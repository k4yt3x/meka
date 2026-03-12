use std::io::{self, Write};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use rand::Rng;
use sha2::{Digest, Sha256};

use crate::config;
use crate::provider::AuthCredential;
use crate::session::TokenStore;

const DEFAULT_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const REDIRECT_URI: &str = "https://platform.claude.com/oauth/code/callback";
const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const TOKEN_URL: &str = "https://api.anthropic.com/v1/oauth/token";
const SCOPES: &str =
    "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers";

fn client_id() -> String {
    std::env::var("CLAUDE_CLIENT_ID").unwrap_or_else(|_| DEFAULT_CLIENT_ID.to_string())
}

fn generate_pkce_pair() -> (String, String) {
    let mut bytes = [0u8; 32];
    rand::rng().fill(&mut bytes);
    let code_verifier = URL_SAFE_NO_PAD.encode(bytes);

    let digest = Sha256::digest(code_verifier.as_bytes());
    let code_challenge = URL_SAFE_NO_PAD.encode(digest);

    (code_verifier, code_challenge)
}

fn generate_state() -> String {
    let mut bytes = [0u8; 32];
    rand::rng().fill(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

fn build_authorize_url(client_id: &str, code_challenge: &str, state: &str) -> String {
    let mut url = reqwest::Url::parse(AUTHORIZE_URL).expect("static URL is valid");
    url.query_pairs_mut()
        .append_pair("code", "true")
        .append_pair("client_id", client_id)
        .append_pair("response_type", "code")
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("scope", SCOPES)
        .append_pair("code_challenge", code_challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", state);
    url.to_string()
}

#[derive(serde::Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<i64>,
}

async fn exchange_code(
    code: &str,
    code_verifier: &str,
    client_id: &str,
    state: &str,
) -> anyhow::Result<AuthCredential> {
    let client = reqwest::Client::new();

    let response = client
        .post(TOKEN_URL)
        .json(&serde_json::json!({
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": REDIRECT_URI,
            "client_id": client_id,
            "state": state,
        }))
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("token exchange failed ({}): {}", status, body);
    }

    let token_response: TokenResponse = response.json().await?;

    let expires_at = token_response.expires_in.map(|seconds| {
        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is before Unix epoch")
            .as_millis() as i64;
        now_millis + (seconds * 1000)
    });

    Ok(AuthCredential::OAuthToken {
        access_token: token_response.access_token,
        refresh_token: token_response.refresh_token,
        expires_at,
    })
}

fn prompt_line(prompt: &str) -> io::Result<String> {
    print!("{}", prompt);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

fn prompt_choice(prompt: &str, options: &[&str]) -> io::Result<usize> {
    println!("{}", prompt);
    for (index, option) in options.iter().enumerate() {
        println!("  {}. {}", index + 1, option);
    }

    loop {
        let input = prompt_line("> ")?;
        if let Ok(choice) = input.parse::<usize>() {
            if choice >= 1 && choice <= options.len() {
                return Ok(choice - 1);
            }
        }
        println!("Please enter a number between 1 and {}.", options.len());
    }
}

pub async fn run_setup(token_store: &TokenStore) -> anyhow::Result<()> {
    println!("Welcome to agsh! Let's set up your configuration.\n");

    let provider_index = prompt_choice("Select a provider:", &["claude", "openai"])?;
    let provider_name = match provider_index {
        0 => "claude",
        _ => "openai",
    };

    println!();

    let mut api_key: Option<String> = None;

    match provider_name {
        "claude" => {
            let auth_index = prompt_choice("Authentication method:", &["OAuth login", "API key"])?;
            println!();

            match auth_index {
                0 => {
                    run_oauth_login(token_store).await?;
                }
                _ => {
                    let key = prompt_line("Enter your Claude API key: ")?;
                    if key.is_empty() {
                        anyhow::bail!("API key cannot be empty");
                    }
                    api_key = Some(key);
                }
            }
        }
        _ => {
            let key = prompt_line("Enter your OpenAI API key: ")?;
            if key.is_empty() {
                anyhow::bail!("API key cannot be empty");
            }
            api_key = Some(key);
        }
    }

    println!();
    let model = prompt_line("Model name: ")?;
    if model.is_empty() {
        anyhow::bail!("model name cannot be empty");
    }

    println!();
    let base_url_input = prompt_line("API base URL (leave empty for default): ")?;
    let base_url = if base_url_input.is_empty() {
        None
    } else {
        Some(base_url_input)
    };

    config::write_config_file(
        provider_name,
        &model,
        api_key.as_deref(),
        base_url.as_deref(),
    )?;

    println!();
    if let Some(path) = config::config_file_path() {
        println!("Configuration saved to {}", path.display());
    } else {
        println!("Configuration saved.");
    }

    Ok(())
}

async fn run_oauth_login(token_store: &TokenStore) -> anyhow::Result<()> {
    let client_id = client_id();
    let (code_verifier, code_challenge) = generate_pkce_pair();
    let state = generate_state();

    let url = build_authorize_url(&client_id, &code_challenge, &state);

    println!("Opening browser for authorization...");
    if open::that(&url).is_err() {
        tracing::debug!("failed to open browser automatically");
    }
    println!();
    println!("If the browser doesn't open, visit this URL:");
    println!("{}", url);
    println!();

    let code_input = prompt_line("After authorizing, paste the authorization code here:\n> ")?;
    if code_input.is_empty() {
        anyhow::bail!("authorization code cannot be empty");
    }

    // The pasted value may include the state after a '#' delimiter (e.g., "code#state")
    let code = code_input.split('#').next().unwrap_or(&code_input);

    let credential = exchange_code(code, &code_verifier, &client_id, &state).await?;
    token_store.save_oauth_token("claude", &credential).await?;

    println!("Login successful! OAuth tokens saved.");

    Ok(())
}
