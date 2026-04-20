//! First-launch interactive configuration wizard. Walks the user through
//! provider selection, runs the Claude OAuth/PKCE flow when applicable, and
//! writes the resulting `~/.config/agsh/config.toml`.

use std::io::{self, Write};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use rand::Rng;
use sha2::{Digest, Sha256};

use crate::config;
use crate::provider::{AuthCredential, DEFAULT_CLAUDE_CLIENT_ID};
use crate::session::TokenStore;
const REDIRECT_URI: &str = "https://platform.claude.com/oauth/code/callback";
const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const TOKEN_URL: &str = "https://api.anthropic.com/v1/oauth/token";
const SCOPES: &str =
    "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers";

fn client_id() -> String {
    std::env::var("CLAUDE_CLIENT_ID").unwrap_or_else(|_| DEFAULT_CLAUDE_CLIENT_ID.to_string())
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

fn build_authorize_url(
    client_id: &str,
    code_challenge: &str,
    state: &str,
) -> anyhow::Result<String> {
    let mut url = reqwest::Url::parse(AUTHORIZE_URL)?;
    url.query_pairs_mut()
        .append_pair("code", "true")
        .append_pair("client_id", client_id)
        .append_pair("response_type", "code")
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("scope", SCOPES)
        .append_pair("code_challenge", code_challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", state);
    Ok(url.to_string())
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

    let expires_at = match token_response.expires_in {
        Some(seconds) => {
            let now_millis = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_millis() as i64)
                .unwrap_or(0);
            Some(now_millis + (seconds * 1000))
        }
        None => None,
    };

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
        if let Ok(choice) = input.parse::<usize>()
            && choice >= 1
            && choice <= options.len()
        {
            return Ok(choice - 1);
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

    if let Some(path) = config::config_file_path() {
        tracing::info!("configuration saved to {}", path.display());
    } else {
        tracing::info!("configuration saved");
    }

    Ok(())
}

async fn run_oauth_login(token_store: &TokenStore) -> anyhow::Result<()> {
    let client_id = client_id();
    let (code_verifier, code_challenge) = generate_pkce_pair();
    let state = generate_state();

    let url = build_authorize_url(&client_id, &code_challenge, &state)?;

    // The URL is printed unconditionally below; silently try to open
    // it as a convenience on desktop, and keep the failure at debug
    // since headless hosts will hit this path every time.
    if let Err(error) = open::that(&url) {
        tracing::debug!("failed to open browser for setup: {}", error);
    }
    println!();
    println!("To authorize, open this URL in your browser:");
    println!("    {}", url);
    println!();

    let code_input = prompt_line("After authorizing, paste the authorization code here:\n> ")?;
    if code_input.is_empty() {
        anyhow::bail!("authorization code cannot be empty");
    }

    // The pasted value may include the state after a '#' delimiter (e.g., "code#state")
    let code = code_input.split('#').next().unwrap_or(&code_input);

    let credential = exchange_code(code, &code_verifier, &client_id, &state).await?;
    token_store.save_oauth_token("claude", &credential).await?;

    tracing::info!("login successful; OAuth tokens saved");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_pkce_pair_lengths() {
        let (verifier, challenge) = generate_pkce_pair();
        assert!(!verifier.is_empty());
        assert!(!challenge.is_empty());
        assert_ne!(verifier, challenge);
    }

    #[test]
    fn test_generate_pkce_pair_challenge_is_sha256_of_verifier() {
        let (verifier, challenge) = generate_pkce_pair();
        let expected_digest = Sha256::digest(verifier.as_bytes());
        let expected_challenge = URL_SAFE_NO_PAD.encode(expected_digest);
        assert_eq!(challenge, expected_challenge);
    }

    #[test]
    fn test_generate_state_not_empty() {
        let state = generate_state();
        assert!(!state.is_empty());
    }

    #[test]
    fn test_generate_state_unique() {
        let state1 = generate_state();
        let state2 = generate_state();
        assert_ne!(state1, state2);
    }

    #[test]
    fn test_build_authorize_url_contains_params() {
        let url = build_authorize_url("test-client-id", "test-challenge", "test-state").unwrap();
        assert!(url.contains("client_id=test-client-id"));
        assert!(url.contains("code_challenge=test-challenge"));
        assert!(url.contains("state=test-state"));
        assert!(url.contains("response_type=code"));
        assert!(url.contains("redirect_uri="));
        assert!(url.contains("code_challenge_method=S256"));
    }

    #[test]
    fn test_build_authorize_url_starts_with_authorize_url() {
        let url = build_authorize_url("cid", "ch", "st").unwrap();
        assert!(url.starts_with(AUTHORIZE_URL));
    }
}
