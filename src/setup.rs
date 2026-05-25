//! First-launch interactive configuration wizard. Walks the user through
//! provider selection, runs the Claude OAuth/PKCE flow when applicable, and
//! writes the resulting `~/.config/agsh/config.toml`.

use std::io::{self, Write};

use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use rand::RngExt;
use sha2::{Digest, Sha256};

use crate::{
    config,
    provider::{AuthCredential, DEFAULT_CLAUDE_CLIENT_ID, DEFAULT_OPENAI_CODEX_CLIENT_ID},
    session::TokenStore,
};
const REDIRECT_URI: &str = "https://platform.claude.com/oauth/code/callback";
const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const TOKEN_URL: &str = "https://api.anthropic.com/v1/oauth/token";
const SCOPES: &str =
    "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers";

/// `openai-codex` OAuth flow constants. Mirror Codex's first-party CLI:
/// the authorization server lives at `auth.openai.com`, the redirect
/// listener binds on `localhost:1455` (matching `temp/codex/codex-rs/login/src/server.rs:51`).
const CODEX_AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CODEX_REDIRECT_PORT: u16 = 1455;
const CODEX_SCOPES: &str =
    "openid profile email offline_access api.connectors.read api.connectors.invoke";
/// Wall-clock budget for the user to complete the in-browser authorization.
const CODEX_CALLBACK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(180);

fn client_id() -> String {
    std::env::var("CLAUDE_CLIENT_ID").unwrap_or_else(|_| DEFAULT_CLAUDE_CLIENT_ID.to_string())
}

fn codex_client_id() -> String {
    std::env::var("CODEX_CLIENT_ID").unwrap_or_else(|_| DEFAULT_OPENAI_CODEX_CLIENT_ID.to_string())
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
        account_id: None,
    })
}

fn prompt_line(prompt: &str) -> io::Result<String> {
    eprint!("{}", prompt);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

fn prompt_choice(prompt: &str, options: &[&str]) -> io::Result<usize> {
    eprintln!("{}", prompt);
    for (index, option) in options.iter().enumerate() {
        eprintln!("  {}. {}", index + 1, option);
    }

    loop {
        let input = prompt_line("> ")?;
        if let Ok(choice) = input.parse::<usize>()
            && choice >= 1
            && choice <= options.len()
        {
            return Ok(choice - 1);
        }
        eprintln!("Please enter a number between 1 and {}.", options.len());
    }
}

pub async fn run_setup(token_store: &TokenStore) -> anyhow::Result<()> {
    eprintln!("Welcome to agsh! Let's set up your configuration.\n");

    let provider_index = prompt_choice("Select a provider:", &[
        "claude-oauth (Claude Code OAuth login)",
        "claude-api (Claude API key)",
        "openai-codex (ChatGPT subscription login)",
        "openai-api (OpenAI API key)",
    ])?;
    let provider_name = match provider_index {
        0 => "claude-oauth",
        1 => "claude-api",
        2 => "openai-codex",
        _ => "openai-api",
    };

    eprintln!();

    let mut api_key: Option<String> = None;

    match provider_name {
        "claude-oauth" => {
            run_oauth_login(token_store).await?;
        }
        "claude-api" => {
            let key = prompt_line("Enter your Claude API key: ")?;
            if key.is_empty() {
                anyhow::bail!("API key cannot be empty");
            }
            api_key = Some(key);
        }
        "openai-codex" => {
            run_codex_oauth_login(token_store).await?;
        }
        _ => {
            let key = prompt_line("Enter your OpenAI API key: ")?;
            if key.is_empty() {
                anyhow::bail!("API key cannot be empty");
            }
            api_key = Some(key);
        }
    }

    eprintln!();
    let model = prompt_line("Model name: ")?;
    if model.is_empty() {
        anyhow::bail!("model name cannot be empty");
    }

    eprintln!();
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
    eprintln!();
    eprintln!("To authorize, open this URL in your browser:");
    eprintln!("    {}", url);
    eprintln!();

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

/// Driver for the `openai-codex` PKCE + authorization-code flow against
/// `auth.openai.com`. Differs from `run_oauth_login` (Claude) in that:
/// 1. The redirect URI is `http://localhost:1455/auth/callback` — a real localhost listener, not a
///    paste-the-code-back-in flow. Matches the first-party Codex CLI.
/// 2. The token exchange is form-urlencoded (Claude uses JSON).
/// 3. The id_token JWT is decoded post-exchange to pull `chatgpt_account_id`, which is sent on
///    every Codex request as the `ChatGPT-Account-ID` header.
async fn run_codex_oauth_login(token_store: &TokenStore) -> anyhow::Result<()> {
    let client_id = codex_client_id();
    let (code_verifier, code_challenge) = generate_pkce_pair();
    let state = generate_state();

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", CODEX_REDIRECT_PORT))
        .await
        .map_err(|error| {
            anyhow::anyhow!(
                "failed to bind callback listener on 127.0.0.1:{}: {}. \
                 Is another agsh / codex login already running?",
                CODEX_REDIRECT_PORT,
                error
            )
        })?;
    let redirect_uri = format!("http://localhost:{}/auth/callback", CODEX_REDIRECT_PORT);

    let url = build_codex_authorize_url(&client_id, &code_challenge, &state, &redirect_uri)?;

    if let Err(error) = open::that(&url) {
        tracing::debug!("failed to open browser for Codex login: {}", error);
    }
    eprintln!();
    eprintln!("To authorize, open this URL in your browser:");
    eprintln!("    {}", url);
    eprintln!();
    eprintln!(
        "Waiting up to {}s for the callback on 127.0.0.1:{}...",
        CODEX_CALLBACK_TIMEOUT.as_secs(),
        CODEX_REDIRECT_PORT
    );

    let (received_code, received_state) =
        accept_codex_callback(listener, CODEX_CALLBACK_TIMEOUT).await?;

    if received_state != state {
        anyhow::bail!("OAuth state mismatch — possible CSRF; aborting");
    }

    let credential =
        exchange_codex_code(&received_code, &code_verifier, &client_id, &redirect_uri).await?;
    token_store
        .save_oauth_token(crate::provider::openai::codex::STORAGE_KEY, &credential)
        .await?;

    tracing::info!("Codex login successful; OAuth tokens saved");

    Ok(())
}

fn build_codex_authorize_url(
    client_id: &str,
    code_challenge: &str,
    state: &str,
    redirect_uri: &str,
) -> anyhow::Result<String> {
    let mut url = reqwest::Url::parse(CODEX_AUTHORIZE_URL)?;
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", client_id)
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("scope", CODEX_SCOPES)
        .append_pair("code_challenge", code_challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("id_token_add_organizations", "true")
        .append_pair("codex_cli_simplified_flow", "true")
        .append_pair("state", state)
        .append_pair("originator", "agsh_cli");
    Ok(url.to_string())
}

/// Accept one HTTP request on the bound listener, parse the OAuth callback
/// path, and return `(code, state)`. Loops past unrelated requests
/// (favicons, browser preflights) until the deadline elapses.
async fn accept_codex_callback(
    listener: tokio::net::TcpListener,
    timeout: std::time::Duration,
) -> anyhow::Result<(String, String)> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            anyhow::bail!("authorization timed out after {}s", timeout.as_secs());
        }
        let accept = match tokio::time::timeout(remaining, listener.accept()).await {
            Ok(Ok(pair)) => pair,
            Ok(Err(error)) => anyhow::bail!("accept failed: {}", error),
            Err(_) => anyhow::bail!("authorization timed out after {}s", timeout.as_secs()),
        };
        let (mut stream, _) = accept;

        // Read until end-of-headers or 64 KiB cap; same approach as MCP's
        // OAuth callback handler.
        const MAX_BYTES: usize = 64 * 1024;
        let mut buffer = Vec::with_capacity(4096);
        let mut temp = [0u8; 4096];
        let headers_complete = loop {
            if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
                break true;
            }
            if buffer.len() >= MAX_BYTES {
                break false;
            }
            let read_remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if read_remaining.is_zero() {
                anyhow::bail!("authorization timed out after {}s", timeout.as_secs());
            }
            match tokio::time::timeout(read_remaining, stream.read(&mut temp)).await {
                Ok(Ok(0)) => break buffer.windows(4).any(|window| window == b"\r\n\r\n"),
                Ok(Ok(n)) => buffer.extend_from_slice(&temp[..n]),
                Ok(Err(error)) => anyhow::bail!("read failed: {}", error),
                Err(_) => anyhow::bail!("authorization timed out after {}s", timeout.as_secs()),
            }
        };

        if !headers_complete {
            let _ = stream
                .write_all(b"HTTP/1.1 431 Request Header Fields Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .await;
            continue;
        }

        let request = String::from_utf8_lossy(&buffer);
        match parse_codex_callback_query(&request) {
            CodexCallback::Match { code, state } => {
                let body = b"<!DOCTYPE html><html><body>\
                    <h1>Codex authorization successful</h1>\
                    <p>You can close this tab and return to agsh.</p>\
                    </body></html>";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.write_all(body).await;
                return Ok((code, state));
            }
            CodexCallback::NotCallback => {
                let _ = stream
                    .write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .await;
                continue;
            }
            CodexCallback::Malformed(message) => {
                let _ = stream
                    .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    .await;
                anyhow::bail!(message);
            }
            CodexCallback::AuthError(message) => {
                let _ = stream
                    .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    .await;
                anyhow::bail!("authorization server returned error: {}", message);
            }
        }
    }
}

enum CodexCallback {
    Match { code: String, state: String },
    NotCallback,
    Malformed(String),
    AuthError(String),
}

fn parse_codex_callback_query(request: &str) -> CodexCallback {
    let Some(first_line) = request.lines().next() else {
        return CodexCallback::Malformed("empty HTTP request".to_string());
    };
    let Some(path) = first_line.split_whitespace().nth(1) else {
        return CodexCallback::Malformed("malformed HTTP request line".to_string());
    };
    let (path_component, query_string) = match path.split_once('?') {
        Some((p, q)) => (p, q),
        None => (path, ""),
    };
    if !path_component.eq_ignore_ascii_case("/auth/callback") {
        return CodexCallback::NotCallback;
    }
    if query_string.is_empty() {
        return CodexCallback::Malformed("no query parameters in callback URL".to_string());
    }

    let mut code = None;
    let mut state = None;
    let mut error_param: Option<String> = None;
    for pair in query_string.split('&') {
        let Some((key, value)) = pair.split_once('=') else {
            continue;
        };
        let decoded = percent_encoding::percent_decode_str(value)
            .decode_utf8_lossy()
            .into_owned();
        match key {
            "code" => code = Some(decoded),
            "state" => state = Some(decoded),
            "error" => error_param = Some(decoded),
            _ => {}
        }
    }

    if let Some(message) = error_param {
        return CodexCallback::AuthError(message);
    }
    match (code, state) {
        (Some(code), Some(state)) => CodexCallback::Match { code, state },
        _ => CodexCallback::Malformed("callback missing 'code' or 'state' parameter".to_string()),
    }
}

async fn exchange_codex_code(
    code: &str,
    code_verifier: &str,
    client_id: &str,
    redirect_uri: &str,
) -> anyhow::Result<AuthCredential> {
    use percent_encoding::{NON_ALPHANUMERIC, utf8_percent_encode};
    let encode = |value: &str| utf8_percent_encode(value, NON_ALPHANUMERIC).to_string();
    let body = format!(
        "grant_type=authorization_code&code={}&redirect_uri={}&client_id={}&code_verifier={}",
        encode(code),
        encode(redirect_uri),
        encode(client_id),
        encode(code_verifier),
    );

    let client = reqwest::Client::new();
    let response = client
        .post(CODEX_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("Codex token exchange failed ({}): {}", status, body);
    }

    #[derive(serde::Deserialize)]
    struct CodexTokenResponse {
        id_token: Option<String>,
        access_token: String,
        refresh_token: Option<String>,
    }

    let token: CodexTokenResponse = response.json().await?;

    let account_id = token.id_token.as_deref().and_then(extract_codex_account_id);

    let expires_at = extract_jwt_expiration_millis(&token.access_token);

    Ok(AuthCredential::OAuthToken {
        access_token: token.access_token,
        refresh_token: token.refresh_token,
        expires_at,
        account_id,
    })
}

/// Decode an OpenAI id_token JWT and extract `chatgpt_account_id` from the
/// nested `https://api.openai.com/auth` claim. Returns `None` on any
/// failure — the absence of an account_id isn't fatal at login time.
fn extract_codex_account_id(jwt: &str) -> Option<String> {
    let payload = jwt.split('.').nth(1)?;
    let bytes = URL_SAFE_NO_PAD.decode(payload).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    value
        .get("https://api.openai.com/auth")
        .and_then(|auth| auth.get("chatgpt_account_id"))
        .and_then(|id| id.as_str())
        .map(|id| id.to_string())
}

/// Decode the `exp` claim of a JWT (in seconds) and return millis. Returns
/// `None` if the claim is missing or the JWT is malformed.
fn extract_jwt_expiration_millis(jwt: &str) -> Option<i64> {
    let payload = jwt.split('.').nth(1)?;
    let bytes = URL_SAFE_NO_PAD.decode(payload).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let exp = value.get("exp")?.as_i64()?;
    Some(exp * 1000)
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

    #[test]
    fn test_build_codex_authorize_url_contains_required_params() {
        let url = build_codex_authorize_url(
            "app_test",
            "challenge_x",
            "state_y",
            "http://localhost:1455/auth/callback",
        )
        .unwrap();
        assert!(url.starts_with(CODEX_AUTHORIZE_URL));
        assert!(url.contains("response_type=code"));
        assert!(url.contains("client_id=app_test"));
        assert!(url.contains("code_challenge=challenge_x"));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("state=state_y"));
        // Codex-specific knobs that distinguish this from Claude's flow.
        assert!(url.contains("id_token_add_organizations=true"));
        assert!(url.contains("codex_cli_simplified_flow=true"));
        assert!(url.contains("originator=agsh_cli"));
        // Scope is space-separated and percent-encoded with `+` for spaces.
        assert!(url.contains("openid"));
        assert!(url.contains("offline_access"));
    }

    #[test]
    fn test_parse_codex_callback_query_match() {
        let request =
            "GET /auth/callback?code=abc123&state=xyz HTTP/1.1\r\nHost: localhost:1455\r\n\r\n";
        match parse_codex_callback_query(request) {
            CodexCallback::Match { code, state } => {
                assert_eq!(code, "abc123");
                assert_eq!(state, "xyz");
            }
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn test_parse_codex_callback_query_decodes_percent_encoding() {
        let request = "GET /auth/callback?code=hello%20world&state=s%23t HTTP/1.1\r\n\r\n";
        match parse_codex_callback_query(request) {
            CodexCallback::Match { code, state } => {
                assert_eq!(code, "hello world");
                assert_eq!(state, "s#t");
            }
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn test_parse_codex_callback_query_non_callback_path() {
        let request = "GET /favicon.ico HTTP/1.1\r\n\r\n";
        assert!(matches!(
            parse_codex_callback_query(request),
            CodexCallback::NotCallback
        ));
    }

    #[test]
    fn test_parse_codex_callback_query_missing_params() {
        let request = "GET /auth/callback HTTP/1.1\r\n\r\n";
        assert!(matches!(
            parse_codex_callback_query(request),
            CodexCallback::Malformed(_)
        ));
    }

    #[test]
    fn test_parse_codex_callback_query_auth_error() {
        let request = "GET /auth/callback?error=access_denied HTTP/1.1\r\n\r\n";
        match parse_codex_callback_query(request) {
            CodexCallback::AuthError(message) => assert_eq!(message, "access_denied"),
            _ => panic!("expected AuthError"),
        }
    }

    #[test]
    fn test_extract_codex_account_id_from_namespaced_claim() {
        let payload = serde_json::json!({
            "sub": "u",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "ws-1"
            }
        });
        let header = URL_SAFE_NO_PAD.encode(b"{}");
        let body = URL_SAFE_NO_PAD.encode(payload.to_string().as_bytes());
        let signature = URL_SAFE_NO_PAD.encode(b"sig");
        let jwt = format!("{}.{}.{}", header, body, signature);
        assert_eq!(extract_codex_account_id(&jwt).as_deref(), Some("ws-1"));
    }

    #[test]
    fn test_extract_codex_account_id_missing_returns_none() {
        let payload = serde_json::json!({"sub": "u"});
        let body = URL_SAFE_NO_PAD.encode(payload.to_string().as_bytes());
        let jwt = format!("{}.{}.{}", "h", body, "s");
        assert!(extract_codex_account_id(&jwt).is_none());
    }

    #[test]
    fn test_extract_jwt_expiration_millis() {
        let payload = serde_json::json!({"exp": 1_700_000_000});
        let body = URL_SAFE_NO_PAD.encode(payload.to_string().as_bytes());
        let jwt = format!("{}.{}.{}", "h", body, "s");
        assert_eq!(extract_jwt_expiration_millis(&jwt), Some(1_700_000_000_000));
    }
}
