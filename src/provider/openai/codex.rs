//! OpenAI Codex (ChatGPT subscription) provider.
//!
//! Talks the Responses API to `chatgpt.com/backend-api/codex/responses`, authenticated by the
//! bearer token + `ChatGPT-Account-ID` header issued by the Codex OAuth flow. Mirrors how OpenAI's
//! own first-party Codex CLI authenticates so the wire shape matches.

mod auth;
mod responses;

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use self::{
    auth::{extract_account_id, extract_expiration_seconds},
    responses::{build_request_body, drive_responses_sse_stream},
};
use crate::{
    error::{MekaError, Result},
    provider::{
        AuthCredential, DEFAULT_OPENAI_CODEX_CLIENT_ID, Message, Provider, StopReason, StreamEvent,
        TokenUsage, ToolDefinition,
    },
    session::TokenStore,
};

/// Default endpoint for OpenAI Codex subscription requests. The path `/backend-api/codex/responses`
/// is appended at request time.
const DEFAULT_BASE_URL: &str = "https://chatgpt.com";

/// Default OAuth token endpoint. Refresh requests POST here as JSON.
const DEFAULT_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";

/// `originator` request header value. Mirrors Codex's `codex_cli_rs` slot, flagged as the calling
/// tool so OpenAI can attribute traffic.
const ORIGINATOR: &str = "meka_cli";

fn now_epoch_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

pub struct OpenAiCodexProvider {
    client: reqwest::Client,
    credential: tokio::sync::RwLock<AuthCredential>,
    base_url: String,
    model: String,
    client_id: String,
    oauth_token_url: String,
    token_store: Option<Arc<TokenStore>>,
    /// Profile name this provider's credential is stored under, so refreshed tokens are written
    /// back to the correct `provider_credentials` row.
    credential_key: String,
    /// `low` / `medium` / `high` for reasoning models (gpt-5, o-series). Forwarded as
    /// `reasoning.effort` in the request body. `None` skips the reasoning block entirely.
    reasoning_effort: Option<String>,
    /// Per-request output token cap from the profile; `None` leaves the Responses API default.
    max_output_tokens: Option<u64>,
    user_agent: String,
}

impl OpenAiCodexProvider {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        credential: AuthCredential,
        model: String,
        base_url: Option<String>,
        client_id: Option<String>,
        oauth_token_url: Option<String>,
        token_store: Option<Arc<TokenStore>>,
        credential_key: String,
        reasoning_effort: Option<String>,
        max_output_tokens: Option<u64>,
    ) -> Result<Self> {
        // chatgpt.com is fronted by Cloudflare; enabling the cookie jar lets bot-clearance cookies
        // (e.g. `__cf_bm`) persist across requests.
        let client = reqwest::Client::builder()
            .cookie_store(true)
            .build()
            .map_err(|error| {
                MekaError::Provider(format!(
                    "failed to build openai-codex HTTP client: {}",
                    error
                ))
            })?;

        Ok(Self {
            client,
            credential: tokio::sync::RwLock::new(credential),
            base_url: base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            model,
            client_id: client_id.unwrap_or_else(|| DEFAULT_OPENAI_CODEX_CLIENT_ID.to_string()),
            oauth_token_url: oauth_token_url.unwrap_or_else(|| DEFAULT_TOKEN_URL.to_string()),
            token_store,
            credential_key,
            reasoning_effort,
            max_output_tokens,
            user_agent: format!(
                "meka/{} ({}; {})",
                env!("CARGO_PKG_VERSION"),
                std::env::consts::OS,
                std::env::consts::ARCH
            ),
        })
    }

    /// Returns the URL the request POSTs to. Codex's own client appends `/backend-api`
    /// automatically when the base URL is one of the chatgpt.com domains, but we keep the path
    /// explicit so users pointing `--base-url` at a custom proxy don't need to know the rewrite
    /// rule.
    fn responses_url(&self) -> String {
        let trimmed = self.base_url.trim_end_matches('/');
        if trimmed.contains("/backend-api") || trimmed.contains("/codex") {
            format!("{}/responses", trimmed)
        } else {
            format!("{}/backend-api/codex/responses", trimmed)
        }
    }

    /// Returns `(bearer_token, account_id)`, refreshing the access token first if it's within 5
    /// minutes of expiry. The account_id is `Option<String>` because free-tier accounts may not
    /// have one (Codex's auth/manager.rs treats the missing claim as non-fatal).
    async fn ensure_valid_credential(&self) -> Result<(String, Option<String>)> {
        {
            let credential = self.credential.read().await;
            let AuthCredential::OAuthToken {
                access_token,
                expires_at,
                account_id,
                ..
            } = &*credential
            else {
                return Err(MekaError::Provider(
                    "openai-codex requires an OAuth token, not an API key".to_string(),
                ));
            };

            let needs_refresh = expires_at.is_some_and(|exp| now_epoch_millis() + 300_000 >= exp);
            if !needs_refresh {
                return Ok((access_token.clone(), account_id.clone()));
            }
        }

        let mut credential = self.credential.write().await;

        // Re-read the latest credential from the DB. Refresh tokens rotate on each successful
        // refresh, and a sibling meka process may have rotated ours since startup. Without this
        // re-read we'd POST a stale refresh_token and the OAuth provider would reject it with
        // `invalid_grant`.
        if let Some(store) = &self.token_store {
            match store.load_provider_credential(&self.credential_key).await {
                Ok(Some(latest)) => *credential = latest,
                Ok(None) => {}
                Err(error) => {
                    tracing::warn!(
                        "failed to re-read Codex OAuth token before refresh: {}",
                        error
                    );
                }
            }
        }

        // Double-check after the DB re-read: another caller (in this process or a sibling meka) may
        // already have rotated to a still-valid access token.
        if let AuthCredential::OAuthToken {
            access_token,
            expires_at,
            account_id,
            ..
        } = &*credential
        {
            let needs_refresh = expires_at.is_some_and(|exp| now_epoch_millis() + 300_000 >= exp);
            if !needs_refresh {
                return Ok((access_token.clone(), account_id.clone()));
            }
        }

        let refresh_token = match &*credential {
            AuthCredential::OAuthToken { refresh_token, .. } => refresh_token.clone(),
            _ => None,
        };
        let Some(refresh_token) = refresh_token else {
            return Err(MekaError::Provider(
                "OAuth access token expired and no refresh token available".to_string(),
            ));
        };

        let new_credential = self.refresh_oauth_token(&refresh_token).await?;
        let (token_value, account_id) = match &new_credential {
            AuthCredential::OAuthToken {
                access_token,
                account_id,
                ..
            } => (access_token.clone(), account_id.clone()),
            _ => unreachable!("refresh always returns OAuthToken"),
        };

        if let Some(store) = &self.token_store
            && let Err(error) = store
                .save_provider_credential(&self.credential_key, &new_credential)
                .await
        {
            tracing::warn!("failed to persist refreshed Codex OAuth token: {}", error);
        }

        *credential = new_credential;
        Ok((token_value, account_id))
    }

    async fn refresh_oauth_token(&self, refresh_token: &str) -> Result<AuthCredential> {
        tracing::info!("refreshing Codex OAuth token");

        let response = self
            .client
            .post(&self.oauth_token_url)
            .json(&serde_json::json!({
                "client_id": self.client_id,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }))
            .send()
            .await
            .map_err(|error| {
                MekaError::Provider(format!(
                    "Codex OAuth token refresh request failed: {}",
                    crate::error::format_reqwest_error(&error)
                ))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_else(|error| {
                tracing::warn!("failed to read Codex OAuth refresh error body: {}", error);
                String::new()
            });
            return Err(MekaError::Provider(format!(
                "Codex OAuth token refresh failed ({}): {}",
                status, body
            )));
        }

        #[derive(Deserialize)]
        struct RefreshResponse {
            id_token: Option<String>,
            access_token: Option<String>,
            refresh_token: Option<String>,
        }

        let data: RefreshResponse = response.json().await.map_err(|error| {
            MekaError::Provider(format!("failed to parse Codex refresh response: {}", error))
        })?;

        let access_token = data.access_token.ok_or_else(|| {
            MekaError::Provider("Codex refresh response missing access_token".to_string())
        })?;

        // Re-extract `chatgpt_account_id` from the new id_token if the server returned one: the
        // workspace association can change.
        let account_id = match data.id_token.as_deref() {
            Some(id_token) => extract_account_id(id_token).ok().flatten(),
            None => None,
        };

        // expires_at comes from the access_token JWT's `exp` claim.
        let expires_at = match extract_expiration_seconds(&access_token) {
            Ok(Some(seconds)) => Some(seconds * 1000),
            _ => None,
        };

        Ok(AuthCredential::OAuthToken {
            access_token,
            refresh_token: data
                .refresh_token
                .or_else(|| Some(refresh_token.to_string())),
            expires_at,
            account_id,
        })
    }

    fn apply_headers(
        &self,
        request: reqwest::RequestBuilder,
        bearer: &str,
        account_id: Option<&str>,
    ) -> reqwest::RequestBuilder {
        let mut request = request
            .header("Authorization", format!("Bearer {}", bearer))
            .header("originator", ORIGINATOR)
            .header("User-Agent", &self.user_agent)
            .header("Accept", "text/event-stream")
            .header("Content-Type", "application/json");
        if let Some(account_id) = account_id {
            request = request.header("ChatGPT-Account-ID", account_id);
        }
        request
    }
}

#[async_trait]
impl Provider for OpenAiCodexProvider {
    async fn complete(
        &self,
        _system_prompt: &str,
        _messages: &[Message],
        _tools: &[ToolDefinition],
    ) -> Result<(
        Message,
        StopReason,
        TokenUsage,
        Vec<crate::provider::Notice>,
    )> {
        // The Responses API on chatgpt.com only ever returns SSE; there is no non-streaming JSON
        // response shape to parse. The agent layer calls `stream` for openai-codex.
        Err(MekaError::Provider(
            "openai-codex does not support non-streaming completion; use streaming mode"
                .to_string(),
        ))
    }

    async fn stream(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        event_sender: mpsc::Sender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        let body = build_request_body(
            &self.model,
            system_prompt,
            messages,
            tools,
            self.reasoning_effort.as_deref(),
            self.max_output_tokens,
            true,
        );

        let (bearer, account_id) = self.ensure_valid_credential().await?;

        let request = self
            .apply_headers(
                self.client.post(self.responses_url()),
                &bearer,
                account_id.as_deref(),
            )
            .json(&body);

        let response = request.send().await.map_err(|error| {
            MekaError::Provider(format!(
                "Codex HTTP request failed: {}",
                crate::error::format_reqwest_error(&error)
            ))
        })?;

        drive_responses_sse_stream(response, event_sender, cancellation).await
    }

    fn name(&self) -> &str {
        "openai-codex"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_credential() -> AuthCredential {
        AuthCredential::OAuthToken {
            access_token: "access-test".to_string(),
            refresh_token: Some("refresh-test".to_string()),
            // 1 day in the future to avoid the refresh path during construction.
            expires_at: Some(now_epoch_millis() + 86_400_000),
            account_id: Some("workspace-test".to_string()),
        }
    }

    fn test_provider() -> OpenAiCodexProvider {
        OpenAiCodexProvider::new(
            test_credential(),
            "gpt-5".to_string(),
            None,
            None,
            None,
            None,
            "test".to_string(),
            Some("high".to_string()),
            None,
        )
        .expect("provider")
    }

    #[test]
    fn test_provider_name() {
        assert_eq!(test_provider().name(), "openai-codex");
    }

    #[test]
    fn test_responses_url_default_appends_backend_api_codex() {
        let provider = test_provider();
        assert_eq!(
            provider.responses_url(),
            "https://chatgpt.com/backend-api/codex/responses"
        );
    }

    #[test]
    fn test_responses_url_user_supplied_backend_api_path_preserved() {
        let provider = OpenAiCodexProvider::new(
            test_credential(),
            "gpt-5".to_string(),
            Some("https://example.com/backend-api/codex".to_string()),
            None,
            None,
            None,
            "test".to_string(),
            None,
            None,
        )
        .expect("provider");
        assert_eq!(
            provider.responses_url(),
            "https://example.com/backend-api/codex/responses"
        );
    }

    #[test]
    fn test_responses_url_strips_trailing_slash() {
        let provider = OpenAiCodexProvider::new(
            test_credential(),
            "gpt-5".to_string(),
            Some("https://chatgpt.com/".to_string()),
            None,
            None,
            None,
            "test".to_string(),
            None,
            None,
        )
        .expect("provider");
        assert_eq!(
            provider.responses_url(),
            "https://chatgpt.com/backend-api/codex/responses"
        );
    }

    #[tokio::test]
    async fn test_ensure_valid_credential_returns_token_and_account_id() {
        let provider = test_provider();
        let (bearer, account_id) = provider
            .ensure_valid_credential()
            .await
            .expect("valid credential");
        assert_eq!(bearer, "access-test");
        assert_eq!(account_id.as_deref(), Some("workspace-test"));
    }

    #[tokio::test]
    async fn test_ensure_valid_credential_rejects_api_key() {
        let provider = OpenAiCodexProvider::new(
            AuthCredential::ApiKey("sk-test".to_string()),
            "gpt-5".to_string(),
            None,
            None,
            None,
            None,
            "test".to_string(),
            None,
            None,
        )
        .expect("provider");
        let result = provider.ensure_valid_credential().await;
        assert!(matches!(result, Err(MekaError::Provider(_))));
    }

    #[tokio::test]
    async fn test_ensure_valid_credential_no_refresh_token_when_expired() {
        // Token already expired, no refresh available → error.
        let provider = OpenAiCodexProvider::new(
            AuthCredential::OAuthToken {
                access_token: "old".to_string(),
                refresh_token: None,
                expires_at: Some(now_epoch_millis() - 1_000),
                account_id: None,
            },
            "gpt-5".to_string(),
            None,
            None,
            None,
            None,
            "test".to_string(),
            None,
            None,
        )
        .expect("provider");
        let result = provider.ensure_valid_credential().await;
        assert!(matches!(result, Err(MekaError::Provider(ref m)) if m.contains("expired")));
    }

    #[tokio::test]
    async fn test_complete_returns_unsupported_error() {
        // openai-codex is streaming-only; complete() must error explicitly rather than silently
        // fall back to a non-streaming code path that doesn't exist for this endpoint.
        let provider = test_provider();
        let result = provider.complete("", &[], &[]).await;
        assert!(matches!(result, Err(MekaError::Provider(_))));
    }
}
