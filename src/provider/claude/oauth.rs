//! Claude OAuth provider. Uses Claude Code attestation / billing header machinery to send requests
//! as the official CLI, and manages OAuth token refresh against the Claude token endpoint.

mod attestation;

use std::sync::{
    Arc,
    atomic::{AtomicI8, Ordering},
};

use async_trait::async_trait;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::shared::{
    self, convert_messages_to_claude_content, convert_tools_to_claude_tools,
    drive_claude_sse_stream, model_is_haiku, model_supports_adaptive_thinking,
    model_supports_effort, model_supports_modern_features, parse_non_streaming_response,
};
use crate::{
    error::{AgshError, Result},
    provider::{
        AuthCredential, DEFAULT_CLAUDE_CLIENT_ID, Message, Notice, Provider, StopReason,
        StreamEvent, TokenUsage, ToolDefinition,
    },
    session::TokenStore,
};

/// Claude Code system prompt prefix.
const CC_SYSTEM_PROMPT_PREFIX: &str = "You are Claude Code, Anthropic's official CLI for Claude.";

fn now_epoch_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

pub struct ClaudeOAuthProvider {
    client: reqwest::Client,
    credential: tokio::sync::RwLock<AuthCredential>,
    base_url: String,
    model: String,
    client_id: String,
    oauth_token_url: String,
    token_store: Option<Arc<TokenStore>>,
    session_id: String,
    device_id: String,
    thinking_enabled: bool,
    thinking_budget_tokens: u64,
    thinking_override: AtomicI8,
    /// Value emitted as `output_config.effort` for effort-capable models. Always one of `"low" |
    /// "medium" | "high"` (validated by config layer).
    effort: String,
    /// When true, request `redacted_thinking` blocks via the `redact-thinking-2026-02-12` beta
    /// header.
    redact_thinking: bool,
    /// Per-session counters incremented when image-redaction events fire.
    session_stats: Option<Arc<crate::stats::SessionStats>>,
}

impl ClaudeOAuthProvider {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        credential: AuthCredential,
        model: String,
        base_url: Option<String>,
        client_id: Option<String>,
        oauth_token_url: Option<String>,
        token_store: Option<Arc<TokenStore>>,
        thinking_enabled: bool,
        thinking_budget_tokens: u64,
        device_id: String,
        effort: String,
        redact_thinking: bool,
        session_stats: Option<Arc<crate::stats::SessionStats>>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            credential: tokio::sync::RwLock::new(credential),
            base_url: base_url.unwrap_or_else(|| "https://api.anthropic.com".to_string()),
            model,
            client_id: client_id.unwrap_or_else(|| DEFAULT_CLAUDE_CLIENT_ID.to_string()),
            oauth_token_url: oauth_token_url
                .unwrap_or_else(|| "https://api.anthropic.com/v1/oauth/token".to_string()),
            token_store,
            session_id: Uuid::new_v4().to_string(),
            device_id,
            thinking_enabled,
            thinking_budget_tokens,
            thinking_override: AtomicI8::new(-1),
            effort,
            redact_thinking,
            session_stats,
        }
    }

    fn is_thinking_enabled(&self) -> bool {
        shared::resolve_thinking_enabled(&self.thinking_override, self.thinking_enabled)
    }

    /// Mirrors Claude Code's `getAllModelBetas` (`utils/betas.ts:234-369`)
    /// for the OAuth-on-1P case. Order matches a recent wire dump
    /// (`claude-cli/2.1.41`) with `claude-opus-4-6` + thinking:
    /// `claude-code-20250219, oauth-2025-04-20, adaptive-thinking-2026-01-28,
    ///  context-management-2025-06-27, prompt-caching-scope-2026-01-05,
    ///  effort-2025-11-24`. `redact-thinking-2026-02-12` is appended
    ///  immediately after the thinking beta when `redact_thinking` is set,
    ///  matching `betas.ts:270-277`.
    fn compute_betas(&self) -> Option<String> {
        let model = self.model.as_str();
        let mut parts: Vec<&'static str> = Vec::with_capacity(7);

        if !model_is_haiku(model) {
            parts.push("claude-code-20250219");
        }
        parts.push("oauth-2025-04-20");

        if self.is_thinking_enabled() && model_supports_modern_features(model) {
            if model_supports_adaptive_thinking(model) {
                parts.push("adaptive-thinking-2026-01-28");
            } else {
                parts.push("interleaved-thinking-2025-05-14");
            }

            if self.redact_thinking {
                parts.push("redact-thinking-2026-02-12");
            }
        }

        if model_supports_modern_features(model) {
            parts.push("context-management-2025-06-27");
        }

        parts.push("prompt-caching-scope-2026-01-05");

        if model_supports_effort(model) {
            parts.push("effort-2025-11-24");
        }

        Some(parts.join(","))
    }

    /// Resolve a valid Authorization header, refreshing the OAuth token if it's within 5 minutes of
    /// expiry.
    ///
    /// Concurrency contract (relevant under multi-session ACP where two sessions may call this in
    /// parallel): the `RwLock` on `credential` doubles as the refresh gate. Two tasks that both
    /// observe an expiring token race for the write lock; the loser re-reads after acquiring it and
    /// finds the winner's fresh token via the double-check at the top of the slow path
    /// (`needs_refresh` block below). Exactly one refresh API call fires under contention; both
    /// callers return a valid token. No separate `Mutex<()>` refresh gate is needed.
    async fn ensure_valid_credential(&self) -> Result<(&'static str, String)> {
        {
            let credential = self.credential.read().await;
            match &*credential {
                AuthCredential::ApiKey(_) => {
                    return Err(AgshError::Provider(
                        "claude-oauth requires an OAuth token, not an API key".to_string(),
                    ));
                }
                AuthCredential::OAuthToken {
                    access_token,
                    expires_at,
                    ..
                } => {
                    let needs_refresh = if let Some(exp) = expires_at {
                        now_epoch_millis() + 300_000 >= *exp
                    } else {
                        false
                    };

                    if !needs_refresh {
                        return Ok(("Authorization", format!("Bearer {}", access_token)));
                    }
                }
            }
        }

        // Token expired — attempt refresh
        let mut credential = self.credential.write().await;

        // Re-read the latest credential from the DB. Refresh tokens rotate on each successful
        // refresh, and a sibling agsh process (or `agsh mcp login` flow) may have rotated ours
        // since startup. Without this re-read we'd POST a stale refresh_token and the OAuth
        // provider would reject it with `invalid_grant`.
        if let Some(store) = &self.token_store {
            match store.load_oauth_token("claude").await {
                Ok(Some(latest)) => *credential = latest,
                Ok(None) => {}
                Err(error) => {
                    tracing::warn!("failed to re-read OAuth token before refresh: {}", error);
                }
            }
        }

        // Double-check after the DB re-read: another process may have already rotated and persisted
        // a new access token that's still valid.
        if let AuthCredential::OAuthToken {
            access_token,
            expires_at,
            ..
        } = &*credential
        {
            let needs_refresh = if let Some(exp) = expires_at {
                now_epoch_millis() + 300_000 >= *exp
            } else {
                false
            };

            if !needs_refresh {
                return Ok(("Authorization", format!("Bearer {}", access_token)));
            }
        }

        let refresh_token = match &*credential {
            AuthCredential::OAuthToken { refresh_token, .. } => refresh_token.clone(),
            _ => None,
        };

        let Some(refresh_token) = refresh_token else {
            return Err(AgshError::Provider(
                "OAuth access token expired and no refresh token available".to_string(),
            ));
        };

        let new_credential = self.refresh_oauth_token(&refresh_token).await?;
        let (header_name, header_value) = new_credential.auth_header();

        // Storage key kept as "claude" to preserve existing users' refresh tokens across the
        // provider rename to "claude-oauth".
        if let Some(store) = &self.token_store
            && let Err(error) = store.save_oauth_token("claude", &new_credential).await
        {
            tracing::warn!("failed to persist refreshed OAuth token: {}", error);
        }

        *credential = new_credential;
        Ok((header_name, header_value))
    }

    async fn refresh_oauth_token(&self, refresh_token: &str) -> Result<AuthCredential> {
        tracing::info!("refreshing OAuth token");

        let response = self
            .client
            .post(&self.oauth_token_url)
            .json(&serde_json::json!({
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
            }))
            .send()
            .await
            .map_err(|error| {
                AgshError::Provider(format!(
                    "OAuth token refresh request failed: {}",
                    crate::error::format_reqwest_error(&error)
                ))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_else(|error| {
                tracing::warn!("failed to read OAuth refresh error body: {}", error);
                String::new()
            });
            return Err(AgshError::Provider(format!(
                "OAuth token refresh failed ({}): {}",
                status, body
            )));
        }

        #[derive(Deserialize)]
        struct RefreshResponse {
            access_token: String,
            refresh_token: Option<String>,
            expires_in: Option<u64>,
        }

        let data: RefreshResponse = response.json().await.map_err(|error| {
            AgshError::Provider(format!("failed to parse refresh response: {}", error))
        })?;

        let expires_at = data
            .expires_in
            .map(|seconds| now_epoch_millis() + (seconds as i64) * 1000);

        Ok(AuthCredential::OAuthToken {
            access_token: data.access_token,
            refresh_token: data
                .refresh_token
                .or_else(|| Some(refresh_token.to_string())),
            expires_at,
            account_id: None,
        })
    }

    pub(super) fn build_request_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> serde_json::Value {
        let claude_messages = convert_messages_to_claude_content(messages);

        let metadata_user_id = serde_json::json!({
            "device_id": self.device_id,
            "account_uuid": "",
            "session_id": self.session_id,
        })
        .to_string();

        // `system` must precede `messages` so the billing header's `cch=00000` is always the first
        // occurrence in the serialized JSON.
        let mut body = serde_json::Map::new();

        if !system_prompt.is_empty() {
            let billing_header = attestation::generate_billing_header(messages);
            // Matches recent Claude Code wire shape: only the user system prompt carries
            // `cache_control`. Billing header and identity prefix are unmarked — the source's
            // "boundary mode" (`utils/api.ts:362-409`) assigns `cacheScope: null` to both. ttl `1h`
            // matches Claude Code's `getCacheControl` for OAuth subscribers (`claude.ts:358-374`).
            body.insert(
                "system".to_string(),
                serde_json::json!([
                    {
                        "type": "text",
                        "text": billing_header
                    },
                    {
                        "type": "text",
                        "text": CC_SYSTEM_PROMPT_PREFIX
                    },
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": { "type": "ephemeral", "ttl": "1h" }
                    }
                ]),
            );
        }

        body.insert("model".to_string(), serde_json::json!(self.model));
        body.insert("messages".to_string(), serde_json::json!(claude_messages));

        shared::insert_thinking_fields(
            &mut body,
            self.is_thinking_enabled(),
            &self.model,
            self.thinking_budget_tokens,
        );

        // Mirrors `getAPIContextManagement` (`compact/apiMicrocompact.ts:64-92`) for the
        // OAuth-without-ant-tool-clearing case: when thinking is on and the model supports context
        // management, preserve thinking blocks across previous assistant turns via
        // `clear_thinking_20251015`.
        if self.is_thinking_enabled() && model_supports_modern_features(&self.model) {
            body.insert(
                "context_management".to_string(),
                serde_json::json!({
                    "edits": [{ "type": "clear_thinking_20251015", "keep": "all" }]
                }),
            );
        }

        if !self.is_thinking_enabled() {
            body.insert("temperature".to_string(), serde_json::json!(1));
        }

        if model_supports_effort(&self.model) {
            body.insert(
                "output_config".to_string(),
                serde_json::json!({ "effort": self.effort }),
            );
        }

        body.insert("stream".to_string(), serde_json::json!(stream));
        body.insert(
            "metadata".to_string(),
            serde_json::json!({ "user_id": metadata_user_id }),
        );

        if !tools.is_empty() {
            body.insert(
                "tools".to_string(),
                serde_json::json!(convert_tools_to_claude_tools(tools)),
            );
        }

        serde_json::Value::Object(body)
    }
}

#[async_trait]
impl Provider for ClaudeOAuthProvider {
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<(Message, StopReason, TokenUsage, Vec<Notice>)> {
        let (body_json, redaction_notice) =
            shared::build_body_within_budget(messages, self.session_stats.as_ref(), |msgs| {
                serde_json::to_string(&self.build_request_body(system_prompt, msgs, tools, false))
                    .map_err(|error| {
                        AgshError::Provider(format!("failed to serialize body: {}", error))
                    })
            })?;
        let body_json = if !system_prompt.is_empty() {
            attestation::patch_request_body(&body_json)?
        } else {
            body_json
        };
        let body_size_mib = body_json.len() / 1_048_576;
        let (auth_header_name, auth_header_value) = self.ensure_valid_credential().await?;

        let betas = self.compute_betas();

        let request = attestation::apply_headers(
            self.client
                .post(format!("{}/v1/messages?beta=true", self.base_url)),
            auth_header_name,
            &auth_header_value,
            &self.session_id,
            betas.as_deref(),
        );

        let response = request.body(body_json).send().await.map_err(|error| {
            AgshError::Provider(format!(
                "HTTP request failed (body {} MiB): {}",
                body_size_mib,
                crate::error::format_reqwest_error(&error),
            ))
        })?;

        let status = response.status();
        let response_text = response
            .text()
            .await
            .map_err(|error| AgshError::Provider(format!("failed to read response: {}", error)))?;

        if !status.is_success() {
            return Err(AgshError::Provider(format!(
                "API returned status {}: {}",
                status, response_text
            )));
        }

        let response_json: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|error| AgshError::Provider(format!("invalid JSON response: {}", error)))?;

        let (message, stop_reason, usage) = parse_non_streaming_response(&response_json)?;
        let notices = redaction_notice.into_iter().collect();
        Ok((message, stop_reason, usage, notices))
    }

    async fn stream(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        event_sender: mpsc::Sender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        let (body_json, redaction_notice) =
            shared::build_body_within_budget(messages, self.session_stats.as_ref(), |msgs| {
                serde_json::to_string(&self.build_request_body(system_prompt, msgs, tools, true))
                    .map_err(|error| {
                        AgshError::Provider(format!("failed to serialize body: {}", error))
                    })
            })?;
        // Surface the redaction notice ahead of any provider text. See the mirror in
        // `provider/claude/api.rs::stream` for the rationale.
        if let Some(notice) = redaction_notice
            && let Err(error) = event_sender.send(StreamEvent::Notice(notice)).await
        {
            tracing::debug!("failed to forward redaction notice into stream: {}", error);
        }
        let body_json = if !system_prompt.is_empty() {
            attestation::patch_request_body(&body_json)?
        } else {
            body_json
        };
        let body_size_mib = body_json.len() / 1_048_576;
        let (auth_header_name, auth_header_value) = self.ensure_valid_credential().await?;

        let betas = self.compute_betas();

        let request = attestation::apply_headers(
            self.client
                .post(format!("{}/v1/messages?beta=true", self.base_url))
                .header("accept-encoding", "identity"),
            auth_header_name,
            &auth_header_value,
            &self.session_id,
            betas.as_deref(),
        );

        let response = request.body(body_json).send().await.map_err(|error| {
            AgshError::Provider(format!(
                "HTTP request failed (body {} MiB): {}",
                body_size_mib,
                crate::error::format_reqwest_error(&error),
            ))
        })?;

        drive_claude_sse_stream(response, event_sender, cancellation).await
    }

    fn name(&self) -> &str {
        "claude-oauth"
    }

    fn set_thinking_override(&self, enabled: Option<bool>) {
        let value = match enabled {
            None => -1,
            Some(false) => 0,
            Some(true) => 1,
        };
        self.thinking_override.store(value, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::{attestation::CC_VERSION, *};
    use crate::provider::{ContentBlock, Role, ToolResultContent};

    fn test_provider() -> ClaudeOAuthProvider {
        ClaudeOAuthProvider::new(
            AuthCredential::ApiKey("test-key".to_string()),
            "claude-sonnet-4-20250514".to_string(),
            None,
            None,
            None,
            None,
            false,
            10000,
            "a".repeat(64),
            "high".to_string(),
            false,
            None,
        )
    }

    #[test]
    fn test_claude_request_body_simple() {
        let provider = test_provider();

        let messages = vec![Message::user("hello")];
        let body = provider.build_request_body("system prompt", &messages, &[], false);

        assert_eq!(body["model"], "claude-sonnet-4-20250514");
        assert_eq!(body["stream"], false);

        let system = body["system"].as_array().expect("system should be array");
        assert_eq!(system.len(), 3);

        assert_eq!(system[0]["type"], "text");
        let billing = system[0]["text"].as_str().unwrap();
        let expected_prefix = format!("x-anthropic-billing-header: cc_version={}.", CC_VERSION);
        assert!(billing.starts_with(&expected_prefix), "{}", billing);
        assert!(billing.contains("cc_entrypoint=cli"));
        assert!(billing.contains("cch=00000"));
        assert!(system[0].get("cache_control").is_none());

        assert_eq!(system[1]["type"], "text");
        assert_eq!(system[1]["text"], CC_SYSTEM_PROMPT_PREFIX);
        // Identity prefix carries no cache_control — matches recent Claude Code wire shape
        // (boundary mode in `utils/api.ts:362-409`).
        assert!(system[1].get("cache_control").is_none());

        assert_eq!(system[2]["type"], "text");
        assert_eq!(system[2]["text"], "system prompt");
        // User system prompt carries cache_control with ttl=1h (matches `getCacheControl` for OAuth
        // subscribers).
        assert_eq!(
            system[2]["cache_control"],
            serde_json::json!({"type": "ephemeral", "ttl": "1h"})
        );

        let body_json = serde_json::to_string(&body).unwrap();
        let system_pos = body_json.find("\"system\"").unwrap();
        let messages_pos = body_json.find("\"messages\"").unwrap();
        assert!(system_pos < messages_pos);

        let user_id_str = body["metadata"]["user_id"].as_str().unwrap();
        let user_id_parsed: serde_json::Value = serde_json::from_str(user_id_str).unwrap();
        assert!(user_id_parsed.get("device_id").is_some());
        assert!(user_id_parsed.get("session_id").is_some());

        let claude_messages = body["messages"]
            .as_array()
            .expect("messages should be array");
        assert_eq!(claude_messages.len(), 1);
        assert_eq!(claude_messages[0]["role"], "user");

        let content = claude_messages[0]["content"]
            .as_array()
            .expect("content should be array");
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "hello");
        assert!(content[0].get("cache_control").is_some());
    }

    #[test]
    fn test_claude_request_body_with_tools() {
        let provider = test_provider();

        let tools = vec![ToolDefinition::new(
            "read_file".to_string(),
            "Read a file".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        )];

        let body = provider.build_request_body("", &[], &tools, false);
        let claude_tools = body["tools"].as_array().expect("tools should be array");
        assert_eq!(claude_tools.len(), 1);
        assert_eq!(claude_tools[0]["name"], "read_file");
        assert_eq!(claude_tools[0]["description"], "Read a file");
        assert!(claude_tools[0].get("input_schema").is_some());
        assert!(claude_tools[0].get("cache_control").is_some());
    }

    #[test]
    fn test_claude_request_body_with_tool_calls() {
        let provider = test_provider();

        let messages = vec![
            Message::user("read /tmp/test.txt"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test.txt"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "file contents here".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];

        let body = provider.build_request_body("", &messages, &[], false);
        let claude_messages = body["messages"]
            .as_array()
            .expect("messages should be array");

        assert_eq!(claude_messages.len(), 3);
        assert_eq!(claude_messages[0]["role"], "user");

        assert_eq!(claude_messages[1]["role"], "assistant");
        let assistant_content = claude_messages[1]["content"]
            .as_array()
            .expect("content should be array");
        assert_eq!(assistant_content[0]["type"], "tool_use");
        assert_eq!(assistant_content[0]["id"], "toolu_1");
        assert_eq!(assistant_content[0]["name"], "read_file");

        assert_eq!(claude_messages[2]["role"], "user");
        let result_content = claude_messages[2]["content"]
            .as_array()
            .expect("content should be array");
        assert_eq!(result_content[0]["type"], "tool_result");
        assert_eq!(result_content[0]["tool_use_id"], "toolu_1");

        let first_content = claude_messages[0]["content"]
            .as_array()
            .expect("content should be array");
        assert!(first_content[0].get("cache_control").is_none());
        assert!(assistant_content[0].get("cache_control").is_none());
        assert!(result_content[0].get("cache_control").is_some());
    }

    #[test]
    fn test_claude_parse_non_streaming_text() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": "Hello there!"
            }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 10, "output_tokens": 5 }
        });

        let (message, stop_reason, _) =
            parse_non_streaming_response(&response).expect("should parse");

        assert_eq!(message.text_content(), "Hello there!");
        assert_eq!(stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn test_claude_parse_non_streaming_tool_use() {
        let response = serde_json::json!({
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I'll read that file for you."
                },
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "read_file",
                    "input": {"path": "/tmp/test.txt"}
                }
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 20, "output_tokens": 15 }
        });

        let (message, stop_reason, _) =
            parse_non_streaming_response(&response).expect("should parse");

        assert_eq!(stop_reason, StopReason::ToolUse);
        assert_eq!(message.text_content(), "I'll read that file for you.");

        let tool_uses = message.tool_uses();
        assert_eq!(tool_uses.len(), 1);

        if let ContentBlock::ToolUse { id, name, input } = &tool_uses[0] {
            assert_eq!(id, "toolu_abc");
            assert_eq!(name, "read_file");
            assert_eq!(input["path"], "/tmp/test.txt");
        } else {
            panic!("expected ToolUse block");
        }
    }

    #[test]
    fn test_patch_request_body_replaces_placeholder() {
        let messages = vec![Message::user("hello")];
        let provider = test_provider();
        let body = provider.build_request_body("system prompt", &messages, &[], false);
        let body_json = serde_json::to_string(&body).unwrap();

        assert!(body_json.contains("cch=00000"));

        let patched = attestation::patch_request_body(&body_json).unwrap();
        assert!(!patched.contains("cch=00000"));
        let cch_idx = patched.find("cch=").expect("cch= must be present");
        let token = &patched[cch_idx + 4..cch_idx + 9];
        assert_eq!(token.len(), 5);
        assert!(token.chars().all(|c| c.is_ascii_hexdigit()), "{}", token);

        let patched2 = attestation::patch_request_body(&body_json).unwrap();
        assert_eq!(patched, patched2);
    }

    #[test]
    fn test_patch_request_body_ignores_cch_in_messages() {
        let messages = vec![Message::user(
            "The billing header contains cch=00000 as a placeholder.",
        )];
        let provider = test_provider();
        let body = provider.build_request_body("system prompt", &messages, &[], false);
        let body_json = serde_json::to_string(&body).unwrap();

        let count = body_json.matches("cch=00000").count();
        assert_eq!(count, 2, "expected 2 occurrences of cch=00000 in body");

        let patched = attestation::patch_request_body(&body_json).unwrap();

        let billing_start = patched.find("x-anthropic-billing-header:").unwrap();
        let billing_region = &patched[billing_start..billing_start + 200];
        assert!(!billing_region.contains("cch=00000"));
        assert!(patched.contains("cch=00000"));
    }

    #[test]
    fn test_claude_no_system_prompt_when_empty() {
        let provider = test_provider();

        let body = provider.build_request_body("", &[], &[], false);
        assert!(body.get("system").is_none());
    }

    #[test]
    fn test_claude_parse_missing_tool_use_id() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "name": "read_file",
                "input": {"path": "/tmp/test.txt"}
            }],
            "stop_reason": "tool_use"
        });

        let result = parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_claude_parse_missing_tool_use_name() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_abc",
                "input": {"path": "/tmp/test.txt"}
            }],
            "stop_reason": "tool_use"
        });

        let result = parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_patch_request_body_cch_in_tool_result() {
        let messages = vec![
            Message::user("run the tool"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"command": "echo cch=00000"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "output: cch=00000".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];
        let provider = test_provider();
        let body = provider.build_request_body("system prompt", &messages, &[], false);
        let body_json = serde_json::to_string(&body).unwrap();
        assert!(body_json.matches("cch=00000").count() >= 2);

        let patched = attestation::patch_request_body(&body_json).unwrap();

        let billing_start = patched.find("x-anthropic-billing-header:").unwrap();
        let billing_end = patched[billing_start..].find(';').unwrap() + billing_start;
        let billing_region = &patched[billing_start..billing_end + 30];
        assert!(!billing_region.contains("cch=00000"));
        assert!(patched.contains("output: cch=00000"));
    }

    #[test]
    fn test_patch_request_body_preserves_length() {
        let provider = test_provider();
        let body = provider.build_request_body("prompt", &[Message::user("hi")], &[], false);
        let body_json = serde_json::to_string(&body).unwrap();
        let patched = attestation::patch_request_body(&body_json).unwrap();
        assert_eq!(body_json.len(), patched.len());
    }

    #[test]
    fn test_claude_request_body_stream_true() {
        let provider = test_provider();
        let body = provider.build_request_body("prompt", &[Message::user("hi")], &[], true);
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn test_claude_request_body_system_and_tools_together() {
        let provider = test_provider();
        let tools = vec![ToolDefinition::new(
            "bash".to_string(),
            "Run a shell command".to_string(),
            serde_json::json!({"type": "object", "properties": {}}),
        )];
        let body =
            provider.build_request_body("system prompt", &[Message::user("hi")], &tools, true);

        assert!(body.get("system").is_some());
        assert!(body.get("tools").is_some());
        assert_eq!(body["stream"], true);

        let json = serde_json::to_string(&body).unwrap();
        assert!(json.find("\"system\"").unwrap() < json.find("\"messages\"").unwrap());

        let tools_array = body["tools"].as_array().unwrap();
        assert!(tools_array.last().unwrap().get("cache_control").is_some());
    }

    #[test]
    fn test_claude_request_body_metadata_fields() {
        let provider = test_provider();
        let body = provider.build_request_body("prompt", &[Message::user("hi")], &[], false);

        let user_id_str = body["metadata"]["user_id"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(user_id_str).unwrap();

        assert_eq!(parsed["device_id"], "a".repeat(64));
        assert_eq!(parsed["account_uuid"], "");
        let session_id = parsed["session_id"].as_str().unwrap();
        assert!(Uuid::parse_str(session_id).is_ok(), "{}", session_id);
    }

    #[test]
    fn test_claude_request_body_no_tools_key_when_empty() {
        let provider = test_provider();
        let body = provider.build_request_body("prompt", &[Message::user("hi")], &[], false);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn test_claude_parse_missing_content_array() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "stop_reason": "end_turn"
        });
        let result = parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_claude_parse_missing_stop_reason_defaults_to_end_turn() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}]
        });
        let (_, stop_reason, _) = parse_non_streaming_response(&response).expect("should parse");
        assert_eq!(stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn test_claude_parse_max_tokens_stop_reason() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "truncated"}],
            "stop_reason": "max_tokens"
        });
        let (_, stop_reason, _) = parse_non_streaming_response(&response).expect("should parse");
        assert_eq!(stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn test_claude_parse_unknown_stop_reason() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "something_new"
        });
        let (_, stop_reason, _) = parse_non_streaming_response(&response).expect("should parse");
        assert_eq!(
            stop_reason,
            StopReason::Unknown("something_new".to_string())
        );
    }

    #[test]
    fn test_claude_parse_empty_content_array() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [],
            "stop_reason": "end_turn"
        });
        let (message, ..) = parse_non_streaming_response(&response).expect("should parse");
        assert!(message.content.is_empty());
        assert_eq!(message.text_content(), "");
    }

    #[test]
    fn test_claude_parse_thinking_block() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm..."},
                {"type": "text", "text": "answer"}
            ],
            "stop_reason": "end_turn"
        });
        let (message, ..) = parse_non_streaming_response(&response).expect("should parse");
        assert_eq!(message.content.len(), 2);
        assert!(
            matches!(&message.content[0], ContentBlock::Thinking { thinking, .. } if thinking == "hmm...")
        );
        assert_eq!(message.text_content(), "answer");
    }

    #[test]
    fn test_claude_parse_unknown_block_type_skipped() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "totally_unknown", "data": "xyz"},
                {"type": "text", "text": "answer"}
            ],
            "stop_reason": "end_turn"
        });
        let (message, ..) = parse_non_streaming_response(&response).expect("should parse");
        assert_eq!(message.content.len(), 1);
        assert_eq!(message.text_content(), "answer");
    }

    #[test]
    fn test_claude_parse_tool_use_missing_input_defaults() {
        let response = serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "list_files"
            }],
            "stop_reason": "tool_use"
        });
        let (message, ..) = parse_non_streaming_response(&response).expect("should parse");
        if let ContentBlock::ToolUse { input, .. } = &message.content[0] {
            assert_eq!(*input, serde_json::json!({}));
        } else {
            panic!("expected ToolUse block");
        }
    }

    fn provider_with(model: &str, thinking: bool) -> ClaudeOAuthProvider {
        provider_full(model, thinking, "high", false)
    }

    fn provider_full(
        model: &str,
        thinking: bool,
        effort: &str,
        redact_thinking: bool,
    ) -> ClaudeOAuthProvider {
        ClaudeOAuthProvider::new(
            AuthCredential::ApiKey("test-key".to_string()),
            model.to_string(),
            None,
            None,
            None,
            None,
            thinking,
            10000,
            "a".repeat(64),
            effort.to_string(),
            redact_thinking,
            None,
        )
    }

    #[test]
    fn test_betas_minimal_no_thinking_non_haiku() {
        // Sonnet 4.0 (not adaptive, not effort-capable), no thinking.
        let provider = provider_with("claude-sonnet-4-20250514", false);
        let betas = provider.compute_betas().unwrap();
        let parts: Vec<&str> = betas.split(',').collect();
        assert_eq!(
            parts,
            vec![
                "claude-code-20250219",
                "oauth-2025-04-20",
                "context-management-2025-06-27",
                "prompt-caching-scope-2026-01-05",
            ],
            "minimal beta set for non-adaptive non-effort sonnet without thinking"
        );
    }

    #[test]
    fn test_betas_haiku_skips_claude_code_and_effort() {
        let provider = provider_with("claude-haiku-4-5-20251001", false);
        let betas = provider.compute_betas().unwrap();
        assert!(
            !betas.contains("claude-code-20250219"),
            "claude-code beta must be skipped for Haiku models: {}",
            betas
        );
        assert!(
            !betas.contains("effort-2025-11-24"),
            "effort beta must be skipped for Haiku models: {}",
            betas
        );
        // Haiku 4.5 still supports modern features (context management, etc.) and OAuth +
        // prompt-caching-scope are unconditional.
        assert!(betas.contains("oauth-2025-04-20"), "{}", betas);
        assert!(betas.contains("context-management-2025-06-27"), "{}", betas);
        assert!(
            betas.contains("prompt-caching-scope-2026-01-05"),
            "{}",
            betas
        );
    }

    #[test]
    fn test_betas_adaptive_with_thinking_matches_wire_dump() {
        // Mirrors the user-supplied wire dump for claude-cli/2.1.41 with an adaptive-capable model
        // and thinking enabled.
        let provider = provider_with("claude-opus-4-6-20250514", true);
        let betas = provider.compute_betas().unwrap();
        assert_eq!(
            betas,
            "claude-code-20250219,oauth-2025-04-20,adaptive-thinking-2026-01-28,\
             context-management-2025-06-27,prompt-caching-scope-2026-01-05,effort-2025-11-24",
            "must exactly match the recent Claude Code wire dump"
        );
    }

    #[test]
    fn test_betas_interleaved_for_non_adaptive_with_thinking() {
        // Sonnet 4.0 supports thinking but NOT adaptive thinking, so the older interleaved-thinking
        // beta is sent instead of adaptive-thinking.
        let provider = provider_with("claude-sonnet-4-20250514", true);
        let betas = provider.compute_betas().unwrap();
        assert!(
            betas.contains("interleaved-thinking-2025-05-14"),
            "non-adaptive thinking model should send interleaved-thinking beta: {}",
            betas
        );
        assert!(
            !betas.contains("adaptive-thinking-2026-01-28"),
            "non-adaptive model must not send adaptive-thinking beta: {}",
            betas
        );
        assert!(
            !betas.contains("effort-2025-11-24"),
            "sonnet-4 (not 4-6) is not effort-capable: {}",
            betas
        );
    }

    #[test]
    fn test_betas_no_thinking_beta_when_thinking_disabled() {
        let provider = provider_with("claude-opus-4-6-20250514", false);
        let betas = provider.compute_betas().unwrap();
        assert!(
            !betas.contains("adaptive-thinking-2026-01-28"),
            "no thinking beta when thinking is off: {}",
            betas
        );
        assert!(
            !betas.contains("interleaved-thinking-2025-05-14"),
            "no thinking beta when thinking is off: {}",
            betas
        );
        // effort beta is independent of thinking; opus-4-6 supports it.
        assert!(betas.contains("effort-2025-11-24"), "{}", betas);
    }

    #[test]
    fn test_betas_oauth_and_prompt_caching_scope_always_present() {
        for model in [
            "claude-opus-4-6-20250514",
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
        ] {
            let provider = provider_with(model, false);
            let betas = provider.compute_betas().unwrap();
            assert!(betas.contains("oauth-2025-04-20"), "{} → {}", model, betas);
            assert!(
                betas.contains("prompt-caching-scope-2026-01-05"),
                "{} → {}",
                model,
                betas
            );
        }
    }

    #[test]
    fn test_context_management_body_when_thinking_enabled() {
        let provider = provider_with("claude-opus-4-6-20250514", true);
        let body = provider.build_request_body("system prompt", &[Message::user("hi")], &[], false);
        let cm = body
            .get("context_management")
            .expect("context_management should be present when thinking is on");
        assert_eq!(cm["edits"][0]["type"], "clear_thinking_20251015");
        assert_eq!(cm["edits"][0]["keep"], "all");
    }

    #[test]
    fn test_output_config_effort_uses_configured_value() {
        for value in ["low", "medium", "high"] {
            let provider = provider_full("claude-opus-4-6-20250514", false, value, false);
            let body =
                provider.build_request_body("system prompt", &[Message::user("hi")], &[], false);
            let oc = body
                .get("output_config")
                .unwrap_or_else(|| panic!("output_config missing for effort={}", value));
            assert_eq!(
                oc["effort"], value,
                "effort body field must reflect configured value"
            );
        }
    }

    #[test]
    fn test_output_config_omitted_when_model_does_not_support_effort() {
        // sonnet-4 (not 4-6) is not effort-capable.
        let provider = provider_full("claude-sonnet-4-20250514", false, "high", false);
        let body = provider.build_request_body("system prompt", &[Message::user("hi")], &[], false);
        assert!(
            body.get("output_config").is_none(),
            "output_config must be omitted when model lacks effort support"
        );
    }

    #[test]
    fn test_betas_redact_thinking_added_when_enabled() {
        // Adaptive-thinking-capable model + thinking on + redact_thinking on.
        let provider = provider_full("claude-opus-4-6-20250514", true, "high", true);
        let betas = provider.compute_betas().unwrap();
        assert!(
            betas.contains("redact-thinking-2026-02-12"),
            "redact-thinking beta must be present when redact_thinking=true: {}",
            betas
        );
    }

    #[test]
    fn test_betas_redact_thinking_omitted_when_disabled() {
        let provider = provider_full("claude-opus-4-6-20250514", true, "high", false);
        let betas = provider.compute_betas().unwrap();
        assert!(
            !betas.contains("redact-thinking-2026-02-12"),
            "redact-thinking beta must be omitted when redact_thinking=false: {}",
            betas
        );
    }

    #[test]
    fn test_betas_redact_thinking_omitted_when_thinking_disabled() {
        // The beta only makes sense when thinking is also enabled — Claude Code's
        // `getAllModelBetas` gates it on `modelSupportsISP(model)` (which we collapse into
        // `model_supports_modern_features`) AND we additionally gate on the thinking toggle since
        // there's no thinking stream to redact when thinking is off.
        let provider = provider_full("claude-opus-4-6-20250514", false, "high", true);
        let betas = provider.compute_betas().unwrap();
        assert!(
            !betas.contains("redact-thinking-2026-02-12"),
            "redact-thinking beta must be omitted when thinking is off: {}",
            betas
        );
    }

    #[test]
    fn test_context_management_body_absent_when_thinking_disabled() {
        let provider = provider_with("claude-opus-4-6-20250514", false);
        let body = provider.build_request_body("system prompt", &[Message::user("hi")], &[], false);
        assert!(
            body.get("context_management").is_none(),
            "context_management must be omitted when thinking is off"
        );
    }

    /// All `cache_control` markers carry `ttl: "1h"` to match recent Claude Code's
    /// `getCacheControl` (returns `{type:"ephemeral", ttl:"1h"}` for OAuth subscribers via
    /// `should1hCacheTTL`).
    #[test]
    fn test_cache_control_uses_one_hour_ttl_everywhere() {
        let provider = test_provider();
        let tools = vec![ToolDefinition::new(
            "read_file",
            "Read a file",
            serde_json::json!({"type": "object"}),
        )];
        let body = provider.build_request_body(
            "user system prompt",
            &[Message::user("hi")],
            &tools,
            false,
        );

        let expected = serde_json::json!({"type": "ephemeral", "ttl": "1h"});

        // System: only the user prompt block (system[2]) has cache_control.
        let system = body["system"].as_array().unwrap();
        assert!(system[0].get("cache_control").is_none());
        assert!(system[1].get("cache_control").is_none());
        assert_eq!(system[2]["cache_control"], expected);

        // Tools: last tool carries cache_control with ttl=1h.
        let tools_arr = body["tools"].as_array().unwrap();
        assert_eq!(
            tools_arr.last().unwrap().get("cache_control").unwrap(),
            &expected,
        );

        // Messages: last block of the last message carries cache_control with ttl=1h.
        let messages_arr = body["messages"].as_array().unwrap();
        let last_msg = messages_arr.last().unwrap();
        let last_block = last_msg["content"].as_array().unwrap().last().unwrap();
        assert_eq!(last_block["cache_control"], expected);
    }

    #[test]
    fn test_now_epoch_millis_reasonable() {
        let ms = now_epoch_millis();
        assert!(ms > 1_577_836_800_000);
        assert!(ms < 4_102_444_800_000);
    }

    // Cache prefix stability tests. These tests simulate multi-turn conversations and tool-use
    // loops to verify that the serialized request bodies share a stable prefix across successive
    // API calls, which is the fundamental requirement for KV cache reuse. A "prefix" here means:
    // the system prompt, tool schemas, and all previously-sent messages must serialize identically
    // (ignoring the `cache_control` marker, which intentionally moves to the newest tail).

    /// Strips every `cache_control` key from every content block in a message so two messages can
    /// be compared purely on semantic content.
    fn strip_cache_control(message: &serde_json::Value) -> serde_json::Value {
        let mut message = message.clone();
        if let Some(content) = message.get_mut("content").and_then(|c| c.as_array_mut()) {
            for block in content.iter_mut() {
                if let Some(obj) = block.as_object_mut() {
                    obj.remove("cache_control");
                }
            }
        }
        message
    }

    /// Strips `cache_control` from every tool schema in an array.
    fn strip_tool_cache_control(tools: &[serde_json::Value]) -> Vec<serde_json::Value> {
        tools
            .iter()
            .map(|tool| {
                let mut tool = tool.clone();
                if let Some(obj) = tool.as_object_mut() {
                    obj.remove("cache_control");
                }
                tool
            })
            .collect()
    }

    /// Asserts that the first `shared_count` messages in two request bodies are semantically
    /// identical (ignoring `cache_control` movement), and that the system prompt and tool schemas
    /// are identical.
    fn assert_prefix_stable(
        body_a: &serde_json::Value,
        body_b: &serde_json::Value,
        shared_message_count: usize,
    ) {
        // System prompt must be byte-identical (before cch patching).
        assert_eq!(
            body_a["system"], body_b["system"],
            "system prompt diverged between requests"
        );

        // Tool schemas must be identical (content-wise, ignoring cache_control which is always on
        // the last tool and doesn't affect tokens).
        let tools_a = body_a["tools"].as_array();
        let tools_b = body_b["tools"].as_array();
        match (tools_a, tools_b) {
            (Some(a), Some(b)) => {
                assert_eq!(
                    strip_tool_cache_control(a),
                    strip_tool_cache_control(b),
                    "tool schemas diverged between requests"
                );
            }
            (None, None) => {}
            _ => panic!("tools presence diverged between requests"),
        }

        let msgs_a = body_a["messages"]
            .as_array()
            .expect("messages array in body_a");
        let msgs_b = body_b["messages"]
            .as_array()
            .expect("messages array in body_b");

        assert!(
            msgs_a.len() >= shared_message_count,
            "body_a has {} messages, expected at least {}",
            msgs_a.len(),
            shared_message_count
        );
        assert!(
            msgs_b.len() >= shared_message_count,
            "body_b has {} messages, expected at least {}",
            msgs_b.len(),
            shared_message_count
        );

        for i in 0..shared_message_count {
            let a = strip_cache_control(&msgs_a[i]);
            let b = strip_cache_control(&msgs_b[i]);
            assert_eq!(a, b, "message at index {} diverged between requests", i);
        }
    }

    /// Counts the total number of `cache_control` markers across all content blocks in the messages
    /// array.
    fn count_message_cache_controls(body: &serde_json::Value) -> usize {
        let mut count = 0;
        if let Some(messages) = body["messages"].as_array() {
            for message in messages {
                if let Some(content) = message["content"].as_array() {
                    for block in content {
                        if block.get("cache_control").is_some() {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    fn test_tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::new(
                "read_file".to_string(),
                "Read a file".to_string(),
                serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            ),
            ToolDefinition::new(
                "execute_command".to_string(),
                "Run a shell command".to_string(),
                serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
            ),
        ]
    }

    #[test]
    fn test_multi_turn_prefix_is_stable() {
        let provider = test_provider();
        let system = "You are a helpful assistant.";
        let tools = test_tools();

        // Turn 1: single user message
        let messages_t1 = vec![Message::user("What files are in /tmp?")];
        let body_t1 = provider.build_request_body(system, &messages_t1, &tools, true);

        // Turn 2: previous turn + assistant response + new user message
        let messages_t2 = vec![
            Message::user("What files are in /tmp?"),
            Message::assistant_text("There are 3 files in /tmp."),
            Message::user("Show me the first one."),
        ];
        let body_t2 = provider.build_request_body(system, &messages_t2, &tools, true);

        // Turn 3: previous turns + another exchange
        let messages_t3 = vec![
            Message::user("What files are in /tmp?"),
            Message::assistant_text("There are 3 files in /tmp."),
            Message::user("Show me the first one."),
            Message::assistant_text("Here is the content of file1.txt."),
            Message::user("Delete it."),
        ];
        let body_t3 = provider.build_request_body(system, &messages_t3, &tools, true);

        // The first message is shared across all three requests.
        assert_prefix_stable(&body_t1, &body_t2, 1);
        // The first three messages are shared between turn 2 and turn 3.
        assert_prefix_stable(&body_t2, &body_t3, 3);
        // The first message is shared across turn 1 and turn 3.
        assert_prefix_stable(&body_t1, &body_t3, 1);
    }

    /// Simulates a two-turn conversation where the user toggles the permission level between turns
    /// and verifies that the cacheable prefix (system prompt + tools array + historical messages)
    /// is byte-identical across the toggle. This is the regression guard for Option 1 of the
    /// higher-permission-visibility work — it proves that `/permission <level>` mid-session does
    /// not invalidate the Claude prompt cache.
    ///
    /// Covers the full agent request-body assembly:
    ///   - [`ToolRegistry::tool_catalogue`] / [`ToolRegistry::definitions_active`]
    ///   - [`crate::context::build_system_prompt`]
    ///   - [`crate::context::build_turn_context`]
    ///   - [`ClaudeOAuthProvider::build_request_body`]
    #[tokio::test]
    async fn test_permission_toggle_preserves_cache_prefix() {
        use std::path::Path;

        use crate::{
            context::{build_system_prompt, build_turn_context},
            permission::{Permission, SharedPermission},
            session::SessionManager,
            tools::ToolRegistry,
        };

        let session_manager = SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("in-memory session manager");
        let shared_permission =
            SharedPermission::new(Permission::Read, crate::permission::EnabledPermissions::ALL);
        let shared_session_id = std::sync::Arc::new(tokio::sync::RwLock::new(None));
        let todo_list = std::sync::Arc::new(tokio::sync::RwLock::new(Vec::new()));
        let registry = ToolRegistry::build_default(
            crate::config::WebClientConfig::default(),
            shared_permission,
            true,
            crate::sandbox::detect(),
            crate::config::SandboxBackend::Landlock,
            crate::sandbox::BackendProbe::Missing {
                reason: "test fixture".to_string(),
            },
            todo_list,
            session_manager,
            shared_session_id,
            crate::skills::SkillCache::for_root(None),
            crate::tools::BuiltinToolFilter::default(),
            crate::agent::test_cwd(),
            std::sync::Arc::new(crate::frontend::SilentFrontend),
        )
        .expect("default web client config should build cleanly");

        let provider = test_provider();

        // The agent fetches these once per turn. None of them take the current permission — that's
        // the invariant we're testing.
        let catalogue = registry.tool_catalogue();
        let system = build_system_prompt(&catalogue, true, &[], None, &[]);
        let tools = registry.definitions_active(&[]);

        let u1_text = {
            let block = build_turn_context(Permission::Read, &[], std::path::Path::new("."));
            format!("{}\n\n{}", block, "list files under /tmp")
        };
        let messages_t1 = vec![Message::user(&u1_text)];
        let body_t1 = provider.build_request_body(&system, &messages_t1, &tools, true);

        // Simulate a `/permission write` toggle: in real code this happens on a different thread
        // via `SharedPermission::set`; here we just re-read the catalogue and rebuild everything to
        // prove the outputs don't depend on the live permission state.

        let catalogue_t2 = registry.tool_catalogue();
        let system_t2 = build_system_prompt(&catalogue_t2, true, &[], None, &[]);
        let tools_t2 = registry.definitions_active(&[]);

        let u2_text = {
            let block = build_turn_context(Permission::Write, &[], std::path::Path::new("."));
            format!("{}\n\n{}", block, "now write 'hi' to /tmp/out.txt")
        };
        let messages_t2 = vec![
            Message::user(&u1_text),
            Message::assistant_text("There are three files in /tmp."),
            Message::user(&u2_text),
        ];
        let body_t2 = provider.build_request_body(&system_t2, &messages_t2, &tools_t2, true);

        // 1. The system prompt is identical. (Breakpoint 2 cache-hit.)
        assert_eq!(
            body_t1["system"], body_t2["system"],
            "system prompt diverged across /permission toggle — cache prefix invalidated"
        );

        // 2. The tools array is identical. (Breakpoint 3 cache-hit.) Reuse the existing helper
        //    which tolerates cache_control movement between the last-tool position across requests.
        assert_prefix_stable(&body_t1, &body_t2, 1);

        // 3. The turn-1 user message is preserved verbatim in turn-2's history — historical
        //    messages must never mutate on toggle, otherwise breakpoint 4 (messages cache)
        //    cascades.
        let t1_msg = strip_cache_control(&body_t1["messages"][0]);
        let t2_msg0 = strip_cache_control(&body_t2["messages"][0]);
        assert_eq!(
            t1_msg, t2_msg0,
            "turn-1 user message changed after permission toggle"
        );

        // 4. Sanity: the two user messages do differ in their permission context (fresh content on
        //    each turn, not cached yet).
        assert!(u1_text.contains("Current permission level: read"));
        assert!(u2_text.contains("Current permission level: write"));
        assert_ne!(u1_text, u2_text);
    }

    /// `load_tool` activation must NOT mutate the cacheable system prompt. This is the regression
    /// guard for the deferred-tool refactor: when the model invokes `load_tool` to expose a
    /// deferred tool's schema, the system prompt block stays byte-identical (so breakpoint 2 cache
    /// hits) — the tools array is what grows, append-only, so its prior entries also cache
    /// (breakpoint 3).
    ///
    /// Mirrors [`test_permission_toggle_preserves_cache_prefix`] structurally.
    #[tokio::test]
    async fn test_load_tool_preserves_system_prompt_cache() {
        use std::path::Path;

        use crate::{
            context::{build_system_prompt, build_turn_context},
            permission::{Permission, SharedPermission},
            session::SessionManager,
            tools::ToolRegistry,
        };

        let session_manager = SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("in-memory session manager");
        let shared_permission = SharedPermission::new(
            Permission::Write,
            crate::permission::EnabledPermissions::ALL,
        );
        let shared_session_id = std::sync::Arc::new(tokio::sync::RwLock::new(None));
        let todo_list = std::sync::Arc::new(tokio::sync::RwLock::new(Vec::new()));
        let registry = ToolRegistry::build_default(
            crate::config::WebClientConfig::default(),
            shared_permission,
            true,
            crate::sandbox::detect(),
            crate::config::SandboxBackend::Landlock,
            crate::sandbox::BackendProbe::Missing {
                reason: "test fixture".to_string(),
            },
            todo_list,
            session_manager,
            shared_session_id,
            crate::skills::SkillCache::for_root(None),
            crate::tools::BuiltinToolFilter::default(),
            crate::agent::test_cwd(),
            std::sync::Arc::new(crate::frontend::SilentFrontend),
        )
        .expect("default web client config should build cleanly");
        // Register a deferred fixture *after* `build_default` so it lands at the tail of the tools
        // vector. Loading it later appends to the end of the API tools array, which is the
        // append-only growth shape the cache prefix invariant relies on.
        crate::tools::tests::register_deferred_fixture(&registry, "fixture_deferred");

        let provider = test_provider();
        let catalogue = registry.tool_catalogue();
        let system = build_system_prompt(&catalogue, true, &[], None, &[]);

        // Turn 1: empty history, fixture_deferred not yet exposed.
        let u1_text = {
            let block = build_turn_context(Permission::Write, &[], std::path::Path::new("."));
            format!("{}\n\n{}", block, "investigate scratchpad")
        };
        let messages_t1 = vec![Message::user(&u1_text)];
        let tools_t1 = registry.definitions_active(&messages_t1);
        let body_t1 = provider.build_request_body(&system, &messages_t1, &tools_t1, true);

        assert!(
            !tools_t1.iter().any(|t| t.name == "fixture_deferred"),
            "fixture_deferred should be deferred in turn 1"
        );

        // Turn 2: the model has called `load_tool` for fixture_deferred, so the next request should
        // expose its schema.
        let messages_t2 = vec![
            Message::user(&u1_text),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "load_tool".to_string(),
                    input: serde_json::json!({"name": "fixture_deferred"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "schema available".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];
        // System prompt is rebuilt the same way every turn — its content is a function of the
        // catalogue, not the messages, so it must not shift when load_tool is invoked.
        let catalogue_t2 = registry.tool_catalogue();
        let system_t2 = build_system_prompt(&catalogue_t2, true, &[], None, &[]);
        let tools_t2 = registry.definitions_active(&messages_t2);
        let body_t2 = provider.build_request_body(&system_t2, &messages_t2, &tools_t2, true);

        // 1. The system prompt is byte-identical. (Breakpoint 2 cache-hit.)
        assert_eq!(
            body_t1["system"], body_t2["system"],
            "system prompt diverged across load_tool invocation — cache prefix invalidated"
        );

        // 2. The tools array gained fixture_deferred (append-only growth).
        assert!(
            tools_t2.iter().any(|t| t.name == "fixture_deferred"),
            "fixture_deferred should be active in turn 2 after load_tool"
        );
        assert_eq!(
            tools_t2.len(),
            tools_t1.len() + 1,
            "tools array should grow by exactly one entry after load_tool"
        );

        // 3. The prior tools (turn-1 set) are present in turn-2 in the same relative order — i.e.,
        //    the prefix is preserved. Stripping cache_control because the marker moves to the new
        //    last tool.
        let tools_arr_t1 =
            strip_tool_cache_control(body_t1["tools"].as_array().expect("tools array in body_t1"));
        let tools_arr_t2 =
            strip_tool_cache_control(body_t2["tools"].as_array().expect("tools array in body_t2"));
        for (idx, tool) in tools_arr_t1.iter().enumerate() {
            assert_eq!(
                &tools_arr_t2[idx], tool,
                "tool at index {} mutated between turns — cache prefix invalidated",
                idx
            );
        }
    }

    /// Compaction must not silently drop the deferred-tool active set. Pre-compaction, the model
    /// loads a deferred fixture via `load_tool`; post-compaction, the
    /// `Event::CompactBoundary::loaded_tools_snapshot` must keep the loaded tool in the API tools
    /// array even though the pre-compaction `load_tool` rows have moved below the materialized
    /// view's logical start.
    #[tokio::test]
    async fn test_compaction_preserves_loaded_tools_active_set() {
        use std::path::Path;

        use crate::{
            conversation::{Conversation, Event, extract_loaded_tool_names_from_events},
            permission::{Permission, SharedPermission},
            session::SessionManager,
            tools::ToolRegistry,
        };

        let session_manager = SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("in-memory session manager");
        let shared_permission = SharedPermission::new(
            Permission::Write,
            crate::permission::EnabledPermissions::ALL,
        );
        let shared_session_id = std::sync::Arc::new(tokio::sync::RwLock::new(None));
        let todo_list = std::sync::Arc::new(tokio::sync::RwLock::new(Vec::new()));
        let registry = ToolRegistry::build_default(
            crate::config::WebClientConfig::default(),
            shared_permission,
            true,
            crate::sandbox::detect(),
            crate::config::SandboxBackend::Landlock,
            crate::sandbox::BackendProbe::Missing {
                reason: "test fixture".to_string(),
            },
            todo_list,
            session_manager,
            shared_session_id,
            crate::skills::SkillCache::for_root(None),
            crate::tools::BuiltinToolFilter::default(),
            crate::agent::test_cwd(),
            std::sync::Arc::new(crate::frontend::SilentFrontend),
        )
        .expect("default web client config should build cleanly");
        crate::tools::tests::register_deferred_fixture(&registry, "fixture_deferred");

        // Pre-compaction: load fixture_deferred via load_tool.
        let mut log = Conversation::new();
        log.append(Message::user("question 1"));
        log.append(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "u1".to_string(),
                name: "load_tool".to_string(),
                input: serde_json::json!({"name": "fixture_deferred"}),
            }],
        });
        log.append(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "u1".to_string(),
                content: vec![ToolResultContent::Text {
                    text: "loaded".to_string(),
                }],
                is_error: false,
            }],
        });

        let pre_loaded = extract_loaded_tool_names_from_events(log.events());
        assert!(pre_loaded.contains("fixture_deferred"));
        let pre_tools = registry.definitions_active_with_loaded(&pre_loaded);
        assert!(pre_tools.iter().any(|t| t.name == "fixture_deferred"));

        // Compact: the snapshot must carry the loaded set forward.
        log.replace_for_compaction(
            Message::user("[summary]"),
            vec![Message::user("question 2")],
            pre_loaded.clone(),
        );

        // The materialized view shrank — but events are append-only.
        let post_loaded = extract_loaded_tool_names_from_events(log.events());
        assert!(
            post_loaded.contains("fixture_deferred"),
            "compaction must preserve the loaded-tools active set via the snapshot"
        );

        // The active tool set the agent sends to the API still includes fixture_deferred
        // post-compaction.
        let post_tools = registry.definitions_active_with_loaded(&post_loaded);
        assert!(post_tools.iter().any(|t| t.name == "fixture_deferred"));

        // The post-compaction event log must have grown, never shrunk.
        let boundary_count = log
            .events()
            .iter()
            .filter(|e| matches!(e, Event::CompactBoundary { .. }))
            .count();
        assert_eq!(boundary_count, 1);
        let append_count = log
            .events()
            .iter()
            .filter(|e| matches!(e, Event::Append(_)))
            .count();
        // 3 pre-compaction Appends + 1 tail Append = 4.
        assert_eq!(append_count, 4);
    }

    /// Same invariant, but exercises every pairwise permission toggle (16 combinations). Catches
    /// any permission state that sneaks back into the cacheable prefix.
    #[tokio::test]
    async fn test_permission_independence_all_levels() {
        use std::path::Path;

        use crate::{
            context::build_system_prompt,
            permission::{Permission, SharedPermission},
            session::SessionManager,
            tools::ToolRegistry,
        };

        let session_manager = SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("in-memory session manager");
        let shared_permission =
            SharedPermission::new(Permission::Read, crate::permission::EnabledPermissions::ALL);
        let shared_session_id = std::sync::Arc::new(tokio::sync::RwLock::new(None));
        let todo_list = std::sync::Arc::new(tokio::sync::RwLock::new(Vec::new()));
        let registry = ToolRegistry::build_default(
            crate::config::WebClientConfig::default(),
            shared_permission.clone(),
            true,
            crate::sandbox::detect(),
            crate::config::SandboxBackend::Landlock,
            crate::sandbox::BackendProbe::Missing {
                reason: "test fixture".to_string(),
            },
            todo_list,
            session_manager,
            shared_session_id,
            crate::skills::SkillCache::for_root(None),
            crate::tools::BuiltinToolFilter::default(),
            crate::agent::test_cwd(),
            std::sync::Arc::new(crate::frontend::SilentFrontend),
        )
        .expect("default web client config should build cleanly");

        let provider = test_provider();
        let levels = [
            Permission::None,
            Permission::Read,
            Permission::Ask,
            Permission::Write,
        ];

        let mut bodies = Vec::with_capacity(levels.len());
        for &level in &levels {
            shared_permission.set_unchecked(level);
            let catalogue = registry.tool_catalogue();
            let system = build_system_prompt(&catalogue, true, &[], None, &[]);
            let tools = registry.definitions_active(&[]);
            let messages = vec![Message::user("hello")];
            bodies.push(provider.build_request_body(&system, &messages, &tools, true));
        }

        // Every pair must agree on the cacheable prefix.
        for i in 0..bodies.len() {
            for j in (i + 1)..bodies.len() {
                assert_eq!(
                    bodies[i]["system"], bodies[j]["system"],
                    "system prompt differs between {:?} and {:?}",
                    levels[i], levels[j]
                );
                assert_prefix_stable(&bodies[i], &bodies[j], 1);
            }
        }
    }

    #[test]
    fn test_tool_loop_prefix_is_stable() {
        let provider = test_provider();
        let system = "You are a helpful assistant.";
        let tools = test_tools();

        // Iteration 1 of tool loop: user asks, model about to respond
        let messages_iter1 = vec![Message::user("Read /tmp/test.txt")];
        let body_iter1 = provider.build_request_body(system, &messages_iter1, &tools, true);

        // Iteration 2: model made a tool call, tool result came back
        let messages_iter2 = vec![
            Message::user("Read /tmp/test.txt"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test.txt"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "hello world".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];
        let body_iter2 = provider.build_request_body(system, &messages_iter2, &tools, true);

        // Iteration 3: model made another tool call
        let messages_iter3 = vec![
            Message::user("Read /tmp/test.txt"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test.txt"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "hello world".to_string(),
                    }],
                    is_error: false,
                }],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "toolu_2".to_string(),
                    name: "execute_command".to_string(),
                    input: serde_json::json!({"command": "wc -l /tmp/test.txt"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_2".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "1 /tmp/test.txt".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];
        let body_iter3 = provider.build_request_body(system, &messages_iter3, &tools, true);

        // Prefix from iter1 is stable in iter2 and iter3
        assert_prefix_stable(&body_iter1, &body_iter2, 1);
        assert_prefix_stable(&body_iter2, &body_iter3, 3);
        assert_prefix_stable(&body_iter1, &body_iter3, 1);
    }

    #[test]
    fn test_exactly_one_message_cache_control_per_request() {
        let provider = test_provider();
        let system = "You are a helpful assistant.";
        let tools = test_tools();

        // Single message
        let body1 = provider.build_request_body(system, &[Message::user("hello")], &tools, true);
        assert_eq!(count_message_cache_controls(&body1), 1);

        // Three messages
        let body3 = provider.build_request_body(
            system,
            &[
                Message::user("hello"),
                Message::assistant_text("hi"),
                Message::user("bye"),
            ],
            &tools,
            true,
        );
        assert_eq!(count_message_cache_controls(&body3), 1);

        // Five messages with tool use
        let body5 = provider.build_request_body(
            system,
            &[
                Message::user("read file"),
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolUse {
                        id: "t1".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({"path": "/tmp/x"}),
                    }],
                },
                Message {
                    role: Role::User,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: "t1".to_string(),
                        content: vec![ToolResultContent::Text {
                            text: "data".to_string(),
                        }],
                        is_error: false,
                    }],
                },
                Message::assistant_text("Here's the file."),
                Message::user("thanks"),
            ],
            &tools,
            true,
        );
        assert_eq!(count_message_cache_controls(&body5), 1);
    }

    #[test]
    fn test_cache_control_shifts_to_new_last_message() {
        let provider = test_provider();
        let system = "system";

        // Build with 2 messages — cache_control should be on message[1]
        let messages_a = vec![Message::user("hello"), Message::assistant_text("hi")];
        let body_a = provider.build_request_body(system, &messages_a, &[], false);
        let msgs_a = body_a["messages"].as_array().unwrap();

        // Message 0 should NOT have cache_control
        let block_0 = &msgs_a[0]["content"].as_array().unwrap()[0];
        assert!(block_0.get("cache_control").is_none());
        // Message 1 (last) SHOULD have cache_control
        let block_1 = &msgs_a[1]["content"].as_array().unwrap()[0];
        assert!(block_1.get("cache_control").is_some());

        // Now append a third message — cache_control should move to message[2]
        let messages_b = vec![
            Message::user("hello"),
            Message::assistant_text("hi"),
            Message::user("bye"),
        ];
        let body_b = provider.build_request_body(system, &messages_b, &[], false);
        let msgs_b = body_b["messages"].as_array().unwrap();

        // Messages 0 and 1 should NOT have cache_control
        assert!(
            msgs_b[0]["content"].as_array().unwrap()[0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            msgs_b[1]["content"].as_array().unwrap()[0]
                .get("cache_control")
                .is_none()
        );
        // Message 2 (new last) SHOULD have cache_control
        assert!(
            msgs_b[2]["content"].as_array().unwrap()[0]
                .get("cache_control")
                .is_some()
        );
    }

    #[test]
    fn test_system_prompt_identical_across_turns() {
        let provider = test_provider();
        let system = "You are a helpful assistant.";
        let tools = test_tools();

        let body1 = provider.build_request_body(system, &[Message::user("turn 1")], &tools, true);
        let body2 = provider.build_request_body(
            system,
            &[
                Message::user("turn 1"),
                Message::assistant_text("response 1"),
                Message::user("turn 2"),
            ],
            &tools,
            true,
        );
        let body3 = provider.build_request_body(
            system,
            &[
                Message::user("turn 1"),
                Message::assistant_text("response 1"),
                Message::user("turn 2"),
                Message::assistant_text("response 2"),
                Message::user("turn 3"),
            ],
            &tools,
            true,
        );

        // System prompt must be byte-identical across all turns.
        assert_eq!(body1["system"], body2["system"]);
        assert_eq!(body2["system"], body3["system"]);

        // Model, max_tokens, metadata must also be identical.
        assert_eq!(body1["model"], body2["model"]);
        assert_eq!(body1["max_tokens"], body2["max_tokens"]);
        assert_eq!(body1["metadata"], body2["metadata"]);
        assert_eq!(body2["model"], body3["model"]);
        assert_eq!(body2["max_tokens"], body3["max_tokens"]);
        assert_eq!(body2["metadata"], body3["metadata"]);
    }

    #[test]
    fn test_tool_schemas_stable_across_turns() {
        let provider = test_provider();
        let tools = test_tools();

        let body1 = provider.build_request_body("system", &[Message::user("a")], &tools, true);
        let body2 = provider.build_request_body(
            "system",
            &[
                Message::user("a"),
                Message::assistant_text("b"),
                Message::user("c"),
            ],
            &tools,
            true,
        );

        // Tool schemas (including cache_control on the last tool) must be identical when the same
        // tools are provided.
        assert_eq!(body1["tools"], body2["tools"]);
    }

    #[test]
    fn test_multi_block_message_cache_control_on_last_block_only() {
        let provider = test_provider();

        // An assistant message with text + tool_use (multiple blocks)
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "Let me read that file.".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/x"}),
                },
            ],
        }];
        let body = provider.build_request_body("system", &messages, &[], false);
        let msg = &body["messages"].as_array().unwrap()[0];
        let blocks = msg["content"].as_array().unwrap();

        // First block (text) should NOT have cache_control
        assert!(blocks[0].get("cache_control").is_none());
        // Second block (tool_use, the last block of the last message) SHOULD
        assert!(blocks[1].get("cache_control").is_some());
    }

    #[test]
    fn test_long_conversation_prefix_stability() {
        let provider = test_provider();
        let system = "You are a helpful assistant.";
        let tools = test_tools();

        // Build up a 10-turn conversation incrementally and verify each step preserves the prefix
        // from the previous step.
        let mut messages: Vec<Message> = Vec::new();
        let mut previous: Option<(serde_json::Value, usize)> = None;

        for turn in 0..10 {
            messages.push(Message::user(format!("User message {}", turn)));
            let body = provider.build_request_body(system, &messages, &tools, true);

            if let Some((prev_body, prev_msg_count)) = &previous {
                // The shared prefix is exactly the messages that were in the previous request body.
                assert_prefix_stable(prev_body, &body, *prev_msg_count);
            }

            assert_eq!(count_message_cache_controls(&body), 1);

            let msg_count = messages.len();
            // Simulate assistant response
            messages.push(Message::assistant_text(format!("Response {}", turn)));
            previous = Some((body, msg_count));
        }
    }

    #[test]
    fn test_tool_loop_with_multiple_sequential_calls() {
        let provider = test_provider();
        let system = "system";
        let tools = test_tools();

        // Simulate a user request that triggers 4 sequential tool calls. Each iteration of the loop
        // adds an assistant tool_use + user tool_result pair. Verify the prefix is stable across
        // all iterations.
        let mut messages: Vec<Message> = vec![Message::user("do several things")];

        let mut previous_body: Option<serde_json::Value> = None;
        let mut previous_len = 0;

        for i in 0..4 {
            let body = provider.build_request_body(system, &messages, &tools, true);

            if let Some(prev) = &previous_body {
                assert_prefix_stable(prev, &body, previous_len);
            }

            assert_eq!(
                count_message_cache_controls(&body),
                1,
                "iteration {} should have exactly 1 message cache_control",
                i
            );

            previous_len = messages.len();
            previous_body = Some(body);

            // Simulate tool call and result
            messages.push(Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: format!("toolu_{}", i),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": format!("/tmp/file{}", i)}),
                }],
            });
            messages.push(Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: format!("toolu_{}", i),
                    content: vec![ToolResultContent::Text {
                        text: format!("contents of file{}", i),
                    }],
                    is_error: false,
                }],
            });
        }

        // Final body after all tool calls
        let final_body = provider.build_request_body(system, &messages, &tools, true);
        assert_prefix_stable(previous_body.as_ref().unwrap(), &final_body, previous_len);
        assert_eq!(count_message_cache_controls(&final_body), 1);
    }

    #[test]
    fn test_empty_messages_produces_no_cache_control() {
        let provider = test_provider();
        let body = provider.build_request_body("system", &[], &[], false);
        assert_eq!(count_message_cache_controls(&body), 0);
        assert!(body["messages"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_cache_control_on_tool_result_block() {
        let provider = test_provider();

        // When the last message is a tool_result, cache_control should still appear on its last
        // content block.
        let messages = vec![
            Message::user("read file"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/x"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "t1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "file data".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];
        let body = provider.build_request_body("system", &messages, &[], false);
        let msgs = body["messages"].as_array().unwrap();

        // Only the tool_result message (last) should have cache_control
        assert!(
            msgs[0]["content"].as_array().unwrap()[0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            msgs[1]["content"].as_array().unwrap()[0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            msgs[2]["content"].as_array().unwrap()[0]
                .get("cache_control")
                .is_some()
        );
        assert_eq!(count_message_cache_controls(&body), 1);
    }

    #[test]
    fn test_claude_cache_control_on_last_message_only() {
        let provider = test_provider();

        let messages = vec![
            Message::user("first"),
            Message::assistant_text("response"),
            Message::user("second"),
        ];
        let body = provider.build_request_body("system", &messages, &[], false);
        let claude_messages = body["messages"].as_array().unwrap();

        let first_content = claude_messages[0]["content"].as_array().unwrap();
        assert!(first_content[0].get("cache_control").is_none());

        let second_content = claude_messages[1]["content"].as_array().unwrap();
        assert!(second_content[0].get("cache_control").is_none());

        let third_content = claude_messages[2]["content"].as_array().unwrap();
        assert!(third_content[0].get("cache_control").is_some());
    }

    #[test]
    fn test_claude_cache_control_on_last_tool() {
        let provider = test_provider();

        let tools = vec![
            ToolDefinition::new(
                "read_file".to_string(),
                "Read a file".to_string(),
                serde_json::json!({"type": "object"}),
            ),
            ToolDefinition::new(
                "write_file".to_string(),
                "Write a file".to_string(),
                serde_json::json!({"type": "object"}),
            ),
        ];
        let body = provider.build_request_body("system", &[Message::user("hi")], &tools, false);
        let claude_tools = body["tools"].as_array().unwrap();

        assert!(claude_tools[0].get("cache_control").is_none());
        assert!(claude_tools[1].get("cache_control").is_some());
    }

    #[test]
    fn test_claude_no_message_cache_control_when_empty() {
        let provider = test_provider();
        let body = provider.build_request_body("system", &[], &[], false);
        let claude_messages = body["messages"].as_array().unwrap();
        assert!(claude_messages.is_empty());
    }

    /// A minimal in-process OAuth refresh endpoint that counts hits. Returns a valid refresh
    /// response on every call so the provider path completes; the test then asserts the hit count.
    async fn run_mock_refresh_endpoint(
        listener: tokio::net::TcpListener,
        hits: Arc<std::sync::atomic::AtomicUsize>,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        loop {
            let (mut socket, _) = match listener.accept().await {
                Ok(pair) => pair,
                Err(_) => return,
            };
            let hits = Arc::clone(&hits);
            tokio::spawn(async move {
                // Drain enough of the request to know we got a full POST body. The OAuth endpoint
                // sends a small JSON body — read until we see two CRLFs (header end) and then
                // enough bytes to satisfy Content-Length.
                let mut buf = Vec::with_capacity(2048);
                let mut headers_end: Option<usize> = None;
                let mut content_length: Option<usize> = None;
                while headers_end.is_none() {
                    let mut chunk = [0u8; 1024];
                    let n = match socket.read(&mut chunk).await {
                        Ok(0) => return,
                        Ok(n) => n,
                        Err(_) => return,
                    };
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(idx) = find_crlf_crlf(&buf) {
                        headers_end = Some(idx);
                        content_length = parse_content_length(&buf[..idx]);
                    }
                }
                if let (Some(end), Some(len)) = (headers_end, content_length) {
                    let body_start = end + 4;
                    while buf.len() < body_start + len {
                        let mut chunk = [0u8; 1024];
                        let n = match socket.read(&mut chunk).await {
                            Ok(0) => break,
                            Ok(n) => n,
                            Err(_) => return,
                        };
                        buf.extend_from_slice(&chunk[..n]);
                    }
                }
                hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                let body = serde_json::json!({
                    "access_token": "fresh-token-xyz",
                    "refresh_token": "fresh-refresh",
                    "expires_in": 3600,
                })
                .to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    }

    fn find_crlf_crlf(buf: &[u8]) -> Option<usize> {
        buf.windows(4).position(|window| window == b"\r\n\r\n")
    }

    fn parse_content_length(headers: &[u8]) -> Option<usize> {
        let headers = std::str::from_utf8(headers).ok()?;
        for line in headers.split("\r\n") {
            if let Some((name, value)) = line.split_once(':')
                && name.trim().eq_ignore_ascii_case("content-length")
            {
                return value.trim().parse().ok();
            }
        }
        None
    }

    /// When many tasks hit `ensure_valid_credential` against a near-expiry credential at the same
    /// instant, exactly **one** refresh API call must fire. The remaining tasks observe the refresh
    /// that already happened via the post-write-lock re-check inside `ensure_valid_credential` and
    /// return the fresh token without re-firing the refresh. This is the invariant relied on by
    /// multi-session ACP where two sessions can race the same credential at the same time.
    #[tokio::test]
    async fn oauth_refresh_fires_once_under_concurrent_demand() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock OAuth endpoint");
        let local = listener.local_addr().expect("local addr");
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        tokio::spawn(run_mock_refresh_endpoint(listener, Arc::clone(&hits)));

        // Credential whose access token already counts as "expiring soon" (the threshold is 5
        // minutes / 300_000 ms). Setting expires_at to "now" forces every caller into the slow path
        // immediately.
        let credential = AuthCredential::OAuthToken {
            access_token: "stale".to_string(),
            refresh_token: Some("rt".to_string()),
            expires_at: Some(now_epoch_millis()),
            account_id: None,
        };

        let provider = Arc::new(ClaudeOAuthProvider::new(
            credential,
            "claude-sonnet-4-20250514".to_string(),
            None,
            None,
            Some(format!("http://{}/", local)),
            None,
            false,
            10000,
            "a".repeat(64),
            "high".to_string(),
            false,
            None,
        ));

        // Fire many concurrent callers. The exact count isn't load- bearing; we just want enough to
        // make a fan-out plausible if the gate broke.
        let mut handles = Vec::new();
        for _ in 0..16 {
            let provider = Arc::clone(&provider);
            handles.push(tokio::spawn(async move {
                provider
                    .ensure_valid_credential()
                    .await
                    .map(|(_, value)| value)
            }));
        }
        let mut results = Vec::with_capacity(handles.len());
        for handle in handles {
            results.push(handle.await.expect("join").expect("ensure_valid"));
        }

        // Every caller must return the same fresh token — proves they observed the refresh that
        // landed, didn't double-refresh.
        for header in &results {
            assert_eq!(header, "Bearer fresh-token-xyz", "stale token leaked",);
        }

        let observed_hits = hits.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            observed_hits, 1,
            "exactly one refresh API call must fire under concurrent demand; got {}",
            observed_hits,
        );
    }
}
