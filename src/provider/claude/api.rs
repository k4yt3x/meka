//! Direct Claude Messages API provider. Uses `x-api-key` auth without the
//! Claude Code fingerprinting / attestation machinery that `claude-oauth`
//! requires. Intended for users bringing their own `CLAUDE_API_KEY`.

use std::sync::Arc;
use std::sync::atomic::{AtomicI8, Ordering};

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::{AgshError, Result};
use crate::provider::{Message, Provider, StopReason, StreamEvent, TokenUsage, ToolDefinition};

use super::shared::{
    self, convert_messages_to_claude_content, convert_tools_to_claude_tools,
    drive_claude_sse_stream, model_supports_adaptive_thinking, parse_non_streaming_response,
};

pub struct ClaudeApiProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    thinking_enabled: bool,
    thinking_budget_tokens: u64,
    thinking_override: AtomicI8,
    /// Per-session counters incremented when image-redaction events fire.
    session_stats: Option<Arc<crate::stats::SessionStats>>,
}

impl ClaudeApiProvider {
    pub fn new(
        api_key: String,
        model: String,
        base_url: Option<String>,
        thinking_enabled: bool,
        thinking_budget_tokens: u64,
        session_stats: Option<Arc<crate::stats::SessionStats>>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.anthropic.com".to_string()),
            model,
            thinking_enabled,
            thinking_budget_tokens,
            thinking_override: AtomicI8::new(-1),
            session_stats,
        }
    }

    fn is_thinking_enabled(&self) -> bool {
        shared::resolve_thinking_enabled(&self.thinking_override, self.thinking_enabled)
    }

    fn compute_betas(&self) -> Option<String> {
        if self.is_thinking_enabled() {
            Some("interleaved-thinking-2025-05-14".to_string())
        } else {
            None
        }
    }

    pub(super) fn build_request_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> serde_json::Value {
        let claude_messages = convert_messages_to_claude_content(messages);

        let mut body = serde_json::Map::new();
        body.insert("model".to_string(), serde_json::json!(self.model));
        if !system_prompt.is_empty() {
            body.insert("system".to_string(), serde_json::json!(system_prompt));
        }
        body.insert("messages".to_string(), serde_json::json!(claude_messages));

        if self.is_thinking_enabled() {
            if model_supports_adaptive_thinking(&self.model) {
                body.insert("max_tokens".to_string(), serde_json::json!(64_000));
                body.insert(
                    "thinking".to_string(),
                    serde_json::json!({ "type": "adaptive" }),
                );
            } else {
                let budget = self.thinking_budget_tokens;
                let max_tokens = std::cmp::max(budget * 2, 32_000);
                body.insert("max_tokens".to_string(), serde_json::json!(max_tokens));
                body.insert(
                    "thinking".to_string(),
                    serde_json::json!({
                        "type": "enabled",
                        "budget_tokens": budget
                    }),
                );
            }
        } else {
            body.insert("max_tokens".to_string(), serde_json::json!(32_000));
        }

        body.insert("stream".to_string(), serde_json::json!(stream));

        if !tools.is_empty() {
            body.insert(
                "tools".to_string(),
                serde_json::json!(convert_tools_to_claude_tools(tools)),
            );
        }

        serde_json::Value::Object(body)
    }

    /// Mirror of `ClaudeOAuthProvider::build_body_within_budget`: serialize
    /// the request body and, if it exceeds [`shared::MAX_REQUEST_BYTES`],
    /// reactively redact the oldest tool-result image blocks (preserving the
    /// final message's cache breakpoint) and shrink past
    /// [`shared::REDACTION_TARGET_BYTES`] so subsequent turns don't
    /// re-trigger.
    fn build_body_within_budget(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> Result<String> {
        // Anthropic rejects images >2000 px on either axis in multi-image
        // requests. Downscale before serialization. See
        // `shared::downscale_oversized_images` for the rationale.
        let prepared = shared::downscale_oversized_images(messages);
        let messages = prepared.as_ref();

        let body_json =
            serde_json::to_string(&self.build_request_body(system_prompt, messages, tools, stream))
                .map_err(|error| {
                    AgshError::Provider(format!("failed to serialize body: {}", error))
                })?;

        if body_json.len() <= shared::MAX_REQUEST_BYTES {
            return Ok(body_json);
        }

        let bytes_to_drop = body_json.len() - shared::REDACTION_TARGET_BYTES;
        let (redacted, stats) = shared::redact_oldest_images(messages, bytes_to_drop);
        let body_json = serde_json::to_string(&self.build_request_body(
            system_prompt,
            redacted.as_ref(),
            tools,
            stream,
        ))
        .map_err(|error| AgshError::Provider(format!("failed to serialize body: {}", error)))?;

        if body_json.len() > shared::MAX_REQUEST_BYTES {
            return Err(AgshError::Provider(format!(
                "request body is {} MiB after redacting old tool-result images; \
                 Anthropic's limit is 32 MiB. Run /compact, remove large attachments \
                 from the most recent turn, or split the work across smaller turns.",
                body_json.len() / 1_048_576,
            )));
        }

        if let Some(session_stats) = &self.session_stats {
            session_stats.record_redaction(stats.images_redacted as u64, stats.bytes_freed as u64);
        }
        crate::render::render_hint(&format!(
            "Redacted {} old image{} (~{} MiB freed). Cache prefix invalidated for those messages.",
            stats.images_redacted,
            if stats.images_redacted == 1 { "" } else { "s" },
            stats.bytes_freed / 1_048_576,
        ));
        tracing::warn!(
            "redacted {} old tool-result image(s); body now {} MiB",
            stats.images_redacted,
            body_json.len() / 1_048_576,
        );
        Ok(body_json)
    }

    fn apply_headers(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut request = request
            .header("accept", "application/json")
            .header("content-type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .header("x-api-key", &self.api_key);

        if let Some(betas) = self.compute_betas() {
            request = request.header("anthropic-beta", betas);
        }

        request
    }
}

#[async_trait]
impl Provider for ClaudeApiProvider {
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<(Message, StopReason, TokenUsage)> {
        let body_json = self.build_body_within_budget(system_prompt, messages, tools, false)?;
        let body_size_mib = body_json.len() / 1_048_576;
        let request = self
            .apply_headers(self.client.post(format!("{}/v1/messages", self.base_url)))
            .body(body_json);

        let response = request.send().await.map_err(|error| {
            AgshError::Provider(format!(
                "HTTP request failed (body {} MiB): {}",
                body_size_mib, error
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

        parse_non_streaming_response(&response_json)
    }

    async fn stream(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        event_sender: mpsc::UnboundedSender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        let body_json = self.build_body_within_budget(system_prompt, messages, tools, true)?;
        let body_size_mib = body_json.len() / 1_048_576;
        let request = self
            .apply_headers(
                self.client
                    .post(format!("{}/v1/messages", self.base_url))
                    .header("accept-encoding", "identity"),
            )
            .body(body_json);

        let response = request.send().await.map_err(|error| {
            AgshError::Provider(format!(
                "HTTP request failed (body {} MiB): {}",
                body_size_mib, error
            ))
        })?;

        drive_claude_sse_stream(response, event_sender, cancellation).await
    }

    fn name(&self) -> &str {
        "claude-api"
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
    use super::*;

    fn test_provider() -> ClaudeApiProvider {
        ClaudeApiProvider::new(
            "test-key".to_string(),
            "claude-sonnet-4-20250514".to_string(),
            None,
            false,
            10000,
            None,
        )
    }

    #[test]
    fn test_api_body_has_no_billing_header() {
        let provider = test_provider();
        let messages = vec![Message::user("hello")];
        let body = provider.build_request_body("be nice", &messages, &[], false);

        let serialized = serde_json::to_string(&body).unwrap();
        assert!(
            !serialized.contains("cc_version"),
            "claude-api body must not contain Claude Code billing header: {}",
            serialized
        );
        assert!(
            !serialized.contains("cc_entrypoint"),
            "claude-api body must not contain Claude Code entrypoint tag: {}",
            serialized
        );
        assert!(
            !serialized.contains("cch="),
            "claude-api body must not contain cch attestation placeholder: {}",
            serialized
        );
    }

    #[test]
    fn test_api_body_has_no_metadata() {
        let provider = test_provider();
        let body = provider.build_request_body("", &[Message::user("hi")], &[], false);
        assert!(
            body.get("metadata").is_none(),
            "claude-api body must not include metadata.user_id"
        );
    }

    #[test]
    fn test_api_body_plain_string_system_prompt() {
        let provider = test_provider();
        let body = provider.build_request_body("my system", &[Message::user("hi")], &[], false);
        let system = body.get("system").unwrap();
        assert_eq!(
            system.as_str(),
            Some("my system"),
            "claude-api should serialize `system` as a plain string"
        );
    }

    #[test]
    fn test_api_body_omits_system_when_empty() {
        let provider = test_provider();
        let body = provider.build_request_body("", &[Message::user("hi")], &[], false);
        assert!(
            body.get("system").is_none(),
            "claude-api should omit `system` when the prompt is empty"
        );
    }
}
