//! `anthropic-messages`: the Anthropic Messages API against any endpoint serving it, with an API
//! key.
//!
//! `POST {base_url}/v1/messages` with `x-api-key`, and none of the Claude Code fingerprinting or
//! attestation machinery [`super::subscription`] needs: the two speak the same protocol and
//! differ only in how they authenticate and whose client they look like. The wire format is shared
//! through [`super::shared`].
//!
//! The key comes from the profile's stored credential, never from the environment: meka reads no
//! provider env vars at all, so an ambient key cannot silently rebind which account a profile
//! bills.

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::shared::{self, convert_messages_to_claude_content, convert_tools_to_claude_tools};
use crate::{
    config::ThinkingMode,
    conversation::Message,
    error::Result,
    provider::{CompletionRequest, Provider, StreamEvent, ThinkingOverride, ToolDefinition},
};

/// The `anthropic-messages` backend: one profile's model, endpoint and API key.
pub(crate) struct AnthropicMessagesProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    thinking: ThinkingMode,
    thinking_budget_tokens: u64,
    /// The settled `output_config.effort` for the request body, resolved once at construction from
    /// the profile's override. `None` (the unconfigured case) omits the field so Anthropic (or
    /// whatever endpoint `base_url` names) applies its own default. The direct Messages API takes
    /// effort with no beta header.
    resolved_effort: Option<String>,
    /// Per-request output token cap from the profile; `None` keeps the built-in default.
    max_output_tokens: Option<u64>,
    /// See [`crate::config::ProfileConfig::max_request_bytes`].
    max_request_bytes: Option<usize>,
}

impl AnthropicMessagesProvider {
    /// `api_key` is the credential `settings` carries, already checked to be one by the builder.
    pub(crate) fn new(api_key: String, settings: crate::provider::ProviderBuilder) -> Result<Self> {
        let crate::provider::ProviderBuilder {
            model,
            base_url,
            thinking,
            thinking_budget_tokens,
            effort,
            max_output_tokens,
            max_request_bytes,
            ..
        } = settings;
        let resolved_effort = crate::provider::resolve_effort_level(effort.as_deref());
        Ok(Self {
            client: crate::provider::build_http_client("anthropic-messages", |builder| builder)?,
            api_key,
            base_url: shared::normalize_claude_base_url(
                base_url
                    .as_deref()
                    .unwrap_or(crate::provider::DEFAULT_ANTHROPIC_BASE_URL),
            ),
            model,
            thinking,
            thinking_budget_tokens,
            resolved_effort,
            max_output_tokens,
            max_request_bytes,
        })
    }

    fn effective_thinking(&self, thinking: ThinkingOverride) -> ThinkingMode {
        shared::effective_thinking(thinking, self.thinking)
    }

    /// The settled effort to send as `output_config.effort` (see [`Self::resolved_effort`]).
    fn wire_effort(&self) -> Option<String> {
        self.resolved_effort.clone()
    }

    fn compute_betas(&self, thinking: ThinkingOverride) -> Option<String> {
        let mut parts: Vec<&str> = Vec::new();
        // Sent to whatever `base_url` names, on purpose: an endpoint that does not know the beta
        // rejects the request, a visible failure, where omitting it would silently degrade
        // thinking on every endpoint that does.
        if self.effective_thinking(thinking).is_on() {
            parts.push("interleaved-thinking-2025-05-14");
        }
        // No `context-1m-2025-08-07`: on the direct Messages API, 1M context is the *default* for
        // the current large-context models (Opus 4.6+, Sonnet 4.6, Fable 5) with no beta header,
        // so the 1M window is already what the request gets. See
        // <https://platform.claude.com/docs/en/build-with-claude/context-windows>. The
        // claude-subscription path does not send it either; see `compute_betas` there.
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(","))
        }
    }

    pub(super) fn build_request_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        stream: bool,
        thinking: ThinkingOverride,
    ) -> serde_json::Value {
        // The API's own TTL. The one-hour breakpoint needs a beta this backend does not send, and
        // this endpoint is whatever `base_url` names.
        let claude_messages =
            convert_messages_to_claude_content(messages, super::shared::CacheBreakpoint::Ephemeral);

        let mut body = serde_json::Map::new();
        body.insert("model".to_string(), serde_json::json!(self.model));
        if !system_prompt.is_empty() {
            body.insert("system".to_string(), serde_json::json!(system_prompt));
        }
        body.insert("messages".to_string(), serde_json::json!(claude_messages));

        shared::insert_thinking_fields(
            &mut body,
            self.effective_thinking(thinking),
            self.thinking_budget_tokens,
            self.max_output_tokens,
        );

        body.insert("stream".to_string(), serde_json::json!(stream));

        if !tools.is_empty() {
            body.insert(
                "tools".to_string(),
                serde_json::json!(convert_tools_to_claude_tools(tools)),
            );
        }

        // The direct Messages API takes `output_config.effort` in the body with no beta header
        // (unlike claude-subscription, which mirrors Claude Code's `effort-2025-11-24` beta). See
        // <https://platform.claude.com/docs/en/build-with-claude/effort>.
        if let Some(effort) = self.wire_effort() {
            body.insert(
                "output_config".to_string(),
                serde_json::json!({ "effort": effort }),
            );
        }

        serde_json::Value::Object(body)
    }

    fn apply_headers(
        &self,
        request: reqwest::RequestBuilder,
        thinking: ThinkingOverride,
    ) -> reqwest::RequestBuilder {
        let mut request = request
            .header("accept", "application/json")
            .header("content-type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .header("x-api-key", &self.api_key);

        if let Some(betas) = self.compute_betas(thinking) {
            request = request.header("anthropic-beta", betas);
        }

        request
    }
}

impl crate::oauth::RefreshesCredential for AnthropicMessagesProvider {}

#[async_trait]
impl shared::ClaudeBackend for AnthropicMessagesProvider {
    fn client(&self) -> &reqwest::Client {
        &self.client
    }

    fn endpoint(&self) -> String {
        format!("{}/v1/messages", self.base_url)
    }

    fn max_request_bytes(&self) -> usize {
        self.max_request_bytes
            .unwrap_or(super::shared::MAX_REQUEST_BYTES)
    }

    fn request_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        stream: bool,
        thinking: ThinkingOverride,
        _attribution: &crate::provider::Attribution,
    ) -> serde_json::Value {
        self.build_request_body(system_prompt, messages, tools, stream, thinking)
    }

    async fn authenticated_request(
        &self,
        request: reqwest::RequestBuilder,
        _has_tools: bool,
        stream: bool,
        thinking: ThinkingOverride,
    ) -> Result<reqwest::RequestBuilder> {
        let request = if stream {
            request.header("accept-encoding", "identity")
        } else {
            request
        };
        Ok(self.apply_headers(request, thinking))
    }
}

#[async_trait]
impl Provider for AnthropicMessagesProvider {
    async fn complete(
        &self,
        request: CompletionRequest<'_>,
    ) -> Result<crate::provider::Completion> {
        shared::complete(self, request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest<'_>,
        event_sender: mpsc::Sender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        shared::stream(self, request, event_sender, cancellation).await
    }

    fn resolved_effort(&self) -> Option<String> {
        self.wire_effort()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::MekaError;

    fn provider_for_test() -> AnthropicMessagesProvider {
        provider("claude-sonnet-4-20250514", None)
    }

    fn provider(model: &str, effort: Option<&str>) -> AnthropicMessagesProvider {
        provider_with_base(model, effort, None)
    }

    fn provider_with_base(
        model: &str,
        effort: Option<&str>,
        base_url: Option<&str>,
    ) -> AnthropicMessagesProvider {
        {
            let api_key: String = "test-key".to_string();
            AnthropicMessagesProvider::new(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::AnthropicMessages,
                    crate::store::AuthCredential::ApiKey(api_key),
                    model.to_string(),
                )
                .base_url(base_url.map(str::to_string))
                .thinking(ThinkingMode::Off, 10000)
                .effort(effort.map(str::to_string))
                .max_output_tokens(None)
                .max_request_bytes(None),
            )
        }
        .expect("build test provider")
    }

    /// A backend that could not reach its endpoint reports a failure the agent loop will retry.
    ///
    /// This is the wiring rather than the rule: [`crate::error::provider_transport_error`] has its
    /// own tests, and what this one asserts is that a real `.send()` site actually calls it. Every
    /// such site in every backend was hand-rolling a bare `MekaError::Provider`, which the retry
    /// loop discards, so a site that quietly goes back to doing that is the regression worth
    /// catching. One backend stands for the pattern; the classifier they all share is what makes
    /// that enough.
    #[tokio::test]
    async fn a_backend_that_could_not_reach_its_endpoint_reports_a_retryable_failure() {
        // Bound and dropped, so the port is refused rather than answered or hung.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
        let port = listener.local_addr().expect("the bound address").port();
        drop(listener);

        let provider = provider_with_base(
            "claude-sonnet-4-20250514",
            None,
            Some(&format!("http://127.0.0.1:{port}")),
        );
        let error = provider
            .complete(CompletionRequest::new("", &[Message::user("hello")], &[]))
            .await
            .expect_err("nothing is listening there");

        assert!(
            matches!(error, MekaError::RetryableProvider { .. }),
            "a call that never reached the provider must be retryable, got: {error}"
        );
    }

    /// The same, for the streaming path, which is the one an interactive turn actually takes.
    ///
    /// Worth its own test rather than trusting the sibling above: `complete` and `stream` build and
    /// send their requests separately, so they are two wirings, and it was the streaming one that
    /// ended a real session by reporting a connection reset as terminal.
    #[tokio::test]
    async fn a_backend_whose_stream_could_not_start_reports_a_retryable_failure() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
        let port = listener.local_addr().expect("the bound address").port();
        drop(listener);

        let provider = provider_with_base(
            "claude-sonnet-4-20250514",
            None,
            Some(&format!("http://127.0.0.1:{port}")),
        );
        let (sender, _receiver) = mpsc::channel(8);
        let error = provider
            .stream(
                CompletionRequest::new("", &[Message::user("hello")], &[]),
                sender,
                CancellationToken::new(),
            )
            .await
            .expect_err("nothing is listening there");

        assert!(
            matches!(error, MekaError::RetryableProvider { .. }),
            "a stream that never started must be retryable, got: {error}"
        );
    }

    /// A read that fails after the headers arrived keeps the hint those headers carried.
    ///
    /// Parsing `Retry-After` and passing it on are separate acts: a read site that parses the
    /// header into a local and then hands [`crate::error::provider_transport_error`] a `None`
    /// retries a provider that just said "wait 60 seconds" on plain 1s/2s backoff. The classifier's
    /// own tests cannot catch that, because they supply the argument themselves; this one makes a
    /// real site parse a real header. Asserting the message as well as the hint is what pins it to
    /// the read site: a 429 whose body *did* arrive is also retryable and also carries the hint,
    /// but says so in `provider_http_error`'s words.
    #[tokio::test]
    async fn a_truncated_body_keeps_the_rate_limit_hint() {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral port");
        let port = listener.local_addr().expect("the bound address").port();
        tokio::spawn(async move {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            // Read the request out in full before answering. A response written while the client is
            // still uploading can be lost to the reset that closing an unread socket sends.
            let mut request = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                let read = match socket.read(&mut chunk).await {
                    Ok(0) => break,
                    Ok(read) => read,
                    Err(_) => return,
                };
                request.extend_from_slice(&chunk[..read]);
                let Some(head_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
                else {
                    continue;
                };
                let length: usize = String::from_utf8_lossy(&request[..head_end])
                    .split("\r\n")
                    .filter_map(|line| line.split_once(':'))
                    .find(|(name, _)| name.trim().eq_ignore_ascii_case("content-length"))
                    .and_then(|(_, value)| value.trim().parse().ok())
                    .unwrap_or(0);
                if request.len() >= head_end + 4 + length {
                    break;
                }
            }

            // A `Content-Length` the body then stops short of, so the read ends at EOF with the
            // status and headers already delivered.
            let head = "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 60\r\n\
                        Content-Type: application/json\r\nContent-Length: 4096\r\n\r\n";
            if socket.write_all(head.as_bytes()).await.is_err() {
                return;
            }
            if socket.write_all(b"{\"error\":").await.is_err() {
                return;
            }
            if socket.shutdown().await.is_err() {
                tracing::debug!("mock endpoint failed to shut its socket down cleanly");
            }
        });

        let provider = provider_with_base(
            "claude-sonnet-4-20250514",
            None,
            Some(&format!("http://127.0.0.1:{port}")),
        );
        let error = provider
            .complete(CompletionRequest::new("", &[Message::user("hello")], &[]))
            .await
            .expect_err("the body stops short of its declared length");

        match error {
            MekaError::RetryableProvider {
                message,
                retry_after,
                ..
            } => {
                assert!(
                    message.starts_with("failed to read response"),
                    "expected the read site rather than the status classifier: {message}"
                );
                assert_eq!(retry_after, Some(std::time::Duration::from_secs(60)));
            }
            other => panic!("expected a retryable failure carrying the hint, got: {other}"),
        }
    }

    #[test]
    fn a_claude_base_url_is_normalized_at_construction() {
        // The shape a gateway publishes for its Anthropic endpoint; meka appends `/v1/messages`
        // itself, so leaving this would request `/v1/v1/messages`.
        let versioned = provider_with_base(
            "claude-sonnet-4-20250514",
            None,
            Some("https://api.synthetic.new/anthropic/v1"),
        );
        assert_eq!(versioned.base_url, "https://api.synthetic.new/anthropic");

        let trailing = provider_with_base(
            "claude-sonnet-4-20250514",
            None,
            Some("https://api.anthropic.com/"),
        );
        assert_eq!(trailing.base_url, "https://api.anthropic.com");

        assert_eq!(provider_for_test().base_url, "https://api.anthropic.com");
    }

    #[test]
    fn an_unconfigured_profile_sends_no_output_config_whatever_the_model() {
        // No model name earns an effort tier. `anthropic-messages` reaches any Anthropic-compatible
        // endpoint, so meka cannot know which tiers the far side implements; omitting the field is
        // how it asks for whatever that endpoint's default is.
        for model in [
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-sonnet-4-20250514",
            "hf.co/bartowski/Qwen3.8-27B-GGUF:Q8_0",
        ] {
            let body = provider(model, None).build_request_body(
                "s",
                &[Message::user("hi")],
                &[],
                false,
                ThinkingOverride::Inherit,
            );
            assert!(body.get("output_config").is_none(), "{model}");
        }
        // A configured value is absolute: sent verbatim on any model, including one meka has never
        // heard of, because the user knows their endpoint and meka does not.
        for model in ["claude-opus-4-8", "hf.co/bartowski/Qwen3.8-27B-GGUF:Q8_0"] {
            let body = provider(model, Some("medium")).build_request_body(
                "s",
                &[Message::user("hi")],
                &[],
                false,
                ThinkingOverride::Inherit,
            );
            assert_eq!(body["output_config"]["effort"], "medium", "{model}");
        }
    }

    #[test]
    fn betas_omit_context_1m() {
        // The direct Messages API serves 1M context by default for 1M-capable models with no beta
        // header, so anthropic-messages never sends `context-1m-2025-08-07` (unlike
        // claude-subscription, which mirrors Claude Code's captured wire). A
        // thinking-enabled request still sends only the interleaved beta.
        let thinking_on = {
            let api_key: String = "test-key".to_string();
            AnthropicMessagesProvider::new(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::AnthropicMessages,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "claude-opus-4-8".to_string(),
                )
                .base_url(None)
                .thinking(ThinkingMode::Adaptive, 10000)
                .effort(None)
                .max_output_tokens(None)
                .max_request_bytes(None),
            )
        }
        .expect("build test provider");
        let betas = thinking_on
            .compute_betas(ThinkingOverride::Inherit)
            .unwrap_or_default();
        assert!(betas.contains("interleaved-thinking-2025-05-14"));
        assert!(
            !betas.contains("context-1m"),
            "1M is the API default; no beta expected: {betas}"
        );
        // Thinking off on a 1M-capable model → no betas at all.
        assert!(
            provider("claude-opus-4-8", None)
                .compute_betas(ThinkingOverride::Inherit)
                .is_none()
        );
    }

    /// A repaired `tool_use` reaches the wire with the arguments the repair put on it, unexamined.
    ///
    /// `DegradeTier::ToolExchanges` empties a refused call in place, which leaves its `input` no
    /// longer matching the tool's declared schema. That is only safe because a replayed `tool_use`
    /// is a record of what happened rather than a request to validate, so nothing on the way out
    /// should be checking it. This pins meka's half of that: whatever the repair wrote is what gets
    /// sent. The other half, that the provider accepts it, is the provider's and cannot be asserted
    /// here.
    #[test]
    fn a_repaired_tool_use_reaches_the_wire_with_its_arguments_unexamined() {
        let provider = provider_for_test();
        let messages = vec![Message::user("read my notes"), Message {
            role: crate::conversation::Role::Assistant,
            content: vec![crate::conversation::ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                // What the repair leaves behind: nothing the tool's schema declares.
                input: serde_json::json!({"[meka harness]": "arguments removed"}),
            }],
        }];
        let body = provider.build_request_body(
            "be nice",
            &messages,
            &[],
            false,
            ThinkingOverride::Inherit,
        );

        let serialized = serde_json::to_string(&body).expect("serialize");
        assert!(
            serialized.contains("[meka harness]"),
            "the repaired input must survive to the wire verbatim: {serialized}"
        );
        assert!(
            serialized.contains("read_file") && serialized.contains("call_1"),
            "and the call keeps its identity, so its result is not orphaned: {serialized}"
        );
    }

    #[test]
    fn api_body_has_no_billing_header() {
        let provider = provider_for_test();
        let messages = vec![Message::user("hello")];
        let body = provider.build_request_body(
            "be nice",
            &messages,
            &[],
            false,
            ThinkingOverride::Inherit,
        );

        let serialized = serde_json::to_string(&body).unwrap();
        assert!(
            !serialized.contains("cc_version"),
            "anthropic-messages body must not contain Claude Code billing header: {serialized}"
        );
        assert!(
            !serialized.contains("cc_entrypoint"),
            "anthropic-messages body must not contain Claude Code entrypoint tag: {serialized}"
        );
        assert!(
            !serialized.contains("cch="),
            "anthropic-messages body must not contain cch attestation placeholder: {serialized}"
        );
    }

    #[test]
    fn api_body_has_no_metadata() {
        let provider = provider_for_test();
        let body = provider.build_request_body(
            "",
            &[Message::user("hi")],
            &[],
            false,
            ThinkingOverride::Inherit,
        );
        assert!(
            body.get("metadata").is_none(),
            "anthropic-messages body must not include metadata.user_id"
        );
    }

    #[test]
    fn api_body_plain_string_system_prompt() {
        let provider = provider_for_test();
        let body = provider.build_request_body(
            "my system",
            &[Message::user("hi")],
            &[],
            false,
            ThinkingOverride::Inherit,
        );
        let system = body.get("system").unwrap();
        assert_eq!(
            system.as_str(),
            Some("my system"),
            "anthropic-messages should serialize `system` as a plain string"
        );
    }

    #[test]
    fn api_body_omits_system_when_empty() {
        let provider = provider_for_test();
        let body = provider.build_request_body(
            "",
            &[Message::user("hi")],
            &[],
            false,
            ThinkingOverride::Inherit,
        );
        assert!(
            body.get("system").is_none(),
            "anthropic-messages should omit `system` when the prompt is empty"
        );
    }
}
