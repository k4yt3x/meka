//! `openai-responses`: the Responses API against any endpoint that serves it, with an API key.
//!
//! The protocol sibling of [`super::chat_completions`] and the auth sibling of
//! [`super::subscription`]. It posts to `{base_url}/responses` with a bearer token, which reaches
//! OpenAI itself and equally reaches Ollama (v0.13.3+), vLLM, LM Studio and OpenRouter, all of
//! which implement the same endpoint. The wire format lives in [`super::responses_wire`].
//!
//! OpenAI's own fields go out by default, each behind a switch for an endpoint or a model that
//! refuses it: [`super::responses_wire::include_encrypted_reasoning`] and
//! [`super::responses_wire::request_reasoning_summary`] behind the profile's `encrypted_reasoning`
//! and `reasoning_summary`, since a model that does not reason may refuse them where one that
//! does needs them, and `prompt_cache_key` behind the account's, since only the endpoint decides
//! that one. Most servers ignore a field they do not know, and the one that refuses names it, so a
//! switch turned off is the remedy rather than a guess made for every endpoint.

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::responses_wire::{
    ResponsesBackend, build_request_body_for, include_encrypted_reasoning,
    request_reasoning_summary,
};
use crate::{
    conversation::Message,
    error::Result,
    provider::{CompletionRequest, Provider, StreamEvent, ToolDefinition},
};

/// The `openai-responses` backend: one profile's model, endpoint and API key.
pub(crate) struct OpenAiResponsesProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    /// The settled `reasoning.effort` for the request body, resolved once at construction from the
    /// profile's override. `None` (the unconfigured case) omits the `reasoning` block so the
    /// endpoint applies its own default, which matters most for the local servers this backend
    /// also reaches.
    resolved_effort: Option<String>,
    /// Per-request output token cap from the profile; `None` leaves the endpoint's default.
    max_output_tokens: Option<u64>,
    /// See [`crate::config::ProfileConfig::max_request_bytes`]; unset means no ceiling here.
    max_request_bytes: Option<usize>,
    /// See `ProfileConfig::deferred_tools`: whether deferred tools are declared with
    /// `defer_loading` and loaded through the endpoint's own `tool_search`.
    deferred_tools: crate::config::DeferredTools,
    /// See `ProfileConfig::encrypted_reasoning`.
    encrypted_reasoning: bool,
    /// See `ProfileConfig::reasoning_summary`.
    reasoning_summary: bool,
    /// See `AccountConfig::prompt_cache_key`.
    prompt_cache_key: bool,
}

impl OpenAiResponsesProvider {
    /// `api_key` is the credential `settings` carries, already checked to be one by the builder.
    pub(crate) fn from_settings(
        api_key: String,
        settings: crate::provider::ProviderBuilder,
    ) -> Result<Self> {
        let crate::provider::ProviderBuilder {
            model,
            base_url,
            effort: reasoning_effort,
            max_output_tokens,
            max_request_bytes,
            deferred_tools,
            encrypted_reasoning,
            reasoning_summary,
            prompt_cache_key,
            ..
        } = settings;
        let resolved_effort = crate::provider::resolve_effort_level(reasoning_effort.as_deref());
        Ok(Self {
            client: crate::provider::build_http_client("openai-responses", |builder| builder)?,
            api_key,
            // The same normalizer Chat Completions uses, and for the same reason:
            // `{base}/responses` composes exactly as `{base}/chat/completions` does, so
            // `https://api.openai.com/v1`, `http://localhost:11434/v1` and
            // `https://openrouter.ai/api/v1` all work verbatim.
            base_url: crate::provider::normalize_base_url(
                base_url
                    .as_deref()
                    .unwrap_or(crate::provider::DEFAULT_OPENAI_BASE_URL),
            ),
            model,
            resolved_effort,
            max_output_tokens,
            max_request_bytes,
            deferred_tools,
            encrypted_reasoning,
            reasoning_summary,
            prompt_cache_key,
        })
    }

    /// The settled reasoning-effort to send as `reasoning.effort` (see [`Self::resolved_effort`]).
    fn wire_effort(&self) -> Option<String> {
        self.resolved_effort.clone()
    }

    fn responses_url(&self) -> String {
        format!("{}/responses", self.base_url)
    }

    /// The request body, exactly as `stream` sends it.
    ///
    /// A named method rather than inline in `stream`, and the *only* body builder this backend has,
    /// so a test asserting what is on the wire is asserting the shipping path rather than a
    /// parallel copy of it.
    fn build_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        deferred_tools: &[ToolDefinition],
        attribution: &crate::provider::Attribution,
    ) -> serde_json::Value {
        let origin = self.encrypted_reasoning.then(|| self.origin()).flatten();
        let mut body = build_request_body_for(
            self.deferred_tool_delivery(),
            deferred_tools,
            &self.model,
            system_prompt,
            messages,
            tools,
            self.wire_effort().as_deref(),
            self.max_output_tokens,
            true,
            origin.as_deref(),
        );
        if self.reasoning_summary {
            request_reasoning_summary(&mut body);
        }
        if self.encrypted_reasoning {
            include_encrypted_reasoning(&mut body);
        }
        // The conversation's id, as `chatgpt-subscription` sends it, so one conversation's
        // requests reach the cache holding its prefix.
        if self.prompt_cache_key
            && let Some(session_id) = attribution.session_id
            && let Some(fields) = body.as_object_mut()
        {
            fields.insert(
                "prompt_cache_key".into(),
                serde_json::Value::String(session_id.to_string()),
            );
        }
        body
    }
}

#[async_trait]
impl crate::oauth::RefreshesCredential for OpenAiResponsesProvider {}

#[async_trait::async_trait]
impl ResponsesBackend for OpenAiResponsesProvider {
    fn client(&self) -> &reqwest::Client {
        &self.client
    }

    fn endpoint(&self) -> String {
        self.responses_url()
    }

    fn request_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        deferred_tools: &[ToolDefinition],
        attribution: &crate::provider::Attribution,
    ) -> serde_json::Value {
        self.build_body(system_prompt, messages, tools, deferred_tools, attribution)
    }

    fn max_request_bytes(&self) -> Option<usize> {
        self.max_request_bytes
    }

    async fn authenticated_request(
        &self,
        request: reqwest::RequestBuilder,
        _attribution: &crate::provider::Attribution,
    ) -> Result<reqwest::RequestBuilder> {
        Ok(request
            .header("Authorization", crate::text::bearer(&self.api_key))
            .header("Accept", "text/event-stream"))
    }
}

#[async_trait]
impl Provider for OpenAiResponsesProvider {
    fn deferred_tool_delivery(&self) -> crate::provider::DeferredToolDelivery {
        match self.deferred_tools {
            crate::config::DeferredTools::Native => crate::provider::DeferredToolDelivery::Searched,
            crate::config::DeferredTools::Append => crate::provider::DeferredToolDelivery::Appended,
        }
    }

    async fn complete(
        &self,
        request: CompletionRequest<'_>,
        cancellation: CancellationToken,
    ) -> Result<crate::provider::Completion> {
        super::responses_wire::complete(self, request, cancellation).await
    }

    async fn stream(
        &self,
        request: CompletionRequest<'_>,
        event_sender: mpsc::Sender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        super::responses_wire::stream(self, request, event_sender, cancellation).await
    }

    fn resolved_effort(&self) -> Option<String> {
        self.wire_effort()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(effort: Option<&str>, base_url: Option<&str>) -> OpenAiResponsesProvider {
        {
            let api_key: String = "test-key".to_string();
            OpenAiResponsesProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiResponses,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-6-astra".to_string(),
                )
                .base_url(base_url.map(str::to_string))
                .effort(effort.map(str::to_string))
                .max_output_tokens(None),
            )
        }
        .expect("build test provider")
    }

    /// A stream that never started is retryable here too.
    ///
    /// The sibling of the pair in `provider::anthropic::messages`, and here for the same reason:
    /// this backend has its own `.send()` site, so it is its own wiring into
    /// [`crate::error::provider_transport_error`] and its own chance to go back to a bare
    /// `MekaError::Provider` that the agent loop discards.
    #[tokio::test]
    async fn a_stream_that_could_not_start_reports_a_retryable_failure() {
        // Bound and dropped, so the port is refused rather than answered or hung.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
        let port = listener.local_addr().expect("the bound address").port();
        drop(listener);

        let provider = provider(None, Some(&format!("http://127.0.0.1:{port}/v1")));
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
            matches!(error, crate::error::MekaError::RetryableProvider { .. }),
            "a stream that never started must be retryable, got: {error}"
        );
    }

    /// The endpoint is composed the same way Chat Completions composes its own.
    ///
    /// `{base}/responses` has to work verbatim for the published base URL of every server that
    /// serves this protocol, which is why it reuses `normalize_base_url` rather than inventing a
    /// second rule.
    #[test]
    fn the_url_composes_against_every_published_base() {
        for (base, expected) in [
            (None, "https://api.openai.com/v1/responses"),
            (
                Some("https://api.openai.com/v1"),
                "https://api.openai.com/v1/responses",
            ),
            (
                Some("http://127.0.0.1:11434/v1"),
                "http://127.0.0.1:11434/v1/responses",
            ),
            (
                Some("https://openrouter.ai/api/v1"),
                "https://openrouter.ai/api/v1/responses",
            ),
            // A trailing slash must not double up.
            (
                Some("http://127.0.0.1:11434/v1/"),
                "http://127.0.0.1:11434/v1/responses",
            ),
        ] {
            assert_eq!(provider(None, base).responses_url(), expected, "{base:?}");
        }
    }

    /// OpenAI's own fields go out by default, as `chatgpt-subscription` sends them: the sealed
    /// reasoning asked for, its summary, and the conversation's cache key.
    #[test]
    fn openai_s_fields_go_out_by_default() {
        let session = uuid::Uuid::new_v4();
        let attribution = crate::provider::Attribution {
            session_id: Some(session),
            ..Default::default()
        };
        let body =
            provider(None, None).build_body("s", &[Message::user("hi")], &[], &[], &attribution);
        assert_eq!(body["include"][0], "reasoning.encrypted_content", "{body}");
        assert_eq!(body["reasoning"]["summary"], "auto", "{body}");
        assert_eq!(body["prompt_cache_key"], session.to_string(), "{body}");
        // And the protocol-level fields every implementation understands are still there.
        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], true);
        assert_eq!(body["instructions"], "s");
    }

    /// Each field has its own switch, for the endpoint or the model that refuses it, and a switch
    /// turned off leaves its field out and the others as they were.
    #[test]
    fn each_of_openai_s_fields_has_its_own_switch() {
        let attribution = crate::provider::Attribution {
            session_id: Some(uuid::Uuid::new_v4()),
            ..Default::default()
        };
        let build = |encrypted_reasoning, reasoning_summary, prompt_cache_key| {
            let mut provider = provider(None, None);
            provider.encrypted_reasoning = encrypted_reasoning;
            provider.reasoning_summary = reasoning_summary;
            provider.prompt_cache_key = prompt_cache_key;
            provider.build_body("", &[Message::user("hi")], &[], &[], &attribution)
        };
        let body = build(false, true, true);
        assert!(body.get("include").is_none(), "{body}");
        assert_eq!(body["reasoning"]["summary"], "auto", "{body}");
        let body = build(true, false, true);
        assert!(body.get("reasoning").is_none(), "{body}");
        assert_eq!(body["include"][0], "reasoning.encrypted_content", "{body}");
        let body = build(true, true, false);
        assert!(body.get("prompt_cache_key").is_none(), "{body}");
    }

    /// Sealed reasoning goes back only to the endpoint that sealed it, and only while the profile
    /// asks for reasoning to be carried: a session resumed here after a `chatgpt-subscription`
    /// turn holds ChatGPT's blobs, which this endpoint cannot open.
    #[test]
    fn sealed_reasoning_goes_back_only_to_the_endpoint_that_sealed_it() {
        let history = |origin: &str| {
            vec![
                Message::user("hi"),
                Message {
                    role: crate::conversation::Role::Assistant,
                    content: vec![
                        crate::conversation::ContentBlock::Thinking {
                            thinking: "summary".to_string(),
                            opaque: Some(crate::conversation::OpaqueReasoning::Sealed {
                                encrypted_content: "SEALED".to_string(),
                                id: Some("rs_1".to_string()),
                                origin: Some(origin.to_string()),
                            }),
                        },
                        crate::conversation::ContentBlock::Text {
                            text: "answer".to_string(),
                        },
                    ],
                },
                Message::user("go on"),
            ]
        };
        let replayed = |body: &serde_json::Value| {
            body["input"]
                .as_array()
                .expect("input")
                .iter()
                .any(|item| item["type"] == "reasoning")
        };
        let attribution = crate::provider::Attribution::default();
        let own = provider(None, None).build_body(
            "s",
            &history("https://api.openai.com"),
            &[],
            &[],
            &attribution,
        );
        assert!(replayed(&own), "{own}");
        let foreign = provider(None, None).build_body(
            "s",
            &history("https://chatgpt.com"),
            &[],
            &[],
            &attribution,
        );
        assert!(!replayed(&foreign), "{foreign}");
        assert!(
            serde_json::to_string(&foreign)
                .expect("serialize")
                .contains("answer"),
            "the turn the reasoning belonged to is still there"
        );
        let mut off = provider(None, None);
        off.encrypted_reasoning = false;
        let off = off.build_body(
            "s",
            &history("https://api.openai.com"),
            &[],
            &[],
            &attribution,
        );
        assert!(!replayed(&off), "{off}");
    }

    /// Effort is sent only when the profile asks, and with the summary off nothing pulls a
    /// `reasoning` block in.
    #[test]
    fn an_unconfigured_effort_omits_the_effort() {
        let mut provider = provider(None, None);
        provider.reasoning_summary = false;
        let body = provider.build_body(
            "",
            &[Message::user("hi")],
            &[],
            &[],
            &crate::provider::Attribution::default(),
        );
        assert!(body.get("reasoning").is_none(), "{body}");
        // An empty system prompt sends no `instructions` rather than an empty one.
        assert!(body.get("instructions").is_none(), "{body}");
    }
}
