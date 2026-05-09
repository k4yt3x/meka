//! LLM provider abstraction. Defines the [`Provider`] trait, the shared
//! message/content/tool types, and the [`ProviderBuilder`] that returns
//! a concrete Claude or OpenAI-compatible implementation.

mod claude;
pub(crate) mod openai;

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::{AgshError, Result};
use crate::session::TokenStore;

pub use claude::{ClaudeApiProvider, ClaudeOAuthProvider};
pub use openai::{OpenAiCodexProvider, OpenAiProvider};

pub(crate) const DEFAULT_CLAUDE_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";

/// Codex's hardcoded OpenAI OAuth client ID. Mirrors the value used by the
/// first-party CLI at `temp/codex/codex-rs/login/src/auth/manager.rs:869`.
pub(crate) const DEFAULT_OPENAI_CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

pub const SUPPORTED_PROVIDERS: &[&str] =
    &["openai-api", "openai-codex", "claude-api", "claude-oauth"];

#[derive(Debug, Clone)]
pub enum AuthCredential {
    ApiKey(String),
    OAuthToken {
        access_token: String,
        refresh_token: Option<String>,
        expires_at: Option<i64>,
        /// Provider-flavoured identity carried alongside the bearer token.
        /// Currently only `openai-codex` populates this — the
        /// `chatgpt_account_id` extracted from the id_token JWT, sent on
        /// every request as `ChatGPT-Account-ID`. Claude OAuth leaves it
        /// `None`.
        account_id: Option<String>,
    },
}

impl AuthCredential {
    pub fn auth_header(&self) -> (&'static str, String) {
        match self {
            AuthCredential::ApiKey(key) => ("x-api-key", key.clone()),
            AuthCredential::OAuthToken { access_token, .. } => {
                ("Authorization", format!("Bearer {}", access_token))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContent {
    Text { text: String },
    Image { source: ImageSource },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        signature: Option<String>,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(deserialize_with = "deserialize_tool_result_content")]
        content: Vec<ToolResultContent>,
        is_error: bool,
    },
}

/// Deserializes ToolResult content from either a string (legacy format) or
/// an array of ToolResultContent (new format).
fn deserialize_tool_result_content<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<ToolResultContent>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrVec {
        String(String),
        Vec(Vec<ToolResultContent>),
    }

    match StringOrVec::deserialize(deserializer)? {
        StringOrVec::String(text) => Ok(vec![ToolResultContent::Text { text }]),
        StringOrVec::Vec(vec) => Ok(vec),
    }
}

impl ContentBlock {
    /// Extract the text content of a ToolResult (for display/logging).
    pub fn tool_result_text_content(content: &[ToolResultContent]) -> String {
        content
            .iter()
            .map(|block| match block {
                ToolResultContent::Text { text } => text.as_str(),
                ToolResultContent::Image { .. } => "[Image]",
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    #[cfg(test)]
    pub fn tool_uses(&self) -> Vec<&ContentBlock> {
        self.content
            .iter()
            .filter(|block| matches!(block, ContentBlock::ToolUse { .. }))
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    /// Human-readable title for the tool, optionally set by MCP servers.
    /// Providers may render this in UIs instead of the machine name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// MCP `tool.annotations`: hints such as `readOnlyHint`,
    /// `destructiveHint`, `openWorldHint`. Passed through verbatim as JSON;
    /// providers that don't recognise the field ignore it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<serde_json::Value>,
    /// MCP `tool.meta` payload, forwarded verbatim so permission heuristics
    /// and audit logs can access it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<serde_json::Value>,
}

#[cfg(test)]
impl ToolDefinition {
    /// Test-only convenience constructor. Production code builds
    /// `ToolDefinition` as a struct literal and explicitly sets the
    /// MCP-specific `title`/`annotations`/`meta` fields; this helper just
    /// keeps test fixtures terse.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            title: None,
            annotations: None,
            meta: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta(String),
    ThinkingDelta(String),
    ThinkingComplete {
        signature: Option<String>,
    },
    ToolUseStart {
        id: String,
        name: String,
    },
    ToolInputDelta(String),
    ToolUseEnd {
        input: serde_json::Value,
    },
    /// Emitted in lieu of `ToolUseEnd` when the accumulated tool-call
    /// arguments fail to parse as JSON. The agent layer must not execute
    /// the tool; it should surface the parse error back to the model as
    /// a `ToolResult { is_error: true }` instead.
    ToolCallRejected {
        id: String,
        name: String,
        reason: String,
    },
    MessageEnd {
        stop_reason: StopReason,
    },
    Usage(TokenUsage),
    Error(String),
}

/// Sentinel key inserted into `ToolUse::input` when the upstream tool-call
/// arguments failed to parse. `resolve_and_execute_tool` checks for this
/// and short-circuits to an error result instead of invoking the tool
/// with a potentially surprising default-filled object.
pub(crate) const INVALID_TOOL_ARGS_MARKER: &str = "_agsh_invalid_arguments";

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Tokens billed at the cache-write tier (content newly cached this
    /// turn). Anthropic-only; OpenAI providers leave this at 0.
    pub cache_creation_input_tokens: u64,
    /// Tokens served from the prompt cache (cache-read tier). Anthropic
    /// returns this in `usage.cache_read_input_tokens`; OpenAI providers
    /// leave it at 0 today.
    pub cache_read_input_tokens: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Unknown(String),
}

#[async_trait]
pub trait Provider: Send + Sync {
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<(Message, StopReason, TokenUsage)>;

    async fn stream(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        event_sender: mpsc::UnboundedSender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()>;

    #[allow(dead_code)]
    fn name(&self) -> &str;

    /// Override thinking for the next API call. `Some(false)` disables,
    /// `Some(true)` enables, `None` restores the default.
    fn set_thinking_override(&self, _enabled: Option<bool>) {}
}

struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

fn finalize_tool_call_accumulators(
    accumulators: &mut std::collections::HashMap<i64, ToolCallAccumulator>,
    event_sender: &mpsc::UnboundedSender<StreamEvent>,
) -> bool {
    let has_tools = !accumulators.is_empty();
    let mut indices: Vec<i64> = accumulators.keys().copied().collect();
    indices.sort();
    for index in indices {
        if let Some(accumulator) = accumulators.remove(&index) {
            if event_sender
                .send(StreamEvent::ToolUseStart {
                    id: accumulator.id.clone(),
                    name: accumulator.name.clone(),
                })
                .is_err()
            {
                tracing::trace!("stream event receiver dropped");
                return has_tools;
            }
            match serde_json::from_str::<serde_json::Value>(&accumulator.arguments) {
                Ok(value) => {
                    if event_sender
                        .send(StreamEvent::ToolUseEnd { input: value })
                        .is_err()
                    {
                        tracing::trace!("stream event receiver dropped");
                        return has_tools;
                    }
                }
                Err(error) => {
                    tracing::warn!(
                        tool = %accumulator.name,
                        "rejecting tool call with unparseable JSON arguments: {}",
                        error
                    );
                    if event_sender
                        .send(StreamEvent::ToolCallRejected {
                            id: accumulator.id.clone(),
                            name: accumulator.name.clone(),
                            reason: format!("invalid JSON arguments: {}", error),
                        })
                        .is_err()
                    {
                        tracing::trace!("stream event receiver dropped");
                        return has_tools;
                    }
                }
            }
        }
    }
    has_tools
}

/// Constructs a concrete [`Provider`] (Claude API, Claude OAuth, or
/// OpenAI-compatible) from a bag of provider-specific settings. Each
/// setter documents which provider(s) consume it; unused settings are
/// silently ignored by providers that don't need them. The only
/// required inputs are the provider name, the credential, and the
/// model — everything else has a sensible default.
pub struct ProviderBuilder {
    provider_name: String,
    credential: AuthCredential,
    model: String,
    base_url: Option<String>,
    client_id: Option<String>,
    oauth_token_url: Option<String>,
    token_store: Option<Arc<TokenStore>>,
    thinking_enabled: bool,
    thinking_budget_tokens: u64,
    reasoning_effort: Option<String>,
    device_id: String,
    effort: String,
    redact_thinking: bool,
    session_stats: Option<Arc<crate::stats::SessionStats>>,
}

impl ProviderBuilder {
    pub fn new(
        provider_name: impl Into<String>,
        credential: AuthCredential,
        model: impl Into<String>,
    ) -> Self {
        Self {
            provider_name: provider_name.into(),
            credential,
            model: model.into(),
            base_url: None,
            client_id: None,
            oauth_token_url: None,
            token_store: None,
            thinking_enabled: false,
            thinking_budget_tokens: 0,
            reasoning_effort: None,
            device_id: String::new(),
            effort: "high".to_string(),
            redact_thinking: false,
            session_stats: None,
        }
    }

    /// Override the HTTP endpoint. Applies to every provider variant;
    /// defaults to the Claude or OpenAI production URL.
    pub fn base_url(mut self, value: Option<String>) -> Self {
        self.base_url = value;
        self
    }

    /// OAuth client ID. Only consumed by `claude-oauth`.
    pub fn client_id(mut self, value: Option<String>) -> Self {
        self.client_id = value;
        self
    }

    /// OAuth token endpoint. Only consumed by `claude-oauth`.
    pub fn oauth_token_url(mut self, value: Option<String>) -> Self {
        self.oauth_token_url = value;
        self
    }

    /// Sink for refreshed OAuth tokens. Only consumed by `claude-oauth`;
    /// when `None`, refreshed tokens are held in memory only.
    pub fn token_store(mut self, value: Option<Arc<TokenStore>>) -> Self {
        self.token_store = value;
        self
    }

    /// Claude-only: turn on extended thinking with the given budget cap.
    /// Ignored by `openai-api`.
    pub fn thinking(mut self, enabled: bool, budget_tokens: u64) -> Self {
        self.thinking_enabled = enabled;
        self.thinking_budget_tokens = budget_tokens;
        self
    }

    /// OpenAI-only: maps to `reasoning.effort` for reasoning models.
    pub fn reasoning_effort(mut self, value: Option<String>) -> Self {
        self.reasoning_effort = value;
        self
    }

    /// Stable device identity embedded in `metadata.user_id`. Only
    /// consumed by `claude-oauth`.
    pub fn device_id(mut self, value: String) -> Self {
        self.device_id = value;
        self
    }

    /// Claude Code `output_config.effort` (`low` / `medium` / `high`).
    /// Only consumed by `claude-oauth`.
    pub fn effort(mut self, value: String) -> Self {
        self.effort = value;
        self
    }

    /// Request `redacted_thinking` blocks. Only consumed by `claude-oauth`.
    pub fn redact_thinking(mut self, value: bool) -> Self {
        self.redact_thinking = value;
        self
    }

    /// Per-session counters incremented when image-redaction events fire.
    /// Currently consumed only by `claude-oauth` and `claude-api`.
    pub fn session_stats(mut self, value: Option<Arc<crate::stats::SessionStats>>) -> Self {
        self.session_stats = value;
        self
    }

    pub fn build(self) -> Result<Arc<dyn Provider>> {
        match self.provider_name.as_str() {
            "openai-api" => {
                let api_key = match self.credential {
                    AuthCredential::ApiKey(key) => key,
                    AuthCredential::OAuthToken { access_token, .. } => access_token,
                };
                Ok(Arc::new(OpenAiProvider::new(
                    api_key,
                    self.model,
                    self.base_url,
                    self.reasoning_effort,
                )))
            }
            "claude-api" => {
                let api_key = match self.credential {
                    AuthCredential::ApiKey(key) => key,
                    AuthCredential::OAuthToken { .. } => {
                        return Err(AgshError::Config(
                            "provider 'claude-api' requires an API key, not an OAuth token. \
                             Use 'claude-oauth' for Claude Code OAuth."
                                .to_string(),
                        ));
                    }
                };
                Ok(Arc::new(ClaudeApiProvider::new(
                    api_key,
                    self.model,
                    self.base_url,
                    self.thinking_enabled,
                    self.thinking_budget_tokens,
                    self.session_stats,
                )))
            }
            "claude-oauth" => {
                if matches!(self.credential, AuthCredential::ApiKey(_)) {
                    return Err(AgshError::Config(
                        "provider 'claude-oauth' requires an OAuth token, not an API key. \
                         Use 'claude-api' for direct API access."
                            .to_string(),
                    ));
                }
                Ok(Arc::new(ClaudeOAuthProvider::new(
                    self.credential,
                    self.model,
                    self.base_url,
                    self.client_id,
                    self.oauth_token_url,
                    self.token_store,
                    self.thinking_enabled,
                    self.thinking_budget_tokens,
                    self.device_id,
                    self.effort,
                    self.redact_thinking,
                    self.session_stats,
                )))
            }
            "openai-codex" => {
                if matches!(self.credential, AuthCredential::ApiKey(_)) {
                    return Err(AgshError::Config(
                        "provider 'openai-codex' requires an OAuth token, not an API key. \
                         Use 'openai-api' for direct API access."
                            .to_string(),
                    ));
                }
                Ok(Arc::new(OpenAiCodexProvider::new(
                    self.credential,
                    self.model,
                    self.base_url,
                    self.client_id,
                    self.oauth_token_url,
                    self.token_store,
                    self.reasoning_effort,
                )?))
            }
            other => Err(AgshError::Config(format!(
                "unknown provider: '{}'. Supported providers: {}",
                other,
                SUPPORTED_PROVIDERS.join(", ")
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the "silent `{}` fallback" bug: a tool call with
    /// unparseable JSON arguments must be rejected via
    /// [`StreamEvent::ToolCallRejected`] rather than replayed with an
    /// empty input object (which would run the tool on whatever defaults
    /// it happens to tolerate).
    #[test]
    fn test_finalize_tool_call_accumulators_rejects_invalid_json() {
        let mut accumulators = std::collections::HashMap::new();
        accumulators.insert(
            0,
            ToolCallAccumulator {
                id: "call-1".to_string(),
                name: "read_file".to_string(),
                arguments: "{not json".to_string(),
            },
        );

        let (sender, mut receiver) = mpsc::unbounded_channel::<StreamEvent>();
        let has_tools = finalize_tool_call_accumulators(&mut accumulators, &sender);
        assert!(has_tools, "accumulator was non-empty");

        let first = receiver.try_recv().expect("ToolUseStart emitted first");
        assert!(
            matches!(first, StreamEvent::ToolUseStart { .. }),
            "expected ToolUseStart, got {:?}",
            first
        );

        let second = receiver.try_recv().expect("follow-up event");
        match second {
            StreamEvent::ToolCallRejected { id, name, reason } => {
                assert_eq!(id, "call-1");
                assert_eq!(name, "read_file");
                assert!(reason.starts_with("invalid JSON arguments"));
            }
            other => panic!("expected ToolCallRejected, got {:?}", other),
        }

        assert!(
            receiver.try_recv().is_err(),
            "no further events after rejection"
        );
    }

    #[test]
    fn test_finalize_tool_call_accumulators_passes_valid_json() {
        let mut accumulators = std::collections::HashMap::new();
        accumulators.insert(
            0,
            ToolCallAccumulator {
                id: "call-2".to_string(),
                name: "read_file".to_string(),
                arguments: r#"{"path": "/tmp/x"}"#.to_string(),
            },
        );

        let (sender, mut receiver) = mpsc::unbounded_channel::<StreamEvent>();
        finalize_tool_call_accumulators(&mut accumulators, &sender);

        let first = receiver.try_recv().expect("ToolUseStart");
        assert!(matches!(first, StreamEvent::ToolUseStart { .. }));

        match receiver.try_recv().expect("ToolUseEnd") {
            StreamEvent::ToolUseEnd { input } => {
                assert_eq!(input["path"], "/tmp/x");
            }
            other => panic!("expected ToolUseEnd, got {:?}", other),
        }
    }

    #[test]
    fn test_message_user() {
        let message = Message::user("hello");
        assert_eq!(message.role, Role::User);
        assert_eq!(message.text_content(), "hello");
    }

    #[test]
    fn test_message_assistant_text() {
        let message = Message::assistant_text("response");
        assert_eq!(message.role, Role::Assistant);
        assert_eq!(message.text_content(), "response");
    }

    #[test]
    fn test_message_tool_uses() {
        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "I'll read that file.".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test.txt"}),
                },
            ],
        };
        assert_eq!(message.tool_uses().len(), 1);
    }

    #[test]
    fn test_content_block_serialization() {
        let block = ContentBlock::Text {
            text: "hello".to_string(),
        };
        let json = serde_json::to_string(&block).expect("should serialize");
        let deserialized: ContentBlock = serde_json::from_str(&json).expect("should deserialize");

        if let ContentBlock::Text { text } = deserialized {
            assert_eq!(text, "hello");
        } else {
            panic!("expected Text block");
        }
    }

    #[test]
    fn test_message_serialization_roundtrip() {
        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "Let me read that.".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test"}),
                },
            ],
        };

        let json = serde_json::to_string(&message).expect("should serialize");
        let deserialized: Message = serde_json::from_str(&json).expect("should deserialize");

        assert_eq!(deserialized.role, Role::Assistant);
        assert_eq!(deserialized.content.len(), 2);
        assert_eq!(deserialized.text_content(), "Let me read that.");
    }

    #[test]
    fn test_create_provider_openai_api() {
        let result = ProviderBuilder::new(
            "openai-api",
            AuthCredential::ApiKey("key".to_string()),
            "gpt-4o",
        )
        .device_id("a".repeat(64))
        .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_provider_claude_api() {
        let result = ProviderBuilder::new(
            "claude-api",
            AuthCredential::ApiKey("key".to_string()),
            "claude-sonnet-4-20250514",
        )
        .thinking(false, 10000)
        .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_provider_claude_oauth() {
        let result = ProviderBuilder::new(
            "claude-oauth",
            AuthCredential::OAuthToken {
                access_token: "sk-ant-oat01-test".to_string(),
                refresh_token: None,
                expires_at: None,
                account_id: None,
            },
            "claude-sonnet-4-20250514",
        )
        .device_id("a".repeat(64))
        .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_provider_claude_api_rejects_oauth_token() {
        let result = ProviderBuilder::new(
            "claude-api",
            AuthCredential::OAuthToken {
                access_token: "sk-ant-oat01-test".to_string(),
                refresh_token: None,
                expires_at: None,
                account_id: None,
            },
            "claude-sonnet-4-20250514",
        )
        .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_create_provider_claude_oauth_rejects_api_key() {
        let result = ProviderBuilder::new(
            "claude-oauth",
            AuthCredential::ApiKey("sk-ant-api03-test".to_string()),
            "claude-sonnet-4-20250514",
        )
        .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_create_provider_openai_codex() {
        let result = ProviderBuilder::new(
            "openai-codex",
            AuthCredential::OAuthToken {
                access_token: "codex-access".to_string(),
                refresh_token: Some("codex-refresh".to_string()),
                expires_at: Some(now_ms_in_far_future()),
                account_id: Some("workspace-1".to_string()),
            },
            "gpt-5",
        )
        .reasoning_effort(Some("high".to_string()))
        .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_provider_openai_codex_rejects_api_key() {
        let result = ProviderBuilder::new(
            "openai-codex",
            AuthCredential::ApiKey("sk-...".to_string()),
            "gpt-5",
        )
        .build();
        assert!(result.is_err());
    }

    fn now_ms_in_far_future() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_millis() as i64 + 86_400_000)
            .unwrap_or(0)
    }

    #[test]
    fn test_create_provider_unknown() {
        let result = ProviderBuilder::new(
            "unknown",
            AuthCredential::ApiKey("key".to_string()),
            "model",
        )
        .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_auth_credential_api_key_header() {
        let credential = AuthCredential::ApiKey("my-key".to_string());
        let (name, value) = credential.auth_header();
        assert_eq!(name, "x-api-key");
        assert_eq!(value, "my-key");
    }

    #[test]
    fn test_auth_credential_oauth_header() {
        let credential = AuthCredential::OAuthToken {
            access_token: "my-token".to_string(),
            refresh_token: None,
            expires_at: None,
            account_id: None,
        };
        let (name, value) = credential.auth_header();
        assert_eq!(name, "Authorization");
        assert_eq!(value, "Bearer my-token");
    }
}
