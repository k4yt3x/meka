mod claude;
mod openai;

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::{AgshError, Result};
use crate::session::TokenStore;

pub use claude::ClaudeProvider;
pub use openai::OpenAiProvider;

pub(crate) const DEFAULT_CLAUDE_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";

#[derive(Debug, Clone)]
pub enum AuthCredential {
    ApiKey(String),
    OAuthToken {
        access_token: String,
        refresh_token: Option<String>,
        expires_at: Option<i64>,
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

    pub fn from_token_string(token: String) -> Self {
        if token.starts_with("sk-ant-oat01-") {
            AuthCredential::OAuthToken {
                access_token: token,
                refresh_token: None,
                expires_at: None,
            }
        } else {
            AuthCredential::ApiKey(token)
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

#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta(String),
    ThinkingDelta(String),
    ThinkingComplete { signature: Option<String> },
    ToolUseStart { id: String, name: String },
    ToolInputDelta(String),
    ToolUseEnd { input: serde_json::Value },
    MessageEnd { stop_reason: StopReason },
    Usage(TokenUsage),
    Error(String),
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
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
            let input = match serde_json::from_str(&accumulator.arguments) {
                Ok(value) => value,
                Err(error) => {
                    tracing::warn!("failed to parse tool arguments: {}", error);
                    serde_json::json!({})
                }
            };
            if event_sender
                .send(StreamEvent::ToolUseEnd { input })
                .is_err()
            {
                tracing::trace!("stream event receiver dropped");
                return has_tools;
            }
        }
    }
    has_tools
}

#[allow(clippy::too_many_arguments)]
pub fn create_provider(
    provider_name: &str,
    credential: AuthCredential,
    model: String,
    base_url: Option<String>,
    client_id: Option<String>,
    oauth_token_url: Option<String>,
    token_store: Option<Arc<TokenStore>>,
    thinking_enabled: bool,
    thinking_budget_tokens: u64,
    reasoning_effort: Option<String>,
) -> Result<Arc<dyn Provider>> {
    match provider_name {
        "openai" => {
            let api_key = match credential {
                AuthCredential::ApiKey(key) => key,
                AuthCredential::OAuthToken { access_token, .. } => access_token,
            };
            Ok(Arc::new(OpenAiProvider::new(
                api_key,
                model,
                base_url,
                reasoning_effort,
            )))
        }
        "claude" => Ok(Arc::new(ClaudeProvider::new(
            credential,
            model,
            base_url,
            client_id,
            oauth_token_url,
            token_store,
            thinking_enabled,
            thinking_budget_tokens,
        ))),
        other => Err(AgshError::Config(format!(
            "unknown provider: '{}'. Supported providers: openai, claude",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_create_provider_openai() {
        let result = create_provider(
            "openai",
            AuthCredential::ApiKey("key".to_string()),
            "gpt-4o".to_string(),
            None,
            None,
            None,
            None,
            false,
            10000,
            None,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_provider_claude() {
        let result = create_provider(
            "claude",
            AuthCredential::ApiKey("key".to_string()),
            "claude-sonnet-4-20250514".to_string(),
            None,
            None,
            None,
            None,
            false,
            10000,
            None,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_provider_unknown() {
        let result = create_provider(
            "unknown",
            AuthCredential::ApiKey("key".to_string()),
            "model".to_string(),
            None,
            None,
            None,
            None,
            false,
            10000,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_auth_credential_from_token_string_api_key() {
        let credential = AuthCredential::from_token_string("sk-ant-api03-test".to_string());
        assert!(matches!(credential, AuthCredential::ApiKey(_)));
    }

    #[test]
    fn test_auth_credential_from_token_string_oauth() {
        let credential = AuthCredential::from_token_string("sk-ant-oat01-test".to_string());
        assert!(matches!(credential, AuthCredential::OAuthToken { .. }));
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
        };
        let (name, value) = credential.auth_header();
        assert_eq!(name, "Authorization");
        assert_eq!(value, "Bearer my-token");
    }
}
