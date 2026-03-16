use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::{AgshError, Result};
use crate::session::TokenStore;

use super::{
    AuthCredential, ContentBlock, DEFAULT_CLAUDE_CLIENT_ID, Message, Provider, Role, StopReason,
    StreamEvent, ToolDefinition,
};

fn now_epoch_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

pub struct ClaudeProvider {
    client: reqwest::Client,
    credential: tokio::sync::RwLock<AuthCredential>,
    base_url: String,
    model: String,
    client_id: String,
    oauth_token_url: String,
    token_store: Option<Arc<TokenStore>>,
}

impl ClaudeProvider {
    pub fn new(
        credential: AuthCredential,
        model: String,
        base_url: Option<String>,
        client_id: Option<String>,
        oauth_token_url: Option<String>,
        token_store: Option<Arc<TokenStore>>,
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
        }
    }

    async fn ensure_valid_credential(&self) -> Result<(&'static str, String)> {
        {
            let credential = self.credential.read().await;
            match &*credential {
                AuthCredential::ApiKey(key) => {
                    return Ok(("x-api-key", key.clone()));
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

        // Double-check after acquiring write lock
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
                AgshError::Provider(format!("OAuth token refresh request failed: {}", error))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
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
        })
    }

    pub(super) fn build_request_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> serde_json::Value {
        let claude_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|message| {
                let role = match message.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                };

                let content: Vec<serde_json::Value> = message
                    .content
                    .iter()
                    .map(|block| match block {
                        ContentBlock::Text { text } => {
                            serde_json::json!({
                                "type": "text",
                                "text": text,
                            })
                        }
                        ContentBlock::ToolUse { id, name, input } => {
                            serde_json::json!({
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": input,
                            })
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            serde_json::json!({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": content,
                                "is_error": is_error,
                            })
                        }
                    })
                    .collect();

                serde_json::json!({
                    "role": role,
                    "content": content,
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": claude_messages,
            "max_tokens": 8192,
            "stream": stream,
        });

        if !system_prompt.is_empty() {
            body["system"] = serde_json::json!(system_prompt);
        }

        if !tools.is_empty() {
            let claude_tools: Vec<serde_json::Value> = tools
                .iter()
                .map(|tool| {
                    serde_json::json!({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.parameters,
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(claude_tools);
        }

        body
    }

    pub(super) fn parse_non_streaming_response(
        &self,
        response: &serde_json::Value,
    ) -> Result<(Message, StopReason)> {
        let stop_reason_str = response
            .get("stop_reason")
            .and_then(|reason| reason.as_str())
            .unwrap_or("end_turn");

        let stop_reason = parse_claude_stop_reason(stop_reason_str);

        let content_array = response
            .get("content")
            .and_then(|content| content.as_array())
            .ok_or_else(|| AgshError::Provider("no content array in response".to_string()))?;

        let mut content_blocks = Vec::new();

        for block in content_array {
            let block_type = block
                .get("type")
                .and_then(|block_type| block_type.as_str())
                .unwrap_or("");

            match block_type {
                "text" => {
                    if let Some(text) = block.get("text").and_then(|text| text.as_str()) {
                        content_blocks.push(ContentBlock::Text {
                            text: text.to_string(),
                        });
                    }
                }
                "tool_use" => {
                    let id = block
                        .get("id")
                        .and_then(|id| id.as_str())
                        .ok_or_else(|| {
                            AgshError::Provider("tool_use block missing 'id' field".to_string())
                        })?
                        .to_string();
                    let name = block
                        .get("name")
                        .and_then(|name| name.as_str())
                        .ok_or_else(|| {
                            AgshError::Provider("tool_use block missing 'name' field".to_string())
                        })?
                        .to_string();
                    let input = block.get("input").cloned().unwrap_or_else(|| {
                        tracing::warn!("missing 'input' in tool_use block");
                        serde_json::json!({})
                    });

                    content_blocks.push(ContentBlock::ToolUse { id, name, input });
                }
                _ => {
                    tracing::warn!("unknown Claude content block type: {}", block_type);
                }
            }
        }

        Ok((
            Message {
                role: Role::Assistant,
                content: content_blocks,
            },
            stop_reason,
        ))
    }
}

#[async_trait]
impl Provider for ClaudeProvider {
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<(Message, StopReason)> {
        let body = self.build_request_body(system_prompt, messages, tools, false);
        let (auth_header_name, auth_header_value) = self.ensure_valid_credential().await?;

        let mut request = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header(auth_header_name, &auth_header_value)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");

        if auth_header_name == "Authorization" {
            request = request.header("anthropic-beta", "oauth-2025-04-20,claude-code-20250219");
        }

        let response = request
            .json(&body)
            .send()
            .await
            .map_err(|error| AgshError::Provider(format!("HTTP request failed: {}", error)))?;

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

        self.parse_non_streaming_response(&response_json)
    }

    async fn stream(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        event_sender: mpsc::UnboundedSender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        use eventsource_stream::Eventsource;
        use futures::StreamExt;

        let body = self.build_request_body(system_prompt, messages, tools, true);
        let (auth_header_name, auth_header_value) = self.ensure_valid_credential().await?;

        let mut request = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header(auth_header_name, &auth_header_value)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");

        if auth_header_name == "Authorization" {
            request = request.header("anthropic-beta", "oauth-2025-04-20,claude-code-20250219");
        }

        let response = request
            .json(&body)
            .send()
            .await
            .map_err(|error| AgshError::Provider(format!("HTTP request failed: {}", error)))?;

        let status = response.status();
        if !status.is_success() {
            let response_text = response.text().await.unwrap_or_default();
            return Err(AgshError::Provider(format!(
                "API returned status {}: {}",
                status, response_text
            )));
        }

        let mut event_stream = response.bytes_stream().eventsource();

        let mut current_tool_input = String::new();
        let mut in_tool_use = false;

        loop {
            tokio::select! {
                _ = cancellation.cancelled() => {
                    return Err(AgshError::Interrupted);
                }
                event = event_stream.next() => {
                    let Some(event) = event else {
                        break;
                    };

                    match event {
                        Ok(event) => {
                            let data: serde_json::Value = match serde_json::from_str(&event.data) {
                                Ok(data) => data,
                                Err(error) => {
                                    tracing::warn!("failed to parse SSE data: {}", error);
                                    continue;
                                }
                            };

                            match event.event.as_str() {
                                "content_block_start" => {
                                    let Some(content_block) = data.get("content_block") else {
                                        continue;
                                    };
                                    let block_type = content_block
                                        .get("type")
                                        .and_then(|block_type| block_type.as_str())
                                        .unwrap_or("");

                                    if block_type == "tool_use" {
                                        let id = content_block
                                            .get("id")
                                            .and_then(|id| id.as_str())
                                            .ok_or_else(|| {
                                                AgshError::Provider(
                                                    "tool_use block missing 'id' field".to_string(),
                                                )
                                            })?
                                            .to_string();
                                        let name = content_block
                                            .get("name")
                                            .and_then(|name| name.as_str())
                                            .ok_or_else(|| {
                                                AgshError::Provider(
                                                    "tool_use block missing 'name' field"
                                                        .to_string(),
                                                )
                                            })?
                                            .to_string();

                                        current_tool_input.clear();
                                        in_tool_use = true;
                                        if event_sender.send(StreamEvent::ToolUseStart {
                                            id,
                                            name,
                                        }).is_err() {
                                            tracing::trace!("stream event receiver dropped");
                                            break;
                                        }
                                    }
                                }
                                "content_block_delta" => {
                                    let Some(delta) = data.get("delta") else {
                                        continue;
                                    };
                                    let delta_type = delta
                                        .get("type")
                                        .and_then(|delta_type| delta_type.as_str())
                                        .unwrap_or("");

                                    match delta_type {
                                        "text_delta" => {
                                            if let Some(text) = delta.get("text").and_then(|text| text.as_str())
                                                && !text.is_empty()
                                                    && event_sender.send(
                                                        StreamEvent::TextDelta(text.to_string()),
                                                    ).is_err() {
                                                        tracing::trace!("stream event receiver dropped");
                                                        break;
                                                    }
                                        }
                                        "input_json_delta" => {
                                            if let Some(partial_json) =
                                                delta.get("partial_json").and_then(|partial_json| partial_json.as_str())
                                            {
                                                current_tool_input.push_str(partial_json);
                                                if event_sender.send(
                                                    StreamEvent::ToolInputDelta(
                                                        partial_json.to_string(),
                                                    ),
                                                ).is_err() {
                                                    tracing::trace!("stream event receiver dropped");
                                                    break;
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                                "content_block_stop" => {
                                    if in_tool_use {
                                        let input = if current_tool_input.is_empty() {
                                            serde_json::json!({})
                                        } else {
                                            match serde_json::from_str(&current_tool_input) {
                                                Ok(value) => value,
                                                Err(error) => {
                                                    tracing::warn!("failed to parse tool input JSON: {}", error);
                                                    serde_json::json!({})
                                                }
                                            }
                                        };
                                        if event_sender
                                            .send(StreamEvent::ToolUseEnd { input })
                                            .is_err()
                                        {
                                            tracing::trace!("stream event receiver dropped");
                                            break;
                                        }
                                        current_tool_input.clear();
                                        in_tool_use = false;
                                    }
                                }
                                "message_delta" => {
                                    let Some(delta) = data.get("delta") else {
                                        continue;
                                    };
                                    if let Some(stop_reason_str) =
                                        delta.get("stop_reason").and_then(|reason| reason.as_str())
                                    {
                                        let stop_reason = parse_claude_stop_reason(stop_reason_str);
                                        if event_sender
                                            .send(StreamEvent::MessageEnd { stop_reason })
                                            .is_err()
                                        {
                                            tracing::trace!("stream event receiver dropped");
                                            break;
                                        }
                                    }
                                }
                                "message_stop" => {
                                    break;
                                }
                                "message_start" | "ping" => {}
                                other => {
                                    tracing::debug!("unknown Claude SSE event: {}", other);
                                }
                            }
                        }
                        Err(error) => {
                            if event_sender.send(StreamEvent::Error(error.to_string())).is_err() {
                                tracing::trace!("stream event receiver dropped");
                            }
                            return Err(AgshError::StreamError(error.to_string()));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "claude"
    }
}

fn parse_claude_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        other => StopReason::Unknown(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claude_request_body_simple() {
        let provider = ClaudeProvider::new(
            AuthCredential::ApiKey("test-key".to_string()),
            "claude-sonnet-4-20250514".to_string(),
            None,
            None,
            None,
            None,
        );

        let messages = vec![Message::user("hello")];
        let body = provider.build_request_body("system prompt", &messages, &[], false);

        assert_eq!(body["model"], "claude-sonnet-4-20250514");
        assert_eq!(body["stream"], false);
        assert_eq!(body["system"], "system prompt");

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
    }

    #[test]
    fn test_claude_request_body_with_tools() {
        let provider = ClaudeProvider::new(
            AuthCredential::ApiKey("test-key".to_string()),
            "claude-sonnet-4-20250514".to_string(),
            None,
            None,
            None,
            None,
        );

        let tools = vec![ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        }];

        let body = provider.build_request_body("", &[], &tools, false);
        let claude_tools = body["tools"].as_array().expect("tools should be array");
        assert_eq!(claude_tools.len(), 1);
        assert_eq!(claude_tools[0]["name"], "read_file");
        assert_eq!(claude_tools[0]["description"], "Read a file");
        assert!(claude_tools[0].get("input_schema").is_some());
    }

    #[test]
    fn test_claude_request_body_with_tool_calls() {
        let provider = ClaudeProvider::new(
            AuthCredential::ApiKey("test-key".to_string()),
            "claude-sonnet-4-20250514".to_string(),
            None,
            None,
            None,
            None,
        );

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
                    content: "file contents here".to_string(),
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
    }

    #[test]
    fn test_claude_parse_non_streaming_text() {
        let provider = ClaudeProvider::new(
            AuthCredential::ApiKey("test-key".to_string()),
            "claude-sonnet-4-20250514".to_string(),
            None,
            None,
            None,
            None,
        );

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

        let (message, stop_reason) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");

        assert_eq!(message.text_content(), "Hello there!");
        assert_eq!(stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn test_claude_parse_non_streaming_tool_use() {
        let provider = ClaudeProvider::new(
            AuthCredential::ApiKey("test-key".to_string()),
            "claude-sonnet-4-20250514".to_string(),
            None,
            None,
            None,
            None,
        );

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

        let (message, stop_reason) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");

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
    fn test_claude_no_system_prompt_when_empty() {
        let provider = ClaudeProvider::new(
            AuthCredential::ApiKey("test-key".to_string()),
            "claude-sonnet-4-20250514".to_string(),
            None,
            None,
            None,
            None,
        );

        let body = provider.build_request_body("", &[], &[], false);
        assert!(body.get("system").is_none());
    }

    #[test]
    fn test_claude_parse_missing_tool_use_id() {
        let provider = ClaudeProvider::new(
            AuthCredential::ApiKey("test-key".to_string()),
            "claude-sonnet-4-20250514".to_string(),
            None,
            None,
            None,
            None,
        );

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

        let result = provider.parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_claude_parse_missing_tool_use_name() {
        let provider = ClaudeProvider::new(
            AuthCredential::ApiKey("test-key".to_string()),
            "claude-sonnet-4-20250514".to_string(),
            None,
            None,
            None,
            None,
        );

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

        let result = provider.parse_non_streaming_response(&response);
        assert!(result.is_err());
    }
}
