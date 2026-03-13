use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::{AgshError, Result};
use crate::session::TokenStore;

pub(crate) const DEFAULT_CLAUDE_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";

// ---------------------------------------------------------------------------
// Authentication credential
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Unified message model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
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

// ---------------------------------------------------------------------------
// Tool definition (for sending to the LLM)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Streaming events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta(String),
    ToolUseStart { id: String, name: String },
    ToolInputDelta(String),
    ToolUseEnd { input: serde_json::Value },
    MessageEnd { stop_reason: StopReason },
    Error(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Unknown(String),
}

// ---------------------------------------------------------------------------
// Provider trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait Provider: Send + Sync {
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<(Message, StopReason)>;

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
}

// ---------------------------------------------------------------------------
// OpenAI provider
// ---------------------------------------------------------------------------

pub struct OpenAiProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl OpenAiProvider {
    pub fn new(api_key: String, model: String, base_url: Option<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
            model,
        }
    }

    fn build_request_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> serde_json::Value {
        let mut openai_messages = Vec::new();

        if !system_prompt.is_empty() {
            openai_messages.push(serde_json::json!({
                "role": "system",
                "content": system_prompt,
            }));
        }

        for message in messages {
            match message.role {
                Role::User => {
                    let has_tool_results = message
                        .content
                        .iter()
                        .any(|block| matches!(block, ContentBlock::ToolResult { .. }));

                    if has_tool_results {
                        for block in &message.content {
                            if let ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                is_error,
                            } = block
                            {
                                let mut tool_msg = serde_json::json!({
                                    "role": "tool",
                                    "tool_call_id": tool_use_id,
                                    "content": content,
                                });
                                if *is_error {
                                    tool_msg["is_error"] = serde_json::json!(true);
                                }
                                openai_messages.push(tool_msg);
                            }
                        }
                    } else {
                        openai_messages.push(serde_json::json!({
                            "role": "user",
                            "content": message.text_content(),
                        }));
                    }
                }
                Role::Assistant => {
                    let tool_calls: Vec<_> = message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::ToolUse { id, name, input } => Some(serde_json::json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": input.to_string(),
                                }
                            })),
                            _ => None,
                        })
                        .collect();

                    if tool_calls.is_empty() {
                        openai_messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": message.text_content(),
                        }));
                    } else {
                        let text = message.text_content();
                        let mut msg = serde_json::json!({
                            "role": "assistant",
                            "tool_calls": tool_calls,
                        });
                        if !text.is_empty() {
                            msg["content"] = serde_json::json!(text);
                        }
                        openai_messages.push(msg);
                    }
                }
            }
        }

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": openai_messages,
            "stream": stream,
        });

        if !tools.is_empty() {
            let openai_tools: Vec<_> = tools
                .iter()
                .map(|tool| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(openai_tools);
        }

        body
    }

    fn parse_non_streaming_response(
        &self,
        response: &serde_json::Value,
    ) -> Result<(Message, StopReason)> {
        let choice = response
            .get("choices")
            .and_then(|choices| choices.get(0))
            .ok_or_else(|| AgshError::Provider("no choices in response".to_string()))?;

        let finish_reason = choice
            .get("finish_reason")
            .and_then(|reason| reason.as_str())
            .unwrap_or("stop");

        let stop_reason = match finish_reason {
            "stop" => StopReason::EndTurn,
            "tool_calls" => StopReason::ToolUse,
            "length" => StopReason::MaxTokens,
            other => StopReason::Unknown(other.to_string()),
        };

        let assistant_message = &choice["message"];
        let mut content_blocks = Vec::new();

        if let Some(text) = assistant_message.get("content").and_then(|c| c.as_str()) {
            if !text.is_empty() {
                content_blocks.push(ContentBlock::Text {
                    text: text.to_string(),
                });
            }
        }

        if let Some(tool_calls) = assistant_message
            .get("tool_calls")
            .and_then(|tc| tc.as_array())
        {
            for tool_call in tool_calls {
                let id = tool_call["id"].as_str().unwrap_or_default().to_string();
                let name = tool_call["function"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let arguments_str = tool_call["function"]["arguments"].as_str().unwrap_or("{}");
                let input: serde_json::Value =
                    serde_json::from_str(arguments_str).unwrap_or(serde_json::json!({}));

                content_blocks.push(ContentBlock::ToolUse { id, name, input });
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
impl Provider for OpenAiProvider {
    async fn complete(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<(Message, StopReason)> {
        let body = self.build_request_body(system_prompt, messages, tools, false);

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
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
        use futures::StreamExt;
        use reqwest_eventsource::{Event, EventSource};

        let body = self.build_request_body(system_prompt, messages, tools, true);

        let request = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body);

        let mut event_source = EventSource::new(request).map_err(|error| {
            AgshError::Provider(format!("failed to create SSE stream: {}", error))
        })?;

        // Accumulate tool call arguments as they stream in
        let mut tool_call_accumulators: std::collections::HashMap<i64, ToolCallAccumulator> =
            std::collections::HashMap::new();

        loop {
            tokio::select! {
                _ = cancellation.cancelled() => {
                    event_source.close();
                    return Err(AgshError::Interrupted);
                }
                event = event_source.next() => {
                    let Some(event) = event else {
                        break;
                    };

                    match event {
                        Ok(Event::Open) => {}
                        Ok(Event::Message(message)) => {
                            if message.data == "[DONE]" {
                                // Finalize any pending tool calls with paired Start+End
                                let has_tools = !tool_call_accumulators.is_empty();
                                let mut indices: Vec<i64> = tool_call_accumulators.keys().copied().collect();
                                indices.sort();
                                for index in indices {
                                    if let Some(accumulator) = tool_call_accumulators.remove(&index) {
                                        let _ = event_sender.send(StreamEvent::ToolUseStart {
                                            id: accumulator.id.clone(),
                                            name: accumulator.name.clone(),
                                        });
                                        let input = serde_json::from_str(&accumulator.arguments)
                                            .unwrap_or(serde_json::json!({}));
                                        let _ = event_sender.send(StreamEvent::ToolUseEnd { input });
                                    }
                                }
                                let stop_reason = if has_tools {
                                    StopReason::ToolUse
                                } else {
                                    StopReason::EndTurn
                                };
                                let _ = event_sender.send(StreamEvent::MessageEnd {
                                    stop_reason,
                                });
                                break;
                            }

                            let data: serde_json::Value = match serde_json::from_str(&message.data) {
                                Ok(data) => data,
                                Err(error) => {
                                    tracing::warn!("failed to parse SSE data: {}", error);
                                    continue;
                                }
                            };

                            let Some(choice) = data.get("choices").and_then(|c| c.get(0)) else {
                                continue;
                            };

                            if let Some(finish_reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
                                let stop_reason = match finish_reason {
                                    "stop" => StopReason::EndTurn,
                                    "tool_calls" => StopReason::ToolUse,
                                    "length" => StopReason::MaxTokens,
                                    other => StopReason::Unknown(other.to_string()),
                                };

                                // Finalize pending tool calls with paired Start+End
                                let mut indices: Vec<i64> = tool_call_accumulators.keys().copied().collect();
                                indices.sort();
                                for index in indices {
                                    if let Some(accumulator) = tool_call_accumulators.remove(&index) {
                                        let _ = event_sender.send(StreamEvent::ToolUseStart {
                                            id: accumulator.id.clone(),
                                            name: accumulator.name.clone(),
                                        });
                                        let input = serde_json::from_str(&accumulator.arguments)
                                            .unwrap_or(serde_json::json!({}));
                                        let _ = event_sender.send(StreamEvent::ToolUseEnd { input });
                                    }
                                }

                                let _ = event_sender.send(StreamEvent::MessageEnd { stop_reason });
                                break;
                            }

                            let delta = &choice["delta"];

                            if let Some(text) = delta.get("content").and_then(|c| c.as_str()) {
                                if !text.is_empty() {
                                    let _ = event_sender.send(StreamEvent::TextDelta(text.to_string()));
                                }
                            }

                            if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
                                for tool_call in tool_calls {
                                    let index = tool_call.get("index").and_then(|i| i.as_i64()).unwrap_or(0);

                                    if let Some(id) = tool_call.get("id").and_then(|id| id.as_str()) {
                                        let name = tool_call["function"]["name"]
                                            .as_str()
                                            .unwrap_or_default()
                                            .to_string();

                                        tool_call_accumulators.insert(index, ToolCallAccumulator {
                                            id: id.to_string(),
                                            name,
                                            arguments: String::new(),
                                        });
                                    }

                                    if let Some(args) = tool_call
                                        .get("function")
                                        .and_then(|f| f.get("arguments"))
                                        .and_then(|a| a.as_str())
                                    {
                                        if !args.is_empty() {
                                            if let Some(accumulator) = tool_call_accumulators.get_mut(&index) {
                                                accumulator.arguments.push_str(args);
                                            }
                                            let _ = event_sender.send(StreamEvent::ToolInputDelta(args.to_string()));
                                        }
                                    }
                                }
                            }
                        }
                        Err(reqwest_eventsource::Error::StreamEnded) => {
                            break;
                        }
                        Err(error) => {
                            let _ = event_sender.send(StreamEvent::Error(error.to_string()));
                            return Err(AgshError::StreamError(error.to_string()));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "openai"
    }
}

struct ToolCallAccumulator {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    name: String,
    arguments: String,
}

// ---------------------------------------------------------------------------
// Claude provider
// ---------------------------------------------------------------------------

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
                    let now_millis = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|duration| duration.as_millis() as i64)
                        .unwrap_or(0);

                    let needs_refresh = if let Some(exp) = expires_at {
                        now_millis + 300_000 >= *exp
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
            let now_millis = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_millis() as i64)
                .unwrap_or(0);

            let needs_refresh = if let Some(exp) = expires_at {
                now_millis + 300_000 >= *exp
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

        if let Some(store) = &self.token_store {
            if let Err(error) = store.save_oauth_token("claude", &new_credential).await {
                tracing::warn!("failed to persist refreshed OAuth token: {}", error);
            }
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

        let expires_at = data.expires_in.map(|seconds| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_millis() as i64)
                .unwrap_or(0);
            now + (seconds as i64) * 1000
        });

        Ok(AuthCredential::OAuthToken {
            access_token: data.access_token,
            refresh_token: data
                .refresh_token
                .or_else(|| Some(refresh_token.to_string())),
            expires_at,
        })
    }

    fn build_request_body(
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

    fn parse_non_streaming_response(
        &self,
        response: &serde_json::Value,
    ) -> Result<(Message, StopReason)> {
        let stop_reason_str = response
            .get("stop_reason")
            .and_then(|reason| reason.as_str())
            .unwrap_or("end_turn");

        let stop_reason = match stop_reason_str {
            "end_turn" => StopReason::EndTurn,
            "tool_use" => StopReason::ToolUse,
            "max_tokens" => StopReason::MaxTokens,
            other => StopReason::Unknown(other.to_string()),
        };

        let content_array = response
            .get("content")
            .and_then(|c| c.as_array())
            .ok_or_else(|| AgshError::Provider("no content array in response".to_string()))?;

        let mut content_blocks = Vec::new();

        for block in content_array {
            let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");

            match block_type {
                "text" => {
                    if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                        content_blocks.push(ContentBlock::Text {
                            text: text.to_string(),
                        });
                    }
                }
                "tool_use" => {
                    let id = block
                        .get("id")
                        .and_then(|id| id.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let name = block
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let input = block.get("input").cloned().unwrap_or(serde_json::json!({}));

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
        use futures::StreamExt;
        use reqwest_eventsource::{Event, EventSource};

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

        let request = request.json(&body);

        let mut event_source = EventSource::new(request).map_err(|error| {
            AgshError::Provider(format!("failed to create SSE stream: {}", error))
        })?;

        let mut current_tool_input = String::new();
        let mut in_tool_use = false;

        loop {
            tokio::select! {
                _ = cancellation.cancelled() => {
                    event_source.close();
                    return Err(AgshError::Interrupted);
                }
                event = event_source.next() => {
                    let Some(event) = event else {
                        break;
                    };

                    match event {
                        Ok(Event::Open) => {}
                        Ok(Event::Message(message)) => {
                            let data: serde_json::Value = match serde_json::from_str(&message.data) {
                                Ok(data) => data,
                                Err(error) => {
                                    tracing::warn!("failed to parse SSE data: {}", error);
                                    continue;
                                }
                            };

                            match message.event.as_str() {
                                "content_block_start" => {
                                    let content_block = &data["content_block"];
                                    let block_type = content_block
                                        .get("type")
                                        .and_then(|t| t.as_str())
                                        .unwrap_or("");

                                    if block_type == "tool_use" {
                                        let id = content_block
                                            .get("id")
                                            .and_then(|id| id.as_str())
                                            .unwrap_or_default()
                                            .to_string();
                                        let name = content_block
                                            .get("name")
                                            .and_then(|n| n.as_str())
                                            .unwrap_or_default()
                                            .to_string();

                                        current_tool_input.clear();
                                        in_tool_use = true;
                                        let _ = event_sender.send(StreamEvent::ToolUseStart {
                                            id,
                                            name,
                                        });
                                    }
                                }
                                "content_block_delta" => {
                                    let delta = &data["delta"];
                                    let delta_type = delta
                                        .get("type")
                                        .and_then(|t| t.as_str())
                                        .unwrap_or("");

                                    match delta_type {
                                        "text_delta" => {
                                            if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                                if !text.is_empty() {
                                                    let _ = event_sender.send(
                                                        StreamEvent::TextDelta(text.to_string()),
                                                    );
                                                }
                                            }
                                        }
                                        "input_json_delta" => {
                                            if let Some(partial_json) =
                                                delta.get("partial_json").and_then(|p| p.as_str())
                                            {
                                                current_tool_input.push_str(partial_json);
                                                let _ = event_sender.send(
                                                    StreamEvent::ToolInputDelta(
                                                        partial_json.to_string(),
                                                    ),
                                                );
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
                                            serde_json::from_str(&current_tool_input)
                                                .unwrap_or(serde_json::json!({}))
                                        };
                                        let _ = event_sender
                                            .send(StreamEvent::ToolUseEnd { input });
                                        current_tool_input.clear();
                                        in_tool_use = false;
                                    }
                                }
                                "message_delta" => {
                                    let delta = &data["delta"];
                                    if let Some(stop_reason_str) =
                                        delta.get("stop_reason").and_then(|s| s.as_str())
                                    {
                                        let stop_reason = match stop_reason_str {
                                            "end_turn" => StopReason::EndTurn,
                                            "tool_use" => StopReason::ToolUse,
                                            "max_tokens" => StopReason::MaxTokens,
                                            other => StopReason::Unknown(other.to_string()),
                                        };
                                        let _ = event_sender
                                            .send(StreamEvent::MessageEnd { stop_reason });
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
                        Err(reqwest_eventsource::Error::StreamEnded) => {
                            break;
                        }
                        Err(error) => {
                            let _ = event_sender.send(StreamEvent::Error(error.to_string()));
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

// ---------------------------------------------------------------------------
// Provider factory
// ---------------------------------------------------------------------------

pub fn create_provider(
    provider_name: &str,
    credential: AuthCredential,
    model: String,
    base_url: Option<String>,
    client_id: Option<String>,
    oauth_token_url: Option<String>,
    token_store: Option<Arc<TokenStore>>,
) -> Result<Arc<dyn Provider>> {
    match provider_name {
        "openai" => {
            let api_key = match credential {
                AuthCredential::ApiKey(key) => key,
                AuthCredential::OAuthToken { access_token, .. } => access_token,
            };
            Ok(Arc::new(OpenAiProvider::new(api_key, model, base_url)))
        }
        "claude" => Ok(Arc::new(ClaudeProvider::new(
            credential,
            model,
            base_url,
            client_id,
            oauth_token_url,
            token_store,
        ))),
        other => Err(AgshError::Config(format!(
            "unknown provider: '{}'. Supported providers: openai, claude",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
    fn test_openai_request_body_simple() {
        let provider = OpenAiProvider::new("test-key".to_string(), "gpt-4o".to_string(), None);

        let messages = vec![Message::user("hello")];
        let body = provider.build_request_body("system prompt", &messages, &[], false);

        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["stream"], false);

        let openai_messages = body["messages"]
            .as_array()
            .expect("messages should be array");
        assert_eq!(openai_messages.len(), 2);
        assert_eq!(openai_messages[0]["role"], "system");
        assert_eq!(openai_messages[0]["content"], "system prompt");
        assert_eq!(openai_messages[1]["role"], "user");
        assert_eq!(openai_messages[1]["content"], "hello");
    }

    #[test]
    fn test_openai_request_body_with_tools() {
        let provider = OpenAiProvider::new("test-key".to_string(), "gpt-4o".to_string(), None);

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
        let openai_tools = body["tools"].as_array().expect("tools should be array");
        assert_eq!(openai_tools.len(), 1);
        assert_eq!(openai_tools[0]["type"], "function");
        assert_eq!(openai_tools[0]["function"]["name"], "read_file");
    }

    #[test]
    fn test_openai_request_body_with_tool_calls() {
        let provider = OpenAiProvider::new("test-key".to_string(), "gpt-4o".to_string(), None);

        let messages = vec![
            Message::user("read /tmp/test.txt"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test.txt"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: "file contents here".to_string(),
                    is_error: false,
                }],
            },
        ];

        let body = provider.build_request_body("", &messages, &[], false);
        let openai_messages = body["messages"]
            .as_array()
            .expect("messages should be array");

        assert_eq!(openai_messages[0]["role"], "user");
        assert_eq!(openai_messages[1]["role"], "assistant");
        assert!(openai_messages[1].get("tool_calls").is_some());
        assert_eq!(openai_messages[2]["role"], "tool");
        assert_eq!(openai_messages[2]["tool_call_id"], "call_1");
    }

    #[test]
    fn test_openai_parse_non_streaming_text() {
        let provider = OpenAiProvider::new("test-key".to_string(), "gpt-4o".to_string(), None);

        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello there!"
                },
                "finish_reason": "stop"
            }]
        });

        let (message, stop_reason) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");

        assert_eq!(message.text_content(), "Hello there!");
        assert_eq!(stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn test_openai_parse_non_streaming_tool_call() {
        let provider = OpenAiProvider::new("test-key".to_string(), "gpt-4o".to_string(), None);

        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\":\"/tmp/test.txt\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let (message, stop_reason) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");

        assert_eq!(stop_reason, StopReason::ToolUse);
        let tool_uses = message.tool_uses();
        assert_eq!(tool_uses.len(), 1);

        if let ContentBlock::ToolUse { id, name, input } = &tool_uses[0] {
            assert_eq!(id, "call_abc");
            assert_eq!(name, "read_file");
            assert_eq!(input["path"], "/tmp/test.txt");
        } else {
            panic!("expected ToolUse block");
        }
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
    fn test_claude_request_body_simple() {
        let provider = ClaudeProvider::new(
            AuthCredential::ApiKey("test-key".to_string()),
            "claude-sonnet-4-20250514".to_string(),
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

        // Assistant message with tool_use block
        assert_eq!(claude_messages[1]["role"], "assistant");
        let assistant_content = claude_messages[1]["content"]
            .as_array()
            .expect("content should be array");
        assert_eq!(assistant_content[0]["type"], "tool_use");
        assert_eq!(assistant_content[0]["id"], "toolu_1");
        assert_eq!(assistant_content[0]["name"], "read_file");

        // User message with tool_result block
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
        );

        let body = provider.build_request_body("", &[], &[], false);
        assert!(body.get("system").is_none());
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
