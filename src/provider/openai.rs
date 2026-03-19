use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::{AgshError, Result};

use super::{
    ContentBlock, Message, Provider, Role, StopReason, StreamEvent, ToolCallAccumulator,
    ToolDefinition, finalize_tool_call_accumulators,
};

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

    pub(super) fn build_request_body(
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

    pub(super) fn parse_non_streaming_response(
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

        let stop_reason = parse_openai_stop_reason(finish_reason);

        let assistant_message = choice
            .get("message")
            .ok_or_else(|| AgshError::Provider("no 'message' in choice".to_string()))?;
        let mut content_blocks = Vec::new();

        if let Some(text) = assistant_message
            .get("content")
            .and_then(|content| content.as_str())
            && !text.is_empty()
        {
            content_blocks.push(ContentBlock::Text {
                text: text.to_string(),
            });
        }

        if let Some(tool_calls) = assistant_message
            .get("tool_calls")
            .and_then(|tool_calls| tool_calls.as_array())
        {
            for tool_call in tool_calls {
                let id = tool_call
                    .get("id")
                    .and_then(|id| id.as_str())
                    .ok_or_else(|| AgshError::Provider("tool call missing 'id' field".to_string()))?
                    .to_string();
                let name = tool_call
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(|name| name.as_str())
                    .or_else(|| tool_call.get("name").and_then(|name| name.as_str()))
                    .ok_or_else(|| {
                        AgshError::Provider("tool call missing 'function.name' field".to_string())
                    })?
                    .to_string();
                let arguments_str = tool_call
                    .get("function")
                    .and_then(|function| function.get("arguments"))
                    .and_then(|arguments| arguments.as_str())
                    .or_else(|| {
                        tool_call
                            .get("arguments")
                            .and_then(|arguments| arguments.as_str())
                    })
                    .unwrap_or("{}");
                let input: serde_json::Value = match serde_json::from_str(arguments_str) {
                    Ok(value) => value,
                    Err(error) => {
                        tracing::warn!("failed to parse tool arguments: {}", error);
                        serde_json::json!({})
                    }
                };

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
        use eventsource_stream::Eventsource;
        use futures::StreamExt;

        let body = self.build_request_body(system_prompt, messages, tools, true);

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
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

        let mut tool_call_accumulators: std::collections::HashMap<i64, ToolCallAccumulator> =
            std::collections::HashMap::new();

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
                            if event.data == "[DONE]" {
                                let has_tools = finalize_tool_call_accumulators(
                                    &mut tool_call_accumulators,
                                    &event_sender,
                                );
                                let stop_reason = if has_tools {
                                    StopReason::ToolUse
                                } else {
                                    StopReason::EndTurn
                                };
                                if event_sender.send(StreamEvent::MessageEnd { stop_reason }).is_err() {
                                    tracing::trace!("stream event receiver dropped");
                                }
                                break;
                            }

                            let data: serde_json::Value = match serde_json::from_str(&event.data) {
                                Ok(data) => data,
                                Err(error) => {
                                    tracing::warn!("failed to parse SSE data: {}", error);
                                    continue;
                                }
                            };

                            let Some(choice) = data.get("choices").and_then(|choices| choices.get(0)) else {
                                continue;
                            };

                            if let Some(finish_reason) = choice.get("finish_reason").and_then(|reason| reason.as_str()) {
                                let stop_reason = parse_openai_stop_reason(finish_reason);
                                finalize_tool_call_accumulators(
                                    &mut tool_call_accumulators,
                                    &event_sender,
                                );
                                if event_sender.send(StreamEvent::MessageEnd { stop_reason }).is_err() {
                                    tracing::trace!("stream event receiver dropped");
                                }
                                break;
                            }

                            let Some(delta) = choice.get("delta") else {
                                continue;
                            };

                            if let Some(text) = delta.get("content").and_then(|content| content.as_str())
                                && !text.is_empty()
                                    && event_sender.send(StreamEvent::TextDelta(text.to_string())).is_err() {
                                        tracing::trace!("stream event receiver dropped");
                                        break;
                                    }

                            if let Some(tool_calls) = delta.get("tool_calls").and_then(|tool_calls| tool_calls.as_array()) {
                                for tool_call in tool_calls {
                                    let index = tool_call.get("index").and_then(|index| index.as_i64()).unwrap_or(0);

                                    let name = tool_call
                                        .get("function")
                                        .and_then(|function| function.get("name"))
                                        .and_then(|name| name.as_str())
                                        .or_else(|| tool_call.get("name").and_then(|name| name.as_str()));

                                    if let Some(id) = tool_call.get("id").and_then(|id| id.as_str()) {
                                        let accumulator = tool_call_accumulators
                                            .entry(index)
                                            .or_insert_with(|| ToolCallAccumulator {
                                                id: id.to_string(),
                                                name: String::new(),
                                                arguments: String::new(),
                                            });
                                        if let Some(name) = name {
                                            if accumulator.name.is_empty() {
                                                accumulator.name = name.to_string();
                                            }
                                        }
                                    } else if let Some(name) = name {
                                        if let Some(accumulator) = tool_call_accumulators.get_mut(&index) {
                                            if accumulator.name.is_empty() {
                                                accumulator.name = name.to_string();
                                            }
                                        }
                                    }

                                    if let Some(args) = tool_call
                                        .get("function")
                                        .and_then(|function| function.get("arguments"))
                                        .and_then(|arguments| arguments.as_str())
                                        .or_else(|| tool_call.get("arguments").and_then(|arguments| arguments.as_str()))
                                        && !args.is_empty() {
                                            if let Some(accumulator) = tool_call_accumulators.get_mut(&index) {
                                                accumulator.arguments.push_str(args);
                                            }
                                            if event_sender.send(StreamEvent::ToolInputDelta(args.to_string())).is_err() {
                                                tracing::trace!("stream event receiver dropped");
                                                break;
                                            }
                                        }
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
        "openai"
    }
}

fn parse_openai_stop_reason(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::EndTurn,
        "tool_calls" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        other => StopReason::Unknown(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_openai_parse_missing_message_in_choice() {
        let provider = OpenAiProvider::new("test-key".to_string(), "gpt-4o".to_string(), None);

        let response = serde_json::json!({
            "choices": [{
                "finish_reason": "stop"
            }]
        });

        let result = provider.parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_openai_parse_missing_tool_call_id() {
        let provider = OpenAiProvider::new("test-key".to_string(), "gpt-4o".to_string(), None);

        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let result = provider.parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_openai_parse_missing_tool_call_function_name() {
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
                            "arguments": "{}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let result = provider.parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_openai_parse_non_streaming_flattened_tool_call() {
        let provider = OpenAiProvider::new("test-key".to_string(), "gpt-4o".to_string(), None);

        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "name": "read_file",
                        "arguments": "{\"path\":\"/tmp/test.txt\"}"
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let (message, stop_reason) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse flattened tool call");

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
    fn test_openai_parse_non_streaming_flattened_missing_name_still_errors() {
        let provider = OpenAiProvider::new("test-key".to_string(), "gpt-4o".to_string(), None);

        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "arguments": "{}"
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let result = provider.parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_openai_tool_definitions_use_standard_chat_completions_format() {
        let provider = OpenAiProvider::new("test-key".to_string(), "gpt-4o".to_string(), None);

        let tools = vec![ToolDefinition {
            name: "write_file".to_string(),
            description: "Create or overwrite a file".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }),
        }];

        let body = provider.build_request_body("", &[], &tools, false);
        let openai_tools = body["tools"].as_array().expect("tools should be array");

        assert_eq!(openai_tools[0]["type"], "function");
        assert_eq!(openai_tools[0]["function"]["name"], "write_file");
        assert_eq!(
            openai_tools[0]["function"]["description"],
            "Create or overwrite a file"
        );
        assert!(openai_tools[0]["function"].get("parameters").is_some());

        // Top-level name/description/parameters must NOT be present to avoid
        // triggering Responses API strict validation on OpenAI/OpenRouter
        assert!(openai_tools[0].get("name").is_none());
        assert!(openai_tools[0].get("description").is_none());
        assert!(openai_tools[0].get("parameters").is_none());
    }
}
