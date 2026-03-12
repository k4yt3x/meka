use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::error::{AgshError, Result};
use crate::permission::SharedPermission;
use crate::provider::{
    ContentBlock, Message, Provider, Role, StopReason, StreamEvent, ToolDefinition,
};
use crate::render::{self, StreamingRenderer};
use crate::session::SessionManager;
use crate::system_prompt::build_system_prompt;
use crate::tools::ToolRegistry;

pub struct Agent {
    provider: Arc<dyn Provider>,
    tool_registry: ToolRegistry,
    session_manager: SessionManager,
    shared_permission: SharedPermission,
    streaming: bool,
    newline_before_prompt: bool,
    newline_after_prompt: bool,
    show_session_id: bool,
    sandboxed_shell: bool,
    context_messages: Option<usize>,
}

impl Agent {
    pub fn new(
        provider: Arc<dyn Provider>,
        tool_registry: ToolRegistry,
        session_manager: SessionManager,
        shared_permission: SharedPermission,
        streaming: bool,
        newline_before_prompt: bool,
        newline_after_prompt: bool,
        show_session_id: bool,
        sandboxed_shell: bool,
        context_messages: Option<usize>,
    ) -> Self {
        Self {
            provider,
            tool_registry,
            session_manager,
            shared_permission,
            streaming,
            newline_before_prompt,
            newline_after_prompt,
            show_session_id,
            sandboxed_shell,
            context_messages,
        }
    }

    pub async fn run_turn(
        &self,
        session_id: &mut Option<Uuid>,
        messages: &mut Vec<Message>,
        user_input: String,
        cancellation: CancellationToken,
    ) -> Result<()> {
        if session_id.is_none() {
            let id = self.session_manager.create_session().await?;
            *session_id = Some(id);
            if self.show_session_id {
                crate::render::render_session_id("Creating new session", &id.to_string());
            }
        }

        if self.newline_after_prompt {
            println!();
        }

        let sid = session_id.expect("session_id should be set");

        let user_message = Message::user(&user_input);
        messages.push(user_message);
        self.session_manager
            .save_message(sid, "user", &user_input)
            .await?;

        loop {
            if cancellation.is_cancelled() {
                return Err(AgshError::Interrupted);
            }

            let permission = self.shared_permission.get();
            let tools = self.available_tools(permission);
            let system_prompt = build_system_prompt(permission, &tools, self.sandboxed_shell);

            let api_messages = truncate_messages_for_context(messages, self.context_messages);

            let (assistant_message, stop_reason) = if self.streaming {
                self.run_streaming(&system_prompt, &api_messages, &tools, cancellation.clone())
                    .await?
            } else {
                self.provider
                    .complete(&system_prompt, &api_messages, &tools)
                    .await?
            };

            let content_json =
                serde_json::to_string(&assistant_message.content).unwrap_or_default();
            self.session_manager
                .save_message(sid, "assistant", &content_json)
                .await?;

            messages.push(assistant_message.clone());

            match stop_reason {
                StopReason::ToolUse => {
                    let tool_results = self
                        .execute_tool_calls(&assistant_message, cancellation.clone())
                        .await;

                    let result_message = Message {
                        role: Role::User,
                        content: tool_results,
                    };

                    let result_json =
                        serde_json::to_string(&result_message.content).unwrap_or_default();
                    self.session_manager
                        .save_message(sid, "tool_results", &result_json)
                        .await?;

                    messages.push(result_message);
                    // Continue the loop to let the model process tool results
                }
                StopReason::EndTurn | StopReason::MaxTokens | StopReason::Unknown(_) => {
                    break;
                }
            }
        }

        if self.newline_before_prompt {
            println!();
        }

        Ok(())
    }

    async fn run_streaming(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        cancellation: CancellationToken,
    ) -> Result<(Message, StopReason)> {
        let (event_sender, mut event_receiver) = mpsc::unbounded_channel::<StreamEvent>();

        let provider = Arc::clone(&self.provider);
        let system_prompt = system_prompt.to_string();
        let messages = messages.to_vec();
        let tools = tools.to_vec();
        let cancellation_clone = cancellation.clone();

        let stream_handle = tokio::spawn(async move {
            provider
                .stream(
                    &system_prompt,
                    &messages,
                    &tools,
                    event_sender,
                    cancellation_clone,
                )
                .await
        });

        let mut renderer = StreamingRenderer::new();
        let mut content_blocks: Vec<ContentBlock> = Vec::new();
        let mut current_text = String::new();
        let mut current_tool_id = String::new();
        let mut current_tool_name = String::new();
        let mut current_tool_input_json = String::new();
        let mut stop_reason = StopReason::EndTurn;

        while let Some(event) = event_receiver.recv().await {
            match event {
                StreamEvent::TextDelta(text) => {
                    current_text.push_str(&text);
                    renderer.push_delta(&text)?;
                }
                StreamEvent::ToolUseStart { id, name } => {
                    // Flush any accumulated text
                    if !current_text.is_empty() {
                        content_blocks.push(ContentBlock::Text {
                            text: std::mem::take(&mut current_text),
                        });
                    }
                    current_tool_id = id;
                    current_tool_name = name;
                    current_tool_input_json.clear();
                }
                StreamEvent::ToolInputDelta(delta) => {
                    current_tool_input_json.push_str(&delta);
                }
                StreamEvent::ToolUseEnd { input } => {
                    renderer.finish()?;
                    render::render_tool_indicator(&current_tool_name, &input);

                    content_blocks.push(ContentBlock::ToolUse {
                        id: std::mem::take(&mut current_tool_id),
                        name: std::mem::take(&mut current_tool_name),
                        input,
                    });
                    current_tool_input_json.clear();
                }
                StreamEvent::MessageEnd {
                    stop_reason: reason,
                } => {
                    stop_reason = reason;
                }
                StreamEvent::Error(error) => {
                    tracing::error!("stream error: {}", error);
                }
            }
        }

        // Flush remaining text
        if !current_text.is_empty() {
            content_blocks.push(ContentBlock::Text { text: current_text });
        }
        renderer.finish()?;

        // Wait for the stream task to complete
        match stream_handle.await {
            Ok(Ok(())) => {}
            Ok(Err(AgshError::Interrupted)) => return Err(AgshError::Interrupted),
            Ok(Err(error)) => return Err(error),
            Err(join_error) => {
                return Err(AgshError::Provider(format!(
                    "stream task panicked: {}",
                    join_error
                )));
            }
        }

        let message = Message {
            role: Role::Assistant,
            content: content_blocks,
        };

        Ok((message, stop_reason))
    }

    async fn execute_tool_calls(
        &self,
        assistant_message: &Message,
        cancellation: CancellationToken,
    ) -> Vec<ContentBlock> {
        let mut results = Vec::new();
        let permission = self.shared_permission.get();

        for block in &assistant_message.content {
            if let ContentBlock::ToolUse { id, name, input } = block {
                if !self.streaming {
                    render::render_tool_indicator(name, input);
                }

                let output = match self.tool_registry.get(name) {
                    None => {
                        let error_msg = format!("Unknown tool: '{}'", name);
                        crate::tools::ToolOutput {
                            content: error_msg,
                            is_error: true,
                        }
                    }
                    Some(tool) => {
                        let required = tool.required_permission();
                        if !permission.allows(required) {
                            let error_msg = format!(
                                "Permission denied: '{}' requires {} permission, current: {}",
                                name, required, permission
                            );
                            crate::tools::ToolOutput {
                                content: error_msg,
                                is_error: true,
                            }
                        } else {
                            match tool.execute(input.clone(), cancellation.clone()).await {
                                Ok(output) => output,
                                Err(AgshError::Interrupted) => crate::tools::ToolOutput {
                                    content: "Tool execution interrupted.".to_string(),
                                    is_error: true,
                                },
                                Err(error) => crate::tools::ToolOutput {
                                    content: format!("Tool error: {}", error),
                                    is_error: true,
                                },
                            }
                        }
                    }
                };

                results.push(ContentBlock::ToolResult {
                    tool_use_id: id.clone(),
                    content: output.content,
                    is_error: output.is_error,
                });
            }
        }

        results
    }

    pub async fn compact_session(
        &self,
        session_id: &mut Option<Uuid>,
        messages: &mut Vec<Message>,
    ) -> Result<()> {
        let Some(sid) = *session_id else {
            return Err(AgshError::Config(
                "no active session to compact".to_string(),
            ));
        };

        if messages.is_empty() {
            return Err(AgshError::Config("no messages to compact".to_string()));
        }

        let system_prompt = "You are a conversation summarizer. The user will ask you to \
             summarize the conversation so far. Produce a concise summary that preserves \
             all important information, decisions made, files discussed, and any ongoing \
             tasks. The summary will be used as the starting context for continuing this \
             conversation. Be thorough but concise. Write the summary in second person \
             (e.g., 'You were working on...').";

        // Clone messages and append a user message so the conversation ends with a
        // user turn. Some providers (e.g., Google) reject requests where the last
        // message has an assistant role.
        let mut compact_messages = messages.clone();
        compact_messages.push(Message::user(
            "Summarize our conversation so far into a concise context message.",
        ));

        let (summary_message, _stop_reason) = self
            .provider
            .complete(system_prompt, &compact_messages, &[])
            .await?;

        let summary_text = summary_message.text_content();
        if summary_text.is_empty() {
            return Err(AgshError::Provider(
                "LLM returned an empty summary".to_string(),
            ));
        }

        self.session_manager.clear_messages(sid).await?;

        messages.clear();

        let context_message = format!(
            "[Conversation summary from session compaction]\n\n{}",
            summary_text
        );
        let user_message = Message::user(&context_message);
        messages.push(user_message);

        self.session_manager
            .save_message(sid, "user", &context_message)
            .await?;

        Ok(())
    }

    fn available_tools(&self, permission: crate::permission::Permission) -> Vec<ToolDefinition> {
        self.tool_registry.definitions_for_permission(permission)
    }
}

fn truncate_messages_for_context(
    messages: &[Message],
    context_messages: Option<usize>,
) -> Vec<Message> {
    let Some(limit) = context_messages else {
        return messages.to_vec();
    };

    if messages.len() <= limit {
        return messages.to_vec();
    }

    let mut start_index = messages.len().saturating_sub(limit);

    // Walk backward to find a safe cut point: a user message that is NOT a
    // tool_results message. This avoids splitting assistant(ToolUse) →
    // user(ToolResult) chains and ensures the first message has role User
    // (required by Claude API).
    loop {
        if start_index == 0 {
            break;
        }

        let message = &messages[start_index];
        if message.role == Role::User && !has_tool_results(&message.content) {
            break;
        }

        start_index -= 1;
    }

    messages[start_index..].to_vec()
}

fn has_tool_results(content: &[ContentBlock]) -> bool {
    content
        .iter()
        .any(|block| matches!(block, ContentBlock::ToolResult { .. }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_msg(text: &str) -> Message {
        Message::user(text)
    }

    fn assistant_msg(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    fn assistant_tool_use() -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                input: serde_json::json!({"path": "/tmp/test"}),
            }],
        }
    }

    fn tool_result_msg() -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: "file contents".to_string(),
                is_error: false,
            }],
        }
    }

    #[test]
    fn test_truncate_no_limit() {
        let messages = vec![user_msg("hello"), assistant_msg("hi")];
        let result = truncate_messages_for_context(&messages, None);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_truncate_under_limit() {
        let messages = vec![user_msg("hello"), assistant_msg("hi")];
        let result = truncate_messages_for_context(&messages, Some(10));
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_truncate_over_limit() {
        let messages = vec![
            user_msg("first"),
            assistant_msg("response1"),
            user_msg("second"),
            assistant_msg("response2"),
            user_msg("third"),
            assistant_msg("response3"),
        ];
        let result = truncate_messages_for_context(&messages, Some(4));
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].role, Role::User);
    }

    #[test]
    fn test_truncate_does_not_split_tool_chain() {
        let messages = vec![
            user_msg("first"),
            assistant_msg("response1"),
            user_msg("second"),
            assistant_tool_use(),
            tool_result_msg(),
            assistant_msg("final"),
        ];
        // Limit 3 would naively start at index 3 (assistant_tool_use), but that
        // splits the tool chain. It should walk back to index 2 (user "second").
        let result = truncate_messages_for_context(&messages, Some(3));
        assert_eq!(result[0].role, Role::User);
        assert!(!has_tool_results(&result[0].content));
        assert!(result.len() >= 3);
    }

    #[test]
    fn test_truncate_starts_with_user() {
        let messages = vec![
            user_msg("first"),
            assistant_msg("response1"),
            assistant_msg("response2"),
            user_msg("second"),
            assistant_msg("response3"),
        ];
        // Limit 2 would naively start at index 3, which is a user message
        let result = truncate_messages_for_context(&messages, Some(2));
        assert_eq!(result[0].role, Role::User);
    }

    #[test]
    fn test_truncate_walks_back_past_tool_result() {
        let messages = vec![
            user_msg("first"),
            assistant_tool_use(),
            tool_result_msg(),
            assistant_msg("response"),
            user_msg("second"),
            assistant_msg("response2"),
        ];
        // Limit 4 would naively start at index 2 (tool_result_msg), should walk
        // back to index 0 (user "first")
        let result = truncate_messages_for_context(&messages, Some(4));
        assert_eq!(result[0].role, Role::User);
        assert!(!has_tool_results(&result[0].content));
    }
}
