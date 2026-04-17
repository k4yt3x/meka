use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::context::build_environment_context;
use crate::error::{AgshError, Result};
use crate::permission::{Permission, SharedPermission};
use crate::provider::{ContentBlock, Message, Provider, StopReason, ToolDefinition};

use super::{Tool, ToolOutput, ToolRegistry};

/// Parameters needed to build a fresh ToolRegistry for sub-agents.
#[derive(Clone)]
pub struct ToolBuilderParams {
    pub user_agent: String,
    pub sandbox_enabled: bool,
    pub sandbox_capability: crate::sandbox::SandboxCapability,
}

pub struct SpawnAgentTool {
    pub provider: Arc<dyn Provider>,
    pub parent_permission: SharedPermission,
    pub tool_builder_params: ToolBuilderParams,
    pub user_instructions: Option<String>,
}

#[async_trait]
impl Tool for SpawnAgentTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "spawn_agent".to_string(),
            description: "Spawn a sub-agent to perform a research or analysis task. The \
                          sub-agent runs with read-only permissions and returns a concise \
                          report. Use this to delegate exploration, search, or analysis \
                          tasks without polluting the main conversation context."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task description for the sub-agent"
                    },
                    "scratchpad": {
                        "type": "string",
                        "description": "If provided, save the output to the scratchpad under this name instead of returning it inline."
                    }
                },
                "required": ["prompt"]
            }),
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let prompt = input["prompt"]
            .as_str()
            .ok_or_else(|| AgshError::ToolExecution {
                tool_name: "spawn_agent".to_string(),
                message: "missing 'prompt' parameter".to_string(),
            })?
            .to_string();

        let parent_perm = self.parent_permission.get();
        let sub_perm = match parent_perm {
            Permission::None => Permission::None,
            _ => Permission::Read,
        };

        // Build a sub-agent tool registry: no spawn_agent (prevents recursion)
        // and no todo_write (parent owns task tracking).
        let sub_shared_perm = SharedPermission::new(sub_perm);
        let sub_registry = ToolRegistry::build_for_subagent(
            self.tool_builder_params.user_agent.clone(),
            sub_shared_perm,
            self.tool_builder_params.sandbox_enabled,
            self.tool_builder_params.sandbox_capability,
        );

        let tools = sub_registry.definitions_for_permission(sub_perm);
        let system_prompt =
            build_subagent_system_prompt(sub_perm, &tools, self.user_instructions.as_deref());

        let environment_context = build_environment_context(sub_perm);
        let augmented_prompt = format!("{}\n{}", environment_context, prompt);
        let mut messages = vec![Message::user(&augmented_prompt)];

        const MAX_REPORT_CHARS: usize = 30_000;

        let mut report = run_subagent_loop(
            &*self.provider,
            &sub_registry,
            &system_prompt,
            &mut messages,
            sub_perm,
            &tools,
            cancellation,
        )
        .await?;

        if report.len() > MAX_REPORT_CHARS {
            let boundary = report.floor_char_boundary(MAX_REPORT_CHARS);
            report.truncate(boundary);
            report.push_str("\n\n... (sub-agent report truncated, showing first 30000 characters)");
        }

        Ok(ToolOutput::text(report, false))
    }
}

fn build_subagent_system_prompt(
    permission: Permission,
    tools: &[ToolDefinition],
    user_instructions: Option<&str>,
) -> String {
    let mut prompt = String::new();
    prompt.push_str(
        "You are a research sub-agent. Complete the assigned task using the \
         available tools, then produce a concise final report summarizing your \
         findings. Do not ask follow-up questions — work with what you have.\n\n",
    );

    prompt.push_str(&format!("## Permission Level: {}\n\n", permission));

    if let Some(instructions) = user_instructions
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        prompt.push_str("## User Instructions\n\n");
        prompt.push_str(
            "These are installation-specific rules set by the user. Treat them as \
             hard constraints unless they conflict with safety requirements.\n\n",
        );
        prompt.push_str(instructions);
        prompt.push_str("\n\n");
    }

    if !tools.is_empty() {
        prompt.push_str("## Available Tools\n\n");
        for tool in tools {
            prompt.push_str(&format!("- **{}**: {}\n", tool.name, tool.description));
        }
        prompt.push('\n');
    }

    prompt
}

async fn run_subagent_loop(
    provider: &dyn Provider,
    tool_registry: &ToolRegistry,
    system_prompt: &str,
    messages: &mut Vec<Message>,
    permission: Permission,
    tools: &[ToolDefinition],
    cancellation: CancellationToken,
) -> Result<String> {
    let max_iterations = 20;

    for _ in 0..max_iterations {
        if cancellation.is_cancelled() {
            return Err(AgshError::Interrupted);
        }

        let (assistant_message, stop_reason, _usage) =
            provider.complete(system_prompt, messages, tools).await?;

        // Strip thinking blocks
        let cleaned = Message {
            role: crate::provider::Role::Assistant,
            content: assistant_message
                .content
                .iter()
                .filter(|block| !matches!(block, ContentBlock::Thinking { .. }))
                .cloned()
                .collect(),
        };
        messages.push(cleaned.clone());

        match stop_reason {
            StopReason::ToolUse => {
                let mut results = Vec::new();
                for block in &cleaned.content {
                    if let ContentBlock::ToolUse { id, name, input } = block {
                        let output = match tool_registry.get(name) {
                            None => ToolOutput::text(format!("Unknown tool: '{}'", name), true),
                            Some(tool) => {
                                let required = tool.required_permission();
                                if !permission.allows(required) {
                                    ToolOutput::text(
                                        format!(
                                            "Permission denied: '{}' requires {}",
                                            name, required
                                        ),
                                        true,
                                    )
                                } else {
                                    match tool.execute(input.clone(), cancellation.clone()).await {
                                        Ok(output) => output,
                                        Err(AgshError::Interrupted) => {
                                            return Err(AgshError::Interrupted);
                                        }
                                        Err(error) => {
                                            ToolOutput::text(format!("Tool error: {}", error), true)
                                        }
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

                messages.push(Message {
                    role: crate::provider::Role::User,
                    content: results,
                });
            }
            StopReason::EndTurn | StopReason::MaxTokens | StopReason::Unknown(_) => {
                return Ok(cleaned.text_content());
            }
        }
    }

    // If we hit the iteration limit, return what we have
    messages
        .last()
        .map(|msg| msg.text_content())
        .ok_or_else(|| AgshError::Provider("sub-agent produced no output".to_string()))
}
