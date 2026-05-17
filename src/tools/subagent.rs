//! `spawn_agent` tool: delegates a self-contained research/exploration task
//! to a fresh sub-agent with its own conversation, returning the
//! sub-agent's final report as a single tool result.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::context::build_environment_context;
use crate::conversation::Conversation;
use crate::error::{AgshError, Result};
use crate::permission::{Permission, SharedPermission};
use crate::provider::{ContentBlock, Message, Provider, StopReason, ToolDefinition};

use super::{BuiltinToolFilter, Tool, ToolOutput, ToolRegistry};

/// Parameters needed to build a fresh ToolRegistry for sub-agents.
#[derive(Clone)]
pub struct ToolBuilderParams {
    pub web_client: crate::config::WebClientConfig,
    pub sandbox_enabled: bool,
    pub sandbox_capability: crate::sandbox::SandboxCapability,
    pub sandbox_backend: crate::config::SandboxBackend,
    pub backend_probe: crate::sandbox::BackendProbe,
    /// Parent's `[tools]` filter — sub-agents inherit it.
    pub builtin_filter: BuiltinToolFilter,
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
            ..Default::default()
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
        let sub_shared_perm = SharedPermission::new(sub_perm, self.parent_permission.enabled());
        let sub_registry = ToolRegistry::build_for_subagent(
            self.tool_builder_params.web_client.clone(),
            sub_shared_perm,
            self.tool_builder_params.sandbox_enabled,
            self.tool_builder_params.sandbox_capability.clone(),
            self.tool_builder_params.sandbox_backend,
            self.tool_builder_params.backend_probe.clone(),
            self.tool_builder_params.builtin_filter.clone(),
        )
        .map_err(|error| AgshError::ToolExecution {
            tool_name: "spawn_agent".to_string(),
            message: format!("failed to build sub-agent tool registry: {}", error),
        })?;

        let tools = sub_registry.definitions_for_permission(sub_perm);
        let system_prompt =
            build_subagent_system_prompt(sub_perm, &tools, self.user_instructions.as_deref());

        let environment_context = build_environment_context(sub_perm);
        let augmented_prompt = format!("{}\n{}", environment_context, prompt);
        let mut messages = Conversation::new();
        messages.append(Message::user(&augmented_prompt));

        // No report-length truncation here: the agent layer's
        // `persist_oversized_results` auto-persists any oversized report to
        // the scratchpad losslessly, and `save_explicit_scratchpad_results`
        // handles explicit redirections.
        let report = run_subagent_loop(
            &*self.provider,
            &sub_registry,
            &system_prompt,
            &mut messages,
            sub_perm,
            &tools,
            cancellation,
        )
        .await?;

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
    messages: &mut Conversation,
    permission: Permission,
    tools: &[ToolDefinition],
    cancellation: CancellationToken,
) -> Result<String> {
    let max_iterations = 20;

    for _ in 0..max_iterations {
        if cancellation.is_cancelled() {
            return Err(AgshError::Interrupted);
        }

        let (assistant_message, stop_reason, _usage) = provider
            .complete(system_prompt, messages.as_slice(), tools)
            .await?;

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
        messages.append(cleaned.clone());

        match stop_reason {
            StopReason::ToolUse => {
                let mut results = Vec::new();
                for block in &cleaned.content {
                    if let ContentBlock::ToolUse { id, name, input } = block {
                        let output = match tool_registry.get(name) {
                            None => ToolOutput::text(format!("Unknown tool: '{}'", name), true),
                            Some(tool) => {
                                let required = tool_registry
                                    .required_permission_for(name)
                                    .unwrap_or_else(|| tool.required_permission());
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

                messages.append(Message {
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
