//! Per-turn agent loop: streams provider output, dispatches tool calls, and
//! persists the resulting messages to the session store. Also handles
//! mid-conversation auto-compaction when the input-token budget is exceeded.

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::context;
use crate::conversation::Conversation;
use crate::error::{AgshError, Result};
use crate::permission::SharedPermission;
use crate::provider::{
    ContentBlock, Message, Provider, Role, StopReason, StreamEvent, ToolDefinition,
};
use crate::render::{self, StreamingRenderer};
use crate::session::SessionManager;
use crate::tools::ToolRegistry;
use crate::tools::todo::SharedTodoList;

/// Trigger auto-compaction once a turn's input tokens exceed this fraction of
/// the configured context window.
const AUTO_COMPACT_THRESHOLD_PERCENT: u64 = 80;

/// Per-turn configuration knobs for [`Agent`]. Constructed once by `main` from
/// the [`crate::config::ResolvedConfig`] and held immutably for the agent's
/// lifetime; mid-session permission cycling and tool loading are handled by
/// shared state (see [`SharedPermission`] and [`ToolRegistry`]) rather than
/// by mutating fields here.
pub struct AgentOptions {
    /// When true, assistant responses stream token-by-token via
    /// `Provider::stream`; otherwise the agent uses the blocking
    /// `Provider::complete`.
    pub streaming: bool,
    pub newline_before_prompt: bool,
    pub newline_after_prompt: bool,
    pub show_session_id_on_create: bool,
    pub show_token_usage: bool,
    /// Whether read-mode `execute_command` calls run inside the platform
    /// sandbox. Forced off when no sandbox backend is available.
    pub sandboxed_shell: bool,
    pub render_mode: crate::render::RenderMode,
    /// Cap on messages sent to the provider per turn. `None` = unlimited;
    /// the agent walks back to a safe boundary so tool-result chains stay
    /// intact (see `truncate_messages_for_context`).
    pub context_messages: Option<usize>,
    /// When true, the agent auto-compacts the conversation once a turn's
    /// input tokens cross [`AUTO_COMPACT_THRESHOLD_PERCENT`] of
    /// [`Self::context_window`]. Requires `context_window > 0`.
    pub auto_compact: bool,
    /// Provider's advertised context window in tokens. Drives auto-compact.
    pub context_window: u64,
    pub thinking_show_content: bool,
    /// User-authored instructions, surfaced in the system prompt and to
    /// sub-agents. Per-run `--instructions` overrides the config-file value.
    pub user_instructions: Option<String>,
    /// Pre-turn MCP readiness gate. When true, a turn is rejected with
    /// `AgshError::McpTurnGated` if any enabled server isn't `Connected`
    /// after [`Self::mcp_grace`].
    pub mcp_strict: bool,
    /// Max time to wait for still-`Pending` MCP servers to reach
    /// `Connected` before applying the strict check.
    pub mcp_grace: std::time::Duration,
}

/// Driver for a single conversation. One [`Agent`] handles one or more
/// sequential turns against a single provider, with a shared tool registry,
/// shared permission state, and a persistent SQLite session. A turn fans
/// out tool calls (in parallel via `join_all`) and persists every assistant
/// and tool-result message to the session store.
///
/// `Agent` is held across turns but not across providers — switching
/// providers requires a fresh instance.
pub struct Agent {
    provider: Arc<dyn Provider>,
    tool_registry: ToolRegistry,
    session_manager: SessionManager,
    shared_permission: SharedPermission,
    options: AgentOptions,
    todo_list: SharedTodoList,
    shared_session_id: Arc<tokio::sync::RwLock<Option<uuid::Uuid>>>,
    approval_sender: Option<std::sync::mpsc::Sender<crate::repl::AgentToReplEvent>>,
    last_input_tokens: std::sync::atomic::AtomicU64,
    /// Per-turn map of `tool_use_id` → scratchpad-name hint. Populated by
    /// MCP tool adapters so oversized-output persistence uses
    /// `mcp_<server>_<tool>` instead of the plain tool name. Cleared
    /// between turns by `persist_oversized_results`.
    scratchpad_hints: Arc<tokio::sync::RwLock<std::collections::HashMap<String, String>>>,
    /// Optional MCP client manager; used to read server-supplied
    /// `InitializeResult.instructions` for inclusion in the system prompt.
    mcp_manager: Option<Arc<crate::mcp::McpClientManager>>,
    /// Counters surfaced by `/status`. Shared with the Claude providers,
    /// which increment the redaction-related fields when oversized
    /// request bodies trigger image-block redaction.
    session_stats: Arc<crate::stats::SessionStats>,
}

impl Agent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider: Arc<dyn Provider>,
        tool_registry: ToolRegistry,
        session_manager: SessionManager,
        shared_permission: SharedPermission,
        options: AgentOptions,
        todo_list: SharedTodoList,
        shared_session_id: Arc<tokio::sync::RwLock<Option<uuid::Uuid>>>,
        approval_sender: Option<std::sync::mpsc::Sender<crate::repl::AgentToReplEvent>>,
        session_stats: Arc<crate::stats::SessionStats>,
    ) -> Self {
        Self {
            provider,
            tool_registry,
            session_manager,
            shared_permission,
            options,
            todo_list,
            shared_session_id,
            approval_sender,
            last_input_tokens: std::sync::atomic::AtomicU64::new(0),
            scratchpad_hints: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            mcp_manager: None,
            session_stats,
        }
    }

    /// Snapshot of the per-session counters used by `/status`. Called
    /// from the REPL on demand.
    pub fn session_stats_snapshot(&self) -> crate::stats::SessionStatsSnapshot {
        self.session_stats.snapshot()
    }

    /// Attach the MCP client manager so server-supplied `initialize`
    /// instructions can be injected into each turn's system prompt.
    pub fn set_mcp_manager(&mut self, manager: Arc<crate::mcp::McpClientManager>) {
        self.mcp_manager = Some(manager);
    }

    /// Per-turn MCP readiness gate. Applies to every turn (not just the
    /// first) so mid-session reconnects also gate cleanly. Awaits
    /// `grace` for Pending servers to finish connecting; then:
    /// - all enabled servers `Connected` → `Ok(())`.
    /// - some still not Connected + `strict` → `Err(McpTurnGated)`.
    /// - some still not Connected + `!strict` → `Ok(())` with a warn.
    ///
    /// No-op when no MCP manager is attached (e.g. sub-agents).
    async fn await_mcp_ready(&self) -> Result<()> {
        let Some(manager) = self.mcp_manager.as_ref() else {
            return Ok(());
        };
        if manager.all_ready() {
            let not_ready = manager.enabled_not_connected().await;
            if not_ready.is_empty() {
                return Ok(());
            }
            return self.handle_mcp_not_ready(not_ready);
        }

        // Best-effort grace wait — we re-check readiness below regardless of
        // whether `await_settled` returned in time. The timeout result is
        // intentionally discarded.
        let _ = tokio::time::timeout(self.options.mcp_grace, manager.await_settled()).await;

        let not_ready = manager.enabled_not_connected().await;
        if not_ready.is_empty() {
            return Ok(());
        }
        self.handle_mcp_not_ready(not_ready)
    }

    fn handle_mcp_not_ready(
        &self,
        not_ready: Vec<(String, crate::mcp::ServerState)>,
    ) -> Result<()> {
        if self.options.mcp_strict {
            let summary: Vec<(String, String)> = not_ready
                .iter()
                .map(|(name, state)| (name.clone(), state.label().to_string()))
                .collect();
            Err(AgshError::McpTurnGated { servers: summary })
        } else {
            let names: Vec<&str> = not_ready.iter().map(|(n, _)| n.as_str()).collect();
            tracing::warn!(
                "mcp: proceeding without {} server(s): {:?} (set [mcp].strict = true to gate)",
                names.len(),
                names
            );
            Ok(())
        }
    }

    pub async fn run_turn(
        &self,
        session_id: &mut Option<Uuid>,
        messages: &mut Conversation,
        user_input: String,
        cancellation: CancellationToken,
    ) -> Result<()> {
        // Gate on MCP readiness BEFORE touching session state / message
        // history so a rejected turn leaves no trace in the conversation.
        self.await_mcp_ready().await?;

        if session_id.is_none() {
            let id = self.session_manager.create_session().await?;
            *session_id = Some(id);
            if self.options.show_session_id_on_create {
                crate::render::render_session_id("Creating new session", &id.to_string());
            }
        }

        let mut spacing = render::OutputSpacing::new();

        if self.options.newline_after_prompt {
            eprintln!();
            spacing.after_prompt();
        }

        let sid = session_id.ok_or(AgshError::Config("session_id not set".into()))?;

        // Keep the shared session ID in sync so scratchpad tools can access it.
        *self.shared_session_id.write().await = Some(sid);

        // Auto-compact if the last turn's input tokens exceeded the threshold
        // fraction of the context window. This runs between turns (not
        // mid-tool-loop) so the stable base_messages invariant is preserved.
        if self.options.auto_compact && self.options.context_window > 0 {
            let last_tokens = self
                .last_input_tokens
                .load(std::sync::atomic::Ordering::Relaxed);
            let threshold = self.options.context_window * AUTO_COMPACT_THRESHOLD_PERCENT / 100;
            if last_tokens > threshold && messages.len() > 1 {
                tracing::info!(
                    "auto-compacting: {} input tokens exceeds {}% of {} context window",
                    last_tokens,
                    AUTO_COMPACT_THRESHOLD_PERCENT,
                    self.options.context_window
                );
                // Automatic lifecycle signpost — hidden at default
                // verbosity, surface with `-v` / `RUST_LOG`.
                tracing::info!("auto-compacting conversation");
                if let Err(error) = self.compact_session(session_id, messages).await {
                    tracing::warn!("auto-compact failed: {}", error);
                }
            }
        }

        let permission = self.shared_permission.get();

        let catalogue = self.tool_registry.tool_catalogue();
        let augmented_input = {
            let todos = self.todo_list.read().await;
            let block = context::build_turn_context(permission, &todos);
            format!("{}\n\n{}", block, user_input)
        };
        let user_message = Message::user(&augmented_input);
        messages.append(user_message);
        let skills = crate::skills::discover_skills();
        let mcp_instructions = self
            .mcp_manager
            .as_ref()
            .map(|manager| manager.server_instructions())
            .unwrap_or_default();
        let system_prompt = context::build_system_prompt(
            &catalogue,
            self.options.sandboxed_shell,
            &skills,
            self.options.user_instructions.as_deref(),
            &mcp_instructions,
        );

        let base_messages =
            truncate_messages_for_context(messages.as_slice(), self.options.context_messages);
        let turn_start_len = messages.len();

        let mut user_saved = false;
        // Accumulate token usage across every provider call within this
        // turn so the per-turn display reflects the whole turn (including
        // tool-execution loops), not just the final round-trip.
        let mut turn_usage = crate::provider::TokenUsage::default();

        let result: Result<()> = 'turn: {
            loop {
                if cancellation.is_cancelled() {
                    break 'turn Err(AgshError::Interrupted);
                }

                let api_messages = if messages.len() > turn_start_len {
                    let mut combined = base_messages.clone();
                    combined.extend_from_slice(&messages.as_slice()[turn_start_len..]);
                    combined
                } else {
                    base_messages.clone()
                };

                // Recompute the active tool set every iteration so a
                // `load_tool` call earlier in this turn becomes visible to
                // the model on the very next request — without mutating
                // any registry state. Append-only growth keeps the tools
                // array's cache prefix stable.
                //
                // Read from events (not the materialized slice) so the
                // deferred-tool snapshot stored on `Event::CompactBoundary`
                // survives across compaction; otherwise tools the model
                // loaded pre-compaction would silently drop out of the
                // active set on the next turn.
                let loaded =
                    crate::conversation::extract_loaded_tool_names_from_events(messages.events());
                let tools = self.tool_registry.definitions_active_with_loaded(&loaded);

                let (assistant_message, stop_reason, usage) = match if self.options.streaming {
                    self.run_streaming(
                        &system_prompt,
                        &api_messages,
                        &tools,
                        cancellation.clone(),
                        &mut spacing,
                    )
                    .await
                } else {
                    self.provider
                        .complete(&system_prompt, &api_messages, &tools)
                        .await
                } {
                    Ok(value) => value,
                    Err(error) => break 'turn Err(error),
                };

                self.last_input_tokens
                    .store(usage.input_tokens, std::sync::atomic::Ordering::Relaxed);
                turn_usage.input_tokens =
                    turn_usage.input_tokens.saturating_add(usage.input_tokens);
                turn_usage.output_tokens =
                    turn_usage.output_tokens.saturating_add(usage.output_tokens);
                turn_usage.cache_creation_input_tokens = turn_usage
                    .cache_creation_input_tokens
                    .saturating_add(usage.cache_creation_input_tokens);
                turn_usage.cache_read_input_tokens = turn_usage
                    .cache_read_input_tokens
                    .saturating_add(usage.cache_read_input_tokens);

                if !user_saved {
                    let user_event =
                        crate::conversation::Event::Append(Message::user(augmented_input.as_str()));
                    if let Err(error) = self.session_manager.save_event(sid, &user_event).await {
                        break 'turn Err(error);
                    }
                    user_saved = true;
                }

                let assistant_event = crate::conversation::Event::Append(assistant_message.clone());
                if let Err(error) = self.session_manager.save_event(sid, &assistant_event).await {
                    break 'turn Err(error);
                }

                // Thinking blocks are preserved in the conversation
                // for the Claude API (interleaved-thinking beta). The provider's
                // build_messages handles stripping trailing thinking from the last
                // assistant message.
                messages.append(assistant_message.clone());

                if cancellation.is_cancelled() {
                    break 'turn Err(AgshError::Interrupted);
                }

                match stop_reason {
                    StopReason::ToolUse => {
                        let mut tool_results = self
                            .execute_tool_calls(
                                &assistant_message,
                                cancellation.clone(),
                                &mut spacing,
                            )
                            .await;

                        if let Err(error) =
                            crate::tools::scratchpad::save_explicit_scratchpad_results(
                                &self.session_manager,
                                sid,
                                &assistant_message,
                                &mut tool_results,
                            )
                            .await
                        {
                            tracing::warn!("failed to save explicit scratchpad results: {}", error);
                        }

                        let hints_snapshot = {
                            let guard = self.scratchpad_hints.read().await;
                            guard.clone()
                        };
                        if let Err(error) = crate::tools::scratchpad::persist_oversized_results(
                            &self.session_manager,
                            sid,
                            &assistant_message,
                            &mut tool_results,
                            &hints_snapshot,
                        )
                        .await
                        {
                            tracing::warn!("failed to persist oversized tool results: {}", error);
                        }
                        // Drop the per-turn hints so a long session doesn't
                        // accumulate entries for tool calls that already ran.
                        self.scratchpad_hints.write().await.clear();

                        let result_message = Message {
                            role: Role::User,
                            content: tool_results,
                        };

                        let result_event =
                            crate::conversation::Event::Append(result_message.clone());
                        if let Err(error) =
                            self.session_manager.save_event(sid, &result_event).await
                        {
                            break 'turn Err(error);
                        }

                        messages.append(result_message);
                    }
                    StopReason::EndTurn | StopReason::MaxTokens | StopReason::Unknown(_) => {
                        break 'turn Ok(());
                    }
                }
            }
        };

        if result.is_ok() {
            // Roll the turn into the session-level counters surfaced by
            // `/status`. Done here (not inside the inner loop) so a single
            // `/status` reading reflects whole turns, not partial state.
            self.session_stats.record_turn(&turn_usage);
            if self.options.show_token_usage {
                crate::render::render_token_usage(&turn_usage);
            }
        }

        if result.is_ok() && self.options.newline_before_prompt {
            eprintln!();
        }

        match &result {
            Err(AgshError::Interrupted) if !user_saved => {
                let user_event =
                    crate::conversation::Event::Append(Message::user(augmented_input.as_str()));
                if let Err(error) = self.session_manager.save_event(sid, &user_event).await {
                    tracing::error!("failed to save user message on interruption: {}", error);
                }
            }
            Err(error) if !matches!(error, AgshError::Interrupted) && !user_saved => {
                messages.pop_unsaved();
            }
            _ => {}
        }

        result
    }

    async fn run_streaming(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        cancellation: CancellationToken,
        spacing: &mut render::OutputSpacing,
    ) -> Result<(Message, StopReason, crate::provider::TokenUsage)> {
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

        let mut renderer = StreamingRenderer::new(self.options.render_mode);
        let mut content_blocks: Vec<ContentBlock> = Vec::new();
        let mut current_text = String::new();
        let mut current_thinking = String::new();
        let mut current_tool_id = String::new();
        let mut current_tool_name = String::new();
        let mut current_tool_input_json = String::new();
        let mut stop_reason = StopReason::EndTurn;
        let mut token_usage = crate::provider::TokenUsage::default();
        let show_thinking = self.options.thinking_show_content;

        while let Some(event) = event_receiver.recv().await {
            match event {
                StreamEvent::ThinkingDelta(text) => {
                    current_thinking.push_str(&text);
                }
                StreamEvent::ThinkingComplete { signature } => {
                    if !current_thinking.is_empty() {
                        if spacing.before_thinking() {
                            eprintln!();
                        }
                        render::render_thinking_block(&current_thinking, show_thinking);
                        content_blocks.push(ContentBlock::Thinking {
                            thinking: std::mem::take(&mut current_thinking),
                            signature,
                        });
                    }
                }
                StreamEvent::TextDelta(text) => {
                    if !renderer.started && spacing.before_text() {
                        eprintln!();
                    }
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
                    if spacing.before_tool_indicator() {
                        eprintln!();
                    }
                    let schema = self
                        .tool_registry
                        .get(&current_tool_name)
                        .map(|t| t.definition().parameters);
                    render::render_tool_indicator(&current_tool_name, &input, schema.as_ref());

                    content_blocks.push(ContentBlock::ToolUse {
                        id: std::mem::take(&mut current_tool_id),
                        name: std::mem::take(&mut current_tool_name),
                        input,
                    });
                    current_tool_input_json.clear();
                }
                StreamEvent::ToolCallRejected { id, name, reason } => {
                    // A malformed tool-call arrived (bad JSON). Emit a
                    // `ToolUse` block with a sentinel marker so the shape
                    // of the assistant message stays valid for the API
                    // round-trip, but `resolve_and_execute_tool` sees the
                    // marker and surfaces an error back to the model
                    // rather than running the tool on a silently-empty
                    // argument object.
                    renderer.finish()?;
                    if spacing.before_tool_indicator() {
                        eprintln!();
                    }
                    let marker_input = serde_json::json!({
                        crate::provider::INVALID_TOOL_ARGS_MARKER: reason,
                    });
                    let schema = self
                        .tool_registry
                        .get(&name)
                        .map(|t| t.definition().parameters);
                    render::render_tool_indicator(&name, &marker_input, schema.as_ref());
                    content_blocks.push(ContentBlock::ToolUse {
                        id,
                        name,
                        input: marker_input,
                    });
                    current_tool_id.clear();
                    current_tool_name.clear();
                    current_tool_input_json.clear();
                }
                StreamEvent::MessageEnd {
                    stop_reason: reason,
                } => {
                    stop_reason = reason;
                }
                StreamEvent::Usage(usage) => {
                    token_usage = usage;
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
            Ok(Err(AgshError::Interrupted)) => {
                // Interrupted — fall through to return partial content.
                // The caller detects interruption via the cancellation token.
            }
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

        Ok((message, stop_reason, token_usage))
    }

    async fn execute_tool_calls(
        &self,
        assistant_message: &Message,
        cancellation: CancellationToken,
        spacing: &mut render::OutputSpacing,
    ) -> Vec<ContentBlock> {
        // Pass 1 (serial): render indicators in source order and collect the
        // plan. Rendering must happen before any await so concurrent tool
        // execution can't interleave indicator lines on stderr.
        let mut planned: Vec<(String, String, serde_json::Value)> = Vec::new();
        for block in &assistant_message.content {
            if let ContentBlock::ToolUse { id, name, input } = block {
                if !self.options.streaming {
                    if spacing.before_tool_indicator() {
                        eprintln!();
                    }
                    let schema = self
                        .tool_registry
                        .get(name)
                        .map(|t| t.definition().parameters);
                    render::render_tool_indicator(name, input, schema.as_ref());
                }
                planned.push((id.clone(), name.clone(), input.clone()));
            }
        }

        // Pass 2 (concurrent): dispatch every tool in this assistant message
        // in parallel. `join_all` preserves the input ordering so the i-th
        // output corresponds to the i-th planned tool call.
        let futures = planned.iter().map(|(_, name, input)| {
            self.resolve_and_execute_tool(name.as_str(), input, cancellation.clone())
        });
        let outputs = futures::future::join_all(futures).await;

        // Pass 3 (serial): accumulate scratchpad hints, build ToolResult
        // blocks in source order, and run the spacing update once after all
        // tools have settled.
        let mut results = Vec::with_capacity(planned.len());
        let mut todo_write_fired = false;
        for ((id, name, _), output) in planned.into_iter().zip(outputs) {
            if name == "todo_write" && !self.todo_list.read().await.is_empty() {
                todo_write_fired = true;
            }
            if let Some(hint) = output.scratchpad_hint.clone() {
                self.scratchpad_hints.write().await.insert(id.clone(), hint);
            }
            results.push(ContentBlock::ToolResult {
                tool_use_id: id,
                content: output.content,
                is_error: output.is_error,
            });
        }
        if todo_write_fired {
            spacing.after_todo_list();
        }

        results
    }

    async fn resolve_and_execute_tool(
        &self,
        name: &str,
        input: &serde_json::Value,
        cancellation: CancellationToken,
    ) -> crate::tools::ToolOutput {
        // If the stream layer couldn't parse this tool call's JSON
        // arguments, it marked the input with a sentinel. Bail out with
        // an error so the model sees the parse failure instead of us
        // silently invoking the tool on a default-filled object.
        if let Some(reason) = input
            .get(crate::provider::INVALID_TOOL_ARGS_MARKER)
            .and_then(|v| v.as_str())
        {
            return crate::tools::ToolOutput::text(format!("Tool call rejected: {}", reason), true);
        }

        let Some(tool) = self.tool_registry.get(name) else {
            return crate::tools::ToolOutput::text(format!("Unknown tool: '{}'", name), true);
        };

        // Read the current permission once, at the enforcement site, so a
        // permission cycle via Shift+Tab during dispatch can't leave us
        // acting on a stale snapshot captured earlier in the loop.
        let permission = self.shared_permission.get();
        let required = self
            .tool_registry
            .required_permission_for(name)
            .unwrap_or_else(|| tool.required_permission());
        if !permission.allows(required) {
            return crate::tools::ToolOutput::text(
                format!(
                    "Permission denied: '{}' requires `{}` permission, current level is `{}`. \
                     Ask the user to run `/permission {}` (or press Shift+Tab) to enable it.",
                    name, required, permission, required
                ),
                true,
            );
        }

        if permission == crate::permission::Permission::Ask {
            return self
                .execute_with_approval(&*tool, name, input, cancellation)
                .await;
        }

        Self::run_tool(&*tool, input, cancellation).await
    }

    async fn execute_with_approval(
        &self,
        tool: &dyn crate::tools::Tool,
        name: &str,
        input: &serde_json::Value,
        cancellation: CancellationToken,
    ) -> crate::tools::ToolOutput {
        let Some(sender) = &self.approval_sender else {
            return crate::tools::ToolOutput::text(
                "Ask mode requires interactive shell for tool approval.".to_string(),
                true,
            );
        };

        let (response_sender, response_receiver) = tokio::sync::oneshot::channel::<bool>();
        let schema = tool.definition().parameters;
        let primary_param = crate::render::resolve_primary_param(name, input, Some(&schema));
        let request = crate::repl::ToolApprovalRequest {
            tool_name: name.to_string(),
            primary_param,
            response_sender,
        };

        if sender
            .send(crate::repl::AgentToReplEvent::ApprovalRequest(request))
            .is_err()
        {
            return crate::tools::ToolOutput::text(
                "Failed to request approval (shell disconnected)".to_string(),
                true,
            );
        }

        // Awaiting the oneshot receiver (rather than a blocking
        // `SyncReceiver::recv()`) keeps the executor thread free, so multiple
        // parallel tool calls in Ask mode each suspend cleanly while waiting
        // for the user's Y/n.
        match response_receiver.await {
            Ok(true) => Self::run_tool(tool, input, cancellation).await,
            Ok(false) => {
                crate::tools::ToolOutput::text("User denied tool execution.".to_string(), true)
            }
            Err(_) => crate::tools::ToolOutput::text(
                "Failed to receive approval response.".to_string(),
                true,
            ),
        }
    }

    async fn run_tool(
        tool: &dyn crate::tools::Tool,
        input: &serde_json::Value,
        cancellation: CancellationToken,
    ) -> crate::tools::ToolOutput {
        match tool.execute(input.clone(), cancellation).await {
            Ok(output) => output,
            Err(AgshError::Interrupted) => {
                crate::tools::ToolOutput::text("Tool execution interrupted.".to_string(), true)
            }
            Err(error) => crate::tools::ToolOutput::text(format!("Tool error: {}", error), true),
        }
    }

    pub async fn compact_session(
        &self,
        session_id: &mut Option<Uuid>,
        messages: &mut Conversation,
    ) -> Result<()> {
        let Some(sid) = *session_id else {
            return Err(AgshError::Config(
                "no active session to compact".to_string(),
            ));
        };

        if messages.is_empty() {
            return Err(AgshError::Config("no messages to compact".to_string()));
        }

        let system_prompt = "You are a conversation summarizer. Produce a structured summary \
             that will replace the conversation. Write in second person \
             (\"You were working on...\").\n\n\
             Cover these sections (skip any that don't apply):\n\n\
             1. **Primary task**: What the user asked for and the overall goal.\n\
             2. **Current state**: What has been completed, what is in progress, what remains.\n\
             3. **Key files**: Files read, created, or modified (list paths).\n\
             4. **Key decisions**: Important choices made and their rationale.\n\
             5. **Errors and fixes**: Problems encountered and how they were resolved.\n\
             6. **User preferences**: Feedback or corrections about how to work.";

        // Split into messages to summarize vs. recent messages to keep verbatim.
        // Walk backward from the target split point to find a safe cut that
        // doesn't orphan tool_use blocks from their tool_result responses.
        let view = messages.as_slice();
        let (to_summarize, to_keep) = if view.len() > 6 {
            let mut split = view.len() - 2;
            loop {
                if split == 0 {
                    break;
                }
                let message = &view[split];
                if message.role == Role::User && !has_tool_results(&message.content) {
                    break;
                }
                split -= 1;
            }
            if split >= 4 {
                (view[..split].to_vec(), view[split..].to_vec())
            } else {
                (view.to_vec(), Vec::new())
            }
        } else {
            (view.to_vec(), Vec::new())
        };

        // Clone and preprocess messages for the summarizer: strip images and
        // truncate large text blocks to avoid overwhelming the summary call.
        let mut compact_messages = to_summarize.clone();
        for message in &mut compact_messages {
            strip_images_and_truncate(&mut message.content);
        }

        // Append a user message so the conversation ends with a user turn.
        compact_messages.push(Message::user(
            "Summarize this conversation into a concise context message.",
        ));

        self.provider.set_thinking_override(Some(false));
        let compact_result = self
            .provider
            .complete(system_prompt, &compact_messages, &[])
            .await;
        self.provider.set_thinking_override(None);
        let (summary_message, _stop_reason, _usage) = compact_result?;

        let summary_text = summary_message.text_content();
        if summary_text.is_empty() {
            return Err(AgshError::Provider(
                "LLM returned an empty summary".to_string(),
            ));
        }

        // Build post-compact context: environment, todos, scratchpad inventory.
        let post_context = self.build_post_compact_context(sid).await;

        let context_message = if post_context.is_empty() {
            format!(
                "[Conversation summary from session compaction]\n\n{}",
                summary_text,
            )
        } else {
            format!(
                "[Conversation summary from session compaction]\n\n{}\n\n\
                 [Post-compaction context]\n\n{}",
                summary_text, post_context,
            )
        };

        // Snapshot the deferred-tool active set BEFORE compaction so the
        // `CompactBoundary` event carries it forward; otherwise tools the
        // model loaded pre-compaction would silently drop out of the
        // active set on the next turn.
        let loaded_tools_snapshot = crate::tools::extract_loaded_tool_names(messages.as_slice());

        let summary_user_message = Message::user(&context_message);
        messages.replace_for_compaction(
            summary_user_message,
            to_keep.clone(),
            loaded_tools_snapshot,
        );

        // Persist the new compaction-boundary event and the re-appended
        // tail. Pre-compaction rows stay in the DB unchanged; the
        // event log on disk grows append-only.
        let boundary_event = messages
            .events()
            .iter()
            .rev()
            .find(|e| matches!(e, crate::conversation::Event::CompactBoundary { .. }))
            .cloned()
            .ok_or_else(|| {
                AgshError::Internal(
                    "compact boundary missing after replace_for_compaction".to_string(),
                )
            })?;
        self.session_manager
            .save_event(sid, &boundary_event)
            .await?;
        for message in &to_keep {
            self.session_manager
                .save_event(sid, &crate::conversation::Event::Append(message.clone()))
                .await?;
        }

        Ok(())
    }

    async fn build_post_compact_context(&self, session_id: Uuid) -> String {
        let permission = self.shared_permission.get();
        let todos = self.todo_list.read().await.clone();
        let entries = self
            .session_manager
            .list_tool_outputs(session_id)
            .await
            .unwrap_or_default();
        context::build_post_compact_context(permission, &todos, &entries)
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

/// Preprocess message content blocks for the compaction summarizer:
/// replace images with "[image]" markers and truncate large text blocks.
fn strip_images_and_truncate(content: &mut [ContentBlock]) {
    use crate::provider::ToolResultContent;

    const MAX_TEXT_CHARS: usize = 2000;
    const HEAD_CHARS: usize = 1000;
    const TAIL_CHARS: usize = 500;

    for block in content.iter_mut() {
        match block {
            ContentBlock::ToolResult {
                content: tool_content,
                ..
            } => {
                for item in tool_content.iter_mut() {
                    match item {
                        ToolResultContent::Image { .. } => {
                            *item = ToolResultContent::Text {
                                text: "[image]".to_string(),
                            };
                        }
                        ToolResultContent::Text { text } => {
                            if text.len() > MAX_TEXT_CHARS {
                                let head_end = text.floor_char_boundary(HEAD_CHARS);
                                let tail_start =
                                    text.floor_char_boundary(text.len().saturating_sub(TAIL_CHARS));
                                *text = format!(
                                    "{}\n... (truncated for compaction) ...\n{}",
                                    &text[..head_end],
                                    &text[tail_start..],
                                );
                            }
                        }
                    }
                }
            }
            ContentBlock::Text { text } if text.len() > MAX_TEXT_CHARS => {
                let head_end = text.floor_char_boundary(HEAD_CHARS);
                let tail_start = text.floor_char_boundary(text.len().saturating_sub(TAIL_CHARS));
                *text = format!(
                    "{}\n... (truncated for compaction) ...\n{}",
                    &text[..head_end],
                    &text[tail_start..],
                );
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ToolResultContent;

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
                content: vec![ToolResultContent::Text {
                    text: "file contents".to_string(),
                }],
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

    // Cache prefix stability tests.
    // These tests simulate the agent's message-assembly logic (stable base +
    // appended tool-loop messages) to verify that the prefix sent to the API
    // remains identical across iterations of the tool-use loop.  This is the
    // core invariant required for KV cache reuse.

    fn assistant_tool_use_named(id: &str, name: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: id.to_string(),
                name: name.to_string(),
                input: serde_json::json!({"path": "/tmp/test"}),
            }],
        }
    }

    fn tool_result_for(tool_use_id: &str, content: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.to_string(),
                content: vec![ToolResultContent::Text {
                    text: content.to_string(),
                }],
                is_error: false,
            }],
        }
    }

    /// Compares two message slices for semantic equality (same role, same
    /// content blocks).  This is what determines whether the KV cache prefix
    /// is reusable.
    fn assert_messages_equal(a: &[Message], b: &[Message], context: &str) {
        assert_eq!(a.len(), b.len(), "{}: length mismatch", context);
        for (i, (ma, mb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                ma.role, mb.role,
                "{}: role mismatch at index {}",
                context, i
            );
            assert_eq!(
                ma.content.len(),
                mb.content.len(),
                "{}: content block count mismatch at index {}",
                context,
                i
            );
            let json_a = serde_json::to_string(&ma.content).unwrap();
            let json_b = serde_json::to_string(&mb.content).unwrap();
            assert_eq!(
                json_a, json_b,
                "{}: content mismatch at index {}",
                context, i
            );
        }
    }

    /// Simulates the tool-loop message assembly logic from `run_turn`:
    ///   base_messages = truncate(messages, limit)   // computed once
    ///   turn_start_len = messages.len()
    ///   loop { api_messages = base + messages[turn_start_len..] }
    fn build_api_messages(
        messages: &[Message],
        base_messages: &[Message],
        turn_start_len: usize,
    ) -> Vec<Message> {
        if messages.len() > turn_start_len {
            let mut combined = base_messages.to_vec();
            combined.extend_from_slice(&messages[turn_start_len..]);
            combined
        } else {
            base_messages.to_vec()
        }
    }

    #[test]
    fn test_stable_base_during_tool_loop() {
        // Simulate a conversation with history, then a tool loop that adds
        // 3 tool call/result pairs.  The base prefix (everything before the
        // tool loop) must be identical across all iterations.
        let mut messages = vec![
            user_msg("first question"),
            assistant_msg("first answer"),
            user_msg("second question"),
        ];

        let base_messages = truncate_messages_for_context(&messages, None);
        let turn_start_len = messages.len();

        let api_iter0 = build_api_messages(&messages, &base_messages, turn_start_len);
        assert_eq!(api_iter0.len(), 3);

        // Iteration 1: model calls a tool
        messages.push(assistant_tool_use_named("t1", "read_file"));
        messages.push(tool_result_for("t1", "file contents"));

        let api_iter1 = build_api_messages(&messages, &base_messages, turn_start_len);
        assert_eq!(api_iter1.len(), 5);

        // The first 3 messages (the base) must be identical.
        assert_messages_equal(&api_iter0[..3], &api_iter1[..3], "iter0→iter1 base");

        // Iteration 2: model calls another tool
        messages.push(assistant_tool_use_named("t2", "execute_command"));
        messages.push(tool_result_for("t2", "command output"));

        let api_iter2 = build_api_messages(&messages, &base_messages, turn_start_len);
        assert_eq!(api_iter2.len(), 7);

        // Base is still identical.
        assert_messages_equal(&api_iter0[..3], &api_iter2[..3], "iter0→iter2 base");
        // And the first 5 (base + iter1's additions) are identical too.
        assert_messages_equal(&api_iter1[..5], &api_iter2[..5], "iter1→iter2 prefix");

        // Iteration 3: yet another tool call
        messages.push(assistant_tool_use_named("t3", "read_file"));
        messages.push(tool_result_for("t3", "more contents"));

        let api_iter3 = build_api_messages(&messages, &base_messages, turn_start_len);
        assert_eq!(api_iter3.len(), 9);

        assert_messages_equal(&api_iter2[..7], &api_iter3[..7], "iter2→iter3 prefix");
        assert_messages_equal(&api_iter0[..3], &api_iter3[..3], "iter0→iter3 base");
    }

    #[test]
    fn test_truncation_boundary_does_not_shift_during_tool_loop() {
        // This is the critical test for the fix: when context_messages is set
        // and we're near the limit, adding tool results within the loop must
        // NOT cause the truncated prefix to shift.  Before the fix, truncation
        // was recomputed inside the loop, causing prefix instability.
        let limit = Some(6);

        // Start with 5 messages (under the limit of 6).
        let mut messages = vec![
            user_msg("msg-1"),
            assistant_msg("resp-1"),
            user_msg("msg-2"),
            assistant_msg("resp-2"),
            user_msg("msg-3"),
        ];

        // Compute the stable base ONCE before the loop (as run_turn does).
        let base_messages = truncate_messages_for_context(&messages, limit);
        let turn_start_len = messages.len();

        // All 5 messages fit within the limit; no truncation yet.
        assert_eq!(base_messages.len(), 5);

        let api_iter0 = build_api_messages(&messages, &base_messages, turn_start_len);
        assert_eq!(api_iter0.len(), 5);

        // Iteration 1: add tool call + result → 7 messages total, over limit.
        // With the old code, truncation would kick in and drop messages from
        // the front.  With the new code, the base is frozen.
        messages.push(assistant_tool_use_named("t1", "read_file"));
        messages.push(tool_result_for("t1", "data"));

        let api_iter1 = build_api_messages(&messages, &base_messages, turn_start_len);
        // Should be base(5) + new(2) = 7
        assert_eq!(api_iter1.len(), 7);

        // The first 5 messages must be identical to iter0.
        assert_messages_equal(&api_iter0[..5], &api_iter1[..5], "iter0→iter1 base");

        // Iteration 2: add another tool call → 9 total, well over limit.
        messages.push(assistant_tool_use_named("t2", "execute_command"));
        messages.push(tool_result_for("t2", "output"));

        let api_iter2 = build_api_messages(&messages, &base_messages, turn_start_len);
        assert_eq!(api_iter2.len(), 9);

        // The first 7 messages must match iter1 exactly.
        assert_messages_equal(&api_iter1[..7], &api_iter2[..7], "iter1→iter2 prefix");
        // And the base (first 5) is still untouched.
        assert_messages_equal(&api_iter0[..5], &api_iter2[..5], "iter0→iter2 base");
    }

    #[test]
    fn test_truncation_with_tool_chain_near_boundary() {
        // Verify that when the conversation includes a tool chain right at the
        // truncation boundary, the base is computed correctly and stays stable.
        let limit = Some(4);

        let mut messages = vec![
            user_msg("old-msg"),
            assistant_msg("old-resp"),
            user_msg("current question"),
            assistant_tool_use_named("t0", "read_file"),
            tool_result_for("t0", "initial data"),
            assistant_msg("here is the data"),
            user_msg("follow-up"),
        ];

        let base_messages = truncate_messages_for_context(&messages, limit);
        let turn_start_len = messages.len();

        // The truncation should keep a safe cut point; verify it starts with
        // a user message and doesn't split tool chains.
        assert_eq!(base_messages[0].role, Role::User);
        assert!(!has_tool_results(&base_messages[0].content));

        let api_iter0 = build_api_messages(&messages, &base_messages, turn_start_len);

        // Add tool loop messages
        messages.push(assistant_tool_use_named("t1", "read_file"));
        messages.push(tool_result_for("t1", "more data"));

        let api_iter1 = build_api_messages(&messages, &base_messages, turn_start_len);

        // The base portion must be identical.
        let base_len = base_messages.len();
        assert_messages_equal(
            &api_iter0[..base_len],
            &api_iter1[..base_len],
            "base stable after tool loop",
        );
    }

    #[test]
    fn test_no_limit_produces_full_prefix() {
        // With no context_messages limit, base_messages includes everything,
        // and tool loop additions are appended without any truncation.
        let mut messages = vec![user_msg("a"), assistant_msg("b"), user_msg("c")];

        let base_messages = truncate_messages_for_context(&messages, None);
        let turn_start_len = messages.len();

        assert_eq!(base_messages.len(), 3);

        let api_iter0 = build_api_messages(&messages, &base_messages, turn_start_len);
        assert_eq!(api_iter0.len(), 3);

        // Add many tool calls
        for i in 0..5 {
            messages.push(assistant_tool_use_named(&format!("t{}", i), "read_file"));
            messages.push(tool_result_for(
                &format!("t{}", i),
                &format!("result {}", i),
            ));
        }

        let api_final = build_api_messages(&messages, &base_messages, turn_start_len);
        assert_eq!(api_final.len(), 13); // 3 base + 10 tool messages

        // Base prefix still matches.
        assert_messages_equal(&api_iter0[..3], &api_final[..3], "full prefix stable");
    }

    #[test]
    fn test_multi_turn_with_truncation_each_turn_gets_stable_base() {
        // Simulate multiple turns, each computing its own stable base.
        // Verify that within each turn's tool loop the base stays fixed,
        // and that across turns the overlapping messages are consistent.
        let limit = Some(6);

        // -- Turn 1 --
        let mut messages: Vec<Message> = vec![user_msg("turn-1 question")];
        let base_t1 = truncate_messages_for_context(&messages, limit);
        let start_t1 = messages.len();

        // Tool loop: 2 iterations
        messages.push(assistant_tool_use_named("t1a", "read_file"));
        messages.push(tool_result_for("t1a", "data-a"));
        let api_t1_iter1 = build_api_messages(&messages, &base_t1, start_t1);

        messages.push(assistant_msg("here's your answer"));
        let api_t1_iter2 = build_api_messages(&messages, &base_t1, start_t1);

        // Base is stable within turn 1.
        assert_messages_equal(
            &api_t1_iter1[..base_t1.len()],
            &api_t1_iter2[..base_t1.len()],
            "turn 1 base stable",
        );

        // -- Turn 2 --
        messages.push(user_msg("turn-2 question"));

        let base_t2 = truncate_messages_for_context(&messages, limit);
        let start_t2 = messages.len();

        messages.push(assistant_tool_use_named("t2a", "execute_command"));
        messages.push(tool_result_for("t2a", "output"));
        let api_t2_iter1 = build_api_messages(&messages, &base_t2, start_t2);

        messages.push(assistant_tool_use_named("t2b", "read_file"));
        messages.push(tool_result_for("t2b", "more"));
        let api_t2_iter2 = build_api_messages(&messages, &base_t2, start_t2);

        // Base is stable within turn 2.
        assert_messages_equal(
            &api_t2_iter1[..base_t2.len()],
            &api_t2_iter2[..base_t2.len()],
            "turn 2 base stable",
        );

        // And the tool-loop prefix from iter1 is preserved in iter2.
        let shared = api_t2_iter1.len();
        assert_messages_equal(
            &api_t2_iter1[..shared],
            &api_t2_iter2[..shared],
            "turn 2 iter1→iter2 prefix",
        );
    }
}
