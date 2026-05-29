//! `meka acp` subcommand. Speaks the Agent Client Protocol (ACP) on stdio so editor / web /
//! messenger clients can drive a meka turn end to end.
//!
//! # Capability surface
//!
//! - **Lifecycle**: `initialize` (with `clientInfo` capture + `agentInfo` advertisement),
//!   `session/new`, `session/load` (replays the persisted conversation as `session/update`
//!   notifications), `session/list` (cwd-filtered, cursor-paginated; sub-agent sessions hidden),
//!   `session/resume`, `session/close`, `session/cancel`, `session/set_mode`.
//! - **Turn**: `session/prompt` streams `agent_message_chunk`, `agent_thought_chunk`, the full
//!   `tool_call` + `tool_call_update` lifecycle (with diff content blocks from
//!   [`crate::frontend::ToolOutputMetadata::Diff`]), and ends with `end_turn` / `cancelled` stop
//!   reasons. `session/request_permission` handles `ask`-mode tool approvals; per-session sticky
//!   always/never sets short-circuit subsequent requests.
//! - **Skills + modes**: installed skills surface as `available_commands_update` palette entries;
//!   the agent resolves `/<skill-name> [extra]` prompts to the rendered skill body before the turn.
//!   `Permission` levels map 1:1 to ACP `SessionMode` ids, advertised on every session-creation
//!   response and mutated live via `session/set_mode`.
//! - **Delegation**: `read_file` / `write_file` / `edit_file` / `execute_command` route through the
//!   client's `fs/read_text_file`, `fs/write_text_file`, and `terminal/*` when the matching
//!   capability is offered, falling back to local syscalls otherwise. `read` permission mode
//!   bypasses `execute_command` delegation so the local Landlock / bwrap / sandbox-exec /
//!   Low-Integrity jail stays in place.
//!
//! Multi-session: any number of sessions can coexist in one `meka acp` process. Each session has
//! its own cwd, permission cell, conversation, cancellation token, and per-session `Agent` +
//! `AcpFrontend`. Sessions share process-wide dependencies (provider, MCP manager, session DB,
//! skill cache) via `Arc`. Two sessions can run `session/prompt` calls in parallel; there is no
//! global mutex serialising turns. Sub-agents reach the parent's client through
//! [`crate::frontend::PermissionForwardingFrontend`], so their permission prompts, fs delegates,
//! and terminal delegates all flow through the parent session's editor UI.

use std::{path::PathBuf, sync::Arc};

use agent_client_protocol::{
    Agent as AcpAgentRole, Client, ConnectionTo, Stdio,
    schema::{
        AgentCapabilities, AvailableCommand, AvailableCommandInput, AvailableCommandsUpdate,
        CancelNotification, ClientCapabilities, CloseSessionRequest, CloseSessionResponse,
        ContentBlock, ContentChunk, CreateTerminalRequest, CurrentModeUpdate, Diff, EnvVariable,
        Implementation, InitializeRequest, InitializeResponse, KillTerminalRequest,
        ListSessionsRequest, ListSessionsResponse, LoadSessionRequest, LoadSessionResponse,
        NewSessionRequest, NewSessionResponse, PermissionOption, PermissionOptionKind,
        PromptCapabilities, PromptRequest, PromptResponse, ReadTextFileRequest,
        ReleaseTerminalRequest, RequestPermissionOutcome, RequestPermissionRequest,
        ResumeSessionRequest, ResumeSessionResponse, SessionCapabilities, SessionCloseCapabilities,
        SessionId, SessionInfo, SessionListCapabilities, SessionMode, SessionModeId,
        SessionModeState, SessionNotification, SessionResumeCapabilities, SessionUpdate,
        SetSessionModeRequest, SetSessionModeResponse, StopReason, TerminalOutputRequest, ToolCall,
        ToolCallContent, ToolCallLocation, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields,
        ToolKind, UnstructuredCommandInput, WaitForTerminalExitRequest, WriteTextFileRequest,
    },
};
use async_trait::async_trait;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::{
    agent::{Agent, SharedCwd, resolve_against_cwd},
    config::ResolvedConfig,
    conversation::Conversation,
    error::MekaError,
    frontend::{
        DelegatedExecOutput, DelegatedExecSpec, Frontend, FrontendError, FrontendEvent,
        PermissionOutcome, PermissionRequest, ToolOutputMetadata,
    },
    mcp,
    permission::{Permission, SharedPermission},
    provider::{AuthCredential, ContentBlock as MekaContentBlock, Role, ToolResultContent},
    session::SessionManager,
    skills::SkillCache,
};

/// Build a JSON-RPC `InvalidParams` error (`-32602`) with a free-form human-readable message in the
/// `data` field. Mirrors [`agent_client_protocol::util::internal_error`] but for the
/// input-validation cases (unknown sessionId, malformed UUID, unsupported mode, non-text content).
/// Clients can rely on the JSON-RPC code to distinguish "bad input" from "server failure".
fn invalid_params_error(message: impl ToString) -> agent_client_protocol::Error {
    agent_client_protocol::Error::invalid_params().data(message.to_string())
}

/// Late-bound view of everything the connected client told us on `initialize`: its advertised
/// capabilities and its self-identifying `Implementation` (name + version). Default is the
/// all-`false` `ClientCapabilities` and a `None` identity, so an `AcpFrontend` constructed before
/// `initialize` arrives correctly reports "delegation unavailable" and "client unknown" until the
/// handler fills it in.
#[derive(Clone, Default)]
pub struct SharedClientState {
    inner: Arc<std::sync::RwLock<ClientStateInner>>,
}

#[derive(Clone, Default)]
struct ClientStateInner {
    capabilities: ClientCapabilities,
    /// Logged once on `initialize`. Read only in tests today; the `#[allow(dead_code)]` stays
    /// until a production reader (e.g. surfacing the client name in response `_meta`) lands.
    #[allow(dead_code)]
    info: Option<Implementation>,
}

impl SharedClientState {
    /// Record both halves of the client-side initialize payload in one write. Called exactly once
    /// per process today (the `initialize` handler), but tolerant of re-initialisation if a future
    /// client ever resends.
    fn record_initialize(&self, capabilities: ClientCapabilities, info: Option<Implementation>) {
        let mut guard = self
            .inner
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *guard = ClientStateInner { capabilities, info };
    }

    fn capabilities(&self) -> ClientCapabilities {
        self.inner
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .capabilities
            .clone()
    }

    #[cfg(test)]
    fn client_info(&self) -> Option<Implementation> {
        self.inner
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .info
            .clone()
    }
}

/// Render an `Implementation` as a `"name version"` pair for the `initialize` log line. `None`
/// renders as `"<unknown> <unknown>"` so the log shape is stable across clients that omit
/// `client_info` entirely.
fn describe_client(info: Option<&Implementation>) -> String {
    match info {
        Some(implementation) => format!("{} {}", implementation.name, implementation.version),
        None => "<unknown> <unknown>".to_string(),
    }
}

/// ACP-side [`Frontend`] impl. Converts the agent loop's streamed events into ACP `session/update`
/// notifications and runs the `session/request_permission` round-trip for tool approvals.
/// Constructed per-session: every field is fully populated at build time, so there's no "not yet
/// bound" `Option` state to handle.
pub struct AcpFrontend {
    connection: ConnectionTo<Client>,
    session_id: SessionId,
    cwd: SharedCwd,
    /// Sticky `allow_always` set; symmetric `never_allowed` below for `reject_always`. Both
    /// short-circuit `request_permission` so the user isn't re-prompted for the same tool in this
    /// session. Per-session (one `AcpFrontend` per session); not persisted.
    always_allowed: std::sync::Mutex<std::collections::HashSet<String>>,
    never_allowed: std::sync::Mutex<std::collections::HashSet<String>>,
    client_state: SharedClientState,
    /// Stdio-level "transport is dead" latch, shared across every per-session `AcpFrontend` in the
    /// process. When `send_notification` fails on any session, we set the latch so every other
    /// session's agent loop short-circuits on its next iteration instead of burning provider
    /// tokens until its own emit also fails.
    ///
    /// This is correct *for stdio*: one closed pipe affects every session in the process, so the
    /// global signal carries no false positives. When a per-session transport (e.g. WebSocket-ACP
    /// or a TCP-multiplexed successor) lands, this field needs a per-session sibling (read both
    /// in `client_disconnected()` and OR them) so a single session's drop doesn't take the
    /// process down with it. Grep for `transport_dead` to find the migration points.
    transport_dead: Arc<std::sync::atomic::AtomicBool>,
}

impl AcpFrontend {
    fn new(
        connection: ConnectionTo<Client>,
        session_id: SessionId,
        cwd: SharedCwd,
        client_state: SharedClientState,
        transport_dead: Arc<std::sync::atomic::AtomicBool>,
    ) -> Self {
        Self {
            connection,
            session_id,
            cwd,
            always_allowed: std::sync::Mutex::new(std::collections::HashSet::new()),
            never_allowed: std::sync::Mutex::new(std::collections::HashSet::new()),
            client_state,
            transport_dead,
        }
    }

    /// Mark the stdio transport as dead. Called from `emit` and the `session/load` replay loop
    /// whenever `send_notification` reports an error. Idempotent. The trait-level
    /// `client_disconnected()` read below surfaces the same flag back to the agent loop.
    fn mark_transport_dead(&self) {
        self.transport_dead
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn is_always_allowed(&self, tool_name: &str) -> bool {
        self.always_allowed
            .lock()
            .map(|guard| guard.contains(tool_name))
            .unwrap_or(false)
    }

    fn is_never_allowed(&self, tool_name: &str) -> bool {
        self.never_allowed
            .lock()
            .map(|guard| guard.contains(tool_name))
            .unwrap_or(false)
    }

    fn remember_allow(&self, tool_name: &str) {
        if let Ok(mut guard) = self.always_allowed.lock() {
            guard.insert(tool_name.to_string());
        }
    }

    fn remember_deny(&self, tool_name: &str) {
        if let Ok(mut guard) = self.never_allowed.lock() {
            guard.insert(tool_name.to_string());
        }
    }
}

#[async_trait]
impl Frontend for AcpFrontend {
    async fn emit(&self, event: FrontendEvent) {
        let connection = self.connection.clone();
        let session_id = self.session_id.clone();

        let update = match event {
            FrontendEvent::AssistantTextDelta(text) => {
                SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                    agent_client_protocol::schema::TextContent::new(text),
                )))
            }
            FrontendEvent::ThinkingBlock { content, .. } => {
                SessionUpdate::AgentThoughtChunk(ContentChunk::new(ContentBlock::Text(
                    agent_client_protocol::schema::TextContent::new(content),
                )))
            }
            FrontendEvent::ToolCallStarted {
                id,
                name,
                input,
                // `display_summary` is REPL-facing chrome; ACP clients render their own tool-call
                // title from `raw_input` and `locations`.
                display_summary: _,
            } => {
                // No separate `pending` state in the agent loop, so the in-progress emit is the
                // first one the client sees.
                let locations = tool_locations(&name, &input, &self.cwd);
                let call = ToolCall::new(id, name.clone())
                    .kind(tool_kind_for(&name))
                    .status(ToolCallStatus::InProgress)
                    .locations(locations)
                    .raw_input(input);
                SessionUpdate::ToolCall(call)
            }
            FrontendEvent::ToolCallCompleted {
                id,
                is_error,
                content,
                metadata,
            } => {
                let status = if is_error {
                    ToolCallStatus::Failed
                } else {
                    ToolCallStatus::Completed
                };
                let acp_content = build_completion_content(&content, metadata);
                let fields = ToolCallUpdateFields::new()
                    .status(status)
                    .content(acp_content);
                SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(id, fields))
            }
            FrontendEvent::Notice(notice) => {
                // No dedicated ACP primitive for advisories; surface inline as an assistant-message
                // chunk with an `[meka]` prefix so editor transcripts record the side-effect and
                // clients can filter or style by that prefix. `notice.level` is unused on the wire
                // today; when ACP grows a typed notice variant, branch on it here.
                let text = format!("[meka] {}", notice.text);
                SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                    agent_client_protocol::schema::TextContent::new(text),
                )))
            }
            FrontendEvent::McpProgress(update) => {
                // ACP has no protocol primitive for tool-progress streams. The REPL renders these
                // inline as a carriage-return-overwrite status line; in the ACP world the editor
                // already has its own visibility into MCP server activity (or can subscribe to the
                // stderr log stream of the spawned agent). Log at info so `-v` users can still see
                // them; don't pollute the assistant-message transcript with per-tick status text.
                tracing::info!(
                    "MCP '{}' {} progress: {}{}{}",
                    update.server_name,
                    update.tool_name,
                    update.progress,
                    update.total.map(|t| format!("/{}", t)).unwrap_or_default(),
                    update
                        .message
                        .as_deref()
                        .map(|m| format!(", {}", m))
                        .unwrap_or_default()
                );
                return;
            }
            // REPL-specific signage (todos, token usage, lifecycle).
            _ => return,
        };

        if let Err(error) =
            connection.send_notification(SessionNotification::new(session_id, update))
        {
            tracing::debug!("AcpFrontend send_notification failed: {}", error);
            self.mark_transport_dead();
        }
    }

    fn client_disconnected(&self) -> bool {
        self.transport_dead
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome {
        // Honor sticky decisions from earlier `*_always` selections.
        if self.is_always_allowed(&request.tool_name) {
            return PermissionOutcome::Allow;
        }
        if self.is_never_allowed(&request.tool_name) {
            return PermissionOutcome::Deny;
        }

        let connection = self.connection.clone();
        let session_id = self.session_id.clone();

        let options = vec![
            PermissionOption::new(OPTION_ALLOW_ONCE, "Allow", PermissionOptionKind::AllowOnce),
            PermissionOption::new(
                OPTION_ALLOW_ALWAYS,
                "Always allow",
                PermissionOptionKind::AllowAlways,
            ),
            PermissionOption::new(OPTION_REJECT_ONCE, "Deny", PermissionOptionKind::RejectOnce),
            PermissionOption::new(
                OPTION_REJECT_ALWAYS,
                "Always deny",
                PermissionOptionKind::RejectAlways,
            ),
        ];

        // Synthetic id: the permission round-trip is its own space, not correlated with the
        // streaming tool_call lifecycle.
        let tool_call_id = format!("perm-{}", uuid::Uuid::new_v4());
        let title = match &request.primary_param {
            Some(param) if !param.is_empty() => format!("{} {}", request.tool_name, param),
            _ => request.tool_name.clone(),
        };
        let fields = ToolCallUpdateFields::new()
            .kind(tool_kind_for(&request.tool_name))
            .title(title)
            .status(ToolCallStatus::Pending);
        let tool_call = ToolCallUpdate::new(tool_call_id, fields);

        let req = RequestPermissionRequest::new(session_id, tool_call, options);
        // Race the round-trip against the per-turn cancellation token. If `session/cancel` fires
        // while we're waiting for the client to answer the permission prompt, we resolve as
        // `Cancelled` instead of holding the runtime mutex forever, which would block
        // `session/close` and `session/set_mode` too.
        let response = tokio::select! {
            biased;
            _ = request.cancellation.cancelled() => {
                return PermissionOutcome::Cancelled;
            }
            result = connection.send_request(req).block_task() => match result {
                Ok(resp) => resp,
                Err(error) => {
                    tracing::debug!("request_permission send_request failed: {}", error);
                    // Spec-conformant clients always reply with a `Selected` or `Cancelled`
                    // outcome, so an `Err` here is almost certainly transport-level. Mark the
                    // connection dropped so the agent loop short-circuits on the next pre-iteration
                    // check instead of running a tool, emitting a denied result, and only then
                    // discovering the client is gone via the next emit. The FS / execute delegates
                    // intentionally don't do this: those paths legitimately receive JSON-RPC error
                    // responses (e.g. terminal/create denied), which would produce false-positive
                    // disconnects.
                    self.mark_transport_dead();
                    return PermissionOutcome::Deny;
                }
            },
        };

        translate_permission_outcome(
            response.outcome,
            &request.tool_name,
            |sticky| match sticky {
                StickyDecision::AllowAlways => self.remember_allow(&request.tool_name),
                StickyDecision::RejectAlways => self.remember_deny(&request.tool_name),
            },
        )
    }

    async fn delegate_fs_read(
        &self,
        path: &std::path::Path,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> Option<Result<String, FrontendError>> {
        let caps = self.client_state.capabilities();
        if !caps.fs.read_text_file {
            return None;
        }
        let connection = self.connection.clone();
        let session_id = self.session_id.clone();
        let mut request = ReadTextFileRequest::new(session_id, path.to_path_buf());
        if let Some(line) = line {
            request = request.line(line);
        }
        if let Some(limit) = limit {
            request = request.limit(limit);
        }
        Some(match connection.send_request(request).block_task().await {
            Ok(response) => Ok(response.content),
            Err(error) => Err(FrontendError::new(format!(
                "fs/read_text_file failed: {}",
                error
            ))),
        })
    }

    async fn delegate_fs_write(
        &self,
        path: &std::path::Path,
        content: &str,
    ) -> Option<Result<(), FrontendError>> {
        let caps = self.client_state.capabilities();
        if !caps.fs.write_text_file {
            return None;
        }
        let connection = self.connection.clone();
        let session_id = self.session_id.clone();
        let request =
            WriteTextFileRequest::new(session_id, path.to_path_buf(), content.to_string());
        Some(match connection.send_request(request).block_task().await {
            Ok(_) => Ok(()),
            Err(error) => Err(FrontendError::new(format!(
                "fs/write_text_file failed: {}",
                error
            ))),
        })
    }

    async fn delegate_execute(
        &self,
        spec: DelegatedExecSpec,
    ) -> Option<Result<DelegatedExecOutput, FrontendError>> {
        let caps = self.client_state.capabilities();
        if !caps.terminal {
            return None;
        }
        let connection = self.connection.clone();
        let session_id = self.session_id.clone();
        Some(run_delegated_execute(connection, session_id, spec).await)
    }

    async fn handle_elicitation(
        &self,
        prompt: crate::mcp::elicitation::ElicitationPrompt,
    ) -> crate::mcp::elicitation::ElicitationResponse {
        // ACP has no protocol primitive for arbitrary server forms. The pragmatic stance until one
        // lands is to auto-decline with an info-level log so editor users can see in their agent
        // stderr that an elicitation arrived and was passed back to the server. A future per-ACP
        // path could synthesize a `session/request_permission` round-trip (the only existing
        // round-trip primitive) by mapping form fields to permission options, but that conflates
        // tool approval and form input, which is the kind of overload the protocol is likely to
        // rule out as it grows a proper elicitation surface.
        tracing::warn!(
            "ACP session received MCP elicitation from '{}' ({}); auto-declining (no ACP \
             primitive for form/URL prompts yet): {}",
            prompt.server_name,
            match &prompt.kind {
                crate::mcp::elicitation::ElicitationKind::Form { .. } => "form",
                crate::mcp::elicitation::ElicitationKind::Url { .. } => "url",
            },
            prompt.message,
        );
        crate::mcp::elicitation::ElicitationResponse::Decline
    }
}

/// Stable string IDs for the four permission options. The agent and the client must agree on these;
/// picking them as `const`s keeps the match arm in [`translate_permission_outcome`] honest.
const OPTION_ALLOW_ONCE: &str = "allow_once";
const OPTION_ALLOW_ALWAYS: &str = "allow_always";
const OPTION_REJECT_ONCE: &str = "reject_once";
const OPTION_REJECT_ALWAYS: &str = "reject_always";

/// Indicates which sticky bucket the user just opted into, so the caller can update its set.
/// Internal to the permission flow.
enum StickyDecision {
    AllowAlways,
    RejectAlways,
}

/// Map an ACP outcome to meka's [`PermissionOutcome`] and fire `record_sticky` when the user picked
/// one of the `*_always` options. Pure function so it's easy to unit-test.
fn translate_permission_outcome<F>(
    outcome: RequestPermissionOutcome,
    tool_name: &str,
    mut record_sticky: F,
) -> PermissionOutcome
where
    F: FnMut(StickyDecision),
{
    match outcome {
        RequestPermissionOutcome::Cancelled => PermissionOutcome::Cancelled,
        RequestPermissionOutcome::Selected(selected) => {
            let option_id: &str = selected.option_id.0.as_ref();
            match option_id {
                OPTION_ALLOW_ONCE => PermissionOutcome::Allow,
                OPTION_ALLOW_ALWAYS => {
                    record_sticky(StickyDecision::AllowAlways);
                    PermissionOutcome::Allow
                }
                OPTION_REJECT_ONCE => PermissionOutcome::Deny,
                OPTION_REJECT_ALWAYS => {
                    record_sticky(StickyDecision::RejectAlways);
                    PermissionOutcome::Deny
                }
                other => {
                    tracing::debug!(
                        "request_permission for '{}' returned unknown option_id '{}'; \
                         defaulting to Deny",
                        tool_name,
                        other,
                    );
                    PermissionOutcome::Deny
                }
            }
        }
        // ACP's `RequestPermissionOutcome` is `#[non_exhaustive]`; any future variant we haven't
        // taught the agent about should fail closed.
        other => {
            tracing::debug!(
                "request_permission for '{}' returned unknown outcome {:?}; \
                 defaulting to Deny",
                tool_name,
                other,
            );
            PermissionOutcome::Deny
        }
    }
}

/// Map meka's tool name to ACP's [`ToolKind`] so clients can pick the right icon and grouping.
/// MCP-loaded tools (named `mcp__server__tool`) and anything unknown fall through to `Other`.
fn tool_kind_for(name: &str) -> ToolKind {
    match name {
        "read_file" | "todo" => ToolKind::Read,
        "edit_file" | "write_file" => ToolKind::Edit,
        "find_files" | "search_contents" => ToolKind::Search,
        "execute_command" => ToolKind::Execute,
        "fetch_url" | "web_search" => ToolKind::Fetch,
        "spawn_agent" => ToolKind::Think,
        // skill, scratchpad_*, render_image, load_tool, mcp__*, and any future
        // built-ins.
        _ => ToolKind::Other,
    }
}

/// Compute the `locations` entries for a tool call. For tools whose primary argument is a path,
/// resolve it against the agent's per-session cwd (ACP requires absolute paths). Anything else
/// returns an empty list; clients fall back to the `raw_input` field.
fn tool_locations(name: &str, input: &serde_json::Value, cwd: &SharedCwd) -> Vec<ToolCallLocation> {
    let raw = match name {
        "read_file" | "edit_file" | "write_file" | "find_files" | "search_contents" => {
            input.get("path").and_then(|v| v.as_str())
        }
        _ => None,
    };
    raw.map(|path| vec![ToolCallLocation::new(resolve_against_cwd(cwd, path))])
        .unwrap_or_default()
}

/// Build the `content` array of a `tool_call_update` from meka's tool output. A populated `Diff`
/// metadata wins over the raw text content so clients (Zed) get the structured diff for apply-UI;
/// if no metadata is set, text/image content blocks pass through.
fn build_completion_content(
    content: &[ToolResultContent],
    metadata: Option<ToolOutputMetadata>,
) -> Vec<ToolCallContent> {
    if let Some(ToolOutputMetadata::Diff {
        path,
        old_text,
        new_text,
    }) = metadata
    {
        let mut diff = Diff::new(path, new_text);
        if let Some(old) = old_text {
            diff = diff.old_text(old);
        }
        return vec![ToolCallContent::Diff(diff)];
    }

    content
        .iter()
        .map(|block| {
            let text = match block {
                ToolResultContent::Text { text } => text.clone(),
                // No ACP analogue for image content yet; collapse to a text marker so the wire
                // stays valid.
                ToolResultContent::Image { .. } => "[image]".to_string(),
            };
            ToolCallContent::from(ContentBlock::Text(
                agent_client_protocol::schema::TextContent::new(text),
            ))
        })
        .collect()
}

/// Walk a hydrated [`Conversation`] and emit one `session/update` notification per content
/// block, mirroring the streaming shape the client would have seen had it been connected during
/// the original turn. Used by `session/load` so an editor that just reopened a session replays the
/// full history into its UI.
///
/// `<context>...</context>` preambles meka prepends to each user message are stripped before emit
/// so the client sees only what the user actually typed.
///
/// Tool calls track open `tool_use_id`s; any tool that never received a matching `ToolResult` (e.g.
/// a crashed turn) is closed out with a `failed` `tool_call_update` so the client doesn't render a
/// stuck spinner.
fn replay_session_updates(
    connection: &ConnectionTo<Client>,
    session_id: &SessionId,
    cwd: &SharedCwd,
    messages: &Conversation,
) {
    let mut open_tools: std::collections::HashSet<String> = std::collections::HashSet::new();
    for message in messages.as_slice() {
        match message.role {
            Role::User => {
                for block in &message.content {
                    match block {
                        MekaContentBlock::Text { text } => {
                            let stripped = crate::session::strip_context_tags(text);
                            if !stripped.is_empty() {
                                send_session_update(
                                    connection,
                                    session_id,
                                    SessionUpdate::UserMessageChunk(ContentChunk::new(
                                        ContentBlock::Text(
                                            agent_client_protocol::schema::TextContent::new(
                                                stripped.to_string(),
                                            ),
                                        ),
                                    )),
                                );
                            }
                        }
                        MekaContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            let status = if *is_error {
                                ToolCallStatus::Failed
                            } else {
                                ToolCallStatus::Completed
                            };
                            let acp_content = build_completion_content(content, None);
                            let fields = ToolCallUpdateFields::new()
                                .status(status)
                                .content(acp_content);
                            send_session_update(
                                connection,
                                session_id,
                                SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                                    tool_use_id.clone(),
                                    fields,
                                )),
                            );
                            open_tools.remove(tool_use_id);
                        }
                        _ => {}
                    }
                }
            }
            Role::Assistant => {
                for block in &message.content {
                    match block {
                        MekaContentBlock::Text { text } => {
                            send_session_update(
                                connection,
                                session_id,
                                SessionUpdate::AgentMessageChunk(ContentChunk::new(
                                    ContentBlock::Text(
                                        agent_client_protocol::schema::TextContent::new(
                                            text.clone(),
                                        ),
                                    ),
                                )),
                            );
                        }
                        MekaContentBlock::Thinking { thinking, .. } => {
                            send_session_update(
                                connection,
                                session_id,
                                SessionUpdate::AgentThoughtChunk(ContentChunk::new(
                                    ContentBlock::Text(
                                        agent_client_protocol::schema::TextContent::new(
                                            thinking.clone(),
                                        ),
                                    ),
                                )),
                            );
                        }
                        MekaContentBlock::ToolUse { id, name, input } => {
                            let locations = tool_locations(name, input, cwd);
                            let call = ToolCall::new(id.clone(), name.clone())
                                .kind(tool_kind_for(name))
                                .status(ToolCallStatus::InProgress)
                                .locations(locations)
                                .raw_input(input.clone());
                            send_session_update(
                                connection,
                                session_id,
                                SessionUpdate::ToolCall(call),
                            );
                            open_tools.insert(id.clone());
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Tool calls without a matching result: close them as failed so the client's "tool running"
    // indicator doesn't get stuck.
    for orphan_id in open_tools {
        let fields = ToolCallUpdateFields::new().status(ToolCallStatus::Failed);
        send_session_update(
            connection,
            session_id,
            SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(orphan_id, fields)),
        );
    }
}

fn send_session_update(
    connection: &ConnectionTo<Client>,
    session_id: &SessionId,
    update: SessionUpdate,
) {
    if let Err(error) =
        connection.send_notification(SessionNotification::new(session_id.clone(), update))
    {
        tracing::debug!("session/load replay send_notification failed: {}", error);
    }
}

/// Drive the ACP `terminal/*` four-call dance for a delegated execute:
/// `terminal/create` → wait for exit (raced against the agent's
/// cancellation token + the spec's timeout) → `terminal/output` →
/// `terminal/release`. On cancel or timeout we send `terminal/kill`
/// first and then still read whatever output the editor buffered.
async fn run_delegated_execute(
    connection: ConnectionTo<Client>,
    session_id: SessionId,
    spec: DelegatedExecSpec,
) -> Result<DelegatedExecOutput, FrontendError> {
    // Build CreateTerminalRequest. Empty `args` / `env` / unset `cwd` / unset `output_byte_limit`
    // are all fine; the builder leaves them at defaults.
    let mut create = CreateTerminalRequest::new(session_id.clone(), spec.command.clone());
    if !spec.args.is_empty() {
        create = create.args(spec.args.clone());
    }
    if !spec.env.is_empty() {
        let env_vars: Vec<EnvVariable> = spec
            .env
            .iter()
            .map(|(name, value)| EnvVariable::new(name.clone(), value.clone()))
            .collect();
        create = create.env(env_vars);
    }
    if let Some(cwd) = spec.cwd.clone() {
        create = create.cwd(cwd);
    }
    if let Some(limit) = spec.output_byte_limit {
        create = create.output_byte_limit(limit);
    }

    let terminal_id = match connection.send_request(create).block_task().await {
        Ok(response) => response.terminal_id,
        Err(error) => {
            return Err(FrontendError::new(format!(
                "terminal/create failed: {}",
                error
            )));
        }
    };

    // Wait for exit, racing the agent's cancellation token + the spec's timeout. On race-loss we
    // kill the terminal first; the follow-up `terminal/output` still returns whatever was buffered.
    //
    // Default cap of 15 minutes when the caller didn't supply one; interactive tools can override
    // per-call. The agent's cancel token is still the primary escape hatch; the timeout is just the
    // worst-case bound if both the cancel path and the underlying process get wedged.
    let timeout = spec
        .timeout
        .unwrap_or_else(|| std::time::Duration::from_secs(60 * 15));
    let wait_request = WaitForTerminalExitRequest::new(session_id.clone(), terminal_id.clone());
    let killed = tokio::select! {
        result = connection.send_request(wait_request).block_task() => {
            match result {
                Ok(_response) => false,
                Err(error) => {
                    return Err(FrontendError::new(format!(
                        "terminal/wait_for_exit failed: {}",
                        error
                    )));
                }
            }
        }
        _ = spec.cancellation.cancelled() => true,
        _ = tokio::time::sleep(timeout) => true,
    };
    if killed
        && let Err(error) = connection
            .send_request(KillTerminalRequest::new(
                session_id.clone(),
                terminal_id.clone(),
            ))
            .block_task()
            .await
    {
        tracing::debug!("terminal/kill failed: {}", error);
    }

    // Fetch the final output regardless of which arm won.
    let output_request = TerminalOutputRequest::new(session_id.clone(), terminal_id.clone());
    let output_response = match connection.send_request(output_request).block_task().await {
        Ok(response) => response,
        Err(error) => {
            // Try to release before bubbling the error so we don't leak the terminal handle on the
            // client side.
            if let Err(release_error) = connection
                .send_request(ReleaseTerminalRequest::new(
                    session_id.clone(),
                    terminal_id.clone(),
                ))
                .block_task()
                .await
            {
                tracing::debug!(
                    "terminal/release after output failure also failed: {}",
                    release_error,
                );
            }
            return Err(FrontendError::new(format!(
                "terminal/output failed: {}",
                error
            )));
        }
    };

    // Best-effort release; errors are non-fatal (the editor cleans up on disconnect anyway). Log at
    // debug for diagnostics.
    if let Err(error) = connection
        .send_request(ReleaseTerminalRequest::new(session_id, terminal_id))
        .block_task()
        .await
    {
        tracing::debug!("terminal/release failed: {}", error);
    }

    let (exit_code, signal) = match output_response.exit_status {
        Some(status) => (
            // ACP wire protocol uses u32 exit codes; meka's `DelegatedExecOutput.exit_code` is
            // i32. Real exit codes are 0-255, so `try_from` always succeeds; the
            // explicit fallback to -1 documents the choice instead of doing a lossy
            // `as`-cast.
            status
                .exit_code
                .map(|code| i32::try_from(code).unwrap_or(-1)),
            status.signal.clone(),
        ),
        None if killed => (None, Some("SIGTERM".to_string())),
        None => (None, None),
    };

    Ok(DelegatedExecOutput {
        output: output_response.output,
        exit_code,
        signal,
        truncated: output_response.truncated,
    })
}

/// Map a meka [`Permission`] to its ACP [`SessionModeId`] string. The mapping is the lowercase
/// debug name (`none` / `read` / `ask` / `write`), the same string `Permission::Display` produces.
/// It is kept as a dedicated function so the inverse parser ([`parse_mode_id`]) reads as the
/// obvious complement.
fn mode_id_for(permission: Permission) -> SessionModeId {
    SessionModeId::from(permission.to_string())
}

/// Parse a `SessionModeId` (treated as a `&str`) into the matching `Permission`. Returns `None` for
/// any unrecognised mode id. The match arms must stay in lock-step with [`mode_id_for`].
fn parse_mode_id(id: &str) -> Option<Permission> {
    match id {
        "none" => Some(Permission::None),
        "read" => Some(Permission::Read),
        "ask" => Some(Permission::Ask),
        "write" => Some(Permission::Write),
        _ => None,
    }
}

/// Human-readable label for a permission mode, shown in editor mode pickers next to each option.
/// Kept in lock-step with the REPL's `/permission` output and the `[permissions]` keys in
/// `config.toml` so users see the same vocabulary everywhere.
fn mode_display_name(permission: Permission) -> &'static str {
    match permission {
        Permission::None => "None",
        Permission::Read => "Read",
        Permission::Ask => "Ask",
        Permission::Write => "Write",
    }
}

/// One-line description of what a permission mode lets the agent do. Shown beneath the mode label
/// in editor pickers.
fn mode_description(permission: Permission) -> &'static str {
    match permission {
        Permission::None => "No tools available.",
        Permission::Read => "File reads and searches only. No writes, no shell.",
        Permission::Ask => "Every write or shell command requires approval.",
        Permission::Write => "All tools allowed without per-call approval.",
    }
}

/// Build the `SessionModeState` advertised on every session-creation response (`session/new`,
/// `session/load`, `session/resume`). Only modes in [`SharedPermission::enabled`] are exposed:
/// picking a non-enabled mode through `session/set_mode` later would just error out, so we don't
/// surface them in the first place.
fn build_mode_state(permission: &SharedPermission) -> SessionModeState {
    let modes: Vec<SessionMode> = permission
        .enabled()
        .iter()
        .map(|mode| {
            SessionMode::new(mode_id_for(mode), mode_display_name(mode))
                .description(mode_description(mode))
        })
        .collect();
    SessionModeState::new(mode_id_for(permission.get()), modes)
}

/// Emit a `session/update: available_commands_update` listing every installed skill as an
/// [`AvailableCommand`]. Editor clients render these as slash commands in their prompt input;
/// picking one inserts `/<name> ` and lets the user type extra context after.
///
/// `SkillCache::current` is mtime-cached, so calling this at the top of every prompt is cheap (one
/// `read_dir`, no parsing on the warm path).
async fn emit_available_commands(
    connection: &ConnectionTo<Client>,
    session_id: &SessionId,
    skills: &Arc<SkillCache>,
) {
    let snapshot = skills.current().await;
    let commands: Vec<AvailableCommand> = snapshot
        .iter()
        .map(|skill| {
            AvailableCommand::new(skill.name.clone(), skill.description.clone()).input(
                AvailableCommandInput::Unstructured(UnstructuredCommandInput::new(
                    "additional context (optional)",
                )),
            )
        })
        .collect();
    send_session_update(
        connection,
        session_id,
        SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(commands)),
    );
}

/// Outcome of running a slash-command parse against an ACP `session/prompt`'s text. Carries enough
/// detail for the prompt handler to either continue with the resolved text or surface a JSON-RPC
/// error explaining what went wrong.
#[derive(Debug)]
enum SlashInvocationError {
    SkillNotFound(String),
    SkillLoadFailed { name: String, source: String },
}

impl std::fmt::Display for SlashInvocationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlashInvocationError::SkillNotFound(name) => {
                write!(f, "unknown skill '{}'", name)
            }
            SlashInvocationError::SkillLoadFailed { name, source } => {
                write!(f, "failed to load skill '{}': {}", name, source)
            }
        }
    }
}

/// Split an ACP prompt that looks like `/<name> [extra]` into the command name and the remainder.
/// Returns `None` if the input isn't in that shape, i.e. doesn't start with `/`, has only
/// whitespace after the slash, or contains a newline before the first whitespace (heuristic: a real
/// slash command never spans lines, but pasted content might).
fn split_acp_slash(prompt_text: &str) -> Option<(String, String)> {
    let rest = prompt_text.strip_prefix('/')?;
    if rest.is_empty() || rest.starts_with(char::is_whitespace) {
        return None;
    }
    Some(match rest.split_once(char::is_whitespace) {
        Some((name, extra)) => (name.to_string(), extra.trim().to_string()),
        None => (rest.to_string(), String::new()),
    })
}

/// Intercept `/<skill-name> [extra]` invocations in an ACP prompt's
/// text. Returns the text the agent should actually run with:
///
/// - Non-slash input: returned unchanged.
/// - Slash followed by a name that isn't a syntactically valid skill identifier (e.g. a pasted path
///   like `/etc/hosts` or a `//` comment): returned unchanged so the model can see it.
/// - `/<skill-name>` matching an installed skill: returns `extra\n\n{body}` where `body` is
///   [`crate::skills::load_skill_body`]'s output (with `${MEKA_SKILL_DIR}` / `${MEKA_SESSION_ID}`
///   substituted). Empty `extra` collapses to just `body`. Same composition the REPL uses at
///   `main.rs:1004`.
/// - `/<name>` with a syntactically valid skill name but no installed skill of that name: error.
///   The shape is too deliberate to be a paste, so a typo deserves a clear "unknown skill" rather
///   than silently going to the model.
async fn slash_to_prompt_text(
    prompt_text: String,
    skills: &Arc<SkillCache>,
    session_id: &str,
) -> Result<String, SlashInvocationError> {
    let Some((name, extra)) = split_acp_slash(&prompt_text) else {
        return Ok(prompt_text);
    };
    // Anything that doesn't even look like a skill identifier was never going to match. Pass
    // through so pasted paths and code comments reach the model unchanged.
    if crate::skills::validate_skill_name(&name).is_err() {
        return Ok(prompt_text);
    }
    let snapshot = skills.current().await;
    let Some(skill) = snapshot.iter().find(|skill| skill.name == name) else {
        return Err(SlashInvocationError::SkillNotFound(name));
    };
    let body = crate::skills::load_skill_body(skill, Some(session_id))
        .await
        .map_err(|source| SlashInvocationError::SkillLoadFailed {
            name: name.clone(),
            source,
        })?;
    Ok(if extra.is_empty() {
        body
    } else {
        format!("{}\n\n{}", extra, body)
    })
}

/// Process-wide ACP server state. The outer `sessions` `RwLock` is held only for map insert /
/// lookup / remove; per-session mutable state lives behind each entry's inner `Mutex` so a
/// long-running prompt on one session never blocks operations on another.
struct ServerState {
    shared: Arc<crate::SharedDeps>,
    client_state: SharedClientState,
    sessions: Arc<tokio::sync::RwLock<std::collections::HashMap<String, SessionEntry>>>,
    /// Shared with every per-session `AcpFrontend`; see the field on `AcpFrontend` for the
    /// stdio-level rationale.
    transport_dead: Arc<std::sync::atomic::AtomicBool>,
}

/// Per-session map entry. Most fields live outside `runtime` so the cancel / set_mode / close
/// handlers can act without waiting for the long-held runtime mutex.
#[derive(Clone)]
struct SessionEntry {
    runtime: Arc<Mutex<SessionRuntime>>,
    /// In-flight turn's cancellation token. Rewritten at turn start inside `runtime`'s lock;
    /// cancel handler reads-and-clones it without touching `runtime`.
    cancellation: Arc<std::sync::RwLock<CancellationToken>>,
    /// Latch for cancels that arrive between turns. The prompt handler checks-and-clears it under
    /// the runtime lock after installing the new token, so a between-turn cancel signal isn't
    /// lost. See `acp_session_cancel_between_turns_applied_to_next_prompt`.
    cancel_pending: Arc<std::sync::atomic::AtomicBool>,
    /// Hoisted out of `SessionRuntime` so `set_mode` can flip the permission without waiting on
    /// the runtime mutex.
    permission: SharedPermission,
    /// Hoisted for the same reason as `permission`: `set_mode` / `close` need the connection to
    /// emit notifications without blocking on the runtime mutex.
    frontend: Arc<AcpFrontend>,
    /// Held purely for its `Drop` side-effect: dropping releases the OS file lock on the persisted
    /// session row. Without this, a second `meka` process could attach to the same id.
    #[allow(dead_code)]
    session_lock: Arc<crate::session::SessionLock>,
}

/// Per-session state held under `SessionEntry.runtime`. Held inside a `Mutex` because
/// `Agent::run_turn` mutates the conversation. The `frontend` field duplicates
/// `SessionEntry.frontend` so the agent (which only knows `Arc<dyn Frontend>`) can reach the
/// connection.
struct SessionRuntime {
    /// Duplicates `frontend.session_id.0`; string form retained for handlers that need it without
    /// re-extracting from the schema.
    #[allow(dead_code)]
    session_id_str: String,
    session_uuid: uuid::Uuid,
    messages: Conversation,
    cwd: SharedCwd,
    permission: SharedPermission,
    agent: Agent,
    #[allow(dead_code)]
    frontend: Arc<AcpFrontend>,
    tool_registry: crate::tools::ToolRegistry,
}

/// Run meka as an ACP agent over stdio. Returns when the client disconnects (stdin EOF).
pub async fn run_acp(
    config: ResolvedConfig,
    session_manager: SessionManager,
    mcp_manager: Option<Arc<mcp::McpClientManager>>,
    mcp_context: Arc<mcp::McpClientContext>,
) -> anyhow::Result<()> {
    // Resolve provider credentials the same way the REPL path does.
    let credential = resolve_credential_for_acp(&config, &session_manager.token_store()).await?;

    // Build process-wide shared deps once. Sessions hold an `Arc<SharedDeps>` and read fields by
    // reference; no work happens here that needs to be re-run per session.
    let shared = Arc::new(
        super::build_shared_deps(
            config,
            session_manager,
            credential,
            mcp_manager,
            mcp_context,
        )
        .await?,
    );

    // Test-only: swap in a scripted provider when the integration harness asks for it. The real
    // provider built above is dropped unused. Only compiled in debug builds. We rebuild SharedDeps
    // with the mock provider before installing it.
    #[cfg(debug_assertions)]
    let shared = if std::env::var("MEKA_ACP_MOCK_PROVIDER").as_deref() == Ok("1") {
        let rounds = crate::provider::mock::load_script_from_env()?.unwrap_or_default();
        let mock = Arc::new(crate::provider::mock::MockProvider::from_rounds(rounds));
        // Replace just the provider field, inheriting the rest from the real SharedDeps.
        // `SharedDeps: Clone` keeps this one-line and means future field additions are picked up
        // automatically; Rust still enforces the exhaustive struct literal at compile time on top.
        let new_inner = crate::SharedDeps {
            provider: mock,
            ..(*shared).clone()
        };
        // Re-publish the mock provider on the MCP context (overwriting the real one) so MCP
        // sampling callbacks hit the mock too.
        new_inner
            .mcp_context
            .set_provider(Arc::clone(&new_inner.provider));
        tracing::info!("MEKA_ACP_MOCK_PROVIDER=1: using scripted mock provider");
        Arc::new(new_inner)
    } else {
        shared
    };

    let client_state = SharedClientState::default();
    let transport_dead = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let state = Arc::new(ServerState {
        shared: Arc::clone(&shared),
        client_state: client_state.clone(),
        sessions: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
        transport_dead,
    });

    let acp_result = AcpAgentRole
        .builder()
        .name("meka")
        .on_receive_request(
            {
                let client_state = client_state.clone();
                async move |req: InitializeRequest, responder, _cx| {
                    // Stash the client's advertised capabilities (so `AcpFrontend`'s delegate_*
                    // methods can gate on them) and the client's self-identifying `Implementation`
                    // (logged here, available for diagnostics elsewhere). Both are small clones.
                    tracing::info!(
                        "ACP client connected: {}",
                        describe_client(req.client_info.as_ref())
                    );
                    client_state.record_initialize(
                        req.client_capabilities.clone(),
                        req.client_info.clone(),
                    );

                    // Advertise the optional session methods. Each marker is an empty struct;
                    // presence signals support.
                    let session_caps = SessionCapabilities::new()
                        .list(Some(SessionListCapabilities::new()))
                        .resume(Some(SessionResumeCapabilities::new()))
                        .close(Some(SessionCloseCapabilities::new()));
                    // meka accepts only `ContentBlock::Text` in
                    // `session/prompt` today. Default
                    // `PromptCapabilities` is `{image: false, audio:
                    // false, embedded_context: false}`, declared
                    // explicitly so the contract is visible in the
                    // initialize response and any future SDK default
                    // change can't quietly flip it.
                    //
                    // `mcp_capabilities` is intentionally omitted:
                    // meka sources MCP servers from its own config
                    // file and does not yet connect to servers passed
                    // through `session/new`'s `mcpServers` array.
                    // Advertising `{ http: true, sse: true }` while
                    // ignoring client-provided servers was misleading;
                    // the marker is dropped until client-MCP
                    // support lands.
                    let capabilities = AgentCapabilities::new()
                        .load_session(true)
                        .session_capabilities(session_caps)
                        .prompt_capabilities(PromptCapabilities::new());
                    // Reject the V0 sentinel explicitly. The schema uses V0 as the "couldn't parse
                    // the requested version" fallback; a clamped `min(V0, LATEST)` would silently
                    // echo it back and let the handshake proceed against a malformed input.
                    if req.protocol_version == agent_client_protocol::schema::ProtocolVersion::V0 {
                        return responder.respond_with_error(invalid_params_error(
                            "protocolVersion 0 is the schema's parse-failure sentinel; \
                             specify a supported version",
                        ));
                    }
                    // Negotiate the protocol version per the ACP spec:
                    // respond with the requested version if we
                    // support it, otherwise pin to the latest stable
                    // version we know about. A naive echo lets a
                    // future client think we support a version we
                    // haven't shipped yet.
                    let negotiated = std::cmp::min(
                        req.protocol_version,
                        agent_client_protocol::schema::ProtocolVersion::LATEST,
                    );
                    let response = InitializeResponse::new(negotiated)
                        .agent_capabilities(capabilities)
                        .agent_info(Implementation::new("meka", env!("CARGO_PKG_VERSION")));
                    responder.respond(response)
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |req: NewSessionRequest, responder, cx: ConnectionTo<Client>| {
                    if !req.cwd.is_absolute() {
                        return responder.respond_with_error(invalid_params_error(format!(
                            "cwd must be an absolute path; got `{}`",
                            req.cwd.display()
                        )));
                    }
                    let session_uuid = match state
                        .shared
                        .session_manager
                        .create_session(Some(req.cwd.clone()))
                        .await
                    {
                        Ok(uuid) => uuid,
                        Err(error) => {
                            return responder.respond_with_error(
                                agent_client_protocol::util::internal_error(format!(
                                    "failed to create meka session: {}",
                                    error
                                )),
                            );
                        }
                    };
                    // Take the OS file lock on the newly created session row so a second `meka acp`
                    // process (or an `meka repl`) can't open the same id and interleave events.
                    let session_lock = match state.shared.session_manager.lock_session(session_uuid)
                    {
                        Ok(lock) => Arc::new(lock),
                        Err(error) => {
                            return responder.respond_with_error(
                                agent_client_protocol::util::internal_error(format!(
                                    "failed to lock session: {}",
                                    error
                                )),
                            );
                        }
                    };
                    let session_id_str = session_uuid.to_string();
                    let session_id: SessionId = session_id_str.clone().into();

                    let runtime = match build_session_runtime(
                        &state.shared,
                        &state.client_state,
                        &state.transport_dead,
                        cx.clone(),
                        session_id.clone(),
                        session_id_str.clone(),
                        session_uuid,
                        req.cwd.clone(),
                        Conversation::new(),
                    )
                    .await
                    {
                        Ok(runtime) => runtime,
                        Err(error) => {
                            return responder.respond_with_error(
                                agent_client_protocol::util::internal_error(format!(
                                    "failed to build session runtime: {}",
                                    error
                                )),
                            );
                        }
                    };

                    let permission = runtime.permission.clone();
                    let frontend = Arc::clone(&runtime.frontend);
                    let entry = SessionEntry {
                        runtime: Arc::new(Mutex::new(runtime)),
                        cancellation: Arc::new(std::sync::RwLock::new(CancellationToken::new())),
                        cancel_pending: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                        permission: permission.clone(),
                        frontend,
                        session_lock,
                    };
                    state.sessions.write().await.insert(session_id_str, entry);

                    if !req.mcp_servers.is_empty() {
                        tracing::warn!(
                            "session/new: client provided {} mcpServers, \
                             ignored (config-driven MCP servers are still \
                             active)",
                            req.mcp_servers.len(),
                        );
                    }

                    // Push the initial skill palette + the configured mode picker so the editor's
                    // UI is populated before the user types their first prompt.
                    let modes = build_mode_state(&permission);
                    emit_available_commands(&cx, &session_id, &state.shared.skills).await;

                    responder.respond(NewSessionResponse::new(session_id).modes(modes))
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |req: PromptRequest, responder, cx: ConnectionTo<Client>| {
                    let state_for_spawn = Arc::clone(&state);
                    cx.spawn(
                        async move { run_prompt_turn(state_for_spawn, req, responder).await },
                    )?;
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |req: LoadSessionRequest, responder, cx: ConnectionTo<Client>| {
                    handle_load_session(Arc::clone(&state), req, responder, cx).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |req: ListSessionsRequest, responder, _cx: ConnectionTo<Client>| {
                    handle_list_sessions(Arc::clone(&state), req, responder).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |req: ResumeSessionRequest, responder, cx: ConnectionTo<Client>| {
                    handle_resume_session(Arc::clone(&state), req, responder, cx).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |req: CloseSessionRequest, responder, _cx: ConnectionTo<Client>| {
                    handle_close_session(Arc::clone(&state), req, responder).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |req: SetSessionModeRequest, responder, _cx: ConnectionTo<Client>| {
                    handle_set_session_mode(Arc::clone(&state), req, responder).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            {
                let state = Arc::clone(&state);
                async move |notif: CancelNotification, _cx: ConnectionTo<Client>| {
                    // Cancel fires through the sibling `cancellation` cell on the `SessionEntry`;
                    // we never touch the per-session runtime mutex, which the prompt handler
                    // holds for the duration of the turn.
                    //
                    // We also set `cancel_pending`: if the cancel arrives between turns (the cell
                    // still holds a stale token from the previous turn, which is now a no-op), the
                    // next prompt handler will check this flag right after installing its fresh
                    // token and cancel it immediately. Without the latch, the cancel signal is
                    // lost.
                    let entry = {
                        let sessions = state.sessions.read().await;
                        sessions.get(notif.session_id.0.as_ref()).cloned()
                    };
                    if let Some(entry) = entry {
                        entry
                            .cancel_pending
                            .store(true, std::sync::atomic::Ordering::SeqCst);
                        let token = entry
                            .cancellation
                            .read()
                            .map(|guard| guard.clone())
                            .unwrap_or_else(|poisoned| poisoned.into_inner().clone());
                        token.cancel();
                    }
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_to(Stdio::new())
        .await;

    acp_result.map_err(|error| anyhow::anyhow!("ACP server error: {}", error))
}

/// Body of the `session/prompt` spawn. Extracted so the closure stays thin. Owns `responder` and
/// replies exactly once.
///
/// Lock ordering: take the outer `sessions` read lock briefly to clone the per-session
/// `Arc<Mutex<SessionRuntime>>`, drop it, then hold *only* the per-session mutex for the duration
/// of the turn. Cancel and other sessions remain unblocked.
async fn run_prompt_turn(
    state: Arc<ServerState>,
    req: PromptRequest,
    responder: agent_client_protocol::Responder<PromptResponse>,
) -> Result<(), agent_client_protocol::Error> {
    // Accept `text` + `resource_link` (the ACP baseline). Other content variants get rejected with
    // `InvalidParams` below.
    let mut prompt_text = String::new();
    for block in &req.prompt {
        match block {
            ContentBlock::Text(text) => {
                if !prompt_text.is_empty() {
                    prompt_text.push('\n');
                }
                prompt_text.push_str(&text.text);
            }
            // `ResourceLink` is part of the ACP baseline that every agent MUST support (alongside
            // `Text`). meka doesn't fetch the resource server-side; the model sees the reference as
            // a structured tag carrying the link's name, uri, and (optional) description so it can
            // decide what to do with it.
            ContentBlock::ResourceLink(link) => {
                if !prompt_text.is_empty() {
                    prompt_text.push('\n');
                }
                let mut tag =
                    format!("<resource_link name=\"{}\" uri=\"{}\"", link.name, link.uri,);
                if let Some(mime) = &link.mime_type {
                    tag.push_str(&format!(" mime=\"{}\"", mime));
                }
                tag.push('>');
                if let Some(description) = &link.description {
                    tag.push_str(description);
                }
                tag.push_str("</resource_link>");
                prompt_text.push_str(&tag);
            }
            _ => {
                return responder.respond_with_error(invalid_params_error(
                    "meka acp accepts only `text` and `resource_link` content blocks in \
                     `prompt`; `image`, `audio`, and `resource` are not yet supported",
                ));
            }
        }
    }

    // Look up the target session by id under the outer read lock, clone the entry (cheap, two
    // `Arc`s), drop the outer guard. From here on, only the per-session runtime mutex is held; the
    // sibling cancellation cell is accessible to the cancel handler throughout the turn.
    let session_id_str = req.session_id.0.as_ref().to_string();
    let entry = {
        let sessions = state.sessions.read().await;
        match sessions.get(&session_id_str) {
            Some(entry) => entry.clone(),
            None => {
                return responder.respond_with_error(invalid_params_error(format!(
                    "unknown sessionId: {}",
                    session_id_str
                )));
            }
        }
    };

    // Acquire the runtime mutex non-blocking. If another prompt is already in flight for this
    // session, reject explicitly: ACP models one prompt at a time per session and silent queueing
    // also enables a race against the sibling cancellation cell (the second prompt would overwrite
    // the first's token before the first finishes, so `session/cancel` would target the wrong
    // turn). The lock guard is held for the entire turn so the token written below cannot be
    // overwritten by a sibling request, and per-session pre-work serialises naturally.
    let mut runtime = match entry.runtime.try_lock() {
        Ok(guard) => guard,
        Err(_) => {
            return responder.respond_with_error(invalid_params_error(
                "session already has a prompt in flight",
            ));
        }
    };

    // Install a fresh cancellation token inside the locked scope so the cancel handler (which reads
    // the sibling cell) always sees the token for the turn currently using the runtime.
    let cancellation = CancellationToken::new();
    {
        let mut guard = entry
            .cancellation
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *guard = cancellation.clone();
    }

    // Close the between-turns race: if a `session/cancel` arrived after the previous turn finished
    // but before we installed this turn's token, the cancel handler set `cancel_pending` and fired
    // the now-dead previous token. Apply the latched signal to the freshly installed token so the
    // spec-mandated cancel isn't lost. `swap` provides the read-and-clear in one step; SeqCst pairs
    // with the same ordering in the cancel handler.
    if entry
        .cancel_pending
        .swap(false, std::sync::atomic::Ordering::SeqCst)
    {
        cancellation.cancel();
    }

    // Refresh the slash-command palette before the prompt body resolves. This uses the per-session
    // frontend so the notification routes to the right ACP connection.
    let frontend = Arc::clone(&runtime.frontend);
    emit_available_commands(
        &frontend.connection,
        &frontend.session_id,
        &state.shared.skills,
    )
    .await;

    let original_prompt_text = prompt_text.clone();
    let prompt_text = match slash_to_prompt_text(
        prompt_text,
        &state.shared.skills,
        session_id_str.as_str(),
    )
    .await
    {
        Ok(text) => text,
        Err(SlashInvocationError::SkillNotFound(name)) => {
            // `slash_to_prompt_text` only returns `SkillNotFound` for strings whose first token is
            // a syntactically-valid skill name. That's deliberately a narrow filter, but it still
            // false-positives on pasted text like `/usr local lib` (parses as name=`usr`,
            // extra=`local lib`). Treat "no such skill" as "this wasn't a skill invocation after
            // all" and feed the original text to the model. It can respond with "I don't know that
            // command" if the user really meant `/<name>`. The alternative (hard-error) breaks
            // paste UX for any string starting with `/word`.
            tracing::debug!(
                "session/prompt: '/{}' didn't match a registered skill; passing through",
                name,
            );
            original_prompt_text
        }
        Err(error @ SlashInvocationError::SkillLoadFailed { .. }) => {
            // The skill name was valid and matched an installed skill; the failure is a server-side
            // problem reading the body (disk I/O, permission, etc.). JSON-RPC `InternalError` is
            // the correct classification; `InvalidParams` would mislead the client into thinking
            // the user's request was malformed.
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                error.to_string(),
            ));
        }
    };

    let SessionRuntime {
        agent,
        session_uuid,
        messages,
        ..
    } = &mut *runtime;
    // ACP sessions always have a UUID pre-allocated at `session/new`, so `run_turn` never mutates
    // this `Option`. Pass it through anyway for API compatibility with the REPL path that does
    // lazy-create sessions on first prompt.
    let mut session_uuid_opt = Some(*session_uuid);
    // Clone the cancellation token so we can probe `is_cancelled()` after the call returns. The
    // spec mandates that any cancel arriving during a turn must surface as `StopReason::Cancelled`,
    // even when the cancellation manifests as a provider / tool error rather than the clean
    // `MekaError::Interrupted` path.
    let cancel_probe = cancellation.clone();
    let result = agent
        .run_turn(&mut session_uuid_opt, messages, prompt_text, cancellation)
        .await;

    let stop_reason = match result {
        Ok(crate::agent::TurnOutcome::EndTurn) => StopReason::EndTurn,
        Ok(crate::agent::TurnOutcome::MaxTokens) => StopReason::MaxTokens,
        Ok(crate::agent::TurnOutcome::Refusal(_)) => StopReason::Refusal,
        Err(MekaError::Interrupted) => StopReason::Cancelled,
        Err(error) => {
            if cancel_probe.is_cancelled() {
                StopReason::Cancelled
            } else {
                return responder.respond_with_error(agent_client_protocol::util::internal_error(
                    format!("meka turn failed: {}", error),
                ));
            }
        }
    };

    responder.respond(PromptResponse::new(stop_reason))
}

/// `session/load`: reopen a previously persisted session and add it to the active sessions map.
/// Replays the persisted history as `session/update` notifications so the client's UI rebuilds the
/// conversation before the response goes out.
async fn handle_load_session(
    state: Arc<ServerState>,
    req: LoadSessionRequest,
    responder: agent_client_protocol::Responder<LoadSessionResponse>,
    cx: ConnectionTo<Client>,
) -> Result<(), agent_client_protocol::Error> {
    let session_id_str = req.session_id.0.as_ref().to_string();
    let session_uuid = match uuid::Uuid::parse_str(&session_id_str) {
        Ok(uuid) => uuid,
        Err(_) => {
            return responder.respond_with_error(invalid_params_error(format!(
                "malformed sessionId: {}",
                session_id_str
            )));
        }
    };

    // Refuse if a session with the same id is already loaded. Collisions between different
    // connections aren't possible (one process serves one ACP client) but a re-load of the same
    // session would discard in-flight state.
    if state.sessions.read().await.contains_key(&session_id_str) {
        return responder.respond_with_error(invalid_params_error(
            "session is already loaded; call session/close first",
        ));
    }

    if !req.cwd.is_absolute() {
        return responder.respond_with_error(invalid_params_error(format!(
            "cwd must be an absolute path; got `{}`",
            req.cwd.display()
        )));
    }

    let summary = match state
        .shared
        .session_manager
        .session_info(session_uuid)
        .await
    {
        Ok(Some(summary)) => summary,
        Ok(None) => {
            return responder.respond_with_error(invalid_params_error(format!(
                "unknown sessionId: {}",
                session_uuid
            )));
        }
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to look up session: {}", error),
            ));
        }
    };

    // Take the on-disk lock now so a concurrent process can't write events while we replay history.
    let session_lock = match state.shared.session_manager.lock_session(session_uuid) {
        Ok(lock) => Arc::new(lock),
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to lock session: {}", error),
            ));
        }
    };

    // The client's cwd wins (consistent with `session/new`'s captured cwd); update the DB row when
    // it differs so `session/list` reflects the live state.
    if summary.cwd.as_deref() != Some(req.cwd.as_path())
        && let Err(error) = state
            .shared
            .session_manager
            .update_session_cwd(session_uuid, &req.cwd)
            .await
    {
        tracing::warn!(
            "session/load: failed to update persisted cwd to {}: {}",
            req.cwd.display(),
            error,
        );
    }

    let events = match state.shared.session_manager.load_events(session_uuid).await {
        Ok(events) => events,
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to load session events: {}", error),
            ));
        }
    };
    let conversation = Conversation::from_events(events);
    let session_id: SessionId = session_id_str.clone().into();

    let runtime = match build_session_runtime(
        &state.shared,
        &state.client_state,
        &state.transport_dead,
        cx.clone(),
        session_id.clone(),
        session_id_str.clone(),
        session_uuid,
        req.cwd.clone(),
        conversation,
    )
    .await
    {
        Ok(runtime) => runtime,
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to build session runtime: {}", error),
            ));
        }
    };

    // Replay before inserting so the client sees the rebuild stream before any new turn-related
    // update could race in.
    replay_session_updates(&cx, &session_id, &runtime.cwd, &runtime.messages);

    let permission = runtime.permission.clone();
    let frontend = Arc::clone(&runtime.frontend);
    let entry = SessionEntry {
        runtime: Arc::new(Mutex::new(runtime)),
        cancellation: Arc::new(std::sync::RwLock::new(CancellationToken::new())),
        cancel_pending: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        permission: permission.clone(),
        frontend,
        session_lock,
    };
    state.sessions.write().await.insert(session_id_str, entry);

    // Refresh the palette + advertise the current mode set: the editor was reopened, so its UI
    // starts blank.
    let modes = build_mode_state(&permission);
    emit_available_commands(&cx, &session_id, &state.shared.skills).await;

    responder.respond(LoadSessionResponse::new().modes(modes))
}

/// `session/list`: paginated index of persisted sessions, filtered by cwd when the client asks.
/// Sub-agent sessions are excluded; they're internal audit rows, not user-facing conversations.
async fn handle_list_sessions(
    state: Arc<ServerState>,
    req: ListSessionsRequest,
    responder: agent_client_protocol::Responder<ListSessionsResponse>,
) -> Result<(), agent_client_protocol::Error> {
    const PAGE_SIZE: u32 = 50;
    let cwd_filter = req.cwd.as_deref();
    let cursor = req.cursor.as_deref();
    let (rows, next_cursor) = match state
        .shared
        .session_manager
        .list_sessions(PAGE_SIZE, false, cwd_filter, cursor)
        .await
    {
        Ok(pair) => pair,
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to list sessions: {}", error),
            ));
        }
    };
    // Fallback cwd for legacy rows that predate the cwd column. The process cwd matches what the
    // agent would use for relative-path resolution if the client picked one of these to load. That
    // is better than refusing to surface them.
    let fallback_cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let sessions = rows
        .into_iter()
        .map(|summary| {
            let cwd = summary.cwd.unwrap_or_else(|| fallback_cwd.clone());
            let mut info =
                SessionInfo::new(summary.id.to_string(), cwd).updated_at(summary.updated_at);
            if !summary.preview.is_empty() {
                info = info.title(summary.preview);
            }
            info
        })
        .collect::<Vec<_>>();

    let mut response = ListSessionsResponse::new(sessions);
    if let Some(token) = next_cursor {
        response = response.next_cursor(token);
    }
    responder.respond(response)
}

/// `session/resume`: adopt an existing session as active without replaying. Used when the client
/// already has the history in its UI and just wants the agent to pick up the conversation context.
async fn handle_resume_session(
    state: Arc<ServerState>,
    req: ResumeSessionRequest,
    responder: agent_client_protocol::Responder<ResumeSessionResponse>,
    cx: ConnectionTo<Client>,
) -> Result<(), agent_client_protocol::Error> {
    let session_id_str = req.session_id.0.as_ref().to_string();
    let session_uuid = match uuid::Uuid::parse_str(&session_id_str) {
        Ok(uuid) => uuid,
        Err(_) => {
            return responder.respond_with_error(invalid_params_error(format!(
                "malformed sessionId: {}",
                session_id_str
            )));
        }
    };

    if state.sessions.read().await.contains_key(&session_id_str) {
        return responder.respond_with_error(invalid_params_error(
            "session is already loaded; call session/close first",
        ));
    }

    if !req.cwd.is_absolute() {
        return responder.respond_with_error(invalid_params_error(format!(
            "cwd must be an absolute path; got `{}`",
            req.cwd.display()
        )));
    }

    let summary = match state
        .shared
        .session_manager
        .session_info(session_uuid)
        .await
    {
        Ok(Some(summary)) => summary,
        Ok(None) => {
            return responder.respond_with_error(invalid_params_error(format!(
                "unknown sessionId: {}",
                session_uuid
            )));
        }
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to look up session: {}", error),
            ));
        }
    };

    let session_lock = match state.shared.session_manager.lock_session(session_uuid) {
        Ok(lock) => Arc::new(lock),
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to lock session: {}", error),
            ));
        }
    };

    if summary.cwd.as_deref() != Some(req.cwd.as_path())
        && let Err(error) = state
            .shared
            .session_manager
            .update_session_cwd(session_uuid, &req.cwd)
            .await
    {
        tracing::warn!(
            "session/resume: failed to update persisted cwd to {}: {}",
            req.cwd.display(),
            error,
        );
    }

    let events = match state.shared.session_manager.load_events(session_uuid).await {
        Ok(events) => events,
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to load session events: {}", error),
            ));
        }
    };
    let conversation = Conversation::from_events(events);
    let session_id: SessionId = session_id_str.clone().into();

    let runtime = match build_session_runtime(
        &state.shared,
        &state.client_state,
        &state.transport_dead,
        cx.clone(),
        session_id.clone(),
        session_id_str.clone(),
        session_uuid,
        req.cwd.clone(),
        conversation,
    )
    .await
    {
        Ok(runtime) => runtime,
        Err(error) => {
            return responder.respond_with_error(agent_client_protocol::util::internal_error(
                format!("failed to build session runtime: {}", error),
            ));
        }
    };

    let permission = runtime.permission.clone();
    let frontend = Arc::clone(&runtime.frontend);
    let entry = SessionEntry {
        runtime: Arc::new(Mutex::new(runtime)),
        cancellation: Arc::new(std::sync::RwLock::new(CancellationToken::new())),
        cancel_pending: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        permission: permission.clone(),
        frontend,
        session_lock,
    };
    state.sessions.write().await.insert(session_id_str, entry);

    let modes = build_mode_state(&permission);
    emit_available_commands(&cx, &session_id, &state.shared.skills).await;

    responder.respond(ResumeSessionResponse::new().modes(modes))
}

/// `session/close`: remove a session from the active map. Cancels any in-flight prompt for that
/// session before removing it from the map so the agent loop unwinds. Detaches the session's tool
/// registry from the MCP manager so live `tools/list_changed` updates stop targeting it.
async fn handle_close_session(
    state: Arc<ServerState>,
    req: CloseSessionRequest,
    responder: agent_client_protocol::Responder<CloseSessionResponse>,
) -> Result<(), agent_client_protocol::Error> {
    let session_id_str = req.session_id.0.as_ref().to_string();
    let removed = state.sessions.write().await.remove(&session_id_str);
    let Some(entry) = removed else {
        return responder.respond_with_error(invalid_params_error("no such session"));
    };
    // Fire cancel via the sibling cell; never blocks on the runtime mutex (which an in-flight
    // prompt may hold for the whole turn).
    let token = entry
        .cancellation
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_else(|poisoned| poisoned.into_inner().clone());
    token.cancel();
    // Detach the session's tool registry from the MCP manager so tools/list_changed updates stop
    // targeting it. Briefly lock the runtime to read the registry handle; the in-flight prompt (if
    // any) was just cancelled and will release the lock shortly.
    let registry = {
        let runtime = entry.runtime.lock().await;
        runtime.tool_registry.clone()
    };
    if let Some(manager) = &state.shared.mcp_manager {
        manager.detach_registry(&registry).await;
    }
    // The inner Arcs live until any in-flight prompt's lock guard drops; the agent loop sees the
    // cancel and returns. The map entry is gone, so further requests for this session id error.
    drop(entry);
    responder.respond(CloseSessionResponse::new())
}

/// `session/set_mode`: switch the active session to a different permission level. Validates against
/// the configured enabled set; modes outside it become JSON-RPC errors rather than silently
/// failing. On success, emit `current_mode_update` so every connected client (the picker UI)
/// reflects the new state.
async fn handle_set_session_mode(
    state: Arc<ServerState>,
    req: SetSessionModeRequest,
    responder: agent_client_protocol::Responder<SetSessionModeResponse>,
) -> Result<(), agent_client_protocol::Error> {
    let session_id_str = req.session_id.0.as_ref().to_string();
    let entry = {
        let sessions = state.sessions.read().await;
        match sessions.get(&session_id_str) {
            Some(entry) => entry.clone(),
            None => {
                return responder.respond_with_error(invalid_params_error("no such session"));
            }
        }
    };
    let permission = match parse_mode_id(req.mode_id.0.as_ref()) {
        Some(p) => p,
        None => {
            return responder.respond_with_error(invalid_params_error(format!(
                "unknown mode id: {}",
                req.mode_id.0.as_ref()
            )));
        }
    };
    // No runtime mutex acquired: `SharedPermission` is `Arc<AtomicU8>` and the frontend cell holds
    // the connection. A user's mid-turn mode change takes effect on the next tool-call permission
    // probe without waiting for the in-flight turn to finish.
    if let Err(disabled) = entry.permission.try_set(permission) {
        return responder.respond_with_error(invalid_params_error(format!(
            "mode '{}' is not enabled in this configuration",
            disabled.0
        )));
    }
    send_session_update(
        &entry.frontend.connection,
        &entry.frontend.session_id,
        SessionUpdate::CurrentModeUpdate(CurrentModeUpdate::new(req.mode_id.clone())),
    );
    responder.respond(SetSessionModeResponse::new())
}

/// Build a fresh [`SessionRuntime`] from the process-wide
/// [`crate::SharedDeps`]. Called from `session/new`, `session/load`,
/// and `session/resume`. Each follows the same shape:
/// 1. Construct the per-session `AcpFrontend` bound to this connection + session id.
/// 2. Build a per-session `SharedPermission` cell seeded from config defaults.
/// 3. Build the per-session `Agent` + `ToolRegistry` via [`crate::build_session_agent`], which also
///    attaches the registry to the MCP manager.
/// 4. Bundle everything into a `SessionRuntime`.
#[allow(clippy::too_many_arguments)]
async fn build_session_runtime(
    shared: &Arc<crate::SharedDeps>,
    client_state: &SharedClientState,
    transport_dead: &Arc<std::sync::atomic::AtomicBool>,
    connection: ConnectionTo<Client>,
    session_id: SessionId,
    session_id_str: String,
    session_uuid: uuid::Uuid,
    cwd_path: PathBuf,
    messages: Conversation,
) -> anyhow::Result<SessionRuntime> {
    let cwd: SharedCwd = Arc::new(std::sync::RwLock::new(cwd_path));
    let permission =
        SharedPermission::new(shared.config.permission, shared.config.enabled_permissions);

    let acp_frontend = Arc::new(AcpFrontend::new(
        connection,
        session_id,
        Arc::clone(&cwd),
        client_state.clone(),
        Arc::clone(transport_dead),
    ));
    let frontend: Arc<dyn Frontend> = acp_frontend.clone();

    let (agent, tool_registry) =
        crate::build_session_agent(shared, permission.clone(), frontend, Arc::clone(&cwd)).await?;

    Ok(SessionRuntime {
        session_id_str,
        session_uuid,
        messages,
        cwd,
        permission,
        agent,
        frontend: acp_frontend,
        tool_registry,
    })
}

/// Mirrors `main::resolve_credential` but stays in this module to avoid widening `main`'s
/// visibility for an ACP-only call site.
async fn resolve_credential_for_acp(
    config: &ResolvedConfig,
    token_store: &crate::session::TokenStore,
) -> anyhow::Result<AuthCredential> {
    let Some(profile) = config.active_profile.as_deref() else {
        anyhow::bail!("meka acp requires a configured provider; run `meka provider add <name>`");
    };
    token_store
        .load_provider_credential(profile)
        .await?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "provider profile '{}' has no stored credential; run `meka provider login {}`",
                profile,
                profile
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::PermissionOutcome;

    // `AcpFrontend` itself can't be unit-tested (requires a live `ConnectionTo<Client>`);
    // per-session behaviour is covered end-to-end in `tests/acp.rs`. The pure helpers below are
    // what this unit-test module owns.

    #[test]
    fn test_tool_kind_for_covers_builtins() {
        assert_eq!(tool_kind_for("read_file"), ToolKind::Read);
        assert_eq!(tool_kind_for("edit_file"), ToolKind::Edit);
        assert_eq!(tool_kind_for("write_file"), ToolKind::Edit);
        assert_eq!(tool_kind_for("find_files"), ToolKind::Search);
        assert_eq!(tool_kind_for("search_contents"), ToolKind::Search);
        assert_eq!(tool_kind_for("execute_command"), ToolKind::Execute);
        assert_eq!(tool_kind_for("fetch_url"), ToolKind::Fetch);
        assert_eq!(tool_kind_for("spawn_agent"), ToolKind::Think);
        // MCP-loaded tools and anything else fall through.
        assert_eq!(tool_kind_for("mcp__github__create_issue"), ToolKind::Other);
        assert_eq!(tool_kind_for("scratchpad_write"), ToolKind::Other);
        assert_eq!(tool_kind_for("totally_unknown"), ToolKind::Other);
    }

    #[test]
    fn test_tool_locations_resolves_relative_against_cwd() {
        let cwd: SharedCwd = Arc::new(std::sync::RwLock::new(PathBuf::from("/home/agent/proj")));
        let input = serde_json::json!({"path": "src/main.rs"});
        let locations = tool_locations("read_file", &input, &cwd);
        assert_eq!(locations.len(), 1);
        assert_eq!(
            locations[0].path,
            PathBuf::from("/home/agent/proj/src/main.rs")
        );
    }

    #[test]
    fn test_tool_locations_passes_absolute_paths_through() {
        let cwd: SharedCwd = Arc::new(std::sync::RwLock::new(PathBuf::from("/some/other/dir")));
        let input = serde_json::json!({"path": "/etc/hosts"});
        let locations = tool_locations("edit_file", &input, &cwd);
        assert_eq!(locations[0].path, PathBuf::from("/etc/hosts"));
    }

    #[test]
    fn test_tool_locations_empty_for_non_path_tools() {
        let cwd: SharedCwd = Arc::new(std::sync::RwLock::new(PathBuf::from("/")));
        let input = serde_json::json!({"command": "ls"});
        assert!(tool_locations("execute_command", &input, &cwd).is_empty());
        assert!(tool_locations("web_search", &input, &cwd).is_empty());
    }

    #[test]
    fn test_build_completion_content_prefers_diff_metadata() {
        let metadata = Some(ToolOutputMetadata::Diff {
            path: PathBuf::from("/tmp/foo.txt"),
            old_text: Some("old".to_string()),
            new_text: "new".to_string(),
        });
        let content = vec![ToolResultContent::Text {
            text: "ignored".to_string(),
        }];
        let blocks = build_completion_content(&content, metadata);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(blocks[0], ToolCallContent::Diff(_)));
    }

    #[test]
    fn test_translate_permission_outcome_maps_each_option() {
        use agent_client_protocol::schema::SelectedPermissionOutcome;

        // Capture sticky pushes via a `Cell` so each call site borrows it fresh; this sidesteps the
        // closure-vs-direct-read borrow conflict that comes from sharing one `&mut Vec`.
        let sticky: std::cell::RefCell<Vec<&'static str>> = std::cell::RefCell::new(Vec::new());
        let record = |s: StickyDecision| {
            sticky.borrow_mut().push(match s {
                StickyDecision::AllowAlways => "allow",
                StickyDecision::RejectAlways => "deny",
            });
        };

        assert_eq!(
            translate_permission_outcome(
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                    OPTION_ALLOW_ONCE,
                )),
                "read_file",
                record,
            ),
            PermissionOutcome::Allow,
        );
        assert!(
            sticky.borrow().is_empty(),
            "allow_once must not record a sticky"
        );

        assert_eq!(
            translate_permission_outcome(
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                    OPTION_ALLOW_ALWAYS,
                )),
                "read_file",
                record,
            ),
            PermissionOutcome::Allow,
        );
        assert_eq!(sticky.borrow().last().copied(), Some("allow"));

        assert_eq!(
            translate_permission_outcome(
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                    OPTION_REJECT_ONCE,
                )),
                "write_file",
                record,
            ),
            PermissionOutcome::Deny,
        );
        assert_eq!(
            sticky.borrow().last().copied(),
            Some("allow"),
            "reject_once must not push"
        );

        assert_eq!(
            translate_permission_outcome(
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                    OPTION_REJECT_ALWAYS,
                )),
                "write_file",
                record,
            ),
            PermissionOutcome::Deny,
        );
        assert_eq!(sticky.borrow().last().copied(), Some("deny"));

        assert_eq!(
            translate_permission_outcome(RequestPermissionOutcome::Cancelled, "read_file", record,),
            PermissionOutcome::Cancelled,
        );
    }

    #[test]
    fn test_translate_permission_outcome_unknown_option_denies() {
        use agent_client_protocol::schema::SelectedPermissionOutcome;
        let result = translate_permission_outcome(
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new("future_option")),
            "read_file",
            &mut |_| {},
        );
        assert_eq!(result, PermissionOutcome::Deny);
    }

    #[test]
    fn test_build_completion_content_falls_back_to_text() {
        let content = vec![ToolResultContent::Text {
            text: "hello".to_string(),
        }];
        let blocks = build_completion_content(&content, None);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(blocks[0], ToolCallContent::Content(_)));
    }

    /// Image content has no ACP analogue today, so `build_completion_content` collapses it to a
    /// `[image]` text marker. Walks the resulting `ContentBlock::Text` to confirm the literal. This
    /// guards against accidentally swapping in the `ImageContent` ACP variant before the wire
    /// format is wired through end-to-end.
    #[test]
    fn test_build_completion_content_image_falls_back_to_marker() {
        use crate::provider::ImageSource;
        let content = vec![ToolResultContent::Image {
            source: ImageSource {
                source_type: "base64".to_string(),
                media_type: "image/png".to_string(),
                data: "irrelevant".to_string(),
            },
        }];
        let blocks = build_completion_content(&content, None);
        assert_eq!(blocks.len(), 1);
        let ToolCallContent::Content(chunk) = &blocks[0] else {
            panic!("expected ToolCallContent::Content; got {:?}", blocks[0]);
        };
        let ContentBlock::Text(text) = &chunk.content else {
            panic!("expected ContentBlock::Text; got {:?}", chunk.content);
        };
        assert_eq!(text.text, "[image]");
    }

    #[test]
    fn test_parse_mode_id_covers_all_levels() {
        assert_eq!(parse_mode_id("none"), Some(Permission::None));
        assert_eq!(parse_mode_id("read"), Some(Permission::Read));
        assert_eq!(parse_mode_id("ask"), Some(Permission::Ask));
        assert_eq!(parse_mode_id("write"), Some(Permission::Write));
    }

    #[test]
    fn test_parse_mode_id_rejects_garbage() {
        assert!(parse_mode_id("READ").is_none(), "case-sensitive");
        assert!(parse_mode_id("admin").is_none());
        assert!(parse_mode_id("").is_none());
    }

    #[test]
    fn test_build_mode_state_lists_only_enabled_modes() {
        use crate::permission::{EnabledPermissions, SharedPermission};
        let enabled =
            EnabledPermissions::from_modes([Permission::Read, Permission::Ask]).expect("non-empty");
        let permission = SharedPermission::new(Permission::Read, enabled);

        let state = build_mode_state(&permission);
        let ids: Vec<&str> = state
            .available_modes
            .iter()
            .map(|m| m.id.0.as_ref())
            .collect();
        assert_eq!(ids, vec!["read", "ask"]);
        assert_eq!(state.current_mode_id.0.as_ref(), "read");
        // Descriptions populated.
        assert!(
            state
                .available_modes
                .iter()
                .all(|m| m.description.is_some()),
            "every mode advertised must carry a description"
        );
    }

    #[test]
    fn test_build_mode_state_reflects_current_after_set() {
        use crate::permission::{EnabledPermissions, SharedPermission};
        let permission = SharedPermission::new(Permission::Read, EnabledPermissions::ALL);
        permission
            .try_set(Permission::Write)
            .expect("write enabled");
        assert_eq!(
            build_mode_state(&permission).current_mode_id.0.as_ref(),
            "write"
        );
    }

    #[tokio::test]
    async fn test_slash_to_prompt_text_passes_through_non_slash() {
        let cache = SkillCache::for_root(None);
        let out = slash_to_prompt_text("just a normal prompt".to_string(), &cache, "sid")
            .await
            .expect("ok");
        assert_eq!(out, "just a normal prompt");
    }

    #[tokio::test]
    async fn test_slash_to_prompt_text_passes_through_paste_shaped_input() {
        // A pasted path like `/etc/hosts is a config file` has an invalid skill-name first token
        // (slash inside the name), so the helper must NOT touch it.
        let cache = SkillCache::for_root(None);
        let out = slash_to_prompt_text("/etc/hosts is the config file".to_string(), &cache, "sid")
            .await
            .expect("pass-through");
        assert_eq!(out, "/etc/hosts is the config file");
    }

    #[tokio::test]
    async fn test_slash_to_prompt_text_passes_through_double_slash_comment() {
        // `//foo` parses as name="/foo", which is invalid; pass through.
        let cache = SkillCache::for_root(None);
        let out = slash_to_prompt_text("//comment line".to_string(), &cache, "sid")
            .await
            .expect("pass-through");
        assert_eq!(out, "//comment line");
    }

    #[tokio::test]
    async fn test_slash_to_prompt_text_unknown_but_valid_name_errors() {
        // A clean `/<name>` shape with a syntactically valid skill name that isn't installed:
        // error, since the only realistic source of this shape is a typo'd palette pick.
        let cache = SkillCache::for_root(None);
        let err = slash_to_prompt_text("/nonexistent".to_string(), &cache, "sid")
            .await
            .expect_err("should error");
        assert!(
            matches!(err, SlashInvocationError::SkillNotFound(ref name) if name == "nonexistent")
        );
    }

    #[tokio::test]
    async fn test_slash_to_prompt_text_known_skill_composes_body() {
        // Drop a SKILL.md under a tempdir, point a fresh cache at it.
        let temp = tempfile::tempdir().expect("tempdir");
        let skill_dir = temp.path().join("demo");
        std::fs::create_dir_all(&skill_dir).expect("mkdir skill");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\ndescription: demo skill\n---\nrun ls in ${MEKA_SKILL_DIR}\n",
        )
        .expect("write SKILL.md");

        let cache = SkillCache::for_root(Some(temp.path().to_path_buf()));
        let out = slash_to_prompt_text("/demo only fetch UK news".to_string(), &cache, "sid-xyz")
            .await
            .expect("ok");
        assert!(
            out.starts_with("only fetch UK news\n\n"),
            "extra context must lead: {}",
            out
        );
        assert!(
            out.contains("run ls in ") && out.contains("demo"),
            "body must include the substituted skill dir: {}",
            out
        );
        assert!(
            out.contains("Base directory for this skill"),
            "skill_context_header must be present: {}",
            out
        );
    }

    #[tokio::test]
    async fn test_slash_to_prompt_text_known_skill_no_extra() {
        let temp = tempfile::tempdir().expect("tempdir");
        let skill_dir = temp.path().join("ping");
        std::fs::create_dir_all(&skill_dir).expect("mkdir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\ndescription: ping\n---\npong\n",
        )
        .expect("write");

        let cache = SkillCache::for_root(Some(temp.path().to_path_buf()));
        let out = slash_to_prompt_text("/ping".to_string(), &cache, "sid")
            .await
            .expect("ok");
        // No `extra\n\n` prefix when the user passed only the skill name; the body stands alone.
        assert!(
            !out.starts_with("\n\n"),
            "bare /skill must not have a leading newline: {:?}",
            out
        );
        assert!(out.contains("pong"));
    }

    #[test]
    fn test_shared_client_state_round_trip() {
        let shared = SharedClientState::default();
        // Default snapshot has every capability false and no client identity recorded.
        let initial = shared.capabilities();
        assert!(!initial.fs.read_text_file);
        assert!(!initial.fs.write_text_file);
        assert!(!initial.terminal);
        assert!(shared.client_info().is_none());

        let updated_caps = ClientCapabilities::new()
            .fs(agent_client_protocol::schema::FileSystemCapabilities::new()
                .read_text_file(true)
                .write_text_file(true))
            .terminal(true);
        let updated_info = Implementation::new("test-editor", "9.9.9");
        shared.record_initialize(updated_caps, Some(updated_info));

        let after_caps = shared.capabilities();
        assert!(after_caps.fs.read_text_file);
        assert!(after_caps.fs.write_text_file);
        assert!(after_caps.terminal);
        let after_info = shared.client_info().expect("info present");
        assert_eq!(after_info.name, "test-editor");
        assert_eq!(after_info.version, "9.9.9");
    }

    #[test]
    fn test_describe_client_formats_known_and_unknown() {
        assert_eq!(describe_client(None), "<unknown> <unknown>");
        let info = Implementation::new("zed", "0.999.0");
        assert_eq!(describe_client(Some(&info)), "zed 0.999.0");
    }
}
