//! `Frontend` — the swappable driver for agent output and approval
//! round-trips.
//!
//! `Agent::run_turn` emits its user-facing output (streamed assistant
//! text, thinking blocks, tool-call indicators, todo lists, token usage)
//! and its tool-approval requests through `Arc<dyn Frontend>` instead of
//! calling `render::*` and `std::sync::mpsc` directly. The REPL today is
//! one impl ([`ReplFrontend`]); ACP, a Telegram bridge, or a web UI
//! become additional impls without touching the agent core.
//!
//! The event-based shape mirrors ACP's `session/update` notification —
//! one channel for every kind of agent-emitted output, discriminated by
//! the [`FrontendEvent`] variant.

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    provider::TokenUsage,
    render::{self, OutputSpacing, RenderMode, StreamingRenderer},
    tools::todo::TodoItem,
};

/// Trait the agent loop talks through to surface output and ask the user
/// to approve tool calls. Implementations are responsible for rendering
/// mode, newline spacing, and any inter-event formatting.
#[async_trait]
pub trait Frontend: Send + Sync {
    /// Emit a one-way UI event. Implementations must tolerate any order
    /// of events but may assume `TurnStarted` precedes any per-turn
    /// activity and `TurnFinished` closes it.
    async fn emit(&self, event: FrontendEvent);

    /// Round-trip request for user approval of a tool call. Used only
    /// when [`crate::permission::Permission::Ask`] is active.
    /// [`PermissionOutcome::Cancelled`] is distinct from
    /// [`PermissionOutcome::Deny`] — it indicates the user cancelled the
    /// enclosing turn (Ctrl+C, `session/cancel`), which ACP will surface
    /// later. Today's REPL collapses it to deny semantics.
    async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome;

    /// Delegate a file read to whatever filesystem the frontend owns
    /// (typically the ACP client's in-buffer view of the file).
    /// `Some(Ok(content))` means the frontend handled it; `Some(Err(_))`
    /// means delegation was attempted and failed (surface the error to
    /// the user — don't silently fall back); `None` means "no delegate
    /// available, do it locally".
    ///
    /// `line` and `limit` follow ACP's 1-based line / line-count
    /// convention.
    async fn delegate_fs_read(
        &self,
        _path: &Path,
        _line: Option<u32>,
        _limit: Option<u32>,
    ) -> Option<Result<String, FrontendError>> {
        None
    }

    /// Delegate a file write. Same `None` / `Some(Err)` / `Some(Ok)`
    /// semantics as [`Self::delegate_fs_read`].
    async fn delegate_fs_write(
        &self,
        _path: &Path,
        _content: &str,
    ) -> Option<Result<(), FrontendError>> {
        None
    }

    /// Delegate a shell command to the frontend's hosted terminal
    /// (e.g. ACP `terminal/*`). Same `None` / `Some(Err)` / `Some(Ok)`
    /// semantics as [`Self::delegate_fs_read`].
    async fn delegate_execute(
        &self,
        _spec: DelegatedExecSpec,
    ) -> Option<Result<DelegatedExecOutput, FrontendError>> {
        None
    }

    /// Returns `true` if the frontend has observed that its client is
    /// no longer reachable (e.g. an ACP client has closed its stdio
    /// connection, so every `session/update` notification returns an
    /// error). The agent loop checks this at every loop iteration and
    /// short-circuits with [`crate::error::AgshError::Interrupted`] so
    /// it doesn't keep burning provider / MCP cycles for an audience
    /// that's gone away.
    ///
    /// REPL and silent frontends never disconnect in this sense, so
    /// the default `false` is correct for them.
    fn client_disconnected(&self) -> bool {
        false
    }
}

/// Error from a frontend-delegated operation ([`Frontend::delegate_fs_read`],
/// [`Frontend::delegate_fs_write`], [`Frontend::delegate_execute`]). Wraps
/// whatever the underlying transport (ACP JSON-RPC, etc.) returned in a
/// stringly form so tools can splice it into their `ToolOutput` text
/// without depending on the transport crate.
#[derive(Debug, Clone)]
pub struct FrontendError(pub String);

impl FrontendError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for FrontendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FrontendError {}

/// Description of the command a delegated `execute_command` should run.
/// The frontend is responsible for spawning the process (via ACP
/// `terminal/create`, an MCP equivalent, etc.) and returning the
/// assembled output via [`DelegatedExecOutput`].
#[derive(Debug, Clone)]
pub struct DelegatedExecSpec {
    /// The executable to run. agsh always picks a shell (e.g. `sh` /
    /// `powershell.exe`) and passes the user-supplied command as an
    /// argument, so the frontend doesn't need to do its own shell-
    /// quoting.
    pub command: String,
    pub args: Vec<String>,
    /// Process environment to set in addition to whatever the frontend
    /// supplies as its baseline. agsh forwards a filtered subset of the
    /// agent's env so things like `PATH` / `LANG` are preserved.
    pub env: Vec<(String, String)>,
    /// Working directory for the spawned process. Almost always
    /// `Some(_)` — agsh's per-session cwd snapshot at the call site.
    pub cwd: Option<PathBuf>,
    /// Hard timeout. The frontend should attempt to kill the process and
    /// return whatever output accumulated. `None` defers to the
    /// frontend's own default.
    pub timeout: Option<Duration>,
    /// Maximum bytes of output the frontend should retain. The frontend
    /// signals truncation via [`DelegatedExecOutput::truncated`].
    pub output_byte_limit: Option<u64>,
    /// Cancellation token from the agent loop. The frontend may use this
    /// to short-circuit `wait_for_exit` and issue a kill.
    pub cancellation: CancellationToken,
}

/// Output of a delegated execute_command. ACP's `terminal/*` returns one
/// combined output stream; we flatten any stdout/stderr separation into
/// the single [`Self::output`] field. agsh's local execute_command renders
/// the same way (stderr is appended to stdout with a separator), so this
/// matches.
#[derive(Debug, Clone)]
pub struct DelegatedExecOutput {
    pub output: String,
    pub exit_code: Option<i32>,
    /// Signal name (e.g. `"SIGTERM"`) when the process was killed.
    /// Mutually exclusive with [`Self::exit_code`] in practice.
    pub signal: Option<String>,
    /// True iff the frontend dropped bytes past
    /// [`DelegatedExecSpec::output_byte_limit`].
    pub truncated: bool,
}

/// One-way UI event emitted by the agent loop.
#[derive(Debug, Clone)]
pub enum FrontendEvent {
    /// A new session was created. Carries the session UUID.
    SessionStarted { id: Uuid },
    /// The agent is about to start a turn. REPL uses this to emit the
    /// `newline_after_prompt` blank line.
    TurnStarted,
    /// The agent finished a turn cleanly. REPL uses this to flush any
    /// open streaming renderer and emit the `newline_before_prompt`
    /// blank line.
    TurnFinished,
    /// A streamed chunk of assistant text. Multiple deltas concatenate
    /// into one logical text run; any non-text event closes the run.
    AssistantTextDelta(String),
    /// A complete thinking block. Emitted after the provider's
    /// `ThinkingComplete` stream event.
    ThinkingBlock {
        content: String,
        /// Provider-opaque signature blob (Claude's extended-thinking
        /// signature). Unread today; kept on the event so future
        /// session/load replay can round-trip it without an event
        /// shape change.
        #[allow(dead_code)]
        signature: Option<String>,
    },
    /// A tool call is about to be dispatched. `schema` is the tool's
    /// `parameters` JSON Schema (cloned from its `ToolDefinition`) when
    /// available, used for primary-param rendering. `id` is the
    /// `tool_use_id` assigned by the provider — frontends use it to
    /// correlate this announcement with the matching
    /// [`Self::ToolCallCompleted`].
    ToolCallStarted {
        id: String,
        name: String,
        input: serde_json::Value,
        schema: Option<serde_json::Value>,
    },
    /// A previously-announced tool call has finished. Emitted once per
    /// tool in source order after the parallel dispatch settles. The
    /// REPL impl ignores this today (tool results render through the
    /// model's next assistant message); the ACP impl translates it to
    /// `session/update: tool_call_update` with `status: completed | failed`.
    ToolCallCompleted {
        id: String,
        is_error: bool,
        content: Vec<crate::provider::ToolResultContent>,
        /// Tool-specific structured side-channel. `edit_file` /
        /// `write_file` populate [`ToolOutputMetadata::Diff`] so ACP can
        /// emit a proper `diff` content block (and Zed can render its
        /// apply-diff UI). `None` for tools that have nothing extra.
        metadata: Option<ToolOutputMetadata>,
    },
    /// The shared todo list was just replaced via `todo_write`. Emitted
    /// by the agent loop after the tool succeeds; the REPL renders the
    /// list and the agent's per-turn `OutputSpacing` is advanced.
    TodoListUpdated(Vec<TodoItem>),
    /// End-of-turn token-usage summary.
    TokenUsage(TokenUsage),
}

/// Structured side-channel a tool can attach to its [`crate::tools::ToolOutput`]
/// for frontends that know how to render it. Frontends that don't
/// understand a variant ignore it (the regular `content` text is still
/// the source of truth for the model and the REPL).
#[derive(Debug, Clone)]
pub enum ToolOutputMetadata {
    /// Pre/post file content produced by `edit_file` / `write_file`.
    /// `old_text == None` means the file did not exist before the call
    /// (the write created it).
    Diff {
        path: std::path::PathBuf,
        old_text: Option<String>,
        new_text: String,
    },
}

/// Round-trip request for tool-call approval.
#[derive(Debug, Clone)]
pub struct PermissionRequest {
    pub tool_name: String,
    /// The most user-meaningful argument for display in the prompt
    /// (e.g. the file path for `read_file`, the command for
    /// `execute_command`). Resolved via [`crate::render::resolve_primary_param`].
    pub primary_param: Option<String>,
    /// Per-turn cancellation token. ACP frontends race their
    /// `session/request_permission` round-trip against this so a
    /// `session/cancel` during an `Ask`-mode prompt resolves
    /// promptly instead of hanging until the client replies.
    pub cancellation: tokio_util::sync::CancellationToken,
}

/// Outcome of a [`Frontend::request_permission`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionOutcome {
    Allow,
    Deny,
    /// The enclosing turn was cancelled while the request was in flight.
    /// The ACP frontend surfaces this as `{outcome: cancelled}`;
    /// the REPL collapses it to a deny-shaped tool error.
    Cancelled,
}

/// Frontend wrapper used by sub-agents when the parent is interactive
/// enough to host permission prompts. `emit` is a no-op (sub-agents
/// don't stream output to the user — their final report flows back
/// through the parent's `spawn_agent` tool result), but
/// `request_permission` is forwarded to the held delegate so the
/// user is prompted in their original UI (REPL approval line / ACP
/// `session/request_permission`).
///
/// Constructed in [`crate::tools::subagent::SpawnAgentTool::execute`]
/// with the parent agent's frontend as the delegate.
pub struct PermissionForwardingFrontend {
    delegate: Arc<dyn Frontend>,
}

impl PermissionForwardingFrontend {
    pub fn new(delegate: Arc<dyn Frontend>) -> Self {
        Self { delegate }
    }
}

#[async_trait]
impl Frontend for PermissionForwardingFrontend {
    async fn emit(&self, _event: FrontendEvent) {}

    fn client_disconnected(&self) -> bool {
        // Sub-agents must observe the parent's disconnect so their
        // own run_turn loop short-circuits — without this forward,
        // a sub-agent under a dropped ACP connection keeps burning
        // provider tokens.
        self.delegate.client_disconnected()
    }

    async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome {
        self.delegate.request_permission(request).await
    }

    async fn delegate_fs_read(
        &self,
        path: &Path,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> Option<Result<String, FrontendError>> {
        self.delegate.delegate_fs_read(path, line, limit).await
    }

    async fn delegate_fs_write(
        &self,
        path: &Path,
        content: &str,
    ) -> Option<Result<(), FrontendError>> {
        self.delegate.delegate_fs_write(path, content).await
    }

    async fn delegate_execute(
        &self,
        spec: DelegatedExecSpec,
    ) -> Option<Result<DelegatedExecOutput, FrontendError>> {
        self.delegate.delegate_execute(spec).await
    }
}

/// Fully-silent frontend: drops every emit and denies every
/// permission request. Used by tests and `agsh tools list`'s
/// reference registry — both want a frontend that never reaches out
/// to a user. Sub-agents use [`PermissionForwardingFrontend`]
/// instead so their permission prompts surface in the parent's UI.
pub struct SilentFrontend;

#[async_trait]
impl Frontend for SilentFrontend {
    async fn emit(&self, _event: FrontendEvent) {}

    async fn request_permission(&self, _request: PermissionRequest) -> PermissionOutcome {
        // No human to ask — safest default.
        PermissionOutcome::Deny
    }
}

/// Construction-time configuration for [`ReplFrontend`]. These fields
/// used to live on `AgentOptions`; they are UI concerns and now belong
/// to the frontend impl.
pub struct ReplFrontendConfig {
    pub render_mode: RenderMode,
    pub newline_before_prompt: bool,
    pub newline_after_prompt: bool,
    pub show_session_id_on_create: bool,
    pub show_token_usage: bool,
    pub thinking_show_content: bool,
    /// Sender for the REPL's `AgentToReplEvent` channel, used to forward
    /// approval requests to the blocking REPL thread.
    pub agent_event_sender: std::sync::mpsc::Sender<crate::repl::AgentToReplEvent>,
}

/// REPL-side [`Frontend`] impl. Owns the [`StreamingRenderer`] and
/// [`OutputSpacing`] state that used to be threaded through
/// `Agent::run_turn` / `run_streaming`, and forwards approval requests
/// over the existing mpsc to the blocking REPL thread.
pub struct ReplFrontend {
    config: ReplFrontendConfig,
    state: Mutex<ReplFrontendState>,
}

struct ReplFrontendState {
    spacing: OutputSpacing,
    /// Open across consecutive `AssistantTextDelta` events; closed by
    /// any non-text event (or `TurnFinished`).
    renderer: Option<StreamingRenderer>,
}

impl ReplFrontend {
    pub fn new(config: ReplFrontendConfig) -> Self {
        Self {
            config,
            state: Mutex::new(ReplFrontendState {
                spacing: OutputSpacing::new(),
                renderer: None,
            }),
        }
    }

    /// Flush and drop any open streaming renderer. Called before any
    /// non-text event so block types don't interleave on stderr.
    fn close_text_run(state: &mut ReplFrontendState) {
        if let Some(mut renderer) = state.renderer.take() {
            // Rendering errors here are typically a broken stderr pipe;
            // log and move on rather than panicking inside `emit`.
            if let Err(error) = renderer.finish() {
                tracing::debug!("frontend renderer finish failed: {}", error);
            }
        }
    }
}

#[async_trait]
impl Frontend for ReplFrontend {
    async fn emit(&self, event: FrontendEvent) {
        // Held briefly across synchronous render calls. The agent loop
        // emits events serially per turn, so contention is effectively
        // zero; the lock is purely a `Send + Sync` discipline check.
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        match event {
            FrontendEvent::SessionStarted { id } => {
                if self.config.show_session_id_on_create {
                    render::render_session_id("Creating new session", &id.to_string());
                }
            }
            FrontendEvent::TurnStarted => {
                if self.config.newline_after_prompt {
                    eprintln!();
                    state.spacing.after_prompt();
                }
            }
            FrontendEvent::TurnFinished => {
                Self::close_text_run(&mut state);
                if self.config.newline_before_prompt {
                    eprintln!();
                }
            }
            FrontendEvent::AssistantTextDelta(text) => {
                if state.renderer.is_none() {
                    if state.spacing.before_text() {
                        eprintln!();
                    }
                    state.renderer = Some(StreamingRenderer::new(self.config.render_mode));
                }
                if let Some(renderer) = state.renderer.as_mut()
                    && let Err(error) = renderer.push_delta(&text)
                {
                    tracing::debug!("frontend renderer push_delta failed: {}", error);
                }
            }
            FrontendEvent::ThinkingBlock {
                content,
                signature: _,
            } => {
                Self::close_text_run(&mut state);
                if state.spacing.before_thinking() {
                    eprintln!();
                }
                render::render_thinking_block(&content, self.config.thinking_show_content);
            }
            FrontendEvent::ToolCallStarted {
                id: _,
                name,
                input,
                schema,
            } => {
                Self::close_text_run(&mut state);
                if state.spacing.before_tool_indicator() {
                    eprintln!();
                }
                render::render_tool_indicator(&name, &input, schema.as_ref());
            }
            // The REPL renders tool results inline through the agent's
            // own message-history path (the next assistant turn). No
            // additional UI is needed at completion time — the model's
            // response that follows already summarizes what happened.
            FrontendEvent::ToolCallCompleted { .. } => {}
            FrontendEvent::TodoListUpdated(items) => {
                Self::close_text_run(&mut state);
                render::render_todo_list(&items);
                state.spacing.after_todo_list();
            }
            FrontendEvent::TokenUsage(usage) => {
                Self::close_text_run(&mut state);
                if self.config.show_token_usage {
                    render::render_token_usage(&usage);
                }
            }
        }
    }

    async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome {
        let (response_sender, response_receiver) = tokio::sync::oneshot::channel::<bool>();
        let approval = crate::repl::ToolApprovalRequest {
            tool_name: request.tool_name,
            primary_param: request.primary_param,
            response_sender,
        };
        if self
            .config
            .agent_event_sender
            .send(crate::repl::AgentToReplEvent::ApprovalRequest(approval))
            .is_err()
        {
            // REPL thread is gone — there is no human to ask. Treat as
            // cancellation rather than denial so the caller's
            // ToolOutput message is honest about the cause.
            return PermissionOutcome::Cancelled;
        }
        match response_receiver.await {
            Ok(true) => PermissionOutcome::Allow,
            Ok(false) => PermissionOutcome::Deny,
            Err(_) => PermissionOutcome::Cancelled,
        }
    }
}

/// Test-only frontend that records every event it receives. Available
/// to the rest of the crate's test suite via
/// `crate::frontend::testing::RecordingFrontend`.
#[cfg(test)]
pub mod testing {
    use super::*;

    pub struct RecordingFrontend {
        events: Mutex<Vec<FrontendEvent>>,
        permission_response: Mutex<PermissionOutcome>,
    }

    impl RecordingFrontend {
        pub fn new() -> Self {
            Self {
                events: Mutex::new(Vec::new()),
                permission_response: Mutex::new(PermissionOutcome::Allow),
            }
        }

        pub fn with_permission(response: PermissionOutcome) -> Self {
            Self {
                events: Mutex::new(Vec::new()),
                permission_response: Mutex::new(response),
            }
        }

        pub fn events(&self) -> Vec<FrontendEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl Default for RecordingFrontend {
        fn default() -> Self {
            Self::new()
        }
    }

    #[async_trait]
    impl Frontend for RecordingFrontend {
        async fn emit(&self, event: FrontendEvent) {
            self.events.lock().unwrap().push(event);
        }

        async fn request_permission(&self, _request: PermissionRequest) -> PermissionOutcome {
            self.permission_response.lock().unwrap().clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{testing::RecordingFrontend, *};

    #[tokio::test]
    async fn test_silent_frontend_emit_is_no_op_and_does_not_panic() {
        let frontend = SilentFrontend;
        frontend.emit(FrontendEvent::TurnStarted).await;
        frontend
            .emit(FrontendEvent::AssistantTextDelta("hello".to_string()))
            .await;
        frontend.emit(FrontendEvent::TurnFinished).await;
    }

    #[tokio::test]
    async fn test_silent_frontend_request_permission_denies() {
        let frontend = SilentFrontend;
        let outcome = frontend
            .request_permission(PermissionRequest {
                tool_name: "read_file".to_string(),
                primary_param: Some("/tmp/foo".to_string()),
                cancellation: tokio_util::sync::CancellationToken::new(),
            })
            .await;
        assert_eq!(outcome, PermissionOutcome::Deny);
    }

    #[tokio::test]
    async fn test_recording_frontend_records_events_in_order() {
        let frontend = RecordingFrontend::new();
        frontend.emit(FrontendEvent::TurnStarted).await;
        frontend
            .emit(FrontendEvent::AssistantTextDelta("hi".to_string()))
            .await;
        frontend.emit(FrontendEvent::TurnFinished).await;
        let events = frontend.events();
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], FrontendEvent::TurnStarted));
        assert!(matches!(events[1], FrontendEvent::AssistantTextDelta(ref s) if s == "hi"));
        assert!(matches!(events[2], FrontendEvent::TurnFinished));
    }

    #[tokio::test]
    async fn test_recording_frontend_returns_configured_permission_outcome() {
        let frontend = RecordingFrontend::with_permission(PermissionOutcome::Deny);
        let outcome = frontend
            .request_permission(PermissionRequest {
                tool_name: "execute_command".to_string(),
                primary_param: Some("rm -rf /".to_string()),
                cancellation: tokio_util::sync::CancellationToken::new(),
            })
            .await;
        assert_eq!(outcome, PermissionOutcome::Deny);
    }

    #[tokio::test]
    async fn test_permission_forwarding_frontend_drops_emits() {
        // Keep a typed handle alongside the trait-object Arc so we
        // can inspect the delegate's recorded events after the
        // wrapper drops the emits.
        let recorder = Arc::new(RecordingFrontend::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate);

        forwarder.emit(FrontendEvent::TurnStarted).await;
        forwarder
            .emit(FrontendEvent::AssistantTextDelta("ignored".into()))
            .await;

        assert!(
            recorder.events().is_empty(),
            "emit must not forward to the delegate",
        );
    }

    #[tokio::test]
    async fn test_permission_forwarding_frontend_delegates_request_permission() {
        let delegate: Arc<dyn Frontend> =
            Arc::new(RecordingFrontend::with_permission(PermissionOutcome::Allow));
        let forwarder = PermissionForwardingFrontend::new(delegate);
        let outcome = forwarder
            .request_permission(PermissionRequest {
                tool_name: "write_file".into(),
                primary_param: Some("/tmp/foo".into()),
                cancellation: tokio_util::sync::CancellationToken::new(),
            })
            .await;
        assert_eq!(outcome, PermissionOutcome::Allow);
    }

    #[tokio::test]
    async fn test_silent_frontend_default_delegate_methods_return_none() {
        // Default impls signal "no delegate available, do it locally".
        let frontend = SilentFrontend;
        assert!(
            frontend
                .delegate_fs_read(Path::new("/tmp/x"), None, None)
                .await
                .is_none()
        );
        assert!(
            frontend
                .delegate_fs_write(Path::new("/tmp/x"), "hi")
                .await
                .is_none()
        );
        assert!(
            frontend
                .delegate_execute(DelegatedExecSpec {
                    command: "true".to_string(),
                    args: Vec::new(),
                    env: Vec::new(),
                    cwd: None,
                    timeout: None,
                    output_byte_limit: None,
                    cancellation: tokio_util::sync::CancellationToken::new(),
                })
                .await
                .is_none()
        );
    }

    /// Test fixture that records what arguments each delegate method
    /// was called with, and lets the test pick the response.
    pub(super) struct DelegatingRecorder {
        pub fs_reads: Mutex<Vec<PathBuf>>,
        pub fs_writes: Mutex<Vec<(PathBuf, String)>>,
        pub execs: Mutex<Vec<DelegatedExecSpec>>,
        pub fs_read_response: Mutex<Option<Result<String, FrontendError>>>,
        pub fs_write_response: Mutex<Option<Result<(), FrontendError>>>,
        pub exec_response: Mutex<Option<Result<DelegatedExecOutput, FrontendError>>>,
    }

    impl DelegatingRecorder {
        fn new() -> Self {
            Self {
                fs_reads: Mutex::new(Vec::new()),
                fs_writes: Mutex::new(Vec::new()),
                execs: Mutex::new(Vec::new()),
                fs_read_response: Mutex::new(Some(Ok("from-delegate".to_string()))),
                fs_write_response: Mutex::new(Some(Ok(()))),
                exec_response: Mutex::new(Some(Ok(DelegatedExecOutput {
                    output: "delegate-out".to_string(),
                    exit_code: Some(0),
                    signal: None,
                    truncated: false,
                }))),
            }
        }
    }

    #[async_trait]
    impl Frontend for DelegatingRecorder {
        async fn emit(&self, _event: FrontendEvent) {}

        async fn request_permission(&self, _request: PermissionRequest) -> PermissionOutcome {
            PermissionOutcome::Allow
        }

        async fn delegate_fs_read(
            &self,
            path: &Path,
            _line: Option<u32>,
            _limit: Option<u32>,
        ) -> Option<Result<String, FrontendError>> {
            self.fs_reads.lock().unwrap().push(path.to_path_buf());
            self.fs_read_response.lock().unwrap().take()
        }

        async fn delegate_fs_write(
            &self,
            path: &Path,
            content: &str,
        ) -> Option<Result<(), FrontendError>> {
            self.fs_writes
                .lock()
                .unwrap()
                .push((path.to_path_buf(), content.to_string()));
            self.fs_write_response.lock().unwrap().take()
        }

        async fn delegate_execute(
            &self,
            spec: DelegatedExecSpec,
        ) -> Option<Result<DelegatedExecOutput, FrontendError>> {
            self.execs.lock().unwrap().push(spec);
            self.exec_response.lock().unwrap().take()
        }
    }

    #[tokio::test]
    async fn test_permission_forwarding_frontend_forwards_fs_read() {
        let recorder = Arc::new(DelegatingRecorder::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate);
        let outcome = forwarder
            .delegate_fs_read(Path::new("/tmp/sub.txt"), None, None)
            .await
            .expect("delegate result");
        assert_eq!(outcome.expect("ok"), "from-delegate");
        assert_eq!(recorder.fs_reads.lock().unwrap().as_slice(), &[
            PathBuf::from("/tmp/sub.txt")
        ],);
    }

    #[tokio::test]
    async fn test_permission_forwarding_frontend_forwards_fs_write() {
        let recorder = Arc::new(DelegatingRecorder::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate);
        forwarder
            .delegate_fs_write(Path::new("/tmp/sub.txt"), "hi from sub-agent")
            .await
            .expect("delegate result")
            .expect("ok");
        let recorded = recorder.fs_writes.lock().unwrap().clone();
        assert_eq!(recorded, vec![(
            PathBuf::from("/tmp/sub.txt"),
            "hi from sub-agent".to_string()
        )]);
    }

    #[tokio::test]
    async fn test_permission_forwarding_frontend_forwards_execute() {
        let recorder = Arc::new(DelegatingRecorder::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate);
        let spec = DelegatedExecSpec {
            command: "ls".to_string(),
            args: vec!["-la".to_string()],
            env: Vec::new(),
            cwd: Some(PathBuf::from("/tmp")),
            timeout: None,
            output_byte_limit: None,
            cancellation: tokio_util::sync::CancellationToken::new(),
        };
        let outcome = forwarder
            .delegate_execute(spec)
            .await
            .expect("delegate result")
            .expect("ok");
        assert_eq!(outcome.output, "delegate-out");
        assert_eq!(recorder.execs.lock().unwrap().len(), 1);
    }
}
