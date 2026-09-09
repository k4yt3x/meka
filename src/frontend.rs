//! `Frontend`: the swappable driver for agent output and approval round-trips.
//!
//! `Agent::run_turn` emits its user-facing output (streamed assistant text, thinking blocks,
//! tool-call indicators, todo lists, token usage) and its tool-approval requests through `Arc<dyn
//! Frontend>` instead of calling `render::*` and `std::sync::mpsc` directly. The REPL today is one
//! impl ([`crate::host::repl::frontend::ReplFrontend`]); ACP, a Telegram bridge, or a web UI become
//! additional impls without touching the agent core.
//!
//! This module owns the trait, the event/permission types, and the two UI-agnostic impls
//! ([`SilentFrontend`], [`PermissionForwardingFrontend`]). Concrete UI impls live with their UI
//! (`ReplFrontend` in `crate::host::repl::editor`, `AcpFrontend` in `crate::host::acp`) so the
//! abstraction layer never depends on a specific frontend by name.
//!
//! The event-based shape mirrors ACP's `session/update` notification: one channel for every kind
//! of agent-emitted output, discriminated by the [`FrontendEvent`] variant.

#[cfg(test)]
use std::path::PathBuf;
use std::{
    collections::{HashSet, VecDeque},
    path::Path,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use uuid::Uuid;

use crate::{stats::TokenUsage, todo::TodoItem};

/// How long a host with a client on the far side waits for an approval answer before denying.
///
/// One figure for ACP and HTTP, because the thing being waited on is the same on both: a human
/// reading a prompt and deciding. It is a backstop against a client that will never answer at all
/// (an editor whose UI thread has wedged, a headless harness that speaks the protocol but shows no
/// prompt), not a deadline on the user, so it is generous. A prompt still open after this long has
/// been abandoned, and the turn holding the session's runtime open for it blocks everything queued
/// behind it. The REPL has no such backstop: a human is at the keyboard, and Ctrl+D or `n` is how
/// they leave a prompt.
pub(crate) const APPROVAL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30 * 60);

/// Trait the agent loop talks through to surface output and ask the user to approve tool calls.
/// Implementations are responsible for rendering mode, newline spacing, and any inter-event
/// formatting.
#[async_trait]
pub(crate) trait Frontend: Send + Sync {
    /// Emit a one-way UI event. Implementations must tolerate any order of events but may assume
    /// `TurnStarted` precedes any per-turn activity and `TurnFinished` closes it.
    async fn emit(&self, event: FrontendEvent);

    /// Round-trip request for user approval of a tool call, sent when the session's approvals
    /// switch submits a call above its level. [`PermissionOutcome::Canceled`] is
    /// distinct from [`PermissionOutcome::Deny`]; it indicates the user canceled the enclosing
    /// turn (Ctrl+C, `session/cancel`), which ACP will surface later. Today's REPL collapses it to
    /// deny semantics.
    async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome;

    /// Delegate a file read to whatever filesystem the frontend owns (typically the ACP client's
    /// in-buffer view of the file).
    ///
    /// The file tools route on the [`DelegateFailure`] carried by [`Delegation::Failed`], under
    /// one rule:
    /// **[`DelegateFailure::UnservablePath`] means the local filesystem is the only route and is
    /// used; anything else means the frontend may own the file, and the operation fails rather
    /// than routing around it.** Which paths a frontend will serve is its own business and differs
    /// between editors, so meka models none of them -- it asks per path and believes the answer.
    ///
    /// `line` and `limit` follow ACP's 1-based line / line-count convention.
    async fn delegate_fs_read(
        &self,
        _path: &Path,
        _line: Option<u32>,
        _limit: Option<u32>,
    ) -> Delegation<String> {
        Delegation::Local
    }

    /// Delegate a file write. Same [`Delegation`] semantics as [`Self::delegate_fs_read`].
    async fn delegate_fs_write(&self, _path: &Path, _content: &str) -> Delegation<()> {
        Delegation::Local
    }

    /// Returns `true` if the frontend has observed that its client is no longer reachable (e.g. an
    /// ACP client has closed its stdio connection, so every `session/update` notification returns
    /// an error). The agent loop checks this at every loop iteration and short-circuits with
    /// [`crate::error::MekaError::Interrupted`] so it doesn't keep burning provider / MCP cycles
    /// for an audience that's gone away.
    ///
    /// REPL and silent frontends never disconnect in this sense, so the default `false` is correct
    /// for them.
    fn client_disconnected(&self) -> bool {
        false
    }

    /// Whether reasoning forwarded to this frontend is kept somewhere the user or client still has
    /// it after the turn moves on.
    ///
    /// Answers the one thing `content_started` needs to know about a
    /// [`FrontendEvent::ThinkingDelta`] and the agent cannot: would retrying the attempt
    /// deliver the same reasoning twice. Only the frontend knows -- the REPL renders deltas
    /// under `[thinking].show_content` and drops them otherwise, and the SSE stream forwards
    /// them only for a session that asked for reasoning.
    ///
    /// It matters more here than it would for the answer, because reasoning is the *first* thing a
    /// turn produces. Marking every turn that reasoned would refuse the retry for almost any
    /// mid-turn provider failure, which is the resilience an overloaded provider most needs; and
    /// not marking one whose reasoning a client is concatenating corrupts what it rebuilds.
    ///
    /// `false` by default, which is right for the frontends that drop reasoning entirely and for
    /// those that read whole blocks. The latter not because a failed attempt is silent -- a block
    /// can complete and the attempt fail after it -- but because neither emitter of
    /// [`FrontendEvent::ThinkingBlock`] can be followed by a retry: the streaming one marks the
    /// turn started in the same branch, and the non-streaming one re-emits a whole message only
    /// once the provider call has returned `Ok`, past its own retry loop.
    fn retains_reasoning(&self) -> bool {
        false
    }

    /// Handle an MCP `elicitation/create` request: the server asked the user for input (either a
    /// structured form or a URL-consent flow). The frontend is responsible for prompting the user
    /// and returning their response. The default impl declines and says so through a
    /// [`Notice`]: the safe behavior when no human is reachable (non-interactive subcommands,
    /// `SilentFrontend`, the test-only `RecordingFrontend`), and the same shape every concrete
    /// impl gives a decline it makes on the user's behalf, so a tool call that then fails is
    /// explained on whatever surface the frontend has.
    ///
    /// Called through the frontend on the tool call's [`crate::tools::ToolContext`]. Concrete
    /// impls today:
    /// [`crate::host::repl::frontend::ReplFrontend`] (routes through the REPL thread),
    /// `crate::host::acp::AcpFrontend` (issues ACP `elicitation/create`, declining when the client
    /// doesn't advertise the mode), and [`PermissionForwardingFrontend`] (hands a sub-agent's
    /// elicitation to the parent).
    async fn handle_elicitation(
        &self,
        prompt: crate::frontend::ElicitationPrompt,
    ) -> crate::frontend::ElicitationResponse {
        self.emit(FrontendEvent::Notice(Notice::elicitation_declined(
            &prompt.server_name,
            "nothing here can show a prompt",
        )))
        .await;
        crate::frontend::ElicitationResponse::Decline
    }
}

/// The allow and deny answers a user gave for the rest of the session, keyed on the tool name
/// alone.
///
/// One definition for every frontend that offers a sticky answer (`always` and `never` at the
/// REPL, `allow_always` and `reject_always` on ACP, `allow_always` and `deny_always` over HTTP), so
/// what such an answer covers cannot drift between hosts: every later call to that tool, whatever
/// its arguments, until the session ends. Held in memory with the frontend and never persisted.
#[derive(Debug, Default)]
pub(crate) struct StickyApprovals {
    remembered: Mutex<StickySets>,
}

#[derive(Debug, Default)]
struct StickySets {
    always_allowed: HashSet<String>,
    never_allowed: HashSet<String>,
}

impl StickyApprovals {
    /// The answer already given for `tool_name`, if any, so the user is not asked again.
    pub(crate) fn remembered(&self, tool_name: &str) -> Option<PermissionOutcome> {
        let sets = crate::sync::lock(&self.remembered);
        if sets.always_allowed.contains(tool_name) {
            Some(PermissionOutcome::Allow)
        } else if sets.never_allowed.contains(tool_name) {
            Some(PermissionOutcome::Deny)
        } else {
            None
        }
    }

    pub(crate) fn remember_allow(&self, tool_name: &str) {
        crate::sync::lock(&self.remembered)
            .always_allowed
            .insert(tool_name.to_string());
    }

    pub(crate) fn remember_deny(&self, tool_name: &str) {
        crate::sync::lock(&self.remembered)
            .never_allowed
            .insert(tool_name.to_string());
    }

    /// Forget every answer, for a frontend that outlives the session its answers were given in.
    pub(crate) fn clear(&self) {
        *crate::sync::lock(&self.remembered) = StickySets::default();
    }
}

/// Why a delegated operation failed, in the only distinction the routing rule turns on.
///
/// Editors differ in which paths they will serve -- some only the project they have open, some
/// anything absolute -- so meka does not model any editor's rule. It asks, and this is the answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DelegateFailure {
    /// The frontend will not serve this path at all (ACP `ResourceNotFound`). It holds no buffer
    /// for the file and never will, so the local filesystem is not a degraded substitute for the
    /// delegate here -- it is the same bytes, and the only route.
    UnservablePath,
    /// Anything else: transport, timeout, an internal error inside the client. The frontend may
    /// well own this file and hold unsaved changes for it, so falling back to the local filesystem
    /// could read stale bytes or overwrite the user's unsaved work.
    Transient,
    /// The user stopped the turn while the round-trip was outstanding, so no answer is coming.
    ///
    /// Routes like [`Self::Transient`] -- the local filesystem is not a substitute for an answer
    /// the frontend never gave -- but reads differently to a caller: nothing failed, and the tool
    /// should end as an interruption rather than report the client broken.
    Canceled,
}

tokio::task_local! {
    /// The cancellation token of the *call* running on this task, when that is not the session's
    /// current turn.
    ///
    /// A frontend that races client round-trips against cancellation has only one token to hand:
    /// the session's, rewritten at every turn start. That is right for a tool call inside the turn
    /// and wrong for one the agent detached with `background: true`, which owns a token of its own
    /// and is documented to outlive the turn that started it. Reading the session cell made a
    /// `session/cancel` on any *later* turn abandon a detached call's `fs/*` request without
    /// sending it, so the task failed with "interrupted" -- the opposite of the promise in
    /// `docs/book/src/usage/background.md`.
    ///
    /// A task-local rather than a parameter because the frontend is a shared `Arc` behind a trait
    /// whose delegation methods take no token, and threading one through would touch every impl and
    /// call site to serve one caller. Mirrors the sub-agent flag on
    /// [`crate::provider::Attribution`], which solves the same "which unit of work is this task"
    /// problem the same way.
    static CALL_CANCELLATION: tokio_util::sync::CancellationToken;
}

/// Run `future` with `token` as the cancellation any frontend delegation on this task should
/// honor.
pub(crate) async fn scope_call_cancellation<F: std::future::Future>(
    token: tokio_util::sync::CancellationToken,
    future: F,
) -> F::Output {
    CALL_CANCELLATION.scope(token, future).await
}

/// The detached call's token, if this task is running one.
pub(crate) fn current_call_cancellation() -> Option<tokio_util::sync::CancellationToken> {
    CALL_CANCELLATION.try_with(Clone::clone).ok()
}

/// What a frontend answered when asked to serve a file operation. The three answers route
/// differently in the file tools, so they are three variants rather than an `Option<Result>` whose
/// `None` reads as "nothing happened".
#[derive(Debug, Clone)]
pub(crate) enum Delegation<T> {
    /// The frontend served it.
    Served(T),
    /// The frontend was asked and failed; the [`DelegateFailure`] says whether the local
    /// filesystem may stand in.
    Failed(FrontendError),
    /// The frontend has no file delegate, so the local filesystem is the only route.
    Local,
}

impl<T> Delegation<T> {
    #[cfg(test)]
    pub(crate) fn served(self) -> Option<T> {
        match self {
            Self::Served(value) => Some(value),
            Self::Failed(_) | Self::Local => None,
        }
    }
}

/// Error from a frontend-delegated operation ([`Frontend::delegate_fs_read`],
/// [`Frontend::delegate_fs_write`]). Carries the underlying transport's message in a stringly form
/// so tools can splice it into their `ToolOutput` text without depending on the transport crate,
/// plus the [`DelegateFailure`] the routing rule needs.
#[derive(Debug, Clone)]
pub(crate) struct FrontendError {
    message: String,
    failure: DelegateFailure,
}

impl FrontendError {
    /// Construct a [`DelegateFailure::Transient`] error -- the conservative default, since it is
    /// the classification that never routes around the frontend.
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            failure: DelegateFailure::Transient,
        }
    }

    /// Construct a [`DelegateFailure::UnservablePath`] error. Only a transport that can tell the
    /// two apart on the wire may call this.
    pub(crate) fn unservable_path(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            failure: DelegateFailure::UnservablePath,
        }
    }

    /// Construct a [`DelegateFailure::Canceled`] error. `what` names the round-trip that was
    /// abandoned, e.g. `"fs/read_text_file"`.
    pub(crate) fn canceled(what: &str) -> Self {
        Self {
            message: format!("{what} was abandoned: the turn was canceled"),
            failure: DelegateFailure::Canceled,
        }
    }

    /// Whether the local filesystem is a safe route for this path: true only when the frontend
    /// said it cannot serve the path at all.
    pub(crate) fn is_unservable_path(&self) -> bool {
        self.failure == DelegateFailure::UnservablePath
    }

    /// Whether this is the turn being stopped rather than a delegation failing. Callers turn it
    /// into [`crate::error::MekaError::Interrupted`] instead of a tool error, so stopping a turn
    /// mid-`fs/*` reads as a stop and not as a broken client.
    pub(crate) fn is_canceled(&self) -> bool {
        self.failure == DelegateFailure::Canceled
    }
}

impl std::fmt::Display for FrontendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for FrontendError {}

/// One-way UI event emitted by the agent loop.
#[derive(Debug, Clone)]
pub(crate) enum FrontendEvent {
    /// A new session was created. Carries the session UUID.
    SessionStarted { id: Uuid },
    /// The agent is about to start a turn. Carries no spacing duty: the `[display]` blanks bracket
    /// the *episode* between two prompts, which is `crate::console`'s to own, and a turn is only
    /// one of the things an episode may contain.
    TurnStarted,
    /// The agent finished a turn cleanly. The REPL closes any open streaming text block on this, as
    /// a block boundary; the closing blank belongs to the episode, not to the turn.
    TurnFinished,
    /// The turn took its own prompt back: it ended before anything from the model reached the
    /// conversation, and the prompt's retention said to; see
    /// [`crate::conversation::PromptRetention`]. Emitted from the places a prompt is withdrawn,
    /// ahead of the host's terminal event, so a client that resends can be told whether the
    /// conversation still holds the copy it sent.
    PromptWithdrawn,
    /// A streamed chunk of assistant text. Multiple deltas concatenate into one logical text run;
    /// any non-text event closes the run.
    AssistantTextDelta(String),
    /// The model is thinking, with the server's running token estimate when it offers one.
    ///
    /// A transient indicator, not content: it is expected to be drawn in place and erased when
    /// anything else prints. Exists because thinking is frequently *silent* (under Claude's
    /// `redact-thinking` beta or display updates no thinking text is ever streamed), so without
    /// this the reasoning phase is an unexplained pause, and a long one on a hard prompt.
    ThinkingProgress { estimated_tokens: Option<u64> },
    /// A thinking block closed without any text to show for it.
    ///
    /// Emitted instead of [`Self::ThinkingBlock`] when the block carried no readable content, which
    /// is every block under Claude's `redact-thinking` beta or display updates. Frontends drawing a
    /// transient indicator use it to close that indicator out at the moment the block actually
    /// ends, rather than leaving the line open until some later event happens to arrive -- a
    /// turn that errors or is interrupted emits no further events at all, and the error text
    /// would land on the indicator's line.
    ThinkingEnded,
    /// A streamed chunk of reasoning, as text to show. Multiple deltas concatenate into one block,
    /// which [`Self::ThinkingBlock`] then closes.
    ///
    /// Emitted for **every** block that carries readable text, including the ones a non-streaming
    /// provider hands back whole, which are forwarded here as a single delta. That invariant is
    /// what lets each frontend pick a lane once and hold it: a consumer either renders the deltas
    /// or renders the block, and never has to remember which of the two a given block produced.
    ThinkingDelta(String),
    /// A complete thinking block, as text to show. Emitted after the provider's `ThinkingComplete`
    /// stream event, and always preceded by the [`Self::ThinkingDelta`]s carrying the same text.
    ///
    /// Still emitted for a consumer that wants reasoning whole -- ACP thought chunks, the blocking
    /// HTTP response, the one-line preview under `show_content = false` -- and as the end-of-block
    /// marker for one that streamed it.
    ///
    /// Deliberately carries no opaque half. No `signature` against a future replay: replay reads
    /// the conversation log, where the block keeps its provider-tagged
    /// [`crate::conversation::OpaqueReasoning`]. A bare blob here would be an undiscriminated
    /// Claude MAC or OpenAI sealed reasoning, which is the conflation that shape exists to
    /// prevent, and it cloned kilobytes per block for a reader that never came.
    ThinkingBlock { content: String },
    /// The model has started composing a tool call: the name has arrived, the arguments have not.
    ///
    /// Pairs with [`Self::ToolCallStarted`] on the same `id`, which marks the end of composition
    /// and carries the arguments. The interval between the two is the time the model spent
    /// generating them, and for a tool whose argument *is* the user-visible text -- a chat
    /// bridge's `send_message`, say -- it is the only signal on the stream that a reply is
    /// being written rather than more work being done. Nothing else distinguishes the two:
    /// assistant text is usually narration *around* a call, and `ToolCallStarted` fires once
    /// the text is already finished.
    ///
    /// Streamed turns only, and unpaired if the turn dies. Under `--no-stream` the provider hands
    /// back each call whole, so there is no composition to report and this never fires; and a turn
    /// that fails or is canceled mid-block emits this with no `ToolCallStarted` after it, so a
    /// consumer holding state per `id` has to close it on the turn's terminal event as well.
    ToolCallComposing { id: String, name: String },
    /// A tool call is about to be dispatched. `id` is the `tool_use_id` assigned by the provider;
    /// frontends use it to correlate this announcement with the matching
    /// [`Self::ToolCallCompleted`]. `display_summary` is the agent-resolved primary argument for
    /// display (e.g. the path for `read_file`, the command for `execute_command`), pre-computed via
    /// [`crate::tools::resolve_primary_param`] so frontends don't need the tool's JSON Schema to
    /// render the indicator. `None` means "no obvious primary arg". Render the bare tool name.
    ToolCallStarted {
        id: String,
        name: String,
        input: serde_json::Value,
        display_summary: Option<String>,
    },
    /// A previously-announced tool call has finished. Emitted once per tool in source order after
    /// the parallel dispatch settles. The REPL impl ignores this today (tool results render through
    /// the model's next assistant message); the ACP impl translates it to `session/update:
    /// tool_call_update` with `status: completed | failed`.
    ToolCallCompleted {
        id: String,
        /// Tool name (matches the [`Self::ToolCallStarted`] `name`). Lets a frontend format the
        /// output per tool, e.g. wrapping `execute_command` output in a console code block.
        name: String,
        is_error: bool,
        content: Vec<crate::conversation::ToolResultContent>,
        /// Tool-specific structured side-channel. `edit_file` / `write_file` populate
        /// [`ToolOutputMetadata::Diff`] so ACP can emit a proper `diff` content block (and Zed can
        /// render its apply-diff UI). `None` for tools that have nothing extra.
        metadata: Option<ToolOutputMetadata>,
    },
    /// Output a still-running tool has produced so far, as it arrives. `id` matches the
    /// [`Self::ToolCallStarted`] `id`. Only `execute_command` emits these today: a build or a test
    /// run is silent for its whole duration otherwise, since [`Self::ToolCallCompleted`] can't fire
    /// until the process exits.
    ///
    /// A delta, not a snapshot, so the emitter doesn't have to hold the whole output. Frontends
    /// that render a replace-the-content protocol (ACP `tool_call_update`) accumulate it
    /// themselves, which is also where throttling belongs -- the emitter fires per read syscall
    /// and has no idea what a given transport costs.
    ///
    /// stdout and stderr arrive interleaved in production order, the way a terminal shows them.
    /// The model-facing result assembled at completion still separates the two streams.
    ToolCallOutputDelta { id: String, chunk: String },
    /// The shared todo list changed via the `todo` tool. Emitted by the agent loop after the tool
    /// succeeds and only when the rendered state actually changed; the REPL renders the list and
    /// the agent's per-turn `OutputSpacing` is advanced. `title` is the heading the agent set
    /// for the list.
    TodoListUpdated {
        title: Option<String>,
        items: Vec<TodoItem>,
    },
    /// A sub-agent running under the `agent_spawn` tool call `tool_call_id` did something worth
    /// showing. `summary` is the *whole* rolling activity block, not a delta, because ACP's
    /// `tool_call_update` replaces a tool call's content rather than appending to it.
    ///
    /// Emitted by [`PermissionForwardingFrontend`], which is the only place that knows both the
    /// sub-agent's events and the parent tool call they belong to.
    SubAgentActivity {
        tool_call_id: String,
        summary: String,
    },
    /// End-of-turn token-usage summary.
    TokenUsage(TokenUsage),
    /// User-visible advisory raised by meka itself or by the provider layer (e.g. image redaction
    /// when the request body would exceed the API limit). `ReplFrontend` renders it in the color
    /// its [`NoticeLevel`] asks for; `AcpFrontend` forwards it as an `AgentMessageChunk` with a
    /// `[meka] ` or `[meka warn] ` prefix so the editor's transcript records the side-effect; the
    /// HTTP and one-shot JSON surfaces carry it as a [`NoticeView`]. `SilentFrontend` drops it.
    Notice(crate::frontend::Notice),
    /// Incremental progress from an in-flight MCP tool (`notifications/progress`). Routed
    /// per-session via the frontend the MCP call registered with its progress token. See
    /// [`crate::mcp::progress::ProgressRegistry`]. `ReplFrontend` renders an inline status line
    /// (carriage-return overwrite); `HttpFrontend` streams it as a `progress` SSE event;
    /// `AcpFrontend` logs at `info!` today (no protocol primitive yet). `SilentFrontend` drops
    /// them.
    McpProgress(crate::frontend::ProgressUpdate),
    /// The conversation was just summarized and the window replaced.
    ///
    /// Emitted for every compaction whatever triggered it, including the automatic ones that fire
    /// mid-turn without anyone asking. A frontend holding its own view of the transcript needs this
    /// because compaction is the one event that makes messages it already rendered stop existing:
    /// the REPL and ACP re-read from the agent each turn and so never noticed, but an HTTP client
    /// polling `GET /messages` sees `total` shrink with no explanation unless it is told.
    Compacted {
        /// `checkpoint`, `checkpoint_text`, or `summarizer`. The three differ in fidelity, not
        /// just mechanism.
        source: &'static str,
        /// How many materialized messages the boundary removed.
        ///
        /// The whole pre-compaction window, including the recent tail that is then re-appended
        /// verbatim -- so this over-counts what the summary actually stands for. It is the figure
        /// the replay algorithm uses, and the one the `compaction` marker on `GET /messages`
        /// reports, so the two always agree.
        replaced_count: usize,
        /// Which compaction this was, counting from 1.
        generation: u64,
    },
}

/// Structured side-channel a tool can attach to its [`crate::tools::ToolOutput`] for frontends that
/// know how to render it. Frontends that don't understand a variant ignore it (the regular
/// `content` text is still the source of truth for the model and the REPL).
#[derive(Debug, Clone)]
pub(crate) enum ToolOutputMetadata {
    /// Pre/post file content produced by `edit_file` / `write_file`. `old_text == None` means the
    /// file did not exist before the call (the write created it).
    Diff {
        path: std::path::PathBuf,
        old_text: Option<String>,
        new_text: String,
    },
    /// How an `execute_command` child ended. The exit code is already spelled out in the tool text
    /// for the model, but a frontend rendering a terminal needs it as a number, and parsing it back
    /// out of the prose would be a guess. `exit_code == None` with a `signal` means the process was
    /// killed; both `None` means it never got far enough to have either (timeout, spawn failure).
    CommandExit {
        exit_code: Option<i32>,
        signal: Option<String>,
    },
}

/// Round-trip request for tool-call approval.
#[derive(Debug, Clone)]
pub(crate) struct PermissionRequest {
    pub(crate) tool_name: String,
    /// The most user-meaningful argument for display in the prompt (e.g. the file path for
    /// `read_file`, the command for `execute_command`). Resolved via
    /// [`crate::tools::resolve_primary_param`]. Still the right shape for a client that wants one
    /// line: the ACP frontend builds its permission title from it.
    pub(crate) primary_param: Option<String>,
    /// Every argument the tool was called with, plus `background` when the call would detach.
    ///
    /// `background` is meka's own parameter and is taken out before any tool sees it, but it
    /// decides whether the call outlives the turn, so it is put back for the asking.
    ///
    /// `primary_param` alone is not enough to authorize a call. It resolves to the *destination*
    /// for every write-shaped tool -- `path` for `write_file` and `edit_file`, `url` for
    /// `fetch_url`, `name` for `scratchpad_write` -- so a prompt built from it asks the user
    /// to approve a write without showing them what is being written. A frontend that gates on
    /// human judgment should render this instead.
    pub(crate) input: serde_json::Value,
    /// Per-turn cancellation token. ACP frontends race their `session/request_permission`
    /// round-trip against this so a `session/cancel` during an approval prompt resolves promptly
    /// instead of hanging until the client replies.
    pub(crate) cancellation: tokio_util::sync::CancellationToken,
}

/// Outcome of a [`Frontend::request_permission`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PermissionOutcome {
    Allow,
    Deny,
    /// The enclosing turn was canceled while the request was in flight. The ACP frontend surfaces
    /// this as `{outcome: canceled}`; the REPL collapses it to a deny-shaped tool error.
    Canceled,
}

/// Frontend wrapper used by sub-agents when the parent is interactive enough to host permission
/// prompts. Streaming output (text, thinking, todos, token usage) is dropped; sub-agents' final
/// reports flow back through the parent's `agent_spawn` tool result, not through this frontend.
/// The exceptions are:
///
/// - `Notice`: provider-side advisories the user should still see, e.g. a redaction during a
///   sub-agent's turn.
/// - `request_permission` and `handle_elicitation`: round-trips forwarded so the user is asked in
///   their original UI (REPL approval line, ACP `session/request_permission` /
///   `elicitation/create`).
/// - `ToolCallStarted`: not forwarded as-is, but rolled up into [`FrontendEvent::SubAgentActivity`]
///   against the parent's `agent_spawn` call so a long delegated task shows its progress instead of
///   an opaque spinner.
///
/// Constructed in [`crate::tools::subagent::AgentSpawnTool`] with the parent agent's frontend as
/// the delegate.
pub(crate) struct PermissionForwardingFrontend {
    delegate: Arc<dyn Frontend>,
    /// The parent's `tool_use_id` for the `agent_spawn` call this sub-agent is running under, when
    /// one is in scope. `None` outside a tool call (tests, direct construction), which disables
    /// activity forwarding rather than guessing at a correlation id.
    tool_call_id: Option<String>,
    /// Rolling record of what the sub-agent has done, oldest first, capped at
    /// [`Self::MAX_ACTIVITY_LINES`].
    activity: std::sync::Mutex<VecDeque<String>>,
}

impl PermissionForwardingFrontend {
    /// How many activity lines the parent tool call shows. The whole block is resent on every
    /// update (ACP replaces content), so this bounds per-update payload as well as height.
    const MAX_ACTIVITY_LINES: usize = 20;

    pub(crate) fn new(delegate: Arc<dyn Frontend>, tool_call_id: Option<String>) -> Self {
        Self {
            delegate,
            tool_call_id,
            activity: std::sync::Mutex::new(VecDeque::new()),
        }
    }

    /// Append `line` and return the whole block to send.
    fn record_activity(&self, line: String) -> String {
        let mut activity = crate::sync::lock(&self.activity);
        if activity.len() == Self::MAX_ACTIVITY_LINES {
            activity.pop_front();
        }
        activity.push_back(line);
        activity
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[async_trait]
impl Frontend for PermissionForwardingFrontend {
    async fn emit(&self, event: FrontendEvent) {
        match event {
            // Provider advisories about the sub-agent's request belong in the user's primary UI.
            FrontendEvent::Notice(_) => self.delegate.emit(event).await,
            // Roll the sub-agent's tool calls up into the parent's `agent_spawn` tool call, so a
            // long-running sub-agent shows what it is doing instead of an opaque spinner. Only the
            // call being *started* is recorded: it answers "where is it now", which is the
            // question an unattended run leaves open, and results still arrive in the report.
            FrontendEvent::ToolCallStarted {
                name,
                display_summary,
                ..
            } => {
                // Without a parent call there is nothing to attach the activity to, so it is
                // dropped rather than sent against a guessed correlation id.
                let Some(tool_call_id) = self.tool_call_id.clone() else {
                    return;
                };
                let line = match display_summary {
                    Some(summary) => format!("{name}: {summary}"),
                    None => name,
                };
                let summary = self.record_activity(line);
                self.delegate
                    .emit(FrontendEvent::SubAgentActivity {
                        tool_call_id,
                        summary,
                    })
                    .await;
            }
            // A nested sub-agent's activity is already summarized as a `agent_spawn` line in this
            // sub-agent's own record; forwarding it too would have two writers fighting over one
            // tool call's content.
            FrontendEvent::SubAgentActivity { .. } => {}
            // Everything else (text deltas, thinking, tool results, todos, token usage, session
            // lifecycle) is sub-agent chrome the user shouldn't see.
            //
            // [`FrontendEvent::ToolCallOutputDelta`] and [`FrontendEvent::ToolCallComposing`] must
            // stay in here rather than being forwarded. Both are keyed by the sub-agent's own
            // `tool_use_id`, which names no tool call the client has been told about, so a frontend
            // that holds state per call would accumulate an entry that nothing ever completes and
            // frees. Composing is the worse of the two: the event that closes it is this level's
            // `ToolCallStarted`, which the arm above turns into a `SubAgentActivity` -- so the id
            // would never be heard from again, and an indicator opened on it would never come down.
            _ => {}
        }
    }

    fn client_disconnected(&self) -> bool {
        // Sub-agents must observe the parent's disconnect so their own run_turn loop
        // short-circuits; without this forward, a sub-agent under a dropped ACP connection
        // keeps burning provider tokens.
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
    ) -> Delegation<String> {
        self.delegate.delegate_fs_read(path, line, limit).await
    }

    async fn delegate_fs_write(&self, path: &Path, content: &str) -> Delegation<()> {
        self.delegate.delegate_fs_write(path, content).await
    }

    /// Forwarded for the same reason as [`Self::request_permission`]: an MCP server called from a
    /// sub-agent needs to reach the same human, and the trait default would decline on their
    /// behalf without ever asking.
    async fn handle_elicitation(
        &self,
        prompt: crate::frontend::ElicitationPrompt,
    ) -> crate::frontend::ElicitationResponse {
        self.delegate.handle_elicitation(prompt).await
    }
}

/// Fully-silent frontend: drops every emit and denies every permission request. Used by tests and
/// `meka tools list`'s reference registry. Both want a frontend that never reaches out to a user.
/// Sub-agents use [`PermissionForwardingFrontend`] instead so their permission prompts surface in
/// the parent's UI.
pub(crate) struct SilentFrontend;

#[async_trait]
impl Frontend for SilentFrontend {
    async fn emit(&self, _event: FrontendEvent) {}

    async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome {
        // Said through `emit` even though this frontend drops it: the decision reads the same at
        // every door, and the silence is this sink's doing rather than the refusal's.
        self.emit(FrontendEvent::Notice(
            Notice::approval_refused_without_asking(&request.tool_name),
        ))
        .await;
        PermissionOutcome::Deny
    }
}

/// Severity hint for a provider-emitted [`Notice`]. Frontends can map these to per-level styling
/// (a dim hint for `Info`, a warn-colored line for `Warn`). `Info` carries the image-redaction
/// notice; `Warn` carries recoverable conditions the user should see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NoticeLevel {
    Info,
    Warn,
}
impl NoticeLevel {
    /// The level's spelling on every JSON surface (HTTP responses, SSE events, one-shot reports).
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warn => "warn",
        }
    }
}
/// A [`Notice`] as every JSON surface serializes it: `level` is [`NoticeLevel::name`].
#[derive(Debug, Clone, serde::Serialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
pub(crate) struct NoticeView {
    pub(crate) level: String,
    pub(crate) text: String,
}
impl From<Notice> for NoticeView {
    fn from(notice: Notice) -> Self {
        Self {
            level: notice.level.name().to_string(),
            text: notice.text,
        }
    }
}
/// User-visible advisory surfaced by a provider during a request. Frontends format the message
/// themselves; the one structured payload is for the agent, not for display.
#[derive(Debug, Clone)]
pub(crate) struct Notice {
    pub(crate) level: NoticeLevel,
    pub(crate) text: String,
    /// Set when the advisory reports an image-redaction pass, so the agent can count it against
    /// the session the request belonged to.
    pub(crate) redaction: Option<crate::stats::Redaction>,
}
impl Notice {
    pub(crate) fn info(text: impl Into<String>) -> Self {
        Self {
            level: NoticeLevel::Info,
            text: text.into(),
            redaction: None,
        }
    }

    pub(crate) fn warn(text: impl Into<String>) -> Self {
        Self {
            level: NoticeLevel::Warn,
            text: text.into(),
            redaction: None,
        }
    }

    pub(crate) fn reporting(mut self, redaction: crate::stats::Redaction) -> Self {
        self.redaction = Some(redaction);
        self
    }

    /// What every frontend says when a call needs approval and nothing can put the question to a
    /// human: the one-shot run, the silent frontend, a REPL whose prompt thread is gone.
    ///
    /// A `warn`, and a notice rather than a log line, because the run is otherwise
    /// indistinguishable from a model that chose not to use its tools, and that has sent people
    /// debugging the prompt instead of the flag.
    pub(crate) fn approval_refused_without_asking(tool_name: &str) -> Self {
        Self::warn(Self::approval_refused_without_asking_text(tool_name))
    }

    /// [`Self::approval_refused_without_asking`] with the host's reason and remedy appended, for a
    /// surface where the caller could have provided a channel and did not.
    pub(crate) fn approval_refused_without_asking_because(tool_name: &str, reason: &str) -> Self {
        Self::warn(format!(
            "{}: {reason}",
            Self::approval_refused_without_asking_text(tool_name)
        ))
    }

    fn approval_refused_without_asking_text(tool_name: &str) -> String {
        format!(
            "approvals are on but nobody can answer here, so '{tool_name}' was refused without \
             asking"
        )
    }

    /// What every frontend says when it declines an MCP elicitation on the user's behalf. `reason`
    /// is the host's: no prompt to show, a mode the client did not advertise, a schema the
    /// protocol cannot express.
    pub(crate) fn elicitation_declined(server_name: &str, reason: &str) -> Self {
        Self::warn(format!(
            "MCP elicitation from '{server_name}' declined: {reason}"
        ))
    }
}

/// User-facing payload the frontend renders.
#[derive(Debug)]
pub(crate) struct ElicitationPrompt {
    pub(crate) server_name: String,
    pub(crate) kind: ElicitationKind,
    pub(crate) message: String,
}
#[derive(Debug)]
pub(crate) enum ElicitationKind {
    /// Structured form: the server sent a JSON schema of fields to fill.
    Form { schema: serde_json::Value },
    /// URL consent: the server wants the user to visit a URL (e.g. to log in to a third-party
    /// service).
    Url { url: String },
}
/// Frontend's response back to the MCP handler.
#[derive(Debug, Clone)]
pub(crate) enum ElicitationResponse {
    Accept { content: Option<serde_json::Value> },
    Decline,
    Cancel,
}

/// Single progress update forwarded to the frontend.
#[derive(Clone, Debug)]
pub(crate) struct ProgressUpdate {
    pub(crate) server_name: String,
    pub(crate) tool_name: String,
    /// The provider's `tool_use_id` for the in-flight call, when one was supplied, so a client
    /// holding state per call (the SSE `progress` event's readers) can attach the update to the
    /// `tool_call.executing` it belongs to.
    pub(crate) tool_use_id: Option<String>,
    pub(crate) progress: f64,
    pub(crate) total: Option<f64>,
    pub(crate) message: Option<String>,
}
#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    pub(crate) struct RecordingFrontend {
        events: Mutex<Vec<FrontendEvent>>,
        permission_response: Mutex<PermissionOutcome>,
        /// Tool names this frontend was asked to approve, in order. Lets a test assert that a gate
        /// actually ran, rather than only that its outcome was survivable.
        permission_requests: Mutex<Vec<String>>,
    }

    impl RecordingFrontend {
        pub(crate) fn new() -> Self {
            Self {
                events: Mutex::new(Vec::new()),
                permission_response: Mutex::new(PermissionOutcome::Allow),
                permission_requests: Mutex::new(Vec::new()),
            }
        }

        pub(crate) fn with_permission(response: PermissionOutcome) -> Self {
            Self {
                events: Mutex::new(Vec::new()),
                permission_response: Mutex::new(response),
                permission_requests: Mutex::new(Vec::new()),
            }
        }

        pub(crate) fn events(&self) -> Vec<FrontendEvent> {
            self.events.lock().unwrap().clone()
        }

        pub(crate) fn permission_requests(&self) -> Vec<String> {
            self.permission_requests.lock().unwrap().clone()
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

        async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome {
            self.permission_requests
                .lock()
                .unwrap()
                .push(request.tool_name);
            self.permission_response.lock().unwrap().clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{testing::RecordingFrontend, *};

    #[tokio::test]
    async fn silent_frontend_emit_is_no_op_and_does_not_panic() {
        let frontend = SilentFrontend;
        frontend.emit(FrontendEvent::TurnStarted).await;
        frontend
            .emit(FrontendEvent::AssistantTextDelta("hello".to_string()))
            .await;
        frontend.emit(FrontendEvent::TurnFinished).await;
    }

    #[tokio::test]
    async fn silent_frontend_request_permission_denies() {
        let frontend = SilentFrontend;
        let outcome = frontend
            .request_permission(PermissionRequest {
                tool_name: "read_file".to_string(),
                primary_param: Some("/tmp/foo".to_string()),
                input: serde_json::json!({"path": "/tmp/foo"}),
                cancellation: tokio_util::sync::CancellationToken::new(),
            })
            .await;
        assert_eq!(outcome, PermissionOutcome::Deny);
    }

    /// The trait default is what every frontend without a prompt of its own answers with, so the
    /// decline it makes on the user's behalf has to be said where that frontend says things. The
    /// recorder does not override the method, so what it records is the default's doing.
    #[tokio::test]
    async fn the_default_elicitation_decline_is_announced_as_a_warn_notice() {
        let frontend = RecordingFrontend::new();
        let response = frontend
            .handle_elicitation(ElicitationPrompt {
                server_name: "notion".to_string(),
                message: "authorize?".to_string(),
                kind: ElicitationKind::Url {
                    url: "https://example.com/".to_string(),
                },
            })
            .await;
        assert!(matches!(response, ElicitationResponse::Decline));
        let events = frontend.events();
        assert!(
            matches!(
                events.as_slice(),
                [FrontendEvent::Notice(notice)]
                    if notice.level == NoticeLevel::Warn && notice.text.contains("'notion'")
            ),
            "the decline must be said once, at warn, naming the server: {events:?}"
        );
    }

    /// A sticky answer covers the tool, not the call, and a session starting over forgets it.
    #[test]
    fn a_sticky_answer_covers_every_later_call_to_the_tool_until_cleared() {
        let sticky = StickyApprovals::default();
        assert_eq!(sticky.remembered("write_file"), None);
        sticky.remember_allow("write_file");
        sticky.remember_deny("execute_command");
        assert_eq!(
            sticky.remembered("write_file"),
            Some(PermissionOutcome::Allow)
        );
        assert_eq!(
            sticky.remembered("execute_command"),
            Some(PermissionOutcome::Deny)
        );
        assert_eq!(sticky.remembered("read_file"), None);
        sticky.clear();
        assert_eq!(sticky.remembered("write_file"), None);
        assert_eq!(sticky.remembered("execute_command"), None);
    }

    /// The activity record is a display aid; a panic that poisoned its lock elsewhere must not
    /// blank the parent's view of the sub-agent for the rest of the run.
    #[test]
    fn activity_is_still_recorded_after_the_lock_was_poisoned() {
        let frontend =
            PermissionForwardingFrontend::new(Arc::new(SilentFrontend), Some("call-1".to_string()));
        std::thread::scope(|scope| {
            let poisoner = scope.spawn(|| {
                let _held = frontend.activity.lock().expect("not yet poisoned");
                panic!("poison the activity lock");
            });
            assert!(poisoner.join().is_err(), "the panic is the point");
        });
        assert!(frontend.activity.is_poisoned());

        assert_eq!(
            frontend.record_activity("read_file: a.txt".to_string()),
            "read_file: a.txt"
        );
    }

    #[tokio::test]
    async fn recording_frontend_records_events_in_order() {
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

    /// The agent pre-resolves the primary argument through [`crate::tools::resolve_primary_param`]
    /// and ships it on [`FrontendEvent::ToolCallStarted`], so no frontend needs the tool's JSON
    /// Schema. End-to-end emission from `Agent::run_turn` is covered by `tests/acp.rs`.
    #[tokio::test]
    async fn tool_call_started_carries_resolved_display_summary() {
        let recorder = RecordingFrontend::new();
        let input = serde_json::json!({"path": "/etc/hosts"});
        let display_summary = crate::tools::resolve_primary_param("read_file", &input, None);
        assert_eq!(display_summary.as_deref(), Some("/etc/hosts"));
        recorder
            .emit(FrontendEvent::ToolCallStarted {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                input: input.clone(),
                display_summary: display_summary.clone(),
            })
            .await;
        let events = recorder.events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            FrontendEvent::ToolCallStarted {
                display_summary, ..
            } => {
                assert_eq!(display_summary.as_deref(), Some("/etc/hosts"));
            }
            other => panic!("expected ToolCallStarted; got {other:?}"),
        }
    }

    #[tokio::test]
    async fn recording_frontend_returns_configured_permission_outcome() {
        let frontend = RecordingFrontend::with_permission(PermissionOutcome::Deny);
        let outcome = frontend
            .request_permission(PermissionRequest {
                tool_name: "execute_command".to_string(),
                primary_param: Some("rm -rf /".to_string()),
                input: serde_json::json!({"command": "rm -rf /"}),
                cancellation: tokio_util::sync::CancellationToken::new(),
            })
            .await;
        assert_eq!(outcome, PermissionOutcome::Deny);
    }

    #[tokio::test]
    async fn permission_forwarding_frontend_drops_sub_agent_chrome() {
        // Sub-agent chrome (text, lifecycle, tool indicators) must NOT bubble up to the parent's
        // UI; the sub-agent's report flows back via the agent_spawn tool result instead.
        let recorder = Arc::new(RecordingFrontend::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate, None);

        forwarder.emit(FrontendEvent::TurnStarted).await;
        forwarder
            .emit(FrontendEvent::AssistantTextDelta("ignored".into()))
            .await;

        assert!(
            recorder.events().is_empty(),
            "sub-agent chrome must not forward to the delegate",
        );
    }

    #[tokio::test]
    async fn permission_forwarding_frontend_rolls_up_sub_agent_tool_calls() {
        let recorder = Arc::new(RecordingFrontend::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate, Some("toolu_parent".into()));

        for (name, summary) in [
            ("read_file", Some("/etc/hosts".to_string())),
            ("find_files", Some("**/*.rs".to_string())),
            ("todo", None),
        ] {
            forwarder
                .emit(FrontendEvent::ToolCallStarted {
                    id: format!("toolu_{name}"),
                    name: name.to_string(),
                    input: serde_json::json!({}),
                    display_summary: summary,
                })
                .await;
        }

        let events = recorder.events();
        assert_eq!(events.len(), 3, "one update per sub-agent tool call");
        // Each update carries the whole block, not a delta: ACP replaces tool call content.
        match &events[2] {
            FrontendEvent::SubAgentActivity {
                tool_call_id,
                summary,
            } => {
                assert_eq!(tool_call_id, "toolu_parent");
                assert_eq!(summary, "read_file: /etc/hosts\nfind_files: **/*.rs\ntodo");
            }
            other => panic!("expected SubAgentActivity; got {other:?}"),
        }
    }

    #[tokio::test]
    async fn permission_forwarding_frontend_caps_activity_lines() {
        let recorder = Arc::new(RecordingFrontend::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate, Some("toolu_parent".into()));

        let total = PermissionForwardingFrontend::MAX_ACTIVITY_LINES + 5;
        for i in 0..total {
            forwarder
                .emit(FrontendEvent::ToolCallStarted {
                    id: format!("toolu_{i}"),
                    name: "read_file".to_string(),
                    input: serde_json::json!({}),
                    display_summary: Some(format!("/file{i}")),
                })
                .await;
        }

        let events = recorder.events();
        let FrontendEvent::SubAgentActivity { summary, .. } = events.last().expect("an event")
        else {
            panic!("expected SubAgentActivity");
        };
        let lines: Vec<&str> = summary.lines().collect();
        assert_eq!(
            lines.len(),
            PermissionForwardingFrontend::MAX_ACTIVITY_LINES
        );
        assert_eq!(
            lines[0],
            format!(
                "read_file: /file{}",
                total - PermissionForwardingFrontend::MAX_ACTIVITY_LINES
            ),
            "the oldest lines are dropped, not the newest"
        );
    }

    #[tokio::test]
    async fn permission_forwarding_frontend_without_tool_call_id_stays_silent() {
        // Outside a tool call there is nothing to correlate the activity with, so it is dropped
        // rather than sent against a guessed id.
        let recorder = Arc::new(RecordingFrontend::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate, None);

        forwarder
            .emit(FrontendEvent::ToolCallStarted {
                id: "toolu_1".into(),
                name: "read_file".into(),
                input: serde_json::json!({}),
                display_summary: Some("/etc/hosts".into()),
            })
            .await;

        assert!(recorder.events().is_empty());
    }

    #[tokio::test]
    async fn permission_forwarding_frontend_drops_nested_sub_agent_activity() {
        // A nested sub-agent's roll-up must not reach the parent: it already appears as a
        // `agent_spawn` line in this level's own record, and two writers on one tool call's
        // content would overwrite each other.
        let recorder = Arc::new(RecordingFrontend::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate, Some("toolu_parent".into()));

        forwarder
            .emit(FrontendEvent::SubAgentActivity {
                tool_call_id: "toolu_nested".into(),
                summary: "read_file: /deep".into(),
            })
            .await;

        assert!(recorder.events().is_empty());
    }

    /// The two events keyed by the sub-agent's own `tool_use_id` must not reach the parent's
    /// client, which has never been told that id exists. Composing is the sharper case: the
    /// `ToolCallStarted` that would close it is rolled up into a `SubAgentActivity` against the
    /// parent's call, so a client pairing the two would hold an indicator open on an id it never
    /// hears about again.
    #[tokio::test]
    async fn permission_forwarding_frontend_drops_events_keyed_by_sub_agent_call_id() {
        let recorder = Arc::new(RecordingFrontend::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate, Some("toolu_parent".into()));

        forwarder
            .emit(FrontendEvent::ToolCallComposing {
                id: "toolu_child".into(),
                name: "read_file".into(),
            })
            .await;
        forwarder
            .emit(FrontendEvent::ToolCallOutputDelta {
                id: "toolu_child".into(),
                chunk: "building...".into(),
            })
            .await;

        assert!(recorder.events().is_empty());
    }

    /// A detached call's token is what frontend delegation must honor on that task.
    ///
    /// `AcpFrontend::until_canceled` reads this before falling back to the session's cell. Without
    /// it, a `session/cancel` on any later turn abandoned a background task's `fs/*` request
    /// without sending it, contradicting `docs/book/src/usage/background.md`.
    #[tokio::test]
    async fn a_scoped_call_token_overrides_the_ambient_one() {
        assert!(
            current_call_cancellation().is_none(),
            "an ordinary turn has no per-call token, so the frontend falls back to the session's"
        );

        let call = tokio_util::sync::CancellationToken::new();
        let seen = scope_call_cancellation(call.clone(), async {
            let inside =
                current_call_cancellation().expect("the detached call publishes its token");
            assert!(!inside.is_cancelled());
            call.cancel();
            inside.is_cancelled()
        })
        .await;
        assert!(
            seen,
            "the token seen inside the scope must be the call's own"
        );

        assert!(
            current_call_cancellation().is_none(),
            "the scope must not leak past the call"
        );
    }

    /// Notices are the one event that *does* forward through `PermissionForwardingFrontend`. Image
    /// redaction during a sub-agent's provider call is a side effect the user needs to see, and the
    /// sub-agent's report has no place to surface it.
    #[tokio::test]
    async fn permission_forwarding_frontend_forwards_notice() {
        let recorder = Arc::new(RecordingFrontend::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate, None);
        forwarder
            .emit(FrontendEvent::Notice(crate::frontend::Notice::info(
                "redacted 2 images",
            )))
            .await;
        let events = recorder.events();
        assert_eq!(events.len(), 1, "exactly one event should forward");
        match &events[0] {
            FrontendEvent::Notice(notice) => {
                assert_eq!(notice.text, "redacted 2 images");
                assert_eq!(notice.level, crate::frontend::NoticeLevel::Info);
            }
            other => panic!("expected Notice, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn permission_forwarding_frontend_delegates_request_permission() {
        let delegate: Arc<dyn Frontend> =
            Arc::new(RecordingFrontend::with_permission(PermissionOutcome::Allow));
        let forwarder = PermissionForwardingFrontend::new(delegate, None);
        let outcome = forwarder
            .request_permission(PermissionRequest {
                tool_name: "write_file".into(),
                primary_param: Some("/tmp/foo".into()),
                input: serde_json::json!({"path": "/tmp/foo"}),
                cancellation: tokio_util::sync::CancellationToken::new(),
            })
            .await;
        assert_eq!(outcome, PermissionOutcome::Allow);
    }

    #[tokio::test]
    async fn silent_frontend_default_delegate_methods_answer_local() {
        // Default impls signal "no delegate available, do it locally".
        let frontend = SilentFrontend;
        assert!(matches!(
            frontend
                .delegate_fs_read(Path::new("/tmp/x"), None, None)
                .await,
            Delegation::Local
        ));
        assert!(matches!(
            frontend.delegate_fs_write(Path::new("/tmp/x"), "hi").await,
            Delegation::Local
        ));
    }

    /// Test fixture that records what arguments each delegate method was called with, and lets the
    /// test pick the response.
    pub(super) struct DelegatingRecorder {
        pub(crate) fs_reads: Mutex<Vec<PathBuf>>,
        pub(crate) fs_writes: Mutex<Vec<(PathBuf, String)>>,
        pub(crate) fs_read_response: Mutex<Option<Delegation<String>>>,
        pub(crate) fs_write_response: Mutex<Option<Delegation<()>>>,
    }

    impl DelegatingRecorder {
        fn new() -> Self {
            Self {
                fs_reads: Mutex::new(Vec::new()),
                fs_writes: Mutex::new(Vec::new()),
                fs_read_response: Mutex::new(Some(Delegation::Served("from-delegate".to_string()))),
                fs_write_response: Mutex::new(Some(Delegation::Served(()))),
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
        ) -> Delegation<String> {
            self.fs_reads.lock().unwrap().push(path.to_path_buf());
            self.fs_read_response
                .lock()
                .unwrap()
                .take()
                .unwrap_or(Delegation::Local)
        }

        async fn delegate_fs_write(&self, path: &Path, content: &str) -> Delegation<()> {
            self.fs_writes
                .lock()
                .unwrap()
                .push((path.to_path_buf(), content.to_string()));
            self.fs_write_response
                .lock()
                .unwrap()
                .take()
                .unwrap_or(Delegation::Local)
        }
    }

    #[tokio::test]
    async fn permission_forwarding_frontend_forwards_fs_read() {
        let recorder = Arc::new(DelegatingRecorder::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate, None);
        let outcome = forwarder
            .delegate_fs_read(Path::new("/tmp/sub.txt"), None, None)
            .await
            .served()
            .expect("delegate result");
        assert_eq!(outcome, "from-delegate");
        assert_eq!(recorder.fs_reads.lock().unwrap().as_slice(), &[
            PathBuf::from("/tmp/sub.txt")
        ],);
    }

    #[tokio::test]
    async fn permission_forwarding_frontend_forwards_fs_write() {
        let recorder = Arc::new(DelegatingRecorder::new());
        let delegate: Arc<dyn Frontend> = recorder.clone();
        let forwarder = PermissionForwardingFrontend::new(delegate, None);
        forwarder
            .delegate_fs_write(Path::new("/tmp/sub.txt"), "hi from sub-agent")
            .await
            .served()
            .expect("delegate result");
        let recorded = recorder.fs_writes.lock().unwrap().clone();
        assert_eq!(recorded, vec![(
            PathBuf::from("/tmp/sub.txt"),
            "hi from sub-agent".to_string()
        )]);
    }
}
