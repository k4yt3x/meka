//! The ACP client as a [`crate::frontend::Frontend`]: session updates, permission round-trips,
//! live terminal output, and the mode and config options an editor is shown.

use super::*;
use crate::frontend::Delegation;

/// Classify an `fs/*` failure into the one distinction the file tools route on.
///
/// `ResourceNotFound` (`-32002`) is the protocol's way for a client to say it will not serve a
/// path. Which paths those are is the client's business and differs between editors -- Zed answers
/// it for anything outside the project it has open, another client may serve any absolute path --
/// which is exactly why meka asks per path instead of modeling any editor's rule. Every other
/// code (transport, timeout, an internal error inside the client) leaves open the possibility that
/// the client owns the file and holds unsaved changes for it, so it must not route around it.
pub(super) fn classify_fs_error(
    method: &str,
    error: &agent_client_protocol::Error,
) -> FrontendError {
    let message = format!("{method} failed: {error}");
    if error.code == agent_client_protocol::ErrorCode::ResourceNotFound {
        FrontendError::unservable_path(message)
    } else {
        FrontendError::new(message)
    }
}
/// Late-bound view of everything the connected client told us on `initialize`: its advertised
/// capabilities and its self-identifying `Implementation` (name + version). Default is the
/// all-`false` `ClientCapabilities` and a `None` identity, so an `AcpFrontend` constructed before
/// `initialize` arrives correctly reports "delegation unavailable" and "client unknown" until the
/// handler fills it in.
#[derive(Clone, Default)]
pub(crate) struct SharedClientState {
    pub(super) inner: Arc<std::sync::RwLock<ClientStateInner>>,
}
#[derive(Clone, Default)]
pub(super) struct ClientStateInner {
    pub(super) capabilities: ClientCapabilities,
    /// Logged once on `initialize`.
    #[allow(dead_code, reason = "recorded for diagnostics; read only by tests")]
    pub(super) info: Option<Implementation>,
}
impl SharedClientState {
    /// Record both halves of the client-side initialize payload in one write. Called exactly once
    /// per process today (the `initialize` handler), but tolerant of re-initialization if a future
    /// client ever resends.
    pub(super) fn record_initialize(
        &self,
        capabilities: ClientCapabilities,
        info: Option<Implementation>,
    ) {
        let mut guard = crate::sync::write(&self.inner);
        *guard = ClientStateInner { capabilities, info };
    }

    pub(super) fn capabilities(&self) -> ClientCapabilities {
        crate::sync::read(&self.inner).capabilities.clone()
    }

    /// Whether the client renders agent-owned terminals from meka's `_meta` frames.
    ///
    /// Deliberately not the typed `terminal` capability: that one says the client implements
    /// `terminal/*` so an agent can run commands *in the client*, which meka never does. A client
    /// can offer that and still have no idea what [`META_TERMINAL_INFO`] means.
    pub(super) fn renders_agent_terminals(&self) -> bool {
        self.capabilities()
            .meta
            .as_ref()
            .and_then(|meta| meta.get(CAP_TERMINAL_OUTPUT))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    }

    #[cfg(test)]
    pub(super) fn client_info(&self) -> Option<Implementation> {
        crate::sync::read(&self.inner).info.clone()
    }
}
/// Render an `Implementation` as a `"name version"` pair for the `initialize` log line. `None`
/// renders as `"<unknown> <unknown>"` so the log shape is stable across clients that omit
/// `client_info` entirely.
pub(super) fn describe_client(info: Option<&Implementation>) -> String {
    match info {
        Some(implementation) => format!("{} {}", implementation.name, implementation.version),
        None => "<unknown> <unknown>".to_string(),
    }
}
/// ACP-side [`Frontend`] impl. Converts the agent loop's streamed events into ACP `session/update`
/// notifications and runs the `session/request_permission` round-trip for tool approvals.
/// Constructed per-session: every field is fully populated at build time, so there's no "not yet
/// bound" `Option` state to handle.
pub(crate) struct AcpFrontend {
    pub(super) connection: ConnectionTo<Client>,
    pub(super) session_id: SessionId,
    pub(super) cwd: SharedCwd,
    /// The `allow_always` and `reject_always` answers given this session, which short-circuit
    /// `request_permission` so the user isn't re-prompted for the same tool. Per-session (one
    /// `AcpFrontend` per session); not persisted.
    pub(super) sticky: crate::frontend::StickyApprovals,
    pub(super) client_state: SharedClientState,
    /// Stdio-level "transport is dead" latch, shared across every per-session `AcpFrontend` in the
    /// process. When `send_notification` fails on any session, the latch is set so every other
    /// session's agent loop short-circuits on its next iteration instead of burning provider
    /// tokens until its own emit also fails.
    ///
    /// Process-wide because the transport is: one closed pipe affects every session, so the global
    /// signal carries no false positives. A per-session transport would need a per-session
    /// sibling.
    pub(super) transport_dead: Arc<std::sync::atomic::AtomicBool>,
    /// Live context-occupancy counter shared with this session's agent through its
    /// [`crate::session::SessionCells`]; read on every `TokenUsage` event to emit an
    /// ACP `usage_update` so editors show "tokens used / context window".
    pub(super) context_tokens: Arc<std::sync::atomic::AtomicU64>,
    /// The resolved window for the `usage_update` `size` field, and the same cell the agent
    /// publishes into on every provider switch. `0` until the agent is built, which suppresses the
    /// update.
    ///
    /// Shared rather than pushed: a copy re-stored by hand from `session/set_config_option`
    /// reports a mid-turn switch as occupancy measured against the profile the turn is still
    /// running on, divided by the window of the one it has not moved to yet.
    pub(super) context_window: Arc<std::sync::atomic::AtomicU64>,
    /// Accumulated live output per in-flight tool call, keyed by `tool_use_id`. ACP replaces a
    /// tool call's whole `content` array on each update rather than appending to it, so the
    /// running total has to be kept somewhere; the emitter sends deltas, and this is where
    /// they are added up. Entries are dropped when the call completes.
    pub(super) live_output: std::sync::Mutex<std::collections::HashMap<String, LiveOutput>>,
    /// The same cell the session entry holds, so a client round-trip started by this frontend can
    /// be abandoned when `session/cancel` fires. Shared rather than copied: every turn installs a
    /// token of its own, and a frontend holding a stale clone would race against a token nobody
    /// signals.
    pub(super) cancel: crate::host::CancelCell,
}
/// How a running command's output is shown to this client.
///
/// The two modes are not cosmetic variants of each other. In [`Self::Terminal`] the client owns a
/// scrollback buffer that meka appends to, so the whole output is available and rendered as a real
/// terminal (ANSI colors, selection, its own scrolling). In [`Self::Text`] the only lever is
/// replacing the tool call's `content`, so meka has to keep the running text itself and can only
/// afford to re-send a window of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LiveOutputMode {
    /// The client set `_meta.terminal_output` on `initialize`, so it renders the agent-owned
    /// terminal frames.
    Terminal,
    /// Fallback: a `console` code block, replaced on each update.
    Text,
}
/// Per-tool-call state for relaying a running command's output.
pub(super) struct LiveOutput {
    pub(super) mode: LiveOutputMode,
    /// Text still to show. In [`LiveOutputMode::Text`] this is the whole (tail-capped) output,
    /// re-sent every tick. In [`LiveOutputMode::Terminal`] it is only the bytes not yet appended,
    /// and it is drained on each send, because the client keeps the scrollback.
    pub(super) text: String,
    /// `None` until the first update goes out, so the first chunk is never delayed.
    pub(super) last_sent: Option<std::time::Instant>,
}
/// Shortest gap between two `tool_call_update`s for the same tool call. A chatty build writes
/// thousands of small chunks a second, and one notification per read syscall would spend more time
/// on the wire than on the build. Coalescing is why both modes need a buffer at all.
pub(super) const LIVE_OUTPUT_INTERVAL: std::time::Duration = std::time::Duration::from_millis(150);
/// How much of the tail to keep visible while a command runs, in [`LiveOutputMode::Text`] only.
/// That mode re-sends its whole buffer on every tick, so an uncapped buffer would make a chatty
/// command cost quadratic in its own output. The terminal mode appends and needs no cap.
pub(super) const LIVE_OUTPUT_TAIL_BYTES: usize = 8 * crate::text::KIB;
impl LiveOutput {
    pub(super) fn new(mode: LiveOutputMode) -> Self {
        Self {
            mode,
            text: String::new(),
            last_sent: None,
        }
    }

    /// Append a delta and return what to send now, or `None` while throttled.
    pub(super) fn push(&mut self, chunk: &str, now: std::time::Instant) -> Option<String> {
        self.text.push_str(chunk);
        if self.mode == LiveOutputMode::Text && self.text.len() > LIVE_OUTPUT_TAIL_BYTES {
            let mut cut = self.text.len() - LIVE_OUTPUT_TAIL_BYTES;
            // A byte count can land inside a multi-byte character, and both slicing and draining
            // there panic. Advance to a boundary before either.
            while cut < self.text.len() && !self.text.is_char_boundary(cut) {
                cut += 1;
            }
            // Prefer opening the view at a line start rather than mid-line.
            if let Some(offset) = self.text[cut..].find('\n') {
                cut += offset + 1;
            }
            self.text.drain(..cut);
        }
        if let Some(last) = self.last_sent
            && now.duration_since(last) < LIVE_OUTPUT_INTERVAL
        {
            return None;
        }
        self.last_sent = Some(now);
        match self.mode {
            // Appending: hand over what has accumulated and start empty again, so the same bytes
            // are never sent twice.
            LiveOutputMode::Terminal => {
                if self.text.is_empty() {
                    None
                } else {
                    Some(std::mem::take(&mut self.text))
                }
            }
            LiveOutputMode::Text => Some(self.text.clone()),
        }
    }

    /// Whatever is still buffered, for the final flush before a call completes. The throttle can
    /// swallow the last chunk of a command that exits right after printing, and in terminal mode
    /// those bytes exist nowhere else.
    pub(super) fn take_pending(&mut self) -> Option<String> {
        match self.mode {
            LiveOutputMode::Terminal if !self.text.is_empty() => {
                Some(std::mem::take(&mut self.text))
            }
            _ => None,
        }
    }
}
/// `clientCapabilities._meta` key a client sets to say it renders agent-owned terminals from the
/// `_meta` frames below. Distinct from the typed `terminal` capability, which means "I implement
/// `terminal/*` requests" -- a client can do that without understanding these frames at all. Zed
/// advertises both; gating on the wrong one would send terminal content blocks to a client that
/// resolves them to nothing and so displays no output, which is the failure this whole path exists
/// to fix.
pub(super) const CAP_TERMINAL_OUTPUT: &str = "terminal_output";
/// `_meta` key naming an agent-owned terminal so the client registers it. Without this the client
/// has nothing to attach output to and buffers it against an id it was never told about (Zed parks
/// it in `pending_terminal_output`, drained only on a matching create), leaving an empty terminal.
///
/// Must ride on the `tool_call` that opens the call, not a later `tool_call_update`: the client
/// reads it only off the former.
pub(super) const META_TERMINAL_INFO: &str = "terminal_info";
/// `_meta` key carrying a chunk of output to append to an agent-owned terminal.
pub(super) const META_TERMINAL_OUTPUT: &str = "terminal_output";
/// `_meta` key marking an agent-owned terminal as finished, with its exit status.
pub(super) const META_TERMINAL_EXIT: &str = "terminal_exit";
/// Build the `_meta` map for one agent-owned-terminal frame.
///
/// This is an extension, not ACP proper: it originates in codex-acp, claude-agent-acp emits the
/// same shape, and Zed consumes it. ACP v2 standardizes the idea as `terminal_update` /
/// `terminal_output_chunk`, which is what this should become once a client speaks v2. `_meta` is
/// specified as ignorable (every `_meta` field deserializes with `DefaultOnError`), so a client
/// that doesn't know these keys drops them rather than failing.
pub(super) fn terminal_meta(
    key: &str,
    payload: serde_json::Value,
) -> agent_client_protocol::schema::v1::Meta {
    let mut meta = agent_client_protocol::schema::v1::Meta::new();
    meta.insert(key.to_string(), payload);
    meta
}
/// The connection one session's frontend talks to the editor over, with the per-connection state
/// it reads: what the client advertised, and whether the transport is still up.
pub(super) struct AcpClientLink {
    pub(super) connection: ConnectionTo<Client>,
    pub(super) session_id: SessionId,
    pub(super) client_state: SharedClientState,
    pub(super) transport_dead: Arc<std::sync::atomic::AtomicBool>,
}

impl AcpFrontend {
    pub(super) fn new(
        link: AcpClientLink,
        cwd: SharedCwd,
        context_tokens: Arc<std::sync::atomic::AtomicU64>,
        context_window: Arc<std::sync::atomic::AtomicU64>,
        cancel: crate::host::CancelCell,
    ) -> Self {
        let AcpClientLink {
            connection,
            session_id,
            client_state,
            transport_dead,
        } = link;
        Self {
            connection,
            session_id,
            cwd,
            sticky: crate::frontend::StickyApprovals::default(),
            client_state,
            transport_dead,
            context_tokens,
            context_window,
            live_output: std::sync::Mutex::new(std::collections::HashMap::new()),
            cancel,
        }
    }

    /// The current turn's cancellation token, cloned out of the cell each turn publishes into for
    /// its duration.
    ///
    /// A fresh token when no turn is live, which reads as "never canceled" and so leaves the
    /// round-trip waiting on the client alone. Callers reach this from inside a turn, so the case
    /// is theoretical, but answering with a token some earlier turn had canceled would abandon a
    /// request that nobody asked to stop.
    pub(super) fn current_cancellation(&self) -> CancellationToken {
        // A detached call's own token wins over the session's. See
        // `crate::frontend::scope_call_cancellation`: without this, canceling any later turn
        // abandoned a background task's `fs/*` round-trip mid-flight.
        if let Some(call) = crate::frontend::current_call_cancellation() {
            return call;
        }
        self.cancel.live().unwrap_or_default()
    }

    /// Await a client round-trip, giving up if the turn is canceled.
    ///
    /// `session/cancel` is the user pressing stop, and a client is entitled to drop an outstanding
    /// `fs/*` or elicitation request rather than answer it once it has canceled. Without this race
    /// the future never resolves: the prompt returns no `stopReason` at all and every later prompt
    /// on that session is refused for the one in flight, so one stop wedges the session for the
    /// life of the process.
    ///
    /// The canceled arm is a [`FrontendError`] rather than a `None` on purpose. Both callers
    /// return `Option<Result<_, FrontendError>>`, where `None` already means "this frontend has no
    /// delegate, do it locally" -- so a `?` on an `Option` here turned pressing stop into a local
    /// write, computing the edit from on-disk bytes and overwriting whatever the editor still held
    /// unsaved. Returning a `Result` makes that `?` a type error instead of a silent one.
    pub(super) async fn until_canceled<T>(
        &self,
        what: &str,
        work: impl std::future::Future<Output = T>,
    ) -> std::result::Result<T, FrontendError> {
        race_against_cancellation(what, &self.current_cancellation(), work).await
    }

    /// Mark the stdio transport as dead. Called from `emit` and the `session/load` replay loop
    /// whenever `send_notification` reports an error. Idempotent. The trait-level
    /// `client_disconnected()` read below surfaces the same flag back to the agent loop.
    pub(super) fn mark_transport_dead(&self) {
        self.transport_dead
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Show a scheduled job's prompt in the transcript, before the turn that answers it runs.
    ///
    /// Goes out as a `UserMessageChunk` rather than a notice because that is what it is: the turn
    /// has a prompt, it just came from a timer instead of a keystroke. Without it the editor shows
    /// a reply with nothing above it explaining the question.
    pub(crate) fn push_scheduled_prompt(&self, wakeup: &crate::scheduler::Wakeup) {
        self.push_out_of_band_prompt(&wakeup.render_prompt());
    }

    /// Show any prompt the user did not type, for the same reason: a reply with nothing above it
    /// explaining the question reads as the agent talking to itself. Used by scheduled jobs and by
    /// background-task outcome reports.
    pub(crate) fn push_out_of_band_prompt(&self, prompt: &str) {
        send_session_update(
            &self.connection,
            &self.session_id,
            crate::host::acp::schedule::out_of_band_prompt_update(prompt),
        );
    }

    /// Push one `session/update` to the client, latching the transport as dead if it fails.
    ///
    /// `send_notification` is synchronous, which is what lets the live-output path hold its buffer
    /// lock across a send to keep chunks ordered.
    pub(super) fn send_update(&self, update: SessionUpdate) {
        if let Err(error) = self
            .connection
            .send_notification(SessionNotification::new(self.session_id.clone(), update))
        {
            tracing::debug!("failed to send session/update: {error}");
            self.mark_transport_dead();
        }
    }

    /// Recover from a poisoned lock rather than propagating it. A panic under this lock would
    /// otherwise disable live output *and* the tool-call completion update for the rest of the
    /// session, leaving the client on a spinner that never resolves; the buffer is display state,
    /// so continuing with whatever it holds is strictly better than going silent.
    pub(super) fn live_output(
        &self,
    ) -> std::sync::MutexGuard<'_, std::collections::HashMap<String, LiveOutput>> {
        crate::sync::lock(&self.live_output)
    }

    /// Open a live-output view for a starting tool call and return how it will be rendered, or
    /// `None` for calls that produce no streamed output.
    ///
    /// Only `execute_command` qualifies: it is the one tool whose result can be minutes away, and
    /// the terminal frames below would be nonsense for anything that isn't a command. The mode is
    /// decided once, here, so the client can't be told about a terminal it never registered.
    pub(super) fn begin_live_output(&self, id: &str, tool_name: &str) -> Option<LiveOutputMode> {
        if tool_name != "execute_command" {
            return None;
        }
        let mode = if self.client_state.renders_agent_terminals() {
            LiveOutputMode::Terminal
        } else {
            LiveOutputMode::Text
        };
        // The single most useful line when a user reports "I see the command but not its output":
        // it says whether the client asked for terminal rendering, which is the whole branch point.
        tracing::debug!("execute_command {id} live output: {mode:?}");
        self.live_output()
            .insert(id.to_string(), LiveOutput::new(mode));
        Some(mode)
    }
}
#[async_trait]
impl Frontend for AcpFrontend {
    async fn emit(&self, event: FrontendEvent) {
        let update = match event {
            FrontendEvent::AssistantTextDelta(text) => {
                SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                    agent_client_protocol::schema::v1::TextContent::new(text),
                )))
            }
            // Nothing to show, and no transient UI to close: ACP thought chunks accumulate, so a
            // counter meant to be drawn over and replaced would leave a trail of stale figures in
            // the thread. Matched explicitly rather than left to the catch-all below, which is for
            // REPL signage an editor has its own UI for -- these two are the opposite case, signals
            // this frontend structurally cannot represent.
            FrontendEvent::ThinkingProgress { .. } | FrontendEvent::ThinkingEnded => return,
            // ACP takes reasoning whole, from `ThinkingBlock` below. The deltas carry the same text
            // and forwarding both would double every thought chunk in the thread.
            FrontendEvent::ThinkingDelta(_) => return,
            FrontendEvent::ThinkingBlock { content, .. } => {
                SessionUpdate::AgentThoughtChunk(ContentChunk::new(ContentBlock::Text(
                    agent_client_protocol::schema::v1::TextContent::new(content),
                )))
            }
            // ACP has a `pending` status this could drive, but a call is announced to the client
            // exactly once, and that announcement carries the title, locations and arguments an
            // editor renders -- none of which exist yet here. Sending a name-only call first would
            // mean two `session/update: tool_call` notifications for one id.
            FrontendEvent::ToolCallComposing { .. } => return,
            FrontendEvent::ToolCallStarted {
                id,
                name,
                input,
                display_summary,
            } => {
                // No separate `pending` state in the agent loop, so the in-progress emit is the
                // first one the client sees. The title carries the resolved primary argument (the
                // command, file path, URL, ...) so editors show what's actually running instead of
                // a bare tool name; `raw_input` still carries the full argument object.
                let locations = tool_locations(&name, &input, &self.cwd);
                let title = tool_call_title(&name, display_summary.as_deref());
                let mut call = ToolCall::new(id.clone(), title)
                    .kind(tool_kind_for(&name))
                    .status(ToolCallStatus::InProgress)
                    .locations(locations)
                    .raw_input(input);
                // Claim the rendering mode for this call up front, because the terminal has to be
                // announced before any output references it. The tool call's own id doubles as the
                // terminal id: it is unique per call and is what the client already correlates on.
                if self.begin_live_output(&id, &name) == Some(LiveOutputMode::Terminal) {
                    // `cwd` is optional in the frame but worth sending: the client labels the
                    // terminal with it, and meka's per-session cwd (which `/cd` moves) is not
                    // something the client could otherwise know.
                    let cwd = self.cwd.get();
                    call = call
                        .content(vec![ToolCallContent::Terminal(
                            agent_client_protocol::schema::v1::Terminal::new(id.clone()),
                        )])
                        .meta(terminal_meta(
                            META_TERMINAL_INFO,
                            serde_json::json!({ "terminal_id": id, "cwd": cwd }),
                        ));
                }
                SessionUpdate::ToolCall(call)
            }
            FrontendEvent::ToolCallOutputDelta { id, chunk } => {
                // stdout and stderr drain on separate tasks, so two deltas for the same call can
                // be in flight on two threads at once. Terminal mode appends whatever it is handed,
                // so draining the buffer and sending it have to be one atomic step: release the
                // lock in between and the two tasks can swap order, interleaving the terminal's
                // contents. Safe to hold across the send because `send_notification` is
                // synchronous -- nothing is awaited under the lock.
                let mut buffers = self.live_output();
                // Absent means the call never opened a live view (not `execute_command`, or it
                // already completed), so there is nothing to attach this to.
                let Some(entry) = buffers.get_mut(&id) else {
                    return;
                };
                let mode = entry.mode;
                // `None` means this tick is throttled away; the buffer keeps the bytes and the next
                // tick that clears the interval sends them.
                let Some(text) = entry.push(&chunk, std::time::Instant::now()) else {
                    return;
                };
                drop(buffers);
                self.send_update(SessionUpdate::ToolCallUpdate(live_output_update(
                    &id, mode, &text,
                )));
                return;
            }
            FrontendEvent::ToolCallCompleted {
                id,
                name,
                is_error,
                content,
                metadata,
            } => {
                let live = {
                    // The completion update carries the authoritative output, so the live view has
                    // done its job either way; take the mode and any bytes the throttle swallowed.
                    self.live_output()
                        .remove(&id)
                        .map(|mut entry| (entry.mode, entry.take_pending()))
                };
                // A command that exits immediately after printing can have its last chunk still
                // inside the throttle window. In terminal mode those bytes live nowhere else -- the
                // client's scrollback is the only copy -- so flush them before marking the call
                // done.
                if let Some((LiveOutputMode::Terminal, Some(pending))) = &live {
                    self.send_update(SessionUpdate::ToolCallUpdate(live_output_update(
                        &id,
                        LiveOutputMode::Terminal,
                        pending,
                    )));
                }
                let status = if is_error {
                    ToolCallStatus::Failed
                } else {
                    ToolCallStatus::Completed
                };
                let mut fields = ToolCallUpdateFields::new().status(status);
                let mut update_meta = None;
                if let Some((LiveOutputMode::Terminal, _)) = live {
                    // Keep the terminal as the call's content: it already holds the full
                    // scrollback, and replacing it with a text block here would
                    // swap a live, scrollable, color-rendered view for a
                    // flattened copy at the moment the command finishes.
                    fields = fields.content(vec![ToolCallContent::Terminal(
                        agent_client_protocol::schema::v1::Terminal::new(id.clone()),
                    )]);
                    update_meta = Some(terminal_meta(
                        META_TERMINAL_EXIT,
                        serde_json::json!({
                            "terminal_id": id,
                            "exit_code": command_exit_code(&metadata, is_error),
                            "signal": command_signal(&metadata),
                        }),
                    ));
                } else {
                    fields = fields.content(build_completion_content(&name, &content, metadata));
                }
                // Surface the structured tool output too, so clients (e.g. Zed's tool-call detail
                // view) can introspect the result beyond the rendered `content` blocks.
                if let Ok(raw) = serde_json::to_value(&content) {
                    fields = fields.raw_output(raw);
                }
                let mut update = ToolCallUpdate::new(id, fields);
                if let Some(meta) = update_meta {
                    update = update.meta(meta);
                }
                SessionUpdate::ToolCallUpdate(update)
            }
            FrontendEvent::SubAgentActivity {
                tool_call_id,
                summary,
            } => {
                // ACP has no sub-agent primitive -- no nested sessions, no nested tool calls -- so
                // a sub-agent is one tool call and its progress is that call's content. Updating
                // content while the call is still `in_progress` is what turns an opaque spinner
                // into a live view of what the sub-agent is doing. The client replaces the content
                // array on each update, which is why `summary` is the whole block.
                let fields = ToolCallUpdateFields::new().content(vec![text_content_block(summary)]);
                SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(tool_call_id, fields))
            }
            FrontendEvent::Notice(notice) => notice_update(&notice),
            // An editor holds a transcript, not a mirror of the model's window, so nothing on
            // screen is invalidated; but the automatic compactions happen with nobody asking, and
            // a user whose next reply forgets the morning deserves to know why.
            FrontendEvent::Compacted {
                source,
                replaced_count,
                generation,
            } => notice_update(&crate::frontend::Notice::info(format!(
                "compacted the conversation: {replaced_count} messages replaced by a summary \
                 ({source}, compaction {generation})"
            ))),
            FrontendEvent::McpProgress(update) => {
                // ACP has no protocol primitive for tool-progress streams. The REPL renders these
                // inline as a carriage-return-overwrite status line; in the ACP world the editor
                // already has its own visibility into MCP server activity (or can subscribe to the
                // stderr log stream of the spawned agent). Log at info so `-v` users can still see
                // them; don't pollute the assistant-message transcript with per-tick status text.
                let total = update
                    .total
                    .map(|total| format!("/{total}"))
                    .unwrap_or_default();
                let message = update
                    .message
                    .as_deref()
                    .map(|message| format!(", {message}"))
                    .unwrap_or_default();
                tracing::info!(
                    "MCP '{server}' {tool} progress: {progress}{total}{message}",
                    server = update.server_name,
                    tool = update.tool_name,
                    progress = update.progress,
                );
                return;
            }
            FrontendEvent::TodoListUpdated { items, .. } => {
                // The `todo` tool's list maps onto ACP's plan panel. The REPL-only `title` has no
                // `Plan` analog and is dropped. The agent loop (`agent/dispatch.rs`) never emits
                // an emptied list, so a cleared plan is not pushed, as on the REPL.
                SessionUpdate::Plan(Plan::new(todo_items_to_plan(&items)))
            }
            FrontendEvent::TokenUsage(_) => {
                // Mirror the REPL's context gauge as an ACP `usage_update` so editors (e.g. Zed)
                // show "tokens used / context window". `used` is read from the shared atomic the
                // agent updates each round (current occupancy: all input tiers + output) rather
                // than the event's per-turn total, which over-counts multi-round tool turns;
                // `size` is the resolved window. Suppress until both are known.
                let used = self
                    .context_tokens
                    .load(std::sync::atomic::Ordering::Relaxed);
                let size = self
                    .context_window
                    .load(std::sync::atomic::Ordering::Relaxed);
                if used == 0 || size == 0 {
                    return;
                }
                SessionUpdate::UsageUpdate(UsageUpdate::new(used, size))
            }
            FrontendEvent::TurnStarted => {
                // Tool calls begin and end inside a turn, so anything still open here belongs to a
                // previous one that never delivered its completion (a canceled turn, or a stream
                // retried after announcing a tool call). Those entries would otherwise accumulate
                // for the life of the session. `TurnFinished` is not a substitute: the agent loop
                // only emits it when the turn succeeded, which is exactly when there is nothing to
                // clean up.
                self.live_output().clear();
                return;
            }
            // REPL signage. The editor named the session it created, and the `session/prompt`
            // response is what tells it the turn is over. A withdrawn prompt is a scheduled fire's,
            // never one the editor sent, so its transcript has nothing to amend. Spelled out rather
            // than left to a catch-all so a variant added later has to be placed here on purpose.
            FrontendEvent::SessionStarted { .. }
            | FrontendEvent::TurnFinished
            | FrontendEvent::PromptWithdrawn => return,
        };

        self.send_update(update);
    }

    fn client_disconnected(&self) -> bool {
        self.transport_dead
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome {
        // Honor sticky decisions from earlier `*_always` selections.
        if let Some(remembered) = self.sticky.remembered(&request.tool_name) {
            return remembered;
        }

        let connection = self.connection.clone();
        let session_id = self.session_id.clone();

        // The sticky options name the *tool*, because that is their scope: the decision is keyed on
        // the tool name alone and applies to every later call to it, whatever its arguments. The
        // prompt's title beside them is `<tool> <primary_param>` -- for `execute_command` that is
        // the specific command line -- so a bare "Always allow" reads as approving the command the
        // user just read, when it actually approves every shell command for the rest of the
        // session. Spelling the tool out is what makes the affordance and the semantics agree, and
        // it is spelled the way the title spells it.
        let options = vec![
            PermissionOption::new(OPTION_ALLOW_ONCE, "Allow", PermissionOptionKind::AllowOnce),
            PermissionOption::new(
                OPTION_ALLOW_ALWAYS,
                sticky_option_label("allow", &request.tool_name),
                PermissionOptionKind::AllowAlways,
            ),
            PermissionOption::new(OPTION_REJECT_ONCE, "Deny", PermissionOptionKind::RejectOnce),
            PermissionOption::new(
                OPTION_REJECT_ALWAYS,
                sticky_option_label("deny", &request.tool_name),
                PermissionOptionKind::RejectAlways,
            ),
        ];

        // Synthetic id: the permission round-trip is its own space, not correlated with the
        // streaming tool_call lifecycle.
        let tool_call_id = format!("permission-{}", uuid::Uuid::new_v4());
        // The title is the one line a client is sure to show, and it names the destination for
        // every write-shaped tool; `raw_input` and the content block carry every argument, so a
        // client that renders either shows what is being written and not only where. See
        // `PermissionRequest::input`.
        let title = tool_call_title(&request.tool_name, request.primary_param.as_deref());
        let fields = ToolCallUpdateFields::new()
            .kind(tool_kind_for(&request.tool_name))
            .title(title)
            .status(ToolCallStatus::Pending)
            .content(vec![text_content_block(approval_arguments_block(
                &request.input,
            ))])
            .raw_input(request.input.clone());
        let tool_call = ToolCallUpdate::new(tool_call_id, fields);

        let acp_request = RequestPermissionRequest::new(session_id, tool_call, options);
        // Race the round-trip against the per-turn cancellation token. If `session/cancel` fires
        // while we're waiting for the client to answer the permission prompt, we resolve as
        // `Canceled` instead of holding the runtime mutex forever, which would block
        // `session/close` and `session/set_mode` too.
        let response = tokio::select! {
            biased;
            _ = request.cancellation.cancelled() => {
                return PermissionOutcome::Canceled;
            }
            // The backstop the cancellation race alone does not provide. A client that is
            // *connected* but never answers -- an editor whose UI thread is wedged, a headless
            // harness that speaks ACP but implements no prompt -- fires no cancellation, so the
            // race above waits on it forever and the turn holds the runtime mutex with it. Deny on
            // expiry rather than allow: an unanswered prompt is not consent.
            _ = tokio::time::sleep(crate::frontend::APPROVAL_TIMEOUT) => {
                tracing::warn!(
                    "the client did not answer the permission prompt for '{tool}' within \
                     {timeout:?}; denying it",
                    tool = request.tool_name,
                    timeout = crate::frontend::APPROVAL_TIMEOUT,
                );
                return PermissionOutcome::Deny;
            }
            result = connection.send_request(acp_request).block_task() => match result {
                Ok(response) => response,
                Err(error) => {
                    tracing::debug!("failed to send session/request_permission: {error}");
                    // Spec-conformant clients always reply with a `Selected` or `Cancelled`
                    // outcome, so an `Err` here is almost certainly transport-level. Mark the
                    // connection dropped so the agent loop short-circuits on the next pre-iteration
                    // check instead of running a tool, emitting a denied result, and only then
                    // discovering the client is gone via the next emit. The FS delegates
                    // intentionally don't do this: those paths legitimately receive JSON-RPC error
                    // responses (a path the client won't serve), which would produce false-positive
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
                StickyDecision::AllowAlways => self.sticky.remember_allow(&request.tool_name),
                StickyDecision::RejectAlways => self.sticky.remember_deny(&request.tool_name),
            },
        )
    }

    async fn delegate_fs_read(
        &self,
        path: &std::path::Path,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> Delegation<String> {
        let caps = self.client_state.capabilities();
        if !caps.fs.read_text_file {
            return Delegation::Local;
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
        let outcome = match self
            .until_canceled(
                "fs/read_text_file",
                connection.send_request(request).block_task(),
            )
            .await
        {
            Ok(outcome) => outcome,
            Err(canceled) => return Delegation::Failed(canceled),
        };
        match outcome {
            Ok(response) => Delegation::Served(response.content),
            Err(error) => Delegation::Failed(classify_fs_error("fs/read_text_file", &error)),
        }
    }

    async fn delegate_fs_write(&self, path: &std::path::Path, content: &str) -> Delegation<()> {
        let caps = self.client_state.capabilities();
        if !caps.fs.write_text_file {
            return Delegation::Local;
        }
        let connection = self.connection.clone();
        let session_id = self.session_id.clone();
        let request =
            WriteTextFileRequest::new(session_id, path.to_path_buf(), content.to_string());
        let outcome = match self
            .until_canceled(
                "fs/write_text_file",
                connection.send_request(request).block_task(),
            )
            .await
        {
            Ok(outcome) => outcome,
            Err(canceled) => return Delegation::Failed(canceled),
        };
        match outcome {
            Ok(_) => Delegation::Served(()),
            Err(error) => Delegation::Failed(classify_fs_error("fs/write_text_file", &error)),
        }
    }

    async fn handle_elicitation(
        &self,
        prompt: crate::frontend::ElicitationPrompt,
    ) -> crate::frontend::ElicitationResponse {
        use crate::frontend::{ElicitationKind, ElicitationResponse};

        let kind = match &prompt.kind {
            ElicitationKind::Form { .. } => "form",
            ElicitationKind::Url { .. } => "url",
        };
        // Form and URL support are advertised independently, so check the mode actually being
        // asked for. An unadvertised mode must not be sent: the client would reject it, and the
        // MCP call is blocked on the answer meanwhile.
        let supported = self
            .client_state
            .capabilities()
            .elicitation
            .is_some_and(|caps| match &prompt.kind {
                ElicitationKind::Form { .. } => caps.form.is_some(),
                ElicitationKind::Url { .. } => caps.url.is_some(),
            });
        // Each decline made on the user's behalf is said in the transcript, since the tool call
        // waiting on the answer fails next and nothing else names the cause.
        if !supported {
            self.emit(FrontendEvent::Notice(
                crate::frontend::Notice::elicitation_declined(
                    &prompt.server_name,
                    &format!("the client does not support {kind} elicitation"),
                ),
            ))
            .await;
            return ElicitationResponse::Decline;
        }

        let Some(request) = elicitation::to_acp_request(&prompt, &self.session_id) else {
            self.emit(FrontendEvent::Notice(
                crate::frontend::Notice::elicitation_declined(
                    &prompt.server_name,
                    "its schema uses a field type ACP cannot express",
                ),
            ))
            .await;
            return ElicitationResponse::Decline;
        };

        // Raced against the turn's cancellation, like every other client round-trip on this
        // frontend. A bare await here was the last one left: an MCP `call_tool` is blocked on this
        // answer, so a client that drops the request rather than answering it -- which it is
        // entitled to do once the user has pressed stop -- left the tool call, the turn, and every
        // later prompt on that session waiting for the life of the process.
        //
        // Declining on cancellation rather than propagating it, because the return type has no
        // third state and the MCP server needs an answer either way. The turn is stopping; what the
        // server does with the refusal no longer changes what the user sees.
        let outcome = match self
            .until_canceled(
                "elicitation/create",
                self.connection.clone().send_request(request).block_task(),
            )
            .await
        {
            Ok(outcome) => outcome,
            Err(_canceled) => {
                tracing::debug!(
                    "MCP elicitation from '{server}' ({kind}) declined: the turn was canceled",
                    server = prompt.server_name,
                );
                return ElicitationResponse::Decline;
            }
        };
        match outcome {
            Ok(response) => elicitation::from_acp_action(response.action),
            Err(error) => {
                self.emit(FrontendEvent::Notice(
                    crate::frontend::Notice::elicitation_declined(
                        &prompt.server_name,
                        &format!("elicitation/create failed: {error}"),
                    ),
                ))
                .await;
                ElicitationResponse::Decline
            }
        }
    }
}
/// An advisory as an assistant-message chunk, since ACP has no primitive for one. The `[meka]`
/// prefix (`[meka warn]` for a [`crate::frontend::NoticeLevel::Warn`]) is what lets an editor's
/// transcript record the side-effect and a client filter or style by it.
///
/// Deliberately not [`crate::conversation::HARNESS_NOTE`], despite the resemblance. That marker is
/// longer because a *model* has to read it and should not have to recall what meka is; this one is
/// read by a person looking at their own editor, where the product name needs no gloss, and by
/// clients matching on the prefix, which a rename would break.
pub(super) fn notice_update(notice: &crate::frontend::Notice) -> SessionUpdate {
    let prefix = match notice.level {
        crate::frontend::NoticeLevel::Info => "[meka]",
        crate::frontend::NoticeLevel::Warn => "[meka warn]",
    };
    SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
        agent_client_protocol::schema::v1::TextContent::new(format!("{prefix} {}", notice.text)),
    )))
}
/// Every argument of a call awaiting approval, as a fenced block for the prompt's content.
///
/// The title names the destination; this is what would be written there. Pretty-printed JSON
/// rather than the REPL's key-per-line rendering because an editor renders markdown, where a code
/// block keeps a multi-line `content` legible and a client-side width does the wrapping.
pub(super) fn approval_arguments_block(input: &serde_json::Value) -> String {
    let rendered = serde_json::to_string_pretty(input).unwrap_or_else(|_| input.to_string());
    format!("```json\n{rendered}\n```")
}
/// Stable string IDs for the four permission options. The agent and the client must agree on these;
/// picking them as `const`s keeps the match arm in [`translate_permission_outcome`] honest.
pub(super) const OPTION_ALLOW_ONCE: &str = "allow_once";
pub(super) const OPTION_ALLOW_ALWAYS: &str = "allow_always";
pub(super) const OPTION_REJECT_ONCE: &str = "reject_once";
pub(super) const OPTION_REJECT_ALWAYS: &str = "reject_always";
/// Label for a sticky (`*Always`) permission option, naming the tool the decision actually covers.
///
/// A function rather than two `format!`s at the call site so the wording is assertable. The sticky
/// options are keyed on the tool name alone, and the prompt beside them shows one specific
/// invocation, so a bare "Always allow" reads as approving the command on screen when it approves
/// every call to that tool for the session. The tool name is the part that must not go missing.
pub(super) fn sticky_option_label(verb: &str, tool_name: &str) -> String {
    format!("Always {verb} any {tool_name}")
}
/// The cancellation race itself, taking the token rather than reading it off an `AcpFrontend`, so
/// it can be exercised without standing up a connection to a client.
///
/// `biased` matters: a turn canceled while the request is already outstanding must lose the race
/// deterministically, not half the time. The client owes no answer to a request the user withdrew.
pub(super) async fn race_against_cancellation<T>(
    what: &str,
    cancellation: &CancellationToken,
    work: impl std::future::Future<Output = T>,
) -> std::result::Result<T, FrontendError> {
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => {
            tracing::debug!("{what} abandoned: the turn was canceled");
            Err(FrontendError::canceled(what))
        }
        result = work => Ok(result),
    }
}
/// Indicates which sticky bucket the user just opted into, so the caller can update its set.
/// Internal to the permission flow.
pub(super) enum StickyDecision {
    AllowAlways,
    RejectAlways,
}
/// Map an ACP outcome to meka's [`PermissionOutcome`] and fire `record_sticky` when the user picked
/// one of the `*_always` options. Pure function so it's easy to unit-test.
pub(super) fn translate_permission_outcome<F>(
    outcome: RequestPermissionOutcome,
    tool_name: &str,
    mut record_sticky: F,
) -> PermissionOutcome
where
    F: FnMut(StickyDecision),
{
    match outcome {
        RequestPermissionOutcome::Cancelled => PermissionOutcome::Canceled,
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
                        "request_permission for '{tool_name}' returned unknown option_id '{other}'; \
                         defaulting to Deny"
                    );
                    PermissionOutcome::Deny
                }
            }
        }
        // ACP's `RequestPermissionOutcome` is `#[non_exhaustive]`; any future variant we haven't
        // taught the agent about should fail closed.
        other => {
            tracing::debug!(
                "request_permission for '{tool_name}' returned unknown outcome {other:?}; \
                 defaulting to Deny"
            );
            PermissionOutcome::Deny
        }
    }
}
/// Map meka's tool name to ACP's [`ToolKind`] so clients can pick the right icon and grouping.
/// MCP-loaded tools (named `mcp__server__tool`) and anything unknown fall through to `Other`.
pub(super) fn tool_kind_for(name: &str) -> ToolKind {
    match name {
        "read_file" | "todo" => ToolKind::Read,
        "edit_file" | "write_file" => ToolKind::Edit,
        "find_files" | "search_contents" => ToolKind::Search,
        "execute_command" => ToolKind::Execute,
        "fetch_url" => ToolKind::Fetch,
        "agent_spawn" => ToolKind::Think,
        // skill, memory_*, scratchpad_*, render_image, load_tool, mcp__*, and any
        // future built-ins.
        _ => ToolKind::Other,
    }
}
/// Build the human-readable `title` for a tool call: the tool's name, then the resolved primary
/// argument (`display_summary`: the command for `execute_command`, the path for `read_file`, the
/// URL for `fetch_url`, ...), so editors show what's running and not only which tool. The name is
/// the one the REPL's indicator and approval prompt show, so the surfaces share one vocabulary.
/// `raw_input` still carries the full argument object for clients that want it.
pub(super) fn tool_call_title(name: &str, display_summary: Option<&str>) -> String {
    let raw = match display_summary.map(str::trim).filter(|s| !s.is_empty()) {
        Some(argument) => format!("{name} {argument}"),
        None => name.to_string(),
    };
    sanitize_title(&raw)
}
/// Collapse internal whitespace (so a multi-line command becomes a one-line title) and cap the
/// length so an editor never gets an unwieldy title. Mirrors claude-agent-acp's `sanitizeTitle`.
pub(super) fn sanitize_title(text: &str) -> String {
    const MAX_TITLE_CHARS: usize = 256;
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= MAX_TITLE_CHARS {
        collapsed
    } else {
        let truncated: String = collapsed.chars().take(MAX_TITLE_CHARS - 1).collect();
        format!("{truncated}…")
    }
}
/// Convert meka's `todo` tool list into ACP [`PlanEntry`] rows for [`SessionUpdate::Plan`]. meka's
/// `Canceled` status has no ACP analog, so it maps to `Completed` ("no longer active") to keep
/// the entry count stable against the model's own todo list. meka tracks no per-item priority, so
/// every entry is reported as `Medium`.
pub(super) fn todo_items_to_plan(items: &[TodoItem]) -> Vec<PlanEntry> {
    items
        .iter()
        .map(|item| {
            let status = match item.status {
                TodoStatus::Pending => PlanEntryStatus::Pending,
                TodoStatus::InProgress => PlanEntryStatus::InProgress,
                TodoStatus::Completed | TodoStatus::Canceled => PlanEntryStatus::Completed,
            };
            PlanEntry::new(item.text.clone(), PlanEntryPriority::Medium, status)
        })
        .collect()
}
/// Compute the `locations` entries for a tool call. For tools whose primary argument is a path,
/// resolve it against the agent's per-session cwd (ACP requires absolute paths). Anything else
/// returns an empty list; clients fall back to the `raw_input` field.
pub(super) fn tool_locations(
    name: &str,
    input: &serde_json::Value,
    cwd: &SharedCwd,
) -> Vec<ToolCallLocation> {
    let raw = match name {
        "read_file" | "edit_file" | "write_file" | "find_files" | "search_contents" => {
            input.get("path").and_then(|v| v.as_str())
        }
        _ => None,
    };
    raw.map(|path| {
        let mut location = ToolCallLocation::new(resolve_against_cwd(cwd, path));
        // For `read_file`, point the client at the first line being read. meka's `offset` is
        // 0-based; ACP line numbers are 1-based.
        if name == "read_file"
            && let Some(offset) = input.get("offset").and_then(|value| value.as_u64())
        {
            location = location.line(u32::try_from(offset.saturating_add(1)).unwrap_or(u32::MAX));
        }
        vec![location]
    })
    .unwrap_or_default()
}
/// Wrap a string as a plain-text [`ToolCallContent`] block.
pub(super) fn text_content_block(text: impl Into<String>) -> ToolCallContent {
    ToolCallContent::from(ContentBlock::Text(
        agent_client_protocol::schema::v1::TextContent::new(text.into()),
    ))
}
/// Build the `tool_call_update` that carries a running command's output, in whichever shape the
/// client understands. Terminal mode appends `text` to the client's scrollback and repeats the
/// content block so the terminal stays attached; text mode replaces the content with the window
/// meka is holding.
pub(super) fn live_output_update(id: &str, mode: LiveOutputMode, text: &str) -> ToolCallUpdate {
    match mode {
        LiveOutputMode::Terminal => {
            let fields = ToolCallUpdateFields::new().content(vec![ToolCallContent::Terminal(
                agent_client_protocol::schema::v1::Terminal::new(id.to_string()),
            )]);
            ToolCallUpdate::new(id.to_string(), fields).meta(terminal_meta(
                META_TERMINAL_OUTPUT,
                serde_json::json!({ "terminal_id": id, "data": text }),
            ))
        }
        LiveOutputMode::Text => {
            let fields = ToolCallUpdateFields::new().content(vec![console_content_block(text)]);
            ToolCallUpdate::new(id.to_string(), fields)
        }
    }
}
/// Exit code for a finished command's terminal frame. Falls back to the coarse "did it fail" bit
/// when the tool didn't report a code (a signal kill, or a tool error raised before the spawn).
pub(super) fn command_exit_code(
    metadata: &Option<ToolOutputMetadata>,
    is_error: bool,
) -> Option<i32> {
    match metadata {
        Some(ToolOutputMetadata::CommandExit { exit_code, .. }) => *exit_code,
        _ => Some(i32::from(is_error)),
    }
}
/// Signal name for a finished command's terminal frame, when it was killed rather than exiting.
pub(super) fn command_signal(metadata: &Option<ToolOutputMetadata>) -> Option<String> {
    match metadata {
        Some(ToolOutputMetadata::CommandExit { signal, .. }) => signal.clone(),
        _ => None,
    }
}
/// Wrap shell output in a `console` code block so editors render it monospaced (mirrors
/// claude-agent-acp's no-terminal fallback). Shared by the live view a command streams while it
/// runs and the final one emitted when it exits, so output doesn't reflow when the call completes.
pub(super) fn console_content_block(output: &str) -> ToolCallContent {
    text_content_block(format!("```console\n{}\n```", output.trim_end()))
}
/// Build the `content` array of a `tool_call_update` from meka's tool output. A populated `Diff`
/// metadata wins (so clients like Zed get the structured diff for apply-UI). `execute_command`
/// output is wrapped in a `console` code block so editors render it monospaced (mirrors
/// claude-agent-acp's no-terminal fallback). Other tools pass their text and image blocks through
/// unchanged, so a tool that looked at an image (`read_file` on a PNG, `render_image`, `fetch_url`)
/// shows the human the same picture the model saw.
pub(super) fn build_completion_content(
    tool_name: &str,
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

    if tool_name == "execute_command" {
        // Reuse the canonical text-flattening; `execute_command` output is text-only, so the
        // `[Image]` marker `tool_result_text_content` would emit for images never appears here.
        let combined = MekaContentBlock::tool_result_text_content(content);
        if combined.trim_end().is_empty() {
            return Vec::new();
        }
        return vec![console_content_block(&combined)];
    }

    content
        .iter()
        .map(|block| match block {
            ToolResultContent::Text { text } => text_content_block(text.clone()),
            // The payload is already provider-normalized (size-capped, converted to a native
            // format) by the time it reaches here, so it can go out as-is. A reference has no
            // payload to show and is named instead.
            ToolResultContent::Image { source } => match source.base64_data() {
                Some(data) => ToolCallContent::from(ContentBlock::Image(ImageContent::new(
                    data.to_string(),
                    source.media_type().to_string(),
                ))),
                None => text_content_block(crate::image::UNRESOLVED_IMAGE_PLACEHOLDER.to_string()),
            },
        })
        .collect()
}
/// Walk a hydrated [`Conversation`] and emit one `session/update` notification per content
/// block, mirroring the streaming shape the client would have seen had it been connected during
/// the original turn. Used by `session/load` so an editor that just reopened a session replays the
/// full history into its UI.
///
/// A user turn's context block is not replayed, so the client sees only what the user typed.
///
/// Tool calls track open `tool_use_id`s; any tool that never received a matching `ToolResult` (e.g.
/// a crashed turn) is closed out with a `failed` `tool_call_update` so the client doesn't render a
/// stuck spinner.
pub(super) fn replay_session_updates(
    connection: &ConnectionTo<Client>,
    session_id: &SessionId,
    cwd: &SharedCwd,
    messages: &Conversation,
) {
    // Map each open `tool_use_id` to its tool name so the result update can format output per tool
    // and the orphan sweep can close stragglers.
    let mut open_tools: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for message in messages.as_slice() {
        match message.role {
            Role::User => {
                for block in &message.content {
                    match block {
                        MekaContentBlock::Text { text } => {
                            if !text.is_empty() {
                                send_session_update(
                                    connection,
                                    session_id,
                                    SessionUpdate::UserMessageChunk(ContentChunk::new(
                                        ContentBlock::Text(
                                            agent_client_protocol::schema::v1::TextContent::new(
                                                text.clone(),
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
                            let tool_name = open_tools
                                .get(tool_use_id)
                                .map(String::as_str)
                                .unwrap_or("");
                            let acp_content = build_completion_content(tool_name, content, None);
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
                        // Re-emit input images so a reopened session shows the attachment.
                        MekaContentBlock::Image { source } => {
                            // A hydrated conversation carries bytes; a reference has nothing to
                            // show and is skipped rather than sent as an empty image.
                            if let Some(data) = source.base64_data() {
                                send_session_update(
                                    connection,
                                    session_id,
                                    SessionUpdate::UserMessageChunk(ContentChunk::new(
                                        ContentBlock::Image(ImageContent::new(
                                            data.to_string(),
                                            source.media_type().to_string(),
                                        )),
                                    )),
                                );
                            }
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
                                        agent_client_protocol::schema::v1::TextContent::new(
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
                                        agent_client_protocol::schema::v1::TextContent::new(
                                            thinking.clone(),
                                        ),
                                    ),
                                )),
                            );
                        }
                        MekaContentBlock::RedactedThinking { .. } => {
                            send_session_update(
                                connection,
                                session_id,
                                SessionUpdate::AgentThoughtChunk(ContentChunk::new(
                                    ContentBlock::Text(
                                        agent_client_protocol::schema::v1::TextContent::new(
                                            "[redacted thinking]".to_string(),
                                        ),
                                    ),
                                )),
                            );
                        }
                        MekaContentBlock::ToolUse { id, name, input } => {
                            let locations = tool_locations(name, input, cwd);
                            // Match the live path's rich title. No tool schema is available on
                            // replay, so only built-in tools resolve a primary argument; MCP tools
                            // fall back to the bare name.
                            let display_summary =
                                crate::tools::resolve_primary_param(name, input, None);
                            let title = tool_call_title(name, display_summary.as_deref());
                            let call = ToolCall::new(id.clone(), title)
                                .kind(tool_kind_for(name))
                                .status(ToolCallStatus::InProgress)
                                .locations(locations)
                                .raw_input(input.clone());
                            send_session_update(
                                connection,
                                session_id,
                                SessionUpdate::ToolCall(call),
                            );
                            open_tools.insert(id.clone(), name.clone());
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Tool calls without a matching result: close them as failed so the client's "tool running"
    // indicator doesn't get stuck.
    for orphan_id in open_tools.into_keys() {
        let fields = ToolCallUpdateFields::new().status(ToolCallStatus::Failed);
        send_session_update(
            connection,
            session_id,
            SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(orphan_id, fields)),
        );
    }
}
pub(super) fn send_session_update(
    connection: &ConnectionTo<Client>,
    session_id: &SessionId,
    update: SessionUpdate,
) {
    if let Err(error) =
        connection.send_notification(SessionNotification::new(session_id.clone(), update))
    {
        // Every `session/update` goes out through here, not just `session/load` replay, so name the
        // notification rather than one caller: this is the line to look at when a client reports
        // that updates aren't arriving.
        tracing::debug!("failed to send session/update: {error}");
    }
}
/// Emit a `session_info_update` carrying the session title exactly once. The title is
/// [`Conversation::title`], which never changes once the first user words are in, so `title_sent`
/// guards against re-emission across the first prompt and any later load/resume of the same
/// session.
pub(super) fn maybe_emit_session_title(
    connection: &ConnectionTo<Client>,
    session_id: &SessionId,
    title_sent: &std::sync::atomic::AtomicBool,
    messages: &Conversation,
) {
    use std::sync::atomic::Ordering;
    if title_sent.load(Ordering::Acquire) {
        return;
    }
    let title = messages.title();
    if title.is_empty() {
        return;
    }
    // Claim the one-shot before sending; if a concurrent path beat us to it, skip.
    if title_sent.swap(true, Ordering::AcqRel) {
        return;
    }
    send_session_update(
        connection,
        session_id,
        SessionUpdate::SessionInfoUpdate(SessionInfoUpdate::new().title(title)),
    );
}
/// Map a meka [`Permission`] to its ACP [`SessionModeId`] string: the same lowercase word
/// `Permission::Display` produces, so the id a client reads back is the one it sees in
/// `config.toml` and on the `--permission` flag.
pub(super) fn mode_id_for(permission: Permission) -> SessionModeId {
    SessionModeId::from(permission.to_string())
}
/// Parse a `SessionModeId` (treated as a `&str`) into the matching `Permission`. Returns `None` for
/// a mode id naming no level, which the caller turns into an error response.
///
/// Delegates to [`Permission`]'s [`std::str::FromStr`] rather than keeping its own table. A second
/// hand-maintained copy is the shape that grants a client a level it did not ask for: the two have
/// to stay in lock-step, and when they drift it is an id silently mapping to the wrong rung. One
/// table cannot drift from itself.
pub(super) fn parse_mode_id(id: &str) -> Option<Permission> {
    id.parse().ok()
}
/// Human-readable label for a permission level, shown in editor mode pickers next to each option.
/// Kept in lock-step with the REPL's `/permission` output and the `[permissions]` keys in
/// `config.toml` so users see the same vocabulary everywhere.
pub(super) fn level_display_name(permission: Permission) -> &'static str {
    match permission {
        Permission::None => "None",
        Permission::Read => "Read",
        Permission::Workspace => "Workspace",
        Permission::Unrestricted => "Unrestricted",
    }
}
/// One-line description of what a permission level lets the agent do. Shown beneath the level's
/// label in editor pickers.
///
/// `Unrestricted` is described by its *reach*, not by the absence of approval prompts: "all tools
/// without per-call approval" is equally true of `Workspace`, so it never distinguished the two.
pub(super) fn level_description(permission: Permission) -> &'static str {
    match permission {
        Permission::None => "No tools available.",
        Permission::Read => "File reads and searches only. No writes, no shell.",
        Permission::Workspace => "Writes confined to the workspace roots.",
        Permission::Unrestricted => "Writes and shell commands reach anywhere on the machine.",
    }
}
/// Build the `SessionModeState` advertised on every session-creation response (`session/new`,
/// `session/load`, `session/resume`). Only levels in [`SharedPermission::enabled`] are exposed:
/// picking a level outside that set through `session/set_mode` later would only error out, so
/// they are not surfaced in the first place.
pub(super) fn build_mode_state(permission: &SharedPermission) -> SessionModeState {
    let modes: Vec<SessionMode> = permission
        .enabled()
        .iter()
        .map(|mode| {
            SessionMode::new(mode_id_for(mode), level_display_name(mode))
                .description(level_description(mode))
        })
        .collect();
    SessionModeState::new(mode_id_for(permission.get()), modes)
}
/// The `configOptions` id for the permission picker, which duplicates the `modes` field.
///
/// Both are advertised, the way the reference adapter does it: `modes` and `configOptions` are
/// separate response fields rendered in separate places, so a client that only understands `modes`
/// keeps its picker, and one that understands `configOptions` gets permission and profile side by
/// side rather than in two unrelated menus.
pub(super) const PERMISSION_CONFIG_ID: &str = "permission";
/// The `configOptions` id for the profile picker. `modes` has no counterpart for this one.
pub(super) const PROFILE_CONFIG_ID: &str = "profile";
/// The `configOptions` id for the approvals switch, a boolean beside the two pickers.
pub(super) const APPROVALS_CONFIG_ID: &str = "approvals";
/// The permission picker as a `configOptions` entry. The same enabled set [`build_mode_state`]
/// exposes, for the same reason: a level the client cannot actually be granted has no business in
/// the list.
pub(super) fn permission_config_option(permission: &SharedPermission) -> SessionConfigOption {
    SessionConfigOption::select(
        PERMISSION_CONFIG_ID,
        "Permission",
        SessionConfigValueId::from(permission.get().to_string()),
        permission
            .enabled()
            .iter()
            .map(|mode| {
                SessionConfigSelectOption::new(
                    SessionConfigValueId::from(mode.to_string()),
                    level_display_name(mode),
                )
                .description(level_description(mode))
            })
            .collect::<Vec<_>>(),
    )
    .category(SessionConfigOptionCategory::Mode)
    .description("What the agent may do without asking.")
}
/// The approvals switch as a `configOptions` entry: whether a call above the level is submitted
/// for approval rather than refused. Reads the same cell the dispatch door reads.
pub(super) fn approvals_config_option(permission: &SharedPermission) -> SessionConfigOption {
    SessionConfigOption::boolean(APPROVALS_CONFIG_ID, "Approvals", permission.approvals())
        .category(SessionConfigOptionCategory::Mode)
        .description(
            "Submit a call above the permission level for approval instead of refusing it.",
        )
}
/// The profile picker as a `configOptions` entry.
///
/// `current` may name no configured profile, which is what a session whose profile was deleted from
/// `config.toml` looks like. Nothing is invented for it: no option matches, so a client renders
/// "nothing selected", which is the truth. Inventing a selection would show a profile the session
/// is not going to run on.
pub(super) fn profile_config_option(
    profiles: &std::collections::BTreeMap<String, crate::config::ProfileConfig>,
    current: &str,
) -> SessionConfigOption {
    SessionConfigOption::select(
        PROFILE_CONFIG_ID,
        "Profile",
        SessionConfigValueId::from(current.to_string()),
        profiles
            .iter()
            .map(|(name, profile)| {
                let option = SessionConfigSelectOption::new(
                    SessionConfigValueId::from(name.clone()),
                    name.clone(),
                );
                // The model, when the profile names one. A profile that leaves it to the provider
                // has nothing truthful to put here, and inventing a label would be meka asserting
                // a fact about someone else's system.
                match &profile.model {
                    Some(model) => option.description(model.clone()),
                    None => option,
                }
            })
            .collect::<Vec<_>>(),
    )
    .category(SessionConfigOptionCategory::Model)
    .description("The profile this session runs on.")
}
/// Build the `configOptions` list advertised on every session-creation response and returned by
/// `session/set_config_option`.
///
/// The profile's current value is read from the session row rather than from the live agent: the
/// row is what the next turn resolves against, and the agent is behind the runtime mutex that an
/// in-flight prompt holds.
pub(super) async fn build_config_options(
    shared: &crate::host::SharedDeps,
    permission: &SharedPermission,
    session_uuid: Option<uuid::Uuid>,
) -> Vec<SessionConfigOption> {
    let current_profile = match session_uuid {
        Some(session_uuid) => shared
            .store
            .recorded_profile(session_uuid)
            .await
            .unwrap_or_else(|error| {
                tracing::warn!(
                    "failed to read the recorded profile for session {session_uuid}: {error}"
                );
                None
            })
            .unwrap_or_default(),
        None => String::new(),
    };
    vec![
        permission_config_option(permission),
        profile_config_option(&shared.config.profiles, &current_profile),
        approvals_config_option(permission),
    ]
}
/// Push a `config_option_update` so a client's pickers reflect a change it did not make, whether
/// that was another surface repinning the session or meka's own `session/set_mode` handler.
pub(super) async fn emit_config_options(
    state: &ServerState,
    entry: &SessionEntry,
    session_uuid: Option<uuid::Uuid>,
) {
    let options =
        build_config_options(&state.shared, &entry.cells().permission, session_uuid).await;
    send_session_update(
        &entry.frontend.connection,
        &entry.frontend.session_id,
        SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate::new(options)),
    );
}
