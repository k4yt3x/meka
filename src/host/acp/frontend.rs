//! The ACP client as a [`crate::frontend::Frontend`]: session updates, permission round-trips,
//! live terminal output, and the mode and config options an editor is shown.

use super::*;
use crate::{
    frontend::{Delegation, PromptedCall},
    host::{LiveOutput, LiveOutputMode},
};

/// Classify an `fs/*` failure into the one distinction the file tools route on.
///
/// `ResourceNotFound` (`-32002`) is the protocol's way for a client to say it will not serve a
/// path. Which paths those are is the client's business and differs between editors -- Zed answers
/// it for anything outside the project it has open, another client may serve any absolute path --
/// which is exactly why Meka asks per path instead of modeling any editor's rule. Every other
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

    /// Whether the client renders `compaction_update`, which the protocol lets an agent send only
    /// once the client advertised `session.compaction`.
    pub(super) fn renders_compactions(&self) -> bool {
        crate::sync::read(&self.inner)
            .capabilities
            .session
            .as_ref()
            .is_some_and(|session| session.compaction.is_some())
    }

    /// Whether the client renders `notice`, which the protocol lets an agent send only once the
    /// client advertised `session.notices`.
    pub(super) fn renders_notices(&self) -> bool {
        crate::sync::read(&self.inner)
            .capabilities
            .session
            .as_ref()
            .is_some_and(|session| session.notices.is_some())
    }

    /// Whether the client renders agent-owned terminals from Meka's `_meta` frames.
    ///
    /// Deliberately not the typed `terminal` capability: that one says the client implements
    /// `terminal/*` so an agent can run commands *in the client*, which Meka never does. A client
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
    /// Accumulated live output per in-flight tool call, keyed by `tool_use_id`. ACP replaces a
    /// tool call's whole `content` array on each update rather than appending to it, so the
    /// running total has to be kept somewhere; the emitter sends deltas, and this is where
    /// they are added up. Entries are dropped when the call completes.
    pub(super) live_output: std::sync::Mutex<std::collections::HashMap<String, LiveOutput>>,
    /// The compaction checkpoint's calls a prompt opened a card for, which the compaction's report
    /// completes rather than opens: the prompt is the first the client hears of such a call.
    pub(super) prompted: std::sync::Mutex<std::collections::HashSet<String>>,
    /// The ids the client knows this session's tool calls by; see [`CallIds`]. The replay a load
    /// sends goes through it too, so a call after the load is told apart from the replayed ones.
    pub(super) call_ids: std::sync::Mutex<CallIds>,
    /// The same cell the session entry holds, so a client round-trip started by this frontend can
    /// be abandoned when `session/cancel` fires. Shared rather than copied: every turn installs a
    /// token of its own, and a frontend holding a stale clone would race against a token nobody
    /// signals.
    pub(super) cancel: crate::host::CancelCell,
    /// Whether the last update sent was a thought, which the next thought starts apart from.
    pub(super) after_thought: std::sync::atomic::AtomicBool,
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
/// The `_meta` key under which a `compaction_update` carries Meka's facts about the compaction:
/// what set it going, and once it lands, the fields of the `context.compacted` event.
pub(super) const META_COMPACTION: &str = "compaction";
/// The title of the card a running compaction is for a client without the protocol's compaction
/// entry: the screen's words for the same moment.
const COMPACTION_CARD_TITLE: &str = "Compacting the conversation";
/// The `_meta` key under which the notice for a background report carries the tasks that reported,
/// each as the HTTP API's task object.
pub(super) const META_BACKGROUND: &str = "background";
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

/// How an approval prompt was answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Answered {
    /// For this call alone.
    Once,
    /// For this call and every later call to the tool.
    ForTheSession,
    /// Not by the user: the prompt timed out, the transport failed, or the turn was stopped.
    Unanswered,
}

/// What the client already shows of a session when its frontend is built.
pub(super) enum EditorThread {
    /// Nothing: a new session, or one `session/load` is about to replay.
    Empty,
    /// A thread no frontend of this process sent: a resumed, forked or reopened session's, whose
    /// cards may carry any id the provider gave.
    Inherited,
}

/// The ids the client knows this session's tool calls by. The protocol wants a call's id unique
/// within the session, and a provider may give one id to every call, or reuse its ids from round
/// to round, while a client fills in the newest card it holds under an id. So a call whose id the
/// client may already hold is sent under one minted here, and every later report of the call goes
/// under what it was sent as.
pub(super) struct CallIds {
    /// Unique to this frontend, so an id minted here is one no earlier frontend sent.
    token: String,
    /// Whether the client may hold cards under ids this frontend never saw.
    inherited: bool,
    minted: usize,
    /// Each provider id's calls still running, by what they were sent as, oldest first.
    running: std::collections::HashMap<String, std::collections::VecDeque<String>>,
    /// Each provider id's newest call, by what it was sent as: every id this frontend has sent a
    /// call under.
    newest: std::collections::HashMap<String, String>,
    /// The calls that run a worker and whose worker has not taken its card yet, by what they were
    /// sent as, oldest first: a provider may give several calls one id, and a worker takes the
    /// oldest card under its call's id rather than the newest call. A worker is built while its
    /// call runs, so the oldest is its own call's, unless two calls under one id build workers at
    /// once and take each other's.
    workers: std::collections::HashMap<String, std::collections::VecDeque<String>>,
}

impl CallIds {
    pub(super) fn new(thread: EditorThread) -> Self {
        Self {
            token: uuid::Uuid::new_v4().simple().to_string()[..8].to_string(),
            inherited: matches!(thread, EditorThread::Inherited),
            minted: 0,
            running: std::collections::HashMap::new(),
            newest: std::collections::HashMap::new(),
            workers: std::collections::HashMap::new(),
        }
    }

    /// The id a call starting under the provider's `id` is sent as.
    pub(super) fn open(&mut self, id: &str) -> String {
        let sent = if !self.inherited && !self.newest.contains_key(id) {
            id.to_string()
        } else {
            self.minted += 1;
            format!("{id}#{}-{}", self.token, self.minted)
        };
        self.running
            .entry(id.to_string())
            .or_default()
            .push_back(sent.clone());
        self.newest.insert(id.to_string(), sent.clone());
        sent
    }

    /// What the oldest call still running under the provider's `id` was sent as.
    pub(super) fn running(&self, id: &str) -> Option<String> {
        self.running
            .get(id)
            .and_then(|calls| calls.front())
            .cloned()
    }

    /// What the call a report under the provider's `id` is about was sent as: the oldest still
    /// running, else the newest, as a sub-agent left running after its call returned reports.
    pub(super) fn current(&self, id: &str) -> Option<String> {
        self.running(id).or_else(|| self.newest.get(id).cloned())
    }

    /// End the oldest call running under the provider's `id`, and say what it was sent as. A
    /// completion for a call that never started is sent under the id it names.
    pub(super) fn close(&mut self, id: &str) -> String {
        self.close_running(id)
            .or_else(|| self.newest.get(id).cloned())
            .unwrap_or_else(|| id.to_string())
    }

    /// End the oldest call running under the provider's `id`, if one is, and say what it was sent
    /// as. A card the call queued for a worker goes with it: a worker is built while its call
    /// runs, so a card still queued belongs to a call refused before its worker started, and the
    /// next worker under the id would take it.
    pub(super) fn close_running(&mut self, id: &str) -> Option<String> {
        let closed = self
            .running
            .get_mut(id)
            .and_then(std::collections::VecDeque::pop_front);
        if self
            .running
            .get(id)
            .is_some_and(std::collections::VecDeque::is_empty)
        {
            self.running.remove(id);
        }
        if let Some(sent) = &closed
            && let Some(cards) = self.workers.get_mut(id)
        {
            cards.retain(|card| card != sent);
            if cards.is_empty() {
                self.workers.remove(id);
            }
        }
        closed
    }

    /// Keep the card of a call that runs a worker, `sent` under the provider's `id`, for its worker
    /// to take.
    pub(super) fn await_worker(&mut self, id: &str, sent: &str) {
        self.workers
            .entry(id.to_string())
            .or_default()
            .push_back(sent.to_string());
    }

    /// The card a worker started by a call under the provider's `id` reports on: the oldest such
    /// call whose worker has not taken its card, else the call [`Self::current`] names.
    fn take_worker(&mut self, id: &str) -> Option<String> {
        let taken = self
            .workers
            .get_mut(id)
            .and_then(std::collections::VecDeque::pop_front);
        if self
            .workers
            .get(id)
            .is_some_and(std::collections::VecDeque::is_empty)
        {
            self.workers.remove(id);
        }
        taken.or_else(|| self.current(id))
    }

    /// The card a worker started by a call under the provider's `id` reports its activity on,
    /// while `live` says a live view is open on that call: [`Self::take_worker`]'s, else `id`. A
    /// call in the background shows its worker on no card, and still takes the one it queued,
    /// which a worker another call under the id builds before this call ends would otherwise
    /// take.
    pub(super) fn worker_card(&mut self, id: &str, live: bool) -> Option<String> {
        let card = self.take_worker(id).unwrap_or_else(|| id.to_string());
        live.then_some(card)
    }

    /// Let go of every card still queued, at the turn's end: a call abandoned there never closes
    /// to let go of its own.
    pub(super) fn forget_workers(&mut self) {
        self.workers.clear();
    }

    /// End every call still running, and say what each was sent as.
    pub(super) fn abandon_running(&mut self) -> Vec<String> {
        self.running.drain().flat_map(|(_, calls)| calls).collect()
    }
}

impl AcpFrontend {
    pub(super) fn new(
        link: AcpClientLink,
        cwd: SharedCwd,
        cancel: crate::host::CancelCell,
        thread: EditorThread,
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
            live_output: std::sync::Mutex::new(std::collections::HashMap::new()),
            prompted: std::sync::Mutex::new(std::collections::HashSet::new()),
            call_ids: std::sync::Mutex::new(CallIds::new(thread)),
            cancel,
            after_thought: std::sync::atomic::AtomicBool::new(false),
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
        self.cancel
            .live()
            .map(|cancellation| cancellation.token().clone())
            .unwrap_or_default()
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

    /// Close the card of a call that never ran, as failed, with the terminal it opened ended.
    fn close_unrun(&self, id: String) {
        let terminal = self.live_output().remove(&id).map(|entry| entry.mode)
            == Some(LiveOutputMode::Terminal);
        let mut update = ToolCallUpdate::new(
            id.clone(),
            ToolCallUpdateFields::new().status(ToolCallStatus::Failed),
        );
        if terminal {
            update = update.meta(terminal_meta(
                META_TERMINAL_EXIT,
                serde_json::json!({ "terminal_id": id, "exit_code": null, "signal": null }),
            ));
        }
        self.send_update(SessionUpdate::ToolCallUpdate(update));
    }

    /// Put `tool_call` to the client and read its answer, raced against the turn's cancellation
    /// and the approval timeout, and how it was answered.
    async fn ask_permission(
        &self,
        request: &PermissionRequest,
        tool_call: ToolCallUpdate,
        options: Vec<PermissionOption>,
    ) -> (PermissionOutcome, Answered) {
        let acp_request =
            RequestPermissionRequest::new(self.session_id.clone(), tool_call, options);
        // Race the round-trip against the per-turn cancellation token. If `session/cancel` fires
        // while we're waiting for the client to answer the approval prompt, we resolve as
        // `Canceled` instead of holding the runtime mutex forever, which would block
        // `session/close` and `session/set_mode` too.
        let response = tokio::select! {
            biased;
            _ = request.cancellation.cancelled() => {
                return (PermissionOutcome::Canceled, Answered::Unanswered);
            }
            // The backstop the cancellation race alone does not provide. A client that is
            // *connected* but never answers -- an editor whose UI thread is wedged, a headless
            // harness that speaks ACP but implements no prompt -- fires no cancellation, so the
            // race above waits on it forever and the turn holds the runtime mutex with it. Deny on
            // expiry rather than allow: an unanswered prompt is not consent.
            _ = tokio::time::sleep(crate::frontend::APPROVAL_TIMEOUT) => {
                tracing::warn!(
                    "the client did not answer the approval prompt for '{tool}' within \
                     {timeout:?}; denying it",
                    tool = request.tool_name,
                    timeout = crate::frontend::APPROVAL_TIMEOUT,
                );
                return (PermissionOutcome::Deny, Answered::Unanswered);
            }
            result = self.connection.send_request(acp_request).block_task() => match result {
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
                    return (PermissionOutcome::Deny, Answered::Unanswered);
                }
            },
        };

        let mut answered = Answered::Once;
        let outcome =
            translate_permission_outcome(response.outcome, &request.tool_name, |sticky| {
                answered = Answered::ForTheSession;
                match sticky {
                    StickyDecision::AllowAlways => self.sticky.remember_allow(&request.tool_name),
                    StickyDecision::RejectAlways => self.sticky.remember_deny(&request.tool_name),
                }
            });
        if outcome == PermissionOutcome::Canceled {
            answered = Answered::Unanswered;
        }
        (outcome, answered)
    }

    /// Push one `session/update` to the client, latching the transport as dead if it fails.
    ///
    /// `send_notification` is synchronous, which is what lets the live-output path hold its buffer
    /// lock across a send to keep chunks ordered.
    pub(super) fn send_update(&self, update: SessionUpdate) {
        self.after_thought.store(
            matches!(update, SessionUpdate::AgentThoughtChunk(_)),
            std::sync::atomic::Ordering::Relaxed,
        );
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
    /// Only `shell_execute` qualifies: it is the one tool whose result can be minutes away, and
    /// the terminal frames below would be nonsense for anything that isn't a command. The mode is
    /// decided once, here, so the client can't be told about a terminal it never registered.
    pub(super) fn begin_live_output(&self, id: &str, tool_name: &str) -> Option<LiveOutputMode> {
        if !crate::host::streams_output(tool_name) {
            return None;
        }
        let mode = if self.client_state.renders_agent_terminals() {
            LiveOutputMode::Terminal
        } else {
            LiveOutputMode::Text
        };
        // The single most useful line when a user reports "I see the command but not its output":
        // it says whether the client asked for terminal rendering, which is the whole branch point.
        tracing::debug!("shell_execute {id} live output: {mode:?}");
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
            // screen signage an editor has its own UI for -- these two are the opposite case,
            // signals this frontend structurally cannot represent.
            FrontendEvent::ThinkingProgress { .. } | FrontendEvent::ThinkingEnded => return,
            // The editor relays the conversation, and an empty reply holds nothing to relay; the
            // turn's stop says how it ended. A round's stop has no place in the protocol; the
            // calls it made are told as they run.
            FrontendEvent::NoAnswer | FrontendEvent::ReplyStopped { .. } => return,
            // ACP takes reasoning whole, from `ThinkingBlock` below. The deltas carry the same text
            // and forwarding both would double every thought chunk in the thread.
            FrontendEvent::ThinkingDelta(_) => return,
            FrontendEvent::ThinkingBlock { content, .. } => thought_chunk(
                &content,
                self.after_thought
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            // ACP has no typed slot for a withheld block, so it is said as a thought, in the words
            // a replay of the log says it in.
            FrontendEvent::RedactedThinking => thought_chunk(
                crate::conversation::REDACTED_THINKING,
                self.after_thought
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
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
                ..
            } => {
                // No separate `pending` state in the agent loop, so the in-progress emit is the
                // first one the client sees. The title carries the resolved primary argument (the
                // command, file path, URL, ...) so editors show what's actually running instead of
                // a bare tool name; `raw_input` still carries the full argument object.
                let id = {
                    let mut call_ids = crate::sync::lock(&self.call_ids);
                    let sent = call_ids.open(&id);
                    if crate::tools::subagent::runs_a_worker(&name) {
                        call_ids.await_worker(&id, &sent);
                    }
                    sent
                };
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
                    // terminal with it, and Meka's per-session cwd (which `/cd` moves) is not
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
                let Some(id) = crate::sync::lock(&self.call_ids).running(&id) else {
                    return;
                };
                // stdout and stderr drain on separate tasks, so two deltas for the same call can
                // be in flight on two threads at once. Terminal mode appends whatever it is handed,
                // so draining the buffer and sending it have to be one atomic step: release the
                // lock in between and the two tasks can swap order, interleaving the terminal's
                // contents. Safe to hold across the send because `send_notification` is
                // synchronous -- nothing is awaited under the lock.
                let mut buffers = self.live_output();
                // Absent means the call never opened a live view (not `shell_execute`, or it
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
                display,
            } => {
                let id = crate::sync::lock(&self.call_ids).close(&id);
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
                    fields = fields.content(build_completion_content(
                        &name,
                        &content,
                        metadata,
                        display.as_ref(),
                    ));
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
                // Already the id the card was sent under; see `for_subagent`.
                let fields = ToolCallUpdateFields::new().content(vec![text_content_block(summary)]);
                SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(tool_call_id, fields))
            }
            FrontendEvent::Notice(notice) => {
                notice_update(&notice, self.client_state.renders_notices())
            }
            FrontendEvent::CompactionStarted { id, trigger } => {
                compaction_started_update(&self.client_state, id, trigger)
            }
            // An editor holds a transcript, not a mirror of the model's window, so nothing on
            // screen is invalidated; but the automatic compactions happen with nobody asking, and
            // a user whose next reply forgets the morning deserves to know why.
            FrontendEvent::Compacted {
                id,
                record,
                generation,
            } => compacted_update(&self.client_state, id, &record, generation, true),
            FrontendEvent::CompactionFailed {
                id,
                trigger,
                error,
                interrupted,
            } => compaction_failed_update(&self.client_state, id, trigger, error, interrupted),
            // Told as it finishes, since what it wrote stays written however the compaction ends.
            // A call put to the client has the card its prompt opened.
            FrontendEvent::CheckpointCallCompleted {
                id,
                name,
                input,
                content,
                is_error,
                display,
            } => {
                let call = checkpoint_call(
                    id,
                    &name,
                    &input,
                    Some((&content, is_error, display.as_ref())),
                    &self.cwd,
                );
                let prompted =
                    crate::sync::lock(&self.prompted).remove(call.tool_call_id.0.as_ref());
                if prompted {
                    SessionUpdate::ToolCallUpdate(ToolCallUpdate::from(call))
                } else {
                    SessionUpdate::ToolCall(call)
                }
            }
            FrontendEvent::McpProgress(update) => {
                // ACP has no protocol primitive for tool-progress streams. The screen renders these
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
            FrontendEvent::ChecklistUpdated { items } => {
                // The checklist maps onto ACP's plan panel; an emptied list is an empty plan,
                // which is how the panel clears once everything is disposed of.
                SessionUpdate::Plan(Plan::new(checklist_to_plan(&items)))
            }
            // The turn's totals are the prompt response's; the gauge follows the measurement.
            FrontendEvent::TokenUsage(_) => return,
            // The screen's context gauge as an ACP `usage_update`, once per provider round and on
            // every turn's rounds, so an editor's "tokens used / context window" moves while a long
            // tool loop runs rather than when it ends. Nothing until the window is known: a `size`
            // of zero is not a small window.
            FrontendEvent::ContextMeasured { used, window } => {
                if window == 0 {
                    return;
                }
                SessionUpdate::UsageUpdate(UsageUpdate::new(used, window))
            }
            FrontendEvent::TurnStarted { .. } => {
                // Tool calls begin and end inside a turn, so anything still open here belongs to a
                // previous one that never delivered its completion (a canceled turn, or a stream
                // retried after announcing a tool call). Those entries would otherwise accumulate
                // for the life of the session. `TurnFinished` is not a substitute: the agent loop
                // only emits it when the turn succeeded, which is exactly when there is nothing to
                // clean up.
                self.live_output().clear();
                return;
            }
            // The protocol has no place for the level a turn opened under.
            FrontendEvent::TurnOpened { .. } => return,
            // What reported is said as the screen says it; the report is the model's to read.
            FrontendEvent::BackgroundReported { tasks, .. } => {
                background_report_update(&tasks, self.client_state.renders_notices())
            }
            // Inbox items added behind a round, a nudge or a cut prompt: the editor did not type
            // them, so they are shown as the user turn they are, as an opening's are.
            FrontendEvent::InboxAppended { blocks } => {
                for block in &blocks {
                    if let crate::conversation::ContentBlock::Inbox { text, .. } = block {
                        self.send_update(super::schedule::out_of_band_prompt_update(text));
                    }
                }
                return;
            }
            // The message a turn opens with, or one an interrupt appends. What the editor did not
            // type is shown as the user turn it is, so the reply has its question above it; the
            // editor's own words and images are in its transcript already, and the context block
            // is Meka's, shown nowhere. A report is said as what reported, as one read mid-turn
            // is.
            FrontendEvent::MessageAppended {
                message, outcomes, ..
            } => {
                for block in &message.content {
                    let text = match block {
                        crate::conversation::ContentBlock::BackgroundReport { .. } => {
                            self.send_update(background_report_update(
                                &outcomes,
                                self.client_state.renders_notices(),
                            ));
                            continue;
                        }
                        crate::conversation::ContentBlock::Scheduled { text, .. }
                        | crate::conversation::ContentBlock::Inbox { text, .. } => text,
                        // What the editor sent, which it drew itself.
                        crate::conversation::ContentBlock::Text { .. }
                        | crate::conversation::ContentBlock::Invocation { .. }
                        | crate::conversation::ContentBlock::Resource { .. }
                        | crate::conversation::ContentBlock::TurnContext { .. }
                        | crate::conversation::ContentBlock::Nudge { .. }
                        | crate::conversation::ContentBlock::HarnessNote { .. }
                        | crate::conversation::ContentBlock::Summary { .. }
                        | crate::conversation::ContentBlock::Refusal { .. }
                        | crate::conversation::ContentBlock::Unrecognized { .. }
                        | crate::conversation::ContentBlock::Image { .. }
                        | crate::conversation::ContentBlock::Thinking { .. }
                        | crate::conversation::ContentBlock::RedactedThinking { .. }
                        | crate::conversation::ContentBlock::ToolUse { .. }
                        | crate::conversation::ContentBlock::ToolResult { .. } => continue,
                    };
                    self.send_update(super::schedule::out_of_band_prompt_update(text));
                }
                return;
            }
            // A call whose arguments never finished was never shown, so it has no card to close.
            FrontendEvent::ToolCallDropped { id, started } => {
                if !started {
                    return;
                }
                let dropped = crate::sync::lock(&self.call_ids).close_running(&id);
                if let Some(id) = dropped {
                    self.close_unrun(id);
                }
                return;
            }
            // A call still running when its turn ends never ran: its stream was cut and the turn
            // ended there. A stop the client asked for is the client's to mark, as the protocol
            // has it do for every call unfinished at a `session/cancel`.
            FrontendEvent::TurnEnded { ending, .. } => {
                let unrun = {
                    let mut call_ids = crate::sync::lock(&self.call_ids);
                    call_ids.forget_workers();
                    call_ids.abandon_running()
                };
                if ending.reason == Some(crate::store::turns::CancelReason::Client) {
                    let mut live = self.live_output();
                    for id in &unrun {
                        live.remove(id);
                    }
                } else {
                    for id in unrun {
                        self.close_unrun(id);
                    }
                }
                return;
            }
            // Screen signage. The editor named the session it created, and the `session/prompt`
            // response is what tells it the turn is over. A withdrawn prompt is a scheduled fire's,
            // never one the editor sent, so its transcript has nothing to amend. Spelled out rather
            // than left to a catch-all so a variant added later has to be placed here on purpose.
            FrontendEvent::SessionStarted { .. }
            | FrontendEvent::TurnFinished
            | FrontendEvent::PromptWithdrawn
            | FrontendEvent::InboxDelivered { .. }
            | FrontendEvent::Nudged { .. } => return,
        };

        self.send_update(update);
    }

    fn client_disconnected(&self) -> bool {
        self.transport_dead
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The default adapter, told the call that runs the worker by the id its card was sent under,
    /// so its activity names that card rather than whichever call holds the provider's id when it
    /// reports; see [`CallIds::worker_card`].
    fn for_subagent(
        &self,
        this: Arc<dyn Frontend>,
        child: uuid::Uuid,
        tool_call_id: Option<String>,
        live: bool,
    ) -> Arc<dyn Frontend> {
        let card =
            tool_call_id.and_then(|id| crate::sync::lock(&self.call_ids).worker_card(&id, live));
        Arc::new(crate::frontend::PermissionForwardingFrontend::new(
            this,
            card,
            Some(child),
        ))
    }

    async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome {
        // Honor sticky decisions from earlier `*_always` selections.
        if let Some(remembered) = self.sticky.remembered(&request.tool_name) {
            return remembered;
        }
        // A turn already stopped has nobody to ask, and a card opened for the question would only
        // close again.
        if request.cancellation.is_cancelled() {
            return PermissionOutcome::Canceled;
        }

        // The sticky options name the *tool*, because that is their scope: the decision is keyed on
        // the tool name alone and applies to every later call to it, whatever its arguments. The
        // prompt's title beside them is `<tool> <primary_param>`, and for `shell_execute` the
        // command line alone, so a bare "Always allow" reads as approving the command the user just
        // read, when it actually approves every shell command for the rest of the session. Spelling
        // the tool out is what makes the affordance and the semantics agree, and it is spelled the
        // way the screen spells it.
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

        // The call the editor already shows turns to awaiting confirmation rather than a second
        // card being drawn. One it was never shown is opened first, so the request names a call
        // the client knows: a checkpoint's, under the id the compaction's report completes it by,
        // or with no call in scope, as for a sub-agent's, a card of the prompt's own, closed once
        // answered since no report follows.
        let (tool_call_id, announced, owned) = match request.tool_call.clone() {
            Some(PromptedCall::Announced(id)) => {
                let sent = crate::sync::lock(&self.call_ids).current(&id);
                (sent.unwrap_or(id), true, false)
            }
            Some(PromptedCall::Unannounced(id)) => (id, false, false),
            None => (format!("permission-{}", uuid::Uuid::new_v4()), false, true),
        };
        // The title is the one line a client is sure to show, and it names the destination for
        // every write-shaped tool; `raw_input` and the content block carry every argument, so a
        // client that renders either shows what is being written and not only where. See
        // `PermissionRequest::input`.
        let title = tool_call_title(&request.tool_name, request.primary_param.as_deref());
        let content = vec![text_content_block(approval_arguments_block(&request.input))];
        if !announced {
            if !owned {
                crate::sync::lock(&self.prompted).insert(tool_call_id.clone());
            }
            self.send_update(SessionUpdate::ToolCall(
                ToolCall::new(tool_call_id.clone(), title.clone())
                    .kind(tool_kind_for(&request.tool_name))
                    .status(ToolCallStatus::Pending)
                    .content(content.clone())
                    .raw_input(request.input.clone()),
            ));
        }
        let fields = ToolCallUpdateFields::new()
            .kind(tool_kind_for(&request.tool_name))
            .title(title)
            .status(ToolCallStatus::Pending)
            .content(content)
            .raw_input(request.input.clone());
        let tool_call = ToolCallUpdate::new(tool_call_id.clone(), fields);

        let (outcome, answered) = self.ask_permission(&request, tool_call, options).await;
        // Said as the screen says the answer, since the card stands for the question: the call
        // itself is reported nowhere the client sees. A stop the client asked for is the client's
        // to mark, as it marks every call unfinished at a `session/cancel`; any other, a stopped
        // background task's, is not one it knows of.
        let client_stopped = outcome == PermissionOutcome::Canceled
            && self.cancel.live().and_then(|turn| turn.reason())
                == Some(crate::store::turns::CancelReason::Client);
        if owned && !client_stopped {
            let mut fields =
                ToolCallUpdateFields::new().status(if outcome == PermissionOutcome::Allow {
                    ToolCallStatus::Completed
                } else {
                    ToolCallStatus::Failed
                });
            if answered != Answered::Unanswered {
                fields = fields.content(vec![text_content_block(
                    crate::frontend::approval_answered(
                        &request.tool_name,
                        outcome == PermissionOutcome::Allow,
                        answered == Answered::ForTheSession,
                    ),
                )]);
            }
            self.send_update(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                tool_call_id,
                fields,
            )));
        }
        outcome
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
        if !self.accepts_fs_writes() {
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

    fn accepts_fs_writes(&self) -> bool {
        self.client_state.capabilities().fs.write_text_file
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
/// The update that says a notice: a `notice` for an advisory when the client renders one, else an
/// assistant-message chunk. The chunk's `[Meka]` prefix (`[Meka warn]` for a
/// [`crate::frontend::NoticeLevel::Warn`]) tells a person reading the transcript the line is
/// Meka's; a client that wants the severity advertises `session.notices` and reads it typed. A
/// stage direction is a chunk whatever the client renders: it annotates a place in the reply, and
/// a notice has no place.
///
/// Shorter than the label a model reads ahead of Meka's notes, since a person looking at their own
/// editor needs no gloss on the product's name.
pub(super) fn notice_update(
    notice: &crate::frontend::Notice,
    client_renders_notices: bool,
) -> SessionUpdate {
    // A stage direction lands between the reply it annotates and the reply that follows, in one
    // agent message the editor builds from chunks, so it has to bring its own paragraph breaks or
    // the two replies read as one sentence with a note in the middle.
    let text = match notice.kind {
        crate::frontend::NoticeKind::StageDirection => format!("\n\n({})\n\n", notice.text),
        crate::frontend::NoticeKind::Advisory if client_renders_notices => {
            let severity = match notice.level {
                crate::frontend::NoticeLevel::Info => NoticeSeverity::Info,
                crate::frontend::NoticeLevel::Warn => NoticeSeverity::Warning,
            };
            return SessionUpdate::Notice(agent_client_protocol::schema::v1::Notice::new(
                severity,
                notice.text.clone(),
            ));
        }
        crate::frontend::NoticeKind::Advisory => {
            let prefix = match notice.level {
                crate::frontend::NoticeLevel::Info => "[Meka]",
                crate::frontend::NoticeLevel::Warn => "[Meka warn]",
            };
            format!("{prefix} {}", notice.text)
        }
    };
    SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
        agent_client_protocol::schema::v1::TextContent::new(text),
    )))
}
/// The update that says what reported, in the screen's words. A `notice` when the client renders
/// one: the stage direction as its title, the tasks' labels as its description, and the tasks
/// themselves under `_meta.background`, so a client that wants the outcome has it without the
/// transcript carrying it. Otherwise the direction as a stage-direction chunk, as a nudge is said
/// and as a replay says it. The report the model read stays out of the transcript either way, as
/// every tool's output does.
pub(super) fn background_report_update(
    tasks: &[crate::store::background::BackgroundTask],
    client_renders_notices: bool,
) -> SessionUpdate {
    let direction = crate::background::report_direction(tasks);
    if !client_renders_notices {
        return notice_update(
            &crate::frontend::Notice::stage_direction(
                crate::frontend::NoticeLevel::Info,
                direction,
            ),
            false,
        );
    }
    let description = match tasks {
        [task] => crate::background::shown_label(task),
        _ => tasks
            .iter()
            .map(|task| {
                format!(
                    "{} {}",
                    task.short_id(),
                    crate::background::shown_label(task)
                )
            })
            .collect::<Vec<_>>()
            .join("\n"),
    };
    let views: Vec<crate::view::BackgroundTaskView> = tasks
        .iter()
        .cloned()
        .map(crate::view::BackgroundTaskView::from)
        .collect();
    SessionUpdate::Notice(
        agent_client_protocol::schema::v1::Notice::new(NoticeSeverity::Info, direction)
            .description(description)
            .meta(terminal_meta(
                META_BACKGROUND,
                serde_json::json!({ "tasks": views }),
            )),
    )
}
/// Every argument of a call awaiting approval, as a fenced block for the prompt's content.
///
/// The title names the destination; this is what would be written there. Pretty-printed JSON
/// rather than the screen's key-per-line rendering because an editor renders markdown, where a code
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
/// Map an ACP outcome to Meka's [`PermissionOutcome`] and fire `record_sticky` when the user picked
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
/// Map Meka's tool name to ACP's [`ToolKind`] so clients can pick the right icon and grouping.
/// MCP-loaded tools (named `mcp__server__tool`) and anything unknown fall through to `Other`.
pub(super) fn tool_kind_for(name: &str) -> ToolKind {
    match name {
        "file_read" | "checklist_read" => ToolKind::Read,
        // A skill is a file in Meka's store, so its change opens the editor's diff view on it.
        "file_edit" | "file_write" | "skill_write" | "skill_edit" => ToolKind::Edit,
        "skill_delete" => ToolKind::Delete,
        "file_find" | "file_search" => ToolKind::Search,
        "shell_execute" => ToolKind::Execute,
        "web_fetch" => ToolKind::Fetch,
        // The list tools plan rather than touch anything, so a client never opens its diff view
        // for them.
        "agent_spawn" | "checklist_add" | "checklist_edit" => ToolKind::Think,
        // skill_read, memory_* (no file for a diff view to open), scratchpad_*, image_render,
        // tool_load, mcp__*, and any future built-ins.
        _ => ToolKind::Other,
    }
}
/// Build the human-readable `title` for a tool call: the tool's name, then the resolved primary
/// argument (`display_summary`: the path for `file_read`, the URL for `web_fetch`, ...), so editors
/// show what's running and not only which tool. The name is the one the screen's indicator and
/// approval prompt show, so the surfaces share one vocabulary. `raw_input` still carries the full
/// argument object for clients that want it.
///
/// A call of kind `execute` is the exception: its title is the command itself, whole, which is what
/// claude-agent-acp sends. A client treats an execute call's title as the command rather than as a
/// label: Zed shows it in the terminal card, its copy button copies it, and it shows no `raw_input`
/// for the kind, so a name in front corrupts the command and a collapsed or cut title is the only
/// copy of it the user ever sees.
pub(super) fn tool_call_title(name: &str, display_summary: Option<&str>) -> String {
    let argument = display_summary.map(str::trim).filter(|s| !s.is_empty());
    if tool_kind_for(name) == ToolKind::Execute
        && let Some(command) = argument
    {
        return command.to_string();
    }
    let raw = match argument {
        Some(argument) => format!("{name} {argument}"),
        None => name.to_string(),
    };
    sanitize_title(&raw)
}
/// Collapse internal whitespace and cap the length, so an editor never gets an unwieldy label.
///
/// The cap is the one claude-agent-acp puts on a session title; it puts none on a tool call's, but
/// its tool titles are a verb and a path, where Meka's primary argument can be a sub-agent's whole
/// prompt.
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
/// Convert Meka's checklist into ACP [`PlanEntry`] rows for [`SessionUpdate::Plan`]. A deferred
/// item is still open from the user's side and ACP has no fourth state, so it reports as
/// `Pending`, with what it waits on in its text. Completed and canceled items have left the list,
/// so the panel only ever shows what remains. Meka tracks no per-item priority, so every entry is
/// reported as `Medium`.
pub(super) fn checklist_to_plan(items: &[ChecklistItem]) -> Vec<PlanEntry> {
    items
        .iter()
        .map(|item| {
            let status = match item.status {
                ChecklistStatus::Pending | ChecklistStatus::Deferred => PlanEntryStatus::Pending,
                ChecklistStatus::InProgress => PlanEntryStatus::InProgress,
            };
            let content = match item.status {
                ChecklistStatus::Deferred => {
                    let reason = item.reason.as_deref().unwrap_or_default();
                    format!("{} (deferred: {reason})", item.text)
                }
                ChecklistStatus::Pending | ChecklistStatus::InProgress => item.text.clone(),
            };
            PlanEntry::new(content, PlanEntryPriority::Medium, status)
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
        "file_read" | "file_edit" | "file_write" | "file_find" | "file_search" => {
            input.get("path").and_then(|v| v.as_str())
        }
        _ => None,
    };
    raw.map(|path| {
        let mut location = ToolCallLocation::new(resolve_against_cwd(cwd, path));
        // For `file_read`, point the client at the first line being read; `start` counts from 1,
        // as ACP's `line` does.
        if name == "file_read"
            && let Some(start) = input.get("start").and_then(|value| value.as_u64())
        {
            location = location.line(u32::try_from(start).unwrap_or(u32::MAX));
        }
        vec![location]
    })
    .unwrap_or_default()
}
/// A recorded change as a unified diff in a `diff` code block. The record holds the change's
/// hunks, not the texts the protocol's `diff` needs, and each line numbered on one side only, so
/// the other side's numbers come from how far the hunks before have moved them: the record keeps
/// the hunks in order and a prefix of them, so those hold every change ahead of this one.
pub(super) fn recorded_change_block(change: &crate::conversation::Change) -> ToolCallContent {
    use crate::conversation::ChangeKind;
    let mut unified = String::new();
    let mut drift: isize = 0;
    for hunk in &change.hunks {
        let Some(first) = hunk.first() else {
            continue;
        };
        let old_lines = hunk
            .iter()
            .filter(|line| line.kind != ChangeKind::Added)
            .count();
        let new_lines = hunk
            .iter()
            .filter(|line| line.kind != ChangeKind::Removed)
            .count();
        let number = isize::try_from(first.number).unwrap_or(isize::MAX);
        let (old_start, new_start) = match first.kind {
            ChangeKind::Added => (number.saturating_sub(drift), number),
            ChangeKind::Kept | ChangeKind::Removed => (number, number.saturating_add(drift)),
        };
        // A side with no lines is numbered by the line before it, as a unified diff numbers it.
        let start = |start: isize, lines: usize| if lines == 0 { start - 1 } else { start };
        unified.push_str(&format!(
            "@@ -{},{old_lines} +{},{new_lines} @@\n",
            start(old_start, old_lines),
            start(new_start, new_lines)
        ));
        for line in hunk {
            unified.push(match line.kind {
                ChangeKind::Kept => ' ',
                ChangeKind::Removed => '-',
                ChangeKind::Added => '+',
            });
            unified.push_str(&line.text);
            unified.push('\n');
        }
        drift += isize::try_from(new_lines).unwrap_or(0) - isize::try_from(old_lines).unwrap_or(0);
    }
    let fence = fence_for(&unified);
    // The screen's words for the lines the record's budget left out.
    let left_out = match change.lines_left_out() {
        0 => String::new(),
        1 => "\n… +1 line".to_string(),
        lines => format!("\n… +{lines} lines"),
    };
    text_content_block(format!("{fence}diff\n{unified}{fence}{left_out}"))
}
/// A code fence no line of `text` can close: one backtick longer than the longest run opening a
/// line, and never shorter than the three a plain block uses. A memory or a skill holds fenced
/// code of its own, and an unchanged line of it sits behind the one space a diff gives context,
/// which is indentation a fence may carry, so a three-backtick block would end there.
fn fence_for(text: &str) -> String {
    let longest = text
        .lines()
        .map(|line| line.trim_start().chars().take_while(|c| *c == '`').count())
        .max()
        .unwrap_or(0);
    "`".repeat(longest.max(2) + 1)
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
/// Meka is holding.
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
/// Build the `content` array of a `tool_call_update` from Meka's tool output. A change the tool
/// reported wins: the words the tool wrote about it are the model's, and the change is what the
/// human reads. `shell_execute` output is wrapped in a `console` code block so editors render it
/// monospaced (mirrors claude-agent-acp's no-terminal fallback). Other tools pass their text and
/// image blocks through unchanged, so a tool that looked at an image (`file_read` on a PNG,
/// `image_render`, `web_fetch`) shows the human the same picture the model saw.
pub(super) fn build_completion_content(
    tool_name: &str,
    content: &[ToolResultContent],
    metadata: Option<ToolOutputMetadata>,
    display: Option<&crate::conversation::ResultDisplay>,
) -> Vec<ToolCallContent> {
    // A file the call left text in: the protocol's `diff`, which names the file and its text after
    // the call, so an editor opens its diff view on it. Only the live run has the texts.
    if let Some(ToolOutputMetadata::Diff {
        path: Some(path),
        old_text,
        new_text: Some(new_text),
    }) = metadata
    {
        let mut diff = Diff::new(path, new_text);
        if let Some(old) = old_text {
            diff = diff.old_text(old);
        }
        return vec![ToolCallContent::Diff(diff)];
    }
    // Any other change, a memory's, which has no file, or a delete's, which leaves no text, as its
    // result records it, so a replay shows it as the live run did.
    if let Some(change) = display
        .and_then(|display| display.change.as_ref())
        .filter(|change| !change.hunks.is_empty())
    {
        return vec![recorded_change_block(change)];
    }

    if tool_name == "shell_execute" {
        // The command's output as the console shows it: its text, a large output's opening as
        // the screen shows it, and no note Meka wrote to the model.
        let combined: String = content
            .iter()
            .filter_map(|block| match block {
                ToolResultContent::Text { text } => Some(text.as_str()),
                ToolResultContent::LargeOutput { preview, .. } => Some(preview.as_str()),
                ToolResultContent::Image { .. }
                | ToolResultContent::ToolReference { .. }
                | ToolResultContent::ImageCaption { .. }
                | ToolResultContent::HarnessNote { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if combined.trim_end().is_empty() {
            return Vec::new();
        }
        return vec![console_content_block(&combined)];
    }

    content
        .iter()
        .filter_map(|block| match block {
            ToolResultContent::Text { text } => Some(text_content_block(text.clone())),
            // The payload is already provider-normalized (size-capped, converted to a native
            // format) by the time it reaches here, so it can go out as-is. A reference has no
            // payload to show and is named instead.
            ToolResultContent::Image { source } => Some(match source.base64_data() {
                Some(data) => ToolCallContent::from(ContentBlock::Image(ImageContent::new(
                    data.to_string(),
                    source.media_type().to_string(),
                ))),
                None => text_content_block(crate::image::UNRESOLVED_IMAGE_PLACEHOLDER.to_string()),
            }),
            ToolResultContent::ToolReference { tool_name } => {
                Some(text_content_block(tool_name.clone()))
            }
            // The caption names the file in the transcript, as the model reads it; a large
            // output shows its opening, as the screen does.
            ToolResultContent::ImageCaption { .. } => match block.wire_part() {
                crate::conversation::ResultPart::Text(caption) => {
                    Some(text_content_block(caption.into_owned()))
                }
                _ => None,
            },
            ToolResultContent::LargeOutput { preview, .. } => {
                Some(text_content_block(preview.clone()))
            }
            // Meka's note to the model is the model's, not the tool's output.
            ToolResultContent::HarnessNote { .. } => None,
        })
        .collect()
}
/// One entry of a session's log as a replay says it: a message, or a compaction where the log has
/// a boundary.
pub(super) enum Replayed<'a> {
    /// A message, and whether it is a reply its record says was cut before it ended.
    Message {
        message: &'a crate::conversation::Message,
        cut: bool,
    },
    Compaction {
        id: String,
        record: &'a crate::conversation::CompactionRecord,
        generation: u64,
    },
    /// A compaction that ended without replacing the window, where the log records it.
    CompactionFailed {
        id: String,
        failure: &'a crate::conversation::FailedCompaction,
    },
}

/// A page of the log as a replay says it: each message, and each boundary as the compaction it
/// recorded, numbered from `first_generation`, by the rules every surface that shows history reads
/// a row by ([`crate::store::Said`]). The page is history, which holds no copy of the turns a
/// compaction kept, so each is said once, where it was first said.
pub(super) fn replay_entries<'a>(
    rows: &'a [crate::store::LogRow],
    rewound: &'a [crate::store::Rewound],
    first_generation: u64,
) -> Vec<Replayed<'a>> {
    use crate::store::Said;
    let mut entries: Vec<Replayed<'a>> = Vec::new();
    let mut generation = first_generation;
    for row in rows {
        let said = row.said(rewound);
        match said {
            Said::Message(message) => entries.push(Replayed::Message {
                message,
                cut: row.reply.as_ref().is_some_and(|reply| reply.cut.is_some()),
            }),
            Said::CompactionFailed { failure } => entries.push(Replayed::CompactionFailed {
                id: format!("replayed-failed-{}", row.id),
                failure,
            }),
            _ => {}
        }
        // Every boundary counts, one a rewind took included, as the history numbers them.
        if matches!(
            row.event,
            crate::conversation::Event::CompactBoundary { .. }
        ) {
            if let Said::Compaction { record } = said {
                entries.push(Replayed::Compaction {
                    id: format!("replayed-{generation}"),
                    record,
                    generation,
                });
            }
            generation += 1;
        }
        // The live run said nothing in an editor about a rewind or a joined prompt.
    }
    entries
}

/// The update a compaction's start is. A client that advertised `session.compaction` renders a
/// compaction entry of its own, opened here and closed by the update that next names the id; the
/// protocol lets an agent send the entry to no other client, so any other gets a card under the
/// same id instead.
pub(super) fn compaction_started_update(
    client_state: &SharedClientState,
    id: String,
    trigger: crate::conversation::CompactionTrigger,
) -> SessionUpdate {
    if client_state.renders_compactions() {
        SessionUpdate::CompactionUpdate(
            CompactionUpdate::new(id, CompactionStatus::InProgress).meta(terminal_meta(
                META_COMPACTION,
                serde_json::json!({ "trigger": trigger.name() }),
            )),
        )
    } else {
        SessionUpdate::ToolCall(
            ToolCall::new(id, COMPACTION_CARD_TITLE)
                .kind(ToolKind::Think)
                .status(ToolCallStatus::InProgress),
        )
    }
}

/// The update a compaction that ended without replacing the window is. A cancel is the user's own
/// act, so the entry says it by its status alone; a card has no canceled status, so it closes as
/// failed and says so.
pub(super) fn compaction_failed_update(
    client_state: &SharedClientState,
    id: String,
    trigger: crate::conversation::CompactionTrigger,
    error: String,
    interrupted: bool,
) -> SessionUpdate {
    if client_state.renders_compactions() {
        let status = if interrupted {
            CompactionStatus::Cancelled
        } else {
            CompactionStatus::Failed
        };
        let mut update = CompactionUpdate::new(id, status).meta(terminal_meta(
            META_COMPACTION,
            serde_json::json!({ "trigger": trigger.name(), "interrupted": interrupted }),
        ));
        if !interrupted {
            update = update.error(error);
        }
        return SessionUpdate::CompactionUpdate(update);
    }
    let said = if interrupted {
        "Interrupted".to_string()
    } else {
        format!("Compaction failed: {error}")
    };
    SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
        id,
        ToolCallUpdateFields::new()
            .status(ToolCallStatus::Failed)
            .content(vec![text_content_block(said)]),
    ))
}

/// The update a finished compaction is, live and on a replay alike: the protocol's compaction
/// entry for a client that advertised `session.compaction`, else the card claude-agent-acp shows
/// such a client, titled by the record's headline. `announced` when the compaction's start opened
/// the card, which a replay has no record of, so its report is the card's first.
pub(super) fn compacted_update(
    client_state: &SharedClientState,
    id: String,
    record: &crate::conversation::CompactionRecord,
    generation: u64,
    announced: bool,
) -> SessionUpdate {
    let facts = serde_json::json!(crate::view::CompactionView::new(record, generation));
    // An empty summary would leave an entry that opens onto nothing.
    let summary =
        (!crate::text::is_blank(&record.written_summary)).then(|| record.written_summary.clone());
    if client_state.renders_compactions() {
        let mut update = CompactionUpdate::new(id, CompactionStatus::Completed)
            .meta(terminal_meta(META_COMPACTION, facts));
        if let Some(summary) = summary {
            update = update.summary(vec![ContentBlock::Text(
                agent_client_protocol::schema::v1::TextContent::new(summary),
            )]);
        }
        return SessionUpdate::CompactionUpdate(update);
    }
    let particulars = record.particulars();
    let mut content = Vec::new();
    if !particulars.is_empty() {
        content.push(text_content_block(particulars.join(" ")));
    }
    if let Some(summary) = summary {
        content.push(text_content_block(summary));
    }
    if announced {
        let mut fields = ToolCallUpdateFields::new()
            .status(ToolCallStatus::Completed)
            .title(record.headline())
            .raw_output(facts);
        if !content.is_empty() {
            fields = fields.content(content);
        }
        SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(id, fields))
    } else {
        SessionUpdate::ToolCall(
            ToolCall::new(id, record.headline())
                .kind(ToolKind::Think)
                .status(ToolCallStatus::Completed)
                .content(content)
                .raw_output(facts),
        )
    }
}

/// The calls a compaction's checkpoint made, from its record, as the tool calls a replay shows
/// after the compaction, in the order the live run told them, each under the id
/// [`crate::frontend::checkpoint_call_id`] gives it. What the checkpoint thought is left out: a
/// thought after the compaction reads as the reply's.
///
/// A round's results are matched to its calls by place, as the record keeps them, since a
/// provider may give one id to every call.
pub(super) fn checkpoint_calls(
    compaction_id: &str,
    rounds: &[crate::conversation::RecordedMessage],
    cwd: &SharedCwd,
) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    for (index, round) in rounds.iter().enumerate() {
        let round = &round.message;
        if round.role != Role::Assistant {
            continue;
        }
        // Paired by place, as the round sent them, so calls a provider numbered alike keep their
        // own results.
        let mut results = rounds
            .get(index + 1)
            .map(|next| &next.message)
            .filter(|next| next.role == Role::User)
            .into_iter()
            .flat_map(|next| &next.content)
            .filter_map(|block| match block {
                MekaContentBlock::ToolResult {
                    content,
                    is_error,
                    display,
                    ..
                } => Some((content.as_slice(), *is_error, display.as_ref())),
                _ => None,
            });
        for block in &round.content {
            let MekaContentBlock::ToolUse { name, input, .. } = block else {
                continue;
            };
            let result = results.next();
            // The call that handed the summary in is the compaction's own, as the live run said.
            if name == "context_replace" {
                continue;
            }
            calls.push(checkpoint_call(
                crate::frontend::checkpoint_call_id(compaction_id, calls.len() + 1),
                name,
                input,
                result,
                cwd,
            ));
        }
    }
    calls
}

/// One call of a compaction's checkpoint as a finished tool call; `result` is `None` for one the
/// checkpoint never finished.
pub(super) fn checkpoint_call(
    id: String,
    name: &str,
    input: &serde_json::Value,
    result: Option<(
        &[ToolResultContent],
        bool,
        Option<&crate::conversation::ResultDisplay>,
    )>,
    cwd: &SharedCwd,
) -> ToolCall {
    let display_summary = crate::tools::resolve_primary_param(name, input);
    let call = ToolCall::new(id, tool_call_title(name, display_summary.as_deref()))
        .kind(tool_kind_for(name))
        .locations(tool_locations(name, input, cwd))
        .raw_input(input.clone());
    let Some((content, is_error, display)) = result else {
        return call.status(ToolCallStatus::Failed);
    };
    let mut call = call
        .status(if is_error {
            ToolCallStatus::Failed
        } else {
            ToolCallStatus::Completed
        })
        .content(build_completion_content(name, content, None, display));
    if let Ok(raw) = serde_json::to_value(content) {
        call = call.raw_output(raw);
    }
    call
}

/// Walk a session's replayed log and emit one `session/update` notification per content
/// block, mirroring the streaming shape the client would have seen had it been connected during
/// the original turn. Used by `session/load` so an editor that just reopened a session replays the
/// full history into its UI.
///
/// A user turn's context block is not replayed, so the client sees only what the user typed. A
/// background report is said as the stage direction the live run showed, from the session's task
/// rows in `tasks`, never as the words the model read.
///
/// Tool calls track open `tool_use_id`s; any tool that never received a matching `ToolResult` (e.g.
/// a crashed turn) is closed out with a `failed` `tool_call_update` so the client doesn't render a
/// stuck spinner.
pub(super) fn replay_session_updates(
    connection: &ConnectionTo<Client>,
    client_state: &SharedClientState,
    session_id: &SessionId,
    cwd: &SharedCwd,
    entries: &[Replayed<'_>],
    tasks: &[crate::store::background::BackgroundTask],
    call_ids: &std::sync::Mutex<CallIds>,
) {
    // Each open call's name, by the id it was sent as, so the result update can format output per
    // tool and the orphan sweep can close stragglers.
    let mut open_tools: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for entry in entries {
        let (message, cut) = match entry {
            Replayed::Message { message, cut } => (*message, *cut),
            // Said as the live run said it, never as the summary's words in a user turn: the
            // summary is the model's, and the turns before it are replayed above it.
            Replayed::Compaction {
                id,
                record,
                generation,
            } => {
                send_session_update(
                    connection,
                    session_id,
                    compacted_update(client_state, id.clone(), record, *generation, false),
                );
                for call in checkpoint_calls(id, &record.checkpoint, cwd) {
                    send_session_update(connection, session_id, SessionUpdate::ToolCall(call));
                }
                continue;
            }
            // Opened and closed as the live run opened and closed it, with the calls its
            // checkpoint made on the way, which the live run told as each finished.
            Replayed::CompactionFailed { id, failure } => {
                send_session_update(
                    connection,
                    session_id,
                    compaction_started_update(client_state, id.clone(), failure.trigger),
                );
                for call in checkpoint_calls(id, &failure.checkpoint, cwd) {
                    send_session_update(connection, session_id, SessionUpdate::ToolCall(call));
                }
                send_session_update(
                    connection,
                    session_id,
                    compaction_failed_update(
                        client_state,
                        id.clone(),
                        failure.trigger,
                        failure.error.clone().unwrap_or_default(),
                        failure.error.is_none(),
                    ),
                );
                continue;
            }
        };
        // A nudge is Meka's words, not the person's: the live run said it as a stage direction,
        // and the replay says the same, or nothing for the nudge after a reply with no visible
        // text, as live. What rode behind it, an inbox item, is the person's and follows as a
        // user chunk like any other text block.
        if let Some(note) = crate::render::nudge_direction(message) {
            send_session_update(
                connection,
                session_id,
                notice_update(
                    &crate::frontend::Notice::stage_direction(
                        crate::frontend::NoticeLevel::Info,
                        note.to_string(),
                    ),
                    false,
                ),
            );
        }
        match message.role {
            Role::User => {
                for block in &message.content {
                    match block {
                        MekaContentBlock::Text { text, .. } => {
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
                        // What reported, as the live run said it: a stage direction in the
                        // transcript rather than a notice, which is a live event and not history.
                        MekaContentBlock::BackgroundReport { task_ids, .. } => {
                            send_session_update(
                                connection,
                                session_id,
                                notice_update(
                                    &crate::frontend::Notice::stage_direction(
                                        crate::frontend::NoticeLevel::Info,
                                        crate::background::replayed_report_direction(
                                            tasks, task_ids,
                                        ),
                                    ),
                                    false,
                                ),
                            );
                        }
                        // A skill by the command an editor invokes it with; an MCP prompt, which
                        // an editor has no command for, by the stage direction the screen says.
                        // The words typed with it follow either way.
                        MekaContentBlock::Invocation {
                            kind,
                            name,
                            directive,
                            ..
                        } => {
                            let words = match kind {
                                crate::conversation::InvocationKind::Skill => {
                                    format!("/{name} {directive}").trim_end().to_string()
                                }
                                crate::conversation::InvocationKind::McpPrompt => {
                                    send_session_update(
                                        connection,
                                        session_id,
                                        notice_update(
                                            &crate::frontend::Notice::stage_direction(
                                                crate::frontend::NoticeLevel::Info,
                                                kind.direction(name),
                                            ),
                                            false,
                                        ),
                                    );
                                    directive.trim().to_string()
                                }
                            };
                            if !words.is_empty() {
                                send_session_update(
                                    connection,
                                    session_id,
                                    SessionUpdate::UserMessageChunk(ContentChunk::new(
                                        ContentBlock::Text(
                                            agent_client_protocol::schema::v1::TextContent::new(
                                                words,
                                            ),
                                        ),
                                    )),
                                );
                            }
                        }
                        // The mention the editor sent, where it sent it, without the contents.
                        MekaContentBlock::Resource { uri, name, .. } => {
                            send_session_update(
                                connection,
                                session_id,
                                SessionUpdate::UserMessageChunk(ContentChunk::new(
                                    ContentBlock::ResourceLink(
                                        agent_client_protocol::schema::v1::ResourceLink::new(
                                            name.clone(),
                                            uri.clone(),
                                        ),
                                    ),
                                )),
                            );
                        }
                        // What arrived for the model, as it read it: the editor showed it as the
                        // user turn it is while the run was live, and the replay says the same.
                        MekaContentBlock::Inbox { text, .. }
                        | MekaContentBlock::Scheduled { text, .. } => {
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
                        MekaContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                            display,
                        } => {
                            let status = if *is_error {
                                ToolCallStatus::Failed
                            } else {
                                ToolCallStatus::Completed
                            };
                            let id = crate::sync::lock(call_ids).close(tool_use_id);
                            let tool_name = open_tools.remove(&id).unwrap_or_default();
                            let mut fields = ToolCallUpdateFields::new().status(status).content(
                                build_completion_content(
                                    &tool_name,
                                    content,
                                    None,
                                    display.as_ref(),
                                ),
                            );
                            if let Ok(raw) = serde_json::to_value(content) {
                                fields = fields.raw_output(raw);
                            }
                            send_session_update(
                                connection,
                                session_id,
                                SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(id, fields)),
                            );
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
                // A call no result followed before the next answer never got one, as live, where
                // a call its round left unrun is dropped before the next round: closed now, so a
                // later result under its id is not taken for it.
                close_orphans(connection, session_id, call_ids, &mut open_tools);
                let mut after_thought = false;
                for block in &message.content {
                    // Of a reply cut before it ended, its words alone, which the live run sent as
                    // they came: its reasoning may not have reached an editor, or not whole, and
                    // its calls never ran, so the record holds what the live run may never have
                    // sent, and a replay says less rather than more.
                    if cut
                        && !matches!(
                            block,
                            MekaContentBlock::Text { .. } | MekaContentBlock::Refusal { .. }
                        )
                    {
                        continue;
                    }
                    if let Some(update) = reply_update(block, &mut after_thought) {
                        send_session_update(connection, session_id, update);
                        continue;
                    }
                    let MekaContentBlock::ToolUse {
                        id, name, input, ..
                    } = block
                    else {
                        continue;
                    };
                    after_thought = false;
                    let locations = tool_locations(name, input, cwd);
                    // Match the live path's rich title, which is read off the call alone.
                    let display_summary = crate::tools::resolve_primary_param(name, input);
                    let title = tool_call_title(name, display_summary.as_deref());
                    let id = crate::sync::lock(call_ids).open(id);
                    let call = ToolCall::new(id.clone(), title)
                        .kind(tool_kind_for(name))
                        .status(ToolCallStatus::InProgress)
                        .locations(locations)
                        .raw_input(input.clone());
                    send_session_update(connection, session_id, SessionUpdate::ToolCall(call));
                    open_tools.insert(id, name.clone());
                }
            }
        }
    }

    // Tool calls without a matching result: close them as failed so the client's "tool running"
    // indicator doesn't get stuck, and so a live call after the load is not reported as one of
    // them.
    close_orphans(connection, session_id, call_ids, &mut open_tools);
}

/// What a replay sends for a reply's `block` when it holds words or reasoning, as the live turn
/// sent it: a refusal's words as the model's, nothing for a part with nothing to read.
/// `after_thought` says whether the last update was a thought, and is kept up to date.
pub(super) fn reply_update(
    block: &MekaContentBlock,
    after_thought: &mut bool,
) -> Option<SessionUpdate> {
    let (update, thought) = match block {
        MekaContentBlock::Text { text, .. } | MekaContentBlock::Refusal { text, .. } => {
            if crate::text::is_blank(text) {
                return None;
            }
            let update = SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                agent_client_protocol::schema::v1::TextContent::new(text.clone()),
            )));
            (update, false)
        }
        MekaContentBlock::Thinking { parts, .. } => {
            let reasoning = crate::conversation::reasoning_text(parts);
            if crate::text::is_blank(&reasoning) {
                return None;
            }
            (thought_chunk(&reasoning, *after_thought), true)
        }
        MekaContentBlock::RedactedThinking { .. } => (
            thought_chunk(crate::conversation::REDACTED_THINKING, *after_thought),
            true,
        ),
        _ => return None,
    };
    *after_thought = thought;
    Some(update)
}

/// A thought as the chunk that carries it. The protocol marks no boundary between chunks, so one
/// that follows another thought starts a paragraph of its own.
fn thought_chunk(text: &str, after_thought: bool) -> SessionUpdate {
    let text = if after_thought {
        format!("\n\n{text}")
    } else {
        text.to_string()
    };
    SessionUpdate::AgentThoughtChunk(ContentChunk::new(ContentBlock::Text(
        agent_client_protocol::schema::v1::TextContent::new(text),
    )))
}

/// Close every call a replay has open as failed, and forget them as running.
fn close_orphans(
    connection: &ConnectionTo<Client>,
    session_id: &SessionId,
    call_ids: &std::sync::Mutex<CallIds>,
    open_tools: &mut std::collections::HashMap<String, String>,
) {
    if open_tools.is_empty() {
        return;
    }
    crate::sync::lock(call_ids).abandon_running();
    for orphan_id in open_tools.drain().map(|(id, _)| id) {
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
/// [`crate::conversation::Conversation::title`]'s definition applied by whoever holds the first
/// user words: the resident log after a first prompt, the session row on a load, resume or fork,
/// since a hydrated log begins at the last compaction boundary. It never changes once those words
/// are in, so `title_sent` guards against re-emission across the first prompt and any later
/// reopening of the same session.
pub(super) fn maybe_emit_session_title(
    connection: &ConnectionTo<Client>,
    session_id: &SessionId,
    title_sent: &std::sync::atomic::AtomicBool,
    title: String,
) {
    use std::sync::atomic::Ordering;
    if title_sent.load(Ordering::Acquire) {
        return;
    }
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
/// Map a Meka [`Permission`] to its ACP [`SessionModeId`] string: the same lowercase word
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
/// Kept in lock-step with the screen's `/permission` output and the `[permissions]` keys in
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
        Permission::Workspace => "Writes confined to the workspace.",
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
/// The project picker, offered only where the store has a project.
pub(super) const PROJECT_CONFIG_ID: &str = "project";
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
                // has nothing truthful to put here, and inventing a label would be Meka asserting
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
/// The project picker as a `configOptions` entry: every project behind a first entry with the
/// empty id, which is no project. No category: the protocol's are the model's and the mode's,
/// and a project is neither.
pub(super) fn project_config_option(projects: &[String], current: &str) -> SessionConfigOption {
    let mut options = vec![SessionConfigSelectOption::new(
        SessionConfigValueId::from(String::new()),
        "No project".to_string(),
    )];
    options.extend(projects.iter().map(|name| {
        SessionConfigSelectOption::new(SessionConfigValueId::from(name.clone()), name.clone())
    }));
    SessionConfigOption::select(
        PROJECT_CONFIG_ID,
        "Project",
        SessionConfigValueId::from(current.to_string()),
        options,
    )
    .description("The project this session is on, whose memories it sees beside the global ones.")
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
    let current_project = match session_uuid {
        Some(session_uuid) => shared
            .store
            .recorded_project(session_uuid)
            .await
            .unwrap_or_else(|error| {
                tracing::warn!(
                    "failed to read the recorded project for session {session_uuid}: {error}"
                );
                None
            })
            .unwrap_or_default(),
        None => String::new(),
    };
    let mut options = vec![
        permission_config_option(permission),
        profile_config_option(&shared.providers.catalog().profiles, &current_profile),
    ];
    // Offered only where there is a project to pick, so an installation without any shows the
    // pickers it always had. Read from the store at every build, so a project added while the
    // server runs is offered on the next update.
    let projects = shared.store.project_names().await.unwrap_or_else(|error| {
        tracing::warn!("failed to list projects for the config options: {error}");
        Vec::new()
    });
    if !projects.is_empty() {
        options.push(project_config_option(&projects, &current_project));
    }
    options.push(approvals_config_option(permission));
    options
}
/// Push a `config_option_update` so a client's pickers reflect a change it did not make, whether
/// that was another surface repinning the session or Meka's own `session/set_mode` handler.
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
