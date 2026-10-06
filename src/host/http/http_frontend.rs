//! `HttpFrontend`, the `meka serve` impl of [`crate::frontend::Frontend`].
//!
//! Blocking mode buffers every emitted event into the turn's recorder; mid-turn pause primitives
//! (permission approval, MCP elicitation) short-circuit to their safe defaults (`Deny`,
//! `Decline`) and append a diagnostic `Notice` so the caller can detect the misconfiguration.
//!
//! Every event also goes out on the session's feed: one `broadcast::Sender` per resident session,
//! installed when the session is loaded and living as long as it does, with a replay ring behind
//! it. `POST /turn` with `stream: true` is a view of that feed scoped to one turn;
//! `GET /v1/sessions/{id}/stream` is the feed itself, across turns, and is how a client sees the
//! turns nobody asked for over HTTP: a scheduled fire, a background outcome, an inbox item. The
//! pause primitives park on a `oneshot::Receiver` until the client POSTs to
//! `/v1/sessions/{id}/responses/{request_id}`, but only for a turn a streaming client attends.
//!
//! The HTTP API deliberately omits frontend-tool delegation (`delegate_fs_read` / `_fs_write` /
//! `_execute`). See the HTTP API docs. The
//! `Frontend` trait defaults already return `None`, which is the correct behavior (the agent
//! falls back to local I/O).

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, oneshot};

use super::sse::{EventIdGenerator, SseEvent, SseEventType, translate};
use crate::frontend::{
    APPROVAL_TIMEOUT, ElicitationPrompt, ElicitationResponse, Frontend, FrontendEvent, Notice,
    PermissionOutcome, PermissionRequest, StickyApprovals,
};

/// How often the parked `request_permission` poll checks whether the SSE consumer has
/// disconnected. `tokio::sync::broadcast::Sender` has no async "wait for subscriber count
/// change" primitive, so we poll `client_disconnected()` on a short interval. 500ms is fast
/// enough to feel instant to a human operator while consuming negligible CPU.
const DISCONNECT_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// The HTTP host's frontend, one per resident session, held on its `SessionEntry` for as long as
/// the session is resident. A turn handler reads the recorded events out of it to assemble the JSON
/// response body.
///
/// One HttpFrontend per session. The blocking-mode recorder is a [`Mutex`] around
/// [`Recorder`]; the session feed is installed once by [`Self::install_feed`], and `emit` fans
/// events out to both.
pub(crate) struct HttpFrontend {
    recorder: Mutex<Recorder>,
    /// Whether events are kept for a blocking response. A sub-agent's feed has no host draining
    /// it, so it publishes and keeps nothing.
    records: bool,
    /// What the feed was installed with, for the feeds of this session's sub-agents.
    wiring: Mutex<Option<FeedWiring>>,
    /// The session's event feed, and the turn currently publishing on it. `None` only on a
    /// frontend nobody installed a feed on, which no resident session is; the slot exists because
    /// the capacities come from the server's config, not from this type.
    feed: Mutex<Option<SessionFeed>>,
    /// In-memory parking lot for mid-turn pause primitives (`request_permission` and
    /// `handle_elicitation`). The HTTP turn handler emits an SSE event with the `request_id`,
    /// then `POST /v1/sessions/{id}/responses/{request_id}` pushes the resolution through the
    /// matching oneshot.
    pending: Arc<Mutex<HashMap<String, PermissionPending>>>,
    /// Per-session capabilities, declared at session creation. Controls SSE event filtering
    /// (`supports_reasoning_stream`) and whether a gated tool parks for approval or is denied
    /// outright (`supports_permission_prompts`).
    capabilities: SessionCapabilities,
    /// The `allow_always` and `deny_always` answers given this session, which short-circuit
    /// `request_permission` without re-emitting the SSE pause event.
    sticky: StickyApprovals,
    /// Set when the turn was canceled because its only SSE consumer fell behind, so the recorded
    /// terminal says so rather than blaming a client that never asked.
    canceled_for_lag: std::sync::atomic::AtomicBool,
    /// Output a running command has produced that the feed has not been sent yet, per tool call.
    /// See [`Self::wire_events`].
    live_output: Mutex<HashMap<String, crate::host::LiveOutput>>,
    /// Event ids, monotonic across the *session* rather than restarting per turn.
    ///
    /// Per-turn ids look tidier and make `Last-Event-ID` unusable: a client holding id 5 from one
    /// turn and reconnecting during the next cannot be told apart from one that is up to date,
    /// because the new turn issues id 5 too. Filtering against it then discards the entire backlog
    /// as already-delivered. Session-scoped ids make a stale position sort strictly below
    /// everything the current turn emitted, so the ordinary `event.id > last` filter is correct
    /// without needing to know which turn the id came from.
    ids: Arc<EventIdGenerator>,
}

/// Per-session capabilities flags declared at create time. Defaults match the bot/bridge use
/// case (server handles everything locally; SSE clients get assistant text + tool calls but not
/// thinking deltas). See the HTTP API docs § "Capabilities".
///
/// `Serialize` / `Deserialize` are derived so the value can be persisted on the session row and
/// re-hydrated by `reattach::ensure_session_loaded` when a GC-evicted session is reconstructed.
/// `ToSchema` is derived so the field can ride on `SessionResponse` in the OpenAPI spec.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(default)]
pub(crate) struct SessionCapabilities {
    /// When `true`, the SSE stream includes `thinking.delta` events for extended-thinking
    /// content. Default `false` so chat-transcript clients (Telegram bridges, etc.) don't
    /// surface reasoning text inline.
    pub(crate) supports_reasoning_stream: bool,
    /// When `false`, a mid-turn permission request is denied immediately with a notice instead of
    /// parking on the SSE channel for [`APPROVAL_TIMEOUT`].
    ///
    /// Streaming mode otherwise assumes the consumer can answer a prompt, which is wrong for a
    /// service-to-service client: it wants streaming for liveness on a long turn and has no
    /// interface to put an approval in front of, so a session with approvals on stalls for the
    /// whole approval timeout per gated call and then denies anyway. That reads as a hang rather
    /// than a misconfiguration. Setting this `false` gets blocking mode's behavior (immediate deny
    /// plus an explanatory notice), which is the same outcome without the stall and legible in
    /// the response rather than only in the timing.
    ///
    /// Defaults to `true`, so a client that declares nothing and an imported `capabilities_json`
    /// that omits the flag both park rather than refuse without asking.
    pub(crate) supports_permission_prompts: bool,
}

impl Default for SessionCapabilities {
    fn default() -> Self {
        Self {
            supports_reasoning_stream: false,
            supports_permission_prompts: true,
        }
    }
}

use super::feed::{Attendance, FEED_BROADCAST_CAPACITY, LiveTurn, SessionFeed, StreamAttachment};
pub(crate) use crate::host::scheduler::TurnSource;

/// One parked permission request: the channel its answer arrives on. What the prompt is about
/// rides the `permission_required` event, which the feed replays while the prompt is parked.
pub(crate) struct PermissionPending {
    pub(crate) sender: oneshot::Sender<PermissionResolution>,
}

/// The `permission_required` SSE event's payload: what a client needs to show the prompt and
/// answer it.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub(crate) struct PermissionRequiredEvent {
    /// Names the parked request on `POST /v1/sessions/{id}/responses/{request_id}`.
    pub(crate) request_id: String,
    pub(crate) tool_name: String,
    /// Every argument the tool was called with, plus `background` when the call would detach. The
    /// prompt has to show what is being written, not only where; see
    /// [`PermissionRequest::input`].
    pub(crate) input: serde_json::Value,
    /// How long the request stays answerable before it is denied, counted from when it was
    /// parked.
    pub(crate) expires_in_seconds: u64,
    /// RFC 3339, when the request is denied unanswered. The absolute form of
    /// `expires_in_seconds`, for a client that receives the event from the replay ring minutes
    /// after it was parked.
    pub(crate) expires_at: String,
    /// The sub-agent whose call this is, when a sub-agent asked through its parent's feed.
    /// Omitted for the session's own call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) subagent_id: Option<uuid::Uuid>,
}

/// Outcome carried by `POST /responses/{request_id}` for permission resolution. `*_always` also
/// records a sticky decision for the tool once the waiting turn has taken the answer.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PermissionResolution {
    Allow,
    AllowAlways,
    Deny,
    DenyAlways,
}

impl PermissionResolution {
    /// The decision's wire word, as `POST /responses/{request_id}` takes it and
    /// `permission_resolved` reports it.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::AllowAlways => "allow_always",
            Self::Deny => "deny",
            Self::DenyAlways => "deny_always",
        }
    }

    const fn outcome(self) -> PermissionOutcome {
        match self {
            Self::Allow | Self::AllowAlways => PermissionOutcome::Allow,
            Self::Deny | Self::DenyAlways => PermissionOutcome::Deny,
        }
    }
}

/// What a blocking-mode turn collects on its way to producing the JSON response. The turn
/// handler reads this back after `run_turn` returns.
pub(crate) type Recorder = Vec<FrontendEvent>;

impl HttpFrontend {
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::with_capabilities(SessionCapabilities::default())
    }

    pub(crate) fn with_capabilities(capabilities: SessionCapabilities) -> Self {
        Self {
            recorder: Mutex::new(Recorder::default()),
            records: true,
            wiring: Mutex::new(None),
            feed: Mutex::new(None),
            pending: Arc::new(Mutex::new(HashMap::new())),
            capabilities,
            sticky: StickyApprovals::default(),
            canceled_for_lag: std::sync::atomic::AtomicBool::new(false),
            live_output: Mutex::new(HashMap::new()),
            ids: Arc::new(EventIdGenerator::default()),
        }
    }

    /// Record that this turn is being canceled for a lagging consumer, before the token fires.
    pub(crate) fn note_canceled_for_lag(&self) {
        self.canceled_for_lag
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Whether the last cancel was for a lagging consumer. Cleared when a new turn begins.
    pub(crate) fn canceled_for_lag(&self) -> bool {
        self.canceled_for_lag
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    #[cfg(test)]
    fn is_always_allowed(&self, tool_name: &str) -> bool {
        self.sticky.remembered(tool_name) == Some(PermissionOutcome::Allow)
    }

    /// True while a streaming client's turn is in flight. A feed subscriber that did not attend is
    /// not one: it may be a bridge or a UI that shows no prompts, which is what
    /// [`Self::prompt_answerable`] asks about.
    fn is_streaming(&self) -> bool {
        let guard = crate::sync::lock(&self.feed);
        guard
            .as_ref()
            .and_then(|feed| feed.turn.as_ref())
            .is_some_and(|turn| turn.attended)
    }

    /// How many feed readers are attending the session right now.
    fn attending(&self) -> usize {
        let guard = crate::sync::lock(&self.feed);
        guard.as_ref().map_or(0, |feed| {
            feed.attending.load(std::sync::atomic::Ordering::SeqCst)
        })
    }

    /// Whether the turn in flight has a streaming client that can answer a prompt: one opened it
    /// over SSE, and the session said such a client shows prompts (`supports_permission_prompts`).
    /// The one place the flag is read, so parking and giving up agree about whom they are waiting
    /// on.
    fn streaming_client_answers(&self) -> bool {
        self.is_streaming() && self.capabilities.supports_permission_prompts
    }

    /// Whether anyone could answer a prompt parked now: the streaming client, if it can, or a feed
    /// reader attending the session, which is that declaration made per connection and so needs
    /// no session-level one. Decides park-or-deny; [`Self::prompt_abandoned`] decides, polled, when
    /// a parked prompt is given up on.
    fn prompt_answerable(&self) -> bool {
        self.streaming_client_answers() || self.attending() > 0
    }

    /// Whether everyone who could have answered a parked prompt has gone: no attendee is left and
    /// none has been for the reattach grace, and the streaming client either cannot answer or has
    /// been away past that same grace. Asked separately from [`Self::prompt_answerable`] because
    /// a reader inside its grace is neither: it cannot answer yet, and the prompt has to wait for
    /// it, as it always has. A client that declared it shows no prompts is not waited on at all:
    /// a prompt parked for an attendee that has since left would otherwise sit out the whole
    /// timeout on a client that never looks.
    fn prompt_abandoned(&self) -> bool {
        self.attenders_gone() && (!self.streaming_client_answers() || self.client_disconnected())
    }

    /// Whether no attending reader is left, and none has been for the feed's reattach grace,
    /// counted from the last one leaving. A reader that never attended has nobody to wait for; a
    /// browser tab reloading or waking from suspension is a departure only once the window it
    /// would have come back inside has passed, the same window a streaming client gets. The
    /// grace is the feed's, not a live turn's: a detached sub-agent parks its prompt after the
    /// parent's turn has ended.
    fn attenders_gone(&self) -> bool {
        let guard = crate::sync::lock(&self.feed);
        let Some(feed) = guard.as_ref() else {
            return true;
        };
        if feed.attending.load(std::sync::atomic::Ordering::SeqCst) > 0 {
            return false;
        }
        let Some(since) = *crate::sync::lock(&feed.unattended_since) else {
            return true;
        };
        let grace = feed.reattach_grace;
        drop(guard);
        since.elapsed() >= grace
    }

    /// Resolve a pending mid-turn permission request by `request_id`. Returns true iff the entry
    /// existed and the waiting turn took the answer. Called by
    /// `POST /v1/sessions/{id}/responses/{request_id}`.
    ///
    /// `request_permission` resolves through a `tokio::select!`, so its `oneshot::Receiver` is
    /// already dropped once the approval timeout expires, the turn is canceled, or the SSE client
    /// disconnects; a reply that loses that race is reported to the caller as
    /// `404 request-not-found`, and an answer nobody received grants nothing: the sticky decision
    /// an `*_always` carries is recorded by the waiter, on receipt, for that reason.
    pub(crate) fn resolve_permission(
        &self,
        request_id: &str,
        resolution: PermissionResolution,
    ) -> bool {
        let entry = {
            let mut guard = crate::sync::lock(&self.pending);
            guard.remove(request_id)
        };
        match entry {
            Some(pending) => pending.sender.send(resolution).is_ok(),
            None => false,
        }
    }

    /// How many prompts are parked on this session right now.
    pub(crate) fn approvals_pending(&self) -> u64 {
        crate::sync::lock(&self.pending).len() as u64
    }

    /// Swap the recorder out for an empty one and return what was collected. Called by the
    /// turn handler after `run_turn` returns; the per-session `HttpFrontend` lives across
    /// turns so consuming `self` isn't an option.
    pub(crate) fn drain(&self) -> Recorder {
        let mut guard = crate::sync::lock(&self.recorder);
        std::mem::take(&mut *guard)
    }

    /// Close the session's feed, when the session leaves the process: dropping the channel is
    /// what ends every `GET /stream` still reading it, since the stream holds this frontend and
    /// nothing else would. A turn on the frontend afterwards installs a fresh one.
    pub(crate) fn close_feed(&self) {
        crate::sync::lock(&self.feed).take();
    }

    /// Install the session's feed. Once, when the session becomes resident; a second call is a
    /// no-op, so the first turn on a frontend a test built bare can install one too.
    pub(crate) fn install_feed(&self, session_id: uuid::Uuid, wiring: &FeedWiring) {
        *crate::sync::lock(&self.wiring) = Some(wiring.clone());
        let mut guard = crate::sync::lock(&self.feed);
        guard.get_or_insert_with(|| {
            SessionFeed::new(
                session_id,
                Arc::clone(&self.ids),
                wiring.capacity,
                wiring.replay_capacity,
                wiring.webhooks.clone(),
                wiring.server.clone(),
                wiring.reattach_grace,
            )
        });
    }

    /// Publish and keep nothing: for a feed no host drains.
    fn without_recorder(mut self) -> Self {
        self.records = false;
        self
    }

    /// The parked prompt's events go to the feed of the sub-agent that asked as well, so a client
    /// watching the worker sees it waiting and sees the wait end.
    fn mirror_to_subagent(
        &self,
        subagent_id: Option<uuid::Uuid>,
        event_type: SseEventType,
        data: &serde_json::Value,
    ) {
        let Some(child) = subagent_id else {
            return;
        };
        let feed = crate::sync::lock(&self.wiring)
            .as_ref()
            .and_then(|wiring| wiring.children.as_ref())
            .and_then(|children| crate::sync::lock(children).get(&child).cloned());
        if let Some(feed) = feed {
            feed.push_sse(event_type, data.clone());
        }
    }

    /// Open a turn on the feed: subscribe, then announce it with a numbered `turn.started` that
    /// the subscription sees first. The receiver is taken before the announcement so a streaming
    /// handler misses nothing between the two.
    ///
    /// `attended` is whether a streaming client opened the turn; only then does
    /// [`Self::client_disconnected`] apply. Installs a feed if none is, since a turn on a
    /// frontend nobody installed one on would otherwise publish nowhere.
    pub(crate) fn begin_turn(
        &self,
        turn_id: uuid::Uuid,
        source: TurnSource,
        attended: bool,
        reattach_grace: Duration,
        replay_capacity: usize,
    ) -> (broadcast::Receiver<SseEvent>, Arc<EventIdGenerator>) {
        self.canceled_for_lag
            .store(false, std::sync::atomic::Ordering::SeqCst);
        let mut guard = crate::sync::lock(&self.feed);
        // A frontend nobody installed a feed on gets one here rather than publishing nowhere;
        // every resident session's was installed with its session id when it was loaded.
        let feed = guard.get_or_insert_with(|| {
            SessionFeed::new(
                uuid::Uuid::nil(),
                Arc::clone(&self.ids),
                FEED_BROADCAST_CAPACITY,
                replay_capacity,
                None,
                None,
                reattach_grace,
            )
        });
        let mut data = serde_json::json!({
            "turn_id": turn_id.to_string(),
            "started_at": chrono::Utc::now().to_rfc3339(),
        });
        source.describe(&mut data);
        feed.turn = Some(LiveTurn {
            turn_id,
            source,
            attended,
            disconnected_since: None,
            reattach_grace,
        });
        let receiver = feed.sender.subscribe();
        feed.publish(SseEventType::TurnStarted, data);
        let ids = Arc::clone(&self.ids);
        drop(guard);
        (receiver, ids)
    }

    /// [`Self::begin_turn`] for a streaming client's own turn, which is the attended case.
    pub(crate) fn install_stream(
        &self,
        capacity: usize,
        replay_capacity: usize,
        reattach_grace: Duration,
        turn_id: uuid::Uuid,
    ) -> (broadcast::Receiver<SseEvent>, Arc<EventIdGenerator>) {
        self.install_feed(
            uuid::Uuid::nil(),
            &FeedWiring::bare(capacity, replay_capacity, reattach_grace),
        );
        self.begin_turn(
            turn_id,
            TurnSource::Client,
            true,
            reattach_grace,
            replay_capacity,
        )
    }

    /// Record and broadcast a turn's terminal event. Kept as the feed's terminal for a client
    /// that reconnects after the turn is over; see [`SessionFeed`].
    pub(crate) fn record_terminal(
        &self,
        event_type: SseEventType,
        data: serde_json::Value,
    ) -> SseEvent {
        let mut guard = crate::sync::lock(&self.feed);
        let Some(feed) = guard.as_mut() else {
            return SseEvent {
                id: Some(self.ids.next()),
                event_type,
                data,
            };
        };
        let turn_id = feed.turn.as_ref().map(|turn| turn.turn_id);
        let event = feed.publish(event_type, data);
        if let Some(turn_id) = turn_id {
            feed.terminal = Some((turn_id, event.clone()));
        }
        drop(guard);
        event
    }

    /// Publish an event the host assembled itself, one no [`FrontendEvent`] describes: an inbox
    /// item withdrawn or given up on.
    pub(crate) fn push_sse(&self, event_type: SseEventType, data: serde_json::Value) {
        let mut guard = crate::sync::lock(&self.feed);
        if let Some(feed) = guard.as_mut() {
            feed.publish(event_type, data);
        }
    }

    /// How many consumers are reading the feed right now.
    pub(crate) fn subscriber_count(&self) -> usize {
        let guard = crate::sync::lock(&self.feed);
        guard
            .as_ref()
            .map_or(0, |feed| feed.sender.receiver_count())
    }

    /// Close the turn's view of the feed. The feed itself stays open: the next turn, whoever
    /// starts it, publishes on the same channel and a subscriber keeps its position.
    pub(crate) fn end_turn(&self) {
        let mut guard = crate::sync::lock(&self.feed);
        if let Some(feed) = guard.as_mut() {
            feed.turn = None;
        }
    }

    /// The terminal event of the most recent turn, once it has one.
    ///
    /// Re-read after a live subscription closes rather than trusted from the attachment snapshot:
    /// a client that attached mid-turn captured `terminal: None` because the turn had not ended
    /// yet. Scoped to `turn_id`: a re-attacher that wakes after the next turn has already started
    /// must not be handed that turn's terminal, or `None` for a turn that actually succeeded.
    /// Returning `None` on a mismatch lets the caller say what is true: the turn ended and its
    /// outcome is no longer held here.
    pub(crate) fn recorded_terminal(&self, turn_id: uuid::Uuid) -> Option<SseEvent> {
        let guard = crate::sync::lock(&self.feed);
        guard
            .as_ref()
            .and_then(|feed| feed.terminal.as_ref())
            .filter(|(recorded, _)| *recorded == turn_id)
            .map(|(_, event)| event.clone())
    }

    /// Attach to the feed, replaying anything after `last_event_id`. With `attend`, the reader is
    /// counted as one that answers approval prompts for as long as the attachment lives.
    ///
    /// The backlog snapshot and the `subscribe()` happen under one lock, and [`Self::emit`] takes
    /// the same lock to append. That is what makes the replay gap-free: without it an event
    /// emitted between the snapshot and the subscribe would be in neither.
    ///
    /// `None` when no feed is installed.
    pub(crate) fn attach_stream(
        &self,
        last_event_id: Option<u64>,
        attend: bool,
    ) -> Option<StreamAttachment> {
        let mut guard = crate::sync::lock(&self.feed);
        let feed = guard.as_mut()?;
        // Someone is listening again, so the grace clock restarts. Cleared here and not only in
        // `client_disconnected`, which the agent loop reaches at provider-round boundaries: a
        // client that reconnects and drops again before the next boundary would otherwise have its
        // second grace measured from the *first* disconnect, and so get almost none of it.
        if let Some(turn) = feed.turn.as_mut() {
            turn.disconnected_since = None;
        }
        // Ids are session-monotonic, so an id at or above the high-water mark was never issued
        // here at all -- a fabricated value, or one carried over from a different session. Discard
        // it rather than filter against it, which would silently deliver nothing.
        let stale = last_event_id.is_some_and(|last| last >= feed.ids.peek());
        let resume_from = if stale { None } else { last_event_id };
        // Taking `pending` while holding `feed` is safe in this order only: `request_permission`
        // releases `pending` before it acquires `feed` (see `park_permission` / `emit_pause`),
        // and `resolve_permission` never touches `feed` at all, so there is no inversion.
        let still_pending = crate::sync::lock(&self.pending);
        let backlog: Vec<SseEvent> = feed
            .replay
            .iter()
            .filter(|event| resume_from.is_none_or(|last| event.id.is_some_and(|id| id > last)))
            // A pause is stateful, not additive. Replaying one the client already answered (or
            // that timed out) would put an approval prompt back on screen for a request that no
            // longer exists, and any decision sent for it comes back 404. Replay it only while it
            // is still actionable, which is exactly while it is still parked.
            .filter(|event| {
                if event.event_type != SseEventType::PermissionRequired {
                    return true;
                }
                event
                    .data
                    .get("request_id")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|request_id| still_pending.contains_key(request_id))
            })
            .cloned()
            .collect();
        drop(still_pending);
        // A hole exists only relative to a position the client actually claims. `id + 1` because
        // resuming from exactly the oldest retained id is contiguous.
        //
        // A client that names no `Last-Event-ID` is joining, not resuming, and has lost nothing:
        // warning it about the events before it arrived would fire on every first attach. A
        // *stale* id is always a gap, because whatever the client was following has ended.
        let gap = stale
            || match (resume_from, feed.replay.front()) {
                (Some(last), Some(oldest)) => {
                    oldest.id.is_some_and(|id| id > last.saturating_add(1))
                }
                // Replay is switched off (`stream_replay_events = 0`), so a client resuming from a
                // position has been handed nothing between there and now. Reporting no gap would
                // be the silent truncation the notice exists to rule out.
                (Some(_), None) => feed.replay_capacity == 0,
                _ => false,
            };
        let turn_id = feed.turn.as_ref().map(|turn| turn.turn_id);
        let turn_source = feed.turn.as_ref().map(|turn| turn.source.clone());
        // Handed over only when no turn is running: with one in flight, the live subscription is
        // where its terminal will arrive, and the previous turn's is history the ring already
        // replays.
        let terminal = match turn_id {
            Some(_) => None,
            None => feed.terminal.as_ref().map(|(_, event)| event.clone()),
        };
        let attendance = attend.then(|| {
            feed.attending
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // A reader back inside the window is a reconnect, not a new departure to wait on.
            *crate::sync::lock(&feed.unattended_since) = None;
            Attendance {
                attending: Arc::clone(&feed.attending),
                unattended_since: Arc::clone(&feed.unattended_since),
            }
        });
        let attachment = StreamAttachment {
            turn_id,
            turn_source,
            attendance,
            backlog,
            receiver: feed.sender.subscribe(),
            terminal,
            gap,
            resume_from,
        };
        drop(guard);
        Some(attachment)
    }

    /// Drop SSE events the per-session capabilities don't enable. Currently only the
    /// `thinking.delta` event is gated (clients opt in via
    /// `capabilities.supports_reasoning_stream`). Returns true if the event should reach the
    /// broadcast.
    fn event_passes_capability_filter(&self, event: &FrontendEvent) -> bool {
        match event {
            FrontendEvent::ThinkingDelta(_) | FrontendEvent::ThinkingBlock { .. } => {
                self.capabilities.supports_reasoning_stream
            }
            _ => true,
        }
    }

    /// Record a warn-level notice in the recorder (and broadcast it if streaming) without
    /// going through the `Frontend::emit` trait method, so it can be called from a context that
    /// already holds `&self` in a non-async fashion.
    async fn record_warn_notice(&self, notice: Notice) {
        self.emit(FrontendEvent::Notice(notice)).await;
    }

    /// Publish an event to the feed and the recorder. Public so a host can push a frontend event
    /// it assembled itself, such as the prompt of a scheduled fire.
    pub(crate) fn push_event(&self, event: FrontendEvent) {
        // Tool calls begin and end inside a turn, so live output still buffered here belongs to a
        // call that never delivered its completion (a canceled turn, or a stream retried after
        // announcing a tool call). Cleared at the next turn's start rather than at an end, as ACP
        // does: the agent only reports a finished turn when it succeeded, which is exactly when
        // there is nothing to clean up.
        if matches!(event, FrontendEvent::TurnStarted { .. }) {
            crate::sync::lock(&self.live_output).clear();
        }
        // Push to the broadcast BEFORE recording, so a slow blocking-mode Mutex can't delay live
        // subscribers.
        //
        // The feed lock is held across the entire publish (`ids.next()` + `send`) so concurrent
        // emitters can't reorder monotonic ids. `broadcast::Sender::send` is synchronous, so
        // there's no await-under-lock hazard.
        {
            let mut guard = crate::sync::lock(&self.feed);
            if let Some(feed) = guard.as_mut()
                && self.event_passes_capability_filter(&event)
            {
                for (event_type, data) in self.wire_events(&event) {
                    feed.publish(event_type, data);
                }
            }
        }

        if self.records {
            crate::sync::lock(&self.recorder).push(event);
        }
    }

    /// The wire events one frontend event becomes: none, one, or, for a tool call completing with
    /// output still buffered, that output ahead of the completion.
    ///
    /// A command's output is coalesced per call through [`crate::host::LiveOutput`], as ACP does.
    /// The shell relays every read, and one wire event per read would let a chatty command lag the
    /// feed's readers, which for a `POST /turn` stream with no other reader cancels the turn.
    fn wire_events(&self, event: &FrontendEvent) -> Vec<(SseEventType, serde_json::Value)> {
        let delta = |id: &str, chunk: String| {
            translate(
                FrontendEvent::ToolCallOutputDelta {
                    id: id.to_string(),
                    chunk,
                },
                self.capabilities,
            )
        };
        match event {
            // A live view opens with the call and closes with its completion, as ACP's does. A
            // command run with `background: true` completes at once with its task id and keeps
            // writing for as long as it runs, into turns it has nothing to do with; its output
            // arrives with the task's outcome instead.
            FrontendEvent::ToolCallStarted { id, name, .. } => {
                if crate::host::streams_output(name) {
                    crate::sync::lock(&self.live_output).insert(
                        id.clone(),
                        crate::host::LiveOutput::new(crate::host::LiveOutputMode::Terminal),
                    );
                }
                translate(event.clone(), self.capabilities)
                    .into_iter()
                    .collect()
            }
            FrontendEvent::ToolCallOutputDelta { id, chunk } => {
                // Absent means the call has completed or never opened a view: dropped rather than
                // filed under whatever turn is running now.
                let due = crate::sync::lock(&self.live_output)
                    .get_mut(id)
                    .and_then(|entry| entry.push(chunk, std::time::Instant::now()));
                due.and_then(|text| delta(id, text)).into_iter().collect()
            }
            FrontendEvent::ToolCallCompleted { id, .. } => {
                let pending = crate::sync::lock(&self.live_output)
                    .remove(id)
                    .and_then(|mut entry| entry.take_pending());
                pending
                    .and_then(|text| delta(id, text))
                    .into_iter()
                    .chain(translate(event.clone(), self.capabilities))
                    .collect()
            }
            _ => translate(event.clone(), self.capabilities)
                .into_iter()
                .collect(),
        }
    }
}

#[async_trait]
impl Frontend for HttpFrontend {
    async fn emit(&self, event: FrontendEvent) {
        self.push_event(event);
    }

    fn for_subagent(
        &self,
        this: Arc<dyn Frontend>,
        child: uuid::Uuid,
        tool_call_id: Option<String>,
    ) -> Arc<dyn Frontend> {
        let forward = crate::frontend::PermissionForwardingFrontend::new(
            this,
            tool_call_id.clone(),
            Some(child),
        );
        let wiring = crate::sync::lock(&self.wiring).clone();
        let Some((wiring, children)) = wiring.and_then(|wiring| {
            let children = wiring.children.clone()?;
            Some((wiring, children))
        }) else {
            return Arc::new(forward);
        };
        let feed = Arc::new(HttpFrontend::with_capabilities(self.capabilities).without_recorder());
        // A worker's feed mirrors nothing to the server feed, where the registry announces its
        // record changes; a grandchild's feed is wired by this one, the feed above it.
        feed.install_feed(child, &FeedWiring {
            webhooks: None,
            server: None,
            children: Some(Arc::clone(&children)),
            ..wiring
        });
        crate::sync::lock(&children).insert(child, Arc::clone(&feed));
        let parent_id = crate::sync::lock(&self.feed)
            .as_ref()
            .map(|feed| feed.session_id);
        Arc::new(SubagentFeed {
            child,
            feed,
            forward,
            children,
            parent_id,
            tool_call_id,
            reattach_grace: wiring.reattach_grace,
            replay_capacity: wiring.replay_capacity,
        })
    }

    async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome {
        if let Some(remembered) = self.sticky.remembered(&request.tool_name) {
            return remembered;
        }

        if !self.prompt_answerable() {
            // A streaming client that declared it shows no prompts, or nobody at all.
            let reason = if self.is_streaming() {
                "the session declared supports_permission_prompts=false; raise its permission \
                 with `PATCH /v1/sessions/{id}`, or attend the feed with `attend=true`"
            } else {
                "no client is attending this turn; use `stream: true` or open the feed with \
                 `attend=true`"
            };
            self.record_warn_notice(Notice::approval_refused_without_asking_because(
                &request.tool_name,
                reason,
            ))
            .await;
            return PermissionOutcome::Deny;
        }

        let request_id = format!("req_{}", uuid::Uuid::new_v4());
        let (sender, receiver) = oneshot::channel::<PermissionResolution>();
        let parked_at = chrono::Utc::now();
        {
            let mut guard = crate::sync::lock(&self.pending);
            guard.insert(request_id.clone(), PermissionPending { sender });
        }

        // Hold the feed lock across `ids.next()` + `sender.send()` to preserve monotonic id
        // ordering, mirroring `emit()`.
        {
            let mut guard = crate::sync::lock(&self.feed);
            if let Some(feed) = guard.as_mut() {
                let payload = serde_json::to_value(PermissionRequiredEvent {
                    request_id: request_id.clone(),
                    tool_name: request.tool_name.clone(),
                    input: request.input.clone(),
                    expires_in_seconds: APPROVAL_TIMEOUT.as_secs(),
                    expires_at: (parked_at + APPROVAL_TIMEOUT).to_rfc3339(),
                    subagent_id: request.subagent_id,
                })
                .unwrap_or_else(|error| {
                    tracing::warn!("failed to serialize the permission_required payload: {error}");
                    serde_json::Value::Null
                });
                // Recorded like any other event: a client that reconnects mid-pause has to learn
                // that the turn is waiting on it, or the turn sits there until the timeout.
                feed.publish(SseEventType::PermissionRequired, payload.clone());
                drop(guard);
                self.mirror_to_subagent(
                    request.subagent_id,
                    SseEventType::PermissionRequired,
                    &payload,
                );
            }
        }

        // Poll-based disconnect detection: `broadcast::Sender` has no async "wait for
        // subscriber count change", so we check `prompt_abandoned()` on a short interval.
        // Without this, a client that drops the SSE connection while the turn is parked here
        // leaves the session stuck in `TurnInFlight` until the approval timeout or a manual
        // `POST /cancel`.
        let disconnect_poll = async {
            loop {
                tokio::time::sleep(DISCONNECT_POLL_INTERVAL).await;
                if self.prompt_abandoned() {
                    break;
                }
            }
        };

        // Every exit names how the prompt closed, so the feed can say so to a reader that was
        // not the one who answered.
        let (outcome, closed_as) = tokio::select! {
            biased;
            _ = request.cancellation.cancelled() => (PermissionOutcome::Canceled, "canceled"),
            _ = disconnect_poll => {
                tracing::info!(
                    "nobody is left to answer permission_required for '{tool}'; auto-canceling",
                    tool = request.tool_name,
                );
                (PermissionOutcome::Canceled, "canceled")
            },
            _ = tokio::time::sleep(APPROVAL_TIMEOUT) => (PermissionOutcome::Deny, "expired"),
            response = receiver => match response {
                Ok(resolution) => {
                    // Recorded by the side that took the answer, so a decision nobody received
                    // sticks to nothing.
                    match resolution {
                        PermissionResolution::AllowAlways => {
                            self.sticky.remember_allow(&request.tool_name);
                        }
                        PermissionResolution::DenyAlways => {
                            self.sticky.remember_deny(&request.tool_name);
                        }
                        PermissionResolution::Allow | PermissionResolution::Deny => {}
                    }
                    (resolution.outcome(), resolution.name())
                }
                Err(_) => (PermissionOutcome::Canceled, "canceled"),
            },
        };
        // Remove the entry if it's still there (timeout, cancellation, or disconnect paths).
        {
            let mut guard = crate::sync::lock(&self.pending);
            guard.remove(&request_id);
        }
        let resolved = serde_json::json!({ "request_id": request_id, "outcome": closed_as });
        self.push_sse(SseEventType::PermissionResolved, resolved.clone());
        self.mirror_to_subagent(
            request.subagent_id,
            SseEventType::PermissionResolved,
            &resolved,
        );
        outcome
    }

    async fn handle_elicitation(&self, prompt: ElicitationPrompt) -> ElicitationResponse {
        // The HTTP API doesn't expose MCP elicitation in either mode: service-to-service
        // callers can't render interactive prompts (see HTTP API docs § Approvals). The
        // notice surfaces the decline so operators can spot misconfigured servers that expect
        // to drive elicitation interactively.
        self.emit(FrontendEvent::Notice(Notice::elicitation_declined(
            &prompt.server_name,
            "the HTTP API does not expose interactive MCP prompts",
        )))
        .await;
        ElicitationResponse::Decline
    }

    // `delegate_fs_read` / `_fs_write` / `_execute` keep the trait defaults (all answer
    // `Delegation::Local`). The HTTP frontend does not expose client-hosted tool delegation.
    // Returning `None` routes the call to the agent's local I/O path.

    /// Reasoning leaves this frontend only as `thinking.delta`, which the stream tells clients to
    /// concatenate to rebuild the block. A retry that re-sent them would double whatever the client
    /// rebuilt, so a session receiving them has to cost the turn its retry.
    ///
    /// Both halves are needed. The capability alone is the session's *permission* to receive
    /// reasoning, not evidence that it did: a blocking turn has nothing listening, so the deltas
    /// reach the recorder and [`super::handlers::turn`] drops them there, serving whole blocks
    /// instead. Answering on the capability alone refuses the retry on every blocking turn of such
    /// a session, for reasoning nobody was sent, and reasoning is the first thing a turn produces.
    ///
    /// A block *can* complete and the attempt fail after it, so what keeps the whole-block readers
    /// safe is not that they hear nothing: it is that neither emitter of the block can be followed
    /// by a retry. The streaming one marks the turn started in the same branch; the non-streaming
    /// one re-emits a whole message only once the provider call has returned `Ok`, past its own
    /// retry loop.
    ///
    /// The second half is [`Self::is_streaming`] or an attending feed reader: both declared that
    /// they render what they are sent. A subscriber that did not attend does not count, because
    /// the feed is always there and a bridge holding it open may well be reading whole blocks off
    /// `GET /messages` afterwards; costing every turn its retry for that reader would be paying
    /// for a duplicate it never sees.
    fn retains_reasoning(&self) -> bool {
        self.capabilities.supports_reasoning_stream && (self.is_streaming() || self.attending() > 0)
    }

    /// SSE-mode disconnect detection, with a reconnect grace period.
    ///
    /// Zero remaining subscribers means the SSE consumer has dropped, and the agent loop
    /// short-circuits so we don't keep burning provider tokens for an audience that has gone away.
    /// It reports the disconnect only once the count has been zero for the turn's reattach grace,
    /// because a client whose connection dropped a moment ago and one that is never coming back
    /// are the same observation until the window expires. A reconnect through
    /// [`Self::attach_stream`] clears the stamp on its next poll.
    ///
    /// Only an attended turn can be disconnected from. A blocking turn has no transport-level
    /// disconnect to observe until the response writes complete, and a turn a driver started runs
    /// for the session whether or not anybody is reading, so the trait default `false` stands for
    /// both.
    fn client_disconnected(&self) -> bool {
        let mut guard = crate::sync::lock(&self.feed);
        let Some(feed) = guard.as_mut() else {
            return false;
        };
        let receivers = feed.sender.receiver_count();
        let Some(turn) = feed.turn.as_mut() else {
            return false;
        };
        if !turn.attended {
            return false;
        }
        if receivers > 0 {
            turn.disconnected_since = None;
            return false;
        }
        // Stamped and evaluated in one step so a zero grace means exactly no grace, rather than
        // "one poll interval": the first observation would otherwise always report `false`.
        let since = *turn
            .disconnected_since
            .get_or_insert_with(std::time::Instant::now);
        let grace = turn.reattach_grace;
        drop(guard);
        since.elapsed() >= grace
    }
}

/// What a session's feed is built with, and what a sub-agent's feed inherits from it.
#[derive(Clone)]
pub(crate) struct FeedWiring {
    pub(crate) capacity: usize,
    pub(crate) replay_capacity: usize,
    pub(crate) reattach_grace: Duration,
    pub(crate) webhooks: Option<super::webhook::WebhookDispatcher>,
    pub(crate) server: Option<super::feed::SharedServerFeed>,
    /// Where a sub-agent's feed is registered for as long as its run lasts, so the stream route
    /// can attach to it by the sub-agent's id.
    pub(crate) children: Option<ChildFeeds>,
}

/// The feeds of the sub-agents this process is running, by their session id.
pub(crate) type ChildFeeds = Arc<Mutex<HashMap<uuid::Uuid, Arc<HttpFrontend>>>>;

impl FeedWiring {
    /// A feed wired to nothing beyond itself: no webhooks, no server feed, no sub-agent feeds.
    pub(crate) fn bare(capacity: usize, replay_capacity: usize, reattach_grace: Duration) -> Self {
        Self {
            capacity,
            replay_capacity,
            reattach_grace,
            webhooks: None,
            server: None,
            children: None,
        }
    }
}

/// What a sub-agent runs under when its parent is an HTTP session: the forwarding adapter every
/// host uses, so its prompts reach the parent's feed and its calls roll into the `agent_spawn`
/// call's activity, plus a feed of the sub-agent's own, published under its id for as long as
/// the run lasts, which `GET /v1/sessions/{child}/stream` attaches to read-only.
pub(crate) struct SubagentFeed {
    child: uuid::Uuid,
    feed: Arc<HttpFrontend>,
    forward: crate::frontend::PermissionForwardingFrontend,
    children: ChildFeeds,
    parent_id: Option<uuid::Uuid>,
    tool_call_id: Option<String>,
    reattach_grace: Duration,
    replay_capacity: usize,
}

/// The terminal a sub-agent's feed records from the ending its row records: the same three
/// events a session's feed carries, built from the stored facts since no host holds the result.
pub(crate) fn subagent_terminal(
    turn_id: uuid::Uuid,
    session_id: uuid::Uuid,
    ending: &crate::store::turns::TurnEnding,
) -> (SseEventType, serde_json::Value) {
    use crate::store::turns::TurnStatus;
    let mut data = serde_json::json!({
        "turn_id": turn_id.to_string(),
        "session_id": session_id.to_string(),
    });
    match ending.status {
        TurnStatus::Succeeded => {
            data["stop_reason"] = serde_json::Value::from(ending.stop_reason.clone());
            data["usage"] = serde_json::to_value(ending.usage).unwrap_or(serde_json::Value::Null);
            (SseEventType::TurnFinished, data)
        }
        TurnStatus::Failed => {
            if let Some(error) = &ending.error {
                data["error"] = serde_json::json!({
                    "type": error.kind.type_uri(),
                    "title": error.kind.title(),
                    "status": super::errors::status_of(error.kind).as_u16(),
                    "detail": error.detail,
                });
            }
            (SseEventType::TurnFailed, data)
        }
        TurnStatus::Canceled => {
            // A worker stops when its parent's turn stops, whoever stopped that.
            data["reason"] = serde_json::Value::from("parent");
            (SseEventType::TurnCanceled, data)
        }
    }
}

#[async_trait]
impl Frontend for SubagentFeed {
    async fn emit(&self, event: FrontendEvent) {
        match &event {
            FrontendEvent::TurnStarted { turn_id } => {
                self.feed.begin_turn(
                    *turn_id,
                    TurnSource::Parent {
                        parent_id: self.parent_id,
                        tool_call_id: self.tool_call_id.clone(),
                    },
                    false,
                    self.reattach_grace,
                    self.replay_capacity,
                );
            }
            FrontendEvent::TurnEnded { turn_id, ending } => {
                let (event_type, data) = subagent_terminal(*turn_id, self.child, ending);
                self.feed.record_terminal(event_type, data);
                self.feed.end_turn();
                // The run is over at its terminal, whoever still holds this frontend (a detached
                // tool's task may): the feed leaves the registry so the route stops attaching to
                // it, and ends for whoever is reading, which is how they learn.
                self.unregister();
            }
            _ => self.feed.push_event(event.clone()),
        }
        self.forward.emit(event).await;
    }

    /// A grandchild's feed is wired by this feed, which has the registry, so it is its own and
    /// names this sub-agent as its parent; the default hook would run the grandchild's turn on
    /// this feed.
    fn for_subagent(
        &self,
        this: Arc<dyn Frontend>,
        child: uuid::Uuid,
        tool_call_id: Option<String>,
    ) -> Arc<dyn Frontend> {
        self.feed.for_subagent(this, child, tool_call_id)
    }

    async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome {
        self.forward.request_permission(request).await
    }

    async fn delegate_fs_read(
        &self,
        path: &std::path::Path,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> crate::frontend::Delegation<String> {
        self.forward.delegate_fs_read(path, line, limit).await
    }

    async fn delegate_fs_write(
        &self,
        path: &std::path::Path,
        content: &str,
    ) -> crate::frontend::Delegation<()> {
        self.forward.delegate_fs_write(path, content).await
    }

    async fn handle_elicitation(&self, prompt: ElicitationPrompt) -> ElicitationResponse {
        self.forward.handle_elicitation(prompt).await
    }

    fn client_disconnected(&self) -> bool {
        self.forward.client_disconnected()
    }
}

impl SubagentFeed {
    /// Take the feed out of the registry and close it. Safe to repeat: the terminal does it, and
    /// the drop does it again for a run that ended without one.
    fn unregister(&self) {
        crate::sync::lock(&self.children).remove(&self.child);
        self.feed.close_feed();
    }
}

impl Drop for SubagentFeed {
    fn drop(&mut self) {
        self.unregister();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wait until `count` permission prompts are parked, rather than sleeping and betting that
    /// the spawned task has registered its entry by then.
    async fn wait_for_pending(frontend: &HttpFrontend, count: usize) {
        for _ in 0..2000 {
            if frontend
                .pending
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .len()
                == count
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        panic!("{count} pending permission request(s) never registered");
    }
    use crate::frontend::{ElicitationKind, ElicitationPrompt, Frontend, FrontendEvent};

    /// `subscriber_count` has to see a re-attached second consumer, because that count is the only
    /// thing standing between one slow reader and everyone else's turn.
    ///
    /// The SSE stream task cancels the turn when a consumer lags, and must not do so
    /// unconditionally: turn events are a broadcast, so a re-attached client or a second consumer
    /// is a separate receiver, and one slow reader would kill the turn out from under the client
    /// that is keeping up. The guard is `subscriber_count() <= 1`. This pins the count's semantics,
    /// including the part the guard depends on: the lagging receiver, which is about to be dropped,
    /// is still included while it lives, which is why the threshold is `<= 1` rather than `== 0`.
    #[tokio::test]
    async fn subscriber_count_sees_every_live_consumer_of_a_turn() {
        let frontend = HttpFrontend::new();
        assert_eq!(
            frontend.subscriber_count(),
            0,
            "no stream installed yet, so nobody is reading"
        );

        let (first, _ids) = frontend.install_stream(
            16,
            16,
            Duration::from_secs(1),
            uuid::Uuid::from_u128(0xfeed),
        );
        assert_eq!(frontend.subscriber_count(), 1, "the turn's own consumer");

        let second = frontend
            .attach_stream(None, false)
            .expect("a live stream accepts a re-attach");
        assert_eq!(
            frontend.subscriber_count(),
            2,
            "a re-attached client is a second receiver; canceling on the first one's lag would \
             take the turn away from it"
        );

        drop(second);
        assert_eq!(
            frontend.subscriber_count(),
            1,
            "and once it goes, the lagging consumer speaks for the whole turn again"
        );
        drop(first);
        assert_eq!(frontend.subscriber_count(), 0);
    }

    /// A turn nobody opened over SSE (an inbox turn here) has nobody to wait on unless a feed
    /// reader said it would answer. A bare subscriber is a bridge or a UI with no prompt to show,
    /// so it is refused without asking; an attendee is asked.
    #[tokio::test]
    async fn a_bare_subscriber_is_refused_where_an_attendee_is_asked() {
        let frontend = Arc::new(HttpFrontend::new());
        frontend.install_feed(
            uuid::Uuid::nil(),
            &FeedWiring::bare(16, 16, Duration::from_secs(30)),
        );
        let (_receiver, _ids) = frontend.begin_turn(
            uuid::Uuid::from_u128(0x1),
            TurnSource::Inbox {
                item_ids: Vec::new(),
            },
            false,
            Duration::from_secs(30),
            16,
        );
        let request = || PermissionRequest {
            tool_name: "file_write".into(),
            primary_param: None,
            input: serde_json::Value::Null,
            cancellation: tokio_util::sync::CancellationToken::new(),
            subagent_id: None,
        };

        let _bare = frontend.attach_stream(None, false).expect("feed installed");
        let refused = tokio::time::timeout(
            Duration::from_secs(5),
            frontend.request_permission(request()),
        )
        .await
        .expect("a bare subscriber must not park the prompt");
        assert_eq!(refused, PermissionOutcome::Deny);

        let attendee = frontend.attach_stream(None, true).expect("feed installed");
        let asked = tokio::spawn({
            let frontend = Arc::clone(&frontend);
            async move { frontend.request_permission(request()).await }
        });
        wait_for_pending(&frontend, 1).await;
        let request_id = frontend
            .pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .next()
            .cloned()
            .expect("one parked prompt");
        assert!(frontend.resolve_permission(&request_id, PermissionResolution::Allow));
        assert_eq!(asked.await.expect("join"), PermissionOutcome::Allow);
        drop(attendee);
    }

    /// `supports_permission_prompts` speaks for the streaming client, which is the one a session
    /// declares it for. An attendee made the opposite declaration itself, per connection, so it is
    /// asked on a session whose creator (a bridge, say) said it had nobody to ask.
    #[tokio::test]
    async fn an_attendee_is_asked_on_a_session_that_declared_no_prompts() {
        let frontend = Arc::new(HttpFrontend::with_capabilities(SessionCapabilities {
            supports_permission_prompts: false,
            ..Default::default()
        }));
        frontend.install_feed(
            uuid::Uuid::nil(),
            &FeedWiring::bare(16, 16, Duration::from_secs(30)),
        );
        let (_receiver, _ids) = frontend.begin_turn(
            uuid::Uuid::from_u128(0x5),
            TurnSource::Inbox {
                item_ids: Vec::new(),
            },
            false,
            Duration::from_secs(30),
            16,
        );
        let attendee = frontend.attach_stream(None, true).expect("feed installed");
        let asked = tokio::spawn({
            let frontend = Arc::clone(&frontend);
            async move {
                frontend
                    .request_permission(PermissionRequest {
                        tool_name: "file_write".into(),
                        primary_param: None,
                        input: serde_json::Value::Null,
                        cancellation: tokio_util::sync::CancellationToken::new(),
                        subagent_id: None,
                    })
                    .await
            }
        });
        wait_for_pending(&frontend, 1).await;
        let request_id = frontend
            .pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .next()
            .cloned()
            .expect("one parked prompt");
        assert!(frontend.resolve_permission(&request_id, PermissionResolution::Allow));
        assert_eq!(asked.await.expect("join"), PermissionOutcome::Allow);
        drop(attendee);
    }

    /// A live view opens with `tool_call.executing` for a command and closes with its completion.
    /// Output arriving after that, which is what a command run with `background: true` produces
    /// for as long as it runs, is dropped rather than stamped with whatever turn is running then;
    /// so is output of a call that never opened a view.
    #[tokio::test]
    async fn output_of_a_completed_or_unopened_call_is_dropped() {
        let frontend = HttpFrontend::new();
        frontend.install_feed(
            uuid::Uuid::nil(),
            &FeedWiring::bare(16, 16, Duration::from_secs(30)),
        );
        let (mut receiver, _ids) = frontend.begin_turn(
            uuid::Uuid::from_u128(0x6),
            TurnSource::Client,
            false,
            Duration::from_secs(30),
            16,
        );
        let started = |id: &str, name: &str| FrontendEvent::ToolCallStarted {
            id: id.into(),
            name: name.into(),
            input: serde_json::Value::Null,
            display_summary: None,
        };
        let output = |id: &str, chunk: &str| FrontendEvent::ToolCallOutputDelta {
            id: id.into(),
            chunk: chunk.into(),
        };
        let completed = |id: &str, name: &str| FrontendEvent::ToolCallCompleted {
            id: id.into(),
            name: name.into(),
            is_error: false,
            content: Vec::new(),
            metadata: None,
        };
        frontend.push_event(started("tu_1", "shell_execute"));
        frontend.push_event(output("tu_1", "live\n"));
        frontend.push_event(completed("tu_1", "shell_execute"));
        // The detached command keeps writing after its call returned its task id.
        frontend.push_event(output("tu_1", "late\n"));
        // A tool that does not stream never opens a view, so nothing of its id is relayed.
        frontend.push_event(started("tu_2", "file_read"));
        frontend.push_event(output("tu_2", "never\n"));
        frontend.push_event(output("tu_9", "unknown\n"));

        let mut seen = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            seen.push(event);
        }
        let types: Vec<SseEventType> = seen.iter().map(|event| event.event_type).collect();
        assert_eq!(types, vec![
            SseEventType::TurnStarted,
            SseEventType::ToolCallExecuting,
            SseEventType::ToolCallOutputDelta,
            SseEventType::ToolCallCompleted,
            SseEventType::ToolCallExecuting,
        ]);
        assert_eq!(seen[2].data["chunk"], "live\n");
    }

    /// A prompt parked for an attendee is the attendee's to answer. When it leaves while a
    /// streaming client that declared it shows no prompts stays connected, nobody is left who
    /// will look, so once the grace is out the prompt is canceled rather than sat out for the
    /// whole timeout on that client; a zero grace is out at once.
    #[tokio::test]
    async fn a_prompt_parked_for_an_attendee_is_canceled_when_it_leaves_though_a_client_stays() {
        let frontend = Arc::new(HttpFrontend::with_capabilities(SessionCapabilities {
            supports_permission_prompts: false,
            ..Default::default()
        }));
        // The streaming client's own turn, with its stream held open for the whole test.
        let (_client, _ids) =
            frontend.install_stream(16, 16, Duration::ZERO, uuid::Uuid::from_u128(0x7));
        let attendee = frontend.attach_stream(None, true).expect("feed installed");
        let asked = tokio::spawn({
            let frontend = Arc::clone(&frontend);
            async move { frontend.request_permission(prompt_for_test()).await }
        });
        wait_for_pending(&frontend, 1).await;
        drop(attendee);
        let outcome = tokio::time::timeout(Duration::from_secs(5), asked)
            .await
            .expect("nobody left who will look, so the prompt is not sat out")
            .expect("join");
        assert_eq!(outcome, PermissionOutcome::Canceled);
    }

    /// The attendee is the one waiting on the prompt, so its leaving cancels the prompt once the
    /// reattach grace is out, the way a streaming client's disconnect does, rather than parking
    /// the turn for the whole timeout. Under a zero grace, at once.
    #[tokio::test]
    async fn a_parked_prompt_is_canceled_when_the_last_attendee_leaves_and_the_grace_is_out() {
        let frontend = Arc::new(HttpFrontend::new());
        frontend.install_feed(uuid::Uuid::nil(), &FeedWiring::bare(16, 16, Duration::ZERO));
        let (_receiver, _ids) = frontend.begin_turn(
            uuid::Uuid::from_u128(0x2),
            TurnSource::Background,
            false,
            Duration::ZERO,
            16,
        );
        let attendee = frontend.attach_stream(None, true).expect("feed installed");
        let asked = tokio::spawn({
            let frontend = Arc::clone(&frontend);
            async move { frontend.request_permission(prompt_for_test()).await }
        });
        wait_for_pending(&frontend, 1).await;
        drop(attendee);
        let outcome = tokio::time::timeout(Duration::from_secs(5), asked)
            .await
            .expect("the prompt is canceled once nobody can answer it")
            .expect("join");
        assert_eq!(outcome, PermissionOutcome::Canceled);
    }

    /// Inside the grace, a parked prompt outlives its attendee: a tab that reloads finds the
    /// prompt still parked and answers it, where cancel-on-disconnect would have failed the
    /// call under it.
    #[tokio::test]
    async fn a_parked_prompt_outlives_its_attendee_inside_the_grace() {
        let frontend = Arc::new(HttpFrontend::new());
        frontend.install_feed(
            uuid::Uuid::nil(),
            &FeedWiring::bare(16, 16, Duration::from_secs(30)),
        );
        let (_receiver, _ids) = frontend.begin_turn(
            uuid::Uuid::from_u128(0x2),
            TurnSource::Background,
            false,
            Duration::from_secs(30),
            16,
        );
        let attendee = frontend.attach_stream(None, true).expect("feed installed");
        let asked = tokio::spawn({
            let frontend = Arc::clone(&frontend);
            async move { frontend.request_permission(prompt_for_test()).await }
        });
        wait_for_pending(&frontend, 1).await;
        drop(attendee);
        // Longer than the disconnect poll, so the poll has looked and left the prompt alone.
        tokio::time::sleep(DISCONNECT_POLL_INTERVAL * 3).await;
        assert_eq!(
            frontend.approvals_pending(),
            1,
            "still parked inside the grace"
        );
        assert!(!asked.is_finished(), "and still waiting on an answer");

        let _back = frontend.attach_stream(None, true).expect("feed installed");
        let request_id = crate::sync::lock(&frontend.pending)
            .keys()
            .next()
            .cloned()
            .expect("the parked prompt");
        frontend.resolve_permission(&request_id, PermissionResolution::Allow);
        let outcome = tokio::time::timeout(Duration::from_secs(5), asked)
            .await
            .expect("answered")
            .expect("join");
        assert_eq!(outcome, PermissionOutcome::Allow);
    }

    /// A prompt parked with no turn live on the feed, as a detached sub-agent's is after the
    /// parent's turn ended, gets the feed's grace rather than none.
    #[tokio::test]
    async fn a_prompt_parked_between_turns_outlives_its_attendee_inside_the_grace() {
        let frontend = Arc::new(HttpFrontend::new());
        frontend.install_feed(
            uuid::Uuid::nil(),
            &FeedWiring::bare(16, 16, Duration::from_secs(30)),
        );
        let attendee = frontend.attach_stream(None, true).expect("feed installed");
        let asked = tokio::spawn({
            let frontend = Arc::clone(&frontend);
            async move { frontend.request_permission(prompt_for_test()).await }
        });
        wait_for_pending(&frontend, 1).await;
        drop(attendee);
        tokio::time::sleep(DISCONNECT_POLL_INTERVAL * 3).await;
        assert_eq!(
            frontend.approvals_pending(),
            1,
            "still parked inside the grace"
        );
        assert!(!asked.is_finished());
        let request_id = crate::sync::lock(&frontend.pending)
            .keys()
            .next()
            .cloned()
            .expect("the parked prompt");
        frontend.resolve_permission(&request_id, PermissionResolution::Deny);
        let outcome = tokio::time::timeout(Duration::from_secs(5), asked)
            .await
            .expect("answered")
            .expect("join");
        assert_eq!(outcome, PermissionOutcome::Deny);
    }

    /// A sub-agent's feed ends with its run, at the terminal, whoever still holds the frontend:
    /// a detached tool's task keeping it alive does not keep a feed open that has nothing left
    /// to say, and the route stops attaching to it.
    #[tokio::test]
    async fn a_sub_agents_feed_leaves_the_registry_at_its_terminal() {
        let children: ChildFeeds = Default::default();
        let root = Arc::new(HttpFrontend::new());
        root.install_feed(uuid::Uuid::from_u128(0x1), &FeedWiring {
            children: Some(Arc::clone(&children)),
            ..FeedWiring::bare(16, 16, Duration::from_secs(30))
        });
        let child = uuid::Uuid::from_u128(0x2);
        let root_as_frontend: Arc<dyn Frontend> = Arc::clone(&root) as Arc<dyn Frontend>;
        let child_frontend = root.for_subagent(root_as_frontend, child, Some("tu_1".into()));
        let turn_id = uuid::Uuid::from_u128(0x20);
        child_frontend
            .emit(FrontendEvent::TurnStarted { turn_id })
            .await;
        let feed = crate::sync::lock(&children)
            .get(&child)
            .cloned()
            .expect("registered while it runs");
        let attachment = feed.attach_stream(None, false).expect("attaches");
        child_frontend
            .emit(FrontendEvent::TurnEnded {
                turn_id,
                ending: crate::store::turns::TurnEnding::succeeded(
                    "end_turn",
                    crate::store::turns::TurnUsage::default(),
                ),
            })
            .await;
        assert!(
            crate::sync::lock(&children).get(&child).is_none(),
            "gone at the terminal, with the frontend still held"
        );
        let mut receiver = attachment.receiver;
        let mut seen = Vec::new();
        while let Ok(event) = receiver.recv().await {
            seen.push(event.event_type);
        }
        assert!(
            seen.contains(&SseEventType::TurnFinished),
            "a reader attached at the time still gets the terminal: {seen:?}"
        );
        drop(child_frontend);
    }

    /// The prompt every parking test parks: a gated write with nothing to show.
    fn prompt_for_test() -> PermissionRequest {
        PermissionRequest {
            tool_name: "file_write".into(),
            primary_param: None,
            input: serde_json::Value::Null,
            cancellation: tokio_util::sync::CancellationToken::new(),
            subagent_id: None,
        }
    }

    /// An attendee declared that it renders what it is sent, reasoning deltas included, so a turn
    /// it attends pays the same retry cost a streaming client's does. A subscriber that did not
    /// attend keeps the retry.
    #[tokio::test]
    async fn an_attendee_costs_a_reasoning_turn_its_retry() {
        let frontend = Arc::new(HttpFrontend::with_capabilities(SessionCapabilities {
            supports_reasoning_stream: true,
            ..Default::default()
        }));
        frontend.install_feed(
            uuid::Uuid::nil(),
            &FeedWiring::bare(16, 16, Duration::from_secs(30)),
        );
        let (_receiver, _ids) = frontend.begin_turn(
            uuid::Uuid::from_u128(0x3),
            TurnSource::Background,
            false,
            Duration::from_secs(30),
            16,
        );
        let _bare = frontend.attach_stream(None, false).expect("feed installed");
        assert!(
            !frontend.retains_reasoning(),
            "a bare subscriber keeps the retry"
        );
        let attendee = frontend.attach_stream(None, true).expect("feed installed");
        assert!(frontend.retains_reasoning(), "an attendee is a renderer");
        drop(attendee);
        assert!(!frontend.retains_reasoning());
    }

    /// Output deltas reach whoever is reading now, coalesced per call and flushed ahead of the
    /// call's completion, and leave no trace behind: no id, so the ids around them stay dense, and
    /// no place in the replay, so a reconnecting client gets the durable events alone.
    #[tokio::test]
    async fn a_transient_event_reaches_live_readers_but_never_the_replay() {
        let frontend = HttpFrontend::new();
        frontend.install_feed(
            uuid::Uuid::nil(),
            &FeedWiring::bare(16, 16, Duration::from_secs(30)),
        );
        let (mut receiver, _ids) = frontend.begin_turn(
            uuid::Uuid::from_u128(0x4),
            TurnSource::Client,
            false,
            Duration::from_secs(30),
            16,
        );
        let output = |chunk: &str| FrontendEvent::ToolCallOutputDelta {
            id: "tu_1".into(),
            chunk: chunk.into(),
        };
        frontend.push_event(FrontendEvent::ToolCallStarted {
            id: "tu_1".into(),
            name: "shell_execute".into(),
            input: serde_json::Value::Null,
            display_summary: None,
        });
        frontend.push_event(output("one\n"));
        // Inside the coalescing interval: buffered, not sent.
        frontend.push_event(output("two\n"));
        frontend.push_event(FrontendEvent::ToolCallCompleted {
            id: "tu_1".into(),
            name: "shell_execute".into(),
            is_error: false,
            content: Vec::new(),
            metadata: None,
        });
        frontend.push_event(FrontendEvent::AssistantTextDelta("done".into()));

        let mut seen = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            seen.push(event);
        }
        let types: Vec<SseEventType> = seen.iter().map(|event| event.event_type).collect();
        assert_eq!(types, vec![
            SseEventType::TurnStarted,
            SseEventType::ToolCallExecuting,
            SseEventType::ToolCallOutputDelta,
            SseEventType::ToolCallOutputDelta,
            SseEventType::ToolCallCompleted,
            SseEventType::AssistantTextDelta,
        ]);
        assert_eq!(seen[2].id, None);
        assert_eq!(seen[2].data["chunk"], "one\n");
        assert_eq!(
            seen[3].data["chunk"], "two\n",
            "the buffered remainder is flushed just ahead of the completion"
        );
        assert_eq!(seen[4].id, Some(2));
        assert_eq!(
            seen[5].id,
            Some(3),
            "ids stay dense across transient events"
        );

        let backlog = frontend
            .attach_stream(None, false)
            .expect("feed installed")
            .backlog;
        assert_eq!(
            backlog.len(),
            4,
            "the replay holds the durable events alone"
        );
        assert!(backlog.iter().all(|event| !event.event_type.is_transient()));
    }

    #[tokio::test]
    async fn emit_buffers_events_in_order() {
        let frontend = HttpFrontend::new();
        frontend
            .emit(FrontendEvent::TurnStarted {
                turn_id: uuid::Uuid::nil(),
            })
            .await;
        frontend
            .emit(FrontendEvent::AssistantTextDelta("hello".into()))
            .await;
        frontend.emit(FrontendEvent::TurnFinished).await;
        let recorder = frontend.drain();
        assert_eq!(recorder.len(), 3);
        assert!(matches!(recorder[0], FrontendEvent::TurnStarted { .. }));
        assert!(matches!(recorder[2], FrontendEvent::TurnFinished));
    }

    #[tokio::test]
    async fn request_permission_returns_deny_and_records_notice() {
        let frontend = HttpFrontend::new();
        let outcome = frontend
            .request_permission(PermissionRequest {
                tool_name: "shell_execute".into(),
                primary_param: Some("rm /tmp/x".into()),
                input: serde_json::Value::Null,
                cancellation: tokio_util::sync::CancellationToken::new(),
                subagent_id: None,
            })
            .await;
        assert_eq!(outcome, PermissionOutcome::Deny);
        let recorder = frontend.drain();
        let notice_count = recorder
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    FrontendEvent::Notice(notice)
                        if notice.text.contains("shell_execute")
                            && notice.text.contains("refused without asking")
                            && notice.text.contains("`stream: true`")
                )
            })
            .count();
        assert_eq!(
            notice_count, 1,
            "blocking-mode deny must surface exactly one diagnostic Notice"
        );
    }

    /// A streaming client that declared it cannot show prompts must be denied immediately rather
    /// than parked: the default path burns the full `APPROVAL_TIMEOUT` before denying anyway,
    /// which is indistinguishable from a hang. Bounded well under that timeout so a regression to
    /// the parking path fails here instead of passing slowly.
    #[tokio::test]
    async fn streaming_denies_immediately_when_prompts_are_unsupported() {
        let frontend = Arc::new(HttpFrontend::with_capabilities(SessionCapabilities {
            supports_permission_prompts: false,
            ..Default::default()
        }));
        let (_receiver, _ids) =
            frontend.install_stream(16, 16, Duration::from_secs(30), uuid::Uuid::nil());

        let outcome = tokio::time::timeout(
            Duration::from_secs(5),
            frontend.request_permission(PermissionRequest {
                tool_name: "shell_execute".into(),
                primary_param: Some("rm /tmp/x".into()),
                input: serde_json::Value::Null,
                cancellation: tokio_util::sync::CancellationToken::new(),
                subagent_id: None,
            }),
        )
        .await
        .expect("must not park on the SSE channel");

        assert_eq!(outcome, PermissionOutcome::Deny);
        assert!(
            frontend
                .pending
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_empty(),
            "nothing should have been parked"
        );
        let recorder = frontend.drain();
        assert!(
            recorder.iter().any(|event| matches!(
                event,
                FrontendEvent::Notice(notice)
                    if notice.text.contains("supports_permission_prompts")
            )),
            "the deny must explain itself in the response, not just in the timing"
        );
    }

    /// `meka session import` stores `capabilities_json` verbatim from a user-supplied archive, so
    /// a hand-written or third-party one can be missing a flag. An absent flag has to mean parking:
    /// silently refusing every gated call in an imported session is a worse failure than a
    /// stall the operator can see.
    #[test]
    fn capabilities_json_missing_a_flag_defaults_to_supporting_prompts() {
        let partial: SessionCapabilities =
            serde_json::from_str(r#"{"supports_reasoning_stream":true}"#).expect("deserialize");
        assert!(partial.supports_reasoning_stream);
        assert!(partial.supports_permission_prompts);
    }

    #[tokio::test]
    async fn sticky_allow_short_circuits_subsequent_requests() {
        let frontend = HttpFrontend::new();
        frontend.sticky.remember_allow("file_read");
        let outcome = frontend
            .request_permission(PermissionRequest {
                tool_name: "file_read".into(),
                primary_param: None,
                input: serde_json::Value::Null,
                cancellation: tokio_util::sync::CancellationToken::new(),
                subagent_id: None,
            })
            .await;
        assert_eq!(
            outcome,
            PermissionOutcome::Allow,
            "tools in always_allowed must not re-prompt"
        );
        let recorder = frontend.drain();
        // No diagnostic notice should be emitted: the sticky path bypasses both the streaming
        // SSE pause and the blocking-mode refusal.
        assert!(
            !recorder
                .iter()
                .any(|event| matches!(event, FrontendEvent::Notice(_))),
            "sticky allow must not emit a diagnostic Notice"
        );
    }

    #[tokio::test]
    async fn sticky_deny_short_circuits_subsequent_requests() {
        let frontend = HttpFrontend::new();
        frontend.sticky.remember_deny("shell_execute");
        let outcome = frontend
            .request_permission(PermissionRequest {
                tool_name: "shell_execute".into(),
                primary_param: None,
                input: serde_json::Value::Null,
                cancellation: tokio_util::sync::CancellationToken::new(),
                subagent_id: None,
            })
            .await;
        assert_eq!(outcome, PermissionOutcome::Deny);
    }

    #[tokio::test]
    async fn resolve_permission_allow_always_records_sticky() {
        let frontend = HttpFrontend::new();
        // Install a stream so request_permission parks instead of blocking-mode short-circuit.
        let (_receiver, _ids) =
            frontend.install_stream(16, 16, Duration::from_secs(30), uuid::Uuid::nil());

        let pending_handle = {
            let frontend_clone = Arc::new(frontend);
            let frontend_inner = Arc::clone(&frontend_clone);
            let request = PermissionRequest {
                tool_name: "file_write".into(),
                primary_param: Some("/tmp/x".into()),
                input: serde_json::Value::Null,
                cancellation: tokio_util::sync::CancellationToken::new(),
                subagent_id: None,
            };
            let join =
                tokio::spawn(async move { frontend_inner.request_permission(request).await });
            wait_for_pending(&frontend_clone, 1).await;

            // Resolve via the AllowAlways path.
            let pending_request_id = {
                let guard = frontend_clone
                    .pending
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                guard.keys().next().cloned().expect("pending entry exists")
            };
            assert!(
                frontend_clone
                    .resolve_permission(&pending_request_id, PermissionResolution::AllowAlways,)
            );
            let outcome = join.await.expect("join");
            assert_eq!(outcome, PermissionOutcome::Allow);
            assert!(
                frontend_clone.is_always_allowed("file_write"),
                "AllowAlways must record the tool as always allowed, and only that"
            );
            frontend_clone
        };
        drop(pending_handle);
    }

    #[tokio::test]
    async fn thinking_delta_is_filtered_when_capability_is_off() {
        let frontend = HttpFrontend::with_capabilities(SessionCapabilities::default());
        let (mut receiver, _ids) =
            frontend.install_stream(16, 16, Duration::from_secs(30), uuid::Uuid::nil());
        // The delta is what carries reasoning onto the wire, so it is what the capability has to
        // gate. Emitting only the block would leave this passing with the filter deleted, since
        // `sse::translate` drops the block whatever the capability says.
        frontend
            .emit(FrontendEvent::ThinkingDelta("musing".into()))
            .await;
        frontend
            .emit(FrontendEvent::ThinkingBlock {
                content: "musing".into(),
            })
            .await;
        frontend
            .emit(FrontendEvent::AssistantTextDelta("answer".into()))
            .await;
        // Drop the stream to close the broadcast and drain.
        frontend.end_turn();
        let mut events = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            events.push(event);
        }
        // The feed opens every turn with its `turn.started`; after it, only the assistant delta.
        assert_eq!(
            events.len(),
            2,
            "only the assistant delta should reach the SSE stream when reasoning is off"
        );
        assert_eq!(events[0].event_type, super::SseEventType::TurnStarted);
        assert_eq!(
            events[1].event_type,
            super::SseEventType::AssistantTextDelta
        );

        // The recorder still has all three (blocking-mode JSON path is unaffected).
        let recorder = frontend.drain();
        assert_eq!(recorder.len(), 3);
    }

    #[tokio::test]
    async fn thinking_delta_streams_when_capability_is_on() {
        let frontend = HttpFrontend::with_capabilities(SessionCapabilities {
            supports_reasoning_stream: true,
            ..Default::default()
        });
        let (mut receiver, _ids) =
            frontend.install_stream(16, 16, Duration::from_secs(30), uuid::Uuid::nil());
        frontend
            .emit(FrontendEvent::ThinkingDelta("musing".into()))
            .await;
        frontend.end_turn();
        let opening = receiver.try_recv().expect("the turn's opening is first");
        assert_eq!(opening.event_type, super::SseEventType::TurnStarted);
        let event = receiver.try_recv().expect("thinking event should stream");
        assert_eq!(event.event_type, super::SseEventType::ThinkingDelta);
    }

    /// A reply that loses the race must grant nothing.
    ///
    /// `resolve_permission` answers the HTTP caller `404 request-not-found` when the waiter has
    /// already gone (canceled, timed out, or disconnected). Recording the sticky decision before
    /// the send would still write the tool into `always_allowed` for the rest of the session: the
    /// caller told nothing was resolved, the tool call denied, and every later call to that tool
    /// silently approved with no prompt and no SSE event.
    #[tokio::test]
    async fn a_reply_that_arrives_too_late_grants_nothing() {
        let frontend = Arc::new(HttpFrontend::new());
        let (_receiver, _ids) = frontend.install_stream(16, 16, Duration::ZERO, uuid::Uuid::nil());

        let frontend_inner = Arc::clone(&frontend);
        let cancellation = tokio_util::sync::CancellationToken::new();
        let waiter = cancellation.clone();
        let join = tokio::spawn(async move {
            frontend_inner
                .request_permission(PermissionRequest {
                    tool_name: "shell_execute".into(),
                    primary_param: Some("rm -rf /".into()),
                    input: serde_json::Value::Null,
                    cancellation: waiter,
                    subagent_id: None,
                })
                .await
        });

        // Let it register, then drop the waiter *without* letting it tidy up.
        //
        // Canceling instead would not reach the race: `request_permission` removes its own entry
        // after the select, so a resolve arriving later finds nothing and returns at the
        // unknown-request arm, never reaching the record at all. Aborting drops the future
        // mid-await -- the receiver dies, the removal never runs -- which leaves exactly
        // the state the fix is about: an entry still in the map whose reader is gone, so
        // `send` fails.
        wait_for_pending(&frontend, 1).await;
        let request_id = frontend
            .pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .next()
            .cloned()
            .expect("the request must be registered before it is answered");
        drop(cancellation);
        join.abort();
        assert!(
            join.await.is_err(),
            "the waiter must be gone before the reply arrives"
        );

        // The user's "always allow" arrives after the waiter is gone.
        let delivered = frontend.resolve_permission(&request_id, PermissionResolution::AllowAlways);
        assert!(
            !delivered,
            "the caller has to be told the reply landed nowhere"
        );
        assert!(
            !frontend.is_always_allowed("shell_execute"),
            "and a reply nobody received must not grant the tool for the rest of the session"
        );
    }

    /// The prompt has to show what is being asked, not only which tool: `input` is what a client
    /// renders for the user to judge, and `expires_in_seconds` is the one figure every host with
    /// a client shares, so a client written against ACP's 30 minutes is not surprised here.
    #[tokio::test]
    async fn a_parked_permission_event_carries_the_arguments_and_the_shared_timeout() {
        let frontend = Arc::new(HttpFrontend::new());
        let (mut receiver, _ids) =
            frontend.install_stream(16, 16, Duration::ZERO, uuid::Uuid::nil());
        let cancellation = tokio_util::sync::CancellationToken::new();
        let join = tokio::spawn({
            let frontend = Arc::clone(&frontend);
            let cancellation = cancellation.clone();
            async move {
                frontend
                    .request_permission(PermissionRequest {
                        tool_name: "file_write".into(),
                        primary_param: Some("/tmp/x".into()),
                        input: serde_json::json!({"path": "/tmp/x", "content": "the payload"}),
                        cancellation,
                        subagent_id: None,
                    })
                    .await
            }
        });
        wait_for_pending(&frontend, 1).await;

        let opening = receiver.try_recv().expect("the turn's opening is first");
        assert_eq!(opening.event_type, SseEventType::TurnStarted);
        let event = receiver
            .try_recv()
            .expect("the pause event is on the stream");
        assert_eq!(event.event_type, SseEventType::PermissionRequired);
        assert_eq!(event.data["tool_name"], "file_write");
        assert_eq!(
            event.data["input"],
            serde_json::json!({"path": "/tmp/x", "content": "the payload"}),
            "the client must be shown every argument, not only the destination"
        );
        assert_eq!(
            event.data["expires_in_seconds"],
            crate::frontend::APPROVAL_TIMEOUT.as_secs()
        );
        assert_eq!(
            event.data["expires_in_seconds"], 1800,
            "thirty minutes, as on ACP; a shorter figure here is the drift this pins"
        );

        cancellation.cancel();
        assert_eq!(
            join.await.expect("task"),
            PermissionOutcome::Canceled,
            "the parked request follows the turn's stop"
        );
    }

    #[tokio::test]
    async fn request_permission_detects_sse_disconnect() {
        let frontend = Arc::new(HttpFrontend::new());
        // Install a stream so request_permission takes the streaming (park) path.
        let (receiver, _ids) = frontend.install_stream(16, 16, Duration::ZERO, uuid::Uuid::nil());

        let frontend_inner = Arc::clone(&frontend);
        let join = tokio::spawn(async move {
            frontend_inner
                .request_permission(PermissionRequest {
                    tool_name: "shell_execute".into(),
                    primary_param: Some("echo hi".into()),
                    input: serde_json::Value::Null,
                    cancellation: tokio_util::sync::CancellationToken::new(),
                    subagent_id: None,
                })
                .await
        });

        wait_for_pending(&frontend, 1).await;
        assert_eq!(
            frontend
                .pending
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .len(),
            1,
            "one pending permission request should be registered"
        );

        // Simulate SSE client disconnect by dropping the broadcast receiver.
        drop(receiver);
        assert!(
            frontend.client_disconnected(),
            "client_disconnected() should return true after receiver is dropped"
        );

        // The permission wait should resolve within a few poll intervals.
        let outcome = tokio::time::timeout(Duration::from_secs(5), join)
            .await
            .expect("should resolve well before the approval timeout")
            .expect("task should not panic");
        assert_eq!(
            outcome,
            PermissionOutcome::Canceled,
            "SSE disconnect must resolve the parked permission to Canceled"
        );

        // The pending map should be cleaned up.
        assert_eq!(
            frontend
                .pending
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .len(),
            0,
            "pending entry should be cleaned up after disconnect"
        );
    }

    /// A dropped consumer is not immediately a departed one. Re-attach exists because networks
    /// drop connections, and for the first instant those two look identical; reporting a disconnect
    /// straight away would cancel exactly the turns a client is about to rejoin.
    #[tokio::test]
    async fn client_disconnected_waits_out_the_reattach_grace() {
        let frontend = HttpFrontend::new();
        let (receiver, _ids) =
            frontend.install_stream(16, 16, Duration::from_millis(150), uuid::Uuid::nil());
        assert!(
            !frontend.client_disconnected(),
            "a live consumer is attached"
        );

        drop(receiver);
        assert!(
            !frontend.client_disconnected(),
            "the first zero-subscriber observation starts the grace period, it does not end it"
        );
        assert!(!frontend.client_disconnected(), "still inside the window");
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            frontend.client_disconnected(),
            "past the window with nobody attached, the client really is gone"
        );
    }

    /// Reconnecting inside the window clears the stamp, so the turn keeps running.
    #[tokio::test]
    async fn reattaching_cancels_a_pending_disconnect() {
        let frontend = HttpFrontend::new();
        let (receiver, _ids) =
            frontend.install_stream(16, 16, Duration::from_millis(100), uuid::Uuid::nil());
        drop(receiver);
        assert!(!frontend.client_disconnected(), "grace period starts");

        let attachment = frontend
            .attach_stream(None, false)
            .expect("a stream is installed");
        let _receiver = attachment.receiver;
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(
            !frontend.client_disconnected(),
            "a reconnect inside the window must clear the pending disconnect, not merely delay it"
        );
    }

    /// The ring is what `Last-Event-ID` replay reads. It must drop oldest-first at capacity and
    /// hand back only what the client has not already seen.
    #[tokio::test]
    async fn replay_ring_is_bounded_and_resumes_after_the_given_id() {
        let frontend = HttpFrontend::new();
        let (_receiver, _ids) =
            frontend.install_stream(64, 3, Duration::from_secs(30), uuid::Uuid::nil());
        for index in 0..5 {
            frontend
                .emit(FrontendEvent::AssistantTextDelta(format!("chunk{index}")))
                .await;
        }

        let all = frontend
            .attach_stream(None, false)
            .expect("stream installed");
        assert_eq!(
            all.backlog.len(),
            3,
            "the ring holds its capacity, not the history"
        );
        // Id 0 is the turn's own `turn.started`; the five chunks are 1 to 5.
        assert_eq!(
            all.backlog[0].id,
            Some(3),
            "oldest-first eviction keeps the newest three"
        );
        assert!(
            !all.gap,
            "a client naming no Last-Event-ID is joining, not resuming, so it has lost nothing"
        );

        let resumed = frontend
            .attach_stream(Some(4), false)
            .expect("stream installed");
        let ids: Vec<u64> = resumed
            .backlog
            .iter()
            .filter_map(|event| event.id)
            .collect();
        assert_eq!(ids, vec![5], "resume delivers strictly after the given id");
        assert!(
            !resumed.gap,
            "id 4 is still buffered, so the replay is contiguous"
        );

        let stale = frontend
            .attach_stream(Some(0), false)
            .expect("stream installed");
        assert!(
            stale.gap,
            "resuming from id 0 when the ring starts at 3 skips events 1 and 2, and must say so"
        );
    }

    /// A client that reconnects after the turn ended still has to learn how it ended.
    #[tokio::test]
    async fn attach_after_the_turn_ends_yields_the_terminal() {
        let frontend = HttpFrontend::new();
        let (_receiver, _ids) =
            frontend.install_stream(16, 16, Duration::from_secs(30), uuid::Uuid::nil());
        frontend
            .emit(FrontendEvent::AssistantTextDelta("hi".into()))
            .await;
        frontend.record_terminal(
            SseEventType::TurnFinished,
            serde_json::json!({"stop_reason": "end_turn"}),
        );
        frontend.end_turn();

        let attachment = frontend
            .attach_stream(None, false)
            .expect("stream retained past the turn");
        assert!(
            attachment.turn_id.is_none(),
            "the turn is over; the feed stays open for the next one"
        );
        let terminal = attachment.terminal.expect("terminal must be retained");
        assert_eq!(terminal.event_type, SseEventType::TurnFinished);
        assert!(
            attachment
                .backlog
                .iter()
                .any(|event| event.event_type.is_terminal()),
            "the terminal is in the ring too, so a replaying client receives it in order"
        );
    }

    /// A pause the client already answered must not come back on reconnect: it would put an
    /// approval prompt on screen for a request that no longer exists, and a decision sent for it
    /// returns 404. Additive events replay; stateful ones only while they are still true.
    #[tokio::test]
    async fn replay_drops_permission_prompts_that_are_no_longer_pending() {
        let frontend = Arc::new(HttpFrontend::new());
        let (_receiver, _ids) =
            frontend.install_stream(16, 16, Duration::from_secs(30), uuid::Uuid::nil());

        let asking = Arc::clone(&frontend);
        let join = tokio::spawn(async move {
            asking
                .request_permission(PermissionRequest {
                    tool_name: "shell_execute".into(),
                    primary_param: Some("echo hi".into()),
                    input: serde_json::Value::Null,
                    cancellation: tokio_util::sync::CancellationToken::new(),
                    subagent_id: None,
                })
                .await
        });
        wait_for_pending(&frontend, 1).await;

        // While parked, the prompt is real and must be replayed.
        let parked = frontend
            .attach_stream(None, false)
            .expect("stream installed");
        assert!(
            parked
                .backlog
                .iter()
                .any(|event| event.event_type == SseEventType::PermissionRequired),
            "a live prompt must reach a reconnecting client, or the turn stalls until timeout"
        );
        let request_id = parked
            .backlog
            .iter()
            .find(|event| event.event_type == SseEventType::PermissionRequired)
            .and_then(|event| event.data.get("request_id"))
            .and_then(serde_json::Value::as_str)
            .expect("prompt carries a request_id")
            .to_string();

        assert!(frontend.resolve_permission(&request_id, PermissionResolution::Allow));
        assert_eq!(join.await.expect("task"), PermissionOutcome::Allow);

        let after = frontend
            .attach_stream(None, false)
            .expect("stream installed");
        assert!(
            !after
                .backlog
                .iter()
                .any(|event| event.event_type == SseEventType::PermissionRequired),
            "an answered prompt must not be replayed"
        );
    }

    /// A finished-but-retained stream must not read as "streaming", or the next blocking turn
    /// would park its permission prompt on a channel nobody is listening to.
    #[tokio::test]
    async fn a_retained_stream_is_not_reported_as_streaming() {
        let frontend = HttpFrontend::new();
        let (_receiver, _ids) =
            frontend.install_stream(16, 16, Duration::from_secs(30), uuid::Uuid::nil());
        assert!(frontend.is_streaming());
        frontend.end_turn();
        assert!(
            !frontend.is_streaming(),
            "the ring outlives the turn; the streaming mode must not"
        );
    }

    #[tokio::test]
    async fn handle_elicitation_returns_decline_and_records_notice() {
        let frontend = HttpFrontend::new();
        let prompt = ElicitationPrompt {
            server_name: "github".into(),
            kind: ElicitationKind::Url {
                url: "https://example.com".into(),
            },
            message: "Open this URL?".into(),
        };
        let response = frontend.handle_elicitation(prompt).await;
        assert!(matches!(response, ElicitationResponse::Decline));
        let recorder = frontend.drain();
        assert!(
            recorder.iter().any(|event| matches!(
                event,
                FrontendEvent::Notice(notice) if notice.text.contains("github")
            )),
            "elicitation decline must surface a diagnostic Notice"
        );
    }

    /// A sub-agent's sub-agent gets a feed of its own, registered beside the others and naming
    /// the sub-agent that spawned it as its parent; its turn never opens on the feed above it.
    #[tokio::test]
    async fn a_grandchilds_feed_is_its_own_and_names_the_child_as_its_parent() {
        let children: ChildFeeds = Default::default();
        let root = Arc::new(HttpFrontend::new());
        root.install_feed(uuid::Uuid::from_u128(0x1), &FeedWiring {
            children: Some(Arc::clone(&children)),
            ..FeedWiring::bare(16, 16, Duration::from_secs(30))
        });
        let child = uuid::Uuid::from_u128(0x2);
        let grandchild = uuid::Uuid::from_u128(0x3);
        let root_as_frontend: Arc<dyn Frontend> = Arc::clone(&root) as Arc<dyn Frontend>;
        let child_frontend = root.for_subagent(root_as_frontend, child, Some("tu_1".into()));
        let grandchild_frontend = child_frontend.for_subagent(
            Arc::clone(&child_frontend),
            grandchild,
            Some("tu_2".into()),
        );
        let feed_of = |id: uuid::Uuid| crate::sync::lock(&children).get(&id).cloned();
        assert!(feed_of(child).is_some(), "the child's feed is registered");
        let grandchild_feed = feed_of(grandchild).expect("the grandchild's feed is registered");

        let turn_id = uuid::Uuid::from_u128(0x30);
        grandchild_frontend
            .emit(FrontendEvent::TurnStarted { turn_id })
            .await;
        let attachment = grandchild_feed
            .attach_stream(None, false)
            .expect("the grandchild's feed attaches");
        assert_eq!(attachment.turn_id, Some(turn_id));
        assert_eq!(
            attachment.turn_source,
            Some(TurnSource::Parent {
                parent_id: Some(child),
                tool_call_id: Some("tu_2".into()),
            }),
            "the grandchild's turn names the child as its parent"
        );
        let child_attachment = feed_of(child)
            .expect("still registered")
            .attach_stream(None, false)
            .expect("the child's feed attaches");
        assert_eq!(
            child_attachment.turn_id, None,
            "the grandchild's turn did not open on the child's feed"
        );

        drop(grandchild_frontend);
        assert!(
            feed_of(grandchild).is_none(),
            "the run over, the feed is gone"
        );
        assert!(feed_of(child).is_some(), "and the child's stays");
    }
}
