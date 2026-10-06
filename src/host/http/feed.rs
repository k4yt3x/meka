//! The session feed: one SSE channel for the life of a resident session, the replay ring behind
//! it, and what a client attaching to it is handed. `HttpFrontend` is its one producer; the
//! stream and feed handlers are its readers.

use std::{sync::Arc, time::Duration};

use tokio::sync::broadcast;

use super::sse::{EventIdGenerator, SseEvent, SseEventType};
use crate::host::scheduler::TurnSource;

/// How many events the feed's broadcast channel holds ahead of a slow consumer before it lags.
pub(crate) const FEED_BROADCAST_CAPACITY: usize = 256;

/// The session's SSE event stream: one channel for the life of the resident session, and a replay
/// ring behind it.
///
/// The ring is what `Last-Event-ID` resumption is built on. Everything else on this stream is
/// additive, so a client that misses an event still holds a prefix of the truth and could limp
/// along; the *terminal* event is not, because a client that never receives one waits forever. So
/// the terminal is recorded here too, by the spawned turn task rather than by the response stream.
/// That distinction is load-bearing: in the case re-attach exists for, the client's connection has
/// already dropped and axum has discarded the response stream, so a terminal computed there would
/// be computed for nobody.
pub(crate) struct SessionFeed {
    pub(super) session_id: uuid::Uuid,
    pub(super) ids: Arc<EventIdGenerator>,
    pub(super) sender: broadcast::Sender<SseEvent>,
    /// Recent events, oldest first, capped at `replay_capacity`.
    pub(super) replay: std::collections::VecDeque<SseEvent>,
    pub(super) replay_capacity: usize,
    /// The turn publishing right now, if one is.
    pub(super) turn: Option<LiveTurn>,
    /// How many feed readers opened the stream with `attend=true`, each holding an
    /// [`Attendance`]. Shared with the guards so a reader that hangs up is counted out without
    /// taking the feed lock.
    pub(super) attending: Arc<std::sync::atomic::AtomicUsize>,
    /// When the last attending reader left, while none has come back; `None` while one attends
    /// or when none ever has. A parked prompt is given up on only once this has stood for the
    /// feed's reattach grace, so a reload or a suspended tab is not a departure.
    pub(super) unattended_since: Arc<std::sync::Mutex<Option<std::time::Instant>>>,
    /// The most recent turn's terminal, keyed by its id.
    pub(super) terminal: Option<(uuid::Uuid, SseEvent)>,
    /// Where an `inbox.delivered` also goes. The agent emits the delivery as a frontend event,
    /// and this is the one place that event is seen with the session's identity beside it.
    pub(super) webhooks: Option<super::webhook::WebhookDispatcher>,
    /// How long a reader that left is waited for before it counts as gone: the wiring's
    /// `stream_reattach_grace`, held on the feed so a prompt parked between turns, by a detached
    /// sub-agent say, is given the same window as one parked inside a turn.
    pub(super) reattach_grace: Duration,
    /// The server feed, where every event that changes this session's record goes as well.
    pub(super) server: Option<SharedServerFeed>,
}

/// What the feed knows about the turn in flight.
pub(super) struct LiveTurn {
    pub(super) turn_id: uuid::Uuid,
    /// Who started it, for the `turn.started` a client attaching mid-turn is given.
    pub(super) source: TurnSource,
    /// Whether a streaming client opened this turn, which is the only case where nobody reading
    /// means nobody waiting: a turn a driver started runs for the session, not for a connection.
    pub(super) attended: bool,
    pub(super) disconnected_since: Option<std::time::Instant>,
    pub(super) reattach_grace: Duration,
}

impl SessionFeed {
    pub(super) fn new(
        session_id: uuid::Uuid,
        ids: Arc<EventIdGenerator>,
        capacity: usize,
        replay_capacity: usize,
        webhooks: Option<super::webhook::WebhookDispatcher>,
        server: Option<SharedServerFeed>,
        reattach_grace: Duration,
    ) -> Self {
        let (sender, _receiver) = broadcast::channel::<SseEvent>(capacity);
        Self {
            session_id,
            ids,
            sender,
            replay: std::collections::VecDeque::with_capacity(replay_capacity.min(64)),
            replay_capacity,
            turn: None,
            attending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            unattended_since: Arc::new(std::sync::Mutex::new(None)),
            terminal: None,
            webhooks,
            server,
            reattach_grace,
        }
    }

    fn record(&mut self, event: SseEvent) {
        if self.replay_capacity == 0 {
            return;
        }
        while self.replay.len() >= self.replay_capacity {
            self.replay.pop_front();
        }
        self.replay.push_back(event);
    }

    /// Number, record and broadcast one event, under the lock the caller holds so ids stay
    /// monotonic across concurrent emitters.
    pub(super) fn publish(
        &mut self,
        event_type: SseEventType,
        mut data: serde_json::Value,
    ) -> SseEvent {
        // Every event names its turn and session, so a feed subscriber can file it without
        // per-connection state; the terminals always did, the rest join them.
        if let Some(object) = data.as_object_mut() {
            if let Some(turn) = &self.turn {
                object
                    .entry("turn_id")
                    .or_insert_with(|| serde_json::Value::String(turn.turn_id.to_string()));
            }
            object
                .entry("session_id")
                .or_insert_with(|| serde_json::Value::String(self.session_id.to_string()));
        }
        let transient = event_type.is_transient();
        let event = SseEvent {
            // Progress, not history: no id, so no `Last-Event-ID` ever names it, and no place in
            // the ring, so a chatty command cannot push out the events a reconnecting client needs.
            id: (!transient).then(|| self.ids.next()),
            event_type,
            data,
        };
        if !transient {
            self.record(event.clone());
        }
        if self.sender.send(event.clone()).is_err() {
            tracing::trace!("no consumer is attached; the event is recorded for a re-attach");
        }
        if event_type == SseEventType::InboxDelivered
            && let Some(webhooks) = &self.webhooks
        {
            webhooks.send(
                super::webhook::WebhookEvent::InboxDelivered,
                event.data.clone(),
            );
        }
        // The server feed never takes a session feed's lock, so taking it here under one cannot
        // invert.
        if ServerFeed::mirrors(event_type)
            && let Some(server) = &self.server
        {
            crate::sync::lock(server).publish(event_type, event.data.clone());
        }
        event
    }
}

/// What a re-attaching client gets: the backlog it missed, plus a live subscription, taken
/// together under one lock so nothing can be emitted in the gap between them.
pub(crate) struct StreamAttachment {
    /// The turn in flight when the client attached, if one was.
    pub(crate) turn_id: Option<uuid::Uuid>,
    /// Who started that turn, so the synthesized `turn.started` can say.
    pub(crate) turn_source: Option<TurnSource>,
    /// Held while the client attends the session; dropping the attachment counts it out.
    pub(crate) attendance: Option<Attendance>,
    /// Buffered events with an id greater than the client's `Last-Event-ID`, oldest first.
    pub(crate) backlog: Vec<SseEvent>,
    /// The live subscription. Always present: the feed outlives every turn.
    pub(crate) receiver: broadcast::Receiver<SseEvent>,
    /// The most recent turn's terminal, when the client attached with no turn in flight. A client
    /// that reconnects after the fact gets it immediately rather than waiting on a stream that
    /// will never produce another event for that turn.
    pub(crate) terminal: Option<SseEvent>,
    /// True when the client's `Last-Event-ID` is older than the oldest event still buffered, so
    /// the replay has a hole in it. Reported rather than papered over: a transcript with a silent
    /// gap is worse than one the client knows is incomplete.
    pub(crate) gap: bool,
    /// The position resumption should actually use, after discarding a `Last-Event-ID` that this
    /// session never issued.
    ///
    /// `None` means "send everything you have".
    ///
    /// Ids run monotonically across the whole session, so an id from an *earlier* turn sorts below
    /// this turn's backlog and filters nothing -- that is the case this field is designed to let
    /// through. What it discards is an id at or above the high-water mark: one fabricated, or
    /// carried over from a different session by a browser `EventSource` that re-sends its stored
    /// id automatically. Honoring such an id would filter the entire backlog, and the terminal
    /// with it, as already-delivered, leaving the client waiting on a turn it can never see end.
    pub(crate) resume_from: Option<u64>,
}

/// A feed reader's declaration that it shows approval prompts and answers them, alive for as long
/// as its stream is. While one exists, a gated call on any turn parks as `permission_required`
/// rather than being refused without asking; when the last one drops, a parked prompt is canceled
/// the way a streaming client's disconnect cancels it.
pub(crate) struct Attendance {
    pub(super) attending: Arc<std::sync::atomic::AtomicUsize>,
    pub(super) unattended_since: Arc<std::sync::Mutex<Option<std::time::Instant>>>,
}

impl Drop for Attendance {
    fn drop(&mut self) {
        // The last one out starts the clock a parked prompt waits on.
        if self
            .attending
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst)
            == 1
        {
            *crate::sync::lock(&self.unattended_since) = Some(std::time::Instant::now());
        }
    }
}

/// The listing's change feed: every event that changes a session record, across every session
/// this process holds, so a client keeping a list of sessions needs one connection to keep it
/// current and opens a session's own feed only for the one it is looking at. Carries exactly what
/// a record is rendered from: a session's creation, change and deletion, the four turn lifecycle
/// events without their deltas, and the parking and closing of a prompt, which move
/// `approvals_pending`. Ids of its own, and a ring of its own for `Last-Event-ID`.
pub(crate) struct ServerFeed {
    ids: EventIdGenerator,
    sender: broadcast::Sender<SseEvent>,
    replay: std::collections::VecDeque<SseEvent>,
    replay_capacity: usize,
}

/// The server feed as every session feed and every handler shares it.
pub(crate) type SharedServerFeed = Arc<std::sync::Mutex<ServerFeed>>;

/// What a client attaching to the server feed gets: the backlog it missed and a live
/// subscription, taken together under one lock so nothing is emitted in the gap between them.
pub(crate) struct ServerAttachment {
    pub(crate) backlog: Vec<SseEvent>,
    pub(crate) receiver: broadcast::Receiver<SseEvent>,
    /// Whether the client's `Last-Event-ID` is older than the ring reaches.
    pub(crate) gap: bool,
}

impl ServerFeed {
    /// A feed with a broadcast channel of `capacity` and a replay ring of `replay_capacity`.
    pub(crate) fn new(capacity: usize, replay_capacity: usize) -> Self {
        let (sender, _receiver) = broadcast::channel::<SseEvent>(capacity);
        Self {
            ids: EventIdGenerator::default(),
            sender,
            replay: std::collections::VecDeque::with_capacity(replay_capacity.min(64)),
            replay_capacity,
        }
    }

    /// Which of a session feed's events the server feed carries too: the ones that change the
    /// session's record.
    pub(crate) const fn mirrors(event_type: SseEventType) -> bool {
        matches!(
            event_type,
            SseEventType::TurnStarted
                | SseEventType::TurnFinished
                | SseEventType::TurnFailed
                | SseEventType::TurnCanceled
                | SseEventType::PermissionRequired
                | SseEventType::PermissionResolved
        )
    }

    /// Number, record and broadcast one event. `data` already names its session.
    pub(crate) fn publish(&mut self, event_type: SseEventType, data: serde_json::Value) {
        let event = SseEvent {
            id: Some(self.ids.next()),
            event_type,
            data,
        };
        if self.replay_capacity > 0 {
            while self.replay.len() >= self.replay_capacity {
                self.replay.pop_front();
            }
            self.replay.push_back(event.clone());
        }
        if self.sender.send(event).is_err() {
            tracing::trace!("no consumer is attached to the server feed; the event is recorded");
        }
    }

    /// The backlog after `last_event_id` and a live subscription. An id at or above the high-water
    /// mark was never issued here and is discarded rather than filtered against.
    pub(crate) fn attach(&self, last_event_id: Option<u64>) -> ServerAttachment {
        let stale = last_event_id.is_some_and(|last| last >= self.ids.peek());
        let resume_from = if stale { None } else { last_event_id };
        let backlog: Vec<SseEvent> = self
            .replay
            .iter()
            .filter(|event| resume_from.is_none_or(|last| event.id.is_some_and(|id| id > last)))
            .cloned()
            .collect();
        let gap = stale
            || match (resume_from, self.replay.front()) {
                (Some(last), Some(oldest)) => {
                    oldest.id.is_some_and(|id| id > last.saturating_add(1))
                }
                (Some(_), None) => self.replay_capacity == 0,
                _ => false,
            };
        ServerAttachment {
            backlog,
            receiver: self.sender.subscribe(),
            gap,
        }
    }
}
