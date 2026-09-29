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
    /// The most recent turn's terminal, keyed by its id.
    pub(super) terminal: Option<(uuid::Uuid, SseEvent)>,
    /// Where an `inbox.delivered` also goes. The agent emits the delivery as a frontend event,
    /// and this is the one place that event is seen with the session's identity beside it.
    pub(super) webhooks: Option<super::webhook::WebhookDispatcher>,
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
            terminal: None,
            webhooks,
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
pub(crate) struct Attendance(pub(super) Arc<std::sync::atomic::AtomicUsize>);

impl Drop for Attendance {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}
