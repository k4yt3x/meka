//! The session feed: one SSE channel for the life of a resident session, the replay ring behind
//! it, and what a client attaching to it is handed. `HttpFrontend` is its one producer; the
//! stream and feed handlers are its readers.

use std::{sync::Arc, time::Duration};

use tokio::sync::broadcast;

use super::sse::{EventIdGenerator, SseEvent, SseEventType};
use crate::host::scheduler::TurnSource;

/// How many events the feed's broadcast channel holds ahead of a slow consumer before it lags.
pub(crate) const FEED_BROADCAST_CAPACITY: usize = 256;

/// The replay ring behind a feed: the newest `capacity` numbered events, oldest first. It is what
/// a reader resuming from a `Last-Event-ID` is handed, and what a reader that fell behind the
/// broadcast is caught up from.
pub(super) struct Ring {
    events: std::collections::VecDeque<SseEvent>,
    capacity: usize,
}

impl Ring {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            events: std::collections::VecDeque::with_capacity(capacity.min(64)),
            capacity,
        }
    }

    /// Keep `event`, dropping the oldest once the ring is full. Nothing is kept when replay is
    /// switched off (`stream_replay_events = 0`).
    pub(super) fn record(&mut self, event: SseEvent) {
        if self.capacity == 0 {
            return;
        }
        while self.events.len() >= self.capacity {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    /// What a reader positioned after `last_event_id` is owed, `next_id` being the id the feed
    /// issues next.
    ///
    /// An id at or above `next_id` was never issued here, a fabricated value or one a browser
    /// carried over from another session, and is discarded rather than filtered against, which
    /// would silently deliver nothing. A reader naming no position is joining, not resuming, and
    /// has lost nothing. A hole is measured from the position the reader claims to the oldest
    /// event the ring holds, or to the present when it holds nothing.
    pub(super) fn replay(&self, last_event_id: Option<u64>, next_id: u64) -> Replay {
        let stale = last_event_id.is_some_and(|last| last >= next_id);
        let resume_from = if stale { None } else { last_event_id };
        let backlog = self
            .events
            .iter()
            .filter(|event| resume_from.is_none_or(|last| event.id.is_some_and(|id| id > last)))
            .cloned()
            .collect();
        let gap = if stale {
            Some(Gap { dropped: None })
        } else {
            resume_from.and_then(|last| {
                let oldest_known = self
                    .events
                    .front()
                    .and_then(|event| event.id)
                    .unwrap_or(next_id);
                let dropped = oldest_known.saturating_sub(last.saturating_add(1));
                (dropped > 0).then_some(Gap {
                    dropped: Some(dropped),
                })
            })
        };
        Replay {
            backlog,
            resume_from,
            gap,
        }
    }
}

/// What [`Ring::replay`] hands a reader.
pub(super) struct Replay {
    /// The ring's events after the reader's position, oldest first.
    pub(super) backlog: Vec<SseEvent>,
    /// The position resumption uses once a never-issued id is discarded; `None` is everything.
    pub(super) resume_from: Option<u64>,
    /// The hole between the reader's position and the backlog, when there is one.
    pub(super) gap: Option<Gap>,
}

/// A hole in what a reader is handed: numbered events between its position and what the feed can
/// still send, which only the records can fill. Said as a `feed.gap` event, which is the reader's
/// and not the session's, so it is never numbered or kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Gap {
    /// How many numbered events the hole holds, when the feed can count them; `None` when the
    /// reader's position was never issued here, so nothing says how far back it stood.
    pub(crate) dropped: Option<u64>,
}

impl Gap {
    /// The `feed.gap` event for this hole, naming its session on a session's feed.
    pub(crate) fn event(self, session_id: Option<uuid::Uuid>) -> SseEvent {
        let mut data = serde_json::Map::new();
        if let Some(session_id) = session_id {
            data.insert("session_id".into(), session_id.to_string().into());
        }
        if let Some(dropped) = self.dropped {
            data.insert("dropped".into(), dropped.into());
        }
        SseEvent {
            id: None,
            event_type: SseEventType::FeedGap,
            data: serde_json::Value::Object(data),
        }
    }
}

/// What a reader that fell behind the broadcast is handed to go on without a silent hole: the
/// ring's events after the last it delivered and a fresh subscription, taken under one lock so
/// nothing is emitted between them, plus the hole when the ring no longer reaches that far. The
/// same replay a reconnect gets, done by the server on the reader's behalf.
pub(crate) struct CaughtUp {
    pub(crate) backlog: Vec<SseEvent>,
    pub(crate) receiver: broadcast::Receiver<SseEvent>,
    pub(crate) gap: Option<Gap>,
}

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
    /// The newest events, for a reader resuming from a `Last-Event-ID` or catching up.
    pub(super) ring: Ring,
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
    /// The id of its `turn.started`, the position a stream opened on the turn stands at before
    /// it has received anything; `None` until the event is published.
    pub(super) started_id: Option<u64>,
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
            ring: Ring::new(replay_capacity),
            turn: None,
            attending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            unattended_since: Arc::new(std::sync::Mutex::new(None)),
            terminal: None,
            webhooks,
            server,
            reattach_grace,
        }
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
            self.ring.record(event.clone());
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
    /// The hole between the client's `Last-Event-ID` and the oldest event still buffered, when
    /// there is one. Reported rather than papered over: a transcript with a silent gap is worse
    /// than one the client knows is incomplete.
    pub(crate) gap: Option<Gap>,
    /// The last id issued before this attachment, after which its live subscription begins: the
    /// position a reader that falls behind is caught up from until it has delivered more.
    pub(crate) joined_after: Option<u64>,
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
    ring: Ring,
}

/// The server feed as every session feed and every handler shares it.
pub(crate) type SharedServerFeed = Arc<std::sync::Mutex<ServerFeed>>;

/// What a client attaching to the server feed gets: the backlog it missed and a live
/// subscription, taken together under one lock so nothing is emitted in the gap between them.
pub(crate) struct ServerAttachment {
    pub(crate) backlog: Vec<SseEvent>,
    pub(crate) receiver: broadcast::Receiver<SseEvent>,
    /// The hole between the client's `Last-Event-ID` and the oldest event the ring holds, when
    /// there is one.
    pub(crate) gap: Option<Gap>,
    /// The last id issued before this attachment; see [`StreamAttachment::joined_after`].
    pub(crate) joined_after: Option<u64>,
}

impl ServerFeed {
    /// A feed with a broadcast channel of `capacity` and a replay ring of `replay_capacity`.
    pub(crate) fn new(capacity: usize, replay_capacity: usize) -> Self {
        let (sender, _receiver) = broadcast::channel::<SseEvent>(capacity);
        Self {
            ids: EventIdGenerator::default(),
            sender,
            ring: Ring::new(replay_capacity),
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
        self.ring.record(event.clone());
        if self.sender.send(event).is_err() {
            tracing::trace!("no consumer is attached to the server feed; the event is recorded");
        }
    }

    /// The backlog after `last_event_id` and a live subscription, taken together so nothing is
    /// emitted between them. A reader that fell behind takes the same door with the last id it
    /// delivered, and is caught up the way a reconnect would be.
    pub(crate) fn attach(&self, last_event_id: Option<u64>) -> ServerAttachment {
        let next_id = self.ids.peek();
        let replay = self.ring.replay(last_event_id, next_id);
        ServerAttachment {
            backlog: replay.backlog,
            receiver: self.sender.subscribe(),
            gap: replay.gap,
            joined_after: next_id.checked_sub(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn server_feed_with(published: u64, ring: usize) -> ServerFeed {
        let mut feed = ServerFeed::new(16, ring);
        for index in 0..published {
            feed.publish(
                SseEventType::SessionUpdated,
                serde_json::json!({ "index": index }),
            );
        }
        feed
    }

    /// A reader's position decides what it is handed: everything after it while the ring reaches,
    /// the hole counted when it does not, nothing said to one that is joining rather than
    /// resuming, and a position this feed never issued discarded with its hole unmeasured.
    #[test]
    fn a_ring_replays_after_a_position_and_counts_the_hole_it_cannot_cover() {
        // Ids 0 through 9 were issued; the ring holds 6 through 9.
        let feed = server_feed_with(10, 4);
        let resumed = feed.attach(Some(2));
        let ids: Vec<u64> = resumed
            .backlog
            .iter()
            .filter_map(|event| event.id)
            .collect();
        assert_eq!(ids, vec![6, 7, 8, 9]);
        assert_eq!(
            resumed.gap.and_then(|gap| gap.dropped),
            Some(3),
            "ids 3, 4 and 5 are gone"
        );
        assert_eq!(
            resumed.joined_after,
            Some(9),
            "the live subscription begins after 9"
        );
        assert!(
            feed.attach(Some(6)).gap.is_none(),
            "a position the ring still holds is contiguous"
        );
        let joining = feed.attach(None);
        assert!(joining.gap.is_none(), "joining is not resuming");
        assert_eq!(joining.backlog.len(), 4);
        let never_issued = feed.attach(Some(42));
        assert_eq!(never_issued.gap, Some(Gap { dropped: None }));
        assert_eq!(
            never_issued.backlog.len(),
            4,
            "a discarded position is handed everything the ring holds"
        );
    }

    /// With replay switched off the ring holds nothing, so a resuming reader's hole runs to the
    /// present, and a reader already at the present has lost nothing.
    #[test]
    fn a_ring_of_nothing_counts_everything_since_the_position() {
        let feed = server_feed_with(5, 0);
        assert_eq!(
            feed.attach(Some(1)).gap.and_then(|gap| gap.dropped),
            Some(3),
            "ids 2, 3 and 4"
        );
        assert!(feed.attach(Some(4)).gap.is_none());
    }

    /// The event says the hole and whose feed it is on, and is nothing of the session's: no id.
    #[test]
    fn a_gap_event_names_its_session_and_counts_when_it_can() {
        let session = uuid::Uuid::from_u128(0x5);
        let counted = Gap { dropped: Some(3) }.event(Some(session));
        assert_eq!(counted.id, None);
        assert_eq!(counted.event_type, SseEventType::FeedGap);
        assert_eq!(
            counted.data,
            serde_json::json!({ "session_id": session.to_string(), "dropped": 3 })
        );
        assert_eq!(
            Gap { dropped: None }.event(None).data,
            serde_json::json!({}),
            "an unmeasured hole on the server feed says only that it is one"
        );
    }
}
