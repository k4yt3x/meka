//! Schedule execution for `meka serve`.
//!
//! This is the durable host for [`crate::schedule`]. The server can revive any session on demand
//! (`reattach::ensure_session_loaded`), so it reaches every schedule in the database rather than
//! only those belonging to a conversation it happens to have open, which is the screen's limit.
//! What it leaves alone is a schedule on a session another process has locked; see [`is_ours`].
//!
//! There is no human on the far end of a scheduled turn. Two consequences run through everything
//! here: approval requests resolve to deny, because no client is attached to answer them, and
//! the turn's only durable output is the `messages` rows `Agent::run_turn` writes. A client sees
//! the result by reading the session back. When a push API lands, this is where it hooks in.

use std::sync::Arc;

use crate::{
    host::http::{reattach, state::ServerState},
    scheduler::SchedulerScope,
};

/// Start the scheduler for a running server. Returns the handle so `run_serve` can abort it during
/// shutdown, exactly as it does for the GC scanner.
pub(crate) fn spawn(state: ServerState) -> tokio::task::JoinHandle<()> {
    let config = state.shared.config.schedule.clone();
    if !config.enabled {
        tracing::info!("scheduler disabled ([schedule] enabled = false)");
        return tokio::spawn(async {});
    }
    tracing::info!(
        "scheduler enabled: poll_interval={poll_interval:?}, missed_grace={missed_grace:?}",
        poll_interval = config.poll_interval,
        missed_grace = config.missed_grace
    );
    let store = Arc::new(state.shared.store.clone());
    // The sessions this process holds or could load, as the inbox driver serves: a schedule on a
    // session another process holds is that process's to check, since its turns are the ones that
    // read the fire. Which host takes an occurrence among those that answer yes is a race between
    // their tickers, but only a race for *which*: `prepare` claims the occurrence with a
    // compare-and-swap (`ScheduleStore::claim_occurrence`), so the hosts that lose it return before
    // running the gate.
    let scope = {
        let sessions = state.sessions.clone();
        let store = state.shared.store.clone();
        SchedulerScope::Schedules(Arc::new(move |schedule: &crate::schedule::Schedule| {
            is_ours(residency_of(&sessions, schedule.session_id), || {
                lock_is_free(&store, schedule.session_id)
            })
        }))
    };
    crate::scheduler::spawn(
        store,
        config,
        state.shared.gate_tools.clone(),
        Arc::new(HttpHooks {
            state: state.clone(),
        }) as Arc<dyn crate::scheduler::ResidentPermissions>,
        scope,
        move |schedule, queued| {
            let state = state.clone();
            async move { announce_fire(&state, &schedule, queued).await }
        },
    )
}

/// Say that `schedule`'s fire is queued, as `schedule.fired` on the session's feed when the session
/// is loaded and to the webhooks, and wake the inbox driver, which opens a turn for it when the
/// session is idle. Said ahead of the wake, so a reader sees the fire before the turn it opens.
async fn announce_fire(
    state: &ServerState,
    schedule: &crate::schedule::Schedule,
    queued: crate::store::inbox::QueuedFire,
) {
    let mut data = serde_json::Map::from_iter([
        ("schedule_id".into(), serde_json::json!(schedule.id)),
        ("session_id".into(), serde_json::json!(schedule.session_id)),
        ("item_id".into(), serde_json::json!(queued.item_id)),
        ("occurrences".into(), serde_json::json!(queued.occurrences)),
    ]);
    if let Some(replaces) = queued.replaces {
        data.insert("replaces".into(), serde_json::json!(replaces));
    }
    let data = serde_json::Value::Object(data);
    if let Some(entry) = state.sessions.read().await.get(&schedule.session_id) {
        entry.frontend.push_sse(
            crate::host::http::sse::SseEventType::ScheduleFired,
            data.clone(),
        );
    }
    state.webhooks.send(
        crate::host::http::webhook::WebhookEvent::ScheduleFired,
        data,
    );
    state.inbox_wake.notify_one();
}

/// Start the inbox driver for a running server, over the sessions that are this process's
/// ([`is_ours`]): one it has loaded, which the driver waits for while a turn runs on it, or one
/// nobody holds, which it revives. Independent of the scheduler's switch: the inbox exists whenever
/// the server does.
pub(crate) fn spawn_inbox_driver(state: ServerState) -> tokio::task::JoinHandle<()> {
    let sessions = state.sessions.clone();
    let store = state.shared.store.clone();
    crate::host::scheduler::spawn_inbox_driver(
        Arc::new(HttpHooks {
            state: state.clone(),
        }),
        Arc::clone(&state.inbox_wake),
        state.shared.config.schedule.poll_interval,
        state.shutdown.clone(),
        move |session_id| {
            is_ours(residency_of(&sessions, session_id), || {
                lock_is_free(&store, session_id)
            })
        },
    )
}

/// Where `session_id` stands in this process's map of the sessions it has loaded. `try_read`
/// rather than `read`: the map is briefly write-locked while a session loads, and blocking a sweep
/// behind that is worse than skipping a tick.
fn residency_of(
    sessions: &crate::host::Sessions<uuid::Uuid, crate::host::http::state::SessionEntry>,
    session_id: uuid::Uuid,
) -> Residency {
    match sessions.try_read() {
        Ok(open) => match open.get(&session_id) {
            Some(entry) if entry.is_busy() => Residency::MidTurn,
            Some(_) => Residency::Resident,
            None => Residency::NotResident,
        },
        Err(_) => Residency::Unknown,
    }
}

/// Whether nobody holds the lock of `session_id`, asked by taking it and letting it go. Asked only
/// through [`is_ours`], of a session this process has not loaded.
fn lock_is_free(store: &crate::store::Store, session_id: uuid::Uuid) -> bool {
    match store.lock_session(session_id) {
        Ok(_lock) => true,
        Err(crate::error::MekaError::SessionLocked(_)) => false,
        // Not "someone else has it" but "we could not ask": a lock file owned by another user, an
        // unwritable or swept lock directory, file descriptors exhausted. Declining is still the
        // right answer -- a host that cannot take the lock cannot run the turn either -- but this
        // must be loud. The symptom is a *partial* outage: resident sessions keep running their
        // schedules and their inboxes while everything else silently stops, which looks like
        // nothing being scheduled rather than like a fault. A persistent cause repeating every
        // sweep is noisy on purpose: a bound on coverage nobody is told about reads as
        // "everything ran".
        Err(error) => {
            tracing::warn!(
                "failed to take the lock of session {session_id}: {error}; this process leaves it \
                 alone"
            );
            false
        }
    }
}

/// Whether a session is one this process currently has loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Residency {
    Resident,
    /// Loaded, with a turn in flight on it.
    MidTurn,
    NotResident,
    /// The session map could not be read without blocking, so we do not know.
    Unknown,
}

/// Whether a session is this process's to run turns on: one it has loaded, mid-turn or not, or
/// one nobody holds, which it can load. The one answer to "does another process hold this
/// session" for the scheduler and the inbox driver alike.
///
/// The order matters. A loaded session has to be answered first because *this* process is the
/// one holding its file lock, so probing would report our own sessions as held elsewhere.
/// Everything else is a lock probe: `lock_session` uses a non-blocking `try_write`, so taking the
/// lock and dropping it is a cheap, synchronous "is anyone else on this".
///
/// `lock_is_free` is deferred rather than passed by value because *not consulting it* is the
/// substance: it must not be asked unless the session is already known not to be ours, since a
/// resident session's lock is held by this very process and probing it would report every session
/// we serve as held elsewhere.
fn is_ours(residency: Residency, lock_is_free: impl FnOnce() -> bool) -> bool {
    match residency {
        Residency::Resident | Residency::MidTurn => true,
        Residency::NotResident => lock_is_free(),
        // Declined rather than probed. We may be the holder, and answering "not ours" for one
        // sweep costs a tick, where answering "free" would hand the session to a door that then
        // has to step back anyway.
        Residency::Unknown => false,
    }
}

/// Start the background-outcome poller. Separate task from the scheduler because the two are
/// independent switches: an installation can want timers without detached work, or the reverse.
///
/// Only sessions already resident are polled. Reviving an evicted session to deliver an outcome
/// would rebuild its whole runtime and pin it in memory, and the outcome is not going anywhere: it
/// keeps its `delivered_at IS NULL` until something opens that session again. The session-load
/// sweep is what guarantees the row exists to be found.
pub(crate) fn spawn_background_poller(state: ServerState) -> tokio::task::JoinHandle<()> {
    let config = state.shared.config.background.clone();
    if !config.enabled {
        return tokio::spawn(async {});
    }
    let poll_interval = state.shared.config.schedule.poll_interval;
    tracing::info!(
        "background tasks enabled: max_tasks={max_tasks}, poll_interval={poll_interval:?}",
        max_tasks = config.max_tasks
    );
    let sessions = state.sessions.clone();
    let shutdown = state.shutdown.clone();
    crate::host::scheduler::spawn_outcome_poller(
        HttpHooks { state },
        sessions,
        poll_interval,
        shutdown,
    )
}

/// `meka serve` around an out-of-band turn: a session another process holds is left to that
/// process, webhooks announce and may veto a delivery, the turn runs under shutdown's token, and
/// the frontend's events are drained as it closes because no client is attached to read them.
pub(super) struct HttpHooks {
    state: ServerState,
}

impl HttpHooks {
    /// The error an inbox event carries, in two copies. The feed's is the Problem Detail a
    /// response would get, under the same relay switch, so the two never disagree about what the
    /// upstream's text may travel in. The webhook's keeps the type, title, status and detail and
    /// never `provider_response`: a webhook payload carries no content, for the reason
    /// `notify_turn_end` gives. The failure is the scheduler's to log, and the turn's terminal's.
    fn event_errors(
        &self,
        error: &crate::error::MekaError,
    ) -> (serde_json::Value, serde_json::Value) {
        let full =
            serde_json::to_value(crate::host::http::errors::ProblemDetail::for_logged_error(
                error,
                self.state.config.relay_provider_errors,
            ))
            .unwrap_or(serde_json::Value::Null);
        let slim = serde_json::json!({
            "type": full.get("type"),
            "title": full.get("title"),
            "status": full.get("status"),
            "detail": full.get("detail"),
        });
        (full, slim)
    }
}

#[async_trait::async_trait]
impl crate::scheduler::ResidentPermissions for HttpHooks {
    async fn live_permission_of(
        &self,
        session_id: uuid::Uuid,
    ) -> Option<crate::permission::Permission> {
        self.state
            .sessions
            .read()
            .await
            .get(&session_id)
            .map(|entry| entry.agent.cells().permission.get())
    }
}

#[async_trait::async_trait]
impl crate::host::scheduler::HostHooks for HttpHooks {
    type Entry = crate::host::http::state::SessionEntry;

    async fn prepare(&self, entry: &Self::Entry) -> anyhow::Result<()> {
        crate::host::apply_recorded_bindings(&self.state.shared, &entry.agent, entry.id).await
    }

    fn refusal_sentence(&self, refusal: &crate::error::MekaError) -> String {
        crate::host::http::errors::api_wording(refusal)
    }

    async fn resident(&self, session_id: uuid::Uuid) -> anyhow::Result<Option<Self::Entry>> {
        match reattach::ensure_session_loaded(&self.state, session_id).await {
            Ok(entry) => Ok(Some(entry)),
            Err(problem) if problem.is(crate::host::http::errors::ErrorKind::SessionLocked) => {
                Ok(None)
            }
            // The detail, where the problem has one, is Meka's sentence naming what is wrong; the
            // title only names the problem's class.
            Err(problem) => Err(anyhow::anyhow!(
                "session unavailable: {}",
                problem.detail.as_deref().unwrap_or(&problem.title)
            )),
        }
    }

    async fn still_resident(&self, entry: &Self::Entry) -> bool {
        self.state
            .sessions
            .read()
            .await
            .get(&entry.id)
            .is_some_and(|held| std::sync::Arc::ptr_eq(&held.agent, &entry.agent))
    }

    /// The same count a client's turn takes, so `max_concurrent_turns` means every turn.
    fn admit(
        &self,
        entry: &Self::Entry,
    ) -> Result<crate::host::TurnGuard, crate::host::TurnRefused> {
        entry.admit_turn(Some((
            &self.state.concurrent_turns,
            self.state.config.max_concurrent_turns,
        )))
    }

    async fn announce(&self, tasks: &[crate::store::background::BackgroundTask]) -> bool {
        self.state
            .webhooks
            .announce_finished_tasks(&self.state.shared.store.background_store(), tasks)
            .await
            .may_deliver()
    }

    fn cancellation(&self) -> crate::agent::TurnCancellation {
        crate::agent::TurnCancellation::child_of(
            &self.state.shutdown,
            crate::store::turns::CancelReason::ServerShutdown,
        )
    }

    fn shutting_down(&self) -> bool {
        self.state.shutdown.is_cancelled()
    }

    /// Unattended: the turn runs for the session whether or not anybody reads the feed.
    fn begin_turn(
        &self,
        entry: &Self::Entry,
        turn_id: uuid::Uuid,
        source: crate::host::scheduler::TurnStarter,
    ) {
        let (_feed, _ids) = entry.frontend.begin_turn(
            turn_id,
            source,
            crate::host::http::feed::Opener::DRIVER,
            self.state.config.stream_reattach_grace,
            self.state.config.stream_replay_events,
        );
    }

    /// The same terminal a client's turn records, so a feed subscriber sees an out-of-band turn
    /// end the way it sees every other. The `turn.finished` / `turn.failed` webhook goes out for a
    /// turn the inbox opened, whatever its items are; an outcome turn is announced by the
    /// `task.finished` that preceded it, so the turn's own would be a second delivery for one
    /// event.
    async fn turn_closed(
        &self,
        entry: &Self::Entry,
        turn_id: uuid::Uuid,
        origin: &crate::host::scheduler::TurnStarter,
        outcome: &Result<crate::agent::TurnOutcome, crate::error::MekaError>,
        cancellation: &crate::agent::TurnCancellation,
    ) {
        use crate::host::http::handlers::turn::{
            message_withdrawn, notify_turn_end, position_for_terminal, served_by_for_terminal,
            terminal_event_parts, usage_from,
        };
        // Drained here rather than in `finished`, which runs after and finds it empty: the usage
        // and the withdrawal ride the terminal.
        let facts = entry.frontend.drain();
        let (event_type, data) = terminal_event_parts(
            Ok(outcome.as_ref()),
            cancellation
                .reason()
                .unwrap_or(crate::store::turns::CancelReason::Client),
            usage_from(&facts),
            turn_id,
            entry.id,
            self.state.config.relay_provider_errors,
            message_withdrawn(&facts),
            position_for_terminal(&self.state.shared.store, entry.id).await,
            served_by_for_terminal(&self.state.shared.store, entry.id, turn_id).await,
        );
        entry.frontend.record_terminal(event_type, data);
        entry.frontend.end_turn();
        if matches!(
            origin,
            crate::host::scheduler::TurnStarter::Inbox { .. }
                | crate::host::scheduler::TurnStarter::Schedule { .. }
        ) {
            notify_turn_end(&self.state.webhooks, event_type, turn_id, entry.id);
        }
    }

    async fn inbox_withdrawn(&self, entry: &Self::Entry, item_id: uuid::Uuid) {
        entry.frontend.push_sse(
            crate::host::http::sse::SseEventType::InboxWithdrawn,
            serde_json::json!({
                "item_id": item_id.to_string(),
                "session_id": entry.id.to_string(),
            }),
        );
    }

    async fn inbox_given_up(
        &self,
        entry: &Self::Entry,
        item: &crate::store::inbox::InboxItem,
        reason: &str,
        error: Option<&crate::error::MekaError>,
    ) {
        let mut data = serde_json::Map::from_iter([
            ("item_id".into(), serde_json::json!(item.id.to_string())),
            ("session_id".into(), serde_json::json!(entry.id.to_string())),
            ("reason".into(), serde_json::json!(reason)),
        ]);
        let mut delivery = data.clone();
        if let Some(error) = error {
            let (full, slim) = self.event_errors(error);
            data.insert("error".into(), full);
            delivery.insert("error".into(), slim);
        }
        entry.frontend.push_sse(
            crate::host::http::sse::SseEventType::InboxFailed,
            serde_json::Value::Object(data),
        );
        self.state.webhooks.send(
            crate::host::http::webhook::WebhookEvent::InboxFailed,
            serde_json::Value::Object(delivery),
        );
    }

    async fn inbox_deferred(
        &self,
        entry: &Self::Entry,
        items: &[crate::store::inbox::InboxItem],
        not_before: chrono::DateTime<chrono::Utc>,
        error: &crate::error::MekaError,
    ) {
        let (full, slim) = self.event_errors(error);
        let mut data = serde_json::Map::from_iter([
            (
                "item_ids".into(),
                serde_json::json!(
                    items
                        .iter()
                        .map(|item| item.id.to_string())
                        .collect::<Vec<_>>()
                ),
            ),
            ("session_id".into(), serde_json::json!(entry.id.to_string())),
            (
                "not_before".into(),
                serde_json::json!(not_before.to_rfc3339()),
            ),
        ]);
        let mut delivery = data.clone();
        data.insert("error".into(), full);
        delivery.insert("error".into(), slim);
        entry.frontend.push_sse(
            crate::host::http::sse::SseEventType::InboxDeferred,
            serde_json::Value::Object(data),
        );
        self.state.webhooks.send(
            crate::host::http::webhook::WebhookEvent::InboxDeferred,
            serde_json::Value::Object(delivery),
        );
    }

    fn inbox_retry_ceiling(&self) -> std::time::Duration {
        self.state.config.inbox_retry_ceiling
    }

    /// The error object the webhook gets, since the item is read back by anyone who lists the
    /// inbox: the upstream's text stays off it for the reason it stays off a webhook.
    fn inbox_error_record(&self, error: &crate::error::MekaError) -> Option<String> {
        Some(self.event_errors(error).1.to_string())
    }

    fn background_enabled(&self) -> bool {
        self.state.shared.config.background.enabled
    }

    fn store(&self) -> &crate::store::Store {
        &self.state.shared.store
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::{Residency, is_ours};

    /// The inbox driver's question takes a session this process holds as its own, mid-turn too,
    /// since the driver waits for the turn, and answers without probing the lock it holds itself.
    #[test]
    fn a_session_this_process_holds_is_its_own_mid_turn_or_not() {
        for residency in [Residency::Resident, Residency::MidTurn] {
            let probed = Cell::new(false);
            let ours = is_ours(residency, || {
                probed.set(true);
                false
            });
            assert!(ours, "{residency:?}");
            assert!(!probed.get(), "{residency:?} was probed");
        }
    }

    /// The case this exists for: a session an operator has open in a screen. Declining here is what
    /// keeps `prepare` from evaluating the schedule's gate for a session whose turns this process
    /// does not run.
    #[test]
    fn a_session_another_process_holds_is_declined() {
        assert!(!is_ours(Residency::NotResident, || false));
        assert!(
            is_ours(Residency::NotResident, || true),
            "a free lock is ours to take"
        );
    }

    /// An unreadable session map means we cannot rule out being the holder, so the probe would be
    /// unsound. Skipping the tick is free: the schedule keeps its occurrence and comes back.
    #[test]
    fn an_unreadable_session_map_declines_rather_than_probing() {
        let probed = Cell::new(false);
        let takeable = is_ours(Residency::Unknown, || {
            probed.set(true);
            true
        });
        assert!(!takeable);
        assert!(
            !probed.get(),
            "probing would be unsound when we may be the holder"
        );
    }
}
