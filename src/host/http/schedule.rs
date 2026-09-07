//! Scheduled-job execution for `meka serve`.
//!
//! This is the durable host for [`crate::schedule`]. The server can revive any session on demand
//! (`reattach::ensure_session_loaded`), so it reaches every job in the database rather than only
//! those belonging to a conversation it happens to have open, which is the REPL's limit. What it
//! leaves alone is a job on a session another process has locked; see [`runnable_here`].
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
    let scope = SchedulerScope::Jobs(runnable_here(&state));
    let hooks = Arc::new(HttpHooks {
        state: state.clone(),
    });
    crate::scheduler::spawn(
        store,
        config,
        state.shared.gate_tools.clone(),
        Arc::clone(&hooks) as Arc<dyn crate::scheduler::ResidentPermissions>,
        scope,
        move |wakeup| {
            let hooks = Arc::clone(&hooks);
            async move { crate::host::scheduler::run_wakeup(&*hooks, wakeup).await }
        },
    )
}

/// "Could this process run a turn for that session right now?", asked once per due job per sweep.
///
/// The server can revive any session, so taking every job looks right and is not. `prepare`
/// evaluates a job's *gate* before the host is offered the wakeup, and a job whose session another
/// process holds comes straight back as a deferral -- which restores its original fire time,
/// already in the past, so it is due again on the very next tick. A gated hourly job on a session
/// an operator has open in a REPL would therefore run its shell command every `poll_interval` for
/// as long as that REPL stayed open. Declining here instead is precisely what [`SchedulerScope`]
/// documents its predicate variant for.
///
/// Every job fires in the session that owns it, so this is asked of every one of them and the
/// answer is always about that session. Which host takes a given occurrence, among those that
/// answer yes, is a race between their tickers, but only a race for *which*: `prepare` claims the
/// occurrence with a compare-and-swap (`ScheduleStore::claim_occurrence`), so the hosts that lose
/// it return before running the gate.
///
/// There are two ways to be runnable, and the order matters. A resident session is runnable by
/// definition, and has to be checked first because *this* process is the one holding its file lock
/// -- probing would report our own sessions as busy. Everything else is a lock probe:
/// `lock_session` already uses a non-blocking `try_write`, so taking the lock and dropping it is a
/// cheap, synchronous "is anyone else on this".
///
/// The window between this and `ensure_session_loaded` is left open deliberately. A lock taken
/// inside it still produces a deferral, which is correct and costs one gate evaluation; what this
/// removes is paying that on every tick, forever.
fn runnable_here(
    state: &ServerState,
) -> Arc<dyn Fn(&crate::schedule::ScheduledJob) -> bool + Send + Sync> {
    let sessions = state.sessions.clone();
    let store = state.shared.store.clone();
    Arc::new(move |job: &crate::schedule::ScheduledJob| {
        runnable(
            || {
                // `try_read` rather than `read`: the map is briefly write-locked while a session
                // loads, and blocking a sweep behind that is worse than skipping a tick.
                match sessions.try_read() {
                    Ok(open) if open.contains_key(&job.session_id) => Residency::Resident,
                    Ok(_) => Residency::NotResident,
                    Err(_) => Residency::Unknown,
                }
            },
            || match store.lock_session(job.session_id) {
                Ok(_lock) => true,
                Err(crate::error::MekaError::SessionLocked(_)) => false,
                // Not "someone else has it" but "we could not ask": a lock file owned by another
                // user, an unwritable or swept lock directory, file descriptors exhausted.
                // Declining is still the right answer -- a host that cannot take the lock cannot
                // run the turn either -- but this must be loud. The symptom is a *partial* outage:
                // jobs on resident sessions keep firing while everything else silently stops, which
                // looks like nothing being scheduled rather than like a fault. A persistent cause
                // repeating every sweep is noisy on purpose; it is the same reasoning as the
                // `held_over` line below, that a bound on coverage nobody is told about reads as
                // "everything ran".
                Err(error) => {
                    tracing::warn!(
                        "failed to take the session lock for job {job_id}: {error}; it will not \
                         fire here",
                        job_id = job.short_id()
                    );
                    false
                }
            },
        )
    })
}

/// Whether a session is one this process currently has loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Residency {
    Resident,
    NotResident,
    /// The session map could not be read without blocking, so we do not know.
    Unknown,
}

/// The rule [`runnable_here`] applies, separated from the state it reads so its short-circuits are
/// under test.
///
/// `lock_is_free` is deferred rather than passed by value because *not consulting it* is the
/// substance: it must not be asked unless the session is already known not to be ours, since a
/// resident session's lock is held by this very process and probing it would report every session
/// we serve as busy. `residency` is a closure only so the test can watch whether the other one was
/// reached.
fn runnable(residency: impl FnOnce() -> Residency, lock_is_free: impl FnOnce() -> bool) -> bool {
    match residency() {
        Residency::Resident => true,
        Residency::NotResident => lock_is_free(),
        // Declined rather than probed. We may be the holder, and answering "busy" for one sweep
        // costs a tick, where answering "free" would hand the job to a `run_wakeup` that then has
        // to defer it anyway.
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
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(poll_interval);
        ticker.tick().await;
        loop {
            tokio::select! {
                _ = state.shutdown.cancelled() => return,
                _ = ticker.tick() => {}
            }
            // Supervised for the same reason the scheduler is: this sweep runs a whole agent turn,
            // so anything in the tool loop can panic, and losing the task would stop every
            // background outcome from ever being delivered -- silently, since nothing joins this
            // handle. A task that finished would then sit `delivered_at`-stamped and unreported
            // forever, which is exactly the promise `background.rs` opens by making.
            let hooks = HttpHooks {
                state: state.clone(),
            };
            let sweep = std::panic::AssertUnwindSafe(
                crate::host::scheduler::deliver_ready_outcomes(&hooks, &state.sessions),
            );
            match futures::FutureExt::catch_unwind(sweep).await {
                Ok(std::ops::ControlFlow::Break(())) => return,
                Ok(std::ops::ControlFlow::Continue(())) => {}
                Err(panic) => tracing::warn!(
                    "background outcome sweep panicked ({panic}); continuing",
                    panic = crate::error::panic_message(&*panic)
                ),
            }
        }
    })
}

/// `meka serve` around an out-of-band turn: a session another process holds is deferred, webhooks
/// announce and may veto a delivery, the turn runs under shutdown's token, and the frontend's
/// events are drained afterwards because no client is attached to read them.
struct HttpHooks {
    state: ServerState,
}

/// The `status` a `schedule.fired` webhook carries for a turn that ran. A stop is neither of the
/// other two: the REPL annotates it as interrupted and reports no failure, and a webhook that said
/// `failed` sent an operator looking for a fault in a job someone had canceled.
pub(super) fn fired_status(outcome: &Result<(), crate::error::MekaError>) -> &'static str {
    match outcome {
        Ok(()) => "completed",
        Err(crate::error::MekaError::Interrupted) => "cancelled",
        Err(_) => "failed",
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

    async fn resident(&self, session_id: uuid::Uuid) -> anyhow::Result<Option<Self::Entry>> {
        match reattach::ensure_session_loaded(&self.state, session_id).await {
            Ok(entry) => Ok(Some(entry)),
            Err(problem) if problem.is(crate::host::http::errors::ErrorKind::SessionLocked) => {
                Ok(None)
            }
            Err(problem) => Err(anyhow::anyhow!("session unavailable: {}", problem.title)),
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

    async fn announce(&self, tasks: &[crate::store::background::BackgroundTask]) -> bool {
        self.state
            .webhooks
            .announce_finished_tasks(&self.state.shared.store.background_store(), tasks)
            .await
            .may_deliver()
    }

    fn cancellation(&self) -> tokio_util::sync::CancellationToken {
        self.state.shutdown.child_token()
    }

    fn shutting_down(&self) -> bool {
        self.state.shutdown.is_cancelled()
    }

    /// A notice rather than a user message, since this surface has no user-message event. It
    /// reaches a stream when one is live and the recorder otherwise, where `finished` drains it;
    /// what it buys today is that every host shows the prompt the same way, so a push channel
    /// added later inherits it.
    fn show_prompt(
        &self,
        entry: &Self::Entry,
        prompt: crate::host::scheduler::OutOfBandPrompt<'_>,
    ) {
        let text = match prompt {
            crate::host::scheduler::OutOfBandPrompt::Outcomes(text) => text.to_string(),
            crate::host::scheduler::OutOfBandPrompt::Scheduled(wakeup) => wakeup.render_prompt(),
        };
        entry
            .frontend
            .push_event(crate::frontend::FrontendEvent::Notice(
                crate::frontend::Notice::info(text),
            ));
    }

    async fn finished(
        &self,
        entry: &Self::Entry,
        job: Option<&crate::schedule::ScheduledJob>,
        outcome: &Result<(), crate::error::MekaError>,
    ) {
        let _scheduled_turn_events = entry.frontend.drain();
        if let Some(job) = job {
            self.notify(job, fired_status(outcome));
        }
    }

    fn failed_before_running(&self, job: &crate::schedule::ScheduledJob, _error: &anyhow::Error) {
        self.notify(job, "failed");
    }

    fn background_enabled(&self) -> bool {
        self.state.shared.config.background.enabled
    }

    fn store(&self) -> &crate::store::Store {
        &self.state.shared.store
    }
}

impl HttpHooks {
    fn notify(&self, job: &crate::schedule::ScheduledJob, status: &str) {
        self.state.webhooks.send(
            crate::host::http::webhook::WebhookEvent::ScheduleFired,
            serde_json::json!({
                "job_id": job.id,
                "session_id": job.session_id,
                "status": status,
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::{Residency, fired_status, runnable};

    /// The ordering that makes the probe safe. Every session `meka serve` has loaded is one whose
    /// file lock *this process* holds, so probing a resident session would report it busy and the
    /// server would stop running its own jobs.
    #[test]
    fn a_resident_session_is_never_probed() {
        let probed = Cell::new(false);
        let takeable = runnable(
            || Residency::Resident,
            || {
                probed.set(true);
                false
            },
        );
        assert!(takeable, "a session we already hold is ours to run");
        assert!(
            !probed.get(),
            "and asking the lock would have said otherwise"
        );
    }

    /// The case this exists for: a session an operator has open in a REPL. Declining here is what
    /// keeps `prepare` from evaluating the job's gate, which is the cost the deferral path could
    /// not avoid -- it runs the command first and finds out afterwards.
    #[test]
    fn a_session_another_process_holds_is_declined() {
        assert!(!runnable(|| Residency::NotResident, || false));
        assert!(
            runnable(|| Residency::NotResident, || true),
            "a free lock is ours to take"
        );
    }

    /// An unreadable session map means we cannot rule out being the holder, so the probe would be
    /// unsound. Skipping the tick is free: the job keeps its occurrence and comes back.
    #[test]
    fn an_unreadable_session_map_declines_rather_than_probing() {
        let probed = Cell::new(false);
        let takeable = runnable(
            || Residency::Unknown,
            || {
                probed.set(true);
                true
            },
        );
        assert!(!takeable);
        assert!(
            !probed.get(),
            "probing would be unsound when we may be the holder"
        );
    }

    /// A stopped fire is reported as canceled, the way the REPL reports it, not as a failure.
    #[test]
    fn a_stopped_fire_is_cancelled_not_failed() {
        assert_eq!(fired_status(&Ok(())), "completed");
        assert_eq!(
            fired_status(&Err(crate::error::MekaError::Interrupted)),
            "cancelled"
        );
        assert_eq!(
            fired_status(&Err(crate::error::MekaError::Provider("boom".to_string()))),
            "failed"
        );
    }
}
