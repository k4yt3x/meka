//! The inbox driver: turns on sessions that are idle with items waiting.
//!
//! One sweeper for the process, woken by an enqueue and by every turn's end, and by a ticker for
//! what the wakes cannot carry: a deferred retry coming due, a session revived after a restart.
//! Each session with work gets a task of its own, so one long inbox turn does not hold every other
//! session's items, deduplicated so a wake mid-drain does not start a second driver on the same
//! session. The turns themselves run through [`crate::host::scheduler::run_inbox_turns`], the same
//! driver a scheduled fire and a background outcome go through.

use super::{schedule::HttpHooks, state::ServerState};

/// Start the inbox driver. Independent of the scheduler's switch: the inbox exists whenever the
/// server does.
pub(crate) fn spawn_inbox_driver(state: ServerState) -> tokio::task::JoinHandle<()> {
    let poll_interval = state.shared.config.schedule.poll_interval;
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(poll_interval);
        ticker.tick().await;
        loop {
            tokio::select! {
                _ = state.shutdown.cancelled() => return,
                _ = state.inbox_wake.notified() => {}
                _ = ticker.tick() => {}
            }
            let sessions = match state
                .shared
                .store
                .inbox_store()
                .sessions_with_work(chrono::Utc::now())
                .await
            {
                Ok(sessions) => sessions,
                Err(error) => {
                    tracing::warn!("failed to list sessions with inbox work: {error}");
                    continue;
                }
            };
            for session_id in sessions {
                if !crate::sync::lock(&state.inbox_draining).insert(session_id) {
                    continue;
                }
                let state = state.clone();
                tokio::spawn(async move {
                    let hooks = HttpHooks::new(state.clone());
                    // Supervised for the reason the scheduler's sweep is: this runs a whole agent
                    // turn, and a panic in the tool loop must not take the driver with it.
                    let drain = std::panic::AssertUnwindSafe(
                        crate::host::scheduler::run_inbox_turns(&hooks, session_id),
                    );
                    if let Err(panic) = futures::FutureExt::catch_unwind(drain).await {
                        tracing::warn!(
                            "inbox driver for session {session_id} panicked ({panic}); continuing",
                            panic = crate::error::panic_message(&*panic)
                        );
                    }
                    crate::sync::lock(&state.inbox_draining).remove(&session_id);
                    // A wake for an item enqueued during the drain found the id still in the set
                    // and was spent on nothing; one more look, and a wake of its own if there is
                    // work, rather than leaving it to the tick.
                    match state
                        .shared
                        .store
                        .inbox_store()
                        .pending_count(session_id)
                        .await
                    {
                        Ok(0) => {}
                        Ok(_) => state.inbox_wake.notify_one(),
                        Err(error) => {
                            tracing::warn!("failed to count inbox items for {session_id}: {error}");
                        }
                    }
                });
            }
        }
    })
}
