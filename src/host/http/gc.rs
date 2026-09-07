//! Background session GC. Periodically scans the in-memory session map and evicts entries
//! whose `last_turn_at` is older than the configured `idle_timeout`. Eviction drops the
//! `SessionEntry` (which in turn drops the `FileLock`, releasing the OS file lock) but
//! leaves the SQLite row in place by default; a later request with the same session ID can
//! re-attach (mirroring ACP's `session/load` semantics).
//!
//! Set `[serve].delete_on_idle = true` to also delete the DB row.

use std::time::Duration;

use crate::host::http::state::ServerState;

/// Spawn the GC scanner task. Returns the `JoinHandle` so the caller can cancel-on-drop or
/// wait for it during shutdown. The task loops forever; cancel by aborting the handle or by
/// the parent runtime shutting down.
pub(crate) fn spawn(state: ServerState) -> tokio::task::JoinHandle<()> {
    let scan_interval = state.config.gc_scan_interval;
    let idle_timeout = state.config.idle_timeout;
    let delete_on_idle = state.config.delete_on_idle;
    if idle_timeout.is_zero() {
        tracing::info!("session GC disabled ([serve] idle_timeout = 0)");
        return tokio::spawn(async {});
    }
    tracing::info!(
        "session GC enabled: idle_timeout={idle_timeout:?}, gc_scan_interval={scan_interval:?}, delete_on_idle={delete_on_idle}"
    );
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(scan_interval);
        // Skip the immediate first tick: give the server a moment to settle before we start
        // scanning.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            // Unguarded, one panicking scan ends eviction for the life of the process, silently:
            // the task dies, nothing joins it, and sessions accumulate until the operator notices
            // the memory. Log it and take the next tick instead.
            let scan =
                std::panic::AssertUnwindSafe(evict_idle(&state, idle_timeout, delete_on_idle));
            if let Err(panic) = futures::FutureExt::catch_unwind(scan).await {
                let panic = crate::error::panic_message(&*panic);
                tracing::warn!("session GC scan panicked ({panic}); continuing");
            }
        }
    })
}

async fn evict_idle(state: &ServerState, idle_timeout: Duration, delete_on_idle: bool) {
    let evicted = state.sessions.sweep_idle(idle_timeout).await;
    if evicted.is_empty() {
        return;
    }

    let evicted_ids: Vec<String> = evicted.iter().map(|(id, _)| id.to_string()).collect();
    tracing::info!(
        count = evicted.len(),
        session_ids = %evicted_ids.join(","),
        "session GC: evicted idle session(s)"
    );

    // Detach each evicted session's tool registry from the MCP manager so its
    // `tools/list_changed` callbacks stop targeting a registry that is about to drop. Mirrors
    // `handle_close_session` in `acp.rs`.
    //
    // Nothing here waits on the conversation mutex: anything blocking stalls the rest of the
    // batch, including the `FileLock`s of sessions already removed from the map but still held by
    // `evicted`, whose owners would get `session-locked` for the duration.
    for (_id, entry) in &evicted {
        entry.release(state.shared.mcp_manager.as_ref()).await;
    }

    if delete_on_idle {
        for (id, _entry) in &evicted {
            if let Err(error) = state.shared.store.delete_session(*id).await {
                tracing::warn!("session GC: failed to delete row for {id}: {error}");
            }
        }
    }
}
