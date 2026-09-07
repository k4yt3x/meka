//! One resident session, whichever host drives it: the agent and its conversation behind the turn
//! mutex, admission of a turn or a compaction, the cancel cell a live turn publishes into, the
//! registry of sessions a host holds and its idle sweep, and the claiming of what a previous
//! process left undelivered.

use super::*;

/// How long a resident session may sit idle before the host releases it, when
/// `[serve].idle_timeout` is unset; ACP's idle sweep releases a session after the same silence.
pub(crate) const DEFAULT_IDLE_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(24 * 60 * 60);
/// How often the idle sweep runs, when `[serve].gc_scan_interval` is unset; ACP sweeps at the same
/// interval.
pub(crate) const DEFAULT_GC_SCAN_INTERVAL: std::time::Duration =
    std::time::Duration::from_secs(5 * 60);

/// A session this process holds open: the agent that drives it, the conversation behind the one
/// mutex a turn holds for its whole duration, and the bookkeeping a host needs to admit, cancel and
/// evict it. Each host wraps one in an entry of its own for what only it needs (HTTP's token and
/// wall-clock timestamps, ACP's title flag), and derefs to this for everything the hosts share.
///
/// The agent is not behind the mutex. Everything it exposes is `&self`, so a host reads the cells,
/// the registry or the binding while a turn runs; only the conversation is exclusive, and holding
/// its lock is what "a turn is in flight" means.
#[derive(Clone)]
pub(crate) struct ResidentSession {
    pub(crate) id: uuid::Uuid,
    pub(crate) agent: Arc<Agent>,
    pub(crate) conversation: Arc<tokio::sync::Mutex<crate::conversation::Conversation>>,
    pub(crate) cancel: CancelCell,
    pub(crate) last_activity: Arc<std::sync::RwLock<std::time::Instant>>,
    /// Turns admitted and not yet finished, counted by the host's turn guards.
    pub(crate) in_flight: Arc<std::sync::atomic::AtomicUsize>,
}
/// The idle decision on its facts alone, so it can be exercised without standing up an agent.
///
/// The busy checks come first and are not an optimization: a session mid-turn has a
/// `last_activity` from before the turn started, so on a long turn the timestamp alone says
/// "idle" while the agent is working. Evicting there would drop the entry out from under it.
fn idle_on(
    timeout: std::time::Duration,
    in_flight: usize,
    runtime_free: bool,
    background_running: usize,
    last_activity: std::time::Instant,
) -> bool {
    if timeout.is_zero() || in_flight > 0 || !runtime_free || background_running > 0 {
        return false;
    }
    last_activity.elapsed() >= timeout
}
/// What forking a session hands back to a host that holds the original's lock.
pub(crate) enum ForkHandoff {
    /// The copy exists and its lock is held. The caller assigns this over its current lock, which
    /// releases the original only once the new one is owned.
    Switched {
        id: uuid::Uuid,
        lock: crate::fs::FileLock,
    },
    /// The copy exists but its lock could not be taken, so the caller stays where it is. The id is
    /// carried so the user can still be told where the copy went.
    LockFailed {
        id: uuid::Uuid,
        error: crate::error::MekaError,
    },
    /// The session being forked no longer exists.
    SourceGone,
}
/// Fork `source` and take the copy's lock, in that order and without touching the caller's own.
///
/// The ordering is the point. Releasing the current lock first and then failing to acquire the new
/// one would leave the REPL running against an unlocked session that a second `meka` process could
/// open and interleave events into. Acquiring first means the failure path is simply "stay put",
/// and the caller drops its old lock only by overwriting it with the new one.
pub(crate) async fn fork_and_lock(
    store: &Store,
    source: uuid::Uuid,
) -> anyhow::Result<ForkHandoff> {
    // Locked before the copy's row exists, not after: a row committed ahead of its lock is one a
    // concurrent `session delete --all` enumerates and deletes, after which this function would
    // lock the vanished id successfully and hand the REPL a session whose next turn dies on a
    // foreign-key violation. See `Store::fork_session_locked`.
    // The REPL holds the source it is forking, so the source stands still by construction and a
    // probe would refuse this process its own session.
    let Some((forked, lock)) = store
        .fork_session_locked(
            source,
            crate::store::ForkOverrides::default(),
            crate::store::SourceLock::HeldByCaller,
        )
        .await?
    else {
        return Ok(ForkHandoff::SourceGone);
    };
    match lock {
        Ok(lock) => Ok(ForkHandoff::Switched {
            id: forked.id,
            lock,
        }),
        Err(error) => Ok(ForkHandoff::LockFailed {
            id: forked.id,
            error,
        }),
    }
}
/// Retire whatever the previous owner left running, for a process that has just taken this session.
///
/// Every path that hydrates a conversation has to call this, not just the CLI resume that first
/// needed it: `meka serve` reattaching an evicted session, and ACP's `session/load`,
/// `session/resume` and `session/fork`, all take the same lease and all inherit the same wreckage.
/// Missing one does not merely skip a report, it strands the row:
/// `list_undelivered_background_tasks` ignores `running`, so the outcome is never delivered, while
/// `list_running_background_tasks` keeps injecting the dead task into `[Background]` on every later
/// turn, telling the model not to restart work that died days ago.
///
/// Non-fatal: a session that cannot sweep must still open.
pub(crate) async fn claim_session(store: &Store, session_id: uuid::Uuid) {
    match store
        .background_store()
        .sweep_interrupted_background_tasks(session_id)
        .await
    {
        Ok(0) => {}
        Ok(swept) => tracing::info!(
            "{swept} background task(s) did not survive the last run; reporting them as interrupted"
        ),
        Err(error) => tracing::warn!("failed to retire interrupted background tasks: {error}"),
    }
}
/// Take this session's undelivered outcomes and stamp them, ready to ride on a turn.
///
/// Empty on a database error rather than propagating, because failing to *report* a finished task
/// must not also fail the turn the caller was about to run. Not delivering is better than
/// delivering forever: without the stamp the next drain would repeat these, and every drain after
/// it.
pub(crate) async fn claim_undelivered_outcomes(
    agent: &crate::agent::Agent,
    store: &Store,
    session_id: uuid::Uuid,
) -> Vec<crate::store::background::BackgroundTask> {
    if !scheduler::a_turn_can_carry_them(agent).await {
        return Vec::new();
    }
    claim_outcomes_now(store, session_id).await
}
/// [`claim_undelivered_outcomes`] without the readiness gate.
///
/// For a caller that is not about to run a turn -- the one-shot's post-turn report, which prints to
/// stderr on the way out -- and for tests that drive the claim concurrently with a store but no
/// agent. A caller that *is* about to run a turn must use the gated form, or a turn refused before
/// it touches the conversation leaves the batch stamped and unreadable.
pub(crate) async fn claim_outcomes_now(
    store: &Store,
    session_id: uuid::Uuid,
) -> Vec<crate::store::background::BackgroundTask> {
    let store = store.background_store();
    let ready = match store.list_undelivered_background_tasks(session_id).await {
        Ok(ready) => ready,
        Err(error) => {
            tracing::warn!("failed to load background task outcomes: {error}");
            return Vec::new();
        }
    };
    if ready.is_empty() {
        return ready;
    }
    let ids: Vec<String> = ready.iter().map(|task| task.id.clone()).collect();
    let claimed = match store.mark_background_tasks_delivered(&ids).await {
        Ok(claimed) => claimed,
        Err(error) => {
            tracing::warn!("failed to stamp background outcomes as delivered: {error}");
            return Vec::new();
        }
    };
    // Only what this caller won. Listing and stamping are two statements, and a poller can claim
    // the same row in the gap: whoever the `WHERE delivered_at IS NULL` refuses reports nothing,
    // rather than both of them reporting it.
    crate::background::only_what_was_won(ready, &claimed)
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Derived from the timeout rather than an evocative hour. `Instant` is measured from boot on
    /// both Linux and Windows, and subtracting more than the host's uptime panics, which is how
    /// this failed on a Windows box 55 minutes after a reboot. Any age past the timeout proves the
    /// same thing, so the test asks for the smallest one that does.
    fn aged(timeout: std::time::Duration) -> std::time::Instant {
        std::time::Instant::now()
            .checked_sub(timeout * 10)
            .expect("the host must have been up longer than a few seconds")
    }

    #[test]
    fn an_untouched_session_with_nothing_running_is_idle() {
        let timeout = std::time::Duration::from_secs(1);
        assert!(
            idle_on(timeout, 0, true, 0, aged(timeout)),
            "untouched for longer than the timeout and nothing running: evictable"
        );
        assert!(
            !idle_on(timeout, 0, false, 0, aged(timeout)),
            "a turn holds the runtime lock, so the session is in use"
        );
        assert!(
            !idle_on(timeout, 1, true, 0, aged(timeout)),
            "a turn admitted but not yet holding the runtime is in use too"
        );
        assert!(
            !idle_on(timeout, 0, true, 1, aged(timeout)),
            "a background task still running keeps its session"
        );
        assert!(
            !idle_on(std::time::Duration::ZERO, 0, true, 0, aged(timeout)),
            "a zero timeout never evicts"
        );
    }

    /// And a session used recently stays, so the sweep does not evict the one the user is on.
    #[test]
    fn a_recently_touched_session_is_not_idle() {
        assert!(!idle_on(
            std::time::Duration::from_secs(60 * 60),
            0,
            true,
            0,
            std::time::Instant::now()
        ));
    }

    #[test]
    fn a_cancel_during_admission_still_stops_the_turn() {
        let cell = CancelCell::default();
        let admission = cell.admit();
        assert!(!cell.cancel(), "nothing is live yet");
        let token = tokio_util::sync::CancellationToken::new();
        let _published = cell.publish(token.clone(), admission);
        assert!(
            token.is_cancelled(),
            "a cancel the caller was told succeeded left the turn running"
        );
    }

    #[test]
    fn a_turn_admitted_with_no_cancel_pending_is_left_alone() {
        let cell = CancelCell::default();
        let admission = cell.admit();
        let token = tokio_util::sync::CancellationToken::new();
        let _published = cell.publish(token.clone(), admission);
        assert!(!token.is_cancelled());
    }

    #[test]
    fn a_cancel_from_before_admission_does_not_abort_the_next_turn() {
        let cell = CancelCell::default();
        cell.cancel();
        let admission = cell.admit();
        let token = tokio_util::sync::CancellationToken::new();
        let _published = cell.publish(token.clone(), admission);
        assert!(
            !token.is_cancelled(),
            "a stale cancel aborted a turn submitted after it"
        );
    }

    #[test]
    fn a_cancel_reaches_the_live_turn_and_no_turn_after_it() {
        let cell = CancelCell::default();
        let token = tokio_util::sync::CancellationToken::new();
        let published = cell.publish(token.clone(), cell.admit());
        assert!(cell.cancel(), "the live turn is what a cancel stops");
        assert!(token.is_cancelled());
        drop(published);
        assert!(cell.live().is_none(), "the cell empties when the turn ends");
        assert!(
            !cell.cancel(),
            "and a cancel between turns has nothing to stop"
        );
    }
}
/// Why a turn was not admitted: the process already runs as many as its host allows.
#[derive(Debug)]
pub(crate) struct TurnRefused {
    pub(crate) cap: usize,
}
/// A turn's claim on its session, and on the process when the host caps concurrency. Dropped when
/// the turn ends, which is what lets the idle sweep and `turn_in_flight` see the truth.
#[must_use = "dropping the guard immediately defeats the in-flight tracking"]
pub(crate) struct TurnGuard {
    process: Option<Arc<std::sync::atomic::AtomicUsize>>,
    session: Arc<std::sync::atomic::AtomicUsize>,
    /// Sampled after the session started reporting the turn in flight; see [`CancelCell`].
    pub(crate) admission: Admission,
}
/// An out-of-band run's claim on its session: a scheduled fire, an outcome delivery, a compaction.
/// Counts as in flight for the idle sweep and for anything that must not overlap a turn, without
/// counting against the process cap.
#[must_use = "dropping the guard immediately defeats the in-flight tracking"]
pub(crate) struct BusyGuard {
    session: Arc<std::sync::atomic::AtomicUsize>,
    /// Sampled after the session started reporting the work in flight, like a turn's; a cancel
    /// that lands between the claim and the publish is then honored rather than lost. Sampling
    /// at publish time compared the epoch with itself.
    pub(crate) admission: Admission,
}
/// The sessions a host holds open, keyed the way that host addresses them, with the one idle-sweep
/// policy. Derefs to the map's lock so a handler reads and writes it directly.
pub(crate) struct Sessions<K, E>(Arc<tokio::sync::RwLock<std::collections::HashMap<K, E>>>);
/// Where a session's live turn publishes the token a cancel fires, and the counter that closes
/// the window between admitting a turn and publishing its token.
///
/// A cancel that lands after admission but before publication would otherwise fire the previous
/// turn's token: canceling something already finished, reporting success, and leaving the new turn
/// untouched. Every cancel bumps the epoch; a turn samples it when admitted and, on publishing,
/// cancels itself if the epoch moved. Between turns the cell is empty, so a cancel then fires
/// nothing and says so. The three hosts once solved the same race three ways: this epoch, a
/// generation counter armed by pending-prompt counts, and a process-wide relay.
#[derive(Clone, Default)]
pub(crate) struct CancelCell {
    token: Arc<std::sync::RwLock<Option<tokio_util::sync::CancellationToken>>>,
    epoch: Arc<std::sync::atomic::AtomicU64>,
}
/// The epoch a turn saw when it was admitted; [`CancelCell::publish`] compares it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Admission {
    epoch: u64,
}
/// Keeps the published token in its cell until the turn ends; dropping it empties the cell.
#[must_use = "dropping the guard empties the cell, so a cancel would find no turn to stop"]
pub(crate) struct Published {
    cell: CancelCell,
}
impl ResidentSession {
    /// Hold `session_lock` for as long as the session is resident, in the cells' slot, so another
    /// process cannot take the session and `/fork` has one place to replace it.
    pub(crate) fn new(
        id: uuid::Uuid,
        agent: Agent,
        conversation: crate::conversation::Conversation,
        cancel: CancelCell,
        session_lock: crate::fs::FileLock,
    ) -> Self {
        *crate::sync::lock(&agent.cells().session_lock) = Some(session_lock);
        Self {
            id,
            agent: Arc::new(agent),
            conversation: Arc::new(tokio::sync::Mutex::new(conversation)),
            cancel,
            last_activity: Arc::new(std::sync::RwLock::new(std::time::Instant::now())),
            in_flight: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// View handles a host already holds as a resident session. For the REPL, whose agent and
    /// conversation exist before the session row does, and whose lock is already in the cells'
    /// slot.
    pub(crate) fn from_parts(
        id: uuid::Uuid,
        agent: Arc<Agent>,
        conversation: Arc<tokio::sync::Mutex<crate::conversation::Conversation>>,
        cancel: CancelCell,
    ) -> Self {
        Self {
            id,
            agent,
            conversation,
            cancel,
            last_activity: Arc::new(std::sync::RwLock::new(std::time::Instant::now())),
            in_flight: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// The live cells, read without waiting on a turn.
    pub(crate) fn cells(&self) -> &SessionCells {
        self.agent.cells()
    }

    /// The registry the session dispatches through, for the MCP detach on the way out.
    pub(crate) fn tool_registry(&self) -> &ToolRegistry {
        self.agent.tool_registry()
    }

    /// Whether the profile this session runs on accepts image input, off the cell the agent
    /// publishes its binding into. Asked once the session's recorded profile has reached the agent
    /// (HTTP moves it on `PATCH`, ACP on the prompt after a switch), so the answer is for the
    /// profile the turn is about to run on rather than the one the session started on.
    pub(crate) fn accepts_images(&self) -> bool {
        self.cells().profile.current().vision
    }

    /// Let the session go: stop the turn in flight, stop the detached work it started, and stop
    /// the MCP manager fanning tool updates into a registry nobody reads. One routine for every
    /// way a session leaves a host, because each host once had its own and they disagreed on which
    /// of the three to do. Returns how many background tasks were signaled.
    ///
    /// Does not wait for the turn to unwind: a host that must (ACP's `session/close`, whose caller
    /// is off the dispatch loop) waits on [`Self::conversation`] first, and a host draining at
    /// shutdown must not.
    pub(crate) async fn release(
        &self,
        mcp_manager: Option<&Arc<crate::mcp::McpClientManager>>,
    ) -> usize {
        release_agent(&self.agent, &self.cancel, mcp_manager).await
    }

    /// Record activity, so the idle sweep leaves the session alone for another `timeout`.
    pub(crate) fn touch(&self) {
        let now = std::time::Instant::now();
        *crate::sync::write(&self.last_activity) = now;
    }

    /// Whether nothing has happened for `timeout` and nothing is happening: no turn admitted or
    /// holding the runtime, and no background task of the session still running. A zero timeout
    /// never evicts. The background count is the one asynchronous fact and is supplied, so the
    /// sweep can re-check a candidate under the map's write lock without awaiting there.
    pub(crate) fn is_idle_given(
        &self,
        timeout: std::time::Duration,
        background_running: usize,
    ) -> bool {
        let last = *crate::sync::read(&self.last_activity);
        idle_on(
            timeout,
            self.in_flight.load(std::sync::atomic::Ordering::Acquire),
            self.conversation.try_lock().is_ok(),
            background_running,
            last,
        )
    }
}
impl Drop for TurnGuard {
    fn drop(&mut self) {
        if let Some(process) = &self.process {
            process.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        }
        self.session
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}
impl Drop for BusyGuard {
    fn drop(&mut self) {
        self.session
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}
impl ResidentSession {
    /// Admit a turn: counted on the process first, against `cap` when the host has one, then on
    /// the session, then the cancel epoch is sampled. That order matters: from the moment the
    /// session reports a turn in flight, a cancel is aimed at this turn, so the epoch has to be
    /// read after that moment for [`CancelCell::publish`] to honor it.
    pub(crate) fn admit_turn(
        &self,
        process: Option<(&Arc<std::sync::atomic::AtomicUsize>, Option<usize>)>,
    ) -> Result<TurnGuard, TurnRefused> {
        let process = match process {
            Some((counter, cap)) => {
                let prior = counter.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                if let Some(cap) = cap
                    && prior >= cap
                {
                    counter.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                    return Err(TurnRefused { cap });
                }
                Some(Arc::clone(counter))
            }
            None => None,
        };
        self.in_flight
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Ok(TurnGuard {
            process,
            session: Arc::clone(&self.in_flight),
            admission: self.cancel.admit(),
        })
    }

    /// Count an out-of-band run as in flight.
    pub(crate) fn mark_busy(&self) -> BusyGuard {
        self.in_flight
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        BusyGuard {
            session: Arc::clone(&self.in_flight),
            admission: self.cancel.admit(),
        }
    }

    /// Claim the session for something that must not overlap a turn, or `None` while one is in
    /// flight.
    pub(crate) fn claim_idle(&self) -> Option<BusyGuard> {
        self.in_flight
            .compare_exchange(
                0,
                1,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .ok()
            .map(|_| BusyGuard {
                session: Arc::clone(&self.in_flight),
                admission: self.cancel.admit(),
            })
    }
}
impl<K, E> Clone for Sessions<K, E> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}
impl<K, E> std::ops::Deref for Sessions<K, E> {
    type Target = tokio::sync::RwLock<std::collections::HashMap<K, E>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<K, E> Sessions<K, E>
where
    K: Eq + std::hash::Hash + Clone,
    E: std::ops::Deref<Target = ResidentSession> + Clone,
{
    pub(crate) fn new() -> Self {
        Self(Arc::new(tokio::sync::RwLock::new(
            std::collections::HashMap::new(),
        )))
    }

    /// Remove and return every session idle for `timeout`. Candidates are chosen under the read
    /// lock and re-checked under the write lock, so a turn admitted in between keeps its session.
    pub(crate) async fn sweep_idle(&self, timeout: std::time::Duration) -> Vec<(K, E)> {
        // The background count is the one asynchronous fact, read under the read lock and carried
        // into the write pass, which then re-checks only what can be read without awaiting. The
        // map's lock is write-preferring, so an await held under the write guard queued every
        // handler behind the sweep.
        let candidates: Vec<(K, usize)> = {
            let sessions = self.0.read().await;
            let mut candidates = Vec::new();
            for (key, entry) in sessions.iter() {
                let background_running =
                    entry.cells().background_tasks.running_count(entry.id).await;
                if entry.is_idle_given(timeout, background_running) {
                    candidates.push((key.clone(), background_running));
                }
            }
            drop(sessions);
            candidates
        };
        if candidates.is_empty() {
            return Vec::new();
        }
        let mut evicted = Vec::with_capacity(candidates.len());
        let mut sessions = self.0.write().await;
        for (key, background_running) in candidates {
            let still_idle = match sessions.get(&key) {
                Some(entry) => entry.is_idle_given(timeout, background_running),
                None => false,
            };
            if still_idle && let Some(entry) = sessions.remove(&key) {
                evicted.push((key, entry));
            }
        }
        evicted
    }
}
impl Drop for Published {
    fn drop(&mut self) {
        self.cell.store(None);
    }
}
impl CancelCell {
    fn store(&self, token: Option<tokio_util::sync::CancellationToken>) {
        *crate::sync::write(&self.token) = token;
    }

    /// Sample the epoch for a turn being admitted, before its token exists.
    pub(crate) fn admit(&self) -> Admission {
        Admission {
            epoch: self.epoch.load(std::sync::atomic::Ordering::SeqCst),
        }
    }

    /// Publish a turn's token. A cancel that arrived since `admission` fires it at once, because
    /// the caller of that cancel was told the turn was canceled, so it is.
    pub(crate) fn publish(
        &self,
        token: tokio_util::sync::CancellationToken,
        admission: Admission,
    ) -> Published {
        self.store(Some(token.clone()));
        if self.epoch.load(std::sync::atomic::Ordering::SeqCst) != admission.epoch {
            tracing::debug!(
                "a cancel arrived while this turn was being admitted; honoring it before the turn runs"
            );
            token.cancel();
        }
        Published { cell: self.clone() }
    }

    /// Cancel the live turn, and record the cancel for a turn being admitted. Returns whether a
    /// turn was live to cancel.
    pub(crate) fn cancel(&self) -> bool {
        self.epoch.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        match self.live() {
            Some(token) => {
                token.cancel();
                true
            }
            None => false,
        }
    }

    /// The live turn's token, or `None` between turns. A poisoned lock still answers: a panicking
    /// turn has to stay cancelable.
    pub(crate) fn live(&self) -> Option<tokio_util::sync::CancellationToken> {
        crate::sync::read(&self.token).clone()
    }
}
