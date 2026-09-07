//! The scheduler: the sweep that finds due jobs, claims each occurrence against every other
//! process on the same store, evaluates its gate and hands a [`Wakeup`] to the host that fires it.
//! What a job *is* lives in [`crate::schedule`]; this is what *runs* one, and it is the one place
//! that knows the store's claim protocol.

use std::time::Duration;

use chrono::{DateTime, Utc};
// Reached through `humantime_serde`'s re-export rather than a direct dependency, which is also
// how `crate::config` gets at it. One duration syntax, one copy of the parser.
use humantime_serde::re::humantime;

use crate::{
    schedule::*,
    store::schedule::{ClaimClosed, ScheduleStore},
};

/// A due job, with everything the turn needs to know about why it is running now.
pub(crate) struct Wakeup {
    pub(crate) job: ScheduledJob,
    /// The gate's stdout, when the job has one. Carried so the model does not re-run the check the
    /// gate just ran.
    pub(crate) gate_output: Option<String>,
    /// How far past its due time this fire is. Near zero in normal operation; large after
    /// downtime.
    pub(crate) late_by: chrono::Duration,
    /// Occurrences this fire stands in for, beyond the one being delivered. Non-zero only after
    /// downtime, since the scheduler collapses a backlog into a single turn.
    pub(crate) coalesced: u32,
}
impl Wakeup {
    /// Render the user-turn text delivered to the model.
    ///
    /// The header is not decoration. Without it the model reads a bare instruction as if a human
    /// had just typed it and answers conversationally -- into an empty terminal at 03:00, to
    /// nobody.
    pub(crate) fn render_prompt(&self) -> String {
        let mut rendered = format!(
            "[Scheduled job {} fired {}]",
            self.job.short_id(),
            crate::text::format_timestamp(Utc::now(), crate::text::Precision::Minutes)
        );
        // Only mention lateness when it is material. A tick's worth of delay is normal and saying
        // so every time would train the model to ignore the line that matters after an outage.
        if self.late_by > chrono::Duration::minutes(1) {
            rendered.push_str(&format!(
                "\n[Late by {}; this fire replaces {} missed occurrence(s)]",
                format_late(self.late_by),
                self.coalesced + 1
            ));
        }
        rendered.push_str("\n\n");
        rendered.push_str(&self.job.prompt);
        if let Some(output) = &self.gate_output {
            rendered.push_str("\n\n[Gate output]\n");
            rendered.push_str(output);
        }
        rendered
    }
}
/// Human-readable lateness, for the header above.
pub(crate) fn format_late(late_by: chrono::Duration) -> String {
    late_by
        .to_std()
        .map(|std| humantime::format_duration(Duration::from_secs(std.as_secs())).to_string())
        .unwrap_or_else(|_| "an unknown interval".to_string())
}
/// What a host did with a job handed to it.
///
/// Exists because deciding to fire and being *able* to fire are separate questions, and the gap
/// between them is where an occurrence can be lost. `prepare` leases the occurrence before handing
/// the wakeup over, so the row is untouched while the host works; a host that turns out to be
/// unable to run the job says so, and the lease is released rather than completed.
///
/// Crash protection is [`MAX_CLAIM_ATTEMPTS`]'s job, not this variant's, so this means only what it
/// says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FireOutcome {
    /// The turn ran, or failed in a way that re-running would not fix.
    Ran,
    /// This host could not take the job and another one should. The concrete case is `meka serve`
    /// finding the session's file lock held by a REPL: that REPL has its own watcher and will run
    /// the job itself, so the occurrence is restored rather than burnt.
    Deferred,
    /// This host owns the job and could not run it, and trying again immediately would not help.
    ///
    /// The claim is left to expire rather than released or completed, which is the treatment a
    /// panicking turn already gets and for the same reason: releasing puts `next_fire_at` back in
    /// the past, so the occurrence comes due on the very next sweep and a gated job re-runs its
    /// probe every `poll_interval`; completing spends the occurrence, which for a one-shot deletes
    /// the row and loses the job outright. Waiting out `claim_lease` retries at a cadence a blip
    /// survives and a persistent fault does not, and `MAX_CLAIM_ATTEMPTS` still parks it in the
    /// end.
    ///
    /// The concrete case is a session whose recorded profile no longer resolves: running
    /// the turn would bill an account the row does not name, and the cause may be a configuration
    /// error the user has yet to fix or a transient `SQLITE_BUSY` from another meka process.
    Unrunnable,
}
/// What this process knows about a session's level that its row may not.
///
/// A resident session's level lives in its cell; every surface that moves it writes the row back,
/// but that write can fail, and until it succeeds the row says what the session used to be. The
/// poller running in the same process as the session must not fire a gate on the row's word when
/// the cell it could have read says otherwise. A session that is not resident here has no cell to
/// read, and the row answers for it.
#[async_trait::async_trait]
pub(crate) trait ResidentPermissions: Send + Sync {
    /// The live level of `session_id` when this process holds it, `None` otherwise.
    async fn live_permission_of(
        &self,
        session_id: uuid::Uuid,
    ) -> Option<crate::permission::Permission>;
}

/// A process holding no session, so every level comes from the row. Only the tests want this: a
/// real host always answers for the sessions it holds.
#[cfg(test)]
pub(crate) struct NoResidents;

#[cfg(test)]
#[async_trait::async_trait]
impl ResidentPermissions for NoResidents {
    async fn live_permission_of(
        &self,
        _session_id: uuid::Uuid,
    ) -> Option<crate::permission::Permission> {
        None
    }
}

/// A process holding every session at one level, for a test that pits the cell against the row.
///
/// Only the Unix-only gate tests in `schedule` build one, so it carries their cfg; under a bare
/// `cfg(test)` a Windows build reports it dead.
#[cfg(all(test, unix))]
pub(crate) struct ResidentAt(pub(crate) crate::permission::Permission);

#[cfg(all(test, unix))]
#[async_trait::async_trait]
impl ResidentPermissions for ResidentAt {
    async fn live_permission_of(
        &self,
        _session_id: uuid::Uuid,
    ) -> Option<crate::permission::Permission> {
        Some(self.0)
    }
}

/// Which jobs a scheduler instance is responsible for.
///
/// Every host answers a predicate rather than claiming a static set, because none of them can
/// actually take every job: `meka serve` can revive any session but not one another process has
/// locked, the REPL owns exactly the conversation it has open, and ACP owns whatever the editor
/// currently has open.
///
/// Asked here rather than letting a host decline afterwards, and that placement is the whole point:
/// `prepare` evaluates a job's *gate* before the host is offered the wakeup, so a scope that
/// admitted everything would run every gated job's probe -- a shell command, or a call to someone
/// else's server -- on every tick for sessions it could never serve.
#[derive(Clone)]
pub(crate) enum SchedulerScope {
    /// Only jobs belonging to this session. The REPL, which has exactly one conversation open.
    OneSession(uuid::Uuid),
    /// Jobs the predicate accepts, re-asked every sweep, so a host whose set of runnable jobs moves
    /// under it (ACP's open editors, serve's session locks) is never working from a snapshot.
    ///
    /// Takes the whole job rather than its session id because the answer is not the only thing a
    /// host produces: `meka serve` names the job it is declining when the lock cannot be probed at
    /// all, and a predicate handed a bare uuid could not say which one.
    Jobs(std::sync::Arc<dyn Fn(&ScheduledJob) -> bool + Send + Sync>),
}
impl std::fmt::Debug for SchedulerScope {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OneSession(id) => write!(formatter, "OneSession({id})"),
            Self::Jobs(_) => formatter.write_str("Jobs(<predicate>)"),
        }
    }
}
impl SchedulerScope {
    /// Every job in the database, whoever it belongs to. Only the tests want this: a real host
    /// always has some job it cannot take.
    #[cfg(test)]
    pub(crate) fn every_job() -> Self {
        Self::Jobs(std::sync::Arc::new(|_| true))
    }

    pub(crate) fn covers(&self, job: &ScheduledJob) -> bool {
        match self {
            Self::OneSession(id) => job.session_id == *id,
            Self::Jobs(predicate) => predicate(job),
        }
    }
}
/// Start the scheduler loop. Returns the handle so the caller can abort it on shutdown; the task
/// runs until then.
///
/// Modeled on [`crate::host::http::gc::spawn`]: a tokio interval that wakes, queries, and hands
/// work to a host-supplied callback. Fires are awaited one at a time rather than spawned, so a
/// process with several due jobs runs one turn at a time. That bounds concurrent model spend, which
/// matters more here than latency: nobody is waiting on these.
pub(crate) fn spawn<Callback, Fired>(
    store: std::sync::Arc<crate::store::Store>,
    config: crate::config::ResolvedScheduleConfig,
    tools: Option<std::sync::Arc<dyn GateTools>>,
    residents: std::sync::Arc<dyn ResidentPermissions>,
    scope: SchedulerScope,
    fire: Callback,
) -> tokio::task::JoinHandle<()>
where
    Callback: Fn(Wakeup) -> Fired + Send + Sync + 'static,
    Fired: std::future::Future<Output = FireOutcome> + Send,
{
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(config.poll_interval);
        // A sweep contains its turns, so it routinely overruns `poll_interval` by minutes. Tokio's
        // default `Burst` then resolves *every* tick missed during it, so a sweep that ran twelve
        // periods long is followed by twelve immediate sweeps, eleven of which find nothing due.
        // `Delay` collapses that to one.
        //
        // It does not create a gap, and nothing here does: `Delay` schedules the next tick a period
        // after the miss is *recognized*, so the first `tick()` following a long sweep still
        // returns at once. Batches are therefore adjacent, and `max_consecutive_fires`
        // splits a backlog without spacing it out.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // The first tick resolves immediately; skip it so startup is not competing with provider
        // and MCP connection setup.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            // A panic must not end the loop either, and here that matters more than for the GC
            // scanner this pattern comes from: under `meka serve` the callback runs a whole agent
            // turn, so the surface that can panic is the entire tool loop. Losing the task would
            // stop every scheduled job for the life of the process, and stop it silently -- nothing
            // joins this handle, so the only symptom is jobs that quietly never fire again.
            let sweep = std::panic::AssertUnwindSafe(run_due(
                &store,
                &config,
                tools.as_deref(),
                &*residents,
                &scope,
                &fire,
            ));
            match futures::FutureExt::catch_unwind(sweep).await {
                // A failed sweep must not end the loop: a transient database error would otherwise
                // silently disable every scheduled job for the life of the process.
                Ok(Err(error)) => tracing::warn!("scheduler tick failed: {error}"),
                Err(panic) => {
                    let message = crate::error::panic_message(&*panic);
                    tracing::error!("scheduler tick panicked ({message}); continuing");
                }
                Ok(Ok(())) => {}
            }
        }
    })
}
/// One sweep: evaluate every due job in scope and fire what survives, up to
/// [`crate::config::ResolvedScheduleConfig::max_consecutive_fires`] turns per session.
///
/// Public because the REPL drives it directly rather than from a timer. There, the agent loop owns
/// the conversation and must be the one to run the turn, so a watcher only nudges reedline awake
/// and this runs on the agent side. `meka serve` reaches it through [`spawn`] instead.
///
/// What the budget buys is a seam, not a ceiling. A sweep contains the turns it fires -- they are
/// awaited here -- so forty due jobs still cost forty turns however small the budget is. What
/// changes is that they arrive in groups, so another session's due job is reached after five of the
/// first session's rather than after all forty. The groups are adjacent rather than spaced -- a
/// sweep that overran `poll_interval` leaves its successor already due -- so this splits a backlog
/// without slowing it. Bounding how much one conversation absorbs in total would mean holding jobs
/// across sweeps, which this deliberately does not do.
pub(crate) async fn run_due<Callback, Fired>(
    store: &crate::store::Store,
    config: &crate::config::ResolvedScheduleConfig,
    tools: Option<&dyn GateTools>,
    residents: &dyn ResidentPermissions,
    scope: &SchedulerScope,
    fire: &Callback,
) -> crate::error::Result<()>
where
    Callback: Fn(Wakeup) -> Fired,
    Fired: std::future::Future<Output = FireOutcome>,
{
    let sweep_started = Utc::now();
    let schedule_store = store.schedule_store();
    let due = schedule_store
        .list_due_scheduled_jobs(sweep_started)
        .await?;
    let mut fired: std::collections::HashMap<uuid::Uuid, usize> = std::collections::HashMap::new();
    let mut held_over = 0usize;
    for job in due {
        if !scope.covers(&job) {
            continue;
        }
        // Checked *before* `prepare`, which is what makes holding a job over free: `prepare` is
        // where a gate runs and where the schedule is advanced, so a job skipped here has done
        // neither and is still due, unchanged, on the next sweep. Reaching `prepare` and then
        // declining would pay a gate evaluation and a lease round trip to arrive at the same
        // place.
        //
        // `list_due_scheduled_jobs` orders by `next_fire_at`, so a held-over job is still the most
        // overdue one next time and goes first. Nothing starves.
        if fired.get(&job.session_id).copied().unwrap_or(0) >= config.max_consecutive_fires {
            held_over += 1;
            continue;
        }
        // Cloned whole, before `prepare` claims it. Claiming a job rewrites its schedule, advances
        // its gate baseline, and for a one-shot deletes the row outright, so nothing short of the
        // original can put it back.
        let original = job.clone();
        let short_id = original.short_id();
        // Per job, not per sweep. A sweep contains the turns it fires, so by the fifth job the
        // sweep's own clock can be minutes old: a recurring job was then advanced from a stale
        // instant to a `next_fire_at` already in the past and fired again on the next tick, and a
        // lease was stamped to expire before it was taken.
        let now = Utc::now();
        // `warn!`, not `?`. The same treatment `complete_claim` was given one level down, and for
        // the reason recorded there: propagating aborted the whole sweep, so one transient
        // `SQLITE_BUSY` skipped every *other* job due in the same tick. A job that was never
        // claimed comes back next tick, so the cost of continuing is nothing.
        let prepared = match prepare(store, config, tools, residents, job, now).await {
            Ok(prepared) => prepared,
            Err(error) => {
                // Deliberately not promising the occurrence is intact. Everything before the claim
                // leaves the job untouched, and that is the common case -- but the one error that
                // can arrive *after* it is a one-shot's restore failing, and there the row is
                // already gone. Saying "will be reconsidered" would then be the opposite of what
                // happened, in the one case a reader most needs to know about.
                tracing::warn!(
                    "failed to prepare job {short_id}: {error}. If it was claimed first, that occurrence is \
                     spent; otherwise it is untouched and the next tick reconsiders it"
                );
                continue;
            }
        };
        if let Some((claim, wakeup)) = prepared {
            // The callback is host code and can panic: a turn that blows up must not take the lease
            // with it for a whole `claim_lease`, nor end the sweep before the jobs behind it.
            // Caught here rather than only at the sweep boundary so the occurrence goes back
            // immediately, and released *without* forgiving the attempt, so a prompt that does this
            // every time climbs to `MAX_CLAIM_ATTEMPTS` and parks instead of looping.
            let outcome =
                futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(fire(wakeup))).await;
            let outcome = match outcome {
                Ok(outcome) => outcome,
                // The lease is kept and left to expire, which is the retry.
                //
                // Releasing it here was the obvious thing and produced the failure the ceiling
                // exists to prevent, by a shorter route than the one it was written for. The row
                // stays due, so the next sweep re-delivers, and since every claim raises
                // `attempts` a prompt that panics reliably is parked after three of them: half a
                // minute at the default `poll_interval`. A recurring job is never retired by
                // `missed_grace`, so it was then dead until a person noticed -- and a panic from a
                // transient condition would kill a healthy job just as fast.
                //
                // Waiting out `claim_lease` gives the same three attempts an hour apart, which a
                // blip survives and a genuinely broken prompt does not. This is the same rule the
                // gate-probe path follows, and there is now only one of it.
                Err(_) => {
                    tracing::warn!(
                        "the turn for job {short_id} panicked; its claim is left to expire, so the retry \
                         waits out [schedule].claim_lease"
                    );
                    continue;
                }
            };
            if outcome == FireOutcome::Unrunnable {
                // Neither released nor completed; see the variant. Same handling as the panicking
                // turn above, and for the same reason: the retry is paced by `claim_lease` rather
                // than by `poll_interval`, so a transient cause survives and a persistent one is
                // parked by `MAX_CLAIM_ATTEMPTS` instead of spinning.
                tracing::warn!(
                    "this host failed to run job {short_id}; its claim is left to expire, so the \
                     retry waits out [schedule].claim_lease"
                );
                continue;
            }
            if outcome == FireOutcome::Deferred {
                tracing::debug!("job {short_id} deferred; releasing the lease");
                // Also `warn!`: propagating would take the rest of the sweep with it, and a lease
                // that is not released expires on its own, so the cost of continuing is a delay
                // rather than a loss. This is the whole gain of leasing over consuming: the
                // failure mode of the handback is now "later" instead of "never".
                if let Err(error) = schedule_store
                    .release_claim(&original.id, &claim.owner)
                    .await
                {
                    tracing::warn!(
                        "job {short_id} was deferred but failed to release its lease: {error}. It will be \
                         reconsidered once the lease expires"
                    );
                }
                // A `false` here needs nothing said: the lease was already gone, and the
                // occurrence this host declined to run is open for whoever holds it now, which is
                // the outcome a deferral wants anyway.
            } else {
                // Delivered, so the occurrence is spent: advance a job that lives on, retire one
                // whose moment has passed. Written after the turn rather than before it, which is
                // what the attempt counter buys.
                match schedule_store
                    .complete_claim(
                        &original.id,
                        &claim.owner,
                        claim.next_fire_at,
                        Some(now),
                        claim.gate_baseline.as_deref(),
                    )
                    .await
                {
                    Ok(ClaimClosed::Yes) => {}
                    // Not a problem, and not silent either: a job that fires and then cancels
                    // itself is an ordinary shape, and the turn that did it should not look like
                    // a fault in the log.
                    Ok(ClaimClosed::RowGone) => tracing::debug!(
                        "job {short_id} ran and its row was removed during the turn, so there was no \
                         occurrence left to close"
                    ),
                    // The same outcome as the `Err` below, reached silently: the turn ran, and the
                    // occurrence it belonged to is still open because the lease expired under it.
                    // Worth its own sentence because the remedy differs -- an error is a database
                    // problem, this is a `claim_lease` shorter than a turn takes.
                    Ok(ClaimClosed::LeaseLost) => tracing::warn!(
                        "job {short_id} ran, but its lease had already expired, so the occurrence stayed \
                         open and may be delivered again. Raise [schedule].claim_lease past how \
                         long this job's turn takes"
                    ),
                    Err(error) => tracing::warn!(
                        "job {short_id} ran but failed to close its occurrence: {error}. The lease expires \
                         on its own and the job is retried, which may deliver it twice"
                    ),
                }
                // Counted only once a turn has actually been spent. A job `prepare` retired (a
                // declining gate, a one-shot past its grace period) and a job the host handed back
                // both cost the conversation nothing, so neither may consume a session's budget --
                // otherwise five quiet watchers would starve the sixth job that had something to
                // say.
                *fired.entry(original.session_id).or_default() += 1;
            }
        }
    }
    // Said out loud: a cap that bounds coverage silently reads as "everything ran". `info!` rather
    // than `warn!` because holding a job over is the budget working, not a fallback -- the jobs are
    // intact and the next sweep takes them.
    if held_over > 0 {
        let ceiling = config.max_consecutive_fires;
        tracing::info!(
            "held over {held_over} due job(s) past [schedule].max_consecutive_fires ({ceiling}); they keep their \
             occurrence and run on the next sweep"
        );
    }
    Ok(())
}
/// The lease this host holds on one occurrence.
///
/// Carried out of [`prepare`] so the host that turns out to be unable to run the job hands back
/// exactly what it took, and so every write that follows is scoped to the claim *this process* won
/// rather than to the job id alone. A stale writer whose lease has since expired and been taken by
/// someone else changes nothing.
#[derive(Debug, Clone)]
pub(crate) struct Claim {
    /// Proof of ownership, matched against `scheduled_jobs.claimed_by`.
    pub(crate) owner: String,
    /// Where the schedule goes once the turn is delivered: `Some` advances a job that lives on,
    /// `None` retires one whose moment is spent.
    pub(crate) next_fire_at: Option<DateTime<Utc>>,
    /// What the gate saw, to be recorded when the occurrence is disposed of.
    ///
    /// Carried rather than written as soon as the gate returns, because a baseline is only true
    /// once the occurrence it belongs to is finished with. A host that evaluates, decides to fire
    /// and then cannot run the turn hands the occurrence back, and if it had already advanced the
    /// baseline the next host would compare the new value against itself, see no change, and never
    /// fire: the change would be swallowed by the handback that was supposed to preserve it.
    pub(crate) gate_baseline: Option<String>,
}
/// What the store said when a session's row was asked for.
///
/// A lookup that failed is kept apart from one that found no row, and both from one that found a
/// row: the first two decide nothing and leave the occurrence for the next sweep, while a row
/// answers for itself. Folding a failed read into "no level" once re-granted gate authority at the
/// host's level on a `SQLITE_BUSY`, so a session recorded at `read` evaluated an unsandboxed shell
/// gate at `unrestricted` for that sweep.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SessionLookup<'a> {
    Read(Option<&'a crate::store::SessionSummary>),
    Failed,
}
/// What a session's permission is *now* according to its row, as opposed to what it was when a
/// gate was authored.
///
/// The row is the authority for a session this process does not hold; for one it does, the live
/// cell outranks it, and [`prepare`] asks [`ResidentPermissions`] first. Every door that creates
/// a session writes its level and every surface that moves it writes the row back, so a row that
/// records nothing is one no host wrote, and it answers `none`: nothing runs on the strength of a
/// level nobody set. The polling process's own `--permission` standing in for a bare row would let
/// a `meka serve` sharing the data directory fire a gate on a session at whatever that daemon was
/// started with.
///
/// Filtered by the enabled set, like the other readers of this column. A row records what a
/// session was set to, not what this installation still permits, and the two diverge the moment an
/// operator narrows `[permissions].enabled` and restarts: the session re-attaches clamped, while
/// this read saw the unclamped row and kept firing the gate. That was verified end to end against
/// a live `meka serve`, and the creation door two files over returns 403 for the same authority,
/// so the two doors disagreed about one job.
///
/// A lookup that failed, or found no row, answers `none` here; the callers that can defer instead
/// decline it before asking. See [`SessionLookup`].
///
/// Shared with [`wake_would_produce_work`] rather than duplicated: the watcher that decides whether
/// to interrupt a prompt has to reach the same answer as the door that decides whether to fire, or
/// it wakes the shell for work that will not happen.
pub(crate) fn live_permission(
    lookup: SessionLookup<'_>,
    config: &crate::config::ResolvedScheduleConfig,
    session_id: uuid::Uuid,
) -> crate::permission::Permission {
    let session = match lookup {
        // Nothing is executable at `none`, which is the level that lets a job neither act nor be
        // woken until the row can be read again.
        SessionLookup::Failed | SessionLookup::Read(None) => {
            return crate::permission::Permission::None;
        }
        SessionLookup::Read(Some(session)) => session,
    };
    config
        .enabled_permissions
        .admit_recorded(session.permission, &format!("session {session_id}"))
        .unwrap_or(crate::permission::Permission::None)
}
/// Whether any of `due` belongs to this session and is not parked.
///
/// Separated from the two database reads around it so the rule can be asserted directly: a job at
/// [`MAX_CLAIM_ATTEMPTS`] stays in the table on purpose (listed, cancelable, reported as held), so
/// "there is a due row" and "something will run" are different questions and the watcher has to ask
/// the second one.
pub(crate) fn has_runnable_job(due: &[ScheduledJob], session_id: uuid::Uuid) -> bool {
    due.iter()
        .any(|job| job.session_id == session_id && job.attempts < MAX_CLAIM_ATTEMPTS)
}
/// Whether waking this session's prompt would actually produce work.
///
/// `list_due_scheduled_jobs` is a pure SQL predicate, while [`prepare`] declines for several
/// further reasons. Waking on the SQL predicate alone interrupts the prompt every poll to run
/// nothing, wherever the row deliberately does not move: a job parked at [`MAX_CLAIM_ATTEMPTS`], or
/// a session whose live level cannot do unattended work.
///
/// This answers the subset of `prepare`'s question that costs a row read and nothing else. It is
/// deliberately *conservative* rather than exact: a gated job still wakes the prompt, because the
/// only way to know whether its gate passes is to run the probe, and running a side-effecting probe
/// twice per poll to answer the same question would be worse than the interruption. The invariant
/// is that anything this refuses, `prepare` would also have refused -- never the reverse.
pub(crate) async fn wake_would_produce_work(
    store: &crate::store::Store,
    config: &crate::config::ResolvedScheduleConfig,
    residents: &dyn ResidentPermissions,
    session_id: uuid::Uuid,
    now: DateTime<Utc>,
) -> crate::error::Result<bool> {
    let due = store.schedule_store().list_due_scheduled_jobs(now).await?;
    if !has_runnable_job(&due, session_id) {
        return Ok(false);
    }
    // The same source `prepare` reads, or the two disagree about the one session this watcher
    // exists for: a row a level change failed to write would have this decline a wake the fire
    // door would take.
    if let Some(level) = residents.live_permission_of(session_id).await {
        return Ok(level.allows_unattended_work());
    }
    // Read once for the session, not once per job: every job here shares it.
    let lookup = store.session_info(session_id).await;
    let lookup = match &lookup {
        Ok(info) => SessionLookup::Read(info.as_ref()),
        // Fail closed on a lookup that could not confirm the level, the same way `prepare` does:
        // a session whose row cannot be read is not one to wake a prompt for.
        Err(error) => {
            tracing::warn!(
                "failed to read session {session_id} while checking for due work: {error}"
            );
            SessionLookup::Failed
        }
    };
    Ok(live_permission(lookup, config, session_id).allows_unattended_work())
}
/// Decide what to do with one due job: retire it, reschedule it quietly, or produce the [`Wakeup`]
/// that spends a turn.
pub(crate) async fn prepare(
    store: &crate::store::Store,
    config: &crate::config::ResolvedScheduleConfig,
    tools: Option<&dyn GateTools>,
    residents: &dyn ResidentPermissions,
    job: ScheduledJob,
    now: DateTime<Utc>,
) -> crate::error::Result<Option<(Claim, Wakeup)>> {
    let schedule_store = store.schedule_store();
    let memory = store.scheduler_memory();
    let late_by = now - job.next_fire_at;
    let recurring = job.schedule.is_recurring();
    let short_id = job.short_id();
    // A one-shot far past its moment is noise rather than a reminder: "join the standup" delivered
    // five days late helps nobody. Recurring jobs need no equivalent rule -- their occurrences are
    // one period apart, so the most recent missed one is always less than a period old.
    if !recurring
        && chrono::Duration::from_std(config.missed_grace)
            .map(|grace| late_by > grace)
            .unwrap_or(false)
    {
        // Claimed rather than deleted outright, so the announcement is made once by whichever host
        // actually removed the row rather than once per host that had it in its due list.
        if schedule_store
            .retire_unclaimed(&job.id, job.next_fire_at, now)
            .await?
        {
            // One of the two doors out of the table that do not go through `delete_scheduled_job`;
            // the other is `complete_claim` retiring a job whose moment is spent, which is reached
            // only after the clears below have already run. Without these the ledgers keyed by job
            // id outlive the job, which a held one-shot reaches routinely: it survives every sweep
            // until its grace window closes, and this is where it ends.
            memory.forget(&job.id);
            let late = format_late(late_by);
            tracing::warn!(
                "dropping one-shot job {short_id}: due {late} ago, past the missed-job grace period"
            );
        }
        return Ok(None);
    }

    // One lookup, for every job rather than only gated ones, serving the working directory a gate
    // runs in and the live level both the checks below need.
    //
    // Not skipped for an ungated job: at `none` such a job would keep firing, waking a model that
    // can read nothing and act on nothing. The query costs one row read per job actually due,
    // against a model turn.
    let looked_up = store.session_info(job.session_id).await;
    let (session, lookup) = match &looked_up {
        Ok(info) => (info.as_ref(), SessionLookup::Read(info.as_ref())),
        Err(error) => {
            // Not silently "no session". A failed lookup means the level cannot be confirmed, and
            // `live_permission` fails closed on it rather than falling back to the host's level.
            let session_id = job.session_id;
            tracing::warn!(
                "failed to read session {session_id} while preparing job {short_id}: {error}"
            );
            (None, SessionLookup::Failed)
        }
    };
    let gate_cwd = session.and_then(|info| info.cwd.clone());
    // A row that could not be read, or is gone, decides nothing, and the occurrence stays for the
    // next sweep, exactly as it does when a gate tool is not yet resolvable. Falling through to the
    // permission branch declined the job at `none`: a recurring job was advanced past an
    // occurrence its probe never ran for, and the log told the operator to raise a session that
    // may be at `unrestricted`, on the strength of one `SQLITE_BUSY`.
    if matches!(lookup, SessionLookup::Failed | SessionLookup::Read(None)) {
        return Ok(None);
    }

    // What the session's permission is *now*, as opposed to what it was when the gate was
    // authored. A session this process holds answers from its cell, which is what the user's
    // Shift+Tab moved and what its own turns dispatch against; the row it writes back can lag on
    // a failed write, and a poller in the same process reading the row then fired a shell gate on
    // authority the user had withdrawn. Every other session answers from its row, filtered by the
    // enabled set; see `live_permission` for why nothing else does.
    let live_permission = match residents.live_permission_of(job.session_id).await {
        Some(level) => level,
        None => live_permission(lookup, config, job.session_id),
    };

    let coalesced = occurrences_between(&job.schedule, job.next_fire_at, now);
    // `Some` only for a job that lives on. A one-shot's moment is spent, and a cron pattern with
    // nothing left in range has no future to be scheduled for; both are retired by `complete_claim`
    // once the turn is delivered, which is the only place the scheduler deletes a job.
    //
    // Derived before the claim because a refusal needs it too: it is what spends a recurring job's
    // occurrence without evaluating anything.
    let next_fire_at = job
        .schedule
        .next_after_delivering(job.next_fire_at, now)
        .filter(|_| recurring);

    // Spend the occurrence of a job refused before its claim was taken.
    //
    // A recurring job advances, which is the documented rule and the same thing a gate that ran and
    // said no does: without it a held job sits permanently due and then reports a month-long
    // backlog the moment it is authorized again.
    //
    // A one-shot is left completely alone, and that asymmetry is the reason this exists rather than
    // the claim being taken first. It is cheaper to refuse before leasing than to lease and hand
    // back, and under the design this replaced it was not merely cheaper but necessary: claiming a
    // one-shot was a `DELETE`, so a refusal that followed one had to re-`INSERT`, and an `INSERT`
    // cannot tell "I deleted this a moment ago" from "the user canceled it in between". Leasing
    // removed that hazard; every refusal that needs nothing but a row read is still made here,
    // because there is no reason to pay for a lease to reach the same answer.
    async fn decline_before_claiming(
        store: &ScheduleStore,
        job: &ScheduledJob,
        now: DateTime<Utc>,
        next_fire_at: Option<DateTime<Utc>>,
    ) -> crate::error::Result<()> {
        // Whether *this* host won the advance does not matter: either way the occurrence has moved
        // off the value every host read, and no host will deliver it.
        if let Some(next) = next_fire_at {
            store
                .advance_unclaimed(&job.id, job.next_fire_at, now, next)
                .await?;
        }
        Ok(())
    }

    // A job that has crashed its host repeatedly is parked rather than retried again.
    //
    // Claiming is a lease now, so nothing else stops a prompt that reliably kills the process from
    // being picked up on every expiry, forever. The row stays exactly where it is: listed,
    // cancelable, and reported as held on every surface, because destroying a user's job over a
    // failure meka cannot diagnose would be worse than leaving it visible and inert.
    if job.attempts >= MAX_CLAIM_ATTEMPTS {
        if memory.declined_for_permission_first_time(&job.id, "crashed") {
            let reason = job_withheld_reason(memory, &job, live_permission, tools)
                .unwrap_or_else(|| "it has been parked".to_string());
            tracing::warn!("job {short_id} not fired: {reason}");
        }
        return Ok(None);
    }

    // Before the gate, and regardless of whether there is one. At `none` nothing the turn could
    // reach is dispatchable: it reads nothing, changes nothing, and `schedule_cancel` is refused
    // too, so it cannot stop itself being woken again. Registration does not depend on the level --
    // the model is shown the job and offered the tool, and only refused when it reaches for one --
    // so the turn is left able to describe its predicament and unable to act on it. Firing anyway
    // spends tokens on that, every interval, until an operator notices.
    if !live_permission.allows_unattended_work() {
        if memory.declined_for_permission_first_time(
            &job.id,
            &format!("unattended-work:{live_permission}"),
        ) {
            tracing::warn!(
                "job {short_id} not fired: the session is at {live_permission}, where no tool is executable, so the turn \
                 could neither act nor cancel the job. Raise the session to restore it"
            );
        } else {
            tracing::debug!(
                "job {short_id} still not fired: the session remains at {live_permission}"
            );
        }
        decline_before_claiming(&schedule_store, &job, now, next_fire_at).await?;
        return Ok(None);
    }

    // The same predicate the two creation doors use, asked again here against both the recorded
    // level and the live one.
    //
    // Checking only the recorded value was a tautology: the row is written by a door that already
    // demanded the level, and nothing ever updates the column, so the recorded value always
    // satisfies whatever admitted it. The comparison could not fail, and the case it was written
    // for -- the session cycles down to `read`, or a `meka serve --permission read` restarts and
    // inherits the row -- went unnoticed. The live level is what makes the withdrawal real; the
    // recorded one still matters because a hand-edited or unparseable `gate_permission` decodes as
    // `Permission::None` and must stay refused.
    //
    // Going through `gate_probe_is_authorized` rather than re-deriving the rule is what keeps the
    // doors in agreement. Asking `allows_unattended_shell` here regardless of probe kind accepted
    // every tool gate at creation and then declined it forever at fire time, with a message about a
    // shell command the job did not have: the headline case (`mcp__…__unseen` at `read`) never
    // called its probe once.
    //
    // The occurrence is declined, exactly as a gate that ran and said no is declined. A gate is the
    // condition on the job, so a gate that could not be evaluated has not passed, and firing anyway
    // converts a conditional job into an unconditional one. The shape that makes this concrete is
    // `every = "1m"` with a `changed` gate: firing it unconditionally turns a near-silent job into
    // a turn a minute, which is the opposite of what the row asks for and expensive besides.
    if let Some(gate) = &job.gate
        && let Some((refusal, level)) = gate_withheld_reason(gate, live_permission, tools)
    {
        // Not decidable yet is not refused. A server mid-handshake, or a host with no
        // dispatcher for this tool at all, cannot say whether the gate passes; declining spent
        // the occurrence anyway, so a `6h` job due at startup was advanced six hours without
        // its probe ever running, and a host without the server advanced rows a host with it
        // would have evaluated. The same rule `job_withheld` applies before it reports.
        if matches!(refusal, GateRefusal::ToolUnavailable)
            && let GateProbe::Tool { name, .. } = &gate.probe
            && tools.is_none_or(|tools| tools.is_still_connecting(name))
        {
            tracing::debug!(
                "job {short_id} not decided: gate tool '{name}' is not resolvable yet; leaving the \
                 occurrence for the next sweep"
            );
            return Ok(None);
        }
        // Said once per decline, not once per evaluation. The condition is a standing state
        // rather than an event: a session left below the bar with an `every = "1m"` job wrote
        // this line every minute for as long as it stayed there, which buries the log it is
        // supposed to be the signal in. The id is cleared the moment the gate is authorized
        // again, so a later withdrawal is announced afresh.
        let explained = refusal.explain(&gate.probe, level);
        if memory.declined_for_permission_first_time(&job.id, &explained) {
            // `gate_withheld_reason` reports the live level first because it is the one an
            // operator can act on, so a refusal carrying the *recorded* level instead means the
            // live level was fine: a hand-edited or damaged row, which no amount of cycling the
            // session will fix. Saying which of the two it was is the only way to tell those
            // apart from the log.
            if level == live_permission {
                // Naming the level it was authorized at only helps when that level would still
                // pass; otherwise it reads as a promise that restoring it is enough.
                match gate_probe_is_authorized(&gate.probe, gate.permission, tools) {
                    Ok(()) => {
                        let recorded = gate.permission;
                        tracing::warn!(
                            "job {short_id} not fired: {explained}. It was authorized at \
                             {recorded}; raise the session back to restore it"
                        );
                    }
                    Err(_) => {
                        tracing::warn!("job {short_id} not fired: {explained}")
                    }
                }
            } else {
                tracing::warn!(
                    "job {short_id} not fired: {explained}, which is the level recorded when the gate was \
                     authorized"
                );
            }
        } else {
            tracing::debug!("job {short_id} still not fired: its gate is still unauthorized");
        }
        decline_before_claiming(&schedule_store, &job, now, next_fire_at).await?;
        return Ok(None);
    }
    // Nothing is holding this job back, so a later withdrawal is announced afresh rather than
    // swallowed by the once-per-decline suppression above. An ungated job passes through here too:
    // it is held by nothing, and the `none` floor above is the only thing that could have stopped
    // it.
    memory.clear_permission_decline(&job.id);

    // Lease the occurrence before doing anything that can fail or hang.
    //
    // This is what arbitrates between hosts, which is why it is conditional. Every `meka serve`,
    // REPL and ACP session polls the same table, so one occurrence is in several hosts' due lists
    // at once; whoever takes the lease owns it, and the rest return here having neither evaluated
    // the gate nor spent the occurrence.
    //
    // The lease is taken *before* the gate runs and released or completed after, so the row is
    // never absent while this host is working: a refusal has nothing to put back, and a crash
    // between the claim and the turn expires rather than spending the occurrence.
    let mut claim = Claim {
        owner: memory.owner().to_string(),
        next_fire_at,
        gate_baseline: None,
    };
    if !schedule_store
        .claim_occurrence(
            &job.id,
            job.next_fire_at,
            &claim.owner,
            now,
            now + chrono::Duration::from_std(config.claim_lease).unwrap_or_else(|_| {
                // Only an out-of-range configured duration reaches this, and a lease that cannot be
                // expressed must not become one that never expires.
                chrono::Duration::hours(1)
            }),
        )
        .await?
    {
        tracing::debug!("job {short_id} was claimed for this occurrence by another host");
        return Ok(None);
    }

    let gate_output = match &job.gate {
        None => None,
        Some(gate) => {
            match evaluate_gate(
                gate,
                config.gate_timeout,
                gate_cwd.as_deref(),
                tools,
                Some(job.session_id),
            )
            .await
            {
                Ok(outcome) => {
                    // An evaluation that produced an answer, whichever answer it was, ends any
                    // standing failure: the probe works.
                    memory.clear_probe_failure(&job.id);
                    // Persist the new baseline even when it did not fire; that is exactly how a
                    // `changed` gate stops firing once it has seen the new value. A retired job has
                    // no row left to write to, and needs none -- it will not be evaluated again.
                    //
                    // `baseline`, not `output`: for a pointer predicate the two differ, and storing
                    // the whole result would put the moving field the pointer excludes back into
                    // the comparison, firing the gate every interval.
                    claim.gate_baseline = Some(outcome.baseline);
                    if !outcome.fired {
                        // The occurrence is spent: the condition was asked and said no. A recurring
                        // job moves to its next occurrence and a one-shot's moment has passed,
                        // which is the documented rule and the same thing `complete_claim` does
                        // after a turn. The only difference is that no turn ran.
                        tracing::debug!("gate for job {short_id} declined to fire");
                        match schedule_store
                            .complete_claim(
                                &job.id,
                                &claim.owner,
                                claim.next_fire_at,
                                None,
                                claim.gate_baseline.as_deref(),
                            )
                            .await
                        {
                            Ok(ClaimClosed::Yes) => {}
                            Ok(ClaimClosed::RowGone) => tracing::debug!(
                                "job {short_id}'s gate declined and its row was removed while the probe \
                                 ran, so there was no occurrence left to close"
                            ),
                            // The lease was taken from under this host while the probe ran, so the
                            // baseline it just measured was not recorded. Said out loud because
                            // the visible symptom is a `changed` gate firing twice for one change,
                            // which reads as a flapping probe rather than a lost write.
                            Ok(ClaimClosed::LeaseLost) => tracing::warn!(
                                "job {short_id}'s gate declined, but this host no longer held the lease, \
                                 so the occurrence stayed open. It may be evaluated again"
                            ),
                            Err(error) => tracing::warn!(
                                "job {short_id} declined but failed to close its occurrence: {error}. The \
                                 lease expires on its own"
                            ),
                        }
                        return Ok(None);
                    }
                    Some(outcome.output)
                }
                Err(error) => {
                    // Loud on purpose. A watcher whose probe breaks produces the same silence as a
                    // watcher with nothing to report, and that is the failure most likely to go
                    // unnoticed for weeks.
                    //
                    // Counted as well as logged, so the *model* hears about it too once the
                    // condition is standing rather than momentary. The log alone reaches only
                    // whoever is reading it, and a scheduled job exists precisely because nobody
                    // is. The row as it stands *now* -- before this arm's own disposal, which
                    // writes neither half of the witness. See [`standing_probe_failure`].
                    let failures = memory.record_probe_failure(&job, &error);
                    tracing::warn!("gate for job {short_id} failed: {error} (failure {failures})");
                    // The same disposal a refusal gets, and for the same reason: the condition was
                    // not answered, so this occurrence is over. A recurring job advances to its
                    // next one; a one-shot keeps its row, because its moment has not been spent on
                    // anything.
                    //
                    // Simply releasing the lease was the obvious thing and was wrong. It leaves
                    // `next_fire_at` where it was, so the row is due again on the very next sweep:
                    // a six-hour job whose server is down was re-probed every `poll_interval`
                    // rather than every six hours, and a probe that *hangs* burned the whole
                    // `gate_timeout` out of each sweep, delaying every job behind it. Under the
                    // old design the schedule had already been advanced to claim the job, so this
                    // was structurally impossible and nothing here had to think about it.
                    //
                    // `fired_at` and `gate_baseline` are both `None`. Nothing fired, and nothing
                    // was measured -- leaving the baseline alone is what makes the recovery
                    // correct, because the next successful evaluation then compares against the
                    // last value actually observed and reports the change that happened while the
                    // probe was broken.
                    match claim.next_fire_at {
                        Some(next) => match schedule_store
                            .complete_claim(&job.id, &claim.owner, Some(next), None, None)
                            .await
                        {
                            Ok(ClaimClosed::Yes) => {}
                            Ok(ClaimClosed::RowGone) => tracing::debug!(
                                "job {short_id}'s gate failed and its row was removed while the probe \
                                 ran, so there was no occurrence left to close"
                            ),
                            // The occurrence stayed open, so the next sweep probes again. Said out
                            // loud because it is the state this whole arm exists to avoid, and it
                            // is otherwise indistinguishable from a probe that is simply failing
                            // often.
                            Ok(ClaimClosed::LeaseLost) => tracing::warn!(
                                "job {short_id}'s gate failed and this host no longer held the lease, so \
                                 the occurrence stayed open and will be probed again"
                            ),
                            Err(error) => tracing::warn!(
                                "job {short_id}'s gate failed and then failed to close its occurrence: {error}. \
                                 The lease expires on its own"
                            ),
                        },
                        // Nothing at all: the lease is *kept*, and left to expire.
                        //
                        // This is the one case the advance above cannot reach -- a schedule with no
                        // next occurrence to move to, which in practice means a one-shot. Releasing
                        // the lease here makes the row due again on the very next sweep, so a probe
                        // that is down gets re-run every `poll_interval` and, since each claim
                        // raises `attempts`, the job is parked after three of them: half a minute
                        // at the default. An MCP server restarting anywhere near a one-shot's due
                        // time would silently destroy the reminder, which is a worse failure than
                        // the one the advance is here to prevent.
                        //
                        // A lease already means "not available until then", so holding it *is* the
                        // backoff, and it needs no new state to express. The job is retried once
                        // per `claim_lease` rather than once per tick, and because each of those
                        // retries is a fresh claim the attempt ceiling still applies -- three
                        // hourly attempts before parking rather than three ten-second ones, which
                        // is a budget a transient outage survives and a broken gate does not.
                        None => tracing::debug!(
                            "job {short_id}'s gate failed and it has no next occurrence; holding the lease \
                             so the retry waits out [schedule].claim_lease"
                        ),
                    }
                    return Ok(None);
                }
            }
        }
    };

    // The schedule is advanced by `run_due` once the turn has actually been delivered, not here.
    // Until then this host holds a lease and the row is untouched, so a crash costs a retry rather
    // than the occurrence.
    let schedule = job.schedule.describe();
    tracing::info!("firing scheduled job {short_id} ({schedule})");
    Ok(Some((claim, Wakeup {
        job,
        gate_output,
        late_by,
        coalesced,
    })))
}

/// Run a gate and decide whether the job it guards should fire.
///
/// Errors are for the gate itself failing (spawn failure, timeout, a tool that cannot be reached),
/// never for the condition being false. The distinction matters: a watcher that goes quiet because
/// its probe broke looks exactly like a healthy watcher with nothing to report, so the caller must
/// surface an `Err` rather than treating it as "no change".
pub(crate) async fn evaluate_gate(
    gate: &Gate,
    timeout: Duration,
    cwd: Option<&std::path::Path>,
    tools: Option<&dyn GateTools>,
    session_id: Option<uuid::Uuid>,
) -> Result<GateOutcome, String> {
    let probe = run_probe(&gate.probe, timeout, cwd, tools, session_id).await?;
    apply_predicate(&gate.predicate, &probe, gate.last_output.as_deref())
}
/// Obtain the value a gate judges, without judging it.
pub(crate) async fn run_probe(
    probe: &GateProbe,
    timeout: Duration,
    cwd: Option<&std::path::Path>,
    tools: Option<&dyn GateTools>,
    session_id: Option<uuid::Uuid>,
) -> Result<ProbeOutcome, String> {
    match probe {
        GateProbe::Shell { command } => run_shell_probe(command, timeout, cwd).await,
        GateProbe::Tool { name, arguments } => {
            let Some(tools) = tools else {
                // Not a misconfiguration to report at creation: the host that authored the job can
                // dispatch tools, and this is a *different* host picking the row up. Declining is
                // the same answer a disconnected server gets, for the same reason.
                return Err(format!(
                    "gate calls `{name}`, which this process cannot dispatch"
                ));
            };
            tools.call(name, arguments, timeout, cwd, session_id).await
        }
    }
}
/// The shell probe: unsandboxed, in the session's directory, bounded by `timeout`.
///
/// Authoring one requires `unrestricted`, which is the same level at which `execute_command` runs
/// arbitrary unsandboxed commands, so a sandbox here would block the ordinary cases (`gh`, `curl`)
/// without raising the bar the agent must clear.
pub(crate) async fn run_shell_probe(
    command: &str,
    timeout: Duration,
    cwd: Option<&std::path::Path>,
) -> Result<ProbeOutcome, String> {
    let mut builder = gate_command_builder(command);
    // The creating session's directory, not the host process's. A gate is almost always written by
    // the model right after verifying the same command through `execute_command`, which runs in the
    // session's cwd -- so a gate that runs anywhere else silently stops matching the command the
    // model tested. Under a `meka serve` systemd unit the process cwd is `/`, where a repo-relative
    // `gh pr checks` exits non-zero with empty stdout, and a `changed` gate then latches onto
    // that empty baseline and never fires again.
    if let Some(directory) = cwd {
        if directory.is_dir() {
            builder.current_dir(directory);
        } else {
            let directory = directory.display();
            tracing::warn!(
                "gate's session directory '{directory}' no longer exists; running it in the current \
                 directory instead"
            );
        }
    }
    builder
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        // Dropping the future on timeout must not leave the command running until the next tick
        // spawns another. Note this reaps the direct child only: a gate whose shell backgrounds
        // something of its own can still orphan it, which is a reason to keep gates to simple
        // checks.
        .kill_on_drop(true);

    let mut child = builder
        .spawn()
        .map_err(|error| format!("failed to start gate: {error}"))?;

    // Read up to the parse limit and one byte more, never the whole pipe: `wait_with_output` held
    // everything a runaway `cat access.log` or `yes` produced until the timeout, so a fast producer
    // took the host's memory inside its own time budget. Past the limit the probe is cut off, since
    // nothing after it can be parsed anyway.
    use tokio::io::AsyncReadExt as _;
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();
    let read_bounded = |pipe: Option<tokio::process::ChildStdout>, limit: usize| async move {
        let mut buffer = Vec::new();
        if let Some(pipe) = pipe {
            tokio::io::AsyncReadExt::take(pipe, limit as u64 + 1)
                .read_to_end(&mut buffer)
                .await?;
        }
        Ok::<_, std::io::Error>(buffer)
    };
    // Drained to its end rather than `take`n: a `take` closes the read end once the cap is reached,
    // and the child's next write to stderr is then `SIGPIPE`, so a gate that logged more than the
    // cap died before it could print its answer and was recorded as a failed probe.
    let read_stderr_bounded = |pipe: Option<tokio::process::ChildStderr>| async move {
        let mut kept = Vec::new();
        if let Some(mut pipe) = pipe {
            let mut chunk = [0u8; 8192];
            loop {
                let read = tokio::io::AsyncReadExt::read(&mut pipe, &mut chunk).await?;
                if read == 0 {
                    break;
                }
                let room = GATE_OUTPUT_LIMIT.saturating_sub(kept.len());
                kept.extend_from_slice(&chunk[..read.min(room)]);
            }
        }
        Ok::<_, std::io::Error>(kept)
    };
    let collected = async {
        let (stdout, stderr) = tokio::join!(
            read_bounded(stdout_pipe, GATE_PARSE_LIMIT),
            read_stderr_bounded(stderr_pipe)
        );
        let mut stdout = stdout?;
        let stderr = stderr?;
        let cut_off = stdout.len() > GATE_PARSE_LIMIT;
        if cut_off {
            stdout.truncate(GATE_PARSE_LIMIT);
            // The producer is still running with nobody reading; waiting for it to finish on its
            // own would be waiting for the timeout.
            if let Err(error) = child.kill().await {
                tracing::debug!("failed to stop a gate that exceeded its output limit: {error}");
            }
        }
        let status = child.wait().await?;
        Ok::<_, std::io::Error>((stdout, stderr, status, cut_off))
    };
    let (stdout_bytes, stderr_bytes, status, cut_off) =
        match tokio::time::timeout(timeout, collected).await {
            Ok(Ok(collected)) => collected,
            Ok(Err(error)) => return Err(format!("gate failed to run: {error}")),
            Err(_) => {
                return Err(format!(
                    "gate exceeded its {} budget",
                    humantime::format_duration(timeout)
                ));
            }
        };
    if cut_off {
        tracing::debug!(
            "gate output exceeded {GATE_PARSE_LIMIT} bytes and was cut off; the command was stopped"
        );
    }

    let stdout = String::from_utf8_lossy(&stdout_bytes);

    // A non-zero exit is reported, not refused.
    //
    // The failure this exists for is a watcher that breaks silently: an expired token has `gh` exit
    // non-zero with empty stdout, the first evaluation stores `""` as the baseline, and every
    // evaluation after compares `"" == ""` and stays quiet forever. The line is `debug!`, so `-vv`
    // is what surfaces it; it cannot be `warn!` for the reason immediately below, which is that a
    // non-zero exit is the *normal* state of a large class of correct gates.
    //
    // Refusing to produce output would not: for a large class of perfectly good gates, a non-zero
    // exit *is* the signal. `diff -q a b` and `git diff --exit-code` exit 1 exactly when there is a
    // difference; `grep ERROR log` exits 1 through the whole quiet period it is watching; `curl -f`
    // exits non-zero until the endpoint comes back. Treating any of those as broken would silence
    // the gate permanently, which is the bug this was meant to fix, pointed the other way. The
    // `succeeded` flag carries the exit status to whichever predicate asked for it instead.
    if !status.success() {
        let stderr = truncate_gate_output(&String::from_utf8_lossy(&stderr_bytes));
        let detail = if stderr.is_empty() {
            String::new()
        } else {
            format!(": {stderr}")
        };
        tracing::debug!(
            "gate command exited with {status}{detail}; its output still stands, since a non-zero \
             exit is how several common gates signal a change"
        );
    }

    Ok(ProbeOutcome::new(&stdout, None, status.success()))
}
/// Decide whether a probe's result means "fire".
///
/// Pure, so every predicate is testable without spawning anything.
pub(crate) fn apply_predicate(
    predicate: &GatePredicate,
    probe: &ProbeOutcome,
    last_output: Option<&str>,
) -> Result<GateOutcome, String> {
    // A first evaluation has no baseline, so "changed" is the honest answer. It also means a
    // freshly created watcher proves itself immediately instead of staying silent until
    // something happens, which is when a typo in the probe would otherwise surface.
    let changed_from = |current: &str| last_output != Some(current);

    match predicate {
        GatePredicate::Changed => Ok(GateOutcome {
            fired: changed_from(&probe.text),
            output: probe.text.clone(),
            baseline: probe.text.clone(),
        }),
        GatePredicate::Succeeded => Ok(GateOutcome {
            fired: probe.succeeded,
            output: probe.text.clone(),
            baseline: probe.text.clone(),
        }),
        GatePredicate::Matches { pattern } => {
            // A pattern that no longer compiles cannot be an error here: `evaluate_gate` reserves
            // `Err` for a probe that broke, and this one ran fine. It is refused at creation, so
            // reaching this means a hand-edited row; declining is the safe direction and the
            // warning says which job to fix.
            let fired = match regex::Regex::new(pattern) {
                Ok(regex) => regex.is_match(&probe.text),
                Err(error) => {
                    tracing::warn!("gate pattern /{pattern}/ does not compile: {error}");
                    false
                }
            };
            Ok(GateOutcome {
                fired,
                output: probe.text.clone(),
                baseline: probe.text.clone(),
            })
        }
        GatePredicate::At { pointer, is } => {
            let value = match pointed_at(probe, pointer) {
                Pointed::Found(value) => Some(value),
                Pointed::Absent => None,
                // An `Err`, like a probe that could not be spawned: the pointer describes a shape
                // this result does not have, so no predicate over it has an honest answer. The
                // caller declines the occurrence and warns, naming the job.
                Pointed::NotADocument => {
                    return Err(format!(
                        "gate points at `{}` but the probe did not return JSON: {}",
                        pointer,
                        elide_for_message(&probe.text)
                    ));
                }
            };
            // Serialized rather than compared as a `Value` because the baseline has to survive a
            // round trip through a TEXT column, and canonically because `serde_json` is built with
            // `preserve_order`: a `Value`'s object keys come back in the order the *input* had
            // them, so a server that emits the same object with its keys in a different order
            // renders as a different string. That is precisely the flap `at` exists to prevent,
            // arriving through the door left open for it. Arrays keep their order, which is part of
            // the value rather than an artifact of how it was written.
            let rendered = value
                .as_ref()
                .map(|value| truncate_gate_output(&canonical_json(value).to_string()))
                .unwrap_or_default();
            let fired = match is {
                PointerTest::NotEmpty => value.as_ref().is_some_and(json_is_non_empty),
                PointerTest::Empty => !value.as_ref().is_some_and(json_is_non_empty),
                PointerTest::Changed => changed_from(&rendered),
            };
            Ok(GateOutcome {
                fired,
                // The turn still sees the whole result: the pointer narrows what is *judged*, not
                // what the model is told, and the surrounding fields are usually the context that
                // makes the fire worth reading.
                output: probe.text.clone(),
                baseline: rendered,
            })
        }
    }
}
/// What resolving a JSON pointer against a probe's result found.
///
/// Three outcomes, not two, because the two failures mean opposite things. A document that parsed
/// and simply lacks the field is an *answer*: an API that omits `chats` when there are none is
/// saying there are none. A result that is not a JSON document at all is a *broken probe*: nothing
/// was measured, so there is nothing to conclude.
///
/// Collapsing them cost real turns. `{"at": "/chats", "is": "empty"}` reads a missing value as
/// empty and fires, so a server that started returning an error string or prose fired the job on
/// every interval, indefinitely -- the exact expense the pointer predicate exists to avoid, aimed
/// the other way. `not-empty` and `changed` fail toward silence, which is survivable; `empty` was
/// alone in failing toward spending.
pub(crate) enum Pointed {
    /// The document parsed and the pointer resolved to this.
    Found(serde_json::Value),
    /// The document parsed; the pointer names nothing in it.
    Absent,
    /// The probe's result is not a JSON document, so the pointer means nothing here.
    NotADocument,
}
/// Resolve a JSON pointer against a probe's result.
///
/// Prefers the structured value, and falls back to parsing the text. The fallback is what makes a
/// pointer usable against the many MCP servers that return JSON as their text content and set no
/// `structuredContent`; there is no fence in that case, so the parse is unambiguous.
pub(crate) fn pointed_at(probe: &ProbeOutcome, pointer: &str) -> Pointed {
    let document = match &probe.structured {
        Some(structured) => std::borrow::Cow::Borrowed(structured),
        None => match serde_json::from_str::<serde_json::Value>(probe.text.trim()) {
            Ok(parsed) => std::borrow::Cow::Owned(parsed),
            Err(_) => return Pointed::NotADocument,
        },
    };
    match document.pointer(pointer) {
        Some(value) => Pointed::Found(value.clone()),
        None => Pointed::Absent,
    }
}
/// The same value with every object's keys in a fixed order, so equal documents render alike.
///
/// See the call site: `preserve_order` makes a `Value` remember its input's key order, which makes
/// a string comparison sensitive to something no watcher means to watch.
pub(crate) fn canonical_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(fields) => {
            let mut sorted: Vec<(&String, &serde_json::Value)> = fields.iter().collect();
            sorted.sort_by(|left, right| left.0.cmp(right.0));
            serde_json::Value::Object(
                sorted
                    .into_iter()
                    .map(|(key, value)| (key.clone(), canonical_json(value)))
                    .collect(),
            )
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(canonical_json).collect())
        }
        scalar => scalar.clone(),
    }
}
/// Whether a pointed-at value counts as "there is something here".
///
/// Containers go by length and scalars by presence, because those are the two ways a watched thing
/// reads as absent: an empty `chats` array, or a `null` field.
pub(crate) fn json_is_non_empty(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Array(items) => !items.is_empty(),
        serde_json::Value::Object(entries) => !entries.is_empty(),
        serde_json::Value::String(text) => !text.is_empty(),
        serde_json::Value::Bool(_) | serde_json::Value::Number(_) => true,
    }
}
/// Build the platform's shell invocation for a gate, mirroring what `execute_command` does on its
/// unsandboxed path (`crate::tools::shell`).
pub(crate) fn gate_command_builder(command: &str) -> tokio::process::Command {
    #[cfg(windows)]
    {
        // Same UTF-8 prelude the shell tool uses: PowerShell 5.1 otherwise emits the legacy console
        // code page and non-ASCII output comes back as `?`, which would make a `changed` gate
        // flap between encodings rather than on the thing it watches.
        let wrapped = crate::sandbox::wrap_command_with_utf8_output(command);
        let mut builder = tokio::process::Command::new("powershell.exe");
        builder
            .arg("-NoProfile")
            .arg("-NonInteractive")
            .arg("-Command")
            .arg(&wrapped);
        builder
    }
    #[cfg(not(windows))]
    {
        let mut builder = tokio::process::Command::new("sh");
        builder.arg("-c").arg(command);
        builder
    }
}
