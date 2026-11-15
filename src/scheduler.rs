//! The scheduler: the sweep that finds due schedules, claims each occurrence against every other
//! process on the same store, evaluates its gate and queues the fire in its session's inbox. What
//! a schedule *is* lives in [`crate::schedule`]; this is what *runs* one, and it is the one place
//! that knows the store's claim protocol.

use std::time::Duration;

use chrono::{DateTime, Utc};
// Reached through `humantime_serde`'s re-export rather than a direct dependency, which is also
// how `crate::config` gets at it. One duration syntax, one copy of the parser.
use humantime_serde::re::humantime;

use crate::{
    schedule::*,
    store::{
        inbox::{Fire, QueuedFire},
        schedule::{ClaimClosed, ScheduleStore},
    },
};

/// Human-readable lateness, for a fire's header and its stage direction.
pub(crate) fn format_late(late_by: chrono::Duration) -> String {
    late_by
        .to_std()
        .map(|std| humantime::format_duration(Duration::from_secs(std.as_secs())).to_string())
        .unwrap_or_else(|_| "an unknown interval".to_string())
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

/// Which schedules a scheduler instance is responsible for.
///
/// Every host answers a predicate rather than claiming a static set, because none of them can
/// actually take every schedule: `meka serve` can revive any session but not one another process
/// has locked, the screen owns exactly the conversation it has open, and ACP owns whatever the
/// editor currently has open.
///
/// Asked here rather than letting a host decline afterwards: `prepare` evaluates a schedule's gate
/// before its fire is queued, so a scope that admitted everything would run every gated schedule's
/// probe (a shell command, or a call to someone else's server) on every tick for sessions it could
/// never serve.
#[derive(Clone)]
pub(crate) enum SchedulerScope {
    /// Schedules the predicate accepts, re-asked every sweep, so a host whose set of sessions
    /// moves under it (the screen's `/resume`, ACP's open editors, serve's session locks) is never
    /// working from a snapshot.
    Schedules(std::sync::Arc<dyn Fn(&Schedule) -> bool + Send + Sync>),
}
impl std::fmt::Debug for SchedulerScope {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Schedules(_) => formatter.write_str("Schedules(<predicate>)"),
        }
    }
}
impl SchedulerScope {
    /// Every schedule in the database, whoever it belongs to. Only the tests want this: a real host
    /// always has some schedule it cannot take.
    #[cfg(test)]
    pub(crate) fn every_schedule() -> Self {
        Self::Schedules(std::sync::Arc::new(|_| true))
    }

    pub(crate) fn covers(&self, schedule: &Schedule) -> bool {
        match self {
            Self::Schedules(predicate) => predicate(schedule),
        }
    }
}
/// Start the scheduler loop. Returns the handle so the caller can abort it on shutdown; the task
/// runs until then.
///
/// Modeled on [`crate::host::http::gc::spawn`]: a tokio interval that wakes, queries, and hands
/// work to a host-supplied callback. Schedules are prepared one at a time, so a process with
/// several due runs one gate at a time; the turns the fires open are the inbox's.
pub(crate) fn spawn<Callback, Told>(
    store: std::sync::Arc<crate::store::Store>,
    config: crate::config::ResolvedScheduleConfig,
    tools: Option<std::sync::Arc<dyn GateTools>>,
    residents: std::sync::Arc<dyn ResidentPermissions>,
    scope: SchedulerScope,
    queued: Callback,
) -> tokio::task::JoinHandle<()>
where
    Callback: Fn(Schedule, QueuedFire) -> Told + Send + Sync + 'static,
    Told: std::future::Future<Output = ()> + Send,
{
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(config.poll_interval);
        // A sweep contains its gates' probes, so it can overrun `poll_interval` by a
        // `gate_timeout` for every gated schedule due. Tokio's default `Burst` then resolves
        // *every* tick missed during it, so a sweep that ran twelve periods long is followed by
        // twelve immediate sweeps, eleven of which find nothing due. `Delay` collapses that to one.
        //
        // It does not create a gap, and nothing here does: `Delay` schedules the next tick a period
        // after the miss is *recognized*, so the first `tick()` following a long sweep still
        // returns at once.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // The first tick resolves immediately; skip it so startup is not competing with provider
        // and MCP connection setup.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            // A panic must not end the loop either: a gate's probe calls into an MCP server's tool
            // and the callback is host code, and nothing joins this handle, so losing the task
            // would stop every schedule silently.
            let sweep = std::panic::AssertUnwindSafe(run_due(
                &store,
                &config,
                tools.as_deref(),
                &*residents,
                &scope,
                &queued,
            ));
            match futures::FutureExt::catch_unwind(sweep).await {
                // A failed sweep must not end the loop: a transient database error would otherwise
                // silently disable every schedule for the life of the process.
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
/// One sweep: evaluate every due schedule in scope and queue the fires that survive, telling
/// `queued` of each once its write has committed: the schedule, and where its fire is now.
///
/// The sweep awaits no turn. A fire waits in its session's inbox, where a running turn reads it at
/// its next pause and an idle session gets a turn for it from its host's driver, so a check runs
/// when it is due whatever its session is doing.
pub(crate) async fn run_due<Callback, Told>(
    store: &crate::store::Store,
    config: &crate::config::ResolvedScheduleConfig,
    tools: Option<&dyn GateTools>,
    residents: &dyn ResidentPermissions,
    scope: &SchedulerScope,
    queued: &Callback,
) -> crate::error::Result<()>
where
    Callback: Fn(Schedule, QueuedFire) -> Told,
    Told: std::future::Future<Output = ()>,
{
    let due = store
        .schedule_store()
        .list_due_schedules(Utc::now())
        .await?;
    for schedule in due {
        if !scope.covers(&schedule) {
            continue;
        }
        let short_id = schedule.short_id().to_string();
        // Per schedule, not per sweep: a sweep contains the gates it runs, so by the fifth
        // schedule the sweep's own clock can be minutes old, which would advance a recurring
        // schedule to a `next_fire_at` already in the past and stamp a lease to expire before it
        // was taken.
        let now = Utc::now();
        // `warn!`, not `?`: propagating would abort the whole sweep, so one transient `SQLITE_BUSY`
        // would skip every other schedule due in the same tick. A schedule that was never claimed
        // comes back next tick.
        match prepare(store, config, tools, residents, schedule, now).await {
            Ok(Some((schedule, fire))) => queued(schedule, fire).await,
            Ok(None) => {}
            Err(error) => tracing::warn!("failed to prepare schedule {short_id}: {error}"),
        }
    }
    Ok(())
}
/// The lease this host holds on one occurrence, and what closing it writes.
///
/// Every write that follows the claim is scoped to the claim *this process* won rather than to the
/// schedule id alone. A stale writer whose lease has since expired and been taken by someone else
/// changes nothing.
#[derive(Debug, Clone)]
pub(crate) struct Claim {
    /// Proof of ownership, matched against `schedules.claimed_by`.
    pub(crate) owner: String,
    /// Where the schedule goes once the occurrence is spent: `Some` advances a schedule that lives
    /// on, `None` retires one whose moment is spent.
    pub(crate) next_fire_at: Option<DateTime<Utc>>,
    /// What the gate saw, to be recorded when the occurrence is disposed of.
    ///
    /// Carried rather than written as soon as the gate returns, because a baseline is only true
    /// once the occurrence it belongs to is finished with: it is written with the fire it decided,
    /// in one transaction. A host that stopped between the gate and that write leaves the
    /// occurrence to the next claim with the baseline as it was, and the change it measured still
    /// fires there rather than being compared against itself.
    pub(crate) gate_baseline: Option<String>,
}
/// What the store said when a session's row was asked for.
///
/// A lookup that failed is kept apart from one that found no row, and both from one that found a
/// row: the first two decide nothing and leave the occurrence for the next sweep, while a row
/// answers for itself. Folding a failed read into "no level" would re-grant gate authority at the
/// host's level on a `SQLITE_BUSY`.
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
/// operator narrows `[permissions].enabled` and restarts: the session re-attaches clamped, and the
/// creation door returns 403 for the same authority, so an unclamped read here would keep firing a
/// gate the two doors had refused.
///
/// A lookup that failed, or found no row, answers `none` here; [`prepare`] leaves the occurrence
/// for the next sweep instead, ahead of asking. See [`SessionLookup`].
pub(crate) fn live_permission(
    lookup: SessionLookup<'_>,
    config: &crate::config::ResolvedScheduleConfig,
    session_id: uuid::Uuid,
) -> crate::permission::Permission {
    let session = match lookup {
        // Nothing is executable at `none`, which is the level that lets a schedule neither act nor
        // be woken until the row can be read again.
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
/// Decide what to do with one due schedule: retire it, reschedule it quietly, or queue its fire,
/// answering the schedule and where its fire is now when it queued one.
pub(crate) async fn prepare(
    store: &crate::store::Store,
    config: &crate::config::ResolvedScheduleConfig,
    tools: Option<&dyn GateTools>,
    residents: &dyn ResidentPermissions,
    schedule: Schedule,
    now: DateTime<Utc>,
) -> crate::error::Result<Option<(Schedule, QueuedFire)>> {
    let schedule_store = store.schedule_store();
    let memory = store.scheduler_memory();
    let late_by = now - schedule.next_fire_at;
    let recurring = schedule.timing.is_recurring();
    let short_id = schedule.short_id();
    // A one-shot far past its moment is noise rather than a reminder: "join the standup" delivered
    // five days late helps nobody. Recurring schedules need no equivalent rule, since their
    // occurrences are one period apart and the most recent missed one is always less than a period
    // old.
    if !recurring
        && chrono::Duration::from_std(config.missed_grace)
            .map(|grace| late_by > grace)
            .unwrap_or(false)
    {
        // Claimed rather than deleted outright, so the announcement is made once by whichever host
        // actually removed the row rather than once per host that had it in its due list.
        if schedule_store
            .retire_unclaimed(&schedule.id, schedule.next_fire_at, now)
            .await?
        {
            // One of the two doors out of the table that do not go through `delete_schedule`;
            // the other is a claim's close retiring a schedule whose moment is spent, which is
            // reached only after the clears below have already run. Without these the ledgers keyed
            // by schedule id outlive the schedule, which a held one-shot reaches routinely: it
            // survives every sweep until its grace window closes, and this is where it ends.
            memory.forget(&schedule.id);
            let late = format_late(late_by);
            tracing::warn!(
                "dropping one-shot schedule {short_id}: due {late} ago, past `[schedule].missed_grace`"
            );
        }
        return Ok(None);
    }

    // One lookup, for every schedule rather than only gated ones, serving the working directory a
    // gate runs in and the live level both the checks below need.
    //
    // Not skipped for an ungated schedule: at `none` such a schedule would keep firing, waking a
    // model that can read nothing and act on nothing. The query costs one row read per schedule
    // actually due, against a model turn.
    let looked_up = store.session_info(schedule.session_id).await;
    let (session, lookup) = match &looked_up {
        Ok(info) => (info.as_ref(), SessionLookup::Read(info.as_ref())),
        Err(error) => {
            // Not silently "no session". A failed lookup means the level cannot be confirmed, and
            // `live_permission` fails closed on it rather than falling back to the host's level.
            let session_id = schedule.session_id;
            tracing::warn!(
                "failed to read session {session_id} while preparing schedule {short_id}: {error}"
            );
            (None, SessionLookup::Failed)
        }
    };
    let gate_cwd = session.and_then(|info| info.cwd.clone());
    // A row that could not be read, or is gone, decides nothing, and the occurrence stays for the
    // next sweep, exactly as it does when a gate tool is not yet resolvable. Falling through to the
    // permission branch would decline the schedule at `none` on the strength of one `SQLITE_BUSY`.
    if matches!(lookup, SessionLookup::Failed | SessionLookup::Read(None)) {
        return Ok(None);
    }

    // What the session's permission is now, as opposed to what it was when the gate was authored.
    // A session this process holds answers from its cell, which is what the user's Shift+Tab moved
    // and what its own turns dispatch against; the row it writes back can lag on a failed write.
    // Every other session answers from its row, filtered by the enabled set; see `live_permission`
    // for why nothing else does.
    let live_permission = match residents.live_permission_of(schedule.session_id).await {
        Some(level) => level,
        None => live_permission(lookup, config, schedule.session_id),
    };

    let coalesced = occurrences_between(&schedule.timing, schedule.next_fire_at, now);
    // `Some` only for a schedule that lives on. A one-shot's moment is spent, and a cron pattern
    // with nothing left in range has no future to be scheduled for; both are retired when the
    // claim is closed.
    //
    // Derived before the claim because a refusal needs it too: it is what spends a recurring
    // schedule's occurrence without evaluating anything.
    let next_fire_at = schedule
        .timing
        .next_after_delivering(schedule.next_fire_at, now)
        .filter(|_| recurring);

    // Spend the occurrence of a schedule refused before its claim was taken.
    //
    // A recurring schedule advances, which is the documented rule and the same thing a gate that
    // ran and said no does: without it a held schedule sits permanently due and then reports a
    // month-long backlog the moment it is authorized again.
    //
    // A one-shot is left completely alone. Every refusal that needs nothing but a row read is made
    // here because there is no reason to pay for a lease to reach the same answer.
    async fn decline_before_claiming(
        store: &ScheduleStore,
        schedule: &Schedule,
        now: DateTime<Utc>,
        next_fire_at: Option<DateTime<Utc>>,
    ) -> crate::error::Result<()> {
        // Whether *this* host won the advance does not matter: either way the occurrence has moved
        // off the value every host read, and no host will deliver it.
        if let Some(next) = next_fire_at {
            store
                .advance_unclaimed(&schedule.id, schedule.next_fire_at, now, next)
                .await?;
        }
        Ok(())
    }

    // A schedule whose claims keep ending without being closed is parked rather than retried
    // again.
    //
    // Claiming is a lease, so nothing else stops an occurrence that reliably takes its host down
    // before the fire is written from being picked up on every expiry, forever. The row stays
    // exactly where it is: listed, cancelable, and reported as held on every surface, because
    // destroying a user's schedule over a failure Meka cannot diagnose would be worse than leaving
    // it visible and inert.
    if schedule.attempts >= MAX_CLAIM_ATTEMPTS {
        if memory.declined_for_permission_first_time(&schedule.id, "crashed") {
            let reason = schedule_withheld_reason(memory, &schedule, live_permission, tools)
                .unwrap_or_else(|| "it has been parked".to_string());
            tracing::warn!("schedule {short_id} not fired: {reason}");
        }
        return Ok(None);
    }

    // Before the gate, and regardless of whether there is one. At `none` nothing the turn could
    // reach is dispatchable: it reads nothing, changes nothing, and `schedule_cancel` is refused
    // too, so it cannot stop itself being woken again. Firing anyway spends tokens on that, every
    // interval, until an operator notices.
    if !live_permission.allows_unattended_work() {
        if memory.declined_for_permission_first_time(
            &schedule.id,
            &format!("unattended-work:{live_permission}"),
        ) {
            tracing::warn!(
                "schedule {short_id} not fired: the session is at `{live_permission}`, where no tool is \
                 executable; raise the session to restore it"
            );
        } else {
            tracing::debug!(
                "schedule {short_id} still not fired: the session remains at {live_permission}"
            );
        }
        decline_before_claiming(&schedule_store, &schedule, now, next_fire_at).await?;
        return Ok(None);
    }

    // The same predicate the two creation doors use, asked again here against both the recorded
    // level and the live one. Checking only the recorded value would be a tautology: the row is
    // written by a door that already demanded the level, and nothing ever updates the column. The
    // live level is what makes a withdrawal real; the recorded one still matters because a
    // hand-edited or unparseable `gate_permission` decodes as `Permission::None` and must stay
    // refused. Going through `gate_probe_is_authorized` rather than re-deriving the rule is what
    // keeps the doors in agreement.
    //
    // The occurrence is declined, exactly as a gate that ran and said no is declined: a gate that
    // could not be evaluated has not passed, and firing anyway would convert a conditional schedule
    // into an unconditional one, a turn a minute for `every = "1m"` with a `changed` gate.
    if let Some(gate) = &schedule.gate
        && let Some((refusal, level)) = gate_withheld_reason(gate, live_permission, tools)
    {
        // Not decidable yet is not refused: a server mid-handshake, or a host with no dispatcher
        // for this tool at all, cannot say whether the gate passes, and declining would spend the
        // occurrence (a `6h` schedule due at startup advanced six hours without its probe running).
        // The same rule `schedule_withheld` applies before it reports.
        if matches!(refusal, GateRefusal::ToolUnavailable)
            && let GateProbe::Tool { name, .. } = &gate.probe
            && tools.is_none_or(|tools| tools.is_still_connecting(name))
        {
            tracing::debug!(
                "schedule {short_id} not decided: gate tool '{name}' is not resolvable yet"
            );
            return Ok(None);
        }
        // Said once per decline, not once per evaluation: the condition is a standing state rather
        // than an event, and an `every = "1m"` schedule would otherwise write this line every
        // minute. The id is cleared the moment the gate is authorized again, so a later withdrawal
        // is announced afresh.
        let explained = refusal.explain(&gate.probe, level);
        if memory.declined_for_permission_first_time(&schedule.id, &explained) {
            // `gate_withheld_reason` reports the live level first because it is the one an
            // operator can act on, so a refusal carrying the recorded level instead means the live
            // level was fine: a hand-edited or damaged row, which no amount of cycling the session
            // will fix.
            if level == live_permission {
                // Naming the level it was authorized at only helps when that level would still
                // pass; otherwise it reads as a promise that restoring it is enough.
                match gate_probe_is_authorized(&gate.probe, gate.permission, tools) {
                    Ok(()) => {
                        let recorded = gate.permission;
                        tracing::warn!(
                            "schedule {short_id} not fired: {explained}; raise the session back to \
                             `{recorded}` to restore it"
                        );
                    }
                    Err(_) => {
                        tracing::warn!("schedule {short_id} not fired: {explained}")
                    }
                }
            } else {
                tracing::warn!(
                    "schedule {short_id} not fired: {explained}, which is the level recorded when the gate was \
                     authorized"
                );
            }
        } else {
            tracing::debug!("schedule {short_id} still not fired: its gate is still unauthorized");
        }
        decline_before_claiming(&schedule_store, &schedule, now, next_fire_at).await?;
        return Ok(None);
    }
    // Nothing is holding this schedule back, so a later withdrawal is announced afresh rather than
    // swallowed by the once-per-decline suppression above. An ungated schedule passes through here
    // too: it is held by nothing, and the `none` floor above is the only thing that could have
    // stopped it.
    memory.clear_permission_decline(&schedule.id);

    // Lease the occurrence before doing anything that can fail or hang.
    //
    // This is what arbitrates between hosts, which is why it is conditional. Every `meka serve`,
    // screen and ACP session polls the same table, so one occurrence is in several hosts' due lists
    // at once; whoever takes the lease owns it, and the rest return here having neither evaluated
    // the gate nor spent the occurrence.
    //
    // The lease is taken *before* the gate runs and closed after, so the row is never absent
    // while this host is working: a refusal has nothing to put back, and a crash between the claim
    // and the fire's write expires rather than spending the occurrence.
    let mut claim = Claim {
        owner: memory.owner().to_string(),
        next_fire_at,
        gate_baseline: None,
    };
    if !schedule_store
        .claim_occurrence(
            &schedule.id,
            schedule.next_fire_at,
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
        tracing::debug!("schedule {short_id} was claimed for this occurrence by another host");
        return Ok(None);
    }

    let gate_output = match &schedule.gate {
        None => None,
        Some(gate) => {
            // A sweep stopped while the gate runs leaves the schedule noted as running, which is
            // how a host on its way out tells the lease this gate holds from one left to run out on
            // purpose.
            memory.gate_started(&schedule.id);
            let evaluated = evaluate_gate(
                gate,
                config.gate_timeout,
                gate_cwd.as_deref(),
                tools,
                Some(schedule.session_id),
            )
            .await;
            memory.gate_ended(&schedule.id);
            match evaluated {
                Ok(outcome) => {
                    // An evaluation that produced an answer, whichever answer it was, ends any
                    // standing failure: the probe works.
                    memory.clear_probe_failure(&schedule.id);
                    // Persist the new baseline even when it did not fire; that is exactly how a
                    // `changed` gate stops firing once it has seen the new value.
                    //
                    // `baseline`, not `output`: for a pointer predicate the two differ, and storing
                    // the whole result would put the moving field the pointer excludes back into
                    // the comparison, firing the gate every interval.
                    claim.gate_baseline = Some(outcome.baseline);
                    if !outcome.fired {
                        // The occurrence is spent: the condition was asked and said no. A recurring
                        // schedule moves to its next occurrence and a one-shot's moment has passed,
                        // which is the documented rule and the same thing a fire's write does. The
                        // only difference is that nothing is queued.
                        tracing::debug!("gate for schedule {short_id} declined to fire");
                        match schedule_store
                            .complete_claim(
                                &schedule.id,
                                &claim.owner,
                                claim.next_fire_at,
                                claim.gate_baseline.as_deref(),
                            )
                            .await
                        {
                            Ok(ClaimClosed::Yes(())) => {}
                            Ok(ClaimClosed::RowGone) => tracing::debug!(
                                "schedule {short_id}'s gate declined and its row was removed while the \
                                 probe ran"
                            ),
                            // The lease was taken from under this host while the probe ran, so the
                            // baseline it just measured was not recorded. Said out loud because
                            // the visible symptom is a `changed` gate firing twice for one change,
                            // which reads as a flapping probe rather than a lost write.
                            Ok(ClaimClosed::LeaseLost) => tracing::warn!(
                                "schedule {short_id}'s gate declined after its lease expired; the \
                                 occurrence may be evaluated again"
                            ),
                            Err(error) => tracing::warn!(
                                "schedule {short_id} declined but failed to close its occurrence: {error}"
                            ),
                        }
                        return Ok(None);
                    }
                    Some(outcome.output)
                }
                Err(error) => {
                    // Loud on purpose: a watcher whose probe breaks produces the same silence as a
                    // watcher with nothing to report, and that is the failure most likely to go
                    // unnoticed for weeks. Counted as well as logged, so the model hears about it
                    // too once the condition is standing, since a schedule exists precisely
                    // because nobody is reading the log. The row as it stands now, before this
                    // arm's own disposal, which writes neither half of the witness; see
                    // [`standing_probe_failure`].
                    let failures = memory.record_probe_failure(&schedule, &error);
                    tracing::warn!(
                        "gate for schedule {short_id} failed: {error} (failure {failures})"
                    );
                    // The same disposal a refusal gets, and for the same reason: the condition was
                    // not answered, so this occurrence is over. A recurring schedule advances to
                    // its next one; releasing the lease instead would leave the row due on the very
                    // next sweep, so a six-hour schedule whose server is down would be re-probed
                    // every `poll_interval`, and a probe that hangs would burn the whole
                    // `gate_timeout` out of each sweep.
                    //
                    // No baseline: nothing was measured, so the next successful evaluation compares
                    // against the last value actually observed and reports the change that
                    // happened while the probe was broken.
                    match claim.next_fire_at {
                        Some(next) => match schedule_store
                            .complete_claim(&schedule.id, &claim.owner, Some(next), None)
                            .await
                        {
                            Ok(ClaimClosed::Yes(())) => {}
                            Ok(ClaimClosed::RowGone) => tracing::debug!(
                                "schedule {short_id}'s gate failed and its row was removed while the \
                                 probe ran"
                            ),
                            // Said out loud because it is the state this whole arm exists to
                            // avoid, and it is otherwise indistinguishable from a probe that is
                            // simply failing often.
                            Ok(ClaimClosed::LeaseLost) => tracing::warn!(
                                "schedule {short_id}'s gate failed after its lease expired; it is probed \
                                 again on the next sweep"
                            ),
                            Err(error) => tracing::warn!(
                                "schedule {short_id}'s gate failed and then failed to close its \
                                 occurrence: {error}"
                            ),
                        },
                        // The lease is kept and left to expire. A one-shot has no next occurrence
                        // to advance to, and releasing the lease would make the row due on the
                        // very next sweep, so an MCP server restarting anywhere near its due time
                        // would park it after three claims. Holding the lease is the backoff: the
                        // schedule is retried once per `claim_lease`, and the attempt ceiling still
                        // applies to those retries.
                        None => tracing::debug!(
                            "schedule {short_id}'s gate failed; it is retried after \
                             `[schedule].claim_lease`"
                        ),
                    }
                    return Ok(None);
                }
            }
        }
    };

    // The occurrence is spent and the fire queued in one write, scoped to this host's lease.
    let fire = Fire {
        schedule_id: schedule.id.clone(),
        gate_output,
        occurrences: coalesced.saturating_add(1),
        due_at: schedule.next_fire_at,
        fired_at: now,
    };
    match schedule_store
        .queue_fire(
            &schedule,
            &claim.owner,
            claim.next_fire_at,
            claim.gate_baseline.as_deref(),
            fire,
        )
        .await
    {
        Ok(ClaimClosed::Yes(queued)) => {
            let timing = schedule.timing.describe();
            tracing::info!("queued schedule {short_id}'s fire ({timing})");
            Ok(Some((schedule, queued)))
        }
        // A cancellation wins: the schedule's row went while its gate ran, and nothing is queued
        // for a schedule that no longer exists.
        Ok(ClaimClosed::RowGone) => {
            tracing::debug!(
                "schedule {short_id} was canceled while its gate ran; nothing is queued"
            );
            Ok(None)
        }
        // Its own sentence because the remedy differs from the `Err` below: an error is a database
        // problem, this is a `claim_lease` shorter than the gate took.
        Ok(ClaimClosed::LeaseLost) => {
            tracing::warn!(
                "schedule {short_id}'s lease expired before its fire was queued, so it is left to \
                 the host holding it now; raise `[schedule].claim_lease` past the gate's run"
            );
            Ok(None)
        }
        Err(error) => {
            tracing::warn!(
                "failed to queue schedule {short_id}'s fire: {error}; it is tried again after \
                 `[schedule].claim_lease`"
            );
            Ok(None)
        }
    }
}

/// Run a gate and decide whether the schedule it guards should fire.
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
                // Not a misconfiguration to report at creation: the host that authored the schedule
                // can dispatch tools, and this is a *different* host picking the row up. Declining
                // is the same answer a disconnected server gets, for the same reason.
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
/// Authoring one requires `unrestricted`, which is the same level at which `shell_execute` runs
/// arbitrary unsandboxed commands, so a sandbox here would block the ordinary cases (`gh`, `curl`)
/// without raising the bar the agent must clear.
pub(crate) async fn run_shell_probe(
    command: &str,
    timeout: Duration,
    cwd: Option<&std::path::Path>,
) -> Result<ProbeOutcome, String> {
    use crate::process::{Capture, Keep, OutputBudget};

    /// How long a probe's output is waited for once its command has ended: a process it started
    /// and left behind can hold the pipe open, and the probe is not held for it.
    const GATE_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

    let mut builder = gate_command_builder(command);
    // The creating session's directory, not the host process's. A gate is almost always written by
    // the model right after verifying the same command through `shell_execute`, which runs in the
    // session's cwd, so a gate that runs anywhere else silently stops matching the command the
    // model tested; under a `meka serve` systemd unit the process cwd is `/`.
    if let Some(directory) = cwd {
        if directory.is_dir() {
            builder.current_dir(directory);
        } else {
            let directory = directory.display();
            tracing::warn!(
                "gate's session directory '{directory}' no longer exists; running in the current \
                 directory"
            );
        }
    }
    builder
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    // A tree of its own, so a timeout, the parse limit or a dropped evaluation ends the whole
    // pipeline (`gh … | jq …`) rather than the shell alone.
    let child = crate::process::spawn_tree(&mut builder)
        .map_err(|error| format!("failed to start gate: {error}"))?;

    // Stdout is spent from the parse limit, never held whole: `wait_with_output` would hold
    // everything a runaway `yes` produced until the timeout. Past the limit the probe is stopped,
    // since nothing after it can be parsed anyway. Stderr keeps its head and is read to its end: a
    // read end closed at the cap would make the child's next write to stderr `SIGPIPE`, so a gate
    // that logs more than the cap would die before it could print its answer.
    let capture = Capture {
        stdout: Keep::Budgeted(std::sync::Arc::new(OutputBudget::new(GATE_PARSE_BYTES))),
        stderr: Keep::Head(GATE_OUTPUT_BYTES),
        sink: None,
    };
    let finished = crate::process::run(child, capture, Some(timeout), None, GATE_DRAIN_TIMEOUT)
        .await
        .map_err(|error| format!("gate failed to run: {error}"))?;
    shell_probe_outcome(finished, timeout)
}
/// What a shell probe's run came to, for a predicate to judge: its output up to the parse limit
/// and whether it succeeded, or the refusal of a run that outlasted `timeout`.
///
/// Apart from the run, so what each ending means is testable without a process.
pub(crate) fn shell_probe_outcome(
    finished: crate::process::Finished,
    timeout: Duration,
) -> Result<ProbeOutcome, String> {
    use crate::process::Ended;

    let status = match finished.ended {
        Ended::Exited(status) => Some(status),
        Ended::OutputBound => None,
        Ended::TimedOut | Ended::Stopped => {
            return Err(format!(
                "gate exceeded its {} budget",
                humantime::format_duration(timeout)
            ));
        }
    };
    let mut stdout_bytes = finished.drained.stdout.bytes;
    let stderr_bytes = finished.drained.stderr.bytes;
    // Past the limit the output is cut off and the probe has not succeeded, whichever the run saw
    // first: a command that fits its last write in the pipe exits before the drain crosses the
    // limit, and one that does not is stopped at it, and which of the two happened is a race the
    // predicate must not be decided by.
    let cut_off = stdout_bytes.len() > GATE_PARSE_BYTES;
    if cut_off {
        stdout_bytes.truncate(GATE_PARSE_BYTES);
        tracing::debug!(
            "gate output exceeded {GATE_PARSE_BYTES} bytes and was cut off; the command was stopped"
        );
    }

    let stdout = String::from_utf8_lossy(&stdout_bytes);

    // A non-zero exit is reported, not refused.
    //
    // The failure this exists for is a watcher that breaks silently: an expired token has `gh` exit
    // non-zero with empty stdout, the first evaluation stores `""` as the baseline, and every
    // evaluation after compares `"" == ""` and stays quiet forever. `debug!`, not `warn!`, because
    // for a large class of correct gates a non-zero exit is the signal (`diff -q`, `git diff
    // --exit-code`, `grep ERROR log`, `curl -f`), and refusing to produce output would silence them
    // permanently. The `succeeded` flag carries the exit status to whichever predicate asked.
    let succeeded = !cut_off && status.is_some_and(|status| status.success());
    if let Some(status) = status
        && !succeeded
    {
        let stderr = truncate_gate_output(&String::from_utf8_lossy(&stderr_bytes));
        let detail = if stderr.is_empty() {
            String::new()
        } else {
            format!(": {stderr}")
        };
        tracing::debug!("gate command exited with {status}{detail}; its output still stands");
    }

    Ok(ProbeOutcome::new(&stdout, None, succeeded))
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
            // warning says which schedule to fix.
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
                // caller declines the occurrence and warns, naming the schedule.
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
/// Collapsing them costs real turns: `{"at": "/chats", "is": "empty"}` reads a missing value as
/// empty and fires, so a server that started returning an error string or prose would fire the
/// schedule on every interval. `not_empty` and `changed` fail toward silence, which is survivable;
/// `empty` alone fails toward spending.
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
/// Build the platform's shell invocation for a gate, mirroring what `shell_execute` does on its
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
