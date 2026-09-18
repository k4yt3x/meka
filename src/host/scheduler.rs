//! What every host does when the scheduler fires a job or the background poller finds finished
//! work: find the session, take its runtime, publish a token, fold undelivered outcomes into the
//! prompt, run the turn, and report. One driver, so the hosts cannot disagree on whether an ungated
//! job waits behind a turn, whether a canceled fire counts as failed, or who announces outcomes and
//! when; what differs is asked through [`HostHooks`].

use std::ops::ControlFlow;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::{
    background::{only_what_was_won, render_outcomes},
    error::MekaError,
    host::{ResidentSession, Sessions, claim_undelivered_outcomes},
    schedule::ScheduledJob,
    scheduler::{FireOutcome, Wakeup},
    store::{Store, background::BackgroundTask},
};

/// Who started a turn the host did not receive over its own door, for the host's own accounting:
/// `serve` names it on the session feed's `turn.started`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TurnOrigin {
    Schedule { job_id: String },
    Background,
    Inbox { item_ids: Vec<uuid::Uuid> },
}

/// First wait before a turn that failed on inbox items is tried again; doubles per attempt.
/// Also the wait for a session another process holds, asked again at this pace until it lets go.
const INBOX_RETRY_BASE: std::time::Duration = std::time::Duration::from_secs(10);
/// Longest wait between two attempts, and the wait for a session that could not be brought up
/// at all, which no backoff from a first try would fit.
const INBOX_RETRY_LONGEST_WAIT: std::time::Duration = std::time::Duration::from_secs(5 * 60);
/// How long an item is retried for before it is given up on, from when it was enqueued. Long
/// enough to ride out a provider blip; short enough that an outage is reported to whoever is
/// waiting rather than left to look like silence.
const INBOX_RETRY_CEILING: chrono::Duration = chrono::Duration::hours(1);

/// A prompt the user did not type, shown where the host shows such things before it runs.
pub(crate) enum OutOfBandPrompt<'a> {
    /// A batch of finished background work, rendered.
    Outcomes(&'a str),
    /// A scheduled job about to fire.
    Scheduled(&'a Wakeup),
}

/// What differs between hosts around a scheduled or outcome-driven turn.
///
/// A host also answers for the live level of every session it holds, through
/// [`crate::scheduler::ResidentPermissions`], so the fire door reads a resident session's cell
/// rather than the row that cell may not have reached yet.
#[async_trait]
pub(crate) trait HostHooks: crate::scheduler::ResidentPermissions {
    type Entry: std::ops::Deref<Target = ResidentSession> + Clone + Send + Sync + 'static;

    /// The resident session a job or outcome belongs to. `Ok(None)` when this host cannot run it
    /// now, which defers the fire: ACP serves only sessions the editor has open, and `serve`
    /// declines a session another process holds.
    async fn resident(&self, session_id: uuid::Uuid) -> anyhow::Result<Option<Self::Entry>>;

    /// Whether `entry` is still the session this host holds under its id, asked once the
    /// conversation lock has been won. A fire queued on that lock can outlive the session: ACP's
    /// `session/close` and `serve`'s idle eviction remove the entry and wait on or skip the same
    /// lock, and the fire then ran a whole turn against a session the editor had closed, pushing
    /// its transcript to a session id the client no longer had. Hosts with a map answer from
    /// the map; the REPL's one session is never released while it runs.
    async fn still_resident(&self, _entry: &Self::Entry) -> bool {
        true
    }

    /// Admit an out-of-band turn on `entry`. Counted on the process where the host caps
    /// concurrency, so `serve`'s `max_concurrent_turns` covers every turn and not only the ones a
    /// client submits; the default counts on the session alone. A door refused steps back ahead
    /// of anything it would have claimed and leaves the work for the next tick or the next turn's
    /// end, since unlike a client it has nobody to answer 429 to.
    fn admit(
        &self,
        entry: &Self::Entry,
    ) -> Result<crate::host::TurnGuard, crate::host::TurnRefused> {
        entry.admit_turn(None)
    }

    /// What the host does with the session before a turn, under the conversation lock. ACP moves
    /// the agent onto the profile the row records; an error here makes the fire unrunnable rather
    /// than failed.
    async fn prepare(&self, _entry: &Self::Entry) -> anyhow::Result<()> {
        Ok(())
    }

    /// Announce finished tasks where the host announces them, and say whether they may be
    /// delivered into a turn. `serve` posts webhooks and lets a delivery webhook veto; the others
    /// have nothing to announce to.
    async fn announce(&self, _tasks: &[BackgroundTask]) -> bool {
        true
    }

    /// Show a prompt the user did not type. ACP pushes it to the editor as a user-message chunk so
    /// the transcript shows what the turn answered; the others show nothing.
    fn show_prompt(&self, _entry: &Self::Entry, _prompt: OutOfBandPrompt<'_>) {}

    /// The token an out-of-band turn runs under. `serve` derives it from shutdown so a stopping
    /// server ends the turn.
    fn cancellation(&self) -> CancellationToken {
        CancellationToken::new()
    }

    /// An out-of-band turn is about to run under `turn_id`. `serve` opens it on the session feed;
    /// the others have no feed to open it on.
    fn begin_turn(&self, _entry: &Self::Entry, _turn_id: uuid::Uuid, _origin: TurnOrigin) {}

    /// An out-of-band turn has ended, before [`Self::finished`] announces it. `serve` records the
    /// terminal on the feed, closes the turn there, and posts the turn webhooks an inbox turn
    /// alone has nothing else to carry.
    async fn turn_closed(
        &self,
        _entry: &Self::Entry,
        _turn_id: uuid::Uuid,
        _origin: &TurnOrigin,
        _outcome: &Result<crate::agent::TurnOutcome, MekaError>,
    ) {
    }

    /// An inbox item was given up on and withdrawn. `serve` tells the feed and the webhooks.
    async fn inbox_given_up(
        &self,
        _entry: &Self::Entry,
        _item: &crate::store::inbox::InboxItem,
        _reason: &str,
    ) {
    }

    /// An inbox item was withdrawn because the turn opened on it was canceled. `serve` tells the
    /// feed, as it does for a withdrawal a client asked for.
    async fn inbox_withdrawn(&self, _entry: &Self::Entry, _item_id: uuid::Uuid) {}

    fn shutting_down(&self) -> bool {
        false
    }

    /// After an out-of-band turn, whichever way it went. `job` is `None` for an outcome delivery.
    async fn finished(
        &self,
        _entry: &Self::Entry,
        _job: Option<&ScheduledJob>,
        _outcome: &Result<(), MekaError>,
    ) {
    }

    /// A job whose session could not be reached at all. The occurrence is not spent: the claim is
    /// left to expire and retried, as an unresolvable profile is, since the cause is as likely a
    /// transient store failure as anything permanent.
    fn failed_before_running(&self, _job: &ScheduledJob, _error: &anyhow::Error) {}

    fn background_enabled(&self) -> bool;

    fn store(&self) -> &Store;
}

/// Fire one scheduled job in its session.
///
/// A gated job waits for the runtime: its probe has just said "now", and the wait is what makes a
/// gate on a busy session mean anything. An ungated one does not queue behind a turn; it defers to
/// the next tick, so a long turn cannot pile fires up behind it. A fire the process is shutting
/// down during is deferred whatever it did, so the occurrence runs again on the next start.
#[allow(
    clippy::significant_drop_tightening,
    reason = "the conversation guard is the fire's exclusivity and lives to the end on purpose"
)]
pub(crate) async fn run_wakeup<H: HostHooks>(hooks: &H, wakeup: Wakeup) -> FireOutcome {
    let job = &wakeup.job;
    let job_id = job.short_id().to_string();
    let entry = match hooks.resident(job.session_id).await {
        Ok(Some(entry)) => entry,
        Ok(None) => {
            tracing::debug!(
                "scheduled job {job_id} belongs to a session this process cannot run now; deferring"
            );
            return FireOutcome::Deferred;
        }
        Err(error) => {
            // Unrunnable rather than run: `Ran` completed the occurrence, which for a one-shot
            // deleted the job, on what may have been one `SQLITE_BUSY` from a sibling process.
            tracing::warn!(
                "scheduled job {job_id} did not run: failed to reach its session: {error}"
            );
            hooks.failed_before_running(job, &error);
            return FireOutcome::Unrunnable;
        }
    };
    entry.touch();
    let mut conversation = if job.gate.is_some() {
        entry.conversation.lock().await
    } else {
        match entry.conversation.try_lock() {
            Ok(conversation) => conversation,
            Err(_) => {
                tracing::debug!("scheduled job {job_id} found its session mid-turn; deferring");
                return FireOutcome::Deferred;
            }
        }
    };
    if !hooks.still_resident(&entry).await {
        tracing::debug!(
            "scheduled job {job_id} won its session's lock after the session was closed; deferring"
        );
        return FireOutcome::Deferred;
    }
    let busy = match hooks.admit(&entry) {
        Ok(busy) => busy,
        Err(refused) => {
            tracing::debug!(
                "scheduled job {job_id} waits: the process is at its cap of {cap} concurrent \
                 turns; deferring",
                cap = refused.cap
            );
            return FireOutcome::Deferred;
        }
    };
    if let Err(error) = hooks.prepare(&entry).await {
        tracing::warn!(
            "scheduled job {job_id} did not run: its session's profile did not resolve \
             ({error}); move it with `meka -r <id> --profile <name>`"
        );
        return FireOutcome::Unrunnable;
    }
    let cancellation = hooks.cancellation();
    let turn_id = uuid::Uuid::new_v4();
    let _published = entry
        .cancel
        .publish_turn(cancellation.clone(), busy.admission, turn_id);

    // Admitted before any outcome is claimed. A claim is one-way, so a prompt refused after it
    // would leave the batch stamped delivered and never handed out again.
    let input = match crate::agent::TurnInput::from_parts(wakeup.render_prompt(), Vec::new()) {
        Ok(input) => input.retaining(job.prompt_retention()),
        Err(empty) => {
            tracing::warn!("scheduled job {job_id} rendered no prompt: {empty}");
            return FireOutcome::Unrunnable;
        }
    };

    // An outcome that did not warrant a turn of its own rides on this one, ahead of the job's
    // prompt rather than as a message of its own: see `background::render_outcomes_riding`.
    let riding = if hooks.background_enabled() {
        claim_undelivered_outcomes(&entry.agent, hooks.store(), job.session_id).await
    } else {
        Vec::new()
    };
    if !riding.is_empty() {
        // The batch is already claimed, so a veto here cannot hold it back; it is delivered into
        // this turn regardless and the refusal is logged, as `submit_turn` logs its own.
        if !hooks.announce(&riding).await {
            tracing::warn!(
                "scheduled job {job_id} carries {outcomes} background outcome(s) a webhook \
                 rejected; delivering them anyway, since they are claimed",
                outcomes = riding.len()
            );
        }
        hooks.show_prompt(&entry, OutOfBandPrompt::Outcomes(&render_outcomes(&riding)));
    }
    hooks.show_prompt(&entry, OutOfBandPrompt::Scheduled(&wakeup));
    let input = input.riding(riding);

    let origin = TurnOrigin::Schedule {
        job_id: job.id.clone(),
    };
    hooks.begin_turn(&entry, turn_id, origin.clone());
    let outcome = entry
        .agent
        .run_turn(&mut conversation, input, cancellation)
        .await;
    entry.touch();
    hooks.turn_closed(&entry, turn_id, &origin, &outcome).await;
    let outcome = outcome.map(|_| ());
    // Before `finished`, not after: a fire the drain cut short is deferred and fires again on the
    // next start, so telling a webhook it failed announced one occurrence twice.
    if hooks.shutting_down() {
        tracing::info!("scheduled job {job_id} fired during shutdown; deferring the occurrence");
        return FireOutcome::Deferred;
    }
    hooks.finished(&entry, Some(job), &outcome).await;
    match &outcome {
        Ok(()) => tracing::info!("scheduled job {job_id} completed"),
        Err(error) => tracing::warn!("scheduled job {job_id} failed: {error}"),
    }
    FireOutcome::Ran
}

/// Deliver finished background work to every resident session that can take a turn for it.
///
/// Only outcomes that wake a host earn a turn of their own; the rest wait to ride the next turn.
/// A session mid-turn is skipped rather than waited for, so one long turn cannot hold every other
/// session's report. Claiming happens only after the host has agreed to deliver and the session
/// has been found able to carry a turn, because a claim is one-way: an outcome stamped delivered
/// and then not delivered is a report nobody ever sees.
pub(crate) async fn deliver_ready_outcomes<H: HostHooks, K>(
    hooks: &H,
    sessions: &Sessions<K, H::Entry>,
) -> ControlFlow<()>
where
    K: Eq + std::hash::Hash + Clone,
{
    let resident: Vec<H::Entry> = sessions.read().await.values().cloned().collect();
    for entry in resident {
        if hooks.shutting_down() {
            return ControlFlow::Break(());
        }
        let session_id = entry.id;
        let store = hooks.store().background_store();
        let unannounced = match store.list_unannounced_background_tasks(session_id).await {
            Ok(unannounced) => unannounced,
            Err(error) => {
                tracing::warn!(
                    "failed to list unannounced background tasks for session {session_id}: {error}"
                );
                continue;
            }
        };
        if !hooks.announce(&unannounced).await {
            continue;
        }
        let Ok(mut conversation) = entry.conversation.try_lock() else {
            continue;
        };
        // The snapshot above is from before the store round trips; an entry the idle sweep evicted
        // and released meanwhile has no registry and no tasks left to report into.
        if !hooks.still_resident(&entry).await {
            continue;
        }
        if !a_turn_can_carry_them(&entry.agent).await {
            continue;
        }
        // Admitted ahead of the stamp below, which is one-way: a batch refused for capacity stays
        // undelivered for the next tick rather than stamped and never handed out.
        let busy = match hooks.admit(&entry) {
            Ok(busy) => busy,
            Err(refused) => {
                tracing::debug!(
                    "background outcomes for session {session_id} wait: the process is at its cap \
                     of {cap} concurrent turns",
                    cap = refused.cap
                );
                continue;
            }
        };
        if let Err(error) = hooks.prepare(&entry).await {
            tracing::warn!(
                "holding a background outcome report for session {session_id} until its profile \
                 resolves: {error}"
            );
            continue;
        }
        let ready = match store.list_undelivered_background_tasks(session_id).await {
            Ok(ready) if !ready.is_empty() => ready,
            Ok(_) => continue,
            Err(error) => {
                tracing::warn!(
                    "failed to list undelivered background tasks for session {session_id}: {error}"
                );
                continue;
            }
        };
        if !ready.iter().any(|task| task.status.wakes_a_host()) {
            continue;
        }
        if !hooks.announce(&ready).await {
            continue;
        }
        let ids: Vec<String> = ready.iter().map(|task| task.id.clone()).collect();
        let claimed = match store.mark_background_tasks_delivered(&ids).await {
            Ok(claimed) => claimed,
            Err(error) => {
                tracing::warn!("failed to stamp background outcomes as delivered: {error}");
                continue;
            }
        };
        let ready = only_what_was_won(ready, &claimed);
        if ready.is_empty() {
            continue;
        }

        entry.touch();
        hooks.show_prompt(&entry, OutOfBandPrompt::Outcomes(&render_outcomes(&ready)));
        let cancellation = hooks.cancellation();
        let turn_id = uuid::Uuid::new_v4();
        let _published = entry
            .cancel
            .publish_turn(cancellation.clone(), busy.admission, turn_id);
        let input = crate::agent::TurnInput::outcomes(ready);
        hooks.begin_turn(&entry, turn_id, TurnOrigin::Background);
        let outcome = entry
            .agent
            .run_turn(&mut conversation, input, cancellation)
            .await;
        entry.touch();
        hooks
            .turn_closed(&entry, turn_id, &TurnOrigin::Background, &outcome)
            .await;
        let outcome = outcome.map(|_| ());
        hooks.finished(&entry, None, &outcome).await;
        if let Err(error) = outcome {
            tracing::warn!("background outcome turn for session {session_id} failed: {error}");
        }
    }
    ControlFlow::Continue(())
}

/// Put every due item of a session off by `wait`, counting the attempt. A failure to write it is
/// warned about and nothing else: the items stay due and the sweeper retries on its next tick.
async fn defer_session_inbox<H: HostHooks>(
    hooks: &H,
    session_id: uuid::Uuid,
    wait: std::time::Duration,
) {
    let not_before =
        chrono::Utc::now() + chrono::Duration::from_std(wait).unwrap_or(chrono::Duration::zero());
    if let Err(error) = hooks
        .store()
        .inbox_store()
        .defer_session_pending(session_id, not_before)
        .await
    {
        tracing::warn!("failed to defer inbox items for session {session_id}: {error}");
    }
}

/// How a drain ended, for the driver that started it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InboxDrain {
    /// Nothing waits that this driver could run now: the inbox is empty, the session is busy or
    /// gone, or a turn failed and its items were put off.
    Done,
    /// The process is at its concurrent-turn cap. The items are untouched and wait for a turn's
    /// end, which wakes the driver, or the next tick; a driver that woke itself over them would
    /// spin until a slot freed.
    AtCapacity,
}

/// Run turns on a session's waiting inbox items until nothing is waiting, the session is busy,
/// or a turn fails.
///
/// A busy session is waited for: the running turn reads steers itself at its round boundaries,
/// and what it leaves pending opens a turn the moment it ends. Turns are chained while items keep
/// arriving, each opening on everything waiting at that moment. A turn that fails before anything
/// reached the conversation has withdrawn its prompt and put its items back; they are deferred by
/// a backoff that doubles per attempt, and given up on past [`INBOX_RETRY_CEILING`], which is
/// when whoever is waiting is told rather than left with silence. A turn that fails after a tool
/// ran keeps its prompt, so its items are in the conversation already and the next turn, whoever
/// starts it, delivers them. A turn canceled from outside withdraws the items it opened on, as a
/// canceled client turn loses its prompt: a cancel that re-ran the same items would not be one.
#[allow(
    clippy::significant_drop_tightening,
    reason = "the conversation guard is the turn's exclusivity and lives to the end on purpose"
)]
pub(crate) async fn run_inbox_turns<H: HostHooks>(hooks: &H, session_id: uuid::Uuid) -> InboxDrain {
    let entry = match hooks.resident(session_id).await {
        Ok(Some(entry)) => entry,
        // Put off rather than left due, in both arms: the sweeper wakes itself whenever due items
        // remain after a pass, so a return that leaves them due is asked again at once, and a
        // session another process keeps for an hour would be probed in a hot loop for the hour.
        // The holder's own turns drain the inbox while it has the session, so its items wait the
        // short interval; a session that cannot be brought up is not a provider blip and waits the
        // longest. Each wait counts as an attempt, so neither holds an item past the ceiling.
        Ok(None) => {
            tracing::debug!(
                "session {session_id} is held elsewhere; its inbox waits {INBOX_RETRY_BASE:?}"
            );
            defer_session_inbox(hooks, session_id, INBOX_RETRY_BASE).await;
            return InboxDrain::Done;
        }
        Err(error) => {
            tracing::warn!(
                "inbox items for session {session_id} wait {INBOX_RETRY_LONGEST_WAIT:?}: session \
                 unavailable: {error}"
            );
            defer_session_inbox(hooks, session_id, INBOX_RETRY_LONGEST_WAIT).await;
            return InboxDrain::Done;
        }
    };
    let inbox = hooks.store().inbox_store();
    loop {
        if hooks.shutting_down() {
            return InboxDrain::Done;
        }
        // Waited for rather than tried: the turn holding it ends with items still pending
        // exactly when they arrived too late for its last boundary, and this is the turn that
        // carries them next.
        let cancellation = hooks.cancellation();
        let mut conversation = tokio::select! {
            guard = entry.conversation.lock() => guard,
            _ = cancellation.cancelled() => return InboxDrain::Done,
        };
        if !hooks.still_resident(&entry).await {
            return InboxDrain::Done;
        }
        // Admitted ahead of the items, which are read rather than taken: a refusal for capacity
        // leaves them pending exactly as they were, for the next turn's end or the next tick, and
        // the driver is told so it does not wake itself over them at once.
        let busy = match hooks.admit(&entry) {
            Ok(busy) => busy,
            Err(refused) => {
                tracing::debug!(
                    "inbox items for session {session_id} wait: the process is at its cap of \
                     {cap} concurrent turns",
                    cap = refused.cap
                );
                return InboxDrain::AtCapacity;
            }
        };
        let now = chrono::Utc::now();
        let items = match inbox.take_pending(session_id, &[], now).await {
            Ok(items) => items,
            Err(error) => {
                tracing::warn!("failed to read the inbox for session {session_id}: {error}");
                return InboxDrain::Done;
            }
        };
        if items.is_empty() {
            return InboxDrain::Done;
        }
        // An item that has failed before and has waited past the ceiling is given up on. A fresh
        // one is never expired: the ceiling bounds retrying, not queueing.
        let (expired, items): (Vec<_>, Vec<_>) = items
            .into_iter()
            .partition(|item| item.attempts > 0 && now - item.created_at >= INBOX_RETRY_CEILING);
        for item in expired {
            let reason = format!(
                "no turn could deliver it in {} attempt(s) over the last hour",
                item.attempts
            );
            let item_id = item.id;
            match inbox.withdraw(item_id, Some(reason.clone())).await {
                Ok(crate::store::inbox::Withdrawal::Withdrawn) => {
                    tracing::warn!("giving up on inbox item {item_id}: {reason}");
                    hooks.inbox_given_up(&entry, &item, &reason).await;
                }
                Ok(_) => {}
                Err(error) => tracing::warn!("failed to withdraw inbox item {item_id}: {error}"),
            }
        }
        if items.is_empty() {
            continue;
        }
        let attempts = items.iter().map(|item| item.attempts).max().unwrap_or(0);
        let item_ids: Vec<uuid::Uuid> = items.iter().map(|item| item.id).collect();

        entry.touch();
        if let Err(error) = hooks.prepare(&entry).await {
            tracing::warn!(
                "inbox items for session {session_id} wait: the session's profile did not \
                 resolve ({error})"
            );
            return InboxDrain::Done;
        }
        let cancellation = hooks.cancellation();
        let turn_id = uuid::Uuid::new_v4();
        let _published = entry
            .cancel
            .publish_turn(cancellation.clone(), busy.admission, turn_id);
        let Ok(input) = crate::agent::TurnInput::inbox(items) else {
            // Unreachable while the door refuses an empty message; left due, the sweeper would
            // ask again at once.
            tracing::warn!("inbox items for session {session_id} carry no words; they wait");
            defer_session_inbox(hooks, session_id, INBOX_RETRY_LONGEST_WAIT).await;
            return InboxDrain::Done;
        };
        let input = input.retaining(crate::conversation::PromptRetention::Withdraw);
        let riding = if hooks.background_enabled() {
            claim_undelivered_outcomes(&entry.agent, hooks.store(), session_id).await
        } else {
            Vec::new()
        };
        if !riding.is_empty() && !hooks.announce(&riding).await {
            tracing::warn!(
                "an inbox turn carries {outcomes} background outcome(s) a webhook rejected; \
                 delivering them anyway, since they are claimed",
                outcomes = riding.len()
            );
        }
        let input = input.riding(riding);
        let origin = TurnOrigin::Inbox {
            item_ids: item_ids.clone(),
        };
        hooks.begin_turn(&entry, turn_id, origin.clone());
        let outcome = entry
            .agent
            .run_turn(&mut conversation, input, cancellation)
            .await;
        entry.touch();
        hooks.turn_closed(&entry, turn_id, &origin, &outcome).await;
        let outcome = outcome.map(|_| ());
        if hooks.shutting_down() {
            return InboxDrain::Done;
        }
        hooks.finished(&entry, None, &outcome).await;
        match outcome {
            Ok(()) => tracing::info!("inbox turn on session {session_id} completed"),
            // Every item it opened on was withdrawn while it was starting; nothing failed.
            Err(MekaError::EmptyPrompt) => {}
            Err(MekaError::Interrupted) => {
                // Withdrawn rather than re-run: the withdrawal put them back pending, and a
                // driver that opened the same turn again would undo the cancel.
                for item_id in item_ids {
                    match inbox
                        .withdraw(
                            item_id,
                            Some("the turn opened on it was canceled".to_string()),
                        )
                        .await
                    {
                        Ok(crate::store::inbox::Withdrawal::Withdrawn) => {
                            hooks.inbox_withdrawn(&entry, item_id).await;
                        }
                        Ok(_) => {}
                        Err(error) => {
                            tracing::warn!("failed to withdraw inbox item {item_id}: {error}");
                        }
                    }
                }
            }
            Err(error) => {
                // Only what the withdrawal put back is pending; what a tool round kept is in the
                // conversation and waits for a turn of any kind.
                let delay = INBOX_RETRY_BASE
                    .saturating_mul(1u32 << attempts.min(16))
                    .min(INBOX_RETRY_LONGEST_WAIT);
                tracing::warn!(
                    "inbox turn on session {session_id} failed: {error}; trying again in \
                     {delay:?}"
                );
                let not_before = chrono::Utc::now()
                    + chrono::Duration::from_std(delay).unwrap_or(chrono::Duration::zero());
                if let Err(error) = inbox.defer_pending(&item_ids, not_before).await {
                    tracing::warn!("failed to defer inbox items: {error}");
                }
                return InboxDrain::Done;
            }
        }
    }
}

/// Whether the turn that would carry these outcomes can start at all.
///
/// `Agent::run_turn` gates on MCP readiness as its first statement, before it touches the
/// conversation, so a required server that is down refuses every turn -- and a batch stamped ahead
/// of one is a batch nobody is ever told about, because the stamp is one-way. Every claimer asks
/// this first: [`claim_undelivered_outcomes`] for the hosts that fold a batch into somebody's
/// prompt, and the two pollers directly, since they stamp with
/// [`crate::store::background::BackgroundStore::mark_background_tasks_delivered`] to render the
/// batch as a turn of its own.
pub(crate) async fn a_turn_can_carry_them(agent: &crate::agent::Agent) -> bool {
    if !agent.background_enabled() {
        return false;
    }
    match agent.ensure_ready_for_turn().await {
        Ok(()) => true,
        Err(error) => {
            tracing::warn!(
                "holding background task outcomes: a turn cannot start right now: {error}"
            );
            false
        }
    }
}
#[cfg(test)]
mod tests {
    /// Asserted against the source, and weaker than a behavioral test: nothing in the suite drives
    /// background-outcome delivery through this driver, so this checks *order* rather than mere
    /// presence.
    ///
    /// What it defends: `list_undelivered_background_tasks` filters on `delivered_at IS NULL`, so a
    /// stamped batch has no re-delivery path. Stamping first lets a provider lookup that comes back
    /// `SQLITE_BUSY`, an ordinary occurrence with a second meka process on the store, destroy a
    /// report the user was waiting on, with one `warn!` and nothing else.
    #[test]
    fn a_background_outcome_is_stamped_only_once_its_turn_can_run() {
        let source = include_str!("scheduler.rs");
        let body = source
            .split("async fn deliver_ready_outcomes<")
            .nth(1)
            .expect("the function this test is about")
            .split("\n#[cfg(test)]")
            .next()
            .expect("splitting always yields a first part");

        let prepared = body
            .find("hooks.prepare(")
            .expect("the turn must run on the profile the host prepares, the row's");
        let stamp = body
            .find("mark_background_tasks_delivered")
            .expect("the batch is stamped here, or this test is watching the wrong function");
        // The call, not the word: prose above it mentions `run_turn` by name, and matching that
        // put the "turn" earlier in the body than the stamp it is supposed to follow.
        let turn = body
            .find(".run_turn(")
            .expect("the turn this whole ordering is about is no longer in this function");
        assert!(
            prepared < stamp,
            "the host must prepare the runtime before the batch is stamped, so a failure leaves \
             the outcomes for the next sweep instead of destroying them; found prepare@{prepared} \
             stamp@{stamp}"
        );
        // Order alone is not the invariant: deleting the `continue;` leaves the sequence intact
        // while the failure falls through to stamp the batch and run the turn on the wrong
        // profile.
        let arm = body.get(prepared..stamp).expect("ordered above");
        assert!(
            arm.contains("continue;"),
            "a preparation failure must move on before the stamp, not merely be logged before it"
        );
        assert!(
            stamp < turn,
            "but the stamp must still precede the turn: an outcome that reliably wedges the \
             process must not be redelivered on every restart; found stamp@{stamp} turn@{turn}"
        );
    }

    /// The prompt is admitted before any outcome is claimed to ride on it, in both places that
    /// fold a batch into a prompt. Asserted against the source for the reason the test above is:
    /// neither `render_prompt` nor the editor can produce an empty prompt today, so the refusal
    /// is unreachable and a behavioral test cannot see the order. The claim is one-way, so the
    /// order is what stops a refused prompt from leaving a batch stamped and never delivered.
    #[test]
    fn a_prompt_is_admitted_before_the_outcomes_that_ride_it_are_claimed() {
        let wakeup = include_str!("scheduler.rs")
            .split("pub(crate) async fn run_wakeup<")
            .nth(1)
            .expect("the function this test is about")
            .split("\npub(crate) async fn deliver_ready_outcomes<")
            .next()
            .expect("splitting always yields a first part");
        let admitted = wakeup
            .find("TurnInput::from_parts(")
            .expect("the fire admits its prompt");
        let claimed = wakeup
            .find("claim_undelivered_outcomes(")
            .expect("the fire claims the outcomes that ride it");
        assert!(
            admitted < claimed,
            "a scheduled fire must admit its prompt before claiming outcomes; found \
             admit@{admitted} claim@{claimed}"
        );

        let typed = include_str!("repl.rs")
            .split("ReplEvent::UserInput(input) => {")
            .nth(1)
            .expect("the REPL's typed-prompt arm")
            .split("ReplEvent::Command(command) => {")
            .next()
            .expect("splitting always yields a first part");
        let admitted = typed
            .find("TurnInput::from_parts(")
            .expect("the REPL admits the typed prompt");
        let claimed = typed
            .find("collect_background_outcomes(")
            .expect("the REPL claims the outcomes that ride it");
        assert!(
            admitted < claimed,
            "the REPL must admit a typed prompt before claiming outcomes; found \
             admit@{admitted} claim@{claimed}"
        );
    }
}
