//! What every host does when the scheduler fires a job or the background poller finds finished
//! work: find the session, take its runtime, publish a token, fold undelivered outcomes into the
//! prompt, run the turn, and report. The three hosts once carried three copies of this loop that
//! agreed on most of it and disagreed, silently, on the rest: whether an ungated job waits behind a
//! turn, whether a canceled fire counts as failed, which host announced outcomes and when. One
//! driver, with the differences asked through [`HostHooks`].

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
    let busy = entry.mark_busy();
    if let Err(error) = hooks.prepare(&entry).await {
        tracing::warn!(
            "job {job_id} did not run: its session's recorded profile could not be resolved. Fix the \
             profile, or move the session with `meka -r <id> --profile <name>`: {error}"
        );
        return FireOutcome::Unrunnable;
    }
    let cancellation = hooks.cancellation();
    let _published = entry.cancel.publish(cancellation.clone(), busy.admission);

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
                "scheduled job {job_id} carries {outcomes} background outcome(s) a delivery \
                 webhook refused; delivering them anyway, since they are claimed",
                outcomes = riding.len()
            );
        }
        hooks.show_prompt(&entry, OutOfBandPrompt::Outcomes(&render_outcomes(&riding)));
    }
    hooks.show_prompt(&entry, OutOfBandPrompt::Scheduled(&wakeup));
    let input = input.riding(riding);

    let outcome = entry
        .agent
        .run_turn(&mut conversation, input, cancellation)
        .await
        .map(|_| ());
    entry.touch();
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
        if let Err(error) = hooks.prepare(&entry).await {
            tracing::warn!(
                "holding a background outcome report for session {session_id}: its recorded profile could \
                 not be resolved. It is retried on the next sweep: {error}"
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
        let busy = entry.mark_busy();
        hooks.show_prompt(&entry, OutOfBandPrompt::Outcomes(&render_outcomes(&ready)));
        let cancellation = hooks.cancellation();
        let _published = entry.cancel.publish(cancellation.clone(), busy.admission);
        let outcome = entry
            .agent
            .run_turn(
                &mut conversation,
                crate::agent::TurnInput::outcomes(ready),
                cancellation,
            )
            .await
            .map(|_| ());
        entry.touch();
        hooks.finished(&entry, None, &outcome).await;
        if let Err(error) = outcome {
            tracing::warn!("background outcome turn for session {session_id} failed: {error}");
        }
    }
    ControlFlow::Continue(())
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
    /// Asserted against the source, and honestly weaker than a behavioral test: nothing in the
    /// suite drives background-outcome delivery through this driver, which a mutation sweep once
    /// confirmed for the ACP copy by replacing it with `()` and staying green. Until that coverage
    /// exists this is what stands between a future edit and a silent regression, so it checks
    /// *order* rather than mere presence.
    ///
    /// What it defends: `list_undelivered_background_tasks` filters on `delivered_at IS NULL`, so a
    /// stamped batch has no re-delivery path. Stamping first meant a provider lookup that came back
    /// `SQLITE_BUSY` -- an ordinary occurrence with a second meka process on the store -- destroyed
    /// a report the user was waiting on, with one `warn!` and nothing else.
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
