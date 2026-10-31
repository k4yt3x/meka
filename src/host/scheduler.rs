//! What every host does when the scheduler fires a job or the background poller finds finished
//! work: find the session, take its runtime, publish a token, fold undelivered outcomes into the
//! prompt, run the turn, and report. One driver, so the hosts cannot disagree on whether an ungated
//! job waits behind a turn, whether a canceled fire counts as failed, or who announces outcomes and
//! when; what differs is asked through [`HostHooks`].

use std::ops::ControlFlow;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::{
    background::only_what_was_won,
    error::MekaError,
    host::{ResidentSession, Sessions, claim_undelivered_outcomes},
    schedule::ScheduledJob,
    scheduler::{FireOutcome, Wakeup},
    store::{Store, background::BackgroundTask},
};

/// Who started a turn, for the host's own accounting: `serve` names it on the session feed's
/// `turn.started`, and the hosts that run an out-of-band turn name it to their hooks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TurnStarter {
    /// `POST /v1/sessions/{id}/turn`.
    Client,
    /// The inbox driver, on these items.
    Inbox { item_ids: Vec<uuid::Uuid> },
    /// A scheduled job firing.
    Schedule { job_id: String },
    /// Finished background work delivered as a turn of its own.
    Background,
    /// `POST /v1/sessions/{id}/compact`: the checkpoint turn a compaction runs.
    Compaction,
    /// A sub-agent's turn, opened by its parent's spawn or follow-up.
    Parent {
        parent_id: Option<uuid::Uuid>,
        tool_call_id: Option<String>,
    },
}

impl TurnStarter {
    /// The `source` member of `turn.started`, the word the turn's row records.
    pub(crate) const fn source(&self) -> crate::store::turns::TurnSource {
        use crate::store::turns::TurnSource;
        match self {
            Self::Client => TurnSource::Client,
            Self::Inbox { .. } => TurnSource::Inbox,
            Self::Schedule { .. } => TurnSource::Schedule,
            Self::Background => TurnSource::Background,
            Self::Compaction => TurnSource::Compaction,
            Self::Parent { .. } => TurnSource::Parent,
        }
    }

    /// Write `source` and what identifies it into a `turn.started` payload: the real one
    /// `HttpFrontend::begin_turn` publishes and the one a feed synthesizes for a client that
    /// attaches mid-turn, so both name the turn the same way.
    pub(crate) fn describe(&self, data: &mut serde_json::Value) {
        let Some(object) = data.as_object_mut() else {
            return;
        };
        object.insert(
            "source".into(),
            serde_json::Value::String(self.source().name().to_string()),
        );
        match self {
            Self::Inbox { item_ids } => {
                object.insert("item_ids".into(), serde_json::json!(item_ids));
            }
            Self::Schedule { job_id } => {
                object.insert("job_id".into(), serde_json::json!(job_id));
            }
            Self::Parent {
                parent_id,
                tool_call_id,
            } => {
                if let Some(parent_id) = parent_id {
                    object.insert("parent_id".into(), serde_json::json!(parent_id));
                }
                if let Some(tool_call_id) = tool_call_id {
                    object.insert("tool_call_id".into(), serde_json::json!(tool_call_id));
                }
            }
            Self::Client | Self::Background | Self::Compaction => {}
        }
    }
}

/// First wait before a turn that failed on inbox items is tried again; doubles per attempt.
/// Also the wait for a session another process holds, asked again at this pace until it lets go.
const INBOX_RETRY_BASE: std::time::Duration = std::time::Duration::from_secs(10);
/// Longest wait between two attempts, and the wait for a session that could not be brought up
/// at all, which no backoff from a first try would fit.
const INBOX_RETRY_LONGEST_WAIT: std::time::Duration = std::time::Duration::from_secs(5 * 60);
/// `[serve].inbox_retry_ceiling` when unset: how long an item is retried for before it is given
/// up on, from when it was enqueued. Long enough to ride out a provider blip; short enough that an
/// outage is reported to whoever is waiting rather than left to look like silence.
pub(crate) const DEFAULT_INBOX_RETRY_CEILING: std::time::Duration =
    std::time::Duration::from_secs(60 * 60);

/// Whether waiting can change the answer a failed inbox turn got: the upstream could not serve the
/// profile and may again (an outage, a dropped stream, a quota that resets), a required MCP
/// server was not ready and may reconnect, or meka's own store or disk refused (a lock another
/// process holds, a disk that is full) and may not next time. Anything else, a rejected
/// credential, a request the upstream judged wrong, a session its profile cannot read, fails the
/// same way on every attempt, so the item is given up on at once and the sender is told now.
fn inbox_retry_can_help(error: &MekaError) -> bool {
    error.is_profile_unavailable()
        || matches!(
            error,
            MekaError::McpTurnGated { .. } | MekaError::Database(_) | MekaError::Io(_)
        )
}

/// When the next attempt on failed items runs: the backoff for the attempt count, pulled earlier
/// to the upstream's own word when it said it would be back sooner, never later. A stated time is
/// an estimate the operator can overtake by topping an account up, so it only ever shortens a
/// wait, and the floor keeps a reset already past from spinning the driver.
fn inbox_retry_delay(attempts: u32, advised: Option<std::time::Duration>) -> std::time::Duration {
    let backoff = INBOX_RETRY_BASE
        .saturating_mul(1u32 << attempts.min(16))
        .min(INBOX_RETRY_LONGEST_WAIT);
    match advised {
        Some(advised) => backoff.min(advised.max(INBOX_RETRY_BASE)),
        None => backoff,
    }
}

/// Why an item is given up on when its retries ran out of the ceiling.
fn ceiling_reason(attempts: u32, ceiling: std::time::Duration) -> String {
    if ceiling.is_zero() {
        return "it could not be delivered at once and `[serve] inbox_retry_ceiling` is 0s"
            .to_string();
    }
    format!(
        "no turn could deliver it in {attempts} attempt(s) within the retry ceiling of {}",
        crate::text::format_duration_words(ceiling)
    )
}

/// A `std` duration as `chrono` measures one, saturating rather than failing on one too long.
fn chrono_duration(duration: std::time::Duration) -> chrono::Duration {
    chrono::Duration::from_std(duration).unwrap_or(chrono::Duration::MAX)
}

/// When retrying `item` ends: its enqueue plus `ceiling`. The one rule for the expiry at a
/// drain's start, for a deferral that would land too late, and for the `retry_until` the item
/// shows, so a client reads the deadline the driver keeps. An item never tried is not expired by
/// it: its first attempt runs whenever its turn comes, and only a failure after it is not retried.
pub(crate) fn inbox_deadline(
    item: &crate::store::inbox::InboxItem,
    ceiling: std::time::Duration,
) -> chrono::DateTime<chrono::Utc> {
    item.created_at
        .checked_add_signed(chrono_duration(ceiling))
        .unwrap_or(chrono::DateTime::<chrono::Utc>::MAX_UTC)
}

/// Whether an attempt on `item` at `at` would fall at or past its deadline.
fn past_ceiling(
    at: chrono::DateTime<chrono::Utc>,
    item: &crate::store::inbox::InboxItem,
    ceiling: std::time::Duration,
) -> bool {
    at >= inbox_deadline(item, ceiling)
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
    /// the map; the screen's one session is never released while it runs.
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

    /// What the host does with the session before a turn, under the conversation lock. The two
    /// servers move the agent onto the profile the row records, through
    /// `crate::host::apply_recorded_bindings`; an error here makes the fire unrunnable rather than
    /// failed.
    async fn prepare(&self, _entry: &Self::Entry) -> anyhow::Result<()> {
        Ok(())
    }

    /// Announce finished tasks where the host announces them, and say whether they may be
    /// delivered into a turn. `serve` posts webhooks and lets a delivery webhook veto; the others
    /// have nothing to announce to.
    async fn announce(&self, _tasks: &[BackgroundTask]) -> bool {
        true
    }

    /// The stop signal an out-of-band turn runs under. `serve` derives it from shutdown so a
    /// stopping server ends the turn, and the turn then says it was the server.
    fn cancellation(&self) -> crate::agent::TurnCancellation {
        crate::agent::TurnCancellation::new()
    }

    /// An out-of-band turn is about to run under `turn_id`. `serve` opens it on the session feed;
    /// the others have no feed to open it on.
    fn begin_turn(&self, _entry: &Self::Entry, _turn_id: uuid::Uuid, _origin: TurnStarter) {}

    /// An out-of-band turn has ended, before [`Self::finished`] announces it. `serve` records the
    /// terminal on the feed, closes the turn there, and posts the turn webhooks an inbox turn
    /// alone has nothing else to carry.
    async fn turn_closed(
        &self,
        _entry: &Self::Entry,
        _turn_id: uuid::Uuid,
        _origin: &TurnStarter,
        _outcome: &Result<crate::agent::TurnOutcome, MekaError>,
        _cancellation: &crate::agent::TurnCancellation,
    ) {
    }

    /// An inbox item was given up on and withdrawn, for `reason`, which `error` decided when it
    /// was a failed turn's. `serve` tells the feed and the webhooks.
    async fn inbox_given_up(
        &self,
        _entry: &Self::Entry,
        _item: &crate::store::inbox::InboxItem,
        _reason: &str,
        _error: Option<&MekaError>,
    ) {
    }

    /// A turn on `items` failed with `error` in a way waiting can change, and they wait for the
    /// next attempt at `not_before`. `serve` tells the feed and the webhooks.
    async fn inbox_deferred(
        &self,
        _entry: &Self::Entry,
        _items: &[crate::store::inbox::InboxItem],
        _not_before: chrono::DateTime<chrono::Utc>,
        _error: &MekaError,
    ) {
    }

    /// How long an item is retried for, from its enqueue, before it is given up on.
    fn inbox_retry_ceiling(&self) -> std::time::Duration {
        DEFAULT_INBOX_RETRY_CEILING
    }

    /// The error a failed turn leaves on its items, as the host's API states an error, for
    /// whoever reads the item later; `None` on a host whose items nobody reads back.
    fn inbox_error_record(&self, _error: &MekaError) -> Option<String> {
        None
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
             ({error}); restore the profile in config.toml or move the session onto another"
        );
        hooks.failed_before_running(job, &error);
        return FireOutcome::Unrunnable;
    }
    let cancellation = hooks.cancellation();
    let turn_id = uuid::Uuid::new_v4();
    let _published = entry
        .cancel
        .publish_turn(cancellation.clone(), busy.admission, turn_id);

    // Admitted before any outcome is claimed. A claim is one-way, so a prompt refused after it
    // would leave the batch stamped delivered and never handed out again.
    let input = match crate::agent::TurnInput::scheduled(
        job.id.clone(),
        job.prompt.clone(),
        wakeup.render_prompt(),
        wakeup.lateness(),
    ) {
        Ok(input) => input
            .retaining(job.prompt_retention())
            .originating(crate::provider::TurnOrigin::Scheduled)
            .identified(turn_id),
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
    }
    let input = input.riding(riding);

    let origin = TurnStarter::Schedule {
        job_id: job.id.clone(),
    };
    hooks.begin_turn(&entry, turn_id, origin.clone());
    let outcome = entry
        .agent
        .run_turn(&mut conversation, input, cancellation.clone())
        .await;
    entry.touch();
    hooks
        .turn_closed(&entry, turn_id, &origin, &outcome, &cancellation)
        .await;
    let outcome = outcome.map(|_| ());
    // Before `finished`, not after: a fire the drain cut short is deferred and fires again on the
    // next start, so telling a webhook it failed announced one occurrence twice.
    if deferred_by_the_drain(hooks.shutting_down(), &outcome) {
        tracing::info!(
            "scheduled job {job_id} was cut short by the shutdown; deferring the occurrence"
        );
        return FireOutcome::Deferred;
    }
    hooks.finished(&entry, Some(job), &outcome).await;
    match &outcome {
        Ok(()) => tracing::info!("scheduled job {job_id} completed"),
        Err(error) => tracing::warn!("scheduled job {job_id} failed: {error}"),
    }
    FireOutcome::Ran
}

/// Deliver finished background outcomes on every tick until `shutdown` fires or the sweep says
/// it is done: the one poller both servers spawn, so the supervision is written once.
///
/// Supervised because this sweep runs a whole agent turn, so anything in the tool loop can
/// panic, and losing the task would stop every background outcome from ever being delivered,
/// silently, since nothing joins the handle. A task that finished would then sit stamped and
/// unreported forever, which is exactly the promise `background.rs` opens by making.
pub(crate) fn spawn_outcome_poller<H, K>(
    hooks: H,
    sessions: Sessions<K, H::Entry>,
    poll_interval: std::time::Duration,
    shutdown: CancellationToken,
) -> tokio::task::JoinHandle<()>
where
    H: HostHooks + 'static,
    K: Eq + std::hash::Hash + Clone + Send + Sync + 'static,
    H::Entry: Send + Sync + 'static,
{
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(poll_interval);
        ticker.tick().await;
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => return,
                _ = ticker.tick() => {}
            }
            let sweep = std::panic::AssertUnwindSafe(deliver_ready_outcomes(&hooks, &sessions));
            match futures::FutureExt::catch_unwind(sweep).await {
                Ok(ControlFlow::Break(())) => return,
                Ok(ControlFlow::Continue(())) => {}
                Err(panic) => tracing::warn!(
                    "background outcome sweep panicked ({panic}); continuing",
                    panic = crate::error::panic_message(&*panic)
                ),
            }
        }
    })
}

/// Whether a fire's occurrence is handed back for the next start rather than completed: only when
/// the drain cut the turn short. A turn that finished, or failed on its own, while the drain began
/// is complete, its messages persisted; deferring it too makes a one-shot reminder deliver twice.
fn deferred_by_the_drain(shutting_down: bool, outcome: &Result<(), MekaError>) -> bool {
    shutting_down && matches!(outcome, Err(MekaError::Interrupted))
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
        // Read ahead of the lock. A session carries a canceled task undelivered until a turn
        // folds it in, and this runs every tick, so the lock is not taken on the chance of
        // something to deliver: a client's turn is refused while it is held, and a hold the
        // poller has not admitted is one `turn_in_flight` says nothing about.
        let ready = match store.list_undelivered_background_tasks(session_id).await {
            Ok(ready) if ready.iter().any(|task| task.status.wakes_a_host()) => ready,
            Ok(_) => continue,
            Err(error) => {
                tracing::warn!(
                    "failed to list undelivered background tasks for session {session_id}: {error}"
                );
                continue;
            }
        };
        let Ok(mut conversation) = entry.conversation.try_lock() else {
            continue;
        };
        // Admitted the moment the lock is held, so the session reports the hold as a turn in
        // flight for as long as it lasts, and ahead of the stamp below, which is one-way: a batch
        // refused for capacity stays undelivered for the next tick rather than stamped and never
        // handed out.
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
                "holding a background outcome report for session {session_id} until its profile \
                 resolves: {error}"
            );
            continue;
        }
        // Asked of the profile just resolved: the turn would refuse it and put the batch back,
        // and the next tick would claim it again.
        if let Err(error) = entry
            .agent
            .refuse_images_the_profile_cannot_read(&conversation)
        {
            tracing::warn!("holding a background outcome report for session {session_id}: {error}");
            continue;
        }
        // A turn that ran between the read and the lock may have folded the batch in already;
        // the stamp below then wins nothing, and the tick moves on.
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
        let cancellation = hooks.cancellation();
        let turn_id = uuid::Uuid::new_v4();
        let _published = entry
            .cancel
            .publish_turn(cancellation.clone(), busy.admission, turn_id);
        let input = crate::agent::TurnInput::outcomes(ready).identified(turn_id);
        hooks.begin_turn(&entry, turn_id, TurnStarter::Background);
        let outcome = entry
            .agent
            .run_turn(&mut conversation, input, cancellation.clone())
            .await;
        entry.touch();
        hooks
            .turn_closed(
                &entry,
                turn_id,
                &TurnStarter::Background,
                &outcome,
                &cancellation,
            )
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
/// reached the conversation has withdrawn its prompt and put its items back; when waiting can
/// change the answer ([`inbox_retry_can_help`]) they are deferred by [`inbox_retry_delay`] and
/// given up on past the host's retry ceiling, which is when whoever is waiting is told rather than
/// left with silence, and otherwise given up on at once. A turn that fails after a tool
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
            _ = cancellation.token().cancelled() => return InboxDrain::Done,
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
            // Put off like the arms above, or the sweeper asks again at once for as long as the
            // read keeps failing.
            Err(error) => {
                tracing::warn!(
                    "inbox items for session {session_id} wait {INBOX_RETRY_LONGEST_WAIT:?}: \
                     failed to read the inbox: {error}"
                );
                defer_session_inbox(hooks, session_id, INBOX_RETRY_LONGEST_WAIT).await;
                return InboxDrain::Done;
            }
        };
        if items.is_empty() {
            return InboxDrain::Done;
        }
        // An item that has failed before and has waited past the ceiling is given up on. A fresh
        // one is never expired: the ceiling bounds retrying, not queueing.
        let ceiling = hooks.inbox_retry_ceiling();
        let (expired, items): (Vec<_>, Vec<_>) = items
            .into_iter()
            .partition(|item| item.attempts > 0 && past_ceiling(now, item, ceiling));
        for item in expired {
            let reason = ceiling_reason(item.attempts, ceiling);
            give_up_inbox_item(hooks, &entry, &item, reason, None).await;
        }
        if items.is_empty() {
            continue;
        }
        let attempts = items.iter().map(|item| item.attempts).max().unwrap_or(0);
        let item_ids: Vec<uuid::Uuid> = items.iter().map(|item| item.id).collect();
        // Kept for the failure arms below, which name the items a turn could not carry.
        let offered = items.clone();

        entry.touch();
        // Put off like the arms above, or the sweeper asks again at once: a session whose row
        // names a profile this process cannot resolve stays that way until somebody moves it.
        if let Err(error) = hooks.prepare(&entry).await {
            tracing::warn!(
                "inbox items for session {session_id} wait {INBOX_RETRY_LONGEST_WAIT:?}: the \
                 session's profile did not resolve ({error})"
            );
            defer_session_inbox(hooks, session_id, INBOX_RETRY_LONGEST_WAIT).await;
            return InboxDrain::Done;
        }
        let cancellation = hooks.cancellation();
        let turn_id = uuid::Uuid::new_v4();
        let _published = entry
            .cancel
            .publish_turn(cancellation.clone(), busy.admission, turn_id);
        let Ok(input) =
            crate::agent::TurnInput::inbox(items).map(|input| input.identified(turn_id))
        else {
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
        let origin = TurnStarter::Inbox {
            item_ids: item_ids.clone(),
        };
        hooks.begin_turn(&entry, turn_id, origin.clone());
        let outcome = entry
            .agent
            .run_turn(&mut conversation, input, cancellation.clone())
            .await;
        entry.touch();
        hooks
            .turn_closed(&entry, turn_id, &origin, &outcome, &cancellation)
            .await;
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
                            None,
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
                if !inbox_retry_can_help(&error) {
                    let reason = format!(
                        "the turn opened on it failed, and waiting would not change that: {}",
                        error.sentence()
                    );
                    tracing::warn!("inbox turn on session {session_id} failed: {error}");
                    for item in &offered {
                        give_up_inbox_item(hooks, &entry, item, reason.clone(), Some(&error)).await;
                    }
                    return InboxDrain::Done;
                }
                let delay = inbox_retry_delay(attempts, error.resumes_after());
                let not_before = chrono::Utc::now() + chrono_duration(delay);
                // An item whose next attempt would land past the ceiling is given up now, so the
                // sender hears at once rather than at the tick that would have found it expired.
                let (hopeless, retried): (Vec<_>, Vec<_>) = offered
                    .into_iter()
                    .partition(|item| past_ceiling(not_before, item, ceiling));
                for item in &hopeless {
                    let reason = ceiling_reason(item.attempts + 1, ceiling);
                    give_up_inbox_item(hooks, &entry, item, reason, Some(&error)).await;
                }
                if retried.is_empty() {
                    return InboxDrain::Done;
                }
                tracing::warn!(
                    "inbox turn on session {session_id} failed: {error}; trying again in \
                     {delay:?}"
                );
                let retried_ids: Vec<uuid::Uuid> = retried.iter().map(|item| item.id).collect();
                let record = hooks.inbox_error_record(&error);
                let deferred = match inbox
                    .defer_pending(&retried_ids, not_before, record.as_deref())
                    .await
                {
                    Ok(deferred) => deferred,
                    Err(error) => {
                        tracing::warn!("failed to defer inbox items: {error}");
                        Vec::new()
                    }
                };
                // Named on the event are the rows the store held back; one a tool round already
                // carried into the conversation is history and waits for no attempt.
                let waiting: Vec<_> = retried
                    .into_iter()
                    .filter(|item| deferred.contains(&item.id))
                    .collect();
                if !waiting.is_empty() {
                    hooks
                        .inbox_deferred(&entry, &waiting, not_before, &error)
                        .await;
                }
                return InboxDrain::Done;
            }
        }
    }
}

/// Withdraw `item` for `reason` and tell the host, with the error that decided it when one did.
async fn give_up_inbox_item<H: HostHooks>(
    hooks: &H,
    entry: &H::Entry,
    item: &crate::store::inbox::InboxItem,
    reason: String,
    error: Option<&MekaError>,
) {
    let item_id = item.id;
    match hooks
        .store()
        .inbox_store()
        .withdraw(
            item_id,
            Some(reason.clone()),
            // The error that decided it, or the last attempt's when the ceiling did with nothing
            // newer known: what a reader of the item is shown as why.
            error
                .and_then(|error| hooks.inbox_error_record(error))
                .or_else(|| item.last_error.clone()),
        )
        .await
    {
        Ok(crate::store::inbox::Withdrawal::Withdrawn) => {
            tracing::warn!("giving up on inbox item {item_id}: {reason}");
            hooks.inbox_given_up(entry, item, &reason, error).await;
        }
        Ok(_) => {}
        Err(error) => tracing::warn!("failed to withdraw inbox item {item_id}: {error}"),
    }
}

/// Whether the turn that would carry these outcomes can start at all.
///
/// `Agent::run_turn` gates on MCP readiness as its first statement, before it touches the
/// conversation, so a required server that is down refuses every turn, and puts back the batch it
/// was to carry. Every claimer asks this first, so a batch is not claimed only to go back:
/// [`claim_undelivered_outcomes`] for the hosts that fold a batch into somebody's prompt, and the
/// two pollers directly, since they stamp with
/// [`crate::store::background::BackgroundStore::mark_background_tasks_delivered`] to render the
/// batch as a turn of its own and would otherwise claim it again on every tick. A turn's own round
/// boundary claims without asking: it is running, so it can carry them.
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
    use super::*;

    /// The next attempt waits the backoff for the attempt count, or less when the upstream said it
    /// would be back sooner; a stated time never lengthens the wait, and one already past still
    /// waits the floor.
    #[test]
    fn an_inbox_retry_waits_the_backoff_or_the_upstream_s_sooner_word() {
        use std::time::Duration;
        assert_eq!(inbox_retry_delay(0, None), Duration::from_secs(10));
        assert_eq!(inbox_retry_delay(3, None), Duration::from_secs(80));
        assert_eq!(inbox_retry_delay(20, None), Duration::from_secs(300));
        assert_eq!(
            inbox_retry_delay(5, Some(Duration::from_secs(45))),
            Duration::from_secs(45)
        );
        assert_eq!(
            inbox_retry_delay(0, Some(Duration::from_secs(7_200))),
            Duration::from_secs(10)
        );
        assert_eq!(
            inbox_retry_delay(2, Some(Duration::ZERO)),
            Duration::from_secs(10)
        );
        assert!(ceiling_reason(3, Duration::from_secs(3_600)).contains("3 attempt(s)"));
        assert!(ceiling_reason(3, Duration::from_secs(3_600)).contains("1 hour"));
        assert!(ceiling_reason(1, Duration::ZERO).contains("inbox_retry_ceiling"));
    }

    /// The deadline is the enqueue plus the ceiling, and a ceiling too long to add leaves the
    /// item never past it rather than failing the arithmetic.
    #[test]
    fn an_item_s_deadline_is_its_enqueue_plus_the_ceiling() {
        let created_at = chrono::Utc::now();
        let item = crate::store::inbox::InboxItem {
            id: uuid::Uuid::nil(),
            session_id: uuid::Uuid::nil(),
            class: crate::conversation::InboxClass::Steer,
            source: None,
            body: "hi".to_string(),
            created_at,
            attempts: 0,
            not_before: None,
            last_error: None,
            appended_at: None,
            delivered_at: None,
            withdrawn_at: None,
            failure: None,
        };
        let hour = std::time::Duration::from_secs(3600);
        assert_eq!(
            inbox_deadline(&item, hour),
            created_at + chrono::Duration::hours(1)
        );
        assert!(past_ceiling(
            created_at + chrono::Duration::hours(1),
            &item,
            hour
        ));
        assert!(!past_ceiling(
            created_at + chrono::Duration::minutes(59),
            &item,
            hour
        ));
        assert_eq!(
            inbox_deadline(&item, std::time::Duration::ZERO),
            created_at,
            "a zero ceiling expires an item at its enqueue"
        );
        assert_eq!(
            inbox_deadline(&item, std::time::Duration::MAX),
            chrono::DateTime::<chrono::Utc>::MAX_UTC,
            "a ceiling too long to add saturates rather than failing"
        );
        assert!(
            !past_ceiling(
                created_at + chrono::Duration::days(365_000),
                &item,
                std::time::Duration::MAX
            ),
            "and is not reached in a thousand years"
        );
    }

    /// Only a failure waiting can change is retried: the upstream's unavailability, a quota
    /// refusal, a required MCP server not ready and meka's own store or disk refusing; a rejected
    /// credential, a bad request, a configuration refusal and an overflow fail the item at once.
    #[test]
    fn only_a_failure_waiting_can_change_is_retried() {
        assert!(inbox_retry_can_help(&MekaError::RetryableProvider {
            message: "529".to_string(),
            retry_after: None,
            server_error_on_completion: false,
        }));
        assert!(inbox_retry_can_help(&MekaError::StreamError(
            "cut".to_string()
        )));
        assert!(inbox_retry_can_help(&MekaError::ProviderExhausted {
            message: "402".to_string(),
            resets_after: None,
        }));
        assert!(inbox_retry_can_help(&MekaError::McpTurnGated {
            servers: vec![("exa".to_string(), "down".to_string())],
        }));
        assert!(inbox_retry_can_help(&MekaError::Database(
            "database is locked".to_string()
        )));
        assert!(inbox_retry_can_help(&MekaError::Io(std::io::Error::other(
            "no space left on device"
        ))));
        assert!(!inbox_retry_can_help(&MekaError::Provider(
            "401".to_string()
        )));
        assert!(!inbox_retry_can_help(&MekaError::InvalidRequest(
            "bad".to_string()
        )));
        assert!(!inbox_retry_can_help(&MekaError::Config(
            "text-only".to_string()
        )));
        assert!(!inbox_retry_can_help(&MekaError::ContextOverflow(
            "too long".to_string()
        )));
    }

    /// The outcome poller runs every tick on every resident session, so it reads what there is to
    /// deliver before it takes the conversation lock, and admits the hold the moment it has it:
    /// the lock refuses a client's turn as in flight, and `turn_in_flight` only knows what was
    /// admitted.
    #[test]
    fn the_outcome_poller_takes_the_lock_only_for_what_wakes_a_host() {
        let poller = include_str!("scheduler.rs")
            .split("pub(crate) async fn deliver_ready_outcomes")
            .nth(1)
            .expect("the poller")
            .split("\n}\n")
            .next()
            .expect("the poller ends");
        let position = |needle: &str| poller.find(needle).unwrap_or_else(|| panic!("{needle}"));
        let listed = position("list_undelivered_background_tasks(");
        let decided = position("wakes_a_host()");
        let locked = position("conversation.try_lock()");
        let admitted = position("hooks.admit(");
        let checked = position("still_resident(");
        let stamped = position("mark_background_tasks_delivered(");
        assert!(
            listed < decided && decided < locked,
            "the read and the decision come first"
        );
        assert!(
            locked < admitted && admitted < checked,
            "admitted as soon as the lock is held"
        );
        assert!(admitted < stamped, "and ahead of the one-way stamp");
    }
    /// A fire that completed before the drain reached it is complete: its reply is in the store,
    /// and a deferred occurrence would deliver it again on the next start. Only the drain's own
    /// interruption hands the occurrence back.
    #[test]
    fn only_a_fire_the_drain_cut_short_is_deferred() {
        use super::deferred_by_the_drain;
        assert!(deferred_by_the_drain(
            true,
            &Err(crate::error::MekaError::Interrupted)
        ));
        assert!(!deferred_by_the_drain(true, &Ok(())));
        assert!(!deferred_by_the_drain(
            true,
            &Err(crate::error::MekaError::Provider("down".to_string()))
        ));
        assert!(!deferred_by_the_drain(
            false,
            &Err(crate::error::MekaError::Interrupted)
        ));
    }

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
            .find("TurnInput::scheduled(")
            .expect("the fire admits its prompt");
        let claimed = wakeup
            .find("claim_undelivered_outcomes(")
            .expect("the fire claims the outcomes that ride it");
        assert!(
            admitted < claimed,
            "a scheduled fire must admit its prompt before claiming outcomes; found \
             admit@{admitted} claim@{claimed}"
        );
        assert!(
            wakeup.contains(".originating(crate::provider::TurnOrigin::Scheduled)"),
            "a fired job's words are typed by nobody, so the fire must restate their origin"
        );

        let typed = include_str!("tui.rs")
            .split("HostEvent::UserInput { parts, images } => {")
            .nth(1)
            .expect("the screen's typed-prompt arm")
            .split("HostEvent::Command(command) => {")
            .next()
            .expect("splitting always yields a first part");
        let admitted = typed
            .find("TurnInput::sent(")
            .expect("the host admits the typed prompt");
        let claimed = typed
            .find("collect_background_outcomes(")
            .expect("the host claims the outcomes that ride it");
        assert!(
            admitted < claimed,
            "the host must admit a typed prompt before claiming outcomes; found \
             admit@{admitted} claim@{claimed}"
        );
    }
}
