//! What the scheduler remembers between sweeps and the predicates that decide whether a job is
//! held back: pure over the job and the live permission, so every surface that reports on a job
//! (the model's `[Scheduled]` block, `meka schedule list`, the HTTP jobs API) reaches the same
//! answer as the sweep that would fire it.

use chrono::{DateTime, Utc};

use super::*;

/// Why this job will not fire right now, phrased for the model, or `None` if it will.
///
/// The whole answer, not just the gate's half: a session at `none` withholds every job, gated or
/// not, and an ungated job is exactly the case a gate-shaped question misses, so it would read as
/// healthy on every surface while never firing.
pub(crate) fn job_withheld_reason(
    memory: &SchedulerMemory,
    job: &ScheduledJob,
    live: crate::permission::Permission,
    tools: Option<&dyn GateTools>,
) -> Option<String> {
    match job_withheld(memory, job, Some(live), tools) {
        Withheld::Yes(reason) => Some(reason),
        Withheld::No | Withheld::Undetermined => None,
    }
}
/// What a reader is entitled to say about whether a job will fire.
///
/// Three answers rather than two, because a reader without a dispatcher cannot resolve a tool gate
/// and "I cannot tell" is not "it is fine". Collapsing them is right for a surface that renders a
/// sentence (there is nothing to say) and wrong for one that renders a column, where the empty
/// cell beside a populated one reads as a verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Withheld {
    /// It will fire, as far as this reader can establish.
    No,
    /// It will not, for this reason.
    Yes(String),
    /// Not establishable by this reader: a tool gate it has no dispatcher for or whose server is
    /// still connecting, or a session whose permission level it could not resolve at all.
    Undetermined,
}
/// The three-way form of [`job_withheld_reason`], for a reader that can express "I cannot tell".
pub(crate) fn job_withheld(
    memory: &SchedulerMemory,
    job: &ScheduledJob,
    live: Option<crate::permission::Permission>,
    tools: Option<&dyn GateTools>,
) -> Withheld {
    // First, because it is the most specific and the only verdict here that needs no permission
    // level: a parked job has a healthy gate and an adequate session, so every other question
    // answers "it will fire", and a reader that could not establish a level still reports the job
    // it can see is dead.
    if job.attempts >= MAX_CLAIM_ATTEMPTS {
        return Withheld::Yes(memory.parked_reason(job));
    }
    let Some(live) = live else {
        return Withheld::Undetermined;
    };
    if !live.allows_unattended_work() {
        return Withheld::Yes(format!(
            "the session is at {live}, where nothing is executable, so a scheduled turn could neither \
             act on this nor cancel it"
        ));
    }
    // Asked last: every other verdict stops the fire door before a host is offered the job, so
    // while one holds, what the host's last fire found is not why the job is not firing.
    let unrunnable = || {
        memory
            .unrunnable_reason(job)
            .map_or(Withheld::No, Withheld::Yes)
    };
    let Some(gate) = job.gate.as_ref() else {
        return unrunnable();
    };
    let Some((refusal, level)) = gate_withheld_reason(gate, live, tools) else {
        return match memory.standing_probe_failure(job) {
            Some(reason) => Withheld::Yes(reason),
            None => unrunnable(),
        };
    };
    // `ToolUnavailable` is the one refusal that can mean "I cannot tell" rather than "it is
    // broken": a caller with no dispatcher (`meka schedule list` has no MCP manager) would
    // otherwise report every tool gate as dead, and a server still completing its first handshake
    // is not a verdict yet. The fire door still declines the occurrence either way.
    if matches!(refusal, GateRefusal::ToolUnavailable)
        && let GateProbe::Tool { name, .. } = &gate.probe
        && tools.is_none_or(|tools| tools.is_still_connecting(name))
    {
        return Withheld::Undetermined;
    }
    Withheld::Yes(refusal.explain(&gate.probe, level))
}
/// Why a host that owns a job could not run its fire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Unrunnable {
    /// The session could not be reached; the error.
    SessionUnreachable(String),
    /// The session's recorded profile or project did not resolve; the error.
    BindingsUnresolved(String),
    /// The job's prompt is empty.
    EmptyPrompt,
}
impl Unrunnable {
    /// What the reporting surfaces say while the cause stands, or `None` when the cause is the
    /// job's own.
    ///
    /// A session that cannot be reached and bindings that do not resolve are facts about the
    /// installation, so the claim does not count toward parking and the job is retried once a
    /// lease until they are fixed. An empty prompt is the job's: no retry fixes it, so its claim
    /// counts as a crashing turn's does.
    pub(crate) fn standing_reason(&self) -> Option<String> {
        match self {
            Self::SessionUnreachable(error) => Some(format!(
                "its last fire did not run because its session could not be reached ({}); it is \
                 retried every `[schedule].claim_lease`",
                elide_for_message(error)
            )),
            Self::BindingsUnresolved(error) => Some(format!(
                "its last fire did not run because its session's profile or project did not \
                 resolve ({}); it is retried every `[schedule].claim_lease`",
                elide_for_message(error)
            )),
            Self::EmptyPrompt => None,
        }
    }
}
/// What a job's row said when a probe failure was counted against it.
///
/// Neither half is written by the failing path (it persists no `fired_at` and no baseline), so
/// either of them moving is proof that something else evaluated this gate afterwards and got an
/// answer. See [`SchedulerMemory::standing_probe_failure`].
pub(crate) type ProbeWitness = (Option<DateTime<Utc>>, Option<String>);
/// The witness for `job` as its row stands now.
pub(crate) fn probe_witness(job: &ScheduledJob) -> ProbeWitness {
    (
        job.last_fired_at,
        job.gate.as_ref().and_then(|gate| gate.last_output.clone()),
    )
}
/// How many consecutive failures, the last one's reason, and the row they were counted against.
pub(crate) type ProbeFailures = std::collections::HashMap<String, (u32, String, ProbeWitness)>;
/// Why a fire could not run, and the row it was recorded against: its `last_fired_at` and the
/// occurrence, its `next_fire_at`.
type UnrunnableFires =
    std::collections::HashMap<String, (String, (Option<DateTime<Utc>>, DateTime<Utc>))>;
/// How many consecutive failures before a broken probe is reported as a standing condition.
///
/// Two, not one: a single failure is as often a blip (a server restarting, a command losing a
/// race) as a break, and the marker states a standing condition, not an event.
///
/// Two evaluations, not two ticks. Evaluations are one occurrence apart for a recurring job and
/// one `claim_lease` apart for a job with no next occurrence, so the marker arrives after two
/// periods (twelve hours for a `6h` job) and this constant says nothing about wall-clock latency.
pub(crate) const PROBE_FAILURES_BEFORE_REPORTING: u32 = 2;
/// What this process has learned about the jobs in one store, and the identity its leases carry.
///
/// Owned by the [`crate::store::Store`]: every reader of these verdicts already holds one, and
/// deleting a job there is the door every cancel path converges on, which is where the entries for
/// a job that no longer exists have to go. Per store handle rather than per process for the same
/// reason the lease owner is: two hosts on one database are two of these, and what one learned
/// says nothing about the other.
pub(crate) struct SchedulerMemory {
    /// Jobs currently held back because their gate's authority was withdrawn.
    ///
    /// Exists only to keep the explanation to once per episode. The check runs on every sweep, and
    /// the state it reports does not change between them, so warning per evaluation turns one
    /// fact into a line a minute for as long as the session stays below write.
    permission_declined: std::sync::Mutex<std::collections::HashMap<String, String>>,
    /// Consecutive failed probe evaluations per job, with the last reason and the row as it stood.
    ///
    /// In memory rather than on the row: persisting it would mean a schema change and a write on
    /// every failed evaluation, to report a condition a restart re-establishes within one poll
    /// interval. The cost is that `meka schedule list`, a separate process, sees nothing here.
    ///
    /// The [`ProbeWitness`] keeps that from becoming a wrong answer rather than a missing one:
    /// only the host that wins `claim_occurrence` evaluates, so a host that recorded two failures
    /// and then stopped winning would otherwise report the job dead forever.
    probe_failures: std::sync::Mutex<ProbeFailures>,
    /// Why this host could not run each job's last fire, with the job's `last_fired_at` and
    /// `next_fire_at` then.
    ///
    /// In memory for the reason the probe failures are, and witnessed the same way: a turn
    /// delivered since, by this host or another, moves `last_fired_at`, and an occurrence closed
    /// without one (a gate that declines on the retry) moves `next_fire_at`; an unrunnable fire
    /// writes neither.
    unrunnable: std::sync::Mutex<UnrunnableFires>,
    /// A token identifying this process to the claim column, for the life of the process.
    ///
    /// Per store handle rather than per sweep: the point is to tell *my* lease from someone
    /// else's, and a value that changed between the claim and the write scoped to it would
    /// defeat both. Random rather than derived from the pid, because a pid is reused and a
    /// reused pid would let a fresh process finish a dead one's claim.
    owner: String,
}

impl Default for SchedulerMemory {
    fn default() -> Self {
        Self {
            permission_declined: std::sync::Mutex::default(),
            probe_failures: std::sync::Mutex::default(),
            unrunnable: std::sync::Mutex::default(),
            owner: uuid::Uuid::new_v4().to_string(),
        }
    }
}

impl SchedulerMemory {
    /// The identity this handle's leases carry.
    pub(crate) fn owner(&self) -> &str {
        &self.owner
    }

    /// Why a parked job stopped, said only as far as the row can support.
    ///
    /// Two things fill `attempts`, with opposite remedies: a prompt that takes the host down, and a
    /// gate probe that can never answer. The probe's error is the discriminator when this process
    /// has one, but `SchedulerMemory` is per-process, so a restart loses it and `meka schedule
    /// list` never had it; guessing the commoner cause from that absence aims the remedy at the
    /// wrong artifact. The row still settles it in one direction: a job with no gate has no probe
    /// that could have failed.
    pub(crate) fn parked_reason(&self, job: &ScheduledJob) -> String {
        let opening = format!("{} claims ended without delivering", job.attempts);
        match (self.probe_failure(&job.id), job.gate.is_some()) {
            (Some((_, error)), _) => format!(
                "{}, because its gate could not be evaluated: {}. It is no longer retried; fix the \
                 check by recreating the job, or cancel it",
                opening,
                elide_for_message(&error)
            ),
            (None, false) => format!(
                "{opening} or handing back, and it has no gate that could have failed, so the host died each \
                 time. It is no longer retried. Cancel it, or recreate it with a prompt that does not \
                 take the process down"
            ),
            (None, true) => format!(
                "{opening} or handing back, so either its gate cannot be evaluated or the turn takes the host \
                 down; this process no longer has the record that would say which. It is no longer \
                 retried. Run the gate's check by hand, and cancel or recreate the job"
            ),
        }
    }

    /// Why an authorized gate is still not firing: its probe keeps breaking.
    ///
    /// A server that changed its schema, a command that was uninstalled, a pointer into a result
    /// that stopped being JSON: each errors on every evaluation, and each looks from the model's
    /// side exactly like a healthy watcher with nothing to report.
    pub(crate) fn standing_probe_failure(&self, job: &ScheduledJob) -> Option<String> {
        let (failures, error, witness) = {
            let held = crate::sync::lock(&self.probe_failures);
            held.get(&job.id).cloned()?
        };
        // The verdict is this process's, but the job is not: only the host that wins
        // `claim_occurrence` evaluates, so a second `meka serve` on the same store can heal the
        // gate without this process ever re-entering the counting path. Neither half of the
        // witness is written by the failing path, so either having moved is proof of a successful
        // evaluation since. A gate that keeps declining with identical output cannot be told
        // apart this way.
        if witness != probe_witness(job) {
            self.clear_probe_failure(&job.id);
            return None;
        }
        if failures < PROBE_FAILURES_BEFORE_REPORTING {
            return None;
        }
        // No count in the sentence: `render_world_state_diff` announces a job to the model when
        // its withheld reason changes, so a running total would re-announce it on every failed
        // evaluation. The count is in the `warn!` at each failure.
        Some(format!(
            "its gate keeps failing and cannot say whether to fire: {}. Fix the check by recreating \
             the job, or cancel it",
            elide_for_message(&error)
        ))
    }

    /// True the first time a job is held back for *this* reason, false while that reason persists.
    ///
    /// Keyed by job and by reason, not by job alone: "the session is at `none`" and "this gate
    /// needs `unrestricted`" have different remedies, and keyed by job alone a move between them
    /// would arrive silently because the entry was already there.
    ///
    /// Entries are dropped when the job is authorized again or when it stops being a job Meka can
    /// see, because a long-lived `meka serve` has no other bound on them.
    pub(crate) fn declined_for_permission_first_time(&self, job_id: &str, reason: &str) -> bool {
        let mut held = crate::sync::lock(&self.permission_declined);
        match held.get(job_id) {
            Some(previous) if previous == reason => false,
            _ => {
                held.insert(job_id.to_string(), reason.to_string());
                true
            }
        }
    }

    /// Count one failed evaluation against a job and return the running total.
    pub(crate) fn record_probe_failure(&self, job: &ScheduledJob, error: &str) -> u32 {
        let mut held = crate::sync::lock(&self.probe_failures);
        let entry = held
            .entry(job.id.clone())
            .or_insert_with(|| (0, String::new(), (None, None)));
        entry.0 = entry.0.saturating_add(1);
        entry.1 = error.to_string();
        entry.2 = probe_witness(job);
        let failures = entry.0;
        drop(held);
        failures
    }

    /// What this process knows about a job's recent probe failures: how many, and why the last one
    /// failed.
    pub(crate) fn probe_failure(&self, job_id: &str) -> Option<(u32, String)> {
        let held = crate::sync::lock(&self.probe_failures);
        held.get(job_id)
            .map(|(failures, error, _)| (*failures, error.clone()))
    }

    /// Forget a job's probe failures, on an evaluation that worked or on the job going away.
    pub(crate) fn clear_probe_failure(&self, job_id: &str) {
        crate::sync::lock(&self.probe_failures).remove(job_id);
    }

    /// Forget a job's held-back state, so the next withdrawal is announced again.
    pub(crate) fn clear_permission_decline(&self, job_id: &str) {
        crate::sync::lock(&self.permission_declined).remove(job_id);
    }

    /// Record why this host could not run `job`'s fire, against the row as it stands.
    pub(crate) fn record_unrunnable(&self, job: &ScheduledJob, reason: String) {
        crate::sync::lock(&self.unrunnable).insert(
            job.id.clone(),
            (reason, (job.last_fired_at, job.next_fire_at)),
        );
    }

    /// Why this host could not run `job`'s last fire, while that occurrence is still the one due
    /// and no turn has been delivered since.
    pub(crate) fn unrunnable_reason(&self, job: &ScheduledJob) -> Option<String> {
        let (reason, witness) = crate::sync::lock(&self.unrunnable).get(&job.id).cloned()?;
        if witness != (job.last_fired_at, job.next_fire_at) {
            self.clear_unrunnable(&job.id);
            return None;
        }
        Some(reason)
    }

    /// Forget why a job's fire could not run, on a fire that did or on the job going away.
    pub(crate) fn clear_unrunnable(&self, job_id: &str) {
        crate::sync::lock(&self.unrunnable).remove(job_id);
    }

    /// Everything recorded against a job, on the job going away.
    pub(crate) fn forget(&self, job_id: &str) {
        self.clear_probe_failure(job_id);
        self.clear_permission_decline(job_id);
        self.clear_unrunnable(job_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host that could not run a job's last fire stops saying so once a turn has been delivered,
    /// by this host or by another one, either of which moves `last_fired_at`.
    #[test]
    fn an_unrunnable_fire_stands_down_once_a_turn_is_delivered() {
        let memory = SchedulerMemory::default();
        let job = ScheduledJob {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: uuid::Uuid::nil(),
            schedule: Schedule::parse_every("1h").expect("parses"),
            prompt: "watch the thing".to_string(),
            gate: None,
            created_at: Utc::now(),
            last_fired_at: None,
            next_fire_at: Utc::now(),
            attempts: 0,
        };
        let cause = Unrunnable::BindingsUnresolved(
            "no profile named 'work' (configured: home)".to_string(),
        );
        memory.record_unrunnable(&job, cause.standing_reason().expect("the installation's"));
        assert!(memory.unrunnable_reason(&job).is_some());

        let mut delivered = job.clone();
        delivered.last_fired_at = Some(Utc::now());
        assert_eq!(
            memory.unrunnable_reason(&delivered),
            None,
            "a turn delivered since is proof the session can run it"
        );

        // The occurrence closed without a turn, as a gate that declines on the retry closes it:
        // the cause was that occurrence's, and the job is waiting for its next one.
        memory.record_unrunnable(&job, cause.standing_reason().expect("the installation's"));
        let mut advanced = job.clone();
        advanced.next_fire_at = job.next_fire_at + chrono::Duration::hours(1);
        assert_eq!(
            memory.unrunnable_reason(&advanced),
            None,
            "a cause recorded for an occurrence does not outlive it"
        );
    }

    /// The held-back explanation is a fact about a standing state, so it is said once: an
    /// `every = "1m"` job would otherwise write it every minute. Restoring the authority arms it
    /// again, so the next withdrawal is not swallowed.
    #[test]
    fn a_job_held_back_for_permission_explains_itself_once_per_downgrade() {
        let job = format!("job-{}", uuid::Uuid::new_v4());
        let gate_bar = "a gate command runs unattended with no sandbox";
        let memory = SchedulerMemory::default();

        assert!(
            memory.declined_for_permission_first_time(&job, gate_bar),
            "the first sweep of a downgrade has to say why"
        );
        assert!(
            !memory.declined_for_permission_first_time(&job, gate_bar),
            "and the ones after it must not repeat"
        );
        assert!(
            !memory.declined_for_permission_first_time(&job, gate_bar),
            "however many there are"
        );

        // A different reason for the same job: keyed by job alone, dropping a session from `read`
        // to `none` (which stops every job, not just the gated one) would arrive with no line.
        assert!(
            memory.declined_for_permission_first_time(&job, "unattended-work:none"),
            "a job held for a different reason is a different fact, and the remedy differs too"
        );
        assert!(
            !memory.declined_for_permission_first_time(&job, "unattended-work:none"),
            "and that one settles into silence in its turn"
        );
        assert!(
            memory.declined_for_permission_first_time(&job, gate_bar),
            "including on the way back"
        );

        memory.clear_permission_decline(&job);
        assert!(
            memory.declined_for_permission_first_time(&job, gate_bar),
            "a later withdrawal is a new fact and is announced again"
        );
    }
}
