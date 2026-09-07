//! What the scheduler remembers between sweeps and the predicates that decide whether a job is
//! held back: pure over the job and the live permission, so every surface that reports on a job
//! (the model's `[Scheduled]` block, `meka schedule list`, the HTTP jobs API) reaches the same
//! answer as the sweep that would fire it.

use chrono::{DateTime, Utc};

use super::*;

/// Why this job will not fire right now, phrased for the model, or `None` if it will.
///
/// The whole answer, not just the gate's half: a session at `none` withholds every job, gated or
/// not, and an *ungated* job is exactly the case a gate-shaped question misses. Without this an
/// ungated reminder on such a session read as perfectly healthy on every surface while never
/// firing -- the same "held and healthy look identical" problem the gate marker exists to solve,
/// one level up, and a disagreement between the creation door (which accepts) and the fire door
/// (which refuses).
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
/// and "I cannot tell" is not "it is fine". Collapsing them is right for the surfaces that render a
/// *sentence* -- there is nothing to say -- and wrong for one that renders a *column*, where the
/// empty cell beside a populated one reads as a verdict.
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
    // First, because it is the most specific, the one nothing else will explain, and the only
    // verdict here that needs no permission level. A parked job has a healthy gate and an adequate
    // session, so every other question answers "it will fire" while the fire door refuses it on
    // every sweep -- and asking it before the level means a reader that could not establish one
    // still reports the job it can see is dead.
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
    let Some(gate) = job.gate.as_ref() else {
        return Withheld::No;
    };
    let Some((refusal, level)) = gate_withheld_reason(gate, live, tools) else {
        return match memory.standing_probe_failure(job) {
            Some(reason) => Withheld::Yes(reason),
            None => Withheld::No,
        };
    };
    // `ToolUnavailable` is the one refusal this function will not report on a guess, because it is
    // the one that can mean "I cannot tell" rather than "it is broken".
    //
    // Two readers hit that. A caller with no dispatcher at all -- `meka schedule list`, which has
    // no MCP manager -- would otherwise report *every* tool gate as dead, libelling healthy jobs to
    // the one audience that cannot check. And a server still completing its first handshake is not
    // a verdict yet: reporting one marks a healthy job dead for the second it takes and announces
    // it alive again a turn later, on every start and every reconnect.
    //
    // The fire door is unaffected in both cases: it still declines the occurrence, because a gate
    // whose tool cannot be resolved cannot be evaluated. Silence here is about what we are entitled
    // to *say*, not about what runs.
    if matches!(refusal, GateRefusal::ToolUnavailable)
        && let GateProbe::Tool { name, .. } = &gate.probe
        && tools.is_none_or(|tools| tools.is_still_connecting(name))
    {
        return Withheld::Undetermined;
    }
    Withheld::Yes(refusal.explain(&gate.probe, level))
}
/// What a job's row said when a probe failure was counted against it.
///
/// Neither half is written by the failing path -- it persists no `fired_at` and no baseline,
/// deliberately -- so either of them moving is proof that *something else* evaluated this gate
/// afterwards and got an answer. See [`SchedulerMemory::standing_probe_failure`].
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
/// How many consecutive failures before a broken probe is reported as a standing condition.
///
/// Two, not one. A single failure is as often a blip as a break -- a server restarting, a network
/// blip, a command losing a race -- and the marker says "this job is not firing", which is a
/// statement about a state rather than an event. A watcher that recovers on its next evaluation
/// never earns one.
///
/// Two *evaluations*, not two ticks, and the difference grew teeth when a failed probe stopped
/// being retried every `poll_interval`. Evaluations are now one occurrence apart for a recurring
/// job and one `claim_lease` apart for a job with no next occurrence, so the marker arrives after
/// two periods rather than twenty seconds: twelve hours for a `6h` job, two hours for a one-shot at
/// the default lease. That is the cost of not re-probing a broken gate at tick cadence, and it is
/// the right way round -- the alternative spent real work to reach the same conclusion sooner --
/// but it does mean this constant no longer implies anything about wall-clock latency.
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
    /// In memory rather than on the row, which is a real limitation and the right trade.
    /// Persisting it would mean a schema change and a write on every failed evaluation, to
    /// report a condition that a restart re-establishes within one poll interval. The cost is
    /// that the reporting surface and the scheduler have to be the same process to agree: they
    /// are for the REPL, ACP and `meka serve`, which are the three that both run jobs and
    /// render `[Scheduled]`. `meka schedule list` is a separate process and sees nothing here,
    /// which is the same thing it already does with tool gates it cannot resolve.
    ///
    /// The [`ProbeWitness`] is what keeps that from becoming a *wrong* answer rather than a
    /// missing one. Only the host that wins `claim_occurrence` evaluates, and which host that
    /// is, is a race between their tickers; a host that recorded two failures and then stopped
    /// winning would go on telling its resident session's model that a job firing every hour
    /// is dead, forever, because nothing else in this process ever re-enters the counting
    /// path.
    probe_failures: std::sync::Mutex<ProbeFailures>,
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
    /// Two things fill `attempts`, and they have opposite remedies: a prompt that takes the host
    /// down, and a gate probe that can never answer. The probe's error is the discriminator
    /// when this process has one, but `SchedulerMemory` is per-process, so a restart -- which
    /// is exactly what an operator does after noticing a job has gone inert -- loses it, and
    /// `meka schedule list` never had it. Asserting the commoner cause from that absence
    /// produced the worst outcome available: telling someone whose MCP server was misconfigured
    /// that their prompt crashes meka, with a remedy aimed at the wrong artifact, on the
    /// model's own `[Scheduled]` block.
    ///
    /// So absence is treated as absence. The row does still settle it in one direction: a job with
    /// no gate has no probe that could have failed, so a crash is the only thing left and can
    /// be named outright.
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

    /// Why an *authorized* gate is still not firing: its probe keeps breaking.
    ///
    /// Authority is not the only way a watcher dies, and it is not the commonest. A server that
    /// changed its schema, a command that was uninstalled, a pointer into a result that stopped
    /// being JSON: each produces a gate that errors on every evaluation, and each looks from
    /// the model's side exactly like a healthy watcher with nothing to report. The marker
    /// existed for that indistinguishability and covered only half of it.
    pub(crate) fn standing_probe_failure(&self, job: &ScheduledJob) -> Option<String> {
        let (failures, error, witness) = {
            let held = crate::sync::lock(&self.probe_failures);
            held.get(&job.id).cloned()?
        };
        // The verdict is this process's, but the job is not. Only the host that wins
        // `claim_occurrence` evaluates, so a second `meka serve` on the same store can take
        // over every occurrence and heal the gate without this process ever hearing.
        // Nothing here re-enters the counting path in that case, so without this check the
        // marker stood forever: the model was told, every turn, that a job firing hourly
        // was dead.
        //
        // Neither half of the witness is written by the failing path -- it persists no `fired_at`
        // and no baseline, deliberately -- so either of them having moved is proof of a successful
        // evaluation since. The gap it cannot close is a gate that keeps evaluating, keeps
        // declining, and keeps producing the identical output: an unchanged row cannot testify to
        // anything.
        if witness != probe_witness(job) {
            self.clear_probe_failure(&job.id);
            return None;
        }
        if failures < PROBE_FAILURES_BEFORE_REPORTING {
            return None;
        }
        // No count in the sentence, deliberately.
        //
        // Every reader of this compares it by equality. `render_world_state_diff` announces a job
        // to the model when its withheld reason *changes*, so a running total made the
        // reason change on every failed evaluation and the model was told "can no longer
        // fire: … 7 evaluations", then 8, then 9, for as long as the gate stayed broken.
        // `context.rs` already leaves next-fire times out of the snapshot for exactly this
        // reason; a counter is the same mistake wearing a different hat.
        //
        // The number is not lost: it is in the `warn!` at each failure, where an event belongs, and
        // `-v` shows it. What the model needs is the standing fact and what to do about it, and
        // that does not change between the second failure and the two-hundredth.
        Some(format!(
            "its gate keeps failing and cannot say whether to fire: {}. Fix the check by recreating \
             the job, or cancel it",
            elide_for_message(&error)
        ))
    }

    /// True the first time a job is held back for *this* reason, false while that reason persists.
    ///
    /// Keyed by job and by reason, not by job alone. The two refusals are different facts with
    /// different remedies -- "the session is at `none`" and "this gate needs `unrestricted`" -- and
    /// a job that moves between them has changed in a way the operator acts on. Keyed by job
    /// alone the second condition arrived silently, because the entry was already there:
    /// dropping a session from `read` to `none` said nothing at all, and raising it back to
    /// `read` said nothing either.
    ///
    /// Entries are dropped when the job is authorized again *or* when it stops being a job meka can
    /// see, so a canceled-while-declined job does not sit here for the life of the process. It is
    /// one short string per held-back job, which is small, but a long-lived `meka serve` has no
    /// other bound on it.
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

    /// Everything recorded against a job, on the job going away.
    pub(crate) fn forget(&self, job_id: &str) {
        self.clear_probe_failure(job_id);
        self.clear_permission_decline(job_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The held-back explanation is a fact about a standing state, so it is said once.
    ///
    /// The sweep re-evaluates every due job, and a session parked below write does not change
    /// between sweeps. An `every = "1m"` job wrote the full explanation every minute for as long as
    /// it stayed there, which turns the one line an operator needs to see into the noise they stop
    /// reading. Restoring the authority arms it again, so the next withdrawal is not swallowed.
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

        // A different reason for the same job. Keyed by job alone this was silent, so dropping a
        // session from `read` to `none` -- which stops *every* job, not just the gated one --
        // arrived with no line at all, and raising it back to `read` said nothing either.
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
