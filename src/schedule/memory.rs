//! What the scheduler remembers between sweeps and the predicates that decide whether a schedule is
//! held back: pure over the schedule and the live permission, so every surface that reports on a
//! schedule (the model's `[Scheduled]` block, `meka schedule list`, the HTTP schedules API) reaches
//! the same answer as the sweep that would fire it.

use chrono::{DateTime, Utc};

use super::*;

/// Why this schedule will not fire right now, phrased for the model, or `None` if it will.
///
/// The whole answer, not just the gate's half: a session at `none` withholds every schedule, gated
/// or not, and an ungated schedule is exactly the case a gate-shaped question misses, so it would
/// read as healthy on every surface while never firing.
pub(crate) fn schedule_withheld_reason(
    memory: &SchedulerMemory,
    schedule: &Schedule,
    live: crate::permission::Permission,
    tools: Option<&dyn GateTools>,
) -> Option<String> {
    match schedule_withheld(memory, schedule, Some(live), tools) {
        Withheld::Yes(reason) => Some(reason),
        Withheld::No | Withheld::Undetermined => None,
    }
}
/// What a reader is entitled to say about whether a schedule will fire.
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
/// The three-way form of [`schedule_withheld_reason`], for a reader that can express "I cannot
/// tell".
pub(crate) fn schedule_withheld(
    memory: &SchedulerMemory,
    schedule: &Schedule,
    live: Option<crate::permission::Permission>,
    tools: Option<&dyn GateTools>,
) -> Withheld {
    // First, because it is the most specific and the only verdict here that needs no permission
    // level: a parked schedule has a healthy gate and an adequate session, so every other question
    // answers "it will fire", and a reader that could not establish a level still reports the
    // schedule it can see is dead.
    if schedule.attempts >= MAX_CLAIM_ATTEMPTS {
        return Withheld::Yes(memory.parked_reason(schedule));
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
    let Some(gate) = schedule.gate.as_ref() else {
        return Withheld::No;
    };
    let Some((refusal, level)) = gate_withheld_reason(gate, live, tools) else {
        return match memory.standing_probe_failure(schedule) {
            Some(reason) => Withheld::Yes(reason),
            None => Withheld::No,
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
/// What a schedule's row said when a probe failure was counted against it.
///
/// Neither half is written by the failing path (it persists no `fired_at` and no baseline), so
/// either of them moving is proof that something else evaluated this gate afterwards and got an
/// answer. See [`SchedulerMemory::standing_probe_failure`].
pub(crate) type ProbeWitness = (Option<DateTime<Utc>>, Option<String>);
/// The witness for `schedule` as its row stands now.
pub(crate) fn probe_witness(schedule: &Schedule) -> ProbeWitness {
    (
        schedule.last_fired_at,
        schedule
            .gate
            .as_ref()
            .and_then(|gate| gate.last_output.clone()),
    )
}
/// How many consecutive failures, the last one's reason, and the row they were counted against.
pub(crate) type ProbeFailures = std::collections::HashMap<String, (u32, String, ProbeWitness)>;
/// How many consecutive failures before a broken probe is reported as a standing condition.
///
/// Two, not one: a single failure is as often a blip (a server restarting, a command losing a
/// race) as a break, and the marker states a standing condition, not an event.
///
/// Two evaluations, not two ticks. Evaluations are one occurrence apart for a recurring schedule
/// and one `claim_lease` apart for a schedule with no next occurrence, so the marker arrives after
/// two periods (twelve hours for a `6h` schedule) and this constant says nothing about wall-clock
/// latency.
pub(crate) const PROBE_FAILURES_BEFORE_REPORTING: u32 = 2;
/// What this process has learned about the schedules in one store, and the identity its leases
/// carry.
///
/// Owned by the [`crate::store::Store`]: every reader of these verdicts already holds one, and
/// deleting a schedule there is the door every cancel path converges on, which is where the entries
/// for a schedule that no longer exists have to go. Per store handle rather than per process for
/// the same reason the lease owner is: two hosts on one database are two of these, and what one
/// learned says nothing about the other.
pub(crate) struct SchedulerMemory {
    /// Schedules currently held back because their gate's authority was withdrawn.
    ///
    /// Exists only to keep the explanation to once per episode. The check runs on every sweep, and
    /// the state it reports does not change between them, so warning per evaluation turns one
    /// fact into a line a minute for as long as the session stays below write.
    permission_declined: std::sync::Mutex<std::collections::HashMap<String, String>>,
    /// Consecutive failed probe evaluations per schedule, with the last reason and the row as it
    /// stood.
    ///
    /// In memory rather than on the row: persisting it would mean a schema change and a write on
    /// every failed evaluation, to report a condition a restart re-establishes within one poll
    /// interval. The cost is that `meka schedule list`, a separate process, sees nothing here.
    ///
    /// The [`ProbeWitness`] keeps that from becoming a wrong answer rather than a missing one:
    /// only the host that wins `claim_occurrence` evaluates, so a host that recorded two failures
    /// and then stopped winning would otherwise report the schedule dead forever.
    probe_failures: std::sync::Mutex<ProbeFailures>,
    /// Schedules whose gate this process is running now, under its lease.
    ///
    /// A sweep stopped inside a gate leaves that schedule here, which is how a host on its way out
    /// tells the lease the gate held from one the sweep left to run out on purpose: the wait of a
    /// one-shot whose gate could not be evaluated, or of a fire the store refused.
    gates_running: std::sync::Mutex<std::collections::HashSet<String>>,
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
            gates_running: std::sync::Mutex::default(),
            owner: uuid::Uuid::new_v4().to_string(),
        }
    }
}

impl SchedulerMemory {
    /// The identity this handle's leases carry.
    pub(crate) fn owner(&self) -> &str {
        &self.owner
    }

    /// Why a parked schedule stopped, said only as far as the row can support.
    ///
    /// Two things fill `attempts`, with opposite remedies: a host that stops, or a store that
    /// refuses the write, before the fire is queued, and a gate probe that can never answer. The
    /// probe's error is the discriminator when this process has one, but `SchedulerMemory` is
    /// per-process, so a restart loses it and `meka schedule list` never had it; guessing the
    /// commoner cause from that absence aims the remedy at the wrong artifact. The row still
    /// settles it in one direction: a schedule with no gate has no probe that could have failed.
    pub(crate) fn parked_reason(&self, schedule: &Schedule) -> String {
        let opening = format!("{} claims ended without queuing a fire", schedule.attempts);
        match (self.probe_failure(&schedule.id), schedule.gate.is_some()) {
            (Some((_, error)), _) => format!(
                "{}, because its gate could not be evaluated: {}. It is no longer retried; fix the \
                 check by recreating the schedule, or cancel it",
                opening,
                elide_for_message(&error)
            ),
            (None, false) => format!(
                "{opening}, and it has no gate that could have failed, so the host stopped or its store \
                 refused the write each time. It is no longer retried; recreate it, or cancel it"
            ),
            (None, true) => format!(
                "{opening}, so either its gate cannot be evaluated or the host stopped before queuing \
                 its fire; this process no longer has the record that would say which. It is no longer \
                 retried. Run the gate's check by hand, and cancel or recreate the schedule"
            ),
        }
    }

    /// Why an authorized gate is still not firing: its probe keeps breaking.
    ///
    /// A server that changed its schema, a command that was uninstalled, a pointer into a result
    /// that stopped being JSON: each errors on every evaluation, and each looks from the model's
    /// side exactly like a healthy watcher with nothing to report.
    pub(crate) fn standing_probe_failure(&self, schedule: &Schedule) -> Option<String> {
        let (failures, error, witness) = {
            let held = crate::sync::lock(&self.probe_failures);
            held.get(&schedule.id).cloned()?
        };
        // The verdict is this process's, but the schedule is not: only the host that wins
        // `claim_occurrence` evaluates, so a second `meka serve` on the same store can heal the
        // gate without this process ever re-entering the counting path. Neither half of the
        // witness is written by the failing path, so either having moved is proof of a successful
        // evaluation since. A gate that keeps declining with identical output cannot be told
        // apart this way.
        if witness != probe_witness(schedule) {
            self.clear_probe_failure(&schedule.id);
            return None;
        }
        if failures < PROBE_FAILURES_BEFORE_REPORTING {
            return None;
        }
        // No count in the sentence: `render_world_state_diff` announces a schedule to the model
        // when its withheld reason changes, so a running total would re-announce it on every failed
        // evaluation. The count is in the `warn!` at each failure.
        Some(format!(
            "its gate keeps failing and cannot say whether to fire: {}. Fix the check by recreating \
             the schedule, or cancel it",
            elide_for_message(&error)
        ))
    }

    /// True the first time a schedule is held back for *this* reason, false while that reason
    /// persists.
    ///
    /// Keyed by schedule and by reason, not by schedule alone: "the session is at `none`" and "this
    /// gate needs `unrestricted`" have different remedies, and keyed by schedule alone a move
    /// between them would arrive silently because the entry was already there.
    ///
    /// Entries are dropped when the schedule is authorized again or when it stops being a schedule
    /// Meka can see, because a long-lived `meka serve` has no other bound on them.
    pub(crate) fn declined_for_permission_first_time(
        &self,
        schedule_id: &str,
        reason: &str,
    ) -> bool {
        let mut held = crate::sync::lock(&self.permission_declined);
        match held.get(schedule_id) {
            Some(previous) if previous == reason => false,
            _ => {
                held.insert(schedule_id.to_string(), reason.to_string());
                true
            }
        }
    }

    /// Count one failed evaluation against a schedule and return the running total.
    pub(crate) fn record_probe_failure(&self, schedule: &Schedule, error: &str) -> u32 {
        let mut held = crate::sync::lock(&self.probe_failures);
        let entry = held
            .entry(schedule.id.clone())
            .or_insert_with(|| (0, String::new(), (None, None)));
        entry.0 = entry.0.saturating_add(1);
        entry.1 = error.to_string();
        entry.2 = probe_witness(schedule);
        let failures = entry.0;
        drop(held);
        failures
    }

    /// What this process knows about a schedule's recent probe failures: how many, and why the last
    /// one failed.
    pub(crate) fn probe_failure(&self, schedule_id: &str) -> Option<(u32, String)> {
        let held = crate::sync::lock(&self.probe_failures);
        held.get(schedule_id)
            .map(|(failures, error, _)| (*failures, error.clone()))
    }

    /// Forget a schedule's probe failures, on an evaluation that worked or on the schedule going
    /// away.
    pub(crate) fn clear_probe_failure(&self, schedule_id: &str) {
        crate::sync::lock(&self.probe_failures).remove(schedule_id);
    }

    /// Forget a schedule's held-back state, so the next withdrawal is announced again.
    pub(crate) fn clear_permission_decline(&self, schedule_id: &str) {
        crate::sync::lock(&self.permission_declined).remove(schedule_id);
    }

    /// Note that `schedule_id`'s gate is about to run under this process's lease.
    pub(crate) fn gate_started(&self, schedule_id: &str) {
        crate::sync::lock(&self.gates_running).insert(schedule_id.to_string());
    }

    /// Note that `schedule_id`'s gate returned, whatever it answered.
    pub(crate) fn gate_ended(&self, schedule_id: &str) {
        crate::sync::lock(&self.gates_running).remove(schedule_id);
    }

    /// The schedules whose gate started and never returned, which is what a sweep that has been
    /// stopped was running. The call forgets them.
    pub(crate) fn take_gates_cut_short(&self) -> Vec<String> {
        crate::sync::lock(&self.gates_running).drain().collect()
    }

    /// Everything recorded against a schedule, on the schedule going away.
    pub(crate) fn forget(&self, schedule_id: &str) {
        self.clear_probe_failure(schedule_id);
        self.clear_permission_decline(schedule_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The held-back explanation is a fact about a standing state, so it is said once: an
    /// `every = "1m"` schedule would otherwise write it every minute. Restoring the authority arms
    /// it again, so the next withdrawal is not swallowed.
    #[test]
    fn a_schedule_held_back_for_permission_explains_itself_once_per_downgrade() {
        let schedule = format!("schedule-{}", uuid::Uuid::new_v4());
        let gate_bar = "a gate command runs unattended with no sandbox";
        let memory = SchedulerMemory::default();

        assert!(
            memory.declined_for_permission_first_time(&schedule, gate_bar),
            "the first sweep of a downgrade has to say why"
        );
        assert!(
            !memory.declined_for_permission_first_time(&schedule, gate_bar),
            "and the ones after it must not repeat"
        );
        assert!(
            !memory.declined_for_permission_first_time(&schedule, gate_bar),
            "however many there are"
        );

        // A different reason for the same schedule: keyed by schedule alone, dropping a session
        // from `read` to `none` (which stops every schedule, not just the gated one) would arrive
        // with no line.
        assert!(
            memory.declined_for_permission_first_time(&schedule, "unattended-work:none"),
            "a schedule held for a different reason is a different fact, and the remedy differs too"
        );
        assert!(
            !memory.declined_for_permission_first_time(&schedule, "unattended-work:none"),
            "and that one settles into silence in its turn"
        );
        assert!(
            memory.declined_for_permission_first_time(&schedule, gate_bar),
            "including on the way back"
        );

        memory.clear_permission_decline(&schedule);
        assert!(
            memory.declined_for_permission_first_time(&schedule, gate_bar),
            "a later withdrawal is a new fact and is announced again"
        );
    }
}
