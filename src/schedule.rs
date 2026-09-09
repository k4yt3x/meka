//! Scheduled wakeups: the agent arranging to be prompted again later.
//!
//! Every other turn meka runs originates outside the agent: a human typing, an editor sending
//! `session/prompt`, a client calling `POST /v1/sessions/{id}/turn`. This module supplies the one
//! trigger nothing else can, a timer: the agent's ability to say "wake me at 09:00" and have that
//! survive the process it was said in.
//!
//! A job pairs a [`Schedule`] with the prompt to deliver, and optionally a *gate*: a cheap shell
//! command run first, whose result decides whether the expensive model turn happens at all. Without
//! one, "watch X every 30s" costs a model turn every 30 seconds; with one it costs a process spawn,
//! and a turn only when something actually changed.
//!
//! Jobs live in the session database rather than `config.toml` because they are runtime data the
//! agent creates, not settings a human writes, and they are keyed to a session so a job dies with
//! the conversation that asked for it.

use std::time::Duration;

use chrono::{DateTime, Local, Utc};
use croner::Cron;
// Reached through `humantime_serde`'s re-export rather than a direct dependency, which is also
// how `crate::config` gets at it. One duration syntax, one copy of the parser.
use humantime_serde::re::humantime;

mod gate;
mod memory;

pub(crate) use self::{gate::*, memory::*};

/// Smallest interval a recurring job may use. Not a policy limit: a zero or sub-second interval
/// makes `next_after` return an instant that is already in the past by the time it is stored, so
/// the job fires every poll tick forever.
const MIN_EVERY: Duration = Duration::from_secs(1);

/// Upper bound on the missed-occurrence count reported to the model. Counting is a courtesy ("you
/// missed 12 checks"), and walking a per-minute cron across a month-long outage to get an exact
/// figure is not worth the tick it would spend.
pub(crate) const MAX_COALESCED_REPORTED: u32 = 1000;

/// How many undelivered claims a job may accumulate before it stops being retried.
///
/// A lease is handed back on expiry, so a prompt that reliably kills the process is otherwise
/// claimed again forever. Counting the claims that ended in neither a delivery nor a handback
/// identifies exactly that job, and this is where it is parked: still listed, still cancelable,
/// reported as held on every surface, and unable to take the daemon down with it. Three, because
/// two is within the range of ordinary bad luck (a deploy during a fire, then a machine restart)
/// and a fourth attempt on something that has failed three times is not going to be the one that
/// works.
///
/// It also bounds the one shape the "advance on a failed probe" rule cannot reach: a job with no
/// next occurrence, which in practice means a one-shot. There the lease is held rather than
/// released, so the retry waits out `claim_lease` instead of coming round on the next tick, and
/// this is what stops that from going on until the grace period closes. Three attempts an hour
/// apart is a budget a transient outage survives and a broken gate does not; three ten-second ones,
/// which is what releasing the lease would have given, is neither.
pub(crate) const MAX_CLAIM_ATTEMPTS: u32 = 3;

/// When a job fires.
///
/// `Cron` is boxed because [`croner::Cron`] carries a parsed component table an order of magnitude
/// larger than the other two variants, and a `Schedule` is cloned per poll tick.
#[derive(Debug, Clone)]
pub(crate) enum Schedule {
    /// Fire once at an instant, then delete. Always absolute: see [`Schedule::parse_at`] for why a
    /// relative input is resolved at creation rather than stored as written.
    At(DateTime<Utc>),
    /// Fire repeatedly, this far apart.
    Every(Duration),
    /// Fire on a 5-field cron pattern, evaluated in the host's local time.
    Cron(Box<Cron>),
}

impl Schedule {
    /// Parse a one-shot time: either an RFC 3339 timestamp or a humantime duration relative to
    /// `now` (`"20m"`, `"2h"`, `"1h 30m"`).
    ///
    /// A relative input is resolved to an absolute instant here and stored that way. Keeping it
    /// relative would be a job that never fires: every process restart would re-parse `"20m"` and
    /// push the target twenty minutes further into the future.
    pub(crate) fn parse_at(input: &str, now: DateTime<Utc>) -> Result<Self, String> {
        let input = input.trim();
        if let Ok(absolute) = DateTime::parse_from_rfc3339(input) {
            return Ok(Self::At(absolute.with_timezone(&Utc)));
        }
        let offset = parse_duration(input).map_err(|error| {
            format!("'{input}' is neither an RFC 3339 timestamp nor a duration: {error}")
        })?;
        let offset = chrono::Duration::from_std(offset)
            .map_err(|_| format!("'{input}' is too far in the future to schedule"))?;
        now.checked_add_signed(offset)
            .map(Self::At)
            .ok_or_else(|| format!("'{input}' is too far in the future to schedule"))
    }

    /// Parse a recurring interval (`"30m"`, `"1h"`).
    ///
    /// Note that an interval shorter than the scheduler's poll tick fires once per tick, not once
    /// per interval; the tick is the real resolution floor.
    pub(crate) fn parse_every(input: &str) -> Result<Self, String> {
        let input = input.trim();
        let interval = parse_duration(input)
            .map_err(|error| format!("'{input}' is not a valid duration: {error}"))?;
        if interval < MIN_EVERY {
            return Err(format!(
                "interval '{}' is below the {}s minimum",
                input,
                MIN_EVERY.as_secs()
            ));
        }
        Ok(Self::Every(interval))
    }

    /// Parse a 5-field cron pattern, evaluated in the host's local time.
    ///
    /// Rejects a well-formed but unsatisfiable pattern like `0 0 30 2 *` (February 30th) at
    /// creation, which is the only way to catch it instead of leaving a job that silently never
    /// fires. croner reports its own search-limit error for those, so this only has to ask it for
    /// one occurrence.
    pub(crate) fn parse_cron(input: &str) -> Result<Self, String> {
        let input = input.trim();
        // Five fields, explicitly. `Cron::from_str` defaults to `Seconds::Optional`, so a six-field
        // pattern would parse with the first field as seconds, and `*/10 * * * * *`, written by a
        // model meaning "every 10 minutes" in the Quartz shape, would become every 10 seconds. The
        // `MIN_EVERY` floor that stops `every` firing on each poll tick does not apply to `cron`,
        // and the confirmation echoes the pattern back verbatim.
        let cron = croner::parser::CronParser::builder()
            .seconds(croner::parser::Seconds::Disallowed)
            .build()
            .parse(input)
            .map_err(|error| format!("'{input}' is not a valid cron expression: {error}"))?;
        let schedule = Self::Cron(Box::new(cron));
        if schedule.next_after(Utc::now()).is_none() {
            return Err(format!(
                "cron expression '{input}' matches no calendar date"
            ));
        }
        Ok(schedule)
    }

    /// Rebuild a schedule from its two persisted columns. The inverse of [`Schedule::kind_str`] +
    /// [`Schedule::spec`].
    pub(crate) fn from_stored(kind: &str, spec: &str) -> Result<Self, String> {
        match kind {
            "at" => DateTime::parse_from_rfc3339(spec)
                .map(|absolute| Self::At(absolute.with_timezone(&Utc)))
                .map_err(|error| format!("stored 'at' spec '{spec}' is not RFC 3339: {error}")),
            "every" => Self::parse_every(spec),
            // Rehydrates through `parse_cron`, not `Cron::from_str`, so a stored spec is read back
            // under the same five-field grammar that accepted it. The permissive parser would read
            // a six-field pattern's first field as seconds, giving a stored job a different meaning
            // on reload than it had at creation. A spec that does not parse under those rules is
            // surfaced as an error rather than silently reinterpreted.
            "cron" => Self::parse_cron(spec)
                .map_err(|error| format!("stored cron spec '{spec}' is invalid: {error}")),
            other => Err(crate::text::unknown_name("schedule kind", other, [
                "at", "every", "cron",
            ])),
        }
    }

    /// Discriminant as persisted in `scheduled_jobs.kind`.
    pub(crate) fn kind_str(&self) -> &'static str {
        match self {
            Self::At(_) => "at",
            Self::Every(_) => "every",
            Self::Cron(_) => "cron",
        }
    }

    /// Round-trippable form as persisted in `scheduled_jobs.spec`.
    pub(crate) fn spec(&self) -> String {
        match self {
            Self::At(instant) => instant.to_rfc3339(),
            Self::Every(interval) => humantime::format_duration(*interval).to_string(),
            Self::Cron(cron) => cron.pattern.to_string(),
        }
    }

    /// Whether firing this schedule leaves a job to reschedule. One-shots are deleted on fire.
    pub(crate) fn is_recurring(&self) -> bool {
        !matches!(self, Self::At(_))
    }

    /// The occurrence to schedule after delivering the one at `delivered`, given the clock is now
    /// `now`.
    ///
    /// This is what a firing job needs, and it is **not** `next_after(now)`. For `Every`, anchoring
    /// on the current time adds the pickup latency to every interval, permanently: the scheduler
    /// polls, notices a job is due some milliseconds late, and schedules the next one a full
    /// interval from that moment rather than from the occurrence it just spent. At `poll_interval
    /// = 1s` an `every = "1s"` job would run at 2.0s, half the requested rate, because the tick one
    /// interval later lands microseconds early and the job loses a whole poll.
    ///
    /// Advancing by whole intervals from `delivered` preserves the phase, so a job fires on the
    /// grid it was created on however late any single pickup is. A backlog is skipped in one
    /// multiplication rather than a loop, which matters after an outage: `occurrences_between`
    /// separately reports how many were coalesced.
    ///
    /// `Cron` is unaffected: its occurrences are absolute wall-clock instants, so the next one
    /// after `now` is the next one, and `At` has no successor at all.
    pub(crate) fn next_after_delivering(
        &self,
        delivered: DateTime<Utc>,
        now: DateTime<Utc>,
    ) -> Option<DateTime<Utc>> {
        match self {
            Self::At(_) => None,
            Self::Every(interval) => {
                let step = chrono::Duration::from_std(*interval).ok()?;
                // Milliseconds, not seconds. `num_seconds` truncates, and `parse_every` accepts
                // anything from 1s up while humantime parses `"1500ms"` and `"1s 500ms"`, which
                // round-trip through `spec()`. With the interval truncated to 1s, a `1500ms` job
                // would advance by whole seconds and walk off its own grid a little further on
                // every fire, which is exactly the property this function exists to hold.
                let step_millis = step.num_milliseconds();
                // A zero or negative interval has no grid to stay on; fall back rather than divide
                // by zero.
                if step_millis <= 0 {
                    return self.next_after(now);
                }
                let mut next = delivered.checked_add_signed(step)?;
                if next <= now {
                    let behind = (now - next).num_milliseconds();
                    let skips = behind / step_millis + 1;
                    next = next.checked_add_signed(chrono::Duration::milliseconds(
                        skips.checked_mul(step_millis)?,
                    ))?;
                }
                Some(next)
            }
            Self::Cron(_) => self.next_after(now),
        }
    }

    /// The first occurrence strictly after `anchor`, or `None` when there is no next occurrence
    /// (a one-shot whose instant has passed, or a cron pattern matching no upcoming date).
    ///
    /// Callers must pass the job's own anchor (`last_fired_at` if it has ever fired, otherwise
    /// `created_at`) and never `Utc::now()`. Anchoring on the current time makes a pinned pattern
    /// such as `30 14 27 2 *` skip to next year whenever the process happens to restart after its
    /// window; anchoring permanently on `created_at` makes a long-lived job replay every occurrence
    /// since it was created.
    pub(crate) fn next_after(&self, anchor: DateTime<Utc>) -> Option<DateTime<Utc>> {
        match self {
            Self::At(instant) => (*instant > anchor).then_some(*instant),
            Self::Every(interval) => {
                let interval = chrono::Duration::from_std(*interval).ok()?;
                anchor.checked_add_signed(interval)
            }
            Self::Cron(cron) => {
                // Cron patterns are wall-clock expressions: "0 9 * * *" means 09:00 where the user
                // lives, so the search runs in local time and the result converts back to the UTC
                // meka stores.
                let local_anchor = anchor.with_timezone(&Local);
                // croner bounds its own forward search and reports a search-limit error for a
                // pattern that matches no calendar date, so its verdict is the whole answer. An
                // extra horizon here would be indistinguishable from that verdict at the call site,
                // and `prepare` retires a job whose schedule has no next occurrence: a 366-day one
                // would delete `0 0 29 2 *` the first time it fired, because the next February
                // 29th is up to four years out.
                let next = cron.find_next_occurrence(&local_anchor, false).ok()?;
                Some(next.with_timezone(&Utc))
            }
        }
    }

    /// One-line human description, for the confirmation a tool hands back and for `schedule list`.
    /// A scheduling mistake is otherwise invisible until it fires, which may be days later.
    pub(crate) fn describe(&self) -> String {
        match self {
            Self::At(instant) => format!(
                "once at {}",
                crate::text::format_timestamp(*instant, crate::text::Precision::Minutes)
            ),
            Self::Every(interval) => {
                format!("every {}", humantime::format_duration(*interval))
            }
            Self::Cron(cron) => cron.pattern.to_string(),
        }
    }
}

/// A persisted wakeup.
#[derive(Debug, Clone)]
pub(crate) struct ScheduledJob {
    pub(crate) id: String,
    pub(crate) session_id: uuid::Uuid,
    pub(crate) schedule: Schedule,
    pub(crate) prompt: String,
    pub(crate) gate: Option<Gate>,
    pub(crate) created_at: DateTime<Utc>,
    pub(crate) last_fired_at: Option<DateTime<Utc>>,
    pub(crate) next_fire_at: DateTime<Utc>,
    /// Claims that ended without delivering the turn and without the *host* declining the job.
    ///
    /// A claim raises it, and it is only cleared by an ending that says the job is fine. Two
    /// endings do not: a host that dies or panics mid-delivery, and a job with no next occurrence
    /// whose gate probe could not be evaluated, which keeps its lease so the retry waits out
    /// `claim_lease` rather than coming round on the next tick. Both leave an occurrence that
    /// nothing has spent, so something has to bound how often it is retried;
    /// [`MAX_CLAIM_ATTEMPTS`] is where a job that keeps doing it stops being retried.
    ///
    /// A deferral does *not* raise it. That is a host saying "not me", which is a fact about the
    /// host rather than the job, so [`crate::store::schedule::ScheduleStore::release_claim`]
    /// resets the count.
    pub(crate) attempts: u32,
}

impl ScheduledJob {
    /// The instant [`Schedule::next_after`] must be measured from for this job.
    ///
    /// Exists so no call site has to remember the rule, because both ways of getting it wrong are
    /// silent: anchoring on `now` skips a pinned pattern to next year after an ill-timed restart,
    /// and anchoring permanently on `created_at` replays every occurrence since creation.
    #[cfg(test)]
    pub(crate) fn anchor(&self) -> DateTime<Utc> {
        self.last_fired_at.unwrap_or(self.created_at)
    }

    /// Short id for display, matching the width `schedule_cancel` accepts.
    pub(crate) fn short_id(&self) -> &str {
        self.id.get(..crate::text::ID_PREFIX).unwrap_or(&self.id)
    }

    /// What should become of this job's prompt if the turn it triggers fails before the model ever
    /// sees it.
    ///
    /// Defined once here rather than at each host, because every host has to answer it and the rule
    /// is not obvious enough to restate three times. A recurring job produces the prompt again on
    /// its next occurrence, so a failure withdrawing it costs nothing and spares the conversation
    /// one unanswered message per fire through an outage. A one-shot does not: its row is retired
    /// the moment the turn is delivered, so once a failed fire has been through `complete_claim`
    /// the unanswered message is the last trace that the reminder ever existed, and withdrawing it
    /// would be the deletion the feature is supposed to prevent.
    ///
    /// The row survives *during* the turn rather than being deleted before it, which is why the
    /// reasoning is about the completion rather than the claim.
    pub(crate) fn prompt_retention(&self) -> crate::conversation::PromptRetention {
        match self.schedule.is_recurring() {
            true => crate::conversation::PromptRetention::Withdraw,
            false => crate::conversation::PromptRetention::Keep,
        }
    }
}

/// How many occurrences of `schedule` fall between `from` and `to`, capped at
/// [`MAX_COALESCED_REPORTED`].
pub(crate) fn occurrences_between(
    schedule: &Schedule,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> u32 {
    match schedule {
        // A one-shot has exactly the one occurrence, which is the fire being delivered.
        Schedule::At(_) => 0,
        Schedule::Every(interval) => {
            // Milliseconds, matching `next_after_delivering`: `as_secs` would truncate, so a
            // `1500ms` job would count its coalesced occurrences against a 1s grid it does not run
            // on.
            let interval = interval.as_millis();
            if interval == 0 {
                return 0;
            }
            let elapsed = (to - from).num_milliseconds().max(0) as u128;
            u32::try_from(elapsed / interval)
                .unwrap_or(MAX_COALESCED_REPORTED)
                .min(MAX_COALESCED_REPORTED)
        }
        Schedule::Cron(_) => {
            // Counts occurrences in `(from, to]`, i.e. everything after the one being delivered,
            // matching the `Every` arm above, which divides the same open interval.
            let mut cursor = from;
            let mut count = 0;
            while count < MAX_COALESCED_REPORTED {
                match schedule.next_after(cursor) {
                    Some(next) if next <= to => {
                        cursor = next;
                        count += 1;
                    }
                    _ => break,
                }
            }
            count
        }
    }
}

/// Parse a humantime duration, via the same re-export `crate::config` uses so `every = "30m"` in a
/// tool call and `idle_timeout = "30m"` in `config.toml` mean the same thing.
///
/// Worth spelling out in any tool description built on this: humantime reads `m` as minutes and
/// `M` as months, so `1m` and `1M` differ by a factor of 43,800. Decimals and compound forms both
/// work (`1.5h` and `1h 30m` are the same duration).
fn parse_duration(input: &str) -> Result<Duration, humantime::DurationError> {
    humantime::parse_duration(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{scheduler::*, store::schedule::*};

    pub(super) fn at(text: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(text)
            .expect("test timestamp parses")
            .with_timezone(&Utc)
    }

    #[test]
    fn parse_at_accepts_rfc3339() {
        let now = at("2026-08-11T12:00:00Z");
        let schedule = Schedule::parse_at("2026-08-12T09:30:00Z", now).expect("parses");
        assert!(matches!(schedule, Schedule::At(instant) if instant == at("2026-08-12T09:30:00Z")));
    }

    /// A relative `at` must be resolved against `now` at parse time. Storing "20m" verbatim would
    /// produce a job that re-bases itself on every restart and therefore never fires.
    #[test]
    fn parse_at_resolves_relative_input_to_an_absolute_instant() {
        let now = at("2026-08-11T12:00:00Z");
        let schedule = Schedule::parse_at("20m", now).expect("parses");
        assert!(matches!(schedule, Schedule::At(instant) if instant == at("2026-08-11T12:20:00Z")));
        assert_eq!(schedule.spec(), at("2026-08-11T12:20:00Z").to_rfc3339());
    }

    #[test]
    fn parse_at_rejects_nonsense() {
        let now = at("2026-08-11T12:00:00Z");
        assert!(Schedule::parse_at("next tuesday", now).is_err());
        assert!(Schedule::parse_at("", now).is_err());
        assert!(
            Schedule::parse_at("2026-08-12", now).is_err(),
            "a bare date is not RFC 3339 and is not a duration either"
        );
    }

    /// `m` is minutes and `M` is months, a factor of roughly 43,800 apart. Pinned because a model
    /// writing `30M` for "half an hour" would schedule two and a half years out, and nothing else
    /// in the system would notice.
    #[test]
    fn duration_units_distinguish_minutes_from_months() {
        let minutes = Schedule::parse_every("30m").expect("parses");
        let months = Schedule::parse_every("30M").expect("parses");
        assert!(
            matches!(minutes, Schedule::Every(interval) if interval == Duration::from_secs(1800))
        );
        assert!(
            matches!(months, Schedule::Every(interval) if interval > Duration::from_secs(60 * 60 * 24 * 365))
        );
    }

    #[test]
    fn parse_every_rejects_sub_minimum_intervals() {
        // A zero interval yields a next-fire that is already in the past, so the job would fire on
        // every poll tick forever.
        assert!(Schedule::parse_every("0s").is_err());
        assert!(Schedule::parse_every("1s").is_ok());
    }

    #[test]
    fn parse_every_reads_m_as_minutes() {
        let schedule = Schedule::parse_every("30m").expect("parses");
        assert!(
            matches!(schedule, Schedule::Every(interval) if interval == Duration::from_secs(1800))
        );
    }

    #[test]
    fn parse_cron_accepts_five_fields() {
        // The reason croner is used over the `cron` crate: `cron` demands a seconds field, so the
        // five-field expressions users and models actually write would fail to parse.
        assert!(Schedule::parse_cron("0 9 * * 1-5").is_ok());
        assert!(Schedule::parse_cron("*/5 * * * *").is_ok());
    }

    /// The five-field rule has to reach the rows already on disk, which is the only population it
    /// matters for: with rehydration left on the permissive parser, a six-field row would keep its
    /// every-ten-seconds reading forever.
    #[test]
    fn a_stored_cron_spec_is_read_with_the_same_grammar_it_was_created_under() {
        assert!(
            Schedule::parse_cron("*/10 * * * * *").is_err(),
            "six fields are refused at creation"
        );
        assert!(
            Schedule::from_stored("cron", "*/10 * * * * *").is_err(),
            "and refused on the way back out of the database"
        );

        // A legitimate five-field row still round-trips.
        let stored = Schedule::from_stored("cron", "0 9 * * 1-5").expect("five fields still load");
        assert_eq!(stored.spec(), "0 9 * * 1-5");
    }

    #[test]
    fn parse_cron_rejects_unsatisfiable_pattern() {
        // Well-formed but matches no calendar date; caught at creation rather than leaving a job
        // that silently never fires.
        assert!(Schedule::parse_cron("0 0 30 2 *").is_err());
    }

    /// A pattern whose next occurrence is years away is satisfiable, and `prepare` retires a job
    /// whose schedule has no next occurrence, so `next_after` must not confuse "far off" with
    /// "never". February 29th is the shortest such case at up to four years.
    #[test]
    fn a_schedule_whose_next_occurrence_is_years_away_still_has_one() {
        use chrono::TimeZone;

        let schedule = Schedule::parse_cron("0 0 29 2 *").expect("Feb 29 is a real date");
        let anchor = Utc
            .with_ymd_and_hms(2026, 8, 16, 12, 0, 0)
            .single()
            .expect("anchor");
        let next = schedule
            .next_after(anchor)
            .expect("a leap day must not read as an unschedulable pattern");
        assert!(next > anchor + chrono::Duration::days(366), "{next}");
    }

    #[test]
    fn parse_cron_rejects_malformed_pattern() {
        assert!(Schedule::parse_cron("not a cron").is_err());
        assert!(Schedule::parse_cron("99 * * * *").is_err());
    }

    #[test]
    fn next_after_at_fires_once_then_never() {
        let instant = at("2026-08-12T09:00:00Z");
        let schedule = Schedule::At(instant);
        assert_eq!(
            schedule.next_after(at("2026-08-11T12:00:00Z")),
            Some(instant)
        );
        // Anchored at or past its own instant, a one-shot has no next occurrence.
        assert_eq!(schedule.next_after(instant), None);
        assert_eq!(schedule.next_after(at("2026-08-13T00:00:00Z")), None);
    }

    #[test]
    fn next_after_every_advances_one_interval_from_the_anchor() {
        let schedule = Schedule::parse_every("30m").expect("parses");
        assert_eq!(
            schedule.next_after(at("2026-08-11T12:00:00Z")),
            Some(at("2026-08-11T12:30:00Z"))
        );
    }

    /// The anchoring rule, stated as a test: a job that last fired ten days ago yields exactly one
    /// next fire, not ten. Coalescing missed occurrences is the scheduler's job, but it depends on
    /// `next_after` returning a single instant rather than a backlog.
    #[test]
    fn next_after_every_yields_one_occurrence_from_a_stale_anchor() {
        let schedule = Schedule::parse_every("1h").expect("parses");
        let stale = at("2026-08-01T12:00:00Z");
        assert_eq!(schedule.next_after(stale), Some(at("2026-08-01T13:00:00Z")));
    }

    #[test]
    fn schedule_round_trips_through_stored_columns() {
        for original in [
            Schedule::parse_at("2026-08-12T09:30:00Z", at("2026-08-11T12:00:00Z")).expect("parses"),
            Schedule::parse_every("45m").expect("parses"),
            Schedule::parse_cron("0 9 * * 1-5").expect("parses"),
        ] {
            let restored = Schedule::from_stored(original.kind_str(), &original.spec())
                .expect("stored form parses back");
            assert_eq!(restored.kind_str(), original.kind_str());
            assert_eq!(restored.spec(), original.spec());
        }
    }

    #[test]
    fn from_stored_rejects_unknown_kind() {
        assert!(Schedule::from_stored("on-exit", "make build").is_err());
    }

    /// A gate command that creates `path`, in whichever shell the host will run it under.
    ///
    /// `evaluate_gate` spawns the platform shell, so a `touch` hardcoded in a fixture is a Unix
    /// command handed to PowerShell on Windows: the gate reports a non-zero exit, the probe never
    /// appears, and the test reads that as the scheduler having declined to fire.
    fn create_file_command(path: &std::path::Path) -> String {
        if cfg!(windows) {
            format!(
                "New-Item -ItemType File -Force -Path '{}' | Out-Null",
                path.display()
            )
        } else {
            format!("touch '{}'", path.display())
        }
    }

    pub(super) fn gate(command: &str, predicate: GatePredicate, last_output: Option<&str>) -> Gate {
        Gate {
            probe: GateProbe::Shell {
                command: command.to_string(),
            },
            predicate,
            last_output: last_output.map(str::to_string),
            // The level every gate is created at. Tests that exercise a gate *running* need it; the
            // one that exercises a withdrawn authority overrides it explicitly.
            permission: crate::permission::Permission::Unrestricted,
        }
    }

    pub(super) const GATE_BUDGET: Duration = Duration::from_secs(10);

    /// A probe result, without running anything. `apply_predicate` is pure, so every predicate can
    /// be exercised directly rather than through a command that has to produce the shape.
    /// `apply_predicate` for tests that expect an answer rather than a broken probe. The `Err`
    /// arm is exercised on its own, by the tests that hand it something that is not a document.
    fn judged(
        predicate: &GatePredicate,
        probe: &ProbeOutcome,
        last_output: Option<&str>,
    ) -> GateOutcome {
        apply_predicate(predicate, probe, last_output).expect("the predicate has an answer")
    }

    pub(super) fn probed(
        text: &str,
        structured: Option<serde_json::Value>,
        succeeded: bool,
    ) -> ProbeOutcome {
        ProbeOutcome {
            text: text.to_string(),
            structured,
            succeeded,
        }
    }

    /// The regression this whole design exists to prevent.
    ///
    /// A structured result carrying anything self-moving is different on every call, so `changed`
    /// over the whole of it fires every single interval and spends exactly the turns a gate is
    /// meant to save. It looks like a working watcher right up until the bill arrives. Pointing at
    /// the field that matters is the only honest way to watch one, so this asserts both halves: the
    /// pointer stays quiet while only the timestamp moves, and the naive predicate over the same
    /// two results does not.
    #[test]
    fn a_pointer_ignores_a_sibling_that_moves_on_its_own() {
        let first = serde_json::json!({"chats": [], "checked_at": "2026-08-25T03:00:00Z"});
        let second = serde_json::json!({"chats": [], "checked_at": "2026-08-25T03:00:30Z"});
        let pointer = GatePredicate::At {
            pointer: "/chats".to_string(),
            is: PointerTest::Changed,
        };

        let baseline = judged(&pointer, &probed("", Some(first), true), None).baseline;
        let outcome = judged(
            &pointer,
            &probed("", Some(second.clone()), true),
            Some(&baseline),
        );
        assert!(
            !outcome.fired,
            "only `checked_at` moved, so the watched list did not change"
        );

        // The same two results under `changed`, to show the pointer is doing the work rather than
        // the values happening to be equal.
        let naive = judged(
            &GatePredicate::Changed,
            &probed(&second.to_string(), None, true),
            Some(
                &serde_json::json!({"chats": [], "checked_at": "2026-08-25T03:00:00Z"}).to_string(),
            ),
        );
        assert!(
            naive.fired,
            "whole-result `changed` sees the timestamp and fires, which is the trap"
        );
    }

    /// The user's case: fire when there is something to read, stay quiet when there is not.
    #[test]
    fn not_empty_follows_the_pointed_at_collection() {
        let predicate = GatePredicate::At {
            pointer: "/chats".to_string(),
            is: PointerTest::NotEmpty,
        };
        let empty = serde_json::json!({"chats": []});
        let full = serde_json::json!({"chats": [{"id": "a"}]});

        assert!(!judged(&predicate, &probed("", Some(empty), true), None).fired);
        assert!(judged(&predicate, &probed("", Some(full), true), None).fired);
    }

    /// Plenty of MCP servers send JSON as their text content and set no `structuredContent`, so a
    /// pointer has to reach that too or it would work against half the servers people run.
    #[test]
    fn a_pointer_falls_back_to_parsing_the_text() {
        let predicate = GatePredicate::At {
            pointer: "/chats".to_string(),
            is: PointerTest::NotEmpty,
        };
        let outcome = judged(
            &predicate,
            &probed(r#"{"chats": [{"id": "a"}]}"#, None, true),
            None,
        );
        assert!(outcome.fired, "the text parsed and the list is non-empty");
    }

    /// And it still parses when the document is larger than the turn is allowed to see.
    ///
    /// The two limits are unrelated: `text` is capped so a runaway probe cannot push the prompt
    /// over the context window, and the cap appends a marker, so a result parsed after capping
    /// would no longer parse and an `at` gate over a large result would report "the probe did not
    /// return JSON" about a probe that did.
    #[test]
    fn a_pointer_reads_a_document_larger_than_the_turn_is_shown() {
        let filler = "x".repeat(GATE_OUTPUT_LIMIT);
        let raw = format!(r#"{{"filler": "{filler}", "chats": [{{"id": "a"}}]}}"#);
        assert!(
            raw.len() > GATE_OUTPUT_LIMIT,
            "the document exceeds the cap"
        );

        let probe = ProbeOutcome::new(&raw, None, true);
        assert!(
            probe.text.ends_with("[gate output truncated]"),
            "the turn is still shown a bounded result"
        );

        let outcome = judged(
            &GatePredicate::At {
                pointer: "/chats".to_string(),
                is: PointerTest::NotEmpty,
            },
            &probe,
            None,
        );
        assert!(
            outcome.fired,
            "the pointer judges the whole document, which is what was measured"
        );
    }

    /// The other half of the split: a document that parsed and simply lacks the field is an
    /// answer, and `empty` still fires on it. An API that omits `chats` when there are none is
    /// saying there are none, which is exactly what the predicate was asked.
    #[test]
    fn a_pointer_at_an_absent_field_in_real_json_still_answers() {
        let predicate = GatePredicate::At {
            pointer: "/chats".to_string(),
            is: PointerTest::Empty,
        };
        assert!(
            judged(&predicate, &probed(r#"{"other": 1}"#, None, true), None).fired,
            "the document parsed, so an absent field is a genuine `empty`"
        );

        let predicate = GatePredicate::At {
            pointer: "/chats".to_string(),
            is: PointerTest::NotEmpty,
        };
        assert!(!judged(&predicate, &probed(r#"{"other": 1}"#, None, true), None).fired);
    }

    #[test]
    fn matches_judges_the_text() {
        let predicate = GatePredicate::Matches {
            pattern: r"ERROR \d+".to_string(),
        };
        assert!(judged(&predicate, &probed("saw ERROR 500", None, true), None).fired);
        assert!(!judged(&predicate, &probed("all quiet", None, true), None).fired);
    }

    /// `succeeded` is the one predicate that reads the probe's own status rather than its output,
    /// which is what lets it mean the same thing for a command's exit code and a tool's error flag.
    #[test]
    fn succeeded_follows_the_probe_status_not_its_output() {
        let predicate = GatePredicate::Succeeded;
        assert!(judged(&predicate, &probed("", None, true), None).fired);
        assert!(!judged(&predicate, &probed("lots of output", None, false), None).fired);
    }

    /// Count the `warn!` lines a block emits, so "said once, not once per tick" is testable.
    ///
    /// The suppression this guards is about log volume, and log volume has no other observable:
    /// the job's state after a sweep is identical whether the line was written or not. Capturing
    /// the output is the only way to tell a fix from a no-op here.
    ///
    /// Capture goes through [`crate::render::log_capture`], which explains why the subscriber
    /// behind it is global rather than a `tracing::subscriber::set_default`.
    ///
    /// `#[tokio::test]` is single-threaded, so `body` is polled on the thread that owns the
    /// capture buffer throughout.
    async fn warnings_from<F>(body: F) -> usize
    where
        F: std::future::Future<Output = ()>,
    {
        crate::render::log_capture::start();
        body.await;
        crate::render::log_capture::warnings()
            .matches("not fired:")
            .count()
    }

    /// A held job says so once, not once per poll interval, including a one-shot.
    ///
    /// Retiring a one-shot must not clear its held-back state on the reasoning that the row is
    /// gone: an authority refusal puts the row back, so a clear that ran before the refusal on
    /// every sweep would make the "first time" check true every time, and a held one-shot would
    /// warn every 10 seconds for up to the whole `missed_grace` window.
    #[tokio::test]
    async fn a_held_job_warns_once_not_once_per_sweep() {
        for (label, schedule) in [
            (
                "one-shot",
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
            ),
            ("recurring", Schedule::parse_every("1h").expect("parses")),
        ] {
            let harness = SchedulerHarness::new().await;
            harness
                .manager
                .update_session(harness.session_id, crate::store::SessionPatch {
                    permission: Some("read".parse().expect("a level")),
                    ..Default::default()
                })
                .await
                .expect("record the level the session was set to");
            harness
                .overdue_job(
                    schedule,
                    Some(gate("true", GatePredicate::Succeeded, None)),
                    chrono::Duration::minutes(5),
                )
                .await;

            let warnings = warnings_from(async {
                for _ in 0..4 {
                    harness.tick().await;
                }
            })
            .await;
            assert_eq!(
                warnings, 1,
                "{label}: a standing condition is announced once, not on every sweep"
            );
        }
    }

    /// A dispatcher that knows nothing yet because its server is mid-handshake.
    #[derive(Debug)]
    pub(super) struct StillConnecting;

    #[async_trait::async_trait]
    impl GateTools for StillConnecting {
        fn resolve(&self, _name: &str) -> Option<crate::permission::Permission> {
            None
        }

        fn is_still_connecting(&self, _name: &str) -> bool {
            true
        }

        async fn call(
            &self,
            _name: &str,
            _arguments: &serde_json::Value,
            _timeout: Duration,
            _cwd: Option<&std::path::Path>,
            _session_id: Option<uuid::Uuid>,
        ) -> std::result::Result<ProbeOutcome, String> {
            Err("not connected".to_string())
        }
    }

    /// Canceling a job that is already gone reports a miss, not a cancellation.
    ///
    /// Both cancel doors resolve an id from a listing and then delete it, and a scheduler sweep can
    /// retire the row in between: a one-shot's occurrence retires it, and deleting a session takes
    /// its jobs through the foreign key. Reporting success regardless would say "Canceled job
    /// abc12345" about a job this call did not cancel, in the same words it uses when it did.
    #[tokio::test]
    async fn deleting_a_job_that_is_already_gone_reports_that_it_removed_nothing() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;
        let store = harness.manager.schedule_store();

        assert!(
            store
                .delete_scheduled_job(&job.id)
                .await
                .expect("the delete runs"),
            "the row was there, so this call is the one that removed it"
        );
        assert!(
            !store
                .delete_scheduled_job(&job.id)
                .await
                .expect("the delete runs"),
            "nothing was removed the second time, and saying otherwise is a claim about work that \
             did not happen"
        );
        assert_eq!(
            harness
                .manager
                .schedule_store()
                .cancel_scheduled_job(harness.session_id, job.short_id())
                .await
                .expect("the cancel runs"),
            None,
            "and the prefix door agrees, since it delegates to the same delete"
        );
    }

    /// A dispatcher whose answers are fixed by the test, so the authority rule can be exercised
    /// without an MCP server.
    #[derive(Debug)]
    pub(super) struct FixedTools(pub(super) Option<crate::permission::Permission>);

    #[async_trait::async_trait]
    impl GateTools for FixedTools {
        fn resolve(&self, _name: &str) -> Option<crate::permission::Permission> {
            self.0
        }

        async fn call(
            &self,
            _name: &str,
            _arguments: &serde_json::Value,
            _timeout: Duration,
            _cwd: Option<&std::path::Path>,
            _session_id: Option<uuid::Uuid>,
        ) -> std::result::Result<ProbeOutcome, String> {
            Ok(probed("{}", None, true))
        }
    }

    pub(super) fn tool_probe() -> GateProbe {
        GateProbe::Tool {
            name: "mcp__bridge__unseen".to_string(),
            arguments: serde_json::json!({}),
        }
    }

    #[tokio::test]
    async fn on_success_gate_follows_the_exit_code() {
        let passing = evaluate_gate(
            &gate("exit 0", GatePredicate::Succeeded, None),
            GATE_BUDGET,
            None,
            None,
            None,
        )
        .await
        .expect("gate ran");
        assert!(passing.fired);

        let failing = evaluate_gate(
            &gate("exit 1", GatePredicate::Succeeded, None),
            GATE_BUDGET,
            None,
            None,
            None,
        )
        .await
        .expect("gate ran");
        assert!(
            !failing.fired,
            "a false condition is not an error, it is just no fire"
        );
    }

    #[tokio::test]
    async fn on_change_gate_fires_on_its_first_evaluation() {
        // No baseline means the watcher has never run. Firing proves the command works instead of
        // leaving a typo undiscovered until the thing being watched finally changes.
        let outcome = evaluate_gate(
            &gate("echo ready", GatePredicate::Changed, None),
            GATE_BUDGET,
            None,
            None,
            None,
        )
        .await
        .expect("gate ran");
        assert!(outcome.fired);
        assert_eq!(outcome.output, "ready");
    }

    /// A gate runs in its session's directory, not the host process's.
    ///
    /// A parked job stays in the table on purpose, so "a row is due" and "something will run" are
    /// different questions. Asking the first one would let a job at `MAX_CLAIM_ATTEMPTS`
    /// interrupt the prompt every poll interval, forever, to run nothing.
    #[test]
    fn a_wake_is_only_worth_it_for_a_job_that_can_still_run() {
        let mine = uuid::Uuid::new_v4();
        let theirs = uuid::Uuid::new_v4();
        let job = |session_id, attempts| ScheduledJob {
            id: "j".to_string(),
            session_id,
            schedule: Schedule::At(Utc::now()),
            prompt: "p".to_string(),
            gate: None,
            created_at: Utc::now(),
            last_fired_at: None,
            next_fire_at: Utc::now(),
            attempts,
        };

        assert!(has_runnable_job(&[job(mine, 0)], mine));
        assert!(
            has_runnable_job(&[job(mine, MAX_CLAIM_ATTEMPTS - 1)], mine),
            "one attempt short of parked is still runnable",
        );
        assert!(
            !has_runnable_job(&[job(mine, MAX_CLAIM_ATTEMPTS)], mine),
            "a parked job must not wake the prompt",
        );
        assert!(
            !has_runnable_job(&[job(theirs, 0)], mine),
            "another session's job is not this prompt's business",
        );
        assert!(!has_runnable_job(&[], mine));
        assert!(
            has_runnable_job(&[job(mine, MAX_CLAIM_ATTEMPTS), job(mine, 0)], mine),
            "one runnable job among parked ones is enough",
        );
    }

    /// A non-zero exit is how several perfectly good `changed` gates signal a change: `diff -q`
    /// and `git diff --exit-code` exit 1 exactly when there is a difference. Refusing to fire on a
    /// non-zero exit would silence those permanently.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_on_change_gate_that_signals_through_its_exit_code_still_fires() {
        let gate = Gate {
            probe: GateProbe::Shell {
                command: "echo 'Files a and b differ'; exit 1".to_string(),
            },
            predicate: GatePredicate::Changed,
            last_output: Some("".to_string()),
            permission: crate::permission::Permission::Unrestricted,
        };
        let outcome = evaluate_gate(&gate, GATE_BUDGET, None, None, None)
            .await
            .expect("a non-zero exit is a signal, not a broken gate");
        assert!(
            outcome.fired,
            "output differs from the baseline, so the gate must fire"
        );
        assert_eq!(outcome.output, "Files a and b differ");
    }

    /// The other half: a watcher in its quiet period exits non-zero with nothing on stdout, every
    /// time, and must stay quiet rather than erroring. `grep PATTERN log` is the canonical shape.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_on_change_gate_quiet_period_is_not_an_error() {
        let gate = Gate {
            probe: GateProbe::Shell {
                command: "exit 1".to_string(),
            },
            predicate: GatePredicate::Changed,
            last_output: Some("".to_string()),
            permission: crate::permission::Permission::Unrestricted,
        };
        let outcome = evaluate_gate(&gate, GATE_BUDGET, None, None, None)
            .await
            .expect("a quiet watcher is not a broken one");
        assert!(!outcome.fired, "nothing changed, so nothing fires");
    }

    #[tokio::test]
    async fn on_change_gate_is_quiet_until_the_output_differs() {
        let unchanged = evaluate_gate(
            &gate("echo steady", GatePredicate::Changed, Some("steady")),
            GATE_BUDGET,
            None,
            None,
            None,
        )
        .await
        .expect("gate ran");
        assert!(!unchanged.fired, "same output must not spend a turn");

        let changed = evaluate_gate(
            &gate("echo moved", GatePredicate::Changed, Some("steady")),
            GATE_BUDGET,
            None,
            None,
            None,
        )
        .await
        .expect("gate ran");
        assert!(changed.fired);
        assert_eq!(changed.output, "moved");
    }

    /// The failure mode this guards is the nastiest one in the feature: a broken watcher that
    /// reports nothing looks identical to a healthy watcher with nothing to report. A gate that
    /// overruns must surface as `Err`, never as a quiet `fired: false`.
    #[tokio::test]
    async fn a_gate_that_overruns_its_budget_is_an_error_not_a_silent_skip() {
        #[cfg(unix)]
        let command = "sleep 30";
        #[cfg(windows)]
        let command = "Start-Sleep -Seconds 30";

        let error = evaluate_gate(
            &gate(command, GatePredicate::Changed, None),
            Duration::from_millis(150),
            None,
            None,
            None,
        )
        .await
        .expect_err("an overrunning gate must not report success");
        assert!(error.contains("budget"), "{error}");
    }

    // --- scheduler ---

    /// What a fire delivered, in the shape the assertions care about.
    #[derive(Debug, Clone)]
    struct FiredRecord {
        job_id: String,
        coalesced: u32,
        gate_output: Option<String>,
        late_by: chrono::Duration,
    }

    struct SchedulerHarness {
        manager: std::sync::Arc<crate::store::Store>,
        session_id: uuid::Uuid,
        config: crate::config::ResolvedScheduleConfig,
        fired: std::sync::Arc<std::sync::Mutex<Vec<FiredRecord>>>,
    }

    impl SchedulerHarness {
        async fn new() -> Self {
            Self::at_session_permission(crate::permission::Permission::Unrestricted).await
        }

        /// A harness whose session's row records `permission`, which is the only level the
        /// scheduler reads. `new` uses `Unrestricted` because that is the ordinary case every
        /// other test needs; a row with no level runs nothing, so the level has to be stated here
        /// rather than inherited by accident.
        async fn at_session_permission(permission: crate::permission::Permission) -> Self {
            let manager = std::sync::Arc::new(crate::store::Store::for_test().await);
            let session_id = Self::session_at(&manager, permission).await;
            Self {
                manager,
                session_id,
                config: crate::config::ResolvedScheduleConfig::default(),
                fired: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            }
        }

        /// A session whose row records `permission`, the way every host writes one.
        async fn session_at(
            manager: &crate::store::Store,
            permission: crate::permission::Permission,
        ) -> uuid::Uuid {
            let session_id = manager
                .create_session(None, "test-profile".to_string())
                .await
                .expect("create session");
            manager
                .update_session(session_id, crate::store::SessionPatch {
                    permission: Some(permission),
                    ..Default::default()
                })
                .await
                .expect("record the level");
            session_id
        }

        /// A second session in the same database, at `unrestricted`, for the per-session budget
        /// tests.
        async fn another_session(&self) -> uuid::Uuid {
            Self::session_at(&self.manager, crate::permission::Permission::Unrestricted).await
        }

        /// Insert a job already overdue by `overdue`.
        async fn overdue_job(
            &self,
            schedule: Schedule,
            gate: Option<Gate>,
            overdue: chrono::Duration,
        ) -> ScheduledJob {
            self.overdue_job_in(self.session_id, schedule, gate, overdue)
                .await
        }

        /// Same, against an explicit session.
        async fn overdue_job_in(
            &self,
            session_id: uuid::Uuid,
            schedule: Schedule,
            gate: Option<Gate>,
            overdue: chrono::Duration,
        ) -> ScheduledJob {
            let now = Utc::now();
            let job = ScheduledJob {
                attempts: 0,
                id: uuid::Uuid::new_v4().to_string(),
                session_id,
                schedule,
                prompt: "do the thing".to_string(),
                gate,
                created_at: now - overdue - chrono::Duration::seconds(1),
                last_fired_at: None,
                next_fire_at: now - overdue,
            };
            self.manager
                .schedule_store()
                .create_scheduled_job(&job)
                .await
                .expect("create job");
            job
        }

        /// Drag a job back to a moment ago, so the next sweep considers it again without waiting
        /// out its interval.
        async fn overdue_now(&self, id: &str) {
            let id = id.to_string();
            let due = (Utc::now() - chrono::Duration::minutes(1)).to_rfc3339();
            self.manager
                .schedule_store()
                .connection
                .call(move |connection| {
                    connection.execute(
                        "UPDATE scheduled_jobs SET next_fire_at = ?2 WHERE id = ?1",
                        rusqlite::params![id, due],
                    )
                })
                .await
                .expect("the row is there");
        }

        /// Swap a shell gate's command, for the tests that watch a probe break and then recover.
        async fn rewrite_gate(&self, id: &str, command: &str) {
            let id = id.to_string();
            let spec = serde_json::json!({
                "shell": { "command": command },
                "when": { "at": { "pointer": "/chats", "is": "not_empty" } },
            })
            .to_string();
            self.manager
                .schedule_store()
                .connection
                .call(move |connection| {
                    connection.execute(
                        "UPDATE scheduled_jobs SET gate_spec_json = ?2 WHERE id = ?1",
                        rusqlite::params![id, spec],
                    )
                })
                .await
                .expect("the row is there");
        }

        async fn tick(&self) {
            self.tick_with(self.config.clone(), None).await;
        }

        /// One sweep under a config other than the harness's own, for the tests that model an
        /// operator changing `config.toml` and restarting while the rows stay as they were.
        async fn tick_with(
            &self,
            config: crate::config::ResolvedScheduleConfig,
            tools: Option<std::sync::Arc<dyn GateTools>>,
        ) {
            let fired = self.fired.clone();
            run_due(
                &self.manager,
                &config,
                tools.as_deref(),
                &NoResidents,
                &SchedulerScope::every_job(),
                &move |wakeup: Wakeup| {
                    let fired = fired.clone();
                    async move {
                        if let Ok(mut guard) = fired.lock() {
                            guard.push(FiredRecord {
                                job_id: wakeup.job.id.clone(),
                                coalesced: wakeup.coalesced,
                                gate_output: wakeup.gate_output.clone(),
                                late_by: wakeup.late_by,
                            });
                        }
                        FireOutcome::Ran
                    }
                },
            )
            .await
            .expect("tick runs");
        }

        /// One sweep whose every fire takes `delay`, for the tests about what a long sweep does to
        /// the jobs behind it.
        async fn tick_slowly(&self, delay: Duration) {
            let fired = self.fired.clone();
            run_due(
                &self.manager,
                &self.config,
                None,
                &NoResidents,
                &SchedulerScope::every_job(),
                &move |wakeup: Wakeup| {
                    let fired = fired.clone();
                    async move {
                        tokio::time::sleep(delay).await;
                        if let Ok(mut guard) = fired.lock() {
                            guard.push(FiredRecord {
                                job_id: wakeup.job.id.clone(),
                                coalesced: wakeup.coalesced,
                                gate_output: wakeup.gate_output.clone(),
                                late_by: wakeup.late_by,
                            });
                        }
                        FireOutcome::Ran
                    }
                },
            )
            .await
            .expect("tick runs");
        }

        fn fired(&self) -> Vec<FiredRecord> {
            self.fired
                .lock()
                .map(|guard| guard.clone())
                .unwrap_or_default()
        }

        async fn jobs(&self) -> Vec<ScheduledJob> {
            self.manager
                .schedule_store()
                .list_scheduled_jobs(self.session_id)
                .await
                .expect("list jobs")
        }
    }

    /// An empty prefix cancels nothing, on the door the model can reach.
    ///
    /// `schedule_cancel {"id": ""}` is a tool call, and `require_str` accepts an empty string, so
    /// this is reachable without a user typing anything. `"".starts_with` is true of every id, so
    /// without the guard it would resolve to whichever job was alone and destroy the agent's own
    /// reminder, reporting success, and only start to error once a second job existed.
    #[tokio::test]
    async fn canceling_an_empty_prefix_destroys_no_job() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job_in(
                harness.session_id,
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::seconds(1),
            )
            .await;

        assert!(
            harness
                .manager
                .schedule_store()
                .cancel_scheduled_job(harness.session_id, "")
                .await
                .expect("an empty prefix is a miss, not an error")
                .is_none(),
            "an empty prefix must name no job"
        );
        assert_eq!(
            harness
                .manager
                .schedule_store()
                .list_scheduled_jobs(harness.session_id)
                .await
                .expect("list")
                .len(),
            1,
            "and the job must survive it"
        );

        // The same id, typed as it was printed, still cancels.
        assert_eq!(
            harness
                .manager
                .schedule_store()
                .cancel_scheduled_job(harness.session_id, &job.id[..8])
                .await
                .expect("cancel"),
            Some(job.id),
            "a real prefix is unaffected by the guard"
        );
    }

    /// Every field a job carries survives the write and the read back.
    ///
    /// `ScheduledJobRow` addresses columns by position, so the `INSERT`, the three `SELECT` lists
    /// and the decoder have to agree on an ordering that is written out four times and checked
    /// nowhere. A field asserted here is one an index shift cannot move silently: the timestamps
    /// stop parsing, and the gate and the schedule come back as something else.
    ///
    /// Both an ungated and a gated job, because the gate occupies four consecutive columns in the
    /// middle of the row and a shift that starts after them is invisible to a job that has none.
    #[tokio::test]
    async fn every_field_of_a_job_survives_the_round_trip() {
        let harness = SchedulerHarness::new().await;
        // Fixed rather than `Utc::now()`: the store round-trips through RFC 3339, so a wall-clock
        // instant would make this assert the precision of that format rather than the ordering of
        // the columns.
        let now = chrono::DateTime::parse_from_rfc3339("2026-03-04T05:06:07Z")
            .expect("parses")
            .with_timezone(&Utc);
        let gate = Gate {
            probe: GateProbe::Tool {
                name: "mcp__bridge__unseen".to_string(),
                arguments: serde_json::json!({"folder": "inbox"}),
            },
            predicate: GatePredicate::At {
                pointer: "/chats".to_string(),
                is: PointerTest::NotEmpty,
            },
            last_output: Some("{\"chats\":[]}".to_string()),
            permission: crate::permission::Permission::Read,
        };

        for (label, gate) in [("ungated", None), ("gated", Some(gate))] {
            let written = ScheduledJob {
                attempts: 2,
                id: uuid::Uuid::new_v4().to_string(),
                session_id: harness.session_id,
                schedule: Schedule::parse_every("90m").expect("parses"),
                prompt: "check the feed".to_string(),
                gate,
                created_at: now,
                last_fired_at: Some(now + chrono::Duration::minutes(5)),
                next_fire_at: now + chrono::Duration::minutes(90),
            };
            harness
                .manager
                .schedule_store()
                .create_scheduled_job(&written)
                .await
                .expect("write");

            let read = harness
                .manager
                .schedule_store()
                .list_scheduled_jobs(harness.session_id)
                .await
                .expect("read back")
                .into_iter()
                .find(|job| job.id == written.id)
                .unwrap_or_else(|| panic!("{label}: the job is not there"));

            assert_eq!(read.session_id, written.session_id, "{label}: session");
            assert_eq!(
                read.schedule.spec(),
                written.schedule.spec(),
                "{label}: schedule"
            );
            assert_eq!(read.prompt, written.prompt, "{label}: prompt");
            assert_eq!(
                read.gate.as_ref().map(|gate| gate.spec()),
                written.gate.as_ref().map(|gate| gate.spec()),
                "{label}: gate"
            );
            assert_eq!(
                read.gate.as_ref().and_then(|gate| gate.last_output.clone()),
                written
                    .gate
                    .as_ref()
                    .and_then(|gate| gate.last_output.clone()),
                "{label}: gate baseline"
            );
            assert_eq!(
                read.gate.as_ref().map(|gate| gate.permission),
                written.gate.as_ref().map(|gate| gate.permission),
                "{label}: gate permission"
            );
            assert_eq!(read.created_at, written.created_at, "{label}: created_at");
            assert_eq!(
                read.last_fired_at, written.last_fired_at,
                "{label}: last_fired_at"
            );
            assert_eq!(
                read.next_fire_at, written.next_fire_at,
                "{label}: next_fire_at"
            );
            // Not written by `create_scheduled_job`, which is itself the point: the column has a
            // default and sits at the end of the row, so a shift lands here first.
            assert_eq!(read.attempts, 0, "{label}: attempts starts unclaimed");
        }
    }

    /// The headline missed-job rule: an outage does not become a burst. A 30-second job that was
    /// due six hours ago has 720 missed occurrences, and must produce exactly one turn.
    #[tokio::test]
    async fn a_long_outage_coalesces_into_a_single_fire() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("30s").expect("parses"),
                None,
                chrono::Duration::hours(6),
            )
            .await;

        harness.tick().await;

        let fired = harness.fired();
        assert_eq!(fired.len(), 1, "one turn, not 720");
        assert_eq!(fired[0].job_id, job.id);
        // Six hours of 30-second occurrences: the one due at the stored time, plus 720 after it.
        assert_eq!(
            fired[0].coalesced, 720,
            "the skipped occurrences are reported, not replayed"
        );
    }

    /// Coalescing bounds one job's backlog; this bounds the whole sweep's. A session that
    /// accumulated more watchers than the budget must not wake to a turn per job, and must not lose
    /// any of them either.
    #[tokio::test]
    async fn the_fire_budget_holds_jobs_over_without_losing_them() {
        let mut harness = SchedulerHarness::new().await;
        harness.config.max_consecutive_fires = 5;
        for _ in 0..8 {
            harness
                .overdue_job(
                    Schedule::parse_every("1h").expect("parses"),
                    None,
                    chrono::Duration::hours(6),
                )
                .await;
        }

        harness.tick().await;
        assert_eq!(harness.fired().len(), 5, "the budget bounds the burst");

        harness.tick().await;
        let fired = harness.fired();
        assert_eq!(fired.len(), 8, "and the next sweep takes the rest");
        let distinct: std::collections::HashSet<&str> =
            fired.iter().map(|record| record.job_id.as_str()).collect();
        assert_eq!(distinct.len(), 8, "every job fired exactly once");
    }

    /// Per session, not per sweep. A budget shared across sessions would let one conversation's
    /// backlog delay another's due job, which under `meka serve` is somebody else's job entirely.
    #[tokio::test]
    async fn the_fire_budget_is_per_session_rather_than_global() {
        let mut harness = SchedulerHarness::new().await;
        harness.config.max_consecutive_fires = 5;
        let other = harness.another_session().await;
        // The backlog is older, so it sorts first and would exhaust a global budget before the
        // quiet session's single job was ever reached.
        for _ in 0..6 {
            harness
                .overdue_job(
                    Schedule::parse_every("1h").expect("parses"),
                    None,
                    chrono::Duration::hours(6),
                )
                .await;
        }
        let lonely = harness
            .overdue_job_in(
                other,
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        let fired = harness.fired();
        assert_eq!(
            fired.len(),
            6,
            "five from the busy session, one from the other"
        );
        assert!(
            fired.iter().any(|record| record.job_id == lonely.id),
            "the quiet session's job was not held behind the busy one's backlog"
        );
    }

    /// A job the gate retires spends no turn, so it must not spend budget either. Otherwise a
    /// handful of quiet watchers would starve the one job that had something to report.
    #[tokio::test]
    async fn a_declining_gate_does_not_consume_the_fire_budget() {
        let mut harness = SchedulerHarness::new().await;
        harness.config.max_consecutive_fires = 2;
        // More overdue, so both are evaluated before the ungated job below.
        for _ in 0..2 {
            harness
                .overdue_job(
                    Schedule::parse_every("1h").expect("parses"),
                    Some(gate("exit 1", GatePredicate::Succeeded, None)),
                    chrono::Duration::hours(6),
                )
                .await;
        }
        let speaks = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        let fired = harness.fired();
        assert_eq!(fired.len(), 1, "the two silent gates cost nothing");
        assert_eq!(fired[0].job_id, speaks.id);
    }

    /// A host handing an occurrence back has not spent a turn on it, so the deferral must not spend
    /// budget either. Otherwise a `meka serve` whose session is held by a REPL would burn its whole
    /// per-sweep budget on jobs it never ran.
    #[tokio::test]
    async fn a_deferral_does_not_consume_the_fire_budget() {
        let mut harness = SchedulerHarness::new().await;
        harness.config.max_consecutive_fires = 1;
        for _ in 0..2 {
            harness
                .overdue_job(
                    Schedule::parse_every("1h").expect("parses"),
                    None,
                    chrono::Duration::hours(6),
                )
                .await;
        }

        let offered = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        run_due(
            &harness.manager,
            &harness.config,
            None,
            &NoResidents,
            &SchedulerScope::every_job(),
            &move |_wakeup: Wakeup| {
                let offered = offered.clone();
                async move {
                    // The first is handed back; only the second spends a turn.
                    match offered.fetch_add(1, std::sync::atomic::Ordering::Relaxed) {
                        0 => FireOutcome::Deferred,
                        _ => FireOutcome::Ran,
                    }
                }
            },
        )
        .await
        .expect("sweep runs");

        // Both jobs reached the host despite a budget of one, because the deferred one cost
        // nothing. The second is the only one whose schedule advanced.
        let still_due: Vec<_> = harness
            .jobs()
            .await
            .into_iter()
            .filter(|job| job.next_fire_at <= Utc::now())
            .collect();
        assert_eq!(
            still_due.len(),
            1,
            "the deferred job kept its occurrence; the other one spent its turn"
        );
    }

    /// The budget is checked before `prepare`, which is where a gate runs, so holding a job over
    /// must cost nothing, not even the shell command whose expense is half the reason gates
    /// exist.
    ///
    /// Observed through a side effect on the filesystem rather than through the job's stored gate
    /// baseline. Enforcing the budget *after* `prepare` and then restoring the job would run the
    /// command and put the baseline back, leaving every column identical to a job that was never
    /// touched. Only the command's own footprint tells the two apart.
    #[tokio::test]
    async fn a_held_over_job_does_not_run_its_gate() {
        let mut harness = SchedulerHarness::new().await;
        harness.config.max_consecutive_fires = 1;
        // Removed on drop rather than at the end of the test, so a failing assertion does not leak
        // it into the temp directory of whoever ran the suite.
        struct Probe(std::path::PathBuf);
        impl Drop for Probe {
            fn drop(&mut self) {
                std::fs::remove_file(&self.0).ok();
            }
        }
        let guard =
            Probe(std::env::temp_dir().join(format!("meka-gate-probe-{}", uuid::Uuid::new_v4())));
        let probe = guard.0.clone();
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::hours(6),
            )
            .await;
        let watcher = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(
                    &create_file_command(&probe),
                    GatePredicate::Changed,
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert!(
            !probe.exists(),
            "the held-over job's gate command never ran"
        );
        let held = harness
            .jobs()
            .await
            .into_iter()
            .find(|job| job.id == watcher.id)
            .expect("the held-over job is still there");
        assert_eq!(
            held.next_fire_at, watcher.next_fire_at,
            "and its schedule was not advanced, so it is still due"
        );

        // The probe is only evidence if it can fire at all: the next sweep has budget for it.
        harness.tick().await;
        assert!(probe.exists(), "and it runs once the budget has room");
        std::fs::remove_file(&probe).expect("clean up the probe");
    }

    /// The rule every host defers to. Withdrawing a failed fire's prompt is only safe when the job
    /// will produce it again; `prepare` deletes a one-shot's row *before* the turn runs, so for
    /// those the unanswered message is the last trace the reminder ever fired.
    #[test]
    fn only_a_recurring_job_lets_a_failed_fire_withdraw_its_prompt() {
        let recurring = |schedule: Schedule| ScheduledJob {
            attempts: 0,
            id: "7f3a1b2c".to_string(),
            session_id: uuid::Uuid::nil(),
            schedule,
            prompt: "check the news".to_string(),
            gate: None,
            created_at: at("2026-08-11T12:00:00Z"),
            last_fired_at: None,
            next_fire_at: at("2026-08-11T12:00:00Z"),
        };

        for schedule in [
            Schedule::parse_every("1h").expect("parses"),
            Schedule::parse_cron("0 9 * * 1-5").expect("parses"),
        ] {
            assert_eq!(
                recurring(schedule).prompt_retention(),
                crate::conversation::PromptRetention::Withdraw,
                "a recurring job regenerates its prompt"
            );
        }
        assert_eq!(
            recurring(Schedule::At(at("2026-08-11T12:00:00Z"))).prompt_retention(),
            crate::conversation::PromptRetention::Keep,
            "a one-shot's row is already gone; its prompt is all that is left"
        );
    }

    /// The `Every` and `Cron` arms must agree on what they count, or the same outage reports a
    /// different backlog depending on how the schedule happened to be written.
    #[test]
    fn occurrence_counting_agrees_between_interval_and_cron() {
        let from = at("2026-08-11T12:00:00Z");
        let to = at("2026-08-11T12:05:00Z");
        let every = occurrences_between(&Schedule::parse_every("1m").expect("parses"), from, to);
        let cron = occurrences_between(
            &Schedule::parse_cron("* * * * *").expect("parses"),
            from,
            to,
        );
        assert_eq!(every, 5);
        assert_eq!(every, cron);
    }

    /// After firing late, the next due time is measured from now rather than from the slot that was
    /// missed. Rescheduling from the missed slot would leave the job still overdue and fire it
    /// again on the very next tick, which is the burst this avoids.
    #[tokio::test]
    async fn a_late_recurring_job_reschedules_from_now() {
        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::hours(5),
            )
            .await;

        harness.tick().await;
        assert_eq!(harness.fired().len(), 1);

        let job = harness.jobs();
        let job = job.await;
        let job = job.first().expect("recurring job survives");
        assert!(
            job.next_fire_at > Utc::now(),
            "next fire must be in the future, not still overdue"
        );

        // A second sweep must find nothing.
        harness.tick().await;
        assert_eq!(
            harness.fired().len(),
            1,
            "no second fire on the same tick cycle"
        );
    }

    #[tokio::test]
    async fn a_one_shot_past_the_grace_period_is_dropped_not_delivered() {
        let harness = SchedulerHarness::new().await;
        let stale = Utc::now() - chrono::Duration::days(5);
        harness
            .overdue_job(Schedule::At(stale), None, chrono::Duration::days(5))
            .await;

        harness.tick().await;

        assert!(
            harness.fired().is_empty(),
            "a five-day-old reminder is noise, not a reminder"
        );
        assert!(harness.jobs().await.is_empty(), "and it is retired");
    }

    #[tokio::test]
    async fn a_one_shot_inside_the_grace_period_fires_and_reports_its_lateness() {
        let harness = SchedulerHarness::new().await;
        let due = Utc::now() - chrono::Duration::hours(3);
        harness
            .overdue_job(Schedule::At(due), None, chrono::Duration::hours(3))
            .await;

        harness.tick().await;

        let fired = harness.fired();
        assert_eq!(fired.len(), 1);
        assert!(fired[0].late_by >= chrono::Duration::hours(3) - chrono::Duration::seconds(5));
        assert!(harness.jobs().await.is_empty(), "one-shots retire on fire");
    }

    /// A gate that declines must still advance the schedule, or the job stays overdue and the gate
    /// runs on every single tick instead of on its interval.
    #[tokio::test]
    async fn a_declining_gate_reschedules_without_spending_a_turn() {
        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate("exit 1", GatePredicate::Succeeded, None)),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert!(harness.fired().is_empty(), "the condition was false");
        let jobs = harness.jobs().await;
        let job = jobs.first().expect("job survives");
        assert!(job.next_fire_at > Utc::now(), "but the schedule moved on");
        assert!(
            job.last_fired_at.is_none(),
            "evaluating is not firing; recording it as fired would misreport the job"
        );
    }

    /// The nastiest failure in the feature: a broken gate must not look like a quiet one. It does
    /// not fire, but it also must not be recorded as having fired.
    #[tokio::test]
    async fn a_broken_gate_does_not_fire_and_does_not_claim_to_have() {
        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(
                    "definitely-not-a-real-command-xyzzy",
                    GatePredicate::Succeeded,
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert!(harness.fired().is_empty());
        let jobs = harness.jobs().await;
        assert!(jobs.first().expect("job survives").last_fired_at.is_none());
    }

    /// A gate is authorized once, at `unrestricted`, and then persists as a row that any process
    /// executes on a timer. Nothing about the creating session's later downgrade (Shift+Tab to
    /// `read`, or a `meka serve --permission read` restart inheriting the job) can reach back to
    /// withdraw it, so the level travels on the row and is re-checked here. Asserted through a real
    /// filesystem side effect rather than through the returned outcome, because "did not fire" and
    /// "did not *run*" are different claims and only the second one is the security property. The
    /// scenario the feature exists for, driven the way production produces it: the gate carries the
    /// `Unrestricted` it was legitimately created with, and the *session* has since dropped to
    /// `read`.
    ///
    /// The sibling below hand-sets `gate.permission` to `Read`, which no creation path can produce:
    /// both `schedule_create` and the HTTP handler demand `Unrestricted` before writing the row,
    /// and nothing updates the column afterwards. That test alone would prove the mechanism on an
    /// input reality never supplies.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_gate_is_not_executed_once_the_session_drops_below_unrestricted() {
        let marker = std::env::temp_dir().join(format!("meka-gate-{}", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&marker);

        let harness =
            SchedulerHarness::at_session_permission(crate::permission::Permission::Read).await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                // Recorded `Unrestricted`, exactly as `schedule_create` would have written it.
                Some(gate(
                    &create_file_command(&marker),
                    GatePredicate::Succeeded,
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert!(
            !marker.exists(),
            "the gate command must not run at all once the host is below write"
        );
        let _ = std::fs::remove_file(&marker);
    }

    /// A recurring job stays on the grid it was created on, however late a pickup is.
    ///
    /// Passing `now` to `next_after` would add each sweep's pickup latency to the interval and keep
    /// it: at `poll_interval = 1s` an `every = "1s"` job would run at 2.0s, half the requested
    /// rate, because the tick one interval later lands microseconds early and the job waits a whole
    /// extra poll. This asserts the property directly rather than by timing a live scheduler, which
    /// would be flaky.
    #[test]
    fn a_recurring_schedule_advances_from_the_occurrence_it_delivered_not_from_now() {
        let every = Schedule::parse_every("5s").expect("parses");
        let delivered = chrono::Utc::now() - chrono::Duration::seconds(12);
        // Pickup is late, and not on a multiple of the interval.
        let now = delivered + chrono::Duration::milliseconds(12_345);

        let next = every
            .next_after_delivering(delivered, now)
            .expect("a recurring schedule always has a next occurrence");
        assert!(next > now, "the next occurrence must be in the future");
        let offset = (next - delivered).num_milliseconds();
        assert_eq!(
            offset % 5_000,
            0,
            "the next fire must sit on a whole multiple of the interval from the delivered \
             occurrence, got {offset}ms"
        );
        assert_eq!(
            offset, 15_000,
            "and it must be the *first* such multiple after now"
        );

        // The drifted shape, kept here so the difference is legible: anchoring on `now` yields a
        // time that is not on the grid at all.
        let drifted = every.next_after(now).expect("every always has a next");
        assert_ne!(
            (drifted - delivered).num_milliseconds() % 5_000,
            0,
            "anchoring on `now` should drift off the grid, or this test proves nothing"
        );

        // A long outage skips whole intervals rather than replaying them, and still lands on the
        // grid.
        let after_outage = delivered + chrono::Duration::seconds(3_601);
        let resumed = every
            .next_after_delivering(delivered, after_outage)
            .expect("still recurring");
        assert!(resumed > after_outage);
        assert_eq!((resumed - delivered).num_milliseconds() % 5_000, 0);

        // Cron is absolute wall-clock and must be untouched by any of this.
        let cron = Schedule::parse_cron("*/5 * * * *").expect("parses");
        assert_eq!(
            cron.next_after_delivering(delivered, now),
            cron.next_after(now),
            "a cron pattern's next occurrence does not depend on which one was just delivered"
        );
    }

    /// And the wiring: the row `prepare` writes back is on the grid, not one interval from now.
    ///
    /// The unit test above covers `next_after_delivering`; reverting the call site to
    /// `next_after(now)` would leave every other schedule test passing. What the user feels is this
    /// value, persisted.
    #[tokio::test]
    async fn a_fired_job_is_rescheduled_onto_its_own_grid() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("5s").expect("parses"),
                None,
                chrono::Duration::seconds(12),
            )
            .await;
        let delivered = job.next_fire_at;

        harness.tick().await;

        let rescheduled = harness
            .jobs()
            .await
            .into_iter()
            .find(|candidate| candidate.id == job.id)
            .expect("a recurring job lives on");
        let offset = (rescheduled.next_fire_at - delivered).num_milliseconds();
        assert_eq!(
            offset % 5_000,
            0,
            "the persisted next fire must be a whole number of intervals from the occurrence just \
             delivered, got {offset}ms -- anchoring on `now` instead makes every job drift"
        );
        assert!(
            rescheduled.next_fire_at > delivered,
            "and must be in the future relative to what was delivered"
        );
    }

    /// Narrowing `[permissions].enabled` disarms a gate whose session row still records a level
    /// the installation no longer permits.
    ///
    /// A row outlives the configuration that produced it. Every session `meka serve` creates
    /// persists its own level, so `--permission` on the host is a *default* and not a ceiling, and
    /// the only thing an operator can narrow that a row cannot exceed is the enabled set. Without
    /// the filter this test guards, that operator restarts, watches the session re-attach at
    /// `read` in the log, and the gate keeps firing at `unrestricted`, while the creation door
    /// two files over returns 403 for the very same authority.
    #[cfg(unix)]
    #[tokio::test]
    async fn narrowing_the_enabled_set_disarms_a_gate_the_row_still_authorizes() {
        let marker = std::env::temp_dir().join(format!("meka-gate-{}", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&marker);

        // The row still says `unrestricted`, and something has to refuse to believe it.
        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(
                    &create_file_command(&marker),
                    GatePredicate::Succeeded,
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;

        // What the operator did: narrowed the installation to `read` and restarted.
        let narrowed = crate::config::ResolvedScheduleConfig {
            enabled_permissions: crate::permission::EnabledPermissions::from_levels([
                crate::permission::Permission::Read,
            ])
            .expect("a single level is a valid set"),
            ..harness.config.clone()
        };
        harness.tick_with(narrowed, None).await;
        assert!(
            !marker.exists(),
            "a level the installation no longer enables must not authorize a gate"
        );

        // And the control: with `unrestricted` still enabled, the identical row does authorize the
        // gate, so the refusal above is the enabled set rather than some unrelated part of the
        // fixture. A second job, because the sweep above claimed the first one's occurrence by
        // advancing it: a declined gate still spends the occurrence, which is what stops a
        // refused watcher from re-running every poll.
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(
                    &create_file_command(&marker),
                    GatePredicate::Succeeded,
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;
        let permitted = crate::config::ResolvedScheduleConfig {
            enabled_permissions: crate::permission::EnabledPermissions::DEFAULT,
            ..harness.config.clone()
        };
        harness.tick_with(permitted, None).await;
        assert!(
            marker.exists(),
            "the same job must still fire while its recorded level is enabled"
        );
        let _ = std::fs::remove_file(&marker);
    }

    /// A panic in one fire must not stop the scheduler.
    ///
    /// Under `meka serve` the callback runs a whole agent turn, so everything the tool loop can do
    /// is inside the surface that can panic. Nothing joins this task, so losing it would produce
    /// no error anywhere: scheduled jobs would simply stop firing, for the life of the process, and
    /// the first sign would be a reminder that never arrived.
    #[tokio::test]
    async fn a_panicking_fire_does_not_stop_the_scheduler() {
        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1s").expect("parses"),
                None,
                chrono::Duration::seconds(30),
            )
            .await;

        let fires = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let config = crate::config::ResolvedScheduleConfig {
            poll_interval: std::time::Duration::from_millis(20),
            // A panicking turn leaves its claim to expire, so without a lease shorter than the
            // test's patience the same job would not come round twice. What is being asserted is
            // that the *loop* survives a panic, not how long the retry waits.
            claim_lease: std::time::Duration::from_millis(1),
            ..harness.config.clone()
        };
        let handle = spawn(
            std::sync::Arc::clone(&harness.manager),
            config,
            None,
            std::sync::Arc::new(NoResidents),
            SchedulerScope::every_job(),
            {
                let fires = std::sync::Arc::clone(&fires);
                move |_wakeup| {
                    let fires = std::sync::Arc::clone(&fires);
                    async move {
                        fires.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        panic!("the turn blew up");
                    }
                }
            },
        );

        // Two fires means the loop survived the first panic, which is the whole claim.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while fires.load(std::sync::atomic::Ordering::SeqCst) < 2 {
            assert!(
                std::time::Instant::now() < deadline,
                "the scheduler stopped after {} fire(s)",
                fires.load(std::sync::atomic::Ordering::SeqCst),
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        handle.abort();
    }

    /// The other half of a withdrawn gate: the job does not fire either.
    ///
    /// "Did not run" and "did not fire" are separate claims and both matter. A gate is the
    /// condition on the job, so a gate that cannot be evaluated has not passed, and delivering the
    /// prompt regardless turns a conditional job into an unconditional one: on an `every = "1m"`
    /// watcher, a turn a minute for as long as the session stayed below `unrestricted`.
    #[tokio::test]
    async fn a_job_whose_gate_cannot_be_run_does_not_fire_regardless() {
        let harness =
            SchedulerHarness::at_session_permission(crate::permission::Permission::Read).await;
        harness
            .overdue_job(
                Schedule::parse_every("1m").expect("parses"),
                Some(gate("true", GatePredicate::Succeeded, None)),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert!(
            harness.fired().is_empty(),
            "an unevaluated gate is not a passed gate",
        );
        let jobs = harness.jobs().await;
        let job = jobs.first().expect("the job survives for the next sweep");
        assert!(
            job.last_fired_at.is_none(),
            "and it must not be recorded as having fired",
        );
        assert!(
            job.next_fire_at > Utc::now(),
            "the occurrence is spent, so a restored session does not get a backlog",
        );
    }

    /// The companion: the same job, same recorded level, on a host still at `unrestricted`, runs.
    /// Without this the test above would pass just as well if gates never ran at all.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_gate_is_executed_while_the_host_still_holds_write() {
        let marker = std::env::temp_dir().join(format!("meka-gate-{}", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&marker);

        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(
                    &create_file_command(&marker),
                    GatePredicate::Succeeded,
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert!(marker.exists(), "a fully authorized gate must still run");
        let _ = std::fs::remove_file(&marker);
    }

    /// A row that could not be read grants nothing, and neither does one that is gone. Read as "no
    /// level recorded", a failed read would fall back to the host's level, so a `SQLITE_BUSY` on a
    /// session recorded at `read` would evaluate its gates at `unrestricted` for that sweep.
    #[tokio::test]
    async fn an_unreadable_session_row_fails_closed() {
        let harness = SchedulerHarness::new().await;
        assert_eq!(
            live_permission(SessionLookup::Failed, &harness.config, harness.session_id),
            crate::permission::Permission::None
        );
        assert_eq!(
            live_permission(
                SessionLookup::Read(None),
                &harness.config,
                harness.session_id
            ),
            crate::permission::Permission::None,
            "a row that is gone grants nothing either"
        );
    }

    /// The clock is read per job. A sweep contains the turns it fires, so the second job's
    /// `fired_at` and `next_fire_at` would otherwise be computed from an instant the first job's
    /// turn had left behind; a recurring job would then be advanced into the past and fire again on
    /// the next tick.
    #[tokio::test]
    async fn each_job_in_a_sweep_is_timed_from_its_own_instant() {
        let harness = SchedulerHarness::new().await;
        let first = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(2),
            )
            .await;
        let other_session = harness.another_session().await;
        let second = harness
            .overdue_job_in(
                other_session,
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(1),
            )
            .await;

        harness.tick_slowly(Duration::from_millis(1100)).await;

        assert_eq!(harness.fired().len(), 2, "both jobs fire in the one sweep");
        let fired_at = |session: uuid::Uuid, id: &str| {
            let store = harness.manager.schedule_store();
            let id = id.to_string();
            async move {
                store
                    .list_scheduled_jobs(session)
                    .await
                    .expect("list")
                    .into_iter()
                    .find(|job| job.id == id)
                    .and_then(|job| job.last_fired_at)
                    .expect("fired")
            }
        };
        let first_fired = fired_at(harness.session_id, &first.id).await;
        let second_fired = fired_at(other_session, &second.id).await;
        assert!(
            second_fired - first_fired >= chrono::Duration::seconds(1),
            "the second job's instant must follow the first's turn: {first_fired} vs {second_fired}"
        );
    }

    /// A row that records no level runs nothing, rather than whatever the polling process was
    /// started with.
    #[tokio::test]
    async fn a_row_with_no_level_fires_nothing() {
        let harness = SchedulerHarness::new().await;
        let bare = harness
            .manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        harness
            .manager
            .erase_recorded_permission(bare)
            .await
            .expect("a row no door writes");
        harness
            .overdue_job_in(
                bare,
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;
        harness.tick().await;
        assert!(
            harness.fired().is_empty(),
            "a session with no recorded level must not be woken"
        );
    }

    /// A session row the sweep cannot read decides nothing, so the occurrence stays. Read as a
    /// session at `none`, the job would be declined, a recurring one advanced past a probe that
    /// never ran, and the operator told to raise a session that may be at `unrestricted`.
    #[tokio::test]
    async fn an_unreadable_session_row_keeps_the_occurrence() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("meka.db");
        let manager = std::sync::Arc::new(
            crate::store::Store::open(Some(&path), &Default::default())
                .await
                .expect("open"),
        );
        let session_id =
            SchedulerHarness::session_at(&manager, crate::permission::Permission::Unrestricted)
                .await;
        let harness = SchedulerHarness {
            manager,
            session_id,
            config: crate::config::ResolvedScheduleConfig::default(),
            fired: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        };
        let job = harness
            .overdue_job(
                Schedule::parse_every("6h").expect("parses"),
                None,
                chrono::Duration::minutes(1),
            )
            .await;

        // Hidden under a live connection, so the row read fails the way a locked store does.
        let hide = |hidden: bool| {
            let statement = if hidden {
                "ALTER TABLE sessions RENAME TO sessions_hidden;"
            } else {
                "ALTER TABLE sessions_hidden RENAME TO sessions;"
            };
            rusqlite::Connection::open(&path)
                .expect("second connection")
                .execute_batch(statement)
                .expect("rename");
        };
        hide(true);
        harness.tick().await;
        hide(false);

        assert!(harness.fired().is_empty(), "nothing can be decided");
        let after = harness.jobs().await;
        let after = after.first().expect("the job survives");
        assert_eq!(
            after.next_fire_at, job.next_fire_at,
            "the occurrence is neither spent nor advanced; the next sweep decides it"
        );
    }

    /// A gate that logs more than the stderr cap still answers: its stderr is drained past the cap
    /// and only the head kept. A `take` at the cap would close the read end, so the child's next
    /// log line would be `SIGPIPE` and the gate would die before it could print its answer.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_chatty_stderr_does_not_kill_the_probe() {
        // One process, so a death on stderr takes the answer with it: 220 KB of logging, well past
        // the cap and any pipe buffer, then the answer on stdout.
        let outcome = crate::scheduler::run_shell_probe(
            "awk 'BEGIN { for (i = 0; i < 20000; i++) print \"xxxxxxxxxx\" > \"/dev/stderr\"; \
             print \"answer\" }'",
            Duration::from_secs(20),
            None,
        )
        .await
        .expect("the probe completes");
        assert_eq!(outcome.text.trim(), "answer");
    }

    /// A gate whose tool cannot be resolved *yet* is left for the next sweep, not declined.
    /// Declining would spend the occurrence, so a `6h` job due while its server was still
    /// connecting would be advanced six hours without its probe ever running.
    #[tokio::test]
    async fn a_gate_whose_server_is_still_connecting_keeps_its_occurrence() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("6h").expect("parses"),
                Some(Gate {
                    probe: tool_probe(),
                    predicate: GatePredicate::Succeeded,
                    last_output: None,
                    permission: crate::permission::Permission::Unrestricted,
                }),
                chrono::Duration::minutes(1),
            )
            .await;
        harness
            .tick_with(
                harness.config.clone(),
                Some(std::sync::Arc::new(StillConnecting)),
            )
            .await;

        assert!(harness.fired().is_empty(), "nothing can be evaluated yet");
        let after = harness.jobs().await;
        let after = after.first().expect("the job survives");
        assert_eq!(
            after.next_fire_at, job.next_fire_at,
            "the occurrence is neither spent nor advanced; the next sweep decides it"
        );
    }

    /// A gate that never stops producing is cut off at the parse limit, not held in memory until
    /// its time budget runs out. `wait_with_output` reads the whole pipe, so `yes` would take the
    /// host's memory inside its own timeout.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_runaway_probe_is_cut_off_at_the_parse_limit() {
        let started = std::time::Instant::now();
        let outcome = crate::scheduler::run_shell_probe("yes", Duration::from_secs(20), None)
            .await
            .expect("the probe still produces an outcome");
        assert!(
            started.elapsed() < Duration::from_secs(15),
            "the probe waited for its timeout instead of stopping the producer: {:?}",
            started.elapsed()
        );
        assert!(!outcome.text.is_empty());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_gate_whose_authority_was_withdrawn_is_not_executed() {
        let marker = std::env::temp_dir().join(format!("meka-gate-{}", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&marker);

        let harness = SchedulerHarness::new().await;
        let mut withdrawn = gate(
            &create_file_command(&marker),
            GatePredicate::Succeeded,
            None,
        );
        withdrawn.permission = crate::permission::Permission::Read;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(withdrawn),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert!(
            !marker.exists(),
            "a gate authorized at read must never reach the shell"
        );
        // And the occurrence is declined rather than delivered ungated; see
        // `a_job_whose_gate_cannot_be_run_does_not_fire_regardless` for why.
        assert!(harness.fired().is_empty());

        let _ = std::fs::remove_file(&marker);
    }

    /// An unreadable `gate_permission` decodes to the level that authorizes nothing.
    ///
    /// This is the arm a row holding a value this build does not resolve hits. It fails closed
    /// (`Permission::None` authorizes no gate), and a fallback to `Unrestricted` would let every
    /// upgraded database silently regain unattended arbitrary shell. The fixtures all plant valid
    /// values, so the one value that matters on upgrade needs its own test.
    #[test]
    fn an_unreadable_gate_permission_authorizes_nothing() {
        let row = |permission: Option<&str>| ScheduledJobRow {
            attempts: 0,
            id: "7f3a1b2c-0000-0000-0000-000000000000".to_string(),
            session_id: uuid::Uuid::nil().to_string(),
            kind: "every".to_string(),
            spec: "1h".to_string(),
            prompt: "do the thing".to_string(),
            gate_kind: Some("shell".to_string()),
            gate_spec_json: Some(r#"{"shell":{"command":"true"},"when":"succeeded"}"#.to_string()),
            gate_last_output: None,
            gate_permission: permission.map(str::to_string),
            created_at: Utc::now().to_rfc3339(),
            last_fired_at: None,
            next_fire_at: Utc::now().to_rfc3339(),
        };

        // A value no build of meka resolves.
        let unreadable = row(Some("elevated"))
            .decode()
            .expect("the row still decodes");
        let gate = unreadable
            .gate
            .expect("the gate survives; only its level is unreadable");
        assert_eq!(gate.permission, crate::permission::Permission::None);
        assert!(
            !gate.permission.allows_unattended_shell(),
            "an unreadable level must authorize no gate, or an upgrade re-arms every one of them"
        );

        // Absent is the same answer, for the same reason.
        let absent = row(None).decode().expect("decodes");
        assert_eq!(
            absent.gate.expect("gate").permission,
            crate::permission::Permission::None
        );

        // The control: a level meka still reads survives intact.
        let current = row(Some("unrestricted")).decode().expect("decodes");
        assert_eq!(
            current.gate.expect("gate").permission,
            crate::permission::Permission::Unrestricted
        );
    }

    /// The session's own recorded level decides, and a withdrawal reaches every process that
    /// polls the row.
    ///
    /// This is the cross-process half of the withdrawal. Were the polling process's own startup
    /// flag to stand in for a row with no level, a `meka serve` sharing the data directory would
    /// use that daemon's `--permission`, not anything the user touched: Shift+Tab-ing the REPL
    /// down would stop the gate in the REPL and leave serve firing it, unattended and unsandboxed,
    /// which is the opposite of what `scheduling.md` promises. The row is written at
    /// session creation and on every level change, and it is the only thing read.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_withdrawn_session_refuses_its_gate() {
        let marker = std::env::temp_dir().join(format!("meka-gate-{}", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&marker);

        let harness = SchedulerHarness::new().await;
        // Authorized when it was created, exactly as `schedule_create` would have written it.
        let authorized = gate(
            &create_file_command(&marker),
            GatePredicate::Succeeded,
            None,
        );
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(authorized),
                chrono::Duration::minutes(5),
            )
            .await;

        // The session has since been withdrawn to `read`: the row the REPL now keeps current.
        harness
            .manager
            .update_session(harness.session_id, crate::store::SessionPatch {
                permission: Some("read".parse().expect("a level")),
                ..Default::default()
            })
            .await
            .expect("record the withdrawal");

        harness.tick().await;

        assert!(
            !marker.exists(),
            "a withdrawn session must stop its gate, or Shift+Tab withdraws nothing while a daemon \
             is up"
        );
        assert!(harness.fired().is_empty());

        let _ = std::fs::remove_file(&marker);
    }

    /// The companion to the above: at `unrestricted` the same gate does run, so the refusal is
    /// about the recorded authority and not about gates having quietly stopped working.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_gate_that_still_holds_unrestricted_is_executed() {
        let marker = std::env::temp_dir().join(format!("meka-gate-{}", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&marker);

        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(
                    &create_file_command(&marker),
                    GatePredicate::Succeeded,
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert!(
            marker.exists(),
            "a gate at unrestricted permission must run"
        );
        let _ = std::fs::remove_file(&marker);
    }

    /// A session this process holds answers for its level from its cell, not its row. The row is
    /// written back on every change and the write can fail; until it lands the row says
    /// `unrestricted` for a session the user has just dropped to `read`, and the poller in the
    /// same process would fire a shell gate on the strength of it.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_resident_sessions_live_level_outranks_its_row() {
        let marker = std::env::temp_dir().join(format!("meka-gate-{}", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&marker);

        // The row still says `unrestricted`, as it does after a failed write.
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(
                    &create_file_command(&marker),
                    GatePredicate::Succeeded,
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;

        let wakeup = prepare(
            &harness.manager,
            &harness.config,
            None,
            &ResidentAt(crate::permission::Permission::Read),
            job,
            Utc::now(),
        )
        .await
        .expect("prepare runs");

        assert!(
            wakeup.is_none(),
            "the cell says `read`, so the gate is withheld"
        );
        assert!(
            !marker.exists(),
            "and its command never ran, whatever the row says"
        );
        let _ = std::fs::remove_file(&marker);
    }

    /// A job on a session at `none` does not fire, gate or no gate.
    ///
    /// The turn it would spend can do nothing: every tool is refused at dispatch, so it reads
    /// nothing, changes nothing, and its `schedule_cancel` is refused too, leaving it unable to
    /// stop itself being woken again. Registration does not depend on the level, so it *sees* the
    /// job in `[Scheduled]` and is offered the tool; the refusal is at the point of use, which is
    /// the worst of both. An ungated `every = "5s"` job would be a turn every five seconds for as
    /// long as the session sat there, stoppable only by an operator. Ungated is the case that
    /// matters here: a gated job is already refused by the authority check.
    #[tokio::test]
    async fn an_ungated_job_does_not_fire_on_a_session_at_none() {
        let harness = SchedulerHarness::new().await;
        harness
            .manager
            .update_session(harness.session_id, crate::store::SessionPatch {
                permission: Some("none".parse().expect("a level")),
                ..Default::default()
            })
            .await
            .expect("record the level the session was set to");
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert!(
            harness.fired().is_empty(),
            "waking a model that cannot act, and cannot cancel the job that woke it, is a turn \
             spent to no purpose"
        );
    }

    /// A one-shot held back for authority survives; it is not spent.
    ///
    /// A one-shot is retired the instant it comes due, *before* the gate is consulted, which is
    /// right when the gate ran and said no: its moment has passed either way. It is wrong when the
    /// gate was never consulted at all. Lowering a session for one minute would destroy every
    /// one-shot that happened to come due in that minute, while the log line says "not fired",
    /// which reads as held rather than deleted. The gate-error path makes this distinction for a
    /// gate that errored; the two authority refusals belong on the same side of it.
    #[tokio::test]
    async fn a_one_shot_held_for_authority_is_kept_rather_than_retired() {
        for (label, level, gate_on_it) in [
            ("session at none, ungated", "none", false),
            ("session below the gate's bar", "read", true),
        ] {
            let harness = SchedulerHarness::new().await;
            harness
                .manager
                .update_session(harness.session_id, crate::store::SessionPatch {
                    permission: Some(level.parse().expect("a level")),
                    ..Default::default()
                })
                .await
                .expect("record the level the session was set to");
            harness
                .overdue_job(
                    Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                    gate_on_it.then(|| gate("true", GatePredicate::Succeeded, None)),
                    chrono::Duration::minutes(1),
                )
                .await;

            harness.tick().await;

            assert!(harness.fired().is_empty(), "{label}: must not fire");
            assert_eq!(
                harness.jobs().await.len(),
                1,
                "{label}: the reminder was never evaluated, so it must still be there once the \
                 level is restored"
            );
        }
    }

    /// A refused job's row is never deleted, not even for the instant it took to put it back.
    ///
    /// Kept-and-restored and never-touched leave identical rows, which is why the resurrection this
    /// prevents was invisible: claiming by deleting a one-shot puts the refusal after the delete,
    /// and the restore is then an `INSERT` that cannot tell "I deleted this a moment ago" from "the
    /// user canceled it in between", so a `schedule_cancel` landing in that window is silently
    /// undone while both cancel doors report success.
    ///
    /// SQLite's `rowid` is the observable: a re-`INSERT` takes `max(rowid) + 1`, so a row that kept
    /// its rowid was never deleted. Restoring the delete-then-restore shape fails this with two
    /// different values.
    ///
    /// The second job is not decoration. With one row in the table the deleted rowid is also `max +
    /// 1`, so SQLite hands the same value straight back and the assertion holds under both
    /// orderings.
    #[tokio::test]
    async fn a_refused_one_shot_keeps_its_row_rather_than_being_deleted_and_restored() {
        let harness = SchedulerHarness::new().await;
        harness
            .manager
            .update_session(harness.session_id, crate::store::SessionPatch {
                permission: Some("read".parse().expect("a level")),
                ..Default::default()
            })
            .await
            .expect("record the level the session was set to");
        let job = harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                Some(gate("true", GatePredicate::Succeeded, None)),
                chrono::Duration::minutes(1),
            )
            .await;
        // Behind it in the table and not due, so it is never considered and only ever raises
        // `max(rowid)`.
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                -chrono::Duration::hours(1),
            )
            .await;

        let rowid = |id: String| {
            let store = harness.manager.schedule_store();
            async move {
                store
                    .connection
                    .call(move |connection| {
                        connection.query_row(
                            "SELECT rowid FROM scheduled_jobs WHERE id = ?1",
                            rusqlite::params![id],
                            |row| row.get::<_, i64>(0),
                        )
                    })
                    .await
                    .expect("the row is there")
            }
        };

        let before = rowid(job.id.clone()).await;
        harness.tick().await;
        let after = rowid(job.id.clone()).await;

        assert_eq!(
            before, after,
            "a held one-shot must keep its row: deleting it to put it back is what loses a cancel \
             issued in between"
        );
    }

    /// A gate whose probe keeps breaking is reported to the model, once the breakage is standing.
    ///
    /// Authority is not the commonest way a watcher dies. A server that changed its schema, a
    /// command that was uninstalled, a pointer into a result that stopped being JSON: each errors
    /// on every evaluation and each is indistinguishable, from the model's side, from a healthy
    /// watcher with nothing to report, which is the whole reason the marker exists. The first
    /// failure is deliberately silent, because one failure is as often a blip as a break.
    #[tokio::test]
    async fn a_gate_whose_probe_keeps_failing_is_reported_after_the_second_failure() {
        let harness = SchedulerHarness::new().await;
        harness
            .manager
            .update_session(harness.session_id, crate::store::SessionPatch {
                permission: Some("unrestricted".parse().expect("a level")),
                ..Default::default()
            })
            .await
            .expect("record the level the session was set to");
        // A pointer into output that is not JSON: an error, not an answer, on every evaluation.
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(
                    "echo not-json-at-all",
                    GatePredicate::At {
                        pointer: "/chats".to_string(),
                        is: PointerTest::NotEmpty,
                    },
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;
        let level = crate::permission::Permission::Unrestricted;

        harness.tick().await;
        assert_eq!(
            job_withheld_reason(harness.manager.scheduler_memory(), &job, level, None),
            None,
            "one failure is a blip, and marking a job dead for it would flap"
        );

        harness.overdue_now(&job.id).await;
        harness.tick().await;
        let reported = job_withheld_reason(harness.manager.scheduler_memory(), &job, level, None)
            .unwrap_or_default();
        assert!(
            reported.contains("keeps failing"),
            "a standing breakage is the model's to act on: {reported}"
        );

        // And the sentence does not move once it has been said. Every reader compares it by
        // equality, so a running total in it would make `render_world_state_diff` re-announce the
        // job to the model on every single failed evaluation.
        harness.overdue_now(&job.id).await;
        harness.tick().await;
        assert_eq!(
            job_withheld_reason(harness.manager.scheduler_memory(), &job, level, None)
                .unwrap_or_default(),
            reported,
            "a third failure says exactly what the second did"
        );
        assert!(
            reported.contains("not JSON") || reported.contains("did not return JSON"),
            "and it has to say what broke: {reported}"
        );

        // A working evaluation ends it, so the marker tracks the gate rather than accumulating.
        harness.overdue_now(&job.id).await;
        // Single-quoted, or `sh` eats the quotes and the probe emits `{chats: []}`, which is not
        // JSON either.
        harness
            .rewrite_gate(&job.id, "echo '{\"chats\": []}'")
            .await;
        harness.tick().await;
        assert_eq!(
            job_withheld_reason(harness.manager.scheduler_memory(), &job, level, None),
            None,
            "the probe answered, so there is nothing left to report"
        );
    }

    /// A crash between the claim and the delivery costs a retry, not the job.
    ///
    /// This is what the lease is for. Consuming the row to claim it (advancing a recurring job's
    /// `next_fire_at`, deleting a one-shot's row outright) leaves the occurrence spent when a
    /// host dies before the turn runs, and for a one-shot the reminder is gone with nothing
    /// anywhere to recover it from. Nothing swept for it either, unlike `background_tasks`, which
    /// is marked `interrupted` when a process takes the session lock.
    #[tokio::test]
    async fn a_crash_between_the_claim_and_the_turn_does_not_lose_the_job() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                None,
                chrono::Duration::minutes(1),
            )
            .await;
        let store = harness.manager.schedule_store();

        // A host takes the occurrence and is killed before it can deliver: no completion, no
        // release, just a lease nobody will ever come back for.
        let died_at = Utc::now() - chrono::Duration::hours(2);
        assert!(
            store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-that-died",
                    died_at,
                    died_at + chrono::Duration::hours(1),
                )
                .await
                .expect("claim")
        );

        harness.tick().await;

        assert_eq!(
            harness.fired().len(),
            1,
            "the reminder is delivered by the next host once the dead one's lease expires"
        );
        assert!(
            harness.jobs().await.is_empty(),
            "and retired properly afterwards, rather than left leased forever"
        );
    }

    /// A job whose turn keeps taking the host down is parked rather than retried forever.
    ///
    /// The counterpart the lease requires. A lease hands the same occurrence back every time, so
    /// without a ceiling a prompt that kills the process is claimed again on every sweep, forever.
    /// The row is kept rather than deleted: it stays listed, cancelable, and reported as held,
    /// because meka cannot tell a poisonous prompt from an unlucky one and destroying a user's job
    /// on that guess would be worse.
    #[tokio::test]
    async fn a_job_that_keeps_crashing_its_host_is_parked_rather_than_retried_forever() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;
        let store = harness.manager.schedule_store();

        // Every claim ends in a crash: taken, never completed, never voluntarily handed back.
        for attempt in 1..=MAX_CLAIM_ATTEMPTS {
            let now = Utc::now();
            assert!(
                store
                    .claim_occurrence(
                        &job.id,
                        job.next_fire_at,
                        &format!("host-{attempt}"),
                        now,
                        now - chrono::Duration::seconds(1),
                    )
                    .await
                    .expect("claim"),
                "attempt {attempt}: an expired lease is takeable"
            );
        }

        harness.tick().await;

        assert!(
            harness.fired().is_empty(),
            "the ceiling is reached, so the job is not handed to a fourth host to kill"
        );
        let parked = harness.jobs().await;
        let parked = parked.first().expect("and the job is kept, not destroyed");
        assert_eq!(parked.attempts, MAX_CLAIM_ATTEMPTS);
        assert_eq!(
            job_withheld_reason(
                harness.manager.scheduler_memory(),
                parked,
                crate::permission::Permission::Unrestricted,
                None
            )
            .as_deref()
            .map(|reason| reason.contains("claims ended without delivering")),
            Some(true),
            "and every surface says so, because a parked job that looks healthy is the thing this \
             whole marker exists to prevent"
        );
    }

    /// A stale *completion* is refused too, not just a stale release.
    ///
    /// The completion of a one-shot is the one `DELETE` the scheduler issues, so an unscoped one
    /// would let a host whose lease expired retire a job the current holder is still delivering:
    /// the reminder vanishing mid-turn, from a write issued by a host that no longer owns it.
    #[tokio::test]
    async fn a_stale_completion_does_not_retire_another_hosts_job() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                None,
                chrono::Duration::minutes(1),
            )
            .await;
        let store = harness.manager.schedule_store();

        let long_ago = Utc::now() - chrono::Duration::hours(2);
        assert!(
            store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-a",
                    long_ago,
                    long_ago + chrono::Duration::minutes(1),
                )
                .await
                .expect("claim")
        );
        let now = Utc::now();
        assert!(
            store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-b",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("host B takes the expired lease")
        );

        // Host A finally finishes and retires the one-shot it thinks it delivered.
        store
            .complete_claim(&job.id, "host-a", None, Some(Utc::now()), None)
            .await
            .expect("complete");

        assert_eq!(
            harness.jobs().await.len(),
            1,
            "host B is still delivering it: a stale completion may not retire the row"
        );
    }

    /// A turn that panics leaves its claim to expire, and is not forgiven for it.
    ///
    /// Two properties in one shape, because they are the same trade. Leaving the lease is what
    /// spaces the retries: giving the occurrence back at once leaves the row due on the next
    /// sweep, so three panics arrive within `3 * poll_interval` (half a minute at the default)
    /// and park a recurring job that `missed_grace` will never retire. *Not* resetting the attempt
    /// count is what stops the same panic being retried forever once the spacing is in place.
    #[tokio::test]
    async fn a_panicking_turn_leaves_its_claim_to_expire_and_still_counts_against_the_ceiling() {
        let mut harness = SchedulerHarness::new().await;
        // Expired by the time the next sweep looks, so one sweep stands in for one `claim_lease`.
        // Set on the resolved config directly because `validate` refuses anything this short.
        harness.config.claim_lease = Duration::from_millis(1);
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;

        let attempts_after = |attempt: u32| {
            let manager = harness.manager.clone();
            let config = harness.config.clone();
            let id = job.id.clone();
            async move {
                run_due(
                    &manager,
                    &config,
                    None,
                    &NoResidents,
                    &SchedulerScope::every_job(),
                    &move |_wakeup: Wakeup| async move {
                        panic!("the turn blew up");
                    },
                )
                .await
                .expect("the sweep survives the panic");
                tokio::time::sleep(Duration::from_millis(3)).await;
                manager
                    .schedule_store()
                    .list_all_scheduled_jobs()
                    .await
                    .expect("list")
                    .into_iter()
                    .find(|job| job.id == id)
                    .unwrap_or_else(|| panic!("attempt {attempt}: the job survives the panic"))
                    .attempts
            }
        };

        assert_eq!(
            attempts_after(1).await,
            1,
            "the crash is counted rather than forgiven"
        );
        assert_eq!(
            attempts_after(2).await,
            2,
            "a second panic counts again: forgiving it would retry this prompt forever"
        );
        assert_eq!(attempts_after(3).await, MAX_CLAIM_ATTEMPTS);

        // The ceiling is reached, so the fourth sweep does not hand it to another turn to kill.
        harness.tick().await;
        assert!(harness.fired().is_empty(), "and the job is parked");
    }

    /// The other half of the reorder: a *recurring* job refused for authority still spends the
    /// occurrence it came due for.
    ///
    /// Moving the checks above the claim must not turn a held job into one that sits permanently
    /// due, accumulating a backlog it would report the moment it was authorized again. Only the
    /// one-shot is spared, because only the one-shot's claim destroys anything.
    #[tokio::test]
    async fn a_refused_recurring_job_still_spends_its_occurrence() {
        let harness = SchedulerHarness::new().await;
        harness
            .manager
            .update_session(harness.session_id, crate::store::SessionPatch {
                permission: Some("read".parse().expect("a level")),
                ..Default::default()
            })
            .await
            .expect("record the level the session was set to");
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate("true", GatePredicate::Succeeded, None)),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        let after = harness.jobs().await;
        let [refused] = after.as_slice() else {
            panic!("the job survives a refusal: {after:?}");
        };
        assert!(harness.fired().is_empty(), "it must not have fired");
        assert!(
            refused.next_fire_at > job.next_fire_at,
            "the occurrence is spent, exactly as it is when a gate runs and says no"
        );
    }

    /// The companion, so the refusal above is about `none` and not about ungated jobs having
    /// quietly stopped working: one rung up, the same job fires.
    #[tokio::test]
    async fn an_ungated_job_fires_on_a_session_at_read() {
        let harness = SchedulerHarness::new().await;
        harness
            .manager
            .update_session(harness.session_id, crate::store::SessionPatch {
                permission: Some("read".parse().expect("a level")),
                ..Default::default()
            })
            .await
            .expect("record the level the session was set to");
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        assert_eq!(
            harness.fired().len(),
            1,
            "`read` can still read, report and cancel, so the reminder is worth waking for"
        );
    }

    /// The fire door has to ask the same question the two creation doors ask.
    ///
    /// Both creation doors call `gate_probe_is_authorized`; a fire door that instead demands
    /// `unrestricted` whatever the probe is would *accept* a tool gate at `read` and then decline
    /// it on every tick forever, warning about an unattended shell command the job does not have.
    /// The headline case, `mcp__…__unseen` at `read`, would never call its probe once.
    ///
    /// Every other fire-time test here uses a shell gate at `unrestricted`, where the two checks
    /// agree, so only this one can tell them apart.
    #[tokio::test]
    async fn a_read_only_tool_gate_fires_at_read() {
        let harness =
            SchedulerHarness::at_session_permission(crate::permission::Permission::Read).await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(Gate {
                    probe: tool_probe(),
                    predicate: GatePredicate::Succeeded,
                    last_output: None,
                    permission: crate::permission::Permission::Read,
                }),
                chrono::Duration::minutes(5),
            )
            .await;

        harness
            .tick_with(
                crate::config::ResolvedScheduleConfig {
                    ..harness.config.clone()
                },
                Some(std::sync::Arc::new(FixedTools(Some(
                    crate::permission::Permission::Read,
                )))),
            )
            .await;

        assert_eq!(
            harness.fired().len(),
            1,
            "a read-only tool gate must fire at `read`, which is the entire point of not holding \
             every probe to the shell bar"
        );
    }

    /// The companion, and the user's second scenario: the same job, after the operator retuned the
    /// tool above `read`. Withdrawn at fire time, without the row changing.
    #[tokio::test]
    async fn a_tool_gate_stops_firing_once_the_tool_resolves_above_read() {
        let harness =
            SchedulerHarness::at_session_permission(crate::permission::Permission::Read).await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(Gate {
                    probe: tool_probe(),
                    predicate: GatePredicate::Succeeded,
                    last_output: None,
                    permission: crate::permission::Permission::Read,
                }),
                chrono::Duration::minutes(5),
            )
            .await;

        harness
            .tick_with(
                crate::config::ResolvedScheduleConfig {
                    ..harness.config.clone()
                },
                Some(std::sync::Arc::new(FixedTools(Some(
                    crate::permission::Permission::Unrestricted,
                )))),
            )
            .await;

        assert!(
            harness.fired().is_empty(),
            "a tool that no longer resolves to `read` must stop being a gate, or retuning \
             `tool_permissions` means nothing to a job already on the timer"
        );
    }

    /// A one-shot is retired the moment it comes due, before its gate is consulted: its moment has
    /// passed either way. The writes that follow a fire must therefore tolerate the row being gone,
    /// rather than issuing unconditionally and relying on the updates happening to match nothing.
    #[tokio::test]
    async fn a_one_shot_with_a_declining_gate_is_retired_without_firing() {
        let harness = SchedulerHarness::new().await;
        let due = Utc::now() - chrono::Duration::minutes(1);
        harness
            .overdue_job(
                Schedule::At(due),
                Some(gate("exit 1", GatePredicate::Succeeded, None)),
                chrono::Duration::minutes(1),
            )
            .await;

        harness.tick().await;

        assert!(harness.fired().is_empty(), "the condition was false");
        assert!(
            harness.jobs().await.is_empty(),
            "and the one-shot is gone rather than left to retry forever"
        );
    }

    #[tokio::test]
    async fn gate_output_rides_along_to_the_turn() {
        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate("echo ci-red", GatePredicate::Changed, None)),
                chrono::Duration::minutes(5),
            )
            .await;

        harness.tick().await;

        let fired = harness.fired();
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].gate_output.as_deref(), Some("ci-red"));
        // The baseline is persisted, so an unchanged second evaluation stays quiet.
        let jobs = harness.jobs().await;
        assert_eq!(
            jobs.first()
                .and_then(|job| job.gate.as_ref())
                .and_then(|gate| gate.last_output.as_deref()),
            Some("ci-red")
        );
    }

    /// The REPL only owns the conversation it has open; a job belonging to another session must be
    /// left for whichever host can actually run it.
    #[tokio::test]
    async fn session_scope_ignores_jobs_from_other_sessions() {
        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;

        let fired = harness.fired.clone();
        run_due(
            &harness.manager,
            &harness.config,
            None,
            &NoResidents,
            &SchedulerScope::OneSession(uuid::Uuid::new_v4()),
            &move |wakeup: Wakeup| {
                let fired = fired.clone();
                async move {
                    if let Ok(mut guard) = fired.lock() {
                        guard.push(FiredRecord {
                            job_id: wakeup.job.id.clone(),
                            coalesced: wakeup.coalesced,
                            gate_output: None,
                            late_by: wakeup.late_by,
                        });
                    }
                    FireOutcome::Ran
                }
            },
        )
        .await
        .expect("tick runs");

        assert!(harness.fired().is_empty());
        assert_eq!(
            harness.jobs().await.len(),
            1,
            "and the job is untouched, not consumed"
        );
    }

    /// A host that cannot take a job must not consume its occurrence. `meka serve` hits this
    /// whenever a REPL holds the session's file lock: `prepare` has already stamped and advanced
    /// the job by the time the lock is attempted, so without the restore the job would be
    /// silently skipped on every tick for as long as the REPL stayed open.
    #[tokio::test]
    async fn a_deferred_job_keeps_its_occurrence() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;
        let due_before = job.next_fire_at;

        run_due(
            &harness.manager,
            &harness.config,
            None,
            &NoResidents,
            &SchedulerScope::every_job(),
            &|_wakeup: Wakeup| std::future::ready(FireOutcome::Deferred),
        )
        .await
        .expect("tick runs");

        let jobs = harness.jobs().await;
        let job = jobs.first().expect("job survives a deferral");
        assert_eq!(
            job.next_fire_at, due_before,
            "the occurrence must be put back, not spent"
        );
        assert!(
            job.last_fired_at.is_none(),
            "and it must not be recorded as having fired"
        );

        // Still due, so the host that *can* run it sees it on its next sweep.
        assert_eq!(
            harness
                .manager
                .schedule_store()
                .list_due_scheduled_jobs(Utc::now())
                .await
                .expect("list due")
                .len(),
            1
        );
    }

    /// The one-shot case of a deferral, which the recurring test above does not reach. Claiming a
    /// one-shot *deletes* its row, so a restore that only updated columns matched nothing, reported
    /// success, and lost the reminder for good. That is the concrete "remind me in 20 minutes"
    /// failure when `meka serve` and a REPL race for the same session.
    #[tokio::test]
    async fn a_deferred_one_shot_is_not_lost() {
        let harness = SchedulerHarness::new().await;
        let due = Utc::now() - chrono::Duration::minutes(1);
        let created = harness
            .overdue_job(Schedule::At(due), None, chrono::Duration::minutes(1))
            .await;

        run_due(
            &harness.manager,
            &harness.config,
            None,
            &NoResidents,
            &SchedulerScope::every_job(),
            &|_wakeup: Wakeup| std::future::ready(FireOutcome::Deferred),
        )
        .await
        .expect("tick runs");

        let jobs = harness.jobs().await;
        assert_eq!(jobs.len(), 1, "the reminder must survive a deferral");
        let job = jobs.first().expect("job present");
        assert_eq!(job.id, created.id);
        assert_eq!(
            job.next_fire_at, created.next_fire_at,
            "still due, for the host that can run it"
        );
        assert!(job.last_fired_at.is_none());
    }

    /// A deferral must also leave the gate's baseline alone.
    ///
    /// The baseline is measured inside `prepare` and rides on the [`Claim`] until the occurrence is
    /// disposed of, which is what makes this work: a host that evaluates, decides to fire and then
    /// cannot run the turn writes nothing. Persisting it at the moment the gate returned would
    /// leave the watcher having already absorbed the change it exists to report, so the next host
    /// would compare the new value against itself, see nothing, and stay quiet forever.
    #[tokio::test]
    async fn a_deferred_gated_job_keeps_its_baseline() {
        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(
                    "echo changed",
                    GatePredicate::Changed,
                    Some("original"),
                )),
                chrono::Duration::minutes(1),
            )
            .await;

        run_due(
            &harness.manager,
            &harness.config,
            None,
            &NoResidents,
            &SchedulerScope::every_job(),
            &|_wakeup: Wakeup| std::future::ready(FireOutcome::Deferred),
        )
        .await
        .expect("tick runs");

        let jobs = harness.jobs().await;
        assert_eq!(
            jobs.first()
                .and_then(|job| job.gate.as_ref())
                .and_then(|gate| gate.last_output.as_deref()),
            Some("original"),
            "the baseline must be as it was, so the change still fires for the next host"
        );
    }

    /// The arbitration between hosts, at the primitive. Two `meka serve` instances sharing a
    /// database both read the same occurrence into their due lists; exactly one of them may take
    /// it. An unconditional write would let both advance the row and both go on to fire.
    ///
    /// One shape for both kinds of schedule, which is the point of leasing rather than consuming:
    /// claiming a one-shot by deleting its row needs a second shape, and the delete has nothing to
    /// hand back.
    #[tokio::test]
    async fn only_one_host_can_claim_an_occurrence() {
        for (label, schedule) in [
            ("recurring", Schedule::parse_every("1h").expect("parses")),
            (
                "one-shot",
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
            ),
        ] {
            let harness = SchedulerHarness::new().await;
            let job = harness
                .overdue_job(schedule, None, chrono::Duration::minutes(5))
                .await;
            let store = harness.manager.schedule_store();
            let now = Utc::now();
            let until = now + chrono::Duration::hours(1);

            assert!(
                store
                    .claim_occurrence(&job.id, job.next_fire_at, "host-a", now, until)
                    .await
                    .expect("claim"),
                "{label}: the first host to reach the row takes the occurrence"
            );
            assert!(
                !store
                    .claim_occurrence(&job.id, job.next_fire_at, "host-b", now, until)
                    .await
                    .expect("claim"),
                "{label}: and the second, still holding the copy it listed, is refused"
            );
            assert_eq!(
                harness.jobs().await.first().map(|job| job.next_fire_at),
                Some(job.next_fire_at),
                "{label}: and the row itself has not moved, because a claim no longer consumes it"
            );
        }
    }

    /// A lease that has run out is takeable, and that is what a crash costs: a delay, not the job.
    ///
    /// The host that dies mid-delivery never releases. Were a claim to consume the row (advance
    /// past the occurrence, or delete a one-shot outright), the occurrence would be gone with
    /// nothing to recover it from and the reminder simply never delivered.
    #[tokio::test]
    async fn an_expired_lease_is_taken_by_the_next_host() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                None,
                chrono::Duration::minutes(1),
            )
            .await;
        let store = harness.manager.schedule_store();
        let died_at = Utc::now() - chrono::Duration::hours(2);

        assert!(
            store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-that-died",
                    died_at,
                    died_at + chrono::Duration::hours(1),
                )
                .await
                .expect("claim"),
            "a host takes the occurrence and then never comes back"
        );
        let now = Utc::now();
        assert!(
            store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-b",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("claim"),
            "an hour later the lease has expired and the job is deliverable again"
        );
        assert_eq!(
            harness.jobs().await.len(),
            1,
            "and it was there to be taken, which a consumed row would not have been"
        );
    }

    /// A gate that cannot be evaluated spends the occurrence, exactly as one that ran and said no
    /// does.
    ///
    /// Releasing the lease instead, since nothing was measured and the job has not had its turn,
    /// would leave `next_fire_at` where it was, so the row would be due again on the very next
    /// sweep: a six-hour job whose server is down would be re-probed every `poll_interval` rather
    /// than every six hours, and a probe that hangs would burn the whole `gate_timeout` out of each
    /// sweep.
    #[tokio::test]
    async fn a_gate_that_cannot_be_evaluated_spends_a_recurring_occurrence() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("6h").expect("parses"),
                // A pointer into output that is not JSON: an error on every evaluation.
                Some(gate(
                    "echo not-json-at-all",
                    GatePredicate::At {
                        pointer: "/chats".to_string(),
                        is: PointerTest::NotEmpty,
                    },
                    Some("the value last actually observed"),
                )),
                chrono::Duration::minutes(1),
            )
            .await;

        for _ in 0..4 {
            harness.tick().await;
        }

        assert!(harness.fired().is_empty(), "a broken gate never fires");
        let after = harness.jobs().await;
        let after = after.first().expect("the job survives");
        assert!(
            after.next_fire_at > job.next_fire_at,
            "the occurrence is spent, so the next probe is a period away and not a tick away: \
             {:?} vs {:?}",
            after.next_fire_at,
            job.next_fire_at
        );
        assert_eq!(
            harness
                .manager
                .scheduler_memory()
                .probe_failure(&job.id)
                .map(|(count, _)| count),
            Some(1),
            "and four sweeps cost one probe, because only one occurrence came due"
        );
        assert_eq!(
            after
                .gate
                .as_ref()
                .and_then(|gate| gate.last_output.as_deref()),
            Some("the value last actually observed"),
            "nothing was measured, so the baseline must survive: the next working evaluation is \
             what reports the change that happened while the probe was broken"
        );
        assert_eq!(
            after.attempts, 0,
            "and the job is not on its way to being parked, because its occurrences are spent \
             rather than accumulating"
        );
    }

    /// The same for a one-shot, which has no next occurrence to spend: the lease is what waits.
    ///
    /// Advancing is not available, so the row stays due, and releasing the lease would make it due
    /// now: re-probed on every sweep, and parked by the attempt ceiling after three of them,
    /// which at the default `poll_interval` is half a minute. A server restarting anywhere near a
    /// one-shot's due time would silently destroy the reminder, which is a worse failure than the
    /// cost the advance exists to avoid. Keeping the lease spaces the retry by `claim_lease`.
    #[tokio::test]
    async fn a_gate_that_cannot_be_evaluated_holds_a_one_shots_lease_rather_than_reprobing() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                Some(gate(
                    "echo not-json-at-all",
                    GatePredicate::At {
                        pointer: "/chats".to_string(),
                        is: PointerTest::NotEmpty,
                    },
                    None,
                )),
                chrono::Duration::minutes(1),
            )
            .await;

        for _ in 0..6 {
            harness.tick().await;
        }

        assert_eq!(
            harness
                .manager
                .scheduler_memory()
                .probe_failure(&job.id)
                .map(|(count, _)| count),
            Some(1),
            "six sweeps inside one lease cost one probe, not six"
        );
        let after = harness.jobs().await;
        let after = after.first().expect("the job is kept, not deleted");
        assert_eq!(
            after.attempts, 1,
            "and it is nowhere near the ceiling, so a server that comes back inside the lease \
             still delivers the reminder"
        );
    }

    /// It does still park, once those spaced-out retries are spent.
    ///
    /// The ceiling has to survive the backoff above, or a one-shot whose gate is permanently broken
    /// is probed once per lease until its grace period closes. Each expiry is a fresh claim, so the
    /// count rises on the retries rather than on the sweeps.
    #[tokio::test]
    async fn a_one_shot_whose_gate_never_works_is_parked_once_its_retries_are_spent() {
        let mut harness = SchedulerHarness::new().await;
        // A lease that has run out by the next sweep, so one tick stands in for one `claim_lease`
        // without the test waiting one out. Set on the resolved config directly because
        // `validate` refuses anything this short: what is being exercised is the expiry, not a
        // setting anyone can configure.
        harness.config.claim_lease = Duration::from_millis(1);
        harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                Some(gate(
                    "echo not-json-at-all",
                    GatePredicate::At {
                        pointer: "/chats".to_string(),
                        is: PointerTest::NotEmpty,
                    },
                    None,
                )),
                chrono::Duration::minutes(1),
            )
            .await;

        for _ in 0..6 {
            tokio::time::sleep(Duration::from_millis(3)).await;
            harness.tick().await;
        }

        let after = harness.jobs().await;
        let after = after.first().expect("a parked job is kept, not deleted");
        assert_eq!(
            after.attempts, MAX_CLAIM_ATTEMPTS,
            "the ceiling still bites: past it the job is refused before its gate runs"
        );
        let reported = job_withheld_reason(
            harness.manager.scheduler_memory(),
            after,
            crate::permission::Permission::Unrestricted,
            None,
        )
        .unwrap_or_default();
        assert!(
            reported.contains("gate could not be evaluated"),
            "and it is reported for what happened, not as a crash: {reported}"
        );
        assert!(
            reported.contains("JSON"),
            "naming the probe's own error, which is the actionable half: {reported}"
        );
    }

    /// An expired lease is "unclaimed" to the paths that hand out work, so it must be unclaimed to
    /// the paths that retire and advance without taking one.
    ///
    /// Nothing clears `claimed_by` but a release, a completion or a fresh claim, so a host that
    /// dies holding a lease leaves it set for good. If those two paths tested the column rather
    /// than the expiry, such a row would be handed to `prepare` on every sweep and be invisible to
    /// both: a one-shot past its grace period would never be retired, never fired and never logged,
    /// and a refused recurring job would never advance.
    #[tokio::test]
    async fn a_lease_left_by_a_dead_host_does_not_wedge_the_occurrence() {
        for (label, schedule, level, overdue) in [
            (
                "one-shot past its grace period is retired",
                Schedule::At(Utc::now() - chrono::Duration::hours(25)),
                crate::permission::Permission::Unrestricted,
                chrono::Duration::hours(25),
            ),
            (
                "recurring job refused at `none` still spends its occurrence",
                Schedule::parse_every("1h").expect("parses"),
                crate::permission::Permission::None,
                chrono::Duration::hours(6),
            ),
        ] {
            let recurring = schedule.is_recurring();
            let harness = SchedulerHarness::at_session_permission(level).await;
            let job = harness.overdue_job(schedule, None, overdue).await;
            let store = harness.manager.schedule_store();
            let died_at = Utc::now() - overdue;
            assert!(
                store
                    .claim_occurrence(
                        &job.id,
                        job.next_fire_at,
                        "host-that-died",
                        died_at,
                        died_at + chrono::Duration::hours(1),
                    )
                    .await
                    .expect("claim"),
                "{label}: a host takes the occurrence and is killed before it can release"
            );

            harness.tick().await;

            assert!(harness.fired().is_empty(), "{label}: nothing is delivered");
            let after = harness.jobs().await;
            match recurring {
                false => assert!(
                    after.is_empty(),
                    "{label}: it must be retired, as it is when no crashed host ever touched it"
                ),
                true => assert!(
                    after
                        .first()
                        .is_some_and(|after| after.next_fire_at > job.next_fire_at),
                    "{label}: {:?} vs {:?}",
                    after.first().map(|after| after.next_fire_at),
                    job.next_fire_at
                ),
            }
        }
    }

    /// A sweep that bounded its own coverage says so.
    ///
    /// The budget holds jobs over, and the next sweep takes them, so nothing is lost, which is
    /// exactly why the line matters: without it a capped run is indistinguishable from a complete
    /// one in the log, and an operator watching a backlog has no way to tell that the cap is what
    /// they are looking at. The count has no reader but this line.
    #[tokio::test]
    async fn a_sweep_that_holds_jobs_over_reports_that_it_did() {
        let mut harness = SchedulerHarness::new().await;
        harness.config.max_consecutive_fires = 2;
        for _ in 0..5 {
            harness
                .overdue_job(
                    Schedule::parse_every("1h").expect("parses"),
                    None,
                    chrono::Duration::hours(6),
                )
                .await;
        }

        crate::render::log_capture::start();
        harness.tick().await;
        let reported = crate::render::log_capture::infos();
        assert!(
            reported.contains("held over 3 due job(s)"),
            "five due jobs against a budget of two leaves three, and the count has to be right or \
             the line is worse than nothing: {reported}"
        );

        // And it stays quiet when the budget did not engage, or it would train the reader to skip
        // it.
        let harness = SchedulerHarness::new().await;
        harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::hours(6),
            )
            .await;
        crate::render::log_capture::start();
        harness.tick().await;
        assert!(
            !crate::render::log_capture::infos().contains("held over"),
            "one job and a budget of five is not a bounded sweep"
        );
    }

    /// Several hosts noticing the same expired one-shot produce one announcement, not one each.
    ///
    /// The delete is scoped to the occurrence, so whoever wins removes the row and everyone else
    /// changes nothing, and the return value is how the winner knows to be the one that speaks.
    /// Every assertion about the row itself passes whichever way that value goes.
    #[tokio::test]
    async fn only_the_host_that_removed_an_expired_one_shot_announces_it() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::hours(30)),
                None,
                chrono::Duration::hours(30),
            )
            .await;
        let store = harness.manager.schedule_store();
        let now = Utc::now();

        assert!(
            store
                .retire_unclaimed(&job.id, job.next_fire_at, now)
                .await
                .expect("retire"),
            "the host whose delete removed the row is the one that announces it"
        );
        assert!(
            !store
                .retire_unclaimed(&job.id, job.next_fire_at, now)
                .await
                .expect("retire"),
            "and a host arriving afterwards stays quiet rather than repeating it"
        );
    }

    /// A parked job is not accused of crashing meka when nothing knows that it did.
    ///
    /// `attempts` is on the row; the reason it rose is in a process-global map. A restart is
    /// exactly what an operator does once a job has gone inert, and `meka schedule list` is a
    /// separate process that never had the map at all, so the commonest way to read this message
    /// is with the cause missing. Asserting the likelier cause from that absence would tell someone
    /// whose MCP server was misconfigured that their prompt takes meka down, with a remedy aimed at
    /// the wrong thing, in the model's own `[Scheduled]` block.
    ///
    /// The row still settles it one way: no gate means no probe that could have failed.
    #[tokio::test]
    async fn a_parked_job_is_only_called_a_crash_when_the_row_can_prove_it() {
        let harness = SchedulerHarness::new().await;
        let mut gated = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate("true", GatePredicate::Succeeded, None)),
                chrono::Duration::minutes(1),
            )
            .await;
        gated.attempts = MAX_CLAIM_ATTEMPTS;
        // A fresh memory, so this process holds no record of why the claims failed, which is
        // the state every reader is in after a restart.
        let memory = SchedulerMemory::default();

        let reported = job_withheld_reason(
            &memory,
            &gated,
            crate::permission::Permission::Unrestricted,
            None,
        )
        .unwrap_or_default();
        assert!(
            !reported.contains("the host died"),
            "a gated job could have been parked by either cause, so neither may be asserted: \
             {reported}"
        );
        assert!(
            reported.contains("gate cannot be evaluated") && reported.contains("takes the host"),
            "both possibilities are named, so the operator knows what to check: {reported}"
        );

        let mut ungated = gated;
        ungated.gate = None;
        let reported = job_withheld_reason(
            &memory,
            &ungated,
            crate::permission::Permission::Unrestricted,
            None,
        )
        .unwrap_or_default();
        assert!(
            reported.contains("the host died"),
            "with no gate there is no probe that could have failed, so the crash can be named: \
             {reported}"
        );
    }

    /// A standing "this gate is broken" verdict retires itself once the row shows otherwise.
    ///
    /// The counter is per process and only the host that wins `claim_occurrence` ever touches it.
    /// Which host wins is a race between their tickers, so a second `meka serve` on the same store
    /// can take over every occurrence and heal the gate while this process's count stays where it
    /// stopped. The marker would then stand forever: the model told, every turn, that a job firing
    /// hourly was dead.
    ///
    /// Driven in one process rather than two. Advancing `last_fired_at` here is exactly what the
    /// other host's `complete_claim` writes, and the counter it has to convince lives in this
    /// process either way, so a second one would add wall-clock and no signal. It would add
    /// *coverage*, though: `GET /v1/schedule` does put this verdict on a wire
    /// (`host::http::handlers::jobs` renders it as `withheld`), so a two-host test is constructible
    /// and would exercise the convergence end to end rather than at the predicate.
    #[tokio::test]
    async fn a_probe_verdict_stands_down_when_another_host_has_evaluated_the_gate() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate("true", GatePredicate::Succeeded, None)),
                chrono::Duration::minutes(1),
            )
            .await;
        let memory = harness.manager.scheduler_memory();

        for _ in 0..PROBE_FAILURES_BEFORE_REPORTING {
            memory.record_probe_failure(&job, "server said no such tool");
        }
        assert!(
            memory.standing_probe_failure(&job).is_some(),
            "two failures with nothing to contradict them is a standing condition"
        );

        // What the other host's `complete_claim` writes when it fires the job. The failing path
        // writes neither this nor the baseline, so it cannot be this process's own doing.
        let mut fired = job.clone();
        fired.last_fired_at = Some(Utc::now());
        assert!(
            memory.standing_probe_failure(&fired).is_none(),
            "the job has fired since the last failure was counted, so somebody evaluated this \
             gate and got an answer"
        );
        assert!(
            memory.standing_probe_failure(&job).is_none(),
            "and the verdict is dropped rather than merely suppressed, so it does not come back \
             the next time this process reads the pre-fire row"
        );
    }

    /// A changed baseline is the other half: a gate that evaluates and declines never advances
    /// `last_fired_at`, but a successful evaluation still records what it saw.
    #[tokio::test]
    async fn a_probe_verdict_also_stands_down_on_a_baseline_another_host_recorded() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate("true", GatePredicate::Changed, Some("old"))),
                chrono::Duration::minutes(1),
            )
            .await;
        let memory = harness.manager.scheduler_memory();

        for _ in 0..PROBE_FAILURES_BEFORE_REPORTING {
            memory.record_probe_failure(&job, "server said no such tool");
        }
        assert!(memory.standing_probe_failure(&job).is_some());

        let mut evaluated = job;
        if let Some(gate) = evaluated.gate.as_mut() {
            gate.last_output = Some("new".to_string());
        }
        assert!(
            memory.standing_probe_failure(&evaluated).is_none(),
            "a baseline this process did not write means the gate answered somewhere else"
        );
    }

    /// Closing an occurrence that is not there any more is not the same as losing the lease.
    ///
    /// Both make the scoped write match nothing, and they mean opposite things. A job that fires
    /// and then cancels itself is an ordinary shape (`schedule_create`'s own reply tells the model
    /// how), and conflating the two would tell it, on every such fire, that a duplicate delivery
    /// was possible and that an unrelated setting should be raised.
    #[tokio::test]
    async fn closing_an_occurrence_tells_a_canceled_job_from_a_lost_lease() {
        let harness = SchedulerHarness::new().await;
        let store = harness.manager.schedule_store();
        let now = Utc::now();

        let canceled = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(1),
            )
            .await;
        assert!(
            store
                .claim_occurrence(
                    &canceled.id,
                    canceled.next_fire_at,
                    "host-a",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("claim")
        );
        store
            .delete_scheduled_job(&canceled.id)
            .await
            .expect("the model cancels the job during its own turn");
        assert_eq!(
            store
                .complete_claim(&canceled.id, "host-a", Some(now), Some(now), None)
                .await
                .expect("complete"),
            ClaimClosed::RowGone,
            "there is no occurrence left to close, and nothing has gone wrong"
        );

        let taken = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(1),
            )
            .await;
        assert!(
            store
                .claim_occurrence(
                    &taken.id,
                    taken.next_fire_at,
                    "host-b",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("claim")
        );
        assert_eq!(
            store
                .complete_claim(&taken.id, "host-a", Some(now), Some(now), None)
                .await
                .expect("complete"),
            ClaimClosed::LeaseLost,
            "the row is there under someone else's claim, so this turn may be delivered again"
        );
    }

    /// A cancellation issued while a host holds the lease is not undone by the handback.
    ///
    /// This is the failure the lease exists for. Claiming a one-shot by deleting its row makes the
    /// handback an `INSERT` that cannot tell "I deleted this a moment ago" from "the user canceled
    /// it in between", and puts the job back either way, silently discarding the cancellation. A
    /// release is scoped to the claim, so it cannot recreate anything.
    #[tokio::test]
    async fn a_cancellation_during_a_claim_is_not_undone_by_the_handback() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                None,
                chrono::Duration::minutes(1),
            )
            .await;
        let store = harness.manager.schedule_store();
        let now = Utc::now();

        assert!(
            store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-a",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("claim")
        );
        assert!(
            store
                .delete_scheduled_job(&job.id)
                .await
                .expect("the operator cancels it while the host works"),
            "the cancel removes a row that is still there, and says so"
        );

        store
            .release_claim(&job.id, "host-a")
            .await
            .expect("the host hands the occurrence back");

        assert!(
            harness.jobs().await.is_empty(),
            "a canceled job stays canceled: the handback may not resurrect it"
        );
    }

    /// What a lost claim must cost: nothing. `prepare` evaluates the gate only after the claim is
    /// won, so a host that arrives second neither spawns the command nor produces a wakeup, and
    /// leaves the winner's schedule exactly as the winner wrote it.
    ///
    /// Observed through a side effect on the filesystem for the same reason
    /// [`a_held_over_job_does_not_run_its_gate`] is: a gate that ran and was then discarded leaves
    /// every column identical to one that never ran.
    #[tokio::test]
    async fn a_lost_claim_runs_no_gate_and_produces_no_wakeup() {
        let harness = SchedulerHarness::new().await;
        struct Probe(std::path::PathBuf);
        impl Drop for Probe {
            fn drop(&mut self) {
                std::fs::remove_file(&self.0).ok();
            }
        }
        let guard =
            Probe(std::env::temp_dir().join(format!("meka-claim-probe-{}", uuid::Uuid::new_v4())));
        let probe = guard.0.clone();
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate(
                    &create_file_command(&probe),
                    GatePredicate::Succeeded,
                    None,
                )),
                chrono::Duration::minutes(5),
            )
            .await;
        let job_next_fire_at = job.next_fire_at;

        // The other host gets there first, while this one is still holding the copy it listed.
        let now = Utc::now();
        assert!(
            harness
                .manager
                .schedule_store()
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "the-other-host",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("the other host claims")
        );

        let wakeup = prepare(
            &harness.manager,
            &harness.config,
            None,
            &NoResidents,
            job,
            Utc::now(),
        )
        .await
        .expect("prepare runs");

        assert!(wakeup.is_none(), "a host that lost the claim does not fire");
        assert!(!probe.exists(), "and never ran the gate command");
        assert_eq!(
            harness
                .jobs()
                .await
                .first()
                .map(|job| job.next_fire_at)
                .expect("job survives"),
            job_next_fire_at,
            "and the winner's row is where the winner left it"
        );
    }

    /// The one-shot half of a lost claim: the host that did not win must not deliver "remind me in
    /// 20 minutes" a second time.
    #[tokio::test]
    async fn a_lost_one_shot_claim_produces_no_wakeup() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::At(Utc::now() - chrono::Duration::minutes(1)),
                None,
                chrono::Duration::minutes(1),
            )
            .await;
        let now = Utc::now();
        assert!(
            harness
                .manager
                .schedule_store()
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "the-other-host",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("the other host claims")
        );

        let wakeup = prepare(
            &harness.manager,
            &harness.config,
            None,
            &NoResidents,
            job,
            Utc::now(),
        )
        .await
        .expect("prepare runs");

        assert!(
            wakeup.is_none(),
            "the reminder belongs to the host holding the lease"
        );
    }

    /// A handback releases the occurrence *this* host holds, and nothing else.
    ///
    /// A whole-row upsert applied by id would let a host that lost the claim and was then refused
    /// the session lock overwrite the winner's `next_fire_at` with a time already in the past,
    /// bringing the job due on the very next tick while the winner is still running the turn. One
    /// hourly occurrence would produce three gate runs and two agent turns that way.
    ///
    /// Scoping to the lease makes that structural rather than careful: the shape below is a host
    /// whose lease expired and was taken over while it was still working, which is the only way two
    /// hosts can now hold opinions about one job at once.
    #[tokio::test]
    async fn a_handback_does_not_reach_past_its_own_lease() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;
        let store = harness.manager.schedule_store();
        let long_ago = Utc::now() - chrono::Duration::hours(2);

        assert!(
            store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-a",
                    long_ago,
                    long_ago + chrono::Duration::minutes(1),
                )
                .await
                .expect("claim"),
            "host A takes the occurrence, and then takes far too long"
        );
        let now = Utc::now();
        assert!(
            store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-b",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("claim"),
            "host B finds the lease expired and takes it over"
        );

        // Host A finally gives up and hands back what it thinks it holds.
        store
            .release_claim(&job.id, "host-a")
            .await
            .expect("release");

        assert!(
            !store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-c",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("claim"),
            "host B still holds it: a stale release must not free an occurrence someone else owns"
        );
    }

    /// The writes that come after a claim are scoped to the lease, and this is what that buys. A
    /// host whose lease expires while its gate is still running has been taken over; an unscoped
    /// write would then stamp this host's fire onto the new holder's row and drag the `changed`
    /// baseline back to a value already reported on.
    #[tokio::test]
    async fn a_late_write_does_not_land_on_another_hosts_occurrence() {
        let harness = SchedulerHarness::new().await;
        let planted = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                Some(gate("echo state", GatePredicate::Changed, None)),
                chrono::Duration::minutes(5),
            )
            .await;
        let store = harness.manager.schedule_store();

        let long_ago = Utc::now() - chrono::Duration::hours(2);
        assert!(
            store
                .claim_occurrence(
                    &planted.id,
                    planted.next_fire_at,
                    "host-a",
                    long_ago,
                    long_ago + chrono::Duration::minutes(1),
                )
                .await
                .expect("claim")
        );
        // Taken over by another host while this one's gate is still running.
        let now = Utc::now();
        assert!(
            store
                .claim_occurrence(
                    &planted.id,
                    planted.next_fire_at,
                    "host-b",
                    now,
                    now + chrono::Duration::hours(1)
                )
                .await
                .expect("the other host takes the expired lease")
        );
        store
            .complete_claim(
                &planted.id,
                "host-b",
                Some(planted.next_fire_at + chrono::Duration::hours(1)),
                None,
                Some("theirs"),
            )
            .await
            .expect("the other host finishes and records its baseline");

        // Host A finally finishes, and writes against the lease it thinks it holds.
        store
            .complete_claim(
                &planted.id,
                "host-a",
                Some(planted.next_fire_at + chrono::Duration::hours(2)),
                Some(Utc::now()),
                Some("ours"),
            )
            .await
            .expect("complete");

        let jobs = harness.jobs().await;
        let job = jobs.first().expect("job survives");
        assert_eq!(
            job.gate
                .as_ref()
                .and_then(|gate| gate.last_output.as_deref()),
            Some("theirs"),
            "a late baseline must not overwrite the one the lease holder recorded"
        );
        assert!(
            job.last_fired_at.is_none(),
            "and a late completion must not land on a row another host owns"
        );
    }

    /// The fallback arm of [`SAME_OCCURRENCE`]. Every writer in meka renders the column with
    /// `to_rfc3339`, so the textual comparison matches in practice, but a row that reached the
    /// database any other way must still be claimable. The failure this guards against is the
    /// quietest one available: a compare-and-swap that matches nothing on every sweep, forever,
    /// with the job simply never firing again and not a line said about it.
    #[tokio::test]
    async fn a_timestamp_stored_in_another_shape_is_still_claimable() {
        let harness = SchedulerHarness::new().await;
        let planted = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::minutes(5),
            )
            .await;
        let store = harness.manager.schedule_store();
        // The same instant, written the way something that is not meka would write it.
        let raw = planted
            .next_fire_at
            .format("%Y-%m-%dT%H:%M:%S%.3fZ")
            .to_string();
        store
            .set_next_fire_at_verbatim_for_test(&planted.id, &raw)
            .await
            .expect("plant the timestamp");

        let due = store
            .list_due_scheduled_jobs(Utc::now())
            .await
            .expect("list due");
        let job = due.first().expect("still due");
        assert!(
            store
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "host-a",
                    Utc::now(),
                    Utc::now() + chrono::Duration::hours(1)
                )
                .await
                .expect("claim"),
            "a job whose timestamp is not in meka's own shape must still be claimable"
        );
    }

    /// A job that really fires records that it fired, and a recurring one past the grace period is
    /// rescheduled rather than deleted.
    ///
    /// The store method has its own test, so this checks that `prepare` calls it: otherwise every
    /// job would read as never-fired in `meka schedule list` and an interval schedule would
    /// re-anchor on `created_at` after a restart and replay everything since.
    ///
    /// And the `!recurring` term in `if !recurring && past_grace` needs a fixture that exercises
    /// it: `DEFAULT_MISSED_GRACE` is 24 hours and every other fixture in the suite is at most 6
    /// hours overdue, so without one a laptop shut for a weekend could have every recurring job
    /// silently retired.
    #[tokio::test]
    async fn a_fire_is_recorded_and_a_long_outage_does_not_retire_a_recurring_job() {
        let harness = SchedulerHarness::new().await;
        // Well past `DEFAULT_MISSED_GRACE`, which is what makes the `!recurring` term the only
        // thing standing between this job and deletion.
        let planted = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::days(3),
            )
            .await;

        // Through a whole sweep rather than `prepare` alone, because the schedule is advanced and
        // the fire recorded once the turn has actually been delivered. That ordering is the point
        // of the lease: a crash before delivery costs a retry rather than the occurrence.
        harness.tick().await;

        assert_eq!(
            harness.fired().len(),
            1,
            "a recurring job is never past a grace period: its occurrences are one interval apart"
        );
        let job = harness
            .jobs()
            .await
            .first()
            .cloned()
            .expect("and the row survives rather than being retired");
        assert!(
            job.last_fired_at.is_some(),
            "a job that fires has to record it, or a restart re-anchors on `created_at` and \
             replays every occurrence since"
        );
        assert!(
            job.next_fire_at > Utc::now(),
            "and be scheduled forward rather than left due"
        );
        assert_eq!(
            (job.attempts, planted.attempts),
            (0, 0),
            "and a delivered occurrence clears the crash count rather than accumulating one"
        );
    }

    #[tokio::test]
    async fn rendered_prompt_marks_the_turn_as_scheduled() {
        let harness = SchedulerHarness::new().await;
        let job = harness
            .overdue_job(
                Schedule::parse_every("1h").expect("parses"),
                None,
                chrono::Duration::seconds(1),
            )
            .await;
        let wakeup = Wakeup {
            job,
            gate_output: Some("ci-red".to_string()),
            late_by: chrono::Duration::seconds(1),
            coalesced: 0,
        };
        let rendered = wakeup.render_prompt();
        assert!(rendered.starts_with("[Scheduled job "));
        assert!(rendered.contains("do the thing"));
        assert!(rendered.contains("[Gate output]\nci-red"));
        assert!(
            !rendered.contains("Late by"),
            "a second of tick latency is not worth reporting"
        );
    }

    #[test]
    fn is_recurring_separates_one_shots() {
        assert!(!Schedule::At(at("2026-08-12T09:00:00Z")).is_recurring());
        assert!(Schedule::parse_every("1h").expect("parses").is_recurring());
        assert!(
            Schedule::parse_cron("0 9 * * *")
                .expect("parses")
                .is_recurring()
        );
    }
}
