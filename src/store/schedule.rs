//! Scheduled jobs and their claims: the `scheduled_jobs` table.

use super::*;
use crate::schedule::*;

/// Scheduling's slice of the session database, handed out by
/// [`crate::store::Store::schedule_store`].
#[derive(Clone)]
pub(crate) struct ScheduleStore {
    pub(crate) connection: std::sync::Arc<tokio_rusqlite::Connection>,
    memory: std::sync::Arc<SchedulerMemory>,
}
impl ScheduleStore {
    pub(crate) fn new(
        connection: std::sync::Arc<tokio_rusqlite::Connection>,
        memory: std::sync::Arc<SchedulerMemory>,
    ) -> Self {
        Self { connection, memory }
    }

    /// Persist a new scheduled job. The caller owns computing `next_fire_at` from the job's anchor
    /// (see `ScheduledJob::anchor`).
    pub(crate) async fn create_scheduled_job(
        &self,
        job: &ScheduledJob,
    ) -> crate::error::Result<()> {
        let id = job.id.clone();
        let session_id = job.session_id.to_string();
        let kind = job.schedule.kind_str().to_string();
        let spec = job.schedule.spec();
        let prompt = job.prompt.clone();
        let gate_kind = job
            .gate
            .as_ref()
            .map(|gate| gate.probe.kind_str().to_string());
        let gate_spec_json = job.gate.as_ref().map(|gate| gate.spec());
        let gate_last_output = job.gate.as_ref().and_then(|gate| gate.last_output.clone());
        let gate_permission = job.gate.as_ref().map(|gate| gate.permission.to_string());
        let created_at = job.created_at.to_rfc3339();
        let last_fired_at = job.last_fired_at.map(|at| at.to_rfc3339());
        let next_fire_at = job.next_fire_at.to_rfc3339();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO scheduled_jobs (id, session_id, kind, spec, prompt, gate_kind, \
                     gate_spec_json, gate_last_output, gate_permission, created_at, \
                     last_fired_at, next_fire_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                    rusqlite::params![
                        id,
                        session_id,
                        kind,
                        spec,
                        prompt,
                        gate_kind,
                        gate_spec_json,
                        gate_last_output,
                        gate_permission,
                        created_at,
                        last_fired_at,
                        next_fire_at
                    ],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to create scheduled job: {error}"))
            })
    }

    /// Every job belonging to one session, soonest first.
    pub(crate) async fn list_scheduled_jobs(
        &self,
        session_id: uuid::Uuid,
    ) -> crate::error::Result<Vec<ScheduledJob>> {
        self.query_scheduled_jobs(
            "SELECT id, session_id, kind, spec, prompt, gate_kind, gate_spec_json, \
             gate_last_output, gate_permission, created_at, last_fired_at, next_fire_at, \
             attempts FROM scheduled_jobs WHERE session_id = ?1 ORDER BY next_fire_at ASC"
                .to_string(),
            vec![session_id.to_string()],
        )
        .await
    }

    /// Every job in the database, soonest first. Backs `meka schedule list` and `meka schedule
    /// cancel`, which work from a job id and so cannot ask the caller which session to look in.
    pub(crate) async fn list_all_scheduled_jobs(&self) -> crate::error::Result<Vec<ScheduledJob>> {
        self.query_scheduled_jobs(
            "SELECT id, session_id, kind, spec, prompt, gate_kind, gate_spec_json, \
             gate_last_output, gate_permission, created_at, last_fired_at, next_fire_at, \
             attempts FROM scheduled_jobs ORDER BY next_fire_at ASC"
                .to_string(),
            Vec::new(),
        )
        .await
    }

    /// Every job across all sessions whose `next_fire_at` has passed, soonest first. The
    /// scheduler's per-tick query; served by `idx_scheduled_jobs_next_fire_at`.
    pub(crate) async fn list_due_scheduled_jobs(
        &self,
        now: chrono::DateTime<chrono::Utc>,
    ) -> crate::error::Result<Vec<ScheduledJob>> {
        self.query_scheduled_jobs(
            format!(
                "SELECT id, session_id, kind, spec, prompt, gate_kind, gate_spec_json, \
                 gate_last_output, gate_permission, created_at, last_fired_at, \
                 next_fire_at, attempts FROM scheduled_jobs \
                 WHERE next_fire_at <= ?1 AND {} \
                 ORDER BY next_fire_at ASC",
                no_live_claim("?1")
            ),
            vec![now.to_rfc3339()],
        )
        .await
    }

    /// Shared row decoder. A row that fails to decode (hand-edited spec, a `kind` from a future
    /// version) is skipped with a warning rather than failing the whole query: one bad row must not
    /// stop every other job in the database from firing.
    pub(crate) async fn query_scheduled_jobs(
        &self,
        sql: String,
        params: Vec<String>,
    ) -> crate::error::Result<Vec<ScheduledJob>> {
        let rows: Vec<ScheduledJobRow> = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(&sql)?;
                let rows = statement
                    .query_map(rusqlite::params_from_iter(params.iter()), |row| {
                        Ok(ScheduledJobRow {
                            id: row.get(0)?,
                            session_id: row.get(1)?,
                            kind: row.get(2)?,
                            spec: row.get(3)?,
                            prompt: row.get(4)?,
                            gate_kind: row.get(5)?,
                            gate_spec_json: row.get(6)?,
                            gate_last_output: row.get(7)?,
                            gate_permission: row.get(8)?,
                            created_at: row.get(9)?,
                            last_fired_at: row.get(10)?,
                            next_fire_at: row.get(11)?,
                            attempts: row.get::<_, i64>(12)?.max(0) as u32,
                        })
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to load scheduled jobs: {error}"))
            })?;

        Ok(rows
            .into_iter()
            .filter_map(|row| {
                let id = row.id.clone();
                row.decode()
                    .inspect_err(|error| {
                        tracing::warn!("skipping unreadable scheduled job {id}: {error}");
                    })
                    .ok()
            })
            .collect())
    }

    /// Delete a job by full or unique-prefix id. Returns the id actually removed, or `None` when
    /// nothing matched *or* when the row was gone by the time the delete ran.
    ///
    /// An ambiguous prefix is an error rather than an arbitrary pick, and a `Config` rather than a
    /// `Database` one. Nothing went wrong with the database; the caller's prefix is
    /// under-specified, and `Config` is the variant that carries that to HTTP as a 422 rather than
    /// a 500. `BackgroundStore::resolve_background_task` says the same thing about the same
    /// condition and already used `Config`, so the two `serve` endpoints answered different
    /// statuses for one mistake. The variant name fits neither of them well; its mapping does, and
    /// a new variant for one condition is not worth the churn through every match.
    pub(crate) async fn cancel_scheduled_job(
        &self,
        session_id: uuid::Uuid,
        id_prefix: &str,
    ) -> crate::error::Result<Option<String>> {
        if !crate::text::is_usable_id_prefix(id_prefix) {
            return Ok(None);
        }
        let wanted = crate::text::id_prefix_for_matching(id_prefix);
        let jobs = self.list_scheduled_jobs(session_id).await?;
        let matches: Vec<&ScheduledJob> = jobs
            .iter()
            .filter(|job| job.id.starts_with(&wanted))
            .collect();
        let id = match matches.as_slice() {
            [] => return Ok(None),
            [job] => job.id.clone(),
            several => {
                return Err(MekaError::Config(format!(
                    "'{}' matches {} jobs; use a longer id",
                    id_prefix,
                    several.len()
                )));
            }
        };

        // `None` when the row was already gone, not `Some(id)`.
        //
        // The listing above and the `DELETE` below are two statements, and a scheduler sweep can
        // retire the row between them: a one-shot's occurrence retires it, and a session deleted
        // elsewhere takes its jobs with it through the foreign key. Reporting the id regardless
        // told the agent "Canceled job abc12345" about a job this call did not cancel, which is
        // the same sentence it gets when it did -- and there is no way to tell them apart
        // afterwards, because both end with no such row.
        match self.delete_scheduled_job(&id).await? {
            true => Ok(Some(id)),
            false => Ok(None),
        }
    }

    /// Delete a job by exact id, without the prefix resolution [`Self::cancel_scheduled_job`] does.
    ///
    /// `true` when a row was actually removed. Callers that report an outcome to a person or to the
    /// model must not treat `false` as success: it means something else removed the job first, and
    /// saying "canceled" then is a claim about work this call did not do.
    pub(crate) async fn delete_scheduled_job(&self, id: &str) -> crate::error::Result<bool> {
        // A job that no longer exists cannot be held back for permission, and the memory is
        // otherwise only cleared when a job is *authorized* again. A job canceled while declined
        // therefore left its id there for the life of a `meka serve`. Forgetting here keeps the
        // memory bounded by the jobs that exist rather than by every job that ever did. This is
        // not the only `DELETE` -- `retire_unclaimed` and `complete_claim` have their own -- but
        // both of those forget the job themselves, on the path that reaches them.
        self.memory.forget(id);
        let id = id.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM scheduled_jobs WHERE id = ?1",
                    rusqlite::params![id],
                )
            })
            .await
            .map(|changed| changed == 1)
            .map_err(|error| {
                MekaError::Database(format!("failed to delete scheduled job: {error}"))
            })
    }

    /// Take this occurrence for this process, by leasing the row rather than consuming it.
    ///
    /// One shape for both kinds of schedule. Arbitrating by consuming the row instead would need
    /// two: a recurring job advances `next_fire_at`, but a one-shot has no next occurrence and can
    /// only be *deleted*, which overloads one piece of state with two facts ("the user still wants
    /// this job" and "this occurrence is available"). A host handing such an occurrence back could
    /// only re-`INSERT`, and an `INSERT` cannot tell "I deleted this a moment ago" from "the user
    /// canceled it in between".
    ///
    /// A lease separates the facts. `claimed_by` says who is delivering this occurrence and
    /// `claim_expires_at` says how long that claim is good for; the row itself stays put, so a
    /// cancellation is an unconditional `DELETE` that always wins and a crash expires rather than
    /// erasing. Every write after this one is scoped to `claimed_by`, so a late writer whose lease
    /// has since been taken changes nothing.
    ///
    /// `occurrence` is the value this host read into its due list, so exactly one host can take a
    /// given occurrence. `attempts` counts claims that neither completed nor were handed back, a
    /// host that died or a probe that could not answer and left its lease to expire; see
    /// [`ScheduledJob::attempts`] and [`MAX_CLAIM_ATTEMPTS`].
    pub(crate) async fn claim_occurrence(
        &self,
        id: &str,
        occurrence: chrono::DateTime<chrono::Utc>,
        owner: &str,
        now: chrono::DateTime<chrono::Utc>,
        until: chrono::DateTime<chrono::Utc>,
    ) -> crate::error::Result<bool> {
        let id = id.to_string();
        let occurrence = occurrence.to_rfc3339();
        let owner = owner.to_string();
        let now = now.to_rfc3339();
        let until = until.to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    &format!(
                        "UPDATE scheduled_jobs \
                         SET claimed_by = ?3, claim_expires_at = ?5, attempts = attempts + 1 \
                         WHERE id = ?1 AND {} AND {}",
                        SAME_OCCURRENCE,
                        no_live_claim("?4")
                    ),
                    rusqlite::params![id, occurrence, owner, now, until],
                )
            })
            .await
            .map(|changed| changed == 1)
            .map_err(|error| MekaError::Database(format!("failed to claim scheduled job: {error}")))
    }

    /// Hand the occurrence back: this host took it and will not deliver it after all.
    ///
    /// Scoped to `owner`, which is what makes it safe: releasing a lease can only affect a row this
    /// host still holds, so a job canceled in the meantime stays canceled and a job another host
    /// has since claimed is left alone.
    ///
    /// `attempts` is reset, because a claim the *host* handed back is not the failure
    /// [`MAX_CLAIM_ATTEMPTS`] is counting. That ceiling is about jobs that cannot be delivered;
    /// this is about a host that could not take one, which says nothing about the job.
    ///
    /// This is the only way a lease is given up early, and that is the whole disposal rule: a claim
    /// is either handed back by a host that declined the work, or left to expire. A claim that
    /// ended *without* delivering (a panicking turn, an unevaluable probe) must be left to expire
    /// rather than released here. Releasing it leaves the row due on the next sweep, so the retries
    /// come one `poll_interval` apart and [`MAX_CLAIM_ATTEMPTS`] is reached in half a minute,
    /// killing jobs whose only problem was a blip.
    ///
    /// Matching nothing is ordinary and is not reported: the lease had already expired and someone
    /// else holds the occurrence, which for a host that was declining it anyway is the outcome it
    /// wanted. Hence `()` rather than a `bool` no caller reads, as with
    /// [`Self::advance_unclaimed`].
    pub(crate) async fn release_claim(&self, id: &str, owner: &str) -> crate::error::Result<()> {
        let id = id.to_string();
        let owner = owner.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE scheduled_jobs SET claimed_by = NULL, claim_expires_at = NULL, \
                     attempts = 0 WHERE id = ?1 AND claimed_by = ?2",
                    rusqlite::params![id, owner],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to release a scheduled job: {error}"))
            })
    }

    /// Finish the occurrence: the turn was delivered, so advance the schedule or retire the job.
    ///
    /// `next_fire_at` is `Some` for a job that lives on and `None` for one whose moment is spent --
    /// a one-shot, or a cron pattern with nothing left in range -- which is the only place a
    /// scheduled job is deleted by the scheduler rather than by a person.
    ///
    /// Written *after* delivery. That ordering existed so a prompt which reliably crashed the
    /// process could not be re-selected on every restart, at the price of losing the occurrence to
    /// any crash at all. The lease's `attempts` counter takes over that job and does it better: a
    /// crash now costs a retry rather than the occurrence, and a job that crashes repeatedly is
    /// parked instead of looping.
    ///
    /// `fired_at` is `None` for an occurrence that was *considered* rather than delivered, which is
    /// what a gate saying no amounts to. `last_fired_at` means a turn happened: recording one for
    /// an evaluation would misreport the job in every listing and re-anchor an interval schedule on
    /// evaluations rather than on fires.
    ///
    /// See [`ClaimClosed`] for what the outcomes mean; only one of them is a problem.
    pub(crate) async fn complete_claim(
        &self,
        id: &str,
        owner: &str,
        next_fire_at: Option<chrono::DateTime<chrono::Utc>>,
        fired_at: Option<chrono::DateTime<chrono::Utc>>,
        gate_baseline: Option<&str>,
    ) -> crate::error::Result<ClaimClosed> {
        let id = id.to_string();
        let owner = owner.to_string();
        let fired_at = fired_at.map(|at| at.to_rfc3339());
        let next_fire_at = next_fire_at.map(|at| at.to_rfc3339());
        let gate_baseline = gate_baseline.map(str::to_string);
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                if match next_fire_at {
                    // `COALESCE` twice, so an evaluation leaves the previous fire time alone rather
                    // than clearing it, and an ungated job leaves the baseline alone rather than
                    // erasing a gate's memory.
                    Some(next) => connection.execute(
                        "UPDATE scheduled_jobs SET next_fire_at = ?3, \
                         last_fired_at = COALESCE(?4, last_fired_at), \
                         gate_last_output = COALESCE(?5, gate_last_output), \
                         claimed_by = NULL, claim_expires_at = NULL, attempts = 0 \
                         WHERE id = ?1 AND claimed_by = ?2",
                        rusqlite::params![id, owner, next, fired_at, gate_baseline],
                    ),
                    None => connection.execute(
                        "DELETE FROM scheduled_jobs WHERE id = ?1 AND claimed_by = ?2",
                        rusqlite::params![id, owner],
                    ),
                }? == 1
                {
                    return Ok(ClaimClosed::Yes);
                }
                // Both statements are scoped to this owner's lease, so a zero count says only "no
                // row carrying my claim" -- which has two causes that mean opposite things. Asked
                // here, on the failure path only, because the answer decides whether the caller
                // has a problem or has merely been canceled.
                let still_there: bool = connection.query_row(
                    "SELECT EXISTS(SELECT 1 FROM scheduled_jobs WHERE id = ?1)",
                    rusqlite::params![id],
                    |row| row.get(0),
                )?;
                Ok(match still_there {
                    true => ClaimClosed::LeaseLost,
                    false => ClaimClosed::RowGone,
                })
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to complete a scheduled job: {error}"))
            })
    }

    /// Write `next_fire_at` verbatim, bypassing the `to_rfc3339` rendering every real writer goes
    /// through.
    ///
    /// Exists so a test can plant the shape [`SAME_OCCURRENCE`]'s second arm is for. Nothing in
    /// meka can produce a timestamp in any other form, so without this the fallback would be
    /// unreachable from the test suite and its guarantee would rest on the comment alone.
    #[cfg(test)]
    pub(crate) async fn set_next_fire_at_verbatim_for_test(
        &self,
        id: &str,
        raw: &str,
    ) -> crate::error::Result<()> {
        let id = id.to_string();
        let raw = raw.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE scheduled_jobs SET next_fire_at = ?2 WHERE id = ?1",
                    rusqlite::params![id, raw],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to plant a timestamp: {error}")))
    }

    /// Drop a job nobody is delivering, for the one-shot whose moment passed while meka was not
    /// running. Returns whether this call is the one that removed it.
    ///
    /// Scoped to the occurrence so that several hosts noticing the same expired job produce one
    /// announcement rather than one each.
    pub(crate) async fn retire_unclaimed(
        &self,
        id: &str,
        occurrence: chrono::DateTime<chrono::Utc>,
        now: chrono::DateTime<chrono::Utc>,
    ) -> crate::error::Result<bool> {
        let id = id.to_string();
        let occurrence = occurrence.to_rfc3339();
        let now = now.to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    &format!(
                        "DELETE FROM scheduled_jobs WHERE id = ?1 AND {} AND {}",
                        SAME_OCCURRENCE,
                        no_live_claim("?3")
                    ),
                    rusqlite::params![id, occurrence, now],
                )
            })
            .await
            .map(|changed| changed == 1)
            .map_err(|error| {
                MekaError::Database(format!("failed to retire scheduled job: {error}"))
            })
    }

    /// Move a job to its next occurrence without claiming or delivering it, for a refusal made
    /// before any lease was taken.
    ///
    /// The occurrence is spent because the job was *considered*: that is the documented rule for a
    /// gate that says no, and a refusal is the same shape. `last_fired_at` is deliberately not
    /// written, because nothing fired.
    ///
    /// Matching nothing is ordinary here and is not reported: another host advanced the same
    /// occurrence first, or took it while this one was deciding, and either way the occurrence has
    /// moved off the value this host read. That is why this returns `()` where
    /// [`Self::retire_unclaimed`] returns `bool` -- there, whether *this* call was the one that
    /// removed the row decides who announces it.
    pub(crate) async fn advance_unclaimed(
        &self,
        id: &str,
        occurrence: chrono::DateTime<chrono::Utc>,
        now: chrono::DateTime<chrono::Utc>,
        next_fire_at: chrono::DateTime<chrono::Utc>,
    ) -> crate::error::Result<()> {
        let id = id.to_string();
        let occurrence = occurrence.to_rfc3339();
        let now = now.to_rfc3339();
        let next_fire_at = next_fire_at.to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    &format!(
                        "UPDATE scheduled_jobs SET next_fire_at = ?4 \
                         WHERE id = ?1 AND {} AND {}",
                        SAME_OCCURRENCE,
                        no_live_claim("?3")
                    ),
                    rusqlite::params![id, occurrence, now, next_fire_at],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to advance a scheduled job: {error}"))
            })
    }
}
/// What became of an attempt to close out an occurrence.
///
/// Three answers rather than two, because "the write matched nothing" has two causes that mean
/// opposite things and prescribe opposite remedies. Reporting the commoner one is how a healthy
/// job came to be told, on every fire, to raise a setting that had nothing to do with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClaimClosed {
    /// Written. The occurrence is spent.
    Yes,
    /// The row is gone: the job was canceled, or its session deleted, while the turn ran. Nothing
    /// is wrong -- there is no occurrence left to close, and a cancellation is meant to win.
    /// `schedule_cancel` is offered to the model in the same breath as `schedule_create`, so a job
    /// that fires, decides it is finished and cancels itself lands here every time.
    RowGone,
    /// The row is there, under someone else's lease. This one is a problem: the occurrence is
    /// still open, so the turn that just ran may be delivered again by whoever holds it now.
    LeaseLost,
}
/// The lease half of every availability test: a row is takeable when nothing holds it, or the hold
/// has run out. `now` names the bound parameter carrying the current instant, because the four
/// queries that ask this number their parameters differently.
///
/// One definition because two are what a stale lease exploits. This clause and `claimed_by IS NULL`
/// look interchangeable and are not: nothing clears `claimed_by` except a release, a completion or
/// a fresh claim, so a host that dies holding a lease leaves it set for good. The due query and
/// [`ScheduleStore::claim_occurrence`] used the expiry; [`ScheduleStore::retire_unclaimed`] and
/// [`ScheduleStore::advance_unclaimed`] used the column, which are the two paths that move an
/// occurrence *without* taking a lease. A row with an expired lease was therefore handed out on
/// every sweep and was invisible to both, so a one-shot past its grace period was never retired,
/// never fired and never logged, and a recurring job refused before its claim never advanced.
pub(crate) fn no_live_claim(now: &str) -> String {
    format!("(claimed_by IS NULL OR claim_expires_at IS NULL OR claim_expires_at < {now})")
}
/// The `next_fire_at` half of every claim-scoped `WHERE`, comparing `?2` against the stored column.
///
/// Textual equality is the fast path and is what matches in practice: every writer renders the
/// column with `DateTime::<Utc>::to_rfc3339`, and re-rendering a value parsed back out of it
/// reproduces the same bytes. The `julianday` arm is there so a row that reached the database any
/// other way -- a hand-edited timestamp, a `Z` suffix, a non-UTC offset -- is still claimable
/// rather than silently unclaimable forever, which is the shape this failure would take: the
/// compare-and-swap would match nothing, on every sweep, and the job would simply never fire again
/// with nothing logged. `julianday` returns `NULL` for anything it cannot parse and `NULL = NULL`
/// is not true, so an unreadable timestamp fails closed. It compares instants at millisecond
/// resolution, which cannot conflate two occurrences: `MIN_EVERY` is a second.
pub(crate) const SAME_OCCURRENCE: &str =
    "(next_fire_at = ?2 OR julianday(next_fire_at) = julianday(?2))";
/// Raw `scheduled_jobs` row, decoded into a [`ScheduledJob`] outside the database closure so parse
/// failures can be logged and skipped individually.
pub(crate) struct ScheduledJobRow {
    pub(crate) id: String,
    pub(crate) session_id: String,
    pub(crate) kind: String,
    pub(crate) spec: String,
    pub(crate) prompt: String,
    pub(crate) gate_kind: Option<String>,
    pub(crate) gate_spec_json: Option<String>,
    pub(crate) gate_last_output: Option<String>,
    pub(crate) gate_permission: Option<String>,
    pub(crate) created_at: String,
    pub(crate) last_fired_at: Option<String>,
    pub(crate) next_fire_at: String,
    pub(crate) attempts: u32,
}
impl ScheduledJobRow {
    pub(crate) fn decode(self) -> std::result::Result<ScheduledJob, String> {
        let parse_time =
            |text: &str| -> std::result::Result<chrono::DateTime<chrono::Utc>, String> {
                chrono::DateTime::parse_from_rfc3339(text)
                    .map(|at| at.with_timezone(&chrono::Utc))
                    .map_err(|error| format!("bad timestamp '{text}': {error}"))
            };

        let gate = match (self.gate_kind, self.gate_spec_json) {
            (Some(kind), Some(spec)) => Some(Gate::from_stored(
                &kind,
                &spec,
                self.gate_last_output,
                // Every write path stores a level alongside the gate, so an absent or unparseable
                // one means a hand-edited or damaged row. Reading that as `Unrestricted` would
                // hand an arbitrary shell command the authority the column exists
                // to record, so it resolves to `None`: the gate is refused at fire
                // time and the user is told to recreate the job. Failing closed
                // costs one re-creation; failing open costs the guarantee.
                //
                // Through `parse_recorded_permission` like the five session-row readers, so the
                // *unreadable* case is heard rather than folded into the absent one. Without the
                // warning the only clue is a later message saying the gate was authorized at
                // `none` -- naming a level the job was never created at.
                crate::permission::parse_recorded_permission(
                    self.gate_permission.as_deref(),
                    &format_args!("the gate on job {}", self.id),
                )
                .unwrap_or(crate::permission::Permission::None),
            )?),
            // A half-written gate is a corrupt row, not a job without a gate: silently dropping the
            // condition would turn a watcher into an unconditional timer.
            (Some(_), None) | (None, Some(_)) => {
                return Err("gate_kind and gate_spec_json must both be set or both be null".into());
            }
            (None, None) => None,
        };

        Ok(ScheduledJob {
            attempts: self.attempts,
            id: self.id,
            session_id: uuid::Uuid::parse_str(&self.session_id)
                .map_err(|error| format!("bad session id '{}': {}", self.session_id, error))?,
            schedule: Schedule::from_stored(&self.kind, &self.spec)?,
            prompt: self.prompt,
            gate,
            created_at: parse_time(&self.created_at)?,
            last_fired_at: self.last_fired_at.as_deref().map(parse_time).transpose()?,
            next_fire_at: parse_time(&self.next_fire_at)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn job_fixture(
        store: &Store,
        session_id: Uuid,
        schedule: crate::schedule::Schedule,
        gate: Option<crate::schedule::Gate>,
    ) -> crate::schedule::ScheduledJob {
        let created_at = chrono::Utc::now();
        let next_fire_at = schedule.next_after(created_at).expect("has a next fire");
        let job = crate::schedule::ScheduledJob {
            attempts: 0,
            id: Uuid::new_v4().to_string(),
            session_id,
            schedule,
            prompt: "check the deploy".to_string(),
            gate,
            created_at,
            last_fired_at: None,
            next_fire_at,
        };
        store
            .schedule_store()
            .create_scheduled_job(&job)
            .await
            .expect("create scheduled job");
        job
    }

    #[tokio::test]
    async fn scheduled_job_round_trips_through_the_database() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let written = job_fixture(
            &store,
            session_id,
            crate::schedule::Schedule::parse_every("30m").expect("parses"),
            Some(crate::schedule::Gate {
                probe: crate::schedule::GateProbe::Shell {
                    command: "gh pr checks 123".to_string(),
                },
                predicate: crate::schedule::GatePredicate::Changed,
                last_output: None,
                permission: crate::permission::Permission::Unrestricted,
            }),
        )
        .await;

        let jobs = store
            .schedule_store()
            .list_scheduled_jobs(session_id)
            .await
            .expect("list jobs");
        assert_eq!(jobs.len(), 1);
        let read = &jobs[0];
        assert_eq!(read.id, written.id);
        assert_eq!(read.prompt, "check the deploy");
        assert_eq!(read.schedule.spec(), written.schedule.spec());
        let gate = read.gate.as_ref().expect("gate survived the round trip");
        assert_eq!(gate.probe, crate::schedule::GateProbe::Shell {
            command: "gh pr checks 123".to_string(),
        });
        assert_eq!(gate.predicate, crate::schedule::GatePredicate::Changed);
    }

    /// Jobs are keyed to the conversation that asked for them, so deleting the session must not
    /// leave a scheduler entry that fires into nothing.
    #[tokio::test]
    async fn deleting_a_session_cascades_to_its_scheduled_jobs() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        job_fixture(
            &store,
            session_id,
            crate::schedule::Schedule::parse_every("30m").expect("parses"),
            None,
        )
        .await;

        store
            .delete_session(session_id)
            .await
            .expect("delete session");

        let due = store
            .schedule_store()
            .list_due_scheduled_jobs(chrono::Utc::now() + chrono::Duration::days(365))
            .await
            .expect("list due");
        assert!(due.is_empty(), "the job should have cascaded away");
    }

    /// `meka schedule list` and `cancel` work from a job id, so they need every job regardless of
    /// when it is due. Approximating it with "due within the next century" leaves a job scheduled
    /// past that horizon invisible and therefore uncancelable.
    #[tokio::test]
    async fn list_all_includes_jobs_beyond_any_due_horizon() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let far_future = chrono::Utc::now() + chrono::Duration::days(365 * 200);
        job_fixture(
            &store,
            session_id,
            crate::schedule::Schedule::At(far_future),
            None,
        )
        .await;

        assert!(
            store
                .schedule_store()
                .list_due_scheduled_jobs(chrono::Utc::now() + chrono::Duration::days(365 * 100))
                .await
                .expect("list due")
                .is_empty(),
            "fixture must actually sit beyond the horizon the old query used"
        );
        assert_eq!(
            store
                .schedule_store()
                .list_all_scheduled_jobs()
                .await
                .expect("list all")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn list_due_selects_only_jobs_whose_time_has_come() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        job_fixture(
            &store,
            session_id,
            crate::schedule::Schedule::parse_every("1h").expect("parses"),
            None,
        )
        .await;

        let none_yet = store
            .schedule_store()
            .list_due_scheduled_jobs(chrono::Utc::now())
            .await
            .expect("list due");
        assert!(none_yet.is_empty(), "not due for another hour");

        let later = store
            .schedule_store()
            .list_due_scheduled_jobs(chrono::Utc::now() + chrono::Duration::hours(2))
            .await
            .expect("list due");
        assert_eq!(later.len(), 1);
    }

    /// The anchoring rule at the storage layer: claiming an occurrence records when the job is next
    /// due and stamping the fire records when it fired, and a subsequent read reconstructs the same
    /// anchor the in-memory scheduler held. Without the `last_fired_at` half, a restart would
    /// re-anchor on `created_at` and replay every occurrence since.
    #[tokio::test]
    async fn stamping_a_fire_persists_the_anchor_for_the_next_process() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let job = job_fixture(
            &store,
            session_id,
            crate::schedule::Schedule::parse_every("1h").expect("parses"),
            None,
        )
        .await;

        let fired_at = chrono::Utc::now();
        let next = job.schedule.next_after(fired_at).expect("has a next fire");
        assert!(
            store
                .schedule_store()
                .claim_occurrence(
                    &job.id,
                    job.next_fire_at,
                    "this-process",
                    fired_at,
                    fired_at + chrono::Duration::hours(1),
                )
                .await
                .expect("claim the occurrence"),
            "the occurrence is unclaimed, so this process takes it"
        );
        store
            .schedule_store()
            .complete_claim(&job.id, "this-process", Some(next), Some(fired_at), None)
            .await
            .expect("record the delivery");

        let reloaded = store
            .schedule_store()
            .list_scheduled_jobs(session_id)
            .await
            .expect("list jobs");
        let reloaded = reloaded.first().expect("job still present");
        assert!(reloaded.last_fired_at.is_some());
        assert_eq!(
            reloaded.anchor(),
            reloaded.last_fired_at.unwrap_or_default()
        );
        // One hour on from the fire, not from creation.
        assert_eq!(
            (reloaded.next_fire_at - fired_at).num_seconds(),
            chrono::Duration::hours(1).num_seconds()
        );
    }

    #[tokio::test]
    async fn cancel_resolves_an_id_prefix_and_reports_ambiguity() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let job = job_fixture(
            &store,
            session_id,
            crate::schedule::Schedule::parse_every("30m").expect("parses"),
            None,
        )
        .await;

        assert!(
            store
                .schedule_store()
                .cancel_scheduled_job(session_id, "nomatch")
                .await
                .expect("cancel runs")
                .is_none()
        );

        let canceled = store
            .schedule_store()
            .cancel_scheduled_job(session_id, job.short_id())
            .await
            .expect("cancel runs")
            .expect("prefix matched");
        assert_eq!(canceled, job.id);
        assert!(
            store
                .schedule_store()
                .list_scheduled_jobs(session_id)
                .await
                .expect("list jobs")
                .is_empty()
        );
    }

    /// A half-written gate is corruption, not "no gate": treating it as an ungated job would
    /// silently promote a watcher into an unconditional timer that fires every interval.
    #[tokio::test]
    async fn a_row_with_a_half_written_gate_is_skipped_not_downgraded() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let job = job_fixture(
            &store,
            session_id,
            crate::schedule::Schedule::parse_every("30m").expect("parses"),
            Some(crate::schedule::Gate {
                probe: crate::schedule::GateProbe::Shell {
                    command: "true".to_string(),
                },
                predicate: crate::schedule::GatePredicate::Succeeded,
                last_output: None,
                permission: crate::permission::Permission::Unrestricted,
            }),
        )
        .await;

        // Prove the row is readable before the corruption, so an empty result afterwards can only
        // be the decoder rejecting it.
        assert_eq!(
            store
                .schedule_store()
                .list_scheduled_jobs(session_id)
                .await
                .expect("list jobs")
                .len(),
            1
        );

        let id = job.id.clone();
        store
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE scheduled_jobs SET gate_spec_json = NULL WHERE id = ?1",
                    rusqlite::params![id],
                )?;
                Ok(())
            })
            .await
            .expect("corrupt the row");

        assert!(
            store
                .schedule_store()
                .list_scheduled_jobs(session_id)
                .await
                .expect("list jobs")
                .is_empty(),
            "the corrupt row is skipped rather than read as ungated"
        );
    }
}
