//! Schedules and their claims: the `schedules` table.

use super::*;
use crate::schedule::*;

/// What creating a schedule did: the row is in, or the session already holds as many as the cap
/// allows and nothing was written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScheduleCreation {
    Created,
    AtCapacity { existing: usize },
}

/// What a cancel removed: the schedule, and the fire it had waiting, which nothing will read now.
/// A host with a feed says the withdrawal on it, as it says any other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CanceledSchedule {
    /// The schedule's whole id.
    pub(crate) id: String,
    /// The session the schedule was on, or `None` when the row named it by something that is no
    /// id, which only a hand edit or damage leaves.
    pub(crate) session_id: Option<uuid::Uuid>,
    /// The inbox items withdrawn with it; one whose row holds no id is withdrawn and not named.
    pub(crate) withdrawn: Vec<uuid::Uuid>,
}

/// Scheduling's slice of the session database, handed out by
/// [`crate::store::Store::schedule_store`].
#[derive(Clone)]
pub(crate) struct ScheduleStore {
    pub(crate) connection: std::sync::Arc<super::connection::Connection>,
    memory: std::sync::Arc<SchedulerMemory>,
}
/// The ids of schedules a listing could not decode and has already warned about, per process.
static UNREADABLE_SCHEDULES_WARNED: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashSet<String>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashSet::new()));

impl ScheduleStore {
    /// A handle on the table over `connection`, sharing the scheduler's memory of held schedules.
    pub(crate) fn new(
        connection: std::sync::Arc<super::connection::Connection>,
        memory: std::sync::Arc<SchedulerMemory>,
    ) -> Self {
        Self { connection, memory }
    }

    /// Persist a new schedule, unless the session already holds `max_schedules`, in which case
    /// nothing is written. The count and the insert are one immediate transaction: two creates
    /// racing for the last slot cannot both find it free, whatever their doors counted ahead of
    /// the call. The caller owns computing `next_fire_at` from the schedule's anchor (see
    /// `Schedule::anchor`).
    pub(crate) async fn create_schedule(
        &self,
        schedule: &Schedule,
        max_schedules: usize,
    ) -> crate::error::Result<ScheduleCreation> {
        let id = schedule.id.clone();
        let session_id = schedule.session_id.to_string();
        let kind = schedule.timing.kind_str().to_string();
        let spec = schedule.timing.spec();
        let timezone = schedule.timing.timezone().map(str::to_string);
        let prompt = schedule.prompt.clone();
        let gate_kind = schedule
            .gate
            .as_ref()
            .map(|gate| gate.probe.kind_str().to_string());
        let gate_spec_json = schedule.gate.as_ref().map(|gate| gate.spec());
        let gate_last_output = schedule
            .gate
            .as_ref()
            .and_then(|gate| gate.last_output.clone());
        let gate_permission = schedule
            .gate
            .as_ref()
            .map(|gate| gate.permission.to_string());
        let created_at = schedule.created_at.to_rfc3339();
        let last_fired_at = schedule.last_fired_at.map(|at| at.to_rfc3339());
        let next_fire_at = schedule.next_fire_at.to_rfc3339();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let transaction = connection
                    .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let existing = transaction.query_row(
                    "SELECT count(*) FROM schedules WHERE session_id = ?1",
                    rusqlite::params![session_id],
                    |row| row.get::<_, i64>(0),
                )?;
                let existing = usize::try_from(existing).unwrap_or(usize::MAX);
                if existing >= max_schedules {
                    return Ok(ScheduleCreation::AtCapacity { existing });
                }
                transaction.execute(
                    "INSERT INTO schedules (id, session_id, kind, spec, prompt, gate_kind, \
                     gate_spec_json, gate_last_output, gate_permission, created_at, \
                     last_fired_at, next_fire_at, timezone) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
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
                        next_fire_at,
                        timezone
                    ],
                )?;
                transaction.commit()?;
                Ok(ScheduleCreation::Created)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to create schedule: {error}")))
    }

    /// Every schedule belonging to one session, soonest first.
    pub(crate) async fn list_schedules(
        &self,
        session_id: uuid::Uuid,
    ) -> crate::error::Result<Vec<Schedule>> {
        self.query_schedules(
            format!(
                "SELECT {SCHEDULE_COLUMNS} FROM schedules WHERE session_id = ?1 \
                 ORDER BY next_fire_at ASC"
            ),
            vec![session_id.to_string()],
        )
        .await
    }

    /// Every schedule in the database, soonest first. Backs `meka schedule list` and `meka schedule
    /// show`, which work from a schedule id and so cannot ask the caller which session to look in.
    pub(crate) async fn list_all_schedules(&self) -> crate::error::Result<Vec<Schedule>> {
        self.query_schedules(
            format!("SELECT {SCHEDULE_COLUMNS} FROM schedules ORDER BY next_fire_at ASC"),
            Vec::new(),
        )
        .await
    }

    /// Every schedule across all sessions whose `next_fire_at` has passed, soonest first. The
    /// scheduler's per-tick query; served by `idx_schedules_next_fire_at`.
    pub(crate) async fn list_due_schedules(
        &self,
        now: chrono::DateTime<chrono::Utc>,
    ) -> crate::error::Result<Vec<Schedule>> {
        self.query_schedules(
            format!(
                "SELECT {SCHEDULE_COLUMNS} FROM schedules WHERE next_fire_at <= ?1 AND {} \
                 ORDER BY next_fire_at ASC",
                no_live_claim("?1")
            ),
            vec![now.to_rfc3339()],
        )
        .await
    }

    /// Shared row decoder. A row that fails to decode (hand-edited spec, a `kind` from a future
    /// version) is skipped with a warning rather than failing the whole query: one bad row must not
    /// stop every other schedule in the database from firing.
    /// Every listing decodes every row, on every poll; the schedules already warned about, so the
    /// warning is said once per process rather than once per poll.
    pub(crate) async fn query_schedules(
        &self,
        sql: String,
        params: Vec<String>,
    ) -> crate::error::Result<Vec<Schedule>> {
        let rows: Vec<ScheduleRow> = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(&sql)?;
                let rows = statement
                    .query_map(rusqlite::params_from_iter(params.iter()), |row| {
                        Ok(ScheduleRow {
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
                            timezone: row.get(13)?,
                        })
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to load schedules: {error}")))?;

        Ok(rows
            .into_iter()
            .filter_map(|row| {
                let id = row.id.clone();
                row.decode()
                    .inspect_err(|error| {
                        // Once per schedule per process: every scheduler poll in every process
                        // lists the schedules, and a row that never decodes would otherwise warn
                        // every five seconds for as long as it stood.
                        if crate::sync::lock(&UNREADABLE_SCHEDULES_WARNED).insert(id.clone()) {
                            tracing::warn!(
                                "skipping unreadable schedule {id}: {error}; cancel it by \
                                 id with `meka schedule cancel`"
                            );
                        } else {
                            tracing::debug!("skipping unreadable schedule {id}: {error}");
                        }
                    })
                    .ok()
            })
            .collect())
    }

    /// The full ids `id_prefix` names: in one session's schedules when `session` names one (the
    /// tool and the screen, which answer inside one conversation), in every session's when it does
    /// not (`meka schedule cancel`, where the operator has an id and no session). An unusable
    /// prefix names nothing, so a blank one cannot reach every schedule.
    ///
    /// Read from the column rather than the decoded listing: a row this build cannot decode is
    /// skipped by every listing, and a schedule nobody can list must still be a schedule somebody
    /// can cancel, or it warns on every poll and keeps its session out of retention for good.
    pub(crate) async fn schedule_ids_matching(
        &self,
        id_prefix: &str,
        session: Option<uuid::Uuid>,
    ) -> crate::error::Result<Vec<String>> {
        if !crate::text::is_usable_id_prefix(id_prefix) {
            return Ok(Vec::new());
        }
        let wanted = crate::text::id_prefix_for_matching(id_prefix);
        let session = session.map(|session| session.to_string());
        let mut ids: Vec<String> = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "SELECT id FROM schedules WHERE ?1 IS NULL OR session_id = ?1 ORDER BY id",
                )?;
                let ids = statement
                    .query_map([session], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<String>>>()?;
                Ok(ids)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to list schedule ids: {error}"))
            })?;
        ids.retain(|id| id.starts_with(&wanted));
        Ok(ids)
    }

    /// Delete a schedule by full or unique-prefix id. Returns what was actually removed, or `None`
    /// when nothing matched *or* when the row was gone by the time the delete ran.
    ///
    /// An ambiguous prefix is an error rather than an arbitrary pick, and a `Config` rather than a
    /// `Database` one. Nothing went wrong with the database; the caller's prefix is
    /// under-specified, and `Config` is the variant that carries that to HTTP as a 422 rather than
    /// a 500. `BackgroundStore::resolve_background_task` says the same thing about the same
    /// condition and already used `Config`, so the two `serve` endpoints answered different
    /// statuses for one mistake. The variant name fits neither of them well; its mapping does, and
    /// a new variant for one condition is not worth the churn through every match.
    pub(crate) async fn cancel_schedule(
        &self,
        session_id: uuid::Uuid,
        id_prefix: &str,
    ) -> crate::error::Result<Option<CanceledSchedule>> {
        let matches = self
            .schedule_ids_matching(id_prefix, Some(session_id))
            .await?;
        let id = match matches.as_slice() {
            [] => return Ok(None),
            [id] => id.clone(),
            several => {
                return Err(MekaError::Usage(format!(
                    "'{}' matches {} schedules; use a longer id",
                    id_prefix,
                    several.len()
                )));
            }
        };

        // `None` when the row was already gone, not `Some(id)`.
        //
        // The listing above and the `DELETE` below are two statements, and a scheduler sweep can
        // retire the row between them: a one-shot's occurrence retires it, and a session deleted
        // elsewhere takes its schedules with it through the foreign key. Reporting the id
        // regardless tells the agent "Canceled schedule abc12345" about a schedule this call did
        // not cancel, which is the same sentence it gets when it did, and there is no way to tell
        // them apart afterwards, because both end with no such row.
        self.delete_schedule(&id).await
    }

    /// Delete a schedule by exact id, without the prefix resolution [`Self::cancel_schedule`] does.
    ///
    /// `Some` when a row was actually removed. Callers that report an outcome to a person or to the
    /// model must not treat `None` as success: it means something else removed the schedule first,
    /// and saying "canceled" then is a claim about work this call did not do.
    ///
    /// The one door every cancel converges on, so it is where a canceled schedule's waiting fire
    /// is withdrawn, in the transaction that removes the row. The scheduler's own retirements do
    /// not come through here: a spent one-shot's fire is the delivery it was spent on.
    pub(crate) async fn delete_schedule(
        &self,
        id: &str,
    ) -> crate::error::Result<Option<CanceledSchedule>> {
        // A schedule that no longer exists cannot be held back for permission, and the memory is
        // otherwise only cleared when a schedule is *authorized* again. A schedule canceled while
        // declined therefore left its id there for the life of a `meka serve`. Forgetting here
        // keeps the memory bounded by the schedules that exist rather than by every schedule that
        // ever did. This is not the only `DELETE` (`retire_unclaimed` and `close_claim_in` have
        // their own), but both of those forget the schedule themselves, on the path that reaches
        // them.
        self.memory.forget(id);
        let id = id.to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let removed = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                let transaction = connection.transaction()?;
                let Some(session_id) = transaction
                    .query_row(
                        "DELETE FROM schedules WHERE id = ?1 RETURNING session_id",
                        rusqlite::params![id],
                        |row| row.get::<_, String>(0),
                    )
                    .optional()?
                else {
                    return Ok(None);
                };
                let withdrawn = super::inbox::withdraw_fires_in(&transaction, &id, &now)?;
                transaction.commit()?;
                Ok(Some((id, session_id, withdrawn)))
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to delete schedule: {error}")))?;
        // Parsed once the cancel has committed: the remedy for a row damaged by hand is to cancel
        // it, so an id that does not parse costs only the announcement that would name it.
        Ok(removed.map(|(id, session_id, withdrawn)| CanceledSchedule {
            id,
            session_id: uuid::Uuid::parse_str(&session_id).ok(),
            withdrawn: withdrawn
                .iter()
                .filter_map(|item_id| uuid::Uuid::parse_str(item_id).ok())
                .collect(),
        }))
    }

    /// Take this occurrence for this process, by leasing the row rather than consuming it.
    ///
    /// One shape for both kinds of schedule. Arbitrating by consuming the row instead would need
    /// two: a recurring schedule advances `next_fire_at`, but a one-shot has no next occurrence and
    /// can only be *deleted*, which overloads one piece of state with two facts ("the user still
    /// wants this schedule" and "this occurrence is available"), and a host that stopped between
    /// the delete and the fire's write would lose the reminder with nothing to recover it from.
    ///
    /// A lease separates the facts. `claimed_by` says who is evaluating this occurrence and
    /// `claim_expires_at` says how long that claim is good for; the row itself stays put, so a
    /// cancellation is an unconditional `DELETE` that always wins and a crash expires rather than
    /// erasing. Every write after this one is scoped to `claimed_by`, so a late writer whose lease
    /// has since been taken changes nothing.
    ///
    /// `occurrence` is the value this host read into its due list, so exactly one host can take a
    /// given occurrence. `attempts` counts claims that were never closed, a host that stopped or a
    /// probe that could not answer and left its lease to expire; see [`Schedule::attempts`] and
    /// [`MAX_CLAIM_ATTEMPTS`].
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
                        "UPDATE schedules \
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
            .map_err(|error| MekaError::Database(format!("failed to claim schedule: {error}")))
    }

    /// Give back the lease of each schedule whose gate this process was running when its sweep was
    /// stopped, taking back the attempt that claim counted, for a host on its way out: the schedule
    /// is due again at the next start rather than once `claim_lease` runs out, and leaving is not
    /// counted toward parking it. A lease the sweep left to run out is not among them: that wait is
    /// the backoff of a one-shot whose gate could not be evaluated, or of a fire the store refused,
    /// and given back at every exit it would have the gate tried at every start and the schedule
    /// never parked.
    pub(crate) async fn give_back_gates_cut_short(&self) -> crate::error::Result<()> {
        let schedule_ids = self.memory.take_gates_cut_short();
        if schedule_ids.is_empty() {
            return Ok(());
        }
        let owner = self.memory.owner().to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "UPDATE schedules SET claimed_by = NULL, claim_expires_at = NULL, \
                     attempts = max(attempts - 1, 0) WHERE id = ?1 AND claimed_by = ?2",
                )?;
                for schedule_id in &schedule_ids {
                    statement.execute(rusqlite::params![schedule_id, owner])?;
                }
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to give back leases: {error}")))
    }

    /// Close the occurrence without a fire: the gate was asked and said no, or could not be
    /// evaluated, so advance the schedule or retire it, recording `gate_baseline` when the gate
    /// measured one.
    ///
    /// See [`close_claim_in`] for the write and [`ClaimClosed`] for what the outcomes mean; only
    /// one of them is a problem.
    pub(crate) async fn complete_claim(
        &self,
        id: &str,
        owner: &str,
        next_fire_at: Option<chrono::DateTime<chrono::Utc>>,
        gate_baseline: Option<&str>,
    ) -> crate::error::Result<ClaimClosed> {
        let id = id.to_string();
        let owner = owner.to_string();
        let next_fire_at = next_fire_at.map(|at| at.to_rfc3339());
        let gate_baseline = gate_baseline.map(str::to_string);
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                close_claim_in(
                    connection,
                    &id,
                    &owner,
                    next_fire_at.as_deref(),
                    None,
                    gate_baseline.as_deref(),
                )
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to complete a schedule: {error}")))
    }

    /// Close the occurrence `fire` spends and queue the fire in the inbox of `schedule`'s session,
    /// in one transaction: the close [`Self::complete_claim`] makes, recording `fire.fired_at` as
    /// the schedule's `last_fired_at`, and, only if that took, the item
    /// ([`super::inbox::queue_fire_in`]). A schedule canceled while its gate ran, or a lease
    /// another host has taken since, queues nothing.
    ///
    /// Written once the gate has passed, never before it: until then this host holds a lease and
    /// the row is untouched, so a crash costs a retry rather than the occurrence.
    pub(crate) async fn queue_fire(
        &self,
        schedule: &Schedule,
        owner: &str,
        next_fire_at: Option<chrono::DateTime<chrono::Utc>>,
        gate_baseline: Option<&str>,
        fire: super::inbox::Fire,
    ) -> crate::error::Result<ClaimClosed<super::inbox::QueuedFire>> {
        let id = schedule.id.clone();
        let session_id = schedule.session_id.to_string();
        let body = schedule.prompt.clone();
        let owner = owner.to_string();
        let next_fire_at = next_fire_at.map(|at| at.to_rfc3339());
        let gate_baseline = gate_baseline.map(str::to_string);
        let now = chrono::Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                // Immediate, because the read of a waiting fire decides the write.
                let transaction = connection
                    .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let fired_at = fire.fired_at.to_rfc3339();
                let closed = match close_claim_in(
                    &transaction,
                    &id,
                    &owner,
                    next_fire_at.as_deref(),
                    Some(&fired_at),
                    gate_baseline.as_deref(),
                )? {
                    ClaimClosed::Yes(()) => ClaimClosed::Yes(super::inbox::queue_fire_in(
                        &transaction,
                        &session_id,
                        &body,
                        &fire,
                        &now,
                    )?),
                    ClaimClosed::RowGone => ClaimClosed::RowGone,
                    ClaimClosed::LeaseLost => ClaimClosed::LeaseLost,
                };
                transaction.commit()?;
                Ok(closed)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to queue a schedule's fire: {error}"))
            })
    }

    /// Write `next_fire_at` verbatim, bypassing the `to_rfc3339` rendering every real writer goes
    /// through.
    ///
    /// Exists so a test can plant the shape [`SAME_OCCURRENCE`]'s second arm is for. Nothing in
    /// Meka can produce a timestamp in any other form, so without this the fallback would be
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
                    "UPDATE schedules SET next_fire_at = ?2 WHERE id = ?1",
                    rusqlite::params![id, raw],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to plant a timestamp: {error}")))
    }

    /// Queue `fire` for `session_id` the way a sweep does, through [`Self::queue_fire`]: a one-shot
    /// of `fire`'s schedule id and `prompt`, due when `fire` was, claimed and spent by the write.
    /// For a test that needs a fire in the inbox and not the sweep that queues one.
    #[cfg(test)]
    pub(crate) async fn queue_fire_for_test(
        &self,
        session_id: uuid::Uuid,
        prompt: &str,
        fire: super::inbox::Fire,
    ) -> crate::error::Result<super::inbox::QueuedFire> {
        let schedule = Schedule {
            attempts: 0,
            id: fire.schedule_id.clone(),
            session_id,
            timing: Timing::At(fire.due_at),
            prompt: prompt.to_string(),
            gate: None,
            created_at: fire.due_at,
            last_fired_at: None,
            next_fire_at: fire.due_at,
        };
        self.create_schedule(&schedule, usize::MAX).await?;
        let now = chrono::Utc::now();
        if !self
            .claim_occurrence(
                &schedule.id,
                schedule.next_fire_at,
                "test",
                now,
                now + chrono::Duration::hours(1),
            )
            .await?
        {
            return Err(MekaError::Database("the occurrence is claimed".to_string()));
        }
        match self.queue_fire(&schedule, "test", None, None, fire).await? {
            ClaimClosed::Yes(queued) => Ok(queued),
            closed => Err(MekaError::Database(format!(
                "the fire was not queued: {closed:?}"
            ))),
        }
    }

    /// Drop a schedule nobody is delivering, for the one-shot whose moment passed while Meka was
    /// not running. Returns whether this call is the one that removed it.
    ///
    /// Scoped to the occurrence so that several hosts noticing the same expired schedule produce
    /// one announcement rather than one each.
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
                        "DELETE FROM schedules WHERE id = ?1 AND {} AND {}",
                        SAME_OCCURRENCE,
                        no_live_claim("?3")
                    ),
                    rusqlite::params![id, occurrence, now],
                )
            })
            .await
            .map(|changed| changed == 1)
            .map_err(|error| MekaError::Database(format!("failed to retire schedule: {error}")))
    }

    /// Move a schedule to its next occurrence without claiming or delivering it, for a refusal made
    /// before any lease was taken.
    ///
    /// The occurrence is spent because the schedule was *considered*: that is the documented rule
    /// for a gate that says no, and a refusal is the same shape. `last_fired_at` is deliberately
    /// not written, because nothing fired.
    ///
    /// Matching nothing is ordinary here and is not reported: another host advanced the same
    /// occurrence first, or took it while this one was deciding, and either way the occurrence has
    /// moved off the value this host read. That is why this returns `()` where
    /// [`Self::retire_unclaimed`] returns `bool`: there, whether *this* call was the one that
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
                        "UPDATE schedules SET next_fire_at = ?4 \
                         WHERE id = ?1 AND {} AND {}",
                        SAME_OCCURRENCE,
                        no_live_claim("?3")
                    ),
                    rusqlite::params![id, occurrence, now, next_fire_at],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to advance a schedule: {error}")))
    }
}
/// Close this host's claim on an occurrence over `connection`: advance a schedule that lives on to
/// `next_fire_at`, or retire one whose moment is spent (`None`: a one-shot, or a cron pattern with
/// nothing left in range), which is the only place a schedule is deleted by the scheduler rather
/// than by a person. The write [`ScheduleStore::complete_claim`] and [`ScheduleStore::queue_fire`]
/// share.
///
/// `fired_at` is `Some` only for an occurrence that fired. One that was *considered*, which is
/// what a gate saying no amounts to, records none: `last_fired_at` means a fire happened, and
/// recording one for an evaluation would misreport the schedule in every listing and re-anchor an
/// interval schedule on evaluations rather than on fires.
fn close_claim_in(
    connection: &rusqlite::Connection,
    id: &str,
    owner: &str,
    next_fire_at: Option<&str>,
    fired_at: Option<&str>,
    gate_baseline: Option<&str>,
) -> rusqlite::Result<ClaimClosed> {
    if match next_fire_at {
        // `COALESCE` twice, so an evaluation leaves the previous fire time alone rather than
        // clearing it, and an ungated schedule leaves the baseline alone rather than erasing a
        // gate's memory.
        Some(next) => connection.execute(
            "UPDATE schedules SET next_fire_at = ?3, \
             last_fired_at = COALESCE(?4, last_fired_at), \
             gate_last_output = COALESCE(?5, gate_last_output), \
             claimed_by = NULL, claim_expires_at = NULL, attempts = 0 \
             WHERE id = ?1 AND claimed_by = ?2",
            rusqlite::params![id, owner, next, fired_at, gate_baseline],
        ),
        None => connection.execute(
            "DELETE FROM schedules WHERE id = ?1 AND claimed_by = ?2",
            rusqlite::params![id, owner],
        ),
    }? == 1
    {
        return Ok(ClaimClosed::Yes(()));
    }
    // Both statements are scoped to this owner's lease, so a zero count says only "no row
    // carrying my claim", which has two causes that mean opposite things. Asked here, on the
    // failure path only, because the answer decides whether the caller has a problem or has
    // merely been canceled.
    let still_there: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM schedules WHERE id = ?1)",
        rusqlite::params![id],
        |row| row.get(0),
    )?;
    Ok(match still_there {
        true => ClaimClosed::LeaseLost,
        false => ClaimClosed::RowGone,
    })
}
/// What became of an attempt to close out an occurrence, with what closing it produced when it
/// took: nothing for an occurrence that did not fire, the queued fire for one that did.
///
/// Three answers rather than two, because "the write matched nothing" has two causes that mean
/// opposite things and prescribe opposite remedies. Reporting the commoner one is how a healthy
/// schedule came to be told, on every fire, to raise a setting that had nothing to do with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClaimClosed<Done = ()> {
    /// Written. The occurrence is spent.
    Yes(Done),
    /// The row is gone: the schedule was canceled, or its session deleted, while its gate ran.
    /// Nothing is wrong: there is no occurrence left to close, and a cancellation is meant to win.
    RowGone,
    /// The row is there, under someone else's lease. This one is a problem: the occurrence is
    /// still open, and whoever holds it now evaluates it again.
    LeaseLost,
}
/// The lease half of every availability test: a row is takeable when nothing holds it, or the hold
/// has run out. `now` names the bound parameter carrying the current instant, because the four
/// queries that ask this number their parameters differently.
///
/// One definition because two are what a stale lease exploits. This clause and `claimed_by IS NULL`
/// look interchangeable and are not: nothing clears `claimed_by` except a completion or a fresh
/// claim, so a host that dies holding a lease leaves it set for good. With the due query
/// and [`ScheduleStore::claim_occurrence`] on the expiry but [`ScheduleStore::retire_unclaimed`]
/// and [`ScheduleStore::advance_unclaimed`] (the two paths that move an occurrence *without*
/// taking a lease) on the column, a row with an expired lease is handed out on every sweep and
/// invisible to both, so a one-shot past its grace period is never retired, never fired and never
/// logged, and a recurring schedule refused before its claim never advances.
pub(crate) fn no_live_claim(now: &str) -> String {
    format!("(claimed_by IS NULL OR claim_expires_at IS NULL OR claim_expires_at < {now})")
}
/// The `next_fire_at` half of every claim-scoped `WHERE`, comparing `?2` against the stored column.
///
/// Textual equality is the fast path and is what matches in practice: every writer renders the
/// column with `DateTime::<Utc>::to_rfc3339`, and re-rendering a value parsed back out of it
/// reproduces the same bytes. The `julianday` arm is there so a row that reached the database any
/// other way (a hand-edited timestamp, a `Z` suffix, a non-UTC offset) is still claimable
/// rather than silently unclaimable forever, which is the shape this failure would take: the
/// compare-and-swap would match nothing, on every sweep, and the schedule would simply never fire
/// again with nothing logged. `julianday` returns `NULL` for anything it cannot parse and `NULL =
/// NULL` is not true, so an unreadable timestamp fails closed. It compares instants at millisecond
/// resolution, which cannot conflate two occurrences: `MIN_EVERY` is a second.
pub(crate) const SAME_OCCURRENCE: &str =
    "(next_fire_at = ?2 OR julianday(next_fire_at) = julianday(?2))";
/// The columns every schedule read selects, in the order [`ScheduleRow`] reads them.
const SCHEDULE_COLUMNS: &str = "id, session_id, kind, spec, prompt, gate_kind, gate_spec_json, \
                           gate_last_output, gate_permission, created_at, last_fired_at, \
                           next_fire_at, attempts, timezone";
/// Raw `schedules` row, decoded into a [`Schedule`] outside the database closure so parse
/// failures can be logged and skipped individually.
pub(crate) struct ScheduleRow {
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
    pub(crate) timezone: Option<String>,
}
impl ScheduleRow {
    /// The row as a [`Schedule`], or what about it did not parse.
    pub(crate) fn decode(self) -> std::result::Result<Schedule, String> {
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
                // time and the user is told to recreate the schedule. Failing closed
                // costs one re-creation; failing open costs the guarantee.
                //
                // Through `parse_recorded_permission` like the five session-row readers, so the
                // *unreadable* case is heard rather than folded into the absent one. Without the
                // warning the only clue is a later message saying the gate was authorized at
                // `none`, naming a level the schedule was never created at.
                crate::permission::parse_recorded_permission(
                    self.gate_permission.as_deref(),
                    &format_args!("the gate on schedule {}", self.id),
                )
                .unwrap_or(crate::permission::Permission::None),
            )?),
            // A half-written gate is a corrupt row, not a schedule without a gate: silently
            // dropping the condition would turn a watcher into an unconditional timer.
            (Some(_), None) | (None, Some(_)) => {
                return Err("gate_kind and gate_spec_json must both be set or both be null".into());
            }
            (None, None) => None,
        };

        Ok(Schedule {
            attempts: self.attempts,
            id: self.id,
            session_id: uuid::Uuid::parse_str(&self.session_id)
                .map_err(|error| format!("bad session id '{}': {}", self.session_id, error))?,
            timing: Timing::from_stored(&self.kind, &self.spec, self.timezone.as_deref())?,
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

    /// A row this build cannot decode is still a schedule somebody can cancel: resolved over the
    /// raw ids, since every listing skips it, and left uncancelable it warns on every poll and
    /// keeps its session out of retention for good.
    #[tokio::test]
    async fn an_unreadable_schedule_can_still_be_canceled_by_id() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let id = Uuid::new_v4().to_string();
        let row_id = id.clone();
        let session = session_id.to_string();
        store
            .connection
            .call(move |connection| -> rusqlite::Result<()> {
                connection.execute(
                    "INSERT INTO schedules (id, session_id, kind, spec, prompt, created_at, \
                     next_fire_at) VALUES (?1, ?2, 'bogus', 'nothing this build reads', 'p', ?3, ?3)",
                    rusqlite::params![row_id, session, chrono::Utc::now().to_rfc3339()],
                )?;
                Ok(())
            })
            .await
            .expect("a row no build decodes");
        assert!(
            store
                .schedule_store()
                .list_schedules(session_id)
                .await
                .expect("list")
                .is_empty(),
            "the listing skips it"
        );
        let canceled = store
            .schedule_store()
            .cancel_schedule(session_id, &id[..8])
            .await
            .expect("cancel resolves by raw id");
        assert_eq!(
            canceled.map(|canceled| canceled.id).as_deref(),
            Some(id.as_str())
        );
        let left: i64 = store
            .connection
            .call(move |connection| {
                connection.query_row(
                    "SELECT count(*) FROM schedules WHERE id = ?1",
                    [id],
                    |row| row.get(0),
                )
            })
            .await
            .expect("count");
        assert_eq!(left, 0, "the row is gone");
    }

    /// A row that names its session, or a fire that names itself, by something that is no id is
    /// still canceled, and the fire withdrawn: canceling is the remedy for a damaged row, and only
    /// what would announce it is left out.
    #[tokio::test]
    async fn a_schedule_whose_ids_do_not_parse_can_still_be_canceled() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let id = Uuid::new_v4().to_string();
        let row_id = id.clone();
        let session = session_id.to_string();
        store
            .connection
            .call(move |connection| -> rusqlite::Result<()> {
                let now = chrono::Utc::now().to_rfc3339();
                // A hand edit the foreign key would refuse.
                connection.execute_batch("PRAGMA foreign_keys = OFF")?;
                connection.execute(
                    "INSERT INTO schedules (id, session_id, kind, spec, prompt, created_at, \
                     next_fire_at) VALUES (?1, 'not-a-session-id', 'every', '1h', 'p', ?2, ?2)",
                    rusqlite::params![row_id, now],
                )?;
                connection.execute_batch("PRAGMA foreign_keys = ON")?;
                connection.execute(
                    "INSERT INTO inbox_items (id, session_id, class, source, body, created_at, \
                     schedule_id, occurrences, due_at, fired_at) \
                     VALUES ('not-an-item-id', ?1, 'steer', '', 'p', ?2, ?3, 1, ?2, ?2)",
                    rusqlite::params![session, now, row_id],
                )?;
                Ok(())
            })
            .await
            .expect("rows no build writes");

        assert_eq!(
            store
                .schedule_store()
                .delete_schedule(&id)
                .await
                .expect("the cancel commits"),
            Some(CanceledSchedule {
                id: id.clone(),
                session_id: None,
                withdrawn: Vec::new(),
            })
        );
        let (left, failure) = store
            .connection
            .call(
                move |connection| -> rusqlite::Result<(i64, Option<String>)> {
                    Ok((
                        connection.query_row(
                            "SELECT count(*) FROM schedules WHERE id = ?1",
                            [id],
                            |row| row.get(0),
                        )?,
                        connection.query_row(
                            "SELECT failure FROM inbox_items WHERE id = 'not-an-item-id'",
                            [],
                            |row| row.get(0),
                        )?,
                    ))
                },
            )
            .await
            .expect("read back");
        assert_eq!(
            (left, failure.as_deref()),
            (0, Some("its schedule was canceled")),
            "the row is gone and its fire withdrawn"
        );
    }

    async fn schedule_fixture(
        store: &Store,
        session_id: Uuid,
        timing: crate::schedule::Timing,
        gate: Option<crate::schedule::Gate>,
    ) -> crate::schedule::Schedule {
        let created_at = chrono::Utc::now();
        let next_fire_at = timing.next_after(created_at).expect("has a next fire");
        let schedule = crate::schedule::Schedule {
            attempts: 0,
            id: Uuid::new_v4().to_string(),
            session_id,
            timing,
            prompt: "check the deploy".to_string(),
            gate,
            created_at,
            last_fired_at: None,
            next_fire_at,
        };
        store
            .schedule_store()
            .create_schedule(&schedule, usize::MAX)
            .await
            .expect("create schedule");
        schedule
    }

    /// The cap is held in the transaction that writes the row, so no door's count-then-insert can
    /// let a race past it, and a refusal writes nothing.
    #[tokio::test]
    async fn the_schedule_cap_is_enforced_where_the_row_is_written() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let schedule = |prompt: &str| {
            let now = chrono::Utc::now();
            crate::schedule::Schedule {
                attempts: 0,
                id: Uuid::new_v4().to_string(),
                session_id,
                timing: crate::schedule::Timing::parse_every("1h").expect("parses"),
                prompt: prompt.to_string(),
                gate: None,
                created_at: now,
                last_fired_at: None,
                next_fire_at: now + chrono::Duration::hours(1),
            }
        };
        let schedule_store = store.schedule_store();
        assert_eq!(
            schedule_store
                .create_schedule(&schedule("first"), 1)
                .await
                .expect("first"),
            ScheduleCreation::Created
        );
        assert_eq!(
            schedule_store
                .create_schedule(&schedule("second"), 1)
                .await
                .expect("second"),
            ScheduleCreation::AtCapacity { existing: 1 }
        );
        assert_eq!(
            schedule_store
                .list_schedules(session_id)
                .await
                .expect("list")
                .len(),
            1,
            "nothing is written past the cap"
        );
        assert_eq!(
            schedule_store
                .create_schedule(&schedule("third"), usize::MAX)
                .await
                .expect("third"),
            ScheduleCreation::Created
        );
    }

    #[tokio::test]
    async fn schedule_round_trips_through_the_database() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let written = schedule_fixture(
            &store,
            session_id,
            crate::schedule::Timing::parse_every("30m").expect("parses"),
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

        let schedules = store
            .schedule_store()
            .list_schedules(session_id)
            .await
            .expect("list schedules");
        assert_eq!(schedules.len(), 1);
        let read = &schedules[0];
        assert_eq!(read.id, written.id);
        assert_eq!(read.prompt, "check the deploy");
        assert_eq!(read.timing.spec(), written.timing.spec());
        let gate = read.gate.as_ref().expect("gate survived the round trip");
        assert_eq!(gate.probe, crate::schedule::GateProbe::Shell {
            command: "gh pr checks 123".to_string(),
        });
        assert_eq!(gate.predicate, crate::schedule::GatePredicate::Changed);
    }

    /// Schedules are keyed to the conversation that asked for them, so deleting the session must
    /// not leave a scheduler entry that fires into nothing.
    #[tokio::test]
    async fn deleting_a_session_cascades_to_its_schedules() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        schedule_fixture(
            &store,
            session_id,
            crate::schedule::Timing::parse_every("30m").expect("parses"),
            None,
        )
        .await;

        store
            .delete_session(session_id)
            .await
            .expect("delete session");

        let due = store
            .schedule_store()
            .list_due_schedules(chrono::Utc::now() + chrono::Duration::days(365))
            .await
            .expect("list due");
        assert!(due.is_empty(), "the schedule should have cascaded away");
    }

    /// `meka schedule list` and `cancel` work from a schedule id, so they need every schedule
    /// regardless of when it is due. Approximating it with "due within the next century" leaves a
    /// schedule due past that horizon invisible and therefore uncancelable.
    #[tokio::test]
    async fn list_all_includes_schedules_beyond_any_due_horizon() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let far_future = chrono::Utc::now() + chrono::Duration::days(365 * 200);
        schedule_fixture(
            &store,
            session_id,
            crate::schedule::Timing::At(far_future),
            None,
        )
        .await;

        assert!(
            store
                .schedule_store()
                .list_due_schedules(chrono::Utc::now() + chrono::Duration::days(365 * 100))
                .await
                .expect("list due")
                .is_empty(),
            "fixture must actually sit beyond the horizon the old query used"
        );
        assert_eq!(
            store
                .schedule_store()
                .list_all_schedules()
                .await
                .expect("list all")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn list_due_selects_only_schedules_whose_time_has_come() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        schedule_fixture(
            &store,
            session_id,
            crate::schedule::Timing::parse_every("1h").expect("parses"),
            None,
        )
        .await;

        let none_yet = store
            .schedule_store()
            .list_due_schedules(chrono::Utc::now())
            .await
            .expect("list due");
        assert!(none_yet.is_empty(), "not due for another hour");

        let later = store
            .schedule_store()
            .list_due_schedules(chrono::Utc::now() + chrono::Duration::hours(2))
            .await
            .expect("list due");
        assert_eq!(later.len(), 1);
    }

    /// The anchoring rule at the storage layer: claiming an occurrence records when the schedule is
    /// next due and stamping the fire records when it fired, and a subsequent read reconstructs the
    /// same anchor the in-memory scheduler held. Without the `last_fired_at` half, a restart would
    /// re-anchor on `created_at` and replay every occurrence since.
    #[tokio::test]
    async fn stamping_a_fire_persists_the_anchor_for_the_next_process() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let schedule = schedule_fixture(
            &store,
            session_id,
            crate::schedule::Timing::parse_every("1h").expect("parses"),
            None,
        )
        .await;

        let fired_at = chrono::Utc::now();
        let next = schedule
            .timing
            .next_after(fired_at)
            .expect("has a next fire");
        assert!(
            store
                .schedule_store()
                .claim_occurrence(
                    &schedule.id,
                    schedule.next_fire_at,
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
            .queue_fire(
                &schedule,
                "this-process",
                Some(next),
                None,
                fire_of(&schedule, fired_at),
            )
            .await
            .expect("record the fire");

        let reloaded = store
            .schedule_store()
            .list_schedules(session_id)
            .await
            .expect("list schedules");
        let reloaded = reloaded.first().expect("schedule still present");
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

    /// A host on its way out gives back the lease of a schedule whose gate its stopped sweep was
    /// running, and takes back the attempt that claim counted, so the occurrence is anyone's again
    /// at once. A lease its sweep left to run out stays, and so does another host's.
    #[tokio::test]
    async fn giving_back_releases_the_gate_a_stopped_sweep_was_running_and_no_other_claim() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let every_hour = || crate::schedule::Timing::parse_every("1h").expect("parses");
        let cut_short = schedule_fixture(&store, session_id, every_hour(), None).await;
        let left_to_run_out = schedule_fixture(&store, session_id, every_hour(), None).await;
        let theirs = schedule_fixture(&store, session_id, every_hour(), None).await;
        let owner = store.scheduler_memory().owner().to_string();
        let now = chrono::Utc::now();
        for (schedule, host) in [
            (&cut_short, owner.as_str()),
            (&left_to_run_out, owner.as_str()),
            (&theirs, "host-b"),
        ] {
            assert!(
                store
                    .schedule_store()
                    .claim_occurrence(
                        &schedule.id,
                        schedule.next_fire_at,
                        host,
                        now,
                        now + chrono::Duration::hours(1),
                    )
                    .await
                    .expect("claim"),
                "the occurrence is takeable"
            );
        }

        // What a sweep stopped inside the first one's gate leaves behind. The second one's gate
        // returned, and its lease was left on purpose.
        let memory = store.scheduler_memory();
        memory.gate_started(&cut_short.id);
        memory.gate_started(&left_to_run_out.id);
        memory.gate_ended(&left_to_run_out.id);

        store
            .schedule_store()
            .give_back_gates_cut_short()
            .await
            .expect("give back");
        let claim = async |id: &str| {
            let id = id.to_string();
            store
                .connection
                .call(
                    move |connection| -> rusqlite::Result<(Option<String>, Option<String>, i64)> {
                        connection.query_row(
                            "SELECT claimed_by, claim_expires_at, attempts FROM schedules \
                             WHERE id = ?1",
                            [id],
                            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                        )
                    },
                )
                .await
                .expect("the row")
        };
        assert_eq!(claim(&cut_short.id).await, (None, None, 0));
        for (schedule, host) in [(&left_to_run_out, owner.as_str()), (&theirs, "host-b")] {
            let (holder, expires, attempts) = claim(&schedule.id).await;
            assert_eq!(
                (holder.as_deref(), expires.is_some(), attempts),
                (Some(host), true, 1),
                "a claim no stopped gate held stays as it was"
            );
        }
    }

    /// The fire a sweep queues for `schedule`'s occurrence, fired at `fired_at`.
    fn fire_of(
        schedule: &crate::schedule::Schedule,
        fired_at: chrono::DateTime<chrono::Utc>,
    ) -> crate::store::inbox::Fire {
        crate::store::inbox::Fire {
            schedule_id: schedule.id.clone(),
            gate_output: None,
            occurrences: 1,
            due_at: schedule.next_fire_at,
            fired_at,
        }
    }

    /// Claim `schedule`'s occurrence for `owner` and queue `fire` as spending it, moving the
    /// schedule on an hour, as a sweep that fired it does.
    async fn fire(
        store: &Store,
        schedule: &crate::schedule::Schedule,
        owner: &str,
        fire: crate::store::inbox::Fire,
    ) -> ClaimClosed<crate::store::inbox::QueuedFire> {
        let now = chrono::Utc::now();
        assert!(
            store
                .schedule_store()
                .claim_occurrence(
                    &schedule.id,
                    schedule.next_fire_at,
                    owner,
                    now,
                    now + chrono::Duration::hours(1),
                )
                .await
                .expect("claim"),
            "the occurrence is takeable"
        );
        store
            .schedule_store()
            .queue_fire(
                schedule,
                owner,
                Some(schedule.next_fire_at + chrono::Duration::hours(1)),
                Some("baseline"),
                fire,
            )
            .await
            .expect("queue the fire")
    }

    /// A fire is queued with its facts, as a `steer` of the schedule's words, by the write that
    /// spends its occurrence: the schedule moves on, records when it fired and what its gate saw,
    /// and gives its claim up.
    #[tokio::test]
    async fn a_fire_is_queued_with_its_facts_as_its_occurrence_is_spent() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let schedule = schedule_fixture(
            &store,
            session_id,
            crate::schedule::Timing::parse_every("1h").expect("parses"),
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
        let fired_at = schedule.next_fire_at + chrono::Duration::seconds(5);
        let mut facts = fire_of(&schedule, fired_at);
        facts.gate_output = Some("state: failed".to_string());
        facts.occurrences = 3;

        let ClaimClosed::Yes(queued) = fire(&store, &schedule, "host-a", facts.clone()).await
        else {
            panic!("the claim closes");
        };
        assert_eq!((queued.replaces, queued.occurrences), (None, 3));
        let item = store
            .inbox_store()
            .get(queued.item_id)
            .await
            .expect("get")
            .expect("the fire is in the inbox");
        assert_eq!(item.origin, crate::store::inbox::InboxOrigin::Fire(facts));
        assert_eq!(
            (item.session_id, item.class, item.body.as_str()),
            (
                session_id,
                crate::store::inbox::InboxClass::Steer,
                "check the deploy"
            )
        );
        assert_eq!(item.state(), crate::store::inbox::InboxState::Pending);

        let after = store
            .schedule_store()
            .list_schedules(session_id)
            .await
            .expect("list")
            .into_iter()
            .next()
            .expect("a recurring schedule lives on");
        assert_eq!(
            after.next_fire_at,
            schedule.next_fire_at + chrono::Duration::hours(1)
        );
        assert_eq!(after.last_fired_at, Some(fired_at));
        assert_eq!(
            after
                .gate
                .as_ref()
                .and_then(|gate| gate.last_output.as_deref()),
            Some("baseline"),
            "and what its gate saw"
        );
        assert_eq!(after.attempts, 0, "and its claim is given up");
    }

    /// The two halves of a fire commit together: a fire the store refuses to queue leaves the
    /// occurrence unspent, claimed and due again once the lease runs out, rather than spent with
    /// nothing in the inbox to show for it.
    #[tokio::test]
    async fn a_fire_the_inbox_refuses_spends_no_occurrence() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let schedule = schedule_fixture(
            &store,
            session_id,
            crate::schedule::Timing::parse_every("1h").expect("parses"),
            None,
        )
        .await;
        store
            .connection
            .call(|connection| {
                connection.execute_batch(
                    "CREATE TEMP TRIGGER refuse_fires BEFORE INSERT ON inbox_items \
                     BEGIN SELECT RAISE(ABORT, 'refused'); END;",
                )
            })
            .await
            .expect("refuse every item");
        let now = chrono::Utc::now();
        assert!(
            store
                .schedule_store()
                .claim_occurrence(
                    &schedule.id,
                    schedule.next_fire_at,
                    "host-a",
                    now,
                    now + chrono::Duration::hours(1),
                )
                .await
                .expect("claim")
        );
        store
            .schedule_store()
            .queue_fire(
                &schedule,
                "host-a",
                Some(schedule.next_fire_at + chrono::Duration::hours(1)),
                Some("baseline"),
                fire_of(&schedule, now),
            )
            .await
            .expect_err("the inbox refuses the fire");

        let after = store
            .schedule_store()
            .list_all_schedules()
            .await
            .expect("list")
            .into_iter()
            .next()
            .expect("the schedule is there");
        assert_eq!(
            (after.next_fire_at, after.last_fired_at, after.attempts),
            (schedule.next_fire_at, None, 1),
            "the occurrence is not spent, and the claim stands to expire"
        );
    }

    /// An occurrence that fires while its schedule's last fire waits joins it, under a new id: the
    /// count grows, and the gate's output and when it fired are the new fire's, while the item
    /// keeps its age, when its first occurrence was due and its retry state.
    #[tokio::test]
    async fn a_second_occurrence_joins_the_waiting_fire_under_a_new_id() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let schedule = schedule_fixture(
            &store,
            session_id,
            crate::schedule::Timing::parse_every("1h").expect("parses"),
            None,
        )
        .await;
        let first_fired = schedule.next_fire_at;
        let mut first = fire_of(&schedule, first_fired);
        first.gate_output = Some("one".to_string());
        let ClaimClosed::Yes(waiting) = fire(&store, &schedule, "host-a", first).await else {
            panic!("the first fire is queued");
        };
        let not_before = chrono::Utc::now() + chrono::Duration::minutes(5);
        store
            .inbox_store()
            .defer_pending(&[waiting.item_id], not_before, Some("{\"title\":\"down\"}"))
            .await
            .expect("the first fire's turn failed and it waits");
        let before = store
            .inbox_store()
            .get(waiting.item_id)
            .await
            .expect("get")
            .expect("the waiting fire");

        let advanced = store
            .schedule_store()
            .list_schedules(session_id)
            .await
            .expect("list")
            .into_iter()
            .next()
            .expect("the schedule");
        let second_fired = advanced.next_fire_at;
        let mut second = fire_of(&advanced, second_fired);
        second.gate_output = Some("two".to_string());
        second.occurrences = 2;
        let ClaimClosed::Yes(joined) = fire(&store, &advanced, "host-a", second).await else {
            panic!("the second fire is queued");
        };

        assert_ne!(
            joined.item_id, waiting.item_id,
            "the joined fire takes a new id"
        );
        assert_eq!(
            (joined.replaces, joined.occurrences),
            (Some(waiting.item_id), 3)
        );
        assert!(
            store
                .inbox_store()
                .get(waiting.item_id)
                .await
                .expect("get")
                .is_none(),
            "no item has the old id"
        );
        let after = store
            .inbox_store()
            .get(joined.item_id)
            .await
            .expect("get")
            .expect("the joined fire");
        assert_eq!(
            after.origin,
            crate::store::inbox::InboxOrigin::Fire(crate::store::inbox::Fire {
                schedule_id: schedule.id.clone(),
                gate_output: Some("two".to_string()),
                occurrences: 3,
                due_at: first_fired,
                fired_at: second_fired,
            })
        );
        assert_eq!(
            (
                after.created_at,
                after.attempts,
                after.not_before,
                after.last_error.clone()
            ),
            (
                before.created_at,
                1,
                before.not_before,
                Some("{\"title\":\"down\"}".to_string())
            ),
            "it keeps its age and its retry state"
        );
        assert_eq!(
            store
                .inbox_store()
                .list_open(session_id)
                .await
                .expect("list")
                .len(),
            1,
            "one waiting fire per schedule"
        );
    }

    /// A fire the conversation already holds is history, so the next occurrence is a fire of its
    /// own rather than a change to one the model has read.
    #[tokio::test]
    async fn a_fire_already_in_the_conversation_is_not_joined() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let schedule = schedule_fixture(
            &store,
            session_id,
            crate::schedule::Timing::parse_every("1h").expect("parses"),
            None,
        )
        .await;
        let ClaimClosed::Yes(read) = fire(
            &store,
            &schedule,
            "host-a",
            fire_of(&schedule, schedule.next_fire_at),
        )
        .await
        else {
            panic!("the first fire is queued");
        };
        store
            .save_event_marking_inbox(
                session_id,
                &crate::conversation::Event::Append(crate::conversation::Message::user_turn(
                    "",
                    "the fire",
                    Vec::new(),
                )),
                &[read.item_id],
                None,
            )
            .await
            .expect("a turn reads it");

        let advanced = store
            .schedule_store()
            .list_schedules(session_id)
            .await
            .expect("list")
            .into_iter()
            .next()
            .expect("the schedule");
        let ClaimClosed::Yes(next) = fire(
            &store,
            &advanced,
            "host-a",
            fire_of(&advanced, advanced.next_fire_at),
        )
        .await
        else {
            panic!("the second fire is queued");
        };
        assert_eq!((next.replaces, next.occurrences), (None, 1));
        assert_ne!(next.item_id, read.item_id);
        let earlier = store
            .inbox_store()
            .get(read.item_id)
            .await
            .expect("get")
            .expect("the fire the turn read");
        assert_eq!(earlier.state(), crate::store::inbox::InboxState::Appended);
        let crate::store::inbox::InboxOrigin::Fire(facts) = earlier.origin else {
            panic!("a fire");
        };
        assert_eq!(
            facts.occurrences, 1,
            "what the model read is left as it read it"
        );
    }

    /// Fire the next occurrence of the session's one schedule, as a sweep does. None joins another
    /// when each is appended before the next.
    async fn fire_next(store: &Store, session_id: Uuid) -> crate::store::inbox::QueuedFire {
        let current = store
            .schedule_store()
            .list_schedules(session_id)
            .await
            .expect("list")
            .into_iter()
            .next()
            .expect("the schedule");
        let ClaimClosed::Yes(queued) = fire(
            store,
            &current,
            "host-a",
            fire_of(&current, current.next_fire_at),
        )
        .await
        else {
            panic!("the fire is queued");
        };
        queued
    }

    /// Put `item_id` in the conversation, as a turn that read it does.
    async fn append_fire(store: &Store, session_id: Uuid, item_id: Uuid) {
        store
            .save_event_marking_inbox(
                session_id,
                &crate::conversation::Event::Append(crate::conversation::Message::user_turn(
                    "",
                    "the fire",
                    Vec::new(),
                )),
                &[item_id],
                None,
            )
            .await
            .expect("a turn appends it");
    }

    /// Canceling a schedule withdraws every fire the model has not read, saying why: the one
    /// waiting, and the one a turn has appended that no provider has accepted yet, which the
    /// turn failing would otherwise offer again. The one a provider accepted is history and stays.
    #[tokio::test]
    async fn canceling_a_schedule_withdraws_its_waiting_fire_and_leaves_one_already_read() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let schedule = schedule_fixture(
            &store,
            session_id,
            crate::schedule::Timing::parse_every("1h").expect("parses"),
            None,
        )
        .await;
        let read = fire_next(&store, session_id).await;
        append_fire(&store, session_id, read.item_id).await;
        store
            .inbox_store()
            .mark_delivered(session_id)
            .await
            .expect("a provider accepts it");
        let appended = fire_next(&store, session_id).await;
        append_fire(&store, session_id, appended.item_id).await;
        let waiting = fire_next(&store, session_id).await;

        let canceled = store
            .schedule_store()
            .delete_schedule(&schedule.id)
            .await
            .expect("cancel")
            .expect("the schedule was there");
        assert_eq!(
            (canceled.id.as_str(), canceled.session_id),
            (schedule.id.as_str(), Some(session_id))
        );
        let mut withdrawn = canceled.withdrawn;
        withdrawn.sort();
        let mut unread = vec![appended.item_id, waiting.item_id];
        unread.sort();
        assert_eq!(
            withdrawn, unread,
            "the cancel answers the fires it withdrew, and not the one the model read"
        );
        for item_id in unread {
            let item = store
                .inbox_store()
                .get(item_id)
                .await
                .expect("get")
                .expect("the fire");
            assert_eq!(
                (item.state(), item.failure.as_deref()),
                (
                    crate::store::inbox::InboxState::Withdrawn,
                    Some("its schedule was canceled")
                )
            );
        }
        assert_eq!(
            store
                .inbox_store()
                .get(read.item_id)
                .await
                .expect("get")
                .expect("the fire the model read")
                .state(),
            crate::store::inbox::InboxState::Delivered,
            "a fire the model read stays"
        );
    }

    /// A fire already in the conversation when its schedule is canceled cannot be unsaid. It is
    /// withdrawn, so no failed turn or rewind offers it again; and when a provider accepts the
    /// request that carries it all the same, the model has read it and the item says so beside the
    /// withdrawal. One whose message left the log instead is in no request and is never delivered.
    #[tokio::test]
    async fn a_canceled_schedule_s_fire_in_an_accepted_request_is_delivered_all_the_same() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let schedule = schedule_fixture(
            &store,
            session_id,
            crate::schedule::Timing::parse_every("1h").expect("parses"),
            None,
        )
        .await;
        let taken_back = fire_next(&store, session_id).await.item_id;
        append_fire(&store, session_id, taken_back).await;
        let accepted = fire_next(&store, session_id).await.item_id;
        append_fire(&store, session_id, accepted).await;
        store
            .schedule_store()
            .delete_schedule(&schedule.id)
            .await
            .expect("cancel")
            .expect("the schedule was there");
        let inbox = store.inbox_store();

        // The first one's turn ends with nothing from the model, and its opening leaves the log.
        inbox
            .reset_pending(&[taken_back], chrono::Utc::now())
            .await
            .expect("reset");
        assert_eq!(
            inbox.mark_delivered(session_id).await.expect("deliver"),
            [accepted],
            "a provider accepted the request the other one was in"
        );
        let read = inbox.get(accepted).await.expect("get").expect("the fire");
        assert_eq!(
            (read.state(), read.withdrawn_at.is_some()),
            (crate::store::inbox::InboxState::Delivered, true),
            "the model read it, and it is still one no reset offers again"
        );
        let gone = inbox.get(taken_back).await.expect("get").expect("the fire");
        assert_eq!(
            (gone.state(), gone.appended_at, gone.delivered_at),
            (crate::store::inbox::InboxState::Withdrawn, None, None),
            "its text is in no request"
        );
        assert!(
            inbox
                .take_pending(session_id, &[], chrono::Utc::now())
                .await
                .expect("pending")
                .is_empty(),
            "and neither is offered again"
        );
    }

    /// A rewind that takes the message of a fire withdrawn in the conversation takes the fire out
    /// of the conversation with it, so no later request delivers what none carries.
    #[tokio::test]
    async fn a_rewind_takes_a_withdrawn_fire_out_of_the_conversation_with_its_message() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let schedule = schedule_fixture(
            &store,
            session_id,
            crate::schedule::Timing::parse_every("1h").expect("parses"),
            None,
        )
        .await;
        let rewound = fire_next(&store, session_id).await.item_id;
        append_fire(&store, session_id, rewound).await;
        store
            .schedule_store()
            .delete_schedule(&schedule.id)
            .await
            .expect("cancel")
            .expect("the schedule was there");
        store
            .save_rewind(session_id, &crate::conversation::Event::Repair {
                replaced_count: 1,
                messages: Vec::new(),
            })
            .await
            .expect("rewind");

        let inbox = store.inbox_store();
        let gone = inbox.get(rewound).await.expect("get").expect("the fire");
        assert_eq!(
            (gone.state(), gone.appended_at),
            (crate::store::inbox::InboxState::Withdrawn, None)
        );
        assert!(
            inbox
                .mark_delivered(session_id)
                .await
                .expect("deliver")
                .is_empty(),
            "the next request carries nothing of it"
        );
        assert!(
            inbox
                .take_pending(session_id, &[], chrono::Utc::now())
                .await
                .expect("pending")
                .is_empty(),
            "and it is not offered again"
        );
    }

    /// A one-shot leaves the list when its fire is queued, and the fire is the delivery it was
    /// spent on: nothing withdraws it.
    #[tokio::test]
    async fn a_one_shot_retired_by_its_fire_leaves_the_fire_to_be_delivered() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let due_at = chrono::Utc::now() - chrono::Duration::minutes(1);
        let queued = store
            .schedule_store()
            .queue_fire_for_test(session_id, "stretch", crate::store::inbox::Fire {
                schedule_id: Uuid::new_v4().to_string(),
                gate_output: None,
                occurrences: 1,
                due_at,
                fired_at: chrono::Utc::now(),
            })
            .await
            .expect("queue the fire");
        assert!(
            store
                .schedule_store()
                .list_all_schedules()
                .await
                .expect("list")
                .is_empty(),
            "the one-shot is retired"
        );
        assert_eq!(
            store
                .inbox_store()
                .get(queued.item_id)
                .await
                .expect("get")
                .expect("its fire")
                .state(),
            crate::store::inbox::InboxState::Pending
        );
    }

    #[tokio::test]
    async fn cancel_resolves_an_id_prefix_and_reports_ambiguity() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let schedule = schedule_fixture(
            &store,
            session_id,
            crate::schedule::Timing::parse_every("30m").expect("parses"),
            None,
        )
        .await;

        assert!(
            store
                .schedule_store()
                .cancel_schedule(session_id, "nomatch")
                .await
                .expect("cancel runs")
                .is_none()
        );

        let canceled = store
            .schedule_store()
            .cancel_schedule(session_id, schedule.short_id())
            .await
            .expect("cancel runs")
            .expect("prefix matched");
        assert_eq!(canceled.id, schedule.id);
        assert!(
            store
                .schedule_store()
                .list_schedules(session_id)
                .await
                .expect("list schedules")
                .is_empty()
        );
    }

    /// A half-written gate is corruption, not "no gate": treating it as an ungated schedule would
    /// silently promote a watcher into an unconditional timer that fires every interval.
    #[tokio::test]
    async fn a_row_with_a_half_written_gate_is_skipped_not_downgraded() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let schedule = schedule_fixture(
            &store,
            session_id,
            crate::schedule::Timing::parse_every("30m").expect("parses"),
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
                .list_schedules(session_id)
                .await
                .expect("list schedules")
                .len(),
            1
        );

        let id = schedule.id.clone();
        store
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE schedules SET gate_spec_json = NULL WHERE id = ?1",
                    rusqlite::params![id],
                )?;
                Ok(())
            })
            .await
            .expect("corrupt the row");

        assert!(
            store
                .schedule_store()
                .list_schedules(session_id)
                .await
                .expect("list schedules")
                .is_empty(),
            "the corrupt row is skipped rather than read as ungated"
        );
    }
}
