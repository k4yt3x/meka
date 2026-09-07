//! Background tasks: the `background_tasks` table.

use chrono::{DateTime, Utc};

use super::*;

/// Background work's slice of the session database, handed out by
/// [`crate::store::Store::background_store`].
#[derive(Clone)]
pub(crate) struct BackgroundStore {
    pub(crate) connection: std::sync::Arc<tokio_rusqlite::Connection>,
}
impl BackgroundStore {
    pub(crate) fn new(connection: std::sync::Arc<tokio_rusqlite::Connection>) -> Self {
        Self { connection }
    }

    /// Record a task as started. Written before the work is spawned, so a process that dies between
    /// the two leaves a `running` row the sweep will retire rather than a task nobody knows about.
    pub(crate) async fn start_background_task(
        &self,
        task: &BackgroundTask,
    ) -> crate::error::Result<()> {
        let id = task.id.clone();
        let session_id = task.session_id.to_string();
        let tool_name = task.tool_name.clone();
        let label = task.label.clone();
        let status = task.status.name().to_string();
        let started_at = task.started_at.to_rfc3339();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO background_tasks \
                     (id, session_id, tool_name, label, status, started_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    rusqlite::params![id, session_id, tool_name, label, status, started_at],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to start background task: {error}"))
            })
    }

    /// Record a terminal outcome.
    ///
    /// Guarded on `status = 'running'` so a task that was canceled, or swept to `interrupted`,
    /// cannot be overwritten by its own work finishing a moment later. The first terminal write
    /// wins, which is what keeps a canceled task from reporting success.
    pub(crate) async fn finish_background_task(
        &self,
        id: &str,
        status: TaskStatus,
        outcome: Option<String>,
        scratchpad_name: Option<String>,
    ) -> crate::error::Result<()> {
        let id = id.to_string();
        let status = status.name().to_string();
        let finished_at = chrono::Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE background_tasks \
                     SET status = ?2, outcome = ?3, scratchpad_name = ?4, finished_at = ?5 \
                     WHERE id = ?1 AND status = 'running'",
                    rusqlite::params![id, status, outcome, scratchpad_name, finished_at],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to finish background task: {error}"))
            })
    }

    /// Every task belonging to one session, newest first.
    pub(crate) async fn list_background_tasks(
        &self,
        session_id: Uuid,
    ) -> crate::error::Result<Vec<BackgroundTask>> {
        self.query_background_tasks(
            "SELECT id, session_id, tool_name, label, status, outcome, scratchpad_name, \
             started_at, finished_at, announced_at, delivered_at FROM background_tasks \
             WHERE session_id = ?1 ORDER BY started_at DESC",
            vec![session_id.to_string()],
        )
        .await
    }

    /// A session's tasks still running, oldest first. Backs the `[Background]` index and the Ctrl+C
    /// survivor line.
    pub(crate) async fn list_running_background_tasks(
        &self,
        session_id: Uuid,
    ) -> crate::error::Result<Vec<BackgroundTask>> {
        self.query_background_tasks(
            "SELECT id, session_id, tool_name, label, status, outcome, scratchpad_name, \
             started_at, finished_at, announced_at, delivered_at FROM background_tasks \
             WHERE session_id = ?1 AND status = 'running' ORDER BY started_at ASC",
            vec![session_id.to_string()],
        )
        .await
    }

    /// A session's finished-but-unreported tasks, oldest first. The delivery poll's query; served
    /// by `idx_background_tasks_session_status`.
    ///
    /// This and [`Self::mark_background_tasks_delivered`] are two statements, not one transaction,
    /// so two processes that both list before either marks would each render the same outcome. That
    /// is currently unreachable -- delivery only happens inside a session, and a session is held by
    /// one process at a time from the moment its row exists -- so the pair rests on the session
    /// lock rather than on its own atomicity. Anything that ever lets two hosts open one
    /// session at once has to make this a transaction first.
    pub(crate) async fn list_undelivered_background_tasks(
        &self,
        session_id: Uuid,
    ) -> crate::error::Result<Vec<BackgroundTask>> {
        self.query_background_tasks(
            "SELECT id, session_id, tool_name, label, status, outcome, scratchpad_name, \
             started_at, finished_at, announced_at, delivered_at FROM background_tasks \
             WHERE session_id = ?1 AND status != 'running' AND delivered_at IS NULL \
             ORDER BY finished_at ASC",
            vec![session_id.to_string()],
        )
        .await
    }

    /// Stamp outcomes as delivered.
    ///
    /// Called *before* the turn runs: an outcome that reliably wedges the process would otherwise
    /// be redelivered on every restart, turning one bad result into a boot loop. Losing one report
    /// is the cheaper failure.
    ///
    ///
    /// [`crate::store::schedule::ScheduleStore::complete_claim`] writes *after* the turn, because a
    /// lease plus an attempt ceiling gives it the same boot-loop protection without paying an
    /// occurrence for every crash. A background outcome has no lease to hold, so it keeps the
    /// cruder rule.
    pub(crate) async fn mark_background_tasks_delivered(
        &self,
        ids: &[String],
    ) -> crate::error::Result<Vec<String>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let ids = ids.to_vec();
        let delivered_at = chrono::Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                // One transaction, so a failure part-way cannot leave half a batch stamped. The
                // caller renders every one of these into a single turn, and stamping only some of
                // them would repeat the rest alongside it on the next tick.
                let transaction = connection.transaction()?;
                let mut claimed = Vec::new();
                {
                    let mut statement = transaction.prepare(
                        "UPDATE background_tasks SET delivered_at = ?2 \
                         WHERE id = ?1 AND delivered_at IS NULL",
                    )?;
                    for id in &ids {
                        if statement.execute(rusqlite::params![id, delivered_at])? == 1 {
                            claimed.push(id.clone());
                        }
                    }
                }
                transaction.commit()?;
                Ok(claimed)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to mark tasks delivered: {error}"))
            })
    }

    /// A session's finished-but-unannounced tasks, oldest first.
    ///
    /// The poller's own question, and deliberately not [`Self::list_undelivered_background_tasks`]:
    /// telling subscribers a task finished needs nothing from a live session, while telling the
    /// model needs a turn. An outcome that waits for one must stay undelivered without also being
    /// re-announced on every poll.
    ///
    /// Bounded to the undelivered pool, which is what keeps it news. `announced_at` is written only
    /// by `meka serve`, so every task a REPL or ACP session ever ran is unannounced forever in a
    /// shared store; without this clause, opening one of those sessions in `meka serve` would fire
    /// `task.finished` for work that finished weeks ago and was reported at the time. A task still
    /// in the pool has been reported to nobody, which is the case worth pushing.
    pub(crate) async fn list_unannounced_background_tasks(
        &self,
        session_id: Uuid,
    ) -> crate::error::Result<Vec<BackgroundTask>> {
        self.query_background_tasks(
            "SELECT id, session_id, tool_name, label, status, outcome, scratchpad_name, \
             started_at, finished_at, announced_at, delivered_at FROM background_tasks \
             WHERE session_id = ?1 AND status != 'running' AND announced_at IS NULL \
             AND delivered_at IS NULL ORDER BY finished_at ASC",
            vec![session_id.to_string()],
        )
        .await
    }

    /// Stamp a batch announced. One transaction, for the reason
    /// [`Self::mark_background_tasks_delivered`] gives.
    pub(crate) async fn mark_background_tasks_announced(
        &self,
        ids: &[String],
    ) -> crate::error::Result<Vec<String>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let ids = ids.to_vec();
        let announced_at = chrono::Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let transaction = connection.transaction()?;
                let mut claimed = Vec::new();
                {
                    let mut statement = transaction.prepare(
                        "UPDATE background_tasks SET announced_at = ?2 \
                         WHERE id = ?1 AND announced_at IS NULL",
                    )?;
                    for id in &ids {
                        if statement.execute(rusqlite::params![id, announced_at])? == 1 {
                            claimed.push(id.clone());
                        }
                    }
                }
                transaction.commit()?;
                Ok(claimed)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to mark tasks announced: {error}"))
            })
    }

    /// Retire every `running` task in this session as [`TaskStatus::Interrupted`], returning how
    /// many were swept.
    ///
    /// Called when a process takes ownership of a session. The session lock
    /// ([`crate::store::Store::lock_session`]) is the lease: holding it means no other
    /// process can still be running this session's tasks, so any row that still says `running`
    /// belongs to a process that is gone. Without this a task in flight at shutdown would leave the
    /// agent waiting on a report that can never arrive, having very likely already promised one.
    pub(crate) async fn sweep_interrupted_background_tasks(
        &self,
        session_id: Uuid,
    ) -> crate::error::Result<usize> {
        let session_id = session_id.to_string();
        let status = TaskStatus::Interrupted.name().to_string();
        let finished_at = chrono::Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let swept = connection.execute(
                    "UPDATE background_tasks SET status = ?2, finished_at = ?3 \
                     WHERE session_id = ?1 AND status = 'running'",
                    rusqlite::params![session_id, status, finished_at],
                )?;
                Ok(swept)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to sweep background tasks: {error}"))
            })
    }

    /// Resolve a full or unique-prefix task id within a session. An ambiguous prefix is an error
    /// rather than an arbitrary pick, matching
    /// [`crate::store::schedule::ScheduleStore::cancel_scheduled_job`] in behavior and in error
    /// variant. That one returned `Database` for the identical condition, so the two `serve`
    /// endpoints answered different HTTP statuses for the same mistake; both now use `Config`,
    /// which reaches the caller as a 422.
    pub(crate) async fn resolve_background_task(
        &self,
        session_id: Uuid,
        id_prefix: &str,
    ) -> crate::error::Result<Option<BackgroundTask>> {
        if !crate::text::is_usable_id_prefix(id_prefix) {
            return Ok(None);
        }
        let wanted = crate::text::id_prefix_for_matching(id_prefix);
        let tasks = self.list_background_tasks(session_id).await?;
        let matches: Vec<BackgroundTask> = tasks
            .into_iter()
            .filter(|task| task.id.starts_with(&wanted))
            .collect();
        match matches.len() {
            0 => Ok(None),
            1 => Ok(matches.into_iter().next()),
            _ => Err(MekaError::Config(format!(
                "task id '{}' is ambiguous; it matches {} tasks",
                id_prefix,
                matches.len()
            ))),
        }
    }

    /// Shared row decoder, mirroring `ScheduleStore::query_scheduled_jobs`: one unreadable row is
    /// skipped with a warning rather than failing every other task in the query.
    pub(crate) async fn query_background_tasks(
        &self,
        sql: &'static str,
        params: Vec<String>,
    ) -> crate::error::Result<Vec<BackgroundTask>> {
        let rows: Vec<BackgroundTaskRow> = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(sql)?;
                let rows = statement
                    .query_map(rusqlite::params_from_iter(params.iter()), |row| {
                        Ok(BackgroundTaskRow {
                            id: row.get(0)?,
                            session_id: row.get(1)?,
                            tool_name: row.get(2)?,
                            label: row.get(3)?,
                            status: row.get(4)?,
                            outcome: row.get(5)?,
                            scratchpad_name: row.get(6)?,
                            started_at: row.get(7)?,
                            finished_at: row.get(8)?,
                            announced_at: row.get(9)?,
                            delivered_at: row.get(10)?,
                        })
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to load background tasks: {error}"))
            })?;

        Ok(rows
            .into_iter()
            .filter_map(|row| {
                let id = row.id.clone();
                row.decode()
                    .inspect_err(|error| {
                        tracing::warn!("skipping unreadable background task {id}: {error}");
                    })
                    .ok()
            })
            .collect())
    }
}
/// Raw `background_tasks` row, decoded outside the database closure so a parse failure can be
/// logged and skipped individually.
pub(crate) struct BackgroundTaskRow {
    pub(crate) id: String,
    pub(crate) session_id: String,
    pub(crate) tool_name: String,
    pub(crate) label: String,
    pub(crate) status: String,
    pub(crate) outcome: Option<String>,
    pub(crate) scratchpad_name: Option<String>,
    pub(crate) started_at: String,
    pub(crate) finished_at: Option<String>,
    pub(crate) announced_at: Option<String>,
    pub(crate) delivered_at: Option<String>,
}
impl BackgroundTaskRow {
    pub(crate) fn decode(self) -> std::result::Result<BackgroundTask, String> {
        let parse_time =
            |text: &str| -> std::result::Result<chrono::DateTime<chrono::Utc>, String> {
                chrono::DateTime::parse_from_rfc3339(text)
                    .map(|at| at.with_timezone(&chrono::Utc))
                    .map_err(|error| format!("bad timestamp '{text}': {error}"))
            };
        let parse_optional = |text: Option<String>| -> std::result::Result<_, String> {
            text.as_deref().map(parse_time).transpose()
        };

        Ok(BackgroundTask {
            id: self.id,
            session_id: Uuid::parse_str(&self.session_id)
                .map_err(|error| format!("bad session id: {error}"))?,
            tool_name: self.tool_name,
            label: self.label,
            status: self.status.parse()?,
            outcome: self.outcome,
            scratchpad_name: self.scratchpad_name,
            started_at: parse_time(&self.started_at)?,
            finished_at: parse_optional(self.finished_at)?,
            announced_at: parse_optional(self.announced_at)?,
            delivered_at: parse_optional(self.delivered_at)?,
        })
    }
}
/// How a task ended, or that it hasn't.
///
/// The four terminal states deliver identically and differ only in their rendered header. Keeping
/// them distinct is what lets the agent tell "your build failed" from "your build never ran".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TaskStatus {
    Running,
    Completed,
    /// The tool returned an error, or panicked.
    Failed,
    /// Stopped on request, via `task_cancel` or a second Ctrl+C.
    Cancelled,
    /// The process holding it went away. Reconstructed by the session-load sweep, never written by
    /// the task itself, which by definition is not around to write it.
    Interrupted,
}
impl TaskStatus {
    /// Every status, in lifecycle order.
    pub(crate) const ALL: [TaskStatus; 5] = [
        Self::Running,
        Self::Completed,
        Self::Failed,
        Self::Cancelled,
        Self::Interrupted,
    ];

    /// Whether a finished task should wake a host, or wait for the next turn to carry it.
    ///
    /// Every terminal outcome is delivered; this decides only whether delivering it is worth a turn
    /// nobody asked for. A cancellation is always somebody's deliberate act -- `/tasks cancel`, the
    /// `task_cancel` tool, a second Ctrl+C, `POST .../cancel` -- so the one party who would learn
    /// something from the turn already knows, and a command whose whole purpose is to stop work
    /// would be starting some. The outcome still reaches the model, on the next turn there is.
    ///
    /// The others are nobody's decision: a build finished, a tool failed, or a host died holding
    /// the task. The agent asked to be told about the first two and cannot infer the third, and
    /// there may be no human about to type.
    pub(crate) fn wakes_a_host(self) -> bool {
        !matches!(self, Self::Cancelled)
    }

    /// The one spelling of this status: the `status` column, the HTTP view and `task_list`.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Interrupted => "interrupted",
        }
    }

    /// The names, joined for a refusal that lists what would have been accepted.
    pub(crate) fn supported() -> String {
        Self::ALL
            .iter()
            .map(|status| status.name())
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub(crate) fn is_terminal(self) -> bool {
        !matches!(self, Self::Running)
    }

    /// The header word the delivered turn leads with.
    pub(crate) fn headline(self) -> &'static str {
        match self {
            Self::Running => "is still running",
            Self::Completed => "finished",
            Self::Failed => "failed",
            Self::Cancelled => "was canceled",
            Self::Interrupted => "was interrupted",
        }
    }
}
impl std::fmt::Display for TaskStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}
impl std::str::FromStr for TaskStatus {
    type Err = String;

    /// Refuses with the names that would have been accepted; `decode` turns the refusal into a
    /// skipped row rather than a hard error, matching how the session store treats a
    /// forward-compatible event it can't read.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|status| status.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not a task status. Supported: {}",
                    Self::supported()
                )
            })
    }
}
/// A persisted background task.
#[derive(Debug, Clone)]
pub(crate) struct BackgroundTask {
    pub(crate) id: String,
    pub(crate) session_id: Uuid,
    /// The tool that was backgrounded, e.g. `execute_command`.
    pub(crate) tool_name: String,
    /// Human-readable summary of what was started, from
    /// [`crate::tools::resolve_primary_param`]. Carried so `task_list` and the delivered turn can
    /// name the work without re-deriving it from arguments that are no longer around.
    pub(crate) label: String,
    pub(crate) status: TaskStatus,
    /// The tool's own output, for a terminal task. Truncated to
    /// [`crate::background::OUTCOME_INLINE_LIMIT`] when it is also spilled to the scratchpad.
    pub(crate) outcome: Option<String>,
    /// Scratchpad entry holding the full output, when it was too large to carry inline.
    pub(crate) scratchpad_name: Option<String>,
    pub(crate) started_at: DateTime<Utc>,
    pub(crate) finished_at: Option<DateTime<Utc>>,
    /// When subscribers were told, which is the poller's job and happens whether or not a session
    /// is ever live again. Separate from `delivered_at` because the two stopped coinciding once an
    /// outcome could wait for a turn instead of causing one.
    pub(crate) announced_at: Option<DateTime<Utc>>,
    pub(crate) delivered_at: Option<DateTime<Utc>>,
}
impl BackgroundTask {
    /// Short id for display, matching the width `task_cancel` accepts. Same convention as
    /// [`crate::schedule::ScheduledJob::short_id`].
    pub(crate) fn short_id(&self) -> &str {
        self.id.get(..crate::text::ID_PREFIX).unwrap_or(&self.id)
    }

    /// How long the task ran, or has been running.
    pub(crate) fn elapsed(&self) -> chrono::Duration {
        self.finished_at.unwrap_or_else(Utc::now) - self.started_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn task_fixture(
        store: &Store,
        session_id: Uuid,
        label: &str,
    ) -> crate::store::background::BackgroundTask {
        let task = crate::store::background::BackgroundTask {
            id: Uuid::new_v4().to_string(),
            session_id,
            tool_name: "execute_command".to_string(),
            label: label.to_string(),
            status: crate::store::background::TaskStatus::Running,
            outcome: None,
            scratchpad_name: None,
            started_at: chrono::Utc::now(),
            finished_at: None,
            announced_at: None,
            delivered_at: None,
        };
        store
            .background_store()
            .start_background_task(&task)
            .await
            .expect("start background task");
        task
    }

    #[tokio::test]
    async fn background_task_round_trips_through_the_database() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let written = task_fixture(&store, session_id, "cargo test --all").await;

        let running = store
            .background_store()
            .list_running_background_tasks(session_id)
            .await
            .expect("list running");
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].id, written.id);
        assert_eq!(running[0].label, "cargo test --all");

        store
            .background_store()
            .finish_background_task(
                &written.id,
                crate::store::background::TaskStatus::Completed,
                Some("42 passed".to_string()),
                Some("task_log".to_string()),
            )
            .await
            .expect("finish");

        let undelivered = store
            .background_store()
            .list_undelivered_background_tasks(session_id)
            .await
            .expect("list undelivered");
        assert_eq!(undelivered.len(), 1);
        assert_eq!(
            undelivered[0].status,
            crate::store::background::TaskStatus::Completed
        );
        assert_eq!(undelivered[0].outcome.as_deref(), Some("42 passed"));
        assert_eq!(undelivered[0].scratchpad_name.as_deref(), Some("task_log"));
        assert!(undelivered[0].finished_at.is_some());
        assert!(
            store
                .background_store()
                .list_running_background_tasks(session_id)
                .await
                .expect("list running")
                .is_empty()
        );
    }

    /// The whole point of `delivered_at`: an outcome reaches the conversation once, including
    /// across a restart that re-runs the delivery poll.
    #[tokio::test]
    async fn a_delivered_outcome_is_not_delivered_again() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let task = task_fixture(&store, session_id, "sleep 60").await;
        store
            .background_store()
            .finish_background_task(
                &task.id,
                crate::store::background::TaskStatus::Completed,
                None,
                None,
            )
            .await
            .expect("finish");

        store
            .background_store()
            .mark_background_tasks_delivered(std::slice::from_ref(&task.id))
            .await
            .expect("mark delivered");

        assert!(
            store
                .background_store()
                .list_undelivered_background_tasks(session_id)
                .await
                .expect("list undelivered")
                .is_empty()
        );
    }

    /// Two claimers, one row, exactly one winner.
    ///
    /// Listing and stamping are two statements, and there are now two claimers per host: a poller
    /// and whichever turn the user sends. Both can read the same row as undelivered before either
    /// writes, so the `WHERE delivered_at IS NULL` on the stamp is the only arbiter -- without it
    /// both render the same outcome and the model is told twice.
    ///
    /// Two managers over one file on disk, not two clones of one store: a clone shares an
    /// `Arc<Connection>` and therefore one worker thread, so the two claims cannot interleave and
    /// the test proves only that the statement is atomic against itself. Separate connections are
    /// what a second meka process actually is.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn only_one_of_two_racing_claimers_takes_each_outcome() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("meka.db");
        let store = Store::open(Some(&path), &Default::default())
            .await
            .expect("open");
        let rival = Store::open(Some(&path), &Default::default())
            .await
            .expect("a second connection to the same file");
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        const ROWS: usize = 24;
        for index in 0..ROWS {
            let task = task_fixture(&store, session_id, &format!("job {index}")).await;
            store
                .background_store()
                .finish_background_task(
                    &task.id,
                    crate::store::background::TaskStatus::Completed,
                    None,
                    None,
                )
                .await
                .expect("finish");
        }

        let first = {
            let store = store.clone();
            tokio::spawn(async move { crate::host::claim_outcomes_now(&store, session_id).await })
        };
        let second = {
            let store = rival;
            tokio::spawn(async move { crate::host::claim_outcomes_now(&store, session_id).await })
        };
        let (first, second) = (first.await.expect("join"), second.await.expect("join"));

        let mut seen: Vec<String> = first
            .iter()
            .chain(second.iter())
            .map(|task| task.id.clone())
            .collect();
        let total = seen.len();
        seen.sort();
        seen.dedup();
        assert_eq!(
            total, ROWS,
            "every outcome must be claimed exactly once across both claimers, not {total}"
        );
        assert_eq!(
            seen.len(),
            ROWS,
            "and no id may appear in both claims: that is one outcome delivered twice"
        );
        assert!(
            store
                .background_store()
                .list_undelivered_background_tasks(session_id)
                .await
                .expect("list undelivered")
                .is_empty(),
            "and nothing may be left behind"
        );
    }

    /// The same arbitration for the announce stamp, which decides who fires `task.finished`.
    ///
    /// Two connections for the reason its sibling gives: one store cloned is one worker thread.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn only_one_of_two_racing_announcers_takes_each_outcome() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("meka.db");
        let store = Store::open(Some(&path), &Default::default())
            .await
            .expect("open");
        let rival = Store::open(Some(&path), &Default::default())
            .await
            .expect("a second connection to the same file");
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        const ROWS: usize = 24;
        let mut ids = Vec::new();
        for index in 0..ROWS {
            let task = task_fixture(&store, session_id, &format!("job {index}")).await;
            store
                .background_store()
                .finish_background_task(
                    &task.id,
                    crate::store::background::TaskStatus::Cancelled,
                    None,
                    None,
                )
                .await
                .expect("finish");
            ids.push(task.id);
        }

        // Both read the row as unannounced first, which is what the two callers really do.
        let first = {
            let (store, ids) = (store.clone(), ids.clone());
            tokio::spawn(async move {
                store
                    .background_store()
                    .mark_background_tasks_announced(&ids)
                    .await
                    .expect("announce")
            })
        };
        let second = {
            let (store, ids) = (rival, ids.clone());
            tokio::spawn(async move {
                store
                    .background_store()
                    .mark_background_tasks_announced(&ids)
                    .await
                    .expect("announce")
            })
        };
        let (first, second) = (first.await.expect("join"), second.await.expect("join"));

        assert_eq!(
            first.len() + second.len(),
            ROWS,
            "a subscriber must hear about each task once: {} + {}",
            first.len(),
            second.len()
        );
        let mut seen: Vec<&String> = first.iter().chain(second.iter()).collect();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), ROWS, "and no id may be won twice");
    }

    /// Each stamp is won once, and the loser is told it won nothing.
    ///
    /// The stamps are compare-and-swaps and they return the rows they took, which is what every
    /// caller filters its report by. A second attempt on an already-stamped row must come back
    /// empty rather than claiming it again -- that empty answer is what stops a second webhook and
    /// a second delivery. The concurrent case is
    /// [`only_one_of_two_racing_claimers_takes_each_outcome`]; this is the sequential contract the
    /// callers read.
    #[tokio::test]
    async fn a_second_attempt_on_a_stamped_row_wins_nothing() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let task = task_fixture(&store, session_id, "cargo build").await;
        store
            .background_store()
            .finish_background_task(
                &task.id,
                crate::store::background::TaskStatus::Completed,
                None,
                None,
            )
            .await
            .expect("finish");
        let ids = vec![task.id.clone()];

        assert_eq!(
            store
                .background_store()
                .mark_background_tasks_delivered(&ids)
                .await
                .expect("deliver"),
            ids,
            "the first claimer wins the row"
        );
        assert!(
            store
                .background_store()
                .mark_background_tasks_delivered(&ids)
                .await
                .expect("deliver")
                .is_empty(),
            "and the second wins nothing, so it reports nothing"
        );

        assert_eq!(
            store
                .background_store()
                .mark_background_tasks_announced(&ids)
                .await
                .expect("announce"),
            ids,
            "the first announcer wins the row"
        );
        assert!(
            store
                .background_store()
                .mark_background_tasks_announced(&ids)
                .await
                .expect("announce")
                .is_empty(),
            "and the second sends no second `task.finished`"
        );
    }

    /// A task that goes terminal between the poller's two queries is still announced.
    ///
    /// The sweep asks `list_unannounced_background_tasks` and then, separately,
    /// `list_undelivered_background_tasks`. A task finishing in the gap is absent from the first
    /// and present in the second, so it was delivered and then unannounceable forever -- both
    /// pools exclude a delivered row. Announcing the claimed batch as well closes it, and is
    /// idempotent because the stamp is a compare-and-swap.
    #[tokio::test]
    async fn a_task_finishing_between_the_two_queries_is_still_announced() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");

        // The announce pass runs first and sees nothing: the task is still running.
        let early = store
            .background_store()
            .list_unannounced_background_tasks(session_id)
            .await
            .expect("list unannounced");
        let task = task_fixture(&store, session_id, "cargo build").await;
        assert!(
            early.iter().all(|seen| seen.id != task.id),
            "the task must not be in the earlier snapshot"
        );

        // It finishes in the gap, so the delivery pass is the first to see it.
        store
            .background_store()
            .finish_background_task(
                &task.id,
                crate::store::background::TaskStatus::Completed,
                None,
                None,
            )
            .await
            .expect("finish");
        let claimed = crate::host::claim_outcomes_now(&store, session_id).await;
        assert_eq!(claimed.len(), 1, "the delivery pass claims it");
        assert!(
            claimed[0].announced_at.is_none(),
            "and it is still unannounced at the point the caller must act on"
        );

        // Which is why the claimed batch is announced too: nothing else ever can.
        let ids: Vec<String> = claimed.iter().map(|task| task.id.clone()).collect();
        let won = store
            .background_store()
            .mark_background_tasks_announced(&ids)
            .await
            .expect("announce");
        assert_eq!(won, ids, "the claim is the last chance to announce it");
        assert!(
            store
                .background_store()
                .list_unannounced_background_tasks(session_id)
                .await
                .expect("list unannounced")
                .is_empty(),
            "and afterwards no pool can return it: a delivered row is excluded from both"
        );
    }

    /// The announce pool is the undelivered pool, so old news is never pushed as new.
    ///
    /// `announced_at` is written only by `meka serve`, and a data directory is shared: every task a
    /// REPL or ACP session ever ran sits unannounced forever. Opening one of those sessions in
    /// `meka serve` must not fire `task.finished` for work that finished weeks ago and was reported
    /// to the model at the time. A task still undelivered has been reported to nobody, which is the
    /// case worth pushing -- including one interrupted before any host could report it.
    #[tokio::test]
    async fn an_outcome_already_reported_is_not_announced_as_news() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let reported = task_fixture(&store, session_id, "cargo build").await;
        let unreported = task_fixture(&store, session_id, "sleep 600").await;
        for task in [&reported, &unreported] {
            store
                .background_store()
                .finish_background_task(
                    &task.id,
                    crate::store::background::TaskStatus::Completed,
                    None,
                    None,
                )
                .await
                .expect("finish");
        }

        // Delivered without being announced, which is exactly what a REPL turn leaves behind.
        store
            .background_store()
            .mark_background_tasks_delivered(std::slice::from_ref(&reported.id))
            .await
            .expect("mark delivered");

        let unannounced = store
            .background_store()
            .list_unannounced_background_tasks(session_id)
            .await
            .expect("list unannounced");
        assert_eq!(
            unannounced.iter().map(|task| &task.id).collect::<Vec<_>>(),
            vec![&unreported.id],
            "only the outcome nobody has had is news"
        );

        store
            .background_store()
            .mark_background_tasks_announced(std::slice::from_ref(&unreported.id))
            .await
            .expect("mark announced");
        assert!(
            store
                .background_store()
                .list_unannounced_background_tasks(session_id)
                .await
                .expect("list unannounced")
                .is_empty(),
            "and it is announced once, not on every poll"
        );
    }

    /// A task in flight when the process died would otherwise leave the agent waiting on a report
    /// that can never arrive, having usually already promised one.
    #[tokio::test]
    async fn the_sweep_retires_tasks_left_running_by_a_dead_process() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        task_fixture(&store, session_id, "sleep 600").await;

        let swept = store
            .background_store()
            .sweep_interrupted_background_tasks(session_id)
            .await
            .expect("sweep");
        assert_eq!(swept, 1);

        let undelivered = store
            .background_store()
            .list_undelivered_background_tasks(session_id)
            .await
            .expect("list undelivered");
        assert_eq!(undelivered.len(), 1);
        assert_eq!(
            undelivered[0].status,
            crate::store::background::TaskStatus::Interrupted,
            "the agent must be told the work stopped, not left waiting"
        );

        // Idempotent: a second attach must not re-report what the first already retired.
        assert_eq!(
            store
                .background_store()
                .sweep_interrupted_background_tasks(session_id)
                .await
                .expect("second sweep"),
            0
        );
    }

    /// The shape a `--oneshot` resume hits: the previous process died mid-task, so the sweep
    /// produces an outcome that no task in *this* process is waiting on. A host that only looked
    /// for outcomes when it had started something itself would answer the prompt and exit with that
    /// report still sitting undelivered.
    #[tokio::test]
    async fn a_swept_outcome_is_pending_for_a_process_that_started_nothing() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        task_fixture(&store, session_id, "sleep 400").await;

        // A fresh process takes the session: it holds no handles, and the sweep is all it knows.
        store
            .background_store()
            .sweep_interrupted_background_tasks(session_id)
            .await
            .expect("sweep");

        let ready = store
            .background_store()
            .list_undelivered_background_tasks(session_id)
            .await
            .expect("list undelivered");
        assert_eq!(ready.len(), 1, "the report must be waiting to be collected");
        assert_eq!(
            ready[0].status,
            crate::store::background::TaskStatus::Interrupted
        );
    }

    /// A canceled task whose work happens to finish a moment later must not overwrite the
    /// cancellation and report success.
    #[tokio::test]
    async fn the_first_terminal_write_wins() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let task = task_fixture(&store, session_id, "sleep 600").await;

        store
            .background_store()
            .finish_background_task(
                &task.id,
                crate::store::background::TaskStatus::Cancelled,
                None,
                None,
            )
            .await
            .expect("cancel");
        store
            .background_store()
            .finish_background_task(
                &task.id,
                crate::store::background::TaskStatus::Completed,
                Some("finished anyway".to_string()),
                None,
            )
            .await
            .expect("late completion is accepted but ignored");

        let undelivered = store
            .background_store()
            .list_undelivered_background_tasks(session_id)
            .await
            .expect("list undelivered");
        assert_eq!(
            undelivered[0].status,
            crate::store::background::TaskStatus::Cancelled
        );
        assert!(undelivered[0].outcome.is_none());
    }

    #[tokio::test]
    async fn resolve_background_task_by_prefix() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let task = task_fixture(&store, session_id, "sleep 60").await;

        let found = store
            .background_store()
            .resolve_background_task(session_id, &task.id[..8])
            .await
            .expect("resolve")
            .expect("matched");
        assert_eq!(found.id, task.id);

        assert!(
            store
                .background_store()
                .resolve_background_task(session_id, "zzzzzzzz")
                .await
                .expect("resolve")
                .is_none()
        );
    }

    #[tokio::test]
    async fn deleting_a_session_cascades_to_its_background_tasks() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        task_fixture(&store, session_id, "sleep 600").await;

        store
            .delete_session(session_id)
            .await
            .expect("delete session");

        assert!(
            store
                .background_store()
                .list_background_tasks(session_id)
                .await
                .expect("list")
                .is_empty()
        );
    }
}
