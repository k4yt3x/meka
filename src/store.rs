//! SQLite-backed session store. The tables this module owns are `sessions` and `messages` for the
//! conversation, `tool_outputs` for results too large to keep inline (referenced from the
//! conversation by handle), `account_credentials` and `mcp_credentials` for secrets, and
//! `scheduled_jobs` and `background_tasks` for work the agent starts and does not wait for. They
//! are not the whole database, and this module does not define them: they and the memory store's
//! tables are created by [`migrations`], the single ledger that also brings an older store forward.
//! `initialize_schema` runs it, then hands the memory search index to
//! `crate::store::memory::reconcile_index`.
//!
//! `prompt_history` is in the ledger like every other table, but it is read and written on a
//! separate synchronous connection ([`history::HistoryStore`]) because the line editor that
//! consumes it is synchronous.
//!
//! Per-session mutual exclusion is provided by an OS-level file lock ([`FileLock`]) so the
//! kernel reclaims it whenever the holder dies: no PID-aliveness check, no risk of stale locks.
//!
//! On Unix the data directory (`0700`), lock directory (`0700`), and the database file itself
//! (`0600`) are tightened after creation so the persisted OAuth tokens, MCP credentials, and
//! conversation content aren't readable by other local users regardless of the user's umask.

use crate::fs::*;

pub(crate) mod background;
mod backup;
mod blobs;
mod credentials;
pub(crate) mod export;
pub(crate) mod history;
mod locks;
pub(crate) mod memory;
pub(crate) mod migrations;
pub(crate) mod schedule;
mod scratchpad;
mod sessions;

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use tokio_rusqlite::Connection;
use uuid::Uuid;

use self::backup::*;
// The lock vocabulary is the store's own; only the list below leaves the module.
use self::locks::*;
pub(crate) use self::{
    credentials::{
        AuthCredential, CredentialWrite, McpCredentialKind, StoredCredential, TokenStore,
    },
    locks::SessionLockSlot,
    scratchpad::{RenameOutcome, ScratchpadEntry},
    sessions::{
        ForkOverrides, ImportSessionRecord, SessionMetaRow, SessionPatch, SessionSummary,
        SessionSweep, SourceLock, SpawnTerms,
    },
};
use crate::error::{MekaError, Result};

#[derive(Clone)]
pub(crate) struct Store {
    connection: Arc<Connection>,
    lock_dir: PathBuf,
    /// Resolved path to the on-disk database (or `:memory:`). Exposed via [`Self::database_path`]
    /// so the REPL can open a second connection for persistent input history.
    database_path: PathBuf,
    /// Set only for an in-memory database, whose lock dir is a fresh temp directory nothing else
    /// would ever clean up. Held behind an `Arc` so the removal happens when the *last* clone of
    /// this store drops, not the first: `Store` is cloned into sub-agents and tool
    /// builders, and any of those may still be locking sessions.
    _ephemeral_lock_dir: Option<Arc<EphemeralLockDir>>,
    /// What this handle's scheduler has learned about the jobs in this database. Here rather than
    /// on the scheduler because the readers that explain a held job, and the delete that has to
    /// forget one, hold the store and not the scheduler.
    scheduler_memory: Arc<crate::schedule::SchedulerMemory>,
    /// Serializes the forks that probe their source within this process, so two concurrent forks
    /// of one dormant session do not refuse each other. `flock` cannot tell a sibling fork
    /// mid-copy from another process mid-turn, and the two deserve opposite answers: a copy is
    /// one short transaction to wait behind, a turn is not.
    probed_forks: Arc<tokio::sync::Mutex<()>>,
}

/// How long to keep trying to convert a rollback-journal database to WAL before giving up.
///
/// A deadline rather than an attempt count, because the two failure modes it spans cost wildly
/// different amounts of time. When SQLite skips the busy handler for this pragma an attempt returns
/// at once, and what is wanted is many of them across the contention window; when it consults the
/// handler an attempt blocks for the full `busy_timeout` first, and ten of those would turn a
/// five-second startup failure into a fifty-second one. Counting time bounds both.
///
/// Only a first run on a fresh install can need any of this: once the database is in WAL mode the
/// pragma takes no exclusive lock and cannot contend.
const WAL_CONVERSION_DEADLINE: std::time::Duration = std::time::Duration::from_secs(5);

/// Pause between those attempts. Blocking rather than async because the whole pragma batch runs on
/// the connection's own thread, where a sleep costs nothing else.
const WAL_CONVERSION_RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(50);

fn default_database_path() -> Result<PathBuf> {
    // One resolver for the data directory, shared with the sandbox masks that hide it from a
    // confined shell: two answers to "where is the store" would leave one of them unmasked.
    let directory = crate::paths::meka_data_dir().ok_or_else(|| {
        MekaError::Config(
            "failed to determine a data directory for the database; set `MEKA_DATA_DIR` to an \
             absolute path"
                .into(),
        )
    })?;
    Ok(directory.join("meka.db"))
}

impl Store {
    /// Open the store, bringing its schema forward if it is behind.
    ///
    /// `context` carries the facts a migration cannot work out for itself; see
    /// [`migrations::Context`]. It is a parameter rather than something read here because this
    /// function must not know what a profile is: the ledger is the only place allowed to
    /// act on an older meka's store, and config is the only place that knows which profile is the
    /// default. A caller with nothing to carry forward passes the default, which every test does
    /// because a store it just created has no sessions to carry.
    pub(crate) async fn open(path: Option<&Path>, context: &migrations::Context) -> Result<Self> {
        let database_path = match path {
            Some(path) => path.to_path_buf(),
            None => default_database_path()?,
        };

        // In-memory SQLite databases (used by tests) have no on-disk parent; give each `open()`
        // call its own ephemeral lock dir under the system temp directory so concurrent tests don't
        // share lock files.
        let is_in_memory = database_path == Path::new(":memory:");
        let lock_dir = if is_in_memory {
            std::env::temp_dir().join(format!("meka-test-locks-{}", Uuid::new_v4()))
        } else {
            if let Some(parent) = database_path.parent() {
                create_private_dir(parent)?;
                // Pre-existing parents inherit their old mode; tighten if so.
                restrict_permissions(parent, 0o700);
            }
            database_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("locks")
        };
        create_private_dir(&lock_dir)?;
        restrict_permissions(&lock_dir, 0o700);
        // Owned from the moment it exists, not from the moment the store is built. Five fallible
        // steps sit between the two (opening the connection, the pragmas, the schema lock), and a
        // return from any of them must not leave the directory created with nothing holding it.
        let ephemeral_lock_dir = is_in_memory.then(|| Arc::new(EphemeralLockDir(lock_dir.clone())));

        // Pre-touched at 0600 so SQLite's `Connection::open` reuses an already-restricted file
        // rather than creating one at umask defaults to be tightened afterwards, which leaves a
        // window where another local user could open it. `-wal`/`-shm` companions still inherit
        // the umask, but the parent directory's 0700 mode keeps them inaccessible to other users.
        #[cfg(unix)]
        if !is_in_memory {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .mode(0o600)
                .open(&database_path)
                .map_err(|error| {
                    MekaError::Database(format!(
                        "failed to pre-touch database '{}': {}",
                        database_path.display(),
                        error
                    ))
                })?;
        }

        let connection = Connection::open(&database_path).await.map_err(|error| {
            MekaError::Database(format!(
                "failed to open database '{}': {}",
                database_path.display(),
                error
            ))
        })?;

        // A file that pre-existed at a more permissive mode (manual setup, restored backup) is
        // tightened now; the pre-touch above is the primary protection for a new one.
        if !is_in_memory {
            restrict_permissions(&database_path, 0o600);
        }

        // SQLite defaults foreign-key enforcement to OFF per-connection; the `FOREIGN KEY` clauses
        // in `CREATE TABLE` are decorative without this. Set before `initialize_schema` so every
        // statement it runs, and every one after, sees enforcement active. Must run outside any
        // transaction to take effect.
        connection
            .call(|connection| -> rusqlite::Result<_> {
                // Restated rather than established: `rusqlite::Connection::open` already installs a
                // five-second busy timeout before any of this runs, so this pragma pins the value
                // meka wants against a future change in that default rather than supplying one.
                // Ordered before the WAL conversion below because that is where it would matter if
                // it were ever the only source.
                connection.execute_batch(
                    "PRAGMA busy_timeout = 5000;\n\
                     PRAGMA foreign_keys = ON;",
                )?;
                // The retry is the part that fixes something. Converting a rollback-journal
                // database to WAL takes an exclusive lock, and SQLite does not always route *that*
                // pragma's acquisition through the busy handler, so with a handler installed and
                // waiting the conversion can still return `database is locked` outright when
                // several meka processes start together. An already-WAL database takes no
                // exclusive lock here and never contends, so this only ever bites a first run on a
                // fresh install (a systemd unit and a shell coming up together, which is the
                // ordinary case).
                //
                // WAL is what lets the REPL's history connection read without blocking the agent's
                // writes, so a database left in rollback mode is a live contention problem rather
                // than a cosmetic one: worth several attempts before giving up. (On `:memory:` the
                // request is silently ignored and the first attempt always succeeds.)
                let giving_up_at = std::time::Instant::now() + WAL_CONVERSION_DEADLINE;
                loop {
                    match connection.execute_batch("PRAGMA journal_mode = WAL;") {
                        Ok(()) => break,
                        Err(error) if std::time::Instant::now() >= giving_up_at => {
                            return Err(error);
                        }
                        Err(_) => std::thread::sleep(WAL_CONVERSION_RETRY_DELAY),
                    }
                }
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to set connection pragmas: {error}"))
            })?;

        let store = Self {
            connection: Arc::new(connection),
            _ephemeral_lock_dir: ephemeral_lock_dir,
            probed_forks: Arc::default(),
            lock_dir,
            database_path,
            scheduler_memory: Arc::default(),
        };
        store.initialize_schema(context).await?;
        store.prune_orphan_lock_files().await;
        Ok(store)
    }

    /// Resolved path to the on-disk database (or `:memory:`). The REPL opens a second synchronous
    /// connection here for persistent input history (see
    /// [`crate::host::repl::history::PromptHistory`]).
    pub(crate) fn database_path(&self) -> &Path {
        &self.database_path
    }

    async fn initialize_schema(&self, context: &migrations::Context) -> Result<()> {
        // Serialize schema work across processes.
        //
        // Two things below need it, and neither is safe on its own. The migration run decides what
        // to do by reading the store and then acts on that answer, so two processes that both read
        // "needs migrating" would both try. And `store::memory::reconcile_index` makes the
        // `sqlite_master` read that decides whether the FTS triggers have drifted *outside* the
        // transaction that replaces them; the replacement itself is one immediate transaction, so
        // no process ever sees a half-applied trigger set, but a second process can see a snapshot
        // the winner is about to invalidate and then act on it after it has stopped being true. A
        // systemd unit and a shell REPL starting together is exactly when that happens.
        //
        // An OS file lock rather than a SQLite transaction, the same primitive `FileLock` uses,
        // held for the whole of the schema work so the check and the write it authorizes cannot be
        // split. The loser waits, then re-runs against the winner's finished schema and no-ops.
        let lock_path = self.lock_dir.join(format!("{SCHEMA_LOCK_STEM}.lock"));
        let schema_lock_file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
            .map_err(|error| {
                MekaError::Database(format!(
                    "failed to open schema lock '{}': {}",
                    lock_path.display(),
                    error
                ))
            })?;
        // Released where `schema_lock_file` closes, at the end of this function. Taken on a
        // blocking thread: `lock()` waits for however long another process holds it, and a runtime
        // worker parked on it is one every other task on this runtime loses for the duration.
        let _schema_lock_file = tokio::task::spawn_blocking({
            let lock_path = lock_path.clone();
            move || {
                schema_lock_file.lock().map_err(|error| {
                    MekaError::Database(format!(
                        "failed to acquire schema lock '{}': {}",
                        lock_path.display(),
                        error
                    ))
                })?;
                Ok::<_, MekaError>(schema_lock_file)
            }
        })
        .await
        .map_err(|error| {
            MekaError::Database(format!("the schema lock task did not complete: {error}"))
        })??;

        // Migrations, not declaration. [`migrations::plan`] decides what this store needs and
        // [`migrations::apply`] performs it, both under the lock taken above, so two processes
        // starting together cannot each decide to migrate and then both try. Everything downstream
        // of this point may assume the current schema unconditionally, which is the whole benefit;
        // [`migrations`] states the rule that keeps it true.
        //
        // What the lock costs, and it is real: opening the store takes a write lock even when there
        // is nothing to do, so a long-running writer elsewhere fails commands that only read. An
        // external `BEGIN IMMEDIATE` held for eight seconds kills `meka --oneshot` at 5.1 seconds
        // with `failed to initialize schema in '<path>': database is locked`, and
        // `meka session list` (a pure read) dies the same way. A rare, loud, retryable startup
        // error is the accepted half of that trade.
        let database_path = self.database_path.clone();
        let context = context.clone();
        let (plan, backup) = self
            .connection
            .call(move |connection| -> std::result::Result<_, MekaError> {
                let plan = migrations::plan(connection)?;
                // Before anything is written, and only when there is something to lose. `from > 0`
                // is what distinguishes carrying an existing store forward from building a new one:
                // a fresh store has no data to preserve, and copying the empty file it does not yet
                // have would leave a `.v0.bak` beside every first run. The copy carries its own
                // `user_version`, so restoring it yields a store that migrates once when next
                // opened rather than one mistaken for already-current.
                let backup = if plan.from > 0 && plan.has_work() {
                    back_up_before_migrating(connection, &database_path, plan.from)?
                } else {
                    None
                };
                migrations::apply(connection, plan, &context)?;
                // After `apply`, never before. Two orderings have to hold at once and only this one
                // gives both: a copy must exist before an older one is removed, which the `?` on
                // `back_up_before_migrating` above guarantees, *and* the older copy must survive a
                // migration that fails. `apply` rolls its own transaction back and reports "The
                // store is unchanged", but deleting a file is not part of that transaction, so
                // pruning first would have a failed upgrade destroy the copy the user is told to
                // fall back on, and every retry take another one. Once `apply` has returned `Ok`,
                // the copies below it are genuinely superseded. See `prune_older_backups`.
                if let Some(target) = &backup {
                    prune_older_backups(&database_path, target);
                }
                // Reconciliation rather than creation, and outside the ledger for that reason: it
                // asks whether this database's FTS triggers are the ones this build requires and
                // makes them so, which is as true of a store created a minute ago as of one carried
                // forward. `crate::store::memory` owns the reasoning.
                crate::store::memory::reconcile_index(connection).map_err(|error| {
                    MekaError::Database(format!("failed to reconcile the memory index: {error}"))
                })?;
                Ok((plan, backup))
            })
            .await
            .map_err(|error| match error {
                tokio_rusqlite::Error::Error(inner) => inner,
                other => MekaError::Database(format!(
                    "failed to initialize schema in '{}': {}",
                    self.database_path.display(),
                    other
                )),
            })?;

        if plan.has_work() {
            match backup {
                Some(path) => tracing::info!(
                    "brought the store forward from schema version {from} to {head}; the pre-migration copy is at {path}",
                    from = plan.from,
                    head = plan.head,
                    path = path.display()
                ),
                None => tracing::info!(
                    "brought the store forward from schema version {from} to {head}",
                    from = plan.from,
                    head = plan.head
                ),
            }
        }
        Ok(())
    }

    /// Run `PRAGMA wal_checkpoint(TRUNCATE)` to flush the SQLite write-ahead log into the main
    /// database file. Called from `meka serve`'s graceful-shutdown path so a `SIGTERM` followed
    /// by a fresh `meka` process invocation doesn't see a long WAL replay on open. A failure is
    /// not fatal to the caller: SQLite recovers from an unflushed WAL on next open.
    pub(crate) async fn checkpoint(&self) -> Result<()> {
        self.connection
            .call(|connection| -> rusqlite::Result<_> {
                connection.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to checkpoint the WAL: {error}")))
    }
}

impl Store {
    /// The credential tables, on this store's connection.
    pub(crate) fn token_store(&self) -> TokenStore {
        TokenStore {
            connection: Arc::clone(&self.connection),
            lock_dir: self.lock_dir.clone(),
            _ephemeral_lock_dir: self._ephemeral_lock_dir.clone(),
        }
    }

    /// The scheduled-jobs table, on this store's connection.
    pub(crate) fn schedule_store(&self) -> crate::store::schedule::ScheduleStore {
        crate::store::schedule::ScheduleStore::new(
            Arc::clone(&self.connection),
            Arc::clone(&self.scheduler_memory),
        )
    }

    /// What this handle's scheduler has learned about the jobs in this database.
    pub(crate) fn scheduler_memory(&self) -> &crate::schedule::SchedulerMemory {
        &self.scheduler_memory
    }

    /// Handle on the memory store. See [`crate::store::memory`] for why it shares this database
    /// rather than owning one: meka has one database, and a second would be a new thing to back
    /// up, lock and explain for the sake of two tables.
    pub(crate) fn memory_store(&self, enabled: bool) -> Arc<crate::store::memory::MemoryStore> {
        crate::store::memory::MemoryStore::new(Arc::clone(&self.connection), enabled)
    }

    /// The background-tasks table, on this store's connection.
    pub(crate) fn background_store(&self) -> crate::store::background::BackgroundStore {
        crate::store::background::BackgroundStore::new(Arc::clone(&self.connection))
    }
}

#[cfg(test)]
impl Store {
    /// An in-memory store at the current schema.
    ///
    /// `:memory:`, spelled out. `None` is not "no path" but *the default path*, so a helper that
    /// passed it created sessions in the developer's own `meka.db` on every `cargo test`, and
    /// migrated and backed it up on the way in.
    pub(crate) async fn for_test() -> Self {
        Self::open(Some(Path::new(":memory:")), &Default::default())
            .await
            .expect("an in-memory store opens")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `MEKA_DATA_DIR` has to be absolute, for a sharper version of the reason `MEKA_CONFIG_DIR`
    /// does: `meka.db` holds every provider credential, so a relative value gives one credential
    /// store per directory meka is launched from, none of them the one the user set up.
    ///
    /// The comment justifying the config-directory hardening asserted this sibling "already refuses
    /// both". It refused only the empty value.
    #[tokio::test]
    async fn a_relative_data_dir_is_refused_rather_than_joined_to_the_cwd() {
        let _env = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        let previous = std::env::var_os("MEKA_DATA_DIR");
        // SAFETY: `MEKA_DATA_DIR` is process-global; the lock above serializes every test that
        // touches it, and the original value is restored before the guard drops.
        unsafe { std::env::set_var("MEKA_DATA_DIR", "relative/data") };
        let relative = default_database_path();

        let absolute_dir = std::env::temp_dir().join("meka-data-dir-test");
        unsafe { std::env::set_var("MEKA_DATA_DIR", &absolute_dir) };
        let absolute = default_database_path();

        unsafe {
            match previous {
                Some(value) => std::env::set_var("MEKA_DATA_DIR", value),
                None => std::env::remove_var("MEKA_DATA_DIR"),
            }
        }

        let relative = relative.expect("a rejected override still resolves a platform default");
        assert!(
            !relative.starts_with("relative"),
            "a relative override must not be joined to the cwd; got {}",
            relative.display(),
        );
        assert_eq!(
            absolute.expect("an absolute override is honored"),
            absolute_dir.join("meka.db"),
            "and an absolute one is still used verbatim",
        );
    }

    /// Regression test for the umask-dependent permission bug: the session database file stores
    /// OAuth tokens and MCP credentials, so it must be readable by the owner only (0600) and the
    /// surrounding directory by the owner only (0700), regardless of the user's umask.
    #[cfg(unix)]
    #[tokio::test]
    async fn session_db_file_mode() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = tempfile::tempdir().expect("tempdir");
        let db_path = temp_dir.path().join("data").join("meka.db");

        let _manager = Store::open(Some(&db_path), &Default::default())
            .await
            .expect("open session");

        let db_mode = std::fs::metadata(&db_path)
            .expect("stat db")
            .permissions()
            .mode();
        assert_eq!(
            db_mode & 0o777,
            0o600,
            "db file should be 0600 (got {:o})",
            db_mode & 0o777
        );

        let dir_mode = std::fs::metadata(db_path.parent().expect("parent"))
            .expect("stat dir")
            .permissions()
            .mode();
        assert_eq!(
            dir_mode & 0o777,
            0o700,
            "data dir should be 0700 (got {:o})",
            dir_mode & 0o777
        );

        let lock_mode = std::fs::metadata(db_path.parent().expect("parent").join("locks"))
            .expect("stat lock dir")
            .permissions()
            .mode();
        assert_eq!(
            lock_mode & 0o777,
            0o700,
            "lock dir should be 0700 (got {:o})",
            lock_mode & 0o777
        );
    }

    /// Opening a fresh database while another connection holds its write lock must wait, and must
    /// come out of it in WAL: converting a rollback journal takes an exclusive lock, and a database
    /// left unconverted is a permanent contention problem, not a slow start.
    ///
    /// What this does *not* isolate is the retry loop. `rusqlite` installs a five-second busy
    /// timeout at open, so any hold shorter than that is waited out on the first attempt and the
    /// retry never runs. The case the retry exists for is the one where SQLite declines to consult
    /// the busy handler for this pragma at all -- observed at a couple of launches per few hundred,
    /// and not reproducible on demand. So this pins the property (a contended open waits and gets
    /// WAL) and the retry's own arm is covered by argument, not by a test.
    ///
    /// The writer is a plain `rusqlite` connection holding `BEGIN EXCLUSIVE`, which is what an
    /// unrelated process mid-transaction looks like from outside.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_first_open_waits_out_a_writer_instead_of_failing() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let db_path = temp_dir.path().join("meka.db");

        // A rollback-journal database with something in it, so the open below has a real conversion
        // to do rather than a no-op on an empty file.
        let blocker = rusqlite::Connection::open(&db_path).expect("open");
        blocker
            .execute_batch("CREATE TABLE placeholder (id INTEGER);")
            .expect("seed");
        blocker.execute_batch("BEGIN EXCLUSIVE;").expect("hold");

        let released = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(600));
            blocker.execute_batch("COMMIT;").expect("release");
        });

        let store = Store::open(Some(&db_path), &Default::default())
            .await
            .expect("a contended first open must wait, not fail");
        released.join().expect("the writer thread finishes");

        let mode: String = store
            .connection
            .call(|connection| {
                connection.query_row("PRAGMA journal_mode", [], |row| row.get::<_, String>(0))
            })
            .await
            .expect("read the journal mode");
        assert_eq!(
            mode.to_lowercase(),
            "wal",
            "and must actually get WAL, not settle for the rollback journal"
        );
    }

    // End-to-end regression tests for `meka session list`'s title. These tests mock the complete
    // pipeline that produces the `Title` column: build the turn-context block the agent actually
    // sends, put it in front of a user prompt the way `agent::Agent::run_turn` does, persist the
    // event, then call `list_sessions` and assert the title matches the raw user prompt. Any future
    // change to `build_turn_context`'s output shape, `run_turn`'s message shape, the event
    // encoding, `list_sessions`'s SQL or `title_of_first_user_row` that breaks the title will fail
    // one of these tests.

    // Child-session tests: parent→sub-agent linkage, cascade-on-delete, and `meka session list`
    // filter behavior.

    // MCP TokenStore tests. Exercise the methods backing `meka mcp login/logout`. In-memory DB
    // keeps each case hermetic.

    /// Two hosts starting together against one unmigrated store. The schema lock is what makes the
    /// loser re-read the winner's answer instead of acting on its own stale one, and a migration
    /// applied twice is how a store gets a duplicated column or a half-converted table.
    ///
    /// Spawned rather than `join!`ed, and that is not a style choice. `initialize_schema` holds a
    /// *blocking* file lock across an `await`, so two opens driven by one task deadlock: the second
    /// blocks the thread that the first needs in order to be polled again and release. Separate
    /// tasks put them on separate workers, which is also what two real hosts are.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn two_hosts_opening_one_unmigrated_store_migrate_it_once() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let database_path = temp_dir.path().join("meka.db");
        {
            let connection = rusqlite::Connection::open(&database_path).expect("open");
            connection
                .execute_batch(crate::store::migrations::baseline_for_test())
                .expect("the baseline builds");
        }

        let one = tokio::spawn({
            let database_path = database_path.clone();
            async move { Store::open(Some(&database_path), &Default::default()).await }
        });
        let other = tokio::spawn({
            let database_path = database_path.clone();
            async move { Store::open(Some(&database_path), &Default::default()).await }
        });
        let first = one
            .await
            .expect("the task finishes")
            .expect("one host opens");
        let second = other
            .await
            .expect("the task finishes")
            .expect("the other host opens too");

        let version: i64 = first
            .connection
            .call(|connection| {
                connection.query_row("SELECT * FROM pragma_user_version", [], |row| row.get(0))
            })
            .await
            .expect("a version");
        assert!(version > 0, "the store was migrated");
        // One migration, not two: a second pass would have tried to add columns that now exist.
        let claim_columns: i64 = second
            .connection
            .call(|connection| {
                connection.query_row(
                    "SELECT count(*) FROM pragma_table_info('scheduled_jobs') WHERE name = 'claimed_by'",
                    [],
                    |row| row.get(0),
                )
            })
            .await
            .expect("column count");
        assert_eq!(claim_columns, 1);
    }
}
