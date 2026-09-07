//! The schema ledger: the only place in meka that knows a previous version of the store existed.
//!
//! `AGENTS.md`'s "Schema and migrations" section states the four rules and what they buy. What
//! follows is what enforcing them looks like here.
//!
//! **Nothing outside this module may know that an older meka existed.** No fallback reader for a
//! superseded shape, no `#[serde(alias)]` for a renamed field, no branch whose condition is "was
//! this row written by an older release". Every other reader assumes the current schema
//! unconditionally, and may do so because [`apply`] has already run by the time it sees the store.
//!
//! **A migration is frozen once any store has run it**, development stores included: `user_version`
//! is a positional index into this list, so removing an entry renumbers every entry after it, and a
//! store stamped between the hole and the new head skips a step it never ran and then stamps itself
//! current.
//!
//! **A migration may not call meka's own code.** [`gates_become_kind_and_spec`] builds its JSON by
//! hand rather than through `Gate::spec`, because a borrowed function starts meaning something else
//! the day the gate types are refactored, years after the users it ran for could notice. It may
//! *receive* a fact it cannot work out: that is [`Context`], and a `String` cannot change meaning
//! under a frozen step the way a function can. [`sessions_name_their_provider`] needs the profile a
//! session with none recorded should adopt, and `config.toml` is the only place that knows.
//!
//! **A migration must be safe to run twice.** `user_version` lives in the file header, and `sqlite3
//! old.db .dump | sqlite3 new.db` drops it where plain `VACUUM` keeps it. Such a store is
//! classified by shape, which can only answer "fresh" or "at the baseline", so every step after the
//! baseline replays over data that already has it. [`gates_become_kind_and_spec`] guards each `ADD
//! COLUMN` on the column's absence and returns early when `gate_command` is already gone; a plain
//! `Step::Sql("ALTER TABLE … ADD COLUMN x")` in that position fails with `duplicate column name`
//! and refuses the store on every start.
//!
//! What is *not* banned elsewhere: guards against hand-editing, corruption and bugs. Those name no
//! release and are equally true of a store created five minutes ago, so they stay where the data is
//! read. `gate_kind` and `gate_spec_json` must both be set or both be null; an unparseable
//! `gate_permission` fails closed.
//!
//! `PRAGMA user_version` is a 32-bit slot in the file header that SQLite reserves for applications
//! and never touches itself. It holds the number of migrations applied, so head is
//! `MIGRATIONS.len()`. It is transactional, so the DDL and the version bump commit or fail
//! together, and `VACUUM INTO` copies it. SQLite's own `PRAGMA schema_version` is an unrelated
//! internal DDL counter: not usable for this, and must not be written.

use crate::error::{MekaError, Result};

/// One step in the ledger. Append only: see the module docs on freezing.
struct Migration {
    /// Identifies the step in logs and in the frozen-prefix test. Never reused.
    name: &'static str,
    step: Step,
}

enum Step {
    /// Statements with no decisions in them.
    Sql(&'static str),
    /// A conversion SQL cannot express. Takes the transaction so it commits with everything else.
    Rust(fn(&rusqlite::Transaction<'_>) -> rusqlite::Result<()>),
    /// A conversion that also needs a fact only the caller can supply. See [`Context`].
    Contextual(fn(&rusqlite::Transaction<'_>, &Context) -> rusqlite::Result<()>),
}

/// Facts a migration cannot work out for itself, supplied by the caller.
///
/// This is the one loosening of rule 2, and the line it draws is between *receiving* data and
/// *calling* code to get it. `Gate::spec` can start meaning something else the day the gate types
/// are refactored, years after the stores it ran against stopped being able to notice; a `String`
/// cannot. So a migration may take one of these and may not reach for a function.
///
/// **Append only, like the ledger itself.** A shipped step's inputs are part of what is frozen
/// about it: adding a field for a later step is additive and safe, but renaming or removing one
/// silently rewrites what an already-run step would do.
#[derive(Debug, Clone, Default)]
pub(crate) struct Context {
    /// The profile a session with no recorded one adopts, or empty when nothing resolves.
    ///
    /// Empty is not a sentinel the readers know about. No profile can be named `""` in a way that
    /// resolves, so an empty value lands on the same "recorded profile is not configured" refusal
    /// a deleted profile produces, which every reader already has to handle.
    pub(crate) default_provider: String,
    /// Why `default_provider` is empty, when it is: because nothing resolved, or because the
    /// caller could not read `config.toml` at all.
    ///
    /// The two are not the same answer and must not produce the same write. "Nothing resolved" is
    /// a fact about a config meka read; "could not read it" is meka knowing nothing, and
    /// stamping every existing session against no profile on the strength of a parse error is
    /// irreversible, since the step runs once and `user_version` moves with it. So a step that
    /// would act on the value refuses instead, which aborts the transaction and leaves the
    /// store exactly as it was for a later run with a readable config.
    ///
    /// A *separate* field rather than making `default_provider` an `Option`, because `Context` is
    /// append-only for the same reason the ledger is: a shipped step's inputs are part of what is
    /// frozen about it, and changing the type of one would silently rewrite what an already-run
    /// step would have done.
    pub(crate) config_unreadable: bool,
    /// The level a root session that recorded none adopts, spelled as `config.toml` spells
    /// it, or empty when the caller knows none. The file's default and not the run's, because the
    /// value is written once and read by every later process.
    ///
    /// Empty leaves such rows as they are, which is what the frozen 0.46 step did whatever the
    /// reason; [`root_rows_take_the_default_level_once_the_config_reads`] refuses instead when the
    /// reason is `config_unreadable` and there is a row to stamp, so the level is recorded on a
    /// later run that can read the file rather than never. A row with no level runs nothing under
    /// the scheduler, which is the safe direction, and the interactive hosts fall back to their
    /// own default when they read one.
    pub(crate) default_permission: String,
}

impl Context {
    /// What a caller that has resolved a profile hands over. `None` means none resolved.
    pub(crate) fn adopting(profile: Option<&str>) -> Self {
        Self {
            default_provider: profile.unwrap_or_default().to_string(),
            config_unreadable: false,
            default_permission: String::new(),
        }
    }

    /// The same, with the level a session that never recorded one adopts.
    pub(crate) fn starting_at(mut self, permission: &str) -> Self {
        self.default_permission = permission.to_string();
        self
    }

    /// What a caller hands over when `config.toml` could not be parsed or read.
    ///
    /// Deliberately still openable: a store already at head runs no step that consults this, so the
    /// commands that exist to *repair* an unreadable config -- `meka mcp remove`, `meka account
    /// remove`, `meka profile remove`, which edit the raw document through `toml_edit` and never
    /// parse it -- keep working.
    /// Only a store that would actually adopt a profile, or stamp a level on a root row, is
    /// refused.
    pub(crate) fn on_unreadable_config() -> Self {
        Self {
            default_provider: String::new(),
            config_unreadable: true,
            default_permission: String::new(),
        }
    }
}

/// Every migration, in order. **Append only, and never edit a shipped entry**: users who already
/// ran it will not run it again, so an edit changes what new stores get and nothing else, which is
/// a divergence no test downstream of it can see. `the_ledger_is_append_only` fails the build on
/// any change to an entry any store has already run.
const MIGRATIONS: &[Migration] = &[
    Migration {
        name: "baseline_0_42",
        step: Step::Sql(BASELINE_0_42),
    },
    Migration {
        name: "gates_become_kind_and_spec",
        step: Step::Rust(gates_become_kind_and_spec),
    },
    Migration {
        name: "sessions_name_their_provider",
        step: Step::Contextual(sessions_name_their_provider),
    },
    Migration {
        name: "sessions_record_their_model_overrides",
        step: Step::Rust(sessions_record_their_model_overrides),
    },
    Migration {
        name: "mcp_credentials_hold_every_kind",
        step: Step::Rust(mcp_credentials_hold_every_kind),
    },
    Migration {
        name: "scheduled_jobs_forget_isolation",
        step: Step::Rust(scheduled_jobs_forget_isolation),
    },
    Migration {
        name: "sessions_forget_their_model_overrides",
        step: Step::Rust(sessions_forget_their_model_overrides),
    },
    Migration {
        name: "mcp_credentials_exist_on_every_store",
        step: Step::Rust(mcp_credentials_exist_on_every_store),
    },
    Migration {
        name: "background_tasks_announce_before_they_deliver",
        step: Step::Rust(background_tasks_announce_before_they_deliver),
    },
    // The REPL's history table was created by its own connection on every open, which made
    // `store/history.rs` a second owner of the schema. Every store that has run the REPL already
    // has it, hence `IF NOT EXISTS`; one that has not gets it here, and the history code
    // assumes it.
    Migration {
        name: "prompt_history_is_in_the_ledger",
        step: Step::Sql(PROMPT_HISTORY_0_46),
    },
    // 0.46 split a provider profile into an account and a profile. A session names the profile,
    // as it always did; the column takes the word the rest of meka now uses for it.
    Migration {
        name: "sessions_record_their_profile",
        step: Step::Rust(sessions_record_their_profile),
    },
    // A credential belongs to the account a login produced it for, which is the half of a
    // provider profile the credential was always keyed by.
    Migration {
        name: "credentials_belong_to_accounts",
        step: Step::Rust(credentials_belong_to_accounts),
    },
    // 0.46 replaced the `ask` level with an approvals switch beside the level. An `ask` session
    // becomes `none` with the switch on, which asks about every call as `ask` did; and a
    // root session that never recorded a level adopts the caller's default, so the scheduler
    // reads a level off every such row instead of falling back to the polling process's own.
    Migration {
        name: "sessions_carry_approvals",
        step: Step::Contextual(sessions_carry_approvals),
    },
    // 0.46 stores the per-turn context meka injects as its own content block ahead of the words.
    Migration {
        name: "user_turns_carry_their_context_as_a_block",
        step: Step::Rust(user_turns_carry_their_context_as_a_block),
    },
    // 0.46 stores image bytes once, in `blobs`, and a message row references them by content hash.
    Migration {
        name: "images_live_in_blobs",
        step: Step::Rust(images_live_in_blobs),
    },
    // 0.46 names every column and index by one rule; the JSON held in two of them follows the
    // serde shapes the rest of meka writes.
    Migration {
        name: "columns_follow_one_naming_rule",
        step: Step::Rust(columns_follow_one_naming_rule),
    },
    // 0.46 tags a thinking block's `opaque` object `type` in a `repair` row too, which the step
    // above passed over: it read each row as a list of blocks, and a repair is an envelope.
    Migration {
        name: "repair_rows_retag_their_thinking",
        step: Step::Rust(repair_rows_retag_their_thinking),
    },
    // The approvals step stamps a root row's level only when the caller could read `config.toml`,
    // and `user_version` moved past it either way, so a 0.46.0 launched against an unreadable
    // config left such a row without a level for good. This one refuses in that situation, as the
    // 0.44 step does for a profile, and stamps on the launch that can read the file.
    Migration {
        name: "root_rows_take_the_default_level_once_the_config_reads",
        step: Step::Contextual(root_rows_take_the_default_level_once_the_config_reads),
    },
    // The reader accepts one spelling of a task status, the American one meka now writes; a task
    // stopped on request under an earlier meka recorded the other.
    Migration {
        name: "background_tasks_spell_canceled_with_one_l",
        step: Step::Sql(BACKGROUND_TASKS_CANCELED),
    },
];

const PROMPT_HISTORY_0_46: &str = "CREATE TABLE IF NOT EXISTS prompt_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command_line TEXT NOT NULL,
    created_at TEXT NOT NULL
)";

const BACKGROUND_TASKS_CANCELED: &str =
    "UPDATE background_tasks SET status = 'canceled' WHERE status = 'cancelled'";

/// What [`plan`] decided, and what [`apply`] will do about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Plan {
    /// The version the store is at, after classifying an unversioned one.
    pub(crate) from: u32,
    /// The version it will be at once [`apply`] returns.
    pub(crate) head: u32,
}

impl Plan {
    /// Whether anything would be written. The overwhelmingly common answer is `false`, and the
    /// caller uses it to skip both the backup and the transaction rather than paying for a
    /// no-op write on every process start.
    pub(crate) fn has_work(&self) -> bool {
        self.from < self.head
    }
}

/// Decide what this store needs, writing nothing.
///
/// Must be called with the schema lock held, and its answer used under the same lock: two processes
/// starting together would otherwise both read "needs migrating" and both try.
pub(crate) fn plan(connection: &rusqlite::Connection) -> Result<Plan> {
    let head = MIGRATIONS.len() as u32;
    let stored = user_version(connection)?;
    // A store from a newer meka. Refused rather than migrated, because the steps that would bring
    // it here do not exist in this binary and running the ones that do would be inventing a
    // downgrade.
    if stored > head {
        return Err(MekaError::Database(format!(
            "{} is at schema version {} and this meka only knows {}, so it was written by a newer \
             release. Nothing has been changed. Upgrade meka, or point MEKA_DATA_DIR at a \
             different store",
            store_name(connection),
            stored,
            head
        )));
    }
    let from = if stored <= RETIRED_INITIALIZED_FLAG {
        classify_by_shape(connection)?
    } else {
        stored
    };
    // A store that says it is current has to look current. The one way to be stamped at head with
    // some other shape is to have been copied without the `-wal` that held the last migration's
    // writes, or restored from a dump that dropped `user_version` and then re-stamped; either way
    // the tables are not what this version number means, and every later statement would fail
    // somewhere unhelpful.
    if from == head {
        let fingerprint = schema_fingerprint(connection).map_err(|error| {
            MekaError::Database(format!(
                "failed to read the schema of {}: {}",
                store_name(connection),
                error
            ))
        })?;
        if fingerprint != HEAD_SCHEMA_FINGERPRINT {
            return Err(MekaError::Database(format!(
                "{} is stamped at schema version {} but its tables do not have that version's \
                 shape. If this store was copied from another machine or directory, copy its \
                 `-wal` and `-shm` companions with it, or run `PRAGMA wal_checkpoint(TRUNCATE)` \
                 on the source first. Nothing has been changed",
                store_name(connection),
                head
            )));
        }
    }
    Ok(Plan { from, head })
}

/// Every table the ledger owns at head, which is what the fingerprint describes. Only these: a
/// digest of everything in `sqlite_master` refused a store the moment another tool added a table
/// beside meka's (a replication tool's sequence table, a hand-made one), and the refusal
/// prescribed a `-wal` remedy that could not apply. A table missing from this list digests as
/// empty, so a store that lost one is still refused. The FTS index and its shadow tables are left
/// out: `memory::store::reconcile_index` rebuilds the index to this build's definition after this
/// check, so its shape is not the ledger's to pin.
const HEAD_TABLES: &[&str] = &[
    "account_credentials",
    "background_tasks",
    "blobs",
    "mcp_credentials",
    "memories",
    "message_blobs",
    "messages",
    "prompt_history",
    "scheduled_jobs",
    "sessions",
    "tool_outputs",
];

/// The shape of the schema at head, as [`schema_fingerprint`] computes it. Pinned by
/// `the_head_schema_fingerprint_is_pinned`, so a new migration updates this alongside the ledger.
const HEAD_SCHEMA_FINGERPRINT: u64 = 4_495_506_424_273_589_879;

/// A digest of every table's columns, independent of how the table came to have them.
///
/// Built from `PRAGMA table_info` rather than the `CREATE TABLE` text in `sqlite_master`, because
/// a column added by `ALTER TABLE` and one written inline produce different text for the same
/// schema, and a store that migrated here and one created at head must agree. The columns are
/// digested sorted by name and without their position, for the same reason one level down: an
/// `ALTER TABLE ... ADD COLUMN` appends, so a migrated store carries the same columns as a fresh
/// baseline in a different order, and a digest that included the order refused every store that
/// had ever been migrated. Only the tables in [`HEAD_TABLES`] are described; see there for why.
pub(crate) fn schema_fingerprint(connection: &rusqlite::Connection) -> rusqlite::Result<u64> {
    let mut description = String::new();
    for name in HEAD_TABLES {
        description.push_str(name);
        description.push('\n');
        let mut columns = connection.prepare("SELECT * FROM pragma_table_info(?1)")?;
        let mut described: Vec<String> = columns
            .query_map([name], |row| {
                Ok((
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, i64>(5)?,
                ))
            })?
            .map(|column| {
                let (column_name, kind, not_null, default, primary_key) = column?;
                Ok(format!(
                    "  {column_name} {kind} {not_null} {} {primary_key}\n",
                    default.unwrap_or_default()
                ))
            })
            .collect::<rusqlite::Result<_>>()?;
        described.sort();
        for column in described {
            description.push_str(&column);
        }
    }
    Ok(fnv1a(&description))
}

/// FNV-1a, written out rather than taken from `DefaultHasher`, whose output Rust does not promise
/// to keep stable across releases.
fn fnv1a(input: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// `1` does not mean what this ledger would mean by it, so it is never taken at face value.
///
/// A store written before 0.42 carries `PRAGMA user_version = 1` from a different schema system,
/// which used it as a one-shot "this database has been initialized" flag rather than as a step
/// counter. Every store that any release up to and including 0.41 finished opening still carries
/// it, which is most stores in existence and was true of the first real one this was tested
/// against.
///
/// Read as a ledger version it says "the baseline is applied", and for a 0.42-shaped store that
/// happens to be true. For a **0.41**-shaped store it is false and the consequence is severe:
/// trusting it skips [`classify_by_shape`], which is where the refusal for that shape lives, so the
/// gate conversion runs against a table with no `gate_permission`, drops the two columns it did
/// read, and commits. The store is then stamped at head, so nothing will revisit it, and every
/// later read of `scheduled_jobs` fails with `no such column: gate_permission`. Reproduced end to
/// end before this guard existed; the only way back was the backup.
///
/// Distrusting it costs nothing, because this ledger cannot produce a `1`: [`apply`] always stamps
/// `MIGRATIONS.len()`, which `the_ledger_can_never_stamp_the_retired_flag` pins at two or more.
const RETIRED_INITIALIZED_FLAG: u32 = 1;

/// Apply everything [`plan`] found pending, and stamp the new version, in one transaction.
///
/// Foreign keys are suspended for the duration, and this is the reason the two halves are split
/// across two functions. `PRAGMA foreign_keys` is a **no-op inside a transaction**, so it has to be
/// set before `BEGIN`, which the transaction-owning half cannot do. SQLite's documented procedure
/// for the table changes `ALTER TABLE` cannot express -- changing a column's type, adding or
/// removing `NOT NULL`, changing a default, dropping a constraint -- is to build a new table, copy,
/// drop the old, and rename, and it requires enforcement off. With it on, `DROP TABLE sessions`
/// cascades through `messages`, `tool_outputs`, `scheduled_jobs` and `background_tasks`, deleting
/// the entire conversation history inside a transaction that then commits successfully. Measured:
/// one child row before, zero after, with the pragma reading `1` throughout because the attempt to
/// turn it off was ignored. `PRAGMA defer_foreign_keys` does not help.
///
/// Neither shipped step rebuilds a table, so this changes nothing today. It is here now because
/// `apply`'s transaction boundary is itself a shipped decision: the first migration that needs a
/// rebuild would otherwise have to change it, and would probably not notice why it had to.
/// [`apply_steps`] runs `foreign_key_check` before committing, since nothing was enforcing
/// references while the steps ran.
pub(crate) fn apply(
    connection: &mut rusqlite::Connection,
    plan: Plan,
    context: &Context,
) -> Result<()> {
    if !plan.has_work() {
        return Ok(());
    }
    let applied = with_foreign_keys_suspended(connection, |connection| {
        apply_steps(connection, plan, context)
    })?;
    // After the commit, so a migration that rolled back is not reported as applied.
    for name in applied {
        tracing::info!("applied schema migration '{name}'");
    }
    Ok(())
}

/// Run `work` with foreign-key enforcement off, and restore it whichever way that goes.
///
/// Separated from [`apply`] so the property can be tested rather than argued for. The thing that
/// has to be true is "a step that rebuilds a table does not cascade-delete its children", and no
/// shipped migration rebuilds one, so with the suspension inlined there was nothing a test could
/// reach: the mutation sweep could only confirm the *count* was read, never that the guard worked.
/// A closure lets a test hand in the rebuild that does not exist in `MIGRATIONS`.
///
/// The restore is not optional and not best-effort. This connection goes on to serve the whole
/// process, and every write after this point expects enforcement to be live; a failure to put it
/// back is reported even when the migration itself succeeded, and logged even when the migration
/// failed too and its error is the one returned.
///
/// A panic inside a step is the one path that skips the restore. No shipped step can panic (none
/// indexes, slices or unwraps), and a future one must not either; that is part of what "a migration
/// is frozen and self-contained" buys.
fn with_foreign_keys_suspended<T>(
    connection: &mut rusqlite::Connection,
    work: impl FnOnce(&mut rusqlite::Connection) -> Result<T>,
) -> Result<T> {
    connection
        .execute_batch("PRAGMA foreign_keys = OFF;")
        .map_err(|error| {
            MekaError::Database(format!(
                "failed to suspend foreign keys for the schema migration: {error}. Nothing has been \
                 changed"
            ))
        })?;
    let outcome = work(connection);
    let restored = connection.execute_batch("PRAGMA foreign_keys = ON;");
    // Said out loud even when the migration is the thing that failed. Returning only the migration
    // error is right -- it is the more useful message and the reason the caller is unwinding -- but
    // dropping this one silently would hide that the connection is now unsafe as well.
    if let Err(error) = &restored {
        tracing::error!(
            "failed to re-enable foreign keys after the schema migration: {error}. Restart meka \
             rather than continuing with enforcement off"
        );
    }
    let outcome = outcome?;
    restored.map_err(|error| {
        MekaError::Database(format!(
            "the schema migration committed but failed to re-enable foreign keys on this \
             connection: {error}. Restart meka rather than continuing with enforcement off"
        ))
    })?;
    Ok(outcome)
}

/// The transaction half. `Immediate`, so the write lock is taken at `BEGIN` rather than on the
/// first write: under WAL a deferred transaction that upgrades later can return `SQLITE_BUSY`
/// without consulting the busy handler at all, which is the same reason [`crate::store::memory`]
/// gives for its own writes.
fn apply_steps(
    connection: &mut rusqlite::Connection,
    plan: Plan,
    context: &Context,
) -> Result<Vec<&'static str>> {
    let transaction = connection
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|error| {
            MekaError::Database(format!("failed to begin the schema migration: {error}"))
        })?;
    // Counted before as well as after, so what fails the migration is damage *it* caused rather
    // than damage it inherited. A store that already carries a dangling reference, which takes
    // hand-editing to arrange because enforcement is on for every normal write, would otherwise be
    // refused every start forever with no way forward.
    let dangling_before = count_dangling_references(&transaction)?;
    let mut applied = Vec::new();
    for (index, migration) in MIGRATIONS.iter().enumerate().skip(plan.from as usize) {
        match &migration.step {
            Step::Sql(sql) => transaction.execute_batch(sql),
            Step::Rust(step) => step(&transaction),
            Step::Contextual(step) => step(&transaction, context),
        }
        .map_err(|error| {
            MekaError::Database(format!(
                "schema migration {} ('{}') failed: {}. The store is unchanged",
                index + 1,
                migration.name,
                error
            ))
        })?;
        applied.push(migration.name);
    }
    // The price of suspending enforcement: a step that orphaned a row would otherwise commit it
    // silently, and the damage would only surface much later as a row pointing at a parent that is
    // not there.
    let dangling_after = count_dangling_references(&transaction)?;
    if dangling_after > dangling_before {
        return Err(MekaError::Database(format!(
            "the schema migration would have left {} row(s) referring to a parent that is not \
             there, so it was rolled back. The store is unchanged",
            dangling_after - dangling_before
        )));
    }
    if dangling_before > 0 {
        tracing::warn!(
            "this store already carried {dangling_before} row(s) referring to a parent that is not there. The \
             migration did not add to them and has not removed them"
        );
    }
    set_user_version(&transaction, plan.head)?;
    transaction.commit().map_err(|error| {
        MekaError::Database(format!(
            "failed to commit the schema migration: {error}. The store is unchanged"
        ))
    })?;
    Ok(applied)
}

/// How many rows point at a parent row that is not there.
///
/// A full scan of every foreign key in the store, so it is run twice per migration and not at all
/// when there is nothing to do. `pragma_foreign_key_check` reports one row per violation; only the
/// count is wanted here, because the migration is refused wholesale either way.
fn count_dangling_references(transaction: &rusqlite::Transaction<'_>) -> Result<i64> {
    transaction
        .query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |row| {
            row.get(0)
        })
        .map_err(|error| {
            MekaError::Database(format!(
                "failed to check foreign keys during the schema migration: {error}. The store is \
                 unchanged"
            ))
        })
}

fn user_version(connection: &rusqlite::Connection) -> Result<u32> {
    connection
        .query_row("SELECT * FROM pragma_user_version", [], |row| {
            row.get::<_, i64>(0)
        })
        // Clamped rather than trusted: the column is a signed 32-bit slot anyone can write, and a
        // negative value would otherwise wrap into a huge version and read as "newer than this
        // binary".
        .map(|version| version.max(0) as u32)
        .map_err(|error| {
            MekaError::Database(format!("failed to read the store's schema version: {error}"))
        })
}

fn set_user_version(transaction: &rusqlite::Transaction<'_>, version: u32) -> Result<()> {
    // Formatted rather than bound because pragmas do not accept bind parameters at all
    // (`PRAGMA user_version = ?` is a parse error). Not an injection surface, and not a candidate
    // for being "fixed" into a bound parameter later: `version` is a `u32` this module computed
    // from `MIGRATIONS.len()` and no caller can influence it.
    transaction
        .execute_batch(&format!("PRAGMA user_version = {version};"))
        .map_err(|error| {
            MekaError::Database(format!("failed to record the new schema version: {error}"))
        })
}

/// Name the file a refusal is about.
///
/// Every refusal here is a *stop*, and the reader's next move is to go look at the store. Saying
/// "this store" leaves them to work out which file that is, and the answer is not obvious: meka
/// creates one on any invocation while `config.toml` appears only once a provider is added, so a
/// machine that ran an old meka once and was never configured has a store the user has no reason to
/// believe in. Told "this store is in the 0.41 shape" there, the honest reading is that meka is
/// wrong. Named, it is a file they can look at.
///
/// `Connection::path` is rusqlite's own, so this stays inside what a migration may call. An
/// in-memory store reports an empty path rather than `None`, hence both arms.
fn store_name(connection: &rusqlite::Connection) -> &str {
    match connection.path() {
        Some(path) if !path.is_empty() => path,
        _ => "the session store",
    }
}

/// Decide what a store already is by looking at it, for the versions that cannot be trusted.
///
/// Reached when `user_version` is 0 (never stamped) or [`RETIRED_INITIALIZED_FLAG`] (stamped by a
/// system that meant something else by it). The answer is stamped by [`apply`], so this runs once
/// per store and never again: it is the whole of the adoption problem, and it is why the ban on
/// version knowledge everywhere else costs nothing.
///
/// Markers first, each the column or table that the release in question introduced. `sessions` has
/// existed for as long as meka has had a store, so its absence means there is nothing of meka's
/// here. `gate_permission` is what 0.42 added, so a `scheduled_jobs` without it is exactly 0.41.
///
/// Then completeness, because returning `1` asserts that *everything* the baseline creates is
/// already there, and the old code that would have quietly filled a gap is gone: it ran
/// `CREATE TABLE IF NOT EXISTS` for every object on every open. It also ran them as six separate
/// statements in autocommit, so a first run of any pre-0.43 release interrupted partway leaves
/// exactly this state, with `background_tasks` and the memory tables likeliest because they were
/// last. Without the check such a store is stamped at head with a table still missing, which
/// nothing will ever revisit; measured, `meka session list` succeeded and left it that way.
/// Refusing names what is wrong instead.
fn classify_by_shape(connection: &rusqlite::Connection) -> Result<u32> {
    // Not a meka store: an empty file, or one carrying tables meka did not write. Nothing to carry
    // forward either way, so build the schema alongside whatever is already there, which is what
    // every release before this one did to such a file too.
    if table_columns(connection, "sessions")?.is_empty() {
        return Ok(0);
    }
    let columns = table_columns(connection, "scheduled_jobs")?;
    if columns.is_empty() {
        return Err(MekaError::Database(format!(
            "{} has tables but no `scheduled_jobs`, so it predates 0.42 and this meka cannot bring \
             it forward. Nothing has been changed. Run the 0.42 release against it once, then this \
             one",
            store_name(connection)
        )));
    }
    if !columns.iter().any(|column| column == "gate_permission") {
        return Err(MekaError::Database(format!(
            "{} is in the 0.41 shape, which this meka cannot bring forward. Nothing has been \
             changed. Run `migrate-0.41-to-0.42.py`, attached to the 0.42 release, once; every \
             upgrade after that is automatic",
            store_name(connection)
        )));
    }
    let mut missing = Vec::new();
    for names in BASELINE_OBJECTS {
        let mut found = false;
        for name in *names {
            let present: i64 = connection
                .query_row(
                    "SELECT count(*) FROM sqlite_master WHERE name = ?1",
                    [name],
                    |row| row.get(0),
                )
                .map_err(|error| {
                    MekaError::Database(format!("failed to inspect the store's objects: {error}"))
                })?;
            if present > 0 {
                found = true;
                break;
            }
        }
        // Reported under the baseline's own name. A store that reaches here has none of the later
        // ones either, and naming a table this meka would create under a different name would send
        // the reader looking for the wrong thing in their backup.
        if !found && let Some(baseline) = names.first() {
            missing.push(*baseline);
        }
    }
    if !missing.is_empty() {
        return Err(MekaError::Database(format!(
            "{} is missing {}, which every release from 0.42 creates, so it was probably left \
             half-built by an interrupted first run. Nothing has been changed. Restore it from a \
             backup, or move it aside and let meka build a new one",
            store_name(connection),
            missing.join(", ")
        )));
    }
    Ok(1)
}

/// Every **table** [`BASELINE_0_42`] creates, by the name it appears under in `sqlite_master`.
///
/// Tables only, deliberately. A missing table means the store is not at the baseline and the
/// classification would be a lie; a missing *index* means the same queries return the same answers
/// more slowly, and refusing to start over one would be worse than the problem. The seven indexes
/// the baseline creates are therefore not checked, and not repaired either: nothing puts a dropped
/// one back. That is a real if small regression, accepted because the alternative is the
/// declare-and-heal pattern the ledger exists to replace.
///
/// Read by [`classify_by_shape`] to check that a store claiming to be at the baseline really is.
/// Deliberately a separate list rather than parsed out of the SQL: it is the *question* asked of an
/// old store, which is frozen for the same reason the migration is, whereas the SQL is the answer
/// given to a new one. A later migration that adds a table does not belong here.
///
/// Each entry is every name the object may legitimately appear under, first the baseline's. More
/// than one only where a later step **renamed** a table the baseline created, because the question
/// this list asks is whether the store was fully built rather than left half written, and a rename
/// does not make it less so. A store that was carried forward and then lost its `user_version`
/// shows the new name, and refusing it would tell the user their complete store was half-built.
/// This is the one list allowed to know that, for the reason [`classify_by_shape`] gives about
/// itself.
const BASELINE_OBJECTS: &[&[&str]] = &[
    &["background_tasks"],
    // Renamed by `mcp_credentials_hold_every_kind` once it held more than OAuth bundles.
    &["mcp_oauth_credentials", "mcp_credentials"],
    &["memories"],
    &["memories_fts"],
    &["messages"],
    // Renamed by `credentials_belong_to_accounts`. The old name survives as a view, so a store at
    // head answers to both; see that step for why.
    &["provider_credentials", "account_credentials"],
    &["scheduled_jobs"],
    &["sessions"],
    &["tool_outputs"],
];

fn table_columns(connection: &rusqlite::Connection, table: &str) -> Result<Vec<String>> {
    let mut statement = connection
        .prepare("SELECT name FROM pragma_table_info(?1)")
        .map_err(|error| MekaError::Database(format!("failed to inspect `{table}`: {error}")))?;
    let names = statement
        .query_map([table], |row| row.get::<_, String>(0))
        .and_then(|rows| rows.collect::<rusqlite::Result<Vec<_>>>())
        .map_err(|error| MekaError::Database(format!("failed to inspect `{table}`: {error}")))?;
    Ok(names)
}

/// Every table, index and virtual table as 0.42 left them.
///
/// A fresh install runs this and then every step after it, so the schema a new store gets is
/// produced by the same code path an upgraded one goes through. That replay is the point: two
/// separate definitions of "the current schema" drift, and
/// `a_fresh_store_and_an_upgraded_one_have_the_same_schema` is what proves this one cannot.
///
/// The FTS *triggers* are deliberately absent. `crate::store::memory`'s `sync_triggers` owns their
/// creation and reconciles them on every open, and a `CREATE TRIGGER IF NOT EXISTS` here would put
/// a trigger that had gone missing back before that comparison could notice, which is a silent
/// index desync that module documents having reproduced.
const BASELINE_0_42: &str = "
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        parent_session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
        cwd TEXT,
        permission TEXT,
        capabilities_json TEXT,
        token_id TEXT,
        additional_roots_json TEXT,
        subagent_spec_json TEXT,
        stat_turns INTEGER NOT NULL DEFAULT 0,
        stat_input_tokens INTEGER NOT NULL DEFAULT 0,
        stat_output_tokens INTEGER NOT NULL DEFAULT 0,
        stat_cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
        stat_cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
        stat_redactions INTEGER NOT NULL DEFAULT 0,
        stat_redacted_images INTEGER NOT NULL DEFAULT 0,
        stat_redacted_bytes INTEGER NOT NULL DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at);

    CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);

    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);

    CREATE TABLE IF NOT EXISTS provider_credentials (
        profile TEXT PRIMARY KEY,
        credentials_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS mcp_oauth_credentials (
        server_name TEXT PRIMARY KEY,
        credentials_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS tool_outputs (
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (session_id, name)
    );

    CREATE TABLE IF NOT EXISTS scheduled_jobs (
        id                TEXT PRIMARY KEY,
        session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        kind              TEXT NOT NULL,
        spec              TEXT NOT NULL,
        prompt            TEXT NOT NULL,
        gate_command      TEXT,
        gate_fire         TEXT,
        gate_last_output  TEXT,
        gate_permission   TEXT,
        isolated          INTEGER NOT NULL DEFAULT 0,
        created_at        TEXT NOT NULL,
        last_fired_at     TEXT,
        next_fire_at      TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_next_fire ON scheduled_jobs(next_fire_at);

    CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_session ON scheduled_jobs(session_id);

    CREATE TABLE IF NOT EXISTS background_tasks (
        id                TEXT PRIMARY KEY,
        session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        tool_name         TEXT NOT NULL,
        label             TEXT NOT NULL,
        status            TEXT NOT NULL,
        outcome           TEXT,
        scratchpad_name   TEXT,
        started_at        TEXT NOT NULL,
        finished_at       TEXT,
        delivered_at      TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_background_tasks_session_status
        ON background_tasks(session_id, status);

    CREATE TABLE IF NOT EXISTS memories (
        id           INTEGER PRIMARY KEY,
        name         TEXT NOT NULL UNIQUE COLLATE NOCASE,
        description  TEXT NOT NULL,
        tags         TEXT NOT NULL DEFAULT '',
        body         TEXT NOT NULL DEFAULT '',
        priority     INTEGER NOT NULL DEFAULT 5,
        recorded_at  TEXT NOT NULL,
        updated_at   TEXT NOT NULL,
        read_count   INTEGER NOT NULL DEFAULT 0,
        last_read_at TEXT
    );

    CREATE INDEX IF NOT EXISTS memories_rank ON memories(priority, recorded_at DESC);

    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        name,
        description,
        tags,
        body,
        content = 'memories',
        content_rowid = 'id',
        tokenize = 'porter unicode61'
    );
";

/// 0.43: a gate becomes `gate_kind` plus a JSON `gate_spec`, and a due occurrence is leased.
///
/// Both old values were written only by meka, so the predicate mapping is a rename rather than a
/// guess.
fn gates_become_kind_and_spec(transaction: &rusqlite::Transaction<'_>) -> rusqlite::Result<()> {
    let columns: Vec<String> = {
        let mut statement = transaction.prepare("SELECT name FROM pragma_table_info(?1)")?;
        let rows = statement.query_map(["scheduled_jobs"], |row| row.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    let has = |column: &str| columns.iter().any(|existing| existing == column);

    for (column, statement) in [
        (
            "gate_kind",
            "ALTER TABLE scheduled_jobs ADD COLUMN gate_kind TEXT",
        ),
        (
            "gate_spec",
            "ALTER TABLE scheduled_jobs ADD COLUMN gate_spec TEXT",
        ),
        (
            "claimed_by",
            "ALTER TABLE scheduled_jobs ADD COLUMN claimed_by TEXT",
        ),
        (
            "claimed_until",
            "ALTER TABLE scheduled_jobs ADD COLUMN claimed_until TEXT",
        ),
        (
            "attempts",
            "ALTER TABLE scheduled_jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0",
        ),
    ] {
        if !has(column) {
            transaction.execute_batch(statement)?;
        }
    }

    // A store whose gates are already in the new shape, which is what a hand-run of the retired
    // script leaves behind. The lease columns above are still worth reaching, because an early
    // build of that script added the gate columns without them.
    if !has("gate_command") {
        return Ok(());
    }

    let mut unconvertible: Vec<String> = Vec::new();
    {
        let mut statement = transaction.prepare(
            "SELECT id, gate_command, gate_fire FROM scheduled_jobs \
             WHERE gate_command IS NOT NULL OR gate_fire IS NOT NULL",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })?;
        for row in rows {
            let (id, command, fire) = row?;
            let converted = match (command, fire) {
                (Some(command), Some(fire)) => predicate_for(&fire).map(|predicate| {
                    // Built through `serde_json` rather than by hand because a gate command is
                    // arbitrary user text: quotes, newlines and non-ASCII all have to survive, and
                    // hand-rolled escaping is where that goes wrong. The *shape* is inlined
                    // deliberately, mirroring `GateSpec` as it stands today without borrowing it;
                    // see the module docs on frozen dependencies.
                    serde_json::json!({ "shell": { "command": command }, "when": predicate })
                        .to_string()
                }),
                _ => None,
            };
            match converted {
                Some(spec) => {
                    transaction.execute(
                        "UPDATE scheduled_jobs SET gate_kind = 'shell', gate_spec = ?2 \
                         WHERE id = ?1",
                        rusqlite::params![id, spec],
                    )?;
                }
                // A row 0.42 refused to load: a half-written gate, or a `gate_fire` value it did
                // not recognise. Preserved in exactly the state it was already in rather than
                // guessed at or deleted, by setting `gate_kind` and leaving `gate_spec` null, which
                // the reader's existing corrupt-row rule refuses the same way 0.42's did. Getting
                // this wrong is expensive and silent in one specific direction: leaving both null
                // reads as *no gate at all*, which turns a watcher that never fired into a timer
                // that fires every interval.
                None => {
                    transaction.execute(
                        "UPDATE scheduled_jobs SET gate_kind = 'shell', gate_spec = NULL \
                         WHERE id = ?1",
                        rusqlite::params![&id],
                    )?;
                    unconvertible.push(id);
                }
            }
        }
    }

    transaction.execute_batch(
        "ALTER TABLE scheduled_jobs DROP COLUMN gate_command;
         ALTER TABLE scheduled_jobs DROP COLUMN gate_fire;",
    )?;

    if !unconvertible.is_empty() {
        tracing::warn!(
            "{} scheduled job(s) had a gate that could not be read and stay inert, as the previous \
             release left them: {}. They will not fire or appear in `meka schedule list`; recreate \
             them from the pre-migration backup if you still want them",
            unconvertible.len(),
            unconvertible.join(", ")
        );
    }
    Ok(())
}

/// 0.44: a session records the provider profile it runs with.
///
/// The column is `NOT NULL` because every session created from here on has a resolved profile at
/// the moment it is written, so the readers get to assume one unconditionally. Existing rows have
/// no such fact to recover, and this is the only place allowed to invent one.
///
/// The value is whatever the caller resolved as the default. When nothing resolved, the sole stored
/// credential is a better guess than nothing: a store with sessions in it was used, and a single
/// credential is almost certainly the profile that ran them. Two or more cannot be told apart from
/// here, so those fall through to the empty string, which no configured profile can be named and so
/// resolves to the refusal a deleted profile already produces.
#[allow(
    clippy::uninlined_format_args,
    reason = "a released migration body is frozen; the ledger digests it byte for byte"
)]
fn sessions_name_their_provider(
    transaction: &rusqlite::Transaction<'_>,
    context: &Context,
) -> rusqlite::Result<()> {
    let columns = {
        let mut statement = transaction.prepare("SELECT name FROM pragma_table_info(?1)")?;
        let rows = statement.query_map(["sessions"], |row| row.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<Vec<String>>>()?
    };
    // Guarded rather than bare, so a store that lost its `user_version` and replays this step does
    // not fail with `duplicate column name` and refuse to open on every start afterwards.
    if !columns.iter().any(|column| column == "provider") {
        transaction
            .execute_batch("ALTER TABLE sessions ADD COLUMN provider TEXT NOT NULL DEFAULT ''")?;
    }

    // Refused, not guessed. An unreadable `config.toml` means the caller could not tell us what
    // these sessions should adopt, and this step gets one attempt: it runs once, `user_version`
    // moves with it, and every row it stamps against no profile stays that way. Erroring aborts the
    // transaction, so the store is untouched and a later run with a readable config migrates it
    // correctly. The `ALTER TABLE` above is inside the same transaction and goes with it.
    if context.config_unreadable {
        let carried: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM sessions WHERE provider = ''",
            [],
            |row| row.get(0),
        )?;
        if carried > 0 {
            // `InvalidParameterName` carries an arbitrary message, which is the idiom `session.rs`
            // already uses to surface a non-SQLite failure through a `rusqlite::Result`. Its
            // `Display` prefixes the variant name, which reads oddly here but is the price of the
            // one variant available without enabling a rusqlite feature for a single error path.
            return Err(rusqlite::Error::InvalidParameterName(format!(
                "cannot record a provider for {} carried-forward session(s) while config.toml \
                 cannot be read; fix the file and start meka again",
                carried
            )));
        }
    }

    let adopted = if context.default_provider.is_empty() {
        sole_credential(transaction)?.unwrap_or_default()
    } else {
        context.default_provider.clone()
    };
    if adopted.is_empty() {
        // Only when there is something to say. Warning unconditionally fired on the literal first
        // run of a fresh install, where the sessions table is empty and there is nothing to leave
        // without a provider, and on every `meka provider add` before a default exists. A warning
        // that appears when nothing is wrong teaches the reader to ignore the one that matters.
        let stranded: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM sessions WHERE provider = ''",
            [],
            |row| row.get(0),
        )?;
        if stranded > 0 {
            tracing::warn!(
                "no provider profile could be resolved, so {} existing session(s) were left \
                 without one. Configure a provider, then resume each with --provider once to \
                 record which it runs on",
                stranded
            );
        }
        return Ok(());
    }
    // Only the rows that have nothing, which is what makes a replay a no-op rather than a rewrite
    // of a session someone has since moved to another profile.
    let adopting = transaction.execute(
        "UPDATE sessions SET provider = ?1 WHERE provider = ''",
        rusqlite::params![&adopted],
    )?;
    // Said out loud, because for anyone with more than one account this is a guess. The alternative
    // was leaving these sessions unresumable, so adopting is right; doing it silently is not, since
    // a session that actually ran on the other account now names this one and resuming it bills
    // here. `info!` and not `warn!` because it is a one-time lifecycle signpost that is simply
    // correct for a single-profile install, and a warning that cries wolf for most readers is one
    // the rest learn to skip.
    if adopting > 0 {
        tracing::info!(
            "recorded provider profile '{}' on {} session(s) that predate meka recording one. \
             Resume any that ran on a different profile once with --provider <name> to correct it",
            adopted,
            adopting
        );
    }
    Ok(())
}

/// One table for every MCP secret, not only the OAuth ones.
///
/// `mcp_oauth_credentials` was named for the only kind it could hold. A server's static bearer and
/// its `client_secret` stopped being config keys in this release and land here beside the OAuth
/// bundles, so the name had become a lie about its own contents.
///
/// `kind` is what a reader consults to know how to interpret `secret`, and it exists so that
/// nothing has to guess from the value's shape. Sniffing would be the same mistake as version
/// sniffing: a rule about what some other component's JSON happens to look like, which goes stale
/// with nothing to notice. The column is `secret` rather than the old `credentials_json` because
/// only one of the three kinds is JSON; a bearer and a client secret are the string itself.
///
/// **The key is `(server_name, kind)`, and that is the whole reason this is a rebuild rather than a
/// rename.** One server can hold two secrets at once: `McpAuthConfig::OAuth` takes an optional
/// `client_secret`, so a confidential client has a long-lived secret *and* the refreshable bundle
/// obtained with it. Under the old `server_name` primary key those two collide, and the first token
/// refresh silently overwrites the client secret. That server then works until the refresh and
/// fails afterwards, which is about the worst shape a failure can take. SQLite cannot alter a
/// primary key, so the table is built anew and the rows copied across.
///
/// Every row that already exists came from the authorization-code flow, which is the only one that
/// ever persisted anything, so they all copy over as `oauth` and there is nothing to work out.
///
/// [`mcp_credentials_exist_on_every_store`] repeats this conversion verbatim, and the duplication
/// is the point rather than an oversight: that step reaches stores this one was renumbered past,
/// and each entry is frozen on its own, so neither may borrow the other's body. The visible cost is
/// that `cargo mutants` reports replacing *this* function with `Ok(())` as a surviving mutant,
/// because the later step puts the table back. Belt and braces, deliberately, on the one table
/// whose absence took down every MCP connection.
fn mcp_credentials_hold_every_kind(
    transaction: &rusqlite::Transaction<'_>,
) -> rusqlite::Result<()> {
    let table_exists = |name: &str| -> rusqlite::Result<bool> {
        let count: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
            [name],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    };

    // Guarded on the target for the reason the module docs give: a store that lost its
    // `user_version` replays every step after the baseline, and a second pass would otherwise fail
    // on `CREATE TABLE` and refuse that store on every start afterwards.
    if table_exists("mcp_credentials")? {
        return Ok(());
    }
    transaction.execute_batch(
        "CREATE TABLE mcp_credentials (
            server_name TEXT NOT NULL,
            kind        TEXT NOT NULL,
            secret      TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            PRIMARY KEY (server_name, kind)
        )",
    )?;
    // Absent only for a store built by a release that never created it, which the baseline check in
    // `classify_by_shape` has already ruled out for anything this reaches. Guarded anyway, because
    // a step that assumes its predecessor's output is the assumption the replay rule exists to
    // break.
    if table_exists("mcp_oauth_credentials")? {
        transaction.execute_batch(
            "INSERT INTO mcp_credentials (server_name, kind, secret, updated_at)
             SELECT server_name, 'oauth', credentials_json, updated_at FROM mcp_oauth_credentials",
        )?;
        transaction.execute_batch("DROP TABLE mcp_oauth_credentials")?;
    }
    Ok(())
}

/// Add the two columns that hold a session's per-session model and endpoint.
///
/// Nullable with no backfill, and that is the correct value rather than a shortcut: absent means
/// "whatever the profile says", which is exactly what every session written before these columns
/// existed did. There is nothing to convert.
///
/// **Superseded by [`sessions_forget_their_model_overrides`], and kept anyway.** The columns lasted
/// one unreleased cycle before a profile became indivisible, so nothing shipped with them and the
/// obvious move was to delete this entry outright. That is wrong, and the reason is the whole point
/// of a positional ledger: `user_version` is an *index*, so removing an entry renumbers every entry
/// after it, and a store stamped anywhere between the hole and the new head silently skips a step
/// it never ran while claiming to be current. A development store sitting at 4 lost
/// `mcp_credentials_hold_every_kind` exactly that way, then stamped itself finished so nothing
/// would ever revisit it. An entry is frozen once *any* store has run it, which is not the same
/// thing as having shipped; append a reversal instead.
///
/// A consequence worth stating, because it looks like a coverage hole and is not: this body is now
/// net-inert on every path, since step 6 drops exactly what it adds. `cargo mutants` therefore
/// reports replacing it with `Ok(())` as a surviving mutant, and no test can kill it. The entry
/// still has to be here, holding index 3 so nothing after it moves.
fn sessions_record_their_model_overrides(
    transaction: &rusqlite::Transaction<'_>,
) -> rusqlite::Result<()> {
    let columns = {
        let mut statement = transaction.prepare("SELECT name FROM pragma_table_info(?1)")?;
        let rows = statement.query_map(["sessions"], |row| row.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<Vec<String>>>()?
    };
    // Guarded for the reason the module docs give: a store that lost its `user_version` replays
    // every step after the baseline, and a bare `ADD COLUMN` would then refuse it forever.
    if !columns.iter().any(|column| column == "model_override") {
        transaction.execute_batch("ALTER TABLE sessions ADD COLUMN model_override TEXT")?;
    }
    if !columns.iter().any(|column| column == "base_url_override") {
        transaction.execute_batch("ALTER TABLE sessions ADD COLUMN base_url_override TEXT")?;
    }
    Ok(())
}

/// 0.44: a session records the profile it runs on and nothing else.
///
/// A provider profile is an indivisible bundle, so a session names one rather than carrying a
/// rewritten copy of part of it. The two columns held the copy; they are dropped rather than left
/// inert for the reason [`scheduled_jobs_forget_isolation`] gives about `isolated`.
///
/// Nothing is converted. A row that pinned a model was pinning it *at its profile's own endpoint*,
/// so the profile still describes where that conversation was had; what is lost is a model
/// override, which was only ever expressible for one unreleased cycle.
fn sessions_forget_their_model_overrides(
    transaction: &rusqlite::Transaction<'_>,
) -> rusqlite::Result<()> {
    let columns: Vec<String> = {
        let mut statement = transaction.prepare("SELECT name FROM pragma_table_info(?1)")?;
        let rows = statement.query_map(["sessions"], |row| row.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    // Guarded on each column independently, so a store that lost its `user_version` and replays
    // this step does not fail on `no such column` and refuse to open ever afterwards.
    if columns.iter().any(|column| column == "model_override") {
        transaction.execute_batch("ALTER TABLE sessions DROP COLUMN model_override")?;
    }
    if columns.iter().any(|column| column == "base_url_override") {
        transaction.execute_batch("ALTER TABLE sessions DROP COLUMN base_url_override")?;
    }
    Ok(())
}

/// Repair a store that a 0.44 development build renumbered past `mcp_credentials_hold_every_kind`.
///
/// That build deleted an entry from the middle of the ledger, which shifted this table's step down
/// one. A store stamped at the old index 4 was therefore judged to have already run it, skipped it,
/// and stamped itself current -- so every MCP connection failed with `no such table:
/// mcp_credentials` while the migration reported success.
///
/// Restoring the ledger's numbering fixes the *cause*, but not those stores: they are now stamped
/// past the step they missed, so nothing replays it. Appending is the only thing that reaches them,
/// which is the same reason the reversal above is appended rather than the entry being edited.
///
/// A no-op on every store that is not in that state, which is all of them but the ones that ran
/// that build. Deliberately self-contained rather than calling [`mcp_credentials_hold_every_kind`]:
/// each entry is frozen on its own, and a step that borrows another's body would start meaning
/// whatever that one meant later.
fn mcp_credentials_exist_on_every_store(
    transaction: &rusqlite::Transaction<'_>,
) -> rusqlite::Result<()> {
    let table_exists = |name: &str| -> rusqlite::Result<bool> {
        let count: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
            [name],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    };

    if table_exists("mcp_credentials")? {
        return Ok(());
    }
    transaction.execute_batch(
        "CREATE TABLE mcp_credentials (
            server_name TEXT NOT NULL,
            kind        TEXT NOT NULL,
            secret      TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            PRIMARY KEY (server_name, kind)
        )",
    )?;
    if table_exists("mcp_oauth_credentials")? {
        transaction.execute_batch(
            "INSERT INTO mcp_credentials (server_name, kind, secret, updated_at)
             SELECT server_name, 'oauth', credentials_json, updated_at FROM mcp_oauth_credentials",
        )?;
        transaction.execute_batch("DROP TABLE mcp_oauth_credentials")?;
    }
    Ok(())
}

/// Separate "subscribers were told" from "the model was told" on a background task.
///
/// `delivered_at` answered both while they were the same instant: the poller stamped it, fired the
/// `task.finished` webhook, and ran the reporting turn in one pass. They stopped coinciding when an
/// outcome became able to wait for a turn rather than cause one, and one column cannot carry two
/// facts that no longer happen together -- leaving the stamp off to defer the model's copy also
/// re-fires the webhook on every poll.
///
/// The backfill copies the stamp across for every delivered row, recording that the outcome was
/// reported rather than that any endpoint received it: a REPL, ACP or one-shot run has no
/// subscribers, and even `meka serve` sends only to endpoints subscribed at the time. The store
/// cannot know which, and this is the only answer available to it.
///
/// It changes no behavior. Every path that decides whether to announce also requires
/// `delivered_at IS NULL`, so a delivered row is never a candidate whatever this column says; the
/// value is visible only on `GET /v1/sessions/{id}/tasks`, whose field documents the same caveat.
/// A row still undelivered is left NULL, which is how a task interrupted before any host could
/// report it finally reaches a subscriber.
fn background_tasks_announce_before_they_deliver(
    transaction: &rusqlite::Transaction<'_>,
) -> rusqlite::Result<()> {
    let columns: Vec<String> = {
        let mut statement = transaction.prepare("SELECT name FROM pragma_table_info(?1)")?;
        let rows = statement.query_map(["background_tasks"], |row| row.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    if columns.iter().any(|column| column == "announced_at") {
        return Ok(());
    }

    transaction.execute_batch(
        "ALTER TABLE background_tasks ADD COLUMN announced_at TEXT;
         UPDATE background_tasks SET announced_at = delivered_at WHERE delivered_at IS NOT NULL;",
    )
}

/// 0.44: a scheduled job always fires in the session that created it.
///
/// `isolated` let a job run its turn in a fresh root session instead of the conversation that
/// created it. The flag was honored only by `meka serve`; the REPL and ACP each drive one
/// conversation per session and ran such a job in the open one, with a warning. It is gone, so
/// every host now does what two of the three already did.
///
/// Nothing is converted, because there is nothing to convert: a job's identity is its session, its
/// schedule and its prompt, and only where the turn landed changes. A row that carried `1` keeps
/// firing on exactly its old schedule, into the conversation it belongs to.
///
/// Dropping the column rather than leaving it inert is what earns every reader the right to stop
/// asking. The row decoder addresses columns by position, so a dead column is not free: it stays in
/// every `SELECT` list and every index below it counts around it, which is a standing invitation to
/// an off-by-one in a place where the symptom is a timestamp that will not parse.
fn scheduled_jobs_forget_isolation(
    transaction: &rusqlite::Transaction<'_>,
) -> rusqlite::Result<()> {
    let columns: Vec<String> = {
        let mut statement = transaction.prepare("SELECT name FROM pragma_table_info(?1)")?;
        let rows = statement.query_map(["scheduled_jobs"], |row| row.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    // Guarded for the reason the module docs give: a store that lost its `user_version` replays
    // every step after the baseline, and a bare `DROP COLUMN` on the second pass fails with
    // `no such column` and refuses that store on every start afterwards.
    if columns.iter().any(|column| column == "isolated") {
        transaction.execute_batch("ALTER TABLE scheduled_jobs DROP COLUMN isolated")?;
    }
    Ok(())
}

/// `sessions.provider` becomes `sessions.profile`.
///
/// A rename, not a conversion: the column held a profile name all along, and 0.46 renamed the
/// concept rather than the values. Guarded on the current column set for rule 3, and the replay
/// case has one more shape than usual. A store that lost its `user_version` replays
/// [`sessions_name_their_provider`], which finds no `provider` column and adds one back beside
/// `profile`; that re-added column is empty bookkeeping the frozen step cannot know is redundant,
/// so it is dropped here, and `profile` keeps what the user's sessions actually record.
fn sessions_record_their_profile(transaction: &rusqlite::Transaction<'_>) -> rusqlite::Result<()> {
    let columns: Vec<String> = {
        let mut statement = transaction.prepare("SELECT name FROM pragma_table_info(?1)")?;
        let rows = statement.query_map(["sessions"], |row| row.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    let has_provider = columns.iter().any(|column| column == "provider");
    let has_profile = columns.iter().any(|column| column == "profile");
    match (has_provider, has_profile) {
        (true, false) => {
            transaction.execute_batch("ALTER TABLE sessions RENAME COLUMN provider TO profile")?;
        }
        (true, true) => {
            transaction.execute_batch("ALTER TABLE sessions DROP COLUMN provider")?;
        }
        (false, _) => {}
    }
    Ok(())
}

/// `provider_credentials(profile, …)` becomes `account_credentials(account, …)`.
///
/// Same rename as [`sessions_record_their_profile`], one table over: the row was always keyed by
/// the name a login was run for, which 0.46 calls the account.
///
/// **The old name is kept as a view**, and that is not a reader's convenience: nothing outside this
/// module names it. It exists for rule 3. [`sessions_name_their_provider`] is frozen and, when no
/// default profile resolves, reads `provider_credentials` to guess one; a store that lost its
/// `user_version` replays that step over this shape, and without the view it would fail with `no
/// such table` and refuse to open on every start afterwards. The view answers the frozen query with
/// the account names, which the replay of [`sessions_record_their_profile`] then discards along
/// with the column they were stamped into.
fn credentials_belong_to_accounts(transaction: &rusqlite::Transaction<'_>) -> rusqlite::Result<()> {
    let object_type = |name: &str| -> rusqlite::Result<Option<String>> {
        let mut statement =
            transaction.prepare("SELECT type FROM sqlite_master WHERE name = ?1 LIMIT 1")?;
        let mut rows = statement.query([name])?;
        match rows.next()? {
            Some(row) => Ok(Some(row.get::<_, String>(0)?)),
            None => Ok(None),
        }
    };
    if object_type("account_credentials")?.is_none()
        && object_type("provider_credentials")?.as_deref() == Some("table")
    {
        transaction
            .execute_batch("ALTER TABLE provider_credentials RENAME TO account_credentials")?;
    }
    let columns: Vec<String> = {
        let mut statement = transaction.prepare("SELECT name FROM pragma_table_info(?1)")?;
        let rows = statement.query_map(["account_credentials"], |row| row.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    if columns.iter().any(|column| column == "profile")
        && !columns.iter().any(|column| column == "account")
    {
        transaction
            .execute_batch("ALTER TABLE account_credentials RENAME COLUMN profile TO account")?;
    }
    if object_type("provider_credentials")?.is_none() {
        transaction.execute_batch(
            "CREATE VIEW provider_credentials AS
                 SELECT account AS profile, credentials_json, updated_at FROM account_credentials",
        )?;
    }
    Ok(())
}

/// `sessions.approvals`, and the `ask` level's retirement.
///
/// `ask` rows become `none` with approvals on: every call asked before, and every call asks now. A
/// root row with no level adopts `context.default_permission`, which is what the process
/// would have run such a session at until now; a sub-agent's row is left alone, since its level
/// travels in `subagent_spec_json`. Each statement tests for the shape it converts from, so a
/// replay finds nothing to do.
fn sessions_carry_approvals(
    transaction: &rusqlite::Transaction<'_>,
    context: &Context,
) -> rusqlite::Result<()> {
    let columns: Vec<String> = {
        let mut statement = transaction.prepare("SELECT name FROM pragma_table_info(?1)")?;
        let rows = statement.query_map(["sessions"], |row| row.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    if !columns.iter().any(|column| column == "approvals") {
        transaction.execute_batch(
            "ALTER TABLE sessions ADD COLUMN approvals INTEGER NOT NULL DEFAULT 0",
        )?;
    }
    transaction.execute(
        "UPDATE sessions SET permission = 'none', approvals = 1 WHERE permission = 'ask'",
        [],
    )?;
    if !context.default_permission.is_empty() {
        transaction.execute(
            "UPDATE sessions SET permission = ?1
             WHERE permission IS NULL
               AND parent_session_id IS NULL
               AND subagent_spec_json IS NULL",
            [&context.default_permission],
        )?;
    }
    Ok(())
}

/// The turn's context block becomes its own content block on every stored user turn.
///
/// Until 0.46 a turn's user message was one text: meka's `<context>…</context>` preamble, a blank
/// line, then the words as typed, and every reader that wanted the words cut the preamble off
/// again by looking for the closing tag. The block is typed now, and the split is made once here: a
/// text that opens with the preamble is divided at its closing tag into a `turn_context` block and
/// a `text` block, and the row moves to the JSON shape (`user_blocks`) that carries more than
/// text. A row that never carried the preamble, or already opens with a `turn_context` block, is
/// left alone, so a replay finds nothing to do.
fn user_turns_carry_their_context_as_a_block(
    transaction: &rusqlite::Transaction<'_>,
) -> rusqlite::Result<()> {
    let rows: Vec<(i64, String, String)> = {
        let mut statement = transaction.prepare(
            "SELECT id, role, content FROM messages
             WHERE (role = 'user' AND content LIKE '<context>%')
                OR (role = 'user_blocks'
                    AND content LIKE '[{\"type\":\"text\",\"text\":\"<context>%')",
        )?;
        let rows = statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    for (id, role, content) in rows {
        let blocks = if role == "user" {
            match split_context_preamble(&content) {
                Some(blocks) => blocks,
                None => continue,
            }
        } else {
            let Ok(mut existing) = serde_json::from_str::<Vec<serde_json::Value>>(&content) else {
                continue;
            };
            let Some(split) = existing
                .first()
                .filter(|first| first["type"] == "text")
                .and_then(|first| first["text"].as_str())
                .and_then(split_context_preamble)
            else {
                continue;
            };
            existing.splice(0..1, split);
            existing
        };
        let encoded = match serde_json::to_string(&blocks) {
            Ok(encoded) => encoded,
            Err(error) => {
                return Err(rusqlite::Error::ToSqlConversionFailure(Box::new(error)));
            }
        };
        transaction.execute(
            "UPDATE messages SET role = 'user_blocks', content = ?1 WHERE id = ?2",
            rusqlite::params![encoded, id],
        )?;
    }
    Ok(())
}

/// A text that opens with the old inline preamble, as the two blocks it becomes, or `None` for a
/// text that does not.
fn split_context_preamble(text: &str) -> Option<Vec<serde_json::Value>> {
    const OPENING: &str = "<context>\n";
    const CLOSING: &str = "</context>";
    if !text.starts_with(OPENING) {
        return None;
    }
    let end = text.find(CLOSING)? + CLOSING.len();
    let context = &text[..end];
    let words = text[end..].trim_start_matches('\n');
    let mut blocks = vec![serde_json::json!({"type": "turn_context", "text": context})];
    if !words.is_empty() {
        blocks.push(serde_json::json!({"type": "text", "text": words}));
    }
    Some(blocks)
}

/// Image bytes move out of message rows into `blobs`, referenced by content hash.
///
/// Every `{"type":"base64","media_type":…,"data":…}` object in a row's JSON, wherever the envelope
/// puts it, becomes `{"type":"blob","hash":…,"media_type":…,"size":…}` and its bytes one row in
/// `blobs`, so the same screenshot read twice is stored once. `message_blobs` records which rows
/// reference which bytes, which is what the sweep after a session delete reads. A payload that is
/// not base64 stays where it is; a row already in the reference shape has nothing left to move.
fn images_live_in_blobs(transaction: &rusqlite::Transaction<'_>) -> rusqlite::Result<()> {
    transaction.execute_batch(
        "CREATE TABLE IF NOT EXISTS blobs (
             hash TEXT PRIMARY KEY,
             media_type TEXT NOT NULL,
             bytes BLOB NOT NULL,
             size INTEGER NOT NULL,
             created_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS message_blobs (
             message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
             hash TEXT NOT NULL REFERENCES blobs(hash),
             PRIMARY KEY (message_id, hash)
         );
         CREATE INDEX IF NOT EXISTS idx_message_blobs_hash ON message_blobs(hash);",
    )?;
    let rows: Vec<(i64, String)> = {
        let mut statement = transaction.prepare(
            "SELECT id, content FROM messages WHERE content LIKE '%\"type\":\"base64\"%'",
        )?;
        let rows = statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    let now = chrono::Utc::now().to_rfc3339();
    for (id, content) in rows {
        let Ok(mut value) = serde_json::from_str::<serde_json::Value>(&content) else {
            continue;
        };
        let mut blobs = Vec::new();
        if !move_inline_images(&mut value, &mut blobs) {
            continue;
        }
        for (hash, media_type, bytes) in &blobs {
            transaction.execute(
                "INSERT OR IGNORE INTO blobs (hash, media_type, bytes, size, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![hash, media_type, bytes, bytes.len() as i64, now],
            )?;
            transaction.execute(
                "INSERT OR IGNORE INTO message_blobs (message_id, hash) VALUES (?1, ?2)",
                rusqlite::params![id, hash],
            )?;
        }
        let encoded = match serde_json::to_string(&value) {
            Ok(encoded) => encoded,
            Err(error) => {
                return Err(rusqlite::Error::ToSqlConversionFailure(Box::new(error)));
            }
        };
        transaction.execute(
            "UPDATE messages SET content = ?1 WHERE id = ?2",
            rusqlite::params![encoded, id],
        )?;
    }
    Ok(())
}

/// Replace every inline image object under `value` with its reference, collecting the bytes.
/// Returns whether anything changed.
fn move_inline_images(
    value: &mut serde_json::Value,
    blobs: &mut Vec<(String, String, Vec<u8>)>,
) -> bool {
    use base64::Engine;
    use sha2::Digest;
    match value {
        serde_json::Value::Object(map) => {
            // Only the `source` of an image block. A tool call's arguments or a text block could
            // carry an object of the same shape, and those are the model's words, not an image
            // meka stored: rewriting them would leave a reference no reader ever resolves.
            let is_image = map.get("type").and_then(|kind| kind.as_str()) == Some("image");
            let inline = is_image
                && map.get("source").is_some_and(|source| {
                    source.get("type").and_then(|kind| kind.as_str()) == Some("base64")
                        && source.get("data").is_some_and(|data| data.is_string())
                        && source
                            .get("media_type")
                            .is_some_and(|media| media.is_string())
                });
            if inline {
                let source = &map["source"];
                let data = source["data"].as_str().unwrap_or_default();
                let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(data.as_bytes())
                else {
                    return false;
                };
                let media_type = source["media_type"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let digest = sha2::Sha256::digest(&bytes);
                let mut hash = String::with_capacity(64);
                for byte in digest {
                    hash.push_str(&format!("{byte:02x}"));
                }
                let size = bytes.len();
                blobs.push((hash.clone(), media_type.clone(), bytes));
                map.insert(
                    "source".to_string(),
                    serde_json::json!({
                        "type": "blob",
                        "hash": hash,
                        "media_type": media_type,
                        "size": size,
                    }),
                );
                return true;
            }
            let mut changed = false;
            for child in map.values_mut() {
                changed |= move_inline_images(child, blobs);
            }
            changed
        }
        serde_json::Value::Array(items) => {
            let mut changed = false;
            for item in items {
                changed |= move_inline_images(item, blobs);
            }
            changed
        }
        _ => false,
    }
}

/// Three columns and four indexes take the names one rule gives them, and the JSON in two columns
/// takes the serde shapes the rest of meka writes.
///
/// `gate_spec` holds JSON and takes the `_json` suffix its siblings carry; `claimed_until` names an
/// instant and takes `_at`; `memories.recorded_at` is the row's creation stamp and is `created_at`
/// like every other table's. An index is `idx_<table>_<columns>`. A thinking block's `opaque`
/// object is tagged `type` like every other tagged object in a message, and a gate's pointer test
/// is spelled `not_empty`, the one value in a stored gate spec that kebab-case and snake_case spell
/// differently.
///
/// Each rename is guarded on what is there and each JSON rewrite tests for the shape it converts
/// from, so a replay finds nothing to do. The frozen 0.43 step puts `gate_spec` and `claimed_until`
/// back, empty, on a replay; [`rename_column`] drops them again. The store a renumbered build left
/// never gained `gate_spec` at all, and has no gate to respell.
fn columns_follow_one_naming_rule(transaction: &rusqlite::Transaction<'_>) -> rusqlite::Result<()> {
    let has_gate_spec_json =
        rename_column(transaction, "scheduled_jobs", "gate_spec", "gate_spec_json")?;
    rename_column(
        transaction,
        "scheduled_jobs",
        "claimed_until",
        "claim_expires_at",
    )?;
    rename_column(transaction, "memories", "recorded_at", "created_at")?;
    transaction.execute_batch(
        "DROP INDEX IF EXISTS idx_sessions_parent;
         CREATE INDEX IF NOT EXISTS idx_sessions_parent_session_id ON sessions(parent_session_id);
         DROP INDEX IF EXISTS idx_scheduled_jobs_next_fire;
         CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_next_fire_at ON scheduled_jobs(next_fire_at);
         DROP INDEX IF EXISTS idx_scheduled_jobs_session;
         CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_session_id ON scheduled_jobs(session_id);
         DROP INDEX IF EXISTS memories_rank;
         CREATE INDEX IF NOT EXISTS idx_memories_rank ON memories(priority, created_at DESC);",
    )?;
    retag_opaque_reasoning(transaction)?;
    if has_gate_spec_json {
        respell_pointer_tests(transaction)?;
    }
    Ok(())
}

/// `ALTER TABLE … RENAME COLUMN`, guarded on the table's current columns so a replay finds nothing
/// to do. Reports whether the table has `to` afterwards, which is false only for a store that never
/// gained `from`.
///
/// Both names present means a frozen step added the old one back, empty, on a replay that had
/// dropped `user_version`, which is what the 0.43 step does to `gate_spec` and `claimed_until`. It
/// is dropped again so the store converges on the head shape instead of being refused on every
/// later start for carrying a column head does not have.
fn rename_column(
    transaction: &rusqlite::Transaction<'_>,
    table: &str,
    from: &str,
    to: &str,
) -> rusqlite::Result<bool> {
    let columns: Vec<String> = {
        let mut statement = transaction.prepare("SELECT name FROM pragma_table_info(?1)")?;
        let rows = statement.query_map([table], |row| row.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    let has = |column: &str| columns.iter().any(|existing| existing == column);
    match (has(from), has(to)) {
        (true, false) => {
            transaction
                .execute_batch(&format!("ALTER TABLE {table} RENAME COLUMN {from} TO {to}"))?;
            Ok(true)
        }
        (true, true) => {
            transaction.execute_batch(&format!("ALTER TABLE {table} DROP COLUMN {from}"))?;
            Ok(true)
        }
        (false, present) => Ok(present),
    }
}

/// A thinking block's `opaque` object, tagged `kind` until 0.46, is retagged `type`.
///
/// Only a top-level thinking block whose `opaque` carries a string `kind` and no `type` is
/// touched: a tool call's arguments are the model's words, whatever their shape. The tag keeps its
/// position, so a converted row is byte for byte what a 0.46 write produces.
fn retag_opaque_reasoning(transaction: &rusqlite::Transaction<'_>) -> rusqlite::Result<()> {
    let rows: Vec<(i64, String)> = {
        let mut statement = transaction
            .prepare("SELECT id, content FROM messages WHERE content LIKE '%\"opaque\"%'")?;
        let rows = statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    for (id, content) in rows {
        let Ok(mut blocks) = serde_json::from_str::<Vec<serde_json::Value>>(&content) else {
            continue;
        };
        let mut changed = false;
        for block in &mut blocks {
            if block.get("type").and_then(|kind| kind.as_str()) != Some("thinking") {
                continue;
            }
            let Some(opaque) = block
                .get_mut("opaque")
                .and_then(|opaque| opaque.as_object_mut())
            else {
                continue;
            };
            if opaque.contains_key("type") || !opaque.get("kind").is_some_and(|tag| tag.is_string())
            {
                continue;
            }
            let mut retagged = serde_json::Map::new();
            for (key, value) in opaque.iter() {
                let key = if key == "kind" { "type" } else { key.as_str() };
                retagged.insert(key.to_string(), value.clone());
            }
            *opaque = retagged;
            changed = true;
        }
        if !changed {
            continue;
        }
        let encoded = match serde_json::to_string(&blocks) {
            Ok(encoded) => encoded,
            Err(error) => {
                return Err(rusqlite::Error::ToSqlConversionFailure(Box::new(error)));
            }
        };
        transaction.execute(
            "UPDATE messages SET content = ?1 WHERE id = ?2",
            rusqlite::params![encoded, id],
        )?;
    }
    Ok(())
}

/// A gate's pointer test spelled `not-empty`, as kebab-case wrote it until 0.46, becomes
/// `not_empty`. Only `when.at.is` is read, so a command that happens to contain the word is left
/// alone.
fn respell_pointer_tests(transaction: &rusqlite::Transaction<'_>) -> rusqlite::Result<()> {
    let rows: Vec<(String, String)> = {
        let mut statement = transaction.prepare(
            "SELECT id, gate_spec_json FROM scheduled_jobs WHERE gate_spec_json LIKE '%not-empty%'",
        )?;
        let rows = statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    for (id, spec) in rows {
        let Ok(mut value) = serde_json::from_str::<serde_json::Value>(&spec) else {
            continue;
        };
        let Some(test) = value.pointer_mut("/when/at/is") else {
            continue;
        };
        if test.as_str() != Some("not-empty") {
            continue;
        }
        *test = serde_json::Value::String("not_empty".to_string());
        let encoded = match serde_json::to_string(&value) {
            Ok(encoded) => encoded,
            Err(error) => {
                return Err(rusqlite::Error::ToSqlConversionFailure(Box::new(error)));
            }
        };
        transaction.execute(
            "UPDATE scheduled_jobs SET gate_spec_json = ?2 WHERE id = ?1",
            rusqlite::params![id, encoded],
        )?;
    }
    Ok(())
}

/// A `repair` row's thinking blocks take the `type` tag [`retag_opaque_reasoning`] gave every other
/// row's.
///
/// That step read each row as a list of blocks and passed over what it could not read as one. A
/// `repair` row is not a list: it is an externally tagged event, `{"Repair":{"replaced_count":..,
/// "messages":[..]}}`, and the messages it carries keep the thinking blocks of the rounds they
/// replaced. Left tagged `kind`, the row no longer decodes, `load_events` drops it, and the
/// conversation replays the very messages the provider had rejected. A `compact_boundary` row's
/// summary is a user message meka wrote and carries no thinking; a `redact` row carries no
/// messages at all.
///
/// The same rule one level down, restated rather than shared: the step that first applied it is
/// frozen, and a helper the two called would change its digest. Only a top-level thinking block
/// whose `opaque` carries a string `kind` and no `type` is touched, so a replay and a row already
/// in the new shape find nothing to do, and a tool call's arguments are the model's words whatever
/// their shape.
fn repair_rows_retag_their_thinking(
    transaction: &rusqlite::Transaction<'_>,
) -> rusqlite::Result<()> {
    let rows: Vec<(i64, String)> = {
        let mut statement = transaction
            .prepare("SELECT id, content FROM messages WHERE content LIKE '%\"opaque\"%'")?;
        let rows = statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    for (id, content) in rows {
        let Ok(mut event) = serde_json::from_str::<serde_json::Value>(&content) else {
            continue;
        };
        let Some(messages) = event
            .get_mut("Repair")
            .and_then(|repair| repair.get_mut("messages"))
            .and_then(|messages| messages.as_array_mut())
        else {
            continue;
        };
        let mut changed = false;
        for message in messages {
            let Some(blocks) = message
                .get_mut("content")
                .and_then(|content| content.as_array_mut())
            else {
                continue;
            };
            for block in blocks {
                changed |= retag_thinking_block(block);
            }
        }
        if !changed {
            continue;
        }
        let encoded = match serde_json::to_string(&event) {
            Ok(encoded) => encoded,
            Err(error) => {
                return Err(rusqlite::Error::ToSqlConversionFailure(Box::new(error)));
            }
        };
        transaction.execute(
            "UPDATE messages SET content = ?1 WHERE id = ?2",
            rusqlite::params![encoded, id],
        )?;
    }
    Ok(())
}

/// Retag one thinking block's `opaque` object from `kind` to `type`, keeping the tag's position so
/// the block is byte for byte what a 0.46 write produces. Reports whether anything changed.
fn retag_thinking_block(block: &mut serde_json::Value) -> bool {
    if block.get("type").and_then(|kind| kind.as_str()) != Some("thinking") {
        return false;
    }
    let Some(opaque) = block
        .get_mut("opaque")
        .and_then(|opaque| opaque.as_object_mut())
    else {
        return false;
    };
    if opaque.contains_key("type") || !opaque.get("kind").is_some_and(|tag| tag.is_string()) {
        return false;
    }
    let mut retagged = serde_json::Map::new();
    for (key, value) in opaque.iter() {
        let key = if key == "kind" { "type" } else { key.as_str() };
        retagged.insert(key.to_string(), value.clone());
    }
    *opaque = retagged;
    true
}

/// A root row that recorded no level takes `[permissions].default` once `config.toml` reads.
///
/// [`sessions_carry_approvals`] stamps such a row only when the caller could read the file, and
/// the ledger moved past it either way, so a launch against an unreadable config left the row
/// without a level and nothing came back for it. This step keeps the frozen step's statement and
/// takes the refusal [`sessions_name_their_provider`] makes for a profile: with a row to stamp and
/// no file to take the level from, it errors, which aborts the transaction and leaves the store as
/// it was for a later run that can read the file. A sub-agent's row is left alone, since its level
/// travels in `subagent_spec_json`, and a caller that read the file and names no level leaves the
/// rows as the frozen step did. Only a row with nothing is written, so a replay finds nothing to
/// do.
fn root_rows_take_the_default_level_once_the_config_reads(
    transaction: &rusqlite::Transaction<'_>,
    context: &Context,
) -> rusqlite::Result<()> {
    let bare: i64 = transaction.query_row(
        "SELECT COUNT(*) FROM sessions
         WHERE permission IS NULL
           AND parent_session_id IS NULL
           AND subagent_spec_json IS NULL",
        [],
        |row| row.get(0),
    )?;
    if bare == 0 {
        return Ok(());
    }
    if context.config_unreadable {
        // `InvalidParameterName` for the reason the 0.44 step gives: the one variant that carries
        // an arbitrary message without enabling a rusqlite feature for a single error path.
        return Err(rusqlite::Error::InvalidParameterName(format!(
            "cannot record a level for {bare} root session(s) while config.toml cannot be read; \
             fix the file and start meka again"
        )));
    }
    if !context.default_permission.is_empty() {
        transaction.execute(
            "UPDATE sessions SET permission = ?1
             WHERE permission IS NULL
               AND parent_session_id IS NULL
               AND subagent_spec_json IS NULL",
            [&context.default_permission],
        )?;
    }
    Ok(())
}

/// The profile named by the only stored credential, when there is exactly one.
///
/// Ambiguity is not resolved by picking: with two credentials there is no evidence here about which
/// of them ran any given session, and a confident wrong answer is worse than an admitted absence.
fn sole_credential(transaction: &rusqlite::Transaction<'_>) -> rusqlite::Result<Option<String>> {
    let mut statement = transaction.prepare("SELECT profile FROM provider_credentials LIMIT 2")?;
    let profiles = statement
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<String>>>()?;
    match profiles.as_slice() {
        [only] => Ok(Some(only.clone())),
        _ => Ok(None),
    }
}

fn predicate_for(fire: &str) -> Option<&'static str> {
    match fire {
        "on-change" => Some("changed"),
        "on-success" => Some("succeeded"),
        _ => None,
    }
}

/// Build a store at head, for a test that needs the schema without a whole
/// [`crate::store::Store`].
///
/// Deliberately [`plan`] then [`apply`], the same pair production calls, rather than a private loop
/// over `MIGRATIONS`. A second way to build the schema is a second thing that can be right while
/// the real one is wrong, and the tests that would notice are exactly the ones using this.
#[cfg(test)]
pub(crate) fn create_for_test(connection: &mut rusqlite::Connection) -> Result<()> {
    let plan = plan(connection)?;
    apply(connection, plan, &Context::default())
}

/// The baseline DDL, for a test that needs to plant a store shaped the way an older meka left one.
#[cfg(test)]
pub(crate) fn baseline_for_test() -> &'static str {
    BASELINE_0_42
}

#[cfg(test)]
mod tests {
    use super::*;

    fn connection_for_test() -> rusqlite::Connection {
        rusqlite::Connection::open_in_memory().expect("an in-memory database")
    }

    /// The schema as SQLite understands it, rather than as it was typed.
    ///
    /// Compared structurally because the two paths this file cares about cannot produce identical
    /// text even when they are identical schemas: one runs `CREATE TABLE`, the other runs that plus
    /// `ALTER TABLE ADD`/`DROP COLUMN`, and SQLite rewrites `sqlite_master.sql` differently for
    /// each. `pragma_table_info` answers what the columns actually are, which is the thing that has
    /// to match.
    fn fingerprint(connection: &rusqlite::Connection) -> String {
        let objects: Vec<(String, String)> = {
            let mut statement = connection
                .prepare(
                    "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' \
                     ORDER BY type, name",
                )
                .expect("sqlite_master is readable");
            let rows = statement
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .expect("sqlite_master rows");
            rows.collect::<rusqlite::Result<_>>()
                .expect("sqlite_master rows")
        };
        let mut lines = Vec::new();
        for (kind, name) in objects {
            lines.push(format!("{kind} {name}"));
            if kind != "table" {
                continue;
            }
            let mut statement = connection
                .prepare(
                    "SELECT cid, name, type, \"notnull\", ifnull(dflt_value, ''), pk \
                     FROM pragma_table_info(?1) ORDER BY cid",
                )
                .expect("pragma_table_info is queryable");
            let columns = statement
                .query_map([&name], |row| {
                    Ok(format!(
                        "  {} {} {} notnull={} default={} pk={}",
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, i64>(5)?,
                    ))
                })
                .expect("column rows")
                .collect::<rusqlite::Result<Vec<_>>>()
                .expect("column rows");
            lines.extend(columns);
        }
        lines.join("\n")
    }

    /// Bring a baseline store to where the binary before the named entry left it: every earlier
    /// step run, and stamped as such. Returns that version, which is what `plan` reports as `from`.
    ///
    /// By name rather than by `head - 1`, so a test about one entry keeps testing that entry after
    /// the next one is appended.
    fn stopped_before(connection: &mut rusqlite::Connection, name: &str) -> u32 {
        let position = MIGRATIONS
            .iter()
            .position(|migration| migration.name == name)
            .unwrap_or_else(|| panic!("no ledger entry named {name}"));
        {
            let transaction = connection.transaction().expect("transaction");
            for migration in MIGRATIONS.iter().take(position) {
                match &migration.step {
                    Step::Sql(sql) => transaction.execute_batch(sql),
                    Step::Rust(run) => run(&transaction),
                    Step::Contextual(run) => run(&transaction, &Context::adopting(Some("p"))),
                }
                .unwrap_or_else(|error| panic!("{} failed: {error}", migration.name));
            }
            transaction.commit().expect("commit");
        }
        connection
            .execute_batch(&format!("PRAGMA user_version = {position};"))
            .expect("stamp the version that binary would have written");
        position as u32
    }

    /// A store as the 0.42 binary left one: the baseline shape, and nothing in `user_version`.
    fn store_as_0_42_left_it() -> rusqlite::Connection {
        let connection = connection_for_test();
        connection
            .execute_batch(BASELINE_0_42)
            .expect("the baseline builds");
        connection
    }

    fn plant_job(
        connection: &rusqlite::Connection,
        id: &str,
        command: Option<&str>,
        fire: Option<&str>,
    ) {
        connection
            .execute(
                "INSERT INTO sessions (id, created_at, updated_at) VALUES ('s', 'now', 'now') \
                 ON CONFLICT(id) DO NOTHING",
                [],
            )
            .expect("a session to hang the job from");
        connection
            .execute(
                "INSERT INTO scheduled_jobs \
                 (id, session_id, kind, spec, prompt, gate_command, gate_fire, gate_permission, \
                  created_at, next_fire_at) \
                 VALUES (?1, 's', 'every', '30s', 'p', ?2, ?3, 'unrestricted', 'now', 'later')",
                rusqlite::params![id, command, fire],
            )
            .expect("a planted job");
    }

    fn gate_of(connection: &rusqlite::Connection, id: &str) -> (Option<String>, Option<String>) {
        connection
            .query_row(
                "SELECT gate_kind, gate_spec_json FROM scheduled_jobs WHERE id = ?1",
                [id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("the row survives")
    }

    /// A fresh install runs the whole chain; an existing store starts partway along it. Those are
    /// two different paths through the same ledger and they must land in the same place. What this
    /// catches is a migration that behaves differently depending on whether the objects it touches
    /// were created a moment ago or were already there: `gates_become_kind_and_spec` has exactly
    /// such a branch in its `has("gate_command")` early return, and a future step guarded the same
    /// way is the likely place for it to happen again.
    ///
    /// It does **not** catch a column added to `BASELINE_0_42` instead of appended as a new
    /// migration, because the 0.42 fixture below is built from that same constant, so an edit lands
    /// in both paths and they agree. Verified by making that exact edit: this test passed and
    /// `the_ledger_is_append_only` failed. That one is the guard for edits; this one is the guard
    /// for divergence.
    #[test]
    fn a_fresh_store_and_an_upgraded_one_have_the_same_schema() {
        let mut built_from_scratch = connection_for_test();
        create_for_test(&mut built_from_scratch).expect("a fresh store reaches head");

        let mut carried_forward = store_as_0_42_left_it();
        let plan = plan(&carried_forward).expect("a 0.42 store is classified");
        assert_eq!(plan.from, 1, "the baseline is version 1, not a fresh store");
        apply(&mut carried_forward, plan, &Context::default()).expect("the remaining steps apply");

        assert_eq!(
            fingerprint(&built_from_scratch),
            fingerprint(&carried_forward),
            "a store built by the whole chain and one carried forward through part of it must end \
             up with the same schema"
        );
    }

    /// Every stopping point in the ledger resumes to the shape a fresh store gets.
    ///
    /// What it does catch: a step that fails or converts differently depending on whether the
    /// objects it touches were made a moment ago by an earlier step or were already there. Every
    /// guarded step is a candidate, and the guards are why this file has so many of them.
    ///
    /// **What it cannot catch is renumbering.** The intermediate store is built by running
    /// `MIGRATIONS[..stopped_at]`, the *current* list, so "a store at version N" means whatever
    /// this build says N is. Delete an entry and the yardstick moves with the ledger: prefix and
    /// suffix still compose to the whole list, the fingerprints still match, and this test passes.
    /// Verified by re-introducing the exact deletion that caused the outage; this test was green
    /// and only [`the_ledger_is_append_only`] failed.
    ///
    /// That is the general shape of every test in this file bar one: the "before" state is built
    /// from the code under test, so no test here can see a defect that depends on what a *previous*
    /// binary left behind. The single guard against renumbering is the hand-written name list in
    /// [`the_ledger_is_append_only`], and it is hand-written for exactly that reason.
    #[test]
    fn every_reachable_version_converges_on_the_fresh_shape() {
        let head = MIGRATIONS.len() as u32;

        let mut reference = store_as_0_42_left_it();
        let reference_plan = plan(&reference).expect("classified");
        apply(
            &mut reference,
            reference_plan,
            &Context::adopting(Some("p")),
        )
        .expect("migrated");
        let expected = fingerprint(&reference);

        for stopped_at in 0..=head {
            // A store as a binary carrying only the first `stopped_at` steps would have left it.
            let mut store = store_as_0_42_left_it();
            {
                let transaction = store.transaction().expect("transaction");
                for migration in MIGRATIONS.iter().take(stopped_at as usize) {
                    match &migration.step {
                        Step::Sql(sql) => transaction.execute_batch(sql),
                        Step::Rust(run) => run(&transaction),
                        Step::Contextual(run) => run(&transaction, &Context::adopting(Some("p"))),
                    }
                    .unwrap_or_else(|error| {
                        panic!(
                            "building a store at {stopped_at} failed on {}: {error}",
                            migration.name
                        )
                    });
                }
                transaction.commit().expect("commit");
            }
            store
                .execute_batch(&format!("PRAGMA user_version = {stopped_at};"))
                .expect("stamp the version that binary would have written");

            // A store at 0 or 1 is classified by shape rather than believed; both are covered by
            // their own tests. What matters here is that the migration completes and converges.
            let step_plan = plan(&store).expect("a store at any reachable version is planned");
            apply(&mut store, step_plan, &Context::adopting(Some("p")))
                .unwrap_or_else(|error| panic!("a store at {stopped_at} must migrate: {error}"));

            assert_eq!(
                fingerprint(&store),
                expected,
                "a store stamped {stopped_at} did not converge on the fresh shape"
            );
            assert_eq!(
                user_version(&store).expect("read the version back"),
                head,
                "a store stamped {stopped_at} did not end up at head"
            );
        }
    }

    /// A store stamped above head is refused, not migrated.
    ///
    /// This fires for a store a *newer* meka wrote, and that is the only thing it is for. 0.44
    /// briefly relied on it for something else: `sessions_record_their_model_overrides` was deleted
    /// rather than reversed, on the reasoning that an unreleased entry belongs to nobody, and this
    /// refusal was supposed to catch the development stores that shortening the list left stranded
    /// above head. It caught nothing, because the stores that mattered were stamped *below* the new
    /// head, not above it: they skipped the step the hole renumbered past and reported success. The
    /// entry was restored and reversed by an appended step, so the list only ever grows and this
    /// test is back to guarding one thing.
    ///
    /// The refusal has to name the version and change nothing, because the remedy is the user's to
    /// choose: recreate the store, restore the pre-upgrade backup, or converge it by hand.
    #[test]
    fn a_store_stamped_above_head_is_refused_rather_than_migrated() {
        let connection = store_as_0_42_left_it();
        let head = MIGRATIONS.len() as u32;
        connection
            .execute_batch(&format!("PRAGMA user_version = {};", head + 1))
            .expect("stamp it as a newer meka would");

        let error = plan(&connection).expect_err("a store from the future must not be migrated");
        let message = error.to_string();
        assert!(
            message.contains(&(head + 1).to_string()) && message.contains(&head.to_string()),
            "the refusal names both versions so the user can tell which way to go: {message}"
        );
        assert!(
            message.contains("Nothing has been changed"),
            "and says it did not touch the store: {message}"
        );

        let still_there = user_version(&connection).expect("read the version back");
        assert_eq!(
            still_there,
            head + 1,
            "a refused plan leaves the version exactly as it found it"
        );
    }

    /// Editing a migration that has already shipped changes what new stores get and nothing else,
    /// because the users who ran it will never run it again. Reordering or renaming one does the
    /// same. None of that is visible downstream, so it is checked here rather than hoped for.
    ///
    /// A `Rust` step's body is not covered, and cannot be: it is a function pointer, so only its
    /// name and position are pinned. That matters more than it sounds, because
    /// `gates_become_kind_and_spec` carries five `ADD COLUMN` and two `DROP COLUMN` statements
    /// -- more DDL than anything here covers apart from the baseline. Editing one of those
    /// (a default, say) passes this test *and* the convergence test, since both paths run the
    /// edited step and therefore agree. The conversion tests below pin the gate JSON, not the
    /// DDL. Treat a `Rust` step's body as guarded by review alone.
    /// The pin, so a schema change is a deliberate edit here as well as in the ledger. Prints the
    /// value it found, which is what the failure message needs.
    #[test]
    fn the_head_schema_fingerprint_is_pinned() {
        let mut connection = rusqlite::Connection::open_in_memory().expect("open");
        let plan = plan(&connection).expect("plan");
        apply(&mut connection, plan, &Context::default()).expect("apply");
        let fingerprint = schema_fingerprint(&connection).expect("fingerprint");
        assert_eq!(
            fingerprint, HEAD_SCHEMA_FINGERPRINT,
            "the head schema changed; set HEAD_SCHEMA_FINGERPRINT to {fingerprint} in the same \
             change that appended the migration"
        );
    }

    /// A store that reached head through the ledger passes the guard. Its added columns sit at the
    /// end of each table, where `ALTER TABLE` put them, while a fresh baseline has them inline; a
    /// digest that included the position refused every store that had ever been migrated, which is
    /// every real one.
    #[test]
    fn a_store_migrated_from_the_baseline_passes_the_fingerprint() {
        let mut connection = store_as_0_42_left_it();
        let plan_before = plan(&connection).expect("plan");
        assert!(
            plan_before.has_work(),
            "the baseline store has steps to run"
        );
        apply(&mut connection, plan_before, &Context::default()).expect("apply");
        let plan_after = plan(&connection).expect("a migrated store is a current store");
        assert!(!plan_after.has_work());
        assert_eq!(
            schema_fingerprint(&connection).expect("fingerprint"),
            HEAD_SCHEMA_FINGERPRINT,
            "the migrated shape must digest like the fresh one"
        );
    }

    /// A table meka did not create is not meka's shape. Digesting every table in `sqlite_master`
    /// refused a store beside which a replication tool kept its own bookkeeping, and told the user
    /// to copy a `-wal` that would have changed nothing.
    #[test]
    fn a_foreign_table_beside_the_ledger_s_own_is_not_a_shape_change() {
        let mut connection = rusqlite::Connection::open_in_memory().expect("open");
        let plan_before = plan(&connection).expect("plan");
        apply(&mut connection, plan_before, &Context::default()).expect("apply");
        connection
            .execute_batch("CREATE TABLE _litestream_seq (id INTEGER PRIMARY KEY, seq INTEGER)")
            .expect("another tool's table");
        let plan_after = plan(&connection).expect("a foreign table is not the ledger's concern");
        assert!(!plan_after.has_work());
    }

    /// A store stamped at head with some other shape is refused, and the message says what to copy.
    #[test]
    fn a_head_stamped_store_with_another_shape_is_refused() {
        let mut connection = rusqlite::Connection::open_in_memory().expect("open");
        let plan_before = plan(&connection).expect("plan");
        apply(&mut connection, plan_before, &Context::default()).expect("apply");
        assert!(plan(&connection).is_ok(), "a store at head plans cleanly");
        connection
            .execute_batch("ALTER TABLE sessions ADD COLUMN stray TEXT")
            .expect("tamper");
        let error = plan(&connection).expect_err("a tampered head store is refused");
        assert!(
            error.to_string().contains("-wal"),
            "the refusal names the companion files: {error}"
        );
    }

    #[test]
    fn the_ledger_is_append_only() {
        // FNV-1a, written out rather than taken from `DefaultHasher`, whose output Rust explicitly
        // does not promise to keep stable across releases. A test that changes its own expectation
        // when the toolchain moves is not a guard.
        fn digest(input: &str) -> u64 {
            let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
            for byte in input.as_bytes() {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            hash
        }
        /// Every entry, in order, because **an entry is frozen once any store has run it** -- which
        /// is not the same as having shipped.
        ///
        /// Pinning the released prefix only, on the reasoning that an unreleased entry belongs to
        /// nobody, is not enough. It does not: a development store has run it, and `user_version`
        /// is a positional index, so removing an entry renumbers every entry after it and a store
        /// stamped between the hole and the new head skips a step it never ran while reporting a
        /// clean migration. That happened. `sessions_record_their_model_overrides` was deleted as
        /// "unreleased, therefore free", and a store at 4 silently lost
        /// `mcp_credentials_hold_every_kind`, then stamped itself current so nothing would revisit
        /// it. Every MCP connection failed with `no such table: mcp_credentials`.
        ///
        /// Still a **prefix** check, so appending stays legal and is the only legal move. Adding a
        /// migration means adding a line here, which is the point: it is a deliberate, reviewable
        /// act rather than a silent renumbering. If this fires for any other reason, append; never
        /// paste the new vector out of the failure, which re-freezes the edit alongside whatever
        /// was legitimately added and launders exactly what this guards.
        ///
        /// Re-pinned once, on 2026-09-06, when the digest began covering a Rust or Contextual
        /// step's body and its helpers; until then such an entry carried only its name, so its
        /// body was frozen by convention alone. The SQL entries kept their values, which is the
        /// check that nothing else moved.
        const SHIPPED: &[(&str, u64)] = &[
            ("baseline_0_42", 9890918125805624612_u64),
            ("gates_become_kind_and_spec", 14871176051668618256_u64),
            ("sessions_name_their_provider", 2839145663817881110_u64),
            (
                "sessions_record_their_model_overrides",
                16650669434822866619_u64,
            ),
            ("mcp_credentials_hold_every_kind", 5768695620423760250_u64),
            ("scheduled_jobs_forget_isolation", 17602669801808082090_u64),
            (
                "sessions_forget_their_model_overrides",
                6081320598876219933_u64,
            ),
            (
                "mcp_credentials_exist_on_every_store",
                10674060935313009008_u64,
            ),
            (
                "background_tasks_announce_before_they_deliver",
                12218580709004178328_u64,
            ),
            ("prompt_history_is_in_the_ledger", 7340959758223132754_u64),
            ("sessions_record_their_profile", 7263607531702221278_u64),
            ("credentials_belong_to_accounts", 10445015279463977740_u64),
            ("sessions_carry_approvals", 11421791245895291692_u64),
            (
                "user_turns_carry_their_context_as_a_block",
                10528060927322803169_u64,
            ),
            ("images_live_in_blobs", 12946522340706447291_u64),
            ("columns_follow_one_naming_rule", 12052506080733391844_u64),
            ("repair_rows_retag_their_thinking", 10024761171275002141_u64),
            (
                "root_rows_take_the_default_level_once_the_config_reads",
                12427590907437771145_u64,
            ),
            (
                "background_tasks_spell_canceled_with_one_l",
                10686117140380679488_u64,
            ),
        ];
        /// The text of the column-zero `fn name(` up to its closing brace, plus, in name order,
        /// every column-zero function it calls, recursively. What a Rust step does is its body and
        /// its helpers', so both are what freezing it means; a shared helper that changes changes
        /// every step that calls it, which is the signal.
        fn source_of(production: &str, name: &str) -> String {
            fn body_of<'a>(production: &'a str, name: &str) -> &'a str {
                let header = format!("\nfn {name}(");
                let start = production
                    .find(&header)
                    .unwrap_or_else(|| panic!("no column-zero `fn {name}` in migrations.rs"));
                let rest = &production[start + 1..];
                let end = rest
                    .find("\n}\n")
                    .expect("a column-zero function ends with a column-zero brace");
                &rest[..end + 2]
            }
            let mut wanted = vec![name.to_string()];
            let mut seen: Vec<String> = Vec::new();
            while let Some(current) = wanted.pop() {
                if seen.contains(&current) {
                    continue;
                }
                let body = body_of(production, &current);
                let mut rest = body;
                while let Some(open) = rest.find('(') {
                    let ident: String = rest[..open]
                        .chars()
                        .rev()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect();
                    if !ident.is_empty()
                        && ident != current
                        && !seen.contains(&ident)
                        && production.contains(&format!("\nfn {ident}("))
                    {
                        wanted.push(ident);
                    }
                    rest = &rest[open + 1..];
                }
                seen.push(current);
            }
            seen.sort();
            seen.iter()
                .map(|name| body_of(production, name))
                .collect::<Vec<_>>()
                .join("\n")
        }
        let production = include_str!("migrations.rs")
            .split("\nmod tests {")
            .next()
            .expect("splitting always yields a first part");
        let current: Vec<(&str, u64)> = MIGRATIONS
            .iter()
            .map(|migration| {
                let body = match &migration.step {
                    Step::Sql(sql) => digest(sql),
                    Step::Rust(_) | Step::Contextual(_) => {
                        digest(&source_of(production, migration.name))
                    }
                };
                (migration.name, digest(migration.name) ^ body)
            })
            .collect();
        assert!(
            current.len() >= SHIPPED.len(),
            "a shipped migration was removed; the ledger is append-only"
        );
        assert_eq!(
            &current[..SHIPPED.len()],
            SHIPPED,
            "a shipped migration changed. Append a *new* migration instead of editing one users \
             have already run; do not repair this by pasting the current values over `SHIPPED`"
        );
    }

    /// Rule 2 from the module docs, enforced rather than remembered. A migration that reached for
    /// `Gate::spec` would keep passing every test in the suite and quietly start doing something
    /// else the day those types are refactored.
    ///
    /// `super::` is checked as well as `crate::`, and that is not belt-and-braces. This module is a
    /// child of `session`, so every item in meka is reachable as `super::super::…`; a scan for
    /// `crate::` alone accepts `super::super::schedule::Gate::spec`, which is precisely the call
    /// the module docs single out as forbidden. Verified against the earlier form of this test,
    /// which passed it.
    ///
    /// Split on `mod tests` rather than on the first `#[cfg(test)]` for the same reason: that
    /// attribute also marks test-only helpers, and one placed above a migration would truncate the
    /// scanned region to nothing and pass vacuously.
    #[test]
    fn no_migration_calls_meka_s_own_code() {
        let source = include_str!("migrations.rs");
        // `"\nmod tests {"` rather than `"mod tests {"`: the module is declared at column zero, so
        // this cannot be truncated by a doc comment that happens to contain the literal. The
        // unanchored form could be, and the sanity check below would not have noticed, because it
        // anchors on a function that sits *above* where such a comment would go.
        let production = source
            .split("\nmod tests {")
            .next()
            .expect("splitting always yields a first part");
        assert!(
            production.contains("fn gates_become_kind_and_spec"),
            "the scanned region no longer covers the migrations, so this test proves nothing"
        );
        for line in production.lines() {
            let code = line.trim_start();
            if code.starts_with("//") || !(code.contains("crate::") || code.contains("super::")) {
                continue;
            }
            assert_eq!(
                code, "use crate::error::{MekaError, Result};",
                "a migration may not reach into meka's own code, by any path; see the module docs"
            );
        }
    }

    fn plant_session(connection: &rusqlite::Connection, id: &str) {
        connection
            .execute(
                "INSERT INTO sessions (id, created_at, updated_at) VALUES (?1, 'now', 'now')",
                [id],
            )
            .expect("a session to carry forward");
    }

    fn plant_credential(connection: &rusqlite::Connection, profile: &str) {
        connection
            .execute(
                "INSERT INTO provider_credentials (profile, credentials_json, updated_at) \
                 VALUES (?1, '{}', 'now')",
                [profile],
            )
            .expect("a credential");
    }

    /// The profile a session's row names once the whole ledger has run.
    fn provider_of(connection: &rusqlite::Connection, id: &str) -> String {
        connection
            .query_row("SELECT profile FROM sessions WHERE id = ?1", [id], |row| {
                row.get(0)
            })
            .expect("the row survives")
    }

    /// The two 0.46 renames: a 0.45 row's `provider` is read back as `profile`, and the credential
    /// stored under `provider_credentials(profile)` as `account_credentials(account)`. A replay
    /// over the renamed store keeps what the row says and drops the column the frozen 0.44 step
    /// adds back, rather than failing or re-stamping.
    #[test]
    fn the_profile_and_account_renames_land_and_replay_as_a_no_op() {
        fn columns(connection: &rusqlite::Connection) -> Vec<String> {
            connection
                .prepare("SELECT name FROM pragma_table_info('sessions')")
                .expect("prepare")
                .query_map([], |row| row.get(0))
                .expect("query")
                .collect::<rusqlite::Result<_>>()
                .expect("rows")
        }
        fn accounts(connection: &rusqlite::Connection) -> Vec<String> {
            connection
                .prepare("SELECT account FROM account_credentials ORDER BY 1")
                .expect("the renamed table with its renamed column")
                .query_map([], |row| row.get(0))
                .expect("query")
                .collect::<rusqlite::Result<_>>()
                .expect("rows")
        }
        let mut connection = store_as_0_42_left_it();
        plant_session(&connection, "carried");
        plant_credential(&connection, "work");
        let first = plan(&connection).expect("classified");
        apply(&mut connection, first, &Context::adopting(Some("work"))).expect("migrated");

        assert!(
            columns(&connection)
                .iter()
                .any(|column| column == "profile")
        );
        assert!(
            !columns(&connection)
                .iter()
                .any(|column| column == "provider")
        );
        assert_eq!(provider_of(&connection, "carried"), "work");
        assert_eq!(accounts(&connection), vec!["work".to_string()]);

        connection
            .execute_batch("PRAGMA user_version = 0;")
            .expect("the round trip that drops the version");
        let replayed = plan(&connection).expect("classified by shape");
        apply(&mut connection, replayed, &Context::adopting(Some("other")))
            .expect("the replay must not fail");
        assert_eq!(
            provider_of(&connection, "carried"),
            "work",
            "the row keeps what it recorded"
        );
        assert!(
            !columns(&connection)
                .iter()
                .any(|column| column == "provider"),
            "the column the frozen step re-added is dropped again"
        );
        assert_eq!(accounts(&connection), vec!["work".to_string()]);
    }

    /// The ordinary upgrade: whatever the caller resolved is what existing sessions adopt.
    #[test]
    fn an_existing_session_adopts_the_resolved_default() {
        let mut connection = store_as_0_42_left_it();
        plant_session(&connection, "carried");
        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::adopting(Some("work"))).expect("migrated");

        assert_eq!(provider_of(&connection, "carried"), "work");
    }

    /// With nothing resolved, one stored credential is the only evidence in the store about which
    /// profile ran these sessions, and it is better than admitting nothing.
    #[test]
    fn a_sole_credential_stands_in_when_no_default_resolves() {
        let mut connection = store_as_0_42_left_it();
        plant_session(&connection, "carried");
        plant_credential(&connection, "the-only-one");
        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::default()).expect("migrated");

        assert_eq!(provider_of(&connection, "carried"), "the-only-one");
    }

    /// Two credentials say nothing about which ran any given session, so nothing is claimed. The
    /// empty value is not special-cased anywhere: it simply does not name a configured profile.
    #[test]
    fn two_credentials_are_not_guessed_between() {
        let mut connection = store_as_0_42_left_it();
        plant_session(&connection, "carried");
        plant_credential(&connection, "one");
        plant_credential(&connection, "two");
        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::default()).expect("migrated");

        assert_eq!(
            provider_of(&connection, "carried"),
            "",
            "an ambiguous store is left saying nothing rather than guessing"
        );
    }

    /// The caller's answer beats the store's, because config is the current statement of intent and
    /// a credential can outlive the profile that used it.
    #[test]
    fn the_resolved_default_beats_a_sole_credential() {
        let mut connection = store_as_0_42_left_it();
        plant_session(&connection, "carried");
        plant_credential(&connection, "stale");
        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::adopting(Some("current"))).expect("migrated");

        assert_eq!(provider_of(&connection, "carried"), "current");
    }

    /// Rule 3. A store that lost its `user_version` replays every step, and this one must not fail
    /// with `duplicate column name` nor overwrite a session that has since moved profile.
    #[test]
    fn a_replay_neither_fails_nor_rewrites_a_chosen_provider() {
        let mut connection = store_as_0_42_left_it();
        plant_session(&connection, "carried");
        let first = plan(&connection).expect("classified");
        apply(&mut connection, first, &Context::adopting(Some("first"))).expect("migrated");

        connection
            .execute(
                "UPDATE sessions SET profile = 'chosen' WHERE id = 'carried'",
                [],
            )
            .expect("the user repins it");
        connection
            .execute_batch("PRAGMA user_version = 0;")
            .expect("the round trip that drops the version");

        let replayed = plan(&connection).expect("classified by shape");
        apply(
            &mut connection,
            replayed,
            &Context::adopting(Some("second")),
        )
        .expect("the replay must not fail");

        assert_eq!(
            provider_of(&connection, "carried"),
            "chosen",
            "a replay only fills what is empty"
        );
    }

    /// The level a session's row records once the whole ledger has run, and its approvals switch.
    fn level_of(connection: &rusqlite::Connection, id: &str) -> (Option<String>, i64) {
        connection
            .query_row(
                "SELECT permission, approvals FROM sessions WHERE id = ?1",
                [id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("the row survives")
    }

    /// `ask` was "every call asks", which is `none` with the switch on; a row that recorded no
    /// level adopts the caller's default, and a sub-agent's row does not.
    #[test]
    fn an_ask_session_becomes_none_with_approvals_and_a_bare_row_adopts_the_default() {
        let mut connection = store_as_0_42_left_it();
        plant_session(&connection, "asked");
        plant_session(&connection, "bare");
        plant_session(&connection, "sub-agent");
        connection
            .execute_batch(
                "UPDATE sessions SET permission = 'ask' WHERE id = 'asked';
                 UPDATE sessions SET parent_session_id = 'bare' WHERE id = 'sub-agent';",
            )
            .expect("shape the rows");
        let first = plan(&connection).expect("classified");
        apply(
            &mut connection,
            first,
            &Context::adopting(Some("p")).starting_at("read"),
        )
        .expect("migrated");

        assert_eq!(
            level_of(&connection, "asked"),
            (Some("none".to_string()), 1)
        );
        assert_eq!(level_of(&connection, "bare"), (Some("read".to_string()), 0));
        assert_eq!(
            level_of(&connection, "sub-agent"),
            (None, 0),
            "a sub-agent's level travels in its spawn terms, not on the row"
        );

        // The user then moves the session, and the round trip that drops the version replays the
        // step over it.
        connection
            .execute_batch(
                "UPDATE sessions SET permission = 'unrestricted', approvals = 0 WHERE id = 'asked'",
            )
            .expect("the user repins it");
        connection
            .execute_batch("PRAGMA user_version = 0;")
            .expect("the round trip that drops the version");
        let replayed = plan(&connection).expect("classified by shape");
        apply(
            &mut connection,
            replayed,
            &Context::adopting(Some("p")).starting_at("workspace"),
        )
        .expect("the replay must not fail");
        assert_eq!(
            level_of(&connection, "asked"),
            (Some("unrestricted".to_string()), 0),
            "a replay only converts what is still in the old shape"
        );
        assert_eq!(level_of(&connection, "bare"), (Some("read".to_string()), 0));
    }

    /// A `repair` row's thinking blocks are retagged where `columns_follow_one_naming_rule` passed
    /// them over, the row decodes again through the store, and a replay rewrites nothing.
    ///
    /// Through `Store::open` and `load_events` rather than the transaction alone, because what the
    /// defect cost was the decode: a row that no longer parsed was dropped with a warning, and the
    /// conversation replayed the messages the repair had withdrawn.
    #[tokio::test]
    async fn a_repair_row_decodes_again_once_its_thinking_is_retagged() {
        use crate::conversation::{ContentBlock, Event, OpaqueReasoning};

        let directory = tempfile::tempdir().expect("a directory for the store");
        let path = directory.path().join("meka.db");
        let session = uuid::Uuid::new_v4().to_string();
        let repair = r#"{"Repair":{"replaced_count":2,"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"hmm","opaque":{"kind":"signed","signature":"SIG"}},{"type":"text","text":"hi"}]}]}}"#;
        // Already in the new shape, and a tool call whose arguments happen to look like one: both
        // are left byte for byte as they were.
        let retagged = r#"{"Repair":{"replaced_count":1,"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"s","opaque":{"type":"sealed","encrypted_content":"E"}}]}]}}"#;
        let arguments = r#"{"Repair":{"replaced_count":1,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"memory_write","input":{"opaque":{"kind":"signed"}}}]}]}}"#;
        {
            let mut connection = rusqlite::Connection::open(&path).expect("open");
            connection
                .execute_batch(BASELINE_0_42)
                .expect("the baseline builds");
            stopped_before(&mut connection, "repair_rows_retag_their_thinking");
            plant_session(&connection, &session);
            plant_message(&connection, &session, "repair", repair);
            plant_message(&connection, &session, "repair", retagged);
            plant_message(&connection, &session, "repair", arguments);
        }

        let store = crate::store::Store::open(Some(&path), &Context::default())
            .await
            .expect("the store migrates on open");
        let events = store
            .load_events(session.parse().expect("a uuid"))
            .await
            .expect("load");
        let Some(Event::Repair {
            replaced_count: 2,
            messages,
        }) = events.first()
        else {
            panic!("the repair row must decode as the first event: {events:?}");
        };
        assert!(
            matches!(
                messages[0].content.first(),
                Some(ContentBlock::Thinking {
                    opaque: Some(OpaqueReasoning::Signed { signature }),
                    ..
                }) if signature == "SIG"
            ),
            "the repair's thinking block survives with its signature: {messages:?}"
        );
        assert_eq!(
            events.len(),
            3,
            "every repair row decodes, the two the step left alone included: {events:?}"
        );

        let mut connection = rusqlite::Connection::open(&path).expect("reopen");
        let after = stored_messages(&connection);
        assert_eq!(after[1].1, retagged, "a row in the new shape is untouched");
        assert_eq!(
            after[2].1, arguments,
            "a tool call's arguments are not a thinking block"
        );
        let transaction = connection.transaction().expect("transaction");
        repair_rows_retag_their_thinking(&transaction).expect("replay");
        transaction.commit().expect("commit");
        assert_eq!(
            stored_messages(&connection),
            after,
            "a replay finds nothing left in the old shape"
        );
    }

    fn plant_message(connection: &rusqlite::Connection, session: &str, role: &str, content: &str) {
        connection
            .execute(
                "INSERT INTO messages (session_id, role, content, created_at) \
                 VALUES (?1, ?2, ?3, 'now')",
                rusqlite::params![session, role, content],
            )
            .expect("a message to carry forward");
    }

    /// Every stored user message, as `(role, content)`, oldest first.
    fn stored_messages(connection: &rusqlite::Connection) -> Vec<(String, String)> {
        let mut statement = connection
            .prepare("SELECT role, content FROM messages ORDER BY id ASC")
            .expect("prepare");
        statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .expect("query")
            .collect::<rusqlite::Result<_>>()
            .expect("rows")
    }

    /// The inline preamble a 0.45 turn stored becomes a `turn_context` block ahead of the words,
    /// on a plain-text row and on one that already carried blocks; a row without the preamble and a
    /// converted row are left alone, so the replay is a no-op.
    #[test]
    fn a_stored_turn_s_preamble_becomes_its_own_block() {
        let mut connection = store_as_0_42_left_it();
        plant_session(&connection, "s");
        plant_message(
            &connection,
            "s",
            "user",
            "<context>\n[Permission context]\nCurrent permission level: read\n</context>\n\nhello \
             world",
        );
        plant_message(
            &connection,
            "s",
            "user_blocks",
            "[{\"type\":\"text\",\"text\":\"<context>\\nx\\n</context>\\n\\nwith image\"},\
             {\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\",\
             \"data\":\"QUJD\"}}]",
        );
        plant_message(&connection, "s", "user", "plain words");
        plant_message(
            &connection,
            "s",
            "user",
            "<context>\nonly context\n</context>\n\n",
        );
        let first = plan(&connection).expect("classified");
        apply(&mut connection, first, &Context::adopting(Some("p"))).expect("migrated");

        let after = stored_messages(&connection);
        let expected = vec![
            (
                "user_blocks".to_string(),
                "[{\"type\":\"turn_context\",\"text\":\"<context>\\n[Permission context]\\nCurrent \
                 permission level: read\\n</context>\"},{\"type\":\"text\",\"text\":\"hello world\"}]"
                    .to_string(),
            ),
            (
                "user_blocks".to_string(),
                // The image goes on to `images_live_in_blobs`, which is why the bytes ("ABC") are
                // a reference here rather than the payload the row was planted with.
                "[{\"type\":\"turn_context\",\"text\":\"<context>\\nx\\n</context>\"},{\"type\":\
                 \"text\",\"text\":\"with image\"},{\"type\":\"image\",\"source\":{\"type\":\
                 \"blob\",\"hash\":\"b5d4045c3f466fa91fe2cc6abe79232a1a57cdf104f7a26e716e0a1e2789df78\",\
                 \"media_type\":\"image/png\",\"size\":3}}]"
                    .to_string(),
            ),
            ("user".to_string(), "plain words".to_string()),
            (
                "user_blocks".to_string(),
                "[{\"type\":\"turn_context\",\"text\":\"<context>\\nonly context\\n</context>\"}]"
                    .to_string(),
            ),
        ];
        assert_eq!(after, expected);

        connection
            .execute_batch("PRAGMA user_version = 0;")
            .expect("the round trip that drops the version");
        let replayed = plan(&connection).expect("classified by shape");
        apply(&mut connection, replayed, &Context::adopting(Some("p"))).expect("replay");
        assert_eq!(
            stored_messages(&connection),
            expected,
            "a replay finds nothing left in the old shape"
        );
    }

    /// One image in two rows becomes one blob with two references, each row in the reference shape;
    /// a row with no image and a converted row are left alone, so the replay is a no-op.
    #[test]
    fn inline_images_move_into_blobs_once() {
        let mut connection = store_as_0_42_left_it();
        plant_session(&connection, "s");
        // "hello", base64.
        let inline = "{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"aGVsbG8=\"}";
        plant_message(
            &connection,
            "s",
            "user_blocks",
            &format!(
                "[{{\"type\":\"text\",\"text\":\"look\"}},{{\"type\":\"image\",\"source\":{inline}}}]"
            ),
        );
        plant_message(
            &connection,
            "s",
            "tool_results",
            &format!(
                "[{{\"type\":\"tool_result\",\"tool_use_id\":\"u1\",\"content\":[{{\"type\":\"image\",\
                 \"source\":{inline}}}],\"is_error\":false}}]"
            ),
        );
        plant_message(&connection, "s", "user", "no image here");
        // A tool call whose arguments happen to be shaped like an image source: the model's words,
        // not an image meka stored, so they must be left exactly as they were.
        let arguments = format!(
            "[{{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"scratchpad_write\",\"input\":{inline}}}]"
        );
        plant_message(&connection, "s", "assistant", &arguments);
        let first = plan(&connection).expect("classified");
        apply(&mut connection, first, &Context::adopting(Some("p"))).expect("migrated");

        let hash = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
        let (count, size, media_type): (i64, i64, String) = connection
            .query_row(
                "SELECT count(*), max(size), max(media_type) FROM blobs",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("blobs");
        assert_eq!((count, size, media_type.as_str()), (1, 5, "image/png"));
        let references: i64 = connection
            .query_row(
                "SELECT count(*) FROM message_blobs WHERE hash = ?1",
                [hash],
                |row| row.get(0),
            )
            .expect("references");
        assert_eq!(references, 2);
        let after = stored_messages(&connection);
        assert!(
            after.iter().take(2).all(|(_, content)| {
                content.contains(&format!("\"hash\":\"{hash}\"")) && !content.contains("aGVsbG8=")
            }),
            "both rows reference the blob and neither carries the bytes: {after:?}"
        );
        assert_eq!(after[2], ("user".to_string(), "no image here".to_string()));
        assert_eq!(
            after[3],
            ("assistant".to_string(), arguments),
            "a tool call's arguments are not an image, whatever their shape"
        );

        connection
            .execute_batch("PRAGMA user_version = 0;")
            .expect("the round trip that drops the version");
        let replayed = plan(&connection).expect("classified by shape");
        apply(&mut connection, replayed, &Context::adopting(Some("p"))).expect("replay");
        assert_eq!(stored_messages(&connection), after);
    }

    /// The 0.46 names land on a store shaped as 0.45 left it: three columns and four indexes are
    /// renamed, the JSON in two of the columns moves with them, and a replay over the renamed
    /// store, which puts `gate_spec` and `claimed_until` back through the frozen 0.43 step, drops
    /// them again and rewrites nothing.
    #[test]
    fn the_0_46_names_land_and_replay_as_a_no_op() {
        fn indexes(connection: &rusqlite::Connection) -> Vec<String> {
            connection
                .prepare("SELECT name FROM sqlite_master WHERE type = 'index' ORDER BY name")
                .expect("prepare")
                .query_map([], |row| row.get(0))
                .expect("query")
                .collect::<rusqlite::Result<_>>()
                .expect("rows")
        }
        fn job_spec(connection: &rusqlite::Connection, id: &str) -> (String, Option<String>) {
            connection
                .query_row(
                    "SELECT gate_spec_json, claim_expires_at FROM scheduled_jobs WHERE id = ?1",
                    [id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .expect("the row survives")
        }
        let mut connection = store_as_0_42_left_it();
        let before = stopped_before(&mut connection, "columns_follow_one_naming_rule");
        plant_session(&connection, "s");
        let watched = r#"{"tool":{"name":"t","arguments":{}},"when":{"at":{"pointer":"/chats","is":"not-empty"}}}"#;
        // A command that merely contains the word is the operator's text, not a pointer test.
        let plain = r#"{"shell":{"command":"echo not-empty"},"when":"changed"}"#;
        for (id, kind, spec) in [("watched", "tool", watched), ("plain", "shell", plain)] {
            connection
                .execute(
                    "INSERT INTO scheduled_jobs (id, session_id, kind, spec, prompt, gate_kind, \
                     gate_spec, gate_permission, claimed_by, claimed_until, created_at, \
                     next_fire_at) \
                     VALUES (?1, 's', 'every', '1h', 'p', ?2, ?3, 'read', 'host', 'later', 'now', \
                     'soon')",
                    rusqlite::params![id, kind, spec],
                )
                .expect("a gated job");
        }
        connection
            .execute(
                "INSERT INTO memories (name, description, recorded_at, updated_at) \
                 VALUES ('m', 'd', 'then', 'now')",
                [],
            )
            .expect("a memory");
        let signed = r#"[{"type":"thinking","thinking":"hmm","opaque":{"kind":"signed","signature":"SIG"}},{"type":"text","text":"hi"}]"#;
        let sealed = r#"[{"type":"thinking","thinking":"s","opaque":{"kind":"sealed","encrypted_content":"E","id":"rs_1"}}]"#;
        // The same shape inside a tool call's arguments is the model's, and stays as it was.
        let arguments = r#"[{"type":"tool_use","id":"t1","name":"memory_write","input":{"opaque":{"kind":"signed"}}}]"#;
        plant_message(&connection, "s", "assistant", signed);
        plant_message(&connection, "s", "assistant", sealed);
        plant_message(&connection, "s", "assistant", arguments);
        plant_message(&connection, "s", "user", "a plain row that says opaque");

        let first = plan(&connection).expect("planned");
        assert_eq!(
            first.from, before,
            "this entry and the ones after it are pending"
        );
        apply(&mut connection, first, &Context::default()).expect("migrated");

        let scheduled = table_columns(&connection, "scheduled_jobs").expect("columns");
        let memories = table_columns(&connection, "memories").expect("columns");
        for (table, gone, renamed) in [
            (&scheduled, "gate_spec", "gate_spec_json"),
            (&scheduled, "claimed_until", "claim_expires_at"),
            (&memories, "recorded_at", "created_at"),
        ] {
            assert!(!table.iter().any(|c| c == gone), "{gone} should be renamed");
            assert!(table.iter().any(|c| c == renamed), "{renamed} should exist");
        }
        assert_eq!(
            job_spec(&connection, "watched"),
            (
                r#"{"tool":{"name":"t","arguments":{}},"when":{"at":{"pointer":"/chats","is":"not_empty"}}}"#.to_string(),
                Some("later".to_string())
            ),
            "the pointer test is respelled and the lease carries over under its new name"
        );
        assert_eq!(job_spec(&connection, "plain").0, plain);
        let created: String = connection
            .query_row(
                "SELECT created_at FROM memories WHERE name = 'm'",
                [],
                |row| row.get(0),
            )
            .expect("the memory survives");
        assert_eq!(created, "then");
        let after = stored_messages(&connection);
        assert_eq!(
            after,
            vec![
                (
                    "assistant".to_string(),
                    r#"[{"type":"thinking","thinking":"hmm","opaque":{"type":"signed","signature":"SIG"}},{"type":"text","text":"hi"}]"#.to_string()
                ),
                (
                    "assistant".to_string(),
                    r#"[{"type":"thinking","thinking":"s","opaque":{"type":"sealed","encrypted_content":"E","id":"rs_1"}}]"#.to_string()
                ),
                ("assistant".to_string(), arguments.to_string()),
                ("user".to_string(), "a plain row that says opaque".to_string()),
            ]
        );
        let names = indexes(&connection);
        for renamed in [
            "idx_sessions_parent_session_id",
            "idx_scheduled_jobs_next_fire_at",
            "idx_scheduled_jobs_session_id",
            "idx_memories_rank",
        ] {
            assert!(names.iter().any(|n| n == renamed), "{renamed} should exist");
        }
        for gone in [
            "idx_sessions_parent",
            "idx_scheduled_jobs_next_fire",
            "idx_scheduled_jobs_session",
            "memories_rank",
        ] {
            assert!(!names.iter().any(|n| n == gone), "{gone} should be renamed");
        }
        let shape = fingerprint(&connection);

        connection
            .execute_batch("PRAGMA user_version = 0;")
            .expect("the round trip that drops the version");
        let replayed = plan(&connection).expect("classified by shape");
        apply(&mut connection, replayed, &Context::default()).expect("replay");
        assert_eq!(
            fingerprint(&connection),
            shape,
            "the columns the frozen step re-added are dropped again"
        );
        assert_eq!(stored_messages(&connection), after);
        assert_eq!(
            job_spec(&connection, "watched").1,
            Some("later".to_string())
        );
    }

    /// A caller that read the file and knows no default leaves a bare row bare rather than
    /// inventing a level; one that could not read the file is refused, below.
    #[test]
    fn a_bare_row_stays_bare_when_no_default_is_known() {
        let mut connection = store_as_0_42_left_it();
        plant_session(&connection, "bare");
        let first = plan(&connection).expect("classified");
        apply(&mut connection, first, &Context::adopting(Some("p"))).expect("migrated");
        assert_eq!(level_of(&connection, "bare"), (None, 0));
    }

    /// The store 0.46.0 left when launched against an unreadable config: the frozen approvals step
    /// ran, stamped nothing, and the ledger moved past it. A root row and a sub-agent's row that
    /// recorded no level, brought to the entry before the head one.
    fn store_the_approvals_step_passed_over() -> (rusqlite::Connection, u32) {
        let mut connection = store_as_0_42_left_it();
        plant_session(&connection, "bare");
        plant_session(&connection, "sub-agent");
        connection
            .execute_batch("UPDATE sessions SET parent_session_id = 'bare' WHERE id = 'sub-agent'")
            .expect("shape the rows");
        let from = stopped_before(
            &mut connection,
            "root_rows_take_the_default_level_once_the_config_reads",
        );
        assert_eq!(
            level_of(&connection, "bare"),
            (None, 0),
            "the frozen step must have left the root row bare for this to test anything"
        );
        (connection, from)
    }

    /// The gap the head step closes: a root row the approvals step passed over takes the file's
    /// default once a run can read it, a sub-agent's row does not, and a replay over the stamped
    /// store rewrites nothing.
    #[test]
    fn a_root_row_the_approvals_step_passed_over_takes_the_default_once_the_config_reads() {
        let (mut connection, _) = store_the_approvals_step_passed_over();
        let first = plan(&connection).expect("classified");
        apply(
            &mut connection,
            first,
            &Context::adopting(Some("p")).starting_at("read"),
        )
        .expect("migrated");
        assert_eq!(level_of(&connection, "bare"), (Some("read".to_string()), 0));
        assert_eq!(
            level_of(&connection, "sub-agent"),
            (None, 0),
            "a sub-agent's level travels in its spawn terms, not on the row"
        );

        connection
            .execute_batch("PRAGMA user_version = 0;")
            .expect("the round trip that drops the version");
        let replayed = plan(&connection).expect("classified by shape");
        apply(
            &mut connection,
            replayed,
            &Context::adopting(Some("p")).starting_at("workspace"),
        )
        .expect("the replay must not fail");
        assert_eq!(
            level_of(&connection, "bare"),
            (Some("read".to_string()), 0),
            "a replay only fills what is empty"
        );
        assert_eq!(level_of(&connection, "sub-agent"), (None, 0));
    }

    /// The 0.44 refusal, for a level: with a root row to stamp and no file to take the level from,
    /// the migration errors rather than advancing past the stamp, and the transaction it aborts
    /// takes `user_version` back with it. The sub-agent's row is not among the rows counted.
    #[test]
    fn a_bare_root_row_refuses_the_migration_while_the_config_cannot_be_read() {
        let (mut connection, from) = store_the_approvals_step_passed_over();
        let pending = plan(&connection).expect("classified");
        assert!(pending.has_work());
        let error = apply(&mut connection, pending, &Context::on_unreadable_config())
            .expect_err("a root row without a level refuses an unreadable config");
        assert!(
            error
                .to_string()
                .contains("cannot record a level for 1 root session(s)"),
            "{error}"
        );
        assert_eq!(
            user_version(&connection).expect("read the version back"),
            from,
            "the transaction rolled back, so the version must not have moved"
        );
        assert_eq!(level_of(&connection, "bare"), (None, 0));
    }

    /// With nothing to stamp there is nothing to refuse: a store whose root rows all record a
    /// level opens under an unreadable config, which is what keeps `meka profile remove` and its
    /// siblings able to repair the file. A sub-agent's row without a level is not a reason.
    #[test]
    fn a_store_with_no_bare_root_row_opens_while_the_config_cannot_be_read() {
        let mut connection = store_as_0_42_left_it();
        plant_session(&connection, "pinned");
        plant_session(&connection, "sub-agent");
        connection
            .execute_batch(
                "UPDATE sessions SET permission = 'read' WHERE id = 'pinned';
                 UPDATE sessions SET parent_session_id = 'pinned' WHERE id = 'sub-agent';",
            )
            .expect("shape the rows");
        stopped_before(
            &mut connection,
            "root_rows_take_the_default_level_once_the_config_reads",
        );
        let pending = plan(&connection).expect("classified");
        apply(&mut connection, pending, &Context::on_unreadable_config())
            .expect("nothing to stamp, so nothing to refuse");
        assert_eq!(
            user_version(&connection).expect("read the version back"),
            MIGRATIONS.len() as u32
        );
        assert_eq!(
            level_of(&connection, "pinned"),
            (Some("read".to_string()), 0)
        );
        assert_eq!(level_of(&connection, "sub-agent"), (None, 0));
    }

    /// The OAuth rows a 0.42 store holds arrive in the new table under kind `oauth`, and the table
    /// they came from is gone. Nothing outside this module may know it ever existed, which is only
    /// true if the copy is complete.
    #[test]
    fn every_oauth_credential_carries_over_and_the_old_table_goes() {
        let mut connection = store_as_0_42_left_it();
        connection
            .execute_batch(
                "INSERT INTO mcp_oauth_credentials (server_name, credentials_json, updated_at) \
                 VALUES ('docs', '{\"access_token\":\"at1\"}', '2026-01-01T00:00:00Z'), \
                        ('api',  '{\"access_token\":\"at2\"}', '2026-01-02T00:00:00Z')",
            )
            .expect("two authorized servers");

        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::adopting(Some("p"))).expect("migrated");

        let mut statement = connection
            .prepare("SELECT server_name, kind, secret, updated_at FROM mcp_credentials ORDER BY 1")
            .expect("the new table exists");
        let rows: Vec<(String, String, String, String)> = statement
            .query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })
            .expect("query")
            .collect::<rusqlite::Result<_>>()
            .expect("rows");
        assert_eq!(
            rows,
            vec![
                (
                    "api".to_string(),
                    "oauth".to_string(),
                    "{\"access_token\":\"at2\"}".to_string(),
                    "2026-01-02T00:00:00Z".to_string(),
                ),
                (
                    "docs".to_string(),
                    "oauth".to_string(),
                    "{\"access_token\":\"at1\"}".to_string(),
                    "2026-01-01T00:00:00Z".to_string(),
                ),
            ],
            "every row carries over, at kind oauth, with its timestamp"
        );

        let old: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master \
                 WHERE type = 'table' AND name = 'mcp_oauth_credentials'",
                [],
                |row| row.get(0),
            )
            .expect("query");
        assert_eq!(old, 0, "the table named for one kind is gone");
    }

    /// Rule 3, for the step that rebuilds a table rather than altering one. A bare `CREATE TABLE`
    /// here would fail with `table … already exists` and refuse the store on every start.
    #[test]
    fn a_replay_neither_fails_nor_discards_a_stored_credential() {
        let mut connection = store_as_0_42_left_it();
        connection
            .execute_batch(
                "INSERT INTO mcp_oauth_credentials (server_name, credentials_json, updated_at) \
                 VALUES ('docs', '{\"access_token\":\"at1\"}', '2026-01-01T00:00:00Z')",
            )
            .expect("one authorized server");
        let first = plan(&connection).expect("classified");
        apply(&mut connection, first, &Context::adopting(Some("first"))).expect("migrated");

        // A secret acquired after the migration, of a kind the old table could not hold.
        connection
            .execute_batch(
                "INSERT INTO mcp_credentials (server_name, kind, secret, updated_at) \
                 VALUES ('docs', 'client_secret', 'cs-not-a-real-secret', '2026-02-01T00:00:00Z')",
            )
            .expect("a client secret beside the bundle");
        connection
            .execute_batch("PRAGMA user_version = 0;")
            .expect("the round trip that drops the version");

        let replayed = plan(&connection).expect("classified by shape");
        apply(
            &mut connection,
            replayed,
            &Context::adopting(Some("second")),
        )
        .expect("the replay must not fail");

        let kinds: Vec<String> = connection
            .prepare("SELECT kind FROM mcp_credentials WHERE server_name = 'docs' ORDER BY 1")
            .expect("prepare")
            .query_map([], |row| row.get(0))
            .expect("query")
            .collect::<rusqlite::Result<_>>()
            .expect("rows");
        assert_eq!(
            kinds,
            vec!["client_secret".to_string(), "oauth".to_string()],
            "a replay leaves both secrets where they were"
        );
    }

    /// The store the broken build actually left, rebuilt by hand rather than by this ledger.
    ///
    /// Every other test here derives its "before" from the code under test, which is why none of
    /// them could see the renumbering and why none of them reaches
    /// [`mcp_credentials_exist_on_every_store`]'s body: by the time index 7 runs, index 4 has
    /// always just created the table. This one writes the damaged shape out in SQL -- stamped past
    /// the step, `mcp_credentials` absent, `mcp_oauth_credentials` still holding a bundle -- so the
    /// repair has something to repair. Without it the `CREATE TABLE`, the `INSERT … SELECT` and the
    /// `DROP TABLE` are all dead under test, and a wrong column name or a mistyped `kind` would
    /// lose every user's OAuth bundle with the suite green.
    #[test]
    fn the_repair_rebuilds_a_table_the_renumbered_ledger_skipped() {
        let mut connection = store_as_0_42_left_it();
        connection
            .execute_batch(
                // What that build had done by the time it stamped 5: named the provider, and
                // skipped the table. Written out rather than run, because a store built from
                // today's list cannot be missing a step today's list contains.
                "ALTER TABLE sessions ADD COLUMN provider TEXT NOT NULL DEFAULT '';
                 INSERT INTO mcp_oauth_credentials (server_name, credentials_json, updated_at) \
                 VALUES ('docs', '{\"access_token\":\"at1\"}', '2026-01-01T00:00:00Z');
                 PRAGMA user_version = 5;",
            )
            .expect("the shape the renumbered ledger left");

        let planned = plan(&connection).expect("classified");
        assert_eq!(planned.from, 5, "taken at its word, as a version is");
        apply(&mut connection, planned, &Context::adopting(Some("p"))).expect("repaired");

        let carried: Vec<(String, String, String)> = connection
            .prepare("SELECT server_name, kind, secret FROM mcp_credentials")
            .expect("the table the broken build never created")
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .expect("query")
            .collect::<rusqlite::Result<_>>()
            .expect("rows");
        assert_eq!(
            carried,
            vec![(
                "docs".to_string(),
                "oauth".to_string(),
                "{\"access_token\":\"at1\"}".to_string()
            )],
            "the bundle is carried across under the kind the reader looks for, not dropped"
        );
        let old: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = \
                 'mcp_oauth_credentials'",
                [],
                |row| row.get(0),
            )
            .expect("count the superseded table");
        assert_eq!(
            old, 0,
            "and the table it came from is gone, as on any other path"
        );
        assert_eq!(
            user_version(&connection).expect("read the version back"),
            MIGRATIONS.len() as u32,
            "a repaired store is at head like any other"
        );
    }

    /// Rule 3 for [`sessions_forget_their_model_overrides`], on the store that makes it bite.
    ///
    /// A fresh install on the broken build never ran the step that *added* the override columns,
    /// yet is stamped past the step that drops them. Replaying the drop over that store is the only
    /// way its guard is ever consulted: on every path built from the current ledger, index 3 has
    /// added the columns before index 6 removes them. Drop the guard and this is the one test that
    /// notices, while such a store would be refused on every start with `no such column`.
    #[test]
    fn dropping_the_override_columns_replays_over_a_store_that_never_had_them() {
        let mut connection = store_as_0_42_left_it();
        connection
            .execute_batch(
                "ALTER TABLE sessions ADD COLUMN provider TEXT NOT NULL DEFAULT '';
                 CREATE TABLE mcp_credentials (
                     server_name TEXT NOT NULL,
                     kind        TEXT NOT NULL,
                     secret      TEXT NOT NULL,
                     updated_at  TEXT NOT NULL,
                     PRIMARY KEY (server_name, kind)
                 );
                 DROP TABLE mcp_oauth_credentials;
                 PRAGMA user_version = 5;",
            )
            .expect("a fresh store as the broken build built one");

        let planned = plan(&connection).expect("classified");
        apply(&mut connection, planned, &Context::adopting(Some("p")))
            .expect("the drop must skip a column that was never added, not fail on it");

        let columns: Vec<String> = connection
            .prepare("SELECT name FROM pragma_table_info('sessions')")
            .expect("prepare")
            .query_map([], |row| row.get(0))
            .expect("query")
            .collect::<rusqlite::Result<_>>()
            .expect("rows");
        assert!(
            !columns.iter().any(|column| column == "model_override")
                && !columns.iter().any(|column| column == "base_url_override"),
            "and leaves the table without them either way: {columns:?}"
        );
    }

    /// Every job survives losing the column, keeps its schedule, and is still readable afterwards.
    ///
    /// The column sat at index 9 of a positionally-decoded row, so dropping it moves four fields
    /// under the reader. Reading the schedule and the fire time back out is what would catch a
    /// half-applied shift: a timestamp read from the wrong column does not parse.
    #[test]
    fn a_job_survives_losing_the_isolated_column() {
        let mut connection = store_as_0_42_left_it();
        connection
            .execute_batch(
                "INSERT INTO sessions (id, created_at, updated_at) \
                 VALUES ('s1', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z');
                 INSERT INTO scheduled_jobs \
                   (id, session_id, kind, spec, prompt, isolated, created_at, next_fire_at) \
                 VALUES ('j-solo', 's1', 'every', '1h', 'check the feed', 1, \
                         '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z'), \
                        ('j-joined', 's1', 'every', '2h', 'check the other feed', 0, \
                         '2026-01-01T00:00:00Z', '2026-01-01T02:00:00Z')",
            )
            .expect("one job of each kind");

        let planned = plan(&connection).expect("classified");
        apply(&mut connection, planned, &Context::adopting(Some("p"))).expect("migrated");

        let columns: Vec<String> = connection
            .prepare("SELECT name FROM pragma_table_info('scheduled_jobs')")
            .expect("prepare")
            .query_map([], |row| row.get(0))
            .expect("query")
            .collect::<rusqlite::Result<_>>()
            .expect("rows");
        assert!(
            !columns.iter().any(|column| column == "isolated"),
            "the column is gone: {columns:?}"
        );

        let rows: Vec<(String, String, String)> = connection
            .prepare("SELECT id, spec, next_fire_at FROM scheduled_jobs ORDER BY id")
            .expect("prepare")
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .expect("query")
            .collect::<rusqlite::Result<_>>()
            .expect("rows");
        assert_eq!(
            rows,
            vec![
                (
                    "j-joined".to_string(),
                    "2h".to_string(),
                    "2026-01-01T02:00:00Z".to_string()
                ),
                (
                    "j-solo".to_string(),
                    "1h".to_string(),
                    "2026-01-01T01:00:00Z".to_string()
                ),
            ],
            "both jobs keep their schedule and their next fire; the one that was isolated is now \
             an ordinary job on the same session"
        );
    }

    /// Rule 3 for the column drop. A bare `ALTER TABLE … DROP COLUMN` would fail with
    /// `no such column: isolated` on the second pass and refuse the store on every start.
    #[test]
    fn dropping_the_isolated_column_replays_cleanly() {
        let mut connection = store_as_0_42_left_it();
        connection
            .execute_batch(
                "INSERT INTO sessions (id, created_at, updated_at) \
                 VALUES ('s1', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z');
                 INSERT INTO scheduled_jobs \
                   (id, session_id, kind, spec, prompt, isolated, created_at, next_fire_at) \
                 VALUES ('j1', 's1', 'every', '1h', 'watch it', 1, \
                         '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z')",
            )
            .expect("a job");
        let first = plan(&connection).expect("classified");
        apply(&mut connection, first, &Context::adopting(Some("p"))).expect("migrated");

        connection
            .execute_batch("PRAGMA user_version = 0;")
            .expect("the round trip that drops the version");
        let replayed = plan(&connection).expect("classified by shape");
        apply(&mut connection, replayed, &Context::adopting(Some("p")))
            .expect("the replay must not fail");

        let surviving: i64 = connection
            .query_row("SELECT COUNT(*) FROM scheduled_jobs", [], |row| row.get(0))
            .expect("query");
        assert_eq!(surviving, 1, "and the job is still there");
    }

    /// The step speaks exactly when it has something to say, and says which profile it chose.
    ///
    /// Both halves matter and both were wrong. It warned on the literal first run of a fresh
    /// install, where there are no sessions to leave without a provider, which teaches a reader to
    /// skip the message that does matter. And it adopted a profile for every carried-forward
    /// session in silence: for anyone with two accounts that is a guess, and a session that ran on
    /// the other one now names this one and bills here when resumed.
    #[test]
    fn the_step_reports_the_profile_it_adopted_and_only_then() {
        #[derive(Clone)]
        struct Capture(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Capture {
            fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
                crate::sync::lock(&self.0).extend_from_slice(buffer);
                Ok(buffer.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Capture {
            type Writer = Self;

            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        // INFO rather than WARN, because the adoption notice is a lifecycle signpost and the
        // empty-profile complaint above it is a warning; capturing both keeps one test honest
        // about which level each uses.
        let migrate = |plant: bool, default: Option<&str>| -> String {
            let capture = Capture(std::sync::Arc::new(std::sync::Mutex::new(Vec::new())));
            let buffer = std::sync::Arc::clone(&capture.0);
            let subscriber = tracing_subscriber::fmt()
                .with_writer(capture)
                .with_max_level(tracing::Level::INFO)
                .finish();
            tracing::subscriber::with_default(subscriber, || {
                let mut connection = store_as_0_42_left_it();
                if plant {
                    plant_session(&connection, "carried");
                }
                let plan = plan(&connection).expect("classified");
                apply(&mut connection, plan, &Context::adopting(default)).expect("migrated");
            });
            String::from_utf8_lossy(&crate::sync::lock(&buffer)).into_owned()
        };

        let text = migrate(true, Some("work"));
        assert!(
            text.contains("recorded provider profile 'work'") && text.contains(" 1 session"),
            "the adoption must name the profile and how many rows it touched: {text}"
        );

        // Nothing carried forward: the step still runs and still has nothing to report.
        let text = migrate(false, Some("work"));
        assert!(
            !text.contains("recorded provider profile"),
            "a store with no sessions has nothing to adopt: {text}"
        );
        let text = migrate(false, None);
        assert!(
            !text.contains("left without one"),
            "a fresh install must not be warned about sessions it does not have: {text}"
        );
    }

    #[test]
    fn each_fire_mode_becomes_its_predicate() {
        for (fire, expected) in [("on-change", "changed"), ("on-success", "succeeded")] {
            let mut connection = store_as_0_42_left_it();
            plant_job(&connection, "job", Some("gh pr checks"), Some(fire));
            let plan = plan(&connection).expect("classified");
            apply(&mut connection, plan, &Context::default()).expect("converted");

            let (kind, spec) = gate_of(&connection, "job");
            assert_eq!(kind.as_deref(), Some("shell"));
            let spec: serde_json::Value =
                serde_json::from_str(&spec.expect("a spec")).expect("valid JSON");
            assert_eq!(spec["shell"]["command"], "gh pr checks");
            assert_eq!(spec["when"], expected);
        }
    }

    /// A command is arbitrary user text and has to survive byte for byte. The corpus is the one the
    /// retired Python script used, for the same reason it chose it.
    #[test]
    fn a_command_with_quotes_newlines_and_non_ascii_survives() {
        let awkward = "curl -f 'https://x' # naïve — ünïcødé ✓ 日本語\nsecond \"line\"\\";
        let mut connection = store_as_0_42_left_it();
        plant_job(&connection, "job", Some(awkward), Some("on-change"));
        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::default()).expect("converted");

        let (_, spec) = gate_of(&connection, "job");
        let spec: serde_json::Value =
            serde_json::from_str(&spec.expect("a spec")).expect("valid JSON");
        assert_eq!(spec["shell"]["command"], awkward);
    }

    /// The expensive failure, and the reason the unconvertible row is written the way it is.
    ///
    /// 0.42 refused to load these, so they never fired. Leaving both gate columns null would read
    /// as *no gate at all*, turning a watcher that never fired into a timer that fires every
    /// interval. Setting `gate_kind` without `gate_spec` keeps them refused by the reader's own
    /// corrupt-row rule, which names no version and would say the same of a hand-edited row.
    #[test]
    fn a_gate_that_cannot_be_converted_stays_refused_rather_than_becoming_ungated() {
        let mut connection = store_as_0_42_left_it();
        plant_job(
            &connection,
            "unknown-fire",
            Some("echo hi"),
            Some("on-tuesday"),
        );
        plant_job(&connection, "half-written", Some("echo hi"), None);
        plant_job(&connection, "no-command", None, Some("on-change"));
        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::default()).expect("converted");

        for id in ["unknown-fire", "half-written", "no-command"] {
            let (kind, spec) = gate_of(&connection, id);
            assert_eq!(kind.as_deref(), Some("shell"), "{id} keeps a gate kind");
            assert_eq!(
                spec, None,
                "{id} must stay unreadable, because a null kind *and* spec reads as ungated"
            );
        }
    }

    #[test]
    fn the_retired_columns_are_gone_and_the_lease_columns_are_present() {
        let mut connection = store_as_0_42_left_it();
        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::default()).expect("converted");

        let columns = table_columns(&connection, "scheduled_jobs").expect("columns");
        for gone in ["gate_command", "gate_fire"] {
            assert!(
                !columns.iter().any(|c| c == gone),
                "{gone} should be dropped"
            );
        }
        for present in [
            "gate_kind",
            "gate_spec_json",
            "claimed_by",
            "claim_expires_at",
            "attempts",
        ] {
            assert!(
                columns.iter().any(|c| c == present),
                "{present} should exist"
            );
        }
    }

    /// Enforcement is suspended for the steps, so `apply` has to put it back. A connection left
    /// with foreign keys off would go on serving the whole process, silently accepting writes the
    /// schema forbids.
    #[test]
    fn foreign_keys_are_suspended_for_the_steps_and_restored_afterwards() {
        let mut connection = store_as_0_42_left_it();
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .expect("enforcement on, as `Store::open` leaves it");
        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::default()).expect("migrated");

        let enforcing: i64 = connection
            .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
            .expect("readable");
        assert_eq!(
            enforcing, 1,
            "foreign keys must be back on after a migration"
        );
    }

    /// The count is what decides whether a migration is refused, so a constant in its place disarms
    /// the guard while every other test still passes. The comparison that reads this is `after >
    /// before`, which any constant return value makes false, so the value itself has to be pinned.
    #[test]
    fn dangling_references_are_counted_rather_than_assumed() {
        let mut connection = store_as_0_42_left_it();
        let transaction = connection.transaction().expect("a transaction");
        assert_eq!(
            count_dangling_references(&transaction).expect("count"),
            0,
            "a store with no rows has nothing dangling"
        );
        drop(transaction);

        // Only reachable with enforcement off, which is how these rows come to exist at all.
        connection
            .execute_batch(
                "PRAGMA foreign_keys = OFF;
                 INSERT INTO scheduled_jobs (id, session_id, kind, spec, prompt, created_at, \
                     next_fire_at) \
                 VALUES ('a', 'no-such-session', 'every', '6h', 'p', 'now', 'later');
                 INSERT INTO scheduled_jobs (id, session_id, kind, spec, prompt, created_at, \
                     next_fire_at) \
                 VALUES ('b', 'no-such-session', 'every', '6h', 'p', 'now', 'later');",
            )
            .expect("two orphaned rows");
        let transaction = connection.transaction().expect("a transaction");
        assert_eq!(
            count_dangling_references(&transaction).expect("count"),
            2,
            "both orphans are counted, not just noticed"
        );
    }

    /// The warning is the only signal a user gets that a job was left inert, and the only place the
    /// ids appear. Nothing else reads the log, so without this the guard can be inverted to fire on
    /// the empty case and stay green.
    #[test]
    fn only_a_store_with_unreadable_gates_is_warned_about() {
        crate::render::log_capture::start();

        let mut clean = store_as_0_42_left_it();
        plant_job(&clean, "fine", Some("gh pr checks"), Some("on-change"));
        let clean_plan = plan(&clean).expect("classified");
        apply(&mut clean, clean_plan, &Context::default()).expect("converted");
        assert!(
            !crate::render::log_capture::warnings().contains("stay inert"),
            "a store whose gates all convert must not be warned about: {}",
            crate::render::log_capture::warnings()
        );

        let mut damaged = store_as_0_42_left_it();
        plant_job(&damaged, "unreadable", Some("echo hi"), Some("on-tuesday"));
        let damaged_plan = plan(&damaged).expect("classified");
        apply(&mut damaged, damaged_plan, &Context::default()).expect("converted");
        let warnings = crate::render::log_capture::warnings();
        assert!(warnings.contains("stay inert"), "{warnings}");
        assert!(
            warnings.contains("unreadable"),
            "the warning must name the row, since nothing else will: {warnings}"
        );
    }

    /// The whole point of suspending enforcement, and until [`with_foreign_keys_suspended`] was
    /// split out there was no way to reach it: no shipped migration rebuilds a table, so nothing
    /// could exercise the case the suspension exists for.
    ///
    /// `sessions` is the parent of `messages`, `tool_outputs`, `scheduled_jobs` and
    /// `background_tasks`, every one `ON DELETE CASCADE`. SQLite's documented procedure for the
    /// table changes `ALTER TABLE` cannot express ends in `DROP TABLE sessions`, and with
    /// enforcement on that deletes the entire conversation history inside a transaction that then
    /// commits successfully. Measured before the fix: one child row before, zero after.
    #[test]
    fn a_step_that_rebuilds_a_parent_table_does_not_delete_its_children() {
        let mut connection = store_as_0_42_left_it();
        connection
            .execute_batch(
                "PRAGMA foreign_keys = ON;
                 INSERT INTO sessions (id, created_at, updated_at) VALUES ('s', 'x', 'x');
                 INSERT INTO messages (session_id, role, content, created_at) \
                     VALUES ('s', 'user', 'keep me', 'x');",
            )
            .expect("a session with a message hanging off it");

        // The rebuild `MIGRATIONS` does not contain, handed in the way a future migration would
        // perform it.
        let rebuilt = with_foreign_keys_suspended(&mut connection, |connection| {
            let transaction = connection
                .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
                .expect("a transaction");
            transaction
                .execute_batch(
                    "CREATE TABLE sessions_new (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, \
                         updated_at TEXT NOT NULL);
                     INSERT INTO sessions_new SELECT id, created_at, updated_at FROM sessions;
                     DROP TABLE sessions;
                     ALTER TABLE sessions_new RENAME TO sessions;",
                )
                .expect("the rebuild runs");
            transaction.commit().expect("and commits");
            Ok(())
        });
        rebuilt.expect("the wrapper returns cleanly");

        let survivors: i64 = connection
            .query_row("SELECT count(*) FROM messages", [], |row| row.get(0))
            .expect("count the messages");
        assert_eq!(
            survivors, 1,
            "rebuilding the parent must not cascade; with enforcement on this reads 0 and the \
             conversation is gone"
        );
        let enforcing: i64 = connection
            .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
            .expect("readable");
        assert_eq!(enforcing, 1, "and enforcement is back on afterwards");
    }

    /// A store round-tripped through `sqlite3 .dump` keeps its schema and loses its version, so
    /// every step after the baseline replays over data that already has it. Plain `VACUUM` keeps
    /// the version; `.dump` does not, and that round trip is what people reach for to repair or
    /// move a database.
    ///
    /// Survivable only because `gates_become_kind_and_spec` guards each `ALTER TABLE` and returns
    /// early once `gate_command` is gone. This pins that, so the day a step is written without
    /// those guards it fails here rather than on a stranger's machine, permanently.
    #[test]
    fn a_store_that_lost_its_version_replays_without_damage() {
        let mut connection = connection_for_test();
        create_for_test(&mut connection).expect("a store at head");
        connection
            .execute(
                "INSERT INTO sessions (id, created_at, updated_at) VALUES ('s', 'now', 'now')",
                [],
            )
            .expect("a session");
        connection
            .execute(
                "INSERT INTO scheduled_jobs (id, session_id, kind, spec, prompt, gate_kind, \
                 gate_spec_json, created_at, next_fire_at) \
                 VALUES ('j', 's', 'every', '6h', 'p', 'shell', ?1, 'now', 'later')",
                [r#"{"shell":{"command":"gh pr checks"},"when":"changed"}"#],
            )
            .expect("an already-converted job");
        let before = fingerprint(&connection);

        // What `.dump` into a fresh database leaves: the schema, the rows, and no version.
        connection
            .execute_batch("PRAGMA user_version = 0;")
            .expect("lose the version");

        let replayed = plan(&connection).expect("classified by shape");
        assert_eq!(
            replayed.from, 1,
            "shape says the baseline is applied, which it is"
        );
        apply(&mut connection, replayed, &Context::default()).expect("the replay must not fail");

        assert_eq!(before, fingerprint(&connection), "the schema is unchanged");
        let (kind, spec) = gate_of(&connection, "j");
        assert_eq!(kind.as_deref(), Some("shell"));
        assert_eq!(
            spec.as_deref(),
            Some(r#"{"shell":{"command":"gh pr checks"},"when":"changed"}"#),
            "the already-converted gate is left alone rather than converted twice"
        );
    }

    /// Damage a store already carries must not block it forever. Enforcement is on for every normal
    /// write, so arranging this takes hand-editing, and refusing the migration would leave such a
    /// user with no way forward at all.
    #[test]
    fn a_dangling_reference_that_predates_the_migration_does_not_block_it() {
        let mut connection = store_as_0_42_left_it();
        // Written with enforcement off, which is the only way this row can exist.
        connection
            .execute_batch(
                "PRAGMA foreign_keys = OFF;
                 INSERT INTO scheduled_jobs (id, session_id, kind, spec, prompt, created_at, \
                     next_fire_at) \
                 VALUES ('orphan', 'no-such-session', 'every', '6h', 'p', 'now', 'later');",
            )
            .expect("an orphaned row");

        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::default())
            .expect("the migration is not blocked by inherited damage");
        assert_eq!(
            user_version(&connection).expect("version"),
            MIGRATIONS.len() as u32
        );
    }

    /// Returning 1 asserts the whole baseline is present. The old code filled a gap silently on
    /// every open; nothing does now, so a store missing a table has to be named rather than
    /// stamped at head with the table still absent.
    #[test]
    fn a_half_built_store_is_named_rather_than_stamped_as_current() {
        let connection = store_as_0_42_left_it();
        connection
            .execute_batch("DROP TABLE background_tasks;")
            .expect("a store an interrupted first run could leave");

        let error = plan(&connection).expect_err("refused");
        let message = error.to_string();
        assert!(message.contains("background_tasks"), "{message}");
        assert!(message.contains("Nothing has been changed"), "{message}");
    }

    #[test]
    fn a_second_run_finds_nothing_to_do() {
        let mut connection = store_as_0_42_left_it();
        plant_job(&connection, "job", Some("echo hi"), Some("on-change"));
        let first = plan(&connection).expect("classified");
        apply(&mut connection, first, &Context::default()).expect("converted");
        let before = fingerprint(&connection);

        let second = plan(&connection).expect("classified again");
        assert!(!second.has_work(), "a converted store has nothing pending");
        apply(&mut connection, second, &Context::default()).expect("a no-op");
        assert_eq!(before, fingerprint(&connection), "nothing moved");
        assert_eq!(
            user_version(&connection).expect("version"),
            MIGRATIONS.len() as u32
        );
    }

    /// The store a hand-run of the retired script left behind: gates already converted, and on an
    /// early build of it, no lease columns. It has to converge here rather than be refused.
    #[test]
    fn a_store_converted_by_hand_still_reaches_head() {
        let mut connection = store_as_0_42_left_it();
        connection
            .execute_batch(
                "ALTER TABLE scheduled_jobs ADD COLUMN gate_kind TEXT;
                 ALTER TABLE scheduled_jobs ADD COLUMN gate_spec TEXT;
                 ALTER TABLE scheduled_jobs DROP COLUMN gate_command;
                 ALTER TABLE scheduled_jobs DROP COLUMN gate_fire;",
            )
            .expect("a hand conversion");
        let plan = plan(&connection).expect("classified");
        apply(&mut connection, plan, &Context::default()).expect("the lease columns still arrive");

        let columns = table_columns(&connection, "scheduled_jobs").expect("columns");
        for present in ["claimed_by", "claim_expires_at", "attempts"] {
            assert!(
                columns.iter().any(|c| c == present),
                "{present} should exist"
            );
        }
    }

    #[test]
    fn a_0_41_store_is_refused_by_name_rather_than_converted() {
        let connection = connection_for_test();
        connection
            .execute_batch(
                "CREATE TABLE sessions (id TEXT PRIMARY KEY);
                 CREATE TABLE scheduled_jobs (id TEXT PRIMARY KEY, gate_command TEXT, \
                     gate_fire TEXT);",
            )
            .expect("a 0.41 shape");
        let error = plan(&connection).expect_err("0.41 is refused");
        let message = error.to_string();
        assert!(message.contains("migrate-0.41-to-0.42.py"), "{message}");
        assert!(message.contains("Nothing has been changed"), "{message}");
    }

    /// A refusal has to name the file it is about. meka creates a store on any invocation while
    /// `config.toml` appears only once a provider is added, so a machine that ran an old meka once
    /// and was never configured holds a store its owner has no reason to believe exists. Told only
    /// that "this store" is in the 0.41 shape, the honest reading there is that meka is wrong about
    /// a database that was never set up.
    ///
    /// Every refusal, not only the one that prompted this: each ends with the reader going to look
    /// at a file, and a later one added without the path would be exactly as unlocatable.
    ///
    /// Matched on the file name rather than the whole path, because SQLite reports the path it
    /// opened and that is not textually the one passed on every platform.
    #[test]
    fn every_refusal_names_the_store_it_is_about() {
        // Each is the smallest shape that reaches one refusal, paired with a phrase unique to it so
        // a case that starts landing on a *different* refusal fails rather than passing by luck.
        let shapes: &[(&str, &str, &str)] = &[
            (
                "newer.db",
                "PRAGMA user_version = 999;",
                "is at schema version",
            ),
            (
                "pre-042.db",
                "CREATE TABLE sessions (id TEXT PRIMARY KEY);",
                "has tables but no `scheduled_jobs`",
            ),
            (
                "shape-041.db",
                "CREATE TABLE sessions (id TEXT PRIMARY KEY);
                 CREATE TABLE scheduled_jobs (id TEXT PRIMARY KEY, gate_command TEXT);",
                "is in the 0.41 shape",
            ),
            (
                "half-built.db",
                "CREATE TABLE sessions (id TEXT PRIMARY KEY);
                 CREATE TABLE scheduled_jobs (id TEXT PRIMARY KEY, gate_permission TEXT);",
                "is missing",
            ),
        ];

        let directory = tempfile::tempdir().expect("a temp dir");
        for (file, shape, phrase) in shapes {
            let connection = rusqlite::Connection::open(directory.path().join(file))
                .expect("a file-backed store");
            connection.execute_batch(shape).expect("the planted shape");

            let message = plan(&connection).expect_err("refused").to_string();
            assert!(
                message.contains(phrase),
                "{file} hit another refusal: {message}"
            );
            assert!(message.contains(file), "{file} must be named: {message}");
        }
    }

    /// The fallback arm, which every unit test above reaches. An in-memory store reports an empty
    /// path rather than `None`, so a bare `unwrap_or` would put the refusal's subject at the front
    /// of the sentence as nothing at all.
    #[test]
    fn a_store_with_no_path_is_still_given_a_subject() {
        let connection = connection_for_test();
        connection
            .execute_batch(
                "CREATE TABLE sessions (id TEXT PRIMARY KEY);
                 CREATE TABLE scheduled_jobs (id TEXT PRIMARY KEY, gate_command TEXT);",
            )
            .expect("a 0.41 shape");

        let message = plan(&connection).expect_err("0.41 is refused").to_string();
        assert!(
            message.contains("the session store is in the 0.41 shape"),
            "{message}"
        );
    }

    /// The whole basis for distrusting a stored `1`, pinned so it cannot quietly stop being true.
    /// If the ledger could ever stamp that value, `plan` would be second-guessing a store this
    /// build itself wrote, and the shape probe would run on every start forever.
    #[test]
    fn the_ledger_can_never_stamp_the_retired_flag() {
        assert!(
            MIGRATIONS.len() as u32 > RETIRED_INITIALIZED_FLAG,
            "`apply` stamps `MIGRATIONS.len()`, so a ledger of one step would write the same \
             number the retired schema system used as its initialized flag, and nothing could tell \
             the two apart afterwards"
        );
    }

    /// The bug [`RETIRED_INITIALIZED_FLAG`] exists for, found by running against a real store.
    ///
    /// Every meka up to 0.41 stamped `user_version = 1` on a store it had finished initializing, so
    /// a 0.41 store in the wild carries it. Taking that at face value skipped the shape probe,
    /// which is where this refusal lives; the gate conversion then ran against a table with no
    /// `gate_permission`, dropped the columns it had read, committed, and stamped head. Reproduced
    /// end to end: every later read failed with `no such column: gate_permission`, and because the
    /// store was stamped current nothing would ever revisit it.
    #[test]
    fn a_0_41_store_carrying_the_retired_stamp_is_still_refused() {
        let connection = connection_for_test();
        connection
            .execute_batch(
                "CREATE TABLE sessions (id TEXT PRIMARY KEY);
                 CREATE TABLE scheduled_jobs (id TEXT PRIMARY KEY, gate_command TEXT, \
                     gate_fire TEXT);
                 PRAGMA user_version = 1;",
            )
            .expect("a 0.41 store as that release left one");
        let error =
            plan(&connection).expect_err("refused, not converted into something unreadable");
        assert!(
            error.to_string().contains("migrate-0.41-to-0.42.py"),
            "{error}"
        );
    }

    /// The other half: distrusting the stamp must not cost a 0.42 store its upgrade. The shape
    /// probe reaches the same answer the stamp claimed, so this one converts normally.
    #[test]
    fn a_0_42_store_carrying_the_retired_stamp_migrates_normally() {
        let mut connection = store_as_0_42_left_it();
        connection
            .execute_batch("PRAGMA user_version = 1;")
            .expect("the stamp every release up to 0.41 left");
        plant_job(&connection, "job", Some("gh pr checks"), Some("on-change"));

        let plan = plan(&connection).expect("classified by shape rather than by the stamp");
        assert_eq!(
            plan.from, 1,
            "the shape says the baseline is applied, and it is"
        );
        apply(&mut connection, plan, &Context::default()).expect("converted");
        let (kind, spec) = gate_of(&connection, "job");
        assert_eq!(kind.as_deref(), Some("shell"));
        assert!(
            spec.is_some(),
            "the gate converted rather than being left unreadable"
        );
    }

    #[test]
    fn a_store_from_a_newer_meka_is_refused_and_left_alone() {
        let mut connection = connection_for_test();
        create_for_test(&mut connection).expect("a fresh store");
        connection
            .execute_batch("PRAGMA user_version = 99;")
            .expect("a version from the future");
        let before = fingerprint(&connection);

        let error = plan(&connection).expect_err("a newer store is refused");
        assert!(error.to_string().contains("newer release"), "{error}");
        assert_eq!(before, fingerprint(&connection), "nothing was touched");
        assert_eq!(user_version(&connection).expect("version"), 99);
    }

    /// A file meka did not write is not a store to carry forward. Every release before this one
    /// created its tables alongside whatever was already there, and so does this.
    #[test]
    fn a_database_that_is_not_a_meka_store_is_built_rather_than_refused() {
        let mut connection = connection_for_test();
        connection
            .execute_batch("CREATE TABLE placeholder (id INTEGER);")
            .expect("an unrelated table");
        let plan = plan(&connection).expect("classified as new");
        assert_eq!(
            plan.from, 0,
            "there is nothing of meka's here to carry forward"
        );
        apply(&mut connection, plan, &Context::default()).expect("the schema is built");
        assert!(
            !table_columns(&connection, "sessions")
                .expect("columns")
                .is_empty()
        );
    }

    /// An unreadable `config.toml` must not be turned into "no profile" and stamped.
    ///
    /// The adopt step runs once and `user_version` moves with it, so a value derived from a parse
    /// error is written to every carried-forward session and never revisited: one typo, and 476
    /// sessions permanently record no profile with nothing said. "Nothing resolved" and "I could
    /// not read the file" are different answers and only the first may be adopted.
    ///
    /// The refusal is conditional on there being rows to stamp, which is what keeps `meka mcp
    /// remove` and `meka provider remove` usable: those edit the raw document through `toml_edit`
    /// and are how a user repairs such a config, so a store with nothing to migrate must still
    /// open.
    #[test]
    fn an_unreadable_config_refuses_to_stamp_carried_sessions_but_not_an_empty_store() {
        let mut carrying = store_as_0_42_left_it();
        carrying
            .execute_batch(
                "INSERT INTO sessions (id, created_at, updated_at) VALUES \
                 ('11111111-1111-4111-8111-111111111111', '2020-01-01', '2020-01-01')",
            )
            .expect("a session that predates meka recording a provider");
        let before = fingerprint(&carrying);

        let carrying_plan = plan(&carrying).expect("classified");
        assert!(carrying_plan.has_work());
        let error = apply(
            &mut carrying,
            carrying_plan,
            &Context::on_unreadable_config(),
        )
        .expect_err("a store with rows to stamp must refuse");
        assert!(
            error.to_string().contains("config.toml"),
            "the refusal must name what has to be fixed: {error}"
        );
        assert_eq!(
            before,
            fingerprint(&carrying),
            "the store must be exactly as it was, so a later run can migrate it correctly"
        );
        assert_eq!(
            user_version(&carrying).expect("version"),
            0,
            "and the version must not move, or the step never runs again"
        );

        // Nothing to stamp, so nothing to get wrong: the repair commands keep working.
        let mut empty = store_as_0_42_left_it();
        let empty_plan = plan(&empty).expect("classified");
        apply(&mut empty, empty_plan, &Context::on_unreadable_config())
            .expect("a store with no carried sessions opens on an unreadable config");
        assert_eq!(
            user_version(&empty).expect("version"),
            MIGRATIONS.len() as u32,
            "and reaches head"
        );
    }

    /// One transaction, so a step that fails leaves the version and the tables where they were.
    /// Without this the store can end up carrying half a migration with nothing recording that.
    #[test]
    fn a_failing_step_leaves_the_store_untouched() {
        let mut connection = store_as_0_42_left_it();
        plant_job(&connection, "job", Some("gh pr checks"), Some("on-change"));
        // A store that classifies cleanly as 0.42 but that the gate conversion cannot finish: the
        // trigger aborts the `UPDATE` that writes the converted spec, after the step has already
        // added five columns. That partial state is exactly what the single transaction has to
        // undo, so forcing the failure *mid-step* is the point rather than an accident of setup.
        connection
            .execute_batch(
                "CREATE TRIGGER refuse_the_conversion BEFORE UPDATE ON scheduled_jobs \
                 BEGIN SELECT RAISE(ABORT, 'this store will not take the update'); END;",
            )
            .expect("a store the next step cannot get through");
        let before = fingerprint(&connection);

        let plan = plan(&connection).expect("classified");
        assert!(plan.has_work());
        let error = apply(&mut connection, plan, &Context::default()).expect_err("the step fails");
        assert!(
            error.to_string().contains("The store is unchanged"),
            "{error}"
        );
        assert_eq!(
            before,
            fingerprint(&connection),
            "the tables are as they were"
        );
        assert_eq!(
            user_version(&connection).expect("version"),
            0,
            "the version must not move without the schema it describes"
        );
        assert!(
            table_columns(&connection, "scheduled_jobs")
                .expect("columns")
                .iter()
                .any(|column| column == "gate_command"),
            "the columns the failed step meant to drop are still there"
        );
    }

    /// The column arrives on an existing store, and the backfill restates history without inventing
    /// any.
    ///
    /// A delivered row was announced in the same statement under the old design, so copying the
    /// stamp across is what did happen. An undelivered one was never announced, and its NULL is
    /// what lets the poller announce it once -- including an `Interrupted` task that no host ever
    /// got to report, which previously fired no `task.finished` at all.
    #[test]
    fn announcing_is_split_from_delivering_and_backfilled_from_what_happened() {
        // Seeded *before* migrating, because the backfill is about rows a previous binary wrote.
        let mut store = store_as_0_42_left_it();
        store
            .execute_batch(
                "INSERT INTO sessions (id, created_at, updated_at) VALUES ('s', 'now', 'now');
                 INSERT INTO background_tasks \
                     (id, session_id, tool_name, label, status, started_at, delivered_at) \
                     VALUES ('done', 's', 't', 'l', 'completed', 'now', 'stamped');
                 INSERT INTO background_tasks \
                     (id, session_id, tool_name, label, status, started_at, delivered_at) \
                     VALUES ('waiting', 's', 't', 'l', 'canceled', 'now', NULL);",
            )
            .expect("seed");

        let plan = plan(&store).expect("classified");
        apply(&mut store, plan, &Context::adopting(Some("p"))).expect("migrated");

        // A `.dump` round trip drops `user_version`, so every step replays over data that has it.
        let transaction = store.transaction().expect("transaction");
        background_tasks_announce_before_they_deliver(&transaction).expect("safe to run twice");
        transaction.commit().expect("commit");

        let announced = |id: &str| -> Option<String> {
            store
                .query_row(
                    "SELECT announced_at FROM background_tasks WHERE id = ?1",
                    [id],
                    |row| row.get::<_, Option<String>>(0),
                )
                .expect("read")
        };
        assert_eq!(
            announced("done").as_deref(),
            Some("stamped"),
            "a delivered row was announced at the same instant, so the stamp carries across"
        );
        assert_eq!(
            announced("waiting"),
            None,
            "an undelivered one was never announced, so the poller still owes it"
        );
    }

    /// A task an earlier meka stopped on request recorded `cancelled`, and the reader accepts only
    /// the spelling meka writes now. Replayed, the step finds nothing left to rewrite.
    #[test]
    fn a_task_status_spelled_with_two_ls_reads_back_as_canceled() {
        let mut store = store_as_0_42_left_it();
        store
            .execute_batch(
                "INSERT INTO sessions (id, created_at, updated_at) VALUES ('s', 'now', 'now');
                 INSERT INTO background_tasks \
                     (id, session_id, tool_name, label, status, started_at) \
                     VALUES ('stopped', 's', 't', 'l', 'cancelled', 'now');
                 INSERT INTO background_tasks \
                     (id, session_id, tool_name, label, status, started_at) \
                     VALUES ('done', 's', 't', 'l', 'completed', 'now');",
            )
            .expect("seed");

        let plan = plan(&store).expect("classified");
        apply(&mut store, plan, &Context::adopting(Some("p"))).expect("migrated");

        let status_of = |id: &str| -> String {
            store
                .query_row(
                    "SELECT status FROM background_tasks WHERE id = ?1",
                    [id],
                    |row| row.get(0),
                )
                .expect("read")
        };
        assert_eq!(
            status_of("stopped").parse::<crate::store::background::TaskStatus>(),
            Ok(crate::store::background::TaskStatus::Canceled),
            "the row reads back through the one spelling the reader accepts"
        );
        assert_eq!(
            status_of("done"),
            "completed",
            "every other status is left as it was"
        );

        // A `.dump` round trip drops `user_version`, so every step replays over data that has it.
        let rewritten = store
            .execute(BACKGROUND_TASKS_CANCELED, [])
            .expect("safe to run twice");
        assert_eq!(rewritten, 0, "a replay finds nothing left to rewrite");
    }
}
