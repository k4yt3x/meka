//! SQLite-backed session store. Persists messages, large tool outputs (so
//! they can be referenced from the conversation by handle), OAuth tokens,
//! and MCP credentials. Per-session mutual exclusion is provided by an
//! OS-level file lock ([`SessionLock`]) so the kernel reclaims it whenever
//! the holder dies — no PID-aliveness check, no risk of stale locks.
//!
//! On Unix the data directory (`0700`), lock directory (`0700`), and the
//! database file itself (`0600`) are tightened after creation so the
//! persisted OAuth tokens, MCP credentials, and conversation content
//! aren't readable by other local users regardless of the user's umask.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use fd_lock::{RwLock as FdRwLock, RwLockWriteGuard as FdRwLockWriteGuard};
use serde::{Deserialize, Serialize};
use tokio_rusqlite::Connection;
use uuid::Uuid;

use crate::error::{AgshError, Result};
use crate::provider::AuthCredential;

/// Raw row from the `messages` table — the on-disk shape of a single
/// [`crate::conversation::Event`]. Internal to the session module: only
/// the encoder and decoder helpers handle these directly. External
/// consumers go through [`SessionManager::save_event`] /
/// [`SessionManager::load_events`].
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredMessage {
    role: String,
    content: String,
    created_at: String,
}

#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub id: Uuid,
    pub updated_at: String,
    pub preview: String,
}

#[derive(Debug, Clone)]
pub struct ToolOutputSummary {
    pub name: String,
    pub size: usize,
    pub created_at: String,
}

#[derive(Clone)]
pub struct SessionManager {
    connection: Arc<Connection>,
    lock_dir: PathBuf,
}

/// RAII handle for an exclusive per-session OS file lock. Holding this value
/// keeps the underlying lock file descriptor open; dropping it (including
/// when the process exits or panics) closes the FD, which causes the kernel
/// to release the `flock`/`LockFileEx` lock automatically. There is no
/// "stale lock" failure mode — even `SIGKILL` is safe.
///
/// Internally this is a self-referential struct: `_guard` borrows from
/// `*_lock` (a `Box` for stable heap address). Field declaration order
/// guarantees `_guard` is dropped before `_lock`, which is the safety
/// invariant of the lifetime transmute used during construction.
pub struct SessionLock {
    _guard: FdRwLockWriteGuard<'static, File>,
    _lock: Box<FdRwLock<File>>,
}

fn default_database_path() -> Result<PathBuf> {
    // `AGSH_DATA_DIR` is the cross-platform override — the only env var that
    // works on every OS, mirroring how `AGSH_CONFIG_DIR` overrides the config
    // directory. The value points at the `agsh` data dir itself (the parent
    // that contains `sessions.db`). Useful for tests, portable installs, and
    // isolating per-project session state from the global one.
    if let Ok(value) = std::env::var("AGSH_DATA_DIR")
        && !value.is_empty()
    {
        return Ok(PathBuf::from(value).join("sessions.db"));
    }

    // `dirs::data_dir()` honors XDG_DATA_HOME on Linux and the platform's
    // standard data directory elsewhere. Fall back to `$HOME/.local/share`
    // so a tilde never reaches the filesystem (which doesn't expand it).
    let base = dirs::data_dir()
        .or_else(|| dirs::home_dir().map(|home| home.join(".local").join("share")))
        .ok_or_else(|| {
            AgshError::Config(
                "could not determine a data directory; set AGSH_DATA_DIR, \
                 XDG_DATA_HOME, or HOME, or pass an explicit session database \
                 path"
                    .into(),
            )
        })?;
    Ok(base.join("agsh").join("sessions.db"))
}

/// Create a directory (and any missing parents) born at mode 0700 on Unix.
/// Avoids the umask window that `create_dir_all` + later `set_permissions`
/// would open: between `mkdir(2)` and `chmod(2)`, the directory would be
/// readable by other local users on a permissive umask. `DirBuilderExt::mode`
/// passes the mode straight to `mkdir`. Pre-existing directories keep their
/// mode — callers that need to tighten an already-existing dir should still
/// follow up with `restrict_permissions`.
#[cfg(unix)]
fn create_private_dir(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .mode(0o700)
        .recursive(true)
        .create(path)
}

#[cfg(not(unix))]
fn create_private_dir(path: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(path)
}

/// Restrict a path's permissions on Unix. Best-effort — if the call fails
/// we log and continue, because on some mounts (`/tmp` under specific
/// overlay setups, NFS without proper support, etc.) `chmod` returns
/// `EPERM`/`EROFS` and refusing to open the session is a strictly worse
/// failure than leaving the file at the umask-derived mode.
#[cfg(unix)]
fn restrict_permissions(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    match std::fs::metadata(path) {
        Ok(metadata) => {
            let mut permissions = metadata.permissions();
            permissions.set_mode(mode);
            if let Err(error) = std::fs::set_permissions(path, permissions) {
                tracing::debug!(
                    "failed to restrict '{}' to mode {:o}: {}",
                    path.display(),
                    mode,
                    error
                );
            }
        }
        Err(error) => {
            tracing::debug!(
                "failed to stat '{}' while restricting permissions: {}",
                path.display(),
                error
            );
        }
    }
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &Path, _mode: u32) {
    // Windows ACLs inherit from the parent directory; leave alone.
}

impl SessionManager {
    pub async fn open(path: Option<&Path>) -> Result<Self> {
        let database_path = match path {
            Some(path) => path.to_path_buf(),
            None => default_database_path()?,
        };

        // In-memory SQLite databases (used by tests) have no on-disk parent;
        // give each `open()` call its own ephemeral lock dir under the system
        // temp directory so concurrent tests don't share lock files.
        let is_in_memory = database_path == Path::new(":memory:");
        let lock_dir = if is_in_memory {
            std::env::temp_dir().join(format!("agsh-test-locks-{}", Uuid::new_v4()))
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

        // Pre-touch the DB file at 0600 so SQLite's `Connection::open` reuses
        // an already-restricted file rather than creating one at umask
        // defaults that we then chmod down — the latter leaves a window
        // where another local user could open the file. `-wal`/`-shm`
        // companions still inherit the umask, but the parent directory's
        // 0700 mode keeps them inaccessible to other users.
        #[cfg(unix)]
        if !is_in_memory {
            use std::os::unix::fs::OpenOptionsExt;
            let _ = std::fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .mode(0o600)
                .open(&database_path)
                .map_err(|error| {
                    AgshError::Database(format!(
                        "failed to pre-touch database '{}': {}",
                        database_path.display(),
                        error
                    ))
                })?;
        }

        let connection = Connection::open(&database_path)
            .await
            .map_err(|error| AgshError::Database(format!("failed to open database: {}", error)))?;

        // Belt-and-braces: if the file pre-existed at a more permissive mode
        // (manual setup, restored backup, etc.), tighten it now. The
        // pre-touch above is the primary protection for newly-created files.
        if !is_in_memory {
            restrict_permissions(&database_path, 0o600);
        }

        let manager = Self {
            connection: Arc::new(connection),
            lock_dir,
        };
        manager.initialize_schema().await?;
        Ok(manager)
    }

    async fn initialize_schema(&self) -> Result<()> {
        self.connection
            .call(|connection| -> rusqlite::Result<_> {
                connection.execute_batch(
                    "CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata TEXT
                    );

                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions(id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_messages_session_id
                        ON messages(session_id);

                    CREATE TABLE IF NOT EXISTS oauth_tokens (
                        provider TEXT PRIMARY KEY,
                        access_token TEXT NOT NULL,
                        refresh_token TEXT,
                        expires_at INTEGER,
                        account_id TEXT,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS mcp_oauth_credentials (
                        server_name TEXT PRIMARY KEY,
                        credentials_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    ",
                )?;

                // Migration: recreate tool_outputs if it has the old integer-ID schema.
                let has_old_schema: bool = connection
                    .query_row(
                        "SELECT COUNT(*) FROM pragma_table_info('tool_outputs') WHERE name = 'id'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;
                if has_old_schema {
                    connection.execute_batch("DROP TABLE tool_outputs")?;
                }

                // Migration: drop the legacy `sessions.locked_by` column.
                // Locks are now OS file locks managed via `SessionLock`, so
                // any value left in this column is meaningless and a stale
                // PID can permanently lock a session if the column survives.
                let has_locked_by: bool = connection
                    .query_row(
                        "SELECT COUNT(*) FROM pragma_table_info('sessions') WHERE name = 'locked_by'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;
                if has_locked_by {
                    connection.execute_batch("ALTER TABLE sessions DROP COLUMN locked_by")?;
                }

                // Migration: add `oauth_tokens.account_id` for openai-codex's
                // ChatGPT-Account-ID header. Existing rows get NULL, which is
                // fine for providers that don't use it (Claude OAuth).
                let has_account_id: bool = connection
                    .query_row(
                        "SELECT COUNT(*) FROM pragma_table_info('oauth_tokens') WHERE name = 'account_id'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;
                if !has_account_id {
                    connection.execute_batch("ALTER TABLE oauth_tokens ADD COLUMN account_id TEXT")?;
                }

                connection.execute_batch(
                    "CREATE TABLE IF NOT EXISTS tool_outputs (
                        session_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (session_id, name),
                        FOREIGN KEY (session_id) REFERENCES sessions(id)
                    );

                    CREATE TABLE IF NOT EXISTS mcp_auth_cache (
                        server_name TEXT PRIMARY KEY,
                        needs_auth INTEGER NOT NULL,
                        cached_at INTEGER NOT NULL
                    );",
                )?;

                Ok(())
            })
            .await
            .map_err(|error| AgshError::Database(format!("failed to initialize schema: {}", error)))
    }

    pub async fn create_session(&self) -> Result<Uuid> {
        let session_id = Uuid::new_v4();
        let now = chrono::Utc::now().to_rfc3339();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO sessions (id, created_at, updated_at) VALUES (?1, ?2, ?3)",
                    rusqlite::params![session_id.to_string(), now, now],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| AgshError::Database(format!("failed to create session: {}", error)))?;

        Ok(session_id)
    }

    /// Acquire an exclusive OS file lock on the session. Returns a
    /// [`SessionLock`] handle whose lifetime owns the lock; drop it (or let
    /// the process exit) to release.
    ///
    /// The session must already exist in the database. Returns
    /// [`AgshError::SessionLocked`] if another live process holds the lock.
    pub fn lock_session(&self, session_id: Uuid) -> Result<SessionLock> {
        let path = self.lock_dir.join(format!("{}.lock", session_id));
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .map_err(|error| {
                AgshError::Database(format!(
                    "failed to open session lock file '{}': {}",
                    path.display(),
                    error
                ))
            })?;

        let mut lock = Box::new(FdRwLock::new(file));
        let guard = match lock.try_write() {
            Ok(guard) => guard,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                return Err(AgshError::SessionLocked(session_id));
            }
            Err(error) => {
                return Err(AgshError::Database(format!(
                    "failed to acquire session lock '{}': {}",
                    path.display(),
                    error
                )));
            }
        };

        // SAFETY: `guard` borrows from `*lock`. We move the box (not the
        // RwLock inside it) into the returned `SessionLock`, so the RwLock's
        // heap address is stable for as long as the box lives. Field
        // declaration order in `SessionLock` ensures `_guard` is dropped
        // before `_lock`, so the borrow never outlives the borrowee.
        let guard: FdRwLockWriteGuard<'static, File> = unsafe { std::mem::transmute(guard) };

        Ok(SessionLock {
            _guard: guard,
            _lock: lock,
        })
    }

    /// Persist a single event from the conversation log. Events are
    /// encoded into the existing `messages(role, content, …)` table:
    ///
    /// - `Event::Append(message)` writes one row with the message's
    ///   role (`user` / `assistant` / `tool_results`).
    /// - `Event::CompactBoundary { … }` writes one row with the
    ///   pseudo-role `compact_boundary` and a JSON-serialized envelope
    ///   in `content`.
    ///
    /// No schema migration: legacy databases (predating this commit) only
    /// contain `Event::Append` rows; loading them via [`Self::load_events`]
    /// yields the same events the in-memory log produced before.
    pub async fn save_event(
        &self,
        session_id: Uuid,
        event: &crate::conversation::Event,
    ) -> Result<()> {
        let (role, content) = encode_event_for_db(event)
            .map_err(|error| AgshError::Database(format!("failed to encode event: {}", error)))?;
        self.save_message(session_id, &role, &content).await
    }

    /// Load every event for a session in chronological order. Legacy
    /// rows (role ∈ {`user`, `assistant`, `tool_results`}) are
    /// reconstructed as `Event::Append`; rows with role
    /// `compact_boundary` are deserialized from the JSON envelope.
    /// Unknown roles are skipped with a warning.
    pub async fn load_events(&self, session_id: Uuid) -> Result<Vec<crate::conversation::Event>> {
        let stored = self.load_messages(session_id).await?;
        let mut events = Vec::with_capacity(stored.len());
        for row in stored {
            match decode_event_from_row(&row) {
                Ok(Some(event)) => events.push(event),
                Ok(None) => {
                    tracing::warn!("dropping unparseable session row (role={})", row.role);
                }
                Err(error) => {
                    tracing::warn!(
                        "failed to decode session row (role={}): {}",
                        row.role,
                        error
                    );
                }
            }
        }
        Ok(events)
    }

    /// Persist a single row into the `messages` table. Internal helper
    /// for [`Self::save_event`]; external consumers go through the event
    /// API. Tests still call this directly to populate fixtures.
    pub(super) async fn save_message(
        &self,
        session_id: Uuid,
        role: &str,
        content: &str,
    ) -> Result<()> {
        let role = role.to_string();
        let content = content.to_string();
        let now = chrono::Utc::now().to_rfc3339();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO messages (session_id, role, content, created_at) VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![session_id.to_string(), role, content, now],
                )?;

                connection.execute(
                    "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![
                        chrono::Utc::now().to_rfc3339(),
                        session_id.to_string()
                    ],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| AgshError::Database(format!("failed to save message: {}", error)))
    }

    /// Fetch raw rows for a session. Internal helper for
    /// [`Self::load_events`]; external consumers go through the event API.
    async fn load_messages(&self, session_id: Uuid) -> Result<Vec<StoredMessage>> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "SELECT role, content, created_at FROM messages WHERE session_id = ?1 ORDER BY id ASC",
                )?;

                let messages = statement
                    .query_map(rusqlite::params![session_id.to_string()], |row| {
                        Ok(StoredMessage {
                            role: row.get(0)?,
                            content: row.get(1)?,
                            created_at: row.get(2)?,
                        })
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;

                Ok(messages)
            })
            .await
            .map_err(|error| AgshError::Database(format!("failed to load messages: {}", error)))
    }

    pub async fn last_session_id(&self) -> Result<Option<Uuid>> {
        self.connection
            .call(|connection| -> rusqlite::Result<_> {
                let result: std::result::Result<String, _> = connection.query_row(
                    "SELECT id FROM sessions ORDER BY updated_at DESC LIMIT 1",
                    [],
                    |row| row.get(0),
                );

                match result {
                    Ok(id_str) => {
                        let uuid = Uuid::parse_str(&id_str).map_err(|error| {
                            rusqlite::Error::InvalidParameterName(error.to_string())
                        })?;
                        Ok(Some(uuid))
                    }
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(error) => Err(error),
                }
            })
            .await
            .map_err(|error| AgshError::Database(format!("failed to get last session: {}", error)))
    }

    pub async fn session_exists(&self, session_id: Uuid) -> Result<bool> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let count: i64 = connection.query_row(
                    "SELECT COUNT(*) FROM sessions WHERE id = ?1",
                    rusqlite::params![session_id.to_string()],
                    |row| row.get(0),
                )?;
                Ok(count > 0)
            })
            .await
            .map_err(|error| {
                AgshError::Database(format!("failed to check session existence: {}", error))
            })
    }

    /// Resolve a session-ID prefix (e.g. `d64`) to the matching full UUIDs.
    ///
    /// Used by `agsh -c <prefix>` so the user doesn't have to type the
    /// whole UUID. Capped at 16 matches; ordered most-recent-first so the
    /// caller's "ambiguous prefix" listing leads with the session the user
    /// most likely meant.
    ///
    /// Anything outside the UUID alphabet (`0-9a-fA-F-`) returns an empty
    /// list — both because such a prefix can't match any real session ID
    /// and to keep SQL `LIKE` wildcards (`%`, `_`) from sneaking through.
    pub async fn find_sessions_by_prefix(&self, prefix: &str) -> Result<Vec<Uuid>> {
        if prefix.is_empty() || !prefix.chars().all(|c| c.is_ascii_hexdigit() || c == '-') {
            return Ok(Vec::new());
        }
        let pattern = format!("{}%", prefix);
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "SELECT id FROM sessions WHERE id LIKE ?1 \
                     ORDER BY updated_at DESC LIMIT 16",
                )?;
                let rows = statement.query_map(rusqlite::params![pattern], |row| {
                    let id: String = row.get(0)?;
                    Ok(id)
                })?;
                let mut ids = Vec::new();
                for row in rows {
                    if let Ok(uuid) = Uuid::parse_str(&row?) {
                        ids.push(uuid);
                    }
                }
                Ok(ids)
            })
            .await
            .map_err(|error| {
                AgshError::Database(format!("failed to find sessions by prefix: {}", error))
            })
    }

    pub async fn list_sessions(&self, limit: u32) -> Result<Vec<SessionSummary>> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "SELECT s.id, s.updated_at,
                            COALESCE(
                              (SELECT content FROM messages
                               WHERE session_id = s.id AND role = 'user'
                               ORDER BY id ASC LIMIT 1),
                              ''
                            ) AS preview
                     FROM sessions s
                     ORDER BY s.updated_at DESC
                     LIMIT ?1",
                )?;

                let rows = statement.query_map(rusqlite::params![limit], |row| {
                    let id_str: String = row.get(0)?;
                    let updated_at: String = row.get(1)?;
                    let preview: String = row.get(2)?;
                    Ok((id_str, updated_at, preview))
                })?;

                let mut summaries = Vec::new();
                for row in rows {
                    let (id_str, updated_at, preview) = row?;
                    let id = Uuid::parse_str(&id_str).map_err(|error| {
                        rusqlite::Error::InvalidParameterName(error.to_string())
                    })?;

                    let preview = truncate_preview(&preview, 80);

                    summaries.push(SessionSummary {
                        id,
                        updated_at,
                        preview,
                    });
                }
                Ok(summaries)
            })
            .await
            .map_err(|error| AgshError::Database(format!("failed to list sessions: {}", error)))
    }

    pub async fn delete_expired_sessions(&self, retention_days: u64) -> Result<u64> {
        let cutoff = chrono::Utc::now() - chrono::TimeDelta::days(retention_days as i64);
        let cutoff_str = cutoff.to_rfc3339();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM tool_outputs WHERE session_id IN (
                        SELECT id FROM sessions WHERE updated_at < ?1
                    )",
                    rusqlite::params![cutoff_str],
                )?;

                connection.execute(
                    "DELETE FROM messages WHERE session_id IN (
                        SELECT id FROM sessions WHERE updated_at < ?1
                    )",
                    rusqlite::params![cutoff_str],
                )?;

                let deleted = connection.execute(
                    "DELETE FROM sessions WHERE updated_at < ?1",
                    rusqlite::params![cutoff_str],
                )?;

                Ok(deleted as u64)
            })
            .await
            .map_err(|error| {
                AgshError::Database(format!("failed to delete expired sessions: {}", error))
            })
    }

    #[cfg(test)]
    pub async fn clear_messages(&self, session_id: Uuid) -> Result<()> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM tool_outputs WHERE session_id = ?1",
                    rusqlite::params![session_id.to_string()],
                )?;

                connection.execute(
                    "DELETE FROM messages WHERE session_id = ?1",
                    rusqlite::params![session_id.to_string()],
                )?;

                connection.execute(
                    "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![chrono::Utc::now().to_rfc3339(), session_id.to_string()],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| AgshError::Database(format!("failed to clear messages: {}", error)))
    }

    pub async fn delete_session(&self, session_id: Uuid) -> Result<bool> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM tool_outputs WHERE session_id = ?1",
                    rusqlite::params![session_id.to_string()],
                )?;

                connection.execute(
                    "DELETE FROM messages WHERE session_id = ?1",
                    rusqlite::params![session_id.to_string()],
                )?;

                let deleted = connection.execute(
                    "DELETE FROM sessions WHERE id = ?1",
                    rusqlite::params![session_id.to_string()],
                )?;

                Ok(deleted > 0)
            })
            .await
            .map_err(|error| AgshError::Database(format!("failed to delete session: {}", error)))
    }

    pub async fn delete_all_sessions(&self) -> Result<u64> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute("DELETE FROM tool_outputs", [])?;
                connection.execute("DELETE FROM messages", [])?;
                let deleted = connection.execute("DELETE FROM sessions", [])?;
                Ok(deleted as u64)
            })
            .await
            .map_err(|error| {
                AgshError::Database(format!("failed to delete all sessions: {}", error))
            })
    }

    pub async fn save_tool_output(
        &self,
        session_id: Uuid,
        name: &str,
        content: &str,
    ) -> Result<()> {
        let name = name.to_string();
        let content = content.to_string();
        let now = chrono::Utc::now().to_rfc3339();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT OR REPLACE INTO tool_outputs (session_id, name, content, created_at) \
                     VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![session_id.to_string(), name, content, now],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| AgshError::Database(format!("failed to save tool output: {}", error)))
    }

    pub async fn update_tool_output(
        &self,
        session_id: Uuid,
        name: &str,
        content: &str,
    ) -> Result<bool> {
        let name = name.to_string();
        let content = content.to_string();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let updated = connection.execute(
                    "UPDATE tool_outputs SET content = ?1 \
                     WHERE session_id = ?2 AND name = ?3",
                    rusqlite::params![content, session_id.to_string(), name],
                )?;
                Ok(updated > 0)
            })
            .await
            .map_err(|error| {
                AgshError::Database(format!("failed to update tool output: {}", error))
            })
    }

    pub async fn delete_tool_output(&self, session_id: Uuid, name: &str) -> Result<bool> {
        let name = name.to_string();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let deleted = connection.execute(
                    "DELETE FROM tool_outputs WHERE session_id = ?1 AND name = ?2",
                    rusqlite::params![session_id.to_string(), name],
                )?;
                Ok(deleted > 0)
            })
            .await
            .map_err(|error| {
                AgshError::Database(format!("failed to delete tool output: {}", error))
            })
    }

    pub async fn list_tool_outputs(&self, session_id: Uuid) -> Result<Vec<ToolOutputSummary>> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "SELECT name, LENGTH(content), created_at \
                     FROM tool_outputs WHERE session_id = ?1 ORDER BY created_at ASC",
                )?;

                let rows = statement
                    .query_map(rusqlite::params![session_id.to_string()], |row| {
                        Ok(ToolOutputSummary {
                            name: row.get(0)?,
                            size: row.get::<_, i64>(1)? as usize,
                            created_at: row.get(2)?,
                        })
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;

                Ok(rows)
            })
            .await
            .map_err(|error| AgshError::Database(format!("failed to list tool outputs: {}", error)))
    }

    pub async fn load_tool_output(&self, session_id: Uuid, name: &str) -> Result<Option<String>> {
        let name = name.to_string();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let result = connection.query_row(
                    "SELECT content FROM tool_outputs \
                     WHERE session_id = ?1 AND name = ?2",
                    rusqlite::params![session_id.to_string(), name],
                    |row| row.get::<_, String>(0),
                );

                match result {
                    Ok(content) => Ok(Some(content)),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(error) => Err(error),
                }
            })
            .await
            .map_err(|error| AgshError::Database(format!("failed to load tool output: {}", error)))
    }

    pub async fn load_all_tool_outputs(&self, session_id: Uuid) -> Result<Vec<(String, String)>> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "SELECT name, content FROM tool_outputs \
                     WHERE session_id = ?1 ORDER BY created_at ASC",
                )?;

                let rows = statement
                    .query_map(rusqlite::params![session_id.to_string()], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;

                Ok(rows)
            })
            .await
            .map_err(|error| AgshError::Database(format!("failed to load tool outputs: {}", error)))
    }

    pub async fn enforce_storage_limit(&self, max_bytes: u64) -> Result<u64> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut deleted: u64 = 0;

                loop {
                    let total_bytes: i64 = connection.query_row(
                        "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM messages",
                        [],
                        |row| row.get(0),
                    )?;

                    if (total_bytes as u64) <= max_bytes {
                        break;
                    }

                    let oldest_id: std::result::Result<String, _> = connection.query_row(
                        "SELECT id FROM sessions ORDER BY updated_at ASC LIMIT 1",
                        [],
                        |row| row.get(0),
                    );

                    match oldest_id {
                        Ok(session_id) => {
                            connection.execute(
                                "DELETE FROM tool_outputs WHERE session_id = ?1",
                                rusqlite::params![session_id],
                            )?;
                            connection.execute(
                                "DELETE FROM messages WHERE session_id = ?1",
                                rusqlite::params![session_id],
                            )?;
                            connection.execute(
                                "DELETE FROM sessions WHERE id = ?1",
                                rusqlite::params![session_id],
                            )?;
                            deleted += 1;
                        }
                        Err(rusqlite::Error::QueryReturnedNoRows) => break,
                        Err(error) => return Err(error),
                    }
                }

                Ok(deleted)
            })
            .await
            .map_err(|error| {
                AgshError::Database(format!("failed to enforce storage limit: {}", error))
            })
    }
}

#[derive(Clone)]
pub struct TokenStore {
    connection: Arc<Connection>,
}

impl TokenStore {
    pub async fn load_oauth_token(&self, provider: &str) -> Result<Option<AuthCredential>> {
        let provider = provider.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let result = connection.query_row(
                    "SELECT access_token, refresh_token, expires_at, account_id \
                     FROM oauth_tokens WHERE provider = ?1",
                    rusqlite::params![provider],
                    |row| {
                        let access_token: String = row.get(0)?;
                        let refresh_token: Option<String> = row.get(1)?;
                        let expires_at: Option<i64> = row.get(2)?;
                        let account_id: Option<String> = row.get(3)?;
                        Ok(AuthCredential::OAuthToken {
                            access_token,
                            refresh_token,
                            expires_at,
                            account_id,
                        })
                    },
                );

                match result {
                    Ok(credential) => Ok(Some(credential)),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(error) => Err(error),
                }
            })
            .await
            .map_err(|error| AgshError::Database(format!("failed to load OAuth token: {}", error)))
    }

    pub async fn load_mcp_credentials(&self, server_name: &str) -> Result<Option<String>> {
        let server_name = server_name.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let result = connection.query_row(
                    "SELECT credentials_json FROM mcp_oauth_credentials WHERE server_name = ?1",
                    rusqlite::params![server_name],
                    |row| row.get::<_, String>(0),
                );

                match result {
                    Ok(json) => Ok(Some(json)),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(error) => Err(error),
                }
            })
            .await
            .map_err(|error| {
                AgshError::Database(format!("failed to load MCP credentials: {}", error))
            })
    }

    pub async fn save_mcp_credentials(&self, server_name: &str, json: &str) -> Result<()> {
        let server_name = server_name.to_string();
        let json = json.to_string();
        let now = chrono::Utc::now().to_rfc3339();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO mcp_oauth_credentials (server_name, credentials_json, updated_at)
                     VALUES (?1, ?2, ?3)
                     ON CONFLICT(server_name) DO UPDATE SET
                         credentials_json = excluded.credentials_json,
                         updated_at = excluded.updated_at",
                    rusqlite::params![server_name, json, now],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                AgshError::Database(format!("failed to save MCP credentials: {}", error))
            })
    }

    pub async fn clear_mcp_credentials(&self, server_name: &str) -> Result<()> {
        let server_name = server_name.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM mcp_oauth_credentials WHERE server_name = ?1",
                    rusqlite::params![server_name],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                AgshError::Database(format!("failed to clear MCP credentials: {}", error))
            })
    }

    /// Load a cached needs-auth verdict for an MCP server, or `None` if
    /// there is no entry or it is older than `ttl`.
    pub async fn load_auth_probe(
        &self,
        server_name: &str,
        ttl: std::time::Duration,
    ) -> Result<Option<bool>> {
        let server_name = server_name.to_string();
        let ttl_seconds = ttl.as_secs() as i64;
        let now = chrono::Utc::now().timestamp();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let result = connection.query_row(
                    "SELECT needs_auth, cached_at FROM mcp_auth_cache WHERE server_name = ?1",
                    rusqlite::params![server_name],
                    |row| {
                        let needs_auth: i64 = row.get(0)?;
                        let cached_at: i64 = row.get(1)?;
                        Ok((needs_auth != 0, cached_at))
                    },
                );
                match result {
                    // Strict `<` so a zero-duration TTL behaves as "never
                    // cache" instead of "cache for the rest of this second".
                    Ok((needs_auth, cached_at)) if now - cached_at < ttl_seconds => {
                        Ok(Some(needs_auth))
                    }
                    Ok(_) => Ok(None),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(error) => Err(error),
                }
            })
            .await
            .map_err(|error| {
                AgshError::Database(format!("failed to load MCP auth probe cache: {}", error))
            })
    }

    /// Persist a needs-auth verdict for an MCP server (TTL is enforced at
    /// load time, so we just record the current timestamp here).
    pub async fn save_auth_probe(&self, server_name: &str, needs_auth: bool) -> Result<()> {
        let server_name = server_name.to_string();
        let now = chrono::Utc::now().timestamp();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO mcp_auth_cache (server_name, needs_auth, cached_at)
                     VALUES (?1, ?2, ?3)
                     ON CONFLICT(server_name) DO UPDATE SET
                         needs_auth = excluded.needs_auth,
                         cached_at = excluded.cached_at",
                    rusqlite::params![server_name, if needs_auth { 1_i64 } else { 0_i64 }, now],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                AgshError::Database(format!("failed to save MCP auth probe cache: {}", error))
            })
    }

    /// Remove any cached needs-auth verdict for an MCP server.
    pub async fn clear_auth_probe(&self, server_name: &str) -> Result<()> {
        let server_name = server_name.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM mcp_auth_cache WHERE server_name = ?1",
                    rusqlite::params![server_name],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                AgshError::Database(format!("failed to clear MCP auth probe cache: {}", error))
            })
    }

    pub async fn save_oauth_token(
        &self,
        provider: &str,
        credential: &AuthCredential,
    ) -> Result<()> {
        let AuthCredential::OAuthToken {
            access_token,
            refresh_token,
            expires_at,
            account_id,
        } = credential
        else {
            return Ok(());
        };

        let provider = provider.to_string();
        let access_token = access_token.clone();
        let refresh_token = refresh_token.clone();
        let expires_at = *expires_at;
        let account_id = account_id.clone();
        let now = chrono::Utc::now().to_rfc3339();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO oauth_tokens \
                         (provider, access_token, refresh_token, expires_at, account_id, updated_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
                     ON CONFLICT(provider) DO UPDATE SET \
                         access_token = excluded.access_token, \
                         refresh_token = excluded.refresh_token, \
                         expires_at = excluded.expires_at, \
                         account_id = excluded.account_id, \
                         updated_at = excluded.updated_at",
                    rusqlite::params![
                        provider,
                        access_token,
                        refresh_token,
                        expires_at,
                        account_id,
                        now
                    ],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| AgshError::Database(format!("failed to save OAuth token: {}", error)))
    }
}

impl SessionManager {
    pub fn token_store(&self) -> TokenStore {
        TokenStore {
            connection: Arc::clone(&self.connection),
        }
    }
}

/// Strip `<context>...</context>` tags from a stored user message,
/// returning only the actual user input.
pub fn strip_context_tags(text: &str) -> &str {
    const CLOSING_TAG: &str = "</context>";
    if let Some(end) = text.find(CLOSING_TAG) {
        let after = &text[end + CLOSING_TAG.len()..];
        after.trim_start_matches('\n')
    } else {
        text
    }
}

/// Pseudo-role used in the `messages` table for `Event::CompactBoundary`
/// rows. Coexists with the legacy `user`/`assistant`/`tool_results` roles
/// without a schema migration.
const COMPACT_BOUNDARY_ROLE: &str = "compact_boundary";

/// Encode an [`crate::conversation::Event`] into the `(role, content)`
/// columns of the `messages` table. `Event::Append` writes the message's
/// natural role; `Event::CompactBoundary` writes a JSON envelope under
/// the [`COMPACT_BOUNDARY_ROLE`] pseudo-role.
fn encode_event_for_db(
    event: &crate::conversation::Event,
) -> std::result::Result<(String, String), serde_json::Error> {
    use crate::conversation::Event;
    use crate::provider::{ContentBlock, Role};

    match event {
        Event::Append(message) => {
            let (role, content) = match message.role {
                Role::User => {
                    if message
                        .content
                        .iter()
                        .any(|block| matches!(block, ContentBlock::ToolResult { .. }))
                    {
                        ("tool_results", serde_json::to_string(&message.content)?)
                    } else {
                        ("user", message.text_content())
                    }
                }
                Role::Assistant => ("assistant", serde_json::to_string(&message.content)?),
            };
            Ok((role.to_string(), content))
        }
        Event::CompactBoundary { .. } => {
            let content = serde_json::to_string(event)?;
            Ok((COMPACT_BOUNDARY_ROLE.to_string(), content))
        }
    }
}

/// Decode one persisted row back into an [`crate::conversation::Event`].
/// Returns `Ok(None)` when the row's role is unrecognised (forward-
/// compat for new variants).
fn decode_event_from_row(
    row: &StoredMessage,
) -> std::result::Result<Option<crate::conversation::Event>, serde_json::Error> {
    use crate::conversation::Event;
    use crate::provider::{ContentBlock, Message, Role};

    match row.role.as_str() {
        "user" => Ok(Some(Event::Append(Message::user(&row.content)))),
        "assistant" => match serde_json::from_str::<Vec<ContentBlock>>(&row.content) {
            Ok(content) => Ok(Some(Event::Append(Message {
                role: Role::Assistant,
                content,
            }))),
            Err(_) => {
                // Legacy or malformed JSON: fall back to text.
                Ok(Some(Event::Append(Message::assistant_text(&row.content))))
            }
        },
        "tool_results" => match serde_json::from_str::<Vec<ContentBlock>>(&row.content) {
            Ok(content) => Ok(Some(Event::Append(Message {
                role: Role::User,
                content,
            }))),
            Err(error) => Err(error),
        },
        role if role == COMPACT_BOUNDARY_ROLE => {
            let event: Event = serde_json::from_str(&row.content)?;
            Ok(Some(event))
        }
        _ => Ok(None),
    }
}

fn truncate_preview(text: &str, max_chars: usize) -> String {
    let text = strip_context_tags(text);
    let first_line = text.lines().next().unwrap_or("");
    if first_line.chars().count() <= max_chars {
        first_line.to_string()
    } else {
        let truncated: String = first_line.chars().take(max_chars).collect();
        format!("{}…", truncated.trim_end())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_manager() -> SessionManager {
        SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("failed to open in-memory database")
    }

    /// Persist one of every event variant via `save_event` and read it
    /// back through `load_events`. Verifies the encoding/decoding round
    /// trip — including the JSON envelope used for `CompactBoundary` —
    /// matches the in-memory shape.
    #[tokio::test]
    async fn test_save_and_load_events_round_trip() {
        use std::collections::HashSet;

        use crate::conversation::Event;
        use crate::provider::{ContentBlock, Message, Role, ToolResultContent};

        let manager = test_manager().await;
        let sid = manager.create_session().await.expect("create session");

        let user_event = Event::Append(Message::user("hello"));
        let assistant_event = Event::Append(Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "thinking aloud".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "u1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/x"}),
                },
            ],
        });
        let tool_result_event = Event::Append(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "u1".to_string(),
                content: vec![ToolResultContent::Text {
                    text: "ok".to_string(),
                }],
                is_error: false,
            }],
        });
        let snapshot: HashSet<String> = ["mcp__notion__fetch".to_string()].into_iter().collect();
        let boundary_event = Event::CompactBoundary {
            summary: Message::user("[summary]"),
            replaced_count: 3,
            loaded_tools_snapshot: snapshot,
        };

        for event in [
            &user_event,
            &assistant_event,
            &tool_result_event,
            &boundary_event,
        ] {
            manager.save_event(sid, event).await.expect("save event");
        }

        let loaded = manager.load_events(sid).await.expect("load events");
        assert_eq!(loaded.len(), 4);

        match &loaded[0] {
            Event::Append(m) => assert_eq!(m.text_content(), "hello"),
            _ => panic!("expected user Append"),
        }
        match &loaded[1] {
            Event::Append(m) => {
                assert_eq!(m.role, Role::Assistant);
                assert_eq!(m.content.len(), 2);
                assert!(matches!(&m.content[1], ContentBlock::ToolUse { id, .. } if id == "u1"));
            }
            _ => panic!("expected assistant Append"),
        }
        match &loaded[2] {
            Event::Append(m) => {
                assert_eq!(m.role, Role::User);
                assert!(matches!(
                    &m.content[0],
                    ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "u1"
                ));
            }
            _ => panic!("expected tool_results Append"),
        }
        match &loaded[3] {
            Event::CompactBoundary {
                replaced_count,
                loaded_tools_snapshot,
                summary,
            } => {
                assert_eq!(*replaced_count, 3);
                assert!(loaded_tools_snapshot.contains("mcp__notion__fetch"));
                assert_eq!(summary.text_content(), "[summary]");
            }
            _ => panic!("expected CompactBoundary"),
        }
    }

    /// Legacy databases (predating PR 2) only contain rows with the
    /// `user` / `assistant` / `tool_results` roles — no `compact_boundary`.
    /// `load_events` must hydrate every legacy row as an `Event::Append`
    /// so resume works without a schema migration.
    #[tokio::test]
    async fn test_load_events_legacy_rows_as_append() {
        use crate::conversation::Event;

        let manager = test_manager().await;
        let sid = manager.create_session().await.expect("create session");

        // Simulate a pre-PR-2 session by writing rows the legacy way.
        manager
            .save_message(sid, "user", "first")
            .await
            .expect("save user");
        let assistant_blocks = serde_json::json!([
            {"type": "text", "text": "answer"}
        ])
        .to_string();
        manager
            .save_message(sid, "assistant", &assistant_blocks)
            .await
            .expect("save assistant");

        let events = manager.load_events(sid).await.expect("load events");
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|e| matches!(e, Event::Append(_))));
    }

    /// A row with an unknown role should be skipped (with a warning) so
    /// a future schema bump that adds new event variants doesn't crash
    /// older binaries reading newer DBs.
    #[tokio::test]
    async fn test_load_events_skips_unknown_role() {
        let manager = test_manager().await;
        let sid = manager.create_session().await.expect("create session");
        manager
            .save_message(sid, "user", "real")
            .await
            .expect("save real row");
        manager
            .save_message(sid, "future_event_kind", "{}")
            .await
            .expect("save unknown row");
        let events = manager.load_events(sid).await.expect("load events");
        assert_eq!(events.len(), 1);
    }

    /// Regression test for the umask-dependent permission bug: the session
    /// database file stores OAuth tokens and MCP credentials, so it must be
    /// readable by the owner only (0600) and the surrounding directory by
    /// the owner only (0700), regardless of the user's umask.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_session_db_file_mode() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = tempfile::tempdir().expect("tempdir");
        let db_path = temp_dir.path().join("data").join("sessions.db");

        let _manager = SessionManager::open(Some(&db_path))
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

    #[tokio::test]
    async fn test_create_session() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session()
            .await
            .expect("failed to create session");
        assert!(
            manager
                .session_exists(session_id)
                .await
                .expect("failed to check")
        );
    }

    #[tokio::test]
    async fn test_save_and_load_messages() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session()
            .await
            .expect("failed to create session");

        manager
            .save_message(session_id, "user", "hello")
            .await
            .expect("failed to save message");
        manager
            .save_message(session_id, "assistant", "hi there")
            .await
            .expect("failed to save message");

        let messages = manager
            .load_messages(session_id)
            .await
            .expect("failed to load messages");

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "hello");
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[1].content, "hi there");
    }

    #[tokio::test]
    async fn test_last_session_id() {
        let manager = test_manager().await;
        assert!(manager.last_session_id().await.expect("failed").is_none());

        let session_id = manager
            .create_session()
            .await
            .expect("failed to create session");
        let last = manager
            .last_session_id()
            .await
            .expect("failed to get last session");
        assert_eq!(last, Some(session_id));
    }

    #[tokio::test]
    async fn test_find_sessions_by_prefix_empty_db() {
        let manager = test_manager().await;
        let matches = manager
            .find_sessions_by_prefix("abc")
            .await
            .expect("failed prefix lookup");
        assert!(matches.is_empty());
    }

    #[tokio::test]
    async fn test_find_sessions_by_prefix_unique_match() {
        let manager = test_manager().await;
        let id = manager
            .create_session()
            .await
            .expect("failed to create session");
        // First 8 hex chars (before the first dash) — guaranteed unique
        // for a freshly-generated random UUID with only one row in the DB.
        let prefix: String = id.to_string().chars().take(8).collect();
        let matches = manager
            .find_sessions_by_prefix(&prefix)
            .await
            .expect("failed prefix lookup");
        assert_eq!(matches, vec![id]);
    }

    #[tokio::test]
    async fn test_find_sessions_by_prefix_no_match() {
        let manager = test_manager().await;
        manager
            .create_session()
            .await
            .expect("failed to create session");
        let matches = manager
            .find_sessions_by_prefix("ffffffff")
            .await
            .expect("failed prefix lookup");
        // Real UUIDs are random; collision with this prefix is astronomically
        // unlikely but theoretically possible — re-create a session if so.
        assert!(matches.is_empty() || matches.len() == 1);
    }

    #[tokio::test]
    async fn test_find_sessions_by_prefix_rejects_non_hex_chars() {
        let manager = test_manager().await;
        manager
            .create_session()
            .await
            .expect("failed to create session");
        // SQL `%` and `_` wildcards must not slip through as prefix chars.
        for bad in ["%", "_", "abc%", "ab_c", "g0g0", "x123"] {
            let matches = manager
                .find_sessions_by_prefix(bad)
                .await
                .expect("failed prefix lookup");
            assert!(
                matches.is_empty(),
                "non-hex prefix {:?} should match nothing",
                bad
            );
        }
    }

    #[tokio::test]
    async fn test_find_sessions_by_prefix_empty_prefix_matches_nothing() {
        let manager = test_manager().await;
        manager
            .create_session()
            .await
            .expect("failed to create session");
        let matches = manager
            .find_sessions_by_prefix("")
            .await
            .expect("failed prefix lookup");
        assert!(
            matches.is_empty(),
            "empty prefix must not match every session"
        );
    }

    #[tokio::test]
    async fn test_session_locking_acquire_and_release() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session()
            .await
            .expect("failed to create session");

        let lock = manager
            .lock_session(session_id)
            .expect("failed to lock session");

        // While the lock handle is alive, a second attempt must fail.
        match manager.lock_session(session_id) {
            Err(AgshError::SessionLocked(id)) => assert_eq!(id, session_id),
            other => panic!("expected SessionLocked, got {:?}", other.map(|_| "Ok(_)")),
        }

        // Dropping the handle releases the OS lock; re-acquisition succeeds.
        drop(lock);
        let _lock2 = manager
            .lock_session(session_id)
            .expect("failed to re-acquire session lock after drop");
    }

    #[tokio::test]
    async fn test_session_not_found() {
        let manager = test_manager().await;
        let fake_id = Uuid::new_v4();
        assert!(
            !manager
                .session_exists(fake_id)
                .await
                .expect("failed to check")
        );
    }

    #[tokio::test]
    async fn test_multiple_sessions() {
        let manager = test_manager().await;
        let session1 = manager.create_session().await.expect("failed");
        let session2 = manager.create_session().await.expect("failed");

        manager
            .save_message(session1, "user", "msg1")
            .await
            .expect("failed");
        manager
            .save_message(session2, "user", "msg2")
            .await
            .expect("failed");

        let messages1 = manager.load_messages(session1).await.expect("failed");
        let messages2 = manager.load_messages(session2).await.expect("failed");

        assert_eq!(messages1.len(), 1);
        assert_eq!(messages1[0].content, "msg1");
        assert_eq!(messages2.len(), 1);
        assert_eq!(messages2[0].content, "msg2");
    }

    #[tokio::test]
    async fn test_delete_expired_sessions() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("failed");
        manager
            .save_message(session_id, "user", "hello")
            .await
            .expect("failed");

        // Backdate the session to 100 days ago
        let old_date = (chrono::Utc::now() - chrono::TimeDelta::days(100)).to_rfc3339();
        manager
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![old_date, session_id.to_string()],
                )?;
                Ok(())
            })
            .await
            .expect("failed to backdate");

        let deleted = manager
            .delete_expired_sessions(30)
            .await
            .expect("failed to delete");
        assert_eq!(deleted, 1);
        assert!(!manager.session_exists(session_id).await.expect("failed"));

        let messages = manager.load_messages(session_id).await.expect("failed");
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn test_delete_expired_sessions_keeps_recent() {
        let manager = test_manager().await;
        let old_session = manager.create_session().await.expect("failed");
        let new_session = manager.create_session().await.expect("failed");

        manager
            .save_message(old_session, "user", "old")
            .await
            .expect("failed");
        manager
            .save_message(new_session, "user", "new")
            .await
            .expect("failed");

        // Backdate only the old session
        let old_date = (chrono::Utc::now() - chrono::TimeDelta::days(100)).to_rfc3339();
        manager
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![old_date, old_session.to_string()],
                )?;
                Ok(())
            })
            .await
            .expect("failed to backdate");

        let deleted = manager
            .delete_expired_sessions(30)
            .await
            .expect("failed to delete");
        assert_eq!(deleted, 1);
        assert!(!manager.session_exists(old_session).await.expect("failed"));
        assert!(manager.session_exists(new_session).await.expect("failed"));
    }

    #[tokio::test]
    async fn test_enforce_storage_limit() {
        let manager = test_manager().await;
        let session1 = manager.create_session().await.expect("failed");

        // Add enough content to exceed a small limit
        let large_content = "x".repeat(1000);
        manager
            .save_message(session1, "user", &large_content)
            .await
            .expect("failed");

        // Backdate session1 so it's the oldest
        let old_date = (chrono::Utc::now() - chrono::TimeDelta::days(10)).to_rfc3339();
        manager
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![old_date, session1.to_string()],
                )?;
                Ok(())
            })
            .await
            .expect("failed to backdate");

        let session2 = manager.create_session().await.expect("failed");
        manager
            .save_message(session2, "user", "small")
            .await
            .expect("failed");

        // Set a limit smaller than the total, but larger than session2 alone
        let deleted = manager
            .enforce_storage_limit(500)
            .await
            .expect("failed to enforce");
        assert_eq!(deleted, 1);
        assert!(!manager.session_exists(session1).await.expect("failed"));
        assert!(manager.session_exists(session2).await.expect("failed"));
    }

    #[tokio::test]
    async fn test_clear_messages() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("failed");

        manager
            .save_message(session_id, "user", "hello")
            .await
            .expect("failed");
        manager
            .save_message(session_id, "assistant", "hi")
            .await
            .expect("failed");

        let messages = manager.load_messages(session_id).await.expect("failed");
        assert_eq!(messages.len(), 2);

        manager
            .clear_messages(session_id)
            .await
            .expect("failed to clear");

        let messages = manager.load_messages(session_id).await.expect("failed");
        assert!(messages.is_empty());

        // Session itself should still exist
        assert!(manager.session_exists(session_id).await.expect("failed"));
    }

    #[test]
    fn test_strip_context_tags_with_context() {
        let input = "<context>\n[Environment context]\nWorking directory: /tmp\nDate: Mon\n</context>\n\nhello world";
        assert_eq!(strip_context_tags(input), "hello world");
    }

    #[test]
    fn test_strip_context_tags_without_context() {
        let input = "hello world";
        assert_eq!(strip_context_tags(input), "hello world");
    }

    #[test]
    fn test_strip_context_tags_empty_after_context() {
        let input = "<context>\nstuff\n</context>\n\n";
        assert_eq!(strip_context_tags(input), "");
    }

    #[test]
    fn test_truncate_preview_with_context_tags() {
        let input = "<context>\n[Environment context]\nWorking directory: /tmp\n</context>\n\nfind all Rust files";
        assert_eq!(truncate_preview(input, 80), "find all Rust files");
    }

    #[test]
    fn test_truncate_preview_without_context_tags() {
        let input = "find all Rust files";
        assert_eq!(truncate_preview(input, 80), "find all Rust files");
    }

    #[test]
    fn test_truncate_preview_old_format_backward_compat() {
        let input = "[Environment context]\nWorking directory: /tmp\n\nfind all Rust files";
        assert_eq!(truncate_preview(input, 80), "[Environment context]");
    }

    #[test]
    fn test_truncate_preview_with_context_tags_long_input() {
        let long_input = format!("<context>\nstuff\n</context>\n\n{}", "x".repeat(100));
        let preview = truncate_preview(&long_input, 80);
        assert!(preview.ends_with('…'));
        assert!(preview.len() <= 84); // 80 chars + "…"
    }

    // End-to-end regression tests for `agsh list`'s preview.
    // These tests mock the complete pipeline that produces the
    // `Preview` column: build the turn-context block the agent
    // actually sends, prepend it to a user prompt the way
    // `agent::Agent::run_turn` does, persist via `save_message`,
    // then call `list_sessions` and assert the preview matches the
    // raw user prompt. Any future change to:
    //   - `context::build_turn_context`'s output shape
    //   - `agent::Agent::run_turn`'s "prefix block, then user input"
    //     format
    //   - `save_message` storage
    //   - `list_sessions`'s SQL / preview rendering
    //   - `strip_context_tags` / `truncate_preview`
    // that breaks the preview will fail one of these tests.
    //
    // The preview has regressed several times historically; these
    // guards exist so the next breakage is caught by CI, not a user.

    /// Reconstructs what `agent::Agent::run_turn` passes to
    /// `save_message` for a fresh user turn: the `<context>...</context>`
    /// block followed by `\n\n` and the user's raw prompt. Kept
    /// structurally identical to the real call-site (see
    /// `src/agent.rs::run_turn` → `augmented_input = format!("{}\n\n{}", block, user_input)`).
    fn mock_run_turn_user_message(
        permission: crate::permission::Permission,
        user_input: &str,
    ) -> String {
        let block = crate::context::build_turn_context(permission, &[]);
        format!("{}\n\n{}", block, user_input)
    }

    #[tokio::test]
    async fn test_list_sessions_preview_is_user_prompt_not_context_wrapper() {
        // The canonical regression: user types a prompt, turn runs,
        // `agsh list` must show the prompt — not `<context>`, not the
        // permission/environment metadata.
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create_session");

        let user_prompt = "find all Rust files under src/";
        let stored = mock_run_turn_user_message(crate::permission::Permission::Read, user_prompt);
        manager
            .save_message(session_id, "user", &stored)
            .await
            .expect("save_message");

        let summaries = manager.list_sessions(10).await.expect("list_sessions");
        let summary = summaries
            .iter()
            .find(|s| s.id == session_id)
            .expect("session missing from list");

        assert_eq!(
            summary.preview, user_prompt,
            "preview regressed: expected user prompt, got {:?}",
            summary.preview
        );
        assert!(
            !summary.preview.contains("<context>"),
            "wrapper leaked into preview: {:?}",
            summary.preview
        );
        assert!(
            !summary.preview.contains("[Permission context]"),
            "permission metadata leaked into preview: {:?}",
            summary.preview
        );
    }

    #[tokio::test]
    async fn test_list_sessions_preview_covers_all_permission_levels() {
        // The context block's shape differs per permission level
        // (Write omits the [Environment context], None differs again).
        // Every level should still surface the user's prompt cleanly.
        let manager = test_manager().await;
        for (label, permission) in &[
            ("none", crate::permission::Permission::None),
            ("read", crate::permission::Permission::Read),
            ("ask", crate::permission::Permission::Ask),
            ("write", crate::permission::Permission::Write),
        ] {
            let session_id = manager.create_session().await.expect("create_session");
            let prompt = format!("ask at {} level", label);
            let stored = mock_run_turn_user_message(*permission, &prompt);
            manager
                .save_message(session_id, "user", &stored)
                .await
                .expect("save_message");

            let summaries = manager.list_sessions(100).await.expect("list_sessions");
            let summary = summaries
                .iter()
                .find(|s| s.id == session_id)
                .unwrap_or_else(|| panic!("session missing for level {}", label));
            assert_eq!(
                summary.preview, prompt,
                "preview mismatch at permission level {}",
                label
            );
        }
    }

    #[tokio::test]
    async fn test_list_sessions_preview_truncates_long_prompt_with_ellipsis() {
        // Long prompts are capped at 80 chars with a trailing ellipsis.
        // The cap must apply to the user's prompt, not the wrapper.
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create_session");

        let long_prompt = "a".repeat(150);
        let stored = mock_run_turn_user_message(crate::permission::Permission::Read, &long_prompt);
        manager
            .save_message(session_id, "user", &stored)
            .await
            .expect("save_message");

        let summaries = manager.list_sessions(10).await.expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();

        assert!(
            summary.preview.starts_with("aaa"),
            "preview should start with the user's content, not the wrapper: {:?}",
            summary.preview
        );
        assert!(
            summary.preview.ends_with('…'),
            "long preview should end with ellipsis: {:?}",
            summary.preview
        );
        assert!(summary.preview.chars().count() <= 81);
    }

    #[tokio::test]
    async fn test_list_sessions_preview_is_first_user_turn_not_later() {
        // Multiple turns in one session — preview must be the FIRST
        // user prompt, not a later one. `ORDER BY id ASC LIMIT 1`
        // guarantees this; guard against that being changed.
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create_session");

        for (i, prompt) in ["first prompt", "second prompt", "third prompt"]
            .iter()
            .enumerate()
        {
            let stored = mock_run_turn_user_message(crate::permission::Permission::Read, prompt);
            manager
                .save_message(session_id, "user", &stored)
                .await
                .expect("save_message");
            // Interleave an assistant reply — real sessions alternate.
            manager
                .save_message(session_id, "assistant", &format!("reply {}", i))
                .await
                .expect("save_message");
        }

        let summaries = manager.list_sessions(10).await.expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(summary.preview, "first prompt");
    }

    #[tokio::test]
    async fn test_list_sessions_preview_multiline_shows_first_line() {
        // Multi-line user prompts collapse to the first line in the
        // list view. The remaining lines are not leaked.
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create_session");

        let stored = mock_run_turn_user_message(
            crate::permission::Permission::Read,
            "line one is the preview\nline two should not appear\nline three either",
        );
        manager
            .save_message(session_id, "user", &stored)
            .await
            .expect("save_message");

        let summaries = manager.list_sessions(10).await.expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(summary.preview, "line one is the preview");
    }

    #[tokio::test]
    async fn test_list_sessions_preview_independent_per_session() {
        // Each session's preview is its own first user turn — no
        // cross-contamination from neighbour sessions.
        let manager = test_manager().await;
        let a = manager.create_session().await.expect("create_session");
        let b = manager.create_session().await.expect("create_session");
        let c = manager.create_session().await.expect("create_session");

        for (sid, prompt) in [(a, "alpha"), (b, "beta"), (c, "gamma")] {
            let stored = mock_run_turn_user_message(crate::permission::Permission::Read, prompt);
            manager
                .save_message(sid, "user", &stored)
                .await
                .expect("save_message");
        }

        let summaries = manager.list_sessions(10).await.expect("list_sessions");
        let preview_of = |id: uuid::Uuid| {
            summaries
                .iter()
                .find(|s| s.id == id)
                .map(|s| s.preview.clone())
                .unwrap_or_default()
        };
        assert_eq!(preview_of(a), "alpha");
        assert_eq!(preview_of(b), "beta");
        assert_eq!(preview_of(c), "gamma");
    }

    #[tokio::test]
    async fn test_list_sessions_preview_empty_session_has_empty_preview() {
        // A session with zero user messages (e.g. created but Ctrl-C'd
        // before first dispatch) falls back to an empty string — it
        // should not panic or render `<no user msg>` scaffolding.
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create_session");

        let summaries = manager.list_sessions(10).await.expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(summary.preview, "");
    }

    #[tokio::test]
    async fn test_list_sessions_preview_compacted_session() {
        // After `/compact`, the agent clears messages and inserts a
        // single new user message starting with
        // `[Conversation summary from session compaction]`. That has
        // no `<context>` wrapper; `list_sessions` should surface the
        // summary's first line, not an empty preview.
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create_session");

        let summary_msg = "[Conversation summary from session compaction]\n\nSummary text here\n\n\
             [Post-compaction context]\n\n…";
        manager
            .save_message(session_id, "user", summary_msg)
            .await
            .expect("save_message");

        let summaries = manager.list_sessions(10).await.expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(
            summary.preview, "[Conversation summary from session compaction]",
            "compacted session should surface the summary marker as preview"
        );
    }

    #[tokio::test]
    async fn test_list_sessions_preview_legacy_unwrapped_user_message() {
        // Backward-compat: older sessions (or future non-wrapper
        // message paths) whose user content has no `<context>` block
        // at all — the stored string IS the prompt, and the preview
        // equals it.
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("create_session");
        manager
            .save_message(session_id, "user", "legacy prompt without any wrapper")
            .await
            .expect("save_message");

        let summaries = manager.list_sessions(10).await.expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(summary.preview, "legacy prompt without any wrapper");
    }

    #[tokio::test]
    async fn test_enforce_storage_limit_no_deletion_needed() {
        let manager = test_manager().await;
        let session_id = manager.create_session().await.expect("failed");
        manager
            .save_message(session_id, "user", "small")
            .await
            .expect("failed");

        let deleted = manager
            .enforce_storage_limit(1_000_000)
            .await
            .expect("failed to enforce");
        assert_eq!(deleted, 0);
        assert!(manager.session_exists(session_id).await.expect("failed"));
    }

    // MCP TokenStore tests.
    // Exercise the methods backing `agsh mcp login/logout` and the auth-probe
    // cache that skips unauthenticated connects after a 401. In-memory DB
    // keeps each case hermetic.

    #[tokio::test]
    async fn mcp_credentials_round_trip() {
        let manager = test_manager().await;
        let store = manager.token_store();

        assert!(
            store
                .load_mcp_credentials("srv")
                .await
                .expect("load absent")
                .is_none(),
            "no credentials should exist yet"
        );

        store
            .save_mcp_credentials("srv", r#"{"tokens":{"access_token":"at1"}}"#)
            .await
            .expect("save");
        assert_eq!(
            store
                .load_mcp_credentials("srv")
                .await
                .expect("load")
                .as_deref(),
            Some(r#"{"tokens":{"access_token":"at1"}}"#)
        );

        // Upsert: second save replaces the first.
        store
            .save_mcp_credentials("srv", r#"{"tokens":{"access_token":"at2"}}"#)
            .await
            .expect("save again");
        assert_eq!(
            store
                .load_mcp_credentials("srv")
                .await
                .expect("load")
                .as_deref(),
            Some(r#"{"tokens":{"access_token":"at2"}}"#)
        );

        store.clear_mcp_credentials("srv").await.expect("clear");
        assert!(
            store
                .load_mcp_credentials("srv")
                .await
                .expect("load after clear")
                .is_none()
        );
    }

    #[tokio::test]
    async fn mcp_credentials_are_scoped_per_server() {
        let manager = test_manager().await;
        let store = manager.token_store();
        store
            .save_mcp_credentials("a", "alpha")
            .await
            .expect("save a");
        store
            .save_mcp_credentials("b", "beta")
            .await
            .expect("save b");
        store.clear_mcp_credentials("a").await.expect("clear a");
        assert!(store.load_mcp_credentials("a").await.unwrap().is_none());
        assert_eq!(
            store.load_mcp_credentials("b").await.unwrap().as_deref(),
            Some("beta")
        );
    }

    #[tokio::test]
    async fn auth_probe_round_trip_and_scoping() {
        let manager = test_manager().await;
        let store = manager.token_store();
        let ttl = std::time::Duration::from_secs(3600);

        assert!(store.load_auth_probe("srv", ttl).await.unwrap().is_none());

        store.save_auth_probe("srv", true).await.expect("save true");
        assert_eq!(store.load_auth_probe("srv", ttl).await.unwrap(), Some(true));

        // Upsert — flip the verdict.
        store
            .save_auth_probe("srv", false)
            .await
            .expect("save false");
        assert_eq!(
            store.load_auth_probe("srv", ttl).await.unwrap(),
            Some(false)
        );

        // A different server name is unaffected.
        assert!(store.load_auth_probe("other", ttl).await.unwrap().is_none());

        store.clear_auth_probe("srv").await.expect("clear");
        assert!(store.load_auth_probe("srv", ttl).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn auth_probe_honours_ttl() {
        let manager = test_manager().await;
        let store = manager.token_store();
        store.save_auth_probe("srv", true).await.expect("save");
        // A TTL of 0 seconds must treat any entry as stale.
        assert!(
            store
                .load_auth_probe("srv", std::time::Duration::from_secs(0))
                .await
                .unwrap()
                .is_none(),
            "entry should be stale under zero TTL"
        );
        // A large TTL still returns the same entry.
        assert_eq!(
            store
                .load_auth_probe("srv", std::time::Duration::from_secs(3600))
                .await
                .unwrap(),
            Some(true)
        );
    }

    #[tokio::test]
    async fn test_oauth_token_round_trip_preserves_all_fields() {
        let manager = SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("memory store");
        let store = manager.token_store();

        let credential = AuthCredential::OAuthToken {
            access_token: "access-1".to_string(),
            refresh_token: Some("refresh-1".to_string()),
            expires_at: Some(1_700_000_000_000),
            account_id: Some("account-abc".to_string()),
        };

        store
            .save_oauth_token("openai-codex", &credential)
            .await
            .expect("save");

        let loaded = store
            .load_oauth_token("openai-codex")
            .await
            .expect("load")
            .expect("present");

        match loaded {
            AuthCredential::OAuthToken {
                access_token,
                refresh_token,
                expires_at,
                account_id,
            } => {
                assert_eq!(access_token, "access-1");
                assert_eq!(refresh_token.as_deref(), Some("refresh-1"));
                assert_eq!(expires_at, Some(1_700_000_000_000));
                assert_eq!(account_id.as_deref(), Some("account-abc"));
            }
            _ => panic!("expected OAuthToken"),
        }
    }

    #[tokio::test]
    async fn test_oauth_token_round_trip_account_id_optional() {
        // Claude OAuth doesn't populate `account_id` — make sure round-tripping
        // a `None` value works without losing other fields.
        let manager = SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("memory store");
        let store = manager.token_store();

        let credential = AuthCredential::OAuthToken {
            access_token: "claude-token".to_string(),
            refresh_token: None,
            expires_at: None,
            account_id: None,
        };

        store
            .save_oauth_token("claude", &credential)
            .await
            .expect("save");

        let loaded = store.load_oauth_token("claude").await.expect("load");

        match loaded {
            Some(AuthCredential::OAuthToken {
                access_token,
                account_id,
                ..
            }) => {
                assert_eq!(access_token, "claude-token");
                assert!(account_id.is_none());
            }
            _ => panic!("expected OAuthToken with account_id=None"),
        }
    }

    /// Two providers can persist independently with different `account_id`
    /// values — verifies the provider PK keeps openai-codex and a hypothetical
    /// future OAuth provider isolated.
    #[tokio::test]
    async fn test_oauth_token_two_providers_independent() {
        let manager = SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("memory store");
        let store = manager.token_store();

        let codex_credential = AuthCredential::OAuthToken {
            access_token: "codex-access".to_string(),
            refresh_token: Some("codex-refresh".to_string()),
            expires_at: Some(2_000_000_000_000),
            account_id: Some("workspace-1".to_string()),
        };
        let claude_credential = AuthCredential::OAuthToken {
            access_token: "claude-access".to_string(),
            refresh_token: Some("claude-refresh".to_string()),
            expires_at: Some(3_000_000_000_000),
            account_id: None,
        };

        store
            .save_oauth_token("openai-codex", &codex_credential)
            .await
            .expect("save codex");
        store
            .save_oauth_token("claude", &claude_credential)
            .await
            .expect("save claude");

        let codex_loaded = store
            .load_oauth_token("openai-codex")
            .await
            .expect("load codex")
            .expect("present");
        let claude_loaded = store
            .load_oauth_token("claude")
            .await
            .expect("load claude")
            .expect("present");

        if let AuthCredential::OAuthToken { account_id, .. } = codex_loaded {
            assert_eq!(account_id.as_deref(), Some("workspace-1"));
        } else {
            panic!("expected OAuthToken");
        }
        if let AuthCredential::OAuthToken { account_id, .. } = claude_loaded {
            assert!(account_id.is_none());
        } else {
            panic!("expected OAuthToken");
        }
    }
}
