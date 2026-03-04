use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio_rusqlite::Connection;
use uuid::Uuid;

use crate::error::{AgshError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMessage {
    pub role: String,
    pub content: String,
    pub created_at: String,
}

#[derive(Clone)]
pub struct SessionManager {
    connection: Arc<Connection>,
}

fn default_database_path() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("~/.local/share"))
        .join("agsh")
        .join("sessions.db")
}

impl SessionManager {
    pub async fn open(path: Option<&Path>) -> Result<Self> {
        let database_path = match path {
            Some(path) => path.to_path_buf(),
            None => default_database_path(),
        };

        if let Some(parent) = database_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let connection = Connection::open(&database_path)
            .await
            .map_err(|error| AgshError::Config(format!("failed to open database: {}", error)))?;

        let manager = Self {
            connection: Arc::new(connection),
        };
        manager.initialize_schema().await?;
        Ok(manager)
    }

    async fn initialize_schema(&self) -> Result<()> {
        self.connection
            .call(|connection| {
                connection.execute_batch(
                    "CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        locked_by TEXT,
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
                        ON messages(session_id);",
                )?;
                Ok(())
            })
            .await
            .map_err(|error| AgshError::Config(format!("failed to initialize schema: {}", error)))
    }

    pub async fn create_session(&self) -> Result<Uuid> {
        let session_id = Uuid::new_v4();
        let now = chrono::Utc::now().to_rfc3339();
        let pid = std::process::id().to_string();

        self.connection
            .call(move |connection| {
                connection.execute(
                    "INSERT INTO sessions (id, created_at, updated_at, locked_by) VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![session_id.to_string(), now, now, pid],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| AgshError::Config(format!("failed to create session: {}", error)))?;

        Ok(session_id)
    }

    pub async fn lock_session(&self, session_id: Uuid) -> Result<()> {
        let pid = std::process::id().to_string();

        self.connection
            .call(move |connection| {
                let existing_lock: Option<String> = connection
                    .query_row(
                        "SELECT locked_by FROM sessions WHERE id = ?1",
                        rusqlite::params![session_id.to_string()],
                        |row| row.get(0),
                    )
                    .map_err(|_| {
                        tokio_rusqlite::Error::Other(Box::new(AgshError::SessionNotFound(
                            session_id,
                        )))
                    })?;

                if let Some(locked_pid) = existing_lock {
                    if locked_pid != pid && is_process_alive(&locked_pid) {
                        return Err(tokio_rusqlite::Error::Other(Box::new(
                            AgshError::SessionLocked(session_id),
                        )));
                    }
                }

                connection.execute(
                    "UPDATE sessions SET locked_by = ?1, updated_at = ?2 WHERE id = ?3",
                    rusqlite::params![pid, chrono::Utc::now().to_rfc3339(), session_id.to_string()],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| match error {
                tokio_rusqlite::Error::Other(inner) => {
                    if let Some(agsh_error) = inner.downcast_ref::<AgshError>() {
                        match agsh_error {
                            AgshError::SessionNotFound(id) => AgshError::SessionNotFound(*id),
                            AgshError::SessionLocked(id) => AgshError::SessionLocked(*id),
                            _ => AgshError::Config(format!("failed to lock session: {}", inner)),
                        }
                    } else {
                        AgshError::Config(format!("failed to lock session: {}", inner))
                    }
                }
                other => AgshError::Config(format!("failed to lock session: {}", other)),
            })
    }

    pub async fn unlock_session(&self, session_id: Uuid) -> Result<()> {
        self.connection
            .call(move |connection| {
                connection.execute(
                    "UPDATE sessions SET locked_by = NULL, updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![chrono::Utc::now().to_rfc3339(), session_id.to_string()],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| AgshError::Config(format!("failed to unlock session: {}", error)))
    }

    pub async fn save_message(&self, session_id: Uuid, role: &str, content: &str) -> Result<()> {
        let role = role.to_string();
        let content = content.to_string();
        let now = chrono::Utc::now().to_rfc3339();

        self.connection
            .call(move |connection| {
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
            .map_err(|error| AgshError::Config(format!("failed to save message: {}", error)))
    }

    pub async fn load_messages(&self, session_id: Uuid) -> Result<Vec<StoredMessage>> {
        self.connection
            .call(move |connection| {
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
            .map_err(|error| AgshError::Config(format!("failed to load messages: {}", error)))
    }

    pub async fn last_session_id(&self) -> Result<Option<Uuid>> {
        self.connection
            .call(|connection| {
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
                    Err(error) => Err(error.into()),
                }
            })
            .await
            .map_err(|error| AgshError::Config(format!("failed to get last session: {}", error)))
    }

    pub async fn session_exists(&self, session_id: Uuid) -> Result<bool> {
        self.connection
            .call(move |connection| {
                let count: i64 = connection.query_row(
                    "SELECT COUNT(*) FROM sessions WHERE id = ?1",
                    rusqlite::params![session_id.to_string()],
                    |row| row.get(0),
                )?;
                Ok(count > 0)
            })
            .await
            .map_err(|error| {
                AgshError::Config(format!("failed to check session existence: {}", error))
            })
    }
}

fn is_process_alive(pid_str: &str) -> bool {
    let Ok(pid) = pid_str.parse::<u32>() else {
        return false;
    };

    #[cfg(unix)]
    {
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }

    #[cfg(windows)]
    {
        std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {}", pid), "/NH"])
            .output()
            .map(|output| {
                let stdout = String::from_utf8_lossy(&output.stdout);
                stdout.contains(&pid.to_string())
            })
            .unwrap_or(false)
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
    async fn test_session_locking() {
        let manager = test_manager().await;
        let session_id = manager
            .create_session()
            .await
            .expect("failed to create session");

        manager
            .lock_session(session_id)
            .await
            .expect("failed to lock session");

        // Same PID should be able to re-lock
        manager
            .lock_session(session_id)
            .await
            .expect("failed to re-lock session");

        manager
            .unlock_session(session_id)
            .await
            .expect("failed to unlock session");
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
}
