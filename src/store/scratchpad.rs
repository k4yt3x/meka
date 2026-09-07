//! Scratchpad rows: the `tool_outputs` table.

use super::*;

/// One scratchpad entry as a listing shows it: its name, size and creation time, not its content.
#[derive(Debug, Clone)]
pub(crate) struct ScratchpadEntry {
    pub(crate) name: String,
    pub(crate) size: usize,
    pub(crate) created_at: String,
}
/// Result of [`Store::rename_scratchpad_entry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RenameOutcome {
    Renamed,
    NotFound,
    TargetExists,
}

impl Store {
    /// Write a scratchpad entry, replacing one of the same name.
    pub(crate) async fn save_scratchpad_entry(
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
            .map_err(|error| {
                MekaError::Database(format!("failed to save the scratchpad entry: {error}"))
            })
    }

    /// Replace an existing entry's content; `false` when there is no entry of that name.
    pub(crate) async fn update_scratchpad_entry(
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
                MekaError::Database(format!("failed to update the scratchpad entry: {error}"))
            })
    }

    /// Remove an entry; `false` when there was none of that name.
    pub(crate) async fn delete_scratchpad_entry(
        &self,
        session_id: Uuid,
        name: &str,
    ) -> Result<bool> {
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
                MekaError::Database(format!("failed to delete the scratchpad entry: {error}"))
            })
    }

    /// Rename an entry, refusing to overwrite one that already has the new name.
    pub(crate) async fn rename_scratchpad_entry(
        &self,
        session_id: Uuid,
        old: &str,
        new: &str,
    ) -> Result<RenameOutcome> {
        let old = old.to_string();
        let new = new.to_string();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                // Pre-check: target must not exist. `tokio_rusqlite` serializes connection access
                // so this and the UPDATE share a consistent view; the `PRIMARY KEY (session_id,
                // name)` constraint at the schema layer is the final backstop.
                let target_exists: i64 = connection.query_row(
                    "SELECT COUNT(*) FROM tool_outputs WHERE session_id = ?1 AND name = ?2",
                    rusqlite::params![session_id.to_string(), new],
                    |row| row.get(0),
                )?;
                if target_exists > 0 {
                    return Ok(RenameOutcome::TargetExists);
                }
                let renamed = connection.execute(
                    "UPDATE tool_outputs SET name = ?1 WHERE session_id = ?2 AND name = ?3",
                    rusqlite::params![new, session_id.to_string(), old],
                )?;
                Ok(if renamed > 0 {
                    RenameOutcome::Renamed
                } else {
                    RenameOutcome::NotFound
                })
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to rename the scratchpad entry: {error}"))
            })
    }

    /// Every entry of a session, oldest first, without content.
    pub(crate) async fn list_scratchpad_entries(
        &self,
        session_id: Uuid,
    ) -> Result<Vec<ScratchpadEntry>> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "SELECT name, LENGTH(content), created_at \
                     FROM tool_outputs WHERE session_id = ?1 ORDER BY created_at ASC",
                )?;

                let rows = statement
                    .query_map(rusqlite::params![session_id.to_string()], |row| {
                        Ok(ScratchpadEntry {
                            name: row.get(0)?,
                            size: row.get::<_, i64>(1)? as usize,
                            created_at: row.get(2)?,
                        })
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;

                Ok(rows)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to list the scratchpad entries: {error}"))
            })
    }

    /// One entry's content, or `None` when there is no entry of that name.
    pub(crate) async fn load_scratchpad_entry(
        &self,
        session_id: Uuid,
        name: &str,
    ) -> Result<Option<String>> {
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
            .map_err(|error| {
                MekaError::Database(format!("failed to load the scratchpad entry: {error}"))
            })
    }

    /// Every entry of a session with its content, oldest first, as `(name, content)`.
    pub(crate) async fn load_all_scratchpad_entries(
        &self,
        session_id: Uuid,
    ) -> Result<Vec<(String, String)>> {
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
            .map_err(|error| {
                MekaError::Database(format!("failed to load the scratchpad entries: {error}"))
            })
    }
}
