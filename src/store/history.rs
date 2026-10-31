//! The screen's input history: the `prompt_history` table, one row per entry, global across
//! sessions and runs so Up-arrow and Ctrl+R recall what the user typed in *any* previous run.
//! Multi-line entries round-trip intact because each entry is a single TEXT column, not a line in a
//! newline-delimited file.
//!
//! This is the SQL alone. What the composer needs of it is a walk and a search in
//! `host::tui::input`, because the store must not carry an editor. It runs on its own synchronous
//! connection: a key press walks the history synchronously, and the store's connection is async.
//! WAL and `busy_timeout` on both sides keep the two safe together. The table is the ledger's
//! (`prompt_history_is_in_the_ledger`), so the store must have been opened once before this.

use std::path::Path;

use rusqlite::{Connection, OptionalExtension, ToSql, params};

use crate::error::{MekaError, Result};

/// Which rows a search covers: an id window, exclusive at both ends, and a text match. Time, cwd,
/// hostname and exit filters do not exist here because this history never records them.
#[derive(Default)]
pub(crate) struct HistoryFilter {
    pub(crate) id_below: Option<i64>,
    pub(crate) id_above: Option<i64>,
    pub(crate) command_line: Option<CommandLineMatch>,
}

/// How a search's text is matched against `command_line`.
pub(crate) enum CommandLineMatch {
    Substring(String),
}

/// Which end of the history a search reads from first.
#[derive(Clone, Copy)]
pub(crate) enum HistoryOrder {
    NewestFirst,
}

impl super::Store {
    /// Delete the screen's input history written before `cutoff`, for the retention sweeps:
    /// nothing a person typed outlives the window the conversation it went into is kept for.
    /// Answers how many entries went.
    pub(crate) async fn delete_prompt_history_before(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
    ) -> Result<usize> {
        let cutoff = cutoff.to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM prompt_history WHERE julianday(created_at) < julianday(?1)",
                    params![cutoff],
                )
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to delete expired input history: {error}"))
            })
    }
}

/// The `prompt_history` table, on its own synchronous connection.
pub(crate) struct HistoryStore {
    connection: Connection,
    /// Maximum number of entries retained; older rows are pruned on append. `0` disables storage.
    capacity: usize,
}

impl HistoryStore {
    /// Open a second connection to the store at `db_path`, retaining at most `capacity` entries.
    pub(crate) fn open(db_path: &Path, capacity: usize) -> Result<Self> {
        let connection = Connection::open(db_path).map_err(database)?;
        connection
            .busy_timeout(std::time::Duration::from_millis(5000))
            .map_err(database)?;
        Ok(Self {
            connection,
            capacity,
        })
    }

    /// Most-recent entries, returned oldest-first for display. `limit == 0` returns all of them.
    pub(crate) fn recent(&self, limit: usize) -> Result<Vec<String>> {
        // SQLite treats a negative LIMIT as unbounded, so `0` maps to `-1` and needs no second
        // query or string building.
        let bound: i64 = if limit == 0 { -1 } else { limit as i64 };
        let mut statement = self
            .connection
            .prepare("SELECT command_line FROM prompt_history ORDER BY id DESC LIMIT ?1")
            .map_err(database)?;
        let mut rows: Vec<String> = statement
            .query_map(params![bound], |row| row.get(0))
            .map_err(database)?
            .collect::<rusqlite::Result<Vec<String>>>()
            .map_err(database)?;
        rows.reverse();
        Ok(rows)
    }

    /// Delete every input-history row, returning the number removed.
    pub(crate) fn clear_all(&self) -> Result<usize> {
        self.connection
            .execute("DELETE FROM prompt_history", [])
            .map_err(database)
    }

    /// The newest entry, which is what a new one is compared against before it is stored.
    pub(crate) fn latest(&self) -> Result<Option<String>> {
        self.connection
            .query_row(
                "SELECT command_line FROM prompt_history ORDER BY id DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(database)
    }

    /// Store one entry and prune the table back to its capacity. Returns the new row's id.
    pub(crate) fn append(&self, command_line: &str) -> Result<i64> {
        let created_at = chrono::Utc::now().to_rfc3339();
        self.connection
            .execute(
                "INSERT INTO prompt_history (command_line, created_at) VALUES (?1, ?2)",
                params![command_line, created_at],
            )
            .map_err(database)?;
        let id = self.connection.last_insert_rowid();
        self.connection
            .execute(
                "DELETE FROM prompt_history WHERE id NOT IN \
                 (SELECT id FROM prompt_history ORDER BY id DESC LIMIT ?1)",
                params![self.capacity as i64],
            )
            .map_err(database)?;
        Ok(id)
    }

    /// Matching rows as `(id, command_line)`, in `order`, at most `limit` of them.
    pub(crate) fn search(
        &self,
        filter: &HistoryFilter,
        order: HistoryOrder,
        limit: Option<i64>,
    ) -> Result<Vec<(i64, String)>> {
        let (conditions, mut bindings) = where_clause(filter);
        let mut sql = String::from("SELECT id, command_line FROM prompt_history");
        if !conditions.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conditions.join(" AND "));
        }
        sql.push_str(match order {
            HistoryOrder::NewestFirst => " ORDER BY id DESC",
        });
        if let Some(limit) = limit {
            sql.push_str(" LIMIT ?");
            bindings.push(Box::new(limit));
        }
        let refs: Vec<&dyn ToSql> = bindings.iter().map(|binding| binding.as_ref()).collect();
        let mut statement = self.connection.prepare(&sql).map_err(database)?;
        let rows = statement
            .query_map(refs.as_slice(), |row| Ok((row.get(0)?, row.get(1)?)))
            .map_err(database)?;
        rows.collect::<rusqlite::Result<Vec<(i64, String)>>>()
            .map_err(database)
    }
}

/// A filter as a SQL `WHERE` clause and its bound parameters.
fn where_clause(filter: &HistoryFilter) -> (Vec<String>, Vec<Box<dyn ToSql>>) {
    let mut conditions: Vec<String> = Vec::new();
    let mut bindings: Vec<Box<dyn ToSql>> = Vec::new();
    if let Some(below) = filter.id_below {
        conditions.push("id < ?".to_string());
        bindings.push(Box::new(below));
    }
    if let Some(above) = filter.id_above {
        conditions.push("id > ?".to_string());
        bindings.push(Box::new(above));
    }
    match &filter.command_line {
        Some(CommandLineMatch::Substring(text)) => {
            conditions.push("command_line LIKE ? ESCAPE '\\'".to_string());
            bindings.push(Box::new(format!("%{}%", like_escape(text))));
        }
        None => {}
    }
    (conditions, bindings)
}

fn database(error: rusqlite::Error) -> MekaError {
    MekaError::Database(format!("prompt history: {error}"))
}

/// Escape `LIKE` wildcards so a prompt containing `%` / `_` / `\` is matched literally (paired with
/// `ESCAPE '\'` in the query).
fn like_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}
