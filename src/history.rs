//! Persistent REPL input history backed by SQLite.
//!
//! [`PromptHistory`] implements reedline's [`History`] trait against a `prompt_history` table in
//! meka's existing SQLite database, so Up-arrow / Ctrl+R recall what the user typed in *any*
//! previous run (shell-style), not just the current process. It reuses the `rusqlite` dependency
//! meka already links (rather than enabling reedline's `sqlite` feature, which would pull a second
//! `libsqlite3-sys` and fail to link against `bundled`). History is global / cross-session, and
//! multi-line entries round-trip intact because each entry is a single TEXT column, not a line in a
//! newline-delimited file.

use std::path::Path;

use reedline::{
    CommandLineSearch, History, HistoryItem, HistoryItemId, HistorySessionId, ReedlineError,
    ReedlineErrorVariants, SearchDirection, SearchQuery,
};
use rusqlite::{Connection, OptionalExtension, ToSql, params};

/// SQLite-backed reedline history. Owns a private synchronous connection to the database file (the
/// agent's async connection is separate); WAL + `busy_timeout` on both sides keep concurrent access
/// safe.
pub struct PromptHistory {
    connection: Connection,
    /// Maximum number of entries retained; older rows are pruned on save. `0` disables storage.
    capacity: usize,
}

impl PromptHistory {
    /// Open (or create) the `prompt_history` table in the database at `db_path`.
    pub fn open(db_path: &Path, capacity: usize) -> rusqlite::Result<Self> {
        let connection = Connection::open(db_path)?;
        connection.busy_timeout(std::time::Duration::from_millis(5000))?;
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS prompt_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command_line TEXT NOT NULL,
                created_at TEXT NOT NULL
            );",
        )?;
        Ok(Self {
            connection,
            capacity,
        })
    }

    /// Most-recent entries, returned oldest-first for display. `limit == 0` returns all of them.
    pub fn recent(&self, limit: usize) -> rusqlite::Result<Vec<String>> {
        // SQLite treats a negative LIMIT as unbounded, so `0` maps to `-1` and needs no second
        // query or string building.
        let bound: i64 = if limit == 0 { -1 } else { limit as i64 };
        let mut statement = self
            .connection
            .prepare("SELECT command_line FROM prompt_history ORDER BY id DESC LIMIT ?1")?;
        let mut rows: Vec<String> = statement
            .query_map(params![bound], |row| row.get(0))?
            .collect::<rusqlite::Result<Vec<String>>>()?;
        rows.reverse();
        Ok(rows)
    }

    /// Delete every input-history row, returning the number removed.
    pub fn clear_all(&self) -> rusqlite::Result<usize> {
        self.connection.execute("DELETE FROM prompt_history", [])
    }

    /// Translate a [`SearchQuery`]'s id bounds and command-line filter into a SQL `WHERE` clause
    /// and its bound parameters. `start_id`/`end_id` are exclusive and direction-dependent,
    /// matching reedline's `FileBackedHistory` semantics. Time / cwd / hostname / exit filters
    /// are ignored (never populated for this history).
    fn build_filter(query: &SearchQuery) -> (Vec<String>, Vec<Box<dyn ToSql>>) {
        let mut conditions: Vec<String> = Vec::new();
        let mut bindings: Vec<Box<dyn ToSql>> = Vec::new();

        match query.direction {
            SearchDirection::Backward => {
                if let Some(start) = query.start_id {
                    conditions.push("id < ?".to_string());
                    bindings.push(Box::new(start.0));
                }
                if let Some(end) = query.end_id {
                    conditions.push("id > ?".to_string());
                    bindings.push(Box::new(end.0));
                }
            }
            SearchDirection::Forward => {
                if let Some(start) = query.start_id {
                    conditions.push("id > ?".to_string());
                    bindings.push(Box::new(start.0));
                }
                if let Some(end) = query.end_id {
                    conditions.push("id < ?".to_string());
                    bindings.push(Box::new(end.0));
                }
            }
        }

        match &query.filter.command_line {
            Some(CommandLineSearch::Prefix(text)) => {
                conditions.push("command_line LIKE ? ESCAPE '\\'".to_string());
                bindings.push(Box::new(format!("{}%", like_escape(text))));
            }
            Some(CommandLineSearch::Substring(text)) => {
                conditions.push("command_line LIKE ? ESCAPE '\\'".to_string());
                bindings.push(Box::new(format!("%{}%", like_escape(text))));
            }
            Some(CommandLineSearch::Exact(text)) => {
                conditions.push("command_line = ?".to_string());
                bindings.push(Box::new(text.clone()));
            }
            None => {}
        }

        // `SearchFilter::not_command_line` (used to skip the currently-shown entry during prefix
        // navigation) is `pub(crate)` in reedline and unreadable here. Stepping by the exclusive
        // `start_id`/`end_id` bounds above already advances to a strictly older row each press, and
        // adjacent duplicates are never stored, so omitting it has no practical effect.

        (conditions, bindings)
    }
}

impl History for PromptHistory {
    fn save(&mut self, h: HistoryItem) -> reedline::Result<HistoryItem> {
        let command_line = h.command_line;
        // Skip blank input and consecutive duplicates (matches FileBackedHistory).
        if self.capacity == 0 || command_line.trim().is_empty() {
            return Ok(construct_entry(None, command_line));
        }
        let previous: Option<String> = self
            .connection
            .query_row(
                "SELECT command_line FROM prompt_history ORDER BY id DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(to_reedline_error)?;
        if previous.as_deref() == Some(command_line.as_str()) {
            return Ok(construct_entry(None, command_line));
        }

        let created_at = chrono::Utc::now().to_rfc3339();
        self.connection
            .execute(
                "INSERT INTO prompt_history (command_line, created_at) VALUES (?1, ?2)",
                params![command_line, created_at],
            )
            .map_err(to_reedline_error)?;
        let id = self.connection.last_insert_rowid();

        // Keep only the most recent `capacity` entries.
        self.connection
            .execute(
                "DELETE FROM prompt_history WHERE id NOT IN \
                 (SELECT id FROM prompt_history ORDER BY id DESC LIMIT ?1)",
                params![self.capacity as i64],
            )
            .map_err(to_reedline_error)?;

        Ok(construct_entry(Some(HistoryItemId::new(id)), command_line))
    }

    fn load(&self, id: HistoryItemId) -> reedline::Result<HistoryItem> {
        let command_line: Option<String> = self
            .connection
            .query_row(
                "SELECT command_line FROM prompt_history WHERE id = ?1",
                params![id.0],
                |row| row.get(0),
            )
            .optional()
            .map_err(to_reedline_error)?;
        match command_line {
            Some(command_line) => Ok(construct_entry(Some(id), command_line)),
            None => Err(ReedlineError(ReedlineErrorVariants::OtherHistoryError(
                "history item does not exist",
            ))),
        }
    }

    fn count(&self, query: SearchQuery) -> reedline::Result<i64> {
        let (conditions, bindings) = Self::build_filter(&query);
        let mut sql = String::from("SELECT COUNT(*) FROM prompt_history");
        if !conditions.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conditions.join(" AND "));
        }
        let refs: Vec<&dyn ToSql> = bindings.iter().map(|b| b.as_ref()).collect();
        self.connection
            .query_row(&sql, refs.as_slice(), |row| row.get(0))
            .map_err(to_reedline_error)
    }

    fn search(&self, query: SearchQuery) -> reedline::Result<Vec<HistoryItem>> {
        let limit = query.limit;
        let direction = query.direction;
        let (conditions, mut bindings) = Self::build_filter(&query);

        let mut sql = String::from("SELECT id, command_line FROM prompt_history");
        if !conditions.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conditions.join(" AND "));
        }
        // Backward yields newest-first; Forward oldest-first (matches FileBackedHistory).
        sql.push_str(match direction {
            SearchDirection::Backward => " ORDER BY id DESC",
            SearchDirection::Forward => " ORDER BY id ASC",
        });
        if let Some(limit) = limit {
            sql.push_str(" LIMIT ?");
            bindings.push(Box::new(limit));
        }

        let refs: Vec<&dyn ToSql> = bindings.iter().map(|b| b.as_ref()).collect();
        let mut statement = self.connection.prepare(&sql).map_err(to_reedline_error)?;
        let rows = statement
            .query_map(refs.as_slice(), |row| {
                let id: i64 = row.get(0)?;
                let command_line: String = row.get(1)?;
                Ok(construct_entry(Some(HistoryItemId::new(id)), command_line))
            })
            .map_err(to_reedline_error)?;
        rows.collect::<rusqlite::Result<Vec<HistoryItem>>>()
            .map_err(to_reedline_error)
    }

    fn update(
        &mut self,
        id: HistoryItemId,
        updater: &dyn Fn(HistoryItem) -> HistoryItem,
    ) -> reedline::Result<()> {
        let updated = updater(self.load(id)?);
        self.connection
            .execute(
                "UPDATE prompt_history SET command_line = ?1 WHERE id = ?2",
                params![updated.command_line, id.0],
            )
            .map_err(to_reedline_error)?;
        Ok(())
    }

    fn clear(&mut self) -> reedline::Result<()> {
        self.connection
            .execute("DELETE FROM prompt_history", [])
            .map_err(to_reedline_error)?;
        Ok(())
    }

    fn delete(&mut self, id: HistoryItemId) -> reedline::Result<()> {
        self.connection
            .execute("DELETE FROM prompt_history WHERE id = ?1", params![id.0])
            .map_err(to_reedline_error)?;
        Ok(())
    }

    fn sync(&mut self) -> std::io::Result<()> {
        // Each write is autocommitted, so there's nothing buffered to flush.
        Ok(())
    }

    fn session(&self) -> Option<HistorySessionId> {
        // No session scoping: recall is global across all runs.
        None
    }
}

/// Build a [`HistoryItem`] carrying only the id and command line; this history stores nothing else.
fn construct_entry(id: Option<HistoryItemId>, command_line: String) -> HistoryItem {
    HistoryItem {
        id,
        start_timestamp: None,
        command_line,
        session_id: None,
        hostname: None,
        cwd: None,
        duration: None,
        exit_status: None,
        more_info: None,
    }
}

/// `ReedlineErrorVariants::HistoryDatabaseError` is gated behind reedline's `sqlite` feature (not
/// enabled here), so wrap database failures through `std::io::Error` instead, and log the real
/// error since the wrapped message isn't surfaced to the user.
fn to_reedline_error(error: rusqlite::Error) -> ReedlineError {
    tracing::warn!("prompt history database error: {}", error);
    ReedlineError::from(std::io::Error::other(error))
}

/// Escape `LIKE` wildcards so a prompt containing `%` / `_` / `\` is matched literally (paired with
/// `ESCAPE '\'` in the query).
fn like_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

#[cfg(test)]
mod tests {
    use reedline::SearchQuery;

    use super::*;

    fn history() -> PromptHistory {
        PromptHistory::open(Path::new(":memory:"), 100).expect("open in-memory history")
    }

    fn save(history: &mut PromptHistory, command_line: &str) -> Option<HistoryItemId> {
        history
            .save(construct_entry(None, command_line.to_string()))
            .expect("save")
            .id
    }

    fn last(history: &PromptHistory) -> Option<String> {
        history
            .search(SearchQuery::everything(SearchDirection::Backward, None))
            .expect("search")
            .first()
            .map(|item| item.command_line.clone())
    }

    #[test]
    fn test_save_and_recall_most_recent() {
        let mut history = history();
        save(&mut history, "first");
        save(&mut history, "second");
        // Backward + limit 1 is what up-arrow issues for the most recent entry.
        let query = SearchQuery {
            limit: Some(1),
            ..SearchQuery::everything(SearchDirection::Backward, None)
        };
        let results = history.search(query).expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].command_line, "second");
    }

    #[test]
    fn test_blank_and_adjacent_duplicates_skipped() {
        let mut history = history();
        assert!(save(&mut history, "   ").is_none(), "blank not stored");
        assert!(save(&mut history, "cmd").is_some());
        assert!(
            save(&mut history, "cmd").is_none(),
            "adjacent duplicate not stored"
        );
        assert_eq!(history.count_all().expect("count"), 1);
    }

    #[test]
    fn test_multiline_roundtrip() {
        let mut history = history();
        save(&mut history, "line one\nline two");
        assert_eq!(last(&history).as_deref(), Some("line one\nline two"));
    }

    #[test]
    fn test_recent_returns_oldest_first_and_honors_limit() {
        let mut history = history();
        save(&mut history, "a");
        save(&mut history, "b");
        save(&mut history, "c");
        assert_eq!(history.recent(0).expect("recent all"), ["a", "b", "c"]);
        assert_eq!(history.recent(2).expect("recent 2"), ["b", "c"]);
    }

    #[test]
    fn test_clear_all_returns_count_and_empties() {
        let mut history = history();
        save(&mut history, "a");
        save(&mut history, "b");
        assert_eq!(history.clear_all().expect("clear"), 2);
        assert_eq!(history.count_all().expect("count"), 0);
    }

    #[test]
    fn test_prefix_and_substring_filters() {
        let mut history = history();
        save(&mut history, "git status");
        save(&mut history, "cargo build");
        save(&mut history, "git commit");

        let prefix = history
            .search(SearchQuery::last_with_prefix("git".to_string(), None))
            .expect("prefix search");
        assert_eq!(prefix.len(), 1);
        assert_eq!(prefix[0].command_line, "git commit");

        let substring = history
            .search(SearchQuery::all_that_contain_rev("car".to_string()))
            .expect("substring search");
        assert_eq!(substring.len(), 1);
        assert_eq!(substring[0].command_line, "cargo build");
    }

    #[test]
    fn test_like_wildcards_are_literal() {
        let mut history = history();
        save(&mut history, "100% sure");
        save(&mut history, "100x sure");
        let results = history
            .search(SearchQuery::all_that_contain_rev("100%".to_string()))
            .expect("search");
        assert_eq!(results.len(), 1, "% must not act as a wildcard");
        assert_eq!(results[0].command_line, "100% sure");
    }

    #[test]
    fn test_backward_id_bound_steps_to_older() {
        let mut history = history();
        let first = save(&mut history, "first").expect("id");
        save(&mut history, "second");
        // Up-arrow again from `second`: start_id = second's cursor → expect the older `first`.
        let newest = history
            .search(SearchQuery {
                limit: Some(1),
                ..SearchQuery::everything(SearchDirection::Backward, None)
            })
            .expect("search")[0]
            .id
            .expect("id");
        let query = SearchQuery {
            start_id: Some(newest),
            limit: Some(1),
            ..SearchQuery::everything(SearchDirection::Backward, None)
        };
        let older = history.search(query).expect("search");
        assert_eq!(older.len(), 1);
        assert_eq!(older[0].command_line, "first");
        assert_eq!(older[0].id, Some(first));
    }

    #[test]
    fn test_capacity_prunes_oldest() {
        let mut history = PromptHistory::open(Path::new(":memory:"), 2).expect("open");
        save(&mut history, "one");
        save(&mut history, "two");
        save(&mut history, "three");
        assert_eq!(history.count_all().expect("count"), 2);
        let all = history
            .search(SearchQuery::everything(SearchDirection::Backward, None))
            .expect("search");
        let lines: Vec<&str> = all.iter().map(|i| i.command_line.as_str()).collect();
        assert_eq!(lines, vec!["three", "two"], "oldest entry pruned");
    }

    #[test]
    fn test_load_delete_and_clear() {
        let mut history = history();
        let id = save(&mut history, "keep").expect("id");
        assert_eq!(history.load(id).expect("load").command_line, "keep");
        save(&mut history, "drop");
        history.delete(id).expect("delete");
        assert!(history.load(id).is_err(), "deleted item is gone");
        history.clear().expect("clear");
        assert_eq!(history.count_all().expect("count"), 0);
    }
}
