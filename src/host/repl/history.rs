//! reedline's [`History`] over the store's input-history table, so Up-arrow and Ctrl+R recall
//! what the user typed in any previous run. The SQL is [`HistoryStore`]; this translates
//! reedline's queries and items to it. It reuses the `rusqlite` meka already links rather than
//! reedline's `sqlite` feature, which would pull a second `libsqlite3-sys` and fail to link against
//! `bundled`.

use std::path::Path;

use reedline::{
    CommandLineSearch, History, HistoryItem, HistoryItemId, HistorySessionId, ReedlineError,
    ReedlineErrorVariants, SearchDirection, SearchQuery,
};

use crate::store::history::{CommandLineMatch, HistoryFilter, HistoryOrder, HistoryStore};

pub(crate) struct PromptHistory {
    store: HistoryStore,
}

impl PromptHistory {
    pub(crate) fn open(db_path: &Path, capacity: usize) -> crate::error::Result<Self> {
        Ok(Self {
            store: HistoryStore::open(db_path, capacity)?,
        })
    }
}

/// A reedline query as the store's filter. `start_id`/`end_id` are exclusive and
/// direction-dependent, matching reedline's `FileBackedHistory` semantics.
///
/// `SearchFilter::not_command_line` (used to skip the currently-shown entry during prefix
/// navigation) is `pub(crate)` in reedline and unreadable here. Stepping by the exclusive bounds
/// already advances to a strictly older row each press, and adjacent duplicates are never stored,
/// so omitting it has no practical effect.
fn filter_of(query: &SearchQuery) -> HistoryFilter {
    let (id_below, id_above) = match query.direction {
        SearchDirection::Backward => (query.start_id, query.end_id),
        SearchDirection::Forward => (query.end_id, query.start_id),
    };
    HistoryFilter {
        id_below: id_below.map(|id| id.0),
        id_above: id_above.map(|id| id.0),
        command_line: query
            .filter
            .command_line
            .as_ref()
            .map(|search| match search {
                CommandLineSearch::Prefix(text) => CommandLineMatch::Prefix(text.clone()),
                CommandLineSearch::Substring(text) => CommandLineMatch::Substring(text.clone()),
                CommandLineSearch::Exact(text) => CommandLineMatch::Exact(text.clone()),
            }),
    }
}

impl History for PromptHistory {
    fn save(&mut self, h: HistoryItem) -> reedline::Result<HistoryItem> {
        let command_line = h.command_line;
        // Skip blank input and consecutive duplicates (matches FileBackedHistory).
        if self.store.capacity() == 0 || command_line.trim().is_empty() {
            return Ok(construct_entry(None, command_line));
        }
        if self.store.latest().map_err(to_reedline_error)?.as_deref() == Some(command_line.as_str())
        {
            return Ok(construct_entry(None, command_line));
        }
        let id = self
            .store
            .append(&command_line)
            .map_err(to_reedline_error)?;
        Ok(construct_entry(Some(HistoryItemId::new(id)), command_line))
    }

    fn load(&self, id: HistoryItemId) -> reedline::Result<HistoryItem> {
        match self.store.get(id.0).map_err(to_reedline_error)? {
            Some(command_line) => Ok(construct_entry(Some(id), command_line)),
            None => Err(ReedlineError(ReedlineErrorVariants::OtherHistoryError(
                "history item does not exist",
            ))),
        }
    }

    fn count(&self, query: SearchQuery) -> reedline::Result<i64> {
        self.store
            .count(&filter_of(&query))
            .map_err(to_reedline_error)
    }

    fn search(&self, query: SearchQuery) -> reedline::Result<Vec<HistoryItem>> {
        // Backward yields newest-first; Forward oldest-first (matches FileBackedHistory).
        let order = match query.direction {
            SearchDirection::Backward => HistoryOrder::NewestFirst,
            SearchDirection::Forward => HistoryOrder::OldestFirst,
        };
        let rows = self
            .store
            .search(&filter_of(&query), order, query.limit)
            .map_err(to_reedline_error)?;
        Ok(rows
            .into_iter()
            .map(|(id, command_line)| construct_entry(Some(HistoryItemId::new(id)), command_line))
            .collect())
    }

    fn update(
        &mut self,
        id: HistoryItemId,
        updater: &dyn Fn(HistoryItem) -> HistoryItem,
    ) -> reedline::Result<()> {
        let updated = updater(self.load(id)?);
        self.store
            .update(id.0, &updated.command_line)
            .map_err(to_reedline_error)
    }

    fn clear(&mut self) -> reedline::Result<()> {
        self.store
            .clear_all()
            .map(|_| ())
            .map_err(to_reedline_error)
    }

    fn delete(&mut self, id: HistoryItemId) -> reedline::Result<()> {
        self.store.delete(id.0).map_err(to_reedline_error)
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
fn to_reedline_error(error: crate::error::MekaError) -> ReedlineError {
    tracing::warn!("prompt history database error: {error}");
    ReedlineError::from(std::io::Error::other(error))
}

#[cfg(test)]
mod tests {
    use reedline::SearchQuery;
    use rusqlite::Connection;

    use super::*;

    /// A file-backed store the ledger has been applied to, since the table is the ledger's now.
    fn history_with_capacity(capacity: usize) -> PromptHistory {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.keep().join("history.db");
        let mut connection = Connection::open(&path).expect("open");
        let plan = crate::store::migrations::plan(&connection).expect("plan");
        crate::store::migrations::apply(&mut connection, plan, &Default::default()).expect("apply");
        PromptHistory::open(&path, capacity).expect("open history")
    }

    fn history() -> PromptHistory {
        history_with_capacity(100)
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

    fn count_all(history: &PromptHistory) -> i64 {
        history
            .store
            .count(&HistoryFilter::default())
            .expect("count")
    }

    #[test]
    fn save_and_recall_most_recent() {
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
    fn blank_and_adjacent_duplicates_skipped() {
        let mut history = history();
        assert!(save(&mut history, "   ").is_none(), "blank not stored");
        assert!(save(&mut history, "cmd").is_some());
        assert!(
            save(&mut history, "cmd").is_none(),
            "adjacent duplicate not stored"
        );
        assert_eq!(count_all(&history), 1);
    }

    #[test]
    fn multiline_roundtrip() {
        let mut history = history();
        save(&mut history, "line one\nline two");
        assert_eq!(last(&history).as_deref(), Some("line one\nline two"));
    }

    #[test]
    fn recent_returns_oldest_first_and_honors_limit() {
        let mut history = history();
        save(&mut history, "a");
        save(&mut history, "b");
        save(&mut history, "c");
        assert_eq!(history.store.recent(0).expect("recent all"), [
            "a", "b", "c"
        ]);
        assert_eq!(history.store.recent(2).expect("recent 2"), ["b", "c"]);
    }

    #[test]
    fn clear_all_returns_count_and_empties() {
        let mut history = history();
        save(&mut history, "a");
        save(&mut history, "b");
        assert_eq!(history.store.clear_all().expect("clear"), 2);
        assert_eq!(count_all(&history), 0);
    }

    #[test]
    fn prefix_and_substring_filters() {
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
    fn like_wildcards_are_literal() {
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
    fn backward_id_bound_steps_to_older() {
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
    fn capacity_prunes_oldest() {
        let mut history = history_with_capacity(2);
        save(&mut history, "one");
        save(&mut history, "two");
        save(&mut history, "three");
        assert_eq!(count_all(&history), 2);
        let all = history
            .search(SearchQuery::everything(SearchDirection::Backward, None))
            .expect("search");
        let lines: Vec<&str> = all.iter().map(|i| i.command_line.as_str()).collect();
        assert_eq!(lines, vec!["three", "two"], "oldest entry pruned");
    }

    #[test]
    fn load_delete_and_clear() {
        let mut history = history();
        let id = save(&mut history, "keep").expect("id");
        assert_eq!(history.load(id).expect("load").command_line, "keep");
        save(&mut history, "drop");
        history.delete(id).expect("delete");
        assert!(history.load(id).is_err(), "deleted item is gone");
        history.clear().expect("clear");
        assert_eq!(count_all(&history), 0);
    }
}
