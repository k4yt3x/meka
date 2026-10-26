//! The `projects` table: every project a session can be on, each a memory tier beside the global
//! one.
//!
//! A project is a row and nothing else, created by `meka project add` or `POST /v1/projects`, so
//! every door handed a name from outside asks [`Store::require_project`] before a row records it:
//! a typo is refused with the names that exist rather than opening an empty tier. A rename moves
//! the row and every session and memory on it in one transaction, and a removal refuses while any
//! still names it.

use super::{Store, sessions::spawned_session_sql};
use crate::error::{MekaError, Result};

/// A project as the store holds it, with what is on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProjectRecord {
    pub(crate) name: String,
    /// RFC 3339, when the project was added.
    pub(crate) created_at: String,
    /// Memories in the project's tier.
    pub(crate) memories: u64,
    /// Root sessions on the project. A sub-agent's row carries the project its parent was on,
    /// and counting it would list a project as busier than the sessions a person can resume.
    pub(crate) sessions: u64,
}

/// What a rename did: the project under its new name, and how many sessions, sub-agents
/// included, and memories followed it there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RenamedProject {
    pub(crate) project: ProjectRecord,
    pub(crate) sessions_moved: u64,
    pub(crate) memories_moved: u64,
}

/// What a door decided inside its transaction.
///
/// A refusal is returned rather than raised so the closure's `rusqlite` errors keep their type;
/// the transaction is dropped uncommitted on that path, which is what makes "refused" mean
/// "nothing moved".
enum Decided<T> {
    Done(T),
    Refused(MekaError),
}

impl<T> Decided<T> {
    fn into_result(self) -> Result<T> {
        match self {
            Decided::Done(value) => Ok(value),
            Decided::Refused(refusal) => Err(refusal),
        }
    }
}

/// The names in `connection`'s `projects` table, sorted.
pub(super) fn names_in(connection: &rusqlite::Connection) -> rusqlite::Result<Vec<String>> {
    let mut statement = connection.prepare("SELECT name FROM projects ORDER BY name ASC")?;
    let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
    rows.collect()
}

/// The projects in `connection` with their counts, sorted by name: every one, or the one `name`
/// names. A writer reads its own row through this inside its transaction, so what it answers is
/// what it committed.
fn records_in(
    connection: &rusqlite::Connection,
    name: Option<&str>,
) -> rusqlite::Result<Vec<ProjectRecord>> {
    let mut statement = connection.prepare(&format!(
        "SELECT p.name, p.created_at,
                (SELECT COUNT(*) FROM memories m WHERE m.project = p.name),
                (SELECT COUNT(*) FROM sessions s
                  WHERE s.project = p.name AND NOT {})
         FROM projects p
         WHERE ?1 IS NULL OR p.name = ?1
         ORDER BY p.name ASC",
        spawned_session_sql("s.")
    ))?;
    let rows = statement.query_map(rusqlite::params![name], |row| {
        Ok(ProjectRecord {
            name: row.get(0)?,
            created_at: row.get(1)?,
            memories: u64::try_from(row.get::<_, i64>(2)?).unwrap_or(0),
            sessions: u64::try_from(row.get::<_, i64>(3)?).unwrap_or(0),
        })
    })?;
    rows.collect()
}

/// The refusal for `name` when no project has it.
fn unknown(name: &str, known: Vec<String>) -> MekaError {
    MekaError::UnknownProject {
        name: name.to_string(),
        known,
    }
}

/// The existing name `name` would collide with, exact or under case folding, if any. Names are
/// compared exactly wherever a row records one, so two that a listing cannot tell apart are
/// refused at the door. `except` is a rename's own name, which it may keep in another case.
fn name_taken_by(
    connection: &rusqlite::Connection,
    name: &str,
    except: Option<&str>,
) -> rusqlite::Result<Option<String>> {
    use rusqlite::OptionalExtension as _;
    connection
        .query_row(
            "SELECT name FROM projects WHERE name = ?1 COLLATE NOCASE AND name != ?2",
            rusqlite::params![name, except.unwrap_or_default()],
            |row| row.get::<_, String>(0),
        )
        .optional()
}

/// The refusal for a name [`name_taken_by`] found.
fn taken(name: &str, by: &str) -> MekaError {
    MekaError::Usage(if by == name {
        format!("a project named '{name}' already exists")
    } else {
        format!("a project named '{by}' already exists, and names differ only by case")
    })
}

/// `count` of `noun`, pluralized the English way `memories` and `sessions` go.
fn counted(count: u64, singular: &str, plural: &str) -> String {
    let noun = if count == 1 { singular } else { plural };
    format!("{count} {noun}")
}

impl Store {
    /// Add a project and answer its row. Refuses a name the entry rule refuses, one that exists,
    /// and one that differs from an existing name only by case.
    pub(crate) async fn add_project(&self, name: &str) -> Result<ProjectRecord> {
        crate::entry::validate_entry_name(name, "project").map_err(MekaError::Usage)?;
        let name = name.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let transaction = connection
                    .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                if let Some(by) = name_taken_by(&transaction, &name, None)? {
                    return Ok(Decided::Refused(taken(&name, &by)));
                }
                let created_at = chrono::Utc::now().to_rfc3339();
                transaction.execute(
                    "INSERT INTO projects (name, created_at) VALUES (?1, ?2)",
                    rusqlite::params![name, created_at],
                )?;
                transaction.commit()?;
                Ok(Decided::Done(ProjectRecord {
                    name,
                    created_at,
                    memories: 0,
                    sessions: 0,
                }))
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to add a project: {error}")))?
            .into_result()
    }

    /// Every project, sorted by name, with how many memories and root sessions are on each.
    pub(crate) async fn list_projects(&self) -> Result<Vec<ProjectRecord>> {
        self.connection
            .call(|connection| records_in(connection, None))
            .await
            .map_err(|error| MekaError::Database(format!("failed to list projects: {error}")))
    }

    /// Every project name, sorted: what a picker offers and a refusal names.
    pub(crate) async fn project_names(&self) -> Result<Vec<String>> {
        self.connection
            .call(|connection| names_in(connection))
            .await
            .map_err(|error| MekaError::Database(format!("failed to list projects: {error}")))
    }

    /// Refuse `name` unless a project has it, naming the ones that do exist. Asked by every door
    /// that takes a project name from outside before any row records it.
    pub(crate) async fn require_project(&self, name: &str) -> Result<()> {
        let known = self.project_names().await?;
        if known.iter().any(|existing| existing == name) {
            return Ok(());
        }
        Err(unknown(name, known))
    }

    /// Rename a project, moving every session and memory on it, and answer the row under its new
    /// name with how many of each moved.
    ///
    /// One transaction over the three tables, so a crash cannot leave the sessions on one name
    /// and their memories on another, which no door could then bring back together.
    pub(crate) async fn rename_project(
        &self,
        name: &str,
        new_name: &str,
    ) -> Result<RenamedProject> {
        crate::entry::validate_entry_name(new_name, "project").map_err(MekaError::Usage)?;
        let name = name.to_string();
        let new_name = new_name.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let transaction = connection
                    .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let known = names_in(&transaction)?;
                if !known.contains(&name) {
                    return Ok(Decided::Refused(unknown(&name, known)));
                }
                if let Some(by) = name_taken_by(&transaction, &new_name, Some(&name))? {
                    return Ok(Decided::Refused(taken(&new_name, &by)));
                }
                transaction.execute(
                    "UPDATE projects SET name = ?2 WHERE name = ?1",
                    rusqlite::params![name, new_name],
                )?;
                let sessions = transaction.execute(
                    "UPDATE sessions SET project = ?2 WHERE project = ?1",
                    rusqlite::params![name, new_name],
                )?;
                let memories = transaction.execute(
                    "UPDATE memories SET project = ?2 WHERE project = ?1",
                    rusqlite::params![name, new_name],
                )?;
                let project = records_in(&transaction, Some(&new_name))?
                    .pop()
                    .ok_or(rusqlite::Error::QueryReturnedNoRows)?;
                transaction.commit()?;
                Ok(Decided::Done(RenamedProject {
                    project,
                    sessions_moved: sessions as u64,
                    memories_moved: memories as u64,
                }))
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to rename a project: {error}")))?
            .into_result()
    }

    /// Remove a project, refusing with the counts while any memory or root session is on it.
    pub(crate) async fn remove_project(&self, name: &str) -> Result<()> {
        let name = name.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let transaction = connection
                    .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let known = names_in(&transaction)?;
                if !known.contains(&name) {
                    return Ok(Decided::Refused(unknown(&name, known)));
                }
                let memories: i64 = transaction.query_row(
                    "SELECT COUNT(*) FROM memories WHERE project = ?1",
                    rusqlite::params![name],
                    |row| row.get(0),
                )?;
                let sessions: i64 = transaction.query_row(
                    &format!(
                        "SELECT COUNT(*) FROM sessions WHERE project = ?1 AND NOT {}",
                        spawned_session_sql("")
                    ),
                    rusqlite::params![name],
                    |row| row.get(0),
                )?;
                if memories + sessions > 0 {
                    return Ok(Decided::Refused(MekaError::Usage(format!(
                        "project '{name}' has {} and {}; move or remove them first",
                        counted(memories.max(0) as u64, "memory", "memories"),
                        counted(sessions.max(0) as u64, "session", "sessions"),
                    ))));
                }
                transaction.execute("DELETE FROM projects WHERE name = ?1", rusqlite::params![
                    name
                ])?;
                transaction.commit()?;
                Ok(Decided::Done(()))
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to remove a project: {error}")))?
            .into_result()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A project is added once: the same name, a case variant and a name the entry rule refuses
    /// are all turned away, and a name no project has is refused with the ones that exist.
    #[tokio::test]
    async fn a_project_is_added_once_and_an_unknown_name_is_refused_with_the_known() {
        let store = Store::for_test().await;
        store.add_project("webdev").await.expect("added");
        let again = store.add_project("webdev").await.expect_err("taken");
        assert!(again.to_string().contains("already exists"), "{again}");
        let cased = store.add_project("Webdev").await.expect_err("case");
        assert!(cased.to_string().contains("differ only by case"), "{cased}");
        let spaced = store.add_project("web dev").await.expect_err("rule");
        assert!(spaced.to_string().contains("project name"), "{spaced}");
        assert_eq!(store.project_names().await.expect("names"), ["webdev"]);
        store.require_project("webdev").await.expect("known");
        let unknown = store.require_project("movie").await.expect_err("unknown");
        assert!(
            matches!(&unknown, MekaError::UnknownProject { name, known } if name == "movie" && known == &["webdev"]),
            "{unknown:?}"
        );
        let listed = store.list_projects().await.expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "webdev");
        assert_eq!((listed[0].memories, listed[0].sessions), (0, 0));
        assert!(
            chrono::DateTime::parse_from_rfc3339(&listed[0].created_at).is_ok(),
            "{}",
            listed[0].created_at
        );
    }

    /// A rename moves the project and every session and memory on it in one step, and the counts
    /// a listing shows follow; a rename onto a taken name, or of a name no project has, moves
    /// nothing. A case-only rename is a rename of the one project.
    #[tokio::test]
    async fn a_rename_moves_every_row_on_the_project_and_the_counts_follow() {
        let store = Store::for_test().await;
        store.add_project("webdev").await.expect("added");
        store.add_project("movie").await.expect("added");
        let root = store
            .create_session_in_project_for_test(None, "profile", "webdev")
            .await
            .expect("a session on webdev");
        let (_child, _lock) = store
            .create_child_session(
                root,
                None,
                Vec::new(),
                None,
                "read".to_string(),
                "profile".to_string(),
                "webdev".to_string(),
            )
            .await
            .expect("a sub-agent of it");
        let memories = store.memory_store(true, crate::config::MEMORY_INLINE_MAX_BYTES_DEFAULT);
        memories
            .write(crate::store::memory::WriteRequest {
                name: "note".to_string(),
                description: Some("d".to_string()),
                view: crate::store::memory::MemoryView::Session("webdev".to_string()),
                ..Default::default()
            })
            .await
            .expect("a webdev memory");
        let counts = |store: Store| async move {
            store
                .list_projects()
                .await
                .expect("list")
                .into_iter()
                .map(|project| (project.name, project.memories, project.sessions))
                .collect::<Vec<_>>()
        };
        assert_eq!(counts(store.clone()).await, [
            ("movie".to_string(), 0, 0),
            ("webdev".to_string(), 1, 1)
        ]);

        let onto_taken = store
            .rename_project("webdev", "Movie")
            .await
            .expect_err("taken under case folding");
        assert!(
            onto_taken.to_string().contains("differ only by case"),
            "{onto_taken}"
        );
        let of_unknown = store
            .rename_project("ghost", "web")
            .await
            .expect_err("unknown");
        assert!(
            matches!(of_unknown, MekaError::UnknownProject { .. }),
            "{of_unknown}"
        );

        let renamed = store.rename_project("webdev", "web").await.expect("rename");
        assert_eq!(
            (renamed.sessions_moved, renamed.memories_moved),
            (2, 1),
            "the root and its sub-agent, plus the one memory"
        );
        assert_eq!(
            (
                renamed.project.name.as_str(),
                renamed.project.memories,
                renamed.project.sessions
            ),
            ("web", 1, 1),
            "the row under its new name, counting the root alone"
        );
        assert_eq!(
            store.recorded_project(root).await.expect("row"),
            Some("web".to_string())
        );
        assert_eq!(counts(store.clone()).await, [
            ("movie".to_string(), 0, 0),
            ("web".to_string(), 1, 1)
        ]);
        store.rename_project("web", "Web").await.expect("case only");
        assert_eq!(store.project_names().await.expect("names"), [
            "Web", "movie"
        ]);
    }

    /// A removal refuses, with the counts, while a memory or a root session is on the project,
    /// and goes through once nothing is.
    #[tokio::test]
    async fn a_removal_refuses_while_rows_name_the_project() {
        let store = Store::for_test().await;
        store.add_project("webdev").await.expect("added");
        let session = store
            .create_session_in_project_for_test(None, "profile", "webdev")
            .await
            .expect("a session on webdev");
        let memories = store.memory_store(true, crate::config::MEMORY_INLINE_MAX_BYTES_DEFAULT);
        memories
            .write(crate::store::memory::WriteRequest {
                name: "note".to_string(),
                description: Some("d".to_string()),
                view: crate::store::memory::MemoryView::Session("webdev".to_string()),
                ..Default::default()
            })
            .await
            .expect("a webdev memory");
        let refused = store.remove_project("webdev").await.expect_err("in use");
        assert_eq!(
            refused.to_string(),
            "project 'webdev' has 1 memory and 1 session; move or remove them first"
        );
        assert_eq!(store.project_names().await.expect("names"), ["webdev"]);

        store
            .update_session(session, crate::store::SessionPatch {
                project: Some(String::new()),
                ..Default::default()
            })
            .await
            .expect("moved off");
        memories
            .delete("note", &crate::store::memory::MemoryView::All)
            .await
            .expect("deleted");
        store.remove_project("webdev").await.expect("removed");
        assert!(store.project_names().await.expect("names").is_empty());
        let gone = store.remove_project("webdev").await.expect_err("gone");
        assert!(matches!(gone, MekaError::UnknownProject { .. }), "{gone}");
    }
}
