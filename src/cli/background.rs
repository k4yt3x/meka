//! Listing and canceling background tasks from outside the agent.
//!
//! Read-and-cancel only, for the same reason [`crate::cli::schedule`] is: *starting* one needs a
//! session for the result to be reported into, and the decision to detach belongs to the agent
//! making the call.

use crate::{
    error::{MekaError, Result},
    store::{Store, background::TaskStatus},
};

/// Ceiling on the tool name. Server-chosen, so nothing else bounds it, and `show` has it in full.
const TOOL_TRUNCATE: usize = 24;

/// Ceiling on the command line a task ran, so the result keeps a share of the row. Chosen with
/// [`TOOL_TRUNCATE`] so that a row whose fixed cells are all at their widest (`completed`, a
/// seven-column elapsed) still leaves the result its floor.
const LABEL_TRUNCATE: usize = 36;

/// The listing's columns: the id prefix `show` and `cancel` accept, three facts, the two cells the
/// model or a program wrote for information, and the result spending the rest.
const TASK_COLUMNS: [crate::text::Column; 6] = [
    crate::text::Column::content("ID"),
    crate::text::Column::content("Status"),
    crate::text::Column::capped("Tool", TOOL_TRUNCATE),
    crate::text::Column::capped("What", LABEL_TRUNCATE),
    crate::text::Column::content("Elapsed"),
    crate::text::Column::remainder("Result"),
];

/// `/task` in the REPL: one row per task in this conversation.
pub(crate) async fn run_list_for_session(store: &Store, session: uuid::Uuid) -> Result<()> {
    render(
        store
            .background_store()
            .list_background_tasks(session)
            .await?,
        crate::render::Stream::Stderr,
    )
}

fn render(
    tasks: Vec<crate::store::background::BackgroundTask>,
    stream: crate::render::Stream,
) -> Result<()> {
    if tasks.is_empty() {
        // stderr: an empty list is a status note, not the data a script asked for.
        crate::streams::write_stderr_line("No background tasks.");
        return Ok(());
    }

    stream.write(crate::text::format_table(&TASK_COLUMNS, &task_rows(&tasks)))?;
    Ok(())
}

/// One row per task, separated from printing so the layout can be asserted.
fn task_rows(tasks: &[crate::store::background::BackgroundTask]) -> Vec<Vec<String>> {
    let ids: Vec<&str> = tasks.iter().map(|task| task.id.as_str()).collect();
    let id_width = crate::text::unique_prefix_len(ids.iter().copied()).max("ID".len());
    tasks
        .iter()
        .map(|task| {
            vec![
                task.id.get(..id_width).unwrap_or(&task.id).to_string(),
                task.status.name().to_string(),
                task.tool_name.clone(),
                crate::text::prose_cell(&task.label),
                format_elapsed(task),
                match &task.outcome {
                    Some(outcome) if task.status.is_terminal() => crate::text::prose_cell(outcome),
                    _ => "-".to_string(),
                },
            ]
        })
        .collect()
}

/// `/task show <id>`: one task, with the id in full.
///
/// The listing shortens an id to whatever distinguishes it, which is only safe because this prints
/// the whole thing. It also carries the two cells a column cannot hold: the command line as
/// written, and the outcome rather than a 40-column excerpt of it.
pub(crate) async fn show(
    store: &Store,
    session: uuid::Uuid,
    id_prefix: &str,
    stream: crate::render::Stream,
) -> Result<()> {
    let Some(task) = store
        .background_store()
        .resolve_background_task(session, id_prefix)
        .await?
    else {
        return Err(MekaError::Config(format!(
            "no background task matching '{id_prefix}'"
        )));
    };

    stream.write(show_lines(&task))?;
    Ok(())
}

/// One `name: value` line per field, separated from printing so the alignment and the sanitizing
/// can be asserted. Five of the values are model-authored or server-chosen, and this is the one
/// surface that prints them untruncated.
fn show_lines(task: &crate::store::background::BackgroundTask) -> String {
    let mut fields = vec![
        ("id", task.id.clone()),
        ("session", task.session_id.to_string()),
        ("status", task.status.name().to_string()),
        (
            "tool",
            crate::text::sanitize_to_line(&task.tool_name, usize::MAX),
        ),
        ("elapsed", format_elapsed(task)),
        (
            "started",
            crate::text::format_timestamp(task.started_at, crate::text::Precision::Seconds),
        ),
        ("finished", match task.finished_at {
            Some(at) => crate::text::format_timestamp(at, crate::text::Precision::Seconds),
            None => "-".to_string(),
        }),
        (
            "what",
            crate::text::sanitize_to_line(&task.label, usize::MAX),
        ),
    ];
    // Named before the result is printed, because the result below is then only the part that fit.
    // `render_outcomes` and `task_list` both tell the model where the rest went; this is the
    // human-facing surface that claims to print the outcome in full, so it is the one place the
    // pointer cannot be missing.
    if let Some(scratchpad) = &task.scratchpad_name {
        fields.push((
            "full output",
            format!(
                "scratchpad entry {}",
                crate::text::sanitize_to_line(scratchpad, usize::MAX)
            ),
        ));
    }
    if task.outcome.is_some() {
        fields.push(("result", String::new()));
    }
    let mut out = crate::text::format_fields(&fields);
    if let Some(outcome) = &task.outcome {
        // Sanitized per line rather than collapsed to one: an outcome is program output and its
        // line structure is most of what makes it readable. Indented, because it is not: a program
        // whose stdout contains `status:    completed` would otherwise render exactly like the
        // fields above it, and this command exists to be believed about what a task did.
        for line in outcome.lines() {
            out.push_str("  ");
            out.push_str(&crate::text::sanitize_to_line(line, usize::MAX));
            out.push('\n');
        }
    }
    out
}

/// Cancel one task in `session` by full or unique-prefix id, or every running one.
///
/// Records the terminal outcome; signaling the live handle is the caller's job, since only the
/// process that started a task holds its token.
pub(crate) async fn cancel(
    store: &Store,
    session: uuid::Uuid,
    id_prefix: Option<&str>,
) -> Result<Vec<String>> {
    let Some(id_prefix) = id_prefix else {
        let store = store.background_store();
        let running: Vec<String> = store
            .list_running_background_tasks(session)
            .await?
            .into_iter()
            .map(|task| task.id)
            .collect();
        for id in &running {
            store
                .finish_background_task(id, TaskStatus::Canceled, None, None)
                .await?;
        }
        return Ok(running);
    };

    let Some(task) = store
        .background_store()
        .resolve_background_task(session, id_prefix)
        .await?
    else {
        return Err(MekaError::Config(format!(
            "no background task matching '{id_prefix}'"
        )));
    };
    if task.status.is_terminal() {
        return Err(MekaError::Config(format!(
            "task {} already {}",
            task.short_id(),
            task.status.name()
        )));
    }
    store
        .background_store()
        .finish_background_task(&task.id, TaskStatus::Canceled, None, None)
        .await?;
    Ok(vec![task.id])
}

/// How long a task ran, or has been running, in a cell the table can budget for.
///
/// `humantime` spells a duration in full (`3years 4months 16days 17h 28m 57s` is 33 columns) and
/// nothing bounds how long a task has been running, so the two coarsest units are kept: they are
/// what a reader takes from this cell anyway.
fn format_elapsed(task: &crate::store::background::BackgroundTask) -> String {
    let seconds = task.elapsed().num_seconds().max(0) as u64;
    let spelled =
        humantime_serde::re::humantime::format_duration(std::time::Duration::from_secs(seconds))
            .to_string();
    spelled
        .split_whitespace()
        .take(2)
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::background::BackgroundTask;

    /// Every field starts its value in the same column, including the one that appears only when
    /// the output overflowed into a scratchpad.
    ///
    /// `full output` is the longest label and the only conditional one, so a width measured against
    /// the fields that always print reads as correct until a task is big enough to need it.
    #[test]
    fn every_field_of_a_task_starts_in_the_same_column() {
        let task = BackgroundTask {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: uuid::Uuid::new_v4(),
            tool_name: "execute_command".to_string(),
            label: "cargo test --all".to_string(),
            status: TaskStatus::Completed,
            outcome: Some("test result: ok".to_string()),
            scratchpad_name: Some("build-log".to_string()),
            started_at: chrono::Utc::now(),
            finished_at: Some(chrono::Utc::now()),
            announced_at: None,
            delivered_at: None,
        };

        let rendered = show_lines(&task);
        // The outcome is indented under `result:` rather than being a field of its own, and
        // `result:` carries no value to line anything up against. A value may hold colons of its
        // own (both timestamps do), so the label's is the first one.
        let columns: Vec<(usize, &str)> = rendered
            .lines()
            .filter(|line| !line.starts_with(' '))
            .filter_map(|line| {
                let label = line.find(':')?;
                let value = line[label + 1..].find(|c: char| c != ' ')? + label + 1;
                Some((value, line))
            })
            .collect();

        assert!(
            columns
                .iter()
                .any(|(_, line)| line.starts_with("full output:")),
            "the longest label has to be in the sample, or this proves nothing: {rendered}"
        );
        let (first_column, first_line) = *columns.first().expect("the fields rendered");
        for (column, line) in &columns {
            assert_eq!(
                *column, first_column,
                "`{line}` starts its value at {column}, `{first_line}` at {first_column}"
            );
        }
    }

    /// The values a model or a server chose reach this surface sanitized.
    ///
    /// `show` is the one command that prints them untruncated, so it is also the one where a forged
    /// `status:` row or a cleared screen would be most convincing. `sanitize_to_line` is asked for
    /// each of them; this checks it was actually asked.
    #[test]
    fn a_task_show_sanitizes_every_value_it_did_not_write() {
        let forged = "done\x1b[2J\rstatus:      completed";
        let task = BackgroundTask {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: uuid::Uuid::new_v4(),
            tool_name: format!("mcp__evil__{forged}"),
            label: forged.to_string(),
            status: TaskStatus::Completed,
            outcome: Some(forged.to_string()),
            scratchpad_name: Some(forged.to_string()),
            started_at: chrono::Utc::now(),
            finished_at: Some(chrono::Utc::now()),
            announced_at: None,
            delivered_at: None,
        };

        let rendered = show_lines(&task);
        assert!(
            !rendered.contains('\x1b'),
            "an escape reached the terminal: {rendered:?}"
        );
        assert!(
            !rendered.contains('\r'),
            "a carriage return can repaint the row above it: {rendered:?}"
        );
        assert!(
            rendered.contains("done"),
            "and the text either side of it still shows: {rendered:?}"
        );
    }

    async fn manager_with_session() -> (Store, uuid::Uuid) {
        let manager = Store::open(Some(std::path::Path::new(":memory:")), &Default::default())
            .await
            .expect("in-memory db");
        let session = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        (manager, session)
    }

    async fn seed(manager: &Store, session: uuid::Uuid, label: &str) -> BackgroundTask {
        let task = BackgroundTask {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: session,
            tool_name: "execute_command".to_string(),
            label: label.to_string(),
            status: TaskStatus::Running,
            outcome: None,
            scratchpad_name: None,
            started_at: chrono::Utc::now(),
            finished_at: None,
            announced_at: None,
            delivered_at: None,
        };
        manager
            .background_store()
            .start_background_task(&task)
            .await
            .expect("start task");
        task
    }

    #[tokio::test]
    async fn cancel_by_prefix_records_the_outcome() {
        let (manager, session) = manager_with_session().await;
        let task = seed(&manager, session, "sleep 600").await;

        let canceled = cancel(&manager, session, Some(&task.id[..8]))
            .await
            .expect("cancel");
        assert_eq!(canceled, vec![task.id.clone()]);

        let undelivered = manager
            .background_store()
            .list_undelivered_background_tasks(session)
            .await
            .expect("list");
        assert_eq!(undelivered.len(), 1);
        assert_eq!(undelivered[0].status, TaskStatus::Canceled);
    }

    /// `/task cancel ""` must not stop the only running task.
    ///
    /// `id.starts_with("")` is true of every id, so an unset variable would resolve to whichever
    /// task happened to be alone and cancel it, and the ambiguity error only appears once a second
    /// task exists. `--all` is the way to mean all of them.
    #[tokio::test]
    async fn canceling_an_empty_prefix_stops_nothing() {
        let (manager, session) = manager_with_session().await;
        let task = seed(&manager, session, "sleep 600").await;

        let error = cancel(&manager, session, Some(""))
            .await
            .expect_err("an empty prefix names no task");
        assert!(
            error.to_string().contains("no background task"),
            "it must read as a miss, not as an ambiguity: {error}"
        );
        assert_eq!(
            manager
                .background_store()
                .list_running_background_tasks(session)
                .await
                .expect("list")
                .first()
                .map(|running| running.id.clone()),
            Some(task.id),
            "and the task must still be running"
        );
    }

    #[tokio::test]
    async fn cancel_all_records_every_running_task() {
        let (manager, session) = manager_with_session().await;
        seed(&manager, session, "sleep 1").await;
        seed(&manager, session, "sleep 2").await;

        let canceled = cancel(&manager, session, None).await.expect("cancel all");
        assert_eq!(canceled.len(), 2);
        assert!(
            manager
                .background_store()
                .list_running_background_tasks(session)
                .await
                .expect("list")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn cancel_rejects_an_unknown_id() {
        let (manager, session) = manager_with_session().await;
        assert!(cancel(&manager, session, Some("deadbeef")).await.is_err());
    }

    #[tokio::test]
    async fn cancel_rejects_an_already_finished_task() {
        let (manager, session) = manager_with_session().await;
        let task = seed(&manager, session, "make").await;
        manager
            .background_store()
            .finish_background_task(&task.id, TaskStatus::Completed, None, None)
            .await
            .expect("finish");

        assert!(
            cancel(&manager, session, Some(&task.id[..8]))
                .await
                .is_err()
        );
    }

    /// The table has a budget, and both cells that carry authored text respect it.
    ///
    /// `tool_name` is chosen by an MCP server, so nothing else bounds it, and a long one would push
    /// every later column off the screen.
    #[test]
    fn the_task_table_fits_its_budget_and_sanitizes_what_it_did_not_write() {
        let task = BackgroundTask {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: uuid::Uuid::new_v4(),
            tool_name: "mcp__a_very_long_server_name__an_even_longer_tool".to_string(),
            label: "cargo test\u{1b}[31m\n1234abcd  running  x  y  z  forged".to_string(),
            status: TaskStatus::Completed,
            outcome: Some("out\u{1b}[31m\nput ".to_string() + &"x".repeat(400)),
            scratchpad_name: None,
            started_at: chrono::Utc::now(),
            finished_at: Some(chrono::Utc::now()),
            announced_at: None,
            delivered_at: None,
        };

        let table = crate::text::format_table(&TASK_COLUMNS, &task_rows(&[task]));
        assert_eq!(table.lines().count(), 2, "one task, one row: {table}");
        for line in table.lines() {
            assert!(
                !line.contains('\u{1b}'),
                "a line reaches a terminal verbatim, so an escape may not survive: {line:?}"
            );
            let width = crate::text::display_width(line);
            assert!(
                width <= crate::text::TABLE_WIDTH,
                "the row spends {width} of a {} budget: {line}",
                crate::text::TABLE_WIDTH
            );
        }
    }
}
