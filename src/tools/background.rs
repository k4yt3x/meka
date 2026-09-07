//! The `task_*` tools: the agent watching and stopping its own background work
//! ([`crate::background`]).
//!
//! Both gate at [`Permission::Read`]. Neither runs anything: listing reads meka's own store, and
//! canceling only signals a token belonging to work whose permission was already checked when it
//! was dispatched. Requiring `unrestricted` to stop something the agent itself started would leave
//! it unable to clean up after a call it had every right to make.
//!
//! Sub-agents deliberately get neither, for the same reason they get no `schedule_*`: a sub-agent's
//! session ends with the single turn that spawned it, so it can neither start a task that outlives
//! that turn nor be around to hear about one.

use std::sync::Arc;

use async_trait::async_trait;

use super::{
    Tool, ToolOutput,
    util::{require_str, resolve_session_id},
};
use crate::{
    background::{BackgroundTasks, TASK_INDEX_TOOL},
    error::Result,
    permission::Permission,
    provider::ToolDefinition,
    store::{Store, background::TaskStatus},
};

/// How much of a finished task's output `task_list` shows. Enough to recognize what happened, not
/// enough to make listing tasks a way to re-read every result.
const OUTCOME_EXCERPT_CHARS: usize = 200;

/// Ceiling on the rendered label. `format_columns` widens a column to its longest cell, so an
/// unbounded command line would push everything after it far off to the right.
const LABEL_EXCERPT_CHARS: usize = 48;

struct TaskContext {
    store: Store,
    site: crate::session::ToolSite,
    tasks: BackgroundTasks,
}

pub(super) struct TaskListTool {
    context: TaskContext,
}

#[async_trait]
impl Tool for TaskListTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: TASK_INDEX_TOOL.to_string(),
            description: "List this session's background tasks: what is still running, and what \
                has already reported. Your per-turn context carries a short index of the running \
                ones, so reach for this when you want the full picture including finished tasks."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        _context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let session_id = resolve_session_id(&self.context.site.session_id, TASK_INDEX_TOOL)?;
        let tasks = self
            .context
            .store
            .background_store()
            .list_background_tasks(session_id)
            .await?;
        if tasks.is_empty() {
            return Ok(ToolOutput::text(
                "No background tasks in this session.".to_string(),
                false,
            ));
        }

        // The same column layout every other listing in meka uses (`scratchpad_list`, `meka session
        // list`, `meka mcp list`), so a reader who has seen one has seen them all.
        let rows: Vec<Vec<String>> = tasks
            .iter()
            .map(|task| {
                vec![
                    task.short_id().to_string(),
                    task.status.name().to_string(),
                    task.tool_name.clone(),
                    crate::background::excerpt(&task.label, LABEL_EXCERPT_CHARS),
                    humantime_serde::re::humantime::format_duration(
                        std::time::Duration::from_secs(task.elapsed().num_seconds().max(0) as u64),
                    )
                    .to_string(),
                    // An excerpt for anything already finished. Outcomes are delivered as their
                    // own turn, but that delivery is stamped before the turn runs, so a turn that
                    // fails (a provider error, an interrupt) consumes the report. Without this the
                    // result would be reachable only by reading the database by hand, which for
                    // the agent means not at all.
                    match (&task.outcome, &task.scratchpad_name) {
                        (_, Some(name)) => format!("in scratchpad '{name}'"),
                        (Some(outcome), None) if task.status.is_terminal() => {
                            crate::background::excerpt(outcome, OUTCOME_EXCERPT_CHARS)
                        }
                        _ => "-".to_string(),
                    },
                ]
            })
            .collect();

        let mut rendered = crate::text::format_columns(
            &["ID", "Status", "Tool", "What", "Elapsed", "Result"],
            &rows,
        );
        rendered.push_str(
            "\nA finished task's full result is delivered to you on its own; this listing is for \
             checking what is still running.",
        );
        Ok(ToolOutput::text(rendered, false))
    }
}

pub(super) struct TaskCancelTool {
    context: TaskContext,
}

#[async_trait]
impl Tool for TaskCancelTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "task_cancel".to_string(),
            description:
                "Stop a running background task by id (the short form from `task_list` is \
                enough), or every one of them with all=true. A canceled task still reports back, \
                so you will be told when it has actually stopped."
                    .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Task id, in full or the short prefix `task_list` shows.",
                    },
                    "all": {
                        "type": "boolean",
                        "default": false,
                        "description": "Cancel every running task in this session instead of one by id. Default: false.",
                    }
                }
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let session_id = resolve_session_id(&self.context.site.session_id, "task_cancel")?;

        if input
            .get("all")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            // Recorded before signaling, exactly as the single-id path below does and for the same
            // reason: the work reacting to its token reports an interruption, which would otherwise
            // land as `failed` and tell the agent its build broke rather than that it was stopped.
            let ids = self.context.tasks.session_task_ids(session_id).await;
            let store = self.context.store.background_store();
            for id in &ids {
                store
                    .finish_background_task(id, TaskStatus::Canceled, None, None)
                    .await?;
            }
            let signaled = self.context.tasks.cancel_session(session_id).await;
            return Ok(ToolOutput::text(
                if signaled == 0 {
                    "No running background tasks to cancel.".to_string()
                } else {
                    format!(
                        "Asked {signaled} background task(s) to stop. Each will report back once it has."
                    )
                },
                false,
            ));
        }

        let id_prefix = require_str(&input, "id", "task_cancel")?;
        let Some(task) = self
            .context
            .store
            .background_store()
            .resolve_background_task(session_id, &id_prefix)
            .await?
        else {
            return Ok(ToolOutput::text(
                format!(
                    "Error: no background task in this session matches '{id_prefix}'. Call `task_list` for \
                     the current ids."
                ),
                true,
            ));
        };

        if task.status.is_terminal() {
            return Ok(ToolOutput::text(
                format!(
                    "Task {} already {} and is not running.",
                    task.short_id(),
                    task.status.name()
                ),
                false,
            ));
        }

        // Record the cancellation before signaling. `finish_background_task` only writes over a
        // `running` row, so whichever of the two lands first wins, and doing it in this order means
        // a task that happens to finish in the same instant cannot report success after the agent
        // was told it was stopped.
        self.context
            .store
            .background_store()
            .finish_background_task(&task.id, TaskStatus::Canceled, None, None)
            .await?;
        let signaled = self.context.tasks.cancel(&task.id).await;
        if !signaled {
            // The row was ours to retire but the handle was not: the task belonged to a process
            // that is gone. Recording it is still the right move, and is what stops the agent
            // waiting forever.
            let short_id = task.short_id();
            tracing::debug!(
                "task {short_id} had no live handle in this process; recorded as canceled"
            );
        }
        Ok(ToolOutput::text(
            format!(
                "Asked task {} ({}) to stop. It will report back once it has.",
                task.short_id(),
                task.label
            ),
            false,
        ))
    }
}

pub(super) fn build(
    store: Store,
    site: crate::session::ToolSite,
    tasks: BackgroundTasks,
) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(TaskListTool {
            context: TaskContext {
                store: store.clone(),
                tasks: tasks.clone(),
                site: site.clone(),
            },
        }),
        Arc::new(TaskCancelTool {
            context: TaskContext { store, tasks, site },
        }),
    ]
}

#[cfg(test)]
mod tests {
    use tokio_util::sync::CancellationToken;
    use uuid::Uuid;

    use super::*;
    use crate::store::background::BackgroundTask;

    async fn context() -> (TaskContext, Uuid) {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        (
            TaskContext {
                store,
                tasks: BackgroundTasks::default(),
                site: crate::session::ToolSite::for_test()
                    .with_session_id(crate::session::SharedSessionId::new(Some(session_id))),
            },
            session_id,
        )
    }

    async fn seed(context: &TaskContext, session_id: Uuid, label: &str) -> BackgroundTask {
        let task = BackgroundTask {
            id: Uuid::new_v4().to_string(),
            session_id,
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
        context
            .store
            .background_store()
            .start_background_task(&task)
            .await
            .expect("start task");
        task
    }

    #[tokio::test]
    async fn task_list_reports_nothing_when_there_is_nothing() {
        let (context, _) = context().await;
        let tool = TaskListTool { context };
        let result = tool
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("ok");
        assert!(!result.is_error);
        assert!(result.text_content().contains("No background tasks"));
    }

    #[tokio::test]
    async fn task_list_names_each_task_and_its_state() {
        let (context, session_id) = context().await;
        seed(&context, session_id, "cargo test --all").await;
        let tool = TaskListTool { context };
        let result = tool
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("ok");
        let text = result.text_content();
        assert!(text.contains("ID"), "column headers: {text}");
        assert!(text.contains("cargo test --all"), "{text}");
        assert!(text.contains("running"), "{text}");
    }

    /// `[failed] … running for 30s` reads as a contradiction and invites the agent to keep waiting
    /// on work that already stopped.
    #[tokio::test]
    async fn task_list_does_not_say_a_finished_task_is_running() {
        let (context, session_id) = context().await;
        let task = seed(&context, session_id, "make").await;
        context
            .store
            .background_store()
            .finish_background_task(&task.id, TaskStatus::Failed, Some("boom".to_string()), None)
            .await
            .expect("finish");

        let tool = TaskListTool { context };
        let result = tool
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("ok");
        let text = result.text_content();
        // Scoped to the task's own row: the trailing note legitimately mentions what is still
        // running, and asserting over the whole output would be testing the footer.
        let row = text
            .lines()
            .find(|line| line.contains(task.short_id()))
            .unwrap_or_else(|| panic!("no row for the task: {text}"));
        assert!(row.contains("failed"), "{row}");
        // The `Status` column is the single place a task's state is stated. An elapsed time
        // labeled "running for" beside a `failed` badge read as a contradiction and invited the
        // agent to keep waiting on work that had already stopped.
        assert!(!row.contains("running"), "{row}");
    }

    /// The cancellation has to be recorded even when the handle is gone, or the agent waits forever
    /// on a task it was told it had stopped.
    #[tokio::test]
    async fn task_cancel_records_a_terminal_outcome() {
        let (context, session_id) = context().await;
        let task = seed(&context, session_id, "sleep 600").await;
        let store = context.store.clone();
        let tool = TaskCancelTool { context };

        let result = tool
            .execute(
                serde_json::json!({"id": &task.id[..8]}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("ok");
        assert!(!result.is_error, "{:?}", result.content);

        let undelivered = store
            .background_store()
            .list_undelivered_background_tasks(session_id)
            .await
            .expect("list");
        assert_eq!(undelivered.len(), 1);
        assert_eq!(undelivered[0].status, TaskStatus::Canceled);
    }

    #[tokio::test]
    async fn task_cancel_rejects_an_unknown_id() {
        let (context, _) = context().await;
        let tool = TaskCancelTool { context };
        let result = tool
            .execute(
                serde_json::json!({"id": "deadbeef"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("ok");
        assert!(result.is_error);
        assert!(result.text_content().contains("no background task"));
    }

    /// Canceling in bulk must record `canceled`, not leave the task's own interruption to land as
    /// `failed`. "Your build failed" and "you stopped your build" call for different next moves.
    /// Outcome delivery is stamped before its turn runs, so a turn that fails consumes the report.
    /// Listing has to be able to recover it, or the result is reachable only by reading the
    /// database by hand.
    #[tokio::test]
    async fn task_list_shows_a_finished_task_s_result() {
        let (context, session_id) = context().await;
        let task = seed(&context, session_id, "cargo test").await;
        context
            .store
            .background_store()
            .finish_background_task(
                &task.id,
                TaskStatus::Completed,
                Some("42 passed; 0 failed".to_string()),
                None,
            )
            .await
            .expect("finish");

        let tool = TaskListTool { context };
        let result = tool
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("ok");
        let text = result.text_content();
        assert!(text.contains("42 passed"), "{text}");
    }

    /// A running task has no result yet, so listing must not invent one.
    #[tokio::test]
    async fn task_list_shows_no_result_for_a_running_task() {
        let (context, session_id) = context().await;
        seed(&context, session_id, "sleep 600").await;
        let tool = TaskListTool { context };
        let result = tool
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("ok");
        let text = result.text_content();
        let row = text
            .lines()
            .find(|line| line.contains("sleep 600"))
            .unwrap_or_else(|| panic!("the running task's row is missing from:\n{text}"));
        assert!(
            row.trim_end().ends_with('-'),
            "a running task's Result cell must be the `-` placeholder: {row}"
        );
    }

    #[tokio::test]
    async fn task_cancel_all_records_canceled_not_failed() {
        let (context, session_id) = context().await;
        let task = seed(&context, session_id, "sleep 600").await;
        // A live handle, so the bulk path has something to enumerate.
        context
            .tasks
            .try_reserve(task.id.clone(), session_id, CancellationToken::new(), 10)
            .await;
        let store = context.store.clone();
        let tool = TaskCancelTool { context };

        let result = tool
            .execute(
                serde_json::json!({"all": true}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("ok");
        assert!(!result.is_error);

        let undelivered = store
            .background_store()
            .list_undelivered_background_tasks(session_id)
            .await
            .expect("list");
        assert_eq!(undelivered.len(), 1);
        assert_eq!(undelivered[0].status, TaskStatus::Canceled);
    }

    #[tokio::test]
    async fn task_cancel_all_is_a_no_op_when_nothing_runs() {
        let (context, _) = context().await;
        let tool = TaskCancelTool { context };
        let result = tool
            .execute(
                serde_json::json!({"all": true}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("ok");
        assert!(!result.is_error);
        assert!(result.text_content().contains("No running background"));
    }
}
