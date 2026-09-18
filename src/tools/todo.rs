//! The `todo_*` family and the shared task-list state: `todo_write` replaces the list, `todo_edit`
//! patches statuses by task number, `todo_read` returns it. Every call echoes the canonical state
//! back, so the model always has authoritative task numbers for its next update. The list is also
//! surfaced in the per-turn context block and the REPL display.

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde::Deserialize;

use super::{Tool, ToolOutput};
use crate::{
    error::Result,
    permission::Permission,
    provider::ToolDefinition,
    todo::{SharedTodoList, TodoItem, TodoItemInput, TodoStatus, format_todo_state},
};

/// Whether a call of `name` can change the list: what the dispatcher watches to announce a change
/// to the display. `todo_read` cannot, so it is not here.
pub(crate) fn changes_the_list(name: &str) -> bool {
    name == "todo_write" || name == "todo_edit"
}

/// The `scratchpad` property every tool advertises. Accepted because a call carrying it was once
/// refused outright; the list is state rather than output, so there is nothing to redirect.
fn scratchpad_property() -> serde_json::Value {
    serde_json::json!({
        "type": "string",
        "description": "Accepted for uniformity with every other tool; the list is kept as state, \
                        so nothing is redirected."
    })
}

const STATUSES: [&str; 4] = ["pending", "in_progress", "completed", "canceled"];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WriteInput {
    title: String,
    items: Vec<TodoItemInput>,
    #[serde(default, rename = "scratchpad")]
    _scratchpad: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EditInput {
    set: BTreeMap<String, TodoStatus>,
    #[serde(default, rename = "scratchpad")]
    _scratchpad: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadInput {
    #[serde(default, rename = "scratchpad")]
    _scratchpad: Option<String>,
}

/// The result every member returns: the model's next `todo_edit` needs the numbers this shows.
fn current_list(list: &SharedTodoList) -> ToolOutput {
    ToolOutput::text(format_todo_state(&list.get()), false)
}

pub(crate) struct TodoWriteTool {
    pub(crate) todo_list: SharedTodoList,
}

#[async_trait]
impl Tool for TodoWriteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "todo_write".to_string(),
            description: "Create or replace the task list for multi-step work: pass `title`, a \
                          short heading for the overall goal, and `items`, the whole list in \
                          order. Each item is a task string (status defaults to pending) or an \
                          object {\"text\":..., \"status\":...}. Tasks are numbered 1..N in order \
                          and the full list is returned. Flip statuses as you work with \
                          `todo_edit`; keep exactly one task in_progress."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "scratchpad": scratchpad_property(),
                    "title": {
                        "type": "string",
                        "description": "A short heading summarizing the overall goal of the list; \
                                        rendered above it and kept across later edits."
                    },
                    "items": {
                        "type": "array",
                        "description": "The whole list. Each entry is a task string (status \
                                        defaults to pending) or an object {text, status}. Tasks \
                                        are numbered 1..N in order.",
                        "items": {
                            "anyOf": [
                                { "type": "string" },
                                {
                                    "type": "object",
                                    "properties": {
                                        "text": {
                                            "type": "string",
                                            "description": "What needs to be done."
                                        },
                                        "status": {
                                            "type": "string",
                                            "enum": STATUSES
                                        }
                                    },
                                    "required": ["text"]
                                }
                            ]
                        }
                    }
                },
                "required": ["title", "items"]
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
        let parsed: WriteInput = match serde_json::from_value(input) {
            Ok(parsed) => parsed,
            Err(error) => {
                return Ok(ToolOutput::text(
                    format!("invalid todo_write input: {error}"),
                    true,
                ));
            }
        };
        let title = parsed.title.trim();
        if title.is_empty() {
            return Ok(ToolOutput::text(
                "todo_write: `title` must not be blank".to_string(),
                true,
            ));
        }
        self.todo_list.update(|state| {
            state.title = Some(title.to_string());
            state.items = parsed.items.into_iter().map(TodoItem::from).collect();
        });
        Ok(current_list(&self.todo_list))
    }
}

pub(crate) struct TodoEditTool {
    pub(crate) todo_list: SharedTodoList,
}

#[async_trait]
impl Tool for TodoEditTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "todo_edit".to_string(),
            description: "Update task statuses by number as you work: pass `set`, e.g. \
                          {\"1\":\"completed\",\"2\":\"in_progress\"}, as you start and finish \
                          each step. Keep exactly one task in_progress; mark a task completed only \
                          when truly done, or canceled if you drop it. Every number is checked \
                          before any status changes, and the full list is returned."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "scratchpad": scratchpad_property(),
                    "set": {
                        "type": "object",
                        "description": "Sparse status update keyed by 1-based task number, e.g. \
                                        {\"1\":\"completed\",\"2\":\"in_progress\"}.",
                        "additionalProperties": {
                            "type": "string",
                            "enum": STATUSES
                        }
                    }
                },
                "required": ["set"]
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
        let parsed: EditInput = match serde_json::from_value(input) {
            Ok(parsed) => parsed,
            Err(error) => {
                return Ok(ToolOutput::text(
                    format!("invalid todo_edit input: {error}"),
                    true,
                ));
            }
        };
        self.todo_list.update(|state| {
            let len = state.items.len();
            // Validate every key before mutating so a single bad id leaves the list untouched.
            let mut patches = Vec::with_capacity(parsed.set.len());
            for (key, status) in parsed.set {
                match key.parse::<usize>() {
                    Ok(id) if (1..=len).contains(&id) => patches.push((id, status)),
                    _ => {
                        return Ok(ToolOutput::text(
                            format!(
                                "todo_edit: '{}' is not a valid task number (the list has {} \
                                 task{})",
                                key,
                                len,
                                if len == 1 { "" } else { "s" }
                            ),
                            true,
                        ));
                    }
                }
            }
            for (id, status) in patches {
                if let Some(item) = state.items.get_mut(id - 1) {
                    item.status = status;
                }
            }
            Ok(ToolOutput::text(format_todo_state(state), false))
        })
    }
}

pub(crate) struct TodoReadTool {
    pub(crate) todo_list: SharedTodoList,
}

#[async_trait]
impl Tool for TodoReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "todo_read".to_string(),
            description: "Return the current task list with task numbers. The list is also shown \
                          in your context at the start of every turn, so this is for a look \
                          mid-turn."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "scratchpad": scratchpad_property()
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
        if let Err(error) = serde_json::from_value::<ReadInput>(input) {
            return Ok(ToolOutput::text(
                format!("invalid todo_read input: {error}"),
                true,
            ));
        }
        Ok(current_list(&self.todo_list))
    }
}

#[cfg(test)]
mod tests {
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::todo::TodoState;

    struct Family {
        write: TodoWriteTool,
        edit: TodoEditTool,
        read: TodoReadTool,
        list: SharedTodoList,
    }

    fn family() -> Family {
        let list = SharedTodoList::default();
        Family {
            write: TodoWriteTool {
                todo_list: list.clone(),
            },
            edit: TodoEditTool {
                todo_list: list.clone(),
            },
            read: TodoReadTool {
                todo_list: list.clone(),
            },
            list,
        }
    }

    async fn call(tool: &dyn Tool, input: serde_json::Value) -> ToolOutput {
        tool.execute(
            input,
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("a todo call answers with a result, never an Err")
    }

    #[tokio::test]
    async fn a_read_takes_no_arguments() {
        let family = family();
        let result = call(&family.read, serde_json::json!({})).await;
        assert!(!result.is_error);
        assert!(result.text_content().contains("(no tasks)"));
        let refused = call(&family.read, serde_json::json!({"set": {"1": "completed"}})).await;
        assert!(refused.is_error, "{}", refused.text_content());
    }

    /// `scratchpad` is the parameter every tool takes. `deny_unknown_fields` refused a call that
    /// carried it and the list was never written.
    #[tokio::test]
    async fn the_universal_scratchpad_parameter_is_accepted() {
        let family = family();
        let result = call(
            &family.write,
            serde_json::json!({ "title": "Plan", "items": ["One"], "scratchpad": "plan" }),
        )
        .await;
        assert!(!result.is_error, "{}", result.text_content());
        assert_eq!(family.list.get().items.len(), 1);
        let result = call(&family.read, serde_json::json!({ "scratchpad": "plan" })).await;
        assert!(!result.is_error, "{}", result.text_content());
    }

    #[tokio::test]
    async fn a_write_replaces_the_list_and_needs_a_title() {
        let family = family();
        call(
            &family.write,
            serde_json::json!({ "title": "Setup", "items": ["First", "Second"] }),
        )
        .await;
        let state = family.list.get();
        assert_eq!(state.title.as_deref(), Some("Setup"));
        assert_eq!(state.items.len(), 2);
        assert_eq!(state.items[0].text, "First");
        assert_eq!(state.items[0].status, TodoStatus::Pending);

        call(
            &family.write,
            serde_json::json!({ "title": "Again", "items": ["Only"] }),
        )
        .await;
        let state = family.list.get();
        assert_eq!(state.title.as_deref(), Some("Again"));
        assert_eq!(state.items.len(), 1, "a write replaces, never appends");

        for input in [
            serde_json::json!({ "items": ["A", "B"] }),
            serde_json::json!({ "title": "  ", "items": ["A", "B"] }),
        ] {
            let refused = call(&family.write, input).await;
            assert!(refused.is_error);
            assert!(
                refused.text_content().contains("title"),
                "{}",
                refused.text_content()
            );
        }
        assert_eq!(
            family.list.get().items.len(),
            1,
            "a refused write leaves the list alone"
        );
    }

    #[tokio::test]
    async fn items_as_objects_honor_status() {
        let family = family();
        call(
            &family.write,
            serde_json::json!({
                "title": "Work",
                "items": [
                    {"text": "A", "status": "in_progress"},
                    {"text": "B"}
                ]
            }),
        )
        .await;
        let state = family.list.get();
        assert_eq!(state.items[0].status, TodoStatus::InProgress);
        assert_eq!(state.items[1].status, TodoStatus::Pending);
    }

    #[tokio::test]
    async fn status_aliases_map_onto_the_canonical_statuses() {
        let family = family();
        call(
            &family.write,
            serde_json::json!({
                "title": "Work",
                "items": [
                    {"text": "A", "status": "done"},
                    {"text": "B", "status": "wip"},
                    {"text": "C", "status": "skipped"},
                    {"text": "D", "status": "cancelled"}
                ]
            }),
        )
        .await;
        let state = family.list.get();
        assert_eq!(state.items[0].status, TodoStatus::Completed);
        assert_eq!(state.items[1].status, TodoStatus::InProgress);
        assert_eq!(state.items[2].status, TodoStatus::Canceled);
        assert_eq!(
            state.items[3].status,
            TodoStatus::Canceled,
            "a model may spell it either way"
        );
        assert_eq!(
            serde_json::to_value(TodoStatus::Canceled).expect("serializes"),
            serde_json::json!("canceled"),
            "meka writes one spelling"
        );
    }

    #[tokio::test]
    async fn an_edit_patches_statuses_and_refuses_a_bad_number_whole() {
        let family = family();
        call(
            &family.write,
            serde_json::json!({ "title": "Work", "items": ["A", "B", "C"] }),
        )
        .await;
        let result = call(
            &family.edit,
            serde_json::json!({ "set": {"1": "completed", "2": "in_progress"} }),
        )
        .await;
        assert!(!result.is_error, "{}", result.text_content());
        let state = family.list.get();
        assert_eq!(
            state.title.as_deref(),
            Some("Work"),
            "the title persists across edits"
        );
        assert_eq!(state.items[0].status, TodoStatus::Completed);
        assert_eq!(state.items[1].status, TodoStatus::InProgress);
        assert_eq!(state.items[2].status, TodoStatus::Pending);

        // One bad number in a batch refuses the whole batch, so the good half is not half-applied.
        let refused = call(
            &family.edit,
            serde_json::json!({ "set": {"3": "completed", "9": "completed"} }),
        )
        .await;
        assert!(refused.is_error);
        assert!(refused.text_content().contains("valid task number"));
        assert_eq!(family.list.get().items[2].status, TodoStatus::Pending);

        let refused = call(
            &family.edit,
            serde_json::json!({ "set": {"abc": "completed"} }),
        )
        .await;
        assert!(refused.is_error);
        let refused = call(&family.edit, serde_json::json!({})).await;
        assert!(refused.is_error, "`set` is required");
    }

    #[tokio::test]
    async fn rejects_unknown_status() {
        let family = family();
        let result = call(
            &family.write,
            serde_json::json!({ "title": "Work", "items": [{"text": "x", "status": "bogus"}] }),
        )
        .await;
        assert!(result.is_error);
        assert!(family.list.get().items.is_empty(), "list must be untouched");
    }

    #[test]
    fn only_the_writing_members_change_the_list() {
        assert!(changes_the_list("todo_write"));
        assert!(changes_the_list("todo_edit"));
        assert!(!changes_the_list("todo_read"));
        assert!(!changes_the_list("file_read"));
    }

    #[test]
    fn format_heading_and_markers() {
        let state = TodoState {
            title: Some("My tasks".to_string()),
            items: vec![
                TodoItem {
                    text: "Pending one".to_string(),
                    status: TodoStatus::Pending,
                },
                TodoItem {
                    text: "Working".to_string(),
                    status: TodoStatus::InProgress,
                },
                TodoItem {
                    text: "Finished".to_string(),
                    status: TodoStatus::Completed,
                },
                TodoItem {
                    text: "Dropped".to_string(),
                    status: TodoStatus::Canceled,
                },
            ],
        };
        let output = format_todo_state(&state);
        assert!(output.starts_with("TODO: My tasks\n\n"));
        assert!(!output.contains("done"));
        assert!(output.contains("- [ ] 1 Pending one"));
        assert!(output.contains("- [~] 2 Working"));
        assert!(output.contains("- [x] 3 Finished"));
        assert!(output.contains("- [-] 4 (canceled) Dropped"));
    }

    #[test]
    fn format_soft_invariant_multiple_in_progress() {
        let state = TodoState {
            items: vec![
                TodoItem {
                    text: "A".to_string(),
                    status: TodoStatus::InProgress,
                },
                TodoItem {
                    text: "B".to_string(),
                    status: TodoStatus::InProgress,
                },
            ],
            ..Default::default()
        };
        let output = format_todo_state(&state);
        assert!(output.contains("(!) tasks #1, #2 are all in_progress"));
    }

    #[test]
    fn format_soft_invariant_none_in_progress() {
        let state = TodoState {
            items: vec![TodoItem {
                text: "A".to_string(),
                status: TodoStatus::Pending,
            }],
            ..Default::default()
        };
        assert!(format_todo_state(&state).contains("(!) no task in_progress"));
    }

    #[test]
    fn an_empty_todo_list_formats_as_no_tasks() {
        assert!(format_todo_state(&TodoState::default()).contains("(no tasks)"));
    }
}
