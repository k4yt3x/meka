//! The task list the agent keeps for multi-step work: the items, their states, and the shared
//! handle every collaborator reads. The `todo` tool that edits it is `tools::todo`; this module is
//! the vocabulary, so the frontends and the system prompt can hold a list without holding a tool.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TodoStatus {
    Pending,
    #[serde(
        alias = "wip",
        alias = "in-progress",
        alias = "in progress",
        alias = "started"
    )]
    InProgress,
    #[serde(alias = "done", alias = "complete", alias = "finished")]
    Completed,
    #[serde(
        alias = "cancelled",
        alias = "skipped",
        alias = "dropped",
        alias = "wontfix"
    )]
    Canceled,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TodoItem {
    pub(crate) text: String,
    pub(crate) status: TodoStatus,
}
/// The full task-list state: a `title` (set by the agent when it builds the list and rendered as
/// the heading) and the ordered items. Task numbers are positional (1-based) and owned by the tool,
/// so they are derived from order rather than stored on the item.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct TodoState {
    pub(crate) title: Option<String>,
    pub(crate) items: Vec<TodoItem>,
}
/// The task list of one session, shared by handle between the `todo` tool that edits it, the turn
/// loop that renders it into the context block, and the frontend that draws it.
///
/// A `std` lock rather than a `tokio` one: the tool edits the state in place inside
/// [`Self::update`] with no `.await` in reach, and every other holder takes a copy. Poisoning is
/// recovered through [`crate::sync`] like every other `std` lock in the tree.
#[derive(Clone, Default)]
pub(crate) struct SharedTodoList(Arc<std::sync::RwLock<TodoState>>);

impl SharedTodoList {
    /// A copy of the current state.
    pub(crate) fn get(&self) -> TodoState {
        crate::sync::read(&self.0).clone()
    }

    /// Edit the state in place. One write lock spans the whole edit, so a reader never sees a list
    /// with its items replaced and its statuses not yet patched.
    pub(crate) fn update<R>(&self, edit: impl FnOnce(&mut TodoState) -> R) -> R {
        edit(&mut crate::sync::write(&self.0))
    }
}
/// One element of the `items` array: either a bare task string (status defaults to `pending`) or an
/// object carrying an explicit status. `Text` must come first so a JSON string matches it before
/// the object variant is tried.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum TodoItemInput {
    Text(String),
    Full {
        text: String,
        #[serde(default)]
        status: Option<TodoStatus>,
    },
}
impl From<TodoItemInput> for TodoItem {
    fn from(input: TodoItemInput) -> Self {
        match input {
            TodoItemInput::Text(text) => TodoItem {
                text,
                status: TodoStatus::Pending,
            },
            TodoItemInput::Full { text, status } => TodoItem {
                text,
                status: status.unwrap_or(TodoStatus::Pending),
            },
        }
    }
}
/// Render the task list as plain text (no ANSI), used both as the `todo` tool result echoed to the
/// model and in the per-turn / post-compaction context blocks. Terminal coloring lives separately
/// in `render::render_todo_list`.
pub(crate) fn format_todo_state(state: &TodoState) -> String {
    if state.items.is_empty() {
        return "(no tasks)\n".to_string();
    }

    // Heading is `TODO: <title>` (defensive fallback when somehow absent), followed by a blank line
    // and the tasks as a markdown checklist.
    let title = state.title.as_deref().unwrap_or("Tasks");
    let mut output = format!("TODO: {title}\n\n");

    for (index, item) in state.items.iter().enumerate() {
        let number = index + 1;
        let marker = match item.status {
            TodoStatus::Pending => "[ ]",
            TodoStatus::InProgress => "[~]",
            TodoStatus::Completed => "[x]",
            TodoStatus::Canceled => "[-]",
        };
        if item.status == TodoStatus::Canceled {
            output.push_str(&format!(
                "- {} {} (canceled) {}\n",
                marker, number, item.text
            ));
        } else {
            output.push_str(&format!("- {} {} {}\n", marker, number, item.text));
        }
    }

    // Soft-invariant footer: report violations of the "exactly one in_progress" convention without
    // blocking the call. The model self-corrects on its next update.
    let in_progress: Vec<usize> = state
        .items
        .iter()
        .enumerate()
        .filter(|(_, item)| item.status == TodoStatus::InProgress)
        .map(|(index, _)| index + 1)
        .collect();
    if in_progress.len() > 1 {
        let ids = in_progress
            .iter()
            .map(|id| format!("#{id}"))
            .collect::<Vec<_>>()
            .join(", ");
        output.push_str(&format!(
            "(!) tasks {ids} are all in_progress; keep exactly one in_progress at a time\n"
        ));
    } else if in_progress.is_empty()
        && state
            .items
            .iter()
            .any(|item| item.status == TodoStatus::Pending)
    {
        output.push_str("(!) no task in_progress; set the next task to in_progress\n");
    }

    output
}
