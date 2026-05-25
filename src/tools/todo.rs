//! `todo_write` tool and the shared todo list state. Lets the agent track
//! multi-step plans and surface their progress in the system prompt and the
//! REPL display.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use super::{Tool, ToolOutput};
use crate::{
    error::{AgshError, Result},
    permission::Permission,
    provider::ToolDefinition,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Done,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    pub id: String,
    pub description: String,
    pub status: TodoStatus,
}

pub type SharedTodoList = Arc<RwLock<Vec<TodoItem>>>;

pub(super) struct TodoWriteTool {
    pub todo_list: SharedTodoList,
}

#[async_trait]
impl Tool for TodoWriteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "todo_write".to_string(),
            description: "Create or update a structured task list. Replaces the entire list each \
                          call. Use this to break down multi-step work, track progress, and \
                          communicate status."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "description": "The complete task list (replaces any existing list)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Unique task identifier"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "What needs to be done"
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "done"],
                                    "description": "Task status"
                                }
                            },
                            "required": ["id", "description", "status"]
                        }
                    }
                },
                "required": ["tasks"]
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
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let tasks_value = input.get("tasks").ok_or_else(|| AgshError::ToolExecution {
            tool_name: "todo_write".to_string(),
            message: "missing 'tasks' parameter".to_string(),
        })?;

        let tasks: Vec<TodoItem> =
            serde_json::from_value(tasks_value.clone()).map_err(|error| {
                AgshError::ToolExecution {
                    tool_name: "todo_write".to_string(),
                    message: format!("invalid tasks format: {}", error),
                }
            })?;

        let count = tasks.len();
        *self.todo_list.write().await = tasks;

        Ok(ToolOutput::text(
            format!("Task list updated ({} tasks)", count),
            false,
        ))
    }
}

pub(super) struct TodoReadTool {
    pub todo_list: SharedTodoList,
}

#[async_trait]
impl Tool for TodoReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "todo_read".to_string(),
            description: "Read the current task list. Returns an empty list if no tasks have \
                          been written. Use this to check progress before deciding the next step."
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
        _cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let items = self.todo_list.read().await;
        let text = if items.is_empty() {
            "[Current task list]\n(empty)\n".to_string()
        } else {
            format_todo_for_context(&items)
        };
        Ok(ToolOutput::text(text, false))
    }
}

pub fn format_todo_for_context(items: &[TodoItem]) -> String {
    let mut output = String::from("[Current task list]\n");
    for item in items {
        let marker = match item.status {
            TodoStatus::Done => "[x]",
            TodoStatus::InProgress => "[~]",
            TodoStatus::Pending => "[ ]",
        };
        output.push_str(&format!("{} {} - {}\n", marker, item.id, item.description));
    }
    output.push('\n');
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tests::text_content;

    fn test_list() -> SharedTodoList {
        Arc::new(RwLock::new(Vec::new()))
    }

    #[tokio::test]
    async fn test_todo_write() {
        let list = test_list();
        let tool = TodoWriteTool {
            todo_list: list.clone(),
        };

        let result = tool
            .execute(
                serde_json::json!({
                    "tasks": [
                        {"id": "1", "description": "First task", "status": "pending"},
                        {"id": "2", "description": "Second task", "status": "done"}
                    ]
                }),
                CancellationToken::new(),
            )
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        assert!(text_content(&result).contains("2 tasks"));

        let items = list.read().await;
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].status, TodoStatus::Pending);
        assert_eq!(items[1].status, TodoStatus::Done);
    }

    #[tokio::test]
    async fn test_todo_write_rejects_unknown_status() {
        let list = test_list();
        let tool = TodoWriteTool {
            todo_list: list.clone(),
        };

        let result = tool
            .execute(
                serde_json::json!({
                    "tasks": [{"id": "1", "description": "x", "status": "bogus"}]
                }),
                CancellationToken::new(),
            )
            .await;

        assert!(
            result.is_err(),
            "an unrecognized status must fail the write"
        );
        assert!(list.read().await.is_empty(), "list must be untouched");
    }

    #[tokio::test]
    async fn test_todo_write_replaces_list() {
        let list = test_list();
        let tool = TodoWriteTool {
            todo_list: list.clone(),
        };

        tool.execute(
            serde_json::json!({
                "tasks": [{"id": "1", "description": "Old", "status": "pending"}]
            }),
            CancellationToken::new(),
        )
        .await
        .expect("should succeed");

        tool.execute(
            serde_json::json!({
                "tasks": [
                    {"id": "a", "description": "New A", "status": "in_progress"},
                    {"id": "b", "description": "New B", "status": "done"}
                ]
            }),
            CancellationToken::new(),
        )
        .await
        .expect("should succeed");

        let items = list.read().await;
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].id, "a");
        assert_eq!(items[1].id, "b");
    }

    #[tokio::test]
    async fn test_todo_read_empty() {
        let list = test_list();
        let tool = TodoReadTool {
            todo_list: list.clone(),
        };

        let result = tool
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .expect("should succeed");

        assert!(!result.is_error);
        let text = text_content(&result);
        assert!(text.contains("[Current task list]"));
        assert!(text.contains("(empty)"));
    }

    #[tokio::test]
    async fn test_todo_read_after_write() {
        let list = test_list();
        let writer = TodoWriteTool {
            todo_list: list.clone(),
        };
        let reader = TodoReadTool {
            todo_list: list.clone(),
        };

        writer
            .execute(
                serde_json::json!({
                    "tasks": [
                        {"id": "1", "description": "Plan", "status": "in_progress"},
                        {"id": "2", "description": "Execute", "status": "pending"}
                    ]
                }),
                CancellationToken::new(),
            )
            .await
            .expect("write should succeed");

        let result = reader
            .execute(serde_json::json!({}), CancellationToken::new())
            .await
            .expect("read should succeed");

        let text = text_content(&result);
        assert!(text.contains("[~] 1 - Plan"));
        assert!(text.contains("[ ] 2 - Execute"));
    }

    #[test]
    fn test_format_todo_for_context() {
        let items = vec![
            TodoItem {
                id: "1".to_string(),
                description: "Do something".to_string(),
                status: TodoStatus::Pending,
            },
            TodoItem {
                id: "2".to_string(),
                description: "Working on it".to_string(),
                status: TodoStatus::InProgress,
            },
            TodoItem {
                id: "3".to_string(),
                description: "Already done".to_string(),
                status: TodoStatus::Done,
            },
        ];

        let output = format_todo_for_context(&items);
        assert!(output.contains("[ ] 1 - Do something"));
        assert!(output.contains("[~] 2 - Working on it"));
        assert!(output.contains("[x] 3 - Already done"));
    }

    #[test]
    fn test_format_todo_empty() {
        let output = format_todo_for_context(&[]);
        assert!(output.contains("[Current task list]"));
    }
}
