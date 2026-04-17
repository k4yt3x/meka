mod file;
mod find;
mod grep;
mod render;
pub(crate) mod scratchpad;
mod shell;
mod skill;
pub(crate) mod subagent;
pub(crate) mod todo;
mod util;
mod web;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

type DeferredSet = Arc<std::sync::RwLock<HashSet<String>>>;

use crate::error::Result;
use crate::permission::Permission;
use crate::provider::{ToolDefinition, ToolResultContent};
use crate::session::SessionManager;

pub type ReadTracker = Arc<RwLock<HashSet<PathBuf>>>;

#[derive(Debug)]
pub struct ToolOutput {
    pub content: Vec<ToolResultContent>,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn text(content: String, is_error: bool) -> Self {
        Self {
            content: vec![ToolResultContent::Text { text: content }],
            is_error,
        }
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn required_permission(&self) -> Permission;
    async fn execute(
        &self,
        input: serde_json::Value,
        cancellation: CancellationToken,
    ) -> Result<ToolOutput>;
}

pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
    deferred: DeferredSet,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            deferred: Arc::new(std::sync::RwLock::new(HashSet::new())),
        }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    /// Mark a tool as deferred. Deferred tools are available for execution but
    /// not included in the API tool definitions until auto-activated.
    pub fn mark_deferred(&self, name: &str) {
        self.deferred
            .write()
            .expect("deferred lock poisoned")
            .insert(name.to_string());
    }

    /// Activate a deferred tool so it appears in subsequent API calls.
    pub fn activate(&self, name: &str) {
        self.deferred
            .write()
            .expect("deferred lock poisoned")
            .remove(name);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|tool| tool.definition().name == name)
            .map(|tool| tool.as_ref())
    }

    /// Check if a tool is registered but currently deferred.
    pub fn is_deferred(&self, name: &str) -> bool {
        self.deferred
            .read()
            .expect("deferred lock poisoned")
            .contains(name)
    }

    /// Returns tool definitions for the API call, excluding deferred tools.
    pub fn definitions_for_permission(&self, permission: Permission) -> Vec<ToolDefinition> {
        let deferred = self.deferred.read().expect("deferred lock poisoned");
        self.tools
            .iter()
            .filter(|tool| {
                permission.allows(tool.required_permission())
                    && !deferred.contains(&tool.definition().name)
            })
            .map(|tool| tool.definition())
            .collect()
    }

    /// Returns name+description summaries for ALL tools including deferred ones.
    pub fn all_tool_summaries(&self, permission: Permission) -> Vec<(String, String)> {
        self.tools
            .iter()
            .filter(|tool| permission.allows(tool.required_permission()))
            .map(|tool| {
                let def = tool.definition();
                (def.name, def.description)
            })
            .collect()
    }

    /// Register the core tools shared by the main agent and sub-agents:
    /// file I/O, search, web, and shell execution.
    fn register_core_tools(
        &mut self,
        user_agent: &str,
        shared_permission: crate::permission::SharedPermission,
        sandbox_enabled: bool,
        sandbox_capability: crate::sandbox::SandboxCapability,
    ) {
        let read_tracker: ReadTracker = Arc::new(RwLock::new(HashSet::new()));
        self.register(Box::new(file::ReadFileTool {
            read_tracker: read_tracker.clone(),
        }));
        self.register(Box::new(file::EditFileTool { read_tracker }));
        self.register(Box::new(file::WriteFileTool));
        self.register(Box::new(find::FindFilesTool));
        self.register(Box::new(grep::SearchContentsTool));
        let web_client = reqwest::Client::builder()
            .user_agent(user_agent)
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        self.register(Box::new(web::FetchUrlTool {
            client: web_client.clone(),
        }));
        self.register(Box::new(web::WebSearchTool { client: web_client }));
        self.register(Box::new(shell::ExecuteCommandTool {
            sandbox_capability,
            shared_permission,
            sandbox_enabled,
        }));
    }

    pub fn build_default(
        user_agent: String,
        shared_permission: crate::permission::SharedPermission,
        sandbox_enabled: bool,
        sandbox_capability: crate::sandbox::SandboxCapability,
        todo_list: todo::SharedTodoList,
        session_manager: SessionManager,
        shared_session_id: Arc<RwLock<Option<Uuid>>>,
    ) -> Self {
        let mut registry = Self::new();
        registry.register_core_tools(
            &user_agent,
            shared_permission,
            sandbox_enabled,
            sandbox_capability,
        );
        registry.register(Box::new(skill::SkillTool {
            session_id: shared_session_id.clone(),
        }));
        registry.register(Box::new(render::RenderImageTool {
            session_id: shared_session_id.clone(),
            session_manager: session_manager.clone(),
        }));
        registry.register(Box::new(todo::TodoWriteTool { todo_list }));
        registry.register(Box::new(scratchpad::ScratchpadWriteTool {
            session_manager: session_manager.clone(),
            session_id: shared_session_id.clone(),
        }));
        registry.register(Box::new(scratchpad::ScratchpadReadTool {
            session_manager: session_manager.clone(),
            session_id: shared_session_id.clone(),
        }));
        registry.mark_deferred("scratchpad_read");
        registry.register(Box::new(scratchpad::ScratchpadEditTool {
            session_manager: session_manager.clone(),
            session_id: shared_session_id.clone(),
        }));
        registry.mark_deferred("scratchpad_edit");
        registry.register(Box::new(scratchpad::ScratchpadListTool {
            session_manager: session_manager.clone(),
            session_id: shared_session_id.clone(),
        }));
        registry.mark_deferred("scratchpad_list");
        registry.register(Box::new(scratchpad::ScratchpadDeleteTool {
            session_manager,
            session_id: shared_session_id,
        }));
        registry.mark_deferred("scratchpad_delete");
        registry
    }

    /// Build a tool registry for sub-agents. Excludes `todo_write` (parent
    /// owns task tracking) and `spawn_agent` (no recursive spawning).
    pub fn build_for_subagent(
        user_agent: String,
        shared_permission: crate::permission::SharedPermission,
        sandbox_enabled: bool,
        sandbox_capability: crate::sandbox::SandboxCapability,
    ) -> Self {
        let mut registry = Self::new();
        registry.register_core_tools(
            &user_agent,
            shared_permission,
            sandbox_enabled,
            sandbox_capability,
        );
        // Sub-agents don't have a session of their own — skills still load but
        // ${AGSH_SESSION_ID} stays unresolved for their invocations.
        registry.register(Box::new(skill::SkillTool {
            session_id: Arc::new(RwLock::new(None)),
        }));
        registry
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn test_shared_permission() -> crate::permission::SharedPermission {
        crate::permission::SharedPermission::new(Permission::Write)
    }

    fn test_todo_list() -> todo::SharedTodoList {
        Arc::new(RwLock::new(Vec::new()))
    }

    async fn test_registry() -> ToolRegistry {
        let session_manager = SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("failed to open in-memory database");
        let shared_session_id = Arc::new(RwLock::new(None));
        ToolRegistry::build_default(
            "test-agent/0.1".to_string(),
            test_shared_permission(),
            true,
            crate::sandbox::detect(),
            test_todo_list(),
            session_manager,
            shared_session_id,
        )
    }

    #[tokio::test]
    async fn test_tool_registry() {
        let registry = test_registry().await;
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("write_file").is_some());
        assert!(registry.get("edit_file").is_some());
        assert!(registry.get("find_files").is_some());
        assert!(registry.get("search_contents").is_some());
        assert!(registry.get("execute_command").is_some());
        assert!(registry.get("fetch_url").is_some());
        assert!(registry.get("web_search").is_some());
        assert!(registry.get("todo_write").is_some());
        assert!(registry.get("scratchpad_write").is_some());
        assert!(registry.get("scratchpad_read").is_some());
        assert!(registry.get("scratchpad_edit").is_some());
        assert!(registry.get("scratchpad_list").is_some());
        assert!(registry.get("scratchpad_delete").is_some());
        assert!(registry.get("skill").is_some());
        assert!(registry.get("render_image").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[tokio::test]
    async fn test_permission_filtering() {
        let registry = test_registry().await;

        let none_tools = registry.definitions_for_permission(Permission::None);
        assert!(none_tools.is_empty());

        let read_tools = registry.definitions_for_permission(Permission::Read);
        assert!(read_tools.iter().any(|t| t.name == "read_file"));
        assert!(read_tools.iter().any(|t| t.name == "find_files"));
        assert!(read_tools.iter().any(|t| t.name == "execute_command"));
        assert!(!read_tools.iter().any(|t| t.name == "write_file"));

        let write_tools = registry.definitions_for_permission(Permission::Write);
        assert!(write_tools.iter().any(|t| t.name == "read_file"));
        assert!(write_tools.iter().any(|t| t.name == "write_file"));
        assert!(write_tools.iter().any(|t| t.name == "execute_command"));
    }
}
