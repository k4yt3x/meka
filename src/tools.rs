//! Tool registry and built-in tool modules. Owns the [`ToolRegistry`] type
//! that the agent loop consults to resolve tool names to executable handlers,
//! plus the per-tool submodules (file, find, grep, scratchpad, shell, etc.).

mod file;
mod find;
mod grep;
pub(crate) mod mcp_resources;
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

#[derive(Debug, Default)]
pub struct ToolOutput {
    pub content: Vec<ToolResultContent>,
    pub is_error: bool,
    /// When `persist_oversized_results` has to spill to the scratchpad, use
    /// this name instead of the caller-supplied tool name. Set by MCP tool
    /// adapters so the persisted blob is namespaced as
    /// `mcp_<server>_<remote_tool>` for easier debugging.
    pub scratchpad_hint: Option<String>,
}

impl ToolOutput {
    pub fn text(content: String, is_error: bool) -> Self {
        Self {
            content: vec![ToolResultContent::Text { text: content }],
            is_error,
            scratchpad_hint: None,
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

type ToolSet = Arc<std::sync::RwLock<Vec<Arc<dyn Tool>>>>;

/// Tool registry. Backed by an `Arc<RwLock<Vec<Arc<dyn Tool>>>>` so MCP
/// notification handlers can swap a server's tools in place on
/// `tools/list_changed`. Individual registrations only hold the write lock
/// briefly; dispatch clones the matching `Arc<dyn Tool>` out of the lock
/// before awaiting `execute`, so no lock is held across `.await`.
#[derive(Clone)]
pub struct ToolRegistry {
    tools: ToolSet,
    deferred: DeferredSet,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: Arc::new(std::sync::RwLock::new(Vec::new())),
            deferred: Arc::new(std::sync::RwLock::new(HashSet::new())),
        }
    }

    /// Register a tool. Returns an error if another tool with the same name
    /// is already registered. Callers that know the tool is unique (e.g.
    /// core builtins) may `.expect()` the result; MCP registration should
    /// log and continue so one bad server can't break startup.
    pub fn register(&self, tool: Arc<dyn Tool>) -> Result<()> {
        let name = tool.definition().name.clone();
        let mut tools = self.tools.write().expect("tools lock poisoned");
        if tools.iter().any(|t| t.definition().name == name) {
            return Err(crate::error::AgshError::ToolRegistration {
                message: format!("tool name '{}' is already registered", name),
            });
        }
        tools.push(tool);
        Ok(())
    }

    /// Replace every tool whose name starts with `<server_name>__` with the
    /// supplied set. Used by `AgshClientHandler::on_tool_list_changed` to
    /// hot-swap a server's tools without restarting the agent. Deferred
    /// markers for removed tool names are cleared so the registry's deferred
    /// set doesn't grow unbounded.
    pub fn replace_server_tools(&self, server_name: &str, new_tools: Vec<Arc<dyn Tool>>) {
        let prefix = format!("{}__", server_name);
        let mut tools = self.tools.write().expect("tools lock poisoned");
        let removed: Vec<String> = tools
            .iter()
            .filter(|t| t.definition().name.starts_with(&prefix))
            .map(|t| t.definition().name)
            .collect();
        tools.retain(|t| !t.definition().name.starts_with(&prefix));
        tools.extend(new_tools);
        drop(tools);

        if !removed.is_empty() {
            let mut deferred = self.deferred.write().expect("deferred lock poisoned");
            for name in &removed {
                deferred.remove(name);
            }
        }
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

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools
            .read()
            .expect("tools lock poisoned")
            .iter()
            .find(|tool| tool.definition().name == name)
            .cloned()
    }

    /// Check if a tool is registered but currently deferred.
    pub fn is_deferred(&self, name: &str) -> bool {
        self.deferred
            .read()
            .expect("deferred lock poisoned")
            .contains(name)
    }

    /// Returns tool definitions for the API call, excluding deferred tools.
    /// Permission-filtered view — used by sub-agents which run at a fixed
    /// permission. The main agent uses [`Self::definitions_active`] so the
    /// tools array remains byte-identical across mid-session `/permission`
    /// toggles, keeping the Anthropic prompt cache warm on subsequent turns.
    pub fn definitions_for_permission(&self, permission: Permission) -> Vec<ToolDefinition> {
        let deferred = self.deferred.read().expect("deferred lock poisoned");
        self.tools
            .read()
            .expect("tools lock poisoned")
            .iter()
            .filter(|tool| {
                permission.allows(tool.required_permission())
                    && !deferred.contains(&tool.definition().name)
            })
            .map(|tool| tool.definition())
            .collect()
    }

    /// Returns every active (non-deferred) tool definition regardless of the
    /// caller's current permission. Blocked calls are rejected at dispatch;
    /// keeping the tools array permission-independent is what preserves the
    /// prompt cache prefix across `/permission` toggles (breakpoint 3 in the
    /// Claude provider's cache layout).
    pub fn definitions_active(&self) -> Vec<ToolDefinition> {
        let deferred = self.deferred.read().expect("deferred lock poisoned");
        self.tools
            .read()
            .expect("tools lock poisoned")
            .iter()
            .filter(|tool| !deferred.contains(&tool.definition().name))
            .map(|tool| tool.definition())
            .collect()
    }

    /// Returns (name, description, required_permission, is_deferred) for every
    /// registered tool. Drives the permission-independent system-prompt tool
    /// catalogue plus the per-turn `[Permission context]` block that names
    /// currently-blocked tools. Sorted by (name) for deterministic output.
    pub fn tool_catalogue(&self) -> Vec<(String, String, Permission, bool)> {
        let deferred = self.deferred.read().expect("deferred lock poisoned");
        let mut entries: Vec<(String, String, Permission, bool)> = self
            .tools
            .read()
            .expect("tools lock poisoned")
            .iter()
            .map(|tool| {
                let def = tool.definition();
                let is_deferred = deferred.contains(&def.name);
                (
                    def.name,
                    def.description,
                    tool.required_permission(),
                    is_deferred,
                )
            })
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries
    }

    /// Register the core tools shared by the main agent and sub-agents:
    /// file I/O, search, web, and shell execution.
    fn register_core_tools(
        &self,
        user_agent: &str,
        shared_permission: crate::permission::SharedPermission,
        sandbox_enabled: bool,
        sandbox_capability: crate::sandbox::SandboxCapability,
    ) {
        let read_tracker: ReadTracker = Arc::new(RwLock::new(HashSet::new()));
        self.register_builtin(Arc::new(file::ReadFileTool {
            read_tracker: read_tracker.clone(),
        }));
        self.register_builtin(Arc::new(file::EditFileTool { read_tracker }));
        self.register_builtin(Arc::new(file::WriteFileTool));
        self.register_builtin(Arc::new(find::FindFilesTool));
        self.register_builtin(Arc::new(grep::SearchContentsTool));
        let web_client = reqwest::Client::builder()
            .user_agent(user_agent)
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        self.register_builtin(Arc::new(web::FetchUrlTool {
            client: web_client.clone(),
        }));
        self.register_builtin(Arc::new(web::WebSearchTool { client: web_client }));
        self.register_builtin(Arc::new(shell::ExecuteCommandTool {
            sandbox_capability,
            shared_permission,
            sandbox_enabled,
        }));
    }

    /// Register a builtin tool. Builtins are statically known to be unique,
    /// so a collision is a programmer error — panic rather than swallow.
    fn register_builtin(&self, tool: Arc<dyn Tool>) {
        self.register(tool).expect("builtin tool name collision");
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
        let registry = Self::new();
        registry.register_core_tools(
            &user_agent,
            shared_permission,
            sandbox_enabled,
            sandbox_capability,
        );
        registry.register_builtin(Arc::new(skill::SkillTool {
            session_id: shared_session_id.clone(),
        }));
        registry.register_builtin(Arc::new(render::RenderImageTool {
            session_id: shared_session_id.clone(),
            session_manager: session_manager.clone(),
        }));
        registry.register_builtin(Arc::new(todo::TodoWriteTool { todo_list }));
        registry.register_builtin(Arc::new(scratchpad::ScratchpadWriteTool {
            session_manager: session_manager.clone(),
            session_id: shared_session_id.clone(),
        }));
        registry.register_builtin(Arc::new(scratchpad::ScratchpadReadTool {
            session_manager: session_manager.clone(),
            session_id: shared_session_id.clone(),
        }));
        registry.mark_deferred("scratchpad_read");
        registry.register_builtin(Arc::new(scratchpad::ScratchpadEditTool {
            session_manager: session_manager.clone(),
            session_id: shared_session_id.clone(),
        }));
        registry.mark_deferred("scratchpad_edit");
        registry.register_builtin(Arc::new(scratchpad::ScratchpadListTool {
            session_manager: session_manager.clone(),
            session_id: shared_session_id.clone(),
        }));
        registry.mark_deferred("scratchpad_list");
        registry.register_builtin(Arc::new(scratchpad::ScratchpadDeleteTool {
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
        let registry = Self::new();
        registry.register_core_tools(
            &user_agent,
            shared_permission,
            sandbox_enabled,
            sandbox_capability,
        );
        // Sub-agents don't have a session of their own — skills still load but
        // ${AGSH_SESSION_ID} stays unresolved for their invocations.
        registry.register_builtin(Arc::new(skill::SkillTool {
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

    #[tokio::test]
    async fn test_definitions_active_includes_write_tools() {
        let registry = test_registry().await;
        let active = registry.definitions_active();
        assert!(active.iter().any(|t| t.name == "read_file"));
        assert!(active.iter().any(|t| t.name == "write_file"));
        assert!(active.iter().any(|t| t.name == "edit_file"));
        assert!(active.iter().any(|t| t.name == "execute_command"));
        assert!(!active.iter().any(|t| t.name == "scratchpad_read"));
    }

    #[tokio::test]
    async fn test_definitions_active_stable_across_permissions() {
        let registry = test_registry().await;
        let a = registry.definitions_active();
        let b = registry.definitions_active();
        assert_eq!(a.len(), b.len());
        let a_names: Vec<_> = a.iter().map(|t| t.name.clone()).collect();
        let b_names: Vec<_> = b.iter().map(|t| t.name.clone()).collect();
        assert_eq!(a_names, b_names);
    }

    #[tokio::test]
    async fn test_tool_catalogue_covers_active_and_deferred() {
        let registry = test_registry().await;
        let entries = registry.tool_catalogue();
        let names: std::collections::HashSet<_> =
            entries.iter().map(|(n, _, _, _)| n.clone()).collect();
        assert!(names.contains("write_file"));
        assert!(names.contains("scratchpad_read"));
        assert!(names.contains("scratchpad_edit"));

        let deferred_names: Vec<_> = entries
            .iter()
            .filter(|(_, _, _, d)| *d)
            .map(|(n, _, _, _)| n.clone())
            .collect();
        assert!(deferred_names.iter().any(|n| n == "scratchpad_read"));

        let required: std::collections::HashMap<_, _> =
            entries.iter().map(|(n, _, p, _)| (n.clone(), *p)).collect();
        assert_eq!(required["read_file"], Permission::Read);
        assert_eq!(required["write_file"], Permission::Write);
    }

    #[tokio::test]
    async fn test_tool_catalogue_is_sorted() {
        let registry = test_registry().await;
        let entries = registry.tool_catalogue();
        let names: Vec<_> = entries.iter().map(|(n, _, _, _)| n.clone()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "tool_catalogue must return sorted entries");
    }

    #[tokio::test]
    async fn test_register_duplicate_returns_error() {
        struct DummyTool;
        #[async_trait::async_trait]
        impl Tool for DummyTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition::new(
                    "dup_tool".to_string(),
                    "dummy".to_string(),
                    serde_json::json!({}),
                )
            }
            fn required_permission(&self) -> Permission {
                Permission::Read
            }
            async fn execute(
                &self,
                _input: serde_json::Value,
                _cancellation: CancellationToken,
            ) -> crate::error::Result<ToolOutput> {
                Ok(ToolOutput::text(String::new(), false))
            }
        }

        let registry = ToolRegistry::new();
        registry
            .register(Arc::new(DummyTool) as Arc<dyn Tool>)
            .expect("first registration succeeds");
        let err = registry
            .register(Arc::new(DummyTool) as Arc<dyn Tool>)
            .expect_err("second registration with same name must fail");
        let message = format!("{}", err);
        assert!(
            message.contains("dup_tool"),
            "error message should mention the duplicate name, got: {}",
            message
        );
    }
}
