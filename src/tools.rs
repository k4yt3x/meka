//! Tool registry and built-in tool modules. Owns the [`ToolRegistry`] type
//! that the agent loop consults to resolve tool names to executable handlers,
//! plus the per-tool submodules (file, find, grep, scratchpad, shell, etc.).

mod file;
mod find;
mod grep;
mod load_tool;
pub(crate) mod mcp_resources;
mod render_image;
pub(crate) mod scratchpad;
mod shell;
mod skill;
pub(crate) mod subagent;
pub(crate) mod todo;
mod util;
mod web;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

type DeferredSet = Arc<std::sync::RwLock<HashSet<String>>>;

/// Name of the meta-tool that loads a deferred tool's schema. Calls to this
/// tool are scanned out of the conversation to compute the per-turn active
/// tool set; see [`extract_loaded_tool_names`].
pub const LOAD_TOOL_NAME: &str = "load_tool";

use crate::error::Result;
use crate::permission::Permission;
use crate::provider::{ContentBlock, Message, ToolDefinition, ToolResultContent};
use crate::session::SessionManager;

/// Walk the conversation and collect the names of tools that have been
/// loaded via successful `load_tool` calls. A `load_tool` `tool_use` block
/// counts only when paired with a non-error `tool_result` whose
/// `tool_use_id` matches; this excludes errored loads (unknown name,
/// malformed args) and orphan `tool_use` blocks awaiting their result.
///
/// The active set is a pure function of the message slice, so the tools
/// array sent to the Claude API is deterministic given the conversation
/// state. Resumed sessions reconstruct the exact active set their suspend
/// time had, with no out-of-band state.
pub fn extract_loaded_tool_names(messages: &[Message]) -> HashSet<String> {
    let mut pending: HashMap<String, String> = HashMap::new();
    let mut loaded: HashSet<String> = HashSet::new();
    for message in messages {
        for block in &message.content {
            match block {
                ContentBlock::ToolUse { id, name, input } if name == LOAD_TOOL_NAME => {
                    if let Some(loaded_name) = input.get("name").and_then(|v| v.as_str()) {
                        pending.insert(id.clone(), loaded_name.to_string());
                    }
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    is_error,
                    ..
                } => {
                    if let Some(loaded_name) = pending.remove(tool_use_id)
                        && !is_error
                    {
                        loaded.insert(loaded_name);
                    }
                }
                _ => {}
            }
        }
    }
    loaded
}

/// Built-in tool policy from `[tools]` in `config.toml`. Mirrors the three
/// knobs [`crate::config::McpServerConfig`] exposes for MCP tools.
#[derive(Debug, Clone, Default)]
pub struct BuiltinToolFilter {
    pub allowed: Option<HashSet<String>>,
    pub disabled: HashSet<String>,
    pub permission_overrides: HashMap<String, Permission>,
}

impl BuiltinToolFilter {
    pub fn from_config(
        allowed: Option<Vec<String>>,
        disabled: Vec<String>,
        permission_overrides: HashMap<String, Permission>,
    ) -> Self {
        // Empty allow-list → None so `admits` treats it as "no restriction".
        let allowed = allowed.and_then(|list| {
            if list.is_empty() {
                None
            } else {
                Some(list.into_iter().collect())
            }
        });
        Self {
            allowed,
            disabled: disabled.into_iter().collect(),
            permission_overrides,
        }
    }

    pub fn admits(&self, name: &str) -> bool {
        if self.disabled.contains(name) {
            return false;
        }
        match &self.allowed {
            Some(list) => list.contains(name),
            None => true,
        }
    }
}

/// Canonical built-in names for the stale-entry warning pass. Update when
/// adding a new built-in in [`ToolRegistry::build_default`].
pub const BUILTIN_TOOL_NAMES: &[&str] = &[
    "edit_file",
    "execute_command",
    "fetch_url",
    "find_files",
    "load_tool",
    "read_file",
    "render_image",
    "scratchpad_delete",
    "scratchpad_edit",
    "scratchpad_list",
    "scratchpad_read",
    "scratchpad_write",
    "search_contents",
    "skill",
    "spawn_agent",
    "todo_write",
    "web_search",
    "write_file",
];

/// Warn (never fail) on `[tools]` entries that don't match any known
/// built-in. Mirrors MCP's `warn_on_stale_tool_config()`.
pub fn warn_on_stale_builtin_tool_config(filter: &BuiltinToolFilter) {
    let known: HashSet<&str> = BUILTIN_TOOL_NAMES.iter().copied().collect();
    if let Some(allowed) = filter.allowed.as_ref() {
        for name in allowed {
            if !known.contains(name.as_str()) {
                tracing::warn!(
                    "[tools].allowed_tools entry '{}' doesn't match any built-in tool",
                    name
                );
            }
        }
    }
    for name in &filter.disabled {
        if !known.contains(name.as_str()) {
            tracing::warn!(
                "[tools].disabled_tools entry '{}' doesn't match any built-in tool",
                name
            );
        }
    }
    for name in filter.permission_overrides.keys() {
        if !known.contains(name.as_str()) {
            tracing::warn!(
                "[tools.tool_permissions] entry '{}' doesn't match any built-in tool",
                name
            );
        }
    }
}

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
    /// Per-tool overrides from `[tools.tool_permissions]`. Immutable after
    /// construction so the cached system-prompt prefix stays byte-stable
    /// across `/permission` toggles.
    permission_overrides: Arc<HashMap<String, Permission>>,
    /// Built-in allow/block-list. MCP tools have their own per-server
    /// filtering in `src/mcp.rs` and bypass this.
    builtin_filter: Arc<BuiltinToolFilter>,
}

impl ToolRegistry {
    /// Empty registry with the default filter — no built-ins, no MCP
    /// tools. Used by out-of-band CLI commands that spin up a manager
    /// for a single RPC (`agsh mcp reconnect`, `agsh mcp tools`) and
    /// don't need a populated registry.
    pub(crate) fn new() -> Self {
        Self::new_with_filter(BuiltinToolFilter::default())
    }

    fn new_with_filter(filter: BuiltinToolFilter) -> Self {
        let overrides = filter.permission_overrides.clone();
        Self {
            tools: Arc::new(std::sync::RwLock::new(Vec::new())),
            deferred: Arc::new(std::sync::RwLock::new(HashSet::new())),
            permission_overrides: Arc::new(overrides),
            builtin_filter: Arc::new(filter),
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

    /// Replace every tool whose name starts with `mcp__<server_name>__` with
    /// the supplied set. Used by `AgshClientHandler::on_tool_list_changed` to
    /// hot-swap a server's tools without restarting the agent. Deferred
    /// markers for removed tool names are cleared so the registry's deferred
    /// set doesn't grow unbounded.
    pub fn replace_server_tools(&self, server_name: &str, new_tools: Vec<Arc<dyn Tool>>) {
        let prefix = format!("mcp__{}__", server_name);
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

    /// Mark a tool as deferred. Deferred tools live in the registry but are
    /// hidden from the per-turn tools array until the model explicitly
    /// loads them via the `load_tool` meta-tool. Discoverability is
    /// preserved by the `## Tool Discovery` section of the system prompt
    /// (built from `tool_catalogue()`), and the active set is recomputed
    /// per turn from the conversation, not from registry state.
    pub fn mark_deferred(&self, name: &str) {
        self.deferred
            .write()
            .expect("deferred lock poisoned")
            .insert(name.to_string());
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools
            .read()
            .expect("tools lock poisoned")
            .iter()
            .find(|tool| tool.definition().name == name)
            .cloned()
    }

    /// Effective required permission: override wins, else the tool's
    /// hardcoded `Tool::required_permission()`. `None` if not registered.
    pub fn required_permission_for(&self, name: &str) -> Option<Permission> {
        if let Some(permission) = self.permission_overrides.get(name) {
            return Some(*permission);
        }
        self.get(name).map(|tool| tool.required_permission())
    }

    /// Returns tool definitions for the API call, excluding deferred tools.
    /// Permission-filtered view — used by sub-agents which run at a fixed
    /// permission. The main agent uses
    /// [`Self::definitions_active_with_loaded`] so the tools array remains
    /// byte-identical across mid-session `/permission` toggles, keeping
    /// the Claude prompt cache warm on subsequent turns.
    pub fn definitions_for_permission(&self, permission: Permission) -> Vec<ToolDefinition> {
        let deferred = self.deferred.read().expect("deferred lock poisoned");
        self.tools
            .read()
            .expect("tools lock poisoned")
            .iter()
            .filter(|tool| {
                let definition = tool.definition();
                let required = self
                    .permission_overrides
                    .get(&definition.name)
                    .copied()
                    .unwrap_or_else(|| tool.required_permission());
                permission.allows(required) && !deferred.contains(&definition.name)
            })
            .map(|tool| tool.definition())
            .collect()
    }

    /// Slice-based convenience wrapper for tests: composes
    /// [`extract_loaded_tool_names`] with [`Self::definitions_active_with_loaded`].
    /// Production code goes through the events-aware path (see
    /// [`crate::conversation::extract_loaded_tool_names_from_events`]) so
    /// `Event::CompactBoundary::loaded_tools_snapshot` survives across
    /// compaction; a slice-only scan loses the snapshot.
    #[cfg(test)]
    pub fn definitions_active(&self, messages: &[Message]) -> Vec<ToolDefinition> {
        let loaded = extract_loaded_tool_names(messages);
        self.definitions_active_with_loaded(&loaded)
    }

    /// Returns every active tool definition regardless of the caller's
    /// current permission. The active set is the union of non-deferred
    /// tools and deferred tools whose schema has been loaded via the
    /// `load_tool` meta-tool — `loaded` is computed by the caller (via
    /// [`crate::conversation::extract_loaded_tool_names_from_events`]
    /// for the agent loop, [`extract_loaded_tool_names`] for tests).
    ///
    /// Blocked calls are rejected at dispatch; keeping the tools array
    /// permission-independent is what preserves the prompt cache prefix
    /// across `/permission` toggles (breakpoint 3 in the Claude provider's
    /// cache layout).
    pub fn definitions_active_with_loaded(&self, loaded: &HashSet<String>) -> Vec<ToolDefinition> {
        let deferred = self.deferred.read().expect("deferred lock poisoned");
        self.tools
            .read()
            .expect("tools lock poisoned")
            .iter()
            .filter(|tool| {
                let name = tool.definition().name;
                !deferred.contains(&name) || loaded.contains(&name)
            })
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
                let required = self
                    .permission_overrides
                    .get(&def.name)
                    .copied()
                    .unwrap_or_else(|| tool.required_permission());
                (def.name, def.description, required, is_deferred)
            })
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries
    }

    /// Register the core tools shared by the main agent and sub-agents:
    /// file I/O, search, web, and shell execution.
    fn register_core_tools(
        &self,
        web_client_config: &crate::config::WebClientConfig,
        shared_permission: crate::permission::SharedPermission,
        sandbox_enabled: bool,
        sandbox_capability: crate::sandbox::SandboxCapability,
    ) -> Result<()> {
        let read_tracker: ReadTracker = Arc::new(RwLock::new(HashSet::new()));
        self.register_builtin(Arc::new(file::ReadFileTool {
            read_tracker: read_tracker.clone(),
        }));
        self.register_builtin(Arc::new(file::EditFileTool { read_tracker }));
        self.register_builtin(Arc::new(file::WriteFileTool));
        self.register_builtin(Arc::new(find::FindFilesTool));
        self.register_builtin(Arc::new(grep::SearchContentsTool));
        // A malformed proxy URL or unreadable CA file surfaces as a
        // startup error rather than silently falling back to an
        // unconfigured client (which would ignore the user's intent).
        let web_client = web::build_web_client(web_client_config)?;
        self.register_builtin(Arc::new(web::FetchUrlTool {
            client: web_client.clone(),
        }));
        self.register_builtin(Arc::new(web::WebSearchTool { client: web_client }));
        self.register_builtin(Arc::new(shell::ExecuteCommandTool {
            sandbox_capability,
            shared_permission,
            sandbox_enabled,
        }));
        Ok(())
    }

    /// Register a builtin. Collisions panic (programmer error). Tools
    /// rejected by the `[tools]` filter are silently skipped.
    fn register_builtin(&self, tool: Arc<dyn Tool>) {
        let name = tool.definition().name;
        if !self.builtin_filter.admits(&name) {
            tracing::info!(
                "skipping built-in tool '{}' (disabled by [tools] config)",
                name
            );
            return;
        }
        self.register(tool).expect("builtin tool name collision");
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_default(
        web_client_config: crate::config::WebClientConfig,
        shared_permission: crate::permission::SharedPermission,
        sandbox_enabled: bool,
        sandbox_capability: crate::sandbox::SandboxCapability,
        todo_list: todo::SharedTodoList,
        session_manager: SessionManager,
        shared_session_id: Arc<RwLock<Option<Uuid>>>,
        builtin_filter: BuiltinToolFilter,
    ) -> Result<Self> {
        let registry = Self::new_with_filter(builtin_filter);
        registry.register_core_tools(
            &web_client_config,
            shared_permission,
            sandbox_enabled,
            sandbox_capability,
        )?;
        registry.register_builtin(Arc::new(load_tool::LoadToolTool {
            tools: Arc::downgrade(&registry.tools),
            deferred: Arc::downgrade(&registry.deferred),
        }));
        registry.register_builtin(Arc::new(skill::SkillTool {
            session_id: shared_session_id.clone(),
        }));
        registry.register_builtin(Arc::new(render_image::RenderImageTool {
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
        Ok(registry)
    }

    /// Build a tool registry for sub-agents. Excludes `todo_write` (parent
    /// owns task tracking) and `spawn_agent` (no recursive spawning).
    pub fn build_for_subagent(
        web_client_config: crate::config::WebClientConfig,
        shared_permission: crate::permission::SharedPermission,
        sandbox_enabled: bool,
        sandbox_capability: crate::sandbox::SandboxCapability,
        builtin_filter: BuiltinToolFilter,
    ) -> Result<Self> {
        let registry = Self::new_with_filter(builtin_filter);
        registry.register_core_tools(
            &web_client_config,
            shared_permission,
            sandbox_enabled,
            sandbox_capability,
        )?;
        // Sub-agents don't have a session of their own — skills still load but
        // ${AGSH_SESSION_ID} stays unresolved for their invocations.
        registry.register_builtin(Arc::new(skill::SkillTool {
            session_id: Arc::new(RwLock::new(None)),
        }));
        Ok(registry)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn test_shared_permission() -> crate::permission::SharedPermission {
        crate::permission::SharedPermission::new(
            Permission::Write,
            crate::permission::EnabledPermissions::ALL,
        )
    }

    fn test_todo_list() -> todo::SharedTodoList {
        Arc::new(RwLock::new(Vec::new()))
    }

    async fn test_registry() -> ToolRegistry {
        build_test_registry(BuiltinToolFilter::default()).await
    }

    async fn build_test_registry(filter: BuiltinToolFilter) -> ToolRegistry {
        let session_manager = SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("failed to open in-memory database");
        let shared_session_id = Arc::new(RwLock::new(None));
        ToolRegistry::build_default(
            crate::config::WebClientConfig::default(),
            test_shared_permission(),
            true,
            crate::sandbox::detect(),
            test_todo_list(),
            session_manager,
            shared_session_id,
            filter,
        )
        .expect("default web client config should build cleanly")
    }

    use crate::provider::Role;

    fn load_tool_use(id: &str, target: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: id.to_string(),
                name: LOAD_TOOL_NAME.to_string(),
                input: serde_json::json!({ "name": target }),
            }],
        }
    }

    fn tool_result(use_id: &str, body: &str, is_error: bool) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: use_id.to_string(),
                content: vec![ToolResultContent::Text {
                    text: body.to_string(),
                }],
                is_error,
            }],
        }
    }

    #[test]
    fn test_extract_loaded_tool_names_empty() {
        assert!(extract_loaded_tool_names(&[]).is_empty());
    }

    #[test]
    fn test_extract_loaded_tool_names_single_success() {
        let messages = vec![
            load_tool_use("u1", "scratchpad_read"),
            tool_result("u1", "loaded", false),
        ];
        let loaded = extract_loaded_tool_names(&messages);
        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains("scratchpad_read"));
    }

    #[test]
    fn test_extract_loaded_tool_names_error_excluded() {
        let messages = vec![
            load_tool_use("u1", "missing_tool"),
            tool_result("u1", "Error: not registered", true),
        ];
        assert!(extract_loaded_tool_names(&messages).is_empty());
    }

    #[test]
    fn test_extract_loaded_tool_names_orphan_use() {
        // load_tool was issued but the tool_result hasn't arrived yet.
        let messages = vec![load_tool_use("u1", "scratchpad_read")];
        assert!(extract_loaded_tool_names(&messages).is_empty());
    }

    #[test]
    fn test_extract_loaded_tool_names_ignores_other_tools() {
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "u1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({ "name": "anything" }),
                }],
            },
            tool_result("u1", "ok", false),
        ];
        assert!(extract_loaded_tool_names(&messages).is_empty());
    }

    #[test]
    fn test_extract_loaded_tool_names_malformed_input() {
        // load_tool called with no `name` field — must not panic, must not
        // pollute the active set.
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "u1".to_string(),
                    name: LOAD_TOOL_NAME.to_string(),
                    input: serde_json::json!({}),
                }],
            },
            tool_result("u1", "Error", true),
        ];
        assert!(extract_loaded_tool_names(&messages).is_empty());
    }

    #[test]
    fn test_extract_loaded_tool_names_multiple_loads_dedup() {
        let messages = vec![
            load_tool_use("u1", "scratchpad_read"),
            tool_result("u1", "ok", false),
            load_tool_use("u2", "scratchpad_edit"),
            tool_result("u2", "ok", false),
            load_tool_use("u3", "scratchpad_read"),
            tool_result("u3", "already available", false),
        ];
        let loaded = extract_loaded_tool_names(&messages);
        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains("scratchpad_read"));
        assert!(loaded.contains("scratchpad_edit"));
    }

    #[test]
    fn test_extract_loaded_tool_names_multi_block_message() {
        // The model can emit several `tool_use` blocks in one assistant
        // message — the matching `tool_result`s come back as separate
        // blocks of one user message. Both must be processed.
        let assistant = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::ToolUse {
                    id: "u1".to_string(),
                    name: LOAD_TOOL_NAME.to_string(),
                    input: serde_json::json!({"name": "scratchpad_read"}),
                },
                ContentBlock::ToolUse {
                    id: "u2".to_string(),
                    name: LOAD_TOOL_NAME.to_string(),
                    input: serde_json::json!({"name": "scratchpad_edit"}),
                },
            ],
        };
        let user_results = Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "u1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "ok".to_string(),
                    }],
                    is_error: false,
                },
                ContentBlock::ToolResult {
                    tool_use_id: "u2".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "ok".to_string(),
                    }],
                    is_error: false,
                },
            ],
        };
        let loaded = extract_loaded_tool_names(&[assistant, user_results]);
        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains("scratchpad_read"));
        assert!(loaded.contains("scratchpad_edit"));
    }

    #[test]
    fn test_extract_loaded_tool_names_mismatched_id() {
        // tool_result references an id that no `load_tool` use claimed.
        // The result is dropped; the orphan use stays unmatched and is
        // not added to the active set.
        let messages = vec![
            load_tool_use("u1", "scratchpad_read"),
            tool_result("u_other", "ok", false),
        ];
        assert!(extract_loaded_tool_names(&messages).is_empty());
    }

    #[test]
    fn test_extract_loaded_tool_names_interleaved_with_other_tool_calls() {
        // load_tool calls share the message stream with regular tool calls;
        // the scanner must pair on tool_use_id, not on positional adjacency.
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::ToolUse {
                        id: "u1".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({"path": "/tmp/x"}),
                    },
                    ContentBlock::ToolUse {
                        id: "u2".to_string(),
                        name: LOAD_TOOL_NAME.to_string(),
                        input: serde_json::json!({"name": "scratchpad_read"}),
                    },
                ],
            },
            Message {
                role: Role::User,
                content: vec![
                    ContentBlock::ToolResult {
                        tool_use_id: "u1".to_string(),
                        content: vec![ToolResultContent::Text {
                            text: "x contents".to_string(),
                        }],
                        is_error: false,
                    },
                    ContentBlock::ToolResult {
                        tool_use_id: "u2".to_string(),
                        content: vec![ToolResultContent::Text {
                            text: "loaded".to_string(),
                        }],
                        is_error: false,
                    },
                ],
            },
        ];
        let loaded = extract_loaded_tool_names(&messages);
        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains("scratchpad_read"));
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
        assert!(registry.get("load_tool").is_some());
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
        let active = registry.definitions_active(&[]);
        assert!(active.iter().any(|t| t.name == "read_file"));
        assert!(active.iter().any(|t| t.name == "write_file"));
        assert!(active.iter().any(|t| t.name == "edit_file"));
        assert!(active.iter().any(|t| t.name == "execute_command"));
        assert!(!active.iter().any(|t| t.name == "scratchpad_read"));
    }

    #[tokio::test]
    async fn test_definitions_active_stable_across_permissions() {
        let registry = test_registry().await;
        let a = registry.definitions_active(&[]);
        let b = registry.definitions_active(&[]);
        assert_eq!(a.len(), b.len());
        let a_names: Vec<_> = a.iter().map(|t| t.name.clone()).collect();
        let b_names: Vec<_> = b.iter().map(|t| t.name.clone()).collect();
        assert_eq!(a_names, b_names);
    }

    #[tokio::test]
    async fn test_definitions_active_exposes_loaded_deferred_tool() {
        // End-to-end: a successful load_tool call in the conversation
        // promotes the named tool into the active set on the next call.
        let registry = test_registry().await;
        let baseline = registry.definitions_active(&[]);
        assert!(!baseline.iter().any(|t| t.name == "scratchpad_read"));

        let messages = vec![
            load_tool_use("u1", "scratchpad_read"),
            tool_result("u1", "ok", false),
        ];
        let after_load = registry.definitions_active(&messages);
        assert!(after_load.iter().any(|t| t.name == "scratchpad_read"));
        // Append-only: the tools array gains exactly one entry.
        assert_eq!(after_load.len(), baseline.len() + 1);
        // Other deferred scratchpad tools are unaffected.
        assert!(!after_load.iter().any(|t| t.name == "scratchpad_edit"));
    }

    #[tokio::test]
    async fn test_definitions_active_errored_load_stays_hidden() {
        // A load_tool call that ended in an error tool_result must NOT
        // expose the deferred tool — the model's parameter shape was
        // wrong, the schema was not delivered.
        let registry = test_registry().await;
        let messages = vec![
            load_tool_use("u1", "scratchpad_read"),
            tool_result("u1", "Error", true),
        ];
        let active = registry.definitions_active(&messages);
        assert!(!active.iter().any(|t| t.name == "scratchpad_read"));
    }

    #[tokio::test]
    async fn test_definitions_active_load_tool_itself_always_visible() {
        // load_tool is the bootstrap meta-tool — it must appear in the
        // active set for an empty conversation, otherwise the model has
        // no way to discover deferred tools.
        let registry = test_registry().await;
        let active = registry.definitions_active(&[]);
        assert!(active.iter().any(|t| t.name == "load_tool"));
    }

    #[tokio::test]
    async fn test_definitions_active_unknown_load_silently_dropped() {
        // load_tool was called for a tool that isn't registered. The
        // scanner records the (errored) result as not loaded, and even
        // if it were loaded, the registry just doesn't contain a tool
        // by that name — no crash, no spurious entry.
        let registry = test_registry().await;
        let messages = vec![
            load_tool_use("u1", "no_such_tool"),
            tool_result("u1", "Error: not registered", true),
        ];
        let active = registry.definitions_active(&messages);
        assert!(!active.iter().any(|t| t.name == "no_such_tool"));
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

    #[test]
    fn test_builtin_filter_default_admits_everything() {
        let filter = BuiltinToolFilter::default();
        assert!(filter.admits("read_file"));
        assert!(filter.admits("write_file"));
        assert!(filter.admits("anything_else"));
    }

    #[test]
    fn test_builtin_filter_allow_list_restricts() {
        let allowed: HashSet<String> = ["read_file", "find_files"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let filter = BuiltinToolFilter {
            allowed: Some(allowed),
            ..Default::default()
        };
        assert!(filter.admits("read_file"));
        assert!(filter.admits("find_files"));
        assert!(!filter.admits("write_file"));
        assert!(!filter.admits("execute_command"));
    }

    #[test]
    fn test_builtin_filter_block_list_wins_over_allow_list() {
        let allowed: HashSet<String> = ["read_file", "write_file"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let disabled: HashSet<String> = ["write_file"].iter().map(|s| s.to_string()).collect();
        let filter = BuiltinToolFilter {
            allowed: Some(allowed),
            disabled,
            ..Default::default()
        };
        assert!(filter.admits("read_file"));
        assert!(!filter.admits("write_file"));
    }

    #[test]
    fn test_builtin_filter_from_config_empty_allow_list_is_none() {
        let filter = BuiltinToolFilter::from_config(Some(Vec::new()), Vec::new(), HashMap::new());
        assert!(
            filter.allowed.is_none(),
            "empty allow-list should drop to None"
        );
        assert!(filter.admits("read_file"));
    }

    #[tokio::test]
    async fn test_registry_filter_drops_disabled_tools() {
        let filter = BuiltinToolFilter::from_config(
            None,
            vec!["web_search".to_string(), "fetch_url".to_string()],
            HashMap::new(),
        );
        let registry = build_test_registry(filter).await;
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("write_file").is_some());
        assert!(
            registry.get("web_search").is_none(),
            "web_search should be filtered out"
        );
        assert!(
            registry.get("fetch_url").is_none(),
            "fetch_url should be filtered out"
        );
    }

    #[tokio::test]
    async fn test_registry_filter_allow_list_keeps_only_listed() {
        let filter = BuiltinToolFilter::from_config(
            Some(vec!["read_file".to_string(), "find_files".to_string()]),
            Vec::new(),
            HashMap::new(),
        );
        let registry = build_test_registry(filter).await;
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("find_files").is_some());
        assert!(registry.get("write_file").is_none());
        assert!(registry.get("execute_command").is_none());
        assert!(registry.get("web_search").is_none());
    }

    #[tokio::test]
    async fn test_registry_permission_override_applied() {
        let mut overrides = HashMap::new();
        overrides.insert("read_file".to_string(), Permission::Write);
        let filter = BuiltinToolFilter::from_config(None, Vec::new(), overrides);
        let registry = build_test_registry(filter).await;

        // Override wins over the Tool impl's hardcoded `Read`.
        assert_eq!(
            registry.required_permission_for("read_file"),
            Some(Permission::Write)
        );
        // Non-overridden tool returns its hardcoded level.
        assert_eq!(
            registry.required_permission_for("write_file"),
            Some(Permission::Write)
        );
        // Catalogue must reflect the override too (the system prompt
        // reads from it).
        let catalogue = registry.tool_catalogue();
        let read_file_required = catalogue
            .iter()
            .find(|(name, _, _, _)| name == "read_file")
            .map(|(_, _, perm, _)| *perm);
        assert_eq!(read_file_required, Some(Permission::Write));
    }

    #[tokio::test]
    async fn test_registry_permission_override_excludes_tool_from_lower_level() {
        let mut overrides = HashMap::new();
        overrides.insert("read_file".to_string(), Permission::Write);
        let filter = BuiltinToolFilter::from_config(None, Vec::new(), overrides);
        let registry = build_test_registry(filter).await;

        // At Read permission, read_file should now be excluded from the
        // permission-filtered definitions because the override raised it
        // to Write.
        let read_defs = registry.definitions_for_permission(Permission::Read);
        assert!(!read_defs.iter().any(|t| t.name == "read_file"));

        let write_defs = registry.definitions_for_permission(Permission::Write);
        assert!(write_defs.iter().any(|t| t.name == "read_file"));
    }

    #[tokio::test]
    async fn test_subagent_registry_honours_filter() {
        let filter =
            BuiltinToolFilter::from_config(None, vec!["web_search".to_string()], HashMap::new());
        let registry = ToolRegistry::build_for_subagent(
            crate::config::WebClientConfig::default(),
            crate::permission::SharedPermission::new(
                Permission::Read,
                crate::permission::EnabledPermissions::ALL,
            ),
            true,
            crate::sandbox::detect(),
            filter,
        )
        .expect("default web client config should build cleanly");
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("web_search").is_none());
    }

    #[test]
    fn test_builtin_tool_names_covers_canonical_set() {
        // Guard against forgetting to add a new built-in to the canonical
        // list that drives stale-entry warnings. Update this assertion
        // deliberately when adding a tool in register_core_tools.
        let names: HashSet<&str> = BUILTIN_TOOL_NAMES.iter().copied().collect();
        for expected in &[
            "read_file",
            "write_file",
            "edit_file",
            "find_files",
            "search_contents",
            "execute_command",
            "fetch_url",
            "web_search",
            "todo_write",
            "scratchpad_read",
            "scratchpad_write",
            "scratchpad_edit",
            "scratchpad_list",
            "scratchpad_delete",
            "skill",
            "render_image",
            "spawn_agent",
            "load_tool",
        ] {
            assert!(
                names.contains(expected),
                "BUILTIN_TOOL_NAMES missing '{}'",
                expected
            );
        }
    }
}
