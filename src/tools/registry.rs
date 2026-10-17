//! Which tools a session has: the registry, the built-in filter and denials, deferred loading, and
//! the builders for each kind of agent.

use super::*;
use crate::{
    config::BuiltinToolFilter,
    session::{CoreMaterials, SessionCells, SessionMaterials},
};

pub(super) type DeferredSet = Arc<std::sync::RwLock<HashSet<String>>>;

#[cfg(test)]
impl ToolRegistry {
    /// Register a [`crate::tools::FixtureDeferredTool`] under `name` and defer it.
    pub(crate) fn register_deferred_fixture(&self, name: &str) {
        self.register(Arc::new(crate::tools::FixtureDeferredTool {
            name: name.to_string(),
        }))
        .expect("fixture tool name must be unique");
        self.mark_deferred(name);
    }
}
use crate::provider::DeferredToolDelivery;
pub(crate) use crate::provider::{LOAD_TOOL_NAME, SEARCH_TOOL_NAME, is_discovery_tool};
/// Most tools one `tool_load` call will render. A batch past this is more likely a model loading a
/// whole server speculatively than a task that genuinely needs them all, and each schema is
/// unbounded in size.
pub(crate) const MAX_LOAD_TOOL_BATCH: usize = 10;
/// Most matches one `tool_search` call renders: one `tool_load` batch, so everything a search
/// turns up can be loaded in the next call.
pub(crate) const TOOL_SEARCH_MAX_RESULTS: usize = MAX_LOAD_TOOL_BATCH;
/// The tool names one `tool_load` call refers to, in order, deduplicated, and capped at
/// [`MAX_LOAD_TOOL_BATCH`].
///
/// Accepts a bare string or an array of strings: a task needing three tools off one server should
/// cost one round trip, not three. Shared with the scanners that recover the active set, so it is
/// derived from exactly the names the tool acted on, cap included. A name dropped by the cap must
/// not become active, since its schema was never rendered.
pub(crate) fn load_tool_names(input: &serde_json::Value) -> Vec<String> {
    let mut names = requested_tool_names(input);
    names.truncate(MAX_LOAD_TOOL_BATCH);
    names
}
/// Every distinct name the call asked for, before [`MAX_LOAD_TOOL_BATCH`] is applied. `tool_load`
/// compares this against [`load_tool_names`] so an over-long batch is reported rather than quietly
/// half-honored, which is the same class of silent shortfall this whole advisory machinery exists
/// to eliminate.
pub(crate) fn requested_tool_names(input: &serde_json::Value) -> Vec<String> {
    let mut names: Vec<String> = match input.get("name") {
        Some(serde_json::Value::String(name)) => vec![name.clone()],
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|item| item.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    };
    let mut seen = HashSet::new();
    names.retain(|name| seen.insert(name.clone()));
    names
}
/// Walk the conversation and collect the names of tools that have been loaded via successful
/// `tool_load` calls. A `tool_load` `tool_use` block counts only when paired with a non-error
/// `tool_result` whose `tool_use_id` matches; this excludes errored loads (unknown name, malformed
/// args) and orphan `tool_use` blocks awaiting their result.
///
/// A batch that resolved some names and not others returns a non-error result, so every name it
/// carried is recorded here. Harmless: a name with no registry entry matches nothing when the
/// active set is assembled (see [`ToolRegistry::active_tools`]).
///
/// Test-only, because the answer it gives is not the one production wants: a materialized slice
/// shows only the `tool_load` exchanges still standing in the current view, and a compaction or a
/// `DegradeTier::ToolExchanges` repair takes them out of it.
/// [`crate::tools::load_tool::extract_loaded_tool_names_from_events`] reads the log instead. The
/// `#[cfg(test)]` is what stops this drifting back into a live path.
#[cfg(test)]
pub(crate) fn extract_loaded_tool_names(messages: &[Message]) -> HashSet<String> {
    let mut pending: HashMap<String, Vec<String>> = HashMap::new();
    let mut loaded: HashSet<String> = HashSet::new();
    for message in messages {
        for block in &message.content {
            match block {
                ContentBlock::ToolUse { id, name, input } if name == LOAD_TOOL_NAME => {
                    let names = load_tool_names(input);
                    if !names.is_empty() {
                        pending.insert(id.clone(), names);
                    }
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    is_error,
                    ..
                } => {
                    if let Some(loaded_names) = pending.remove(tool_use_id)
                        && !is_error
                    {
                        loaded.extend(loaded_names);
                    }
                }
                _ => {}
            }
        }
    }
    loaded
}
/// What a sub-agent registry may hold: `[subagents]` narrowed by the `allow_servers`,
/// `allow_tools`, `deny_servers` and `deny_tools` of the `agent_spawn` call that built it.
/// Unrestricted on the primary agent's registry, which is what makes this a sub-agent concept
/// rather than a second `[tools]` filter.
///
/// Both kinds of list only narrow, and a denial wins on overlap, the rule `[tools]` and each
/// `[[mcp.servers]]` already apply: an allow-list is exhaustive over what it names, so nothing
/// outside it appears, and inside it anything a deny names is removed too. A scope only ever
/// narrows ([`Self::narrow`]): denials accumulate and allow-lists intersect, so no depth of nesting
/// can hand a descendant something config or an ancestor took away.
#[derive(Debug, Clone, Default)]
pub(crate) struct ToolScope {
    /// The MCP servers a worker may see at all, or `None` for every configured one.
    pub(super) allowed_servers: Option<HashSet<String>>,
    /// The tools a worker may see, built-ins and namespaced MCP tools in one namespace, or `None`
    /// for every one its registry would otherwise hold.
    pub(super) allowed_tools: Option<HashSet<String>>,
    pub(super) denied_servers: HashSet<String>,
    pub(super) denied_tools: HashSet<String>,
}
impl ToolScope {
    /// A scope that denies `servers` and `tools` and allows everything else.
    pub(crate) fn new(
        servers: impl IntoIterator<Item = String>,
        tools: impl IntoIterator<Item = String>,
    ) -> Self {
        Self {
            allowed_servers: None,
            allowed_tools: None,
            denied_servers: servers.into_iter().collect(),
            denied_tools: tools.into_iter().collect(),
        }
    }

    /// This scope with the allow-lists set: `None` leaves a side unrestricted.
    pub(crate) fn allowing(
        mut self,
        servers: Option<impl IntoIterator<Item = String>>,
        tools: Option<impl IntoIterator<Item = String>>,
    ) -> Self {
        self.allowed_servers = servers.map(|list| list.into_iter().collect());
        self.allowed_tools = tools.map(|list| list.into_iter().collect());
        self
    }

    /// Whether this scope admits everything, so a caller can skip filtering altogether.
    pub(crate) fn is_unrestricted(&self) -> bool {
        self.allowed_servers.is_none()
            && self.allowed_tools.is_none()
            && self.denied_servers.is_empty()
            && self.denied_tools.is_empty()
    }

    /// Whether a worker may see `server_name` at all: its tools, its resources and its prompts.
    pub(crate) fn admits_server(&self, server_name: &str) -> bool {
        if self.denied_servers.contains(server_name) {
            return false;
        }
        self.allowed_servers
            .as_ref()
            .is_none_or(|servers| servers.contains(server_name))
    }

    /// Whether a worker may see the tool `name`: not denied by its own registry name, within the
    /// tool allow-list when there is one, and, for a namespaced MCP tool, from a server the scope
    /// admits. The server arm is what makes `deny_servers` and `allow_servers` cover tools
    /// registered through paths that never see a server name, so the lists cannot disagree.
    pub(crate) fn admits_tool(&self, name: &str) -> bool {
        if self.denied_tools.contains(name) {
            return false;
        }
        if let Some(server) = server_of_tool(name)
            && !self.admits_server(server)
        {
            return false;
        }
        self.allowed_tools
            .as_ref()
            .is_none_or(|tools| tools.contains(name))
    }

    /// This scope narrowed by `other`: everything either denies is denied, and what both allow is
    /// allowed, an absent allow-list being everything. Used when an `agent_spawn` call narrows what
    /// config and its own ancestors already set, and again when the child's own `agent_spawn`
    /// inherits the result. Never wider than either side.
    pub(crate) fn narrow(&self, other: &Self) -> Self {
        fn intersect(
            left: &Option<HashSet<String>>,
            right: &Option<HashSet<String>>,
        ) -> Option<HashSet<String>> {
            match (left, right) {
                (Some(left), Some(right)) => Some(left.intersection(right).cloned().collect()),
                (Some(only), None) | (None, Some(only)) => Some(only.clone()),
                (None, None) => None,
            }
        }
        Self {
            allowed_servers: intersect(&self.allowed_servers, &other.allowed_servers),
            allowed_tools: intersect(&self.allowed_tools, &other.allowed_tools),
            denied_servers: self
                .denied_servers
                .union(&other.denied_servers)
                .cloned()
                .collect(),
            denied_tools: self
                .denied_tools
                .union(&other.denied_tools)
                .cloned()
                .collect(),
        }
    }

    /// Sorted, for the persisted spawn spec and for tests. `HashSet` iteration order is not stable,
    /// and a spec that round-trips differently every time is one nobody can diff.
    pub(crate) fn denied_server_list(&self) -> Vec<String> {
        sorted(&self.denied_servers)
    }

    pub(crate) fn denied_tool_list(&self) -> Vec<String> {
        sorted(&self.denied_tools)
    }

    /// The allow-lists, sorted, or `None` for a side that is unrestricted.
    pub(crate) fn allowed_server_list(&self) -> Option<Vec<String>> {
        self.allowed_servers.as_ref().map(sorted)
    }

    pub(crate) fn allowed_tool_list(&self) -> Option<Vec<String>> {
        self.allowed_tools.as_ref().map(sorted)
    }
}
fn sorted(names: &HashSet<String>) -> Vec<String> {
    let mut names: Vec<String> = names.iter().cloned().collect();
    names.sort();
    names
}
/// Which list of a [`ToolScope`] an entry came from, for a report that names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ScopeList {
    AllowedServers,
    AllowedTools,
    DeniedServers,
    DeniedTools,
}
/// The entries of `scope` that match nothing, each with the list it came from: a server name no
/// `[[mcp.servers]]` configures, and a tool name that is neither a built-in nor namespaced under
/// a configured server. One rule for the config lists and the `agent_spawn` lists, which report
/// under their own key names. An entry that matches nothing restricts nothing, and the reader gets
/// no signal either way: they asked for a narrowed worker and would get a wider one believing
/// otherwise.
///
/// Checked against configured server names rather than advertised tool names, because the config
/// pass runs at startup before any server has completed its handshake. So
/// `mcp__notion__craete_page` with `notion` configured is accepted here and never warned about:
/// nothing else checks it either (`mcp::warn_on_stale_tool_config` only inspects the per-server
/// `[[mcp.servers]]` lists, against raw un-namespaced names). Catching it would mean deferring this
/// pass until every server has connected, which is a different shape of startup. Server names,
/// which are the coarse lever and the more consequential half, are checked exactly.
pub(crate) fn stale_scope_entries(
    scope: &ToolScope,
    configured_servers: &[String],
) -> Vec<(ScopeList, String)> {
    let servers: HashSet<&str> = configured_servers.iter().map(String::as_str).collect();
    let known: HashSet<&str> = BUILTIN_TOOL_NAMES.iter().copied().collect();
    let tool_is_stale = |name: &str| {
        !known.contains(name)
            && !server_of_tool(name).is_some_and(|server| servers.contains(server))
    };
    let mut stale = Vec::new();
    for (list, names) in [
        (ScopeList::AllowedServers, scope.allowed_server_list()),
        (ScopeList::DeniedServers, Some(scope.denied_server_list())),
    ] {
        for name in names.unwrap_or_default() {
            if !servers.contains(name.as_str()) {
                stale.push((list, name));
            }
        }
    }
    for (list, names) in [
        (ScopeList::AllowedTools, scope.allowed_tool_list()),
        (ScopeList::DeniedTools, Some(scope.denied_tool_list())),
    ] {
        for name in names.unwrap_or_default() {
            if tool_is_stale(&name) {
                stale.push((list, name));
            }
        }
    }
    stale
}
/// How the wire of the session that published `profile` carries deferred tools. A cell no session
/// published to (a test's, a gate probe's) appends, like a wire with no declaration of its own.
pub(super) fn wire_delivery(
    profile: &std::sync::OnceLock<crate::provider::PublishedProfile>,
) -> DeferredToolDelivery {
    profile
        .get()
        .map(|profile| profile.current().provider.deferred_tool_delivery())
        .unwrap_or(DeferredToolDelivery::Appended)
}
/// Whether the wire of the session that published `profile` carries the tool `name`. A backend
/// may refuse a name outright, and a session can move onto one mid-conversation, so the tool stays
/// registered and every door that would carry, show, load or run it asks here, per request. A
/// cell no session published to carries everything.
pub(super) fn is_carried_by_wire(
    profile: &std::sync::OnceLock<crate::provider::PublishedProfile>,
    name: &str,
) -> bool {
    !profile
        .get()
        .is_some_and(|profile| profile.current().provider.refuses_tool_name(name))
}
/// The one sentence a load or a call of a tool the wire does not carry is answered with.
pub(crate) fn not_carried_refusal(name: &str) -> String {
    format!("Tool '{name}' is not available on this profile's backend.")
}
/// Server name out of a namespaced MCP tool name, or `None` for a built-in. Server names cannot
/// contain `__` (`mcp::sanitize::normalize_server_name`), so the first occurrence splits them.
pub(crate) fn server_of_tool(tool_name: &str) -> Option<&str> {
    let rest = tool_name.strip_prefix("mcp__")?;
    rest.split_once("__").map(|(server, _tool)| server)
}
/// The MCP resource and prompt meta-tools, which `mcp_resources::register_all` registers directly
/// rather than through `register_builtin`. Deniable via `disabled_tools`, but deliberately outside
/// `allowed_tools`; see [`BuiltinToolFilter::denies`].
pub(crate) const MCP_META_TOOL_NAMES: &[&str] = &[
    "mcp_prompt_get",
    "mcp_prompt_list",
    "mcp_resource_list",
    "mcp_resource_read",
    "mcp_resource_subscribe",
    "mcp_resource_unsubscribe",
    "mcp_resource_updates_list",
];
/// Canonical built-in names for the stale-entry warning pass, sorted.
///
/// Every name a user may legitimately put in `[tools]` or `[subagents]`, which is wider than
/// [`ToolRegistry::build_default`]'s own list: the conditionally-registered families
/// (`schedule_*`, `task_*`) and the tools registered outside `register_builtin` (the MCP
/// meta-tools) are all deniable, so leaving them out made the warning fire on correct entries.
/// Update when adding any new built-in.
pub(crate) const BUILTIN_TOOL_NAMES: &[&str] = &[
    "agent_delete",
    "agent_followup",
    "agent_list",
    "agent_spawn",
    "agent_steer",
    "checklist_add",
    "checklist_edit",
    "checklist_read",
    "context_check",
    "context_compact",
    "conversation_read",
    "conversation_search",
    "file_edit",
    "file_find",
    "file_read",
    "file_search",
    "file_write",
    "image_render",
    "mcp_prompt_get",
    "mcp_prompt_list",
    "mcp_resource_list",
    "mcp_resource_read",
    "mcp_resource_subscribe",
    "mcp_resource_unsubscribe",
    "mcp_resource_updates_list",
    "memory_delete",
    "memory_read",
    "memory_search",
    "memory_write",
    "schedule_cancel",
    "schedule_create",
    "schedule_list",
    "scratchpad_delete",
    "scratchpad_edit",
    "scratchpad_list",
    "scratchpad_load_file",
    "scratchpad_merge",
    "scratchpad_read",
    "scratchpad_rename",
    "scratchpad_save_file",
    "scratchpad_write",
    "session_list",
    "session_search",
    "shell_execute",
    "skill_delete",
    "skill_read",
    "skill_search",
    "skill_write",
    "task_cancel",
    "task_list",
    "tool_load",
    "tool_search",
    "web_fetch",
];
/// Tools a checkpoint turn may reach, on top of the `context_replace` it is given.
///
/// An allow-list, because the guiding rule is that **a checkpoint can save, but not act**: the turn
/// exists to preserve what already happened, not to do more work while the window is about to be
/// rewritten. So no `shell_execute`, no `file_write` / `file_edit`, no `agent_spawn`, no
/// `schedule_*`, and no MCP tools.
///
/// The read tools *are* here, because deciding what is worth keeping sometimes means checking
/// something first. The two delete tools are not: deleting is not saving, and a mistaken
/// `memory_delete` in an unattended checkpoint is unrecoverable, while the agent can still delete
/// on any ordinary turn. Nor are the checklist edits: a checkpoint's calls are not recorded, and
/// the list is what the recorded calls add up to, so an edit made here would show in the summary
/// and be undone by the next reading of the log.
///
/// Kept sorted, and every entry must exist in [`BUILTIN_TOOL_NAMES`]; a test enforces both.
pub(crate) const CHECKPOINT_TOOL_NAMES: &[&str] = &[
    "checklist_read",
    "conversation_read",
    "conversation_search",
    "file_find",
    "file_read",
    "file_search",
    "memory_read",
    "memory_search",
    "memory_write",
    "scratchpad_edit",
    "scratchpad_list",
    "scratchpad_read",
    "scratchpad_write",
];
/// Warn (never fail) on `[tools]` entries that don't match any known built-in. Mirrors MCP's
/// `warn_on_stale_tool_config()`.
pub(crate) fn warn_on_stale_builtin_tool_config(filter: &BuiltinToolFilter) {
    let known: HashSet<&str> = BUILTIN_TOOL_NAMES.iter().copied().collect();
    if let Some(allowed) = filter.allowed.as_ref() {
        for name in allowed {
            if !known.contains(name.as_str()) {
                let hint = builtin_name_hint(name);
                tracing::warn!(
                    "`[tools].allowed_tools` entry '{name}' matches no built-in tool.{hint}"
                );
            } else if MCP_META_TOOL_NAMES.contains(&name.as_str()) {
                // These register outside the allow-list (see
                // `ToolRegistry::admits_infrastructure`), so naming one here is inert while
                // looking like it keeps the tool.
                tracing::warn!(
                    "`[tools].allowed_tools` entry '{name}' has no effect; name it in \
                     `[tools].disabled_tools` to remove it"
                );
            }
        }
    }
    for name in &filter.disabled {
        if !known.contains(name.as_str()) {
            let hint = builtin_name_hint(name);
            tracing::warn!(
                "`[tools].disabled_tools` entry '{name}' matches no built-in tool.{hint}"
            );
        }
    }
    for name in filter.permission_overrides.keys() {
        if !known.contains(name.as_str()) {
            let hint = builtin_name_hint(name);
            tracing::warn!(
                "`[tools.tool_permissions]` entry '{name}' matches no built-in tool.{hint}"
            );
        }
    }
}
/// Suggest a built-in for a `[tools]` entry that matches none.
///
/// The same service `did_you_mean_hint` does for the model, for the other reader of these names.
/// A rename leaves stale entries in config files, and a bare "matches nothing" makes a `[tools]`
/// block that silently stopped applying look like a typo the user has to hunt for.
pub(super) fn builtin_name_hint(name: &str) -> String {
    did_you_mean_hint(name, BUILTIN_TOOL_NAMES.iter().copied())
}
/// Warn (never fail) on `[subagents]` entries that match nothing; see [`stale_scope_entries`]
/// for the rule and why. A typo here denies nothing at all while reading as a restriction, which
/// is the worst of both: the user believes a sub-agent cannot reach a server it can reach.
pub(crate) fn warn_on_stale_subagent_config(scope: &ToolScope, configured_servers: &[String]) {
    for (list, name) in stale_scope_entries(scope, configured_servers) {
        match list {
            ScopeList::DeniedServers => tracing::warn!(
                "`[subagents].disabled_servers` entry '{name}' matches no configured MCP server"
            ),
            ScopeList::DeniedTools => tracing::warn!(
                "`[subagents].disabled_tools` entry '{name}' matches no built-in tool and no \
                 configured MCP server"
            ),
            // Config has no allow-lists for sub-agents; the call site does.
            ScopeList::AllowedServers | ScopeList::AllowedTools => {}
        }
    }
}
pub(super) type ToolSet = Arc<std::sync::RwLock<Registered>>;
/// Tool registry. Backed by an `Arc<RwLock<Vec<Arc<dyn Tool>>>>` so MCP notification handlers can
/// swap a server's tools in place on `tools/list_changed`. Individual registrations only hold the
/// write lock briefly; dispatch clones the matching `Arc<dyn Tool>` out of the lock before awaiting
/// `execute`, so no lock is held across `.await`.
#[derive(Clone)]
pub(crate) struct ToolRegistry {
    pub(super) tools: ToolSet,
    pub(super) deferred: DeferredSet,
    /// Per-tool overrides from `[tools.tool_permissions]`. Immutable after construction so the
    /// cached system-prompt prefix stays byte-stable across `/permission` toggles.
    pub(super) permission_overrides: Arc<HashMap<String, Permission>>,
    /// The scope built during `register_core_tools`, so the session-scoped pass can hand the same
    /// one to `scratchpad_save_file` without re-deriving it from a different set of inputs.
    pub(super) write_scope: Arc<std::sync::RwLock<Option<crate::workspace::WriteScope>>>,
    /// Built-in allow/block-list. MCP tools have their own per-server filtering in `src/mcp.rs`
    /// and bypass this.
    pub(super) builtin_filter: Arc<BuiltinToolFilter>,
    /// The sub-agent's tool scope, unrestricted on the root agent's registry. Unlike
    /// `builtin_filter` this covers MCP tools too, and is read back out by
    /// [`crate::tools::mcp_adapter::install_on_worker_registry`] and
    /// [`mcp_resources::register_all`] so the registry stays the single place the answer lives.
    pub(super) tool_scope: Arc<ToolScope>,
    /// Files read this session, shared with the file tools so `file_edit` can require a prior
    /// read. Cleared on conversation compaction.
    pub(super) read_tracker: ReadTracker,
    /// Back-reference to the MCP manager, filled in by
    /// [`crate::tools::mcp_adapter::attach_session_registry`] for session registries and
    /// [`crate::tools::mcp_adapter::install_on_worker_registry`] for sub-agent ones. Both
    /// `tool_load` and `Agent::resolve_and_execute_tool` read it to distinguish "no such tool"
    /// from "that tool's server isn't connected". `Weak` because the manager owns the registry
    /// list, not the other way round.
    pub(super) mcp_manager: Arc<std::sync::OnceLock<std::sync::Weak<crate::mcp::McpClientManager>>>,
    /// Whether the `background` property is spliced into the schemas this registry hands the
    /// provider. Off leaves it out entirely rather than refusing it at dispatch: a parameter the
    /// model can see but cannot use is worse than one it cannot see.
    pub(super) background_enabled: Arc<std::sync::atomic::AtomicBool>,
    /// The session's published profile, whose provider says how the wire carries deferred tools.
    /// Read per request rather than copied, so a profile switch is one store into the cell and
    /// nothing to keep in step. Unset on a registry no session published to (a test's, a gate
    /// probe's), which appends.
    pub(super) profile: Arc<std::sync::OnceLock<crate::provider::PublishedProfile>>,
    /// Scratchpad entries a parent lent this session read-only, recorded by the session-scoped
    /// pass so the turn's universal `scratchpad` parameter refuses them like the tools do. Empty
    /// on the root agent's registry.
    pub(super) inherited_scratchpad_names: Arc<std::sync::RwLock<Vec<String>>>,
}
/// The registered tools in registration order, each with the definition it answered when it was
/// registered, indexed by name.
///
/// The order is kept because the provider sees it: a stable tools array is what a cache prefix
/// is. The index is what makes a lookup by name one hash rather than a walk, and the stored
/// definition is what makes handing one out a clone rather than a rebuild: a built-in constructs
/// its schema from `json!` on every `definition()` call, and a registry of a few hundred MCP tools
/// would answer a name lookup by constructing all of them. The estimate beside each is the
/// schema's token cost, summed per round for the context gauge, so a round pays one addition per
/// active tool rather than one serialization.
#[derive(Default)]
pub(super) struct Registered {
    entries: Vec<RegisteredTool>,
    by_name: HashMap<String, usize>,
}

/// One registered tool: the tool, the definition it gave, and what that definition costs.
pub(super) struct RegisteredTool {
    pub(super) tool: Arc<dyn Tool>,
    pub(super) definition: ToolDefinition,
    pub(super) schema_tokens: u64,
}

impl Registered {
    pub(super) fn contains(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    pub(super) fn get(&self, name: &str) -> Option<&RegisteredTool> {
        self.by_name
            .get(name)
            .and_then(|&index| self.entries.get(index))
    }

    /// Append `tool`, which the caller has checked is not registered under its name.
    pub(super) fn push(&mut self, tool: Arc<dyn Tool>) {
        let definition = tool.definition();
        let schema_tokens = crate::tokens::estimate_text(&definition.name)
            .saturating_add(crate::tokens::estimate_text(&definition.description))
            .saturating_add(crate::tokens::estimate_text(
                &definition.parameters.to_string(),
            ));
        self.by_name
            .insert(definition.name.clone(), self.entries.len());
        self.entries.push(RegisteredTool {
            tool,
            definition,
            schema_tokens,
        });
    }

    /// Drop every tool `keep` refuses, keeping the order of the rest.
    pub(super) fn retain(&mut self, mut keep: impl FnMut(&ToolDefinition) -> bool) {
        self.entries.retain(|entry| keep(&entry.definition));
        self.by_name = self
            .entries
            .iter()
            .enumerate()
            .map(|(index, entry)| (entry.definition.name.clone(), index))
            .collect();
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &RegisteredTool> {
        self.entries.iter()
    }

    pub(super) fn names(&self) -> impl Iterator<Item = &str> {
        self.entries
            .iter()
            .map(|entry| entry.definition.name.as_str())
    }

    pub(super) fn tools(&self) -> impl Iterator<Item = &Arc<dyn Tool>> {
        self.entries.iter().map(|entry| &entry.tool)
    }
}

/// The tools a request carries, and what their schemas cost the window.
pub(crate) struct ActiveTools {
    pub(crate) definitions: Vec<ToolDefinition>,
    /// The deferred tools a declaring wire lists beside `definitions` with `defer_loading`; empty
    /// on a wire that appends a loaded tool to `definitions` instead.
    pub(crate) deferred_definitions: Vec<ToolDefinition>,
    pub(crate) schema_tokens: u64,
}

/// The registry's tools as one request carries them; see [`ToolRegistry::carried`].
struct Carried<'a> {
    delivery: DeferredToolDelivery,
    /// Every carried tool that is not deferred, in registration order, the two discovery tools
    /// among them whether or not they ride.
    active: Vec<&'a RegisteredTool>,
    /// Every carried deferred tool, in registration order.
    deferred: Vec<&'a RegisteredTool>,
    /// Whether the two discovery tools ride as functions: only while something is deferred for
    /// them to find, and never on a wire whose own search item does both jobs.
    discovery_rides: bool,
    /// Whether the deferred tools are declared outright, schemas and all, because nothing else
    /// rides. A declaring endpoint refuses an array that is deferred through, and an appending
    /// wire would leave such a scope (a sub-agent allow-listed to one MCP tool, with no
    /// `tool_load` in it) nothing it can call; with nothing active there is no prefix to keep
    /// stable either way.
    declared_outright: bool,
}

impl Carried<'_> {
    /// Whether `entry` is in the array: every active tool but a discovery tool that does not ride.
    fn rides(&self, entry: &RegisteredTool) -> bool {
        self.discovery_rides || !is_discovery_tool(&entry.definition.name)
    }
}

impl ToolRegistry {
    /// Empty registry with the default filter: no built-ins, no MCP tools. Used by out-of-band CLI
    /// commands that spin up a manager for a single RPC (`meka mcp reconnect`, `meka mcp tools`)
    /// and don't need a populated registry.
    pub(crate) fn new() -> Self {
        Self::new_with_filter(BuiltinToolFilter::default())
    }

    pub(super) fn new_with_filter(filter: BuiltinToolFilter) -> Self {
        Self::new_with_filter_and_scope(filter, ToolScope::default())
    }

    pub(super) fn new_with_filter_and_scope(
        filter: BuiltinToolFilter,
        tool_scope: ToolScope,
    ) -> Self {
        let overrides = filter.permission_overrides.clone();
        Self {
            tools: Arc::new(std::sync::RwLock::new(Registered::default())),
            deferred: Arc::new(std::sync::RwLock::new(HashSet::new())),
            permission_overrides: Arc::new(overrides),
            builtin_filter: Arc::new(filter),
            tool_scope: Arc::new(tool_scope),
            mcp_manager: Arc::new(std::sync::OnceLock::new()),
            background_enabled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            profile: Arc::new(std::sync::OnceLock::new()),
            read_tracker: Arc::new(RwLock::new(HashMap::new())),
            write_scope: Arc::new(std::sync::RwLock::new(None)),
            inherited_scratchpad_names: Arc::new(std::sync::RwLock::new(Vec::new())),
        }
    }

    /// The scratchpad entries a parent lent this session read-only.
    pub(crate) fn inherited_scratchpad_names(&self) -> Vec<String> {
        crate::sync::read(&self.inherited_scratchpad_names).clone()
    }

    /// The scope `register_core_tools` built, or a deny-all one.
    ///
    /// `None` only on a registry whose core tools were never registered. Both production builders
    /// register them first and only then run the session-scoped pass, so the fallback is
    /// unreachable there; it fails closed so that a registry which cannot tell whether a boundary
    /// applies refuses rather than assumes.
    pub(super) fn write_scope_or_deny_all(&self) -> crate::workspace::WriteScope {
        crate::sync::read(&self.write_scope)
            .clone()
            .unwrap_or_else(crate::workspace::WriteScope::deny_all)
    }

    /// What this registry refuses to register. Read by the MCP installation paths so they can skip
    /// a withheld server's discovery round trip and its resource/prompt meta-tools, rather than
    /// each caller threading its own copy of the lists.
    pub(crate) fn tool_scope(&self) -> &ToolScope {
        &self.tool_scope
    }

    /// Whether this registry will accept a tool by that name: it has to pass both the `[tools]`
    /// filter and the sub-agent's tool scope.
    ///
    /// The single predicate behind [`Self::register_builtin`] and the two paths that call
    /// [`Self::register`] directly ([`mcp_resources::register_all`] and
    /// [`subagent::register_subagent_tools`]), whose tools are built from collaborators the generic
    /// builder does not have; routing them past `register_builtin` would route them past the
    /// filters with it. Any future direct registration should come through here too.
    pub(crate) fn admits(&self, name: &str) -> bool {
        if !self.builtin_filter.admits(name) {
            tracing::info!("skipping tool '{name}': excluded by `[tools]`");
            return false;
        }
        self.not_denied(name)
    }

    /// Like [`Self::admits`] but consulting only `[tools].disabled_tools`, not `allowed_tools`.
    ///
    /// For the MCP meta-tools, which an exhaustive `allowed_tools` does not reach. See
    /// [`BuiltinToolFilter::denies`] for why widening it now would break working configs.
    pub(crate) fn admits_infrastructure(&self, name: &str) -> bool {
        if self.builtin_filter.denies(name) {
            tracing::info!("skipping tool '{name}': disabled by `[tools].disabled_tools`");
            return false;
        }
        self.not_denied(name)
    }

    pub(super) fn not_denied(&self, name: &str) -> bool {
        if !self.tool_scope.admits_tool(name) {
            tracing::info!("skipping tool '{name}' for sub-agent: outside its tool scope");
            return false;
        }
        true
    }

    /// Clear the read-tracker. Called when the conversation is rewritten (a compaction, a
    /// rewind): the model's context is reset, so a follow-up `file_edit` should re-read the file
    /// rather than trust a read the model no longer holds.
    pub(crate) async fn clear_read_tracker(&self) {
        self.read_tracker.write().await.clear();
    }

    /// How many files the tracker holds a read of.
    #[cfg(test)]
    pub(crate) async fn tracked_reads_for_test(&self) -> usize {
        self.read_tracker.read().await.len()
    }

    /// Record a read of `path` as `file_read` would, for a test of what forgets it.
    #[cfg(test)]
    pub(crate) async fn record_read_for_test(&self, path: PathBuf) {
        self.read_tracker
            .write()
            .await
            .insert(path, crate::tools::ReadStamp::of_delegated(""));
    }

    /// Every clone of one registry reports the same number, and no live registry shares it with
    /// another. `crate::tools::mcp_adapter` unsubscribes a session's registry by it.
    pub(crate) fn inner_identity(&self) -> usize {
        Arc::as_ptr(&self.tools) as usize
    }

    /// Register a tool. Returns an error if another tool with the same name is already registered.
    pub(crate) fn register(&self, tool: Arc<dyn Tool>) -> Result<()> {
        let name = tool.definition().name;
        let mut tools = crate::sync::write(&self.tools);
        if tools.contains(&name) {
            return Err(crate::error::MekaError::ToolRegistration {
                message: format!("tool name '{name}' is already registered"),
            });
        }
        tools.push(tool);
        drop(tools);
        Ok(())
    }

    /// Replace every tool whose name starts with `mcp__<server_name>__` with the supplied set. Used
    /// by `MekaClientHandler::on_tool_list_changed` to hot-swap a server's tools without restarting
    /// the agent. Deferred markers for removed tool names are cleared so the registry's deferred
    /// set doesn't grow unbounded.
    pub(crate) fn replace_server_tools(&self, server_name: &str, new_tools: Vec<Arc<dyn Tool>>) {
        // Filter here rather than only at the call sites: a `tools/list_changed` notification
        // arriving mid-run goes through this same path, so a denied server would otherwise walk its
        // tools back in the moment it re-advertised them.
        let new_tools: Vec<Arc<dyn Tool>> = if self.tool_scope.is_unrestricted() {
            new_tools
        } else {
            new_tools
                .into_iter()
                .filter(|tool| self.tool_scope.admits_tool(&tool.definition().name))
                .collect()
        };
        let prefix = format!("mcp__{server_name}__");
        let mut tools = crate::sync::write(&self.tools);
        let removed: Vec<String> = tools
            .names()
            .filter(|name| name.starts_with(&prefix))
            .map(str::to_string)
            .collect();
        tools.retain(|definition| !definition.name.starts_with(&prefix));
        for tool in new_tools {
            tools.push(tool);
        }
        drop(tools);

        if !removed.is_empty() {
            let mut deferred = crate::sync::write(&self.deferred);
            for name in &removed {
                deferred.remove(name);
            }
        }
    }

    /// Register just `tool_load`, for tests that exercise its unfindable-name path against a real
    /// registry + manager pair without building the whole builtin set.
    #[cfg(test)]
    pub(crate) fn register_load_tool_for_test(&self) {
        self.register_builtin(Arc::new(load_tool::LoadToolTool {
            tools: Arc::downgrade(&self.tools),
            deferred: Arc::downgrade(&self.deferred),
            mcp_manager: Arc::downgrade(&self.mcp_manager),
            permission_overrides: Arc::clone(&self.permission_overrides),
            permission: crate::tools::tests::shared_permission_for_test(),
            profile: Arc::clone(&self.profile),
        }));
    }

    /// Record the manager whose servers back this registry's MCP tools. Called by
    /// [`crate::tools::mcp_adapter::attach_session_registry`] (session registries) and
    /// [`crate::tools::mcp_adapter::install_on_worker_registry`] (sub-agent registries); a second
    /// call is ignored, since a registry is only ever wired to one manager.
    pub(crate) fn set_mcp_manager(&self, manager: std::sync::Weak<crate::mcp::McpClientManager>) {
        // `set` fails only when the slot is already filled, which is the documented no-op.
        let _already_set = self.mcp_manager.set(manager);
    }

    /// The manager recorded by [`Self::set_mcp_manager`], if it is still alive.
    ///
    /// The registry is the right place to ask "which manager owns these tools": a sub-agent's
    /// [`crate::agent::Agent`] has no `mcp_manager` of its own, but its registry does.
    pub(crate) fn mcp_manager(&self) -> Option<Arc<crate::mcp::McpClientManager>> {
        self.mcp_manager.get()?.upgrade()
    }

    /// Mark a tool as deferred. Deferred tools live in the registry but are hidden from the
    /// per-turn tools array until the model explicitly loads them via the `tool_load` meta-tool.
    /// Discoverability is preserved by the `[Tool discovery]` section of the per-turn `<context>`
    /// block (built from `tool_catalog()`), and the active set is recomputed per turn from the
    /// conversation, not from registry state.
    pub(crate) fn mark_deferred(&self, name: &str) {
        crate::sync::write(&self.deferred).insert(name.to_string());
    }

    /// Offer `background` on every tool this registry hands the provider.
    ///
    /// Set once, from `[background] enabled`. Sub-agent registries are deliberately left off: a
    /// sub-agent's session ends with the single turn that spawned it, so a task outliving that turn
    /// would have no conversation to report back into.
    pub(crate) fn enable_background(&self) {
        self.background_enabled
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn background_enabled(&self) -> bool {
        self.background_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Hand the registry the session's published profile; see [`Self::profile`]. A second call is
    /// ignored, since every session's registries share one cell.
    pub(crate) fn publish_profile(&self, profile: &crate::provider::PublishedProfile) {
        // `set` fails only when the slot is already filled, which is the documented no-op.
        let _already_set = self.profile.set(profile.clone());
    }

    /// Whether this session's wire carries the tool `name` now; see [`is_carried_by_wire`].
    pub(crate) fn is_carried(&self, name: &str) -> bool {
        is_carried_by_wire(&self.profile, name)
    }

    /// How this session's wire carries deferred tools now; see [`wire_delivery`].
    pub(crate) fn deferred_tool_delivery(&self) -> DeferredToolDelivery {
        wire_delivery(&self.profile)
    }

    /// Every registered tool name the wire carries. Unlike [`Self::tool_catalog`] this clones no
    /// descriptions and imposes no order, which is all a "did you mean" lookup needs.
    pub(crate) fn registered_tool_names(&self) -> Vec<String> {
        crate::sync::read(&self.tools)
            .names()
            .filter(|name| self.is_carried(name))
            .map(str::to_string)
            .collect()
    }

    /// Whether `name` is deferred, i.e. whether the model has to have loaded it to have ever seen
    /// its schema.
    pub(crate) fn is_deferred(&self, name: &str) -> bool {
        crate::sync::read(&self.deferred).contains(name)
    }

    pub(crate) fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        crate::sync::read(&self.tools)
            .get(name)
            .map(|entry| Arc::clone(&entry.tool))
    }

    /// The level a call to `name` must clear, by [`crate::tools::effective_permission`]; `None`
    /// for a name that is not registered, an override or not, so an override cannot admit a
    /// probe that matches nothing.
    pub(crate) fn required_permission_for(&self, name: &str) -> Option<Permission> {
        self.get(name).map(|tool| {
            crate::tools::effective_permission(
                &self.permission_overrides,
                name,
                tool.required_permission(),
            )
        })
    }

    /// Slice-based convenience wrapper for tests: composes [`extract_loaded_tool_names`] with
    /// [`Self::active_tools`]. Production code goes through the events-aware path
    /// (see [`crate::tools::load_tool::extract_loaded_tool_names_from_events`]) so
    /// `Event::CompactBoundary::loaded_tools_snapshot` survives across compaction; a slice-only
    /// scan loses the snapshot.
    #[cfg(test)]
    pub(crate) fn definitions_active(&self, messages: &[Message]) -> Vec<ToolDefinition> {
        // Sorted so the slice-only path is deterministic; the events-aware path preserves real load
        // order, which is what the append-only cache prefix depends on.
        let mut loaded: Vec<String> = extract_loaded_tool_names(messages).into_iter().collect();
        loaded.sort();
        self.definitions_active_with_loaded(&loaded)
    }

    /// Returns every active tool definition regardless of the caller's current permission. The
    /// active set is the union of non-deferred tools and deferred tools whose schema has been
    /// loaded via the `tool_load` meta-tool. `loaded` is computed by the caller (via
    /// [`crate::tools::load_tool::extract_loaded_tool_names_from_events`], which is the only door
    /// for that question outside tests).
    ///
    /// Blocked calls are rejected at dispatch; keeping the tools array permission-independent is
    /// what preserves the prompt cache prefix across `/permission` toggles (breakpoint 3 in the
    /// Claude provider's cache layout). `loaded` must be in load order (see
    /// [`crate::tools::load_tool::extract_loaded_tool_names_from_events`]).
    ///
    /// Emits always-active tools first in registration order, then loaded-deferred tools in the
    /// order they were loaded. Both halves grow only at the tail as a session proceeds, which is
    /// what keeps this array a stable cache prefix: `tool_load` only ever appends to the
    /// conversation, so the second half only ever gains entries.
    ///
    /// Filtering the registration-ordered list in place would *not* be append-only. With tools
    /// registered `[a, b]`, loading `b` then `a` yields `[b]` and then `[a, b]`, inserting `a`
    /// ahead of `b`. The array precedes every message, so that edit re-caches the whole
    /// conversation.
    ///
    /// The one thing that does disturb the first half is [`Self::replace_server_tools`], which
    /// removes an MCP server's tools and re-appends the new set at the tail. That genuinely
    /// rewrites part of the array and cannot be avoided when a server withdraws a tool, but it is
    /// confined here: the system prompt ahead of the array is unaffected, so the blast radius is
    /// the array rather than the whole conversation.
    #[cfg(test)]
    pub(crate) fn definitions_active_with_loaded(&self, loaded: &[String]) -> Vec<ToolDefinition> {
        self.active_tools(loaded).definitions
    }

    /// The tools a request carries, by the wire's delivery. Every tool that is not deferred, in
    /// registration order; then, on a wire that appends, the deferred ones `loaded` names in the
    /// order they were loaded, so a replay of the same conversation builds the same array. On a
    /// wire that declares, every deferred tool rides separately as a deferred definition whatever
    /// `loaded` holds, and `loaded` only prices the gauge. With what their schemas cost.
    ///
    /// `tool_load` and `tool_search` ride only while something is deferred for them to find, and
    /// never on a wire whose own search item does both jobs: with every tool declared or hidden
    /// they would be two schemas the model can only call for nothing.
    pub(crate) fn active_tools(&self, loaded: &[String]) -> ActiveTools {
        let deferred = crate::sync::read(&self.deferred);
        let tools = crate::sync::read(&self.tools);
        let carried = self.carried(&tools, &deferred);
        let delivery = carried.delivery;

        let mut definitions: Vec<ToolDefinition> = Vec::new();
        let mut deferred_definitions: Vec<ToolDefinition> = Vec::new();
        let mut schema_tokens: u64 = 0;
        for entry in carried.active.iter().filter(|entry| carried.rides(entry)) {
            definitions.push(entry.definition.clone());
            schema_tokens = schema_tokens.saturating_add(entry.schema_tokens);
        }
        for entry in &carried.deferred {
            if carried.declared_outright {
                definitions.push(entry.definition.clone());
                schema_tokens = schema_tokens.saturating_add(entry.schema_tokens);
            } else if delivery.declares() {
                deferred_definitions.push(entry.definition.clone());
            }
        }

        // On a wire that appends, a loaded tool joins the array in load order, so a replay of the
        // same conversation builds the same array; on a wire that declares, its schema reaches the
        // model through the reference rather than the array, so the load only prices it. A name
        // that isn't deferred is in the active half already; one with no registry entry was loaded
        // and later removed (an MCP `tools/list_changed` swap) and drops out; a set declared
        // outright is in the array and priced.
        let mut loaded_names: HashSet<&str> = HashSet::new();
        for name in loaded {
            if carried.declared_outright || !deferred.contains(name) || !self.is_carried(name) {
                continue;
            }
            if let Some(entry) = tools.get(name) {
                loaded_names.insert(name.as_str());
                if !delivery.declares() {
                    definitions.push(entry.definition.clone());
                }
                schema_tokens = schema_tokens.saturating_add(entry.schema_tokens);
            }
        }
        drop(carried);
        drop(tools);
        drop(deferred);

        if self.background_enabled() {
            for definition in &mut definitions {
                if crate::tools::detachable(&definition.name)
                    && offer_background(&mut definition.parameters)
                {
                    schema_tokens = schema_tokens.saturating_add(*BACKGROUND_PROPERTY_TOKENS);
                }
            }
            // A deferred declaration carries the property so the schema the endpoint expands
            // offers it, but the window holds only the schemas the model loaded, so only those
            // price it.
            for definition in &mut deferred_definitions {
                if crate::tools::detachable(&definition.name)
                    && offer_background(&mut definition.parameters)
                    && loaded_names.contains(definition.name.as_str())
                {
                    schema_tokens = schema_tokens.saturating_add(*BACKGROUND_PROPERTY_TOKENS);
                }
            }
        }

        ActiveTools {
            definitions,
            deferred_definitions,
            schema_tokens,
        }
    }

    /// What a request carries of the registry, before pricing: one classification for the array
    /// and the catalog, so the catalog never calls a tool deferred that the array declares.
    fn carried<'a>(&self, tools: &'a Registered, deferred_names: &HashSet<String>) -> Carried<'a> {
        let delivery = self.deferred_tool_delivery();
        let (deferred, active): (Vec<&RegisteredTool>, Vec<&RegisteredTool>) = tools
            .iter()
            .filter(|entry| self.is_carried(&entry.definition.name))
            .partition(|entry| deferred_names.contains(&entry.definition.name));
        let discovery_rides = !deferred.is_empty() && delivery != DeferredToolDelivery::Searched;
        let mut carried = Carried {
            delivery,
            active,
            deferred,
            discovery_rides,
            declared_outright: false,
        };
        carried.declared_outright = !carried.deferred.is_empty()
            && !carried.active.iter().any(|entry| carried.rides(entry));
        carried
    }

    /// Returns (name, description, required_permission, is_deferred) for every registered tool.
    /// Drives the permission-independent `[Available tools]` catalog plus the per-turn
    /// `[Permission context]` block that names currently-blocked tools. Sorted by (name) for
    /// deterministic output.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "the classification borrows the registry guard until the entries are built; the guard \
                  is scoped to that block and the sort runs after it is dropped"
    )]
    pub(crate) fn tool_catalog(&self) -> Vec<(String, String, Permission, bool)> {
        let entry_of = |entry: &RegisteredTool, is_deferred: bool| {
            let def = &entry.definition;
            let required = crate::tools::effective_permission(
                &self.permission_overrides,
                &def.name,
                entry.tool.required_permission(),
            );
            (
                def.name.clone(),
                def.description.clone(),
                required,
                is_deferred,
            )
        };
        let mut entries: Vec<(String, String, Permission, bool)> = {
            let deferred = crate::sync::read(&self.deferred);
            let tools = crate::sync::read(&self.tools);
            let carried = self.carried(&tools, &deferred);
            carried
                .active
                .iter()
                // The wire's own search item answers to `tool_search`'s name; `tool_load` has no
                // counterpart there and leaves the catalog as it leaves the request.
                .filter(|entry| {
                    !(entry.definition.name == LOAD_TOOL_NAME
                        && carried.delivery == DeferredToolDelivery::Searched)
                })
                .map(|entry| entry_of(entry, false))
                // A set declared outright is in the array with its schemas, so to the model it is
                // not deferred.
                .chain(
                    carried
                        .deferred
                        .iter()
                        .map(|entry| entry_of(entry, !carried.declared_outright)),
                )
                .collect()
        };
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries
    }

    /// Register the core tools shared by the main agent and sub-agents: file I/O, search, web, and
    /// shell execution. Takes a [`crate::session::ToolSite`] rather than the whole
    /// [`SessionCells`], because a gate probe has no session to take one from.
    pub(super) fn register_core_tools(
        &self,
        core: &CoreMaterials,
        site: crate::session::ToolSite,
    ) -> Result<()> {
        let web_client_config = &core.web_client;
        let sandbox_enabled = core.sandbox_enabled;
        let sandbox_capability = core.sandbox_capability.clone();
        let sandbox_backend = core.sandbox_backend;
        let backend_probe = core.backend_probe.clone();
        let read_tracker = self.read_tracker.clone();
        // One scope for every tool that writes a path the user named, so `file_write`, `file_edit`
        // and `scratchpad_save_file` cannot end up judging the boundary differently.
        let write_scope = crate::workspace::WriteScope::new(
            site.permission.clone(),
            site.roots.clone(),
            core.write_locks.clone(),
        );
        *crate::sync::write(&self.write_scope) = Some(write_scope.clone());
        self.register_builtin(Arc::new(file::ReadFileTool {
            read_tracker: read_tracker.clone(),
            site: site.clone(),
        }));
        self.register_builtin(Arc::new(file::EditFileTool {
            scope: write_scope.clone(),
            read_tracker: read_tracker.clone(),
            site: site.clone(),
        }));
        self.register_builtin(Arc::new(file::WriteFileTool {
            scope: write_scope.clone(),
            read_tracker,
            site: site.clone(),
        }));
        self.register_builtin(Arc::new(find::FindFilesTool { site: site.clone() }));
        self.register_builtin(Arc::new(grep::SearchContentsTool { site: site.clone() }));
        // A malformed proxy URL or unreadable CA file surfaces as a startup error rather than
        // silently falling back to an unconfigured client (which would ignore the user's intent).
        if web::web_fetch_registered(site.network) {
            let web_client = web::build_web_client(web_client_config)?;
            self.register_builtin(Arc::new(web::FetchUrlTool {
                refuse_private_destinations: web_client_config.refuse_private_destinations,
                client: web_client,
            }));
        }
        self.register_builtin(Arc::new(shell::ExecuteCommandTool {
            scope: write_scope,
            #[cfg(windows)]
            windows_grants: Arc::clone(crate::sandbox::windows::process_grants()),
            sandbox_capability,
            sandbox_backend,
            backend_probe,
            sandbox_enabled,
            site,
        }));
        Ok(())
    }

    /// Register a builtin. Collisions panic (programmer error). Tools rejected by the `[tools]`
    /// filter are silently skipped.
    pub(super) fn register_builtin(&self, tool: Arc<dyn Tool>) {
        let name = tool.definition().name;
        if !self.admits(&name) {
            return;
        }
        #[allow(
            clippy::expect_used,
            reason = "two builtins sharing a name is a bug the first build must surface"
        )]
        self.register(tool).expect("builtin tool name collision");
    }

    /// Register the session-scoped tools (tool_load, skill_*, image_render, checklist_*,
    /// scratchpad_*) on the registry. Shared between [`Self::build_default`] and
    /// [`Self::build_for_subagent`] so adding a new such tool to the parent automatically gives
    /// it to sub-agents too.
    ///
    /// `parent_session_id` + `inherited_scratchpad_names` configure read-only scratchpad
    /// inheritance for sub-agents. Both are `None`/empty on the root agent's registry, so no
    /// fallback path is taken there.
    pub(super) fn register_session_scoped_tools(
        &self,
        materials: &SessionMaterials,
        cells: &SessionCells,
        scope: SessionScope,
    ) {
        let SessionScope {
            skills_managed,
            memory_access,
            parent_session_id,
            inherited_scratchpad_names,
            schedule,
            gate_tools,
            background,
            context_ceiling_percent,
            auto_compact,
        } = scope;
        let store = materials.store.clone();
        let site = cells.site();
        let checklist = cells.checklist.clone();
        let skills = materials.skills.clone();
        let memories = materials.memories.clone();
        let background = background.then(|| cells.background_tasks.clone());
        *crate::sync::write(&self.inherited_scratchpad_names) = inherited_scratchpad_names.clone();
        // The two registry meta-tools, root and sub-agent alike: the same door for both is what
        // keeps a worker's picture of its tools identical to its parent's.
        let loader = load_tool::LoadToolTool {
            tools: Arc::downgrade(&self.tools),
            deferred: Arc::downgrade(&self.deferred),
            mcp_manager: Arc::downgrade(&self.mcp_manager),
            permission_overrides: Arc::clone(&self.permission_overrides),
            permission: cells.permission.clone(),
            profile: Arc::clone(&self.profile),
        };
        // In the order the model meets them: a search finds, a load follows.
        self.register_builtin(Arc::new(load_tool::ToolSearchTool {
            loader: loader.clone(),
        }));
        self.register_builtin(Arc::new(loader));
        // Both stores register their tools only when the subsystem is switched on. Skipping is
        // the whole point of the config switch: a disabled subsystem must keep its schemas out of
        // every request, not ship tools that can only ever fail.
        if skills.enabled() {
            self.register_builtin(Arc::new(skill::SkillReadTool {
                skills: skills.clone(),
            }));
            self.register_builtin(Arc::new(skill::SkillSearchTool {
                skills: skills.clone(),
            }));
            // Authoring is opt-in per installation and never reaches a sub-agent. Same reasoning
            // as `MemoryAccess::Write` being unreachable from `agent_spawn`, and stronger here: a
            // sub-agent that inferred something from one narrow task should not rewrite the store
            // every future agent reasons from, and a skill is additionally spawnable *as* a task.
            if skills_managed {
                self.register_builtin(Arc::new(skill::SkillWriteTool {
                    skills: skills.clone(),
                }));
                self.register_builtin(Arc::new(skill::SkillDeleteTool { skills }));
            }
        }
        // Two independent gates: `enabled()` is whether this installation keeps memories at all,
        // `memory_access` is how much of that store the agent in front of us may reach.
        if memories.enabled() && memory_access != crate::config::MemoryAccess::None {
            // Registration order is the order the tools reach the provider, and the tool array
            // heads the prompt-cache prefix, so the gates are interleaved rather than grouped to
            // keep a full-access agent's order stable.
            if memory_access == crate::config::MemoryAccess::Write {
                self.register_builtin(Arc::new(memory::MemoryWriteTool {
                    memories: memories.clone(),
                }));
            }
            self.register_builtin(Arc::new(memory::MemoryReadTool {
                memories: memories.clone(),
            }));
            self.register_builtin(Arc::new(memory::MemorySearchTool {
                memories: memories.clone(),
            }));
            if memory_access == crate::config::MemoryAccess::Write {
                self.register_builtin(Arc::new(memory::MemoryDeleteTool { memories }));
            }
        }
        self.register_builtin(Arc::new(render_image::RenderImageTool {
            store: store.clone(),
            site: site.clone(),
        }));
        if let Some(schedule_config) = schedule
            && schedule_config.enabled
        {
            for tool in schedule::build(store.clone(), site.clone(), schedule_config, gate_tools) {
                self.register_builtin(tool);
            }
        }
        if let Some(tasks) = background
            && self.background_enabled()
        {
            for tool in background::build(store.clone(), site.clone(), tasks) {
                self.register_builtin(tool);
            }
        }
        self.register_builtin(Arc::new(checklist::ChecklistAddTool {
            checklist: checklist.clone(),
        }));
        self.register_builtin(Arc::new(checklist::ChecklistEditTool {
            checklist: checklist.clone(),
            store: store.clone(),
            site: site.clone(),
        }));
        self.register_builtin(Arc::new(checklist::ChecklistReadTool { checklist }));
        self.register_builtin(Arc::new(scratchpad::ScratchpadWriteTool {
            store: store.clone(),
            inherited_names: inherited_scratchpad_names.clone(),
            site: site.clone(),
        }));
        self.register_builtin(Arc::new(scratchpad::ScratchpadReadTool {
            store: store.clone(),
            parent_session_id,
            inherited_names: inherited_scratchpad_names.clone(),
            site: site.clone(),
            gauge: context::ContextGauge::new(cells, context_ceiling_percent, auto_compact),
        }));
        self.register_builtin(Arc::new(conversation::ConversationSearchTool {
            store: store.clone(),
            site: site.clone(),
        }));
        self.register_builtin(Arc::new(conversation::ConversationReadTool {
            store: store.clone(),
            site: site.clone(),
        }));
        self.register_builtin(Arc::new(conversation::SessionListTool {
            store: store.clone(),
            site: site.clone(),
        }));
        self.register_builtin(Arc::new(conversation::SessionSearchTool {
            store: store.clone(),
            site: site.clone(),
        }));
        self.register_builtin(Arc::new(scratchpad::ScratchpadEditTool {
            store: store.clone(),
            inherited_names: inherited_scratchpad_names.clone(),
            site: site.clone(),
        }));
        self.register_builtin(Arc::new(scratchpad::ScratchpadListTool {
            store: store.clone(),
            parent_session_id,
            inherited_names: inherited_scratchpad_names.clone(),
            site: site.clone(),
        }));
        self.register_builtin(Arc::new(scratchpad::ScratchpadMergeTool {
            store: store.clone(),
            parent_session_id,
            inherited_names: inherited_scratchpad_names.clone(),
            site: site.clone(),
        }));
        self.register_builtin(Arc::new(scratchpad::ScratchpadDeleteTool {
            store: store.clone(),
            inherited_names: inherited_scratchpad_names.clone(),
            site: site.clone(),
        }));
        self.register_builtin(Arc::new(scratchpad::ScratchpadRenameTool {
            store: store.clone(),
            inherited_names: inherited_scratchpad_names.clone(),
            site: site.clone(),
        }));
        self.register_builtin(Arc::new(scratchpad::ScratchpadLoadFileTool {
            store: store.clone(),
            inherited_names: inherited_scratchpad_names.clone(),
            site: site.clone(),
        }));
        self.register_builtin(Arc::new(scratchpad::ScratchpadSaveFileTool {
            scope: self.write_scope_or_deny_all(),
            store,
            parent_session_id,
            inherited_names: inherited_scratchpad_names,
            read_tracker: Arc::clone(&self.read_tracker),
            site,
        }));
    }

    /// Register the always-on half of the `context_*` family.
    ///
    /// Outside [`Self::build_default`] for the same reason `agent_spawn` is: the counters these
    /// read are owned by the `Agent`, which does not exist when the registry is built. The caller
    /// makes them, hands them here, and points the agent at the same handles.
    ///
    /// `context_replace` is deliberately absent. It only exists inside a checkpoint turn, which
    /// builds its own tool list; registering it here would offer the model a way to blank its
    /// context during ordinary work.
    ///
    /// Sub-agents get neither. A sub-agent's window is its own, but the compaction it would be
    /// asking for happens inside a turn its parent is waiting on, and the parent owns that
    /// conversation.
    pub(crate) fn register_context_tools(
        &self,
        gauge: context::ContextGauge,
        pending: crate::session::PendingCompaction,
        checkpoint_enabled: bool,
        materials: &SessionMaterials,
        cells: &SessionCells,
    ) {
        self.register_builtin(Arc::new(context::ContextCheckTool {
            gauge,
            store: materials.store.clone(),
            site: cells.site(),
        }));
        self.register_builtin(Arc::new(context::ContextCompactTool {
            pending,
            checkpoint_enabled,
        }));
    }

    /// The tools a checkpoint turn may use: this registry's, filtered to [`CHECKPOINT_TOOL_NAMES`]
    /// and the caller's permission, plus a fresh `context_replace` bound to `slot`. With
    /// `approvals` on the level does not filter: a call above it is put to the user at dispatch,
    /// the way it is in an ordinary turn, so the checkpoint offers what it can ask about.
    ///
    /// Read *out of* the live registry rather than built fresh, so a tool the user disabled in
    /// `[tools]` stays disabled here, and every tool keeps the session's cwd, permission and
    /// frontend already baked into it.
    pub(crate) fn checkpoint_tools(
        &self,
        permission: Permission,
        approvals: bool,
        slot: context::SubmissionSlot,
    ) -> Vec<Arc<dyn Tool>> {
        let mut tools: Vec<Arc<dyn Tool>> = CHECKPOINT_TOOL_NAMES
            .iter()
            .filter_map(|name| self.get(name))
            .filter(|tool| {
                let definition = tool.definition();
                let required = crate::tools::effective_permission(
                    &self.permission_overrides,
                    &definition.name,
                    tool.required_permission(),
                );
                approvals || permission.allows(required)
            })
            .collect();
        // Unconditional, deliberately outside the permission filter above. `context_replace` has
        // no effect outside this process - it writes a summary into a slot the caller owns - and
        // without it a checkpoint has no way to finish, so filtering it out would silently
        // downgrade every compaction at `none` permission to the fallback summarizer. That leaves
        // a `none`-permission checkpoint holding exactly one tool, which is the right shape: it
        // can still write the summary, it just has nowhere to save anything.
        tools.push(Arc::new(context::ContextReplaceTool { slot }));
        tools
    }

    /// The root agent's registry: the core tools and the session-scoped set, with the
    /// `background` parameter offered when `[background]` is on.
    pub(crate) fn build_default(
        materials: &SessionMaterials,
        cells: &SessionCells,
        options: &crate::session::AgentOptions,
    ) -> Result<Self> {
        let registry = Self::new_with_filter(materials.core.builtin_filter.clone());
        registry.publish_profile(&cells.profile);
        // Before the tools are registered, because both the `task_*` registration and the schema
        // splice read this flag.
        if materials.background.enabled {
            registry.enable_background();
        }
        registry.register_core_tools(&materials.core, cells.site())?;
        registry.register_session_scoped_tools(materials, cells, SessionScope {
            skills_managed: materials.skills_agent_managed,
            memory_access: crate::config::MemoryAccess::Write,
            parent_session_id: None,
            inherited_scratchpad_names: Vec::new(),
            schedule: Some(materials.schedule.clone()),
            gate_tools: options.gate_tools.clone(),
            background: true,
            context_ceiling_percent: options.context_ceiling_percent,
            auto_compact: options.auto_compact,
        });
        // The `context_*` tools read the same cells the agent gauges with, so the two never
        // disagree about occupancy or about a compaction one of them asked for.
        registry.register_context_tools(
            context::ContextGauge::new(
                cells,
                options.context_ceiling_percent,
                options.auto_compact,
            ),
            Arc::clone(&cells.pending_compaction),
            options.compact_checkpoint,
            materials,
            cells,
        );
        // `subagent_max_depth == 0` disables sub-agents entirely (the root gets no `agent_spawn`);
        // `>= 1` seeds the root's soft recursion budget, and `absolute_depth` starts at 0 for the
        // root. The predicate folds in the `[tools]` half too, and is named because `meka tool
        // list` has to reach the same answer with no provider to assemble a session with.
        if super::subagent::agent_tools_registered(
            &materials.core.builtin_filter,
            materials.subagent_max_depth,
        ) {
            let config_scope = ToolScope::new(
                materials.subagents.disabled_servers.clone(),
                materials.subagents.disabled_tools.clone(),
            );
            super::subagent::register_subagent_tools(&registry, super::subagent::AgentSpawnTool {
                parent_permission: cells.permission.clone(),
                tool_builder_params: super::subagent::ToolBuilderParams {
                    materials: materials.clone(),
                    cells: cells.clone(),
                    // The root agent holds the whole store, so that is the ceiling on what it
                    // can grant a sub-agent. Whether a sub-agent gets anything is decided per
                    // `agent_spawn` call, and defaults to nothing.
                    memory_access: crate::config::MemoryAccess::Write,
                    config_scope: config_scope.clone(),
                    parent_options: options.clone(),
                },
                inherited_scope: config_scope,
                remaining_depth: materials.subagent_max_depth,
                absolute_depth: 0,
            })?;
        }
        Ok(registry)
    }

    /// Build a registry holding only the core tools, for a scheduled gate's tool probe.
    ///
    /// Nothing session-scoped is registered: a gate is a predicate, and `memory_*` / `skill_*` /
    /// `checklist_*` are not questions about the world. What it does get is the read-only built-ins
    /// a watcher wants (`file_read`, `web_fetch`), built against the job's cwd rather
    /// than the host process's, for the same reason a shell gate runs there.
    ///
    /// Construction is allocation only, no I/O, so a caller may build one per evaluation.
    /// `SilentFrontend` because nobody is watching a scheduled fire, and empty roots.
    ///
    /// Empty roots is not the same as nowhere: `WriteScope::confined_to` at `read` still admits the
    /// job's own directory. A gate may only call a tool that resolves to `read`, and nothing meka
    /// ships writes at that level, but `tool_permissions` can lower one that does, and it then
    /// writes under the session's cwd, unattended, for as long as the job exists: the operator's
    /// own instruction, which is why this does not claim writes are impossible. `shell_execute`
    /// cannot be opened this way, since it re-derives its own confinement rather than trusting
    /// the level.
    pub(crate) fn for_gate(core: &CoreMaterials, site: crate::session::ToolSite) -> Result<Self> {
        let registry = Self::new_with_filter(core.builtin_filter.clone());
        registry.register_core_tools(core, site)?;
        Ok(registry)
    }

    /// Build a tool registry for a sub-agent. Sub-agents get the same session-scoped tools as the
    /// parent (tool_load, skill_*, memory_*, image_render, checklist_*, scratchpad_*) scoped to
    /// their own ephemeral child session, through `cells` that are the sub-agent's own.
    ///
    /// `agent_spawn` is deliberately not registered here, but sub-agents *can* nest: the caller
    /// adds it afterwards when the recursion budget allows (`AgentSpawnTool::execute`), because
    /// the child's depth counters aren't known until the parent's `max_depth` override has been
    /// resolved. Registering it outside this builder mirrors the root's own registration in
    /// `assemble_agent`.
    ///
    /// `scope` is what makes a sub-agent's registry narrower than its parent's rather than a copy
    /// of it. Its tool scope and memory level are already narrowed and resolved by the caller:
    /// this builder applies them, it does not decide them.
    pub(crate) fn build_for_subagent(
        materials: &SessionMaterials,
        cells: &SessionCells,
        scope: RegistryScope,
    ) -> Result<Self> {
        let registry = Self::new_with_filter_and_scope(
            materials.core.builtin_filter.clone(),
            scope.tool_scope,
        );
        registry.publish_profile(&cells.profile);
        registry.register_core_tools(&materials.core, cells.site())?;
        registry.register_session_scoped_tools(materials, cells, SessionScope {
            // Never, whatever the installation's `agent_managed` says. See the registration site.
            skills_managed: false,
            memory_access: scope.memory_access,
            parent_session_id: scope.parent_session_id,
            inherited_scratchpad_names: scope.inherited_scratchpad_names,
            schedule: None,
            gate_tools: None,
            background: false,
            context_ceiling_percent: scope.context_ceiling_percent,
            auto_compact: scope.auto_compact,
        });
        Ok(registry)
    }
}

/// What distinguishes a sub-agent's registry from its parent's.
pub(crate) struct RegistryScope {
    pub(crate) tool_scope: ToolScope,
    pub(crate) memory_access: crate::config::MemoryAccess,
    /// The parent, when the sub-agent may read some of its scratchpad; see
    /// [`ToolRegistry::build_for_subagent`].
    pub(crate) parent_session_id: Option<Uuid>,
    pub(crate) inherited_scratchpad_names: Vec<String>,
    /// The sub-agent's own `AgentOptions::context_ceiling_percent` and `auto_compact`, for the
    /// gauge its reads are sized against and `context_check` reports.
    pub(crate) context_ceiling_percent: u64,
    pub(crate) auto_compact: bool,
}

/// Which of the session-scoped tools a registry gets, and at what level.
pub(super) struct SessionScope {
    pub(super) skills_managed: bool,
    pub(super) memory_access: crate::config::MemoryAccess,
    pub(super) parent_session_id: Option<Uuid>,
    pub(super) inherited_scratchpad_names: Vec<String>,
    /// The `schedule_*` tools and their ceilings; `None` leaves them out.
    pub(super) schedule: Option<crate::config::ResolvedScheduleConfig>,
    /// The dispatcher the `schedule_*` tools report gates against.
    pub(super) gate_tools: Option<Arc<dyn crate::schedule::GateTools>>,
    /// The `task_*` tools, when the registry offers `background` at all.
    pub(super) background: bool,
    /// `AgentOptions::context_ceiling_percent` and `auto_compact` of the agent the registry
    /// serves, for the context gauge `scratchpad_read` sizes a whole read against.
    pub(super) context_ceiling_percent: u64,
    pub(super) auto_compact: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::BuiltinToolFilter,
        conversation::Role,
        store::Store,
        tools::tests::{
            checklist_for_test, shared_permission_for_test, tool_registry_for_test,
            tool_registry_for_test_with_filter,
        },
    };

    /// The shell tool the production builder hands out writes into the process-wide grant ledger,
    /// not one of its own: `WindowsGrants` is a singleton because a sub-agent finishing its task
    /// must not revoke ACEs the parent is still writing through, and every other Windows test
    /// builds its tool by hand with a fresh `WindowsGrants::default()`.
    ///
    /// Asserted behaviorally rather than by identity because `ToolRegistry` hands back
    /// `Arc<dyn Tool>` with no downcast: run a real confined command through the registry's own
    /// `shell_execute`, then ask the process ledger whether it heard about the root.
    #[cfg(windows)]
    #[tokio::test]
    async fn the_shell_the_registry_builds_grants_through_the_process_ledger() {
        let temp = tempfile::tempdir().expect("tempdir");
        let workspace = crate::workspace::canonical_for_test(temp.path());

        let permission = crate::permission::SharedPermission::new(
            crate::permission::Permission::Workspace,
            crate::permission::EnabledPermissions::ALL,
        );
        let store = Store::for_test().await;
        let sandbox_capability = crate::sandbox::detect();
        let registry = ToolRegistry::build_default(
            &crate::session::SessionMaterials {
                providers: std::sync::Arc::new(crate::provider::ProviderRegistry::for_test(
                    store.token_store(),
                    &["test-profile"],
                )),
                core: crate::session::CoreMaterials {
                    web_client: crate::config::WebClientConfig::default(),
                    sandbox_enabled: true,
                    sandbox_capability: sandbox_capability.clone(),
                    sandbox_backend: crate::config::SandboxBackend::Landlock,
                    backend_probe: crate::sandbox::BackendProbe::Ok(sandbox_capability),
                    builtin_filter: BuiltinToolFilter::default(),
                    write_locks: crate::workspace::WriteLocks::default(),
                    running_subagents: Default::default(),
                },
                skills: crate::skills::SkillCache::for_root(None),
                skills_agent_managed: false,
                memories: crate::store::memory::MemoryStore::detached(),
                schedule: crate::config::ResolvedScheduleConfig::default(),
                background: crate::config::ResolvedBackgroundConfig::default(),
                ..crate::session::SessionMaterials::for_test(store)
            },
            &crate::session::SessionCells {
                session_id: crate::session::SharedSessionId::default(),
                checklist: checklist_for_test(),
                background_tasks: crate::background::BackgroundTasks::default(),
                ..crate::session::SessionCells::for_test(
                    permission,
                    crate::workspace::SharedCwd::new(workspace.clone()),
                    crate::workspace::SharedRoots::new(vec![workspace.clone()]),
                    Arc::new(crate::frontend::SilentFrontend),
                )
            },
            &crate::session::AgentOptions::for_test(),
        )
        .expect("registry builds");

        let shell_execute = registry
            .get("shell_execute")
            .expect("shell_execute is registered");
        let result = shell_execute
            .execute(
                serde_json::json!({"command": "cmd /c echo ok"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        assert!(result.is_ok(), "the confined command must run: {result:?}");

        let granted = crate::sandbox::windows::process_grants().granted_roots();
        assert!(
            granted.contains(&workspace),
            "the registry's shell must grant through the process-wide ledger, but it holds \
             {granted:?} and not {}",
            workspace.display()
        );
    }

    /// The registry's write boundary is bound to the session's own permission cell, so moving that
    /// cell moves the boundary. Building the shared `WriteScope` from a permanently-`Unrestricted`
    /// handle instead would sever the session's level from `file_write`, `file_edit`,
    /// `scratchpad_save_file` and `shell_execute` at once, failing open in one line.
    ///
    /// Driven through the production builder and the real tool rather than through the helpers,
    /// because the helpers are what that edit leaves working.
    #[tokio::test]
    async fn the_write_boundary_follows_the_session_permission_cell() {
        let temp = tempfile::tempdir().expect("tempdir");
        let workspace = crate::workspace::canonical_for_test(temp.path());
        let outside = crate::workspace::canonical_for_test(std::env::temp_dir())
            .join(format!("meka-outside-{}.txt", uuid::Uuid::new_v4()));

        let permission = crate::permission::SharedPermission::new(
            crate::permission::Permission::Workspace,
            crate::permission::EnabledPermissions::ALL,
        );
        let store = Store::for_test().await;
        let sandbox_capability = crate::sandbox::detect();
        let registry = ToolRegistry::build_default(
            &crate::session::SessionMaterials {
                providers: std::sync::Arc::new(crate::provider::ProviderRegistry::for_test(
                    store.token_store(),
                    &["test-profile"],
                )),
                core: crate::session::CoreMaterials {
                    web_client: crate::config::WebClientConfig::default(),
                    sandbox_enabled: true,
                    sandbox_capability: sandbox_capability.clone(),
                    sandbox_backend: crate::config::SandboxBackend::Landlock,
                    backend_probe: crate::sandbox::BackendProbe::Ok(sandbox_capability),
                    builtin_filter: BuiltinToolFilter::default(),
                    write_locks: crate::workspace::WriteLocks::default(),
                    running_subagents: Default::default(),
                },
                skills: crate::skills::SkillCache::for_root(None),
                skills_agent_managed: false,
                memories: crate::store::memory::MemoryStore::detached(),
                schedule: crate::config::ResolvedScheduleConfig::default(),
                background: crate::config::ResolvedBackgroundConfig::default(),
                ..crate::session::SessionMaterials::for_test(store)
            },
            &crate::session::SessionCells {
                session_id: crate::session::SharedSessionId::default(),
                checklist: checklist_for_test(),
                background_tasks: crate::background::BackgroundTasks::default(),
                ..crate::session::SessionCells::for_test(
                    permission.clone(),
                    crate::workspace::SharedCwd::new(workspace.clone()),
                    crate::workspace::roots_for_test(),
                    Arc::new(crate::frontend::SilentFrontend),
                )
            },
            &crate::session::AgentOptions::for_test(),
        )
        .expect("registry builds");

        let file_write = registry
            .get("file_write")
            .expect("file_write is registered");
        let write_outside = || {
            file_write.execute(
                serde_json::json!({
                    "path": outside.to_str().expect("path"),
                    "content": "payload",
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
        };

        // At `workspace` the cell confines it.
        //
        // Either refusal shape counts. `file_write` returns `Err(MekaError::ToolExecution)` from
        // `resolve_write_target` while `file_edit` returns `Ok(ToolOutput { is_error: true })`;
        // both reach the model as a failed tool call, and pinning one here would make this test
        // fail for a reason that has nothing to do with the boundary.
        let refused = write_outside().await;
        let refused_message = match &refused {
            Err(error) => error.to_string(),
            Ok(output) => {
                assert!(
                    output.is_error,
                    "at `workspace` a write outside every root must be refused: {output:?}"
                );
                output.text_content()
            }
        };
        assert!(
            refused_message.contains("outside the workspace"),
            "the refusal must name the boundary: {refused_message}"
        );
        assert!(
            !outside.exists(),
            "and must not have been written: {}",
            outside.display()
        );

        // The same registry, the same tool object, one cell mutation later.
        permission
            .try_set(crate::permission::Permission::Unrestricted)
            .expect("unrestricted is enabled in this fixture");
        let allowed = write_outside().await.expect("write");
        assert!(
            !allowed.is_error,
            "at `unrestricted` the same write must land: {allowed:?}"
        );
        assert_eq!(
            std::fs::read_to_string(&outside).expect("read back"),
            "payload"
        );
        let _ = std::fs::remove_file(&outside);
    }

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
    fn extract_loaded_tool_names_empty() {
        assert!(extract_loaded_tool_names(&[]).is_empty());
    }

    /// The scanner has one name. A call under any other name, however close, loads nothing: a
    /// store's history carries the meta-tool under the one name the ledger gives it, so nothing
    /// here tolerates a second spelling.
    #[test]
    fn a_call_under_any_other_name_loads_nothing() {
        let loaded = vec![
            load_tool_use("u1", "scratchpad_read"),
            tool_result("u1", "loaded", false),
        ];
        assert!(extract_loaded_tool_names(&loaded).contains("scratchpad_read"));

        let mut other = loaded.clone();
        let ContentBlock::ToolUse { name, .. } = &mut other[0].content[0] else {
            panic!("the first block is the call");
        };
        *name = "not_the_loader".to_string();
        assert!(extract_loaded_tool_names(&other).is_empty());
    }

    #[test]
    fn extract_loaded_tool_names_single_success() {
        let messages = vec![
            load_tool_use("u1", "scratchpad_read"),
            tool_result("u1", "loaded", false),
        ];
        let loaded = extract_loaded_tool_names(&messages);
        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains("scratchpad_read"));
    }

    #[test]
    fn extract_loaded_tool_names_from_a_batch() {
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "u1".to_string(),
                    name: LOAD_TOOL_NAME.to_string(),
                    input: serde_json::json!({"name": ["alpha", "beta"]}),
                }],
            },
            tool_result("u1", "loaded", false),
        ];
        let loaded = extract_loaded_tool_names(&messages);
        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains("alpha") && loaded.contains("beta"));
    }

    #[test]
    fn load_tool_names_dedups_and_caps() {
        let input = serde_json::json!({"name": ["a", "a", "b"]});
        assert_eq!(load_tool_names(&input), vec![
            "a".to_string(),
            "b".to_string()
        ]);

        let many: Vec<String> = (0..MAX_LOAD_TOOL_BATCH + 5)
            .map(|index| format!("tool_{index}"))
            .collect();
        let input = serde_json::json!({"name": many});
        assert_eq!(load_tool_names(&input).len(), MAX_LOAD_TOOL_BATCH);
    }

    #[test]
    fn load_tool_names_rejects_a_non_string_name() {
        assert!(load_tool_names(&serde_json::json!({})).is_empty());
        assert!(load_tool_names(&serde_json::json!({"name": 7})).is_empty());
        assert!(load_tool_names(&serde_json::json!({"name": [7, false]})).is_empty());
    }

    #[tokio::test]
    async fn background_is_absent_until_enabled_and_present_after() {
        let registry = ToolRegistry::new();
        registry.register_deferred_fixture("fixture_alpha");
        let has_background = |registry: &ToolRegistry| {
            registry
                .definitions_active_with_loaded(&["fixture_alpha".to_string()])
                .iter()
                .any(|definition| {
                    definition
                        .parameters
                        .get("properties")
                        .and_then(|properties| properties.get(BACKGROUND_PARAMETER))
                        .is_some()
                })
        };

        assert!(
            !has_background(&registry),
            "a parameter the model can see but cannot use is worse than one it cannot see"
        );
        registry.enable_background();
        assert!(has_background(&registry));
    }

    #[test]
    fn extract_loaded_tool_names_error_excluded() {
        let messages = vec![
            load_tool_use("u1", "missing_tool"),
            tool_result("u1", "Error: not registered", true),
        ];
        assert!(extract_loaded_tool_names(&messages).is_empty());
    }

    #[test]
    fn extract_loaded_tool_names_orphan_use() {
        // tool_load was issued but the tool_result hasn't arrived yet.
        let messages = vec![load_tool_use("u1", "scratchpad_read")];
        assert!(extract_loaded_tool_names(&messages).is_empty());
    }

    #[test]
    fn extract_loaded_tool_names_ignores_other_tools() {
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "u1".to_string(),
                    name: "file_read".to_string(),
                    input: serde_json::json!({ "name": "anything" }),
                }],
            },
            tool_result("u1", "ok", false),
        ];
        assert!(extract_loaded_tool_names(&messages).is_empty());
    }

    #[test]
    fn extract_loaded_tool_names_malformed_input() {
        // tool_load called with no `name` field: must not panic, must not pollute the active set.
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
    fn extract_loaded_tool_names_multiple_loads_dedup() {
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
    fn extract_loaded_tool_names_multi_block_message() {
        // The model can emit several `tool_use` blocks in one assistant message; the matching
        // `tool_result`s come back as separate blocks of one user message. Both must be processed.
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
    fn extract_loaded_tool_names_mismatched_id() {
        // tool_result references an id that no `tool_load` use claimed. The result is dropped; the
        // orphan use stays unmatched and is not added to the active set.
        let messages = vec![
            load_tool_use("u1", "scratchpad_read"),
            tool_result("u_other", "ok", false),
        ];
        assert!(extract_loaded_tool_names(&messages).is_empty());
    }

    #[test]
    fn extract_loaded_tool_names_interleaved_with_other_tool_calls() {
        // tool_load calls share the message stream with regular tool calls; the scanner must pair
        // on tool_use_id, not on positional adjacency.
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::ToolUse {
                        id: "u1".to_string(),
                        name: "file_read".to_string(),
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

    /// The other half of the rename hazard: a `[tools]` block naming the old tool silently stops
    /// applying, and "matches nothing" alone reads as a typo the user has to hunt for.
    #[test]
    fn stale_config_entry_is_pointed_at_the_renamed_tool() {
        let hint = builtin_name_hint("skill");
        assert!(hint.contains("`skill_read`"), "{hint:?}");
        // Still silent when the entry really is nonsense, so the hint stays worth reading.
        assert_eq!(builtin_name_hint("frobnicate_widget"), "");
    }

    /// A sub-agent never gets the authoring tools, whatever the installation says. Mirrors
    /// `MemoryAccess::Write` being unreachable from `agent_spawn`, and matters more here because a
    /// skill is spawnable: a sub-agent could otherwise rewrite the instructions its siblings run
    /// on.
    #[tokio::test]
    async fn subagent_never_gets_skill_authoring_tools() {
        let registry =
            subagent_registry(ToolScope::default(), crate::config::MemoryAccess::Write).await;
        assert!(
            registry.get("skill_read").is_some(),
            "a sub-agent still reads skills"
        );
        assert!(registry.get("skill_search").is_some());
        assert!(registry.get("skill_write").is_none());
        assert!(registry.get("skill_delete").is_none());
    }

    /// Disabling a subsystem keeps its schemas out of the request entirely, which is the whole
    /// point of the config switch.
    #[tokio::test]
    async fn disabled_stores_register_no_tools() {
        let store = Store::for_test().await;
        let sandbox_capability = crate::sandbox::detect();
        let backend_probe = crate::sandbox::BackendProbe::Ok(sandbox_capability.clone());
        let registry = ToolRegistry::build_default(
            &crate::session::SessionMaterials {
                providers: std::sync::Arc::new(crate::provider::ProviderRegistry::for_test(
                    store.token_store(),
                    &["test-profile"],
                )),
                core: crate::session::CoreMaterials {
                    web_client: crate::config::WebClientConfig::default(),
                    sandbox_enabled: true,
                    sandbox_capability,
                    sandbox_backend: crate::config::SandboxBackend::Landlock,
                    backend_probe,
                    builtin_filter: BuiltinToolFilter::default(),
                    write_locks: crate::workspace::WriteLocks::default(),
                    running_subagents: Default::default(),
                },
                skills: crate::skills::SkillCache::disabled(),
                skills_agent_managed: false,
                memories: crate::store::memory::MemoryStore::disabled(),
                schedule: crate::config::ResolvedScheduleConfig::default(),
                background: crate::config::ResolvedBackgroundConfig::default(),
                ..crate::session::SessionMaterials::for_test(store)
            },
            &crate::session::SessionCells {
                session_id: crate::session::SharedSessionId::default(),
                checklist: checklist_for_test(),
                background_tasks: crate::background::BackgroundTasks::default(),
                ..crate::session::SessionCells::for_test(
                    shared_permission_for_test(),
                    crate::workspace::cwd_for_test(),
                    crate::workspace::roots_for_test(),
                    Arc::new(crate::frontend::SilentFrontend),
                )
            },
            &crate::session::AgentOptions::for_test(),
        )
        .expect("default web client config should build cleanly");

        assert!(registry.get("skill_read").is_none());
        for name in [
            "memory_write",
            "memory_read",
            "memory_search",
            "memory_delete",
        ] {
            assert!(registry.get(name).is_none(), "{name} must not register");
        }
        // Unrelated built-ins are untouched.
        assert!(registry.get("file_read").is_some());
    }

    #[tokio::test]
    async fn definitions_active_includes_write_tools() {
        let registry = tool_registry_for_test().await;
        let active = registry.definitions_active(&[]);
        assert!(active.iter().any(|t| t.name == "file_read"));
        assert!(active.iter().any(|t| t.name == "file_write"));
        assert!(active.iter().any(|t| t.name == "file_edit"));
        assert!(active.iter().any(|t| t.name == "shell_execute"));
        // All five scratchpad tools ship default: no `tool_load` round-trip.
        assert!(active.iter().any(|t| t.name == "scratchpad_write"));
        assert!(active.iter().any(|t| t.name == "scratchpad_read"));
        assert!(active.iter().any(|t| t.name == "scratchpad_edit"));
        assert!(active.iter().any(|t| t.name == "scratchpad_list"));
        assert!(active.iter().any(|t| t.name == "scratchpad_delete"));
    }

    /// The tools array precedes every message in the request, so it must only ever grow at the
    /// tail: a mid-array insertion re-caches the entire conversation behind it.
    ///
    /// Loading in reverse registration order is the case that breaks a naive implementation.
    /// Filtering the registration-ordered list in place puts `alpha` in front of `omega` once both
    /// are loaded, even though `omega` was loaded first.
    #[tokio::test]
    async fn definitions_active_grows_only_at_the_tail() {
        let registry = tool_registry_for_test().await;
        registry.register_deferred_fixture("alpha_fixture");
        registry.register_deferred_fixture("omega_fixture");

        let names = |loaded: &[String]| -> Vec<String> {
            registry
                .definitions_active_with_loaded(loaded)
                .into_iter()
                .map(|definition| definition.name)
                .collect()
        };

        // Reverse registration order: `omega_fixture` was registered second but loads first.
        let none = names(&[]);
        let one = names(&["omega_fixture".to_string()]);
        let two = names(&["omega_fixture".to_string(), "alpha_fixture".to_string()]);

        assert_eq!(
            one[..none.len()],
            none[..],
            "loading a tool must not disturb the tools already in the array"
        );
        assert_eq!(
            two[..one.len()],
            one[..],
            "loading a second tool must not reorder the first: got {two:?} after {one:?}",
        );
        assert_eq!(two[none.len()..], [
            "omega_fixture".to_string(),
            "alpha_fixture".to_string()
        ]);
    }

    /// The active tool set is byte-identical at every permission level, which is what lets the
    /// Claude prompt-cache prefix survive a mid-session Shift+Tab. This is the property
    /// `Permission::allows` is shaped around: `Workspace` and `Unrestricted` are deliberately equal
    /// there, and scope is enforced at the write door instead of by hiding tools.
    ///
    /// Compares the serialized definitions rather than the names, because "byte-identical" is the
    /// actual claim: a description or a schema that varied by level would break the cache just as
    /// thoroughly as a missing tool, and a name list cannot see either.
    #[tokio::test]
    async fn the_tools_array_is_byte_identical_at_every_permission_level() {
        let permission = crate::permission::SharedPermission::new(
            crate::permission::Permission::None,
            crate::permission::EnabledPermissions::ALL,
        );
        let store = Store::for_test().await;
        let sandbox_capability = crate::sandbox::detect();
        let registry = ToolRegistry::build_default(
            &crate::session::SessionMaterials {
                providers: std::sync::Arc::new(crate::provider::ProviderRegistry::for_test(
                    store.token_store(),
                    &["test-profile"],
                )),
                core: crate::session::CoreMaterials {
                    web_client: crate::config::WebClientConfig::default(),
                    sandbox_enabled: true,
                    sandbox_capability: sandbox_capability.clone(),
                    sandbox_backend: crate::config::SandboxBackend::Landlock,
                    backend_probe: crate::sandbox::BackendProbe::Ok(sandbox_capability),
                    builtin_filter: BuiltinToolFilter::default(),
                    write_locks: crate::workspace::WriteLocks::default(),
                    running_subagents: Default::default(),
                },
                skills: crate::skills::SkillCache::for_root(None),
                skills_agent_managed: false,
                memories: crate::store::memory::MemoryStore::detached(),
                schedule: crate::config::ResolvedScheduleConfig::default(),
                background: crate::config::ResolvedBackgroundConfig::default(),
                ..crate::session::SessionMaterials::for_test(store)
            },
            &crate::session::SessionCells {
                session_id: crate::session::SharedSessionId::default(),
                checklist: checklist_for_test(),
                background_tasks: crate::background::BackgroundTasks::default(),
                ..crate::session::SessionCells::for_test(
                    permission.clone(),
                    crate::workspace::cwd_for_test(),
                    crate::workspace::roots_for_test(),
                    Arc::new(crate::frontend::SilentFrontend),
                )
            },
            &crate::session::AgentOptions::for_test(),
        )
        .expect("registry builds");

        let render = |registry: &ToolRegistry| {
            serde_json::to_string(&registry.definitions_active(&[])).expect("definitions serialize")
        };

        let mut baseline: Option<(crate::permission::Permission, String)> = None;
        for level in [
            crate::permission::Permission::None,
            crate::permission::Permission::Read,
            crate::permission::Permission::Workspace,
            crate::permission::Permission::Unrestricted,
        ] {
            permission
                .try_set(level)
                .expect("every level is enabled here");
            let rendered = render(&registry);
            assert!(
                !rendered.is_empty() && rendered != "[]",
                "{level} produced no tools at all, so equality below would be vacuous"
            );
            match &baseline {
                None => baseline = Some((level, rendered)),
                Some((first_level, first)) => assert_eq!(
                    first, &rendered,
                    "the tools array differs between {first_level} and {level}, which invalidates \
                     the cached prompt prefix on every level toggle"
                ),
            }
        }
    }

    #[tokio::test]
    async fn definitions_active_exposes_loaded_deferred_tool() {
        // End-to-end: a successful tool_load call in the conversation promotes the named tool into
        // the active set on the next call.
        let registry = tool_registry_for_test().await;
        registry.register_deferred_fixture("fixture_alpha");
        registry.register_deferred_fixture("fixture_beta");

        let baseline = registry.definitions_active(&[]);
        assert!(!baseline.iter().any(|t| t.name == "fixture_alpha"));
        assert!(!baseline.iter().any(|t| t.name == "fixture_beta"));

        let messages = vec![
            load_tool_use("u1", "fixture_alpha"),
            tool_result("u1", "ok", false),
        ];
        let after_load = registry.definitions_active(&messages);
        assert!(after_load.iter().any(|t| t.name == "fixture_alpha"));
        // Append-only: the tools array gains exactly one entry.
        assert_eq!(after_load.len(), baseline.len() + 1);
        // Sibling deferred fixtures remain hidden.
        assert!(!after_load.iter().any(|t| t.name == "fixture_beta"));
    }

    #[tokio::test]
    async fn definitions_active_errored_load_stays_hidden() {
        // A tool_load call that ended in an error tool_result must NOT expose the deferred tool:
        // the model's parameter shape was wrong, so the schema was not delivered.
        let registry = tool_registry_for_test().await;
        registry.register_deferred_fixture("fixture_alpha");

        let messages = vec![
            load_tool_use("u1", "fixture_alpha"),
            tool_result("u1", "Error", true),
        ];
        let active = registry.definitions_active(&messages);
        assert!(!active.iter().any(|t| t.name == "fixture_alpha"));
    }

    /// What a request carries follows the wire. Appending: a loaded deferred tool joins the array.
    /// Declaring by reference: every deferred tool rides as a deferred definition whatever was
    /// loaded, and the discovery tools stay. Declaring by search: the same, minus the discovery
    /// tools, whose job the endpoint's own search item does.
    #[tokio::test]
    async fn the_active_set_follows_the_wire_s_delivery() {
        use crate::provider::DeferredToolDelivery;
        let registry = tool_registry_for_test().await;
        registry.register_deferred_fixture("fixture_alpha");
        let publish = |delivery: DeferredToolDelivery| {
            registry
                .profile
                .get()
                .expect("the test registry published its cells' profile")
                .store(&crate::provider::ResolvedProfile {
                    provider: std::sync::Arc::new(
                        crate::provider::mock::MockProvider::from_rounds(Vec::new())
                            .delivering(delivery),
                    ),
                    profile: "test".to_string(),
                    context_window: 1_000,
                    vision: true,
                });
        };
        let names = |definitions: &[ToolDefinition]| -> Vec<String> {
            definitions
                .iter()
                .map(|definition| definition.name.clone())
                .collect()
        };
        let loaded = vec!["fixture_alpha".to_string()];

        publish(DeferredToolDelivery::Appended);
        let appended = registry.active_tools(&loaded);
        assert!(names(&appended.definitions).contains(&"fixture_alpha".to_string()));
        assert!(appended.deferred_definitions.is_empty());

        publish(DeferredToolDelivery::Referenced);
        let referenced = registry.active_tools(&loaded);
        assert_eq!(names(&referenced.deferred_definitions), vec![
            "fixture_alpha"
        ]);
        let active = names(&referenced.definitions);
        assert!(!active.contains(&"fixture_alpha".to_string()), "{active:?}");
        assert!(
            active.iter().any(|name| name == LOAD_TOOL_NAME),
            "{active:?}"
        );
        assert!(
            active.iter().any(|name| name == SEARCH_TOOL_NAME),
            "{active:?}"
        );

        publish(DeferredToolDelivery::Searched);
        let searched = registry.active_tools(&[]);
        assert_eq!(names(&searched.deferred_definitions), vec!["fixture_alpha"]);
        let active = names(&searched.definitions);
        assert!(
            !active.iter().any(|name| is_discovery_tool(name)),
            "{active:?}"
        );
        // The catalog names what the request carries: the endpoint's item answers to
        // `tool_search`'s name, and nothing on that wire answers to `tool_load`'s.
        let catalog: Vec<String> = registry
            .tool_catalog()
            .into_iter()
            .map(|(name, ..)| name)
            .collect();
        assert!(
            catalog.iter().any(|name| name == SEARCH_TOOL_NAME),
            "{catalog:?}"
        );
        assert!(
            !catalog.iter().any(|name| name == LOAD_TOOL_NAME),
            "{catalog:?}"
        );
    }

    /// With nothing active the deferred tools are declared outright on a declaring wire, since the
    /// API refuses an array that is deferred through; the moment something active is there they go
    /// back to riding as deferred declarations.
    #[tokio::test]
    async fn an_all_deferred_set_is_declared_outright() {
        use crate::provider::DeferredToolDelivery;
        let registry = ToolRegistry::new();
        registry.register_deferred_fixture("mcp__probe__fetch");
        let published =
            crate::provider::PublishedProfile::detached(&crate::provider::ResolvedProfile {
                provider: std::sync::Arc::new(
                    crate::provider::mock::MockProvider::from_rounds(Vec::new())
                        .delivering(DeferredToolDelivery::Referenced),
                ),
                profile: "test".to_string(),
                context_window: 1_000,
                vision: true,
            });
        registry.publish_profile(&published);
        let alone = registry.active_tools(&[]);
        assert_eq!(
            alone
                .definitions
                .iter()
                .map(|d| d.name.as_str())
                .collect::<Vec<_>>(),
            vec!["mcp__probe__fetch"]
        );
        assert!(alone.deferred_definitions.is_empty());
        assert!(alone.schema_tokens > 0, "declared outright, it is priced");
        let loaded_too = registry.active_tools(&["mcp__probe__fetch".to_string()]);
        assert_eq!(
            loaded_too.schema_tokens, alone.schema_tokens,
            "a load of a tool declared outright prices it once"
        );
        // The catalog says what the array does: a tool declared with its schema is not deferred
        // to the model, so the context block lists it as available and asks for no load.
        let catalog = registry.tool_catalog();
        assert_eq!(catalog.len(), 1);
        assert!(
            !catalog[0].3,
            "declared outright, so not deferred: {catalog:?}"
        );

        // An appending wire declares such a set outright too: with no `tool_load` in the scope,
        // nothing else could ever make the tool callable.
        published.store(&crate::provider::ResolvedProfile {
            provider: std::sync::Arc::new(
                crate::provider::mock::MockProvider::from_rounds(Vec::new())
                    .delivering(DeferredToolDelivery::Appended),
            ),
            profile: "test".to_string(),
            context_window: 1_000,
            vision: true,
        });
        let appended = registry.active_tools(&[]);
        assert_eq!(
            appended
                .definitions
                .iter()
                .map(|d| d.name.as_str())
                .collect::<Vec<_>>(),
            vec!["mcp__probe__fetch"]
        );
        published.store(&crate::provider::ResolvedProfile {
            provider: std::sync::Arc::new(
                crate::provider::mock::MockProvider::from_rounds(Vec::new())
                    .delivering(DeferredToolDelivery::Referenced),
            ),
            profile: "test".to_string(),
            context_window: 1_000,
            vision: true,
        });

        registry.register_load_tool_for_test();
        let with_discovery = registry.active_tools(&[]);
        assert_eq!(
            with_discovery
                .deferred_definitions
                .iter()
                .map(|d| d.name.as_str())
                .collect::<Vec<_>>(),
            vec!["mcp__probe__fetch"]
        );
        assert!(!with_discovery.definitions.is_empty());
    }

    /// A name the wire refuses leaves the request, whether the tool is active, deferred, or
    /// deferred and loaded, and leaves the catalog with it; the rest of the registry is untouched,
    /// and the tool stays registered for the profile the session may move back to.
    #[tokio::test]
    async fn a_name_the_wire_refuses_leaves_the_request_and_the_catalog() {
        use crate::provider::DeferredToolDelivery;
        let registry = tool_registry_for_test().await;
        registry.register_deferred_fixture("mcp_probe");
        registry.register_deferred_fixture("fixture_alpha");
        let publish = |delivery: DeferredToolDelivery| {
            registry
                .profile
                .get()
                .expect("the test registry published its cells' profile")
                .store(&crate::provider::ResolvedProfile {
                    provider: std::sync::Arc::new(
                        crate::provider::mock::MockProvider::from_rounds(Vec::new())
                            .delivering(delivery)
                            .refusing("mcp_probe")
                            .refusing("file_read"),
                    ),
                    profile: "test".to_string(),
                    context_window: 1_000,
                    vision: true,
                });
        };
        let names = |definitions: &[ToolDefinition]| -> Vec<String> {
            definitions
                .iter()
                .map(|definition| definition.name.clone())
                .collect()
        };
        let loaded = vec!["mcp_probe".to_string(), "fixture_alpha".to_string()];

        publish(DeferredToolDelivery::Appended);
        let appended = names(&registry.active_tools(&loaded).definitions);
        assert!(!appended.contains(&"file_read".to_string()), "{appended:?}");
        assert!(!appended.contains(&"mcp_probe".to_string()), "{appended:?}");
        assert!(
            appended.contains(&"fixture_alpha".to_string()),
            "{appended:?}"
        );
        assert!(appended.contains(&"file_write".to_string()), "{appended:?}");

        publish(DeferredToolDelivery::Referenced);
        let referenced = registry.active_tools(&[]);
        assert_eq!(names(&referenced.deferred_definitions), vec![
            "fixture_alpha"
        ]);
        assert!(!names(&referenced.definitions).contains(&"file_read".to_string()));

        let catalog: Vec<String> = registry
            .tool_catalog()
            .into_iter()
            .map(|(name, ..)| name)
            .collect();
        assert!(!catalog.contains(&"mcp_probe".to_string()), "{catalog:?}");
        assert!(!catalog.contains(&"file_read".to_string()), "{catalog:?}");
        assert!(
            catalog.contains(&"fixture_alpha".to_string()),
            "{catalog:?}"
        );
        assert!(registry.get("mcp_probe").is_some(), "still registered");
        assert!(!registry.is_carried("mcp_probe"));
        assert!(registry.is_carried("fixture_alpha"));
    }

    /// The two discovery tools exist to find deferred tools: with none deferred they are two
    /// schemas the model can only call for nothing, and the moment one is deferred they ride.
    #[tokio::test]
    async fn the_discovery_tools_ride_only_while_something_is_deferred() {
        let registry = tool_registry_for_test().await;
        let names = |registry: &ToolRegistry| -> Vec<String> {
            registry
                .definitions_active(&[])
                .into_iter()
                .map(|definition| definition.name)
                .collect()
        };
        let bare = names(&registry);
        assert!(!bare.iter().any(|name| is_discovery_tool(name)), "{bare:?}");
        registry.register_deferred_fixture("fixture_alpha");
        let with_deferred = names(&registry);
        assert!(with_deferred.iter().any(|name| name == LOAD_TOOL_NAME));
        assert!(with_deferred.iter().any(|name| name == SEARCH_TOOL_NAME));
        assert!(!with_deferred.iter().any(|name| name == "fixture_alpha"));
    }

    /// A deferred tool the wire refuses is nothing for the discovery tools to find: with no other
    /// deferred tool they leave the array, as they leave the context block, whose catalog never
    /// listed the refused tool.
    #[tokio::test]
    async fn the_discovery_tools_do_not_ride_for_a_deferred_tool_the_wire_refuses() {
        let registry = tool_registry_for_test().await;
        registry.register_deferred_fixture("mcp_probe");
        registry
            .profile
            .get()
            .expect("the test registry published its cells' profile")
            .store(&crate::provider::ResolvedProfile {
                provider: std::sync::Arc::new(
                    crate::provider::mock::MockProvider::from_rounds(Vec::new())
                        .refusing("mcp_probe"),
                ),
                profile: "test".to_string(),
                context_window: 1_000,
                vision: true,
            });
        let active = registry.definitions_active(&[]);
        assert!(
            !active.iter().any(|tool| is_discovery_tool(&tool.name)),
            "{:?}",
            active.iter().map(|tool| &tool.name).collect::<Vec<_>>()
        );
    }

    /// The gauge prices what the window holds. A deferred declaration carries the background
    /// property so the expanded schema offers it, but until the model loads the tool neither the
    /// schema nor the property is in the window, so a declaring wire prices the same as an
    /// appending one with the same loads.
    #[tokio::test]
    async fn a_declaring_wire_prices_a_deferred_tool_only_once_loaded() {
        use crate::provider::DeferredToolDelivery;
        let registry = tool_registry_for_test().await;
        registry.enable_background();
        registry.register_deferred_fixture("mcp__kv__fetch");
        let publish = |delivery: DeferredToolDelivery| {
            registry
                .profile
                .get()
                .expect("the test registry published its cells' profile")
                .store(&crate::provider::ResolvedProfile {
                    provider: std::sync::Arc::new(
                        crate::provider::mock::MockProvider::from_rounds(Vec::new())
                            .delivering(delivery),
                    ),
                    profile: "test".to_string(),
                    context_window: 1_000,
                    vision: true,
                });
        };
        let loaded = vec!["mcp__kv__fetch".to_string()];
        publish(DeferredToolDelivery::Appended);
        let appended_bare = registry.active_tools(&[]).schema_tokens;
        let appended_loaded = registry.active_tools(&loaded).schema_tokens;
        publish(DeferredToolDelivery::Referenced);
        let referenced_bare = registry.active_tools(&[]);
        let referenced_loaded = registry.active_tools(&loaded);
        assert!(
            referenced_bare.deferred_definitions[0].parameters["properties"]
                .get("background")
                .is_some(),
            "the declaration offers the property"
        );
        assert_eq!(
            referenced_bare.schema_tokens, appended_bare,
            "nothing loaded"
        );
        assert_eq!(
            referenced_loaded.schema_tokens, appended_loaded,
            "one loaded"
        );
        assert!(appended_loaded > appended_bare);
    }

    #[tokio::test]
    async fn definitions_active_unknown_load_silently_dropped() {
        // tool_load was called for a tool that isn't registered. The scanner records the (errored)
        // result as not loaded, and even if it were loaded, the registry just doesn't contain a
        // tool by that name: no crash, no spurious entry.
        let registry = tool_registry_for_test().await;
        let messages = vec![
            load_tool_use("u1", "no_such_tool"),
            tool_result("u1", "Error: not registered", true),
        ];
        let active = registry.definitions_active(&messages);
        assert!(!active.iter().any(|t| t.name == "no_such_tool"));
    }

    #[tokio::test]
    async fn tool_catalog_covers_active_and_deferred() {
        let registry = tool_registry_for_test().await;
        registry.register_deferred_fixture("fixture_alpha");

        let entries = registry.tool_catalog();
        let names: std::collections::HashSet<_> = entries.iter().map(|(n, ..)| n.clone()).collect();
        assert!(names.contains("file_write"));
        assert!(names.contains("scratchpad_read"));
        assert!(names.contains("fixture_alpha"));

        let by_name: std::collections::HashMap<_, _> =
            entries.iter().map(|(n, _, _, d)| (n.clone(), *d)).collect();
        assert!(
            by_name["fixture_alpha"],
            "deferred fixture must be flagged deferred"
        );
        assert!(
            !by_name["scratchpad_read"],
            "scratchpad_read ships active and must not be flagged deferred"
        );
        assert!(!by_name["file_write"], "file_write is an active builtin");

        let required: std::collections::HashMap<_, _> =
            entries.iter().map(|(n, _, p, _)| (n.clone(), *p)).collect();
        assert_eq!(required["file_read"], Permission::Read);
        assert_eq!(required["file_write"], Permission::Workspace);
    }

    #[tokio::test]
    async fn scratchpad_tools_default_to_active() {
        // Every scratchpad tool ships active: an asymmetry where `scratchpad_write` is active but
        // its siblings are deferred behind `tool_load` trips agents up.
        let registry = tool_registry_for_test().await;
        let entries = registry.tool_catalog();
        for name in [
            "scratchpad_write",
            "scratchpad_read",
            "scratchpad_edit",
            "scratchpad_list",
            "scratchpad_merge",
            "scratchpad_delete",
            "scratchpad_rename",
            "scratchpad_load_file",
            "scratchpad_save_file",
        ] {
            let entry = entries
                .iter()
                .find(|(n, ..)| n == name)
                .unwrap_or_else(|| panic!("{name} missing from catalog"));
            assert!(
                !entry.3,
                "{name} must not be deferred (would force a tool_load round-trip)",
            );
        }
    }

    #[tokio::test]
    async fn tool_catalog_is_sorted() {
        let registry = tool_registry_for_test().await;
        let entries = registry.tool_catalog();
        let names: Vec<_> = entries.iter().map(|(n, ..)| n.clone()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "tool_catalog must return sorted entries");
    }

    /// `CHECKPOINT_TOOL_NAMES` is an allow-list read at compaction time, so an entry that is not a
    /// real built-in is silently inert rather than an error: `checkpoint_tools` just finds nothing
    /// under that name and the checkpoint quietly loses a capability.
    #[test]
    fn checkpoint_tool_names_are_sorted_and_real() {
        let mut sorted = CHECKPOINT_TOOL_NAMES.to_vec();
        sorted.sort_unstable();
        assert_eq!(CHECKPOINT_TOOL_NAMES, sorted.as_slice());

        let known: HashSet<&str> = BUILTIN_TOOL_NAMES.iter().copied().collect();
        for name in CHECKPOINT_TOOL_NAMES {
            assert!(known.contains(name), "{name} is not a built-in tool");
        }
    }

    /// The rule the allow-list encodes: a checkpoint can save, but not act. Asserted by name
    /// because the list is the only thing standing between an unattended turn and the shell.
    #[test]
    fn checkpoint_tool_names_exclude_acting_and_deleting() {
        for name in [
            "shell_execute",
            "file_write",
            "file_edit",
            "agent_spawn",
            "schedule_create",
            "web_fetch",
            "memory_delete",
            "scratchpad_delete",
            "checklist_add",
            "checklist_edit",
        ] {
            assert!(
                !CHECKPOINT_TOOL_NAMES.contains(&name),
                "{name} must not be reachable from a checkpoint turn"
            );
        }
        assert!(CHECKPOINT_TOOL_NAMES.contains(&"checklist_read"));
    }

    /// `context_replace` is always supplied, even against a registry holding nothing else, because
    /// a checkpoint with no way to submit could only ever fall through to the text tier.
    #[tokio::test]
    async fn checkpoint_tools_are_the_allow_list_plus_context_replace() {
        let registry = tool_registry_for_test().await;
        let slot = Arc::new(std::sync::Mutex::new(None));
        let names: HashSet<String> = registry
            .checkpoint_tools(Permission::Unrestricted, false, slot)
            .iter()
            .map(|tool| tool.definition().name)
            .collect();

        assert!(names.contains("context_replace"));
        assert!(names.contains("memory_write"));
        assert!(names.contains("file_read"));
        assert!(!names.contains("shell_execute"));
        assert!(!names.contains("file_write"));
        assert!(!names.contains("agent_spawn"));
    }

    /// A tool the user switched off in `[tools]` must stay off inside a checkpoint too: the list is
    /// read out of the live registry precisely so config keeps applying.
    #[tokio::test]
    async fn checkpoint_tools_respect_disabled_tools() {
        let registry = tool_registry_for_test_with_filter(BuiltinToolFilter::from_config(
            None,
            vec!["memory_write".to_string()],
            std::collections::HashMap::new(),
        ))
        .await;
        let slot = Arc::new(std::sync::Mutex::new(None));
        let names: HashSet<String> = registry
            .checkpoint_tools(Permission::Unrestricted, false, slot)
            .iter()
            .map(|tool| tool.definition().name)
            .collect();

        assert!(!names.contains("memory_write"));
        assert!(names.contains("context_replace"));
    }

    /// `context_replace` is deliberately absent from `BUILTIN_TOOL_NAMES`. Listing it would make
    /// `disabled_tools = ["context_replace"]` a valid-looking entry that silently downgrades every
    /// compaction to the fallback summarizer; left out, it warns as unrecognized instead.
    #[test]
    fn context_replace_is_not_a_configurable_builtin() {
        assert!(!BUILTIN_TOOL_NAMES.contains(&"context_replace"));
        assert!(BUILTIN_TOOL_NAMES.contains(&"context_check"));
        assert!(BUILTIN_TOOL_NAMES.contains(&"context_compact"));
    }

    #[test]
    fn builtin_filter_default_admits_everything() {
        let filter = BuiltinToolFilter::default();
        assert!(filter.admits("file_read"));
        assert!(filter.admits("file_write"));
        assert!(filter.admits("anything_else"));
    }

    #[test]
    fn builtin_filter_allow_list_restricts() {
        let allowed: HashSet<String> = ["file_read", "file_find"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let filter = BuiltinToolFilter {
            allowed: Some(allowed),
            ..Default::default()
        };
        assert!(filter.admits("file_read"));
        assert!(filter.admits("file_find"));
        assert!(!filter.admits("file_write"));
        assert!(!filter.admits("shell_execute"));
    }

    #[test]
    fn builtin_filter_block_list_wins_over_allow_list() {
        let allowed: HashSet<String> = ["file_read", "file_write"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let disabled: HashSet<String> = ["file_write"].iter().map(|s| s.to_string()).collect();
        let filter = BuiltinToolFilter {
            allowed: Some(allowed),
            disabled,
            ..Default::default()
        };
        assert!(filter.admits("file_read"));
        assert!(!filter.admits("file_write"));
    }

    #[test]
    fn builtin_filter_from_config_empty_allow_list_is_none() {
        let filter = BuiltinToolFilter::from_config(Some(Vec::new()), Vec::new(), HashMap::new());
        assert!(
            filter.allowed.is_none(),
            "empty allow-list should drop to None"
        );
        assert!(filter.admits("file_read"));
    }

    #[tokio::test]
    async fn registry_permission_override_applied() {
        let mut overrides = HashMap::new();
        overrides.insert("file_read".to_string(), Permission::Unrestricted);
        let filter = BuiltinToolFilter::from_config(None, Vec::new(), overrides);
        let registry = tool_registry_for_test_with_filter(filter).await;

        // Override wins over the Tool impl's hardcoded `Read`.
        assert_eq!(
            registry.required_permission_for("file_read"),
            Some(Permission::Unrestricted)
        );
        // Non-overridden tool returns its hardcoded level.
        assert_eq!(
            registry.required_permission_for("file_write"),
            Some(Permission::Workspace)
        );
        // Catalog must reflect the override too (the world-state block reads from it).
        let catalog = registry.tool_catalog();
        let read_file_required = catalog
            .iter()
            .find(|(name, ..)| name == "file_read")
            .map(|(_, _, permission, _)| *permission);
        assert_eq!(read_file_required, Some(Permission::Unrestricted));
    }

    #[tokio::test]
    async fn registry_permission_override_excludes_tool_from_lower_level() {
        let mut overrides = HashMap::new();
        overrides.insert("file_read".to_string(), Permission::Unrestricted);
        let filter = BuiltinToolFilter::from_config(None, Vec::new(), overrides);
        let registry = tool_registry_for_test_with_filter(filter).await;

        // At Read permission, file_read should now be excluded from the checkpoint
        // tools because the override raised it to `unrestricted`.
        let read_defs = registry.checkpoint_tools(
            Permission::Read,
            false,
            Arc::new(std::sync::Mutex::new(None)),
        );
        assert!(!read_defs.iter().any(|t| t.definition().name == "file_read"));

        let write_defs = registry.checkpoint_tools(
            Permission::Unrestricted,
            false,
            Arc::new(std::sync::Mutex::new(None)),
        );
        assert!(
            write_defs
                .iter()
                .any(|t| t.definition().name == "file_read")
        );
    }

    #[tokio::test]
    async fn subagent_registry_honors_filter() {
        let filter =
            BuiltinToolFilter::from_config(None, vec!["web_fetch".to_string()], HashMap::new());
        let sandbox_capability = crate::sandbox::detect();
        let backend_probe = crate::sandbox::BackendProbe::Ok(sandbox_capability.clone());
        let store = Store::for_test().await;
        let shared_session_id = crate::session::SharedSessionId::default();
        let registry = ToolRegistry::build_for_subagent(
            &crate::session::SessionMaterials {
                providers: std::sync::Arc::new(crate::provider::ProviderRegistry::for_test(
                    store.token_store(),
                    &["test-profile"],
                )),
                core: crate::session::CoreMaterials {
                    web_client: crate::config::WebClientConfig::default(),
                    sandbox_enabled: true,
                    sandbox_capability,
                    sandbox_backend: crate::config::SandboxBackend::Landlock,
                    backend_probe,
                    builtin_filter: filter,
                    write_locks: crate::workspace::WriteLocks::default(),
                    running_subagents: Default::default(),
                },
                skills: crate::skills::SkillCache::for_root(None),
                memories: crate::store::memory::MemoryStore::detached(),
                ..crate::session::SessionMaterials::for_test(store)
            },
            &crate::session::SessionCells {
                session_id: shared_session_id,
                checklist: checklist_for_test(),
                ..crate::session::SessionCells::for_test(
                    crate::permission::SharedPermission::new(
                        Permission::Read,
                        crate::permission::EnabledPermissions::ALL,
                    ),
                    crate::workspace::cwd_for_test(),
                    crate::workspace::roots_for_test(),
                    Arc::new(crate::frontend::SilentFrontend),
                )
            },
            crate::tools::registry::RegistryScope {
                tool_scope: ToolScope::default(),
                memory_access: crate::config::MemoryAccess::Write,
                parent_session_id: None,
                inherited_scratchpad_names: Vec::new(),
                context_ceiling_percent: 80,
                auto_compact: false,
            },
        )
        .expect("default web client config should build cleanly");
        assert!(registry.get("file_read").is_some());
        assert!(registry.get("web_fetch").is_none());
        assert!(registry.get("checklist_add").is_some());
        assert!(registry.get("agent_spawn").is_none());
    }

    /// Minimal named tool, for registry-level tests that only care whether a name is present.
    struct StubTool {
        name: String,
    }

    #[async_trait::async_trait]
    impl Tool for StubTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new(self.name.clone(), "stub".to_string(), serde_json::json!({}))
        }

        fn required_permission(&self) -> Permission {
            Permission::Read
        }

        async fn execute(
            &self,
            _input: serde_json::Value,
            _context: crate::tools::ToolContext,
        ) -> crate::error::Result<ToolOutput> {
            Ok(ToolOutput::text(String::new(), false))
        }
    }

    /// Test helper for the sub-agent registry: everything but the two knobs under test held at its
    /// default, so a case reads as "denials X, memory Y" rather than nineteen positional arguments.
    async fn subagent_registry(
        tool_scope: ToolScope,
        memory_access: crate::config::MemoryAccess,
    ) -> ToolRegistry {
        let sandbox_capability = crate::sandbox::detect();
        let backend_probe = crate::sandbox::BackendProbe::Ok(sandbox_capability.clone());
        let store = Store::for_test().await;
        ToolRegistry::build_for_subagent(
            &crate::session::SessionMaterials {
                providers: std::sync::Arc::new(crate::provider::ProviderRegistry::for_test(
                    store.token_store(),
                    &["test-profile"],
                )),
                core: crate::session::CoreMaterials {
                    web_client: crate::config::WebClientConfig::default(),
                    sandbox_enabled: true,
                    sandbox_capability,
                    sandbox_backend: crate::config::SandboxBackend::Landlock,
                    backend_probe,
                    builtin_filter: BuiltinToolFilter::default(),
                    write_locks: crate::workspace::WriteLocks::default(),
                    running_subagents: Default::default(),
                },
                skills: crate::skills::SkillCache::for_root(None),
                memories: crate::store::memory::MemoryStore::detached(),
                ..crate::session::SessionMaterials::for_test(store)
            },
            &crate::session::SessionCells {
                session_id: crate::session::SharedSessionId::default(),
                checklist: checklist_for_test(),
                ..crate::session::SessionCells::for_test(
                    crate::permission::SharedPermission::new(
                        Permission::Unrestricted,
                        crate::permission::EnabledPermissions::ALL,
                    ),
                    crate::workspace::cwd_for_test(),
                    crate::workspace::roots_for_test(),
                    Arc::new(crate::frontend::SilentFrontend),
                )
            },
            crate::tools::registry::RegistryScope {
                tool_scope,
                memory_access,
                parent_session_id: None,
                inherited_scratchpad_names: Vec::new(),
                context_ceiling_percent: 80,
                auto_compact: false,
            },
        )
        .expect("subagent registry should build")
    }

    const MEMORY_TOOLS: [&str; 4] = [
        "memory_read",
        "memory_search",
        "memory_write",
        "memory_delete",
    ];

    #[tokio::test]
    async fn subagent_memory_access_none_registers_no_memory_tools() {
        let registry =
            subagent_registry(ToolScope::default(), crate::config::MemoryAccess::None).await;
        for name in MEMORY_TOOLS {
            assert!(
                registry.get(name).is_none(),
                "memory = \"none\" must not register '{name}'"
            );
        }
        // The gate is memory-specific, not a blanket refusal.
        assert!(registry.get("file_read").is_some());
    }

    #[tokio::test]
    async fn subagent_memory_access_read_registers_exactly_the_readers() {
        let registry =
            subagent_registry(ToolScope::default(), crate::config::MemoryAccess::Read).await;
        assert!(registry.get("memory_read").is_some());
        assert!(registry.get("memory_search").is_some());
        assert!(
            registry.get("memory_write").is_none(),
            "read access must not let a worker edit the store the parent reasons from"
        );
        assert!(registry.get("memory_delete").is_none());
    }

    #[tokio::test]
    async fn subagent_memory_access_write_registers_all_four() {
        let registry =
            subagent_registry(ToolScope::default(), crate::config::MemoryAccess::Write).await;
        for name in MEMORY_TOOLS {
            assert!(registry.get(name).is_some(), "expected '{name}'");
        }
    }

    #[tokio::test]
    async fn subagent_denied_builtin_is_absent() {
        let registry = subagent_registry(
            ToolScope::new(Vec::new(), vec!["file_write".to_string()]),
            crate::config::MemoryAccess::Write,
        )
        .await;
        assert!(registry.get("file_write").is_none());
        assert!(
            registry.get("file_edit").is_some(),
            "denying one tool must not take its neighbors"
        );
    }

    /// A denied server's tools never reach the registry even when handed to it directly, which is
    /// the path a mid-run `tools/list_changed` takes.
    #[tokio::test]
    async fn subagent_denied_server_tools_are_dropped_on_replace() {
        let registry = subagent_registry(
            ToolScope::new(vec!["blocked".to_string()], Vec::new()),
            crate::config::MemoryAccess::Write,
        )
        .await;
        registry.replace_server_tools("blocked", vec![Arc::new(StubTool {
            name: "mcp__blocked__send".to_string(),
        })]);
        registry.replace_server_tools("allowed", vec![Arc::new(StubTool {
            name: "mcp__allowed__send".to_string(),
        })]);
        assert!(registry.get("mcp__blocked__send").is_none());
        assert!(registry.get("mcp__allowed__send").is_some());
    }

    /// The refresh door asks the same predicate as registration: a server outside the allow-list,
    /// and a tool outside the tool allow-list, never reach the registry through
    /// `tools/list_changed` either.
    #[tokio::test]
    async fn subagent_tools_outside_the_allow_lists_are_dropped_on_replace() {
        let registry = subagent_registry(
            // `mcp__other__send` is in the tool allow-list, so only `allow_servers` drops it.
            ToolScope::default().allowing(
                Some(vec!["allowed".to_string()]),
                Some(vec![
                    "mcp__allowed__send".to_string(),
                    "mcp__other__send".to_string(),
                ]),
            ),
            crate::config::MemoryAccess::Write,
        )
        .await;
        registry.replace_server_tools("other", vec![Arc::new(StubTool {
            name: "mcp__other__send".to_string(),
        })]);
        registry.replace_server_tools("allowed", vec![
            Arc::new(StubTool {
                name: "mcp__allowed__send".to_string(),
            }),
            Arc::new(StubTool {
                name: "mcp__allowed__delete".to_string(),
            }),
        ]);
        assert!(registry.get("mcp__other__send").is_none());
        assert!(registry.get("mcp__allowed__send").is_some());
        assert!(
            registry.get("mcp__allowed__delete").is_none(),
            "an allowed server's tool outside the tool allow-list"
        );
    }

    /// A scope only ever narrows across levels: denials from both sides accumulate and the
    /// allow-lists intersect, an absent one being everything, so a sub-agent cannot shed its
    /// parent's limits, nor name back a tool its parent lost, by spawning one more level down.
    #[test]
    fn a_nested_scope_intersects_allow_lists_and_unions_denials() {
        let parent = ToolScope::new(vec!["one".to_string()], vec!["alpha".to_string()]).allowing(
            None::<Vec<String>>,
            Some(vec!["file_read".to_string(), "file_find".to_string()]),
        );
        let child = parent.narrow(
            &ToolScope::new(vec!["two".to_string()], vec!["beta".to_string()]).allowing(
                Some(vec!["notion".to_string()]),
                Some(vec!["file_read".to_string(), "file_write".to_string()]),
            ),
        );
        assert_eq!(child.denied_server_list(), vec!["one", "two"]);
        assert_eq!(child.denied_tool_list(), vec!["alpha", "beta"]);
        assert_eq!(
            child.allowed_server_list(),
            Some(vec!["notion".to_string()])
        );
        assert_eq!(
            child.allowed_tool_list(),
            Some(vec!["file_read".to_string()]),
            "only what both allow"
        );
        assert!(
            parent.admits_tool("file_find") && !child.admits_tool("file_find"),
            "the child cannot keep what it did not name"
        );
        assert!(
            !parent.admits_tool("file_write") && !child.admits_tool("file_write"),
            "the child cannot name back what the parent lost"
        );
        assert!(!parent.admits_server("one") && parent.admits_server("two"));
        assert!(ToolScope::default().is_unrestricted() && !parent.is_unrestricted());
    }

    /// Withholding a server withholds its tools without naming each one, so the server lists and
    /// the tool lists cannot disagree about a server already ruled out or never let in.
    #[test]
    fn a_scope_withholding_a_server_withholds_its_tools() {
        let denied = ToolScope::new(vec!["notion".to_string()], Vec::new());
        assert!(!denied.admits_tool("mcp__notion__create_page"));
        assert!(denied.admits_tool("mcp__linear__create_issue"));
        assert!(denied.admits_tool("file_write"));
        let allowed =
            ToolScope::default().allowing(Some(vec!["linear".to_string()]), None::<Vec<String>>);
        assert!(!allowed.admits_tool("mcp__notion__create_page"));
        assert!(allowed.admits_tool("mcp__linear__create_issue"));
        assert!(
            allowed.admits_tool("file_write"),
            "an allowed-servers list leaves built-ins alone"
        );
    }

    /// An allow-list is exhaustive over what it names, a deny wins on overlap, and the two kinds of
    /// list compose: a tool must pass every list that has an opinion about it.
    #[test]
    fn a_scope_admits_only_what_its_allow_lists_name_and_never_what_a_deny_names() {
        let scope = ToolScope::new(Vec::new(), vec!["file_write".to_string()]).allowing(
            Some(vec!["notion".to_string()]),
            Some(vec![
                "file_read".to_string(),
                "file_write".to_string(),
                "mcp__notion__search".to_string(),
                "mcp__linear__search".to_string(),
            ]),
        );
        assert!(scope.admits_tool("file_read"));
        assert!(
            !scope.admits_tool("file_write"),
            "a deny wins over an allow"
        );
        assert!(!scope.admits_tool("file_find"), "not named, not seen");
        assert!(scope.admits_tool("mcp__notion__search"));
        assert!(
            !scope.admits_tool("mcp__linear__search"),
            "named in allow_tools but from a server outside allow_servers"
        );
        assert!(!scope.admits_server("linear") && scope.admits_server("notion"));
        assert!(!scope.is_unrestricted());
    }

    /// Every list of a scope is checked for entries that match nothing, each reported with the
    /// list it came from, so the call site and the config can name the key the reader wrote.
    #[test]
    fn the_stale_entries_of_a_scope_are_reported_with_their_list() {
        let scope = ToolScope::new(vec!["ghost".to_string()], vec!["file_wrtie".to_string()])
            .allowing(
                Some(vec!["notion".to_string(), "phantom".to_string()]),
                Some(vec![
                    "file_read".to_string(),
                    "mcp__notion__anything".to_string(),
                    "mcp__phantom__anything".to_string(),
                ]),
            );
        let mut stale = stale_scope_entries(&scope, &["notion".to_string()]);
        stale.sort();
        assert_eq!(stale, vec![
            (ScopeList::AllowedServers, "phantom".to_string()),
            (
                ScopeList::AllowedTools,
                "mcp__phantom__anything".to_string()
            ),
            (ScopeList::DeniedServers, "ghost".to_string()),
            (ScopeList::DeniedTools, "file_wrtie".to_string()),
        ]);
    }

    /// `[subagents]` restricts sub-agents, not the agent doing the delegating. The root registry's
    /// own denial set is empty, so `admits` is what keeps `disabled_tools = ["agent_spawn"]` (the
    /// natural way to write "sub-agents may not spawn sub-agents") from deleting the root agent's
    /// ability to delegate at all.
    #[tokio::test]
    async fn registry_admits_is_answered_by_the_registry_not_the_config() {
        let root =
            subagent_registry(ToolScope::default(), crate::config::MemoryAccess::Write).await;
        assert!(root.admits("agent_spawn"));
        assert!(root.admits("agent_delete"));

        let worker = subagent_registry(
            ToolScope::new(Vec::new(), vec!["agent_spawn".to_string()]),
            crate::config::MemoryAccess::Write,
        )
        .await;
        assert!(!worker.admits("agent_spawn"));
        assert!(worker.admits("agent_list"));
    }

    /// The MCP meta-tools are registered outside `register_builtin`, so `admits` is the only thing
    /// standing between them and a deny list that names them.
    #[tokio::test]
    async fn registry_admits_covers_the_directly_registered_tools() {
        let registry = subagent_registry(
            ToolScope::new(Vec::new(), vec!["mcp_resource_read".to_string()]),
            crate::config::MemoryAccess::Write,
        )
        .await;
        assert!(!registry.admits("mcp_resource_read"));
        assert!(registry.admits("mcp_resource_list"));
    }

    /// `allowed_tools` is exhaustive, and before the MCP meta-tools were filterable at all they
    /// registered regardless of it. Applying it to them now would delete seven tools from every
    /// install that has an allow-list, on upgrade, with nothing in the config naming them. The
    /// block-list half does apply, because naming a tool there has always meant "remove this one".
    #[tokio::test]
    async fn allowed_tools_does_not_reach_the_mcp_meta_tools() {
        // Asserted through `register_all` rather than the predicate alone: the regression this
        // guards was in the wiring (which predicate `register_all` calls), so a test that only
        // exercised `admits_infrastructure` would pass with the wiring wrong.
        let manager = crate::mcp::McpClientManager::prepare(
            &[crate::config::McpServerConfig::for_test("notes")],
            None,
            None,
            crate::mcp::McpClientContext::new(),
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("prepare");

        let allow_listed = ToolRegistry::new_with_filter(BuiltinToolFilter::from_config(
            Some(vec!["file_read".to_string()]),
            Vec::new(),
            HashMap::new(),
        ));
        mcp_resources::register_all(&allow_listed, std::sync::Arc::clone(&manager));
        assert!(
            allow_listed.get("mcp_resource_read").is_some(),
            "an exhaustive allowed_tools must not silently take the MCP meta-tools"
        );
        assert!(
            !allow_listed.admits("file_write"),
            "while still biting the tools it always did"
        );

        let block_listed = ToolRegistry::new_with_filter(BuiltinToolFilter::from_config(
            None,
            vec!["mcp_resource_read".to_string()],
            HashMap::new(),
        ));
        mcp_resources::register_all(&block_listed, manager);
        assert!(
            block_listed.get("mcp_resource_read").is_none(),
            "naming one explicitly does remove it"
        );
        assert!(block_listed.get("mcp_prompt_get").is_some());
    }

    /// Every meta-tool name must be a real one, and must be in `BUILTIN_TOOL_NAMES` too: the
    /// allow-list warning treats "not in `BUILTIN_TOOL_NAMES`" and "is a meta-tool" as different
    /// cases, so a name in neither list would fall through both.
    #[test]
    fn mcp_meta_tool_names_are_a_subset_of_the_builtins() {
        let known: HashSet<&str> = BUILTIN_TOOL_NAMES.iter().copied().collect();
        for name in MCP_META_TOOL_NAMES {
            assert!(
                known.contains(name),
                "BUILTIN_TOOL_NAMES is missing '{name}'"
            );
        }
        let mut sorted = MCP_META_TOOL_NAMES.to_vec();
        sorted.sort_unstable();
        assert_eq!(
            MCP_META_TOOL_NAMES,
            sorted.as_slice(),
            "keep the list sorted"
        );
        assert_eq!(MCP_META_TOOL_NAMES.len(), 7);
    }

    /// Every name a user may put in `[tools]` or `[subagents]` has to be in `BUILTIN_TOOL_NAMES`,
    /// or the stale-entry warning fires on a correct entry and invites them to "fix" it.
    #[test]
    fn builtin_tool_names_covers_the_deniable_families() {
        let known: HashSet<&str> = BUILTIN_TOOL_NAMES.iter().copied().collect();
        for name in [
            "conversation_search",
            "conversation_read",
            "schedule_create",
            "task_list",
            "mcp_resource_read",
            "mcp_resource_updates_list",
            "agent_followup",
            "agent_steer",
        ] {
            assert!(
                known.contains(name),
                "BUILTIN_TOOL_NAMES is missing '{name}'"
            );
        }
        let mut sorted = BUILTIN_TOOL_NAMES.to_vec();
        sorted.sort_unstable();
        assert_eq!(
            BUILTIN_TOOL_NAMES,
            sorted.as_slice(),
            "keep the list sorted"
        );
    }

    #[test]
    fn server_of_tool_splits_on_the_first_separator() {
        assert_eq!(server_of_tool("mcp__notion__create_page"), Some("notion"));
        // Tool names may themselves contain `__`; server names may not.
        assert_eq!(server_of_tool("mcp__notion__a__b"), Some("notion"));
        assert_eq!(server_of_tool("file_write"), None);
        assert_eq!(server_of_tool("mcp__malformed"), None);
    }

    #[test]
    fn builtin_tool_names_covers_canonical_set() {
        // Guard against forgetting to add a new built-in to the canonical list that drives
        // stale-entry warnings. Update this assertion deliberately when adding a tool in
        // register_core_tools.
        let names: HashSet<&str> = BUILTIN_TOOL_NAMES.iter().copied().collect();
        for expected in &[
            "file_read",
            "file_write",
            "file_edit",
            "file_find",
            "file_search",
            "shell_execute",
            "web_fetch",
            "checklist_add",
            "scratchpad_read",
            "scratchpad_write",
            "scratchpad_edit",
            "scratchpad_list",
            "scratchpad_delete",
            "skill_read",
            "skill_search",
            "skill_write",
            "skill_delete",
            "image_render",
            "agent_spawn",
            "tool_load",
        ] {
            assert!(
                names.contains(expected),
                "BUILTIN_TOOL_NAMES missing '{expected}'"
            );
        }
    }
}
