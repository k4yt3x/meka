//! Which of a server's tools are registered, eagerly loaded, and at what level: pure config
//! policy over `[[mcp.servers]]`, with no client state behind it.

use super::*;

/// Decide whether a tool advertised by a server should be registered. Applies `allowed_tools`
/// (restrict-in, when set and non-empty) then `disabled_tools` (always-remove). Both fields can
/// coexist: the allow-list acts as a restriction, and the block-list subtracts from whatever
/// remains. A tool passes iff it survives both checks.
pub(crate) fn tool_is_allowed(server_config: &McpServerConfig, tool_raw_name: &str) -> bool {
    if let Some(allow) = server_config.allowed_tools.as_deref()
        && !allow.is_empty()
        && !allow.iter().any(|t| t == tool_raw_name)
    {
        return false;
    }
    if let Some(deny) = server_config.disabled_tools.as_deref()
        && deny.iter().any(|t| t == tool_raw_name)
    {
        return false;
    }
    true
}

/// Whether the given raw tool name is in this server's
/// [`eager_load_tools`][McpServerConfig::eager_load_tools] list. Mirrors [`tool_is_allowed`]'s
/// shape. When true, the registration sites skip `mark_deferred` so the tool ships in the cacheable
/// tools-array prefix from the first turn instead of after a `tool_load` round-trip.
pub(crate) fn tool_should_eager_load(server_config: &McpServerConfig, tool_raw_name: &str) -> bool {
    server_config
        .eager_load_tools
        .as_ref()
        .is_some_and(|list| list.iter().any(|n| n == tool_raw_name))
}

/// Warn once per entry in `allowed_tools` / `disabled_tools` / `eager_load_tools` /
/// `tool_permissions` that names nothing the server currently advertises, and once per tool that
/// is both disabled and eager-loaded. A warning rather than a failed connect, because tool lists
/// change between server releases and a hard error on every rename would be hostile.
pub(crate) fn warn_on_stale_tool_config(
    server_name: &str,
    server_config: &McpServerConfig,
    advertised: &std::collections::HashSet<&str>,
) {
    if let Some(allow) = server_config.allowed_tools.as_deref() {
        for name in allow {
            if !advertised.contains(name.as_str()) {
                tracing::warn!(
                    "MCP server '{server_name}': allowed_tools entry '{name}' names no advertised tool"
                );
            }
        }
    }
    if let Some(deny) = server_config.disabled_tools.as_deref() {
        for name in deny {
            if !advertised.contains(name.as_str()) {
                tracing::warn!(
                    "MCP server '{server_name}': disabled_tools entry '{name}' names no advertised tool"
                );
            }
        }
    }
    if let Some(eager) = server_config.eager_load_tools.as_deref() {
        let disabled = server_config.disabled_tools.as_deref().unwrap_or(&[]);
        for name in eager {
            if !advertised.contains(name.as_str()) {
                tracing::warn!(
                    "MCP server '{server_name}': eager_load_tools entry '{name}' names no advertised tool"
                );
            }
            if disabled.iter().any(|d| d == name) {
                tracing::warn!(
                    "MCP server '{server_name}': eager_load_tools entry '{name}' is also in disabled_tools, \
                     so it is never registered"
                );
            }
        }
    }
    if let Some(permissions) = server_config.tool_permissions.as_ref() {
        for key in permissions.keys() {
            if !advertised.contains(key.as_str()) {
                tracing::warn!(
                    "MCP server '{server_name}': tool_permissions key '{key}' names no advertised tool"
                );
            }
        }
    }
}

/// Resolve the required permission for a single MCP tool. Applies the
/// layered policy documented in `docs/book/src/configuration/config-file.md`:
///
/// 1. `server.tool_permissions[tool]`: per-tool user override.
/// 2. `server.permission`: server-level user override.
/// 3. `tool.annotations.readOnlyHint` advertised by the server: `true` → Read, `false` →
///    Unrestricted. The `true` half is skipped when the server sets `trust_read_only_hint = false`.
/// 4. `mcp.default_permission`: global fallback when no hint exists.
/// 5. Hardcoded `Unrestricted`: ultimate strict fallback.
///
/// User config at steps 1/2 always beats the server's hints. Hints beat the global fallback so a
/// `readOnlyHint = false` destructive tool isn't silently promoted to Read just because the user
/// opted into a lenient global default.
pub(crate) fn resolve_tool_permission(
    tool_raw_name: &str,
    tool_annotations: Option<&rmcp::model::ToolAnnotations>,
    server_config: &McpServerConfig,
    mcp_default: Option<Permission>,
) -> Permission {
    resolve_tool_permission_with_source(tool_raw_name, tool_annotations, server_config, mcp_default)
        .0
}

/// Identifies which step of the 5-step resolution chain produced a tool's permission. Used by `meka
/// mcp tools <name>` so users can see which knob is driving each tool's classification when editing
/// allow/block lists or per-tool overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PermissionSource {
    ToolOverride,
    ServerOverride,
    ReadOnlyHint,
    GlobalDefault,
    Fallback,
}

impl PermissionSource {
    /// Short human label matching the config keys users would edit.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::ToolOverride => "tool_permission",
            Self::ServerOverride => "server_permission",
            Self::ReadOnlyHint => "readOnlyHint",
            Self::GlobalDefault => "default_permission",
            Self::Fallback => "fallback",
        }
    }
}

/// A tool advertised by an MCP server, paired with the resolved permission and the source step of
/// the resolution chain. Returned by [`McpClientManager::list_advertised_tools`] and printed by
/// `meka mcp tools <server>`.
pub(crate) struct AdvertisedTool {
    /// Raw name as advertised by the server. Use this value in `allowed_tools` / `disabled_tools`
    /// / `tool_permissions` config.
    pub(crate) raw_name: String,
    /// Sanitized + truncated description (same pipeline as registered tools).
    pub(crate) description: String,
    /// Output of the 5-step permission resolution.
    pub(crate) resolved_permission: Permission,
    /// Which step of the chain won.
    pub(crate) permission_source: PermissionSource,
    /// `false` if currently filtered out by `allowed_tools` / `disabled_tools`, i.e. the agent
    /// would never see this tool.
    pub(crate) allowed: bool,
    /// The server advertised `readOnlyHint: true` and `trust_read_only_hint = false` withheld it,
    /// so resolution fell through to the steps below.
    ///
    /// Carried separately because [`Self::permission_source`] names only what *won*, and a
    /// declined hint by definition did not. Without it the one thing the setting exists to do
    /// is invisible at the one place a user checks it: `meka mcp tools` would show
    /// `default_permission` either way, so a server advertising no hint and a server whose
    /// hint was refused would read identically.
    pub(crate) read_only_hint_declined: bool,
}

/// Same resolution as [`resolve_tool_permission`] but also returns which step of the chain fired,
/// so `meka mcp tools` can show the user exactly why a given tool has its current permission.
pub(super) fn resolve_tool_permission_with_source(
    tool_raw_name: &str,
    tool_annotations: Option<&rmcp::model::ToolAnnotations>,
    server_config: &McpServerConfig,
    mcp_default: Option<Permission>,
) -> (Permission, PermissionSource) {
    // 1. Per-tool override.
    if let Some(permission) = server_config
        .tool_permissions
        .as_ref()
        .and_then(|map| map.get(tool_raw_name))
    {
        return (*permission, PermissionSource::ToolOverride);
    }
    // 2. Server-level override.
    if let Some(permission) = server_config.permission {
        return (permission, PermissionSource::ServerOverride);
    }
    // 3. Server-advertised readOnlyHint.
    //
    // The two directions are not symmetric, so they are gated differently. A hint of `false` only
    // ever *raises* the requirement to Unrestricted, so believing it costs nothing and it is always
    // honored. A hint of `true` *lowers* the requirement to Read, and that is the direction in
    // which a wrong or dishonest hint matters: MCP tools run in the server's own process with no
    // sandbox, so a tool wrongly classified Read can write the user's tree while meka sits at
    // `read`. `trust_read_only_hint = false` withholds exactly that, leaving the hint advisory for
    // display and dropping the tool through to the strict fallback, past the global default, for
    // the reason step 4 gives.
    let mut hint_declined = false;
    if let Some(annotations) = tool_annotations
        && let Some(hint) = annotations.read_only_hint
    {
        if !hint {
            return (Permission::Unrestricted, PermissionSource::ReadOnlyHint);
        }
        if server_config.trust_read_only_hint.unwrap_or(true) {
            return (Permission::Read, PermissionSource::ReadOnlyHint);
        }
        hint_declined = true;
    }
    // 4. Global [mcp].default_permission, but not for a hint this server was refused.
    //
    // A declined hint skips straight to the strict fallback, because otherwise the knob is
    // display-only in exactly the configuration where it matters most. `default_permission =
    // "read"` would send a refused `readOnlyHint: true` back to `Read` here, which is bit-for-bit
    // the outcome of trusting it: the tool registers at `Read` and dispatches unapproved at
    // `--permission read`. `"none"` is worse, since a required level of `None` is permitted at
    // every tier. Either way the user set a per-server flag saying "do not take this server's word
    // for it" and a global convenience setting would quietly take its word for it anyway.
    //
    // Per-server beats global, which is the direction the rest of this chain already runs: steps 1
    // and 2 are the per-server `tool_permissions` / `permission` overrides and they are checked
    // above. Those remain the way to put a distrusted server's tool back within reach of `read`.
    if !hint_declined && let Some(permission) = mcp_default {
        return (permission, PermissionSource::GlobalDefault);
    }
    // 5. Hardcoded strict fallback.
    //
    // `Unrestricted`, never `Workspace`, and this is load-bearing rather than incidental. An MCP
    // tool runs inside the server's own process, which meka does not sandbox and cannot confine to
    // a workspace root, so an unannotated tool reachable from `workspace` would make that level's
    // central promise false for every MCP user while looking exactly like it worked. The rung has
    // to be the one that promises no boundary, because that is the only one this tool honors.
    (Permission::Unrestricted, PermissionSource::Fallback)
}
