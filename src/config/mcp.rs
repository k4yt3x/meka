//! The `[mcp]` section: the servers, their transports and their auth, as `config.toml` states
//! them. Read by the MCP client and the `meka mcp` commands through `crate::config`.

use super::*;

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct McpConfig {
    /// Fallback permission for MCP tools when nothing more specific applies (no `tool_permissions`
    /// override, no server-level `permission`, no `readOnlyHint` from the server). If this is also
    /// unset the hardcoded fallback is `Unrestricted`, i.e. strict. Typed, like `[permissions]`,
    /// so a level meka does not have is refused where the file is parsed.
    pub(crate) default_permission: Option<Permission>,
    pub(crate) servers: Option<Vec<McpServerConfig>>,
    /// Default for each server's [`McpServerConfig::required`]. When true, every enabled server
    /// gates the turn; when false (the default) only servers that opt in with `required = true`
    /// do. A gated turn is refused outright rather than sent to the model.
    ///
    /// Defaults to false because whether a missing server should stop the turn is a property of
    /// that server, not of the installation: one that is essential on a workstation may be
    /// irrelevant inside a container that lacks its binary.
    pub(crate) default_required: Option<bool>,
    /// Per-turn cap on how long to wait for still-`Pending` MCP servers to settle before the
    /// readiness gate decides. Default `"3s"`; `"0s"` skips the wait.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) grace: Option<std::time::Duration>,
    /// Per-server wrap around connect + `initialize` + `list_tools`. A hung stdio spawn or slow
    /// HTTPS handshake can't stall the whole fleet past this bound. Default `"30s"`; `"0s"` is
    /// refused at startup.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) connect_timeout: Option<std::time::Duration>,
    /// How many stdio servers the startup connector spawns at once. Default 3; `0` is refused at
    /// startup.
    pub(crate) stdio_concurrency: Option<usize>,
    /// How many HTTP servers the startup connector connects at once. Default 20; `0` is refused
    /// at startup.
    pub(crate) http_concurrency: Option<usize>,
}
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct McpServerConfig {
    pub(crate) name: String,
    pub(crate) transport: McpTransport,
    pub(crate) command: Option<String>,
    pub(crate) args: Option<Vec<String>>,
    pub(crate) env: Option<std::collections::HashMap<String, String>>,
    pub(crate) url: Option<String>,
    pub(crate) headers: Option<std::collections::HashMap<String, String>>,
    /// Optional path to an executable that, when run, prints dynamic HTTP headers to stdout in
    /// `Name: Value\n` form. Merged over [`Self::headers`] (dynamic wins). Useful for SSO flows
    /// where bearer tokens rotate. The script is spawned with `MEKA_MCP_SERVER_NAME` and
    /// `MEKA_MCP_SERVER_URL` in its environment so one helper can drive multiple servers. Non-zero
    /// exit fails the connect.
    pub(crate) headers_helper: Option<String>,
    pub(crate) auth: Option<McpAuthConfig>,
    /// Server-wide permission override. Typed, like `[permissions]`, so a level meka does not have
    /// is refused where the file is parsed rather than when the server connects.
    pub(crate) permission: Option<Permission>,
    /// Optional allow-list of raw tool names (the server-advertised form, not the
    /// `mcp__<server>__<tool>` namespaced form). When set and non-empty, only these tools from
    /// this server are registered.
    pub(crate) allowed_tools: Option<Vec<String>>,
    /// Optional block-list of raw tool names. Applied after [`Self::allowed_tools`]. Tools listed
    /// here are never registered.
    pub(crate) disabled_tools: Option<Vec<String>>,
    /// Raw tool names (server-advertised, not the `mcp__<server>__<tool>` namespaced form) that
    /// should ship eager-loaded instead of deferred. Saves a `tool_load` round-trip and keeps the
    /// schema in the cacheable tools-array prefix. Names that don't match an advertised tool
    /// surface as a `warn!` via [`crate::mcp::warn_on_stale_tool_config`].
    pub(crate) eager_load_tools: Option<Vec<String>>,
    /// Optional per-tool permission overrides keyed by raw tool name. Beats the server-level
    /// `permission` and the server's `readOnlyHint` annotation when resolving a tool's required
    /// permission at registration time. Typed for the same reason [`Self::permission`] is.
    pub(crate) tool_permissions: Option<std::collections::HashMap<String, Permission>>,
    /// Whether this server's `readOnlyHint` annotation may classify a tool as `read`. Defaults to
    /// true, so a server that says a tool only reads is believed.
    ///
    /// The hint is asserted by the server, not verified by meka, and MCP tools execute in the
    /// server's own process with no sandbox. A server that advertises `readOnlyHint: true` for a
    /// tool that in fact writes therefore gets to write while meka sits at `read`. That is the
    /// reason this knob exists: setting it to `false` makes the hint advisory for display only, so
    /// the tool falls through to the strict `Unrestricted` fallback, and nothing from this server
    /// is reachable at `read` without an explicit [`Self::tool_permissions`] or
    /// [`Self::permission`] entry.
    ///
    /// A refused hint deliberately skips `[mcp].default_permission` on the way. That is a global
    /// convenience and this is a per-server audit decision, so the per-server one wins, the same
    /// direction the two overrides above already run. Falling through to it meant that with
    /// `default_permission = "read"` the knob changed nothing at all: the tool landed back on
    /// `Read` and dispatched unapproved at `--permission read`, which is precisely the outcome
    /// setting it to `false` was meant to prevent.
    ///
    /// Defaulting to true keeps existing configurations working and keeps the `read` level useful
    /// with well-behaved servers; the trade is that the `read` level's filesystem guarantee covers
    /// meka's built-in tools plus whichever MCP servers the user has chosen to trust.
    pub(crate) trust_read_only_hint: Option<bool>,
    /// When true, this server is skipped at startup: no process is spawned, no HTTP connect
    /// attempt is made. Lets users mute a flaky or in-development server without removing the
    /// entry. Unset means false.
    pub(crate) disabled: Option<bool>,
    /// Whether a turn may proceed while this server is unavailable. `None` inherits
    /// [`McpConfig::default_required`] (false by default), so a server is optional unless it says
    /// otherwise.
    ///
    /// The other half of the availability pair with [`Self::disabled`]: `disabled` says "don't
    /// even try", `required` says "if trying failed, stop the turn". Resolved once in
    /// [`crate::config::ResolvedConfig::resolve`], so every later consumer reads a plain `bool`.
    pub(crate) required: Option<bool>,
}

#[cfg(test)]
impl McpServerConfig {
    /// A bare HTTP server entry with nothing configured but its name.
    pub(crate) fn for_test(name: &str) -> Self {
        Self {
            name: name.to_string(),
            transport: McpTransport::Http,
            command: None,
            args: None,
            env: None,
            url: Some("https://example".to_string()),
            headers: None,
            headers_helper: None,
            auth: None,
            permission: None,
            allowed_tools: None,
            disabled_tools: None,
            eager_load_tools: None,
            tool_permissions: None,
            trust_read_only_hint: None,
            disabled: None,
            required: None,
        }
    }
}
/// How an MCP server is reached. One spelling, [`Self::name`], on `transport` in the file and on
/// `meka mcp add --transport`.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(try_from = "String", into = "String")]
pub(crate) enum McpTransport {
    Stdio,
    Http,
}
impl McpTransport {
    /// Every transport, in the order the names sort.
    pub(crate) const ALL: [McpTransport; 2] = [Self::Http, Self::Stdio];

    /// The one spelling `transport` takes.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Stdio => "stdio",
            Self::Http => "http",
        }
    }

    /// The names, joined for a refusal that lists what would have been accepted.
    pub(crate) fn supported() -> String {
        Self::ALL
            .iter()
            .map(|transport| transport.name())
            .collect::<Vec<_>>()
            .join(", ")
    }
}
impl std::fmt::Display for McpTransport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}
impl std::str::FromStr for McpTransport {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|transport| transport.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not a transport. Supported: {}",
                    Self::supported()
                )
            })
    }
}
impl TryFrom<String> for McpTransport {
    type Error = String;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        value.parse()
    }
}
impl From<McpTransport> for String {
    fn from(transport: McpTransport) -> Self {
        transport.name().to_string()
    }
}
/// How an HTTP MCP server authenticates. The secret itself is never here: it lives in
/// `mcp_credentials`, keyed by server name, exactly as an account's key lives in
/// `account_credentials`. This block says *which* flow to run and with what public parameters.
///
/// `deny_unknown_fields` so a key this does not model is refused rather than silently dropped,
/// which is the same strictness [`McpServerConfig`] already has. A secret quietly ignored is worse
/// than one refused: the connect fails later, somewhere else, for a reason that names nothing.
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum McpAuthConfig {
    ClientCredentials {
        client_id: String,
        scopes: Option<Vec<String>>,
        resource: Option<String>,
    },
    ClientCredentialsJwt {
        client_id: String,
        signing_key_path: String,
        signing_algorithm: Option<String>,
        scopes: Option<Vec<String>>,
        resource: Option<String>,
    },
    #[serde(rename = "oauth")]
    OAuth {
        client_id: Option<String>,
        scopes: Option<Vec<String>>,
        redirect_port: Option<u16>,
    },
}
impl std::fmt::Debug for McpServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpServerConfig")
            .field("name", &self.name)
            .field("transport", &self.transport)
            .field("command", &self.command)
            .field("args", &self.args)
            .field("env", &redact_map(&self.env))
            .field("url", &self.url)
            .field("headers", &redact_map(&self.headers))
            .field("headers_helper", &self.headers_helper)
            .field("auth", &self.auth)
            .field("permission", &self.permission)
            .field("allowed_tools", &self.allowed_tools)
            .field("disabled_tools", &self.disabled_tools)
            .field("eager_load_tools", &self.eager_load_tools)
            .field("tool_permissions", &self.tool_permissions)
            .field("trust_read_only_hint", &self.trust_read_only_hint)
            .field("disabled", &self.disabled)
            .field("required", &self.required)
            .finish()
    }
}
