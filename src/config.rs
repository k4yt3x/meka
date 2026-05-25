//! Configuration: parses `~/.config/agsh/config.toml`, layers CLI overrides and environment
//! variables on top, and produces a [`ResolvedConfig`] that the rest of the binary consumes.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use serde::Deserialize;

use crate::{
    cli::Cli,
    permission::{EnabledPermissions, Permission},
    provider::AuthCredential,
    render::RenderMode,
};

/// In-memory shape of `config.toml`. Each top-level `[section]` deserializes into its own
/// sub-struct; missing sections fall back to `Default`. This is the raw deserialized form —
/// `resolve_config` merges it with CLI flags and env vars to produce a [`ResolvedConfig`].
#[derive(Debug, Deserialize, Default)]
pub struct ConfigFile {
    pub provider: Option<ProviderConfig>,
    pub display: Option<DisplayConfig>,
    pub web: Option<WebConfig>,
    pub shell: Option<ShellConfig>,
    pub session: Option<SessionConfig>,
    pub thinking: Option<ThinkingConfig>,
    pub mcp: Option<McpConfig>,
    pub prompt: Option<PromptConfig>,
    pub tools: Option<ToolsConfig>,
    pub permissions: Option<PermissionsConfig>,
    pub agent: Option<AgentConfig>,
}

/// `[agent]` table: agent-loop knobs that don't belong to any of the other domain-specific
/// sections. Currently only `max_turn_requests`.
#[derive(Debug, Deserialize, Default)]
pub struct AgentConfig {
    /// Cap on provider requests per turn. Surfaces as ACP `max_turn_requests` stop reason when
    /// exceeded. Default 100.
    pub max_turn_requests: Option<usize>,
}

/// `[permissions]` table: choose which modes are reachable at runtime and which mode the session
/// starts in. See `docs/book/src/usage/permissions.md`.
#[derive(Debug, Deserialize, Default)]
pub struct PermissionsConfig {
    pub default: Option<String>,
    pub enabled: Option<Vec<String>>,
}

/// Built-in tool filters, mirroring the per-server knobs on [`McpServerConfig`]. Applied at
/// registration time by [`crate::tools::ToolRegistry`].
#[derive(Debug, Deserialize, Default)]
pub struct ToolsConfig {
    pub allowed_tools: Option<Vec<String>>,
    pub disabled_tools: Option<Vec<String>>,
    pub tool_permissions: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize, Default)]
pub struct PromptConfig {
    pub instructions: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ThinkingConfig {
    pub enabled: Option<bool>,
    pub budget_tokens: Option<u64>,
    pub show_content: Option<bool>,
}

#[derive(Debug, Deserialize, Default)]
pub struct McpConfig {
    /// Fallback permission for MCP tools when nothing more specific applies (no `tool_permissions`
    /// override, no server-level `permission`, no `readOnlyHint` from the server). If this is also
    /// unset the hardcoded fallback is `Write` — i.e. strict.
    pub default_permission: Option<String>,
    pub servers: Option<Vec<McpServerConfig>>,
    /// When true (default), every turn is gated on all enabled MCP servers being `Connected`. If
    /// any are not, the turn is rejected with a shell-style error instead of sending the request.
    pub strict: Option<bool>,
    /// Per-turn cap on how long to wait for still-`Pending` MCP servers to settle before applying
    /// the strict check. Default: 3.
    pub grace_seconds: Option<u64>,
    /// Per-server wrap around connect + `initialize` + `list_tools`. A hung stdio spawn or slow
    /// HTTPS handshake can't stall the whole fleet past this bound. Default: 30.
    pub connect_timeout_seconds: Option<u64>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct McpServerConfig {
    pub name: String,
    pub transport: McpTransport,
    pub command: Option<String>,
    pub args: Option<Vec<String>>,
    pub env: Option<std::collections::HashMap<String, String>>,
    pub url: Option<String>,
    pub auth_token: Option<String>,
    pub headers: Option<std::collections::HashMap<String, String>>,
    /// Optional path to an executable that, when run, prints dynamic HTTP headers to stdout in
    /// `Name: Value\n` form. Merged over [`Self::headers`] (dynamic wins). Useful for SSO flows
    /// where bearer tokens rotate. The script is spawned with `AGSH_MCP_SERVER_NAME` and
    /// `AGSH_MCP_SERVER_URL` in its environment so one helper can drive multiple servers. Non-zero
    /// exit fails the connect.
    pub headers_helper: Option<String>,
    pub auth: Option<McpAuthConfig>,
    pub permission: Option<String>,
    /// Optional allow-list of raw tool names (the server-advertised form, not the
    /// `mcp__<server>__<tool>` namespaced form). When set and non-empty, only these tools from
    /// this server are registered.
    pub allowed_tools: Option<Vec<String>>,
    /// Optional block-list of raw tool names. Applied after [`Self::allowed_tools`] — tools listed
    /// here are never registered.
    pub disabled_tools: Option<Vec<String>>,
    /// Raw tool names (server-advertised, not the `mcp__<server>__<tool>` namespaced form) that
    /// should ship eager-loaded instead of deferred. Saves a `load_tool` round-trip and keeps the
    /// schema in the cacheable tools-array prefix. Names that don't match an advertised tool
    /// surface as a `warn!` via [`crate::mcp::warn_on_stale_tool_config`].
    pub eager_load_tools: Option<Vec<String>>,
    /// Optional per-tool permission overrides keyed by raw tool name. Beats the server-level
    /// `permission` and the server's `readOnlyHint` annotation when resolving a tool's required
    /// permission at registration time.
    pub tool_permissions: Option<std::collections::HashMap<String, String>>,
    /// Allow this server to issue `sampling/createMessage` requests. When false (default), any
    /// such request is rejected with `METHOD_NOT_FOUND`. Use with caution: sampling lets the
    /// server inject arbitrary messages into your LLM context and spend your provider quota.
    #[serde(default)]
    pub sampling: bool,
    /// Cap on the number of sampling calls this server may issue per agsh session. Only meaningful
    /// when `sampling = true`. Default: 10.
    pub sampling_limit: Option<u32>,
    /// When true, this server is skipped at startup — no process is spawned, no HTTP connect
    /// attempt is made. Lets users mute a flaky or in-development server without removing the
    /// entry.
    #[serde(default)]
    pub disabled: bool,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum McpTransport {
    Stdio,
    Http,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpAuthConfig {
    ClientCredentials {
        client_id: String,
        client_secret: String,
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
        client_secret: Option<String>,
        scopes: Option<Vec<String>>,
        redirect_port: Option<u16>,
    },
}

#[derive(Debug, Deserialize, Default)]
pub struct DisplayConfig {
    pub newline_before_prompt: Option<bool>,
    pub newline_after_prompt: Option<bool>,
    pub show_session_id_on_create: Option<bool>,
    pub show_session_id_on_exit: Option<bool>,
    pub show_path_in_prompt: Option<bool>,
    pub show_token_usage: Option<bool>,
    pub render_mode: Option<RenderMode>,
    /// Style applied to the REPL input buffer so submitted prompts stand out in scrollback. Parsed
    /// by [`parse_input_style`]. Accepts `bold`, `dim`, `none`, or a colour name (`cyan`,
    /// `yellow`, …).
    pub input_style: Option<String>,
    /// When set to `Some(N)` with `N > 0`, resuming a session reprints the last `N` turns (user
    /// prompts plus the agent's response, styled like the live REPL) instead of just the last
    /// assistant message. Unset preserves the legacy behaviour.
    pub resume_show_recent: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
pub struct WebConfig {
    pub user_agent: Option<String>,
    pub request_timeout_seconds: Option<u64>,
    pub connect_timeout_seconds: Option<u64>,
    pub read_timeout_seconds: Option<u64>,
    /// Max number of redirects reqwest will follow. `0` disables redirects entirely. Default:
    /// `10`.
    pub max_redirects: Option<u64>,
    /// Proxy URL. Accepts `http://…`, `https://…`, `socks5://…`, `socks5h://…`. The literal string
    /// `"none"` explicitly disables env-var auto-detection (overriding `HTTP_PROXY` etc.). Unset
    /// honours the env vars.
    pub proxy: Option<String>,
    /// Path to a PEM file containing one or more root CAs to trust on top of the system trust
    /// store. Used for corporate MITM proxies or self-signed internal services.
    pub ca_cert_file: Option<String>,
    /// Reject plain `http://` URLs — only `https://` allowed.
    pub https_only: Option<bool>,
    /// Minimum TLS version: `"1.0"`, `"1.1"`, `"1.2"`, or `"1.3"`. Anything else logs a warn and
    /// falls back to reqwest's default.
    pub min_tls_version: Option<String>,
    /// DANGER: disable TLS certificate validation entirely. Allows MITM; only use against trusted
    /// local dev servers.
    pub danger_accept_invalid_certs: Option<bool>,
    /// DANGER: accept certificates whose hostname doesn't match. Allows MITM; only use against
    /// trusted local dev servers.
    pub danger_accept_invalid_hostnames: Option<bool>,
}

/// Minimum TLS version accepted by the web-tools client. Normalised from `[web].min_tls_version` so
/// the rest of the crate doesn't pass free-form strings around.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinTlsVersion {
    V1_0,
    V1_1,
    V1_2,
    V1_3,
}

impl MinTlsVersion {
    /// Parse the config-file string form. Returns `None` on unknown input; callers are expected to
    /// log and fall through to the reqwest backend default.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim() {
            "1.0" => Some(Self::V1_0),
            "1.1" => Some(Self::V1_1),
            "1.2" => Some(Self::V1_2),
            "1.3" => Some(Self::V1_3),
            _ => None,
        }
    }
}

/// Fully-resolved web-tools HTTP client configuration. Carried on [`ResolvedConfig`] and consumed
/// by `crate::tools::web::build_web_client` at registry-build time.
#[derive(Debug, Clone)]
pub struct WebClientConfig {
    pub user_agent: String,
    pub request_timeout: std::time::Duration,
    pub connect_timeout: Option<std::time::Duration>,
    pub read_timeout: Option<std::time::Duration>,
    /// `0` means "no redirects" (`Policy::none`); any other value becomes `Policy::limited(n)`.
    pub max_redirects: usize,
    pub proxy: Option<String>,
    pub ca_cert_file: Option<std::path::PathBuf>,
    pub https_only: bool,
    pub min_tls_version: Option<MinTlsVersion>,
    pub danger_accept_invalid_certs: bool,
    pub danger_accept_invalid_hostnames: bool,
}

impl Default for WebClientConfig {
    fn default() -> Self {
        Self {
            user_agent: DEFAULT_WEB_USER_AGENT.to_string(),
            request_timeout: std::time::Duration::from_secs(30),
            connect_timeout: None,
            read_timeout: None,
            max_redirects: 10,
            proxy: None,
            ca_cert_file: None,
            https_only: false,
            min_tls_version: None,
            danger_accept_invalid_certs: false,
            danger_accept_invalid_hostnames: false,
        }
    }
}

impl WebClientConfig {
    /// Build a resolved config from the TOML section. Invalid `min_tls_version` strings log a warn
    /// and fall through to reqwest's default rather than aborting startup.
    pub fn from_file(file: &WebConfig) -> Self {
        let user_agent = file
            .user_agent
            .clone()
            .unwrap_or_else(|| DEFAULT_WEB_USER_AGENT.to_string());

        let min_tls_version =
            file.min_tls_version
                .as_deref()
                .and_then(|raw| match MinTlsVersion::parse(raw) {
                    Some(v) => Some(v),
                    None => {
                        tracing::warn!(
                            "ignoring unknown [web].min_tls_version '{}' \
                         (expected '1.0', '1.1', '1.2', or '1.3')",
                            raw
                        );
                        None
                    }
                });

        Self {
            user_agent,
            request_timeout: file
                .request_timeout_seconds
                .filter(|n| *n > 0)
                .map(std::time::Duration::from_secs)
                .unwrap_or_else(|| std::time::Duration::from_secs(30)),
            connect_timeout: file
                .connect_timeout_seconds
                .filter(|n| *n > 0)
                .map(std::time::Duration::from_secs),
            read_timeout: file
                .read_timeout_seconds
                .filter(|n| *n > 0)
                .map(std::time::Duration::from_secs),
            max_redirects: file
                .max_redirects
                .map(|n| usize::try_from(n).unwrap_or(usize::MAX))
                .unwrap_or(10),
            proxy: file.proxy.clone(),
            ca_cert_file: file.ca_cert_file.clone().map(std::path::PathBuf::from),
            https_only: file.https_only.unwrap_or(false),
            min_tls_version,
            danger_accept_invalid_certs: file.danger_accept_invalid_certs.unwrap_or(false),
            danger_accept_invalid_hostnames: file.danger_accept_invalid_hostnames.unwrap_or(false),
        }
    }
}

/// Default UA for the web tools when `[web].user_agent` is unset. Kept in sync with what real
/// Chrome emits so anti-bot filters don't single out agsh by default.
pub const DEFAULT_WEB_USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36";

/// Max conversation messages kept in the per-turn API window by default.
const DEFAULT_CONTEXT_MESSAGES: usize = 200;
/// Default session-retention window, in days.
const DEFAULT_RETENTION_DAYS: u64 = 90;
/// Default cap on total session storage, in bytes (50 MiB).
const DEFAULT_MAX_STORAGE_BYTES: u64 = 50 * 1024 * 1024;
/// Default extended-thinking token budget.
const DEFAULT_THINKING_BUDGET_TOKENS: u64 = 16_000;

#[derive(Debug, Deserialize, Default)]
pub struct ShellConfig {
    pub sandbox: Option<bool>,
    /// Linux-only choice between `"landlock"` and `"bubblewrap"`. When omitted, the resolver
    /// auto-picks bubblewrap if available and falls back to landlock with a one-shot warning (see
    /// `src/sandbox.rs` and `Warn 2` in `warn_if_sandbox_issues`).
    pub sandbox_backend: Option<SandboxBackend>,
}

/// Linux sandbox backend. Silently ignored on macOS and Windows
/// (sandbox-exec and Low-integrity respectively are the only options
/// on those platforms). The absence of a default impl is intentional:
/// an unset value is meaningful and triggers auto-resolution in
/// [`ResolvedConfig::from_cli`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SandboxBackend {
    Landlock,
    Bubblewrap,
}

impl SandboxBackend {
    /// Brand-cased name for user-facing prose (logs, errors). Also the `Display` impl's output.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Landlock => "Landlock",
            Self::Bubblewrap => "Bubblewrap",
        }
    }
}

impl std::fmt::Display for SandboxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.display_name())
    }
}

#[derive(Debug, Deserialize, Default)]
pub struct SessionConfig {
    pub context_messages: Option<usize>,
    pub retention_days: Option<u64>,
    pub max_storage_bytes: Option<u64>,
    pub auto_compact: Option<bool>,
    pub context_window: Option<u64>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ProviderConfig {
    pub name: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub oauth_token: Option<String>,
    pub oauth_token_url: Option<String>,
    pub base_url: Option<String>,
    pub reasoning_effort: Option<String>,
    pub device_id: Option<String>,
    /// `claude-oauth` only: value emitted as `output_config.effort`. Mirrors Claude Code's effort
    /// knob — see `temp/claude-code/src/utils/effort.ts`. Accepted values: `"low" | "medium" |
    /// "high"`. Defaults to `"high"`.
    pub effort: Option<String>,
    /// `claude-oauth` only: when true, agsh sends the `redact-thinking-2026-02-12` beta header so
    /// the server returns `redacted_thinking` blocks instead of full thinking summaries (saves
    /// bandwidth, but the redacted payloads can't be replayed back to the server in multi-turn
    /// conversations). Defaults to false.
    pub redact_thinking: Option<bool>,
}

/// Merged + validated runtime view of [`ConfigFile`], CLI flags, and env vars. This is what the
/// rest of the binary reads — `ConfigFile` is for deserialization only. Resolution lives in
/// `resolve_config` (Linux) and the non-Linux variant below it.
#[derive(Debug)]
pub struct ResolvedConfig {
    pub provider_name: Option<String>,
    pub model: Option<String>,
    pub auth_credential: Option<AuthCredential>,
    pub base_url: Option<String>,
    pub client_id: Option<String>,
    pub oauth_token_url: Option<String>,
    pub permission: Permission,
    pub enabled_permissions: EnabledPermissions,
    pub streaming: bool,
    /// Cap on provider requests per turn. When the agent loop makes this many calls without
    /// finishing, the turn resolves as the ACP `max_turn_requests` stop reason. Defaults to 100 —
    /// generous for legitimate tool chains, low enough to bound a runaway model.
    pub max_turn_requests: usize,
    pub continue_session: Option<String>,
    pub prompt: Option<String>,
    pub oneshot: bool,
    pub newline_before_prompt: bool,
    pub newline_after_prompt: bool,
    pub show_session_id_on_create: bool,
    pub show_session_id_on_exit: bool,
    pub show_path_in_prompt: bool,
    pub show_token_usage: bool,
    pub resume_show_recent: Option<usize>,
    pub web_client: WebClientConfig,
    pub sandbox: bool,
    /// Resolved Linux sandbox backend. Auto-picked at startup when `[shell].sandbox_backend` was
    /// not set: prefers bubblewrap, falls back to landlock. Silently `Landlock` on macOS / Windows
    /// (those platforms have their own backend and ignore the field).
    pub sandbox_backend: SandboxBackend,
    /// True when [`Self::sandbox_backend`] was auto-resolved (i.e. the user did not pin a value in
    /// `[shell].sandbox_backend`). Used to gate the "stronger sandbox available; install bwrap"
    /// startup warn — we don't want to nag users who explicitly chose landlock.
    pub sandbox_auto_resolved: bool,
    /// Cached probe of the resolved backend. Consulted by `warn_if_sandbox_issues` and by the lazy
    /// hard-error path in `src/tools/shell.rs` when read-mode `execute_command` is invoked.
    pub backend_probe: crate::sandbox::BackendProbe,
    pub render_mode: RenderMode,
    pub context_messages: Option<usize>,
    pub retention_days: Option<u64>,
    pub max_storage_bytes: Option<u64>,
    pub thinking_enabled: bool,
    pub thinking_budget_tokens: u64,
    pub thinking_show_content: bool,
    pub reasoning_effort: Option<String>,
    /// Stable per-device identifier for `claude-oauth`'s `metadata.user_id`. Empty string for
    /// non-`claude-oauth` providers (the value is ignored downstream).
    pub device_id: String,
    /// `claude-oauth` `output_config.effort` value. Always one of `"low" | "medium" | "high"`
    /// after `validate()`. Default `"high"`.
    pub effort: String,
    /// `claude-oauth`: when true, request `redacted_thinking` blocks via
    /// `redact-thinking-2026-02-12` beta. Default false.
    pub redact_thinking: bool,
    pub auto_compact: bool,
    pub context_window: Option<u64>,
    pub mcp_servers: Vec<McpServerConfig>,
    /// Parsed [`Permission`] from `[mcp].default_permission`, carried so per-turn tool-permission
    /// resolution in `src/mcp.rs` doesn't have to re-read the config file. `None` means "no
    /// `[mcp]` default configured" — resolution falls through to the hardcoded Write.
    pub mcp_default_permission: Option<Permission>,
    pub user_instructions: Option<String>,
    pub builtin_allowed_tools: Option<Vec<String>>,
    pub builtin_disabled_tools: Vec<String>,
    pub builtin_tool_permissions: HashMap<String, Permission>,
    pub input_style: nu_ansi_term::Style,
    /// Per-turn MCP readiness gate. When true, a turn is rejected if any enabled server isn't
    /// `Connected` after `mcp_grace`.
    pub mcp_strict: bool,
    /// First-turn await cap for still-connecting MCP servers.
    pub mcp_grace: std::time::Duration,
    /// Per-server connect+initialize timeout.
    pub mcp_connect_timeout: std::time::Duration,
}

/// Default input style: bold, white-ish foreground, slate-blue background. Uses truecolor RGB (not
/// palette indices or named colours) so the visual is consistent across terminals that remap the
/// standard 16 colours to match their theme.
pub fn default_input_style() -> nu_ansi_term::Style {
    use nu_ansi_term::{Color, Style};
    Style::new()
        .bold()
        .fg(Color::Rgb(240, 240, 240))
        .on(Color::Rgb(55, 75, 110))
}

/// Parse a `[display].input_style` value. `"default"` (or unset) yields [`default_input_style`];
/// `"none"` yields no styling; simple keywords pick a single colour or attribute. Unknown keywords
/// warn and fall back to the default so a typo doesn't lose the session to a panic.
pub fn parse_input_style(raw: &str) -> nu_ansi_term::Style {
    use nu_ansi_term::{Color, Style};
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "default" => default_input_style(),
        "none" => Style::default(),
        "reverse" => Style::new().reverse(),
        "bold" => Style::new().bold(),
        "dim" => Style::new().dimmed(),
        "italic" => Style::new().italic(),
        "underline" => Style::new().underline(),
        "black" => Style::new().fg(Color::Black),
        "red" => Style::new().fg(Color::Red),
        "green" => Style::new().fg(Color::Green),
        "yellow" => Style::new().fg(Color::Yellow),
        "blue" => Style::new().fg(Color::Blue),
        "magenta" | "purple" => Style::new().fg(Color::Magenta),
        "cyan" => Style::new().fg(Color::Cyan),
        "white" => Style::new().fg(Color::White),
        other => {
            tracing::warn!(
                "ignoring unknown [display].input_style '{}' (expected \
                 default, none, reverse, bold, dim, italic, underline, or a colour name)",
                other
            );
            default_input_style()
        }
    }
}

/// Returns the agsh config directory (the directory that contains `config.toml` and `skills/`).
/// Honours the `AGSH_CONFIG_DIR` env var — used by tests for per-run isolation and by power users
/// who want a non-standard location — before falling back to the platform-native
/// `dirs::config_dir().join("agsh")`. The env-var route is the only reliable way to isolate state
/// on macOS and Windows, where `dirs::config_dir()` doesn't honour `XDG_CONFIG_HOME`.
pub fn agsh_config_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("AGSH_CONFIG_DIR") {
        return Some(PathBuf::from(dir));
    }
    dirs::config_dir().map(|directory| directory.join("agsh"))
}

pub(crate) fn config_file_path() -> Option<PathBuf> {
    agsh_config_dir().map(|dir| dir.join("config.toml"))
}

pub(crate) fn config_file_exists() -> bool {
    config_file_path().is_some_and(|path| path.exists())
}

/// Write `content` to `path` atomically: serialise to `<path>.tmp` in the same directory,
/// `sync_all` the fd, then `rename` over the target. Also creates the parent directory (0700 on
/// Unix) and chmods the final file to 0600 on Unix so `auth_token` / OAuth-derived secrets aren't
/// world-readable regardless of the user's umask.
pub(crate) fn write_config_atomic(path: &Path, content: &str) -> std::io::Result<()> {
    use std::io::Write as _;

    if let Some(parent) = path.parent() {
        // Create newly-missing parents already at 0700 to avoid the umask window left by
        // `create_dir_all` followed by `set_permissions`. `DirBuilderExt::mode` passes the mode
        // straight to `mkdir(2)`.
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            std::fs::DirBuilder::new()
                .mode(0o700)
                .recursive(true)
                .create(parent)?;
        }
        #[cfg(not(unix))]
        std::fs::create_dir_all(parent)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            // Pre-existing dirs may have a different mode (e.g. user pre-created `~/.config` at
            // 0755). Best-effort tighten to 0700; failure here gets a warning rather than aborting
            // the write.
            if let Ok(metadata) = std::fs::metadata(parent) {
                let mut perms = metadata.permissions();
                if perms.mode() & 0o777 != 0o700 {
                    perms.set_mode(0o700);
                    if let Err(error) = std::fs::set_permissions(parent, perms) {
                        tracing::warn!(
                            "failed to tighten '{}' to 0700: {}",
                            parent.display(),
                            error
                        );
                    }
                }
            }
        }
    }

    let file_name = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "config path has no file name",
        )
    })?;
    let mut tmp_name = file_name.to_os_string();
    tmp_name.push(".tmp");
    let tmp_path = path.with_file_name(tmp_name);

    // Create the tmp file with restrictive perms on Unix before any bytes land on disk, so a
    // concurrent reader never sees the partial content with a looser mode.
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&tmp_path)?;
    file.write_all(content.as_bytes())?;
    file.sync_all()?;
    drop(file);

    std::fs::rename(&tmp_path, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp_path);
    })
}

/// Write a freshly initialized `config.toml` from the setup wizard. No `[shell]` table is emitted —
/// the sandbox backend auto-resolves at runtime (bubblewrap preferred, landlock fallback; see
/// [`resolve_sandbox_backend`]), so a pinned value is only ever needed when the user wants to
/// override that default by hand.
pub(crate) fn write_config_file(
    provider_name: &str,
    model: &str,
    api_key: Option<&str>,
    base_url: Option<&str>,
) -> std::io::Result<()> {
    let path = config_file_path().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "could not determine config directory",
        )
    })?;

    let mut provider_table = toml::map::Map::new();
    provider_table.insert(
        "name".to_string(),
        toml::Value::String(provider_name.to_string()),
    );
    provider_table.insert("model".to_string(), toml::Value::String(model.to_string()));
    if let Some(key) = api_key {
        provider_table.insert("api_key".to_string(), toml::Value::String(key.to_string()));
    }
    if let Some(url) = base_url {
        provider_table.insert("base_url".to_string(), toml::Value::String(url.to_string()));
    }

    let mut root = toml::map::Map::new();
    root.insert("provider".to_string(), toml::Value::Table(provider_table));

    let content = toml::to_string_pretty(&root)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string()))?;

    write_config_atomic(&path, &content)
}

fn load_config_file() -> ConfigFile {
    let Some(path) = config_file_path() else {
        return ConfigFile::default();
    };

    match std::fs::read_to_string(&path) {
        Ok(contents) => match toml::from_str(&contents) {
            Ok(config) => config,
            Err(error) => {
                tracing::warn!("failed to parse config file {}: {}", path.display(), error);
                ConfigFile::default()
            }
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => ConfigFile::default(),
        Err(error) => {
            tracing::warn!("failed to read config file {}: {}", path.display(), error);
            ConfigFile::default()
        }
    }
}

/// Resolve the runtime permission level and the set of enabled modes from the layered config
/// sources. CLI > env > config file > built-in defaults. Invalid entries warn and are dropped;
/// out-of-set overrides (e.g. `--permission ask` when `ask` is disabled) warn and clamp to the
/// configured default rather than refusing to start, mirroring the `[tools.tool_permissions]`
/// warn-and-skip pattern.
fn resolve_permission(
    cli_permission: Option<Permission>,
    env_permission: Option<&str>,
    file_default: Option<&str>,
    file_enabled: Option<&[String]>,
) -> (Permission, EnabledPermissions) {
    let enabled = match file_enabled {
        Some(list) => {
            let parsed: Vec<Permission> = list
                .iter()
                .filter_map(|raw| match raw.parse::<Permission>() {
                    Ok(mode) => Some(mode),
                    Err(_) => {
                        tracing::warn!(
                            "ignoring invalid [permissions].enabled entry '{}' \
                             (expected none, read, ask, or write)",
                            raw
                        );
                        None
                    }
                })
                .collect();
            match EnabledPermissions::from_modes(parsed) {
                Some(set) => set,
                None => {
                    tracing::warn!(
                        "[permissions].enabled was empty after filtering; falling back \
                         to defaults (none, read, write)"
                    );
                    EnabledPermissions::DEFAULT
                }
            }
        }
        None => EnabledPermissions::DEFAULT,
    };

    let configured_default = file_default.and_then(|raw| match raw.parse::<Permission>() {
        Ok(mode) => Some(mode),
        Err(_) => {
            tracing::warn!(
                "ignoring invalid [permissions].default '{}' (expected none, read, \
                 ask, or write)",
                raw
            );
            None
        }
    });

    let resolved_default = match configured_default {
        Some(mode) if enabled.is_enabled(mode) => mode,
        Some(mode) => {
            tracing::warn!(
                "[permissions].default = '{}' is not in [permissions].enabled; \
                 falling back",
                mode
            );
            if enabled.is_enabled(Permission::Read) {
                Permission::Read
            } else {
                enabled.lowest()
            }
        }
        None => {
            if enabled.is_enabled(Permission::Read) {
                Permission::Read
            } else {
                enabled.lowest()
            }
        }
    };

    let env_override = env_permission.and_then(|raw| match raw.parse::<Permission>() {
        Ok(mode) => Some(mode),
        Err(_) => {
            tracing::warn!(
                "ignoring invalid AGSH_PERMISSION='{}' (expected none, read, ask, or \
                 write)",
                raw
            );
            None
        }
    });

    let raw_choice = cli_permission.or(env_override);
    let permission = match raw_choice {
        Some(mode) if enabled.is_enabled(mode) => mode,
        Some(mode) => {
            tracing::warn!(
                "requested start mode '{}' is not in [permissions].enabled; using '{}'",
                mode,
                resolved_default
            );
            resolved_default
        }
        None => resolved_default,
    };

    (permission, enabled)
}

/// Resolve the active Linux sandbox backend.
///
/// When the user pinned `[shell].sandbox_backend = "..."` in `config.toml`, that choice is binding
/// — no silent fallback at runtime; an unavailable explicit backend surfaces at use time via the
/// `BackendProbe::Missing` / `UserNamespaceDenied` variants.
///
/// When the value is unset (`None`), agsh probes bubblewrap and picks it if available, falling back
/// to landlock otherwise. The `auto_resolved` flag is propagated so the startup warn helper can
/// nudge the user once toward installing bwrap (without nagging users who explicitly pinned
/// landlock).
#[cfg(target_os = "linux")]
fn resolve_sandbox_backend(
    configured: Option<SandboxBackend>,
) -> (SandboxBackend, bool, crate::sandbox::BackendProbe) {
    use crate::sandbox::{BackendProbe, probe_backend};

    // Probe Bubblewrap only when its result is load-bearing for the resolution: either the user
    // pinned it explicitly, or no value was configured (so we need the probe to decide whether to
    // auto-pick it). When the user pinned Landlock, the Bubblewrap smoke test would be pure waste
    // (~500 ms on every agsh start).
    let (backend, auto_resolved, cached_bubblewrap_probe) = match configured {
        Some(explicit) => (explicit, false, None),
        None => {
            let probe = probe_backend(SandboxBackend::Bubblewrap);
            let picked = match &probe {
                BackendProbe::Ok(_) => SandboxBackend::Bubblewrap,
                _ => SandboxBackend::Landlock,
            };
            (picked, true, Some(probe))
        }
    };
    // The Landlock arm discards `cached_bubblewrap_probe` because the auto-resolve path that
    // populated it landed on Bubblewrap (it only falls through to Landlock when Bubblewrap probes
    // unavailable, and that probe isn't useful for the chosen backend's status).
    let backend_probe = match (backend, cached_bubblewrap_probe) {
        (SandboxBackend::Bubblewrap, Some(probe)) => probe,
        (SandboxBackend::Bubblewrap, None) => probe_backend(SandboxBackend::Bubblewrap),
        (SandboxBackend::Landlock, _) => probe_backend(SandboxBackend::Landlock),
    };
    (backend, auto_resolved, backend_probe)
}

/// Non-Linux platforms have a single platform-native sandbox (`sandbox-exec` on macOS,
/// Low-integrity on Windows, nothing elsewhere). `[shell].sandbox_backend` is documented as
/// Linux-only and is ignored here: the resolved capability comes from [`crate::sandbox::detect`]
/// and is surfaced through the same `BackendProbe::Ok` envelope so the downstream wiring in
/// `src/main.rs` doesn't need a platform branch.
#[cfg(not(target_os = "linux"))]
fn resolve_sandbox_backend(
    _configured: Option<SandboxBackend>,
) -> (SandboxBackend, bool, crate::sandbox::BackendProbe) {
    use crate::sandbox::{BackendProbe, SandboxCapability};

    let probe = match crate::sandbox::detect() {
        SandboxCapability::Unavailable => BackendProbe::Missing {
            reason: "no platform sandbox backend available".to_string(),
        },
        capability => BackendProbe::Ok(capability),
    };
    // `SandboxBackend::Landlock` is a stand-in here — the field exists for Linux config parity but
    // is never consulted on this platform.
    (SandboxBackend::Landlock, false, probe)
}

/// Merge `--eager-load-tool SERVER:TOOL` CLI values into the matching server's
/// [`McpServerConfig::eager_load_tools`] list. Malformed entries and unknown server names warn and
/// are skipped — same philosophy as `warn_on_stale_tool_config`. Appends to (never replaces) the
/// configured list, and deduplicates so a CLI flag that overlaps with `config.toml` doesn't grow
/// the list.
fn apply_cli_eager_load_overrides(raw_pairs: &[String], servers: &mut [McpServerConfig]) {
    for raw in raw_pairs {
        let (server_name, tool_name) = match raw.split_once(':') {
            Some((server, tool)) => {
                let server = server.trim();
                let tool = tool.trim();
                if server.is_empty() || tool.is_empty() {
                    tracing::warn!(
                        "ignoring --eager-load-tool '{}' (expected SERVER:TOOL with both \
                         parts non-empty)",
                        raw
                    );
                    continue;
                }
                (server, tool)
            }
            None => {
                tracing::warn!(
                    "ignoring --eager-load-tool '{}' (expected SERVER:TOOL format)",
                    raw
                );
                continue;
            }
        };
        match servers.iter_mut().find(|s| s.name == server_name) {
            Some(server) => {
                let list = server.eager_load_tools.get_or_insert_with(Vec::new);
                if !list.iter().any(|existing| existing == tool_name) {
                    list.push(tool_name.to_string());
                }
            }
            None => {
                tracing::warn!(
                    "--eager-load-tool '{}': no MCP server named '{}' is configured",
                    raw,
                    server_name
                );
            }
        }
    }
}

impl ResolvedConfig {
    pub fn from_cli(cli: &Cli) -> Self {
        let config_file = load_config_file();
        let file_provider = config_file.provider.unwrap_or_default();
        let file_display = config_file.display.unwrap_or_default();
        let file_web = config_file.web.unwrap_or_default();
        let file_shell = config_file.shell.unwrap_or_default();
        let file_session = config_file.session.unwrap_or_default();
        let file_thinking = config_file.thinking.unwrap_or_default();
        let file_prompt = config_file.prompt.unwrap_or_default();
        let file_tools = config_file.tools.unwrap_or_default();
        // Destructure the [mcp] table into its two independent fields so we don't have to re-open
        // the config file later for resolution.
        let (
            mcp_default_permission_str,
            mut mcp_servers,
            mcp_strict,
            mcp_grace,
            mcp_connect_timeout,
        ) = match config_file.mcp {
            Some(mcp) => (
                mcp.default_permission,
                mcp.servers.unwrap_or_default(),
                mcp.strict.unwrap_or(true),
                std::time::Duration::from_secs(mcp.grace_seconds.unwrap_or(3)),
                std::time::Duration::from_secs(mcp.connect_timeout_seconds.unwrap_or(30)),
            ),
            None => (
                None,
                Vec::new(),
                true,
                std::time::Duration::from_secs(3),
                std::time::Duration::from_secs(30),
            ),
        };
        apply_cli_eager_load_overrides(&cli.eager_load_tool, &mut mcp_servers);
        let mcp_default_permission = match mcp_default_permission_str.as_deref() {
            Some(raw) => match raw.parse::<Permission>() {
                Ok(permission) => Some(permission),
                Err(_) => {
                    tracing::warn!(
                        "ignoring invalid [mcp].default_permission '{}' (expected \
                         none, read, ask, or write)",
                        raw
                    );
                    None
                }
            },
            None => None,
        };

        let user_instructions = cli
            .instructions
            .clone()
            .or_else(|| std::env::var("AGSH_INSTRUCTIONS").ok())
            .or(file_prompt.instructions)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());

        let builtin_allowed_tools = file_tools
            .allowed_tools
            .filter(|list| !list.is_empty())
            .map(|list| list.into_iter().map(|s| s.trim().to_string()).collect());
        let builtin_disabled_tools = file_tools
            .disabled_tools
            .unwrap_or_default()
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let builtin_tool_permissions = file_tools
            .tool_permissions
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(name, level)| {
                let name = name.trim().to_string();
                if name.is_empty() {
                    return None;
                }
                match level.parse::<Permission>() {
                    Ok(permission) => Some((name, permission)),
                    Err(_) => {
                        tracing::warn!(
                            "ignoring invalid [tools.tool_permissions] entry '{}' = '{}' \
                             (expected none, read, ask, or write)",
                            name,
                            level
                        );
                        None
                    }
                }
            })
            .collect();

        let provider_name = cli
            .provider
            .clone()
            .or_else(|| std::env::var("AGSH_PROVIDER").ok())
            .or_else(|| file_provider.name.clone());

        let model = cli
            .model
            .clone()
            .or_else(|| std::env::var("AGSH_MODEL").ok())
            .or_else(|| file_provider.model.clone());

        let auth_credential = credential::resolve(provider_name.as_deref(), &file_provider);

        let base_url = cli
            .base_url
            .clone()
            .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
            .or_else(|| file_provider.base_url.clone());

        let file_permissions = config_file.permissions.unwrap_or_default();
        let (permission, enabled_permissions) = resolve_permission(
            cli.permission,
            std::env::var("AGSH_PERMISSION").ok().as_deref(),
            file_permissions.default.as_deref(),
            file_permissions.enabled.as_deref(),
        );

        // Compute device_id before the struct literal so we can borrow `provider_name` here without
        // conflicting with the `provider_name` field move below.
        let device_id =
            device_id::resolve(provider_name.as_deref(), file_provider.device_id.as_deref());
        let effort = effort::resolve(file_provider.effort.as_deref());
        let redact_thinking = file_provider.redact_thinking.unwrap_or(false);

        // Only probe the sandbox backend when sandboxing is actually enabled. Skipping the probe
        // for `sandbox = false` saves the smoke-test cost on every invocation of subcommands that
        // don't touch the shell (`agsh list`, `agsh export`, `agsh mcp list`, etc.) when the user
        // has disabled sandboxing globally. The placeholder probe is never consulted in that state
        // — the shell tool short-circuits on `sandbox_enabled = false`, and the warn helper early-
        // returns on `!state.enabled`.
        let (sandbox_backend, sandbox_auto_resolved, backend_probe) =
            if file_shell.sandbox.unwrap_or(true) {
                resolve_sandbox_backend(file_shell.sandbox_backend)
            } else {
                (
                    file_shell
                        .sandbox_backend
                        .unwrap_or(SandboxBackend::Landlock),
                    false,
                    crate::sandbox::BackendProbe::Missing {
                        reason: "sandbox disabled in config".to_string(),
                    },
                )
            };

        Self {
            provider_name,
            model,
            auth_credential,
            base_url,
            client_id: std::env::var("CLAUDE_CLIENT_ID").ok(),
            oauth_token_url: file_provider.oauth_token_url.clone(),
            permission,
            enabled_permissions,
            streaming: !cli.no_stream,
            max_turn_requests: config_file
                .agent
                .as_ref()
                .and_then(|a| a.max_turn_requests)
                .unwrap_or(100),
            continue_session: cli.continue_session.clone(),
            prompt: cli.prompt.clone(),
            oneshot: cli.oneshot,
            newline_before_prompt: file_display.newline_before_prompt.unwrap_or(true),
            newline_after_prompt: file_display.newline_after_prompt.unwrap_or(true),
            show_session_id_on_create: file_display.show_session_id_on_create.unwrap_or(false),
            show_session_id_on_exit: file_display.show_session_id_on_exit.unwrap_or(true),
            show_token_usage: file_display.show_token_usage.unwrap_or(false),
            resume_show_recent: file_display.resume_show_recent,
            show_path_in_prompt: file_display.show_path_in_prompt.unwrap_or(true),
            web_client: WebClientConfig::from_file(&file_web),
            sandbox: file_shell.sandbox.unwrap_or(true),
            sandbox_backend,
            sandbox_auto_resolved,
            backend_probe,
            render_mode: cli
                .render_mode
                .or(file_display.render_mode)
                .unwrap_or_default(),
            context_messages: file_session
                .context_messages
                .or(Some(DEFAULT_CONTEXT_MESSAGES)),
            retention_days: file_session.retention_days.or(Some(DEFAULT_RETENTION_DAYS)),
            max_storage_bytes: file_session
                .max_storage_bytes
                .or(Some(DEFAULT_MAX_STORAGE_BYTES)),
            thinking_enabled: cli
                .thinking
                .unwrap_or_else(|| file_thinking.enabled.unwrap_or(true)),
            thinking_budget_tokens: cli.thinking_budget.unwrap_or_else(|| {
                file_thinking
                    .budget_tokens
                    .unwrap_or(DEFAULT_THINKING_BUDGET_TOKENS)
            }),
            thinking_show_content: file_thinking.show_content.unwrap_or(false),
            reasoning_effort: file_provider.reasoning_effort.clone(),
            device_id,
            effort,
            redact_thinking,
            auto_compact: file_session.auto_compact.unwrap_or(true),
            context_window: file_session.context_window,
            mcp_servers,
            mcp_default_permission,
            user_instructions,
            builtin_allowed_tools,
            builtin_disabled_tools,
            builtin_tool_permissions,
            input_style: file_display
                .input_style
                .as_deref()
                .map(parse_input_style)
                .unwrap_or_else(default_input_style),
            mcp_strict,
            mcp_grace,
            mcp_connect_timeout,
        }
    }

    pub fn validate(&self) -> crate::error::Result<()> {
        match self.provider_name.as_deref() {
            None => {
                return Err(crate::error::AgshError::Config(format!(
                    "no provider configured. Set --provider, AGSH_PROVIDER env var, or \
                     provider.name in config file (~/.config/agsh/config.toml). Supported \
                     providers: {}",
                    crate::provider::SUPPORTED_PROVIDERS.join(", "),
                )));
            }
            Some(name) if !crate::provider::SUPPORTED_PROVIDERS.contains(&name) => {
                return Err(crate::error::AgshError::Config(format!(
                    "'{}' is not a valid provider. Supported providers: {}",
                    name,
                    crate::provider::SUPPORTED_PROVIDERS.join(", "),
                )));
            }
            Some(_) => {}
        }
        if self.model.is_none() {
            return Err(crate::error::AgshError::Config(
                "no model configured. Set --model, AGSH_MODEL env var, \
                 or provider.model in config file (~/.config/agsh/config.toml)"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

pub fn context_window_for_model(model: &str) -> u64 {
    if model.contains("claude") {
        200_000
    } else if model.contains("gpt-4.1") {
        1_047_576
    } else if model.contains("gpt-4o") {
        128_000
    } else if model.contains("o3") || model.contains("o4-mini") || model.contains("o1") {
        200_000
    } else {
        128_000
    }
}

/// Stable per-device identity for `claude-oauth` (embedded in `metadata.user_id`). Other providers
/// get an empty string — we don't write a stub config file just to hold an unused value.
mod device_id {
    use std::path::Path;

    use super::{config_file_path, write_config_atomic};

    /// Lookup order: configured → persisted → Claude Code's `~/.claude.json` userID → freshly
    /// generated. The claude.json fallback lets agsh and Claude Code on the same machine share a
    /// device identity.
    pub(super) fn resolve(provider_name: Option<&str>, configured: Option<&str>) -> String {
        if provider_name != Some("claude-oauth") {
            return String::new();
        }

        if let Some(id) = configured
            && !id.is_empty()
        {
            return id.to_string();
        }

        let path = match config_file_path() {
            Some(path) => path,
            None => return read_claude_code_user_id().unwrap_or_else(generate),
        };

        if let Ok(contents) = std::fs::read_to_string(&path)
            && let Ok(doc) = contents.parse::<toml_edit::DocumentMut>()
            && let Some(id) = doc
                .get("provider")
                .and_then(|t| t.get("device_id"))
                .and_then(|v| v.as_str())
            && !id.is_empty()
        {
            return id.to_string();
        }

        let (id, source) = match read_claude_code_user_id() {
            Some(id) => (id, "~/.claude.json"),
            None => (generate(), "random"),
        };
        tracing::info!("seeded claude-oauth device_id from {}", source);
        if let Err(error) = persist(&path, &id) {
            tracing::warn!("failed to persist device_id: {}", error);
        }
        id
    }

    fn generate() -> String {
        use rand::RngExt;
        let mut bytes = [0u8; 32];
        rand::rng().fill(&mut bytes);
        bytes.iter().map(|byte| format!("{:02x}", byte)).collect()
    }

    fn read_claude_code_user_id() -> Option<String> {
        read_user_id_from(&dirs::home_dir()?.join(".claude.json"))
    }

    pub(super) fn read_user_id_from(path: &Path) -> Option<String> {
        let contents = match std::fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(error) => {
                tracing::debug!("could not read user-id file {}: {}", path.display(), error);
                return None;
            }
        };
        let document: serde_json::Value = match serde_json::from_str(&contents) {
            Ok(document) => document,
            Err(error) => {
                tracing::debug!("could not parse user-id file {}: {}", path.display(), error);
                return None;
            }
        };
        let id = document.get("userID")?.as_str()?.trim();
        if id.is_empty() {
            return None;
        }
        Some(id.to_string())
    }

    fn persist(path: &Path, id: &str) -> std::io::Result<()> {
        let contents = std::fs::read_to_string(path).unwrap_or_default();
        let mut doc: toml_edit::DocumentMut = contents
            .parse()
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;

        if !doc.contains_key("provider") {
            doc["provider"] = toml_edit::Item::Table(toml_edit::Table::new());
        }
        doc["provider"]["device_id"] = toml_edit::value(id);

        write_config_atomic(path, &doc.to_string())
    }
}

/// `[provider].effort` normalisation for Claude Code's `output_config.effort`.
mod effort {
    /// Resolves to one of `"low" | "medium" | "high"`, falling back to `"high"` for missing or
    /// unrecognised values (with a warn log for the latter so a typo isn't silently lost).
    pub(super) fn resolve(configured: Option<&str>) -> String {
        const DEFAULT: &str = "high";
        let Some(value) = configured else {
            return DEFAULT.to_string();
        };
        let trimmed = value.trim();
        match trimmed.to_ascii_lowercase().as_str() {
            "low" | "medium" | "high" => trimmed.to_ascii_lowercase(),
            other => {
                tracing::warn!(
                    "ignoring [provider].effort = {:?}: expected one of \"low\", \"medium\", \"high\"; \
                     falling back to \"{}\"",
                    other,
                    DEFAULT,
                );
                DEFAULT.to_string()
            }
        }
    }
}

/// Provider credential lookup: env var → config file, with provider-specific env-var precedence.
/// OAuth database fallback happens in `main.rs`.
mod credential {
    use super::{AuthCredential, ProviderConfig};

    pub(super) fn resolve(
        provider_name: Option<&str>,
        file_provider: &ProviderConfig,
    ) -> Option<AuthCredential> {
        match provider_name {
            Some("claude-api") => {
                if let Ok(key) = std::env::var("CLAUDE_API_KEY") {
                    return Some(AuthCredential::ApiKey(key));
                }
                if let Some(key) = &file_provider.api_key {
                    return Some(AuthCredential::ApiKey(key.clone()));
                }
                None
            }
            Some("claude-oauth") => {
                if let Ok(token) = std::env::var("CLAUDE_OAUTH_TOKEN") {
                    return Some(AuthCredential::OAuthToken {
                        access_token: token,
                        refresh_token: None,
                        expires_at: None,
                        account_id: None,
                    });
                }
                if let Some(token) = &file_provider.oauth_token {
                    return Some(AuthCredential::OAuthToken {
                        access_token: token.clone(),
                        refresh_token: None,
                        expires_at: None,
                        account_id: None,
                    });
                }
                // Database fallback happens in main.rs
                None
            }
            Some("openai-codex") => {
                if let Ok(token) = std::env::var("OPENAI_CODEX_TOKEN") {
                    return Some(AuthCredential::OAuthToken {
                        access_token: token,
                        refresh_token: None,
                        expires_at: None,
                        account_id: None,
                    });
                }
                if let Some(token) = &file_provider.oauth_token {
                    return Some(AuthCredential::OAuthToken {
                        access_token: token.clone(),
                        refresh_token: None,
                        expires_at: None,
                        account_id: None,
                    });
                }
                // Database fallback happens in main.rs
                None
            }
            _ => {
                let key = std::env::var("OPENAI_API_KEY")
                    .ok()
                    .or_else(|| file_provider.api_key.clone())?;
                Some(AuthCredential::ApiKey(key))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_server(name: &str) -> McpServerConfig {
        McpServerConfig {
            name: name.to_string(),
            transport: McpTransport::Http,
            command: None,
            args: None,
            env: None,
            url: Some("https://example".to_string()),
            auth_token: None,
            headers: None,
            headers_helper: None,
            auth: None,
            permission: None,
            allowed_tools: None,
            disabled_tools: None,
            eager_load_tools: None,
            tool_permissions: None,
            sampling: false,
            sampling_limit: None,
            disabled: false,
        }
    }

    #[test]
    fn test_eager_load_override_appends_to_matching_server() {
        let mut servers = vec![fixture_server("notion"), fixture_server("github")];
        let raw = vec![
            "notion:search".to_string(),
            "github:create_issue".to_string(),
        ];
        apply_cli_eager_load_overrides(&raw, &mut servers);

        assert_eq!(
            servers[0].eager_load_tools.as_deref(),
            Some(&["search".to_string()][..])
        );
        assert_eq!(
            servers[1].eager_load_tools.as_deref(),
            Some(&["create_issue".to_string()][..])
        );
    }

    #[test]
    fn test_eager_load_override_appends_to_existing_list() {
        let mut servers = vec![fixture_server("notion")];
        servers[0].eager_load_tools = Some(vec!["search".to_string()]);
        apply_cli_eager_load_overrides(&["notion:fetch".to_string()], &mut servers);
        assert_eq!(
            servers[0].eager_load_tools.as_deref(),
            Some(&["search".to_string(), "fetch".to_string()][..])
        );
    }

    #[test]
    fn test_eager_load_override_dedupes_existing_entry() {
        let mut servers = vec![fixture_server("notion")];
        servers[0].eager_load_tools = Some(vec!["search".to_string()]);
        apply_cli_eager_load_overrides(&["notion:search".to_string()], &mut servers);
        assert_eq!(
            servers[0].eager_load_tools.as_deref(),
            Some(&["search".to_string()][..]),
            "duplicate tool name must not double the list"
        );
    }

    #[test]
    fn test_eager_load_override_skips_unknown_server() {
        let mut servers = vec![fixture_server("notion")];
        apply_cli_eager_load_overrides(&["nope:search".to_string()], &mut servers);
        // The matching `notion` entry must remain untouched; the unknown `nope` entry simply
        // produces a warn log (not captured here).
        assert!(servers[0].eager_load_tools.is_none());
    }

    #[test]
    fn test_eager_load_override_skips_malformed_values() {
        let mut servers = vec![fixture_server("notion")];
        let raw = vec![
            "no-colon".to_string(),
            ":missing-server".to_string(),
            "missing-tool:".to_string(),
            "".to_string(),
        ];
        apply_cli_eager_load_overrides(&raw, &mut servers);
        assert!(servers[0].eager_load_tools.is_none());
    }

    #[test]
    fn test_eager_load_override_trims_whitespace() {
        let mut servers = vec![fixture_server("notion")];
        apply_cli_eager_load_overrides(&["  notion : search  ".to_string()], &mut servers);
        assert_eq!(
            servers[0].eager_load_tools.as_deref(),
            Some(&["search".to_string()][..])
        );
    }

    #[test]
    fn test_web_config_all_fields_parse() {
        let toml_str = r#"
[web]
user_agent = "agsh-test"
request_timeout_seconds = 60
connect_timeout_seconds = 5
read_timeout_seconds = 10
max_redirects = 3
proxy = "socks5h://127.0.0.1:1080"
ca_cert_file = "/etc/ssl/corp.pem"
https_only = true
min_tls_version = "1.3"
danger_accept_invalid_certs = true
danger_accept_invalid_hostnames = true
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("parse toml");
        let web = config.web.expect("web present");
        assert_eq!(web.user_agent.as_deref(), Some("agsh-test"));
        assert_eq!(web.request_timeout_seconds, Some(60));
        assert_eq!(web.connect_timeout_seconds, Some(5));
        assert_eq!(web.read_timeout_seconds, Some(10));
        assert_eq!(web.max_redirects, Some(3));
        assert_eq!(web.proxy.as_deref(), Some("socks5h://127.0.0.1:1080"));
        assert_eq!(web.ca_cert_file.as_deref(), Some("/etc/ssl/corp.pem"));
        assert_eq!(web.https_only, Some(true));
        assert_eq!(web.min_tls_version.as_deref(), Some("1.3"));
        assert_eq!(web.danger_accept_invalid_certs, Some(true));
        assert_eq!(web.danger_accept_invalid_hostnames, Some(true));
    }

    #[test]
    fn test_web_client_config_defaults_from_empty_file() {
        // Empty [web] → sensible defaults; no user-surprising failures.
        let file = WebConfig::default();
        let cfg = WebClientConfig::from_file(&file);
        assert_eq!(cfg.user_agent, DEFAULT_WEB_USER_AGENT);
        assert_eq!(cfg.request_timeout, std::time::Duration::from_secs(30));
        assert!(cfg.connect_timeout.is_none());
        assert!(cfg.read_timeout.is_none());
        assert_eq!(cfg.max_redirects, 10);
        assert!(cfg.proxy.is_none());
        assert!(cfg.ca_cert_file.is_none());
        assert!(!cfg.https_only);
        assert!(cfg.min_tls_version.is_none());
        assert!(!cfg.danger_accept_invalid_certs);
        assert!(!cfg.danger_accept_invalid_hostnames);
    }

    #[test]
    fn test_web_client_config_resolves_full_file() {
        let file = WebConfig {
            user_agent: Some("ua".to_string()),
            request_timeout_seconds: Some(60),
            connect_timeout_seconds: Some(5),
            read_timeout_seconds: Some(10),
            max_redirects: Some(0),
            proxy: Some("http://proxy.local:8080".to_string()),
            ca_cert_file: Some("/tmp/ca.pem".to_string()),
            https_only: Some(true),
            min_tls_version: Some("1.3".to_string()),
            danger_accept_invalid_certs: Some(true),
            danger_accept_invalid_hostnames: Some(true),
        };
        let cfg = WebClientConfig::from_file(&file);
        assert_eq!(cfg.user_agent, "ua");
        assert_eq!(cfg.request_timeout, std::time::Duration::from_secs(60));
        assert_eq!(cfg.connect_timeout, Some(std::time::Duration::from_secs(5)));
        assert_eq!(cfg.read_timeout, Some(std::time::Duration::from_secs(10)));
        assert_eq!(cfg.max_redirects, 0);
        assert_eq!(cfg.proxy.as_deref(), Some("http://proxy.local:8080"));
        assert_eq!(
            cfg.ca_cert_file.as_deref(),
            Some(std::path::Path::new("/tmp/ca.pem"))
        );
        assert!(cfg.https_only);
        assert_eq!(cfg.min_tls_version, Some(MinTlsVersion::V1_3));
        assert!(cfg.danger_accept_invalid_certs);
        assert!(cfg.danger_accept_invalid_hostnames);
    }

    #[test]
    fn test_min_tls_version_parse_accepts_all_valid() {
        assert_eq!(MinTlsVersion::parse("1.0"), Some(MinTlsVersion::V1_0));
        assert_eq!(MinTlsVersion::parse("1.1"), Some(MinTlsVersion::V1_1));
        assert_eq!(MinTlsVersion::parse("1.2"), Some(MinTlsVersion::V1_2));
        assert_eq!(MinTlsVersion::parse("1.3"), Some(MinTlsVersion::V1_3));
        // Whitespace trimming.
        assert_eq!(MinTlsVersion::parse("  1.2  "), Some(MinTlsVersion::V1_2));
    }

    #[test]
    fn test_min_tls_version_parse_rejects_invalid() {
        assert!(MinTlsVersion::parse("1.5").is_none());
        assert!(MinTlsVersion::parse("tls1.3").is_none());
        assert!(MinTlsVersion::parse("").is_none());
    }

    #[test]
    fn test_web_client_config_rejects_bad_min_tls_falls_back() {
        // Invalid min_tls_version string logs a warn but doesn't abort — we fall through to
        // reqwest's default rather than failing startup on a typo.
        let file = WebConfig {
            min_tls_version: Some("1.5".to_string()),
            ..WebConfig::default()
        };
        let cfg = WebClientConfig::from_file(&file);
        assert!(cfg.min_tls_version.is_none());
    }

    #[test]
    fn test_web_client_config_zero_timeout_uses_default() {
        // `0` in the config is treated as "fall through to default" so users can't accidentally set
        // request_timeout = 0 and disable timeouts entirely.
        let file = WebConfig {
            request_timeout_seconds: Some(0),
            connect_timeout_seconds: Some(0),
            read_timeout_seconds: Some(0),
            ..WebConfig::default()
        };
        let cfg = WebClientConfig::from_file(&file);
        assert_eq!(cfg.request_timeout, std::time::Duration::from_secs(30));
        assert!(cfg.connect_timeout.is_none());
        assert!(cfg.read_timeout.is_none());
    }

    #[test]
    fn test_mcp_runtime_fields_parse() {
        let toml_str = r#"
[mcp]
default_permission = "read"
strict = false
grace_seconds = 5
connect_timeout_seconds = 60
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("parse toml");
        let mcp = config.mcp.expect("mcp present");
        assert_eq!(mcp.strict, Some(false));
        assert_eq!(mcp.grace_seconds, Some(5));
        assert_eq!(mcp.connect_timeout_seconds, Some(60));
    }

    #[test]
    fn test_mcp_server_disabled_parses() {
        let toml_str = r#"
[[mcp.servers]]
name = "flaky"
transport = "stdio"
command = "npx"
disabled = true
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("parse toml");
        let servers = config.mcp.unwrap().servers.unwrap();
        assert_eq!(servers.len(), 1);
        assert!(servers[0].disabled);
    }

    #[test]
    fn test_mcp_server_disabled_defaults_false() {
        let toml_str = r#"
[[mcp.servers]]
name = "normal"
transport = "stdio"
command = "npx"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("parse toml");
        let servers = config.mcp.unwrap().servers.unwrap();
        assert!(!servers[0].disabled);
    }

    #[test]
    fn test_config_file_deserialization() {
        let toml_str = r#"
[provider]
name = "openai-api"
model = "gpt-4o"
api_key = "sk-test"
base_url = "https://api.openai.com/v1"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let provider = config.provider.expect("provider should be present");
        assert_eq!(provider.name.as_deref(), Some("openai-api"));
        assert_eq!(provider.model.as_deref(), Some("gpt-4o"));
        assert_eq!(provider.api_key.as_deref(), Some("sk-test"));
        assert_eq!(
            provider.base_url.as_deref(),
            Some("https://api.openai.com/v1")
        );
    }

    #[test]
    fn test_empty_config_file() {
        let config: ConfigFile = toml::from_str("").expect("failed to parse empty toml");
        assert!(config.provider.is_none());
    }

    #[test]
    fn test_partial_config_file() {
        let toml_str = r#"
[provider]
name = "claude-oauth"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let provider = config.provider.expect("provider should be present");
        assert_eq!(provider.name.as_deref(), Some("claude-oauth"));
        assert!(provider.model.is_none());
        assert!(provider.api_key.is_none());
        assert!(provider.base_url.is_none());
    }

    #[test]
    fn test_resolve_effort_default_high() {
        assert_eq!(effort::resolve(None), "high");
    }

    #[test]
    fn test_resolve_effort_recognized_values() {
        assert_eq!(effort::resolve(Some("low")), "low");
        assert_eq!(effort::resolve(Some("medium")), "medium");
        assert_eq!(effort::resolve(Some("high")), "high");
        // Case-insensitive + trims surrounding whitespace.
        assert_eq!(effort::resolve(Some("  Medium ")), "medium");
        assert_eq!(effort::resolve(Some("HIGH")), "high");
    }

    #[test]
    fn test_resolve_effort_unknown_falls_back_to_high() {
        assert_eq!(effort::resolve(Some("max")), "high");
        assert_eq!(effort::resolve(Some("")), "high");
        assert_eq!(effort::resolve(Some("ultra")), "high");
    }

    #[test]
    fn test_resolve_device_id_returns_empty_for_non_claude_oauth() {
        // Should not generate / persist anything when the provider doesn't need a device_id. Empty
        // string flows through but is ignored by non-claude-oauth providers.
        assert_eq!(device_id::resolve(Some("openai-api"), None), "");
        assert_eq!(device_id::resolve(Some("claude-api"), None), "");
        assert_eq!(device_id::resolve(None, None), "");
        // Even an explicit configured value is suppressed when the provider isn't claude-oauth —
        // the field is provider-scoped.
        assert_eq!(device_id::resolve(Some("openai-api"), Some("explicit")), "");
    }

    #[test]
    fn test_resolve_device_id_uses_configured_value_for_claude_oauth() {
        let id = "deadbeef".repeat(8);
        assert_eq!(
            device_id::resolve(Some("claude-oauth"), Some(&id)),
            id,
            "configured value must be used verbatim for claude-oauth"
        );
    }

    #[test]
    fn test_read_user_id_from_valid_claude_json() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(
            &path,
            r#"{"userID": "af5986c7cb3b5e8d00eaf3da3b81730c6f523b1e68e1720c7128a96167534be3", "other": "stuff"}"#,
        )
        .expect("write");
        assert_eq!(
            device_id::read_user_id_from(&path).as_deref(),
            Some("af5986c7cb3b5e8d00eaf3da3b81730c6f523b1e68e1720c7128a96167534be3")
        );
    }

    #[test]
    fn test_read_user_id_from_missing_file_returns_none() {
        let path = std::path::Path::new("/nonexistent/path/claude.json");
        assert!(device_id::read_user_id_from(path).is_none());
    }

    #[test]
    fn test_read_user_id_from_malformed_json_returns_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(&path, "{not valid json").expect("write");
        assert!(device_id::read_user_id_from(&path).is_none());
    }

    #[test]
    fn test_read_user_id_from_missing_field_returns_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(&path, r#"{"foo": "bar"}"#).expect("write");
        assert!(device_id::read_user_id_from(&path).is_none());
    }

    #[test]
    fn test_read_user_id_from_empty_string_returns_none() {
        // An empty `userID` in claude.json shouldn't override agsh's own random-generation fallback
        // — treat it as "not configured".
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(&path, r#"{"userID": ""}"#).expect("write");
        assert!(device_id::read_user_id_from(&path).is_none());
    }

    #[test]
    fn test_read_user_id_from_whitespace_only_returns_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(&path, r#"{"userID": "   "}"#).expect("write");
        assert!(device_id::read_user_id_from(&path).is_none());
    }

    #[test]
    fn test_read_user_id_from_non_string_returns_none() {
        // A non-string `userID` (number, object, …) shouldn't crash — just decline to use the
        // value.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(&path, r#"{"userID": 12345}"#).expect("write");
        assert!(device_id::read_user_id_from(&path).is_none());
    }

    #[test]
    fn test_read_user_id_from_trims_surrounding_whitespace() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("claude.json");
        std::fs::write(&path, r#"{"userID": "  abcdef123  "}"#).expect("write");
        assert_eq!(
            device_id::read_user_id_from(&path).as_deref(),
            Some("abcdef123")
        );
    }

    #[test]
    fn test_provider_config_deserializes_effort_and_redact_thinking() {
        let toml_str = r#"
[provider]
name = "claude-oauth"
model = "claude-opus-4-6-20250514"
effort = "medium"
redact_thinking = true
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let provider = config.provider.expect("provider should be present");
        assert_eq!(provider.effort.as_deref(), Some("medium"));
        assert_eq!(provider.redact_thinking, Some(true));
    }

    #[test]
    fn test_session_config_deserialization() {
        let toml_str = r#"
[session]
context_messages = 100
retention_days = 90
max_storage_bytes = 52428800
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let session = config.session.expect("session should be present");
        assert_eq!(session.context_messages, Some(100));
        assert_eq!(session.retention_days, Some(90));
        assert_eq!(session.max_storage_bytes, Some(52428800));
    }

    #[test]
    fn test_session_config_partial() {
        let toml_str = r#"
[session]
context_messages = 50
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let session = config.session.expect("session should be present");
        assert_eq!(session.context_messages, Some(50));
        assert!(session.retention_days.is_none());
        assert!(session.max_storage_bytes.is_none());
    }

    #[test]
    fn test_session_defaults_applied() {
        let file_session = SessionConfig::default();
        let context_messages = file_session.context_messages.or(Some(200));
        let retention_days = file_session.retention_days.or(Some(90));
        let max_storage_bytes = file_session.max_storage_bytes.or(Some(52_428_800));

        assert_eq!(context_messages, Some(200));
        assert_eq!(retention_days, Some(90));
        assert_eq!(max_storage_bytes, Some(52_428_800));
    }

    #[test]
    fn test_mcp_config_deserialization() {
        let toml_str = r#"
[[mcp.servers]]
name = "postgres"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-postgres"]
permission = "read"

[[mcp.servers]]
name = "web-api"
transport = "http"
url = "http://localhost:8080/mcp"
permission = "write"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let mcp = config.mcp.expect("mcp should be present");
        let servers = mcp.servers.expect("servers should be present");
        assert_eq!(servers.len(), 2);
        assert_eq!(servers[0].name, "postgres");
        assert_eq!(servers[0].transport, McpTransport::Stdio);
        assert_eq!(servers[0].command.as_deref(), Some("npx"));
        assert_eq!(
            servers[0].args.as_deref(),
            Some(
                ["-y", "@modelcontextprotocol/server-postgres"]
                    .map(String::from)
                    .as_slice()
            )
        );
        assert_eq!(servers[0].permission.as_deref(), Some("read"));
        assert_eq!(servers[1].name, "web-api");
        assert_eq!(servers[1].transport, McpTransport::Http);
        assert_eq!(servers[1].url.as_deref(), Some("http://localhost:8080/mcp"));
        assert_eq!(servers[1].permission.as_deref(), Some("write"));
    }

    #[test]
    fn test_mcp_config_empty() {
        let config: ConfigFile = toml::from_str("").expect("failed to parse empty toml");
        assert!(config.mcp.is_none());
    }

    #[test]
    fn test_session_config_overrides_defaults() {
        let toml_str = r#"
[session]
context_messages = 50
retention_days = 30
max_storage_bytes = 10485760
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let file_session = config.session.unwrap_or_default();
        let context_messages = file_session.context_messages.or(Some(200));
        let retention_days = file_session.retention_days.or(Some(90));
        let max_storage_bytes = file_session.max_storage_bytes.or(Some(52_428_800));

        assert_eq!(context_messages, Some(50));
        assert_eq!(retention_days, Some(30));
        assert_eq!(max_storage_bytes, Some(10_485_760));
    }

    #[test]
    fn test_mcp_auth_client_credentials() {
        let toml_str = r#"
[[mcp.servers]]
name = "api"
transport = "http"
url = "https://api.example.com/mcp"

[mcp.servers.auth]
type = "client_credentials"
client_id = "my-client"
client_secret = "my-secret"
scopes = ["read", "write"]
resource = "https://api.example.com"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let servers = config.mcp.unwrap().servers.unwrap();
        assert_eq!(servers.len(), 1);
        let auth = servers[0].auth.as_ref().expect("auth should be present");
        match auth {
            McpAuthConfig::ClientCredentials {
                client_id,
                client_secret,
                scopes,
                resource,
            } => {
                assert_eq!(client_id, "my-client");
                assert_eq!(client_secret, "my-secret");
                assert_eq!(
                    scopes.as_deref(),
                    Some(["read".to_string(), "write".to_string()].as_slice())
                );
                assert_eq!(resource.as_deref(), Some("https://api.example.com"));
            }
            other => panic!("expected ClientCredentials, got {:?}", other),
        }
    }

    #[test]
    fn test_mcp_auth_client_credentials_jwt() {
        let toml_str = r#"
[[mcp.servers]]
name = "api"
transport = "http"
url = "https://api.example.com/mcp"

[mcp.servers.auth]
type = "client_credentials_jwt"
client_id = "my-client"
signing_key_path = "/path/to/key.pem"
signing_algorithm = "ES256"
scopes = ["admin"]
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let servers = config.mcp.unwrap().servers.unwrap();
        let auth = servers[0].auth.as_ref().expect("auth should be present");
        match auth {
            McpAuthConfig::ClientCredentialsJwt {
                client_id,
                signing_key_path,
                signing_algorithm,
                scopes,
                resource,
            } => {
                assert_eq!(client_id, "my-client");
                assert_eq!(signing_key_path, "/path/to/key.pem");
                assert_eq!(signing_algorithm.as_deref(), Some("ES256"));
                assert_eq!(scopes.as_deref(), Some(["admin".to_string()].as_slice()));
                assert!(resource.is_none());
            }
            other => panic!("expected ClientCredentialsJwt, got {:?}", other),
        }
    }

    #[test]
    fn test_mcp_auth_oauth() {
        let toml_str = r#"
[[mcp.servers]]
name = "github"
transport = "http"
url = "https://mcp.example.com"

[mcp.servers.auth]
type = "oauth"
client_id = "my-app"
scopes = ["repo", "user"]
redirect_port = 9000
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let servers = config.mcp.unwrap().servers.unwrap();
        let auth = servers[0].auth.as_ref().expect("auth should be present");
        match auth {
            McpAuthConfig::OAuth {
                client_id,
                client_secret,
                scopes,
                redirect_port,
            } => {
                assert_eq!(client_id.as_deref(), Some("my-app"));
                assert!(client_secret.is_none());
                assert_eq!(
                    scopes.as_deref(),
                    Some(["repo".to_string(), "user".to_string()].as_slice())
                );
                assert_eq!(*redirect_port, Some(9000));
            }
            other => panic!("expected OAuth, got {:?}", other),
        }
    }

    #[test]
    fn test_mcp_auth_oauth_minimal() {
        let toml_str = r#"
[[mcp.servers]]
name = "api"
transport = "http"
url = "https://api.example.com/mcp"

[mcp.servers.auth]
type = "oauth"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let servers = config.mcp.unwrap().servers.unwrap();
        let auth = servers[0].auth.as_ref().expect("auth should be present");
        match auth {
            McpAuthConfig::OAuth {
                client_id,
                client_secret,
                scopes,
                redirect_port,
            } => {
                assert!(client_id.is_none());
                assert!(client_secret.is_none());
                assert!(scopes.is_none());
                assert!(redirect_port.is_none());
            }
            other => panic!("expected OAuth, got {:?}", other),
        }
    }

    #[test]
    fn test_mcp_no_auth() {
        let toml_str = r#"
[[mcp.servers]]
name = "simple"
transport = "http"
url = "https://api.example.com/mcp"
auth_token = "bearer-token"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let servers = config.mcp.unwrap().servers.unwrap();
        assert!(servers[0].auth.is_none());
        assert_eq!(servers[0].auth_token.as_deref(), Some("bearer-token"));
    }

    #[test]
    fn test_parse_input_style_known_values() {
        use nu_ansi_term::{Color, Style};
        assert_eq!(parse_input_style("bold"), Style::new().bold());
        assert_eq!(parse_input_style("BOLD"), Style::new().bold());
        assert_eq!(parse_input_style("dim"), Style::new().dimmed());
        assert_eq!(parse_input_style("cyan"), Style::new().fg(Color::Cyan));
        assert_eq!(parse_input_style("purple"), Style::new().fg(Color::Magenta));
    }

    #[test]
    fn test_parse_input_style_none_is_plain() {
        use nu_ansi_term::Style;
        assert_eq!(parse_input_style("none"), Style::default());
    }

    #[test]
    fn test_parse_input_style_default_and_empty_yield_preset() {
        let preset = default_input_style();
        assert_eq!(parse_input_style(""), preset);
        assert_eq!(parse_input_style("default"), preset);
        assert!(preset.is_bold);
        assert!(preset.foreground.is_some(), "default must set foreground");
        assert!(preset.background.is_some(), "default must set background");
    }

    #[test]
    fn test_parse_input_style_reverse() {
        assert!(parse_input_style("reverse").is_reverse);
    }

    #[test]
    fn test_parse_input_style_unknown_falls_back_to_default() {
        // Invalid keywords warn but must not panic — fall back to the same preset used when the key
        // is unset.
        assert_eq!(parse_input_style("superbold"), default_input_style());
    }

    #[test]
    fn test_tools_config_deserialization() {
        let toml_str = r#"
[tools]
allowed_tools = ["read_file", "find_files"]
disabled_tools = ["web_search"]

[tools.tool_permissions]
execute_command = "write"
read_file = "ask"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let tools = config.tools.expect("tools should be present");
        assert_eq!(
            tools.allowed_tools.as_deref(),
            Some(["read_file".to_string(), "find_files".to_string()].as_slice())
        );
        assert_eq!(
            tools.disabled_tools.as_deref(),
            Some(["web_search".to_string()].as_slice())
        );
        let perms = tools.tool_permissions.expect("tool_permissions set");
        assert_eq!(
            perms.get("execute_command").map(String::as_str),
            Some("write")
        );
        assert_eq!(perms.get("read_file").map(String::as_str), Some("ask"));
    }

    #[test]
    fn test_tools_config_missing_is_none() {
        let config: ConfigFile = toml::from_str("").expect("failed to parse empty toml");
        assert!(config.tools.is_none());
    }

    #[test]
    fn test_tools_config_invalid_permission_drops_entry() {
        // Drive the post-parse filter directly — ResolvedConfig::from_cli runs this loop. Checks
        // that a bad level string is filtered out without panicking and that valid entries still
        // land.
        let raw: HashMap<String, String> = [
            ("read_file".to_string(), "write".to_string()),
            ("write_file".to_string(), "superuser".to_string()),
            ("find_files".to_string(), "read".to_string()),
        ]
        .into_iter()
        .collect();
        let parsed: HashMap<String, Permission> = raw
            .into_iter()
            .filter_map(|(name, level)| level.parse::<Permission>().ok().map(|p| (name, p)))
            .collect();
        assert_eq!(parsed.get("read_file").copied(), Some(Permission::Write));
        assert_eq!(parsed.get("find_files").copied(), Some(Permission::Read));
        assert!(
            !parsed.contains_key("write_file"),
            "invalid level 'superuser' must be dropped"
        );
    }

    #[test]
    fn test_prompt_config_deserialization() {
        let toml_str = r#"
[prompt]
instructions = "Never use pip."
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let prompt = config.prompt.expect("prompt should be present");
        assert_eq!(prompt.instructions.as_deref(), Some("Never use pip."));
    }

    #[test]
    fn test_prompt_config_missing() {
        let config: ConfigFile = toml::from_str("").expect("failed to parse empty toml");
        assert!(config.prompt.is_none());
    }

    #[test]
    fn test_prompt_config_multiline() {
        let toml_str = r#"
[prompt]
instructions = """
Rule 1.
Rule 2.
"""
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let prompt = config.prompt.expect("prompt should be present");
        let instructions = prompt.instructions.expect("instructions should be set");
        assert!(instructions.contains("Rule 1."));
        assert!(instructions.contains("Rule 2."));
    }

    #[test]
    fn test_user_instructions_whitespace_only_is_none() {
        let file_prompt = PromptConfig {
            instructions: Some("   \n\t  ".to_string()),
        };
        let resolved = file_prompt
            .instructions
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        assert!(resolved.is_none());
    }

    #[test]
    fn test_user_instructions_trimmed() {
        let file_prompt = PromptConfig {
            instructions: Some("  hello  \n".to_string()),
        };
        let resolved = file_prompt
            .instructions
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        assert_eq!(resolved.as_deref(), Some("hello"));
    }

    #[test]
    fn test_user_instructions_cli_overrides_config() {
        // Replicates the merge chain in `from_cli`: CLI value wins over config.
        let cli: Option<String> = Some("from cli".to_string());
        let env: Option<String> = None;
        let file: Option<String> = Some("from config".to_string());
        let resolved = cli
            .or(env)
            .or(file)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        assert_eq!(resolved.as_deref(), Some("from cli"));
    }

    #[test]
    fn test_user_instructions_falls_through_to_config_when_cli_unset() {
        let cli: Option<String> = None;
        let env: Option<String> = None;
        let file: Option<String> = Some("from config".to_string());
        let resolved = cli
            .or(env)
            .or(file)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        assert_eq!(resolved.as_deref(), Some("from config"));
    }

    /// End-to-end check that AGSH_INSTRUCTIONS overrides `[prompt].instructions` when
    /// `--instructions` is not on the CLI. Drives the actual `from_cli` path against a
    /// tempdir-backed config to catch any regression where the env-var read silently no-ops.
    ///
    /// Touches process env, so it serializes against any other env-var test in this file via
    /// `ENV_LOCK`.
    #[test]
    fn test_env_var_overrides_config_file_instructions() {
        use std::sync::{Mutex, OnceLock};
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            "[prompt]\ninstructions = \"FROM CONFIG FILE\"\n",
        )
        .expect("write config.toml");

        // SAFETY: ENV_LOCK serializes this with any other env-var test.
        unsafe {
            std::env::set_var("AGSH_CONFIG_DIR", dir.path());
            std::env::set_var("AGSH_INSTRUCTIONS", "FROM ENV VAR");
        }

        use clap::Parser;
        let cli = crate::cli::Cli::parse_from(["agsh"]);
        let resolved = ResolvedConfig::from_cli(&cli);

        // SAFETY: same as above — ENV_LOCK held for the full set→read→clear cycle.
        unsafe {
            std::env::remove_var("AGSH_CONFIG_DIR");
            std::env::remove_var("AGSH_INSTRUCTIONS");
        }

        assert_eq!(
            resolved.user_instructions.as_deref(),
            Some("FROM ENV VAR"),
            "AGSH_INSTRUCTIONS should override [prompt].instructions in the config file",
        );
    }

    #[test]
    fn test_write_config_atomic_writes_content_and_no_tmp_left() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("sub").join("config.toml");
        write_config_atomic(&path, "[x]\nk = 1\n").expect("atomic write");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read back"),
            "[x]\nk = 1\n"
        );
        // The temporary file must not be left behind after a successful write.
        let tmp = dir.path().join("sub").join("config.toml.tmp");
        assert!(!tmp.exists(), "temp file should not remain: {:?}", tmp);
    }

    #[test]
    fn test_write_config_atomic_overwrites_existing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "old contents that are LONGER than the new ones").expect("seed file");
        write_config_atomic(&path, "new\n").expect("atomic overwrite");
        assert_eq!(std::fs::read_to_string(&path).expect("read back"), "new\n");
    }

    #[cfg(unix)]
    #[test]
    fn test_write_config_atomic_sets_unix_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let parent = dir.path().join("agsh");
        let path = parent.join("config.toml");
        write_config_atomic(&path, "x = 1\n").expect("atomic write");

        let file_mode = std::fs::metadata(&path)
            .expect("stat file")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            file_mode, 0o600,
            "config file should be 0600, got {:o}",
            file_mode
        );

        let dir_mode = std::fs::metadata(&parent)
            .expect("stat dir")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            dir_mode, 0o700,
            "config dir should be 0700, got {:o}",
            dir_mode
        );
    }

    fn enabled_set(modes: &[Permission]) -> EnabledPermissions {
        EnabledPermissions::from_modes(modes.iter().copied()).unwrap()
    }

    fn enabled_strings(modes: &[&str]) -> Vec<String> {
        modes.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_resolve_permission_no_config() {
        let (perm, enabled) = resolve_permission(None, None, None, None);
        assert_eq!(perm, Permission::Read);
        assert_eq!(enabled, EnabledPermissions::DEFAULT);
    }

    #[test]
    fn test_resolve_permission_explicit_enabled_all() {
        let list = enabled_strings(&["none", "read", "ask", "write"]);
        let (_perm, enabled) = resolve_permission(None, None, None, Some(&list));
        assert!(enabled.is_enabled(Permission::Ask));
        assert_eq!(enabled.iter().count(), 4);
    }

    #[test]
    fn test_resolve_permission_invalid_entry_warns_and_drops() {
        let list = enabled_strings(&["read", "lol", "write"]);
        let (perm, enabled) = resolve_permission(None, None, None, Some(&list));
        assert_eq!(enabled, enabled_set(&[Permission::Read, Permission::Write]));
        assert_eq!(perm, Permission::Read);
    }

    #[test]
    fn test_resolve_permission_empty_enabled_falls_back_to_default() {
        let list: Vec<String> = vec![];
        let (perm, enabled) = resolve_permission(None, None, None, Some(&list));
        assert_eq!(enabled, EnabledPermissions::DEFAULT);
        assert_eq!(perm, Permission::Read);
    }

    #[test]
    fn test_resolve_permission_default_not_in_enabled_clamps() {
        let list = enabled_strings(&["read", "write"]);
        let (perm, _enabled) = resolve_permission(None, None, Some("ask"), Some(&list));
        // `ask` is filtered out → fall back to Read because it's enabled.
        assert_eq!(perm, Permission::Read);
    }

    #[test]
    fn test_resolve_permission_default_not_in_enabled_no_read_falls_to_lowest() {
        let list = enabled_strings(&["ask", "write"]);
        let (perm, _enabled) = resolve_permission(None, None, Some("none"), Some(&list));
        // none isn't enabled, Read isn't either → lowest enabled is Ask.
        assert_eq!(perm, Permission::Ask);
    }

    #[test]
    fn test_resolve_permission_invalid_default_falls_back() {
        let (perm, _enabled) = resolve_permission(None, None, Some("weird"), None);
        assert_eq!(perm, Permission::Read);
    }

    #[test]
    fn test_resolve_permission_explicit_default_used() {
        let (perm, _enabled) = resolve_permission(None, None, Some("write"), None);
        assert_eq!(perm, Permission::Write);
    }

    #[test]
    fn test_resolve_permission_cli_override_disabled_clamps_to_default() {
        // `ask` not enabled → CLI request for ask warns and clamps to the configured default
        // (Read).
        let (perm, _enabled) = resolve_permission(Some(Permission::Ask), None, None, None);
        assert_eq!(perm, Permission::Read);
    }

    #[test]
    fn test_resolve_permission_cli_override_enabled_wins() {
        let list = enabled_strings(&["none", "read", "ask", "write"]);
        let (perm, _enabled) = resolve_permission(Some(Permission::Ask), None, None, Some(&list));
        assert_eq!(perm, Permission::Ask);
    }

    #[test]
    fn test_resolve_permission_env_override_used() {
        let (perm, _enabled) = resolve_permission(None, Some("write"), None, None);
        assert_eq!(perm, Permission::Write);
    }

    #[test]
    fn test_resolve_permission_env_override_disabled_clamps() {
        // env asks for ask, but ask isn't in DEFAULT enabled set.
        let (perm, _enabled) = resolve_permission(None, Some("ask"), None, None);
        assert_eq!(perm, Permission::Read);
    }

    #[test]
    fn test_resolve_permission_cli_beats_env() {
        let (perm, _enabled) =
            resolve_permission(Some(Permission::None), Some("write"), None, None);
        assert_eq!(perm, Permission::None);
    }

    #[test]
    fn test_permissions_config_deserialization() {
        let toml_str = r#"
[permissions]
default = "write"
enabled = ["read", "write"]
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("parse toml");
        let perms = config.permissions.expect("permissions present");
        assert_eq!(perms.default.as_deref(), Some("write"));
        assert_eq!(
            perms.enabled.as_deref(),
            Some(&[String::from("read"), String::from("write")][..])
        );
    }

    /// `sandbox_backend = "bubblewrap"` and `"landlock"` deserialize cleanly. Any other value,
    /// including the prior internal alias `"bwrap"`, must be rejected — we don't want alias creep
    /// that would silently desync `agsh setup`-generated configs from hand-edited ones.
    #[test]
    fn test_sandbox_backend_deserializes_strict_values() {
        let bubblewrap: ShellConfig =
            toml::from_str(r#"sandbox_backend = "bubblewrap""#).expect("deserialize bubblewrap");
        assert_eq!(bubblewrap.sandbox_backend, Some(SandboxBackend::Bubblewrap));
        let landlock: ShellConfig =
            toml::from_str(r#"sandbox_backend = "landlock""#).expect("deserialize landlock");
        assert_eq!(landlock.sandbox_backend, Some(SandboxBackend::Landlock));
        // No aliases / case variants accepted.
        assert!(toml::from_str::<ShellConfig>(r#"sandbox_backend = "bwrap""#).is_err());
        assert!(toml::from_str::<ShellConfig>(r#"sandbox_backend = "Bubblewrap""#).is_err());
        assert!(toml::from_str::<ShellConfig>(r#"sandbox_backend = "none""#).is_err());
    }

    /// When the user pins `sandbox_backend = "..."` explicitly, the resolver returns that choice
    /// with `auto_resolved == false` — no silent fallback even if the probe would suggest
    /// otherwise.
    #[cfg(target_os = "linux")]
    #[test]
    fn test_resolve_sandbox_backend_explicit_value_is_binding() {
        let (backend, auto_resolved, _probe) =
            resolve_sandbox_backend(Some(SandboxBackend::Landlock));
        assert_eq!(backend, SandboxBackend::Landlock);
        assert!(!auto_resolved);

        let (backend, auto_resolved, _probe) =
            resolve_sandbox_backend(Some(SandboxBackend::Bubblewrap));
        assert_eq!(backend, SandboxBackend::Bubblewrap);
        assert!(!auto_resolved);
    }

    /// When the user has not pinned a backend, resolve_sandbox_backend must surface `auto_resolved
    /// == true`. The exact backend it picks depends on whether the host has bwrap installed and
    /// supports user namespaces, so we just assert the auto flag is set and one of the two backends
    /// came back.
    #[cfg(target_os = "linux")]
    #[test]
    fn test_resolve_sandbox_backend_auto_resolves_when_unset() {
        let (backend, auto_resolved, _probe) = resolve_sandbox_backend(None);
        assert!(auto_resolved);
        assert!(matches!(
            backend,
            SandboxBackend::Bubblewrap | SandboxBackend::Landlock
        ));
    }

    /// On macOS / Windows the `sandbox_backend` config field is documented as ignored. The resolver
    /// must still return a probe that reflects the platform's native sandbox capability rather than
    /// the never-applicable Linux defaults, so the downstream wiring in `src/main.rs` can map it to
    /// `SandboxCapability` for `sandbox-exec` / Low-integrity. Guards against the regression that
    /// surfaces when only the Linux probe paths are wired up.
    #[cfg(not(target_os = "linux"))]
    #[test]
    fn test_resolve_sandbox_backend_uses_platform_sandbox_on_non_linux() {
        use crate::sandbox::{BackendProbe, SandboxCapability};

        // Explicit `Some(...)` is ignored on non-Linux — the field is documented as Linux-only.
        let (_backend, auto_resolved, probe) =
            resolve_sandbox_backend(Some(SandboxBackend::Bubblewrap));
        assert!(!auto_resolved);
        // The probe should reflect what `detect()` reports for this host, surfaced as `Ok(...)` so
        // the consumer can drop into the platform's spawn path.
        match probe {
            BackendProbe::Ok(SandboxCapability::Unavailable) => {
                panic!("Ok(Unavailable) is incoherent — expected a real capability or Missing")
            }
            BackendProbe::Ok(_) | BackendProbe::Missing { .. } => {}
            other => panic!("unexpected probe variant on non-Linux: {:?}", other),
        }
    }
}
