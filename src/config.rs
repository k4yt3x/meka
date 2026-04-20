//! Configuration: parses `~/.config/agsh/config.toml`, layers CLI overrides
//! and environment variables on top, and produces a [`ResolvedConfig`] that
//! the rest of the binary consumes.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::cli::Cli;
use crate::permission::Permission;
use crate::provider::AuthCredential;
use crate::render::RenderMode;

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
    /// Fallback permission for MCP tools when nothing more specific
    /// applies (no `tool_permissions` override, no server-level
    /// `permission`, no `readOnlyHint` from the server). If this is
    /// also unset the hardcoded fallback is `Write` — i.e. strict.
    pub default_permission: Option<String>,
    pub servers: Option<Vec<McpServerConfig>>,
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
    /// Optional path to an executable that, when run, prints dynamic HTTP
    /// headers to stdout in `Name: Value\n` form. Merged over [`Self::headers`]
    /// (dynamic wins). Useful for SSO flows where bearer tokens rotate.
    /// The script is spawned with `AGSH_MCP_SERVER_NAME` and
    /// `AGSH_MCP_SERVER_URL` in its environment so one helper can drive
    /// multiple servers. Non-zero exit fails the connect.
    pub headers_helper: Option<String>,
    pub auth: Option<McpAuthConfig>,
    pub permission: Option<String>,
    /// Optional allow-list of raw tool names (the server-advertised
    /// form, not the `<server>__<tool>` namespaced form). When set and
    /// non-empty, only these tools from this server are registered.
    pub allowed_tools: Option<Vec<String>>,
    /// Optional block-list of raw tool names. Applied after
    /// [`allowed_tools`] — tools listed here are never registered.
    pub disabled_tools: Option<Vec<String>>,
    /// Optional per-tool permission overrides keyed by raw tool name.
    /// Beats the server-level `permission` and the server's
    /// `readOnlyHint` annotation when resolving a tool's required
    /// permission at registration time.
    pub tool_permissions: Option<std::collections::HashMap<String, String>>,
    /// Allow this server to issue `sampling/createMessage` requests. When
    /// false (default), any such request is rejected with `METHOD_NOT_FOUND`.
    /// Use with caution: sampling lets the server inject arbitrary messages
    /// into your LLM context and spend your provider quota.
    #[serde(default)]
    pub sampling: bool,
    /// Cap on the number of sampling calls this server may issue per agsh
    /// session. Only meaningful when `sampling = true`. Default: 10.
    pub sampling_limit: Option<u32>,
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
    pub render_mode: Option<RenderMode>,
}

#[derive(Debug, Deserialize, Default)]
pub struct WebConfig {
    pub user_agent: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ShellConfig {
    pub sandbox: Option<bool>,
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
}

#[derive(Debug)]
pub struct ResolvedConfig {
    pub provider_name: Option<String>,
    pub model: Option<String>,
    pub auth_credential: Option<AuthCredential>,
    pub base_url: Option<String>,
    pub client_id: Option<String>,
    pub oauth_token_url: Option<String>,
    pub permission: Permission,
    pub streaming: bool,
    pub continue_session: Option<String>,
    pub prompt: Option<String>,
    pub newline_before_prompt: bool,
    pub newline_after_prompt: bool,
    pub show_session_id_on_create: bool,
    pub show_session_id_on_exit: bool,
    pub show_path_in_prompt: bool,
    pub user_agent: String,
    pub sandbox: bool,
    pub render_mode: RenderMode,
    pub context_messages: Option<usize>,
    pub retention_days: Option<u64>,
    pub max_storage_bytes: Option<u64>,
    pub thinking_enabled: bool,
    pub thinking_budget_tokens: u64,
    pub thinking_show_content: bool,
    pub reasoning_effort: Option<String>,
    pub auto_compact: bool,
    pub context_window: Option<u64>,
    pub mcp_servers: Vec<McpServerConfig>,
    /// Parsed [`Permission`] from `[mcp].default_permission`, carried so
    /// per-turn tool-permission resolution in `src/mcp.rs` doesn't have
    /// to re-read the config file. `None` means "no `[mcp]` default
    /// configured" — resolution falls through to the hardcoded Write.
    pub mcp_default_permission: Option<Permission>,
    pub user_instructions: Option<String>,
}

pub(crate) fn config_file_path() -> Option<PathBuf> {
    dirs::config_dir().map(|directory| directory.join("agsh").join("config.toml"))
}

pub(crate) fn config_file_exists() -> bool {
    config_file_path().is_some_and(|path| path.exists())
}

/// Write `content` to `path` atomically: serialise to `<path>.tmp` in the
/// same directory, `sync_all` the fd, then `rename` over the target. Also
/// creates the parent directory (0700 on Unix) and chmods the final file
/// to 0600 on Unix so `auth_token` / OAuth-derived secrets aren't
/// world-readable regardless of the user's umask.
pub(crate) fn write_config_atomic(path: &Path, content: &str) -> std::io::Result<()> {
    use std::io::Write as _;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            // Best-effort — a pre-existing dir with different perms stays as-is.
            if let Ok(metadata) = std::fs::metadata(parent) {
                let mut perms = metadata.permissions();
                if perms.mode() & 0o777 != 0o700 {
                    perms.set_mode(0o700);
                    let _ = std::fs::set_permissions(parent, perms);
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

    // Create the tmp file with restrictive perms on Unix before any bytes
    // land on disk, so a concurrent reader never sees the partial content
    // with a looser mode.
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
        // Destructure the [mcp] table into its two independent fields so
        // we don't have to re-open the config file later for resolution.
        let (mcp_default_permission_str, mcp_servers) = match config_file.mcp {
            Some(mcp) => (mcp.default_permission, mcp.servers.unwrap_or_default()),
            None => (None, Vec::new()),
        };
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

        let user_instructions = file_prompt
            .instructions
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());

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

        let auth_credential = resolve_auth_credential(provider_name.as_deref(), &file_provider);

        let base_url = cli
            .base_url
            .clone()
            .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
            .or_else(|| file_provider.base_url.clone());

        let permission = cli
            .permission
            .or_else(|| {
                std::env::var("AGSH_PERMISSION")
                    .ok()
                    .and_then(|s| s.parse().ok())
            })
            .unwrap_or(Permission::Read);

        Self {
            provider_name,
            model,
            auth_credential,
            base_url,
            client_id: std::env::var("CLAUDE_CLIENT_ID").ok(),
            oauth_token_url: file_provider.oauth_token_url.clone(),
            permission,
            streaming: !cli.no_stream,
            continue_session: cli.continue_session.clone(),
            prompt: cli.prompt.clone(),
            newline_before_prompt: file_display.newline_before_prompt.unwrap_or(true),
            newline_after_prompt: file_display.newline_after_prompt.unwrap_or(true),
            show_session_id_on_create: file_display.show_session_id_on_create.unwrap_or(false),
            show_session_id_on_exit: file_display.show_session_id_on_exit.unwrap_or(true),
            show_path_in_prompt: file_display.show_path_in_prompt.unwrap_or(true),
            user_agent: file_web.user_agent.unwrap_or_else(|| {
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.3"
                    .to_string()
            }),
            sandbox: file_shell.sandbox.unwrap_or(true),
            render_mode: cli
                .render_mode
                .or(file_display.render_mode)
                .unwrap_or_default(),
            context_messages: file_session.context_messages.or(Some(200)),
            retention_days: file_session.retention_days.or(Some(90)),
            max_storage_bytes: file_session.max_storage_bytes.or(Some(52_428_800)),
            thinking_enabled: cli
                .thinking
                .unwrap_or_else(|| file_thinking.enabled.unwrap_or(true)),
            thinking_budget_tokens: cli
                .thinking_budget
                .unwrap_or_else(|| file_thinking.budget_tokens.unwrap_or(16_000)),
            thinking_show_content: file_thinking.show_content.unwrap_or(false),
            reasoning_effort: file_provider.reasoning_effort.clone(),
            auto_compact: file_session.auto_compact.unwrap_or(true),
            context_window: file_session.context_window,
            mcp_servers,
            mcp_default_permission,
            user_instructions,
        }
    }

    pub fn validate(&self) -> crate::error::Result<()> {
        if self.provider_name.is_none() {
            return Err(crate::error::AgshError::Config(
                "no provider configured. Set --provider, AGSH_PROVIDER env var, \
                 or provider.name in config file (~/.config/agsh/config.toml)"
                    .to_string(),
            ));
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

fn resolve_auth_credential(
    provider_name: Option<&str>,
    file_provider: &ProviderConfig,
) -> Option<AuthCredential> {
    match provider_name {
        Some("claude") => {
            // 1. CLAUDE_API_KEY env var (auto-detects OAuth vs API key by prefix)
            if let Ok(key) = std::env::var("CLAUDE_API_KEY") {
                return Some(AuthCredential::from_token_string(key));
            }
            // 2. CLAUDE_OAUTH_TOKEN env var (always treated as OAuth)
            if let Ok(token) = std::env::var("CLAUDE_OAUTH_TOKEN") {
                return Some(AuthCredential::OAuthToken {
                    access_token: token,
                    refresh_token: None,
                    expires_at: None,
                });
            }
            // 3. provider.api_key from config (auto-detects)
            if let Some(key) = &file_provider.api_key {
                return Some(AuthCredential::from_token_string(key.clone()));
            }
            // 4. provider.oauth_token from config
            if let Some(token) = &file_provider.oauth_token {
                return Some(AuthCredential::OAuthToken {
                    access_token: token.clone(),
                    refresh_token: None,
                    expires_at: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_file_deserialization() {
        let toml_str = r#"
[provider]
name = "openai"
model = "gpt-4o"
api_key = "sk-test"
base_url = "https://api.openai.com/v1"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let provider = config.provider.expect("provider should be present");
        assert_eq!(provider.name.as_deref(), Some("openai"));
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
name = "claude"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let provider = config.provider.expect("provider should be present");
        assert_eq!(provider.name.as_deref(), Some("claude"));
        assert!(provider.model.is_none());
        assert!(provider.api_key.is_none());
        assert!(provider.base_url.is_none());
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
}
