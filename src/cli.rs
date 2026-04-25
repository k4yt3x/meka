//! Clap-derived CLI definition. Owns the top-level argument struct, the
//! subcommand enum (`setup`, `export`, `delete`, `list`), and the small
//! parsers for permission/render-mode flag values.

use clap::Parser;

use crate::permission::Permission;

// `Mcp { action: McpAction }` is bigger than every other variant because
// `McpAction::Add` holds every CLI flag inline, but the enum is only ever
// constructed once per process by clap and held on the stack of `main`,
// so the few extra words of padding on the other variants aren't worth
// the indirection cost of boxing.
#[allow(clippy::large_enum_variant)]
#[derive(clap::Subcommand, Debug)]
pub enum Command {
    /// Run the interactive configuration wizard
    Setup,
    /// Export a session as Markdown
    Export {
        /// Session UUID to export
        session_id: uuid::Uuid,
        /// Output file path (default: `session-<id>.md`). Use "-" for stdout.
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Delete one or more sessions
    Delete {
        /// Session UUIDs to delete
        session_ids: Vec<uuid::Uuid>,
        /// Delete all sessions
        #[arg(long)]
        all: bool,
    },
    /// List past sessions
    List {
        /// Maximum number of sessions to show
        #[arg(short = 'n', long, default_value = "20")]
        limit: u32,
    },
    /// Manage MCP servers (list, add, remove, reconnect, login, logout)
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },
    /// Inspect built-in tool filters (allow/disable/permission overrides)
    Tools {
        #[command(subcommand)]
        action: ToolsAction,
    },
}

#[derive(clap::Subcommand, Debug)]
pub enum ToolsAction {
    /// List every built-in tool with its effective permission and status.
    List,
}

// Same reasoning as `Command` above: `Add` is the outlier and the enum
// lives on `main`'s stack for exactly one dispatch, not in a hot
// collection, so boxing would trade clarity for nothing.
#[allow(clippy::large_enum_variant)]
#[derive(clap::Subcommand, Debug)]
pub enum McpAction {
    /// List all configured MCP servers
    List,
    /// Print the configuration for one server
    Get { name: String },
    /// Connect once and print `ok` if the handshake succeeds
    Reconnect { name: String },
    /// List tools advertised by a server, with resolved permission per tool.
    Tools { name: String },
    /// Authenticate a server interactively (OAuth assumed for HTTP)
    Login { name: String },
    /// Revoke cached credentials for a server
    Logout { name: String },
    /// Add a server to config.toml.
    ///
    /// Examples:
    ///   agsh mcp add pg npx -y @modelcontextprotocol/server-postgres
    ///   agsh mcp add notion https://mcp.notion.com/mcp
    ///   agsh mcp add api https://api.example.com/mcp --auth-token $API_TOKEN
    ///   agsh mcp add notion https://mcp.notion.com/mcp --auth oauth
    // `rustdoc::bare_urls` normally turns URLs like https://example into
    // auto-links, but these doc lines are ALSO the text clap prints for
    // `agsh mcp add --help`. Angle-brackets would leak into the CLI
    // help. Allow bare URLs just on this variant.
    #[allow(rustdoc::bare_urls)]
    Add {
        /// Unique server name (alphanumerics, `-`, `_` only)
        name: String,
        /// URL (for HTTP) or executable path (for stdio). Transport is
        /// auto-detected from this value unless `--transport` is given.
        location: Option<String>,
        /// Arguments to pass to the stdio command.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,

        /// Force transport (stdio or http); auto-detected otherwise
        #[arg(long, value_parser = parse_mcp_transport)]
        transport: Option<crate::config::McpTransport>,

        /// Environment variable for stdio (KEY=VALUE, repeatable)
        #[arg(long = "env", value_name = "KEY=VALUE")]
        env: Vec<String>,

        /// HTTP header (KEY=VALUE, repeatable)
        #[arg(long = "header", value_name = "KEY=VALUE")]
        header: Vec<String>,

        /// Authentication: oauth | client-credentials | client-credentials-jwt
        #[arg(long, value_parser = parse_mcp_auth_kind)]
        auth: Option<McpAuthKind>,

        /// Static bearer token for HTTP (mutually exclusive with --auth)
        #[arg(long)]
        auth_token: Option<String>,

        /// OAuth / client-credentials client ID
        #[arg(long)]
        client_id: Option<String>,

        /// OAuth / client-credentials client secret
        #[arg(long)]
        client_secret: Option<String>,

        /// JWT signing key path (for client-credentials-jwt)
        #[arg(long)]
        signing_key: Option<String>,

        /// JWT signing algorithm (RS256, RS384, RS512, ES256, ES384)
        #[arg(long)]
        signing_algorithm: Option<String>,

        /// OAuth scope (repeatable)
        #[arg(long = "scope", value_name = "SCOPE")]
        scope: Vec<String>,

        /// Fixed OAuth redirect port (default: ephemeral)
        #[arg(long)]
        redirect_port: Option<u16>,

        /// Permission: none, read, ask, write (default: read)
        #[arg(long)]
        permission: Option<String>,

        /// Allow this server to call sampling/createMessage
        #[arg(long)]
        sampling: bool,

        /// Max sampling calls per agsh session (default 10)
        #[arg(long)]
        sampling_limit: Option<u32>,

        /// Raw tool name to allow (repeatable). When set, only listed
        /// tools from this server are registered.
        #[arg(long = "allow-tool", value_name = "TOOL")]
        allow_tool: Vec<String>,

        /// Raw tool name to block (repeatable). Applied after --allow-tool.
        #[arg(long = "disable-tool", value_name = "TOOL")]
        disable_tool: Vec<String>,

        /// Per-tool permission override (repeatable).
        /// Format: `TOOL=LEVEL`, where LEVEL is none/read/ask/write.
        #[arg(long = "tool-permission", value_name = "TOOL=LEVEL")]
        tool_permission: Vec<String>,

        /// Skip the post-add auto-login even if the server requires auth.
        /// The server is still persisted; run `agsh mcp login <name>`
        /// later to authorise.
        #[arg(long = "no-login")]
        no_login: bool,

        /// Add the server entry with `disabled = true` so it's skipped
        /// at startup until `agsh mcp enable <name>` runs.
        #[arg(long = "disabled")]
        disabled: bool,
    },
    /// Remove a server from config.toml and clear stored creds
    Remove { name: String },
    /// Temporarily turn off a server without removing it from config
    Disable { name: String },
    /// Turn a disabled server back on
    Enable { name: String },
}

/// Authentication flavours selectable from the CLI. Maps onto the
/// [`crate::config::McpAuthConfig`] variants, except `None` which means
/// "no `[auth]` block at all" (static token or unauthenticated).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum McpAuthKind {
    OAuth,
    ClientCredentials,
    ClientCredentialsJwt,
}

fn parse_mcp_transport(s: &str) -> std::result::Result<crate::config::McpTransport, String> {
    match s.to_ascii_lowercase().as_str() {
        "stdio" => Ok(crate::config::McpTransport::Stdio),
        "http" => Ok(crate::config::McpTransport::Http),
        other => Err(format!(
            "unknown transport '{}' (expected stdio or http)",
            other
        )),
    }
}

fn parse_mcp_auth_kind(s: &str) -> std::result::Result<McpAuthKind, String> {
    match s.to_ascii_lowercase().as_str() {
        "oauth" => Ok(McpAuthKind::OAuth),
        "client-credentials" | "client_credentials" => Ok(McpAuthKind::ClientCredentials),
        "client-credentials-jwt" | "client_credentials_jwt" => {
            Ok(McpAuthKind::ClientCredentialsJwt)
        }
        other => Err(format!(
            "unknown auth '{}' (expected oauth, client-credentials, or client-credentials-jwt)",
            other
        )),
    }
}

#[derive(Parser, Debug)]
#[command(name = "agsh", version, about = "An agentic shell")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Run a one-shot prompt and exit
    pub prompt: Option<String>,

    /// Continue a session. Use -c to resume the last session,
    /// or -c SESSION_ID to resume a specific session.
    #[arg(short = 'c', long = "continue", num_args = 0..=1, default_missing_value = "last")]
    pub continue_session: Option<String>,

    /// Initial permission mode (none, read, write)
    #[arg(long = "permission", value_parser = parse_permission)]
    pub permission: Option<Permission>,

    /// LLM provider to use (openai-api, openai-codex, claude-api, claude-oauth)
    #[arg(long = "provider")]
    pub provider: Option<String>,

    /// Model name
    #[arg(short = 'm', long = "model")]
    pub model: Option<String>,

    /// API base URL (for OpenAI-compatible providers)
    #[arg(long = "base-url")]
    pub base_url: Option<String>,

    /// Disable streaming mode
    #[arg(long = "no-stream")]
    pub no_stream: bool,

    /// Output render mode: 'rich' (default) or 'raw' (markdown with ANSI highlighting)
    #[arg(long = "render-mode", value_parser = parse_render_mode)]
    pub render_mode: Option<crate::render::RenderMode>,

    /// Enable extended thinking (Claude-only)
    #[arg(long = "thinking")]
    pub thinking: Option<bool>,

    /// Token budget for extended thinking (Claude-only)
    #[arg(long = "thinking-budget")]
    pub thinking_budget: Option<u64>,

    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count)]
    pub verbosity: u8,
}

fn parse_permission(s: &str) -> std::result::Result<Permission, String> {
    s.parse()
}

fn parse_render_mode(s: &str) -> std::result::Result<crate::render::RenderMode, String> {
    s.parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_defaults() {
        let cli = Cli::parse_from(["agsh"]);
        assert!(cli.command.is_none());
        assert!(cli.prompt.is_none());
        assert!(cli.continue_session.is_none());
        assert!(cli.permission.is_none());
        assert!(cli.provider.is_none());
        assert!(cli.model.is_none());
        assert!(cli.base_url.is_none());
        assert!(!cli.no_stream);
        assert!(cli.render_mode.is_none());
        assert_eq!(cli.verbosity, 0);
    }

    #[test]
    fn test_cli_oneshot_prompt() {
        let cli = Cli::parse_from(["agsh", "hello world"]);
        assert_eq!(cli.prompt.as_deref(), Some("hello world"));
    }

    #[test]
    fn test_cli_continue_last() {
        let cli = Cli::parse_from(["agsh", "-c"]);
        assert_eq!(cli.continue_session.as_deref(), Some("last"));
    }

    #[test]
    fn test_cli_continue_specific_session() {
        let cli = Cli::parse_from(["agsh", "-c", "550e8400-e29b-41d4-a716-446655440000"]);
        assert_eq!(
            cli.continue_session.as_deref(),
            Some("550e8400-e29b-41d4-a716-446655440000")
        );
    }

    #[test]
    fn test_cli_flags() {
        let cli = Cli::parse_from([
            "agsh",
            "--provider",
            "openai-api",
            "--model",
            "gpt-4o",
            "--no-stream",
            "-c",
            "-vv",
        ]);
        assert_eq!(cli.provider.as_deref(), Some("openai-api"));
        assert_eq!(cli.model.as_deref(), Some("gpt-4o"));
        assert!(cli.no_stream);
        assert_eq!(cli.continue_session.as_deref(), Some("last"));
        assert_eq!(cli.verbosity, 2);
    }

    #[test]
    fn test_cli_permission_flag() {
        let cli = Cli::parse_from(["agsh", "--permission", "write"]);
        assert_eq!(cli.permission, Some(Permission::Write));
    }

    #[test]
    fn test_cli_continue_long_form() {
        let cli = Cli::parse_from(["agsh", "--continue"]);
        assert_eq!(cli.continue_session.as_deref(), Some("last"));
    }

    #[test]
    fn test_cli_continue_long_form_with_id() {
        let cli = Cli::parse_from(["agsh", "--continue", "550e8400-e29b-41d4-a716-446655440000"]);
        assert_eq!(
            cli.continue_session.as_deref(),
            Some("550e8400-e29b-41d4-a716-446655440000")
        );
    }

    #[test]
    fn test_cli_setup_subcommand() {
        let cli = Cli::parse_from(["agsh", "setup"]);
        assert!(matches!(cli.command, Some(Command::Setup)));
    }
}
