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
        /// Output file (default: `session-<id>.md`; `-` = stdout)
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
        /// Include sub-agent sessions (children of a parent session) in the
        /// listing. Hidden by default to keep the view focused on
        /// user-initiated conversations.
        #[arg(long)]
        include_children: bool,
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
    /// Manage user skills (list, add, remove, show)
    Skill {
        #[command(subcommand)]
        action: SkillAction,
    },
}

#[derive(clap::Subcommand, Debug)]
pub enum ToolsAction {
    /// List every built-in tool with its effective permission and status.
    List,
}

// `Add` is the outlier with several flags inline; same one-shot CLI
// dispatch reasoning as [`Command`] and [`McpAction`] above.
#[allow(clippy::large_enum_variant)]
#[derive(clap::Subcommand, Debug)]
pub enum SkillAction {
    /// List installed skills
    List,
    /// Print one skill's frontmatter and on-disk paths
    Get { name: String },
    /// Print the rendered skill body
    Show { name: String },
    /// Scaffold a new skill at `~/.config/agsh/skills/<name>/SKILL.md`.
    ///
    /// Examples:
    ///   agsh skill add demo --description "X"
    ///   agsh skill add custom --from-file ./template.md
    #[command(verbatim_doc_comment)]
    Add {
        /// Unique skill name (alphanumerics, `-`, `_` only)
        name: String,

        /// One-line description for the system prompt
        #[arg(long)]
        description: Option<String>,

        /// Version label
        #[arg(long)]
        version: Option<String>,

        /// Author, in `Name <email>` form
        #[arg(long)]
        author: Option<String>,

        /// https:// URL the skill can be re-fetched from by `skill update`
        #[arg(long = "source-url", value_name = "URL")]
        source_url: Option<String>,

        /// Copy this file's contents instead of the default template
        #[arg(long = "from-file", value_name = "PATH")]
        from_file: Option<std::path::PathBuf>,

        /// Overwrite the skill directory if it exists
        #[arg(long)]
        force: bool,

        /// Open the new SKILL.md in $EDITOR after scaffolding
        #[arg(long)]
        edit: bool,
    },
    /// Remove a skill's directory
    Remove { name: String },
    /// Re-fetch skills from their `source_url` and replace them on disk.
    ///
    /// Examples:
    ///   agsh skill update my-skill
    ///   agsh skill update --all          # dry run: lists what would update
    ///   agsh skill update --all --yes    # applies the updates
    #[command(verbatim_doc_comment)]
    Update {
        /// Skill name to update. Omit and pass --all to update every skill.
        name: Option<String>,

        /// Update every skill that declares a source_url
        #[arg(long)]
        all: bool,

        /// Apply --all updates (without this, --all only lists)
        #[arg(long)]
        yes: bool,
    },
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
    /// List a server's advertised tools with their resolved permissions
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
    // Preserve line breaks in the `Examples:` block; clap's default
    // joins consecutive `///` lines into one re-wrapped paragraph.
    #[command(verbatim_doc_comment)]
    Add {
        /// Unique server name (alphanumerics, `-`, `_` only)
        name: String,
        /// URL (HTTP) or executable path (stdio); transport auto-detected
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

        /// Raw tool name to allow (repeatable; restricts which register)
        #[arg(long = "allow-tool", value_name = "TOOL")]
        allow_tool: Vec<String>,

        /// Raw tool name to block (repeatable; applied after --allow-tool)
        #[arg(long = "disable-tool", value_name = "TOOL")]
        disable_tool: Vec<String>,

        /// Raw tool name to eager-load (repeatable; skips load_tool)
        #[arg(long = "eager-load-tool", value_name = "TOOL")]
        eager_load_tool: Vec<String>,

        /// Per-tool permission override (TOOL=LEVEL, repeatable)
        #[arg(long = "tool-permission", value_name = "TOOL=LEVEL")]
        tool_permission: Vec<String>,

        /// Skip post-add auto-login; run `agsh mcp login <name>` later
        #[arg(long = "no-login")]
        no_login: bool,

        /// Persist with disabled=true; re-enable via `agsh mcp enable`
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

    /// Resume a session (`-c` for last, `-c <UUID-PREFIX>` for specific)
    #[arg(short = 'c', long = "continue", num_args = 0..=1, default_missing_value = "last")]
    pub continue_session: Option<String>,

    /// Initial permission mode (none, read, ask, write)
    #[arg(long = "permission", value_parser = parse_permission)]
    pub permission: Option<Permission>,

    /// LLM provider: openai-api, openai-codex, claude-api, claude-oauth
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

    /// Markdown render mode: bat (default), rich, or raw
    #[arg(long = "render-mode", value_parser = parse_render_mode)]
    pub render_mode: Option<crate::render::RenderMode>,

    /// Enable extended thinking (Claude-only)
    #[arg(long = "thinking")]
    pub thinking: Option<bool>,

    /// Token budget for extended thinking (Claude-only)
    #[arg(long = "thinking-budget")]
    pub thinking_budget: Option<u64>,

    /// Override `[prompt].instructions` for this run (replaces config value).
    #[arg(long = "instructions", value_name = "STRING")]
    pub instructions: Option<String>,

    /// Invoke a user-invocable skill on the first turn.
    #[arg(long = "skill", value_name = "NAME")]
    pub skill: Option<String>,

    /// Exit after the first turn finishes (requires a prompt or `--skill`).
    #[arg(long = "oneshot")]
    pub oneshot: bool,

    /// Eager-load an MCP tool this session (raw SERVER:TOOL, repeatable)
    #[arg(long = "eager-load-tool", value_name = "SERVER:TOOL")]
    pub eager_load_tool: Vec<String>,

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
        assert!(cli.skill.is_none());
        assert!(!cli.oneshot);
        assert!(cli.eager_load_tool.is_empty());
        assert_eq!(cli.verbosity, 0);
    }

    #[test]
    fn test_cli_eager_load_tool_repeatable() {
        let cli = Cli::parse_from([
            "agsh",
            "--eager-load-tool",
            "notion:search",
            "--eager-load-tool",
            "github:create_issue",
        ]);
        assert_eq!(
            cli.eager_load_tool,
            vec![
                "notion:search".to_string(),
                "github:create_issue".to_string()
            ]
        );
    }

    #[test]
    fn test_cli_oneshot_flag() {
        let cli = Cli::parse_from(["agsh", "--oneshot", "do thing"]);
        assert!(cli.oneshot);
        assert_eq!(cli.prompt.as_deref(), Some("do thing"));
    }

    #[test]
    fn test_cli_oneshot_prompt() {
        let cli = Cli::parse_from(["agsh", "hello world"]);
        assert_eq!(cli.prompt.as_deref(), Some("hello world"));
    }

    #[test]
    fn test_cli_skill_flag_alone() {
        let cli = Cli::parse_from(["agsh", "--skill", "demo"]);
        assert_eq!(cli.skill.as_deref(), Some("demo"));
        assert!(cli.prompt.is_none());
    }

    #[test]
    fn test_cli_skill_flag_with_extra_prompt() {
        let cli = Cli::parse_from(["agsh", "--skill", "demo", "extra context"]);
        assert_eq!(cli.skill.as_deref(), Some("demo"));
        assert_eq!(cli.prompt.as_deref(), Some("extra context"));
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
