use clap::Parser;

use crate::permission::Permission;

#[derive(clap::Subcommand, Debug)]
pub enum Command {
    /// Run the interactive configuration wizard
    Setup,
    /// Export a session as Markdown
    Export {
        /// Session UUID to export
        session_id: uuid::Uuid,
        /// Output file path (default: session-<id>.md). Use "-" for stdout.
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

    /// LLM provider to use (openai, claude)
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
            "openai",
            "--model",
            "gpt-4o",
            "--no-stream",
            "-c",
            "-vv",
        ]);
        assert_eq!(cli.provider.as_deref(), Some("openai"));
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
