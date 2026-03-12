use clap::Parser;

use crate::permission::Permission;

#[derive(clap::Subcommand, Debug)]
pub enum Command {
    /// Run the interactive configuration wizard
    Setup,
}

#[derive(Parser, Debug)]
#[command(name = "agsh", version, about = "An agentic shell")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Run a one-shot prompt and exit
    pub prompt: Option<String>,

    /// Session ID to resume
    #[arg(short = 's', long = "session")]
    pub session_id: Option<uuid::Uuid>,

    /// Continue the last session
    #[arg(short = 'c', long = "continue")]
    pub continue_last: bool,

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

    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count)]
    pub verbosity: u8,
}

fn parse_permission(s: &str) -> std::result::Result<Permission, String> {
    s.parse()
}
