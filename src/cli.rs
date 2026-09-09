//! Clap-derived CLI definition: the root argument struct, the subcommand enums, and the value enums
//! a flag parses into through their `FromStr`.

use clap::Parser;

use crate::permission::Permission;

pub(crate) mod account;
pub(crate) mod background;
pub(crate) mod history;
pub(crate) mod instructions;
pub(crate) mod mcp;
pub(crate) mod memory;
pub(crate) mod profile;
pub(crate) mod schedule;
pub(crate) mod session;
pub(crate) mod skills;
pub(crate) mod tools;

#[allow(
    clippy::large_enum_variant,
    reason = "`McpAction::Add` holds every flag inline; clap builds the enum once and `main` holds it on the stack, so boxing buys nothing"
)]
#[derive(clap::Subcommand, Debug)]
pub(crate) enum Command {
    /// Manage accounts and their credentials
    Account {
        #[command(subcommand)]
        action: AccountAction,
    },
    /// Manage profiles
    Profile {
        #[command(subcommand)]
        action: ProfileAction,
    },
    /// Manage stored sessions
    Session {
        #[command(subcommand)]
        action: SessionAction,
    },
    /// View or clear REPL input history
    History {
        #[command(subcommand)]
        action: HistoryAction,
    },
    /// Manage MCP servers
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },
    /// Inspect built-in tool filters
    Tools {
        #[command(subcommand)]
        action: ToolsAction,
    },
    /// Manage user skills
    Skill {
        #[command(subcommand)]
        action: SkillAction,
    },
    /// Manage the agent's saved memories
    Memory {
        #[command(subcommand)]
        action: MemoryAction,
    },
    /// Show the standing instructions the agent is given
    Instructions {
        #[command(subcommand)]
        action: InstructionsAction,
    },
    /// Inspect and cancel scheduled jobs
    Schedule {
        #[command(subcommand)]
        action: ScheduleAction,
    },
    /// Run meka as an ACP (Agent Client Protocol) agent over stdio
    ///
    /// Speaks newline-framed JSON-RPC on stdin and stdout; diagnostics go to stderr.
    Acp,
    /// Run meka as a long-lived HTTP service
    ///
    /// Exposes the agent over HTTP and JSON; auth, session GC and SSE streaming are configured
    /// under `[serve]` in config.toml.
    Serve {
        /// Override `[serve].bind`
        #[arg(long, value_name = "ADDR")]
        bind: Option<String>,
    },
}

#[derive(clap::Subcommand, Debug)]
pub(crate) enum ToolsAction {
    /// List every built-in tool with its effective permission and status
    List {
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
}

#[derive(clap::Subcommand, Debug)]
pub(crate) enum SessionAction {
    /// List past sessions
    List {
        /// Maximum number of sessions to show
        #[arg(short = 'n', long, default_value = "20")]
        limit: u32,
        /// Include sub-agent sessions in the listing
        ///
        /// Hidden by default, so the view stays on conversations you started.
        #[arg(long)]
        include_children: bool,
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
    /// Export a session as Markdown or JSON
    Export {
        /// Session id, or any unique prefix of one
        session_id: String,
        /// Output file (`-` for stdout)
        ///
        /// Defaults to `session-<id>.md` for markdown or `session-<id>.json` for json, written to
        /// the current directory.
        #[arg(short, long)]
        output: Option<String>,
        /// Output format: markdown or json
        ///
        /// `json` is structured and round-trippable via `meka session import`, and includes any
        /// sub-agent child sessions. `markdown` is rendered and covers the single session only.
        #[arg(long, default_value = "markdown")]
        format: SessionExportFormat,
    },
    /// Delete sessions by id, by age, or all of them
    Delete {
        /// Session ids, or any unique prefix of each
        session_ids: Vec<String>,
        /// Delete all sessions
        // Conflicts with explicit ids: naming some sessions and then asking for every session is
        // two different requests, and running the wider one makes the narrower one look honored.
        #[arg(long, conflicts_with_all = ["older_than_days", "session_ids"])]
        all: bool,
        /// Delete sessions not updated in this many days
        // Conflicts with explicit ids rather than ignoring them: a listed session younger than the
        // window would otherwise be silently spared.
        #[arg(
            long = "older-than-days",
            value_name = "DAYS",
            conflicts_with = "session_ids"
        )]
        older_than_days: Option<u64>,
    },
    /// Import a session from a JSON export
    ///
    /// Recreates the session and any sub-agent children under fresh ids, and prints the new root
    /// session id.
    Import {
        /// Export file to read (`-` for stdin)
        input: String,
    },
    /// Fork a session into an independent copy
    ///
    /// The copy carries the full conversation; the original is untouched. Prints the new session
    /// id.
    Fork {
        /// Session id, or any unique prefix of one
        session_id: String,
    },
    /// Show a session's full details
    Show {
        /// Session id, or any unique prefix of one
        session_id: String,
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
    /// Drop the most recent turns from a session
    ///
    /// Cuts at a user boundary, so no tool call is separated from its result, and `meka session
    /// export` still shows the dropped turns. Recovers a session the provider rejects.
    Rewind {
        /// Session id, or any unique prefix of one
        session_id: String,
        /// Number of turns to drop
        #[arg(short = 'n', long, default_value = "1")]
        turns: usize,
    },
}

#[derive(clap::Subcommand, Debug)]
pub(crate) enum HistoryAction {
    /// List recorded input history
    List {
        /// Maximum number of entries to show (0 = all)
        #[arg(short = 'n', long, default_value = "50")]
        limit: u32,
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
    /// Delete all recorded input history
    Clear,
}

/// Accounts: what a login produces and what bills. `add`/`login`/`list`/`remove` manage the
/// `[accounts.<name>]` tables and the credential each holds; `usage`/`whoami`/`stats` are the
/// read-only views, each reached through a profile because a request needs a model.
#[allow(
    clippy::large_enum_variant,
    reason = "`Add` holds several flags inline; built once per process, so boxing buys nothing"
)]
#[derive(clap::Subcommand, Debug)]
pub(crate) enum AccountAction {
    /// Add an account and authenticate it
    ///
    /// Prompts for the backend and base URL when not flagged, then acquires the secret (an OAuth
    /// login for the subscription backends, an API-key prompt for the rest) and saves it to the
    /// store.
    Add {
        /// Account name
        name: String,
        /// Backend: anthropic-messages, chatgpt-subscription, claude-subscription,
        /// openai-chat-completions, openai-responses
        #[arg(long, value_name = "BACKEND")]
        backend: Option<String>,
        /// API base URL; any endpoint serving the backend's protocol
        #[arg(long = "base-url", value_name = "URL")]
        base_url: Option<String>,
        /// OAuth token endpoint override, used for the code exchange and every refresh
        #[arg(long = "oauth-token-url", value_name = "URL")]
        oauth_token_url: Option<String>,
        /// OAuth client id override (subscription backends only)
        #[arg(long = "client-id", value_name = "ID")]
        client_id: Option<String>,
        /// Read the API key from stdin (API-key backends only); needs `--backend`
        #[arg(long = "api-key-stdin")]
        api_key_stdin: bool,
    },
    /// List configured accounts
    List {
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
    /// Re-authenticate an account, keeping its settings
    Login {
        /// Account name
        name: String,
        /// Read the API key from stdin (API-key backends only)
        #[arg(long = "api-key-stdin")]
        api_key_stdin: bool,
    },
    /// Remove an account and clear its stored credential
    ///
    /// Refused while any profile names the account.
    Remove {
        /// Account name
        name: String,
    },
    /// Rename an account; its credential and every profile on it follow
    Rename {
        /// Account name
        name: String,
        /// New name
        new_name: String,
    },
    /// Show the account's rate-limit usage
    ///
    /// The session and weekly windows, with their reset times.
    Usage {
        /// Profile to reach the account through (default: the profile a new session runs on)
        #[arg(long, value_name = "NAME")]
        profile: Option<String>,
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
    /// Show who the account is and whether it is logged in
    ///
    /// The plan, tier, organization and role the backend reports, and the state of the stored
    /// credential.
    Whoami {
        /// Profile to reach the account through (default: the profile a new session runs on)
        #[arg(long, value_name = "NAME")]
        profile: Option<String>,
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
    /// Show the account's historical usage
    ///
    /// Lifetime tokens, streaks and per-day counts, as the backend reports them.
    Stats {
        /// Profile to reach the account through (default: the profile a new session runs on)
        #[arg(long, value_name = "NAME")]
        profile: Option<String>,
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
}

/// Profiles: an account plus the model and every model-tied setting. A session records the profile
/// it runs on; `use` is the only command that writes `default_profile`.
#[allow(
    clippy::large_enum_variant,
    reason = "`Add` holds several flags inline; built once per process, so boxing buys nothing"
)]
#[derive(clap::Subcommand, Debug)]
pub(crate) enum ProfileAction {
    /// Add a profile on an account
    ///
    /// Prompts for the account and model when not flagged, then offers an optional advanced step
    /// (thinking, context window, effort). Does not touch `default_profile`: a sole profile is
    /// the default, and `meka profile use` picks among several.
    Add {
        /// Profile name
        name: String,
        /// Account the profile bills
        #[arg(long, value_name = "NAME")]
        account: Option<String>,
        /// Model name
        #[arg(long)]
        model: Option<String>,
        /// Context window in tokens (default: 1000000)
        #[arg(long = "context-window", value_name = "TOKENS")]
        context_window: Option<u64>,
        /// Per-request output token cap; unset leaves the backend's default
        #[arg(long = "max-output-tokens", value_name = "TOKENS")]
        max_output_tokens: Option<u64>,
        /// Reasoning effort; unset leaves the provider's default
        #[arg(long, value_name = "EFFORT")]
        effort: Option<String>,
        /// Accept image input (default: true)
        #[arg(long, hide_possible_values = true, value_name = "BOOL")]
        vision: Option<bool>,
        /// Thinking mode: adaptive, budgeted, off (Anthropic Messages backends only; default:
        /// adaptive)
        #[arg(long, value_enum, hide_possible_values = true, value_name = "MODE")]
        thinking: Option<crate::config::ThinkingMode>,
        /// Token budget when thinking = budgeted (default: `[thinking].budget`, then 16000)
        #[arg(long = "thinking-budget", value_name = "TOKENS")]
        thinking_budget: Option<u64>,
        /// Largest request body in bytes before old images are redacted (Anthropic backends
        /// default to 30 MiB)
        #[arg(long = "max-request-bytes", value_name = "BYTES")]
        max_request_bytes: Option<u64>,
        /// Thinking display: updates, summarized, redacted (claude-subscription only; default:
        /// updates)
        #[arg(
            long = "thinking-display",
            value_enum,
            hide_possible_values = true,
            value_name = "DISPLAY"
        )]
        thinking_display: Option<crate::config::ThinkingDisplay>,
    },
    /// List configured profiles
    List {
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
    /// Change one setting on a profile
    ///
    /// Keys: model, context_window, max_output_tokens, effort, vision, thinking, thinking_budget,
    /// max_request_bytes, thinking_display. `account` is not settable; add a profile on the other
    /// account instead.
    Set {
        /// Profile name
        name: String,
        /// Setting to write
        key: String,
        /// New value; omit it and pass `--unset` to remove the setting
        value: Option<String>,
        /// Remove the setting, so the profile falls back to the default
        #[arg(long, conflicts_with = "value")]
        unset: bool,
    },
    /// Set the default profile
    Use {
        /// Profile name
        name: String,
    },
    /// Remove a profile
    Remove {
        /// Profile name
        name: String,
    },
    /// Rename a profile; its sessions and `default_profile` follow
    Rename {
        /// Profile name
        name: String,
        /// New name
        new_name: String,
    },
}

#[allow(
    clippy::large_enum_variant,
    reason = "`Add` holds several flags inline; built once per process, so boxing buys nothing"
)]
#[derive(clap::Subcommand, Debug)]
pub(crate) enum SkillAction {
    /// List installed skills
    List {
        /// Also show where each skill is on disk
        #[arg(long)]
        paths: bool,
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
    /// Print one skill's frontmatter and on-disk paths
    Get {
        /// Skill name
        name: String,
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
    /// Print the rendered skill body
    Show {
        /// Skill name
        name: String,
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
    /// Scaffold a new skill at `~/.config/meka/skills/<name>/SKILL.md`
    Add {
        /// Unique skill name (lowercase letters, digits, hyphens)
        name: String,

        /// One-line description for the system prompt
        #[arg(long)]
        description: Option<String>,

        /// Priority 0-9, lower first (default: 5)
        #[arg(long, value_parser = clap::value_parser!(u8).range(0..=9))]
        priority: Option<u8>,

        /// Frontmatter metadata (repeatable)
        #[arg(long, value_name = "KEY=VALUE")]
        metadata: Vec<String>,

        /// Copy this file instead of the template
        #[arg(long = "from-file", value_name = "PATH")]
        from_file: Option<std::path::PathBuf>,

        /// Overwrite the skill directory if it exists
        #[arg(long)]
        force: bool,

        /// Open the new SKILL.md afterwards in `$VISUAL`, then `$EDITOR`
        #[arg(long)]
        edit: bool,
    },
    /// Remove a skill's directory
    Remove {
        /// Skill name
        name: String,
    },
}

/// Inspect and cancel the wakeups the agent scheduled for itself through the `schedule_*` tools.
/// Read-and-cancel only: creating a job needs a session to attach it to, which is the agent's job.
#[derive(clap::Subcommand, Debug)]
pub(crate) enum ScheduleAction {
    /// List scheduled jobs
    List {
        /// One session's jobs, by id or prefix (default: all)
        #[arg(long)]
        session: Option<String>,
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
    /// Show a job's full details
    Show {
        /// Job id, or any unique prefix of one
        id: String,
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
    /// Cancel a job by id or unique prefix
    Cancel {
        /// Job id, or any unique prefix of one
        id: String,
    },
}

#[derive(clap::Subcommand, Debug)]
pub(crate) enum InstructionsAction {
    /// Print the resolved instructions and where they came from
    ///
    /// Resolution order is `MEKA_INSTRUCTIONS`, `MEKA_INSTRUCTIONS_FILE`, then `instructions.md`
    /// (or `instructions/`) in the config directory; `--instructions` belongs to a run and is not
    /// consulted. The text goes to stdout and the source to stderr.
    Show,
    /// Print the paths checked for instructions, and whether each exists
    Path,
}

/// Inspect and curate the agent's durable notes. The agent maintains these itself through the
/// `memory_*` tools; these subcommands are for reading, auditing, and pruning them by hand.
#[derive(clap::Subcommand, Debug)]
pub(crate) enum MemoryAction {
    /// List saved memories and the priority distribution
    List {
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
    /// Print one memory's stored fields
    Get {
        /// Memory name
        name: String,
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
    /// Print a memory's body
    Show {
        /// Memory name
        name: String,
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
    /// Write a memory by hand
    Add {
        /// Unique name (alphanumerics, `-`, `_` only)
        name: String,

        /// Fact shown in every session's memory index
        #[arg(long)]
        description: String,

        /// Priority 0-9, lower first (default: 5)
        #[arg(long, value_parser = clap::value_parser!(u8).range(0..=9))]
        priority: Option<u8>,

        /// Label for grouping and filtering (repeatable)
        #[arg(long = "tag", value_name = "TAG")]
        tags: Vec<String>,

        /// Detail loaded only on `memory_read`
        #[arg(long)]
        body: Option<String>,

        /// Read the body from this file, not `--body`
        #[arg(long = "from-file", value_name = "PATH")]
        from_file: Option<std::path::PathBuf>,

        /// Update an existing memory instead of refusing
        #[arg(long)]
        force: bool,
    },
    /// Open a memory's body in `$VISUAL`, then `$EDITOR`
    ///
    /// The body only; `meka memory add <name> --force` changes the description, priority or tags.
    Edit {
        /// Name of the memory to edit
        name: String,
    },
    /// Delete a memory permanently
    Remove {
        /// Memory name
        name: String,
    },
    /// Check the search index against the stored memories
    ///
    /// The index is derived from the memories, so rebuilding it cannot lose a note.
    Verify {
        /// Regenerate the index instead of only checking it
        #[arg(long)]
        rebuild: bool,
    },
    /// Write every memory out as Markdown, one file per memory
    Export {
        /// Directory to write into; must be new or empty (default: ./meka-memory-export)
        #[arg(long, value_name = "PATH")]
        dir: Option<std::path::PathBuf>,
    },
}

pub(crate) use crate::config::OutputFormat;

/// Output format for `meka session export`. One spelling, [`Self::name`], on `--format`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionExportFormat {
    /// Rendered Markdown (single session).
    Markdown,
    /// Structured JSON (round-trippable; includes sub-agent children).
    Json,
}

impl SessionExportFormat {
    /// Every format, in the order the names sort.
    pub(crate) const ALL: [SessionExportFormat; 2] = [Self::Json, Self::Markdown];

    /// The one spelling `--format` takes.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Markdown => "markdown",
            Self::Json => "json",
        }
    }

    /// The names, joined for a refusal that lists what would have been accepted.
    pub(crate) fn supported() -> String {
        Self::ALL
            .iter()
            .map(|format| format.name())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl std::fmt::Display for SessionExportFormat {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for SessionExportFormat {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|format| format.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not an export format. Supported: {}",
                    Self::supported()
                )
            })
    }
}

#[allow(
    clippy::large_enum_variant,
    reason = "`Add` holds every flag inline; built once per process, so boxing buys nothing"
)]
#[derive(clap::Subcommand, Debug)]
pub(crate) enum McpAction {
    /// List configured MCP servers
    List {
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
    /// Print the configuration for one server
    Get {
        /// Name of a server in config.toml
        name: String,
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
    /// Connect once and exit non-zero if the handshake fails
    Reconnect {
        /// Name of a server in config.toml
        name: String,
    },
    /// List a server's advertised tools with their resolved permissions
    Tools {
        /// Name of a server in config.toml
        name: String,
        /// Output format: plain or json
        #[arg(long, default_value = "plain")]
        format: OutputFormat,
    },
    /// Authenticate a server interactively
    ///
    /// With neither flag, runs the OAuth authorization-code flow. With one, stores the secret read
    /// from stdin and exits, which is also how an existing one is rotated.
    Login {
        /// Name of a server in config.toml
        name: String,

        /// Store a static bearer token read from stdin
        #[arg(long = "auth-token-stdin", conflicts_with = "client_secret_stdin")]
        auth_token_stdin: bool,

        /// Store an OAuth client secret read from stdin
        #[arg(long = "client-secret-stdin")]
        client_secret_stdin: bool,
    },
    /// Clear every stored credential for a server, revoking OAuth first
    Logout {
        /// Name of a server in config.toml
        name: String,
    },
    /// Add a server to config.toml
    Add {
        /// Unique server name (alphanumerics, `-`, `_` only)
        name: String,
        /// URL (HTTP) or executable path (stdio); transport auto-detected
        location: Option<String>,
        /// Arguments to pass to the stdio command
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,

        /// Force transport (stdio or http); auto-detected otherwise
        #[arg(long)]
        transport: Option<crate::config::McpTransport>,

        /// Environment variable for a stdio server (repeatable)
        #[arg(long = "env", value_name = "KEY=VALUE")]
        env: Vec<String>,

        /// HTTP header (repeatable)
        #[arg(long = "header", value_name = "KEY=VALUE")]
        header: Vec<String>,

        /// Authentication: oauth, client_credentials, client_credentials_jwt
        #[arg(long)]
        auth: Option<McpAuthKind>,

        /// Read a static bearer token from stdin (excludes `--auth`)
        ///
        /// Kept in the store, never in config.toml.
        #[arg(long = "auth-token-stdin", conflicts_with = "client_secret_stdin")]
        auth_token_stdin: bool,

        /// OAuth or client_credentials client id
        #[arg(long, value_name = "ID")]
        client_id: Option<String>,

        /// Read the OAuth client secret from stdin (excludes `--auth-token-stdin`)
        ///
        /// Kept in the store, never in config.toml.
        #[arg(long = "client-secret-stdin")]
        client_secret_stdin: bool,

        /// JWT signing key path (for client_credentials_jwt)
        #[arg(long, value_name = "KEY")]
        signing_key: Option<String>,

        /// JWT signing algorithm (RS256, RS384, RS512, ES256, ES384)
        #[arg(long, value_name = "ALGORITHM")]
        signing_algorithm: Option<String>,

        /// OAuth scope (repeatable)
        #[arg(long = "scope", value_name = "SCOPE")]
        scope: Vec<String>,

        /// Fixed OAuth redirect port (default: ephemeral)
        #[arg(long, value_name = "PORT")]
        redirect_port: Option<u16>,

        /// Permission: none, read, workspace, unrestricted (default: read)
        #[arg(long, value_name = "LEVEL")]
        permission: Option<String>,

        /// Raw tool name to allow (repeatable; restricts which register)
        #[arg(long = "allow-tool", value_name = "TOOL")]
        allow_tool: Vec<String>,

        /// Raw tool name to block (repeatable; applied after `--allow-tool`)
        #[arg(long = "disable-tool", value_name = "TOOL")]
        disable_tool: Vec<String>,

        /// Raw tool name to eager-load (repeatable; skips `load_tool`)
        #[arg(long = "eager-load-tool", value_name = "TOOL")]
        eager_load_tool: Vec<String>,

        /// Per-tool permission override (repeatable)
        #[arg(long = "tool-permission", value_name = "TOOL=LEVEL")]
        tool_permission: Vec<String>,

        /// Skip post-add auto-login; run `meka mcp login <name>` later
        #[arg(long = "no-login")]
        no_login: bool,

        /// Persist with `disabled = true`; re-enable via `meka mcp enable`
        #[arg(long = "disabled")]
        disabled: bool,

        /// Gate turns on this server: decline the turn if it is not connected
        #[arg(long = "required")]
        required: bool,
    },
    /// Remove a server from config.toml and clear every stored credential
    Remove {
        /// Name of a server in config.toml
        name: String,
    },
    /// Temporarily turn off a server without removing it from config
    Disable {
        /// Name of a server in config.toml
        name: String,
    },
    /// Turn a disabled server back on
    Enable {
        /// Name of a server in config.toml
        name: String,
    },
}

/// Authentication flavors selectable from the CLI. Maps onto the [`crate::config::McpAuthConfig`]
/// variants, except `None` which means "no `[auth]` block at all" (static token or
/// unauthenticated).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum McpAuthKind {
    OAuth,
    ClientCredentials,
    ClientCredentialsJwt,
}

impl McpAuthKind {
    pub(crate) const ALL: [Self; 3] = [
        Self::OAuth,
        Self::ClientCredentials,
        Self::ClientCredentialsJwt,
    ];

    /// The one spelling, shared with the `type` the `[auth]` block records.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::OAuth => "oauth",
            Self::ClientCredentials => "client_credentials",
            Self::ClientCredentialsJwt => "client_credentials_jwt",
        }
    }
}

impl std::fmt::Display for McpAuthKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for McpAuthKind {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|kind| kind.name() == value)
            .ok_or_else(|| {
                crate::text::unknown_name("auth", value, Self::ALL.iter().map(|kind| kind.name()))
            })
    }
}

#[derive(Parser, Debug)]
#[command(name = "meka", version, about = "A general-purpose AI agent harness")]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Option<Command>,

    /// Prompt for the first turn; `-` reads it from stdin
    #[arg(short = 'p', long = "prompt", value_name = "TEXT")]
    pub(crate) prompt: Option<String>,

    /// Continue the most recent session
    #[arg(short = 'c', long = "continue", conflicts_with = "resume")]
    pub(crate) continue_last: bool,

    /// Resume a session by id or any unique prefix
    #[arg(short = 'r', long = "resume", value_name = "SESSION")]
    pub(crate) resume: Option<String>,

    /// Initial permission level (none, read, workspace, unrestricted)
    #[arg(long = "permission", value_name = "LEVEL")]
    pub(crate) permission: Option<Permission>,

    /// Extra directory writable at `workspace` permission (repeatable)
    ///
    /// The working directory is always writable at that level, as are any folders an ACP client
    /// supplies. This adds to them.
    // A flag rather than a config key: which folders this run may write is per-run scope, like the
    // working directory itself, not a preference to persist.
    #[arg(long = "writable-root", value_name = "PATH")]
    pub(crate) writable_root: Vec<std::path::PathBuf>,

    /// Profile for this session; on a resume, repins it
    ///
    /// Selects a whole profile (account, model and every model-tied setting); change a field with
    /// `meka profile set`.
    #[arg(long = "profile", value_name = "NAME")]
    pub(crate) profile: Option<String>,

    /// Linux sandbox backend: landlock or bubblewrap
    #[arg(long = "sandbox-backend", value_name = "BACKEND")]
    pub(crate) sandbox_backend: Option<crate::config::SandboxBackend>,

    /// Disable streaming for this run (see `[display] stream`)
    #[arg(long = "no-stream")]
    pub(crate) no_stream: bool,

    /// Markdown render mode: termimad (default), syntect, or raw
    #[arg(long = "render-mode", value_name = "RENDERER")]
    pub(crate) render_mode: Option<crate::config::RenderMode>,

    /// Standing instructions for this run, replacing the discovered ones
    #[arg(long = "instructions", value_name = "STRING")]
    pub(crate) instructions: Option<String>,

    /// Invoke a user-invocable skill on the first turn
    #[arg(long = "skill", value_name = "NAME")]
    pub(crate) skill: Option<String>,

    /// Exit after the first turn finishes (requires `--prompt` or `--skill`)
    #[arg(long = "oneshot")]
    pub(crate) oneshot: bool,

    /// Output format: plain or json
    #[arg(long = "format", default_value = "plain", value_name = "FORMAT")]
    pub(crate) format: OutputFormat,

    /// Eager-load an MCP tool this session (repeatable)
    #[arg(long = "eager-load-tool", value_name = "SERVER:TOOL")]
    pub(crate) eager_load_tool: Vec<String>,

    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count)]
    pub(crate) verbosity: u8,
}

impl Cli {
    /// Replace `-p -` with what stdin holds, read to end of input.
    ///
    /// Done here rather than in clap, which cannot read a stream, and before `overrides()` so every
    /// consumer sees the words. Trailing newlines are dropped, since `echo hi | meka -p -` means
    /// `hi`; an empty read is refused, because a prompt of nothing is a turn spent on nothing.
    pub(crate) fn read_prompt_from_stdin_if_asked(&mut self) -> anyhow::Result<()> {
        if self.prompt.as_deref() != Some("-") {
            return Ok(());
        }
        let mut text = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut text)?;
        let text = text.trim_end_matches(['\n', '\r']).to_string();
        if text.trim().is_empty() {
            anyhow::bail!("`-p -` read nothing from stdin");
        }
        self.prompt = Some(text);
        Ok(())
    }

    /// The root flags as the value `config` resolves from. One place knows both names.
    pub(crate) fn overrides(&self) -> crate::config::CliOverrides {
        crate::config::CliOverrides {
            profile: self.profile.clone(),
            eager_load_tools: self.eager_load_tool.clone(),
            instructions: self.instructions.clone(),
            permission: self.permission,
            sandbox_backend: self.sandbox_backend,
            writable_roots: self.writable_root.clone(),
            no_stream: self.no_stream,
            render_mode: self.render_mode,
            resume: self.resume.clone(),
            continue_last: self.continue_last,
            prompt: self.prompt.clone(),
            oneshot: self.oneshot,
            output_format: self.format,
        }
    }
}
/// Read a secret from stdin when its `--…-stdin` flag was passed, and return `None` when it was
/// not.
///
/// stdin is read to end, so exactly one secret can be taken per command; the flags that reach here
/// conflict with each other in clap for that reason. An empty stream is an error rather than
/// `None`: a caller that asked for a token and got nothing should hear it here, not from the server
/// later.
pub(crate) fn read_secret_from_stdin(
    from_stdin: bool,
    label: &str,
) -> anyhow::Result<Option<String>> {
    if !from_stdin {
        return Ok(None);
    }
    use std::io::Read as _;
    let mut buffer = String::new();
    std::io::stdin().read_to_string(&mut buffer)?;
    let secret = buffer.trim().to_string();
    if secret.is_empty() {
        anyhow::bail!("no {label} was read from stdin");
    }
    Ok(Some(secret))
}

/// A `show` command's answer under `--format json`: one pretty-printed object, and nothing else on
/// stdout.
pub(crate) fn write_json(document: &impl serde::Serialize) -> std::io::Result<()> {
    crate::render::write_stdout_line(serde_json::to_string_pretty(document)?)
}

/// A `list` command's answer under `--format json`: `{"<nouns>": [...]}`, the envelope the HTTP
/// API's collection endpoints answer with, so one client type reads both. An empty listing is the
/// envelope around an empty array; the `No <nouns>.` note belongs to the plain rendering.
pub(crate) fn write_json_listing(
    nouns: &str,
    items: &impl serde::Serialize,
) -> std::io::Result<()> {
    let mut document = serde_json::Map::new();
    document.insert(nouns.to_string(), serde_json::to_value(items)?);
    write_json(&document)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_defaults() {
        let cli = Cli::parse_from(["meka"]);
        assert!(cli.command.is_none());
        assert!(cli.prompt.is_none());
        assert!(!cli.continue_last);
        assert!(cli.resume.is_none());
        assert!(cli.permission.is_none());
        assert!(cli.profile.is_none());
        assert_eq!(cli.format, OutputFormat::Plain);
        assert!(!cli.no_stream);
        assert!(cli.render_mode.is_none());
        assert!(cli.skill.is_none());
        assert!(!cli.oneshot);
        assert!(cli.eager_load_tool.is_empty());
        assert_eq!(cli.verbosity, 0);
    }

    #[test]
    fn cli_eager_load_tool_repeatable() {
        let cli = Cli::parse_from([
            "meka",
            "--eager-load-tool",
            "notion:search",
            "--eager-load-tool",
            "github:create_issue",
        ]);
        assert_eq!(cli.eager_load_tool, vec![
            "notion:search".to_string(),
            "github:create_issue".to_string()
        ]);
    }

    #[test]
    fn cli_oneshot_flag() {
        let cli = Cli::parse_from(["meka", "--oneshot", "-p", "do thing"]);
        assert!(cli.oneshot);
        assert_eq!(cli.prompt.as_deref(), Some("do thing"));
        let cli = Cli::parse_from(["meka", "--oneshot", "-p", "do thing", "--format", "json"]);
        assert_eq!(cli.format, OutputFormat::Json);
    }

    #[test]
    fn cli_prompt_flag() {
        let cli = Cli::parse_from(["meka", "--prompt", "hello world"]);
        assert_eq!(cli.prompt.as_deref(), Some("hello world"));
    }

    /// The bare positional is gone: a mistyped subcommand is an error, not a session opened with
    /// the typo as its first turn.
    #[test]
    fn an_unknown_subcommand_is_refused_rather_than_taken_as_a_prompt() {
        assert!(Cli::try_parse_from(["meka", "unknowncommand"]).is_err());
        assert!(Cli::try_parse_from(["meka", "hello world"]).is_err());
    }

    #[test]
    fn cli_skill_flag_alone() {
        let cli = Cli::parse_from(["meka", "--skill", "demo"]);
        assert_eq!(cli.skill.as_deref(), Some("demo"));
        assert!(cli.prompt.is_none());
    }

    #[test]
    fn cli_skill_flag_with_extra_prompt() {
        let cli = Cli::parse_from(["meka", "--skill", "demo", "-p", "extra context"]);
        assert_eq!(cli.skill.as_deref(), Some("demo"));
        assert_eq!(cli.prompt.as_deref(), Some("extra context"));
    }

    #[test]
    fn cli_continue_last() {
        let cli = Cli::parse_from(["meka", "-c"]);
        assert!(cli.continue_last);
        assert!(cli.resume.is_none());
    }

    #[test]
    fn cli_resume_specific_session() {
        let cli = Cli::parse_from(["meka", "-r", "550e8400-e29b-41d4-a716-446655440000"]);
        assert_eq!(
            cli.resume.as_deref(),
            Some("550e8400-e29b-41d4-a716-446655440000")
        );
        assert!(!cli.continue_last);
    }

    /// `-c` takes no value: it would otherwise be the only root flag that could swallow the next
    /// argument, and a prompt after it would be read as a session prefix.
    #[test]
    fn cli_continue_does_not_consume_the_prompt() {
        let cli = Cli::parse_from(["meka", "-c", "-p", "fix the bug"]);
        assert!(cli.continue_last);
        assert_eq!(cli.prompt.as_deref(), Some("fix the bug"));
    }

    #[test]
    fn cli_resume_takes_the_id_and_leaves_the_prompt() {
        let cli = Cli::parse_from(["meka", "-r", "550e8400", "-p", "fix the bug"]);
        assert_eq!(cli.resume.as_deref(), Some("550e8400"));
        assert_eq!(cli.prompt.as_deref(), Some("fix the bug"));
    }

    #[test]
    fn cli_continue_and_resume_are_mutually_exclusive() {
        assert!(Cli::try_parse_from(["meka", "-c", "-r", "550e8400"]).is_err());
    }

    #[test]
    fn cli_flags() {
        let cli = Cli::parse_from(["meka", "--profile", "work", "--no-stream", "-c", "-vv"]);
        assert_eq!(cli.profile.as_deref(), Some("work"));
        assert!(cli.no_stream);
        assert!(cli.continue_last);
        assert_eq!(cli.verbosity, 2);
    }

    /// `-p` is the prompt, and the profile has no short form: were the two one letter apart, a
    /// `-p work` meant to pick a profile would become a turn that said "work".
    #[test]
    fn cli_prompt_short_form_and_profile_long_form() {
        let cli = Cli::parse_from(["meka", "-p", "fix the bug", "--profile", "work"]);
        assert_eq!(cli.profile.as_deref(), Some("work"));
        assert_eq!(cli.prompt.as_deref(), Some("fix the bug"));
    }

    #[test]
    fn cli_permission_flag() {
        let cli = Cli::parse_from(["meka", "--permission", "workspace"]);
        assert_eq!(cli.permission, Some(Permission::Workspace));
        let cli = Cli::parse_from(["meka", "--permission", "unrestricted"]);
        assert_eq!(cli.permission, Some(Permission::Unrestricted));
        // One spelling per level: the prompt's indicator character is display, not a name the
        // flag takes.
        assert!(Cli::try_parse_from(["meka", "--permission", "u"]).is_err());
    }

    /// One spelling per value on every flag that takes an enum: an `md` alias and a case variant
    /// are refused, and the refusal lists what would have been accepted.
    #[test]
    fn every_enum_flag_takes_one_spelling_and_refuses_the_rest() {
        let cli = Cli::parse_from(["meka", "session", "export", "x", "--format", "json"]);
        assert!(matches!(
            cli.command,
            Some(Command::Session {
                action: SessionAction::Export {
                    format: SessionExportFormat::Json,
                    ..
                }
            })
        ));
        let refused: [&[&str]; 7] = [
            &["meka", "session", "export", "x", "--format", "md"],
            &["meka", "session", "export", "x", "--format", "JSON"],
            &["meka", "--oneshot", "-p", "x", "--format", "Json"],
            &["meka", "--render-mode", "RAW", "-p", "x"],
            &["meka", "--sandbox-backend", "Landlock", "-p", "x"],
            &["meka", "--permission", "Read", "-p", "x"],
            &["meka", "mcp", "add", "s", "http://x", "--transport", "HTTP"],
        ];
        for arguments in refused {
            let error = Cli::try_parse_from(arguments)
                .expect_err("a second spelling must not parse")
                .to_string();
            assert!(error.contains("Supported:"), "{arguments:?}: {error}");
        }
        let cli = Cli::parse_from(["meka", "mcp", "add", "s", "x", "--transport", "stdio"]);
        assert!(matches!(
            cli.command,
            Some(Command::Mcp {
                action: McpAction::Add {
                    transport: Some(crate::config::McpTransport::Stdio),
                    ..
                }
            })
        ));
    }

    /// A level meka does not have must fail *loudly* at the CLI rather than resolve to anything.
    ///
    /// This is the surface where an invocation is most likely to be automated, and a hard exit is
    /// the good outcome there: a nonzero status stops a script, where quietly picking a level would
    /// let it run on with authority nobody chose.
    #[test]
    fn the_permission_flag_refuses_a_level_meka_does_not_have() {
        let error = Cli::try_parse_from(["meka", "--permission", "elevated"])
            .expect_err("an unknown level must not parse");
        let rendered = error.to_string();
        for level in ["none", "read", "workspace", "unrestricted"] {
            assert!(
                rendered.contains(level),
                "clap must list {level}: {rendered}"
            );
        }
    }

    #[test]
    fn cli_continue_long_form() {
        let cli = Cli::parse_from(["meka", "--continue"]);
        assert!(cli.continue_last);
    }

    #[test]
    fn cli_resume_long_form() {
        let cli = Cli::parse_from(["meka", "--resume", "550e8400-e29b-41d4-a716-446655440000"]);
        assert_eq!(
            cli.resume.as_deref(),
            Some("550e8400-e29b-41d4-a716-446655440000")
        );
    }

    #[test]
    fn cli_account_add_subcommand() {
        let cli = Cli::parse_from([
            "meka",
            "account",
            "add",
            "work",
            "--backend",
            "claude-subscription",
        ]);
        match cli.command {
            Some(Command::Account {
                action: AccountAction::Add { name, backend, .. },
            }) => {
                assert_eq!(name, "work");
                assert_eq!(backend.as_deref(), Some("claude-subscription"));
            }
            other => panic!("expected account add, got {other:?}"),
        }
    }

    #[test]
    fn cli_profile_add_subcommand() {
        let cli = Cli::parse_from([
            "meka",
            "profile",
            "add",
            "daily",
            "--account",
            "work",
            "--model",
            "claude-opus-5",
        ]);
        match cli.command {
            Some(Command::Profile {
                action:
                    ProfileAction::Add {
                        name,
                        account,
                        model,
                        ..
                    },
            }) => {
                assert_eq!(name, "daily");
                assert_eq!(account.as_deref(), Some("work"));
                assert_eq!(model.as_deref(), Some("claude-opus-5"));
            }
            other => panic!("expected profile add, got {other:?}"),
        }
    }

    #[test]
    fn cli_session_list_subcommand() {
        let cli = Cli::parse_from(["meka", "session", "list"]);
        match cli.command {
            Some(Command::Session {
                action:
                    SessionAction::List {
                        limit,
                        include_children,
                        format,
                    },
            }) => {
                assert_eq!(limit, 20);
                assert!(!include_children);
                assert_eq!(format, OutputFormat::Plain);
            }
            other => panic!("expected session list, got {other:?}"),
        }
    }

    /// Every listing and `show` command takes `--format` with the one vocabulary `OutputFormat`
    /// has, defaulting to plain, and refuses a spelling it does not have.
    #[test]
    fn every_listing_and_show_command_takes_the_same_format_flag() {
        let commands: [&[&str]; 17] = [
            &["session", "list"],
            &["session", "show", "0e5f"],
            &["account", "list"],
            &["profile", "list"],
            &["mcp", "list"],
            &["mcp", "get", "s"],
            &["mcp", "tools", "s"],
            &["schedule", "list"],
            &["schedule", "show", "7f3a"],
            &["memory", "list"],
            &["memory", "get", "m"],
            &["memory", "show", "m"],
            &["tools", "list"],
            &["history", "list"],
            &["skill", "list"],
            &["skill", "get", "s"],
            &["skill", "show", "s"],
        ];
        for command in commands {
            let mut arguments = vec!["meka"];
            arguments.extend_from_slice(command);
            let plain = Cli::try_parse_from(&arguments)
                .unwrap_or_else(|error| panic!("{command:?} must parse without the flag: {error}"));
            arguments.extend_from_slice(&["--format", "json"]);
            let json = Cli::try_parse_from(&arguments)
                .unwrap_or_else(|error| panic!("{command:?} must take --format json: {error}"));
            // The flag is inside each variant, so its value is read back through `Debug`: one
            // assertion shape for seventeen variants.
            assert!(
                format!("{:?}", plain.command).contains("format: Plain"),
                "{command:?} must default to plain: {:?}",
                plain.command
            );
            assert!(
                format!("{:?}", json.command).contains("format: Json"),
                "{command:?} must record json: {:?}",
                json.command
            );
            arguments.pop();
            arguments.push("JSON");
            let refused = Cli::try_parse_from(&arguments)
                .expect_err("a second spelling must not parse")
                .to_string();
            assert!(refused.contains("Supported:"), "{command:?}: {refused}");
        }
    }

    #[test]
    fn cli_session_delete_all_subcommand() {
        let cli = Cli::parse_from(["meka", "session", "delete", "--all"]);
        match cli.command {
            Some(Command::Session {
                action:
                    SessionAction::Delete {
                        session_ids, all, ..
                    },
            }) => {
                assert!(session_ids.is_empty());
                assert!(all);
            }
            other => panic!("expected session delete, got {other:?}"),
        }
    }

    /// Naming sessions and then asking for every session are two different requests.
    ///
    /// Taking both and quietly honoring the wider one would let `meka session delete "$ID" --all`
    /// with `$ID` unset delete the store and report the empty id as a failure: a complete success
    /// reported as an error, over work nobody asked for. Same reasoning as the `--older-than-days`
    /// conflicts beside it.
    #[test]
    fn cli_session_delete_all_conflicts_with_explicit_ids() {
        let id = "550e8400-e29b-41d4-a716-446655440000";
        assert!(
            Cli::try_parse_from(["meka", "session", "delete", id, "--all"]).is_err(),
            "naming an id and asking for --all must be refused"
        );
    }

    /// The manual counterpart to `[session].retention`, so it has to actually parse.
    #[test]
    fn cli_session_delete_older_than_days() {
        let cli = Cli::parse_from(["meka", "session", "delete", "--older-than-days", "90"]);
        match cli.command {
            Some(Command::Session {
                action:
                    SessionAction::Delete {
                        session_ids,
                        all,
                        older_than_days,
                    },
            }) => {
                assert!(session_ids.is_empty());
                assert!(!all);
                assert_eq!(older_than_days, Some(90));
            }
            other => panic!("expected session delete, got {other:?}"),
        }
    }

    /// Both other selectors must be refused alongside it. `--all` because the two windows
    /// disagree, and explicit ids because a listed session younger than the window would be
    /// silently spared while the user watched a different count come back.
    #[test]
    fn cli_session_delete_older_than_days_conflicts() {
        let id = "550e8400-e29b-41d4-a716-446655440000";
        for arguments in [
            vec![
                "meka",
                "session",
                "delete",
                "--older-than-days",
                "90",
                "--all",
            ],
            vec!["meka", "session", "delete", "--older-than-days", "90", id],
        ] {
            assert!(
                Cli::try_parse_from(&arguments).is_err(),
                "{arguments:?} must be refused"
            );
        }
    }

    #[test]
    fn cli_session_export_stdout_subcommand() {
        let id = "550e8400-e29b-41d4-a716-446655440000";
        let cli = Cli::parse_from(["meka", "session", "export", id, "-o", "-"]);
        match cli.command {
            Some(Command::Session {
                action: SessionAction::Export { output, .. },
            }) => assert_eq!(output.as_deref(), Some("-")),
            other => panic!("expected session export, got {other:?}"),
        }
    }

    #[test]
    fn cli_session_fork_subcommand() {
        let id = "550e8400-e29b-41d4-a716-446655440000";
        let cli = Cli::parse_from(["meka", "session", "fork", id]);
        match cli.command {
            Some(Command::Session {
                action: SessionAction::Fork { session_id },
            }) => assert_eq!(session_id, id),
            other => panic!("expected session fork, got {other:?}"),
        }
    }

    #[test]
    fn cli_history_list_subcommand() {
        let cli = Cli::parse_from(["meka", "history", "list", "-n", "10"]);
        match cli.command {
            Some(Command::History {
                action: HistoryAction::List { limit, .. },
            }) => assert_eq!(limit, 10),
            other => panic!("expected history list, got {other:?}"),
        }
    }

    #[test]
    fn cli_history_clear_subcommand() {
        let cli = Cli::parse_from(["meka", "history", "clear"]);
        assert!(matches!(
            cli.command,
            Some(Command::History {
                action: HistoryAction::Clear
            })
        ));
    }
}
