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
pub(crate) mod project;
pub(crate) mod schedule;
pub(crate) mod session;
pub(crate) mod skills;
pub(crate) mod tool;

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
    /// Manage projects
    Project {
        #[command(subcommand)]
        action: ProjectAction,
    },
    /// Manage stored sessions
    Session {
        #[command(subcommand)]
        action: SessionAction,
    },
    /// View or clear screen input history
    History {
        #[command(subcommand)]
        action: HistoryAction,
    },
    /// Manage MCP servers
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },
    /// Inspect built-in tools
    Tool {
        #[command(subcommand)]
        action: ToolAction,
    },
    /// Manage user skills
    Skill {
        #[command(subcommand)]
        action: SkillAction,
    },
    /// Manage the memories the agent keeps
    Memory {
        #[command(subcommand)]
        action: MemoryAction,
    },
    /// Show the standing instructions
    Instructions {
        #[command(subcommand)]
        action: InstructionsAction,
    },
    /// Inspect and cancel scheduled jobs
    Schedule {
        #[command(subcommand)]
        action: ScheduleAction,
    },
    /// Run as an ACP agent over stdio
    ///
    /// Speaks newline-framed JSON-RPC on stdin and stdout; diagnostics go to stderr.
    Acp,
    /// Run as an HTTP service
    ///
    /// Exposes the agent over HTTP and JSON; auth, session GC and SSE streaming are configured
    /// under `[serve]` in config.toml.
    Serve {
        /// Address to bind
        #[arg(long, value_name = "ADDR")]
        bind: Option<String>,
    },
    /// Run a command inside the sandbox's Landlock layer (internal)
    ///
    /// Spawned by `shell_execute` inside Bubblewrap, which has no Landlock support of its own; not
    /// for direct use.
    #[cfg(target_os = "linux")]
    #[command(hide = true)]
    Confine {
        /// Directory the command may write beneath (repeatable)
        #[arg(long = "writable", value_name = "PATH")]
        writable: Vec<std::path::PathBuf>,
        /// Directory the command may read beneath (repeatable); given, reads elsewhere are refused
        #[arg(long = "readable", value_name = "PATH")]
        readable: Vec<std::path::PathBuf>,
        /// Throwaway space the command may fill, without reaching a socket in it (repeatable)
        #[arg(long = "scratch", value_name = "PATH")]
        scratch: Vec<std::path::PathBuf>,
        /// The command to become, after `--`
        #[arg(last = true, required = true, value_name = "COMMAND")]
        command: Vec<std::ffi::OsString>,
    },
}

#[derive(clap::Subcommand, Debug)]
pub(crate) enum ToolAction {
    /// List built-in tools with their permissions
    List {
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Show a built-in tool's schema
    Show {
        /// Tool name
        name: String,
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
}

#[derive(clap::Subcommand, Debug)]
pub(crate) enum SessionAction {
    /// List past sessions
    List {
        /// Maximum number of sessions to show (0 = all)
        #[arg(short = 'n', long, default_value = "20")]
        limit: u32,
        /// Include sub-agent sessions
        ///
        /// Hidden by default, so the view stays on conversations you started.
        #[arg(long)]
        include_children: bool,
        /// Only sessions on this project
        #[arg(long, value_name = "NAME")]
        project: Option<String>,
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Export a session
    Export {
        /// Session id or unique prefix
        session_id: String,
        /// Output file (`-` for stdout)
        ///
        /// Defaults to `session-<id>.md` for markdown or `session-<id>.json` for json, written to
        /// the current directory.
        #[arg(short, long)]
        output: Option<String>,
        #[arg(
            long,
            default_value = "markdown",
            help = session_export_format_help(),
            long_help = format!(
                "{}\n\n`json` is structured and round-trippable via `meka session import`, and \
                 includes any sub-agent child sessions. `markdown` is rendered and covers the \
                 single session only.",
                session_export_format_help()
            )
        )]
        format: SessionExportFormat,
    },
    /// Remove sessions
    Remove {
        /// Session ids or unique prefixes
        session_ids: Vec<String>,
        /// Remove all sessions
        // Conflicts with explicit ids: naming some sessions and then asking for every session is
        // two different requests, and running the wider one makes the narrower one look honored.
        #[arg(long, conflicts_with_all = ["older_than_days", "session_ids"])]
        all: bool,
        /// Remove sessions not updated in this many days
        // Conflicts with explicit ids rather than ignoring them: a listed session younger than the
        // window would otherwise be silently spared.
        #[arg(
            long = "older-than-days",
            value_name = "DAYS",
            conflicts_with = "session_ids"
        )]
        older_than_days: Option<u64>,
    },
    /// Import a session
    ///
    /// Recreates the session and any sub-agent children under fresh ids, and prints the new root
    /// session id.
    Import {
        /// Export file to read (`-` for stdin)
        input: String,
    },
    /// Fork a session
    ///
    /// The copy carries the full conversation; the original is untouched. Prints the new session
    /// id.
    Fork {
        /// Session id or unique prefix
        session_id: String,
    },
    /// Show a session
    Show {
        /// Session id or unique prefix
        session_id: String,
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Drop the most recent turns from a session
    ///
    /// Cuts at a user boundary, so no tool call is separated from its result, and `meka session
    /// export` still shows the dropped turns. Recovers a session the provider rejects.
    Rewind {
        /// Session id or unique prefix
        session_id: String,
        /// Number of turns to drop
        #[arg(short = 'n', long, default_value = "1")]
        turns: usize,
    },
    /// Set a session's title or pin it
    ///
    /// `title` takes the text, or an empty string to label the session by its first words again;
    /// `pinned` takes `true` or `false`. Neither moves the session in the listing's recency order.
    Set {
        /// Session id or unique prefix
        session_id: String,
        #[arg(help = listed("Field", SessionField::ALL.iter().map(|field| field.name()), ""))]
        field: SessionField,
        /// New value
        value: String,
    },
    /// Search sessions
    ///
    /// Searches what was said, not what tools were called with or returned, and lists a session
    /// whose title holds every word first.
    Search {
        /// Words to search for
        query: String,
        /// Maximum number of sessions to show (0 = all)
        #[arg(short = 'n', long, default_value = "20")]
        limit: u32,
        /// Include sub-agent sessions
        #[arg(long)]
        include_children: bool,
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
}

/// The field `meka session set` writes. One spelling, [`Self::name`], on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionField {
    /// The label every listing shows for the session.
    Title,
    /// Whether the session is listed first, ahead of every unpinned one.
    Pinned,
}

impl SessionField {
    /// Every field, in the order the names sort.
    pub(crate) const ALL: [SessionField; 2] = [Self::Pinned, Self::Title];

    /// The one spelling the command line takes.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Title => "title",
            Self::Pinned => "pinned",
        }
    }
}

impl std::fmt::Display for SessionField {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for SessionField {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|field| field.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not a session field. Supported: {}",
                    Self::ALL
                        .iter()
                        .map(|field| field.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
    }
}

#[derive(clap::Subcommand, Debug)]
pub(crate) enum HistoryAction {
    /// List input history
    List {
        /// Maximum number of entries to show (0 = all)
        #[arg(short = 'n', long, default_value = "50")]
        limit: u32,
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Clear input history
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
    /// Add an account
    ///
    /// Prompts for the backend and base URL when not flagged, then acquires the secret (an OAuth
    /// login for the subscription backends, an API-key prompt for the rest) and saves it to the
    /// store.
    Add {
        /// Account name
        name: String,
        #[arg(
            long,
            value_name = "BACKEND",
            help = listed("Backend", crate::config::Backend::ALL.iter().map(|backend| backend.name()), "")
        )]
        backend: Option<String>,
        /// API base URL
        #[arg(long = "base-url", value_name = "URL")]
        base_url: Option<String>,
        /// OAuth token endpoint
        #[arg(long = "oauth-token-url", value_name = "URL")]
        oauth_token_url: Option<String>,
        /// OAuth client id
        #[arg(long = "client-id", value_name = "ID")]
        client_id: Option<String>,
        /// Send the interleaved-thinking beta (default: true)
        #[arg(
            long = "interleaved-thinking",
            hide_possible_values = true,
            value_name = "BOOL"
        )]
        interleaved_thinking: Option<bool>,
        /// Read the API key from stdin
        #[arg(long = "api-key-stdin")]
        api_key_stdin: bool,
    },
    /// List configured accounts
    List {
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Re-authenticate an account
    Login {
        /// Account name
        name: String,
        /// Read the API key from stdin
        #[arg(long = "api-key-stdin")]
        api_key_stdin: bool,
    },
    /// Remove an account
    ///
    /// Refused while any profile names the account.
    Remove {
        /// Account name
        name: String,
    },
    /// Rename an account
    Rename {
        /// Current name
        name: String,
        /// New name
        new_name: String,
    },
    /// Show the account's rate-limit usage
    ///
    /// The session and weekly windows, with their reset times.
    Usage {
        /// Account name (default: the default profile's account)
        name: Option<String>,
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Show who the account is logged in as
    ///
    /// The plan, tier, organization and role the backend reports, and the state of the stored
    /// credential.
    Whoami {
        /// Account name (default: the default profile's account)
        name: Option<String>,
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Show the account's historical usage
    ///
    /// Lifetime tokens, streaks and per-day counts, as the backend reports them.
    Stats {
        /// Account name (default: the default profile's account)
        name: Option<String>,
        #[arg(long, default_value = "plain", help = output_format_help())]
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
    /// Add a profile
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
        /// Fallback profile (repeatable, tried in order)
        #[arg(long = "fallback", value_name = "NAME")]
        fallback: Vec<String>,
        /// Context window in tokens
        #[arg(long = "context-window", value_name = "TOKENS")]
        context_window: Option<u64>,
        /// Per-request output token cap
        #[arg(long = "max-output-tokens", value_name = "TOKENS")]
        max_output_tokens: Option<u64>,
        /// Request body size in bytes above which old images are redacted
        #[arg(long = "max-request-bytes", value_name = "BYTES")]
        max_request_bytes: Option<u64>,
        /// Reasoning effort
        #[arg(long, value_name = "EFFORT")]
        effort: Option<String>,
        /// Accept image input (default: true)
        #[arg(long, hide_possible_values = true, value_name = "BOOL")]
        vision: Option<bool>,
        #[arg(
            long = "deferred-tools",
            value_name = "MODE",
            help = listed(
                "How a request carries deferred tools",
                crate::config::DeferredTools::ALL.iter().map(|setting| setting.name()),
                ""
            )
        )]
        deferred_tools: Option<crate::config::DeferredTools>,
        #[arg(
            long,
            value_enum,
            hide_possible_values = true,
            value_name = "MODE",
            help = listed(
                "Thinking mode",
                crate::config::ThinkingMode::ALL.iter().map(|mode| mode.name()),
                " (default: adaptive)"
            )
        )]
        thinking: Option<crate::config::ThinkingMode>,
        /// Token budget for budgeted thinking
        #[arg(long = "thinking-budget", value_name = "TOKENS")]
        thinking_budget: Option<u64>,
        /// Send the moving cache breakpoint (default: true)
        #[arg(
            long = "cache-control",
            hide_possible_values = true,
            value_name = "BOOL"
        )]
        cache_control: Option<bool>,
        #[arg(
            long = "thinking-display",
            value_enum,
            hide_possible_values = true,
            value_name = "DISPLAY",
            help = listed(
                "Thinking display",
                crate::config::ThinkingDisplay::ALL.iter().map(|display| display.name()),
                " (default: summarized)"
            )
        )]
        thinking_display: Option<crate::config::ThinkingDisplay>,
    },
    /// List configured profiles
    List {
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Set one of a profile's settings
    #[command(long_about = profile_set_about())]
    Set {
        /// Profile name
        name: String,
        /// Setting to write
        key: String,
        /// New value
        value: Option<String>,
        /// Remove the setting
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
    /// Rename a profile
    Rename {
        /// Current name
        name: String,
        /// New name
        new_name: String,
    },
}

/// Projects: the names a session is on, each a memory tier of its own beside the global one. A
/// project is a row in the store; a session is put on one with `--project`, `/project` or the
/// session's `project` field.
#[derive(clap::Subcommand, Debug)]
pub(crate) enum ProjectAction {
    /// Add a project
    Add {
        /// Project name (alphanumerics, `-`, `_` only)
        name: String,
    },
    /// List projects
    List {
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Remove a project
    Remove {
        /// Project name
        name: String,
    },
    /// Rename a project
    Rename {
        /// Current name
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
        /// Show each skill's path
        #[arg(long)]
        paths: bool,
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Show a skill's frontmatter
    Get {
        /// Skill name
        name: String,
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Show a skill's body
    Show {
        /// Skill name
        name: String,
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Add a skill
    Add {
        /// Skill name (lowercase letters, digits, hyphens)
        name: String,

        /// One-line description
        #[arg(long)]
        description: Option<String>,

        /// Priority 0-9, lower first (default: 5)
        #[arg(long, value_parser = clap::value_parser!(u8).range(0..=9))]
        priority: Option<u8>,

        /// Frontmatter metadata (repeatable)
        #[arg(long, value_name = "KEY=VALUE")]
        metadata: Vec<String>,

        /// Read the body from this file
        #[arg(long = "from-file", value_name = "PATH")]
        from_file: Option<std::path::PathBuf>,

        /// Overwrite an existing skill
        #[arg(long)]
        force: bool,

        /// Open the new skill in an editor
        #[arg(long)]
        edit: bool,
    },
    /// Remove a skill
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
        /// Only one session's jobs
        #[arg(long)]
        session: Option<String>,
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Show a job
    Show {
        /// Job id or unique prefix
        id: String,
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Cancel a job
    Cancel {
        /// Job id or unique prefix
        id: String,
    },
}

#[derive(clap::Subcommand, Debug)]
pub(crate) enum InstructionsAction {
    /// Show the resolved instructions
    ///
    /// Resolution order is `MEKA_INSTRUCTIONS`, `MEKA_INSTRUCTIONS_FILE`, then `instructions.md`
    /// (or `instructions/`) in the config directory; `--instructions` belongs to a run and is not
    /// consulted. The files `[instructions].files` names are read against the current directory
    /// and follow. The text goes to stdout and the sources to stderr.
    Show,
    /// Show where standing instructions are looked for
    ///
    /// The files `[instructions].files` names are listed after the config directory's, resolved
    /// against the current directory.
    Path,
}

/// Inspect and curate the agent's durable notes. The agent maintains these itself through the
/// `memory_*` tools; these subcommands are for reading, auditing, and pruning them by hand.
#[derive(clap::Subcommand, Debug)]
pub(crate) enum MemoryAction {
    /// List saved memories
    List {
        /// Only this project's memories
        #[arg(long, value_name = "NAME")]
        project: Option<String>,
        /// Only memories referencing this target
        #[arg(long, value_name = "REFERENCE")]
        referencing: Option<String>,
        /// Only memories not verified within this long
        #[arg(long, value_name = "DURATION")]
        unverified: Option<String>,
        /// Only expired memories
        #[arg(long)]
        expired: bool,
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Show a memory's fields
    Get {
        /// Memory name
        name: String,
        /// Project the memory is in
        #[arg(long, value_name = "NAME")]
        project: Option<String>,
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Show a memory's body
    Show {
        /// Memory name
        name: String,
        /// Project the memory is in
        #[arg(long, value_name = "NAME")]
        project: Option<String>,
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Add a memory
    Add {
        /// Memory name (alphanumerics, `-`, `_` only)
        name: String,
        /// One-line description
        #[arg(long)]
        description: String,
        /// Body text
        #[arg(long)]
        body: Option<String>,
        /// Read the body from this file
        #[arg(long = "from-file", value_name = "PATH")]
        from_file: Option<std::path::PathBuf>,
        /// Project to file it in
        #[arg(long, value_name = "NAME")]
        project: Option<String>,
        /// Priority 0-9, lower first (default: 5)
        #[arg(long, value_parser = clap::value_parser!(u8).range(0..=9))]
        priority: Option<u8>,
        #[arg(
            long,
            value_name = "LOAD",
            help = listed(
                "Load policy",
                crate::memory::Load::ALL.iter().map(|load| load.name()),
                " (default: index)"
            )
        )]
        load: Option<crate::memory::Load>,
        /// Record the fact as verified now
        #[arg(long)]
        verified: bool,
        /// Expiry: RFC 3339 or YYYY-MM-DD
        #[arg(long, value_name = "WHEN")]
        expires: Option<String>,
        /// Where the fact came from
        #[arg(long)]
        source: Option<String>,
        /// What it depends on (repeatable)
        #[arg(long = "reference", value_name = "REFERENCE")]
        references: Vec<String>,
        /// Label for grouping and filtering (repeatable)
        #[arg(long = "tag", value_name = "TAG")]
        tags: Vec<String>,
        /// Update an existing memory
        #[arg(long)]
        force: bool,
    },
    /// Edit a memory's body in an editor
    ///
    /// The body only; `meka memory add <name> --force` changes the description, priority or tags.
    Edit {
        /// Memory name
        name: String,
        /// Project the memory is in
        #[arg(long, value_name = "NAME")]
        project: Option<String>,
    },
    /// Remove a memory
    Remove {
        /// Memory name
        name: String,
        /// Project the memory is in
        #[arg(long, value_name = "NAME")]
        project: Option<String>,
    },
    /// Check the search index against the stored memories
    ///
    /// The index is derived from the memories, so rebuilding it cannot lose a note.
    Verify {
        /// Rebuild the index
        #[arg(long)]
        rebuild: bool,
    },
    /// Export memories as Markdown files
    Export {
        /// Directory to write into (default: ./meka-memory-export)
        #[arg(long, value_name = "DIR")]
        directory: Option<std::path::PathBuf>,
        /// Only this project's memories
        #[arg(long, value_name = "NAME")]
        project: Option<String>,
    },
    /// Import an export directory
    Import {
        /// Directory `meka memory export` wrote
        #[arg(value_name = "DIR")]
        directory: std::path::PathBuf,
        /// Overwrite a memory that exists
        #[arg(long)]
        force: bool,
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
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Show a server's configuration
    Get {
        /// Server name
        name: String,
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Check that a server connects
    Reconnect {
        /// Server name
        name: String,
    },
    /// List a server's tools
    Tools {
        /// Server name
        name: String,
        #[arg(long, default_value = "plain", help = output_format_help())]
        format: OutputFormat,
    },
    /// Log in to a server
    ///
    /// With neither flag, runs the OAuth authorization-code flow. With one, stores the secret read
    /// from stdin and exits, which is also how an existing one is rotated.
    Login {
        /// Server name
        name: String,

        /// Read a bearer token from stdin
        #[arg(long = "auth-token-stdin", conflicts_with = "client_secret_stdin")]
        auth_token_stdin: bool,

        /// Read the OAuth client secret from stdin
        #[arg(long = "client-secret-stdin")]
        client_secret_stdin: bool,
    },
    /// Clear a server's stored credentials
    Logout {
        /// Server name
        name: String,
    },
    /// Add a server
    Add {
        /// Server name (alphanumerics, `-`, `_` only)
        name: String,
        /// URL or executable path
        location: Option<String>,
        /// Arguments for a stdio command
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,

        #[arg(
            long,
            help = listed(
                "Transport",
                crate::config::McpTransport::ALL.iter().map(|transport| transport.name()),
                ""
            )
        )]
        transport: Option<crate::config::McpTransport>,

        /// Environment variable for a stdio server (repeatable)
        #[arg(long = "env", value_name = "KEY=VALUE")]
        env: Vec<String>,

        /// HTTP header (repeatable)
        #[arg(long = "header", value_name = "KEY=VALUE")]
        header: Vec<String>,

        #[arg(
            long,
            help = listed("Authentication", McpAuthKind::ALL.iter().map(|kind| kind.name()), "")
        )]
        auth: Option<McpAuthKind>,

        /// Read a bearer token from stdin
        ///
        /// Kept in the store, never in config.toml.
        #[arg(long = "auth-token-stdin", conflicts_with = "client_secret_stdin")]
        auth_token_stdin: bool,

        /// OAuth client id
        #[arg(long, value_name = "ID")]
        client_id: Option<String>,

        /// Read the OAuth client secret from stdin
        ///
        /// Kept in the store, never in config.toml.
        #[arg(long = "client-secret-stdin")]
        client_secret_stdin: bool,

        /// JWT signing key path
        #[arg(long, value_name = "KEY")]
        signing_key: Option<String>,

        /// JWT signing algorithm: RS256, RS384, RS512, ES256, ES384
        #[arg(long, value_name = "ALGORITHM")]
        signing_algorithm: Option<String>,

        /// OAuth scope (repeatable)
        #[arg(long = "scope", value_name = "SCOPE")]
        scope: Vec<String>,

        /// OAuth redirect port
        #[arg(long, value_name = "PORT")]
        redirect_port: Option<u16>,

        #[arg(
            long,
            value_name = "LEVEL",
            help = listed(
                "Permission",
                Permission::ALL.iter().map(|level| level.name()),
                ""
            )
        )]
        permission: Option<String>,

        /// Tool to allow (repeatable)
        #[arg(long = "allow-tool", value_name = "TOOL")]
        allow_tool: Vec<String>,

        /// Tool to disable (repeatable)
        #[arg(long = "disable-tool", value_name = "TOOL")]
        disable_tool: Vec<String>,

        /// Tool to eager-load (repeatable)
        #[arg(long = "eager-load-tool", value_name = "TOOL")]
        eager_load_tool: Vec<String>,

        #[arg(
            long = "deferral",
            value_name = "MODE",
            help = listed(
                "Tool loading",
                crate::config::Deferral::ALL.iter().map(|deferral| deferral.name()),
                ""
            )
        )]
        deferral: Option<crate::config::Deferral>,

        /// Per-tool permission override (repeatable)
        #[arg(long = "tool-permission", value_name = "TOOL=LEVEL")]
        tool_permission: Vec<String>,

        /// Skip the login after adding
        #[arg(long = "no-login")]
        no_login: bool,

        /// Add the server disabled
        #[arg(long = "disabled")]
        disabled: bool,

        /// Decline turns while the server is not connected
        #[arg(long = "required")]
        required: bool,
    },
    /// Remove a server and its credentials
    Remove {
        /// Server name
        name: String,
    },
    /// Stop connecting to a server
    Disable {
        /// Server name
        name: String,
    },
    /// Connect to a disabled server again
    Enable {
        /// Server name
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
                format!(
                    "'{value}' is not an auth kind. Supported: {}",
                    Self::ALL
                        .iter()
                        .map(|kind| kind.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
    }
}

/// The `--format` summary every listing shares, from the table the parser reads.
fn output_format_help() -> String {
    listed(
        "Output format",
        OutputFormat::ALL.iter().map(|format| format.name()),
        "",
    )
}

/// The `session export` format summary, from the table the parser reads.
fn session_export_format_help() -> String {
    listed(
        "Output format",
        SessionExportFormat::ALL.iter().map(|format| format.name()),
        "",
    )
}

/// A help line listing a vocabulary from the table that defines it, so `-h` cannot fall behind a
/// value added to an `ALL`.
fn listed(lead: &str, names: impl IntoIterator<Item = &'static str>, tail: &str) -> String {
    format!(
        "{lead}: {}{tail}",
        names.into_iter().collect::<Vec<_>>().join(", ")
    )
}

/// The help of `--sandbox-backend`: this platform's values, from the table the probe refuses by.
fn sandbox_backend_help() -> String {
    let values = crate::config::SandboxBackend::ON_THIS_PLATFORM;
    if values.is_empty() {
        return "Sandbox backend: this platform takes none".to_string();
    }
    listed(
        "Sandbox backend",
        values.iter().map(|backend| backend.name()),
        "",
    )
}

/// The long help of `profile set`: the settable keys, from the table `profile set` refuses by.
fn profile_set_about() -> String {
    format!(
        "Set one of a profile's settings\n\nKeys: {}. `account` is not settable; add a profile on the \
         other account instead.",
        profile::SETTABLE_PROFILE_KEYS.join(", ")
    )
}

#[derive(Parser, Debug)]
#[command(name = "meka", version, about = "A general-purpose AI agent harness")]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Option<Command>,
    /// Prompt for the first turn; `-` reads it from stdin
    #[arg(short = 'p', long = "prompt", value_name = "TEXT")]
    pub(crate) prompt: Option<String>,
    /// Skill to invoke on the first turn
    #[arg(long = "skill", value_name = "NAME")]
    pub(crate) skill: Option<String>,
    /// Continue the most recent session
    #[arg(short = 'c', long = "continue", conflicts_with = "resume")]
    pub(crate) continue_last: bool,
    /// Resume a session by id or unique prefix, or pick one from a list
    #[arg(short = 'r', long = "resume", value_name = "SESSION", num_args = 0..=1)]
    pub(crate) resume: Option<Option<String>>,
    /// Profile to use for this session
    ///
    /// Selects a whole profile (account, model and every model-tied setting); change a field with
    /// `meka profile set`.
    #[arg(long = "profile", value_name = "NAME")]
    pub(crate) profile: Option<String>,
    /// Project this session belongs to
    ///
    /// With `-c`, continues the latest session on that project, or starts one there. With
    /// `-r <SESSION>`, moves that session onto the project.
    #[arg(long = "project", value_name = "NAME")]
    pub(crate) project: Option<String>,
    #[arg(
        long = "permission",
        value_name = "LEVEL",
        help = listed(
            "Initial permission level",
            Permission::ALL.iter().map(|level| level.name()),
            ""
        )
    )]
    pub(crate) permission: Option<Permission>,
    /// Standing instructions for this run
    #[arg(long = "instructions", value_name = "STRING")]
    pub(crate) instructions: Option<String>,
    /// Directory added to this run's workspace (repeatable)
    ///
    /// Part of the workspace with the working directory's rights: writable at `workspace`,
    /// read-only at `read`, searched with it. An ACP client's folders join the same list.
    // A flag rather than a config key: which directories this run works in is per-run scope, like
    // the working directory itself, not a preference to persist.
    #[arg(long = "additional-workspace-directory", value_name = "DIR")]
    pub(crate) additional_workspace_directory: Vec<std::path::PathBuf>,
    /// Directory this run may read (repeatable)
    ///
    /// The per-run form of `[shell].additional_readable_directories`; in effect where reads are
    /// confined.
    #[arg(long = "additional-readable-directory", value_name = "DIR")]
    pub(crate) additional_readable_directory: Vec<std::path::PathBuf>,
    #[arg(
        long = "sandbox-backend",
        value_name = "BACKEND",
        help = sandbox_backend_help()
    )]
    pub(crate) sandbox_backend: Option<crate::config::SandboxBackend>,
    /// MCP tool to eager-load (repeatable)
    #[arg(long = "eager-load-tool", value_name = "SERVER:TOOL")]
    pub(crate) eager_load_tool: Vec<String>,
    /// Disable streaming for this run
    #[arg(long = "no-stream")]
    pub(crate) no_stream: bool,
    #[arg(long = "format", default_value = "plain", value_name = "FORMAT", help = output_format_help())]
    pub(crate) format: OutputFormat,
    /// Exit after the first turn
    #[arg(long = "oneshot")]
    pub(crate) oneshot: bool,
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
            project: self.project.clone(),
            eager_load_tools: self.eager_load_tool.clone(),
            instructions: self.instructions.clone(),
            permission: self.permission,
            sandbox_backend: self.sandbox_backend,
            additional_workspace_directories: self.additional_workspace_directory.clone(),
            additional_readable_directories: self.additional_readable_directory.clone(),
            no_stream: self.no_stream,
            resume: self.resume.clone(),
            continue_last: self.continue_last,
            prompt: self.prompt.clone(),
            oneshot: self.oneshot,
            output_format: self.format,
            serve: matches!(self.command, Some(Command::Serve { .. })),
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
/// The terminal's side of a browser login, an account's or an MCP server's: the URL on stderr,
/// and the pasted line read from stdin. The screen has its own, since it owns both.
pub(crate) struct TerminalLogin;

#[async_trait::async_trait]
impl crate::oauth::LoginPrompt for TerminalLogin {
    fn authorize_at(&self, url: &str) {
        // The URL, exactly once, and no browser launched: which browser, and on which machine, is
        // the user's to decide, and a launch from an SSH session or a container reached nothing or
        // the wrong desktop.
        crate::streams::write_stderr_line(format!(
            "To authorize, open this URL in your browser:\n\n{url}\n"
        ));
    }

    fn awaiting(&self, completion: crate::oauth::Completion) {
        use crate::oauth::Paste;

        let line = match (completion.listener_seconds, completion.paste) {
            (Some(seconds), Some(_)) => format!(
                "Waiting up to {seconds}s for the callback, or paste the callback URL here and \
                 press Enter:"
            ),
            (Some(seconds), None) => format!("Waiting up to {seconds}s for the callback."),
            (None, Some(Paste::CallbackUrl)) => {
                "Paste the full callback URL the browser was sent to and press Enter:".to_string()
            }
            (None, Some(Paste::AuthorizationCode)) => {
                "After authorizing, paste the authorization code here and press Enter:".to_string()
            }
            (None, None) => return,
        };
        crate::streams::write_stderr_line(line);
    }

    fn accepts_paste(&self) -> bool {
        std::io::IsTerminal::is_terminal(&std::io::stdin())
    }

    async fn read_pasted_line(&self) -> std::result::Result<Option<String>, String> {
        use tokio::io::{AsyncBufReadExt, BufReader};

        let mut reader = BufReader::new(tokio::io::stdin());
        let mut line = String::new();
        match reader.read_line(&mut line).await {
            Err(error) => Err(format!("failed to read stdin: {error}")),
            // EOF: stdin was closed before the user pasted anything.
            Ok(0) => Ok(None),
            Ok(_) => Ok(Some(line)),
        }
    }

    fn paste_refused(&self, reason: &str) {
        // Said as a line of meka's own, which starts with a capital whatever the refusal does.
        let mut line = reason.to_string();
        if let Some(first) = line.get(..1) {
            line.replace_range(..1, &first.to_ascii_uppercase());
        }
        crate::streams::write_stderr_line(line);
    }
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
    fn session_set_and_search_parse_and_an_unknown_field_is_refused() {
        let cli = Cli::parse_from(["meka", "session", "set", "abc", "title", "My  title"]);
        match cli.command {
            Some(Command::Session {
                action:
                    SessionAction::Set {
                        session_id,
                        field,
                        value,
                    },
            }) => {
                assert_eq!(session_id, "abc");
                assert_eq!(field, SessionField::Title);
                assert_eq!(value, "My  title");
            }
            other => panic!("expected `session set`, got {other:?}"),
        }
        let cli = Cli::parse_from(["meka", "session", "set", "abc", "pinned", "true"]);
        assert!(matches!(
            cli.command,
            Some(Command::Session {
                action: SessionAction::Set {
                    field: SessionField::Pinned,
                    ..
                }
            })
        ));
        let error = Cli::try_parse_from(["meka", "session", "set", "abc", "color", "x"])
            .expect_err("an unknown field must not parse")
            .to_string();
        assert!(error.contains("pinned, title"), "{error}");

        let cli = Cli::parse_from([
            "meka",
            "session",
            "search",
            "blind spot",
            "-n",
            "5",
            "--include-children",
        ]);
        match cli.command {
            Some(Command::Session {
                action:
                    SessionAction::Search {
                        query,
                        limit,
                        include_children,
                        format,
                    },
            }) => {
                assert_eq!(query, "blind spot");
                assert_eq!(limit, 5);
                assert!(include_children);
                assert_eq!(format, OutputFormat::Plain);
            }
            other => panic!("expected `session search`, got {other:?}"),
        }
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
            cli.resume,
            Some(Some("550e8400-e29b-41d4-a716-446655440000".to_string()))
        );
        assert!(!cli.continue_last);
    }

    /// `-r` alone asks for the picker. The value can be optional because the root command has
    /// no positional: the prompt is `-p`, so nothing after a bare `-r` is mistaken for a session.
    #[test]
    fn cli_bare_resume_asks_for_the_picker() {
        let cli = Cli::parse_from(["meka", "-r"]);
        assert_eq!(cli.resume, Some(None));
        let cli = Cli::parse_from(["meka", "-r", "-p", "hi"]);
        assert_eq!(cli.resume, Some(None));
        assert_eq!(cli.prompt.as_deref(), Some("hi"));
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
        assert_eq!(
            cli.resume.as_ref().and_then(|given| given.as_deref()),
            Some("550e8400")
        );
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
        let refused: [&[&str]; 6] = [
            &["meka", "session", "export", "x", "--format", "md"],
            &["meka", "session", "export", "x", "--format", "JSON"],
            &["meka", "--oneshot", "-p", "x", "--format", "Json"],
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
            cli.resume.as_ref().and_then(|given| given.as_deref()),
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
            "claude-opus-5-5",
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
                assert_eq!(model.as_deref(), Some("claude-opus-5-5"));
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
                        project,
                        format,
                    },
            }) => {
                assert_eq!(limit, 20);
                assert!(project.is_none());
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
        let commands: [&[&str]; 20] = [
            &["session", "list"],
            &["session", "show", "0e5f"],
            &["account", "list"],
            &["account", "usage"],
            &["account", "whoami"],
            &["account", "stats"],
            &["profile", "list"],
            &["mcp", "list"],
            &["mcp", "get", "s"],
            &["mcp", "tools", "s"],
            &["schedule", "list"],
            &["schedule", "show", "7f3a"],
            &["memory", "list"],
            &["memory", "get", "m"],
            &["memory", "show", "m"],
            &["tool", "list"],
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
            // assertion shape for twenty variants.
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
    fn cli_session_remove_all_subcommand() {
        let cli = Cli::parse_from(["meka", "session", "remove", "--all"]);
        match cli.command {
            Some(Command::Session {
                action:
                    SessionAction::Remove {
                        session_ids, all, ..
                    },
            }) => {
                assert!(session_ids.is_empty());
                assert!(all);
            }
            other => panic!("expected session remove, got {other:?}"),
        }
    }

    /// Naming sessions and then asking for every session are two different requests.
    ///
    /// Taking both and quietly honoring the wider one would let `meka session remove "$ID" --all`
    /// with `$ID` unset delete the store and report the empty id as a failure: a complete success
    /// reported as an error, over work nobody asked for. Same reasoning as the `--older-than-days`
    /// conflicts beside it.
    #[test]
    fn cli_session_remove_all_conflicts_with_explicit_ids() {
        let id = "550e8400-e29b-41d4-a716-446655440000";
        assert!(
            Cli::try_parse_from(["meka", "session", "remove", id, "--all"]).is_err(),
            "naming an id and asking for --all must be refused"
        );
    }

    /// The manual counterpart to `[session].retention`, so it has to actually parse.
    #[test]
    fn cli_session_remove_older_than_days() {
        let cli = Cli::parse_from(["meka", "session", "remove", "--older-than-days", "90"]);
        match cli.command {
            Some(Command::Session {
                action:
                    SessionAction::Remove {
                        session_ids,
                        all,
                        older_than_days,
                    },
            }) => {
                assert!(session_ids.is_empty());
                assert!(!all);
                assert_eq!(older_than_days, Some(90));
            }
            other => panic!("expected session remove, got {other:?}"),
        }
    }

    /// Both other selectors must be refused alongside it. `--all` because the two windows
    /// disagree, and explicit ids because a listed session younger than the window would be
    /// silently spared while the user watched a different count come back.
    #[test]
    fn cli_session_remove_older_than_days_conflicts() {
        let id = "550e8400-e29b-41d4-a716-446655440000";
        for arguments in [
            vec![
                "meka",
                "session",
                "remove",
                "--older-than-days",
                "90",
                "--all",
            ],
            vec!["meka", "session", "remove", "--older-than-days", "90", id],
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
