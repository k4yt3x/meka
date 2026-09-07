//! Configuration: parses `~/.config/meka/config.toml`, layers CLI overrides and environment
//! variables on top, and produces a [`ResolvedConfig`] that the rest of the binary consumes.

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{
    fs::*,
    paths::{config_file_path, expand_user_path, meka_config_dir},
    permission::{EnabledPermissions, Permission},
};

/// Accounts and profiles: the `backend` vocabulary, an account and a profile as written and a
/// profile as resolved through its account, and which profile a run selects.
mod profile;

#[cfg(test)]
pub(crate) use profile::PROFILE_KEY_ORDER;
pub(crate) use profile::{
    AccountConfig, Backend, ProfileConfig, ProfileRequest, ProfileSettings, account_for,
    default_profile_on_disk, require_profile, resolve_device_id, resolve_profile, select_profile,
    sort_account_keys, sort_profile_keys, validate_max_output_tokens,
};

/// In-memory shape of `config.toml`. Each top-level `[section]` deserializes into its own
/// sub-struct; missing sections fall back to `Default`. This is the raw deserialized form;
/// [`ResolvedConfig::resolve`] merges it with CLI flags and env vars.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct ConfigFile {
    /// Name of the profile to use when no `--profile` flag is given.
    pub(crate) default_profile: Option<String>,
    /// Named accounts, parsed from `[accounts.<name>]`. Each pins a `backend` plus the endpoint
    /// and OAuth settings a login needs; credentials live in the DB keyed by the account name.
    #[serde(default)]
    pub(crate) accounts: std::collections::BTreeMap<String, AccountConfig>,
    /// Named profiles, parsed from `[profiles.<name>]`. Each names an account plus the model and
    /// every model-tied setting.
    #[serde(default)]
    pub(crate) profiles: std::collections::BTreeMap<String, ProfileConfig>,
    pub(crate) display: Option<DisplayConfig>,
    pub(crate) web: Option<WebConfig>,
    pub(crate) shell: Option<ShellConfig>,
    pub(crate) session: Option<SessionConfig>,
    pub(crate) thinking: Option<ThinkingConfig>,
    pub(crate) mcp: Option<McpConfig>,
    pub(crate) tools: Option<ToolsConfig>,
    pub(crate) skills: Option<SkillsConfig>,
    pub(crate) memory: Option<MemoryConfig>,
    pub(crate) permissions: Option<PermissionsConfig>,
    pub(crate) schedule: Option<ScheduleConfig>,
    pub(crate) background: Option<BackgroundConfig>,
    pub(crate) subagents: Option<SubagentsConfig>,
    pub(crate) serve: Option<ServeConfig>,
}

/// `[skills]` table: the user-authored skill store ([`crate::skills`]).
///
/// Same shape and rationale as [`MemoryConfig`]: config-only, no env var, no CLI flag. Turning it
/// off keeps the `skill` tool's schema out of every request and stops the `[Skills]` section from
/// rendering, for an installation that never uses skills.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct SkillsConfig {
    pub(crate) enabled: Option<bool>,
    /// Whether the agent may author skills, adding `skill_write` and `skill_delete` to its
    /// registry. Default `false`.
    ///
    /// Off by default because for the ordinary terminal session the human authors and curates
    /// skills, and an agent rewriting that store is not something the user asked for. It earns its
    /// keep in the opposite deployment: a long-running dispatcher that spawns sub-agents, where a
    /// skill is the only artifact that both outlives the session and can be handed to a sub-agent
    /// as its task, so writing one is how the agent gets a refined sub-agent brief to the next
    /// sub-agent without routing it through its own context window.
    ///
    /// Never granted to a sub-agent whatever this says; see the registration site in
    /// [`crate::tools`].
    pub(crate) agent_managed: Option<bool>,
    /// Additional directories to scan for skills, **read-only**. Default empty.
    ///
    /// meka writes only to its own `skills/` directory under the config dir; these roots are never
    /// created and never written to, so listing one that does not exist costs nothing and leaves
    /// no trace. That is what makes it safe to point at a shared location like `~/.agents/skills`,
    /// which other Agent Skills clients populate: meka reads what is already there rather than
    /// putting anything in `$HOME` itself.
    ///
    /// Empty by default, and deliberately not defaulted to the cross-client convention: whether a
    /// directory outside meka's own namespace should be read is the user's call, not meka's.
    ///
    /// A leading `~` is expanded (see [`expand_user_path`]). A relative path is resolved against
    /// the process working directory, which is why nothing here is scanned implicitly: meka does
    /// not auto-trust the cwd, so a project's skills are read only when a path names them.
    pub(crate) extra_paths: Option<Vec<String>>,
}

/// `[schedule]` table: agent-created wakeups ([`crate::schedule`]).
///
/// Config-only, like `[skills]` and `[memory]`: whether an agent may schedule its own future turns
/// is a property of the installation. Turning it off keeps the three `schedule_*` tool schemas out
/// of every request, stops the `[Scheduled]` section rendering, and leaves the scheduler unstarted,
/// so existing jobs stay on disk without firing.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct ScheduleConfig {
    pub(crate) enabled: Option<bool>,
    /// How often the scheduler looks for due jobs. Accepts humantime strings like `"10s"`, `"1m"`.
    /// Default `"10s"`. This is the real resolution floor: a job with a shorter interval fires
    /// once per tick, not once per interval.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) poll_interval: Option<std::time::Duration>,
    /// How late a *one-shot* job may be and still fire after downtime. Accepts humantime strings
    /// like `"24h"`. Default `"24h"`.
    ///
    /// Recurring jobs need no equivalent: their occurrences are one period apart, so the most
    /// recent missed one is always less than a period old, and the scheduler coalesces the rest.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) missed_grace: Option<std::time::Duration>,
    /// Wall-clock budget for a gate probe. Accepts humantime strings like `"30s"`. Default
    /// `"30s"`. A gate that overruns is a failure, not a silent skip.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) gate_timeout: Option<std::time::Duration>,
    /// How long a host's claim on an occurrence is good for. Default `"1h"`.
    ///
    /// The window in which a crashed host's job is nobody's, so it is the delay before another
    /// host picks it up. It should comfortably exceed a gate probe plus a model turn: a lease
    /// that expires while the holder is still working lets a second host take the same
    /// occurrence, and although the session lock then catches it and defers, that costs the
    /// occurrence a round trip and re-runs the gate probe. An hour is far longer than any
    /// realistic turn and still short enough that a machine which rebooted overnight has caught
    /// up by morning.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) claim_lease: Option<std::time::Duration>,
    /// Ceiling on jobs per session, refused at `schedule_create`. Default 50.
    pub(crate) max_jobs: Option<usize>,
    /// How many of one session's jobs may spend a turn in a single sweep. Default 5. Jobs past it
    /// keep their occurrence and fire on the next sweep.
    pub(crate) max_consecutive_fires: Option<usize>,
}

/// [`ScheduleConfig`] with every default filled in.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedScheduleConfig {
    pub(crate) enabled: bool,
    /// The levels this installation permits at all.
    ///
    /// The fire-time gate check reads a session's recorded level straight out of the row, and a
    /// row outlives the configuration that produced it. Without this filter, narrowing
    /// `[permissions].enabled` and restarting would leave an `unrestricted` row firing its gate
    /// forever while the live session re-attaches at `read`: the exact fail-open the live re-check
    /// exists to prevent, reached from the one reader that did not apply it.
    pub(crate) enabled_permissions: crate::permission::EnabledPermissions,
    pub(crate) poll_interval: std::time::Duration,
    pub(crate) missed_grace: std::time::Duration,
    pub(crate) gate_timeout: std::time::Duration,
    /// See [`ScheduleConfig::claim_lease`].
    pub(crate) claim_lease: std::time::Duration,
    pub(crate) max_jobs: usize,
    pub(crate) max_consecutive_fires: usize,
}

impl ResolvedScheduleConfig {
    /// An hour: far longer than any realistic gate plus turn, so a lease cannot expire under a
    /// host that is still working, and short enough that a machine which rebooted overnight has
    /// caught up by morning.
    const DEFAULT_CLAIM_LEASE: std::time::Duration = std::time::Duration::from_secs(60 * 60);
    const DEFAULT_GATE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
    /// Five is well above what a healthy session has due at once, since coalescing means one job
    /// contributes one fire however long the outage was, so the budget only engages on a real
    /// backlog. What it buys there is interleaving, not throttling: sweeps do not overlap and the
    /// next begins as soon as the last ends, so a backlog produces the same number of turns either
    /// way, but another session's due job is reached after five of this one's rather than after
    /// all of them.
    const DEFAULT_MAX_CONSECUTIVE_FIRES: usize = 5;
    const DEFAULT_MAX_JOBS: usize = 50;
    /// A day. Long enough that an overnight outage still delivers this morning's reminder, short
    /// enough that a laptop closed for a week does not wake to a pile of stale ones.
    const DEFAULT_MISSED_GRACE: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);
    /// Ten seconds trades scheduling precision against wakeups: a minute-granularity cron never
    /// needs better, and it keeps a `--oneshot` run from paying for a tick it will never use.
    const DEFAULT_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);

    fn resolve(
        raw: Option<ScheduleConfig>,
        enabled_permissions: crate::permission::EnabledPermissions,
    ) -> Self {
        let raw = raw.unwrap_or_default();
        Self {
            enabled_permissions,
            enabled: raw.enabled.unwrap_or(true),
            poll_interval: raw.poll_interval.unwrap_or(Self::DEFAULT_POLL_INTERVAL),
            missed_grace: raw.missed_grace.unwrap_or(Self::DEFAULT_MISSED_GRACE),
            gate_timeout: raw.gate_timeout.unwrap_or(Self::DEFAULT_GATE_TIMEOUT),
            claim_lease: raw.claim_lease.unwrap_or(Self::DEFAULT_CLAIM_LEASE),
            max_jobs: raw.max_jobs.unwrap_or(Self::DEFAULT_MAX_JOBS),
            max_consecutive_fires: raw
                .max_consecutive_fires
                .unwrap_or(Self::DEFAULT_MAX_CONSECUTIVE_FIRES),
        }
    }
}

impl Default for ResolvedScheduleConfig {
    fn default() -> Self {
        // `DEFAULT` for the enabled set rather than something narrower: it can only ever *narrow*
        // what a row already recorded, and `from_levels` guarantees a non-empty set anyway.
        Self::resolve(None, crate::permission::EnabledPermissions::DEFAULT)
    }
}

/// `[background]` table: tool calls the agent starts and does not wait for ([`crate::background`]).
///
/// Config-only, like `[schedule]`, `[skills]`, and `[memory]`.
///
/// Alone among them it defaults **off**, which is deliberate rather than caution. Those three add
/// capability without changing when a turn ends; this changes the contract of the primary
/// interaction from "you asked, it answered" into "and something else may interrupt you later",
/// which is right for an unattended assistant and wrong for someone using the REPL as a command
/// line. A scheduled job also takes an explicit, visible act to create, whereas `background` is
/// reachable from any tool call, so an agent will reach for it unprompted.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct BackgroundConfig {
    pub(crate) enabled: Option<bool>,
    /// Ceiling on tasks running at once per session, refused at dispatch. Default 10.
    pub(crate) max_tasks: Option<usize>,
}

/// [`BackgroundConfig`] with every default filled in.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedBackgroundConfig {
    pub(crate) enabled: bool,
    pub(crate) max_tasks: usize,
}

impl ResolvedBackgroundConfig {
    /// Enough for a fan-out of parallel work, low enough that a runaway loop is capped before it
    /// has spawned a hundred shells.
    const DEFAULT_MAX_TASKS: usize = 10;

    fn resolve(raw: Option<BackgroundConfig>) -> Self {
        let raw = raw.unwrap_or_default();
        Self {
            enabled: raw.enabled.unwrap_or(false),
            max_tasks: raw.max_tasks.unwrap_or(Self::DEFAULT_MAX_TASKS),
        }
    }
}

impl Default for ResolvedBackgroundConfig {
    fn default() -> Self {
        Self::resolve(None)
    }
}

/// `[memory]` table: the agent's durable note store ([`crate::memory`]).
///
/// Config-only, with no env var and no CLI flag: whether an agent keeps memories is a property of
/// the installation, not something to vary per run. Turning it off keeps the four `memory_*` tool
/// schemas out of every request, which is the point for someone running lean coding sessions.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct MemoryConfig {
    pub(crate) enabled: Option<bool>,
}

/// `[permissions]` table: choose which levels are reachable at runtime and which level the
/// session starts in. See `docs/book/src/usage/permissions.md`.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct PermissionsConfig {
    /// Typed rather than read as strings, so a level meka does not have is refused where the file
    /// is parsed, with its line, the way an unknown key is.
    pub(crate) default: Option<Permission>,
    pub(crate) enabled: Option<Vec<Permission>>,
    /// Whether a tool call needing more than the session's level is submitted to the user for
    /// approval rather than refused. Off unless set; a session records its own afterwards.
    pub(crate) approvals: Option<bool>,
}

/// Built-in tool filters, mirroring the per-server knobs on [`McpServerConfig`]. Applied at
/// registration time by [`crate::tools::ToolRegistry`].
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct ToolsConfig {
    pub(crate) allowed_tools: Option<Vec<String>>,
    pub(crate) disabled_tools: Option<Vec<String>>,
    /// Typed, like `[permissions]`, so a level meka does not have is refused where the file is
    /// parsed, with its line, rather than dropped with a warning.
    pub(crate) tool_permissions: Option<HashMap<String, Permission>>,
}

/// `[subagents]` table: capabilities a sub-agent may never hold.
///
/// Deliberately only the two deny lists. The distinction that decides what belongs here: a
/// *capability* is something config can genuinely withhold, because a tool the registry never
/// registered cannot be reached however the parent phrases the task. *Context* (the memory store,
/// the instructions file) cannot be withheld the same way, because a parent holding it can copy
/// it into the sub-agent's prompt; a config key promising otherwise would read as a boundary while
/// being none. Context is therefore granted per call by `agent_spawn`, defaulting to nothing, and
/// has no key here.
///
/// The forgetting failure these lists exist for is real: MCP servers are inherited by default, so a
/// parent that never considers the question hands a sub-agent every tool it has, including one
/// that messages the user. `agent_spawn`'s `deny_servers` / `deny_tools` union with these, so a
/// parent can restrict further; there is deliberately no call-site allow-list, which would let a
/// parent widen what config denied.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct SubagentsConfig {
    /// Whole MCP servers a sub-agent cannot see. The coarse lever and the one that matters: naming
    /// a server removes its tools, its resources, and its prompts.
    pub(crate) disabled_servers: Option<Vec<String>>,
    /// Individual tools a sub-agent cannot see, by registry name, so built-ins and namespaced MCP
    /// tools (`mcp__<server>__<tool>`) share one namespace.
    pub(crate) disabled_tools: Option<Vec<String>>,
}

/// How much of the memory store an agent may reach.
///
/// Not a config key: the root agent always holds [`MemoryAccess::Write`], and a sub-agent holds
/// whatever its `agent_spawn` call granted, defaulting to [`MemoryAccess::None`]. Ordered `None <
/// Read < Write` so `min` reads as "the more restrictive of two", which is how a grant is clamped
/// against what the granting agent itself holds.
///
/// `Write` is deliberately unreachable from `agent_spawn` (see [`Self::parse_grant`]). A sub-agent
/// that inferred something from one narrow task should not be able to write it into the store every
/// future turn then reasons from; the parent, which has the whole conversation, is better placed to
/// decide what is worth remembering, and can record it after reading the sub-agent's report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum MemoryAccess {
    /// No memory tools at all.
    None,
    /// `memory_read` and `memory_search`; no `memory_write` or `memory_delete`.
    Read,
    /// All four memory tools. The root agent only.
    Write,
}

impl MemoryAccess {
    /// `None`, as a `#[serde(default)]` target for a persisted spec that lost the field: an absent
    /// grant must cost the sub-agent its memory tools rather than hand it the store.
    pub(crate) fn none() -> Self {
        Self::None
    }

    /// Parse a `agent_spawn` grant. `"write"` is refused with its reason rather than clamped, so a
    /// parent asking for it learns that no sub-agent can have it instead of silently getting
    /// `read`.
    pub(crate) fn parse_grant(text: &str) -> std::result::Result<Self, String> {
        match text.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "read" => Ok(Self::Read),
            "write" => Err(
                "memory = \"write\" is not available to sub-agents; grant \"read\" and record \
                 anything worth keeping yourself"
                    .to_string(),
            ),
            other => Err(format!(
                "memory = \"{other}\" is not an access level; use \"none\" or \"read\""
            )),
        }
    }
}

/// Whether an agent was handed the installation's instructions file.
///
/// Like [`MemoryAccess`], granted per `agent_spawn` call rather than configured, and defaulting to
/// [`InstructionAccess::None`]. Instructions describe the root agent (its persona, how to address
/// the user, what to volunteer), so a sub-agent inherits none of it unless the parent decides this
/// particular task needs the project's rules.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum InstructionAccess {
    /// A clean slate: the sub-agent's system prompt carries no user instructions.
    #[default]
    None,
    /// The sub-agent receives the instructions verbatim, as the parent has them.
    Inherit,
}

impl InstructionAccess {
    /// Parse a `agent_spawn` grant.
    pub(crate) fn parse_grant(text: &str) -> std::result::Result<Self, String> {
        match text.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "inherit" => Ok(Self::Inherit),
            other => Err(format!(
                "instructions = \"{other}\" is not an instruction grant; use \"none\" or \"inherit\""
            )),
        }
    }
}

/// [`SubagentsConfig`] with defaults filled in.
#[derive(Debug, Clone, Default)]
pub(crate) struct ResolvedSubagentsConfig {
    pub(crate) disabled_servers: Vec<String>,
    pub(crate) disabled_tools: Vec<String>,
}

impl ResolvedSubagentsConfig {
    fn resolve(raw: Option<SubagentsConfig>) -> Self {
        let raw = raw.unwrap_or_default();
        Self {
            disabled_servers: raw.disabled_servers.unwrap_or_default(),
            disabled_tools: raw.disabled_tools.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct ThinkingConfig {
    /// Budget for [`crate::config::ThinkingMode::Budgeted`]; ignored under the other modes.
    ///
    /// The installation-wide fallback, which `[profiles.<name>].thinking_budget` takes precedence
    /// over: one number for every profile that does not state its own.
    pub(crate) budget: Option<u64>,
    pub(crate) show_content: Option<bool>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct DisplayConfig {
    pub(crate) newline_before_prompt: Option<bool>,
    pub(crate) newline_after_prompt: Option<bool>,
    pub(crate) show_session_id_on_create: Option<bool>,
    pub(crate) show_session_id_on_exit: Option<bool>,
    pub(crate) show_path_in_prompt: Option<bool>,
    pub(crate) show_context_in_prompt: Option<bool>,
    pub(crate) show_token_usage: Option<bool>,
    pub(crate) render_mode: Option<RenderMode>,
    /// How much of a tool call's input the `[tool X]` indicator shows: `off`, `summary` (default),
    /// or `full`.
    pub(crate) tool_params: Option<ToolParams>,
    /// Widest line meka composes from model output, in terminal columns. Unset follows the
    /// terminal, which is what keeps a line from ever wrapping; a set value is honored exactly,
    /// so one wider than the terminal will wrap. Covers meka's own output only: assistant
    /// markdown reflows to the real terminal through `render_mode` either way.
    pub(crate) max_width: Option<usize>,
    /// Style applied to the REPL input buffer so submitted prompts stand out in scrollback. Parsed
    /// by [`parse_input_style`]. Accepts `bold`, `dim`, `none`, or a color name (`cyan`,
    /// `yellow`, …).
    pub(crate) input_style: Option<String>,
    /// When set to `Some(N)` with `N > 0`, resuming a session reprints the last `N` turns (user
    /// prompts plus the agent's response, styled like the live REPL). Unset reprints only the last
    /// assistant message.
    pub(crate) resume_show_recent: Option<usize>,
    /// Whether answers stream as they arrive. `false` is what `--no-stream` asks for one run; a
    /// preference belongs in the file, like every other display setting.
    pub(crate) stream: Option<bool>,
}

/// Whether answers stream: the flag turns it off for one run, the file is the standing preference,
/// and either saying no wins.
pub(crate) fn streaming_enabled(no_stream_flag: bool, configured: Option<bool>) -> bool {
    !no_stream_flag && configured.unwrap_or(true)
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct WebConfig {
    pub(crate) user_agent: Option<String>,
    /// Total request budget: connect, TLS and read. Default `"30s"`; `"0s"` is refused at
    /// startup.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) request_timeout: Option<std::time::Duration>,
    /// Separate cap on the TCP and TLS handshake. Unset means no separate cap; `"0s"` is refused
    /// at startup.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) connect_timeout: Option<std::time::Duration>,
    /// Per-chunk idle timeout, for a body that stalls mid-stream. Unset means none; `"0s"` is
    /// refused at startup.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) read_timeout: Option<std::time::Duration>,
    /// Max number of redirects reqwest will follow. `0` disables redirects entirely. Default:
    /// `10`.
    pub(crate) max_redirects: Option<u64>,
    /// Proxy URL. Accepts `http://…`, `https://…`, `socks5://…`, `socks5h://…`. The literal string
    /// `"none"` explicitly disables env-var auto-detection (overriding `HTTP_PROXY` etc.). Unset
    /// honors the env vars.
    pub(crate) proxy: Option<String>,
    /// Path to a PEM file containing one or more root CAs to trust on top of the system trust
    /// store. Used for corporate MITM proxies or self-signed internal services.
    pub(crate) ca_cert_file: Option<String>,
    /// Reject plain `http://` URLs: only `https://` allowed.
    pub(crate) https_only: Option<bool>,
    /// Minimum TLS version: `"1.0"`, `"1.1"`, `"1.2"`, or `"1.3"`. Anything else logs a warn and
    /// falls back to reqwest's default.
    pub(crate) min_tls_version: Option<String>,
    /// DANGER: disable TLS certificate validation entirely. Allows MITM; only use against trusted
    /// local dev servers.
    pub(crate) danger_accept_invalid_certs: Option<bool>,
    /// DANGER: accept certificates whose hostname doesn't match. Allows MITM; only use against
    /// trusted local dev servers.
    pub(crate) danger_accept_invalid_hostnames: Option<bool>,
}

/// Minimum TLS version accepted by the web-tools client. Normalized from `[web].min_tls_version` so
/// the rest of the crate doesn't pass free-form strings around.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MinTlsVersion {
    V1_0,
    V1_1,
    V1_2,
    V1_3,
}

impl MinTlsVersion {
    /// Parse the config-file string form. Returns `None` on unknown input; callers are expected to
    /// log and fall through to the reqwest backend default.
    pub(crate) fn parse(raw: &str) -> Option<Self> {
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
pub(crate) struct WebClientConfig {
    pub(crate) user_agent: String,
    pub(crate) request_timeout: std::time::Duration,
    pub(crate) connect_timeout: Option<std::time::Duration>,
    pub(crate) read_timeout: Option<std::time::Duration>,
    /// `0` means "no redirects" (`Policy::none`); any other value becomes `Policy::limited(n)`.
    pub(crate) max_redirects: usize,
    pub(crate) proxy: Option<String>,
    pub(crate) ca_cert_file: Option<std::path::PathBuf>,
    pub(crate) https_only: bool,
    pub(crate) min_tls_version: Option<MinTlsVersion>,
    pub(crate) danger_accept_invalid_certs: bool,
    pub(crate) danger_accept_invalid_hostnames: bool,
}

impl Default for WebClientConfig {
    fn default() -> Self {
        Self {
            user_agent: DEFAULT_WEB_USER_AGENT.to_string(),
            request_timeout: DEFAULT_WEB_REQUEST_TIMEOUT,
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
    pub(crate) fn from_file(file: &WebConfig) -> Self {
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
                            "ignoring [web].min_tls_version '{raw}'; supported: 1.0, 1.1, 1.2, 1.3"
                        );
                        None
                    }
                });

        Self {
            user_agent,
            // A zero is carried through rather than filtered: `ResolvedConfig::validate` refuses
            // it, and filtering here is what let a `0` silently mean "the default".
            request_timeout: file.request_timeout.unwrap_or(DEFAULT_WEB_REQUEST_TIMEOUT),
            connect_timeout: file.connect_timeout,
            read_timeout: file.read_timeout,
            max_redirects: file
                .max_redirects
                .map(|n| usize::try_from(n).unwrap_or(usize::MAX))
                .unwrap_or(10),
            proxy: file.proxy.clone(),
            // Through the same `~` expansion as every other path setting; a `~/certs/x.pem` was
            // read verbatim and failed at the web client build with "No such file". A form the
            // expansion does not take (`~user/...`) is kept as written.
            ca_cert_file: file.ca_cert_file.as_deref().map(|path| {
                expand_user_path(path).unwrap_or_else(|| std::path::PathBuf::from(path))
            }),
            https_only: file.https_only.unwrap_or(false),
            min_tls_version,
            danger_accept_invalid_certs: file.danger_accept_invalid_certs.unwrap_or(false),
            danger_accept_invalid_hostnames: file.danger_accept_invalid_hostnames.unwrap_or(false),
        }
    }
}

/// Default UA for the web tools when `[web].user_agent` is unset. Kept in sync with what real
/// Chrome emits so anti-bot filters don't single out meka by default.
pub(crate) const DEFAULT_WEB_USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36";
/// `[web].request_timeout` when unset.
pub(crate) const DEFAULT_WEB_REQUEST_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(30);
/// `[mcp].grace` when unset.
pub(crate) const DEFAULT_MCP_GRACE: std::time::Duration = std::time::Duration::from_secs(3);
/// `[mcp].connect_timeout` when unset. `crate::mcp::DEFAULT_MCP_REQUEST_TIMEOUT` matches it.
pub(crate) const DEFAULT_MCP_CONNECT_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(30);
/// `[mcp].stdio_concurrency` when unset: each is a process launch.
pub(crate) const DEFAULT_MCP_STDIO_CONCURRENCY: usize = 3;
/// `[mcp].http_concurrency` when unset: a connect is a request, not a process.
pub(crate) const DEFAULT_MCP_HTTP_CONCURRENCY: usize = 20;

/// Max conversation messages kept in the per-turn API window by default.
const DEFAULT_CONTEXT_MESSAGES: usize = 200;
/// Default extended-thinking token budget.
pub(crate) const DEFAULT_THINKING_BUDGET_TOKENS: u64 = 16_000;
/// Default maximum sub-agent recursion depth (root spawns down to grandchild).
const DEFAULT_SUBAGENT_MAX_DEPTH: usize = 3;

/// Narrowest `[display].max_width` that still leaves room for anything.
///
/// Every budget derived from it subtracts fixed chrome first (`[tool ` and its brackets, the
/// `Thinking... ` prefix, a block's indent), so below roughly this the subtraction leaves nothing
/// and output degrades to punctuation. Clamped rather than rejected: a too-narrow width is a
/// preference stated badly, not a broken config, and refusing to start over it would be
/// disproportionate.
const MIN_CONFIGURED_WIDTH: usize = 40;

/// Widest `[display].max_width` worth honoring.
///
/// Not a taste judgment about long lines. Fitting text to a column budget re-measures a growing
/// prefix, so the work to compose one line grows with the square of the width: a stray `max_width =
/// 100000` turns each elided line into billions of width computations and hangs the renderer. No
/// terminal is this wide, so nothing legitimate is refused.
const MAX_CONFIGURED_WIDTH: usize = 1000;

fn clamp_max_width(configured: usize) -> usize {
    if configured < MIN_CONFIGURED_WIDTH {
        tracing::warn!(
            "[display].max_width = {configured} is below the {MIN_CONFIGURED_WIDTH}-column \
             minimum; using {MIN_CONFIGURED_WIDTH}"
        );
        return MIN_CONFIGURED_WIDTH;
    }
    if configured > MAX_CONFIGURED_WIDTH {
        tracing::warn!(
            "[display].max_width = {configured} is above the {MAX_CONFIGURED_WIDTH}-column \
             maximum; using {MAX_CONFIGURED_WIDTH}"
        );
        return MAX_CONFIGURED_WIDTH;
    }
    configured
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct ShellConfig {
    pub(crate) sandbox: Option<bool>,
    /// Linux-only choice between `"landlock"` and `"bubblewrap"`. When omitted, the resolver
    /// auto-picks bubblewrap if available and falls back to landlock with a one-shot warning (see
    /// `src/sandbox.rs` and `Warn 2` in `warn_if_sandbox_issues`).
    pub(crate) sandbox_backend: Option<SandboxBackend>,
}

/// Linux sandbox backend. Silently ignored on macOS and Windows (sandbox-exec and Low-integrity
/// respectively are the only options on those platforms). The absence of a default impl is
/// intentional: an unset value is meaningful and triggers auto-resolution in
/// [`ResolvedConfig::resolve`].
///
/// The wire spelling is [`Self::name`], which the file, the flag and the variable take; `Display`
/// prints the brand-cased [`Self::display_name`] for prose, so a message reads "Landlock" while the
/// config key still says `landlock`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub(crate) enum SandboxBackend {
    Landlock,
    Bubblewrap,
}

impl SandboxBackend {
    /// Every backend, in the order the names sort.
    pub(crate) const ALL: [SandboxBackend; 2] = [Self::Bubblewrap, Self::Landlock];

    /// The one spelling `[shell].sandbox_backend`, `--sandbox-backend` and `MEKA_SANDBOX_BACKEND`
    /// take.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Landlock => "landlock",
            Self::Bubblewrap => "bubblewrap",
        }
    }

    /// Brand-cased name for user-facing prose (logs, errors). Also the `Display` impl's output.
    pub(crate) const fn display_name(self) -> &'static str {
        match self {
            Self::Landlock => "Landlock",
            Self::Bubblewrap => "Bubblewrap",
        }
    }

    /// The names, joined for a refusal that lists what would have been accepted.
    pub(crate) fn supported() -> String {
        Self::ALL
            .iter()
            .map(|backend| backend.name())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl std::fmt::Display for SandboxBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.display_name())
    }
}

impl std::str::FromStr for SandboxBackend {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|backend| backend.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not a sandbox backend. Supported: {}",
                    Self::supported()
                )
            })
    }
}

impl TryFrom<String> for SandboxBackend {
    type Error = String;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<SandboxBackend> for String {
    fn from(backend: SandboxBackend) -> Self {
        backend.name().to_string()
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct SessionConfig {
    pub(crate) context_messages: Option<usize>,
    /// Delete sessions whose `updated_at` is older than this, at agent startup. Unset means keep
    /// everything: conversation history is not reproducible, so meka never discards it unasked.
    /// `"0s"` is refused at startup, since it would delete every session on each launch.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) retention: Option<std::time::Duration>,
    pub(crate) auto_compact: Option<bool>,
    /// Run a checkpoint turn before each compaction, letting the agent save what must survive and
    /// write the summary itself. Default `true`. Costs one extra model call per compaction; off
    /// falls back to the standalone summarizer, which has no tools and none of the agent's
    /// identity.
    pub(crate) compact_checkpoint: Option<bool>,
    pub(crate) context_window: Option<u64>,
    pub(crate) subagent_max_depth: Option<usize>,
}

impl Backend {
    /// Every backend, in the order the names sort, which is the order a user is shown them.
    pub(crate) const ALL: [Backend; 5] = [
        Self::AnthropicMessages,
        Self::ChatGptSubscription,
        Self::ClaudeSubscription,
        Self::OpenAiChatCompletions,
        Self::OpenAiResponses,
    ];

    /// The name an account's `backend` gives this backend.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::AnthropicMessages => "anthropic-messages",
            Self::ChatGptSubscription => "chatgpt-subscription",
            Self::ClaudeSubscription => "claude-subscription",
            Self::OpenAiChatCompletions => "openai-chat-completions",
            Self::OpenAiResponses => "openai-responses",
        }
    }

    /// The names, joined for a refusal that lists what would have been accepted.
    pub(crate) fn supported() -> String {
        Self::ALL
            .iter()
            .map(|backend| backend.name())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Whether this backend's requests carry a `thinking` field at all, i.e. whether it speaks
    /// Anthropic's Messages API. `/status` and `meka profile add` both key off this, so they agree
    /// about which profiles the [`ThinkingMode`] setting is even meaningful for.
    ///
    /// A `name().starts_with("claude")` test holds only while every Anthropic backend is *named*
    /// for Claude: the protocol name `anthropic-messages` silently falls out of it, and nothing
    /// about the failure is visible except a missing line.
    pub(crate) const fn takes_thinking(self) -> bool {
        matches!(self, Self::AnthropicMessages | Self::ClaudeSubscription)
    }

    /// Whether this backend actually sends the profile key `key`.
    ///
    /// [`Self::takes_thinking`] answers for the request *field*, which is the right question for
    /// `thinking` and `thinking_budget` and the wrong one for `redact_thinking`. That key gates the
    /// `redact-thinking-…` beta header, which only `claude-subscription` sends: the string does not
    /// appear in the Messages API backend at all. So `profile set apikey redact_thinking false`
    /// reported success and did nothing, which is exactly what `refuse_an_inert_key` exists to
    /// stop one field over.
    ///
    /// Every other key is read by every backend that has it, so the fallback is `true` rather than
    /// an enumeration that would need editing whenever a field is added; `max_request_bytes` is
    /// among those, with a default on the Anthropic backends and none on the others. The two CLI
    /// write doors and the load-time check all ask this one function.
    pub(crate) fn reads_profile_key(self, key: &str) -> bool {
        match key {
            "redact_thinking" => matches!(self, Self::ClaudeSubscription),
            "thinking" | "thinking_budget" => self.takes_thinking(),
            _ => true,
        }
    }

    /// Whether this backend actually reads the account key `key`: `client_id` and `oauth_token_url`
    /// are read only by the two OAuth backends. The sibling of [`Self::reads_profile_key`], asked
    /// by `account add` and the load-time check.
    pub(crate) fn reads_account_key(self, key: &str) -> bool {
        match key {
            "client_id" | "oauth_token_url" => {
                matches!(self, Self::ClaudeSubscription | Self::ChatGptSubscription)
            }
            _ => true,
        }
    }
}

/// Say so for every profile or account key its backend never reads.
///
/// The CLI write doors refuse or drop such a key, but a hand-written `config.toml` is another door
/// with no check at all, and AGENTS.md's profile rule is that no field may state a mismatch
/// silently: a profile on an `openai-responses` account with `thinking = "off"` read as reasoning
/// turned off while the backend ignored the key. Asked once at load, not in `resolve_profile`,
/// which runs per request. An unrecognized `backend`, and a profile whose account is missing, are
/// reported elsewhere and skipped here.
fn warn_about_inert_profile_keys(
    accounts: &std::collections::BTreeMap<String, AccountConfig>,
    profiles: &std::collections::BTreeMap<String, ProfileConfig>,
) {
    for (name, account) in accounts {
        let Ok(backend) = account.backend.parse::<Backend>() else {
            continue;
        };
        let set = [
            ("client_id", account.client_id.is_some()),
            ("oauth_token_url", account.oauth_token_url.is_some()),
        ];
        for (key, is_set) in set {
            if is_set && !backend.reads_account_key(key) {
                tracing::warn!(
                    "account '{name}' sets `{key}`, which a '{backend}' account never reads"
                );
            }
        }
    }
    for (name, profile) in profiles {
        let Ok(backend) = accounts
            .get(&profile.account)
            .and_then(|account| account.backend.parse::<Backend>().ok())
            .ok_or(())
        else {
            continue;
        };
        let set = [
            ("thinking", profile.thinking.is_some()),
            ("thinking_budget", profile.thinking_budget.is_some()),
            ("redact_thinking", profile.redact_thinking.is_some()),
        ];
        for (key, is_set) in set {
            if is_set && !backend.reads_profile_key(key) {
                tracing::warn!(
                    "profile '{name}' sets `{key}`, which its '{backend}' account never sends"
                );
            }
        }
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for Backend {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, String> {
        Self::ALL
            .iter()
            .copied()
            .find(|backend| backend.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not a backend. Supported: {}",
                    Self::supported()
                )
            })
    }
}

/// Which session a run should pick up, resolved from the mutually exclusive `--continue` and
/// `--resume` flags.
///
/// A real type rather than an `Option<String>` with a `"last"` sentinel: the sentinel makes `-c`
/// take an optional value, which would make it the only flag on the root command that can swallow
/// the following argument. `meka -c "fix the bug"` read the prompt as a session prefix and failed
/// with either a confusing lookup error or, under `--oneshot`, a claim that no prompt was given.
/// Splitting the two intents into a boolean and a value-taking flag puts both in line with every
/// other root flag and removes the ambiguity at the parser rather than guessing at it afterwards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SessionResume {
    /// `--continue`: the most recently updated session.
    Last,
    /// `--resume <SESSION>`: a specific UUID, or a leading prefix of one.
    Id(String),
}

impl SessionResume {
    /// `None` when neither flag was given. Clap enforces that they are mutually exclusive, so the
    /// order of these checks is not load-bearing.
    pub(crate) fn from_flags(resume: Option<String>, continue_last: bool) -> Option<Self> {
        if let Some(id) = resume {
            return Some(Self::Id(id));
        }
        continue_last.then_some(Self::Last)
    }
}

/// Output format for a command's stdout: `meka account usage`, `whoami`, `stats`, and a
/// `--oneshot` run. The one output-format vocabulary meka has; every command that offers a choice
/// takes `--format`, and `cli` parses the flag into this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum OutputFormat {
    /// Human-readable text.
    #[default]
    Plain,
    /// JSON (stable shape, for scripts).
    Json,
}

impl OutputFormat {
    /// Every format, in the order the names sort.
    pub(crate) const ALL: [OutputFormat; 2] = [Self::Json, Self::Plain];

    /// The one spelling `--format` takes.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
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

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|format| format.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not an output format. Supported: {}",
                    Self::supported()
                )
            })
    }
}

/// What the command line contributes to resolution: every root flag that overrides or requests
/// something. Built by `cli` from the clap struct, so `config` never has to know what a flag is
/// called.
#[derive(Debug, Default, Clone)]
pub(crate) struct CliOverrides {
    pub(crate) profile: Option<String>,
    /// `--format` on the run, read by a `--oneshot` turn.
    pub(crate) output_format: OutputFormat,
    pub(crate) eager_load_tools: Vec<String>,
    pub(crate) instructions: Option<String>,
    pub(crate) permission: Option<Permission>,
    pub(crate) sandbox_backend: Option<SandboxBackend>,
    pub(crate) writable_roots: Vec<std::path::PathBuf>,
    pub(crate) no_stream: bool,
    pub(crate) render_mode: Option<RenderMode>,
    pub(crate) resume: Option<String>,
    pub(crate) continue_last: bool,
    pub(crate) prompt: Option<String>,
    pub(crate) oneshot: bool,
}

/// What one invocation asked for: the prompt, the session to resume, and the overrides the command
/// line carried. Everything else on [`ResolvedConfig`] is a setting.
#[derive(Debug, Default)]
pub(crate) struct RunRequest {
    pub(crate) prompt: Option<String>,
    pub(crate) oneshot: bool,
    /// What a `--oneshot` run writes to stdout: the answer as text, or one JSON object describing
    /// the turn. `plain` unless `--format json` was passed.
    pub(crate) output_format: OutputFormat,
    pub(crate) session_resume: Option<SessionResume>,
    /// The permission level the user asked for on this run with `--permission` or
    /// `MEKA_PERMISSION`, when the request was honored.
    ///
    /// Distinct from [`ResolvedConfig::permission`], which is the resolved level and is set
    /// whether or not anyone asked for it. Resuming needs the two apart: a session runs at the
    /// level it recorded, and only an explicit request overrides that.
    pub(crate) requested_permission: Option<Permission>,
    /// The profile the user named on the command line with `--profile`, if any.
    ///
    /// Distinct from [`ResolvedConfig::default_profile`], which is the *resolved* default and is
    /// set whether or not anyone asked for it. Resuming needs to tell the two apart: a session
    /// runs on the profile it recorded, and only an explicit request rewrites that.
    pub(crate) requested_profile: Option<String>,
    /// Extra workspace roots from `--writable-root`, joined with the cwd (and, under ACP, the
    /// client's folders) to form the write boundary at `workspace` permission.
    ///
    /// CLI-only by design. Which folders a run may write is a per-run scope like the working
    /// directory, not a preference worth persisting, so there is deliberately no `config.toml`
    /// key and no env tier for it.
    pub(crate) writable_roots: Vec<std::path::PathBuf>,
    /// CLI-flag override for `[serve].bind` (`meka serve --bind ...`). Wins over the config file
    /// when set.
    pub(crate) serve_bind_override: Option<String>,
    /// `--instructions` verbatim; the host resolves it against the persistent tiers at startup
    /// through [`crate::instructions::resolve`].
    pub(crate) instructions: Option<String>,
}

/// One configured profile as a listing shows it: its name, the account it bills, that account's
/// `backend` as written, and the model when the profile names one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProfileSummary {
    pub(crate) name: String,
    pub(crate) account: String,
    pub(crate) backend: String,
    pub(crate) model: Option<String>,
}

/// Merged + validated runtime view of [`ConfigFile`], CLI flags, and env vars. This is what the
/// rest of the binary reads; `ConfigFile` is for deserialization only. Resolution is
/// [`ResolvedConfig::resolve`].
#[derive(Debug)]
pub(crate) struct ResolvedConfig {
    /// Backend of the selected profile's account; `None` when no profile could be selected or its
    /// account's backend is not one meka knows, in which case [`Self::provider_error`] says so.
    pub(crate) backend: Option<Backend>,
    /// Name of the selected profile.
    ///
    /// The process default only. A session records the profile it runs with and resolves that one
    /// out of [`Self::profiles`] instead, which is why the maps below are retained rather than
    /// dropped once this name is picked.
    pub(crate) default_profile: Option<String>,
    /// The profile `default_profile` picks, ignoring `--profile`.
    ///
    /// Only the migration ledger wants this. It stamps a profile onto sessions that predate meka
    /// recording one, which is a guess it makes once and cannot revisit, so the answer must not
    /// depend on a flag the first invocation after an upgrade happened to carry: `meka --profile
    /// side -p "..."` would otherwise move a whole history onto `side` because of one run.
    /// Everything else wants [`Self::default_profile`], where the flag is the point.
    pub(crate) configured_default_profile: Option<String>,
    /// Every configured account, by name. A profile's credential is stored under its account.
    pub(crate) accounts: std::collections::BTreeMap<String, AccountConfig>,
    /// Every configured profile, by name.
    ///
    /// Kept so a provider can be built for a profile that is not the process default. Selection
    /// picks one name at startup, but a session names its own, and resolving that one needs the
    /// profile still to be here.
    pub(crate) profiles: std::collections::BTreeMap<String, ProfileConfig>,
    /// Set when profile selection failed (no profiles, ambiguous, an unknown name, or a profile
    /// whose account is missing); surfaced by [`Self::validate`] with guidance.
    pub(crate) provider_error: Option<String>,
    /// `config.toml` failed to read or parse. Raised by [`Self::validate`] before anything else,
    /// since every other field is a default that silently ignores what the user wrote.
    pub(crate) config_error: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) permission: Permission,
    pub(crate) enabled_permissions: EnabledPermissions,
    /// What a new session's approvals switch starts as. A resumed session brings its own.
    pub(crate) approvals: bool,
    pub(crate) streaming: bool,
    pub(crate) newline_before_prompt: bool,
    pub(crate) newline_after_prompt: bool,
    pub(crate) show_session_id_on_create: bool,
    pub(crate) show_session_id_on_exit: bool,
    pub(crate) show_path_in_prompt: bool,
    pub(crate) show_context_in_prompt: bool,
    pub(crate) show_token_usage: bool,
    pub(crate) resume_show_recent: Option<usize>,
    pub(crate) web_client: WebClientConfig,
    pub(crate) sandbox: bool,
    /// `[shell].sandbox_backend` (or its flag and variable), before any auto-pick. Linux-only;
    /// the host resolves and probes it at startup ([`crate::sandbox::resolve_backend`]) because
    /// the pick depends on what the machine has.
    pub(crate) sandbox_backend: Option<SandboxBackend>,
    pub(crate) render_mode: RenderMode,
    pub(crate) tool_params: ToolParams,
    /// Resolved `[display].max_width`. `None`, the default, follows the terminal.
    pub(crate) max_width: Option<usize>,
    pub(crate) context_messages: Option<usize>,
    /// Resolved `[session].retention`. `None`, the default, disables startup cleanup.
    pub(crate) retention: Option<std::time::Duration>,
    pub(crate) thinking: crate::config::ThinkingMode,
    /// The selected profile's resolved thinking budget, the sibling of [`Self::thinking`] and
    /// [`Self::max_output_tokens`]. Answers "what would a brand-new session get", which is exactly
    /// what [`ResolvedConfig::validate_default_profile`] needs to check the pair against each
    /// other. Not the seed: that is [`Self::default_thinking_budget`].
    pub(crate) thinking_budget: u64,
    pub(crate) thinking_show_content: bool,
    pub(crate) auto_compact: bool,
    pub(crate) compact_checkpoint: bool,
    /// `[session].context_window` verbatim, *before* any profile is applied to it.
    ///
    /// Unresolved on purpose, and the one flat field here that must stay that way. It is the seed
    /// [`crate::provider::ProviderRegistry`] hands `resolve_profile` for every profile it builds,
    /// so a value that had already been through the *active* profile would become that profile's
    /// window for every other profile that states none: a session on a small local model would
    /// gauge itself against the big cloud default and never compact, and the reverse pairing would
    /// compact every turn. Resolving it here and again per profile is the same mistake twice.
    pub(crate) session_context_window: Option<u64>,
    /// `[thinking].budget` verbatim, unresolved for the reason [`Self::session_context_window`]
    /// gives at length: it is the seed every profile falls back to, so a value already resolved
    /// through the *active* profile would become that profile's budget for every profile stating
    /// none.
    pub(crate) default_thinking_budget: Option<u64>,
    /// Maximum sub-agent recursion depth. `1` lets the root agent spawn but denies its sub-agents
    /// the same; `0` disables `agent_spawn` entirely. Seeds the root `AgentSpawnTool`'s depth
    /// budget in `main.rs`.
    pub(crate) subagent_max_depth: usize,
    /// Whether the selected profile's model accepts image input (the ACP `image` prompt
    /// capability). Resolved from `[profiles.<name>].vision`, defaulting to `true`.
    pub(crate) vision: bool,
    /// Per-request output (completion) token cap from `[profiles.<name>].max_output_tokens`.
    /// `None` leaves each backend's built-in default in place.
    pub(crate) max_output_tokens: Option<u64>,
    pub(crate) mcp_servers: Vec<McpServerConfig>,
    /// Parsed [`Permission`] from `[mcp].default_permission`, carried so per-turn tool-permission
    /// resolution in `src/mcp.rs` doesn't have to re-read the config file. `None` means "no
    /// `[mcp]` default configured"; resolution falls through to the hardcoded `Unrestricted`.
    pub(crate) mcp_default_permission: Option<Permission>,
    /// Non-secret projection of the configured profiles, for `GET /v1/profiles`.
    ///
    /// A projection rather than the profiles themselves, so a field added to [`ProfileConfig`]
    /// later cannot start appearing on the API by accident. Credentials were never in here to
    /// begin with; they live in the database keyed by account name.
    pub(crate) profile_summaries: Vec<ProfileSummary>,
    /// Whether the `skill_read` / `skill_search` tools are registered and the `[Skills]` index
    /// rendered. Defaults to `true`; see [`SkillsConfig`].
    pub(crate) skills_enabled: bool,
    /// Whether `skill_write` / `skill_delete` are additionally registered. Defaults to `false`;
    /// see [`SkillsConfig::agent_managed`].
    pub(crate) skills_agent_managed: bool,
    /// Read-only skill roots scanned in addition to meka's own; see [`SkillsConfig::extra_paths`].
    /// Already tilde-expanded, and empty unless configured.
    ///
    /// Read through [`Self::skill_roots`] rather than directly, so the "native first, then these"
    /// order has one owner.
    pub(crate) skills_extra_paths: Vec<std::path::PathBuf>,
    /// Whether the `memory_*` tools are registered and the `[Memory]` index rendered. Defaults to
    /// `true`; see [`MemoryConfig`].
    pub(crate) memory_enabled: bool,
    /// `[schedule]` with defaults filled. Resolved here rather than at the call site because both
    /// the REPL and `meka serve` start a scheduler and would otherwise each re-derive it.
    pub(crate) schedule: ResolvedScheduleConfig,
    /// `[background]` with defaults filled. Off unless the installation asks for it; see
    /// [`BackgroundConfig`].
    pub(crate) background: ResolvedBackgroundConfig,
    /// `[subagents]` with defaults filled. Governs what a spawned sub-agent does *not* inherit;
    /// see [`SubagentsConfig`].
    pub(crate) subagents: ResolvedSubagentsConfig,
    pub(crate) builtin_allowed_tools: Option<Vec<String>>,
    pub(crate) builtin_disabled_tools: Vec<String>,
    pub(crate) builtin_tool_permissions: HashMap<String, Permission>,
    pub(crate) input_style: nu_ansi_term::Style,
    /// First-turn await cap for still-connecting MCP servers.
    pub(crate) mcp_grace: std::time::Duration,
    /// Per-server connect+initialize timeout.
    pub(crate) mcp_connect_timeout: std::time::Duration,
    /// How many stdio servers the startup connector spawns at once.
    pub(crate) mcp_stdio_concurrency: usize,
    /// How many HTTP servers the startup connector connects at once.
    pub(crate) mcp_http_concurrency: usize,
    /// Raw `[serve]` section. Resolved (defaults filled, env vars substituted, token files read)
    /// at `meka serve` startup by [`crate::host::http::config::ResolvedServeConfig::resolve`], not
    /// here, so [`ResolvedConfig::resolve`] can stay infallible.
    pub(crate) serve: Option<ServeConfig>,
    /// What this invocation asked for, as distinct from how meka is configured.
    pub(crate) request: RunRequest,
}

/// Deserialize an optional humantime duration string (e.g. `"24h"`, `"5m"`, `"30s"`).
/// `serde(deserialize_with)` on `Option<Duration>` requires this wrapper because `humantime_serde`
/// only provides a `deserialize` for `Duration`, not `Option<Duration>`.
pub(crate) fn deserialize_optional_duration<'de, D>(
    deserializer: D,
) -> Result<Option<std::time::Duration>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<String> = Option::deserialize(deserializer)?;
    match opt {
        None => Ok(None),
        Some(s) => humantime_serde::re::humantime::parse_duration(&s)
            .map(Some)
            .map_err(serde::de::Error::custom),
    }
}

/// Replace `${VAR}` occurrences with the corresponding environment variable. `$$` is an escape
/// for a literal `$`. Unknown variables return an error rather than expanding to empty; silent
/// expansion to empty would produce a runtime auth-bypass-shaped configuration.
///
/// Returns the substituted string plus a boolean indicating whether at least one `${VAR}` expansion
/// happened; callers use this to classify token provenance (`TokenSource::EnvVar` vs
/// `TokenSource::Inline`) for the startup warning.
pub(crate) fn substitute_env(input: &str) -> Result<(String, bool), String> {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    let mut substituted = false;
    while let Some(c) = chars.next() {
        if c != '$' {
            out.push(c);
            continue;
        }
        match chars.peek() {
            Some('$') => {
                chars.next();
                out.push('$');
            }
            Some('{') => {
                chars.next();
                let mut name = String::new();
                let mut closed = false;
                for inner in chars.by_ref() {
                    if inner == '}' {
                        closed = true;
                        break;
                    }
                    name.push(inner);
                }
                if !closed {
                    return Err("unclosed `${` in a substituted config value".into());
                }
                let value = std::env::var(&name)
                    .map_err(|_| format!("env var `{name}` referenced in config is unset"))?;
                out.push_str(&value);
                substituted = true;
            }
            _ => out.push('$'),
        }
    }
    Ok((out, substituted))
}

/// Default input style: bold, white-ish foreground, slate-blue background. Uses truecolor RGB (not
/// palette indices or named colors) so the visual is consistent across terminals that remap the
/// standard 16 colors to match their theme.
pub(crate) fn default_input_style() -> nu_ansi_term::Style {
    use nu_ansi_term::{Color, Style};
    Style::new()
        .bold()
        .fg(Color::Rgb(240, 240, 240))
        .on(Color::Rgb(55, 75, 110))
}

/// Parse a `[display].input_style` value. `"default"` (or unset) yields [`default_input_style`];
/// `"none"` yields no styling; simple keywords pick a single color or attribute. Unknown keywords
/// warn and fall back to the default so a typo doesn't lose the session to a panic.
pub(crate) fn parse_input_style(raw: &str) -> nu_ansi_term::Style {
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
                "ignoring [display].input_style '{other}'; supported: default, none, reverse, \
                 bold, dim, italic, underline, a color name"
            );
            default_input_style()
        }
    }
}

/// Resolve `[skills] extra_paths` into the read-only roots discovery will scan.
///
/// A *function* rather than a chain inside [`ResolvedConfig::resolve`], because each of the four
/// things it drops has its own reason and each needs saying out loud:
///
/// - An empty entry, because [`expand_user_path`] maps `""` to the home directory. That is right
///   for `/cd` (type `cd` and you go home) and catastrophic here: `$HOME` would become a skills
///   root and warn once per subdirectory on every rediscovery.
/// - An entry whose `~` cannot be expanded, rather than treating the tilde as a directory name.
/// - A repeat of an earlier entry, which discovery would otherwise walk twice and then report every
///   skill in it as shadowed *by itself*, a warning naming one path twice, which an operator can
///   neither act on nor dismiss.
/// - meka's own root, for the same reason. It is always scanned first; naming it again adds a
///   second pass and the same self-shadowing report.
///
/// Compared as written after expansion, not canonicalized: resolving symlinks would touch the
/// filesystem, and the promise that a configured-but-absent root leaves no trace is worth more than
/// catching two spellings of one directory. Discovery's own first-wins rule handles that case.
fn resolve_skills_extra_paths(raw: &[String], native: Option<&Path>) -> Vec<PathBuf> {
    let mut resolved: Vec<PathBuf> = Vec::new();
    for entry in raw {
        if entry.trim().is_empty() {
            tracing::warn!("[skills] extra_paths contains an empty entry; ignoring it");
            continue;
        }
        let Some(path) = expand_user_path(entry) else {
            tracing::warn!(
                "ignoring [skills] extra_paths entry '{entry}': no home directory to expand `~` \
                 into"
            );
            continue;
        };
        if Some(path.as_path()) == native {
            tracing::warn!(
                "[skills] extra_paths entry '{entry}' is meka's own skills directory, which is \
                 always scanned first; ignoring it"
            );
            continue;
        }
        if resolved.contains(&path) {
            tracing::warn!(
                "[skills] extra_paths lists '{entry}' more than once; ignoring the repeat"
            );
            continue;
        }
        resolved.push(path);
    }
    resolved
}

/// Exclusive cross-process lock over `config.toml`, held for a whole read-modify-write.
///
/// Every editor of the file reads it, mutates a `toml_edit` document, and writes the whole thing
/// back. Two of those interleaving means the second write is computed from a snapshot taken before
/// the first, so the first is silently discarded; `write_file_atomic` makes each *write* atomic,
/// which does nothing for a lost update. This is not a hypothetical race between two humans running
/// CLI commands: an ordinary launch races `meka mcp add`, because `device_id::persist` runs from
/// `ProviderRegistry::device_id_for` the first time a `claude-subscription` profile that states
/// none is resolved.
///
/// Taken on the config directory wherever the platform allows it, so a tree people keep under
/// version control gains no lock file; [`open_config_lock_target`] has why the directory rather
/// than the file, and the platform that cannot do either.
///
/// Reentrant within a thread. `flock` is associated with the open file description, so a second
/// `open` in the same process conflicts with the first exactly as another process would, and a
/// nested `lock_config_file()` would self-deadlock; the depth counter is what keeps the outermost
/// acquisition the only one that touches the file.
pub(crate) enum ConfigFileLock {
    /// The outermost acquisition on this thread; owns the `flock` and releases it on drop.
    Held { _lock: PathLock },
    /// A nested acquisition. The outer guard still holds the file; this one only keeps the depth
    /// accurate so the outer release happens at the right moment.
    Reentrant,
}

thread_local! {
    /// How many [`ConfigFileLock`]s this thread currently holds.
    ///
    /// Thread-local while the `flock` underneath it is per open file description, which only holds
    /// because no `ConfigFileLock` is held across an `.await` on a task that can migrate. A task
    /// resuming on another worker thread would break both halves at once: a nested acquisition
    /// there would see depth 0 and block on an `flock` this process already holds, and the guard
    /// would decrement the wrong thread's counter, leaving the original stuck above zero so every
    /// later acquisition on it silently took no lock at all.
    static CONFIG_LOCK_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl Drop for ConfigFileLock {
    fn drop(&mut self) {
        CONFIG_LOCK_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// Take [`ConfigFileLock`]. Blocks until the holder releases it, which is right for a CLI edit:
/// the alternative is failing a `meka mcp add` because a background `device_id` write happened to
/// be in flight.
pub(crate) fn lock_config_file() -> std::io::Result<ConfigFileLock> {
    if CONFIG_LOCK_DEPTH.with(|depth| depth.get()) > 0 {
        CONFIG_LOCK_DEPTH.with(|depth| depth.set(depth.get() + 1));
        return Ok(ConfigFileLock::Reentrant);
    }

    let directory = meka_config_dir()
        .ok_or_else(|| std::io::Error::other("failed to determine the config directory"))?;
    // 0700 straight from `mkdir(2)`, as [`write_file_atomic`] does. A pre-existing directory is
    // left alone here and tightened by that function on the write this lock is being taken for,
    // which is also the only place that knows whether a symlink took the write out of meka's tree.
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .mode(0o700)
            .recursive(true)
            .create(&directory)?;
    }
    #[cfg(not(unix))]
    std::fs::create_dir_all(&directory)?;

    let lock = lock_path(&directory, open_config_lock_target)?;
    CONFIG_LOCK_DEPTH.with(|depth| depth.set(1));
    Ok(ConfigFileLock::Held { _lock: lock })
}

/// Load `config.toml`, returning the parsed file and any error that stopped it from parsing.
///
/// Warning and falling back to `ConfigFile::default()` is the worst of the options: one mistyped
/// key silently reconfigures the whole agent, running it with no provider profiles, no MCP servers
/// and default permissions, off a single line the user can easily scroll past. The error is carried
/// instead and raised by [`ResolvedConfig::validate`], alongside the profile-selection failures, so
/// [`ResolvedConfig::resolve`] can stay infallible.
///
/// The line between failing and continuing is whether a command *consults* the parsed config. One
/// that does can only answer from empty defaults, which reads as fact: `meka profile list`
/// printing "No profiles." over a file full of them. Those callers use
/// [`load_config_file_or_err`] / [`ResolvedConfig::require_readable_config`].
///
/// Commands that instead edit the raw document through `toml_edit` are unaffected by an unknown key
/// and are how a broken config gets fixed from the CLI, so they run anyway: `meka mcp add` /
/// `remove` / `enable` / `disable` note the failure via
/// [`ResolvedConfig::warn_if_config_unreadable`], and `meka account remove` and `meka profile
/// remove` never load the parsed config at all. Each re-reads the file itself, and each must fail
/// on a *read* error rather than substitute an empty document, or the write-back truncates what it
/// couldn't read.
pub(crate) fn load_config_file() -> (ConfigFile, Option<String>) {
    let Some(path) = config_file_path() else {
        return (ConfigFile::default(), None);
    };

    match std::fs::read_to_string(&path) {
        Ok(contents) => match toml::from_str(&contents) {
            Ok(config) => (config, None),
            Err(error) => (
                ConfigFile::default(),
                Some(format!("failed to parse {}: {}", path.display(), error)),
            ),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (ConfigFile::default(), None),
        Err(error) => (
            ConfigFile::default(),
            Some(format!("failed to read {}: {}", path.display(), error)),
        ),
    }
}

/// [`load_config_file`] for callers that read profiles but have no `validate()` to route the error
/// through: the `meka account` and `meka profile` subcommands.
///
/// Fails rather than falling back to an empty `ConfigFile`, because for these callers "empty" is
/// indistinguishable from "your profiles are gone". `meka profile add <existing>` is the sharp
/// edge: its duplicate guard is the parsed map, so an empty one lets the add through to
/// `upsert_profile_document`, which replaces the profile's table wholesale and drops every field
/// the flags did not set.
pub(crate) fn load_config_file_or_err() -> crate::error::Result<ConfigFile> {
    let (config_file, error) = load_config_file();
    match error {
        Some(error) => Err(crate::error::MekaError::Config(error)),
        None => Ok(config_file),
    }
}

/// Resolve the runtime permission level and the set of enabled levels from the layered config
/// sources. CLI > env > config file > built-in defaults. A level meka does not have never reaches
/// here, because [`PermissionsConfig`] is typed and serde refuses it at parse; out-of-set overrides
/// (e.g. `--permission unrestricted` when it is disabled) warn and clamp to the configured default
/// rather than refusing to start, mirroring the `[tools.tool_permissions]` warn-and-skip pattern.
///
/// The third element is the level the user asked for on this run and got, or `None` when the level
/// came from the config file or the built-in default. Resuming needs the distinction: a session
/// runs at the level it recorded, and only an explicit request overrides that. The level
/// `config.toml` alone gives a new session, for what is written into rows and archives: the store's
/// ledger and an import. Unlike [`ResolvedConfig::resolve`] it reads no CLI flag and no environment
/// variable, because those belong to one run and the row outlives it.
pub(crate) fn default_permission_on_disk() -> crate::error::Result<Permission> {
    let config_file = load_config_file_or_err()?;
    let permissions = config_file.permissions.unwrap_or_default();
    // No CLI or environment tier: what this answers is written into rows and archives that every
    // later process reads, so only the file's own statement may decide it.
    let (permission, _enabled, _requested) = resolve_permission(
        None,
        None,
        permissions.default,
        permissions.enabled.as_deref(),
    );
    Ok(permission)
}

fn resolve_permission(
    cli_permission: Option<Permission>,
    env_permission: Option<&str>,
    file_default: Option<Permission>,
    file_enabled: Option<&[Permission]>,
) -> (Permission, EnabledPermissions, Option<Permission>) {
    let enabled = match file_enabled {
        Some(list) => match EnabledPermissions::from_levels(list.iter().copied()) {
            Some(set) => set,
            None => {
                // `{read}`, not `DEFAULT`. The user wrote a list, and an empty one asks for
                // nothing; `DEFAULT` would answer that by handing back *four* levels including
                // `unrestricted`. Falling back to the narrowest useful set keeps an empty list
                // from widening authority.
                tracing::warn!(
                    "[permissions].enabled names no level; falling back to `read` alone"
                );
                EnabledPermissions::from_levels([Permission::Read])
                    .unwrap_or(EnabledPermissions::DEFAULT)
            }
        },
        None => EnabledPermissions::DEFAULT,
    };

    let resolved_default = match file_default {
        Some(level) if enabled.is_enabled(level) => level,
        Some(level) => {
            tracing::warn!(
                "[permissions].default = '{level}' is not in [permissions].enabled; falling back"
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
        Ok(level) => Some(level),
        Err(error) => {
            tracing::warn!("ignoring MEKA_PERMISSION='{raw}': {error}");
            None
        }
    });

    let raw_choice = cli_permission.or(env_override);
    // A request that had to be clamped is not a request that was honored, so it does not suppress a
    // resumed session's recorded level either.
    let requested = match raw_choice {
        Some(level) if enabled.is_enabled(level) => Some(level),
        Some(level) => {
            tracing::warn!(
                "requested start level '{level}' is not in [permissions].enabled; using \
                 '{resolved_default}'"
            );
            None
        }
        None => None,
    };

    (requested.unwrap_or(resolved_default), enabled, requested)
}

/// `MEKA_SANDBOX_BACKEND` overrides `[shell].sandbox_backend` for non-interactive / containerized
/// runs (the `mekabox` wrapper sets it to pin Landlock and silence the auto-resolve warning when
/// it mounts the host config read-only). Takes the one spelling the file takes; an unrecognized
/// value is warned about and ignored. Like the config field, this only affects the resolved
/// backend on Linux.
fn sandbox_backend_override() -> Option<SandboxBackend> {
    parse_sandbox_backend_override(&std::env::var("MEKA_SANDBOX_BACKEND").ok()?)
}

/// Parse a `MEKA_SANDBOX_BACKEND` value (trimmed). Empty or unrecognized values yield `None`;
/// unrecognized ones also warn. Split from [`sandbox_backend_override`] so the parsing is
/// unit-testable without mutating process env.
fn parse_sandbox_backend_override(value: &str) -> Option<SandboxBackend> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Env tolerates a bad value (warn and ignore); the CLI path uses `FromStr` directly and errors.
    match trimmed.parse() {
        Ok(backend) => Some(backend),
        Err(message) => {
            tracing::warn!("ignoring MEKA_SANDBOX_BACKEND: {message}");
            None
        }
    }
}

/// `MEKA_RENDER_MODE` overrides `[display].render_mode`, with the same tolerance as
/// [`sandbox_backend_override`]: a value meka does not have is warned about and ignored.
fn render_mode_override() -> Option<RenderMode> {
    parse_render_mode_override(&std::env::var("MEKA_RENDER_MODE").ok()?)
}

/// Parse a `MEKA_RENDER_MODE` value (trimmed). Empty or unrecognized values yield `None`;
/// unrecognized ones also warn, so a retired spelling does not silently fall to the default.
fn parse_render_mode_override(value: &str) -> Option<RenderMode> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.parse() {
        Ok(mode) => Some(mode),
        Err(message) => {
            tracing::warn!("ignoring MEKA_RENDER_MODE: {message}");
            None
        }
    }
}

/// Merge `--eager-load-tool SERVER:TOOL` CLI values into the matching server's
/// [`McpServerConfig::eager_load_tools`] list. A malformed entry or an unknown server name warns
/// and is skipped. Appends to the configured list, without repeating an entry already there.
fn apply_cli_eager_load_overrides(raw_pairs: &[String], servers: &mut [McpServerConfig]) {
    for raw in raw_pairs {
        let (server_name, tool_name) = match raw.split_once(':') {
            Some((server, tool)) => {
                let server = server.trim();
                let tool = tool.trim();
                if server.is_empty() || tool.is_empty() {
                    tracing::warn!("ignoring --eager-load-tool '{raw}': expected SERVER:TOOL");
                    continue;
                }
                (server, tool)
            }
            None => {
                tracing::warn!("ignoring --eager-load-tool '{raw}': expected SERVER:TOOL");
                continue;
            }
        };
        match servers.iter_mut().find(|server| server.name == server_name) {
            Some(server) => {
                let list = server.eager_load_tools.get_or_insert_with(Vec::new);
                if !list.iter().any(|existing| existing == tool_name) {
                    list.push(tool_name.to_string());
                }
            }
            None => {
                let refusal = crate::text::unknown_name(
                    "MCP server",
                    server_name,
                    servers.iter().map(|server| server.name.as_str()),
                );
                tracing::warn!("--eager-load-tool '{raw}': {refusal}");
            }
        }
    }
}

impl ResolvedConfig {
    pub(crate) fn resolve(overrides: CliOverrides) -> Self {
        let (config_file, config_error) = load_config_file();
        let accounts = config_file.accounts;
        let profiles = config_file.profiles;
        // `--profile`, else `default_profile`, else the sole profile. A miss is a deferred error
        // `validate()` raises, so `resolve` stays infallible.
        let (profile_request, requested_active) = match overrides.profile.clone() {
            Some(name) => (ProfileRequest::Flag, Some(name)),
            None => (
                ProfileRequest::DefaultProfile,
                config_file.default_profile.clone(),
            ),
        };
        let (default_profile, mut provider_error) =
            select_profile(requested_active, profile_request, &profiles);
        // Resolved a second time without the flag, for the ledger. Its error is discarded: a
        // default nothing can pick is already reported by the resolution above, and the migration's
        // own fallbacks handle having no answer.
        let (configured_default_profile, _) = select_profile(
            config_file.default_profile.clone(),
            ProfileRequest::DefaultProfile,
            &profiles,
        );
        let active = default_profile.as_ref().and_then(|name| profiles.get(name));
        let file_display = config_file.display.unwrap_or_default();
        let file_web = config_file.web.unwrap_or_default();
        let file_shell = config_file.shell.unwrap_or_default();
        let file_session = config_file.session.unwrap_or_default();
        let file_thinking = config_file.thinking.unwrap_or_default();
        let file_tools = config_file.tools.unwrap_or_default();
        let file_skills = config_file.skills.unwrap_or_default();
        let skills_enabled = file_skills.enabled.unwrap_or(true);
        let skills_agent_managed = file_skills.agent_managed.unwrap_or(false);
        let skills_extra_paths = resolve_skills_extra_paths(
            file_skills.extra_paths.as_deref().unwrap_or_default(),
            crate::paths::skills_dir().as_deref(),
        );
        // The two keys read as independent but are not: `enabled = false` registers no skill tools
        // at all, so the authoring pair never appears however this is set. Saying so beats leaving
        // someone to conclude that `agent_managed` is broken.
        if skills_agent_managed && !skills_enabled {
            tracing::warn!(
                "[skills] agent_managed is on but enabled is false, so no skill tools are \
                 registered at all; set enabled = true to let the agent author skills"
            );
        }
        let memory_enabled = config_file
            .memory
            .unwrap_or_default()
            .enabled
            .unwrap_or(true);
        let background = ResolvedBackgroundConfig::resolve(config_file.background);
        let subagents = ResolvedSubagentsConfig::resolve(config_file.subagents);
        let file_mcp = config_file.mcp.unwrap_or_default();
        let mut mcp_servers = file_mcp.servers.unwrap_or_default();
        let mcp_default_required = file_mcp.default_required.unwrap_or(false);
        let mcp_grace = file_mcp.grace.unwrap_or(DEFAULT_MCP_GRACE);
        let mcp_connect_timeout = file_mcp
            .connect_timeout
            .unwrap_or(DEFAULT_MCP_CONNECT_TIMEOUT);
        let mcp_stdio_concurrency = file_mcp
            .stdio_concurrency
            .unwrap_or(DEFAULT_MCP_STDIO_CONCURRENCY);
        let mcp_http_concurrency = file_mcp
            .http_concurrency
            .unwrap_or(DEFAULT_MCP_HTTP_CONCURRENCY);
        let mcp_default_permission = file_mcp.default_permission;
        apply_cli_eager_load_overrides(&overrides.eager_load_tools, &mut mcp_servers);
        // Settle `required` here so no later consumer has to remember that `None` means "inherit
        // `default_required`". Everything downstream reads a plain `Some(bool)`, and the default
        // deliberately isn't carried on `ResolvedConfig`: a second copy of the same answer is one
        // a future reader could gate on, believing it still decides turns on its own. It doesn't.
        for server in &mut mcp_servers {
            server.required = Some(server.required.unwrap_or(mcp_default_required));
        }

        let profile_summaries: Vec<ProfileSummary> = profiles
            .iter()
            .map(|(name, profile)| ProfileSummary {
                name: name.clone(),
                account: profile.account.clone(),
                backend: accounts
                    .get(&profile.account)
                    .map(|account| account.backend.clone())
                    .unwrap_or_default(),
                model: profile.model.clone(),
            })
            .collect();

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
                (!name.is_empty()).then_some((name, level))
            })
            .collect();

        // The selected profile's settings, through the same [`resolve_profile`] every other profile
        // goes through; nothing on the command line rewrites a field inside it. The flat fields
        // below answer "what would a brand-new session get", and a session naming another profile
        // resolves it again by name. The account's `device_id` verbatim, never a freshly seeded
        // one: seeding writes `config.toml`, and belongs in [`crate::provider::ProviderRegistry`],
        // where it happens once.
        let default_thinking_budget = file_thinking.budget;
        let active_settings = match active.map(|profile| {
            let account = account_for(
                default_profile.as_deref().unwrap_or("?"),
                profile,
                &accounts,
            )?;
            resolve_profile(
                profile,
                account,
                file_session.context_window,
                default_thinking_budget,
                account.device_id.clone().unwrap_or_default(),
            )
        }) {
            Some(Ok(settings)) => Some(settings),
            // Reported where a selection failure is, and only when a run needs a provider: a
            // subcommand that never builds one must not fail over a typo in an unrelated field.
            Some(Err(error)) => {
                provider_error.get_or_insert(error);
                None
            }
            None => None,
        };
        let backend = active_settings.as_ref().map(|settings| settings.backend);
        let model = active_settings
            .as_ref()
            .and_then(|settings| settings.model.clone());

        let file_permissions = config_file.permissions.unwrap_or_default();
        let (permission, enabled_permissions, requested_permission) = resolve_permission(
            overrides.permission,
            std::env::var("MEKA_PERMISSION").ok().as_deref(),
            file_permissions.default,
            file_permissions.enabled.as_deref(),
        );

        // After `resolve_permission`, because the fire-time gate check filters a row's recorded
        // level by the enabled set, which is the one thing an operator can narrow that a row
        // cannot exceed.
        let schedule = ResolvedScheduleConfig::resolve(config_file.schedule, enabled_permissions);

        // `--sandbox-backend` > `MEKA_SANDBOX_BACKEND` > `[shell].sandbox_backend`. Not probed
        // here: the host picks and probes at startup, and only when a shell can run.
        let sandbox_backend = overrides
            .sandbox_backend
            .or_else(sandbox_backend_override)
            .or(file_shell.sandbox_backend);

        // A root that does not resolve contributes nothing to the write boundary, because
        // `writable_roots` drops what it cannot canonicalize. Silently, until a write is refused
        // with a boundary the user believed included this path. The path is kept regardless of the
        // warning: a build directory that does not exist yet is a legitimate root, and the boundary
        // is recomputed on every write rather than frozen here.
        for root in &overrides.writable_roots {
            if let Err(error) = std::fs::canonicalize(root) {
                tracing::warn!(
                    "--writable-root {root} does not resolve ({error}); it grants nothing until \
                     it exists",
                    root = root.display()
                );
            }
        }

        Self {
            backend,
            default_profile,
            configured_default_profile,
            accounts,
            profiles,
            provider_error,
            config_error,
            model,
            permission,
            enabled_permissions,
            approvals: file_permissions.approvals.unwrap_or(false),
            streaming: streaming_enabled(overrides.no_stream, file_display.stream),
            newline_before_prompt: file_display.newline_before_prompt.unwrap_or(true),
            newline_after_prompt: file_display.newline_after_prompt.unwrap_or(true),
            show_session_id_on_create: file_display.show_session_id_on_create.unwrap_or(false),
            show_session_id_on_exit: file_display.show_session_id_on_exit.unwrap_or(true),
            show_token_usage: file_display.show_token_usage.unwrap_or(false),
            resume_show_recent: file_display.resume_show_recent,
            show_path_in_prompt: file_display.show_path_in_prompt.unwrap_or(true),
            show_context_in_prompt: file_display.show_context_in_prompt.unwrap_or(false),
            web_client: WebClientConfig::from_file(&file_web),
            sandbox: file_shell.sandbox.unwrap_or(true),
            sandbox_backend,
            render_mode: overrides
                .render_mode
                .or_else(render_mode_override)
                .or(file_display.render_mode)
                .unwrap_or_default(),
            tool_params: file_display.tool_params.unwrap_or_default(),
            max_width: file_display.max_width.map(clamp_max_width),
            context_messages: file_session
                .context_messages
                .or(Some(DEFAULT_CONTEXT_MESSAGES)),
            retention: file_session.retention,
            thinking: active_settings
                .as_ref()
                .map(|settings| settings.thinking)
                .unwrap_or_default(),
            thinking_budget: active_settings
                .as_ref()
                .map(|settings| settings.thinking_budget)
                .or(default_thinking_budget)
                .unwrap_or(DEFAULT_THINKING_BUDGET_TOKENS),
            thinking_show_content: file_thinking.show_content.unwrap_or(false),
            auto_compact: file_session.auto_compact.unwrap_or(true),
            compact_checkpoint: file_session.compact_checkpoint.unwrap_or(true),
            // Carried through unresolved; see the field. The profile > `[session]` precedence is
            // applied once, per profile, in `resolve_profile`, and the call sites in `main.rs`
            // apply `DEFAULT_CONTEXT_WINDOW` when both are unset.
            session_context_window: file_session.context_window,
            // Unresolved for the same reason, and by the same rule: every profile that states no
            // budget of its own falls back to this one.
            default_thinking_budget,
            subagent_max_depth: file_session
                .subagent_max_depth
                .unwrap_or(DEFAULT_SUBAGENT_MAX_DEPTH),
            vision: active_settings
                .as_ref()
                .is_none_or(|settings| settings.vision),
            max_output_tokens: active_settings
                .as_ref()
                .and_then(|settings| settings.max_output_tokens),
            mcp_servers,
            mcp_default_permission,
            profile_summaries,
            skills_enabled,
            skills_agent_managed,
            skills_extra_paths,
            memory_enabled,
            schedule,
            background,
            subagents,
            builtin_allowed_tools,
            builtin_disabled_tools,
            builtin_tool_permissions,
            input_style: file_display
                .input_style
                .as_deref()
                .map(parse_input_style)
                .unwrap_or_else(default_input_style),
            mcp_grace,
            mcp_connect_timeout,
            mcp_stdio_concurrency,
            mcp_http_concurrency,
            serve: config_file.serve,
            request: RunRequest {
                writable_roots: overrides.writable_roots,
                requested_permission,
                requested_profile: overrides.profile,
                session_resume: SessionResume::from_flags(
                    overrides.resume,
                    overrides.continue_last,
                ),
                prompt: overrides.prompt,
                oneshot: overrides.oneshot,
                output_format: overrides.output_format,
                serve_bind_override: None,
                instructions: overrides.instructions,
            },
        }
    }

    /// Refuse to answer from empty defaults when `config.toml` did not parse.
    ///
    /// For the subcommands that never reach [`Self::validate`] but do read the parsed config:
    /// `meka mcp list` over a config full of servers would otherwise print "No MCP servers.", and
    /// `meka mcp get <name>` would say the server does not exist. Both are indistinguishable from
    /// the truthful answer, which is what makes them worth failing on.
    pub(crate) fn require_readable_config(&self) -> crate::error::Result<()> {
        match &self.config_error {
            Some(error) => Err(crate::error::MekaError::Config(error.clone())),
            None => Ok(()),
        }
    }

    /// Every directory skills are read from, in precedence order.
    ///
    /// A method rather than the four `skill_roots(&config.skills_extra_paths)` spellings it
    /// replaces: the derivation is "meka's own first, then the configured extras", and a caller
    /// that assembled it itself was a caller that could get the order wrong.
    pub(crate) fn skill_roots(&self) -> Vec<PathBuf> {
        crate::paths::skill_roots(&self.skills_extra_paths)
    }

    /// Note an unreadable `config.toml` without failing, for the commands that edit the raw
    /// document through `toml_edit` and so work fine on one meka can't parse. They are how the file
    /// gets repaired from the CLI, so they must run; the warning is there because their view of the
    /// config is empty and any message they print about it would otherwise mislead.
    pub(crate) fn warn_if_config_unreadable(&self) {
        if let Some(error) = &self.config_error {
            tracing::warn!("ignoring config file: {error}");
        }
    }

    pub(crate) fn validate(&self) -> crate::error::Result<()> {
        // Profile-selection failure (none / ambiguous / unknown name) is reported first with its
        // specific guidance. Ahead of the provider check: with an unparsed config there are no
        // profiles at all, and "no provider configured" would send the user chasing the wrong
        // problem.
        if let Some(error) = &self.config_error {
            return Err(crate::error::MekaError::Config(error.clone()));
        }
        self.validate_default_profile()?;
        warn_about_inert_profile_keys(&self.accounts, &self.profiles);
        // `retention = "0s"` means "delete anything not updated in the last zero seconds", i.e.
        // every session, on every startup. Nobody means that, and the cost of guessing wrong is
        // unrecoverable, so refuse rather than run it once and find out.
        if self.retention.is_some_and(|retention| retention.is_zero()) {
            return Err(crate::error::MekaError::Config(
                "`[session].retention = \"0s\"` would delete every session on each startup; remove \
                 the key to keep sessions"
                    .to_string(),
            ));
        }
        // A zero timeout fails every request before it is sent. The integer keys these replaced
        // read `0` as "the default", so a file converted by hand may still say it and has to be
        // told, rather than handed a client that cannot fetch anything.
        for (key, timeout) in [
            ("request_timeout", Some(self.web_client.request_timeout)),
            ("connect_timeout", self.web_client.connect_timeout),
            ("read_timeout", self.web_client.read_timeout),
        ] {
            if timeout.is_some_and(|timeout| timeout.is_zero()) {
                return Err(crate::error::MekaError::Config(format!(
                    "`[web].{key} = \"0s\"` would time out every request before it is sent; remove \
                     the key for the default"
                )));
            }
        }
        if self.mcp_connect_timeout.is_zero() {
            return Err(crate::error::MekaError::Config(
                "`[mcp].connect_timeout = \"0s\"` would time out every server before it connects; \
                 remove the key for the default of \"30s\""
                    .to_string(),
            ));
        }
        // The connector buffers this many connects at once; zero would wait forever for the first.
        for (key, limit) in [
            ("stdio_concurrency", self.mcp_stdio_concurrency),
            ("http_concurrency", self.mcp_http_concurrency),
        ] {
            if limit == 0 {
                return Err(crate::error::MekaError::Config(format!(
                    "`[mcp].{key} = 0` would connect no server; remove the key for the default"
                )));
            }
        }
        // `context_messages = 0` reads as "send no history", which is not a thing the provider APIs
        // accept: a request needs at least the current user message. Indexing one past the end of
        // the message slice on the first turn panics: fatal in the REPL, and under ACP it leaves
        // the client's `session/prompt` waiting forever with no response.
        if self.context_messages == Some(0) {
            return Err(crate::error::MekaError::Config(format!(
                "`[session].context_messages = 0` would send no conversation; remove the key for \
                 the default of {DEFAULT_CONTEXT_MESSAGES}"
            )));
        }
        // `tokio::time::interval` panics on a zero period, so this would be a config value taking
        // the process down rather than a setting behaving oddly.
        if self.schedule.poll_interval.is_zero() {
            return Err(crate::error::MekaError::Config(
                "`[schedule].poll_interval = \"0s\"` would never tick; remove the key for the \
                 default of \"10s\""
                    .to_string(),
            ));
        }
        // A zero budget fails every gate before it can produce output, so every gated job would
        // report a broken watcher forever.
        if self.schedule.gate_timeout.is_zero() {
            return Err(crate::error::MekaError::Config(
                "`[schedule].gate_timeout = \"0s\"` would time out every gate before it runs; \
                 remove the key for the default of \"30s\""
                    .to_string(),
            ));
        }
        if self.schedule.max_jobs == 0 {
            return Err(crate::error::MekaError::Config(
                "`[schedule].max_jobs = 0` would refuse every job; set `[schedule] enabled = \
                 false` to turn scheduling off"
                    .to_string(),
            ));
        }
        if self.schedule.max_consecutive_fires == 0 {
            return Err(crate::error::MekaError::Config(
                "`[schedule].max_consecutive_fires = 0` would hold every due job over forever; set \
                 `[schedule] enabled = false` to turn scheduling off"
                    .to_string(),
            ));
        }
        // A lease has to outlast the work it covers, and the probe is the part meka can check. One
        // that expires while a host is still working lets a second host take the same occurrence,
        // which the session lock catches at the cost of a deferral and a re-run gate probe. Checked
        // against `gate_timeout` rather than against a fixed floor because that is the only bound
        // on the work meka knows; the turn after it is unbounded, which is why the documentation
        // asks for headroom on top rather than this settling the question.
        if self.schedule.claim_lease <= self.schedule.gate_timeout {
            return Err(crate::error::MekaError::Config(format!(
                "`[schedule].claim_lease` ({}) must outlast `[schedule].gate_timeout` ({}); remove \
                 the key for the default of \"1h\"",
                humantime_serde::re::humantime::format_duration(self.schedule.claim_lease),
                humantime_serde::re::humantime::format_duration(self.schedule.gate_timeout),
            )));
        }
        Ok(())
    }

    /// Everything this run needs to be true of the **process default** profile: that one could be
    /// picked at all, that its backend is one meka speaks, and that it names a model.
    ///
    /// Skipped entirely for a resume, because a resumed session answers all three from its own row
    /// and never reads [`Self::default_profile`]. Each is re-asked against the profile that
    /// actually applies, with a message naming it: `require_profile` for the first,
    /// [`crate::provider::ProviderBuilder::build`]'s fallthrough for the second,
    /// [`crate::provider::ProviderRegistry::resolve`] for the third.
    ///
    /// Firing these unconditionally would do two bad things. A session whose profile is still
    /// configured could not be resumed at all, though its row holds the answer and `meka session
    /// list` prints it. And a session whose profile *has* been deleted would be refused with
    /// whichever of these tripped first rather than with the specific "no profile named 'work'": a
    /// generic message masking a precise one, and only when two or more profiles happen to remain.
    ///
    /// Deferred, not dropped: a resume that finds no session, or a row with no recorded profile,
    /// still needs a default, and `resolve_session_profile` carries [`Self::provider_error`] into
    /// that path so the same guidance appears there.
    fn validate_default_profile(&self) -> crate::error::Result<()> {
        if self.request.session_resume.is_some() {
            return Ok(());
        }
        if let Some(error) = &self.provider_error {
            return Err(crate::error::MekaError::Config(error.clone()));
        }
        if self.backend.is_none() {
            return Err(crate::error::MekaError::Config(
                "no profile configured; run `meka account add <name>`, then `meka profile add \
                 <name>`"
                    .to_string(),
            ));
        }
        if self.model.is_none() {
            return Err(crate::error::MekaError::Config(format!(
                "profile '{0}' names no model; set one with `meka profile set {0} model <model>`",
                self.default_profile.as_deref().unwrap_or("?"),
            )));
        }
        validate_max_output_tokens(
            self.default_profile.as_deref().unwrap_or("?"),
            self.backend,
            self.max_output_tokens,
            self.thinking,
            self.thinking_budget,
        )?;
        Ok(())
    }
}

/// The one lock serializing every test in this crate that mutates `MEKA_CONFIG_DIR` or
/// `MEKA_DATA_DIR`.
///
/// The var is process-global and unit tests share a process, so a per-module lock only serializes a
/// module against itself. Two of them are worse than none: with `cli/skills.rs` holding its own
/// lock, a config test's `remove_var` could land mid-test and send `crate::cli::skills::run_add` at
/// the developer's real `~/.config/meka`. A `tokio::sync::Mutex` because the skills tests are async
/// and hold the guard across `.await`; the synchronous tests here take it with `blocking_lock`.
#[cfg(test)]
pub(crate) static CONFIG_DIR_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Built-in tool policy from `[tools]` in `config.toml`. Mirrors the three knobs
/// [`crate::config::McpServerConfig`] exposes for MCP tools.
#[derive(Debug, Clone, Default)]
pub(crate) struct BuiltinToolFilter {
    pub(crate) allowed: Option<HashSet<String>>,
    pub(crate) disabled: HashSet<String>,
    pub(crate) permission_overrides: HashMap<String, Permission>,
}
impl BuiltinToolFilter {
    pub(crate) fn from_config(
        allowed: Option<Vec<String>>,
        disabled: Vec<String>,
        permission_overrides: HashMap<String, Permission>,
    ) -> Self {
        // Empty allow-list → None so `admits` treats it as "no restriction".
        let allowed = allowed.and_then(|list| {
            if list.is_empty() {
                None
            } else {
                Some(list.into_iter().collect())
            }
        });
        Self {
            allowed,
            disabled: disabled.into_iter().collect(),
            permission_overrides,
        }
    }

    pub(crate) fn admits(&self, name: &str) -> bool {
        if self.denies(name) {
            return false;
        }
        match &self.allowed {
            Some(list) => list.contains(name),
            None => true,
        }
    }

    /// The block-list half alone, for tools that `allowed_tools` was never able to reach.
    ///
    /// `allowed_tools` is exhaustive: naming five tools removes everything else. Anyone who wrote
    /// one before the MCP meta-tools were filterable wrote it against a world where those seven
    /// registered unconditionally, so applying the allow-list to them now would delete
    /// `mcp_resource_read` and friends from working installations on upgrade, with nothing in the
    /// config to explain it. `disabled_tools` has no such problem: naming a tool there has always
    /// meant "remove this one", so honoring it is what the user already asked for.
    pub(crate) fn denies(&self, name: &str) -> bool {
        self.disabled.contains(name)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(try_from = "String", into = "String")]
pub(crate) enum RenderMode {
    /// syntect-based highlighter. Named after the `syntect` crate that does the in-process
    /// highlighting. Shows the markdown source as the model wrote it, reflowing nothing, so a wide
    /// table runs past the terminal edge.
    Syntect,
    /// Rendered CommonMark, reflowed to the terminal (default).
    ///
    /// The default because meka's own output is table-heavy: `task_list`, `scratchpad_list`, and
    /// anything the model formats as a table all wrap inside their box here and run off the right
    /// edge under `syntect`. Reading rendered prose is also the common case; wanting to see the
    /// markers is the exception, and `syntect` is one config line away.
    ///
    /// One spelling, `termimad`, on the flag, in the environment and in `config.toml` alike.
    #[default]
    Termimad,
    Raw,
}
impl RenderMode {
    /// Every mode, in the order the names sort.
    pub(crate) const ALL: [RenderMode; 3] = [Self::Raw, Self::Syntect, Self::Termimad];

    /// The one spelling `[display].render_mode`, `--render-mode` and `MEKA_RENDER_MODE` take.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Syntect => "syntect",
            Self::Termimad => "termimad",
            Self::Raw => "raw",
        }
    }

    /// The names, joined for a refusal that lists what would have been accepted.
    pub(crate) fn supported() -> String {
        Self::ALL
            .iter()
            .map(|mode| mode.name())
            .collect::<Vec<_>>()
            .join(", ")
    }
}
impl std::fmt::Display for RenderMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}
impl TryFrom<String> for RenderMode {
    type Error = String;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        value.parse()
    }
}
impl From<RenderMode> for String {
    fn from(mode: RenderMode) -> Self {
        mode.name().to_string()
    }
}
/// How much of a tool call's input the tool indicator shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ToolParams {
    /// Name only: `[tool execute_command]`. The only setting under which a model-supplied string
    /// never reaches the terminal at all.
    Off,
    /// Name plus the one argument [`crate::tools::resolve_primary_param`] picks out, on one line
    /// (default).
    #[default]
    Summary,
    /// Every argument, as an indented block under the name. See `render_tool_params` in `render`.
    Full,
}
impl std::fmt::Display for ToolParams {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolParams::Off => write!(formatter, "off"),
            ToolParams::Summary => write!(formatter, "summary"),
            ToolParams::Full => write!(formatter, "full"),
        }
    }
}
impl std::str::FromStr for RenderMode {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|mode| mode.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not a render mode. Supported: {}",
                    Self::supported()
                )
            })
    }
}

/// Whether a Claude request asks for extended thinking, and which wire encoding it uses.
///
/// One knob rather than two. Stated rather than inferred from the model name: `anthropic-messages`
/// reaches any Anthropic-compatible endpoint, so meka cannot tell which encoding the far side
/// implements. The profile states it, and is the user's to keep correct if they later change
/// `model`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub(crate) enum ThinkingMode {
    /// The model sets its own budget. Claude 4.6 and newer.
    ///
    /// Sends the adaptive thinking block, and is the default because it is what the current models
    /// want. The wire shapes each mode produces live in
    /// `anthropic::shared::insert_thinking_fields`.
    #[default]
    Adaptive,
    /// A fixed budget, from the profile's `thinking_budget` or `[thinking].budget`. Required by
    /// pre-4.6 Claude.
    ///
    /// The older encoding, and the one most third-party Anthropic-compatible servers implement.
    Budgeted,
    /// No thinking requested.
    Off,
}
impl ThinkingMode {
    /// Every mode, in the order the names sort.
    pub(crate) const ALL: [ThinkingMode; 3] = [Self::Adaptive, Self::Budgeted, Self::Off];

    /// The one spelling of each mode: what the profile key, `profile add --thinking` and the
    /// `/status` block all use. One copy, because a per-surface match is several hand-written
    /// mappings that have to agree with each other, with nothing checking them against each other.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Adaptive => "adaptive",
            Self::Budgeted => "budgeted",
            Self::Off => "off",
        }
    }

    /// The names, joined for a refusal that lists what would have been accepted.
    pub(crate) fn supported() -> String {
        Self::ALL
            .iter()
            .map(|mode| mode.name())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Whether the request asks for thinking in any encoding. The betas and the `temperature` gate
    /// key off this rather than off a specific encoding.
    pub(crate) fn is_on(self) -> bool {
        !matches!(self, ThinkingMode::Off)
    }
}
impl std::fmt::Display for ThinkingMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}
impl std::str::FromStr for ThinkingMode {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|mode| mode.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not a thinking mode. Supported: {}",
                    Self::supported()
                )
            })
    }
}
impl TryFrom<String> for ThinkingMode {
    type Error = String;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        value.parse()
    }
}
impl From<ThinkingMode> for String {
    fn from(mode: ThinkingMode) -> Self {
        mode.name().to_string()
    }
}

/// `headers` and `env` are the two places a user is most likely to put a bearer token, and a
/// `{:?}` on this struct (in a connect error, a `tracing::debug!`, a panic message) printed them.
/// Names are kept: knowing that `Authorization` was set is the diagnostic; knowing its value is the
/// leak.
fn redact_map(
    map: &Option<std::collections::HashMap<String, String>>,
) -> Option<std::collections::BTreeMap<&str, String>> {
    map.as_ref().map(|entries| {
        entries
            .iter()
            .map(|(name, value)| (name.as_str(), format!("[REDACTED len={}]", value.len())))
            .collect()
    })
}

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
    /// should ship eager-loaded instead of deferred. Saves a `load_tool` round-trip and keeps the
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

/// `[serve]` table: HTTP server config for `meka serve`. All fields optional with sensible
/// defaults, but at least one `[[serve.tokens]]` entry is required; the server refuses to start
/// without one.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct ServeConfig {
    /// Listen address. Default `127.0.0.1:8080`: bind to loopback so a fresh deploy isn't
    /// accidentally world-reachable. Operators front with a reverse proxy (nginx, caddy) for
    /// TLS termination and put a public address there.
    pub(crate) bind: Option<String>,
    /// Idle-timeout for session eviction. Sessions with no turn activity for this long are dropped
    /// from the in-memory map by the GC scanner. Accepts humantime strings like `"24h"`, `"30m"`,
    /// `"86400s"`. Default `"24h"`.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) idle_timeout: Option<std::time::Duration>,
    /// How often the GC scanner sweeps the session map. Accepts humantime strings like `"5m"`,
    /// `"300s"`. Default `"5m"`.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) gc_scan_interval: Option<std::time::Duration>,
    /// When true, GC also deletes the SQLite row for idle sessions; default false (keep the row so
    /// a future request with the same session ID can re-attach, mirroring ACP's `session/load`).
    pub(crate) delete_on_idle: Option<bool>,
    /// On SIGTERM / SIGINT, wait at most this long for in-flight turns to finish before forcibly
    /// aborting. Accepts humantime strings like `"30s"`, `"1m"`. Default `"30s"`.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) shutdown_drain_timeout: Option<std::time::Duration>,
    /// Process-wide cap on concurrent in-flight turns across all sessions. None = unbounded
    /// (default). Returns 429 with `concurrency-limit` when exceeded.
    pub(crate) max_concurrent_turns: Option<usize>,
    /// Request body size limit (bytes). Default 10 MiB.
    pub(crate) max_body_bytes: Option<usize>,
    /// Whether a 502 carries the failing provider call's error text, as a `provider_response`
    /// extension member. Default `true`. `detail` is meka's own sentence and is identical either
    /// way.
    ///
    /// On, because the upstream's error type is the actionable part of a failed turn and "consult
    /// the server log" is no answer to anyone running against a meka they do not operate. `meka
    /// acp` honors this key too: the same policy decides what a failed turn's `error.data`
    /// carries.
    ///
    /// **What it can expose.** Usually the upstream's response body, which can name the
    /// *operator's* provider account and its rate-limit posture. Not always, though: the member
    /// carries the failing call's error message, and for some failures that is meka's own sentence
    /// about the call rather than anything the provider sent.
    ///
    /// **Who can read it.** `sessions:r`, not just `sessions:w`. Submitting a turn takes the write
    /// scope, but the failure also rides the terminal `turn.failed` event, and `GET
    /// /v1/sessions/{id}/stream` replays that to any reader.
    ///
    /// Turn it off where read-only tokens go to people who may watch a session but are not
    /// entitled to the account behind it. `/errors/mcp-unavailable` is not covered either way: it
    /// reports server names only, and that reason is meka's own subprocess text.
    pub(crate) relay_provider_errors: Option<bool>,
    /// Whether to serve the Swagger UI and the OpenAPI document at `/v1/docs` and
    /// `/v1/openapi.json`. Default `false`.
    ///
    /// Off by default because they are unauthenticated (as are the two health probes, which
    /// publish nothing) and what they publish is the shape of every endpoint the deployment
    /// exposes. That is useful while building a client and pure reconnaissance value once the
    /// deployment is real. Turn it on deliberately, on a deployment where anyone who can reach the
    /// port is entitled to the map.
    pub(crate) docs: Option<bool>,
    /// How many SSE events per turn to retain so a client reconnecting with `Last-Event-ID` can
    /// replay what it missed. Default 256, matching the live broadcast channel's capacity.
    ///
    /// Raising it buys a longer reconnect window at the cost of holding more per-session memory
    /// during a turn; `0` switches replay off, so a reconnect gets only what happens from then on.
    pub(crate) stream_replay_events: Option<usize>,
    /// How long a streaming turn keeps running after its SSE consumer disconnects, waiting for a
    /// reconnect. Accepts humantime strings like `"30s"`. Default `"30s"`.
    ///
    /// `"0s"` cancels the turn the moment the stream drops. That spends fewer provider tokens on
    /// abandoned work, and makes re-attach useful only for turns that already finished.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) stream_reattach_grace: Option<std::time::Duration>,
    /// Bearer tokens configured for this deployment. An empty list is refused at startup: `meka
    /// serve` exits rather than binding a port nothing can authenticate against.
    pub(crate) tokens: Option<Vec<ServeTokenConfig>>,
    /// Outbound webhook endpoints. Empty (the default) means meka never makes an outbound request.
    pub(crate) webhooks: Option<Vec<WebhookConfig>>,
}
/// One entry in `[[serve.webhooks]]`.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct WebhookConfig {
    /// Where to POST. `http://` is accepted for loopback development but logged as a warning:
    /// deliveries are signed, not encrypted, so anything on the path can read them.
    pub(crate) url: String,
    /// Shared secret for the `X-Meka-Signature` HMAC. Supports `${ENV_VAR}` substitution.
    /// Mutually exclusive with `secret_file`.
    pub(crate) secret: Option<String>,
    /// Path to a file whose contents (trimmed) are the secret. chmod 0600 recommended.
    pub(crate) secret_file: Option<std::path::PathBuf>,
    /// Which events to deliver. Required and non-empty: an endpoint subscribed to nothing is
    /// almost certainly a mistake, and silently never firing is the worst way to find out.
    pub(crate) events: Vec<String>,
    /// Per-attempt request timeout. Accepts humantime strings like `"10s"`. Default `"10s"`.
    #[serde(default, deserialize_with = "deserialize_optional_duration")]
    pub(crate) timeout: Option<std::time::Duration>,
    /// Retries after the first attempt, with exponential backoff. Default 3.
    pub(crate) max_retries: Option<u32>,
}
// Manual `Debug` so a secret cannot reach a log through the *raw* struct either. Nothing prints
// this today, but the derived one would have been the single unredacted path in the webhook chain,
// and that is exactly the kind of thing a later `dbg!` finds.
impl std::fmt::Debug for WebhookConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebhookConfig")
            .field("url", &self.url)
            .field("secret", &self.secret.as_ref().map(|_| "[REDACTED]"))
            .field("secret_file", &self.secret_file)
            .field("events", &self.events)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}
/// One entry in `[serve.tokens]`. Tokens identify callers; scopes gate what they can do. See the
/// Auth section of the HTTP API docs for the full scope catalog.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct ServeTokenConfig {
    /// Inline token value. Supports `${ENV_VAR}` substitution at config-load time. Mutually
    /// exclusive with `token_file`.
    pub(crate) token: Option<String>,
    /// Path to a file whose contents (trimmed) are the token. chmod 0600 recommended.
    pub(crate) token_file: Option<std::path::PathBuf>,
    /// Free-form description, surfaced in startup logs. Operators use it to remember which caller
    /// a token belongs to (e.g. "telegram bridge", "ci debug").
    pub(crate) description: Option<String>,
    /// Scopes granted to this token. See the HTTP API docs for the catalog.
    pub(crate) scopes: Vec<String>,
}
// Manual `Debug` for the same reason as [`WebhookConfig`]'s, and more urgently: `ServeConfig` and
// `ResolvedConfig` both derive `Debug` and `ResolvedConfig` owns the whole `[serve]` table, so the
// derived impl made every bearer token reachable from a single `{:?}` on the config. Redacting only
// `ResolvedServeToken` left that path open, because the raw form survives resolution.
impl std::fmt::Debug for ServeTokenConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServeTokenConfig")
            .field(
                "token",
                &self
                    .token
                    .as_ref()
                    .map(|token| format_args!("[REDACTED len={}]", token.len()).to_string()),
            )
            .field("token_file", &self.token_file)
            .field("description", &self.description)
            .field("scopes", &self.scopes)
            .finish()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instructions::resolve as resolve_instructions, sandbox::resolve_sandbox_backend};

    /// A tilde is expanded whichever separator follows it.
    ///
    /// `~\projects` is what a Windows user types, because it is what PowerShell's own expansion
    /// produces. Matching a literal `"~/"` left it alone, so `/cd` reported the tilde back as a
    /// missing directory and `[skills] extra_paths` resolved to a root that never existed.
    /// `[web].ca_cert_file` is a path setting like the others, so `~` reaches the home directory
    /// instead of the web client failing on a file literally named `~/...`.
    #[test]
    fn the_ca_cert_path_goes_through_tilde_expansion() {
        let home = crate::paths::expand_user_path("~").expect("a home directory");
        let file = WebConfig {
            ca_cert_file: Some("~/certs/corp.pem".to_string()),
            ..WebConfig::default()
        };
        assert_eq!(
            WebClientConfig::from_file(&file).ca_cert_file,
            Some(home.join("certs").join("corp.pem"))
        );
    }

    #[test]
    fn expand_user_path_accepts_either_separator_after_the_tilde() {
        let home = dirs::home_dir().expect("a home directory");

        assert_eq!(expand_user_path("~"), Some(home.clone()));
        assert_eq!(expand_user_path(""), Some(home.clone()));
        assert_eq!(expand_user_path("~/projects"), Some(home.join("projects")));
        #[cfg(windows)]
        assert_eq!(expand_user_path(r"~\projects"), Some(home.join("projects")));

        // Not a tilde-prefixed path: `~foo` is a user reference meka deliberately does not expand,
        // and it must stay a literal rather than silently becoming `$HOME/foo`.
        assert_eq!(
            expand_user_path("~user/projects"),
            Some(PathBuf::from("~user/projects"))
        );
    }

    /// `extra_paths` resolution drops what discovery cannot use, and says so each time.
    ///
    /// A repeat, or meka's own root listed again, makes discovery walk the directory twice and then
    /// report every skill in it as shadowed *by itself*, a warning naming one path twice, which
    /// an operator can neither act on nor dismiss. The empty entry is the dangerous one: it would
    /// expand to `$HOME` and make the whole home directory a skills root.
    #[test]
    fn extra_paths_drops_repeats_the_native_root_and_empty_entries() {
        let native = PathBuf::from("/config/skills");
        let resolved = resolve_skills_extra_paths(
            &[
                "/a/skills".to_string(),
                "/a/skills".to_string(),
                "/config/skills".to_string(),
                "  ".to_string(),
                "/b/skills".to_string(),
            ],
            Some(native.as_path()),
        );
        assert_eq!(
            resolved,
            vec![PathBuf::from("/a/skills"), PathBuf::from("/b/skills")],
            "order is precedence, so it has to be preserved"
        );

        // With no native root there is nothing to compare against, and the rest still applies.
        assert_eq!(
            resolve_skills_extra_paths(&["/a".to_string(), "/a".to_string()], None),
            vec![PathBuf::from("/a")]
        );
    }

    fn fixture_server(name: &str) -> McpServerConfig {
        McpServerConfig {
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

    #[test]
    fn eager_load_override_appends_to_matching_server() {
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
    fn eager_load_override_appends_to_existing_list() {
        let mut servers = vec![fixture_server("notion")];
        servers[0].eager_load_tools = Some(vec!["search".to_string()]);
        apply_cli_eager_load_overrides(&["notion:fetch".to_string()], &mut servers);
        assert_eq!(
            servers[0].eager_load_tools.as_deref(),
            Some(&["search".to_string(), "fetch".to_string()][..])
        );
    }

    #[test]
    fn eager_load_override_dedupes_existing_entry() {
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
    fn eager_load_override_skips_unknown_server() {
        let mut servers = vec![fixture_server("notion")];
        apply_cli_eager_load_overrides(&["nope:search".to_string()], &mut servers);
        // The matching `notion` entry must remain untouched; the unknown `nope` entry simply
        // produces a warn log (not captured here).
        assert!(servers[0].eager_load_tools.is_none());
    }

    #[test]
    fn eager_load_override_skips_malformed_values() {
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

    /// `DisplayConfig` denies unknown fields, so a name or spelling that does not match is a
    /// startup error rather than a silently ignored line. Worth pinning both the key and the three
    /// values, since the config file is the only way to reach this setting.
    #[test]
    fn display_tool_params_parses_each_value() {
        for (written, expected) in [
            ("off", ToolParams::Off),
            ("summary", ToolParams::Summary),
            ("full", ToolParams::Full),
        ] {
            let toml_str = format!("[display]\ntool_params = \"{written}\"\n");
            let config: ConfigFile = toml::from_str(&toml_str).expect("parse toml");
            let display = config.display.expect("display present");
            assert_eq!(display.tool_params, Some(expected));
        }
    }

    /// Omitting it leaves today's one-line indicator, so an existing config keeps its output.
    #[test]
    fn display_tool_params_defaults_to_summary() {
        assert_eq!(ToolParams::default(), ToolParams::Summary);
    }

    /// Deserializing `DisplayConfig` is not the same as the setting working: a key that parses but
    /// never reaches `ResolvedConfig` is a dead feature that every test on either side still
    /// passes. This is the one that fails if the resolution line is dropped.
    ///
    /// Unset means "follow the terminal", so the resolved value stays `None` rather than becoming a
    /// number that would then be honored exactly and pin output to a guess.
    #[test]
    fn display_max_width_reaches_the_resolved_config() {
        assert_eq!(
            resolve_with_config("[display]\nmax_width = 120\n").max_width,
            Some(120)
        );
        assert_eq!(resolve_with_config("[display]\n").max_width, None);
    }

    /// Every budget subtracts fixed chrome first, so below the floor the subtraction leaves nothing
    /// and output degrades to punctuation. Above the ceiling the cost of composing a line grows
    /// quadratically and the renderer stalls. Clamped rather than rejected at both ends: a width
    /// stated badly is a preference, not a broken config.
    #[test]
    fn display_max_width_is_clamped_to_what_can_be_rendered() {
        assert_eq!(
            resolve_with_config("[display]\nmax_width = 5\n").max_width,
            Some(MIN_CONFIGURED_WIDTH)
        );
        assert_eq!(
            resolve_with_config(&format!("[display]\nmax_width = {MIN_CONFIGURED_WIDTH}\n"))
                .max_width,
            Some(MIN_CONFIGURED_WIDTH)
        );
        assert_eq!(
            resolve_with_config("[display]\nmax_width = 100000\n").max_width,
            Some(MAX_CONFIGURED_WIDTH)
        );
        assert_eq!(
            resolve_with_config("[display]\nmax_width = 120\n").max_width,
            Some(120)
        );
    }

    /// The same resolution check for `tool_params`, whose default is a value rather than an
    /// absence: unset must land on `summary`, not on whatever `ToolParams` derives.
    #[test]
    fn display_tool_params_reaches_the_resolved_config() {
        assert_eq!(
            resolve_with_config("[display]\ntool_params = \"full\"\n").tool_params,
            ToolParams::Full
        );
        assert_eq!(
            resolve_with_config("[display]\n").tool_params,
            ToolParams::Summary
        );
    }

    #[test]
    fn eager_load_override_trims_whitespace() {
        let mut servers = vec![fixture_server("notion")];
        apply_cli_eager_load_overrides(&["  notion : search  ".to_string()], &mut servers);
        assert_eq!(
            servers[0].eager_load_tools.as_deref(),
            Some(&["search".to_string()][..])
        );
    }

    #[test]
    fn web_config_all_fields_parse() {
        let toml_str = r#"
[web]
user_agent = "meka-test"
request_timeout = "60s"
connect_timeout = "5s"
read_timeout = "10s"
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
        assert_eq!(web.user_agent.as_deref(), Some("meka-test"));
        assert_eq!(
            web.request_timeout,
            Some(std::time::Duration::from_secs(60))
        );
        assert_eq!(web.connect_timeout, Some(std::time::Duration::from_secs(5)));
        assert_eq!(web.read_timeout, Some(std::time::Duration::from_secs(10)));
        assert_eq!(web.max_redirects, Some(3));
        assert_eq!(web.proxy.as_deref(), Some("socks5h://127.0.0.1:1080"));
        assert_eq!(web.ca_cert_file.as_deref(), Some("/etc/ssl/corp.pem"));
        assert_eq!(web.https_only, Some(true));
        assert_eq!(web.min_tls_version.as_deref(), Some("1.3"));
        assert_eq!(web.danger_accept_invalid_certs, Some(true));
        assert_eq!(web.danger_accept_invalid_hostnames, Some(true));
    }

    #[test]
    fn web_client_config_defaults_from_empty_file() {
        // Empty [web] → sensible defaults; no user-surprising failures.
        let file = WebConfig::default();
        let config = WebClientConfig::from_file(&file);
        assert_eq!(config.user_agent, DEFAULT_WEB_USER_AGENT);
        assert_eq!(config.request_timeout, std::time::Duration::from_secs(30));
        assert!(config.connect_timeout.is_none());
        assert!(config.read_timeout.is_none());
        assert_eq!(config.max_redirects, 10);
        assert!(config.proxy.is_none());
        assert!(config.ca_cert_file.is_none());
        assert!(!config.https_only);
        assert!(config.min_tls_version.is_none());
        assert!(!config.danger_accept_invalid_certs);
        assert!(!config.danger_accept_invalid_hostnames);
    }

    #[test]
    fn web_client_config_resolves_full_file() {
        let file = WebConfig {
            user_agent: Some("ua".to_string()),
            request_timeout: Some(std::time::Duration::from_secs(60)),
            connect_timeout: Some(std::time::Duration::from_secs(5)),
            read_timeout: Some(std::time::Duration::from_secs(10)),
            max_redirects: Some(0),
            proxy: Some("http://proxy.local:8080".to_string()),
            ca_cert_file: Some("/tmp/ca.pem".to_string()),
            https_only: Some(true),
            min_tls_version: Some("1.3".to_string()),
            danger_accept_invalid_certs: Some(true),
            danger_accept_invalid_hostnames: Some(true),
        };
        let config = WebClientConfig::from_file(&file);
        assert_eq!(config.user_agent, "ua");
        assert_eq!(config.request_timeout, std::time::Duration::from_secs(60));
        assert_eq!(
            config.connect_timeout,
            Some(std::time::Duration::from_secs(5))
        );
        assert_eq!(
            config.read_timeout,
            Some(std::time::Duration::from_secs(10))
        );
        assert_eq!(config.max_redirects, 0);
        assert_eq!(config.proxy.as_deref(), Some("http://proxy.local:8080"));
        assert_eq!(
            config.ca_cert_file.as_deref(),
            Some(std::path::Path::new("/tmp/ca.pem"))
        );
        assert!(config.https_only);
        assert_eq!(config.min_tls_version, Some(MinTlsVersion::V1_3));
        assert!(config.danger_accept_invalid_certs);
        assert!(config.danger_accept_invalid_hostnames);
    }

    #[test]
    fn min_tls_version_parse_accepts_all_valid() {
        assert_eq!(MinTlsVersion::parse("1.0"), Some(MinTlsVersion::V1_0));
        assert_eq!(MinTlsVersion::parse("1.1"), Some(MinTlsVersion::V1_1));
        assert_eq!(MinTlsVersion::parse("1.2"), Some(MinTlsVersion::V1_2));
        assert_eq!(MinTlsVersion::parse("1.3"), Some(MinTlsVersion::V1_3));
        // Whitespace trimming.
        assert_eq!(MinTlsVersion::parse("  1.2  "), Some(MinTlsVersion::V1_2));
    }

    #[test]
    fn min_tls_version_parse_rejects_invalid() {
        assert!(MinTlsVersion::parse("1.5").is_none());
        assert!(MinTlsVersion::parse("tls1.3").is_none());
        assert!(MinTlsVersion::parse("").is_none());
    }

    #[test]
    fn web_client_config_rejects_bad_min_tls_falls_back() {
        // Invalid min_tls_version string logs a warn but doesn't abort; we fall through to
        // reqwest's default rather than failing startup on a typo.
        let file = WebConfig {
            min_tls_version: Some("1.5".to_string()),
            ..WebConfig::default()
        };
        let config = WebClientConfig::from_file(&file);
        assert!(config.min_tls_version.is_none());
    }

    /// A zero timeout fails every request before it is sent. The integer keys read `0` as "the
    /// default", which is a guess a file that says zero never asked for; each of the three is
    /// refused by name instead.
    #[test]
    fn a_zero_web_timeout_is_refused_at_startup() {
        for key in ["request_timeout", "connect_timeout", "read_timeout"] {
            let resolved = resolve_with_config(&format!(
                r#"
default_profile = "p"

[accounts.p]
backend = "openai-chat-completions"

[profiles.p]
account = "p"
model = "m"

[web]
{key} = "0s"
"#
            ));
            let error = resolved
                .validate()
                .expect_err("a zero timeout must not be accepted");
            assert!(
                error.to_string().contains(&format!("[web].{key}")),
                "{error}"
            );
        }
        let resolved = resolve_with_config(
            r#"
default_profile = "p"

[accounts.p]
backend = "openai-chat-completions"

[profiles.p]
account = "p"
model = "m"

[web]
request_timeout = "45s"
"#,
        );
        resolved.validate().expect("a positive timeout is fine");
        assert_eq!(
            resolved.web_client.request_timeout,
            std::time::Duration::from_secs(45)
        );
    }

    #[test]
    fn mcp_runtime_fields_parse() {
        let toml_str = r#"
[mcp]
default_permission = "read"
default_required = false
grace = "5s"
connect_timeout = "1m"
stdio_concurrency = 2
http_concurrency = 40
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("parse toml");
        let mcp = config.mcp.expect("mcp present");
        assert_eq!(mcp.default_permission, Some(Permission::Read));
        assert_eq!(mcp.default_required, Some(false));
        assert_eq!(mcp.grace, Some(std::time::Duration::from_secs(5)));
        assert_eq!(
            mcp.connect_timeout,
            Some(std::time::Duration::from_secs(60))
        );
        assert_eq!(mcp.stdio_concurrency, Some(2));
        assert_eq!(mcp.http_concurrency, Some(40));
    }

    /// The two limits reach the connector from the file, with their documented defaults, and a
    /// zero is refused: `buffer_unordered(0)` would wait forever for the first server.
    #[test]
    fn mcp_concurrency_comes_from_the_file_and_zero_is_refused() {
        let provider = r#"
default_profile = "p"

[accounts.p]
backend = "openai-chat-completions"

[profiles.p]
account = "p"
model = "m"
"#;
        let resolved = resolve_with_config(provider);
        assert_eq!(resolved.mcp_stdio_concurrency, 3);
        assert_eq!(resolved.mcp_http_concurrency, 20);
        let resolved = resolve_with_config(&format!(
            "{provider}\n[mcp]\nstdio_concurrency = 1\nhttp_concurrency = 5\n"
        ));
        resolved.validate().expect("positive limits are fine");
        let runtime = crate::mcp::McpRuntimeConfig::from_config(&resolved);
        assert_eq!(runtime.stdio_concurrency, 1);
        assert_eq!(runtime.http_concurrency, 5);
        for key in ["stdio_concurrency", "http_concurrency"] {
            let resolved = resolve_with_config(&format!("{provider}\n[mcp]\n{key} = 0\n"));
            let error = resolved.validate().expect_err("zero must be refused");
            assert!(
                error.to_string().contains(&format!("[mcp].{key}")),
                "{error}"
            );
        }
    }

    /// A zero connect timeout fails every server before it connects; unlike `grace`, where zero
    /// means "do not wait", there is no reading of it anyone wants.
    #[test]
    fn a_zero_mcp_connect_timeout_is_refused_and_a_zero_grace_is_not() {
        let provider = r#"
default_profile = "p"

[accounts.p]
backend = "openai-chat-completions"

[profiles.p]
account = "p"
model = "m"
"#;
        let resolved =
            resolve_with_config(&format!("{provider}\n[mcp]\nconnect_timeout = \"0s\"\n"));
        let error = resolved.validate().expect_err("zero must be refused");
        assert!(
            error.to_string().contains("[mcp].connect_timeout"),
            "{error}"
        );
        let resolved = resolve_with_config(&format!("{provider}\n[mcp]\ngrace = \"0s\"\n"));
        resolved.validate().expect("a zero grace skips the wait");
        assert!(resolved.mcp_grace.is_zero());
    }

    /// `[mcp].default_permission`, a server's `permission` and its `tool_permissions` are refused
    /// where the file is parsed, with the line, the way `[permissions]` is. Dropping a level meka
    /// does not have with a warning, or failing the one server's connection later, tells the user
    /// nothing about where to look.
    #[test]
    fn an_mcp_permission_level_meka_does_not_have_is_refused_at_parse() {
        for body in [
            "[mcp]\ndefault_permission = \"write\"\n",
            "[[mcp.servers]]\nname = \"s\"\ntransport = \"http\"\nurl = \"http://x\"\npermission = \"write\"\n",
            "[[mcp.servers]]\nname = \"s\"\ntransport = \"http\"\nurl = \"http://x\"\n[mcp.servers.tool_permissions]\nsearch = \"write\"\n",
        ] {
            let error = toml::from_str::<ConfigFile>(body)
                .expect_err("write is not a level")
                .to_string();
            assert!(
                error.contains("write") && error.contains("unrestricted"),
                "names the value and the levels meka has: {error}"
            );
        }
    }

    /// Every key `migrate-0.45-to-0.46.py` renames is refused by its old name, so a file the script
    /// did not see fails at the line rather than being read with the setting silently dropped.
    #[test]
    fn a_renamed_config_key_is_refused_by_its_old_name() {
        for (old_key, body) in [
            (
                "request_timeout_seconds",
                "[web]\nrequest_timeout_seconds = 30\n",
            ),
            (
                "connect_timeout_seconds",
                "[web]\nconnect_timeout_seconds = 5\n",
            ),
            ("read_timeout_seconds", "[web]\nread_timeout_seconds = 5\n"),
            ("grace_seconds", "[mcp]\ngrace_seconds = 3\n"),
            (
                "connect_timeout_seconds",
                "[mcp]\nconnect_timeout_seconds = 30\n",
            ),
            ("strict", "[mcp]\nstrict = true\n"),
            ("retention_days", "[session]\nretention_days = 30\n"),
            ("budget_tokens", "[thinking]\nbudget_tokens = 1000\n"),
        ] {
            let error = toml::from_str::<ConfigFile>(body)
                .expect_err("the old key must not parse")
                .to_string();
            assert!(error.contains(old_key), "{old_key}: {error}");
        }
    }

    #[test]
    fn mcp_server_disabled_parses() {
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
        assert_eq!(servers[0].disabled, Some(true));
    }

    #[test]
    fn mcp_server_disabled_defaults_false() {
        let toml_str = r#"
[[mcp.servers]]
name = "normal"
transport = "stdio"
command = "npx"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("parse toml");
        let servers = config.mcp.unwrap().servers.unwrap();
        assert_eq!(servers[0].disabled, None);
    }

    #[test]
    fn config_file_deserialization() {
        let toml_str = r#"
default_profile = "work"

[accounts.work]
backend = "openai-chat-completions"
base_url = "https://api.openai.com/v1"

[profiles.work]
account = "work"
model = "gpt-4o"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        assert_eq!(config.default_profile.as_deref(), Some("work"));
        let account = config
            .accounts
            .get("work")
            .expect("account should be present");
        assert_eq!(account.backend, "openai-chat-completions");
        assert_eq!(
            account.base_url.as_deref(),
            Some("https://api.openai.com/v1")
        );
        let profile = config
            .profiles
            .get("work")
            .expect("profile should be present");
        assert_eq!(profile.account, "work");
        assert_eq!(profile.model.as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn empty_config_file() {
        let config: ConfigFile = toml::from_str("").expect("failed to parse empty toml");
        assert!(config.accounts.is_empty());
        assert!(config.profiles.is_empty());
        assert!(config.default_profile.is_none());
    }

    #[test]
    fn partial_config_file() {
        let toml_str = r#"
[accounts.main]
backend = "claude-subscription"

[profiles.main]
account = "main"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let account = config
            .accounts
            .get("main")
            .expect("account should be present");
        assert_eq!(account.backend, "claude-subscription");
        assert!(account.base_url.is_none());
        let profile = config
            .profiles
            .get("main")
            .expect("profile should be present");
        assert!(profile.model.is_none());
    }

    #[test]
    fn provider_profile_deserializes_effort_and_redact_thinking() {
        let toml_str = r#"
[accounts.work]
backend = "claude-subscription"

[profiles.work]
account = "work"
model = "claude-opus-4-6-20250514"
effort = "medium"
thinking = "budgeted"
redact_thinking = true
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let profile = config
            .profiles
            .get("work")
            .expect("profile should be present");
        assert_eq!(profile.effort.as_deref(), Some("medium"));
        assert_eq!(
            profile.thinking,
            Some(crate::config::ThinkingMode::Budgeted)
        );
        assert_eq!(profile.redact_thinking, Some(true));
    }

    #[test]
    fn the_thinking_block_parses_its_keys() {
        let config: ConfigFile =
            toml::from_str("[thinking]\nbudget = 10000\nshow_content = true\n").expect("parse");
        let thinking = config.thinking.expect("[thinking] present");
        assert_eq!(thinking.budget, Some(10_000));
        assert_eq!(thinking.show_content, Some(true));
    }

    #[test]
    fn unknown_config_keys_are_rejected() {
        // `deny_unknown_fields`: an unrecognized key errors at load instead of silently doing
        // nothing. An unknown key inside a provider profile (here `reasoning_effort`, which meka
        // does not accept because the provider owns that default).
        let stale_profile = r#"
[accounts.work]
backend = "chatgpt-subscription"

[profiles.work]
account = "work"
model = "gpt-5.6-sol"
reasoning_effort = "high"
"#;
        let error = toml::from_str::<ConfigFile>(stale_profile)
            .expect_err("unknown profile key must be rejected")
            .to_string();
        assert!(
            error.contains("reasoning_effort") || error.contains("unknown field"),
            "{error}"
        );
        // A typo'd key inside `[session]`.
        assert!(toml::from_str::<ConfigFile>("[session]\ncontex_window = 1000").is_err());
        // A typo'd top-level section.
        assert!(toml::from_str::<ConfigFile>("[sesion]\nfoo = 1").is_err());
        // The corrected config still parses.
        assert!(toml::from_str::<ConfigFile>("[session]\ncontext_window = 1000000").is_ok());
    }

    #[test]
    fn provider_profile_deserializes_capability_knobs() {
        let toml_str = r#"
[accounts.work]
backend = "openai-chat-completions"

[profiles.work]
account = "work"
model = "gpt-5.5"
context_window = 1000000
vision = false
max_output_tokens = 64000
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let profile = config
            .profiles
            .get("work")
            .expect("profile should be present");
        assert_eq!(profile.context_window, Some(1_000_000));
        assert_eq!(profile.vision, Some(false));
        assert_eq!(profile.max_output_tokens, Some(64_000));
    }

    #[test]
    fn provider_profile_capability_knobs_default_to_none() {
        let toml_str = r#"
[accounts.work]
backend = "openai-chat-completions"

[profiles.work]
account = "work"
model = "gpt-5.5"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let profile = config.profiles.get("work").expect("profile present");
        assert_eq!(profile.context_window, None);
        assert_eq!(profile.vision, None);
        assert_eq!(profile.max_output_tokens, None);
    }

    /// Deserialization only: `required` reaches the per-server config, and an omitted one stays
    /// `None` so resolution can seed it from `default_required`. What that resolution then
    /// produces is covered by `default_required_seeds_required_and_per_server_wins`.
    #[test]
    fn mcp_required_deserialization() {
        let config: ConfigFile = toml::from_str(
            r#"
[mcp]
default_required = true

[[mcp.servers]]
name = "ida"
transport = "stdio"
command = "ida-mcp"
required = false

[[mcp.servers]]
name = "bridge"
transport = "http"
url = "http://127.0.0.1:9100/mcp"
"#,
        )
        .expect("parse");
        let mcp = config.mcp.expect("mcp table");
        assert_eq!(mcp.default_required, Some(true));
        let servers = mcp.servers.expect("servers");
        assert_eq!(servers[0].required, Some(false), "explicit opt-out");
        assert_eq!(
            servers[1].required, None,
            "inherits default_required at resolve time"
        );
    }

    #[test]
    fn memory_and_skills_config_deserialization() {
        let config: ConfigFile = toml::from_str(
            r#"
[memory]
enabled = false

[skills]
enabled = true
"#,
        )
        .expect("failed to parse toml");
        assert_eq!(
            config.memory.expect("memory table present").enabled,
            Some(false)
        );
        assert_eq!(
            config.skills.expect("skills table present").enabled,
            Some(true)
        );
    }

    #[test]
    fn schedule_config_deserialization() {
        let config: ConfigFile = toml::from_str(
            r#"
[schedule]
enabled = true
poll_interval = "5s"
missed_grace = "2h"
gate_timeout = "45s"
max_jobs = 10
max_consecutive_fires = 3
"#,
        )
        .expect("failed to parse toml");
        let schedule = ResolvedScheduleConfig::resolve(
            config.schedule,
            crate::permission::EnabledPermissions::DEFAULT,
        );
        assert!(schedule.enabled);
        assert_eq!(schedule.poll_interval, std::time::Duration::from_secs(5));
        assert_eq!(schedule.missed_grace, std::time::Duration::from_secs(7200));
        assert_eq!(schedule.gate_timeout, std::time::Duration::from_secs(45));
        assert_eq!(schedule.max_jobs, 10);
        assert_eq!(schedule.max_consecutive_fires, 3);
    }

    #[test]
    fn schedule_config_defaults_are_filled_when_absent() {
        let config: ConfigFile = toml::from_str("").expect("empty config parses");
        assert!(config.schedule.is_none());
        let schedule = ResolvedScheduleConfig::resolve(
            config.schedule,
            crate::permission::EnabledPermissions::DEFAULT,
        );
        assert!(schedule.enabled, "scheduling is on unless turned off");
        assert!(!schedule.poll_interval.is_zero());
        assert!(!schedule.gate_timeout.is_zero());
        assert!(schedule.max_jobs > 0);
        // Pinned to the value rather than to non-zero: both doc pages state 5, and drifting either
        // way is silent: large disables the interleaving, 1 serializes every session.
        assert_eq!(schedule.max_consecutive_fires, 5);
    }

    #[test]
    fn background_config_deserialization() {
        let config: ConfigFile = toml::from_str(
            "[background]
enabled = true
max_tasks = 3
",
        )
        .expect("config parses");
        let background = ResolvedBackgroundConfig::resolve(config.background);
        assert!(background.enabled);
        assert_eq!(background.max_tasks, 3);
    }

    #[test]
    fn subagents_config_deserialization() {
        let config: ConfigFile = toml::from_str(
            r#"[subagents]
disabled_servers = ["mekabridge"]
disabled_tools = ["mcp__notion__create_page", "write_file"]
"#,
        )
        .expect("config parses");
        let subagents = ResolvedSubagentsConfig::resolve(config.subagents);
        assert_eq!(subagents.disabled_servers, vec!["mekabridge".to_string()]);
        assert_eq!(subagents.disabled_tools, vec![
            "mcp__notion__create_page".to_string(),
            "write_file".to_string()
        ]);
    }

    /// Absent `[subagents]` denies nothing. What a sub-agent *receives* is not configured at all:
    /// it is granted per `agent_spawn` call and defaults to nothing.
    #[test]
    fn subagents_defaults_deny_nothing() {
        let config: ConfigFile = toml::from_str("").expect("empty config parses");
        assert!(config.subagents.is_none());
        let subagents = ResolvedSubagentsConfig::resolve(config.subagents);
        assert!(subagents.disabled_servers.is_empty());
        assert!(subagents.disabled_tools.is_empty());
    }

    /// `[subagents]` deliberately has no `memory` key: config withholds capabilities, and the
    /// memory store is context a parent can copy into a prompt regardless. Setting one must be an
    /// error rather than silently ignored.
    #[test]
    fn subagents_has_no_memory_key() {
        let error = toml::from_str::<ConfigFile>("[subagents]\nmemory = \"none\"\n")
            .expect_err("an unrecognized key must not be silently ignored");
        assert!(error.to_string().contains("memory"), "{error}");
    }

    /// Sub-agents can be granted read access but never write: a sub-agent recording what it
    /// inferred from one narrow task puts it in front of every future turn.
    #[test]
    fn memory_grant_parsing_refuses_write() {
        assert_eq!(MemoryAccess::parse_grant("none"), Ok(MemoryAccess::None));
        assert_eq!(MemoryAccess::parse_grant("  READ "), Ok(MemoryAccess::Read));
        let error = MemoryAccess::parse_grant("write").expect_err("write is not grantable");
        assert!(error.contains("not available to sub-agents"), "{error}");
        assert!(error.contains("yourself"), "and says what to do instead");
        assert!(MemoryAccess::parse_grant("readonly").is_err());
    }

    #[test]
    fn instruction_grant_parsing() {
        assert_eq!(
            InstructionAccess::parse_grant("inherit"),
            Ok(InstructionAccess::Inherit)
        );
        assert_eq!(
            InstructionAccess::parse_grant("None"),
            Ok(InstructionAccess::None)
        );
        assert!(InstructionAccess::parse_grant("yes").is_err());
        assert_eq!(InstructionAccess::default(), InstructionAccess::None);
    }

    /// `min` is "the more restrictive of two", which is what clamps a grant against what the
    /// granting agent itself holds.
    #[test]
    fn memory_access_orders_restrictive_first() {
        assert!(MemoryAccess::None < MemoryAccess::Read);
        assert!(MemoryAccess::Read < MemoryAccess::Write);
        assert_eq!(
            MemoryAccess::Read.min(MemoryAccess::None),
            MemoryAccess::None
        );
        assert_eq!(
            InstructionAccess::Inherit.min(InstructionAccess::None),
            InstructionAccess::None
        );
    }

    /// The one capability block that is off unless asked for. See [`BackgroundConfig`] for why it
    /// departs from `[schedule]` / `[skills]` / `[memory]`.
    #[test]
    fn background_is_off_by_default() {
        let config: ConfigFile = toml::from_str("").expect("empty config parses");
        assert!(config.background.is_none());
        let background = ResolvedBackgroundConfig::resolve(config.background);
        assert!(
            !background.enabled,
            "background tasks are off unless turned on"
        );
        assert!(background.max_tasks > 0);
    }

    #[test]
    fn background_config_rejects_unknown_keys() {
        assert!(
            toml::from_str::<ConfigFile>(
                "[background]
detach = true
"
            )
            .is_err()
        );
    }

    #[test]
    fn schedule_config_rejects_unknown_keys() {
        assert!(
            toml::from_str::<ConfigFile>(
                "[schedule]
on_exit = \"make build\"
"
            )
            .is_err()
        );
    }

    /// Absent tables must stay absent rather than deserializing to something opinionated; the
    /// default-on decision lives in `ResolvedConfig::resolve`'s `unwrap_or(true)`, not in the
    /// parsed shape.
    #[test]
    fn memory_and_skills_absent_by_default() {
        let config: ConfigFile = toml::from_str("").expect("empty config parses");
        assert!(config.memory.is_none());
        assert!(config.skills.is_none());
        assert!(MemoryConfig::default().enabled.is_none());
        assert!(SkillsConfig::default().enabled.is_none());
    }

    /// Both tables are `deny_unknown_fields`, so a typo is a startup error rather than a setting
    /// that silently does nothing.
    #[test]
    fn memory_and_skills_reject_unknown_keys() {
        assert!(
            toml::from_str::<ConfigFile>(
                "[memory]
enabeld = false
"
            )
            .is_err()
        );
        assert!(
            toml::from_str::<ConfigFile>(
                "[skills]
path = \"/tmp\"
"
            )
            .is_err()
        );
    }

    #[test]
    fn session_config_deserialization() {
        let toml_str = r#"
[session]
context_messages = 100
retention = "90d"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let session = config.session.expect("session should be present");
        assert_eq!(session.context_messages, Some(100));
        assert_eq!(
            session.retention,
            Some(std::time::Duration::from_secs(90 * 86_400))
        );
    }

    /// Builds a `ResolvedConfig` from a real config file, so the resolution steps in
    /// `ResolvedConfig::resolve` are exercised rather than re-implemented in the test.
    fn resolve_with_config(body: &str) -> ResolvedConfig {
        use clap::Parser;
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.toml"), body).expect("write config");
        // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serializes every test
        // that touches it, and the guard is held across the whole set → resolve → clear cycle.
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let resolved = ResolvedConfig::resolve(crate::cli::Cli::parse_from(["meka"]).overrides());
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        resolved
    }

    /// The registry gating is tested in `crate::tools`, but nothing joined the two: a key that
    /// never reaches `ResolvedConfig` leaves the tools correctly gated on a flag that is always
    /// false, and every test on either side still passes.
    #[test]
    fn skills_agent_managed_resolves_from_the_config_file() {
        let resolved = resolve_with_config("");
        assert!(resolved.skills_enabled, "skills default on");
        assert!(
            !resolved.skills_agent_managed,
            "authoring must be off unless asked for"
        );

        let resolved = resolve_with_config(
            r#"
[skills]
agent_managed = true
"#,
        );
        assert!(resolved.skills_agent_managed);
        assert!(
            resolved.skills_enabled,
            "agent_managed alone must not disturb the enabled default"
        );
    }

    /// `default_required` is only a default for `required`; every consumer downstream reads the
    /// per-server flag, so if this resolution stopped happening `default_required = true` would
    /// silently stop gating.
    #[test]
    fn default_required_seeds_required_and_per_server_wins() {
        let resolved = resolve_with_config(
            r#"
[mcp]
default_required = true

[[mcp.servers]]
name = "gates"
transport = "http"
url = "http://localhost/mcp"

[[mcp.servers]]
name = "optout"
transport = "http"
url = "http://localhost/mcp"
required = false
"#,
        );
        let by_name = |n: &str| {
            resolved
                .mcp_servers
                .iter()
                .find(|s| s.name == n)
                .unwrap_or_else(|| panic!("{n} present"))
                .required
        };
        assert_eq!(by_name("gates"), Some(true), "inherits default_required");
        assert_eq!(by_name("optout"), Some(false), "explicit opt-out wins");
    }

    /// The default is off: an unavailable server degrades the session instead of stopping it.
    #[test]
    fn servers_are_optional_without_default_required() {
        let resolved = resolve_with_config(
            r#"
[[mcp.servers]]
name = "ida"
transport = "stdio"
command = "ida-mcp"
"#,
        );
        assert_eq!(resolved.mcp_servers[0].required, Some(false));
        assert!(resolved.retention.is_none(), "no cleanup by default");
    }

    /// Zero is "delete everything, every startup". Refuse rather than discover it after the fact.
    /// `context_messages = 0` would send an empty conversation, which no provider accepts, and the
    /// assembly path indexes on the assumption that the window is non-empty. Rejecting it at load
    /// is what keeps that from becoming a panic mid-turn; nothing asserted the rejection.
    #[test]
    fn context_messages_zero_is_rejected() {
        let resolved = resolve_with_config(
            r#"
default_profile = "p"

[accounts.p]
backend = "openai-chat-completions"

[profiles.p]
account = "p"
model = "m"

[session]
context_messages = 0
"#,
        );
        assert_eq!(
            resolved.context_messages,
            Some(0),
            "fixture must actually set it"
        );
        let error = resolved
            .validate()
            .expect_err("context_messages = 0 must not be accepted");
        assert!(error.to_string().contains("context_messages"), "{error}");
    }

    #[test]
    fn a_zero_retention_is_refused() {
        // Needs a usable provider: `validate` reports a missing one first, and rightly so - it
        // blocks the run outright, where retention only bites at the next startup sweep.
        let resolved = resolve_with_config(
            r#"
default_profile = "p"

[accounts.p]
backend = "openai-chat-completions"

[profiles.p]
account = "p"
model = "m"

[session]
retention = "0s"
"#,
        );
        assert_eq!(
            resolved.retention,
            Some(std::time::Duration::ZERO),
            "fixture must actually set it"
        );
        let error = resolved
            .validate()
            .expect_err("retention = \"0s\" must not be accepted");
        assert!(error.to_string().contains("[session].retention"), "{error}");
    }

    /// `tokio::time::interval` panics on a zero period, so without this check a config value takes
    /// the whole process down at scheduler startup rather than merely behaving oddly.
    #[test]
    fn schedule_poll_interval_zero_is_rejected() {
        let resolved = resolve_with_config(
            r#"
default_profile = "p"

[accounts.p]
backend = "openai-chat-completions"

[profiles.p]
account = "p"
model = "m"

[schedule]
poll_interval = "0s"
"#,
        );
        assert!(
            resolved.schedule.poll_interval.is_zero(),
            "fixture must actually set it"
        );
        let error = resolved
            .validate()
            .expect_err("a zero poll interval must not be accepted");
        assert!(error.to_string().contains("poll_interval"), "{error}");
    }

    #[test]
    fn schedule_gate_timeout_and_max_jobs_zero_are_rejected() {
        for (key, value, needle) in [
            ("gate_timeout", "\"0s\"", "gate_timeout"),
            ("max_jobs", "0", "max_jobs"),
            ("max_consecutive_fires", "0", "max_consecutive_fires"),
        ] {
            let resolved = resolve_with_config(&format!(
                r#"
default_profile = "p"

[accounts.p]
backend = "openai-chat-completions"

[profiles.p]
account = "p"
model = "m"

[schedule]
{key} = {value}
"#
            ));
            let error = resolved
                .validate()
                .expect_err("a zero value must not be accepted");
            assert!(error.to_string().contains(needle), "{key}: {error}");
        }
    }

    /// The lease was the one schedule duration nothing checked.
    ///
    /// Zero is the obvious mistake and is not the only one: any lease shorter than the gate budget
    /// can expire while the host that took it is still running its own probe, which hands the same
    /// occurrence to a second host. The session lock stops that becoming a second turn, but only
    /// after the occurrence has been round-tripped and the probe re-run.
    #[test]
    fn schedule_claim_lease_must_outlast_the_gate_budget() {
        for (lease, accepted) in [("\"0s\"", false), ("\"10s\"", false), ("\"90s\"", true)] {
            let resolved = resolve_with_config(&format!(
                r#"
default_profile = "p"

[accounts.p]
backend = "openai-chat-completions"

[profiles.p]
account = "p"
model = "m"

[schedule]
gate_timeout = "30s"
claim_lease = {lease}
"#
            ));
            match accepted {
                true => assert!(
                    resolved.validate().is_ok(),
                    "{lease}: a lease with room for the probe and a turn is fine"
                ),
                false => {
                    let error = resolved.validate().expect_err(
                        "a lease that cannot outlast its own gate must not be accepted",
                    );
                    assert!(
                        error.to_string().contains("claim_lease"),
                        "{lease}: {error}"
                    );
                }
            }
        }
    }

    /// A config that does not parse must stop meka, not be swapped for defaults: one mistyped key
    /// would otherwise silently run the agent with no profiles, no MCP servers and default
    /// permissions, off one warn line among the rest of startup.
    #[test]
    fn unparseable_config_is_reported_not_ignored() {
        // `load_config_file` carries the parse failure; `validate` is what turns it into an exit.
        let (file, error) = {
            let dir = tempfile::tempdir().expect("tempdir");
            std::fs::write(dir.path().join("config.toml"), "[session]\nnot_a_key = 1\n")
                .expect("write config");
            // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serializes every
            // test that touches it, and the guard is held across the whole set → read → clear
            // cycle.
            let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();
            unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
            let loaded = load_config_file();
            unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
            loaded
        };
        assert!(
            file.session.is_none(),
            "a failed parse yields defaults, not partial config"
        );
        let error = error.expect("the parse failure must be reported");
        assert!(error.contains("not_a_key"), "{error}");
    }

    /// Carrying the parse failure is only half of it: `validate` is what turns it into an exit.
    /// Without this, deleting the `config_error` arm leaves the whole suite green while meka goes
    /// back to silently running the agent on defaults.
    #[test]
    fn validate_rejects_an_unparseable_config() {
        let resolved = resolve_with_config("[session]\nnot_a_key = 1\n");
        let error = resolved
            .validate()
            .expect_err("an unparseable config must not start the agent");
        assert!(error.to_string().contains("not_a_key"), "{error}");
        // Ahead of the provider check: this fixture has no profiles either, and "no provider
        // configured" would send the user chasing the wrong problem.
        assert!(resolved.provider_error.is_some(), "fixture has no profiles");
    }

    /// Session cleanup is driven by age alone, so `max_storage_bytes` is not a knob. It must be
    /// rejected outright rather than parsed and ignored, which would leave a config that asks for
    /// a size cap looking like it got one.
    #[test]
    fn size_based_cleanup_is_not_configurable() {
        let error = toml::from_str::<ConfigFile>("[session]\nmax_storage_bytes = 52428800\n")
            .expect_err("the key must not parse");
        assert!(error.to_string().contains("max_storage_bytes"), "{error}");
    }

    #[test]
    fn session_config_partial() {
        let toml_str = r#"
[session]
context_messages = 50
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let session = config.session.expect("session should be present");
        assert_eq!(session.context_messages, Some(50));
        assert!(session.retention.is_none());
    }

    #[test]
    fn session_defaults_applied() {
        let file_session = SessionConfig::default();
        let context_messages = file_session.context_messages.or(Some(200));
        let subagent_max_depth = file_session.subagent_max_depth.unwrap_or(3);

        assert_eq!(context_messages, Some(200));
        assert_eq!(subagent_max_depth, 3);
        // Retention has no default: unset means keep every session forever.
        assert!(file_session.retention.is_none());
    }

    #[test]
    fn mcp_config_deserialization() {
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
permission = "workspace"
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
        assert_eq!(servers[0].permission, Some(Permission::Read));
        assert_eq!(servers[1].name, "web-api");
        assert_eq!(servers[1].transport, McpTransport::Http);
        assert_eq!(servers[1].url.as_deref(), Some("http://localhost:8080/mcp"));
        assert_eq!(servers[1].permission, Some(Permission::Workspace));
    }

    #[test]
    fn mcp_config_empty() {
        let config: ConfigFile = toml::from_str("").expect("failed to parse empty toml");
        assert!(config.mcp.is_none());
    }

    #[test]
    fn session_config_overrides_defaults() {
        let toml_str = r#"
[session]
context_messages = 50
retention = "30d"
subagent_max_depth = 5
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let file_session = config.session.unwrap_or_default();
        let context_messages = file_session.context_messages.or(Some(200));
        let subagent_max_depth = file_session.subagent_max_depth.unwrap_or(3);

        assert_eq!(context_messages, Some(50));
        assert_eq!(
            file_session.retention,
            Some(std::time::Duration::from_secs(30 * 86_400))
        );
        assert_eq!(subagent_max_depth, 5);
    }

    #[test]
    fn mcp_auth_client_credentials() {
        let toml_str = r#"
[[mcp.servers]]
name = "api"
transport = "http"
url = "https://api.example.com/mcp"

[mcp.servers.auth]
type = "client_credentials"
client_id = "my-client"
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
                scopes,
                resource,
            } => {
                assert_eq!(client_id, "my-client");
                assert_eq!(
                    scopes.as_deref(),
                    Some(["read".to_string(), "write".to_string()].as_slice())
                );
                assert_eq!(resource.as_deref(), Some("https://api.example.com"));
            }
            other => panic!("expected ClientCredentials, got {other:?}"),
        }
    }

    #[test]
    fn mcp_auth_client_credentials_jwt() {
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
            other => panic!("expected ClientCredentialsJwt, got {other:?}"),
        }
    }

    #[test]
    fn mcp_auth_oauth() {
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
                scopes,
                redirect_port,
            } => {
                assert_eq!(client_id.as_deref(), Some("my-app"));
                assert_eq!(
                    scopes.as_deref(),
                    Some(["repo".to_string(), "user".to_string()].as_slice())
                );
                assert_eq!(*redirect_port, Some(9000));
            }
            other => panic!("expected OAuth, got {other:?}"),
        }
    }

    #[test]
    fn mcp_auth_oauth_minimal() {
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
                scopes,
                redirect_port,
            } => {
                assert!(client_id.is_none());
                assert!(scopes.is_none());
                assert!(redirect_port.is_none());
            }
            other => panic!("expected OAuth, got {other:?}"),
        }
    }

    #[test]
    fn mcp_no_auth() {
        let toml_str = r#"
[[mcp.servers]]
name = "simple"
transport = "http"
url = "https://api.example.com/mcp"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let servers = config.mcp.unwrap().servers.unwrap();
        assert!(servers[0].auth.is_none());
    }

    /// A secret in `config.toml` is refused, not ignored.
    ///
    /// `auth_token` and `client_secret` moved into `mcp_credentials`, and the only thing standing
    /// between a user with an old config and a silently unauthenticated server is
    /// `deny_unknown_fields`. Dropped instead of refused, the key would parse away and the connect
    /// would fail later with a 401 that names nothing; `McpAuthConfig` in particular did not carry
    /// the attribute until this change, so its `client_secret` would have vanished quietly.
    ///
    /// The message is deliberately serde's own. Naming the retired key specially would be code that
    /// remembers a previous version, which only the migration ledger may do.
    #[test]
    fn a_secret_left_in_config_is_refused_rather_than_ignored() {
        let bearer = r#"
[[mcp.servers]]
name = "simple"
transport = "http"
url = "https://api.example.com/mcp"
auth_token = "bearer-token"
"#;
        let error = toml::from_str::<ConfigFile>(bearer).expect_err("a retired key must not parse");
        assert!(
            error.to_string().contains("auth_token"),
            "the refusal must name the offending key: {error}"
        );

        let secret = r#"
[[mcp.servers]]
name = "simple"
transport = "http"
url = "https://api.example.com/mcp"

[mcp.servers.auth]
type = "client_credentials"
client_id = "my-client"
client_secret = "my-secret"
"#;
        let error = toml::from_str::<ConfigFile>(secret).expect_err("a retired key must not parse");
        assert!(
            error.to_string().contains("client_secret"),
            "the refusal must name the offending key: {error}"
        );
    }

    #[test]
    fn parse_input_style_known_values() {
        use nu_ansi_term::{Color, Style};
        assert_eq!(parse_input_style("bold"), Style::new().bold());
        assert_eq!(parse_input_style("BOLD"), Style::new().bold());
        assert_eq!(parse_input_style("dim"), Style::new().dimmed());
        assert_eq!(parse_input_style("cyan"), Style::new().fg(Color::Cyan));
        assert_eq!(parse_input_style("purple"), Style::new().fg(Color::Magenta));
    }

    #[test]
    fn parse_input_style_none_is_plain() {
        use nu_ansi_term::Style;
        assert_eq!(parse_input_style("none"), Style::default());
    }

    #[test]
    fn parse_input_style_default_and_empty_yield_preset() {
        let preset = default_input_style();
        assert_eq!(parse_input_style(""), preset);
        assert_eq!(parse_input_style("default"), preset);
        assert!(preset.is_bold);
        assert!(preset.foreground.is_some(), "default must set foreground");
        assert!(preset.background.is_some(), "default must set background");
    }

    #[test]
    fn parse_input_style_reverse() {
        assert!(parse_input_style("reverse").is_reverse);
    }

    #[test]
    fn parse_input_style_unknown_falls_back_to_default() {
        // Invalid keywords warn but must not panic; fall back to the same preset used when the key
        // is unset.
        assert_eq!(parse_input_style("superbold"), default_input_style());
    }

    #[test]
    fn tools_config_deserialization() {
        let toml_str = r#"
[tools]
allowed_tools = ["read_file", "find_files"]
disabled_tools = ["search_web"]

[tools.tool_permissions]
execute_command = "workspace"
read_file = "unrestricted"
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("failed to parse toml");
        let tools = config.tools.expect("tools should be present");
        assert_eq!(
            tools.allowed_tools.as_deref(),
            Some(["read_file".to_string(), "find_files".to_string()].as_slice())
        );
        assert_eq!(
            tools.disabled_tools.as_deref(),
            Some(["search_web".to_string()].as_slice())
        );
        let permissions = tools.tool_permissions.expect("tool_permissions set");
        assert_eq!(
            permissions.get("execute_command").copied(),
            Some(Permission::Workspace)
        );
        assert_eq!(
            permissions.get("read_file").copied(),
            Some(Permission::Unrestricted)
        );
    }

    #[test]
    fn tools_config_missing_is_none() {
        let config: ConfigFile = toml::from_str("").expect("failed to parse empty toml");
        assert!(config.tools.is_none());
    }

    /// A level meka does not have is refused where the file is parsed, naming the value and the
    /// levels it has, rather than dropped with a warning that leaves the tool at its built-in
    /// level.
    #[test]
    fn a_tool_permission_level_meka_does_not_have_is_refused_at_parse() {
        let error =
            toml::from_str::<ConfigFile>("[tools.tool_permissions]\nwrite_file = \"superuser\"\n")
                .expect_err("superuser is not a level")
                .to_string();
        assert!(
            error.contains("superuser") && error.contains("unrestricted"),
            "{error}"
        );
    }

    #[test]
    fn the_resume_selector_follows_the_continue_and_resume_flags() {
        use clap::Parser as _;

        let parse = |args: &[&str]| {
            let overrides = crate::cli::Cli::parse_from(args).overrides();
            SessionResume::from_flags(overrides.resume, overrides.continue_last)
        };

        assert_eq!(parse(&["meka"]), None);
        assert_eq!(parse(&["meka", "-c"]), Some(SessionResume::Last));
        assert_eq!(
            parse(&["meka", "-r", "550e8400"]),
            Some(SessionResume::Id("550e8400".to_string())),
        );
        // A prompt alongside either flag is a prompt, not a session; that ambiguity is why `-c`
        // takes no value.
        assert_eq!(
            parse(&["meka", "-c", "-p", "fix the bug"]),
            Some(SessionResume::Last)
        );
        assert_eq!(
            parse(&["meka", "-r", "550e8400", "-p", "fix the bug"]),
            Some(SessionResume::Id("550e8400".to_string())),
        );
    }

    /// The conventional path is the tier with no configuration at all, so a regression here is
    /// invisible: instructions simply stop applying and nothing says so.
    #[test]
    fn instructions_resolve_from_the_conventional_file() {
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("instructions.md"), "  from the file  \n")
            .expect("write instructions.md");

        // SAFETY: `CONFIG_DIR_ENV_LOCK` serializes this with every other env-var test.
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let resolved = resolve_instructions(None);
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let found = resolved.expect("ok").expect("some");
        assert_eq!(found.text, "from the file", "trimmed");
        assert!(matches!(
            found.source,
            crate::instructions::InstructionsSource::Files(_)
        ));
    }

    /// Splitting a grown `instructions.md` into a directory should be a rename, so the directory
    /// has to win rather than the two silently concatenating or the file shadowing it.
    #[test]
    fn instructions_directory_wins_over_the_single_file() {
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("instructions.md"), "single file").expect("write file");
        std::fs::create_dir(dir.path().join("instructions")).expect("mkdir");
        std::fs::write(dir.path().join("instructions/10-a.md"), "split one").expect("write a");
        std::fs::write(dir.path().join("instructions/20-b.md"), "split two").expect("write b");

        // SAFETY: guarded above.
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let resolved = resolve_instructions(None);
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        assert_eq!(
            resolved.expect("ok").expect("some").text,
            "split one\n\nsplit two"
        );
    }

    /// The `mekabox` case: the config directory is mounted read-only, so the container's own
    /// instructions can only arrive as a string. If the env tier ever stopped beating the
    /// conventional file, that wrapper would silently run with the host's instructions instead.
    #[test]
    fn inline_env_beats_the_conventional_file() {
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("instructions.md"), "FROM THE FILE").expect("write");

        // SAFETY: guarded above.
        unsafe {
            std::env::set_var("MEKA_CONFIG_DIR", dir.path());
            std::env::set_var("MEKA_INSTRUCTIONS", "FROM THE ENV");
        }
        let resolved = resolve_instructions(None);
        unsafe {
            std::env::remove_var("MEKA_CONFIG_DIR");
            std::env::remove_var("MEKA_INSTRUCTIONS");
        }

        let found = resolved.expect("ok").expect("some");
        assert_eq!(found.text, "FROM THE ENV");
        assert_eq!(found.source, crate::instructions::InstructionsSource::Env);
    }

    #[test]
    fn instructions_file_env_reads_the_named_path() {
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("elsewhere.md");
        std::fs::write(&path, "from the named path").expect("write");

        // SAFETY: guarded above.
        unsafe { std::env::set_var("MEKA_INSTRUCTIONS_FILE", &path) };
        let resolved = resolve_instructions(None);
        unsafe { std::env::remove_var("MEKA_INSTRUCTIONS_FILE") };

        assert_eq!(
            resolved.expect("ok").expect("some").text,
            "from the named path"
        );
    }

    /// No reading of "both set" means both, so resolving one silently would hide the mistake until
    /// the agent behaved unexpectedly.
    #[test]
    fn both_instruction_env_vars_is_an_error() {
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();
        // SAFETY: guarded above.
        unsafe {
            std::env::set_var("MEKA_INSTRUCTIONS", "inline");
            std::env::set_var("MEKA_INSTRUCTIONS_FILE", "/nonexistent.md");
        }
        let resolved = resolve_instructions(None);
        unsafe {
            std::env::remove_var("MEKA_INSTRUCTIONS");
            std::env::remove_var("MEKA_INSTRUCTIONS_FILE");
        }

        let error = resolved.expect_err("must refuse").to_string();
        assert!(error.contains("MEKA_INSTRUCTIONS"), "{error}");
        assert!(error.contains("MEKA_INSTRUCTIONS_FILE"), "{error}");
    }

    #[test]
    fn flag_beats_every_env_tier_and_trims() {
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();
        // SAFETY: guarded above.
        unsafe { std::env::set_var("MEKA_INSTRUCTIONS", "from the env") };
        let resolved = resolve_instructions(Some("  from the flag  "));
        let blank = resolve_instructions(Some("   \n\t "));
        unsafe { std::env::remove_var("MEKA_INSTRUCTIONS") };

        assert_eq!(resolved.expect("ok").expect("some").text, "from the flag");
        assert!(
            blank.expect("ok").is_none(),
            "an all-whitespace flag is no instructions, not empty ones"
        );
    }

    /// Touches process env, so it serializes against any other env-var test in this file via
    /// [`CONFIG_DIR_ENV_LOCK`].
    #[test]
    fn a_profiles_thinking_mode_reaches_the_resolved_config() {
        // The resolution step, not the parse: `ProfileConfig.thinking` deserializing is separate
        // from `ResolvedConfig::resolve` actually consulting it, and dropping the profile arm here
        // is invisible to every wire-level test: those construct providers directly.
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();

        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            r#"
default_profile = "local"

[accounts.local]
backend = "anthropic-messages"

[profiles.local]
account = "local"
model = "some-local-model"
thinking = "budgeted"
"#,
        )
        .expect("write config.toml");

        // SAFETY: `CONFIG_DIR_ENV_LOCK` serializes this with any other env-var test.
        unsafe {
            std::env::set_var("MEKA_CONFIG_DIR", dir.path());
        }
        use clap::Parser;
        let from_profile =
            ResolvedConfig::resolve(crate::cli::Cli::parse_from(["meka"]).overrides());
        // SAFETY: same as above; the guard is held for the full set→read→clear cycle.
        unsafe {
            std::env::remove_var("MEKA_CONFIG_DIR");
        }

        assert_eq!(
            from_profile.thinking,
            crate::config::ThinkingMode::Budgeted,
            "the profile's thinking mode is the only thing that decides this now"
        );
    }

    /// `MEKA_INSTRUCTIONS` beats the conventional file when `--instructions` is not given.
    ///
    /// Touches process env, so it serializes against any other env-var test in this file via
    /// [`CONFIG_DIR_ENV_LOCK`].
    #[test]
    fn env_var_overrides_config_file_instructions() {
        // The module-level lock, not a private one: a function-local static only serializes the
        // test against itself, which let concurrent tests clobber each other's MEKA_CONFIG_DIR.
        let _guard = CONFIG_DIR_ENV_LOCK.blocking_lock();

        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("instructions.md"), "FROM THE FILE")
            .expect("write instructions.md");

        // SAFETY: `CONFIG_DIR_ENV_LOCK` serializes this with any other env-var test.
        unsafe {
            std::env::set_var("MEKA_CONFIG_DIR", dir.path());
            std::env::set_var("MEKA_INSTRUCTIONS", "FROM ENV VAR");
        }

        let resolved = resolve_instructions(None).expect("instructions resolve");

        // SAFETY: same as above, the `CONFIG_DIR_ENV_LOCK` guard is held for the full
        // set→read→clear cycle.
        unsafe {
            std::env::remove_var("MEKA_CONFIG_DIR");
            std::env::remove_var("MEKA_INSTRUCTIONS");
        }

        assert_eq!(
            resolved.as_ref().map(|found| found.text.as_str()),
            Some("FROM ENV VAR"),
            "MEKA_INSTRUCTIONS should override the conventional instructions file",
        );
    }

    #[test]
    fn the_sandbox_backend_variable_takes_one_spelling_and_ignores_the_rest() {
        assert_eq!(
            parse_sandbox_backend_override("landlock"),
            Some(SandboxBackend::Landlock)
        );
        assert_eq!(
            parse_sandbox_backend_override("  landlock  "),
            Some(SandboxBackend::Landlock),
            "value is trimmed"
        );
        assert_eq!(
            parse_sandbox_backend_override("Bubblewrap"),
            None,
            "one spelling per backend: a case variant is ignored, not folded"
        );
        assert_eq!(parse_sandbox_backend_override(""), None);
        assert_eq!(
            parse_sandbox_backend_override("nonsense"),
            None,
            "unrecognized values are ignored, not fatal"
        );
    }

    #[test]
    fn sandbox_backend_from_str() {
        assert_eq!(
            "landlock".parse::<SandboxBackend>(),
            Ok(SandboxBackend::Landlock)
        );
        assert_eq!(
            "bubblewrap".parse::<SandboxBackend>(),
            Ok(SandboxBackend::Bubblewrap)
        );
        assert!(
            "Bubblewrap".parse::<SandboxBackend>().is_err(),
            "one spelling per backend"
        );
        // Unlike the env path, the CLI parse surfaces an error for a bad value.
        assert!("bogus".parse::<SandboxBackend>().is_err());
    }

    /// Every value enum with a wire spelling follows `Backend`: `name()` round-trips through
    /// `FromStr` and `Display`, and a case variant is refused rather than folded, so the flag, the
    /// variable and the file cannot each accept a different form of the same value.
    #[test]
    fn every_value_enum_has_one_spelling_and_refuses_the_rest() {
        fn check<T>(all: &[T], name: fn(T) -> &'static str)
        where
            T: Copy + PartialEq + std::fmt::Debug + std::fmt::Display + std::str::FromStr,
            T::Err: std::fmt::Debug,
        {
            for value in all.iter().copied() {
                assert_eq!(name(value).parse::<T>().ok(), Some(value));
                assert_eq!(value.to_string(), name(value));
                assert!(
                    name(value).to_uppercase().parse::<T>().is_err(),
                    "{} must be the only spelling of {value:?}",
                    name(value)
                );
            }
            assert!("nonsense".parse::<T>().is_err());
        }
        check(&RenderMode::ALL, RenderMode::name);
        check(&ThinkingMode::ALL, ThinkingMode::name);
        check(&McpTransport::ALL, McpTransport::name);
        check(&OutputFormat::ALL, OutputFormat::name);
        check(
            &crate::cli::SessionExportFormat::ALL,
            crate::cli::SessionExportFormat::name,
        );
        check(
            &crate::store::background::TaskStatus::ALL,
            crate::store::background::TaskStatus::name,
        );
        // The sandbox backend displays its brand case for prose; the wire spelling still
        // round-trips and nothing else is taken.
        for backend in SandboxBackend::ALL {
            assert_eq!(backend.name().parse::<SandboxBackend>(), Ok(backend));
            assert!(backend.display_name().parse::<SandboxBackend>().is_err());
            assert_eq!(backend.to_string(), backend.display_name());
        }
        // A retired spelling of a render mode is refused everywhere, including the variable.
        assert!("rich".parse::<RenderMode>().is_err());
        assert_eq!(parse_render_mode_override("rich"), None);
        assert_eq!(parse_render_mode_override(" raw "), Some(RenderMode::Raw));
        assert_eq!(parse_render_mode_override(""), None);
        assert!(toml::from_str::<DisplayConfig>("render_mode = \"Raw\"").is_err());
        assert!(toml::from_str::<McpServerConfig>("name = \"s\"\ntransport = \"HTTP\"").is_err());
    }

    /// `write_file_atomic` makes each *write* atomic, which does nothing for a lost update: two
    /// editors that each read, mutate and write back will silently discard whichever finished
    /// first. The lock is what makes the read and the write one critical section.
    #[tokio::test]
    async fn the_config_lock_serializes_a_read_modify_write() {
        let _env = CONFIG_DIR_ENV_LOCK.lock().await;
        let dir = tempfile::tempdir().expect("tempdir");
        // SAFETY: `MEKA_CONFIG_DIR` is process-global; the guard above serializes every test that
        // touches it and is held across the whole set → use → clear cycle.
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };

        let path = dir.path().join("config.toml");
        std::fs::write(&path, "counter = 0\n").expect("seed");

        // Each thread does the full read-modify-write under the lock. Without it the two reads
        // race and one increment is lost; with it the file ends at 2.
        let threads: Vec<_> = (0..2)
            .map(|_| {
                let path = path.clone();
                std::thread::spawn(move || {
                    let _lock = lock_config_file().expect("lock");
                    let contents = std::fs::read_to_string(&path).expect("read");
                    let current: u32 = contents
                        .trim()
                        .strip_prefix("counter = ")
                        .and_then(|value| value.parse().ok())
                        .expect("parse");
                    // Widen the window the lock has to cover, so an unlocked version loses
                    // reliably rather than occasionally.
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    write_file_atomic(&path, &format!("counter = {}\n", current + 1))
                        .expect("write");
                })
            })
            .collect();
        for thread in threads {
            thread.join().expect("thread");
        }

        let final_contents = std::fs::read_to_string(&path).expect("read");
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        assert_eq!(
            final_contents.trim(),
            "counter = 2",
            "both increments must survive; one lost means the reads interleaved"
        );
    }

    /// `flock` is per open file description, so a second `open` in the same process conflicts with
    /// the first exactly as another process would. Nesting is not hypothetical: `meka mcp add` on
    /// an HTTPS URL holds the lock through `run_add` and then calls `persist_auth_block_for`,
    /// which wants it too.
    #[tokio::test]
    async fn the_config_lock_can_be_taken_again_by_the_thread_already_holding_it() {
        let _env = CONFIG_DIR_ENV_LOCK.lock().await;
        let dir = tempfile::tempdir().expect("tempdir");
        // SAFETY: `MEKA_CONFIG_DIR` is process-global; the guard above serializes every test that
        // touches it and is held across the whole set -> use -> clear cycle.
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };

        let outer = lock_config_file().expect("outer");
        let inner = lock_config_file().expect("a nested acquisition must not block");
        drop(inner);
        // Still held by the outer guard, so a third nested take is also fine.
        let another = lock_config_file().expect("still reentrant");
        drop(another);
        drop(outer);

        // Once the outermost is dropped the file is free again, so a fresh acquisition takes the
        // real lock rather than believing it is still nested.
        let reacquired = lock_config_file().expect("reacquire after full release");
        assert!(matches!(reacquired, ConfigFileLock::Held { .. }));
        drop(reacquired);

        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
    }

    /// Taking the lock must not add a file to the config directory. People keep that directory in a
    /// dotfiles repository, and a lock file there is a working-tree change meka had no reason to
    /// make. Windows is excluded because it writes `.config.toml.lock` there by design; see
    /// [`open_config_lock_target`].
    #[cfg(unix)]
    #[tokio::test]
    async fn taking_the_config_lock_adds_nothing_to_the_config_directory() {
        let _env = CONFIG_DIR_ENV_LOCK.lock().await;
        let dir = tempfile::tempdir().expect("tempdir");
        // SAFETY: `MEKA_CONFIG_DIR` is process-global; the guard above serializes every test that
        // touches it and is held across the whole set -> use -> clear cycle.
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };

        let held = lock_config_file().expect("lock");
        let entries: Vec<String> = std::fs::read_dir(dir.path())
            .expect("read config dir")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        drop(held);

        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };
        // The directory is the lock, so a command that takes it and then bails writes nothing at
        // all, not even an empty `config.toml`.
        assert!(entries.is_empty(), "locking created {entries:?}");
    }

    /// A write must not end the critical section it happens inside.
    ///
    /// Callers hold the lock past their write: `purge_server` revokes a credential over the network
    /// after writing `config.toml`. Keying the lock to the file's inode would let the next arrival
    /// in the instant the publishing `rename` lands, while the holder is still working.
    #[tokio::test]
    async fn writing_the_config_does_not_release_the_lock_held_across_it() {
        let _env = CONFIG_DIR_ENV_LOCK.lock().await;
        let dir = tempfile::tempdir().expect("tempdir");
        // SAFETY: as above.
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };

        let path = dir.path().join("config.toml");
        std::fs::write(&path, "counter = 0\n").expect("seed");

        let held = lock_config_file().expect("lock");
        write_file_atomic(&path, "counter = 1\n").expect("publish over the file being guarded");

        let contender = std::thread::spawn(|| drop(lock_config_file().expect("contender")));
        std::thread::sleep(std::time::Duration::from_millis(250));
        let contender_got_in = contender.is_finished();

        drop(held);
        contender.join().expect("contender");
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        assert!(
            !contender_got_in,
            "the write let a second process in while the first was still holding the lock"
        );
    }

    fn enabled_set(levels: &[Permission]) -> EnabledPermissions {
        EnabledPermissions::from_levels(levels.iter().copied()).unwrap()
    }

    #[test]
    fn resolve_permission_no_config() {
        let (permission, enabled, _requested) = resolve_permission(None, None, None, None);
        assert_eq!(permission, Permission::Read);
        assert_eq!(enabled, EnabledPermissions::DEFAULT);
    }

    #[test]
    fn resolve_permission_explicit_enabled_all() {
        let list = [
            Permission::None,
            Permission::Read,
            Permission::Workspace,
            Permission::Unrestricted,
        ];
        let (_permission, enabled, _requested) = resolve_permission(None, None, None, Some(&list));
        assert!(enabled.is_enabled(Permission::Workspace));
        assert_eq!(enabled.iter().count(), 4);
    }

    /// A level meka does not have is refused where the file is parsed, with its line, the way an
    /// unknown key is: `ask` above all, which the migration script rewrites, rather than dropped
    /// with a warning and replaced by `read`.
    #[test]
    fn a_permission_level_meka_does_not_have_is_refused_at_parse() {
        let error = toml::from_str::<ConfigFile>("[permissions]\ndefault = \"ask\"\n")
            .expect_err("ask is not a level")
            .to_string();
        assert!(
            error.contains("ask") && error.contains("unrestricted"),
            "names the value and the levels meka has: {error}"
        );
        let error = toml::from_str::<ConfigFile>("[permissions]\nenabled = [\"read\", \"lol\"]\n")
            .expect_err("lol is not a level")
            .to_string();
        assert!(error.contains("lol"), "{error}");
    }

    /// An empty list must not resolve to a set *wider* than the one written.
    ///
    /// `EnabledPermissions::DEFAULT` holds four levels including `unrestricted`, so falling back to
    /// it would answer `enabled = []`, which asks for nothing, with the full ladder reachable by
    /// Shift+Tab. The fallback has to move down the ladder, never up.
    #[test]
    fn an_empty_enabled_list_falls_back_to_read_alone() {
        let list: [Permission; 0] = [];
        let (permission, enabled, _requested) = resolve_permission(None, None, None, Some(&list));
        assert_eq!(enabled, enabled_set(&[Permission::Read]));
        assert!(
            !enabled.is_enabled(Permission::Unrestricted),
            "an empty list must not enable a level the user did not write"
        );
        assert_eq!(permission, Permission::Read);
    }

    #[test]
    fn resolve_permission_default_not_in_enabled_clamps() {
        let list = [Permission::Read];
        let (permission, _enabled, _requested) =
            resolve_permission(None, None, Some(Permission::Unrestricted), Some(&list));
        // `unrestricted` is not enabled → fall back to Read because it is.
        assert_eq!(permission, Permission::Read);
    }

    #[test]
    fn resolve_permission_default_not_in_enabled_no_read_falls_to_lowest() {
        let list = [Permission::Workspace, Permission::Unrestricted];
        let (permission, _enabled, _requested) =
            resolve_permission(None, None, Some(Permission::None), Some(&list));
        // none isn't enabled, Read isn't either → lowest enabled is Workspace.
        assert_eq!(permission, Permission::Workspace);
    }

    #[test]
    fn resolve_permission_explicit_default_used() {
        let (permission, _enabled, _requested) =
            resolve_permission(None, None, Some(Permission::Unrestricted), None);
        assert_eq!(permission, Permission::Unrestricted);
    }

    #[test]
    fn resolve_permission_cli_override_disabled_clamps_to_default() {
        // `unrestricted` not enabled → the CLI request warns and clamps to the configured default
        // (Read).
        let list = [Permission::Read];
        let (permission, _enabled, _requested) =
            resolve_permission(Some(Permission::Unrestricted), None, None, Some(&list));
        assert_eq!(permission, Permission::Read);
    }

    #[test]
    fn resolve_permission_cli_override_enabled_wins() {
        let list = [
            Permission::None,
            Permission::Read,
            Permission::Workspace,
            Permission::Unrestricted,
        ];
        let (permission, _enabled, _requested) =
            resolve_permission(Some(Permission::Workspace), None, None, Some(&list));
        assert_eq!(permission, Permission::Workspace);
    }

    #[test]
    fn resolve_permission_env_override_used() {
        let (permission, _enabled, _requested) =
            resolve_permission(None, Some("unrestricted"), None, None);
        assert_eq!(permission, Permission::Unrestricted);
    }

    #[test]
    fn resolve_permission_env_override_disabled_clamps() {
        // env asks for `unrestricted`, which this config does not enable.
        let list = [Permission::Read];
        let (permission, _enabled, _requested) =
            resolve_permission(None, Some("unrestricted"), None, Some(&list));
        assert_eq!(permission, Permission::Read);
    }

    /// The env value has to be one meka still accepts, or this proves nothing: an unparseable
    /// `MEKA_PERMISSION` is dropped before the precedence rule is ever consulted, so the CLI wins
    /// by default rather than by rule, and reversing the precedence leaves the test green.
    #[test]
    fn resolve_permission_cli_beats_env() {
        let (permission, _enabled, _requested) =
            resolve_permission(Some(Permission::None), Some("unrestricted"), None, None);
        assert_eq!(permission, Permission::None);
    }

    #[test]
    fn permissions_config_deserialization() {
        let toml_str = r#"
[permissions]
default = "workspace"
enabled = ["read", "workspace"]
"#;
        let config: ConfigFile = toml::from_str(toml_str).expect("parse toml");
        let permissions = config.permissions.expect("permissions present");
        assert_eq!(permissions.default, Some(Permission::Workspace));
        assert_eq!(
            permissions.enabled,
            Some(vec![Permission::Read, Permission::Workspace])
        );
    }

    /// `sandbox_backend = "bubblewrap"` and `"landlock"` deserialize cleanly. Any other value,
    /// including the obvious abbreviation `"bwrap"`, must be rejected; we don't want alias creep
    /// that would silently desync generated configs from hand-edited ones.
    #[test]
    fn sandbox_backend_deserializes_strict_values() {
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
    /// with `auto_resolved == false`: no silent fallback even if the probe would suggest
    /// otherwise.
    #[cfg(target_os = "linux")]
    #[test]
    fn resolve_sandbox_backend_explicit_value_is_binding() {
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
    fn resolve_sandbox_backend_auto_resolves_when_unset() {
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
    fn resolve_sandbox_backend_uses_platform_sandbox_on_non_linux() {
        use crate::sandbox::{BackendProbe, SandboxCapability};

        // Explicit `Some(...)` is ignored on non-Linux; the field is documented as Linux-only.
        let (_backend, auto_resolved, probe) =
            resolve_sandbox_backend(Some(SandboxBackend::Bubblewrap));
        assert!(!auto_resolved);
        // The probe should reflect what `detect()` reports for this host, surfaced as `Ok(...)` so
        // the consumer can drop into the platform's spawn path.
        match probe {
            BackendProbe::Ok(SandboxCapability::Unavailable) => {
                panic!("Ok(Unavailable) is incoherent: expected a real capability or Missing")
            }
            BackendProbe::Ok(_) | BackendProbe::Missing { .. } => {}
            other => panic!("unexpected probe variant on non-Linux: {other:?}"),
        }
    }

    /// `[display] stream` is the standing preference and `--no-stream` the per-run one; either
    /// saying no wins.
    #[test]
    fn streaming_follows_the_file_and_the_flag() {
        assert!(streaming_enabled(false, None));
        assert!(!streaming_enabled(false, Some(false)));
        assert!(!streaming_enabled(true, Some(true)));
        assert!(streaming_enabled(false, Some(true)));
    }

    /// The redacting `Debug` impls exist to keep a secret out of a log; without this, each could be
    /// deleted and the suite would stay green while `{:?}` started printing secrets again.
    ///
    /// `McpAuthConfig` is not among them and needs no redaction: its secret lives in
    /// `mcp_credentials`, so every field it carries is public configuration.
    // The serve half needs the resolved token type, which exists only with the feature.
    #[cfg(feature = "serve")]
    #[test]
    fn a_secret_never_reaches_a_debug_rendering() {
        use crate::host::http::config::{ResolvedServeToken, TokenSource};
        let mut server = fixture_server("s");
        server.headers = Some(std::collections::HashMap::from([(
            "Authorization".to_string(),
            "Bearer HEADERSECRET".to_string(),
        )]));
        server.env = Some(std::collections::HashMap::from([(
            "API_KEY".to_string(),
            "ENVSECRET".to_string(),
        )]));
        server.auth = Some(McpAuthConfig::ClientCredentials {
            client_id: "public-client-id".to_string(),
            scopes: None,
            resource: None,
        });

        let rendered = format!("{server:?}");
        for secret in ["HEADERSECRET", "ENVSECRET"] {
            assert!(
                !rendered.contains(secret),
                "'{secret}' leaked into {rendered}"
            );
        }
        // The names around the secrets are the diagnostic, and must survive.
        assert!(rendered.contains("Authorization"), "{rendered}");
        assert!(rendered.contains("API_KEY"), "{rendered}");
        assert!(rendered.contains("public-client-id"), "{rendered}");
        assert!(rendered.contains("REDACTED"), "{rendered}");

        let serve_token = ResolvedServeToken {
            token: "SERVETOKEN".to_string(),
            description: None,
            scopes: Default::default(),
            source: TokenSource::Inline,
        };
        let rendered = format!("{serve_token:?}");
        assert!(!rendered.contains("SERVETOKEN"), "{rendered}");

        // The *raw* config form too, and by the route that actually reaches a log: `ResolvedConfig`
        // derives `Debug` and owns the whole `[serve]` table, so redacting only the resolved token
        // left every configured bearer token one `{:?}` away. Asserting on the raw struct alone
        // would not have caught that; this walks the containment chain.
        let raw_token = ServeTokenConfig {
            token: Some("RAWSERVETOKEN".to_string()),
            token_file: None,
            description: Some("ci".to_string()),
            scopes: vec!["sessions:r".to_string()],
        };
        let rendered = format!("{raw_token:?}");
        assert!(!rendered.contains("RAWSERVETOKEN"), "{rendered}");
        assert!(
            rendered.contains("ci"),
            "the label must survive: {rendered}"
        );

        let serve = ServeConfig {
            tokens: Some(vec![raw_token]),
            ..Default::default()
        };
        let rendered = format!("{serve:?}");
        assert!(
            !rendered.contains("RAWSERVETOKEN"),
            "a token must not reach a log through the enclosing [serve] table: {rendered}"
        );

        let webhook = WebhookConfig {
            url: "https://example/hook".to_string(),
            secret: Some("WEBHOOKSECRET".to_string()),
            secret_file: None,
            events: Vec::new(),
            timeout: None,
            max_retries: None,
        };
        let rendered = format!("{webhook:?}");
        assert!(!rendered.contains("WEBHOOKSECRET"), "{rendered}");
    }
}
