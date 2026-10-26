//! The layer between a host and the agent.
//!
//! What every host shares: the process-wide [`SharedDeps`], agent assembly, the doors that decide
//! whether a session may be driven at all, session resume and repin, and the sweeps a host runs
//! on the way in and out. The hosts themselves are the children: the interactive screen, the
//! one-shot run, and the ACP and HTTP servers.
pub(crate) mod acp;
mod assembly;
#[cfg(feature = "serve")]
pub(crate) mod http;
pub(crate) mod oneshot;
pub(crate) mod scheduler;
mod session;
pub(crate) mod terminal;
pub(crate) mod tui;

use std::sync::Arc;

pub(crate) use assembly::*;
pub(crate) use session::*;

use crate::{
    agent::Agent,
    config::ResolvedConfig,
    error::AlreadyReported,
    permission::SharedPermission,
    session::{AgentOptions, CoreMaterials, SessionCells, SessionMaterials},
    store::{Store, TokenStore},
    tools::ToolRegistry,
};

/// Render a `--skill <name>` invocation into the user-message string that drives the first turn.
/// Returns `Ok(None)` when `--skill` is not set so callers can leave the typed prompt untouched.
///
/// Mirrors the screen's `SlashCommand::SkillInvoke` handler in what it composes: the same
/// `format!("{extra}\n\n{body}")` order when the positional `[PROMPT]` is supplied.
///
/// It does *not* mirror the lookup, and the difference is deliberate. This runs before the agent
/// exists, so it walks the roots itself; the screen reads `agent.skills().current()`, which is the
/// live cache. Both resolve the same name to the same file, but only the screen's sees a skill
/// added mid-session.
pub(crate) async fn build_skill_prompt(
    skill: Option<&str>,
    prompt: Option<&str>,
    roots: &[std::path::PathBuf],
) -> anyhow::Result<Option<String>> {
    let Some(name) = skill else {
        return Ok(None);
    };
    let skill =
        crate::skills::require_skill(name, roots).map_err(crate::error::MekaError::Config)?;
    let body = crate::skills::load_skill_body(&skill)
        .await
        .map_err(|error| anyhow::anyhow!("failed to load skill '{name}': {error}"))?;
    let combined = match prompt {
        Some(extra) if !extra.is_empty() => format!("{extra}\n\n{body}"),
        _ => body,
    };
    Ok(Some(combined))
}
/// Delete the sessions `[session].retention` has expired, sparing any another process holds.
///
/// Opt-in only, and never by size. Conversation history is not reproducible, and a byte budget is
/// unpredictable in a way a time window is not: which sessions it takes depends on the total
/// corpus, so one long conversation today can silently destroy an unrelated one from months ago.
/// `warn!` rather than `info!` because a deletion the user configured is still a deletion they
/// should see at the default log level.
///
/// A session is spared by its file lock and by nothing else, which is why the two hosts that
/// resume a named session call this only after `resolve_session_resume` has taken that lock. Run
/// before it, the sweep deleted the very session `meka -r <id>` was about to open, and the run then
/// failed on "no session matches" for a conversation the user had just listed.
pub(crate) async fn sweep_expired_sessions(
    retention: Option<std::time::Duration>,
    store: &Store,
) -> anyhow::Result<()> {
    let Some(retention) = retention else {
        return Ok(());
    };
    let sweep = store.delete_expired_sessions(retention).await?;
    if sweep.deleted > 0 {
        let deleted = sweep.deleted;
        let window = humantime_serde::re::humantime::format_duration(retention);
        tracing::warn!(
            "deleted {deleted} session(s) not updated in {window} (`[session].retention`)"
        );
    }
    // Only turns bump `updated_at`, so a screen idle past the window looks expired while a human is
    // sitting in front of it. Saying nothing here would leave an operator wondering why their
    // retention setting never takes: the answer is that it did, and spared the one session that
    // was in use.
    if sweep.attached_elsewhere > 0 {
        let spared = sweep.attached_elsewhere;
        tracing::info!("spared {spared} session(s) another meka process has open");
    }
    Ok(())
}

/// Delete, in every session, the turns `[session].turn_retention` has expired and a compaction
/// has summarized. No session is spared for being open: the sweep takes nothing from the view a
/// process works from. Reported at `warn` for the reason the session sweep is.
pub(crate) async fn sweep_expired_turns(
    turn_retention: Option<std::time::Duration>,
    store: &Store,
) -> anyhow::Result<()> {
    let Some(window) = turn_retention else {
        return Ok(());
    };
    let cutoff = crate::text::horizon_before(chrono::Utc::now(), window);
    let sweep = store.delete_expired_turns(cutoff, None).await?;
    if !sweep.is_empty() {
        let line = sweep.describe(window);
        tracing::warn!("{line}");
    }
    Ok(())
}

/// The two retention windows of `[session]`, read off the configuration once so the sweeps and
/// the task that repeats them take values rather than the configuration.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RetentionWindows {
    /// `[session].retention`: how long an untouched session is kept.
    pub(crate) sessions: Option<std::time::Duration>,
    /// `[session].turn_retention`: how long an ended turn is kept.
    pub(crate) turns: Option<std::time::Duration>,
}

impl RetentionWindows {
    pub(crate) fn from_config(config: &ResolvedConfig) -> Self {
        Self {
            sessions: config.retention,
            turns: config.turn_retention,
        }
    }

    fn is_unset(self) -> bool {
        self.sessions.is_none() && self.turns.is_none()
    }
}

/// Both retention sweeps, in the order that leaves the second the least to read: a session gone
/// whole has no turns to date.
pub(crate) async fn sweep_expired(windows: RetentionWindows, store: &Store) -> anyhow::Result<()> {
    sweep_expired_sessions(windows.sessions, store).await?;
    sweep_expired_turns(windows.turns, store).await
}

/// How often a host that stays up re-runs the retention sweeps. A window is written in days or
/// hours, so an hour bounds how long anything outlives it while a process runs.
const EXPIRY_SWEEP_INTERVAL: std::time::Duration = std::time::Duration::from_secs(3600);

/// Re-run [`sweep_expired`] every [`EXPIRY_SWEEP_INTERVAL`] for as long as the task lives, which
/// is the caller's to end. A host that stays up for months would otherwise apply a retention
/// window once, at startup, and never again. Finished at once when neither window is set.
pub(crate) fn spawn_expiry_sweeper(
    windows: RetentionWindows,
    store: Store,
) -> tokio::task::JoinHandle<()> {
    if windows.is_unset() {
        return tokio::spawn(async {});
    }
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(EXPIRY_SWEEP_INTERVAL);
        // A machine back from sleep owes one sweep, not one per hour missed.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // The startup sweep has just run; the first tick fires immediately and would repeat it.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            // Unguarded, one panicking sweep ends retention for the life of the process, silently.
            let sweep = std::panic::AssertUnwindSafe(sweep_expired(windows, &store));
            match futures::FutureExt::catch_unwind(sweep).await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => tracing::warn!("retention sweep failed: {error}"),
                Err(panic) => {
                    let panic = crate::error::panic_message(&*panic);
                    tracing::warn!("retention sweep panicked ({panic}); continuing");
                }
            }
        }
    })
}

/// Wait for a termination signal: SIGTERM or Ctrl+C on Unix, Ctrl+C elsewhere. The one watch both
/// servers drain on, so the two cannot differ about what ends a process.
pub(crate) async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(stream) => stream,
                Err(error) => {
                    tracing::warn!(
                        "failed to install SIGTERM handler: {error}; relying on Ctrl+C only"
                    );
                    if let Err(error) = tokio::signal::ctrl_c().await {
                        tracing::warn!("failed to listen for Ctrl+C: {error}");
                    }
                    return;
                }
            };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => tracing::info!("SIGINT received, draining"),
            _ = term.recv() => tracing::info!("SIGTERM received, draining"),
        }
    }
    #[cfg(not(unix))]
    {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::warn!("failed to listen for Ctrl+C: {error}");
        }
        tracing::info!("Ctrl+C received, draining");
    }
}

/// A slash command a host answers itself. The screen offers every one; an editor over ACP is told
/// about the ones marked `for_editors`, as its `available_commands`.
///
/// The execution-side grammar (aliases, argument splitting, `/mcp` and `/skill` subcommands) stays
/// in `parse_slash_command`; this table only models the names that are completed and documented.
pub(crate) struct HostCommand {
    pub(crate) name: &'static str,
    pub(crate) help: &'static str,
    /// Argument syntax shown after the name in help, empty for no-argument commands. A non-empty
    /// hint is the "takes an argument" predicate that drives completion's trailing space.
    pub(crate) arg_hint: &'static str,
    /// Whether an ACP client is offered it. Only what makes sense with no terminal: the three
    /// that print a report.
    pub(crate) for_editors: bool,
}

pub(crate) const COMMANDS: &[HostCommand] = &[
    HostCommand {
        name: "help",
        help: "Show this help message",
        arg_hint: "",
        for_editors: false,
    },
    HostCommand {
        name: "exit",
        help: "Leave",
        arg_hint: "",
        for_editors: false,
    },
    HostCommand {
        name: "session",
        help: "Show the current session id",
        arg_hint: "",
        for_editors: false,
    },
    HostCommand {
        name: "title",
        help: "Show or set this session's title",
        arg_hint: "[text]",
        for_editors: false,
    },
    HostCommand {
        name: "permission",
        help: "Pick or set the permission level",
        arg_hint: "[level]",
        for_editors: false,
    },
    HostCommand {
        name: "approvals",
        help: "Pick or set whether calls above the level are submitted for approval",
        arg_hint: "[on|off]",
        for_editors: false,
    },
    HostCommand {
        name: "profile",
        help: "Pick or set the profile this session runs on",
        arg_hint: "[name]",
        for_editors: false,
    },
    HostCommand {
        name: "project",
        help: "Pick, open or close the project this session is on",
        arg_hint: "[open <name>|close]",
        for_editors: false,
    },
    HostCommand {
        name: "resume",
        help: "Pick a session to switch to, or switch to one by id or unique prefix",
        arg_hint: "[session]",
        for_editors: false,
    },
    HostCommand {
        name: "compact",
        help: "Summarize and compact the session, optionally saying what to keep",
        arg_hint: "[instructions]",
        for_editors: false,
    },
    HostCommand {
        name: "rewind",
        help: "Drop the last N turns from the conversation (default 1)",
        arg_hint: "[N]",
        for_editors: false,
    },
    HostCommand {
        name: "export",
        help: "Export the current session as Markdown",
        arg_hint: "",
        for_editors: false,
    },
    HostCommand {
        name: "fork",
        help: "Fork this session and continue in the copy",
        arg_hint: "",
        for_editors: false,
    },
    HostCommand {
        name: "cd",
        help: "Change working directory (bare: back to where meka started)",
        arg_hint: "[path]",
        for_editors: false,
    },
    HostCommand {
        name: "skill",
        help: "Pick a skill to run, or run one by name with extra context",
        arg_hint: "[name] [extra]",
        for_editors: false,
    },
    HostCommand {
        name: "memory",
        help: "List saved memories, or show one by name",
        arg_hint: "[name]",
        for_editors: false,
    },
    HostCommand {
        name: "schedule",
        help: "List this session's scheduled jobs, show one, or cancel one by id",
        arg_hint: "[show|cancel <id>]",
        for_editors: false,
    },
    HostCommand {
        name: "task",
        help: "List background tasks, show one, or cancel one by id or all",
        arg_hint: "[show|cancel <id>]",
        for_editors: false,
    },
    HostCommand {
        name: "mcp",
        help: "Manage MCP servers and prompts",
        arg_hint: "<subcommand>",
        for_editors: true,
    },
    HostCommand {
        name: "status",
        help: "Show the profile, model, context use and cumulative session stats",
        arg_hint: "",
        for_editors: true,
    },
    HostCommand {
        name: "usage",
        help: "Show account rate-limit usage (subscription backends)",
        arg_hint: "",
        for_editors: true,
    },
    HostCommand {
        name: "log",
        help: "Show the recent log lines, at every level",
        arg_hint: "",
        for_editors: false,
    },
];

/// The session status every host shows, from the agent's own profile rather than the process
/// default: reading config here reported the default profile's model and backend beside a window
/// and an effort that came from the session's, so `/status` and `/profile` contradicted each
/// other on any resume onto a non-default profile.
pub(crate) async fn format_status(
    agent: &Agent,
    providers: &crate::provider::ProviderRegistry,
    message_count: usize,
) -> String {
    let snap = agent.session_stats_snapshot();
    let (context_tokens, context_window, compactions) = match agent.session_id() {
        Some(session_id) => {
            let budget = agent.context_budget(session_id).await;
            (budget.used, budget.window, budget.generation)
        }
        // The screen's first turn has no row yet, so nothing could have been compacted.
        None => {
            let (context_tokens, context_window) = agent.context_usage();
            (context_tokens, context_window, 0)
        }
    };
    let effort = agent.resolved_effort();
    let profile = agent.profile();
    let project = agent.project();
    let settings = providers.settings(&profile);
    crate::render::format_session_status(
        &snap,
        &crate::render::ModelStatus {
            model: settings
                .as_ref()
                .ok()
                .and_then(|settings| settings.model.as_deref()),
            profile: Some(profile.as_str()),
            project: project.as_deref(),
            account: settings
                .as_ref()
                .ok()
                .map(|settings| settings.account.as_str()),
            backend: settings.as_ref().ok().map(|settings| settings.backend),
            effort: effort.as_deref(),
            thinking: settings
                .as_ref()
                .map(|settings| settings.thinking)
                .unwrap_or_default(),
        },
        message_count,
        context_tokens,
        context_window,
        compactions,
    )
}

/// How a reader shows a running command's output, which decides what one update carries.
///
/// The two modes are not cosmetic variants of each other. In [`Self::Terminal`] the reader owns a
/// scrollback that meka appends to, so an update carries only the bytes not yet sent and the whole
/// output is available at the far end: an ACP client rendering agent-owned terminals, or the HTTP
/// feed's `tool_call.output_delta`. In [`Self::Text`] the reader can only replace a block of text,
/// so meka keeps the running output itself and re-sends a window of it: an ACP client without
/// terminal rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LiveOutputMode {
    /// The reader appends what it is sent.
    Terminal,
    /// The reader replaces what it shows with what it is sent.
    Text,
}
/// Whether a tool's output is relayed while it runs. Only `shell_execute` emits
/// [`crate::frontend::FrontendEvent::ToolCallOutputDelta`], and it is the one tool whose result
/// can be minutes away; a live view opened for anything else would show nothing. One answer for
/// both hosts, so a second tool that starts streaming is wired up in one place.
pub(crate) fn streams_output(tool_name: &str) -> bool {
    tool_name == "shell_execute"
}
/// Per-tool-call state for relaying a running command's output.
pub(crate) struct LiveOutput {
    pub(crate) mode: LiveOutputMode,
    /// Text still to show. In [`LiveOutputMode::Text`] this is the whole (tail-capped) output,
    /// re-sent every tick. In [`LiveOutputMode::Terminal`] it is only the bytes not yet appended,
    /// and it is drained on each send, because the client keeps the scrollback.
    pub(crate) text: String,
    /// `None` until the first update goes out, so the first chunk is never delayed.
    pub(crate) last_sent: Option<std::time::Instant>,
}
/// Shortest gap between two `tool_call_update`s for the same tool call. A chatty build writes
/// thousands of small chunks a second, and one notification per read syscall would spend more time
/// on the wire than on the build. Coalescing is why both modes need a buffer at all.
pub(crate) const LIVE_OUTPUT_INTERVAL: std::time::Duration = std::time::Duration::from_millis(150);
/// How much of the tail to keep visible while a command runs, in [`LiveOutputMode::Text`] only.
/// That mode re-sends its whole buffer on every tick, so an uncapped buffer would make a chatty
/// command cost quadratic in its own output. The terminal mode appends and needs no cap.
pub(crate) const LIVE_OUTPUT_TAIL_BYTES: usize = 8 * crate::text::KIB;
impl LiveOutput {
    pub(crate) fn new(mode: LiveOutputMode) -> Self {
        Self {
            mode,
            text: String::new(),
            last_sent: None,
        }
    }

    /// Append a delta and return what to send now, or `None` while throttled.
    pub(crate) fn push(&mut self, chunk: &str, now: std::time::Instant) -> Option<String> {
        self.text.push_str(chunk);
        if self.mode == LiveOutputMode::Text && self.text.len() > LIVE_OUTPUT_TAIL_BYTES {
            let mut cut = self.text.len() - LIVE_OUTPUT_TAIL_BYTES;
            // A byte count can land inside a multi-byte character, and both slicing and draining
            // there panic. Advance to a boundary before either.
            while cut < self.text.len() && !self.text.is_char_boundary(cut) {
                cut += 1;
            }
            // Prefer opening the view at a line start rather than mid-line.
            if let Some(offset) = self.text[cut..].find('\n') {
                cut += offset + 1;
            }
            self.text.drain(..cut);
        }
        if let Some(last) = self.last_sent
            && now.duration_since(last) < LIVE_OUTPUT_INTERVAL
        {
            return None;
        }
        self.last_sent = Some(now);
        match self.mode {
            // Appending: hand over what has accumulated and start empty again, so the same bytes
            // are never sent twice.
            LiveOutputMode::Terminal => {
                if self.text.is_empty() {
                    None
                } else {
                    Some(std::mem::take(&mut self.text))
                }
            }
            LiveOutputMode::Text => Some(self.text.clone()),
        }
    }

    /// Whatever is still buffered, for the final flush before a call completes. The throttle can
    /// swallow the last chunk of a command that exits right after printing, and in terminal mode
    /// those bytes exist nowhere else.
    pub(crate) fn take_pending(&mut self) -> Option<String> {
        match self.mode {
            LiveOutputMode::Terminal if !self.text.is_empty() => {
                Some(std::mem::take(&mut self.text))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host that stays up re-applies the windows on the hour; what aged since startup goes at
    /// the next tick. The one path the startup sweep does not cover, so the wiring is the test.
    #[tokio::test(start_paused = true)]
    async fn the_hourly_sweeper_takes_what_aged_since_startup() {
        use crate::{
            conversation::{Event, Message},
            store::turns::{TurnEnding, TurnUsage},
        };

        let store = Store::for_test().await;
        let session = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        let turn = uuid::Uuid::new_v4();
        let long_ago = (chrono::Utc::now() - chrono::TimeDelta::days(100)).to_rfc3339();
        store
            .turn_store()
            .open_turn(session, turn, "client", &long_ago)
            .await
            .expect("open");
        store
            .save_event(session, &Event::Append(Message::user("old")), Some(turn))
            .await
            .expect("save");
        store
            .turn_store()
            .close_turn(
                turn,
                &TurnEnding::succeeded("end_turn", TurnUsage::default()),
            )
            .await
            .expect("close");
        store
            .run_sql_for_test(&format!(
                "UPDATE turns SET ended_at = '{long_ago}' WHERE id = '{turn}'"
            ))
            .await;
        store
            .save_event(
                session,
                &Event::CompactBoundary {
                    summary: Message::user("summary"),
                    replaced_count: 1,
                    record: crate::conversation::CompactionRecord::for_test(),
                    loaded_tools_snapshot: Default::default(),
                    checklist_snapshot: Default::default(),
                },
                None,
            )
            .await
            .expect("boundary");
        assert_eq!(store.load_events(session).await.expect("load").len(), 2);

        let unset = spawn_expiry_sweeper(RetentionWindows::default(), store.clone());
        unset
            .await
            .expect("a sweeper with no window finishes at once");

        let sweeper = spawn_expiry_sweeper(
            RetentionWindows {
                sessions: None,
                turns: Some(std::time::Duration::from_secs(30 * 86_400)),
            },
            store.clone(),
        );
        // Polled once before the clock moves, so the task takes its immediate first tick now and
        // the advance below is what fires the hourly one. Advanced first, the interval is created
        // after the move and the hourly tick fires only if the clock auto-advances while a store
        // reply is awaited, which is a race the runtime wins or loses by platform.
        tokio::task::yield_now().await;
        tokio::time::advance(EXPIRY_SWEEP_INTERVAL).await;
        // The sweep runs on the store's own thread, so give the task a few turns to finish it.
        let mut remaining = 2;
        for _ in 0..50 {
            tokio::task::yield_now().await;
            remaining = store.load_events(session).await.expect("load").len();
            if remaining == 1 {
                break;
            }
        }
        sweeper.abort();
        assert_eq!(remaining, 1, "the boundary alone is left");
    }
}
