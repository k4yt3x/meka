//! `meka`: a general-purpose AI agent harness where you describe what you want in natural language
//! and an LLM-backed agent decides which tools to run.
//!
//! The binary wires together: a [`provider`] (Claude or OpenAI), a [`session`] store backed by
//! SQLite, a [`tools`] registry, an MCP client manager, and a [`host::repl`] input loop. The
//! [`agent`] module owns the per-turn loop that streams provider output and dispatches tool calls.

// A test panics on failure by design, so the `[lints.clippy]` panic lints in `Cargo.toml` are
// relaxed for test builds alone. A build without the HTTP API leaves the items only it reads
// unused; that configuration is secondary, and the items are not.
#![cfg_attr(not(feature = "serve"), allow(dead_code))]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable,
        clippy::indexing_slicing,
        clippy::match_wildcard_for_single_variants,
        clippy::significant_drop_tightening,
        clippy::redundant_clone,
        clippy::unused_async
    )
)]

mod agent;
mod background;
mod cli;
mod config;
mod console;
mod conversation;
mod entry;
mod error;
mod frontend;
mod fs;
mod host;
mod image;
mod instructions;
mod mcp;
mod memory;
mod oauth;
mod paths;
mod permission;
mod prompt;
mod provider;
mod relay;
mod render;
mod sandbox;
mod schedule;
mod scheduler;
mod session;
mod skills;
mod stats;
mod store;
mod streams;
mod sync;
mod text;
mod todo;
mod tokens;
mod tools;
mod view;
mod workspace;

use std::sync::Arc;

use clap::Parser;

use crate::{config::ResolvedConfig, store::Store};

/// A failure whose message has already been printed in meka's own format.
///
/// Returning the error itself would print it twice, since `main`'s `anyhow::Result` prints whatever
/// it is given; returning `Ok(())` tells every supervisor and wrapper script that a session meka
/// refused to open was a successful run. This carries the exit status and nothing else, so the host
/// keeps its own rendering (color, and the provider hint underneath) and still fails.
#[derive(Debug)]
pub(crate) struct AlreadyReported;

impl std::fmt::Display for AlreadyReported {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("already reported")
    }
}

impl std::error::Error for AlreadyReported {}

fn main() -> anyhow::Result<()> {
    let mut cli = cli::Cli::parse();
    // Before anything else reads the prompt: `-p -` is stdin's, and every later consumer
    // (`overrides()`, the skill prompt, the oneshot check) wants the words rather than the dash.
    cli.read_prompt_from_stdin_if_asked()?;

    let log_level = match cli.verbosity {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    // Route tracing through `relay::RELAY` so the REPL can later install a reedline
    // `ExternalPrinter` and have warnings printed *above* the live prompt instead of racing
    // reedline's redraw. Without a printer installed (non-interactive subcommands, pre-REPL startup
    // window) the relay falls back to plain stderr.
    let rust_log = std::env::var("RUST_LOG").ok();
    tracing_subscriber::fmt()
        .with_env_filter(build_log_filter(rust_log.as_deref(), log_level))
        .with_writer(relay::RELAY.clone())
        .init();

    // Multi-threaded, and one caller depends on that rather than merely preferring it:
    // `GateToolset::resolve` answers a scheduled gate's authority question synchronously and blocks
    // its worker on an MCP snapshot read. On a current-thread runtime that would park the only
    // thread the release has to run on. Anything that narrows this needs to make that resolver
    // async first.
    let runtime = tokio::runtime::Runtime::new()?;
    let result = run_on_runtime(&runtime, cli);
    // Detach any lingering blocking threads instead of joining them on drop. `tokio::io::stdin()`
    // (used by the OAuth paste fallback) spawns a blocking worker that sits on a `read()` syscall
    // until stdin has bytes or EOF; when the user Ctrl-Cs during the wait, the future is dropped
    // but that worker can't be canceled from the outside. Without this the default `Runtime::drop`
    // joins that thread and hangs the process after a clean rollback.
    runtime.shutdown_background();

    // Ahead of the interrupt arm below, which exits without unwinding. Placed here rather than
    // duplicated into that arm because this is the funnel: every ordinary end of the process, clean
    // or interrupted, passes this line.
    crate::sandbox::release_process_grants();

    // User-initiated interrupts are already acknowledged by the rollback warn log ("interrupted;
    // rolling back …") and the shell typically echoes `^C` itself; anyhow's default "Error:
    // agent interrupted by user" on top of that is just noise. Exit with the conventional
    // SIGINT code (128 + 2) silently instead.
    if let Err(error) = &result
        && let Some(crate::error::MekaError::Interrupted) =
            error.downcast_ref::<crate::error::MekaError>()
    {
        std::process::exit(130);
    }

    // Same shape, one line further along: the host has printed this one already, so all that is
    // left of it is the status. Both arms sit below `release_process_grants` for that reason.
    if let Err(error) = &result
        && error.downcast_ref::<AlreadyReported>().is_some()
    {
        std::process::exit(1);
    }

    // A reader that stopped reading is not a failure of the command. `meka session export … | head`
    // ends the moment `head` has its line, and every tool in a pipeline is entitled to do that, so
    // the exit code says the command did what it was asked. Only for a *broken pipe*: every other
    // way a write can fail lost data nobody chose to discard, and those keep their status. One arm
    // here rather than a check per command, because there is a `print` in nearly every one of them.
    //
    // The whole chain, not the outermost error: a command returning `MekaError` hands back
    // `Io(BrokenPipe)` wrapped in it, so asking only the type anyhow is holding never matches.
    //
    // And the payload, not the kind. `BrokenPipe` says a pipe broke, not *which*, and the one that
    // earns a zero is the reader of the answer walking away. A `session export --output <fifo>`
    // whose reader leaves raises the same kind from a destination the user named, and reporting
    // success over data that never landed is worse than the crash this replaced.
    if let Err(error) = &result
        && error.chain().any(|link| {
            link.downcast_ref::<std::io::Error>()
                .and_then(std::io::Error::get_ref)
                .is_some_and(|inner| inner.downcast_ref::<render::ReaderHungUp>().is_some())
        })
    {
        return Ok(());
    }
    result
}

fn run_on_runtime(runtime: &tokio::runtime::Runtime, cli: cli::Cli) -> anyhow::Result<()> {
    // `meka acp` and `meka serve` are heavyweight (full config + credential resolution + MCP
    // setup) so they route through `async_main` rather than the lightweight subcommand block
    // below.
    let acp_mode = matches!(cli.command, Some(cli::Command::Acp));
    let serve_mode = matches!(cli.command, Some(cli::Command::Serve { .. }));

    // Handle subcommands that don't need full config resolution.
    if let Some(command) = cli.command.as_ref()
        && !acp_mode
        && !serve_mode
    {
        let cli_ref = &cli;
        return runtime.block_on(async move {
            // Read off disk rather than from a `ResolvedConfig` this path deliberately does not
            // build. Opening the store is what migrates it, and a store carried forward by `meka
            // account list` must record the same profile one carried forward by `meka` would.
            //
            // Two readers, and the difference matters. The ledger takes the one `default_profile`
            // picks, ignoring `--profile`: it stamps a profile onto every session that predates
            // meka recording one, once and irreversibly, and that must not turn on a flag the first
            // invocation after an upgrade happened to carry. `meka session import` takes the flag
            // itself beside that default and the configured profiles, and settles per run in
            // `plan_import`, where choosing per run is exactly what `--profile` is for. An
            // unreadable `config.toml` is carried to the ledger as itself rather than collapsing
            // into "nothing resolved", because the two must not produce the same write: the adopt
            // step runs once and irreversibly, so a parse error read as "nothing resolved" strands
            // every existing session against no profile with nothing said. It is not turned into a
            // hard error here, though, because `meka mcp remove` and `meka account remove` edit
            // the raw document through `toml_edit` and are how a user *repairs* such a file;
            // refusing every subcommand would close the only door out. The migration refuses
            // instead, and only when it actually has rows to stamp.
            let (installation, default_permission, context) =
                match config::default_profile_on_disk(None) {
                    Ok(adopted) => {
                        // The file is read again for its profile table, which the default reader
                        // does not hand back.
                        let configured = config::load_config_file_or_err()?.profiles;
                        let permission = config::default_permission_on_disk()?;
                        let context = store::migrations::Context::adopting(adopted.as_deref())
                            .starting_at(&permission.to_string());
                        (Some((adopted, configured)), Some(permission), context)
                    }
                    Err(error) => {
                        tracing::warn!(
                            "failed to read config.toml, so no profile can be adopted for older \
                             sessions: {error}"
                        );
                        (
                            None,
                            None,
                            store::migrations::Context::on_unreadable_config(),
                        )
                    }
                };
            let store = Store::open(None, &context).await?;
            match command {
                cli::Command::Account { action } => {
                    crate::cli::account::run(action, &store, cli_ref).await
                }
                cli::Command::Profile { action } => {
                    crate::cli::profile::run(action, &store).await
                }
                cli::Command::Session { action } => {
                    let (default, configured) = match &installation {
                        Some((default, configured)) => (default.as_deref(), Some(configured)),
                        None => (None, None),
                    };
                    let profiles = crate::store::export::ImportProfiles {
                        selected: cli_ref.profile.as_deref(),
                        default,
                        configured,
                    };
                    crate::cli::session::run_session_subcommand(
                        &store,
                        action,
                        profiles,
                        default_permission,
                    )
                    .await
                }
                cli::Command::History { action } => {
                    cli::history::run_history_subcommand(&store, action)
                }
                cli::Command::Mcp { action } => {
                    cli::mcp::run_mcp_subcommand(&store, action, cli_ref).await
                }
                cli::Command::Tool { action } => {
                    cli::tool::run_tool_subcommand(&store, action, cli_ref)
                }
                cli::Command::Skill { action } => {
                    cli::skills::run_skill_subcommand(action, cli_ref).await
                }
                cli::Command::Memory { action } => {
                    cli::memory::run_memory_subcommand(&store, action).await
                }
                cli::Command::Instructions { action } => {
                    cli::instructions::run_instructions_subcommand(action)
                }
                cli::Command::Schedule { action } => {
                    crate::cli::schedule::run(&store, action, cli_ref).await
                }
                #[allow(
                    clippy::unreachable,
                    reason = "this block is entered only when neither `acp_mode` nor `serve_mode` is set"
                )]
                cli::Command::Acp | cli::Command::Serve { .. } => {
                    unreachable!("Acp / Serve route through async_main above");
                }
            }
        });
    }

    // Refused before any setup, so a run with nothing to do never opens the store.
    if cli.oneshot && cli.prompt.is_none() && cli.skill.is_none() {
        return Err(anyhow::anyhow!(
            "`--oneshot` requires `--prompt` or `--skill`"
        ));
    }
    // Refused rather than ignored: the flag shapes what a one-shot run prints, the REPL has no use
    // for it, and a script that wrote `--format json` without `--oneshot` meant to.
    if cli.format != config::OutputFormat::Plain && !cli.oneshot {
        return Err(anyhow::anyhow!("`--format` needs `--oneshot`"));
    }

    // Refused rather than ignored. Both name *this run's session*, and a long-lived host has no
    // such thing: it creates a session per `session/new` or `POST /v1/sessions`. Accepting them
    // silently was worse than it sounds, because `-c` / `-r` set `session_resume`, which switches
    // off the default-profile check a host with no default needs most.
    //
    // `--profile` is deliberately not in this list: it selects which configured profile the host
    // defaults to, which is a property of the host rather than of one session.
    if acp_mode || serve_mode {
        let host = if acp_mode { "acp" } else { "serve" };
        let offending = [
            (cli.continue_last, "--continue"),
            (cli.resume.is_some(), "--resume"),
        ]
        .into_iter()
        .filter_map(|(given, flag)| given.then_some(flag))
        .collect::<Vec<_>>();
        if !offending.is_empty() {
            // The remedy differs by host: ACP resumes through `session/load`, and an HTTP client
            // names the session in the path.
            let remedy = if acp_mode {
                "resume one with `session/load`"
            } else {
                "address one by id under `/v1/sessions/{id}`"
            };
            anyhow::bail!(
                "`meka {host}` does not take {}; {remedy}",
                offending.join(", "),
            );
        }
    }

    let mut config = ResolvedConfig::resolve(cli.overrides());

    // If --skill is set, validate and render the body upfront so an invalid name fails fast
    // before any session/MCP setup. The combined string (extra + body, mirroring the REPL's `/skill
    // <name> [extra...]`) then takes the place of cli.prompt as the first-turn input. Resolved
    // config comes first because `[skills] extra_paths` decides which roots the lookup sees.
    let skill_prompt = runtime.block_on(host::build_skill_prompt(
        cli.skill.as_deref(),
        cli.prompt.as_deref(),
        &config.skill_roots(),
    ))?;

    if let Some(prompt) = skill_prompt {
        config.request.prompt = Some(prompt);
    }
    // `--bind` on `meka serve` overrides the config-file `[serve].bind`. Apply here so
    // `async_main` sees a single resolved binding without re-parsing the CLI.
    if let Some(cli::Command::Serve { bind: Some(bind) }) = cli.command.as_ref() {
        config.request.serve_bind_override = Some(bind.clone());
    }
    // Before anything renders. The renderers read this rather than taking it as a parameter because
    // the approval prompt sits several call sites below `run_repl`, which takes flat scalars; the
    // functions that compose a line still take an explicit width, so tests never touch it.
    render::set_max_width(config.max_width);
    runtime.block_on(async_main(config, acp_mode, serve_mode))
}

/// Build the `tracing` filter for meka.
///
/// `RUST_LOG` is honored verbatim. Otherwise the `-v` level is the floor, with two rmcp log sites
/// that fire on every retry quieted:
///
/// 1. An MCP server behind a CDN closes idle HTTP streams after about 100 s, which trips
///    `rmcp::transport::common::client_side_sse`'s `warn!` before rmcp reconnects on its own via
///    `Last-Event-ID`. The real failure ("max retry times reached") is an `error!` from the same
///    module, so an `=error` floor keeps it and drops the noise.
/// 2. `rmcp::transport::worker` logs an `error!` each time a transport fails to come up, and an
///    unreachable server is retried for the life of the process, so that lands on the prompt every
///    few minutes saying nothing `record_connect_failure` has not already said once. Silenced
///    outright, because the noise is at the error level itself.
///
/// Verified against rmcp 2.1.
fn build_log_filter(rust_log: Option<&str>, log_level: &str) -> tracing_subscriber::EnvFilter {
    use tracing_subscriber::EnvFilter;
    if let Some(value) = rust_log
        && let Ok(filter) = EnvFilter::try_new(value)
    {
        return filter;
    }
    #[allow(
        clippy::expect_used,
        reason = "a compile-time literal in a known-good shape"
    )]
    let sse = "rmcp::transport::common::client_side_sse=error"
        .parse()
        .expect("valid tracing directive");
    #[allow(
        clippy::expect_used,
        reason = "a compile-time literal in a known-good shape"
    )]
    let worker = "rmcp::transport::worker=off"
        .parse()
        .expect("valid tracing directive");
    EnvFilter::new(log_level)
        .add_directive(sse)
        .add_directive(worker)
}

async fn async_main(
    config: ResolvedConfig,
    acp_mode: bool,
    serve_mode: bool,
) -> anyhow::Result<()> {
    // Before the store is opened or a credential resolved, so a misconfigured profile is reported
    // as such rather than as a credential error.
    config.validate()?;

    // The file's default rather than this run's level: a row the migration stamps is read by every
    // later process, and a `--permission` given once must not become what old sessions run at.
    let store = Store::open(
        None,
        &store::migrations::Context::adopting(config.configured_default_profile.as_deref())
            .starting_at(&config::default_permission_on_disk()?.to_string()),
    )
    .await?;
    let token_store = store.token_store();

    // The long-lived hosts sweep here: neither resumes one particular session at startup, so there
    // is no target to protect. The REPL and `--oneshot` sweep only once `resolve_session_resume`
    // holds the lock on the session they were asked for; see `sweep_expired_sessions`.
    if serve_mode || acp_mode {
        host::sweep_expired_sessions(&config, &store).await?;
    }

    let mcp_context = mcp::McpClientContext::new();
    let mcp_manager = if !config.mcp_servers.is_empty() {
        let manager = mcp::McpClientManager::prepare(
            &config.mcp_servers,
            config.mcp_default_permission,
            Some(token_store.clone()),
            Arc::clone(&mcp_context),
        )
        .await?;
        mcp_context.set_manager(Arc::downgrade(&manager));
        Some(manager)
    } else {
        None
    };

    // `meka acp` and `meka serve` reuse every step above (credential resolution, MCP setup,
    // session-manager housekeeping) and then enter their respective transport loops instead of
    // the REPL.
    if serve_mode {
        #[cfg(feature = "serve")]
        return host::http::run_serve(config, store, mcp_manager).await;
        #[cfg(not(feature = "serve"))]
        {
            let _ = (config, store, mcp_manager);
            anyhow::bail!("this meka was built without the `serve` feature, so it has no HTTP API");
        }
    }
    if acp_mode {
        return host::acp::run_acp(config, store, mcp_manager).await;
    }

    if config.request.oneshot {
        // Startup already refused `--oneshot` without a prompt or `--skill`; this is the same
        // refusal for a request that arrived by another route, in place of a panic.
        let Some(prompt) = config.request.prompt.clone() else {
            anyhow::bail!("`--oneshot` requires `--prompt` or `--skill`");
        };
        return host::oneshot::run_oneshot(config, store, prompt, mcp_manager).await;
    }

    let initial_prompt = config.request.prompt.clone();
    host::repl::run_interactive(config, store, token_store, initial_prompt, mcp_manager).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{terminal::*, *};

    /// A host that leaves through one of its early `?` paths still flushes what it had streamed and
    /// still settles the row.
    ///
    /// Both halves of what the guard is for, and the second is the one with a screen behind it:
    /// `repl_handle.await?` fires when the REPL thread panicked, which can be mid-wake with the
    /// prompt it broke out of still drawn.
    ///
    /// Only the mechanism is pinned here. Reaching those paths for real needs a failing provider
    /// registry or a panicking REPL thread behind a terminal, and an assertion the guard's absence
    /// cannot fail is not coverage.
    #[test]
    fn the_last_episode_closes_however_the_host_leaves() {
        let console = Arc::new(std::sync::Mutex::new(console::Console::new(
            console::Spacing {
                newline_before_prompt: true,
                newline_after_prompt: true,
            },
            crate::config::RenderMode::Raw,
        )));
        with_console(&console, |console| {
            console.open_episode(console::RowState::Empty, console::Neighbor::Shell)
        });
        {
            let _last_episode = LastEpisode(Arc::clone(&console));
            with_console(&console, |console| {
                console.text_delta("half an answer");
                // The drawing API cannot reach a parked prompt without a terminal. A wake that
                // streamed would have settled the row first, so this pairs a parked prompt with an
                // open block to exercise both halves of the close at once rather than to reproduce
                // one state the REPL reaches.
                console.force_row(console::RowState::PromptParked);
            });
            assert!(
                with_console(&console, |console| console.has_open_text()),
                "the block has to be open, or the guard has nothing to close"
            );
        }
        assert!(
            !with_console(&console, |console| console.has_open_text()),
            "leaving the host's scope closes it, whether or not the host reached its own close"
        );
        assert_eq!(
            with_console(&console, |console| console.row()),
            console::RowState::Empty,
            "and the stale prompt goes with it, rather than sitting under the shell's"
        );
    }

    /// The Ctrl+C ladder, which nothing else can reach.
    ///
    /// `install_interrupt_handler` is a spawned listener on `tokio::signal::ctrl_c()` ending in
    /// `std::process::exit(130)`, so no test drives it; the four mutants that survived the sweep
    /// all lived in these two decisions. Both are behavior: collapsing the second press into the
    /// third makes Ctrl+C Ctrl+C kill the process instead of the background tasks, which is the
    /// unrecoverable outcome the ladder exists to put one more keystroke in front of.
    #[test]
    fn the_interrupt_ladder_escalates_one_press_at_a_time() {
        assert_eq!(
            escalation_for(1),
            Escalation::CancelTurn,
            "the first press is the shell's contract: the foreground job, and nothing else"
        );
        assert_eq!(
            escalation_for(2),
            Escalation::CancelBackgroundTasks,
            "the second stops the background work, and is the rung whose absence would make the \
             second press fatal"
        );
        for press in [3, 4, 99] {
            assert_eq!(
                escalation_for(press),
                Escalation::Leave,
                "press {press} is past the ladder and leaves"
            );
        }
        // Zero is unreachable -- the counter is incremented before this is asked -- so what matters
        // is that it does not land on a rung, not which one it picks.
        assert_eq!(escalation_for(0), Escalation::Leave);
    }

    /// Nothing is announced when nothing was stopped, and the plural agrees with the count.
    #[test]
    fn the_background_cancellation_notice_counts_what_it_stopped() {
        assert_eq!(
            background_cancellation_notice(0),
            None,
            "a second press with nothing running must not claim to have stopped anything"
        );
        assert_eq!(
            background_cancellation_notice(1).as_deref(),
            Some("stopping 1 background task")
        );
        assert_eq!(
            background_cancellation_notice(4).as_deref(),
            Some("stopping 4 background tasks")
        );
    }

    /// Both arms, against real rows rather than a fabricated `parent_id`.
    ///
    /// The refusal is what keeps a sub-agent's spawn terms meaningful, so a predicate that answered
    /// wrongly in either direction is serious in both directions: admitting a sub-agent reopens the
    /// escalation, and refusing a root session would break every host at once.
    ///
    /// `None` is asserted too, because that is what every fresh session passes and a check that
    /// tried to read a row for it would refuse the case it exists to allow.
    #[tokio::test]
    async fn a_session_another_one_spawned_cannot_be_built_as_a_plain_agent() {
        let manager = Store::for_test().await;
        let parent = manager
            .create_session(None, "profile")
            .await
            .expect("a root session");
        let (sub_agent, _lock) = manager
            .create_child_session(
                parent,
                None,
                Vec::new(),
                None,
                "read".to_string(),
                "profile".to_string(),
            )
            .await
            .expect("a sub-agent of that session");

        refuse_a_spawned_session(&manager, None)
            .await
            .expect("a session that does not exist yet has nothing to refuse");
        refuse_a_spawned_session(&manager, Some(parent))
            .await
            .expect("a root session is the ordinary case and must be admitted");

        let error = refuse_a_spawned_session(&manager, Some(sub_agent))
            .await
            .expect_err("a session with a parent cannot be built as a plain agent");
        assert!(
            matches!(
                error.downcast_ref::<crate::error::MekaError>(),
                Some(crate::error::MekaError::SessionNotDrivable(_))
            ),
            "the refusal has to be the variant the HTTP layer answers 422 for: {error}"
        );
        // Both ids and the way through, because a client handed a sub-agent's id may not know what
        // spawned it, and "no" without a remedy sends it looking for a bug in meka.
        let text = error.to_string();
        assert!(
            text.contains(&sub_agent.to_string())
                && text.contains(&parent.to_string())
                && text.contains("agent_followup"),
            "the refusal must name the sub-agent, its parent, and the door that can: {text}"
        );
    }

    /// A sub-agent whose parent did not survive an import is still a sub-agent.
    ///
    /// `session export` on a sub-agent alone writes a `parent_id` pointing outside the archive, and
    /// `import_sessions` resolves an unknown parent to `NULL` while copying `subagent_spec_json`
    /// verbatim. Keying the refusal on the parent alone therefore left export-then-import as a
    /// two-command promotion of a sub-agent into a drivable root session -- the same laundering
    /// `fork_session` was fixed for, one door over, and reproducible with a real shell.
    ///
    /// The spec is the fact worth refusing on: a row holding the terms another session spawned it
    /// under is a sub-agent's conversation whether or not the link survived.
    #[tokio::test]
    async fn spawn_terms_without_a_parent_are_still_a_sub_agent() {
        let manager = Store::for_test().await;
        // Written through the real importer, because that is the only door that produces this
        // shape: `plan_import` resolves a parent outside the archive to `None` and carries
        // `subagent_spec_json` regardless.
        let orphaned = uuid::Uuid::new_v4();
        manager
            .import_sessions(
                vec![crate::store::ImportSessionRecord {
                    new_id: orphaned,
                    new_parent_id: None,
                    created_at: "2026-08-31T00:00:00Z".to_string(),
                    cwd: None,
                    permission: crate::permission::Permission::Read,
                    approvals: false,
                    capabilities_json: None,
                    additional_roots: Vec::new(),
                    subagent_spec_json: Some("{\"tools\":[]}".to_string()),
                    profile: "profile".to_string(),
                    stats: Default::default(),
                    events: Vec::new(),
                    tool_outputs: Vec::new(),
                }],
                Vec::new(),
            )
            .await
            .expect("the archive a lone sub-agent exports to");

        let error = refuse_a_spawned_session(&manager, Some(orphaned))
            .await
            .expect_err("spawn terms with no parent are what an imported sub-agent looks like");
        assert!(
            matches!(
                error.downcast_ref::<crate::error::MekaError>(),
                Some(crate::error::MekaError::SessionNotDrivable(_))
            ),
            "the same refusal the parented case gets: {error}"
        );
        let text = error.to_string();
        assert!(
            text.contains(&orphaned.to_string()) && !text.contains("agent_followup"),
            "there is no parent to point at, so it must not name a door that is not there: {text}"
        );
    }

    /// `/cd` reaches the row, which is what makes the recorded directory mean "where the session
    /// is" rather than "where it was created". Nothing else covers this: the REPL loop that sends
    /// the event needs a terminal, so a test cannot drive `/cd` itself.
    #[tokio::test]
    async fn a_cd_records_the_directory_on_the_session_row() {
        let manager = Store::for_test().await;
        let temp = tempfile::tempdir().expect("tempdir");
        let moved = crate::workspace::canonical_for_test(temp.path());
        let id = manager
            .create_session(
                Some(std::path::PathBuf::from("/somewhere/else")),
                "p".to_string(),
            )
            .await
            .expect("create");

        record_session_cwd(&manager, Some(id), &moved).await;

        let recorded = manager
            .session_info(id)
            .await
            .expect("read the row")
            .and_then(|info| info.cwd);
        assert_eq!(
            recorded,
            Some(moved),
            "the row must say where `/cd` moved the session",
        );
    }

    /// Before the first turn there is no row to correct: the directory the creation snapshot reads
    /// is the cell `/cd` has already written. So this is a no-op, and in particular it must not
    /// reach for some other session's row -- a REPL sharing a store with a `meka serve` has plenty
    /// to choose from.
    #[tokio::test]
    async fn a_cd_before_the_first_turn_writes_nobody_elses_row() {
        let manager = Store::for_test().await;
        let temp = tempfile::tempdir().expect("tempdir");
        let bystander = manager
            .create_session(
                Some(std::path::PathBuf::from("/its/own/place")),
                "p".to_string(),
            )
            .await
            .expect("create");

        record_session_cwd(&manager, None, temp.path()).await;

        let untouched = manager
            .session_info(bystander)
            .await
            .expect("read the row")
            .and_then(|info| info.cwd);
        assert_eq!(
            untouched,
            Some(std::path::PathBuf::from("/its/own/place")),
            "a `/cd` with no session of its own must leave every other row alone",
        );
    }

    /// A resumed session opens where it was recorded. At `workspace` that directory is also the
    /// writable boundary, so adopting the shell's would widen it behind the user's back.
    #[test]
    fn a_resume_prefers_the_recorded_directory_over_the_launch_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let recorded = temp.path().join("project");
        let launch = temp.path().join("elsewhere");
        std::fs::create_dir_all(&recorded).expect("recorded dir");
        std::fs::create_dir_all(&launch).expect("launch dir");

        assert_eq!(
            resume_working_directory(Some(recorded.clone()), &launch, None),
            recorded,
        );
    }

    /// The two cases that have to keep the run going: a row carrying no directory (an imported
    /// archive may omit it) and one naming a directory that has since been removed.
    #[test]
    fn a_resume_falls_back_to_the_launch_directory_when_the_recording_cannot_serve() {
        let temp = tempfile::tempdir().expect("tempdir");
        let launch = temp.path().join("elsewhere");
        std::fs::create_dir_all(&launch).expect("launch dir");

        assert_eq!(resume_working_directory(None, &launch, None), launch);
        assert_eq!(
            resume_working_directory(Some(temp.path().join("gone")), &launch, None),
            launch,
        );
        // A file is not a directory to open in either, and `is_dir` is what separates them.
        let file = temp.path().join("a-file");
        std::fs::write(&file, b"x").expect("write file");
        assert_eq!(resume_working_directory(Some(file), &launch, None), launch);
    }

    /// The regression the whole feature exists for, at the one door every turn-running site goes
    /// through: a session that named a profile keeps it, whatever the process default now is.
    #[tokio::test]
    async fn a_recorded_binding_beats_the_process_default() {
        let manager = Store::for_test().await;
        let id = manager
            .create_session(None, "openaiprof".to_string())
            .await
            .expect("create");

        let resolved = provider::resolve_session_profile(&manager, Ok("claudeprof"), Some(id))
            .await
            .expect("resolves");

        assert_eq!(resolved, "openaiprof");
    }

    /// A session that does not exist yet is the only case the configured default answers.
    #[tokio::test]
    async fn a_session_that_does_not_exist_yet_takes_the_default() {
        let manager = Store::for_test().await;

        let resolved = provider::resolve_session_profile(&manager, Ok("claudeprof"), None)
            .await
            .expect("resolves");

        assert_eq!(resolved, "claudeprof");
    }

    /// With nothing configured there is no profile to fall back to, and inventing one would be the
    /// silent redirection this door exists to prevent.
    #[tokio::test]
    async fn no_configured_profile_is_an_error_rather_than_an_empty_one() {
        let manager = Store::for_test().await;

        let error = provider::resolve_session_profile(
            &manager,
            Err("no profiles configured. Run `meka profile add <name>`."),
            None,
        )
        .await
        .expect_err("nothing to resolve to");

        assert!(
            error.to_string().contains("meka profile add"),
            "the refusal should say how to fix it: {error}"
        );
    }

    /// The reason travels: `validate()` raises no ambiguous default for a resume, so a resume that
    /// *does* fall through to needing one has to carry the message that says what to do rather
    /// than a generic "nothing configured".
    #[tokio::test]
    async fn a_resume_that_needs_a_default_reports_why_there_is_none() {
        let manager = Store::for_test().await;
        let ambiguous = "multiple profiles configured (personal, side); run \
                         `meka profile use <name>` to pick a default, or pass `--profile <name>`.";

        // `None` session id: `-c` on a store with nothing to resume lands here.
        let error = provider::resolve_session_profile(&manager, Err(ambiguous), None)
            .await
            .expect_err("no default to fall back to");

        let error::MekaError::Config(message) = error else {
            panic!("a missing default is a configuration error: {error}");
        };
        assert_eq!(message, ambiguous);
    }

    /// Zero days means "not updated since this instant", i.e. everything. Easy to type when you
    /// meant "today's", and unrecoverable, so it is refused rather than run.
    #[tokio::test]
    async fn delete_older_than_zero_days_is_refused() {
        let manager = Store::for_test().await;
        let error = crate::cli::session::delete_sessions(&manager, &[], false, Some(0))
            .await
            .expect_err("zero must be refused");
        assert!(error.to_string().contains("--all"), "{error}");
    }

    /// The flag has to reach `delete_expired_sessions(days)` and nothing else: routing it to
    /// `delete_all_sessions` would pass every error-path test in this file while wiping the DB.
    #[tokio::test]
    async fn delete_older_than_days_deletes_only_the_old() {
        let manager = Store::for_test().await;
        let old = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create old");
        let recent = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create recent");
        let backdated = (chrono::Utc::now() - chrono::TimeDelta::days(100)).to_rfc3339();
        manager
            .set_session_updated_at_for_test(old, &backdated)
            .await
            .expect("backdate");

        crate::cli::session::delete_sessions(&manager, &[], false, Some(30))
            .await
            .expect("sweep");

        assert!(!manager.session_exists(old).await.expect("exists"));
        assert!(manager.session_exists(recent).await.expect("exists"));
    }

    /// No selector at all should say what the options are, not silently do nothing.
    #[tokio::test]
    async fn delete_with_no_selector_explains_itself() {
        let manager = Store::for_test().await;
        let error = crate::cli::session::delete_sessions(&manager, &[], false, None)
            .await
            .expect_err("no selector must be an error");
        let text = error.to_string();
        assert!(text.contains("--older-than-days"), "{text}");
        assert!(text.contains("--all"), "{text}");
    }

    /// Both directives exist to stop a retried MCP connect from repeating itself on the user's
    /// prompt. `RUST_LOG` must still win outright, or there is no way to see them when debugging.
    #[test]
    fn log_filter_quiets_rmcp_retry_noise() {
        let filter = build_log_filter(None, "warn").to_string();
        assert!(
            filter.contains("rmcp::transport::worker=off"),
            "the per-attempt transport error must be silenced: {filter}"
        );
        assert!(
            filter.contains("rmcp::transport::common::client_side_sse=error"),
            "the per-reconnect sse warning must be floored: {filter}"
        );

        let overridden = build_log_filter(Some("rmcp=debug"), "warn").to_string();
        assert!(
            !overridden.contains("rmcp::transport::worker=off"),
            "RUST_LOG must replace the defaults wholesale: {overridden}"
        );
    }
    #[test]
    fn parents_first_order_orders_parents_before_children() {
        // Given out of order (child, root, middle), each node must land after its parent.
        let nodes = vec![
            ("c".to_string(), Some("b".to_string())),
            ("a".to_string(), None),
            ("b".to_string(), Some("a".to_string())),
        ];
        let order = crate::store::export::parents_first_order(&nodes).expect("order");
        let position = |id: &str| order.iter().position(|&i| nodes[i].0 == id).unwrap();
        assert!(position("a") < position("b"));
        assert!(position("b") < position("c"));
    }

    #[test]
    fn parents_first_order_treats_external_parent_as_root() {
        // A parent absent from the set (e.g. the exported root was itself a sub-agent) is not an
        // error; the node is ordered as a root.
        let nodes = vec![("only".to_string(), Some("outside".to_string()))];
        assert_eq!(
            crate::store::export::parents_first_order(&nodes).expect("order"),
            vec![0]
        );
    }

    fn user_msg(text: &str) -> crate::conversation::Message {
        crate::conversation::Message::user(text)
    }

    fn assistant_text(text: &str) -> crate::conversation::Message {
        crate::conversation::Message::assistant_text(text)
    }

    fn assistant_tool_use(id: &str, name: &str) -> crate::conversation::Message {
        crate::conversation::Message {
            role: crate::conversation::Role::Assistant,
            content: vec![crate::conversation::ContentBlock::ToolUse {
                id: id.to_string(),
                name: name.to_string(),
                input: serde_json::json!({}),
            }],
        }
    }

    fn tool_result(tool_use_id: &str) -> crate::conversation::Message {
        crate::conversation::Message {
            role: crate::conversation::Role::User,
            content: vec![crate::conversation::ContentBlock::ToolResult {
                tool_use_id: tool_use_id.to_string(),
                content: vec![crate::conversation::ToolResultContent::Text {
                    text: "ok".to_string(),
                }],
                is_error: false,
            }],
        }
    }

    fn build_log(messages: Vec<crate::conversation::Message>) -> conversation::Conversation {
        conversation::Conversation::from_vec(messages)
    }

    #[test]
    fn validate_valid_chain() {
        let mut log = build_log(vec![
            user_msg("hello"),
            assistant_tool_use("c1", "read_file"),
            tool_result("c1"),
            assistant_text("done"),
        ]);
        let dropped = log.sanitize_orphans();
        assert!(dropped.is_empty());
        assert_eq!(log.len(), 4);
    }

    #[test]
    fn validate_orphaned_tool_use_dropped() {
        let mut log = build_log(vec![
            user_msg("hello"),
            assistant_tool_use("c1", "read_file"),
            // Missing tool_result for c1
            assistant_text("done"),
        ]);
        let dropped = log.sanitize_orphans();
        assert_eq!(dropped.len(), 1);
        assert_eq!(log.len(), 2);
        let view = log.as_slice();
        assert_eq!(view[0].role, crate::conversation::Role::User);
        assert_eq!(view[1].role, crate::conversation::Role::Assistant);
        assert_eq!(view[1].text_content(), "done");
    }

    #[test]
    fn validate_orphaned_at_end() {
        let mut log = build_log(vec![
            user_msg("hello"),
            assistant_tool_use("c1", "read_file"),
        ]);
        log.sanitize_orphans();
        assert_eq!(log.len(), 1);
        assert_eq!(log.as_slice()[0].text_content(), "hello");
    }

    #[test]
    fn validate_mismatched_ids() {
        let mut log = build_log(vec![
            user_msg("hello"),
            assistant_tool_use("c1", "read_file"),
            tool_result("c2"), // Wrong ID
        ]);
        log.sanitize_orphans();
        // The assistant message is dropped because c1 has no matching result.
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn validate_text_only_preserved() {
        let mut log = build_log(vec![
            user_msg("hello"),
            assistant_text("hi"),
            user_msg("bye"),
        ]);
        log.sanitize_orphans();
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn validate_multiple_chains() {
        let mut log = build_log(vec![
            user_msg("start"),
            assistant_tool_use("c1", "read_file"),
            tool_result("c1"),
            assistant_tool_use("c2", "write_file"),
            // Missing tool_result for c2
            assistant_text("done"),
        ]);
        log.sanitize_orphans();
        // c2 should be dropped, rest preserved.
        assert_eq!(log.len(), 4);
        assert_eq!(log.as_slice()[3].text_content(), "done");
    }

    // -- log filter --

    /// The default filter (no `RUST_LOG`) floors rmcp's SSE-reconnect module at `error`.
    #[test]
    fn default_log_filter_downgrades_rmcp_sse_warns() {
        let rendered = format!("{}", build_log_filter(None, "warn"));
        assert!(
            rendered.contains("rmcp::transport::common::client_side_sse=error"),
            "expected SSE-reconnect target to be floored at `error` in the default \
             filter, got: {rendered}"
        );
    }

    /// `RUST_LOG` is honored verbatim, or there is no way to see rmcp's internals when debugging.
    #[test]
    fn explicit_rust_log_is_not_overridden() {
        let rendered = format!("{}", build_log_filter(Some("rmcp=debug"), "warn"));
        assert!(
            !rendered.contains("rmcp::transport::common::client_side_sse=error"),
            "explicit RUST_LOG must not be augmented; got: {rendered}"
        );
        assert!(
            rendered.contains("rmcp=debug"),
            "user's RUST_LOG should pass through unchanged; got: {rendered}"
        );
    }

    #[test]
    fn export_without_compaction_renders_plain_turns() {
        let mut log = conversation::Conversation::new();
        log.append(user_msg("hello"));
        log.append(assistant_text("hi there"));
        let markdown = crate::conversation::format_session_as_markdown(
            uuid::Uuid::nil(),
            log.events(),
            &std::collections::HashMap::new(),
        );
        assert!(markdown.contains("## User") && markdown.contains("hello"));
        assert!(markdown.contains("## Assistant") && markdown.contains("hi there"));
        // No compaction happened, so no boundary marker.
        assert!(!markdown.contains("Session compaction"));
    }
}
