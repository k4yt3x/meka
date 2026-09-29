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
    // `meka confine` is the inside of the Bubblewrap sandbox: bwrap execs meka so the Landlock
    // ruleset can be enacted after its mounts, and meka then becomes the command. Ahead of tracing
    // and the runtime, because `landlock_restrict_self` binds the calling thread and `execve`
    // replaces the process, so nothing started here would outlive it either way.
    #[cfg(target_os = "linux")]
    if let Some(cli::Command::Confine {
        writable,
        scratch,
        command,
    }) = &cli.command
    {
        let error = crate::sandbox::run_confined(writable, scratch, command);
        eprintln!("meka confine: failed to run the command inside the sandbox: {error}");
        std::process::exit(126);
    }
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

    // Refused rather than ignored, ahead of every subcommand: the flag shapes what a one-shot run
    // prints and nothing else reads it, so a script that wrote `--format json` ahead of
    // `profile list` meant the subcommand's own `--format`.
    if cli.format != config::OutputFormat::Plain && !cli.oneshot {
        return Err(anyhow::anyhow!("`--format` needs `--oneshot`"));
    }

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
                            "failed to read config.toml, so its settings are ignored and no \
                             profile is adopted for older sessions: {error}"
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
                #[cfg(target_os = "linux")]
                cli::Command::Confine { .. } => {
                    anyhow::bail!("`confine` is handled before startup")
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
}
