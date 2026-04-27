//! `agsh` — an agentic shell where you describe what you want in natural
//! language and an LLM-backed agent decides which tools to run.
//!
//! The binary wires together: a [`provider`] (Claude or OpenAI), a [`session`]
//! store backed by SQLite, a [`tools`] registry, an MCP client manager, and a
//! [`repl`] input loop. The [`agent`] module owns the per-turn loop that streams
//! provider output and dispatches tool calls.

mod agent;
mod cli;
mod config;
mod context;
mod error;
mod image;
mod mcp;
mod permission;
mod provider;
mod render;
mod repl;
mod sandbox;
mod session;
mod setup;
mod skills;
mod tools;

use clap::Parser;
use tokio_util::sync::CancellationToken;

use std::sync::Arc;

use crate::agent::{Agent, AgentOptions};
use crate::config::ResolvedConfig;
use crate::permission::SharedPermission;
use crate::provider::{AuthCredential, ProviderBuilder};
use crate::repl::ReplEvent;
use crate::session::{SessionManager, TokenStore};
use crate::tools::ToolRegistry;

fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();

    let log_level = match cli.verbosity {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(build_log_filter(log_level))
        .with_writer(std::io::stderr)
        .init();

    let runtime = tokio::runtime::Runtime::new()?;
    let result = run_on_runtime(&runtime, cli);
    // Detach any lingering blocking threads instead of joining them on
    // drop. `tokio::io::stdin()` (used by the OAuth paste fallback)
    // spawns a blocking worker that sits on a `read()` syscall until
    // stdin has bytes or EOF; when the user Ctrl-Cs during the wait,
    // the future is dropped but that worker can't be cancelled from
    // the outside. Without this the default `Runtime::drop` joins that
    // thread and hangs the process after a clean rollback.
    runtime.shutdown_background();

    // User-initiated interrupts are already acknowledged by the rollback
    // warn log ("interrupted — rolling back …") and the shell typically
    // echoes `^C` itself; anyhow's default "Error: agent interrupted by
    // user" on top of that is just noise. Exit with the conventional
    // SIGINT code (128 + 2) silently instead.
    if let Err(error) = &result
        && let Some(crate::error::AgshError::Interrupted) =
            error.downcast_ref::<crate::error::AgshError>()
    {
        std::process::exit(130);
    }
    result
}

fn run_on_runtime(runtime: &tokio::runtime::Runtime, cli: cli::Cli) -> anyhow::Result<()> {
    // Handle subcommands that don't need full config resolution.
    if cli.command.is_some() {
        let cli_ref = &cli;
        return runtime.block_on(async move {
            let session_manager = SessionManager::open(None).await?;
            let command = cli_ref.command.as_ref().expect("checked above");
            match command {
                cli::Command::Setup => {
                    let token_store = session_manager.token_store();
                    setup::run_setup(&token_store).await
                }
                cli::Command::Export { session_id, output } => {
                    export_session(&session_manager, *session_id, output.as_deref()).await
                }
                cli::Command::Delete { session_ids, all } => {
                    delete_sessions(&session_manager, session_ids, *all).await
                }
                cli::Command::List { limit } => list_sessions(&session_manager, *limit).await,
                cli::Command::Mcp { action } => {
                    run_mcp_subcommand(&session_manager, action, cli_ref).await
                }
                cli::Command::Tools { action } => run_tools_subcommand(action, cli_ref).await,
            }
        });
    }

    // Auto-detect first launch: no config file and no env-based provider.
    if !config::config_file_exists() && std::env::var("AGSH_PROVIDER").is_err() {
        runtime.block_on(async {
            let session_manager = SessionManager::open(None).await?;
            let token_store = session_manager.token_store();
            setup::run_setup(&token_store).await
        })?;
    }

    let config = ResolvedConfig::from_cli(&cli);
    runtime.block_on(async_main(config))
}

/// Build the `tracing` filter for agsh.
///
/// When the user sets `RUST_LOG` we honour it verbatim — no hidden
/// overrides, so debugging with `RUST_LOG=rmcp=debug` works as expected.
/// Otherwise we start from `log_level` (derived from `-v` / `-vv`) and
/// add a single directive that downgrades rmcp's SSE-reconnect warning
/// to `error`:
///
/// MCP servers behind a CDN / edge (Cloudflare, Fastly, …) close idle
/// HTTP streams after ~100 s, which trips
/// `rmcp::transport::common::client_side_sse`'s `warn!("sse stream
/// error: …")` before rmcp transparently reconnects via `Last-Event-ID`.
/// The warn fires on every expected reconnect; the real failure mode
/// (`"max retry times reached"`) is emitted at `error!` from the same
/// module, so an `=error` floor keeps the useful signal and drops the
/// noise. Verified against rmcp 1.5.
fn build_log_filter(log_level: &str) -> tracing_subscriber::EnvFilter {
    use tracing_subscriber::EnvFilter;
    if let Ok(filter) = EnvFilter::try_from_default_env() {
        return filter;
    }
    EnvFilter::new(log_level).add_directive(
        "rmcp::transport::common::client_side_sse=error"
            .parse()
            .expect("valid tracing directive"),
    )
}

async fn async_main(mut config: ResolvedConfig) -> anyhow::Result<()> {
    // Validate provider name and model before opening the session store or
    // resolving credentials so the user sees a clear "not configured" or
    // "invalid value" message instead of the downstream credential error.
    config.validate()?;

    let session_manager = SessionManager::open(None).await?;
    let token_store = session_manager.token_store();

    if let Some(retention_days) = config.retention_days {
        let deleted = session_manager
            .delete_expired_sessions(retention_days)
            .await?;
        if deleted > 0 {
            tracing::info!("deleted {} expired sessions", deleted);
        }
    }

    if let Some(max_bytes) = config.max_storage_bytes {
        let deleted = session_manager.enforce_storage_limit(max_bytes).await?;
        if deleted > 0 {
            tracing::info!("deleted {} sessions to enforce storage limit", deleted);
        }
    }

    // If no credential from env/config, try loading from database.
    // Storage key stays "claude" across the rename to "claude-oauth" so existing
    // users keep their tokens; "openai-codex" matches the provider name.
    let oauth_storage_key = match config.provider_name.as_deref() {
        Some("claude-oauth") => Some("claude"),
        Some("openai-codex") => Some(crate::provider::openai::codex::STORAGE_KEY),
        _ => None,
    };
    if config.auth_credential.is_none()
        && let Some(storage_key) = oauth_storage_key
    {
        match token_store.load_oauth_token(storage_key).await {
            Ok(Some(credential)) => {
                tracing::info!("loaded OAuth token from database");
                config.auth_credential = Some(credential);
            }
            Ok(None) => {}
            Err(error) => {
                tracing::warn!("failed to load OAuth token from database: {}", error);
            }
        }
    }

    // Save OAuth token from env/config to database for future use, so the
    // refresh path has a place to land updated tokens.
    if let Some(credential @ AuthCredential::OAuthToken { .. }) = &config.auth_credential
        && let Some(storage_key) = oauth_storage_key
        && let Err(error) = token_store.save_oauth_token(storage_key, credential).await
    {
        tracing::warn!("failed to save OAuth token to database: {}", error);
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

    if let Some(prompt) = config.prompt.clone() {
        return run_oneshot(
            config,
            session_manager,
            token_store,
            prompt,
            mcp_manager,
            mcp_context,
        )
        .await;
    }

    run_interactive(
        config,
        session_manager,
        token_store,
        mcp_manager,
        mcp_context,
    )
    .await
}

// Top-level entry point for assembling the agent; splitting its inputs
// further would force callers to pre-bundle unrelated collaborators
// (config, session manager, permission mode, credential, MCP plumbing,
// approval channel) just to appease the arg-count lint.
#[allow(clippy::too_many_arguments)]
async fn create_agent_from_config(
    config: &ResolvedConfig,
    session_manager: SessionManager,
    shared_permission: SharedPermission,
    token_store: TokenStore,
    credential: AuthCredential,
    mcp_manager: Option<&Arc<mcp::McpClientManager>>,
    mcp_context: Option<&Arc<mcp::McpClientContext>>,
    approval_sender: Option<std::sync::mpsc::Sender<repl::AgentToReplEvent>>,
) -> anyhow::Result<Agent> {
    config.validate()?;

    let provider_name = config
        .provider_name
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("provider_name missing after validation"))?;
    let needs_token_store = matches!(credential, AuthCredential::OAuthToken { .. });

    let model = config
        .model
        .clone()
        .ok_or_else(|| anyhow::anyhow!("model missing after validation"))?;
    let provider = ProviderBuilder::new(provider_name, credential, model)
        .base_url(config.base_url.clone())
        .client_id(config.client_id.clone())
        .oauth_token_url(config.oauth_token_url.clone())
        .token_store(if needs_token_store {
            Some(Arc::new(token_store))
        } else {
            None
        })
        .thinking(config.thinking_enabled, config.thinking_budget_tokens)
        .reasoning_effort(config.reasoning_effort.clone())
        .device_id(config.device_id.clone())
        .effort(config.effort.clone())
        .redact_thinking(config.redact_thinking)
        .build()?;

    let sandbox_capability = crate::sandbox::detect();
    let sandboxed_shell = config.sandbox
        && !matches!(
            sandbox_capability,
            crate::sandbox::SandboxCapability::Unavailable
        );

    let todo_list: crate::tools::todo::SharedTodoList =
        std::sync::Arc::new(tokio::sync::RwLock::new(Vec::new()));

    let shared_session_id: std::sync::Arc<tokio::sync::RwLock<Option<uuid::Uuid>>> =
        std::sync::Arc::new(tokio::sync::RwLock::new(None));

    let builtin_filter = crate::tools::BuiltinToolFilter::from_config(
        config.builtin_allowed_tools.clone(),
        config.builtin_disabled_tools.clone(),
        config.builtin_tool_permissions.clone(),
    );

    let tool_registry = ToolRegistry::build_default(
        config.web_client.clone(),
        shared_permission.clone(),
        config.sandbox,
        sandbox_capability,
        todo_list.clone(),
        session_manager.clone(),
        shared_session_id.clone(),
        builtin_filter.clone(),
    )?;

    // Register the sub-agent tool with access to the provider
    if builtin_filter.admits("spawn_agent") {
        tool_registry
            .register(Arc::new(crate::tools::subagent::SpawnAgentTool {
                provider: Arc::clone(&provider),
                parent_permission: shared_permission.clone(),
                tool_builder_params: crate::tools::subagent::ToolBuilderParams {
                    web_client: config.web_client.clone(),
                    sandbox_enabled: config.sandbox,
                    sandbox_capability,
                    builtin_filter: builtin_filter.clone(),
                },
                user_instructions: config.user_instructions.clone(),
            }))
            .expect("builtin subagent tool name collision");
    }

    crate::tools::warn_on_stale_builtin_tool_config(&builtin_filter);

    if let Some(manager) = mcp_manager {
        // Register MCP resource meta-tools upfront — they delegate through
        // `ServerEntry::require_connected` so they tolerate Pending /
        // Failed servers until a specific one is called.
        crate::tools::mcp_resources::register_all(&tool_registry, Arc::clone(manager));
        // Kick off the background connector. Each server's adapters are
        // installed into `tool_registry` via `replace_server_tools` +
        // `mark_deferred` as it reaches `Connected`. The REPL is free to
        // paint while this runs in the background.
        manager.start_connector(
            tool_registry.clone(),
            crate::mcp::McpRuntimeConfig::from_config(config),
        );
    }

    // Now that provider and registry exist, publish them on the MCP client
    // context so notification handlers (`tools/list_changed`) and sampling
    // callbacks (`sampling/createMessage`) can reach them.
    if let Some(context) = mcp_context {
        context.set_provider(Arc::clone(&provider));
        context.set_registry(tool_registry.clone());
    }

    let mut agent = Agent::new(
        provider,
        tool_registry,
        session_manager,
        shared_permission,
        AgentOptions {
            streaming: config.streaming,
            newline_before_prompt: config.newline_before_prompt,
            newline_after_prompt: config.newline_after_prompt,
            show_session_id_on_create: config.show_session_id_on_create,
            sandboxed_shell,
            render_mode: config.render_mode,
            context_messages: config.context_messages,
            auto_compact: config.auto_compact,
            context_window: config.context_window.unwrap_or_else(|| {
                config
                    .model
                    .as_deref()
                    .map(crate::config::context_window_for_model)
                    .unwrap_or(128_000)
            }),
            thinking_show_content: config.thinking_show_content,
            user_instructions: config.user_instructions.clone(),
            mcp_strict: config.mcp_strict,
            mcp_grace: config.mcp_grace,
        },
        todo_list,
        shared_session_id,
        approval_sender,
    );
    if let Some(manager) = mcp_manager {
        agent.set_mcp_manager(Arc::clone(manager));
    }
    Ok(agent)
}

async fn run_oneshot(
    config: ResolvedConfig,
    session_manager: SessionManager,
    token_store: TokenStore,
    prompt: String,
    mcp_manager: Option<Arc<mcp::McpClientManager>>,
    mcp_context: Arc<mcp::McpClientContext>,
) -> anyhow::Result<()> {
    let shared_permission = SharedPermission::new(config.permission);
    let credential = resolve_credential(&config)?;
    let agent = create_agent_from_config(
        &config,
        session_manager,
        shared_permission,
        token_store,
        credential,
        mcp_manager.as_ref(),
        Some(&mcp_context),
        None,
    )
    .await?;

    let cancellation = CancellationToken::new();
    let cancellation_clone = cancellation.clone();

    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            cancellation_clone.cancel();
        }
    });

    let mut session_id = None;
    let mut messages = Vec::new();

    match agent
        .run_turn(&mut session_id, &mut messages, prompt, cancellation)
        .await
    {
        Ok(()) => {}
        Err(error::AgshError::Interrupted) => {
            eprintln!("\nInterrupted.");
        }
        Err(error) => return Err(error.into()),
    }

    if let Some(id) = session_id
        && config.show_session_id_on_exit
    {
        render::render_session_id("Leaving session", &id.to_string());
    }

    if let Some(manager) = mcp_manager {
        shutdown_mcp_manager(manager).await;
    }

    Ok(())
}

async fn run_interactive(
    config: ResolvedConfig,
    session_manager: SessionManager,
    token_store: TokenStore,
    mcp_manager: Option<Arc<mcp::McpClientManager>>,
    mcp_context: Arc<mcp::McpClientContext>,
) -> anyhow::Result<()> {
    let shared_permission = SharedPermission::new(config.permission);

    // Resolve session resumption BEFORE spawning the REPL so the
    // "Resuming session" message appears before the first prompt.
    let (mut session_id, mut messages, mut session_lock) =
        resolve_session_resume(&session_manager, &config).await?;

    if !messages.is_empty() {
        reprint_last_message(&messages, config.render_mode);
    }

    let (input_sender, mut input_receiver) = tokio::sync::mpsc::unbounded_channel::<ReplEvent>();
    let (agent_event_sender, agent_event_receiver) =
        std::sync::mpsc::channel::<repl::AgentToReplEvent>();
    let approval_sender = agent_event_sender.clone();

    // Wire progress/elicitation notifications from MCP handlers through the
    // same agent→shell channel so the REPL can render them inline.
    {
        let sender_for_progress = agent_event_sender.clone();
        mcp::progress::set_ui_sink(Box::new(move |update| {
            if sender_for_progress
                .send(repl::AgentToReplEvent::McpProgress(update))
                .is_err()
            {
                tracing::debug!("MCP progress dropped (REPL disconnected)");
            }
        }));
        let sender_for_elicitation = agent_event_sender.clone();
        mcp::elicitation::set_shell_sink(Some(Box::new(move |prompt| {
            if sender_for_elicitation
                .send(repl::AgentToReplEvent::McpElicitation(prompt))
                .is_err()
            {
                tracing::debug!("MCP elicitation dropped (REPL disconnected)");
            }
        })));
    }

    let repl_permission = shared_permission.clone();
    let show_path_in_prompt = config.show_path_in_prompt;
    let input_style = config.input_style;
    let repl_handle = tokio::task::spawn_blocking(move || {
        repl::run_repl(
            repl_permission,
            show_path_in_prompt,
            input_style,
            input_sender,
            agent_event_receiver,
        );
    });

    // Try to create the agent (may fail if config is incomplete)
    let credential = match resolve_credential(&config) {
        Ok(credential) => credential,
        Err(error) => {
            render::render_error(&error);
            render::render_provider_setup_hint();
            drop(agent_event_sender);
            repl_handle.await?;
            return Ok(());
        }
    };
    let agent = match create_agent_from_config(
        &config,
        session_manager.clone(),
        shared_permission,
        token_store.clone(),
        credential,
        mcp_manager.as_ref(),
        Some(&mcp_context),
        Some(approval_sender),
    )
    .await
    {
        Ok(agent) => agent,
        Err(error) => {
            render::render_error(&error);
            render::render_provider_setup_hint();
            drop(agent_event_sender);
            repl_handle.await?;
            return Ok(());
        }
    };

    while let Some(event) = input_receiver.recv().await {
        match event {
            ReplEvent::UserInput(input) => {
                let cancellation = CancellationToken::new();

                let cancellation_for_signal = cancellation.clone();
                let signal_handle = tokio::spawn(async move {
                    if tokio::signal::ctrl_c().await.is_ok() {
                        cancellation_for_signal.cancel();
                    }
                });

                match agent
                    .run_turn(&mut session_id, &mut messages, input, cancellation)
                    .await
                {
                    Ok(()) => {}
                    Err(error::AgshError::Interrupted) => {
                        eprintln!("\nInterrupted.");
                        if config.newline_before_prompt {
                            println!();
                        }
                    }
                    Err(error) => {
                        render::render_error(&error);
                        if config.newline_before_prompt {
                            println!();
                        }
                    }
                }

                // The first turn creates the session if one wasn't resumed;
                // claim the file lock as soon as the ID is known so a second
                // agsh invocation can't attach to it.
                if session_lock.is_none()
                    && let Some(id) = session_id
                {
                    match session_manager.lock_session(id) {
                        Ok(lock) => session_lock = Some(lock),
                        Err(error) => render::render_error(&error),
                    }
                }

                signal_handle.abort();

                if agent_event_sender
                    .send(repl::AgentToReplEvent::Done)
                    .is_err()
                {
                    break;
                }
            }
            ReplEvent::Command(command) => {
                match command {
                    repl::SlashCommand::Session => match &session_id {
                        Some(id) => render::render_session_id("Current session", &id.to_string()),
                        None => eprintln!("No active session yet."),
                    },
                    repl::SlashCommand::Compact => {
                        match agent.compact_session(&mut session_id, &mut messages).await {
                            Ok(()) => {
                                render::render_hint("Session compacted.");
                            }
                            Err(error) => {
                                render::render_error(&error);
                            }
                        }
                    }
                    repl::SlashCommand::Export => match &session_id {
                        Some(id) => {
                            if let Err(error) = export_session(&session_manager, *id, None).await {
                                render::render_error(&error);
                            }
                        }
                        None => eprintln!("No active session to export."),
                    },
                    repl::SlashCommand::McpList => {
                        if let Err(error) =
                            mcp::cli::run_list(&config.mcp_servers, mcp_manager.as_ref()).await
                        {
                            render::render_error(&error);
                        }
                    }
                    repl::SlashCommand::McpReconnect { server } => {
                        if let Err(error) =
                            mcp::cli::run_reconnect(&config.mcp_servers, &token_store, &server)
                                .await
                        {
                            render::render_error(&error);
                        }
                    }
                    repl::SlashCommand::McpLogin { server } => {
                        if let Err(error) =
                            mcp::cli::run_login(&config.mcp_servers, &token_store, &server).await
                        {
                            render::render_error(&error);
                        }
                    }
                    repl::SlashCommand::McpLogout { server } => {
                        if let Err(error) =
                            mcp::cli::run_logout(&config.mcp_servers, &token_store, &server).await
                        {
                            render::render_error(&error);
                        }
                    }
                    repl::SlashCommand::McpPrompt {
                        server,
                        prompt: prompt_name,
                        args,
                    } => match mcp_manager.as_ref() {
                        Some(manager) => {
                            let entry = manager.server_entry(&server);
                            let Some(entry) = entry else {
                                eprintln!(
                                    "unknown MCP server '{}'; configured: {:?}",
                                    server,
                                    manager.server_names()
                                );
                                continue;
                            };
                            // Map positional args to declared prompt argument
                            // names (lookup via prompts/list).
                            let arg_names = match mcp::list_prompts(&entry).await {
                                Ok(prompts) => prompts
                                    .into_iter()
                                    .find(|p| p.name == prompt_name)
                                    .and_then(|p| p.arguments)
                                    .map(|args| {
                                        args.into_iter().map(|a| a.name).collect::<Vec<_>>()
                                    })
                                    .unwrap_or_default(),
                                Err(error) => {
                                    eprintln!("list_prompts failed: {}", error);
                                    Vec::new()
                                }
                            };
                            let mut arguments: Option<serde_json::Map<String, serde_json::Value>> =
                                None;
                            if !arg_names.is_empty() {
                                let mut map = serde_json::Map::new();
                                for (i, name) in arg_names.iter().enumerate() {
                                    if let Some(value) = args.get(i) {
                                        map.insert(
                                            name.clone(),
                                            serde_json::Value::String(value.clone()),
                                        );
                                    }
                                }
                                arguments = Some(map);
                            }
                            match mcp::get_prompt(&entry, prompt_name.clone(), arguments).await {
                                Ok(result) => {
                                    // Render the prompt messages as a single
                                    // user turn — same shape as the
                                    // `get_mcp_prompt` tool output.
                                    let mut body = String::new();
                                    for message in &result.messages {
                                        let role = match message.role {
                                            rmcp::model::PromptMessageRole::User => "user",
                                            rmcp::model::PromptMessageRole::Assistant => {
                                                "assistant"
                                            }
                                        };
                                        if let rmcp::model::PromptMessageContent::Text { text } =
                                            &message.content
                                        {
                                            body.push_str(&format!("{}: {}\n", role, text));
                                        }
                                    }
                                    let user_input = body.trim().to_string();
                                    if !user_input.is_empty()
                                        && let Err(error) = agent
                                            .run_turn(
                                                &mut session_id,
                                                &mut messages,
                                                user_input,
                                                CancellationToken::new(),
                                            )
                                            .await
                                    {
                                        render::render_error(&error);
                                    }
                                }
                                Err(error) => {
                                    eprintln!("get_prompt failed: {}", error);
                                }
                            }
                        }
                        None => {
                            eprintln!("no MCP servers configured");
                        }
                    },
                    _ => {}
                }

                if agent_event_sender
                    .send(repl::AgentToReplEvent::Done)
                    .is_err()
                {
                    break;
                }
            }
            ReplEvent::Exit => {
                break;
            }
        }
    }

    drop(agent_event_sender);
    repl_handle.await?;

    if let Some(id) = session_id
        && config.show_session_id_on_exit
    {
        render::render_session_id("Leaving session", &id.to_string());
    }
    // Drop after the "Leaving session" message so the lock is held until the
    // very end; the OS releases the underlying flock when the FD closes.
    drop(session_lock);

    if let Some(manager) = mcp_manager {
        shutdown_mcp_manager(manager).await;
    }

    Ok(())
}

/// Unwrap the shared MCP manager and drive its shutdown. The manager is held
/// behind an `Arc` because resource/prompt tools keep clones of it; once the
/// agent and tool registry have been dropped, try_unwrap should succeed.
async fn shutdown_mcp_manager(manager: Arc<mcp::McpClientManager>) {
    match Arc::try_unwrap(manager) {
        Ok(manager) => manager.shutdown().await,
        Err(_arc) => {
            tracing::debug!(
                "MCP manager still referenced at shutdown; relying on drop guards for cleanup"
            );
        }
    }
}

async fn export_session(
    session_manager: &SessionManager,
    session_id: uuid::Uuid,
    output: Option<&str>,
) -> anyhow::Result<()> {
    if !session_manager.session_exists(session_id).await? {
        anyhow::bail!("session not found: {}", session_id);
    }

    let stored_messages = session_manager.load_messages(session_id).await?;
    let tool_outputs: std::collections::HashMap<String, String> = session_manager
        .load_all_tool_outputs(session_id)
        .await?
        .into_iter()
        .collect();
    let markdown = format_session_as_markdown(session_id, &stored_messages, &tool_outputs);

    match output {
        Some("-") => {
            print!("{}", markdown);
        }
        Some(path) => {
            std::fs::write(path, &markdown)?;
            tracing::info!("exported session to {}", path);
        }
        None => {
            let path = format!("session-{}.md", session_id);
            std::fs::write(&path, &markdown)?;
            tracing::info!("exported session to {}", path);
        }
    }

    Ok(())
}

async fn run_mcp_subcommand(
    session_manager: &SessionManager,
    action: &cli::McpAction,
    cli_args: &cli::Cli,
) -> anyhow::Result<()> {
    let config = ResolvedConfig::from_cli(cli_args);
    let token_store = session_manager.token_store();
    match action {
        cli::McpAction::List => mcp::cli::run_list(&config.mcp_servers, None).await?,
        cli::McpAction::Get { name } => mcp::cli::run_get(&config.mcp_servers, name).await?,
        cli::McpAction::Reconnect { name } => {
            mcp::cli::run_reconnect(&config.mcp_servers, &token_store, name).await?
        }
        cli::McpAction::Tools { name } => {
            mcp::cli::run_tools(
                &config.mcp_servers,
                config.mcp_default_permission,
                &token_store,
                name,
            )
            .await?
        }
        cli::McpAction::Login { name } => {
            mcp::cli::run_login(&config.mcp_servers, &token_store, name).await?
        }
        cli::McpAction::Logout { name } => {
            mcp::cli::run_logout(&config.mcp_servers, &token_store, name).await?
        }
        cli::McpAction::Add {
            name,
            location,
            args,
            transport,
            env,
            header,
            auth,
            auth_token,
            client_id,
            client_secret,
            signing_key,
            signing_algorithm,
            scope,
            redirect_port,
            permission,
            sampling,
            sampling_limit,
            no_login,
            allow_tool,
            disable_tool,
            tool_permission,
            disabled,
        } => {
            mcp::cli::run_add(
                mcp::cli::AddArgs {
                    name: name.clone(),
                    location: location.clone(),
                    args: args.clone(),
                    transport: transport.clone(),
                    env: env.clone(),
                    header: header.clone(),
                    auth: auth.clone(),
                    auth_token: auth_token.clone(),
                    client_id: client_id.clone(),
                    client_secret: client_secret.clone(),
                    signing_key: signing_key.clone(),
                    signing_algorithm: signing_algorithm.clone(),
                    scope: scope.clone(),
                    redirect_port: *redirect_port,
                    permission: permission.clone(),
                    sampling: *sampling,
                    sampling_limit: *sampling_limit,
                    no_login: *no_login,
                    allow_tool: allow_tool.clone(),
                    disable_tool: disable_tool.clone(),
                    tool_permission: tool_permission.clone(),
                    disabled: *disabled,
                },
                &token_store,
            )
            .await?
        }
        cli::McpAction::Remove { name } => mcp::cli::run_remove(name, &token_store).await?,
        cli::McpAction::Disable { name } => mcp::cli::run_disable(name).await?,
        cli::McpAction::Enable { name } => mcp::cli::run_enable(name).await?,
    }
    Ok(())
}

/// Handle `agsh tools <action>`.
async fn run_tools_subcommand(
    action: &cli::ToolsAction,
    cli_args: &cli::Cli,
) -> anyhow::Result<()> {
    match action {
        cli::ToolsAction::List => {
            let config = ResolvedConfig::from_cli(cli_args);
            let filter = crate::tools::BuiltinToolFilter::from_config(
                config.builtin_allowed_tools.clone(),
                config.builtin_disabled_tools.clone(),
                config.builtin_tool_permissions.clone(),
            );
            crate::tools::warn_on_stale_builtin_tool_config(&filter);

            // Build with no filter so the catalogue carries every tool's
            // hardcoded level; overlay the real filter for status/source.
            let session_manager = SessionManager::open(None).await?;
            let shared_permission = SharedPermission::new(config.permission);
            let sandbox_capability = crate::sandbox::detect();
            let todo_list: crate::tools::todo::SharedTodoList =
                std::sync::Arc::new(tokio::sync::RwLock::new(Vec::new()));
            let shared_session_id: std::sync::Arc<tokio::sync::RwLock<Option<uuid::Uuid>>> =
                std::sync::Arc::new(tokio::sync::RwLock::new(None));
            let reference = ToolRegistry::build_default(
                config.web_client.clone(),
                shared_permission,
                config.sandbox,
                sandbox_capability,
                todo_list,
                session_manager,
                shared_session_id,
                crate::tools::BuiltinToolFilter::default(),
            )?;

            let catalogue = reference.tool_catalogue();
            println!(
                "{:<20} {:<9} {:<9} {:<10} description",
                "NAME", "REQUIRED", "SOURCE", "VISIBILITY"
            );
            println!("{}", "-".repeat(78));
            for (name, description, required, is_deferred) in &catalogue {
                let override_entry = filter.permission_overrides.get(name);
                let effective = override_entry.copied().unwrap_or(*required);
                let source = if override_entry.is_some() {
                    "override"
                } else {
                    "builtin"
                };
                let visibility = if filter.admits(name) {
                    if *is_deferred { "deferred" } else { "enabled" }
                } else {
                    "disabled"
                };
                let short = description
                    .lines()
                    .next()
                    .unwrap_or("")
                    .chars()
                    .take(40)
                    .collect::<String>();
                println!(
                    "{:<20} {:<9} {:<9} {:<10} {}",
                    name,
                    effective.to_string(),
                    source,
                    visibility,
                    short
                );
            }
        }
    }
    Ok(())
}

async fn list_sessions(session_manager: &SessionManager, limit: u32) -> anyhow::Result<()> {
    let sessions = session_manager.list_sessions(limit).await?;

    if sessions.is_empty() {
        println!("No sessions found.");
        return Ok(());
    }

    println!("{:<36}  {:<20}  Preview", "ID", "Updated");

    for session in &sessions {
        let updated = format_timestamp(&session.updated_at);
        println!("{:<36}  {:<20}  {}", session.id, updated, session.preview);
    }

    Ok(())
}

async fn delete_sessions(
    session_manager: &SessionManager,
    session_ids: &[uuid::Uuid],
    all: bool,
) -> anyhow::Result<()> {
    if all {
        let deleted = session_manager.delete_all_sessions().await?;
        tracing::info!("deleted {} session(s)", deleted);
        return Ok(());
    }

    if session_ids.is_empty() {
        anyhow::bail!("specify one or more session IDs, or use --all to delete all sessions");
    }

    let mut deleted = 0u64;
    for session_id in session_ids {
        if session_manager.delete_session(*session_id).await? {
            deleted += 1;
        } else {
            // User-facing error: they asked to delete a specific ID and
            // we couldn't find it, so stderr (not silent) is right.
            eprintln!("Session not found: {}", session_id);
        }
    }

    tracing::info!("deleted {} session(s)", deleted);
    Ok(())
}

fn format_timestamp(rfc3339: &str) -> String {
    chrono::DateTime::parse_from_rfc3339(rfc3339)
        .map(|datetime| datetime.format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_else(|_| rfc3339.to_string())
}

fn format_session_as_markdown(
    session_id: uuid::Uuid,
    messages: &[session::StoredMessage],
    tool_outputs: &std::collections::HashMap<String, String>,
) -> String {
    use std::fmt::Write;

    let mut output = String::new();
    writeln!(output, "# Session {}\n", session_id).ok();

    for message in messages {
        match message.role.as_str() {
            "user" => {
                writeln!(output, "## User\n").ok();
                writeln!(output, "{}\n", message.content).ok();
            }
            "assistant" => {
                if let Ok(blocks) =
                    serde_json::from_str::<Vec<provider::ContentBlock>>(&message.content)
                {
                    writeln!(output, "## Assistant\n").ok();
                    for block in &blocks {
                        match block {
                            provider::ContentBlock::Text { text } => {
                                writeln!(output, "{}\n", text).ok();
                            }
                            provider::ContentBlock::ToolUse { name, input, .. } => {
                                let input_pretty = serde_json::to_string_pretty(input)
                                    .unwrap_or_else(|_| input.to_string());
                                writeln!(output, "<details>").ok();
                                writeln!(output, "<summary>Tool call: {}</summary>\n", name).ok();
                                writeln!(output, "```json\n{}\n```\n", input_pretty).ok();
                                writeln!(output, "</details>\n").ok();
                            }
                            provider::ContentBlock::ToolResult { .. }
                            | provider::ContentBlock::Thinking { .. } => {}
                        }
                    }
                } else {
                    writeln!(output, "## Assistant\n").ok();
                    writeln!(output, "{}\n", message.content).ok();
                }
            }
            "tool_results" => {
                if let Ok(blocks) =
                    serde_json::from_str::<Vec<provider::ContentBlock>>(&message.content)
                {
                    for block in &blocks {
                        if let provider::ContentBlock::ToolResult {
                            content, is_error, ..
                        } = block
                        {
                            let label = if *is_error {
                                "Tool result (error)"
                            } else {
                                "Tool result"
                            };
                            writeln!(output, "<details>").ok();
                            writeln!(output, "<summary>{}</summary>\n", label).ok();
                            let text = provider::ContentBlock::tool_result_text_content(content);
                            let text = resolve_large_output_tags(&text, tool_outputs);
                            writeln!(output, "```\n{}\n```\n", text).ok();
                            writeln!(output, "</details>\n").ok();
                        }
                    }
                }
            }
            _ => {}
        }
    }

    output
}

fn resolve_large_output_tags(
    text: &str,
    tool_outputs: &std::collections::HashMap<String, String>,
) -> String {
    let re = match regex::Regex::new(r#"<large-output name="([^"]+)"[^>]*>[\s\S]*?</large-output>"#)
    {
        Ok(re) => re,
        Err(_) => return text.to_string(),
    };

    re.replace_all(text, |caps: &regex::Captures| {
        let name = &caps[1];
        match tool_outputs.get(name) {
            Some(content) => content.clone(),
            None => caps[0].to_string(),
        }
    })
    .into_owned()
}

fn resolve_credential(config: &ResolvedConfig) -> anyhow::Result<AuthCredential> {
    match &config.auth_credential {
        Some(credential) => Ok(credential.clone()),
        None => Err(anyhow::anyhow!(
            "no API key or OAuth token configured. Set OPENAI_API_KEY, CLAUDE_API_KEY, \
             or CLAUDE_OAUTH_TOKEN env var, or provider.api_key / provider.oauth_token \
             in config file (~/.config/agsh/config.toml)"
        )),
    }
}

fn reprint_last_message(messages: &[provider::Message], render_mode: render::RenderMode) {
    let Some(last) = messages.last() else {
        return;
    };

    let text = match last.role {
        provider::Role::Assistant => {
            let text = last.text_content();
            if text.is_empty() {
                return;
            }
            text
        }
        provider::Role::User => {
            let raw = last.text_content();
            let stripped = session::strip_context_tags(&raw);
            if stripped.is_empty() {
                return;
            }
            stripped.to_string()
        }
    };

    let mut renderer = render::StreamingRenderer::new(render_mode);
    if let Err(error) = renderer.push_delta(&text) {
        tracing::debug!("failed to render last message delta: {}", error);
    }
    if let Err(error) = renderer.finish() {
        tracing::debug!("failed to finish rendering last message: {}", error);
    }
    println!();
}

async fn resolve_session_resume(
    session_manager: &SessionManager,
    config: &ResolvedConfig,
) -> anyhow::Result<(
    Option<uuid::Uuid>,
    Vec<provider::Message>,
    Option<session::SessionLock>,
)> {
    let Some(value) = &config.continue_session else {
        return Ok((None, Vec::new(), None));
    };

    if value == "last" {
        match session_manager.last_session_id().await? {
            Some(id) => {
                let lock = session_manager.lock_session(id)?;
                render::render_session_id("Continuing session", &id.to_string());
                if config.newline_after_prompt {
                    println!();
                }
                let messages = load_session_messages(session_manager, id).await?;
                Ok((Some(id), messages, Some(lock)))
            }
            None => Ok((None, Vec::new(), None)),
        }
    } else {
        let id: uuid::Uuid = value
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid session ID: {}", value))?;
        if !session_manager.session_exists(id).await? {
            anyhow::bail!("session not found: {}", id);
        }
        let lock = session_manager.lock_session(id)?;
        render::render_session_id("Continuing session", &id.to_string());
        if config.newline_after_prompt {
            println!();
        }
        let messages = load_session_messages(session_manager, id).await?;
        Ok((Some(id), messages, Some(lock)))
    }
}

async fn load_session_messages(
    session_manager: &SessionManager,
    session_id: uuid::Uuid,
) -> anyhow::Result<Vec<provider::Message>> {
    let stored = session_manager.load_messages(session_id).await?;
    let mut messages = Vec::new();

    for stored_message in stored {
        match stored_message.role.as_str() {
            "user" => {
                messages.push(provider::Message::user(&stored_message.content));
            }
            "assistant" => {
                // Content is stored as JSON array of ContentBlock
                if let Ok(content) =
                    serde_json::from_str::<Vec<provider::ContentBlock>>(&stored_message.content)
                {
                    messages.push(provider::Message {
                        role: provider::Role::Assistant,
                        content,
                    });
                } else {
                    tracing::warn!(
                        "failed to parse assistant message as ContentBlock array, treating as text"
                    );
                    messages.push(provider::Message::assistant_text(&stored_message.content));
                }
            }
            "tool_results" => {
                match serde_json::from_str::<Vec<provider::ContentBlock>>(&stored_message.content) {
                    Ok(content) => {
                        messages.push(provider::Message {
                            role: provider::Role::User,
                            content,
                        });
                    }
                    Err(error) => {
                        tracing::warn!("failed to parse tool_results message, dropping: {}", error,);
                    }
                }
            }
            _ => {
                tracing::warn!("unknown message role: {}", stored_message.role);
            }
        }
    }

    // Validate: every assistant ToolUse must be followed by a matching ToolResult.
    // Drop orphaned assistant messages to prevent API errors.
    validate_tool_use_chains(&mut messages);

    Ok(messages)
}

fn validate_tool_use_chains(messages: &mut Vec<provider::Message>) {
    let mut index = 0;
    while index < messages.len() {
        if messages[index].role != provider::Role::Assistant {
            index += 1;
            continue;
        }

        let tool_use_ids: Vec<String> = messages[index]
            .content
            .iter()
            .filter_map(|block| {
                if let provider::ContentBlock::ToolUse { id, .. } = block {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect();

        if tool_use_ids.is_empty() {
            index += 1;
            continue;
        }

        // Check if the next message has matching ToolResult blocks
        let has_results = messages
            .get(index + 1)
            .is_some_and(|next| {
                next.role == provider::Role::User
                    && tool_use_ids.iter().all(|id| {
                        next.content.iter().any(|block| {
                            matches!(block, provider::ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == id)
                        })
                    })
            });

        if has_results {
            index += 1;
        } else {
            tracing::warn!(
                "dropping assistant message with orphaned tool_use IDs: {:?}",
                tool_use_ids,
            );
            messages.remove(index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_msg(text: &str) -> provider::Message {
        provider::Message::user(text)
    }

    fn assistant_text(text: &str) -> provider::Message {
        provider::Message::assistant_text(text)
    }

    fn assistant_tool_use(id: &str, name: &str) -> provider::Message {
        provider::Message {
            role: provider::Role::Assistant,
            content: vec![provider::ContentBlock::ToolUse {
                id: id.to_string(),
                name: name.to_string(),
                input: serde_json::json!({}),
            }],
        }
    }

    fn tool_result(tool_use_id: &str) -> provider::Message {
        provider::Message {
            role: provider::Role::User,
            content: vec![provider::ContentBlock::ToolResult {
                tool_use_id: tool_use_id.to_string(),
                content: vec![provider::ToolResultContent::Text {
                    text: "ok".to_string(),
                }],
                is_error: false,
            }],
        }
    }

    #[test]
    fn test_validate_valid_chain() {
        let mut messages = vec![
            user_msg("hello"),
            assistant_tool_use("c1", "read_file"),
            tool_result("c1"),
            assistant_text("done"),
        ];
        validate_tool_use_chains(&mut messages);
        assert_eq!(messages.len(), 4);
    }

    #[test]
    fn test_validate_orphaned_tool_use_dropped() {
        let mut messages = vec![
            user_msg("hello"),
            assistant_tool_use("c1", "read_file"),
            // Missing tool_result for c1
            assistant_text("done"),
        ];
        validate_tool_use_chains(&mut messages);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, provider::Role::User);
        assert_eq!(messages[1].role, provider::Role::Assistant);
        assert_eq!(messages[1].text_content(), "done");
    }

    #[test]
    fn test_validate_orphaned_at_end() {
        let mut messages = vec![user_msg("hello"), assistant_tool_use("c1", "read_file")];
        validate_tool_use_chains(&mut messages);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text_content(), "hello");
    }

    #[test]
    fn test_validate_mismatched_ids() {
        let mut messages = vec![
            user_msg("hello"),
            assistant_tool_use("c1", "read_file"),
            tool_result("c2"), // Wrong ID
        ];
        validate_tool_use_chains(&mut messages);
        // The assistant message should be dropped because c1 has no matching result
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_validate_text_only_preserved() {
        let mut messages = vec![user_msg("hello"), assistant_text("hi"), user_msg("bye")];
        validate_tool_use_chains(&mut messages);
        assert_eq!(messages.len(), 3);
    }

    #[test]
    fn test_validate_multiple_chains() {
        let mut messages = vec![
            user_msg("start"),
            assistant_tool_use("c1", "read_file"),
            tool_result("c1"),
            assistant_tool_use("c2", "write_file"),
            // Missing tool_result for c2
            assistant_text("done"),
        ];
        validate_tool_use_chains(&mut messages);
        // c2 should be dropped, rest preserved
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[3].text_content(), "done");
    }

    // -- log filter --

    /// The default filter (no `RUST_LOG`) floors rmcp's SSE-reconnect
    /// module at `error`. Guards against a future refactor silently
    /// dropping the directive and letting the noisy warning back in.
    #[test]
    fn default_log_filter_downgrades_rmcp_sse_warns() {
        // Belt-and-braces: clear RUST_LOG so the `try_from_default_env`
        // branch doesn't short-circuit under a developer-set env var.
        // SAFETY: tests run in a single process; we don't read this env
        // var from other threads.
        unsafe { std::env::remove_var("RUST_LOG") };
        let rendered = format!("{}", build_log_filter("warn"));
        assert!(
            rendered.contains("rmcp::transport::common::client_side_sse=error"),
            "expected SSE-reconnect target to be floored at `error` in the default \
             filter, got: {}",
            rendered
        );
    }

    /// When the user sets `RUST_LOG` we honour it verbatim — no hidden
    /// directive overlay — so debugging rmcp internals with e.g.
    /// `RUST_LOG=rmcp=debug` works as expected.
    #[test]
    fn explicit_rust_log_is_not_overridden() {
        // SAFETY: tests run in a single process; we don't read RUST_LOG
        // from other threads.
        unsafe { std::env::set_var("RUST_LOG", "rmcp=debug") };
        let rendered = format!("{}", build_log_filter("warn"));
        unsafe { std::env::remove_var("RUST_LOG") };
        assert!(
            !rendered.contains("rmcp::transport::common::client_side_sse=error"),
            "explicit RUST_LOG must not be augmented; got: {}",
            rendered
        );
        assert!(
            rendered.contains("rmcp=debug"),
            "user's RUST_LOG should pass through unchanged; got: {}",
            rendered
        );
    }
}
