mod agent;
mod cli;
mod config;
mod error;
mod mcp;
mod permission;
mod provider;
mod render;
mod sandbox;
mod session;
mod setup;
mod shell;
mod system_prompt;
mod tools;

use clap::Parser;
use tokio_util::sync::CancellationToken;

use std::sync::Arc;

use crate::agent::{Agent, AgentOptions};
use crate::config::ResolvedConfig;
use crate::permission::SharedPermission;
use crate::provider::{AuthCredential, create_provider};
use crate::session::{SessionManager, TokenStore};
use crate::shell::ShellEvent;
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
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level)),
        )
        .with_writer(std::io::stderr)
        .init();

    let runtime = tokio::runtime::Runtime::new()?;

    // Handle subcommands that don't need full config resolution
    if let Some(command) = cli.command {
        return runtime.block_on(async {
            let session_manager = SessionManager::open(None).await?;
            match command {
                cli::Command::Setup => {
                    let token_store = session_manager.token_store();
                    setup::run_setup(&token_store).await
                }
                cli::Command::Export { session_id, output } => {
                    export_session(&session_manager, session_id, output.as_deref()).await
                }
                cli::Command::Delete { session_ids, all } => {
                    delete_sessions(&session_manager, &session_ids, all).await
                }
                cli::Command::List { limit } => list_sessions(&session_manager, limit).await,
            }
        });
    }

    // Auto-detect first launch: no config file and no env-based provider
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

async fn async_main(mut config: ResolvedConfig) -> anyhow::Result<()> {
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

    // If no credential from env/config, try loading from database
    if config.auth_credential.is_none()
        && let Some(provider_name) = config.provider_name.as_deref()
        && provider_name == "claude"
    {
        match token_store.load_oauth_token("claude").await {
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

    // Save OAuth token from env/config to database for future use
    if let Some(credential @ AuthCredential::OAuthToken { .. }) = &config.auth_credential
        && let Some(provider_name) = config.provider_name.as_deref()
        && let Err(error) = token_store
            .save_oauth_token(provider_name, credential)
            .await
    {
        tracing::warn!("failed to save OAuth token to database: {}", error);
    }

    let mcp_manager = if !config.mcp_servers.is_empty() {
        Some(mcp::McpClientManager::connect_all(&config.mcp_servers, Some(&token_store)).await?)
    } else {
        None
    };

    if let Some(prompt) = config.prompt.clone() {
        return run_oneshot(config, session_manager, token_store, prompt, mcp_manager).await;
    }

    run_interactive(config, session_manager, token_store, mcp_manager).await
}

async fn create_agent_from_config(
    config: &ResolvedConfig,
    session_manager: SessionManager,
    shared_permission: SharedPermission,
    token_store: TokenStore,
    credential: AuthCredential,
    mcp_manager: Option<&mcp::McpClientManager>,
) -> anyhow::Result<Agent> {
    config.validate()?;

    let provider_name = config
        .provider_name
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("provider_name missing after validation"))?;
    let needs_token_store = matches!(credential, AuthCredential::OAuthToken { .. });

    let provider = create_provider(
        provider_name,
        credential,
        config
            .model
            .clone()
            .ok_or_else(|| anyhow::anyhow!("model missing after validation"))?,
        config.base_url.clone(),
        config.client_id.clone(),
        config.oauth_token_url.clone(),
        if needs_token_store {
            Some(Arc::new(token_store))
        } else {
            None
        },
    )?;

    let sandbox_capability = crate::sandbox::detect();
    let sandboxed_shell = config.sandbox
        && !matches!(
            sandbox_capability,
            crate::sandbox::SandboxCapability::Unavailable
        );

    let mut tool_registry = ToolRegistry::build_default(
        config.user_agent.clone(),
        shared_permission.clone(),
        config.sandbox,
        sandbox_capability,
    );

    if let Some(manager) = mcp_manager {
        for mcp_config in &config.mcp_servers {
            let mcp_tools = manager
                .discover_tools_for_server(&mcp_config.name, mcp_config.permission.as_deref())
                .await?;
            for tool in mcp_tools {
                tool_registry.register(Box::new(tool));
            }
        }
    }

    Ok(Agent::new(
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
        },
    ))
}

async fn run_oneshot(
    config: ResolvedConfig,
    session_manager: SessionManager,
    token_store: TokenStore,
    prompt: String,
    mcp_manager: Option<mcp::McpClientManager>,
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
        manager.shutdown().await;
    }

    Ok(())
}

async fn run_interactive(
    config: ResolvedConfig,
    session_manager: SessionManager,
    token_store: TokenStore,
    mcp_manager: Option<mcp::McpClientManager>,
) -> anyhow::Result<()> {
    let shared_permission = SharedPermission::new(config.permission);

    // Resolve session resumption BEFORE spawning the REPL so the
    // "Resuming session" message appears before the first prompt.
    let (mut session_id, mut messages) = if config.continue_last {
        match session_manager.last_session_id().await? {
            Some(id) => {
                session_manager.lock_session(id).await?;
                render::render_session_id("Resuming session", &id.to_string());
                let messages = load_session_messages(&session_manager, id).await?;
                (Some(id), messages)
            }
            None => (None, Vec::new()),
        }
    } else if let Some(id) = config.session_id {
        if !session_manager.session_exists(id).await? {
            anyhow::bail!("session not found: {}", id);
        }
        session_manager.lock_session(id).await?;
        render::render_session_id("Resuming session", &id.to_string());
        let messages = load_session_messages(&session_manager, id).await?;
        (Some(id), messages)
    } else {
        (None, Vec::new())
    };

    let (input_sender, mut input_receiver) = tokio::sync::mpsc::unbounded_channel::<ShellEvent>();
    let (agent_done_sender, agent_done_receiver) = std::sync::mpsc::channel::<()>();

    let repl_permission = shared_permission.clone();
    let show_path_in_prompt = config.show_path_in_prompt;
    let repl_handle = tokio::task::spawn_blocking(move || {
        shell::run_repl(
            repl_permission,
            show_path_in_prompt,
            input_sender,
            agent_done_receiver,
        );
    });

    // Try to create the agent (may fail if config is incomplete)
    let credential = match resolve_credential(&config) {
        Ok(credential) => credential,
        Err(error) => {
            eprintln!("Error: {}", error);
            eprintln!("Configure a provider and model to use agsh.");
            eprintln!("Example: agsh --provider openai --model gpt-4o \"hello\"");
            eprintln!(
                "Or set AGSH_PROVIDER, AGSH_MODEL, and OPENAI_API_KEY environment variables."
            );
            drop(agent_done_sender);
            repl_handle.await?;
            return Ok(());
        }
    };
    let agent = match create_agent_from_config(
        &config,
        session_manager.clone(),
        shared_permission,
        token_store,
        credential,
        mcp_manager.as_ref(),
    )
    .await
    {
        Ok(agent) => agent,
        Err(error) => {
            eprintln!("Error: {}", error);
            eprintln!("Configure a provider and model to use agsh.");
            eprintln!("Example: agsh --provider openai --model gpt-4o \"hello\"");
            eprintln!(
                "Or set AGSH_PROVIDER, AGSH_MODEL, and OPENAI_API_KEY environment variables."
            );
            drop(agent_done_sender);
            repl_handle.await?;
            return Ok(());
        }
    };

    while let Some(event) = input_receiver.recv().await {
        match event {
            ShellEvent::UserInput(input) => {
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
                    }
                    Err(error) => {
                        eprintln!("Error: {}", error);
                    }
                }

                signal_handle.abort();

                if agent_done_sender.send(()).is_err() {
                    break;
                }
            }
            ShellEvent::Command(command) => {
                match command {
                    shell::SlashCommand::Session => match &session_id {
                        Some(id) => render::render_session_id("Current session", &id.to_string()),
                        None => eprintln!("No active session yet."),
                    },
                    shell::SlashCommand::Compact => {
                        match agent.compact_session(&mut session_id, &mut messages).await {
                            Ok(()) => {
                                render::render_hint("Session compacted.");
                            }
                            Err(error) => {
                                eprintln!("Error: {}", error);
                            }
                        }
                    }
                    _ => {}
                }

                if agent_done_sender.send(()).is_err() {
                    break;
                }
            }
            ShellEvent::Exit => {
                break;
            }
        }
    }

    drop(agent_done_sender);
    repl_handle.await?;

    if let Some(id) = session_id {
        if let Err(error) = session_manager.unlock_session(id).await {
            tracing::warn!("failed to unlock session: {}", error);
        }
        if config.show_session_id_on_exit {
            render::render_session_id("Leaving session", &id.to_string());
        }
    }

    if let Some(manager) = mcp_manager {
        manager.shutdown().await;
    }

    Ok(())
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
    let markdown = format_session_as_markdown(session_id, &stored_messages);

    match output {
        Some("-") => {
            print!("{}", markdown);
        }
        Some(path) => {
            std::fs::write(path, &markdown)?;
            eprintln!("Exported to {}", path);
        }
        None => {
            let path = format!("session-{}.md", session_id);
            std::fs::write(&path, &markdown)?;
            eprintln!("Exported to {}", path);
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
        eprintln!("Deleted {} session(s).", deleted);
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
            eprintln!("Session not found: {}", session_id);
        }
    }

    eprintln!("Deleted {} session(s).", deleted);
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
                            provider::ContentBlock::ToolResult { .. } => {}
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
                            writeln!(output, "```\n{}\n```\n", content).ok();
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
                    messages.push(provider::Message::assistant_text(&stored_message.content));
                }
            }
            "tool_results" => {
                if let Ok(content) =
                    serde_json::from_str::<Vec<provider::ContentBlock>>(&stored_message.content)
                {
                    messages.push(provider::Message {
                        role: provider::Role::User,
                        content,
                    });
                }
            }
            _ => {
                tracing::warn!("unknown message role: {}", stored_message.role);
            }
        }
    }

    Ok(messages)
}
