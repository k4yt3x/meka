mod agent;
mod cli;
mod config;
mod error;
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

use crate::agent::Agent;
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

    // Handle the setup subcommand
    if let Some(cli::Command::Setup) = cli.command {
        return runtime.block_on(async {
            let session_manager = SessionManager::open(None).await?;
            let token_store = session_manager.token_store();
            setup::run_setup(&token_store).await
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
    if config.auth_credential.is_none() {
        if let Some(provider_name) = config.provider_name.as_deref() {
            if provider_name == "claude" {
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
        }
    }

    // Save OAuth token from env/config to database for future use
    if let Some(AuthCredential::OAuthToken { .. }) = &config.auth_credential {
        if let Some(provider_name) = config.provider_name.as_deref() {
            if let Err(error) = token_store
                .save_oauth_token(
                    provider_name,
                    config.auth_credential.as_ref().expect("checked above"),
                )
                .await
            {
                tracing::warn!("failed to save OAuth token to database: {}", error);
            }
        }
    }

    if let Some(prompt) = config.prompt.clone() {
        return run_oneshot(config, session_manager, token_store, prompt).await;
    }

    run_interactive(config, session_manager, token_store).await
}

fn create_agent_from_config(
    config: &ResolvedConfig,
    session_manager: SessionManager,
    shared_permission: SharedPermission,
    token_store: TokenStore,
    credential: AuthCredential,
) -> anyhow::Result<Agent> {
    config.validate()?;

    let provider_name = config.provider_name.as_deref().expect("validated");
    let needs_token_store = matches!(credential, AuthCredential::OAuthToken { .. });

    let provider = create_provider(
        provider_name,
        credential,
        config.model.clone().expect("validated"),
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

    let tool_registry = ToolRegistry::build_default(
        config.user_agent.clone(),
        shared_permission.clone(),
        config.sandbox,
        sandbox_capability,
    );

    Ok(Agent::new(
        provider,
        tool_registry,
        session_manager,
        shared_permission,
        config.streaming,
        config.newline_before_prompt,
        config.newline_after_prompt,
        config.show_session_id,
        sandboxed_shell,
        config.context_messages,
    ))
}

async fn run_oneshot(
    config: ResolvedConfig,
    session_manager: SessionManager,
    token_store: TokenStore,
    prompt: String,
) -> anyhow::Result<()> {
    let shared_permission = SharedPermission::new(config.permission);
    let credential = resolve_credential(&config)?;
    let agent = create_agent_from_config(
        &config,
        session_manager,
        shared_permission,
        token_store,
        credential,
    )?;

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

    if let Some(id) = session_id {
        if config.show_session_id {
            render::render_hint(&format!(
                "Run `agsh -c` or `agsh -s {}` to continue this session.",
                id
            ));
        }
    }

    Ok(())
}

async fn run_interactive(
    config: ResolvedConfig,
    session_manager: SessionManager,
    token_store: TokenStore,
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
    let repl_handle = tokio::task::spawn_blocking(move || {
        shell::run_repl(repl_permission, input_sender, agent_done_receiver);
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
    ) {
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
        if config.show_session_id {
            render::render_hint(&format!(
                "Run `agsh -c` or `agsh -s {}` to continue this session.",
                id
            ));
        }
    }

    Ok(())
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
