mod agent;
mod cli;
mod config;
mod error;
mod permission;
mod provider;
mod render;
mod session;
mod shell;
mod system_prompt;
mod tools;

use clap::Parser;
use tokio_util::sync::CancellationToken;

use crate::agent::Agent;
use crate::config::ResolvedConfig;
use crate::permission::SharedPermission;
use crate::provider::create_provider;
use crate::session::SessionManager;
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

    let config = ResolvedConfig::from_cli(&cli);

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async_main(config))
}

async fn async_main(config: ResolvedConfig) -> anyhow::Result<()> {
    let session_manager = SessionManager::open(None).await?;

    if let Some(prompt) = config.prompt.clone() {
        return run_oneshot(config, session_manager, prompt).await;
    }

    run_interactive(config, session_manager).await
}

fn create_agent_from_config(
    config: &ResolvedConfig,
    session_manager: SessionManager,
    shared_permission: SharedPermission,
) -> anyhow::Result<Agent> {
    config.validate()?;

    let provider = create_provider(
        config.provider_name.as_deref().expect("validated"),
        config.api_key.clone().expect("validated"),
        config.model.clone().expect("validated"),
        config.base_url.clone(),
    )?;

    let tool_registry = ToolRegistry::build_default(config.user_agent.clone());

    Ok(Agent::new(
        provider,
        tool_registry,
        session_manager,
        shared_permission,
        config.streaming,
        config.newline_before_prompt,
        config.newline_after_prompt,
        config.show_session_id,
    ))
}

async fn run_oneshot(
    config: ResolvedConfig,
    session_manager: SessionManager,
    prompt: String,
) -> anyhow::Result<()> {
    let shared_permission = SharedPermission::new(config.permission);
    let agent = create_agent_from_config(&config, session_manager, shared_permission)?;

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
    let agent = match create_agent_from_config(&config, session_manager.clone(), shared_permission)
    {
        Ok(agent) => agent,
        Err(error) => {
            eprintln!("Error: {}", error);
            eprintln!("Configure a provider and model to use agsh.");
            eprintln!("Example: agsh --provider openai --model gpt-4o -p \"hello\"");
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
