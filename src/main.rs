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
    approval_sender: Option<std::sync::mpsc::Sender<shell::AgentToShellEvent>>,
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
        config.thinking_enabled,
        config.thinking_budget_tokens,
        config.reasoning_effort.clone(),
    )?;

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

    let mut tool_registry = ToolRegistry::build_default(
        config.user_agent.clone(),
        shared_permission.clone(),
        config.sandbox,
        sandbox_capability,
        todo_list.clone(),
        session_manager.clone(),
        shared_session_id.clone(),
    );

    // Register the sub-agent tool with access to the provider
    tool_registry.register(Box::new(crate::tools::subagent::SpawnAgentTool {
        provider: Arc::clone(&provider),
        parent_permission: shared_permission.clone(),
        tool_builder_params: crate::tools::subagent::ToolBuilderParams {
            user_agent: config.user_agent.clone(),
            sandbox_enabled: config.sandbox,
            sandbox_capability,
        },
    }));

    if let Some(manager) = mcp_manager {
        for mcp_config in &config.mcp_servers {
            let mcp_tools = manager
                .discover_tools_for_server(&mcp_config.name, mcp_config.permission.as_deref())
                .await?;
            for tool in mcp_tools {
                use crate::tools::Tool as _;
                let name = tool.definition().name.clone();
                tool_registry.register(Box::new(tool));
                tool_registry.mark_deferred(&name);
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
            auto_compact: config.auto_compact,
            context_window: config.context_window.unwrap_or_else(|| {
                config
                    .model
                    .as_deref()
                    .map(crate::config::context_window_for_model)
                    .unwrap_or(128_000)
            }),
            thinking_show_content: config.thinking_show_content,
        },
        todo_list,
        shared_session_id,
        approval_sender,
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
    let (mut session_id, mut messages) = resolve_session_resume(&session_manager, &config).await?;

    if !messages.is_empty() {
        reprint_last_message(&messages, config.render_mode);
    }

    let (input_sender, mut input_receiver) = tokio::sync::mpsc::unbounded_channel::<ShellEvent>();
    let (agent_event_sender, agent_event_receiver) =
        std::sync::mpsc::channel::<shell::AgentToShellEvent>();
    let approval_sender = agent_event_sender.clone();

    let repl_permission = shared_permission.clone();
    let show_path_in_prompt = config.show_path_in_prompt;
    let repl_handle = tokio::task::spawn_blocking(move || {
        shell::run_repl(
            repl_permission,
            show_path_in_prompt,
            input_sender,
            agent_event_receiver,
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
            drop(agent_event_sender);
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
        Some(approval_sender),
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
            drop(agent_event_sender);
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
                        if config.newline_before_prompt {
                            println!();
                        }
                    }
                    Err(error) => {
                        eprintln!("Error: {}", error);
                        if config.newline_before_prompt {
                            println!();
                        }
                    }
                }

                signal_handle.abort();

                if agent_event_sender
                    .send(shell::AgentToShellEvent::Done)
                    .is_err()
                {
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
                    shell::SlashCommand::Export => match &session_id {
                        Some(id) => {
                            if let Err(error) = export_session(&session_manager, *id, None).await {
                                eprintln!("Error: {}", error);
                            }
                        }
                        None => eprintln!("No active session to export."),
                    },
                    _ => {}
                }

                if agent_event_sender
                    .send(shell::AgentToShellEvent::Done)
                    .is_err()
                {
                    break;
                }
            }
            ShellEvent::Exit => {
                break;
            }
        }
    }

    drop(agent_event_sender);
    repl_handle.await?;

    // Create a lock guard for the session so it gets unlocked even on panic.
    // The guard's Drop impl spawns an async unlock task.
    let _lock_guard = session_id.map(|id| {
        if config.show_session_id_on_exit {
            render::render_session_id("Leaving session", &id.to_string());
        }
        session::SessionLockGuard::new(session_manager, id)
    });

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

    render::render_hint("Last message:");
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
) -> anyhow::Result<(Option<uuid::Uuid>, Vec<provider::Message>)> {
    let Some(value) = &config.continue_session else {
        return Ok((None, Vec::new()));
    };

    if value == "last" {
        match session_manager.last_session_id().await? {
            Some(id) => {
                session_manager.lock_session(id).await?;
                render::render_session_id("Resuming session", &id.to_string());
                let messages = load_session_messages(session_manager, id).await?;
                Ok((Some(id), messages))
            }
            None => Ok((None, Vec::new())),
        }
    } else {
        let id: uuid::Uuid = value
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid session ID: {}", value))?;
        if !session_manager.session_exists(id).await? {
            anyhow::bail!("session not found: {}", id);
        }
        session_manager.lock_session(id).await?;
        render::render_session_id("Resuming session", &id.to_string());
        let messages = load_session_messages(session_manager, id).await?;
        Ok((Some(id), messages))
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
}
