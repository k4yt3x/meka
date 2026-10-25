//! The slash commands: parsed on the screen, answered by whichever side holds what they need.
//!
//! The commands that need something only the host loop has: the live `Agent`, the conversation,
//! the session id that `/fork` moves, are forwarded and answered in [`answer`], and refused while
//! the host is busy. The rest the screen answers itself, the store's listings and the cancels
//! included, which is what lets them run while a turn does. What a command has to show goes to a
//! panel; what it did is one row on the transcript.
//!
//! **One owner, checked by the compiler.** [`SlashCommand::answered_by`] and [`answer`] are both
//! exhaustive, so a new variant fails to compile until both have said which side owns it.

use std::sync::Arc;

use super::frontend::Ui;
use crate::{
    agent, cli, conversation, mcp, provider, render, skills, store::Store, streams::capture,
};

/// What answering a command needs from the host loop.
pub(crate) struct HostCommandContext<'a> {
    pub(crate) agent: &'a agent::Agent,
    /// What a `/profile` switch resolves against, and the name the status row shows.
    pub(crate) shared: &'a crate::host::SharedDeps,
    pub(crate) current_profile: &'a Arc<std::sync::RwLock<String>>,
    /// Where a turn this command runs publishes its token for Ctrl+C.
    pub(crate) cancel: &'a crate::host::CancelCell,
    pub(crate) config: &'a crate::config::ResolvedConfig,
    pub(crate) ui: &'a Ui,
    pub(crate) mcp_manager: &'a Option<Arc<mcp::McpClientManager>>,
    pub(crate) messages: &'a mut conversation::Conversation,
    pub(crate) providers: &'a Arc<provider::ProviderRegistry>,
    /// `/fork` moves the session the loop is serving, so this is the loop's own cell.
    pub(crate) session_id: &'a mut Option<uuid::Uuid>,
    pub(crate) session_lock: &'a crate::store::SessionLockSlot,
    pub(crate) store: &'a Store,
    pub(crate) token_store: &'a crate::store::TokenStore,
    /// Where the shell was, for a resumed session whose recorded directory is gone.
    pub(crate) launch_cwd: &'a std::path::Path,
}

/// What [`attach_session`] moves: the loop's own cells and the agent's, which every tool reads.
struct AttachContext<'a> {
    store: &'a Store,
    config: &'a crate::config::ResolvedConfig,
    shared: &'a crate::host::SharedDeps,
    agent: &'a agent::Agent,
    session_id: &'a mut Option<uuid::Uuid>,
    session_lock: &'a crate::store::SessionLockSlot,
    messages: &'a mut conversation::Conversation,
    current_profile: &'a Arc<std::sync::RwLock<String>>,
    launch_cwd: &'a std::path::Path,
    ui: &'a Ui,
}

/// Make the session `value` names this run's, in place: what a resume at startup does, done
/// between turns. Everything of the new session is taken and read before anything of the old one
/// is let go, so a failure leaves the screen where it was, and the new lock drops with it.
async fn attach_session(value: &str, context: AttachContext<'_>) -> anyhow::Result<()> {
    let AttachContext {
        store,
        config,
        shared,
        agent,
        session_id,
        session_lock,
        messages,
        current_profile,
        launch_cwd,
        ui,
    } = context;
    let id = store.resolve_session_id(value).await?;
    if *session_id == Some(id) {
        ui.reply("Already on this session.");
        return Ok(());
    }
    crate::host::refuse_a_spawned_session(store, Some(id)).await?;
    // Locked, then read, so the row this run moves to is the one it now owns.
    let (lock, recorded) = store.open_session_row(id).await?;
    let hydrated = crate::host::hydrate_conversation(store, id).await?;
    let profile = crate::provider::profile_for_config(store, config, Some(id)).await?;
    let resolved = crate::host::resolve_profile_switch(shared, &profile).await?;
    // The row's project, refused by name when the store no longer has it, as the profile is.
    let project = (!recorded.project.is_empty()).then(|| recorded.project.clone());
    if let Some(name) = &project {
        crate::host::resolve_project_switch(shared, name).await?;
    }
    // The row's level, clamped to what the configuration still allows, as every resume reads it.
    let level = config
        .enabled_permissions
        .admit_recorded(recorded.permission, &format!("session {id}"))
        .unwrap_or(config.permission);

    // Nothing above has moved; from here the switch is made whole.
    crate::host::terminal::hold_session_lock(session_lock, Some(lock));
    *session_id = Some(id);
    let cells = agent.cells();
    cells.session_id.set(id);
    agent.set_provider(resolved);
    agent.set_project(project);
    *crate::sync::write(current_profile) = profile;
    if let Err(error) = cells.permission.try_set(level) {
        tracing::warn!("{error}");
    }
    cells.permission.set_approvals(recorded.approvals);
    cells.cwd.set(crate::host::resume_working_directory(
        recorded.cwd,
        launch_cwd,
        Some(id),
    ));
    // The agent's markers against message positions, its read tracker and its world-state
    // delta all describe the conversation being let go of; the one reset every rewrite calls
    // clears them and seeds the gauge with the new conversation's estimate.
    agent.reset_conversation_markers(&hydrated).await;
    *messages = hydrated;
    if config.show_session_id_on_resume {
        ui.session_id("Resuming session", &id.to_string());
    }
    ui.reload();
    Ok(())
}

/// Answer one forwarded command. Exhaustive over [`SlashCommand`] on purpose; see the module docs.
pub(crate) async fn answer(command: SlashCommand, context: HostCommandContext<'_>) {
    let HostCommandContext {
        agent,
        shared,
        current_profile,
        cancel,
        config,
        ui,
        mcp_manager,
        messages,
        providers,
        session_id,
        session_lock,
        store,
        token_store,
        launch_cwd,
    } = context;
    let session_id_cell = session_id;
    let mut session_id = *session_id_cell;
    match command {
        SlashCommand::Title(text) => match (&session_id, text) {
            (None, _) => ui.reply("No active session yet."),
            (Some(id), Some(text)) => match crate::store::normalize_title(&text) {
                Ok(title) => {
                    let shown = title.clone().unwrap_or_default();
                    match crate::host::record_session_change(
                        store,
                        *id,
                        crate::store::SessionPatch {
                            title: Some(title),
                            ..Default::default()
                        },
                    )
                    .await
                    {
                        Ok(()) => ui.note(format!(
                            "Title set to: {}",
                            crate::text::sanitize_to_line(&shown, usize::MAX)
                        )),
                        Err(error) => ui.reply(format!("Cannot set the title: {error}")),
                    }
                }
                Err(error) => ui.reply(error.to_string()),
            },
            (Some(id), None) => match store.session_info(*id).await {
                Ok(Some(info)) if !info.title.is_empty() => {
                    ui.reply(crate::text::sanitize_to_line(&info.title, usize::MAX));
                }
                Ok(_) => ui.reply("No title yet."),
                Err(error) => ui.reply(format!("Cannot read the title: {error}")),
            },
        },
        SlashCommand::Compact(instructions) => {
            let request = crate::session::CompactRequest {
                trigger: crate::conversation::CompactionTrigger::Manual,
                instructions: instructions
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string),
                keep_recent: None,
                request_in_flight: None,
            };
            match crate::host::terminal::compact_interruptible(
                cancel,
                cancel.admit(),
                agent,
                &mut session_id,
                messages,
                request,
            )
            .await
            {
                // The row is the compaction's own, drawn from the event every compaction emits.
                Ok(_) => {}
                Err(error) => ui.report_failure(&error),
            }
        }
        SlashCommand::RewindInvalid(argument) => {
            ui.reply(format!(
                "/rewind takes a turn count of 1 or more, not '{argument}'"
            ));
        }
        SlashCommand::Rewind(turns) => {
            let turns = turns.unwrap_or(1);
            // `rewind(0)` returns `None` unconditionally; the count is what is wrong, not the
            // conversation.
            let rewound = if turns == 0 {
                None
            } else {
                messages.rewind(turns)
            };
            match (session_id, rewound) {
                (Some(id), Some(event)) => {
                    if let Err(error) = store.save_rewind(id, &event).await {
                        // Put the turns back rather than leave memory and disk disagreeing, which
                        // would resurrect them on the next resume.
                        messages.pop_repair();
                        ui.reply(error.to_string());
                    } else {
                        agent.reset_conversation_markers(messages).await;
                        // The log's own rewind row says so once the screen reads it again.
                        ui.reload();
                    }
                }
                // No session means nothing was ever persisted, so the in-memory rewind is the
                // whole story.
                (None, Some(_)) => {
                    agent.reset_conversation_markers(messages).await;
                    ui.note(format!(
                        "Rewound {turns} turn{}.",
                        if turns == 1 { "" } else { "s" }
                    ));
                    ui.reload();
                }
                (_, None) if turns == 0 => {
                    ui.reply("Nothing to rewind: /rewind takes a turn count of 1 or more.");
                }
                (_, None) => {
                    ui.reply(format!(
                        "Nothing to rewind: the conversation has fewer than {turns} turn(s)."
                    ));
                }
            }
        }
        SlashCommand::Export => match &session_id {
            Some(id) => {
                match crate::cli::session::export_session(
                    store,
                    *id,
                    None,
                    cli::SessionExportFormat::Markdown,
                )
                .await
                {
                    // Shown absolute: `/cd` moves the session's directory while the export lands
                    // in the process's, so a bare filename would point at the wrong one.
                    Ok(Some(path)) => {
                        let shown = std::env::current_dir()
                            .map(|dir| dir.join(&path))
                            .unwrap_or(path);
                        ui.note(format!("Exported session to {}", shown.display()));
                    }
                    Ok(None) => {}
                    Err(error) => ui.reply(error.to_string()),
                }
            }
            None => ui.reply("No active session to export."),
        },
        SlashCommand::Fork => match session_id {
            Some(id) => match crate::host::fork_and_lock(store, id).await {
                Ok(crate::host::ForkHandoff::Switched { id, lock }) => {
                    // Replacing the slot's contents drops the original guard only now that the new
                    // one is held; see `crate::host::fork_and_lock`.
                    crate::host::terminal::hold_session_lock(session_lock, Some(lock));
                    session_id = Some(id);
                    // The agent's cell is what every tool reads, so the copy has to move it too.
                    agent.cells().session_id.set(id);
                    // `messages` is deliberately untouched, so the branch happens at the current
                    // head and the next turn continues in the copy.
                    ui.session_id("Forked session", &id.to_string());
                    ui.reload();
                }
                Ok(crate::host::ForkHandoff::LockFailed { id, error }) => {
                    ui.reply(format!(
                        "{error}\nStaying in the original. The copy exists: {id}"
                    ));
                }
                Ok(crate::host::ForkHandoff::SourceGone) => {
                    ui.reply(crate::error::MekaError::SessionNotFound(id).to_string());
                }
                Err(error) => ui.reply(format!("Failed to fork session: {error}")),
            },
            None => ui.reply("No active session to fork."),
        },
        // A picker rather than a table: each server with its state, and the pick offers the
        // three verbs below for it. The credential orphans `meka mcp list` reports are the CLI's.
        SlashCommand::McpList => {
            if config.mcp_servers.is_empty() {
                ui.reply("No MCP servers.");
            } else {
                let mut servers = Vec::with_capacity(config.mcp_servers.len());
                for server in &config.mcp_servers {
                    let entry = mcp_manager
                        .as_ref()
                        .and_then(|manager| manager.server_entry(&server.name));
                    let state = match entry {
                        Some(entry) => match entry.state().await {
                            crate::mcp::ServerState::Failed { error, .. } => format!(
                                "failed: {}",
                                crate::text::sanitize_to_line(&error, usize::MAX)
                            ),
                            state @ crate::mcp::ServerState::Connected { .. } => {
                                match entry.protocol_version().await {
                                    Some(version) => format!("{}, MCP {version}", state.label()),
                                    None => state.label().to_string(),
                                }
                            }
                            state => state.label().to_string(),
                        },
                        None => "(unknown)".to_string(),
                    };
                    servers.push(super::frontend::McpServerRow {
                        name: server.name.clone(),
                        state,
                    });
                }
                ui.mcp_servers(servers);
            }
        }
        // These three report success at `info!` and print nothing, which is right for the CLI
        // (the exit code carries it) and wrong here, where silence is indistinguishable from the
        // command never having run.
        SlashCommand::McpReconnect { server } => {
            let (outcome, printed) = capture(crate::cli::mcp::run_reconnect(
                &config.mcp_servers,
                token_store,
                &server,
            ))
            .await;
            if !printed.trim().is_empty() {
                ui.reply(printed);
            }
            match outcome {
                // "Connected", not "Reconnected": this is a smoke test on a throwaway client, and
                // the session's own connection to that server is untouched.
                Ok(()) => ui.note(format!("Connected to '{server}'.")),
                Err(error) => ui.reply(error.to_string()),
            }
        }
        SlashCommand::McpLogin { server } => {
            let (outcome, printed) = capture(crate::cli::mcp::run_login(
                &config.mcp_servers,
                token_store,
                &server,
                Arc::new(ScreenLogin::new(ui.clone(), server.clone())),
            ))
            .await;
            if !printed.trim().is_empty() {
                ui.reply(printed);
            }
            match outcome {
                Ok(()) => ui.note(format!("Authorized '{server}'.")),
                Err(error) => ui.reply(error.to_string()),
            }
        }
        SlashCommand::McpLogout { server } => {
            let (outcome, printed) = capture(crate::cli::mcp::run_logout(
                &config.mcp_servers,
                token_store,
                &server,
            ))
            .await;
            if !printed.trim().is_empty() {
                ui.reply(printed);
            }
            match outcome {
                Ok(()) => ui.note(format!("Cleared credentials for '{server}'.")),
                Err(error) => ui.reply(error.to_string()),
            }
        }
        SlashCommand::McpMissingServer { verb } => {
            ui.reply(format!("/mcp {verb} takes a server name"));
        }
        SlashCommand::McpUnknownVerb { verb } => {
            ui.reply(format!(
                "'{verb}' is not an `/mcp` verb: `/mcp` takes {} or <server>:<prompt>",
                super::input::MCP_SUBCOMMANDS.join(", ")
            ));
        }
        SlashCommand::McpPrompt {
            server,
            prompt: prompt_name,
            args,
        } => 'prompt: {
            let Some(manager) = mcp_manager.as_ref() else {
                ui.reply("No MCP servers.");
                break 'prompt;
            };
            let Some(entry) = manager.server_entry(&server) else {
                ui.reply(crate::text::unknown_name(
                    "MCP server",
                    &server,
                    manager.server_names(),
                ));
                break 'prompt;
            };
            // Map positional args to the declared prompt argument names.
            let arg_names = match mcp::list_prompts(
                &entry,
                &tokio_util::sync::CancellationToken::new(),
            )
            .await
            {
                Ok(prompts) => prompts
                    .into_iter()
                    .find(|prompt| prompt.name == prompt_name)
                    .and_then(|prompt| prompt.arguments)
                    .map(|declared| {
                        declared
                            .into_iter()
                            .map(|argument| argument.name)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
                Err(error) => {
                    ui.reply(error.to_string());
                    Vec::new()
                }
            };
            let mut arguments: Option<serde_json::Map<String, serde_json::Value>> = None;
            if !arg_names.is_empty() {
                let mut map = serde_json::Map::new();
                for (index, name) in arg_names.iter().enumerate() {
                    if let Some(value) = args.get(index) {
                        map.insert(name.clone(), serde_json::Value::String(value.clone()));
                    }
                }
                arguments = Some(map);
            }
            match mcp::get_prompt(
                &entry,
                prompt_name.clone(),
                arguments,
                &tokio_util::sync::CancellationToken::new(),
            )
            .await
            {
                Ok(result) => {
                    // The prompt's messages as one user turn, the shape `mcp_prompt_get` produces.
                    let mut body = String::new();
                    for message in &result.messages {
                        let role = match message.role {
                            rmcp::model::Role::User => "user",
                            rmcp::model::Role::Assistant => "assistant",
                        };
                        if let rmcp::model::ContentBlock::Text(text) = &message.content {
                            body.push_str(&format!("{}: {}\n", role, text.text));
                        }
                    }
                    let user_input = body.trim().to_string();
                    if user_input.is_empty() {
                        ui.reply(format!(
                            "'{server}:{prompt_name}' rendered an empty prompt."
                        ));
                    } else {
                        let Ok(input) = crate::agent::TurnInput::from_parts(user_input, Vec::new())
                        else {
                            ui.reply("Nothing to send.");
                            break 'prompt;
                        };
                        ui.note(format!("Running prompt '{server}:{prompt_name}'."));
                        match crate::host::terminal::run_turn_interruptible(
                            cancel,
                            cancel.admit(),
                            agent,
                            &mut session_id,
                            messages,
                            input,
                        )
                        .await
                        {
                            Ok(_) => {}
                            Err(error) => ui.report_failure(&error),
                        }
                    }
                }
                Err(error) => ui.reply(error.to_string()),
            }
        }
        // The row moves first. If recording the change failed but the agent had already
        // switched, the next `meka -c` would silently go back to the old profile.
        SlashCommand::Profile(Some(name)) => 'switch: {
            let resolved = match crate::host::resolve_profile_switch(shared, &name).await {
                Ok(resolved) => resolved,
                Err(error) => {
                    ui.reply(error.to_string());
                    break 'switch;
                }
            };
            if let Some(id) = session_id
                && let Err(error) = crate::host::record_profile_switch(store, id, &resolved).await
            {
                ui.reply(error.to_string());
                break 'switch;
            }
            agent.set_provider(resolved);
            *crate::sync::write(current_profile) = name.clone();
            ui.note(format!("Profile set to: {name}"));
        }
        // The row moves first, for the reason the profile's does: the row is where every other
        // process reads which tier this session's memories are in.
        SlashCommand::Project(ProjectVerb::Open(name)) => 'open: {
            let project = match crate::host::resolve_project_switch(shared, &name).await {
                Ok(project) => project,
                Err(error) => {
                    ui.reply(error.to_string());
                    break 'open;
                }
            };
            if let Some(id) = session_id
                && let Err(error) =
                    crate::host::record_project_switch(store, id, Some(&project)).await
            {
                ui.reply(error.to_string());
                break 'open;
            }
            agent.set_project(Some(project.clone()));
            ui.note(format!("Project set to: {project}"));
        }
        SlashCommand::Project(ProjectVerb::Close) => 'close: {
            if let Some(id) = session_id
                && let Err(error) = crate::host::record_project_switch(store, id, None).await
            {
                ui.reply(error.to_string());
                break 'close;
            }
            agent.set_project(None);
            ui.note("Project set to: none");
        }
        SlashCommand::Project(ProjectVerb::Unknown(_)) => {
            ui.reply("Usage: /project [open <name>|close]");
        }
        SlashCommand::Resume(Some(value)) => {
            if let Err(error) = attach_session(&value, AttachContext {
                store,
                config,
                shared,
                agent,
                session_id: session_id_cell,
                session_lock,
                messages,
                current_profile,
                launch_cwd,
                ui,
            })
            .await
            {
                ui.reply(error.to_string());
            }
        }
        SlashCommand::SkillInvoke { name, extra } => 'invoke: {
            let installed = agent.skills().current().await;
            let Some(skill) = installed.find(&name) else {
                let message = match installed.skip_reason(&name) {
                    // A file that is there and unreadable is not a missing skill.
                    Some(_) => installed.unavailable(&name),
                    None => crate::text::unknown_name(
                        "skill",
                        &name,
                        installed.skills.iter().map(|skill| skill.name.as_str()),
                    ),
                };
                ui.reply(message);
                break 'invoke;
            };
            let body = match skills::load_skill_body(skill).await {
                Ok(body) => body,
                Err(error) => {
                    ui.reply(format!("failed to load skill '{name}': {error}"));
                    break 'invoke;
                }
            };
            // The user's directive first, then the skill's body as the method.
            let body = if extra.is_empty() {
                body
            } else {
                format!("{extra}\n\n{body}")
            };
            let Ok(input) = crate::agent::TurnInput::from_parts(body, Vec::new()) else {
                ui.reply("Nothing to send.");
                break 'invoke;
            };
            // The body is the model's to read; the row says what the person did.
            ui.note(format!("Running skill '{name}'."));
            match crate::host::terminal::run_turn_interruptible(
                cancel,
                cancel.admit(),
                agent,
                &mut session_id,
                messages,
                input,
            )
            .await
            {
                Ok(_) => {}
                Err(error) => ui.report_failure(&error),
            }
        }
        SlashCommand::Status => {
            ui.reply(crate::host::format_status(agent, providers, messages.len()).await);
        }
        SlashCommand::Usage => {
            let body = match agent.fetch_usage().await {
                Ok(Some(usage)) => render::format_account_usage(&usage),
                Ok(None) => "Account usage is not available for this backend.".to_string(),
                Err(error) => error.to_string(),
            };
            ui.reply(body);
        }
        // The screen answers these itself, listed rather than swept into a wildcard, so adding a
        // variant fails here until someone has decided which side owns it.
        SlashCommand::Cd(_)
        | SlashCommand::Exit
        | SlashCommand::Help
        | SlashCommand::Permission(_)
        | SlashCommand::Approvals(_)
        | SlashCommand::Profile(None)
        | SlashCommand::Project(ProjectVerb::Picker)
        | SlashCommand::Resume(None)
        | SlashCommand::Session
        | SlashCommand::SkillList
        | SlashCommand::Log
        | SlashCommand::MemoryList
        | SlashCommand::MemoryShow { .. }
        | SlashCommand::ScheduleList
        | SlashCommand::ScheduleShow { .. }
        | SlashCommand::ScheduleCancel { .. }
        | SlashCommand::TaskList
        | SlashCommand::TaskShow { .. }
        | SlashCommand::TaskCancel { .. } => {
            debug_assert!(
                false,
                "`answered_by` says the screen answers this, so it should not have been forwarded"
            );
        }
    }
    *session_id_cell = session_id;
    ui.done();
}

/// The screen's side of an interactive MCP login. The terminal is the screen's, so the URL cannot
/// go to stderr and a pasted callback cannot be read from stdin: both are one question in the
/// composer's place, the URL in its message and the paste as its one field, and Esc on it ends
/// the login.
struct ScreenLogin {
    ui: Ui,
    server: String,
    url: std::sync::Mutex<Option<String>>,
    seconds: std::sync::atomic::AtomicU64,
}

impl ScreenLogin {
    fn new(ui: Ui, server: String) -> Self {
        Self {
            ui,
            server,
            url: std::sync::Mutex::new(None),
            seconds: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

#[async_trait::async_trait]
impl mcp::auth::LoginPrompt for ScreenLogin {
    fn authorize_at(&self, url: &str) {
        *crate::sync::lock(&self.url) = Some(url.to_string());
    }

    fn awaiting_callback(&self, seconds: u64, _accepts_paste: bool) {
        self.seconds
            .store(seconds, std::sync::atomic::Ordering::Relaxed);
    }

    fn accepts_paste(&self) -> bool {
        true
    }

    async fn read_pasted_line(&self) -> std::result::Result<Option<String>, String> {
        let url = crate::sync::lock(&self.url).clone().unwrap_or_default();
        let seconds = self.seconds.load(std::sync::atomic::Ordering::Relaxed);
        let (responder, receiver) = tokio::sync::oneshot::channel();
        self.ui.send(super::frontend::UiEvent::Elicitation {
            prompt: crate::frontend::ElicitationPrompt {
                server_name: self.server.clone(),
                kind: crate::frontend::ElicitationKind::Form {
                    schema: serde_json::json!({
                        "properties": {
                            "callback_url": {
                                "type": "string",
                                "description": "paste the callback URL from the browser's \
                                                address bar if the browser cannot reach meka, \
                                                or leave empty to keep waiting"
                            }
                        }
                    }),
                },
                message: format!(
                    "To authorize, open this URL in your browser: {url} (waiting up to \
                     {seconds}s for the callback)"
                ),
            },
            responder,
        });
        match receiver.await {
            Ok(crate::frontend::ElicitationResponse::Accept { content }) => Ok(content
                .as_ref()
                .and_then(|content| content.get("callback_url"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)),
            Ok(
                crate::frontend::ElicitationResponse::Decline
                | crate::frontend::ElicitationResponse::Cancel,
            ) => Err("the login was declined".to_string()),
            // The screen is gone; the listener alone may still be answered.
            Err(_) => Ok(None),
        }
    }
}

/// What `/project` was asked to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ProjectVerb {
    /// Bare `/project`: the picker over the configured projects.
    Picker,
    Open(String),
    Close,
    /// Anything else after `/project`, answered with the usage line.
    Unknown(String),
}

/// One line typed after `/`, parsed; which side answers it is [`Self::answered_by`]'s to say.
pub(crate) enum SlashCommand {
    Exit,
    Help,
    Session,
    /// `/title [text]`: show this session's title, or set it.
    Title(Option<String>),
    Permission(Option<String>),
    /// `/approvals [on|off]`: show or set whether calls above the level are submitted for
    /// approval.
    Approvals(Option<String>),
    /// `/profile [name]`: show which profile this session runs on, or move it to another.
    Profile(Option<String>),
    /// `/project`, `/project open <name>`, `/project close`: pick the project this session is on,
    /// move it onto one, or take it off its project.
    Project(ProjectVerb),
    /// `/resume [session]`: pick a session to switch this screen to, or switch to one by id or
    /// unique prefix.
    Resume(Option<String>),
    /// `/compact [instructions]`: compact now, optionally saying what to keep or drop.
    Compact(Option<String>),
    Export,
    /// `/fork`: copy the current session and continue in the copy.
    Fork,
    Cd(Option<String>),
    /// `/mcp <server>:<prompt> [args...]`: render an MCP prompt and send its messages as the next
    /// user turn.
    McpPrompt {
        server: String,
        prompt: String,
        args: Vec<String>,
    },
    McpList,
    McpReconnect {
        server: String,
    },
    McpLogin {
        server: String,
    },
    McpLogout {
        server: String,
    },
    /// `/mcp reconnect`, `/mcp login` or `/mcp logout` typed without the server it acts on.
    McpMissingServer {
        verb: String,
    },
    /// `/mcp <word>` where the word is neither a verb nor a `<server>:<prompt>` spec.
    McpUnknownVerb {
        verb: String,
    },
    MemoryList,
    ScheduleList,
    ScheduleShow {
        id: String,
    },
    ScheduleCancel {
        id: String,
    },
    TaskList,
    TaskShow {
        id: String,
    },
    /// `/task cancel <id>`, or `/task cancel --all`; `None` means every running task.
    TaskCancel {
        id: Option<String>,
    },
    MemoryShow {
        name: String,
    },
    SkillList,
    /// `/skill <name> [extra...]`: invoke a skill, with anything after the name prepended to its
    /// body so the model reads the user's directive first.
    SkillInvoke {
        name: String,
        extra: String,
    },
    Status,
    Usage,
    /// The recent log lines the screen kept, at every level.
    Log,
    /// `/rewind [N]`: drop the last `N` turns (default 1), cutting at a clean user boundary.
    Rewind(Option<usize>),
    /// `/rewind` with an argument that is not a turn count.
    RewindInvalid(String),
}

/// Who answers a slash command: the screen, when it has everything to hand, or the host loop,
/// when the command needs the live `Agent`, the conversation, or the session id `/fork` moves.
/// A command the host answers is refused while the host is busy, so the screen's side is also
/// what can run during a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Answerer {
    Screen,
    Host,
}

impl SlashCommand {
    /// Exhaustive, and both sides read it, so a new variant fails to compile until both have been
    /// considered.
    pub(crate) fn answered_by(&self) -> Answerer {
        match self {
            SlashCommand::Cd { .. } => Answerer::Screen,
            SlashCommand::Exit => Answerer::Screen,
            SlashCommand::Help => Answerer::Screen,
            SlashCommand::Permission { .. } => Answerer::Screen,
            SlashCommand::Approvals { .. } => Answerer::Screen,
            SlashCommand::Profile(None) => Answerer::Screen,
            SlashCommand::Profile(Some(_)) => Answerer::Host,
            SlashCommand::Project(ProjectVerb::Picker) => Answerer::Screen,
            SlashCommand::Project(_) => Answerer::Host,
            SlashCommand::Resume(None) => Answerer::Screen,
            SlashCommand::Resume(Some(_)) => Answerer::Host,
            SlashCommand::Compact { .. } => Answerer::Host,
            SlashCommand::Export => Answerer::Host,
            SlashCommand::Fork => Answerer::Host,
            SlashCommand::McpList => Answerer::Host,
            SlashCommand::McpLogin { .. } => Answerer::Host,
            SlashCommand::McpLogout { .. } => Answerer::Host,
            SlashCommand::McpMissingServer { .. } => Answerer::Host,
            SlashCommand::McpUnknownVerb { .. } => Answerer::Host,
            SlashCommand::McpPrompt { .. } => Answerer::Host,
            SlashCommand::McpReconnect { .. } => Answerer::Host,
            SlashCommand::MemoryList => Answerer::Screen,
            SlashCommand::MemoryShow { .. } => Answerer::Screen,
            SlashCommand::Rewind { .. } => Answerer::Host,
            SlashCommand::Title { .. } => Answerer::Host,
            SlashCommand::RewindInvalid(_) => Answerer::Host,
            SlashCommand::ScheduleCancel { .. } => Answerer::Screen,
            SlashCommand::ScheduleList => Answerer::Screen,
            SlashCommand::ScheduleShow { .. } => Answerer::Screen,
            SlashCommand::Session => Answerer::Screen,
            SlashCommand::SkillInvoke { .. } => Answerer::Host,
            SlashCommand::SkillList => Answerer::Screen,
            SlashCommand::Status => Answerer::Host,
            SlashCommand::TaskCancel { .. } => Answerer::Screen,
            SlashCommand::TaskList => Answerer::Screen,
            SlashCommand::TaskShow { .. } => Answerer::Screen,
            SlashCommand::Usage => Answerer::Host,
            SlashCommand::Log => Answerer::Screen,
        }
    }

    /// The refusal for a host-answered command typed while a turn runs: the command word alone,
    /// since that is what the rule is about.
    pub(crate) fn busy_refusal(line: &str) -> String {
        let command = line.split(char::is_whitespace).next().unwrap_or(line);
        format!("`{command}` cannot run while a turn is running.")
    }
}

/// The command a typed line names; `None` for a line that is not one the screen knows.
pub(crate) fn parse_slash_command(input: &str) -> Option<SlashCommand> {
    let input = input.strip_prefix('/')?;
    let mut parts = input.splitn(2, char::is_whitespace);
    let command = parts.next()?;
    let argument = parts.next().map(|s| s.trim().to_string());

    match command {
        "exit" | "quit" => Some(SlashCommand::Exit),
        "help" | "?" => Some(SlashCommand::Help),
        "session" => Some(SlashCommand::Session),
        "title" => Some(SlashCommand::Title(
            argument.filter(|text| !text.is_empty()),
        )),
        "memory" => Some(parse_memory_slash(argument.as_deref().unwrap_or(""))),
        "schedule" => Some(parse_schedule_slash(argument.as_deref().unwrap_or(""))),
        "task" => Some(parse_task_slash(argument.as_deref().unwrap_or(""))),
        "permission" => Some(SlashCommand::Permission(argument)),
        "approvals" => Some(SlashCommand::Approvals(argument)),
        "profile" => Some(SlashCommand::Profile(argument)),
        "project" => Some(parse_project_slash(argument.as_deref().unwrap_or(""))),
        "resume" => Some(SlashCommand::Resume(argument)),
        "compact" => Some(SlashCommand::Compact(argument)),
        "rewind" => Some(match argument.as_deref().map(str::trim) {
            None | Some("") => SlashCommand::Rewind(None),
            Some(value) => match value.parse::<usize>() {
                Ok(turns) => SlashCommand::Rewind(Some(turns)),
                Err(_) => SlashCommand::RewindInvalid(value.to_string()),
            },
        }),
        "export" => Some(SlashCommand::Export),
        "fork" => Some(SlashCommand::Fork),
        "cd" => Some(SlashCommand::Cd(argument)),
        "mcp" => Some(parse_mcp_slash(argument.as_deref().unwrap_or(""))),
        "skill" => Some(parse_skill_slash(argument.as_deref().unwrap_or(""))),
        "status" => Some(SlashCommand::Status),
        "usage" => Some(SlashCommand::Usage),
        "log" => Some(SlashCommand::Log),
        _ => None,
    }
}

/// `/project`: bare opens the picker; `open <name>` moves the session onto a project; `close`
/// takes it off its project.
fn parse_project_slash(rest: &str) -> SlashCommand {
    let rest = rest.trim();
    if rest.is_empty() {
        return SlashCommand::Project(ProjectVerb::Picker);
    }
    if rest == "close" {
        return SlashCommand::Project(ProjectVerb::Close);
    }
    match rest.strip_prefix("open").map(str::trim) {
        Some(name) if !name.is_empty() && !name.contains(char::is_whitespace) => {
            SlashCommand::Project(ProjectVerb::Open(name.to_string()))
        }
        _ => SlashCommand::Project(ProjectVerb::Unknown(rest.to_string())),
    }
}

/// `/memory`: bare lists; otherwise the whole argument is a memory name to display.
fn parse_memory_slash(rest: &str) -> SlashCommand {
    let rest = rest.trim();
    if rest.is_empty() {
        return SlashCommand::MemoryList;
    }
    SlashCommand::MemoryShow {
        name: rest.to_string(),
    }
}

/// `/schedule`: bare lists; `show <id>` prints one in full; `cancel <id>` cancels. There is no
/// `create`: a job's prompt is prose the agent writes for its own future self.
fn parse_schedule_slash(rest: &str) -> SlashCommand {
    let rest = rest.trim();
    if let Some(id) = rest.strip_prefix("show").map(str::trim)
        && !id.is_empty()
    {
        return SlashCommand::ScheduleShow { id: id.to_string() };
    }
    match rest.strip_prefix("cancel").map(str::trim) {
        Some(id) if !id.is_empty() => SlashCommand::ScheduleCancel { id: id.to_string() },
        _ => SlashCommand::ScheduleList,
    }
}

/// `/task`: bare lists; `show <id>`; `cancel <id>`; `cancel --all`.
fn parse_task_slash(rest: &str) -> SlashCommand {
    let rest = rest.trim();
    if let Some(id) = rest.strip_prefix("show").map(str::trim)
        && !id.is_empty()
    {
        return SlashCommand::TaskShow { id: id.to_string() };
    }
    match rest.strip_prefix("cancel").map(str::trim) {
        Some("--all" | "all") => SlashCommand::TaskCancel { id: None },
        Some(id) if !id.is_empty() => SlashCommand::TaskCancel {
            id: Some(id.to_string()),
        },
        // A bare `cancel` names nothing; listing is the safe reading, and the user sees the ids.
        _ => SlashCommand::TaskList,
    }
}

/// `/skill`: bare lists; otherwise the first token is the skill and the rest is free-form extra
/// context prepended to the body.
fn parse_skill_slash(rest: &str) -> SlashCommand {
    let rest = rest.trim();
    if rest.is_empty() {
        return SlashCommand::SkillList;
    }
    let (name, extra) = match rest.split_once(char::is_whitespace) {
        Some((name, extra)) => (name.to_string(), extra.trim().to_string()),
        None => (rest.to_string(), String::new()),
    };
    SlashCommand::SkillInvoke { name, extra }
}

/// `/mcp`: every shape parses to a command, so nothing falls through to "Unknown command".
fn parse_mcp_slash(rest: &str) -> SlashCommand {
    let rest = rest.trim();
    if rest.is_empty() || rest == "list" {
        return SlashCommand::McpList;
    }
    let (verb, argument) = rest
        .split_once(char::is_whitespace)
        .map_or((rest, ""), |(verb, argument)| (verb, argument.trim()));
    let with_server: Option<fn(String) -> SlashCommand> = match verb {
        "reconnect" => Some(|server| SlashCommand::McpReconnect { server }),
        "login" => Some(|server| SlashCommand::McpLogin { server }),
        "logout" => Some(|server| SlashCommand::McpLogout { server }),
        _ => None,
    };
    if let Some(build) = with_server {
        return if argument.is_empty() {
            SlashCommand::McpMissingServer {
                verb: verb.to_string(),
            }
        } else {
            build(argument.to_string())
        };
    }
    if let Some((server, prompt)) = verb.split_once(':')
        && !server.is_empty()
        && !prompt.is_empty()
    {
        return SlashCommand::McpPrompt {
            server: server.to_string(),
            prompt: prompt.to_string(),
            args: argument.split_whitespace().map(str::to_string).collect(),
        };
    }
    SlashCommand::McpUnknownVerb {
        verb: verb.to_string(),
    }
}

/// The line for a slash command meka does not know. It names the command and only the command:
/// `/frob a b` is refused as `/frob`, because the argument was never read.
pub(crate) fn unknown_command_message(line: &str) -> String {
    let command = line.split(char::is_whitespace).next().unwrap_or(line);
    format!("Unknown command: {command}. Type /help for available commands.")
}

/// The `/mcp` subcommands are arguments, not top-level commands, so they are absent from
/// `COMMANDS`; `/help` lists them under `/mcp` so it still documents the full grammar.
const MCP_HELP_ROWS: [(&str, &str); 5] = [
    ("/mcp list", "Pick a server to reconnect, log in or log out"),
    (
        "/mcp reconnect <server>",
        "Reconnect smoke-test for one server",
    ),
    ("/mcp login <server>", "Run the OAuth flow for a server"),
    (
        "/mcp logout <server>",
        "Clear stored credentials for a server",
    ),
    (
        "/mcp <server>:<prompt> [args]",
        "Render an MCP prompt as the next turn",
    ),
];

/// The `Commands:` block of `/help`: one row per command, every description starting in the same
/// column.
fn help_command_rows() -> String {
    let mut rows: Vec<(String, &str)> = Vec::new();
    for command in crate::host::COMMANDS {
        let left = if command.arg_hint.is_empty() {
            format!("/{}", command.name)
        } else {
            format!("/{} {}", command.name, command.arg_hint)
        };
        rows.push((left, command.help));
        if command.name == "mcp" {
            rows.extend(
                MCP_HELP_ROWS
                    .iter()
                    .map(|(left, help)| ((*left).to_string(), *help)),
            );
        }
    }
    let width = rows
        .iter()
        .map(|(left, _)| crate::text::display_width(left))
        .max()
        .unwrap_or(0);
    let mut out = String::from("Commands:\n");
    for (left, help) in &rows {
        out.push_str("  ");
        out.push_str(&crate::text::format_columns_row(&[left, help], &[width]));
    }
    out
}

/// What `/help` shows.
pub(crate) fn help_text() -> String {
    let mut out = help_command_rows();
    out.push_str("\nShortcuts:\n");
    out.push_str("  !<command>     Run a shell command in the session's directory\n");
    out.push_str("  Shift+Tab      Cycle the permission level\n");
    out.push_str("  Alt+Enter      Insert a newline\n");
    out.push_str("  Ctrl+T         Show or hide every thinking block\n");
    out.push_str("  Ctrl+O         Expand or collapse every tool call\n");
    out.push_str("  Alt+Up/Down    Pick a block; Enter or Space toggles it\n");
    out.push_str("  Alt+S          Cycle how Enter sends while a turn runs\n");
    out.push_str("  Ctrl+R         Search what you sent before\n");
    out.push_str("  Ctrl+V         Attach the image on the clipboard to the next prompt\n");
    out.push_str("  PgUp/PgDn      Scroll the transcript; older turns load as you reach the top\n");
    out.push_str("  Ctrl+Home/End  The top of what is loaded; back to the end\n");
    out.push_str("  Esc, Ctrl+C    Stop the turn, then the background tasks, then leave\n");
    out.push_str("  Esc, q         Back to the bottom, or close what is open, when nothing runs\n");
    out.push_str("  Ctrl+L         Redraw the screen\n");
    out.push_str("  Ctrl+D         Leave\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_help_description_starts_in_the_same_column() {
        let text = help_command_rows();
        // A description follows the one run of two or more spaces on its row.
        let columns: std::collections::BTreeSet<usize> = text
            .lines()
            .skip(1)
            .map(|row| row.rfind("  ").map_or(0, |gap| gap + 2))
            .collect();
        assert_eq!(columns.len(), 1, "{text}");
    }

    #[test]
    fn the_mcp_grammar_parses_every_shape_to_a_command() {
        assert!(matches!(
            parse_slash_command("/mcp"),
            Some(SlashCommand::McpList)
        ));
        assert!(matches!(
            parse_slash_command("/mcp login"),
            Some(SlashCommand::McpMissingServer { verb }) if verb == "login"
        ));
        assert!(matches!(
            parse_slash_command("/mcp exa:search a b"),
            Some(SlashCommand::McpPrompt { server, prompt, args }) if server == "exa" && prompt == "search" && args.len() == 2
        ));
        assert!(matches!(
            parse_slash_command("/mcp frob"),
            Some(SlashCommand::McpUnknownVerb { verb }) if verb == "frob"
        ));
        assert!(parse_slash_command("/frob").is_none());
    }

    /// Every shape `/project` takes parses to its verb, and anything else is answered with the
    /// usage line rather than taken as a name.
    #[test]
    fn the_project_grammar_parses_every_shape_to_a_verb() {
        assert!(matches!(
            parse_slash_command("/project"),
            Some(SlashCommand::Project(ProjectVerb::Picker))
        ));
        assert!(matches!(
            parse_slash_command("/project open webdev"),
            Some(SlashCommand::Project(ProjectVerb::Open(name))) if name == "webdev"
        ));
        assert!(matches!(
            parse_slash_command("/project close"),
            Some(SlashCommand::Project(ProjectVerb::Close))
        ));
        for line in [
            "/project webdev",
            "/project open",
            "/project open a b",
            "/project drop x",
        ] {
            assert!(
                matches!(
                    parse_slash_command(line),
                    Some(SlashCommand::Project(ProjectVerb::Unknown(_)))
                ),
                "{line}"
            );
        }
    }

    /// `/resume` swaps the conversation for another session's, so what it calls is the one reset
    /// every rewrite calls: a door that resets its own subset leaves the rest stale.
    #[test]
    fn the_switch_resets_every_marker_the_conversation_held() {
        let attach = include_str!("commands.rs")
            .split("async fn attach_session(")
            .nth(1)
            .expect("the attach step")
            .split("\n}\n")
            .next()
            .expect("the step ends");
        assert!(attach.contains("reset_conversation_markers("), "{attach}");
        assert!(
            !attach.contains("hydrate_checklist("),
            "the reset reads the checklist itself: {attach}"
        );
    }
}
