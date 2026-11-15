//! The interactive host: the terminal UI that draws the conversation from what it knows, and the
//! loop that owns the session behind it.
//!
//! Two tasks. The screen ([`screen`]) owns the terminal: it draws, reads the keyboard and the
//! mouse, and sends the host what the user asked for. This loop owns the session: it runs turns,
//! answers the commands that need the live agent, and tells the screen what happened through the
//! same stream the agent's own events travel on ([`frontend`]).

pub(crate) mod commands;
pub(crate) mod frontend;
pub(crate) mod input;
pub(crate) mod model;
pub(crate) mod panel;
pub(crate) mod screen;
pub(crate) mod view;

use self::frontend::{HostEvent, Ui, UiEvent};
use super::{terminal::*, *};

/// Lines `tracing` can queue for the screen before the oldest falls through to stderr.
const LOG_QUEUE_LINES: usize = 512;

/// Entries the recall keeps.
const RECALL_CAPACITY: usize = 5000;

/// Run the interactive host: resume or create the session, open the screen, and answer what it
/// sends until the user leaves.
pub(crate) async fn run_interactive(
    config: ResolvedConfig,
    store: Store,
    token_store: TokenStore,
    opening: Option<crate::conversation::ContentBlock>,
    mcp_manager: Option<Arc<crate::mcp::McpClientManager>>,
) -> anyhow::Result<()> {
    let config = Arc::new(config);
    // Where the shell was. Kept apart from the per-session `cwd` below because the two part
    // company the moment a resumed session opens somewhere else, and a bare `/cd` returns here.
    let launch_cwd = std::env::current_dir().unwrap_or_else(|error| {
        tracing::warn!("failed to read the process working directory at startup: {error}");
        std::path::PathBuf::from(".")
    });

    // From here on every log line is the screen's to show, including the MCP connection warnings
    // that fire while the session is still being assembled: the screen drains the queue once it
    // opens, so nothing lands on the terminal it is about to take. A refusal below hands the queue
    // back to stderr on its way out, ahead of its own line.
    let screen_logs = crate::relay::RELAY.attach_screen(LOG_QUEUE_LINES);

    let (ui_sender, ui_receiver) = tokio::sync::mpsc::unbounded_channel::<UiEvent>();
    let ui = Ui::new(ui_sender);

    let ResumedSession {
        mut session_id,
        messages,
        lock: resumed_lock,
        permission: start_permission,
        approvals: start_approvals,
        repin,
        permission_to_record,
        project,
        project_repin,
        cwd: recorded_cwd,
        announce,
    } = resolve_session_resume(&store, &config).await?;
    // After the resume, whose lock is what spares the session this run was asked for.
    let windows = RetentionWindows::from_config(&config);
    sweep_expired(windows, &store).await?;
    // A screen left open for months is a host that stays up.
    let expiry_sweeper = spawn_expiry_sweeper(windows, store.clone());

    // Where the log ends as the screen opens: the screen reads the pages below this as the view
    // scrolls up, and this run's items go after it. Read before the agent runs anything, so a
    // prompt given on the command line cannot land a row on both sides of the line.
    let log_end = match session_id {
        Some(id) => match store.last_log_row(id).await {
            Ok(end) => end,
            Err(error) => {
                tracing::warn!("failed to read the log's end; opening without history: {error}");
                None
            }
        },
        None => None,
    };

    // Per-session working directory, shared by reference between the screen (the status row and
    // `/cd`) and the agent. The process directory is never moved. A resumed session opens where it
    // recorded rather than where this shell is; see `resume_working_directory`.
    let cwd = crate::workspace::SharedCwd::new(resume_working_directory(
        recorded_cwd,
        &launch_cwd,
        session_id,
    ));

    // What the screen opens with: the banner where the log has nothing to show, the resume row,
    // then the conversation as far back as the log goes. The banner and the rows after it are
    // pushed once the agent is built, since the banner says what the agent carries.
    let mut transcript = model::Transcript::new(model::Settings {
        show_session_id_on_create: config.show_session_id_on_create,
        turn_end_summary: config.turn_end_summary.clone(),
        fold_steps: config.fold_steps,
        expand_streaming_thinking: config.expand_streaming_thinking,
        prompt_padding: config.prompt_padding,
    });
    if let Some(id) = session_id {
        match store.turn_store().load_turns(id).await {
            Ok(turns) => transcript.set_turn_records(turns),
            Err(error) => tracing::warn!("failed to read the session's turns: {error}"),
        }
    }

    // The same deps the long-lived hosts build, so all four hosts assemble a session identically.
    // `providers` is held past agent construction because `/profile` rebuilds one mid-session and
    // the cache is what makes the profile it left reusable when it comes back.
    let shared =
        match build_shared_deps(Arc::clone(&config), store.clone(), mcp_manager.clone()).await {
            Ok(shared) => Arc::new(shared),
            Err(error) => {
                screen_logs.release();
                crate::render::render_error(&error);
                return Err(AlreadyReported.into());
            }
        };
    let providers = Arc::clone(&shared.providers);

    let shared_permission = SharedPermission::new(start_permission, config.enabled_permissions)
        .with_approvals(start_approvals);
    let sandbox_state = crate::sandbox::SandboxState::new(
        config.sandbox,
        config.confine_reads,
        config.network,
        &shared.sandbox,
    );
    if start_permission == crate::permission::Permission::Read {
        crate::sandbox::warn_if_sandbox_issues(
            &sandbox_state,
            crate::sandbox::WarnContext::InitialReadLevel,
        );
    }

    let (host_sender, mut host_receiver) = tokio::sync::mpsc::unbounded_channel::<HostEvent>();

    // A prompt or skill given without `--oneshot` is the first turn's input. The screen opens
    // busy, so Enter during that turn goes where it goes during any other.
    let initial_turn_pending = opening.is_some();
    if let Some(opening) = &opening {
        #[allow(
            clippy::expect_used,
            reason = "the receiver is `host_receiver` above, still owned here, so `send` cannot fail"
        )]
        host_sender
            .send(HostEvent::UserInput {
                parts: vec![opening.clone()],
                images: Vec::new(),
            })
            .expect("a fresh channel accepts its first send");
    }
    let frontend = Arc::new(frontend::TuiFrontend::new(ui.clone()));

    // The status row's context gauge: the agent writes it after each turn and the screen reads it
    // each frame. The window is seeded from the process default and corrected to the session's own
    // as soon as the agent resolves it.
    let context_window_gauge = Arc::new(std::sync::atomic::AtomicU64::new(
        config
            .session_context_window
            .unwrap_or(crate::config::DEFAULT_CONTEXT_WINDOW),
    ));
    let context_tokens = Arc::new(std::sync::atomic::AtomicU64::new(0));
    // Seeded with an estimate on a resume, so the gauge is not blank until the first turn measures.
    if !messages.is_empty() {
        context_tokens.store(
            crate::tokens::estimate_messages(messages.as_slice()),
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    // Before the agent, because the agent resolves the row: a repin that has not landed yet would
    // build this run on the profile the session is leaving.
    if let Err(error) =
        commit_resume_repin(&store, &providers, session_id, repin, project_repin).await
    {
        screen_logs.release();
        crate::render::render_error(&error);
        return Err(AlreadyReported.into());
    }
    let agent = match build_session_agent(&shared, SessionSpec {
        session_id,
        permission: shared_permission.clone(),
        frontend: Arc::clone(&frontend) as Arc<dyn crate::frontend::Frontend>,
        cwd: cwd.clone(),
        roots: crate::workspace::SharedRoots::new(
            config.request.additional_workspace_directories.clone(),
        ),
        read_only_roots: crate::workspace::SharedRoots::new(
            config.request.additional_readable_directories.clone(),
        ),
        context_tokens: Arc::clone(&context_tokens),
        context_overhead: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        context_window: Arc::clone(&context_window_gauge),
        project,
    })
    .await
    {
        Ok(agent) => agent,
        Err(error) => {
            screen_logs.release();
            crate::render::render_error(&error);
            let gone = match session_id {
                Some(id) => crate::provider::recorded_profile_is_gone(&store, &config, id).await,
                None => false,
            };
            let move_to = config
                .default_profile
                .as_deref()
                .or_else(|| config.profiles.keys().next().map(String::as_str));
            if let Some((session_id, move_to)) = session_id.filter(|_| gone).zip(move_to) {
                crate::render::render_profile_setup_hint(crate::render::MissingSessionProfile {
                    session_id,
                    move_to,
                });
            }
            return Err(AlreadyReported.into());
        }
    };
    // After the agent, so a start the builder refused leaves the row at the level it had.
    record_resume_permission(&store, session_id, permission_to_record).await;
    if log_end.is_none() {
        // The build, then what the agent carries for the session: the servers the configuration
        // enables (whether each connected is `/mcp list`'s to say), the installed skills, the
        // memories the session's opening view holds, and the pieces of standing instructions in
        // effect.
        let memories = shared
            .memories
            .count(&crate::store::memory::MemoryView::for_session(
                agent.project().as_deref(),
            ))
            .await?;
        transcript.push(model::Item::Banner {
            version: env!("CARGO_PKG_VERSION").to_string(),
            revision: option_env!("MEKA_GIT_REVISION").map(str::to_string),
            debug: cfg!(debug_assertions),
            mcp_servers: config
                .mcp_servers
                .iter()
                .filter(|server| server.disabled != Some(true))
                .count(),
            skills: shared.skills.current().await.skills.len(),
            memories: usize::try_from(memories).unwrap_or(usize::MAX),
            instructions: agent.user_instructions_count(),
        });
    }
    if let Some(id) = announce {
        transcript.push(model::Item::SessionId {
            label: "Resuming session".to_string(),
            id: id.to_string(),
        });
    }
    // Typed words are drawn as typed; an invocation is said where its message is appended.
    if let Some(crate::conversation::ContentBlock::Text { text, .. }) = opening {
        transcript.push(model::Item::Prompt {
            text,
            class: None,
            inbox_id: None,
            delivered: true,
            ran_under: model::RanUnder::Pending,
        });
    }
    // The resumed log is what the list is; read it before the first prompt rather than at the
    // first turn, so `/status` and a `/fork` between them see the session as it was left.
    agent.hydrate_checklist(&messages).await;
    let mut agent = Arc::new(agent);
    let mut conversation = Arc::new(tokio::sync::Mutex::new(messages));
    let shared_agent = SharedAgent::new(Arc::clone(&agent));
    let cancel = crate::host::CancelCell::default();
    // In raw mode Ctrl+C is a key the screen answers; this listener is for a SIGINT sent from
    // outside, and says what it did through the screen like everything else.
    install_interrupt_handler(
        cancel.clone(),
        store.clone(),
        {
            let shared_agent = shared_agent.clone();
            move || shared_agent.get().background_tasks()
        },
        {
            let ui = ui.clone();
            move |note: &str| ui.note(note)
        },
        {
            // Through the loop below, which is what gives the terminal back and tears the
            // session down; an exit from inside the listener would leave the screen on.
            let host = host_sender.clone();
            move || {
                if host.send(HostEvent::Exit { interrupted: true }).is_err() {
                    tracing::debug!("the host loop has already ended");
                }
            }
        },
        // A hangup or a termination leaves through the loop too.
        {
            let host = host_sender.clone();
            move || {
                if host.send(HostEvent::Exit { interrupted: true }).is_err() {
                    tracing::debug!("the host loop has already ended");
                }
            }
        },
    );
    let mut resident = on_screen(session_id, &agent, &conversation, &cancel);

    // Persistent, cross-session recall in the store. On failure the composer walks nothing rather
    // than the run failing.
    let recall =
        match crate::store::recall::RecallStore::open(store.database_path(), RECALL_CAPACITY) {
            Ok(recall) => Some(recall),
            Err(error) => {
                tracing::warn!("failed to open the store for recall: {error}");
                None
            }
        };

    // The screen draws on the terminal and reads the keyboard; without one there is nothing to
    // draw on, and a scripted run has its own door. Asked here rather than at startup so a session
    // that cannot be driven at all (a profile that is gone, a credential refused) is refused for
    // its own reason first, which is what `tests/cli.rs` reaches through pipes.
    {
        use std::io::IsTerminal;
        if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
            anyhow::bail!(
                "the interactive screen needs a terminal; use `--oneshot` for a scripted run"
            );
        }
    }
    // Opened once there is an agent to answer it, which is what makes every refusal above final.
    let screen_watch = (host_sender.clone(), cancel.clone());
    let screen_handle = tokio::spawn(screen::run_screen(screen::ScreenLaunch {
        ui_receiver,
        ui: ui.clone(),
        log_end,
        host: host_sender,
        logs: screen_logs,
        transcript,
        sandbox_state,
        launch_cwd: launch_cwd.clone(),
        status_row: config.status_row.clone(),
        recall,
        completer: input::Completer {
            mcp_server_names: config
                .mcp_servers
                .iter()
                .map(|server| server.name.clone())
                .collect(),
            skill_names: Arc::new(std::sync::RwLock::new(Vec::new())),
            profile_names: config
                .profile_summaries
                .iter()
                .map(|profile| profile.name.clone())
                .collect(),
            project_names: store.project_names().await?,
        },
        skill_roots: config.skill_roots(),
        configured_profiles: config.profile_summaries.clone(),
        schedule: config.schedule.clone(),
        store: store.clone(),
        agent: shared_agent.clone(),
        running_subagents: shared.running_subagents.clone(),
        cancel: cancel.clone(),
        send_class: inbox_class(config.send_class),
        mouse: config.mouse,
        wheel_rows: config.wheel_rows,
        busy: initial_turn_pending,
        pick_session: config.request.session_resume == Some(crate::config::SessionResume::Pick),
        pick_project: config.request.requested_project.clone(),
    }));
    // Ends the host loop when the screen stops on its own, a panic included: the loop otherwise
    // waits for an exit only the screen sends, and the signal listener's sender keeps the
    // channel open. The turn is stopped with it, since nothing shows it any more.
    let screen_handle = tokio::spawn(async move {
        let outcome = screen_handle.await;
        let (host, cancel) = screen_watch;
        cancel.cancel(crate::store::turns::CancelReason::Client);
        if host.send(HostEvent::Exit { interrupted: false }).is_err() {
            tracing::debug!("the host loop has already ended");
        }
        match outcome {
            Ok(result) => result,
            Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
            Err(error) => Err(anyhow::anyhow!("the screen task was canceled: {error}")),
        }
    });

    // One slot for the session lock from here on, whichever way the session was reached.
    let mut session_lock = agent.session_lock_slot();
    hold_session_lock(&session_lock, resumed_lock);

    // Mirrors the loop's `session_id` for the watcher below, which runs on another task and cannot
    // borrow it.
    let shared_session_id = Arc::new(std::sync::RwLock::new(session_id));

    // The sweep, for the session on screen, asked of its agent on every sweep so a `/resume` and a
    // fresh session's first turn are followed: it checks a due schedule whatever the session is
    // doing and queues its fire in the inbox, which a running turn reads at its next pause, and
    // wakes the screen so an idle one gets a turn for it. Once the process is told to leave it
    // takes no schedule, since nothing would read the fire.
    let schedule_sweep = config.schedule.enabled.then(|| {
        let on_screen = shared_agent.clone();
        let ui = ui.clone();
        crate::scheduler::spawn(
            Arc::new(store.clone()),
            config.schedule.clone(),
            shared.gate_tools.clone(),
            Arc::new(ScreenResidents {
                agent: shared_agent.clone(),
            }),
            crate::scheduler::SchedulerScope::Schedules(Arc::new(
                move |schedule: &crate::schedule::Schedule| {
                    !INTERRUPT_RELAY.is_told_to_leave()
                        && on_screen.get().session_id() == Some(schedule.session_id)
                },
            )),
            move |_schedule, _queued| {
                ui.send(UiEvent::Wake);
                std::future::ready(())
            },
        )
    });

    // Watcher, not a driver: it only tells the screen something waits. This loop owns the
    // conversation, so it has to be the thing that runs the turn.
    let schedule_watcher = {
        let store = store.clone();
        let shared_session_id = Arc::clone(&shared_session_id);
        let poll_interval = config.schedule.poll_interval;
        let background_enabled = config.background.enabled;
        let ui = ui.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(poll_interval);
            ticker.tick().await;
            loop {
                ticker.tick().await;
                let Some(current) = *crate::sync::read(&shared_session_id) else {
                    continue;
                };
                // Behind no switch, unlike the one below: the inbox exists whenever the screen
                // does.
                let mut due = match store.inbox_store().pending_count(current).await {
                    Ok(pending) => pending > 0,
                    Err(error) => {
                        tracing::warn!("failed to poll the inbox: {error}");
                        false
                    }
                };
                if background_enabled {
                    match store
                        .background_store()
                        .list_undelivered_background_tasks(current)
                        .await
                    {
                        Ok(ready) if ready.iter().any(|task| task.status.wakes_a_host()) => {
                            due = true;
                        }
                        Ok(_) => {}
                        Err(error) => tracing::warn!("failed to poll background tasks: {error}"),
                    }
                }
                if due {
                    ui.send(UiEvent::Wake);
                }
            }
        })
    };

    let mut interrupted_exit = false;
    while let Some(event) = host_receiver.recv().await {
        match event {
            // Once told to leave, the process waits on the background tasks it stopped before it
            // sends the exit, and a turn opened meanwhile would run whole first. Every command is
            // refused, not only those that open a turn: the session is about to go.
            HostEvent::UserInput { .. } | HostEvent::Command(_)
                if INTERRUPT_RELAY.is_told_to_leave() =>
            {
                ui.done();
            }
            // Recorded on the row, then nothing else: this is not a turn.
            HostEvent::PermissionChanged { session, level } => {
                let Some(id) = session.get() else {
                    continue;
                };
                if let Err(error) =
                    crate::host::record_session_change(&store, id, crate::store::SessionPatch {
                        permission: Some(level),
                        ..Default::default()
                    })
                    .await
                {
                    tracing::warn!("{error}");
                }
            }
            HostEvent::ApprovalsChanged { session, approvals } => {
                let Some(id) = session.get() else {
                    continue;
                };
                if let Err(error) =
                    crate::host::record_session_change(&store, id, crate::store::SessionPatch {
                        approvals: Some(approvals),
                        ..Default::default()
                    })
                    .await
                {
                    tracing::warn!("{error}");
                }
            }
            HostEvent::CwdChanged { session, path } => {
                record_session_cwd(&store, session.get(), &path).await;
            }
            HostEvent::Wake => {
                // Sampled ahead of the drain, so a stop of a turn it opens is seen below.
                let admission = cancel.admit();
                if let Some(entry) = resident.clone() {
                    let hooks = TuiHooks::new(entry.clone(), &store, &config, &ui);
                    hooks.drain_inbox().await;
                    // Not after a stop, which may be the screen leaving: a turn opened then would
                    // run with nothing left to show or stop it.
                    if config.background.enabled && !cancel.canceled_since(admission) {
                        let sessions = crate::host::Sessions::<uuid::Uuid, SessionEntry>::new();
                        sessions
                            .write()
                            .await
                            .insert(entry.id, SessionEntry(entry.clone()));
                        // The host is leaving (a hangup, a termination, a SIGINT that is the
                        // ladder's last press, or the terminal found gone), and the `Exit` that
                        // ends the loop as interrupted is on its way: a signal listener sends it
                        // once the stopped background tasks have had their moment, and the screen
                        // has sent its own.
                        if crate::host::scheduler::deliver_ready_outcomes(&hooks, &sessions)
                            .await
                            .is_break()
                        {
                            continue;
                        }
                    }
                }
                ui.done();
            }
            HostEvent::UserInput { parts, images } => {
                // Admitted here, before anything below can wait, so a Ctrl+C in the window counts
                // against this turn rather than climbing the ladder past it.
                let admission = cancel.admit();
                let input = match crate::agent::TurnInput::sent(parts, images) {
                    Ok(input) => input,
                    Err(empty) => {
                        ui.error(&empty);
                        ui.done();
                        continue;
                    }
                };
                // An outcome that did not warrant a turn of its own rides on this one.
                let outcomes = if config.background.enabled {
                    collect_background_outcomes(&agent, &store, session_id).await
                } else {
                    Vec::new()
                };
                let input = input.riding(outcomes);
                let mut messages = conversation.lock().await;
                match run_turn_interruptible(
                    &shared,
                    &cancel,
                    admission,
                    &agent,
                    &mut session_id,
                    &mut messages,
                    input,
                )
                .await
                {
                    Ok(_) => {}
                    Err(crate::error::MekaError::Interrupted) => {
                        ui.item(model::Item::interrupted());
                        report_background_survivors_to(&agent, &ui).await;
                    }
                    Err(error) => ui.error(&error),
                }
                drop(messages);
                // Not after a stop, which may be the screen leaving: a turn opened then would run
                // with nothing left to show or stop it.
                if !cancel.canceled_since(admission)
                    && let Some(entry) = on_screen(session_id, &agent, &conversation, &cancel)
                {
                    TuiHooks::new(entry, &store, &config, &ui)
                        .drain_inbox()
                        .await;
                }
                ui.done();
            }
            HostEvent::Command(command) => {
                // Admitted here, as a prompt is: a command that opens a turn fetches what it sends
                // first, and a Ctrl+C or a signal meanwhile counts against that turn.
                let admission = cancel.admit();
                let opens_a_turn = matches!(
                    command,
                    commands::SlashCommand::SkillInvoke { .. }
                        | commands::SlashCommand::McpPrompt { .. }
                        | commands::SlashCommand::Compact(_)
                );
                let mut messages = conversation.lock().await;
                let opened = commands::answer(command, commands::HostCommandContext {
                    agent: &agent,
                    shared: &shared,
                    cancel: &cancel,
                    admission,
                    config: &config,
                    ui: &ui,
                    mcp_manager: &mcp_manager,
                    messages: &mut messages,
                    providers: &providers,
                    session_id: &mut session_id,
                    session_lock: &session_lock,
                    store: &store,
                    token_store: &token_store,
                    launch_cwd: &launch_cwd,
                })
                .await;
                drop(messages);
                // `/resume` opened the session it moves to whole; the one on screen is let go only
                // now, as a screen that leaves lets it go, and everything that reads the session on
                // screen moves to the new one in this step.
                if let Some(opened) = opened {
                    // Ahead of the switch's own word to the screen, so nothing the screen does
                    // once it has shown the session reads the one let go.
                    shared_agent.set(Arc::clone(&opened.agent));
                    // The checklist the session announced as it opened reached the screen ahead of
                    // the switch, which lets it go with the session left; said again after it, it
                    // waits for the session's first page, as a launch's does.
                    ui.switched();
                    ui.send(UiEvent::Agent(
                        crate::frontend::FrontendEvent::ChecklistUpdated {
                            items: opened.cells().checklist.get().items,
                        },
                    ));
                    crate::host::release_agent(&agent, &cancel, mcp_manager.as_ref()).await;
                    hold_session_lock(&session_lock, None);
                    agent = Arc::clone(&opened.agent);
                    conversation = Arc::clone(&opened.conversation);
                    session_lock = agent.session_lock_slot();
                    session_id = Some(opened.id);
                    if config.show_session_id_on_resume {
                        ui.session_id("Resuming session", &opened.id.to_string());
                    }
                    ui.reload();
                }
                // As after a typed prompt. Not after the other commands: one that reloads the
                // screen would have the drain's turn drawn while the screen reads the log again.
                if opens_a_turn
                    && !cancel.canceled_since(admission)
                    && let Some(entry) = on_screen(session_id, &agent, &conversation, &cancel)
                {
                    TuiHooks::new(entry, &store, &config, &ui)
                        .drain_inbox()
                        .await;
                }
                ui.done();
            }
            HostEvent::Exit { interrupted } => {
                interrupted_exit = interrupted;
                break;
            }
        }
        // The loop's own `session_id` is authoritative; the watcher reads this mirror.
        *crate::sync::write(&shared_session_id) = session_id;
        if resident.as_ref().map(|entry| entry.id) != session_id {
            // The loop has moved from one session to another (`/fork`, `/resume`), so what was
            // answered "for the rest of the session" ends here.
            if resident.is_some() {
                frontend.forget_session_answers();
            }
            resident = on_screen(session_id, &agent, &conversation, &cancel);
        }
    }

    schedule_watcher.abort();
    if let Some(sweep) = schedule_sweep {
        sweep.abort();
        // Waited for before the lease of a gate it was running is given back, so nothing the sweep
        // writes lands after that; the gate is stopped with it. Left behind, that lease would hold
        // its schedule until `claim_lease` runs out and count toward parking it.
        if let Err(error) = sweep.await
            && error.is_panic()
        {
            tracing::warn!("the schedule sweep panicked: {error}");
        }
        if let Err(error) = store.schedule_store().give_back_gates_cut_short().await {
            tracing::warn!("{error}");
        }
    }
    expiry_sweeper.abort();
    // The screen gives the terminal back before anything below prints to it, and hands the log
    // queue back to stderr as it goes.
    ui.send(UiEvent::Leave);
    drop(ui);
    // A screen that panicked has had its terminal given back by its hook; the teardown below is
    // still owed, so the panic is reported after it rather than skipping it.
    let screen_panic = match screen_handle.await {
        Ok(Ok(())) => None,
        Ok(Err(error)) => {
            tracing::warn!("the screen ended with an error: {error}");
            None
        }
        Err(error) => Some(error),
    };

    if let Some(id) = session_id
        && config.show_session_id_on_exit
    {
        crate::render::render_session_id("Leaving session", &id.to_string());
    }
    // Emptied after the "Leaving session" line so the lock is held until the very end; the OS
    // releases the underlying flock when the FD closes.
    hold_session_lock(&session_lock, None);

    // Stop this process's background tasks on the way out: a detached `shell_execute` is
    // `setsid()`-ed, so nothing else reaches it, and the next session open would sweep its row to
    // `interrupted` while it may still be writing to the workspace.
    let stopped = crate::host::release_agent(&agent, &cancel, mcp_manager.as_ref()).await;
    if stopped > 0 {
        crate::render::render_annotation(&format!(
            "Stopping {} background task{}",
            stopped,
            if stopped == 1 { "" } else { "s" }
        ));
        // Waited for, not just signaled: the runtime shuts down right after this, which drops
        // every task where it stands.
        if tokio::time::timeout(BACKGROUND_EXIT_GRACE, agent.background_tasks().wait_all())
            .await
            .is_err()
        {
            tracing::warn!(
                "background task(s) still running after {seconds}s; leaving them to the next \
                 session open",
                seconds = BACKGROUND_EXIT_GRACE.as_secs()
            );
        }
    }

    drop(agent);

    if let Some(manager) = mcp_manager {
        shutdown_mcp_manager(manager).await;
    }

    if let Some(error) = screen_panic {
        return Err(anyhow::anyhow!("the screen stopped: {error}"));
    }
    // After the teardown, so the ladder's last press leaves nothing behind that `/exit` would
    // not; `main` turns this into the silent exit status 130.
    if interrupted_exit {
        return Err(crate::error::MekaError::Interrupted.into());
    }
    Ok(())
}

/// The inbox class a configured send class names. Two enums for one set of words, because the
/// config is read below the store; this is where they meet, and the test below holds them to the
/// same spellings.
fn inbox_class(class: crate::config::SendClass) -> crate::store::inbox::InboxClass {
    match class {
        crate::config::SendClass::Steer => crate::store::inbox::InboxClass::Steer,
        crate::config::SendClass::Followup => crate::store::inbox::InboxClass::Followup,
        crate::config::SendClass::Interrupt => crate::store::inbox::InboxClass::Interrupt,
    }
}

/// What survived an interrupt, said on the screen: Ctrl+C stops the turn, not the detached work,
/// and the user may not have registered that anything was detached.
async fn report_background_survivors_to(agent: &Agent, ui: &Ui) {
    let running = agent.background_tasks().unstopped_count_all().await;
    if running > 0 {
        ui.note(format!(
            "{running} background task(s) still running; stop them with `/task cancel --all`"
        ));
    }
}

/// The host's one session, as the scheduler driver addresses it.
#[derive(Clone)]
struct SessionEntry(crate::host::ResidentSession);

impl std::ops::Deref for SessionEntry {
    type Target = crate::host::ResidentSession;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// The session the loop is on, from the loop's own parts, or `None` before a fresh session's first
/// turn has given it an id.
fn on_screen(
    session_id: Option<uuid::Uuid>,
    agent: &Arc<Agent>,
    conversation: &Arc<tokio::sync::Mutex<crate::conversation::Conversation>>,
    cancel: &crate::host::CancelCell,
) -> Option<crate::host::ResidentSession> {
    session_id.map(|id| {
        crate::host::ResidentSession::from_parts(
            id,
            Arc::clone(agent),
            Arc::clone(conversation),
            cancel.clone(),
        )
    })
}

/// What the host does around an outcome delivery or a turn on the inbox: it owns the session
/// outright, so it never defers, and it reports through the screen like a typed turn does.
struct TuiHooks {
    entry: SessionEntry,
    store: Store,
    background_enabled: bool,
    ui: Ui,
}

impl TuiHooks {
    fn new(
        entry: crate::host::ResidentSession,
        store: &Store,
        config: &ResolvedConfig,
        ui: &Ui,
    ) -> Self {
        Self {
            entry: SessionEntry(entry),
            store: store.clone(),
            background_enabled: config.background.enabled,
            ui: ui.clone(),
        }
    }

    /// Run turns on what waits in the inbox of the session on screen until nothing does: what the
    /// inbox driver does for `meka serve` and `meka acp`, done by the host loop here because it
    /// owns the conversation. Asked on a wake and after a turn the loop ran itself, so an item that
    /// came too late for that turn's last pause is answered at once rather than at the watcher's
    /// next tick. Nothing caps the screen's turns, so a drain never steps back for capacity.
    async fn drain_inbox(&self) {
        crate::host::scheduler::run_inbox_turns(self, self.entry.id).await;
    }
}

/// The host's one session as the sweep sees it: resident whenever it is the session on screen, at
/// the level its permission cell holds.
struct ScreenResidents {
    agent: SharedAgent,
}

/// The agent of the session on screen, which `/resume` replaces with the one it opens: the one
/// place the screen, the schedule sweep and the signal listeners read a session's cells from, so a
/// switch leaves none of them on the session it let go of.
#[derive(Clone)]
pub(crate) struct SharedAgent(Arc<std::sync::RwLock<Arc<Agent>>>);

impl SharedAgent {
    fn new(agent: Arc<Agent>) -> Self {
        Self(Arc::new(std::sync::RwLock::new(agent)))
    }

    /// The agent of the session on screen now.
    pub(crate) fn get(&self) -> Arc<Agent> {
        Arc::clone(&crate::sync::read(&self.0))
    }

    fn set(&self, agent: Arc<Agent>) {
        *crate::sync::write(&self.0) = agent;
    }
}

#[async_trait::async_trait]
impl crate::scheduler::ResidentPermissions for ScreenResidents {
    async fn live_permission_of(
        &self,
        session_id: uuid::Uuid,
    ) -> Option<crate::permission::Permission> {
        let agent = self.agent.get();
        (agent.session_id() == Some(session_id)).then(|| agent.cells().permission.get())
    }
}

#[async_trait::async_trait]
impl crate::scheduler::ResidentPermissions for TuiHooks {
    async fn live_permission_of(
        &self,
        session_id: uuid::Uuid,
    ) -> Option<crate::permission::Permission> {
        (self.entry.id == session_id).then(|| self.entry.agent.cells().permission.get())
    }
}

#[async_trait::async_trait]
impl crate::host::scheduler::HostHooks for TuiHooks {
    type Entry = SessionEntry;

    async fn resident(&self, session_id: uuid::Uuid) -> anyhow::Result<Option<Self::Entry>> {
        Ok((self.entry.id == session_id).then(|| self.entry.clone()))
    }

    fn cancellation(&self) -> crate::agent::TurnCancellation {
        // Published through the resident's cell by the driver, where Ctrl+C reaches it; the
        // escalation count starts over for the same reason a typed turn's does.
        crate::host::terminal::reset_interrupt_escalation();
        crate::agent::TurnCancellation::new()
    }

    async fn inbox_deferred(
        &self,
        _entry: &Self::Entry,
        items: &[crate::store::inbox::InboxItem],
        not_before: chrono::DateTime<chrono::Utc>,
        _error: &crate::error::MekaError,
    ) {
        self.ui
            .note(crate::host::scheduler::deferral_notice(items, not_before));
    }

    async fn inbox_given_up(
        &self,
        _entry: &Self::Entry,
        item: &crate::store::inbox::InboxItem,
        reason: &str,
        _error: Option<&crate::error::MekaError>,
    ) {
        self.ui.send(UiEvent::InboxWithdrawn {
            item_ids: vec![item.id],
        });
        self.ui.item(model::Item::Notice {
            level: crate::frontend::NoticeLevel::Warn,
            text: crate::host::scheduler::give_up_notice(item, reason),
        });
    }

    async fn inbox_withdrawn(&self, _entry: &Self::Entry, item_id: uuid::Uuid) {
        self.ui.send(UiEvent::InboxWithdrawn {
            item_ids: vec![item_id],
        });
    }

    async fn finished(
        &self,
        entry: &Self::Entry,
        _origin: &crate::host::scheduler::TurnStarter,
        outcome: &Result<(), crate::error::MekaError>,
    ) {
        match outcome {
            Ok(()) => {}
            Err(crate::error::MekaError::Interrupted) => {
                self.ui.item(model::Item::interrupted());
                report_background_survivors_to(&entry.agent, &self.ui).await;
            }
            Err(error) => self.ui.error(error),
        }
    }

    fn background_enabled(&self) -> bool {
        self.background_enabled
    }

    fn store(&self) -> &Store {
        &self.store
    }

    /// From the hangup, the termination, the terminal found gone or the SIGINT that ends the ladder
    /// on, while the exit is on its way. A drain or a delivery begun meanwhile would run a turn
    /// nothing stops, on what waits in the inbox or on the outcome of a task the stop ended.
    fn shutting_down(&self) -> bool {
        crate::host::terminal::INTERRUPT_RELAY.is_told_to_leave()
    }

    fn draws_its_log(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The config's spelling of each class is the inbox's, value for value.
    #[test]
    fn the_send_classes_spell_the_inbox_classes() {
        for class in crate::config::SendClass::ALL {
            assert_eq!(class.name(), inbox_class(class).name(), "{class:?}");
        }
        assert_eq!(
            crate::config::SendClass::ALL.len(),
            crate::store::inbox::InboxClass::ALL.len()
        );
    }
}
