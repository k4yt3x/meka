//! The interactive host: the loop that owns the conversation and answers the editor thread.

pub(crate) mod commands;
pub(crate) mod editor;
pub(crate) mod frontend;
pub(crate) mod history;
pub(crate) mod prompts;

use super::{terminal::*, *};

/// Run the interactive host: resume or create the session, spawn the editor thread, and answer its
/// events until it leaves.
pub(crate) async fn run_interactive(
    config: ResolvedConfig,
    store: Store,
    token_store: TokenStore,
    initial_prompt: Option<String>,
    mcp_manager: Option<Arc<crate::mcp::McpClientManager>>,
) -> anyhow::Result<()> {
    let config = Arc::new(config);
    // Where the shell was. Kept separate from the per-session `cwd` below rather than folded into
    // it, because the two part company the moment a resumed session opens somewhere else, and a
    // bare `/cd` returns here.
    let launch_cwd = std::env::current_dir().unwrap_or_else(|error| {
        tracing::warn!("failed to read the process working directory at startup: {error}");
        std::path::PathBuf::from(".")
    });

    // Everything printed between two prompts, wherever it came from. One instance, shared by the
    // agent's frontend, the blocking REPL thread and this loop, because the blank lines that
    // bracket an episode follow from what the episode did rather than from which of the three
    // happened to answer it.
    let console = Arc::new(std::sync::Mutex::new(crate::console::Console::new(
        crate::console::Spacing {
            newline_before_prompt: config.newline_before_prompt,
            newline_after_prompt: config.newline_after_prompt,
        },
        config.render_mode,
    )));
    // Before the first thing that logs. Off-prompt, `tracing` writes straight to stderr, and the
    // row it lands on may be one the console intends to erase; giving the relay this handle is what
    // lets a mid-turn `warn!` settle that row first instead of being wiped with it.
    crate::relay::RELAY.install_console(&console);
    let repl_console = Arc::clone(&console);

    // Before the REPL thread exists, so the resume banner lands above the first prompt, and before
    // the permission and cwd cells, which a resumed session brings with it.
    let ResumedSession {
        mut session_id,
        messages,
        lock: resumed_lock,
        permission: start_permission,
        approvals: start_approvals,
        repin,
        permission_to_record,
        cwd: recorded_cwd,
        follows,
    } = resolve_session_resume(&store, &config, &console).await?;
    // The first episode: the replayed history and any prompt queued on the command line belong to
    // it, and it ends at the first prompt, which is what gives them their closing bracket. Opened
    // after the resume because the banner is what it follows: printed, the banner stands in for the
    // line you typed and the blank below it answers to `newline_after_prompt`; hidden, the shell's
    // own command line is above and no opening blank is owed.
    with_console(&console, |console| {
        console.open_episode(crate::console::RowState::Empty, follows)
    });
    let _last_episode = LastEpisode(Arc::clone(&console));
    // After the resume, whose lock is what spares the session this run was asked for.
    sweep_expired_sessions(&config, &store).await?;

    // The same deps the long-lived hosts build, so all four hosts assemble a session identically.
    // `providers` is held past agent construction because `/profile` rebuilds one mid-session and
    // the cache is what makes the profile it left reusable when it comes back.
    let shared =
        match build_shared_deps(Arc::clone(&config), store.clone(), mcp_manager.clone()).await {
            Ok(shared) => Arc::new(shared),
            Err(error) => {
                with_console(&console, |console| console.error(&error));
                return Err(AlreadyReported.into());
            }
        };
    let providers = Arc::clone(&shared.providers);

    // Per-session working directory, shared by reference between the REPL (prompt + `/cd`) and the
    // agent (file/shell/find/grep tools + environment-context block). Process cwd is never mutated.
    // A resumed session opens where it recorded rather than where this shell is; see
    // `resume_working_directory`.
    let cwd = crate::workspace::SharedCwd::new(resume_working_directory(
        recorded_cwd,
        &launch_cwd,
        session_id,
    ));

    let shared_permission = SharedPermission::new(start_permission, config.enabled_permissions)
        .with_approvals(start_approvals);
    if start_permission == crate::permission::Permission::Read {
        crate::sandbox::warn_if_sandbox_issues(
            &crate::sandbox::SandboxState::new(config.sandbox, &shared.sandbox),
            crate::sandbox::WarnContext::InitialReadLevel,
        );
    }

    if !messages.is_empty() {
        // The replay announces itself at its first row and the console decides the blank above it,
        // as for the first output of any episode. A replay that renders nothing (a tail of tool
        // calls with no text) announces nothing, so no blank is spent on an empty region.
        let announce = || with_console(&console, |console| console.announce_foreign_output());
        match config.resume_show_recent {
            Some(n) if n > 0 => {
                crate::render::render_message_history(
                    crate::render::last_n_turns(messages.as_slice(), n),
                    &history_render_options(&config),
                    announce,
                );
            }
            _ => reprint_last_message(messages.as_slice(), config.render_mode, announce),
        }
    }

    let (input_sender, mut input_receiver) = tokio::sync::mpsc::unbounded_channel::<ReplEvent>();

    // A prompt or skill given without `--oneshot` is queued as the first turn's input. The flag
    // tells the REPL to wait for that turn's events before drawing its first prompt; otherwise
    // reedline's prompt collides with the agent's output.
    let initial_turn_pending = initial_prompt.is_some();
    if let Some(prompt) = initial_prompt {
        #[allow(
            clippy::expect_used,
            reason = "the receiver is `input_receiver` below, still owned here, so `send` cannot fail"
        )]
        input_sender
            .send(ReplEvent::UserInput(prompt))
            .expect("freshly created input channel must accept first send");
    }
    let (agent_event_sender, agent_event_receiver) =
        std::sync::mpsc::channel::<crate::host::repl::editor::AgentToReplEvent>();
    // The REPL frontend forwards approval requests to the same channel the REPL thread already
    // reads from for `Done` / MCP elicitation / MCP progress events.
    let repl_frontend = Arc::new(crate::host::repl::frontend::ReplFrontend::new(
        crate::host::repl::frontend::ReplFrontendConfig {
            console: Arc::clone(&console),
            show_session_id_on_create: config.show_session_id_on_create,
            show_token_usage: config.show_token_usage,
            thinking_show_content: config.thinking_show_content,
            tool_params: config.tool_params,
            agent_event_sender: agent_event_sender.clone(),
        },
    ));

    let repl_permission = shared_permission.clone();
    let show_path_in_prompt = config.show_path_in_prompt;
    let input_style = config.input_style;
    let repl_sandbox_state = crate::sandbox::SandboxState::new(config.sandbox, &shared.sandbox);
    let repl_cwd = cwd.clone();
    let repl_mcp_server_names: Vec<String> = config
        .mcp_servers
        .iter()
        .map(|server| server.name.clone())
        .collect();
    let repl_skill_roots = config.skill_roots();
    let repl_history_db_path = Some(store.database_path().to_path_buf());

    // The prompt's context gauge: the agent writes it after each turn and the prompt reads it each
    // render. Created before the agent so the REPL thread can hold it. The window is seeded from
    // the process default and corrected to the session's own as soon as the agent resolves it: the
    // session may be pinned to another profile, and `/profile` may move it again, and a prompt
    // dividing by a window the agent is not gauging against contradicts `/status`.
    let context_window_gauge = Arc::new(std::sync::atomic::AtomicU64::new(
        config
            .session_context_window
            .unwrap_or(crate::provider::DEFAULT_CONTEXT_WINDOW),
    ));
    let context_tokens = Arc::new(std::sync::atomic::AtomicU64::new(0));
    // Seeded with an estimate on a resume, so the gauge is not blank until the first turn measures.
    if !messages.is_empty() {
        context_tokens.store(
            crate::tokens::estimate_messages(messages.as_slice()),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    let context_indicator = config.show_context_in_prompt.then(|| {
        (
            Arc::clone(&context_tokens),
            Arc::clone(&context_window_gauge),
        )
    });

    // Shared with reedline: the scheduler watcher sets it, `read_line` polls it and returns
    // `Signal::ExternalBreak` so a due job can interrupt an idle prompt.
    let schedule_wake = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let repl_wake = Arc::clone(&schedule_wake);

    // What `/profile` reports and rewrites. Seeded through the same door the agent resolves with,
    // so a resumed session shows the profile it recorded rather than the configured default. A
    // session that cannot resolve one never reaches a prompt at all, so the default keeps the seed
    // total.
    let current_profile = Arc::new(std::sync::RwLock::new(match &repin {
        // The repin has not been committed yet (it waits for the registry, below), but it is what
        // the row will say by the time anything reads this cell.
        Some(profile) => profile.clone(),
        None => crate::provider::profile_for_config(&store, &config, session_id)
            .await
            .unwrap_or_default(),
    }));
    let repl_current_profile = Arc::clone(&current_profile);
    // Name, account and backend, so `/profile` can say what each profile *is* rather than only
    // what it is called. `profiles` is a `BTreeMap`, so this is already in name order.
    let repl_configured_profiles: Vec<crate::config::ProfileSummary> =
        config.profile_summaries.clone();

    // Before the agent, because the agent resolves the row: a repin that has not landed yet would
    // build this run on the profile the session is leaving.
    if let Err(error) = commit_resume_repin(&store, &providers, session_id, repin).await {
        with_console(&console, |console| console.error(&error));
        return Err(AlreadyReported.into());
    }
    let agent = match build_session_agent(&shared, SessionSpec {
        session_id,
        permission: shared_permission,
        frontend: Arc::clone(&repl_frontend) as Arc<dyn crate::frontend::Frontend>,
        cwd: cwd.clone(),
        // ACP is the other source of extra workspace roots; here they come from
        // `--writable-root`. Both land in the same handle because they mean the same thing, so a
        // named folder is searched and, at `workspace` permission, writable.
        roots: crate::workspace::SharedRoots::new(config.request.writable_roots.clone()),
        context_tokens: Arc::clone(&context_tokens),
        // The REPL reads occupancy through `/status`, which goes via the agent it owns, so only the
        // token gauge is the host's; the overhead counter can be a fresh one.
        context_overhead: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        // The prompt's own gauge, handed in rather than seeded and corrected: it *is* the cell the
        // agent publishes into, so `/profile` cannot move one without the other.
        context_window: Arc::clone(&context_window_gauge),
    })
    .await
    {
        Ok(agent) => agent,
        Err(error) => {
            with_console(&console, |console| console.error(&error));
            // No `SessionNotDrivable` arm here. `session_id` reaches this builder only from
            // `resolve_session_resume`, which refuses a sub-agent before returning one, so the
            // builder's own refusal is unreachable from this call site and an arm suppressing the
            // provider advice below could never fire. The refusal the user actually sees is raised
            // there, several steps before any of this runs.
            let gone = match session_id {
                Some(id) => crate::provider::recorded_profile_is_gone(&store, &config, id).await,
                None => false,
            };
            // The default when there is one, so the suggested move is the profile the rest of this
            // config already runs on. With no profile at all there is nowhere to move to.
            let move_to = config
                .default_profile
                .as_deref()
                .or_else(|| config.profiles.keys().next().map(String::as_str));
            crate::render::render_profile_setup_hint(session_id.filter(|_| gone).zip(move_to).map(
                |(session_id, move_to)| crate::render::MissingSessionProfile {
                    session_id,
                    move_to,
                },
            ));
            return Err(AlreadyReported.into());
        }
    };
    // After the agent, so a start the builder refused leaves the row at the level it had.
    record_resume_permission(&store, session_id, permission_to_record).await;
    // Installed by the host, with the console: the escalation arms print, and a bare `eprintln!`
    // from a spawned task lands wherever the cursor happens to be, which on a second Ctrl+C during
    // a turn is the middle of the thinking indicator's row.
    let agent = Arc::new(agent);
    let conversation = Arc::new(tokio::sync::Mutex::new(messages));
    let cancel = crate::host::CancelCell::default();
    install_interrupt_handler(cancel.clone(), &agent, Arc::clone(&console));
    // The session as the shared host machinery sees it: made once a session exists, so the
    // scheduler driver can find it, and remade when `/fork` moves the loop to a copy.
    let mut resident = session_id.map(|id| {
        crate::host::ResidentSession::from_parts(
            id,
            Arc::clone(&agent),
            Arc::clone(&conversation),
            cancel.clone(),
        )
    });

    // Spawned once there is an agent to answer it, which is what makes every refusal above final:
    // started earlier, the prompt outlives a failed construction and the user types into a shell
    // that answers nothing.
    let repl_handle = tokio::task::spawn_blocking(move || {
        crate::host::repl::editor::run_repl(crate::host::repl::editor::ReplLaunch {
            shared_permission: repl_permission,
            show_path_in_prompt,
            context_indicator,
            input_style,
            initial_turn_pending,
            sandbox_state: repl_sandbox_state,
            input_sender,
            agent_event_receiver,
            cwd: repl_cwd,
            launch_cwd,
            mcp_server_names: repl_mcp_server_names,
            skill_roots: repl_skill_roots,
            history_db_path: repl_history_db_path,
            wake: repl_wake,
            current_profile: repl_current_profile,
            configured_profiles: repl_configured_profiles,
            console: repl_console,
        });
    });

    // One slot for the session lock from here on, whichever way the session was reached: the agent
    // fills it the moment it creates one, and a resumed session's lock -- taken above, before the
    // REPL thread existed -- moves into the same place. `/fork` replaces what is in it and the exit
    // path empties it, so neither has to know which of the two put it there.
    let session_lock = agent.session_lock_slot();
    hold_session_lock(&session_lock, resumed_lock);

    // Mirrors the loop's `session_id` for the watcher below, which runs on another task and cannot
    // borrow it. Written after every event, which is the only thing that can change it.
    let repl_shared_session_id = Arc::new(std::sync::RwLock::new(session_id));

    // Watcher, not a scheduler: it only nudges reedline awake. The agent loop below owns the
    // conversation, so it has to be the thing that evaluates gates and runs the turn -- otherwise
    // two tasks would be appending to `messages`. Background outcomes ride the same watcher and the
    // same flag for the same reason.
    let schedule_watcher = {
        let store = store.clone();
        let shared_session_id = Arc::clone(&repl_shared_session_id);
        let poll_interval = config.schedule.poll_interval;
        let schedule_enabled = config.schedule.enabled;
        let background_enabled = config.background.enabled;
        // The watcher asks whether a wake would produce work, and part of that answer is the
        // session's live permission resolved against this installation's enabled set.
        let watcher_schedule_config = config.schedule.clone();
        let watcher_residents = ReplResidents {
            session_id: Arc::clone(&repl_shared_session_id),
            permission: agent.cells().permission.clone(),
        };
        tokio::spawn(async move {
            if !schedule_enabled && !background_enabled {
                return;
            }
            let mut ticker = tokio::time::interval(poll_interval);
            ticker.tick().await;
            loop {
                ticker.tick().await;
                let Some(current) = *crate::sync::read(&shared_session_id) else {
                    continue;
                };
                if schedule_enabled {
                    // The same question the fire door asks, as far as it can be answered without
                    // running a gate probe. Asking the weaker "is a row due" here is what let a
                    // parked job interrupt the prompt every `poll_interval` to run nothing.
                    match crate::scheduler::wake_would_produce_work(
                        &store,
                        &watcher_schedule_config,
                        &watcher_residents,
                        current,
                        chrono::Utc::now(),
                    )
                    .await
                    {
                        Ok(true) => {
                            schedule_wake.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        Ok(false) => {}
                        Err(error) => tracing::warn!("failed to poll the schedule: {error}"),
                    }
                }
                if background_enabled {
                    match store
                        .background_store()
                        .list_undelivered_background_tasks(current)
                        .await
                    {
                        // Only for an outcome that warrants a turn, mirroring
                        // `wake_would_produce_work` above: waking for one that will wait tears the
                        // prompt down and redraws it to do nothing.
                        Ok(ready) if ready.iter().any(|task| task.status.wakes_a_host()) => {
                            schedule_wake.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        Ok(_) => {}
                        Err(error) => tracing::warn!("failed to poll background tasks: {error}"),
                    }
                }
            }
        })
    };

    while let Some(event) = input_receiver.recv().await {
        match event {
            // Recorded on the row, then straight back to the prompt: this is not a turn and must
            // not look like one, so no `AgentToReplEvent::Done` and no spacing.
            //
            // The row is what a *scheduled gate* is re-checked against at fire time, and a row that
            // carries no level falls back to the polling process's own startup flag, so a `meka
            // serve` sharing the data directory would keep running a gate the user has just
            // withdrawn with Shift+Tab.
            ReplEvent::PermissionChanged(level) => {
                let Some(id) = session_id else {
                    // No row yet: the level the first turn creates the session with is this one, so
                    // there is nothing to correct.
                    continue;
                };
                // The user's own level has already moved in this process; what a failed write
                // costs is `record_session_change`'s to decide.
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
            // Recorded like the level, and for the same reader: the row is what a resume brings
            // back.
            ReplEvent::ApprovalsChanged(approvals) => {
                let Some(id) = session_id else {
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
            // Recorded for the same reasons the level above is, and it lands here rather than in
            // the REPL thread because that thread is not async and holds no `Store`. The
            // row is where the *next* resume opens the session, and it is the directory a scheduled
            // tool-gate is re-checked in; a `/cd` that stopped at the in-memory cell left both
            // answering with the directory the session was created in.
            ReplEvent::CwdChanged(path) => {
                record_session_cwd(&store, session_id, &path).await;
            }
            // The row moves first. If recording the change failed but the agent had already
            // switched, the next `meka -c` would silently go back to the old profile, which is the
            // surprise this whole feature exists to remove.
            ReplEvent::ProfileChange(name) => {
                // A labeled block rather than `continue`, so every way out passes the `Done`
                // below; otherwise the REPL thread paints the next prompt while this task is still
                // deciding what to print, and the answer lands on the line being typed.
                'switch: {
                    // The profile as configured. `/profile` moves the session to that bundle
                    // entire, which is the only thing naming a profile can mean.
                    let resolved = match resolve_profile_switch(&shared, &name).await {
                        Ok(resolved) => resolved,
                        Err(error) => {
                            with_console(&console, |console| console.error(&error));
                            break 'switch;
                        }
                    };
                    // The row moves first; a row that cannot be written leaves the agent where it
                    // was, so the next `meka -c` cannot silently go back to the old profile.
                    if let Some(id) = session_id
                        && let Err(error) = record_profile_switch(&store, id, &resolved).await
                    {
                        with_console(&console, |console| console.error(&error));
                        break 'switch;
                    }
                    // The prompt gauge is the cell the agent publishes into, so this moves it too.
                    agent.set_provider(resolved);
                    *crate::sync::write(&current_profile) = name.clone();
                    with_console(&console, |console| {
                        console.line(&format!("Profile set to: {name}"))
                    });
                }
                if agent_event_sender
                    .send(crate::host::repl::editor::AgentToReplEvent::Done)
                    .is_err()
                {
                    break;
                }
            }
            ReplEvent::Wake => {
                // Nothing can be due before the session exists; the watcher would not have woken
                // us, but the loop must not assume that.
                if let Some(entry) = resident.clone() {
                    let hooks = ReplHooks {
                        entry: SessionEntry(entry.clone()),
                        store: store.clone(),
                        background_enabled: config.background.enabled,
                        console: Arc::clone(&console),
                    };
                    // Gated on the switch, not just on having been woken. `run_due` has no
                    // `enabled` check of its own -- the flag is enforced by whoever decides to poll
                    // -- and this arm is also reached by a finished background task setting the
                    // same wake flag. Without this, turning scheduling off while background calls
                    // are on would still fire the jobs already in the database.
                    if config.schedule.enabled
                        && let Err(error) = crate::scheduler::run_due(
                            &store,
                            &config.schedule,
                            shared.gate_tools.as_deref(),
                            &hooks,
                            &crate::scheduler::SchedulerScope::OneSession(entry.id),
                            &|wakeup: crate::scheduler::Wakeup| {
                                crate::host::scheduler::run_wakeup(&hooks, wakeup)
                            },
                        )
                        .await
                    {
                        with_console(&console, |console| console.error(&error));
                    }
                    // Outcomes that warrant a turn of their own, delivered the way every host
                    // delivers them; the quiet ones ride the next turn, fired or typed.
                    if config.background.enabled {
                        let sessions = crate::host::Sessions::<uuid::Uuid, SessionEntry>::new();
                        sessions
                            .write()
                            .await
                            .insert(entry.id, SessionEntry(entry.clone()));
                        if crate::host::scheduler::deliver_ready_outcomes(&hooks, &sessions)
                            .await
                            .is_break()
                        {
                            break;
                        }
                    }
                }
                if agent_event_sender
                    .send(crate::host::repl::editor::AgentToReplEvent::Done)
                    .is_err()
                {
                    break;
                }
            }
            ReplEvent::UserInput(input) => {
                // Admitted here, before anything below can wait. The outcome claim waits for the
                // MCP servers to settle, and a Ctrl+C in that window must count: sampling the epoch
                // at publish, after the press has bumped it, runs the turn anyway and sends a
                // second press straight to the escalation ladder, canceling every
                // background task.
                let admission = cancel.admit();
                // Admitted before any outcome is claimed to ride on it. A claim is one-way, so a
                // prompt refused after it would leave the batch stamped delivered and never
                // handed out again.
                let input = match crate::agent::TurnInput::from_parts(input, Vec::new()) {
                    Ok(input) => input,
                    Err(empty) => {
                        with_console(&console, |console| console.error(&empty));
                        if agent_event_sender
                            .send(crate::host::repl::editor::AgentToReplEvent::Done)
                            .is_err()
                        {
                            break;
                        }
                        continue;
                    }
                };
                // An outcome that did not warrant a turn of its own rides on this one. Joined to
                // the prompt rather than appended as its own message: see `render_outcomes_riding`.
                let outcomes = if config.background.enabled {
                    collect_background_outcomes(&agent, &store, session_id).await
                } else {
                    Vec::new()
                };
                let input = input.riding(outcomes);
                let mut messages = conversation.lock().await;
                match run_turn_interruptible(
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
                        with_console(&console, |console| console.annotation("interrupted"));
                        report_background_survivors(&agent).await;
                    }
                    Err(error) => {
                        with_console(&console, |console| console.error(&error));
                    }
                }

                if agent_event_sender
                    .send(crate::host::repl::editor::AgentToReplEvent::Done)
                    .is_err()
                {
                    break;
                }
            }
            ReplEvent::Command(command) => {
                // Exhaustive over `SlashCommand`: a match ending in `_ => {}` is how a forwarded
                // command arrives, matches nothing, and still gets its episode brackets.
                let after = {
                    let mut messages = conversation.lock().await;
                    crate::host::repl::commands::answer(
                        command,
                        crate::host::repl::commands::HostCommandContext {
                            cancel: &cancel,
                            agent: &agent,
                            agent_event_sender: &agent_event_sender,
                            config: &config,
                            console: &console,
                            mcp_manager: &mcp_manager,
                            messages: &mut messages,
                            providers: &providers,
                            session_id: &mut session_id,
                            session_lock: &session_lock,
                            store: &store,
                            token_store: &token_store,
                        },
                    )
                    .await
                };
                if matches!(after, crate::host::repl::commands::AfterCommand::Leave) {
                    break;
                }
            }
            ReplEvent::Exit => {
                break;
            }
        }
        // The loop's own `session_id` is authoritative; the watcher reads this mirror. An event is
        // the only thing that can create or replace a session, so syncing here is sufficient.
        *crate::sync::write(&repl_shared_session_id) = session_id;
        if resident.as_ref().map(|entry| entry.id) != session_id {
            // The loop has moved from one session to another (`/fork` today), so what was answered
            // "for the rest of the session" ends here. The first session's answers are the
            // frontend's own business: it clears on `SessionStarted`, which a turn emits before
            // any call it could be asked about, so this is asked only of a move *between*
            // sessions and cannot take an answer given during that first turn.
            if resident.is_some() {
                repl_frontend.forget_session_answers();
            }
            resident = session_id.map(|id| {
                crate::host::ResidentSession::from_parts(
                    id,
                    Arc::clone(&agent),
                    Arc::clone(&conversation),
                    cancel.clone(),
                )
            });
        }
    }

    schedule_watcher.abort();
    drop(agent_event_sender);
    repl_handle.await?;

    if let Some(id) = session_id
        && config.show_session_id_on_exit
    {
        with_console(&console, |console| {
            console.session_id("Leaving session", &id.to_string())
        });
    }
    // The last episode has no prompt after it, but closing it is still what settles the row and
    // flushes anything a turn left open. Not necessarily the end of output: the background-task
    // notice below can still follow, and prints flush against this line, as two `Chrome` blocks
    // always do.
    close_console_episode(&console);
    // Emptied after the "Leaving session" message so the lock is held until the very end; the OS
    // releases the underlying flock when the FD closes. Emptied rather than dropped: the slot is
    // shared with the agent, which is still alive here, so letting this handle fall out of scope
    // would release nothing.
    hold_session_lock(&session_lock, None);

    // Stop this process's background tasks on the way out. `BackgroundTasks` has no `Drop`, and a
    // detached `execute_command` is `setsid()`-ed, so nothing else reaches it: left alone it runs
    // on untracked, and the next session open sweeps its row to `interrupted` while it may still be
    // writing to the workspace.
    let stopped = crate::host::release_agent(&agent, &cancel, mcp_manager.as_ref()).await;
    if stopped > 0 {
        with_console(&console, |console| {
            console.annotation(&format!(
                "stopping {} background task{}",
                stopped,
                if stopped == 1 { "" } else { "s" }
            ))
        });
        // The shutdown notice is the last thing the terminal sees, so it gets the same closing
        // treatment as everything else. Closing twice is free.
        close_console_episode(&console);
        // Waited for, not just signaled. `run_on_runtime` returns into `shutdown_background`
        // immediately after this, which drops every task where it stands: a task parked at an
        // await is never polled again, so it runs neither `kill_child_tree` nor
        // `finish_background_task` and the cancel achieves exactly nothing. Bounded, because
        // canceling only asks -- a task that does not answer must not hold the terminal, and its
        // row is swept to `interrupted` on the next open, which is what that sweep is for.
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

    Ok(())
}

/// The REPL's one session, as the scheduler driver addresses it.
#[derive(Clone)]
struct SessionEntry(crate::host::ResidentSession);

impl std::ops::Deref for SessionEntry {
    type Target = crate::host::ResidentSession;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// What the REPL does around a scheduled fire or an outcome delivery: it owns the session outright,
/// so it never defers, and it reports through the console like a typed turn does.
struct ReplHooks {
    entry: SessionEntry,
    store: Store,
    background_enabled: bool,
    console: Arc<std::sync::Mutex<crate::console::Console>>,
}

/// The REPL's one session as the watcher task sees it: resident whenever it is the loop's current
/// session, at the level the prompt's cell holds.
struct ReplResidents {
    session_id: Arc<std::sync::RwLock<Option<uuid::Uuid>>>,
    permission: SharedPermission,
}

#[async_trait::async_trait]
impl crate::scheduler::ResidentPermissions for ReplResidents {
    async fn live_permission_of(
        &self,
        session_id: uuid::Uuid,
    ) -> Option<crate::permission::Permission> {
        (*crate::sync::read(&self.session_id) == Some(session_id)).then(|| self.permission.get())
    }
}

#[async_trait::async_trait]
impl crate::scheduler::ResidentPermissions for ReplHooks {
    async fn live_permission_of(
        &self,
        session_id: uuid::Uuid,
    ) -> Option<crate::permission::Permission> {
        (self.entry.id == session_id).then(|| self.entry.agent.cells().permission.get())
    }
}

#[async_trait::async_trait]
impl crate::host::scheduler::HostHooks for ReplHooks {
    type Entry = SessionEntry;

    async fn resident(&self, session_id: uuid::Uuid) -> anyhow::Result<Option<Self::Entry>> {
        Ok((self.entry.id == session_id).then(|| self.entry.clone()))
    }

    fn cancellation(&self) -> tokio_util::sync::CancellationToken {
        // Published through the resident's cell by the driver, where Ctrl+C reaches it; the
        // escalation count starts over for the same reason a typed turn's does.
        crate::host::terminal::reset_interrupt_escalation();
        tokio_util::sync::CancellationToken::new()
    }

    /// Dim, the way a notice is: the reply that follows would otherwise appear under nothing, as
    /// though the model had spoken unprompted.
    fn show_prompt(
        &self,
        _entry: &Self::Entry,
        prompt: crate::host::scheduler::OutOfBandPrompt<'_>,
    ) {
        let text = match prompt {
            crate::host::scheduler::OutOfBandPrompt::Outcomes(text) => text.to_string(),
            crate::host::scheduler::OutOfBandPrompt::Scheduled(wakeup) => wakeup.render_prompt(),
        };
        with_console(&self.console, |console| {
            console.notice(&crate::frontend::Notice::info(text))
        });
    }

    async fn finished(
        &self,
        entry: &Self::Entry,
        _job: Option<&crate::schedule::ScheduledJob>,
        outcome: &Result<(), crate::error::MekaError>,
    ) {
        match outcome {
            Ok(()) => {}
            Err(crate::error::MekaError::Interrupted) => {
                with_console(&self.console, |console| console.annotation("interrupted"));
                report_background_survivors(&entry.agent).await;
            }
            Err(error) => with_console(&self.console, |console| console.error(error)),
        }
    }

    fn background_enabled(&self) -> bool {
        self.background_enabled
    }

    fn store(&self) -> &Store {
        &self.store
    }
}
