//! Answering the slash commands the REPL thread cannot answer itself.
//!
//! `SlashCommand` is parsed on the REPL thread ([`crate::host::repl`]), and about half of its
//! variants need something only the host loop has: the live `Agent`, the conversation, the session
//! id that `/fork` moves. Those are forwarded and answered here.
//!
//! **One owner, checked by the compiler.** A hand-written forwarding list and a `match` ending in
//! `_ => {}` have to agree with nothing making them: a variant added to one and forgotten in the
//! other is sent, silently does nothing, and still gets its episode brackets around no output.
//! [`SlashCommand::answered_by`] and [`answer`] are both exhaustive, so a new variant fails to
//! compile in both places.

use std::sync::Arc;

use crate::{
    agent, cli, conversation, error,
    host::{repl::editor as repl, terminal::with_console},
    mcp, provider, render, skills,
    store::Store,
};

/// Say how a turn, or a `/compact`, ended when it did not succeed.
///
/// An interrupt is the user's own act, so it is annotated the way every turn-running host annotates
/// one rather than reported as an error. One place for the three commands here that run provider
/// calls, so `/compact` cannot drift from `/skill` and `/mcp <server>:<prompt>` again.
fn report_failure(console: &std::sync::Mutex<crate::console::Console>, error: &error::MekaError) {
    with_console(console, |console| match error {
        error::MekaError::Interrupted => console.annotation("interrupted"),
        error => console.error(error),
    });
}

/// What the host loop does once a command has been answered.
///
/// A return value rather than a `break` inside the arm, because the dispatcher sits outside the
/// loop. Only two of the arms need it: `/fork` and the rest leave the loop running, while a REPL
/// thread that has gone away ends it.
pub(crate) enum AfterCommand {
    Continue,
    Leave,
}

/// Answer one forwarded command.
///
/// Exhaustive over [`SlashCommand`] on purpose; see the module docs. The variants the REPL thread
/// answers itself are listed rather than swept into a wildcard, so that adding one is a decision
/// taken twice instead of a silent no-op.
pub(crate) async fn answer(command: SlashCommand, context: HostCommandContext<'_>) -> AfterCommand {
    // Destructured back into the names the arms already use, rather than `context.` at every site.
    // `agent` shadows nothing: a module path lives in the type namespace and this binding in the
    // value namespace, so `agent::PromptRetention` still resolves alongside it.
    let HostCommandContext {
        agent,
        cancel,
        messages,
        agent_event_sender,
        config,
        console,
        mcp_manager,
        providers,
        session_id,
        session_lock,
        store,
        token_store,
    } = context;
    let session_id_cell = session_id;
    let mut session_id = *session_id_cell;
    // Every command answered here says something, even if only that a list is empty, and much of it
    // prints through the `cli` modules the console cannot see. One announcement covers all of them.
    //
    // There is deliberately no "does this one answer by running a turn" exception any more.
    // Announcing is idempotent within an episode -- the opening blank is spent once, by whichever
    // writer gets there first -- so a command that runs a turn is spaced identically whether the
    // turn happens or it bails first. A predicate making that distinction leaves `/skill
    // nosuchskill` printing its error flush against both the line above and the prompt below.
    //
    // "Answered here" is the qualification, and it is why this is gated rather than unconditional:
    // the arm below for the six the REPL owns prints nothing at all. In debug it trips an
    // assertion, but in release it is a silent no-op, and announcing first would give it exactly
    // the blank-line sandwich this call's own doc warns against -- in the builds where nothing is
    // watching.
    if command.answered_by() == Answerer::Host {
        with_console(console, |console| console.announce_foreign_output());
    }
    match command {
        SlashCommand::Session => match &session_id {
            Some(id) => with_console(console, |console| {
                console.session_id("Current session", &id.to_string())
            }),
            None => crate::streams::write_stderr_line("No active session yet."),
        },
        SlashCommand::Compact(instructions) => {
            let request = crate::session::CompactRequest {
                origin: crate::session::CompactOrigin::Manual,
                instructions: instructions
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string),
                keep_recent: None,
                prompt_id: None,
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
                Ok(outcome) => {
                    with_console(console, |console| {
                        console.hint(&render::compaction_summary(&outcome))
                    });
                }
                Err(error) => report_failure(console, &error),
            }
        }
        SlashCommand::RewindInvalid(argument) => {
            with_console(console, |console| {
                console.error(&format!(
                    "/rewind takes a turn count of 1 or more, not '{argument}'"
                ))
            });
        }
        SlashCommand::Rewind(turns) => {
            let turns = turns.unwrap_or(1);
            // `rewind(0)` returns `None` unconditionally, so without this the `None`
            // arm below would report "fewer than 0 turn(s)". The count is what's
            // wrong, not the conversation.
            let rewound = if turns == 0 {
                None
            } else {
                messages.rewind(turns)
            };
            match (session_id, rewound) {
                (Some(id), Some(event)) => {
                    if let Err(error) = store.save_event(id, &event).await {
                        // Put the turns back rather than leave memory and disk
                        // disagreeing, which would resurrect them on the next resume
                        // and make the rewind look like it silently un-did itself.
                        messages.pop_repair();
                        with_console(console, |console| console.error(&error));
                    } else {
                        agent.reset_conversation_markers().await;
                        with_console(console, |console| {
                            console.hint(&format!("Rewound {turns} turn(s)."))
                        });
                    }
                }
                // No session means nothing was ever persisted, so the in-memory rewind
                // (which did happen) is the whole story.
                (None, Some(_)) => {
                    agent.reset_conversation_markers().await;
                    with_console(console, |console| {
                        console.hint(&format!("Rewound {turns} turn(s)."))
                    });
                }
                (_, None) if turns == 0 => {
                    crate::streams::write_stderr_line(
                        "Nothing to rewind: /rewind takes a turn count of 1 or more.",
                    );
                }
                (_, None) => {
                    crate::streams::write_stderr_line(format!(
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
                    // The name is generated from the session id and the file lands in
                    // the working directory, so a REPL user who is not told it has
                    // nowhere to look. `meka session export` stays quiet: there the
                    // shell is the one that knows.
                    Ok(Some(path)) => {
                        // Shown absolute: `/cd` moves the session's directory while
                        // the export lands in the process's, so a bare filename would
                        // point at the wrong one.
                        let shown = std::env::current_dir()
                            .map(|dir| dir.join(&path))
                            .unwrap_or(path);
                        crate::streams::write_stderr_line(format!(
                            "Exported session to {}",
                            shown.display()
                        ));
                    }
                    Ok(None) => {}
                    Err(error) => with_console(console, |console| console.error(&error)),
                }
            }
            None => crate::streams::write_stderr_line("No active session to export."),
        },
        SlashCommand::Fork => match session_id {
            Some(id) => match crate::host::fork_and_lock(store, id).await {
                Ok(crate::host::ForkHandoff::Switched { id, lock }) => {
                    // Replacing the slot's contents drops the original guard only now
                    // that the new one is held; see
                    // `crate::host::fork_and_lock`.
                    crate::host::terminal::hold_session_lock(session_lock, Some(lock));
                    session_id = Some(id);
                    // The agent's cell is what every tool reads, so the copy has to move it too,
                    // not wait for the next turn to notice.
                    agent.cells().session_id.set(id);
                    // `messages` is deliberately untouched, so the branch happens at
                    // the current head and the next turn continues in the copy.
                    with_console(console, |console| {
                        console.session_id("Forked session", &id.to_string())
                    });
                }
                Ok(crate::host::ForkHandoff::LockFailed { id, error }) => {
                    with_console(console, |console| console.error(&error));
                    with_console(console, |console| {
                        console.hint(&format!("Staying in the original. The copy exists: {id}"))
                    });
                }
                Ok(crate::host::ForkHandoff::SourceGone) => {
                    with_console(console, |console| {
                        console.error(&crate::error::MekaError::SessionNotFound(id))
                    });
                }
                Err(error) => {
                    crate::streams::write_stderr_line(format!("Failed to fork session: {error}"))
                }
            },
            None => crate::streams::write_stderr_line("No active session to fork."),
        },
        SlashCommand::McpList => {
            if let Err(error) = crate::cli::mcp::run_list(
                &config.mcp_servers,
                mcp_manager.as_ref(),
                &store.token_store(),
            )
            .await
            {
                with_console(console, |console| console.error(&error));
            }
        }
        // These three report success at `info!` and print nothing, which is right for
        // the `meka mcp …` CLI (the exit code carries it) and wrong here: a REPL
        // command has no exit code, so silence is indistinguishable from the command
        // never having run, and it leaves the `[display]` blank lines wrapped around
        // an empty region. `/permission` sets the precedent for confirming a state
        // change the user asked for.
        SlashCommand::McpReconnect { server } => {
            match crate::cli::mcp::run_reconnect(&config.mcp_servers, token_store, &server).await {
                // "Connected", not "Reconnected": this is a smoke test on a throwaway
                // client, and the session's own connection to that server is untouched.
                Ok(()) => crate::streams::write_stderr_line(format!("Connected to '{server}'.")),
                Err(error) => with_console(console, |console| console.error(&error)),
            }
        }
        SlashCommand::McpLogin { server } => {
            match crate::cli::mcp::run_login(&config.mcp_servers, token_store, &server).await {
                Ok(()) => crate::streams::write_stderr_line(format!("Authorized '{server}'.")),
                Err(error) => with_console(console, |console| console.error(&error)),
            }
        }
        SlashCommand::McpLogout { server } => {
            match crate::cli::mcp::run_logout(&config.mcp_servers, token_store, &server).await {
                Ok(()) => crate::streams::write_stderr_line(format!(
                    "Cleared credentials for '{server}'."
                )),
                Err(error) => with_console(console, |console| console.error(&error)),
            }
        }
        SlashCommand::McpMissingServer { verb } => {
            with_console(console, |console| {
                console.error(&format!("/mcp {verb} takes a server name"))
            });
        }
        SlashCommand::McpUnknownVerb { verb } => {
            with_console(console, |console| {
                console.error(&format!(
                    "'{verb}' is not an `/mcp` verb: `/mcp` takes {} or <server>:<prompt>",
                    repl::MCP_SUBCOMMANDS.join(", ")
                ))
            });
        }
        SlashCommand::McpPrompt {
            server,
            prompt: prompt_name,
            args,
        } => 'prompt: {
            let Some(manager) = mcp_manager.as_ref() else {
                crate::streams::write_stderr_line("No MCP servers.");
                break 'prompt;
            };
            let entry = manager.server_entry(&server);
            let Some(entry) = entry else {
                // Labeled break, not `continue`: `continue` targets the agent
                // loop, skipping the `AgentToReplEvent::Done` send below and
                // leaving the REPL thread parked in `wait_for_agent` with no
                // prompt, for good. Same reason as `SkillInvoke`'s `'invoke`.
                crate::streams::write_stderr_line(crate::text::unknown_name(
                    "MCP server",
                    &server,
                    manager.server_names(),
                ));
                break 'prompt;
            };
            // Map positional args to declared prompt argument names (lookup via
            // prompts/list).
            let arg_names = match mcp::list_prompts(
                &entry,
                &tokio_util::sync::CancellationToken::new(),
            )
            .await
            {
                Ok(prompts) => prompts
                    .into_iter()
                    .find(|p| p.name == prompt_name)
                    .and_then(|p| p.arguments)
                    .map(|args| args.into_iter().map(|a| a.name).collect::<Vec<_>>())
                    .unwrap_or_default(),
                Err(error) => {
                    // The `McpConnection` error already names the server and the
                    // operation, so wrapping it here would say both twice.
                    with_console(console, |console| console.error(&error));
                    Vec::new()
                }
            };
            let mut arguments: Option<serde_json::Map<String, serde_json::Value>> = None;
            if !arg_names.is_empty() {
                let mut map = serde_json::Map::new();
                for (i, name) in arg_names.iter().enumerate() {
                    if let Some(value) = args.get(i) {
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
                    // Render the prompt messages as a single user turn, same shape
                    // as the `mcp_prompt_get` tool output.
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
                        // Every other exit from this arm prints; a server whose
                        // prompt renders to nothing would otherwise return the user
                        // straight to a fresh prompt, which reads as "the command
                        // did nothing" rather than "the prompt was empty".
                        crate::streams::write_stderr_line(format!(
                            "'{server}:{prompt_name}' rendered an empty prompt."
                        ));
                    } else {
                        let Ok(input) = crate::agent::TurnInput::from_parts(user_input, Vec::new())
                        else {
                            crate::streams::write_stderr_line("Nothing to send.");
                            return AfterCommand::Continue;
                        };
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
                            Err(error) => report_failure(console, &error),
                        }
                    }
                }
                Err(error) => {
                    with_console(console, |console| console.error(&error));
                }
            }
        }
        SlashCommand::MemoryList => {
            if let Err(error) = crate::cli::memory::run_list(&store.memory_store(true)).await {
                with_console(console, |console| console.error(&error));
            }
        }
        SlashCommand::MemoryShow { name } => {
            if let Err(error) = crate::cli::memory::run_show(&store.memory_store(true), &name).await
            {
                with_console(console, |console| console.error(&error));
            }
        }
        SlashCommand::ScheduleShow { id } => match session_id {
            Some(session) => {
                if let Err(error) =
                    crate::cli::schedule::show(store, &id, &config.schedule, Some(session)).await
                {
                    with_console(console, |console| console.error(&error));
                }
            }
            None => crate::streams::write_stderr_line("No active session yet."),
        },
        // Scoped to the session in the REPL, unlike `meka schedule list`, which has no
        // conversation to be "this one" and so shows every session's jobs.
        SlashCommand::ScheduleList => match session_id {
            Some(id) => {
                if let Err(error) =
                    crate::cli::schedule::run_list_for_session(store, id, &config.schedule).await
                {
                    with_console(console, |console| console.error(&error));
                }
            }
            None => crate::streams::write_stderr_line("No active session yet."),
        },
        SlashCommand::ScheduleCancel { id } => match session_id {
            Some(session) => {
                match store
                    .schedule_store()
                    .cancel_scheduled_job(session, &id)
                    .await
                {
                    Ok(Some(canceled)) => {
                        crate::streams::write_stderr_line(format!(
                            "Canceled job {}.",
                            &canceled[..8.min(canceled.len())]
                        ));
                    }
                    Ok(None) => {
                        crate::streams::write_stderr_line(format!(
                            "No scheduled job matching '{id}'."
                        ));
                    }
                    Err(error) => with_console(console, |console| console.error(&error)),
                }
            }
            None => crate::streams::write_stderr_line("No active session yet."),
        },
        SlashCommand::TaskList => match session_id {
            Some(id) => {
                if let Err(error) = crate::cli::background::run_list_for_session(store, id).await {
                    with_console(console, |console| console.error(&error));
                }
            }
            None => crate::streams::write_stderr_line("No active session yet."),
        },
        SlashCommand::TaskShow { id } => match session_id {
            Some(session) => {
                if let Err(error) =
                    crate::cli::background::show(store, session, &id, crate::render::Stream::Stderr)
                        .await
                {
                    with_console(console, |console| console.error(&error));
                }
            }
            None => crate::streams::write_stderr_line("No active session yet."),
        },
        SlashCommand::TaskCancel { id } => match session_id {
            Some(session) => {
                // Recorded first, then signaled: `finish_background_task` only
                // overwrites a `running` row, so a task finishing in the same instant
                // cannot report success after the user was told it stopped.
                match crate::cli::background::cancel(store, session, id.as_deref()).await {
                    Ok(canceled) if canceled.is_empty() => {
                        crate::streams::write_stderr_line("No running background tasks.")
                    }
                    Ok(canceled) => {
                        for task_id in &canceled {
                            agent.background_tasks().cancel(task_id).await;
                        }
                        crate::streams::write_stderr_line(format!(
                            "Canceling {} background task(s).",
                            canceled.len()
                        ));
                    }
                    Err(error) => with_console(console, |console| console.error(&error)),
                }
            }
            None => crate::streams::write_stderr_line("No active session yet."),
        },
        SlashCommand::SkillList => {
            if let Err(error) = crate::cli::skills::run_list(&config.skill_roots(), false) {
                with_console(console, |console| console.error(&error));
            }
        }
        SlashCommand::SkillInvoke { name, extra } => 'invoke: {
            // Labeled block so the early-exit error paths can `break 'invoke` out of
            // the arm body without skipping the `AgentToReplEvent::Done` send below;
            // `continue` would short-circuit the outer `while let`, leaving the REPL
            // stuck in `wait_for_agent` and never drawing the next prompt.
            let installed = agent.skills().current().await;
            let Some(skill) = installed.find(&name) else {
                let message = match installed.skip_reason(&name) {
                    // The same distinction `skill_read` draws for the model, in the
                    // same words: a file that is there and unreadable is not a missing
                    // skill.
                    Some(_) => installed.unavailable(&name),
                    None => crate::text::unknown_name(
                        "skill",
                        &name,
                        installed.skills.iter().map(|skill| skill.name.as_str()),
                    ),
                };
                with_console(console, |console| console.error(&message));
                break 'invoke;
            };
            let body = match skills::load_skill_body(skill).await {
                Ok(body) => body,
                Err(error) => {
                    with_console(console, |console| {
                        console.error(&format!("failed to load skill '{name}': {error}"))
                    });
                    break 'invoke;
                }
            };
            // Prepend the user's free-form directive to the skill body when present.
            // The blank-line separator gives the model a visual cue that the first
            // paragraph is the user's "do this skill, but with this twist" and the rest
            // is the skill's static body.
            let body = if extra.is_empty() {
                body
            } else {
                format!("{extra}\n\n{body}")
            };
            let Ok(input) = crate::agent::TurnInput::from_parts(body, Vec::new()) else {
                crate::streams::write_stderr_line("Nothing to send.");
                return AfterCommand::Continue;
            };
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
                Err(error) => report_failure(console, &error),
            }
        }
        SlashCommand::Status => {
            render::render_session_status(&crate::host::format_status(
                agent,
                providers,
                messages.len(),
            ));
        }
        SlashCommand::Usage => match agent.fetch_usage().await {
            Ok(Some(usage)) => render::render_account_usage(&usage),
            Ok(None) => with_console(console, |console| {
                console.hint("Account usage is not available for this backend.")
            }),
            Err(error) => with_console(console, |console| console.error(&error)),
        },
        SlashCommand::History(limit) => {
            let materialized = messages.as_slice();
            let slice = match limit {
                Some(n) => render::last_n_turns(materialized, n),
                None => materialized,
            };
            // Say so rather than printing nothing, like every other list command
            // (`/tasks`, `/memory`, `/skill`). Silence here would be ambiguous between
            // "no history" and "the command did not run", and it would leave the
            // `[display]` blank lines bracketing an empty region. `/history 0` asks
            // for nothing and gets the neutral wording: there may well be a
            // conversation, it just wasn't what was asked for.
            // Announced above with every other host-answered command, so nothing is owed at the
            // first row.
            if !render::render_message_history(
                slice,
                &crate::host::terminal::history_render_options(config),
                || {},
            ) {
                if materialized.is_empty() {
                    crate::streams::write_stderr_line("No conversation history yet.");
                } else {
                    crate::streams::write_stderr_line("Nothing to show.");
                }
            }
        }
        // The REPL thread answers these itself, listed rather than swept into a wildcard. The
        // wildcard is what let a forwarded command match nothing and still get its episode
        // brackets; naming them means adding a variant fails here until someone has decided
        // which side owns it.
        SlashCommand::Cd(_)
        | SlashCommand::Clear
        | SlashCommand::Exit
        | SlashCommand::Help
        | SlashCommand::Permission(_)
        | SlashCommand::Approvals(_)
        | SlashCommand::Profile(_) => {
            debug_assert!(
                false,
                "`answered_by` says the REPL answers this, so the forwarding arm should not have \
                 sent it here"
            );
        }
    }
    *session_id_cell = session_id;
    if agent_event_sender
        .send(repl::AgentToReplEvent::Done)
        .is_err()
    {
        return AfterCommand::Leave;
    }
    AfterCommand::Continue
}

/// What answering a command needs from the host loop.
///
/// Explicit rather than implicit in a closure over `run_interactive`'s locals, which is worth the
/// noise: this is the list of things a slash command can reach, readable without the match arm.
pub(crate) struct HostCommandContext<'a> {
    pub(crate) agent: &'a agent::Agent,
    /// Where a turn this command runs publishes its token for Ctrl+C.
    pub(crate) cancel: &'a crate::host::CancelCell,
    pub(crate) agent_event_sender: &'a std::sync::mpsc::Sender<repl::AgentToReplEvent>,
    pub(crate) config: &'a crate::config::ResolvedConfig,
    pub(crate) console: &'a Arc<std::sync::Mutex<crate::console::Console>>,
    pub(crate) mcp_manager: &'a Option<Arc<mcp::McpClientManager>>,
    pub(crate) messages: &'a mut conversation::Conversation,
    pub(crate) providers: &'a Arc<provider::ProviderRegistry>,
    /// `/fork` moves the session the loop is serving, so this is the loop's own cell.
    pub(crate) session_id: &'a mut Option<uuid::Uuid>,
    pub(crate) session_lock: &'a crate::store::SessionLockSlot,
    pub(crate) store: &'a Store,
    pub(crate) token_store: &'a crate::store::TokenStore,
}

pub(crate) enum SlashCommand {
    Exit,
    Help,
    Clear,
    Session,
    Permission(Option<String>),
    /// `/approvals [on|off]`: show or set whether calls above the level are submitted for
    /// approval.
    Approvals(Option<String>),
    /// `/profile [name]`: show which profile this session runs on, or move it to another.
    Profile(Option<String>),
    /// `/compact [instructions]`: compact now, optionally saying what to keep or drop.
    Compact(Option<String>),
    Export,
    /// `/fork`: copy the current session and continue in the copy. The in-memory conversation is
    /// untouched, so the branch happens at the current head and the original freezes where it was.
    Fork,
    Cd(Option<String>),
    /// `/mcp <server>:<prompt> [args...]`: render an MCP prompt and send its messages as the next
    /// user turn.
    McpPrompt {
        server: String,
        prompt: String,
        args: Vec<String>,
    },
    /// `/mcp list`: display configured MCP servers.
    McpList,
    /// `/mcp reconnect <server>`: smoke-test connect for one server.
    McpReconnect {
        server: String,
    },
    /// `/mcp login <server>`: run the OAuth flow from the REPL.
    McpLogin {
        server: String,
    },
    /// `/mcp logout <server>`: clear stored credentials + revoke.
    McpLogout {
        server: String,
    },
    /// `/mcp reconnect`, `/mcp login` or `/mcp logout` typed without the server it acts on.
    /// Refused by name rather than as an unknown command: the command is known, its argument is
    /// missing.
    McpMissingServer {
        verb: String,
    },
    /// `/mcp <word>` where the word is neither a verb nor a `<server>:<prompt>` spec.
    McpUnknownVerb {
        verb: String,
    },
    /// `/memory` (no argument): list saved memories, most important first.
    MemoryList,
    ScheduleList,
    /// One job in full, including the whole prompt and what a gate runs.
    ///
    /// The table dropped `When` and `Check` to fit its budget, and `/schedule` is the surface that
    /// has no `meka schedule show` to fall back on without leaving the REPL.
    ScheduleShow {
        id: String,
    },
    ScheduleCancel {
        id: String,
    },
    /// `/tasks`: list this session's background tasks.
    TaskList,
    /// One task in full, including the id the listing shortens.
    TaskShow {
        id: String,
    },
    /// `/tasks cancel <id>`, or `/tasks cancel --all`.
    TaskCancel {
        /// `None` means every running task in this session.
        id: Option<String>,
    },
    /// `/memory <name>`: print one memory's body, the in-session equivalent of
    /// `meka memory show`.
    MemoryShow {
        name: String,
    },
    /// `/skill` (no argument): list installed skills.
    SkillList,
    /// `/skill <name> [extra...]`: invoke a user-invocable skill directly. Anything the user types
    /// after the skill name is captured verbatim in `extra` and prepended to the rendered skill
    /// body before the agent turn, so the model reads the user's directive first and the skill body
    /// as the method. Empty when the user just typed `/skill <name>`.
    SkillInvoke {
        name: String,
        extra: String,
    },
    /// `/status`: print the profile, model, context use and the cumulative session stats (turns,
    /// tokens, cache hit ratio, image redactions).
    Status,
    /// `/usage`: fetch and print the account's rate-limit usage from the active provider.
    Usage,
    /// `/rewind [N]`: drop the last `N` turns (default 1) from the conversation, cutting at a
    /// clean user boundary so no `tool_use` is separated from its `tool_result`. The event log is
    /// append-only, so `meka session export` still shows what was dropped.
    ///
    /// The manual counterpart to `run_turn`'s automatic repair: it reaches content the automatic
    /// path cannot, namely anything the provider refuses that was committed before this turn.
    Rewind(Option<usize>),
    /// `/rewind` with an argument that is not a turn count. Parsing it as absent rewound one turn
    /// and reported that count, which the user had not asked for.
    RewindInvalid(String),
    /// `/history [N]`: reprint past conversation in REPL style. Bare `/history` dumps every
    /// materialized message; `/history N` shows the last `N` turns (turn = user prompt + the agent
    /// work it triggered). Any non-numeric argument (e.g. `all`) falls back to the dump-everything
    /// path.
    History(Option<usize>),
}
/// Who answers a slash command.
///
/// The split is not arbitrary: a command needs the host loop exactly when it needs the live
/// `Agent`, the conversation, or the session id that `/fork` moves. Everything else the REPL thread
/// has to hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Answerer {
    /// Answered on the REPL thread, where it was parsed.
    Repl,
    /// Forwarded to the host loop; see [`super::commands::answer`].
    Host,
}
impl SlashCommand {
    /// Which side answers this command.
    ///
    /// Exhaustive, and both sides read it, so a new variant fails to compile until both have been
    /// considered; see the module docs.
    pub(crate) fn answered_by(&self) -> Answerer {
        match self {
            SlashCommand::Cd { .. } => Answerer::Repl,
            SlashCommand::Clear => Answerer::Repl,
            SlashCommand::Exit => Answerer::Repl,
            SlashCommand::Help => Answerer::Repl,
            SlashCommand::Permission { .. } => Answerer::Repl,
            SlashCommand::Approvals { .. } => Answerer::Repl,
            SlashCommand::Profile { .. } => Answerer::Repl,
            SlashCommand::Compact { .. } => Answerer::Host,
            SlashCommand::Export => Answerer::Host,
            SlashCommand::Fork => Answerer::Host,
            SlashCommand::History { .. } => Answerer::Host,
            SlashCommand::McpList => Answerer::Host,
            SlashCommand::McpLogin { .. } => Answerer::Host,
            SlashCommand::McpLogout { .. } => Answerer::Host,
            SlashCommand::McpMissingServer { .. } => Answerer::Host,
            SlashCommand::McpUnknownVerb { .. } => Answerer::Host,
            SlashCommand::McpPrompt { .. } => Answerer::Host,
            SlashCommand::McpReconnect { .. } => Answerer::Host,
            SlashCommand::MemoryList => Answerer::Host,
            SlashCommand::MemoryShow { .. } => Answerer::Host,
            SlashCommand::Rewind { .. } => Answerer::Host,
            SlashCommand::RewindInvalid(_) => Answerer::Host,
            SlashCommand::ScheduleCancel { .. } => Answerer::Host,
            SlashCommand::ScheduleList => Answerer::Host,
            SlashCommand::ScheduleShow { .. } => Answerer::Host,
            SlashCommand::Session => Answerer::Host,
            SlashCommand::SkillInvoke { .. } => Answerer::Host,
            SlashCommand::SkillList => Answerer::Host,
            SlashCommand::Status => Answerer::Host,
            SlashCommand::TaskCancel { .. } => Answerer::Host,
            SlashCommand::TaskList => Answerer::Host,
            SlashCommand::TaskShow { .. } => Answerer::Host,
            SlashCommand::Usage => Answerer::Host,
        }
    }
}
pub(crate) fn parse_slash_command(input: &str) -> Option<SlashCommand> {
    let input = input.strip_prefix('/')?;
    let mut parts = input.splitn(2, char::is_whitespace);
    let command = parts.next()?;
    let argument = parts.next().map(|s| s.trim().to_string());

    match command {
        "exit" | "quit" => Some(SlashCommand::Exit),
        "help" | "?" => Some(SlashCommand::Help),
        "clear" => Some(SlashCommand::Clear),
        "session" => Some(SlashCommand::Session),
        "memory" => Some(parse_memory_slash(argument.as_deref().unwrap_or(""))),
        "schedule" => Some(parse_schedule_slash(argument.as_deref().unwrap_or(""))),
        "tasks" => Some(parse_tasks_slash(argument.as_deref().unwrap_or(""))),
        "permission" => Some(SlashCommand::Permission(argument)),
        "approvals" => Some(SlashCommand::Approvals(argument)),
        "profile" => Some(SlashCommand::Profile(argument)),
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
        "history" => Some(SlashCommand::History(
            argument
                .as_deref()
                .and_then(|s| s.trim().parse::<usize>().ok()),
        )),
        _ => None,
    }
}
/// Parse the argument to `/memory …`.
///
/// - Empty argument (bare `/memory`) → list saved memories.
/// - Otherwise the whole argument is a memory name to display. Unlike `/skill` there is no
///   free-form trailer: showing a memory is a read, not a turn, so there is nothing to prepend it
///   to. Extra tokens would be silently dropped, so they make the name invalid instead and the
///   lookup reports it.
pub(super) fn parse_memory_slash(rest: &str) -> SlashCommand {
    let rest = rest.trim();
    if rest.is_empty() {
        return SlashCommand::MemoryList;
    }
    SlashCommand::MemoryShow {
        name: rest.to_string(),
    }
}
/// Parse the argument to `/schedule …`.
///
/// Bare `/schedule` lists; `/schedule show <id>` prints one in full; `/schedule cancel <id>`
/// cancels. There is no `create`: a job's prompt is prose the agent writes for its own future self,
/// and typing one at the REPL would be doing the agent's job badly.
pub(super) fn parse_schedule_slash(rest: &str) -> SlashCommand {
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
/// Parse the argument to `/tasks …`.
///
/// Bare `/tasks` lists; `/tasks show <id>` prints one in full; `/tasks cancel <id>` stops one;
/// `/tasks cancel --all` stops them all. There is no way to *start* one here, for the same reason
/// `/schedule` has no `create`: the decision to detach belongs to the agent making the call.
pub(super) fn parse_tasks_slash(rest: &str) -> SlashCommand {
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
/// Parse the argument to `/skill …`.
///
/// - Empty argument (bare `/skill`) → list installed skills. There is no `list` keyword: that token
///   would be treated as a skill name to invoke.
/// - Otherwise: first whitespace-separated token is the skill name; the remainder (if any) is
///   free-form extra context that gets prepended to the skill body before the agent turn. The
///   remainder is trimmed so trailing whitespace doesn't bloat the body.
pub(super) fn parse_skill_slash(rest: &str) -> SlashCommand {
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
/// Parse the argument to `/mcp …`.
///
/// Every shape parses to a command, so nothing here falls through to "Unknown command": a verb
/// typed without its server name and a first word that is neither a verb nor a `<server>:<prompt>`
/// spec are answered as what they are, an `/mcp` the user has to finish.
pub(super) fn parse_mcp_slash(rest: &str) -> SlashCommand {
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
    // `<server>:<prompt> [args...]`: the first word is the prompt spec.
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
/// The line for a slash command the REPL does not know. It names the command and only the command:
/// `/frob a b` is refused as `/frob`, because the argument was never read.
///
/// Not the `unknown_name` template: listing every slash command on one line is noise, and `/help`
/// already is the list.
pub(crate) fn unknown_command_message(line: &str) -> String {
    let command = line.split(char::is_whitespace).next().unwrap_or(line);
    format!("Unknown command: {command}. Type /help for available commands.")
}
pub(super) fn print_help() {
    crate::streams::write_stderr_line("Commands:");
    for command in crate::host::COMMANDS {
        let left = if command.arg_hint.is_empty() {
            format!("/{}", command.name)
        } else {
            format!("/{} {}", command.name, command.arg_hint)
        };
        crate::streams::write_stderr_line(format!("  {left:<33}  {}", command.help));
        if command.name == "mcp" {
            // The /mcp subcommands are arguments, not top-level commands, so they are absent from
            // COMMANDS; list them here so help still documents the full grammar. Keep this set in
            // step with `parse_mcp_slash` and `MCP_SUBCOMMANDS`.
            crate::streams::write_stderr_line(format!(
                "  {:<33}  List configured MCP servers",
                "/mcp list"
            ));
            crate::streams::write_stderr_line(format!(
                "  {:<33}  Reconnect smoke-test for one server",
                "/mcp reconnect <server>"
            ));
            crate::streams::write_stderr_line(format!(
                "  {:<33}  Run the OAuth flow for a server",
                "/mcp login <server>"
            ));
            crate::streams::write_stderr_line(format!(
                "  {:<33}  Clear stored credentials for a server",
                "/mcp logout <server>"
            ));
            crate::streams::write_stderr_line(format!(
                "  {:<33}  Render an MCP prompt as the next turn",
                "/mcp <server>:<prompt> [args]"
            ));
        }
    }
    crate::streams::write_stderr_line("");
    crate::streams::write_stderr_line("Shortcuts:");
    crate::streams::write_stderr_line("  !<command>    Execute a shell command directly");
    crate::streams::write_stderr_line("  Shift+Tab     Cycle permission level");
    crate::streams::write_stderr_line("  Ctrl+D        Exit the shell");
}
