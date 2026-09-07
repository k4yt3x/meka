//! The `--oneshot` host: one turn, printed, and out.

use super::{terminal::*, *};

/// The frontend a `--format json` run drives: nothing reaches stdout during the turn, and what the
/// turn produced is read back afterwards to assemble the one object the run prints.
///
/// Permission requests are denied, as they are on the plain one-shot path, where the approval
/// channel has no receiver: there is nobody to ask. The refusal is recorded as the same notice that
/// path prints, so the report's `notices` says why a call the model made did not run.
struct JsonFrontend {
    events: std::sync::Mutex<Vec<crate::frontend::FrontendEvent>>,
    /// Where the one line the report does not carry goes; see `emit`.
    console: Arc<std::sync::Mutex<crate::console::Console>>,
    show_session_id_on_create: bool,
}

#[async_trait::async_trait]
impl crate::frontend::Frontend for JsonFrontend {
    async fn emit(&self, event: crate::frontend::FrontendEvent) {
        // Printed as the REPL frontend prints it, on stderr through the console, so the setting
        // holds under `--format json` too and a run that never reaches its report has still said
        // which session it made. Stdout stays the report's alone.
        if let crate::frontend::FrontendEvent::SessionStarted { id } = &event
            && self.show_session_id_on_create
        {
            with_console(&self.console, |console| {
                console.session_id("Creating new session", &id.to_string())
            });
        }
        crate::sync::lock(&self.events).push(event);
    }

    async fn request_permission(
        &self,
        request: crate::frontend::PermissionRequest,
    ) -> crate::frontend::PermissionOutcome {
        self.emit(crate::frontend::FrontendEvent::Notice(
            crate::frontend::Notice::approval_refused_without_asking(&request.tool_name),
        ))
        .await;
        crate::frontend::PermissionOutcome::Deny
    }
}

/// One tool call as the JSON report lists it.
#[derive(serde::Serialize)]
struct ToolCallReport {
    name: String,
    input: serde_json::Value,
    is_error: bool,
}

/// The object a `--oneshot --format json` run prints: the whole turn, once, on stdout.
#[derive(serde::Serialize)]
struct TurnReport {
    /// Omitted for a turn interrupted before the session existed, which is the one way a run ends
    /// with a report and no session.
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<uuid::Uuid>,
    profile: String,
    /// `end_turn`, `max_tokens`, `refusal`, or `interrupted`.
    stop_reason: &'static str,
    /// The assistant's text, the rounds joined by a blank line.
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    refusal_text: Option<String>,
    tool_calls: Vec<ToolCallReport>,
    usage: crate::stats::TokenUsage,
    /// Provider advisories raised during the turn, in order; the same shape the HTTP API's turn
    /// response carries them in.
    notices: Vec<crate::frontend::NoticeView>,
}

/// Fold the recorded events into the report. Usage is summed over the `TokenUsage` events the turn
/// emitted, one per turn today; the sum keeps the report right should that become one per round.
fn assemble_report(
    events: Vec<crate::frontend::FrontendEvent>,
    outcome: Option<crate::agent::TurnOutcome>,
    session_id: Option<uuid::Uuid>,
    profile: String,
) -> TurnReport {
    use crate::frontend::FrontendEvent;
    let mut text = String::new();
    // Set by a tool call, so the next round's text starts a paragraph rather than running on from
    // the sentence the model wrote before it called the tool.
    let mut round_boundary = false;
    let mut tool_calls: Vec<ToolCallReport> = Vec::new();
    let mut started: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut usage = crate::stats::TokenUsage::default();
    let mut notices: Vec<crate::frontend::NoticeView> = Vec::new();
    for event in events {
        match event {
            FrontendEvent::AssistantTextDelta(delta) => {
                if round_boundary && !text.is_empty() {
                    text.push_str("\n\n");
                }
                round_boundary = false;
                text.push_str(&delta);
            }
            FrontendEvent::ToolCallStarted {
                id, name, input, ..
            } => {
                round_boundary = true;
                started.insert(id, tool_calls.len());
                tool_calls.push(ToolCallReport {
                    name,
                    input,
                    is_error: false,
                });
            }
            FrontendEvent::ToolCallCompleted { id, is_error, .. } => {
                if let Some(index) = started.get(&id)
                    && let Some(call) = tool_calls.get_mut(*index)
                {
                    call.is_error = is_error;
                }
            }
            FrontendEvent::TokenUsage(round) => {
                usage.input_tokens += round.input_tokens;
                usage.output_tokens += round.output_tokens;
                usage.cache_creation_input_tokens += round.cache_creation_input_tokens;
                usage.cache_read_input_tokens += round.cache_read_input_tokens;
            }
            FrontendEvent::Notice(notice) => notices.push(notice.into()),
            _ => {}
        }
    }
    let (stop_reason, refusal_text) = match outcome {
        Some(crate::agent::TurnOutcome::EndTurn) => ("end_turn", None),
        Some(crate::agent::TurnOutcome::MaxTokens) => ("max_tokens", None),
        Some(crate::agent::TurnOutcome::Refusal(reason)) => {
            ("refusal", (!reason.is_empty()).then_some(reason))
        }
        None => ("interrupted", None),
    };
    TurnReport {
        session_id,
        profile,
        stop_reason,
        text,
        refusal_text,
        tool_calls,
        usage,
        notices,
    }
}

pub(crate) async fn run_oneshot(
    config: ResolvedConfig,
    store: Store,
    prompt: String,
    mcp_manager: Option<Arc<crate::mcp::McpClientManager>>,
) -> anyhow::Result<()> {
    let config = Arc::new(config);
    // Ahead of everything with a side effect: a run refused for an empty prompt must not have
    // created a session row, taken a lock or swept anything on the way to saying so. The outcomes
    // it carries ride on it once the session is known.
    let input = crate::agent::TurnInput::from_parts(prompt, Vec::new())?;
    // Oneshot has no REPL, so approval requests can't reach a human. The channel below is
    // intentionally disconnected on the receiver side: `ReplFrontend::request_permission`'s `send`
    // will fail, the call is denied, and a notice on the console says which tool was refused.
    let (noninteractive_sender, _) =
        std::sync::mpsc::channel::<crate::host::repl::editor::AgentToReplEvent>();
    // One episode for the whole run. Oneshot draws no prompt, so there is no line above to space
    // away from and no prompt below to space towards: both blanks are configured off here rather
    // than left to a bracket that would have nothing to bracket against. What the console still
    // buys is the rest of it -- one owner for the streaming renderer, and the row-settling that
    // keeps a turn's last paragraph off an MCP progress line.
    let console = Arc::new(std::sync::Mutex::new(crate::console::Console::new(
        crate::console::Spacing {
            newline_before_prompt: false,
            newline_after_prompt: false,
        },
        config.render_mode,
    )));
    // Before the first thing that logs. Off-prompt, `tracing` writes straight to stderr, and the
    // row it lands on may be one the console intends to erase; giving the relay this handle is what
    // lets a mid-turn `warn!` settle that row first instead of being wiped with it.
    crate::relay::RELAY.install_console(&console);
    with_console(&console, |console| {
        console.open_episode(
            crate::console::RowState::Empty,
            crate::console::Neighbor::Shell,
        )
    });
    let _last_episode = LastEpisode(Arc::clone(&console));
    // `--format json` records the turn instead of rendering it: stdout carries one object at the
    // end and nothing before it, so `meka -p … --format json | jq` sees only the object.
    let json_frontend = matches!(
        config.request.output_format,
        crate::config::OutputFormat::Json
    )
    .then(|| {
        Arc::new(JsonFrontend {
            events: std::sync::Mutex::new(Vec::new()),
            console: Arc::clone(&console),
            show_session_id_on_create: config.show_session_id_on_create,
        })
    });
    let oneshot_frontend: Arc<dyn crate::frontend::Frontend> = match &json_frontend {
        Some(frontend) => Arc::clone(frontend) as Arc<dyn crate::frontend::Frontend>,
        None => Arc::new(crate::host::repl::frontend::ReplFrontend::new(
            crate::host::repl::frontend::ReplFrontendConfig {
                console: Arc::clone(&console),
                show_session_id_on_create: config.show_session_id_on_create,
                show_token_usage: config.show_token_usage,
                thinking_show_content: config.thinking_show_content,
                tool_params: config.tool_params,
                agent_event_sender: noninteractive_sender,
            },
        )),
    };
    let launch_cwd = std::env::current_dir().unwrap_or_else(|error| {
        tracing::warn!("failed to read the process working directory at startup: {error}");
        std::path::PathBuf::from(".")
    });
    // Resolved before the agent is built, not after: which session this is decides which provider
    // profile the agent runs on, which level it runs at and which directory it opens in, and the
    // agent carries all three.
    let ResumedSession {
        mut session_id,
        mut messages,
        lock: _session_lock,
        permission: start_permission,
        approvals: start_approvals,
        repin,
        permission_to_record,
        cwd: recorded_cwd,
    } = resolve_session_resume(&store, &config, &console).await?;
    // After the resume, whose lock is what spares the session this run was asked for.
    sweep_expired_sessions(&config, &store).await?;

    // The same deps the long-lived hosts build, so all four hosts assemble a session identically.
    let shared =
        Arc::new(build_shared_deps(Arc::clone(&config), store.clone(), mcp_manager.clone()).await?);
    let providers = Arc::clone(&shared.providers);
    // A resumed session reopens where it was, not where this shell is. See
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
    // Approvals have nowhere to ask from here: `oneshot_frontend` is built on a channel whose
    // receiver is dropped, so every approval request fails to send and the tool is refused. Say so
    // once, up front, rather than letting the run look like the model simply chose not to use its
    // tools.
    //
    // Against the switch the run actually starts with, which a resumed session brings with it,
    // rather than against the configured default.
    if start_approvals && start_permission != crate::permission::Permission::Unrestricted {
        tracing::warn!(
            "approvals are on but a one-shot run cannot prompt, so every call above the level is \
             refused; run at the `--permission` it needs"
        );
    }

    // Before the agent, because the agent resolves the row: a repin that has not landed yet would
    // build this run on the binding the session is leaving.
    commit_resume_repin(&store, &providers, session_id, repin).await?;
    let agent = build_session_agent(&shared, SessionSpec {
        session_id,
        permission: shared_permission,
        frontend: oneshot_frontend,
        cwd,
        roots: crate::workspace::SharedRoots::new(config.request.writable_roots.clone()),
        // `--oneshot` prints one answer and exits; nothing reads a live gauge.
        context_tokens: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        context_overhead: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        context_window: Arc::new(std::sync::atomic::AtomicU64::new(0)),
    })
    .await?;
    // After the agent, so a start the builder refused leaves the row at the level it had.
    record_resume_permission(&store, session_id, permission_to_record).await;
    let cancel = crate::host::CancelCell::default();
    install_interrupt_handler(cancel.clone(), &agent, Arc::clone(&console));
    // Admitted before the outcome claim below, which can wait on the MCP servers: a Ctrl+C in
    // that window has to reach the turn rather than be lost to an epoch sampled after it.
    let admission = cancel.admit();

    // A resume inherits whatever the last process left undelivered, and this turn is the only one
    // this run has. Joined to the prompt rather than appended as its own message, for the reason
    // `crate::background::render_outcomes_riding` gives. The collection *after* the turn stays: it
    // reports what finished while this turn ran, which no turn is left to carry.
    let outcomes = if config.background.enabled {
        collect_background_outcomes(&agent, &store, session_id).await
    } else {
        Vec::new()
    };
    let input = input.riding(outcomes);

    // A failure is carried to the end rather than returned here. Everything below is the run's
    // exit path -- the wait for detached work, the outcome report, the registry detach, the MCP
    // shutdown -- and a turn that failed after detaching a command needs it as much as one that
    // succeeded: returning early dropped the runtime with the task parked at an await, so the
    // child process group ran on with nothing tracking it and its row stayed `running` until a
    // later open swept it to `interrupted`.
    let (turn_error, outcome) = match run_turn_interruptible(
        &cancel,
        admission,
        &agent,
        &mut session_id,
        &mut messages,
        input,
    )
    .await
    {
        Ok(outcome) => (None, Some(outcome)),
        // Kept as the run's error, so the process leaves with 130 the way every host does on a
        // Ctrl+C, rather than with 0 over a partial answer. The report below still goes out for it.
        Err(error @ crate::error::MekaError::Interrupted) => {
            with_console(&console, |console| console.annotation("interrupted"));
            (Some(error), None)
        }
        Err(error) => (Some(error), None),
    };
    let interrupted = matches!(turn_error, Some(crate::error::MekaError::Interrupted));
    // Closed whichever way the turn went, so a turn that streamed a partial answer and then failed
    // still shows what it streamed. `TurnFinished` closes the happy path; nothing closed this one.
    close_console_episode(&console);
    // The report goes out for a turn that ended, whichever way, an interrupt included: its
    // `stop_reason` says so, and the exit code says so again. A failed turn is reported on stderr
    // and by the exit code alone, because an object beside it would read as an answer.
    if let Some(frontend) = &json_frontend
        && (turn_error.is_none() || interrupted)
    {
        let events = std::mem::take(&mut *crate::sync::lock(&frontend.events));
        let report = assemble_report(events, outcome, session_id, agent.profile());
        crate::render::write_stdout_line(&serde_json::to_string(&report)?)?;
    }

    // A one-shot run exits with the turn, so there is no later turn to deliver an outcome into.
    // Waiting here degrades a background call into a slow synchronous one, which is a worse deal
    // than the agent asked for but an honest one; exiting instead would leave a promise nothing can
    // keep, and kill the work halfway through besides.
    if config.background.enabled
        && let Some(id) = session_id
    {
        let outstanding = agent.background_tasks().running_count(id).await;
        if outstanding > 0 {
            tracing::info!(
                "waiting for {outstanding} background task(s) before exiting; a one-shot run has no \
                 later turn to report them in"
            );
            // Raced against Ctrl+C: a one-shot has no REPL loop and no per-turn signal listener by
            // this point, so an unbounded await would ignore the key for as long as a task ran. The
            // outcomes collected just below still report whatever did finish.
            let tasks = agent.background_tasks();
            tokio::select! {
                _ = tasks.wait_for_session(id) => {}
                _ = INTERRUPT_RELAY.pressed.notified() => {
                    with_console(&console, |console| {
                        console.annotation("stopped waiting for background tasks")
                    });
                }
            }
        }
        // Collected unconditionally, not only when this process started something. Resuming a
        // session sweeps whatever the *last* process left running into `interrupted`, and without
        // this a one-shot resume would answer the prompt and exit while that report sat
        // undelivered.
        //
        // Ungated, unlike the claim before the turn: this one is printed for the human on the way
        // out, so whether a turn could start is not a question about it.
        let outcomes = match session_id {
            Some(id) => crate::host::claim_outcomes_now(&store, id).await,
            None => Vec::new(),
        };
        if !outcomes.is_empty() {
            // Printed rather than delivered as a turn: the agent's answer has already been given
            // and the process is on its way out, so this is for the human reading the output.
            //
            // Through the console, which owns every row this run writes; spacing is off here, so
            // the leading blank is this block's own. Nothing in a one-shot leaves a transient row
            // to settle today (MCP progress is dropped for want of a REPL to draw it), so this is
            // the module's one-owner rule rather than a fix for a reachable glitch.
            with_console(&console, |console| {
                console.chrome(|| {
                    crate::streams::write_stderr_line("");
                    crate::streams::write_stderr(crate::background::render_outcomes(&outcomes));
                })
            });
        }
    }

    if let Some(id) = session_id
        && config.show_session_id_on_exit
    {
        with_console(&console, |console| {
            console.session_id("Leaving session", &id.to_string())
        });
    }

    // Same pairing the REPL does: this path attached the registry to the manager when the agent was
    // built, so it detaches it here rather than leaving the cycle for the process teardown.
    if let Some(manager) = &mcp_manager {
        crate::tools::mcp_adapter::detach_session_registry(manager, agent.tool_registry()).await;
    }
    drop(agent);

    if let Some(manager) = mcp_manager {
        shutdown_mcp_manager(manager).await;
    }

    if let Some(error) = turn_error {
        return Err(error.into());
    }

    // A one-shot run is the host that gets scripted, and its answer is the whole point of invoking
    // it. A stdout that would not take that answer has to reach the exit code, or `meka -p … >
    // out.txt` against a full disk reports success over an empty file. A reader that hung up is
    // excluded upstream: it stopped reading on purpose, and every tool in a pipeline is entitled to
    // do that. Reported here rather than at the write, so the turn still finishes and the session
    // still records what the model said.
    if let Some(error) = with_console(&console, |console| console.take_lost_output()) {
        return Err(anyhow::Error::new(error).context("the answer did not reach stdout"));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A report for a turn that never got a session leaves `session_id` out rather than sending
    /// `null`, which is the rule every other wire shape follows and what the docs promise.
    #[test]
    fn a_report_without_a_session_omits_the_key() {
        let report = assemble_report(Vec::new(), None, None, "work".to_string());
        let value = serde_json::to_value(&report).expect("serializable");
        assert!(
            value.get("session_id").is_none(),
            "a missing session is an absent key, not `null`: {value}"
        );

        let id = uuid::Uuid::nil();
        let report = assemble_report(Vec::new(), None, Some(id), "work".to_string());
        let value = serde_json::to_value(&report).expect("serializable");
        assert_eq!(value["session_id"], id.to_string());
    }
}
