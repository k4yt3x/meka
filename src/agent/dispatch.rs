//! Turning tool calls into tool results: resolution, approval, the background hand-off, and the
//! call itself.

use super::*;

/// The arguments an approval prompt is shown, which is not quite the arguments the tool receives.
///
/// `background` is Meka's own parameter, spliced by the registry into the schema of every tool that
/// can run in the background and taken out again before dispatch so no tool sees a key it never
/// advertised. It is also the argument that
/// decides whether the call detaches and outlives the turn, so a prompt that showed everything
/// except that would be asking about a different call than the one about to run.
pub(super) fn approval_input(input: &serde_json::Value, detach: bool) -> serde_json::Value {
    let mut shown = input.clone();
    if detach && let Some(fields) = shown.as_object_mut() {
        fields.insert("background".to_string(), serde_json::Value::Bool(true));
    }
    shown
}

/// What the loop resolved: the call's id, its tool's name and its arguments as the model sent
/// them.
pub(super) struct ResolvedCall<'a> {
    pub(super) tool_call_id: &'a str,
    pub(super) name: &'a str,
    pub(super) input: &'a serde_json::Value,
    /// What the model sent when it was not arguments, which refuses the call.
    pub(super) unparsed: Option<&'a crate::conversation::UnparsedArguments>,
}

/// Which door a resolved call comes through; see [`Agent::run_admitted_call`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CallDoor {
    /// Inline dispatch: every door, and a `background: true` is honored.
    Dispatch,
    /// A checkpoint tool: every door, and a `background: true` is stripped, since a checkpoint
    /// tool runs inline whatever the model asked.
    Checkpoint,
    /// The checkpoint's own `context_replace`, which no level refuses: it performs no action, it
    /// hands the summary back, and asking the user to approve the checkpoint's own conclusion
    /// would let a denial silently drop the compaction to the fallback summarizer.
    CheckpointSubmission,
}

impl Agent {
    /// What a screen draws of `output` beyond its words, as its result records it and the
    /// announcement of its end carries it: how a command ended, the change a call made, the task
    /// a call that went to the background became, and the list a checklist call left, as the
    /// batch left it, since the batch has run by the time a result is recorded. The one place a
    /// result's display is made, for the turn and the checkpoint alike. A change is worked out on a
    /// blocking thread, since diffing two large texts can take up to the diff's timeout.
    pub(super) async fn result_display(
        &self,
        name: &str,
        output: &crate::tools::ToolOutput,
    ) -> Option<crate::conversation::ResultDisplay> {
        let mut display = match output.frontend_metadata.clone() {
            Some(metadata @ crate::frontend::ToolOutputMetadata::Diff { .. }) => {
                match tokio::task::spawn_blocking(move || metadata.display()).await {
                    Ok(display) => display,
                    Err(error) => {
                        tracing::warn!("failed to work out the change a call made: {error}");
                        crate::conversation::ResultDisplay::default()
                    }
                }
            }
            Some(metadata) => metadata.display(),
            None => crate::conversation::ResultDisplay::default(),
        };
        if !output.is_error && crate::tools::checklist::changes_the_list(name) {
            display.checklist = Some(self.cells.checklist.get().items);
        }
        display.recorded()
    }

    /// `loaded` is the turn's active-tool set, used only to tell a call made against a schema the
    /// model has actually seen from one made blind; see [`Self::schema_advisory`]. The third value
    /// names the calls whose result is a word about the call rather than the tool's output, which
    /// the `scratchpad` redirect leaves inline; see [`crate::tools::ToolOutput::is_output`].
    pub(super) async fn execute_tool_calls(
        &self,
        assistant_message: &Message,
        loaded: &[String],
        attribution: &crate::provider::Attribution,
        cancellation: CancellationToken,
    ) -> (
        Vec<ContentBlock>,
        Vec<crate::provider::ToolDuration>,
        Vec<String>,
    ) {
        // Emit tool-call indicators in source order. The streaming path already emitted these as
        // `ToolUseEnd` events; this loop only fires for the blocking provider path. Serial so
        // concurrent execution below can't interleave indicators.
        let mut planned: Vec<(&str, &str, &serde_json::Value, Option<&_>)> = Vec::new();
        for block in &assistant_message.content {
            if let ContentBlock::ToolUse {
                id,
                name,
                input,
                unparsed,
                ..
            } = block
            {
                if !self.options.streaming {
                    let display_summary = unparsed
                        .is_none()
                        .then(|| crate::tools::resolve_primary_param(name, input))
                        .flatten();
                    self.cells
                        .frontend
                        .emit(FrontendEvent::ToolCallStarted {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                            display_summary,
                            unparsed: unparsed.clone(),
                        })
                        .await;
                }
                planned.push((id, name, input, unparsed.as_ref()));
            }
        }

        // Dispatch concurrently. `join_all` preserves input ordering so the i-th output corresponds
        // to the i-th planned call.
        let futures = planned.iter().map(|&(id, name, input, unparsed)| {
            let cancellation = cancellation.clone();
            async move {
                use futures::FutureExt;
                let started = std::time::Instant::now();
                // Caught as the background path catches it. The assistant's `tool_use` is already
                // in memory when the tools run, so a panic that takes the turn down leaves the
                // conversation ending on a call with no result, and every later request is
                // rejected until the session is evicted.
                let run = self.resolve_and_execute_tool(
                    ResolvedCall {
                        tool_call_id: id,
                        name,
                        input,
                        unparsed,
                    },
                    loaded,
                    attribution,
                    cancellation,
                );
                let output = match std::panic::AssertUnwindSafe(run).catch_unwind().await {
                    Ok(output) => output,
                    Err(_) => {
                        tracing::error!("tool '{name}' panicked");
                        crate::tools::ToolOutput::text(
                            format!("The tool '{name}' panicked while running."),
                            true,
                        )
                    }
                };
                (output, started.elapsed())
            }
        });
        let outputs = futures::future::join_all(futures).await;

        // Serial pass to accumulate scratchpad hints, emit per-tool completion events in source
        // order, build ToolResult blocks, and emit a single ChecklistUpdated event if any
        // `checklist_*` call landed and actually changed the state.
        let mut results = Vec::with_capacity(planned.len());
        let mut durations = Vec::with_capacity(planned.len());
        let mut inline_results = Vec::new();
        let mut checklist_fired = false;
        for ((id, name, ..), (output, elapsed)) in planned.into_iter().zip(outputs) {
            let (id, name) = (id.to_string(), name.to_string());
            if !output.is_output {
                inline_results.push(id.clone());
            }
            durations.push(crate::provider::ToolDuration {
                name: name.clone(),
                elapsed,
            });
            if crate::tools::checklist::changes_the_list(&name) {
                checklist_fired = true;
            }
            if output.spill_hint != crate::tools::SpillHint::default() {
                self.scratchpad_hints
                    .write()
                    .await
                    .insert(id.clone(), output.spill_hint.clone());
            }
            let display = self.result_display(&name, &output).await;
            // Notify the frontend of completion BEFORE building the ToolResult content block so ACP
            // `tool_call_update` notifications arrive before the next assistant turn's text starts
            // streaming.
            self.cells
                .frontend
                .emit(FrontendEvent::ToolCallCompleted {
                    id: id.clone(),
                    name: name.clone(),
                    is_error: output.is_error,
                    content: output.content.clone(),
                    metadata: output.frontend_metadata.clone(),
                    display: display.clone(),
                })
                .await;
            results.push(ContentBlock::ToolResult {
                tool_use_id: id,
                content: output.content,
                is_error: output.is_error,
                display,
            });
        }
        if checklist_fired {
            let state = self.cells.checklist.get();
            // Suppress re-renders for edits that change nothing. Drop the guard before awaiting
            // the emit. A list emptied by its last disposition is a change like any other: every
            // surface clears or says so on an empty one.
            let changed = {
                let mut last = self.last_rendered_checklist.write().await;
                let shown = last.clone().unwrap_or_default();
                let changed = !state.shows_the_same_items(&shown);
                if changed {
                    *last = Some(state.clone());
                }
                changed
            };
            if changed {
                self.cells
                    .frontend
                    .emit(FrontendEvent::ChecklistUpdated { items: state.items })
                    .await;
            }
        }

        (results, durations, inline_results)
    }

    pub(super) async fn resolve_and_execute_tool(
        &self,
        call: ResolvedCall<'_>,
        loaded: &[String],
        attribution: &crate::provider::Attribution,
        cancellation: CancellationToken,
    ) -> crate::tools::ToolOutput {
        let name = call.name;
        let Some(tool) = self.tool_registry.get(name) else {
            // A tool from a server that never connected was never registered, so it lands here.
            // Saying "unknown" would be false (it exists and is unreachable) and would teach the
            // agent to stop asking for a capability that may be seconds from returning.
            //
            // Asks the registry rather than `self.mcp_manager`: a sub-agent has no manager of its
            // own (see `new_subagent`) but its registry does, and it deserves the same answer.
            //
            // Denied names are excluded: the explanation names the server and its state, so
            // offering it to a worker whose denial is meant to make that server invisible would
            // confirm the server exists and let the worker enumerate the rest by guessing.
            if self.tool_registry.tool_scope().admits_tool(name)
                && let Some(manager) = self.tool_registry.mcp_manager()
                && let Some(reason) = manager.unavailable_tool_reason(name).await
            {
                return crate::tools::ToolOutput::text(reason, true);
            }
            // Namespaced MCP names are long and easy to mangle, and the commonest slip is dropping
            // the `mcp__<server>__` prefix entirely, which no amount of re-reading the catalog
            // fixes if the reply is a bare "unknown".
            let registered = self.tool_registry.registered_tool_names();
            let hint = crate::tools::did_you_mean_hint(name, registered.iter().map(String::as_str));
            return crate::tools::ToolOutput::text(format!("Unknown tool: '{name}'.{hint}"), true);
        };
        // Registered but not on this wire: a model that saw the schema on an earlier profile may
        // still ask for it after the session moved.
        if !self.tool_registry.is_carried(name) {
            return crate::tools::ToolOutput::text(crate::tools::not_carried_refusal(name), true);
        }

        self.run_admitted_call(
            &tool,
            call,
            loaded,
            attribution,
            cancellation,
            CallDoor::Dispatch,
        )
        .await
    }

    /// Run one call the loop has resolved to `tool`, through every door in order: the level read
    /// at the enforcement site, `admit_tool_call`, `admit_arguments`, the tool's own refusal at
    /// that level, the approval prompt, then the tool itself, and the schema advisory on the way
    /// back. The one sequence for inline dispatch and the checkpoint turn, so a door added here is
    /// a door both have; a copy per caller is a door one of them forgets.
    ///
    /// Nothing ahead of the tool's own `execute` awaits for a tool at the level of the checklist
    /// tools, and `execute_tool_calls` polls a message's calls in source order, which is what lets
    /// `SharedChecklist::in_turn_order` queue them in that order; an await added to this prelude
    /// would have to keep it.
    pub(super) async fn run_admitted_call(
        &self,
        tool: &Arc<dyn crate::tools::Tool>,
        call: ResolvedCall<'_>,
        loaded: &[String],
        attribution: &crate::provider::Attribution,
        cancellation: CancellationToken,
        door: CallDoor,
    ) -> crate::tools::ToolOutput {
        let ResolvedCall {
            tool_call_id,
            name,
            input,
            unparsed,
        } = call;
        // Ahead of every refusal, as Claude Code records a call once its tool is found, and ahead
        // of the run, so a sub-agent the call spawns names it too. Not the checkpoint's own
        // submission: that is the compaction handing back its summary, which Claude Code's
        // compaction writes as text and never lists.
        if !matches!(door, CallDoor::CheckpointSubmission) {
            self.note_tool_called(name);
        }
        // Read at the enforcement site, so a level cycled or a switch toggled during dispatch
        // means the next call rather than a snapshot captured earlier in the loop.
        let required = self
            .tool_registry
            .required_permission_for(name)
            .unwrap_or_else(|| tool.required_permission());
        let permission = self.cells.permission.get();
        let admission = match door {
            CallDoor::Dispatch | CallDoor::Checkpoint => crate::tools::admit_tool_call(
                name,
                required,
                permission,
                self.cells.permission.approvals(),
                tool.runs_outside_confinement(),
            ),
            CallDoor::CheckpointSubmission => crate::tools::Admission::Run,
        };
        if let crate::tools::Admission::Refuse(refusal) = admission {
            return *refusal;
        }
        // Scope the id across both dispatch paths, so a tool that has to correlate itself with the
        // client's view of this call (`agent_spawn`, routing its sub-agent's activity back into
        // the tool call already on screen) can read it without every other tool's signature
        // growing a parameter it ignores.
        //
        // The session id rides alongside for the same reason, so an MCP `tools/call` can name the
        // conversation it came from. `run_turn` populates `cells.session_id` before the tool loop,
        // so this is only `None` on paths that never established a session.
        let session_id = self.cells.session_id.get();
        let definition = tool.definition();
        let schema = &definition.parameters;

        let (input, detach) =
            match crate::tools::admit_arguments(&**tool, &definition, input, unparsed) {
                Ok(admitted) => admitted,
                Err(refusal) => return refusal,
            };
        let input = &input;
        // A checkpoint tool runs inline whatever the model asked: the flag is stripped like
        // everywhere else and not honored, since a task outliving the checkpoint would report
        // into a conversation the summary has replaced. A detaching call's tool has a start step,
        // `admit_arguments` having refused the flag on any other.
        let detachable = tool
            .as_detachable()
            .filter(|_| detach && matches!(door, CallDoor::Dispatch));
        let detach = detachable.is_some();

        // A call that cannot detach is refused here, ahead of the approval prompt: the session,
        // the ceiling and the slot are known now, so asking first would put a question whose answer
        // could not matter. What the tool's own start step refuses comes after approval, as it
        // would for the call inline.
        if detach && let Err(refusal) = self.admit_detach(session_id).await {
            return refusal;
        }

        if matches!(admission, crate::tools::Admission::Ask) {
            // The same rule for the tool's own doors: an approved call runs at the level, so a
            // refusal the level already decides (the shell with nothing to confine it, a write
            // outside the roots) is returned here rather than asked about and then made anyway.
            if let Some(refusal) = tool.refusal_at_level(permission, input).await {
                return refusal;
            }
            // Approval resolves *before* a detach, never inside it. A prompt surfacing minutes
            // after the turn that caused it, with nothing on screen to explain it, is worse than
            // the round trip it would save.
            //
            // A checkpoint's call was never announced: the compaction tells each call as it
            // finishes, under the id its caller resolved the call by. Its own submission is no
            // call the compaction tells, and no level asks about it.
            let prompted = match door {
                CallDoor::Dispatch => Some(PromptedCall::Announced(tool_call_id.to_string())),
                CallDoor::Checkpoint => Some(PromptedCall::Unannounced(tool_call_id.to_string())),
                CallDoor::CheckpointSubmission => None,
            };
            if let Some(denial) = self
                .request_approval(prompted, name, input, detach, &cancellation)
                .await
            {
                return denial;
            }
        }

        let mut output = if let Some(detachable) = detachable {
            self.start_background_call(
                tool,
                detachable,
                tool_call_id,
                name,
                input,
                session_id,
                attribution,
                cancellation,
            )
            .await
        } else {
            Self::run_tool(&**tool, input, crate::tools::ToolContext {
                session_id,
                tool_call_id: Some(tool_call_id.to_string()),
                prompt_id: attribution.prompt_id,
                frontend: Arc::clone(&self.cells.frontend),
                cancellation,
                task_id: None,
            })
            .await
        };
        if let Some(advisory) = self.schema_advisory(name, input, schema, loaded, !output.is_error)
        {
            output.push_note(advisory);
        }
        output
    }

    /// The advisory to append to this call's result when the arguments and the tool's advertised
    /// schema disagree, or `None`.
    ///
    /// Deferred tools are dispatchable whether or not the model loaded them (see
    /// [`Self::resolve_and_execute_tool`]), and a model that never loaded one has only ever seen
    /// the truncated one-line summary from `[Tool discovery]`, so a call that succeeds on a
    /// silently wrong default has no error to read.
    ///
    /// Emitted at most once per tool per process. The result stays in the conversation, so
    /// repeating it buys nothing and costs context on every subsequent call.
    ///
    /// `ran` gates only the *bookkeeping*: a call that never executed (an interactive permission
    /// denial, say) still gets the advisory, but must not spend the tool's one slot, or the retry
    /// after the user grants permission would be the silent call this exists to prevent.
    pub(super) fn schema_advisory(
        &self,
        name: &str,
        input: &serde_json::Value,
        schema: &serde_json::Value,
        loaded: &[String],
        ran: bool,
    ) -> Option<String> {
        let blind =
            self.tool_registry.is_deferred(name) && !loaded.iter().any(|entry| entry == name);
        let advisory = crate::tools::schema_disagreement(name, input, schema, blind)?;
        if !ran {
            return Some(advisory);
        }
        let first_time = crate::sync::lock(&self.schema_advisories_sent).insert(name.to_string());
        first_time.then_some(advisory)
    }

    /// Ask the user to approve one call the approvals switch submitted. `None` means run it;
    /// `Some` is the result to return instead.
    ///
    /// Split out of the dispatch path so approval can be settled *before* a `background` call
    /// detaches. Left inline, the prompt would surface minutes later with nothing on screen to
    /// explain what it belonged to.
    pub(super) async fn request_approval(
        &self,
        tool_call: Option<PromptedCall>,
        name: &str,
        input: &serde_json::Value,
        detach: bool,
        cancellation: &CancellationToken,
    ) -> Option<crate::tools::ToolOutput> {
        let primary_param = crate::tools::resolve_primary_param(name, input);
        let outcome = self
            .cells
            .frontend
            .request_permission(PermissionRequest {
                tool_name: name.to_string(),
                primary_param,
                input: approval_input(input, detach),
                cancellation: cancellation.clone(),
                tool_call,
                subagent_id: None,
            })
            .await;
        match outcome {
            PermissionOutcome::Allow => None,
            PermissionOutcome::Deny => Some(crate::tools::ToolOutput::text(
                "User denied tool execution.".to_string(),
                true,
            )),
            PermissionOutcome::Canceled => Some(crate::tools::ToolOutput::text(
                "Approval request was canceled.".to_string(),
                true,
            )),
        }
    }

    /// Whether a detached call can start at all: a session to report into, a ceiling above zero,
    /// and a free slot under it. `Ok` carries the session; `Err` is the result the model gets
    /// instead.
    ///
    /// One predicate for the two doors that ask it, the approval prompt and the start itself, so
    /// the user is never asked about a call that cannot detach. The slot check here is a
    /// preview: [`crate::background::BackgroundTasks::try_reserve`] is the atomic
    /// claim, and both refuse in the same words.
    pub(super) async fn admit_detach(
        &self,
        session_id: Option<Uuid>,
    ) -> std::result::Result<Uuid, crate::tools::ToolOutput> {
        let Some(session_id) = session_id else {
            return Err(crate::tools::ToolOutput::text(
                "Error: background calls need a session to report back into. Run this one \
                 normally, without `background`."
                    .to_string(),
                true,
            ));
        };
        if self.background_max_tasks == 0 {
            return Err(crate::tools::ToolOutput::text(
                "Error: background calls are disabled on this installation. Run this one normally, \
                 without `background`."
                    .to_string(),
                true,
            ));
        }
        if self.cells.background_tasks.running_count(session_id).await >= self.background_max_tasks
        {
            return Err(background_limit_refusal(self.background_max_tasks));
        }
        Ok(session_id)
    }

    /// Detach one call: record it, start it, spawn its run, and hand the model the started answer
    /// instead of a result.
    ///
    /// The row is written before the start step, so a process that dies anywhere after leaves a
    /// `running` row the sweep retires rather than work, or a worker's session, nobody recorded.
    /// The start step runs under the turn's token, so a turn interrupted while the call starts
    /// interrupts the call; that, a refusal from the start step, or a panic in it withdraws the
    /// row and frees the slot, and the call reads as an inline call's would. The run gets a
    /// **fresh** token rather than the turn's. Sharing the turn's would kill the task the instant
    /// the turn ended, which is the whole thing this exists to avoid; the cost is that Ctrl+C no
    /// longer reaches it, which is why `task_cancel` and the second-press escalation exist.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn start_background_call(
        &self,
        tool: &Arc<dyn crate::tools::Tool>,
        detachable: &dyn crate::tools::Detachable,
        tool_call_id: &str,
        name: &str,
        input: &serde_json::Value,
        session_id: Option<Uuid>,
        attribution: &crate::provider::Attribution,
        turn_cancellation: CancellationToken,
    ) -> crate::tools::ToolOutput {
        let prompt_id = attribution.prompt_id;
        let session_id = match self.admit_detach(session_id).await {
            Ok(session_id) => session_id,
            Err(refusal) => return refusal,
        };
        let label =
            crate::tools::resolve_primary_param(name, input).unwrap_or_else(|| name.to_string());
        let task = crate::store::background::BackgroundTask {
            id: Uuid::new_v4().to_string(),
            session_id,
            tool: name.to_string(),
            label,
            status: crate::store::background::TaskStatus::Running,
            outcome: None,
            scratchpad_entry: None,
            started_at: chrono::Utc::now(),
            finished_at: None,
            announced_at: None,
            delivered_at: None,
            subagent_id: None,
        };
        // Claim a slot before anything else. Atomic against the sibling calls in this same
        // assistant message, which `execute_tool_calls` dispatches concurrently: a
        // count-then-register would let four calls all read "zero running" and every one of
        // them start.
        let cancellation = CancellationToken::new();
        if !self
            .cells
            .background_tasks
            .try_reserve(
                task.id.clone(),
                session_id,
                cancellation.clone(),
                self.background_max_tasks,
            )
            .await
        {
            return background_limit_refusal(self.background_max_tasks);
        }
        if let Err(error) = self
            .store
            .background_store()
            .start_background_task(&task)
            .await
        {
            // Hand the slot back, or a failed start would shrink the ceiling for the session's
            // lifetime.
            self.cells.background_tasks.forget(&task.id).await;
            return crate::tools::ToolOutput::text(
                format!("Error: could not record the background task: {error}"),
                true,
            );
        }

        // Published for the frontend as well as passed to the tool: a delegated `fs/*` or
        // elicitation must race the token the work runs under, not the session's current turn.
        // See `crate::frontend::scope_call_cancellation`.
        let context = |cancellation: CancellationToken| crate::tools::ToolContext {
            session_id: Some(session_id),
            tool_call_id: Some(tool_call_id.to_string()),
            prompt_id,
            frontend: Arc::clone(&self.cells.frontend),
            cancellation,
            task_id: Some(task.id.clone()),
        };
        let begun = async {
            use std::ops::ControlFlow;
            let started = crate::frontend::scope_call_cancellation(
                turn_cancellation.clone(),
                detachable.start(input.clone(), &context(turn_cancellation.clone())),
            )
            .await;
            match started {
                Ok(crate::tools::Started::Running(running))
                    if !turn_cancellation.is_cancelled() =>
                {
                    ControlFlow::Continue(running)
                }
                // Stopped while it started, by a start step that did not wait on the stop: the
                // run sees it at once and settles what the start began, as it would inline.
                Ok(crate::tools::Started::Running(running)) => ControlFlow::Break(
                    crate::frontend::scope_call_cancellation(
                        turn_cancellation.clone(),
                        running.run(context(turn_cancellation.clone())),
                    )
                    .await,
                ),
                Ok(crate::tools::Started::Answered(output)) => ControlFlow::Break(Ok(output)),
                Err(error) => ControlFlow::Break(Err(error)),
            }
        };
        // A panic is caught here only to take the row and the slot with it, and carried on to the
        // batch, which words it as it words any tool's.
        use futures::FutureExt;
        let running = match std::panic::AssertUnwindSafe(begun).catch_unwind().await {
            Ok(std::ops::ControlFlow::Continue(running)) => running,
            Ok(std::ops::ControlFlow::Break(result)) => {
                self.withdraw_background_call(&task.id).await;
                return Self::tool_result(&**tool, result);
            }
            Err(panic) => {
                self.withdraw_background_call(&task.id).await;
                std::panic::resume_unwind(panic);
            }
        };
        let subagent = running.subagent;
        // A failed write is a warning: the model holds the sub-agent's id from the answer below
        // either way, and what goes without the link is the listings' word on which task runs it.
        if let Some(subagent) = subagent
            && let Err(error) = self
                .store
                .background_store()
                .record_task_subagent(&task.id, subagent)
                .await
        {
            let task_id = &task.id;
            tracing::warn!("failed to record the sub-agent of task {task_id}: {error}");
        }

        let requested_entry = crate::tools::scratchpad::requested_entry(input).map(str::to_string);
        let inherited_names = self.tool_registry.inherited_scratchpad_names();
        let context = context(cancellation.clone());
        let join = tokio::spawn({
            let tool = Arc::clone(tool);
            let store = self.store.clone();
            let tasks = self.cells.background_tasks.clone();
            let task_id = task.id.clone();
            let tool_name = task.tool.clone();
            async move {
                let run =
                    crate::frontend::scope_call_cancellation(cancellation, running.run(context));
                // A panic must not escape this task. Nothing awaits its `JoinHandle` outside
                // `--oneshot`, so an unwind here would skip both the outcome write and the slot
                // release: the agent would wait forever on a report that is never coming, and the
                // ceiling would be permanently one lower. Turning it into a `failed` outcome is
                // what the rest of the machinery already knows how to deliver.
                use futures::FutureExt;
                let output = match std::panic::AssertUnwindSafe(run).catch_unwind().await {
                    Ok(result) => Self::tool_result(&*tool, result),
                    Err(_) => {
                        tracing::error!("background task {task_id} panicked");
                        crate::tools::ToolOutput::text(
                            "The tool panicked while running in the background.".to_string(),
                            true,
                        )
                    }
                };

                let status = tasks.outcome_status(&task_id, output.is_error).await;
                let (outcome, scratchpad_entry, named_entry) = background_outcome(
                    &store,
                    session_id,
                    &task_id,
                    &tool_name,
                    output,
                    requested_entry.as_deref(),
                    &inherited_names,
                )
                .await;
                record_background_outcome(
                    &store,
                    &task_id,
                    status,
                    outcome,
                    scratchpad_entry,
                    named_entry,
                )
                .await;
                tasks.forget(&task_id).await;
            }
        });
        self.cells.background_tasks.attach(&task.id, join).await;

        let mut notice = crate::background::started_notice(task.short_id(), &task.label);
        if let Some(subagent) = subagent {
            notice.push_str(&format!(" It runs sub-agent {subagent}."));
            if self.tool_registry.get("agent_steer").is_some() {
                notice.push_str(" Use `agent_steer` to message it.");
            }
        }
        if self.tool_registry.get("task_list").is_some() {
            notice.push_str(" Use `task_list` for status.");
        }
        if self.tool_registry.get("task_cancel").is_some() {
            notice.push_str(" Use `task_cancel` to stop it.");
        }
        crate::tools::ToolOutput {
            frontend_metadata: Some(crate::frontend::ToolOutputMetadata::BackgroundStarted {
                task: task.id,
                subagent,
            }),
            ..crate::tools::ToolOutput::text(notice, false)
        }
    }

    /// Undo the record of a call whose start step did not start it: its row goes and its slot is
    /// freed, so the call leaves no task behind.
    async fn withdraw_background_call(&self, task_id: &str) {
        if let Err(error) = self
            .store
            .background_store()
            .withdraw_background_task(task_id)
            .await
        {
            tracing::warn!("failed to withdraw background task {task_id}: {error}");
        }
        self.cells.background_tasks.forget(task_id).await;
    }

    /// Invoke a tool and turn what it returned into the output the model reads; see
    /// [`Self::tool_result`].
    pub(super) async fn run_tool(
        tool: &dyn crate::tools::Tool,
        input: &serde_json::Value,
        context: crate::tools::ToolContext,
    ) -> crate::tools::ToolOutput {
        Self::tool_result(tool, tool.execute(input.clone(), context).await)
    }

    /// Turn what `tool` returned into the output the model reads, marking the tool's output
    /// ([`crate::tools::ToolOutput::is_output`]): a result it returned, or a failure it reports
    /// when it says its failures are its output. This is the one door a tool's result comes
    /// through, inline, from a start step that refused, and from a run in the background, so the
    /// mark is set here and nowhere else.
    pub(super) fn tool_result(
        tool: &dyn crate::tools::Tool,
        result: Result<crate::tools::ToolOutput>,
    ) -> crate::tools::ToolOutput {
        match result {
            Ok(output) => said_even_when_empty(&tool.definition().name, crate::tools::ToolOutput {
                is_output: !output.is_error || tool.reports_failure_as_output(),
                // A failed call changed nothing, whatever it reports, and every frontend's
                // completion event is built from what leaves here. How a command ended is a fact
                // about the failure itself and stays.
                frontend_metadata: match output.frontend_metadata {
                    Some(crate::frontend::ToolOutputMetadata::Diff { .. }) if output.is_error => {
                        None
                    }
                    metadata => metadata,
                },
                ..output
            }),
            Err(MekaError::Interrupted) => {
                crate::tools::ToolOutput::text("Tool execution interrupted.".to_string(), true)
            }
            Err(error) => crate::tools::ToolOutput::from_error(&error),
        }
    }
}

/// A result with nothing in it, said as one: an empty result at the end of a request reads to
/// the model as the end of its turn, and a wire may reject the empty text block it would be.
fn said_even_when_empty(name: &str, output: crate::tools::ToolOutput) -> crate::tools::ToolOutput {
    let empty = output.content.iter().all(|item| {
        matches!(item, crate::conversation::ToolResultContent::Text { text } if crate::text::is_blank(text))
    });
    if !empty {
        return output;
    }
    crate::tools::ToolOutput {
        content: vec![crate::conversation::ToolResultContent::Text {
            text: format!("{name} returned no output."),
        }],
        ..output
    }
}

/// The refusal for a session whose background tasks are at the ceiling, worded once for the
/// preview in [`Agent::admit_detach`] and the claim in [`Agent::start_background_call`].
fn background_limit_refusal(max_tasks: usize) -> crate::tools::ToolOutput {
    crate::tools::ToolOutput::text(
        format!(
            "Error: {max_tasks} background tasks are already running, which is the limit. Wait for \
             one to report, cancel one with `task_cancel`, or run this call without `background`."
        ),
        true,
    )
}

/// What a background run's `output` is recorded as: the text its report carries, the entry a spill
/// wrote, and the entry the call named with what goes in it, through the doors a round's result
/// takes. Its images go to scratchpad entries first, since a report carries text alone. Then the
/// entry the call named takes the output, as a round's result's does, and otherwise a text over
/// the report's bound spills whole, the report keeping its head, under the name the tool hinted.
/// The bound is the report's own, below any a tool sizes its result to, so it cuts a result the
/// tool sized to the context too; the note on a refused entry follows the head, not the tail.
async fn background_outcome(
    store: &crate::store::Store,
    session_id: Uuid,
    task_id: &str,
    tool_name: &str,
    mut output: crate::tools::ToolOutput,
    requested_entry: Option<&str>,
    inherited_names: &[String],
) -> (
    String,
    Option<String>,
    Option<crate::store::background::SavedOutput>,
) {
    use crate::tools::scratchpad::{Redirect, notes_of, redirect, save_images, spill};
    let wanted = output
        .spill_hint
        .name
        .clone()
        .unwrap_or_else(|| crate::background::spill_entry_name(task_id, tool_name));
    if let Err(error) = save_images(store, session_id, &mut output.content, &wanted).await {
        tracing::warn!(
            "background task {task_id}: failed to save an image to the scratchpad: {error}"
        );
    }
    let mut note = String::new();
    if let Some(name) = requested_entry.filter(|_| output.is_output) {
        match redirect(inherited_names, name, &output.content) {
            Some(Redirect::Save {
                name,
                text,
                reference,
            }) => {
                let entry = crate::store::background::SavedOutput {
                    session_id,
                    name,
                    content: text,
                };
                let notes = crate::conversation::ContentBlock::tool_result_text_content(&notes_of(
                    &output.content,
                ));
                let report = if notes.is_empty() {
                    reference
                } else {
                    format!("{reference}\n\n{notes}")
                };
                return (report, None, Some(entry));
            }
            Some(Redirect::Refused(refusal)) => note = refusal,
            None => {}
        }
    }
    let text = crate::conversation::ContentBlock::tool_result_text_content(&output.content);
    let (inline, spilled) = crate::background::split_outcome(&text);
    let entry = match spilled {
        None => None,
        Some(full) => match spill(store, session_id, &full, wanted).await {
            Ok(name) => Some(name),
            // Not fatal: the head still reaches the model, and losing the tail is far better
            // than losing the whole report.
            Err(error) => {
                tracing::warn!(
                    "background task {task_id}: failed to spill output to the scratchpad: {error}"
                );
                None
            }
        },
    };
    (inline + &note, entry, None)
}

/// How long a failed outcome write waits before its one retry: enough for the `SQLITE_BUSY` a
/// sibling process's write produces to clear, short enough not to hold the task's slot noticeably.
const OUTCOME_RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(250);

/// Record a finished task's outcome so its row is never left `running` with no task behind it.
///
/// A failed write is retried once, and a second failure marks the row `failed` with the error as
/// its outcome. Warning and moving on left the row `running`, which the next session open swept
/// to `interrupted`: the model was then told the work died, when it finished and merely went
/// unrecorded.
async fn record_background_outcome(
    store: &crate::store::Store,
    task_id: &str,
    status: crate::store::background::TaskStatus,
    outcome: String,
    scratchpad_entry: Option<String>,
    named_entry: Option<crate::store::background::SavedOutput>,
) {
    let background = store.background_store();
    let finish = |outcome: String, named_entry: Option<crate::store::background::SavedOutput>| {
        let background = background.clone();
        let scratchpad_entry = scratchpad_entry.clone();
        async move {
            match named_entry {
                Some(entry) => {
                    background
                        .finish_background_task_into_entry(task_id, status, outcome, entry)
                        .await
                }
                None => {
                    background
                        .finish_background_task(task_id, status, Some(outcome), scratchpad_entry)
                        .await
                }
            }
        }
    };
    let Err(error) = finish(outcome.clone(), named_entry.clone()).await else {
        return;
    };
    tracing::warn!("background task {task_id} finished but failed to record it; retrying: {error}");
    tokio::time::sleep(OUTCOME_RETRY_DELAY).await;
    let Err(error) = finish(outcome, named_entry).await else {
        return;
    };
    tracing::warn!(
        "background task {task_id} finished but failed to record it; marking it failed: {error}"
    );
    let failure = format!("The tool finished, but its result could not be recorded: {error}");
    if let Err(error) = background
        .finish_background_task(
            task_id,
            crate::store::background::TaskStatus::Failed,
            Some(failure),
            None,
        )
        .await
    {
        tracing::error!(
            "failed to mark background task {task_id} failed; it stays listed as running until the next \
             session opens: {error}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tests::{SendFileFixture, agent_with_registry_for_test, send_file_registry};

    /// A result with nothing in it, as an MCP server's empty `content` arrives, is said as one,
    /// with the failure flag it came with; a result with words in it is left alone.
    #[test]
    fn an_empty_result_says_it_returned_nothing() {
        let said = said_even_when_empty(
            "mcp__drive__sync",
            crate::tools::ToolOutput::text(String::new(), true),
        );
        assert!(said.is_error);
        assert!(matches!(
            said.content.as_slice(),
            [crate::conversation::ToolResultContent::Text { text }]
                if text == "mcp__drive__sync returned no output."
        ));
        let kept = said_even_when_empty(
            "file_read",
            crate::tools::ToolOutput::text("words".to_string(), false),
        );
        assert!(matches!(
            kept.content.as_slice(),
            [crate::conversation::ToolResultContent::Text { text }] if text == "words"
        ));
    }

    /// The prompt claims to show every argument the call was made with. `background` is taken out
    /// of the arguments before dispatch, so without this it would be the one argument a user could
    /// not see, and it is the one that decides whether the call keeps running after the turn ends.
    #[test]
    fn a_detaching_call_says_so_at_the_prompt() {
        let input = serde_json::json!({"command": "sleep 600"});
        assert_eq!(
            super::approval_input(&input, true),
            serde_json::json!({"command": "sleep 600", "background": true})
        );
        assert_eq!(super::approval_input(&input, false), input);
    }

    /// A non-object input has nowhere to put the flag, and inventing a shape for it would be worse
    /// than leaving it alone.
    #[test]
    fn a_non_object_input_is_left_alone() {
        let input = serde_json::json!("bare");
        assert_eq!(super::approval_input(&input, true), input);
    }

    /// The tool's output is marked as such and a word about the call is not: a result the tool
    /// returned, and a failure it reports when its failures are its output, as a command's exit
    /// is; neither a failure from a tool whose failed result is a complaint nor Meka's own
    /// refusal. The `scratchpad` redirect takes the former and leaves the latter inline, so a
    /// caller that redirected a refused call still reads why.
    #[tokio::test]
    async fn the_tools_output_is_told_apart_from_a_word_about_the_call() {
        use crate::provider::mock::MockProvider;

        struct Fixture(&'static str);

        #[async_trait::async_trait]
        impl crate::tools::Tool for Fixture {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: self.0.to_string(),
                    description: "fixture".to_string(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                    ..Default::default()
                }
            }

            fn required_permission(&self) -> crate::permission::Permission {
                crate::permission::Permission::Read
            }

            fn reports_failure_as_output(&self) -> bool {
                self.0 == "command"
            }

            async fn execute(
                &self,
                _input: serde_json::Value,
                _context: crate::tools::ToolContext,
            ) -> crate::error::Result<crate::tools::ToolOutput> {
                match self.0 {
                    "command" | "complaining" => {
                        Ok(crate::tools::ToolOutput::text("failed".to_string(), true))
                    }
                    "succeeding" => Ok(crate::tools::ToolOutput::text("done".to_string(), false)),
                    _ => Err(crate::error::MekaError::ToolExecution {
                        tool_name: self.0.to_string(),
                        message: "refused".to_string(),
                    }),
                }
            }
        }

        let registry = crate::tools::ToolRegistry::new();
        let names = ["command", "complaining", "succeeding", "refusing"];
        for name in names {
            registry
                .register(Arc::new(Fixture(name)))
                .expect("registration");
        }
        let (agent, _store) =
            agent_with_registry_for_test(Arc::new(MockProvider::from_rounds(vec![])), registry)
                .await;
        let message = crate::conversation::Message {
            role: crate::conversation::Role::Assistant,
            content: names
                .iter()
                .enumerate()
                .map(|(index, name)| crate::conversation::ContentBlock::ToolUse {
                    id: format!("call-{}", index + 1),
                    name: name.to_string(),
                    input: serde_json::json!({ "scratchpad": "probe" }),
                    unparsed: None,
                    provider_fields: None,
                })
                .collect(),
            provider_fields: None,
        };
        let (results, _, inline_results) = agent
            .execute_tool_calls(
                &message,
                &[],
                &crate::provider::Attribution::default(),
                CancellationToken::new(),
            )
            .await;
        assert_eq!(results.len(), 4);
        assert_eq!(
            inline_results,
            vec!["call-2".to_string(), "call-4".to_string()],
            "a complaint and a refusal stay inline; a command's failure and a result are output"
        );
    }

    /// A tool that panics is a failed result, not a dead turn: the assistant's `tool_use` is
    /// already in memory when the tools run, and a panic that unwinds the turn leaves the
    /// conversation ending on a call with no result, rejected by the provider on every later
    /// request until the session is evicted.
    #[tokio::test]
    async fn a_tool_that_panics_is_a_failed_result() {
        use crate::provider::mock::MockProvider;

        struct Panicking;

        #[async_trait::async_trait]
        impl crate::tools::Tool for Panicking {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "panicking".to_string(),
                    description: "fixture".to_string(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                    ..Default::default()
                }
            }

            fn required_permission(&self) -> crate::permission::Permission {
                crate::permission::Permission::Read
            }

            async fn execute(
                &self,
                _input: serde_json::Value,
                _context: crate::tools::ToolContext,
            ) -> crate::error::Result<crate::tools::ToolOutput> {
                panic!("the fixture panics")
            }
        }

        let registry = crate::tools::ToolRegistry::new();
        registry
            .register(Arc::new(Panicking))
            .expect("registration");
        let (agent, _store) =
            agent_with_registry_for_test(Arc::new(MockProvider::from_rounds(vec![])), registry)
                .await;
        let call = crate::conversation::Message {
            role: crate::conversation::Role::Assistant,
            content: vec![crate::conversation::ContentBlock::ToolUse {
                id: "call-1".to_string(),
                name: "panicking".to_string(),
                input: serde_json::json!({}),
                unparsed: None,
                provider_fields: None,
            }],
            provider_fields: None,
        };
        let (results, _durations, _) = agent
            .execute_tool_calls(
                &call,
                &[],
                &crate::provider::Attribution::default(),
                CancellationToken::new(),
            )
            .await;
        let crate::conversation::ContentBlock::ToolResult {
            content, is_error, ..
        } = &results[0]
        else {
            panic!("a result for the call: {results:?}");
        };
        assert!(is_error);
        let text = crate::conversation::ContentBlock::tool_result_text_content(content);
        assert!(text.contains("panicked"), "{text}");
    }

    /// A tool that runs outside any confinement Meka can apply is refused at `workspace`, for
    /// every requirement above it, not only for `Unrestricted`.
    ///
    /// Deleting the gate fails open, because `resolve_tool_permission`'s fallback for an
    /// unannotated MCP tool is `Unrestricted` and `Workspace.allows(Unrestricted)` is `true` by
    /// design. It is keyed on `is_within` rather than on a literal pair of rungs, so a comparison
    /// naming one rung cannot stand in for the order.
    ///
    /// The two controls matter as much as the refusals. A confinable tool with the same requirement
    /// must still dispatch, or the gate is just a permission check; and a requirement the level
    /// does cover must dispatch even when the tool is unconfinable.
    #[tokio::test]
    async fn an_unconfinable_tool_is_refused_at_workspace_for_every_requirement_above_it() {
        use crate::provider::mock::MockProvider;

        struct Fixture {
            name: &'static str,
            required: crate::permission::Permission,
            unconfinable: bool,
        }

        #[async_trait::async_trait]
        impl crate::tools::Tool for Fixture {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: self.name.to_string(),
                    description: "fixture".to_string(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                    ..Default::default()
                }
            }

            fn required_permission(&self) -> crate::permission::Permission {
                self.required
            }

            fn runs_outside_confinement(&self) -> bool {
                self.unconfinable
            }

            async fn execute(
                &self,
                _input: serde_json::Value,
                _context: crate::tools::ToolContext,
            ) -> crate::error::Result<crate::tools::ToolOutput> {
                Ok(crate::tools::ToolOutput::text(
                    "dispatched".to_string(),
                    false,
                ))
            }
        }

        let registry = crate::tools::ToolRegistry::new();
        for (name, required, unconfinable) in [
            (
                "mcp__x__unrestricted",
                crate::permission::Permission::Unrestricted,
                true,
            ),
            ("mcp__x__read", crate::permission::Permission::Read, true),
            (
                "builtin_unrestricted",
                crate::permission::Permission::Unrestricted,
                false,
            ),
        ] {
            registry
                .register(Arc::new(Fixture {
                    name,
                    required,
                    unconfinable,
                }))
                .expect("registration");
        }

        let (agent, _store) =
            agent_with_registry_for_test(Arc::new(MockProvider::from_rounds(vec![])), registry)
                .await;
        agent
            .cells
            .permission
            .try_set(crate::permission::Permission::Workspace)
            .expect("workspace is enabled in this fixture");

        for (name, refused) in [
            ("mcp__x__unrestricted", true),
            // Within the level, so the gate has no business firing.
            ("mcp__x__read", false),
            // Above the level but confinable: it has its own door downstream.
            ("builtin_unrestricted", false),
        ] {
            let output = agent
                .resolve_and_execute_tool(
                    ResolvedCall {
                        tool_call_id: "call-1",
                        name,
                        input: &serde_json::json!({}),
                        unparsed: None,
                    },
                    &[],
                    &crate::provider::Attribution::default(),
                    CancellationToken::new(),
                )
                .await;
            let body = output.text_content();
            if refused {
                assert!(
                    output.is_error,
                    "{name} must be refused at workspace: {body}"
                );
                assert!(
                    body.contains("does not sandbox"),
                    "{name}'s refusal must say why: {body}"
                );
            } else {
                assert!(
                    !body.contains("does not sandbox"),
                    "{name} must not be refused by the confinement gate: {body}"
                );
            }
        }
    }

    /// A call whose arguments were not an object is refused with the reason, at the one door every
    /// call passes, and the tool never runs on arguments the model did not give.
    #[tokio::test]
    async fn a_call_whose_arguments_were_not_an_object_is_refused_before_the_tool_runs() {
        use crate::provider::mock::MockProvider;

        struct Fixture(Arc<std::sync::atomic::AtomicBool>);

        #[async_trait::async_trait]
        impl crate::tools::Tool for Fixture {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "read_thing".to_string(),
                    description: "fixture".to_string(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                    ..Default::default()
                }
            }

            fn required_permission(&self) -> crate::permission::Permission {
                crate::permission::Permission::Read
            }

            async fn execute(
                &self,
                _input: serde_json::Value,
                _context: crate::tools::ToolContext,
            ) -> crate::error::Result<crate::tools::ToolOutput> {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(crate::tools::ToolOutput::text("ran".to_string(), false))
            }
        }

        let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let registry = crate::tools::ToolRegistry::new();
        registry
            .register(Arc::new(Fixture(Arc::clone(&ran))))
            .expect("registration");
        let (agent, _store) =
            agent_with_registry_for_test(Arc::new(MockProvider::from_rounds(vec![])), registry)
                .await;
        let unparsed = crate::conversation::UnparsedArguments {
            text: Some("{\"path\": ".to_string()),
            error: "invalid JSON arguments: EOF while parsing".to_string(),
        };
        let output = agent
            .resolve_and_execute_tool(
                ResolvedCall {
                    tool_call_id: "call-1",
                    name: "read_thing",
                    input: &serde_json::json!({}),
                    unparsed: Some(&unparsed),
                },
                &[],
                &crate::provider::Attribution::default(),
                CancellationToken::new(),
            )
            .await;
        assert!(output.is_error);
        assert_eq!(
            output.text_content(),
            "Tool call refused: invalid JSON arguments: EOF while parsing"
        );
        assert!(
            !ran.load(std::sync::atomic::Ordering::SeqCst),
            "the tool never ran"
        );
    }

    /// A detached call the background machinery would refuse is refused before anyone is asked
    /// to approve it: with the switch on and the ceiling at zero, the prompt would put a question
    /// to the user whose answer could not matter.
    #[tokio::test]
    async fn a_detached_call_that_cannot_start_is_refused_before_approval_is_asked() {
        use crate::provider::mock::MockProvider;

        struct Fixture;

        #[async_trait::async_trait]
        impl crate::tools::Tool for Fixture {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "write_thing".to_string(),
                    description: "fixture".to_string(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                    ..Default::default()
                }
            }

            fn required_permission(&self) -> crate::permission::Permission {
                crate::permission::Permission::Workspace
            }

            async fn execute(
                &self,
                _input: serde_json::Value,
                _context: crate::tools::ToolContext,
            ) -> crate::error::Result<crate::tools::ToolOutput> {
                Ok(crate::tools::ToolOutput::text("ran".to_string(), false))
            }
        }

        let registry = crate::tools::ToolRegistry::new();
        registry
            .register(Arc::new(crate::tools::DetachableForTest(Arc::new(Fixture))))
            .expect("registration");
        // Deliberately no `enable_background`: the ceiling is zero, as on a default installation.
        let (mut agent, _store) =
            agent_with_registry_for_test(Arc::new(MockProvider::from_rounds(vec![])), registry)
                .await;
        let frontend = Arc::new(crate::frontend::testing::RecordingFrontend::new());
        agent.cells.frontend = Arc::clone(&frontend) as Arc<dyn crate::frontend::Frontend>;
        // Above the level (`read`) with the switch on, so the call would be asked about.
        agent.cells.permission.set_approvals(true);
        agent.cells.session_id.set(Uuid::new_v4());

        let output = agent
            .resolve_and_execute_tool(
                ResolvedCall {
                    tool_call_id: "call-1",
                    name: "write_thing",
                    input: &serde_json::json!({"background": true}),
                    unparsed: None,
                },
                &[],
                &crate::provider::Attribution::default(),
                CancellationToken::new(),
            )
            .await;
        let body = output.text_content();
        assert!(output.is_error, "{body}");
        assert!(body.contains("background calls are disabled"), "{body}");
        assert!(
            frontend.permission_requests().is_empty(),
            "nobody is asked to approve a call that is refused anyway: {:?}",
            frontend.permission_requests()
        );
    }

    /// A failed call changed nothing, whatever it reports: the change a tool attached to a
    /// failure is dropped at the result door, so no frontend draws it; a success keeps its own,
    /// and a failed command keeps how it ended, which is a fact about the failure.
    #[tokio::test]
    async fn a_failed_call_carries_no_change() {
        struct Fixture {
            fails: bool,
        }

        #[async_trait::async_trait]
        impl crate::tools::Tool for Fixture {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "change_thing".to_string(),
                    description: "fixture".to_string(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                    ..Default::default()
                }
            }

            fn required_permission(&self) -> crate::permission::Permission {
                crate::permission::Permission::Read
            }

            async fn execute(
                &self,
                _input: serde_json::Value,
                _context: crate::tools::ToolContext,
            ) -> crate::error::Result<crate::tools::ToolOutput> {
                Ok(
                    crate::tools::ToolOutput::text("ran".to_string(), self.fails).with_metadata(
                        crate::frontend::ToolOutputMetadata::Diff {
                            path: None,
                            old_text: Some("before".to_string()),
                            new_text: Some("after".to_string()),
                        },
                    ),
                )
            }
        }

        let context = || crate::tools::ToolContext::for_test(CancellationToken::new());
        let failed =
            Agent::run_tool(&Fixture { fails: true }, &serde_json::json!({}), context()).await;
        assert!(failed.is_error);
        assert!(
            failed.frontend_metadata.is_none(),
            "{:?}",
            failed.frontend_metadata
        );
        let succeeded =
            Agent::run_tool(&Fixture { fails: false }, &serde_json::json!({}), context()).await;
        assert!(succeeded.frontend_metadata.is_some());

        struct Exited;

        #[async_trait::async_trait]
        impl crate::tools::Tool for Exited {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "exit_thing".to_string(),
                    description: "fixture".to_string(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                    ..Default::default()
                }
            }

            fn required_permission(&self) -> crate::permission::Permission {
                crate::permission::Permission::Read
            }

            async fn execute(
                &self,
                _input: serde_json::Value,
                _context: crate::tools::ToolContext,
            ) -> crate::error::Result<crate::tools::ToolOutput> {
                Ok(
                    crate::tools::ToolOutput::text("exit 7".to_string(), true).with_metadata(
                        crate::frontend::ToolOutputMetadata::CommandExit {
                            exit_code: Some(7),
                            signal: None,
                        },
                    ),
                )
            }
        }

        let exited = Agent::run_tool(&Exited, &serde_json::json!({}), context()).await;
        assert!(
            matches!(
                exited.frontend_metadata,
                Some(crate::frontend::ToolOutputMetadata::CommandExit {
                    exit_code: Some(7),
                    ..
                })
            ),
            "{:?}",
            exited.frontend_metadata
        );
    }

    /// With `[shell].sandbox = false`, `shell_execute` at `read` is refused however the user
    /// answers: an approved call runs at the level, and nothing can confine it there. Asking first
    /// put a question to the user whose yes could not matter, the way `admit_detach` already
    /// refuses a detached call ahead of the prompt. The refusal reaches the model unasked, in the
    /// words `execute` would have used.
    #[tokio::test]
    async fn a_call_the_level_alone_refuses_is_not_asked_about() {
        use crate::provider::mock::MockProvider;

        let registry = crate::tools::ToolRegistry::new();
        registry
            .register(Arc::new(crate::tools::shell::ExecuteCommandTool {
                scope: crate::workspace::WriteScope::unconfined(),
                #[cfg(windows)]
                windows_grants: Arc::new(crate::sandbox::windows::WindowsGrants::default()),
                sandbox_capability: crate::sandbox::SandboxCapability::Unavailable,
                sandbox_backend: crate::config::SandboxBackend::Landlock,
                backend_probe: crate::sandbox::BackendProbe::Missing {
                    reason: "test fixture".to_string(),
                },
                sandbox_enabled: false,
                site: crate::session::ToolSite::for_test()
                    .with_permission(crate::permission::SharedPermission::new(
                        crate::permission::Permission::Read,
                        crate::permission::EnabledPermissions::ALL,
                    ))
                    .with_cwd(crate::workspace::cwd_for_test()),
            }))
            .expect("registration");
        let (mut agent, _store) =
            agent_with_registry_for_test(Arc::new(MockProvider::from_rounds(vec![])), registry)
                .await;
        let frontend = Arc::new(crate::frontend::testing::RecordingFrontend::new());
        agent.cells.frontend = Arc::clone(&frontend) as Arc<dyn crate::frontend::Frontend>;
        // `read` with the switch on: the tool requires `unrestricted`, so this call is asked about
        // unless something ahead of the prompt has the answer.
        agent.cells.permission.set_approvals(true);

        let output = agent
            .resolve_and_execute_tool(
                ResolvedCall {
                    tool_call_id: "call-1",
                    name: "shell_execute",
                    input: &serde_json::json!({"command": "true"}),
                    unparsed: None,
                },
                &[],
                &crate::provider::Attribution::default(),
                CancellationToken::new(),
            )
            .await;
        let body = output.text_content();
        assert!(output.is_error, "{body}");
        assert!(body.contains("`[shell].sandbox = false`"), "{body}");
        assert!(
            frontend.permission_requests().is_empty(),
            "nobody is asked to approve a call the level refuses anyway: {:?}",
            frontend.permission_requests()
        );
    }

    /// A call that never executed must not spend the tool's one advisory. Otherwise a permission
    /// denial swallows the hint, and the retry after the user grants permission is exactly the
    /// silent call this machinery exists to prevent.
    #[tokio::test]
    async fn a_call_that_did_not_run_keeps_its_advisory_slot() {
        use crate::provider::mock::MockProvider;

        let (agent, _store) = agent_with_registry_for_test(
            Arc::new(MockProvider::from_rounds(vec![])),
            send_file_registry(),
        )
        .await;
        let name = "mcp__bridge__send_file";
        let schema = crate::tools::Tool::definition(&SendFileFixture).parameters;
        let input = serde_json::json!({"path": "/tmp/a.png"});

        let denied = agent.schema_advisory(name, &input, &schema, &[], false);
        assert!(denied.is_some(), "a denied call is still told");

        let retried = agent.schema_advisory(name, &input, &schema, &[], true);
        assert!(retried.is_some(), "the denial must not have spent the slot");

        let third = agent.schema_advisory(name, &input, &schema, &[], true);
        assert!(third.is_none(), "but a call that ran spends it");
    }

    /// Only a call that ran and succeeded spends the tool's one advisory. A call that ran and
    /// failed is about to be retried, and a retry on silently defaulted arguments is the call the
    /// advisory exists to prevent.
    #[tokio::test]
    async fn a_failed_call_keeps_the_advisory_slot_and_a_successful_one_spends_it() {
        use crate::provider::mock::MockProvider;

        async fn send(agent: &Agent, path: &str) -> crate::tools::ToolOutput {
            agent
                .resolve_and_execute_tool(
                    ResolvedCall {
                        tool_call_id: "call-1",
                        name: "mcp__bridge__send_file",
                        input: &serde_json::json!({"path": path}),
                        unparsed: None,
                    },
                    &[],
                    &crate::provider::Attribution::default(),
                    CancellationToken::new(),
                )
                .await
        }
        const ADVISORY: &str = "Called without loading its schema";

        let (agent, _store) = agent_with_registry_for_test(
            Arc::new(MockProvider::from_rounds(vec![])),
            send_file_registry(),
        )
        .await;

        let failed = send(&agent, "/missing.png").await;
        assert!(failed.is_error, "{}", failed.text_content());
        assert!(
            failed.text_content().contains(ADVISORY),
            "{}",
            failed.text_content()
        );
        let failed_again = send(&agent, "/missing.png").await;
        assert!(
            failed_again.text_content().contains(ADVISORY),
            "a failed call must not have spent the slot: {}",
            failed_again.text_content()
        );

        let succeeded = send(&agent, "/tmp/a.png").await;
        assert!(!succeeded.is_error, "{}", succeeded.text_content());
        assert!(
            succeeded.text_content().contains(ADVISORY),
            "the first call that succeeds is still told: {}",
            succeeded.text_content()
        );
        let again = send(&agent, "/tmp/a.png").await;
        assert!(
            !again.text_content().contains(ADVISORY),
            "and it spends the slot: {}",
            again.text_content()
        );
    }

    /// The request after a round reports how long each of the round's calls took, in dispatch
    /// order, which is only knowable here.
    #[tokio::test]
    async fn a_round_reports_the_duration_of_each_call_in_dispatch_order() {
        use crate::provider::mock::MockProvider;

        let (agent, store) = agent_with_registry_for_test(
            Arc::new(MockProvider::from_rounds(vec![])),
            crate::tools::ToolRegistry::new(),
        )
        .await;
        agent
            .tool_registry
            .register(Arc::new(crate::tools::checklist::ChecklistAddTool {
                checklist: agent.cells.checklist.clone(),
                store: store.clone(),
                site: agent.cells.site(),
            }))
            .expect("register");
        let message = Message {
            role: crate::conversation::Role::Assistant,
            content: vec![
                ContentBlock::ToolUse {
                    id: "call-1".to_string(),
                    name: "checklist_add".to_string(),
                    input: serde_json::json!({"items": ["first"]}),
                    unparsed: None,
                    provider_fields: None,
                },
                ContentBlock::ToolUse {
                    id: "call-2".to_string(),
                    name: "no_such_tool".to_string(),
                    input: serde_json::json!({}),
                    unparsed: None,
                    provider_fields: None,
                },
            ],
            provider_fields: None,
        };
        let (results, durations, _) = agent
            .execute_tool_calls(
                &message,
                &[],
                &crate::provider::Attribution::default(),
                CancellationToken::new(),
            )
            .await;
        assert_eq!(results.len(), 2);
        let names: Vec<&str> = durations
            .iter()
            .map(|duration| duration.name.as_str())
            .collect();
        assert_eq!(names, vec!["checklist_add", "no_such_tool"]);
    }

    /// The calls of one assistant message run concurrently, and an edit that names a task awaits
    /// the store while one that does not finishes at once; the replay applies them in source
    /// order, so the live list has to be built in source order too.
    #[tokio::test]
    async fn two_edits_in_one_message_land_in_the_order_the_model_wrote_them() {
        use crate::{provider::mock::MockProvider, store::background::TaskStatus};

        let (agent, store) = agent_with_registry_for_test(
            Arc::new(MockProvider::from_rounds(vec![])),
            crate::tools::ToolRegistry::new(),
        )
        .await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("a session");
        agent.cells.session_id.set(session_id);
        store
            .background_store()
            .start_background_task(&crate::store::background::BackgroundTask {
                id: "1a2b3c4d-0000-4000-8000-000000000000".to_string(),
                session_id,
                tool: "shell_execute".to_string(),
                label: "make".to_string(),
                status: TaskStatus::Running,
                outcome: None,
                scratchpad_entry: None,
                started_at: chrono::Utc::now(),
                finished_at: None,
                announced_at: None,
                delivered_at: None,
                subagent_id: None,
            })
            .await
            .expect("a task");
        agent
            .tool_registry
            .register(Arc::new(crate::tools::checklist::ChecklistAddTool {
                checklist: agent.cells.checklist.clone(),
                store: store.clone(),
                site: agent.cells.site(),
            }))
            .expect("registration");
        agent
            .tool_registry
            .register(Arc::new(crate::tools::checklist::ChecklistEditTool {
                checklist: agent.cells.checklist.clone(),
                store: store.clone(),
                site: agent.cells.site(),
            }))
            .expect("registration");
        agent.cells.checklist.update(|state| {
            state.add(vec![crate::checklist::NewItem {
                text: "build".to_string(),
                status: crate::checklist::ChecklistStatus::Pending,
            }]);
        });

        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::ToolUse {
                    id: "edit-1".to_string(),
                    name: "checklist_edit".to_string(),
                    input: serde_json::json!({
                        "id": 1, "status": "deferred", "reason": "building", "task": "1a2b3c4d"
                    }),
                    unparsed: None,
                    provider_fields: None,
                },
                ContentBlock::ToolUse {
                    id: "edit-2".to_string(),
                    name: "checklist_edit".to_string(),
                    input: serde_json::json!({"id": 1, "status": "in_progress"}),
                    unparsed: None,
                    provider_fields: None,
                },
            ],
            provider_fields: None,
        };
        let (results, ..) = agent
            .execute_tool_calls(
                &message,
                &[],
                &crate::provider::Attribution::default(),
                CancellationToken::new(),
            )
            .await;
        assert!(
            results
                .iter()
                .all(|block| matches!(block, ContentBlock::ToolResult {
                    is_error: false,
                    ..
                })),
            "both edits apply: {results:?}"
        );
        let item = &agent.cells.checklist.get().items[0];
        assert_eq!(
            item.status,
            crate::checklist::ChecklistStatus::InProgress,
            "the second edit in the message is the last word, as the replay will read it"
        );
        assert_eq!(item.task, None);
    }

    /// Every result records with it what a screen draws of the call: the code a command exited
    /// with, the change a call made, and the list a checklist call's batch left, so a resumed
    /// session draws the call as the live run did.
    #[tokio::test]
    async fn a_result_records_what_a_screen_draws_of_it() {
        use crate::{conversation::ResultDisplay, provider::mock::MockProvider};

        struct Reporting {
            name: &'static str,
            fails: bool,
            metadata: crate::frontend::ToolOutputMetadata,
        }

        #[async_trait::async_trait]
        impl crate::tools::Tool for Reporting {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: self.name.to_string(),
                    description: "fixture".to_string(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                    ..Default::default()
                }
            }

            fn required_permission(&self) -> crate::permission::Permission {
                crate::permission::Permission::Read
            }

            async fn execute(
                &self,
                _input: serde_json::Value,
                _context: crate::tools::ToolContext,
            ) -> crate::error::Result<crate::tools::ToolOutput> {
                Ok(
                    crate::tools::ToolOutput::text("out".to_string(), self.fails)
                        .with_metadata(self.metadata.clone()),
                )
            }
        }

        let (agent, store) = agent_with_registry_for_test(
            Arc::new(MockProvider::from_rounds(vec![])),
            crate::tools::ToolRegistry::new(),
        )
        .await;
        agent
            .tool_registry
            .register(Arc::new(Reporting {
                name: "exits",
                fails: true,
                metadata: crate::frontend::ToolOutputMetadata::CommandExit {
                    exit_code: Some(3),
                    signal: None,
                },
            }))
            .expect("registration");
        agent
            .tool_registry
            .register(Arc::new(Reporting {
                name: "changes",
                fails: false,
                metadata: crate::frontend::ToolOutputMetadata::Diff {
                    path: None,
                    old_text: Some("a\n".to_string()),
                    new_text: Some("b\n".to_string()),
                },
            }))
            .expect("registration");
        agent
            .tool_registry
            .register(Arc::new(crate::tools::checklist::ChecklistAddTool {
                checklist: agent.cells.checklist.clone(),
                store: store.clone(),
                site: agent.cells.site(),
            }))
            .expect("registration");
        let calls = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::ToolUse {
                    id: "exits-1".to_string(),
                    name: "exits".to_string(),
                    input: serde_json::json!({}),
                    unparsed: None,
                    provider_fields: None,
                },
                ContentBlock::ToolUse {
                    id: "changes-1".to_string(),
                    name: "changes".to_string(),
                    input: serde_json::json!({}),
                    unparsed: None,
                    provider_fields: None,
                },
                ContentBlock::ToolUse {
                    id: "add-1".to_string(),
                    name: "checklist_add".to_string(),
                    input: serde_json::json!({"items": ["first"]}),
                    unparsed: None,
                    provider_fields: None,
                },
            ],
            provider_fields: None,
        };
        let (results, ..) = agent
            .execute_tool_calls(
                &calls,
                &[],
                &crate::provider::Attribution::default(),
                CancellationToken::new(),
            )
            .await;
        let displays: Vec<Option<ResultDisplay>> = results
            .into_iter()
            .map(|block| match block {
                ContentBlock::ToolResult { display, .. } => display,
                other => panic!("a result: {other:?}"),
            })
            .collect();
        assert_eq!(
            displays[0],
            Some(ResultDisplay {
                exit_code: Some(3),
                ..Default::default()
            })
        );
        assert_eq!(
            displays[1],
            Some(ResultDisplay {
                change: Some(crate::conversation::Change::between(
                    Some("a\n"),
                    Some("b\n")
                )),
                ..Default::default()
            })
        );
        assert_eq!(
            displays[2]
                .as_ref()
                .and_then(|display| display.checklist.as_ref())
                .map(|items| items
                    .iter()
                    .map(|item| item.text.as_str())
                    .collect::<Vec<_>>()),
            Some(vec!["first"])
        );
    }

    /// One `ChecklistUpdated` per change. A read changes nothing and announces nothing; an edit
    /// that empties the list announces the empty list, which is what lets a surface clear it.
    #[tokio::test]
    async fn an_unchanged_checklist_is_not_re_announced_and_an_emptied_one_is() {
        use crate::provider::mock::MockProvider;

        let (mut agent, store) = agent_with_registry_for_test(
            Arc::new(MockProvider::from_rounds(vec![])),
            crate::tools::ToolRegistry::new(),
        )
        .await;
        agent
            .tool_registry
            .register(Arc::new(crate::tools::checklist::ChecklistAddTool {
                checklist: agent.cells.checklist.clone(),
                store: store.clone(),
                site: agent.cells.site(),
            }))
            .expect("registration");
        agent
            .tool_registry
            .register(Arc::new(crate::tools::checklist::ChecklistReadTool {
                checklist: agent.cells.checklist.clone(),
                store: store.clone(),
                site: agent.cells.site(),
            }))
            .expect("registration");
        agent
            .tool_registry
            .register(Arc::new(crate::tools::checklist::ChecklistEditTool {
                checklist: agent.cells.checklist.clone(),
                store: store.clone(),
                site: crate::session::ToolSite::for_test(),
            }))
            .expect("registration");
        let frontend = Arc::new(crate::frontend::testing::RecordingFrontend::new());
        agent.cells.frontend = Arc::clone(&frontend) as Arc<dyn crate::frontend::Frontend>;

        let call = |name: &str, input: serde_json::Value| Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "checklist-1".to_string(),
                name: name.to_string(),
                input,
                unparsed: None,
                provider_fields: None,
            }],
            provider_fields: None,
        };
        let rendered = || {
            frontend
                .events()
                .iter()
                .filter(|event| matches!(event, FrontendEvent::ChecklistUpdated { .. }))
                .count()
        };

        agent
            .execute_tool_calls(
                &call("checklist_add", serde_json::json!({"items": ["first"]})),
                &[],
                &crate::provider::Attribution::default(),
                CancellationToken::new(),
            )
            .await;
        assert_eq!(rendered(), 1, "a new item is announced once");
        agent
            .execute_tool_calls(
                &call("checklist_read", serde_json::json!({})),
                &[],
                &crate::provider::Attribution::default(),
                CancellationToken::new(),
            )
            .await;
        assert_eq!(rendered(), 1, "a read announces nothing");
        agent
            .execute_tool_calls(
                &call(
                    "checklist_edit",
                    serde_json::json!({"id": 1, "status": "completed"}),
                ),
                &[],
                &crate::provider::Attribution::default(),
                CancellationToken::new(),
            )
            .await;
        assert_eq!(
            rendered(),
            2,
            "the emptied list is announced, so a surface can clear it"
        );
        assert!(matches!(
            frontend.events().last(),
            Some(FrontendEvent::ChecklistUpdated { items }) if items.is_empty()
        ));
    }

    /// What a test asks a [`StartFixture`]'s start step to do.
    #[derive(Clone, Copy)]
    enum StartAs {
        /// Hand back a run driving this sub-agent, or none.
        Running(Option<Uuid>),
        /// Refuse the call.
        Refused,
        /// Answer the call without running anything.
        Answered,
        /// Wait until the call is stopped, as a start step mid-request does.
        Stopped,
        /// Panic.
        Panics,
    }

    /// A tool whose start step does what [`StartAs`] says.
    struct StartFixture(StartAs);

    #[async_trait::async_trait]
    impl crate::tools::Tool for StartFixture {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "start_fixture".to_string(),
                description: "fixture".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
                ..Default::default()
            }
        }

        fn required_permission(&self) -> crate::permission::Permission {
            crate::permission::Permission::Read
        }

        fn as_detachable(&self) -> Option<&dyn crate::tools::Detachable> {
            Some(self)
        }

        async fn execute(
            &self,
            input: serde_json::Value,
            context: crate::tools::ToolContext,
        ) -> crate::error::Result<crate::tools::ToolOutput> {
            crate::tools::start_and_run(self, input, context).await
        }
    }

    #[async_trait::async_trait]
    impl crate::tools::Detachable for StartFixture {
        async fn start(
            &self,
            _input: serde_json::Value,
            context: &crate::tools::ToolContext,
        ) -> crate::error::Result<crate::tools::Started> {
            match self.0 {
                StartAs::Running(subagent) => Ok(crate::tools::Started::running(
                    subagent,
                    |context| async move {
                        if context.cancellation.is_cancelled() {
                            return Err(MekaError::Interrupted);
                        }
                        Ok(crate::tools::ToolOutput::text("done".to_string(), false))
                    },
                )),
                StartAs::Refused => Err(MekaError::ToolExecution {
                    tool_name: "start_fixture".to_string(),
                    message: "refused at the start".to_string(),
                }),
                StartAs::Answered => Ok(crate::tools::Started::Answered(
                    crate::tools::ToolOutput::text("refused at the start".to_string(), true),
                )),
                StartAs::Stopped => {
                    context.cancellation.cancelled().await;
                    Err(MekaError::Interrupted)
                }
                StartAs::Panics => panic!("the start step broke"),
            }
        }
    }

    /// An agent holding a [`StartFixture`] that starts as `start`, with background calls on and a
    /// session of its own.
    async fn agent_with_start(start: StartAs) -> (Agent, crate::store::Store, Uuid) {
        use crate::provider::mock::MockProvider;

        let registry = crate::tools::ToolRegistry::new();
        registry
            .register(Arc::new(StartFixture(start)))
            .expect("registration");
        let (mut agent, store) =
            agent_with_registry_for_test(Arc::new(MockProvider::from_rounds(vec![])), registry)
                .await;
        agent.enable_background_for_test(1);
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("a session");
        agent.cells.session_id.set(session_id);
        (agent, store, session_id)
    }

    /// A detached call answers with the sub-agent its start step drives, records it with the
    /// result for a screen to draw, and its task's row links the sub-agent by the time the answer
    /// is given, so a model or a client acting on the answer finds both.
    #[tokio::test]
    async fn a_detached_call_names_the_sub_agent_its_start_step_drives() {
        let worker = Uuid::new_v4();
        let (agent, store, session_id) = agent_with_start(StartAs::Running(Some(worker))).await;

        let output = agent
            .resolve_and_execute_tool(
                ResolvedCall {
                    tool_call_id: "call-1",
                    name: "start_fixture",
                    input: &serde_json::json!({"background": true}),
                    unparsed: None,
                },
                &[],
                &crate::provider::Attribution::default(),
                CancellationToken::new(),
            )
            .await;

        let body = output.text_content();
        assert!(!output.is_error, "{body}");
        assert!(
            body.starts_with("Started in the background as task ")
                && body.contains(&format!("It runs sub-agent {worker}.")),
            "{body}"
        );
        let rows = store
            .background_store()
            .list_background_tasks(session_id)
            .await
            .expect("tasks");
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].subagent_id,
            Some(worker),
            "linked before the answer"
        );
        let Some(metadata) = &output.frontend_metadata else {
            panic!("the answer records the start: {body}");
        };
        assert_eq!(
            metadata.display().background,
            Some(crate::conversation::BackgroundStart {
                task: rows[0].id.clone(),
                subagent: Some(worker),
            })
        );
    }

    /// A call its start step refuses, by an error or by an answer, reads as an inline call's
    /// refusal would and leaves no task: its row is withdrawn and its slot freed, so the next call
    /// can take the one slot there is.
    #[tokio::test]
    async fn a_call_its_start_step_refuses_leaves_no_task() {
        for start in [StartAs::Refused, StartAs::Answered] {
            let (agent, store, session_id) = agent_with_start(start).await;
            let input = serde_json::json!({"background": true});
            let attribution = crate::provider::Attribution::default();

            for attempt in 0..2 {
                let output = agent
                    .resolve_and_execute_tool(
                        ResolvedCall {
                            tool_call_id: "call-1",
                            name: "start_fixture",
                            input: &input,
                            unparsed: None,
                        },
                        &[],
                        &attribution,
                        CancellationToken::new(),
                    )
                    .await;
                let body = output.text_content();
                assert!(output.is_error, "{body}");
                assert!(
                    body.contains("refused at the start"),
                    "attempt {attempt}, not a ceiling refusal: {body}"
                );
            }
            assert!(
                store
                    .background_store()
                    .list_background_tasks(session_id)
                    .await
                    .expect("tasks")
                    .is_empty()
            );
            assert_eq!(
                agent.cells.background_tasks.running_count(session_id).await,
                0
            );
        }
    }

    /// A start step that panics takes its row and its slot with it, and the panic goes on to the
    /// batch that words it, as an inline tool's does.
    #[tokio::test]
    async fn a_start_step_that_panics_leaves_no_task() {
        use futures::FutureExt;

        let (agent, store, session_id) = agent_with_start(StartAs::Panics).await;
        let input = serde_json::json!({"background": true});
        let attribution = crate::provider::Attribution::default();
        let call = agent.resolve_and_execute_tool(
            ResolvedCall {
                tool_call_id: "call-1",
                name: "start_fixture",
                input: &input,
                unparsed: None,
            },
            &[],
            &attribution,
            CancellationToken::new(),
        );
        assert!(
            std::panic::AssertUnwindSafe(call)
                .catch_unwind()
                .await
                .is_err(),
            "the panic reaches the batch"
        );
        assert!(
            store
                .background_store()
                .list_background_tasks(session_id)
                .await
                .expect("tasks")
                .is_empty()
        );
        assert_eq!(
            agent.cells.background_tasks.running_count(session_id).await,
            0
        );
    }

    /// A turn stopped while a start step that never waits on the stop runs still stops the call:
    /// its run sees the stop at once, inline, and the call reads as interrupted with no task left.
    #[tokio::test]
    async fn a_start_step_that_finishes_after_the_stop_is_interrupted_and_leaves_no_task() {
        let (agent, store, session_id) = agent_with_start(StartAs::Running(None)).await;
        let stop = CancellationToken::new();
        stop.cancel();
        let output = agent
            .resolve_and_execute_tool(
                ResolvedCall {
                    tool_call_id: "call-1",
                    name: "start_fixture",
                    input: &serde_json::json!({"background": true}),
                    unparsed: None,
                },
                &[],
                &crate::provider::Attribution::default(),
                stop,
            )
            .await;

        assert_eq!(output.text_content(), "Tool execution interrupted.");
        assert!(
            store
                .background_store()
                .list_background_tasks(session_id)
                .await
                .expect("tasks")
                .is_empty()
        );
        assert_eq!(
            agent.cells.background_tasks.running_count(session_id).await,
            0
        );
    }

    /// The row is written before the start step runs, and a call stopped while it starts is an
    /// interrupted call: the row is withdrawn and the slot freed, whatever a second press that
    /// stopped every task recorded on it meanwhile, since nothing ran for a report to tell of.
    #[tokio::test]
    async fn a_call_stopped_while_it_starts_is_interrupted_and_leaves_no_task() {
        let (agent, store, session_id) = agent_with_start(StartAs::Stopped).await;
        let stop = CancellationToken::new();
        let input = serde_json::json!({"background": true});
        let attribution = crate::provider::Attribution::default();
        let call = agent.resolve_and_execute_tool(
            ResolvedCall {
                tool_call_id: "call-1",
                name: "start_fixture",
                input: &input,
                unparsed: None,
            },
            &[],
            &attribution,
            stop.clone(),
        );
        let stopper = async {
            loop {
                let rows = store
                    .background_store()
                    .list_background_tasks(session_id)
                    .await
                    .expect("tasks");
                if !rows.is_empty() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
            agent
                .cells
                .background_tasks
                .stop_all(
                    &store.background_store(),
                    crate::store::background::TaskStatus::Interrupted,
                )
                .await;
            stop.cancel();
        };
        let (output, ()) = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            tokio::join!(call, stopper)
        })
        .await
        .expect("the row appears while the start step waits, and the stop ends it");

        assert!(output.is_error);
        assert_eq!(output.text_content(), "Tool execution interrupted.");
        assert!(
            store
                .background_store()
                .list_background_tasks(session_id)
                .await
                .expect("tasks")
                .is_empty()
        );
        assert_eq!(
            agent.cells.background_tasks.running_count(session_id).await,
            0
        );
    }

    /// A background run's output reaches its report through the doors a round's result takes:
    /// an image goes to a scratchpad entry of its base64 bytes, the form `image_render` takes one
    /// in, rather than becoming `[Image]`; a text over the report's bound spills under the name the
    /// tool hinted, one the tool sized to the context included; and an entry a parent lent
    /// read-only is refused rather than shadowed, the refusal following the head of a long output.
    #[tokio::test]
    async fn a_background_outcome_takes_the_doors_a_result_takes() {
        use crate::conversation::ToolResultContent;

        let store = crate::store::Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("a session");
        let image = crate::image::ImageSource::Base64 {
            media_type: "image/png".to_string(),
            data: "iVBORw0KGgo=".to_string(),
        };
        let output = |content: Vec<ToolResultContent>, hint: crate::tools::SpillHint| {
            crate::tools::ToolOutput {
                content,
                is_output: true,
                spill_hint: hint,
                ..Default::default()
            }
        };
        let text = |text: &str| ToolResultContent::Text {
            text: text.to_string(),
        };
        let task = "7f3a1c22-0000-4000-8000-000000000000";

        let (outcome, spilled, named) = background_outcome(
            &store,
            session_id,
            task,
            "mcp__kv__shot",
            output(
                vec![text("a shot: "), ToolResultContent::Image {
                    source: image.clone(),
                }],
                crate::tools::SpillHint::under("mcp_kv_shot".to_string()),
            ),
            None,
            &[],
        )
        .await;
        assert_eq!(
            outcome,
            "a shot: [Image in scratchpad entry 'mcp_kv_shot_image_1']"
        );
        assert!(spilled.is_none() && named.is_none());
        let saved = store
            .load_scratchpad_entry(session_id, "mcp_kv_shot_image_1")
            .await
            .expect("read")
            .expect("the image's entry");
        assert_eq!(saved, "iVBORw0KGgo=", "its base64 bytes");

        let long = "x".repeat(crate::background::OUTCOME_INLINE_BYTES + 10);
        let (outcome, spilled, _) = background_outcome(
            &store,
            session_id,
            task,
            "mcp__kv__fetch",
            output(
                vec![text(&long)],
                crate::tools::SpillHint::under("mcp_kv_fetch".to_string()),
            ),
            None,
            &[],
        )
        .await;
        assert_eq!(outcome.len(), crate::background::OUTCOME_INLINE_BYTES);
        assert_eq!(spilled.as_deref(), Some("mcp_kv_fetch"));
        let (outcome, spilled, _) = background_outcome(
            &store,
            session_id,
            task,
            "shell_execute",
            output(
                vec![text(&long)],
                crate::tools::SpillHint::sized_to_context(),
            ),
            None,
            &[],
        )
        .await;
        assert_eq!(
            outcome.len(),
            crate::background::OUTCOME_INLINE_BYTES,
            "the report's bound, below what the tool sized to"
        );
        assert!(spilled.is_some());

        let (outcome, spilled, named) = background_outcome(
            &store,
            session_id,
            task,
            "shell_execute",
            output(vec![text(&long)], crate::tools::SpillHint::default()),
            Some("parent-notes"),
            &["parent-notes".to_string()],
        )
        .await;
        assert!(named.is_none(), "nothing written under the lent name");
        assert!(spilled.is_some_and(|name| name != "parent-notes"));
        assert!(
            outcome.starts_with('x')
                && outcome.trim_end().ends_with(']')
                && outcome.contains("[Not saved to the scratchpad:"),
            "{outcome}"
        );
        let (outcome, _, named) = background_outcome(
            &store,
            session_id,
            task,
            "shell_execute",
            output(vec![text("built")], crate::tools::SpillHint::default()),
            Some("build-log"),
            &[],
        )
        .await;
        let named = named.expect("the entry the call named");
        assert_eq!(
            (named.name.as_str(), named.content.as_str()),
            ("build-log", "built")
        );
        assert!(outcome.contains("'build-log'"), "{outcome}");
    }
}
