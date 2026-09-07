//! Turning tool calls into tool results: resolution, approval, the background hand-off, and the
//! call itself.

use super::*;

/// The arguments an approval prompt is shown, which is not quite the arguments the tool receives.
///
/// `background` is meka's own parameter, spliced into every schema by the registry and taken out
/// again before dispatch so no tool sees a key it never advertised. It is also the argument that
/// decides whether the call detaches and outlives the turn, so a prompt that showed everything
/// except that would be asking about a different call than the one about to run.
pub(super) fn approval_input(input: &serde_json::Value, detach: bool) -> serde_json::Value {
    let mut shown = input.clone();
    if detach && let Some(fields) = shown.as_object_mut() {
        fields.insert("background".to_string(), serde_json::Value::Bool(true));
    }
    shown
}

impl Agent {
    /// `loaded` is the turn's active-tool set, used only to tell a call made against a schema the
    /// model has actually seen from one made blind; see [`Self::schema_advisory`].
    pub(super) async fn execute_tool_calls(
        &self,
        assistant_message: &Message,
        loaded: &[String],
        prompt_id: Option<Uuid>,
        cancellation: CancellationToken,
    ) -> Vec<ContentBlock> {
        // Emit tool-call indicators in source order. The streaming path already emitted these as
        // `ToolUseEnd` events; this loop only fires for the blocking provider path. Serial so
        // concurrent execution below can't interleave indicators.
        let mut planned: Vec<(String, String, serde_json::Value)> = Vec::new();
        for block in &assistant_message.content {
            if let ContentBlock::ToolUse { id, name, input } = block {
                if !self.options.streaming {
                    let schema = self
                        .tool_registry
                        .get(name)
                        .map(|t| t.definition().parameters);
                    let display_summary =
                        crate::tools::resolve_primary_param(name, input, schema.as_ref());
                    self.cells
                        .frontend
                        .emit(FrontendEvent::ToolCallStarted {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                            display_summary,
                        })
                        .await;
                }
                planned.push((id.clone(), name.clone(), input.clone()));
            }
        }

        // Dispatch concurrently. `join_all` preserves input ordering so the i-th output corresponds
        // to the i-th planned call.
        let futures = planned.iter().map(|(id, name, input)| {
            self.resolve_and_execute_tool(
                id.as_str(),
                name.as_str(),
                input,
                loaded,
                prompt_id,
                cancellation.clone(),
            )
        });
        let outputs = futures::future::join_all(futures).await;

        // Serial pass to accumulate scratchpad hints, emit per-tool completion events in source
        // order, build ToolResult blocks, and emit a single TodoListUpdated event if any `todo`
        // call landed and actually changed the rendered state.
        let mut results = Vec::with_capacity(planned.len());
        let mut todo_fired = false;
        for ((id, name, _), output) in planned.into_iter().zip(outputs) {
            if name == "todo" {
                todo_fired = true;
            }
            if let Some(hint) = output.scratchpad_hint.clone() {
                self.scratchpad_hints.write().await.insert(id.clone(), hint);
            }
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
                })
                .await;
            results.push(ContentBlock::ToolResult {
                tool_use_id: id,
                content: output.content,
                is_error: output.is_error,
            });
        }
        if todo_fired {
            let state = self.cells.todo_list.get();
            // Suppress re-renders for reads and rewrites that change nothing. Drop the guard before
            // awaiting the emit.
            let changed = {
                let mut last = self.last_rendered_todo.write().await;
                let changed = last.as_ref() != Some(&state);
                if changed {
                    *last = Some(state.clone());
                }
                changed
            };
            // An empty list renders nothing, so emitting it would be a no-op event that also
            // corrupts REPL spacing; require something to show.
            let should_emit = !state.items.is_empty() && changed;
            if should_emit {
                self.cells
                    .frontend
                    .emit(FrontendEvent::TodoListUpdated {
                        title: state.title,
                        items: state.items,
                    })
                    .await;
            }
        }

        results
    }

    pub(super) async fn resolve_and_execute_tool(
        &self,
        tool_call_id: &str,
        name: &str,
        input: &serde_json::Value,
        loaded: &[String],
        prompt_id: Option<Uuid>,
        cancellation: CancellationToken,
    ) -> crate::tools::ToolOutput {
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
            if !self.tool_registry.denials().denies_tool(name)
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

        // Read at the enforcement site, so a level cycled or a switch toggled during dispatch
        // means the next call rather than a snapshot captured earlier in the loop.
        let required = self
            .tool_registry
            .required_permission_for(name)
            .unwrap_or_else(|| tool.required_permission());
        let permission = self.cells.permission.get();
        let admission = admit_tool_call(
            name,
            required,
            permission,
            self.cells.permission.approvals(),
            tool.runs_outside_confinement(),
        );
        if let Admission::Refuse(refusal) = admission {
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
        let schema = tool.definition().parameters;

        let (input, detach) = match crate::tools::admit_arguments(name, input, &schema) {
            Ok(admitted) => admitted,
            Err(refusal) => return refusal,
        };
        let input = &input;

        // A detached call that cannot start is refused here, ahead of the approval prompt. Every
        // refusal `start_background_call` makes needs nothing but what is known now, so asking the
        // user first put a question to them whose answer could not matter: the call was refused the
        // moment they said yes.
        if detach && let Err(refusal) = self.admit_detach(session_id).await {
            return refusal;
        }

        if matches!(admission, Admission::Ask) {
            // The same rule for the tool's own doors: an approved call runs at the level, so a
            // refusal the level already decides (the shell with nothing to confine it, a write
            // outside the roots) is returned here rather than asked about and then made anyway.
            if let Some(refusal) = tool.refusal_at_level(permission, input).await {
                return refusal;
            }
            // Approval resolves *before* a detach, never inside it. A prompt surfacing minutes
            // after the turn that caused it, with nothing on screen to explain it, is worse than
            // the round trip it would save.
            if let Some(denial) = self
                .request_approval(name, input, detach, &schema, &cancellation)
                .await
            {
                return denial;
            }
        }

        let mut output = if detach {
            self.start_background_call(&tool, tool_call_id, name, input, session_id, prompt_id)
                .await
        } else {
            Self::run_tool(&*tool, input, crate::tools::ToolContext {
                session_id,
                tool_call_id: Some(tool_call_id.to_string()),
                prompt_id,
                frontend: Arc::clone(&self.cells.frontend),
                cancellation,
            })
            .await
        };
        if let Some(advisory) = self.schema_advisory(name, input, &schema, loaded, !output.is_error)
        {
            output.append_notice(&advisory);
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
        name: &str,
        input: &serde_json::Value,
        detach: bool,
        schema: &serde_json::Value,
        cancellation: &CancellationToken,
    ) -> Option<crate::tools::ToolOutput> {
        let primary_param = crate::tools::resolve_primary_param(name, input, Some(schema));
        let outcome = self
            .cells
            .frontend
            .request_permission(PermissionRequest {
                tool_name: name.to_string(),
                primary_param,
                input: approval_input(input, detach),
                cancellation: cancellation.clone(),
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
    /// the user is never asked about a call that is refused the moment they approve it. The slot
    /// check here is a preview: [`crate::background::BackgroundTasks::try_reserve`] is the atomic
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

    /// Detach one call: record it, spawn it, and hand the model a task id instead of a result.
    ///
    /// The spawned work gets a **fresh** cancellation token rather than the turn's. Sharing the
    /// turn's would kill the task the instant the turn ended, which is the whole thing this exists
    /// to avoid; the cost is that Ctrl+C no longer reaches it, which is why `task_cancel` and the
    /// second-press escalation exist.
    pub(super) async fn start_background_call(
        &self,
        tool: &Arc<dyn crate::tools::Tool>,
        tool_call_id: &str,
        name: &str,
        input: &serde_json::Value,
        session_id: Option<Uuid>,
        prompt_id: Option<Uuid>,
    ) -> crate::tools::ToolOutput {
        let session_id = match self.admit_detach(session_id).await {
            Ok(session_id) => session_id,
            Err(refusal) => return refusal,
        };
        let schema = tool.definition().parameters;
        let label = crate::tools::resolve_primary_param(name, input, Some(&schema))
            .unwrap_or_else(|| name.to_string());
        let task = crate::store::background::BackgroundTask {
            id: Uuid::new_v4().to_string(),
            session_id,
            tool_name: name.to_string(),
            label,
            status: crate::store::background::TaskStatus::Running,
            outcome: None,
            scratchpad_name: None,
            started_at: chrono::Utc::now(),
            finished_at: None,
            announced_at: None,
            delivered_at: None,
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

        // Recorded before the spawn, so a process that dies in between leaves a `running` row the
        // sweep can retire rather than work nobody knows happened.
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

        let join = tokio::spawn({
            let tool = Arc::clone(tool);
            let input = input.clone();
            let frontend = Arc::clone(&self.cells.frontend);
            let store = self.store.clone();
            let tasks = self.cells.background_tasks.clone();
            let cancellation = cancellation.clone();
            let tool_call_id = tool_call_id.to_string();
            let task_id = task.id.clone();
            let tool_name = task.tool_name.clone();
            async move {
                // Published for the frontend as well as passed to the tool: a delegated `fs/*` or
                // elicitation must race *this* token, not the session's current turn. See
                // `crate::frontend::scope_call_cancellation`.
                let scoped = cancellation.clone();
                let context = crate::tools::ToolContext {
                    session_id: Some(session_id),
                    tool_call_id: Some(tool_call_id),
                    prompt_id,
                    frontend,
                    cancellation: scoped.clone(),
                };
                let run = crate::frontend::scope_call_cancellation(scoped, async move {
                    Self::run_tool(&*tool, &input, context).await
                });
                // A panic must not escape this task. Nothing awaits its `JoinHandle` outside
                // `--oneshot`, so an unwind here would skip both the outcome write and the slot
                // release: the agent would wait forever on a report that is never coming, and the
                // ceiling would be permanently one lower. Turning it into a `failed` outcome is
                // what the rest of the machinery already knows how to deliver.
                use futures::FutureExt;
                let output = match std::panic::AssertUnwindSafe(run).catch_unwind().await {
                    Ok(output) => output,
                    Err(_) => {
                        tracing::error!("background task {task_id} panicked");
                        crate::tools::ToolOutput::text(
                            "The tool panicked while running in the background.".to_string(),
                            true,
                        )
                    }
                };

                let text =
                    crate::conversation::ContentBlock::tool_result_text_content(&output.content);
                let (inline, spilled) = crate::background::split_outcome(&text);
                let mut scratchpad_name = None;
                if let Some(full) = spilled {
                    let name = crate::background::spill_entry_name(&task_id, &tool_name);
                    match store.save_scratchpad_entry(session_id, &name, &full).await {
                        Ok(()) => scratchpad_name = Some(name),
                        // Not fatal: the head still reaches the model, and losing the tail is far
                        // better than losing the whole report.
                        Err(error) => tracing::warn!(
                            "background task {task_id}: failed to spill output to the scratchpad: {error}"
                        ),
                    }
                }

                let status = if output.is_error {
                    crate::store::background::TaskStatus::Failed
                } else {
                    crate::store::background::TaskStatus::Completed
                };
                record_background_outcome(&store, &task_id, status, inline, scratchpad_name).await;
                tasks.forget(&task_id).await;
            }
        });
        self.cells.background_tasks.attach(&task.id, join).await;

        crate::tools::ToolOutput::text(
            format!(
                "Started in the background as task {} ({}). It is still running; its result will \
                 be delivered to you when it finishes. Do not wait for it here. Use `task_list` to \
                 check on it and `task_cancel` with \"{}\" to stop it.",
                task.short_id(),
                task.label,
                task.short_id(),
            ),
            false,
        )
    }

    /// Invoke a tool and turn its failure into the output the model reads.
    pub(super) async fn run_tool(
        tool: &dyn crate::tools::Tool,
        input: &serde_json::Value,
        context: crate::tools::ToolContext,
    ) -> crate::tools::ToolOutput {
        match tool.execute(input.clone(), context).await {
            Ok(output) => output,
            Err(MekaError::Interrupted) => {
                crate::tools::ToolOutput::text("Tool execution interrupted.".to_string(), true)
            }
            Err(error) => crate::tools::ToolOutput::from_error(&error),
        }
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
    scratchpad_name: Option<String>,
) {
    let background = store.background_store();
    let Err(error) = background
        .finish_background_task(
            task_id,
            status,
            Some(outcome.clone()),
            scratchpad_name.clone(),
        )
        .await
    else {
        return;
    };
    tracing::warn!("background task {task_id} finished but failed to record it; retrying: {error}");
    tokio::time::sleep(OUTCOME_RETRY_DELAY).await;
    let Err(error) = background
        .finish_background_task(task_id, status, Some(outcome), scratchpad_name)
        .await
    else {
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

/// What the door decided about one tool call.
pub(super) enum Admission {
    /// Within the level: run it.
    Run,
    /// Above the level, and the session submits such calls for approval.
    Ask,
    /// Above the level, with nobody to ask: the result the model gets instead.
    ///
    /// Boxed because a `ToolOutput` alone crosses clippy's `large_enum_variant` threshold on
    /// Windows, where `PathBuf` is eight bytes wider.
    Refuse(Box<crate::tools::ToolOutput>),
}

/// The one rule for whether a tool call runs, is submitted for approval, or is refused.
///
/// Dispatch and the checkpoint both ask this, so the two doors cannot disagree. The level bounds
/// what runs unattended and the approvals switch decides what happens at the edge of it. An
/// approved call still runs *at the level*: the write fence and the shell's confinement read the
/// same cell, so approval never widens reach, it only turns a refusal into a question.
///
/// `unconfinable` is the door `Permission::allows` cannot provide for a tool meka cannot confine.
/// `allows` treats `workspace` and `unrestricted` as equal on purpose, so a tool requiring
/// `unrestricted` dispatches at `workspace` with no prompt; every built-in that matters has its
/// own door downstream (the write fence, or `execute_command`'s refusal when it cannot be
/// sandboxed), and an MCP adapter has neither. Such a call is a question when approvals are on and
/// a refusal otherwise, the way the shell refuses when its sandbox is unavailable: half a boundary
/// reported as a whole one is worse than an error saying so. Only `workspace` promises a boundary
/// it might fail to apply, so only there does the flag matter.
pub(super) fn admit_tool_call(
    name: &str,
    required: crate::permission::Permission,
    permission: crate::permission::Permission,
    approvals: bool,
    unconfinable: bool,
) -> Admission {
    if !permission.allows(required) {
        if approvals {
            return Admission::Ask;
        }
        return Admission::Refuse(Box::new(crate::tools::ToolOutput::text(
            format!(
                "'{name}' requires `{required}`; the session is at `{permission}`. Ask the user to \
                 raise it to `{required}`."
            ),
            true,
        )));
    }
    if permission == crate::permission::Permission::Workspace
        && !required.is_within(permission)
        && unconfinable
    {
        if approvals {
            return Admission::Ask;
        }
        return Admission::Refuse(Box::new(crate::tools::ToolOutput::text(
            format!(
                "'{name}' runs inside its MCP server's own process, which meka does not sandbox, so \
                 `workspace` cannot confine what it writes. Ask the user for `unrestricted`, or \
                 grant it explicitly with `[mcp.servers.*].tool_permissions` in the config."
            ),
            true,
        )));
    }
    Admission::Run
}

#[cfg(test)]
mod tests {
    use super::{Admission, admit_tool_call, *};
    use crate::{
        agent::tests::{SendFileFixture, agent_with_registry_for_test, send_file_registry},
        permission::Permission,
    };

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

    fn decide(required: Permission, level: Permission, approvals: bool) -> &'static str {
        match admit_tool_call("t", required, level, approvals, false) {
            Admission::Run => "run",
            Admission::Ask => "ask",
            Admission::Refuse(_) => "refuse",
        }
    }

    /// The one rule, at every level: within the level runs, above it is refused, and the switch
    /// turns that refusal into a question. Nothing sits above `unrestricted`, so it never asks.
    #[test]
    fn a_call_above_the_level_is_refused_or_asked_and_never_run() {
        assert_eq!(
            decide(Permission::Workspace, Permission::Read, false),
            "refuse"
        );
        assert_eq!(decide(Permission::Workspace, Permission::Read, true), "ask");
        assert_eq!(decide(Permission::Read, Permission::Read, true), "run");
        assert_eq!(decide(Permission::Read, Permission::None, true), "ask");
        assert_eq!(decide(Permission::Read, Permission::None, false), "refuse");
        assert_eq!(
            decide(Permission::Unrestricted, Permission::Unrestricted, true),
            "run"
        );
        // `workspace` runs a tool that requires `unrestricted` when meka can confine it; the
        // approval question is never asked for a call the level already covers.
        assert_eq!(
            decide(Permission::Unrestricted, Permission::Workspace, true),
            "run"
        );
    }

    /// A tool meka cannot confine is the one case where `workspace` covers the requirement and
    /// still cannot deliver it: refused, or asked about when the switch is on.
    #[test]
    fn an_unconfinable_call_at_workspace_is_asked_about_when_approvals_are_on() {
        let refused = admit_tool_call(
            "mcp__x__t",
            Permission::Unrestricted,
            Permission::Workspace,
            false,
            true,
        );
        assert!(matches!(refused, Admission::Refuse(_)));
        let asked = admit_tool_call(
            "mcp__x__t",
            Permission::Unrestricted,
            Permission::Workspace,
            true,
            true,
        );
        assert!(matches!(asked, Admission::Ask));
        // Not at `unrestricted`, which promises no boundary to fail to apply.
        let run = admit_tool_call(
            "mcp__x__t",
            Permission::Unrestricted,
            Permission::Unrestricted,
            true,
            true,
        );
        assert!(matches!(run, Admission::Run));
    }

    /// The refusal names the level it would take, which is what the model relays to the user.
    #[test]
    fn a_refusal_names_the_required_level() {
        let Admission::Refuse(output) = admit_tool_call(
            "write_file",
            Permission::Workspace,
            Permission::Read,
            false,
            false,
        ) else {
            panic!("a call above the level with approvals off is refused");
        };
        let text = output.text_content();
        assert!(output.is_error);
        assert!(text.contains("requires `workspace`"), "{text}");
        assert!(text.contains("the session is at `read`"), "{text}");
        // meka's own decision is a refusal; "denied" is the user's answer at the prompt.
        assert!(!text.contains("denied"), "{text}");
    }

    /// A tool that runs outside any confinement meka can apply is refused at `workspace`, for
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
                    "call-1",
                    name,
                    &serde_json::json!({}),
                    &[],
                    None,
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
        registry.register(Arc::new(Fixture)).expect("registration");
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
                "call-1",
                "write_thing",
                &serde_json::json!({"background": true}),
                &[],
                None,
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

    /// With `[shell].sandbox = false`, `execute_command` at `read` is refused however the user
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
                "call-1",
                "execute_command",
                &serde_json::json!({"command": "true"}),
                &[],
                None,
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
                    "call-1",
                    "mcp__bridge__send_file",
                    &serde_json::json!({"path": path}),
                    &[],
                    None,
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

    /// One `TodoListUpdated` per change with something to show. A rewrite that changes nothing
    /// re-renders nothing, and a list emptied out renders nothing either: an empty render is a
    /// no-op event that also corrupts REPL spacing.
    #[tokio::test]
    async fn an_unchanged_or_emptied_todo_list_is_not_re_rendered() {
        use crate::provider::mock::MockProvider;

        let (mut agent, _store) = agent_with_registry_for_test(
            Arc::new(MockProvider::from_rounds(vec![])),
            crate::tools::ToolRegistry::new(),
        )
        .await;
        agent
            .tool_registry
            .register(Arc::new(crate::tools::todo::TodoTool {
                todo_list: agent.cells.todo_list.clone(),
            }))
            .expect("registration");
        let frontend = Arc::new(crate::frontend::testing::RecordingFrontend::new());
        agent.cells.frontend = Arc::clone(&frontend) as Arc<dyn crate::frontend::Frontend>;

        let todo_call = |input: serde_json::Value| Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "todo-1".to_string(),
                name: "todo".to_string(),
                input,
            }],
        };
        let rendered = || {
            frontend
                .events()
                .iter()
                .filter(|event| matches!(event, FrontendEvent::TodoListUpdated { .. }))
                .count()
        };

        let list = serde_json::json!({"title": "Plan", "items": ["first"]});
        agent
            .execute_tool_calls(
                &todo_call(list.clone()),
                &[],
                None,
                CancellationToken::new(),
            )
            .await;
        assert_eq!(rendered(), 1, "a new list is rendered once");
        agent
            .execute_tool_calls(&todo_call(list), &[], None, CancellationToken::new())
            .await;
        assert_eq!(rendered(), 1, "rewriting the same list renders nothing");
        agent
            .execute_tool_calls(
                &todo_call(serde_json::json!({"items": []})),
                &[],
                None,
                CancellationToken::new(),
            )
            .await;
        assert_eq!(rendered(), 1, "an emptied list has nothing to render");
    }
}
