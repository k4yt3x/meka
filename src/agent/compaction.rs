//! Compaction and the checkpoint turn that precedes it: what is summarized, where the split falls,
//! and what the model is asked to save first.

use super::*;
use crate::session::{CompactOrigin, CompactRequest, compaction_tail_budget};

/// How many compactions a single turn may honor on the agent's own request.
///
/// Each one costs a summarizer call and, with `[session].compact_checkpoint` on, up to
/// [`CHECKPOINT_MAX_ITERATIONS`] more. A model that asks again in the same turn is answered by the
/// tool and then dropped, not carried forward: it can ask again on a later turn, and meanwhile a
/// confused agent is bounded to one round rather than a loop.
pub(super) const MAX_REQUESTED_COMPACTIONS: u32 = 1;
/// Say what a checkpoint durably wrote, on every path out of a compaction that ran one.
///
/// `/compact` reports this to the user directly (`render::compaction_summary`); the four automatic
/// paths discard the outcome, so without this a reactive, proactive or agent-requested compaction
/// would write instance-scoped notes with no trace at any verbosity. A function rather than a line
/// at the end because an interrupt now returns early, and the writes are durable the moment they
/// run: the report has to leave with the error too, not only with the summary.
///
/// `info!` rather than `warn!`: a lifecycle signpost, not a problem.
pub(super) fn report_checkpoint_memories(memories_written: &[String]) {
    if memories_written.is_empty() {
        return;
    }
    let count = memories_written.len();
    let noun = if count == 1 { "memory" } else { "memories" };
    let names = memories_written.join(", ");
    tracing::info!("checkpoint wrote {count} {noun}: {names}");
}
/// Which strategy produced the summary that replaced the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompactSource {
    /// The checkpoint turn, via `context_replace`. The intended path.
    Checkpoint,
    /// The checkpoint turn ran but never called `context_replace`, so its closing text was used.
    /// `Provider::complete` has no `tool_choice`, so the call cannot be forced and this is
    /// reachable on any backend.
    CheckpointText,
    /// The standalone summarizer: the emergency path, a disabled checkpoint, or a checkpoint that
    /// produced nothing usable.
    Summarizer,
}
/// What a compaction did, for the caller to report. `/compact` is the only consumer today; the
/// automatic triggers discard it.
#[derive(Debug, Clone)]
pub(crate) struct CompactOutcome {
    pub(crate) source: CompactSource,
    /// Memories the checkpoint turn wrote, observed from its `memory_write` calls rather than
    /// self-reported, so this cannot disagree with what actually landed on disk.
    pub(crate) memories_written: Vec<String>,
    pub(crate) kept_recent: bool,
}
/// Provider round-trips one checkpoint turn may take before its summary is read off.
///
/// A bound, not a budget: the turn should need one or two (write a memory, then submit). This only
/// stops a model that keeps finding more to save from compacting forever, and a turn that reaches
/// it still lands on the text fallback rather than failing outright.
pub(super) const CHECKPOINT_MAX_ITERATIONS: usize = 8;
/// What a checkpoint turn produced.
pub(super) struct Checkpoint {
    summary: String,
    source: CompactSource,
    keep_recent: Option<bool>,
}
/// The user message that turns an ordinary turn into a checkpoint.
///
/// A *user* message and not a system-prompt swap, which is the whole design in one detail: the
/// agent's own prompt is what makes a checkpoint worth more than the standalone summarizer, and
/// replacing it would discard exactly the identity, instructions and memory index that make the
/// difference.
pub(super) fn checkpoint_instruction(request: &CompactRequest) -> String {
    let mut instruction = String::from("[Checkpoint: your context is about to be summarized]\n\n");
    instruction.push_str(match request.origin {
        CompactOrigin::Requested => "You asked for this compaction.\n\n",
        CompactOrigin::Manual => "The user asked for this compaction.\n\n",
        // Reactive and Proactive alike: the agent did not choose the moment, so say so rather than
        // letting it read an involuntary interruption as its own decision.
        _ => {
            "The conversation has grown close to the context window, so this is happening now \
              rather than when you would have chosen.\n\n"
        }
    });
    instruction.push_str(
        "Everything above is about to be replaced by a summary you write here, except for a short \
         run of the most recent turns, which is kept as-is. This is the one moment you can act \
         before that happens.\n\n\
         First, save whatever must outlive this conversation. `memory_write` is for what should \
         still be true in a future session: facts about the user, standing preferences, decisions \
         and the reasons behind them. The scratchpad is for working material this task still \
         needs. Prefer updating an existing memory to writing a near-duplicate.\n\n\
         Then call `context_replace` with a summary written for yourself, in your own voice, \
         covering: what is being worked on and why, what is done and what is left, decisions and \
         their reasons, what the user asked for or corrected (quote any constraint on what not to \
         do verbatim, so it keeps applying), commitments you have made but not yet delivered, and \
         the immediate next step.\n\n\
         The full history stays on disk and `conversation_search` reaches it, so do not try to \
         reproduce it here. Write what someone would need to carry the work on without it.",
    );
    if let Some(extra) = &request.instructions {
        instruction.push_str(&format!(
            "\n\nInstructions for this specific compaction, which take precedence over the \
             above:\n{extra}"
        ));
    }
    if request.keep_recent == Some(false) {
        instruction.push_str(
            "\n\nThis compaction was asked to keep nothing verbatim, so those recent turns will be \
             discarded as well. Your summary has to cover them too.",
        );
    }
    instruction
}
/// Split a conversation for compaction into `(to_summarize, to_keep)`. The kept tail is the largest
/// recent suffix whose estimated tokens stay within `keep_budget`, then snapped backward to a clean
/// `User`-without-`tool_results` boundary so a tool_use/tool_result pair is never orphaned and the
/// kept window starts on a valid user turn. If that leaves fewer than `MIN_SUMMARIZE` messages to
/// summarize, the whole conversation is summarized and no tail is kept (a smaller head saves too
/// little to be worth a boundary), except for a trailing user message nobody has answered: that is
/// the request the model is about to answer, and it stays verbatim however short the conversation,
/// or the model answers a summary of it and the user's words are gone from the window.
pub(super) fn compute_compaction_split(
    view: &[Message],
    keep_budget: u64,
) -> (Vec<Message>, Vec<Message>) {
    const MIN_SUMMARIZE: usize = 4;
    if view.len() <= MIN_SUMMARIZE {
        return summarize_all_but_a_trailing_prompt(view);
    }

    // Grow the tail from the end while it fits the budget; always keep at least the last message.
    let mut split = view.len();
    let mut tail_tokens = 0u64;
    while let Some(message) = split.checked_sub(1).and_then(|previous| view.get(previous)) {
        let candidate = tail_tokens.saturating_add(crate::tokens::estimate_message(message));
        if candidate > keep_budget && split < view.len() {
            break;
        }
        tail_tokens = candidate;
        split -= 1;
    }

    // Snap back to a clean user boundary. This only grows the tail, so it never orphans a
    // tool_result and guarantees the kept window starts on a User turn.
    while split > 0 {
        let Some(message) = view.get(split) else {
            break;
        };
        if message.role == Role::User && !has_tool_results(&message.content) {
            break;
        }
        split -= 1;
    }

    if split >= MIN_SUMMARIZE {
        let (head, tail) = view.split_at(split);
        (head.to_vec(), tail.to_vec())
    } else {
        summarize_all_but_a_trailing_prompt(view)
    }
}

/// The whole conversation as the head, save a trailing unanswered prompt, which is the tail.
///
/// Only a plain user message qualifies: a trailing `tool_result` belongs to the `tool_use` before
/// it, and keeping it alone would orphan the pair the snap-back exists to protect.
fn summarize_all_but_a_trailing_prompt(view: &[Message]) -> (Vec<Message>, Vec<Message>) {
    match view.split_last() {
        Some((last, head))
            if !head.is_empty() && last.role == Role::User && !has_tool_results(&last.content) =>
        {
            (head.to_vec(), vec![last.clone()])
        }
        _ => (view.to_vec(), Vec::new()),
    }
}
/// Cap a checkpoint tool result at the size a normal turn would allow inline.
///
/// A normal turn spills anything larger to the scratchpad
/// ([`crate::tools::scratchpad::persist_oversized_results`]), but that runs in `run_turn` and the
/// checkpoint loop is not a turn. Without a bound here the checkpoint would be the one place in
/// meka where a tool result enters the conversation at unlimited size, and it would do so at the
/// worst possible moment: the window is near full, which is why compaction is running at all. A
/// single `read_file` could then overflow the request, and the only visible consequence would be a
/// warn line and a silent fall back to the summarizer.
///
/// Truncated rather than spilled, because spilling would create scratchpad entries nobody asked
/// for during an automatic operation, and a checkpoint needs enough of a result to decide what to
/// write down, not all of it.
pub(super) fn bound_checkpoint_result(
    content: Vec<crate::conversation::ToolResultContent>,
) -> Vec<crate::conversation::ToolResultContent> {
    use crate::conversation::ToolResultContent;

    let limit = crate::tools::scratchpad::MAX_INLINE_RESULT_BYTES;
    content
        .into_iter()
        .map(|item| match item {
            ToolResultContent::Text { text } if text.len() > limit => {
                let end = text.floor_char_boundary(limit);
                ToolResultContent::Text {
                    text: format!(
                        "{}\n... (truncated: this result was too large to carry into a \
                         checkpoint)",
                        &text[..end]
                    ),
                }
            }
            other => other,
        })
        .collect()
}

impl Agent {
    /// Replace the conversation's head with a summary of it, keeping a recent tail verbatim.
    ///
    /// Two strategies produce the summary. The checkpoint turn ([`Self::run_checkpoint_turn`]) is
    /// the agent summarizing itself, which is what lets it persist to memory on the way past;
    /// [`Self::summarize_via_provider`] is a standalone call that knows nothing but the transcript.
    /// Everything after the summary text is chosen is common to both.
    pub(crate) async fn compact_session(
        &self,
        messages: &mut Conversation,
        request: CompactRequest,
        cancellation: CancellationToken,
    ) -> Result<CompactOutcome> {
        let Some(session_id) = self.cells.session_id.get() else {
            return Err(MekaError::Config(
                "no active session to compact".to_string(),
            ));
        };

        if messages.is_empty() {
            return Err(MekaError::Config("no messages to compact".to_string()));
        }

        // Owned here rather than returned by the checkpoint, because a `memory_write` is durable
        // the moment it runs and the turn can still fail or be canceled afterwards. Returned by
        // value it would be dropped on exactly those paths, so the notes would be on disk --
        // overwriting whatever was there -- while the caller was told none were written, with no
        // trace at any verbosity. `CompactResponse::memories_written` promises the opposite.
        let mut memories_written: Vec<String> = Vec::new();
        let checkpoint =
            if request.origin == CompactOrigin::Emergency || !self.options.compact_checkpoint {
                None
            } else {
                match self
                    .run_checkpoint_turn(
                        &request,
                        Some(session_id),
                        messages.as_slice(),
                        cancellation.clone(),
                        &mut memories_written,
                    )
                    .await
                {
                    Ok(result) => result,
                    // Never fatal. Compaction is what keeps a session alive, and the summarizer
                    // below can always do the job, so a checkpoint that fails costs fidelity and
                    // nothing else.
                    Err(error) => {
                        tracing::warn!("checkpoint turn failed; summarizing instead: {error}");
                        None
                    }
                }
            };

        let keep_recent = match &checkpoint {
            // `context_replace` decided last and knew most, having just read the conversation, so
            // its answer outranks the one the caller guessed at.
            Some(checkpoint) => checkpoint
                .keep_recent
                .or(request.keep_recent)
                .unwrap_or(true),
            // A `keep_recent: false` is a bet that the checkpoint captured everything worth
            // keeping, usually into memory. With no checkpoint the bet was never placed: nothing
            // was saved, and the summarizer works from a copy with every long block truncated.
            // Honoring the request here would discard the verbatim tail *and* summarize the rest
            // from an excerpt, compounding a failure into real data loss. Keep the tail instead.
            None => true,
        };

        // A trailing user message is one nobody has answered yet. `CompactOrigin::Proactive` fires
        // *after* `run_turn` appends this turn's prompt and *before* `base_messages` is built from
        // the compacted conversation, so honoring `keep_recent: false` there would delete the
        // request the model is about to answer and then answer the summary instead - the user's
        // words gone from the window, and the reply addressed to whatever "next step" the summary
        // happened to name.
        //
        // Phrased as a property of the conversation rather than a check on the origin, so a future
        // call site cannot reintroduce it by picking a different one.
        let keep_recent = keep_recent
            || messages
                .as_slice()
                .last()
                .is_some_and(|last| last.role == Role::User && !has_tool_results(&last.content));

        // Split into a head to summarize and a recent tail to keep verbatim. The tail is the
        // largest recent suffix that fits a token budget (~10% of the window, capped), snapped back
        // to a clean user boundary so tool_use/tool_result pairs are never orphaned.
        let (to_summarize, to_keep) = if keep_recent {
            let keep_budget = compaction_tail_budget(self.context_window());
            compute_compaction_split(messages.as_slice(), keep_budget)
        } else {
            // Keeping nothing means the summary has to cover everything, tail included. Only the
            // checkpoint turn is in a position to ask for this, because it read the whole
            // conversation; the summarizer is handed the head alone and would drop the rest on the
            // floor. Honored on the summarizer path anyway by widening what it summarizes.
            (messages.as_slice().to_vec(), Vec::new())
        };

        // The first of two places a fired token ends an automatic compaction; the second, after
        // the summary is built, catches an interrupt that lands inside the summarizer. This one is
        // for a token fired before or during the checkpoint, which answers it with `Ok(None)` and
        // would otherwise send the conversation to the summarizer: a full request paid for and
        // then thrown away by the check below. Same exception for `Manual`, and for the same
        // reason: `/compact` is itself the thing asked for, so an interrupt there ends the
        // checkpoint and falls back rather than abandoning the request.
        if request.origin != CompactOrigin::Manual && cancellation.is_cancelled() {
            report_checkpoint_memories(&memories_written);
            return Err(MekaError::Interrupted);
        }

        // `memories_written` is deliberately *not* re-read from the checkpoint here: the
        // accumulator above holds what actually ran, including on the fallback path where the
        // checkpoint half-completed and then failed.
        let (summary_text, source) = match checkpoint {
            Some(checkpoint) => (checkpoint.summary, checkpoint.source),
            None => (
                self.summarize_via_provider(&request, to_summarize, &cancellation)
                    .await?,
                CompactSource::Summarizer,
            ),
        };

        // Build post-compact context: environment, todos, scratchpad inventory.
        let post_context = self.build_post_compact_context(session_id).await;

        let mut context_message =
            format!("[Conversation summary from session compaction]\n\n{summary_text}");
        if !post_context.is_empty() {
            context_message.push_str(&format!("\n\n[Post-compaction context]\n\n{post_context}"));
        }
        // Behavioral directive (always last, most salient): pick the work back up rather than
        // narrate the summary. Without it, the turn after an auto-compaction tends to open with
        // "Based on the summary, I'll continue..." preambles that waste output and add nothing.
        context_message.push_str(
            "\n\n[Continue the work directly from the summary above. Do not acknowledge or recap \
             this summary; resume as if the conversation had not been interrupted.]",
        );

        // Snapshot the deferred-tool active set BEFORE compaction so the `CompactBoundary` event
        // carries it forward; otherwise tools the model loaded pre-compaction would silently drop
        // out of the active set on the next turn.
        //
        // Read from the events, like the per-turn active set, and not from the materialized slice.
        // A slice scan can only see `load_tool` exchanges still standing in the current view, and
        // two things routinely take them out of it. `DegradeTier::ToolExchanges` empties a refused
        // call in place, so its `input` no longer names anything and its result is marked
        // `is_error`; a *previous* compaction replaced everything before it with a summary, which
        // names nothing at all. Either way the snapshot came out short, `prune_compacted_events`
        // then dropped the events that could have corrected it, and a tool the model had loaded
        // disappeared from its array mid-session -- while a resume, reading the full log off disk,
        // brought it back. `ToolRegistry::definitions_active` says this in its own doc comment; the
        // one production caller was not doing it.
        let loaded_tools_snapshot: std::collections::HashSet<String> =
            crate::tools::load_tool::extract_loaded_tool_names_from_events(messages.events())
                .into_iter()
                .collect();

        // The last point before the window is destroyed, and the one that catches an interrupt
        // arriving *inside* the summarizer, whose `provider.complete` takes no token at all.
        // Without this a stop lands mid-summary and the conversation is replaced by a summary
        // written without the agent -- the checkpoint it would have had is exactly what the fired
        // token skipped.
        //
        // Every origin but `Manual`, and the exception is the point rather than an oversight. A
        // compaction the *turn* asked for is incidental to work the user has just stopped, so
        // stopping it is what was meant. `/compact` is the opposite: the compaction is itself the
        // thing asked for, and an interrupt there ends the checkpoint and falls back to the
        // summarizer rather than abandoning the request. That is pinned by
        // `an_interrupt_ends_the_checkpoint_and_falls_back`, and is why this cannot simply test
        // the token.
        //
        // Memories the checkpoint already wrote stay written; they are durable the moment they run
        // and are not this call's to undo.
        if request.origin != CompactOrigin::Manual && cancellation.is_cancelled() {
            // Reported before returning for the same reason the success path reports it: a
            // checkpoint that wrote memories and then hit the interrupt has left durable,
            // instance-scoped notes on disk, and every automatic path discards the outcome. The
            // `info!` below is unreachable from here, so without this the writes leave no trace at
            // any verbosity.
            report_checkpoint_memories(&memories_written);
            return Err(MekaError::Interrupted);
        }

        let summary_user_message = Message::user(&context_message);
        messages.replace_for_compaction(
            summary_user_message,
            to_keep.clone(),
            loaded_tools_snapshot,
        );

        // Persist the new compaction-boundary event and the re-appended tail. Pre-compaction rows
        // stay in the DB unchanged; the event log on disk grows append-only.
        let boundary_event = messages
            .events()
            .iter()
            .rev()
            .find(|e| matches!(e, crate::conversation::Event::CompactBoundary { .. }))
            .cloned()
            .ok_or_else(|| {
                MekaError::Internal(
                    "compact boundary missing after replace_for_compaction".to_string(),
                )
            })?;
        let replaced_count = match &boundary_event {
            crate::conversation::Event::CompactBoundary { replaced_count, .. } => *replaced_count,
            // Unreachable: the `find` above matched on this variant. Zero rather than a panic
            // because a miscounted advisory figure is not worth failing a compaction over.
            _ => 0,
        };
        // One transaction, for the reason `save_events_atomic` exists: a boundary that commits and
        // a tail write that then fails leaves the database holding a *valid* boundary with a
        // truncated tail, which puts those messages permanently outside the materialised view of
        // every future load. Silent, unrecoverable, and reported to the caller as a failure. The
        // whole rewrite is one unit or none of it is.
        let mut compaction_events = Vec::with_capacity(to_keep.len() + 1);
        compaction_events.push(boundary_event);
        compaction_events.extend(
            to_keep
                .iter()
                .map(|message| crate::conversation::Event::Append(message.clone())),
        );
        if let Err(error) = self
            .store
            .save_events_atomic(session_id, compaction_events)
            .await
        {
            // Put the conversation back. The rewrite above already happened in memory, so without
            // this the caller is told the compaction failed while the model goes on reasoning from
            // a summary the database has never heard of -- and `GET /messages`, reading the DB,
            // still serves the full history with `revision` unmoved. `POST /rewind` guards the
            // same hazard with `pop_repair`; this is the compaction-shaped half of it.
            messages.pop_compaction();
            return Err(error);
        }

        // Pre-boundary events are now fully superseded and already persisted; drop them so the
        // in-memory log doesn't grow unbounded across repeated compactions.
        messages.prune_compacted_events();

        // Compaction rewrites the conversation, so a length recorded against the old one no longer
        // identifies any particular message. Cleared here rather than at the call sites so
        // `/compact` and both auto-compact paths are covered by construction.
        self.last_accepted_len
            .store(LAST_ACCEPTED_UNKNOWN, std::sync::atomic::Ordering::Relaxed);

        // The model's view of which files it has read is reset by the summary; drop the
        // read-tracker so `edit_file` re-reads rather than trusting a pre-compaction read (also
        // bounds its growth).
        //
        // Since `context_compact` began draining mid-turn this also forgets reads the *current*
        // turn made, so an `edit_file` after a compaction it asked for is refused until the file is
        // read again. Kept deliberately: whether the read survived depends on where the kept tail
        // was cut, and re-reading costs a call where trusting a read that fell out of the window
        // costs a blind edit.
        self.tool_registry.clear_read_tracker().await;

        // Same reasoning for the tool/skill/MCP picture: the turns that carried it are now behind
        // the boundary and may have been summarized away, so forget what the model was told and let
        // the next turn re-state it in full. Compaction re-caches the conversation anyway, so the
        // extra tokens cost nothing that wasn't already spent.
        *self.last_rendered_world.write().await = None;

        // And the same for the schema advisories. Each one records "the model has already been
        // shown how this tool's arguments go wrong", which was true of a conversation the summary
        // has just replaced. Left standing, a model that repeats the mistake after a boundary gets
        // no hint for the rest of the session.
        crate::sync::lock(&self.schema_advisories_sent).clear();

        // Seed the live context gauge with an estimate of the compacted working set so `/status`
        // (and the prompt indicator) immediately reflect the smaller size; the next real turn
        // overwrites it with the exact provider-reported total.
        self.cells.context_tokens.store(
            crate::tokens::estimate_messages(messages.as_slice()),
            std::sync::atomic::Ordering::Relaxed,
        );

        report_checkpoint_memories(&memories_written);

        // One more generation of remove from the original turns. Left alone when the count has not
        // been read yet, so the lazy seed below still picks up the true figure including this one.
        let generation = self
            .compaction_generation
            .load(std::sync::atomic::Ordering::Relaxed);
        if generation != GENERATION_UNKNOWN {
            self.compaction_generation.store(
                generation.saturating_add(1),
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        // Announced after everything is persisted and the counters are settled, so a frontend
        // acting on it (an SSE client refetching `/messages`) cannot observe a half-applied
        // compaction. Every trigger reaches here, including the automatic ones nobody asked for,
        // which are exactly the ones a remote client would otherwise never learn about.
        // Read back rather than reusing the counter above: when the cache was `GENERATION_UNKNOWN`
        // the bump was skipped, and only this call goes to the database for the true figure, which
        // now includes the boundary just saved.
        let reported_generation = self.compaction_generation(session_id).await;
        self.cells
            .frontend
            .emit(FrontendEvent::Compacted {
                source: match source {
                    CompactSource::Checkpoint => "checkpoint",
                    CompactSource::CheckpointText => "checkpoint_text",
                    CompactSource::Summarizer => "summarizer",
                },
                replaced_count,
                generation: reported_generation,
            })
            .await;

        Ok(CompactOutcome {
            source,
            memories_written,
            // The outcome, not the intent: `compute_compaction_split` yields an empty tail whenever
            // the snapped boundary lands below `MIN_SUMMARIZE`, which is routine in a session whose
            // user turns are separated by long tool runs. Reporting the request would tell the user
            // their recent turns survived on exactly the occasions they did not.
            kept_recent: !to_keep.is_empty(),
        })
    }

    /// Who a compaction's requests are for: the turn that asked, or an id minted for a host-driven
    /// `/compact`, which answers no prompt of its own.
    fn compaction_attribution(&self, request: &CompactRequest) -> crate::provider::Attribution {
        crate::provider::Attribution {
            subagent: self.role.is_worker(),
            prompt_id: Some(request.prompt_id.unwrap_or_else(Uuid::new_v4)),
            previous_request: Some(Arc::clone(&self.previous_request)),
        }
    }

    /// Let the agent summarize itself, saving anything durable on the way past.
    ///
    /// Returns `None` when the turn produced nothing usable, which is the caller's cue to fall back
    /// to [`Self::summarize_via_provider`].
    ///
    /// Three things make this better than the standalone summarizer, and all three come from it
    /// being an ordinary turn rather than a special one:
    ///
    /// - It runs on the agent's *real* system prompt, so its persona, user instructions and memory
    ///   index are all present. The checkpoint instruction rides an appended user message rather
    ///   than replacing that prompt.
    /// - It has tools, so the moment information is about to be destroyed is finally a moment the
    ///   agent can act in. `memory_write` is the point of the exercise.
    /// - It sees full text (only images are stripped), so it judges what actually happened rather
    ///   than a head-and-tail excerpt of it.
    ///
    /// Cancellable through the caller's token. A bare `CancellationToken::new()` here would be a
    /// token with no signal source, which `run_turn_interruptible` documents as silently swallowing
    /// Ctrl+C - and the checkpoint is the longest thing compaction does: up to
    /// `CHECKPOINT_MAX_ITERATIONS` full-conversation calls, plus a prompt, with approvals on, that
    /// blocks until a human answers.
    pub(super) async fn run_checkpoint_turn(
        &self,
        request: &CompactRequest,
        session_id: Option<Uuid>,
        messages: &[Message],
        cancellation: CancellationToken,
        // Accumulated in the caller's buffer rather than returned, so a checkpoint that writes a
        // memory and *then* fails or is canceled still reports what landed on disk.
        memories_written: &mut Vec<String>,
    ) -> Result<Option<Checkpoint>> {
        let attribution = self.compaction_attribution(request);
        let slot: crate::tools::context::SubmissionSlot = Arc::new(std::sync::Mutex::new(
            None::<crate::tools::context::Submission>,
        ));
        let tools = self.tool_registry.checkpoint_tools(
            self.cells.permission.get(),
            self.cells.permission.approvals(),
            Arc::clone(&slot),
        );
        let definitions: Vec<ToolDefinition> = tools.iter().map(|tool| tool.definition()).collect();
        let by_name: std::collections::HashMap<String, Arc<dyn crate::tools::Tool>> = tools
            .into_iter()
            .map(|tool| (tool.definition().name, tool))
            .collect();

        let system_prompt = match &self.options.system_prompt_override {
            Some(prompt) => prompt.clone(),
            None => prompt::build_system_prompt(
                self.options.sandboxed_shell,
                self.options.user_instructions.as_deref(),
            ),
        };

        // Bounded by the same window a normal turn uses. Without this the checkpoint would be the
        // largest request meka ever sends: `context_messages` defaults to 200, and the reactive
        // trigger means "the last 200-message request already filled 80% of the window", so handing
        // the whole log over invites an overflow whose only trace is a warn line and a silent
        // fallback - the checkpoint quietly doing nothing in exactly the long sessions it exists
        // for.
        let mut checkpoint_messages: Vec<Message> =
            truncate_messages_for_context(messages, self.options.context_messages);
        for message in &mut checkpoint_messages {
            strip_images(&mut message.content);
        }

        // Deliver the instruction as a trailing text block on an existing user message when the
        // conversation already ends with one, and only otherwise as a message of its own.
        //
        // `CompactOrigin::Proactive` is why: it fires *after* this turn's user message is appended
        // (`run_turn`, the `messages.append(user_message)` above the pre-send check), so blindly
        // pushing would produce two consecutive user turns. Anthropic rejects that, and the failure
        // is near-silent - `compact_session` catches the error and falls back to the summarizer -
        // so the proactive trigger would quietly never checkpoint at all, which is exactly the kind
        // of degradation that never shows up in a test.
        let instruction = checkpoint_instruction(request);
        match checkpoint_messages.last_mut() {
            Some(last) if last.role == Role::User => {
                last.content.push(ContentBlock::Text { text: instruction });
            }
            _ => checkpoint_messages.push(Message::user(instruction)),
        }

        let mut last_text = String::new();

        for _ in 0..CHECKPOINT_MAX_ITERATIONS {
            // Checked per round as well as inside the tools, so an interrupt ends the checkpoint at
            // the next boundary instead of running out the whole iteration budget. Returning `None`
            // hands the caller to the summarizer, which is the right outcome: the user asked for
            // this to stop, not for the compaction to fail the turn.
            if cancellation.is_cancelled() {
                tracing::warn!("checkpoint turn interrupted; summarizing instead");
                return Ok(None);
            }
            let crate::provider::Completion {
                message: assistant_message,
                usage,
                notices,
                ..
            } = complete_with_retry(
                &self.provider(),
                CompletionRequest::new(&system_prompt, &checkpoint_messages, &definitions)
                    .attributed(attribution.clone()),
                &cancellation,
            )
            .await?;
            self.session_stats.record_untracked_tokens(&usage);
            for notice in notices {
                self.forward_notice(notice).await;
            }

            let text = assistant_message.text_content();
            if !text.trim().is_empty() {
                last_text = text;
            }

            let tool_uses: Vec<(String, String, serde_json::Value)> = assistant_message
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolUse { id, name, input } => {
                        Some((id.clone(), name.clone(), input.clone()))
                    }
                    _ => None,
                })
                .collect();
            if tool_uses.is_empty() {
                break;
            }

            checkpoint_messages.push(assistant_message);
            let mut results = Vec::with_capacity(tool_uses.len());
            for (tool_use_id, name, input) in tool_uses {
                let output = match by_name.get(&name) {
                    // Approvals put a call above the level to the user, and a checkpoint's writes
                    // are actions: `memory_write` overwrites an existing note in place, durably and
                    // instance-wide. Dispatching straight to `run_tool` would make the checkpoint
                    // the one place that silently ignores the level, and invisibly, since the loop
                    // emits no tool-call indicators either. The door is `admit_tool_call`, the one
                    // dispatch uses, so the approvals switch is honored here too. `context_replace`
                    // is exempt, for the same reason it bypasses the permission filter in
                    // `checkpoint_tools`: it performs no action, it hands the summary back to the
                    // caller. Prompting for it would ask the user to approve the checkpoint's own
                    // conclusion, and a denial would silently discard the summary and drop the
                    // whole compaction to the fallback summarizer.
                    Some(tool) => {
                        let schema = tool.definition().parameters;
                        // A checkpoint tool runs inline whatever the model asked; the detach flag
                        // is stripped like everywhere else and not honored. Read here, at the
                        // enforcement site, like `resolve_and_execute_tool` does: the checkpoint is
                        // up to eight round trips long, and a user who cycles the permission during
                        // it means the next call, not the last one. The set offered above was
                        // filtered at the level the checkpoint began with; this is what stops a
                        // call that set admitted from running after the level dropped.
                        let required = self
                            .tool_registry
                            .required_permission_for(&name)
                            .unwrap_or_else(|| tool.required_permission());
                        let admission = if name == "context_replace" {
                            super::dispatch::Admission::Run
                        } else {
                            super::dispatch::admit_tool_call(
                                &name,
                                required,
                                self.cells.permission.get(),
                                self.cells.permission.approvals(),
                                tool.runs_outside_confinement(),
                            )
                        };
                        match (
                            crate::tools::admit_arguments(&name, &input, &schema),
                            admission,
                        ) {
                            (Err(refusal), _) => refusal,
                            (Ok(_), super::dispatch::Admission::Refuse(refusal)) => refusal,
                            (Ok((input, _detach)), admission) => {
                                if matches!(admission, super::dispatch::Admission::Ask)
                                    && let Some(denial) = self
                                        .request_approval(
                                            &name,
                                            &input,
                                            false,
                                            &schema,
                                            &cancellation,
                                        )
                                        .await
                                {
                                    denial
                                } else {
                                    Self::run_tool(
                                        tool.as_ref(),
                                        &input,
                                        crate::tools::ToolContext {
                                            session_id,
                                            tool_call_id: Some(tool_use_id.clone()),
                                            prompt_id: attribution.prompt_id,
                                            frontend: Arc::clone(&self.cells.frontend),
                                            cancellation: cancellation.clone(),
                                        },
                                    )
                                    .await
                                }
                            }
                        }
                    }
                    // Names the constraint rather than reporting the tool as missing, which would
                    // read as "meka has no such tool" and invite the model to look for a synonym.
                    None => crate::tools::ToolOutput::text(
                        format!(
                            "'{name}' is not available during a checkpoint. A checkpoint can save what \
                             already happened, not do more work. Save what must last, then call \
                             `context_replace`."
                        ),
                        true,
                    ),
                };
                // Observed, never self-reported: a derived list cannot disagree with what landed on
                // disk. Only successful writes count.
                if name == "memory_write"
                    && !output.is_error
                    && let Some(memory) = input["name"].as_str()
                {
                    memories_written.push(memory.to_string());
                }
                results.push(ContentBlock::ToolResult {
                    tool_use_id,
                    content: bound_checkpoint_result(output.content),
                    is_error: output.is_error,
                });
            }
            checkpoint_messages.push(Message {
                role: Role::User,
                content: results,
            });

            if crate::sync::lock(&slot).is_some() {
                break;
            }
        }

        let submission = crate::sync::lock(&slot).take();

        // Tier 1: the tool was called, which is the path everything else is a hedge against.
        if let Some(submission) = submission {
            return Ok(Some(Checkpoint {
                summary: submission.summary,
                source: CompactSource::Checkpoint,
                keep_recent: submission.keep_recent,
            }));
        }

        // Tier 2. `Provider::complete` carries no `tool_choice` on any backend, so the call cannot
        // be forced and a model that summarized in prose instead has still done the work.
        let last_text = last_text.trim();
        if !last_text.is_empty() {
            tracing::warn!(
                "checkpoint turn ended without calling context_replace; using its closing text"
            );
            return Ok(Some(Checkpoint {
                summary: last_text.to_string(),
                source: CompactSource::CheckpointText,
                // Prose carries no answer to this, so take the safe direction explicitly rather
                // than returning `None`: `None` defers to the *caller's* `keep_recent`, and a
                // `context_compact(keep_recent: false)` would then discard the tail on the
                // strength of a summary the model never actually submitted.
                keep_recent: Some(true),
            }));
        }

        tracing::warn!("checkpoint turn produced no summary; falling back to the summarizer");
        Ok(None)
    }

    /// Summarize `to_summarize` in one standalone call that carries no tools and none of the
    /// agent's own identity.
    ///
    /// This is the original compaction mechanism, kept for the two cases the checkpoint turn cannot
    /// serve: [`CompactOrigin::Emergency`], where the provider has already refused a request this
    /// size, and any checkpoint that fails or comes back empty. Both want the same thing, a call
    /// deliberately smaller than the conversation, which is what stripping images and truncating
    /// long blocks buys.
    pub(super) async fn summarize_via_provider(
        &self,
        request: &CompactRequest,
        to_summarize: Vec<Message>,
        // The caller's, so a retry between attempts is interruptible for the same reason the
        // checkpoint's rounds are. Not used for anything else here: the call itself is one
        // `complete`, which this cannot reach inside.
        cancellation: &CancellationToken,
    ) -> Result<String> {
        let mut system_prompt = String::from(
            "You are a conversation summarizer. Produce a structured summary \
             that will replace the conversation. Write in second person \
             (\"You were working on...\").\n\n\
             Cover these sections (skip any that don't apply):\n\n\
             1. **Primary task**: What the user asked for and the overall goal.\n\
             2. **Current state**: What has been completed, what is in progress, what remains.\n\
             3. **Key files**: Files read, created, or modified (list paths).\n\
             4. **Key decisions**: Important choices made and their rationale.\n\
             5. **Errors and fixes**: Problems encountered and how they were resolved.\n\
             6. **Standing commitments**: Anything promised to the user but not yet delivered, \
             and any deadline or follow-up still outstanding.\n\
             7. **User preferences and constraints**: Feedback or corrections about how to \
             work. Preserve any security-relevant instructions verbatim (sensitive files or \
             data to avoid, operations that must not be performed, secret-handling rules) so \
             they keep applying after compaction.\n\
             8. **All user requests**: Every distinct request the user made, in order, so none \
             of their intent is lost.\n\
             9. **Next step**: The immediate next action. If a task was mid-flight, quote the \
             user's most recent request verbatim so the work does not drift.",
        );
        // Last, so it outranks the standing sections it may contradict ("drop the debugging").
        if let Some(instructions) = &request.instructions {
            system_prompt.push_str(&format!(
                "\n\nThe following instructions were given for this specific compaction and take \
                 precedence over the sections above:\n{instructions}"
            ));
        }

        // Clone and preprocess messages for the summarizer: strip images and truncate large text
        // blocks to avoid overwhelming the summary call.
        let mut compact_messages = to_summarize;
        for message in &mut compact_messages {
            strip_images_and_truncate(&mut message.content);
        }

        // Append a user message so the conversation ends with a user turn.
        compact_messages.push(Message::user(
            "Summarize this conversation into a concise context message.",
        ));

        // A summary is not worth reasoning over, so this one request turns thinking off. The
        // override travels with the request rather than living on the shared provider, so a
        // sibling sub-agent's in-flight turn keeps whatever thinking its profile configured.
        let crate::provider::Completion {
            message: summary_message,
            usage,
            notices,
            ..
        } = complete_with_retry(
            &self.provider(),
            CompletionRequest::new(&system_prompt, &compact_messages, &[])
                .attributed(self.compaction_attribution(request))
                .without_thinking(),
            cancellation,
        )
        .await?;
        self.session_stats.record_untracked_tokens(&usage);
        // Surface any provider notices from the summary call (e.g. image redaction on a very large
        // compaction window). Rare in practice; emitting before we mutate the conversation keeps
        // the user-facing order stable.
        for notice in notices {
            self.forward_notice(notice).await;
        }

        let summary_text = summary_message.text_content();
        if summary_text.is_empty() {
            return Err(MekaError::Provider(
                "LLM returned an empty summary".to_string(),
            ));
        }
        Ok(summary_text)
    }

    /// How many compactions this session has been through, reading the database once and caching.
    ///
    /// A read that fails reports zero rather than propagating: an unknown generation is a missing
    /// line in a context block, not a reason to fail a turn.
    pub(super) async fn compaction_generation(&self, session_id: Uuid) -> u64 {
        let cached = self
            .compaction_generation
            .load(std::sync::atomic::Ordering::Relaxed);
        if cached != GENERATION_UNKNOWN {
            return cached;
        }
        let counted = self
            .store
            .count_compactions(session_id)
            .await
            .unwrap_or_else(|error| {
                tracing::debug!("failed to count compactions: {error}");
                0
            });
        self.compaction_generation
            .store(counted, std::sync::atomic::Ordering::Relaxed);
        counted
    }

    pub(super) async fn build_post_compact_context(&self, session_id: Uuid) -> String {
        let permission = self.cells.permission.get();
        let todos = self.cells.todo_list.get();
        // Degrades rather than fails, since the summary is what keeps the session alive, but not
        // silently: an empty inventory tells the model it saved nothing, and it re-derives work it
        // already wrote down.
        let entries = match self.store.list_scratchpad_entries(session_id).await {
            Ok(entries) => entries,
            Err(error) => {
                tracing::warn!(
                    "post-compaction context omits the scratchpad inventory; failed to list it: \
                     {error}"
                );
                Vec::new()
            }
        };
        prompt::build_post_compact_context(
            permission,
            &todos,
            &entries,
            &self.cells.cwd.get(),
            &self.cells.roots.get(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        agent::tests::{
            agent_for_test, agent_with_registry_for_test, assistant_message, assistant_tool_use,
            tool_result_message, user_message,
        },
        permission::SharedPermission,
        provider::mock::{MockEvent, MockProvider, MockStopReason, text_round},
        session::{CompactOrigin, CompactRequest},
    };

    /// The attempt cap binds, and the counter that enforces it counts up.
    ///
    /// The sibling of the cancellation test: that one never reaches `retries += 1`, so the counter
    /// itself was unguarded and two mutants survived on it. Both are live failures rather than
    /// arithmetic trivia. `*=` leaves the count at zero forever, so a provider that keeps failing
    /// is retried until [`crate::provider::retry::RETRY_BUDGET`] runs out instead of three times --
    /// five minutes of a user waiting, and up to a completion billed per attempt. `-=` underflows
    /// on the first retry and panics the turn.
    ///
    /// Four rounds against a cap of three attempts: the fourth would succeed, so a run that
    /// reaches it is exactly the runaway being guarded against, and `completions()` says which
    /// happened. Virtual time keeps the 1s and 2s of backoff free; `should_retry_provider_error`
    /// measures its budget on `std::time::Instant`, which `start_paused` does not move, so the
    /// budget cannot fire first and steal the assertion.
    #[tokio::test(start_paused = true)]
    async fn compaction_stops_retrying_at_the_attempt_cap() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let failure = || {
            vec![MockEvent::FailRetryable {
                message: "529 overloaded".to_string(),
                retry_after_secs: None,
            }]
        };
        let mock = Arc::new(MockProvider::from_rounds(vec![
            failure(),
            failure(),
            failure(),
            vec![
                MockEvent::Text {
                    text: "a summary the cap should never let us reach".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let provider: Arc<dyn Provider> = mock.clone();

        let error = complete_with_retry(
            &provider,
            CompletionRequest::new("system", &[], &[]),
            &CancellationToken::new(),
        )
        .await
        .expect_err("three failures spend the cap, so the fourth round is never asked for");
        assert!(
            matches!(error, MekaError::RetryableProvider { .. }),
            "the provider's own last refusal is what the caller has to see: {error}"
        );
        assert_eq!(
            mock.completions().len(),
            usize::try_from(crate::provider::retry::MAX_PROVIDER_RETRIES).unwrap_or(usize::MAX) + 1,
            "one initial attempt plus MAX_PROVIDER_RETRIES, and not one more"
        );
    }

    /// The wait between compaction's retries races the caller's token.
    ///
    /// Giving compaction a retry loop is what made this reachable: before it, the call was one
    /// `complete` with no sleep in it, so there was nothing for a Ctrl+C to sit through.
    /// [`Agent::run_checkpoint_turn`]'s own doc says it is cancellable through the caller's token,
    /// and it checks that per round -- but a bare `tokio::time::sleep` inside the round would sit
    /// out a `Retry-After` of up to [`crate::provider::retry::RETRY_AFTER_CAP`] first, once per
    /// attempt, once per iteration. That is compaction becoming the one provider call the user
    /// cannot stop.
    ///
    /// The hint is five seconds and the cancel lands a tenth of a second in, so the fix returns
    /// almost at once and its absence sleeps: a neutered `select!` takes the full five and then
    /// answers `Ok` from the second round, failing both assertions rather than hanging.
    ///
    /// Canceled *during* the wait rather than before the call. Starting canceled would prove
    /// less than it looks: `compact_session` reaches this helper with an already-canceled token on
    /// its ordinary interrupt path, and that call is meant to go through, so a token read before
    /// the first attempt would be a behavior change rather than a stricter test.
    #[tokio::test]
    async fn a_retry_wait_ends_when_the_turn_is_cancelled() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider: Arc<dyn Provider> = Arc::new(MockProvider::from_rounds(vec![
            vec![MockEvent::FailRetryable {
                message: "overloaded".to_string(),
                retry_after_secs: Some(5),
            }],
            vec![
                MockEvent::Text {
                    text: "a summary nobody asked for any more".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));

        let cancellation = CancellationToken::new();
        tokio::spawn({
            let cancellation = cancellation.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                cancellation.cancel();
            }
        });
        let started = std::time::Instant::now();
        let error = complete_with_retry(
            &provider,
            CompletionRequest::new("system", &[], &[]),
            &cancellation,
        )
        .await
        .expect_err("a canceled turn does not wait out the provider's hint");

        assert!(
            matches!(error, MekaError::Interrupted),
            "the user stopped it, so that is what the caller has to hear rather than the \
             provider's complaint: {error}"
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "the wait has to end on the token, not run to the hint: {:?}",
            started.elapsed()
        );
    }

    /// A conversation too short to be worth a boundary is summarized whole, except for the
    /// request the model is about to answer. `compact_session` forces `keep_recent` for a trailing
    /// prompt, and the split honored that only above `MIN_SUMMARIZE`: below it the prompt went
    /// into the summary and the model answered a paraphrase of the user's words.
    #[test]
    fn compaction_split_small_summarizes_all_but_a_trailing_prompt() {
        let messages = vec![user_message("a"), assistant_message("b"), user_message("c")];
        let (head, tail) = compute_compaction_split(&messages, 10_000);
        assert_eq!(head.len(), 2);
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].text_content(), "c");

        // Answered, so nothing is owed verbatim.
        let messages = vec![
            user_message("a"),
            assistant_message("b"),
            user_message("c"),
            assistant_message("d"),
        ];
        let (head, tail) = compute_compaction_split(&messages, 10_000);
        assert_eq!(head.len(), 4);
        assert!(tail.is_empty());

        // A trailing tool result belongs to the call before it; alone it would be an orphan.
        let messages = vec![
            user_message("a"),
            assistant_tool_use(),
            tool_result_message(),
        ];
        let (head, tail) = compute_compaction_split(&messages, 10_000);
        assert_eq!(head.len(), 3);
        assert!(tail.is_empty());
    }

    #[test]
    fn compaction_split_keeps_recent_tail_within_budget() {
        let mut messages = Vec::new();
        for i in 0..6 {
            messages.push(user_message(&format!("user {i}")));
            messages.push(assistant_message(&format!("assistant {i}")));
        }
        let (head, tail) = compute_compaction_split(&messages, 30);
        // The split partitions the whole conversation.
        assert_eq!(head.len() + tail.len(), messages.len());
        // A small budget keeps only a recent slice, leaving a real head to summarize.
        assert!(head.len() >= 4);
        assert!(!tail.is_empty() && tail.len() < messages.len());
        // The kept window starts on a clean user boundary.
        assert_eq!(tail[0].role, Role::User);
        assert!(!has_tool_results(&tail[0].content));
    }

    #[test]
    fn compaction_split_does_not_orphan_tool_results() {
        let messages = vec![
            user_message("first"),
            assistant_message("r1"),
            user_message("second"),
            assistant_message("r2"),
            user_message("third"),
            assistant_tool_use(),
            tool_result_message(),
            assistant_message("final"),
        ];
        // A budget that naively cuts inside the assistant(tool_use)->user(tool_result) chain must
        // snap back to the user boundary before it.
        let (head, tail) = compute_compaction_split(&messages, 20);
        assert_eq!(head.len() + tail.len(), messages.len());
        assert_eq!(tail[0].role, Role::User);
        assert!(!has_tool_results(&tail[0].content));
    }

    // Compaction strategy selection and the fallback ladder.
    //
    // The ladder exists because `Provider::complete` carries no `tool_choice` on any backend, so
    // `context_replace` cannot be forced. Each rung is reachable in production and so is asserted
    // here.
    fn replace_round(summary: &str, keep_recent: Option<bool>) -> Vec<MockEvent> {
        let mut input = serde_json::json!({ "summary": summary });
        if let Some(keep_recent) = keep_recent {
            input["keep_recent"] = serde_json::json!(keep_recent);
        }
        vec![
            MockEvent::ToolUseStart {
                id: "call-1".to_string(),
                name: "context_replace".to_string(),
            },
            MockEvent::ToolUseEnd { input },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::ToolUse,
            },
        ]
    }

    /// Ten alternating turns, each large enough that the whole conversation overruns the tail
    /// budget.
    ///
    /// Size is load-bearing, not incidental: `compute_compaction_split` grows the tail until the
    /// budget stops it, so a conversation that fits entirely inside the budget is summarized whole
    /// with *no* tail at all. A small fixture would make every `keep_recent` assertion below pass
    /// for the wrong reason.
    fn conversation() -> Conversation {
        let body = "x".repeat(4_000);
        let mut conversation = Conversation::new();
        for index in 0..5 {
            conversation.append(Message::user(format!("user {index} {body}")));
            conversation.append(Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: format!("assistant {index} {body}"),
                }],
            });
        }
        conversation
    }

    /// A tool that exists only to be found by name, so a checkpoint call resolves and reaches
    /// the dispatch path under test. It records what each call was told: the session it
    /// serves and the arguments that reached it.
    type SeenCall = (Option<Uuid>, serde_json::Value);

    struct StubTool {
        name: String,
        calls: Arc<std::sync::Mutex<Vec<SeenCall>>>,
    }

    #[async_trait::async_trait]
    impl crate::tools::Tool for StubTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name.clone(),
                description: "stub".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
                title: None,
                annotations: None,
                meta: None,
            }
        }

        fn required_permission(&self) -> crate::permission::Permission {
            crate::permission::Permission::Read
        }

        async fn execute(
            &self,
            input: serde_json::Value,
            context: crate::tools::ToolContext,
        ) -> Result<crate::tools::ToolOutput> {
            self.calls
                .lock()
                .expect("calls")
                .push((context.session_id, input));
            Ok(crate::tools::ToolOutput::text(
                "stub ran".to_string(),
                false,
            ))
        }
    }

    /// A checkpoint tool is told which session it serves and receives its arguments through
    /// the same admission as an inline call. Before the context traveled with the call, the
    /// checkpoint installed only the frontend, so an MCP tool called from a checkpoint carried
    /// no `meka/sessionId`; and it handed the model's arguments over unexamined, so a
    /// `background: true` reached a tool that never advertised the key.
    #[tokio::test]
    async fn a_checkpoint_tool_is_told_its_session_and_gets_admitted_arguments() {
        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![
                MockEvent::ToolUseStart {
                    id: "call-1".to_string(),
                    name: "memory_write".to_string(),
                },
                MockEvent::ToolUseEnd {
                    input: serde_json::json!({"name": "note", "background": true}),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::ToolUse,
                },
            ],
            replace_round("summary", None),
        ]));
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let registry = crate::tools::ToolRegistry::new();
        registry
            .register(Arc::new(StubTool {
                name: "memory_write".to_string(),
                calls: calls.clone(),
            }))
            .expect("register stub");
        let (agent, store) = agent_with_registry_and_checkpoint(provider, registry).await;
        let mut messages = conversation();
        agent.cells().session_id.set(
            store
                .create_session(None, "test-profile".to_string())
                .await
                .expect("create session"),
        );

        agent
            .compact_session(
                &mut messages,
                CompactRequest::new(CompactOrigin::Manual),
                CancellationToken::new(),
            )
            .await
            .expect("compaction");

        let session_id = agent.session_id();
        let calls = calls.lock().expect("calls");
        assert_eq!(calls.len(), 1, "the checkpoint dispatched the stub once");
        assert_eq!(
            calls[0].0, session_id,
            "the tool is told the session it serves"
        );
        assert_eq!(
            calls[0].1,
            serde_json::json!({"name": "note"}),
            "meka's own parameter is taken out at the checkpoint door too"
        );
    }

    async fn agent_with_registry_and_checkpoint(
        provider: Arc<dyn Provider>,
        registry: crate::tools::ToolRegistry,
    ) -> (Agent, Store) {
        let (mut agent, store) = agent_with_registry_for_test(provider, registry).await;
        agent.options.compact_checkpoint = true;
        agent.set_context_window_for_test(40_000);
        (agent, store)
    }

    async fn agent_with_checkpoint(
        provider: Arc<dyn Provider>,
        checkpoint: bool,
    ) -> (Agent, Store) {
        let (mut agent, store) = agent_for_test(provider).await;
        agent.options.compact_checkpoint = checkpoint;
        agent.set_context_window_for_test(40_000);
        (agent, store)
    }

    async fn compact(
        agent: &Agent,
        store: &Store,
        messages: &mut Conversation,
        request: CompactRequest,
    ) -> CompactOutcome {
        agent.cells().session_id.set(
            store
                .create_session(None, "test-profile".to_string())
                .await
                .expect("create session"),
        );
        agent
            .compact_session(messages, request, CancellationToken::new())
            .await
            .expect("compaction")
    }

    /// The summary is the one request that turns thinking off, and it says so on the request
    /// itself. A flag on the shared provider could not promise that: a sibling sub-agent's turn
    /// in flight over the same `Arc<dyn Provider>` would have lost its thinking too, and the
    /// guard that skipped the flag for sub-agents left their summaries paying for reasoning.
    #[tokio::test]
    async fn the_summary_turns_thinking_off_on_its_own_request_only() {
        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![
                MockEvent::Text {
                    text: "the summary".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
            vec![
                MockEvent::Text {
                    text: "a turn after it".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let (agent, store) = agent_with_checkpoint(provider.clone(), false).await;
        let mut messages = conversation();
        agent.cells().session_id.set(
            store
                .create_session(None, "test-profile".to_string())
                .await
                .expect("create session"),
        );

        agent
            .compact_session(
                &mut messages,
                CompactRequest::new(CompactOrigin::Manual),
                CancellationToken::new(),
            )
            .await
            .expect("compaction");
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("carry on".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("a turn after compaction");

        assert_eq!(
            provider.completion_thinking(),
            vec![crate::provider::ThinkingOverride::Off],
            "the summary is the one request that asks for thinking off"
        );
        let streamed: Vec<_> = provider
            .streams()
            .into_iter()
            .map(|request| request.thinking)
            .collect();
        assert_eq!(
            streamed,
            vec![crate::provider::ThinkingOverride::Inherit],
            "the turn after it inherits the profile's setting"
        );
    }

    #[tokio::test]
    async fn checkpoint_summary_comes_from_the_tool_call() {
        let provider = Arc::new(MockProvider::from_rounds(vec![replace_round(
            "what I was doing",
            None,
        )]));
        let (agent, store) = agent_with_checkpoint(provider, true).await;
        let mut messages = conversation();

        let outcome = compact(
            &agent,
            &store,
            &mut messages,
            CompactRequest::new(CompactOrigin::Reactive),
        )
        .await;

        assert_eq!(outcome.source, CompactSource::Checkpoint);
        assert!(outcome.kept_recent);
        assert!(
            messages.len() > 1,
            "a tail should survive when keep_recent is left unset"
        );
        assert!(
            messages.as_slice()[0]
                .text_content()
                .contains("what I was doing"),
            "summary should be the tool argument, got {:?}",
            messages.as_slice()[0].text_content()
        );
    }

    /// Tier 2. The model summarized in prose instead of submitting, which is still the work
    /// done, so it is used rather than thrown away for a second model call.
    #[tokio::test]
    async fn checkpoint_falls_back_to_its_closing_text() {
        let provider = Arc::new(MockProvider::from_rounds(vec![text_round(
            "here is the state of things",
        )]));
        let (agent, store) = agent_with_checkpoint(provider, true).await;
        let mut messages = conversation();

        let outcome = compact(
            &agent,
            &store,
            &mut messages,
            CompactRequest::new(CompactOrigin::Reactive),
        )
        .await;

        assert_eq!(outcome.source, CompactSource::CheckpointText);
        assert!(
            messages.as_slice()[0]
                .text_content()
                .contains("here is the state of things")
        );
    }

    /// Tier 3. A checkpoint that produces neither a call nor text must not lose the conversation;
    /// the standalone summarizer takes the next round.
    #[tokio::test]
    async fn checkpoint_producing_nothing_falls_back_to_the_summarizer() {
        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![MockEvent::MessageEnd {
                stop_reason: MockStopReason::EndTurn,
            }],
            text_round("summarized separately"),
        ]));
        let (agent, store) = agent_with_checkpoint(provider, true).await;
        let mut messages = conversation();

        let outcome = compact(
            &agent,
            &store,
            &mut messages,
            CompactRequest::new(CompactOrigin::Reactive),
        )
        .await;

        assert_eq!(outcome.source, CompactSource::Summarizer);
        assert!(
            messages.as_slice()[0]
                .text_content()
                .contains("summarized separately")
        );
    }

    /// The emergency path runs after the provider refused the request for being too large. A
    /// checkpoint turn re-sends that same conversation, so it would be refused identically; the
    /// degraded summarizer is the only call that can still get through.
    #[tokio::test]
    async fn emergency_skips_the_checkpoint_even_when_it_is_enabled() {
        let provider = Arc::new(MockProvider::from_rounds(vec![text_round(
            "emergency summary",
        )]));
        let (agent, store) = agent_with_checkpoint(provider, true).await;
        let mut messages = conversation();

        let outcome = compact(
            &agent,
            &store,
            &mut messages,
            CompactRequest::new(CompactOrigin::Emergency),
        )
        .await;

        assert_eq!(outcome.source, CompactSource::Summarizer);
    }

    #[tokio::test]
    async fn disabled_checkpoint_uses_the_summarizer_on_every_origin() {
        for origin in [
            CompactOrigin::Reactive,
            CompactOrigin::Proactive,
            CompactOrigin::Manual,
            CompactOrigin::Requested,
        ] {
            let provider = Arc::new(MockProvider::from_rounds(vec![text_round("plain summary")]));
            let (agent, store) = agent_with_checkpoint(provider, false).await;
            let mut messages = conversation();

            let outcome = compact(&agent, &store, &mut messages, CompactRequest::new(origin)).await;

            assert_eq!(outcome.source, CompactSource::Summarizer, "{origin:?}");
        }
    }

    /// Turning the page: the summary is all that is left, with no verbatim tail behind it.
    #[tokio::test]
    async fn keep_recent_false_leaves_only_the_summary() {
        let provider = Arc::new(MockProvider::from_rounds(vec![replace_round(
            "the whole day",
            Some(false),
        )]));
        let (agent, store) = agent_with_checkpoint(provider, true).await;
        let mut messages = conversation();

        let outcome = compact(
            &agent,
            &store,
            &mut messages,
            CompactRequest::new(CompactOrigin::Requested),
        )
        .await;

        assert!(!outcome.kept_recent);
        assert_eq!(messages.len(), 1, "only the summary should remain");
    }

    /// `context_replace` knows more than the caller did, because it ran after reading the
    /// conversation, so its answer wins over the request's.
    #[tokio::test]
    async fn the_tools_tail_decision_overrides_the_requests() {
        let provider = Arc::new(MockProvider::from_rounds(vec![replace_round(
            "still need the recent turns",
            Some(true),
        )]));
        let (agent, store) = agent_with_checkpoint(provider, true).await;
        let mut messages = conversation();

        let outcome = compact(&agent, &store, &mut messages, CompactRequest {
            origin: CompactOrigin::Requested,
            instructions: None,
            keep_recent: Some(false),
            prompt_id: None,
        })
        .await;

        assert!(outcome.kept_recent);
        assert!(messages.len() > 1);
    }

    /// The proactive trigger fires *after* this turn's user message is appended, so a
    /// checkpoint that blindly pushed its instruction would send two consecutive user turns.
    /// Anthropic rejects that, and `compact_session` swallows the error into the summarizer
    /// fallback, so the damage would be a permanently-degraded trigger and one warn line.
    #[tokio::test]
    async fn the_checkpoint_instruction_never_creates_two_user_turns() {
        let recorded = Arc::new(MockProvider::from_rounds(vec![replace_round("ok", None)]));
        let (agent, store) =
            agent_with_checkpoint(Arc::clone(&recorded) as Arc<dyn Provider>, true).await;
        let mut messages = conversation();
        // Exactly the shape the proactive path compacts in: a user message on the end.
        messages.append(Message::user("the request that pushed us over"));

        compact(
            &agent,
            &store,
            &mut messages,
            CompactRequest::new(CompactOrigin::Proactive),
        )
        .await;

        // What the provider was actually handed, not a local reconstruction of it: an
        // assertion built from the test's own arithmetic passes just as happily when the
        // production path is reverted.
        let sent = recorded.completions();
        let sent = sent.first().expect("the checkpoint made a call");
        for pair in sent.windows(2) {
            assert!(
                !(pair[0].role == Role::User && pair[1].role == Role::User),
                "two consecutive user messages would be refused by the provider"
            );
        }
        let last = sent.last().expect("non-empty");
        assert!(
            last.text_content().contains("Checkpoint"),
            "the instruction must be the last thing the model reads"
        );
        assert!(
            last.text_content()
                .contains("the request that pushed us over"),
            "merging must not drop the message it merged into"
        );
    }

    /// `keep_recent: false` is a bet that the checkpoint saved what mattered. When the
    /// checkpoint never ran, the bet was never placed, so discarding the tail on top of a
    /// truncated summary would turn one failure into permanent data loss.
    #[tokio::test]
    async fn a_failed_checkpoint_does_not_honor_keep_recent_false() {
        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![MockEvent::Fail {
                message: "checkpoint call failed".to_string(),
            }],
            text_round("fallback summary"),
        ]));
        let (agent, store) = agent_with_checkpoint(provider, true).await;
        let mut messages = conversation();

        let outcome = compact(&agent, &store, &mut messages, CompactRequest {
            origin: CompactOrigin::Requested,
            instructions: None,
            keep_recent: Some(false),
            prompt_id: None,
        })
        .await;

        assert_eq!(outcome.source, CompactSource::Summarizer);
        assert!(
            outcome.kept_recent,
            "a failed checkpoint must not also cost the verbatim tail"
        );
        assert!(messages.len() > 1);
    }

    /// The checkpoint runs when the window is nearly full, so an unbounded tool result would
    /// overflow the very request that is supposed to shrink it. A normal turn spills oversized
    /// results to the scratchpad; the checkpoint loop is not a turn, so it truncates instead.
    #[test]
    fn checkpoint_tool_results_are_bounded_like_a_normal_turn() {
        use crate::conversation::ToolResultContent;

        let limit = crate::tools::scratchpad::MAX_INLINE_RESULT_BYTES;
        let bounded = bound_checkpoint_result(vec![ToolResultContent::Text {
            text: "x".repeat(limit * 3),
        }]);
        let ToolResultContent::Text { text } = &bounded[0] else {
            panic!("text in, text out");
        };
        assert!(text.len() < limit * 2, "still {} bytes", text.len());
        assert!(
            text.contains("truncated"),
            "the cut has to be visible to the model"
        );

        // Anything already within the limit is passed through untouched, so the common case
        // costs nothing and reads exactly as the tool wrote it.
        let small = bound_checkpoint_result(vec![ToolResultContent::Text {
            text: "short".to_string(),
        }]);
        let ToolResultContent::Text { text } = &small[0] else {
            panic!("text in, text out");
        };
        assert_eq!(text, "short");
    }

    /// A compaction must not inflate the turn count `/status` reports: it is work meka did on
    /// its own, not a turn the user asked for. The matching "the tokens *are* billed" half
    /// lives in `crate::stats`, which can observe a non-zero usage the mock cannot produce.
    #[tokio::test]
    async fn compaction_is_not_counted_as_a_turn() {
        let provider = Arc::new(MockProvider::from_rounds(vec![replace_round("done", None)]));
        let (agent, store) = agent_with_checkpoint(provider, true).await;
        let mut messages = conversation();

        let before = agent.session_stats.snapshot();
        compact(
            &agent,
            &store,
            &mut messages,
            CompactRequest::new(CompactOrigin::Manual),
        )
        .await;
        let after = agent.session_stats.snapshot();

        assert_eq!(
            after.turns, before.turns,
            "a compaction is not a turn the user asked for"
        );
        // The token half is asserted in `crate::stats`: `MockProvider` reports
        // `TokenUsage::default()`, so there is nothing here for a total to grow by.
    }

    /// The proactive trigger fires after this turn's user message is appended and before
    /// `base_messages` is rebuilt, so a `keep_recent: false` there would delete the request the
    /// model is about to answer, and it would answer the summary instead.
    #[tokio::test]
    async fn a_trailing_unanswered_request_is_never_discarded() {
        let provider = Arc::new(MockProvider::from_rounds(vec![replace_round(
            "everything is covered",
            Some(false),
        )]));
        let (agent, store) = agent_with_checkpoint(provider, true).await;
        let mut messages = conversation();
        messages.append(Message::user("refactor this to async"));

        let outcome = compact(
            &agent,
            &store,
            &mut messages,
            CompactRequest::new(CompactOrigin::Proactive),
        )
        .await;

        assert!(outcome.kept_recent, "the pending request must survive");
        let survived = messages
            .as_slice()
            .iter()
            .any(|message| message.text_content().contains("refactor this to async"));
        assert!(survived, "the user's unanswered request was compacted away");
    }

    /// Tier 2 never saw a `context_replace`, so it cannot know whether the tail is covered.
    /// Returning `None` deferred that to the caller, letting a `context_compact(keep_recent:
    /// false)` discard the tail on the strength of a summary the model never submitted.
    #[tokio::test]
    async fn the_text_fallback_keeps_the_tail_even_when_the_caller_asked_not_to() {
        let provider = Arc::new(MockProvider::from_rounds(vec![text_round(
            "now let me save the last one",
        )]));
        let (agent, store) = agent_with_checkpoint(provider, true).await;
        let mut messages = conversation();

        let outcome = compact(&agent, &store, &mut messages, CompactRequest {
            origin: CompactOrigin::Requested,
            instructions: None,
            keep_recent: Some(false),
            prompt_id: None,
        })
        .await;

        assert_eq!(outcome.source, CompactSource::CheckpointText);
        assert!(
            outcome.kept_recent,
            "a stray sentence must not become the whole context"
        );
        assert!(messages.len() > 1);
    }

    /// The checkpoint must never be the largest request meka sends. The reactive trigger means
    /// the last `context_messages`-bounded request already filled the window, so handing over
    /// the whole log would overflow and degrade to the summarizer precisely in the long
    /// sessions the checkpoint exists for.
    #[tokio::test]
    async fn the_checkpoint_respects_the_context_message_window() {
        let recorded = Arc::new(MockProvider::from_rounds(vec![replace_round("ok", None)]));
        let (mut agent, store) =
            agent_with_checkpoint(Arc::clone(&recorded) as Arc<dyn Provider>, true).await;
        agent.options.context_messages = Some(4);
        let mut messages = conversation();
        let full = messages.len();

        compact(
            &agent,
            &store,
            &mut messages,
            CompactRequest::new(CompactOrigin::Reactive),
        )
        .await;

        let sent = recorded.completions();
        let sent = sent.first().expect("the checkpoint made a call");
        assert!(
            sent.len() < full,
            "checkpoint sent {} of {full} messages; the window was not applied",
            sent.len()
        );
        // The cap, plus the appended instruction when it lands as its own message. Snapping to
        // a user boundary can only keep fewer, never more.
        assert!(
            sent.len() <= 5,
            "checkpoint sent {} messages against a limit of 4",
            sent.len()
        );
    }

    /// A checkpoint runs unattended and can call `memory_write`, which overwrites a note in
    /// place, durably and instance-wide. With approvals on, a call above the level is a
    /// question for the user, and the checkpoint has to ask it the way dispatch does rather
    /// than run the write because nobody was watching. At `none` every tool is above the
    /// level, so this is also the shape an `ask` session migrates to.
    #[tokio::test]
    async fn approvals_are_honored_inside_the_checkpoint() {
        use crate::frontend::{PermissionOutcome, testing::RecordingFrontend};

        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![
                MockEvent::ToolUseStart {
                    id: "call-1".to_string(),
                    name: "memory_write".to_string(),
                },
                MockEvent::ToolUseEnd {
                    input: serde_json::json!({"name": "note", "description": "d"}),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::ToolUse,
                },
            ],
            replace_round("summary after a refusal", None),
        ]));
        let frontend = Arc::new(RecordingFrontend::with_permission(PermissionOutcome::Deny));
        // A registry that actually holds `memory_write`, so the call resolves and reaches the
        // gate. Against an empty registry it would fall through to "not available during a
        // checkpoint" and the test would assert nothing.
        let registry = crate::tools::ToolRegistry::new();
        registry
            .register(Arc::new(StubTool {
                name: "memory_write".to_string(),
                calls: Default::default(),
            }))
            .expect("register stub");
        let (mut agent, store) = agent_with_registry_and_checkpoint(provider, registry).await;
        agent.cells.frontend = frontend.clone();
        agent.cells.permission = SharedPermission::new(
            crate::permission::Permission::None,
            crate::permission::EnabledPermissions::ALL,
        )
        .with_approvals(true);
        let mut messages = conversation();

        let outcome = compact(
            &agent,
            &store,
            &mut messages,
            CompactRequest::new(CompactOrigin::Manual),
        )
        .await;

        // The denial is a tool error, not a dead end: the checkpoint carries on and still
        // submits, so a refused write costs a note rather than the whole summary.
        assert_eq!(outcome.source, CompactSource::Checkpoint);
        assert_eq!(
            frontend.permission_requests(),
            vec!["memory_write".to_string()],
            "the write must be gated, and `context_replace` must not be"
        );
        assert!(
            outcome.memories_written.is_empty(),
            "a denied write must not be reported as written"
        );
    }

    /// The permission is read at each tool call, not once when the checkpoint began. A user
    /// who drops the level during the checkpoint means the next call, and a snapshot taken
    /// eight round trips earlier let `memory_write` run at `none`.
    #[tokio::test]
    async fn the_checkpoint_reads_the_permission_at_each_call() {
        let calls: Arc<std::sync::Mutex<Vec<SeenCall>>> = Default::default();
        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![
                // Long enough for the level to drop while the round is in flight.
                MockEvent::Sleep { ms: 80 },
                MockEvent::ToolUseStart {
                    id: "call-1".to_string(),
                    name: "memory_write".to_string(),
                },
                MockEvent::ToolUseEnd {
                    input: serde_json::json!({"name": "note", "description": "d"}),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::ToolUse,
                },
            ],
            replace_round("summary after a refusal", None),
        ]));
        let registry = crate::tools::ToolRegistry::new();
        registry
            .register(Arc::new(StubTool {
                name: "memory_write".to_string(),
                calls: Arc::clone(&calls),
            }))
            .expect("register stub");
        let (mut agent, store) = agent_with_registry_and_checkpoint(provider, registry).await;
        agent.cells.permission = SharedPermission::new(
            crate::permission::Permission::Read,
            crate::permission::EnabledPermissions::ALL,
        );
        let permission = agent.cells.permission.clone();
        let mut messages = conversation();

        let (outcome, ()) = tokio::join!(
            compact(
                &agent,
                &store,
                &mut messages,
                CompactRequest::new(CompactOrigin::Manual),
            ),
            async {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                permission
                    .try_set(crate::permission::Permission::None)
                    .expect("none is enabled");
            }
        );

        assert_eq!(outcome.source, CompactSource::Checkpoint);
        assert!(
            calls.lock().expect("calls").is_empty(),
            "a call admitted at `read` must not run once the level is `none`"
        );
    }

    /// The checkpoint is the longest thing compaction does, and at `ask` it can block on a
    /// human. A bare token with no signal source would make Ctrl+C a no-op, which
    /// `run_turn_interruptible` documents as the bug to avoid.
    #[tokio::test]
    async fn an_interrupt_ends_the_checkpoint_and_falls_back() {
        let provider = Arc::new(MockProvider::from_rounds(vec![text_round("fallback")]));
        let (agent, store) = agent_with_checkpoint(provider, true).await;
        let mut messages = conversation();
        agent.cells().session_id.set(
            store
                .create_session(None, "test-profile".to_string())
                .await
                .expect("create session"),
        );

        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let outcome = agent
            .compact_session(
                &mut messages,
                CompactRequest::new(CompactOrigin::Manual),
                cancellation,
            )
            .await
            .expect("compaction still completes");

        // Interrupting the checkpoint must not fail the compaction: the user asked for the
        // checkpoint to stop, not for the window to stay full.
        assert_eq!(outcome.source, CompactSource::Summarizer);
    }

    /// An automatic compaction that finds its token fired ends before the summarizer is paid
    /// for. The checkpoint answers a fired token with `Ok(None)`, which must not hand the
    /// conversation to the summarizer only for the check after it to throw the request away.
    #[tokio::test]
    async fn an_interrupt_before_the_summarizer_costs_no_request() {
        let provider = Arc::new(MockProvider::from_rounds(vec![text_round("unwanted")]));
        let (agent, store) =
            agent_with_checkpoint(Arc::clone(&provider) as Arc<dyn Provider>, false).await;
        let mut messages = conversation();
        agent.cells().session_id.set(
            store
                .create_session(None, "test-profile".to_string())
                .await
                .expect("create session"),
        );

        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let error = agent
            .compact_session(
                &mut messages,
                CompactRequest::new(CompactOrigin::Reactive),
                cancellation,
            )
            .await
            .expect_err("a fired token ends an automatic compaction");

        assert!(matches!(error, MekaError::Interrupted), "{error:?}");
        assert!(
            provider.completions().is_empty(),
            "the summarizer must not have been called"
        );
    }

    /// A redaction the summarizer's request reports is counted on the session like one during
    /// a turn. Both compaction doors emitted the notice and skipped the count, so `/status`
    /// under-reported on exactly the largest requests meka sends.
    #[tokio::test]
    async fn a_redaction_during_compaction_is_counted() {
        let provider = Arc::new(MockProvider::from_rounds(vec![vec![
            MockEvent::Redaction {
                images: 3,
                bytes: 6_000_000,
                positions: Vec::new(),
            },
            MockEvent::Text {
                text: "summary".to_string(),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::EndTurn,
            },
        ]]));
        let (agent, store) = agent_with_checkpoint(provider, false).await;
        let mut messages = conversation();

        compact(
            &agent,
            &store,
            &mut messages,
            CompactRequest::new(CompactOrigin::Reactive),
        )
        .await;

        let snapshot = agent.session_stats_snapshot();
        assert_eq!(snapshot.redactions, 1);
        assert_eq!(snapshot.redacted_images, 3);
        assert_eq!(snapshot.redacted_bytes, 6_000_000);
    }

    /// A model that never submits must not compact forever. The cap ends the loop, and the run
    /// still yields a summary via the text tier rather than failing.
    #[tokio::test]
    async fn the_iteration_cap_ends_a_checkpoint_that_never_submits() {
        let mut rounds = Vec::new();
        for index in 0..CHECKPOINT_MAX_ITERATIONS {
            rounds.push(vec![
                MockEvent::Text {
                    text: format!("thinking {index}"),
                },
                MockEvent::ToolUseStart {
                    id: format!("call-{index}"),
                    name: "memory_write".to_string(),
                },
                MockEvent::ToolUseEnd {
                    input: serde_json::json!({"name": "note"}),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::ToolUse,
                },
            ]);
        }
        let provider = Arc::new(MockProvider::from_rounds(rounds));
        let (agent, store) = agent_with_checkpoint(provider, true).await;
        let mut messages = conversation();

        let outcome = compact(
            &agent,
            &store,
            &mut messages,
            CompactRequest::new(CompactOrigin::Reactive),
        )
        .await;

        assert_eq!(outcome.source, CompactSource::CheckpointText);
        // The registry here is empty, so `memory_write` was refused as unavailable and nothing
        // may be claimed as written.
        assert!(outcome.memories_written.is_empty());
    }

    /// Compaction rewrites the head of the conversation; every subsequent one summarizes the
    /// previous summary. The count is what tells the model how far from the original it is.
    #[tokio::test]
    async fn each_compaction_advances_the_generation() {
        let provider = Arc::new(MockProvider::from_rounds(vec![
            text_round("first"),
            text_round("second"),
        ]));
        let (agent, store) = agent_with_checkpoint(provider, false).await;
        agent.cells().session_id.set(
            store
                .create_session(None, "test-profile".to_string())
                .await
                .expect("create session"),
        );
        let mut messages = conversation();

        assert_eq!(
            agent
                .compaction_generation(agent.session_id().expect("session"))
                .await,
            0
        );
        for expected in 1..=2 {
            agent
                .compact_session(
                    &mut messages,
                    CompactRequest::new(CompactOrigin::Manual),
                    CancellationToken::new(),
                )
                .await
                .expect("compaction");
            assert_eq!(
                agent
                    .compaction_generation(agent.session_id().expect("session"))
                    .await,
                expected
            );
        }

        // The database is the authority the in-memory counter is seeded from, and
        // `prune_compacted_events` has already dropped the earlier boundary from the log.
        assert_eq!(
            store
                .count_compactions(agent.session_id().expect("session"))
                .await
                .expect("count"),
            2
        );
    }
}
