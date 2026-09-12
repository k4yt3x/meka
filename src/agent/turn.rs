//! One turn of the loop: admission, the streaming rounds against the provider, and what the turn
//! settles to. Everything here runs under a held session lock.

use super::*;
use crate::{
    conversation::PromptRetention,
    session::{AUTO_COMPACT_THRESHOLD_PERCENT, CompactOrigin, CompactRequest},
};

/// Why an [`Agent::run_turn`] invocation finished cleanly. Callers that drive a user-facing
/// protocol (e.g. the ACP `session/prompt` response) use this to map to a protocol-level stop
/// reason; REPL and one-shot callers discard it. `Interrupted` is not represented here. It
/// surfaces as `Err(MekaError::Interrupted)` so the success-path return type stays straightforward.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TurnOutcome {
    /// The model returned a natural end-of-turn (or an unrecognized stop reason, treated as
    /// end-of-turn since we have nothing better to surface).
    EndTurn,
    /// The provider stopped because the model hit its maximum output tokens. The assistant message
    /// may be truncated; clients can reflect this in their UI.
    MaxTokens,
    /// The model refused to comply with the request (Claude `stop_reason: "refusal"`, OpenAI
    /// equivalent). The string carries the model's refusal text when available so clients can
    /// render it instead of a generic "request failed."
    Refusal(String),
}
/// What a turn is asked. Either a prompt somebody typed (a user, a scheduled job, a spawning
/// agent) or a batch of finished background work that earned a turn of its own, with the images
/// attached, whether the prompt is withdrawn if the turn fails, and the outcomes riding ahead of
/// it. Built by every host and every out-of-band driver alike, so the join into one user message
/// and the retention rule live here and nowhere else.
pub(crate) struct TurnInput {
    prompt: Prompt,
    images: Vec<ImageSource>,
    retention: PromptRetention,
    /// Finished background work folded in ahead of the prompt, already claimed.
    riding: Vec<crate::store::background::BackgroundTask>,
}

enum Prompt {
    Typed(String),
    Outcomes(Vec<crate::store::background::BackgroundTask>),
}

impl TurnInput {
    /// A typed prompt. Blank text with no image is refused with [`MekaError::EmptyPrompt`]: it
    /// costs a provider round-trip to produce nothing, and the model cannot tell it from a prompt
    /// whose content went missing. This is the one check; no host repeats it ahead of the call.
    pub(crate) fn from_parts(
        prompt: String,
        images: Vec<ImageSource>,
    ) -> crate::error::Result<Self> {
        if prompt.trim().is_empty() && images.is_empty() {
            return Err(MekaError::EmptyPrompt);
        }
        Ok(Self {
            prompt: Prompt::Typed(prompt),
            images,
            retention: PromptRetention::Keep,
            riding: Vec::new(),
        })
    }

    /// The typed words after a host expanded a slash command in them. The prompt was admitted on
    /// what the user typed, ahead of the session being taken, and an expansion only adds to it;
    /// nothing to replace on a batch of outcomes, which no host expands.
    pub(crate) fn with_words(mut self, words: String) -> Self {
        if let Prompt::Typed(text) = &mut self.prompt {
            *text = words;
        }
        self
    }

    /// A batch of outcomes that earned a turn of its own. Always kept: the rows are stamped
    /// delivered before the turn starts and are never handed out again.
    pub(crate) fn outcomes(outcomes: Vec<crate::store::background::BackgroundTask>) -> Self {
        Self {
            prompt: Prompt::Outcomes(outcomes),
            images: Vec::new(),
            retention: PromptRetention::Keep,
            riding: Vec::new(),
        }
    }

    /// What becomes of the prompt if the turn ends unanswered: the scheduler's answer per job, the
    /// HTTP client's per turn; see [`PromptRetention`].
    pub(crate) fn retaining(mut self, retention: PromptRetention) -> Self {
        self.retention = retention;
        self
    }

    /// Fold claimed outcomes in ahead of the prompt. A prompt carrying outcomes is kept whatever
    /// its own retention says, because the rows exist nowhere else once stamped.
    pub(crate) fn riding(
        mut self,
        outcomes: Vec<crate::store::background::BackgroundTask>,
    ) -> Self {
        self.retention = crate::background::retention_carrying(&outcomes, self.retention);
        self.riding = outcomes;
        self
    }

    /// The words as typed, or nothing for a turn that only delivers background outcomes.
    fn words(&self) -> String {
        match &self.prompt {
            Prompt::Typed(text) => text.clone(),
            Prompt::Outcomes(_) => String::new(),
        }
    }

    /// The finished background work this turn delivers, rendered for the turn's context block, or
    /// `None`. It rides the one user message the turn appends rather than being sent as a message
    /// of its own, because `withdraw_unanswered_prompt` and `ends_on_a_turn_opening` assume a turn
    /// opens with exactly one.
    fn delivered_outcomes(&self) -> Option<String> {
        match &self.prompt {
            Prompt::Outcomes(outcomes) => Some(crate::background::render_outcomes(outcomes)),
            Prompt::Typed(_) if self.riding.is_empty() => None,
            Prompt::Typed(_) => Some(crate::background::render_outcomes_riding(&self.riding)),
        }
    }
}

/// How many times a single turn may emergency-compact-and-retry after the provider reports a
/// context-window overflow before giving up. One pass shrinks the request dramatically; if it still
/// overflows, looping won't help.
pub(super) const MAX_OVERFLOW_RETRIES: u32 = 1;
/// Whether a world-state render made when the conversation held `rendered_at` messages is still
/// inside the window [`truncate_messages_for_context`] will send.
///
/// The render lives in exactly one user message, at index `rendered_at`. The window keeps the last
/// `context_messages` entries, so that message survives while `current_len - rendered_at` stays
/// within the limit. Once it falls out, the model can no longer see the tool catalog, the skill
/// list, or any MCP server's instructions, and the picture has to be restated in full.
///
/// Deliberately one turn conservative (`<` rather than `<=`), for two reasons: `current_len` is
/// read before this turn's own message is appended, and `truncate_messages_for_context` walks
/// backward from the cut to land on a user-message boundary, which can only keep *more*. Restating
/// a turn early costs tokens once per window; restating a turn late means a request with no
/// catalog in it at all.
pub(super) fn world_state_still_visible(
    rendered_at: usize,
    current_len: usize,
    context_messages: Option<usize>,
) -> bool {
    context_messages.is_none_or(|limit| current_len.saturating_sub(rendered_at) < limit)
}
/// Assemble the message list for one provider call inside a turn: the turn's stable base plus
/// whatever the tool loop has appended since, re-truncated as a whole.
///
/// A named function rather than four lines inline, so the tests that protect this windowing drive
/// the real path rather than a copy that could omit the truncation.
pub(super) fn assemble_api_messages(
    messages: &[Message],
    base_messages: &[Message],
    turn_start_len: usize,
    context_messages: Option<usize>,
) -> Vec<Message> {
    if messages.len() > turn_start_len {
        let mut combined = base_messages.to_vec();
        combined.extend_from_slice(&messages[turn_start_len..]);
        truncate_messages_for_context(&combined, context_messages)
    } else {
        base_messages.to_vec()
    }
}
pub(super) fn truncate_messages_for_context(
    messages: &[Message],
    context_messages: Option<usize>,
) -> Vec<Message> {
    let Some(limit) = context_messages else {
        return messages.to_vec();
    };

    if messages.len() <= limit {
        return messages.to_vec();
    }

    // Clamped to a valid index before the walk below reads `messages[start_index]`. `limit == 0` is
    // rejected at config load, but this function is also called with the value threaded through
    // `AgentOptions`, and an out-of-bounds index here is a panic that takes the process (or, under
    // `serve`, the turn task) down. Costing one message is the right trade against that.
    let mut start_index = messages
        .len()
        .saturating_sub(limit)
        .min(messages.len().saturating_sub(1));

    // A safe cut point is a user message that is NOT a tool_results message: it neither splits an
    // assistant(ToolUse) → user(ToolResult) chain nor leaves the window starting on a role the
    // Claude API rejects.
    let is_safe_cut = |index: usize| {
        messages.get(index).is_some_and(|message| {
            message.role == Role::User && !has_tool_results(&message.content)
        })
    };

    // Search forward first, which drops the leading tool chain whole rather than reaching back
    // over it: reaching back alone makes the cap advisory, since one long tool loop with no plain
    // user message inside it drags `start_index` to 0. Cutting forward can keep fewer messages than
    // asked for, which is what a maximum means.
    if let Some(index) = (start_index..messages.len()).find(|&index| is_safe_cut(index)) {
        return messages[index..].to_vec();
    }

    // Nothing ahead is safe (the tail is one unbroken tool chain), so reach back for the last cut
    // point that is. Exceeding the cap beats sending a conversation the provider will reject.
    while start_index > 0 && !is_safe_cut(start_index) {
        start_index -= 1;
    }

    messages[start_index..].to_vec()
}
pub(super) fn has_tool_results(content: &[ContentBlock]) -> bool {
    content
        .iter()
        .any(|block| matches!(block, ContentBlock::ToolResult { .. }))
}
/// Split the unavailable MCP servers into the ones that stop the turn and the ones that don't.
///
/// Only `required` servers gate. Whether a missing server should halt work is a property of that
/// server, not of the installation (the same config runs on a workstation that has the binary and
/// in a container that does not), so a single installation-wide switch could only ever be right
/// for one of them. `[mcp].strict` survives as the default each server inherits.
///
/// A free function rather than a method because it reads nothing from the agent, which also makes
/// the gating decision directly testable.
pub(super) fn gate_on_required_servers(not_ready: Vec<crate::mcp::NotConnected>) -> Result<()> {
    let (required, optional): (Vec<_>, Vec<_>) =
        not_ready.into_iter().partition(|server| server.required);

    if !optional.is_empty() {
        let names: Vec<&str> = optional.iter().map(|s| s.name.as_str()).collect();
        // `debug!`, not `warn!`: this runs on *every* turn, and a server that is down stays down,
        // so at warn level a single unreachable server would print a line before every reply for
        // the life of the session. The connector already reports the failure once, and `/mcp list`
        // in the REPL shows live state on demand; repeating it per turn is noise.
        let count = names.len();
        tracing::debug!("mcp: proceeding without {count} optional server(s): {names:?}");
    }

    if required.is_empty() {
        return Ok(());
    }
    Err(MekaError::McpTurnGated {
        servers: required
            .iter()
            .map(|server| {
                // Carry the cause, not just the label. This message is the only thing the user gets
                // when a required server blocks every turn, and the connector's warn fires once at
                // startup and then stays quiet, so "ida (failed)" on its own leaves them with
                // nothing to act on. The cause replaces the label rather than joining it: every
                // one of them already describes a failure ("failed to spawn process: …"), so
                // prefixing would render as "failed: failed to …".
                let detail = match &server.state {
                    crate::mcp::ServerState::Failed { error, .. } => error.clone(),
                    other => other.label().to_string(),
                };
                (server.name.clone(), detail)
            })
            .collect(),
    })
}
/// Whether an assistant turn produced any user-visible text: a `Text` block with non-whitespace
/// content. `Thinking`, `ToolUse`, `ToolResult`, and `Image` blocks are not user-visible prose, so
/// a thinking-only turn returns `false` here.
pub(super) fn has_visible_text(content: &[ContentBlock]) -> bool {
    content
        .iter()
        .any(|block| matches!(block, ContentBlock::Text { text } if !text.trim().is_empty()))
}
/// Human-readable stand-in for a terminal turn that produced no content (e.g. a hard refusal, or an
/// empty `max_tokens` / `end_turn`). Used both as the persisted assistant text (so the message
/// isn't empty) and as the line surfaced to the user.
pub(super) fn empty_turn_notice(stop_reason: &StopReason) -> String {
    match stop_reason {
        StopReason::Refusal(text) if !text.is_empty() => text.clone(),
        StopReason::Refusal(_) => "[The model declined to respond to this request.]".to_string(),
        StopReason::MaxTokens => {
            "[The model reached its output limit before producing a response.]".to_string()
        }
        // Surface the raw reason so an unrecognized stop reason is visible instead of being
        // swallowed as a blank turn.
        StopReason::Unknown(reason) => {
            format!("[The model returned an empty response (stop reason: {reason}).]")
        }
        _ => "[The model returned an empty response.]".to_string(),
    }
}
/// Meta message injected to coax a user-visible response out of a turn that produced only thinking
/// (or nothing).
pub(super) const THINKING_ONLY_NUDGE: &str = "[Your previous response contained no visible output. Please \
                                   continue and produce a user-visible response.]";
/// Whether to nudge the model for a user-visible response after a turn that made no tool call and
/// produced no visible text (a thinking-only turn): at most once per turn, and only for a terminal
/// stop reason without its own handling, since `MaxTokens` and `Refusal` carry their own outcomes
/// and a no-text turn under those reasons falls through to [`empty_turn_notice`].
pub(super) fn should_nudge_thinking_only(
    has_tool_calls: bool,
    has_visible_text: bool,
    stop_reason: &StopReason,
    already_nudged: bool,
) -> bool {
    !has_tool_calls
        && !has_visible_text
        && !already_nudged
        && matches!(stop_reason, StopReason::EndTurn | StopReason::Unknown(_))
}
/// Replace every image with a `[image]` placeholder, leaving all text intact.
///
/// The checkpoint turn's preprocessing, and deliberately only half of what
/// [`strip_images_and_truncate`] does. Images are the expensive part of re-sending a conversation
/// and almost never what a summary needs to carry; text is exactly what the agent has to read to
/// judge what matters, so truncating it would hand the agent the same degraded view that made the
/// standalone summarizer worth replacing.
pub(super) fn strip_images(content: &mut [ContentBlock]) {
    use crate::conversation::ToolResultContent;

    for block in content.iter_mut() {
        match block {
            ContentBlock::ToolResult {
                content: tool_content,
                ..
            } => {
                for item in tool_content.iter_mut() {
                    if matches!(item, ToolResultContent::Image { .. }) {
                        *item = ToolResultContent::Text {
                            text: "[image]".to_string(),
                        };
                    }
                }
            }
            ContentBlock::Image { .. } => {
                *block = ContentBlock::Text {
                    text: "[image]".to_string(),
                };
            }
            _ => {}
        }
    }
}
/// Preprocess message content blocks for the compaction summarizer: replace images with "[image]"
/// markers and truncate large text blocks.
pub(super) fn strip_images_and_truncate(content: &mut [ContentBlock]) {
    use crate::conversation::ToolResultContent;

    const MAX_TEXT_CHARS: usize = 2000;
    const HEAD_CHARS: usize = 1000;
    const TAIL_CHARS: usize = 500;

    for block in content.iter_mut() {
        match block {
            ContentBlock::ToolResult {
                content: tool_content,
                ..
            } => {
                for item in tool_content.iter_mut() {
                    match item {
                        ToolResultContent::Image { .. } => {
                            *item = ToolResultContent::Text {
                                text: "[image]".to_string(),
                            };
                        }
                        ToolResultContent::Text { text } => {
                            if text.len() > MAX_TEXT_CHARS {
                                let head_end = text.floor_char_boundary(HEAD_CHARS);
                                let tail_start =
                                    text.floor_char_boundary(text.len().saturating_sub(TAIL_CHARS));
                                *text = format!(
                                    "{}\n... (truncated for compaction) ...\n{}",
                                    &text[..head_end],
                                    &text[tail_start..],
                                );
                            }
                        }
                    }
                }
            }
            // The context block too: it is the larger of the two on most turns, and the summarizer
            // needs no more of it than it needs of a long prompt.
            ContentBlock::Text { text } | ContentBlock::TurnContext { text }
                if text.len() > MAX_TEXT_CHARS =>
            {
                let head_end = text.floor_char_boundary(HEAD_CHARS);
                let tail_start = text.floor_char_boundary(text.len().saturating_sub(TAIL_CHARS));
                *text = format!(
                    "{}\n... (truncated for compaction) ...\n{}",
                    &text[..head_end],
                    &text[tail_start..],
                );
            }
            _ => {}
        }
    }
}

/// What a streaming attempt reported about itself while it ran, read back by the retry policy and
/// by the turn when the attempt fails.
#[derive(Debug, Default)]
pub(super) struct StreamProgress {
    /// Set the instant anything user-visible is forwarded to the frontend, so a failure after it
    /// is never retried: the user has seen output the retry would re-send.
    pub(crate) content_started: bool,
    /// What had streamed when the attempt failed, if any of it was text. The user watched it
    /// arrive, so it is theirs to keep whatever the stream did afterwards.
    pub(crate) partial: Option<Message>,
}

impl Agent {
    /// Whether a turn could start right now, asked before anything irreversible is done for it.
    ///
    /// [`Self::run_turn`] gates on this itself, deliberately before it touches the conversation, so
    /// a refused turn leaves no trace. That ordering is what makes it unsafe to *claim* a
    /// background outcome first: the claim stamps a row that is never handed out again, and a turn
    /// refused here would carry it nowhere. `background::claim_undelivered_outcomes` asks this
    /// before stamping anything, which is why the answer is public.
    pub(crate) async fn ensure_ready_for_turn(&self) -> Result<()> {
        self.await_mcp_ready().await
    }

    /// Per-turn MCP readiness gate. Applies to every turn (not just the first) so mid-session
    /// reconnects also gate cleanly. Awaits `grace` for Pending servers to finish connecting, then
    /// hands whatever is still not `Connected` to [`gate_on_required_servers`], which rejects the
    /// turn only if one of them is `required`.
    ///
    /// No-op when no MCP manager is attached (e.g. sub-agents).
    pub(super) async fn await_mcp_ready(&self) -> Result<()> {
        let Some(manager) = self.mcp_manager() else {
            return Ok(());
        };
        if manager.all_ready() {
            let not_ready = manager.enabled_not_connected().await;
            if not_ready.is_empty() {
                return Ok(());
            }
            return self.handle_mcp_not_ready(not_ready);
        }

        // Best-effort grace wait. We re-check readiness below regardless of whether `await_settled`
        // returned in time. The timeout result is intentionally discarded.
        if tokio::time::timeout(self.options.mcp_grace, manager.await_settled())
            .await
            .is_err()
        {
            tracing::debug!("MCP servers were still connecting when the grace period ended");
        }

        let not_ready = manager.enabled_not_connected().await;
        if not_ready.is_empty() {
            return Ok(());
        }
        self.handle_mcp_not_ready(not_ready)
    }

    pub(super) fn handle_mcp_not_ready(
        &self,
        not_ready: Vec<crate::mcp::NotConnected>,
    ) -> Result<()> {
        gate_on_required_servers(not_ready)
    }

    /// One turn on behalf of whoever asked for it; what becomes of its prompt if it fails is the
    /// input's retention.
    pub(crate) async fn run_turn(
        &self,
        messages: &mut Conversation,
        input: TurnInput,
        cancellation: CancellationToken,
    ) -> Result<TurnOutcome> {
        self.run_attributed_turn(messages, input, cancellation)
            .await
    }

    /// Who this turn's requests are for. A worker answers its parent's prompt, so it carries the
    /// id its parent's turn handed the spawning call rather than minting one; the root agent
    /// mints one per turn. The previous-request slot is the agent's own, so a conversation names
    /// its own last response and never a sibling's.
    fn turn_attribution(&self) -> crate::provider::Attribution {
        crate::provider::Attribution {
            subagent: self.role.is_worker(),
            prompt_id: Some(self.role.inherited_prompt_id().unwrap_or_else(Uuid::new_v4)),
            previous_request: Some(Arc::clone(&self.previous_request)),
            previous_message: Some(Arc::clone(&self.previous_message)),
        }
    }

    /// Park the lock on a session this agent has just created where the host can reach it.
    ///
    /// Claiming the lock in the REPL's post-turn block instead would leave no lock file for the
    /// whole of a first turn, so a second `meka -c --oneshot` in that window would write into the
    /// same conversation and leave a log with non-alternating roles that the provider refuses.
    ///
    /// `None` means the claim could not be made at all, which
    /// [`crate::store::Store::create_session_locked`] has already warned about. The turn runs
    /// regardless: the only way to get here is a filesystem problem with the lock directory, and
    /// refusing to run over that would break installations that work today.
    fn hold_the_lock_on_a_created_session(&self, lock: Option<crate::fs::FileLock>) {
        *crate::sync::lock(&self.cells.session_lock) = lock;
    }

    pub(super) async fn run_attributed_turn(
        &self,
        messages: &mut Conversation,
        input: TurnInput,
        cancellation: CancellationToken,
    ) -> Result<TurnOutcome> {
        let retention = input.retention;
        let words = input.words();
        let outcomes = input.delivered_outcomes();
        let TurnInput { images, .. } = input;
        let attribution = self.turn_attribution();
        // Gate on MCP readiness BEFORE touching session state / message history so a rejected turn
        // leaves no trace in the conversation.
        self.await_mcp_ready().await?;

        let existing = self.cells.session_id.get();
        let session_id = if let Some(session_id) = existing {
            session_id
        } else {
            // Created and locked in one step, the lock taken first. See
            // [`crate::store::Store::create_session_locked`] for why the order is the
            // whole of it.
            let (created, lock) = self
                .store
                .create_session_locked(
                    Some(self.cells.cwd.get()),
                    // The level this session starts at. The row is the one answer every process
                    // reads for a scheduled gate, and a row with no level runs nothing; the REPL
                    // keeps it current through `ReplEvent::PermissionChanged`, ACP through
                    // `session/set_mode`. The approvals switch travels beside it for a resume.
                    self.cells.permission.get().to_string(),
                    self.cells.permission.approvals(),
                    None,
                    None,
                    self.profile(),
                )
                .await?;
            let id = created.id;
            self.hold_the_lock_on_a_created_session(lock.ok());
            // The cell is the one holder: every tool and the host read the id from here.
            self.cells.session_id.set(id);
            self.cells
                .frontend
                .emit(FrontendEvent::SessionStarted { id })
                .await;
            id
        };

        self.cells.frontend.emit(FrontendEvent::TurnStarted).await;

        // Auto-compact if the last turn's context occupancy exceeded the threshold fraction of the
        // context window. This check runs between turns, before the loop opens, which is why it
        // needs no re-anchoring of its own. Compaction itself is not confined here: the emergency
        // retry and the agent's own `context_compact` both run inside the loop and re-anchor
        // through `TurnRecovery::after_conversation_rewrite`.
        if let Some(threshold) = self.auto_compact_threshold() {
            let last_tokens = self
                .cells
                .context_tokens
                .load(std::sync::atomic::Ordering::Relaxed);
            if last_tokens > threshold && messages.len() > 1 {
                let window = self.context_window();
                tracing::info!(
                    "auto-compacting: {last_tokens} tokens in context exceeds \
                     {AUTO_COMPACT_THRESHOLD_PERCENT}% of the {window} window"
                );
                if let Err(error) = self
                    .compact_session(
                        messages,
                        CompactRequest::new(CompactOrigin::Reactive)
                            .attributed_to(attribution.prompt_id),
                        cancellation.clone(),
                    )
                    .await
                {
                    tracing::warn!("auto-compact failed: {error}");
                }
            }
        }

        let permission = self.cells.permission.get();
        let approvals = self.cells.permission.approvals();

        let catalog = self.tool_registry.tool_catalog();
        let skills = self.skills.current().await;
        // A store that cannot be read degrades rather than failing the turn: this runs on every
        // prompt, and a transient `SQLITE_BUSY` should not cost the turn itself.
        //
        // `memories_readable` is what stops that degradation becoming a lie: an empty `Vec` here is
        // indistinguishable from an empty store, so the world-state diff would read it as every
        // memory having been deleted and announce them all as written again on the next successful
        // read. Skipped outright when no tool can open the index, exactly as the schedule and
        // background reads are: `index()` materializes every row, and `WorldSnapshot::new` would
        // then decline to render any of it. "Readable" for a store nobody asked about is `true`:
        // nothing failed, so there is nothing for the diff to carry forward.
        let (memories, memories_readable) = match prompt::memory_index_is_live(&catalog) {
            false => (Vec::new(), true),
            true => match self.memories.index().await {
                Ok(memories) => (memories, true),
                Err(error) => {
                    tracing::warn!("failed to read the memory index: {error}");
                    (Vec::new(), false)
                }
            },
        };
        let mcp_instructions = self
            .mcp_manager()
            .map(|manager| manager.server_instructions())
            .unwrap_or_default();

        // Tools, skills, and MCP instructions move mid-session, so they ride in the user message
        // and only what changed since the model was last told is rendered. `world_state_rollback`
        // is the snapshot before this turn claimed to have announced the change: a turn that fails
        // early pops its user message, which is the only place the announcement lives, so the claim
        // is withdrawn with it. Skipped for sub-agents, whose `system_prompt_override` already
        // lists a tool set fixed at spawn. Read fresh each turn and rendered outside the
        // world-state diff: running tasks are live state, like the todo list, not a record of what
        // the model has been told. Skipped entirely when the `task_*` tools are unregistered, which
        // is the default.
        let background_tasks =
            match prompt::background_index_is_live(&catalog).then_some(session_id) {
                Some(id) => self
                    .store
                    .background_store()
                    .list_running_background_tasks(id)
                    .await
                    .unwrap_or_else(|error| {
                        tracing::warn!("failed to load background tasks for context: {error}");
                        Vec::new()
                    }),
                None => Vec::new(),
            };

        let (world_state, world_state_rollback) = if self.options.system_prompt_override.is_some() {
            (String::new(), None)
        } else {
            // Read fresh rather than cached: a job can be added or canceled by `meka schedule`, by
            // another attached client, or by the scheduler retiring a fired one-shot, none of which
            // pass through this agent. Skipped outright when the tool that opens the index is not
            // registered: without this an installation with `[schedule] enabled = false` pays a
            // database round trip on every single turn for a section that will be discarded.
            let scheduled = match prompt::schedule_index_is_live(&catalog).then_some(session_id) {
                Some(id) => self
                    .store
                    .schedule_store()
                    .list_scheduled_jobs(id)
                    .await
                    .unwrap_or_else(|error| {
                        tracing::warn!("failed to load scheduled jobs for context: {error}");
                        Vec::new()
                    }),
                None => Vec::new(),
            };
            let mut current = prompt::WorldSnapshot::new(
                &catalog,
                &skills,
                &memories,
                &mcp_instructions,
                &scheduled,
            )
            // The live level, not the one recorded on the row: this answers "can it fire *now*",
            // which is the same question `prepare` asks a moment later on the scheduler's thread.
            .with_gate_authority(
                self.store.scheduler_memory(),
                &scheduled,
                self.cells.permission.get(),
                self.options.gate_tools.as_deref(),
            );
            let mut last = self.last_rendered_world.write().await;
            // An unreadable store carries the previous snapshot's memories forward, so the diff
            // compares that half against itself and says nothing about it; advancing to an empty
            // list would announce the whole store as deleted. Nothing to carry (the first turn of a
            // session) leaves the list empty, which renders no `[Memory]` section at all.
            if !memories_readable && let Some((previous, _)) = last.as_ref() {
                current.carry_memories_from(previous);
            }
            // Treat a render that has scrolled out of the API window as never having happened. The
            // window keeps the last `context_messages` entries, so a render at index `i` is gone
            // once the conversation grows past `i + limit`. Rendering in full then puts a fresh
            // copy at the new tail, good for another window's worth of turns.
            let still_visible = last.as_ref().filter(|(_, rendered_at)| {
                world_state_still_visible(
                    *rendered_at,
                    messages.len(),
                    self.options.context_messages,
                )
            });
            let rendered = prompt::render_world_state(&current, still_visible.map(|(s, _)| s));
            // This turn's user message is about to be appended, so that is where the render lands.
            let previous = last.replace((current, messages.len()));
            drop(last);
            (rendered, previous)
        };

        // Taken before the block is built, since the block is where it lands. A freshly spawned
        // sub-agent never sees it (its conversation is empty, so `from_events` never set it), but a
        // followed-up one does, and that is deliberate: it really is running against a fresh
        // registry, a fresh read tracker and an empty todo list. See
        // `crate::tools::subagent::AgentFollowupTool`.
        let resumed = messages.take_resumed_notice();

        let context_block = {
            let todos = self.cells.todo_list.get();
            let cwd = self.cells.cwd.get();
            let roots = self.cells.roots.get();
            prompt::build_turn_context(prompt::TurnContext {
                permission,
                approvals,
                todos: &todos,
                cwd: &cwd,
                roots: &roots,
                world_state: &world_state,
                budget: Some(self.context_budget(session_id).await),
                background: &background_tasks,
                outcomes: outcomes.as_deref(),
                resumed,
            })
        };
        // Build the user message once (context block, the words, any input images) and reuse it
        // for both the in-memory append and every persist path below, so attached images survive
        // resume.
        let user_message = Message::user_turn(context_block, words, images);
        // Captured around the append rather than in the `TurnRecovery` literal below, which is
        // built after a proactive compaction may have moved the conversation under both. See their
        // field documentation for what each one is measured against. The compaction, if it runs,
        // then replaces this with `SUSPECT_FLOOR_AFTER_REWRITE`.
        let mut suspect_floor = messages.len();
        messages.append(user_message.clone());
        let prompt_only_events = messages.events_len();
        // Persist the user message eagerly, before the first provider call, or a crash during the
        // provider round trip would lose it from disk. On a transient database failure the lazy
        // save path below retries; `user_eagerly_saved` suppresses double-writes on the happy path.
        let user_event = crate::conversation::Event::Append(user_message.clone());
        let mut user_eagerly_saved = match self.store.save_event(session_id, &user_event).await {
            Ok(()) => true,
            Err(error) => {
                tracing::warn!(
                    "failed to persist the user message; retrying on the first provider \
                     response: {error}"
                );
                false
            }
        };
        let system_prompt: Arc<str> = match &self.options.system_prompt_override {
            Some(prompt) => Arc::from(prompt.as_str()),
            None => Arc::from(prompt::build_system_prompt(
                self.options.sandboxed_shell,
                self.options.user_instructions.as_deref(),
            )),
        };

        // Proactive pre-send compaction. The reactive check at the top of the turn reads the
        // *previous* round's reported usage, so a turn whose own input jumps over the window (a
        // huge paste, a large tool result carried in) would be sent uncompacted and hard-fail.
        // Project this request locally (conversation + system prompt) and compact before sending if
        // it would cross the threshold. `estimate_messages` under-reads (no tool schemas), so this
        // is a floor that complements, not replaces, the reactive check and the overflow recovery
        // below.
        if let Some(threshold) = self.auto_compact_threshold()
            && messages.len() > 1
        {
            let projected = crate::tokens::estimate_messages(messages.as_slice())
                .saturating_add(crate::tokens::estimate_text(&system_prompt));
            if projected > threshold {
                let window = self.context_window();
                tracing::info!(
                    "proactive compaction: projected {projected} input tokens exceeds \
                     {AUTO_COMPACT_THRESHOLD_PERCENT}% of the {window} window"
                );
                match self
                    .compact_session(
                        messages,
                        CompactRequest::new(CompactOrigin::Proactive)
                            .attributed_to(attribution.prompt_id),
                        cancellation.clone(),
                    )
                    .await
                {
                    // The floor captured above counts messages that no longer exist. Left as it
                    // was, it lands past the end of the collapsed conversation, clamps to the
                    // length, and leaves the degrade-and-retry nothing to look at for the whole
                    // turn. See `SUSPECT_FLOOR_AFTER_REWRITE`.
                    //
                    // The rewrite persisted the prompt too, in the kept tail or inside the
                    // boundary's summary, whatever became of the eager save. Saving it again on
                    // the 2xx would put a second copy after the boundary.
                    Ok(_) => {
                        suspect_floor = SUSPECT_FLOOR_AFTER_REWRITE;
                        user_eagerly_saved = true;
                    }
                    Err(error) => tracing::warn!("proactive compaction failed: {error}"),
                }
            }
        }

        let mut recovery = TurnRecovery {
            base_messages: Arc::from(truncate_messages_for_context(
                messages.as_slice(),
                self.options.context_messages,
            )),
            turn_start_len: messages.len(),
            suspect_floor,
            prompt_only_events,
            overflow_retries: 0,
            requested_compactions: 0,
            tiers_tried: 0,
            pending_repair: None,
            user_saved: user_eagerly_saved,
            thinking_only_nudged: false,
            outage_reprieve_used: false,
        };

        // Accumulate token usage across every provider call within this turn so the per-turn
        // display reflects the whole turn (including tool-execution loops), not just the final
        // round-trip.
        let mut turn_usage = crate::stats::TokenUsage::default();

        let result: Result<TurnOutcome> = 'turn: {
            loop {
                // Both exits undo an unvindicated repair first. A degrade that has been applied but
                // not yet retried is parked in memory with no `Event::Repair` on disk, and these
                // are the two ways out of the round that skip the error arm entirely: a repair
                // fires, `continue` returns here, and the turn ends before the provider ever judged
                // it. Leaving it applied would have the model reasoning from a conversation the
                // store has never heard of, while `GET /messages` still serves the original with
                // its revision unmoved. Under ACP `client_disconnected` becomes true precisely
                // while a turn is stalled on a failing provider, which is when a repair is most
                // likely to be in flight.
                if cancellation.is_cancelled() {
                    recovery.undo_rejected_repair(self, messages);
                    break 'turn Err(MekaError::Interrupted);
                }
                // Bail out if the frontend has noticed its client went away (e.g. ACP stdio
                // disconnect). No point burning more provider tokens for an audience that won't see
                // the output. REPL frontends report `false` here, so this is a no-op for them.
                if self.cells.frontend.client_disconnected() {
                    recovery.undo_rejected_repair(self, messages);
                    break 'turn Err(MekaError::Interrupted);
                }

                // Conversation length behind this request, stamped onto `last_accepted_len` when
                // the provider takes it.
                let sent_len = messages.len();

                // Re-truncate the assembled request, not just the turn's starting point: everything
                // the tool loop appends is spliced onto a `base_messages` capped once at turn
                // start, so `[session] context_messages` would otherwise stop applying at the
                // second provider call. This costs cache, since the cut walks forward to the first
                // safe boundary and the prefix sent to the provider moves within one turn, but an
                // unbounded request eventually hits the context limit the setting exists to avoid.
                //
                // Whatever a request that never reached this point reported is not about the view
                // this request is built from.
                crate::sync::lock(&self.pending_redactions).clear();
                let api_messages: Arc<[Message]> = if messages.len() > recovery.turn_start_len {
                    Arc::from(assemble_api_messages(
                        messages.as_slice(),
                        &recovery.base_messages,
                        recovery.turn_start_len,
                        self.options.context_messages,
                    ))
                } else {
                    // What `assemble_api_messages` returns with nothing appended, reusing the
                    // allocation instead of copying it. `base_messages` was truncated at turn
                    // start, so there is nothing left for the cap to do.
                    Arc::clone(&recovery.base_messages)
                };

                // Recompute the active tool set every iteration so a `load_tool` call earlier in
                // this turn becomes visible to the model on the very next request, without
                // mutating any registry state. Append-only growth keeps the tools array's cache
                // prefix stable.
                //
                // Read from events (not the materialized slice) so the deferred-tool snapshot
                // stored on `Event::CompactBoundary` survives across compaction; otherwise tools
                // the model loaded pre-compaction would silently drop out of the active set on the
                // next turn.
                let loaded = crate::tools::load_tool::extract_loaded_tool_names_from_events(
                    messages.events(),
                );
                let tools: Arc<[ToolDefinition]> =
                    Arc::from(self.tool_registry.definitions_active_with_loaded(&loaded));

                // The part of the window that is not conversation, for `context_check` to report.
                // Re-stamped per round because the active tool set grows as `load_tool` pulls in
                // deferred schemas. Written, never read by the agent: an estimate is fine for
                // informing the model's decision, while the agent's own thresholds run off the
                // provider's exact numbers.
                self.cells.context_overhead.store(
                    tools
                        .iter()
                        .map(|tool| {
                            crate::tokens::estimate_text(&tool.name)
                                .saturating_add(crate::tokens::estimate_text(&tool.description))
                                .saturating_add(crate::tokens::estimate_text(
                                    &tool.parameters.to_string(),
                                ))
                        })
                        .fold(
                            crate::tokens::estimate_text(&system_prompt),
                            |total, cost| total.saturating_add(cost),
                        ),
                    std::sync::atomic::Ordering::Relaxed,
                );

                // Streaming and blocking paths converge on `(Message, StopReason, TokenUsage)`. The
                // blocking provider call surfaces notices in its return tuple (no event channel);
                // we forward them to the frontend here so the user sees the same advisories the
                // streaming path emits inline via `StreamEvent::Notice`.
                let mut progress = StreamProgress::default();
                let call_result: Result<(Message, StopReason, crate::stats::TokenUsage)> = if self
                    .options
                    .streaming
                {
                    self.run_streaming(
                        Arc::clone(&system_prompt),
                        api_messages,
                        tools,
                        attribution.clone(),
                        cancellation.clone(),
                        &mut progress,
                    )
                    .await
                } else {
                    // Non-streaming is fully atomic (nothing is visible until this returns
                    // `Ok`), so `content_started` is always `false` here, so every retryable
                    // failure is retried up to the cap regardless of prior attempts.
                    let mut retries = 0u32;
                    let started = std::time::Instant::now();
                    loop {
                        match self
                            .provider()
                            .complete(
                                CompletionRequest::new(&system_prompt, &api_messages, &tools)
                                    .attributed(attribution.clone()),
                                cancellation.clone(),
                            )
                            .await
                        {
                            Ok(crate::provider::Completion {
                                message,
                                stop_reason,
                                usage,
                                notices,
                            }) => {
                                for notice in notices {
                                    self.forward_notice(notice).await;
                                }
                                break Ok((message, stop_reason, usage));
                            }
                            Err(error) => {
                                match should_retry_provider_error(
                                    &error,
                                    false,
                                    retries,
                                    started.elapsed(),
                                ) {
                                    Some(delay) => {
                                        retries += 1;
                                        let ceiling = crate::provider::retry::MAX_PROVIDER_RETRIES;
                                        tracing::warn!(
                                            "provider request failed transiently (attempt \
                                             {retries}/{ceiling}), retrying in {delay:?}: {error}"
                                        );
                                        tokio::select! {
                                            _ = tokio::time::sleep(delay) => {}
                                            _ = cancellation.cancelled() => break Err(MekaError::Interrupted),
                                        }
                                    }
                                    None => break Err(error),
                                }
                            }
                        }
                    }
                };

                let (mut assistant_message, stop_reason, usage) = match call_result {
                    Ok(value) => value,
                    Err(MekaError::ContextOverflow(message))
                        if self.auto_compact_threshold().is_some()
                            && messages.len() > 1
                            && recovery.overflow_retries < MAX_OVERFLOW_RETRIES =>
                    {
                        if let Err(error) = recovery
                            .recover_from_context_overflow(self, messages, &cancellation, message)
                            .await
                        {
                            break 'turn Err(error);
                        }
                        continue;
                    }
                    Err(error) => {
                        // Unconditionally, before deciding anything else. A repair still applied
                        // here is one the provider has just refused a second time, so it was not
                        // the fix whatever happens next: another tier measures itself against the
                        // conversation as it really is, and a turn that gives up leaves memory and
                        // store agreeing.
                        recovery.undo_rejected_repair(self, messages);
                        if !refusal_may_blame_content(&error, progress.content_started) {
                            // The same treatment an interrupt gives a half-streamed answer: the
                            // text the user watched arrive is kept, without the tool calls that
                            // never ran.
                            //
                            // The prompt goes first, as it does on the interrupt arm below: this
                            // is the one exit ahead of the 2xx that persists a row, so a partial
                            // written while the prompt's eager save had failed would replay as an
                            // answer ahead of its question. A prompt the store still cannot take
                            // gets no row for its answer either.
                            if let Some(partial) = progress.partial.take() {
                                match recovery
                                    .ensure_prompt_saved(self, session_id, &user_message)
                                    .await
                                {
                                    Ok(()) => {
                                        self.keep_partial_answer(session_id, messages, partial)
                                            .await
                                    }
                                    Err(save_error) => tracing::warn!(
                                        "dropping the partial answer: failed to persist its \
                                         prompt: {save_error}"
                                    ),
                                }
                            }
                            break 'turn Err(error);
                        }
                        if let Err(error) = recovery
                            .repair_rejected_content(self, messages, error, &cancellation)
                            .await
                        {
                            break 'turn Err(error);
                        }
                        continue;
                    }
                };

                // The provider accepted this body, so everything in it is known-good and only what
                // comes after can be blamed for a later rejection.
                self.last_accepted_len
                    .store(sent_len, std::sync::atomic::Ordering::Relaxed);
                recovery.note_request_accepted();

                // Total of all tiers including output = everything in context as of this exchange,
                // which is what the next request re-sends (minus the new user prompt). Summing the
                // input tiers + output (Claude reports cached tokens in separate fields) is the
                // true occupancy and what the `/status` gauge and auto-compact threshold read.
                self.cells.context_tokens.store(
                    usage
                        .input_tokens
                        .saturating_add(usage.cache_creation_input_tokens)
                        .saturating_add(usage.cache_read_input_tokens)
                        .saturating_add(usage.output_tokens),
                    std::sync::atomic::Ordering::Relaxed,
                );
                turn_usage.input_tokens =
                    turn_usage.input_tokens.saturating_add(usage.input_tokens);
                turn_usage.output_tokens =
                    turn_usage.output_tokens.saturating_add(usage.output_tokens);
                turn_usage.cache_creation_input_tokens = turn_usage
                    .cache_creation_input_tokens
                    .saturating_add(usage.cache_creation_input_tokens);
                turn_usage.cache_read_input_tokens = turn_usage
                    .cache_read_input_tokens
                    .saturating_add(usage.cache_read_input_tokens);

                if let Err(error) = recovery
                    .ensure_prompt_saved(self, session_id, &user_message)
                    .await
                {
                    // The one exit between a 2xx and the persist below, so the repair the 2xx just
                    // vindicated has to be put back rather than left applied: persisting it on top
                    // of a store whose opening message is missing is exactly what the failure above
                    // forbids. Undoing also restores the trailing `Event::Append` that the
                    // post-loop `pop_unsaved` looks for, which a trailing `Event::Repair` would
                    // have made it silently skip, stranding the prompt in memory too.
                    recovery.undo_rejected_repair(self, messages);
                    break 'turn Err(error);
                }

                recovery.persist_vindicated_repair(self, session_id).await;

                // What the request budget took out of the body, recorded once, so the body that fit
                // is the body every later request sends and the cache prefix ahead of this turn
                // holds. Ahead of the interrupt and thinking-only exits below, because a redaction
                // is a fact about the request the provider just accepted rather than about how the
                // round ends. Before the round's own messages are appended, since the positions
                // are relative to the tail of the view as the request saw it. The turn's base slice
                // is rebuilt too, or the next request would be assembled from a copy that still
                // carries the images.
                let redacted = std::mem::take(&mut *crate::sync::lock(&self.pending_redactions));
                if !redacted.is_empty() {
                    let redaction = messages.redact_images(redacted);
                    recovery.refresh_base_messages(self, messages);
                    // Its own write rather than part of the round's, which the exits below never
                    // reach. A failed write leaves the view redacted in memory, so the turn goes
                    // on; the cost is one more redaction after a resume.
                    if let Err(error) = self.store.save_event(session_id, &redaction).await {
                        tracing::warn!("failed to persist the image redaction: {error}");
                    }
                }

                if cancellation.is_cancelled() {
                    // Interrupted mid-stream. Persist the partial assistant text so it survives
                    // resume instead of being discarded, but drop any `tool_use` blocks first: no
                    // tools run on an interrupt, so a persisted `tool_use` would be orphaned (no
                    // matching `tool_result`) and the provider would reject the next request. Only
                    // persist when text actually streamed; a partial with no text (interrupted
                    // before any output, or mid-thinking) has nothing worth restoring.
                    let partial = assistant_message.without_tool_use();
                    if partial
                        .content
                        .iter()
                        .any(|block| matches!(block, ContentBlock::Text { .. }))
                    {
                        messages.append(partial.clone());
                        if let Err(error) = self
                            .store
                            .save_events_atomic(session_id, vec![
                                crate::conversation::Event::Append(partial),
                            ])
                            .await
                        {
                            tracing::warn!(
                                "failed to persist interrupted partial assistant message: {error}"
                            );
                        }
                    }
                    break 'turn Err(MekaError::Interrupted);
                }

                // Run tools based on the *presence* of tool-call blocks, not the reported stop
                // reason: stop reasons are advisory and providers sometimes mislabel a tool turn as
                // a plain end, but any tool call the model made must be answered with a result or
                // the next request is invalid. Only complete tool calls reach the content blocks,
                // so executing whatever is present is safe.
                let has_tool_calls = assistant_message
                    .content
                    .iter()
                    .any(|block| matches!(block, ContentBlock::ToolUse { .. }));
                let has_visible_text = has_visible_text(&assistant_message.content);

                // The blocking path returns the message whole, with no event channel to have put
                // anything on while it was being written. Every frontend renders assistant text
                // from `AssistantTextDelta` and nothing else carries it, so without this a turn
                // that succeeded shows the user nothing at all. Emitted per round, matching the
                // streaming path, so text the model writes before a tool call still precedes the
                // indicator `execute_tool_calls` emits for it, and ahead of the thinking-only
                // nudge, so a round that reasoned without answering still shows its reasoning as
                // it does when streaming.
                //
                // Reasoning goes through the same shape, delta then block, which is what makes
                // `ThinkingDelta`'s promise hold on this path too: a frontend reading the deltas
                // gets one covering the whole block rather than nothing at all.
                if !self.options.streaming {
                    for block in &assistant_message.content {
                        match block {
                            ContentBlock::Text { text } => {
                                self.cells
                                    .frontend
                                    .emit(FrontendEvent::AssistantTextDelta(text.clone()))
                                    .await;
                            }
                            // `trim` rather than `is_empty`, matching the question replay asks of
                            // the same block in `render::render_message_history`: a block of
                            // whitespace has nothing to show, and the two must not disagree about
                            // that or a resumed transcript gains a block the live turn skipped.
                            ContentBlock::Thinking { thinking, .. }
                                if !thinking.trim().is_empty() =>
                            {
                                self.cells
                                    .frontend
                                    .emit(FrontendEvent::ThinkingDelta(thinking.clone()))
                                    .await;
                                self.cells
                                    .frontend
                                    .emit(FrontendEvent::ThinkingBlock {
                                        content: thinking.clone(),
                                    })
                                    .await;
                            }
                            ContentBlock::RedactedThinking { .. } => {
                                self.cells
                                    .frontend
                                    .emit(FrontendEvent::ThinkingDelta(
                                        crate::conversation::REDACTED_THINKING.to_string(),
                                    ))
                                    .await;
                                self.cells
                                    .frontend
                                    .emit(FrontendEvent::ThinkingBlock {
                                        content: crate::conversation::REDACTED_THINKING.to_string(),
                                    })
                                    .await;
                            }
                            _ => {}
                        }
                    }
                }

                if should_nudge_thinking_only(
                    has_tool_calls,
                    has_visible_text,
                    &stop_reason,
                    recovery.thinking_only_nudged,
                ) {
                    if let Err(error) = recovery
                        .nudge_thinking_only(
                            self,
                            session_id,
                            messages,
                            &assistant_message,
                            &stop_reason,
                        )
                        .await
                    {
                        break 'turn Err(error);
                    }
                    continue;
                }

                // No tool call and no visible text, and the nudge above didn't fire (already used
                // this turn, or a stop reason with its own handling such as refusal / max tokens).
                // Surface a stand-in in the assistant's place and persist it so the message is
                // non-empty: an empty content array is invalid on the next request and breaks
                // resume, and a silent turn leaves the user with nothing.
                if !has_tool_calls && !has_visible_text {
                    let notice = empty_turn_notice(&stop_reason);
                    self.cells
                        .frontend
                        .emit(FrontendEvent::AssistantTextDelta(notice.clone()))
                        .await;
                    assistant_message
                        .content
                        .push(ContentBlock::Text { text: notice });
                }

                // Append in memory now so the next iteration sees the full state; defer the DB save
                // to the branches below (atomic with results on the tool path, standalone
                // otherwise).
                messages.append(assistant_message.clone());
                let mut round_events = vec![crate::conversation::Event::Append(
                    assistant_message.clone(),
                )];

                if has_tool_calls {
                    // Surface a provider that mislabeled the stop reason, the bug this presence
                    // check guards against.
                    if !matches!(stop_reason, StopReason::ToolUse) {
                        tracing::warn!(
                            "assistant message carries tool calls but stop_reason is \
                             {stop_reason:?}; executing them anyway"
                        );
                    }

                    let mut tool_results = self
                        .execute_tool_calls(
                            &assistant_message,
                            &loaded,
                            attribution.prompt_id,
                            cancellation.clone(),
                        )
                        .await;

                    if let Err(error) = crate::tools::scratchpad::save_explicit_scratchpad_results(
                        &self.store,
                        session_id,
                        &self.tool_registry.inherited_scratchpad_names(),
                        &assistant_message,
                        &mut tool_results,
                    )
                    .await
                    {
                        tracing::warn!("failed to save explicit scratchpad results: {error}");
                    }

                    // Take the per-turn hints. This both snapshots them for the call below and
                    // clears them, so a long session doesn't accumulate entries for tool calls that
                    // already ran. No clone needed.
                    let hints_snapshot = std::mem::take(&mut *self.scratchpad_hints.write().await);
                    if let Err(error) = crate::tools::scratchpad::persist_oversized_results(
                        &self.store,
                        session_id,
                        &assistant_message,
                        &mut tool_results,
                        &hints_snapshot,
                    )
                    .await
                    {
                        tracing::warn!("failed to persist oversized tool results: {error}");
                    }

                    let result_message = Message {
                        role: Role::User,
                        content: tool_results,
                    };

                    // Save assistant + tool-results together in one transaction. Both rows commit
                    // or neither does: no dangling assistant-with-tool_use that the provider would
                    // reject on the next iteration.
                    //
                    // The results reach memory *before* the save is judged, for the same reason.
                    // The tools have run, so the only conversation that describes what happened
                    // is one that ends on their results; breaking out with the assistant's calls
                    // still unanswered left the session refused by the provider on every later
                    // turn until a `/rewind`. A store that cannot take the round leaves disk one
                    // round behind memory, which a resume reads as a prompt with no reply and
                    // never as a half round, since the pair is written as one unit or not at all.
                    round_events.push(crate::conversation::Event::Append(result_message.clone()));
                    let tool_calls = result_message.content.len();
                    messages.append(result_message);
                    if let Err(error) = self
                        .store
                        .save_events_atomic(session_id, round_events)
                        .await
                    {
                        tracing::warn!(
                            "failed to persist a tool round ({tool_calls} tool call(s)); a resume \
                             will not carry it: {error}"
                        );
                        break 'turn Err(error);
                    }

                    // A compaction `context_compact` asked for, run here rather than after the
                    // loop so the agent that chose the moment gets to act on the result: it takes
                    // its checkpoint, then this turn carries on against the summary.
                    //
                    // After the whole batch, not the moment the tool ran: a `context_compact`
                    // issued alongside other calls lets their results into the conversation being
                    // summarized, and `keep_recent` (default true) keeps that fresh tail verbatim.
                    //
                    // The guard is dropped before the `.await` below; held across one it would make
                    // this future non-`Send` and break every `tokio::spawn` of a turn.
                    let requested = crate::sync::lock(&self.cells.pending_compaction).take();
                    if let Some(request) = requested {
                        // An early-out, not the safety net. `context_compact` ignores its
                        // cancellation token, so the request outlives an interrupt; starting a
                        // compaction on a turn the user has stopped spends a checkpoint attempt
                        // and a summarizer call for a result that is then thrown away. The
                        // guarantee that the window survives lives at the other end, in
                        // `compact_session`, which refuses to rewrite on a fired token and catches
                        // the interrupt that arrives after this point too. The loop head breaks
                        // the turn on the next pass, so dropping the request here is all that is
                        // owed.
                        if cancellation.is_cancelled() {
                            tracing::debug!("dropping a compaction request on an interrupted turn");
                        } else if recovery.requested_compactions < MAX_REQUESTED_COMPACTIONS {
                            recovery.requested_compactions += 1;
                            tracing::info!("compacting at the agent's request");
                            match self
                                .compact_session(messages, request, cancellation.clone())
                                .await
                            {
                                // Every index the turn holds addresses the conversation this just
                                // replaced.
                                Ok(_) => recovery.after_conversation_rewrite(self, messages),
                                // An interrupt is not a failure to report: the loop's own check
                                // breaks the turn on the next pass, and warning here would put a
                                // line about compaction in front of every Ctrl+C that lands
                                // during one.
                                Err(_) if cancellation.is_cancelled() => {}
                                // Non-fatal otherwise, as it is on the post-loop path: the turn's
                                // own work is what the user asked for, and it can still finish
                                // uncompacted.
                                Err(error) => {
                                    tracing::warn!("requested compaction failed: {error}")
                                }
                            }
                        } else {
                            tracing::info!(
                                "ignoring a second compaction request in one turn; the agent may \
                                 ask again next turn"
                            );
                        }
                    }
                } else {
                    // No tool calls: the assistant message stands alone and ends the turn. Save it
                    // before breaking so the persistent log includes it.
                    if let Err(error) = self
                        .store
                        .save_events_atomic(session_id, round_events)
                        .await
                    {
                        break 'turn Err(error);
                    }
                    break 'turn match stop_reason {
                        StopReason::MaxTokens => Ok(TurnOutcome::MaxTokens),
                        StopReason::Refusal(text) if !text.is_empty() => {
                            Ok(TurnOutcome::Refusal(text))
                        }
                        // An empty refusal body carries no text, so fall back to the assistant
                        // message's text (the model's own refusal, or the stand-in above).
                        StopReason::Refusal(_) => {
                            Ok(TurnOutcome::Refusal(assistant_message.text_content()))
                        }
                        _ => Ok(TurnOutcome::EndTurn),
                    };
                }
            }
        };

        if result.is_ok() {
            // Roll the turn into the session-level counters surfaced by `/status`. Done here (not
            // inside the inner loop) so a single `/status` reading reflects whole turns, not
            // partial state.
            self.session_stats.record_turn(&turn_usage);
            // Persist the cumulative counters onto the session row so `/status` survives resume.
            // Best-effort: a DB hiccup must not fail the turn. Only the root agent writes; a
            // sub-agent shares the parent's `SessionStats` (rolling its usage into the parent's
            // totals) but owns a child session row, so letting it write would stamp the
            // parent-inclusive totals onto the child.
            if self.role.is_root()
                && let Err(error) = self
                    .store
                    .save_session_stats(session_id, &self.session_stats.snapshot())
                    .await
            {
                tracing::warn!("failed to persist session stats: {error}");
            }
            self.cells
                .frontend
                .emit(FrontendEvent::TokenUsage(turn_usage))
                .await;
            self.cells.frontend.emit(FrontendEvent::TurnFinished).await;
        }

        match &result {
            // A prompt its caller will produce again is withdrawn when the turn produced nothing at
            // all, however it ended: a recurring job regenerates it on its next occurrence, and a
            // drained `meka serve` hands the occurrence back, so keeping it guarantees a duplicate;
            // an HTTP client that asked for this resends. The log-length condition is the one that
            // decides: an unchanged count since the prompt means nothing appended, where the
            // materialized tail alone cannot tell a prompt from a compaction summary.
            // `a_fire_interrupted_before_it_began_withdraws_its_prompt` pins the shape.
            Err(_)
                if retention == PromptRetention::Withdraw
                    && messages.events_len() == recovery.prompt_only_events
                    && messages.ends_on_a_turn_opening() =>
            {
                recovery
                    .withdraw_unanswered_prompt(self, session_id, messages)
                    .await;
                *self.last_rendered_world.write().await = world_state_rollback;
                if resumed {
                    messages.restore_resumed_notice();
                }
            }
            Err(MekaError::Interrupted) if !recovery.user_saved => {
                let user_event = crate::conversation::Event::Append(user_message.clone());
                if let Err(error) = self.store.save_event(session_id, &user_event).await {
                    tracing::warn!("failed to save user message on interruption: {error}");
                }
            }
            // Saved rather than popped, because `Keep` means the prompt carries something that
            // exists nowhere else. This arm is reached only when the eager persist failed, so
            // popping would take a delivered background outcome out of the conversation as well as
            // off disk, and its row is already stamped and never handed out again.
            //
            // Reached by dropping `messages` under a live connection; see
            // `a_kept_prompt_survives_a_turn_whose_store_could_not_persist_it`.
            Err(error)
                if !matches!(error, MekaError::Interrupted)
                    && !recovery.user_saved
                    && retention == PromptRetention::Keep =>
            {
                let user_event = crate::conversation::Event::Append(user_message.clone());
                if let Err(error) = self.store.save_event(session_id, &user_event).await {
                    tracing::warn!("failed to save user message after a failed turn: {error}");
                }
            }
            Err(error) if !matches!(error, MekaError::Interrupted) && !recovery.user_saved => {
                if messages.pop_unsaved().is_some() {
                    self.cells
                        .frontend
                        .emit(FrontendEvent::PromptWithdrawn)
                        .await;
                }
                // The popped message carried this turn's world-state announcement, so put the
                // snapshot back to what the model has actually seen. The next turn then re-renders
                // the change rather than assuming it was already delivered.
                *self.last_rendered_world.write().await = world_state_rollback;
                // Same withdrawal for the resume notice, which rode that message and nothing else.
                if resumed {
                    messages.restore_resumed_notice();
                }
            }
            _ => {}
        }

        // The sweeper for a request the tool loop's own drain never reached. A turn that parked one
        // and then failed before that drain is the only way to arrive here holding a request, and
        // the `result.is_ok()` below then declines to act on it, so this exists to empty the slot,
        // not to compact: the slot outlives the turn.
        //
        // Taken in its own binding rather than inside the `if` below so the `std::sync::MutexGuard`
        // is dropped before the `.await`; held across one it would make this future non-`Send` and
        // break every `tokio::spawn` of a turn.
        //
        // Taken unconditionally, so a request left behind by a turn that then failed cannot linger
        // and fire against a later, unrelated turn.
        let requested_compaction = crate::sync::lock(&self.cells.pending_compaction).take();
        // Acted on only when the turn succeeded: an interrupted or failed turn has just popped or
        // repaired its own messages, and compacting on top of that would rewrite a conversation
        // still being put back together. The request is dropped rather than deferred; the agent can
        // ask again on a turn that works.
        if let Some(request) = requested_compaction
            && result.is_ok()
        {
            tracing::info!("compacting at the agent's request");
            if let Err(error) = self
                .compact_session(messages, request, cancellation.clone())
                .await
            {
                // Non-fatal, and deliberately not surfaced as a turn error: the turn itself
                // succeeded, and its answer is what the user asked for.
                tracing::warn!("requested compaction failed: {error}");
            }
        }

        result
    }

    /// Hand a provider advisory to the frontend, counting what it reports against this session.
    ///
    /// Counted here rather than by the provider, which is cached per profile and cannot tell whose
    /// session the request was. The one door for the turn's two attempt paths and both compaction
    /// requests, so a redaction during a summary reaches `/status` like one during a turn.
    pub(super) async fn forward_notice(&self, notice: crate::frontend::Notice) {
        if let Some(redaction) = &notice.redaction {
            // A retried attempt rebuilds the same body and reports the same positions again. Each
            // is recorded once, and a report that adds none is neither counted nor shown: the
            // counters would otherwise grow with every transient failure, and the user would read
            // one redaction as two.
            let adds_nothing = {
                let mut pending = crate::sync::lock(&self.pending_redactions);
                let fresh: Vec<_> = redaction
                    .positions
                    .iter()
                    .filter(|position| !pending.contains(position))
                    .cloned()
                    .collect();
                let adds_nothing = fresh.is_empty() && !redaction.positions.is_empty();
                pending.extend(fresh);
                adds_nothing
            };
            if adds_nothing {
                return;
            }
            self.session_stats.record_redaction(redaction);
        }
        self.cells
            .frontend
            .emit(FrontendEvent::Notice(notice))
            .await;
    }

    /// Keep the text of an answer the stream did not finish, in memory and in the store.
    async fn keep_partial_answer(
        &self,
        session_id: uuid::Uuid,
        messages: &mut Conversation,
        partial: Message,
    ) {
        let partial = partial.without_tool_use();
        if !partial
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::Text { .. }))
        {
            return;
        }
        messages.append(partial.clone());
        if let Err(error) = self
            .store
            .save_events_atomic(session_id, vec![crate::conversation::Event::Append(
                partial,
            )])
            .await
        {
            tracing::warn!("failed to persist the partial assistant message: {error}");
        }
    }

    /// Streaming provider call with bounded retry-with-backoff on transient failures
    /// ([`MekaError::RetryableProvider`]): 429/5xx (including Anthropic's 529 "overloaded") and,
    /// for Claude, a mid-stream `event: error` of a retryable type. Each attempt runs
    /// [`Self::run_streaming_attempt`] fresh (new channel, new spawned task, all accumulator state
    /// reinitialized); retries only fire when that attempt reports `content_started == false`, i.e.
    /// nothing has been forwarded to the frontend yet this attempt; retrying after the user has
    /// already seen partial output would duplicate or corrupt what's on screen. Nothing is
    /// persisted to the session DB until the whole turn's result is resolved (see `run_turn`),
    /// so a discarded attempt never leaves a partial write behind.
    pub(super) async fn run_streaming(
        &self,
        system_prompt: Arc<str>,
        messages: Arc<[Message]>,
        tools: Arc<[ToolDefinition]>,
        attribution: crate::provider::Attribution,
        cancellation: CancellationToken,
        progress: &mut StreamProgress,
    ) -> Result<(Message, StopReason, crate::stats::TokenUsage)> {
        let mut retries = 0u32;
        let started = std::time::Instant::now();
        loop {
            // Reported back out as well as read here, because `run_turn` needs the same fact for
            // the same reason: whatever it does with a failure, it must not re-send a request whose
            // output the user has already seen.
            *progress = StreamProgress::default();
            match self
                .run_streaming_attempt(
                    Arc::clone(&system_prompt),
                    Arc::clone(&messages),
                    Arc::clone(&tools),
                    attribution.clone(),
                    cancellation.clone(),
                    progress,
                )
                .await
            {
                Ok(value) => return Ok(value),
                Err(error) => match should_retry_provider_error(
                    &error,
                    progress.content_started,
                    retries,
                    started.elapsed(),
                ) {
                    Some(delay) => {
                        retries += 1;
                        let ceiling = crate::provider::retry::MAX_PROVIDER_RETRIES;
                        tracing::warn!(
                            "provider stream failed transiently (attempt {retries}/{ceiling}), \
                             retrying in {delay:?}: {error}"
                        );
                        tokio::select! {
                            _ = tokio::time::sleep(delay) => {}
                            _ = cancellation.cancelled() => return Err(MekaError::Interrupted),
                        }
                    }
                    None => return Err(error),
                },
            }
        }
    }

    /// A single streaming attempt: spawns `provider.stream(...)`, drains its `StreamEvent`s into a
    /// `Message`. `content_started` is set the instant anything user-visible is forwarded to the
    /// frontend; see [`Self::run_streaming`], which reads it back to decide whether a failure is
    /// safe to retry.
    pub(super) async fn run_streaming_attempt(
        &self,
        system_prompt: Arc<str>,
        messages: Arc<[Message]>,
        tools: Arc<[ToolDefinition]>,
        attribution: crate::provider::Attribution,
        cancellation: CancellationToken,
        progress: &mut StreamProgress,
    ) -> Result<(Message, StopReason, crate::stats::TokenUsage)> {
        // Bounded so a provider streaming faster than the renderer consumes can't grow memory
        // without limit. 1024 is far above any realistic in-flight backlog, so backpressure
        // effectively never engages.
        let (event_sender, mut event_receiver) = mpsc::channel::<StreamEvent>(1024);

        let provider = self.provider();
        let cancellation_clone = cancellation.clone();
        // The request travels into the spawned task whole, attribution included: nothing here
        // relies on task-local state surviving a spawn, because nothing does.
        let stream_handle = tokio::spawn(async move {
            provider
                .stream(
                    CompletionRequest::new(&system_prompt, &messages, &tools)
                        .attributed(attribution),
                    event_sender,
                    cancellation_clone,
                )
                .await
        });

        let mut accumulator = crate::provider::MessageAccumulator::new();
        // Whether the provider has answered at all. A driver emits nothing before `succeeded` has
        // seen a 2xx, except the redaction notice, which the Claude drivers queue before the
        // request is even sent, so any other event is proof the request was judged.
        let mut response_started = false;

        while let Some(event) = event_receiver.recv().await {
            if !matches!(event, StreamEvent::Notice(_)) {
                response_started = true;
            }
            // What the user sees is decided here, event by event; what the conversation keeps is
            // the accumulator's one answer, shared with every other reader of a stream.
            match &event {
                StreamEvent::ThinkingDelta(text) => {
                    // Marked like `StreamEvent::TextDelta` marks its first chunk, and for the same
                    // reason: this is model output a consumer now holds, and a retry would send it
                    // again. Not the `Notice` exemption, since a notice is meka's own advisory,
                    // queued before the request is even sent.
                    //
                    // Asked of the frontend rather than assumed, because reasoning is the *first*
                    // thing a turn produces: marking every turn that reasoned would refuse the
                    // retry for almost any mid-turn failure, which is exactly when an overloaded
                    // provider needs one. A frontend that discards these loses nothing to a retry.
                    if self.cells.frontend.retains_reasoning() {
                        progress.content_started = true;
                    }
                    self.cells
                        .frontend
                        .emit(FrontendEvent::ThinkingDelta(text.clone()))
                        .await;
                }
                StreamEvent::ThinkingProgress { estimated_tokens } => {
                    // Deliberately does not set `content_started`: this is a transient indicator
                    // the frontend erases, not output the turn produced. Counting it would make an
                    // interrupted think look like a partial answer.
                    self.cells
                        .frontend
                        .emit(FrontendEvent::ThinkingProgress {
                            estimated_tokens: *estimated_tokens,
                        })
                        .await;
                }
                StreamEvent::ThinkingComplete { .. } => {
                    let content = accumulator.pending_thinking();
                    // Whether there is anything to *show*, which is the question
                    // `render::render_message_history` asks of the same block on replay, so a
                    // resumed transcript cannot gain or lose a block against the live turn.
                    //
                    // The deltas above are not held to it: whitespace is content mid-block (the
                    // Responses API separates two summary parts with a bare `\n\n` delta), so the
                    // question can only be asked once the block is whole, which is here. Whether
                    // the block is *kept* is the accumulator's answer, and asks the raw text: what
                    // the provider will accept back is not a question about what is worth showing.
                    let showable = !content.trim().is_empty();
                    if showable {
                        progress.content_started = true;
                        self.cells
                            .frontend
                            .emit(FrontendEvent::ThinkingBlock {
                                content: content.to_string(),
                            })
                            .await;
                    } else {
                        // Nothing to render, but the block is over: say so, so a frontend showing a
                        // live indicator can close it here instead of holding the line open for an
                        // event that may never come.
                        self.cells.frontend.emit(FrontendEvent::ThinkingEnded).await;
                    }
                }
                StreamEvent::RedactedThinking { .. } => {
                    progress.content_started = true;
                    // Delta then block, like every other block carrying visible text. The marker is
                    // meka's own words rather than the model's, but a frontend that reads the
                    // deltas would otherwise be the only one that never hears about a redacted
                    // block at all.
                    self.cells
                        .frontend
                        .emit(FrontendEvent::ThinkingDelta(
                            crate::conversation::REDACTED_THINKING.to_string(),
                        ))
                        .await;
                    self.cells
                        .frontend
                        .emit(FrontendEvent::ThinkingBlock {
                            content: crate::conversation::REDACTED_THINKING.to_string(),
                        })
                        .await;
                }
                StreamEvent::TextDelta(text) => {
                    progress.content_started = true;
                    self.cells
                        .frontend
                        .emit(FrontendEvent::AssistantTextDelta(text.clone()))
                        .await;
                }
                StreamEvent::ToolUseStart { id, name } => {
                    // Deliberately does not set `content_started`, for the same reason
                    // `ThinkingProgress` doesn't: a call whose arguments never finish produced no
                    // output, and the flag is what decides whether a mid-stream failure is still
                    // safe to retry.
                    self.cells
                        .frontend
                        .emit(FrontendEvent::ToolCallComposing {
                            id: id.clone(),
                            name: name.clone(),
                        })
                        .await;
                }
                StreamEvent::ToolUseEnd { input } => {
                    progress.content_started = true;
                    let (id, name) = accumulator.pending_tool();
                    let schema = self
                        .tool_registry
                        .get(name)
                        .map(|t| t.definition().parameters);
                    let display_summary =
                        crate::tools::resolve_primary_param(name, input, schema.as_ref());
                    self.cells
                        .frontend
                        .emit(FrontendEvent::ToolCallStarted {
                            id: id.to_string(),
                            name: name.to_string(),
                            input: input.clone(),
                            display_summary,
                        })
                        .await;
                }
                StreamEvent::ToolCallRejected { id, name, reason } => {
                    // A malformed tool-call arrived (bad JSON). The accumulator keeps a `ToolUse`
                    // block with a sentinel marker so the shape of the assistant message stays
                    // valid for the API round-trip, and `resolve_and_execute_tool` sees the marker
                    // and surfaces an error back to the model rather than running the tool on a
                    // silently-empty argument object. Shown the same way.
                    progress.content_started = true;
                    let marker_input = serde_json::json!({
                        crate::provider::INVALID_TOOL_ARGS_MARKER: reason,
                    });
                    let schema = self
                        .tool_registry
                        .get(name)
                        .map(|t| t.definition().parameters);
                    let display_summary =
                        crate::tools::resolve_primary_param(name, &marker_input, schema.as_ref());
                    self.cells
                        .frontend
                        .emit(FrontendEvent::ToolCallStarted {
                            id: id.clone(),
                            name: name.clone(),
                            input: marker_input,
                            display_summary,
                        })
                        .await;
                }
                StreamEvent::MessageEnd { .. } | StreamEvent::Usage(_) => {}
                StreamEvent::Notice(notice) => {
                    // Provider advisories (image redaction) are emitted inline so the user sees
                    // them in order with the assistant text that follows.
                    //
                    // Deliberately does not set `content_started`: that flag exists so a retry
                    // cannot double-emit model output, and a notice is not model output. The
                    // Claude providers queue the image-redaction advisory before the request is
                    // even sent, so marking it would disable retry for the whole turn from the
                    // first event onward; `forward_notice` drops the advisory a retry repeats.
                    self.forward_notice(notice.clone()).await;
                }
                StreamEvent::Error(error) => {
                    // Log only, no return: every producer of this event sends it immediately
                    // before its own typed `Err` return (see `provider::sse`), so the channel
                    // closes and the `stream_handle.await` below surfaces the original typed
                    // error, which `run_streaming`'s retry logic needs to see.
                    //
                    // Close out any thinking before logging: a failed turn emits no `TurnFinished`
                    // and `ThinkingComplete` only comes from a `content_block_stop` this stream
                    // never reached, so a frontend drawing a live indicator would hold its line
                    // open, and the log below would print onto that row. Sent unconditionally
                    // because every frontend ignores it when nothing is drawn.
                    self.cells.frontend.emit(FrontendEvent::ThinkingEnded).await;
                    tracing::error!("stream error: {error}");
                }
            }
            accumulator.push(event);
        }

        match stream_handle.await {
            Ok(Ok(())) => {}
            Ok(Err(MekaError::Interrupted)) if !response_started => {
                // Stopped before the provider answered, so nothing has been judged: this is the
                // whole-reply interrupt and not a partial answer. Falling through would book the
                // request as accepted and persist a repair the provider never saw.
                return Err(MekaError::Interrupted);
            }
            Ok(Err(MekaError::Interrupted)) => {
                // Interrupted mid-stream, after a 2xx. Fall through to return partial content. The
                // caller detects interruption via the cancellation token.
            }
            Ok(Err(error)) => {
                progress.partial = accumulator.partial();
                return Err(error);
            }
            Err(join_error) => {
                return Err(MekaError::Provider(format!(
                    "stream task panicked: {join_error}"
                )));
            }
        }

        let crate::provider::Completion {
            message,
            stop_reason,
            usage,
            ..
        } = accumulator.finish();
        Ok((message, stop_reason, usage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        agent::tests::{
            REJECTION, agent_for_test, agent_that_compacts_for_test, agent_with_registry_for_test,
            assistant_message, assistant_tool_use, build_test_agent, image_source,
            send_file_registry, tool_result_message, user_message,
        },
        conversation::ToolResultContent,
        provider::mock::text_round,
    };
    /// The text that streamed before the connection died is kept: it was on screen, and losing it
    /// from the conversation and the store makes the user re-ask for what they had already read.
    #[tokio::test]
    async fn a_mid_stream_failure_keeps_the_text_that_had_arrived() {
        let provider = Arc::new(MockProvider::from_rounds(vec![vec![
            MockEvent::Text {
                text: "the first half".to_string(),
            },
            MockEvent::FailStream {
                message: "connection reset".to_string(),
            },
        ]]));
        let (agent, store) = agent_for_test(provider as Arc<dyn Provider>).await;
        let mut messages = Conversation::new();
        let outcome = agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("hello".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await;
        assert!(
            matches!(outcome, Err(MekaError::StreamError(_))),
            "the failure is still reported: {outcome:?}"
        );
        let last = messages.as_slice().last().expect("the partial answer");
        assert_eq!(last.role, Role::Assistant);
        assert_eq!(last.text_content(), "the first half");
        let stored = store
            .load_events(agent.session_id().expect("the turn created the session"))
            .await
            .expect("load");
        assert!(
            stored.iter().any(|event| matches!(
                event,
                crate::conversation::Event::Append(message) if message.text_content() == "the first half"
            )),
            "and it is in the store, so a resume shows it: {stored:?}"
        );
    }

    use std::sync::Arc;

    use tokio_util::sync::CancellationToken;

    use super::PromptRetention;
    use crate::{
        agent::tests::agent_at_for_test,
        conversation::{Conversation, Message, Role},
        error::MekaError,
        provider::{
            Provider,
            mock::{MockEvent, MockProvider},
        },
    };

    fn unreachable_provider(rounds: usize) -> Arc<MockProvider> {
        Arc::new(MockProvider::from_rounds(
            (0..rounds)
                .map(|_| {
                    vec![MockEvent::Fail {
                        message: "error sending request: connection refused".to_string(),
                    }]
                })
                .collect(),
        ))
    }

    /// The case that motivated this: meka up, provider unreachable, a job firing on its interval
    /// for the length of the outage. Each fire persists its prompt before the call and then fails,
    /// so without withdrawal the conversation collects one unanswered message per fire.
    #[tokio::test]
    async fn a_provider_outage_leaves_no_residue_from_scheduled_fires() {
        let (agent, store) = agent_for_test(unreachable_provider(12)).await;
        let mut messages = Conversation::new();

        for _ in 0..12 {
            agent
                .run_turn(
                    &mut messages,
                    crate::agent::TurnInput::from_parts(
                        "[Scheduled job 7f3a1b2c fired] check the news".to_string(),
                        Vec::new(),
                    )
                    .expect("a prompt")
                    .retaining(PromptRetention::Withdraw),
                    CancellationToken::new(),
                )
                .await
                .expect_err("the provider is unreachable");
        }

        assert!(
            messages.is_empty(),
            "a day of failed fires must not accumulate: {:?}",
            messages.as_slice()
        );
        // And the withdrawal reached disk, so resuming the session does not bring them back.
        let session_id = agent
            .session_id()
            .expect("the first turn created the session");
        let events = store.load_events(session_id).await.expect("load events");
        assert!(
            Conversation::from_events(events).is_empty(),
            "the materialized view is empty after a reload too"
        );
    }

    /// A `Keep` prompt survives a failed turn, in the conversation and on disk.
    ///
    /// `Keep` exists because the prompt may carry something that exists nowhere else: a
    /// background outcome, whose row is stamped delivered before the turn starts and is never
    /// handed out again. A failed turn that discards it therefore destroys the only copy, and the
    /// user's retry finds nothing left to retry with.
    ///
    /// This covers the ordinary failure, where the eager persist succeeded and the withdrawal arm
    /// must decline to fire. The `!user_saved` arm beside it is covered by
    /// `a_kept_prompt_survives_a_turn_whose_store_could_not_persist_it`.
    #[tokio::test]
    async fn a_kept_prompt_survives_a_failed_turn() {
        let (agent, store) = agent_for_test(unreachable_provider(1)).await;
        let mut messages = Conversation::new();

        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts(
                    "[Background task reporting] 7f3a1c22 was canceled after 11s.".to_string(),
                    Vec::new(),
                )
                .expect("a prompt")
                .retaining(PromptRetention::Keep),
                CancellationToken::new(),
            )
            .await
            .expect_err("the provider is unreachable");

        assert!(
            messages
                .as_slice()
                .iter()
                .any(|message| format!("{message:?}").contains("was canceled")),
            "the outcome must still be in the conversation: {:?}",
            messages.as_slice()
        );
        let session_id = agent
            .session_id()
            .expect("the first turn created the session");
        let events = store.load_events(session_id).await.expect("load events");
        assert!(
            !Conversation::from_events(events).is_empty(),
            "and on disk, so a resume still carries it"
        );
    }

    /// A `Keep` prompt survives a failed turn whose *store* failed too.
    ///
    /// This is the arm the sibling above cannot reach: `pop_unsaved` runs only when the eager
    /// persist failed, so a healthy store never gets there, and popping would take a delivered
    /// background outcome out of the conversation as well as off disk, its row already stamped and
    /// never handed out again. `SQLITE_BUSY` with a second meka on the store is the ordinary way
    /// in.
    ///
    /// The store is broken by dropping the table under a live connection, the same way
    /// `a_turn_whose_store_breaks_does_not_announce_every_memory_as_deleted` does it: a real
    /// failure on the real write path rather than a stub.
    #[tokio::test]
    async fn a_kept_prompt_survives_a_turn_whose_store_could_not_persist_it() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("meka.db");
        let (agent, store) = agent_at_for_test(unreachable_provider(1), &path).await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        agent.cells().session_id.set(session_id);
        let mut messages = Conversation::new();

        // Between creating the session and running the turn, so the session exists and only the
        // message write fails, which is exactly the state the arm is guarded on.
        rusqlite::Connection::open(&path)
            .expect("second connection")
            .execute_batch("DROP TABLE messages;")
            .expect("drop the table");

        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts(
                    "[Background task reporting] 7f3a1c22 was canceled after 11s.".to_string(),
                    Vec::new(),
                )
                .expect("a prompt")
                .retaining(PromptRetention::Keep),
                CancellationToken::new(),
            )
            .await
            .expect_err("the provider is unreachable");

        assert!(
            messages
                .as_slice()
                .iter()
                .any(|message| format!("{message:?}").contains("was canceled")),
            "the outcome is the only copy there is, so a store that cannot hold it must not be a \
             reason to drop it as well: {:?}",
            messages.as_slice()
        );
    }

    /// Hide the store's `messages` table under a live connection, and put it back. The eager save
    /// fails while it is hidden; every write after the restore succeeds, which is the sequence a
    /// transient `SQLITE_BUSY` on the first write produces.
    fn hide_messages_table(path: &std::path::Path, hidden: bool) {
        let statement = if hidden {
            "ALTER TABLE messages RENAME TO messages_hidden;"
        } else {
            "ALTER TABLE messages_hidden RENAME TO messages;"
        };
        rusqlite::Connection::open(path)
            .expect("second connection")
            .execute_batch(statement)
            .expect("rename the table");
    }

    /// The row order on disk after a turn whose prompt could not be saved eagerly and whose
    /// stream then died with text on screen: the prompt first, then the partial answer. A partial
    /// persisted ahead of the lazy prompt save would replay as an answer before its question, and
    /// a `Withdraw` turn's `pop_unsaved` would take it out of memory in the prompt's place.
    #[tokio::test]
    async fn a_partial_answer_never_lands_on_disk_ahead_of_its_prompt() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("meka.db");
        let provider = Arc::new(MockProvider::from_rounds(vec![vec![
            MockEvent::Sleep { ms: 300 },
            MockEvent::Text {
                text: "the first half".to_string(),
            },
            MockEvent::FailStream {
                message: "connection reset".to_string(),
            },
        ]]));
        let (agent, store) = agent_at_for_test(provider as Arc<dyn Provider>, &path).await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        agent.cells().session_id.set(session_id);
        let mut messages = Conversation::new();

        hide_messages_table(&path, true);
        let restore = tokio::spawn({
            let path = path.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                hide_messages_table(&path, false);
            }
        });

        let outcome = agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("hello".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await;
        restore.await.expect("the table was restored");
        assert!(
            matches!(outcome, Err(MekaError::StreamError(_))),
            "the failure is still reported: {outcome:?}"
        );

        let stored: Vec<String> = store
            .load_events(session_id)
            .await
            .expect("load")
            .into_iter()
            .filter_map(|event| match event {
                crate::conversation::Event::Append(message) => Some(message.text_content()),
                _ => None,
            })
            .collect();
        let prompt_at = stored
            .iter()
            .position(|text| text.contains("hello"))
            .expect("the prompt reached disk once the store recovered");
        let partial_at = stored
            .iter()
            .position(|text| text == "the first half")
            .expect("the partial answer reached disk");
        assert!(
            prompt_at < partial_at,
            "the prompt must precede its answer on disk: {stored:?}"
        );
        assert_eq!(
            messages.as_slice().len(),
            stored.len(),
            "memory and disk describe the same conversation: {:?} vs {stored:?}",
            messages.as_slice()
        );
    }

    /// A compaction that runs inside the turn persists the prompt with its tail, so it counts as
    /// the prompt's persist, or the lazy save on the 2xx writes a second copy after the boundary
    /// whenever the eager one had failed.
    #[tokio::test]
    async fn a_compaction_inside_the_turn_counts_as_the_prompts_persist() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("meka.db");
        let provider = Arc::new(MockProvider::from_rounds(vec![
            // The summarizer, slowed so the store is back before the rewrite is persisted.
            vec![MockEvent::Sleep { ms: 300 }, MockEvent::Text {
                text: "summary".to_string(),
            }],
            vec![MockEvent::Text {
                text: "the answer".to_string(),
            }],
        ]));
        let (mut agent, store) = agent_at_for_test(provider as Arc<dyn Provider>, &path).await;
        agent.options.auto_compact = true;
        // Two prior messages of ~1000 estimated tokens each overrun 80% of this window, so the
        // proactive projection compacts before the first request.
        agent.set_context_window_for_test(2_000);
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        agent.cells().session_id.set(session_id);
        let mut messages = Conversation::new();
        let body = "x".repeat(4_000);
        messages.append(Message::user(format!("earlier {body}")));
        messages.append(Message {
            role: Role::Assistant,
            content: vec![crate::conversation::ContentBlock::Text {
                text: format!("earlier answer {body}"),
            }],
        });

        hide_messages_table(&path, true);
        let restore = tokio::spawn({
            let path = path.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                hide_messages_table(&path, false);
            }
        });

        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("the prompt".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the turn succeeds once the store is back");
        restore.await.expect("the table was restored");

        let events = store.load_events(session_id).await.expect("load");
        assert!(
            events
                .iter()
                .any(|event| matches!(event, crate::conversation::Event::CompactBoundary { .. })),
            "the proactive compaction ran: {events:?}"
        );
        let prompt_copies = events
            .iter()
            .filter(|event| matches!(
                event,
                crate::conversation::Event::Append(message)
                    if message.role == Role::User && message.text_content().contains("the prompt")
            ))
            .count();
        assert_eq!(
            prompt_copies, 1,
            "the prompt is on disk exactly once: {events:?}"
        );
        assert_eq!(
            Conversation::from_events(events).as_slice().len(),
            messages.as_slice().len(),
            "a replay of the store is the conversation in memory: {:?}",
            messages.as_slice()
        );
    }

    /// A tool round whose save fails still answers every `tool_use` in memory.
    ///
    /// The tools have run, so the only conversation that describes what happened is one that ends
    /// on their results; breaking out with the assistant's calls unanswered would leave the
    /// provider refusing every later request until a `/rewind`. Disk may be one round behind; what
    /// it must never hold is half a round.
    #[tokio::test]
    async fn a_tool_round_whose_save_fails_still_answers_its_calls_in_memory() {
        use crate::provider::mock::MockStopReason;

        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("meka.db");
        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![
                MockEvent::Sleep { ms: 300 },
                MockEvent::ToolUseStart {
                    id: "call-1".to_string(),
                    name: "does_not_exist".to_string(),
                },
                MockEvent::ToolUseEnd {
                    input: serde_json::json!({}),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::ToolUse,
                },
            ],
            text_round("carrying on"),
        ]));
        let (agent, store) = agent_at_for_test(provider as Arc<dyn Provider>, &path).await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        agent.cells().session_id.set(session_id);
        let mut messages = Conversation::new();

        // After the eager prompt save and before the round's own, which is the window the sleep
        // above holds open.
        let hide = tokio::spawn({
            let path = path.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                hide_messages_table(&path, true);
            }
        });
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("go".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect_err("the round's save fails");
        hide.await.expect("the table was hidden");

        let last = messages.as_slice().last().expect("the round is in memory");
        assert_eq!(last.role, Role::User, "{:?}", messages.as_slice());
        assert!(
            matches!(
                last.content.first(),
                Some(ContentBlock::ToolResult { tool_use_id, .. }) if tool_use_id == "call-1"
            ),
            "memory ends on the tool result, so every call is answered: {:?}",
            messages.as_slice()
        );

        hide_messages_table(&path, false);
        let stored = store.load_events(session_id).await.expect("load");
        assert!(
            !stored.iter().any(|event| matches!(
                event,
                crate::conversation::Event::Append(message)
                    if message.content.iter().any(|block| matches!(block, ContentBlock::ToolUse { .. }))
            )),
            "disk is one round behind, never half a round: {stored:?}"
        );
        let mut resumed = Conversation::from_events(stored);
        assert!(
            resumed.sanitize_orphans().is_empty(),
            "so a resume has nothing to drop"
        );

        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("more".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the next turn is not refused for an unanswered tool call");
    }

    /// The thinking-only nudge moves memory only once its pair is on disk: appended first, a save
    /// that then fails leaves the reasoning in the conversation with nothing behind it in the
    /// store.
    #[tokio::test]
    async fn a_nudge_whose_save_fails_leaves_memory_untouched() {
        use crate::provider::mock::MockStopReason;

        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("meka.db");
        let provider = Arc::new(MockProvider::from_rounds(vec![vec![
            MockEvent::Sleep { ms: 300 },
            MockEvent::ThinkingDelta {
                text: "weighing the options".to_string(),
            },
            MockEvent::ThinkingComplete { opaque: None },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::EndTurn,
            },
        ]]));
        let (agent, store) = agent_at_for_test(provider as Arc<dyn Provider>, &path).await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        agent.cells().session_id.set(session_id);
        let mut messages = Conversation::new();

        let hide = tokio::spawn({
            let path = path.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                hide_messages_table(&path, true);
            }
        });
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("hello".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect_err("the nudge's save fails");
        hide.await.expect("the table was hidden");

        assert_eq!(
            messages.as_slice().len(),
            1,
            "the prompt alone: neither the reasoning nor the nudge reached the store, so neither \
             may be in memory: {:?}",
            messages.as_slice()
        );
    }

    /// An interrupted round still records what the request budget redacted: a redaction is a fact
    /// about the request the provider accepted, not about how the round ended, and dropping it
    /// with the round makes every later request redact the same image again.
    #[tokio::test]
    async fn an_interrupted_round_still_records_its_redaction() {
        use crate::{
            conversation::{Event, RedactedImage},
            provider::mock::MockStopReason,
        };

        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![
                MockEvent::Redaction {
                    images: 1,
                    bytes: 4,
                    // The request is [user(image), assistant, user(prompt)]: the image sits
                    // three from the end, second block of its message.
                    positions: vec![RedactedImage {
                        from_end: 3,
                        block: 1,
                        item: None,
                    }],
                },
                // Stalled rather than slept: a stop cuts a slept reply short, and this one has
                // to arrive, redaction and all, for there to be anything to record.
                MockEvent::Stall { ms: 300 },
                MockEvent::Text {
                    text: "half".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
            text_round("again"),
        ]));
        let (mut agent, store) = agent_for_test(Arc::clone(&provider) as Arc<dyn Provider>).await;
        // Blocking, so the mock records the messages behind each request in `completions`, and so
        // the stop lands between the provider's answer and the round's own bookkeeping.
        agent.options.streaming = false;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("session");
        agent.cells().session_id.set(session_id);
        for event in [
            Event::Append(Message::user_with_images("look", vec![image_source()])),
            Event::Append(Message::assistant_text("seen")),
        ] {
            store
                .save_event(session_id, &event)
                .await
                .expect("seed the conversation");
        }
        let mut events = store.load_events(session_id).await.expect("load");
        store.inline_blobs(&mut events).await.expect("inline");
        let mut messages = Conversation::from_events(events);

        let cancellation = CancellationToken::new();
        let stop = tokio::spawn({
            let cancellation = cancellation.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                cancellation.cancel();
            }
        });
        let outcome = agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("go".to_string(), Vec::new())
                    .expect("a prompt"),
                cancellation,
            )
            .await;
        stop.await.expect("the stop landed");
        assert!(
            matches!(outcome, Err(MekaError::Interrupted)),
            "the round was interrupted: {outcome:?}"
        );

        let stored = store.load_events(session_id).await.expect("load again");
        assert!(
            stored
                .iter()
                .any(|event| matches!(event, Event::Redact { images } if images.len() == 1)),
            "the redaction is on disk although the round was interrupted: {stored:?}"
        );

        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("more".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the next turn completes");
        let sent = provider.completions();
        assert_eq!(sent.len(), 2, "one request per turn");
        assert!(
            matches!(
                &sent[1][0].content[1],
                ContentBlock::Text { text } if text == crate::conversation::IMAGE_REDACTION_PLACEHOLDER
            ),
            "the next request carries the placeholder, not the image: {:?}",
            sent[1][0].content
        );
    }

    /// A whole reply is silent until the provider has finished it, and a stop must not wait that
    /// out. The mock's slept reply stands in for one still being generated.
    #[tokio::test]
    async fn an_interrupt_drops_a_whole_reply_still_being_generated() {
        use crate::provider::mock::MockStopReason;

        let provider = Arc::new(MockProvider::from_rounds(vec![vec![
            MockEvent::Sleep { ms: 3000 },
            MockEvent::Text {
                text: "never delivered".to_string(),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::EndTurn,
            },
        ]]));
        let (mut agent, store) = agent_for_test(Arc::clone(&provider) as Arc<dyn Provider>).await;
        agent.options.streaming = false;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("session");
        agent.cells().session_id.set(session_id);
        let mut messages = Conversation::new();

        let cancellation = CancellationToken::new();
        let stop = tokio::spawn({
            let cancellation = cancellation.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                cancellation.cancel();
            }
        });
        let started = std::time::Instant::now();
        let outcome = agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("go".to_string(), Vec::new())
                    .expect("a prompt"),
                cancellation,
            )
            .await;
        stop.await.expect("the stop landed");
        assert!(
            matches!(outcome, Err(MekaError::Interrupted)),
            "the turn was interrupted: {outcome:?}"
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "the stop must drop the reply, not wait for it: {:?}",
            started.elapsed()
        );
    }

    /// `meka serve` cancels its shutdown token before draining, and a scheduled turn's token is a
    /// child of it, so a job due during a shutdown is interrupted with its prompt already on disk,
    /// and the occurrence is then handed back, so the identical prompt arrives again on the next
    /// run. Keeping it would guarantee a duplicate, which is why interruption withdraws too.
    #[tokio::test]
    async fn a_fire_interrupted_before_it_began_withdraws_its_prompt() {
        let (agent, _store) = agent_for_test(unreachable_provider(1)).await;
        let mut messages = Conversation::new();
        let canceled = CancellationToken::new();
        canceled.cancel();

        let error = agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts(
                    "[Scheduled job 7f3a1b2c fired] check the news".to_string(),
                    Vec::new(),
                )
                .expect("a prompt")
                .retaining(PromptRetention::Withdraw),
                canceled,
            )
            .await
            .expect_err("a canceled token stops the turn before the provider is reached");
        assert!(matches!(error, crate::error::MekaError::Interrupted));
        assert!(
            messages.is_empty(),
            "the occurrence comes back, so the prompt must not linger: {:?}",
            messages.as_slice()
        );
    }

    /// The announcement of a change rides the prompt of the first turn to see it, and the snapshot
    /// advances as if the model had been told. Withdrawing that prompt takes the only copy of the
    /// announcement with it, so the snapshot has to go back too, or the next turn believes the
    /// change was delivered and never mentions it.
    #[tokio::test]
    async fn a_withdrawn_prompt_gives_its_announcement_back_to_the_next_turn() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Store::open(Some(&temp.path().join("meka.db")), &Default::default())
            .await
            .expect("open");
        let memories = store.memory_store(true);
        let registry = crate::tools::ToolRegistry::new();
        registry
            .register(Arc::new(MemoryReadFixture))
            .expect("register memory_read");
        let provider = Arc::new(MockProvider::from_rounds(vec![
            text_round("first"),
            vec![MockEvent::Fail {
                message: "error sending request: connection refused".to_string(),
            }],
            text_round("third"),
        ]));
        let (mut agent, _unused) =
            agent_with_registry_for_test(provider as Arc<dyn Provider>, registry).await;
        agent.store = store.clone();
        agent.memories = memories.clone();
        agent.options.system_prompt_override = None;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("first".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("first turn");

        memories
            .write(crate::store::memory::WriteRequest {
                name: "deploy-policy".to_string(),
                description: Some("Never deploy on Fridays".to_string()),
                tags: None,
                body: None,
                priority: Some(3),
            })
            .await
            .expect("write");

        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("doomed".to_string(), Vec::new())
                    .expect("a prompt")
                    .retaining(PromptRetention::Withdraw),
                CancellationToken::new(),
            )
            .await
            .expect_err("the provider is unreachable");
        assert!(
            !messages
                .as_slice()
                .iter()
                .any(|message| message.wire_text().contains("deploy-policy")),
            "the premise: the withdrawn prompt took the announcement with it"
        );

        let before = messages.len();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("third".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("third turn");
        let third: String = messages.as_slice()[before..]
            .iter()
            .map(|message| message.wire_text())
            .collect();
        assert!(
            third.contains("deploy-policy"),
            "the change was never delivered, so the next turn has to say it: {third}"
        );
    }

    /// The withdrawal is announced where it happens, so a host can tell a client whether the
    /// conversation still holds the prompt it sent. A kept prompt announces nothing: keeping is
    /// the absence of the act, and the host reads it as such.
    #[tokio::test]
    async fn a_withdrawal_is_announced_once_and_a_kept_prompt_not_at_all() {
        let (agent, frontend) = agent_recording_for_test(unreachable_provider(2)).await;
        let announced = |frontend: &crate::frontend::testing::RecordingFrontend| {
            frontend
                .events()
                .iter()
                .filter(|event| matches!(event, crate::frontend::FrontendEvent::PromptWithdrawn))
                .count()
        };
        let mut messages = Conversation::new();

        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("kept".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect_err("the provider is unreachable");
        assert_eq!(
            announced(&frontend),
            0,
            "a kept prompt is not announced as withdrawn: {:?}",
            frontend.events()
        );

        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("withdrawn".to_string(), Vec::new())
                    .expect("a prompt")
                    .retaining(PromptRetention::Withdraw),
                CancellationToken::new(),
            )
            .await
            .expect_err("the provider is unreachable");
        assert_eq!(
            announced(&frontend),
            1,
            "the withdrawal is announced exactly once: {:?}",
            frontend.events()
        );
    }

    /// The mirror image, and the reason this is not simply `run_turn`'s behavior: a human can see
    /// the error and retype, so their prompt stays exactly where it was.
    #[tokio::test]
    async fn a_typed_prompt_survives_the_same_failure() {
        let (agent, _store) = agent_for_test(unreachable_provider(1)).await;
        let mut messages = Conversation::new();

        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("check the news".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect_err("the provider is unreachable");

        assert_eq!(messages.len(), 1, "a typed prompt is never withdrawn");
    }

    /// A thinking-only reply is answered with a nudge, which is itself a plain `User` message
    /// carrying no tool result, so from the outside it looks exactly like a turn-opening prompt.
    /// If the retry then fails, withdrawal must not take the nudge: doing so leaves the prompt (the
    /// message the feature exists to remove) while retracting one meka had just committed.
    #[tokio::test]
    async fn a_failure_after_a_thinking_only_nudge_withdraws_nothing() {
        use crate::provider::mock::MockStopReason;

        let provider = Arc::new(MockProvider::from_rounds(vec![
            // A reply with a thinking block and no text: `run_turn` appends the nudge and retries.
            vec![
                MockEvent::ThinkingDelta {
                    text: "hmm".to_string(),
                },
                MockEvent::ThinkingComplete { opaque: None },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
            vec![MockEvent::Fail {
                message: "error sending request: connection refused".to_string(),
            }],
        ]));
        let (agent, _store) = agent_for_test(provider).await;
        let mut messages = Conversation::new();

        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts(
                    "[Scheduled job 7f3a1b2c fired] check the news".to_string(),
                    Vec::new(),
                )
                .expect("a prompt")
                .retaining(PromptRetention::Withdraw),
                CancellationToken::new(),
            )
            .await
            .expect_err("the retry fails");

        // Three messages went in (prompt, thinking-only assistant, nudge) and all three stay: the
        // turn moved past its prompt, so there is no longer a lone prompt to withdraw.
        assert_eq!(
            messages.len(),
            3,
            "nothing is retracted once the turn has moved on: {:?}",
            messages.as_slice()
        );
    }

    /// Withdrawal is only for a turn that produced nothing. One that failed after a tool round has
    /// real work behind it (a command that ran, a file that was written) and erasing the prompt
    /// would orphan the record of it.
    #[tokio::test]
    async fn a_fire_that_got_as_far_as_a_tool_call_keeps_everything() {
        use crate::provider::mock::MockStopReason;

        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![
                MockEvent::ToolUseStart {
                    id: "call_1".to_string(),
                    name: "does_not_exist".to_string(),
                },
                MockEvent::ToolUseEnd {
                    input: serde_json::json!({}),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::ToolUse,
                },
            ],
            vec![MockEvent::Fail {
                message: "error sending request: connection refused".to_string(),
            }],
        ]));
        let (agent, _store) = agent_for_test(provider).await;
        let mut messages = Conversation::new();

        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts(
                    "[Scheduled job 7f3a1b2c fired] check the news".to_string(),
                    Vec::new(),
                )
                .expect("a prompt")
                .retaining(PromptRetention::Withdraw),
                CancellationToken::new(),
            )
            .await
            .expect_err("the second round fails");

        // Asserted on content, not on a message count: withdrawal drops exactly the last message,
        // and here that is the tool result, so a count-based check would still pass while the
        // record of what the tool returned had been erased out from under its `tool_use`.
        let blocks: Vec<_> = messages
            .as_slice()
            .iter()
            .flat_map(|message| message.content.iter())
            .collect();
        assert!(
            blocks.iter().any(|block| matches!(
                block,
                crate::conversation::ContentBlock::ToolUse { id, .. } if id == "call_1"
            )),
            "the call the model made is still on record: {:?}",
            messages.as_slice()
        );
        assert!(
            blocks.iter().any(|block| matches!(
                block,
                crate::conversation::ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "call_1"
            )),
            "and so is what it returned, so the pair is not orphaned: {:?}",
            messages.as_slice()
        );
    }

    /// [`agent_for_test`] with a frontend that records what the turn emitted.
    async fn agent_recording_for_test(
        provider: Arc<dyn Provider>,
    ) -> (Agent, Arc<crate::frontend::testing::RecordingFrontend>) {
        let frontend = Arc::new(crate::frontend::testing::RecordingFrontend::new());
        let (mut agent, _store) =
            agent_with_registry_for_test(provider, crate::tools::ToolRegistry::new()).await;
        agent.cells.frontend = frontend.clone();
        (agent, frontend)
    }

    /// A round that only reasoned is nudged for an answer, and on the blocking path the reasoning
    /// the streaming path shows as it arrives has to be shown before that nudge's `continue`.
    #[tokio::test]
    async fn a_thinking_only_round_shows_its_thinking_without_streaming() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![
                MockEvent::ThinkingDelta {
                    text: "weighing the options".to_string(),
                },
                MockEvent::ThinkingComplete { opaque: None },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
            vec![MockEvent::Text {
                text: "the answer".to_string(),
            }],
        ]));
        let (mut agent, frontend) =
            agent_recording_for_test(Arc::clone(&provider) as Arc<dyn Provider>).await;
        agent.options.streaming = false;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("hello".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");

        let events = frontend.events();
        assert!(
            events.iter().any(|event| matches!(
                event,
                FrontendEvent::ThinkingBlock { content } if content == "weighing the options"
            )),
            "the thinking-only round's reasoning reaches the frontend: {events:?}"
        );
        assert!(
            events.iter().any(|event| matches!(
                event,
                FrontendEvent::AssistantTextDelta(text) if text == "the answer"
            )),
            "and the nudged answer follows: {events:?}"
        );
    }

    /// `--no-stream` must still show the answer.
    ///
    /// Every frontend renders assistant text from `AssistantTextDelta`, and the blocking path
    /// produces the whole message at once with no event channel to put one on. Nothing else carries
    /// the text: `TurnFinished` is a signal, and the message goes straight into the conversation
    /// log. So a turn that works perfectly prints nothing at all.
    #[tokio::test]
    async fn a_turn_that_does_not_stream_still_shows_what_the_model_said() {
        use crate::provider::mock::{MockEvent, MockProvider};

        let provider = Arc::new(MockProvider::from_rounds(vec![vec![MockEvent::Text {
            text: "the answer".to_string(),
        }]]));
        let (mut agent, frontend) =
            agent_recording_for_test(Arc::clone(&provider) as Arc<dyn Provider>).await;
        agent.options.streaming = false;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("hello".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");

        let events = frontend.events();
        let shown: String = events
            .iter()
            .filter_map(|event| match event {
                FrontendEvent::AssistantTextDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            shown, "the answer",
            "nothing reached the frontend: {events:?}"
        );
    }

    /// The reactive check fires *above* the threshold, not at it, and never on one message.
    ///
    /// [`Agent::auto_compact_threshold`] pins the number; this pins the comparisons that read it,
    /// which the tests that force compaction never approach. `>=` is the interesting one: it would
    /// compact a session sitting exactly on 80%, and since a compaction resets occupancy well below
    /// the line it would not loop, just fire one turn early, forever, invisibly.
    #[tokio::test]
    async fn the_reactive_compaction_fires_above_the_threshold_and_not_at_it() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let round = || {
            vec![
                MockEvent::Text {
                    text: "ok".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ]
        };
        // Turn one's stream, then the summarizer turn two triggers, then turn two's own stream.
        let provider = Arc::new(MockProvider::from_rounds(vec![round(), round(), round()]));
        let handle: Arc<dyn Provider> = Arc::clone(&provider) as Arc<dyn Provider>;
        let (mut agent, _manager) = agent_that_compacts_for_test(handle).await;

        let occupancy = Arc::new(std::sync::atomic::AtomicU64::new(0));
        agent.set_context_tokens_for_test(Arc::clone(&occupancy));
        let threshold = agent
            .auto_compact_threshold()
            .expect("the compacting harness enables auto-compaction");
        assert_eq!(threshold, 160_000, "80% of the harness's 200k window");

        // Long enough that the split has a head and a tail to work with.
        let mut messages = Conversation::new();
        for index in 0..4 {
            messages.append(Message::user(format!("question {index}")));
            messages.append(Message::assistant_text(format!("answer {index}")));
        }

        // Exactly on the line. `>` must not fire here; `>=` would.
        occupancy.store(threshold, std::sync::atomic::Ordering::Relaxed);
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("first".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the turn runs");
        assert_eq!(
            provider.completions().len(),
            0,
            "occupancy exactly at the threshold must not compact: the check is `>`, not `>=`"
        );

        // One token over. The turn writes the counter itself, so re-arm it first.
        occupancy.store(threshold + 1, std::sync::atomic::Ordering::Relaxed);
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("second".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the turn runs");
        assert_eq!(
            provider.completions().len(),
            1,
            "one token over the threshold must compact, or the window is never respected"
        );
    }

    /// A conversation with nothing to summarize is left alone however full it is.
    ///
    /// The `messages.len() > 1` half of the same guard. Relaxed to `>= 1` it would try to compact a
    /// single message, which is the one shape the splitter cannot produce a summary from.
    #[tokio::test]
    async fn a_single_message_conversation_is_never_auto_compacted() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![vec![
            MockEvent::Text {
                text: "ok".to_string(),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::EndTurn,
            },
        ]]));
        let handle: Arc<dyn Provider> = Arc::clone(&provider) as Arc<dyn Provider>;
        let (mut agent, _manager) = agent_that_compacts_for_test(handle).await;

        let occupancy = Arc::new(std::sync::atomic::AtomicU64::new(0));
        agent.set_context_tokens_for_test(Arc::clone(&occupancy));
        // Far over the line, so only the message count can be what holds compaction back.
        occupancy.store(10_000_000, std::sync::atomic::Ordering::Relaxed);

        let mut messages = Conversation::new();
        messages.append(Message::user("the only thing said so far".to_string()));

        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("and now this".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the turn runs");
        assert_eq!(
            provider.completions().len(),
            0,
            "a one-message conversation has no summary to make, whatever the occupancy says"
        );
    }

    /// The wiring, not the method: nothing else fails when the call in `run_turn` is deleted.
    /// What it guards is the failure the `[Memory]` section exists to prevent in its sharpest
    /// form: a store that cannot be read for one turn looks like a store that is empty, and the
    /// diff would announce every memory as deleted, by name, then re-announce them all as written
    /// on the next turn that succeeds.
    ///
    /// The store is broken by dropping the table under a live connection, which is a real error
    /// through the real path rather than a stubbed one.
    #[tokio::test]
    async fn a_turn_whose_store_breaks_does_not_announce_every_memory_as_deleted() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = Store::open(Some(&temp.path().join("meka.db")), &Default::default())
            .await
            .expect("open");
        let memories = store.memory_store(true);
        memories
            .write(crate::store::memory::WriteRequest {
                name: "deploy-policy".to_string(),
                description: Some("Never deploy on Fridays".to_string()),
                tags: None,
                body: None,
                priority: Some(3),
            })
            .await
            .expect("write");

        // The index only renders when something can open it, so the registry has to carry a tool by
        // that name. A fixture rather than the real one, which lives in a private module.
        let registry = crate::tools::ToolRegistry::new();
        registry
            .register(Arc::new(MemoryReadFixture))
            .expect("register memory_read");
        let provider = Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
            text_round("first"),
            text_round("second"),
        ]));
        let (mut agent, _unused) =
            agent_with_registry_for_test(provider as Arc<dyn Provider>, registry).await;
        agent.store = store.clone();
        agent.memories = memories.clone();
        // The harness builds sub-agent-shaped agents, and a `system_prompt_override` skips the
        // per-turn world state entirely, which is the block under test.
        agent.options.system_prompt_override = None;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("first".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("first turn");
        assert!(
            messages
                .as_slice()
                .iter()
                .any(|message| message.wire_text().contains("deploy-policy")),
            "the premise: the first turn told the model about the memory"
        );

        // Broken under the agent, between turns, through a second connection to the same file:
        // a real failure on the real read path rather than a stub that returns an error.
        rusqlite::Connection::open(temp.path().join("meka.db"))
            .expect("second connection")
            .execute_batch("DROP TABLE memories;")
            .expect("drop the table");
        assert!(
            memories.index().await.is_err(),
            "the premise: the store now fails to read"
        );

        let before = messages.len();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("second".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("second turn");

        let second: String = messages.as_slice()[before..]
            .iter()
            .map(|message| message.wire_text())
            .collect();
        assert!(
            !second.contains("Memories deleted"),
            "a store that cannot be read is not a store that is empty: {second}"
        );
        assert!(
            !second.contains("deploy-policy"),
            "and it says nothing about memory at all rather than restating a guess: {second}"
        );
    }

    /// Stands in for `memory_read` so the catalog reports the index as live. Only the name
    /// matters: `prompt::memory_index_is_live` asks whether anything can open a memory, not what.
    struct MemoryReadFixture;

    #[async_trait::async_trait]
    impl crate::tools::Tool for MemoryReadFixture {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "memory_read".to_string(),
                description: "Load one memory in full.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"name": {"type": "string", "description": "Memory name"}},
                    "required": ["name"]
                }),
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
        ) -> Result<crate::tools::ToolOutput> {
            Ok(crate::tools::ToolOutput::text(String::new(), false))
        }
    }

    /// A streamed request reaches the provider still knowing which prompt it serves.
    ///
    /// `run_streaming_attempt` hands `provider.stream(...)` to `tokio::spawn`, and a task-local
    /// does not cross a spawn; the only visible effect of losing the attribution is on the wire,
    /// where `claude-subscription`'s billing header loses `cc_prompt_id` and `cc_prev_req`.
    /// Recording the attribution the mock provider saw is the only way to see it from a test.
    #[tokio::test]
    async fn a_streamed_request_knows_which_prompt_it_serves() {
        use crate::provider::mock::{MockEvent, MockProvider};

        let provider = Arc::new(MockProvider::from_rounds(vec![vec![MockEvent::Text {
            text: "ok".to_string(),
        }]]));
        let provider_handle: Arc<dyn Provider> = Arc::clone(&provider) as Arc<dyn Provider>;
        let (agent, _store) = agent_for_test(provider_handle).await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("hello".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the turn runs");

        let requests = provider.streams();
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0].prompt_id.is_some(),
            "the spawned request lost the prompt it was serving"
        );
    }

    /// An overflow the agent cannot compact away has to surface once, not loop.
    ///
    /// This is the guard, not the recovery: `agent_for_test` sets `auto_compact: false` and
    /// `context_window: 0`, so the match arm never fires here. The recovery itself is covered by
    /// `an_overflow_it_can_compact_away_is_compacted_and_retried_once`. What this proves is that
    /// the overflow keeps its own error type and is attempted exactly once; the recorded requests
    /// are the only way to see the second part.
    #[tokio::test]
    async fn an_overflow_it_cannot_compact_away_surfaces_instead_of_looping() {
        use crate::provider::mock::{MockEvent, MockProvider};

        let provider = Arc::new(MockProvider::from_rounds(vec![vec![
            MockEvent::FailContextOverflow {
                message: "prompt is too long: 250000 tokens > 200000 maximum".to_string(),
            },
        ]]));
        let provider_handle: Arc<dyn Provider> = Arc::clone(&provider) as Arc<dyn Provider>;
        let (agent, _store) = agent_for_test(provider_handle).await;

        let mut messages = Conversation::new();
        let error = agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("summarize the log".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect_err("an overflow nothing can shrink must reach the caller");
        assert!(
            matches!(error, MekaError::ContextOverflow(_)),
            "the overflow must keep its own type, not become a generic provider error: {error:?}"
        );

        // One attempt, not a retry storm. Recording the requests is the only way to see this: the
        // returned error is identical whether the loop ran once or a thousand times.
        let requests = provider.streams();
        assert_eq!(
            requests.len(),
            1,
            "a compaction that cannot help must not be retried"
        );

        // And the one attempt carried the turn meka meant to send, which is what distinguishes
        // "the provider refused a real request" from "meka sent something malformed and the
        // overflow was incidental".
        let attempt = &requests[0];
        assert!(
            attempt
                .messages
                .iter()
                .any(|message| message.text_content().contains("summarize the log")),
            "the prompt must reach the provider: {:?}",
            attempt.messages
        );
        assert!(
            !attempt.system_prompt.is_empty(),
            "a turn always carries a system prompt"
        );
        assert!(
            attempt.tools.is_empty(),
            "this harness registers no tools, so none should be advertised"
        );
    }

    /// A notice is not model output, so it must not disable the turn's retry.
    ///
    /// `content_started` exists to stop a retry double-emitting what the user already saw. The
    /// Claude providers queue the image-redaction advisory *before* the request is sent, so a
    /// notice that set the flag would disable retry from the first event of every image-bearing
    /// turn: the next dropped connection would fail outright, having produced nothing, and the user
    /// would pay to re-send the images by hand.
    #[tokio::test]
    async fn a_notice_before_a_dropped_stream_does_not_disable_the_retry() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![
            // An advisory, then the connection drops with nothing user-visible emitted.
            vec![
                MockEvent::Notice {
                    message: "an image was too large and was downscaled".to_string(),
                },
                MockEvent::FailStream {
                    message: "connection reset".to_string(),
                },
            ],
            // The retry, which must happen.
            vec![
                MockEvent::Text {
                    text: "answered on the retry".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let provider_handle: Arc<dyn Provider> = Arc::clone(&provider) as Arc<dyn Provider>;
        let (agent, _store) = agent_for_test(provider_handle).await;

        let mut messages = Conversation::new();
        let outcome = agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("go".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the turn must retry past a notice-then-drop, not fail");

        assert!(
            matches!(outcome, TurnOutcome::EndTurn),
            "the retry must carry the turn to a normal end: {outcome:?}",
        );
        assert!(
            messages
                .iter()
                .any(|message| message.text_content().contains("answered on the retry")),
            "and the retry's answer is what lands in the conversation",
        );
    }

    /// A retried attempt rebuilds the body and reports the same redaction again; the session
    /// counts it once and the user reads it once.
    #[tokio::test]
    async fn a_retried_attempt_reporting_the_same_redaction_is_counted_and_shown_once() {
        use crate::{
            conversation::RedactedImage,
            provider::mock::{MockEvent, MockProvider, MockStopReason},
        };

        let redaction = || MockEvent::Redaction {
            images: 1,
            bytes: 4,
            positions: vec![RedactedImage {
                from_end: 1,
                block: 1,
                item: None,
            }],
        };
        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![redaction(), MockEvent::FailRetryable {
                message: "overloaded".to_string(),
                retry_after_secs: Some(0),
            }],
            vec![
                redaction(),
                MockEvent::Text {
                    text: "done".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let (agent, frontend) =
            agent_recording_for_test(Arc::clone(&provider) as Arc<dyn Provider>).await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("go".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the retry carries the turn");

        let snapshot = agent.session_stats_snapshot();
        assert_eq!(snapshot.redactions, 1);
        assert_eq!(snapshot.redacted_images, 1);
        assert_eq!(snapshot.redacted_bytes, 4);
        let shown = frontend
            .events()
            .iter()
            .filter(|event| {
                matches!(event, FrontendEvent::Notice(notice) if notice.redaction.is_some())
            })
            .count();
        assert_eq!(shown, 1, "the same redaction is announced once");
    }

    /// A redaction is counted against the session the request belonged to, from the notice the
    /// provider sends, because the provider is cached per profile and serves every session on it.
    #[tokio::test]
    async fn a_redaction_notice_is_counted_on_the_sessions_stats() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![vec![
            MockEvent::Redaction {
                images: 2,
                bytes: 4_000_000,
                positions: Vec::new(),
            },
            MockEvent::Text {
                text: "done".to_string(),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::EndTurn,
            },
        ]]));
        let provider_handle: Arc<dyn Provider> = Arc::clone(&provider) as Arc<dyn Provider>;
        let (agent, _store) = agent_for_test(provider_handle).await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("go".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the turn completes");

        let snapshot = agent.session_stats_snapshot();
        assert_eq!(snapshot.redactions, 1);
        assert_eq!(snapshot.redacted_images, 2);
        assert_eq!(snapshot.redacted_bytes, 4_000_000);
    }

    /// What a turn stores: one user message of two blocks, the context meka injected and the words
    /// as typed, in that order, so a reader of the row can take the words alone.
    #[tokio::test]
    async fn a_turn_stores_its_context_and_its_words_as_two_blocks() {
        use crate::{conversation::Event, provider::mock::MockProvider};

        let provider = Arc::new(MockProvider::from_rounds(vec![text_round("ok")]));
        let (agent, store) = agent_for_test(Arc::clone(&provider) as Arc<dyn Provider>).await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("session");
        agent.cells().session_id.set(session_id);
        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("hello there".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the turn completes");

        let events = store.load_events(session_id).await.expect("load");
        let Some(Event::Append(stored)) = events.first() else {
            panic!("the first event is the user turn: {events:?}");
        };
        assert_eq!(stored.content.len(), 2, "{stored:?}");
        assert!(
            matches!(
                &stored.content[0],
                ContentBlock::TurnContext { text } if text.contains("[Permission context]")
            ),
            "{stored:?}"
        );
        assert_eq!(stored.content[1], ContentBlock::Text {
            text: "hello there".to_string()
        });
        assert_eq!(stored.text_content(), "hello there");
    }

    /// The summarizer's copy of a turn truncates the context block as it does a long prompt: the
    /// block is the larger of the two on most turns, and the summary needs no more of it.
    #[test]
    fn the_summarizer_truncates_the_context_block_too() {
        let mut content = vec![
            ContentBlock::TurnContext {
                text: "x".repeat(5_000),
            },
            ContentBlock::Text {
                text: "short".to_string(),
            },
        ];
        super::turn::strip_images_and_truncate(&mut content);
        let ContentBlock::TurnContext { text } = &content[0] else {
            panic!("the block keeps its type: {content:?}");
        };
        assert!(text.len() < 2_000, "{}", text.len());
        assert!(text.contains("truncated for compaction"));
        assert_eq!(content[1], ContentBlock::Text {
            text: "short".to_string()
        });
    }

    /// A redaction the budget reports is recorded on the conversation, once: the image the
    /// provider redacted from the body is the placeholder in the view and on disk from then on, so
    /// the next request carries the placeholder rather than redacting the image all over again.
    #[tokio::test]
    async fn a_redaction_is_recorded_once_and_the_next_request_sends_the_placeholder() {
        use crate::{
            conversation::{Event, RedactedImage},
            provider::mock::{MockEvent, MockProvider, MockStopReason},
        };

        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![
                MockEvent::Redaction {
                    images: 1,
                    bytes: 4,
                    // The request the mock answers is [user(image), assistant, user(prompt)]: the
                    // image sits three from the end, second block of its message.
                    positions: vec![RedactedImage {
                        from_end: 3,
                        block: 1,
                        item: None,
                    }],
                },
                // A tool call the registry does not have, so the loop answers it with an error
                // and sends a second request inside the same turn.
                MockEvent::ToolUseStart {
                    id: "call-1".to_string(),
                    name: "nonexistent".to_string(),
                },
                MockEvent::ToolUseEnd {
                    input: serde_json::json!({}),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::ToolUse,
                },
            ],
            text_round("noted"),
            text_round("again"),
        ]));
        let (mut agent, store) = agent_for_test(Arc::clone(&provider) as Arc<dyn Provider>).await;
        // Blocking, so the mock records the messages behind each request in `completions`.
        agent.options.streaming = false;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("session");
        agent.cells().session_id.set(session_id);
        for event in [
            Event::Append(Message::user_with_images("look", vec![image_source()])),
            Event::Append(Message::assistant_text("seen")),
        ] {
            store
                .save_event(session_id, &event)
                .await
                .expect("seed the conversation");
        }
        let mut events = store.load_events(session_id).await.expect("load");
        store.inline_blobs(&mut events).await.expect("inline");
        let mut messages = Conversation::from_events(events);

        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("go".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the turn completes");

        let placeholder = |message: &Message| {
            matches!(
                &message.content[1],
                ContentBlock::Text { text } if text == crate::conversation::IMAGE_REDACTION_PLACEHOLDER
            )
        };
        assert!(
            placeholder(&messages.as_slice()[0]),
            "the view carries the placeholder: {:?}",
            messages.as_slice()[0].content
        );
        let stored = store.load_events(session_id).await.expect("load again");
        assert!(
            stored
                .iter()
                .any(|event| matches!(event, Event::Redact { images } if images.len() == 1)),
            "the redaction is on disk as its own event: {stored:?}"
        );
        assert!(
            placeholder(&Conversation::from_events(stored).as_slice()[0]),
            "and a resume replays it into the same placeholder"
        );

        // Every later request is built from the recorded view, so nothing is left to redact: the
        // tool loop's second round inside the same turn, and the next turn.
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("more".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the second turn completes");
        let sent = provider.completions();
        assert_eq!(
            sent.len(),
            3,
            "two rounds in the first turn, one in the second"
        );
        assert!(
            placeholder(&sent[1][0]),
            "the same turn's next round carries the placeholder, not the image: {:?}",
            sent[1][0].content
        );
        assert!(
            placeholder(&sent[2][0]),
            "and so does the next turn: {:?}",
            sent[2][0].content
        );
    }

    /// The same count on the blocking path, where the advisory rides on the return rather than the
    /// stream: sub-agents and auto-compaction run this way.
    #[tokio::test]
    async fn a_redaction_reported_without_a_stream_is_counted_too() {
        use crate::provider::mock::{MockEvent, MockProvider};

        let provider = Arc::new(MockProvider::from_rounds(vec![vec![
            MockEvent::Redaction {
                images: 1,
                bytes: 2_000_000,
                positions: Vec::new(),
            },
            MockEvent::Text {
                text: "done".to_string(),
            },
        ]]));
        let (mut agent, _frontend) =
            agent_recording_for_test(Arc::clone(&provider) as Arc<dyn Provider>).await;
        agent.options.streaming = false;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("go".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the turn completes");

        let snapshot = agent.session_stats_snapshot();
        assert_eq!(snapshot.redactions, 1);
        assert_eq!(snapshot.redacted_images, 1);
        assert_eq!(snapshot.redacted_bytes, 2_000_000);
    }

    /// The emergency arm, actually reached: an overflow the agent can compact away is compacted
    /// and the turn retried once. Its sibling above exercises the case where the guard
    /// short-circuits, since `agent_for_test`'s `auto_compact: false` and `context_window: 0`
    /// keep `recover_from_context_overflow` from being called there.
    #[tokio::test]
    async fn an_overflow_it_can_compact_away_is_compacted_and_retried_once() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![
            // The turn's first request: too large.
            vec![MockEvent::FailContextOverflow {
                message: "prompt is too long: 250000 tokens > 200000 maximum".to_string(),
            }],
            // The summarizer, which `CompactOrigin::Emergency` always runs through `complete`.
            vec![
                MockEvent::Text {
                    text: "the log said everything was fine".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
            // The retry, against the compacted conversation.
            vec![
                MockEvent::Text {
                    text: "done".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let provider_handle: Arc<dyn Provider> = Arc::clone(&provider) as Arc<dyn Provider>;
        let (agent, _store) = agent_that_compacts_for_test(provider_handle).await;

        // Long enough to have something to compact: the split keeps everything as head below five
        // messages, so a shorter conversation has no summary to make and correctly surfaces.
        let mut messages = Conversation::new();
        for round in 0..4 {
            messages.append(Message::user(format!("question {round}")));
            messages.append(Message::assistant_text(format!("answer {round}")));
        }

        let outcome = agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("summarize the log".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the compacted retry must succeed");
        assert!(
            matches!(outcome, TurnOutcome::EndTurn),
            "the retry must end the turn cleanly: {outcome:?}",
        );

        assert_eq!(
            provider.completions().len(),
            1,
            "the emergency summarizer must have run",
        );
        assert_eq!(
            provider.streams().len(),
            2,
            "the turn is attempted once, compacted, and attempted once more",
        );
        // The retry is the point: it must carry less than the request that overflowed.
        let requests = provider.streams();
        assert!(
            requests[1].messages.len() < requests[0].messages.len(),
            "the retry sent {} messages against the original's {}; compaction achieved nothing",
            requests[1].messages.len(),
            requests[0].messages.len(),
        );
    }

    /// A retryable failure on the first attempt costs a retry, not the turn.
    ///
    /// The mock hands back a [`MekaError::RetryableProvider`] directly, so nothing here reaches
    /// `provider_transport_error`; the classification is pinned by
    /// `error::tests::a_provider_call_that_never_answered_is_retryable` and the Anthropic messages
    /// test that reports an unreachable endpoint as retryable. This pins the loop: a failure typed
    /// this way is retried and leaves no trace in the conversation for a later turn to resend.
    #[tokio::test]
    async fn run_turn_retries_a_call_that_never_answered() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![MockEvent::FailRetryable {
                message: "HTTP request failed (body 2.0 MiB): connection reset".to_string(),
                retry_after_secs: None,
            }],
            vec![
                MockEvent::Text {
                    text: "Here is the chart.".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let (agent, _store) = agent_for_test(provider.clone()).await;

        let mut messages = Conversation::new();
        let outcome = agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("read the chart".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the turn survives a first attempt that got no response");
        assert_eq!(outcome, TurnOutcome::EndTurn);
        assert_eq!(
            provider.streams().len(),
            2,
            "one attempt that failed and one that did not"
        );
        assert!(
            messages
                .iter()
                .flat_map(|message| message.content.iter())
                .any(
                    |block| matches!(block, ContentBlock::Text { text } if text.contains("chart"))
                ),
            "the answer from the surviving attempt is what lands in the conversation"
        );
        assert!(
            !messages
                .iter()
                .any(|message| message.text_content().contains("connection reset")),
            "and the failure that was retried away leaves no trace to resend"
        );
    }

    /// The whole point of the feature: a rejection of content meka just appended must not end the
    /// turn, and the repair must be persisted so a resume doesn't walk back into it.
    #[tokio::test]
    async fn run_turn_degrades_rejected_content_and_continues() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![MockEvent::FailInvalidRequest {
                message: REJECTION.to_string(),
            }],
            vec![
                MockEvent::Text {
                    text: "I could not see that image.".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let (agent, store) = agent_for_test(provider).await;

        let mut messages = Conversation::new();
        let outcome = agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("look at this".to_string(), vec![
                    image_source(),
                ])
                .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the turn recovers instead of dying");
        assert_eq!(outcome, TurnOutcome::EndTurn);

        // The image is gone from the live conversation, replaced by an explanation.
        let user = &messages.as_slice()[0];
        assert!(
            user.content
                .iter()
                .all(|block| !matches!(block, ContentBlock::Image { .. })),
            "the refused image must not survive in the conversation"
        );
        assert!(
            user.text_content().contains("image/jpeg"),
            "carries the reason"
        );

        // And it is gone on disk too, or the next resume would re-poison the session.
        let session_id = agent.session_id().expect("session created");
        let reloaded =
            Conversation::from_events(store.load_events(session_id).await.expect("load events"));
        assert!(
            reloaded
                .iter()
                .flat_map(|message| message.content.iter())
                .all(|block| !matches!(block, ContentBlock::Image { .. })),
            "the repair must be persisted, not just applied in memory"
        );
    }

    /// A stop that lands before the provider has answered has judged nothing. On the streaming
    /// path the send answers a stop with `Interrupted` ahead of any response, and treating that
    /// like a stop mid-stream would book the request as accepted and persist a repair the
    /// provider never saw: the refused image would leave the store for good on the strength of a
    /// keystroke.
    #[tokio::test]
    async fn a_stop_before_the_provider_answers_vindicates_nothing() {
        use crate::{
            conversation::Event,
            provider::mock::{MockEvent, MockProvider, MockStopReason},
        };
        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![MockEvent::FailInvalidRequest {
                message: REJECTION.to_string(),
            }],
            vec![
                MockEvent::Sleep { ms: 3000 },
                MockEvent::Text {
                    text: "never delivered".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let (mut agent, store) = agent_for_test(Arc::clone(&provider) as Arc<dyn Provider>).await;
        agent.options.streaming = true;
        let mut messages = Conversation::new();
        let cancellation = CancellationToken::new();
        let stop = tokio::spawn({
            let cancellation = cancellation.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                cancellation.cancel();
            }
        });
        let outcome = agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("look at this".to_string(), vec![
                    image_source(),
                ])
                .expect("a prompt"),
                cancellation,
            )
            .await;
        stop.await.expect("the stop landed");
        assert!(
            matches!(outcome, Err(MekaError::Interrupted)),
            "the retry was stopped: {outcome:?}"
        );
        // The repair was applied for the retry and undone when the retry was stopped unjudged:
        // the image is back in the live conversation and never left the store.
        let user = &messages.as_slice()[0];
        assert!(
            user.content
                .iter()
                .any(|block| matches!(block, ContentBlock::Image { .. })),
            "an unjudged repair is undone in memory: {:?}",
            user.content
        );
        let session_id = agent.session_id().expect("session created");
        let events = store.load_events(session_id).await.expect("load events");
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Event::Repair { .. })),
            "a repair the provider never saw must not be persisted: {events:?}"
        );
    }

    /// One refused round, as the mock's script sees it: a `500` shaped like the incident.
    ///
    /// `retry_after_secs: Some(0)` is honored verbatim by `backoff_delay`, so a test using this
    /// spends no wall clock proving something about classification.
    fn refused_with_a_five_hundred() -> Vec<crate::provider::mock::MockEvent> {
        vec![crate::provider::mock::MockEvent::FailRetryable {
            message: "API returned status 500 Internal Server Error: {\"error\":\"An exception \
                      occurred while loading IMAGE data at index 21\"}"
                .to_string(),
            retry_after_secs: Some(0),
        }]
    }

    /// Every request one `run_streaming` call makes before it gives up.
    ///
    /// Derived, not written out, so raising [`crate::provider::retry::MAX_PROVIDER_RETRIES`]
    /// lengthens the scripts below rather than silently making them assert the ordinary retry path
    /// instead.
    fn one_spent_retry_sequence() -> Vec<Vec<crate::provider::mock::MockEvent>> {
        (0..=crate::provider::retry::MAX_PROVIDER_RETRIES)
            .map(|_| refused_with_a_five_hundred())
            .collect()
    }

    /// The incident the [`MekaError::RetryableProvider`] arm exists for.
    ///
    /// A gateway that answers `500` because its own image decoder threw is indistinguishable, from
    /// here, from one that is overloaded. So the retry is honored first and in full, and then the
    /// outage reprieve re-sends the body unchanged one more time; this is what happens once *that*
    /// has been refused too. The alternative to degrading is not "wait longer", it is failing the
    /// turn with that body committed, which fails every later turn in the session the same way.
    ///
    /// Paused time, because the reprieve is eight seconds of deliberate waiting and this test is
    /// about what happens after it, not about how long it is.
    #[tokio::test(start_paused = true)]
    async fn a_five_hundred_outliving_its_retries_degrades_rather_than_stranding_the_session() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        // Two sequences: the original request, then the reprieve's re-send of the same body. Only
        // once both have failed does the turn conclude the content is the problem.
        let mut rounds = one_spent_retry_sequence();
        rounds.extend(one_spent_retry_sequence());
        rounds.push(vec![
            MockEvent::Text {
                text: "I could not see that image.".to_string(),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::EndTurn,
            },
        ]);
        let provider = Arc::new(MockProvider::from_rounds(rounds));
        let (agent, store) = agent_for_test(provider).await;

        let mut messages = Conversation::new();
        let outcome = agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("look at this".to_string(), vec![
                    image_source(),
                ])
                .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("a spent retry budget degrades rather than ending the turn");
        assert_eq!(outcome, TurnOutcome::EndTurn);

        let user = &messages.as_slice()[0];
        assert!(
            user.content
                .iter()
                .all(|block| !matches!(block, ContentBlock::Image { .. })),
            "the content the provider kept refusing must not survive the turn"
        );
        assert!(
            user.text_content().contains("500"),
            "the model is told what the provider said: {}",
            user.text_content()
        );

        let session_id = agent.session_id().expect("session created");
        let reloaded =
            Conversation::from_events(store.load_events(session_id).await.expect("load events"));
        assert!(
            reloaded
                .iter()
                .flat_map(|message| message.content.iter())
                .all(|block| !matches!(block, ContentBlock::Image { .. })),
            "and on disk, or the next resume walks straight back into it"
        );
    }

    /// The case the reprieve exists for: an outage that ends, and content that survives it.
    ///
    /// A `529` burst lasting a few seconds outlives the whole retry sequence, which is two attempts
    /// across three seconds of backoff. Without the reprieve the turn would read that as a verdict
    /// on its own body, degrade, and have the degraded retry succeed because the burst had ended,
    /// so `persist_vindicated_repair` would write the loss to the store as proven-good.
    ///
    /// The script says exactly that: one spent sequence, then a success. If the reprieve fires, the
    /// success answers the *unmodified* request and the image is still there.
    #[tokio::test(start_paused = true)]
    async fn an_outage_that_ends_costs_a_wait_rather_than_the_turn_s_attachment() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let mut rounds = one_spent_retry_sequence();
        rounds.push(vec![
            MockEvent::Text {
                text: "I can see the image.".to_string(),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::EndTurn,
            },
        ]);
        let provider = Arc::new(MockProvider::from_rounds(rounds));
        let (agent, store) = agent_for_test(provider).await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("look at this".to_string(), vec![
                    image_source(),
                ])
                .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the re-sent request succeeded");

        assert!(
            messages.as_slice()[0]
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::Image { .. })),
            "the attachment must survive an outage that merely ended: {:?}",
            messages.as_slice()
        );
        let session_id = agent.session_id().expect("session created");
        let reloaded =
            Conversation::from_events(store.load_events(session_id).await.expect("load events"));
        assert!(
            reloaded
                .iter()
                .flat_map(|message| message.content.iter())
                .any(|block| matches!(block, ContentBlock::Image { .. })),
            "and no `Event::Repair` may have been written for a loss that never happened"
        );
    }

    /// End to end: a reprieve that *worked* is available again later in the same turn.
    ///
    /// This is the wiring: the unit test pins what `note_request_accepted` does, and this pins
    /// when it is called. Called from `persist_vindicated_repair` it would never run on the one
    /// path that matters, since a successful reprieve applies no repair and there is nothing to
    /// vindicate.
    ///
    /// Counted rather than inspected, because `TurnRecovery` is local to `run_turn`. Each reprieve
    /// costs one extra *sequence* of unchanged re-sends, so the request tally separates the two
    /// behaviors: eleven requests if the second reprieve fires, eight if it does not.
    #[tokio::test(start_paused = true)]
    async fn a_reprieve_that_worked_is_available_again_later_in_the_turn() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        // A tool round first, so the outages that follow have a tool exchange to degrade. Without
        // one, `repair_rejected_content` finds no tier and fails the turn before the reprieve is
        // ever consulted.
        let tool_round = |id: &str| {
            vec![
                MockEvent::ToolUseStart {
                    id: id.to_string(),
                    name: "no_such_tool".to_string(),
                },
                MockEvent::ToolUseEnd {
                    input: serde_json::json!({"path": "notes.md"}),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::ToolUse,
                },
            ]
        };

        let mut rounds = vec![tool_round("call_1")];
        // First outage: a spent sequence, then the reprieve's re-send, which succeeds and calls a
        // tool again so the turn continues.
        rounds.extend(one_spent_retry_sequence());
        rounds.push(tool_round("call_2"));
        // Second, unrelated outage. Its own spent sequence, then the *second* reprieve's re-send,
        // failing too, so a tier finally applies.
        rounds.extend(one_spent_retry_sequence());
        rounds.extend(one_spent_retry_sequence());
        rounds.push(vec![
            MockEvent::Text {
                text: "done".to_string(),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::EndTurn,
            },
        ]);

        let mock = Arc::new(MockProvider::from_rounds(rounds));
        let (agent, _store) = agent_for_test(Arc::clone(&mock) as Arc<dyn Provider>).await;
        agent
            .run_turn(
                &mut Conversation::new(),
                crate::agent::TurnInput::from_parts("read my notes".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the turn recovers");

        assert_eq!(
            mock.streams().len(),
            12,
            "the second outage has to buy its own unchanged re-send; nine requests would mean the \
             reprieve stayed spent after the first one succeeded"
        );
    }

    /// A turn that compacted on the way in can still degrade.
    ///
    /// `suspect_floor` is captured before the prompt is appended, so it counts messages of the
    /// conversation the compaction then replaces. Left alone it lands past the end of the collapsed
    /// one, the clamp in `repair_rejected_content` reads the window as empty, both tiers find
    /// nothing, and the turn dies with the refused attachment committed, silently: no tier was
    /// spent, so even the rewind hint stays quiet.
    #[tokio::test]
    async fn a_turn_that_compacted_on_the_way_in_can_still_degrade() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![
            // The proactive compaction's summarizer.
            vec![
                MockEvent::Text {
                    text: "a summary of the work so far".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
            vec![MockEvent::FailInvalidRequest {
                message: REJECTION.to_string(),
            }],
            vec![
                MockEvent::Text {
                    text: "I could not see that image.".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let (agent, _store) = agent_that_compacts_for_test(provider as Arc<dyn Provider>).await;

        // Two constraints pull against each other, so the numbers are deliberate. The window has to
        // be small enough that the history below trips the pre-send projection (80% of it), which
        // is what puts the compaction ahead of the turn's first request. It also has to be large
        // enough that `compaction_tail_budget` (~10% of it) can hold this turn's prompt, or the
        // split keeps no tail, the prompt is summarized away with its attachment, and the test
        // passes for the wrong reason: nothing left to degrade.
        agent.set_context_window_for_test(20_000);

        let mut messages = Conversation::new();
        for index in 0..40 {
            messages.append(Message::user(format!(
                "earlier request {index}: {}",
                "x".repeat(1_000)
            )));
            messages.append(Message::assistant_text(format!(
                "earlier reply {index}: {}",
                "y".repeat(1_000)
            )));
        }
        let before = messages.len();

        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("look at this".to_string(), vec![
                    image_source(),
                ])
                .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the degrade has to be reachable after the compaction");

        assert!(
            messages.len() < before,
            "precondition: the proactive compaction ran, or this proves nothing"
        );
        assert!(
            messages
                .iter()
                .flat_map(|message| message.content.iter())
                .all(|block| !matches!(block, ContentBlock::Image { .. })),
            "the refused attachment must not survive the turn: {:?}",
            messages.as_slice()
        );
    }

    /// The other half of the bargain: a turn that degrades and is refused anyway must leave the
    /// conversation exactly as it found it, and must fail with the provider's own error rather than
    /// one invented by the repair.
    ///
    /// The second matters at the HTTP surface, where [`MekaError::InvalidRequest`] answers 4xx.
    /// Reclassifying an upstream 500 as a client error would blame the caller for a fault that was
    /// never theirs.
    #[tokio::test(start_paused = true)]
    async fn a_degrade_that_does_not_help_restores_the_content_and_keeps_the_error() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        // Three full sequences and no more: the original request, the reprieve's unchanged re-send,
        // and the one carrying `Attachments`. The window here is a prompt with an attachment and no
        // tool exchange, so `ToolExchanges` then declines without spending a round trip.
        // Over-provisioning would not fail loudly if the count were wrong:
        // `MockProvider::from_rounds` yields an empty round once the script runs out, and
        // that folds into a successful empty turn.
        let mut rounds = one_spent_retry_sequence();
        rounds.extend(one_spent_retry_sequence());
        rounds.extend(one_spent_retry_sequence());
        rounds.push(vec![MockEvent::MessageEnd {
            stop_reason: MockStopReason::EndTurn,
        }]);
        let provider = Arc::new(MockProvider::from_rounds(rounds));
        let (agent, _store) = agent_for_test(provider).await;

        let mut messages = Conversation::new();
        let error = agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("look at this".to_string(), vec![
                    image_source(),
                ])
                .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect_err("nothing the turn tried satisfied the provider");
        assert!(
            matches!(error, MekaError::RetryableProvider { .. }),
            "the turn fails with the provider's fault, not a reclassified one: {error}"
        );
        assert!(
            messages.as_slice()[0]
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::Image { .. })),
            "a guess that did not pay off costs a round trip, never the content"
        );
    }

    /// A refused tool round with nothing but text in it recovers, on the first attempt.
    ///
    /// This is the shape the whole tier list exists for: `Attachments` has nothing to remove here,
    /// and without the second tier the refused body would stay committed and every later turn in
    /// the session would re-send it. The single refusal in the script is the point: finding
    /// nothing must make a tier step aside, not spend a round trip proving it.
    ///
    /// Driven with a call to a tool that does not exist: the dispatcher answers an unknown name
    /// with an error `tool_result` rather than an `Err`, which is a real `tool_use` / `tool_result`
    /// pair without a fixture tool to register.
    #[tokio::test]
    async fn a_refused_text_only_tool_round_recovers_without_spending_a_tier_on_attachments() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![
                MockEvent::ToolUseStart {
                    id: "call_1".to_string(),
                    name: "no_such_tool".to_string(),
                },
                MockEvent::ToolUseEnd {
                    input: serde_json::json!({"path": "notes.md"}),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::ToolUse,
                },
            ],
            vec![MockEvent::FailInvalidRequest {
                message: REJECTION.to_string(),
            }],
            vec![
                MockEvent::Text {
                    text: "done".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let (agent, _store) = agent_for_test(provider).await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("read my notes".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the second tier gets the turn through");

        let blocks: Vec<&ContentBlock> = messages
            .iter()
            .flat_map(|message| message.content.iter())
            .collect();
        assert!(
            blocks
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolUse { name, .. } if name == "no_such_tool")),
            "the call stays a call, so nothing can be orphaned: {:?}",
            messages.as_slice()
        );
        let results: Vec<&str> = blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolResult { content, .. } => match content.first() {
                    Some(ToolResultContent::Text { text }) => Some(text.as_str()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert!(
            results.iter().any(|text| text.contains("notes.md")),
            "and its arguments moved into the result it reports: {results:?}"
        );
    }

    /// The rewind hint reports, so it must fire exactly when there is something to report.
    ///
    /// It is earned by having spent a tier: real content was degraded, refused anyway, and put
    /// back where every later turn re-sends it. A turn that never found a tier to spend was
    /// refused over the request rather than its contents, and sending that user to delete a turn
    /// would be advice to destroy the wrong thing.
    #[tokio::test]
    async fn the_rewind_hint_is_earned_by_a_tier_that_did_not_help() {
        use crate::provider::mock::{MockEvent, MockProvider};

        async fn hinted(attachments: Vec<ImageSource>) -> bool {
            let refused = || {
                vec![MockEvent::FailInvalidRequest {
                    message: REJECTION.to_string(),
                }]
            };
            let provider = Arc::new(MockProvider::from_rounds(vec![
                refused(),
                refused(),
                refused(),
            ]));
            let (agent, frontend) = agent_recording_for_test(provider).await;
            agent
                .run_turn(
                    &mut Conversation::new(),
                    crate::agent::TurnInput::from_parts("look at this".to_string(), attachments)
                        .expect("a prompt"),
                    CancellationToken::new(),
                )
                .await
                .expect_err("every attempt was refused");
            frontend.events().iter().any(|event| {
                matches!(event, FrontendEvent::Notice(notice) if notice.text.contains("rewind"))
            })
        }

        assert!(
            hinted(vec![image_source()]).await,
            "an attachment was degraded and restored, so the session is carrying it again"
        );
        assert!(
            !hinted(Vec::new()).await,
            "a prose-only turn offered no tier anything; the refusal was not about its contents"
        );
    }

    /// A resumed conversation is told so, once. Its own history reads as proof that a tool call
    /// happened, which is true, and as proof that the effect still holds, which a restart makes
    /// false: the read tracker is gone and any MCP server has reconnected.
    #[tokio::test]
    async fn resumed_conversation_is_told_once() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let round = || {
            vec![
                MockEvent::Text {
                    text: "ok".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ]
        };
        let provider = Arc::new(MockProvider::from_rounds(vec![round(), round(), round()]));
        let (agent, store) = agent_for_test(provider).await;

        // A session with one real turn behind it, then reloaded the way a restart would.
        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("first".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("first turn");
        assert!(
            !messages.as_slice()[0]
                .wire_text()
                .contains("[Session resumed]"),
            "a session nobody resumed must not claim to have been"
        );

        let session_id = agent.session_id().expect("session created");
        let mut resumed =
            Conversation::from_events(store.load_events(session_id).await.expect("load events"));
        let before = resumed.len();
        agent.cells().session_id.set(session_id);
        agent
            .run_turn(
                &mut resumed,
                crate::agent::TurnInput::from_parts("second".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("turn after resume");
        assert!(
            resumed.as_slice()[before]
                .wire_text()
                .contains("[Session resumed]"),
            "the first turn after a resume carries the notice"
        );

        let next = resumed.len();
        agent.cells().session_id.set(session_id);
        agent
            .run_turn(
                &mut resumed,
                crate::agent::TurnInput::from_parts("third".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("third turn");
        assert!(
            !resumed.as_slice()[next]
                .wire_text()
                .contains("[Session resumed]"),
            "and only that turn: repeating it every turn would make it scenery"
        );
    }

    /// A first turn that fails still delivered the notice, because the user message carrying it is
    /// persisted before the provider is called and so survives the failure. It must therefore not
    /// be offered a second time on the retry. The withdrawal in the error arm is for the narrower
    /// case where that save itself failed and the message is popped, which is exactly the pairing
    /// `world_state_rollback` has beside it.
    #[tokio::test]
    async fn resume_notice_is_not_repeated_after_a_failed_turn() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![
                MockEvent::Text {
                    text: "ok".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
            vec![MockEvent::FailInvalidRequest {
                message: "nope".to_string(),
            }],
            vec![
                MockEvent::Text {
                    text: "ok".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let (agent, store) = agent_for_test(provider).await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("first".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("first turn");
        let session_id = agent.session_id().expect("session created");

        let mut resumed =
            Conversation::from_events(store.load_events(session_id).await.expect("load events"));
        agent.cells().session_id.set(session_id);
        assert!(
            agent
                .run_turn(
                    &mut resumed,
                    crate::agent::TurnInput::from_parts("doomed".to_string(), Vec::new())
                        .expect("a prompt"),
                    CancellationToken::new()
                )
                .await
                .is_err(),
            "the fixture must actually fail"
        );

        assert!(
            resumed
                .iter()
                .any(|message| message.wire_text().contains("[Session resumed]")),
            "the failed turn's user message is persisted, so the notice was delivered"
        );

        let before = resumed.len();
        agent.cells().session_id.set(session_id);
        agent
            .run_turn(
                &mut resumed,
                crate::agent::TurnInput::from_parts("retry".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("retry");
        assert!(
            !resumed.as_slice()[before]
                .wire_text()
                .contains("[Session resumed]"),
            "and having been delivered, it must not be said again"
        );
    }

    /// A rejection that degrading doesn't fix must cost one round trip and nothing else.
    #[tokio::test]
    async fn run_turn_restores_content_when_the_repair_does_not_help() {
        use crate::provider::mock::{MockEvent, MockProvider};

        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![MockEvent::FailInvalidRequest {
                message: REJECTION.to_string(),
            }],
            vec![MockEvent::FailInvalidRequest {
                message: REJECTION.to_string(),
            }],
        ]));
        let (agent, _store) = agent_for_test(provider).await;

        let mut messages = Conversation::new();
        let error = agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("look at this".to_string(), vec![
                    image_source(),
                ])
                .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect_err("both attempts were refused");
        assert!(
            matches!(&error, MekaError::InvalidRequest(message) if message.contains("image/jpeg")),
            "the provider's own error surfaces, not one about the repair: {error}"
        );

        assert!(
            messages.as_slice()[0]
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::Image { .. })),
            "a repair that didn't help must leave the conversation untouched"
        );
        assert!(
            !messages
                .events()
                .iter()
                .any(|event| matches!(event, crate::conversation::Event::Repair { .. })),
            "and must leave no repair behind in the log"
        );
    }

    /// `/rewind` shortens the conversation behind a live agent, which invalidates the length the
    /// recovery measures its suspect window back from. Left stale, that length lands at or past the
    /// end of the shortened conversation, the window comes out empty, and the recovery silently
    /// never fires again for the rest of the session.
    #[tokio::test]
    async fn recovery_still_fires_after_the_conversation_is_rewound() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let text_round = |text: &str| {
            vec![
                MockEvent::Text {
                    text: text.to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ]
        };
        let provider = Arc::new(MockProvider::from_rounds(vec![
            text_round("first answer"),
            vec![MockEvent::FailInvalidRequest {
                message: REJECTION.to_string(),
            }],
            text_round("second answer, without the image"),
        ]));
        let (agent, _store) = agent_for_test(provider).await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("first".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("first turn succeeds");

        assert!(messages.rewind(1).is_some(), "the turn is rewound away");
        agent.reset_conversation_markers().await;

        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("look at this".to_string(), vec![
                    image_source(),
                ])
                .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the recovery must still fire on a rewound conversation");
        assert!(
            messages
                .iter()
                .flat_map(|message| message.content.iter())
                .all(|block| !matches!(block, ContentBlock::Image { .. })),
            "the refused image should have been degraded away"
        );
    }

    fn send_file_round(input: serde_json::Value) -> Vec<crate::provider::mock::MockEvent> {
        use crate::provider::mock::{MockEvent, MockStopReason};
        vec![
            MockEvent::ToolUseStart {
                id: "call-1".to_string(),
                name: "mcp__bridge__send_file".to_string(),
            },
            MockEvent::ToolUseEnd { input },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::ToolUse,
            },
        ]
    }

    /// The whole incident, end to end: a deferred tool is callable without `load_tool`, so a model
    /// working from the truncated `[Tool discovery]` summary takes a silently wrong default. The
    /// result has to say so, because the call itself succeeds and there is no error to read.
    #[tokio::test]
    async fn blind_deferred_call_is_told_what_it_omitted() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![
            send_file_round(serde_json::json!({"path": "/tmp/a.png"})),
            vec![
                MockEvent::Text {
                    text: "sent".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let (agent, _store) = agent_with_registry_for_test(provider, send_file_registry()).await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("send the picture".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");

        let results: String = messages
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|block| match block {
                ContentBlock::ToolResult { content, .. } => {
                    Some(ContentBlock::tool_result_text_content(content))
                }
                _ => None,
            })
            .collect();

        assert!(results.contains("Sent (message id 1)"), "{results}");
        assert!(results.contains("as_photo"), "the omitted flag: {results}");
        assert!(results.contains("load_tool"), "{results}");
    }

    /// A thinking block that carries no text still has to announce that it ended.
    ///
    /// This is the contract the live indicator rests on: it holds a line open across the reasoning
    /// phase, and under `redact-thinking` or display updates no text ever arrives to close it.
    /// Without this event the line stays open until some later event happens to occur, and a
    /// turn that errors or is interrupted emits none, so an error message would print onto the
    /// indicator's row.
    #[tokio::test]
    async fn a_silent_thinking_block_announces_that_it_ended() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![vec![
            // No `ThinkingDelta`: the block produces a signature and nothing readable, which is
            // every block under `redact-thinking` or display updates.
            MockEvent::ThinkingComplete {
                opaque: Some(crate::conversation::OpaqueReasoning::Signed {
                    signature: "sig".to_string(),
                }),
            },
            MockEvent::Text {
                text: "done".to_string(),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::EndTurn,
            },
        ]]));
        let (agent, frontend) = agent_recording_for_test(provider).await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("hello".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");

        let events = frontend.events();
        assert!(
            events
                .iter()
                .any(|event| matches!(event, FrontendEvent::ThinkingEnded)),
            "a silent block must report its end: {events:?}",
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, FrontendEvent::ThinkingBlock { .. })),
            "there is no text to render, so no block event belongs here: {events:?}",
        );
    }

    /// Reasoning the turn was handed has to reach the conversation, or the next request cannot
    /// replay it.
    ///
    /// The Responses backend carries two opaque values on a thinking block: `encrypted_content` as
    /// the signature, and the reasoning item's `rs_...` as the id. Neither is readable and neither
    /// is reconstructible, so dropping either here is invisible until the model's chain of thought
    /// quietly stops carrying across tool calls.
    #[tokio::test]
    async fn a_turn_records_the_opaque_reasoning_it_was_handed() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![vec![
            MockEvent::ThinkingDelta {
                text: "weighing it up".to_string(),
            },
            MockEvent::ThinkingComplete {
                opaque: Some(crate::conversation::OpaqueReasoning::Sealed {
                    encrypted_content: "OPAQUE".to_string(),
                    id: Some("rs_1".to_string()),
                }),
            },
            MockEvent::Text {
                text: "done".to_string(),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::EndTurn,
            },
        ]]));
        let (agent, _frontend) = agent_recording_for_test(provider).await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("hello".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");

        let recorded = messages
            .iter()
            .flat_map(|message| message.content.iter())
            .find_map(|block| match block {
                ContentBlock::Thinking { thinking, opaque } => {
                    Some((thinking.clone(), opaque.clone()))
                }
                _ => None,
            })
            .expect("the turn must record its thinking block");

        assert_eq!(
            recorded,
            (
                "weighing it up".to_string(),
                Some(crate::conversation::OpaqueReasoning::Sealed {
                    encrypted_content: "OPAQUE".to_string(),
                    id: Some("rs_1".to_string()),
                })
            )
        );
    }

    /// A stream that dies has to close out any thinking in flight.
    ///
    /// The failing turn emits no `TurnFinished`, and `ThinkingComplete` only arrives from a
    /// `content_block_stop` the stream never reached, so without this the frontend's live
    /// indicator keeps its line open and the error message prints onto that row.
    #[tokio::test]
    async fn a_failed_stream_closes_out_thinking() {
        use crate::provider::mock::{MockEvent, MockProvider};

        let provider = Arc::new(MockProvider::from_rounds(vec![vec![MockEvent::Fail {
            message: "Overloaded".to_string(),
        }]]));
        let (agent, frontend) = agent_recording_for_test(provider).await;

        let mut messages = Conversation::new();
        let outcome = agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("hello".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await;
        assert!(outcome.is_err(), "the turn is supposed to fail here");

        let events = frontend.events();
        assert!(
            events
                .iter()
                .any(|event| matches!(event, FrontendEvent::ThinkingEnded)),
            "a dying stream must release the indicator's line: {events:?}",
        );
    }

    /// The other direction: a block with readable text renders as a block, and must not also
    /// report an empty ending, since the frontend erases the indicator for one and keeps it for
    /// the other, so emitting both would erase a line and then commit nothing.
    #[tokio::test]
    async fn a_thinking_block_with_text_reports_only_the_block() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![vec![
            MockEvent::ThinkingDelta {
                text: "weighing the options".to_string(),
            },
            MockEvent::ThinkingComplete {
                opaque: Some(crate::conversation::OpaqueReasoning::Signed {
                    signature: "sig".to_string(),
                }),
            },
            MockEvent::Text {
                text: "done".to_string(),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::EndTurn,
            },
        ]]));
        let (agent, frontend) = agent_recording_for_test(provider).await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("hello".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");

        let events = frontend.events();
        assert!(
            events
                .iter()
                .any(|event| matches!(event, FrontendEvent::ThinkingBlock { .. })),
            "readable thinking must render as a block: {events:?}",
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, FrontendEvent::ThinkingEnded)),
            "the block event already closes the indicator: {events:?}",
        );
    }

    /// The window a client can draw "writing a message" over: it opens when the tool's name
    /// arrives and closes when its arguments are complete.
    ///
    /// The dispatch event alone puts the whole of that window on the wrong side of the signal,
    /// because by the time it fires the arguments (the message, for a tool that sends one) are
    /// already written.
    #[tokio::test]
    async fn a_tool_call_announces_composition_before_dispatch() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![
                MockEvent::ToolUseStart {
                    id: "tu_1".to_string(),
                    name: "read_file".to_string(),
                },
                MockEvent::ToolUseEnd {
                    input: serde_json::json!({"path": "/tmp/a.txt"}),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::ToolUse,
                },
            ],
            vec![
                MockEvent::Text {
                    text: "done".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let (agent, frontend) = agent_recording_for_test(provider).await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("read it".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");

        let events = frontend.events();
        let composing = events
            .iter()
            .position(|event| {
                matches!(event, FrontendEvent::ToolCallComposing { id, name }
                    if id == "tu_1" && name == "read_file")
            })
            .expect("the call names itself while its arguments are still streaming");
        let dispatched = events
            .iter()
            .position(
                |event| matches!(event, FrontendEvent::ToolCallStarted { id, .. } if id == "tu_1"),
            )
            .expect("the call is dispatched");
        assert!(
            composing < dispatched,
            "composition has to open before the dispatch that ends it: {events:?}",
        );
    }

    /// A tool that blocks until canceled, standing in for a long build.
    struct SlowFixture;

    #[async_trait::async_trait]
    impl crate::tools::Tool for SlowFixture {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "execute_command".to_string(),
                description: "Run a shell command.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command"}
                    },
                    "required": ["command"]
                }),
                ..Default::default()
            }
        }

        fn required_permission(&self) -> crate::permission::Permission {
            crate::permission::Permission::Read
        }

        async fn execute(
            &self,
            _input: serde_json::Value,
            context: crate::tools::ToolContext,
        ) -> Result<crate::tools::ToolOutput> {
            let cancellation = context.cancellation.clone();
            cancellation.cancelled().await;
            Err(MekaError::Interrupted)
        }
    }

    /// A tool that returns at once, for the ordinary completion path.
    struct QuickFixture;

    #[async_trait::async_trait]
    impl crate::tools::Tool for QuickFixture {
        fn definition(&self) -> ToolDefinition {
            SlowFixture.definition()
        }

        fn required_permission(&self) -> crate::permission::Permission {
            crate::permission::Permission::Read
        }

        async fn execute(
            &self,
            _input: serde_json::Value,
            _context: crate::tools::ToolContext,
        ) -> Result<crate::tools::ToolOutput> {
            Ok(crate::tools::ToolOutput::text(
                "42 passed".to_string(),
                false,
            ))
        }
    }

    async fn background_agent(
        provider: Arc<dyn Provider>,
        tool: Arc<dyn crate::tools::Tool>,
    ) -> (Agent, Store) {
        let registry = crate::tools::ToolRegistry::new();
        registry.enable_background();
        registry.register(tool).expect("register fixture");
        let (mut agent, store) = agent_with_registry_for_test(provider, registry).await;
        agent.enable_background_for_test(2);
        (agent, store)
    }

    /// [`background_agent`] against a store on disk, for a test that has to break it from outside.
    async fn background_agent_at(
        provider: Arc<dyn Provider>,
        tool: Arc<dyn crate::tools::Tool>,
        path: &std::path::Path,
    ) -> (Agent, Store) {
        let registry = crate::tools::ToolRegistry::new();
        registry.enable_background();
        registry.register(tool).expect("register fixture");
        let store = Store::open(Some(path), &Default::default())
            .await
            .expect("open");
        let mut agent = build_test_agent(provider, registry, &store);
        agent.enable_background_for_test(2);
        (agent, store)
    }

    /// Make the next `count` writes that would finish a task as `completed` fail, under a live
    /// connection. `RAISE(FAIL)` keeps the counter the trigger bumped rather than rolling it back
    /// with the statement, so the failures are counted down and the write after them succeeds.
    fn fail_completed_writes(path: &std::path::Path, count: usize) {
        rusqlite::Connection::open(path)
            .expect("second connection")
            .execute_batch(&format!(
                "CREATE TABLE injected_failures (remaining INTEGER NOT NULL); \
                 INSERT INTO injected_failures VALUES ({count}); \
                 CREATE TRIGGER injected_failure BEFORE UPDATE ON background_tasks \
                 WHEN NEW.status = 'completed' AND (SELECT remaining FROM injected_failures) > 0 \
                 BEGIN \
                   UPDATE injected_failures SET remaining = remaining - 1; \
                   SELECT RAISE(FAIL, 'injected store failure'); \
                 END;"
            ))
            .expect("install the trigger");
    }

    /// Run one detached call to completion and return its row.
    async fn finished_background_task(
        agent: &Agent,
        store: &Store,
    ) -> crate::store::background::BackgroundTask {
        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("run the suite".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the turn must not block on the task");
        agent.background_tasks().wait_all().await;
        let mut tasks = store
            .background_store()
            .list_background_tasks(agent.session_id().expect("session"))
            .await
            .expect("list tasks");
        assert_eq!(tasks.len(), 1, "{tasks:?}");
        tasks.remove(0)
    }

    /// A finished task whose outcome write fails once is recorded on the retry.
    #[tokio::test]
    async fn a_background_outcome_write_is_retried_once() {
        use crate::provider::mock::MockProvider;

        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("meka.db");
        let provider = Arc::new(MockProvider::from_rounds(vec![
            background_round("cargo test"),
            text_round("started it"),
        ]));
        let (agent, store) = background_agent_at(provider, Arc::new(QuickFixture), &path).await;
        fail_completed_writes(&path, 1);

        let task = finished_background_task(&agent, &store).await;
        assert_eq!(
            task.status,
            crate::store::background::TaskStatus::Completed,
            "{task:?}"
        );
        assert_eq!(task.outcome.as_deref(), Some("42 passed"), "{task:?}");
    }

    /// A finished task whose outcome cannot be recorded is marked `failed` with the error, not
    /// left `running`. A `running` row with no task behind it is swept to `interrupted` on the
    /// next session open, which tells the model the work died when it finished.
    #[tokio::test]
    async fn a_background_task_whose_outcome_cannot_be_recorded_is_marked_failed() {
        use crate::provider::mock::MockProvider;

        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("meka.db");
        let provider = Arc::new(MockProvider::from_rounds(vec![
            background_round("cargo test"),
            text_round("started it"),
        ]));
        let (agent, store) = background_agent_at(provider, Arc::new(QuickFixture), &path).await;
        fail_completed_writes(&path, usize::MAX);

        let task = finished_background_task(&agent, &store).await;
        assert_eq!(
            task.status,
            crate::store::background::TaskStatus::Failed,
            "{task:?}"
        );
        let outcome = task.outcome.as_deref().unwrap_or_default();
        assert!(
            outcome.contains("could not be recorded") && outcome.contains("injected store failure"),
            "the row carries why: {task:?}"
        );
    }

    fn background_round(command: &str) -> Vec<crate::provider::mock::MockEvent> {
        use crate::provider::mock::{MockEvent, MockStopReason};
        vec![
            MockEvent::ToolUseStart {
                id: "call-1".to_string(),
                name: "execute_command".to_string(),
            },
            MockEvent::ToolUseEnd {
                input: serde_json::json!({"command": command, "background": true}),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::ToolUse,
            },
        ]
    }

    /// The whole point: the turn ends without waiting, and the model is handed a task id rather
    /// than a result.
    #[tokio::test]
    async fn a_background_call_returns_a_handle_and_ends_the_turn() {
        use crate::provider::mock::MockProvider;

        let provider = Arc::new(MockProvider::from_rounds(vec![
            background_round("sleep 600"),
            text_round("started it"),
        ]));
        let (agent, store) = background_agent(provider, Arc::new(SlowFixture)).await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("run the suite".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the turn must not block on the task");

        let results: String = messages
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|block| match block {
                ContentBlock::ToolResult { content, .. } => {
                    Some(ContentBlock::tool_result_text_content(content))
                }
                _ => None,
            })
            .collect();
        assert!(results.contains("Started in the background"), "{results}");
        assert!(results.contains("task_cancel"), "{results}");

        let running = store
            .background_store()
            .list_running_background_tasks(agent.session_id().expect("session"))
            .await
            .expect("list running");
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].label, "sleep 600");

        // Leave nothing behind for the next test's runtime to trip over.
        agent.background_tasks().cancel_all().await;
    }

    /// With `[background] enabled = false` the parameter is never offered, so a model asking for it
    /// is guessing. It still must not be silently ignored: running a twenty-minute command in the
    /// foreground because a flag was dropped is exactly the surprise the whole feature exists to
    /// avoid, and the refusal tells the model to reissue the call plainly.
    #[tokio::test]
    async fn background_is_refused_when_the_installation_disabled_it() {
        use crate::provider::mock::MockProvider;

        let registry = crate::tools::ToolRegistry::new();
        registry
            .register(Arc::new(QuickFixture))
            .expect("register fixture");
        // Deliberately no `enable_background`: this is what a default installation looks like.
        let provider = Arc::new(MockProvider::from_rounds(vec![
            background_round("cargo test"),
            text_round("understood"),
        ]));
        let (agent, store) = agent_with_registry_for_test(provider, registry).await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("run the suite".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");

        let results: String = messages
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|block| match block {
                ContentBlock::ToolResult { content, .. } => {
                    Some(ContentBlock::tool_result_text_content(content))
                }
                _ => None,
            })
            .collect();
        assert!(
            results.contains("background calls are disabled"),
            "{results}"
        );
        assert!(results.contains("without `background`"), "{results}");
        assert!(
            !results.contains("42 passed"),
            "the call must be refused, not quietly run in the foreground: {results}"
        );
        assert!(
            store
                .background_store()
                .list_background_tasks(agent.session_id().expect("session"))
                .await
                .expect("list")
                .is_empty(),
            "a disabled installation must not record tasks"
        );
    }

    /// A finished task's outcome has to reach the conversation, and exactly once.
    #[tokio::test]
    async fn a_finished_task_is_recorded_for_delivery_once() {
        use crate::provider::mock::MockProvider;

        let provider = Arc::new(MockProvider::from_rounds(vec![
            background_round("cargo test"),
            text_round("started it"),
        ]));
        let (agent, store) = background_agent(provider, Arc::new(QuickFixture)).await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("run the suite".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");
        let session_id = agent.session_id().expect("session");
        agent.background_tasks().wait_for_session(session_id).await;

        let ready = store
            .background_store()
            .list_undelivered_background_tasks(session_id)
            .await
            .expect("list undelivered");
        assert_eq!(ready.len(), 1);
        assert_eq!(
            ready[0].status,
            crate::store::background::TaskStatus::Completed
        );
        assert_eq!(ready[0].outcome.as_deref(), Some("42 passed"));

        let rendered = crate::background::render_outcomes(&ready);
        assert!(rendered.contains("42 passed"), "{rendered}");
        assert!(rendered.contains("cargo test"), "{rendered}");

        let ids: Vec<String> = ready.iter().map(|task| task.id.clone()).collect();
        store
            .background_store()
            .mark_background_tasks_delivered(&ids)
            .await
            .expect("stamp");
        assert!(
            store
                .background_store()
                .list_undelivered_background_tasks(session_id)
                .await
                .expect("list undelivered")
                .is_empty(),
            "an outcome must reach the conversation once, not on every tick"
        );
    }

    /// Nothing awaits a background task's `JoinHandle` outside `--oneshot`, so an unwind that
    /// escaped would skip both the outcome write and the slot release: a report that never comes,
    /// and a ceiling permanently one lower.
    #[tokio::test]
    async fn a_panicking_background_tool_still_reports_and_frees_its_slot() {
        use crate::provider::mock::MockProvider;

        struct PanicFixture;

        #[async_trait::async_trait]
        impl crate::tools::Tool for PanicFixture {
            fn definition(&self) -> ToolDefinition {
                SlowFixture.definition()
            }

            fn required_permission(&self) -> crate::permission::Permission {
                crate::permission::Permission::Read
            }

            async fn execute(
                &self,
                _input: serde_json::Value,
                _context: crate::tools::ToolContext,
            ) -> Result<crate::tools::ToolOutput> {
                panic!("tool exploded");
            }
        }

        let provider = Arc::new(MockProvider::from_rounds(vec![
            background_round("boom"),
            text_round("started"),
        ]));
        let (agent, store) = background_agent(
            provider,
            Arc::new(PanicFixture) as Arc<dyn crate::tools::Tool>,
        )
        .await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("run it".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");
        let session_id = agent.session_id().expect("session");
        agent.background_tasks().wait_for_session(session_id).await;

        let ready = store
            .background_store()
            .list_undelivered_background_tasks(session_id)
            .await
            .expect("list undelivered");
        assert_eq!(ready.len(), 1, "the panic must still produce a report");
        assert_eq!(
            ready[0].status,
            crate::store::background::TaskStatus::Failed
        );

        assert_eq!(
            agent.background_tasks().running_count(session_id).await,
            0,
            "the slot must be released, or the ceiling shrinks for the session's lifetime"
        );
    }

    /// The concurrency ceiling refuses rather than silently queueing, so the model can decide
    /// whether to wait or run the call in the foreground.
    #[tokio::test]
    async fn the_task_ceiling_refuses_with_something_actionable() {
        use crate::provider::mock::MockProvider;

        let provider = Arc::new(MockProvider::from_rounds(vec![
            background_round("sleep 1"),
            background_round("sleep 2"),
            background_round("sleep 3"),
            text_round("done"),
        ]));
        let (agent, _store) = background_agent(provider, Arc::new(SlowFixture)).await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("start three".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");

        let results: String = messages
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|block| match block {
                ContentBlock::ToolResult { content, .. } => {
                    Some(ContentBlock::tool_result_text_content(content))
                }
                _ => None,
            })
            .collect();
        assert!(results.contains("which is the limit"), "{results}");
        assert!(results.contains("without `background`"), "{results}");

        agent.background_tasks().cancel_all().await;
    }

    /// `background` is meka's own. A tool must never see it, least of all a remote MCP server that
    /// never advertised the key.
    #[tokio::test]
    async fn the_background_flag_never_reaches_the_tool() {
        use crate::provider::mock::MockProvider;

        struct RecordingFixture(Arc<std::sync::Mutex<Option<serde_json::Value>>>);

        #[async_trait::async_trait]
        impl crate::tools::Tool for RecordingFixture {
            fn definition(&self) -> ToolDefinition {
                SlowFixture.definition()
            }

            fn required_permission(&self) -> crate::permission::Permission {
                crate::permission::Permission::Read
            }

            async fn execute(
                &self,
                input: serde_json::Value,
                _context: crate::tools::ToolContext,
            ) -> Result<crate::tools::ToolOutput> {
                *self
                    .0
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(input);
                Ok(crate::tools::ToolOutput::text("ok".to_string(), false))
            }
        }

        let seen = Arc::new(std::sync::Mutex::new(None));
        let provider = Arc::new(MockProvider::from_rounds(vec![
            background_round("make"),
            text_round("started"),
        ]));
        let (agent, _store) = background_agent(
            provider,
            Arc::new(RecordingFixture(Arc::clone(&seen))) as Arc<dyn crate::tools::Tool>,
        )
        .await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("build".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");
        agent
            .background_tasks()
            .wait_for_session(agent.session_id().expect("session"))
            .await;

        let seen = seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let seen = seen.expect("the tool ran");
        assert_eq!(seen, serde_json::json!({"command": "make"}));
    }

    /// A wrong-typed `background` refuses the call and says what to send, rather than running it in
    /// the foreground. Models that stringify every argument (GLM through OpenRouter does) would
    /// otherwise get a twenty-minute block where they asked for a detach, with nothing in the
    /// transcript accounting for it; here they are told, and the retry is theirs to make.
    #[tokio::test]
    async fn a_wrong_typed_background_flag_refuses_the_call() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));

        struct WitnessFixture(Arc<std::sync::atomic::AtomicBool>);

        #[async_trait::async_trait]
        impl crate::tools::Tool for WitnessFixture {
            fn definition(&self) -> ToolDefinition {
                SlowFixture.definition()
            }

            fn required_permission(&self) -> crate::permission::Permission {
                crate::permission::Permission::Read
            }

            async fn execute(
                &self,
                _input: serde_json::Value,
                _context: crate::tools::ToolContext,
            ) -> Result<crate::tools::ToolOutput> {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(crate::tools::ToolOutput::text("ok".to_string(), false))
            }
        }

        let stringified = vec![
            MockEvent::ToolUseStart {
                id: "call-1".to_string(),
                name: "execute_command".to_string(),
            },
            MockEvent::ToolUseEnd {
                input: serde_json::json!({"command": "make", "background": "true"}),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::ToolUse,
            },
        ];
        let provider = Arc::new(MockProvider::from_rounds(vec![
            stringified,
            text_round("understood"),
        ]));
        let (agent, _store) = background_agent(
            provider,
            Arc::new(WitnessFixture(Arc::clone(&ran))) as Arc<dyn crate::tools::Tool>,
        )
        .await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("build".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");

        assert!(
            !ran.load(std::sync::atomic::Ordering::SeqCst),
            "a call meka could not read must not run at all, least of all in the foreground",
        );
        let results: String = messages
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|block| match block {
                ContentBlock::ToolResult { content, .. } => {
                    Some(ContentBlock::tool_result_text_content(content))
                }
                _ => None,
            })
            .collect();
        assert!(results.contains("background"), "{results}");
        assert!(results.contains("boolean"), "{results}");
    }

    /// Once the model has loaded the schema, the same call is its own business.
    #[tokio::test]
    async fn loaded_tool_call_gets_no_advisory() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let registry = send_file_registry();
        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![
                MockEvent::ToolUseStart {
                    id: "load-1".to_string(),
                    name: crate::tools::LOAD_TOOL_NAME.to_string(),
                },
                MockEvent::ToolUseEnd {
                    input: serde_json::json!({"name": "mcp__bridge__send_file"}),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::ToolUse,
                },
            ],
            send_file_round(serde_json::json!({"path": "/tmp/a.png"})),
            vec![
                MockEvent::Text {
                    text: "sent".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let (agent, _store) = agent_with_registry_for_test(provider, registry).await;

        let mut messages = Conversation::new();
        agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("send the picture".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("turn succeeds");

        let results: String = messages
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|block| match block {
                ContentBlock::ToolResult { content, .. } => {
                    Some(ContentBlock::tool_result_text_content(content))
                }
                _ => None,
            })
            .collect();

        assert!(results.contains("Sent (message id 1)"), "{results}");
        assert!(
            !results.contains(crate::conversation::HARNESS_NOTE),
            "the schema was loaded; nothing to advise: {results}"
        );
    }

    /// A 400 that isn't about content (`max_tokens` over the ceiling, a bad header) has nothing to
    /// degrade, so it must fail immediately rather than spend a retry.
    #[tokio::test]
    async fn run_turn_does_not_retry_a_rejection_with_nothing_to_degrade() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![MockEvent::FailInvalidRequest {
                message: "max_tokens: 999999 > 8192, the maximum for this model".to_string(),
            }],
            // Reaching this round would mean a retry was spent on an unrepairable request.
            vec![
                MockEvent::Text {
                    text: "should never run".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let (agent, _store) = agent_for_test(provider).await;

        let mut messages = Conversation::new();
        let error = agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("plain text only".to_string(), Vec::new())
                    .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect_err("nothing to repair, so the turn fails");
        assert!(matches!(error, MekaError::InvalidRequest(_)), "{error}");
    }

    /// The degrade notice has to name the way back to what it removed.
    ///
    /// A repair is not a deletion: the log is append-only, so the superseded rows stay on disk and
    /// `format_session_as_markdown` renders them above the repair marker. That is only useful to
    /// somebody who knows it, and `meka session export` is not a command a user would guess at the
    /// moment their attachment disappears. The notice is the one place they are certainly looking.
    #[tokio::test]
    async fn the_degrade_notice_says_where_the_original_went() {
        use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

        let provider = Arc::new(MockProvider::from_rounds(vec![
            vec![MockEvent::FailInvalidRequest {
                message: REJECTION.to_string(),
            }],
            vec![
                MockEvent::Text {
                    text: "done".to_string(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
        ]));
        let (agent, frontend) = agent_recording_for_test(provider).await;
        agent
            .run_turn(
                &mut Conversation::new(),
                crate::agent::TurnInput::from_parts("look at this".to_string(), vec![
                    image_source(),
                ])
                .expect("a prompt"),
                CancellationToken::new(),
            )
            .await
            .expect("the degrade got the turn through");

        let notices: Vec<String> = frontend
            .events()
            .iter()
            .filter_map(|event| match event {
                FrontendEvent::Notice(notice) => Some(notice.text.clone()),
                _ => None,
            })
            .collect();
        assert!(
            notices
                .iter()
                .any(|text| text.contains("meka session export --format json")),
            "a user whose content just vanished has to be told it is still recoverable: {notices:?}"
        );
    }

    #[test]
    fn empty_turn_notice_includes_unknown_stop_reason() {
        assert_eq!(
            empty_turn_notice(&StopReason::Refusal("custom refusal".to_string())),
            "custom refusal"
        );
        assert!(
            empty_turn_notice(&StopReason::Refusal(String::new())).contains("declined to respond")
        );
        assert!(empty_turn_notice(&StopReason::MaxTokens).contains("output limit"));
        // The raw reason of an unrecognized stop reason must be surfaced, not swallowed.
        let notice = empty_turn_notice(&StopReason::Unknown("pause_turn".to_string()));
        assert!(notice.contains("pause_turn"), "got: {notice}");
        assert!(empty_turn_notice(&StopReason::EndTurn).contains("empty response"));
    }

    #[test]
    fn only_non_blank_text_counts_as_visible() {
        assert!(!has_visible_text(&[]));
        assert!(!has_visible_text(&[ContentBlock::Thinking {
            thinking: "pondering".to_string(),
            opaque: None,
        }]));
        // Whitespace-only text is not visible output.
        assert!(!has_visible_text(&[ContentBlock::Text {
            text: "   \n".to_string(),
        }]));
        assert!(!has_visible_text(&[ContentBlock::ToolUse {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            input: serde_json::json!({}),
        }]));
        assert!(has_visible_text(&[ContentBlock::Text {
            text: "hello".to_string(),
        }]));
        // A thinking block followed by real text still counts as visible.
        assert!(has_visible_text(&[
            ContentBlock::Thinking {
                thinking: "pondering".to_string(),
                opaque: None,
            },
            ContentBlock::Text {
                text: "answer".to_string(),
            },
        ]));
    }

    fn not_connected(name: &str, required: bool) -> crate::mcp::NotConnected {
        crate::mcp::NotConnected {
            name: name.to_string(),
            required,
            state: crate::mcp::ServerState::Failed {
                error: "boom".to_string(),
                at: std::time::Instant::now(),
            },
        }
    }

    /// The point of the change: an optional server that is down must not stop the session. A
    /// container without `ida-mcp` should still run every turn that doesn't need IDA.
    #[test]
    fn optional_servers_do_not_gate_the_turn() {
        assert!(gate_on_required_servers(vec![]).is_ok());
        assert!(
            gate_on_required_servers(vec![
                not_connected("ida", false),
                not_connected("exa", false)
            ])
            .is_ok()
        );
    }

    /// The rejection must name the cause, not just "failed": it is the only thing the user sees
    /// when a required server blocks every turn, and the connector's warn fires once and stops.
    #[test]
    fn required_server_gates_the_turn() {
        let error = gate_on_required_servers(vec![not_connected("bridge", true)])
            .expect_err("a required server must gate");
        match error {
            MekaError::McpTurnGated { servers } => {
                assert_eq!(servers.len(), 1);
                assert_eq!(servers[0].0, "bridge");
                // The cause, not the "failed" label: every cause already reads as a failure, so
                // the label would only produce "failed: failed to ...".
                assert_eq!(servers[0].1, "boom");
            }
            other => panic!("expected McpTurnGated, got {other:?}"),
        }
        assert!(
            gate_on_required_servers(vec![crate::mcp::NotConnected {
                name: "slow".to_string(),
                required: true,
                state: crate::mcp::ServerState::Pending,
            }])
            .expect_err("pending required server still gates")
            .to_string()
            .contains("pending")
        );
    }

    /// A mixed fleet gates on the required one and names only it: listing the optional servers
    /// would imply they are the problem.
    #[test]
    fn gate_names_only_the_required_servers() {
        let error = gate_on_required_servers(vec![
            not_connected("ida", false),
            not_connected("bridge", true),
            not_connected("exa", false),
        ])
        .expect_err("a required server must gate even alongside optional ones");
        match error {
            MekaError::McpTurnGated { servers } => {
                let names: Vec<&str> = servers.iter().map(|(n, _)| n.as_str()).collect();
                assert_eq!(names, vec!["bridge"]);
            }
            other => panic!("expected McpTurnGated, got {other:?}"),
        }
    }

    #[test]
    fn a_thinking_only_round_is_nudged_once() {
        // Thinking-only end_turn: nudge once.
        assert!(should_nudge_thinking_only(
            false,
            false,
            &StopReason::EndTurn,
            false
        ));
        // Thinking-only unrecognized reason (e.g. pause_turn): nudge once.
        assert!(should_nudge_thinking_only(
            false,
            false,
            &StopReason::Unknown("pause_turn".to_string()),
            false,
        ));
        // Already nudged this turn: no second nudge (prevents loops).
        assert!(!should_nudge_thinking_only(
            false,
            false,
            &StopReason::EndTurn,
            true
        ));
        // Visible text present: nothing to recover.
        assert!(!should_nudge_thinking_only(
            false,
            true,
            &StopReason::EndTurn,
            false
        ));
        // Tool calls present: the tool path drives continuation.
        assert!(!should_nudge_thinking_only(
            true,
            false,
            &StopReason::EndTurn,
            false
        ));
        // MaxTokens and Refusal carry their own outcomes; don't retry them.
        assert!(!should_nudge_thinking_only(
            false,
            false,
            &StopReason::MaxTokens,
            false
        ));
        assert!(!should_nudge_thinking_only(
            false,
            false,
            &StopReason::Refusal(String::new()),
            false,
        ));
    }

    #[test]
    fn truncate_no_limit() {
        let messages = vec![user_message("hello"), assistant_message("hi")];
        let result = truncate_messages_for_context(&messages, None);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn truncate_under_limit() {
        let messages = vec![user_message("hello"), assistant_message("hi")];
        let result = truncate_messages_for_context(&messages, Some(10));
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn truncate_over_limit() {
        let messages = vec![
            user_message("first"),
            assistant_message("response1"),
            user_message("second"),
            assistant_message("response2"),
            user_message("third"),
            assistant_message("response3"),
        ];
        let result = truncate_messages_for_context(&messages, Some(4));
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].role, Role::User);
    }

    #[test]
    fn truncate_does_not_split_tool_chain() {
        let messages = vec![
            user_message("first"),
            assistant_message("response1"),
            user_message("second"),
            assistant_tool_use(),
            tool_result_message(),
            assistant_message("final"),
        ];
        // Limit 3 would naively start at index 3 (assistant_tool_use), but that splits the tool
        // chain. It should walk back to index 2 (user "second").
        let result = truncate_messages_for_context(&messages, Some(3));
        assert_eq!(result[0].role, Role::User);
        assert!(!has_tool_results(&result[0].content));
        assert!(result.len() >= 3);
    }

    #[test]
    fn truncate_starts_with_user() {
        let messages = vec![
            user_message("first"),
            assistant_message("response1"),
            assistant_message("response2"),
            user_message("second"),
            assistant_message("response3"),
        ];
        // Limit 2 would naively start at index 3, which is a user message
        let result = truncate_messages_for_context(&messages, Some(2));
        assert_eq!(result[0].role, Role::User);
    }

    #[test]
    fn truncate_skips_forward_past_tool_result() {
        let messages = vec![
            user_message("first"),
            assistant_tool_use(),
            tool_result_message(),
            assistant_message("response"),
            user_message("second"),
            assistant_message("response2"),
        ];
        // Limit 4 lands on index 2 (tool_result_message), which would orphan the tool_use above it.
        // The next safe cut ahead is index 4 (user "second"); the pair is dropped whole.
        let result = truncate_messages_for_context(&messages, Some(4));
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].role, Role::User);
        assert!(!has_tool_results(&result[0].content));
    }

    /// `context_messages` is a maximum, and reaching back over a tool chain to find a cut point
    /// would let one long tool loop ignore it entirely.
    #[test]
    fn a_long_tool_loop_cannot_carry_the_window_past_its_cap() {
        let mut messages = vec![user_message("go")];
        for _ in 0..5 {
            messages.push(assistant_tool_use());
            messages.push(tool_result_message());
        }
        messages.push(user_message("and now this"));

        let result = truncate_messages_for_context(&messages, Some(4));
        assert!(
            result.len() <= 4,
            "{} messages survived the cap",
            result.len()
        );
        assert_eq!(result[0].role, Role::User);
        assert!(!has_tool_results(&result[0].content));
    }

    /// When nothing ahead is a safe cut, reaching back is still right: an invalid conversation the
    /// provider rejects is worse than one over the cap.
    #[test]
    fn an_unbroken_trailing_tool_chain_falls_back_to_reaching_back() {
        let mut messages = vec![user_message("go")];
        for _ in 0..5 {
            messages.push(assistant_tool_use());
            messages.push(tool_result_message());
        }

        let result = truncate_messages_for_context(&messages, Some(4));
        assert_eq!(result.len(), messages.len());
        assert_eq!(result[0].role, Role::User);
        assert!(!has_tool_results(&result[0].content));
    }

    fn assistant_tool_use_named(id: &str, name: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: id.to_string(),
                name: name.to_string(),
                input: serde_json::json!({"path": "/tmp/test"}),
            }],
        }
    }

    fn tool_result_for(tool_use_id: &str, content: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.to_string(),
                content: vec![ToolResultContent::Text {
                    text: content.to_string(),
                }],
                is_error: false,
            }],
        }
    }

    /// Compares two message slices for semantic equality (same role, same content blocks), which
    /// is what determines whether the KV cache prefix is reusable.
    fn assert_messages_equal(a: &[Message], b: &[Message], context: &str) {
        assert_eq!(a.len(), b.len(), "{context}: length mismatch");
        for (i, (ma, mb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(ma.role, mb.role, "{context}: role mismatch at index {i}");
            assert_eq!(
                ma.content.len(),
                mb.content.len(),
                "{context}: content block count mismatch at index {i}"
            );
            let json_a = serde_json::to_string(&ma.content).unwrap();
            let json_b = serde_json::to_string(&mb.content).unwrap();
            assert_eq!(json_a, json_b, "{context}: content mismatch at index {i}");
        }
    }

    #[test]
    fn stable_base_during_tool_loop() {
        // A conversation with history, then a tool loop that adds 3 tool call/result pairs. The
        // base prefix (everything before the tool loop) must be identical across all iterations.
        let mut messages = vec![
            user_message("first question"),
            assistant_message("first answer"),
            user_message("second question"),
        ];

        let base_messages = truncate_messages_for_context(&messages, None);
        let turn_start_len = messages.len();

        let api_iter0 = assemble_api_messages(&messages, &base_messages, turn_start_len, None);
        assert_eq!(api_iter0.len(), 3);

        // Iteration 1: model calls a tool
        messages.push(assistant_tool_use_named("t1", "read_file"));
        messages.push(tool_result_for("t1", "file contents"));

        let api_iter1 = assemble_api_messages(&messages, &base_messages, turn_start_len, None);
        assert_eq!(api_iter1.len(), 5);

        // The first 3 messages (the base) must be identical.
        assert_messages_equal(&api_iter0[..3], &api_iter1[..3], "iter0→iter1 base");

        // Iteration 2: model calls another tool
        messages.push(assistant_tool_use_named("t2", "execute_command"));
        messages.push(tool_result_for("t2", "command output"));

        let api_iter2 = assemble_api_messages(&messages, &base_messages, turn_start_len, None);
        assert_eq!(api_iter2.len(), 7);

        // Base is still identical.
        assert_messages_equal(&api_iter0[..3], &api_iter2[..3], "iter0→iter2 base");
        // And the first 5 (base + iter1's additions) are identical too.
        assert_messages_equal(&api_iter1[..5], &api_iter2[..5], "iter1→iter2 prefix");

        // Iteration 3: yet another tool call
        messages.push(assistant_tool_use_named("t3", "read_file"));
        messages.push(tool_result_for("t3", "more contents"));

        let api_iter3 = assemble_api_messages(&messages, &base_messages, turn_start_len, None);
        assert_eq!(api_iter3.len(), 9);

        assert_messages_equal(&api_iter2[..7], &api_iter3[..7], "iter2→iter3 prefix");
        assert_messages_equal(&api_iter0[..3], &api_iter3[..3], "iter0→iter3 base");
    }

    /// Once a tool loop pushes the assembled request past `context_messages`, the window moves
    /// forward with it. This is the trade the per-round truncation makes.
    ///
    /// Applying the cap once at turn start would freeze the base for the whole turn, which is what
    /// would make `context_messages` stop applying the moment a turn makes its second provider
    /// call.
    ///
    /// The cost is real: a prefix that moves is a prefix the provider cannot serve from cache, so
    /// a long tool loop re-reads its window several times per turn. The alternative is a cap that
    /// does not hold, which is worse, since an unbounded request eventually hits the context limit
    /// the setting exists to avoid.
    #[test]
    fn the_window_moves_forward_when_a_tool_loop_pushes_past_the_cap() {
        let limit = Some(6);

        let mut messages = vec![
            user_message("msg-1"),
            assistant_message("resp-1"),
            user_message("msg-2"),
            assistant_message("resp-2"),
            user_message("msg-3"),
        ];

        let base_messages = truncate_messages_for_context(&messages, limit);
        let turn_start_len = messages.len();
        assert_eq!(base_messages.len(), 5, "five fits under a cap of six");

        let api_iter0 = assemble_api_messages(&messages, &base_messages, turn_start_len, limit);
        assert_eq!(api_iter0.len(), 5, "nothing appended yet");

        // Round 1 takes the assembled request to seven, over the cap.
        messages.push(assistant_tool_use_named("t1", "read_file"));
        messages.push(tool_result_for("t1", "data"));
        let api_iter1 = assemble_api_messages(&messages, &base_messages, turn_start_len, limit);

        // Round 2 takes it to nine.
        messages.push(assistant_tool_use_named("t2", "execute_command"));
        messages.push(tool_result_for("t2", "output"));
        let api_iter2 = assemble_api_messages(&messages, &base_messages, turn_start_len, limit);

        for (round, request) in [(1, &api_iter1), (2, &api_iter2)] {
            assert!(
                request.len() <= 6,
                "round {round} sent {} messages under a cap of 6",
                request.len(),
            );
            assert_eq!(
                request.first().map(|message| &message.role),
                Some(&Role::User),
                "round {round} must start on a role the provider accepts",
            );
            assert!(
                !has_tool_results(&request.first().expect("non-empty").content),
                "round {round} must not start mid tool chain",
            );
        }

        // What the round costs: the request no longer opens on the same message it did before, so
        // the cached prefix ends where the two diverge.
        let first_of = |request: &[Message]| serde_json::to_string(&request[0].content).unwrap();
        assert_ne!(
            first_of(&api_iter1),
            first_of(&api_iter2),
            "the window is expected to move once the cap bites; if this ever holds, the cap has \
             stopped applying inside the turn again",
        );

        // And the newest messages always survive: the cut only ever comes off the front.
        let newest = serde_json::to_string(&messages[messages.len() - 1].content).unwrap();
        assert_eq!(
            serde_json::to_string(&api_iter2[api_iter2.len() - 1].content).unwrap(),
            newest,
        );
    }

    #[test]
    fn truncation_with_tool_chain_near_boundary() {
        // Verify that when the conversation includes a tool chain right at the truncation boundary,
        // the base is computed correctly and stays stable.
        let limit = Some(4);

        let mut messages = vec![
            user_message("old-msg"),
            assistant_message("old-resp"),
            user_message("current question"),
            assistant_tool_use_named("t0", "read_file"),
            tool_result_for("t0", "initial data"),
            assistant_message("here is the data"),
            user_message("follow-up"),
        ];

        let base_messages = truncate_messages_for_context(&messages, limit);
        let turn_start_len = messages.len();

        // The truncation should keep a safe cut point; verify it starts with a user message and
        // doesn't split tool chains.
        assert_eq!(base_messages[0].role, Role::User);
        assert!(!has_tool_results(&base_messages[0].content));

        let api_iter0 = assemble_api_messages(&messages, &base_messages, turn_start_len, limit);

        // Add tool loop messages
        messages.push(assistant_tool_use_named("t1", "read_file"));
        messages.push(tool_result_for("t1", "more data"));

        let api_iter1 = assemble_api_messages(&messages, &base_messages, turn_start_len, limit);

        // The base portion must be identical.
        let base_len = base_messages.len();
        assert_messages_equal(
            &api_iter0[..base_len],
            &api_iter1[..base_len],
            "base stable after tool loop",
        );
    }

    /// The tool catalog, skill list and MCP instructions do not live in the system prompt, which
    /// is sent unconditionally. They now live in one user message, which
    /// `truncate_messages_for_context` will drop once the conversation outgrows `context_messages`
    /// (200 by default). Without this check the snapshot would still claim the model had been told,
    /// and a long session would run with no catalog at all.
    #[test]
    fn world_state_is_restated_once_it_scrolls_out_of_the_window() {
        // Rendered at index 0, window of 200.
        assert!(
            world_state_still_visible(0, 10, Some(200)),
            "a fresh render is visible"
        );
        assert!(
            world_state_still_visible(0, 199, Some(200)),
            "still inside the window one message before the cut"
        );
        assert!(
            !world_state_still_visible(0, 200, Some(200)),
            "the render has reached the edge and must be restated"
        );
        assert!(
            !world_state_still_visible(0, 5_000, Some(200)),
            "a long session must not run on a render that scrolled away"
        );

        // A restatement lands at the current tail and buys another window.
        assert!(world_state_still_visible(4_900, 5_000, Some(200)));

        // No limit means nothing is ever dropped, so a single render lasts the session.
        assert!(world_state_still_visible(0, 100_000, None));
    }

    #[test]
    fn no_limit_produces_full_prefix() {
        // With no context_messages limit, base_messages includes everything, and tool loop
        // additions are appended without any truncation.
        let mut messages = vec![user_message("a"), assistant_message("b"), user_message("c")];

        let base_messages = truncate_messages_for_context(&messages, None);
        let turn_start_len = messages.len();

        assert_eq!(base_messages.len(), 3);

        let api_iter0 = assemble_api_messages(&messages, &base_messages, turn_start_len, None);
        assert_eq!(api_iter0.len(), 3);

        // Add many tool calls
        for i in 0..5 {
            messages.push(assistant_tool_use_named(&format!("t{i}"), "read_file"));
            messages.push(tool_result_for(&format!("t{i}"), &format!("result {i}")));
        }

        let api_final = assemble_api_messages(&messages, &base_messages, turn_start_len, None);
        assert_eq!(api_final.len(), 13); // 3 base + 10 tool messages

        // Base prefix still matches.
        assert_messages_equal(&api_iter0[..3], &api_final[..3], "full prefix stable");
    }

    #[test]
    fn multi_turn_truncation_keeps_every_request_well_formed() {
        // Two turns, each computing its own base. Turn 1 stays under the cap, so its base is
        // stable across the loop; turn 2 crosses it, so the window moves and only the
        // well-formedness invariants hold.
        let limit = Some(6);

        // -- Turn 1 --
        let mut messages: Vec<Message> = vec![user_message("turn-1 question")];
        let base_t1 = truncate_messages_for_context(&messages, limit);
        let start_t1 = messages.len();

        // Tool loop: 2 iterations
        messages.push(assistant_tool_use_named("t1a", "read_file"));
        messages.push(tool_result_for("t1a", "data-a"));
        let api_t1_iter1 = assemble_api_messages(&messages, &base_t1, start_t1, limit);

        messages.push(assistant_message("here's your answer"));
        let api_t1_iter2 = assemble_api_messages(&messages, &base_t1, start_t1, limit);

        // Base is stable within turn 1.
        assert_messages_equal(
            &api_t1_iter1[..base_t1.len()],
            &api_t1_iter2[..base_t1.len()],
            "turn 1 base stable",
        );

        // -- Turn 2 --
        messages.push(user_message("turn-2 question"));

        let base_t2 = truncate_messages_for_context(&messages, limit);
        let start_t2 = messages.len();

        messages.push(assistant_tool_use_named("t2a", "execute_command"));
        messages.push(tool_result_for("t2a", "output"));
        let api_t2_iter1 = assemble_api_messages(&messages, &base_t2, start_t2, limit);

        messages.push(assistant_tool_use_named("t2b", "read_file"));
        messages.push(tool_result_for("t2b", "more"));
        let api_t2_iter2 = assemble_api_messages(&messages, &base_t2, start_t2, limit);

        // Turn 2 is the one where the cap bites, and there the base is not stable: the request
        // is re-truncated each round, so the window walks forward. What survives is the invariant
        // that matters: the cap holds and the request stays well-formed.
        for (round, request) in [(1, &api_t2_iter1), (2, &api_t2_iter2)] {
            assert!(
                request.len() <= 6,
                "turn 2 round {round} sent {} messages under a cap of 6",
                request.len(),
            );
            assert_eq!(
                request.first().map(|message| &message.role),
                Some(&Role::User),
                "turn 2 round {round} must start on a role the provider accepts",
            );
            assert!(
                !has_tool_results(&request.first().expect("non-empty").content),
                "turn 2 round {round} must not start mid tool chain",
            );
        }
    }
}
