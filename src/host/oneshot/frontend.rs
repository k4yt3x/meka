//! The one-shot run's [`Frontend`]: how streamed output, tool indicators and thinking reach the
//! terminal when the answer is written once, to stdout, and nothing can be asked of a human.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::{
    frontend::{Frontend, FrontendEvent, Notice, PermissionOutcome, PermissionRequest},
    host::terminal::with_console,
    render::{self},
};

/// Construction-time configuration for [`PlainFrontend`]. UI concerns, so they live on the
/// frontend impl rather than on `AgentOptions`.
pub(crate) struct PlainFrontendConfig {
    /// Where everything printed goes. The frontend decides *what* to say and the console decides
    /// how it is spaced.
    pub(crate) console: Arc<Mutex<crate::console::Console>>,
    pub(crate) show_session_id_on_create: bool,
    pub(crate) show_token_usage: bool,
    pub(crate) thinking_show_content: bool,
    pub(crate) tool_params: crate::config::ToolParams,
}
/// The write-once [`Frontend`]: a translator from [`FrontendEvent`] to
/// [`crate::console::Console`], plus the thinking indicator's own bookkeeping.
///
/// It decides *what* to say and nothing about spacing. The blank lines belong to the episode the
/// turn happens inside, which is longer than the turn and outlives one that fails, so an owner that
/// only exists while a turn is running cannot be the one that closes them.
pub(crate) struct PlainFrontend {
    pub(super) config: PlainFrontendConfig,
    pub(super) state: Mutex<PlainFrontendState>,
}
pub(super) struct PlainFrontendState {
    /// The thinking indicator currently drawn on the cursor's line. `None` when nothing is drawn.
    /// The row it occupies is the console's business; what the indicator *says* is this struct's.
    pub(super) thinking_indicator: Option<ThinkingIndicator>,
}
/// The thinking indicator currently on screen.
pub(super) struct ThinkingIndicator {
    /// The highest estimate drawn for the thinking block in progress.
    ///
    /// The server's figure is not monotonic -- a single block was observed bouncing 100 <-> 150
    /// repeatedly -- and a counter that runs backwards reads as a bug rather than as progress.
    /// Real thinking spend only accumulates, so the peak is both the steadier reading and the
    /// truer one.
    pub(super) peak_estimate: Option<u64>,
}
/// What closing out the thinking indicator means for a given event.
///
/// A pure decision, separated from `emit` so it can be asserted over every [`FrontendEvent`]
/// variant: a catch-all in that dispatch silently absorbs any variant added later, and a test that
/// enumerates the alternatives is the only thing that makes a wrong default visible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum IndicatorAction {
    /// Leave the line open: the indicator is redrawing itself.
    Keep,
    /// Erase without a trace, because the thinking block's own label is about to render in its
    /// place and would otherwise print `Thinking...` twice for one phase of reasoning.
    Erase,
    /// Write the withheld newline, making the last figure drawn a permanent line.
    Commit,
    /// Forget the indicator without writing anything.
    ///
    /// Reached only when a turn ended mid-block without closing it, which today means an interrupt:
    /// a stream error emits [`FrontendEvent::ThinkingEnded`] and so commits. The `(interrupted)`
    /// annotation has already replaced the indicator's row by the time the next turn starts, so
    /// committing here would write a stray newline.
    Drop,
}
/// Exhaustive on purpose. A catch-all absorbs a new variant into `Commit` silently, which is right
/// for most events and wrong for any that renders in the indicator's place or arrives after its
/// line is gone, and the test below can only catch that for variants someone thought to list.
/// Spelling every one out moves the gate to the compiler, which does not forget.
/// `renders_reasoning` is `[thinking].show_content`: whether this frontend is about to print the
/// reasoning itself. It decides the delta's answer and nothing else.
pub(super) fn indicator_action(event: &FrontendEvent, renders_reasoning: bool) -> IndicatorAction {
    match event {
        FrontendEvent::ThinkingProgress { .. } => IndicatorAction::Keep,
        // Erased only when something is about to render in its place, which would otherwise print
        // `Thinking...` twice for one stretch of reasoning. Erasing is idempotent, which is what
        // lets every delta after the first ask for it without checking.
        //
        // Under the default the deltas render nothing, so erasing would take the counter down for
        // an empty replacement -- and Claude sends an estimate and a delta from one wire event
        // under the token-count beta, so the next estimate would reopen the indicator rather than
        // redraw it, resetting the peak the redraw exists to hold.
        FrontendEvent::ThinkingDelta(_) if !renders_reasoning => IndicatorAction::Keep,
        FrontendEvent::ThinkingDelta(_) | FrontendEvent::ThinkingBlock { .. } => {
            IndicatorAction::Erase
        }
        FrontendEvent::TurnStarted { .. } => IndicatorAction::Drop,
        // Everything else means the thinking phase is over and nothing further will describe it,
        // so the indicator is the only record that the model spent that time.
        FrontendEvent::SessionStarted { .. }
        | FrontendEvent::TurnFinished
        | FrontendEvent::TurnEnded { .. }
        | FrontendEvent::AssistantTextDelta(_)
        | FrontendEvent::ThinkingEnded
        | FrontendEvent::ToolCallComposing { .. }
        | FrontendEvent::ToolCallStarted { .. }
        | FrontendEvent::ToolCallCompleted { .. }
        | FrontendEvent::ToolCallOutputDelta { .. }
        | FrontendEvent::ChecklistUpdated { .. }
        | FrontendEvent::SubAgentActivity { .. }
        | FrontendEvent::TokenUsage(_)
        | FrontendEvent::Notice(_)
        | FrontendEvent::McpProgress(_)
        | FrontendEvent::PromptWithdrawn
        | FrontendEvent::InboxDelivered { .. }
        | FrontendEvent::Nudged { .. }
        | FrontendEvent::Compacted { .. } => IndicatorAction::Commit,
    }
}
/// The figure to draw, given the peak already drawn for the block in progress.
///
/// `incoming` is `None` only when a thinking block opens (the provider drops the null estimate that
/// closes a block), so a `None` resets the peak instead of redrawing the previous block's total: a
/// fresh block is a fresh count, and holding the old figure would attribute it to the new one.
pub(super) fn peak_estimate(previous: Option<u64>, incoming: Option<u64>) -> Option<u64> {
    incoming.map(|estimate| previous.map_or(estimate, |peak| peak.max(estimate)))
}
impl PlainFrontend {
    pub(crate) fn new(config: PlainFrontendConfig) -> Self {
        Self {
            config,
            state: Mutex::new(PlainFrontendState {
                thinking_indicator: None,
            }),
        }
    }

    /// Close out the thinking indicator, if one is drawn, by keeping it.
    ///
    /// Writes the newline the redraw loop deliberately withheld, so the last figure drawn becomes a
    /// permanent line. The reasoning phase is then legible after the fact -- the same way a visible
    /// thinking block stays on screen -- rather than vanishing the instant the answer starts.
    pub(super) fn commit_thinking_indicator(&self, state: &mut PlainFrontendState) {
        if state.thinking_indicator.take().is_some() {
            with_console(&self.config.console, |console| console.commit_transient());
        }
    }

    /// Drop the thinking indicator without keeping it.
    ///
    /// Only for the case where a thinking block with real text is about to render: that block opens
    /// with the same `Thinking...` prefix, so committing first would print the word twice for one
    /// phase of reasoning.
    pub(super) fn erase_thinking_indicator(&self, state: &mut PlainFrontendState) {
        if state.thinking_indicator.take().is_some() {
            with_console(&self.config.console, |console| console.erase_transient());
        }
    }
}
#[async_trait]
impl Frontend for PlainFrontend {
    /// Reasoning stays on screen only when the deltas are what render it. Under the default the
    /// line shown is built from the completed block, which an attempt that failed never reaches,
    /// so a retry repeats nothing.
    fn retains_reasoning(&self) -> bool {
        self.config.thinking_show_content
    }

    #[allow(
        clippy::significant_drop_tightening,
        reason = "held across the whole dispatch: releasing it between arms lets two events interleave on the indicator row"
    )]
    async fn emit(&self, event: FrontendEvent) {
        // Held briefly across synchronous render calls. The agent loop emits events serially per
        // turn, so contention is effectively zero; the lock is purely a `Send + Sync` discipline
        // check. `clippy::await_holding_lock` (deny-level, see Cargo.toml) enforces that no
        // `.await` appears between the lock acquisition and its drop.
        let mut state = crate::sync::lock(&self.state);

        // Close out the indicator before anything else reaches the screen. Done once, here, rather
        // than in each arm that prints: the indicator sits on a line the next write would otherwise
        // overwrite halfway, and a missed call site is the kind of omission that only shows up on
        // the one path nobody exercised.
        match indicator_action(&event, self.config.thinking_show_content) {
            IndicatorAction::Keep => {}
            IndicatorAction::Erase => self.erase_thinking_indicator(&mut state),
            IndicatorAction::Commit => self.commit_thinking_indicator(&mut state),
            IndicatorAction::Drop => state.thinking_indicator = None,
        }

        match event {
            FrontendEvent::SessionStarted { id } => {
                if self.config.show_session_id_on_create {
                    with_console(&self.config.console, |console| {
                        console.session_id("Creating new session", &id.to_string())
                    });
                }
            }
            // Neither is a spacing signal any more. The blanks belong to the episode, which is
            // longer than a turn and outlives one that fails: a turn is simply one of the things
            // that can happen inside it.
            FrontendEvent::TurnStarted { .. } => {}
            // The terminal is the host's to print from the turn's result; the ending here is for
            // a frontend that has no result in hand.
            FrontendEvent::TurnEnded { .. } => {}
            // Nothing to draw. A typed prompt is never withdrawn here, and a scheduled fire's
            // withdrawal is the schedule's own bookkeeping, not news for the person at the
            // keyboard.
            FrontendEvent::PromptWithdrawn => {}
            // Nothing to draw either: a one-shot run has no producer for the inbox, so an item
            // here came from a parent agent steering this worker, whose report is what the
            // person reads.
            FrontendEvent::InboxDelivered { .. } => {}
            // The words are the model's to read; the stage direction that follows is what the
            // person sees of a nudge.
            FrontendEvent::Nudged { .. } => {}
            // Closed here so a completed turn does not hold its last paragraph until the prompt,
            // and closed again by the episode for a turn that died without reaching this.
            FrontendEvent::TurnFinished => {
                with_console(&self.config.console, |console| console.close_stream());
            }
            FrontendEvent::AssistantTextDelta(text) => {
                with_console(&self.config.console, |console| console.text_delta(&text));
            }
            FrontendEvent::ThinkingProgress { estimated_tokens } => {
                // The first estimate of a block opens it, which closes whatever was streaming --
                // the indicator draws from column zero and is kept rather than erased, so it would
                // otherwise overwrite the last row of an open run and leave the damage on screen.
                // Later estimates redraw their own row and close nothing. `Console` decides both,
                // and declines outright while reasoning is already streaming; see
                // `console::indicator_may_draw`.
                //
                // Nothing below produces output without a terminal to redraw on, and the steps
                // that follow move state shared with every other event: closing an open run and
                // taking a blank line from `spacing`. Doing either for an indicator that cannot be
                // drawn shifts the layout of output that *is* produced.
                if !render::live_indicator_supported() {
                    return;
                }
                let opening = state.thinking_indicator.is_none();
                let shown = peak_estimate(
                    state
                        .thinking_indicator
                        .as_ref()
                        .and_then(|indicator| indicator.peak_estimate),
                    estimated_tokens,
                );
                let drawn = with_console(&self.config.console, |console| {
                    console.thinking_indicator(opening, shown)
                });
                state.thinking_indicator = drawn.then_some(ThinkingIndicator {
                    peak_estimate: shown,
                });
            }
            // The indicator was committed by the hook above, which is the whole point of the
            // event. It also ends a streamed thinking block, for the turns that die mid-reasoning:
            // the renderer holds a trailing paragraph back until a blank line settles it, so an
            // unclosed block would surface under the error that follows it on the same stream.
            //
            // Only a *thinking* block. This is not a failure signal -- it fires on any block that
            // completes with nothing readable, which is every block under sealed reasoning and
            // under `redact-thinking` or display updates, on turns whose answer is streaming
            // perfectly well.
            FrontendEvent::ThinkingEnded => {
                with_console(&self.config.console, |console| console.close_thinking());
            }
            // Which of these two arms renders is decided by config, not by what this block did, so
            // neither has to ask whether the other already spoke for it.
            FrontendEvent::ThinkingDelta(text) => {
                if self.config.thinking_show_content {
                    with_console(&self.config.console, |console| {
                        console.thinking_delta(&text)
                    });
                }
            }
            FrontendEvent::ThinkingBlock { content } => {
                with_console(&self.config.console, |console| {
                    if self.config.thinking_show_content {
                        // The text is already on screen; this is only the end of the block.
                        console.close_thinking();
                    } else {
                        console.thinking_preview(&content);
                    }
                });
            }
            // The indicator is drawn at `ToolCallStarted`, where the arguments exist to draw it
            // from. Announcing the bare name first would print every call twice, and the wait it
            // marks is one the terminal already shows as the cursor sitting still.
            FrontendEvent::ToolCallComposing { .. } => {}
            FrontendEvent::ToolCallStarted {
                id: _,
                name,
                input,
                display_summary,
            } => {
                let params = self.config.tool_params;
                with_console(&self.config.console, |console| {
                    console.tool_indicator(&name, &input, display_summary.as_deref(), params)
                });
            }
            // A one-shot run shows a tool's result through the model's next message, which
            // summarizes what happened; the indicator is the whole of what the call itself gets.
            FrontendEvent::ToolCallCompleted { .. } => {}
            // Same reasoning as `ToolCallCompleted`: a one-shot run shows no tool output, so a
            // command's streamed output would be the only tool output it ever printed, and the
            // screen is where output is read as it arrives.
            FrontendEvent::ToolCallOutputDelta { .. } => {}
            FrontendEvent::ChecklistUpdated { items } => {
                with_console(&self.config.console, |console| console.checklist(&items));
            }
            FrontendEvent::TokenUsage(usage) => {
                if self.config.show_token_usage {
                    with_console(&self.config.console, |console| console.token_usage(&usage));
                }
            }
            FrontendEvent::Notice(notice) => {
                with_console(&self.config.console, |console| console.notice(&notice));
            }
            // The run already prints the sub-agent's tool indicators as they happen, via the
            // parent's own renderer; a rolling rewrite of one tool call's content has no place on
            // a stream written once.
            FrontendEvent::SubAgentActivity { .. } => {}
            // Nothing to draw: the transcript on screen is a scrollback the user wrote, not a view
            // of the model's window, so a compaction does not invalidate anything they can see.
            // `/compact` reports its own outcome through `render::compaction_summary`, and the
            // automatic paths log at `info!`. The event exists for clients that hold a *mirror* of
            // the conversation and would otherwise watch it shrink; see `host::http::sse`.
            FrontendEvent::Compacted { .. } => {}
            FrontendEvent::McpProgress(update) => {
                let line = format_progress_update(&update);
                with_console(&self.config.console, |console| {
                    // Always drawn as far as this can tell: `write_stderr` accepts a failed write
                    // rather than reporting it down the stream that just refused it.
                    console.transient(|| {
                        crate::streams::write_stderr(&line);
                        true
                    })
                });
            }
        }
    }

    async fn request_permission(&self, request: PermissionRequest) -> PermissionOutcome {
        // There is no human to ask. Said on the console rather than logged: a run whose every
        // tool is refused otherwise reads as a model that chose not to use them. Denied rather
        // than canceled, because nothing stopped the turn; the call was refused, and the tool
        // result should say so.
        self.emit(FrontendEvent::Notice(
            Notice::approval_refused_without_asking(&request.tool_name),
        ))
        .await;
        PermissionOutcome::Deny
    }

    async fn handle_elicitation(
        &self,
        prompt: crate::frontend::ElicitationPrompt,
    ) -> crate::frontend::ElicitationResponse {
        // No human to ask, as for an approval: declined, and said on the console, because the
        // tool call that needed the answer then fails with nothing else naming the cause.
        self.emit(FrontendEvent::Notice(Notice::elicitation_declined(
            &prompt.server_name,
            "no interactive prompt is available here",
        )))
        .await;
        crate::frontend::ElicitationResponse::Decline
    }
}

/// The one-row progress line for a running MCP tool. Sanitizes server-controlled strings so a
/// server cannot inject escapes to clear the screen or spoof a prompt.
///
/// Every field is flattened to a single line and width-bounded. The line opens with meka's own
/// `\r` to overwrite the previous progress, so anything that survived a newline in a server's
/// string would be painted at column zero on a fresh row, below chrome the user has already read.
fn format_progress_update(update: &crate::frontend::ProgressUpdate) -> String {
    let server = crate::text::sanitize_to_line(&update.server_name, usize::MAX);
    let tool = crate::text::sanitize_to_line(&update.tool_name, usize::MAX);
    let counter = match update.total {
        Some(total) if total > 0.0 => format!("{:.0}/{:.0}", update.progress, total),
        _ => format!("{:.0}", update.progress),
    };
    let prefix = format!("[mcp:{server}/{tool}] {counter} ");
    // What is left of the row after meka's own chrome and the trailing pad. Saturating: a narrow
    // terminal or a long name leaves no room for the message rather than underflowing.
    let pad = 5;
    let budget = crate::render::output_width()
        .saturating_sub(prefix.chars().count())
        .saturating_sub(pad);
    let message = update
        .message
        .as_deref()
        .map(|raw| crate::text::sanitize_to_line(raw, budget))
        .unwrap_or_default();
    // Padded with a few spaces so the next draw clears the tail of a longer previous line.
    format!("\r{}{}{}", prefix, message, " ".repeat(pad))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::{Frontend, FrontendEvent};

    fn console() -> Arc<Mutex<crate::console::Console>> {
        Arc::new(Mutex::new(crate::console::Console::new(
            crate::config::RenderMode::Termimad,
        )))
    }

    fn frontend_on(console: Arc<Mutex<crate::console::Console>>) -> PlainFrontend {
        PlainFrontend::new(PlainFrontendConfig {
            console,
            show_session_id_on_create: false,
            show_token_usage: false,
            thinking_show_content: false,
            tool_params: crate::config::ToolParams::Summary,
        })
    }

    fn showing_thinking(console: Arc<Mutex<crate::console::Console>>) -> PlainFrontend {
        PlainFrontend::new(PlainFrontendConfig {
            console,
            show_session_id_on_create: false,
            show_token_usage: false,
            thinking_show_content: true,
            tool_params: crate::config::ToolParams::Summary,
        })
    }

    /// Every door out of a streamed thinking block, one per case.
    ///
    /// The renderer holds a trailing paragraph back until a blank line settles it, so a block left
    /// open does not merely linger: its tail surfaces under whatever prints next. Each of these is
    /// a path that leaves one open if the close is missing from that arm alone, which is why they
    /// are enumerated rather than sampled.
    #[tokio::test]
    async fn every_way_out_of_a_thinking_block_closes_it() {
        for (name, closer) in [
            ("the block ended", FrontendEvent::ThinkingBlock {
                content: "weighing it".into(),
            }),
            ("the block ended empty", FrontendEvent::ThinkingEnded),
            ("the turn finished", FrontendEvent::TurnFinished),
            ("a tool call followed", FrontendEvent::ToolCallStarted {
                id: "1".into(),
                name: "file_read".into(),
                input: serde_json::json!({"path": "/etc/hosts"}),
                display_summary: None,
            }),
            (
                "the answer started",
                FrontendEvent::AssistantTextDelta("the answer".into()),
            ),
        ] {
            let console = console();
            let frontend = showing_thinking(Arc::clone(&console));
            frontend
                .emit(FrontendEvent::ThinkingDelta("weighing it".into()))
                .await;
            assert!(
                with_console(&console, |console| console.has_open_thinking()),
                "a thinking delta opens a block",
            );
            frontend.emit(closer).await;
            assert!(
                !with_console(&console, |console| console.has_open_thinking()),
                "the block outlived its end: {name}",
            );
        }
    }

    /// The two streamed kinds share one slot, so reasoning arriving mid-answer has to close the
    /// answer rather than append to it. Without that the reasoning would be rendered by the
    /// answer's renderer, onto stdout, inside the paragraph it interrupted.
    #[tokio::test]
    async fn reasoning_and_the_answer_never_stream_at_once() {
        let console = console();
        let frontend = showing_thinking(Arc::clone(&console));

        frontend
            .emit(FrontendEvent::AssistantTextDelta("part one".into()))
            .await;
        assert!(with_console(&console, |console| console.has_open_text()));

        frontend
            .emit(FrontendEvent::ThinkingDelta("second thoughts".into()))
            .await;
        assert!(with_console(&console, |console| console.has_open_thinking()));
        assert!(
            !with_console(&console, |console| console.has_open_text()),
            "the answer must be flushed before reasoning prints over it",
        );

        frontend
            .emit(FrontendEvent::AssistantTextDelta("part two".into()))
            .await;
        assert!(with_console(&console, |console| console.has_open_text()));
        assert!(!with_console(&console, |console| console.has_open_thinking()));
    }

    /// A thinking block that ends with nothing readable must not end the *answer* with it.
    ///
    /// `ThinkingEnded` is not a failure signal: it fires whenever a block completes carrying no
    /// visible text, which is every block under sealed reasoning and under Claude's
    /// `redact-thinking` or display updates. Those turns stream their answer through the same one
    /// slot, so closing it indiscriminately cuts a paragraph in two on the stream a caller
    /// pipes.
    #[tokio::test]
    async fn a_silent_thinking_block_does_not_close_the_answer() {
        for ender in [FrontendEvent::ThinkingEnded, FrontendEvent::ThinkingBlock {
            content: "reasoning".into(),
        }] {
            let console = console();
            let frontend = showing_thinking(Arc::clone(&console));
            frontend
                .emit(FrontendEvent::AssistantTextDelta("part one, ".into()))
                .await;
            assert!(with_console(&console, |console| console.has_open_text()));
            frontend.emit(ender).await;
            assert!(
                with_console(&console, |console| console.has_open_text()),
                "the answer's run was closed by a thinking event",
            );
        }
    }

    /// The default. Deltas are ignored outright and the one-line preview arrives with the block, so
    /// nothing streams and the console never opens a thinking block at all.
    #[tokio::test]
    async fn without_show_content_a_thinking_delta_prints_nothing() {
        let console = console();
        let frontend = frontend_on(Arc::clone(&console));
        frontend
            .emit(FrontendEvent::ThinkingDelta("weighing it".into()))
            .await;
        assert!(
            !with_console(&console, |console| console.has_open_thinking()),
            "the preview is rendered from the block, not streamed",
        );
    }

    /// `Agent::run_turn` emits `TurnFinished` only when the turn succeeded, so an interrupt or a
    /// provider error leaves the text block holding whatever arrived first. Ending the *episode* is
    /// what flushes it, which is what puts it under the turn it belongs to. Flushed by the next
    /// turn's `TurnStarted`, it prints under the following prompt as though the model had said it
    /// in answer to something else.
    #[tokio::test]
    async fn a_failed_turn_flushes_its_partial_answer_when_the_episode_ends() {
        let console = console();
        let frontend = frontend_on(Arc::clone(&console));
        frontend
            .emit(FrontendEvent::AssistantTextDelta(
                "abandoned partial".into(),
            ))
            .await;
        assert!(
            with_console(&console, |console| console.has_open_text()),
            "a text delta opens a block"
        );

        // No `TurnFinished`: this is the failed turn. The run's output still ends.
        with_console(&console, |console| console.finish());

        assert!(
            !with_console(&console, |console| console.has_open_text()),
            "the run must not leave a block open under the shell's prompt"
        );
    }

    /// Nothing here can show a prompt, so an elicitation declined for want of one has to be said
    /// on the console, where the rest of the run's chrome is, rather than logged where the default
    /// `warn` floor may or may not show it. The tool call waiting on the answer then fails, and
    /// this is the only thing that names the cause. Asserted on the console rather than on the
    /// return value, which no assertion could catch: `Decline` is correct either way.
    #[tokio::test]
    async fn an_elicitation_nobody_can_answer_is_declined_on_the_console() {
        let console = console();
        with_console(&console, |console| {
            console.start(crate::console::RowState::Empty)
        });
        let response = frontend_on(Arc::clone(&console))
            .handle_elicitation(crate::frontend::ElicitationPrompt {
                server_name: "notion".to_string(),
                message: "authorize?".to_string(),
                kind: crate::frontend::ElicitationKind::Url {
                    url: "https://example.com/".to_string(),
                },
            })
            .await;
        assert!(matches!(
            response,
            crate::frontend::ElicitationResponse::Decline
        ));
        assert!(
            with_console(&console, |console| console.has_printed()),
            "the decline must be said where the run's other chrome is"
        );
    }

    /// The same for the tool-approval half: a one-shot run with approvals on refuses every tool
    /// that needs approval, and without a line per refusal the run is indistinguishable from a
    /// model that simply chose not to use its tools.
    ///
    /// `Deny`, not `Canceled`: nothing stopped the turn, the call was refused, and the tool result
    /// the model reads should say which.
    #[tokio::test]
    async fn a_tool_nobody_can_approve_is_denied_and_said_on_the_console() {
        let console = console();
        with_console(&console, |console| {
            console.start(crate::console::RowState::Empty)
        });
        let outcome = frontend_on(Arc::clone(&console))
            .request_permission(crate::frontend::PermissionRequest {
                tool_name: "shell_execute".to_string(),
                primary_param: Some("rm -rf /".to_string()),
                input: serde_json::json!({"command": "rm -rf /"}),
                cancellation: tokio_util::sync::CancellationToken::new(),
                subagent_id: None,
            })
            .await;
        assert_eq!(outcome, PermissionOutcome::Deny);
        assert!(
            with_console(&console, |console| console.has_printed()),
            "the refusal must be said where the run's other chrome is"
        );
    }

    /// The happy path still closes on `TurnFinished`, so a completed turn doesn't hold its last
    /// paragraph until the following prompt.
    #[tokio::test]
    async fn turn_finished_closes_the_text_block() {
        let console = console();
        let frontend = frontend_on(Arc::clone(&console));
        frontend
            .emit(FrontendEvent::AssistantTextDelta("done".into()))
            .await;
        frontend.emit(FrontendEvent::TurnFinished).await;

        assert!(!with_console(&console, |console| console.has_open_text()));
    }

    /// Pins the events whose action is *not* the common one, plus a sample of those that are.
    ///
    /// Completeness is the compiler's job, not this test's: `indicator_action` matches every
    /// variant explicitly, so a new one fails to build until somebody chooses. What is left here is
    /// the part a type cannot state -- that the three exceptions are the exceptions.
    #[test]
    fn the_indicator_exceptions_are_the_exceptions() {
        use crate::frontend::FrontendEvent as E;

        let cases: Vec<(E, IndicatorAction)> = vec![
            (
                E::ThinkingProgress {
                    estimated_tokens: Some(50),
                },
                IndicatorAction::Keep,
            ),
            (
                E::ThinkingDelta("reasoning".to_string()),
                IndicatorAction::Erase,
            ),
            (
                E::ThinkingBlock {
                    content: "reasoning".to_string(),
                },
                IndicatorAction::Erase,
            ),
            (
                E::TurnStarted {
                    turn_id: uuid::Uuid::nil(),
                },
                IndicatorAction::Drop,
            ),
            // The rest close the phase out: the indicator is the only record of it.
            (E::ThinkingEnded, IndicatorAction::Commit),
            (E::TurnFinished, IndicatorAction::Commit),
            (
                E::SessionStarted {
                    id: uuid::Uuid::nil(),
                },
                IndicatorAction::Commit,
            ),
            (
                E::AssistantTextDelta("hi".to_string()),
                IndicatorAction::Commit,
            ),
            (
                E::ToolCallComposing {
                    id: "1".to_string(),
                    name: "file_read".to_string(),
                },
                IndicatorAction::Commit,
            ),
            (
                E::ToolCallStarted {
                    id: "1".to_string(),
                    name: "file_read".to_string(),
                    input: serde_json::json!({}),
                    display_summary: None,
                },
                IndicatorAction::Commit,
            ),
            (
                E::ToolCallCompleted {
                    id: "1".to_string(),
                    name: "file_read".to_string(),
                    is_error: false,
                    content: Vec::new(),
                    metadata: None,
                },
                IndicatorAction::Commit,
            ),
            (
                E::ToolCallOutputDelta {
                    id: "1".to_string(),
                    chunk: String::new(),
                },
                IndicatorAction::Commit,
            ),
            (
                E::ChecklistUpdated { items: Vec::new() },
                IndicatorAction::Commit,
            ),
            (
                E::SubAgentActivity {
                    tool_call_id: "1".to_string(),
                    summary: String::new(),
                },
                IndicatorAction::Commit,
            ),
            (
                E::TokenUsage(crate::stats::TokenUsage::default()),
                IndicatorAction::Commit,
            ),
        ];

        for (event, expected) in &cases {
            assert_eq!(
                indicator_action(event, true),
                *expected,
                "unexpected indicator action for {event:?}",
            );
        }

        // The one event whose answer depends on the setting. Under the default the deltas render
        // nothing, so taking the counter down would replace it with an empty row -- and since an
        // estimate and a delta arrive from one wire event under the token-count beta, the next
        // estimate would reopen the indicator instead of redrawing it, losing the peak.
        assert_eq!(
            indicator_action(&E::ThinkingDelta("reasoning".to_string()), false),
            IndicatorAction::Keep,
        );
    }
    /// The server's estimate is not monotonic -- a single thinking block was observed reporting
    /// 100, then 150, then 100 again -- so drawing it raw makes the counter appear to count down.
    /// Real thinking spend only accumulates, so the indicator holds the peak.
    #[test]
    fn thinking_estimate_never_runs_backwards() {
        let mut shown = None;
        for reported in [Some(100), Some(150), Some(100), Some(150), Some(100)] {
            let next = peak_estimate(shown, reported);
            assert!(
                next >= shown,
                "the drawn figure fell from {shown:?} to {next:?} on a reported {reported:?}",
            );
            shown = next;
        }
        assert_eq!(shown, Some(150), "the peak is what stays on screen");
    }
    /// A `None` marks a new block opening, and a new block is a new count: carrying the previous
    /// block's peak forward would credit this block with thinking it has not done.
    #[test]
    fn a_new_thinking_block_restarts_the_estimate() {
        let carried = peak_estimate(Some(900), None);
        assert_eq!(carried, None, "a block opening resets rather than inherits");
        assert_eq!(peak_estimate(carried, Some(50)), Some(50));
    }

    #[test]
    fn format_progress_update_strips_ansi_escapes() {
        let update = crate::frontend::ProgressUpdate {
            server_name: "svr".to_string(),
            tool_name: "tool".to_string(),
            tool_use_id: None,
            message: Some("\x1b[2Jspoofed\x1b[H".to_string()),
            progress: 1.0,
            total: Some(4.0),
        };
        let line = format_progress_update(&update);
        assert!(
            !line.contains('\x1b'),
            "ANSI escape leaked into progress line: {line:?}"
        );
        assert!(line.contains("spoofed"));
        assert!(line.contains("[mcp:svr/tool]"));
    }

    /// The progress line opens with meka's own `\r` to overwrite the previous one. A newline in the
    /// server's message would therefore commit that row and start painting at column zero on a
    /// fresh line, below chrome the user has already read: a forged approval block needing no
    /// escape sequence at all.
    #[test]
    fn a_progress_message_cannot_open_a_second_row() {
        let update = crate::frontend::ProgressUpdate {
            server_name: "svr".to_string(),
            tool_name: "tool".to_string(),
            tool_use_id: None,
            message: Some(
                "working\n[approval] shell_execute\n  command: ls -la\nAllow? (Y/n) ".to_string(),
            ),
            progress: 1.0,
            total: None,
        };

        let line = format_progress_update(&update);

        // One leading `\r` (meka's own) and no other line break of any kind.
        assert_eq!(line.matches('\r').count(), 1, "{line:?}");
        assert!(line.starts_with('\r'), "{line:?}");
        assert!(!line.contains('\n'), "{line:?}");
    }

    /// A server that pads its message must not be able to scroll the transcript by writing a line
    /// longer than the terminal.
    #[test]
    fn a_progress_message_is_bounded_by_the_terminal_width() {
        let update = crate::frontend::ProgressUpdate {
            server_name: "svr".to_string(),
            tool_name: "tool".to_string(),
            tool_use_id: None,
            message: Some("x".repeat(10_000)),
            progress: 1.0,
            total: None,
        };

        let line = format_progress_update(&update);
        let visible = line.trim_start_matches('\r').chars().count();
        assert!(
            visible <= crate::render::output_width(),
            "progress line ran to {visible} columns: {line:?}"
        );
    }

    /// Server and tool names are normalized at registration; the line drops the bidi and
    /// zero-width characters itself so a handler that someday forgets cannot be used to spoof it.
    #[test]
    fn format_progress_update_strips_rtl_override_in_names() {
        let update = crate::frontend::ProgressUpdate {
            server_name: "sv\u{202E}r".to_string(),
            tool_name: "t\u{200B}ool".to_string(),
            tool_use_id: None,
            message: None,
            progress: 0.5,
            total: None,
        };
        let line = format_progress_update(&update);
        assert!(!line.contains('\u{202E}'));
        assert!(!line.contains('\u{200B}'));
    }
}
