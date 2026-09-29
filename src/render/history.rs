//! Replay of a conversation's history on resume: which turns to show and how each message is
//! rendered so a resumed transcript reads like the live turn did.

use super::*;

/// Walk backwards through `messages` and return the suffix that starts at the `n`th most recent
/// user turn. A "turn" begins at a User-role message whose content is not purely `ToolResult`
/// blocks, i.e. an actual user prompt, not an agent-driven tool result echoed back as a User
/// message. `n == 0` or no qualifying turns returns an empty slice.
pub(crate) fn last_n_turns(
    messages: &[crate::conversation::Message],
    n: usize,
) -> &[crate::conversation::Message] {
    if n == 0 || messages.is_empty() {
        return &[];
    }
    // Walk backwards, tracking the earliest qualifying boundary seen so far. If we hit `n`
    // boundaries we stop there; if we exhaust the slice without reaching `n`, we return everything
    // from the earliest boundary we did find (so `N=999` on a 2-turn session still returns both
    // turns, not an empty slice).
    let mut seen = 0usize;
    let mut earliest_boundary: Option<usize> = None;
    for (index, message) in messages.iter().enumerate().rev() {
        if is_user_prompt_boundary(message) {
            seen += 1;
            earliest_boundary = Some(index);
            if seen == n {
                break;
            }
        }
    }
    match earliest_boundary {
        Some(start) => &messages[start..],
        None => &[],
    }
}

/// True when `message` opens a turn from the user's perspective: `Message::opens_turn`, the one
/// spelling of the rule, so a steer that lands beside a round's results does not start a replay
/// in the middle of the turn it steered.
pub(super) fn is_user_prompt_boundary(message: &crate::conversation::Message) -> bool {
    message.opens_turn()
}

/// Knobs for [`render_message_history`]. Mirrors the fields the live REPL reads off
/// `ResolvedConfig` so resumed/dumped history matches what the user sees during a live turn.
pub(crate) struct HistoryRenderOptions {
    pub(crate) render_mode: RenderMode,
    pub(crate) show_thinking: bool,
    /// Mirrors `[display].tool_params`, so a replayed tool call carries the same detail the live
    /// one did.
    pub(crate) tool_params: ToolParams,
    pub(crate) input_style: nu_ansi_term::Style,
    /// Blank line before each user prompt (mirrors `[display].newline_before_prompt`).
    pub(crate) newline_before_prompt: bool,
    /// Blank line after each user prompt (mirrors `[display].newline_after_prompt`). Acts as the
    /// visual separator between the prompt and the agent's first response block.
    pub(crate) newline_after_prompt: bool,
}

/// Reprint a slice of historical messages styled to match the live REPL output. Inter-block spacing
/// flows through [`OutputSpacing`] (the same state machine the live loop uses) so transitions like
/// tool-indicator → text get a blank line; user-prompt spacing follows the `newline_before_prompt`
/// / `newline_after_prompt` config flags just like the live REPL.
///
/// `on_first_output` runs immediately before the first row this prints, and not at all for a slice
/// that renders to nothing, so the caller can announce the output to the console the moment it
/// starts: the blank above the history is then the console's to decide, as for the first output of
/// any episode. Returns whether anything reached the terminal. A slice can render to nothing (it is
/// empty, or it holds only tool results and blank text), and the caller has to know, or it brackets
/// a region with nothing in it.
pub(crate) fn render_message_history(
    messages: &[crate::conversation::Message],
    opts: &HistoryRenderOptions,
    on_first_output: impl FnOnce(),
) -> bool {
    use crate::conversation::{ContentBlock, Role};
    if messages.is_empty() {
        return false;
    }
    let mut spacing = OutputSpacing::new();
    // The blank above this history is the console's, spent when `on_first_output` announces the
    // first row. So the very first user prompt rendered skips its own `newline_before_prompt`, or
    // the two would stack. Once anything has been emitted, the inner spacing rules take over and
    // turn-to-turn transitions get their own blanks naturally.
    let mut emitted_any = false;
    let mut first_output = Some(on_first_output);
    for message in messages {
        for block in &message.content {
            match block {
                ContentBlock::Text { text } => match message.role {
                    Role::Assistant => {
                        if text.trim().is_empty() {
                            continue;
                        }
                        separate(spacing.before_text(), &mut first_output);
                        render_assistant_text(text, opts.render_mode);
                        emitted_any = true;
                    }
                    Role::User => {
                        // Decided here rather than in the renderer, because the announcement must
                        // not run for a prompt that prints nothing.
                        if text.trim().is_empty() {
                            continue;
                        }
                        let newline_before = match first_output.take() {
                            // The first row: the console has just decided the blank above it.
                            Some(announce) => {
                                announce();
                                false
                            }
                            None => opts.newline_before_prompt && emitted_any,
                        };
                        render_user_prompt(text, opts.input_style, newline_before);
                        if opts.newline_after_prompt {
                            write_stderr_line("");
                        }
                        spacing.after_prompt();
                        emitted_any = true;
                    }
                },
                // meka's own preamble for the turn; a replay shows what was typed.
                ContentBlock::TurnContext { .. } => {}
                // Input images (from an ACP client) have no terminal rendering; show a marker so a
                // replayed/exported transcript notes the attachment instead of dropping it
                // silently.
                ContentBlock::Image { .. } => {
                    separate(spacing.before_text(), &mut first_output);
                    write_stderr_line("[image]");
                    emitted_any = true;
                }
                ContentBlock::Thinking { thinking, .. } => {
                    if opts.show_thinking && !thinking.trim().is_empty() {
                        separate(spacing.before_thinking(), &mut first_output);
                        render_thinking_block(thinking, opts.render_mode);
                        emitted_any = true;
                    }
                }
                ContentBlock::RedactedThinking { .. } => {
                    if opts.show_thinking {
                        separate(spacing.before_thinking(), &mut first_output);
                        render_thinking_block(REDACTED_THINKING, opts.render_mode);
                        emitted_any = true;
                    }
                }
                ContentBlock::ToolUse { name, input, .. } => {
                    separate(
                        spacing.before_tool_indicator(opts.tool_params),
                        &mut first_output,
                    );
                    render_tool_indicator(name, input, None, opts.tool_params);
                    emitted_any = true;
                }
                // Tool results are intentionally hidden; the live REPL doesn't echo them either,
                // so showing them in history would be a fidelity regression. The user sees the tool
                // indicator (above) and whatever the assistant's next text block says about the
                // result.
                ContentBlock::ToolResult { .. } => {}
            }
        }
    }
    emitted_any
}

pub(super) fn render_assistant_text(text: &str, render_mode: RenderMode) {
    // Caller has already emitted the leading blank line (via `OutputSpacing::before_text`) when
    // needed, and verified the text is non-empty. We just stream the markdown, no trailing blank,
    // because the next block's `before_*` will add one if appropriate.
    let mut renderer = StreamingRenderer::new(render_mode);
    if let Err(error) = renderer.push_delta(text) {
        report_lost_output("a replayed message did not reach stdout", &error);
    }
    if let Err(error) = renderer.finish() {
        report_lost_output("a replayed message did not reach stdout", &error);
    }
}

/// Emit what separates a block from what precedes it: at the first row the caller's announcement,
/// which is where the console decides the blank; after that the spacing machine's separator.
///
/// The announcement is taken whether or not `needed` is set, so the first block spends it and no
/// later one can run it again. Left armed behind a block that asked for its own separator, it would
/// put the console's blank in the middle of the replayed history instead of above it.
pub(super) fn separate<F: FnOnce()>(needed: bool, first_output: &mut Option<F>) {
    if let Some(announce) = first_output.take() {
        announce();
    } else if needed {
        write_stderr_line("");
    }
}

/// Render a user prompt with the cyan `>` gutter plus `input_style` applied to each line,
/// optionally preceded by a blank line. The caller has already skipped a prompt that trims to
/// nothing.
pub(super) fn render_user_prompt(
    text: &str,
    input_style: nu_ansi_term::Style,
    newline_before: bool,
) {
    if newline_before {
        write_stderr_line("");
    }
    for line in text.trim().lines() {
        // Sanitized like the assistant text a few lines above. "User" here names the *role*, not
        // necessarily a person at this terminal: an ACP or HTTP client wrote it, or a `--skill`
        // body did, and a replayed session shows whatever the row holds. Leaving it raw made the
        // one message class meka replays without filtering the one an attacker controls end to end.
        write_stderr_line(format!(
            "{} {}",
            ">".with(Color::Cyan),
            input_style.paint(sanitize_stream_text(line))
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        conversation::{ContentBlock, Message, Role},
        render::tests::{assistant_text, tool_result_message, user_prompt},
    };

    #[test]
    fn is_user_prompt_boundary_classification() {
        assert!(is_user_prompt_boundary(&user_prompt("hi")));
        assert!(!is_user_prompt_boundary(&assistant_text("hi")));
        assert!(!is_user_prompt_boundary(&tool_result_message("u", "out")));

        // A steer rides the round's results in one user message; the turn it steered opened
        // earlier, so a replay that started here would begin mid-turn.
        let mixed = Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "u".to_string(),
                    content: vec![],
                    is_error: false,
                },
                ContentBlock::Text {
                    text: "follow-up".to_string(),
                },
            ],
        };
        assert!(!is_user_prompt_boundary(&mixed));
    }

    /// The announcement is spent by the first block to print, whether or not that block also asked
    /// for a separator of its own. Left armed there, it would run for a later block and put the
    /// console's blank in the middle of the replayed history instead of above it.
    #[test]
    fn separate_spends_the_announcement_even_when_the_block_asked_for_a_separator() {
        let announced = std::cell::Cell::new(0);
        let mut both = Some(|| announced.set(announced.get() + 1));
        separate(true, &mut both);
        assert!(
            both.is_none(),
            "a block with its own separator must still spend it"
        );
        assert_eq!(announced.get(), 1);

        let mut armed_only = Some(|| announced.set(announced.get() + 1));
        separate(false, &mut armed_only);
        assert!(armed_only.is_none());
        assert_eq!(announced.get(), 2);

        let mut spent: Option<fn()> = None;
        separate(false, &mut spent);
        assert!(spent.is_none());
        assert_eq!(announced.get(), 2, "a spent announcement never runs again");
    }
}
