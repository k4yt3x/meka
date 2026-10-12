//! The terminal of a one-shot run: the one writer for everything it prints, and the one rule for
//! the space between two things it prints.
//!
//! Every block (the answer, a thinking block, a tool indicator, a notice, the session id) is set
//! off from the one before it by a single blank line, and the first prints flush under the
//! shell's command line. [`RowState`] exists because a blank line only makes a gap when the
//! cursor is at column zero: two writers park it mid-row deliberately (the thinking indicator and
//! the MCP progress line), and whatever prints next erases that row first. The blank above an
//! erased row stays where it is, so the row's replacement owes none.

use std::io::Write;

use crate::{
    config::{RenderMode, ToolParams},
    render,
    render::StreamingRenderer,
};

/// What occupies the row the cursor is sitting on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowState {
    /// Column zero of a row nothing has written to. Anything may print immediately.
    Empty,
    /// An in-place status line the writer expects to overwrite or erase: the thinking indicator, or
    /// an MCP progress line whose text comes from the server. Neither is content, so meka's own
    /// output replaces it rather than following it.
    Transient,
}

/// Something that happens to the console, as a value, so the decision it forces can be tested
/// without a terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Action {
    /// A `tracing` line is about to print. It takes a row of its own, because the row it would land
    /// on may be a status line about to be erased.
    AnnounceLog,
    /// A block is about to print.
    Block,
    /// An in-place status line is about to be drawn.
    OpenTransient,
    /// Keep the transient line by writing the newline its writer withheld.
    CommitTransient,
    /// Discard the transient line, because something is about to render in its place.
    EraseTransient,
}

/// What has to happen to the current row before anything else prints on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Settle {
    Nothing,
    /// End the row, keeping what is on it.
    Terminate,
    /// Blank the row and return to column zero, discarding what is on it.
    Erase,
}

/// Everything an action writes before its own content, and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Emit {
    pub(crate) settle: Settle,
    /// The blank that sets a block off from the one above it.
    pub(crate) separator_blank: bool,
}

/// The console's state, separated from the console so [`step`] can be a pure function of it.
#[derive(Clone, Copy)]
pub(crate) struct State {
    pub(crate) row: RowState,
    /// Whether a block has printed and nothing has yet been set off from it: the next block owes
    /// a blank above. Cleared when the blank is spent on a transient row, which keeps it when the
    /// row is erased, so its replacement does not add a second.
    owes_separator: bool,
}

impl State {
    fn new(row: RowState) -> Self {
        Self {
            row,
            owes_separator: false,
        }
    }
}

/// The whole decision, as a pure function.
///
/// Every blank line a one-shot run prints is the one in [`Emit`], and this is the only place it is
/// decided. Split out from the printing because, in a dispatch that mixes the two, the only way to
/// see a wrong answer is to run a terminal and look at it.
pub(crate) fn step(state: State, action: Action) -> (Emit, State) {
    let mut next = state;
    match action {
        Action::AnnounceLog => {
            let settle = settle_row(&mut next);
            (
                Emit {
                    settle,
                    separator_blank: false,
                },
                next,
            )
        }
        Action::Block => {
            let settle = settle_row(&mut next);
            let separator_blank = state.owes_separator;
            next.owes_separator = true;
            (
                Emit {
                    settle,
                    separator_blank,
                },
                next,
            )
        }
        Action::OpenTransient => {
            // A redraw overwrites its own row, so nothing has to be moved past; the first draw
            // takes the blank the block above is owed, and the row is not a block until it is
            // committed.
            let separator_blank = state.owes_separator;
            next.owes_separator = false;
            next.row = RowState::Transient;
            (
                Emit {
                    settle: Settle::Nothing,
                    separator_blank,
                },
                next,
            )
        }
        Action::CommitTransient => {
            let settle = if state.row == RowState::Transient {
                Settle::Terminate
            } else {
                Settle::Nothing
            };
            next.row = RowState::Empty;
            next.owes_separator = true;
            (
                Emit {
                    settle,
                    separator_blank: false,
                },
                next,
            )
        }
        Action::EraseTransient => {
            let settle = if state.row == RowState::Transient {
                Settle::Erase
            } else {
                Settle::Nothing
            };
            next.row = RowState::Empty;
            (
                Emit {
                    settle,
                    separator_blank: false,
                },
                next,
            )
        }
    }
}

/// Settle the row so the next thing printed starts at column zero of a row of its own.
fn settle_row(next: &mut State) -> Settle {
    let settle = match next.row {
        RowState::Empty => Settle::Nothing,
        RowState::Transient => Settle::Erase,
    };
    next.row = RowState::Empty;
    settle
}

/// A streamed block in progress, and which kind it is.
///
/// The kind is what decides whether an arriving delta continues this block or ends it, and whether
/// closing it owes stdout a flush.
struct OpenStream {
    kind: StreamKind,
    renderer: StreamingRenderer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamKind {
    /// The answer, on stdout.
    Text,
    /// Reasoning, on stderr, behind `Thinking... `.
    Thinking,
}

/// What a failed write of `kind` lost, for the report.
///
/// Named per kind rather than once, because the two are not the same loss: an answer that did not
/// arrive is what a scripted host fails on, and reasoning that did not is chrome. The `Thinking`
/// arm is unreachable while its sink hands nothing back, and is named anyway so the message cannot
/// drift from the stream it describes.
fn lost_output_of(kind: StreamKind) -> &'static str {
    match kind {
        StreamKind::Text => "an answer did not reach stdout",
        StreamKind::Thinking => "a thinking block did not reach the terminal",
    }
}

/// Whether a thinking indicator may be drawn over what the console is already showing.
///
/// Split from the drawing for the same reason [`step`] is: with a terminal in the way, a wrong
/// answer here is only visible by running one and looking at it.
///
/// The block's own text is a better progress signal than a running count of it, and the two cannot
/// share the row: drawing the indicator closes the streamed block to make room, so a later estimate
/// cuts the block in half and labels the remainder as a new one. The Claude providers send an
/// estimate and a text delta from a single wire event whenever the token-count beta meets visible
/// thinking, which makes that one cut per delta.
fn indicator_may_draw(open: Option<StreamKind>) -> bool {
    !matches!(open, Some(StreamKind::Thinking))
}

/// The single writer for everything a one-shot run prints.
///
/// Shared by the host and the agent's frontend task, which also gives the two threads that write
/// to the terminal one lock to contend on rather than none.
pub(crate) struct Console {
    state: State,
    render_mode: RenderMode,
    /// Open across consecutive deltas of one kind; closed by any other block, and by the end of
    /// the run.
    ///
    /// One slot, not one per kind. The two alternate but never overlap -- a provider streams one
    /// block at a time -- so a second slot could only ever hold a block nothing had closed, and
    /// the whole point of closing is that the renderer holds a trailing paragraph back until a
    /// blank line settles it, which would then surface under whatever printed next.
    stream: Option<OpenStream>,
    /// The first write failure stdout reported.
    ///
    /// Kept rather than discarded because a host that scripts its output has to be able to fail
    /// on it, which is what [`Self::take_lost_output`] is for. Saying so is
    /// [`render::report_lost_output`]'s job, and it says it once per process.
    lost_output: Option<std::io::Error>,
}

impl Console {
    pub(crate) fn new(render_mode: RenderMode) -> Self {
        Self {
            state: State::new(RowState::Empty),
            render_mode,
            stream: None,
            lost_output: None,
        }
    }

    /// Name a write that did not reach stdout, and keep it for the caller.
    fn lost_output(&mut self, what: &str, error: std::io::Error) {
        render::report_lost_output(what, &error);
        // A reader that hung up chose to stop reading, so it is not a failure of the run. Anything
        // else lost the answer to something nobody chose, and a host that is being scripted has to
        // be able to say so in its exit code.
        if error.kind() != std::io::ErrorKind::BrokenPipe && self.lost_output.is_none() {
            self.lost_output = Some(error);
        }
    }

    /// Take the failure that cost this run its output, if one did.
    pub(crate) fn take_lost_output(&mut self) -> Option<std::io::Error> {
        self.lost_output.take()
    }

    fn act(&mut self, action: Action) {
        // A streaming block spends its label ahead of the text it holds, so between two deltas
        // its cursor may sit mid-row. `step` does not model that row, and does not need to: it is
        // ended here, ahead of every action, so nothing that prints can land on it. Ended by the
        // renderer rather than with a newline of this module's own, because the block resumes
        // below behind its label only if its lead saw the interruption.
        let ended = self
            .stream
            .as_mut()
            .map(|open| (open.kind, open.renderer.interrupt()));
        if let Some((kind, Err(error))) = ended {
            self.lost_output(lost_output_of(kind), error);
        }
        let (emit, next) = step(self.state, action);
        self.state = next;
        match emit.settle {
            Settle::Nothing => {}
            Settle::Terminate => crate::streams::write_stderr_line(""),
            Settle::Erase => render::begin_own_line(),
        }
        if emit.separator_blank {
            crate::streams::write_stderr_line("");
        }
    }

    /// Begin the run's output, given what is on the row the cursor sits on.
    pub(crate) fn start(&mut self, row: RowState) {
        // A new run is a new chance to say that output is not arriving.
        render::reset_lost_output_report();
        self.state = State::new(row);
    }

    /// End the run's output: flush the open block, and erase a status line left drawn, which is
    /// not content and must not survive into the shell's prompt.
    pub(crate) fn finish(&mut self) {
        self.close_stream();
        self.act(Action::EraseTransient);
    }

    /// Declare that a `tracing` line is about to print.
    ///
    /// Called by the relay for every line it writes to stderr off-prompt. See
    /// [`Action::AnnounceLog`] for what a log line does.
    pub(crate) fn announce_log_line(&mut self) {
        self.act(Action::AnnounceLog);
    }

    /// A [`crate::frontend::Notice`], in the color its level asks for: dim for `Info`, the warning
    /// color for `Warn`.
    pub(crate) fn notice(&mut self, notice: &crate::frontend::Notice) {
        self.close_stream();
        self.act(Action::Block);
        if notice.kind == crate::frontend::NoticeKind::StageDirection {
            render::render_annotation(&notice.text);
            return;
        }
        match notice.level {
            crate::frontend::NoticeLevel::Info => render::render_hint(&notice.text),
            crate::frontend::NoticeLevel::Warn => render::render_warning(&notice.text),
        }
    }

    pub(crate) fn session_id(&mut self, label: &str, id: &str) {
        self.close_stream();
        self.act(Action::Block);
        render::render_session_id(label, id);
    }

    /// A stage direction about the output: `(interrupted)`. See [`render::render_annotation`].
    pub(crate) fn annotation(&mut self, note: &str) {
        self.close_stream();
        self.act(Action::Block);
        render::render_annotation(note);
    }

    /// Print through a closure, for a caller whose painter takes arguments this module has no
    /// reason to model.
    pub(crate) fn chrome(&mut self, paint: impl FnOnce()) {
        self.close_stream();
        self.act(Action::Block);
        paint();
    }

    pub(crate) fn tool_indicator(
        &mut self,
        name: &str,
        input: &serde_json::Value,
        display_summary: Option<&str>,
        params: ToolParams,
    ) {
        self.close_stream();
        self.act(Action::Block);
        render::render_tool_indicator(name, input, display_summary, params);
    }

    /// The one dimmed line shown for a thinking block under `show_content = false`.
    pub(crate) fn thinking_preview(&mut self, content: &str) {
        self.close_stream();
        self.act(Action::Block);
        render::render_thinking_preview(content);
    }

    /// The list as it stands after a change, as a block of its own. An emptied list prints
    /// nothing: the indicator of the edit that emptied it already says so, and a heading over
    /// nothing would say less.
    pub(crate) fn checklist(&mut self, items: &[crate::checklist::ChecklistItem]) {
        if items.is_empty() {
            return;
        }
        self.close_stream();
        self.act(Action::Block);
        render::render_checklist(items);
    }

    pub(crate) fn token_usage(&mut self, usage: &crate::stats::TokenUsage) {
        self.close_stream();
        self.act(Action::Block);
        render::render_token_usage(usage);
    }

    /// Draw an in-place status line, returning whether anything reached the terminal.
    ///
    /// Off a tty this returns before the action, where redrawing in place would instead accumulate
    /// one line per update. Nothing is spent: an indicator that cannot be drawn must not take the
    /// blank above it or move the row, or it shifts the layout of the output that *is* produced.
    ///
    /// A `draw` that answers false *past* that gate has already spent whatever blank was owed, so
    /// only the row is restored: the line may be partly on screen. No caller does today -- both
    /// write through helpers that accept a failed write -- but the contract is the one a fallible
    /// drawer would need.
    pub(crate) fn transient(&mut self, draw: impl FnOnce() -> bool) -> bool {
        if !render::live_indicator_supported() {
            return false;
        }
        self.act(Action::OpenTransient);
        let drawn = draw();
        if !drawn {
            self.state.row = RowState::Empty;
        }
        drawn
    }

    /// Draw the thinking indicator, returning whether it reached the terminal.
    ///
    /// `opening` is the first draw of a block, which closes whatever was streaming: the indicator
    /// draws from column zero, and an open text run would have its last line overwritten, damage
    /// that stays on screen since the indicator is kept rather than erased. Redraws overwrite
    /// their own row and close nothing.
    pub(crate) fn thinking_indicator(&mut self, opening: bool, estimate: Option<u64>) -> bool {
        if !indicator_may_draw(self.open_stream_kind()) {
            return false;
        }
        if !render::live_indicator_supported() {
            return false;
        }
        if opening {
            self.close_stream();
        }
        self.act(Action::OpenTransient);
        let drawn = render::render_thinking_indicator(estimate);
        if !drawn {
            self.state.row = RowState::Empty;
        }
        drawn
    }

    /// Keep the transient line by writing the newline its writer withheld.
    pub(crate) fn commit_transient(&mut self) {
        self.act(Action::CommitTransient);
    }

    /// Discard the transient line, for the one case where something is about to render in its
    /// place.
    pub(crate) fn erase_transient(&mut self) {
        self.act(Action::EraseTransient);
    }

    /// Which kind of block is streaming, if one is.
    fn open_stream_kind(&self) -> Option<StreamKind> {
        self.stream.as_ref().map(|open| open.kind)
    }

    /// Whether a text block is still open, for the frontend's tests: a run left open past the end
    /// of the run is what flushes a failed turn's tail under the shell's prompt.
    #[cfg(test)]
    pub(crate) fn has_open_text(&self) -> bool {
        self.open_stream_kind() == Some(StreamKind::Text)
    }

    /// The same question for reasoning. Separate rather than one predicate over both, because the
    /// property under test is that the two never overlap.
    #[cfg(test)]
    pub(crate) fn has_open_thinking(&self) -> bool {
        self.open_stream_kind() == Some(StreamKind::Thinking)
    }

    /// Whether a streaming block has left the cursor mid-row, for the tests of the door that ends
    /// it.
    #[cfg(test)]
    pub(crate) fn has_open_row(&self) -> bool {
        self.stream
            .as_ref()
            .is_some_and(|open| open.renderer.is_row_open())
    }

    /// Whether a block has printed, for the frontend's tests: the one bit of console state a
    /// caller outside this module can observe without a terminal.
    #[cfg(test)]
    pub(crate) fn has_printed(&self) -> bool {
        self.state.owes_separator
    }

    /// Put the row into a state a test cannot reach through the drawing API.
    ///
    /// `transient` and `thinking_indicator` both return early unless
    /// `render::live_indicator_supported()`, which is false without a terminal, so the row a
    /// mid-turn log line actually collides with is otherwise unreachable under `cargo test`.
    #[cfg(test)]
    pub(crate) fn force_row(&mut self, row: RowState) {
        self.state.row = row;
    }

    /// What the cursor is sitting on, for the same tests.
    #[cfg(test)]
    pub(crate) fn row(&self) -> RowState {
        self.state.row
    }

    pub(crate) fn text_delta(&mut self, text: &str) {
        self.stream_delta(StreamKind::Text, text);
    }

    /// A chunk of reasoning, shown as it arrives under `show_content = true`.
    pub(crate) fn thinking_delta(&mut self, text: &str) {
        self.stream_delta(StreamKind::Thinking, text);
    }

    /// Append to the open block of `kind`, opening one and closing any block of the other kind
    /// first.
    fn stream_delta(&mut self, kind: StreamKind, text: &str) {
        if self.stream.as_ref().is_some_and(|open| open.kind != kind) {
            self.close_stream();
        }
        if self.stream.is_none() {
            self.act(Action::Block);
            self.stream = Some(OpenStream {
                kind,
                renderer: match kind {
                    StreamKind::Text => StreamingRenderer::new(self.render_mode),
                    StreamKind::Thinking => StreamingRenderer::for_thinking(self.render_mode),
                },
            });
        }
        let failure = self
            .stream
            .as_mut()
            .and_then(|open| open.renderer.push_delta(text).err());
        if let Some(error) = failure {
            self.lost_output(lost_output_of(kind), error);
        }
    }

    /// Close a streamed *thinking* block, leaving an open answer alone.
    ///
    /// The thinking events fire on turns that stream no reasoning at all: a block completing with
    /// nothing readable emits `ThinkingEnded`, which is the ordinary shape under sealed reasoning
    /// and under Claude's `redact-thinking` or display updates. Closing indiscriminately there ends
    /// the *answer's* run mid-paragraph, and the next delta opens a second one -- splitting one
    /// paragraph across two on the stream a caller pipes.
    pub(crate) fn close_thinking(&mut self) {
        if self.open_stream_kind() == Some(StreamKind::Thinking) {
            self.close_stream();
        }
    }

    /// Flush and drop any open streamed block, so block types don't interleave.
    pub(crate) fn close_stream(&mut self) {
        let Some(mut open) = self.stream.take() else {
            return;
        };
        if let Err(error) = open.renderer.finish() {
            self.lost_output(lost_output_of(open.kind), error);
        }
        // Only the answer is owed this, and only the answer can report a failure: a thinking block
        // writes to stderr, where a report would go down the stream that just refused it.
        if open.kind != StreamKind::Text {
            return;
        }
        // Redundant on every path `finish` completes, which ends in a flush of its own, and kept
        // for the one it does not: a write that failed part-way leaves `finish` returning early
        // through `?`, and the assistant's text goes to stdout while the closing blank and the next
        // prompt go to stderr, so a paragraph left in the line buffer would surface underneath
        // them.
        if let Err(error) = std::io::stdout().flush() {
            self.lost_output("an answer did not reach stdout", error);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Replay a sequence from a fresh console, returning what each action emitted.
    fn run(actions: &[Action]) -> Vec<Emit> {
        let mut state = State::new(RowState::Empty);
        actions
            .iter()
            .map(|action| {
                let (emit, next) = step(state, *action);
                state = next;
                emit
            })
            .collect()
    }

    /// The first block prints flush under the shell's command line; every later one is set off
    /// from the one above by one blank, whatever kind either is.
    #[test]
    fn every_block_after_the_first_is_set_off_by_one_blank() {
        let emits = run(&[Action::Block, Action::Block, Action::Block]);
        assert!(!emits[0].separator_blank, "nothing above the first");
        assert!(emits[1].separator_blank);
        assert!(emits[2].separator_blank);
        assert!(
            emits.iter().all(|emit| emit.settle == Settle::Nothing),
            "an empty row needs no settling"
        );
    }

    /// A blank line only makes a gap from column zero. An in-place status line is discarded rather
    /// than terminated, because its row is not content, and the blank spent above it serves the
    /// block that replaces it.
    #[test]
    fn a_transient_row_is_erased_by_whatever_prints_next_and_its_blank_is_kept() {
        let emits = run(&[
            Action::Block,
            Action::OpenTransient,
            Action::OpenTransient,
            Action::Block,
        ]);
        assert!(
            emits[1].separator_blank,
            "the first draw is set off from the block above"
        );
        assert!(!emits[2].separator_blank, "a redraw overwrites its own row");
        assert_eq!(emits[3].settle, Settle::Erase);
        assert!(
            !emits[3].separator_blank,
            "the blank above the erased row already sets the block off"
        );
        let erased = run(&[
            Action::Block,
            Action::OpenTransient,
            Action::EraseTransient,
            Action::Block,
        ]);
        assert_eq!(erased[2].settle, Settle::Erase);
        assert!(!erased[3].separator_blank);
    }

    /// A committed status line is a block: the next one is set off from it.
    #[test]
    fn committing_a_transient_line_keeps_it_and_sets_the_next_block_off() {
        let emits = run(&[
            Action::OpenTransient,
            Action::CommitTransient,
            Action::Block,
        ]);
        assert_eq!(emits[1].settle, Settle::Terminate);
        assert!(emits[2].separator_blank);
        let nothing = run(&[Action::CommitTransient, Action::EraseTransient]);
        assert!(
            nothing.iter().all(|emit| emit.settle == Settle::Nothing),
            "with no transient row drawn there is nothing to keep or discard"
        );
    }

    /// A log line takes a row of its own: the row it would land on may be a status line about to
    /// be erased. It is not a block, so it spends no blank.
    #[test]
    fn a_log_line_settles_its_row_and_spends_no_blank() {
        let emits = run(&[Action::Block, Action::OpenTransient, Action::AnnounceLog]);
        assert_eq!(emits[2].settle, Settle::Erase);
        assert!(!emits[2].separator_blank);
        let flush = run(&[Action::AnnounceLog, Action::Block]);
        assert!(
            !flush[1].separator_blank,
            "a warning at startup earns the first block no blank"
        );
    }

    /// A log line arriving while a thinking block holds its first paragraph lands under the label
    /// rather than on its row, and the block stays open to continue below it.
    #[test]
    fn a_log_line_ends_the_row_a_streaming_block_left_open() {
        let mut console = Console::new(RenderMode::Raw);
        console.start(RowState::Empty);
        console.thinking_delta("partial");
        assert!(
            console.has_open_row(),
            "the label leaves the cursor mid-row"
        );
        console.announce_log_line();
        assert!(
            !console.has_open_row(),
            "the row is ended before the foreign line"
        );
        assert!(
            console.has_open_thinking(),
            "and the block is still open to continue"
        );
    }

    /// Finishing closes the open block and erases a status line left drawn, which must not
    /// survive into the shell's prompt; finishing twice is finishing once.
    #[test]
    fn finishing_closes_the_block_and_erases_a_leftover_status_line() {
        let mut console = Console::new(RenderMode::Raw);
        console.start(RowState::Empty);
        console.text_delta("half an answer");
        console.force_row(RowState::Transient);
        console.finish();
        assert!(!console.has_open_text());
        assert_eq!(console.row(), RowState::Empty);
        console.finish();
        assert_eq!(console.row(), RowState::Empty);
    }

    /// Claude sends an estimate and a text delta from one wire event once the token-count beta
    /// meets visible thinking, so the two arrive alternately for the whole block. Redrawing the
    /// counter over streamed reasoning closes the block to make room, and the next delta opens a
    /// fresh one behind a second `Thinking... ` label -- once per delta, which shreds the block
    /// into one labeled fragment per chunk.
    #[test]
    fn a_progress_estimate_never_draws_over_reasoning_already_on_screen() {
        assert!(
            indicator_may_draw(None),
            "nothing is streaming, so the counter is the only signal there is",
        );
        assert!(
            indicator_may_draw(Some(StreamKind::Text)),
            "an answer streams to stdout; the indicator has its own row on stderr",
        );
        assert!(
            !indicator_may_draw(Some(StreamKind::Thinking)),
            "the block's own text is the progress, and drawing over it would close the block",
        );
    }
}
