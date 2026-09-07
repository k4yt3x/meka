//! Writing to the terminal: the two streams, the spacing state between writes, a reader that hung
//! up, and the styled one-line notices.

use super::*;
use crate::streams::{write_stderr, write_stderr_line};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LastOutput {
    Nothing,
    Prompt,
    Text,
    Thinking,
    ToolIndicator,
    TodoList,
}
/// Tracks what was last printed to decide if a blank line is needed next.
///
/// `Copy` so [`crate::console`] can run a transition against a scratch copy and return both the
/// blank line it implies and the state that follows, which is what lets the console's whole
/// decision be a pure function a test can enumerate.
#[derive(Clone, Copy)]
pub(crate) struct OutputSpacing {
    pub(super) last: LastOutput,
}
impl OutputSpacing {
    pub(crate) fn new() -> Self {
        Self {
            last: LastOutput::Nothing,
        }
    }

    /// Call before printing streamed text. Returns true if a blank line should be emitted first.
    pub(crate) fn before_text(&mut self) -> bool {
        let need_blank = matches!(self.last, LastOutput::ToolIndicator | LastOutput::Thinking);
        self.last = LastOutput::Text;
        need_blank
    }

    /// Call before printing a tool indicator. Returns true if a blank line should be emitted first.
    ///
    /// Two adjacent indicators normally sit flush, which is what makes a run of them read as a list
    /// of steps. Under [`ToolParams::Full`] each one is a multi-line block instead, so flush means
    /// the next `[tool ...]` header butts against the previous call's last argument and the two
    /// read as one call with too many parameters.
    pub(crate) fn before_tool_indicator(&mut self, params: ToolParams) -> bool {
        let need_blank = match self.last {
            LastOutput::Text | LastOutput::Thinking => true,
            LastOutput::ToolIndicator => params == ToolParams::Full,
            _ => false,
        };
        self.last = LastOutput::ToolIndicator;
        need_blank
    }

    /// Call before printing a thinking block. Returns true if a blank line should be emitted first.
    pub(crate) fn before_thinking(&mut self) -> bool {
        let need_blank = matches!(self.last, LastOutput::Text | LastOutput::ToolIndicator);
        self.last = LastOutput::Thinking;
        need_blank
    }

    /// Call after the todo list is rendered (it has its own trailing newline).
    pub(crate) fn after_todo_list(&mut self) {
        self.last = LastOutput::TodoList;
    }

    /// Call after newline_after_prompt is printed.
    pub(crate) fn after_prompt(&mut self) {
        self.last = LastOutput::Prompt;
    }
}
/// Write to stdout, returning a failure rather than panicking on it.
///
/// `print!` and `println!` panic when the write fails, which is the wrong answer for the one stream
/// a caller is meant to pipe: `meka --oneshot … | head` closes it mid-answer, and where every other
/// tool exits, meka crashes. Every write in [`StreamingRenderer`] goes through this or
/// [`write_stdout_line`], which is what makes the `io::Result` its methods already return the truth
/// about what reached the terminal rather than a formality.
pub(crate) fn write_stdout(text: impl std::fmt::Display) -> io::Result<()> {
    let mut out = io::stdout().lock();
    write!(out, "{text}")
        .and_then(|()| out.flush())
        .map_err(reader_hung_up)
}
/// [`write_stdout`] plus the newline, without building a second string to hold it.
pub(crate) fn write_stdout_line(line: impl std::fmt::Display) -> io::Result<()> {
    let mut out = io::stdout().lock();
    writeln!(out, "{line}")
        .and_then(|()| out.flush())
        .map_err(reader_hung_up)
}
/// Where a command's rendered output goes.
///
/// The same table is data on one host and chrome on another: `meka memory list` is invoked for
/// its table, so it belongs on stdout, while `/memory` inside the REPL is the user glancing at the
/// UI, where stdout carries only the model's answers. The renderer takes the stream from its door
/// rather than deciding, so one function serves both without a second copy of the rendering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stream {
    /// The data a command was invoked to obtain.
    Stdout,
    /// Everything else, including a REPL slash command's output.
    Stderr,
}

impl Stream {
    /// Write `text` to the stream. A stderr failure is dropped by design (see
    /// [`crate::streams::write_stderr`]), so only stdout can report one.
    pub(crate) fn write(self, text: impl std::fmt::Display) -> io::Result<()> {
        match self {
            Stream::Stdout => write_stdout(text),
            Stream::Stderr => {
                crate::streams::write_stderr(text);
                Ok(())
            }
        }
    }

    /// [`Self::write`] plus the newline.
    pub(crate) fn write_line(self, line: impl std::fmt::Display) -> io::Result<()> {
        match self {
            Stream::Stdout => write_stdout_line(line),
            Stream::Stderr => {
                crate::streams::write_stderr_line(line);
                Ok(())
            }
        }
    }
}
/// The payload marking a broken pipe as *this process's stdout* rather than any other.
///
/// A reader that stops reading is its own decision and meka exits 0 for it, but that has to mean
/// the reader of the stream the command was writing its answer to. A `BrokenPipe` reaching the same
/// place from somewhere else -- `session export --output <fifo>`, where the user named a
/// destination and the data did not land -- is a failure, and answering 0 to it reports success
/// over lost data. The kind alone cannot tell those apart, so the ones from here carry this.
#[derive(Debug)]
pub(crate) struct ReaderHungUp;
impl std::fmt::Display for ReaderHungUp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("the reader of stdout stopped reading")
    }
}
impl std::error::Error for ReaderHungUp {}
/// Tag a stdout failure that was the reader hanging up, leaving every other failure alone.
///
/// The kind stays `BrokenPipe`, so [`report_lost_output`] and `Console::lost_output` keep reading
/// it the way they always have; only the payload is added.
pub(super) fn reader_hung_up(error: io::Error) -> io::Error {
    if error.kind() == io::ErrorKind::BrokenPipe {
        return io::Error::new(io::ErrorKind::BrokenPipe, ReaderHungUp);
    }
    error
}
/// Whether the terminal has already been told that output is not arriving.
///
/// Global rather than per-`Console` because the writers are: `Console::text_delta` asks once per
/// streamed delta and history replay once per replayed message, and the replay path has no console
/// to hang a flag on. Without this the correction is hundreds of identical lines, which buries the
/// one that matters.
///
/// Cleared by [`reset_lost_output_report`] when an episode opens, so a REPL that runs for hours
/// says it once per prompt rather than once ever.
pub(super) static LOST_OUTPUT_REPORTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
/// Let the next episode speak again. See [`LOST_OUTPUT_REPORTED`].
pub(crate) fn reset_lost_output_report() {
    LOST_OUTPUT_REPORTED.store(false, std::sync::atomic::Ordering::Relaxed);
}
/// Whether this is the first report since the last reset, claiming the right to be it.
///
/// Split from the static so a test can drive it with a latch of its own. The global one is cleared
/// by `Console::open_episode`, which several other tests in this binary call, and a clear landing
/// between two reports made a test that used it flaky roughly once in four hundred runs.
pub(super) fn claim_first_report(latch: &std::sync::atomic::AtomicBool) -> bool {
    !latch.swap(true, std::sync::atomic::Ordering::Relaxed)
}
/// Report output that did not reach stdout, at the level the failure deserves, once.
///
/// A broken pipe is the reader's own decision -- `meka … | head` -- so saying so by default would
/// put a line on stderr about something the user did on purpose. Any other failure lost the model's
/// answer to something they did not choose, and a full disk that reports at `debug!` is a silent
/// one.
pub(crate) fn report_lost_output(what: &str, error: &io::Error) {
    if !claim_first_report(&LOST_OUTPUT_REPORTED) {
        return;
    }
    if error.kind() == io::ErrorKind::BrokenPipe {
        tracing::debug!("{what}: {error}");
    } else {
        tracing::warn!("{what}: {error}");
    }
}
/// Which stream a [`StreamingRenderer`] writes to, and therefore whose width it measures.
///
/// The two streams differ in what a failure means, so one enum decides both: `Stdout` hands
/// failures back, because a scripted host has to fail on a lost answer, and `Stderr` swallows them,
/// because a report would go to the stream that just refused it (see [`write_stderr`]).
///
/// It also picks which stream is asked for the terminal width. One field for both is what stops the
/// two from disagreeing: a renderer measuring a stream it does not write to wraps to a width its
/// own output never had.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Sink {
    Stdout,
    Stderr,
}
impl Sink {
    pub(super) fn write(self, text: &str) -> io::Result<()> {
        match self {
            Sink::Stdout => write_stdout(text),
            Sink::Stderr => {
                write_stderr(text);
                Ok(())
            }
        }
    }

    pub(super) fn is_terminal(self) -> bool {
        match self {
            Sink::Stdout => std::io::IsTerminal::is_terminal(&io::stdout()),
            Sink::Stderr => std::io::IsTerminal::is_terminal(&io::stderr()),
        }
    }

    pub(super) fn flush(self) -> io::Result<()> {
        match self {
            Sink::Stdout => io::stdout().flush(),
            Sink::Stderr => Ok(()),
        }
    }
}
/// Split `text` into what to write now and how many row endings to hold back.
///
/// The last byte is not always the row ending: syntect closes a highlighted line with a reset
/// *after* its newline, so a plain `trim_end_matches('\n')` finds nothing to hold on that path.
/// Trailing escapes are stepped over and rejoin the text they belong to, which puts the reset ahead
/// of the row ending rather than behind it: an attribute left open across one is what `ESC[K` then
/// erases with (see [`write_own_line_prelude`]).
pub(super) fn split_held_newlines(text: &str) -> (String, usize) {
    let escapes: Vec<(usize, usize)> = CSI_PATTERN
        .find_iter(text)
        .map(|found| (found.start(), found.end()))
        .collect();
    let mut end = text.len();
    let mut unread = escapes.len();
    let mut newlines = 0;
    let mut peeled = String::new();
    // Newlines and escapes *alternate* at the end, so both have to be walked. A highlighted code
    // block is emitted one styled line at a time, so its trailing blank rows arrive as
    // `\n <reset> \n <color> \n <reset>` -- stepping over one final run of escapes leaves the
    // earlier newlines inside the body, where they print as the blank rows the caller is trying to
    // decide about.
    loop {
        if let Some(&(start, stop)) = unread.checked_sub(1).and_then(|last| escapes.get(last))
            && stop == end
        {
            unread -= 1;
            peeled.insert_str(0, text.get(start..stop).unwrap_or_default());
            end = start;
            continue;
        }
        if end
            .checked_sub(1)
            .and_then(|last| text.as_bytes().get(last))
            == Some(&b'\n')
        {
            newlines += 1;
            end -= 1;
            continue;
        }
        break;
    }
    (
        format!("{}{}", text.get(..end).unwrap_or(text), peeled),
        newlines,
    )
}
pub(crate) fn render_hint(message: &str) {
    write_stderr_line(message.with(Color::DarkGrey));
}
/// A warn-level notice: something recoverable the user should see without `-v`.
///
/// Yellow, for the reason [`render_annotation`] gives: nothing has failed, so this is not
/// [`render_error`]'s red, and it must not recede into the gray a hint or a thinking block uses, or
/// a refused approval reads as one more line of the model's musings.
pub(crate) fn render_warning(message: &str) {
    write_stderr_line(message.with(Color::Yellow));
}
/// Print a single-line CLI error to stderr in the project's standard format.
pub(crate) fn render_error(error: &dyn std::fmt::Display) {
    write_stderr_line(format!("{} {}", "Error:".with(Color::Red), error));
}
/// The heading above a block of command output, in the color every other one uses.
///
/// Exists so the color is decided once rather than copied per heading.
pub(crate) fn render_heading(heading: &str) {
    write_stderr_line(heading.with(Color::Cyan));
}
/// A stage direction about the output rather than output of its own: `(interrupted)`.
///
/// Yellow, not red. None of these is a failure -- an interrupt is the user's own doing, and the
/// background-task notices describe meka doing as it was asked. [`Color::Red`] belongs to
/// [`render_error`] alone, and is worth keeping at one meaning. Yellow already carries "worth
/// noticing, nothing went wrong" here: it is the `read` permission indicator and an in-progress
/// todo. Not [`Color::DarkGrey`] either, which is the right *class* but is what thinking blocks
/// use, and the mark saying an answer is incomplete should not recede as far as the model's
/// musings -- spotting it in scrollback is the whole point, since at the time you already knew.
///
/// Parenthesised and lowercase because it annotates the transcript rather than speaking:
/// `Interrupted.` reads as meka saying something, `(interrupted)` as a note on the answer that
/// stopped, in the same register as `(truncated)`.
///
/// Every caller passes one of meka's own strings, so there is nothing here to sanitize.
pub(crate) fn render_annotation(note: &str) {
    write_stderr_line(format!("({note})").with(Color::Yellow));
}

/// Capturing this process's `tracing` output for the current thread, for tests that assert on a
/// log line.
///
/// Here rather than in either module that needs it, because there can only be one of these. The
/// subscriber has to be installed **globally**: `tracing` caches a callsite's interest process-wide
/// the first time it is evaluated, so a thread-local subscriber loses a race it cannot see -- a
/// sibling test reaching the same `warn!` first, with no subscriber installed, registers the
/// callsite as never-enabled, and every later capture of it comes back empty. That is a flake of
/// roughly 2 runs in 10, which is worse than a loud failure because it reads as a CI hiccup.
///
/// Only one global can be installed, so a second copy of this helper does not merely duplicate
/// code: the loser's `set_global_default` fails, its buffer is never written to, and its tests
/// break. The buffer stays thread-local, which is what keeps concurrent tests out of each other's
/// output.
#[cfg(test)]
pub(crate) mod log_capture {
    use std::{cell::RefCell, io, sync::OnceLock};

    thread_local! {
        static BUFFER: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    }

    struct ThreadLocalWriter;

    impl io::Write for ThreadLocalWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            BUFFER.with(|buffer| buffer.borrow_mut().extend_from_slice(bytes));
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for ThreadLocalWriter {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            ThreadLocalWriter
        }
    }

    /// Begin capturing on this thread, discarding anything already buffered.
    ///
    /// Safe to call from any number of threads and any number of times; the subscriber is installed
    /// once and the buffer it writes to is whichever thread is logging.
    ///
    /// Installed at `INFO` rather than `WARN` because one caller needs to assert an `info!`: the
    /// line that says a sweep was bounded, which exists so a capped run does not read as a complete
    /// one. Only one global subscriber can exist, so the level has to satisfy every caller and each
    /// one filters what it wants -- see [`warnings`] and [`infos`]. Capturing more than is asserted
    /// is the safe direction; a caller that asserts *silence* must filter, or an unrelated `info!`
    /// will fail it.
    pub(crate) fn start() {
        static INSTALLED: OnceLock<()> = OnceLock::new();
        INSTALLED.get_or_init(|| {
            let subscriber = tracing_subscriber::fmt()
                .with_writer(ThreadLocalWriter)
                .with_max_level(tracing::Level::INFO)
                .with_ansi(false)
                .without_time()
                .finish();
            // An already-installed global is not worth failing a test over: what this needs is for
            // the callsites it asserts on to be *enabled*. Reported rather than discarded, since a
            // future change that breaks capture would otherwise do it silently and every assertion
            // built on this would start passing vacuously.
            if let Err(error) = tracing::subscriber::set_global_default(subscriber) {
                eprintln!("log capture: a global subscriber was already installed: {error}");
            }
        });
        BUFFER.with(|buffer| buffer.borrow_mut().clear());
    }

    /// What this thread has logged since [`start`], every level together.
    pub(crate) fn captured() -> String {
        BUFFER.with(|buffer| String::from_utf8_lossy(&buffer.borrow()).into_owned())
    }

    /// Only the `WARN` lines. What a caller asserting "this warned once, not once per tick" wants,
    /// and what a caller asserting silence *must* use.
    pub(crate) fn warnings() -> String {
        at_level("WARN")
    }

    /// Only the `INFO` lines.
    pub(crate) fn infos() -> String {
        at_level("INFO")
    }

    /// The subscriber writes the level as the first token of each event, so selecting one is a
    /// filter over the text. A multi-line event keeps its continuation lines with the line that
    /// names the level.
    ///
    /// Matched as that leading token and not with `contains`, which finds a level name anywhere in
    /// the line, including inside the *message*: a `WARN` about a gate watching a log (`grep ERROR
    /// ...`) would be filed as ERROR and dropped, and an assertion counting warnings would silently
    /// undercount.
    fn at_level(level: &str) -> String {
        let mut kept = String::new();
        let mut keeping = false;
        for line in captured().lines() {
            let leading = line.split_whitespace().next().unwrap_or_default();
            if ["ERROR", "WARN", "INFO", "DEBUG", "TRACE"].contains(&leading) {
                keeping = leading == level;
            }
            if keeping {
                kept.push_str(line);
                kept.push('\n');
            }
        }
        kept
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A level name inside a *message* must not be mistaken for the line's level: a gate watching
    /// a log (`grep ERROR ...`) puts exactly that into a warning, and every assertion built on
    /// `warnings()` would undercount in silence.
    #[test]
    fn log_capture_files_a_line_by_its_level_not_by_its_message() {
        log_capture::start();
        tracing::warn!("gate for job abc failed: grep ERROR /var/log/app returned nothing");
        tracing::info!("held over 3 due job(s)");

        let warnings = log_capture::warnings();
        assert!(
            warnings.contains("grep ERROR"),
            "a warning whose message names another level is still a warning: {warnings:?}"
        );
        assert!(
            !warnings.contains("held over"),
            "and an info line is not one: {warnings:?}"
        );
        assert!(
            log_capture::infos().contains("held over"),
            "which is where it does belong"
        );
    }

    /// A stdout that stopped taking writes is named once, and named again after a reset.
    ///
    /// Once per process is right for a one-shot run and wrong for a shell left open all day, where
    /// the first lost answer would otherwise be the only one mentioned. The repetition is real:
    /// `Console::text_delta` asks per streamed delta and history replay per replayed message.
    ///
    /// Driven through a latch of this test's own, not the global. `Console::open_episode` clears
    /// that one, several tests in this binary call it, and they run in parallel; asserting across
    /// two reports on a shared flag is a race. What that leaves untested is the wiring -- that
    /// `report_lost_output` reads the global and `open_episode` clears it -- which is two lines.
    #[test]
    fn a_lost_answer_is_named_once_until_the_next_episode() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let latch = AtomicBool::new(false);
        assert!(
            claim_first_report(&latch),
            "the first report is the one that speaks"
        );
        assert!(!claim_first_report(&latch), "and every later one is silent");

        latch.store(false, Ordering::Relaxed);
        assert!(
            claim_first_report(&latch),
            "until an episode opens, which lets the next lost answer be named too"
        );
    }

    /// A run of one-line indicators reads as a list of steps, and spacing them out would stretch a
    /// six-call turn down the screen for nothing.
    #[test]
    fn summary_indicators_stay_flush_with_each_other() {
        let mut spacing = super::OutputSpacing::new();
        assert!(!spacing.before_tool_indicator(ToolParams::Summary));
        assert!(!spacing.before_tool_indicator(ToolParams::Summary));
    }

    /// Under `full` each indicator is a block, so flush would run the next `[tool ...]` header into
    /// the previous call's last argument.
    #[test]
    fn full_indicators_are_separated_from_each_other() {
        let mut spacing = super::OutputSpacing::new();
        assert!(!spacing.before_tool_indicator(ToolParams::Full));
        assert!(spacing.before_tool_indicator(ToolParams::Full));
    }

    #[test]
    fn an_indicator_after_text_is_separated_whatever_the_style() {
        for style in [ToolParams::Off, ToolParams::Summary, ToolParams::Full] {
            let mut spacing = super::OutputSpacing::new();
            spacing.before_text();
            assert!(spacing.before_tool_indicator(style), "{}", style);
        }
    }
}
