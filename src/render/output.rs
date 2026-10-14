//! Writing to the terminal: the two streams, a reader that hung up, and the styled one-line
//! notices.

use super::*;
use crate::streams::write_stderr_line;

/// Write to stdout, returning a failure rather than panicking on it.
///
/// `print!` and `println!` panic when the write fails, which is the wrong answer for the one stream
/// a caller is meant to pipe: `meka --oneshot … | head` closes it mid-answer, and where every other
/// tool exits, meka crashes. Every write of an answer goes through this or [`write_stdout_line`],
/// which is what makes the `io::Result` the truth about what reached the terminal rather than a
/// formality.
pub(crate) fn write_stdout(text: impl std::fmt::Display) -> io::Result<()> {
    if crate::streams::captured(&text.to_string()) {
        return Ok(());
    }
    let mut out = io::stdout().lock();
    write!(out, "{text}")
        .and_then(|()| out.flush())
        .map_err(reader_hung_up)
}
/// [`write_stdout`] plus the newline, without building a second string to hold it.
pub(crate) fn write_stdout_line(line: impl std::fmt::Display) -> io::Result<()> {
    if crate::streams::captured(&format!("{line}\n")) {
        return Ok(());
    }
    let mut out = io::stdout().lock();
    writeln!(out, "{line}")
        .and_then(|()| out.flush())
        .map_err(reader_hung_up)
}
/// Where a command's rendered output goes.
///
/// The same table is data on one host and chrome on another: `meka memory list` is invoked for
/// its table, so it belongs on stdout, while `/memory` inside the screen is the user glancing at
/// the UI, where stdout carries only the model's answers. The renderer takes the stream from its
/// door rather than deciding, so one function serves both without a second copy of the rendering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stream {
    /// The data a command was invoked to obtain.
    Stdout,
    /// Everything else, including a screen slash command's output.
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
/// The kind stays `BrokenPipe`, so a caller reading the kind sees what it always did; only the
/// payload is added.
pub(super) fn reader_hung_up(error: io::Error) -> io::Error {
    if error.kind() == io::ErrorKind::BrokenPipe {
        return io::Error::new(io::ErrorKind::BrokenPipe, ReaderHungUp);
    }
    error
}
pub(crate) fn render_hint(message: &str) {
    write_stderr_line(rows_of(message).with(Color::DarkGrey));
}

/// A message's text, ending where its last row does: a trailing newline would print an empty row of
/// its own under it.
fn rows_of(message: &str) -> &str {
    message.trim_end_matches('\n')
}
/// Print a single-line CLI error to stderr in the project's standard format.
pub(crate) fn render_error(error: &dyn std::fmt::Display) {
    write_stderr_line(format!("{} {}", "Error:".with(Color::Red), error));
}
/// A stage direction about the output rather than output of its own: `(interrupted)`.
///
/// Yellow, not red. None of these is a failure -- an interrupt is the user's own doing, and the
/// background-task notices describe meka doing as it was asked. [`Color::Red`] belongs to
/// [`render_error`] alone, and is worth keeping at one meaning. Yellow already carries "worth
/// noticing, nothing went wrong" here: it is the `read` permission indicator and an in-progress
/// checklist item. Not [`Color::DarkGrey`] either, which is the right *class* but is what thinking
/// blocks use, and the mark saying an answer is incomplete should not recede as far as the model's
/// musings -- spotting it in scrollback is the whole point, since at the time you already knew.
///
/// Parenthesised and lowercase because it annotates the transcript rather than speaking:
/// `Interrupted.` reads as meka saying something, `(interrupted)` as a note on the answer that
/// stopped, in the same register as `(truncated)`.
///
/// Every caller passes one of meka's own strings, so there is nothing here to sanitize.
pub(crate) fn render_annotation(note: &str) {
    write_stderr_line(format!("({})", rows_of(note)).with(Color::Yellow));
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
}
