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
/// `text` in `color` where stderr is a terminal, and as it is otherwise: a log file or the next
/// tool in a pipeline would keep the escapes as bytes.
pub(crate) fn colored_for_stderr(text: impl std::fmt::Display, color: Color) -> String {
    use std::io::IsTerminal;
    if std::io::stderr().is_terminal() {
        crossterm::style::style(text).with(color).to_string()
    } else {
        text.to_string()
    }
}

pub(crate) fn render_hint(message: &str) {
    write_stderr_line(colored_for_stderr(rows_of(message), Color::DarkGrey));
}

/// A message's text, ending where its last row does: a trailing newline would print an empty row of
/// its own under it.
fn rows_of(message: &str) -> &str {
    message.trim_end_matches('\n')
}
/// Print a single-line CLI error to stderr in the project's standard format.
pub(crate) fn render_error(error: &dyn std::fmt::Display) {
    write_stderr_line(format!(
        "{} {}",
        colored_for_stderr("Error:", Color::Red),
        error
    ));
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
    write_stderr_line(colored_for_stderr(
        format!("({})", rows_of(note)),
        Color::Yellow,
    ));
}
