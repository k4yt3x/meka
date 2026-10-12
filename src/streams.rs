//! Raw writes to the process's standard streams, for every layer that owns a line of chrome.
//!
//! Only stderr lives here so far: it is the stream a failure report would go to, so a failed write
//! has nowhere to be reported and is dropped by design. Stdout's writers stay with the renderer,
//! whose lost-output reporting they feed.

use std::io::{self, Write};

tokio::task_local! {
    /// Where a line goes instead of the terminal while a host that owns the screen runs a command
    /// written for the CLI.
    ///
    /// The slash commands reuse the `cli` handlers, which print their tables and confirmations
    /// through the writers here. A host drawing on the alternate screen has no stream a line can
    /// land on without corrupting the display, so it captures what the handler printed and shows it
    /// as one block. Task-local rather than global because the handler runs on the host's own task,
    /// and nothing another task prints meanwhile is the command's.
    static CAPTURE: std::cell::RefCell<String>;
}

/// Run `future` with every line it prints collected rather than written, and return both.
pub(crate) async fn capture<F: std::future::Future>(future: F) -> (F::Output, String) {
    let buffer = std::cell::RefCell::new(String::new());
    CAPTURE
        .scope(buffer, async move {
            let output = future.await;
            let captured = CAPTURE.with(|buffer| std::mem::take(&mut *buffer.borrow_mut()));
            (output, captured)
        })
        .await
}

/// Append `text` to the capture, if one is active on this task.
pub(crate) fn captured(text: &str) -> bool {
    CAPTURE
        .try_with(|buffer| buffer.borrow_mut().push_str(text))
        .is_ok()
}

/// Write chrome to stderr, and accept that a failure here cannot be reported.
///
/// `eprint!` and `eprintln!` panic when the write fails, which turns `meka … 2>&1 | head` into a
/// crash. Unlike stdout there is nothing to hand back: this is the stream a report would go to, so
/// a caller could only try to say so down the pipe that just refused it. The exit code still
/// carries whatever the run concluded, which is the part a script reads.
pub(crate) fn write_stderr(text: impl std::fmt::Display) {
    if captured(&text.to_string()) {
        return;
    }
    let mut out = io::stderr().lock();
    // `.ok()` rather than `?` or a log: see above. Both would write to this same stream.
    write!(out, "{text}").ok();
    out.flush().ok();
}
/// [`write_stderr`] plus the newline.
pub(crate) fn write_stderr_line(line: impl std::fmt::Display) {
    if captured(&format!("{line}\n")) {
        return;
    }
    let mut out = io::stderr().lock();
    writeln!(out, "{line}").ok();
    out.flush().ok();
}
