//! Raw writes to the process's standard streams, for every layer that owns a line of chrome.
//!
//! Only stderr lives here so far: it is the stream a failure report would go to, so a failed write
//! has nowhere to be reported and is dropped by design. Stdout's writers stay with the renderer,
//! whose lost-output reporting they feed.

use std::io::{self, Write};

/// Write chrome to stderr, and accept that a failure here cannot be reported.
///
/// `eprint!` and `eprintln!` panic when the write fails, which turns `meka … 2>&1 | head` into a
/// crash. Unlike stdout there is nothing to hand back: this is the stream a report would go to, so
/// a caller could only try to say so down the pipe that just refused it. The exit code still
/// carries whatever the run concluded, which is the part a script reads.
pub(crate) fn write_stderr(text: impl std::fmt::Display) {
    let mut out = io::stderr().lock();
    // `.ok()` rather than `?` or a log: see above. Both would write to this same stream.
    write!(out, "{text}").ok();
    out.flush().ok();
}
/// [`write_stderr`] plus the newline.
pub(crate) fn write_stderr_line(line: impl std::fmt::Display) {
    let mut out = io::stderr().lock();
    writeln!(out, "{line}").ok();
    out.flush().ok();
}
