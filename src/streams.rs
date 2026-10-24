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

/// Test support: what the code under test logs on this thread, read back as text.
#[cfg(test)]
pub(crate) mod capture {
    use std::sync::{Arc, Mutex};

    /// Everything `tracing` emits on this thread at the level asked for or above, for as long as
    /// this lives.
    pub(crate) struct CapturedLogs {
        bytes: Arc<Mutex<Vec<u8>>>,
        _scoped: tracing::subscriber::DefaultGuard,
        _registered_twice: tracing::Dispatch,
    }

    impl CapturedLogs {
        /// The lines captured so far.
        pub(crate) fn logged(&self) -> String {
            String::from_utf8_lossy(&crate::sync::lock(&self.bytes)).into_owned()
        }
    }

    #[derive(Clone)]
    struct Writer(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for Writer {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            crate::sync::lock(&self.0).extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Writer {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Capture what this thread logs at `level` or above until the value is dropped.
    ///
    /// The subscriber is registered twice for as long as the capture lives. With exactly one
    /// scoped subscriber registered, `tracing-core` computes a callsite's interest, on whichever
    /// thread first reaches the callsite, from that thread's own default subscriber: a sibling
    /// test's thread reaching a callsite first caches `never` on it, and this thread's event at it
    /// is then dropped before any subscriber sees it. With two registered, every thread computes
    /// the interest from the registered subscribers instead.
    pub(crate) fn capture_logs(level: tracing::Level) -> CapturedLogs {
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let subscriber = Arc::new(
            tracing_subscriber::fmt()
                .with_writer(Writer(Arc::clone(&bytes)))
                .with_max_level(level)
                .with_ansi(false)
                .finish(),
        );
        let registered_twice = tracing::Dispatch::new(Arc::clone(&subscriber));
        let scoped = tracing::subscriber::set_default(subscriber);
        CapturedLogs {
            bytes,
            _scoped: scoped,
            _registered_twice: registered_twice,
        }
    }
}
