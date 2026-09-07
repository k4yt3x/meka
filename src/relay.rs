//! Relays `tracing` output around the live REPL prompt.
//!
//! Without this layer, `tracing` writes directly to `std::io::stderr`, and a line written while
//! reedline is in raw mode is overwritten by its next redraw of the prompt.
//!
//! [`Relay`] holds an optional [`reedline::ExternalPrinter`] (a crossbeam channel reedline drains
//! every poll tick to print messages *above* the prompt without clobbering it). Tracing output goes
//! through the printer when the REPL has registered one; otherwise it falls back to plain stderr so
//! the non-interactive paths (`meka session export`, `meka session list`, etc.) and the pre-REPL
//! startup window still see logs.
//!
//! Crucially, reedline only drains that channel while `read_line()` is running, so the printer is
//! used *only* while the prompt is live (tracked via [`Relay::set_at_prompt`]). Off-prompt windows,
//! most importantly during a turn while the REPL thread is blocked on the agent, write straight to
//! stderr instead, so warnings surface as they happen rather than being buffered until the turn
//! ends and the next prompt is drawn.

use std::{
    io::{self, Write},
    sync::{
        Arc, LazyLock, Mutex, RwLock, Weak,
        atomic::{AtomicBool, Ordering},
    },
};

use reedline::ExternalPrinter;
use tracing_subscriber::fmt::MakeWriter;

use crate::console::Console;

/// Process-global relay. Tracing's `MakeWriter` clones this; the REPL installs its printer at
/// startup. Stays uninstalled for non-interactive commands, so they keep getting plain stderr
/// output.
pub(crate) static RELAY: LazyLock<Relay> = LazyLock::new(Relay::new);

/// Routes log output through reedline's [`ExternalPrinter`] when the interactive REPL has installed
/// one; falls back to stderr otherwise.
#[derive(Clone)]
pub(crate) struct Relay {
    printer: Arc<RwLock<Option<ExternalPrinter<String>>>>,
    /// True only while reedline's `read_line()` owns the terminal. reedline drains the
    /// `ExternalPrinter` channel only inside that loop, so a line routed through it at any other
    /// time would sit until the next prompt is drawn; off-prompt the terminal is in cooked mode
    /// and stderr is safe.
    at_prompt: Arc<AtomicBool>,
    /// The host's console, so an off-prompt log line can settle the row before landing on it.
    ///
    /// Weak on purpose: this static outlives every REPL, and an owning handle would keep a console
    /// (and the renderer behind it) alive for the life of the process after the host that made it
    /// has gone. A dead weak pointer answers the same as no console at all, which is the
    /// non-interactive behavior.
    console: Arc<RwLock<Weak<Mutex<Console>>>>,
}

impl Relay {
    fn new() -> Self {
        Self {
            printer: Arc::new(RwLock::new(None)),
            at_prompt: Arc::new(AtomicBool::new(false)),
            console: Arc::new(RwLock::new(Weak::new())),
        }
    }

    /// Register the console a log line should settle the row through, off-prompt.
    ///
    /// Both CLI hosts install one. `--oneshot` has no reedline and so no `ExternalPrinter`, but it
    /// draws the same thinking indicator, so it has the same row to be written over.
    pub(crate) fn install_console(&self, console: &Arc<Mutex<Console>>) {
        *crate::sync::write(&self.console) = Arc::downgrade(console);
    }

    /// Register an [`ExternalPrinter`] so subsequent log lines get printed above the live prompt
    /// instead of racing reedline's redraw. Caller keeps a clone of the same printer to hand to
    /// [`reedline::Reedline::with_external_printer`].
    pub(crate) fn install(&self, printer: ExternalPrinter<String>) {
        *crate::sync::write(&self.printer) = Some(printer);
    }

    /// Mark whether reedline's `read_line()` is currently active; the REPL sets this around each
    /// call.
    pub(crate) fn set_at_prompt(&self, at_prompt: bool) {
        self.at_prompt.store(at_prompt, Ordering::Relaxed);
    }
}

impl<'a> MakeWriter<'a> for Relay {
    type Writer = RelayWriter;

    fn make_writer(&'a self) -> Self::Writer {
        let printer = self.printer.read().ok().and_then(|guard| guard.clone());
        let console = self.console.read().ok().and_then(|guard| guard.upgrade());
        RelayWriter {
            printer,
            at_prompt: Arc::clone(&self.at_prompt),
            console,
        }
    }
}

/// Per-write borrow handed back to the tracing formatter. Holds a clone of the printer (cheap: it's
/// a pair of crossbeam channel handles) captured at the moment `make_writer` was called, so a
/// printer install or clear racing with an in-flight write doesn't tear. `at_prompt` is read at
/// write time so the routing reflects the live REPL state, not whatever it was at `make_writer`.
pub(crate) struct RelayWriter {
    printer: Option<ExternalPrinter<String>>,
    at_prompt: Arc<AtomicBool>,
    console: Option<Arc<Mutex<Console>>>,
}

impl Write for RelayWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        // Only hand the line to the printer while the prompt is live: reedline drains that channel
        // only inside `read_line()`, so an off-prompt line would sit until the next prompt.
        if self.at_prompt.load(Ordering::Relaxed)
            && let Some(printer) = &self.printer
        {
            // The printer adds its own line break, so the newline tracing's formatter appends is
            // stripped, and an empty message (formatter buffering) would print as a blank line.
            match std::str::from_utf8(buffer) {
                Ok(text) => {
                    let trimmed = text.trim_end_matches('\n');
                    if trimmed.is_empty() {
                        return Ok(buffer.len());
                    }
                    // `try_send`, never `print`: the printer is a bounded channel reedline drains
                    // once per poll, and `print` blocks when it is full, which hangs the REPL
                    // thread for good when that thread, the only drainer, is the one logging. A
                    // full queue falls through to stderr below.
                    match printer.sender().try_send(trimmed.to_string()) {
                        Ok(()) => return Ok(buffer.len()),
                        Err(error) if error.is_disconnected() => {
                            tracing::trace!("the editor has gone; a relayed line was dropped");
                            return Ok(buffer.len());
                        }
                        Err(_full) => {}
                    }
                }
                Err(_) => {
                    // Non-UTF-8 bytes fall through to stderr rather than being dropped.
                }
            }
        }
        self.settle_the_row();
        io::stderr().write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        io::stderr().flush()
    }
}

impl RelayWriter {
    /// Tell the console that output it cannot see is about to land on the current row.
    ///
    /// Off-prompt is when the row may not be free: a turn draws the thinking indicator as a
    /// [`crate::console::RowState::Transient`] line, and a `warn!` printed onto it is wiped by the
    /// next `Settle::Erase`.
    ///
    /// `try_lock`, never `lock`: [`Console`] methods log through `render::report_lost_output`, so a
    /// thread already inside one re-enters here holding the lock, and a blocking acquire would
    /// deadlock the REPL. Contention falls through to the raw stderr write.
    ///
    /// A poisoned mutex is recovered rather than treated as contention: `TryLockError` folds the
    /// two together, and treating them alike would let one panic inside a console method stop
    /// row-settling for the rest of the process.
    ///
    /// Idempotent, because a formatter may split one event across several `write` calls. The row
    /// is left `Empty` on the strength of tracing's formatter terminating every event with a
    /// newline.
    fn settle_the_row(&self) {
        let Some(console) = &self.console else {
            return;
        };
        match console.try_lock() {
            Ok(mut console) => console.announce_foreign_output(),
            Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                poisoned.into_inner().announce_foreign_output()
            }
            Err(std::sync::TryLockError::WouldBlock) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::RenderMode,
        console::{Console, Neighbor, RowState, Spacing},
    };

    fn console() -> Arc<Mutex<Console>> {
        Arc::new(Mutex::new(Console::new(
            Spacing {
                newline_before_prompt: true,
                newline_after_prompt: true,
            },
            RenderMode::Raw,
        )))
    }

    /// A full printer queue does not block the writer. The queue is drained only inside reedline's
    /// `read_line`, so a write that waited for room could wait on the very thread doing the
    /// writing.
    #[test]
    fn a_full_printer_queue_does_not_block_a_log_line() {
        let relay = Relay::new();
        let printer = ExternalPrinter::<String>::new(1);
        relay.install(printer);
        relay.set_at_prompt(true);
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::Write as _;

            use tracing_subscriber::fmt::MakeWriter;
            for line in ["first\n", "second\n", "third\n"] {
                relay
                    .make_writer()
                    .write_all(line.as_bytes())
                    .expect("write");
            }
            done.send(()).expect("report");
        });
        finished
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the third line blocked on a queue nobody drains");
    }

    /// An off-prompt log line settles the row instead of landing on it.
    ///
    /// This is the wiring, not the state machine: `step` already knows that foreign output erases a
    /// transient row. The row is the thinking indicator's, and the line that lands on it is the
    /// retry path's `warn!`, which fires at default verbosity.
    ///
    /// `force_row` because the drawing API cannot reach `Transient` without a terminal:
    /// `thinking_indicator` returns early on `!live_indicator_supported()`.
    #[test]
    fn an_off_prompt_log_line_settles_the_row_it_would_have_landed_on() {
        let console = console();
        let relay = Relay::new();
        relay.install_console(&console);
        relay.set_at_prompt(false);

        console
            .lock()
            .expect("console")
            .open_episode(RowState::Empty, Neighbor::Prompt);
        console
            .lock()
            .expect("console")
            .force_row(RowState::Transient);

        relay
            .make_writer()
            .write_all(b"WARN provider stream failed transiently, retrying\n")
            .expect("the stderr fall-through always succeeds");

        let console = console.lock().expect("console");
        assert_eq!(
            console.row(),
            RowState::Empty,
            "the indicator's row has to be settled before the line lands, or the next \
             `Settle::Erase` takes the line with it"
        );
        assert!(
            console.has_printed(),
            "and the episode has to know it printed, or its closing blank is skipped"
        );
    }

    /// A console already locked by this thread is written past, not deadlocked on.
    ///
    /// `Console::text_delta`, `close_stream` and the stdout flush all report through
    /// `render::report_lost_output` when their renderer fails, so a thread inside a console
    /// method reaches this writer while holding the very lock it wants.
    ///
    /// The guard is held across the write, which is exactly the reentrant shape, and the test
    /// completing at all is the assertion.
    #[test]
    fn a_log_line_from_inside_the_console_falls_through_rather_than_deadlocking() {
        let console = console();
        let relay = Relay::new();
        relay.install_console(&console);
        relay.set_at_prompt(false);

        let mut held = console.lock().expect("console");
        held.open_episode(RowState::Empty, Neighbor::Prompt);

        relay
            .make_writer()
            .write_all(b"DEBUG console renderer push_delta failed\n")
            .expect("the stderr fall-through always succeeds");

        assert!(
            !held.has_printed(),
            "the console could not be consulted, so it must not have been told anything either"
        );
    }

    /// With no console installed (every non-interactive command) nothing changes.
    #[test]
    fn a_host_with_no_console_still_writes_to_stderr() {
        let relay = Relay::new();
        relay.set_at_prompt(false);
        relay
            .make_writer()
            .write_all(b"WARN config.toml could not be read\n")
            .expect("plain stderr is the fallback for `meka mcp list` and friends");
    }
}
