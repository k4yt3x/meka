//! Relays `tracing` output around the live REPL prompt.
//!
//! Without this layer, `tracing` writes directly to `std::io::stderr`. When reedline enters raw
//! mode and redraws the prompt, anything just written to stderr risks getting overwritten by
//! reedline's cursor positioning. The symptom users see is "an error log flashes by, then
//! disappears."
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
        Arc, LazyLock, RwLock,
        atomic::{AtomicBool, Ordering},
    },
};

use reedline::ExternalPrinter;
use tracing_subscriber::fmt::MakeWriter;

/// Process-global relay. Tracing's `MakeWriter` clones this; the REPL installs its printer at
/// startup. Stays uninstalled for non-interactive commands, so they keep getting plain stderr
/// output.
pub static RELAY: LazyLock<Relay> = LazyLock::new(Relay::new);

/// Routes log output through reedline's [`ExternalPrinter`] when the interactive REPL has installed
/// one; falls back to stderr otherwise.
#[derive(Clone)]
pub struct Relay {
    printer: Arc<RwLock<Option<ExternalPrinter<String>>>>,
    /// True only while reedline's `read_line()` owns the terminal (raw mode, prompt drawn).
    /// reedline drains the `ExternalPrinter` channel exclusively inside that loop, so routing a
    /// log line through the printer at any other time (e.g. during a turn, while the REPL thread
    /// is blocked waiting on the agent) would buffer it until the next prompt is drawn. When this
    /// is false the terminal is in cooked mode, so writing straight to stderr is both safe and
    /// immediate.
    at_prompt: Arc<AtomicBool>,
}

impl Relay {
    fn new() -> Self {
        Self {
            printer: Arc::new(RwLock::new(None)),
            at_prompt: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Register an [`ExternalPrinter`] so subsequent log lines get printed above the live prompt
    /// instead of racing reedline's redraw. Caller keeps a clone of the same printer to hand to
    /// [`reedline::Reedline::with_external_printer`].
    pub fn install(&self, printer: ExternalPrinter<String>) {
        *self
            .printer
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(printer);
    }

    /// Drop the registered printer. Called on REPL teardown so tracing reverts to plain stderr
    /// (e.g. interrupt handlers that fire after reedline has exited).
    #[allow(dead_code)]
    pub fn clear(&self) {
        *self
            .printer
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    }

    /// Mark whether reedline's `read_line()` is currently active. The REPL sets this true around
    /// each `read_line()` call and false otherwise, so log lines route through the
    /// `ExternalPrinter` only while the prompt is live (and reedline is draining it) and go
    /// straight to stderr the rest of the time, surfacing immediately instead of buffering until
    /// the next prompt.
    pub fn set_at_prompt(&self, at_prompt: bool) {
        self.at_prompt.store(at_prompt, Ordering::Relaxed);
    }
}

impl<'a> MakeWriter<'a> for Relay {
    type Writer = RelayWriter;

    fn make_writer(&'a self) -> Self::Writer {
        let printer = self.printer.read().ok().and_then(|guard| guard.clone());
        RelayWriter {
            printer,
            at_prompt: Arc::clone(&self.at_prompt),
        }
    }
}

/// Per-write borrow handed back to the tracing formatter. Holds a clone of the printer (cheap: it's
/// a pair of crossbeam channel handles) captured at the moment `make_writer` was called, so a
/// printer install or clear racing with an in-flight write doesn't tear. `at_prompt` is read at
/// write time so the routing reflects the live REPL state, not whatever it was at `make_writer`.
pub struct RelayWriter {
    printer: Option<ExternalPrinter<String>>,
    at_prompt: Arc<AtomicBool>,
}

impl Write for RelayWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // Only hand the line to reedline's ExternalPrinter while the prompt is live: reedline
        // drains that channel exclusively inside `read_line()`, so off-prompt (during a turn) the
        // line would sit buffered until the next prompt. Off-prompt the terminal is in cooked mode,
        // so the stderr fall-through below is both safe and immediate.
        if self.at_prompt.load(Ordering::Relaxed)
            && let Some(printer) = &self.printer
        {
            // Reedline's ExternalPrinter prints each message as a fresh line above the prompt and
            // adds its own line break, so we strip the trailing newline tracing's formatter
            // appends. Empty messages are dropped to avoid blank-line spam from formatter
            // buffering.
            match std::str::from_utf8(buf) {
                Ok(text) => {
                    let trimmed = text.trim_end_matches('\n');
                    if !trimmed.is_empty() {
                        let _ = printer.print(trimmed.to_string());
                    }
                    return Ok(buf.len());
                }
                Err(_) => {
                    // Non-UTF-8 bytes from tracing are unexpected; fall through to stderr so
                    // they're not silently dropped.
                }
            }
        }
        io::stderr().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        io::stderr().flush()
    }
}
