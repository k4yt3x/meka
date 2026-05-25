//! Relays `tracing` output around the live REPL prompt.
//!
//! Without this layer, `tracing` writes directly to `std::io::stderr`. When reedline enters raw
//! mode and redraws the prompt, anything just written to stderr risks getting overwritten by
//! reedline's cursor positioning — the symptom users see is "an error log flashes by, then
//! disappears."
//!
//! [`Relay`] holds an optional [`reedline::ExternalPrinter`] (a crossbeam channel reedline drains
//! every poll tick to print messages *above* the prompt without clobbering it). Tracing output goes
//! through the printer when the REPL has registered one; otherwise it falls back to plain stderr so
//! the non-interactive paths (`agsh export`, `agsh list`, etc.) and the pre-REPL startup window
//! still see logs.

use std::{
    io::{self, Write},
    sync::{Arc, LazyLock, RwLock},
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
}

impl Relay {
    fn new() -> Self {
        Self {
            printer: Arc::new(RwLock::new(None)),
        }
    }

    /// Register an [`ExternalPrinter`] so subsequent log lines get printed above the live prompt
    /// instead of racing reedline's redraw. Caller keeps a clone of the same printer to hand to
    /// [`reedline::Reedline::with_external_printer`].
    pub fn install(&self, printer: ExternalPrinter<String>) {
        *self.printer.write().expect("relay lock poisoned") = Some(printer);
    }

    /// Drop the registered printer. Called on REPL teardown so tracing reverts to plain stderr
    /// (e.g. interrupt handlers that fire after reedline has exited).
    #[allow(dead_code)]
    pub fn clear(&self) {
        *self.printer.write().expect("relay lock poisoned") = None;
    }
}

impl<'a> MakeWriter<'a> for Relay {
    type Writer = RelayWriter;

    fn make_writer(&'a self) -> Self::Writer {
        let printer = self.printer.read().ok().and_then(|guard| guard.clone());
        RelayWriter { printer }
    }
}

/// Per-write borrow handed back to the tracing formatter. Holds a clone of the printer (cheap —
/// it's a pair of crossbeam channel handles) captured at the moment `make_writer` was called, so a
/// printer install or clear racing with an in-flight write doesn't tear.
pub struct RelayWriter {
    printer: Option<ExternalPrinter<String>>,
}

impl Write for RelayWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if let Some(printer) = &self.printer {
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
