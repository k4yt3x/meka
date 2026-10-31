//! Relays `tracing` output to wherever the host can show it.
//!
//! Without this layer, `tracing` writes directly to `std::io::stderr`, which the interactive screen
//! has taken over: a line written there lands on the alternate screen and is painted over by the
//! next frame.
//!
//! [`Relay`] holds the sending half of an optional channel the screen drains every frame and shows
//! as items, fed by [`ScreenLayer`] with each event's level and message while the screen is up
//! (from [`Relay::attach_screen`] until the [`ScreenLogs`] it returns is dropped). Otherwise the
//! formatted line falls back to plain stderr, so the non-interactive paths (`meka session export`,
//! `meka session list`, etc.) and the startup window before a screen exists still see logs.
use std::{
    io::{self, Write},
    sync::{
        Arc, LazyLock, RwLock,
        atomic::{AtomicBool, Ordering},
        mpsc::{SyncSender, TrySendError},
    },
};

use tracing_subscriber::fmt::MakeWriter;

/// Process-global relay. Tracing's `MakeWriter` clones this; the screen installs its sender at
/// startup. Stays uninstalled for non-interactive commands, so they keep getting plain stderr
/// output.
pub(crate) static RELAY: LazyLock<Relay> = LazyLock::new(Relay::new);

/// One log event as the screen shows it: the level decides the row's glyph, the target is named
/// only on the lines `-v` lets through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LogLine {
    pub(crate) at: chrono::DateTime<chrono::Utc>,
    pub(crate) level: tracing::Level,
    pub(crate) target: String,
    pub(crate) message: String,
    /// Whether the user's own log filter passes it: drawn on the transcript when it does, kept
    /// for `/log` either way.
    pub(crate) shown: bool,
}

/// Routes log output to the screen when it has installed its sender; falls back to stderr
/// otherwise.
#[derive(Clone)]
pub(crate) struct Relay {
    /// The sending half of the screen's log channel; the screen keeps the receiving half, so this
    /// is all a logging thread can hold.
    sender: Arc<RwLock<Option<SyncSender<LogLine>>>>,
    /// True only while the screen owns the terminal and drains the channel. Off it, the terminal
    /// is the shell's again and stderr is safe.
    screen_attached: Arc<AtomicBool>,
}

impl Relay {
    fn new() -> Self {
        Self {
            sender: Arc::new(RwLock::new(None)),
            screen_attached: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Register the sending half of the screen's log channel, so subsequent log lines are shown
    /// on the screen instead of written over it.
    fn install(&self, sender: SyncSender<LogLine>) {
        *crate::sync::write(&self.sender) = Some(sender);
    }

    /// Start queuing every event for a screen, up to `capacity` lines at a time, and hand back
    /// the queue's receiving end. From here on nothing the relay sees reaches stderr until the
    /// returned value is dropped, which is what makes the queue safe to open before the screen
    /// is: what the screen never drew goes to stderr then, rather than nowhere.
    pub(crate) fn attach_screen(&self, capacity: usize) -> ScreenLogs {
        let (sender, receiver) = std::sync::mpsc::sync_channel(capacity);
        self.install(sender);
        self.set_screen_attached(true);
        ScreenLogs { receiver }
    }

    /// Mark whether a screen is up and draining the channel.
    fn set_screen_attached(&self, screen_attached: bool) {
        self.screen_attached
            .store(screen_attached, Ordering::Relaxed);
    }

    fn is_screen_attached(&self) -> bool {
        self.screen_attached.load(Ordering::Relaxed)
    }

    /// Hand one event to the screen, while it is up. Nothing is reported about a line that could
    /// not be handed over: an event raised inside tracing's own layer is dropped by its
    /// re-entrancy guard.
    fn offer(&self, line: LogLine) {
        if !self.is_screen_attached() {
            return;
        }
        let Some(sender) = self.sender.read().ok().and_then(|guard| guard.clone()) else {
            return;
        };
        // `try_send`, never `send`: the channel is bounded and drained once per frame, and `send`
        // blocks when it is full, which would hang the screen for good when the screen, the only
        // drainer, is the task logging. A full queue loses the line.
        match sender.try_send(line) {
            Ok(()) | Err(TrySendError::Disconnected(_) | TrySendError::Full(_)) => {}
        }
    }
}

/// The screen's end of the log queue, from the moment the relay starts queuing until the screen
/// has drained it for the last time. Dropping it stops the queuing and writes whatever was queued
/// and never drawn to stderr: the lines logged while a session was assembled, when the run fails
/// before the screen opens, and the lines a turn the exit cut short logs after the screen is gone.
pub(crate) struct ScreenLogs {
    receiver: std::sync::mpsc::Receiver<LogLine>,
}

impl ScreenLogs {
    /// The next queued line, or `None` when the queue is empty; never waits.
    pub(crate) fn try_recv(&self) -> Option<LogLine> {
        self.receiver.try_recv().ok()
    }

    /// Stop queuing and write what was queued to stderr, where it would have gone unqueued: the
    /// lines the user's own filter passes, the screen having queued the rest for `/log` alone. Done
    /// before an error is printed on the way out, so the lines read in the order they happened.
    pub(crate) fn release(&self) {
        RELAY.set_screen_attached(false);
        for line in self.drain_shown() {
            crate::streams::write_stderr_line(line);
        }
    }

    /// The queued lines the user's filter passes, as stderr would have shown them, emptying the
    /// queue.
    fn drain_shown(&self) -> Vec<String> {
        std::iter::from_fn(|| self.receiver.try_recv().ok())
            .filter(|line| line.shown)
            .map(|line| {
                format!(
                    "{} {:>5} {}: {}",
                    crate::text::format_timestamp(line.at, crate::text::Precision::Seconds),
                    line.level,
                    line.target,
                    line.message
                )
            })
            .collect()
    }
}

impl Drop for ScreenLogs {
    fn drop(&mut self) {
        self.release();
    }
}

/// The layer that hands each event to the screen with its level and message, beside the
/// formatting layer that writes the line for every other host. It is given a wider filter than
/// the user asked for, so `/log` has the lines a `-v` would have shown; the user's own filter
/// decides which of them the transcript draws.
pub(crate) struct ScreenLayer {
    shown: tracing_subscriber::EnvFilter,
    /// Where the lines go: the process's relay, or a private one under test.
    relay: Relay,
}

impl ScreenLayer {
    /// A layer whose lines are drawn on the transcript when `shown` passes them.
    pub(crate) fn new(shown: tracing_subscriber::EnvFilter) -> Self {
        Self {
            shown,
            relay: RELAY.clone(),
        }
    }
}

impl<S> tracing_subscriber::Layer<S> for ScreenLayer
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        context: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if !self.relay.is_screen_attached() {
            return;
        }
        // A record the `log` crate bridged arrives under metadata whose target is `log`, carrying
        // its own target and location as `log.*` fields. The formatting layer names the record's
        // target and hides those fields, and the screen's row reads as its line would.
        use tracing_log::NormalizeEvent as _;
        let normalized = event.normalized_metadata();
        let metadata = normalized.as_ref().unwrap_or(event.metadata());
        let shown = tracing_subscriber::layer::Filter::enabled(&self.shown, metadata, &context);
        let mut fields = Fields::default();
        event.record(&mut fields);
        self.relay.offer(LogLine {
            at: chrono::Utc::now(),
            level: *metadata.level(),
            target: metadata.target().to_string(),
            message: fields.into_message(),
            shown,
        });
    }
}

/// An event's fields as the formatting layer would print them: the message first, every other
/// field after it as `name=value`.
#[derive(Default)]
struct Fields {
    message: String,
    rest: Vec<String>,
}

impl Fields {
    fn into_message(self) -> String {
        let mut out = self.message;
        for field in self.rest {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&field);
        }
        out
    }

    fn record(&mut self, field: &tracing::field::Field, value: String) {
        // The `log.*` fields are a bridged record's target and location: the target names the
        // line already, and the location is the library's own business.
        if field.name().starts_with("log.") {
            return;
        }
        if field.name() == "message" {
            self.message = value;
        } else {
            self.rest.push(format!("{}={value}", field.name()));
        }
    }
}

impl tracing::field::Visit for Fields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.record(field, format!("{value:?}"));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.record(field, value.to_string());
    }
}

impl<'a> MakeWriter<'a> for Relay {
    type Writer = RelayWriter;

    fn make_writer(&'a self) -> Self::Writer {
        RelayWriter {
            screen_attached: Arc::clone(&self.screen_attached),
        }
    }
}

/// Per-write borrow handed back to the tracing formatter. `screen_attached` is read at write time
/// so the routing reflects the live screen state, not whatever it was at `make_writer`.
pub(crate) struct RelayWriter {
    screen_attached: Arc<AtomicBool>,
}

impl Write for RelayWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        // While the screen is up the layer has already handed it the event, and the formatted
        // line has nowhere to go: stderr is the alternate screen.
        if self.screen_attached.load(Ordering::Relaxed) {
            return Ok(buffer.len());
        }
        io::stderr().write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        io::stderr().flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A full queue does not block the layer. The queue is drained only by the screen's frame
    /// loop, so an offer that waited for room could wait on the very task doing the logging.
    #[test]
    fn a_full_queue_does_not_block_a_log_line() {
        let relay = Relay::new();
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        relay.install(sender);
        relay.set_screen_attached(true);
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for message in ["first", "second", "third"] {
                relay.offer(LogLine {
                    at: chrono::Utc::now(),
                    level: tracing::Level::WARN,
                    target: "meka".to_string(),
                    message: message.to_string(),
                    shown: true,
                });
            }
            done.send(()).expect("report");
        });
        finished
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the third line blocked on a queue nobody drains");
        // Held to here: once the receiver is gone the channel answers `Disconnected`, which never
        // blocks either, and the test would pass without a full queue.
        drop(receiver);
    }

    /// A queue handed back to stderr writes the lines the user's filter shows, not the ones the
    /// screen queued at a finer level for `/log`.
    #[test]
    fn a_released_queue_writes_only_what_the_filter_shows() {
        let (sender, receiver) = std::sync::mpsc::sync_channel(4);
        let logs = ScreenLogs { receiver };
        for (message, shown) in [("kept", true), ("hidden", false)] {
            sender
                .send(LogLine {
                    at: chrono::Utc::now(),
                    level: tracing::Level::INFO,
                    target: "meka".to_string(),
                    message: message.to_string(),
                    shown,
                })
                .expect("queue");
        }
        let lines = logs.drain_shown();
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(lines[0].ends_with("meka: kept"), "{lines:?}");
    }

    /// An event's fields read as the formatting layer prints them: the message, then the rest.
    #[test]
    fn an_event_s_fields_become_its_message() {
        use tracing::field::Visit;
        let mut fields = Fields::default();
        fields.record_str(&field("message"), "stream error");
        fields.record_debug(&field("attempt"), &2);
        assert_eq!(fields.into_message(), "stream error attempt=2");
    }

    /// A record the `log` crate bridged is the row the formatting layer's line would be: named by
    /// the record's own target, read against the user's filter under that target, and without
    /// the `log.*` fields that carry it.
    #[test]
    fn a_bridged_log_record_reads_as_its_line_would() {
        use tracing_subscriber::layer::SubscriberExt as _;
        let relay = Relay::new();
        let (sender, receiver) = std::sync::mpsc::sync_channel(4);
        relay.install(sender);
        relay.set_screen_attached(true);
        let layer = ScreenLayer {
            shown: tracing_subscriber::EnvFilter::new("warn,arboard=error"),
            relay: relay.clone(),
        };
        let _scoped = tracing::subscriber::set_default(tracing_subscriber::registry().with(layer));
        tracing_log::format_trace(
            &tracing_log::log::Record::builder()
                .level(tracing_log::log::Level::Warn)
                .target("arboard::platform::linux::x11")
                .module_path(Some("arboard::platform::linux::x11"))
                .file(Some("x11.rs"))
                .line(Some(791))
                .args(format_args!("Could not hand the clipboard contents over."))
                .build(),
        )
        .expect("the record reaches the thread's subscriber");
        let line = receiver.try_recv().expect("the record is a line");
        assert_eq!(line.level, tracing::Level::WARN);
        assert_eq!(line.target, "arboard::platform::linux::x11");
        assert_eq!(line.message, "Could not hand the clipboard contents over.");
        assert!(
            !line.shown,
            "`arboard=error` is read against the record's target, not against `log`"
        );
    }

    fn field(name: &'static str) -> tracing::field::Field {
        use tracing::{callsite::Callsite, metadata::Kind};
        struct Site;
        static META: tracing::Metadata<'static> = tracing::Metadata::new(
            "test",
            "test",
            tracing::Level::INFO,
            None,
            None,
            None,
            tracing::field::FieldSet::new(
                &["message", "attempt"],
                tracing::callsite::Identifier(&SITE),
            ),
            Kind::EVENT,
        );
        static SITE: Site = Site;
        impl Callsite for Site {
            fn set_interest(&self, _: tracing::subscriber::Interest) {}

            fn metadata(&self) -> &tracing::Metadata<'_> {
                &META
            }
        }
        META.fields()
            .field(name)
            .expect("a field of the test's set")
    }

    /// With no screen attached (every non-interactive command) the whole line reaches stderr, none
    /// of it swallowed by a screen that is not there.
    #[test]
    fn a_host_without_a_screen_writes_to_stderr() {
        let relay = Relay::new();
        relay.set_screen_attached(false);
        let line = b"WARN config.toml could not be read\n";
        let written = relay
            .make_writer()
            .write(line)
            .expect("plain stderr is the fallback for `meka mcp list` and friends");
        assert_eq!(
            written,
            line.len(),
            "nothing is dropped on the way to stderr"
        );
    }
}
