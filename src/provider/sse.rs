//! The SSE read loop every streaming driver shares.
//!
//! Three protocols decode their frames differently and agree on everything around them: a stall is
//! bounded by [`STREAM_IDLE_TIMEOUT`] and is a transport failure, cancellation is
//! [`MekaError::Interrupted`], a caller that hangs up ends the read without an error, and every
//! failure after the response head is announced on the channel before it is returned. That last one
//! is a contract with the agent's stream consumer, which logs the announcement and reports the
//! typed error the driver returns, so the classification (retryable or not) survives the channel. A
//! non-2xx status and an interrupt return without an announcement: nothing has been streamed, so
//! there is no thinking indicator to close and nothing for the consumer to log.

use eventsource_stream::Eventsource;
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::{STREAM_IDLE_TIMEOUT, StreamEvent};
use crate::error::{MAX_REPLY_BYTES, MekaError, Result};

/// What a protocol says after one frame.
pub(crate) enum Step {
    Continue,
    /// The message has ended; nothing after this frame is part of it.
    Finished,
    /// A send failed: nobody is listening any more.
    ReceiverGone,
}

/// How a read ended when nothing failed.
pub(crate) enum End {
    /// The protocol saw the message end.
    Finished,
    /// The byte stream stopped before the protocol saw the message end. Whether that is a cut
    /// message is the protocol's to say: one that has already seen a stop reason is complete.
    Ended,
    /// The caller left. There is no half-written answer for anyone to act on, so this is not a
    /// failure; reported as one, an abandoned turn goes back through the retry path.
    ReceiverGone,
}

/// One protocol's frame decoding, kept between frames.
#[async_trait::async_trait]
pub(crate) trait Protocol: Send {
    /// Handle one decoded frame. An `Err` is announced on the channel by the loop before it is
    /// returned, so an implementation returns it bare.
    async fn frame(
        &mut self,
        event: eventsource_stream::Event,
        event_sender: &mpsc::Sender<StreamEvent>,
    ) -> Result<Step>;
}

/// Announce `message` on the channel, then hand back the error to return for it.
pub(crate) async fn stream_error(
    event_sender: &mpsc::Sender<StreamEvent>,
    message: String,
) -> MekaError {
    if event_sender
        .send(StreamEvent::Error(message.clone()))
        .await
        .is_err()
    {
        tracing::trace!("stream event receiver dropped");
    }
    MekaError::StreamError(message)
}

/// A frame's `data` as JSON, or `None` with a warning for one that is not, which a protocol skips.
pub(crate) fn frame_json(what: &str, data: &str) -> Option<serde_json::Value> {
    match serde_json::from_str(data) {
        Ok(data) => Some(data),
        Err(error) => {
            tracing::warn!("failed to parse {what} SSE data: {error}");
            None
        }
    }
}

/// Read `response` as SSE to its end, handing each frame to `protocol`. `what` names the protocol
/// in messages.
pub(crate) async fn drive<P: Protocol>(
    response: reqwest::Response,
    what: &str,
    event_sender: &mpsc::Sender<StreamEvent>,
    cancellation: &CancellationToken,
    protocol: &mut P,
) -> Result<End> {
    drive_within(
        response,
        what,
        event_sender,
        cancellation,
        protocol,
        MAX_REPLY_BYTES,
    )
    .await
}

/// [`drive`] with one event bounded at `max_event_bytes`.
async fn drive_within<P: Protocol>(
    response: reqwest::Response,
    what: &str,
    event_sender: &mpsc::Sender<StreamEvent>,
    cancellation: &CancellationToken,
    protocol: &mut P,
    max_event_bytes: usize,
) -> Result<End> {
    let response = super::succeeded(
        response,
        what,
        crate::error::ProviderRequest::Completion,
        cancellation,
    )
    .await?;
    let mut event_stream = bounded_events(response.bytes_stream(), max_event_bytes).eventsource();
    loop {
        tokio::select! {
            _ = cancellation.cancelled() => {
                return Err(MekaError::Interrupted);
            }
            event = tokio::time::timeout(STREAM_IDLE_TIMEOUT, event_stream.next()) => {
                // Bounds silence, not the turn: a model still emitting deltas resets this on every
                // one. Without it a connection that dies without an RST leaves the turn parked on
                // a socket that will never speak again, for as long as the process runs.
                let Ok(event) = event else {
                    let message = format!(
                        "idle timeout waiting for a {what} SSE event after {}s",
                        STREAM_IDLE_TIMEOUT.as_secs()
                    );
                    return Err(stream_error(event_sender, message).await);
                };
                let Some(event) = event else {
                    return Ok(End::Ended);
                };
                let event = match event {
                    Ok(event) => event,
                    // Permanent rather than a stream error: the next attempt reads the same event.
                    Err(eventsource_stream::EventStreamError::Transport(
                        BodyError::EventTooLarge,
                    )) => {
                        let message = format!(
                            "refused a {what} SSE event larger than {}",
                            crate::text::format_size(max_event_bytes)
                        );
                        if event_sender
                            .send(StreamEvent::Error(message.clone()))
                            .await
                            .is_err()
                        {
                            tracing::trace!("stream event receiver dropped");
                        }
                        return Err(MekaError::Provider(message));
                    }
                    Err(error) => return Err(stream_error(event_sender, error.to_string()).await),
                };
                match protocol.frame(event, event_sender).await {
                    Ok(Step::Continue) => {}
                    Ok(Step::Finished) => return Ok(End::Finished),
                    Ok(Step::ReceiverGone) => return Ok(End::ReceiverGone),
                    Err(error) => {
                        if event_sender
                            .send(StreamEvent::Error(error.to_string()))
                            .await
                            .is_err()
                        {
                            tracing::trace!("stream event receiver dropped");
                        }
                        return Err(error);
                    }
                }
            }
        }
    }
}

/// Why a byte of a streamed body is withheld from the SSE decoder.
#[derive(Debug, thiserror::Error)]
enum BodyError<E: std::fmt::Display + std::fmt::Debug> {
    /// The connection failed; shown as the transport shows it, so the decoder's error reads the
    /// same with the bound in between as without.
    #[error("{0}")]
    Transport(E),
    /// One event ran past the bound with no blank line to end it.
    #[error("event too large")]
    EventTooLarge,
}

/// The bytes of a streamed body, failing once an event runs past `max_bytes` without the blank
/// line that ends it. The decoder buffers an event whole until that line, so a reply with no line
/// end would otherwise be held until the connection closed, whatever its size.
fn bounded_events<S, B, E>(
    body: S,
    max_bytes: usize,
) -> impl futures::Stream<Item = std::result::Result<B, BodyError<E>>>
where
    S: futures::Stream<Item = std::result::Result<B, E>>,
    B: AsRef<[u8]>,
    E: std::fmt::Display + std::fmt::Debug,
{
    body.scan(EventBound::new(max_bytes), |bound, chunk| {
        futures::future::ready(Some(match chunk {
            Ok(bytes) if bound.admit(bytes.as_ref()) => Ok(bytes),
            Ok(_) => Err(BodyError::EventTooLarge),
            Err(error) => Err(BodyError::Transport(error)),
        }))
    })
}

/// Bytes since the last blank line, and the line state that recognizes the next one.
///
/// An event ends at an empty line: two line ends in a row, with CRLF read as one, which is every
/// blank line the SSE grammar admits (CR, LF and CRLF all end a line).
struct EventBound {
    max_bytes: usize,
    since_boundary: usize,
    at_line_start: bool,
    after_carriage_return: bool,
}

impl EventBound {
    fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            since_boundary: 0,
            at_line_start: true,
            after_carriage_return: false,
        }
    }

    /// Count `bytes` toward the open event; `false` once it is past the bound.
    fn admit(&mut self, bytes: &[u8]) -> bool {
        for &byte in bytes {
            match byte {
                // The second half of a CRLF: the line ended at the CR, which was counted.
                b'\n' if self.after_carriage_return => {
                    self.after_carriage_return = false;
                    continue;
                }
                b'\r' | b'\n' => {
                    if self.at_line_start {
                        self.since_boundary = 0;
                    } else {
                        self.since_boundary += 1;
                    }
                    self.at_line_start = true;
                    self.after_carriage_return = byte == b'\r';
                }
                _ => {
                    self.since_boundary += 1;
                    self.at_line_start = false;
                    self.after_carriage_return = false;
                }
            }
            if self.since_boundary > self.max_bytes {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The count restarts at every blank line the grammar admits, and nowhere else: a CRLF is one
    /// line end, so a lone LF after a CR opens nothing.
    #[test]
    fn an_event_is_counted_between_blank_lines() {
        let mut bound = EventBound::new(12);
        assert!(bound.admit(b"data: 1234\n\n"));
        assert!(bound.admit(b"data: 5678\r\n\r\n"));
        assert!(bound.admit(b"data: 9\r\r"));
        assert!(bound.admit(b"data: 1234"));
        assert!(!bound.admit(b"567"), "a line past the bound with no end");

        let mut split = EventBound::new(4);
        assert!(split.admit(b"ab\r\n"));
        assert!(
            split.admit(b"\r\n"),
            "the blank line arrives in its own chunk"
        );
        assert!(split.admit(b"abcd"));
        assert!(!split.admit(b"e"));
    }

    struct Discard;

    #[async_trait::async_trait]
    impl Protocol for Discard {
        async fn frame(
            &mut self,
            _event: eventsource_stream::Event,
            _event_sender: &mpsc::Sender<StreamEvent>,
        ) -> Result<Step> {
            Ok(Step::Continue)
        }
    }

    /// An event past the bound ends the stream with a permanent error, announced on the channel
    /// like every other failure after the head, rather than a stream error the turn would retry.
    #[tokio::test]
    async fn an_oversized_event_ends_the_stream_for_good() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("address");
        let peer = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut head = Vec::new();
            let mut chunk = [0u8; 1024];
            while !head.ends_with(b"\r\n\r\n") {
                let read = socket.read(&mut chunk).await.expect("the request arrives");
                assert!(read > 0, "the client closed the connection");
                head.extend_from_slice(&chunk[..read]);
            }
            let mut raw = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                            transfer-encoding: chunked\r\n\r\n806\r\ndata: "
                .to_vec();
            raw.extend(std::iter::repeat_n(b'x', 2048));
            raw.extend_from_slice(b"\r\n0\r\n\r\n");
            socket.write_all(&raw).await.expect("answer");
        });

        let response = reqwest::Client::new()
            .get(format!("http://{address}/"))
            .send()
            .await
            .expect("the head arrives");
        let (event_sender, mut events) = mpsc::channel(8);
        let Err(error) = drive_within(
            response,
            "test",
            &event_sender,
            &CancellationToken::new(),
            &mut Discard,
            1024,
        )
        .await
        else {
            panic!("an oversized event is refused");
        };
        peer.await.expect("the peer finished");

        assert!(matches!(error, MekaError::Provider(_)), "{error}");
        assert!(error.to_string().contains("1.0 KiB"), "{error}");
        assert!(
            matches!(events.try_recv(), Ok(StreamEvent::Error(_))),
            "the failure is announced on the channel first"
        );
    }
}
