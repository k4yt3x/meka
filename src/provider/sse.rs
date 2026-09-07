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
use crate::error::{MekaError, Result};

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
    /// failure; reporting it as one sent an abandoned turn back through the retry path.
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
    let response = super::succeeded(response, what).await?;
    let mut event_stream = response.bytes_stream().eventsource();
    loop {
        tokio::select! {
            _ = cancellation.cancelled() => {
                return Err(MekaError::Interrupted);
            }
            event = tokio::time::timeout(STREAM_IDLE_TIMEOUT, event_stream.next()) => {
                // Bounds silence, not the turn: a model still emitting deltas resets this on every
                // one. Without it a connection that died without an RST left the turn parked on a
                // socket that would never speak again, for as long as the process ran.
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
