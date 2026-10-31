//! The request-size budget every backend applies before a send.
//!
//! The oldest tool-result images are redacted until the serialized body fits the profile's
//! `max_request_bytes`, and a body that still does not fit is refused as
//! [`MekaError::RequestTooLarge`], which the turn answers by degrading its own attachments. The
//! Anthropic backends run this against a default ceiling, Anthropic's cap less headroom; the OpenAI
//! backends run it only when the profile states one, because their endpoints' caps are the
//! endpoints' own facts and meka will not guess at a number it would then send against. Without
//! this on those backends, a window that has accumulated a few image reads ships a body of tens
//! of megabytes on every turn with nothing to say so.

use std::borrow::Cow;

use crate::{
    conversation::{ContentBlock, Message, ToolResultContent},
    error::{MekaError, Result},
    image::RedactedImage,
};

/// The most images the Messages API takes in one request. Fails toward sending less: a backend
/// behind a `base_url` that takes more loses its oldest images early, and one that takes fewer
/// refuses the request as it would without this.
pub(super) const ANTHROPIC_MAX_REQUEST_IMAGES: usize = 100;

/// The images `messages` carries, as attachments and in tool results.
fn image_count(messages: &[Message]) -> usize {
    messages
        .iter()
        .flat_map(|message| &message.content)
        .map(|block| match block {
            ContentBlock::Image { .. } => 1,
            ContentBlock::ToolResult { content, .. } => content
                .iter()
                .filter(|item| matches!(item, ToolResultContent::Image { .. }))
                .count(),
            _ => 0,
        })
        .sum()
}

/// How far below the ceiling a redaction stops, so the next several turns do not re-trigger it.
/// Mirrors Claude Code's `apiMicrocompact` watermark (180k → 140k = ~78% of trigger): a stable
/// cache prefix between redactions matters more than minimum-impact redaction per event. For the
/// Anthropic default of 30 MiB this lands at 24 MiB.
pub(super) const REDACTION_HEADROOM_BYTES: usize = 6 * crate::text::MIB;

/// Where a redaction stops for a given ceiling: [`REDACTION_HEADROOM_BYTES`] below it, capped at
/// half the ceiling so a small one still leaves something to send.
pub(super) fn redaction_target(max_request_bytes: usize) -> usize {
    let headroom = REDACTION_HEADROOM_BYTES.min(max_request_bytes / 2);
    max_request_bytes - headroom
}

pub(super) use crate::conversation::IMAGE_REDACTION_PLACEHOLDER;

/// The request body as the bytes that go on the wire, which is what the budget measures.
pub(super) fn serialize_body(body: &serde_json::Value) -> Result<String> {
    serde_json::to_string(body)
        .map_err(|error| MekaError::Provider(format!("failed to serialize body: {error}")))
}

/// Stats from a single [`redact_oldest_images`] invocation. Returned to callers so they can surface
/// a user-visible advisory and increment a per-session redaction counter.
#[derive(Debug, Clone, Default)]
pub(super) struct RedactionStats {
    pub(crate) images_redacted: usize,
    pub(crate) bytes_freed: usize,
    /// Where each redacted image sat, tail-relative, for the conversation to record.
    pub(crate) positions: Vec<RedactedImage>,
}

/// Walk `messages` oldest-first and replace `ToolResultContent::Image` payloads with
/// [`IMAGE_REDACTION_PLACEHOLDER`] until at least `bytes_to_drop` base64 bytes have been removed.
/// The LAST message is never touched: it is the request the model is about to answer, and on the
/// Claude backends it carries the moving `cache_control` breakpoint, which disturbing would
/// invalidate the cache for the new turn unnecessarily.
///
/// Returns `Cow::Borrowed` if no work was needed (`bytes_to_drop == 0`). Otherwise returns
/// `Cow::Owned` with whatever redaction was possible. Even when the budget couldn't be met, the
/// cloned messages are still returned so the caller can re-serialize and decide whether the body
/// fits.
pub(super) fn redact_oldest_images(
    messages: &[Message],
    bytes_to_drop: usize,
) -> (Cow<'_, [Message]>, RedactionStats) {
    if bytes_to_drop == 0 {
        return (Cow::Borrowed(messages), RedactionStats::default());
    }
    redact_oldest_images_until(messages, |stats| stats.bytes_freed >= bytes_to_drop)
}

/// [`redact_oldest_images`] by count: the oldest images until `images_to_drop` are gone.
fn redact_oldest_image_count(
    messages: &[Message],
    images_to_drop: usize,
) -> (Cow<'_, [Message]>, RedactionStats) {
    if images_to_drop == 0 {
        return (Cow::Borrowed(messages), RedactionStats::default());
    }
    redact_oldest_images_until(messages, |stats| stats.images_redacted >= images_to_drop)
}

/// The walk under both: oldest first, the last message untouched, stopping once `done` says so.
fn redact_oldest_images_until(
    messages: &[Message],
    done: impl Fn(&RedactionStats) -> bool,
) -> (Cow<'_, [Message]>, RedactionStats) {
    if messages.len() <= 1 {
        return (Cow::Borrowed(messages), RedactionStats::default());
    }

    let mut redacted: Vec<Message> = messages.to_vec();
    let total = redacted.len();
    let last = total - 1;
    let mut stats = RedactionStats::default();

    'outer: for (index, message) in redacted.iter_mut().take(last).enumerate() {
        let from_end = total - index;
        for (block_index, block) in message.content.iter_mut().enumerate() {
            match block {
                ContentBlock::ToolResult { content, .. } => {
                    for (item_index, item) in content.iter_mut().enumerate() {
                        if let ToolResultContent::Image { source } = item {
                            stats.bytes_freed = stats
                                .bytes_freed
                                .saturating_add(source.base64_data().map_or(0, str::len));
                            stats.images_redacted = stats.images_redacted.saturating_add(1);
                            stats.positions.push(RedactedImage {
                                from_end,
                                block: block_index,
                                item: Some(item_index),
                            });
                            *item = ToolResultContent::Text {
                                text: IMAGE_REDACTION_PLACEHOLDER.to_string(),
                            };
                            if done(&stats) {
                                break 'outer;
                            }
                        }
                    }
                }
                // Input images (ACP @-mentions) count toward the same 32 MiB cap; collapse them to
                // the placeholder text just like tool-result images.
                ContentBlock::Image { source } => {
                    let freed = source.base64_data().map_or(0, str::len);
                    stats.bytes_freed = stats.bytes_freed.saturating_add(freed);
                    stats.images_redacted = stats.images_redacted.saturating_add(1);
                    stats.positions.push(RedactedImage {
                        from_end,
                        block: block_index,
                        item: None,
                    });
                    *block = ContentBlock::Text {
                        text: IMAGE_REDACTION_PLACEHOLDER.to_string(),
                    };
                    if done(&stats) {
                        break 'outer;
                    }
                }
                _ => {}
            }
        }
    }

    (Cow::Owned(redacted), stats)
}

/// Build a request body that fits `max_request_bytes`, redacting the oldest tool-result images when
/// the first attempt does not.
///
/// `build` takes a `messages` slice and returns the serialized JSON. It is called once on the
/// messages as given and, if oversized, a second time on the redacted set.
///
/// Returns the serialized body plus an optional [`crate::frontend::Notice`]: on a redaction pass,
/// the notice describes what was dropped, for the agent to forward to the frontend and count
/// against the session through the [`crate::stats::Redaction`] it carries. On the happy path the
/// notice is `None`.
pub(super) fn fit_body_to_budget<F>(
    messages: &[Message],
    max_request_bytes: usize,
    // The most images a request may carry, on a wire whose API counts them.
    max_images: Option<usize>,
    mut build: F,
) -> Result<(String, Option<crate::frontend::Notice>)>
where
    F: FnMut(&[Message]) -> Result<String>,
{
    let images_to_drop = max_images.map_or(0, |max| image_count(messages).saturating_sub(max));
    let (counted, mut stats) = redact_oldest_image_count(messages, images_to_drop);
    let body_json = build(counted.as_ref())?;

    if body_json.len() <= max_request_bytes && stats.images_redacted == 0 {
        return Ok((body_json, None));
    }

    let body_json = if body_json.len() <= max_request_bytes {
        body_json
    } else {
        let bytes_to_drop = body_json.len() - redaction_target(max_request_bytes);
        let (redacted, by_size) = redact_oldest_images(counted.as_ref(), bytes_to_drop);
        stats.images_redacted += by_size.images_redacted;
        stats.bytes_freed += by_size.bytes_freed;
        stats.positions.extend(by_size.positions);
        build(redacted.as_ref())?
    };

    // `RequestTooLarge`, meka's own refusal: the request never left the process, so publishing it
    // as a provider failure would tell a caller to look in the server log for a provider response
    // that does not exist. Not `Provider` either, and for a reason the turn depends on: the
    // refusal is deterministic on the body and about its newest content, since every older image
    // was just redacted, which is exactly what the turn's degrade-and-retry exists to strip. As
    // `Provider` it would read as an outage, nothing would be degraded, and the session would stay
    // unusable until a rewind because every later request carries the same attachments.
    if body_json.len() > max_request_bytes {
        return Err(MekaError::RequestTooLarge(format!(
            "request body is {} after redacting old tool-result images, over this profile's \
             `max_request_bytes` of {}; run `/compact`",
            crate::text::format_size(body_json.len()),
            crate::text::format_size(max_request_bytes),
        )));
    }

    let notice_text = format!(
        "Redacted {} old image{} (~{} freed).",
        stats.images_redacted,
        if stats.images_redacted == 1 { "" } else { "s" },
        crate::text::format_size(stats.bytes_freed),
    );
    tracing::warn!(
        "redacted {images_redacted} old tool-result image(s); body now {size}",
        images_redacted = stats.images_redacted,
        size = crate::text::format_size(body_json.len()),
    );
    Ok((
        body_json,
        Some(
            crate::frontend::Notice::info(notice_text).reporting(crate::stats::Redaction {
                images: stats.images_redacted as u64,
                bytes: stats.bytes_freed as u64,
                positions: stats.positions,
            }),
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::Role;

    fn message(text: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    /// A body that will not fit is meka's own refusal, not a provider failure.
    ///
    /// The distinction is the whole point of the variant: nothing was sent, so a host publishing
    /// this as a provider error sends its caller to a server log for an upstream response that
    /// does not exist. The sentence carries the ceiling and the remedy, so it is the answer rather
    /// than a pointer to one.
    #[test]
    fn a_body_over_the_ceiling_is_refused_as_meka_s_own() {
        let messages = vec![message("first"), message(&"x".repeat(4096))];
        let error = fit_body_to_budget(&messages, 64, None, |messages| {
            Ok(serde_json::to_string(&serde_json::json!(
                messages.iter().map(|_| "x".repeat(64)).collect::<Vec<_>>()
            ))
            .unwrap_or_default())
        })
        .expect_err("a body over the ceiling must be refused");
        assert!(
            matches!(error, MekaError::RequestTooLarge(_)),
            "meka's own ceiling is not a provider failure: {error:?}"
        );
        let sentence = error.to_string();
        assert!(
            sentence.contains("max_request_bytes") && sentence.contains("/compact"),
            "the refusal must name the ceiling and the remedy: {sentence}"
        );
    }

    /// The happy path still returns the body, so the refusal above is not simply "always refuse".
    #[test]
    fn a_body_that_fits_is_returned_unchanged() {
        let messages = vec![message("first")];
        let (body, notice) =
            fit_body_to_budget(&messages, 4096, None, |_| Ok("{}".to_string())).expect("fits");
        assert_eq!(body, "{}");
        assert!(notice.is_none(), "nothing was redacted");
    }

    /// A request over the API's image count loses its oldest images until it is at the count,
    /// however small its body, and says so; the message being answered keeps its own.
    #[test]
    fn a_request_over_the_image_count_loses_its_oldest_images() {
        let image = || ContentBlock::Image {
            source: crate::image::ImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: "aGk=".to_string(),
            },
        };
        let messages = vec![
            Message {
                role: Role::User,
                content: vec![image(), image()],
            },
            message("between"),
            Message {
                role: Role::User,
                content: vec![image()],
            },
        ];
        let mut sent = Vec::new();
        let (_, notice) = fit_body_to_budget(&messages, 1 << 20, Some(2), |messages| {
            sent = messages.to_vec();
            Ok("{}".to_string())
        })
        .expect("fits");
        assert_eq!(image_count(&sent), 2, "{sent:?}");
        assert!(
            matches!(sent[0].content[0], ContentBlock::Text { .. }),
            "the oldest image goes first: {sent:?}"
        );
        assert!(matches!(sent[2].content[0], ContentBlock::Image { .. }));
        let notice = notice.expect("a redaction is reported");
        assert!(
            notice.text.starts_with("Redacted 1 old image "),
            "{}",
            notice.text
        );
    }
}
