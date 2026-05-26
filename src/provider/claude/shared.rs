//! Helpers shared by [`super::api::ClaudeApiProvider`] and [`super::oauth::ClaudeOAuthProvider`].
//! Everything in this module is independent of the authentication scheme: message/tool conversion
//! to the Claude wire format, SSE streaming, response parsing, per-model capability detection, and
//! the thinking-override helper.

use std::{borrow::Cow, sync::atomic::AtomicI8};

use eventsource_stream::Eventsource;
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    error::{AgshError, Result},
    provider::{
        ContentBlock, Message, Role, StopReason, StreamEvent, TokenUsage, ToolDefinition,
        ToolResultContent,
    },
};

/// Anthropic's hard request-body cap is 32 MiB; we reserve ~2 MiB headroom for headers, URL,
/// attestation patches, and serialization slack. Bodies above this threshold are reactively shrunk
/// by [`redact_oldest_images`] before they're posted.
pub(super) const MAX_REQUEST_BYTES: usize = 30 * 1024 * 1024;

/// When redaction fires, drop the body to roughly this size — leaves a ~6 MiB buffer below
/// [`MAX_REQUEST_BYTES`] so the next several turns don't re-trigger redaction. Mirrors Claude
/// Code's `apiMicrocompact` watermark (180k → 140k = ~78% of trigger). Stable cache prefix between
/// redactions matters more than minimum-impact redaction per event.
pub(super) const REDACTION_TARGET_BYTES: usize = 24 * 1024 * 1024;

/// Placeholder text that replaces a `ToolResultContent::Image` payload when the request body would
/// otherwise exceed [`MAX_REQUEST_BYTES`].
pub(super) const IMAGE_REDACTION_PLACEHOLDER: &str = "[image redacted to fit request size budget]";

/// Anthropic accepts up to 8000 px per axis on a *single*-image request, but rejects anything over
/// 2000 px on either axis once the request contains more than one image. We always downscale to fit
/// so a session can freely accumulate images without tripping the multi-image cap. This is enforced
/// at the Claude provider layer only — non-Claude providers don't need it (and shouldn't pay the
/// resize cost).
pub(super) const MAX_IMAGE_DIMENSION_PX: u32 = 2000;

/// Extract a `TokenUsage` from an Anthropic `usage` object. Used by both the non-streaming response
/// parser and the SSE driver — Anthropic emits the same shape (`input_tokens`, `output_tokens`,
/// `cache_creation_input_tokens`, `cache_read_input_tokens`) in both places. Missing fields default
/// to 0 (older API responses, or providers that don't surface cache stats).
pub(super) fn parse_usage_object(usage: &serde_json::Value) -> TokenUsage {
    let field = |key: &str| usage.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
    TokenUsage {
        input_tokens: field("input_tokens"),
        output_tokens: field("output_tokens"),
        cache_creation_input_tokens: field("cache_creation_input_tokens"),
        cache_read_input_tokens: field("cache_read_input_tokens"),
    }
}

/// Resolves the effective thinking state given the override atomic's raw value (`-1` = unset, `0` =
/// forced off, `1` = forced on) and the configured default. Kept separate from the atomic itself so
/// the providers can own their own `AtomicI8` without duplicating the branching logic.
pub(super) fn is_thinking_enabled(override_raw: i8, default: bool) -> bool {
    match override_raw {
        0 => false,
        1 => true,
        _ => default,
    }
}

/// Convenience wrapper that loads the atomic with relaxed ordering and applies
/// [`is_thinking_enabled`]. Most callers want this form.
pub(super) fn resolve_thinking_enabled(override_atomic: &AtomicI8, default: bool) -> bool {
    is_thinking_enabled(
        override_atomic.load(std::sync::atomic::Ordering::Relaxed),
        default,
    )
}

/// Mirrors Claude Code's `modelSupportsAdaptiveThinking` (`utils/thinking.ts:113-144`): explicit
/// allowlist for `opus-4-6` / `sonnet-4-6`, explicit deny for any other named opus/sonnet/haiku
/// (covers Claude 4.0 / 4.5 and Haiku 4.5), default-true for unknown 1P model strings.
pub(super) fn model_supports_adaptive_thinking(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    if lower.contains("opus-4-6") || lower.contains("sonnet-4-6") {
        return true;
    }
    if lower.contains("opus") || lower.contains("sonnet") || lower.contains("haiku") {
        return false;
    }
    true
}

pub(super) fn model_is_haiku(model: &str) -> bool {
    model.to_ascii_lowercase().contains("haiku")
}

/// Insert the `max_tokens` + `thinking` fields shared by both Claude providers' request bodies.
/// Adaptive-thinking models get a fixed 64k ceiling; others get `max(budget*2, 32k)` with an
/// explicit budget; thinking-off uses a flat 32k.
pub(super) fn insert_thinking_fields(
    body: &mut serde_json::Map<String, serde_json::Value>,
    thinking_enabled: bool,
    model: &str,
    budget_tokens: u64,
) {
    if thinking_enabled {
        if model_supports_adaptive_thinking(model) {
            body.insert("max_tokens".to_string(), serde_json::json!(64_000));
            body.insert(
                "thinking".to_string(),
                serde_json::json!({ "type": "adaptive" }),
            );
        } else {
            let max_tokens = std::cmp::max(budget_tokens * 2, 32_000);
            body.insert("max_tokens".to_string(), serde_json::json!(max_tokens));
            body.insert(
                "thinking".to_string(),
                serde_json::json!({
                    "type": "enabled",
                    "budget_tokens": budget_tokens
                }),
            );
        }
    } else {
        body.insert("max_tokens".to_string(), serde_json::json!(32_000));
    }
}

/// Mirrors Claude Code's `modelSupportsThinking` (and the equivalent
/// `modelSupportsISP` / `modelSupportsContextManagement`) on the 1P API:
/// any Claude 4+ model. Claude-3.x is excluded.
pub(super) fn model_supports_modern_features(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("claude") && !lower.contains("claude-3-")
}

/// Mirrors Claude Code's `modelSupportsEffort` (`utils/effort.ts:23-49`):
/// `opus-4-6` / `sonnet-4-6` allowlist, explicit deny for other named
/// opus/sonnet/haiku, default-true for unknown model strings (agsh is 1P).
pub(super) fn model_supports_effort(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    if lower.contains("opus-4-6") || lower.contains("sonnet-4-6") {
        return true;
    }
    if lower.contains("opus") || lower.contains("sonnet") || lower.contains("haiku") {
        return false;
    }
    true
}

pub(super) fn convert_messages_to_claude_content(messages: &[Message]) -> Vec<serde_json::Value> {
    let message_count = messages.len();
    let mut claude_messages: Vec<serde_json::Value> = messages
        .iter()
        .enumerate()
        .map(|(message_index, message)| {
            let role = match message.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };

            let is_last_message = message_index + 1 == message_count;
            let block_count = message.content.len();

            let content: Vec<serde_json::Value> = message
                .content
                .iter()
                .enumerate()
                .map(|(block_index, block)| {
                    let mut value = match block {
                        ContentBlock::Text { text } => {
                            serde_json::json!({
                                "type": "text",
                                "text": text,
                            })
                        }
                        ContentBlock::ToolUse { id, name, input } => {
                            serde_json::json!({
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": input,
                            })
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            serde_json::json!({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": content,
                                "is_error": is_error,
                            })
                        }
                        ContentBlock::Thinking {
                            thinking,
                            signature,
                        } => {
                            let mut obj = serde_json::json!({
                                "type": "thinking",
                                "thinking": thinking
                            });
                            if let Some(sig) = signature {
                                obj["signature"] = serde_json::json!(sig);
                            }
                            obj
                        }
                    };

                    if is_last_message
                        && block_index + 1 == block_count
                        && let Some(obj) = value.as_object_mut()
                    {
                        obj.insert(
                            "cache_control".to_string(),
                            serde_json::json!({"type": "ephemeral", "ttl": "1h"}),
                        );
                    }

                    value
                })
                .collect();

            serde_json::json!({
                "role": role,
                "content": content,
            })
        })
        .collect();

    // Strip trailing thinking blocks from the last assistant message (Claude API requirement).
    if let Some(last_assistant) = claude_messages
        .iter_mut()
        .rev()
        .find(|msg| msg.get("role").and_then(|r| r.as_str()) == Some("assistant"))
        && let Some(content) = last_assistant
            .get_mut("content")
            .and_then(|c| c.as_array_mut())
    {
        while content
            .last()
            .and_then(|b| b.get("type"))
            .and_then(|t| t.as_str())
            == Some("thinking")
        {
            content.pop();
        }
        if content.is_empty() {
            content.push(serde_json::json!({
                "type": "text",
                "text": "[No message content]"
            }));
        }
    }

    claude_messages
}

pub(super) fn convert_tools_to_claude_tools(tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
    let tool_count = tools.len();
    tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            let mut schema = serde_json::json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            });
            if index + 1 == tool_count
                && let Some(obj) = schema.as_object_mut()
            {
                obj.insert(
                    "cache_control".to_string(),
                    serde_json::json!({"type": "ephemeral", "ttl": "1h"}),
                );
            }
            schema
        })
        .collect()
}

pub(super) fn parse_claude_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        // Claude does not include the refusal text alongside the streaming `stop_reason` delta; the
        // model's text content is what the user sees as the refusal. Surface an empty refusal
        // payload — the assistant message blocks carry the human-readable explanation already.
        "refusal" => StopReason::Refusal(String::new()),
        other => StopReason::Unknown(other.to_string()),
    }
}

pub(super) fn parse_non_streaming_response(
    response: &serde_json::Value,
) -> Result<(Message, StopReason, TokenUsage)> {
    let stop_reason_str = response
        .get("stop_reason")
        .and_then(|reason| reason.as_str())
        .unwrap_or("end_turn");

    let stop_reason = parse_claude_stop_reason(stop_reason_str);

    let token_usage = response
        .get("usage")
        .map(parse_usage_object)
        .unwrap_or_default();

    let content_array = response
        .get("content")
        .and_then(|content| content.as_array())
        .ok_or_else(|| AgshError::Provider("no content array in response".to_string()))?;

    let mut content_blocks = Vec::new();

    for block in content_array {
        let block_type = block
            .get("type")
            .and_then(|block_type| block_type.as_str())
            .unwrap_or("");

        match block_type {
            "text" => {
                if let Some(text) = block.get("text").and_then(|text| text.as_str()) {
                    content_blocks.push(ContentBlock::Text {
                        text: text.to_string(),
                    });
                }
            }
            "tool_use" => {
                let id = block
                    .get("id")
                    .and_then(|id| id.as_str())
                    .ok_or_else(|| {
                        AgshError::Provider("tool_use block missing 'id' field".to_string())
                    })?
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|name| name.as_str())
                    .ok_or_else(|| {
                        AgshError::Provider("tool_use block missing 'name' field".to_string())
                    })?
                    .to_string();
                let input = block.get("input").cloned().unwrap_or_else(|| {
                    tracing::warn!("missing 'input' in tool_use block");
                    serde_json::json!({})
                });

                content_blocks.push(ContentBlock::ToolUse { id, name, input });
            }
            "thinking" => {
                if let Some(thinking) = block.get("thinking").and_then(|t| t.as_str()) {
                    let signature = block
                        .get("signature")
                        .and_then(|s| s.as_str())
                        .map(|s| s.to_string());
                    content_blocks.push(ContentBlock::Thinking {
                        thinking: thinking.to_string(),
                        signature,
                    });
                }
            }
            "redacted_thinking" => {
                let signature = block
                    .get("signature")
                    .and_then(|s| s.as_str())
                    .map(|s| s.to_string());
                content_blocks.push(ContentBlock::Thinking {
                    thinking: "[redacted]".to_string(),
                    signature,
                });
            }
            _ => {
                tracing::warn!("unknown Claude content block type: {}", block_type);
            }
        }
    }

    Ok((
        Message {
            role: Role::Assistant,
            content: content_blocks,
        },
        stop_reason,
        token_usage,
    ))
}

pub(super) async fn drive_claude_sse_stream(
    response: reqwest::Response,
    event_sender: mpsc::Sender<StreamEvent>,
    cancellation: CancellationToken,
) -> Result<()> {
    let status = response.status();
    if !status.is_success() {
        let response_text = response.text().await.unwrap_or_else(|error| {
            tracing::warn!("failed to read Claude error response body: {}", error);
            String::new()
        });
        return Err(AgshError::Provider(format!(
            "API returned status {}: {}",
            status, response_text
        )));
    }

    let mut event_stream = response.bytes_stream().eventsource();

    let mut current_tool_input = String::new();
    let mut in_tool_use = false;
    let mut in_thinking = false;
    let mut current_thinking_signature: Option<String> = None;

    loop {
        tokio::select! {
            _ = cancellation.cancelled() => {
                return Err(AgshError::Interrupted);
            }
            event = event_stream.next() => {
                let Some(event) = event else {
                    break;
                };

                match event {
                    Ok(event) => {
                        let data: serde_json::Value = match serde_json::from_str(&event.data) {
                            Ok(data) => data,
                            Err(error) => {
                                tracing::warn!("failed to parse SSE data: {}", error);
                                continue;
                            }
                        };

                        match event.event.as_str() {
                            "content_block_start" => {
                                let Some(content_block) = data.get("content_block") else {
                                    continue;
                                };
                                let block_type = content_block
                                    .get("type")
                                    .and_then(|block_type| block_type.as_str())
                                    .unwrap_or("");

                                if block_type == "thinking" {
                                    in_thinking = true;
                                } else if block_type == "redacted_thinking" {
                                    // Emit a stub thinking block so the UI shows something for
                                    // redacted content.
                                    let _ = event_sender
                                        .send(StreamEvent::ThinkingDelta(
                                            "[redacted]".to_string(),
                                        ))
                                        .await;
                                    let signature = content_block
                                        .get("signature")
                                        .and_then(|s| s.as_str())
                                        .map(|s| s.to_string());
                                    let _ = event_sender
                                        .send(StreamEvent::ThinkingComplete { signature })
                                        .await;
                                } else if block_type == "tool_use" {
                                    let id = content_block
                                        .get("id")
                                        .and_then(|id| id.as_str())
                                        .ok_or_else(|| {
                                            AgshError::Provider(
                                                "tool_use block missing 'id' field".to_string(),
                                            )
                                        })?
                                        .to_string();
                                    let name = content_block
                                        .get("name")
                                        .and_then(|name| name.as_str())
                                        .ok_or_else(|| {
                                            AgshError::Provider(
                                                "tool_use block missing 'name' field"
                                                    .to_string(),
                                            )
                                        })?
                                        .to_string();

                                    current_tool_input.clear();
                                    in_tool_use = true;
                                    if event_sender
                                        .send(StreamEvent::ToolUseStart { id, name })
                                        .await
                                        .is_err()
                                    {
                                        tracing::trace!("stream event receiver dropped");
                                        break;
                                    }
                                }
                            }
                            "content_block_delta" => {
                                let Some(delta) = data.get("delta") else {
                                    continue;
                                };
                                let delta_type = delta
                                    .get("type")
                                    .and_then(|delta_type| delta_type.as_str())
                                    .unwrap_or("");

                                match delta_type {
                                    "thinking_delta" => {
                                        if let Some(thinking) = delta.get("thinking").and_then(|t| t.as_str())
                                            && !thinking.is_empty()
                                                && event_sender.send(
                                                    StreamEvent::ThinkingDelta(thinking.to_string()),
                                                ).await.is_err() {
                                                    tracing::trace!("stream event receiver dropped");
                                                    break;
                                                }
                                    }
                                    "text_delta" => {
                                        if let Some(text) = delta.get("text").and_then(|text| text.as_str())
                                            && !text.is_empty()
                                                && event_sender.send(
                                                    StreamEvent::TextDelta(text.to_string()),
                                                ).await.is_err() {
                                                    tracing::trace!("stream event receiver dropped");
                                                    break;
                                                }
                                    }
                                    "signature_delta" => {
                                        if let Some(sig) = delta.get("signature").and_then(|s| s.as_str()) {
                                            current_thinking_signature = Some(
                                                current_thinking_signature
                                                    .map_or_else(|| sig.to_string(), |existing| existing + sig),
                                            );
                                        }
                                    }
                                    "input_json_delta" => {
                                        if let Some(partial_json) =
                                            delta.get("partial_json").and_then(|partial_json| partial_json.as_str())
                                        {
                                            current_tool_input.push_str(partial_json);
                                            if event_sender.send(
                                                StreamEvent::ToolInputDelta(
                                                    partial_json.to_string(),
                                                ),
                                            ).await.is_err() {
                                                tracing::trace!("stream event receiver dropped");
                                                break;
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            "content_block_stop" => {
                                if in_thinking {
                                    in_thinking = false;
                                    let signature = current_thinking_signature.take();
                                    if event_sender
                                        .send(StreamEvent::ThinkingComplete { signature })
                                        .await
                                        .is_err()
                                    {
                                        tracing::trace!("stream event receiver dropped");
                                        break;
                                    }
                                } else if in_tool_use {
                                    let input = if current_tool_input.is_empty() {
                                        serde_json::json!({})
                                    } else {
                                        match serde_json::from_str(&current_tool_input) {
                                            Ok(value) => value,
                                            Err(error) => {
                                                tracing::warn!("failed to parse tool input JSON: {}", error);
                                                serde_json::json!({})
                                            }
                                        }
                                    };
                                    if event_sender
                                        .send(StreamEvent::ToolUseEnd { input })
                                        .await
                                        .is_err()
                                    {
                                        tracing::trace!("stream event receiver dropped");
                                        break;
                                    }
                                    current_tool_input.clear();
                                    in_tool_use = false;
                                }
                            }
                            "message_delta" => {
                                let Some(delta) = data.get("delta") else {
                                    continue;
                                };
                                if let Some(usage) = data.get("usage") {
                                    let token_usage = parse_usage_object(usage);
                                    if event_sender
                                        .send(StreamEvent::Usage(token_usage))
                                        .await
                                        .is_err()
                                    {
                                        tracing::trace!("stream event receiver dropped");
                                        break;
                                    }
                                }
                                if let Some(stop_reason_str) =
                                    delta.get("stop_reason").and_then(|reason| reason.as_str())
                                {
                                    let stop_reason = parse_claude_stop_reason(stop_reason_str);
                                    if event_sender
                                        .send(StreamEvent::MessageEnd { stop_reason })
                                        .await
                                        .is_err()
                                    {
                                        tracing::trace!("stream event receiver dropped");
                                        break;
                                    }
                                }
                            }
                            "message_stop" => {
                                break;
                            }
                            "message_start" => {
                                if let Some(usage) =
                                    data.get("message").and_then(|m| m.get("usage"))
                                {
                                    let token_usage = parse_usage_object(usage);
                                    if event_sender
                                        .send(StreamEvent::Usage(token_usage))
                                        .await
                                        .is_err()
                                    {
                                        tracing::trace!("stream event receiver dropped");
                                        break;
                                    }
                                }
                            }
                            "ping" => {}
                            other => {
                                tracing::debug!("unknown Claude SSE event: {}", other);
                            }
                        }
                    }
                    Err(error) => {
                        if event_sender
                            .send(StreamEvent::Error(error.to_string()))
                            .await
                            .is_err()
                        {
                            tracing::trace!("stream event receiver dropped");
                        }
                        return Err(AgshError::StreamError(error.to_string()));
                    }
                }
            }
        }
    }

    Ok(())
}

/// Stats from a single [`redact_oldest_images`] invocation. Returned to callers so they can surface
/// a user-visible advisory and increment a per-session redaction counter.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct RedactionStats {
    pub images_redacted: usize,
    pub bytes_freed: usize,
}

/// Walk `messages` oldest-first and replace `ToolResultContent::Image` payloads with
/// [`IMAGE_REDACTION_PLACEHOLDER`] until at least `bytes_to_drop` base64 bytes have been removed.
/// The LAST message is never touched — it carries the moving `cache_control` breakpoint set in
/// [`convert_messages_to_claude_content`] and disturbing it would invalidate the cache for the new
/// turn unnecessarily.
///
/// Returns `Cow::Borrowed` if no work was needed (`bytes_to_drop == 0`). Otherwise returns
/// `Cow::Owned` with whatever redaction was possible — even when the budget couldn't be met, the
/// cloned messages are still returned so the caller can re-serialize and decide whether the body
/// fits.
pub(super) fn redact_oldest_images(
    messages: &[Message],
    bytes_to_drop: usize,
) -> (Cow<'_, [Message]>, RedactionStats) {
    if bytes_to_drop == 0 || messages.len() <= 1 {
        return (Cow::Borrowed(messages), RedactionStats::default());
    }

    let mut redacted: Vec<Message> = messages.to_vec();
    let last = redacted.len() - 1;
    let mut stats = RedactionStats::default();

    'outer: for message in &mut redacted[..last] {
        for block in &mut message.content {
            if let ContentBlock::ToolResult { content, .. } = block {
                for item in content.iter_mut() {
                    if let ToolResultContent::Image { source } = item {
                        stats.bytes_freed = stats.bytes_freed.saturating_add(source.data.len());
                        stats.images_redacted = stats.images_redacted.saturating_add(1);
                        *item = ToolResultContent::Text {
                            text: IMAGE_REDACTION_PLACEHOLDER.to_string(),
                        };
                        if stats.bytes_freed >= bytes_to_drop {
                            break 'outer;
                        }
                    }
                }
            }
        }
    }

    (Cow::Owned(redacted), stats)
}

/// Walk `messages` and downscale any `ToolResultContent::Image` whose pixel dimensions exceed
/// [`MAX_IMAGE_DIMENSION_PX`] on either axis. The body bytes (base64) for those images are replaced
/// with a re-encoded PNG that fits within the cap; smaller images are left alone. Returns
/// `Cow::Borrowed` when no work was needed.
///
/// Anthropic-specific: the 2000 px cap only matters for Anthropic's multi-image requests; this
/// helper is intentionally not applied to non-Claude providers. Decode/resize cost is incurred per
/// turn for each oversized image — typical sessions have few oversized images, and the cheap
/// [`crate::image::read_image_dimensions`] header read short-circuits the common case.
pub(super) fn downscale_oversized_images(messages: &[Message]) -> Cow<'_, [Message]> {
    use base64::Engine;
    use image::ImageFormat;

    fn parse_format(media_type: &str) -> Option<ImageFormat> {
        ImageFormat::from_mime_type(media_type)
    }

    // First pass: detect whether any image needs downscaling. Cheap — just base64-decode to peek at
    // header bytes. If nothing's oversized, skip the clone+rewrite entirely and return
    // Cow::Borrowed.
    let needs_work = messages.iter().any(|message| {
        message.content.iter().any(|block| match block {
            ContentBlock::ToolResult { content, .. } => content.iter().any(|item| match item {
                ToolResultContent::Image { source } => {
                    let Some(format) = parse_format(&source.media_type) else {
                        return false;
                    };
                    let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(&source.data)
                    else {
                        return false;
                    };
                    crate::image::read_image_dimensions(&bytes, format)
                        .map(|(w, h)| w > MAX_IMAGE_DIMENSION_PX || h > MAX_IMAGE_DIMENSION_PX)
                        .unwrap_or(false)
                }
                _ => false,
            }),
            _ => false,
        })
    });
    if !needs_work {
        return Cow::Borrowed(messages);
    }

    let mut owned: Vec<Message> = messages.to_vec();
    for message in owned.iter_mut() {
        for block in message.content.iter_mut() {
            if let ContentBlock::ToolResult { content, .. } = block {
                for item in content.iter_mut() {
                    if let ToolResultContent::Image { source } = item {
                        let Some(format) = parse_format(&source.media_type) else {
                            continue;
                        };
                        let Ok(bytes) =
                            base64::engine::general_purpose::STANDARD.decode(&source.data)
                        else {
                            continue;
                        };
                        let Ok((w, h)) = crate::image::read_image_dimensions(&bytes, format) else {
                            continue;
                        };
                        if w <= MAX_IMAGE_DIMENSION_PX && h <= MAX_IMAGE_DIMENSION_PX {
                            continue;
                        }
                        match crate::image::downscale_to_dim_cap(
                            &bytes,
                            format,
                            MAX_IMAGE_DIMENSION_PX,
                        ) {
                            Ok(png) => {
                                source.media_type = "image/png".to_string();
                                source.data =
                                    base64::engine::general_purpose::STANDARD.encode(&png);
                            }
                            Err(error) => {
                                tracing::warn!(
                                    "failed to downscale {}x{} {} image: {}",
                                    w,
                                    h,
                                    source.media_type,
                                    error,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    Cow::Owned(owned)
}

/// Serialize a Claude request body, downscaling oversized images first and reactively redacting old
/// tool-result image blocks if the serialized JSON still exceeds [`MAX_REQUEST_BYTES`]. Both Claude
/// providers run this same redact-and-retry loop; the caller supplies the body builder via `build`
/// so each provider's thinking / metadata wiring stays in its own file.
///
/// `build` takes a `messages` slice (the downscaled-then-maybe-redacted view) and returns the
/// serialized JSON. It's called once on the original messages and, if oversized, a second time on
/// the redacted set.
///
/// Returns the serialized body plus an optional [`crate::provider::Notice`]: on a successful
/// redaction pass, the notice describes what was dropped so the caller can forward it to the active
/// frontend (REPL renders via `render_hint`; ACP surfaces in the session/update stream). On the
/// happy path (no redaction needed), the notice is `None`. The function also records
/// [`RedactionStats`] on `session_stats` when one is provided.
pub(super) fn build_body_within_budget<F>(
    messages: &[Message],
    session_stats: Option<&std::sync::Arc<crate::stats::SessionStats>>,
    mut build: F,
) -> Result<(String, Option<crate::provider::Notice>)>
where
    F: FnMut(&[Message]) -> Result<String>,
{
    let prepared = downscale_oversized_images(messages);
    let body_json = build(prepared.as_ref())?;

    if body_json.len() <= MAX_REQUEST_BYTES {
        return Ok((body_json, None));
    }

    let bytes_to_drop = body_json.len() - REDACTION_TARGET_BYTES;
    let (redacted, stats) = redact_oldest_images(prepared.as_ref(), bytes_to_drop);
    let body_json = build(redacted.as_ref())?;

    if body_json.len() > MAX_REQUEST_BYTES {
        return Err(AgshError::Provider(format!(
            "request body is {} MiB after redacting old tool-result images; \
             Anthropic's limit is 32 MiB. Run /compact, remove large attachments \
             from the most recent turn, or split the work across smaller turns.",
            body_json.len() / 1_048_576,
        )));
    }

    if let Some(session_stats) = session_stats {
        session_stats.record_redaction(stats.images_redacted as u64, stats.bytes_freed as u64);
    }
    let notice_text = format!(
        "Redacted {} old image{} (~{} MiB freed). Cache prefix invalidated for those messages.",
        stats.images_redacted,
        if stats.images_redacted == 1 { "" } else { "s" },
        stats.bytes_freed / 1_048_576,
    );
    tracing::warn!(
        "redacted {} old tool-result image(s); body now {} MiB",
        stats.images_redacted,
        body_json.len() / 1_048_576,
    );
    Ok((body_json, Some(crate::provider::Notice::info(notice_text))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ImageSource;

    #[test]
    fn test_is_thinking_enabled_override_off() {
        assert!(!is_thinking_enabled(0, true));
        assert!(!is_thinking_enabled(0, false));
    }

    #[test]
    fn test_is_thinking_enabled_override_on() {
        assert!(is_thinking_enabled(1, true));
        assert!(is_thinking_enabled(1, false));
    }

    #[test]
    fn test_is_thinking_enabled_unset_uses_default() {
        assert!(is_thinking_enabled(-1, true));
        assert!(!is_thinking_enabled(-1, false));
        // Any non-0/1 value should be treated as "unset" and fall through to the configured
        // default.
        assert!(is_thinking_enabled(42, true));
        assert!(!is_thinking_enabled(-99, false));
    }

    #[test]
    fn test_model_supports_modern_features() {
        assert!(model_supports_modern_features("claude-opus-4-6-20250514"));
        assert!(model_supports_modern_features("claude-sonnet-4-20250514"));
        assert!(model_supports_modern_features("claude-haiku-4-5-20251001"));
        assert!(!model_supports_modern_features(
            "claude-3-5-sonnet-20241022"
        ));
        assert!(!model_supports_modern_features("claude-3-opus-20240229"));
        assert!(!model_supports_modern_features("gpt-4o"));
    }

    #[test]
    fn test_model_supports_effort() {
        assert!(model_supports_effort("claude-opus-4-6-20250514"));
        assert!(model_supports_effort("claude-sonnet-4-6"));
        // Older / non-4-6 sonnet/opus/haiku are explicitly denied.
        assert!(!model_supports_effort("claude-sonnet-4-20250514"));
        assert!(!model_supports_effort("claude-opus-4-1"));
        assert!(!model_supports_effort("claude-haiku-4-5-20251001"));
        // Unknown 1P model defaults to true.
        assert!(model_supports_effort("claude-future-experimental-7"));
    }

    #[test]
    fn test_model_is_haiku() {
        assert!(model_is_haiku("claude-haiku-4-5-20251001"));
        assert!(model_is_haiku("claude-haiku-4-5"));
        assert!(!model_is_haiku("claude-opus-4-6-20250514"));
        assert!(!model_is_haiku("claude-sonnet-4-20250514"));
    }

    #[test]
    fn test_parse_claude_stop_reason_all_variants() {
        assert_eq!(parse_claude_stop_reason("end_turn"), StopReason::EndTurn);
        assert_eq!(parse_claude_stop_reason("tool_use"), StopReason::ToolUse);
        assert_eq!(
            parse_claude_stop_reason("max_tokens"),
            StopReason::MaxTokens
        );
        assert_eq!(
            parse_claude_stop_reason("something_else"),
            StopReason::Unknown("something_else".to_string())
        );
    }

    fn image_block(tool_use_id: &str, payload: &str) -> ContentBlock {
        ContentBlock::ToolResult {
            tool_use_id: tool_use_id.to_string(),
            content: vec![ToolResultContent::Image {
                source: ImageSource {
                    source_type: "base64".to_string(),
                    media_type: "image/png".to_string(),
                    data: payload.to_string(),
                },
            }],
            is_error: false,
        }
    }

    fn user_with_block(block: ContentBlock) -> Message {
        Message {
            role: Role::User,
            content: vec![block],
        }
    }

    fn assistant_text(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    #[test]
    fn test_redact_no_op_when_under_threshold() {
        let messages = vec![
            user_with_block(image_block("call_a", "AAAA")),
            assistant_text("ack"),
        ];
        let (result, stats) = redact_oldest_images(&messages, 0);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(stats.images_redacted, 0);
        assert_eq!(stats.bytes_freed, 0);
    }

    #[test]
    fn test_redact_drops_oldest_image_first() {
        // Two images: one in msg[0] (older), one in msg[1] (last). The helper must only touch the
        // older one — the last message carries the moving cache_control marker.
        let payload_a = "A".repeat(1024);
        let payload_b = "B".repeat(1024);
        let messages = vec![
            user_with_block(image_block("call_a", &payload_a)),
            user_with_block(image_block("call_b", &payload_b)),
        ];
        let (result, stats) = redact_oldest_images(&messages, 1);
        assert_eq!(stats.images_redacted, 1);
        assert_eq!(stats.bytes_freed, 1024);
        let owned = match result {
            Cow::Owned(v) => v,
            Cow::Borrowed(_) => panic!("expected owned redacted vec"),
        };
        // msg[0] image redacted to placeholder text.
        match &owned[0].content[0] {
            ContentBlock::ToolResult { content, .. } => match &content[0] {
                ToolResultContent::Text { text } => {
                    assert_eq!(text, IMAGE_REDACTION_PLACEHOLDER);
                }
                other => panic!("expected text placeholder, got {:?}", other),
            },
            other => panic!("expected ToolResult, got {:?}", other),
        }
        // msg[1] (last) image untouched.
        match &owned[1].content[0] {
            ContentBlock::ToolResult { content, .. } => match &content[0] {
                ToolResultContent::Image { source } => {
                    assert_eq!(source.data, payload_b);
                }
                other => panic!("expected untouched image, got {:?}", other),
            },
            other => panic!("expected ToolResult, got {:?}", other),
        }
    }

    #[test]
    fn test_redact_stops_when_target_reached() {
        // Three images each 1 KiB. Target = 1500 bytes. Only the FIRST image should be redacted;
        // the second remains because we hit the budget after one (1024 >= 1500 is false, but
        // saturating_add gets us past after the first redaction since we then loop-check before the
        // second image is considered? — no: the check is `bytes_dropped >= bytes_to_drop`, so 1024
        // < 1500 means we redact the second too). Clarify by setting target = 1024.
        let payload = "X".repeat(1024);
        let messages = vec![
            user_with_block(image_block("call_a", &payload)),
            user_with_block(image_block("call_b", &payload)),
            assistant_text("end"),
        ];
        let (result, stats) = redact_oldest_images(&messages, 1024);
        assert_eq!(stats.images_redacted, 1);
        assert_eq!(stats.bytes_freed, 1024);
        let owned = match result {
            Cow::Owned(v) => v,
            Cow::Borrowed(_) => panic!("expected owned"),
        };
        // First image redacted.
        match &owned[0].content[0] {
            ContentBlock::ToolResult { content, .. } => match &content[0] {
                ToolResultContent::Text { text } => {
                    assert_eq!(text, IMAGE_REDACTION_PLACEHOLDER);
                }
                _ => panic!("first should be redacted"),
            },
            _ => unreachable!(),
        }
        // Second image preserved (budget already met).
        match &owned[1].content[0] {
            ContentBlock::ToolResult { content, .. } => match &content[0] {
                ToolResultContent::Image { .. } => {}
                _ => panic!("second image should still be intact"),
            },
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_redact_preserves_last_message() {
        // Single image, in the LAST message. Helper must not touch it even when the budget is huge.
        let payload = "P".repeat(8 * 1024);
        let messages = vec![
            assistant_text("setup"),
            user_with_block(image_block("call_only", &payload)),
        ];
        let (result, stats) = redact_oldest_images(&messages, usize::MAX);
        // No redactable images outside the last message → 0 redactions.
        assert_eq!(stats.images_redacted, 0);
        assert_eq!(stats.bytes_freed, 0);
        let owned = match result {
            Cow::Owned(v) => v,
            Cow::Borrowed(_) => panic!("expected owned (cloned even when no redaction)"),
        };
        match &owned[1].content[0] {
            ContentBlock::ToolResult { content, .. } => match &content[0] {
                ToolResultContent::Image { source } => assert_eq!(source.data, payload),
                _ => panic!("last-message image must survive"),
            },
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_redact_handles_no_images() {
        let messages = vec![
            assistant_text("hello"),
            assistant_text("world"),
            assistant_text("end"),
        ];
        let (result, stats) = redact_oldest_images(&messages, 1024);
        assert_eq!(stats.images_redacted, 0);
        assert_eq!(stats.bytes_freed, 0);
        let owned = match result {
            Cow::Owned(v) => v,
            Cow::Borrowed(_) => panic!("expected owned (cloned even when no images)"),
        };
        assert_eq!(owned.len(), 3);
        for (orig, copy) in messages.iter().zip(owned.iter()) {
            assert_eq!(orig.content.len(), copy.content.len());
        }
    }

    fn synthesize_png_base64(width: u32, height: u32) -> String {
        use std::io::Cursor;

        use base64::Engine;
        use image::{ImageFormat, RgbaImage};
        let img = RgbaImage::from_pixel(width, height, image::Rgba([100, 150, 200, 255]));
        let mut bytes = Vec::new();
        img.write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .expect("encode png");
        base64::engine::general_purpose::STANDARD.encode(&bytes)
    }

    fn user_with_image_block(tool_use_id: &str, base64_payload: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.to_string(),
                content: vec![ToolResultContent::Image {
                    source: crate::provider::ImageSource {
                        source_type: "base64".to_string(),
                        media_type: "image/png".to_string(),
                        data: base64_payload.to_string(),
                    },
                }],
                is_error: false,
            }],
        }
    }

    #[test]
    fn test_downscale_no_op_when_all_within_cap() {
        let small = synthesize_png_base64(800, 600);
        let messages = vec![
            user_with_image_block("call_a", &small),
            assistant_text("ack"),
        ];
        assert!(matches!(
            downscale_oversized_images(&messages),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn test_downscale_resizes_oversized_image() {
        use base64::Engine;
        use image::ImageFormat;
        let big = synthesize_png_base64(2400, 1200);
        let small = synthesize_png_base64(800, 600);
        let messages = vec![
            user_with_image_block("call_big", &big),
            user_with_image_block("call_small", &small),
        ];
        let result = downscale_oversized_images(&messages);
        let owned = match result {
            Cow::Owned(v) => v,
            Cow::Borrowed(_) => panic!("expected owned (resize triggered)"),
        };
        // First image was downscaled to fit 2000 px on each axis.
        match &owned[0].content[0] {
            ContentBlock::ToolResult { content, .. } => match &content[0] {
                ToolResultContent::Image { source } => {
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(&source.data)
                        .expect("decode");
                    let decoded =
                        image::load_from_memory_with_format(&bytes, ImageFormat::Png).expect("png");
                    assert!(decoded.width() <= MAX_IMAGE_DIMENSION_PX);
                    assert!(decoded.height() <= MAX_IMAGE_DIMENSION_PX);
                    // 2:1 aspect ratio preserved.
                    assert_eq!(decoded.width() / decoded.height(), 2);
                }
                _ => panic!("expected resized image"),
            },
            _ => unreachable!(),
        }
        // Second image was within cap → unchanged.
        match &owned[1].content[0] {
            ContentBlock::ToolResult { content, .. } => match &content[0] {
                ToolResultContent::Image { source } => assert_eq!(source.data, small),
                _ => panic!("small image should be untouched"),
            },
            _ => unreachable!(),
        }
    }

    /// Locks in the contract that `build_body_within_budget` returns a user-visible
    /// [`crate::provider::Notice`] (rather than printing to stderr directly) when redaction kicks
    /// in. The agent loop then forwards it through `Frontend::emit`, which is how ACP clients see
    /// the redaction signal that used to silently bypass them.
    #[test]
    fn test_build_body_within_budget_returns_notice_on_redaction() {
        use std::cell::Cell;

        // Two messages, the first containing an oversized image and the second a small one. The
        // redactor only touches non-last messages, so the older image is the one that gets
        // dropped.
        let big_payload = "X".repeat(2 * 1024 * 1024);
        let messages = vec![
            user_with_block(image_block("call_a", &big_payload)),
            user_with_block(image_block("call_b", "BBB")),
            assistant_text("ack"),
        ];

        let call_count: Cell<usize> = Cell::new(0);
        let build = |_msgs: &[Message]| -> Result<String> {
            let n = call_count.get();
            call_count.set(n + 1);
            if n == 0 {
                // First serialization: oversize. Use a slim payload so the test stays cheap; the
                // function only cares about `.len() > MAX_REQUEST_BYTES`.
                Ok("X".repeat(MAX_REQUEST_BYTES + 1024))
            } else {
                Ok("{}".to_string())
            }
        };

        let (body, notice) =
            build_body_within_budget(&messages, None, build).expect("redaction should succeed");
        assert_eq!(body, "{}");
        let notice = notice.expect("redaction must surface a Notice");
        assert_eq!(notice.level, crate::provider::NoticeLevel::Info);
        assert!(
            notice.text.starts_with("Redacted "),
            "notice text should describe the redaction: {:?}",
            notice.text,
        );
        assert_eq!(call_count.get(), 2, "build closure called twice");
    }

    /// On the happy path (no redaction needed), the function returns `None` for the notice. Locks
    /// the contract: frontends never see a no-op advisory.
    #[test]
    fn test_build_body_within_budget_no_notice_when_within_budget() {
        let messages = vec![assistant_text("hi")];
        let build = |_msgs: &[Message]| -> Result<String> { Ok("{}".to_string()) };
        let (body, notice) = build_body_within_budget(&messages, None, build).expect("happy path");
        assert_eq!(body, "{}");
        assert!(notice.is_none());
    }
}
