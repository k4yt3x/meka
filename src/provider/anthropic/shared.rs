//! Helpers shared by [`super::messages::AnthropicMessagesProvider`] and
//! [`super::subscription::ClaudeSubscriptionProvider`]. Everything in this module is independent of
//! the authentication scheme: message/tool conversion to the Claude wire format, SSE streaming,
//! response parsing, per-model capability detection, the thinking override, and the one driver
//! both providers send through.

use std::borrow::Cow;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    config::ThinkingMode,
    conversation::{ContentBlock, Message, OpaqueReasoning, Role, ToolResultContent},
    error::{MekaError, Result},
    provider::{
        CompletionRequest, StopReason, StreamEvent, ThinkingOverride, ToolDefinition,
        sse::{End, Step},
    },
    stats::TokenUsage,
};

/// Normalize a Claude-family base URL: trailing slashes, then one trailing `/v1`.
///
/// A Claude base URL is the *host root*, not the versioned prefix. `AnthropicMessagesProvider`
/// appends `/v1/messages` and `/v1/models/{id}`; `ClaudeSubscriptionProvider` appends those and
/// also `/api/oauth/usage`, `/api/oauth/profile` and `/api/oauth/claude_cli/roles`, which sit under
/// a different root entirely. Baking `/v1` into the base would put that second set out of reach, so
/// the root is the only prefix both can share. The OpenAI family has the opposite convention (`/v1`
/// belongs in the base), which is what makes this worth normalizing rather than merely documenting:
/// a gateway serving both APIs publishes one URL per family, and its Anthropic one is routinely
/// written with the `/v1` its OpenAI sibling needs.
///
/// Left alone, that pastes through to `/v1/v1/messages`, which no gateway routes. Stripping it is
/// safe precisely because there is no reading under which it could have been meant: the segment is
/// re-added on every request. Only a *trailing* `/v1` goes, so a base whose path legitimately
/// contains one mid-way (`https://gateway.ai.cloudflare.com/v1/{account}/{gateway}/anthropic`)
/// survives intact.
pub(crate) fn normalize_claude_base_url(url: &str) -> String {
    let trimmed = crate::provider::normalize_base_url(url);
    match trimmed.strip_suffix("/v1") {
        Some(without_version) => {
            // Reports `trimmed`, not `url`: when both steps apply, the slash-trim has already
            // logged `url` -> `trimmed`, so quoting the original again would read as two unrelated
            // rewrites of the same string rather than one chain.
            tracing::debug!(
                "dropped the trailing '/v1' from Claude base URL '{trimmed}'; meka appends it per request"
            );
            without_version.to_string()
        }
        None => trimmed,
    }
}

/// The Messages API request size limit documented by Anthropic is 32 MiB; ~2 MiB is held back for
/// headers, URL, attestation patches, and serialization slack. Bodies above this threshold are
/// reactively shrunk by `redact_oldest_images` in `crate::provider::budget` before they're posted.
pub(super) const MAX_REQUEST_BYTES: usize = 30 * crate::text::MIB;

// The redaction itself, its stopping point and the refusal live in `crate::provider::budget`, since
// every backend applies them; the constant above is the Anthropic default ceiling they run against.
// The redaction tests live beside the downscale tests, hence the test-only import.
pub(super) use crate::provider::budget::serialize_body;
#[cfg(test)]
pub(super) use crate::provider::budget::{IMAGE_REDACTION_PLACEHOLDER, redact_oldest_images};

/// Anthropic accepts up to 8000 px per axis on a *single*-image request, but rejects anything over
/// 2000 px on either axis once the request contains more than one image. Every image is downscaled
/// to fit so a session can freely accumulate images without tripping the multi-image cap. Enforced
/// by the Claude backends only; the others need not pay the resize cost.
pub(super) const MAX_IMAGE_DIMENSION_PX: u32 = 2000;

/// Extract a `TokenUsage` from an Anthropic `usage` object. Used by both the non-streaming response
/// parser and the SSE driver. Anthropic emits the same shape (`input_tokens`, `output_tokens`,
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

/// The mode a request actually uses: the profile's, unless the request turned thinking off for
/// this call (the compaction summary does, so it doesn't pay for reasoning). An override only ever
/// turns thinking *off*, never on, so it cannot resurrect a mode the profile disabled.
pub(super) fn effective_thinking(
    thinking: ThinkingOverride,
    configured: ThinkingMode,
) -> ThinkingMode {
    match thinking {
        ThinkingOverride::Off => ThinkingMode::Off,
        ThinkingOverride::Inherit => configured,
    }
}

/// Parse a `(major, minor)` version out of a Claude model name. The version is written as
/// hyphen-separated digit groups, which sit after the family on the 4.x line (`claude-opus-4-8`)
/// but before it on the 3.x line (`claude-3-5-sonnet`); in both layouts it is the first one or two
/// *short* numeric segments. The trailing date stamp (`-20250514`) is skipped because it has more
/// than two digits. The first short number becomes the major and the next the minor (defaulting to
/// 0); returns `None` when no version-like segment is present.
fn parse_model_version(model: &str) -> Option<(u32, u32)> {
    let mut numbers = model
        .split('-')
        .filter(|segment| {
            !segment.is_empty()
                && segment.len() <= 2
                && segment.bytes().all(|byte| byte.is_ascii_digit())
        })
        .filter_map(|segment| segment.parse::<u32>().ok());
    let major = numbers.next()?;
    let minor = numbers.next().unwrap_or(0);
    Some((major, minor))
}

/// Whether a model name is in the Haiku family.
pub(super) fn model_is_haiku(model: &str) -> bool {
    model.to_ascii_lowercase().contains("haiku")
}

/// Insert the `max_tokens` + `thinking` fields shared by both Claude providers' request bodies.
/// [`ThinkingMode::Adaptive`] gets a fixed 64k ceiling, [`ThinkingMode::Budgeted`] gets
/// `max(budget*2, 32k)` with an explicit budget, and [`ThinkingMode::Off`] a flat 32k. A
/// `max_output_tokens` override (the profile knob) replaces whichever default would otherwise
/// apply; under `Budgeted` it is clamped to stay above `budget_tokens` (the API rejects
/// `max_tokens <= thinking.budget_tokens`).
///
/// The encoding comes from the profile rather than from the model name: this is what a request
/// reaching an arbitrary Anthropic-compatible endpoint needs, since meka cannot tell which of the
/// two forms that endpoint implements.
///
/// `display` is the backend's to add, through [`with_display`]: `claude-subscription` sends Claude
/// Code's display mode, and `anthropic-messages` sends none, because the display values are a
/// first-party feature an arbitrary endpoint may not implement.
///
/// The `max_tokens` sent when the profile states no `max_output_tokens` are Claude Code 2.1.263's
/// (verified by wire capture): 64000 under adaptive thinking, 32000 otherwise, raised to twice
/// the budget under budgeted thinking when that is more. The API requires the field, so omitting
/// it is not a way to ask for a default.
pub(super) fn insert_thinking_fields(
    body: &mut serde_json::Map<String, serde_json::Value>,
    thinking: ThinkingMode,
    budget_tokens: u64,
    max_output_tokens: Option<u64>,
    display: Option<&str>,
) {
    match thinking {
        ThinkingMode::Adaptive => {
            let max_tokens = max_output_tokens.unwrap_or(64_000);
            body.insert("max_tokens".to_string(), serde_json::json!(max_tokens));
            body.insert(
                "thinking".to_string(),
                with_display(serde_json::json!({ "type": "adaptive" }), display),
            );
        }
        ThinkingMode::Budgeted => {
            let default_max = std::cmp::max(budget_tokens.saturating_mul(2), 32_000);
            // Clamp above the budget so an override that's too small can't produce a 400.
            let max_tokens = max_output_tokens
                .unwrap_or(default_max)
                .max(budget_tokens.saturating_add(1));
            body.insert("max_tokens".to_string(), serde_json::json!(max_tokens));
            body.insert(
                "thinking".to_string(),
                with_display(
                    serde_json::json!({
                        "type": "enabled",
                        "budget_tokens": budget_tokens
                    }),
                    display,
                ),
            );
        }
        ThinkingMode::Off => {
            let max_tokens = max_output_tokens.unwrap_or(32_000);
            body.insert("max_tokens".to_string(), serde_json::json!(max_tokens));
        }
    }
}

/// The thinking object with Claude Code's `display` added when the backend asks for one.
fn with_display(mut thinking: serde_json::Value, display: Option<&str>) -> serde_json::Value {
    if let (Some(display), Some(object)) = (display, thinking.as_object_mut()) {
        object.insert("display".to_string(), serde_json::json!(display));
    }
    thinking
}

/// Mirrors Claude Code's `modelSupportsThinking` (and the equivalent
/// `modelSupportsISP` / `modelSupportsContextManagement`) on the 1P API:
/// any Claude 4+ model. Claude-3.x is excluded.
pub(super) fn model_supports_modern_features(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("claude") && !lower.contains("claude-3-")
}

/// Whether a Claude model accepts the `temperature` sampling parameter. Mirrors Claude Code
/// 2.1.263's `rQo`, which is an **allowlist** of the older models that still accept sampling
/// params: the Claude 3.x line, Opus 4.0/4.1/4.5/4.6, Sonnet 4.0/4.5/4.6, and Haiku 4.5. Everything
/// newer (Opus 4.7/4.8/5, Sonnet 5, Fable/Mythos 5) rejects `temperature` with a 400.
///
/// The allowlist direction is the point: an unrecognized model, which in practice means one newer
/// than this list, resolves to `false` and the parameter is omitted. A denylist would instead send
/// `temperature` to every future model and earn a 400 until the list was updated. Matching is by
/// family + [`parse_model_version`] rather than Claude Code's exact string equality because meka
/// accepts date-stamped ids (`claude-haiku-4-5-20251001`) as well as canonical ones.
pub(super) fn model_supports_temperature(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    if lower.contains("claude-3-") {
        return true;
    }
    let Some(version) = parse_model_version(&lower) else {
        return false;
    };
    if lower.contains("opus") {
        matches!(version, (4, 0) | (4, 1) | (4, 5) | (4, 6))
    } else if lower.contains("sonnet") {
        matches!(version, (4, 0) | (4, 5) | (4, 6))
    } else if lower.contains("haiku") {
        version == (4, 5)
    } else {
        false
    }
}

/// Whether a Claude model accepts `output_config.effort`.
///
/// A denylist mirroring Claude Code 2.1.263's own gate, which excludes the Claude 3.x line, Opus
/// 4.0/4.1, Sonnet 4.0/4.5 and Haiku 4.5 and sends the field to everything else on the first-party
/// endpoint. Same reasoning and same single caller as
/// [`model_supports_mid_conversation_system`]: an unrecognized name here is one *newer* than the
/// list, and effort is what those models are for.
pub(super) fn model_supports_effort(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    if lower.contains("claude-3-") {
        return false;
    }
    let Some(version) = parse_model_version(&lower) else {
        return true;
    };
    if lower.contains("opus") {
        !matches!(version, (4, 0) | (4, 1))
    } else if lower.contains("sonnet") {
        !matches!(version, (4, 0) | (4, 5))
    } else if lower.contains("haiku") {
        version != (4, 5)
    } else {
        true
    }
}

/// The effort `claude-subscription` sends when the profile configures none.
///
/// Claude Code reads a per-model `default_effort` out of a table bundled in its binary and clamps
/// it to what that model accepts; almost every effort-capable model in the 2.1.263 table comes out
/// of that as `high`, and `high` is also the value Claude Code falls back to for any model the
/// table does not list.
///
/// One constant, and deliberately not a transcription of that table. A per-model figure would be a
/// fact about someone else's data that goes stale on their release schedule with nothing in the
/// build to notice, and it would buy nothing: the server cannot tell a default meka chose from a
/// value the user configured, so the only thing a wrong entry could produce is meka quietly asking
/// for the wrong tier. Anyone who wants a different one sets `effort` on the profile.
pub(super) const DEFAULT_EFFORT: &str = "high";

/// Whether a Claude model supports mid-conversation system messages (the
/// `mid-conversation-system-2026-04-07` beta).
///
/// A **denylist**, mirroring Claude Code 2.1.263's gate model for model: the Claude 3.x line, Opus
/// 4.0/4.1/4.5/4.6/4.7, Sonnet 4.0/4.5/4.6 and Haiku 4.5 are excluded, and everything else on the
/// first-party endpoint is sent it. The direction is the opposite of
/// [`model_supports_temperature`]'s and deliberately so, because the two fail in opposite ways: an
/// unrecognized model here is one *newer* than the list, which Claude Code sends the beta to, and
/// withholding it would silently drop the mid-conversation system messages meka relies on. It is
/// safe only because this gate has exactly one caller, `claude-subscription`, whose endpoint is
/// always Anthropic's, so an unrecognized name there is necessarily a real Claude. Do not reach
/// for it from a backend a `base_url` can point anywhere.
pub(super) fn model_supports_mid_conversation_system(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    if lower.contains("claude-3-") {
        return false;
    }
    let Some(version) = parse_model_version(&lower) else {
        return true;
    };
    if lower.contains("opus") {
        !matches!(version, (4, 0) | (4, 1) | (4, 5) | (4, 6) | (4, 7))
    } else if lower.contains("sonnet") {
        !matches!(version, (4, 0) | (4, 5) | (4, 6))
    } else if lower.contains("haiku") {
        version != (4, 5)
    } else {
        true
    }
}

/// The name of an SSE frame: the `type` the data names, else the `event:` line.
///
/// Anthropic sends both and they agree. A gateway that forwards the data and drops the `event:`
/// line leaves `eventsource-stream` reporting `message`, and dispatching on that alone would
/// discard every frame of a good turn. The Responses driver keys the same way for the same reason.
fn claude_frame_name<'a>(data: &'a serde_json::Value, event_name: &'a str) -> &'a str {
    data.get("type")
        .and_then(|value| value.as_str())
        .unwrap_or(event_name)
}

/// The cache breakpoint a backend puts on the last block it sends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CacheBreakpoint {
    /// `{"type": "ephemeral"}`: the API's own TTL, which needs no beta and which every endpoint
    /// speaking this protocol accepts.
    Ephemeral,
    /// `{"type": "ephemeral", "ttl": "1h"}`, admitted by the `extended-cache-ttl-2025-04-11` beta
    /// the subscription backend sends, and pinned by its captured wire.
    OneHour,
}

impl CacheBreakpoint {
    fn value(self) -> serde_json::Value {
        match self {
            Self::Ephemeral => serde_json::json!({"type": "ephemeral"}),
            Self::OneHour => serde_json::json!({"type": "ephemeral", "ttl": "1h"}),
        }
    }
}

/// An image block for this wire. A base64 source serializes to exactly Anthropic's `source` object;
/// a blob reference the store did not resolve is not an image the API can take, so it goes out as
/// the same placeholder text the OpenAI encoders send, rather than as a shape the API rejects
/// whole.
fn claude_image_block(source: &crate::image::ImageSource) -> serde_json::Value {
    match source {
        crate::image::ImageSource::Base64 { .. } => serde_json::json!({
            "type": "image",
            "source": source,
        }),
        crate::image::ImageSource::Blob { .. } => serde_json::json!({
            "type": "text",
            "text": crate::image::UNRESOLVED_IMAGE_PLACEHOLDER,
        }),
    }
}

/// The conversation as the `messages` array of a Claude request, with the cache breakpoint on its
/// last block.
pub(super) fn convert_messages_to_claude_content(
    messages: &[Message],
    breakpoint: CacheBreakpoint,
) -> Vec<serde_json::Value> {
    let mut claude_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|message| {
            let role = match message.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };

            let content: Vec<serde_json::Value> = message
                .content
                .iter()
                .map(|block| {
                    match block {
                        // The turn's context block is text on this wire, ahead of the words.
                        ContentBlock::Text { text } | ContentBlock::TurnContext { text } => {
                            serde_json::json!({
                                "type": "text",
                                "text": text,
                            })
                        }
                        ContentBlock::Image { source } => claude_image_block(source),
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
                            let content: Vec<serde_json::Value> = content
                                .iter()
                                .map(|item| match item {
                                    ToolResultContent::Text { text } => {
                                        serde_json::json!({"type": "text", "text": text})
                                    }
                                    ToolResultContent::Image { source } => {
                                        claude_image_block(source)
                                    }
                                })
                                .collect();
                            serde_json::json!({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": content,
                                "is_error": is_error,
                            })
                        }
                        // Only a signature belongs on this wire. A `Sealed` block reaches here
                        // when a session recorded against the Responses API is resumed under
                        // Claude, and its encrypted content is not a Claude signature, so the
                        // block goes out unsigned rather than carrying a foreign blob.
                        ContentBlock::Thinking { thinking, opaque } => {
                            let mut obj = serde_json::json!({
                                "type": "thinking",
                                "thinking": thinking
                            });
                            if let Some(OpaqueReasoning::Signed { signature }) = opaque {
                                obj["signature"] = serde_json::json!(signature);
                            }
                            obj
                        }
                        // Replayed verbatim: the API needs the opaque `data` unchanged to continue
                        // the redacted reasoning chain.
                        ContentBlock::RedactedThinking { data } => {
                            serde_json::json!({
                                "type": "redacted_thinking",
                                "data": data,
                            })
                        }
                    }
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
        .find(|message| message.get("role").and_then(|r| r.as_str()) == Some("assistant"))
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

    // After the strip, not before it: attached to the last block first, the breakpoint would go
    // out on a trailing thinking block and leave with it, so a conversation ending on one would
    // carry no breakpoint at all and re-bill its whole prefix at the write tier.
    if let Some(last) = claude_messages.last_mut()
        && let Some(content) = last.get_mut("content").and_then(|c| c.as_array_mut())
        && let Some(block) = content.last_mut().and_then(|b| b.as_object_mut())
    {
        block.insert("cache_control".to_string(), breakpoint.value());
    }

    claude_messages
}

/// The tool definitions as the `tools` array of a Claude request.
pub(super) fn convert_tools_to_claude_tools(tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
    // No per-tool `cache_control`: Claude Code's captured CLI wire leaves tools unmarked, and the
    // rolling last-message breakpoint already caches the tools+system prefix cumulatively, so the
    // extra tool breakpoint is redundant.
    tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })
        })
        .collect()
}

/// A Claude `stop_reason` as meka's own.
pub(super) fn parse_claude_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        // Claude does not include the refusal text alongside the streaming `stop_reason` delta; the
        // model's text content is what the user sees as the refusal. Surface an empty refusal
        // payload; the assistant message blocks carry the human-readable explanation already.
        "refusal" => StopReason::Refusal(String::new()),
        // Log unrecognized reasons (e.g. `pause_turn`) so a recurrence is diagnosable; the raw
        // string is otherwise discarded once mapped to `Unknown`.
        other => {
            tracing::warn!(
                "Claude returned unrecognized stop_reason {other:?}; mapping to Unknown"
            );
            StopReason::Unknown(other.to_string())
        }
    }
}

/// The assistant message, stop reason and usage out of a non-streaming Claude response.
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
        .ok_or_else(|| MekaError::Provider("no content array in response".to_string()))?;

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
                        MekaError::Provider("tool_use block missing 'id' field".to_string())
                    })?
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|name| name.as_str())
                    .ok_or_else(|| {
                        MekaError::Provider("tool_use block missing 'name' field".to_string())
                    })?
                    .to_string();
                let input = block.get("input").cloned().unwrap_or_else(|| {
                    tracing::warn!("tool_use block missing 'input' field");
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
                        opaque: signature.map(|signature| OpaqueReasoning::Signed { signature }),
                    });
                }
            }
            "redacted_thinking" => {
                if let Some(data) = block.get("data").and_then(|d| d.as_str()) {
                    content_blocks.push(ContentBlock::RedactedThinking {
                        data: data.to_string(),
                    });
                } else {
                    tracing::warn!("redacted_thinking block missing 'data' field");
                }
            }
            _ => {
                tracing::warn!("unknown Claude content block type: {block_type}");
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

/// What the two Claude providers differ in, so one driver serves both. The API-key backend has
/// nothing to refresh and no decoration; the subscription backend rotates a rejected credential
/// once, patches the attestation into the serialized body, and remembers the response's request id.
#[async_trait::async_trait]
pub(super) trait ClaudeBackend: crate::oauth::RefreshesCredential + Send + Sync {
    /// The HTTP client every request goes through.
    fn client(&self) -> &reqwest::Client;
    /// The URL a completion is posted to.
    fn endpoint(&self) -> String;
    /// Largest request body this backend sends before redacting old images: the profile's
    /// `max_request_bytes`, else [`MAX_REQUEST_BYTES`].
    fn max_request_bytes(&self) -> usize;
    /// The request body for one completion, before serialization.
    fn request_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        stream: bool,
        thinking: ThinkingOverride,
        attribution: &crate::provider::Attribution,
    ) -> serde_json::Value;
    /// The serialized body after any decoration that needs the whole of it.
    fn finish_body(&self, _system_prompt: &str, body_json: String) -> Result<String> {
        Ok(body_json)
    }
    /// One attempt's authenticated request. Called again after a rejected credential, so a backend
    /// that can refresh one does it here.
    async fn authenticated_request(
        &self,
        request: reqwest::RequestBuilder,
        has_tools: bool,
        stream: bool,
        thinking: ThinkingOverride,
    ) -> Result<reqwest::RequestBuilder>;
    /// Record the response's request id where the next request can name it; a no-op for a backend
    /// whose wire carries none.
    fn remember_request_id(
        &self,
        _attribution: &crate::provider::Attribution,
        _headers: &reqwest::header::HeaderMap,
    ) {
    }
}

/// Called from inside the `map_err` closure rather than beside the length it reads, so the size
/// `String` is allocated on the failure path only: every provider request pays for what happens
/// here, and the value is read by none of the ones that succeed.
fn transport_error(body_length: usize, error: &reqwest::Error) -> MekaError {
    crate::error::provider_transport_error(
        &format!(
            "HTTP request failed (body {})",
            crate::text::format_size(body_length)
        ),
        error,
        None,
    )
}

/// One non-streaming Claude call: the body within budget, one send, one retry on a credential the
/// backend could refresh, and the login remedy on a second rejection.
pub(super) async fn complete<B: ClaudeBackend>(
    backend: &B,
    request: CompletionRequest<'_>,
    cancellation: CancellationToken,
) -> Result<crate::provider::Completion> {
    let CompletionRequest {
        system_prompt,
        messages,
        tools,
        thinking,
        attribution,
        ..
    } = request;
    let (body_json, redaction_notice) =
        build_body_within_budget(messages, backend.max_request_bytes(), |messages| {
            serialize_body(&backend.request_body(
                system_prompt,
                messages,
                tools,
                false,
                thinking,
                &attribution,
            ))
        })?;
    let body_json = backend.finish_body(system_prompt, body_json)?;
    let body_length = body_json.len();

    let response = crate::oauth::send_with_one_refresh(
        backend,
        crate::error::ProviderRequest::Completion,
        |error| transport_error(body_length, error),
        || async {
            Ok(backend
                .authenticated_request(
                    backend.client().post(backend.endpoint()),
                    !tools.is_empty(),
                    false,
                    thinking,
                )
                .await?
                .body(body_json.clone()))
        },
        &cancellation,
    )
    .await?;

    let status = response.status();
    let retry_after = crate::error::parse_retry_after(response.headers());
    backend.remember_request_id(&attribution, response.headers());
    let response_text =
        crate::provider::read_whole_reply(response, retry_after, &cancellation).await?;
    if !status.is_success() {
        return Err(crate::error::provider_http_error(
            status,
            &response_text,
            retry_after,
            crate::error::ProviderRequest::Completion,
        ));
    }

    let response_json: serde_json::Value = serde_json::from_str(&response_text)
        .map_err(|error| MekaError::Provider(format!("invalid JSON response: {error}")))?;
    if let Some(message_id) = response_json.get("id").and_then(|id| id.as_str()) {
        attribution.record_message_id(message_id);
    }
    let (message, stop_reason, usage) = parse_non_streaming_response(&response_json)?;
    Ok(crate::provider::Completion {
        message,
        stop_reason,
        usage,
        notices: redaction_notice.into_iter().collect(),
    })
}

/// One streaming Claude call, with the same retry-once policy as [`complete`]. A redaction notice
/// goes out as the first stream event so the frontend renders it before any provider text; the
/// agent's `run_streaming` turns it into a `FrontendEvent::Notice`.
pub(super) async fn stream<B: ClaudeBackend>(
    backend: &B,
    request: CompletionRequest<'_>,
    event_sender: mpsc::Sender<StreamEvent>,
    cancellation: CancellationToken,
) -> Result<()> {
    let CompletionRequest {
        system_prompt,
        messages,
        tools,
        thinking,
        attribution,
        ..
    } = request;
    let (body_json, redaction_notice) =
        build_body_within_budget(messages, backend.max_request_bytes(), |messages| {
            serialize_body(&backend.request_body(
                system_prompt,
                messages,
                tools,
                true,
                thinking,
                &attribution,
            ))
        })?;
    // A send error here means the consumer hung up already, which the SSE driver reports itself.
    if let Some(notice) = redaction_notice
        && let Err(error) = event_sender.send(StreamEvent::Notice(notice)).await
    {
        tracing::debug!("failed to forward redaction notice into stream: {error}");
    }
    let body_json = backend.finish_body(system_prompt, body_json)?;
    let body_length = body_json.len();

    let response = crate::oauth::send_with_one_refresh(
        backend,
        crate::error::ProviderRequest::Completion,
        |error| transport_error(body_length, error),
        || async {
            Ok(backend
                .authenticated_request(
                    backend.client().post(backend.endpoint()),
                    !tools.is_empty(),
                    true,
                    thinking,
                )
                .await?
                .body(body_json.clone()))
        },
        &cancellation,
    )
    .await?;
    backend.remember_request_id(&attribution, response.headers());
    drive_claude_sse_stream(
        response,
        event_sender,
        cancellation,
        attribution.previous_message.clone(),
    )
    .await
}

/// Read a Claude SSE response to its end, forwarding each event on the channel.
pub(super) async fn drive_claude_sse_stream(
    response: reqwest::Response,
    event_sender: mpsc::Sender<StreamEvent>,
    cancellation: CancellationToken,
    previous_message: Option<crate::provider::PreviousMessageSlot>,
) -> Result<()> {
    let mut protocol = ClaudeStream {
        previous_message,
        ..ClaudeStream::default()
    };
    match crate::provider::sse::drive(
        response,
        "Claude",
        &event_sender,
        &cancellation,
        &mut protocol,
    )
    .await?
    {
        End::Finished | End::ReceiverGone => Ok(()),
        // The byte stream ending is not the same as the message ending.
        //
        // An intermediary (a gateway named by `base_url`, a CDN edge, a load balancer closing an
        // idle connection) can terminate a chunked response cleanly mid-message. Treating that as
        // success hands the agent a half-written answer with `stop_reason` left at its `EndTurn`
        // default: no error, so no retry, and nothing to distinguish a truncated reply from a
        // complete one. Worse mid-tool-call, where the accumulated call is dropped entirely because
        // `ToolUseEnd` never arrives. Reporting it as a `StreamError` routes it to the same retry
        // path a dropped connection already takes.
        //
        // A stop reason already seen makes the message complete without `message_stop`: a gateway
        // that forwards the deltas and closes without the final frame delivers a whole answer.
        End::Ended if protocol.saw_terminal_event => Ok(()),
        End::Ended => Err(crate::provider::sse::stream_error(
            &event_sender,
            "stream ended before a stop reason".to_string(),
        )
        .await),
    }
}

/// The Claude driver's state between frames.
#[derive(Default)]
struct ClaudeStream {
    /// Where `message_start`'s id is recorded for the next request's
    /// `diagnostics.previous_message_id`; `None` on a stream that is not a conversation turn.
    previous_message: Option<crate::provider::PreviousMessageSlot>,
    current_tool_input: String,
    in_tool_use: bool,
    /// Retained past `ToolUseStart` so a call whose arguments never parse can be *rejected* by id
    /// rather than silently run with `{}`.
    current_tool_id: String,
    current_tool_name: String,
    /// Whether the message reached its end rather than the byte stream simply stopping.
    saw_terminal_event: bool,
    in_thinking: bool,
    current_thinking_signature: Option<String>,
}

#[async_trait::async_trait]
impl crate::provider::sse::Protocol for ClaudeStream {
    async fn frame(
        &mut self,
        event: eventsource_stream::Event,
        event_sender: &mpsc::Sender<StreamEvent>,
    ) -> Result<Step> {
        let Some(data) = crate::provider::sse::frame_json("Claude", &event.data) else {
            return Ok(Step::Continue);
        };
        match claude_frame_name(&data, &event.event) {
            "content_block_start" => {
                let Some(content_block) = data.get("content_block") else {
                    return Ok(Step::Continue);
                };
                let block_type = content_block
                    .get("type")
                    .and_then(|block_type| block_type.as_str())
                    .unwrap_or("");

                if block_type == "thinking" {
                    self.in_thinking = true;
                    // Both schemas make `signature` required on a thinking block
                    // sent back, and require it verbatim, but it does not always
                    // arrive as a `signature_delta`. Anthropic opens the block with
                    // an empty one and fills it by delta; OpenRouter sends no delta
                    // at all for a non-Anthropic model, leaving that empty string
                    // as the value. Losing it there costs every later request in
                    // the session, rejected for the missing field.
                    //
                    // Assigned rather than merged, so a block that opens without
                    // the field cannot inherit the signature of an earlier one.
                    self.current_thinking_signature = content_block
                        .get("signature")
                        .and_then(|signature| signature.as_str())
                        .map(str::to_string);
                    // Announce the block itself, before any estimate: this is the
                    // earliest point the pause becomes explainable, and on a
                    // redacted block it is otherwise the only thing that happens
                    // for seconds at a time.
                    if event_sender
                        .send(StreamEvent::ThinkingProgress {
                            estimated_tokens: None,
                        })
                        .await
                        .is_err()
                    {
                        return Ok(Step::ReceiverGone);
                    }
                } else if block_type == "redacted_thinking" {
                    // The opaque `data` arrives whole in the start event; forward
                    // it so the agent can replay it verbatim on later turns.
                    if let Some(data) = content_block.get("data").and_then(|d| d.as_str())
                        && event_sender
                            .send(StreamEvent::RedactedThinking {
                                data: data.to_string(),
                            })
                            .await
                            .is_err()
                    {
                        return Ok(Step::ReceiverGone);
                    }
                } else if block_type == "tool_use" {
                    let id = content_block
                        .get("id")
                        .and_then(|id| id.as_str())
                        .ok_or_else(|| {
                            MekaError::Provider("tool_use block missing 'id' field".to_string())
                        })?
                        .to_string();
                    let name = content_block
                        .get("name")
                        .and_then(|name| name.as_str())
                        .ok_or_else(|| {
                            MekaError::Provider("tool_use block missing 'name' field".to_string())
                        })?
                        .to_string();

                    self.current_tool_input.clear();
                    self.in_tool_use = true;
                    self.current_tool_id = id.clone();
                    self.current_tool_name = name.clone();
                    if event_sender
                        .send(StreamEvent::ToolUseStart { id, name })
                        .await
                        .is_err()
                    {
                        return Ok(Step::ReceiverGone);
                    }
                }
            }
            "content_block_delta" => {
                let Some(delta) = data.get("delta") else {
                    return Ok(Step::Continue);
                };
                let delta_type = delta
                    .get("type")
                    .and_then(|delta_type| delta_type.as_str())
                    .unwrap_or("");

                match delta_type {
                    "thinking_delta" => {
                        // `estimated_tokens` is the server's running count of
                        // thinking spent so far (the `thinking-token-count` beta).
                        // It is the only progress signal on a redacted block,
                        // where `thinking` is `""` on every delta. The final delta
                        // of a block carries `null`; skipping it leaves the last
                        // real figure on screen rather than blanking the display
                        // just before the block ends.
                        if let Some(estimated) =
                            delta.get("estimated_tokens").and_then(|t| t.as_u64())
                            && event_sender
                                .send(StreamEvent::ThinkingProgress {
                                    estimated_tokens: Some(estimated),
                                })
                                .await
                                .is_err()
                        {
                            return Ok(Step::ReceiverGone);
                        }
                        if let Some(thinking) = delta.get("thinking").and_then(|t| t.as_str())
                            && !thinking.is_empty()
                            && event_sender
                                .send(StreamEvent::ThinkingDelta(thinking.to_string()))
                                .await
                                .is_err()
                        {
                            return Ok(Step::ReceiverGone);
                        }
                    }
                    "text_delta" => {
                        if let Some(text) = delta.get("text").and_then(|text| text.as_str())
                            && !text.is_empty()
                            && event_sender
                                .send(StreamEvent::TextDelta(text.to_string()))
                                .await
                                .is_err()
                        {
                            return Ok(Step::ReceiverGone);
                        }
                    }
                    "signature_delta" => {
                        if let Some(sig) = delta.get("signature").and_then(|s| s.as_str()) {
                            self.current_thinking_signature = Some(
                                self.current_thinking_signature
                                    .take()
                                    .map_or_else(|| sig.to_string(), |existing| existing + sig),
                            );
                        }
                    }
                    "input_json_delta" => {
                        if let Some(partial_json) = delta
                            .get("partial_json")
                            .and_then(|partial_json| partial_json.as_str())
                        {
                            self.current_tool_input.push_str(partial_json);
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                if self.in_thinking {
                    self.in_thinking = false;
                    let signature = self.current_thinking_signature.take();
                    if event_sender
                        .send(StreamEvent::ThinkingComplete {
                            opaque: signature
                                .map(|signature| OpaqueReasoning::Signed { signature }),
                        })
                        .await
                        .is_err()
                    {
                        return Ok(Step::ReceiverGone);
                    }
                } else if self.in_tool_use {
                    // An empty accumulator is a legitimate zero-argument call and
                    // becomes `{}`. Arguments that arrived but do not parse are a
                    // different thing entirely, and are rejected rather than
                    // replaced.
                    let event = crate::provider::tool_use_event(
                        self.current_tool_id.clone(),
                        self.current_tool_name.clone(),
                        &self.current_tool_input,
                    );
                    if event_sender.send(event).await.is_err() {
                        return Ok(Step::ReceiverGone);
                    }
                    self.current_tool_input.clear();
                    self.in_tool_use = false;
                }
            }
            "message_delta" => {
                let Some(delta) = data.get("delta") else {
                    return Ok(Step::Continue);
                };
                if let Some(usage) = data.get("usage") {
                    let token_usage = parse_usage_object(usage);
                    if event_sender
                        .send(StreamEvent::Usage(token_usage))
                        .await
                        .is_err()
                    {
                        return Ok(Step::ReceiverGone);
                    }
                }
                if let Some(stop_reason_str) =
                    delta.get("stop_reason").and_then(|reason| reason.as_str())
                {
                    // The stop reason is what ends a message; `message_stop` is
                    // the framing around it. Requiring the frame would make meka
                    // strictly less tolerant than the wire format needs: a gateway
                    // named by `base_url` that forwards the deltas and closes
                    // without the final event delivers a complete answer, and
                    // every turn through it would fail. What the check is for (a
                    // cut mid-`content_block_delta`) never gets this far.
                    self.saw_terminal_event = true;
                    let stop_reason = parse_claude_stop_reason(stop_reason_str);
                    if event_sender
                        .send(StreamEvent::MessageEnd { stop_reason })
                        .await
                        .is_err()
                    {
                        return Ok(Step::ReceiverGone);
                    }
                }
            }
            "message_stop" => {
                self.saw_terminal_event = true;
                return Ok(Step::Finished);
            }
            "message_start" => {
                if let (Some(slot), Some(message_id)) = (
                    &self.previous_message,
                    data.get("message")
                        .and_then(|message| message.get("id"))
                        .and_then(|id| id.as_str()),
                ) {
                    crate::provider::record_message_id(slot, message_id);
                }
                if let Some(usage) = data.get("message").and_then(|m| m.get("usage")) {
                    let token_usage = parse_usage_object(usage);
                    if event_sender
                        .send(StreamEvent::Usage(token_usage))
                        .await
                        .is_err()
                    {
                        return Ok(Step::ReceiverGone);
                    }
                }
            }
            "ping" => {}
            // Anthropic can send this *after* the 200 response has already started streaming
            // (typically right after `message_start`, before any visible content), e.g.
            // `overloaded_error` during a capacity spike. Letting it fall into the `other`
            // catch-all below would make an overloaded turn look like it succeeded with truncated
            // or empty content, so it is forwarded on the channel for visibility and then returned
            // as the classified error for the caller to retry or not.
            "error" => {
                let error_type = data
                    .get("error")
                    .and_then(|error| error.get("type"))
                    .and_then(|kind| kind.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let message = data
                    .get("error")
                    .and_then(|error| error.get("message"))
                    .and_then(|message| message.as_str())
                    .unwrap_or("stream error event")
                    .to_string();
                return Err(crate::error::provider_stream_error(&error_type, message));
            }
            other => {
                tracing::debug!("unknown Claude SSE event: {other}");
            }
        }
        Ok(Step::Continue)
    }
}

/// Two independently-salted hashes plus the source length. See [`downscale_cache_key`].
type DownscaleCacheKey = (u64, u64, usize);

/// Downscaled payloads, keyed by a hash of the base64 they were made from.
///
/// The same oversized image rides in the conversation on *every* turn, and without this each turn
/// decodes it, resizes it and re-encodes a PNG again, identical work for an identical result,
/// inside the request-building path. A 4000x3000 screenshot costs tens of milliseconds a turn that
/// way, and a session that pasted three of them pays it three times over for as long as they stay
/// in the window.
///
/// Hashed rather than keyed on the string itself so the map does not hold a second copy of every
/// original. A hash collision would serve the wrong image, which is why the stored entry keeps the
/// source length as a cheap discriminator; `DefaultHasher` is not a security boundary and is not
/// being asked to be one, since both inputs are already in this process's memory.
///
/// Cleared wholesale when full rather than evicted one at a time: the working set is the images in
/// one conversation, so either they all fit or the conversation has moved on.
static DOWNSCALE_CACHE: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<DownscaleCacheKey, String>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Entries the downscale cache holds before it is cleared. Each is one base64 PNG, so this bounds
/// residency by roughly this many downscaled images.
const DOWNSCALE_CACHE_ENTRIES: usize = 16;

fn downscale_cache_key(source_base64: &str) -> DownscaleCacheKey {
    use std::hash::{Hash, Hasher};
    // Two independently-seeded hashes plus the length. A single 64-bit hash keys a *whole image*
    // on one collision: two different screenshots landing on the same bucket would serve the
    // wrong one to the model, silently, with the right length. 128 bits of discriminator puts that
    // past the point where it can happen by accident, which is the only way it can happen here:
    // both inputs are already in this process's memory, so there is no attacker to defend against.
    // The salt must be a constant, not a fresh `RandomState`: the key has to be reproducible
    // between the `put` and the `get` that follows it, and a per-call seed makes every lookup miss.
    const SALT: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source_base64.hash(&mut hasher);
    let mut second = std::collections::hash_map::DefaultHasher::new();
    SALT.hash(&mut second);
    source_base64.hash(&mut second);
    (hasher.finish(), second.finish(), source_base64.len())
}

fn downscale_cache_get(source_base64: &str) -> Option<String> {
    let cache = crate::sync::lock(&DOWNSCALE_CACHE);
    cache.get(&downscale_cache_key(source_base64)).cloned()
}

fn downscale_cache_put(source_base64: &str, downscaled_base64: &str) {
    let mut cache = crate::sync::lock(&DOWNSCALE_CACHE);
    if cache.len() >= DOWNSCALE_CACHE_ENTRIES {
        cache.clear();
    }
    cache.insert(
        downscale_cache_key(source_base64),
        downscaled_base64.to_string(),
    );
}

/// Walk `messages` and downscale any image whose pixel dimensions exceed
/// [`MAX_IMAGE_DIMENSION_PX`] on either axis. The body bytes (base64) for those images are replaced
/// with a re-encoded PNG that fits within the cap; smaller images are left alone. Returns
/// `Cow::Borrowed` when no work was needed.
///
/// Anthropic-specific: the 2000 px cap only matters for Anthropic's multi-image requests, so the
/// other backends do not run it. Decode and resize cost is incurred per turn for each oversized
/// image, but typical sessions have few, and the cheap [`crate::image::read_image_dimensions`]
/// header read short-circuits the common case.
pub(super) fn downscale_oversized_images(messages: &[Message]) -> Cow<'_, [Message]> {
    use base64::Engine;
    use image::ImageFormat;

    fn parse_format(media_type: &str) -> Option<ImageFormat> {
        ImageFormat::from_mime_type(media_type)
    }

    // True when this image decodes and exceeds the per-axis pixel cap.
    fn oversized(source: &crate::image::ImageSource) -> bool {
        let Some(format) = parse_format(source.media_type()) else {
            return false;
        };
        let Some(data) = source.base64_data() else {
            return false;
        };
        let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(data) else {
            return false;
        };
        crate::image::read_image_dimensions(&bytes, format)
            .map(|(w, h)| w > MAX_IMAGE_DIMENSION_PX || h > MAX_IMAGE_DIMENSION_PX)
            .unwrap_or(false)
    }

    // Re-encode `source` to a within-cap PNG in place; no-op if it can't be decoded or already
    // fits. Served from the cache when this exact payload has been downscaled before.
    fn downscale_in_place(source: &mut crate::image::ImageSource) {
        // A reference has no bytes to downscale; hydration resolves every one before a request.
        let crate::image::ImageSource::Base64 { media_type, data } = source else {
            return;
        };
        let Some(format) = parse_format(media_type) else {
            return;
        };
        if let Some(cached) = downscale_cache_get(data) {
            *media_type = "image/png".to_string();
            *data = cached;
            return;
        }
        let original = data.clone();
        let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(&*data) else {
            return;
        };
        let Ok((w, h)) = crate::image::read_image_dimensions(&bytes, format) else {
            return;
        };
        if w <= MAX_IMAGE_DIMENSION_PX && h <= MAX_IMAGE_DIMENSION_PX {
            return;
        }
        match crate::image::downscale_to_dim_cap(&bytes, format, MAX_IMAGE_DIMENSION_PX) {
            Ok(png) => {
                let encoded = base64::engine::general_purpose::STANDARD.encode(&png);
                downscale_cache_put(&original, &encoded);
                *media_type = "image/png".to_string();
                *data = encoded;
            }
            Err(error) => {
                tracing::warn!("failed to downscale {w}x{h} {media_type} image: {error}",);
            }
        }
    }

    // First pass: whether any image, tool-result or input (`ContentBlock::Image`), needs
    // downscaling. A cheap header read; when nothing is oversized the clone and rewrite are
    // skipped.
    let needs_work = messages.iter().any(|message| {
        message.content.iter().any(|block| match block {
            ContentBlock::ToolResult { content, .. } => content.iter().any(
                |item| matches!(item, ToolResultContent::Image { source } if oversized(source)),
            ),
            ContentBlock::Image { source } => oversized(source),
            _ => false,
        })
    });
    if !needs_work {
        return Cow::Borrowed(messages);
    }

    let mut owned: Vec<Message> = messages.to_vec();
    for message in owned.iter_mut() {
        for block in message.content.iter_mut() {
            match block {
                ContentBlock::ToolResult { content, .. } => {
                    for item in content.iter_mut() {
                        if let ToolResultContent::Image { source } = item {
                            downscale_in_place(source);
                        }
                    }
                }
                ContentBlock::Image { source } => downscale_in_place(source),
                _ => {}
            }
        }
    }
    Cow::Owned(owned)
}

/// Serialize a Claude request body: downscale oversized images first, then the budget every
/// backend applies, [`crate::provider::budget::fit_body_to_budget`]. Both Claude providers run
/// this; the caller supplies the body builder via `build` so each provider's thinking / metadata
/// wiring stays in its own file. The downscale is the Anthropic-specific half, for the 2000 px
/// multi-image cap; the redaction and the refusal are the shared half.
pub(super) fn build_body_within_budget<F>(
    messages: &[Message],
    max_request_bytes: usize,
    build: F,
) -> Result<(String, Option<crate::frontend::Notice>)>
where
    F: FnMut(&[Message]) -> Result<String>,
{
    let prepared = downscale_oversized_images(messages);
    crate::provider::budget::fit_body_to_budget(prepared.as_ref(), max_request_bytes, build)
}

#[cfg(test)]
mod tests {
    /// `message_start` is where the streaming path learns the message id the next request names
    /// as `diagnostics.previous_message_id`.
    #[tokio::test]
    async fn message_start_records_the_message_id_on_the_attribution() {
        let slot = crate::provider::PreviousMessageSlot::default();
        let mut protocol = super::ClaudeStream {
            previous_message: Some(std::sync::Arc::clone(&slot)),
            ..super::ClaudeStream::default()
        };
        let (sender, _receiver) = tokio::sync::mpsc::channel(8);
        let event = eventsource_stream::Event {
            event: "message_start".to_string(),
            data: r#"{"type":"message_start","message":{"id":"msg_011Cepw4KgcVvSiBJbcdRbqv","usage":{"input_tokens":1,"output_tokens":0}}}"#.to_string(),
            id: String::new(),
            retry: None,
        };
        crate::provider::sse::Protocol::frame(&mut protocol, event, &sender)
            .await
            .expect("frame");
        assert_eq!(
            crate::sync::lock(&slot).as_deref(),
            Some("msg_011Cepw4KgcVvSiBJbcdRbqv")
        );
    }

    use super::*;
    use crate::image::ImageSource;

    /// Drive the SSE decoder over a canned body, returning what it emitted and how it ended.
    ///
    /// Built from an `http::Response` rather than a socket: these are decoder properties, and the
    /// audit's standing complaint about this module was that every test sat above the
    /// `StreamEvent` boundary and so could not see them at all.
    async fn decode_sse(body: &str) -> (Vec<StreamEvent>, Result<()>) {
        let response: reqwest::Response = http::Response::builder()
            .status(200)
            .header("content-type", "text/event-stream")
            .body(body.to_string())
            .expect("build response")
            .into();
        let (sender, mut receiver) = mpsc::channel(64);
        let outcome =
            drive_claude_sse_stream(response, sender, CancellationToken::new(), None).await;
        let mut events = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            events.push(event);
        }
        (events, outcome)
    }

    /// A gateway that forwards the data and drops the `event:` lines still delivers the turn: the
    /// frame is named by its `type`. Dispatching on the `event:` line alone discarded every frame.
    #[tokio::test]
    async fn a_stream_without_event_lines_is_read_by_its_data_types() {
        let (events, outcome) = decode_sse(concat!(
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        ))
        .await;

        assert!(outcome.is_ok(), "expected success, got {:?}", outcome.err());
        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::TextDelta(text) if text == "hi")),
            "the text must arrive: {events:?}",
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::MessageEnd { .. })),
            "and so must the stop reason: {events:?}",
        );
    }

    /// The breakpoint lands on the last block that is actually sent. Attached before the trailing
    /// thinking strip, it left with the block it was on.
    #[test]
    fn the_breakpoint_survives_the_trailing_thinking_strip() {
        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "answer".to_string(),
                },
                ContentBlock::Thinking {
                    thinking: "trailing".to_string(),
                    opaque: None,
                },
            ],
        };
        let converted = convert_messages_to_claude_content(&[message], CacheBreakpoint::Ephemeral);
        let content = converted[0]["content"].as_array().expect("content");
        assert_eq!(
            content.len(),
            1,
            "the trailing thinking block is stripped: {content:?}"
        );
        assert_eq!(
            content[0]["cache_control"],
            serde_json::json!({"type": "ephemeral"})
        );
    }

    /// A turn's two blocks reach this wire as two text blocks in order, the context ahead of the
    /// words, so the model reads what meka injected before what the user typed.
    #[test]
    fn the_context_block_precedes_the_words() {
        let message = Message::user_turn("[Permission context]", "hello", Vec::new());
        let converted = convert_messages_to_claude_content(&[message], CacheBreakpoint::Ephemeral);
        let content = converted[0]["content"].as_array().expect("content");
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "[Permission context]");
        assert_eq!(content[1]["text"], "hello");
    }

    /// A blob reference the store did not resolve goes out as the placeholder, not as a `source`
    /// object the API would reject along with the whole request.
    #[test]
    fn an_unresolved_blob_is_sent_as_the_placeholder() {
        let source = crate::image::ImageSource::Blob {
            hash: "abc".to_string(),
            media_type: "image/png".to_string(),
            size: 3,
        };
        let message = Message {
            role: Role::User,
            content: vec![
                ContentBlock::Image {
                    source: source.clone(),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "t1".to_string(),
                    content: vec![ToolResultContent::Image { source }],
                    is_error: false,
                },
            ],
        };
        let converted = convert_messages_to_claude_content(&[message], CacheBreakpoint::Ephemeral);
        let content = converted[0]["content"].as_array().expect("content");
        assert_eq!(content[0]["type"], "text");
        assert_eq!(
            content[0]["text"],
            crate::image::UNRESOLVED_IMAGE_PLACEHOLDER
        );
        assert_eq!(content[1]["content"][0]["type"], "text");
        assert_eq!(
            content[1]["content"][0]["text"],
            crate::image::UNRESOLVED_IMAGE_PLACEHOLDER
        );
    }

    /// The ceiling is the profile's, not Anthropic's: a body well under 30 MiB is still redacted
    /// when the profile says so, and refused when redaction cannot bring it under.
    #[test]
    fn the_request_ceiling_is_the_profiles() {
        let message = Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "x".repeat(2_000),
            }],
        };
        let build = |messages: &[Message]| {
            serde_json::to_string(messages).map_err(|error| MekaError::Provider(error.to_string()))
        };
        let (body, notice) = build_body_within_budget(std::slice::from_ref(&message), 4_000, build)
            .expect("under the ceiling");
        assert!(body.len() <= 4_000 && notice.is_none());
        let refused = build_body_within_budget(&[message], 1_000, build)
            .expect_err("text cannot be redacted, so the profile's ceiling refuses it");
        assert!(
            refused.to_string().contains("max_request_bytes"),
            "{refused}"
        );
        // The variant the turn's `refusal_may_blame_content` keys on: as `Provider` the refusal
        // ended the turn with nothing degraded, and the next turn carried the same body.
        //
        // `RequestTooLarge` rather than `InvalidRequest`, which means the *provider* refused a
        // body: nothing was sent here, so a host publishing this as a provider failure sent its
        // caller looking for an upstream response that does not exist. It arms the same retry.
        assert!(
            matches!(refused, MekaError::RequestTooLarge(_)),
            "the refusal must arm the degrade-and-retry: {refused:?}"
        );
    }

    /// A gateway that forwards every delta and closes without Anthropic's final framing event has
    /// delivered a complete message: `message_delta` already carried the stop reason.
    ///
    /// Requiring `message_stop` itself made meka stricter than the format needs and broke every
    /// turn through such a shim, for a check whose actual subject -- a response cut mid-content --
    /// never reaches a stop reason at all.
    #[tokio::test]
    async fn a_stream_that_stops_after_its_stop_reason_is_complete() {
        let (events, outcome) = decode_sse(concat!(
            "event: content_block_delta\n",
            "data: {\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "event: message_delta\n",
            "data: {\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        ))
        .await;

        assert!(outcome.is_ok(), "expected success, got {:?}", outcome.err());
        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::MessageEnd { .. })),
            "and the stop reason must still reach the agent: {events:?}",
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, StreamEvent::Error(_))),
            "no error may be reported: {events:?}",
        );
    }

    /// Unparseable tool arguments are the model's intent, mangled. Running the tool with `{}`
    /// instead executes something the model never asked for -- `write_file` with no path, a shell
    /// command with no command -- and reports success for it. Rejecting hands the model back a
    /// result it can act on.
    #[tokio::test]
    async fn a_tool_call_with_unparseable_arguments_is_rejected_not_run_empty() {
        let (events, outcome) = decode_sse(concat!(
            "event: content_block_start\n",
            "data: {\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"write_file\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\": \"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n",
        ))
        .await;
        outcome.expect("a rejected tool call is not a stream failure");

        assert!(
            events.iter().any(|event| matches!(
                event,
                StreamEvent::ToolCallRejected { name, .. } if name == "write_file"
            )),
            "the call must be rejected: {events:?}",
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, StreamEvent::ToolUseEnd { .. })),
            "and must not also be dispatched: {events:?}",
        );
    }

    /// The case the check exists for: the bytes stop partway through the content, so no stop reason
    /// ever arrives and the agent would otherwise commit a half-written answer as a complete one.
    #[tokio::test]
    async fn a_stream_cut_before_its_stop_reason_is_an_error() {
        let (events, outcome) = decode_sse(concat!(
            "event: content_block_delta\n",
            "data: {\"delta\":{\"type\":\"text_delta\",\"text\":\"half an ans\"}}\n\n",
        ))
        .await;

        assert!(
            matches!(outcome, Err(MekaError::StreamError(_))),
            "a truncated stream must reach the retry path, got {outcome:?}",
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::Error(_))),
            "and say so on the stream: {events:?}",
        );
    }

    fn thinking_opaque(events: &[StreamEvent]) -> Option<OpaqueReasoning> {
        events
            .iter()
            .find_map(|event| match event {
                StreamEvent::ThinkingComplete { opaque } => Some(opaque.clone()),
                _ => None,
            })
            .expect("a thinking block must complete")
    }

    /// A signature carried only by `content_block_start` must survive to the echo. OpenRouter sends
    /// it there and no `signature_delta` at all for a non-Anthropic model, and its schema, like
    /// Anthropic's, makes the field required on the way back. Dropping it rejected every request
    /// after the first with `invalid_union` at `messages[n].content`, killing the session.
    #[tokio::test]
    async fn a_thinking_signature_sent_only_on_the_start_event_is_kept() {
        let (events, outcome) = decode_sse(concat!(
            "event: content_block_start\n",
            "data: {\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        ))
        .await;

        assert!(outcome.is_ok(), "expected success, got {:?}", outcome.err());
        assert_eq!(
            thinking_opaque(&events),
            Some(OpaqueReasoning::Signed {
                signature: String::new(),
            }),
            "an empty signature is still the value the API returned: {events:?}",
        );
    }

    /// Anthropic sends the real signature as trailing deltas, so the start event must seed the
    /// accumulator rather than displace what those deltas append.
    #[tokio::test]
    async fn a_thinking_signature_sent_as_deltas_still_accumulates() {
        let (events, outcome) = decode_sse(concat!(
            "event: content_block_start\n",
            "data: {\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"abc\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"def\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        ))
        .await;

        assert!(outcome.is_ok(), "expected success, got {:?}", outcome.err());
        assert_eq!(
            thinking_opaque(&events),
            Some(OpaqueReasoning::Signed {
                signature: "abcdef".to_string(),
            }),
            "{events:?}",
        );
    }

    /// The other direction: a backend that returns no signature at all gets none back. Anthropic
    /// requires the field "exactly as returned by the API", so inventing an empty one to satisfy
    /// the schema would be sending a value that was never issued.
    #[tokio::test]
    async fn a_thinking_block_returned_unsigned_stays_unsigned() {
        let (events, outcome) = decode_sse(concat!(
            "event: content_block_start\n",
            "data: {\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        ))
        .await;

        assert!(outcome.is_ok(), "expected success, got {:?}", outcome.err());
        assert_eq!(thinking_opaque(&events), None, "{events:?}");
    }

    /// One accumulator serves every block, and only `content_block_stop` clears it. A gateway that
    /// omits that event would otherwise hand the next block the previous one's signature, pairing a
    /// signature with reasoning it does not authenticate -- which Anthropic rejects as a modified
    /// block. `anthropic-messages` reaches any `base_url`, so the stream is untrusted input.
    #[tokio::test]
    async fn an_unterminated_thinking_block_does_not_lend_its_signature_to_the_next() {
        let (events, outcome) = decode_sse(concat!(
            "event: content_block_start\n",
            "data: {\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"FIRST\"}}\n\n",
            "event: content_block_start\n",
            "data: {\"index\":1,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"index\":1,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"unrelated\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"index\":1}\n\n",
            "event: message_delta\n",
            "data: {\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        ))
        .await;

        assert!(outcome.is_ok(), "expected success, got {:?}", outcome.err());
        assert_eq!(
            thinking_opaque(&events),
            None,
            "the second block was opened without a signature: {events:?}",
        );
    }

    #[test]
    fn a_claude_base_url_drops_a_trailing_version_segment() {
        // The shape a gateway publishes for its Anthropic endpoint when it mirrors the OpenAI one.
        assert_eq!(
            normalize_claude_base_url("https://api.synthetic.new/anthropic/v1"),
            "https://api.synthetic.new/anthropic"
        );
        // Trailing slashes go first, so the version segment is still recognized behind them.
        assert_eq!(
            normalize_claude_base_url("https://api.synthetic.new/anthropic/v1/"),
            "https://api.synthetic.new/anthropic"
        );
        assert_eq!(
            normalize_claude_base_url("https://api.anthropic.com/v1"),
            "https://api.anthropic.com"
        );
    }

    #[test]
    fn a_claude_base_url_already_in_the_canonical_shape_is_untouched() {
        assert_eq!(
            normalize_claude_base_url("https://api.anthropic.com"),
            "https://api.anthropic.com"
        );
        assert_eq!(
            normalize_claude_base_url("https://api.synthetic.new/anthropic/"),
            "https://api.synthetic.new/anthropic"
        );
    }

    #[test]
    fn only_a_trailing_version_segment_is_dropped_from_a_claude_base_url() {
        // Cloudflare's AI Gateway puts the version early and the vendor last. Stripping a `/v1`
        // anywhere but the end would silently route to the wrong account.
        assert_eq!(
            normalize_claude_base_url(
                "https://gateway.ai.cloudflare.com/v1/account/gateway/anthropic"
            ),
            "https://gateway.ai.cloudflare.com/v1/account/gateway/anthropic"
        );
        // Exactly one segment: a doubled one is pathological either way, but stripping to the host
        // would discard a path the user did write.
        assert_eq!(
            normalize_claude_base_url("https://api.anthropic.com/v1/v1"),
            "https://api.anthropic.com/v1"
        );
        // `/v1` must be a whole segment, not a suffix of one.
        assert_eq!(
            normalize_claude_base_url("https://example.com/openaiv1"),
            "https://example.com/openaiv1"
        );
    }

    #[test]
    fn convert_messages_serializes_input_image() {
        let message = crate::conversation::Message::user_with_images("look at this", vec![
            ImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: "QUJD".to_string(),
            },
        ]);
        let converted = convert_messages_to_claude_content(&[message], CacheBreakpoint::OneHour);
        let blocks = converted[0]["content"].as_array().expect("content array");
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "image");
        assert_eq!(blocks[1]["source"]["type"], "base64");
        assert_eq!(blocks[1]["source"]["media_type"], "image/png");
        assert_eq!(blocks[1]["source"]["data"], "QUJD");
    }

    #[test]
    fn parse_redacted_thinking_block() {
        let response = serde_json::json!({
            "content": [
                { "type": "redacted_thinking", "data": "ENCRYPTED_OPAQUE_BLOB" },
                { "type": "text", "text": "done" },
            ],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 1, "output_tokens": 1 },
        });
        let (message, ..) = parse_non_streaming_response(&response).unwrap();
        assert!(matches!(
            &message.content[0],
            ContentBlock::RedactedThinking { data } if data == "ENCRYPTED_OPAQUE_BLOB"
        ));
    }

    #[test]
    fn redacted_thinking_round_trips_verbatim() {
        let message = crate::conversation::Message {
            role: crate::conversation::Role::Assistant,
            content: vec![
                ContentBlock::RedactedThinking {
                    data: "ENCRYPTED_OPAQUE_BLOB".to_string(),
                },
                ContentBlock::Text {
                    text: "hi".to_string(),
                },
            ],
        };
        let converted = convert_messages_to_claude_content(&[message], CacheBreakpoint::OneHour);
        let block = &converted[0]["content"].as_array().unwrap()[0];
        assert_eq!(block["type"], "redacted_thinking");
        assert_eq!(block["data"], "ENCRYPTED_OPAQUE_BLOB");
        // The opaque data must never be wrapped in `thinking`/`signature` fields.
        assert!(block.get("thinking").is_none());
        assert!(block.get("signature").is_none());
    }

    /// The mirror of the Responses encoder's refusal, for a session recorded against OpenAI and
    /// resumed under Claude. Sealed reasoning is not a Claude signature, so it must not be
    /// serialized as one; the block goes out with its summary text and unsigned.
    #[test]
    fn sealed_reasoning_is_never_sent_to_claude_as_a_signature() {
        let message = crate::conversation::Message {
            role: crate::conversation::Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "a summary".to_string(),
                    opaque: Some(OpaqueReasoning::Sealed {
                        encrypted_content: "OPENAI_SEALED".to_string(),
                        id: Some("rs_1".to_string()),
                    }),
                },
                // Trailing thinking is stripped from the last assistant message, so the block has
                // to be followed by something for this to be testing the encoder at all.
                ContentBlock::Text {
                    text: "answer".to_string(),
                },
            ],
        };
        let converted = convert_messages_to_claude_content(&[message], CacheBreakpoint::OneHour);
        let block = &converted[0]["content"].as_array().expect("content")[0];

        assert_eq!(block["type"], "thinking");
        assert_eq!(block["thinking"], "a summary");
        assert!(block.get("signature").is_none(), "{block}");
        assert!(
            !serde_json::to_string(&converted)
                .expect("serialize")
                .contains("OPENAI_SEALED")
        );
    }

    #[test]
    fn empty_thinking_block_with_signature_serializes_signature() {
        let message = crate::conversation::Message {
            role: crate::conversation::Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: String::new(),
                    opaque: Some(OpaqueReasoning::Signed {
                        signature: "SIG620".to_string(),
                    }),
                },
                ContentBlock::Text {
                    text: "answer".to_string(),
                },
            ],
        };
        let converted = convert_messages_to_claude_content(&[message], CacheBreakpoint::OneHour);
        let block = &converted[0]["content"].as_array().unwrap()[0];
        assert_eq!(block["type"], "thinking");
        assert_eq!(block["thinking"], "");
        assert_eq!(block["signature"], "SIG620");
    }

    #[test]
    fn redact_oldest_images_redacts_input_image() {
        let big = "x".repeat(2_000);
        let messages = vec![
            crate::conversation::Message::user_with_images("first", vec![ImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: big,
            }]),
            // A trailing message: the last message is never redacted.
            crate::conversation::Message::assistant_text("ok"),
        ];
        let (redacted, stats) = redact_oldest_images(&messages, 1_000);
        assert_eq!(stats.images_redacted, 1);
        assert!(stats.bytes_freed >= 2_000);
        // The input image block became a text placeholder.
        assert!(matches!(&redacted[0].content[1], ContentBlock::Text { .. }));
    }

    #[test]
    fn each_thinking_mode_writes_its_own_encoding_and_ceiling() {
        // The encoding now comes from the profile, not from the model name, so the same model can
        // legitimately be asked for either form - which is what an arbitrary Anthropic-compatible
        // endpoint needs.
        let mut adaptive = serde_json::Map::new();
        insert_thinking_fields(&mut adaptive, ThinkingMode::Adaptive, 10_000, None, None);
        assert_eq!(adaptive["max_tokens"], 64_000);
        assert_eq!(adaptive["thinking"]["type"], "adaptive");
        assert!(adaptive["thinking"].get("budget_tokens").is_none());
        // No `display` field: real Claude Code sends `{type:"adaptive"}` only.
        assert!(adaptive["thinking"].get("display").is_none());

        let mut budgeted = serde_json::Map::new();
        insert_thinking_fields(&mut budgeted, ThinkingMode::Budgeted, 10_000, None, None);
        assert_eq!(budgeted["max_tokens"], 32_000);
        assert_eq!(budgeted["thinking"]["type"], "enabled");
        assert_eq!(budgeted["thinking"]["budget_tokens"], 10_000);

        let mut off = serde_json::Map::new();
        insert_thinking_fields(&mut off, ThinkingMode::Off, 10_000, None, None);
        assert_eq!(off["max_tokens"], 32_000);
        assert!(off.get("thinking").is_none());
    }

    #[test]
    fn a_max_output_override_replaces_the_default_but_never_undercuts_the_budget() {
        let mut adaptive = serde_json::Map::new();
        insert_thinking_fields(
            &mut adaptive,
            ThinkingMode::Adaptive,
            10_000,
            Some(80_000),
            None,
        );
        assert_eq!(adaptive["max_tokens"], 80_000);

        let mut off = serde_json::Map::new();
        insert_thinking_fields(&mut off, ThinkingMode::Off, 10_000, Some(50_000), None);
        assert_eq!(off["max_tokens"], 50_000);

        // Budgeted draws the budget from `max_tokens`, so an override at or below it would be a
        // 400. Clamped rather than rejected: the config guard catches the configured case, and this
        // keeps a request valid regardless.
        let mut clamped = serde_json::Map::new();
        insert_thinking_fields(
            &mut clamped,
            ThinkingMode::Budgeted,
            20_000,
            Some(5_000),
            None,
        );
        assert_eq!(clamped["max_tokens"], 20_001);
        assert_eq!(clamped["thinking"]["budget_tokens"], 20_000);
    }

    #[test]
    fn an_override_turns_thinking_off_and_never_on() {
        assert_eq!(
            effective_thinking(ThinkingOverride::Inherit, ThinkingMode::Adaptive),
            ThinkingMode::Adaptive
        );
        // The compaction summary sends `Off` so it doesn't pay for reasoning.
        assert_eq!(
            effective_thinking(ThinkingOverride::Off, ThinkingMode::Adaptive),
            ThinkingMode::Off
        );
        assert_eq!(
            effective_thinking(ThinkingOverride::Off, ThinkingMode::Budgeted),
            ThinkingMode::Off
        );
        // It cannot resurrect thinking for a profile that asked for none, so the two settings can
        // never disagree about whether a request asks for thinking.
        assert_eq!(
            effective_thinking(ThinkingOverride::Inherit, ThinkingMode::Off),
            ThinkingMode::Off
        );
    }

    #[test]
    fn only_the_4_x_line_supports_modern_features() {
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
    fn a_model_version_parses_from_the_name_and_ignores_the_date_stamp() {
        assert_eq!(parse_model_version("claude-opus-4-8"), Some((4, 8)));
        // Trailing date stamp is ignored (too many digits).
        assert_eq!(
            parse_model_version("claude-opus-4-6-20250514"),
            Some((4, 6))
        );
        assert_eq!(parse_model_version("claude-sonnet-4-5"), Some((4, 5)));
        // 3.x line carries the version before the family.
        assert_eq!(
            parse_model_version("claude-3-5-sonnet-20241022"),
            Some((3, 5))
        );
        // A single version segment -> minor defaults to 0.
        assert_eq!(parse_model_version("claude-3-opus-20240229"), Some((3, 0)));
        assert_eq!(
            parse_model_version("claude-sonnet-4-20250514"),
            Some((4, 0))
        );
        // No version-like segment at all.
        assert_eq!(parse_model_version("claude-custom"), None);
    }

    #[test]
    fn a_haiku_model_is_recognized_by_its_family_segment() {
        assert!(model_is_haiku("claude-haiku-4-5-20251001"));
        assert!(model_is_haiku("claude-haiku-4-5"));
        assert!(!model_is_haiku("claude-opus-4-6-20250514"));
        assert!(!model_is_haiku("claude-sonnet-4-20250514"));
    }

    #[test]
    fn temperature_is_sent_only_to_the_allowlisted_models() {
        // The allowlist: the 3.x line, Opus 4.0/4.1/4.5/4.6, Sonnet 4.0/4.5/4.6, Haiku 4.5. Dated
        // and canonical spellings both resolve (`claude-sonnet-4-20250514` parses as 4.0).
        for model in [
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20241022",
            "claude-opus-4-0",
            "claude-opus-4-1",
            "claude-opus-4-5-20251101",
            "claude-opus-4-6-20250514",
            "claude-sonnet-4-20250514",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
        ] {
            assert!(model_supports_temperature(model), "{model}");
        }
        // Sampling-params-removed models (400 on `temperature`).
        for model in [
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-mythos-5",
        ] {
            assert!(!model_supports_temperature(model), "{model}");
        }
        // The allowlist fails safe: anything it doesn't recognize (a model newer than this list)
        // omits `temperature` rather than earning a 400. This is what a denylist got wrong.
        for model in [
            "claude-opus-6",
            "claude-sonnet-6-0",
            "claude-future-experimental-7",
            "claude-custom",
        ] {
            assert!(!model_supports_temperature(model), "{model}");
        }
    }

    #[test]
    fn mid_conversation_system_is_sent_to_every_model_outside_the_denylist() {
        // Everything outside Claude Code's denylist sends the beta, which is every current model
        // and every future one.
        for model in [
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-opus-5-0",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-mythos-5",
            "claude-opus-7",
            "claude-something-new",
        ] {
            assert!(model_supports_mid_conversation_system(model), "{model}");
        }
        // The denylist itself: Opus 4.7 and down, Sonnet 4.6 and down, Haiku 4.5, Claude 3.x.
        for model in [
            "claude-opus-4-7",
            "claude-opus-4-6-20250514",
            "claude-opus-4-0",
            "claude-sonnet-4-6",
            "claude-sonnet-4-5",
            "claude-haiku-4-5-20251001",
            "claude-3-5-sonnet-20241022",
        ] {
            assert!(!model_supports_mid_conversation_system(model), "{model}");
        }
    }

    #[test]
    fn parse_claude_stop_reason_all_variants() {
        assert_eq!(parse_claude_stop_reason("end_turn"), StopReason::EndTurn);
        assert_eq!(parse_claude_stop_reason("tool_use"), StopReason::ToolUse);
        assert_eq!(
            parse_claude_stop_reason("max_tokens"),
            StopReason::MaxTokens
        );
        assert_eq!(
            parse_claude_stop_reason("refusal"),
            StopReason::Refusal(String::new())
        );
        // `pause_turn` is unrecognized and maps to `Unknown` carrying the raw string, so the
        // literal reaches the warn log and the empty-turn stand-in intact.
        assert_eq!(
            parse_claude_stop_reason("pause_turn"),
            StopReason::Unknown("pause_turn".to_string())
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
                source: ImageSource::Base64 {
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
    fn redact_no_op_when_under_threshold() {
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
    fn redact_drops_oldest_image_first() {
        // Two images: one in message[0] (older), one in message[1] (last). The helper must only
        // touch the older one; the last message carries the moving cache_control marker.
        let payload_a = "A".repeat(1024);
        let payload_b = "B".repeat(1024);
        let messages = vec![
            user_with_block(image_block("call_a", &payload_a)),
            user_with_block(image_block("call_b", &payload_b)),
        ];
        let (result, stats) = redact_oldest_images(&messages, 1);
        assert_eq!(stats.images_redacted, 1);
        assert_eq!(stats.bytes_freed, 1024);
        // Named tail-relative for the conversation to record: the older of two messages is two
        // from the end, and the image is the first item of its first block.
        assert_eq!(stats.positions, vec![crate::image::RedactedImage {
            from_end: 2,
            block: 0,
            item: Some(0),
        }]);
        let owned = match result {
            Cow::Owned(v) => v,
            Cow::Borrowed(_) => panic!("expected owned redacted vec"),
        };
        // message[0] image redacted to placeholder text.
        match &owned[0].content[0] {
            ContentBlock::ToolResult { content, .. } => match &content[0] {
                ToolResultContent::Text { text } => {
                    assert_eq!(text, IMAGE_REDACTION_PLACEHOLDER);
                }
                other => panic!("expected text placeholder, got {other:?}"),
            },
            other => panic!("expected ToolResult, got {other:?}"),
        }
        // message[1] (last) image untouched.
        match &owned[1].content[0] {
            ContentBlock::ToolResult { content, .. } => match &content[0] {
                ToolResultContent::Image { source } => {
                    assert_eq!(source.base64_data(), Some(&*payload_b));
                }
                other => panic!("expected untouched image, got {other:?}"),
            },
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn redact_stops_when_target_reached() {
        // Three images each 1 KiB. Target = 1500 bytes. Only the FIRST image should be redacted;
        // the second remains because we hit the budget after one (1024 >= 1500 is false, but
        // saturating_add gets us past after the first redaction since we then loop-check before the
        // second image is considered? No: the check is `bytes_dropped >= bytes_to_drop`, so 1024
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
    fn redact_preserves_last_message() {
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
                ToolResultContent::Image { source } => {
                    assert_eq!(source.base64_data(), Some(&*payload))
                }
                _ => panic!("last-message image must survive"),
            },
            _ => unreachable!(),
        }
    }

    #[test]
    fn redact_handles_no_images() {
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
                    source: crate::image::ImageSource::Base64 {
                        media_type: "image/png".to_string(),
                        data: base64_payload.to_string(),
                    },
                }],
                is_error: false,
            }],
        }
    }

    #[test]
    fn downscale_no_op_when_all_within_cap() {
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

    /// The same oversized image is downscaled once, not once per turn.
    ///
    /// It rides in the conversation for as long as it is in the window, and the request-building
    /// path ran a decode, a resize and a PNG encode over it on every turn -- identical work for an
    /// identical result. Asserted by counting the encodes: a cached hit must produce byte-identical
    /// output *and* not have gone through the encoder to get it, which is the part a naive
    /// equality check cannot see.
    #[test]
    fn a_repeated_image_is_downscaled_once_and_then_served_from_the_cache() {
        let big = synthesize_png_base64(2400, 1200);
        let message = |data: &str| {
            vec![crate::conversation::Message::user_with_images(
                "look",
                vec![crate::image::ImageSource::Base64 {
                    media_type: "image/png".to_string(),
                    data: data.to_string(),
                }],
            )]
        };

        let first = downscale_oversized_images(&message(&big)).into_owned();
        assert!(
            downscale_cache_get(&big).is_some(),
            "the first pass must populate the cache",
        );

        let second = downscale_oversized_images(&message(&big)).into_owned();
        let bytes_of = |messages: &[Message]| match &messages[0].content[1] {
            ContentBlock::Image { source } => source.base64_data().unwrap_or_default().to_string(),
            other => panic!("expected a downscaled image; got {other:?}"),
        };
        assert_eq!(
            bytes_of(&first),
            bytes_of(&second),
            "a cached hit must be the same payload",
        );

        // And a *different* image is not served the first one's result.
        let other = synthesize_png_base64(2400, 1600);
        let third = downscale_oversized_images(&message(&other)).into_owned();
        assert_ne!(
            bytes_of(&first),
            bytes_of(&third),
            "the key must distinguish two different sources",
        );
    }

    #[test]
    fn downscale_resizes_oversized_input_image() {
        use base64::Engine;
        use image::ImageFormat;
        // A user message with a top-level input image (ACP @-mention / pasted screenshot) must be
        // downscaled the same as a tool-result image, or Anthropic rejects the multi-image request.
        let big = synthesize_png_base64(2400, 1200);
        let messages = vec![crate::conversation::Message::user_with_images(
            "look",
            vec![crate::image::ImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: big,
            }],
        )];
        let owned = match downscale_oversized_images(&messages) {
            Cow::Owned(v) => v,
            Cow::Borrowed(_) => panic!("expected owned (input-image resize triggered)"),
        };
        // content[0] = Text, content[1] = the downscaled input image.
        match &owned[0].content[1] {
            ContentBlock::Image { source } => {
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(source.base64_data().expect("inline bytes"))
                    .expect("decode");
                let decoded =
                    image::load_from_memory_with_format(&bytes, ImageFormat::Png).expect("png");
                assert!(decoded.width() <= MAX_IMAGE_DIMENSION_PX);
                assert!(decoded.height() <= MAX_IMAGE_DIMENSION_PX);
            }
            other => panic!("expected downscaled input image; got {other:?}"),
        }
    }

    #[test]
    fn downscale_resizes_oversized_image() {
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
                        .decode(source.base64_data().expect("inline bytes"))
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
                ToolResultContent::Image { source } => {
                    assert_eq!(source.base64_data(), Some(&*small))
                }
                _ => panic!("small image should be untouched"),
            },
            _ => unreachable!(),
        }
    }

    /// Locks in the contract that `build_body_within_budget` returns a user-visible
    /// [`crate::frontend::Notice`] (rather than printing to stderr directly) when redaction kicks
    /// in. The agent loop then forwards it through `Frontend::emit`, which is how ACP clients see
    /// the redaction signal at all.
    #[test]
    fn build_body_within_budget_returns_notice_on_redaction() {
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

        let (body, notice) = build_body_within_budget(&messages, MAX_REQUEST_BYTES, build)
            .expect("redaction should succeed");
        assert_eq!(body, "{}");
        let notice = notice.expect("redaction must surface a Notice");
        assert_eq!(notice.level, crate::frontend::NoticeLevel::Info);
        let redaction = notice
            .redaction
            .expect("the notice carries what was removed, for the session to count");
        assert!(
            redaction.images >= 1 && redaction.bytes > 0,
            "{redaction:?}"
        );
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
    fn build_body_within_budget_no_notice_when_within_budget() {
        let messages = vec![assistant_text("hi")];
        let build = |_msgs: &[Message]| -> Result<String> { Ok("{}".to_string()) };
        let (body, notice) =
            build_body_within_budget(&messages, MAX_REQUEST_BYTES, build).expect("happy path");
        assert_eq!(body, "{}");
        assert!(notice.is_none());
    }
}
