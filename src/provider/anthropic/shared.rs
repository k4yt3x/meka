//! Helpers shared by [`super::messages::AnthropicMessagesProvider`] and
//! [`super::subscription::ClaudeSubscriptionProvider`]. Everything in this module is independent of
//! the authentication scheme: message/tool conversion to the Claude wire format, SSE streaming,
//! response parsing, per-model capability detection, the thinking override, and the one driver
//! both providers send through.

use std::{borrow::Cow, collections::HashSet};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    config::ThinkingMode,
    conversation::{ContentBlock, Message, OpaqueReasoning, ResultPart, Role, ToolResultContent},
    error::{MekaError, Result},
    provider::{
        CompletionRequest, Stop, StopReason, StreamEvent, ToolDefinition,
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
                "dropped the trailing '/v1' from Claude base URL '{trimmed}'; Meka appends it per request"
            );
            without_version.to_string()
        }
        None => trimmed,
    }
}

// The redaction itself, its stopping point and the refusal live in `crate::provider::budget`, since
// every backend applies them; the constant above is the Anthropic default ceiling they run against.
// The redaction tests live beside the downscale tests, hence the test-only import.
#[cfg(test)]
pub(super) use crate::provider::budget::redact_oldest_images;
pub(super) use crate::provider::budget::serialize_body;

/// Anthropic accepts up to 8000 px per axis on a *single*-image request, but rejects anything over
/// 2000 px on either axis once the request contains more than one image. Every image is downscaled
/// to fit so a session can freely accumulate images without tripping the multi-image cap. Enforced
/// by the Claude backends only; the others need not pay the resize cost.
pub(super) const MAX_IMAGE_DIMENSION_PX: u32 = 2000;

/// Extract a `TokenUsage` from an Anthropic `usage` object, as `message_start` and `message_delta`
/// carry it (`input_tokens`, `output_tokens`, `cache_creation_input_tokens`,
/// `cache_read_input_tokens`). Missing fields default to 0 (older API responses, or providers that
/// don't surface cache stats).
pub(super) fn parse_usage_object(usage: &serde_json::Value) -> TokenUsage {
    tracing::debug!("claude usage: {usage}");
    let field = |key: &str| usage.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
    TokenUsage {
        input_tokens: field("input_tokens"),
        output_tokens: field("output_tokens"),
        cache_creation_input_tokens: field("cache_creation_input_tokens"),
        cache_read_input_tokens: field("cache_read_input_tokens"),
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
/// [`ThinkingMode::Adaptive`] gets a fixed 128k ceiling, [`ThinkingMode::Budgeted`] gets
/// `max(budget*2, 32k)` with an explicit budget, and [`ThinkingMode::Off`] a flat 32k. A
/// `max_output_tokens` override (the profile knob) replaces whichever default would otherwise
/// apply; under `Budgeted` it is clamped to stay above `budget_tokens` (the API rejects
/// `max_tokens <= thinking.budget_tokens`).
///
/// The encoding comes from the profile rather than from the model name: this is what a request
/// reaching an arbitrary Anthropic-compatible endpoint needs, since Meka cannot tell which of the
/// two forms that endpoint implements.
///
/// `display` is the backend's to add, through [`with_display`]: `claude-subscription` sends Claude
/// Code's display mode, and `anthropic-messages` sends none, because the display values are a
/// first-party feature an arbitrary endpoint may not implement.
///
/// The `max_tokens` sent when the profile states no `max_output_tokens`: 128000 under adaptive
/// thinking, which is what Claude Code 2.1.322 sends for Opus 5.5 (verified by wire capture), the
/// model `profile add` suggests, and the most every model that takes adaptive thinking accepts;
/// 32000 otherwise, raised to twice the budget under budgeted thinking when that is more. Claude
/// Code reads the figure off a per-model catalog, which says the same for the rest of the 5.5
/// line, Fable 5.2 and Mythos 5.2 and 64000 for the rest of the line-up;
/// Meka carries one figure rather than that catalog, and the profile's `max_output_tokens` is
/// where a user states another. The API requires the field, so omitting it is not a way to ask
/// for a default.
pub(super) fn insert_thinking_fields(
    body: &mut serde_json::Map<String, serde_json::Value>,
    thinking: ThinkingMode,
    budget_tokens: u64,
    max_output_tokens: Option<u64>,
    display: Option<&str>,
) {
    match thinking {
        ThinkingMode::Adaptive => {
            let max_tokens = max_output_tokens.unwrap_or(128_000);
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
/// 2.1.284's gate, an **allowlist** of the models released before Opus 4.7, which still accept
/// sampling params: the Claude 3.x line, Opus 4.0/4.1/4.5/4.6, Sonnet 4.0/4.5/4.6, and Haiku 4.5.
/// Everything newer (Opus 4.7 and up, Sonnet 5, Fable/Mythos 5) rejects `temperature` with a 400.
///
/// The allowlist direction is the point: an unrecognized model, which in practice means one newer
/// than this list, resolves to `false` and the parameter is omitted. A denylist would instead send
/// `temperature` to every future model and earn a 400 until the list was updated. Matching is by
/// family + [`parse_model_version`] rather than Claude Code's exact string equality because Meka
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
/// A denylist mirroring Claude Code 2.1.322's own gate, which excludes the Claude 3.x line, Opus
/// 4.0/4.1, Sonnet 4.0/4.5 and Haiku 4.5 and sends the field to everything else on the first-party
/// endpoint. Same reasoning and same single caller as
/// [`model_supports_mid_conversation_tool_changes`]: an unrecognized name here is one *newer* than
/// the list, and effort is what those models are for.
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
/// it to what that model accepts. `medium` is the 2.1.322 table's figure for Opus 5.5, the model
/// `profile add` suggests, and for the rest of the 5.5 line, Fable 5.2 and Mythos 5.2; Opus 4.7
/// says `xhigh`, the other entries say `high`, and `high` is what Claude Code falls back to for a
/// model the table gives none.
///
/// One constant, and deliberately not a transcription of that table. A per-model figure would be a
/// fact about someone else's data that goes stale on their release schedule with nothing in the
/// build to notice, and it would buy nothing: the server cannot tell a default Meka chose from a
/// value the user configured, so the only thing a wrong entry could produce is Meka quietly asking
/// for the wrong tier. Anyone who wants a different one sets `effort` on the profile.
pub(super) const DEFAULT_EFFORT: &str = "medium";

/// Whether a Claude model takes Claude Code's per-turn effort statements (the
/// `per-turn-control-2026-07-01` beta).
///
/// An **allowlist**, because that is the direction Claude Code 2.1.322's own gate points: it
/// reads the `per_turn_effort` capability off the model catalog bundled in its binary, and a
/// model the catalog does not list gets nothing. The catalog grants it to Opus 5.5, Sonnet 5.5,
/// Haiku 5.5, Fable 5.1, Fable 5.2 and Mythos 5.2 alone, so an unrecognized name, which in
/// practice means one newer than this list, resolves to `false` and the beta is omitted, which
/// every model tolerates.
pub(super) fn model_supports_per_turn_effort(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    let Some(version) = parse_model_version(&lower) else {
        return false;
    };
    if lower.contains("opus") || lower.contains("sonnet") || lower.contains("haiku") {
        version == (5, 5)
    } else if lower.contains("fable") {
        matches!(version, (5, 1) | (5, 2))
    } else if lower.contains("mythos") {
        version == (5, 2)
    } else {
        false
    }
}

/// Whether a Claude model accepts tool changes mid-conversation (the
/// `mid-conversation-tool-changes-2026-07-01` beta).
///
/// A **denylist**, mirroring Claude Code 2.1.322's gate model for model: it reads the
/// `mid_conv_tool_change` capability off its bundled catalog, which withholds it from the Claude
/// 3.x line, Opus 4.0 through 4.7, Sonnet 4.0 through 5.0 and Haiku 4.5, and sends the beta to a
/// model the catalog does not list. An unrecognized name here is one *newer* than the list, which
/// Claude Code sends the beta to, and withholding it would silently drop the tool changes Meka
/// relies on. Safe only because this gate has exactly one caller, `claude-subscription`, whose
/// endpoint is always Anthropic's, so an unrecognized name there is necessarily a real Claude. Do
/// not reach for it from a backend a `base_url` can point anywhere.
pub(super) fn model_supports_mid_conversation_tool_changes(model: &str) -> bool {
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
        !matches!(version, (4, 0) | (4, 5) | (4, 6) | (5, 0))
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
    pub(super) fn value(self) -> serde_json::Value {
        match self {
            Self::Ephemeral => serde_json::json!({"type": "ephemeral"}),
            Self::OneHour => serde_json::json!({"type": "ephemeral", "ttl": "1h"}),
        }
    }

    /// The same breakpoint on the block that opens the cached prefix, with Claude Code's
    /// `scope: "global"` (the `prompt-caching-scope-2026-01-05` beta) so sessions share it.
    pub(super) fn global_scope_value(self) -> serde_json::Value {
        match self {
            Self::Ephemeral => serde_json::json!({"type": "ephemeral", "scope": "global"}),
            Self::OneHour => {
                serde_json::json!({"type": "ephemeral", "ttl": "1h", "scope": "global"})
            }
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

/// The text block that stands in for an assistant message the encoder left empty, since the API
/// rejects a message with no content.
fn no_message_content() -> serde_json::Value {
    serde_json::json!({
        "type": "text",
        "text": "[No message content]"
    })
}

/// The conversation as the `messages` array of a Claude request, with the cache breakpoint, when
/// the profile sends one, on its last block.
pub(super) fn convert_messages_to_claude_content(
    messages: &[Message],
    breakpoint: Option<CacheBreakpoint>,
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
                .filter_map(|block| {
                    Some(match block {
                        // The turn's context block, a nudge, a note and what arrived for the model
                        // are text on this wire, in the order the message holds them.
                        // A blank one is left out: the API rejects a text block with no
                        // non-whitespace text in it, and a reply that streamed a newline before
                        // it was cut is saved as one.
                        ContentBlock::Text { .. }
                        | ContentBlock::TurnContext { .. }
                        | ContentBlock::Nudge { .. }
                        | ContentBlock::HarnessNote { .. }
                        | ContentBlock::Summary { .. }
                        | ContentBlock::Inbox { .. }
                        | ContentBlock::Scheduled { .. }
                        | ContentBlock::BackgroundReport { .. }
                        | ContentBlock::Invocation { .. }
                        | ContentBlock::Resource { .. }
                        | ContentBlock::Refusal { .. } => {
                            let text = block.model_text()?;
                            if crate::text::is_blank(&text) {
                                return None;
                            }
                            let mut text_block = serde_json::json!({
                                "type": "text",
                                "text": text,
                            });
                            // The citations a text block came with go back on it, where the
                            // spec defines them.
                            if let ContentBlock::Text {
                                provider_fields: Some(fields),
                                ..
                            } = block
                                && let Some(citations) = fields
                                    .get("citations")
                                    .filter(|citations| {
                                        citations.as_array().is_some_and(|list| !list.is_empty())
                                    })
                                && let Some(object) = text_block.as_object_mut()
                            {
                                object.insert("citations".to_string(), citations.clone());
                            }
                            text_block
                        }
                        ContentBlock::Image { source } => claude_image_block(source),
                        // Kept for the record; nothing on this wire carries it.
                        ContentBlock::Unrecognized { .. } => return None,
                        ContentBlock::ToolUse { id, name, input, .. } => {
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
                            ..
                        } => {
                            let content: Vec<serde_json::Value> = content
                                .iter()
                                .filter(|item| {
                                    !matches!(item, ToolResultContent::Text { text } if crate::text::is_blank(text))
                                })
                                .map(|item| match item.wire_part() {
                                    ResultPart::Text(text) => {
                                        serde_json::json!({"type": "text", "text": text})
                                    }
                                    ResultPart::Image(source) => claude_image_block(source),
                                    // The API expands the reference into the tool's definition
                                    // where it sits, under the deferred-tools beta the declaring
                                    // backends send.
                                    ResultPart::ToolReference(tool_name) => {
                                        serde_json::json!({
                                            "type": "tool_reference",
                                            "tool_name": tool_name,
                                        })
                                    }
                                })
                                .collect();
                            // `is_error` is the API's optional flag with false as its default, and
                            // Claude Code writes it only on a failure; a success carries no key.
                            let mut block = serde_json::Map::from_iter([
                                ("type".into(), serde_json::json!("tool_result")),
                                ("tool_use_id".into(), serde_json::json!(tool_use_id)),
                                ("content".into(), serde_json::json!(content)),
                            ]);
                            if *is_error {
                                block.insert("is_error".into(), serde_json::Value::Bool(true));
                            }
                            serde_json::Value::Object(block)
                        }
                        // A thinking block goes back with the signature it came with, the empty
                        // string OpenRouter gives a model that does not sign included. One with no
                        // signature at all, sealed by the Responses API or reasoning a Chat
                        // Completions endpoint returned in the clear, has no place here: the
                        // protocol makes `signature` required.
                        ContentBlock::Thinking { parts, opaque, .. } => {
                            let Some(OpaqueReasoning::Signed { signature }) = opaque else {
                                return None;
                            };
                            serde_json::json!({
                                "type": "thinking",
                                "thinking": crate::conversation::reasoning_as_streamed(parts),
                                "signature": signature,
                            })
                        }
                        // Replayed verbatim: the API needs the opaque `data` unchanged to continue
                        // the redacted reasoning chain.
                        ContentBlock::RedactedThinking { data, .. } => {
                            serde_json::json!({
                                "type": "redacted_thinking",
                                "data": data,
                            })
                        }
                    })
                })
                .collect();
            // An assistant turn that was unsigned thinking and nothing else is now empty, which
            // the API rejects; it goes out as the placeholder the trailing strip below uses.
            let content = if content.is_empty() {
                vec![no_message_content()]
            } else {
                content
            };

            serde_json::json!({
                "role": role,
                "content": content,
            })
        })
        .collect();

    // Strip trailing thinking blocks from an assistant message the request ends on (Claude API
    // requirement). Only there: one with a message after it, the reply a thinking-only nudge
    // answers, goes out whole as every later request sends it, since Claude rejects a block whose
    // earlier conversation changed and the nudge exists to keep that reasoning.
    if let Some(last_assistant) = claude_messages
        .last_mut()
        .filter(|message| message.get("role").and_then(|r| r.as_str()) == Some("assistant"))
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
            content.push(no_message_content());
        }
    }

    // After the strip, not before it: attached to the last block first, the breakpoint would go
    // out on a trailing thinking block and leave with it, so a conversation ending on one would
    // carry no breakpoint at all and re-bill its whole prefix at the write tier.
    if let Some(breakpoint) = breakpoint
        && let Some(last) = claude_messages.last_mut()
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

/// The deferred tools of a request, declared with `defer_loading` so the API withholds their
/// schemas until a `tool_reference` names them: listed beside the active tools, never in the
/// prompt, which is what keeps the cached prefix whole across a load.
pub(super) fn convert_deferred_tools_to_claude_tools(
    tools: &[ToolDefinition],
) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
                "defer_loading": true,
            })
        })
        .collect()
}

/// Append `deferred` to the body's `tools`, creating the array for a request whose active tools
/// are none, and keep a `tool_reference` block in the history only where this request can send it.
/// The API rejects a reference to a tool its `tools` array lacks (a server withdrew or renamed it,
/// or the session moved to a profile that appends), and a result mixing references with anything
/// else. So a result of references keeps those this request declares and leaves the rest out (a
/// tool now active is in the array whole); one with none declared, or with other content beside
/// them, sends each as the text of its name, in place.
pub(super) fn declare_deferred_tools(body: &mut serde_json::Value, deferred: &[ToolDefinition]) {
    let declared: HashSet<&str> = deferred.iter().map(|tool| tool.name.as_str()).collect();
    let is_reference = |item: &serde_json::Value| item["type"] == "tool_reference";
    let is_declared_reference = |item: &serde_json::Value| {
        is_reference(item)
            && item["tool_name"]
                .as_str()
                .is_some_and(|name| declared.contains(name))
    };
    let results = body
        .get_mut("messages")
        .and_then(serde_json::Value::as_array_mut)
        .into_iter()
        .flatten()
        .filter_map(|message| message.get_mut("content")?.as_array_mut())
        .flatten()
        .filter(|block| block["type"] == "tool_result")
        .filter_map(|block| block.get_mut("content")?.as_array_mut());
    for items in results {
        if items.iter().all(is_reference) && items.iter().any(is_declared_reference) {
            items.retain(is_declared_reference);
            continue;
        }
        for item in items.iter_mut().filter(|item| is_reference(item)) {
            let name = serde_json::Value::String(
                item["tool_name"].as_str().unwrap_or_default().to_string(),
            );
            if let Some(fields) = item.as_object_mut() {
                fields.remove("tool_name");
                fields.insert("type".to_string(), "text".into());
                fields.insert("text".to_string(), name);
            }
        }
    }
    if deferred.is_empty() {
        return;
    }
    let declared = convert_deferred_tools_to_claude_tools(deferred);
    match body.get_mut("tools").and_then(|tools| tools.as_array_mut()) {
        Some(tools) => tools.extend(declared),
        None => {
            body["tools"] = serde_json::Value::Array(declared);
        }
    }
}

/// A Claude `stop_reason` as Meka's own.
pub(super) fn parse_claude_stop_reason(reason: &str) -> StopReason {
    match reason {
        // A stop sequence ends the reply where the request asked; `stop_sequence` in the details
        // says which.
        "end_turn" | "stop_sequence" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        "model_context_window_exceeded" => StopReason::ContextWindow,
        "refusal" => StopReason::Refusal,
        // Logged so a recurrence is diagnosable; the word itself is recorded with the reply.
        other => {
            tracing::warn!(
                "Claude returned unrecognized stop_reason {other:?}; mapping to Unknown"
            );
            StopReason::Unknown
        }
    }
}

/// How a Claude reply ended, from the `message_delta` that says so: its `stop_reason` classed, the
/// word itself, and beside it `stop_details` (a refusal's category and explanation) and
/// `stop_sequence` when the delta carries them.
fn claude_stop(delta: &serde_json::Value, word: &str) -> Stop {
    let details: serde_json::Map<String, serde_json::Value> = ["stop_details", "stop_sequence"]
        .into_iter()
        .filter_map(|field| {
            delta
                .get(field)
                .filter(|value| !value.is_null())
                .map(|value| (field.to_string(), value.clone()))
        })
        .collect();
    Stop {
        reason: parse_claude_stop_reason(word),
        word: Some(word.to_string()),
        details: (!details.is_empty()).then_some(details),
    }
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
    /// `max_request_bytes`, else [`crate::config::ANTHROPIC_MAX_REQUEST_BYTES`].
    fn max_request_bytes(&self) -> usize;
    /// The request body for one completion, before serialization.
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per request field, as the request has"
    )]
    fn request_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        deferred_tools: &[ToolDefinition],
        stream: bool,
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
        declares_deferred: bool,
        stream: bool,
        attribution: &crate::provider::Attribution,
    ) -> Result<reqwest::RequestBuilder>;
    /// Whether a thinking block's text is the summary the request asked for (Claude's
    /// `thinking.display = "summarized"`) rather than the reasoning itself, which the fold then
    /// records as a summary part. `false` where the request states no display, since what the
    /// endpoint returns is then its own fact.
    fn summarizes_thinking(&self) -> bool {
        false
    }
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

/// A non-streaming Claude call is the streaming one folded in place, so a wire has one request
/// shape. A gateway may keep separate caches for streamed and unstreamed requests (OpenRouter's
/// Vertex route does, measured 2026-10-26 against captured bodies), and a turn streams, so a
/// checkpoint or a summary sent unstreamed would read none of the turn's prefix. The captured
/// subscription wire streams every request too.
pub(super) async fn complete<B: ClaudeBackend>(
    backend: &B,
    request: CompletionRequest<'_>,
    cancellation: CancellationToken,
) -> Result<crate::provider::Completion> {
    let (event_sender, event_receiver) = mpsc::channel::<StreamEvent>(1024);
    let (streamed, folded) = tokio::join!(
        stream(backend, request, event_sender, cancellation),
        crate::provider::aggregate_stream(event_receiver),
    );
    streamed?;
    Ok(folded)
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
        deferred_tools,
        attribution,
        ..
    } = request;
    let (body_json, redaction_notice) =
        build_body_within_budget(messages, backend.max_request_bytes(), |messages| {
            serialize_body(&backend.request_body(
                system_prompt,
                messages,
                tools,
                deferred_tools,
                true,
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
                    !deferred_tools.is_empty(),
                    true,
                    &attribution,
                )
                .await?
                .body(body_json.clone()))
        },
        &cancellation,
    )
    .await?;
    backend.remember_request_id(&attribution, response.headers());
    // A refusal is no response of the conversation's: Claude Code holds a request id, and so
    // records a response, only once the request is answered with a success.
    if response.status().is_success() {
        attribution.note_response_arrived();
    }
    drive_claude_sse_stream(
        response,
        event_sender,
        cancellation,
        Some(attribution),
        backend.summarizes_thinking(),
    )
    .await
}

/// Read a Claude SSE response to its end, forwarding each event on the channel.
pub(super) async fn drive_claude_sse_stream(
    response: reqwest::Response,
    event_sender: mpsc::Sender<StreamEvent>,
    cancellation: CancellationToken,
    attribution: Option<crate::provider::Attribution>,
    summarized_thinking: bool,
) -> Result<()> {
    let mut protocol = ClaudeStream {
        attribution,
        summarized_thinking,
        ..ClaudeStream::default()
    };
    let ended = crate::provider::sse::drive(
        response,
        "Claude",
        &event_sender,
        &cancellation,
        &mut protocol,
    )
    .await;
    let ended = match ended {
        Ok(End::Finished | End::ReceiverGone) => Ok(()),
        // The byte stream ending is not the same as the message ending.
        //
        // An intermediary (a gateway named by `base_url`, a CDN edge, a load balancer closing an
        // idle connection) can terminate a chunked response cleanly mid-message. Treating that as
        // success hands the agent a half-written answer with `stop_reason` left at its `EndTurn`
        // default: no error, so no retry, and nothing to distinguish a truncated reply from a
        // complete one. Worse mid-tool-call, whose arguments never finished. Reporting it as a
        // `StreamError` routes it to the same retry path a dropped connection already takes.
        //
        // A stop reason already seen makes the message complete without `message_stop`: a gateway
        // that forwards the deltas and closes without the final frame delivers a whole answer.
        Ok(End::Ended) if protocol.saw_terminal_event => Ok(()),
        Ok(End::Ended) => Err(crate::provider::sse::stream_error(
            &event_sender,
            "stream ended before a stop reason".to_string(),
        )
        .await),
        Err(error) => Err(error),
    };
    // A block the stream left open is handed over after the failure is said, so it goes to the
    // attempt's partial reply and leaves the retry the failure allows.
    if !protocol.close_open(&event_sender).await {
        tracing::trace!("stream event receiver dropped");
    }
    ended
}

/// The Claude driver's state between frames.
#[derive(Default)]
struct ClaudeStream {
    /// Who the request was for, whose slots `message_start`'s id is recorded in: the next
    /// request's `diagnostics.previous_message_id`, and on the main thread its last response.
    /// `None` on a stream read for no conversation.
    attribution: Option<crate::provider::Attribution>,
    /// Whether the message reached its end rather than the byte stream simply stopping.
    saw_terminal_event: bool,
    /// Whether a thinking block's text is a summary; see [`ClaudeBackend::summarizes_thinking`].
    summarized_thinking: bool,
    /// Whether a `message_delta` has stated the stop, which the reply's record then holds.
    stop_stated: bool,
    /// The message as `message_start` stated it and each `message_delta` updated it, folded as
    /// the reference client folds them, without its content.
    response: serde_json::Map<String, serde_json::Value>,
    /// The blocks open now, by `index`: a block's index is its place in the message's content,
    /// and a stream may open one before an earlier one stops.
    open: std::collections::BTreeMap<u64, OpenBlock>,
    /// The `index` of the block opened last, after which a frame that names no block is kept.
    latest: u64,
    /// The place the events sent so far are of; see [`StreamEvent::Place`].
    place: u64,
}

/// A content block between its start and its stop.
struct OpenBlock {
    kind: BlockKind,
    /// What arrived inside the block that the reference client does not fold (a delta of a type
    /// the block does not read, or an event of a type it does not know), kept whole after the
    /// block so the block is not split.
    unread: Vec<serde_json::Value>,
}

/// What an open block is, with what its stop needs.
enum BlockKind {
    /// A block of a type Meka has no block for, or one without what the spec requires of its
    /// type: the frames that start it and carry its deltas, whole, as they came, each naming the
    /// block by its `index`.
    Unread(Vec<serde_json::Value>),
    /// A text block, with its fields beyond its type and text.
    Text(Option<crate::conversation::ProviderFields>),
    Thinking {
        signature: Option<String>,
        /// The block's fields beyond its type, thinking and signature.
        fields: Option<crate::conversation::ProviderFields>,
    },
    /// A `redacted_thinking` block, with the opaque `data` its start carried whole and its fields
    /// beyond its type and data.
    Redacted {
        data: String,
        fields: Option<crate::conversation::ProviderFields>,
    },
    Call {
        id: String,
        name: String,
        /// The arguments as they streamed.
        arguments: String,
        /// The `input` the block opened with when it held anything, which stands as its arguments
        /// when no `input_json_delta` follows, as the reference client keeps it.
        opened_with: Option<serde_json::Value>,
        /// The block's fields beyond its type, id, name and input.
        fields: Option<crate::conversation::ProviderFields>,
    },
}

#[async_trait::async_trait]
impl crate::provider::sse::Protocol for ClaudeStream {
    async fn frame(
        &mut self,
        event: eventsource_stream::Event,
        event_sender: &mpsc::Sender<StreamEvent>,
    ) -> Result<Step> {
        let data = crate::provider::sse::frame_json("Claude", &event.data)?;
        let name = claude_frame_name(&data, &event.event).to_string();
        if !name.starts_with("content_block_") {
            return self.message_frame(&name, data, event_sender).await;
        }
        // Every block event names its block's `index`; one that does not is a shape the protocol
        // does not define, kept as it came.
        let delivered = match data.get("index").and_then(serde_json::Value::as_u64) {
            Some(index) => match name.as_str() {
                "content_block_start" => self.start(index, data, event_sender).await,
                "content_block_delta" => self.delta(index, data, event_sender).await,
                // A stop for a block that is not open closes nothing and is kept where it names,
                // as a delta for one is.
                "content_block_stop" if !self.open.contains_key(&index) => {
                    self.send(index, StreamEvent::Unrecognized(data), event_sender)
                        .await
                }
                "content_block_stop" => self.close(index, true, event_sender).await,
                _ => self.keep_unread(data, event_sender).await,
            },
            None => self.keep_unread(data, event_sender).await,
        };
        Ok(if delivered {
            Step::Continue
        } else {
            Step::ReceiverGone
        })
    }
}

impl ClaudeStream {
    /// The message's fields nothing else holds: all but its role, content, model and how it
    /// stopped, which the reply and its record hold. The raw `usage` stays, and so does a stop
    /// the message states before any `message_delta` does, which the record never receives, and
    /// content other than the empty list the spec opens the message with, which no block holds.
    fn response_fields(&self) -> Option<crate::conversation::ProviderFields> {
        let mut held = vec!["role", "model"];
        if self
            .response
            .get("content")
            .is_none_or(|content| content.as_array().is_some_and(Vec::is_empty))
        {
            held.push("content");
        }
        held.extend(
            ["stop_reason", "stop_sequence", "stop_details"]
                .into_iter()
                .filter(|key| {
                    self.stop_stated || self.response.get(*key).is_none_or(|value| value.is_null())
                }),
        );
        crate::provider::fields_beyond(&self.response, &held)
    }

    /// Tell the message's usage as it stands, folded from every statement of it so far, which the
    /// reply's usage takes whole; `false` once the receiver is gone.
    async fn tell_usage(&self, event_sender: &mpsc::Sender<StreamEvent>) -> bool {
        match self.response.get("usage").filter(|usage| usage.is_object()) {
            Some(usage) => event_sender
                .send(StreamEvent::Usage(parse_usage_object(usage)))
                .await
                .is_ok(),
            None => true,
        }
    }

    /// Tell the message's fields as they stand; `false` once the receiver is gone.
    async fn tell_response(&self, event_sender: &mpsc::Sender<StreamEvent>) -> bool {
        match self.response_fields() {
            Some(fields) => event_sender
                .send(StreamEvent::ResponseFields(fields))
                .await
                .is_ok(),
            None => true,
        }
    }

    /// Send `event` of the block at `index`, saying first which block it is of when the event sent
    /// before it was another's. Whether the receiver is still there.
    async fn send(
        &mut self,
        index: u64,
        event: StreamEvent,
        event_sender: &mpsc::Sender<StreamEvent>,
    ) -> bool {
        if index != self.place {
            if event_sender.send(StreamEvent::Place(index)).await.is_err() {
                return false;
            }
            self.place = index;
        }
        event_sender.send(event).await.is_ok()
    }

    /// Keep `frame`, which names no open block, as it came: inside the block opened last while
    /// that block is open, so the block is not split, else after it.
    async fn keep_unread(
        &mut self,
        frame: serde_json::Value,
        event_sender: &mpsc::Sender<StreamEvent>,
    ) -> bool {
        match self.open.get_mut(&self.latest) {
            Some(block) => {
                block.unread.push(frame);
                true
            }
            None => {
                self.send(self.latest, StreamEvent::Unrecognized(frame), event_sender)
                    .await
            }
        }
    }

    /// Open the block `frame` starts at `index`, handing over what its start already carries. A
    /// start at an index already open closes the block there first, as one that never finished.
    async fn start(
        &mut self,
        index: u64,
        frame: serde_json::Value,
        event_sender: &mpsc::Sender<StreamEvent>,
    ) -> bool {
        if self.open.contains_key(&index) && !self.close(index, false, event_sender).await {
            return false;
        }
        self.latest = index;
        let content_block = frame
            .get("content_block")
            .and_then(serde_json::Value::as_object);
        let block_type = content_block
            .and_then(|block| block.get("type"))
            .and_then(serde_json::Value::as_str);
        let field = |name: &str| {
            content_block
                .and_then(|block| block.get(name))
                .and_then(serde_json::Value::as_str)
        };
        // What a text or reasoning block opens with is its first, as the reference client folds
        // it.
        let opening = |name: &str| {
            field(name)
                .filter(|text| !text.is_empty())
                .map(str::to_string)
        };
        let fields_beyond = |read: &[&str]| {
            content_block.and_then(|block| crate::provider::fields_beyond(block, read))
        };
        let mut events = Vec::new();
        let kind = match (block_type, field("data"), field("id").zip(field("name"))) {
            (Some("text"), ..) => {
                events.extend(opening("text").map(StreamEvent::TextDelta));
                BlockKind::Text(fields_beyond(&["type", "text"]))
            }
            (Some("thinking"), ..) => {
                // Announce the block itself, before any estimate: this is the earliest point the
                // pause becomes explainable, and on a redacted block it is otherwise the only thing
                // that happens for seconds at a time.
                events.push(StreamEvent::ThinkingProgress {
                    estimated_tokens: None,
                });
                if self.summarized_thinking {
                    events.push(StreamEvent::ThinkingPart {
                        kind: crate::conversation::ReasoningKind::Summary,
                        number: None,
                    });
                }
                events.extend(opening("thinking").map(StreamEvent::ThinkingDelta));
                // Both schemas make `signature` required on a thinking block sent back, and
                // require it verbatim, but it does not always arrive as a `signature_delta`.
                // Anthropic opens the block with an empty one and fills it by delta; OpenRouter
                // sends no delta at all for a non-Anthropic model, leaving that empty string as the
                // value. Losing it there costs every later request in the session, rejected for the
                // missing field.
                BlockKind::Thinking {
                    signature: field("signature").map(str::to_string),
                    fields: fields_beyond(&["type", "thinking", "signature"]),
                }
            }
            // The opaque `data` arrives whole in the start event and is handed over when the block
            // stops, after the progress the server streams inside it; the block is announced as a
            // thinking block is.
            (Some("redacted_thinking"), Some(data), _) => {
                events.push(StreamEvent::ThinkingProgress {
                    estimated_tokens: None,
                });
                BlockKind::Redacted {
                    data: data.to_string(),
                    fields: fields_beyond(&["type", "data"]),
                }
            }
            (Some("tool_use"), _, Some((id, name))) => {
                let (id, name) = (id.to_string(), name.to_string());
                events.push(StreamEvent::ToolUseStart {
                    id: id.clone(),
                    name: name.clone(),
                });
                BlockKind::Call {
                    id,
                    name,
                    arguments: String::new(),
                    opened_with: content_block
                        .and_then(|block| block.get("input"))
                        .filter(|input| {
                            !input.is_null()
                                && input.as_object().is_none_or(|input| !input.is_empty())
                        })
                        .cloned(),
                    fields: fields_beyond(&["type", "id", "name", "input"]),
                }
            }
            // A block of a type Meka has no block for, or without what the spec requires of its
            // type (a call without its id and name, redacted reasoning without its data), kept
            // whole with its deltas and never run.
            _ => BlockKind::Unread(vec![frame.clone()]),
        };
        self.open.insert(index, OpenBlock {
            kind,
            unread: Vec::new(),
        });
        for event in events {
            if !self.send(index, event, event_sender).await {
                return false;
            }
        }
        true
    }

    /// Fold a delta into the block at `index` as the reference client folds it. A delta of a type
    /// its block does not read is kept unread after the block, and one for a block that is not
    /// open is kept where it names.
    async fn delta(
        &mut self,
        index: u64,
        frame: serde_json::Value,
        event_sender: &mpsc::Sender<StreamEvent>,
    ) -> bool {
        let Some(block) = self.open.get_mut(&index) else {
            return self
                .send(index, StreamEvent::Unrecognized(frame), event_sender)
                .await;
        };
        let delta = frame.get("delta");
        let field = |name: &str| {
            delta
                .and_then(|delta| delta.get(name))
                .and_then(serde_json::Value::as_str)
        };
        // `estimated_tokens` is the server's running count of thinking spent so far (the
        // `thinking-token-count` beta). The final delta of a block carries `null`; skipping it
        // leaves the last real figure on screen rather than blanking the display just before the
        // block ends.
        let progress = delta
            .and_then(|delta| delta.get("estimated_tokens"))
            .and_then(serde_json::Value::as_u64)
            .map(|estimated| StreamEvent::ThinkingProgress {
                estimated_tokens: Some(estimated),
            });
        let thinking = field("thinking").filter(|thinking| !thinking.is_empty());
        let mut events = Vec::new();
        match (&mut block.kind, field("type").unwrap_or("")) {
            (BlockKind::Unread(frames), _) => frames.push(frame.clone()),
            (BlockKind::Thinking { .. }, "thinking_delta") => {
                events.extend(progress);
                events.extend(
                    thinking.map(|thinking| StreamEvent::ThinkingDelta(thinking.to_string())),
                );
            }
            // Under the `redact-thinking` beta the server streams the withheld block's progress as
            // `thinking_delta` estimates with no words, which the reference client folds into
            // nothing; words there have no block to go on and are kept unread.
            (BlockKind::Redacted { .. }, "thinking_delta") if thinking.is_none() => {
                events.extend(progress);
            }
            // The delta states the signature whole, and replaces what the start held, as the
            // reference client folds it.
            (BlockKind::Thinking { signature, .. }, "signature_delta") => {
                if let Some(stated) = field("signature") {
                    *signature = Some(stated.to_string());
                }
            }
            (BlockKind::Text(_), "text_delta") => {
                if let Some(text) = field("text").filter(|text| !text.is_empty()) {
                    events.push(StreamEvent::TextDelta(text.to_string()));
                }
            }
            // A citation goes on its text block's `citations`, as the reference client folds it.
            (BlockKind::Text(fields), "citations_delta") => {
                if let Some(citation) = delta.and_then(|delta| delta.get("citation")) {
                    let citations = fields
                        .get_or_insert_with(Default::default)
                        .entry("citations".to_string())
                        .or_insert(serde_json::Value::Null);
                    match citations.as_array_mut() {
                        Some(citations) => citations.push(citation.clone()),
                        None => *citations = serde_json::json!([citation]),
                    }
                }
            }
            (BlockKind::Call { arguments, .. }, "input_json_delta") => {
                if let Some(partial_json) = field("partial_json") {
                    arguments.push_str(partial_json);
                    events.push(StreamEvent::ToolArgumentsDelta(partial_json.to_string()));
                }
            }
            _ => block.unread.push(frame.clone()),
        }
        for event in events {
            if !self.send(index, event, event_sender).await {
                return false;
            }
        }
        true
    }

    /// Close the block at `index` and hand it over, then what arrived unread inside it. A block
    /// that `finished` is handed over as it stopped; one the stream ended inside, or a second
    /// start at its index replaced, as one that never finished: its reasoning unsigned, since a
    /// signature that arrived before the block ended may not cover all of it, and a call refused
    /// with the arguments that had streamed. A block that is not open closes nothing. Whether the
    /// receiver is still there.
    async fn close(
        &mut self,
        index: u64,
        finished: bool,
        event_sender: &mpsc::Sender<StreamEvent>,
    ) -> bool {
        let Some(block) = self.open.remove(&index) else {
            return true;
        };
        let events = match block.kind {
            BlockKind::Unread(frames) => {
                frames.into_iter().map(StreamEvent::Unrecognized).collect()
            }
            BlockKind::Text(provider_fields) => vec![StreamEvent::TextEnd { provider_fields }],
            BlockKind::Thinking { signature, fields } => vec![StreamEvent::ThinkingComplete {
                opaque: signature
                    .filter(|_| finished)
                    .map(|signature| OpaqueReasoning::Signed { signature }),
                provider_fields: fields,
            }],
            BlockKind::Redacted { data, fields } => vec![StreamEvent::RedactedThinking {
                data,
                provider_fields: fields,
            }],
            BlockKind::Call {
                id,
                name,
                arguments,
                opened_with,
                fields,
            } => {
                let arguments = match opened_with {
                    Some(input) if arguments.is_empty() => input.to_string(),
                    _ => arguments,
                };
                // An empty body is a legitimate zero-argument call and becomes `{}`; arguments that
                // arrived but do not parse are refused rather than replaced.
                vec![if finished {
                    crate::provider::tool_use_event(id, name, &arguments, fields)
                } else {
                    StreamEvent::ToolCallRefused {
                        id,
                        name,
                        unparsed: crate::provider::unfinished_call(arguments),
                        provider_fields: fields,
                    }
                }]
            }
        };
        let unread = block.unread.into_iter().map(StreamEvent::Unrecognized);
        for event in events.into_iter().chain(unread) {
            if !self.send(index, event, event_sender).await {
                return false;
            }
        }
        true
    }

    /// Close every block still open as one that never finished, in index order: the stream is
    /// over, whether the message ended or the stream broke off and what arrived of it is the reply
    /// so far. Whether the receiver is still there.
    async fn close_open(&mut self, event_sender: &mpsc::Sender<StreamEvent>) -> bool {
        let indices: Vec<u64> = self.open.keys().copied().collect();
        for index in indices {
            if !self.close(index, false, event_sender).await {
                return false;
            }
        }
        true
    }

    /// Fold a frame of the message's own, outside its blocks.
    async fn message_frame(
        &mut self,
        name: &str,
        data: serde_json::Value,
        event_sender: &mpsc::Sender<StreamEvent>,
    ) -> Result<Step> {
        match name {
            "message_delta" => {
                // Folded into the message alone, as the reference client folds it, and no block is
                // closed here: a gateway may restate the running usage between a block's deltas,
                // and a block's own stop, else the stream's end, closes it. The delta's fields and
                // the event's own set the message's, and each usage field stated overwrites, since
                // the counts are running totals. The reference sets only the fields it types; the
                // rest (OpenRouter's `cost`, a later `service_tier`) are set the same way. A null
                // states nothing, so a stop once stated stands against a trailing delta that
                // restates the fields as null.
                if data
                    .get("delta")
                    .and_then(|delta| delta.get("stop_reason"))
                    .is_some_and(serde_json::Value::is_string)
                {
                    self.stop_stated = true;
                }
                // A `usage` that is not an object is a shape the spec does not define: kept as it
                // came, it leaves the counts stated so far standing.
                let unread_usage = data
                    .get("usage")
                    .is_some_and(|usage| !usage.is_object() && !usage.is_null());
                for (key, value) in data.as_object().into_iter().flatten() {
                    match (key.as_str(), value) {
                        ("type", _) => {}
                        ("usage", _) if unread_usage => {}
                        ("delta" | "usage", serde_json::Value::Object(fields)) => {
                            let target = if key == "delta" {
                                &mut self.response
                            } else {
                                let usage = self
                                    .response
                                    .entry("usage".to_string())
                                    .or_insert(serde_json::Value::Null);
                                if !usage.is_object() {
                                    *usage = serde_json::Value::Object(serde_json::Map::new());
                                }
                                match usage.as_object_mut() {
                                    Some(usage) => usage,
                                    None => continue,
                                }
                            };
                            for (field, value) in fields {
                                crate::provider::state_latest(target, field, value);
                            }
                        }
                        _ => crate::provider::state_latest(&mut self.response, key, value),
                    }
                }
                if !self.tell_response(event_sender).await
                    || (data.get("usage").is_some() && !self.tell_usage(event_sender).await)
                    || (unread_usage && !self.keep_unread(data.clone(), event_sender).await)
                {
                    return Ok(Step::ReceiverGone);
                }
                let Some(delta) = data.get("delta") else {
                    return Ok(Step::Continue);
                };
                if let Some(stop_reason_str) =
                    delta.get("stop_reason").and_then(|reason| reason.as_str())
                {
                    // The stop reason is what ends a message; `message_stop` is
                    // the framing around it. Requiring the frame would make Meka
                    // strictly less tolerant than the wire format needs: a gateway
                    // named by `base_url` that forwards the deltas and closes
                    // without the final event delivers a complete answer, and
                    // every turn through it would fail. What the check is for (a
                    // cut mid-`content_block_delta`) never gets this far.
                    self.saw_terminal_event = true;
                    let stop = claude_stop(delta, stop_reason_str);
                    if event_sender
                        .send(StreamEvent::MessageEnd { stop })
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
                if let Some(message) = data.get("message").and_then(|message| message.as_object()) {
                    self.response = message.clone();
                    if !self.tell_response(event_sender).await {
                        return Ok(Step::ReceiverGone);
                    }
                }
                if let (Some(attribution), Some(message_id)) = (
                    &self.attribution,
                    data.get("message")
                        .and_then(|message| message.get("id"))
                        .and_then(|id| id.as_str()),
                ) {
                    if let Some(slot) = &attribution.previous_message {
                        crate::provider::record_message_id(slot, message_id);
                    }
                    attribution.note_message_opened(message_id);
                }
                if let Some(model) = data
                    .get("message")
                    .and_then(|message| message.get("model"))
                    .and_then(|model| model.as_str())
                    .filter(|model| !model.is_empty())
                    && event_sender
                        .send(StreamEvent::Model(model.to_string()))
                        .await
                        .is_err()
                {
                    return Ok(Step::ReceiverGone);
                }
                if self.response.contains_key("usage") && !self.tell_usage(event_sender).await {
                    return Ok(Step::ReceiverGone);
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
            // An event of a type Meka does not read, kept whole where it came, or after the block
            // it came inside.
            _ => {
                if !self.keep_unread(data, event_sender).await {
                    return Ok(Step::ReceiverGone);
                }
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
    std::sync::Mutex<std::collections::HashMap<DownscaleCacheKey, (&'static str, String)>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Entries the downscale cache holds before it is cleared. Each is one downscaled image, its media
/// type and its base64, so this bounds residency by roughly this many downscaled images.
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

fn downscale_cache_get(source_base64: &str) -> Option<(&'static str, String)> {
    let cache = crate::sync::lock(&DOWNSCALE_CACHE);
    cache.get(&downscale_cache_key(source_base64)).cloned()
}

fn downscale_cache_put(source_base64: &str, media_type: &'static str, downscaled_base64: &str) {
    let mut cache = crate::sync::lock(&DOWNSCALE_CACHE);
    if cache.len() >= DOWNSCALE_CACHE_ENTRIES {
        cache.clear();
    }
    cache.insert(
        downscale_cache_key(source_base64),
        (media_type, downscaled_base64.to_string()),
    );
}

/// Walk `messages` and downscale any image whose pixel dimensions exceed
/// [`MAX_IMAGE_DIMENSION_PX`] on either axis. The body bytes (base64) for those images are replaced
/// with a re-encoding that fits within the cap ([`crate::image::downscale_to_dim_cap`]); smaller
/// images are left alone. Returns
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

    // Re-encode `source` within the cap in place; no-op if it can't be decoded or already fits.
    // Served from the cache when this exact payload has been downscaled before.
    fn downscale_in_place(source: &mut crate::image::ImageSource) {
        // A reference has no bytes to downscale; hydration resolves every one before a request.
        let crate::image::ImageSource::Base64 { media_type, data } = source else {
            return;
        };
        let Some(format) = parse_format(media_type) else {
            return;
        };
        if let Some((cached_type, cached)) = downscale_cache_get(data) {
            *media_type = cached_type.to_string();
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
            Ok((downscaled_type, downscaled)) => {
                let encoded = base64::engine::general_purpose::STANDARD.encode(&downscaled);
                downscale_cache_put(&original, downscaled_type, &encoded);
                *media_type = downscaled_type.to_string();
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
/// backend applies, [`crate::provider::budget::fit_body_to_budget`], with the API's image count.
/// Both Claude providers run this; the caller supplies the body builder via `build` so each
/// provider's thinking / metadata wiring stays in its own file. The downscale is the
/// Anthropic-specific half, for the 2000 px multi-image cap; the redaction and the refusal are the
/// shared half.
pub(super) fn build_body_within_budget<F>(
    messages: &[Message],
    max_request_bytes: usize,
    build: F,
) -> Result<(String, Option<crate::frontend::Notice>)>
where
    F: FnMut(&[Message]) -> Result<String>,
{
    let prepared = downscale_oversized_images(messages);
    crate::provider::budget::fit_body_to_budget(
        prepared.as_ref(),
        max_request_bytes,
        Some(crate::provider::budget::ANTHROPIC_MAX_REQUEST_IMAGES),
        build,
    )
}

#[cfg(test)]
mod tests {
    /// `message_start` is where the streaming path learns the message id the next request names
    /// as `diagnostics.previous_message_id`.
    #[tokio::test]
    async fn message_start_records_the_message_id_on_the_attribution() {
        let slot = crate::provider::PreviousMessageSlot::default();
        let mut protocol = super::ClaudeStream {
            attribution: Some(crate::provider::Attribution {
                previous_message: Some(std::sync::Arc::clone(&slot)),
                ..Default::default()
            }),
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

    /// On the main thread, a response whose message opens is the last response, with the id and
    /// the position of the same request, and answers what a compaction armed: even when the
    /// stream then fails, as Claude Code forgets it once the message has started. Off the main
    /// thread nothing is recorded.
    #[tokio::test]
    async fn an_opened_message_is_the_main_thread_s_last_response_and_answers_the_armed_one() {
        let position = Some(crate::stats::TurnPosition {
            prompt_index: 3,
            turn_index: 4,
        });
        let armed = crate::provider::MainResponse {
            message_id: Some("msg_01armed".to_string()),
            turn_position: None,
        };
        let main_thread = crate::provider::MainThreadResponses::default();
        *crate::sync::lock(&main_thread.before_compaction) = Some(armed.clone());
        let attribution = crate::provider::Attribution {
            main_thread: Some(main_thread.clone()),
            turn_position: position,
            ..Default::default()
        };
        attribution.note_response_arrived();
        assert_eq!(
            *crate::sync::lock(&main_thread.last),
            Some(crate::provider::MainResponse {
                message_id: None,
                turn_position: position,
            }),
            "a response is the last once it arrives"
        );
        assert_eq!(
            attribution.response_before_compaction(),
            Some(armed.clone())
        );

        let response: reqwest::Response = http::Response::builder()
            .status(200)
            .header("content-type", "text/event-stream")
            .body(
                "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01opened\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n"
                    .to_string(),
            )
            .expect("build response")
            .into();
        let (sender, _receiver) = mpsc::channel(64);
        let outcome = drive_claude_sse_stream(
            response,
            sender,
            CancellationToken::new(),
            Some(attribution.clone()),
            false,
        )
        .await;
        assert!(outcome.is_err(), "the stream ended before a stop reason");
        assert_eq!(
            *crate::sync::lock(&main_thread.last),
            Some(crate::provider::MainResponse {
                message_id: Some("msg_01opened".to_string()),
                turn_position: position,
            })
        );
        assert_eq!(attribution.response_before_compaction(), None);

        let elsewhere = crate::provider::Attribution {
            turn_position: position,
            ..Default::default()
        };
        elsewhere.note_response_arrived();
        elsewhere.note_message_opened("msg_01elsewhere");
        assert_eq!(elsewhere.response_before_compaction(), None);
        assert_eq!(
            crate::sync::lock(&main_thread.last)
                .as_ref()
                .and_then(|response| response.message_id.as_deref()),
            Some("msg_01opened"),
            "a request off the main thread records nothing there"
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
        let (sender, mut receiver) = mpsc::channel(1024);
        let outcome =
            drive_claude_sse_stream(response, sender, CancellationToken::new(), None, false).await;
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
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
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

    /// An inbox item read at a round boundary is text beside the round's tool results. The API
    /// wants the results first in that message, and this encoder keeps blocks in the order the
    /// loop appended them, which is what puts the text after them.
    #[test]
    fn text_beside_tool_results_stays_after_them_in_one_message() {
        let message = Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "contents".to_string(),
                    }],
                    is_error: false,
                    display: None,
                },
                ContentBlock::Text {
                    text: "[Message from test, arrived now]\nalso, what is 17*3?".to_string(),
                    provider_fields: None,
                },
            ],
            provider_fields: None,
        };
        let converted =
            convert_messages_to_claude_content(&[message], Some(CacheBreakpoint::Ephemeral));
        let content = converted[0]["content"].as_array().expect("content array");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[1]["type"], "text");
        assert_eq!(
            content[1]["text"],
            "[Message from test, arrived now]\nalso, what is 17*3?"
        );
        assert!(
            content[1].get("cache_control").is_some(),
            "the moving breakpoint lands on the new last block"
        );
    }

    /// An invocation and an attached resource are text on this wire, each where the message holds
    /// it, so the model reads the skill's body and the file among the words.
    #[test]
    fn an_invocation_and_a_resource_are_read_in_place() {
        let converted = convert_messages_to_claude_content(
            &[Message::invoked_with_a_resource_for_test()],
            None,
        );
        let texts: Vec<&str> = converted[0]["content"]
            .as_array()
            .expect("content array")
            .iter()
            .filter_map(|block| block["text"].as_str())
            .collect();
        assert_eq!(texts, [
            "<context/>",
            "the parser\n\nthe skill's body",
            "compare ",
            "<resource uri=\"file:///a.rs\">fn a() {}</resource>",
            " with the old one",
        ]);
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
                    provider_fields: None,
                },
                ContentBlock::Thinking {
                    parts: vec![crate::conversation::ReasoningPart::reasoning("trailing")],
                    opaque: None,
                    provider_fields: None,
                },
            ],
            provider_fields: None,
        };
        let converted =
            convert_messages_to_claude_content(&[message], Some(CacheBreakpoint::Ephemeral));
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

    /// A reply of nothing but thinking, answered by a nudge, goes out whole: the request after the
    /// model's next reply sends it so, and Claude rejects a block whose earlier conversation
    /// changed between the two. Only a request that ends on an assistant message loses its
    /// trailing thinking.
    #[test]
    fn a_thinking_only_reply_followed_by_a_nudge_keeps_its_thinking() {
        let signed = ContentBlock::Thinking {
            parts: vec![crate::conversation::ReasoningPart::reasoning(
                "the reasoning",
            )],
            opaque: Some(OpaqueReasoning::Signed {
                signature: "sig".to_string(),
            }),
            provider_fields: None,
        };
        let reply = Message {
            role: Role::Assistant,
            content: vec![signed.clone()],
            provider_fields: None,
        };
        let nudged = convert_messages_to_claude_content(
            &[
                Message::user("a question"),
                reply,
                Message::user("Please reply"),
            ],
            None,
        );
        assert_eq!(
            nudged[1]["content"],
            serde_json::json!([{"type": "thinking", "thinking": "the reasoning", "signature": "sig"}])
        );

        let ending = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "an answer".to_string(),
                    provider_fields: None,
                },
                signed,
            ],
            provider_fields: None,
        };
        let ended =
            convert_messages_to_claude_content(&[Message::user("a question"), ending], None);
        assert_eq!(
            ended[1]["content"],
            serde_json::json!([{"type": "text", "text": "an answer"}]),
            "a request ending on the assistant still strips"
        );
    }

    /// A turn's two blocks reach this wire as two text blocks in order, the context ahead of the
    /// words, so the model reads what Meka injected before what the user typed.
    #[test]
    fn the_context_block_precedes_the_words() {
        let message = Message::user_turn("[Permission context]", "hello", Vec::new());
        let converted =
            convert_messages_to_claude_content(&[message], Some(CacheBreakpoint::Ephemeral));
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
                    display: None,
                },
            ],
            provider_fields: None,
        };
        let converted =
            convert_messages_to_claude_content(&[message], Some(CacheBreakpoint::Ephemeral));
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
                provider_fields: None,
            }],
            provider_fields: None,
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
        // The variant the turn's `rejection_may_blame_content` keys on: as `Provider` the refusal
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
    /// Requiring `message_stop` itself made Meka stricter than the format needs and broke every
    /// turn through such a shim, for a check whose actual subject -- a response cut mid-content --
    /// never reaches a stop reason at all.
    #[tokio::test]
    async fn a_stream_that_stops_after_its_stop_reason_is_complete() {
        let (events, outcome) = decode_sse(concat!(
            "event: content_block_start\n",
            "data: {\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        ))
        .await;

        assert!(outcome.is_ok(), "expected success, got {:?}", outcome.err());
        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::TextDelta(text) if text == "hi")),
            "the answer must arrive: {events:?}",
        );
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
    /// instead executes something the model never asked for -- `file_write` with no path, a shell
    /// command with no command -- and reports success for it. Rejecting hands the model back a
    /// result it can act on.
    #[tokio::test]
    async fn a_tool_call_with_unparseable_arguments_is_rejected_not_run_empty() {
        let (events, outcome) = decode_sse(concat!(
            "event: content_block_start\n",
            "data: {\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"file_write\"}}\n\n",
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
                StreamEvent::ToolCallRefused { name, .. } if name == "file_write"
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
            "event: content_block_start\n",
            "data: {\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"half an ans\"}}\n\n",
        ))
        .await;
        assert!(
            events.iter().any(
                |event| matches!(event, StreamEvent::TextDelta(text) if text == "half an ans")
            ),
            "what arrived is the reply so far: {events:?}",
        );

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
                StreamEvent::ThinkingComplete { opaque, .. } => Some(opaque.clone()),
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

    /// A `signature_delta` states the signature whole and replaces what the start held, as the
    /// reference client folds it, so a signature restated by a later delta is kept once.
    #[tokio::test]
    async fn a_signature_delta_replaces_the_signature_before_it() {
        let (events, outcome) = decode_sse(concat!(
            "event: content_block_start\n",
            "data: {\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"abc\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"abc\"}}\n\n",
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
                signature: "abc".to_string(),
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

    /// A signature belongs to the block whose `index` it arrived at. A gateway that omits a
    /// block's `content_block_stop` must not have the next block take the previous one's
    /// signature, pairing a signature with reasoning it does not authenticate, which Anthropic
    /// rejects as a modified block. `anthropic-messages` reaches any `base_url`, so the stream is
    /// untrusted input.
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
        let converted =
            convert_messages_to_claude_content(&[message], Some(CacheBreakpoint::OneHour));
        let blocks = converted[0]["content"].as_array().expect("content array");
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "image");
        assert_eq!(blocks[1]["source"]["type"], "base64");
        assert_eq!(blocks[1]["source"]["media_type"], "image/png");
        assert_eq!(blocks[1]["source"]["data"], "QUJD");
    }

    #[test]
    fn redacted_thinking_round_trips_verbatim() {
        let message = crate::conversation::Message {
            role: crate::conversation::Role::Assistant,
            content: vec![
                ContentBlock::RedactedThinking {
                    data: "ENCRYPTED_OPAQUE_BLOB".to_string(),
                    provider_fields: None,
                },
                ContentBlock::Text {
                    text: "hi".to_string(),
                    provider_fields: None,
                },
            ],
            provider_fields: None,
        };
        let converted =
            convert_messages_to_claude_content(&[message], Some(CacheBreakpoint::OneHour));
        let block = &converted[0]["content"].as_array().unwrap()[0];
        assert_eq!(block["type"], "redacted_thinking");
        assert_eq!(block["data"], "ENCRYPTED_OPAQUE_BLOB");
        // The opaque data must never be wrapped in `thinking`/`signature` fields.
        assert!(block.get("thinking").is_none());
        assert!(block.get("signature").is_none());
    }

    /// Deferred tools are declared beside the active ones with `defer_loading`, whether or not the
    /// request had a `tools` array to extend, and a request with none to declare is left alone.
    #[test]
    fn declare_deferred_tools_lists_them_with_defer_loading() {
        let deferred = [ToolDefinition::new(
            "mcp__kv__probe",
            "probe",
            serde_json::json!({"type": "object"}),
        )];
        let mut with_tools = serde_json::json!({"tools": [{"name": "file_read"}]});
        declare_deferred_tools(&mut with_tools, &deferred);
        assert_eq!(with_tools["tools"][0]["name"], "file_read");
        assert_eq!(with_tools["tools"][1]["name"], "mcp__kv__probe");
        assert_eq!(with_tools["tools"][1]["defer_loading"], true);
        assert!(with_tools["tools"][0].get("defer_loading").is_none());

        let mut without_tools = serde_json::json!({"model": "m"});
        declare_deferred_tools(&mut without_tools, &deferred);
        assert_eq!(without_tools["tools"][0]["defer_loading"], true);

        let mut nothing = serde_json::json!({"model": "m"});
        declare_deferred_tools(&mut nothing, &[]);
        assert!(nothing.get("tools").is_none());
    }

    /// A result whose references outlive some of their tools (a server renamed or withdrew one)
    /// keeps the references the request declares and leaves the rest out: the API rejects a
    /// reference it cannot find in `tools`, and text beside a reference as well. The result block
    /// keeps the breakpoint the converter set on it.
    #[test]
    fn a_result_keeps_only_the_references_the_request_declares() {
        let reference = |tool_name: &str| ToolResultContent::ToolReference {
            tool_name: tool_name.to_string(),
        };
        let message = crate::conversation::Message {
            role: crate::conversation::Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "toolu_1".to_string(),
                content: vec![
                    reference("mcp__gone__tool"),
                    reference("mcp__kv__probe"),
                    reference("mcp__gone__other"),
                ],
                is_error: false,
                display: None,
            }],
            provider_fields: None,
        };
        let mut body = serde_json::json!({
            "messages": convert_messages_to_claude_content(
                &[message],
                Some(CacheBreakpoint::Ephemeral),
            ),
        });
        let deferred = [ToolDefinition::new(
            "mcp__kv__probe",
            "probe",
            serde_json::json!({"type": "object"}),
        )];
        declare_deferred_tools(&mut body, &deferred);
        let result = &body["messages"][0]["content"][0];
        assert_eq!(
            result["content"],
            serde_json::json!([{"type": "tool_reference", "tool_name": "mcp__kv__probe"}]),
            "{result}"
        );
        assert_eq!(result["cache_control"]["type"], "ephemeral", "{result}");
    }

    /// A result with no reference the request can send names its tools in text, in place: every
    /// reference when none is declared, and every reference beside other content, which no
    /// declaration makes sendable.
    #[test]
    fn a_result_with_no_sendable_reference_names_its_tools_in_text() {
        let result = |content: serde_json::Value| {
            serde_json::json!({
                "messages": [{
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": content}],
                }],
            })
        };
        let deferred = [ToolDefinition::new(
            "mcp__kv__probe",
            "probe",
            serde_json::json!({"type": "object"}),
        )];

        let mut gone = result(serde_json::json!([
            {"type": "tool_reference", "tool_name": "mcp__gone__tool"},
            {"type": "tool_reference", "tool_name": "mcp__gone__other"},
        ]));
        declare_deferred_tools(&mut gone, &deferred);
        assert_eq!(
            gone["messages"][0]["content"][0]["content"],
            serde_json::json!([
                {"type": "text", "text": "mcp__gone__tool"},
                {"type": "text", "text": "mcp__gone__other"},
            ])
        );

        let mut beside_text = result(serde_json::json!([
            {"type": "text", "text": "loaded"},
            {"type": "tool_reference", "tool_name": "mcp__kv__probe"},
        ]));
        declare_deferred_tools(&mut beside_text, &deferred);
        assert_eq!(
            beside_text["messages"][0]["content"][0]["content"],
            serde_json::json!([
                {"type": "text", "text": "loaded"},
                {"type": "text", "text": "mcp__kv__probe"},
            ])
        );

        let mut undeclared = result(serde_json::json!([
            {"type": "tool_reference", "tool_name": "mcp__kv__probe"},
        ]));
        declare_deferred_tools(&mut undeclared, &[]);
        assert_eq!(
            undeclared["messages"][0]["content"][0]["content"],
            serde_json::json!([{"type": "text", "text": "mcp__kv__probe"}]),
            "nothing declared, nothing referenced"
        );
    }

    /// A successful result carries no `is_error`, the API's default and the shape Claude Code
    /// sends; a failed one says `true`.
    #[test]
    fn a_result_names_is_error_only_on_a_failure() {
        let result = |is_error: bool| crate::conversation::Message {
            role: crate::conversation::Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "toolu_1".to_string(),
                content: vec![ToolResultContent::Text {
                    text: "done".to_string(),
                }],
                is_error,
                display: None,
            }],
            provider_fields: None,
        };
        let ok = convert_messages_to_claude_content(&[result(false)], None);
        assert!(ok[0]["content"][0].get("is_error").is_none(), "{ok:?}");
        assert_eq!(
            ok[0]["content"][0]
                .as_object()
                .map(|block| block.keys().cloned().collect::<Vec<_>>()),
            Some(vec![
                "type".to_string(),
                "tool_use_id".to_string(),
                "content".to_string()
            ])
        );
        let failed = convert_messages_to_claude_content(&[result(true)], None);
        assert_eq!(failed[0]["content"][0]["is_error"], true);
    }

    /// A tool reference in a result goes to Claude as the `tool_reference` block the API expands.
    #[test]
    fn a_tool_reference_in_a_result_is_a_reference_block() {
        let message = crate::conversation::Message {
            role: crate::conversation::Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "toolu_1".to_string(),
                content: vec![
                    ToolResultContent::Text {
                        text: "loaded".to_string(),
                    },
                    ToolResultContent::ToolReference {
                        tool_name: "mcp__kv__probe".to_string(),
                    },
                ],
                is_error: false,
                display: None,
            }],
            provider_fields: None,
        };
        let converted = convert_messages_to_claude_content(&[message], None);
        let content = &converted[0]["content"][0]["content"];
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "tool_reference", "{content}");
        assert_eq!(content[1]["tool_name"], "mcp__kv__probe");
    }

    /// The mirror of the other encoders' refusals, for a session recorded against OpenAI or a
    /// Chat Completions endpoint and resumed under Claude. Sealed reasoning is not a Claude
    /// signature and verbatim reasoning has none, and a thinking block without one is rejected,
    /// so the block is left out: neither the blob nor the text reaches the wire.
    #[test]
    fn another_wires_reasoning_is_never_sent_to_claude_as_a_signature() {
        for opaque in [
            OpaqueReasoning::Sealed {
                encrypted_content: "OPENAI_SEALED".to_string(),
                id: Some("rs_1".to_string()),
                origin: Some("https://chatgpt.com".to_string()),
            },
            OpaqueReasoning::Verbatim,
        ] {
            let message = crate::conversation::Message {
                role: crate::conversation::Role::Assistant,
                content: vec![
                    ContentBlock::Thinking {
                        parts: vec![crate::conversation::ReasoningPart::reasoning(
                            "the reasoning",
                        )],
                        opaque: Some(opaque.clone()),
                        provider_fields: None,
                    },
                    // Trailing thinking is stripped from the last assistant message, so the block
                    // has to be followed by something for this to be testing the encoder at all.
                    ContentBlock::Text {
                        text: "answer".to_string(),
                        provider_fields: None,
                    },
                ],
                provider_fields: None,
            };
            let converted =
                convert_messages_to_claude_content(&[message], Some(CacheBreakpoint::OneHour));
            let content = converted[0]["content"].as_array().expect("content");

            assert_eq!(content.len(), 1, "{opaque:?}: {content:?}");
            assert_eq!(content[0]["type"], "text");
            assert_eq!(content[0]["text"], "answer");
            let serialized = serde_json::to_string(&converted).expect("serialize");
            assert!(
                !serialized.contains("OPENAI_SEALED") && !serialized.contains("the reasoning"),
                "{opaque:?}: {serialized}"
            );
        }
    }

    /// An Anthropic-compatible endpoint that does not sign its thinking (the `synthetic` backend)
    /// leaves `opaque` empty. Replayed as-is, such a block fails a strict endpoint's validation
    /// with `signature: expected string` before any repair tier runs; left out, the session
    /// resumes anywhere.
    #[test]
    fn an_unsigned_thinking_block_is_left_out_of_a_claude_request() {
        let message = crate::conversation::Message {
            role: crate::conversation::Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    parts: vec![crate::conversation::ReasoningPart::reasoning(
                        "unsigned reasoning",
                    )],
                    opaque: None,
                    provider_fields: None,
                },
                ContentBlock::Text {
                    text: "answer".to_string(),
                    provider_fields: None,
                },
            ],
            provider_fields: None,
        };
        let converted =
            convert_messages_to_claude_content(&[message], Some(CacheBreakpoint::OneHour));
        let content = converted[0]["content"].as_array().expect("content");

        assert_eq!(content.len(), 1, "{content:?}");
        assert_eq!(content[0]["text"], "answer");
        assert!(
            !serde_json::to_string(&converted)
                .expect("serialize")
                .contains("unsigned reasoning")
        );
    }

    /// OpenRouter answers for a non-Anthropic model with `signature: ""`, which the accumulator
    /// keeps as the value the API returned, and it goes back so: the endpoint that returned it
    /// takes it, and one that rejects it says so.
    #[test]
    fn an_empty_signature_goes_back_as_returned() {
        let message = crate::conversation::Message {
            role: crate::conversation::Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    parts: vec![crate::conversation::ReasoningPart::reasoning(
                        "openrouter reasoning",
                    )],
                    opaque: Some(OpaqueReasoning::Signed {
                        signature: String::new(),
                    }),
                    provider_fields: None,
                },
                ContentBlock::Text {
                    text: "answer".to_string(),
                    provider_fields: None,
                },
            ],
            provider_fields: None,
        };
        let converted =
            convert_messages_to_claude_content(&[message], Some(CacheBreakpoint::OneHour));
        let content = converted[0]["content"].as_array().expect("content");

        assert_eq!(content.len(), 2, "{content:?}");
        assert_eq!(
            content[0],
            serde_json::json!({"type": "thinking", "thinking": "openrouter reasoning",
                "signature": ""})
        );
        assert_eq!(content[1]["text"], "answer");
    }

    /// The citations a text block came with go back on it, where the Messages request defines
    /// them; an empty list and the block's other fields do not.
    #[test]
    fn a_text_block_s_citations_go_back_on_it() {
        let citation = serde_json::json!({"type": "web_search_result_location", "url": "u",
            "title": "t", "encrypted_index": "E", "cited_text": "c"});
        let text = |fields: serde_json::Value| ContentBlock::Text {
            text: "cited".to_string(),
            provider_fields: fields.as_object().cloned(),
        };
        let message = crate::conversation::Message {
            role: crate::conversation::Role::Assistant,
            content: vec![
                text(serde_json::json!({"citations": [citation.clone()], "other": 1})),
                text(serde_json::json!({"citations": []})),
            ],
            provider_fields: None,
        };
        let converted = convert_messages_to_claude_content(&[message], None);
        let content = converted[0]["content"].as_array().expect("content");
        assert_eq!(
            content[0],
            serde_json::json!({"type": "text", "text": "cited", "citations": [citation]})
        );
        assert_eq!(
            content[1],
            serde_json::json!({"type": "text", "text": "cited"})
        );
    }

    /// A reply recorded with nothing in it goes back as the placeholder, since Claude requires
    /// every message to hold a block.
    #[test]
    fn an_empty_reply_goes_out_as_the_placeholder() {
        let messages = [
            crate::conversation::Message::user("hi"),
            crate::conversation::Message {
                role: crate::conversation::Role::Assistant,
                content: Vec::new(),
                provider_fields: None,
            },
            crate::conversation::Message::user("again"),
        ];
        let converted = convert_messages_to_claude_content(&messages, None);
        assert_eq!(
            converted[1]["content"],
            serde_json::json!([{"type": "text", "text": "[No message content]"}])
        );
    }

    /// Leaving the block out must not leave the message empty, which the API rejects too. The
    /// emptied turn sits ahead of a later assistant turn, so the trailing strip is not what fills
    /// it.
    #[test]
    fn a_message_of_only_unsigned_thinking_goes_out_as_the_placeholder() {
        let messages = [
            crate::conversation::Message {
                role: crate::conversation::Role::Assistant,
                content: vec![ContentBlock::Thinking {
                    parts: vec![crate::conversation::ReasoningPart::reasoning(
                        "unsigned reasoning",
                    )],
                    opaque: None,
                    provider_fields: None,
                }],
                provider_fields: None,
            },
            crate::conversation::Message::user("next"),
            crate::conversation::Message::assistant_text("later"),
        ];
        let converted =
            convert_messages_to_claude_content(&messages, Some(CacheBreakpoint::OneHour));
        let content = converted[0]["content"].as_array().expect("content");

        assert_eq!(content.len(), 1, "{content:?}");
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "[No message content]");
    }

    /// A reply that streamed only whitespace before it was cut is saved as such, and the API
    /// refuses a text block with nothing but whitespace in it: left in, every later request
    /// fails. It goes out as the placeholder, and a blank line in a tool's result is left out.
    #[test]
    fn whitespace_only_text_never_reaches_the_wire() {
        let messages = [
            crate::conversation::Message::user("first"),
            crate::conversation::Message::assistant_text("\n\n"),
            crate::conversation::Message {
                role: crate::conversation::Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call".to_string(),
                    content: vec![
                        ToolResultContent::Text {
                            text: " ".to_string(),
                        },
                        ToolResultContent::Text {
                            text: "kept".to_string(),
                        },
                    ],
                    is_error: false,
                    display: None,
                }],
                provider_fields: None,
            },
        ];
        let converted = convert_messages_to_claude_content(&messages, None);
        assert_eq!(
            converted[1]["content"],
            serde_json::json!([{"type": "text", "text": "[No message content]"}])
        );
        assert_eq!(
            converted[2]["content"][0]["content"],
            serde_json::json!([{"type": "text", "text": "kept"}])
        );
    }

    #[test]
    fn empty_thinking_block_with_signature_serializes_signature() {
        let message = crate::conversation::Message {
            role: crate::conversation::Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    parts: vec![crate::conversation::ReasoningPart::reasoning("")],
                    opaque: Some(OpaqueReasoning::Signed {
                        signature: "SIG620".to_string(),
                    }),
                    provider_fields: None,
                },
                ContentBlock::Text {
                    text: "answer".to_string(),
                    provider_fields: None,
                },
            ],
            provider_fields: None,
        };
        let converted =
            convert_messages_to_claude_content(&[message], Some(CacheBreakpoint::OneHour));
        let block = &converted[0]["content"].as_array().unwrap()[0];
        assert_eq!(block["type"], "thinking");
        assert_eq!(block["thinking"], "");
        assert_eq!(block["signature"], "SIG620");
    }

    /// What Meka writes into a conversation reaches this wire rendered: the summary, a note behind
    /// Meka's label, a call whose arguments were not an object with `{}`, the protocol's own
    /// placeholder for a call's input, and a result's caption and large output's reference.
    #[test]
    fn what_meka_wrote_reaches_the_wire_rendered() {
        let sent = serde_json::to_string(&convert_messages_to_claude_content(
            &Message::typed_content_for_test(),
            None,
        ))
        .expect("json");
        for needle in [
            "the summary",
            "[Meka harness] the note",
            r#""input":{}"#,
            "[Meka harness] the schema note",
            "[Image: a.png]",
            r#"<large-output name=\"the_entry\""#,
        ] {
            assert!(sent.contains(needle), "{needle} is not sent: {sent}");
        }
        assert!(
            !sent.contains("path"),
            "nothing of what the model sent: {sent}"
        );
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
        // The input image block became Meka's note.
        assert!(matches!(
            &redacted[0].content[1],
            ContentBlock::HarnessNote { .. }
        ));
    }

    #[test]
    fn each_thinking_mode_writes_its_own_encoding_and_ceiling() {
        // The encoding now comes from the profile, not from the model name, so the same model can
        // legitimately be asked for either form - which is what an arbitrary Anthropic-compatible
        // endpoint needs.
        let mut adaptive = serde_json::Map::new();
        insert_thinking_fields(&mut adaptive, ThinkingMode::Adaptive, 10_000, None, None);
        assert_eq!(adaptive["max_tokens"], 128_000);
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
            "claude-opus-5-5",
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
    fn per_turn_effort_goes_to_the_six_models_the_catalog_grants_it() {
        for model in [
            "claude-opus-5-5",
            "claude-sonnet-5-5",
            "claude-haiku-5-5",
            "claude-fable-5-1",
            "claude-fable-5-2",
            "claude-mythos-5-2",
        ] {
            assert!(model_supports_per_turn_effort(model), "{model}");
        }
        // An allowlist: the rest of the line-up and anything newer are left out, since Claude
        // Code sends nothing for a model its catalog does not list.
        for model in [
            "claude-opus-5",
            "claude-opus-4-8",
            "claude-fable-5",
            "claude-sonnet-5",
            "claude-mythos-5-1",
            "claude-opus-6",
            "claude-custom",
        ] {
            assert!(!model_supports_per_turn_effort(model), "{model}");
        }
    }

    #[test]
    fn mid_conversation_tool_changes_follow_the_catalog_s_denylist() {
        // Every catalog entry with `mid_conv_tool_change`, and anything the catalog does not list.
        for model in [
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-opus-5-5",
            "claude-sonnet-5-5",
            "claude-haiku-5-5",
            "claude-fable-5-1",
            "claude-mythos-5",
            "claude-opus-7",
            "claude-something-new",
        ] {
            assert!(
                model_supports_mid_conversation_tool_changes(model),
                "{model}"
            );
        }
        // The denylist itself: Opus 4.7 and down, Sonnet 5 and down, Haiku 4.5, Claude 3.x.
        for model in [
            "claude-sonnet-5",
            "claude-sonnet-4-6",
            "claude-sonnet-4-5",
            "claude-opus-4-7",
            "claude-opus-4-6-20250514",
            "claude-opus-4-0",
            "claude-haiku-4-5",
            "claude-3-5-sonnet-20241022",
        ] {
            assert!(
                !model_supports_mid_conversation_tool_changes(model),
                "{model}"
            );
        }
    }

    /// Every field a Claude reply returns has a home: a citation folds onto its text block as the
    /// reference client folds it, a delta Meka has no block for is kept whole after the block it
    /// came in without splitting its text, a block Meka has no block for is kept whole in its
    /// place, and the message's own fields, folded with its `message_delta`, are the record's,
    /// with the raw usage and without what the reply holds.
    #[tokio::test]
    async fn every_field_a_claude_reply_returns_has_a_home() {
        let frames = [
            serde_json::json!({"type": "message_start", "message": {"id": "msg_1",
                "type": "message", "role": "assistant", "model": "claude-x", "content": [],
                "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": 5, "output_tokens": 1}}}),
            serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": "", "citations": null}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "Per the "}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "citations_delta", "citation": {"cited_text": "a"}}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "future_delta", "x": 1}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "docs."}}),
            serde_json::json!({"type": "content_block_stop", "index": 0}),
            serde_json::json!({"type": "content_block_start", "index": 1,
                "content_block": {"type": "server_tool_use", "id": "srv_1", "name": "web_search",
                    "input": {}}}),
            serde_json::json!({"type": "content_block_delta", "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": "{}"}}),
            serde_json::json!({"type": "content_block_stop", "index": 1}),
            serde_json::json!({"type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": null, "container": {"id": "c1"}},
                "usage": {"output_tokens": 9}}),
            serde_json::json!({"type": "message_stop"}),
        ];
        let mut protocol = super::ClaudeStream::default();
        let (sender, mut receiver) = tokio::sync::mpsc::channel(64);
        for frame in frames {
            let name = frame["type"].as_str().expect("a type").to_string();
            let event = eventsource_stream::Event {
                event: name,
                data: frame.to_string(),
                id: String::new(),
                retry: None,
            };
            crate::provider::sse::Protocol::frame(&mut protocol, event, &sender)
                .await
                .expect("frame");
        }
        drop(sender);
        let mut accumulator = crate::provider::MessageAccumulator::new();
        while let Some(event) = receiver.recv().await {
            accumulator.push(event);
        }
        let completion = accumulator.finish();
        let [
            ContentBlock::Text {
                text,
                provider_fields: Some(text_fields),
            },
            ContentBlock::Unrecognized { parts: delta },
            ContentBlock::Unrecognized { parts: block },
        ] = completion.message.content.as_slice()
        else {
            panic!("{:?}", completion.message.content);
        };
        assert_eq!(text, "Per the docs.", "one block, unsplit");
        assert_eq!(
            text_fields["citations"],
            serde_json::json!([{"cited_text": "a"}])
        );
        assert_eq!(
            delta.as_slice(),
            [
                serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "future_delta", "x": 1}})
            ],
            "kept as the event it came in, so it names its block"
        );
        assert_eq!(block.len(), 2, "{block:?}");
        assert_eq!(block[0]["content_block"]["type"], "server_tool_use");
        assert_eq!(block[1]["delta"]["type"], "input_json_delta");
        let record = completion
            .response_fields
            .expect("the message's own fields");
        assert_eq!(record["id"], "msg_1");
        assert_eq!(record["container"], serde_json::json!({"id": "c1"}));
        assert_eq!(record["usage"]["output_tokens"], 9);
        assert_eq!(record["usage"]["input_tokens"], 5);
        for held in ["content", "model", "role", "stop_reason", "stop_sequence"] {
            assert!(
                !record.contains_key(held),
                "{held} is the reply's: {record:?}"
            );
        }
    }

    /// The events `frames` stream as through the driver, each sent as the SSE event its `type`
    /// names, with whatever the stream's end closes; the stream must end as a message does.
    async fn claude_events(frames: &[serde_json::Value]) -> Vec<StreamEvent> {
        let body: String = frames
            .iter()
            .map(|frame| {
                format!(
                    "event: {}\ndata: {frame}\n\n",
                    frame["type"].as_str().expect("a type")
                )
            })
            .collect();
        let (events, outcome) = decode_sse(&body).await;
        outcome.expect("the stream ends as a message");
        events
    }

    /// The completion `frames` fold to.
    async fn claude_completion(frames: &[serde_json::Value]) -> crate::provider::Completion {
        let mut accumulator = crate::provider::MessageAccumulator::new();
        for event in claude_events(frames).await {
            accumulator.push(event);
        }
        accumulator.finish()
    }

    /// A block's `index` is its place in the message, so blocks that overlap on the wire (as
    /// OpenRouter streams them) fold in index order, each whole: a text block that holds only its
    /// citations keeps them, and a block that opens inside another follows it. Each is handed
    /// over as it arrives, the answer before the block open around it closes.
    #[tokio::test]
    async fn blocks_fold_in_index_order_when_they_overlap() {
        let start = serde_json::json!({"type": "message_start", "message": {"id": "msg_1",
            "type": "message", "role": "assistant", "model": "m", "content": [],
            "usage": {"input_tokens": 1, "output_tokens": 1}}});
        let end = serde_json::json!({"type": "message_delta",
            "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 9}});
        let citation = serde_json::json!({"type": "web_search_result_location", "url": "u"});
        let around = [
            start.clone(),
            serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": "", "citations": []}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "citations_delta", "citation": citation}}),
            serde_json::json!({"type": "content_block_start", "index": 1,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
            serde_json::json!({"type": "content_block_delta", "index": 1,
                "delta": {"type": "thinking_delta", "thinking": "hmm"}}),
            serde_json::json!({"type": "content_block_delta", "index": 1,
                "delta": {"type": "signature_delta", "signature": "S"}}),
            serde_json::json!({"type": "content_block_stop", "index": 1}),
            serde_json::json!({"type": "content_block_start", "index": 2,
                "content_block": {"type": "text", "text": "", "citations": []}}),
            serde_json::json!({"type": "content_block_delta", "index": 2,
                "delta": {"type": "text_delta", "text": "The answer."}}),
            serde_json::json!({"type": "content_block_stop", "index": 2}),
            serde_json::json!({"type": "content_block_stop", "index": 0}),
            end.clone(),
        ];
        let events = claude_events(&around).await;
        let position = |wanted: fn(&StreamEvent) -> bool| {
            events
                .iter()
                .position(wanted)
                .unwrap_or_else(|| panic!("{events:?}"))
        };
        let answered = position(
            |event| matches!(event, StreamEvent::TextDelta(text) if text == "The answer."),
        );
        let cited = position(|event| {
            matches!(event, StreamEvent::TextEnd {
                provider_fields: Some(_)
            })
        });
        assert!(answered < cited, "{events:?}");
        let completion = claude_completion(&around).await;
        let [
            ContentBlock::Text {
                text: cited,
                provider_fields: Some(cited_fields),
            },
            ContentBlock::Thinking {
                parts,
                opaque: Some(OpaqueReasoning::Signed { signature }),
                ..
            },
            ContentBlock::Text { text: answer, .. },
        ] = completion.message.content.as_slice()
        else {
            panic!("{:?}", completion.message.content);
        };
        assert_eq!(cited, "");
        assert_eq!(cited_fields["citations"], serde_json::json!([citation]));
        assert_eq!(crate::conversation::reasoning_text(parts), "hmm");
        assert_eq!(signature, "S");
        assert_eq!(answer, "The answer.");

        let completion = claude_completion(&[
            start,
            serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "summary"}}),
            serde_json::json!({"type": "content_block_start", "index": 1,
                "content_block": {"type": "redacted_thinking", "data": "SEALED"}}),
            serde_json::json!({"type": "content_block_stop", "index": 1}),
            serde_json::json!({"type": "content_block_stop", "index": 0}),
            serde_json::json!({"type": "content_block_start", "index": 2,
                "content_block": {"type": "text", "text": "", "citations": []}}),
            serde_json::json!({"type": "content_block_delta", "index": 2,
                "delta": {"type": "text_delta", "text": "899"}}),
            serde_json::json!({"type": "content_block_stop", "index": 2}),
            end,
        ])
        .await;
        let kinds: Vec<_> = completion
            .message
            .content
            .iter()
            .map(|block| match block {
                ContentBlock::Thinking { .. } => "thinking",
                ContentBlock::RedactedThinking { .. } => "redacted_thinking",
                ContentBlock::Text { .. } => "text",
                _ => "other",
            })
            .collect();
        assert_eq!(kinds, ["thinking", "redacted_thinking", "text"]);
    }

    /// Under the `redact-thinking` beta a withheld block streams its progress as `thinking_delta`
    /// estimates with no words: each is shown as progress while the block runs, the block is
    /// handed over when it stops, ahead of the next, and the record holds the block alone.
    #[tokio::test]
    async fn a_redacted_block_shows_its_progress_and_ends_when_it_stops() {
        let progress = |estimated: u64| {
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "", "estimated_tokens": estimated}})
        };
        let frames = [
            serde_json::json!({"type": "message_start", "message": {"id": "msg_1",
                "type": "message", "role": "assistant", "model": "m", "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 1}}}),
            serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "redacted_thinking", "data": "SEALED"}}),
            progress(3),
            progress(9),
            serde_json::json!({"type": "content_block_stop", "index": 0}),
            serde_json::json!({"type": "content_block_start", "index": 1,
                "content_block": {"type": "text", "text": ""}}),
            serde_json::json!({"type": "content_block_delta", "index": 1,
                "delta": {"type": "text_delta", "text": "answer"}}),
            serde_json::json!({"type": "content_block_stop", "index": 1}),
            serde_json::json!({"type": "message_delta",
                "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 9}}),
        ];
        let events = claude_events(&frames).await;
        let order: Vec<String> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ThinkingProgress { estimated_tokens } => Some(format!(
                    "progress {}",
                    estimated_tokens.map_or("-".to_string(), |estimated| estimated.to_string())
                )),
                StreamEvent::RedactedThinking { data, .. } => Some(format!("redacted {data}")),
                StreamEvent::TextDelta(text) => Some(format!("text {text}")),
                _ => None,
            })
            .collect();
        assert_eq!(order, [
            "progress -",
            "progress 3",
            "progress 9",
            "redacted SEALED",
            "text answer"
        ]);
        let completion = claude_completion(&frames).await;
        assert!(
            matches!(completion.message.content.as_slice(), [
                ContentBlock::RedactedThinking { data, .. },
                ContentBlock::Text { text, .. },
            ] if data == "SEALED" && text == "answer"),
            "{:?}",
            completion.message.content
        );
    }

    /// A block whose opening event already carries content keeps it, as the reference client
    /// does: text and reasoning start with what the start held, and a call that streams no
    /// argument deltas runs with the input its start carried.
    #[tokio::test]
    async fn a_block_that_arrives_whole_in_its_opening_event_keeps_what_it_carried() {
        let completion = claude_completion(&[
            serde_json::json!({"type": "message_start", "message": {"id": "msg_1",
                "type": "message", "role": "assistant", "model": "m", "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 1}}}),
            serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "thinking", "thinking": "whole", "signature": "S"}}),
            serde_json::json!({"type": "content_block_stop", "index": 0}),
            serde_json::json!({"type": "content_block_start", "index": 1,
                "content_block": {"type": "text", "text": "Hello "}}),
            serde_json::json!({"type": "content_block_delta", "index": 1,
                "delta": {"type": "text_delta", "text": "world"}}),
            serde_json::json!({"type": "content_block_stop", "index": 1}),
            serde_json::json!({"type": "content_block_start", "index": 2,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "file_read",
                    "input": {"path": "a"}}}),
            serde_json::json!({"type": "content_block_stop", "index": 2}),
            serde_json::json!({"type": "message_delta",
                "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}}),
        ])
        .await;
        let [
            ContentBlock::Thinking {
                parts,
                opaque: Some(OpaqueReasoning::Signed { signature }),
                ..
            },
            ContentBlock::Text { text, .. },
            ContentBlock::ToolUse {
                input, unparsed, ..
            },
        ] = completion.message.content.as_slice()
        else {
            panic!("{:?}", completion.message.content);
        };
        assert_eq!(crate::conversation::reasoning_text(parts), "whole");
        assert_eq!(signature, "S");
        assert_eq!(text, "Hello world");
        assert_eq!(input, &serde_json::json!({"path": "a"}));
        assert!(unparsed.is_none());
    }

    /// A block that never closed before the message ended is handed over as one that never
    /// finished, its reasoning unsigned, since a signature that arrived before the block ended may
    /// not cover all of it, and the blocks after it keep their own places and words.
    #[tokio::test]
    async fn a_block_left_open_is_unfinished_and_the_ones_after_it_keep_their_places() {
        let start = serde_json::json!({"type": "message_start", "message": {"id": "msg_1",
            "type": "message", "role": "assistant", "model": "m", "content": [],
            "usage": {"input_tokens": 1, "output_tokens": 1}}});
        let later_text = [
            serde_json::json!({"type": "content_block_start", "index": 1,
                "content_block": {"type": "text", "text": ""}}),
            serde_json::json!({"type": "content_block_delta", "index": 1,
                "delta": {"type": "text_delta", "text": "later"}}),
            serde_json::json!({"type": "content_block_stop", "index": 1}),
        ];
        let end = serde_json::json!({"type": "message_delta",
            "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 9}});
        let mut frames = vec![
            start.clone(),
            serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "open"}}),
        ];
        frames.extend(later_text.clone());
        frames.push(end.clone());
        let completion = claude_completion(&frames).await;
        let texts: Vec<_> = completion
            .message
            .content
            .iter()
            .map(|block| match block {
                ContentBlock::Text { text, .. } => text.as_str(),
                _ => "other",
            })
            .collect();
        assert_eq!(texts, ["open", "later"]);

        let mut frames = vec![
            start,
            serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "partial"}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "signature_delta", "signature": "S"}}),
        ];
        frames.extend(later_text);
        frames.push(end.clone());
        let completion = claude_completion(&frames).await;
        let [
            ContentBlock::Thinking { parts, opaque, .. },
            ContentBlock::Text { text, .. },
        ] = completion.message.content.as_slice()
        else {
            panic!("{:?}", completion.message.content);
        };
        assert_eq!(crate::conversation::reasoning_text(parts), "partial");
        assert_eq!(opaque, &None);
        assert_eq!(text, "later");

        // Redacted reasoning arrives whole in its start and is handed over only when its block
        // closes, which for one left open is the stream's end.
        let completion = claude_completion(&[
            serde_json::json!({"type": "message_start", "message": {"id": "msg_1",
                "type": "message", "role": "assistant", "model": "m", "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 1}}}),
            serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "redacted_thinking", "data": "SEALED"}}),
            end,
        ])
        .await;
        let [ContentBlock::RedactedThinking { data, .. }] = completion.message.content.as_slice()
        else {
            panic!("{:?}", completion.message.content);
        };
        assert_eq!(data, "SEALED");
    }

    /// A thinking block whose text the request asked to be summarized is recorded as a summary
    /// part, and as the reasoning itself otherwise; what goes back to Claude is the same text.
    #[tokio::test]
    async fn summarized_thinking_is_recorded_as_a_summary() {
        for (summarized, kind) in [
            (true, crate::conversation::ReasoningKind::Summary),
            (false, crate::conversation::ReasoningKind::Reasoning),
        ] {
            let body = concat!(
                "event: content_block_start\n",
                "data: {\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n",
                "event: content_block_delta\n",
                "data: {\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"a plan\"}}\n\n",
                "event: content_block_delta\n",
                "data: {\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"S\"}}\n\n",
                "event: content_block_stop\n",
                "data: {\"index\":0}\n\n",
                "event: message_delta\n",
                "data: {\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
            );
            let response: reqwest::Response = http::Response::builder()
                .status(200)
                .header("content-type", "text/event-stream")
                .body(body.to_string())
                .expect("build response")
                .into();
            let (sender, mut receiver) = mpsc::channel(64);
            drive_claude_sse_stream(response, sender, CancellationToken::new(), None, summarized)
                .await
                .expect("a whole stream");
            let mut accumulator = crate::provider::MessageAccumulator::new();
            while let Ok(event) = receiver.try_recv() {
                accumulator.push(event);
            }
            let content = accumulator.finish().message.content;
            let [ContentBlock::Thinking { parts, .. }] = content.as_slice() else {
                panic!("{content:?}");
            };
            assert_eq!(parts.len(), 1, "{parts:?}");
            assert_eq!(parts[0].kind, kind);
            assert_eq!(crate::conversation::reasoning_as_streamed(parts), "a plan");
        }
    }

    /// A `message_delta` without a stop reason between a block's deltas restates the running
    /// usage, as a gateway may send one after every delta, and closes nothing: each block folds
    /// whole, its reasoning signed, as the reference client folds it.
    #[tokio::test]
    async fn a_message_delta_without_a_stop_reason_leaves_its_blocks_open() {
        let running = |output_tokens: u64| {
            serde_json::json!({"type": "message_delta",
                "delta": {"stop_reason": null, "stop_sequence": null},
                "usage": {"input_tokens": 5, "output_tokens": output_tokens}})
        };
        let completion = claude_completion(&[
            serde_json::json!({"type": "message_start", "message": {"id": "msg_1",
                "type": "message", "role": "assistant", "model": "m", "content": [],
                "usage": {"input_tokens": 0, "output_tokens": 0}}}),
            running(0),
            serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "The"}}),
            running(1),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "thinking_delta", "thinking": " user"}}),
            running(2),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "signature_delta", "signature": "S"}}),
            serde_json::json!({"type": "content_block_stop", "index": 0}),
            serde_json::json!({"type": "content_block_start", "index": 1,
                "content_block": {"type": "text", "text": ""}}),
            serde_json::json!({"type": "content_block_delta", "index": 1,
                "delta": {"type": "text_delta", "text": "**EPT"}}),
            running(3),
            serde_json::json!({"type": "content_block_delta", "index": 1,
                "delta": {"type": "text_delta", "text": "** is"}}),
            running(4),
            serde_json::json!({"type": "content_block_stop", "index": 1}),
            serde_json::json!({"type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                "usage": {"input_tokens": 5, "output_tokens": 4}}),
            serde_json::json!({"type": "message_stop"}),
        ])
        .await;
        let [
            ContentBlock::Thinking {
                parts,
                opaque: Some(OpaqueReasoning::Signed { signature }),
                ..
            },
            ContentBlock::Text { text, .. },
        ] = completion.message.content.as_slice()
        else {
            panic!("{:?}", completion.message.content);
        };
        assert_eq!(crate::conversation::reasoning_text(parts), "The user");
        assert_eq!(signature, "S");
        assert_eq!(text, "**EPT** is");
        assert_eq!(completion.usage.output_tokens, 4);
    }

    /// A block whose stop arrives after the delta that states the message's stop is whole when it
    /// stops, its reasoning signed and its call run, as the reference client folds it: the stop
    /// reason closes no block.
    #[tokio::test]
    async fn a_block_that_stops_after_the_stop_reason_is_whole() {
        let completion = claude_completion(&[
            serde_json::json!({"type": "message_start", "message": {"id": "msg_1",
                "type": "message", "role": "assistant", "model": "m", "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 1}}}),
            serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "a"}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "signature_delta", "signature": "S"}}),
            serde_json::json!({"type": "content_block_start", "index": 1,
                "content_block": {"type": "tool_use", "id": "t1", "name": "n", "input": {}}}),
            serde_json::json!({"type": "content_block_delta", "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": "{\"a\": 1}"}}),
            serde_json::json!({"type": "message_delta",
                "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}),
            serde_json::json!({"type": "content_block_stop", "index": 0}),
            serde_json::json!({"type": "content_block_stop", "index": 1}),
            serde_json::json!({"type": "message_stop"}),
        ])
        .await;
        let [
            ContentBlock::Thinking {
                opaque: Some(OpaqueReasoning::Signed { signature }),
                ..
            },
            ContentBlock::ToolUse {
                input,
                unparsed: None,
                ..
            },
        ] = completion.message.content.as_slice()
        else {
            panic!("{:?}", completion.message.content);
        };
        assert_eq!(signature, "S");
        assert_eq!(input, &serde_json::json!({"a": 1}));
    }

    /// The reply's usage is the message's as its statements fold: a count stated again replaces
    /// the last, zero included, and one left out or null keeps it, which is the reference client's
    /// fold of every count the spec lets a delta leave out. The reply's counts and the usage its
    /// record keeps are one.
    #[tokio::test]
    async fn the_reply_s_usage_is_the_message_s_as_its_statements_fold() {
        let completion = claude_completion(&[
            serde_json::json!({"type": "message_start", "message": {"id": "msg_1",
                "type": "message", "role": "assistant", "model": "m", "content": [],
                "usage": {"input_tokens": 10, "output_tokens": 1,
                    "cache_creation_input_tokens": 2, "cache_read_input_tokens": 5}}}),
            serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "x"}}),
            serde_json::json!({"type": "message_delta", "delta": {"stop_reason": null},
                "usage": {"input_tokens": null, "output_tokens": 3}}),
            serde_json::json!({"type": "content_block_stop", "index": 0}),
            serde_json::json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 7, "cache_read_input_tokens": 0}}),
            serde_json::json!({"type": "message_stop"}),
        ])
        .await;
        assert_eq!(completion.usage, TokenUsage {
            input_tokens: 10,
            output_tokens: 7,
            cache_creation_input_tokens: 2,
            cache_read_input_tokens: 0,
        });
        let recorded = completion.response_fields.expect("the response's fields");
        assert_eq!(parse_usage_object(&recorded["usage"]), completion.usage);
    }

    /// A text block that opened and stopped with no words in it is the reply's as returned, ahead
    /// of the call after it.
    #[tokio::test]
    async fn a_text_block_with_no_words_is_kept_as_returned() {
        let completion = claude_completion(&[
            serde_json::json!({"type": "message_start", "message": {"id": "msg_1",
                "type": "message", "role": "assistant", "model": "m", "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 1}}}),
            serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}}),
            serde_json::json!({"type": "content_block_stop", "index": 0}),
            serde_json::json!({"type": "content_block_start", "index": 1,
                "content_block": {"type": "tool_use", "id": "t1", "name": "n", "input": {}}}),
            serde_json::json!({"type": "content_block_stop", "index": 1}),
            serde_json::json!({"type": "message_delta",
                "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}),
            serde_json::json!({"type": "message_stop"}),
        ])
        .await;
        let [
            ContentBlock::Text { text, .. },
            ContentBlock::ToolUse { .. },
        ] = completion.message.content.as_slice()
        else {
            panic!("{:?}", completion.message.content);
        };
        assert_eq!(text, "");
    }

    /// What the spec does not define is kept as it came: content the message states rather than
    /// streams, by `message_start` or `message_delta`, a stop for a block that is not open, a stop
    /// the message states before any `message_delta` does, which the reply's record never
    /// receives, and a `usage` that is not an object, which leaves the counts stated so far
    /// standing.
    #[tokio::test]
    async fn frames_the_spec_does_not_define_are_kept_as_they_came() {
        let stray_stop = serde_json::json!({"type": "content_block_stop", "index": 0});
        let text_block = [
            serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "hi"}}),
            serde_json::json!({"type": "content_block_stop", "index": 0}),
        ];
        let mut frames = vec![serde_json::json!({"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [{"type": "text", "text": "early"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}}})];
        frames.extend(text_block.clone());
        frames.extend([
            stray_stop.clone(),
            serde_json::json!({"type": "message_stop"}),
        ]);
        let completion = claude_completion(&frames).await;
        let [
            ContentBlock::Text { text, .. },
            ContentBlock::Unrecognized { parts: stopped },
        ] = completion.message.content.as_slice()
        else {
            panic!("{:?}", completion.message.content);
        };
        assert_eq!(text, "hi");
        assert_eq!(stopped, &vec![stray_stop]);
        let fields = completion.response_fields.expect("the response's fields");
        assert_eq!(
            fields["content"],
            serde_json::json!([{"type": "text", "text": "early"}])
        );
        assert_eq!(fields["stop_reason"], "end_turn");

        let closing = serde_json::json!({"type": "message_delta",
            "delta": {"stop_reason": "end_turn", "content": "late"}, "usage": 5});
        let mut frames = vec![serde_json::json!({"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m", "content": [],
            "usage": {"input_tokens": 10, "output_tokens": 1}}})];
        frames.extend(text_block);
        frames.extend([closing.clone(), serde_json::json!({"type": "message_stop"})]);
        let completion = claude_completion(&frames).await;
        let [
            ContentBlock::Text { .. },
            ContentBlock::Unrecognized { parts },
        ] = completion.message.content.as_slice()
        else {
            panic!("{:?}", completion.message.content);
        };
        assert_eq!(parts, &vec![closing]);
        assert_eq!(completion.usage.input_tokens, 10);
        let fields = completion.response_fields.expect("the response's fields");
        assert_eq!(fields["content"], "late");
        assert_eq!(fields["usage"]["input_tokens"], 10);
        assert!(fields.get("stop_reason").is_none(), "{fields:?}");
    }

    /// A stream that ends inside a block says it failed before it hands the block over, so a retry
    /// the failure allows stays possible, and the block goes over as one that never finished: a
    /// call refused with the arguments that had streamed, reasoning unsigned.
    #[tokio::test]
    async fn a_stream_cut_inside_a_block_says_it_failed_first_and_leaves_the_block_unfinished() {
        let start = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",",
            "\"role\":\"assistant\",\"model\":\"m\",\"content\":[],",
            "\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
        );
        let kinds = |events: &[StreamEvent]| -> Vec<&'static str> {
            events
                .iter()
                .filter_map(|event| match event {
                    StreamEvent::Error(_) => Some("error"),
                    StreamEvent::ToolUseEnd { .. } => Some("end"),
                    StreamEvent::ToolCallRefused { .. } => Some("refused"),
                    StreamEvent::ThinkingComplete { opaque: None, .. } => Some("unsigned"),
                    StreamEvent::ThinkingComplete { .. } => Some("signed"),
                    _ => None,
                })
                .collect()
        };
        let (events, outcome) = decode_sse(&format!(
            "{start}{}",
            concat!(
                "event: content_block_start\n",
                "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":",
                "{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"file_read\",\"input\":{}}}\n\n",
                "event: content_block_delta\n",
                "data: {\"type\":\"content_block_delta\",\"index\":0,",
                "\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"pa\"}}\n\n",
            )
        ))
        .await;
        assert!(outcome.is_err(), "a stream that ends early fails");
        assert_eq!(kinds(&events), ["error", "refused"], "{events:?}");
        assert!(
            events.iter().any(|event| matches!(event,
                StreamEvent::ToolCallRefused { unparsed, .. }
                    if unparsed.text.as_deref() == Some("{\"pa")
                        && unparsed.error == crate::provider::unfinished_call(String::new()).error)),
            "{events:?}"
        );
        let (events, outcome) = decode_sse(&format!(
            "{start}{}",
            concat!(
                "event: content_block_start\n",
                "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":",
                "{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n",
                "event: content_block_delta\n",
                "data: {\"type\":\"content_block_delta\",\"index\":0,",
                "\"delta\":{\"type\":\"signature_delta\",\"signature\":\"S\"}}\n\n",
            )
        ))
        .await;
        assert!(outcome.is_err(), "a stream that ends early fails");
        assert_eq!(kinds(&events), ["error", "unsigned"], "{events:?}");
    }

    /// A stream that breaks off inside a block of a type Meka reads nothing from still keeps the
    /// block as it came; and a call without the id or name the spec
    /// requires is kept the same way and never run, rather than failing the reply.
    #[tokio::test]
    async fn a_block_kept_as_it_came_survives_a_cut_and_an_unnamed_call_is_one() {
        let (sender, mut receiver) = tokio::sync::mpsc::channel(64);
        let mut protocol = super::ClaudeStream::default();
        let frames = [
            serde_json::json!({"type": "message_start", "message": {"id": "msg_1",
                "type": "message", "role": "assistant", "model": "m", "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 1}}}),
            serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "server_tool_use", "id": "srv_1", "name": "web_search"}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "{\"query\""}}),
        ];
        for frame in &frames {
            let event = eventsource_stream::Event {
                event: frame["type"].as_str().expect("a type").to_string(),
                data: frame.to_string(),
                id: String::new(),
                retry: None,
            };
            crate::provider::sse::Protocol::frame(&mut protocol, event, &sender)
                .await
                .expect("frame");
        }
        assert!(protocol.close_open(&sender).await, "released");
        drop(sender);
        let mut accumulator = crate::provider::MessageAccumulator::new();
        while let Some(event) = receiver.recv().await {
            accumulator.push(event);
        }
        let partial = accumulator.partial().expect("what arrived");
        assert!(
            matches!(&partial.message.content[..], [ContentBlock::Unrecognized { parts }] if parts.len() == 2),
            "{:?}",
            partial.message.content
        );

        let completion = claude_completion(&[
            frames[0].clone(),
            serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "tool_use", "name": "file_read", "input": {}}}),
            serde_json::json!({"type": "content_block_stop", "index": 0}),
            serde_json::json!({"type": "message_delta",
                "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}}),
        ])
        .await;
        assert!(completion.message.tool_uses().is_empty());
        assert!(
            matches!(&completion.message.content[..], [
                ContentBlock::Unrecognized { .. }
            ]),
            "{:?}",
            completion.message.content
        );
    }

    /// A refusal's stop keeps what Claude said beside the word, the category and explanation of
    /// its `stop_details`, for a surface to show; the reply's words stay the model's alone.
    #[test]
    fn a_stop_keeps_what_claude_said_beside_its_word() {
        let stop = claude_stop(
            &serde_json::json!({
                "stop_reason": "refusal",
                "stop_sequence": null,
                "stop_details": {"type": "refusal", "category": "cyber", "explanation": "no"},
            }),
            "refusal",
        );
        assert_eq!(stop.reason, StopReason::Refusal);
        assert_eq!(stop.word.as_deref(), Some("refusal"));
        assert_eq!(
            stop.details,
            Some(serde_json::Map::from_iter([(
                "stop_details".to_string(),
                serde_json::json!({"type": "refusal", "category": "cyber", "explanation": "no"}),
            )]))
        );
    }

    #[test]
    fn parse_claude_stop_reason_all_variants() {
        assert_eq!(parse_claude_stop_reason("end_turn"), StopReason::EndTurn);
        assert_eq!(parse_claude_stop_reason("tool_use"), StopReason::ToolUse);
        assert_eq!(
            parse_claude_stop_reason("max_tokens"),
            StopReason::MaxTokens
        );
        assert_eq!(parse_claude_stop_reason("refusal"), StopReason::Refusal);
        assert_eq!(
            parse_claude_stop_reason("stop_sequence"),
            StopReason::EndTurn
        );
        assert_eq!(
            parse_claude_stop_reason("model_context_window_exceeded"),
            StopReason::ContextWindow
        );
        // A word Meka has no class for is `Unknown`; the word itself is kept in the stop record.
        assert_eq!(parse_claude_stop_reason("pause_turn"), StopReason::Unknown);
        assert_eq!(
            parse_claude_stop_reason("something_else"),
            StopReason::Unknown
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
            display: None,
        }
    }

    fn user_with_block(block: ContentBlock) -> Message {
        Message {
            role: Role::User,
            content: vec![block],
            provider_fields: None,
        }
    }

    fn assistant_text(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
                provider_fields: None,
            }],
            provider_fields: None,
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
                ToolResultContent::HarnessNote { note } => {
                    assert_eq!(note, crate::conversation::BUDGET_REDACTION_NOTE);
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
                ToolResultContent::HarnessNote { note } => {
                    assert_eq!(note, crate::conversation::BUDGET_REDACTION_NOTE);
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
                display: None,
            }],
            provider_fields: None,
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
                // function only cares about `.len() > crate::config::ANTHROPIC_MAX_REQUEST_BYTES`.
                Ok("X".repeat(crate::config::ANTHROPIC_MAX_REQUEST_BYTES + 1024))
            } else {
                Ok("{}".to_string())
            }
        };

        let (body, notice) =
            build_body_within_budget(&messages, crate::config::ANTHROPIC_MAX_REQUEST_BYTES, build)
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
            build_body_within_budget(&messages, crate::config::ANTHROPIC_MAX_REQUEST_BYTES, build)
                .expect("happy path");
        assert_eq!(body, "{}");
        assert!(notice.is_none());
    }
}
