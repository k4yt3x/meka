//! OpenAI Responses API encoder + SSE decoder.
//!
//! The protocol, not a backend: both [`super::responses`] (API key, any endpoint serving
//! `/v1/responses`) and [`super::subscription`] (ChatGPT's `/backend-api/codex/responses`) drive
//! this. It is the OpenAI counterpart of [`crate::provider::anthropic`]'s `shared`, and the reason
//! Chat Completions is a separate module rather than a flag: the two are different wire formats,
//! not two dialects of one.
//!
//! Nothing here may assume which endpoint is on the other end. `store: false` is set
//! unconditionally because meka replays the whole conversation every turn and never uses
//! server-side state, which also happens to be all the stateless implementations (Ollama, vLLM)
//! support. Anything that *is* endpoint-specific (the encrypted-reasoning `include`, auth
//! headers, the URL) belongs to the backend, not here.
//!
//! The on-the-wire request shape is documented at
//! <https://platform.openai.com/docs/guides/function-calling?api-mode=responses>. Verified against
//! the first-party Codex client:
//! - request shape: `codex-rs/codex-api/src/common.rs`
//! - input items:   `codex-rs/protocol/src/models.rs`
//! - SSE events:    `codex-rs/codex-api/src/sse/responses.rs`

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    conversation::{ContentBlock, Message, OpaqueReasoning, Role, ToolResultContent},
    error::{MekaError, Result},
    provider::{
        CompletionRequest, DeferredToolDelivery, LOAD_TOOL_NAME, SEARCH_TOOL_NAME, StopReason,
        StreamEvent, ToolDefinition, search_tool_parameters,
        sse::{End, Step},
    },
    stats::TokenUsage,
};

/// Build the JSON body POSTed to `/responses`. Translates the meka internal `Message` /
/// `ContentBlock` shape into Responses API `input` items (`message`, `function_call`,
/// `function_call_output`).
///
/// The request body for one completion, for a backend that may declare its deferred tools.
///
/// The result carries only what every Responses implementation understands, and replays sealed
/// reasoning sealed at `replays_reasoning_from` alone: reasoning from another endpoint is a blob
/// this one cannot open. A backend that knows more about its own endpoint adds OpenAI's own fields
/// afterwards; see [`include_encrypted_reasoning`].
///
/// Under [`DeferredToolDelivery::Searched`] the endpoint's own `tool_search` is declared with
/// client execution in place of meka's two discovery tools, `deferred_tools` ride with
/// `defer_loading`, a recorded `tool_search` call replays as the endpoint's `tool_search_call`
/// item, and its result as a `tool_search_output` carrying the definitions the result referenced,
/// since only that item registers a definition on this wire. Under the other deliveries nothing
/// changes but the deferred entries, which the agent hands over empty when the wire appends.
#[allow(
    clippy::too_many_arguments,
    reason = "one argument per request field, as the wrapper has"
)]
pub(super) fn build_request_body_for(
    delivery: DeferredToolDelivery,
    deferred_tools: &[ToolDefinition],
    model: &str,
    system_prompt: &str,
    messages: &[Message],
    tools: &[ToolDefinition],
    reasoning_effort: Option<&str>,
    max_output_tokens: Option<u64>,
    stream: bool,
    replays_reasoning_from: Option<&str>,
) -> serde_json::Value {
    let searched = searched_call_ids(delivery, messages);
    let mut input = Vec::with_capacity(messages.len());

    for message in messages {
        match message.role {
            Role::User => {
                encode_user_message(message, &mut input, &searched, tools, deferred_tools)
            }
            Role::Assistant => {
                encode_assistant_message(message, &mut input, &searched, replays_reasoning_from)
            }
        }
    }

    let mut body = serde_json::Map::from_iter([
        ("model".into(), serde_json::json!(model)),
        ("input".into(), serde_json::json!(input)),
        ("tool_choice".into(), serde_json::json!("auto")),
        ("store".into(), serde_json::json!(false)),
        ("stream".into(), serde_json::json!(stream)),
    ]);

    if !system_prompt.is_empty() {
        body.insert(
            "instructions".into(),
            serde_json::Value::String(system_prompt.to_string()),
        );
    }

    let declared = encode_tools(tools, deferred_tools, delivery, !searched.is_empty());
    if !declared.is_empty() {
        body.insert("tools".into(), serde_json::Value::Array(declared));
    }

    if let Some(effort) = reasoning_effort {
        body.insert("reasoning".into(), serde_json::json!({"effort": effort}));
    }

    if let Some(max_output) = max_output_tokens {
        body.insert("max_output_tokens".into(), serde_json::json!(max_output));
    }

    serde_json::Value::Object(body)
}

/// Ask the server to round-trip its reasoning as `reasoning.encrypted_content`.
///
/// Not in [`build_request_body_for`] because it is OpenAI's field rather than the protocol's:
/// `chatgpt-subscription` always sends it, as the first-party Codex client does, and
/// `openai-responses` sends it unless the profile's `encrypted_reasoning` is off. A model reasons
/// whether or not the request carries a `reasoning` object, so this asks regardless of one.
pub(super) fn include_encrypted_reasoning(body: &mut serde_json::Value) {
    body["include"] = serde_json::json!(["reasoning.encrypted_content"]);
}

/// Ask the server for a reasoning summary, the only part of its reasoning it will show a person.
///
/// Without this the model still reasons, but `summary` comes back empty and there is nothing for
/// the screen to render, so a long think looks like a hang. Codex asks for `auto` by default
/// (`ReasoningSummary::Auto`, and its `supports_reasoning_summary_parameter` defaults to true), so
/// this is what the first-party client's own users see.
///
/// Establishes the `reasoning` object when the profile configured no effort, as Codex does: it
/// sends `reasoning` on every request and omits only the fields it has no value for.
///
/// Not in [`build_request_body_for`] for the reason [`include_encrypted_reasoning`] is not:
/// `summary` is an OpenAI parameter, which `chatgpt-subscription` always sends and
/// `openai-responses` sends unless the profile's `reasoning_summary` is off.
///
/// Takes the body [`build_request_body_for`] returns, where `reasoning` is either an object or
/// absent; an absent one is created here, which is how the no-effort case gets its `reasoning`
/// block.
pub(super) fn request_reasoning_summary(body: &mut serde_json::Value) {
    if let Some(fields) = body.as_object_mut()
        && let Some(reasoning) = fields
            .entry("reasoning")
            .or_insert_with(|| serde_json::json!({}))
            .as_object_mut()
    {
        reasoning.insert("summary".into(), serde_json::json!("auto"));
    }
}

/// Build the `output` field of a `function_call_output` item from a slice of `ToolResultContent`.
/// The Responses API accepts either a plain string or an array of `input_text` / `input_image` /
/// `input_file` content items (per OpenAI's docs: "For functions that return images or files, you
/// can pass an array of image or file objects instead of a string."). The array form is emitted
/// when at least one image is present, to preserve it; otherwise the simpler string.
///
/// Sent unconditionally: a non-vision model returns a clear API error, which beats guessing at
/// model capabilities client-side. The Claude path also sends images without a model gate.
fn build_tool_result_output(content: &[ToolResultContent]) -> serde_json::Value {
    let has_image = content
        .iter()
        .any(|block| matches!(block, ToolResultContent::Image { .. }));

    if !has_image {
        return serde_json::Value::String(ContentBlock::tool_result_text_content(content));
    }

    let parts: Vec<serde_json::Value> = content
        .iter()
        .map(|block| match block {
            ToolResultContent::Text { text } => serde_json::json!({
                "type": "input_text",
                "text": text,
            }),
            ToolResultContent::Image { source } => input_image_part(source),
            ToolResultContent::ToolReference { tool_name } => serde_json::json!({
                "type": "input_text",
                "text": tool_name,
            }),
        })
        .collect();
    serde_json::Value::Array(parts)
}

/// Build a Responses API `input_image` content part from an image source. Shared by the tool-result
/// and user-message encoders.
fn input_image_part(source: &crate::image::ImageSource) -> serde_json::Value {
    match super::data_url(source) {
        Some(url) => serde_json::json!({
            "type": "input_image",
            "image_url": url,
            "detail": "auto",
        }),
        None => serde_json::json!({
            "type": "input_text",
            "text": crate::image::UNRESOLVED_IMAGE_PLACEHOLDER,
        }),
    }
}

/// The ids of the `tool_search` calls a conversation recorded, which replay as the endpoint's own
/// search item under [`DeferredToolDelivery::Searched`]; empty under every other delivery, where
/// the same name is meka's function tool.
fn searched_call_ids(
    delivery: DeferredToolDelivery,
    messages: &[Message],
) -> std::collections::HashSet<String> {
    if delivery != DeferredToolDelivery::Searched {
        return std::collections::HashSet::new();
    }
    messages
        .iter()
        .filter(|message| message.role == Role::Assistant)
        .flat_map(|message| message.content.iter())
        .filter_map(|block| match block {
            ContentBlock::ToolUse { id, name, .. } if name == SEARCH_TOOL_NAME => Some(id.clone()),
            _ => None,
        })
        .collect()
}

/// One function entry of the `tools` array; `deferred` marks it `defer_loading`, which withholds
/// the schema until a `tool_search_output` carries it.
fn function_declaration(tool: &ToolDefinition, deferred: bool) -> serde_json::Value {
    let mut declaration = serde_json::Map::from_iter([
        ("type".into(), serde_json::json!("function")),
        ("name".into(), serde_json::json!(tool.name)),
        ("description".into(), serde_json::json!(tool.description)),
        ("strict".into(), serde_json::json!(false)),
        ("parameters".into(), serde_json::json!(tool.parameters)),
    ]);
    if deferred {
        declaration.insert("defer_loading".into(), serde_json::Value::Bool(true));
    }
    serde_json::Value::Object(declaration)
}

/// The endpoint's own search item, declared with client execution so meka answers it. One door
/// for finding by keyword and loading by name, because only its output registers a definition on
/// this wire.
fn tool_search_declaration() -> serde_json::Value {
    serde_json::json!({
        "type": "tool_search",
        "execution": "client",
        "description": "Find deferred tools by keyword, or load them by exact name from the \
                        catalog. Every tool returned becomes callable.",
        "parameters": search_tool_parameters(DeferredToolDelivery::Searched),
    })
}

fn encode_user_message(
    message: &Message,
    input: &mut Vec<serde_json::Value>,
    searched: &std::collections::HashSet<String>,
    tools: &[ToolDefinition],
    deferred_tools: &[ToolDefinition],
) {
    let mut text_parts: Vec<&str> = Vec::new();
    let mut image_parts: Vec<serde_json::Value> = Vec::new();

    for block in &message.content {
        match block {
            // The context block, a nudge and what arrived for the model are text on this wire
            // too, in the order the message holds them.
            ContentBlock::Text { text }
            | ContentBlock::TurnContext { text }
            | ContentBlock::Nudge { text, .. }
            | ContentBlock::Inbox { text, .. }
            | ContentBlock::Scheduled { text, .. }
            | ContentBlock::BackgroundReport { text, .. }
            | ContentBlock::Invocation { text, .. }
            | ContentBlock::Resource { text, .. } => text_parts.push(text),
            // Responses takes `input_image` content parts on the user message. No model gate;
            // non-vision models return a clear error.
            ContentBlock::Image { source } => image_parts.push(input_image_part(source)),
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                if searched.contains(tool_use_id) {
                    // The definitions the result referenced, looked up in what this request
                    // declares; a name no longer registered (a server withdrew it) is left out.
                    let definitions: Vec<serde_json::Value> = content
                        .iter()
                        .filter_map(|item| match item {
                            ToolResultContent::ToolReference { tool_name } => tools
                                .iter()
                                .chain(deferred_tools)
                                .find(|tool| &tool.name == tool_name)
                                .map(|tool| function_declaration(tool, false)),
                            _ => None,
                        })
                        .collect();
                    input.push(serde_json::json!({
                        "type": "tool_search_output",
                        "call_id": tool_use_id,
                        "execution": "client",
                        "status": "completed",
                        "tools": definitions,
                    }));
                    // The output item carries definitions and nothing else, so what the search
                    // said in words, a match list or why nothing was loaded, follows as the
                    // message's text.
                    text_parts.extend(content.iter().filter_map(|item| match item {
                        ToolResultContent::Text { text } => Some(text.as_str()),
                        _ => None,
                    }));
                } else {
                    input.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": tool_use_id,
                        "output": build_tool_result_output(content),
                    }));
                }
            }
            // ToolUse / Thinking on a user message would be malformed; ignore defensively to match
            // the Chat Completions encoder's behavior.
            _ => {}
        }
    }

    // One part per block, in order: the context block ahead of the words, as the other wires send
    // them, rather than the two joined into one part.
    let mut content_parts: Vec<serde_json::Value> = text_parts
        .iter()
        .map(|text| serde_json::json!({"type": "input_text", "text": text}))
        .collect();
    content_parts.extend(image_parts);
    if !content_parts.is_empty() {
        input.push(serde_json::json!({
            "type": "message",
            "role": "user",
            "content": content_parts,
        }));
    }
}

/// Encode one assistant turn as `input` items, in the order its blocks arrived.
///
/// Order is load-bearing here in a way it is not for the user encoder. A `reasoning` item must be
/// followed by the output that reasoning produced, or the API rejects the whole request: `Item
/// '<id>' of type 'reasoning' was provided without its required following item`. Emitting all the
/// text first and then all the calls breaks that pairing as soon as a turn thinks twice.
fn encode_assistant_message(
    message: &Message,
    input: &mut Vec<serde_json::Value>,
    searched: &std::collections::HashSet<String>,
    replays_reasoning_from: Option<&str>,
) {
    let mut pending_text = String::new();

    for (index, block) in message.content.iter().enumerate() {
        match block {
            ContentBlock::Text { text } => pending_text.push_str(text),
            // Replayed only when it carries `encrypted_content`, which is the reasoning itself and
            // the only part the model can resume from, and only to the endpoint that sealed it. A
            // summary without it is a digest written for a human, so sending one back buys
            // nothing and would put an item shape on the wire for endpoints that were never asked
            // for reasoning in the first place.
            ContentBlock::Thinking {
                thinking,
                opaque:
                    Some(OpaqueReasoning::Sealed {
                        encrypted_content,
                        id,
                        origin: Some(origin),
                    }),
            } if replays_reasoning_from == Some(origin.as_str())
                && an_emitted_item_follows(&message.content, index) =>
            {
                flush_assistant_text(&mut pending_text, input);
                input.push(reasoning_input_item(
                    id.as_deref(),
                    thinking,
                    encrypted_content,
                ));
            }
            ContentBlock::ToolUse {
                id,
                name,
                input: arguments,
            } => {
                flush_assistant_text(&mut pending_text, input);
                if searched.contains(id) {
                    input.push(serde_json::json!({
                        "type": "tool_search_call",
                        "call_id": id,
                        "arguments": arguments,
                        "execution": "client",
                        "status": "completed",
                    }));
                } else {
                    input.push(serde_json::json!({
                        "type": "function_call",
                        "name": name,
                        "call_id": id,
                        "arguments": arguments.to_string(),
                    }));
                }
            }
            _ => {}
        }
    }

    flush_assistant_text(&mut pending_text, input);
}

/// Emit the assistant text accumulated so far as one `message` item, and clear it.
fn flush_assistant_text(pending: &mut String, input: &mut Vec<serde_json::Value>) {
    if pending.is_empty() {
        return;
    }
    input.push(serde_json::json!({
        "type": "message",
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": std::mem::take(pending),
        }],
    }));
}

/// Whether anything this encoder will actually emit follows `index` in the same turn.
///
/// A reasoning item left dangling is a rejected request, and meka's own history reaches that shape
/// without any help from the model: [`Message::without_tool_use`] drops an interrupted turn's calls
/// but keeps its thinking, and a rewind can cut anywhere.
///
/// Only output counts, so a run of reasoning items is carried by whatever follows the run rather
/// than by each other, which is the order the server produced them in, and so the order they may
/// be replayed in. What this refuses is a turn that ends in reasoning with no output at all.
fn an_emitted_item_follows(content: &[ContentBlock], index: usize) -> bool {
    content.iter().skip(index + 1).any(|block| match block {
        ContentBlock::Text { text } => !text.is_empty(),
        ContentBlock::ToolUse { .. } => true,
        _ => false,
    })
}

/// A `reasoning` item, shaped as the server sent it so it can be replayed verbatim.
///
/// meka flattens a multi-part summary into one string as it streams (the parts are display
/// sections, separated by a blank line), and does not try to reconstruct the original parts here:
/// splitting back on the separator would invent boundaries wherever a single part contained a
/// paragraph break. The whole text goes back as one part, which preserves what was said.
fn reasoning_input_item(
    id: Option<&str>,
    summary: &str,
    encrypted_content: &str,
) -> serde_json::Value {
    let summary = if summary.is_empty() {
        serde_json::json!([])
    } else {
        serde_json::json!([{ "type": "summary_text", "text": summary }])
    };
    let mut item = serde_json::Map::from_iter([
        ("type".into(), serde_json::json!("reasoning")),
        ("summary".into(), serde_json::json!(summary)),
        (
            "encrypted_content".into(),
            serde_json::json!(encrypted_content),
        ),
    ]);
    if let Some(id) = id {
        item.insert("id".into(), serde_json::Value::String(id.to_string()));
    }
    serde_json::Value::Object(item)
}

fn encode_tools(
    tools: &[ToolDefinition],
    deferred_tools: &[ToolDefinition],
    delivery: DeferredToolDelivery,
    replays_a_search: bool,
) -> Vec<serde_json::Value> {
    let searched = delivery == DeferredToolDelivery::Searched;
    let mut declared = Vec::with_capacity(tools.len() + deferred_tools.len() + 1);
    // The endpoint's search item rides while there is something for it to find, or while the
    // history replays a search it answered; with neither it is a tool the model can only call
    // for nothing, as meka's own discovery tools would be.
    if searched && (!deferred_tools.is_empty() || replays_a_search) {
        declared.push(tool_search_declaration());
    }
    declared.extend(
        tools
            .iter()
            // The endpoint's search item does both jobs here; meka's two function tools would
            // be a second door whose results register nothing.
            .filter(|tool| {
                !(searched && (tool.name == LOAD_TOOL_NAME || tool.name == SEARCH_TOOL_NAME))
            })
            .map(|tool| function_declaration(tool, false)),
    );
    declared.extend(
        deferred_tools
            .iter()
            .map(|tool| function_declaration(tool, true)),
    );
    declared
}

/// Mutable state threaded through SSE event processing. Tracks the in-flight tool call's
/// accumulated arguments so a parsed `ToolUseEnd` can be returned even if the server elides the
/// final `arguments` field.
#[derive(Default)]
pub(super) struct SseState {
    active_tool_call: Option<ActiveToolCall>,
    in_reasoning: bool,
    /// Whether this response emitted any `function_call` output item. The Responses API reports
    /// `status: "completed"` even for tool-call turns, so the stop reason has to be inferred from
    /// the presence of function calls rather than the status (mirrors the Chat Completions
    /// `finish_reason: "tool_calls"` mapping).
    emitted_tool_call: bool,
    /// Once `response.completed` (or `response.failed` / `response.incomplete`) has been
    /// processed, the driver should stop pulling new events.
    pub(super) finished: bool,
    /// The origin of the endpoint answering, which every sealed reasoning block records.
    pub(super) origin: Option<String>,
}

struct ActiveToolCall {
    arguments_buffer: String,
}

/// Which Responses frame this is: the payload's `type`, falling back to the SSE `event:` line.
///
/// The payload field is the spec's discriminator and is always present; the `event:` line is an
/// optional convenience only some servers set. ChatGPT and OpenAI send both, but an API-key
/// backend can point anywhere, and OpenRouter streams bare `data:` frames: read from the event
/// name alone, every one of its frames looks unhandled and a perfectly good turn dies with "stream
/// ended before a terminal response event". The fallback keeps working for any server that names
/// the event but omits `type`.
fn frame_name<'a>(data: &'a serde_json::Value, event_name: &'a str) -> &'a str {
    data.get("type")
        .and_then(|value| value.as_str())
        .unwrap_or(event_name)
}

/// Pure SSE-event handler. Inspects the named event and parsed JSON payload, updates `state`, and
/// returns the meka-level [`StreamEvent`]s to forward to the agent. Returns `Err` when the server
/// reports a fatal stream error; the driver propagates this back to the caller.
pub(super) fn process_event(
    event_name: &str,
    data: &serde_json::Value,
    state: &mut SseState,
) -> Result<Vec<StreamEvent>> {
    let mut out = Vec::new();
    match event_name {
        "response.created" | "response.in_progress" => {}

        "response.output_text.delta" => {
            if let Some(delta) = data.get("delta").and_then(|v| v.as_str())
                && !delta.is_empty()
            {
                out.push(StreamEvent::TextDelta(delta.to_string()));
            }
        }

        "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
            if let Some(delta) = data.get("delta").and_then(|v| v.as_str())
                && !delta.is_empty()
            {
                state.in_reasoning = true;
                out.push(StreamEvent::ThinkingDelta(delta.to_string()));
            }
        }

        // A reasoning summary arrives as several parts, each its own section with its own heading,
        // and the deltas carry no separator of their own. Codex renders a break here
        // (`on_reasoning_section_break`); meka's thinking is flat text, so the break is a blank
        // line. The first part opens the block rather than separating anything, which is what
        // `in_reasoning` distinguishes.
        "response.reasoning_summary_part.added" => {
            if state.in_reasoning {
                out.push(StreamEvent::ThinkingDelta("\n\n".to_string()));
            }
        }

        "response.output_item.added" => {
            let Some(item) = data.get("item") else {
                return Ok(out);
            };
            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if item_type == "function_call" || item_type == "tool_search_call" {
                let id = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                // The endpoint's search item has no name of its own; it dispatches as meka's
                // search tool, which answers both a query and a list of names.
                let name = if item_type == "tool_search_call" {
                    SEARCH_TOOL_NAME.to_string()
                } else {
                    item.get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string()
                };
                state.active_tool_call = Some(ActiveToolCall {
                    arguments_buffer: String::new(),
                });
                state.emitted_tool_call = true;
                out.push(StreamEvent::ToolUseStart { id, name });
            }
        }

        "response.function_call_arguments.delta" | "response.custom_tool_call_input.delta" => {
            if let Some(delta) = data.get("delta").and_then(|v| v.as_str())
                && !delta.is_empty()
                && let Some(tool) = state.active_tool_call.as_mut()
            {
                tool.arguments_buffer.push_str(delta);
            }
        }

        "response.output_item.done" => {
            let Some(item) = data.get("item") else {
                return Ok(out);
            };
            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if item_type == "tool_search_call" {
                // Arrives whole: the arguments are an object on the done item, with no deltas
                // before it.
                state.active_tool_call.take();
                let call_id = item
                    .get("call_id")
                    .or_else(|| item.get("id"))
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_string();
                let arguments = match item.get("arguments") {
                    Some(serde_json::Value::String(raw)) => raw.clone(),
                    Some(value) if !value.is_null() => value.to_string(),
                    _ => "{}".to_string(),
                };
                out.push(crate::provider::tool_use_event(
                    call_id,
                    SEARCH_TOOL_NAME.to_string(),
                    &arguments,
                ));
            } else if item_type == "function_call" {
                let buffered = state.active_tool_call.take();
                // Read off the completed item: `ActiveToolCall` carries only the argument buffer,
                // and the item is the authoritative copy of the identity anyway.
                let call_id = item
                    .get("call_id")
                    .or_else(|| item.get("id"))
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_string();
                let call_name = item
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_string();
                // The final `arguments` string from the item wins over the accumulated buffer;
                // the server may normalize it.
                let arguments_str = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| buffered.map(|tool| tool.arguments_buffer))
                    .unwrap_or_default();
                out.push(crate::provider::tool_use_event(
                    call_id,
                    call_name,
                    &arguments_str,
                ));
            } else if item_type == "reasoning" {
                // True when this item streamed anything readable, by either spelling: a requested
                // summary, or the raw reasoning text a local server emits unprompted.
                let showed_its_reasoning = state.in_reasoning;
                state.in_reasoning = false;
                let encrypted_content = item
                    .get("encrypted_content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                // Deliberately not gated on that. `encrypted_content` is what the next request
                // replays, and it arrives whether or not summaries were requested, so gating its
                // capture on visible text would silently throw the reasoning chain away for
                // exactly the configuration that asked for it. An item with neither is worth
                // nothing to either the reader or the next turn, so it makes no block.
                if showed_its_reasoning || encrypted_content.is_some() {
                    let id = item
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    out.push(StreamEvent::ThinkingComplete {
                        opaque: encrypted_content.map(|encrypted_content| {
                            OpaqueReasoning::Sealed {
                                encrypted_content,
                                id,
                                origin: state.origin.clone(),
                            }
                        }),
                    });
                }
            }
        }

        "response.completed" => {
            state.finished = true;
            if let Some(response) = data.get("response") {
                if let Some(usage) = response.get("usage") {
                    out.push(StreamEvent::Usage(usage_from(usage)));
                }
                // The Responses API reports `status: "completed"` even when the output is function
                // calls, so a tool-call turn must be surfaced as `ToolUse` regardless of status;
                // otherwise the agent warns and mislabels the turn as `EndTurn`.
                let stop_reason = if state.emitted_tool_call {
                    StopReason::ToolUse
                } else {
                    response
                        .get("status")
                        .and_then(|v| v.as_str())
                        .map(parse_response_status)
                        .unwrap_or(StopReason::EndTurn)
                };
                out.push(StreamEvent::MessageEnd { stop_reason });
            } else if state.emitted_tool_call {
                out.push(StreamEvent::MessageEnd {
                    stop_reason: StopReason::ToolUse,
                });
            } else {
                out.push(StreamEvent::MessageEnd {
                    stop_reason: StopReason::EndTurn,
                });
            }
        }

        "response.failed" => {
            state.finished = true;
            // Sending `StreamEvent::Error` is handled by the caller (`drive_responses_sse_stream`),
            // which has channel access.
            let error_object = data
                .get("response")
                .and_then(|response| response.get("error"))
                .unwrap_or(&serde_json::Value::Null);
            return Err(crate::error::provider_stream_error_object(
                error_object,
                "response.failed event",
            ));
        }

        // The stream's own top-level error frame, distinct from `response.failed`: the request
        // died rather than the response. Unhandled, it would fall into the catch-all and the turn
        // be reported as a stream that ended early, retried twice against an error that repeats.
        "error" => {
            state.finished = true;
            // The frame's fields sit at its top level on the public endpoint and under `error` on
            // the ChatGPT backend; the fallback label says nothing about the cause, so the frame
            // itself goes to the log.
            tracing::debug!("Responses error frame: {data}");
            let fields = data
                .get("error")
                .filter(|value| value.is_object())
                .unwrap_or(data);
            return Err(crate::error::provider_stream_error_object(
                fields,
                "error event",
            ));
        }

        "response.incomplete" => {
            state.finished = true;
            let response = data.get("response");
            let reason = response
                .and_then(|response| response.get("incomplete_details"))
                .and_then(|details| details.get("reason"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            // The output cap is a stop reason, not a failure: the two sibling protocols report it
            // as `MaxTokens` and commit the message, and the agent labels the turn so the user
            // sees a truncated answer rather than an error. Reported as an error, a
            // `max_output_tokens` profile loses every answer that reaches it, whole on the blocking
            // path. Anything else (a content filter, an unknown reason) is deterministic on the
            // request and ends the turn; never retryable.
            if reason == "max_output_tokens" {
                if let Some(usage) = response.and_then(|response| response.get("usage")) {
                    out.push(StreamEvent::Usage(usage_from(usage)));
                }
                out.push(StreamEvent::MessageEnd {
                    stop_reason: StopReason::MaxTokens,
                });
                return Ok(out);
            }
            let message = format!("response.incomplete: {reason}");
            return Err(MekaError::Provider(message));
        }

        other => {
            tracing::debug!("unhandled Responses SSE event: {other}");
        }
    }
    Ok(out)
}

/// The `usage` object of a terminal response event, as the agent counts it.
fn usage_from(usage: &serde_json::Value) -> TokenUsage {
    // Without the per-item `attribution` a ChatGPT response adds, which runs to kilobytes per
    // request; the totals are the diagnostic.
    if tracing::enabled!(tracing::Level::DEBUG)
        && let Some(object) = usage.as_object()
    {
        let totals: serde_json::Map<String, serde_json::Value> = object
            .iter()
            .filter(|(key, _)| *key != "attribution")
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        tracing::debug!(
            "responses usage: {totals}",
            totals = serde_json::Value::Object(totals)
        );
    }
    super::parse_usage(
        usage,
        "input_tokens",
        "input_tokens_details",
        "output_tokens",
    )
}

fn parse_response_status(status: &str) -> StopReason {
    match status {
        "completed" => StopReason::EndTurn,
        "incomplete" => StopReason::MaxTokens,
        other => {
            tracing::warn!(
                "the Responses endpoint returned an unrecognized response status {other:?}; mapping to Unknown"
            );
            StopReason::Unknown(other.to_string())
        }
    }
}

/// What the two Responses providers differ in, so one driver serves both: the endpoint, how a
/// request is authenticated, and whether a rejected credential can be refreshed for one more try.
#[async_trait::async_trait]
pub(super) trait ResponsesBackend: crate::oauth::RefreshesCredential + Send + Sync {
    /// The HTTP client every request goes through.
    fn client(&self) -> &reqwest::Client;
    /// The URL a completion is posted to.
    fn endpoint(&self) -> String;
    /// The origin of [`Self::endpoint`], which the reasoning it seals records; see
    /// [`OpaqueReasoning::Sealed`].
    fn origin(&self) -> Option<String> {
        crate::provider::endpoint_origin(&self.endpoint())
    }
    /// The request body for one completion, before serialization.
    fn request_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        deferred_tools: &[ToolDefinition],
        attribution: &crate::provider::Attribution,
    ) -> serde_json::Value;
    /// The profile's request ceiling, when it states one; see [`crate::provider::budget`]. `None`
    /// sends the body as built.
    fn max_request_bytes(&self) -> Option<usize>;
    /// One attempt's authenticated request. Called again after a rejected credential, so a backend
    /// that can refresh one does it here.
    async fn authenticated_request(
        &self,
        request: reqwest::RequestBuilder,
        attribution: &crate::provider::Attribution,
    ) -> Result<reqwest::RequestBuilder>;
}

/// A non-streaming Responses call is the streaming one aggregated: the API is stream-first and the
/// two providers never needed a second wire shape.
pub(super) async fn complete<B: ResponsesBackend>(
    backend: &B,
    request: CompletionRequest<'_>,
    cancellation: CancellationToken,
) -> Result<crate::provider::Completion> {
    let (event_sender, event_receiver) = mpsc::channel::<StreamEvent>(1024);
    let (stream_result, aggregated) = tokio::join!(
        stream(backend, request, event_sender, cancellation),
        crate::provider::aggregate_stream(event_receiver),
    );
    stream_result?;
    Ok(aggregated)
}

/// One streaming Responses call: one send, one retry on a credential the backend could refresh,
/// and the login remedy on a second rejection.
pub(super) async fn stream<B: ResponsesBackend>(
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
    let (body_json, redaction_notice) = body_within_budget(
        backend,
        system_prompt,
        messages,
        tools,
        deferred_tools,
        &attribution,
    )?;
    // A send error here means the consumer hung up already, which the SSE driver reports itself.
    if let Some(notice) = redaction_notice
        && event_sender
            .send(StreamEvent::Notice(notice))
            .await
            .is_err()
    {
        tracing::trace!("stream event receiver dropped");
    }

    let response = crate::oauth::send_with_one_refresh(
        backend,
        crate::error::ProviderRequest::Completion,
        |error| {
            crate::error::provider_transport_error("Responses HTTP request failed", error, None)
        },
        || async {
            Ok(backend
                .authenticated_request(backend.client().post(backend.endpoint()), &attribution)
                .await?
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body_json.clone()))
        },
        &cancellation,
    )
    .await?;
    drive_responses_sse_stream(response, backend.origin(), event_sender, cancellation).await
}

/// The serialized request, within the profile's ceiling when it states one.
pub(super) fn body_within_budget<B: ResponsesBackend + ?Sized>(
    backend: &B,
    system_prompt: &str,
    messages: &[Message],
    tools: &[ToolDefinition],
    deferred_tools: &[ToolDefinition],
    attribution: &crate::provider::Attribution,
) -> Result<(String, Option<crate::frontend::Notice>)> {
    match backend.max_request_bytes() {
        Some(max_request_bytes) => crate::provider::budget::fit_body_to_budget(
            messages,
            max_request_bytes,
            None,
            |messages| {
                crate::provider::budget::serialize_body(&backend.request_body(
                    system_prompt,
                    messages,
                    tools,
                    deferred_tools,
                    attribution,
                ))
            },
        ),
        None => Ok((
            crate::provider::budget::serialize_body(&backend.request_body(
                system_prompt,
                messages,
                tools,
                deferred_tools,
                attribution,
            ))?,
            None,
        )),
    }
}

/// Drive the SSE stream for a Responses API call. Pulls events off the transport, runs them through
/// [`process_event`], and forwards the resulting [`StreamEvent`]s to the agent.
pub(super) async fn drive_responses_sse_stream(
    response: reqwest::Response,
    origin: Option<String>,
    event_sender: mpsc::Sender<StreamEvent>,
    cancellation: CancellationToken,
) -> Result<()> {
    let mut protocol = ResponsesStream {
        state: SseState {
            origin,
            ..SseState::default()
        },
    };
    match crate::provider::sse::drive(
        response,
        "Responses",
        &event_sender,
        &cancellation,
        &mut protocol,
    )
    .await?
    {
        End::Finished | End::ReceiverGone => Ok(()),
        // The stream ended without `response.completed`, `response.failed` or
        // `response.incomplete`. Falling through here would commit a truncated turn as a complete
        // one: the agent would see whatever text had arrived, write it to the conversation, and
        // move on, with the retry path never consulted. A connection cut mid-response is exactly
        // what that path exists for.
        End::Ended => Err(crate::provider::sse::stream_error(
            &event_sender,
            "the Responses stream ended before a terminal response event".to_string(),
        )
        .await),
    }
}

/// The Responses driver's state between frames: [`process_event`]'s accumulator.
struct ResponsesStream {
    state: SseState,
}

#[async_trait::async_trait]
impl crate::provider::sse::Protocol for ResponsesStream {
    async fn frame(
        &mut self,
        event: eventsource_stream::Event,
        event_sender: &mpsc::Sender<StreamEvent>,
    ) -> Result<Step> {
        let Some(data) = crate::provider::sse::frame_json("Responses", &event.data) else {
            return Ok(Step::Continue);
        };
        let events = process_event(frame_name(&data, &event.event), &data, &mut self.state)?;
        for emit in events {
            if event_sender.send(emit).await.is_err() {
                tracing::trace!("stream event receiver dropped");
                return Ok(Step::ReceiverGone);
            }
        }
        Ok(if self.state.finished {
            Step::Finished
        } else {
            Step::Continue
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::ToolResultContent;

    /// The origin every sealed block these tests write records, as ChatGPT's would.
    const CHATGPT_ORIGIN: &str = "https://chatgpt.com";

    /// An appending request with nothing deferred, which is what most of these tests build.
    fn build_request_body(
        model: &str,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        reasoning_effort: Option<&str>,
        max_output_tokens: Option<u64>,
        stream: bool,
    ) -> serde_json::Value {
        build_request_body_for(
            DeferredToolDelivery::Appended,
            &[],
            model,
            system_prompt,
            messages,
            tools,
            reasoning_effort,
            max_output_tokens,
            stream,
            Some(CHATGPT_ORIGIN),
        )
    }

    /// An invocation and an attached resource are text on this wire, in the order the message
    /// holds them, so the model reads the skill's body and the file among the words.
    #[test]
    fn an_invocation_and_a_resource_are_read_in_place() {
        let body = build_request_body(
            "gpt-5",
            "",
            &[Message::invoked_with_a_resource_for_test()],
            &[],
            None,
            None,
            true,
        );
        let sent = body["input"].to_string();
        let at = |needle: &str| {
            sent.find(needle)
                .unwrap_or_else(|| panic!("{needle} is not sent: {sent}"))
        };
        assert!(at("the skill's body") < at("compare "), "{sent}");
        assert!(at("compare ") < at("fn a() {}"), "{sent}");
        assert!(at("fn a() {}") < at(" with the old one"), "{sent}");
    }

    #[test]
    fn a_minimal_request_body_states_only_the_defaults() {
        let body = build_request_body("gpt-5", "", &[Message::user("hi")], &[], None, None, true);
        assert_eq!(body["model"], "gpt-5");
        assert_eq!(body["stream"], true);
        assert_eq!(body["tool_choice"], "auto");
        assert!(body.get("parallel_tool_calls").is_none());
        assert_eq!(body["store"], false);
        assert!(body.get("instructions").is_none());
        assert!(body.get("tools").is_none());
        assert!(body.get("reasoning").is_none());
    }

    /// The context block and the words are two `input_text` parts in order, the context first.
    #[test]
    fn the_context_block_precedes_the_words() {
        let body = build_request_body(
            "gpt-5",
            "",
            &[Message::user_turn("ctx", "hello", Vec::new())],
            &[],
            None,
            None,
            true,
        );
        let parts = body["input"][0]["content"]
            .as_array()
            .expect("content parts");
        assert_eq!(parts.len(), 2, "{parts:?}");
        assert_eq!(
            parts[0],
            serde_json::json!({"type": "input_text", "text": "ctx"})
        );
        assert_eq!(
            parts[1],
            serde_json::json!({"type": "input_text", "text": "hello"})
        );
    }

    #[test]
    fn request_body_includes_instructions_when_system_prompt_set() {
        let body = build_request_body(
            "gpt-5",
            "be helpful",
            &[Message::user("hi")],
            &[],
            None,
            None,
            true,
        );
        assert_eq!(body["instructions"], "be helpful");
    }

    #[test]
    fn request_body_user_message_uses_input_text() {
        let body = build_request_body(
            "gpt-5",
            "",
            &[Message::user("hello")],
            &[],
            None,
            None,
            true,
        );
        let input = body["input"].as_array().expect("input array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][0]["text"], "hello");
    }

    #[test]
    fn request_body_assistant_text_uses_output_text() {
        let messages = vec![
            Message::user("a"),
            Message::assistant_text("b"),
            Message::user("c"),
        ];
        let body = build_request_body("gpt-5", "", &messages, &[], None, None, true);
        let input = body["input"].as_array().expect("input array");
        assert_eq!(input.len(), 3);
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[1]["content"][0]["type"], "output_text");
        assert_eq!(input[1]["content"][0]["text"], "b");
    }

    /// An inbox item read at a round boundary is text beside the round's tool results: every
    /// `function_call_output` first, then one user item, in that order, or the API complains that
    /// a call has no output.
    #[test]
    fn text_beside_tool_results_becomes_a_user_item_after_the_outputs() {
        let messages = vec![
            Message::user("read /tmp/x"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_abc".to_string(),
                    name: "file_read".to_string(),
                    input: serde_json::json!({"path": "/tmp/x"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![
                    ContentBlock::ToolResult {
                        tool_use_id: "call_abc".to_string(),
                        content: vec![ToolResultContent::Text {
                            text: "contents".to_string(),
                        }],
                        is_error: false,
                    },
                    ContentBlock::Text {
                        text: "[Message from test, arrived now]\nalso, what is 17*3?".to_string(),
                    },
                ],
            },
        ];

        let body = build_request_body("gpt-5", "", &messages, &[], None, None, true);
        let input = body["input"].as_array().expect("input array");

        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call_abc");
        assert_eq!(input[3]["type"], "message");
        assert_eq!(input[3]["role"], "user");
        assert_eq!(
            input[3]["content"][0]["text"],
            "[Message from test, arrived now]\nalso, what is 17*3?"
        );
        assert_eq!(input.len(), 4);
    }

    #[test]
    fn request_body_tool_use_emits_function_call_item() {
        let messages = vec![
            Message::user("read /tmp/x"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_abc".to_string(),
                    name: "file_read".to_string(),
                    input: serde_json::json!({"path": "/tmp/x"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_abc".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "contents".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];

        let body = build_request_body("gpt-5", "", &messages, &[], None, None, true);
        let input = body["input"].as_array().expect("input array");

        // [0] user message, [1] function_call, [2] function_call_output
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["name"], "file_read");
        assert_eq!(input[1]["call_id"], "call_abc");
        // arguments must be a JSON string, not a parsed object
        assert!(input[1]["arguments"].is_string());

        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call_abc");
        assert_eq!(input[2]["output"], "contents");
    }

    #[test]
    fn request_body_tools_use_responses_api_flat_shape() {
        let tools = vec![ToolDefinition::new(
            "demo",
            "A demo tool",
            serde_json::json!({"type": "object", "properties": {}}),
        )];
        let body = build_request_body("gpt-5", "", &[], &tools, None, None, true);
        let tools_arr = body["tools"].as_array().expect("tools");
        assert_eq!(tools_arr[0]["type"], "function");
        // Top-level `name` / `description` / `parameters` (NOT wrapped under a `function` object
        // like Chat Completions). This is the Responses API shape.
        assert_eq!(tools_arr[0]["name"], "demo");
        assert_eq!(tools_arr[0]["description"], "A demo tool");
        assert!(tools_arr[0].get("parameters").is_some());
        assert!(tools_arr[0].get("function").is_none());
    }

    /// The shared body carries only what every Responses implementation understands.
    ///
    /// `include` is an OpenAI extension, so it is not the protocol's to add: it was moved out of
    /// here when `openai-responses` arrived, because that backend reaches servers where an
    /// unrecognized field is a rejected request. The subscription backend opts in explicitly, and
    /// its half of this split is asserted alongside.
    #[test]
    fn the_shared_body_asks_for_reasoning_but_never_for_an_openai_extension() {
        let mut body = build_request_body(
            "gpt-5",
            "",
            &[Message::user("think hard")],
            &[],
            Some("high"),
            None,
            true,
        );
        assert_eq!(body["reasoning"]["effort"], "high");
        assert!(body.get("include").is_none(), "{body}");

        // The subscription backend adds it, so reasoning survives a stateless round trip against
        // ChatGPT, whose support for it is a fact rather than a guess.
        include_encrypted_reasoning(&mut body);
        let include = body["include"].as_array().expect("include");
        assert!(
            include
                .iter()
                .any(|value| value == "reasoning.encrypted_content")
        );
    }

    #[test]
    fn request_body_user_image_emits_input_image() {
        let message =
            Message::user_with_images("describe", vec![crate::image::ImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: "QUJD".to_string(),
            }]);
        let body = build_request_body("gpt-5", "", &[message], &[], None, None, true);
        let input = body["input"].as_array().expect("input array");
        let content = input[0]["content"].as_array().expect("content array");
        assert_eq!(content[0]["type"], "input_text");
        assert_eq!(content[0]["text"], "describe");
        assert_eq!(content[1]["type"], "input_image");
        assert_eq!(content[1]["image_url"], "data:image/png;base64,QUJD");
    }

    #[test]
    fn request_body_sets_max_output_tokens_when_overridden() {
        let body = build_request_body(
            "gpt-5",
            "",
            &[Message::user("hi")],
            &[],
            None,
            Some(40_000),
            true,
        );
        assert_eq!(body["max_output_tokens"], 40_000);
    }

    #[test]
    fn request_body_omits_max_output_tokens_when_unset() {
        let body = build_request_body("gpt-5", "", &[Message::user("hi")], &[], None, None, true);
        assert!(body.get("max_output_tokens").is_none());
    }

    #[test]
    fn request_body_omits_reasoning_when_effort_unset() {
        let body = build_request_body("gpt-5", "", &[Message::user("hi")], &[], None, None, true);
        assert!(body.get("reasoning").is_none());
        assert!(body.get("include").is_none());
    }

    #[test]
    fn request_body_user_message_with_tool_result_only_no_text_block() {
        // A user turn that's *only* a tool_result must produce only a function_call_output input
        // item, no empty user message.
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: vec![ToolResultContent::Text {
                    text: "result".to_string(),
                }],
                is_error: false,
            }],
        }];
        let body = build_request_body("gpt-5", "", &messages, &[], None, None, true);
        let input = body["input"].as_array().expect("input array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "function_call_output");
    }

    fn run_events(
        events: &[(&str, serde_json::Value)],
    ) -> (Vec<StreamEvent>, std::result::Result<(), MekaError>) {
        let mut state = SseState::default();
        let mut emitted = Vec::new();
        let mut outcome = Ok(());
        for (name, data) in events {
            match process_event(name, data, &mut state) {
                Ok(events) => emitted.extend(events),
                Err(error) => {
                    // process_event still yields a StreamEvent::Error before
                    // returning. Draining it via re-running with a fresh state
                    // would be wrong; instead we capture both outcomes by
                    // running once and then preserving the events that were
                    // emitted before the error. For the error path the caller
                    // populates `out` *and* returns Err, so the events the
                    // caller would forward are already in `out` for the call
                    // that errored. Re-process with a side-channel:
                    if let Some(message) = error.to_string().strip_prefix("Provider error: ") {
                        emitted.push(StreamEvent::Error(message.to_string()));
                    }
                    outcome = Err(error);
                    break;
                }
            }
            if state.finished {
                break;
            }
        }
        (emitted, outcome)
    }

    /// A frame is named by its payload, not by the SSE `event:` line.
    ///
    /// Found against OpenRouter, which sends bare `data:` frames. That is valid SSE and the
    /// Responses spec's own discriminator is the payload's `type`; the `event:` line is an optional
    /// convenience. Keying off the event name alone worked only while ChatGPT and OpenAI -- which
    /// send both -- were the only endpoints meka reached, and failed the moment an API-key backend
    /// could point anywhere. The symptom was total: every frame looked unhandled, so a turn that
    /// streamed perfectly well died with "stream ended before a terminal response event".
    ///
    /// Both orderings are asserted, because a fix that read only the payload would break the
    /// endpoints that name the event and send a payload without one.
    #[test]
    fn a_frame_is_identified_by_its_payload_type_not_the_sse_event_name() {
        // How OpenRouter streams: bare `data:`, no event name, type in the payload.
        let bare = serde_json::json!({"type": "response.output_text.delta", "delta": "hi"});
        assert_eq!(frame_name(&bare, ""), "response.output_text.delta");

        // How ChatGPT streams: the event named and the payload agreeing.
        assert_eq!(
            frame_name(&bare, "response.output_text.delta"),
            "response.output_text.delta"
        );

        // A payload with no `type` still falls back to the event name, so an endpoint that names
        // the event and omits the field keeps working.
        let untyped = serde_json::json!({"delta": "hi"});
        assert_eq!(
            frame_name(&untyped, "response.output_text.delta"),
            "response.output_text.delta"
        );

        // And the name that comes out actually drives the decoder.
        let (emitted, _) =
            run_events(&[(frame_name(&bare, ""), serde_json::json!({"delta": "hi"}))]);
        assert!(
            matches!(emitted.as_slice(), [StreamEvent::TextDelta(text)] if text == "hi"),
            "{emitted:?}"
        );
    }

    #[test]
    fn process_event_text_delta() {
        let mut state = SseState::default();
        let events = process_event(
            "response.output_text.delta",
            &serde_json::json!({"delta": "hello"}),
            &mut state,
        )
        .expect("ok");
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::TextDelta(ref t) if t == "hello"));
    }

    #[test]
    fn process_event_text_delta_empty_emits_nothing() {
        let mut state = SseState::default();
        let events = process_event(
            "response.output_text.delta",
            &serde_json::json!({"delta": ""}),
            &mut state,
        )
        .expect("ok");
        assert!(events.is_empty());
    }

    #[test]
    fn process_event_reasoning_delta_emits_thinking() {
        let mut state = SseState::default();
        let events = process_event(
            "response.reasoning_text.delta",
            &serde_json::json!({"delta": "hmm"}),
            &mut state,
        )
        .expect("ok");
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::ThinkingDelta(ref t) if t == "hmm"));
        assert!(state.in_reasoning);
    }

    #[test]
    fn process_event_tool_call_full_lifecycle() {
        let (events, outcome) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({
                    "item": {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "file_read"
                    }
                }),
            ),
            (
                "response.function_call_arguments.delta",
                serde_json::json!({"delta": "{\"path\":"}),
            ),
            (
                "response.function_call_arguments.delta",
                serde_json::json!({"delta": "\"/tmp/x\"}"}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({
                    "item": {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "file_read",
                        "arguments": "{\"path\":\"/tmp/x\"}"
                    }
                }),
            ),
            (
                "response.completed",
                serde_json::json!({
                    "response": {"id": "r1", "status": "completed"}
                }),
            ),
        ]);
        outcome.expect("clean stream");
        assert!(matches!(
            events[0],
            StreamEvent::ToolUseStart { ref id, ref name } if id == "c1" && name == "file_read"
        ));
        match &events[1] {
            StreamEvent::ToolUseEnd { input } => assert_eq!(input["path"], "/tmp/x"),
            other => panic!("expected ToolUseEnd, got {other:?}"),
        }
        // The Responses API reports `status: "completed"` even for tool-call turns; the presence of
        // the function call must still surface as `ToolUse`, not `EndTurn`.
        assert!(matches!(events[2], StreamEvent::MessageEnd {
            stop_reason: StopReason::ToolUse
        }));
    }

    /// The Responses stream closes one item before it opens the next, so one argument buffer is
    /// enough: each call must reach the agent as its own start and end pair, in stream order.
    #[test]
    fn a_response_with_two_function_calls_yields_both_tool_uses_in_order() {
        let (events, outcome) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({
                    "output_index": 0,
                    "item": {"type": "function_call", "call_id": "c1", "name": "file_read"}
                }),
            ),
            (
                "response.function_call_arguments.delta",
                serde_json::json!({"output_index": 0, "delta": "{\"path\":\"/tmp/a\"}"}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "file_read",
                        "arguments": "{\"path\":\"/tmp/a\"}"
                    }
                }),
            ),
            (
                "response.output_item.added",
                serde_json::json!({
                    "output_index": 1,
                    "item": {"type": "function_call", "call_id": "c2", "name": "file_read"}
                }),
            ),
            (
                "response.function_call_arguments.delta",
                serde_json::json!({"output_index": 1, "delta": "{\"path\":\"/tmp/b\"}"}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "call_id": "c2",
                        "name": "file_read",
                        "arguments": "{\"path\":\"/tmp/b\"}"
                    }
                }),
            ),
            (
                "response.completed",
                serde_json::json!({
                    "response": {"id": "r1", "status": "completed"}
                }),
            ),
        ]);
        outcome.expect("clean stream");
        let calls: Vec<(&str, &str)> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ToolUseStart { id, name } => Some((id.as_str(), name.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(calls, vec![("c1", "file_read"), ("c2", "file_read")]);
        let paths: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ToolUseEnd { input } => input["path"].as_str(),
                _ => None,
            })
            .collect();
        assert_eq!(paths, vec!["/tmp/a", "/tmp/b"]);
        assert!(matches!(
            events.last(),
            Some(StreamEvent::MessageEnd {
                stop_reason: StopReason::ToolUse
            })
        ));
    }

    /// `run_events` drives `process_event` directly, so it can never reach the fall-through at the
    /// bottom of `drive_responses_sse_stream` -- the code that decides what a stream ending with
    /// no terminal event means. Drive the real loop instead, the way the Claude decoder's
    /// counterpart test does.
    async fn decode_sse(body: &str) -> (Vec<StreamEvent>, Result<()>) {
        let response: reqwest::Response = http::Response::builder()
            .status(200)
            .header("content-type", "text/event-stream")
            .body(body.to_string())
            .expect("build response")
            .into();
        let (sender, mut receiver) = mpsc::channel(64);
        let outcome = drive_responses_sse_stream(
            response,
            Some(CHATGPT_ORIGIN.to_string()),
            sender,
            CancellationToken::new(),
        )
        .await;
        let mut events = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            events.push(event);
        }
        (events, outcome)
    }

    /// A stream of bare `data:` frames decodes, end to end, through the real driver.
    ///
    /// This is how OpenRouter streams: valid SSE with no `event:` line, the frame named only by the
    /// payload's `type`. It has to be asserted here rather than against [`frame_name`] alone,
    /// because the bug was the *call site* -- the helper can be perfect while the driver still
    /// passes it the wrong string. Every other case in this harness carries an `event:` line and a
    /// payload without a `type`, so they exercise only the fallback branch and all stayed green
    /// while every OpenRouter turn died with "stream ended before a terminal response event".
    #[tokio::test]
    async fn a_stream_of_bare_data_frames_decodes_end_to_end() {
        let (events, outcome) = decode_sse(concat!(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"PO\"}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"NG\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\
             \"usage\":{\"input_tokens\":1,\"output_tokens\":2}}}\n\n",
        ))
        .await;
        assert!(outcome.is_ok(), "{outcome:?}");
        let text: String = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::TextDelta(delta) => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "PONG", "{events:?}");
        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::MessageEnd { .. })),
            "the terminal frame must be recognized too: {events:?}"
        );
    }

    /// A connection cut partway through the answer never sends `response.completed`. Committing
    /// that as a finished turn hands the agent half a response and never consults the retry path,
    /// which is the one case that path exists for.
    #[tokio::test]
    async fn a_stream_cut_before_its_terminal_event_is_an_error() {
        let (events, outcome) = decode_sse(concat!(
            "event: response.output_text.delta\n",
            "data: {\"delta\":\"half an ans\"}\n\n",
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

    /// The other half of the same boundary: a stream that did reach `response.completed` is
    /// finished, and must not be turned into a spurious retry by the check above.
    #[tokio::test]
    async fn a_stream_that_reaches_its_terminal_event_is_complete() {
        let (_events, outcome) = decode_sse(concat!(
            "event: response.output_text.delta\n",
            "data: {\"delta\":\"a whole answer\"}\n\n",
            "event: response.completed\n",
            "data: {\"response\": {\"id\": \"r1\", \"status\": \"completed\"}}\n\n",
        ))
        .await;

        outcome.expect("a stream with a terminal event is complete");
    }

    /// The Responses half of the same boundary the Anthropic decoder has: arguments that do not
    /// parse are the model's intent, mangled, and running the tool with `{}` executes something
    /// it never asked for while reporting success.
    #[test]
    fn a_tool_call_with_unparseable_arguments_is_rejected_not_run_empty() {
        let (events, outcome) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({
                    "item": {"type": "function_call", "call_id": "c1", "name": "file_write"}
                }),
            ),
            (
                "response.function_call_arguments.delta",
                serde_json::json!({"delta": "{\"path\": "}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({
                    "item": {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "file_write",
                        "arguments": "{\"path\": "
                    }
                }),
            ),
        ]);
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

    #[test]
    fn process_event_completed_without_tool_call_is_end_turn() {
        let (events, outcome) = run_events(&[
            (
                "response.output_text.delta",
                serde_json::json!({"delta": "final answer"}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"id": "r1", "status": "completed"}}),
            ),
        ]);
        outcome.expect("clean stream");
        assert!(matches!(
            events.last(),
            Some(StreamEvent::MessageEnd {
                stop_reason: StopReason::EndTurn
            })
        ));
    }

    #[test]
    fn process_event_tool_call_recovers_arguments_from_done_only() {
        // Server elides per-delta events and sends arguments only on `done`.
        let (events, outcome) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({
                    "item": {"type": "function_call", "call_id": "c1", "name": "x"}
                }),
            ),
            (
                "response.output_item.done",
                serde_json::json!({
                    "item": {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "x",
                        "arguments": "{\"k\":1}"
                    }
                }),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"id": "r1", "status": "completed"}}),
            ),
        ]);
        outcome.expect("clean stream");
        let input = events
            .iter()
            .find_map(|event| match event {
                StreamEvent::ToolUseEnd { input } => Some(input),
                _ => None,
            })
            .expect("ToolUseEnd present");
        assert_eq!(input["k"], 1);
    }

    #[test]
    fn process_event_completed_emits_token_usage() {
        let mut state = SseState::default();
        let events = process_event(
            "response.completed",
            &serde_json::json!({
                "response": {
                    "id": "r1",
                    "status": "completed",
                    "usage": {"input_tokens": 42, "output_tokens": 7}
                }
            }),
            &mut state,
        )
        .expect("ok");
        let usage = events
            .iter()
            .find_map(|event| match event {
                StreamEvent::Usage(usage) => Some(usage),
                _ => None,
            })
            .expect("Usage event");
        assert_eq!(usage.input_tokens, 42);
        assert_eq!(usage.output_tokens, 7);
        assert!(state.finished);
    }

    #[test]
    fn process_event_failed_yields_error_and_propagates() {
        let mut state = SseState::default();
        let result = process_event(
            "response.failed",
            &serde_json::json!({
                "response": {"error": {"message": "too many tokens"}}
            }),
            &mut state,
        );
        assert!(state.finished);
        assert!(
            matches!(result, Err(MekaError::Provider(ref message)) if message.contains("too many tokens"))
        );
    }

    #[test]
    fn process_event_failed_with_server_error_code_is_retryable() {
        let mut state = SseState::default();
        let result = process_event(
            "response.failed",
            &serde_json::json!({
                "response": {"error": {"code": "server_error", "message": "internal error"}}
            }),
            &mut state,
        );
        assert!(matches!(result, Err(MekaError::RetryableProvider { .. })));
    }

    /// The stream's own error frame names its cause whether the fields sit at the top level or
    /// under `error`, and a numeric or server code in either place is retried.
    #[test]
    fn an_error_frame_names_its_cause_from_either_shape() {
        let mut state = SseState::default();
        let flat = process_event(
            "error",
            &serde_json::json!({"type": "error", "code": "invalid_request", "message": "flat cause"}),
            &mut state,
        );
        assert!(state.finished);
        assert!(
            matches!(flat, Err(MekaError::Provider(ref message)) if message == "flat cause"),
            "{flat:?}"
        );
        let mut state = SseState::default();
        // The ChatGPT backend's overload frame as captured on 2026-10-16: nested, and transient.
        let nested = process_event(
            "error",
            &serde_json::json!({"type": "error", "sequence_number": 3, "error": {
                "message": "Our servers are currently overloaded. Please try again later.",
                "type": "service_unavailable_error", "param": null, "code": "server_is_overloaded",
            }}),
            &mut state,
        );
        assert!(
            matches!(nested, Err(MekaError::RetryableProvider { ref message, .. }) if message.starts_with("Our servers")),
            "{nested:?}"
        );
    }

    #[test]
    fn process_event_failed_without_code_stays_permanent() {
        // No `code`/`type` field at all: the default is not-retryable.
        let mut state = SseState::default();
        let result = process_event(
            "response.failed",
            &serde_json::json!({
                "response": {"error": {"message": "bad request"}}
            }),
            &mut state,
        );
        assert!(matches!(result, Err(MekaError::Provider(_))));
    }

    /// The stream's top-level `error` frame is a failure with a reason, not an early end.
    #[test]
    fn the_streams_own_error_frame_ends_the_turn_with_its_reason() {
        let mut state = SseState::default();
        let result = process_event(
            "error",
            &serde_json::json!({"type": "error", "code": "server_error", "message": "try later"}),
            &mut state,
        );
        assert!(state.finished);
        assert!(
            matches!(result, Err(MekaError::RetryableProvider { ref message, .. }) if message.contains("try later")),
            "{result:?}"
        );
    }

    /// The output cap ends the message as `MaxTokens`, with its usage, the way the Claude and Chat
    /// Completions drivers report `max_tokens` and `length`; reported as an error, the whole
    /// answer was lost on the blocking path.
    /// The Responses driver applies the profile's ceiling the same way, for both of its backends.
    #[test]
    fn a_stated_ceiling_redacts_old_images_on_responses() {
        let with_ceiling = |max_request_bytes: Option<usize>| {
            let api_key: String = "test-key".to_string();
            crate::provider::openai::responses::OpenAiResponsesProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiResponses,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-5".to_string(),
                )
                .base_url(None)
                .effort(None)
                .max_output_tokens(None)
                .max_request_bytes(max_request_bytes),
            )
            .expect("build test provider")
        };
        let image = "A".repeat(8_000);
        let messages = vec![
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call-1".to_string(),
                    content: vec![ToolResultContent::Image {
                        source: crate::image::ImageSource::Base64 {
                            media_type: "image/png".to_string(),
                            data: image.clone(),
                        },
                    }],
                    is_error: false,
                }],
            },
            Message::user("and now this"),
        ];

        let (body, notice) = body_within_budget(
            &with_ceiling(Some(4_000)),
            "",
            &messages,
            &[],
            &[],
            &Default::default(),
        )
        .expect("fits after redaction");
        assert!(
            body.contains(crate::provider::budget::IMAGE_REDACTION_PLACEHOLDER),
            "the old image is redacted: {body}"
        );
        assert!(notice.is_some(), "the redaction is announced");

        let (body, notice) = body_within_budget(
            &with_ceiling(None),
            "",
            &messages,
            &[],
            &[],
            &Default::default(),
        )
        .expect("no ceiling, no refusal");
        assert!(body.contains(&image), "sent as built without a ceiling");
        assert!(notice.is_none());
    }

    #[test]
    fn process_event_incomplete_for_the_output_cap_ends_as_max_tokens() {
        let mut state = SseState::default();
        let events = process_event(
            "response.incomplete",
            &serde_json::json!({
                "response": {
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "usage": {"input_tokens": 12, "output_tokens": 512}
                }
            }),
            &mut state,
        )
        .expect("a stop reason, not a failure");
        assert!(state.finished);
        assert!(
            matches!(
                events.as_slice(),
                [
                    StreamEvent::Usage(usage),
                    StreamEvent::MessageEnd {
                        stop_reason: StopReason::MaxTokens
                    }
                ] if usage.output_tokens == 512
            ),
            "{events:?}"
        );
    }

    /// Any other reason is deterministic on the request and fails the turn.
    #[test]
    fn process_event_incomplete_for_a_content_filter_yields_error() {
        let mut state = SseState::default();
        let result = process_event(
            "response.incomplete",
            &serde_json::json!({
                "response": {"incomplete_details": {"reason": "content_filter"}}
            }),
            &mut state,
        );
        assert!(state.finished);
        assert!(matches!(
            result,
            Err(MekaError::Provider(ref message)) if message.contains("content_filter")
        ));
    }

    #[test]
    fn process_event_status_incomplete_maps_to_max_tokens() {
        let mut state = SseState::default();
        let events = process_event(
            "response.completed",
            &serde_json::json!({
                "response": {"id": "r1", "status": "incomplete"}
            }),
            &mut state,
        )
        .expect("ok");
        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::MessageEnd {
                    stop_reason: StopReason::MaxTokens
                }))
        );
    }

    #[test]
    fn process_event_unknown_event_silently_skipped() {
        let mut state = SseState::default();
        let events = process_event(
            "response.output_audio_transcript.delta",
            &serde_json::json!({"delta": "audio"}),
            &mut state,
        )
        .expect("ok");
        assert!(events.is_empty());
        assert!(!state.finished);
    }

    #[test]
    fn process_event_reasoning_done_emits_thinking_complete_with_signature() {
        let mut state = SseState {
            in_reasoning: true,
            ..SseState::default()
        };
        let events = process_event(
            "response.output_item.done",
            &serde_json::json!({
                "item": {
                    "type": "reasoning",
                    "id": "rs_123",
                    "summary": [],
                    "encrypted_content": "OPAQUE"
                }
            }),
            &mut state,
        )
        .expect("ok");
        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::ThinkingComplete {
                opaque:
                    Some(OpaqueReasoning::Sealed {
                        encrypted_content,
                        id,
                        ..
                    }),
            } => {
                assert_eq!(encrypted_content, "OPAQUE");
                assert_eq!(id.as_deref(), Some("rs_123"));
            }
            other => panic!("expected ThinkingComplete, got {other:?}"),
        }
        assert!(!state.in_reasoning);
    }

    /// The defect this whole path had: `encrypted_content` arrives whether or not a summary was
    /// asked for, and gating its capture on having seen a summary delta threw the reasoning chain
    /// away for every request that did not also ask for summaries.
    #[test]
    fn encrypted_reasoning_is_captured_without_a_summary_to_show_for_it() {
        let mut state = SseState::default();
        let events = process_event(
            "response.output_item.done",
            &serde_json::json!({
                "item": {
                    "type": "reasoning",
                    "id": "rs_silent",
                    "summary": [],
                    "encrypted_content": "OPAQUE"
                }
            }),
            &mut state,
        )
        .expect("ok");
        assert!(
            matches!(
                events.as_slice(),
                [StreamEvent::ThinkingComplete {
                    opaque: Some(OpaqueReasoning::Sealed {
                        encrypted_content,
                        id,
                        ..
                    }),
                }] if id.as_deref() == Some("rs_silent") && encrypted_content == "OPAQUE"
            ),
            "silent reasoning must still be captured, got {events:?}"
        );
    }

    /// A reasoning item with neither summary nor encrypted content has nothing to render and
    /// nothing to replay, so it should not manufacture a block.
    #[test]
    fn an_empty_reasoning_item_produces_no_event() {
        let mut state = SseState::default();
        let events = process_event(
            "response.output_item.done",
            &serde_json::json!({ "item": { "type": "reasoning", "summary": [] } }),
            &mut state,
        )
        .expect("ok");
        assert!(events.is_empty(), "got {events:?}");
    }

    /// Each summary part is its own section. Without a break between them the parts run together
    /// into one paragraph, and the first part must not be preceded by one.
    #[test]
    fn summary_parts_are_separated_but_the_first_one_is_not() {
        let mut state = SseState::default();
        let opening = process_event(
            "response.reasoning_summary_part.added",
            &serde_json::json!({ "summary_index": 0 }),
            &mut state,
        )
        .expect("ok");
        assert!(opening.is_empty(), "got {opening:?}");

        process_event(
            "response.reasoning_summary_text.delta",
            &serde_json::json!({ "delta": "**First**\nthinking" }),
            &mut state,
        )
        .expect("ok");

        let between = process_event(
            "response.reasoning_summary_part.added",
            &serde_json::json!({ "summary_index": 1 }),
            &mut state,
        )
        .expect("ok");
        assert!(
            matches!(between.as_slice(), [StreamEvent::ThinkingDelta(text)] if text == "\n\n"),
            "got {between:?}"
        );
    }

    fn image_content(media_type: &str, data: &str) -> ToolResultContent {
        ToolResultContent::Image {
            source: crate::image::ImageSource::Base64 {
                media_type: media_type.to_string(),
                data: data.to_string(),
            },
        }
    }

    #[test]
    fn build_tool_result_output_text_only_returns_string() {
        let content = vec![ToolResultContent::Text {
            text: "result".to_string(),
        }];
        let out = build_tool_result_output(&content);
        assert_eq!(out, serde_json::Value::String("result".to_string()));
    }

    #[test]
    fn build_tool_result_output_with_image_returns_array() {
        let content = vec![
            ToolResultContent::Text {
                text: "before".to_string(),
            },
            image_content("image/png", "AAAA"),
            ToolResultContent::Text {
                text: "after".to_string(),
            },
        ];
        let out = build_tool_result_output(&content);
        let array = out.as_array().expect("should be array when image present");
        assert_eq!(array.len(), 3);
        assert_eq!(array[0]["type"], "input_text");
        assert_eq!(array[0]["text"], "before");
        assert_eq!(array[1]["type"], "input_image");
        assert_eq!(array[1]["image_url"], "data:image/png;base64,AAAA");
        assert_eq!(array[1]["detail"], "auto");
        assert_eq!(array[2]["type"], "input_text");
        assert_eq!(array[2]["text"], "after");
    }

    #[test]
    fn build_tool_result_output_image_only_returns_array() {
        let content = vec![image_content("image/jpeg", "DEAD")];
        let out = build_tool_result_output(&content);
        let array = out.as_array().expect("should be array");
        assert_eq!(array.len(), 1);
        assert_eq!(array[0]["type"], "input_image");
        assert_eq!(array[0]["image_url"], "data:image/jpeg;base64,DEAD");
    }

    #[test]
    fn function_call_output_carries_image_array_in_request_body() {
        // End-to-end: build_request_body wires build_tool_result_output via encode_user_message;
        // confirm the function_call_output's `output` field is the array form when an image is
        // present.
        let mut messages = vec![Message::user("look at this"), Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "screenshot".to_string(),
                input: serde_json::json!({}),
            }],
        }];
        messages.push(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: vec![image_content("image/png", "QkFTRTY0")],
                is_error: false,
            }],
        });
        let body = build_request_body("gpt-5", "", &messages, &[], None, None, true);
        let input = body["input"].as_array().expect("input array");
        let output_item = input
            .iter()
            .find(|item| item["type"] == "function_call_output")
            .expect("function_call_output present");
        let output = output_item["output"]
            .as_array()
            .expect("output should be array (image present)");
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "input_image");
        assert_eq!(output[0]["image_url"], "data:image/png;base64,QkFTRTY0");
    }

    fn thinking(summary: &str, encrypted: Option<&str>, id: Option<&str>) -> ContentBlock {
        ContentBlock::Thinking {
            thinking: summary.to_string(),
            opaque: encrypted.map(|encrypted_content| OpaqueReasoning::Sealed {
                encrypted_content: encrypted_content.to_string(),
                id: id.map(str::to_string),
                origin: Some("https://chatgpt.com".to_string()),
            }),
        }
    }

    fn tool_call(id: &str) -> ContentBlock {
        ContentBlock::ToolUse {
            id: id.to_string(),
            name: "file_read".to_string(),
            input: serde_json::json!({}),
        }
    }

    fn definition(name: &str) -> ToolDefinition {
        ToolDefinition::new(
            name,
            format!("{name} does a thing"),
            serde_json::json!({"type": "object"}),
        )
    }

    /// On a searched wire the endpoint's own search item is declared with client execution in
    /// place of meka's two discovery tools, and every deferred tool rides with `defer_loading`.
    /// A referenced wire declares the deferred tools and keeps the discovery tools; an appending
    /// wire sends the array as built.
    #[test]
    fn a_searched_wire_declares_the_endpoint_s_search_item_and_the_deferred_tools() {
        let tools = [
            definition(LOAD_TOOL_NAME),
            definition(SEARCH_TOOL_NAME),
            definition("file_read"),
        ];
        let deferred = [definition("mcp__kv__probe")];
        let declared = |delivery: DeferredToolDelivery, deferred: &[ToolDefinition]| {
            let body = build_request_body_for(
                delivery,
                deferred,
                "gpt-5",
                "",
                &[Message::user("hi")],
                &tools,
                None,
                None,
                true,
                Some(CHATGPT_ORIGIN),
            );
            body["tools"].as_array().cloned().unwrap_or_default()
        };

        let searched = declared(DeferredToolDelivery::Searched, &deferred);
        assert_eq!(searched[0]["type"], "tool_search", "{searched:?}");
        assert_eq!(searched[0]["execution"], "client");
        assert!(
            searched[0]["parameters"]["properties"]
                .get("names")
                .is_some(),
            "the item is the one door that loads here: {searched:?}"
        );
        let functions: Vec<(&str, bool)> = searched[1..]
            .iter()
            .map(|tool| {
                (
                    tool["name"].as_str().unwrap_or(""),
                    tool["defer_loading"] == true,
                )
            })
            .collect();
        assert_eq!(functions, vec![
            ("file_read", false),
            ("mcp__kv__probe", true)
        ]);

        let referenced = declared(DeferredToolDelivery::Referenced, &deferred);
        let names: Vec<&str> = referenced
            .iter()
            .map(|tool| tool["name"].as_str().unwrap_or(""))
            .collect();
        assert_eq!(names, vec![
            LOAD_TOOL_NAME,
            SEARCH_TOOL_NAME,
            "file_read",
            "mcp__kv__probe"
        ]);
        assert!(referenced.iter().all(|tool| tool["type"] == "function"));
        assert_eq!(referenced[3]["defer_loading"], true);
        assert!(referenced[2].get("defer_loading").is_none());

        let appended = declared(DeferredToolDelivery::Appended, &[]);
        assert_eq!(appended.len(), 3);
        assert!(
            appended
                .iter()
                .all(|tool| tool.get("defer_loading").is_none())
        );

        // Nothing deferred and no search in the history: the item leaves with the discovery
        // functions, so a compaction's request or a session with every tool eager declares none.
        let bare = declared(DeferredToolDelivery::Searched, &[]);
        assert_eq!(bare.len(), 1, "{bare:?}");
        assert_eq!(bare[0]["name"], "file_read");
    }

    /// A history that replays a search keeps the endpoint's item declared even when nothing is
    /// deferred any longer, since the replayed call names it.
    #[test]
    fn a_replayed_search_keeps_the_search_item_with_nothing_deferred() {
        let history = [
            Message::user("hi"),
            assistant(vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: SEARCH_TOOL_NAME.to_string(),
                input: serde_json::json!({"names": ["mcp__kv__probe"]}),
            }]),
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: vec![ToolResultContent::ToolReference {
                        tool_name: "mcp__kv__probe".to_string(),
                    }],
                    is_error: false,
                }],
            },
        ];
        let body = build_request_body_for(
            DeferredToolDelivery::Searched,
            &[],
            "gpt-5",
            "",
            &history,
            &[definition("file_read")],
            None,
            None,
            true,
            Some(CHATGPT_ORIGIN),
        );
        let tools = body["tools"].as_array().cloned().unwrap_or_default();
        assert_eq!(tools[0]["type"], "tool_search", "{tools:?}");
        assert_eq!(body["input"][2]["type"], "tool_search_output");
        assert_eq!(
            body["input"][2]["tools"].as_array().map(Vec::len),
            Some(0),
            "nothing declared to deliver"
        );
    }

    /// A recorded `tool_search` call replays as the endpoint's own items on a searched wire, its
    /// result as a `tool_search_output` carrying the definitions it referenced; on any other
    /// delivery the same history is a function call whose output reads the reference as text.
    #[test]
    fn a_recorded_search_replays_as_the_endpoint_s_items_on_a_searched_wire() {
        let history = [
            Message::user("hi"),
            assistant(vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: SEARCH_TOOL_NAME.to_string(),
                input: serde_json::json!({"names": ["mcp__kv__probe"]}),
            }]),
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: vec![
                        ToolResultContent::Text {
                            text: "loaded".to_string(),
                        },
                        ToolResultContent::ToolReference {
                            tool_name: "mcp__kv__probe".to_string(),
                        },
                    ],
                    is_error: false,
                }],
            },
        ];
        let deferred = [definition("mcp__kv__probe")];
        let input = |delivery: DeferredToolDelivery| {
            build_request_body_for(
                delivery,
                &deferred,
                "gpt-5",
                "",
                &history,
                &[],
                None,
                None,
                true,
                Some(CHATGPT_ORIGIN),
            )["input"]
                .as_array()
                .cloned()
                .expect("input array")
        };

        let searched = input(DeferredToolDelivery::Searched);
        assert_eq!(searched[1]["type"], "tool_search_call", "{searched:?}");
        assert_eq!(searched[1]["call_id"], "call_1");
        assert_eq!(searched[1]["arguments"]["names"][0], "mcp__kv__probe");
        assert_eq!(searched[1]["execution"], "client");
        assert_eq!(searched[2]["type"], "tool_search_output");
        assert_eq!(searched[2]["call_id"], "call_1");
        assert_eq!(searched[2]["tools"][0]["name"], "mcp__kv__probe");
        assert!(searched[2]["tools"][0].get("defer_loading").is_none());
        // The words of the result follow the item, which has no place for them.
        assert_eq!(searched[3]["type"], "message", "{searched:?}");
        assert_eq!(searched[3]["content"][0]["text"], "loaded");
        assert_eq!(searched.len(), 4);

        let appended = input(DeferredToolDelivery::Appended);
        assert_eq!(appended[1]["type"], "function_call", "{appended:?}");
        assert_eq!(appended[2]["type"], "function_call_output");
        let output = appended[2]["output"].as_str().expect("text output");
        assert!(
            output.contains("loaded") && output.contains("mcp__kv__probe"),
            "{output}"
        );
    }

    /// The endpoint's search item arrives whole on its done frame and dispatches as meka's search
    /// tool, which answers both a query and a list of names.
    #[test]
    fn a_tool_search_call_item_dispatches_as_the_search_tool() {
        let (events, outcome) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"item": {"type": "tool_search_call", "call_id": "call_1", "execution": "client", "arguments": {}, "status": "in_progress"}}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"item": {"type": "tool_search_call", "call_id": "call_1", "execution": "client", "arguments": {"names": ["mcp__kv__probe"]}, "status": "completed"}}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"id": "r1", "status": "completed"}}),
            ),
        ]);
        outcome.expect("clean stream");
        assert!(
            matches!(
                events.as_slice(),
                [
                    StreamEvent::ToolUseStart { id, name },
                    StreamEvent::ToolUseEnd { input },
                    StreamEvent::MessageEnd { stop_reason: StopReason::ToolUse },
                ] if id == "call_1" && name == SEARCH_TOOL_NAME && input["names"][0] == "mcp__kv__probe"
            ),
            "{events:?}"
        );
    }

    fn assistant(content: Vec<ContentBlock>) -> Message {
        Message {
            role: Role::Assistant,
            content,
        }
    }

    fn input_of(messages: &[Message]) -> Vec<serde_json::Value> {
        build_request_body("gpt-5", "", messages, &[], Some("high"), None, true)["input"]
            .as_array()
            .expect("input array")
            .clone()
    }

    /// The whole point of the `include`: reasoning meka asked the server to encrypt has to go back
    /// out on the next request, or the model restarts its chain of thought at every tool call.
    #[test]
    fn encrypted_reasoning_is_replayed_on_the_next_request() {
        let input = input_of(&[
            Message::user("hi"),
            assistant(vec![
                thinking("weighing it up", Some("OPAQUE"), Some("rs_1")),
                tool_call("call_1"),
            ]),
        ]);

        let reasoning = input
            .iter()
            .find(|item| item["type"] == "reasoning")
            .expect("the reasoning item must survive the round trip");
        assert_eq!(reasoning["id"], "rs_1");
        assert_eq!(reasoning["encrypted_content"], "OPAQUE");
        assert_eq!(reasoning["summary"][0]["type"], "summary_text");
        assert_eq!(reasoning["summary"][0]["text"], "weighing it up");
    }

    /// The API pairs a reasoning item with the output it produced and rejects one left dangling
    /// ("was provided without its required following item"). `without_tool_use` reaches that shape
    /// on any interrupted turn, so the encoder has to refuse it rather than the server.
    #[test]
    fn reasoning_with_nothing_after_it_is_not_sent() {
        let interrupted = assistant(vec![
            thinking("weighing it up", Some("OPAQUE"), Some("rs_1")),
            tool_call("call_1"),
        ])
        .without_tool_use();
        let input = input_of(&[Message::user("hi"), interrupted]);

        assert!(
            !input.iter().any(|item| item["type"] == "reasoning"),
            "a dangling reasoning item would be rejected: {input:?}"
        );
    }

    /// What the API wants after a reasoning item is the output that reasoning produced. Another
    /// reasoning item is not that, so a turn cut down to nothing but thinking sends none of it --
    /// otherwise the last one still dangles and takes the whole request down with it.
    #[test]
    fn reasoning_does_not_count_as_another_reasoning_item_s_follower() {
        let input = input_of(&[
            Message::user("hi"),
            assistant(vec![
                thinking("first", Some("ONE"), Some("rs_1")),
                thinking("second", Some("TWO"), Some("rs_2")),
            ]),
        ]);

        assert!(
            !input.iter().any(|item| item["type"] == "reasoning"),
            "got {input:?}"
        );
    }

    /// Reasoning has to precede the output it produced, so the encoder cannot emit every text block
    /// first and every call after: a turn that thought twice would come back in an order the API
    /// rejects.
    #[test]
    fn reasoning_precedes_the_output_it_produced_across_two_thinking_rounds() {
        let input = input_of(&[
            Message::user("hi"),
            assistant(vec![
                ContentBlock::Text {
                    text: "let me look".to_string(),
                },
                thinking("first", Some("ONE"), Some("rs_1")),
                tool_call("call_1"),
                thinking("second", Some("TWO"), Some("rs_2")),
                ContentBlock::Text {
                    text: "done".to_string(),
                },
            ]),
        ]);

        let shape: Vec<&str> = input
            .iter()
            .filter_map(|item| item["type"].as_str())
            .collect();
        assert_eq!(shape, [
            // The user turn, then text the model wrote before it thought, then each round of
            // reasoning immediately ahead of what that reasoning produced.
            "message",
            "message",
            "reasoning",
            "function_call",
            "reasoning",
            "message"
        ]);
        assert_eq!(input[2]["encrypted_content"], "ONE");
        assert_eq!(input[4]["encrypted_content"], "TWO");
    }

    /// A session is not bound to the provider that recorded it: `meka -c -p other` resumes one
    /// under anything. A thinking block another wire recorded must not be replayed here. A Claude
    /// signature is a MAC from another cryptosystem and its `thinking` is the reasoning itself,
    /// so sending them as `encrypted_content` and a `summary` hands OpenAI a blob it cannot
    /// decrypt and leaks the whole Claude reasoning beside it; a Chat Completions block is the
    /// reasoning in the clear with nothing sealed to replay at all.
    #[test]
    fn another_wires_reasoning_is_never_replayed_to_openai() {
        for opaque in [
            OpaqueReasoning::Signed {
                signature: "CLAUDE_MAC".to_string(),
            },
            OpaqueReasoning::Verbatim,
        ] {
            let foreign_turn = assistant(vec![
                ContentBlock::Thinking {
                    thinking: "the full reasoning".to_string(),
                    opaque: Some(opaque.clone()),
                },
                ContentBlock::Text {
                    text: "answer".to_string(),
                },
            ]);
            let serialized = serde_json::to_string(&serde_json::Value::Array(input_of(&[
                Message::user("hi"),
                foreign_turn,
            ])))
            .expect("serialize");

            assert!(
                !serialized.contains("CLAUDE_MAC"),
                "{opaque:?}: {serialized}"
            );
            assert!(
                !serialized.contains("the full reasoning"),
                "{opaque:?}: {serialized}"
            );
            assert!(
                !serialized.contains("\"reasoning\""),
                "{opaque:?}: {serialized}"
            );
        }
    }

    /// A summary is a digest written for a person; without `encrypted_content` there is no
    /// reasoning to resume, so replaying it would only put an unexpected item shape on the wire of
    /// an endpoint that was never asked for reasoning.
    #[test]
    fn a_summary_without_encrypted_content_is_not_replayed() {
        let input = input_of(&[
            Message::user("hi"),
            assistant(vec![thinking("visible only", None, None), tool_call("c1")]),
        ]);

        assert!(
            !input.iter().any(|item| item["type"] == "reasoning"),
            "got {input:?}"
        );
    }

    /// A reasoning item that never carried an id is still replayable; the id is optional upstream
    /// too (`Option<ResponseItemId>`, skipped when absent).
    #[test]
    fn a_reasoning_item_without_an_id_is_still_replayed() {
        let input = input_of(&[
            Message::user("hi"),
            assistant(vec![thinking("", Some("OPAQUE"), None), tool_call("c1")]),
        ]);

        let reasoning = input
            .iter()
            .find(|item| item["type"] == "reasoning")
            .expect("present");
        assert!(reasoning.get("id").is_none(), "got {reasoning:?}");
        assert_eq!(reasoning["summary"], serde_json::json!([]));
    }

    /// Without this the model still reasons and the user sees nothing, which reads as a hang.
    /// Codex asks for `auto`; so does meka, on the one backend whose endpoint is always OpenAI.
    #[test]
    fn the_two_asks_compose_into_a_reasoning_block_even_with_no_effort_set() {
        let mut body = build_request_body("gpt-5", "", &[], &[], None, None, true);
        assert!(body.get("reasoning").is_none(), "no effort, no reasoning");

        request_reasoning_summary(&mut body);
        assert_eq!(body["reasoning"]["summary"], "auto");

        // And settling `reasoning` is what lets the `include` apply to a default profile, which
        // states no reasoning of its own.
        include_encrypted_reasoning(&mut body);
        assert_eq!(
            body["include"],
            serde_json::json!(["reasoning.encrypted_content"])
        );
    }

    /// The two halves composed: a real ChatGPT reasoning turn off the wire, folded into a message,
    /// and handed straight back to the request builder.
    ///
    /// Each half is pinned on its own above, but the thing that actually has to hold is that what
    /// the parser produces is what the encoder can replay. A field renamed on one side and not the
    /// other passes every unit test here and still loses the reasoning chain in production.
    #[tokio::test]
    async fn reasoning_survives_the_wire_and_goes_straight_back_out() {
        let (events, outcome) = decode_sse(concat!(
            "data: {\"type\":\"response.reasoning_summary_part.added\",\"summary_index\":0}\n\n",
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"**Plan**\"}\n\n",
            "data: {\"type\":\"response.reasoning_summary_part.added\",\"summary_index\":1}\n\n",
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"**Act**\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\
             \"id\":\"rs_live\",\"summary\":[],\"encrypted_content\":\"OPAQUE\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"here you go\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\
             \"usage\":{\"input_tokens\":1,\"output_tokens\":2}}}\n\n",
        ))
        .await;
        assert!(outcome.is_ok(), "{outcome:?}");

        let (sender, receiver) = tokio::sync::mpsc::channel(64);
        for event in events {
            sender.send(event).await.expect("buffered");
        }
        drop(sender);
        let message = crate::provider::aggregate_stream(receiver).await.message;

        let input = input_of(&[Message::user("hi"), message]);
        let reasoning = input
            .iter()
            .find(|item| item["type"] == "reasoning")
            .expect("the turn's reasoning must reach the next request");
        assert_eq!(reasoning["id"], "rs_live");
        assert_eq!(reasoning["encrypted_content"], "OPAQUE");
        assert_eq!(reasoning["summary"][0]["text"], "**Plan**\n\n**Act**");
        // And ahead of the answer it produced, which is what the API pairs it with.
        let shape: Vec<&str> = input
            .iter()
            .filter_map(|item| item["type"].as_str())
            .collect();
        assert_eq!(shape, ["message", "reasoning", "message"]);
    }

    /// The summary request must stay out of the shared body for the same reason the `include`
    /// does: `openai-responses` reaches endpoints that never agreed to an OpenAI parameter.
    #[test]
    fn the_shared_body_never_asks_for_a_reasoning_summary() {
        let body = build_request_body("gpt-5", "", &[], &[], Some("high"), None, true);
        assert_eq!(body["reasoning"]["effort"], "high");
        assert!(
            body["reasoning"].get("summary").is_none(),
            "got {:?}",
            body["reasoning"]
        );
    }
}
