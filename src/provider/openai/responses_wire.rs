//! OpenAI Responses API encoder + SSE decoder.
//!
//! The protocol, not a backend: both [`super::responses`] (API key, any endpoint serving
//! `/v1/responses`) and [`super::subscription`] (ChatGPT's `/backend-api/codex/responses`) drive
//! this. It is the OpenAI counterpart of [`crate::provider::anthropic`]'s `shared`, and the reason
//! Chat Completions is a separate module rather than a flag: the two are different wire formats,
//! not two dialects of one.
//!
//! Nothing here may assume which endpoint is on the other end. `store: false` is set
//! unconditionally because Meka replays the whole conversation every turn and never uses
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
    conversation::{ContentBlock, Message, OpaqueReasoning, ResultPart, Role, ToolResultContent},
    error::Result,
    provider::{
        CompletionRequest, DeferredToolDelivery, LOAD_TOOL_NAME, SEARCH_TOOL_NAME, Stop,
        StopReason, StreamEvent, ToolDefinition, search_tool_parameters,
        sse::{End, Step},
    },
    stats::TokenUsage,
};

/// Build the JSON body POSTed to `/responses`. Translates the Meka internal `Message` /
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
/// client execution in place of Meka's two discovery tools, `deferred_tools` ride with
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
        .map(|block| match block.wire_part() {
            ResultPart::Text(text) => serde_json::json!({
                "type": "input_text",
                "text": text,
            }),
            ResultPart::Image(source) => input_image_part(source),
            ResultPart::ToolReference(tool_name) => serde_json::json!({
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
/// the same name is Meka's function tool.
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

/// The endpoint's own search item, declared with client execution so Meka answers it. One door
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
    let mut text_parts: Vec<std::borrow::Cow<'_, str>> = Vec::new();
    let mut image_parts: Vec<serde_json::Value> = Vec::new();

    for block in &message.content {
        match block {
            // The context block, a nudge, a note and what arrived for the model are text on this
            // wire too, in the order the message holds them.
            ContentBlock::Text { .. }
            | ContentBlock::TurnContext { .. }
            | ContentBlock::Nudge { .. }
            | ContentBlock::HarnessNote { .. }
            | ContentBlock::Summary { .. }
            | ContentBlock::Inbox { .. }
            | ContentBlock::Scheduled { .. }
            | ContentBlock::BackgroundReport { .. }
            | ContentBlock::Invocation { .. }
            | ContentBlock::Resource { .. } => text_parts.extend(block.model_text()),
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
                    text_parts.extend(content.iter().filter_map(|item| match item.wire_part() {
                        ResultPart::Text(text) => Some(text),
                        ResultPart::Image(_) | ResultPart::ToolReference(_) => None,
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
            ContentBlock::Thinking { .. }
            | ContentBlock::RedactedThinking { .. }
            | ContentBlock::ToolUse { .. }
            | ContentBlock::Refusal { .. }
            | ContentBlock::Unrecognized { .. } => {}
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

/// Send `part`, a text or refusal part of an assistant message, in the `message` item it came in:
/// the one the block before it went into when both name the same item, else a new one with the
/// `phase` its item carried.
fn message_part(
    input: &mut Vec<serde_json::Value>,
    open_message: &mut Option<(String, usize)>,
    part: serde_json::Value,
    provider_fields: Option<&crate::conversation::ProviderFields>,
) {
    let id = provider_fields
        .and_then(|fields| fields.get("id"))
        .and_then(serde_json::Value::as_str);
    if let (Some(id), Some((open_id, at))) = (id, open_message.as_ref())
        && id == open_id
        && let Some(content) = input
            .get_mut(*at)
            .and_then(|item| item.get_mut("content"))
            .and_then(serde_json::Value::as_array_mut)
    {
        content.push(part);
        return;
    }
    let mut item = serde_json::json!({
        "type": "message",
        "role": "assistant",
        "content": [part],
    });
    if let Some(phase) = provider_fields
        .and_then(|fields| fields.get("phase"))
        .filter(|phase| phase.is_string())
        && let Some(object) = item.as_object_mut()
    {
        object.insert("phase".to_string(), phase.clone());
    }
    input.push(item);
    *open_message = id.map(|id| (id.to_string(), input.len().saturating_sub(1)));
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
    // The `message` item the last block went into, by its id, which the blocks of its other parts
    // join.
    let mut open_message: Option<(String, usize)> = None;
    for (index, block) in message.content.iter().enumerate() {
        if !matches!(
            block,
            ContentBlock::Text { .. } | ContentBlock::Refusal { .. }
        ) {
            open_message = None;
        }
        match block {
            // A `message` item per item the server returned, its parts in it as they came, each
            // item with its `phase`, which the spec asks to be sent back on every assistant
            // message.
            ContentBlock::Text {
                text,
                provider_fields,
            } => {
                if text.is_empty() {
                    continue;
                }
                // The fields the spec defines on an `output_text` part go back as the part held
                // them.
                let mut part = serde_json::Map::from_iter([
                    ("type".to_string(), serde_json::json!("output_text")),
                    ("text".to_string(), serde_json::json!(text)),
                ]);
                if let Some(held) = provider_fields
                    .as_ref()
                    .and_then(|fields| fields.get("content"))
                    .and_then(|parts| parts.get(0))
                    .and_then(serde_json::Value::as_object)
                {
                    for key in ["annotations", "logprobs"] {
                        if let Some(value) = held.get(key) {
                            part.insert(key.to_string(), value.clone());
                        }
                    }
                }
                message_part(
                    input,
                    &mut open_message,
                    serde_json::Value::Object(part),
                    provider_fields.as_ref(),
                );
            }
            ContentBlock::Refusal {
                text,
                provider_fields,
            } => message_part(
                input,
                &mut open_message,
                serde_json::json!({"type": "refusal", "refusal": text}),
                provider_fields.as_ref(),
            ),
            // Reasoning goes back as a `reasoning` item, whichever endpoint returned it: its
            // summary and reasoning parts as returned, and its `encrypted_content` with the id it
            // names only to the host that sealed it. With `store: false` an id names nothing the
            // endpoint kept but the sealed copy beside it. One with no output after it is left
            // out; see `an_emitted_item_follows`.
            ContentBlock::Thinking { parts, opaque, .. }
                if an_emitted_item_follows(&message.content, index) =>
            {
                let sealed = match opaque {
                    Some(OpaqueReasoning::Sealed {
                        encrypted_content,
                        id,
                        origin: Some(origin),
                    }) if replays_reasoning_from == Some(origin.as_str()) => {
                        Some((id.as_deref(), encrypted_content.as_str()))
                    }
                    _ => None,
                };
                if sealed.is_some() || parts.iter().any(|part| !part.text.is_empty()) {
                    input.push(reasoning_input_item(parts, sealed));
                }
            }
            ContentBlock::ToolUse {
                id,
                name,
                input: arguments,
                unparsed,
                ..
            } => {
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
                        "arguments": crate::conversation::call_arguments_text(
                            arguments,
                            unparsed.as_ref(),
                        ),
                    }));
                }
            }
            // Reasoning with no output after it, and what Meka could not read, are never sent; the
            // rest belongs to a user message.
            ContentBlock::Thinking { .. }
            | ContentBlock::RedactedThinking { .. }
            | ContentBlock::Unrecognized { .. }
            | ContentBlock::TurnContext { .. }
            | ContentBlock::Nudge { .. }
            | ContentBlock::HarnessNote { .. }
            | ContentBlock::Summary { .. }
            | ContentBlock::Inbox { .. }
            | ContentBlock::Scheduled { .. }
            | ContentBlock::BackgroundReport { .. }
            | ContentBlock::Invocation { .. }
            | ContentBlock::Resource { .. }
            | ContentBlock::Image { .. }
            | ContentBlock::ToolResult { .. } => {}
        }
    }
}

/// Whether anything this encoder will actually emit follows `index` in the same turn.
///
/// OpenAI rejects a request whose reasoning item has nothing it led to after it, and that shape is
/// nobody's choice: [`Message::without_tool_use`] drops a cut reply's calls but keeps its thinking,
/// the thinking-only nudge follows a reply that is only reasoning, and a reply can stop
/// mid-reasoning. Leaving such an item out keeps the session going, and the record keeps it; an
/// endpoint that returns reasoning after its output does not get that reasoning back.
///
/// Only output counts, so a run of reasoning items is carried by whatever follows the run rather
/// than by each other, which is the order the server produced them in, and so the order they may
/// be replayed in. What this refuses is a turn that ends in reasoning with no output at all.
fn an_emitted_item_follows(content: &[ContentBlock], index: usize) -> bool {
    content.iter().skip(index + 1).any(|block| match block {
        ContentBlock::Text { text, .. } => !text.is_empty(),
        ContentBlock::ToolUse { .. } | ContentBlock::Refusal { .. } => true,
        _ => false,
    })
}

/// A `reasoning` item, shaped as the server sent it so it can be replayed verbatim: each summary
/// part as a `summary_text` entry and each part of reasoning as a `reasoning_text` one, in the
/// order they came, with the id and `encrypted_content` of the endpoint's seal when it goes back
/// to the endpoint that sealed it.
fn reasoning_input_item(
    parts: &[crate::conversation::ReasoningPart],
    sealed: Option<(Option<&str>, &str)>,
) -> serde_json::Value {
    use crate::conversation::ReasoningKind;
    let entries = |kind: ReasoningKind, entry_type: &str| -> Vec<serde_json::Value> {
        parts
            .iter()
            .filter(|part| part.kind == kind)
            .map(|part| serde_json::json!({ "type": entry_type, "text": part.text }))
            .collect()
    };
    let summary = entries(ReasoningKind::Summary, "summary_text");
    let content = entries(ReasoningKind::Reasoning, "reasoning_text");
    let mut item = serde_json::Map::from_iter([
        ("type".into(), serde_json::json!("reasoning")),
        ("summary".into(), serde_json::json!(summary)),
    ]);
    if !content.is_empty() {
        item.insert("content".into(), serde_json::json!(content));
    }
    if let Some((id, encrypted_content)) = sealed {
        item.insert(
            "encrypted_content".into(),
            serde_json::json!(encrypted_content),
        );
        if let Some(id) = id {
            item.insert("id".into(), serde_json::Value::String(id.to_string()));
        }
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
    // for nothing, as Meka's own discovery tools would be.
    if searched && (!deferred_tools.is_empty() || replays_a_search) {
        declared.push(tool_search_declaration());
    }
    declared.extend(
        tools
            .iter()
            // The endpoint's search item does both jobs here; Meka's two function tools would
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

/// What [`process_event`] carries between frames: the output items in progress, by the index the
/// spec addresses each by, and what the response has said of itself.
#[derive(Default)]
pub(super) struct SseState {
    /// The output items open now, by `output_index`: every item and content event names its item
    /// by it, and items may stream at once.
    items: std::collections::BTreeMap<u64, OpenItem>,
    /// Whether the stream opened or finished any output item, which makes what it sent the reply
    /// rather than the terminal response's output; see [`Self::finish_from_output`].
    sent_an_item: bool,
    /// The `output_index` of the item opened last, after which an event that names no item is
    /// kept.
    latest: u64,
    /// The place the events returned so far are of; see [`StreamEvent::Place`].
    place: u64,
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
    /// The fields Meka's request set, which the response object restates and which are not the
    /// response's own to record.
    pub(super) restated: Vec<String>,
    /// What a failed response states of itself, its usage and fields, told ahead of its failure so
    /// the partial reply keeps them.
    told_before_failing: Vec<StreamEvent>,
    /// What a failed response's output finished of the items still open, handed over after its
    /// failure with the items it leaves open.
    finished_by_failure: Vec<StreamEvent>,
    /// The name each item the stream sent gave itself, once per item. A name given to more than
    /// one item finds none of them in the terminal output; see [`Self::finish_from_output`].
    named: Vec<ItemIdentity>,
    /// The response's own fields as last stated, which [`Self::response_events`] are told with.
    fields: Option<crate::conversation::ProviderFields>,
    /// Events of the response itself that Meka reads nothing from, ChatGPT's `response.metadata`
    /// among them, each as it came under its type, several of one type as a list. They are the
    /// response's, not its output's, so the reply's record keeps them beside its fields rather
    /// than in its content.
    response_events: crate::conversation::ProviderFields,
}

/// The fields of the response object that are the response's own: all but its output, which the
/// reply holds, its model and how it ended, which the reply's record holds, and what restates the
/// request. `service_tier` is the response's even when the request set it: the spec defines it
/// there as the tier actually used, which may differ from the one asked for.
fn response_fields(
    response: &serde_json::Value,
    restated: &[String],
) -> Option<crate::conversation::ProviderFields> {
    let response = response.as_object()?;
    let fields: crate::conversation::ProviderFields = response
        .iter()
        .filter(|(key, _)| {
            !["output", "model", "status", "incomplete_details"].contains(&key.as_str())
                && (*key == "service_tier" || !restated.contains(key))
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    (!fields.is_empty()).then_some(fields)
}

/// An output item's fields beyond `held`, the ones its blocks hold, with each list of parts under
/// a key of `parts` kept as the parts' own remainders, in place: each part's fields beyond those
/// listed beside the key. A part left with nothing is an empty object, so the places hold; a list
/// whose parts all are is left out, since its blocks restate it. An empty list stays, since no
/// block tells it from a list never stated, and so does a value under the key that is no list.
fn item_fields(
    item: &serde_json::Value,
    held: &[&str],
    parts: &[(&str, &[&str])],
) -> Option<crate::conversation::ProviderFields> {
    let item = item.as_object()?;
    let mut fields = crate::provider::fields_beyond(item, held).unwrap_or_default();
    for (key, part_held) in parts {
        let Some(list) = item.get(*key).and_then(serde_json::Value::as_array) else {
            continue;
        };
        fields.remove(*key);
        let remainders: Vec<serde_json::Value> = list
            .iter()
            .map(|part| {
                serde_json::Value::Object(
                    part.as_object()
                        .and_then(|part| crate::provider::fields_beyond(part, part_held))
                        .unwrap_or_default(),
                )
            })
            .collect();
        if remainders.is_empty()
            || remainders
                .iter()
                .any(|part| part.as_object().is_some_and(|part| !part.is_empty()))
        {
            fields.insert((*key).to_string(), serde_json::Value::Array(remainders));
        }
    }
    (!fields.is_empty()).then_some(fields)
}

/// An output item between its `output_item.added` and its `output_item.done`.
#[derive(Default)]
struct OpenItem {
    kind: ItemKind,
    /// What the item that opened it named it; see [`item_identity`].
    identity: Option<ItemIdentity>,
    parts: Parts,
    /// Events addressed to the item in a shape the spec does not allow (a part event naming no
    /// part), kept whole after the item so it is not split.
    unread: Vec<serde_json::Value>,
}

/// What an open item is, with what its end needs.
#[derive(Default)]
enum ItemKind {
    /// A function call or the endpoint's search call: its id and name as the item that opened it
    /// named them, `None` when it did not, which leaves the call to the item that finishes it, and
    /// its arguments as they streamed.
    Call {
        opened: Option<(String, String)>,
        arguments: String,
    },
    /// A reasoning item, and whether it streamed anything readable.
    Reasoning { showed: bool },
    /// A message.
    Message,
    /// An item of a type Meka reads nothing from until it is done.
    #[default]
    Other,
}

/// One of a reasoning item's lists of parts: the key it is listed under, and the index the spec
/// numbers its parts with.
type ReasoningList = (&'static str, fn(u64) -> PartIndex);

/// A part of an item, by the index the spec numbers it with: a message's content and a reasoning
/// item's raw text by `content_index`, a reasoning summary by `summary_index`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PartIndex {
    Content(u64),
    Summary(u64),
}

impl PartIndex {
    /// The event that begins this part of a reasoning item, with its kind and number.
    fn opening(self) -> StreamEvent {
        let (kind, number) = match self {
            Self::Summary(number) => (crate::conversation::ReasoningKind::Summary, number),
            Self::Content(number) => (crate::conversation::ReasoningKind::Reasoning, number),
        };
        StreamEvent::ThinkingPart {
            kind,
            number: Some(number),
        }
    }

    /// Whether `self` and `other` are numbered by the same index.
    fn same_kind(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::Content(_), Self::Content(_)) | (Self::Summary(_), Self::Summary(_))
        )
    }
}

/// An item's parts in progress, handed over one part at a time and each kind in its indexes'
/// order, as the reference client folds them by index: the words of a part go on the part in
/// progress, so what a part streams while another part streams, or while a lower part of its kind
/// is open, waits, and follows when that part ends.
#[derive(Default)]
struct Parts {
    /// The parts begun and not ended.
    open: std::collections::BTreeSet<PartIndex>,
    /// The parts that have begun, so a reasoning part opens once.
    begun: std::collections::BTreeSet<PartIndex>,
    /// The parts that have ended, whose words are whole.
    ended: std::collections::BTreeSet<PartIndex>,
    /// The parts that received words, by delta or whole at their end.
    worded: std::collections::BTreeSet<PartIndex>,
    /// The part streaming now: it has handed over events and not ended.
    streaming: Option<PartIndex>,
    /// What parts streamed while they waited.
    waiting: std::collections::BTreeMap<PartIndex, Vec<StreamEvent>>,
}

impl Parts {
    /// Whether `part` waits: another part is streaming, or a lower part of its kind is open.
    fn held_back(&self, part: PartIndex) -> bool {
        self.streaming.is_some_and(|streaming| streaming != part)
            || self.open.range(..part).any(|lower| lower.same_kind(part))
    }

    /// `events` of `part`, which is open from its first: returned to go out now, or kept until
    /// the lower parts of its kind end.
    fn stream(&mut self, part: PartIndex, events: Vec<StreamEvent>) -> Vec<StreamEvent> {
        self.open.insert(part);
        if self.held_back(part) || self.waiting.contains_key(&part) {
            self.waiting.entry(part).or_default().extend(events);
            return Vec::new();
        }
        if !events.is_empty() {
            self.streaming = Some(part);
        }
        events
    }

    /// End `part`, and return what waited on it, in order, as far as no open part holds it back.
    fn end(&mut self, part: PartIndex) -> Vec<StreamEvent> {
        self.open.remove(&part);
        self.ended.insert(part);
        if self.streaming == Some(part) {
            self.streaming = None;
        }
        let mut released = Vec::new();
        while let Some(next) = self
            .waiting
            .keys()
            .copied()
            .find(|waiting| !self.held_back(*waiting))
        {
            let events = self.waiting.remove(&next).unwrap_or_default();
            // A part still open streams on from what it held, ahead of the rest.
            if self.open.contains(&next) && !events.is_empty() {
                self.streaming = Some(next);
            }
            released.extend(events);
        }
        released
    }

    /// `events`, the words of message part `part`, after the mark that begins it when they are
    /// its first, so the part is a block of its own; see [`StreamEvent::ContentPart`].
    fn begin_content(&mut self, part: PartIndex, mut events: Vec<StreamEvent>) -> Vec<StreamEvent> {
        if let PartIndex::Content(number) = part
            && !events.is_empty()
            && self.begun.insert(part)
        {
            events.insert(0, StreamEvent::ContentPart(number));
        }
        events
    }

    /// Everything still waiting, in order, once the item ends.
    fn finish(&mut self) -> Vec<StreamEvent> {
        self.open.clear();
        self.streaming = None;
        std::mem::take(&mut self.waiting)
            .into_values()
            .flatten()
            .collect()
    }
}

/// The parts an item lists under `key`, as its done item states them.
fn parts_of<'a>(
    item: &'a serde_json::Value,
    key: &str,
) -> impl Iterator<Item = &'a serde_json::Value> {
    item.get(key)
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
}

/// The words a part states under `key`, when it states any.
fn text_of(part: &serde_json::Value, key: &str) -> Option<String> {
    part.get(key)
        .and_then(serde_json::Value::as_str)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

/// The id and name a call item names, which the spec has the item that opens a call and the one
/// that finishes it both carry. The endpoint's search item has no name of its own; it dispatches
/// as Meka's search tool, which answers both a query and a list of names.
fn call_identity(item: &serde_json::Value) -> Option<(String, String)> {
    let text = |key: &str| item.get(key).and_then(serde_json::Value::as_str);
    match item.get("type").and_then(serde_json::Value::as_str) {
        Some("tool_search_call") => Some((
            text("call_id").or_else(|| text("id"))?.to_string(),
            SEARCH_TOOL_NAME.to_string(),
        )),
        _ => Some((text("call_id")?.to_string(), text("name")?.to_string())),
    }
}

/// What an output item names itself, by which the terminal response's output is matched to an item
/// the stream left open: its place there need not be the `output_index` the stream gave it.
#[derive(Clone, PartialEq, Eq)]
enum ItemIdentity {
    /// An item that carries a `call_id`, a call, by it.
    Call(String),
    /// Any other item, by its `id`. Never compared with a call's `call_id`: an endpoint may give
    /// reasoning an `id` equal to the `call_id` of the call it led to, as OpenRouter's Gemini
    /// route does.
    Item(String),
}

/// What `item` names itself, when it names itself at all; see [`ItemIdentity`].
fn item_identity(item: &serde_json::Value) -> Option<ItemIdentity> {
    let text = |key: &str| {
        item.get(key)
            .and_then(serde_json::Value::as_str)
            .filter(|identity| !identity.is_empty())
            .map(str::to_string)
    };
    match item.get("call_id") {
        Some(_) => text("call_id").map(ItemIdentity::Call),
        None => text("id").map(ItemIdentity::Item),
    }
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

impl SseState {
    /// Add `events` of the item at `index` to `out`, saying first which item they are of when the
    /// events before them were another's.
    fn at(&mut self, index: u64, events: Vec<StreamEvent>, out: &mut Vec<StreamEvent>) {
        if events.is_empty() {
            return;
        }
        if index != self.place {
            out.push(StreamEvent::Place(index));
            self.place = index;
        }
        out.extend(events);
    }

    /// Keep `frame`, an item event in a shape the spec does not allow, as it came: after the item
    /// it names, or, naming none, inside the item opened last while it is open so the item is not
    /// split, else after it.
    fn keep_unread(
        &mut self,
        index: Option<u64>,
        frame: &serde_json::Value,
        out: &mut Vec<StreamEvent>,
    ) {
        let index = match index {
            Some(index) => index,
            None => match self.items.get_mut(&self.latest) {
                Some(item) => {
                    item.unread.push(frame.clone());
                    return;
                }
                None => self.latest,
            },
        };
        self.at(index, vec![StreamEvent::Unrecognized(frame.clone())], out);
    }

    /// The response's fields to tell, `stated` by the response object when it states them, with
    /// the response's own events beside them.
    fn fields_with(
        &mut self,
        stated: Option<crate::conversation::ProviderFields>,
    ) -> Option<StreamEvent> {
        if let Some(stated) = stated {
            self.fields = Some(stated);
        }
        let mut fields = self.fields.clone().unwrap_or_default();
        fields.extend(self.response_events.clone());
        (!fields.is_empty()).then_some(StreamEvent::ResponseFields(fields))
    }

    /// Keep `frame`, an event of a type Meka reads nothing from, as it came: inside the item it
    /// names, unless that item is of a type Meka reads nothing from, whose done item restates
    /// what it streamed; after the item it names when that item is not open; and, naming none,
    /// as the response's own, which its record keeps under the event's type.
    fn keep_event(&mut self, frame: &serde_json::Value, out: &mut Vec<StreamEvent>) {
        let Some(index) = frame
            .get("output_index")
            .and_then(serde_json::Value::as_u64)
        else {
            let name = frame
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("event")
                .to_string();
            let kept = match self.response_events.remove(&name) {
                None => frame.clone(),
                Some(serde_json::Value::Array(mut frames)) => {
                    frames.push(frame.clone());
                    serde_json::Value::Array(frames)
                }
                Some(earlier) => serde_json::json!([earlier, frame]),
            };
            self.response_events.insert(name, kept);
            out.extend(self.fields_with(None));
            return;
        };
        match self.items.get_mut(&index) {
            Some(OpenItem {
                kind: ItemKind::Other,
                ..
            }) => {}
            Some(item) => item.unread.push(frame.clone()),
            None => self.keep_unread(Some(index), frame, out),
        }
    }

    /// Open the item `output_item.added` names. A repeated index closes the item open there
    /// first, as one that never finished, so each keeps what it streamed.
    fn open_item(&mut self, data: &serde_json::Value, out: &mut Vec<StreamEvent>) {
        let index = data.get("output_index").and_then(serde_json::Value::as_u64);
        // An `added` without its item says nothing, as the reference client reads it.
        let Some(item) = data.get("item") else {
            return;
        };
        let Some(index) = index else {
            self.keep_unread(None, data, out);
            return;
        };
        if let Some(earlier) = self.items.remove(&index) {
            self.close_unfinished(index, earlier, out);
        }
        self.sent_an_item = true;
        self.latest = index;
        let kind = match item.get("type").and_then(serde_json::Value::as_str) {
            Some("function_call" | "tool_search_call") => {
                let opened = call_identity(item);
                if let Some((id, name)) = opened.clone() {
                    self.emitted_tool_call = true;
                    self.at(index, vec![StreamEvent::ToolUseStart { id, name }], out);
                }
                ItemKind::Call {
                    opened,
                    arguments: String::new(),
                }
            }
            Some("reasoning") => ItemKind::Reasoning { showed: false },
            Some("message") => ItemKind::Message,
            _ => ItemKind::Other,
        };
        let identity = item_identity(item);
        self.named.extend(identity.clone());
        self.items.insert(index, OpenItem {
            kind,
            identity,
            ..OpenItem::default()
        });
    }

    /// Fold an event of an item in progress into it: a delta to the part its index names, a part
    /// beginning or ending. One for an item that is not open is kept as it came; one of a kind its
    /// item does not read is left to the item's done, as the reference client leaves it.
    fn item_event(
        &mut self,
        event_name: &str,
        data: &serde_json::Value,
        out: &mut Vec<StreamEvent>,
    ) {
        let index = data.get("output_index").and_then(serde_json::Value::as_u64);
        let Some(item) = index.and_then(|index| self.items.get_mut(&index)) else {
            self.keep_unread(index, data, out);
            return;
        };
        let index = index.unwrap_or_default();
        let number = |key: &str| data.get(key).and_then(serde_json::Value::as_u64);
        let delta = data
            .get("delta")
            .and_then(serde_json::Value::as_str)
            .filter(|delta| !delta.is_empty());
        let part = match event_name {
            "response.reasoning_summary_text.delta"
            | "response.reasoning_summary_part.added"
            | "response.reasoning_summary_part.done" => {
                number("summary_index").map(PartIndex::Summary)
            }
            "response.function_call_arguments.delta" | "response.custom_tool_call_input.delta" => {
                None
            }
            _ => number("content_index").map(PartIndex::Content),
        };
        // A part's words are whole once it ends; more for it is a shape the spec does not allow.
        if part.is_some_and(|part| item.parts.ended.contains(&part)) {
            item.unread.push(data.clone());
            return;
        }
        let events = match (&mut item.kind, event_name, part) {
            (
                ItemKind::Call { arguments, .. },
                "response.function_call_arguments.delta" | "response.custom_tool_call_input.delta",
                _,
            ) => {
                let Some(delta) = delta else {
                    return;
                };
                arguments.push_str(delta);
                vec![StreamEvent::ToolArgumentsDelta(delta.to_string())]
            }
            // A part event naming no part has no place in its item.
            (ItemKind::Message | ItemKind::Reasoning { .. }, _, None)
                if !event_name.starts_with("response.function_call")
                    && !event_name.starts_with("response.custom_tool_call") =>
            {
                item.unread.push(data.clone());
                return;
            }
            // A part ends whole: one that streamed no words states them in its done, as the
            // reference client takes the part.
            (ItemKind::Message, "response.content_part.done", Some(part)) => {
                let stated = data.get("part");
                let whole: Vec<StreamEvent> = if item.parts.worded.contains(&part) {
                    Vec::new()
                } else {
                    match stated
                        .and_then(|stated| stated.get("type"))
                        .and_then(serde_json::Value::as_str)
                    {
                        Some("output_text") => stated
                            .and_then(|stated| text_of(stated, "text"))
                            .map(StreamEvent::TextDelta),
                        Some("refusal") => stated
                            .and_then(|stated| text_of(stated, "refusal"))
                            .map(StreamEvent::RefusalDelta),
                        _ => None,
                    }
                    .into_iter()
                    .collect()
                };
                if !whole.is_empty() {
                    item.parts.worded.insert(part);
                }
                let whole = item.parts.begin_content(part, whole);
                let mut events = item.parts.stream(part, whole);
                events.extend(item.parts.end(part));
                events
            }
            (
                ItemKind::Reasoning { showed },
                "response.content_part.done" | "response.reasoning_summary_part.done",
                Some(part),
            ) => {
                let mut whole = Vec::new();
                if !item.parts.worded.contains(&part)
                    && let Some(text) = data.get("part").and_then(|stated| text_of(stated, "text"))
                {
                    if item.parts.begun.insert(part) {
                        whole.push(part.opening());
                    }
                    whole.push(StreamEvent::ThinkingDelta(text));
                    *showed = true;
                    item.parts.worded.insert(part);
                }
                let mut events = item.parts.stream(part, whole);
                events.extend(item.parts.end(part));
                events
            }
            (
                ItemKind::Message | ItemKind::Reasoning { .. },
                "response.content_part.added",
                Some(part),
            ) => item.parts.stream(part, Vec::new()),
            (ItemKind::Message, "response.output_text.delta", Some(part)) => {
                if delta.is_some() {
                    item.parts.worded.insert(part);
                }
                let words = delta
                    .map(|delta| StreamEvent::TextDelta(delta.to_string()))
                    .into_iter()
                    .collect();
                let words = item.parts.begin_content(part, words);
                item.parts.stream(part, words)
            }
            (ItemKind::Message, "response.refusal.delta", Some(part)) => {
                if delta.is_some() {
                    item.parts.worded.insert(part);
                }
                let words = delta
                    .map(|delta| StreamEvent::RefusalDelta(delta.to_string()))
                    .into_iter()
                    .collect();
                let words = item.parts.begin_content(part, words);
                item.parts.stream(part, words)
            }
            // Each reasoning part is recorded as the part it is: a summary part opens with its
            // `added`, raw reasoning with its first words.
            (
                ItemKind::Reasoning { showed },
                "response.reasoning_text.delta"
                | "response.reasoning_summary_text.delta"
                | "response.reasoning_summary_part.added",
                Some(part),
            ) => {
                let opens =
                    event_name == "response.reasoning_summary_part.added" || delta.is_some();
                let mut events = Vec::new();
                if opens && item.parts.begun.insert(part) {
                    events.push(part.opening());
                }
                if let Some(delta) = delta {
                    *showed = true;
                    item.parts.worded.insert(part);
                    events.push(StreamEvent::ThinkingDelta(delta.to_string()));
                }
                item.parts.stream(part, events)
            }
            // What an item of a type Meka reads nothing from streams is restated whole by its
            // done item, which is kept as it came.
            (ItemKind::Other, ..) => return,
            // An event of a kind its item does not have is a shape the spec does not allow.
            _ => {
                item.unread.push(data.clone());
                return;
            }
        };
        self.at(index, events, out);
    }

    /// Close the item `output_item.done` finishes, as its done item states it: what its parts still
    /// held, then its block, then what it received unread. The done item alone makes an item too,
    /// as the reference client takes it.
    fn finish_item(&mut self, data: &serde_json::Value, out: &mut Vec<StreamEvent>) {
        let Some(item) = data.get("item") else {
            return;
        };
        let Some(index) = data.get("output_index").and_then(serde_json::Value::as_u64) else {
            self.keep_unread(None, data, out);
            return;
        };
        self.sent_an_item = true;
        let open = self.items.remove(&index).unwrap_or_else(|| {
            self.named.extend(item_identity(item));
            OpenItem::default()
        });
        let events = self.finished(item, open);
        self.at(index, events, out);
    }

    /// The events that finish `open` as its done `item` states it.
    fn finished(&mut self, item: &serde_json::Value, mut open: OpenItem) -> Vec<StreamEvent> {
        let mut events = open.parts.finish();
        let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if item_type == "tool_search_call" || item_type == "function_call" {
            let (opened, buffered) = match open.kind {
                ItemKind::Call { opened, arguments } => (opened, Some(arguments)),
                _ => (None, None),
            };
            // The call is the one the item that finishes it names, as the reference client records
            // the done item; one that names none is the one the item that opened it named. A call
            // neither names is in a shape the spec does not define, kept as it came and never run.
            let Some((call_id, call_name)) = call_identity(item).or_else(|| opened.clone()) else {
                events.push(StreamEvent::Unrecognized(item.clone()));
                events.extend(open.unread.into_iter().map(StreamEvent::Unrecognized));
                return events;
            };
            if opened.is_none() {
                self.emitted_tool_call = true;
                events.push(StreamEvent::ToolUseStart {
                    id: call_id.clone(),
                    name: call_name.clone(),
                });
            }
            if item_type == "tool_search_call" {
                // Arrives whole: the arguments are an object on the done item, with no deltas
                // before it.
                let arguments = match item.get("arguments") {
                    Some(serde_json::Value::String(raw)) => raw.clone(),
                    Some(value) if !value.is_null() => value.to_string(),
                    _ => "{}".to_string(),
                };
                events.push(crate::provider::tool_use_event(
                    call_id,
                    SEARCH_TOOL_NAME.to_string(),
                    &arguments,
                    item.as_object().and_then(|item| {
                        crate::provider::fields_beyond(item, &["type", "call_id", "arguments"])
                    }),
                ));
            } else {
                // The final `arguments` from the item wins over the accumulated buffer, since the
                // server may normalize it; one that is not the string the spec defines refuses
                // the call with what it was.
                let arguments = match item.get("arguments") {
                    Some(serde_json::Value::Null) | None => {
                        crate::provider::finalize_tool_arguments(
                            &call_name,
                            &buffered.unwrap_or_default(),
                        )
                    }
                    arguments => crate::provider::arguments_from_reply(&call_name, arguments),
                };
                events.push(crate::provider::call_event(
                    call_id,
                    call_name,
                    arguments,
                    item.as_object().and_then(|item| {
                        crate::provider::fields_beyond(item, &[
                            "type",
                            "call_id",
                            "name",
                            "arguments",
                        ])
                    }),
                ));
            }
        } else if item_type == "message" {
            // A part that received no words, by delta or whole at its own end, is stated whole by
            // the done item, as the reference client takes it: a block in its place by its
            // number, one with no words too, which still holds its own fields and its kind. A
            // part of a type the spec does not define for a message is kept as it came.
            let mut unread = Vec::new();
            for (number, part) in (0_u64..).zip(parts_of(item, "content")) {
                if open.parts.worded.contains(&PartIndex::Content(number)) {
                    continue;
                }
                let words = match part.get("type").and_then(serde_json::Value::as_str) {
                    Some("output_text") => text_of(part, "text").map(StreamEvent::TextDelta),
                    Some("refusal") => Some(StreamEvent::RefusalDelta(
                        text_of(part, "refusal").unwrap_or_default(),
                    )),
                    _ => {
                        unread.push(StreamEvent::Unrecognized(part.clone()));
                        continue;
                    }
                };
                events.push(StreamEvent::ContentPart(number));
                events.extend(words);
            }
            // One part of the reply: its text closes as a block of its own, with what the item
            // carries beside what its blocks hold, `phase` among it, and each content part's own
            // fields, `annotations` among them.
            events.push(StreamEvent::TextEnd {
                provider_fields: item_fields(item, &["type", "role"], &[("content", &[
                    "type", "text", "refusal",
                ])]),
            });
            events.extend(unread);
        } else if item_type == "reasoning" {
            // True when this item has anything readable, by either spelling: a requested summary,
            // or the raw reasoning text a local server emits unprompted. A part that received no
            // words, by delta or whole at its own end, is stated whole by the done item, as the
            // reference client takes it, and the block keeps it in its place by its number.
            let mut showed_its_reasoning =
                matches!(open.kind, ItemKind::Reasoning { showed: true });
            let lists: [ReasoningList; 2] = [
                ("summary", PartIndex::Summary),
                ("content", PartIndex::Content),
            ];
            for (key, index_of) in lists {
                for (number, part) in (0_u64..).zip(parts_of(item, key)) {
                    if open.parts.worded.contains(&index_of(number)) {
                        continue;
                    }
                    if let Some(text) = text_of(part, "text") {
                        showed_its_reasoning = true;
                        events.push(index_of(number).opening());
                        events.push(StreamEvent::ThinkingDelta(text));
                    }
                }
            }
            let encrypted_content = item
                .get("encrypted_content")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            // Deliberately not gated on that. `encrypted_content` is what the next request
            // replays, and it arrives whether or not summaries were requested, so gating its
            // capture on visible text would silently throw the reasoning chain away for exactly
            // the configuration that asked for it. An item with neither is worth nothing to
            // either the reader or the next turn, so it makes no block.
            if showed_its_reasoning || encrypted_content.is_some() {
                let id = item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                // What the block does not hold: the id when nothing sealed it, and each summary
                // and reasoning part's own fields.
                let held: &[&str] = if encrypted_content.is_some() {
                    &["type", "id", "encrypted_content"]
                } else {
                    &["type"]
                };
                let provider_fields = item_fields(item, held, &[
                    ("summary", &["type", "text"]),
                    ("content", &["type", "text"]),
                ]);
                events.push(StreamEvent::ThinkingComplete {
                    opaque: encrypted_content.map(|encrypted_content| OpaqueReasoning::Sealed {
                        encrypted_content,
                        id,
                        origin: self.origin.clone(),
                    }),
                    provider_fields,
                });
            } else {
                // Nothing to read and nothing to carry: no block stands for it, so it is kept as
                // it came.
                events.push(StreamEvent::Unrecognized(item.clone()));
            }
        } else {
            // An item of a type Meka reads nothing from, kept as it came.
            events.push(StreamEvent::Unrecognized(item.clone()));
        }
        events.extend(open.unread.into_iter().map(StreamEvent::Unrecognized));
        events
    }

    /// Close `item`, open at `index`, as one that never finished, so what opens there next is
    /// an item of its own: its text and reasoning as they stand, a call refused with the
    /// arguments that had streamed.
    fn close_unfinished(&mut self, index: u64, mut item: OpenItem, out: &mut Vec<StreamEvent>) {
        let mut events = item.parts.finish();
        match item.kind {
            ItemKind::Call {
                opened: Some((id, name)),
                arguments,
            } => events.push(StreamEvent::ToolCallRefused {
                id,
                name,
                unparsed: crate::provider::unfinished_call(arguments),
                provider_fields: None,
            }),
            ItemKind::Message => events.push(StreamEvent::TextEnd {
                provider_fields: None,
            }),
            ItemKind::Reasoning { .. } => events.push(StreamEvent::ThinkingComplete {
                opaque: None,
                provider_fields: None,
            }),
            ItemKind::Call { opened: None, .. } | ItemKind::Other => {}
        }
        events.extend(item.unread.into_iter().map(StreamEvent::Unrecognized));
        self.at(index, events, out);
    }

    /// Finish the reply from the terminal response's output, whichever event ended it. What the
    /// stream sent is the reply, as Codex takes it: an item still open is finished as the output
    /// item naming itself exactly as the open one did states it, never found by its place there,
    /// and one no output item names is left to [`Self::release_open`]. A name the stream or the
    /// output gave more than one item names none of them for certain, so it finishes nothing. An
    /// item the stream never sent is not added, since a server may restate what it streamed under
    /// new names. A stream that sent no item at all takes the output as its reply, as
    /// openai-python takes the completed response.
    fn finish_from_output(&mut self, data: &serde_json::Value, out: &mut Vec<StreamEvent>) {
        let output: Vec<&serde_json::Value> = data
            .get("response")
            .and_then(|response| response.get("output"))
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .collect();
        if !self.sent_an_item {
            for (index, item) in (0_u64..).zip(output) {
                let events = self.finished(item, OpenItem::default());
                self.at(index, events, out);
            }
            return;
        }
        let output_names: Vec<Option<ItemIdentity>> =
            output.iter().map(|item| item_identity(item)).collect();
        for (item, identity) in output.iter().zip(&output_names) {
            let Some(identity) = identity else {
                continue;
            };
            let streamed = self.named.iter().filter(|name| *name == identity).count();
            let listed = output_names
                .iter()
                .filter(|name| name.as_ref() == Some(identity))
                .count();
            if streamed != 1 || listed != 1 {
                continue;
            }
            let open = self
                .items
                .iter()
                .find(|(_, open)| open.identity.as_ref() == Some(identity))
                .map(|(index, _)| *index);
            if let Some(index) = open {
                self.finish_item(
                    &serde_json::json!({"output_index": index, "item": item}),
                    out,
                );
            }
        }
    }

    /// Hand over what the items still open hold, in index order: the response ended, or the
    /// stream broke off, before they were done.
    fn release_open(&mut self, out: &mut Vec<StreamEvent>) {
        for (index, mut item) in std::mem::take(&mut self.items) {
            let mut events = item.parts.finish();
            events.extend(item.unread.into_iter().map(StreamEvent::Unrecognized));
            self.at(index, events, out);
        }
    }
}

/// Pure SSE-event handler. Inspects the named event and parsed JSON payload, updates `state`, and
/// returns the Meka-level [`StreamEvent`]s to forward to the agent. Returns `Err` when the server
/// reports a fatal stream error; the driver propagates this back to the caller.
pub(super) fn process_event(
    event_name: &str,
    data: &serde_json::Value,
    state: &mut SseState,
) -> Result<Vec<StreamEvent>> {
    let mut out = Vec::new();
    match event_name {
        // The model the response names, which the accumulator keeps the first of.
        "response.created" | "response.queued" | "response.in_progress" => {
            if let Some(model) = data
                .get("response")
                .and_then(|response| response.get("model"))
                .and_then(|model| model.as_str())
                .filter(|model| !model.is_empty())
            {
                out.push(StreamEvent::Model(model.to_string()));
            }
            // Told as soon as it is known, so a reply cut short keeps it; the terminal event's
            // response is the whole of it.
            let stated = data
                .get("response")
                .and_then(|response| response_fields(response, &state.restated));
            out.extend(state.fields_with(stated));
        }

        "response.output_item.added" => state.open_item(data, &mut out),
        "response.output_item.done" => state.finish_item(data, &mut out),
        "response.output_text.delta"
        | "response.refusal.delta"
        | "response.reasoning_text.delta"
        | "response.reasoning_summary_text.delta"
        | "response.reasoning_summary_part.added"
        | "response.reasoning_summary_part.done"
        | "response.content_part.added"
        | "response.content_part.done"
        | "response.function_call_arguments.delta"
        | "response.custom_tool_call_input.delta" => state.item_event(event_name, data, &mut out),

        "response.completed" => {
            state.finished = true;
            state.finish_from_output(data, &mut out);
            state.release_open(&mut out);
            if let Some(response) = data.get("response") {
                if let Some(usage) = response.get("usage") {
                    out.push(StreamEvent::Usage(usage_from(usage)));
                }
                // The Responses API reports `status: "completed"` even when the output is function
                // calls, so a tool-call turn must be surfaced as `ToolUse` regardless of status;
                // otherwise the agent warns and mislabels the turn as `EndTurn`.
                let mut stop = responses_stop(response, "completed");
                if state.emitted_tool_call {
                    stop.reason = StopReason::ToolUse;
                }
                let stated = response_fields(response, &state.restated);
                out.extend(state.fields_with(stated));
                out.push(StreamEvent::MessageEnd { stop });
            } else if state.emitted_tool_call {
                out.push(StreamEvent::MessageEnd {
                    stop: crate::provider::Stop::of(StopReason::ToolUse),
                });
            } else {
                out.push(StreamEvent::MessageEnd {
                    stop: crate::provider::Stop::of(StopReason::EndTurn),
                });
            }
        }

        "response.failed" => {
            state.finished = true;
            let response = data.get("response");
            if let Some(usage) = response
                .and_then(|response| response.get("usage"))
                .filter(|usage| usage.is_object())
            {
                state
                    .told_before_failing
                    .push(StreamEvent::Usage(usage_from(usage)));
            }
            let stated = response.and_then(|response| response_fields(response, &state.restated));
            if let Some(fields) = state.fields_with(stated) {
                state.told_before_failing.push(fields);
            }
            let mut finished = Vec::new();
            state.finish_from_output(data, &mut finished);
            state.finished_by_failure = finished;
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

        // An incomplete response is a stop, not a failure: the output cap, a content filter or a
        // reason Meka has no class for each end the reply where it stopped, with what streamed.
        "response.incomplete" => {
            state.finished = true;
            state.finish_from_output(data, &mut out);
            state.release_open(&mut out);
            let response = data.get("response").unwrap_or(&serde_json::Value::Null);
            if let Some(usage) = response.get("usage") {
                out.push(StreamEvent::Usage(usage_from(usage)));
            }
            let stated = response_fields(response, &state.restated);
            out.extend(state.fields_with(stated));
            out.push(StreamEvent::MessageEnd {
                stop: responses_stop(response, "incomplete"),
            });
            return Ok(out);
        }

        // What the part's or the item's done states whole again.
        "response.output_text.done"
        | "response.output_text.annotation.added"
        | "response.refusal.done"
        | "response.reasoning_text.done"
        | "response.reasoning_summary_text.done"
        | "response.function_call_arguments.done"
        | "response.custom_tool_call_input.done" => {}

        other => {
            tracing::debug!("Responses SSE event Meka reads nothing from: {other}");
            state.keep_event(data, &mut out);
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

/// How a response ended: its `status` as the word, classed by the reason `incomplete_details`
/// gives for an incomplete one, which is recorded beside it. `ended` is the status the terminal
/// event itself stands for, which classes a response that states none.
fn responses_stop(response: &serde_json::Value, ended: &str) -> Stop {
    let word = response.get("status").and_then(|status| status.as_str());
    let incomplete_details = response
        .get("incomplete_details")
        .filter(|details| !details.is_null());
    let reason = match word.unwrap_or(ended) {
        "completed" => StopReason::EndTurn,
        "incomplete" => match incomplete_details
            .and_then(|details| details.get("reason"))
            .and_then(|reason| reason.as_str())
        {
            Some("max_output_tokens") => StopReason::MaxTokens,
            Some("content_filter") => StopReason::ContentFilter,
            _ => StopReason::Unknown,
        },
        other => {
            tracing::warn!(
                "the Responses endpoint returned an unrecognized response status {other:?}; mapping to Unknown"
            );
            StopReason::Unknown
        }
    };
    Stop {
        reason,
        word: word.map(str::to_string),
        details: incomplete_details.map(|details| {
            serde_json::Map::from_iter([("incomplete_details".to_string(), details.clone())])
        }),
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
    drive_responses_sse_stream(
        response,
        backend.origin(),
        restated_fields(&body_json),
        event_sender,
        cancellation,
    )
    .await
}

/// The top-level fields of the serialized request `body`, which a response object restates.
fn restated_fields(body: &str) -> Vec<String> {
    match serde_json::from_str::<std::collections::BTreeMap<String, serde::de::IgnoredAny>>(body) {
        Ok(fields) => fields.into_keys().collect(),
        Err(error) => {
            tracing::warn!("failed to read the request's fields back: {error}");
            Vec::new()
        }
    }
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
    restated: Vec<String>,
    event_sender: mpsc::Sender<StreamEvent>,
    cancellation: CancellationToken,
) -> Result<()> {
    let mut protocol = ResponsesStream {
        state: SseState {
            origin,
            restated,
            ..SseState::default()
        },
    };
    let ended = match crate::provider::sse::drive(
        response,
        "Responses",
        &event_sender,
        &cancellation,
        &mut protocol,
    )
    .await
    {
        Ok(End::Finished | End::ReceiverGone) => Ok(()),
        // The stream ended without `response.completed`, `response.failed` or
        // `response.incomplete`. Falling through here would commit a truncated turn as a complete
        // one: the agent would see whatever text had arrived, write it to the conversation, and
        // move on, with the retry path never consulted. A connection cut mid-response is exactly
        // what that path exists for.
        Ok(End::Ended) => Err(crate::provider::sse::stream_error(
            &event_sender,
            "the Responses stream ended before a terminal response event".to_string(),
        )
        .await),
        Err(error) => Err(error),
    };
    // What the items still open hold is handed over after the failure is said, so it goes to the
    // attempt's partial reply and leaves the retry the failure allows.
    let mut held = std::mem::take(&mut protocol.state.finished_by_failure);
    protocol.state.release_open(&mut held);
    for event in held {
        if event_sender.send(event).await.is_err() {
            tracing::trace!("stream event receiver dropped");
            break;
        }
    }
    ended
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
        let data = crate::provider::sse::frame_json("Responses", &event.data)?;
        let events = match process_event(frame_name(&data, &event.event), &data, &mut self.state) {
            Ok(events) => events,
            Err(error) => {
                for told in std::mem::take(&mut self.state.told_before_failing) {
                    if event_sender.send(told).await.is_err() {
                        tracing::trace!("stream event receiver dropped");
                        break;
                    }
                }
                return Err(error);
            }
        };
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
    use crate::{conversation::ToolResultContent, error::MekaError};

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

    /// What Meka writes into a conversation reaches this wire rendered: the summary, a note behind
    /// Meka's label, a call's arguments as the model sent them, which the spec defines as a
    /// string, and a result's caption and large output's reference.
    #[test]
    fn what_meka_wrote_reaches_the_wire_rendered() {
        let body = build_request_body(
            "gpt-5",
            "",
            &Message::typed_content_for_test(),
            &[],
            None,
            None,
            true,
        );
        let sent = body["input"].to_string();
        for needle in [
            "the summary",
            "[Meka harness] the note",
            r#""arguments":"{\"path\": ""#,
            "[Meka harness] the schema note",
            "[Image: a.png]",
            "<large-output name=\\\"the_entry\\\"",
        ] {
            assert!(sent.contains(needle), "{needle} is not sent: {sent}");
        }
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
                    unparsed: None,
                    provider_fields: None,
                }],
                provider_fields: None,
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
                        display: None,
                    },
                    ContentBlock::Text {
                        text: "[Message from test, arrived now]\nalso, what is 17*3?".to_string(),
                        provider_fields: None,
                    },
                ],
                provider_fields: None,
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
                    unparsed: None,
                    provider_fields: None,
                }],
                provider_fields: None,
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_abc".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "contents".to_string(),
                    }],
                    is_error: false,
                    display: None,
                }],
                provider_fields: None,
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
                display: None,
            }],
            provider_fields: None,
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
    /// send both -- were the only endpoints Meka reached, and failed the moment an API-key backend
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
        let (emitted, _) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0, "item": {"type": "message"}}),
            ),
            (
                frame_name(&bare, ""),
                serde_json::json!({"output_index": 0, "content_index": 0, "delta": "hi"}),
            ),
        ]);
        assert!(
            matches!(emitted.as_slice(),
                [StreamEvent::ContentPart(0), StreamEvent::TextDelta(text)] if text == "hi"),
            "{emitted:?}"
        );
    }

    /// The message `events` fold to, as the turn folds them.
    fn folded(events: Vec<StreamEvent>) -> Message {
        let mut accumulator = crate::provider::MessageAccumulator::new();
        for event in events {
            accumulator.push(event);
        }
        accumulator.finish().message
    }

    /// Items that stream at once fold by their `output_index`, as the reference client folds
    /// them: each call keeps its own id, name and arguments, and the reply lays them out in index
    /// order.
    #[test]
    fn calls_that_stream_at_once_fold_by_their_output_index() {
        let call = |index: u64, id: &str, name: &str| {
            serde_json::json!({"output_index": index,
                "item": {"type": "function_call", "call_id": id, "name": name}})
        };
        let arguments =
            |index: u64, delta: &str| serde_json::json!({"output_index": index, "delta": delta});
        let (events, outcome) = run_events(&[
            ("response.output_item.added", call(0, "c0", "file_read")),
            ("response.output_item.added", call(1, "c1", "shell_execute")),
            (
                "response.function_call_arguments.delta",
                arguments(1, "{\"command\":"),
            ),
            (
                "response.function_call_arguments.delta",
                arguments(0, "{\"path\":"),
            ),
            (
                "response.function_call_arguments.delta",
                arguments(1, "\"ls\"}"),
            ),
            (
                "response.function_call_arguments.delta",
                arguments(0, "\"a\"}"),
            ),
            ("response.output_item.done", call(1, "c1", "shell_execute")),
            ("response.output_item.done", call(0, "c0", "file_read")),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ]);
        outcome.expect("clean stream");
        let message = folded(events);
        let calls: Vec<(&str, &str, &serde_json::Value)> = message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse {
                    id, name, input, ..
                } => Some((id.as_str(), name.as_str(), input)),
                _ => None,
            })
            .collect();
        assert_eq!(calls, [
            ("c0", "file_read", &serde_json::json!({"path": "a"})),
            ("c1", "shell_execute", &serde_json::json!({"command": "ls"})),
        ]);
    }

    /// A reasoning item and a message that stream at once each keep their own words, and the
    /// reply lays them out in index order.
    #[test]
    fn reasoning_and_a_message_that_stream_at_once_keep_their_own_words() {
        let (events, outcome) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0, "item": {"type": "reasoning", "id": "rs_1"}}),
            ),
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 1, "item": {"type": "message"}}),
            ),
            (
                "response.reasoning_summary_part.added",
                serde_json::json!({"output_index": 0, "summary_index": 0}),
            ),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 1, "content_index": 0, "delta": "The "}),
            ),
            (
                "response.reasoning_summary_text.delta",
                serde_json::json!({"output_index": 0, "summary_index": 0, "delta": "Plan "}),
            ),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 1, "content_index": 0, "delta": "answer."}),
            ),
            (
                "response.reasoning_summary_text.delta",
                serde_json::json!({"output_index": 0, "summary_index": 0, "delta": "first."}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 1, "item": {"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "The answer."}]}}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": {"type": "reasoning", "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "Plan first."}]}}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ]);
        outcome.expect("clean stream");
        let message = folded(events);
        let [
            ContentBlock::Thinking { parts, .. },
            ContentBlock::Text { text, .. },
        ] = message.content.as_slice()
        else {
            panic!("{:?}", message.content);
        };
        assert_eq!(crate::conversation::reasoning_text(parts), "Plan first.");
        assert_eq!(text, "The answer.");
    }

    /// An item's parts fold in their index order, as the reference client addresses them: a
    /// message's later part that streams while an earlier one is open follows it, and so does a
    /// reasoning summary's.
    #[test]
    fn an_item_s_parts_fold_in_their_index_order() {
        let text = |part: u64, delta: &str| serde_json::json!({"output_index": 0, "content_index": part, "delta": delta});
        let content = |part: u64| serde_json::json!({"output_index": 0, "content_index": part});
        let (events, outcome) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0, "item": {"type": "message"}}),
            ),
            ("response.content_part.added", content(0)),
            ("response.content_part.added", content(1)),
            ("response.output_text.delta", text(1, "world")),
            ("response.output_text.delta", text(0, "Hello ")),
            ("response.content_part.done", content(0)),
            ("response.content_part.done", content(1)),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": {"type": "message", "role": "assistant"}}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ]);
        outcome.expect("clean stream");
        let deltas: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::TextDelta(delta) => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas, ["Hello ", "world"]);

        let summary = |part: u64| serde_json::json!({"output_index": 0, "summary_index": part});
        let words = |part: u64, delta: &str| serde_json::json!({"output_index": 0, "summary_index": part, "delta": delta});
        let (events, outcome) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0, "item": {"type": "reasoning", "id": "rs_1"}}),
            ),
            ("response.reasoning_summary_part.added", summary(0)),
            ("response.reasoning_summary_part.added", summary(1)),
            ("response.reasoning_summary_text.delta", words(1, "Second")),
            ("response.reasoning_summary_text.delta", words(0, "First")),
            ("response.reasoning_summary_part.done", summary(0)),
            ("response.reasoning_summary_part.done", summary(1)),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": {"type": "reasoning", "id": "rs_1",
                    "encrypted_content": "E"}}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ]);
        outcome.expect("clean stream");
        let message = folded(events);
        let [ContentBlock::Thinking { parts, .. }] = message.content.as_slice() else {
            panic!("{:?}", message.content);
        };
        let parts: Vec<&str> = parts.iter().map(|part| part.text.as_str()).collect();
        assert_eq!(parts, ["First", "Second"]);
    }

    /// A reasoning item's summary and raw reasoning that stream at once each keep their own
    /// words: one part streams at a time, and the other follows when it ends.
    #[test]
    fn a_summary_and_raw_reasoning_that_stream_at_once_keep_their_own_words() {
        let (events, outcome) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0, "item": {"type": "reasoning", "id": "rs_1"}}),
            ),
            (
                "response.reasoning_summary_part.added",
                serde_json::json!({"output_index": 0, "summary_index": 0}),
            ),
            (
                "response.reasoning_text.delta",
                serde_json::json!({"output_index": 0, "content_index": 0, "delta": "raw"}),
            ),
            (
                "response.reasoning_summary_text.delta",
                serde_json::json!({"output_index": 0, "summary_index": 0, "delta": "summary"}),
            ),
            (
                "response.reasoning_summary_part.done",
                serde_json::json!({"output_index": 0, "summary_index": 0}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": {"type": "reasoning", "id": "rs_1",
                    "encrypted_content": "E"}}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ]);
        outcome.expect("clean stream");
        let message = folded(events);
        let [ContentBlock::Thinking { parts, .. }] = message.content.as_slice() else {
            panic!("{:?}", message.content);
        };
        let parts: Vec<(crate::conversation::ReasoningKind, &str)> = parts
            .iter()
            .map(|part| (part.kind, part.text.as_str()))
            .collect();
        assert_eq!(parts, [
            (crate::conversation::ReasoningKind::Summary, "summary"),
            (crate::conversation::ReasoningKind::Reasoning, "raw"),
        ]);
    }

    /// An event of a kind its item does not have, or one for a part already whole, is a shape
    /// the spec does not allow: kept as it came after the item, never read into it.
    #[test]
    fn an_event_its_item_or_part_does_not_take_is_kept_as_it_came() {
        let misplaced =
            serde_json::json!({"output_index": 0, "content_index": 0, "delta": "thought"});
        let late = serde_json::json!({"output_index": 0, "content_index": 0, "delta": " late"});
        let (events, outcome) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0, "item": {"type": "message"}}),
            ),
            ("response.reasoning_text.delta", misplaced.clone()),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 0, "content_index": 0, "delta": "words"}),
            ),
            (
                "response.content_part.done",
                serde_json::json!({"output_index": 0, "content_index": 0}),
            ),
            ("response.output_text.delta", late.clone()),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": {"type": "message", "role": "assistant"}}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ]);
        outcome.expect("clean stream");
        let message = folded(events);
        assert!(
            matches!(message.content.as_slice(), [
                ContentBlock::Text { text, .. },
                ContentBlock::Unrecognized { parts },
            ] if text == "words" && *parts == [misplaced.clone(), late.clone()]),
            "{:?}",
            message.content
        );
    }

    /// An item that streams none of its words states them whole in its done item, which the
    /// reply takes as the reference client takes it: a message's text and refusal, and a
    /// reasoning item's summary.
    #[test]
    fn an_item_that_arrives_whole_in_its_done_keeps_its_words() {
        let (events, outcome) = run_events(&[
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": {"type": "reasoning", "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "Plan."}]}}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 1, "item": {"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "Whole."}]}}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 2, "item": {"type": "message", "role": "assistant",
                    "content": [{"type": "refusal", "refusal": "Not that."}]}}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ]);
        outcome.expect("clean stream");
        let message = folded(events);
        let [
            ContentBlock::Thinking { parts, .. },
            ContentBlock::Text { text, .. },
            ContentBlock::Refusal { text: refusal, .. },
        ] = message.content.as_slice()
        else {
            panic!("{:?}", message.content);
        };
        assert_eq!(crate::conversation::reasoning_text(parts), "Plan.");
        assert_eq!(text, "Whole.");
        assert_eq!(refusal, "Not that.");
    }

    /// An event for an item that was never opened, or one naming no item, is a shape the spec
    /// does not allow: kept as it came, never read as the reply's words.
    #[test]
    fn an_event_for_no_open_item_is_kept_as_it_came() {
        let stray = serde_json::json!({"output_index": 3, "content_index": 0, "delta": "stray"});
        let unnamed = serde_json::json!({"content_index": 0, "delta": "unnamed"});
        let (events, outcome) = run_events(&[
            ("response.output_text.delta", stray.clone()),
            ("response.output_text.delta", unnamed.clone()),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ]);
        outcome.expect("clean stream");
        let message = folded(events);
        let mut kept = Vec::new();
        for block in &message.content {
            match block {
                ContentBlock::Unrecognized { parts } => kept.extend(parts.iter()),
                other => panic!("only what was kept: {other:?}"),
            }
        }
        assert_eq!(kept, [&unnamed, &stray]);
    }

    /// A second item opened at an index already open closes the first as one that never finished,
    /// so each keeps what it streamed: a call refused with the arguments it had, then the item that
    /// replaced it.
    #[test]
    fn a_repeated_output_index_closes_the_item_open_there() {
        let (events, outcome) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0,
                    "item": {"type": "function_call", "call_id": "c0", "name": "file_read"}}),
            ),
            (
                "response.function_call_arguments.delta",
                serde_json::json!({"output_index": 0, "delta": "{\"pa"}),
            ),
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0, "item": {"type": "message"}}),
            ),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 0, "content_index": 0, "delta": "words"}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": {"type": "message", "role": "assistant"}}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ]);
        outcome.expect("clean stream");
        let message = folded(events);
        assert!(
            matches!(message.content.as_slice(), [
                ContentBlock::ToolUse { id, unparsed: Some(unparsed), .. },
                ContentBlock::Text { text, .. },
            ] if id == "c0" && unparsed.text.as_deref() == Some("{\"pa") && text == "words"),
            "{:?}",
            message.content
        );
    }

    /// What an item holds when the stream breaks off, a part that waited on an earlier one among
    /// it, is handed over after the failure is said, so it reaches the attempt's partial reply
    /// and leaves the retry the failure allows.
    #[tokio::test]
    async fn what_an_open_item_holds_is_handed_over_after_a_cut() {
        let (events, outcome) = decode_sse(concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\
             \"item\":{\"type\":\"message\"}}\n\n",
            "data: {\"type\":\"response.content_part.added\",\"output_index\":0,\
             \"content_index\":0}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\
             \"content_index\":1,\"delta\":\"later\"}\n\n",
        ))
        .await;
        assert!(outcome.is_err(), "a stream that ends early fails");
        let order: Vec<String> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::Error(_) => Some("error".to_string()),
                StreamEvent::TextDelta(text) => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(order, ["error", "later"]);
    }

    #[test]
    fn process_event_text_delta() {
        let mut state = SseState::default();
        process_event(
            "response.output_item.added",
            &serde_json::json!({"output_index": 0, "item": {"type": "message"}}),
            &mut state,
        )
        .expect("ok");
        let events = process_event(
            "response.output_text.delta",
            &serde_json::json!({"output_index": 0, "content_index": 0, "delta": "hello"}),
            &mut state,
        )
        .expect("ok");
        assert!(
            matches!(events.as_slice(),
                [StreamEvent::ContentPart(0), StreamEvent::TextDelta(t)] if t == "hello"),
            "{events:?}"
        );
    }

    #[test]
    fn process_event_text_delta_empty_emits_nothing() {
        let mut state = SseState::default();
        process_event(
            "response.output_item.added",
            &serde_json::json!({"output_index": 0, "item": {"type": "message"}}),
            &mut state,
        )
        .expect("ok");
        let events = process_event(
            "response.output_text.delta",
            &serde_json::json!({"output_index": 0, "content_index": 0, "delta": ""}),
            &mut state,
        )
        .expect("ok");
        assert!(events.is_empty());
    }

    #[test]
    fn process_event_reasoning_delta_emits_thinking() {
        let mut state = SseState::default();
        let added = process_event(
            "response.output_item.added",
            &serde_json::json!({"output_index": 0, "item": {"type": "reasoning", "id": "rs_1"}}),
            &mut state,
        )
        .expect("ok");
        assert!(added.is_empty(), "{added:?}");
        let events = process_event(
            "response.reasoning_text.delta",
            &serde_json::json!({"output_index": 0, "content_index": 0, "delta": "hmm"}),
            &mut state,
        )
        .expect("ok");
        assert!(
            matches!(
                events.as_slice(),
                [
                    StreamEvent::ThinkingPart {
                        kind: crate::conversation::ReasoningKind::Reasoning,
                        number: Some(0),
                    },
                    StreamEvent::ThinkingDelta(text),
                ] if text == "hmm"
            ),
            "{events:?}"
        );
        assert!(matches!(
            state.items.get(&0),
            Some(OpenItem {
                kind: ItemKind::Reasoning { showed: true },
                ..
            })
        ));
    }

    #[test]
    fn process_event_tool_call_full_lifecycle() {
        let (events, outcome) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "file_read"
                    }
                }),
            ),
            (
                "response.function_call_arguments.delta",
                serde_json::json!({"output_index": 0, "delta": "{\"path\":"}),
            ),
            (
                "response.function_call_arguments.delta",
                serde_json::json!({"output_index": 0, "delta": "\"/tmp/x\"}"}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({
                    "output_index": 0,
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
        let deltas: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ToolArgumentsDelta(delta) => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas, ["{\"path\":", "\"/tmp/x\"}"]);
        match &events[3] {
            StreamEvent::ToolUseEnd { input, .. } => assert_eq!(input["path"], "/tmp/x"),
            other => panic!("expected ToolUseEnd, got {other:?}"),
        }
        // The Responses API reports `status: "completed"` even for tool-call turns; the presence of
        // the function call must still surface as `ToolUse`, not `EndTurn`.
        assert!(matches!(&events[4], StreamEvent::ResponseFields(fields) if fields["id"] == "r1"));
        assert!(matches!(events[5], StreamEvent::MessageEnd {
            stop: crate::provider::Stop {
                reason: StopReason::ToolUse,
                ..
            },
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
                StreamEvent::ToolUseEnd { input, .. } => input["path"].as_str(),
                _ => None,
            })
            .collect();
        assert_eq!(paths, vec!["/tmp/a", "/tmp/b"]);
        assert!(matches!(
            events.last(),
            Some(StreamEvent::MessageEnd {
                stop: crate::provider::Stop {
                    reason: StopReason::ToolUse,
                    ..
                },
            })
        ));
    }

    /// The events `frames` stream as through the driver, each named by its `type`.
    async fn responses_events(
        frames: &[(&str, serde_json::Value)],
    ) -> (Vec<StreamEvent>, Result<()>) {
        let body: String = frames
            .iter()
            .map(|(name, data)| {
                let mut data = data.clone();
                if let Some(object) = data.as_object_mut() {
                    object.insert("type".to_string(), serde_json::json!(name));
                }
                format!("event: {name}\ndata: {data}\n\n")
            })
            .collect();
        decode_sse(&body).await
    }

    /// The completion `events` fold to.
    fn completion_of(events: Vec<StreamEvent>) -> crate::provider::Completion {
        let mut accumulator = crate::provider::MessageAccumulator::new();
        for event in events {
            accumulator.push(event);
        }
        accumulator.finish()
    }

    /// A part that streams no words states them whole in its done, as the reference client takes
    /// the part: a message's second part after a first that streamed, a block of its own, and a
    /// summary part opened and ended with no deltas, which stays one part. The done item restating
    /// them otherwise changes nothing.
    #[tokio::test]
    async fn a_part_that_streamed_no_words_takes_them_from_its_done() {
        let text = |text: &str| serde_json::json!({"type": "output_text", "text": text});
        let summary = |text: &str| serde_json::json!({"type": "summary_text", "text": text});
        let (events, outcome) = responses_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0,
                "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
            ),
            (
                "response.reasoning_summary_part.added",
                serde_json::json!({"output_index": 0,
                "summary_index": 0, "part": summary("")}),
            ),
            (
                "response.reasoning_summary_part.done",
                serde_json::json!({"output_index": 0,
                "summary_index": 0, "part": summary("Whole summary")}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0,
                "item": {"type": "reasoning", "id": "rs_1",
                    "summary": [summary("Summary, restated")], "encrypted_content": "SEALED"}}),
            ),
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 1,
                "item": {"type": "message", "id": "m_1", "role": "assistant", "content": []}}),
            ),
            (
                "response.content_part.added",
                serde_json::json!({"output_index": 1,
                "content_index": 0, "part": text("")}),
            ),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 1,
                "content_index": 0, "delta": "Alpha."}),
            ),
            (
                "response.content_part.done",
                serde_json::json!({"output_index": 1,
                "content_index": 0, "part": text("Alpha.")}),
            ),
            (
                "response.content_part.added",
                serde_json::json!({"output_index": 1,
                "content_index": 1, "part": text("")}),
            ),
            (
                "response.content_part.done",
                serde_json::json!({"output_index": 1,
                "content_index": 1, "part": text("Beta.")}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 1,
                "item": {"type": "message", "id": "m_1", "role": "assistant",
                    "content": [text("Alpha."), text("Beta, restated.")]}}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ])
        .await;
        outcome.expect("clean stream");
        let content = completion_of(events).message.content;
        let [
            ContentBlock::Thinking { parts, .. },
            ContentBlock::Text { text: first, .. },
            ContentBlock::Text { text: second, .. },
        ] = content.as_slice()
        else {
            panic!("{content:?}");
        };
        assert_eq!(parts, &[crate::conversation::ReasoningPart::summary(
            "Whole summary"
        )]);
        assert_eq!((first.as_str(), second.as_str()), ("Alpha.", "Beta."));
    }

    /// A call the stream left open is finished as the terminal response's output states it, and
    /// an item only the output lists is not added beside what the stream sent.
    #[tokio::test]
    async fn the_terminal_output_finishes_what_the_stream_left() {
        let (events, outcome) = responses_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0,
                "item": {"type": "function_call", "call_id": "call_1", "name": "file_read",
                    "arguments": ""}}),
            ),
            (
                "response.function_call_arguments.delta",
                serde_json::json!({"output_index": 0,
                "delta": "{\"path\":"}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed",
                "output": [
                    {"type": "function_call", "call_id": "call_1", "name": "file_read",
                        "arguments": "{\"path\":\"a\"}"},
                    {"type": "message", "id": "msg_1", "role": "assistant",
                        "content": [{"type": "output_text", "text": "Only in completed"}]},
                ]}}),
            ),
        ])
        .await;
        outcome.expect("clean stream");
        let completion = completion_of(events);
        let [
            ContentBlock::ToolUse {
                id,
                input,
                unparsed: None,
                ..
            },
        ] = completion.message.content.as_slice()
        else {
            panic!("{:?}", completion.message.content);
        };
        assert_eq!(id, "call_1");
        assert_eq!(input, &serde_json::json!({"path": "a"}));
        assert_eq!(completion.stop.reason, StopReason::ToolUse);

        // An item that names nothing is the reply's when the stream sent no item at all.
        let (events, outcome) = responses_events(&[(
            "response.completed",
            serde_json::json!({"response": {"status": "completed", "output": [
                {"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "Whole"}]},
            ]}}),
        )])
        .await;
        outcome.expect("clean stream");
        let content = completion_of(events).message.content;
        assert!(
            matches!(content.as_slice(), [ContentBlock::Text { text, .. }] if text == "Whole"),
            "{content:?}"
        );
    }

    /// A stream that sent no item at all takes the terminal response's output as its reply, as
    /// the reference client takes the completed response: every item it lists, in its order,
    /// whatever it names itself.
    #[tokio::test]
    async fn a_stream_that_sent_no_item_takes_the_terminal_output_as_its_reply() {
        let (events, outcome) = responses_events(&[(
            "response.completed",
            serde_json::json!({"response": {"status": "completed", "output": [
                {"type": "reasoning", "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "First."}]},
                {"type": "message", "id": "msg_1", "role": "assistant",
                    "content": [{"type": "output_text", "text": "A."}]},
                {"type": "web_search_call", "id": "ws_1"},
                {"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "B."}]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                    "name": "file_read", "arguments": "{}"},
            ]}}),
        )])
        .await;
        outcome.expect("clean stream");
        let completion = completion_of(events);
        let laid_out: Vec<String> = completion
            .message
            .content
            .iter()
            .map(|block| match block {
                ContentBlock::Thinking { parts, .. } => crate::conversation::reasoning_text(parts),
                ContentBlock::Text { text, .. } => text.clone(),
                ContentBlock::Unrecognized { parts } => parts
                    .iter()
                    .map(|part| part["id"].as_str().unwrap_or_default().to_string())
                    .collect(),
                ContentBlock::ToolUse { id, .. } => id.clone(),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(laid_out, ["First.", "A.", "ws_1", "B.", "call_1"]);
        assert_eq!(completion.stop.reason, StopReason::ToolUse);
    }

    /// What the stream sent is the reply, so an item it sent is recorded once however it
    /// numbered it: from one rather than zero, or at an index a finished item used before. An item
    /// the output lists and the stream left out is not added beside what it sent. A part that
    /// streamed and states its words again at its end is recorded once too.
    #[tokio::test]
    async fn the_terminal_output_records_each_streamed_item_once() {
        let call = serde_json::json!({"type": "function_call", "id": "fc_1", "call_id": "call_1",
            "name": "file_read", "arguments": "{}"});
        let message = serde_json::json!({"type": "message", "id": "msg_1", "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello."}]});
        let reasoning = serde_json::json!({"type": "reasoning", "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Plan."}]});
        let unnamed = serde_json::json!({"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello."}]});
        let streamed = |index: u64, item: &serde_json::Value| {
            vec![
                (
                    "response.output_item.added",
                    serde_json::json!({"output_index": index,
                    "item": item}),
                ),
                (
                    "response.output_item.done",
                    serde_json::json!({"output_index": index,
                    "item": item}),
                ),
            ]
        };
        let completed = |output: serde_json::Value| {
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed",
                "output": output}}),
            )
        };
        // Each shape with the calls, texts and reasoning it must come to, no more and no fewer.
        let shapes = [
            // Numbered from one.
            (
                [streamed(1, &call), vec![completed(serde_json::json!([
                    call
                ]))]]
                .concat(),
                (1, 0, 0),
            ),
            // A finished message's index reused by the call after it.
            (
                [streamed(0, &message), streamed(0, &call), vec![completed(
                    serde_json::json!([message, call]),
                )]]
                .concat(),
                (1, 1, 0),
            ),
            // A message that names nothing, streamed and listed again.
            (
                [streamed(0, &unnamed), vec![completed(serde_json::json!([
                    unnamed
                ]))]]
                .concat(),
                (0, 1, 0),
            ),
            // The output lists reasoning the stream left out, ahead of what it sent.
            (
                [streamed(0, &message), vec![completed(serde_json::json!([
                    reasoning, message
                ]))]]
                .concat(),
                (0, 1, 0),
            ),
        ];
        for (frames, expected) in shapes {
            let (events, outcome) = responses_events(&frames).await;
            outcome.expect("clean stream");
            let content = completion_of(events).message.content;
            let count = |wanted: fn(&ContentBlock) -> bool| {
                content.iter().filter(|block| wanted(block)).count()
            };
            assert_eq!(
                (
                    count(|block| matches!(block, ContentBlock::ToolUse { .. })),
                    count(|block| matches!(block, ContentBlock::Text { .. })),
                    count(|block| matches!(block, ContentBlock::Thinking { .. })),
                ),
                expected,
                "{content:?}"
            );
        }

        let (events, outcome) = responses_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0,
                "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
            ),
            (
                "response.reasoning_summary_part.added",
                serde_json::json!({"output_index": 0,
                "summary_index": 0, "part": {"type": "summary_text", "text": ""}}),
            ),
            (
                "response.reasoning_summary_text.delta",
                serde_json::json!({"output_index": 0,
                "summary_index": 0, "delta": "Plan."}),
            ),
            (
                "response.reasoning_summary_part.done",
                serde_json::json!({"output_index": 0,
                "summary_index": 0, "part": {"type": "summary_text", "text": "Plan."}}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0,
                "item": reasoning}),
            ),
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 1,
                "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []}}),
            ),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 1,
                "content_index": 0, "delta": "Hello."}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 1,
                "item": message}),
            ),
            completed(serde_json::json!([reasoning, message])),
        ])
        .await;
        outcome.expect("clean stream");
        let content = completion_of(events).message.content;
        let [
            ContentBlock::Thinking { parts, .. },
            ContentBlock::Text { text, .. },
        ] = content.as_slice()
        else {
            panic!("{content:?}");
        };
        assert_eq!(parts, &[crate::conversation::ReasoningPart::summary(
            "Plan."
        )]);
        assert_eq!(text, "Hello.");
    }

    /// A server that restates what it streamed under new names, as vLLM's did, gets each item
    /// once: the stream's, with the call the stream named, so the call runs once.
    #[tokio::test]
    async fn a_terminal_output_that_renames_the_streamed_items_adds_none_of_them() {
        let reasoning = |id: &str| {
            serde_json::json!({"type": "reasoning", "id": id,
                "summary": [{"type": "summary_text", "text": "Plan."}]})
        };
        let message = |id: &str| {
            serde_json::json!({"type": "message", "id": id, "role": "assistant",
                "content": [{"type": "output_text", "text": "Reading it."}]})
        };
        let call = |id: &str, call_id: &str| {
            serde_json::json!({"type": "function_call", "id": id, "call_id": call_id,
                "name": "file_read", "arguments": "{\"path\":\"a\"}"})
        };
        let streamed = [reasoning("rs_a"), message("msg_a"), call("fc_a", "call_a")];
        let mut frames: Vec<(&str, serde_json::Value)> = (0_u64..)
            .zip(&streamed)
            .flat_map(|(index, item)| {
                [
                    (
                        "response.output_item.added",
                        serde_json::json!({"output_index": index, "item": item}),
                    ),
                    (
                        "response.output_item.done",
                        serde_json::json!({"output_index": index, "item": item}),
                    ),
                ]
            })
            .collect();
        frames.push((
            "response.completed",
            serde_json::json!({"response": {"status": "completed", "output": [
                reasoning("rs_b"), message("msg_b"), call("fc_b", "call_b"),
            ]}}),
        ));
        let (events, outcome) = responses_events(&frames).await;
        outcome.expect("clean stream");
        let completion = completion_of(events);
        let [
            ContentBlock::Thinking { .. },
            ContentBlock::Text { text, .. },
            ContentBlock::ToolUse { id, .. },
        ] = completion.message.content.as_slice()
        else {
            panic!("{:?}", completion.message.content);
        };
        assert_eq!(text, "Reading it.");
        assert_eq!(id, "call_a");
        assert_eq!(completion.stop.reason, StopReason::ToolUse);
    }

    /// A call is found in the terminal output by its `call_id` alone, and any other item by its
    /// `id` alone, since an endpoint may give reasoning an `id` equal to a call's `call_id`, as
    /// OpenRouter's Gemini route does. A call left open is finished as the output states it, not
    /// as the reasoning sharing its name; a call only the output lists is not added beside what
    /// the stream sent.
    #[tokio::test]
    async fn a_call_and_an_item_sharing_a_name_are_never_taken_for_each_other() {
        let reasoning = serde_json::json!({"type": "reasoning", "id": "call_7",
            "summary": [], "encrypted_content": "SEALED"});
        let call = serde_json::json!({"type": "function_call", "id": "fc_tmp", "call_id": "call_7",
            "name": "file_read", "arguments": "{\"path\":\"note.txt\"}"});
        let reasoning_streamed = [
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0,
                "item": {"type": "reasoning", "id": "call_7", "summary": []}}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": reasoning}),
            ),
        ];

        let mut frames = reasoning_streamed.to_vec();
        frames.extend([
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 1,
                "item": {"type": "function_call", "id": "fc_tmp", "call_id": "call_7",
                    "name": "file_read", "arguments": ""}}),
            ),
            (
                "response.function_call_arguments.delta",
                serde_json::json!({"output_index": 1, "delta": "{\"path\":"}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed",
                "output": [reasoning, call]}}),
            ),
        ]);
        let (events, outcome) = responses_events(&frames).await;
        outcome.expect("clean stream");
        let completion = completion_of(events);
        let [
            ContentBlock::Thinking { .. },
            ContentBlock::ToolUse {
                id,
                input,
                unparsed: None,
                ..
            },
        ] = completion.message.content.as_slice()
        else {
            panic!("{:?}", completion.message.content);
        };
        assert_eq!(id, "call_7");
        assert_eq!(input, &serde_json::json!({"path": "note.txt"}));
        assert_eq!(completion.stop.reason, StopReason::ToolUse);

        let mut frames = reasoning_streamed.to_vec();
        frames.push((
            "response.completed",
            serde_json::json!({"response": {"status": "completed",
            "output": [reasoning, call]}}),
        ));
        let (events, outcome) = responses_events(&frames).await;
        outcome.expect("clean stream");
        let completion = completion_of(events);
        assert!(
            matches!(completion.message.content.as_slice(), [
                ContentBlock::Thinking { .. }
            ]),
            "{:?}",
            completion.message.content
        );
        assert_eq!(completion.stop.reason, StopReason::EndTurn);
    }

    /// Every terminal response finishes the reply from its output, not the completed one alone:
    /// an incomplete response's output is the reply of a stream that sent no item, and an item
    /// left open takes the sealed reasoning its output entry states, on an incomplete response and
    /// in the partial reply of a failed one.
    #[tokio::test]
    async fn an_incomplete_or_failed_response_finishes_the_reply_from_its_output() {
        let (events, outcome) = responses_events(&[(
            "response.incomplete",
            serde_json::json!({"response": {"status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [{"type": "message", "id": "msg_1", "role": "assistant",
                    "content": [{"type": "output_text", "text": "Partial answer"}]}]}}),
        )])
        .await;
        outcome.expect("a stop, not a failure");
        let completion = completion_of(events);
        assert!(
            matches!(completion.message.content.as_slice(),
                [ContentBlock::Text { text, .. }] if text == "Partial answer"),
            "{:?}",
            completion.message.content
        );
        assert_eq!(completion.stop.reason, StopReason::MaxTokens);

        let open_reasoning = [
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0,
                "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
            ),
            (
                "response.reasoning_summary_text.delta",
                serde_json::json!({"output_index": 0, "summary_index": 0, "delta": "Plan."}),
            ),
        ];
        let sealed = serde_json::json!([{"type": "reasoning", "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Plan."}],
            "encrypted_content": "SEALED"}]);
        let is_sealed = |content: &[ContentBlock]| {
            matches!(content, [ContentBlock::Thinking {
                opaque: Some(OpaqueReasoning::Sealed { encrypted_content, id, .. }),
                ..
            }] if encrypted_content == "SEALED" && id.as_deref() == Some("rs_1"))
        };

        let mut frames = open_reasoning.to_vec();
        frames.push((
            "response.incomplete",
            serde_json::json!({"response": {"status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"}, "output": sealed}}),
        ));
        let (events, outcome) = responses_events(&frames).await;
        outcome.expect("a stop, not a failure");
        let content = completion_of(events).message.content;
        assert!(is_sealed(&content), "{content:?}");

        let mut frames = open_reasoning.to_vec();
        frames.push((
            "response.failed",
            serde_json::json!({"response": {"status": "failed",
                "error": {"code": "server_error", "message": "boom"}, "output": sealed}}),
        ));
        let (events, outcome) = responses_events(&frames).await;
        assert!(outcome.is_err(), "{outcome:?}");
        let mut accumulator = crate::provider::MessageAccumulator::new();
        for event in events {
            accumulator.push(event);
        }
        let partial = accumulator.partial().expect("what arrived");
        assert!(
            is_sealed(&partial.message.content),
            "{:?}",
            partial.message.content
        );
    }

    /// A name the stream gave two items finishes neither from the terminal output, whose entries
    /// under that name cannot be told apart: each keeps what it streamed, rather than the one left
    /// open taking the first entry's words.
    #[tokio::test]
    async fn a_name_two_items_share_finishes_neither_from_the_output() {
        let first = serde_json::json!({"type": "message", "id": "msg_same", "role": "assistant",
            "phase": "commentary", "content": [{"type": "output_text", "text": "First"}]});
        let second = serde_json::json!({"type": "message", "id": "msg_same", "role": "assistant",
            "phase": "final_answer", "content": [{"type": "output_text", "text": "Second"}]});
        let (events, outcome) = responses_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0, "item": first}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": first}),
            ),
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 1, "item": {"type": "message",
                    "id": "msg_same", "role": "assistant", "content": []}}),
            ),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 1, "content_index": 0, "delta": "Second"}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed",
                    "output": [first, second]}}),
            ),
        ])
        .await;
        outcome.expect("clean stream");
        let content = completion_of(events).message.content;
        let texts: Vec<(&str, Option<&serde_json::Value>)> = content
            .iter()
            .map(|block| match block {
                ContentBlock::Text {
                    text,
                    provider_fields,
                } => (
                    text.as_str(),
                    provider_fields
                        .as_ref()
                        .and_then(|fields| fields.get("phase")),
                ),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(texts, [
            ("First", Some(&serde_json::json!("commentary"))),
            ("Second", None)
        ]);
    }

    /// Parts keep their order when one arrives whole at its end before the next one streams, and
    /// a reasoning part only the done item states is kept after the parts that streamed.
    #[tokio::test]
    async fn parts_arriving_whole_keep_their_order_and_their_words() {
        let text = |text: &str| serde_json::json!({"type": "output_text", "text": text});
        let summary = |text: &str| serde_json::json!({"type": "summary_text", "text": text});
        let (events, outcome) = responses_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0,
                "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
            ),
            (
                "response.reasoning_summary_part.added",
                serde_json::json!({"output_index": 0,
                "summary_index": 0, "part": summary("")}),
            ),
            (
                "response.reasoning_summary_text.delta",
                serde_json::json!({"output_index": 0,
                "summary_index": 0, "delta": "First."}),
            ),
            (
                "response.reasoning_summary_part.done",
                serde_json::json!({"output_index": 0,
                "summary_index": 0, "part": summary("First.")}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0,
                "item": {"type": "reasoning", "id": "rs_1",
                    "summary": [summary("First."), summary("Second.")]}}),
            ),
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 1,
                "item": {"type": "message", "id": "m_1", "role": "assistant", "content": []}}),
            ),
            (
                "response.content_part.added",
                serde_json::json!({"output_index": 1,
                "content_index": 0, "part": text("")}),
            ),
            (
                "response.content_part.done",
                serde_json::json!({"output_index": 1,
                "content_index": 0, "part": text("Alpha.")}),
            ),
            (
                "response.content_part.added",
                serde_json::json!({"output_index": 1,
                "content_index": 1, "part": text("")}),
            ),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 1,
                "content_index": 1, "delta": "Beta."}),
            ),
            (
                "response.content_part.done",
                serde_json::json!({"output_index": 1,
                "content_index": 1, "part": text("Beta.")}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 1,
                "item": {"type": "message", "id": "m_1", "role": "assistant",
                    "content": [text("Alpha."), text("Beta.")]}}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ])
        .await;
        outcome.expect("clean stream");
        let content = completion_of(events).message.content;
        let [
            ContentBlock::Thinking { parts, .. },
            ContentBlock::Text { text: first, .. },
            ContentBlock::Text { text: second, .. },
        ] = content.as_slice()
        else {
            panic!("{content:?}");
        };
        assert_eq!(parts, &[
            crate::conversation::ReasoningPart::summary("First."),
            crate::conversation::ReasoningPart::summary("Second."),
        ]);
        assert_eq!((first.as_str(), second.as_str()), ("Alpha.", "Beta."));
    }

    /// Several events of the response itself of one type are all kept, as a list, in the order
    /// they came.
    #[test]
    fn repeated_response_events_are_all_kept() {
        let mut state = SseState::default();
        let first = serde_json::json!({"type": "response.metadata", "metadata": {"n": 1}});
        let second = serde_json::json!({"type": "response.metadata", "metadata": {"n": 2}});
        process_event("response.metadata", &first, &mut state).expect("ok");
        let events = process_event("response.metadata", &second, &mut state).expect("ok");
        assert!(
            matches!(events.as_slice(), [StreamEvent::ResponseFields(fields)]
                if fields.get("response.metadata") == Some(&serde_json::json!([first, second]))),
            "{events:?}"
        );
    }

    /// A part the stream never mentioned is stated whole by its item's done, in its place among
    /// the parts that received words, whether they streamed or arrived whole at their own end,
    /// which keep those words whatever the done item restates.
    #[tokio::test]
    async fn a_part_only_the_done_item_states_is_kept() {
        let text = |text: &str| serde_json::json!({"type": "output_text", "text": text});
        let (events, outcome) = responses_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0,
                "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []}}),
            ),
            (
                "response.content_part.added",
                serde_json::json!({"output_index": 0,
                "content_index": 0, "part": text("")}),
            ),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 0,
                "content_index": 0, "delta": "One."}),
            ),
            (
                "response.content_part.done",
                serde_json::json!({"output_index": 0,
                "content_index": 0, "part": text("One.")}),
            ),
            (
                "response.content_part.added",
                serde_json::json!({"output_index": 0,
                "content_index": 1, "part": text("")}),
            ),
            (
                "response.content_part.done",
                serde_json::json!({"output_index": 0,
                "content_index": 1, "part": text("Two.")}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0,
                "item": {"type": "message", "id": "msg_1", "role": "assistant",
                    "content": [text("One, restated."), text("Two, restated."), text("Three.")]}}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ])
        .await;
        outcome.expect("clean stream");
        let content = completion_of(events).message.content;
        let texts: Vec<&str> = content
            .iter()
            .map(|block| match block {
                ContentBlock::Text { text, .. } => text.as_str(),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(texts, ["One.", "Two.", "Three."]);
    }

    /// A call is the one the item that finishes it names, as the reference client records the
    /// done item.
    #[tokio::test]
    async fn a_call_is_the_one_its_done_item_names() {
        let (events, outcome) = responses_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0,
                "item": {"type": "function_call", "call_id": "call_a", "name": "file_read",
                    "arguments": ""}}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0,
                "item": {"type": "function_call", "call_id": "call_b", "name": "file_find",
                    "arguments": "{\"x\":1}"}}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ])
        .await;
        outcome.expect("clean stream");
        let content = completion_of(events).message.content;
        let [
            ContentBlock::ToolUse {
                id, name, input, ..
            },
        ] = content.as_slice()
        else {
            panic!("{content:?}");
        };
        assert_eq!((id.as_str(), name.as_str()), ("call_b", "file_find"));
        assert_eq!(input, &serde_json::json!({"x": 1}));
    }

    /// A failed response states its usage and fields, which the partial reply keeps.
    #[tokio::test]
    async fn a_failed_response_keeps_its_usage_and_fields() {
        let (events, outcome) = responses_events(&[
            (
                "response.created",
                serde_json::json!({"response": {"id": "resp_1",
                "usage": null}}),
            ),
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0,
                "item": {"type": "message", "role": "assistant", "content": []}}),
            ),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 0,
                "content_index": 0, "delta": "partial"}),
            ),
            (
                "response.failed",
                serde_json::json!({"response": {"id": "resp_1",
                "status": "failed", "error": {"code": "server_error", "message": "boom"},
                "usage": {"input_tokens": 10, "output_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 4}}}}),
            ),
        ])
        .await;
        assert!(outcome.is_err(), "{outcome:?}");
        let mut accumulator = crate::provider::MessageAccumulator::new();
        for event in events {
            accumulator.push(event);
        }
        let partial = accumulator.partial().expect("what arrived");
        assert_eq!(
            (
                partial.usage.input_tokens,
                partial.usage.cache_read_input_tokens,
                partial.usage.output_tokens
            ),
            (6, 4, 2)
        );
        let fields = partial.response_fields.expect("the response's fields");
        assert_eq!(fields["usage"]["input_tokens"], 10);
    }

    /// A message item of several parts is a block per part, each holding its own part's fields
    /// beside its item's, and goes back as the one item it was, its parts in it as they came:
    /// each text part with its own annotations, the refusal as a refusal, under the item's phase.
    #[tokio::test]
    async fn a_message_of_several_parts_is_a_block_per_part_and_goes_back_as_one_item() {
        let cited = |text: &str, url: &str| {
            serde_json::json!({"type": "output_text", "text": text,
                "annotations": [{"type": "url_citation", "url": url}]})
        };
        let item = serde_json::json!({"type": "message", "id": "msg_1", "role": "assistant",
        "phase": "final_answer", "content": [
            cited("First.", "https://a.example"),
            cited("Second.", "https://b.example"),
            {"type": "refusal", "refusal": "Not that."},
        ]});
        let (events, outcome) = responses_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0, "item": {"type": "message", "id": "msg_1",
                    "role": "assistant", "phase": "final_answer", "content": []}}),
            ),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 0, "content_index": 0, "delta": "First."}),
            ),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 0, "content_index": 1, "delta": "Second."}),
            ),
            (
                "response.refusal.delta",
                serde_json::json!({"output_index": 0, "content_index": 2, "delta": "Not that."}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": item}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ])
        .await;
        outcome.expect("clean stream");
        let content = completion_of(events).message.content;
        let urls: Vec<(&str, Option<&serde_json::Value>)> = content
            .iter()
            .map(|block| match block {
                ContentBlock::Text {
                    text,
                    provider_fields,
                }
                | ContentBlock::Refusal {
                    text,
                    provider_fields,
                } => {
                    let fields = provider_fields.as_ref().expect("the item's fields");
                    assert_eq!(fields["phase"], "final_answer", "{text}");
                    (
                        text.as_str(),
                        fields
                            .get("content")
                            .and_then(|parts| parts.get(0))
                            .and_then(|part| part.get("annotations"))
                            .and_then(|annotations| annotations.get(0))
                            .and_then(|annotation| annotation.get("url")),
                    )
                }
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(urls, [
            ("First.", Some(&serde_json::json!("https://a.example"))),
            ("Second.", Some(&serde_json::json!("https://b.example"))),
            ("Not that.", None),
        ]);

        let input = input_of(&[Message::user("hi"), assistant(content)]);
        let replayed: Vec<&serde_json::Value> = input
            .iter()
            .filter(|item| item["type"] == "message" && item["role"] == "assistant")
            .collect();
        let [replayed] = replayed.as_slice() else {
            panic!("one item: {input:?}");
        };
        assert_eq!(replayed["phase"], "final_answer");
        assert_eq!(
            replayed["content"],
            serde_json::json!([
                cited("First.", "https://a.example"),
                cited("Second.", "https://b.example"),
                {"type": "refusal", "refusal": "Not that."},
            ])
        );
    }

    /// An item's parts are kept in the order their numbers give them, as its done item lists
    /// them, when one only the done item states comes before one that streamed: a message's,
    /// which goes back in that order, and a reasoning summary's.
    #[tokio::test]
    async fn parts_keep_the_order_of_their_numbers_however_they_arrive() {
        let text = |text: &str| serde_json::json!({"type": "output_text", "text": text});
        let summary = |text: &str| serde_json::json!({"type": "summary_text", "text": text});
        let (events, outcome) = responses_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0,
                "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
            ),
            (
                "response.reasoning_summary_text.delta",
                serde_json::json!({"output_index": 0, "summary_index": 1, "delta": "Second."}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": {"type": "reasoning", "id": "rs_1",
                    "summary": [summary("First."), summary("Second.")]}}),
            ),
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 1, "item": {"type": "message", "id": "m_1",
                    "role": "assistant", "content": []}}),
            ),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 1, "content_index": 1, "delta": "B."}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 1, "item": {"type": "message", "id": "m_1",
                    "role": "assistant", "content": [text("A."), text("B.")]}}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ])
        .await;
        outcome.expect("clean stream");
        let content = completion_of(events).message.content;
        let [
            ContentBlock::Thinking { parts, .. },
            ContentBlock::Text { text: first, .. },
            ContentBlock::Text { text: second, .. },
        ] = content.as_slice()
        else {
            panic!("{content:?}");
        };
        assert_eq!(parts, &[
            crate::conversation::ReasoningPart::summary("First."),
            crate::conversation::ReasoningPart::summary("Second."),
        ]);
        assert_eq!((first.as_str(), second.as_str()), ("A.", "B."));
        let input = input_of(&[Message::user("hi"), assistant(content.clone())]);
        let replayed = input
            .iter()
            .find(|item| item["type"] == "message" && item["role"] == "assistant")
            .unwrap_or_else(|| panic!("{input:?}"));
        assert_eq!(
            replayed["content"],
            serde_json::json!([
                {"type": "output_text", "text": "A."},
                {"type": "output_text", "text": "B."},
            ])
        );
    }

    /// An item of one part keeps its fields as it returned them, an empty `content` list
    /// included; in an item of several, a part with no words is a block of its own, which keeps
    /// its own part's fields, between the parts around it.
    #[tokio::test]
    async fn every_part_keeps_its_own_fields() {
        let added = |index: u64| {
            (
                "response.output_item.added",
                serde_json::json!({"output_index": index, "item": {"type": "message",
                    "id": format!("m_{index}"), "role": "assistant", "content": []}}),
            )
        };
        let delta = |index: u64, words: &str| {
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": index, "content_index": 0, "delta": words}),
            )
        };
        let (events, outcome) = responses_events(&[
            added(0),
            delta(0, "Hi."),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": {"type": "message", "id": "m_0",
                    "role": "assistant", "status": "completed", "content": []}}),
            ),
            added(1),
            delta(1, "A."),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 1, "item": {"type": "message", "id": "m_1",
                "role": "assistant", "content": [
                    {"type": "output_text", "text": "A."},
                    {"type": "output_text", "text": "",
                        "annotations": [{"type": "url_citation", "url": "https://b.example"}]},
                    {"type": "output_text", "text": "C."},
                ]}}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ])
        .await;
        outcome.expect("clean stream");
        let content = completion_of(events).message.content;
        let fields: Vec<(&str, Option<serde_json::Value>)> = content
            .iter()
            .map(|block| match block {
                ContentBlock::Text {
                    text,
                    provider_fields,
                } => (
                    text.as_str(),
                    provider_fields
                        .as_ref()
                        .and_then(|fields| fields.get("content").cloned()),
                ),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(fields, [
            ("Hi.", Some(serde_json::json!([]))),
            ("A.", Some(serde_json::json!([{}]))),
            (
                "",
                Some(serde_json::json!([{
                    "annotations": [{"type": "url_citation", "url": "https://b.example"}]
                }]))
            ),
            ("C.", Some(serde_json::json!([{}]))),
        ]);
    }

    /// A message part of a type the spec does not define is kept as it came, after its item's
    /// parts, and a refusal part with no words is still a refusal.
    #[tokio::test]
    async fn a_part_of_no_known_type_is_kept_and_an_empty_refusal_stays_one() {
        let audio = serde_json::json!({"type": "output_audio", "transcript": "x", "data": "zzz"});
        let (events, outcome) = responses_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0, "item": {"type": "message", "id": "m_1",
                    "role": "assistant", "content": []}}),
            ),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 0, "content_index": 0, "delta": "A."}),
            ),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 0, "content_index": 2, "delta": "B."}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": {"type": "message", "id": "m_1",
                "role": "assistant", "content": [
                    {"type": "output_text", "text": "A."},
                    audio,
                    {"type": "output_text", "text": "B."},
                    {"type": "refusal", "refusal": ""},
                ]}}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ])
        .await;
        outcome.expect("clean stream");
        let content = completion_of(events).message.content;
        let [
            ContentBlock::Text { text: first, .. },
            ContentBlock::Text { text: second, .. },
            ContentBlock::Refusal { text: refused, .. },
            ContentBlock::Unrecognized { parts },
        ] = content.as_slice()
        else {
            panic!("{content:?}");
        };
        assert_eq!((first.as_str(), second.as_str()), ("A.", "B."));
        assert!(refused.is_empty());
        assert_eq!(parts, &[audio]);
    }

    /// A thinking block is shown whole in the order it is recorded in, its parts by their
    /// numbers, however they arrived.
    #[tokio::test]
    async fn a_thinking_block_is_shown_in_the_order_it_is_recorded() {
        let summary = |text: &str| serde_json::json!({"type": "summary_text", "text": text});
        let (events, outcome) = responses_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0,
                "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
            ),
            (
                "response.reasoning_summary_part.added",
                serde_json::json!({"output_index": 0, "summary_index": 0, "part": summary("")}),
            ),
            (
                "response.reasoning_summary_part.done",
                serde_json::json!({"output_index": 0, "summary_index": 0, "part": summary("")}),
            ),
            (
                "response.reasoning_summary_text.delta",
                serde_json::json!({"output_index": 0, "summary_index": 1, "delta": "Second."}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": {"type": "reasoning", "id": "rs_1",
                    "summary": [summary("First."), summary("Second.")]}}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ])
        .await;
        outcome.expect("clean stream");
        let mut accumulator = crate::provider::MessageAccumulator::new();
        let mut shown = None;
        for event in events {
            if matches!(event, StreamEvent::ThinkingComplete { .. }) {
                shown = Some(accumulator.pending_thinking());
            }
            accumulator.push(event);
        }
        let content = accumulator.finish().message.content;
        let [ContentBlock::Thinking { parts, .. }] = content.as_slice() else {
            panic!("{content:?}");
        };
        assert_eq!(parts, &[
            crate::conversation::ReasoningPart::summary("First."),
            crate::conversation::ReasoningPart::summary("Second."),
        ]);
        assert_eq!(shown, Some(crate::conversation::reasoning_text(parts)));
    }

    /// A refusal goes back with the `phase` its message item carried, as text does.
    #[test]
    fn a_refusal_goes_back_with_its_phase() {
        let input = input_of(&[
            Message::user("hi"),
            assistant(vec![ContentBlock::Refusal {
                text: "no".to_string(),
                provider_fields: Some(serde_json::Map::from_iter([(
                    "phase".to_string(),
                    serde_json::json!("final_answer"),
                )])),
            }]),
        ]);
        let refusal = input
            .iter()
            .find(|item| item["type"] == "message" && item["role"] == "assistant")
            .unwrap_or_else(|| panic!("{input:?}"));
        assert_eq!(refusal["phase"], "final_answer");
        assert_eq!(refusal["content"][0]["refusal"], "no");
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
        let (sender, mut receiver) = mpsc::channel(1024);
        let outcome = drive_responses_sse_stream(
            response,
            Some(CHATGPT_ORIGIN.to_string()),
            Vec::new(),
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
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\
             \"item\":{\"type\":\"message\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\
             \"content_index\":0,\"delta\":\"PO\"}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\
             \"content_index\":0,\"delta\":\"NG\"}\n\n",
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
                    "output_index": 0,
                    "item": {"type": "function_call", "call_id": "c1", "name": "file_write"}
                }),
            ),
            (
                "response.function_call_arguments.delta",
                serde_json::json!({"output_index": 0, "delta": "{\"path\": "}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({
                    "output_index": 0,
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
                stop: crate::provider::Stop {
                    reason: StopReason::EndTurn,
                    ..
                },
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
                    "output_index": 0,
                    "item": {"type": "function_call", "call_id": "c1", "name": "x"}
                }),
            ),
            (
                "response.output_item.done",
                serde_json::json!({
                    "output_index": 0,
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
                StreamEvent::ToolUseEnd { input, .. } => Some(input),
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
                    display: None,
                }],
                provider_fields: None,
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
            body.contains(crate::conversation::BUDGET_REDACTION_NOTE),
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

    /// The output cap ends the message as `MaxTokens`, with its usage, the way the Claude and Chat
    /// Completions drivers report `max_tokens` and `length`: it is a stop, not a failure, so the
    /// blocking path keeps the answer.
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
                                StreamEvent::ResponseFields(fields),
                                StreamEvent::MessageEnd {
             stop: crate::provider::Stop { reason: StopReason::MaxTokens, .. },
            }
                            ] if usage.output_tokens == 512
                                && fields["usage"]["output_tokens"] == 512
                                && !fields.contains_key("incomplete_details")
                        ),
            "{events:?}"
        );
    }

    /// A content filter ends the reply where it stopped, as a stop of its own class with the
    /// details recorded, rather than failing the turn.
    #[test]
    fn process_event_incomplete_for_a_content_filter_is_a_stop() {
        let mut state = SseState::default();
        let events = process_event(
            "response.incomplete",
            &serde_json::json!({
                "response": {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "content_filter"}
                }
            }),
            &mut state,
        )
        .expect("a stop, not a failure");
        assert!(state.finished);
        let [StreamEvent::MessageEnd { stop }] = events.as_slice() else {
            panic!("{events:?}");
        };
        assert_eq!(stop.reason, StopReason::ContentFilter);
        assert_eq!(stop.word.as_deref(), Some("incomplete"));
        assert_eq!(
            stop.details,
            Some(serde_json::Map::from_iter([(
                "incomplete_details".to_string(),
                serde_json::json!({"reason": "content_filter"})
            )]))
        );
    }

    /// An incomplete response that gives no reason is one Meka cannot class, recorded with its
    /// word rather than guessed to be the output cap.
    #[test]
    fn an_incomplete_status_without_a_reason_is_unknown() {
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
                stop: crate::provider::Stop { reason: StopReason::Unknown, word: Some(word), .. },
            } if word == "incomplete")),
            "{events:?}"
        );
    }

    /// An event of the response itself that Meka reads nothing from, as ChatGPT's
    /// `response.metadata` is, is kept as it came in the response's record under its type, and
    /// stays there when the terminal response states the fields again; a `.done` that restates
    /// what its item's done states whole adds nothing.
    #[test]
    fn an_event_meka_reads_nothing_from_is_kept_as_it_came() {
        let mut state = SseState::default();
        let frame = serde_json::json!({"type": "response.metadata", "metadata": {"a": 1}});
        let events = process_event("response.metadata", &frame, &mut state).expect("ok");
        assert!(
            matches!(events.as_slice(), [StreamEvent::ResponseFields(fields)]
                if fields.get("response.metadata") == Some(&frame)),
            "{events:?}"
        );
        assert!(!state.finished);
        let restated = process_event(
            "response.output_text.done",
            &serde_json::json!({"output_index": 0, "content_index": 0, "text": "whole"}),
            &mut state,
        )
        .expect("ok");
        assert!(restated.is_empty(), "{restated:?}");
        let completed = process_event(
            "response.completed",
            &serde_json::json!({"response": {"id": "resp_1", "status": "completed"}}),
            &mut state,
        )
        .expect("ok");
        assert!(
            completed
                .iter()
                .any(|event| matches!(event, StreamEvent::ResponseFields(fields)
                if fields.get("response.metadata") == Some(&frame) && fields.contains_key("id"))),
            "{completed:?}"
        );
    }

    #[test]
    fn process_event_reasoning_done_emits_thinking_complete_with_signature() {
        let mut state = SseState::default();
        state.items.insert(0, OpenItem {
            kind: ItemKind::Reasoning { showed: true },
            ..OpenItem::default()
        });
        let events = process_event(
            "response.output_item.done",
            &serde_json::json!({
                "output_index": 0,
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
                ..
            } => {
                assert_eq!(encrypted_content, "OPAQUE");
                assert_eq!(id.as_deref(), Some("rs_123"));
            }
            other => panic!("expected ThinkingComplete, got {other:?}"),
        }
        assert!(state.items.is_empty());
    }

    /// A list of parts an item states empty, or a value it states where a list of parts goes,
    /// is the item's own and kept as stated, since no block tells it from one never stated.
    #[test]
    fn an_item_s_empty_or_unlisted_parts_are_kept_as_stated() {
        let (events, outcome) = run_events(&[
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": {"type": "reasoning", "id": "rs_1",
                    "summary": [], "content": [], "encrypted_content": "E"}}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 1, "item": {"type": "message", "role": "assistant",
                    "content": null}}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ]);
        outcome.expect("clean stream");
        let message = folded(events);
        let [
            ContentBlock::Thinking {
                provider_fields: Some(reasoning),
                ..
            },
            ContentBlock::Text {
                provider_fields: Some(answer),
                ..
            },
        ] = message.content.as_slice()
        else {
            panic!("{:?}", message.content);
        };
        assert_eq!(reasoning["summary"], serde_json::json!([]));
        assert_eq!(reasoning["content"], serde_json::json!([]));
        assert_eq!(answer["content"], serde_json::Value::Null);
        assert!(answer.contains_key("content"), "{answer:?}");
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
                "output_index": 0,
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
                    }), .. }] if id.as_deref() == Some("rs_silent") && encrypted_content == "OPAQUE"
            ),
            "silent reasoning must still be captured, got {events:?}"
        );
    }

    /// A reasoning item with neither summary nor encrypted content has nothing to render and
    /// nothing to replay, so no block stands for it, and it is kept as it came.
    #[test]
    fn an_empty_reasoning_item_is_kept_as_it_came() {
        let mut state = SseState::default();
        let item = serde_json::json!({ "type": "reasoning", "id": "rs_1", "summary": [] });
        let events = process_event(
            "response.output_item.done",
            &serde_json::json!({ "output_index": 0, "item": item }),
            &mut state,
        )
        .expect("ok");
        assert!(
            matches!(events.as_slice(), [StreamEvent::Unrecognized(kept)] if *kept == item),
            "got {events:?}"
        );
    }

    /// Two message items are two parts, each with the `phase` its item carried, and they go back
    /// as two items with their phases; a refusal part is the model's words, classed a refusal, and
    /// goes back as a refusal part.
    #[test]
    fn message_items_stay_apart_with_their_phase_and_a_refusal_is_its_own() {
        let (events, outcome) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0, "item": {"type": "message"}}),
            ),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 0, "content_index": 0, "delta": "Looking."}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": {
                    "type": "message", "role": "assistant", "phase": "commentary",
                    "content": [{"type": "output_text", "text": "Looking."}],
                }}),
            ),
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 1, "item": {"type": "message"}}),
            ),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 1, "content_index": 0, "delta": "Done."}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 1, "item": {
                    "type": "message", "role": "assistant", "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "Done."}],
                }}),
            ),
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 2, "item": {"type": "message"}}),
            ),
            (
                "response.refusal.delta",
                serde_json::json!({"output_index": 2, "content_index": 0, "delta": "Not that part."}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ]);
        outcome.expect("clean stream");
        let mut accumulator = crate::provider::MessageAccumulator::new();
        for event in events {
            accumulator.push(event);
        }
        let completion = accumulator.finish();
        assert_eq!(completion.stop.reason, StopReason::Refusal);
        let phases: Vec<(&str, Option<&serde_json::Value>)> = completion
            .message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text {
                    text,
                    provider_fields,
                } => Some((
                    text.as_str(),
                    provider_fields
                        .as_ref()
                        .and_then(|fields| fields.get("phase")),
                )),
                _ => None,
            })
            .collect();
        assert_eq!(phases, [
            ("Looking.", Some(&serde_json::json!("commentary"))),
            ("Done.", Some(&serde_json::json!("final_answer"))),
        ]);
        let input = input_of(&[Message::user("hi"), completion.message]);
        let sent: Vec<&serde_json::Value> = input
            .iter()
            .filter(|item| item["role"] == "assistant")
            .collect();
        assert_eq!(sent.len(), 3, "{input:?}");
        assert_eq!(sent[0]["phase"], "commentary");
        assert_eq!(sent[1]["phase"], "final_answer");
        assert_eq!(sent[2]["content"][0]["type"], "refusal");
        assert_eq!(sent[2]["content"][0]["refusal"], "Not that part.");
    }

    /// Every field a Responses reply returns has a home: a message item's own fields and each
    /// content part's beyond its words ride its text, a reasoning item's own ride its thinking,
    /// and the response object's are the record's, without what restates the request, save the
    /// tier actually used.
    #[test]
    fn every_field_a_responses_reply_returns_has_a_home() {
        let mut state = SseState {
            restated: ["model", "instructions", "tools", "service_tier"]
                .map(str::to_string)
                .to_vec(),
            ..SseState::default()
        };
        let mut events = Vec::new();
        for (name, data) in [
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0, "item": {"type": "reasoning", "id": "rs_1"}}),
            ),
            (
                "response.reasoning_summary_part.added",
                serde_json::json!({"output_index": 0, "summary_index": 0}),
            ),
            (
                "response.reasoning_summary_text.delta",
                serde_json::json!({"output_index": 0, "summary_index": 0, "delta": "Plan"}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": {"type": "reasoning", "id": "rs_1", "status": "completed",
                    "summary": [{"type": "summary_text", "text": "Plan"}],
                    "encrypted_content": "E"}}),
            ),
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 1, "item": {"type": "message", "id": "msg_1"}}),
            ),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 1, "content_index": 0, "delta": "Done."}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 1, "item": {"type": "message", "id": "msg_1", "role": "assistant",
                    "status": "completed", "content": [{"type": "output_text", "text": "Done.",
                    "annotations": [{"type": "url_citation", "url": "u"}]}]}}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"id": "resp_1", "object": "response",
                    "status": "completed", "model": "gpt-x", "instructions": "a long prompt",
                    "tools": [{"type": "function", "name": "t"}], "service_tier": "default",
                    "temperature": 1.0, "output": [], "usage": {"input_tokens": 3,
                    "output_tokens": 2, "output_tokens_details": {"reasoning_tokens": 1}}}}),
            ),
        ] {
            events.extend(process_event(name, &data, &mut state).expect("a frame"));
        }
        let mut accumulator = crate::provider::MessageAccumulator::new();
        for event in events {
            accumulator.push(event);
        }
        let completion = accumulator.finish();
        let [
            ContentBlock::Thinking {
                provider_fields: Some(thinking_fields),
                ..
            },
            ContentBlock::Text {
                provider_fields: Some(text_fields),
                ..
            },
        ] = completion.message.content.as_slice()
        else {
            panic!("{:?}", completion.message.content);
        };
        assert_eq!(
            serde_json::Value::Object(thinking_fields.clone()),
            serde_json::json!({"status": "completed"})
        );
        assert_eq!(
            serde_json::Value::Object(text_fields.clone()),
            serde_json::json!({"id": "msg_1", "status": "completed",
                "content": [{"annotations": [{"type": "url_citation", "url": "u"}]}]})
        );
        let record = completion
            .response_fields
            .expect("the response's own fields");
        assert_eq!(record["id"], "resp_1");
        assert_eq!(
            record["service_tier"], "default",
            "the tier used is the response's"
        );
        assert_eq!(
            record["temperature"], 1.0,
            "a default the request did not set is stated"
        );
        assert_eq!(
            record["usage"]["output_tokens_details"]["reasoning_tokens"],
            1
        );
        for held in ["instructions", "tools", "model", "status", "output"] {
            assert!(!record.contains_key(held), "{held}: {record:?}");
        }
    }

    /// A call is the one the item that opened it names, and one it did not name is named by the
    /// item that finishes it; a call neither names is kept as it came and never run. Each way the
    /// reply holds the call once.
    #[test]
    fn a_call_is_named_by_the_item_that_opens_or_finishes_it() {
        let blocks = |added: serde_json::Value, done: serde_json::Value| {
            let (events, outcome) = run_events(&[
                (
                    "response.output_item.added",
                    serde_json::json!({"output_index": 0, "item": added}),
                ),
                (
                    "response.function_call_arguments.delta",
                    serde_json::json!({"output_index": 0, "delta": "{}"}),
                ),
                (
                    "response.output_item.done",
                    serde_json::json!({"output_index": 0, "item": done}),
                ),
            ]);
            if let Err(error) = outcome {
                panic!("{error}");
            }
            let mut accumulator = crate::provider::MessageAccumulator::new();
            for event in events {
                accumulator.push(event);
            }
            accumulator
                .finish()
                .message
                .content
                .iter()
                .map(|block| match block {
                    ContentBlock::ToolUse {
                        id, name, unparsed, ..
                    } => format!("call {id} {name} refused={}", unparsed.is_some()),
                    ContentBlock::Unrecognized { .. } => "unrecognized".to_string(),
                    other => format!("{other:?}"),
                })
                .collect::<Vec<_>>()
        };
        let named = serde_json::json!({"type": "function_call", "call_id": "c1",
            "name": "file_read", "arguments": "{}"});
        let unnamed = serde_json::json!({"type": "function_call", "arguments": "{}"});
        assert_eq!(blocks(named.clone(), unnamed.clone()), [
            "call c1 file_read refused=false"
        ]);
        assert_eq!(blocks(unnamed.clone(), named), [
            "call c1 file_read refused=false"
        ]);
        assert_eq!(blocks(unnamed.clone(), unnamed), ["unrecognized"]);
    }

    /// A finished call whose `arguments` is not the string the spec defines is refused with what
    /// was there, never run with arguments it did not send; one that sends none reads what its
    /// deltas carried.
    #[test]
    fn a_finished_call_s_arguments_are_read_as_the_spec_defines_them() {
        let call = |arguments: Option<serde_json::Value>| {
            let mut item = serde_json::json!({"type": "function_call", "call_id": "c1",
                "name": "file_read"});
            if let Some(arguments) = arguments {
                item["arguments"] = arguments;
            }
            let (events, outcome) = run_events(&[
                (
                    "response.output_item.added",
                    serde_json::json!({"output_index": 0, "item": {"type": "function_call",
                        "call_id": "c1", "name": "file_read"}}),
                ),
                (
                    "response.function_call_arguments.delta",
                    serde_json::json!({"output_index": 0, "delta": "{\"path\":\"a\"}"}),
                ),
                (
                    "response.output_item.done",
                    serde_json::json!({"output_index": 0, "item": item}),
                ),
            ]);
            if let Err(error) = outcome {
                panic!("{error}");
            }
            events
                .into_iter()
                .find_map(|event| match event {
                    StreamEvent::ToolUseEnd { input, .. } => Some(Ok(input)),
                    StreamEvent::ToolCallRefused { unparsed, .. } => Some(Err(unparsed)),
                    _ => None,
                })
                .expect("the call ends")
        };
        assert_eq!(call(None), Ok(serde_json::json!({"path": "a"})));
        assert_eq!(
            call(Some(serde_json::Value::Null)),
            Ok(serde_json::json!({"path": "a"}))
        );
        let refused = call(Some(serde_json::json!({"path": "b"}))).expect_err("refused");
        assert_eq!(refused.text.as_deref(), Some("{\"path\":\"b\"}"));
    }

    /// The fields the spec defines on an `output_text` part, its `annotations` and `logprobs`, go
    /// back on the part as it held them; its other fields do not.
    #[test]
    fn a_text_part_s_annotations_go_back_on_it() {
        let annotations = serde_json::json!([{"type": "url_citation", "url": "u"}]);
        let input = input_of(&[Message::user("hi"), Message {
            role: crate::conversation::Role::Assistant,
            content: vec![ContentBlock::Text {
                text: "Done.".to_string(),
                provider_fields: serde_json::json!({"id": "msg_1",
                        "content": [{"annotations": annotations, "logprobs": [], "other": 1}]})
                .as_object()
                .cloned(),
            }],
            provider_fields: None,
        }]);
        assert_eq!(
            input[1]["content"],
            serde_json::json!([{"type": "output_text", "text": "Done.",
                "annotations": annotations, "logprobs": []}])
        );
    }

    /// A reply recorded with nothing in it puts no item on the wire, which carries no empty output,
    /// whether it holds no part or a `message` item that streamed no words.
    #[test]
    fn an_empty_reply_puts_no_item_on_the_wire() {
        let reply = |content| crate::conversation::Message {
            role: crate::conversation::Role::Assistant,
            content,
            provider_fields: None,
        };
        let input = input_of(&[
            Message::user("hi"),
            reply(Vec::new()),
            Message::user("again"),
            reply(vec![ContentBlock::Text {
                text: String::new(),
                provider_fields: Some(serde_json::Map::from_iter([(
                    "phase".to_owned(),
                    serde_json::json!("final_answer"),
                )])),
            }]),
            Message::user("once more"),
        ]);
        assert_eq!(input.len(), 3, "{input:?}");
        assert!(input.iter().all(|item| item["role"] == "user"), "{input:?}");
    }

    /// An output item of a type Meka reads nothing from is kept as it came, and never sent back.
    #[test]
    fn an_unknown_output_item_is_kept_and_never_sent() {
        let (events, outcome) = run_events(&[
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": {"type": "web_search_call", "id": "ws_1"}}),
            ),
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 1, "item": {"type": "message"}}),
            ),
            (
                "response.output_text.delta",
                serde_json::json!({"output_index": 1, "content_index": 0, "delta": "Found it."}),
            ),
            (
                "response.completed",
                serde_json::json!({"response": {"status": "completed"}}),
            ),
        ]);
        outcome.expect("clean stream");
        let mut accumulator = crate::provider::MessageAccumulator::new();
        for event in events {
            accumulator.push(event);
        }
        let message = accumulator.finish().message;
        assert!(
            matches!(&message.content[0], ContentBlock::Unrecognized { parts }
                if parts[0]["type"] == "web_search_call"),
            "{:?}",
            message.content
        );
        let input = input_of(&[Message::user("hi"), message]);
        assert!(
            !input.iter().any(|item| item["type"] == "web_search_call"),
            "{input:?}"
        );
    }

    /// Each summary part and each part of raw reasoning is recorded as the part it is, with
    /// nothing Meka wrote between them, and goes back in the arrays it came in.
    #[test]
    fn reasoning_parts_are_kept_as_returned_and_go_back_as_returned() {
        let (events, outcome) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({ "output_index": 0, "item": {"type": "reasoning", "id": "rs_1"} }),
            ),
            (
                "response.reasoning_summary_part.added",
                serde_json::json!({ "output_index": 0, "summary_index": 0 }),
            ),
            (
                "response.reasoning_summary_text.delta",
                serde_json::json!({ "output_index": 0, "summary_index": 0, "delta": "**First**\nthinking" }),
            ),
            (
                "response.reasoning_summary_part.done",
                serde_json::json!({ "output_index": 0, "summary_index": 0 }),
            ),
            (
                "response.reasoning_summary_part.added",
                serde_json::json!({ "output_index": 0, "summary_index": 1 }),
            ),
            (
                "response.reasoning_summary_text.delta",
                serde_json::json!({ "output_index": 0, "summary_index": 1, "delta": "**Second**" }),
            ),
            (
                "response.reasoning_summary_part.done",
                serde_json::json!({ "output_index": 0, "summary_index": 1 }),
            ),
            (
                "response.reasoning_text.delta",
                serde_json::json!({ "output_index": 0, "content_index": 0, "delta": "raw" }),
            ),
            (
                "response.output_item.done",
                serde_json::json!({ "output_index": 0, "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "SEALED",
                }}),
            ),
        ]);
        outcome.expect("clean stream");
        let mut accumulator = crate::provider::MessageAccumulator::new();
        for event in events {
            accumulator.push(event);
        }
        let message = accumulator.finish().message;
        let [ContentBlock::Thinking { parts, .. }] = message.content.as_slice() else {
            panic!("{:?}", message.content);
        };
        assert_eq!(parts, &[
            crate::conversation::ReasoningPart::summary("**First**\nthinking"),
            crate::conversation::ReasoningPart::summary("**Second**"),
            crate::conversation::ReasoningPart::reasoning("raw"),
        ]);
        // Through the encoder, as the next request sends it: the block was sealed by no endpoint
        // this request reaches, so its readable parts go back without the seal.
        let mut reply = message.clone();
        reply.content.push(ContentBlock::Text {
            text: "The answer.".to_string(),
            provider_fields: None,
        });
        let body = build_request_body("m", "", &[Message::user("q"), reply], &[], None, None, true);
        let item = body["input"]
            .as_array()
            .and_then(|input| input.iter().find(|item| item["type"] == "reasoning"))
            .unwrap_or_else(|| panic!("{body}"));
        assert!(item.get("encrypted_content").is_none(), "{item}");
        assert!(item.get("id").is_none(), "{item}");
        assert_eq!(
            item["summary"],
            serde_json::json!([
                {"type": "summary_text", "text": "**First**\nthinking"},
                {"type": "summary_text", "text": "**Second**"},
            ])
        );
        assert_eq!(
            item["content"],
            serde_json::json!([{"type": "reasoning_text", "text": "raw"}])
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
                unparsed: None,
                provider_fields: None,
            }],
            provider_fields: None,
        }];
        messages.push(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: vec![image_content("image/png", "QkFTRTY0")],
                is_error: false,
                display: None,
            }],
            provider_fields: None,
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
            parts: (!summary.is_empty())
                .then(|| crate::conversation::ReasoningPart::summary(summary))
                .into_iter()
                .collect(),
            opaque: encrypted.map(|encrypted_content| OpaqueReasoning::Sealed {
                encrypted_content: encrypted_content.to_string(),
                id: id.map(str::to_string),
                origin: Some("https://chatgpt.com".to_string()),
            }),
            provider_fields: None,
        }
    }

    fn tool_call(id: &str) -> ContentBlock {
        ContentBlock::ToolUse {
            id: id.to_string(),
            name: "file_read".to_string(),
            input: serde_json::json!({}),
            unparsed: None,
            provider_fields: None,
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
    /// place of Meka's two discovery tools, and every deferred tool rides with `defer_loading`.
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
                unparsed: None,
                provider_fields: None,
            }]),
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: vec![ToolResultContent::ToolReference {
                        tool_name: "mcp__kv__probe".to_string(),
                    }],
                    is_error: false,
                    display: None,
                }],
                provider_fields: None,
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
                unparsed: None,
                provider_fields: None,
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
                    display: None,
                }],
                provider_fields: None,
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

    /// The endpoint's search item arrives whole on its done frame and dispatches as Meka's search
    /// tool, which answers both a query and a list of names.
    #[test]
    fn a_tool_search_call_item_dispatches_as_the_search_tool() {
        let (events, outcome) = run_events(&[
            (
                "response.output_item.added",
                serde_json::json!({"output_index": 0, "item": {"type": "tool_search_call", "call_id": "call_1", "execution": "client", "arguments": {}, "status": "in_progress"}}),
            ),
            (
                "response.output_item.done",
                serde_json::json!({"output_index": 0, "item": {"type": "tool_search_call", "call_id": "call_1", "execution": "client", "arguments": {"names": ["mcp__kv__probe"]}, "status": "completed"}}),
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
                                StreamEvent::ToolUseEnd { input, .. },
                                StreamEvent::ResponseFields(_),
                                StreamEvent::MessageEnd {
             stop: crate::provider::Stop { reason: StopReason::ToolUse, .. },
            },
                            ] if id == "call_1" && name == SEARCH_TOOL_NAME && input["names"][0] == "mcp__kv__probe"
                        ),
            "{events:?}"
        );
    }

    fn assistant(content: Vec<ContentBlock>) -> Message {
        Message {
            role: Role::Assistant,
            content,
            provider_fields: None,
        }
    }

    fn input_of(messages: &[Message]) -> Vec<serde_json::Value> {
        build_request_body("gpt-5", "", messages, &[], Some("high"), None, true)["input"]
            .as_array()
            .expect("input array")
            .clone()
    }

    /// The whole point of the `include`: reasoning Meka asked the server to encrypt has to go back
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
                    provider_fields: None,
                },
                thinking("first", Some("ONE"), Some("rs_1")),
                tool_call("call_1"),
                thinking("second", Some("TWO"), Some("rs_2")),
                ContentBlock::Text {
                    text: "done".to_string(),
                    provider_fields: None,
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
    /// under anything. Reasoning another wire recorded goes back here as the reasoning it reads as,
    /// and what only its own endpoint can open stays behind: a Claude signature is a MAC from
    /// another cryptosystem, never sent as `encrypted_content`.
    #[test]
    fn another_wires_reasoning_goes_back_as_it_reads_without_its_seal() {
        for opaque in [
            OpaqueReasoning::Signed {
                signature: "CLAUDE_MAC".to_string(),
            },
            OpaqueReasoning::Verbatim,
        ] {
            let foreign_turn = assistant(vec![
                ContentBlock::Thinking {
                    parts: vec![crate::conversation::ReasoningPart::reasoning(
                        "the full reasoning",
                    )],
                    opaque: Some(opaque.clone()),
                    provider_fields: None,
                },
                ContentBlock::Text {
                    text: "answer".to_string(),
                    provider_fields: None,
                },
            ]);
            let serialized = serde_json::to_string(&serde_json::Value::Array(input_of(&[
                Message::user("hi"),
                foreign_turn.clone(),
            ])))
            .expect("serialize");

            assert!(
                !serialized.contains("CLAUDE_MAC"),
                "{opaque:?}: {serialized}"
            );
            assert!(
                !serialized.contains("encrypted_content"),
                "{opaque:?}: {serialized}"
            );
            let input = input_of(&[Message::user("hi"), foreign_turn]);
            let reasoning = input
                .iter()
                .find(|item| item["type"] == "reasoning")
                .unwrap_or_else(|| panic!("{opaque:?}: {serialized}"));
            assert_eq!(
                reasoning["content"],
                serde_json::json!([{"type": "reasoning_text", "text": "the full reasoning"}]),
                "{opaque:?}"
            );
        }
    }

    /// Reasoning nothing sealed goes back as the parts it holds, since the wire has a slot for
    /// them, with no seal and no id.
    #[test]
    fn a_summary_without_encrypted_content_goes_back_as_it_reads() {
        let input = input_of(&[
            Message::user("hi"),
            assistant(vec![thinking("visible only", None, None), tool_call("c1")]),
        ]);

        let reasoning = input
            .iter()
            .find(|item| item["type"] == "reasoning")
            .unwrap_or_else(|| panic!("{input:?}"));
        assert_eq!(
            reasoning["summary"],
            serde_json::json!([{"type": "summary_text", "text": "visible only"}])
        );
        assert!(reasoning.get("encrypted_content").is_none(), "{reasoning}");
        assert!(reasoning.get("id").is_none(), "{reasoning}");
    }

    /// A sealed reasoning item that never carried an id is still replayed to its sealer, as Codex
    /// replays one (`Option<ResponseItemId>`, skipped when absent).
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
    /// Codex asks for `auto`; so does Meka, on the one backend whose endpoint is always OpenAI.
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
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\
             \"item\":{\"type\":\"reasoning\",\"id\":\"rs_live\"}}\n\n",
            "data: {\"type\":\"response.reasoning_summary_part.added\",\"output_index\":0,\
             \"summary_index\":0}\n\n",
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\
             \"summary_index\":0,\"delta\":\"**Plan**\"}\n\n",
            "data: {\"type\":\"response.reasoning_summary_part.done\",\"output_index\":0,\
             \"summary_index\":0}\n\n",
            "data: {\"type\":\"response.reasoning_summary_part.added\",\"output_index\":0,\
             \"summary_index\":1}\n\n",
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\
             \"summary_index\":1,\"delta\":\"**Act**\"}\n\n",
            "data: {\"type\":\"response.reasoning_summary_part.done\",\"output_index\":0,\
             \"summary_index\":1}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\
             \"item\":{\"type\":\"reasoning\",\"id\":\"rs_live\",\"summary\":[],\
             \"encrypted_content\":\"OPAQUE\"}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\
             \"item\":{\"type\":\"message\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\
             \"content_index\":0,\"delta\":\"here you go\"}\n\n",
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
        assert_eq!(
            reasoning["summary"],
            serde_json::json!([
                {"type": "summary_text", "text": "**Plan**"},
                {"type": "summary_text", "text": "**Act**"},
            ])
        );
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
