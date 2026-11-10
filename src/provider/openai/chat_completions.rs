//! `openai-chat-completions`: the Chat Completions API against any endpoint serving it, with an
//! API key.
//!
//! `POST {base_url}/chat/completions`, reaching OpenAI and anything implementing that format
//! (Ollama, vLLM, LM Studio, OpenRouter, Synthetic, LiteLLM, local proxies). This is *not* the
//! legacy `/v1/completions`, a different protocol with no tool calling that several of those same
//! servers also expose and that Meka does not implement.
//!
//! The protocol sibling is [`super::responses`], which takes the same key against OpenAI's newer
//! format; unlike the Responses pair, Chat Completions has one implementation here and shares no
//! wire module.
//!
//! The key comes from the profile's stored credential, never from the environment.

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    conversation::{ContentBlock, Message, OpaqueReasoning, Role},
    error::{MekaError, Result},
    provider::{
        CompletionRequest, Provider, Stop, StopReason, StreamEvent, ToolCallAccumulator,
        ToolDefinition, finalize_tool_call_accumulators,
        sse::{End, Step},
    },
    stats::TokenUsage,
};

/// The `openai-chat-completions` backend: one profile's model, endpoint and API key.
pub(crate) struct OpenAiChatCompletionsProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    /// The settled `reasoning_effort` for the request body, resolved once at construction from the
    /// profile's override. `None` (the unconfigured case) omits the field so the endpoint applies
    /// its own default, which matters most for the local servers this backend also reaches.
    resolved_effort: Option<String>,
    max_output_tokens: Option<u64>,
    /// See [`crate::config::ProfileConfig::max_request_bytes`]; unset means no ceiling here.
    max_request_bytes: Option<usize>,
    /// See `AccountConfig::stream_usage`.
    stream_usage: bool,
    /// See [`crate::config::ProfileConfig::parameters`].
    parameters: serde_json::Map<String, serde_json::Value>,
}

impl OpenAiChatCompletionsProvider {
    /// `api_key` is the credential `settings` carries, already checked to be one by the builder.
    pub(crate) fn from_settings(
        api_key: String,
        settings: crate::provider::ProviderBuilder,
    ) -> Result<Self> {
        let crate::provider::ProviderBuilder {
            model,
            base_url,
            effort: reasoning_effort,
            max_output_tokens,
            max_request_bytes,
            stream_usage,
            parameters,
            ..
        } = settings;
        let resolved_effort = crate::provider::resolve_effort_level(reasoning_effort.as_deref());
        Ok(Self {
            client: crate::provider::build_http_client("openai-chat-completions", |builder| {
                builder
            })?,
            api_key,
            base_url: crate::provider::normalize_base_url(
                base_url
                    .as_deref()
                    .unwrap_or(crate::provider::DEFAULT_OPENAI_BASE_URL),
            ),
            model,
            resolved_effort,
            max_output_tokens,
            max_request_bytes,
            stream_usage,
            parameters,
        })
    }

    /// The settled reasoning-effort to send as `reasoning_effort` (see [`Self::resolved_effort`]).
    fn wire_effort(&self) -> Option<String> {
        self.resolved_effort.clone()
    }

    /// The serialized request, within the profile's ceiling when it states one; see
    /// [`crate::provider::budget`].
    fn request_body_within_budget(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> Result<(String, Option<crate::frontend::Notice>)> {
        match self.max_request_bytes {
            Some(max_request_bytes) => crate::provider::budget::fit_body_to_budget(
                messages,
                max_request_bytes,
                None,
                |messages| {
                    crate::provider::budget::serialize_body(&self.build_request_body(
                        system_prompt,
                        messages,
                        tools,
                        stream,
                    ))
                },
            ),
            None => Ok((
                crate::provider::budget::serialize_body(&self.build_request_body(
                    system_prompt,
                    messages,
                    tools,
                    stream,
                ))?,
                None,
            )),
        }
    }

    /// The request body: the conversation as Chat Completions messages, the tools, and the
    /// profile's effort and output cap when it states them.
    pub(super) fn build_request_body(
        &self,
        system_prompt: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> serde_json::Value {
        let mut openai_messages = Vec::new();

        if !system_prompt.is_empty() {
            openai_messages.push(serde_json::json!({
                "role": "system",
                "content": system_prompt,
            }));
        }

        for message in messages {
            match message.role {
                Role::User => {
                    let has_tool_results = message.carries_tool_results();

                    if has_tool_results {
                        for block in &message.content {
                            if let ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                ..
                            } = block
                            {
                                // Chat Completions restricts the `tool` role's content to text: the
                                // reference defines `ChatCompletionToolMessageParam.content` as
                                // `string | array of ChatCompletionContentPartText` and notes "for
                                // tool messages, only type `text` is supported", with vision on
                                // `user` messages only. So image blocks collapse to the literal
                                // "[Image]" via `tool_result_text_content`. The Responses API takes
                                // `input_image` in `function_call_output.output`, and its encoder
                                // emits them.
                                let text = ContentBlock::tool_result_text_content(content);
                                let tool_message = serde_json::json!({
                                    "role": "tool",
                                    "tool_call_id": tool_use_id,
                                    "content": text,
                                });
                                // No `is_error`: the `tool` message has no such field in this
                                // API, and an endpoint that validates its schema rejects the
                                // whole request for it. The text carries the failure.
                                openai_messages.push(tool_message);
                            }
                        }
                    }
                    // No `match` on `ContentBlock` here means the compiler can't force this
                    // path to handle `Image`; it must be done by hand. When the user message
                    // carries images, Chat Completions wants a `content` array of `text` +
                    // `image_url` parts (vision is user-role only); otherwise a plain string.
                    let has_images = message
                        .content
                        .iter()
                        .any(|block| matches!(block, ContentBlock::Image { .. }));
                    // Text beside tool results is a message that arrived while the tools ran (an
                    // inbox item), and this wire has no place for it inside a `tool` message: it
                    // follows them as a `user` message, which the spec allows after the run of
                    // tool messages a call demands. A message of tool results alone adds nothing.
                    let has_text = !message.wire_text().is_empty();
                    if !has_tool_results || has_text || has_images {
                        if has_images {
                            let mut parts: Vec<serde_json::Value> = Vec::new();
                            // The context block and the words, as one text part.
                            let text = message.wire_text();
                            if !text.is_empty() {
                                parts.push(serde_json::json!({"type": "text", "text": text}));
                            }
                            for block in &message.content {
                                if let ContentBlock::Image { source } = block {
                                    parts.push(match super::data_url(source) {
                                        Some(url) => serde_json::json!({
                                            "type": "image_url",
                                            "image_url": { "url": url },
                                        }),
                                        None => serde_json::json!({
                                            "type": "text",
                                            "text": crate::image::UNRESOLVED_IMAGE_PLACEHOLDER,
                                        }),
                                    });
                                }
                            }
                            openai_messages.push(serde_json::json!({
                                "role": "user",
                                "content": parts,
                            }));
                        } else {
                            openai_messages.push(serde_json::json!({
                                "role": "user",
                                "content": message.wire_text(),
                            }));
                        }
                    }
                }
                Role::Assistant => {
                    let tool_calls: Vec<_> = message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::ToolUse {
                                id,
                                name,
                                input,
                                unparsed,
                                ..
                            } => Some(serde_json::json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": crate::conversation::call_arguments_text(
                                        input,
                                        unparsed.as_ref(),
                                    ),
                                }
                            })),
                            _ => None,
                        })
                        .collect();

                    let mut assistant_message = if tool_calls.is_empty() {
                        serde_json::Map::from_iter([
                            ("role".into(), serde_json::json!("assistant")),
                            ("content".into(), serde_json::json!(message.text_content())),
                        ])
                    } else {
                        let text = message.text_content();
                        let mut with_calls = serde_json::Map::from_iter([
                            ("role".into(), serde_json::json!("assistant")),
                            ("tool_calls".into(), serde_json::json!(tool_calls)),
                        ]);
                        if !text.is_empty() {
                            with_calls.insert("content".into(), serde_json::json!(text));
                        }
                        with_calls
                    };
                    // The words the model declined in, in the field the spec keeps for them.
                    let refusal: String = message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Refusal { text, .. } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect();
                    if !refusal.is_empty() {
                        assistant_message.insert("refusal".into(), serde_json::json!(refusal));
                    }
                    // The readable reasoning goes back in the field this wire keeps for it,
                    // whichever endpoint returned it: DeepSeek's thinking mode wants every
                    // earlier turn's reasoning once tools are sent, tool call or not. A Claude
                    // signature or a sealed Responses blob is nothing this endpoint can read,
                    // and has no place here.
                    let reasoning = message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Thinking { parts, .. } => {
                                Some(crate::conversation::reasoning_as_streamed(parts))
                            }
                            _ => None,
                        })
                        .collect::<String>();
                    if !reasoning.is_empty() {
                        assistant_message
                            .insert("reasoning_content".into(), serde_json::json!(reasoning));
                    }
                    openai_messages.push(serde_json::Value::Object(assistant_message));
                }
            }
        }

        let mut body = serde_json::Map::from_iter([
            ("model".into(), serde_json::json!(self.model)),
            ("messages".into(), serde_json::json!(openai_messages)),
            ("stream".into(), serde_json::json!(stream)),
        ]);

        // OpenAI omits `usage` from streamed responses unless explicitly asked; without this the
        // final usage-only chunk never arrives and token accounting (the `/status` context gauge,
        // auto-compact) silently reads zero for streaming turns. An endpoint that refuses the field
        // is the account's `stream_usage` to turn off.
        if stream && self.stream_usage {
            body.insert(
                "stream_options".into(),
                serde_json::json!({ "include_usage": true }),
            );
        }

        let reasoning_effort = self.wire_effort();
        if let Some(effort) = &reasoning_effort {
            body.insert("reasoning_effort".into(), serde_json::json!(effort));
        }
        // Only the profile's own cap: the endpoint's default is the endpoint's fact, and this
        // backend reaches whatever `base_url` names. A floor sent whenever `effort` is set would be
        // rejected by every model with a smaller output cap, and each rejection would cost a
        // degraded retry that fails the same way.
        if let Some(max_output) = self.max_output_tokens {
            body.insert(
                "max_completion_tokens".into(),
                serde_json::json!(max_output),
            );
        }

        if !tools.is_empty() {
            let openai_tools: Vec<_> = tools
                .iter()
                .map(|tool| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                    })
                })
                .collect();
            body.insert("tools".into(), serde_json::json!(openai_tools));
        }
        crate::provider::add_parameters(&mut body, &self.parameters);

        serde_json::Value::Object(body)
    }

    /// The assistant message, stop reason and usage out of a non-streaming response.
    pub(super) fn parse_non_streaming_response(
        &self,
        response: &serde_json::Value,
    ) -> Result<(Message, Stop, TokenUsage)> {
        let choice = response
            .get("choices")
            .and_then(|choices| choices.get(0))
            .ok_or_else(|| MekaError::Provider("no choices in response".to_string()))?;

        let stated_stop = choice
            .get("finish_reason")
            .and_then(|reason| reason.as_str())
            .map(|finish_reason| chat_stop(choice, finish_reason));

        let assistant_message = choice
            .get("message")
            .ok_or_else(|| MekaError::Provider("no 'message' in choice".to_string()))?;
        let mut content_blocks = Vec::new();

        let reasoning = reasoning_of(assistant_message);
        if let Some((_, reasoning)) = reasoning {
            content_blocks.push(ContentBlock::reasoning(
                reasoning,
                Some(OpaqueReasoning::Verbatim),
            ));
        }

        match assistant_message.get("content") {
            Some(serde_json::Value::String(text)) if !text.is_empty() => {
                content_blocks.push(ContentBlock::Text {
                    text: text.clone(),
                    provider_fields: None,
                });
            }
            Some(serde_json::Value::String(_) | serde_json::Value::Null) | None => {}
            // Content in a shape the spec does not define, kept as it came.
            Some(other) => content_blocks.push(ContentBlock::Unrecognized {
                parts: vec![other.clone()],
            }),
        }
        if let Some(refusal) = assistant_message
            .get("refusal")
            .and_then(|refusal| refusal.as_str())
            .filter(|refusal| !refusal.is_empty())
        {
            content_blocks.push(ContentBlock::Refusal {
                text: refusal.to_string(),
                provider_fields: None,
            });
        }

        if let Some(tool_calls) = assistant_message
            .get("tool_calls")
            .and_then(|tool_calls| tool_calls.as_array())
        {
            for tool_call in tool_calls {
                // A call in a shape the spec does not define, without an id or a name, is kept as
                // it came and never run; an empty string names nothing.
                let (Some(id), Some(name)) = (
                    tool_call
                        .get("id")
                        .and_then(|id| id.as_str())
                        .filter(|id| !id.is_empty()),
                    tool_call
                        .get("function")
                        .and_then(|function| function.get("name"))
                        .and_then(|name| name.as_str())
                        .filter(|name| !name.is_empty()),
                ) else {
                    content_blocks.push(ContentBlock::Unrecognized {
                        parts: vec![tool_call.clone()],
                    });
                    continue;
                };
                let (id, name) = (id.to_string(), name.to_string());
                let arguments = tool_call
                    .get("function")
                    .and_then(|function| function.get("arguments"));
                let provider_fields = tool_call
                    .as_object()
                    .and_then(|call| crate::provider::fields_beyond(call, CALL_FIELDS_READ));
                content_blocks.push(
                    match crate::provider::arguments_from_reply(&name, arguments) {
                        Ok(input) => ContentBlock::ToolUse {
                            id,
                            name,
                            input,
                            unparsed: None,
                            provider_fields,
                        },
                        Err(unparsed) => {
                            crate::provider::refused_tool_use(id, name, unparsed, provider_fields)
                        }
                    },
                );
            }
        }

        let token_usage = response
            .get("usage")
            .map(|usage| {
                super::parse_usage(
                    usage,
                    "prompt_tokens",
                    "prompt_tokens_details",
                    "completion_tokens",
                )
            })
            .unwrap_or_default();

        let has_calls = content_blocks
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolUse { .. }));
        let stop = stated_stop
            .unwrap_or_else(|| unstated_stop(has_calls))
            .with_refusal_of(&content_blocks);
        Ok((
            Message {
                role: Role::Assistant,
                content: content_blocks,
                provider_fields: message_remainder(
                    assistant_message,
                    reasoning.map(|(field, _)| field),
                ),
            },
            stop,
            token_usage,
        ))
    }
}

// An API key has nothing to refresh; the impl exists so every send goes through the one site that
// races a stop, `crate::oauth::send_with_one_refresh`.
impl crate::oauth::RefreshesCredential for OpenAiChatCompletionsProvider {}

#[async_trait]
impl Provider for OpenAiChatCompletionsProvider {
    async fn complete(
        &self,
        request: CompletionRequest<'_>,
        cancellation: CancellationToken,
    ) -> Result<crate::provider::Completion> {
        let CompletionRequest {
            system_prompt,
            messages,
            tools,
            ..
        } = request;
        let (body_json, redaction_notice) =
            self.request_body_within_budget(system_prompt, messages, tools, false)?;

        let response = crate::oauth::send_with_one_refresh(
            self,
            crate::error::ProviderRequest::Completion,
            |error| crate::error::provider_transport_error("HTTP request failed", error, None),
            || async {
                Ok(self
                    .client
                    .post(format!("{}/chat/completions", self.base_url))
                    .header("Authorization", crate::text::bearer(&self.api_key))
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .body(body_json.clone()))
            },
            &cancellation,
        )
        .await?;

        let retry_after = crate::error::parse_retry_after(response.headers());
        let response = crate::provider::succeeded(
            response,
            "completion",
            crate::error::ProviderRequest::Completion,
            &cancellation,
        )
        .await?;
        let response_text =
            crate::error::read_whole_reply(response, retry_after, &cancellation).await?;

        let response_json: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|error| MekaError::Provider(format!("invalid JSON response: {error}")))?;

        let (message, stop, usage) = self.parse_non_streaming_response(&response_json)?;
        Ok(crate::provider::Completion {
            message,
            stop,
            model: response_model(&response_json),
            usage,
            response_fields: response_json
                .as_object()
                .and_then(|response| response_remainder(response, Some(0))),
            notices: redaction_notice.into_iter().collect(),
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest<'_>,
        event_sender: mpsc::Sender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        let CompletionRequest {
            system_prompt,
            messages,
            tools,
            ..
        } = request;
        let (body_json, redaction_notice) =
            self.request_body_within_budget(system_prompt, messages, tools, true)?;
        // A send error here means the consumer hung up already, which the SSE driver reports
        // itself.
        if let Some(notice) = redaction_notice
            && event_sender
                .send(StreamEvent::Notice(notice))
                .await
                .is_err()
        {
            tracing::trace!("stream event receiver dropped");
        }

        let response = crate::oauth::send_with_one_refresh(
            self,
            crate::error::ProviderRequest::Completion,
            |error| crate::error::provider_transport_error("HTTP request failed", error, None),
            || async {
                Ok(self
                    .client
                    .post(format!("{}/chat/completions", self.base_url))
                    .header("Authorization", crate::text::bearer(&self.api_key))
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .body(body_json.clone()))
            },
            &cancellation,
        )
        .await?;

        let mut protocol = ChatCompletionsStream::default();
        let ended = crate::provider::sse::drive(
            response,
            "Chat Completions",
            &event_sender,
            &cancellation,
            &mut protocol,
        )
        .await;
        match ended {
            Ok(end) => conclude_stream(end, protocol, &event_sender).await,
            // A stream cut short still delivered what arrived: the calls and the fields it held.
            Err(error) => {
                settle_stream(&mut protocol, &event_sender).await;
                Err(error)
            }
        }
    }

    fn resolved_effort(&self) -> Option<String> {
        self.wire_effort()
    }
}

/// End the stream: close reasoning nothing followed, finalize pending tool calls and emit the
/// single `MessageEnd`, preferring the recorded `finish_reason` and falling back to tool presence
/// when none arrived.
///
/// A stream that stopped without saying so is a failure, not a short turn. The read ending before
/// `[DONE]` (a proxy closing the chunked response, a dropped connection) is indistinguishable
/// from the terminal frame unless it is tracked, and treating the two alike commits a truncated
/// message as a finished one: `final_stop` is `None`, so the fallback stamps `EndTurn` and the
/// partial answer is persisted as complete with no retry. `StreamError` is retryable, so the
/// existing retry path applies. A close *after* the `finish_reason` chunk is the other case: the
/// stream has said everything it had to, and a gateway that omits the `[DONE]` sentinel is read
/// the way the Claude driver reads a close after `message_delta`. Failing it would drop every
/// tool call of the turn.
async fn conclude_stream(
    end: End,
    mut protocol: ChatCompletionsStream,
    event_sender: &mpsc::Sender<StreamEvent>,
) -> Result<()> {
    if matches!(end, End::Ended) && protocol.final_stop.is_none() {
        let message = "OpenAI stream ended before a terminal event".to_string();
        tracing::warn!("{message}");
        let error = crate::provider::sse::stream_error(event_sender, message).await;
        settle_stream(&mut protocol, event_sender).await;
        return Err(error);
    }
    let Some(has_tools) = settle_stream(&mut protocol, event_sender).await else {
        return Ok(());
    };
    let stop = protocol
        .final_stop
        .take()
        .unwrap_or_else(|| unstated_stop(has_tools));
    if event_sender
        .send(StreamEvent::MessageEnd { stop })
        .await
        .is_err()
    {
        tracing::trace!("stream event receiver dropped");
    }
    Ok(())
}

/// Hand over what the stream holds, however it ended: the reasoning closed, each call it
/// assembled, and the message's and the response's own fields as folded. Whether it made calls;
/// `None` once the receiver is gone.
async fn settle_stream(
    protocol: &mut ChatCompletionsStream,
    event_sender: &mpsc::Sender<StreamEvent>,
) -> Option<bool> {
    if !protocol.close_reasoning(event_sender).await {
        tracing::trace!("stream event receiver dropped");
        return None;
    }
    let has_tools =
        finalize_tool_call_accumulators(&mut protocol.tool_call_accumulators, event_sender).await;
    let followed = place_of(&protocol.completion, protocol.choice_index);
    let message_fields = followed
        .and_then(|followed| {
            protocol
                .completion
                .get("choices")
                .and_then(|choices| choices.get(followed))
                .and_then(|choice| choice.get("message"))
        })
        .and_then(|message| message_remainder(message, protocol.reasoning_field));
    let response_fields = response_remainder(&protocol.completion, followed);
    for event in message_fields
        .map(StreamEvent::MessageFields)
        .into_iter()
        .chain(response_fields.map(StreamEvent::ResponseFields))
    {
        if event_sender.send(event).await.is_err() {
            tracing::trace!("stream event receiver dropped");
            return None;
        }
    }
    Some(has_tools)
}

/// The stop of a reply that named no `finish_reason`, which the spec requires: a round when it
/// made calls, else an end, with no word of the provider's recorded for it. One rule for a whole
/// reply and a streamed one.
fn unstated_stop(has_calls: bool) -> Stop {
    Stop::of(if has_calls {
        StopReason::ToolUse
    } else {
        StopReason::EndTurn
    })
}

/// The Chat Completions driver's state between frames.
#[derive(Default)]
struct ChatCompletionsStream {
    tool_call_accumulators: std::collections::HashMap<i64, ToolCallAccumulator>,
    /// Set when a `finish_reason` chunk arrives. The read keeps going afterward so the trailing
    /// usage chunk (emitted by `stream_options.include_usage`) is captured; finalization and the
    /// single MessageEnd run once the stream ends.
    final_stop: Option<Stop>,
    /// A reasoning block is open: its deltas have streamed and nothing has followed them yet. The
    /// wire marks no end of reasoning, so the first content or tool-call delta closes it, and the
    /// stream's end closes one that nothing followed.
    reasoning_open: bool,
    /// The `index` of the choice this stream follows: the first one it names. A profile's
    /// `parameters` may ask for `n` answers, which stream interleaved, one choice per chunk.
    choice_index: Option<u64>,
    /// Whether the model the chunks name has been passed on: every chunk names it.
    model_named: bool,
    /// The completion as its chunks state it, folded as the reference client folds them.
    completion: serde_json::Map<String, serde_json::Value>,
    /// The field the followed choice's reasoning came under, which its blocks hold.
    reasoning_field: Option<&'static str>,
}

impl ChatCompletionsStream {
    /// Close the open reasoning block, if any, so its text becomes a thinking block ahead of
    /// whatever comes next. `false` once the receiver is gone.
    async fn close_reasoning(&mut self, event_sender: &mpsc::Sender<StreamEvent>) -> bool {
        if !self.reasoning_open {
            return true;
        }
        self.reasoning_open = false;
        event_sender
            .send(StreamEvent::ThinkingComplete {
                opaque: Some(OpaqueReasoning::Verbatim),
                provider_fields: None,
            })
            .await
            .is_ok()
    }
}

/// How a choice ended: its `finish_reason` classed, the word itself, and OpenRouter's
/// `native_finish_reason`, the upstream provider's own word, beside it when the choice carries one.
fn chat_stop(choice: &serde_json::Value, finish_reason: &str) -> Stop {
    let details = choice
        .get("native_finish_reason")
        .filter(|native| !native.is_null())
        .map(|native| {
            serde_json::Map::from_iter([("native_finish_reason".to_string(), native.clone())])
        });
    Stop {
        reason: parse_openai_stop_reason(finish_reason),
        word: Some(finish_reason.to_string()),
        details,
    }
}

/// The fields of a tool call Meka reads; every other is the provider's, recorded with the call.
const CALL_FIELDS_READ: &[&str] = &["index", "id", "type", "function"];

/// The reasoning an endpoint attached to a message or delta, under either spelling of the field,
/// with the field it came under: `reasoning_content` (DeepSeek, Kimi, Qwen, GLM, xAI) or
/// `reasoning` (vLLM, OpenRouter). An empty string is what a content chunk carries for it, and is
/// nothing.
fn reasoning_of(carrier: &serde_json::Value) -> Option<(&'static str, &str)> {
    ["reasoning_content", "reasoning"]
        .into_iter()
        .find_map(|field| {
            carrier
                .get(field)
                .and_then(|reasoning| reasoning.as_str())
                .filter(|reasoning| !reasoning.is_empty())
                .map(|reasoning| (field, reasoning))
        })
}

/// A message's fields nothing else holds: every one but its role and what its blocks hold, the
/// content, the refusal, the calls and the reasoning field Meka read.
fn message_remainder(
    message: &serde_json::Value,
    reasoning_field: Option<&str>,
) -> Option<crate::conversation::ProviderFields> {
    let mut held = vec!["role", "content", "refusal", "tool_calls"];
    held.extend(reasoning_field);
    message
        .as_object()
        .and_then(|message| crate::provider::fields_beyond(message, &held))
}

/// A completion's fields nothing else holds: all of it but the model, and in the choice at
/// `followed` its message, which the reply holds, and how it finished, which its stop holds. The
/// raw `usage` stays, since Meka holds only its own count of it.
fn response_remainder(
    completion: &serde_json::Map<String, serde_json::Value>,
    followed: Option<usize>,
) -> Option<crate::conversation::ProviderFields> {
    let mut remainder = completion.clone();
    remainder.remove("model");
    if let Some(choice) = followed.and_then(|followed| {
        remainder
            .get_mut("choices")
            .and_then(serde_json::Value::as_array_mut)
            .and_then(|choices| choices.get_mut(followed))
            .and_then(serde_json::Value::as_object_mut)
    }) {
        for held in ["message", "delta", "finish_reason", "native_finish_reason"] {
            choice.remove(held);
        }
    }
    (!remainder.is_empty()).then_some(remainder)
}

/// Fold `chunk` into `completion` as the reference client folds a streamed completion
/// (openai-python `lib/streaming/chat/_completions.py`): a choice's `delta` folds into its
/// `message` and its `logprobs` into its own, by [`crate::provider::accumulate_delta`]. The
/// reference takes a choice's `finish_reason` from the latest chunk that states one, and keeps any
/// other field from the chunk that first stated it, which drops what a later chunk adds
/// (`service_tier`, OpenRouter's `native_finish_reason`); every field here takes its latest
/// non-null statement, as `finish_reason` does there.
fn fold_chunk(
    completion: &mut serde_json::Map<String, serde_json::Value>,
    chunk: &serde_json::Value,
) {
    use serde_json::Value;
    let Some(chunk) = chunk.as_object() else {
        return;
    };
    for (key, value) in chunk {
        if key != "choices" {
            crate::provider::state_latest(completion, key, value);
            continue;
        }
        let Some(choices) = value.as_array() else {
            continue;
        };
        let folded = completion
            .entry("choices".to_string())
            .or_insert_with(|| Value::Array(Vec::new()));
        let Some(folded) = folded.as_array_mut() else {
            continue;
        };
        for choice in choices.iter().filter_map(Value::as_object) {
            let index = choice.get("index").cloned().unwrap_or(Value::from(0));
            let place = match folded
                .iter()
                .position(|held| held.get("index") == Some(&index))
            {
                Some(place) => place,
                None => {
                    folded.push(serde_json::json!({ "index": index }));
                    folded.len() - 1
                }
            };
            let Some(entry) = folded.get_mut(place).and_then(Value::as_object_mut) else {
                continue;
            };
            for (field, value) in choice {
                let target = match field.as_str() {
                    "delta" => "message",
                    "logprobs" => "logprobs",
                    _ => {
                        crate::provider::state_latest(entry, field, value);
                        continue;
                    }
                };
                let Some(value) = value.as_object() else {
                    continue;
                };
                let slot = entry.entry(target.to_string()).or_insert(Value::Null);
                if !slot.is_object() {
                    *slot = Value::Object(serde_json::Map::new());
                }
                if let Some(slot) = slot.as_object_mut() {
                    crate::provider::accumulate_delta(slot, value);
                }
            }
        }
    }
}

/// Where in `completion`'s folded choices the choice under `index` sits.
fn place_of(
    completion: &serde_json::Map<String, serde_json::Value>,
    index: Option<u64>,
) -> Option<usize> {
    let index = index?;
    completion
        .get("choices")
        .and_then(serde_json::Value::as_array)?
        .iter()
        .position(|choice| choice.get("index").and_then(serde_json::Value::as_u64) == Some(index))
}

#[async_trait]
impl crate::provider::sse::Protocol for ChatCompletionsStream {
    async fn frame(
        &mut self,
        event: eventsource_stream::Event,
        event_sender: &mpsc::Sender<StreamEvent>,
    ) -> Result<Step> {
        if event.data == "[DONE]" {
            return Ok(Step::Finished);
        }
        let data = crate::provider::sse::frame_json("Chat Completions", &event.data)?;
        match handle_stream_chunk(&data, self, event_sender).await {
            ChunkOutcome::Continue => Ok(Step::Continue),
            ChunkOutcome::Stop => Ok(Step::ReceiverGone),
            ChunkOutcome::Fail(error) => Err(error),
        }
    }
}

/// Whether the caller should keep reading the stream after a chunk. `Stop` means the event receiver
/// has been dropped, so there is nobody left to stream to; `Fail` means the endpoint reported an
/// error inside the stream, which ends the turn the way a failed request would; the read loop
/// announces it on the channel.
#[derive(Debug)]
enum ChunkOutcome {
    Continue,
    Stop,
    Fail(MekaError),
}

/// Folds one Chat Completions streaming chunk into the in-progress response: forwards usage,
/// reasoning and text, accumulates tool-call fragments by index, and records the stop reason.
///
/// Extracted from the read loop so the chunk shapes real endpoints emit can be tested without an
/// HTTP server.
async fn handle_stream_chunk(
    data: &serde_json::Value,
    stream: &mut ChatCompletionsStream,
    event_sender: &mpsc::Sender<StreamEvent>,
) -> ChunkOutcome {
    // An error object in a 200 stream is how OpenRouter, vLLM and OpenAI itself report a failure
    // that began after the headers went out. It has no `choices`, so reading past it would commit
    // whatever had streamed as a finished turn with no error and no retry.
    //
    // `"error": null` is a field some servers emit on every chunk, and is no error.
    if let Some(error) = data.get("error").filter(|error| !error.is_null()) {
        return ChunkOutcome::Fail(crate::error::provider_stream_error_object(
            error,
            "the endpoint reported an error mid-stream",
        ));
    }

    // The response as the chunks state it: its envelope told as soon as it is known, so a reply
    // cut short keeps it, and the whole of it at the stream's end.
    let first = stream.completion.is_empty();
    fold_chunk(&mut stream.completion, data);
    if first
        && let Some(fields) = response_remainder(
            &stream
                .completion
                .iter()
                .filter(|(key, _)| *key != "choices")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
            None,
        )
        && event_sender
            .send(StreamEvent::ResponseFields(fields))
            .await
            .is_err()
    {
        tracing::trace!("stream event receiver dropped");
        return ChunkOutcome::Stop;
    }

    if !stream.model_named
        && let Some(model) = response_model(data)
    {
        stream.model_named = true;
        if event_sender.send(StreamEvent::Model(model)).await.is_err() {
            tracing::trace!("stream event receiver dropped");
            return ChunkOutcome::Stop;
        }
    }

    // A chunk states the usage whole or not at all. `"usage": null` (OpenAI puts it on every chunk
    // ahead of the usage chunk) states nothing, as the response's record reads a null, so a
    // gateway that repeats it after the usage chunk leaves the reply's counts and its record
    // agreeing.
    if let Some(usage) = data.get("usage").filter(|usage| usage.is_object()) {
        tracing::debug!("chat completions usage: {usage}");
        let token_usage = super::parse_usage(
            usage,
            "prompt_tokens",
            "prompt_tokens_details",
            "completion_tokens",
        );
        if event_sender
            .send(StreamEvent::Usage(token_usage))
            .await
            .is_err()
        {
            tracing::trace!("stream event receiver dropped");
            return ChunkOutcome::Stop;
        }
    }

    // The first choice the stream names is the answer, whatever number a server gives it; a choice
    // under another `index` is one more of `n` answers, left out rather than mixed in.
    let index_of = |choice: &serde_json::Value| {
        choice
            .get("index")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
    };
    let Some(choice) = data
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| {
            let followed = *stream
                .choice_index
                .get_or_insert(index_of(choices.first()?));
            choices.iter().find(|choice| index_of(choice) == followed)
        })
    else {
        return ChunkOutcome::Continue;
    };

    if let Some(finish_reason) = choice
        .get("finish_reason")
        .and_then(|reason| reason.as_str())
    {
        // Record the stop reason but keep reading: with `stream_options.include_usage` the usage
        // arrives in a trailing chunk AFTER this one (and before `[DONE]`). Finalization and the
        // single MessageEnd happen once the stream ends, back in the caller.
        //
        // Fall through to the delta below rather than returning here: OpenAI itself sends
        // `finish_reason` alone with an empty delta, but vLLM-backed endpoints coalesce the final
        // content or tool_calls delta into this same chunk whenever generation ends before the
        // stream flushes it separately. Skipping the delta would drop it, which for a tool call
        // leaves `finish_reason: "tool_calls"` with no tool-use block at all: a silently empty
        // turn.
        stream.final_stop = Some(chat_stop(choice, finish_reason));
    }

    let Some(delta) = choice.get("delta") else {
        return ChunkOutcome::Continue;
    };

    if let Some((field, reasoning)) = reasoning_of(delta) {
        stream.reasoning_field.get_or_insert(field);
        stream.reasoning_open = true;
        if event_sender
            .send(StreamEvent::ThinkingDelta(reasoning.to_string()))
            .await
            .is_err()
        {
            tracing::trace!("stream event receiver dropped");
            return ChunkOutcome::Stop;
        }
    }

    // The answer streams as a string, which the spec defines; anything else is a shape Meka
    // reads nothing from, kept as it came.
    let content = match delta.get("content") {
        Some(serde_json::Value::String(text)) if !text.is_empty() => {
            Some(StreamEvent::TextDelta(text.clone()))
        }
        Some(serde_json::Value::String(_) | serde_json::Value::Null) | None => None,
        Some(other) => Some(StreamEvent::Unrecognized(other.clone())),
    };
    let refusal = delta
        .get("refusal")
        .and_then(|refusal| refusal.as_str())
        .filter(|refusal| !refusal.is_empty())
        .map(|refusal| StreamEvent::RefusalDelta(refusal.to_string()));
    for event in content.into_iter().chain(refusal) {
        if !stream.close_reasoning(event_sender).await || event_sender.send(event).await.is_err() {
            tracing::trace!("stream event receiver dropped");
            return ChunkOutcome::Stop;
        }
    }

    let Some(tool_calls) = delta
        .get("tool_calls")
        .and_then(|tool_calls| tool_calls.as_array())
    else {
        return ChunkOutcome::Continue;
    };
    if !tool_calls.is_empty() && !stream.close_reasoning(event_sender).await {
        tracing::trace!("stream event receiver dropped");
        return ChunkOutcome::Stop;
    }
    let accumulators = &mut stream.tool_call_accumulators;

    for tool_call in tool_calls {
        // A fragment the spec places by its `index`; one without is a shape it does not define,
        // kept as it came.
        let Some(index) = tool_call.get("index").and_then(|index| index.as_i64()) else {
            if event_sender
                .send(StreamEvent::Unrecognized(tool_call.clone()))
                .await
                .is_err()
            {
                tracing::trace!("stream event receiver dropped");
                return ChunkOutcome::Stop;
            }
            continue;
        };

        // An empty string names nothing, so the fragment that names the call fills it in.
        let name = tool_call
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(|name| name.as_str())
            .filter(|name| !name.is_empty());

        // Keyed on the index alone. The id is filled in by whichever fragment carries it, so an
        // endpoint that sends the name and the first arguments before the id still assembles one
        // call; a call no fragment names is kept as its fragments came.
        let accumulator = accumulators.entry(index).or_default();
        accumulator.fragments.push(tool_call.clone());
        // Whatever else a fragment carries is the provider's, kept as it came (Gemini's
        // `extra_content`).
        if let Some(fields) = tool_call
            .as_object()
            .and_then(|call| crate::provider::fields_beyond(call, CALL_FIELDS_READ))
        {
            crate::provider::accumulate_delta(
                accumulator
                    .provider_fields
                    .get_or_insert_with(Default::default),
                &fields,
            );
        }
        if let Some(id) = tool_call
            .get("id")
            .and_then(|id| id.as_str())
            .filter(|id| !id.is_empty())
            && accumulator.id.is_none()
        {
            accumulator.id = Some(id.to_string());
        }
        if let Some(name) = name
            && accumulator.name.is_none()
        {
            accumulator.name = Some(name.to_string());
        }

        match tool_call
            .get("function")
            .and_then(|function| function.get("arguments"))
        {
            Some(serde_json::Value::String(args)) => {
                accumulator.carried_arguments = true;
                accumulator.arguments.push_str(args);
            }
            Some(serde_json::Value::Null) | None => {}
            // Arguments the spec defines as a string, sent as something else: the call is
            // refused with what was sent, never run on the arguments it did not give.
            Some(other) => accumulator.unread_arguments = Some(other.clone()),
        }
    }

    ChunkOutcome::Continue
}

/// The model a reply or a chunk names, when it names one.
fn response_model(data: &serde_json::Value) -> Option<String> {
    data.get("model")
        .and_then(|model| model.as_str())
        .filter(|model| !model.is_empty())
        .map(str::to_string)
}

fn parse_openai_stop_reason(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::EndTurn,
        "tool_calls" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        "content_filter" => StopReason::ContentFilter,
        other => {
            tracing::warn!(
                "OpenAI returned unrecognized finish_reason {other:?}; mapping to Unknown"
            );
            StopReason::Unknown
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::ToolResultContent;

    /// A reply is recorded as the wire returned it: a refusal is the model's words in a block of
    /// its own that classes the stop, a call's and a message's fields beyond those Meka reads are
    /// kept with them and not sent back, a `content` the spec does not define is kept verbatim and
    /// never sent, and a null `reasoning_content` does not hide the `reasoning` beside it.
    #[test]
    fn a_whole_reply_is_recorded_as_returned_and_sends_back_only_what_its_wire_defines() {
        let provider = provider_for_test();
        let response = serde_json::json!({
            "model": "gpt-test",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Some words.",
                    "refusal": "I cannot help with that.",
                    "reasoning_content": null,
                    "reasoning": "thought it over",
                    "reasoning_details": [{"type": "reasoning.text", "text": "thought it over"}],
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "file_read", "arguments": "{}"},
                        "extra_content": {"google": {"thought_signature": "SIG"}},
                    }],
                },
            }],
        });
        let (message, stop, _) = provider
            .parse_non_streaming_response(&response)
            .expect("parses");
        assert_eq!(stop.reason, StopReason::Refusal, "{stop:?}");
        assert_eq!(stop.word.as_deref(), Some("stop"));
        assert!(
            matches!(&message.content[0], ContentBlock::Thinking { parts, .. }
                if crate::conversation::reasoning_text(parts) == "thought it over"),
            "{:?}",
            message.content
        );
        let fields = message
            .provider_fields
            .as_ref()
            .expect("the message's own fields");
        assert!(fields.contains_key("reasoning_details"), "{fields:?}");
        assert!(
            !fields.contains_key("reasoning"),
            "the reasoning Meka read is the block's: {fields:?}"
        );
        assert!(matches!(&message.content[1], ContentBlock::Text {
            provider_fields: None,
            ..
        }));
        assert!(
            matches!(&message.content[2], ContentBlock::Refusal { text, .. } if text == "I cannot help with that.")
        );
        let ContentBlock::ToolUse {
            provider_fields: Some(call_fields),
            ..
        } = &message.content[3]
        else {
            panic!("{:?}", message.content);
        };
        assert!(call_fields.contains_key("extra_content"), "{call_fields:?}");

        let body = provider.build_request_body("", &[Message::user("hi"), message], &[], false);
        let sent = &body["messages"][1];
        assert_eq!(sent["refusal"], "I cannot help with that.", "{sent}");
        assert!(sent.get("reasoning_details").is_none(), "{sent}");
        assert!(
            sent["tool_calls"][0].get("extra_content").is_none(),
            "{sent}"
        );

        let unread = serde_json::json!({
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "parts"}]},
            }],
        });
        let (message, ..) = provider
            .parse_non_streaming_response(&unread)
            .expect("parses");
        assert!(
            matches!(&message.content[..], [ContentBlock::Unrecognized { parts }]
                if parts[0][0]["text"] == "parts"),
            "{:?}",
            message.content
        );
        let body = provider.build_request_body("", &[Message::user("hi"), message], &[], false);
        assert_eq!(body["messages"][1]["content"], "", "{body}");
    }

    /// A frame that is not JSON fails the attempt as a stream error, which is retried like any
    /// other, rather than being skipped while the reply commits with a hole in it.
    #[tokio::test]
    async fn a_frame_that_does_not_parse_fails_the_attempt() {
        let (sender, _receiver) = mpsc::channel::<StreamEvent>(8);
        let mut protocol = ChatCompletionsStream::default();
        let frame = eventsource_stream::Event {
            event: "message".to_string(),
            data: "{\"choices\": [".to_string(),
            id: String::new(),
            retry: None,
        };
        let outcome = crate::provider::sse::Protocol::frame(&mut protocol, frame, &sender).await;
        assert!(matches!(outcome, Err(MekaError::StreamError(_))));
    }

    /// Every stop word is classed, and the word is kept beside the class: `content_filter` is a
    /// stop of its own, not a failure, and a word Meka has no class for is `unknown`.
    #[test]
    fn each_finish_reason_is_classed_and_its_word_kept() {
        for (word, class) in [
            ("stop", StopReason::EndTurn),
            ("tool_calls", StopReason::ToolUse),
            ("length", StopReason::MaxTokens),
            ("content_filter", StopReason::ContentFilter),
            ("something_new", StopReason::Unknown),
        ] {
            let stop = chat_stop(&serde_json::json!({"native_finish_reason": "SAFETY"}), word);
            assert_eq!(stop.reason, class, "{word}");
            assert_eq!(stop.word.as_deref(), Some(word));
            assert_eq!(
                stop.details
                    .as_ref()
                    .and_then(|details| details.get("native_finish_reason")),
                Some(&serde_json::json!("SAFETY"))
            );
        }
    }

    /// Reasoning returned in the clear goes back in `reasoning_content` as it streamed, with
    /// nothing written between two blocks of it.
    #[test]
    fn reasoning_in_the_clear_goes_back_as_it_streamed() {
        let provider = provider_for_test();
        let reply = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::reasoning("first", Some(OpaqueReasoning::Verbatim)),
                ContentBlock::Text {
                    text: "answer".to_string(),
                    provider_fields: None,
                },
                ContentBlock::reasoning(" then more", Some(OpaqueReasoning::Verbatim)),
            ],
            provider_fields: None,
        };
        let body = provider.build_request_body("", &[Message::user("hi"), reply], &[], false);
        assert_eq!(body["messages"][1]["reasoning_content"], "first then more");
    }

    /// A message's own fields are kept on the message whether or not it has text, and a streamed
    /// reply keeps them as the reference client folds them, so it is recorded as the same reply
    /// sent whole would be; the completion's own fields are the record's, stated once, with the raw
    /// usage, and without the model or the followed choice's message and stop, which the reply
    /// holds.
    #[tokio::test]
    async fn every_field_a_reply_returns_has_a_home_streamed_or_whole() {
        let provider = provider_for_test();
        let whole = serde_json::json!({
            "id": "gen-1",
            "object": "chat.completion",
            "model": "a-model",
            "provider": "Anthropic",
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls",
                "native_finish_reason": "tool_use",
                "message": {
                    "role": "assistant",
                    "content": null,
                    "reasoning": "Think",
                    "reasoning_details": [
                        {"index": 0, "type": "reasoning.text", "text": "Think", "signature": "S"},
                    ],
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "file_read", "arguments": "{}"},
                    }],
                },
            }],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4,
                "completion_tokens_details": {"reasoning_tokens": 3}},
        });
        let (message, ..) = provider
            .parse_non_streaming_response(&whole)
            .expect("parses");
        let expected_message_fields = serde_json::json!({"reasoning_details": [
            {"index": 0, "type": "reasoning.text", "text": "Think", "signature": "S"},
        ]});
        assert_eq!(
            message
                .provider_fields
                .clone()
                .map(serde_json::Value::Object),
            Some(expected_message_fields.clone()),
            "a reply with no text keeps its message's own fields"
        );
        let record = response_remainder(whole.as_object().expect("an object"), Some(0))
            .expect("the response's own fields");
        assert_eq!(record["id"], "gen-1");
        assert_eq!(record["provider"], "Anthropic");
        assert_eq!(
            record["usage"]["completion_tokens_details"]["reasoning_tokens"],
            3
        );
        assert!(!record.contains_key("model"));
        assert_eq!(record["choices"], serde_json::json!([{"index": 0}]));

        let chunk = |choice: serde_json::Value| {
            serde_json::json!({"id": "gen-1", "object": "chat.completion.chunk",
                "model": "a-model", "provider": "Anthropic", "choices": [choice]})
        };
        let (sender, mut receiver) = mpsc::channel::<StreamEvent>(64);
        let mut stream = ChatCompletionsStream::default();
        for data in [
            chunk(
                serde_json::json!({"index": 0, "delta": {"role": "assistant", "reasoning": "Th",
                "reasoning_details": [{"index": 0, "type": "reasoning.text", "text": "Th"}]}}),
            ),
            chunk(serde_json::json!({"index": 0, "delta": {"reasoning": "ink",
                "reasoning_details": [{"index": 0, "text": "ink", "signature": "S"}]}})),
            chunk(
                serde_json::json!({"index": 0, "delta": {"tool_calls": [{"index": 0,
                "id": "call_1", "type": "function",
                "function": {"name": "file_read", "arguments": "{}"}}]}}),
            ),
            chunk(serde_json::json!({"index": 0, "delta": {},
                "finish_reason": "tool_calls", "native_finish_reason": "tool_use"})),
        ] {
            handle_stream_chunk(&data, &mut stream, &sender).await;
        }
        conclude_stream(End::Finished, stream, &sender)
            .await
            .expect("a whole stream");
        drop(sender);
        let mut accumulator = crate::provider::MessageAccumulator::new();
        while let Some(event) = receiver.recv().await {
            accumulator.push(event);
        }
        let completion = accumulator.finish();
        assert_eq!(
            completion
                .message
                .provider_fields
                .map(serde_json::Value::Object),
            Some(expected_message_fields),
            "streamed, the message's own fields fold to what the whole reply carries"
        );
        let record = completion
            .response_fields
            .expect("the response's own fields");
        assert_eq!(record["id"], "gen-1", "stated once, not run together");
        assert_eq!(record["provider"], "Anthropic");
        assert_eq!(record["choices"], serde_json::json!([{"index": 0}]));
    }

    /// A reply recorded with nothing in it goes back as the wire's own empty message.
    #[test]
    fn an_empty_reply_goes_out_as_an_empty_message() {
        let messages = [
            Message::user("hi"),
            crate::conversation::Message {
                role: crate::conversation::Role::Assistant,
                content: Vec::new(),
                provider_fields: None,
            },
            Message::user("again"),
        ];
        let body = provider_for_test().build_request_body("", &messages, &[], false);
        assert_eq!(
            body["messages"][1],
            serde_json::json!({"role": "assistant", "content": ""})
        );
        assert_eq!(body["messages"][2]["role"], "user");
    }

    /// A provider against the default endpoint with nothing set on the profile.
    fn provider_for_test() -> OpenAiChatCompletionsProvider {
        let api_key: String = "test-key".to_string();
        OpenAiChatCompletionsProvider::from_settings(
            api_key.clone(),
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::OpenAiChatCompletions,
                crate::store::AuthCredential::ApiKey(api_key),
                "gpt-4o".to_string(),
            )
            .base_url(None)
            .effort(None)
            .max_output_tokens(None),
        )
        .expect("build test provider")
    }

    /// What Meka writes into a conversation reaches this wire rendered: the summary, a note behind
    /// Meka's label, a call's arguments as the model sent them, which the spec defines as a
    /// string, and a result's note, caption and large output's reference in its one string.
    #[test]
    fn what_meka_wrote_reaches_the_wire_rendered() {
        let body = provider_for_test().build_request_body(
            "",
            &Message::typed_content_for_test(),
            &[],
            true,
        );
        let sent = body["messages"].to_string();
        for needle in [
            "the summary",
            "[Meka harness] the note",
            r#""arguments":"{\"path\": ""#,
            "the output\\n\\n[Meka harness] the schema note[Image: a.png]<large-output",
        ] {
            assert!(sent.contains(needle), "{needle} is not sent: {sent}");
        }
    }

    /// An invocation and an attached resource are text on this wire, in the order the message
    /// holds them, so the model reads the skill's body and the file among the words.
    #[test]
    fn an_invocation_and_a_resource_are_read_in_place() {
        let messages = [Message::invoked_with_a_resource_for_test()];
        let body = provider_for_test().build_request_body("", &messages, &[], true);
        let sent = body["messages"].to_string();
        let at = |needle: &str| {
            sent.find(needle)
                .unwrap_or_else(|| panic!("{needle} is not sent: {sent}"))
        };
        assert!(at("the skill's body") < at("compare "), "{sent}");
        assert!(at("compare ") < at("fn a() {}"), "{sent}");
        assert!(at("fn a() {}") < at(" with the old one"), "{sent}");
    }

    /// A streamed request asks for its usage unless the account says its endpoint refuses the
    /// field, and an unstreamed one never asks.
    #[test]
    fn a_streamed_request_asks_for_its_usage_unless_the_account_turns_it_off() {
        let messages = [Message::user("hi")];
        let body = provider_for_test().build_request_body("", &messages, &[], true);
        assert_eq!(body["stream_options"]["include_usage"], true, "{body}");
        let body = provider_for_test().build_request_body("", &messages, &[], false);
        assert!(body.get("stream_options").is_none(), "{body}");
        let mut off = provider_for_test();
        off.stream_usage = false;
        let body = off.build_request_body("", &messages, &[], true);
        assert!(body.get("stream_options").is_none(), "{body}");
    }

    /// Drives [`handle_stream_chunk`] over a sequence of chunks and returns everything it produced,
    /// mirroring what the read loop in `stream_completion` does with a live SSE stream.
    async fn drive_chunks(
        chunks: &[serde_json::Value],
    ) -> (
        Vec<StreamEvent>,
        std::collections::HashMap<i64, ToolCallAccumulator>,
        Option<Stop>,
    ) {
        let (sender, mut receiver) = mpsc::channel::<StreamEvent>(64);
        let mut stream = ChatCompletionsStream::default();

        for chunk in chunks {
            handle_stream_chunk(chunk, &mut stream, &sender).await;
        }
        drop(sender);

        let mut events = Vec::new();
        while let Some(event) = receiver.recv().await {
            events.push(event);
        }
        (events, stream.tool_call_accumulators, stream.final_stop)
    }

    /// One streaming chunk whose `delta` holds the given fields.
    fn delta_chunk(fields: &[(&str, serde_json::Value)]) -> serde_json::Value {
        let delta: serde_json::Map<String, serde_json::Value> = fields
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect();
        serde_json::json!({"choices": [{"index": 0, "delta": delta}]})
    }

    /// A chunk states the usage whole, and `"usage": null` on a later chunk states nothing, so the
    /// reply keeps the usage the stream stated.
    #[tokio::test]
    async fn a_null_usage_on_a_later_chunk_keeps_the_stated_one() {
        let (events, ..) = drive_chunks(&[
            delta_chunk(&[("content", serde_json::json!("hi"))]),
            serde_json::json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            serde_json::json!({"choices": [],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4}}),
            serde_json::json!({"choices": [], "usage": null}),
        ])
        .await;
        let mut accumulator = crate::provider::MessageAccumulator::new();
        for event in events {
            accumulator.push(event);
        }
        let usage = accumulator.finish().usage;
        assert_eq!((usage.input_tokens, usage.output_tokens), (9, 4));
    }

    /// The reasoning an endpoint streams ahead of its answer, under either spelling of the field,
    /// becomes one thinking block that carries its own text for replay and closes when the answer
    /// begins, so the block lands ahead of the text.
    #[tokio::test]
    async fn streamed_reasoning_is_a_verbatim_thinking_block_ahead_of_the_text() {
        for field in ["reasoning_content", "reasoning"] {
            let (events, _, final_stop) = drive_chunks(&[
                delta_chunk(&[
                    ("role", serde_json::json!("assistant")),
                    (field, serde_json::json!("let me")),
                    ("content", serde_json::Value::Null),
                ]),
                delta_chunk(&[(field, serde_json::json!(" think"))]),
                delta_chunk(&[
                    (field, serde_json::json!("")),
                    ("content", serde_json::json!("The answer")),
                ]),
                serde_json::json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            ])
            .await;
            let shape: Vec<String> = events
                .iter()
                .map(|event| match event {
                    StreamEvent::ThinkingDelta(text) => format!("thinking:{text}"),
                    StreamEvent::ThinkingComplete { opaque, .. } => format!("complete:{opaque:?}"),
                    StreamEvent::TextDelta(text) => format!("text:{text}"),
                    other => format!("{other:?}"),
                })
                .collect();
            assert_eq!(
                shape,
                vec![
                    "thinking:let me",
                    "thinking: think",
                    "complete:Some(Verbatim)",
                    "text:The answer",
                ],
                "{field}"
            );
            assert_eq!(
                final_stop.map(|stop| stop.reason),
                Some(StopReason::EndTurn),
                "{field}"
            );
        }
    }

    /// A tool call ends the reasoning the same way text does: the block closes before the call's
    /// fragments are taken, so the thinking block precedes the call in the recorded message.
    #[tokio::test]
    async fn a_tool_call_delta_closes_the_open_reasoning_block() {
        let (events, accumulators, _) = drive_chunks(&[
            delta_chunk(&[("reasoning_content", serde_json::json!("which file?"))]),
            delta_chunk(&[(
                "tool_calls",
                serde_json::json!([{
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "file_read", "arguments": "{\"path\":\"a\"}"}
                }]),
            )]),
        ])
        .await;
        assert!(
            matches!(
                events.as_slice(),
                [
                    StreamEvent::ThinkingDelta(text),
                    StreamEvent::ThinkingComplete {
                        opaque: Some(OpaqueReasoning::Verbatim), .. }
                ] if text == "which file?"
            ),
            "{events:?}"
        );
        assert_eq!(
            accumulators.get(&0).and_then(|call| call.name.as_deref()),
            Some("file_read")
        );
    }

    /// Reasoning that nothing followed, a turn cut by `length` mid-thought, still becomes a block
    /// when the stream ends: the wire marks no end of reasoning, so the end of the stream is it.
    #[tokio::test]
    async fn the_end_of_the_stream_closes_reasoning_that_nothing_followed() {
        let (sender, mut receiver) = mpsc::channel::<StreamEvent>(8);
        let protocol = ChatCompletionsStream {
            final_stop: Some(Stop::of(StopReason::MaxTokens)),
            reasoning_open: true,
            ..Default::default()
        };
        conclude_stream(End::Finished, protocol, &sender)
            .await
            .expect("a stream that said how it stopped");
        drop(sender);
        let mut events = Vec::new();
        while let Some(event) = receiver.recv().await {
            events.push(event);
        }
        assert!(
            matches!(events.as_slice(), [
                StreamEvent::ThinkingComplete {
                    opaque: Some(OpaqueReasoning::Verbatim),
                    ..
                },
                StreamEvent::MessageEnd {
                    stop: crate::provider::Stop {
                        reason: StopReason::MaxTokens,
                        ..
                    },
                }
            ]),
            "{events:?}"
        );
    }

    /// A non-streaming reply's reasoning takes the same block, ahead of the text, under either
    /// spelling, so a `complete` and a streamed turn record the same message.
    #[test]
    fn a_non_streaming_reply_keeps_its_reasoning_as_a_verbatim_thinking_block() {
        let provider = provider_for_test();
        for field in ["reasoning_content", "reasoning"] {
            let response = serde_json::json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Hello there!",
                        field: "a greeting is expected",
                    },
                    "finish_reason": "stop"
                }]
            });
            let (message, ..) = provider
                .parse_non_streaming_response(&response)
                .expect("should parse");
            assert!(
                matches!(
                    message.content.as_slice(),
                    [
                        ContentBlock::Thinking {
                            parts,
                            opaque: Some(OpaqueReasoning::Verbatim), .. },
                        ContentBlock::Text { text, .. }
                    ] if crate::conversation::reasoning_text(parts) == "a greeting is expected"
                        && text == "Hello there!"
                ),
                "{field}: {:?}",
                message.content
            );
        }
    }

    /// Readable reasoning goes back as the assistant message's `reasoning_content` whichever
    /// endpoint returned it, with and without tool calls, because DeepSeek's thinking mode rejects
    /// a request with tools that omits an earlier turn's reasoning; a Claude signature or a sealed
    /// Responses blob is nothing this endpoint can read and never reaches the wire.
    #[test]
    fn readable_reasoning_from_any_endpoint_is_replayed_as_reasoning_content() {
        let provider = provider_for_test();
        let thinking = |opaque: Option<OpaqueReasoning>| ContentBlock::Thinking {
            parts: vec![crate::conversation::ReasoningPart::reasoning(
                "the reasoning",
            )],
            opaque,
            provider_fields: None,
        };
        let call = ContentBlock::ToolUse {
            id: "call_1".to_string(),
            name: "file_read".to_string(),
            input: serde_json::json!({"path": "a"}),
            unparsed: None,
            provider_fields: None,
        };
        let text = ContentBlock::Text {
            text: "answer".to_string(),
            provider_fields: None,
        };

        for content in [
            vec![thinking(Some(OpaqueReasoning::Verbatim)), text.clone()],
            vec![thinking(Some(OpaqueReasoning::Verbatim)), call.clone()],
        ] {
            let body = provider.build_request_body(
                "",
                &[Message::user("hi"), Message {
                    role: Role::Assistant,
                    content,
                    provider_fields: None,
                }],
                &[],
                false,
            );
            assert_eq!(
                body["messages"][1]["reasoning_content"], "the reasoning",
                "{body}"
            );
        }

        for opaque in [
            Some(OpaqueReasoning::Signed {
                signature: "MAC".to_string(),
            }),
            Some(OpaqueReasoning::Sealed {
                encrypted_content: "SEALED".to_string(),
                id: None,
                origin: Some("https://chatgpt.com".to_string()),
            }),
            None,
        ] {
            let body = provider.build_request_body(
                "",
                &[Message::user("hi"), Message {
                    role: Role::Assistant,
                    content: vec![thinking(opaque.clone()), text.clone()],
                    provider_fields: None,
                }],
                &[],
                false,
            );
            let serialized = serde_json::to_string(&body).expect("serialize");
            assert_eq!(
                body["messages"][1]["reasoning_content"], "the reasoning",
                "{opaque:?}: {serialized}"
            );
            assert!(
                !serialized.contains("MAC") && !serialized.contains("SEALED"),
                "{opaque:?}: {serialized}"
            );
        }
    }

    /// A streamed call whose `arguments` is not the string the spec defines is refused with what
    /// came, never run without it; and a reply that made calls without naming a `finish_reason`
    /// stopped for them, streamed or whole alike.
    #[tokio::test]
    async fn a_call_s_arguments_and_an_unstated_stop_read_one_way_streamed_or_whole() {
        let (sender, mut receiver) = mpsc::channel::<StreamEvent>(64);
        let mut protocol = ChatCompletionsStream::default();
        let chunk = serde_json::json!({"choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "id": "call_1", "type": "function",
            "function": {"name": "file_read", "arguments": {"path": "a"}}}]}}]});
        handle_stream_chunk(&chunk, &mut protocol, &sender).await;
        conclude_stream(End::Finished, protocol, &sender)
            .await
            .expect("the stream finished");
        drop(sender);
        let mut accumulator = crate::provider::MessageAccumulator::new();
        while let Some(event) = receiver.recv().await {
            accumulator.push(event);
        }
        let streamed = accumulator.finish();
        let [
            ContentBlock::ToolUse {
                unparsed: Some(unparsed),
                ..
            },
        ] = streamed.message.content.as_slice()
        else {
            panic!("{:?}", streamed.message.content);
        };
        assert_eq!(unparsed.text.as_deref(), Some("{\"path\":\"a\"}"));
        assert_eq!(streamed.stop.reason, StopReason::ToolUse);

        let (_, whole, _) = provider_for_test()
            .parse_non_streaming_response(&serde_json::json!({"choices": [{"message": {
                "role": "assistant", "content": null, "tool_calls": [{"id": "call_1",
                "type": "function", "function": {"name": "file_read", "arguments": "{}"}}]}}]}))
            .expect("parses");
        assert_eq!(whole.reason, StopReason::ToolUse);
    }

    /// A stream read to its end with no `finish_reason` is truncated: the failure is said ahead of
    /// the calls it held, which a retry would otherwise find already shown.
    #[tokio::test]
    async fn a_truncated_stream_says_it_failed_ahead_of_the_calls_it_held() {
        let (sender, mut receiver) = mpsc::channel::<StreamEvent>(64);
        let mut protocol = ChatCompletionsStream::default();
        let chunk = serde_json::json!({"choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "id": "call_1", "type": "function",
            "function": {"name": "file_read", "arguments": "{}"}}]}}]});
        handle_stream_chunk(&chunk, &mut protocol, &sender).await;
        let ended = conclude_stream(End::Ended, protocol, &sender).await;
        assert!(ended.is_err(), "a truncated stream fails");
        drop(sender);
        let mut kinds = Vec::new();
        while let Some(event) = receiver.recv().await {
            kinds.extend(match event {
                StreamEvent::Error(_) => Some("error"),
                StreamEvent::ToolUseStart { .. } => Some("start"),
                StreamEvent::ToolUseEnd { .. } => Some("end"),
                _ => None,
            });
        }
        assert_eq!(kinds, ["error", "start", "end"]);
    }

    /// A streamed call is read as the whole reply's call is: one no fragment names an id for is
    /// kept as its fragments came and not run, one no fragment carries arguments for is refused
    /// for it, and an empty id names nothing, so the fragment that names the call fills it in.
    #[tokio::test]
    async fn a_streamed_call_is_read_as_a_whole_reply_s_call_is() {
        let (sender, mut receiver) = mpsc::channel::<StreamEvent>(64);
        let mut protocol = ChatCompletionsStream::default();
        let unnamed = serde_json::json!({"index": 0, "type": "function",
            "function": {"name": "file_read", "arguments": "{}"}});
        let bare = serde_json::json!({"index": 1, "id": "", "type": "function",
            "function": {"name": "file_read"}});
        let named_later = serde_json::json!({"index": 1, "id": "call_2"});
        for call in [&unnamed, &bare, &named_later] {
            let chunk =
                serde_json::json!({"choices": [{"index": 0, "delta": {"tool_calls": [call]}}]});
            handle_stream_chunk(&chunk, &mut protocol, &sender).await;
        }
        conclude_stream(End::Finished, protocol, &sender)
            .await
            .expect("the stream finished");
        drop(sender);
        let mut accumulator = crate::provider::MessageAccumulator::new();
        while let Some(event) = receiver.recv().await {
            accumulator.push(event);
        }
        let completion = accumulator.finish();
        let [
            ContentBlock::Unrecognized { parts },
            ContentBlock::ToolUse {
                id,
                unparsed: Some(_),
                ..
            },
        ] = completion.message.content.as_slice()
        else {
            panic!("{:?}", completion.message.content);
        };
        assert_eq!(parts, std::slice::from_ref(&unnamed));
        assert_eq!(id, "call_2");
    }

    /// A stream that breaks off still delivers what arrived before the break, its calls among it,
    /// so a failed reply is recorded with them; the attempt fails all the same.
    #[tokio::test]
    async fn a_stream_that_breaks_off_delivers_the_calls_that_arrived() {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral port");
        let port = listener.local_addr().expect("the bound address").port();
        tokio::spawn(async move {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let mut request = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                let read = match socket.read(&mut chunk).await {
                    Ok(0) => break,
                    Ok(read) => read,
                    Err(_) => return,
                };
                request.extend_from_slice(&chunk[..read]);
                let Some(head_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
                else {
                    continue;
                };
                let length: usize = String::from_utf8_lossy(&request[..head_end])
                    .split("\r\n")
                    .filter_map(|line| line.split_once(':'))
                    .find(|(name, _)| name.trim().eq_ignore_ascii_case("content-length"))
                    .and_then(|(_, value)| value.trim().parse().ok())
                    .unwrap_or(0);
                if request.len() >= head_end + 4 + length {
                    break;
                }
            }
            let frames = concat!(
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,",
                "\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"file_read\",",
                "\"arguments\":\"{\\\"path\\\":\\\"a\\\"}\"}}]}}]}\n\n",
                "data: {not json\n\n",
            );
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n{frames}",
                frames.len()
            );
            if socket.write_all(response.as_bytes()).await.is_err() {
                return;
            }
            if socket.shutdown().await.is_err() {
                tracing::debug!("mock endpoint failed to shut its socket down cleanly");
            }
        });

        let provider = OpenAiChatCompletionsProvider::from_settings(
            "test-key".to_string(),
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::OpenAiChatCompletions,
                crate::store::AuthCredential::ApiKey("test-key".to_string()),
                "gpt-4o".to_string(),
            )
            .base_url(Some(format!("http://127.0.0.1:{port}")))
            .effort(None)
            .max_output_tokens(None),
        )
        .expect("build test provider");
        let (sender, mut receiver) = mpsc::channel::<StreamEvent>(64);
        let ended = provider
            .stream(
                crate::provider::CompletionRequest::new("", &[Message::user("hello")], &[]),
                sender,
                tokio_util::sync::CancellationToken::new(),
            )
            .await;
        assert!(ended.is_err(), "the attempt fails");
        let mut accumulator = crate::provider::MessageAccumulator::new();
        let mut said_failed = false;
        while let Some(event) = receiver.recv().await {
            if matches!(event, StreamEvent::ToolUseStart { .. }) {
                assert!(
                    said_failed,
                    "the failure is said ahead of what the stream held"
                );
            }
            said_failed |= matches!(event, StreamEvent::Error(_));
            accumulator.push(event);
        }
        let partial = accumulator.partial().expect("what arrived");
        let [ContentBlock::ToolUse { id, input, .. }] = partial.message.content.as_slice() else {
            panic!("{:?}", partial.message.content);
        };
        assert_eq!(id, "call_1");
        assert_eq!(input, &serde_json::json!({"path": "a"}));
    }

    /// A gateway that ends the response after the final `finish_reason` chunk and never writes
    /// `[DONE]` has delivered a complete message; failing it dropped every tool call of the turn.
    /// A close with no `finish_reason` seen is still the truncation the check exists for.
    #[tokio::test]
    async fn a_close_after_finish_reason_completes_the_message() {
        let (sender, mut receiver) = mpsc::channel::<StreamEvent>(8);
        let protocol = ChatCompletionsStream {
            final_stop: Some(Stop::of(StopReason::EndTurn)),
            ..Default::default()
        };
        conclude_stream(End::Ended, protocol, &sender)
            .await
            .expect("the stop reason was seen, so the message is whole");
        drop(sender);
        let mut events = Vec::new();
        while let Some(event) = receiver.recv().await {
            events.push(event);
        }
        assert!(
            matches!(events.as_slice(), [StreamEvent::MessageEnd {
                stop: crate::provider::Stop {
                    reason: StopReason::EndTurn,
                    ..
                },
            }]),
            "{events:?}"
        );

        let (sender, _receiver) = mpsc::channel::<StreamEvent>(8);
        let error = conclude_stream(End::Ended, ChatCompletionsStream::default(), &sender)
            .await
            .expect_err("no stop reason and no terminal frame is a truncated stream");
        assert!(matches!(error, MekaError::StreamError(_)), "{error:?}");
    }

    /// A profile that sends `n` gets its answers interleaved, one choice per chunk under its own
    /// `index`; the first is the answer, and the others are not mixed into it.
    #[tokio::test]
    async fn a_second_answer_streamed_under_n_is_left_out() {
        let (events, _, final_stop) = drive_chunks(&[
            serde_json::json!({"choices": [{"index": 0, "delta": {"content": "first"}}]}),
            serde_json::json!({"choices": [{"index": 1, "delta": {"content": "second"}}]}),
            serde_json::json!({"choices": [{"index": 1, "delta": {}, "finish_reason": "length"}]}),
            serde_json::json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        ])
        .await;
        let text: String = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::TextDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "first");
        assert_eq!(
            final_stop.map(|stop| stop.reason),
            Some(StopReason::EndTurn)
        );

        // A server that numbers its only choice otherwise is still followed, an empty first chunk
        // fixing nothing.
        let (events, ..) = drive_chunks(&[
            serde_json::json!({"choices": []}),
            serde_json::json!({"choices": [{"index": 1, "delta": {"content": "only"}}]}),
        ])
        .await;
        assert!(
            matches!(events.as_slice(), [StreamEvent::TextDelta(text)] if text == "only"),
            "{events:?}"
        );
    }

    /// Every field this builder writes, streamed or whole, is one a profile's `parameters` may not
    /// name, and the parameters go out on both.
    #[test]
    fn the_reserved_fields_are_what_the_builder_writes() {
        let provider = OpenAiChatCompletionsProvider::from_settings(
            "test-key".to_string(),
            crate::provider::ProviderBuilder::new(
                crate::config::Backend::OpenAiChatCompletions,
                crate::store::AuthCredential::ApiKey("test-key".to_string()),
                "gpt-4o".to_string(),
            )
            .effort(Some("high".to_string()))
            .max_output_tokens(Some(4096))
            .parameters(crate::provider::parameters_for_test()),
        )
        .expect("build test provider");
        let tools = [ToolDefinition::new(
            "read",
            "reads",
            serde_json::json!({"type": "object"}),
        )];
        let streamed = provider.build_request_body("s", &[Message::user("hi")], &tools, true);
        crate::provider::assert_writes_its_reserved_fields(
            crate::config::Backend::OpenAiChatCompletions,
            &streamed,
        );
        let whole = provider.build_request_body("s", &[Message::user("hi")], &tools, false);
        let reserved = crate::config::Backend::OpenAiChatCompletions.reserved_request_fields();
        for field in whole.as_object().expect("an object").keys() {
            assert!(
                reserved.iter().any(|(reserved, _)| reserved == field)
                    || crate::provider::parameters_for_test().contains_key(field),
                "a whole request writes `{field}`, which is not reserved"
            );
        }
        assert_eq!(whole["temperature"], 0.3);
    }

    /// `"error": null` rides on every chunk from some servers and is not a failure.
    #[tokio::test]
    async fn a_null_error_field_is_not_an_error() {
        let (events, _, final_stop) = drive_chunks(&[serde_json::json!({
            "error": null,
            "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]
        })])
        .await;
        assert!(
            matches!(
                events.as_slice(),
                [StreamEvent::ResponseFields(_), StreamEvent::TextDelta(text)] if text == "hi"
            ),
            "{events:?}"
        );
        assert_eq!(
            final_stop.map(|stop| stop.reason),
            Some(StopReason::EndTurn)
        );
    }

    /// vLLM-backed endpoints coalesce the final tool-call delta into the same chunk as
    /// `finish_reason`, rather than sending `finish_reason` alone with an empty delta the way
    /// OpenAI does. The chunk below is a real capture. Skipping the delta on a chunk that carries a
    /// `finish_reason` drops the tool call outright, leaving `StopReason::ToolUse` with no tool-use
    /// block: an empty reply in place of the call.
    #[tokio::test]
    async fn stream_chunk_keeps_tool_call_coalesced_with_finish_reason() {
        let chunk = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {
                    "content": null,
                    "tool_calls": [{
                        "id": "chatcmpl-tool-a445012e51c3a83d",
                        "type": "function",
                        "index": 0,
                        "function": {
                            "name": "mcp__exa__web_search_exa",
                            "arguments": "{\"numResults\": 8, \"query\": \"top global news\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let (_, accumulators, final_stop) = drive_chunks(&[chunk]).await;

        assert_eq!(
            final_stop.map(|stop| stop.reason),
            Some(StopReason::ToolUse)
        );
        let accumulator = accumulators
            .get(&0)
            .expect("tool call in a finish_reason chunk must still be accumulated");
        assert_eq!(
            accumulator.id.as_deref(),
            Some("chatcmpl-tool-a445012e51c3a83d")
        );
        assert_eq!(
            accumulator.name.as_deref(),
            Some("mcp__exa__web_search_exa")
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&accumulator.arguments)
                .expect("arguments parse")["query"],
            "top global news"
        );
    }

    /// The same coalescing applies to plain text: the tail of a response must not be swallowed
    /// because it shared a chunk with `finish_reason: "stop"`.
    #[tokio::test]
    async fn stream_chunk_keeps_text_coalesced_with_finish_reason() {
        let chunk = serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {"content": " final words"},
                "finish_reason": "stop"
            }]
        });

        let (events, _, final_stop) = drive_chunks(&[chunk]).await;

        assert_eq!(
            final_stop.map(|stop| stop.reason),
            Some(StopReason::EndTurn)
        );
        assert!(
            events.iter().any(
                |event| matches!(event, StreamEvent::TextDelta(text) if text == " final words")
            ),
            "text sharing a chunk with finish_reason must still stream, got {events:?}"
        );
    }

    /// The canonical OpenAI shape - tool call streamed across chunks, then `finish_reason` alone
    /// with an empty delta - must keep working. The same endpoint emits this shape too; which of
    /// the two arrives is a timing race.
    #[tokio::test]
    async fn stream_chunk_handles_finish_reason_in_its_own_chunk() {
        let chunks = [
            serde_json::json!({
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "id": "call_abc",
                            "index": 0,
                            "function": {"name": "shell_execute", "arguments": "{\"command\":"}
                        }]
                    },
                    "finish_reason": null
                }]
            }),
            serde_json::json!({
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": " \"pwd\"}"}
                        }]
                    },
                    "finish_reason": null
                }]
            }),
            serde_json::json!({
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
            }),
        ];

        let (_, accumulators, final_stop) = drive_chunks(&chunks).await;

        assert_eq!(
            final_stop.map(|stop| stop.reason),
            Some(StopReason::ToolUse)
        );
        let accumulator = accumulators.get(&0).expect("accumulated tool call");
        assert_eq!(accumulator.id.as_deref(), Some("call_abc"));
        assert_eq!(accumulator.name.as_deref(), Some("shell_execute"));
        assert_eq!(accumulator.arguments, "{\"command\": \"pwd\"}");
    }

    #[test]
    fn an_openai_base_url_is_normalized_at_construction() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-4o".to_string(),
                )
                .base_url(Some("https://openrouter.ai/api/v1/".to_string()))
                .effort(None)
                .max_output_tokens(None),
            )
        }
        .expect("build test provider");
        // Without this the request path would carry a doubled separator, since the endpoint is
        // appended as `{base}/chat/completions`.
        assert_eq!(provider.base_url, "https://openrouter.ai/api/v1");

        // The version segment belongs in an OpenAI-family base and must survive.
        let default = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-4o".to_string(),
                )
                .base_url(None)
                .effort(None)
                .max_output_tokens(None),
            )
        }
        .expect("build test provider");
        assert_eq!(default.base_url, "https://api.openai.com/v1");
    }

    /// The context block and the words are one `content` string with a blank line between, the
    /// context first: this wire has no typed blocks to keep them apart.
    #[test]
    fn the_context_block_precedes_the_words() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-4o".to_string(),
                )
                .base_url(None)
                .effort(None)
                .max_output_tokens(None),
            )
        }
        .expect("build test provider");
        let messages = vec![Message::user_turn("ctx", "hello", Vec::new())];
        let body = provider.build_request_body("", &messages, &[], false);
        assert_eq!(body["messages"][0]["content"], "ctx\n\nhello");
    }

    #[test]
    fn openai_request_body_simple() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-4o".to_string(),
                )
                .base_url(None)
                .effort(None)
                .max_output_tokens(None),
            )
        }
        .expect("build test provider");

        let messages = vec![Message::user("hello")];
        let body = provider.build_request_body("system prompt", &messages, &[], false);

        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["stream"], false);

        let openai_messages = body["messages"]
            .as_array()
            .expect("messages should be array");
        assert_eq!(openai_messages.len(), 2);
        assert_eq!(openai_messages[0]["role"], "system");
        assert_eq!(openai_messages[0]["content"], "system prompt");
        assert_eq!(openai_messages[1]["role"], "user");
        assert_eq!(openai_messages[1]["content"], "hello");
    }

    #[test]
    fn openai_request_body_user_image_uses_content_array() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-4o".to_string(),
                )
                .base_url(None)
                .effort(None)
                .max_output_tokens(None),
            )
        }
        .expect("build test provider");
        let message =
            Message::user_with_images("what is this", vec![crate::image::ImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: "QUJD".to_string(),
            }]);
        let body = provider.build_request_body("", &[message], &[], false);
        let user = &body["messages"].as_array().expect("messages")[0];
        assert_eq!(user["role"], "user");
        let parts = user["content"].as_array().expect("content array");
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "what is this");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,QUJD");
    }

    /// A profile that states `max_request_bytes` is held to it here too. This backend renders a
    /// tool-result image as the text `[Image]`, so there is nothing for the redaction to remove;
    /// what the ceiling still buys is the refusal: a body the endpoint would reject is refused
    /// before the send as `InvalidRequest`, which the turn answers by degrading its own
    /// attachments instead of failing every later turn against the same body. Without a ceiling
    /// the body is sent as built.
    #[test]
    fn a_stated_ceiling_refuses_an_oversize_body_here_too() {
        let with_ceiling = |max_request_bytes: Option<usize>| {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-4o".to_string(),
                )
                .base_url(None)
                .effort(None)
                .max_output_tokens(None)
                .max_request_bytes(max_request_bytes),
            )
            .expect("build test provider")
        };
        let image = "A".repeat(8_000);
        let messages = vec![Message::user_with_images("what is this", vec![
            crate::image::ImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: image.clone(),
            },
        ])];

        let refused = with_ceiling(Some(4_000))
            .request_body_within_budget("", &messages, &[], false)
            .expect_err("the newest message's image cannot be redacted, so the ceiling refuses");
        // Meka's own ceiling, so `RequestTooLarge` rather than the provider's `InvalidRequest`;
        // both arm the degrade-and-retry, and only one of them claims an upstream judged this.
        assert!(
            matches!(refused, MekaError::RequestTooLarge(_)),
            "the refusal arms the degrade-and-retry: {refused:?}"
        );

        let (body, notice) = with_ceiling(None)
            .request_body_within_budget("", &messages, &[], false)
            .expect("no ceiling, no refusal");
        assert!(body.contains(&image), "sent as built without a ceiling");
        assert!(notice.is_none());
    }

    #[test]
    fn openai_request_body_max_output_tokens_override_without_effort() {
        // No reasoning_effort, but an explicit cap: `max_completion_tokens` is set.
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-4o".to_string(),
                )
                .base_url(None)
                .effort(None)
                .max_output_tokens(Some(8_000)),
            )
        }
        .expect("build test provider");
        let body = provider.build_request_body("", &[Message::user("hi")], &[], false);
        assert_eq!(body["max_completion_tokens"], 8_000);
    }

    /// `effort` alone sends no output cap: the endpoint's default is its own fact, and a guessed
    /// 32k floor was refused by every model with a smaller cap.
    #[test]
    fn openai_request_body_effort_alone_sends_no_output_cap() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-5".to_string(),
                )
                .base_url(None)
                .effort(Some("high".to_string()))
                .max_output_tokens(None),
            )
        }
        .expect("build test provider");
        let body = provider.build_request_body("", &[Message::user("hi")], &[], false);
        assert_eq!(body["reasoning_effort"], "high");
        assert!(body.get("max_completion_tokens").is_none(), "{body}");
    }

    #[test]
    fn openai_request_body_max_output_tokens_override_wins_over_effort_default() {
        // With effort and a profile cap, the cap is sent alongside.
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-5".to_string(),
                )
                .base_url(None)
                .effort(Some("high".to_string()))
                .max_output_tokens(Some(120_000)),
            )
        }
        .expect("build test provider");
        let body = provider.build_request_body("", &[Message::user("hi")], &[], false);
        assert_eq!(body["max_completion_tokens"], 120_000);
        assert_eq!(body["reasoning_effort"], "high");
    }

    #[test]
    fn openai_request_body_no_cap_without_effort_or_override() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-4o".to_string(),
                )
                .base_url(None)
                .effort(None)
                .max_output_tokens(None),
            )
        }
        .expect("build test provider");
        let body = provider.build_request_body("", &[Message::user("hi")], &[], false);
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn an_unconfigured_profile_sends_no_reasoning_effort_whatever_the_model() {
        // Recognized reasoning model or local weights, the answer is the same: OpenAI owns the
        // default and Meka asks for it by omitting the field.
        for model in ["gpt-6-astra", "o3", "llama3.1"] {
            let provider = {
                let api_key: String = "test-key".to_string();
                OpenAiChatCompletionsProvider::from_settings(
                    api_key.clone(),
                    crate::provider::ProviderBuilder::new(
                        crate::config::Backend::OpenAiChatCompletions,
                        crate::store::AuthCredential::ApiKey(api_key),
                        model.to_string(),
                    )
                    .base_url(None)
                    .effort(None)
                    .max_output_tokens(None),
                )
            }
            .expect("build test provider");
            let body = provider.build_request_body("", &[Message::user("hi")], &[], false);
            assert!(body.get("reasoning_effort").is_none(), "{model}");
        }
        // A configured value is absolute, including on a model Meka does not recognize.
        let configured = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "llama3.1".to_string(),
                )
                .base_url(None)
                .effort(Some("low".to_string()))
                .max_output_tokens(None),
            )
        }
        .expect("build test provider");
        let body = configured.build_request_body("", &[Message::user("hi")], &[], false);
        assert_eq!(body["reasoning_effort"], "low");
    }

    #[test]
    fn openai_request_body_with_tools() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-4o".to_string(),
                )
                .base_url(None)
                .effort(None)
                .max_output_tokens(None),
            )
        }
        .expect("build test provider");

        let tools = vec![ToolDefinition::new(
            "file_read".to_string(),
            "Read a file".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        )];

        let body = provider.build_request_body("", &[], &tools, false);
        let openai_tools = body["tools"].as_array().expect("tools should be array");
        assert_eq!(openai_tools.len(), 1);
        assert_eq!(openai_tools[0]["type"], "function");
        assert_eq!(openai_tools[0]["function"]["name"], "file_read");
    }

    #[test]
    fn openai_request_body_with_tool_calls() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-4o".to_string(),
                )
                .base_url(None)
                .effort(None)
                .max_output_tokens(None),
            )
        }
        .expect("build test provider");

        let messages = vec![
            Message::user("read /tmp/test.txt"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "file_read".to_string(),
                    input: serde_json::json!({"path": "/tmp/test.txt"}),
                    unparsed: None,
                    provider_fields: None,
                }],
                provider_fields: None,
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "file contents here".to_string(),
                    }],
                    is_error: false,
                    display: None,
                }],
                provider_fields: None,
            },
        ];

        let body = provider.build_request_body("", &messages, &[], false);
        let openai_messages = body["messages"]
            .as_array()
            .expect("messages should be array");

        assert_eq!(openai_messages[0]["role"], "user");
        assert_eq!(openai_messages[1]["role"], "assistant");
        assert!(openai_messages[1].get("tool_calls").is_some());
        assert_eq!(openai_messages[2]["role"], "tool");
        assert_eq!(openai_messages[2]["tool_call_id"], "call_1");
    }

    /// An inbox item read at a round boundary is text beside the round's tool results. This wire
    /// has no room for it inside a `tool` message, so it follows them as a `user` message; before
    /// this the branch that emitted tool messages dropped every text block beside them.
    #[test]
    fn text_beside_tool_results_follows_the_tool_messages_as_a_user_message() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-4o".to_string(),
                )
                .base_url(None)
                .effort(None)
                .max_output_tokens(None),
            )
        }
        .expect("build test provider");
        let messages = vec![
            Message::user("read /tmp/test.txt"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "file_read".to_string(),
                    input: serde_json::json!({"path": "/tmp/test.txt"}),
                    unparsed: None,
                    provider_fields: None,
                }],
                provider_fields: None,
            },
            Message {
                role: Role::User,
                content: vec![
                    ContentBlock::ToolResult {
                        tool_use_id: "call_1".to_string(),
                        content: vec![ToolResultContent::Text {
                            text: "file contents here".to_string(),
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

        let body = provider.build_request_body("", &messages, &[], false);
        let openai_messages = body["messages"]
            .as_array()
            .expect("messages should be array");

        assert_eq!(openai_messages[2]["role"], "tool");
        assert_eq!(openai_messages[2]["tool_call_id"], "call_1");
        assert_eq!(openai_messages[3]["role"], "user");
        assert_eq!(
            openai_messages[3]["content"],
            "[Message from test, arrived now]\nalso, what is 17*3?"
        );
        assert_eq!(
            openai_messages.len(),
            4,
            "tool results alone add no user message"
        );
    }

    #[test]
    fn openai_parse_non_streaming_text() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-4o".to_string(),
                )
                .base_url(None)
                .effort(None)
                .max_output_tokens(None),
            )
        }
        .expect("build test provider");

        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello there!"
                },
                "finish_reason": "stop"
            }]
        });

        let (message, stop, _) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");

        assert_eq!(message.text_content(), "Hello there!");
        assert_eq!(stop.reason, StopReason::EndTurn);
    }

    #[test]
    fn openai_parse_non_streaming_tool_call() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-4o".to_string(),
                )
                .base_url(None)
                .effort(None)
                .max_output_tokens(None),
            )
        }
        .expect("build test provider");

        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "file_read",
                            "arguments": "{\"path\":\"/tmp/test.txt\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let (message, stop, _) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");

        assert_eq!(stop.reason, StopReason::ToolUse);
        let tool_uses = message.tool_uses();
        assert_eq!(tool_uses.len(), 1);

        if let ContentBlock::ToolUse {
            id, name, input, ..
        } = &tool_uses[0]
        {
            assert_eq!(id, "call_abc");
            assert_eq!(name, "file_read");
            assert_eq!(input["path"], "/tmp/test.txt");
        } else {
            panic!("expected ToolUse block");
        }
    }

    #[test]
    fn openai_parse_non_streaming_malformed_tool_args() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-4o".to_string(),
                )
                .base_url(None)
                .effort(None)
                .max_output_tokens(None),
            )
        }
        .expect("build test provider");

        let response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "file_read",
                            "arguments": "{not valid json"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let (message, ..) = provider
            .parse_non_streaming_response(&response)
            .expect("envelope should parse even with bad tool args");

        let tool_uses = message.tool_uses();
        assert_eq!(tool_uses.len(), 1);
        if let ContentBlock::ToolUse {
            input, unparsed, ..
        } = &tool_uses[0]
        {
            assert_eq!(input, &serde_json::json!({}));
            let unparsed = unparsed.as_ref().expect("malformed arguments are refused");
            assert_eq!(unparsed.text.as_deref(), Some("{not valid json"));
            assert!(unparsed.error.starts_with("invalid JSON arguments"));
        } else {
            panic!("expected ToolUse block");
        }
    }

    #[test]
    fn openai_parse_missing_message_in_choice() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-4o".to_string(),
                )
                .base_url(None)
                .effort(None)
                .max_output_tokens(None),
            )
        }
        .expect("build test provider");

        let response = serde_json::json!({
            "choices": [{
                "finish_reason": "stop"
            }]
        });

        let result = provider.parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    /// A call the spec does not allow, without its `id` or its `function.name`, or with the name
    /// and arguments beside `function` rather than in it, is kept as it came and never run, and the
    /// reply still parses.
    #[test]
    fn a_whole_call_the_spec_does_not_allow_is_kept_and_not_run() {
        let provider = provider_for_test();
        for call in [
            serde_json::json!({"type": "function", "function": {"name": "file_read", "arguments": "{}"}}),
            serde_json::json!({"id": "call_abc", "type": "function", "function": {"arguments": "{}"}}),
            serde_json::json!({"id": "call_abc", "type": "function", "name": "file_read",
                "arguments": "{\"path\":\"/tmp/test.txt\"}"}),
            serde_json::json!({"id": "call_abc", "type": "function", "arguments": "{}"}),
        ] {
            let response = serde_json::json!({
                "choices": [{
                    "message": {"role": "assistant", "content": null, "tool_calls": [call.clone()]},
                    "finish_reason": "tool_calls"
                }]
            });
            let (message, stop, _) = provider
                .parse_non_streaming_response(&response)
                .expect("parses");
            assert!(message.tool_uses().is_empty(), "{call}");
            assert!(
                matches!(&message.content[..], [ContentBlock::Unrecognized { parts }] if parts == std::slice::from_ref(&call)),
                "{call}: {:?}",
                message.content
            );
            assert_eq!(stop.reason, StopReason::ToolUse, "{call}");
        }
    }

    #[test]
    fn openai_tool_definitions_use_standard_chat_completions_format() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-4o".to_string(),
                )
                .base_url(None)
                .effort(None)
                .max_output_tokens(None),
            )
        }
        .expect("build test provider");

        let tools = vec![ToolDefinition::new(
            "file_write".to_string(),
            "Create or overwrite a file".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }),
        )];

        let body = provider.build_request_body("", &[], &tools, false);
        let openai_tools = body["tools"].as_array().expect("tools should be array");

        assert_eq!(openai_tools[0]["type"], "function");
        assert_eq!(openai_tools[0]["function"]["name"], "file_write");
        assert_eq!(
            openai_tools[0]["function"]["description"],
            "Create or overwrite a file"
        );
        assert!(openai_tools[0]["function"].get("parameters").is_some());

        // Top-level name/description/parameters must NOT be present to avoid triggering Responses
        // API strict validation on OpenAI/OpenRouter
        assert!(openai_tools[0].get("name").is_none());
        assert!(openai_tools[0].get("description").is_none());
        assert!(openai_tools[0].get("parameters").is_none());
    }

    /// An error object in a 200 stream ends the turn as a failure. Read past, it left the partial
    /// answer committed as complete with no error and no retry. The read loop announces the
    /// failure on the channel; this asserts the classification it hands back.
    #[tokio::test]
    async fn an_error_object_mid_stream_fails_the_turn() {
        let (sender, _receiver) = mpsc::channel::<StreamEvent>(8);
        let outcome = handle_stream_chunk(
            &serde_json::json!({"error": {"message": "upstream capacity", "type": "server_error"}}),
            &mut ChatCompletionsStream::default(),
            &sender,
        )
        .await;
        assert!(
            matches!(
                outcome,
                ChunkOutcome::Fail(MekaError::RetryableProvider { .. })
            ),
            "a server_error is the transient kind: {outcome:?}"
        );
    }

    /// A tool call whose id arrives after its first arguments, or never, is still one call.
    #[tokio::test]
    async fn a_tool_call_is_accumulated_before_its_id_arrives() {
        let (_events, accumulators, _) = drive_chunks(&[
            serde_json::json!({"choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "function": {"name": "file_read", "arguments": "{\"path\":"}
            }]}}]}),
            serde_json::json!({"choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "id": "call_1", "function": {"arguments": " \"a.txt\"}"}
            }]}}]}),
        ])
        .await;
        let accumulator = accumulators.get(&0).expect("one call accumulated");
        assert_eq!(accumulator.id.as_deref(), Some("call_1"));
        assert_eq!(accumulator.name.as_deref(), Some("file_read"));
        assert_eq!(accumulator.arguments, "{\"path\": \"a.txt\"}");
    }

    /// A call fragment without the `index` the spec places it by is kept as it came, each one
    /// whole, and opens no call to run.
    #[tokio::test]
    async fn an_index_less_call_fragment_is_kept_and_not_run() {
        let fragments = [
            serde_json::json!({"id": "call_a", "function": {"name": "file_read", "arguments": "{\"a\":1}"}}),
            serde_json::json!({"id": "call_b", "function": {"name": "file_read", "arguments": "{\"b\":2}"}}),
        ];
        let (events, accumulators, _) = drive_chunks(
            &fragments
                .iter()
                .map(|fragment| {
                    serde_json::json!({"choices": [{"index": 0, "delta": {"tool_calls": [fragment]}}]})
                })
                .collect::<Vec<_>>(),
        )
        .await;
        assert!(accumulators.is_empty(), "no call opened");
        let kept: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::Unrecognized(value) => Some(value.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(kept, fragments);
    }

    /// The `tool` message has no `is_error` field in this API; a strict endpoint rejects the
    /// request for one, and only on the turns where a tool failed.
    #[test]
    fn a_failed_tool_result_carries_no_is_error_key() {
        let provider = {
            let api_key: String = "key".to_string();
            OpenAiChatCompletionsProvider::from_settings(
                api_key.clone(),
                crate::provider::ProviderBuilder::new(
                    crate::config::Backend::OpenAiChatCompletions,
                    crate::store::AuthCredential::ApiKey(api_key),
                    "gpt-4o".to_string(),
                )
                .base_url(None)
                .effort(None)
                .max_output_tokens(None),
            )
        }
        .expect("provider");
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "file_read".to_string(),
                    input: serde_json::json!({"path": "x"}),
                    unparsed: None,
                    provider_fields: None,
                }],
                provider_fields: None,
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "Error: no such file".to_string(),
                    }],
                    is_error: true,
                    display: None,
                }],
                provider_fields: None,
            },
        ];
        let body = provider.build_request_body("system", &messages, &[], false);
        let tool_message = body["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .find(|message| message["role"] == "tool")
            .expect("the tool message");
        assert!(tool_message.get("is_error").is_none(), "{tool_message}");
        assert_eq!(tool_message["content"], "Error: no such file");
    }
}
