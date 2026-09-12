//! `openai-chat-completions`: the Chat Completions API against any endpoint serving it, with an
//! API key.
//!
//! `POST {base_url}/chat/completions`, reaching OpenAI and anything implementing that format
//! (Ollama, vLLM, LM Studio, OpenRouter, Synthetic, LiteLLM, local proxies). This is *not* the
//! legacy `/v1/completions`, a different protocol with no tool calling that several of those same
//! servers also expose and that meka does not implement.
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
    conversation::{ContentBlock, Message, Role},
    error::{MekaError, Result},
    provider::{
        CompletionRequest, Provider, StopReason, StreamEvent, ToolCallAccumulator, ToolDefinition,
        finalize_tool_call_accumulators,
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
}

impl OpenAiChatCompletionsProvider {
    /// `api_key` is the credential `settings` carries, already checked to be one by the builder.
    pub(crate) fn new(api_key: String, settings: crate::provider::ProviderBuilder) -> Result<Self> {
        let crate::provider::ProviderBuilder {
            model,
            base_url,
            effort: reasoning_effort,
            max_output_tokens,
            max_request_bytes,
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
                    let has_tool_results = message
                        .content
                        .iter()
                        .any(|block| matches!(block, ContentBlock::ToolResult { .. }));

                    if has_tool_results {
                        for block in &message.content {
                            if let ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                is_error: _,
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
                    } else {
                        // No `match` on `ContentBlock` here means the compiler can't force this
                        // path to handle `Image`; it must be done by hand. When the user message
                        // carries images, Chat Completions wants a `content` array of `text` +
                        // `image_url` parts (vision is user-role only); otherwise a plain string.
                        let has_images = message
                            .content
                            .iter()
                            .any(|block| matches!(block, ContentBlock::Image { .. }));
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
                            ContentBlock::Thinking { .. } => None,
                            ContentBlock::ToolUse { id, name, input } => Some(serde_json::json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": input.to_string(),
                                }
                            })),
                            _ => None,
                        })
                        .collect();

                    if tool_calls.is_empty() {
                        openai_messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": message.text_content(),
                        }));
                    } else {
                        let text = message.text_content();
                        let mut assistant_message = serde_json::json!({
                            "role": "assistant",
                            "tool_calls": tool_calls,
                        });
                        if !text.is_empty() {
                            assistant_message["content"] = serde_json::json!(text);
                        }
                        openai_messages.push(assistant_message);
                    }
                }
            }
        }

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": openai_messages,
            "stream": stream,
        });

        // OpenAI omits `usage` from streamed responses unless explicitly asked; without this the
        // final usage-only chunk never arrives and token accounting (the `/status` context gauge,
        // auto-compact) silently reads zero for streaming turns.
        if stream {
            body["stream_options"] = serde_json::json!({ "include_usage": true });
        }

        let reasoning_effort = self.wire_effort();
        if let Some(effort) = &reasoning_effort {
            body["reasoning_effort"] = serde_json::json!(effort);
        }
        // Only the profile's own cap: the endpoint's default is the endpoint's fact, and this
        // backend reaches whatever `base_url` names. A floor sent whenever `effort` is set would be
        // rejected by every model with a smaller output cap, and each rejection would cost a
        // degraded retry that fails the same way.
        if let Some(max_output) = self.max_output_tokens {
            body["max_completion_tokens"] = serde_json::json!(max_output);
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
            body["tools"] = serde_json::json!(openai_tools);
        }

        body
    }

    /// The assistant message, stop reason and usage out of a non-streaming response.
    pub(super) fn parse_non_streaming_response(
        &self,
        response: &serde_json::Value,
    ) -> Result<(Message, StopReason, TokenUsage)> {
        let choice = response
            .get("choices")
            .and_then(|choices| choices.get(0))
            .ok_or_else(|| MekaError::Provider("no choices in response".to_string()))?;

        let finish_reason = choice
            .get("finish_reason")
            .and_then(|reason| reason.as_str())
            .unwrap_or("stop");

        let stop_reason = parse_openai_stop_reason(finish_reason);

        let assistant_message = choice
            .get("message")
            .ok_or_else(|| MekaError::Provider("no 'message' in choice".to_string()))?;
        let mut content_blocks = Vec::new();

        if let Some(text) = assistant_message
            .get("content")
            .and_then(|content| content.as_str())
            && !text.is_empty()
        {
            content_blocks.push(ContentBlock::Text {
                text: text.to_string(),
            });
        }

        if let Some(tool_calls) = assistant_message
            .get("tool_calls")
            .and_then(|tool_calls| tool_calls.as_array())
        {
            for tool_call in tool_calls {
                let id = tool_call
                    .get("id")
                    .and_then(|id| id.as_str())
                    .ok_or_else(|| MekaError::Provider("tool call missing 'id' field".to_string()))?
                    .to_string();
                let name = tool_call
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(|name| name.as_str())
                    .or_else(|| tool_call.get("name").and_then(|name| name.as_str()))
                    .ok_or_else(|| {
                        MekaError::Provider("tool call missing 'function.name' field".to_string())
                    })?
                    .to_string();
                let arguments_str = tool_call
                    .get("function")
                    .and_then(|function| function.get("arguments"))
                    .and_then(|arguments| arguments.as_str())
                    .or_else(|| {
                        tool_call
                            .get("arguments")
                            .and_then(|arguments| arguments.as_str())
                    })
                    .unwrap_or("{}");
                content_blocks.push(
                    match crate::provider::finalize_tool_arguments(&name, arguments_str) {
                        Ok(input) => ContentBlock::ToolUse { id, name, input },
                        Err(reason) => crate::provider::rejected_tool_use(id, name, reason),
                    },
                );
            }
        }

        let token_usage = TokenUsage {
            input_tokens: response
                .get("usage")
                .and_then(|u| u.get("prompt_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            output_tokens: response
                .get("usage")
                .and_then(|u| u.get("completion_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            ..TokenUsage::default()
        };

        Ok((
            Message {
                role: Role::Assistant,
                content: content_blocks,
            },
            stop_reason,
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

        let status = response.status();
        let retry_after = crate::error::parse_retry_after(response.headers());
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

        let (message, stop_reason, usage) = self.parse_non_streaming_response(&response_json)?;
        Ok(crate::provider::Completion {
            message,
            stop_reason,
            usage,
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
        let end = crate::provider::sse::drive(
            response,
            "Chat Completions",
            &event_sender,
            &cancellation,
            &mut protocol,
        )
        .await?;
        conclude_stream(end, protocol, &event_sender).await
    }

    fn resolved_effort(&self) -> Option<String> {
        self.wire_effort()
    }
}

/// End the stream: finalize pending tool calls and emit the single `MessageEnd`, preferring the
/// recorded `finish_reason` and falling back to tool presence when none arrived.
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
        return Err(crate::provider::sse::stream_error(event_sender, message).await);
    }

    let has_tools =
        finalize_tool_call_accumulators(&mut protocol.tool_call_accumulators, event_sender).await;
    let stop_reason = protocol.final_stop.unwrap_or(if has_tools {
        StopReason::ToolUse
    } else {
        StopReason::EndTurn
    });
    if event_sender
        .send(StreamEvent::MessageEnd { stop_reason })
        .await
        .is_err()
    {
        tracing::trace!("stream event receiver dropped");
    }
    Ok(())
}

/// The Chat Completions driver's state between frames.
#[derive(Default)]
struct ChatCompletionsStream {
    tool_call_accumulators: std::collections::HashMap<i64, ToolCallAccumulator>,
    /// Set when a `finish_reason` chunk arrives. The read keeps going afterward so the trailing
    /// usage chunk (emitted by `stream_options.include_usage`) is captured; finalization and the
    /// single MessageEnd run once the stream ends.
    final_stop: Option<StopReason>,
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
        let Some(data) = crate::provider::sse::frame_json("Chat Completions", &event.data) else {
            return Ok(Step::Continue);
        };
        match handle_stream_chunk(
            &data,
            &mut self.tool_call_accumulators,
            &mut self.final_stop,
            event_sender,
        )
        .await
        {
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

/// Folds one Chat Completions streaming chunk into the in-progress response: forwards usage and
/// text, accumulates tool-call fragments by index, and records the stop reason.
///
/// Extracted from the read loop so the chunk shapes real endpoints emit can be tested without an
/// HTTP server.
async fn handle_stream_chunk(
    data: &serde_json::Value,
    accumulators: &mut std::collections::HashMap<i64, ToolCallAccumulator>,
    final_stop: &mut Option<StopReason>,
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

    if let Some(usage) = data.get("usage") {
        let token_usage = TokenUsage {
            input_tokens: usage
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            output_tokens: usage
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            ..TokenUsage::default()
        };
        if event_sender
            .send(StreamEvent::Usage(token_usage))
            .await
            .is_err()
        {
            tracing::trace!("stream event receiver dropped");
            return ChunkOutcome::Stop;
        }
    }

    let Some(choice) = data.get("choices").and_then(|choices| choices.get(0)) else {
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
        *final_stop = Some(parse_openai_stop_reason(finish_reason));
    }

    let Some(delta) = choice.get("delta") else {
        return ChunkOutcome::Continue;
    };

    if let Some(text) = delta.get("content").and_then(|content| content.as_str())
        && !text.is_empty()
        && event_sender
            .send(StreamEvent::TextDelta(text.to_string()))
            .await
            .is_err()
    {
        tracing::trace!("stream event receiver dropped");
        return ChunkOutcome::Stop;
    }

    let Some(tool_calls) = delta
        .get("tool_calls")
        .and_then(|tool_calls| tool_calls.as_array())
    else {
        return ChunkOutcome::Continue;
    };

    for tool_call in tool_calls {
        let index = tool_call
            .get("index")
            .and_then(|index| index.as_i64())
            .unwrap_or(0);

        let name = tool_call
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(|name| name.as_str())
            .or_else(|| tool_call.get("name").and_then(|name| name.as_str()));

        // Keyed on the index alone. The id is filled in by whichever fragment carries it, so an
        // endpoint that sends the name and the first arguments before the id, or never sends one,
        // still accumulates a call rather than streaming its arguments to the screen and dropping
        // them from the request.
        let accumulator = accumulators
            .entry(index)
            .or_insert_with(|| ToolCallAccumulator {
                id: String::new(),
                name: String::new(),
                arguments: String::new(),
            });
        if let Some(id) = tool_call.get("id").and_then(|id| id.as_str())
            && accumulator.id.is_empty()
        {
            accumulator.id = id.to_string();
        }
        if let Some(name) = name
            && accumulator.name.is_empty()
        {
            accumulator.name = name.to_string();
        }

        if let Some(args) = tool_call
            .get("function")
            .and_then(|function| function.get("arguments"))
            .and_then(|arguments| arguments.as_str())
            .or_else(|| {
                tool_call
                    .get("arguments")
                    .and_then(|arguments| arguments.as_str())
            })
            && !args.is_empty()
        {
            accumulator.arguments.push_str(args);
        }
    }

    ChunkOutcome::Continue
}

fn parse_openai_stop_reason(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::EndTurn,
        "tool_calls" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        other => {
            tracing::warn!(
                "OpenAI returned unrecognized finish_reason {other:?}; mapping to Unknown"
            );
            StopReason::Unknown(other.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::ToolResultContent;

    /// Drives [`handle_stream_chunk`] over a sequence of chunks and returns everything it produced,
    /// mirroring what the read loop in `stream_completion` does with a live SSE stream.
    async fn drive_chunks(
        chunks: &[serde_json::Value],
    ) -> (
        Vec<StreamEvent>,
        std::collections::HashMap<i64, ToolCallAccumulator>,
        Option<StopReason>,
    ) {
        let (sender, mut receiver) = mpsc::channel::<StreamEvent>(64);
        let mut accumulators = std::collections::HashMap::new();
        let mut final_stop = None;

        for chunk in chunks {
            handle_stream_chunk(chunk, &mut accumulators, &mut final_stop, &sender).await;
        }
        drop(sender);

        let mut events = Vec::new();
        while let Some(event) = receiver.recv().await {
            events.push(event);
        }
        (events, accumulators, final_stop)
    }

    /// A gateway that ends the response after the final `finish_reason` chunk and never writes
    /// `[DONE]` has delivered a complete message; failing it dropped every tool call of the turn.
    /// A close with no `finish_reason` seen is still the truncation the check exists for.
    #[tokio::test]
    async fn a_close_after_finish_reason_completes_the_message() {
        let (sender, mut receiver) = mpsc::channel::<StreamEvent>(8);
        let protocol = ChatCompletionsStream {
            final_stop: Some(StopReason::EndTurn),
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
                stop_reason: StopReason::EndTurn
            }]),
            "{events:?}"
        );

        let (sender, _receiver) = mpsc::channel::<StreamEvent>(8);
        let error = conclude_stream(End::Ended, ChatCompletionsStream::default(), &sender)
            .await
            .expect_err("no stop reason and no terminal frame is a truncated stream");
        assert!(matches!(error, MekaError::StreamError(_)), "{error:?}");
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
            matches!(events.as_slice(), [StreamEvent::TextDelta(text)] if text == "hi"),
            "{events:?}"
        );
        assert_eq!(final_stop, Some(StopReason::EndTurn));
    }

    /// vLLM-backed endpoints coalesce the final tool-call delta into the same chunk as
    /// `finish_reason`, rather than sending `finish_reason` alone with an empty delta the way
    /// OpenAI does. The chunk below is a real capture. Skipping the delta on a chunk that carries a
    /// `finish_reason` drops the tool call outright, leaving `StopReason::ToolUse` with no tool-use
    /// block, which the agent surfaces as "the model returned an empty response".
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

        assert_eq!(final_stop, Some(StopReason::ToolUse));
        let accumulator = accumulators
            .get(&0)
            .expect("tool call in a finish_reason chunk must still be accumulated");
        assert_eq!(accumulator.id, "chatcmpl-tool-a445012e51c3a83d");
        assert_eq!(accumulator.name, "mcp__exa__web_search_exa");
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

        assert_eq!(final_stop, Some(StopReason::EndTurn));
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
                            "function": {"name": "execute_command", "arguments": "{\"command\":"}
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

        assert_eq!(final_stop, Some(StopReason::ToolUse));
        let accumulator = accumulators.get(&0).expect("accumulated tool call");
        assert_eq!(accumulator.id, "call_abc");
        assert_eq!(accumulator.name, "execute_command");
        assert_eq!(accumulator.arguments, "{\"command\": \"pwd\"}");
    }

    #[test]
    fn an_openai_base_url_is_normalized_at_construction() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::new(
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
            OpenAiChatCompletionsProvider::new(
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
            OpenAiChatCompletionsProvider::new(
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
            OpenAiChatCompletionsProvider::new(
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
            OpenAiChatCompletionsProvider::new(
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
            OpenAiChatCompletionsProvider::new(
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
        // meka's own ceiling, so `RequestTooLarge` rather than the provider's `InvalidRequest`;
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
            OpenAiChatCompletionsProvider::new(
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
            OpenAiChatCompletionsProvider::new(
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
            OpenAiChatCompletionsProvider::new(
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
            OpenAiChatCompletionsProvider::new(
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
        // default and meka asks for it by omitting the field.
        for model in ["gpt-5.6-sol", "o3", "llama3.1"] {
            let provider = {
                let api_key: String = "test-key".to_string();
                OpenAiChatCompletionsProvider::new(
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
        // A configured value is absolute, including on a model meka does not recognize.
        let configured = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::new(
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
            OpenAiChatCompletionsProvider::new(
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
            "read_file".to_string(),
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
        assert_eq!(openai_tools[0]["function"]["name"], "read_file");
    }

    #[test]
    fn openai_request_body_with_tool_calls() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::new(
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
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test.txt"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "file contents here".to_string(),
                    }],
                    is_error: false,
                }],
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

    #[test]
    fn openai_parse_non_streaming_text() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::new(
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

        let (message, stop_reason, _) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");

        assert_eq!(message.text_content(), "Hello there!");
        assert_eq!(stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn openai_parse_non_streaming_tool_call() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::new(
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
                            "name": "read_file",
                            "arguments": "{\"path\":\"/tmp/test.txt\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let (message, stop_reason, _) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse");

        assert_eq!(stop_reason, StopReason::ToolUse);
        let tool_uses = message.tool_uses();
        assert_eq!(tool_uses.len(), 1);

        if let ContentBlock::ToolUse { id, name, input } = &tool_uses[0] {
            assert_eq!(id, "call_abc");
            assert_eq!(name, "read_file");
            assert_eq!(input["path"], "/tmp/test.txt");
        } else {
            panic!("expected ToolUse block");
        }
    }

    #[test]
    fn openai_parse_non_streaming_malformed_tool_args() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::new(
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
                            "name": "read_file",
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
        if let ContentBlock::ToolUse { input, .. } = &tool_uses[0] {
            assert!(
                input
                    .get(crate::provider::INVALID_TOOL_ARGS_MARKER)
                    .and_then(|reason| reason.as_str())
                    .is_some(),
                "malformed args must surface the invalid-args sentinel, got: {input}"
            );
        } else {
            panic!("expected ToolUse block");
        }
    }

    #[test]
    fn openai_parse_missing_message_in_choice() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::new(
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

    #[test]
    fn openai_parse_missing_tool_call_id() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::new(
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
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let result = provider.parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn openai_parse_missing_tool_call_function_name() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::new(
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
                            "arguments": "{}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let result = provider.parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn openai_parse_non_streaming_flattened_tool_call() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::new(
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
                        "name": "read_file",
                        "arguments": "{\"path\":\"/tmp/test.txt\"}"
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let (message, stop_reason, _) = provider
            .parse_non_streaming_response(&response)
            .expect("should parse flattened tool call");

        assert_eq!(stop_reason, StopReason::ToolUse);
        let tool_uses = message.tool_uses();
        assert_eq!(tool_uses.len(), 1);

        if let ContentBlock::ToolUse { id, name, input } = &tool_uses[0] {
            assert_eq!(id, "call_abc");
            assert_eq!(name, "read_file");
            assert_eq!(input["path"], "/tmp/test.txt");
        } else {
            panic!("expected ToolUse block");
        }
    }

    #[test]
    fn openai_parse_non_streaming_flattened_missing_name_still_errors() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::new(
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
                        "arguments": "{}"
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let result = provider.parse_non_streaming_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn openai_tool_definitions_use_standard_chat_completions_format() {
        let provider = {
            let api_key: String = "test-key".to_string();
            OpenAiChatCompletionsProvider::new(
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
            "write_file".to_string(),
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
        assert_eq!(openai_tools[0]["function"]["name"], "write_file");
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
        let mut accumulators = std::collections::HashMap::new();
        let mut final_stop = None;
        let outcome = handle_stream_chunk(
            &serde_json::json!({"error": {"message": "upstream capacity", "type": "server_error"}}),
            &mut accumulators,
            &mut final_stop,
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
                "index": 0, "function": {"name": "read_file", "arguments": "{\"path\":"}
            }]}}]}),
            serde_json::json!({"choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "id": "call_1", "function": {"arguments": " \"a.txt\"}"}
            }]}}]}),
        ])
        .await;
        let accumulator = accumulators.get(&0).expect("one call accumulated");
        assert_eq!(accumulator.id, "call_1");
        assert_eq!(accumulator.name, "read_file");
        assert_eq!(accumulator.arguments, "{\"path\": \"a.txt\"}");
    }

    /// The `tool` message has no `is_error` field in this API; a strict endpoint rejects the
    /// request for one, and only on the turns where a tool failed.
    #[test]
    fn a_failed_tool_result_carries_no_is_error_key() {
        let provider = {
            let api_key: String = "key".to_string();
            OpenAiChatCompletionsProvider::new(
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
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "x"}),
                }],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "Error: no such file".to_string(),
                    }],
                    is_error: true,
                }],
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
