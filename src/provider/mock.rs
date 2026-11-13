//! Scripted [`Provider`] for tests. Replays a queue of per-round `StreamEvent` lists so an
//! integration test can drive a multi-round `Agent::run_turn` (tool-use round → tool-result round →
//! final text round) without touching the network.
//!
//! Activated when `MEKA_MOCK_PROVIDER` is set to `1`, which every surface honors: `meka acp`,
//! `meka serve`, and the CLI entry point behind the screen and `--oneshot`. That last one is what
//! lets a test ask what two `meka` processes do to each other's sessions. A second variable,
//! `MEKA_MOCK_PROVIDER_SCRIPT`, names the file holding the JSON-encoded script (see
//! [`crate::provider::mock::load_script`]). Both are read by `ResolvedConfig::resolve`, like
//! every other `MEKA_*` variable, and acted on by `host::build_shared_deps`. Nothing here is
//! reachable without them, and the whole module is compiled out of a release build.
//!
//! The mock is intentionally minimal: text deltas, thinking deltas, tool-use lifecycle,
//! `MessageEnd`, plus a synthetic `Fail` event that returns an error from [`Provider::stream`] so
//! the agent's non-Interrupted error path can be exercised end-to-end. Image content is not
//! supported; tests that need it should extend the mock first.

use std::{collections::VecDeque, sync::Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    conversation::{ContentBlock, Message, Role},
    error::Result,
    provider::{CompletionRequest, Provider, StopReason, StreamEvent, ToolDefinition},
    stats::TokenUsage,
};

/// Serialized event used by [`MockProvider`]. Mirrors the runtime [`StreamEvent`] enum but uses
/// owned struct-tagged variants so scripts can be loaded from JSON (`serde`'s internally-tagged
/// enums don't accept tuple/newtype variants). `Sleep` and `Stall` are the two non-stream-event
/// variants; both delay the mock so a test can act mid-turn, and they differ in whether
/// cancellation cuts them short.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum MockEvent {
    Text {
        text: String,
    },
    /// Words in the field a provider keeps for declining (OpenAI's `refusal`).
    Refusal {
        text: String,
    },
    /// A part Meka reads nothing from, kept as the provider sent it.
    Unrecognized {
        value: serde_json::Value,
    },
    ThinkingDelta {
        text: String,
    },
    /// A running estimate of the block's size, as Claude's token-count beta reports it.
    ///
    /// Scriptable because the beta emits one of these from the *same* wire event that carries a
    /// visible delta, so the two alternate for the whole block. Nothing else can produce that
    /// interleaving, which is the shape the screen's thinking row counts under.
    ThinkingProgress {
        /// `None` is the count the backend reports before the beta has one, and is the shape a
        /// script wants for "the block is running but nothing can be said about its size yet".
        #[serde(default)]
        estimated_tokens: Option<u64>,
    },
    /// Caps an in-flight thinking block. `opaque` carries whichever wire shape the script is
    /// standing in for: a Claude signature, or a Responses reasoning item's sealed content. The
    /// agent treats it as pass-through (see [`crate::frontend::FrontendEvent::ThinkingBlock`]), so
    /// a script sets it to check that the turn it records can be replayed. Defaulted so existing
    /// scripts keep loading.
    ThinkingComplete {
        #[serde(default)]
        opaque: Option<crate::conversation::OpaqueReasoning>,
    },
    /// A `redacted_thinking` block, whole, as a Claude stream opens one.
    RedactedThinking {
        data: String,
    },
    ToolUseStart {
        id: String,
        name: String,
    },
    /// The end of the call the last `ToolUseStart` at this place began.
    ToolUseEnd {
        input: serde_json::Value,
    },
    /// The events that follow are of the part at `place`, as a Claude stream's `index` places a
    /// block that opens before an earlier one stops.
    Place {
        place: u64,
    },
    MessageEnd {
        stop_reason: MockStopReason,
    },
    Sleep {
        ms: u64,
    },
    /// A delay cancellation does *not* cut short, standing in for the part of a turn a drain has
    /// to wait out rather than interrupt: a tool call already inside a syscall, an MCP round-trip
    /// under its own timeout, the conversation commit at the end. Cancellation is checked between
    /// events, so a real turn's response to it is never instant either.
    ///
    /// [`Self::Sleep`] is the opposite and stays the default choice: a test asserting that cancel
    /// *works* wants the sleep to end the moment the token fires. This one exists for the tests
    /// asserting what happens to a turn that is still unwinding.
    Stall {
        ms: u64,
    },
    /// Synthetic provider failure. The stream returns `Err(MekaError::Provider(message))`
    /// immediately, exercising the non-Interrupted error arm of `Agent::run_turn` (which the ACP
    /// layer maps to a JSON-RPC `internal_error`).
    Fail {
        message: String,
    },
    /// Fails the round the way [`Self::Fail`] does unless the request's system prompt contains
    /// `contains`. A script's answers never depend on what it was sent, so this is how a test
    /// driving a real host asserts on the prompt that host built: the recording below is readable
    /// in-process only.
    ExpectSystemPrompt {
        contains: String,
    },
    /// Synthetic *transient* provider failure. The stream returns
    /// `Err(MekaError::RetryableProvider { .. })` immediately, exercising `Agent::run_streaming`'s
    /// retry-with-backoff path. Each retry consumes one more round from the script, so a
    /// `[FailRetryable, ..success events..]` script simulates "first attempt overloaded, retry
    /// succeeds", the same shape a real transient 429/529 followed by success takes.
    FailRetryable {
        message: String,
        retry_after_secs: Option<u64>,
    },
    /// Synthetic quota or billing rejection. The stream returns
    /// `Err(MekaError::ProviderExhausted { .. })` at once, which is never retried in place, so a
    /// profile with fallbacks hands the turn to the next one and the next round is theirs.
    FailExhausted {
        message: String,
        resets_after_secs: Option<u64>,
    },
    /// A login that must be redone. The stream returns
    /// `Err(MekaError::AccountLoginSpent { .. })` at once, as a subscription driver does when its
    /// token is rejected after a refresh or its refresh is refused.
    FailLoginSpent {
        account: String,
        message: String,
    },
    /// Synthetic *malformed-request* failure. The stream returns
    /// `Err(MekaError::InvalidRequest(message))` immediately, exercising `Agent::run_turn`'s
    /// degrade-and-retry path. Each attempt consumes one round, so
    /// `[FailInvalidRequest, ..success events..]` simulates "the provider rejected the content Meka
    /// just appended, the retry without it succeeds".
    FailInvalidRequest {
        message: String,
    },
    /// A provider-side advisory forwarded to the frontend mid-stream. Exists so a test can put a
    /// notice *before* a failure and assert the retry still fires: notices are the one event the
    /// agent forwards without marking the turn as having produced output.
    Notice {
        message: String,
    },
    /// An image-redaction advisory carrying what it removed, which the agent counts against the
    /// session, and where it sat, which the agent records on the conversation.
    Redaction {
        images: u64,
        bytes: u64,
        #[serde(default)]
        positions: Vec<crate::image::RedactedImage>,
    },
    /// Synthetic *transport* failure: the stream returns `Err(MekaError::StreamError(message))`,
    /// which the agent retries only while nothing user-visible has been emitted. Each attempt
    /// consumes one round.
    FailStream {
        message: String,
    },
    /// [`Self::FailStream`] from a driver that held part of the reply when its stream broke: the
    /// failure is said, then `held` is handed over, as a driver delivers what it assembled (the
    /// calls a Chat Completions stream builds to its end, the blocks a Claude stream left open).
    FailStreamHolding {
        message: String,
        held: Vec<MockEvent>,
    },
    /// Synthetic *context-window overflow*. The stream returns `Err(MekaError::ContextOverflow)`,
    /// which is the one recovery path in `Agent::run_turn` nothing else can drive: the emergency
    /// compact-and-retry only fires on this error. Each
    /// attempt consumes one round, so `[FailContextOverflow, ..success events..]` is "the request
    /// was too large, the compacted retry fit".
    FailContextOverflow {
        message: String,
    },
    /// The input tokens the round reports, as a provider's `usage` does. What the agent gauges the
    /// context by, so a script can put the occupancy where a test needs it.
    Usage {
        input_tokens: u64,
    },
    /// Holds the reply until `gate` fires, and like [`Self::Sleep`] ends at once on a stop. The
    /// pacing a test keys on an event of its own rather than on the clock: a test that has to act
    /// before a round ends releases the gate once it has, where a fixed delay only hopes to be
    /// long enough on a slow machine. Never scripted, so never written or read.
    #[serde(skip)]
    #[allow(
        dead_code,
        reason = "constructed by tests alone, which pace a round on an event of their own"
    )]
    Hold {
        gate: tokio_util::sync::CancellationToken,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MockStopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    /// Model-side refusal, as Claude's classifier states one: the stop alone, whatever text
    /// streamed before it.
    Refusal,
}

impl From<MockStopReason> for StopReason {
    fn from(reason: MockStopReason) -> Self {
        match reason {
            MockStopReason::EndTurn => StopReason::EndTurn,
            MockStopReason::ToolUse => StopReason::ToolUse,
            MockStopReason::MaxTokens => StopReason::MaxTokens,
            MockStopReason::Refusal => StopReason::Refusal,
        }
    }
}

/// A scripted multi-round response. Each call to [`Provider::stream`] *or* [`Provider::complete`]
/// drains one round (`Vec<MockEvent>`); subsequent rounds satisfy subsequent agent loop iterations
/// after tool results return. The two paths share the one queue, so a script that spawns a
/// sub-agent must budget a round for each of the sub-agent's turns.
#[derive(Debug, Default)]
pub(crate) struct MockProvider {
    rounds: Mutex<VecDeque<Vec<MockEvent>>>,
    /// Messages handed to each [`Provider::complete`] call, in order.
    ///
    /// Recorded because some behavior is only observable in the *request*: whether the checkpoint
    /// turn is handed the whole conversation, or emits two consecutive user turns. A test that
    /// rebuilds the expected list itself asserts on its own arithmetic and passes even when the
    /// production path is reverted, which is worse than no test at all.
    completions: Mutex<Vec<Vec<Message>>>,
    /// The prompt each `complete` call was attributed to, in call order: a sub-agent under
    /// `--no-stream` runs this way, and whether it carried its parent's prompt id is visible
    /// nowhere else.
    completion_prompt_ids: Mutex<Vec<Option<uuid::Uuid>>>,
    /// Where each `complete` call said its prompt came from, in call order, for the same reason.
    completion_turn_origins: Mutex<Vec<Option<crate::provider::TurnOrigin>>>,
    /// Where each `complete` call said its turn stood, in call order, for the same reason.
    completion_turn_positions: Mutex<Vec<Option<crate::stats::TurnPosition>>>,
    /// How each `complete` call named its worker, in call order, for the same reason.
    completion_workers: Mutex<Vec<Option<crate::provider::WorkerIdentity>>>,
    /// The tools each `complete` call said the session had called, in call order, for the same
    /// reason.
    completion_tools_used: Mutex<Vec<Vec<crate::stats::ToolUsed>>>,
    /// The tool names each `complete` call carried, in call order: whether a checkpoint's request
    /// carries the turn's catalog is visible nowhere else.
    completion_tools: Mutex<Vec<Vec<String>>>,
    /// What each [`Provider::stream`] call was handed, in order.
    ///
    /// The streaming counterpart to [`Self::completions`], and added for the same reason plus one
    /// more: some behavior is only observable in the request and only on the streaming path.
    /// Whether every round of a turn carries the whole conversation, for instance, cannot be seen
    /// in the response at all, so a test that did not record this had nothing to assert against
    /// and passed with the production path reverted.
    streams: Mutex<Vec<StreamRequest>>,
    /// How this mock says its wire carries deferred tools; unset reads as appended, like a wire
    /// with no declaration of its own.
    delivery: Mutex<Option<crate::provider::DeferredToolDelivery>>,
    /// Tool names this mock's wire rejects; empty, like a wire that rejects none.
    rejected_names: Mutex<Vec<String>>,
    /// The model this mock's requests name; unset, like a provider that names none.
    model: Option<String>,
}

/// One recorded [`Provider::stream`] call. Owned rather than borrowed because a test inspects it
/// after the turn has finished and the caller's slices are long gone.
///
/// Only ever read from `#[cfg(test)]` code, but recorded unconditionally: the recording lives in
/// `stream`, which is one function compiled for both builds, and splitting it would put a `cfg` in
/// the middle of the mock's hot path to save three fields in a binary that already carries the
/// whole mock.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone)]
pub(crate) struct StreamRequest {
    pub(crate) system_prompt: String,
    pub(crate) messages: Vec<Message>,
    pub(crate) tools: Vec<ToolDefinition>,
    /// The prompt this request was attributed to, read the way a real provider reads it while
    /// building its billing header.
    ///
    /// Recorded for the same reason the fields above are, with the sharpest version of it: the
    /// streaming path hands `stream` to `tokio::spawn`, task-locals do not cross a spawn, and
    /// nothing in the response reveals whether the attribution made it. Without this the wrapper
    /// that carries it over can be deleted and every test still passes.
    pub(crate) prompt_id: Option<uuid::Uuid>,
    /// Where that prompt came from, recorded for the same reason.
    pub(crate) turn_origin: Option<crate::provider::TurnOrigin>,
    /// Where that prompt's turn stood, recorded for the same reason.
    pub(crate) turn_position: Option<crate::stats::TurnPosition>,
    /// The session the request named, for the same reason: a ChatGPT request sends it as its
    /// cache affinity, and a turn that reads it before the session exists sends none.
    pub(crate) session_id: Option<uuid::Uuid>,
    /// The response before a compaction the request named, for the same reason.
    pub(crate) response_before_compaction: Option<crate::provider::MainResponse>,
    /// The tools the session had called when the request was built, for the same reason.
    pub(crate) tools_used: Vec<crate::stats::ToolUsed>,
}

impl MockProvider {
    /// What each `stream` call was handed so far, in order.
    #[cfg(test)]
    pub(crate) fn streams(&self) -> Vec<StreamRequest> {
        crate::sync::lock(&self.streams).clone()
    }

    /// The messages behind each `complete` call so far, in order.
    #[cfg(test)]
    pub(crate) fn completions(&self) -> Vec<Vec<Message>> {
        crate::sync::lock(&self.completions).clone()
    }

    /// The tool names behind each `complete` call so far, in order.
    #[cfg(test)]
    pub(crate) fn completion_tools(&self) -> Vec<Vec<String>> {
        crate::sync::lock(&self.completion_tools).clone()
    }

    /// The prompt id behind each `complete` call so far, in order.
    #[cfg(test)]
    pub(crate) fn completion_prompt_ids(&self) -> Vec<Option<uuid::Uuid>> {
        crate::sync::lock(&self.completion_prompt_ids).clone()
    }

    /// The turn origin behind each `complete` call so far, in order.
    #[cfg(test)]
    pub(crate) fn completion_turn_origins(&self) -> Vec<Option<crate::provider::TurnOrigin>> {
        crate::sync::lock(&self.completion_turn_origins).clone()
    }

    /// The turn position behind each `complete` call so far, in order.
    #[cfg(test)]
    pub(crate) fn completion_turn_positions(&self) -> Vec<Option<crate::stats::TurnPosition>> {
        crate::sync::lock(&self.completion_turn_positions).clone()
    }

    /// The worker identity behind each `complete` call so far, in order.
    #[cfg(test)]
    pub(crate) fn completion_workers(&self) -> Vec<Option<crate::provider::WorkerIdentity>> {
        crate::sync::lock(&self.completion_workers).clone()
    }

    /// The tools called behind each `complete` call so far, in order.
    #[cfg(test)]
    pub(crate) fn completion_tools_used(&self) -> Vec<Vec<crate::stats::ToolUsed>> {
        crate::sync::lock(&self.completion_tools_used).clone()
    }

    /// A provider that replays `rounds`, one per call, and answers nothing once they are spent.
    pub(crate) fn from_rounds(rounds: Vec<Vec<MockEvent>>) -> Self {
        Self {
            rounds: Mutex::new(rounds.into()),
            ..Default::default()
        }
    }

    /// The same mock answering [`Provider::deferred_tool_delivery`] with `delivery`.
    #[cfg(test)]
    pub(crate) fn delivering(self, delivery: crate::provider::DeferredToolDelivery) -> Self {
        *crate::sync::lock(&self.delivery) = Some(delivery);
        self
    }

    /// The same mock rejecting the tool `name` by its name.
    #[cfg(test)]
    pub(crate) fn rejecting(self, name: &str) -> Self {
        crate::sync::lock(&self.rejected_names).push(name.to_string());
        self
    }

    /// The same mock naming `model` in its requests.
    #[cfg(test)]
    pub(crate) fn naming_model(self, model: &str) -> Self {
        Self {
            model: Some(model.to_string()),
            ..self
        }
    }
}

#[async_trait]
impl Provider for MockProvider {
    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn deferred_tool_delivery(&self) -> crate::provider::DeferredToolDelivery {
        crate::sync::lock(&self.delivery).unwrap_or(crate::provider::DeferredToolDelivery::Appended)
    }

    fn rejects_tool_name(&self, name: &str) -> bool {
        crate::sync::lock(&self.rejected_names)
            .iter()
            .any(|rejected| rejected == name)
    }

    /// Drains one round and folds it into a finished message, so a single script drives either
    /// path. Non-streaming is not an exotic corner: `--no-stream` runs this way, sub-agents
    /// included, as does auto-compaction's summarizer.
    async fn complete(
        &self,
        request: CompletionRequest<'_>,
        cancellation: CancellationToken,
    ) -> Result<crate::provider::Completion> {
        let CompletionRequest {
            system_prompt,
            messages,
            tools,
            attribution,
            ..
        } = request;
        crate::sync::lock(&self.completion_tools)
            .push(tools.iter().map(|tool| tool.name.clone()).collect());
        crate::sync::lock(&self.completions).push(messages.to_vec());
        crate::sync::lock(&self.completion_prompt_ids).push(attribution.prompt_id);
        crate::sync::lock(&self.completion_turn_origins).push(attribution.turn_origin);
        crate::sync::lock(&self.completion_turn_positions).push(attribution.turn_position);
        crate::sync::lock(&self.completion_workers).push(attribution.worker.clone());
        crate::sync::lock(&self.completion_tools_used).push(attribution.tools_used.clone());

        let events = {
            let mut rounds = crate::sync::lock(&self.rounds);
            rounds.pop_front().unwrap_or_default()
        };

        let mut content: Vec<ContentBlock> = Vec::new();
        let mut text = String::new();
        let mut refusal = String::new();
        let mut thinking = String::new();
        let mut pending_tool: Option<(String, String)> = None;
        let mut stop_reason = StopReason::EndTurn;
        let mut notices = Vec::new();
        let mut usage = TokenUsage::default();

        for event in events {
            match event {
                MockEvent::Fail { message } => {
                    return Err(crate::error::MekaError::Provider(message));
                }
                MockEvent::ExpectSystemPrompt { contains } => {
                    if !system_prompt.contains(&contains) {
                        return Err(crate::error::MekaError::Provider(system_prompt_complaint(
                            &contains,
                        )));
                    }
                }
                MockEvent::FailStream { message }
                | MockEvent::FailStreamHolding { message, .. } => {
                    return Err(crate::error::MekaError::StreamError(message));
                }
                // No channel on the non-streaming path: advisories ride on the return, as the real
                // providers' do, and the agent forwards them from there.
                MockEvent::Notice { message } => {
                    notices.push(crate::frontend::Notice::info(message))
                }
                MockEvent::Redaction {
                    images,
                    bytes,
                    positions,
                } => {
                    notices.push(redaction_notice(images, bytes, positions));
                }
                MockEvent::FailContextOverflow { message } => {
                    return Err(crate::error::MekaError::ContextOverflow(message));
                }
                MockEvent::FailRetryable {
                    message,
                    retry_after_secs,
                } => {
                    return Err(crate::error::MekaError::RetryableProvider {
                        message,
                        retry_after: retry_after_secs.map(std::time::Duration::from_secs),
                        // Models the shape the classifier gives a 5xx answering a completion,
                        // which is the one the degrade-and-retry keys on. A test wanting the
                        // other shape drives `error::provider_transport_error` directly.
                        server_error_on_completion: true,
                    });
                }
                MockEvent::FailExhausted {
                    message,
                    resets_after_secs,
                } => {
                    return Err(crate::error::MekaError::ProviderExhausted {
                        message,
                        resets_after: resets_after_secs.map(std::time::Duration::from_secs),
                    });
                }
                MockEvent::FailLoginSpent { account, message } => {
                    return Err(crate::error::MekaError::AccountLoginSpent { account, message });
                }
                MockEvent::FailInvalidRequest { message } => {
                    return Err(crate::error::MekaError::InvalidRequest(message));
                }
                MockEvent::Sleep { ms } => {
                    // Raced like the real send is: a stop drops a whole reply still being
                    // generated. `Stall` is the reply that had already left when the stop came.
                    tokio::select! {
                        _ = tokio::time::sleep(std::time::Duration::from_millis(ms)) => {}
                        _ = cancellation.cancelled() => {
                            return Err(crate::error::MekaError::Interrupted);
                        }
                    }
                }
                MockEvent::Stall { ms } => {
                    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                }
                MockEvent::Hold { gate } => {
                    tokio::select! {
                        _ = gate.cancelled() => {}
                        _ = cancellation.cancelled() => {
                            return Err(crate::error::MekaError::Interrupted);
                        }
                    }
                }
                MockEvent::Text { text: chunk } => text.push_str(&chunk),
                MockEvent::Refusal { text: chunk } => refusal.push_str(&chunk),
                MockEvent::Unrecognized { value } => {
                    content.push(ContentBlock::Unrecognized { parts: vec![value] })
                }
                MockEvent::ThinkingDelta { text: chunk } => thinking.push_str(&chunk),
                // A live indicator's running estimate. Nothing survives a non-streaming turn, which
                // has no indicator to feed.
                MockEvent::ThinkingProgress { .. } => {}
                // Folded into a block like the real providers return it. `run_turn`'s non-streaming
                // branch reads these back out to reach the frontend, so a script that scripts
                // thinking has to produce one here or that path has nothing to render.
                MockEvent::ThinkingComplete { opaque } => {
                    // Flush first, for the same reason `ToolUseStart` does: a block's position
                    // relative to the text around it is what a replayed turn preserves.
                    if !text.is_empty() {
                        content.push(ContentBlock::Text {
                            text: std::mem::take(&mut text),
                            provider_fields: None,
                        });
                    }
                    if let Some(block) = ContentBlock::replayable_thinking(
                        vec![crate::conversation::ReasoningPart::reasoning(
                            std::mem::take(&mut thinking),
                        )],
                        opaque,
                        None,
                    ) {
                        content.push(block);
                    }
                }
                MockEvent::RedactedThinking { data } => {
                    if !text.is_empty() {
                        content.push(ContentBlock::Text {
                            text: std::mem::take(&mut text),
                            provider_fields: None,
                        });
                    }
                    content.push(ContentBlock::RedactedThinking {
                        data,
                        provider_fields: None,
                    });
                }
                // The whole-reply path lays parts out in the order the script gives them.
                MockEvent::Place { .. } => {}
                MockEvent::ToolUseStart { id, name } => {
                    // Flush first, so text that preceded this call stays ahead of it and text
                    // between two calls stays between them. Accumulating everything and appending
                    // at the end would reorder blocks relative to a real provider.
                    if !text.is_empty() {
                        content.push(ContentBlock::Text {
                            text: std::mem::take(&mut text),
                            provider_fields: None,
                        });
                    }
                    pending_tool = Some((id, name));
                }
                MockEvent::ToolUseEnd { input } => {
                    if let Some((id, name)) = pending_tool.take() {
                        content.push(ContentBlock::ToolUse {
                            id,
                            name,
                            input,
                            unparsed: None,
                            provider_fields: None,
                        });
                    }
                }
                MockEvent::MessageEnd {
                    stop_reason: reason,
                } => stop_reason = reason.into(),
                MockEvent::Usage { input_tokens } => usage.input_tokens = input_tokens,
            }
        }

        if !text.is_empty() {
            content.push(ContentBlock::Text {
                text,
                provider_fields: None,
            });
        }
        if !refusal.is_empty() {
            content.push(ContentBlock::Refusal {
                text: refusal,
                provider_fields: None,
            });
        }

        // Classed as every whole-reply parser classes it, a refusal block making a refusal.
        let stop = crate::provider::Stop::of(stop_reason).with_refusal_of(&content);
        Ok(crate::provider::Completion {
            message: Message {
                role: Role::Assistant,
                content,
                provider_fields: None,
            },
            stop,
            model: None,
            usage,
            notices,
            response_fields: None,
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
            attribution,
            ..
        } = request;
        crate::sync::lock(&self.streams).push(StreamRequest {
            system_prompt: system_prompt.to_string(),
            messages: messages.to_vec(),
            tools: tools.to_vec(),
            prompt_id: attribution.prompt_id,
            turn_origin: attribution.turn_origin,
            turn_position: attribution.turn_position,
            session_id: attribution.session_id,
            response_before_compaction: attribution.response_before_compaction(),
            tools_used: attribution.tools_used.clone(),
        });

        let events = {
            let mut rounds = crate::sync::lock(&self.rounds);
            rounds.pop_front().unwrap_or_default()
        };
        let mut script = ScriptedStream::default();
        for event in events {
            // A stop that has already landed is answered the way `sse::drive` answers one, as the
            // interrupt it is: `Ok(())` with nothing streamed reads to the agent as a reply the
            // provider accepted and ended, since a real driver never returns it without a 2xx, and
            // a repair riding the request would be booked as vindicated.
            if cancellation.is_cancelled() {
                return Err(crate::error::MekaError::Interrupted);
            }
            match event {
                // Each failure sends `StreamEvent::Error` before returning its typed error, the
                // way the real SSE drivers do. `Agent`'s handler documents that as an invariant of
                // every producer and acts on it, so a mock that skipped it would exercise a path
                // production never takes.
                MockEvent::Fail { message } => {
                    send_stream_error(&event_sender, &message).await;
                    return Err(crate::error::MekaError::Provider(message));
                }
                MockEvent::ExpectSystemPrompt { contains } => {
                    if !system_prompt.contains(&contains) {
                        let message = system_prompt_complaint(&contains);
                        send_stream_error(&event_sender, &message).await;
                        return Err(crate::error::MekaError::Provider(message));
                    }
                }
                MockEvent::FailStream { message } => {
                    send_stream_error(&event_sender, &message).await;
                    return Err(crate::error::MekaError::StreamError(message));
                }
                MockEvent::FailStreamHolding { message, held } => {
                    send_stream_error(&event_sender, &message).await;
                    for event in held
                        .into_iter()
                        .filter_map(|event| script.stream_event_of(event))
                    {
                        if event_sender.send(event).await.is_err() {
                            break;
                        }
                    }
                    return Err(crate::error::MekaError::StreamError(message));
                }
                MockEvent::Notice { message } => {
                    if event_sender
                        .send(StreamEvent::Notice(crate::frontend::Notice::info(message)))
                        .await
                        .is_err()
                    {
                        return Ok(());
                    }
                }
                MockEvent::Redaction {
                    images,
                    bytes,
                    positions,
                } => {
                    if event_sender
                        .send(StreamEvent::Notice(redaction_notice(
                            images, bytes, positions,
                        )))
                        .await
                        .is_err()
                    {
                        return Ok(());
                    }
                }
                MockEvent::FailContextOverflow { message } => {
                    send_stream_error(&event_sender, &message).await;
                    return Err(crate::error::MekaError::ContextOverflow(message));
                }
                MockEvent::FailRetryable {
                    message,
                    retry_after_secs,
                } => {
                    send_stream_error(&event_sender, &message).await;
                    return Err(crate::error::MekaError::RetryableProvider {
                        message,
                        retry_after: retry_after_secs.map(std::time::Duration::from_secs),
                        // Models the shape the classifier gives a 5xx answering a completion,
                        // which is the one the degrade-and-retry keys on. A test wanting the
                        // other shape drives `error::provider_transport_error` directly.
                        server_error_on_completion: true,
                    });
                }
                MockEvent::FailExhausted {
                    message,
                    resets_after_secs,
                } => {
                    send_stream_error(&event_sender, &message).await;
                    return Err(crate::error::MekaError::ProviderExhausted {
                        message,
                        resets_after: resets_after_secs.map(std::time::Duration::from_secs),
                    });
                }
                MockEvent::FailLoginSpent { account, message } => {
                    send_stream_error(&event_sender, &message).await;
                    return Err(crate::error::MekaError::AccountLoginSpent { account, message });
                }
                MockEvent::FailInvalidRequest { message } => {
                    send_stream_error(&event_sender, &message).await;
                    return Err(crate::error::MekaError::InvalidRequest(message));
                }
                MockEvent::Sleep { ms } => {
                    // Raced against cancellation, and answered the way `sse::drive` answers it,
                    // so a mid-turn `session/cancel` neither waits out the delay nor looks like a
                    // stream that ended on its own.
                    tokio::select! {
                        _ = tokio::time::sleep(std::time::Duration::from_millis(ms)) => {}
                        _ = cancellation.cancelled() => {
                            return Err(crate::error::MekaError::Interrupted);
                        }
                    }
                    continue;
                }
                MockEvent::Stall { ms } => {
                    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                    continue;
                }
                MockEvent::Hold { gate } => {
                    tokio::select! {
                        _ = gate.cancelled() => {}
                        _ = cancellation.cancelled() => {
                            return Err(crate::error::MekaError::Interrupted);
                        }
                    }
                    continue;
                }
                event => {
                    let Some(stream_event) = script.stream_event_of(event) else {
                        continue;
                    };
                    if event_sender.send(stream_event).await.is_err() {
                        // Receiver dropped; the test ended early. Not an error.
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }
}

/// What a scripted stream has said so far that a later step depends on.
#[derive(Default)]
struct ScriptedStream {
    /// The place the steps arriving now are of; see [`MockEvent::Place`].
    place: u64,
    /// The id and name of the call in progress at each place, which its end repeats.
    calls: std::collections::HashMap<u64, (String, String)>,
}

impl ScriptedStream {
    /// The event a scripted step streams as, or `None` for a step that only steers the script (a
    /// sleep, a hold, a failure), which the stream loop acts on itself.
    fn stream_event_of(&mut self, event: MockEvent) -> Option<StreamEvent> {
        Some(match event {
            MockEvent::Notice { message } => {
                StreamEvent::Notice(crate::frontend::Notice::info(message))
            }
            MockEvent::Redaction {
                images,
                bytes,
                positions,
            } => StreamEvent::Notice(redaction_notice(images, bytes, positions)),
            MockEvent::Text { text } => StreamEvent::TextDelta(text),
            MockEvent::Refusal { text } => StreamEvent::RefusalDelta(text),
            MockEvent::Unrecognized { value } => StreamEvent::Unrecognized(value),
            MockEvent::ThinkingDelta { text } => StreamEvent::ThinkingDelta(text),
            MockEvent::ThinkingProgress { estimated_tokens } => {
                StreamEvent::ThinkingProgress { estimated_tokens }
            }
            MockEvent::ThinkingComplete { opaque } => StreamEvent::ThinkingComplete {
                opaque,
                provider_fields: None,
            },
            MockEvent::RedactedThinking { data } => StreamEvent::RedactedThinking {
                data,
                provider_fields: None,
            },
            MockEvent::ToolUseStart { id, name } => {
                self.calls.insert(self.place, (id.clone(), name.clone()));
                StreamEvent::ToolUseStart { id, name }
            }
            MockEvent::ToolUseEnd { input } => {
                let (id, name) = self.calls.remove(&self.place).unwrap_or_default();
                StreamEvent::ToolUseEnd {
                    id,
                    name,
                    input,
                    provider_fields: None,
                }
            }
            MockEvent::Place { place } => {
                self.place = place;
                StreamEvent::Place(place)
            }
            MockEvent::MessageEnd { stop_reason } => StreamEvent::MessageEnd {
                stop: crate::provider::Stop::of(stop_reason.into()),
            },
            MockEvent::Usage { input_tokens } => StreamEvent::Usage(TokenUsage {
                input_tokens,
                ..TokenUsage::default()
            }),
            MockEvent::Sleep { .. }
            | MockEvent::Stall { .. }
            | MockEvent::Hold { .. }
            | MockEvent::Fail { .. }
            | MockEvent::ExpectSystemPrompt { .. }
            | MockEvent::FailStream { .. }
            | MockEvent::FailStreamHolding { .. }
            | MockEvent::FailRetryable { .. }
            | MockEvent::FailExhausted { .. }
            | MockEvent::FailLoginSpent { .. }
            | MockEvent::FailInvalidRequest { .. }
            | MockEvent::FailContextOverflow { .. } => return None,
        })
    }
}

/// The JSON-encoded script at `path`, which `MEKA_MOCK_PROVIDER_SCRIPT` named.
pub(crate) fn load_script(path: &std::path::Path) -> Result<Vec<Vec<MockEvent>>> {
    let body = std::fs::read_to_string(path).map_err(|error| {
        crate::error::MekaError::Config(format!(
            "failed to read MEKA_MOCK_PROVIDER_SCRIPT='{}': {error}",
            path.display()
        ))
    })?;
    serde_json::from_str(&body).map_err(|error| {
        crate::error::MekaError::Config(format!(
            "MEKA_MOCK_PROVIDER_SCRIPT='{}' is not valid JSON: {error}",
            path.display()
        ))
    })
}

/// Send the `StreamEvent::Error` that every real driver emits immediately before returning its own
/// typed error. A dropped receiver is the normal shutdown race and carries no information here.
async fn send_stream_error(event_sender: &mpsc::Sender<StreamEvent>, message: &str) {
    if event_sender
        .send(StreamEvent::Error(message.to_string()))
        .await
        .is_err()
    {
        tracing::trace!("stream event receiver dropped");
    }
}

/// What a round scripted with [`MockEvent::ExpectSystemPrompt`] fails with. Names the words and
/// not the prompt: a test reads this off a host's stderr, and the prompt is what it is asserting.
fn system_prompt_complaint(contains: &str) -> String {
    format!("mock provider: the system prompt does not contain {contains:?}")
}

/// The notice a Claude provider sends when it redacts old images, with the report the agent counts.
fn redaction_notice(
    images: u64,
    bytes: u64,
    positions: Vec<crate::image::RedactedImage>,
) -> crate::frontend::Notice {
    crate::frontend::Notice::info(format!("Redacted {images} old image(s)")).reporting(
        crate::stats::Redaction {
            images,
            bytes,
            positions,
        },
    )
}
/// One scripted round of assistant text that ends the turn.
#[cfg(test)]
pub(crate) fn text_round(text: &str) -> Vec<MockEvent> {
    vec![
        MockEvent::Text {
            text: text.to_string(),
        },
        MockEvent::MessageEnd {
            stop_reason: MockStopReason::EndTurn,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_provider_drains_one_round_per_stream_call() {
        let provider = MockProvider::from_rounds(vec![
            vec![
                MockEvent::Text {
                    text: "hello".into(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::EndTurn,
                },
            ],
            vec![MockEvent::Text {
                text: "second".into(),
            }],
        ]);

        let (tx, mut rx) = mpsc::channel(8);
        provider
            .stream(
                CompletionRequest::new("", &[], &[]),
                tx,
                CancellationToken::new(),
            )
            .await
            .expect("first round");
        // First round emits two events then the channel sender drops.
        assert!(matches!(
            rx.recv().await,
            Some(StreamEvent::TextDelta(ref t)) if t == "hello"
        ));
        assert!(matches!(
            rx.recv().await,
            Some(StreamEvent::MessageEnd { .. })
        ));

        // Second call drains the second round.
        let (tx2, mut rx2) = mpsc::channel(8);
        provider
            .stream(
                CompletionRequest::new("", &[], &[]),
                tx2,
                CancellationToken::new(),
            )
            .await
            .expect("second round");
        assert!(matches!(
            rx2.recv().await,
            Some(StreamEvent::TextDelta(ref t)) if t == "second"
        ));
    }

    #[tokio::test]
    async fn mock_provider_completes_when_script_exhausted() {
        let provider = MockProvider::from_rounds(vec![]);
        let (tx, mut rx) = mpsc::channel(8);
        provider
            .stream(
                CompletionRequest::new("", &[], &[]),
                tx,
                CancellationToken::new(),
            )
            .await
            .expect("empty script");
        assert!(rx.recv().await.is_none(), "exhausted script emits nothing");
    }

    /// `Fail` returns `Err(MekaError::Provider(_))` from [`Provider::stream`], preceded by the
    /// `StreamEvent::Error` every real driver sends before its typed return. The agent loop turns
    /// the typed error into a non-Interrupted `run_turn` error, which the ACP layer maps to a
    /// JSON-RPC `internal_error` response.
    #[tokio::test]
    async fn mock_provider_fail_event_returns_error() {
        let provider = MockProvider::from_rounds(vec![vec![MockEvent::Fail {
            message: "boom".into(),
        }]]);
        let (tx, mut rx) = mpsc::channel(8);
        let result = provider
            .stream(
                CompletionRequest::new("", &[], &[]),
                tx,
                CancellationToken::new(),
            )
            .await;
        let error = result.expect_err("Fail must propagate as Err");
        assert!(
            matches!(&error, crate::error::MekaError::Provider(message) if message == "boom"),
            "unexpected error: {error:?}"
        );
        // The error rides the channel first, then the channel closes. `Agent` depends on that
        // ordering: its `StreamEvent::Error` arm deliberately does not return, because the typed
        // error arriving from the join handle is the one carrying the retry classification.
        assert!(
            matches!(rx.recv().await, Some(StreamEvent::Error(message)) if message == "boom"),
            "Fail must announce itself on the stream before returning",
        );
        assert!(rx.recv().await.is_none(), "and then the stream is over");
    }

    /// `FailRetryable` returns `Err(MekaError::RetryableProvider { .. })` carrying the configured
    /// `retry_after`. It mirrors `Fail`'s shape, including the preceding `StreamEvent::Error`, but
    /// with the typed variant `Agent::run_streaming`'s retry loop pattern-matches on.
    #[tokio::test]
    async fn mock_provider_fail_retryable_event_returns_typed_error() {
        let provider = MockProvider::from_rounds(vec![vec![MockEvent::FailRetryable {
            message: "overloaded".into(),
            retry_after_secs: Some(3),
        }]]);
        let (tx, mut rx) = mpsc::channel(8);
        let result = provider
            .stream(
                CompletionRequest::new("", &[], &[]),
                tx,
                CancellationToken::new(),
            )
            .await;
        match result.expect_err("FailRetryable must propagate as Err") {
            crate::error::MekaError::RetryableProvider {
                message,
                retry_after,
                ..
            } => {
                assert_eq!(message, "overloaded");
                assert_eq!(retry_after, Some(std::time::Duration::from_secs(3)));
            }
            other => panic!("expected RetryableProvider, got {other:?}"),
        }
        assert!(
            matches!(rx.recv().await, Some(StreamEvent::Error(message)) if message == "overloaded"),
            "FailRetryable must announce itself on the stream before returning",
        );
        assert!(rx.recv().await.is_none(), "and then the stream is over");
    }

    /// `complete` folds one round into a message, preserving block order: text that preceded a tool
    /// call stays ahead of it and text between two calls stays between them. Sub-agents take this
    /// path, so a reordering here would show up as a sub-agent's narration landing after its work.
    #[tokio::test]
    async fn mock_provider_complete_folds_a_round_preserving_block_order() {
        let provider = MockProvider::from_rounds(vec![
            vec![
                MockEvent::Text {
                    text: "before ".into(),
                },
                MockEvent::ToolUseStart {
                    id: "call-1".into(),
                    name: "file_read".into(),
                },
                MockEvent::ToolUseEnd {
                    input: serde_json::json!({"path": "a.txt"}),
                },
                MockEvent::Text {
                    text: "after".into(),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::ToolUse,
                },
            ],
            vec![MockEvent::Text {
                text: "second round".into(),
            }],
        ]);

        let crate::provider::Completion { message, stop, .. } = provider
            .complete(
                CompletionRequest::new("", &[], &[]),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .expect("complete");
        assert!(matches!(message.role, Role::Assistant));
        assert_eq!(message.content.len(), 3);
        assert!(
            matches!(&message.content[0], ContentBlock::Text { text, .. } if text == "before ")
        );
        assert!(
            matches!(&message.content[1], ContentBlock::ToolUse { id, name, .. } if id == "call-1" && name == "file_read")
        );
        assert!(matches!(&message.content[2], ContentBlock::Text { text, .. } if text == "after"));
        assert_eq!(stop.reason, StopReason::ToolUse);

        // The queue is shared with `stream`, so the first round is gone for both paths.
        let message = provider
            .complete(
                CompletionRequest::new("", &[], &[]),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .expect("second round")
            .message;
        assert!(
            matches!(&message.content[0], ContentBlock::Text { text, .. } if text == "second round")
        );
    }

    /// An exhausted script yields an empty assistant message rather than an error, matching
    /// `stream`'s "drains nothing, returns Ok" behavior.
    #[tokio::test]
    async fn mock_provider_complete_on_exhausted_script_is_empty() {
        let provider = MockProvider::from_rounds(vec![]);
        let crate::provider::Completion { message, stop, .. } = provider
            .complete(
                CompletionRequest::new("", &[], &[]),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .expect("complete");
        assert!(message.content.is_empty());
        assert_eq!(stop.reason, StopReason::EndTurn);
    }

    /// `ThinkingDelta` + `ThinkingComplete` map straight through to the same-named `StreamEvent`
    /// variants. The agent loop collapses the pair into a single `FrontendEvent::ThinkingBlock`,
    /// which the ACP frontend renders as a `SessionUpdate::AgentThoughtChunk` notification.
    #[tokio::test]
    async fn mock_provider_emits_thinking_delta_and_complete() {
        let provider = MockProvider::from_rounds(vec![vec![
            MockEvent::ThinkingDelta {
                text: "let me think...".into(),
            },
            MockEvent::ThinkingComplete { opaque: None },
            MockEvent::Text {
                text: "done".into(),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::EndTurn,
            },
        ]]);

        let (tx, mut rx) = mpsc::channel(8);
        provider
            .stream(
                CompletionRequest::new("", &[], &[]),
                tx,
                CancellationToken::new(),
            )
            .await
            .expect("stream");
        assert!(matches!(
            rx.recv().await,
            Some(StreamEvent::ThinkingDelta(ref t)) if t == "let me think..."
        ));
        assert!(matches!(
            rx.recv().await,
            Some(StreamEvent::ThinkingComplete { opaque: None, .. })
        ));
        assert!(matches!(
            rx.recv().await,
            Some(StreamEvent::TextDelta(ref t)) if t == "done"
        ));
        assert!(matches!(
            rx.recv().await,
            Some(StreamEvent::MessageEnd { .. })
        ));
    }
}
