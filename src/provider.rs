//! LLM provider abstraction. Defines the [`Provider`] trait, the shared message/content/tool types,
//! and the [`ProviderBuilder`] that returns a concrete backend.
//!
//! A backend is named for the wire protocol it speaks, not for a vendor or an auth method, because
//! neither of those identifies the request shape: one vendor serves several protocols (OpenAI has
//! both Chat Completions and Responses) and one protocol is served by many vendors (`/v1/messages`
//! by Anthropic, LiteLLM, Databricks, Ollama, …). The two subscription backends carry a vendor name
//! instead, because what they select is a billing relationship whose endpoint and client shape come
//! with it.

mod anthropic;
mod budget;
/// Scripted provider used by the integration tests. Compiled in debug builds and under the
/// `mock-provider` feature, so a release build does not carry it. Activated by `MEKA_MOCK_PROVIDER`
/// inside `host::build_shared_deps`, which every host builds through.
#[cfg(any(debug_assertions, feature = "mock-provider"))]
pub(crate) mod mock;
pub(crate) mod openai;
/// Backoff policy for retrying [`crate::error::MekaError::RetryableProvider`] failures.
pub(crate) mod retry;
pub(crate) mod sse;

use std::{sync::Arc, time::Duration};

pub(crate) use anthropic::{AnthropicMessagesProvider, ClaudeSubscriptionProvider};
use async_trait::async_trait;
pub(crate) use openai::{
    ChatGptSubscriptionProvider, OpenAiChatCompletionsProvider, OpenAiResponsesProvider,
};
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    config::ResolvedConfig,
    error::{MekaError, Result},
    store::{Store, TokenStore},
};

mod registry;
mod types;

pub(crate) use self::{registry::*, types::*};
use crate::{
    config::{Backend, ThinkingMode},
    conversation::{ContentBlock, Message, OpaqueReasoning, Role},
    frontend::Notice,
    stats::TokenUsage,
    store::AuthCredential,
};

/// Claude Code's OAuth client id, which the `claude-subscription` backend authenticates as.
pub(crate) const DEFAULT_CLAUDE_SUBSCRIPTION_CLIENT_ID: &str =
    "9d1c250a-e61b-44d9-88ed-5944d1962f5e";

/// Codex's hardcoded OpenAI OAuth client ID. Mirrors the value used by the first-party CLI at
/// `codex-rs/login/src/auth/manager.rs`.
pub(crate) const DEFAULT_CHATGPT_SUBSCRIPTION_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

/// The endpoint each backend talks to when a profile sets no `base_url`.
///
/// Named rather than written inline at each constructor so `meka account add` can *show* the
/// default it is about to apply. A prompt carrying its own copy of the string would eventually
/// offer one URL while the request went to another.
pub(crate) const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";
/// See [`DEFAULT_ANTHROPIC_BASE_URL`].
pub(crate) const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
/// See [`DEFAULT_ANTHROPIC_BASE_URL`].
pub(crate) const DEFAULT_CHATGPT_BASE_URL: &str = "https://chatgpt.com";

/// The default endpoint for `backend`.
pub(crate) fn default_base_url(backend: Backend) -> &'static str {
    match backend {
        Backend::AnthropicMessages | Backend::ClaudeSubscription => DEFAULT_ANTHROPIC_BASE_URL,
        Backend::OpenAiChatCompletions | Backend::OpenAiResponses => DEFAULT_OPENAI_BASE_URL,
        Backend::ChatGptSubscription => DEFAULT_CHATGPT_BASE_URL,
    }
}

/// How long a provider stream may go without producing anything before it is treated as dead.
///
/// This bounds *silence*, never a turn. A model thinking hard, or a request queued behind a busy
/// endpoint, keeps sending: reasoning deltas, text deltas, ping events. Nothing here caps how long
/// a turn may legitimately run, how many tool calls it may make, or how many tokens it may spend.
///
/// Measured in decodable SSE *events*, not bytes. All three drivers wrap `event_stream.next()`, and
/// `eventsource-stream` discards comment lines and refuses to dispatch a data-empty event, so a
/// keepalive that is only `: ping` does not reset this clock: an endpoint sending nothing else for
/// five minutes is treated as silent, which is the intended reading of it but not what "without a
/// byte" would mean. Bounding actual bytes would mean timing the response body underneath the SSE
/// decoder in all three drivers, for a shape no provider meka targets produces. Every provider in
/// use sends `ping` / `keep-alive` events with a data field, which do reset it.
pub(crate) const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(300);

/// How long to wait for the TCP + TLS handshake before giving up on a provider endpoint.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// The context window meka assumes when a profile doesn't state one.
///
/// meka does not look this up, probe for it, or cache it. The window is a local budgeting number
/// (it drives compaction timing, the keep-budget and the `/status` gauge, and is never sent on the
/// wire), so a wrong value cannot fail a request, and the user can state the real one via
/// `[profiles.<name>].context_window`.
///
/// 1M is right for the current flagships on both vendors and too generous for the smaller and older
/// models. That direction is deliberate but not free: planned compaction never fires when the real
/// window is smaller, so those turns hit the provider's own limit and recover through the
/// `ContextOverflow` compact-and-retry path instead, paying one rejected round trip each time. The
/// opposite default would make the common case compact at a fraction of its real window, which is
/// worse every day rather than occasionally.
pub(crate) const DEFAULT_CONTEXT_WINDOW: u64 = 1_000_000;

/// The HTTP client every provider backend uses.
///
/// Deliberately sets `connect_timeout` and `read_timeout` and *not* `timeout`. A whole-request
/// deadline would kill a legitimate long turn, which is the one thing the harness must not do;
/// `read_timeout` resets on every successful read, so it fires only when the connection has gone
/// quiet. Without either, a dropped route left the turn waiting on a socket that would never
/// produce another byte, with no error and no retry.
pub(crate) fn build_http_client(
    backend: &str,
    configure: impl FnOnce(reqwest::ClientBuilder) -> reqwest::ClientBuilder,
) -> Result<reqwest::Client> {
    configure(
        reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(STREAM_IDLE_TIMEOUT),
    )
    .build()
    .map_err(|error| MekaError::Provider(format!("failed to build {backend} HTTP client: {error}")))
}

/// The slot a conversation keeps its most recent provider request id in, so the next request can
/// name it. Held by the [`crate::agent::Agent`] and carried on each of its requests.
pub(crate) type PreviousRequestSlot = Arc<std::sync::Mutex<Option<String>>>;

/// Who a request is for, as the billing header has to describe it.
///
/// None of this can live on the provider: one `Arc<dyn Provider>` serves the main agent and every
/// sub-agent at once, so the answer differs between two requests it is handling concurrently. It
/// rides a task-local instead, mirroring the `AsyncLocalStorage` Claude Code threads the same three
/// facts through.
///
/// Every field is optional in the same way the wire segment is: absent means the request genuinely
/// has no such attribution, which is how meka's own side queries come out bare and how a
/// conversation's first request omits `cc_prev_req`.
#[derive(Clone, Debug, Default)]
pub(crate) struct Attribution {
    /// Whether this is a sub-agent's request (`cc_is_subagent`).
    pub(crate) subagent: bool,
    /// The prompt this request serves (`cc_prompt_id`).
    pub(crate) prompt_id: Option<Uuid>,
    /// Where the last response's id is kept, for `cc_prev_req`. Shared with the conversation that
    /// owns it, which is why it is a handle and not a value: turn two has to be able to name turn
    /// one's last response. Claude Code reads that off the last assistant message in its history
    /// (`t0E`); meka's conversation doesn't carry request ids, so the provider deposits them here.
    pub(crate) previous_request: Option<PreviousRequestSlot>,
}

impl Attribution {
    /// The id of the response before this request in its conversation, if one was recorded.
    pub(crate) fn previous_request_id(&self) -> Option<String> {
        self.previous_request
            .as_ref()
            .and_then(|slot| crate::sync::lock(slot).clone())
    }

    /// Remember a response's request id where the conversation's next request reads it. A
    /// malformed id is ignored rather than sent back: the header is Anthropic's to shape.
    pub(crate) fn record_request_id(&self, request_id: &str) {
        if !is_anthropic_request_id(request_id) {
            tracing::debug!("ignoring malformed provider request id '{request_id}'");
            return;
        }
        if let Some(slot) = &self.previous_request {
            *crate::sync::lock(slot) = Some(request_id.to_string());
        }
    }
}

fn is_anthropic_request_id(value: &str) -> bool {
    let Some(rest) = value.strip_prefix("req_") else {
        return false;
    };
    !rest.is_empty()
        && rest.len() <= 36
        && rest
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

/// Strip trailing slashes so a provider can append its own path with a leading `/`.
///
/// Every backend builds request URLs as `format!("{base}/some/path")`, so a base the user pasted
/// with a trailing slash would otherwise produce a doubled separator (`https://host//v1/messages`).
/// Servers are not obliged to treat that as the same route, and the ones that don't return a 404
/// that names nothing.
pub(crate) fn normalize_base_url(url: &str) -> String {
    let normalized = url.trim().trim_end_matches('/');
    if normalized != url {
        tracing::debug!("normalized provider base URL '{url}' to '{normalized}'");
    }
    normalized.to_string()
}

/// Abstraction over an LLM provider backend, each named for the wire protocol it speaks or the
/// account it bills (see the module header). Implementors are held behind
/// `Arc<dyn Provider>` and shared across concurrent tool dispatch; calls must be safe to make in
/// parallel from multiple sub-agents in one turn.
#[async_trait]
pub(crate) trait Provider: Send + Sync {
    /// Single round-trip request. Returns the assistant message, stop-reason, token-usage metadata,
    /// and any user-visible notices that arose during the request (e.g. the redaction hint from
    /// `anthropic::shared::build_body_within_budget`). The caller is expected to forward each
    /// notice to the active frontend; an empty `Vec` means nothing to surface. No streaming;
    /// the agent awaits the full response.
    async fn complete(&self, request: CompletionRequest<'_>) -> Result<Completion>;

    /// Streaming variant. The provider pushes `StreamEvent`s onto `event_sender` as they arrive.
    /// Cancellation is observed via `cancellation`; implementors must check the token and abort
    /// in-flight HTTP work when it fires.
    async fn stream(
        &self,
        request: CompletionRequest<'_>,
        event_sender: mpsc::Sender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()>;

    /// Fetch the account's current rate-limit usage (session / weekly windows and reset times).
    /// Returns `Ok(None)` for a backend with no per-account usage endpoint, which is every API-key
    /// backend; the subscription backends override this. Errors propagate so the caller can surface
    /// a refresh failure rather than silently showing nothing.
    async fn fetch_usage(&self) -> Result<Option<AccountUsage>> {
        Ok(None)
    }

    /// Fetch the account's identity (display name, plan/tier, organization, role). Same `Ok(None)`
    /// contract as [`Self::fetch_usage`]: only OAuth subscription providers override it.
    async fn fetch_identity(&self) -> Result<Option<AccountIdentity>> {
        Ok(None)
    }

    /// Fetch the account's historical usage (lifetime tokens, streaks, per-day counts, first-used
    /// date). Same `Ok(None)` contract as [`Self::fetch_usage`].
    async fn fetch_history(&self) -> Result<Option<UsageHistory>> {
        Ok(None)
    }

    /// The reasoning-effort value this provider will send on the wire (its settled
    /// `output_config.effort` / `reasoning.effort`), or `None` when it sends none. For display only
    /// (the `/status` model block); the request path reads the same settled slot. Default: `None`
    /// (providers with no effort knob).
    fn resolved_effort(&self) -> Option<String> {
        None
    }
}

struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

/// The response if it succeeded, else the classified HTTP error carrying whatever body could be
/// read. Every driver refuses a non-success status the same way, before it starts reading events.
pub(crate) async fn succeeded(
    response: reqwest::Response,
    what: &str,
) -> Result<reqwest::Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let retry_after = crate::error::parse_retry_after(response.headers());
    let response_text = response.text().await.unwrap_or_else(|error| {
        tracing::warn!("failed to read the {what} error response body: {error}");
        String::new()
    });
    Err(crate::error::provider_http_error(
        status,
        &response_text,
        retry_after,
        crate::error::ProviderRequest::Completion,
    ))
}

/// What a tool call's raw argument text becomes, for every driver. An empty body is a legitimate
/// zero-argument call (a tool with no parameters streams `""`, not `{}`); anything else must parse,
/// and what does not parse is rejected with the reason rather than replaced with `{}`, which would
/// run the tool on whatever defaults it tolerates: a valid call the model never made. Truncated
/// argument JSON is the ordinary shape of a `max_tokens` cutoff mid-call, so this is not an exotic
/// path.
pub(crate) fn finalize_tool_arguments(
    name: &str,
    raw: &str,
) -> std::result::Result<serde_json::Value, String> {
    if raw.trim().is_empty() {
        return Ok(serde_json::json!({}));
    }
    serde_json::from_str(raw).map_err(|error| {
        tracing::warn!("refusing tool call '{name}' with unparseable JSON arguments: {error}");
        format!("invalid JSON arguments: {error}")
    })
}

/// [`finalize_tool_arguments`] as the streaming drivers emit it: the call's end, or its rejection.
pub(crate) fn tool_use_event(id: String, name: String, raw: &str) -> StreamEvent {
    match finalize_tool_arguments(&name, raw) {
        Ok(input) => StreamEvent::ToolUseEnd { input },
        Err(reason) => StreamEvent::ToolCallRejected { id, name, reason },
    }
}

/// A rejected call as a content block: the reason rides under [`INVALID_TOOL_ARGS_MARKER`] so the
/// dispatch loop refuses it and the model is told why.
pub(crate) fn rejected_tool_use(id: String, name: String, reason: String) -> ContentBlock {
    ContentBlock::ToolUse {
        id,
        name,
        input: serde_json::json!({ INVALID_TOOL_ARGS_MARKER: reason }),
    }
}

/// What one completed request produced.
#[derive(Debug)]
pub(crate) struct Completion {
    pub(crate) message: Message,
    pub(crate) stop_reason: StopReason,
    pub(crate) usage: TokenUsage,
    /// Advisories the provider raised while building or sending the request, for the frontend.
    pub(crate) notices: Vec<Notice>,
}

/// How a stream of [`StreamEvent`]s becomes the message it described.
///
/// The one fold, for the agent (which also shows each event as it arrives) and for a Responses
/// `complete` (the stream aggregated with nobody watching). Two copies can disagree on when a
/// thinking block is kept, and a resumed transcript then differs from the live turn.
#[derive(Default)]
pub(crate) struct MessageAccumulator {
    content: Vec<ContentBlock>,
    text: String,
    thinking: String,
    tool_id: String,
    tool_name: String,
    stop_reason: Option<StopReason>,
    usage: TokenUsage,
    notices: Vec<Notice>,
}

impl MessageAccumulator {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The thinking of the block in progress, whole once `ThinkingComplete` arrives.
    pub(crate) fn pending_thinking(&self) -> &str {
        &self.thinking
    }

    /// The `(id, name)` of the tool call whose arguments are in progress.
    pub(crate) fn pending_tool(&self) -> (&str, &str) {
        (&self.tool_id, &self.tool_name)
    }

    fn flush_text(&mut self) {
        if !self.text.is_empty() {
            self.content.push(ContentBlock::Text {
                text: std::mem::take(&mut self.text),
            });
        }
    }

    /// Fold one event. The display-only events (`ThinkingProgress`, `Error`) change nothing here; a
    /// `Notice` is kept for [`Self::finish`].
    pub(crate) fn push(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::TextDelta(text) => self.text.push_str(&text),
            StreamEvent::ThinkingDelta(text) => self.thinking.push_str(&text),
            StreamEvent::ThinkingComplete { opaque } => {
                // Kept whenever it carries replayable state: visible text and/or something opaque.
                // Under `redact-thinking` the text is empty but the signature must survive to
                // continue the reasoning chain on the next turn, and under the Responses API the
                // sealed reasoning is the whole of what can be replayed.
                let thinking = std::mem::take(&mut self.thinking);
                if !thinking.is_empty() || opaque.is_some() {
                    self.content
                        .push(ContentBlock::Thinking { thinking, opaque });
                }
            }
            StreamEvent::RedactedThinking { data } => {
                self.content.push(ContentBlock::RedactedThinking { data });
            }
            StreamEvent::ThinkingProgress { .. } | StreamEvent::Error(_) => {}
            StreamEvent::ToolUseStart { id, name } => {
                // Text that preceded the call stays ahead of it, and text between two calls stays
                // between them.
                self.flush_text();
                self.tool_id = id;
                self.tool_name = name;
            }
            StreamEvent::ToolUseEnd { input } => self.content.push(ContentBlock::ToolUse {
                id: std::mem::take(&mut self.tool_id),
                name: std::mem::take(&mut self.tool_name),
                input,
            }),
            StreamEvent::ToolCallRejected { id, name, reason } => {
                // A malformed call keeps the assistant message's shape valid for the round trip;
                // the marker is what `resolve_and_execute_tool` refuses to run.
                self.content.push(rejected_tool_use(id, name, reason));
                self.tool_id.clear();
                self.tool_name.clear();
            }
            StreamEvent::MessageEnd { stop_reason } => self.stop_reason = Some(stop_reason),
            // Merged rather than overwritten: Anthropic streams the input and cache tiers on
            // `message_start` and the output on `message_delta`, so last-event-wins would drop the
            // input count.
            StreamEvent::Usage(usage) => self.usage.merge_stream(&usage),
            StreamEvent::Notice(notice) => self.notices.push(notice),
        }
    }

    /// What arrived before a stream failed, for a caller that keeps a partial answer: `None`
    /// unless some text did.
    pub(crate) fn partial(mut self) -> Option<Message> {
        self.flush_text();
        let has_text = self
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::Text { .. }));
        has_text.then(|| Message {
            role: Role::Assistant,
            content: self.content,
        })
    }

    /// The message, with `EndTurn` for a stream that never said how it ended.
    pub(crate) fn finish(mut self) -> Completion {
        self.flush_text();
        Completion {
            message: Message {
                role: Role::Assistant,
                content: self.content,
            },
            stop_reason: self.stop_reason.unwrap_or(StopReason::EndTurn),
            usage: self.usage,
            notices: self.notices,
        }
    }
}

async fn finalize_tool_call_accumulators(
    accumulators: &mut std::collections::HashMap<i64, ToolCallAccumulator>,
    event_sender: &mpsc::Sender<StreamEvent>,
) -> bool {
    let has_tools = !accumulators.is_empty();
    let mut indices: Vec<i64> = accumulators.keys().copied().collect();
    indices.sort();
    for index in indices {
        if let Some(accumulator) = accumulators.remove(&index) {
            if event_sender
                .send(StreamEvent::ToolUseStart {
                    id: accumulator.id.clone(),
                    name: accumulator.name.clone(),
                })
                .await
                .is_err()
            {
                tracing::trace!("stream event receiver dropped");
                return has_tools;
            }
            if event_sender
                .send(tool_use_event(
                    accumulator.id.clone(),
                    accumulator.name.clone(),
                    &accumulator.arguments,
                ))
                .await
                .is_err()
            {
                tracing::trace!("stream event receiver dropped");
                return has_tools;
            }
        }
    }
    has_tools
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;
    use crate::{conversation::Role, image::ImageSource};

    /// What a host reports for a session it has not loaded, pinned to three distinct answers.
    ///
    /// The whole function could be replaced with `None`, `Some(0)` or `Some(1)` with the suite
    /// green. Its one non-test caller is `GET /v1/sessions/{id}/context`, which answers with it for
    /// an evicted session, so a wrong number here is a percentage a client divides by, and `None`
    /// specifically is the "meka cannot say" signal that stops a client inventing one.
    #[tokio::test]
    async fn a_bindings_window_is_its_profiles_or_the_documented_default_or_nothing() {
        let mut profiles = profiles(&["stated", "silent"]);
        profiles.profile("stated").context_window = Some(32_000);
        let registry = provider_registry_for_test(profiles).await;

        assert_eq!(
            profile_context_window(&registry, "stated"),
            Some(32_000),
            "a profile that states a window reports it"
        );
        assert_eq!(
            profile_context_window(&registry, "silent"),
            Some(DEFAULT_CONTEXT_WINDOW),
            "one that states none reports the documented default, not its neighbor's value"
        );
        assert_eq!(
            profile_context_window(&registry, "departed"),
            None,
            "a profile that has left config.toml reports nothing, so no client divides by a guess"
        );
    }

    /// One `anthropic-messages` account per name, and a profile of the same name on each.
    fn profiles(names: &[&str]) -> Configured {
        Configured {
            accounts: names
                .iter()
                .map(|name| {
                    (name.to_string(), crate::config::AccountConfig {
                        backend: "anthropic-messages".to_string(),
                        ..Default::default()
                    })
                })
                .collect(),
            profiles: names
                .iter()
                .map(|name| {
                    (name.to_string(), crate::config::ProfileConfig {
                        account: name.to_string(),
                        ..Default::default()
                    })
                })
                .collect(),
        }
    }

    /// What a registry is built over, as a test states it.
    struct Configured {
        accounts: std::collections::BTreeMap<String, crate::config::AccountConfig>,
        profiles: std::collections::BTreeMap<String, crate::config::ProfileConfig>,
    }

    impl Configured {
        fn profile(&mut self, name: &str) -> &mut crate::config::ProfileConfig {
            self.profiles.get_mut(name).expect("a configured profile")
        }

        fn account(&mut self, name: &str) -> &mut crate::config::AccountConfig {
            self.accounts.get_mut(name).expect("a configured account")
        }
    }

    /// A registry over `configured`, with everything a resolution needs and nothing it does not.
    async fn provider_registry_for_test(configured: Configured) -> ProviderRegistry {
        let store = crate::store::Store::for_test().await;
        ProviderRegistry {
            accounts: configured.accounts,
            profiles: configured.profiles,
            session_context_window: None,
            default_thinking_budget: Some(4_096),
            device_ids: std::sync::Mutex::new(std::collections::HashMap::new()),
            token_store: Arc::new(store.token_store()),
            built: std::sync::Mutex::new(std::collections::HashMap::new()),
            #[cfg(any(debug_assertions, feature = "mock-provider"))]
            scripted: std::sync::Mutex::new(None),
        }
    }

    /// Each profile resolves its own thinking mode, and nothing at the run level can outrank it.
    ///
    /// A registry-level `--thinking` is exactly the shape this pins shut: a flag applied where a
    /// profile's value is read makes one profile's mode every profile's, so a session resolving a
    /// different profile gets a mode its own never stated.
    #[tokio::test]
    async fn each_profile_resolves_its_own_thinking_mode() {
        let mut profiles = profiles(&["adaptive", "off"]);
        profiles.profile("adaptive").thinking = Some(ThinkingMode::Adaptive);
        profiles.profile("off").thinking = Some(ThinkingMode::Off);
        let registry = provider_registry_for_test(profiles).await;

        assert_eq!(
            registry.settings("adaptive").expect("resolve").thinking,
            ThinkingMode::Adaptive
        );
        assert_eq!(
            registry.settings("off").expect("resolve").thinking,
            ThinkingMode::Off,
            "one registry, two profiles, two modes"
        );
    }

    /// Resolving a `claude-subscription` device id mints one and writes `config.toml`, and
    /// `profiles` here is a snapshot, so a resolver called per request never saw what it had just
    /// persisted. It minted another every time, for an identifier whose whole purpose is to stay
    /// the same, and took the config lock to write it on every context poll and status query.
    ///
    /// Isolated, because the thing under test *writes a config file and takes a cross-process file
    /// lock*. Unisolated it read the developer's real `~/.claude.json`, then `create_dir_all`ed and
    /// `flock`ed `~/.config/meka` with no timeout, so a `meka` running in another
    /// terminal would hang `cargo test` outright. It escaped rewriting the real `config.toml` only
    /// because `persist` bails when no profile of that name is there to write into, which is luck
    /// rather than design: a developer with a profile named `sub` had their device id rewritten.
    #[tokio::test]
    async fn a_device_id_is_resolved_once_per_profile_however_often_it_is_asked_for() {
        let mut profiles = profiles(&["sub"]);
        profiles.account("sub").backend = "claude-subscription".to_string();
        let registry = provider_registry_for_test(profiles).await;
        let config_dir = tempfile::tempdir().expect("tempdir");
        // Seeded with the account, so `persist` takes its write path rather than bailing against
        // the real file.
        std::fs::write(
            config_dir.path().join("config.toml"),
            "[accounts.sub]\nbackend = \"claude-subscription\"\n",
        )
        .expect("write config.toml");

        // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serializes every test
        // that touches it, and the guard is held across the whole set -> resolve -> clear cycle.
        let _env = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", config_dir.path()) };

        let first = registry.settings("sub").expect("resolve").device_id;
        let second = registry.settings("sub").expect("resolve").device_id;

        let persisted = std::fs::read_to_string(config_dir.path().join("config.toml"))
            .expect("read config.toml back");
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        assert!(!first.is_empty(), "a subscription account gets one");
        assert_eq!(
            first, second,
            "asking twice must not mint a second identifier"
        );
        assert!(
            persisted.contains(&first),
            "the minted id must be written back to the account, or the next process mints another"
        );
    }

    /// The device id is on disk before the first request asks for it. Resolved lazily, the first
    /// `settings` call wrote `config.toml` under the config lock on whatever runtime thread asked.
    #[tokio::test]
    async fn device_ids_are_resolved_ahead_of_the_first_request() {
        let mut profiles = profiles(&["sub"]);
        profiles.account("sub").backend = "claude-subscription".to_string();
        let registry = provider_registry_for_test(profiles).await;
        let config_dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            config_dir.path().join("config.toml"),
            "[accounts.sub]\nbackend = \"claude-subscription\"\n",
        )
        .expect("write config.toml");

        // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serializes every test
        // that touches it, and the guard is held across the whole set -> resolve -> clear cycle.
        let _env = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", config_dir.path()) };

        registry.preload_device_ids().await;
        let persisted = std::fs::read_to_string(config_dir.path().join("config.toml"))
            .expect("read config.toml back");
        let asked = registry.settings("sub").expect("resolve").device_id;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        assert!(
            persisted.contains("device_id"),
            "the id is written before anything asks: {persisted}"
        );
        assert!(
            persisted.contains(&asked),
            "and the one later asks get is the one that was written"
        );
    }

    /// Keyed on the profile and its overrides alone, the memo lets a built provider outlive the
    /// credential it was built from: `meka account login work` against the store a running `meka
    /// serve` is using rotates the key, and every later build in that process still hands out the
    /// provider holding the revoked one. The writer is another process, so this has to be a pull
    /// rather than an invalidation hook.
    #[tokio::test]
    async fn rotating_a_credential_retires_the_provider_built_from_it() {
        let mut profiles = profiles(&["work"]);
        profiles.profile("work").model = Some("claude-opus-5".to_string());
        let registry = provider_registry_for_test(profiles).await;
        registry
            .token_store
            .save_account_credential("work", &AuthCredential::ApiKey("first".to_string()))
            .await
            .expect("store a credential");

        let (first, _) = registry.resolve("work").await.expect("build");
        let (again, _) = registry.resolve("work").await.expect("build again");
        assert!(
            Arc::ptr_eq(&first, &again),
            "an unmoved credential must still be served from the memo -- keeping the connection \
             pool is the whole reason this caches"
        );

        registry
            .token_store
            .save_account_credential("work", &AuthCredential::ApiKey("second".to_string()))
            .await
            .expect("rotate the credential");

        let (after, _) = registry
            .resolve("work")
            .await
            .expect("build after the rotation");
        assert!(
            !Arc::ptr_eq(&first, &after),
            "a rotated credential must retire the provider built from its predecessor"
        );
    }

    /// Two profiles on one account are two providers over one stored credential: the row is keyed
    /// by the account, so one login serves every model configured against it.
    #[tokio::test]
    async fn two_profiles_on_one_account_share_its_credential() {
        let mut configured = profiles(&["work", "fast"]);
        configured.accounts.remove("fast");
        configured.profile("fast").account = "work".to_string();
        configured.profile("work").model = Some("claude-opus-5".to_string());
        configured.profile("fast").model = Some("claude-haiku-4-5".to_string());
        let registry = provider_registry_for_test(configured).await;
        registry
            .token_store
            .save_account_credential("work", &AuthCredential::ApiKey("shared".to_string()))
            .await
            .expect("one credential, under the account");

        let (work, _) = registry.resolve("work").await.expect("build work");
        let (fast, _) = registry.resolve("fast").await.expect("build fast");
        assert!(!Arc::ptr_eq(&work, &fast), "two profiles are two providers");
        assert_eq!(
            registry
                .token_store
                .list_credential_accounts()
                .await
                .expect("list"),
            vec!["work".to_string()],
            "and nothing was stored under a profile name"
        );
    }

    /// The refusal every turn-running site depends on. Falling back to the default here would put
    /// the conversation on another account without saying so, which is the bug being fixed.
    #[tokio::test]
    async fn an_unconfigured_profile_is_refused_by_name_and_lists_what_exists() {
        let registry = provider_registry_for_test(profiles(&["work", "personal"])).await;
        let error = registry
            .settings("retired")
            .expect_err("a profile that is not configured cannot resolve");

        assert!(
            matches!(error, MekaError::ProfileNotConfigured { .. }),
            "refused by type, so every host maps it once: {error:?}"
        );
        let error = error.to_string();
        assert!(
            error.contains("'retired'"),
            "names the missing one: {error}"
        );
        assert!(error.contains("work"), "lists what is available: {error}");
        assert!(error.contains("personal"), "lists all of them: {error}");
    }

    /// The empty case says so rather than trailing a colon into nothing. This is also the state a
    /// store carried forward with no resolvable provider lands in.
    #[tokio::test]
    async fn no_profiles_at_all_says_so() {
        let registry = provider_registry_for_test(profiles(&[])).await;
        let error = registry
            .settings("anything")
            .expect_err("nothing can resolve")
            .to_string();

        assert!(error.contains("(none configured)"), "{error}");
    }

    #[tokio::test]
    async fn a_configured_profile_resolves() {
        let registry = provider_registry_for_test(profiles(&["work"])).await;
        assert_eq!(
            registry.settings("work").expect("configured").account,
            "work"
        );
    }

    /// `AuthCredential` derived `Debug` over its plaintext, and a provider struct holding one is
    /// exactly the kind of thing that lands in a `{:?}` during a bad afternoon.
    #[test]
    fn only_a_well_formed_request_id_is_carried_forward() {
        for rejected in [
            "011CeJwF1NDYzXkUu6cyFJp2",
            "req_",
            "req_has spaces",
            "req_has;semicolon",
            "req_0123456789012345678901234567890123456789",
        ] {
            let slot = PreviousRequestSlot::default();
            let attribution = Attribution {
                previous_request: Some(Arc::clone(&slot)),
                ..Default::default()
            };
            attribution.record_request_id(rejected);
            assert_eq!(
                attribution.previous_request_id(),
                None,
                "'{rejected}' must not reach the billing header"
            );
        }
        let attribution = Attribution {
            previous_request: Some(PreviousRequestSlot::default()),
            ..Default::default()
        };
        attribution.record_request_id("req_011CeJwF1NDYzXkUu6cyFJp2");
        assert_eq!(
            attribution.previous_request_id().as_deref(),
            Some("req_011CeJwF1NDYzXkUu6cyFJp2")
        );
    }

    /// A sub-agent records its own responses in its own slot, so a sub-agent's last response never
    /// lands in its spawner's header and vice versa. The slots are the agents' own; the
    /// attributions merely carry them.
    #[test]
    fn an_attribution_without_a_slot_records_nothing() {
        let attribution = Attribution::default();
        attribution.record_request_id("req_011CeJwF1NDYzXkUu6cyFJp2");
        assert_eq!(attribution.previous_request_id(), None);
    }

    #[test]
    fn a_credential_never_reaches_a_debug_rendering() {
        let rendered = format!(
            "{:?}",
            AuthCredential::ApiKey("sk-APIKEYSECRET".to_string())
        );
        assert!(!rendered.contains("APIKEYSECRET"), "{rendered}");
        assert!(rendered.contains("REDACTED"), "{rendered}");

        let rendered = format!("{:?}", AuthCredential::OAuthToken {
            access_token: "ACCESSSECRET".to_string(),
            refresh_token: Some("REFRESHSECRET".to_string()),
            expires_at: Some(42),
            account_id: Some("acct-visible".to_string()),
        });
        assert!(!rendered.contains("ACCESSSECRET"), "{rendered}");
        assert!(!rendered.contains("REFRESHSECRET"), "{rendered}");
        // The non-secret identity stays: it is what a wrong-account diagnosis needs.
        assert!(rendered.contains("acct-visible"), "{rendered}");
    }

    #[test]
    fn a_base_url_keeps_its_path_and_loses_only_trailing_slashes() {
        assert_eq!(
            normalize_base_url("https://openrouter.ai/api/v1/"),
            "https://openrouter.ai/api/v1"
        );
        assert_eq!(
            normalize_base_url("https://openrouter.ai/api/v1///"),
            "https://openrouter.ai/api/v1"
        );
        assert_eq!(
            normalize_base_url("  https://openrouter.ai/api/v1  "),
            "https://openrouter.ai/api/v1"
        );
        // Already clean: byte-identical, so the common path rewrites nothing.
        assert_eq!(
            normalize_base_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1"
        );
    }

    #[test]
    fn the_generic_normalizer_leaves_the_version_segment_alone() {
        // The OpenAI family carries `/v1` in the base by convention, so stripping it here would
        // break every profile pasted from a provider's own documentation.
        assert_eq!(
            normalize_base_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            normalize_base_url("https://api.synthetic.new/openai/v1/"),
            "https://api.synthetic.new/openai/v1"
        );
    }

    /// The four ways a thinking mode is spelled must agree.
    ///
    /// A mode is written to `config.toml` by `meka profile add`, read back by serde, accepted on
    /// the CLI by clap, and printed by `/status`. Each of those has its own derivation of the
    /// string, and today they coincide only because every variant happens to be one lowercase word:
    /// clap kebab-cases the variant name, serde lowercases it, and the other two are hand-written.
    /// Adding a two-word variant would split them silently - clap taking `some-mode` and serde
    /// `somemode` - so the agreement is pinned here rather than left to that coincidence.
    #[test]
    fn every_spelling_of_a_thinking_mode_agrees() {
        #[derive(Deserialize)]
        struct Profile {
            thinking: ThinkingMode,
        }

        for mode in ThinkingMode::ALL {
            let written = mode.name();
            let parsed: Profile = toml::from_str(&format!("thinking = \"{written}\""))
                .unwrap_or_else(|error| panic!("serde must read back `{written}`: {error}"));
            assert_eq!(parsed.thinking, mode);
            assert_eq!(
                written.parse::<ThinkingMode>(),
                Ok(mode),
                "the CLI must accept what the profile is written with: {written}"
            );
        }
    }

    /// The prompt's default endpoint has to be the constructor's default endpoint.
    ///
    /// `meka account add` shows this URL as the value an empty answer accepts, and each provider
    /// reads the constant directly, so the backend-name mapping is the one joint where they can
    /// disagree. A wrong arm here compiles, passes every other test, and quietly tells every user
    /// of that backend their requests go somewhere they do not.
    #[test]
    fn the_advertised_default_endpoint_is_the_one_the_backend_uses() {
        assert_eq!(
            default_base_url(Backend::AnthropicMessages),
            DEFAULT_ANTHROPIC_BASE_URL
        );
        assert_eq!(
            default_base_url(Backend::ClaudeSubscription),
            DEFAULT_ANTHROPIC_BASE_URL
        );
        assert_eq!(
            default_base_url(Backend::OpenAiChatCompletions),
            DEFAULT_OPENAI_BASE_URL
        );
        // The two OpenAI protocols share a default host: `openai-responses` is a different wire
        // format against the same API, not a different service. Aiming it at the subscription's
        // `chatgpt.com` would compile and pass a mere is-some check, so it is named explicitly.
        assert_eq!(
            default_base_url(Backend::OpenAiResponses),
            DEFAULT_OPENAI_BASE_URL
        );
        assert_eq!(
            default_base_url(Backend::ChatGptSubscription),
            DEFAULT_CHATGPT_BASE_URL
        );
    }

    #[test]
    fn an_unconfigured_effort_is_omitted_so_the_provider_applies_its_own() {
        // The whole policy: meka names a tier only when the profile did. It cannot know which tiers
        // a given endpoint implements - `anthropic-messages` and `openai-chat-completions` reach
        // any compatible server - and an unimplemented tier is a rejected request, not a
        // graceful ignore.
        assert_eq!(resolve_effort_level(None), None);
        // A blank override reads as unset rather than as an empty wire value the API would reject.
        assert_eq!(resolve_effort_level(Some("")), None);
        assert_eq!(resolve_effort_level(Some("   ")), None);
        // A configured value is absolute: verbatim, trimmed and lowercased, never clamped, and
        // never rejected for the model it is aimed at.
        assert_eq!(
            resolve_effort_level(Some("medium")).as_deref(),
            Some("medium")
        );
        assert_eq!(
            resolve_effort_level(Some("  XHigh ")).as_deref(),
            Some("xhigh")
        );
        assert_eq!(resolve_effort_level(Some("max")).as_deref(), Some("max"));
    }

    #[test]
    fn user_with_images_appends_image_blocks_after_text() {
        let images = vec![
            ImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: "AAAA".to_string(),
            },
            ImageSource::Base64 {
                media_type: "image/jpeg".to_string(),
                data: "BBBB".to_string(),
            },
        ];
        let message = Message::user_with_images("look at these", images);
        assert_eq!(message.role, Role::User);
        assert_eq!(message.content.len(), 3);
        assert!(
            matches!(&message.content[0], ContentBlock::Text { text } if text == "look at these")
        );
        assert!(
            matches!(&message.content[1], ContentBlock::Image { source } if source.media_type() == "image/png")
        );
        assert!(
            matches!(&message.content[2], ContentBlock::Image { source } if source.media_type() == "image/jpeg")
        );
        // No images yields the same shape as `Message::user`.
        assert_eq!(Message::user_with_images("hi", vec![]).content.len(), 1);
    }

    /// A `tool_result` whose `content` is a bare string is refused, not silently wrapped.
    ///
    /// `content` is a list, and the bare-string form must stay unparseable: the release script is
    /// the only thing that converts those rows, and a silent wrap here would make it optional while
    /// leaving it the only converter.
    #[test]
    fn a_tool_result_content_written_as_a_string_no_longer_parses() {
        let stored =
            r#"{"type":"tool_result","tool_use_id":"toolu_1","content":"bare","is_error":false}"#;
        assert!(
            serde_json::from_str::<ContentBlock>(stored).is_err(),
            "the string form must not resurrect as a silent wrap"
        );

        let converted = r#"{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"bare"}],"is_error":false}"#;
        let block: ContentBlock = serde_json::from_str(converted).expect("the converted shape");
        assert!(matches!(
            block,
            ContentBlock::ToolResult { ref content, .. } if content.len() == 1
        ));
    }

    /// Both shapes survive a round trip, and `Sealed` omits an id it does not have.
    ///
    /// Named for what it can actually check. There is no backward compatibility to test: 0.42 reads
    /// only this shape, and a store written by an earlier release is brought forward by the
    /// migration script, not by serde. The `skip_serializing_if` is the part worth pinning -- drop
    /// it and every id-less sealed block starts writing `"id":null` into the log.
    #[test]
    fn both_opaque_shapes_round_trip_and_an_absent_id_is_not_written() {
        for (stored, expected) in [
            (
                r#"{"type":"thinking","thinking":"hmm","opaque":{"type":"signed","signature":"SIG"}}"#,
                OpaqueReasoning::Signed {
                    signature: "SIG".to_string(),
                },
            ),
            (
                r#"{"type":"thinking","thinking":"s","opaque":{"type":"sealed","encrypted_content":"E"}}"#,
                OpaqueReasoning::Sealed {
                    encrypted_content: "E".to_string(),
                    id: None,
                },
            ),
        ] {
            let block: ContentBlock = serde_json::from_str(stored).expect("must load");
            assert!(
                matches!(&block, ContentBlock::Thinking { opaque: Some(opaque), .. }
                    if *opaque == expected),
                "got {block:?}"
            );
            assert_eq!(
                serde_json::to_string(&block).expect("serialize"),
                stored,
                "an absent id must not be written back as null"
            );
        }
    }

    #[test]
    fn without_tool_use_keeps_text_and_thinking_drops_tool_use() {
        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "hmm".to_string(),
                    opaque: None,
                },
                ContentBlock::Text {
                    text: "let me check".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({}),
                },
            ],
        };
        let stripped = message.without_tool_use();
        assert_eq!(stripped.role, Role::Assistant);
        assert_eq!(stripped.content.len(), 2);
        assert!(matches!(
            &stripped.content[0],
            ContentBlock::Thinking { .. }
        ));
        assert!(
            matches!(&stripped.content[1], ContentBlock::Text { text } if text == "let me check")
        );
        // A tool-use-only message strips to empty content.
        let only_tool = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_2".to_string(),
                name: "x".to_string(),
                input: serde_json::json!({}),
            }],
        };
        assert!(only_tool.without_tool_use().content.is_empty());
    }

    #[test]
    fn token_usage_merge_stream_keeps_input_from_start_output_from_delta() {
        // Anthropic streaming: `message_start` carries the input/cache tiers (output a
        // placeholder), `message_delta` carries the final output with the input/cache
        // fields absent (parsed as 0).
        let mut usage = TokenUsage::default();
        usage.merge_stream(&TokenUsage {
            input_tokens: 1000,
            output_tokens: 1,
            cache_creation_input_tokens: 200,
            cache_read_input_tokens: 5000,
        });
        usage.merge_stream(&TokenUsage {
            input_tokens: 0,
            output_tokens: 250,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        });
        assert_eq!(
            usage.input_tokens, 1000,
            "input retained from message_start"
        );
        assert_eq!(usage.cache_creation_input_tokens, 200);
        assert_eq!(usage.cache_read_input_tokens, 5000);
        assert_eq!(usage.output_tokens, 250, "output taken from message_delta");
    }

    #[test]
    fn token_usage_merge_stream_single_event_is_verbatim() {
        // OpenAI/Codex emit a single usage event; merging from default keeps it as-is.
        let mut usage = TokenUsage::default();
        usage.merge_stream(&TokenUsage {
            input_tokens: 800,
            output_tokens: 120,
            ..Default::default()
        });
        assert_eq!(usage.input_tokens, 800);
        assert_eq!(usage.output_tokens, 120);
        assert_eq!(usage.cache_read_input_tokens, 0);
    }

    #[test]
    fn auth_credential_json_round_trip() {
        // `AuthCredential` is serialized to JSON for storage in `account_credentials`; both
        // variants must survive a round-trip intact.
        let api_key = AuthCredential::ApiKey("sk-test".to_string());
        let json = serde_json::to_string(&api_key).expect("serialize ApiKey");
        match serde_json::from_str::<AuthCredential>(&json).expect("deserialize ApiKey") {
            AuthCredential::ApiKey(key) => assert_eq!(key, "sk-test"),
            other => panic!("expected ApiKey, got {other:?}"),
        }

        let oauth = AuthCredential::OAuthToken {
            access_token: "access".to_string(),
            refresh_token: Some("refresh".to_string()),
            expires_at: Some(1_700_000_000_000),
            account_id: Some("acct".to_string()),
        };
        let json = serde_json::to_string(&oauth).expect("serialize OAuthToken");
        match serde_json::from_str::<AuthCredential>(&json).expect("deserialize OAuthToken") {
            AuthCredential::OAuthToken {
                access_token,
                refresh_token,
                expires_at,
                account_id,
            } => {
                assert_eq!(access_token, "access");
                assert_eq!(refresh_token.as_deref(), Some("refresh"));
                assert_eq!(expires_at, Some(1_700_000_000_000));
                assert_eq!(account_id.as_deref(), Some("acct"));
            }
            other => panic!("expected OAuthToken, got {other:?}"),
        }
    }

    /// Regression test for the "silent `{}` fallback" bug: a tool call with unparseable JSON
    /// arguments must be rejected via [`StreamEvent::ToolCallRejected`] rather than replayed with
    /// an empty input object (which would run the tool on whatever defaults it happens to
    /// tolerate).
    #[tokio::test]
    async fn finalize_tool_call_accumulators_rejects_invalid_json() {
        let mut accumulators = std::collections::HashMap::new();
        accumulators.insert(0, ToolCallAccumulator {
            id: "call-1".to_string(),
            name: "read_file".to_string(),
            arguments: "{not json".to_string(),
        });

        let (sender, mut receiver) = mpsc::channel::<StreamEvent>(16);
        let has_tools = finalize_tool_call_accumulators(&mut accumulators, &sender).await;
        assert!(has_tools, "accumulator was non-empty");

        let first = receiver.try_recv().expect("ToolUseStart emitted first");
        assert!(
            matches!(first, StreamEvent::ToolUseStart { .. }),
            "expected ToolUseStart, got {first:?}"
        );

        let second = receiver.try_recv().expect("follow-up event");
        match second {
            StreamEvent::ToolCallRejected { id, name, reason } => {
                assert_eq!(id, "call-1");
                assert_eq!(name, "read_file");
                assert!(reason.starts_with("invalid JSON arguments"));
            }
            other => panic!("expected ToolCallRejected, got {other:?}"),
        }

        assert!(
            receiver.try_recv().is_err(),
            "no further events after rejection"
        );
    }

    #[tokio::test]
    async fn finalize_tool_call_accumulators_passes_valid_json() {
        let mut accumulators = std::collections::HashMap::new();
        accumulators.insert(0, ToolCallAccumulator {
            id: "call-2".to_string(),
            name: "read_file".to_string(),
            arguments: r#"{"path": "/tmp/x"}"#.to_string(),
        });

        let (sender, mut receiver) = mpsc::channel::<StreamEvent>(16);
        finalize_tool_call_accumulators(&mut accumulators, &sender).await;

        let first = receiver.try_recv().expect("ToolUseStart");
        assert!(matches!(first, StreamEvent::ToolUseStart { .. }));

        match receiver.try_recv().expect("ToolUseEnd") {
            StreamEvent::ToolUseEnd { input } => {
                assert_eq!(input["path"], "/tmp/x");
            }
            other => panic!("expected ToolUseEnd, got {other:?}"),
        }
    }

    #[test]
    fn a_user_message_carries_its_text() {
        let message = Message::user("hello");
        assert_eq!(message.role, Role::User);
        assert_eq!(message.text_content(), "hello");
    }

    #[test]
    fn an_assistant_text_message_carries_its_text() {
        let message = Message::assistant_text("response");
        assert_eq!(message.role, Role::Assistant);
        assert_eq!(message.text_content(), "response");
    }

    #[test]
    fn message_tool_uses() {
        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "I'll read that file.".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test.txt"}),
                },
            ],
        };
        assert_eq!(message.tool_uses().len(), 1);
    }

    #[test]
    fn a_text_block_round_trips_through_serde() {
        let block = ContentBlock::Text {
            text: "hello".to_string(),
        };
        let json = serde_json::to_string(&block).expect("should serialize");
        let deserialized: ContentBlock = serde_json::from_str(&json).expect("should deserialize");

        if let ContentBlock::Text { text } = deserialized {
            assert_eq!(text, "hello");
        } else {
            panic!("expected Text block");
        }
    }

    #[test]
    fn message_serialization_roundtrip() {
        let message = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "Let me read that.".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test"}),
                },
            ],
        };

        let json = serde_json::to_string(&message).expect("should serialize");
        let deserialized: Message = serde_json::from_str(&json).expect("should deserialize");

        assert_eq!(deserialized.role, Role::Assistant);
        assert_eq!(deserialized.content.len(), 2);
        assert_eq!(deserialized.text_content(), "Let me read that.");
    }

    /// Every backend must actually build with the credential it takes.
    ///
    /// Iterating [`Backend::ALL`] rather than naming five backends is the point: a sixth is covered
    /// the day it is added.
    #[test]
    fn every_supported_backend_builds() {
        for backend in Backend::ALL {
            // Hand each backend the credential shape it accepts; the mismatch cases are asserted
            // by `every_supported_backend_refuses_the_credential_kind_it_does_not_take`.
            let credential = if backend.name().ends_with("-subscription") {
                AuthCredential::OAuthToken {
                    access_token: "token".to_string(),
                    refresh_token: None,
                    expires_at: None,
                    account_id: None,
                }
            } else {
                AuthCredential::ApiKey("key".to_string())
            };
            let built = ProviderBuilder::new(backend, credential, "some-model")
                .credential_key(Some("work".to_string()))
                .device_id("a".repeat(64))
                .build();
            assert!(
                built.is_ok(),
                "{backend} is supported but does not build: {:?}",
                built.err()
            );
        }
    }

    /// Every backend refuses the credential kind it cannot serve, and names one that can.
    ///
    /// The inverse of [`every_supported_backend_builds`], and iterating the list for the same
    /// reason: a sixth backend is covered the day it is added.
    ///
    /// Written after `openai-responses` and `openai-chat-completions` were found silently
    /// *accepting* an OAuth bundle: they matched it, kept the bearer, discarded `refresh_token` and
    /// `expires_at`, and were constructed with no `token_store`, so the credential could never
    /// refresh and the profile stopped working at expiry with nothing naming why. Their three
    /// siblings had refused the mismatch since they were written. Only these two had no test, and
    /// only these two were wrong, which is the whole argument for asserting this over the list
    /// rather than one backend at a time.
    #[test]
    fn every_supported_backend_refuses_the_credential_kind_it_does_not_take() {
        for backend in Backend::ALL {
            // The same rule `every_supported_backend_builds` pairs by, read the other way round.
            let takes_oauth = backend.name().ends_with("-subscription");
            let mismatched = if takes_oauth {
                AuthCredential::ApiKey("key".to_string())
            } else {
                AuthCredential::OAuthToken {
                    access_token: "token".to_string(),
                    refresh_token: Some("refresh".to_string()),
                    expires_at: None,
                    account_id: None,
                }
            };
            let Err(error) = ProviderBuilder::new(backend, mismatched, "some-model")
                .device_id("a".repeat(64))
                .build()
            else {
                panic!("{backend} accepted the credential kind it cannot serve");
            };
            let message = error.to_string();
            assert!(
                message.contains(backend.name()),
                "{backend}'s refusal does not name it: {message}"
            );
            // Naming a backend that *does* take this credential is what makes the refusal
            // actionable: without it the user is told what is wrong and not what to switch to.
            assert!(
                Backend::ALL
                    .iter()
                    .any(|other| *other != backend && message.contains(other.name())),
                "{backend}'s refusal names no backend to switch to: {message}"
            );
        }
    }

    #[test]
    fn a_chat_completions_profile_builds_from_an_api_key() {
        let result = ProviderBuilder::new(
            Backend::OpenAiChatCompletions,
            AuthCredential::ApiKey("key".to_string()),
            "gpt-4o",
        )
        .device_id("a".repeat(64))
        .build();
        assert!(result.is_ok());
    }

    #[test]
    fn an_anthropic_messages_profile_builds_from_an_api_key() {
        let result = ProviderBuilder::new(
            Backend::AnthropicMessages,
            AuthCredential::ApiKey("key".to_string()),
            "claude-sonnet-4-20250514",
        )
        .thinking(ThinkingMode::Off, 10000)
        .build();
        assert!(result.is_ok());
    }

    #[test]
    fn a_claude_subscription_profile_builds_from_an_oauth_token() {
        let result = ProviderBuilder::new(
            Backend::ClaudeSubscription,
            AuthCredential::OAuthToken {
                access_token: "sk-ant-oat01-test".to_string(),
                refresh_token: None,
                expires_at: None,
                account_id: None,
            },
            "claude-sonnet-4-20250514",
        )
        .credential_key(Some("work".to_string()))
        .device_id("a".repeat(64))
        .build();
        assert!(result.is_ok());
    }

    #[test]
    fn anthropic_messages_refuses_an_oauth_token() {
        let result = ProviderBuilder::new(
            Backend::AnthropicMessages,
            AuthCredential::OAuthToken {
                access_token: "sk-ant-oat01-test".to_string(),
                refresh_token: None,
                expires_at: None,
                account_id: None,
            },
            "claude-sonnet-4-20250514",
        )
        .build();
        assert!(result.is_err());
    }

    #[test]
    fn claude_subscription_refuses_an_api_key() {
        let result = ProviderBuilder::new(
            Backend::ClaudeSubscription,
            AuthCredential::ApiKey("sk-ant-api03-test".to_string()),
            "claude-sonnet-4-20250514",
        )
        .build();
        assert!(result.is_err());
    }

    #[test]
    fn a_chatgpt_subscription_profile_builds_from_an_oauth_token() {
        let result = ProviderBuilder::new(
            Backend::ChatGptSubscription,
            AuthCredential::OAuthToken {
                access_token: "codex-access".to_string(),
                refresh_token: Some("codex-refresh".to_string()),
                expires_at: Some(now_ms_in_far_future()),
                account_id: Some("workspace-1".to_string()),
            },
            "gpt-5",
        )
        .credential_key(Some("work".to_string()))
        .effort(Some("high".to_string()))
        .build();
        assert!(result.is_ok());
    }

    /// A subscription provider is built only with the account its credential is stored under.
    ///
    /// It writes refreshed tokens back to that `account_credentials` row. A default in its place
    /// (the backend name) was a second answer to a question the registry already answers: the row
    /// would be one no account names, `account list` would report it as an orphan, and the real
    /// account's credential would stay stale.
    #[test]
    fn a_subscription_provider_is_not_built_without_the_account_its_credential_lives_under() {
        for backend in [Backend::ClaudeSubscription, Backend::ChatGptSubscription] {
            let builder = || {
                ProviderBuilder::new(
                    backend,
                    AuthCredential::OAuthToken {
                        access_token: "token".to_string(),
                        refresh_token: None,
                        expires_at: None,
                        account_id: None,
                    },
                    "some-model",
                )
                .device_id("a".repeat(64))
            };
            let Err(error) = builder().build() else {
                panic!("{backend} built with no account to write refreshed tokens to");
            };
            assert!(error.to_string().contains("account"), "{error}");
            assert!(
                builder()
                    .credential_key(Some("work".to_string()))
                    .build()
                    .is_ok(),
                "{backend} builds once the account is named"
            );
        }
    }

    /// The registry refuses a profile whose model is the empty string the way it refuses one with
    /// no model: `""` is valid TOML that `is_none()` does not see, and it would go to the provider
    /// as the model's name.
    #[tokio::test]
    async fn the_registry_refuses_a_profile_whose_model_is_empty() {
        let mut profiles = profiles(&["blank", "named"]);
        profiles.profile("blank").model = Some(String::new());
        profiles.profile("named").model = Some("some-model".to_string());
        let registry = provider_registry_for_test(profiles).await;
        for account in ["blank", "named"] {
            registry
                .token_store
                .save_account_credential(account, &AuthCredential::ApiKey("key".to_string()))
                .await
                .expect("save");
        }
        let Err(error) = registry.resolve("blank").await else {
            panic!("an empty model was handed to a provider");
        };
        assert!(error.to_string().contains("names no model"), "{error}");
        assert!(
            registry.resolve("named").await.is_ok(),
            "a profile that names a model resolves"
        );
    }

    #[test]
    fn chatgpt_subscription_refuses_an_api_key() {
        let result = ProviderBuilder::new(
            Backend::ChatGptSubscription,
            AuthCredential::ApiKey("sk-...".to_string()),
            "gpt-5",
        )
        .build();
        assert!(result.is_err());
    }

    fn now_ms_in_far_future() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_millis() as i64 + 86_400_000)
            .unwrap_or(0)
    }

    #[test]
    fn auth_credential_api_key_header() {
        let credential = AuthCredential::ApiKey("my-key".to_string());
        let (name, value) = credential.auth_header();
        assert_eq!(name, "x-api-key");
        assert_eq!(value, "my-key");
    }

    #[test]
    fn auth_credential_oauth_header() {
        let credential = AuthCredential::OAuthToken {
            access_token: "my-token".to_string(),
            refresh_token: None,
            expires_at: None,
            account_id: None,
        };
        let (name, value) = credential.auth_header();
        assert_eq!(name, "Authorization");
        assert_eq!(value, "Bearer my-token");
    }
}
