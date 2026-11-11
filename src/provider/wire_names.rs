//! Tool names as a wire carries them.
//!
//! Every wire caps a tool name, and Meka's own names, an MCP server's prefixed tools above all, can
//! pass the cap. Meka keeps a tool's whole name everywhere it records one: configuration, history,
//! sub-agent specs, task rows. This is the one place a request's names are cut to the serving
//! wire's cap and a reply's are given back, so a session moves between wires with its history
//! intact. A cut name is the head of the whole name, `_`, and the first twelve hex digits of the
//! whole name's SHA-256: the same on every request, so the prompt cache holds.

use std::{borrow::Cow, collections::HashMap, sync::Arc};

use async_trait::async_trait;
use sha2::Digest;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    conversation::{ContentBlock, Message, ToolResultContent},
    error::Result,
    provider::{
        AccountIdentity, AccountUsage, Completion, CompletionRequest, DeferredToolDelivery,
        Provider, SEARCH_TOOL_NAME, StreamEvent, ToolDefinition, UsageStats,
    },
};

/// Hex digits of the digest a cut name ends with.
const DIGEST_HEX_CHARS: usize = 12;

/// `provider` with every tool name its requests carry held to `max_bytes`.
pub(crate) fn cap_tool_names(provider: Arc<dyn Provider>, max_bytes: usize) -> Arc<dyn Provider> {
    Arc::new(ToolNameCap {
        inner: provider,
        max_bytes,
    })
}

/// `name` as a wire capped at `max_bytes` carries it.
pub(crate) fn wire_name(name: &str, max_bytes: usize) -> Cow<'_, str> {
    if name.len() <= max_bytes {
        return Cow::Borrowed(name);
    }
    let digest = sha2::Sha256::digest(name.as_bytes());
    let mut suffix = String::from("_");
    for byte in digest.iter().take(DIGEST_HEX_CHARS / 2) {
        suffix.push_str(&format!("{byte:02x}"));
    }
    let mut head = max_bytes.saturating_sub(suffix.len()).min(name.len());
    while !name.is_char_boundary(head) {
        head -= 1;
    }
    Cow::Owned(format!("{}{suffix}", &name[..head]))
}

struct ToolNameCap {
    inner: Arc<dyn Provider>,
    max_bytes: usize,
}

/// A request whose names were cut, owned so the copy it sends can borrow from it, and the map
/// from each cut name back to the name it stands for.
struct CutRequest {
    messages: Vec<Message>,
    tools: Vec<ToolDefinition>,
    deferred_tools: Vec<ToolDefinition>,
    whole: HashMap<String, String>,
}

impl CutRequest {
    /// The request with its names cut, or `None` when every name already fits, which is every
    /// request but one naming a tool past the cap.
    fn of(request: &CompletionRequest<'_>, max_bytes: usize) -> Option<Self> {
        let over = |name: &str| name.len() > max_bytes;
        // The names a search call asked for, which go out cut as the model sent them.
        let searched = |name: &str, input: &serde_json::Value| -> Vec<String> {
            if name != SEARCH_TOOL_NAME {
                return Vec::new();
            }
            input
                .get("names")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|name| name.as_str().map(str::to_string))
                .collect()
        };
        let named_over = |block: &ContentBlock| {
            match block {
            ContentBlock::ToolUse { name, input, .. } => {
                over(name) || searched(name, input).iter().any(|name| over(name))
            }
            ContentBlock::ToolResult { content, .. } => content.iter().any(|item| {
                matches!(item, ToolResultContent::ToolReference { tool_name } if over(tool_name))
            }),
            _ => false,
        }
        };
        let needed = request
            .tools
            .iter()
            .chain(request.deferred_tools)
            .any(|tool| over(&tool.name))
            || request
                .messages
                .iter()
                .any(|message| message.content.iter().any(named_over));
        if !needed {
            return None;
        }
        let mut whole = HashMap::new();
        let mut cut = |name: &mut String| {
            if let Cow::Owned(short) = wire_name(name, max_bytes) {
                whole.insert(short.clone(), std::mem::replace(name, short));
            }
        };
        let mut tools = request.tools.to_vec();
        let mut deferred_tools = request.deferred_tools.to_vec();
        for tool in tools.iter_mut().chain(deferred_tools.iter_mut()) {
            cut(&mut tool.name);
        }
        let mut messages = request.messages.to_vec();
        for block in messages
            .iter_mut()
            .flat_map(|message| message.content.iter_mut())
        {
            match block {
                ContentBlock::ToolUse { name, input, .. } => {
                    if name == SEARCH_TOOL_NAME
                        && let Some(names) = input
                            .get_mut("names")
                            .and_then(serde_json::Value::as_array_mut)
                    {
                        for searched in names {
                            if let Some(whole) = searched.as_str() {
                                let mut cut_name = whole.to_string();
                                cut(&mut cut_name);
                                *searched = serde_json::Value::String(cut_name);
                            }
                        }
                    }
                    cut(name);
                }
                ContentBlock::ToolResult { content, .. } => {
                    for item in content.iter_mut() {
                        if let ToolResultContent::ToolReference { tool_name } = item {
                            cut(tool_name);
                        }
                    }
                }
                _ => {}
            }
        }
        Some(Self {
            messages,
            tools,
            deferred_tools,
            whole,
        })
    }

    /// `original` sending the cut names in place of its own.
    fn request<'a>(&'a self, original: &CompletionRequest<'a>) -> CompletionRequest<'a> {
        CompletionRequest {
            system_prompt: original.system_prompt,
            messages: &self.messages,
            tools: &self.tools,
            deferred_tools: &self.deferred_tools,
            attribution: original.attribution.clone(),
        }
    }

    /// The whole name a reply's `name` stands for. A name the request did not cut passes as it
    /// came, the model's own invention included, so it reaches the registry's refusal as written.
    fn restore(&self, name: String) -> String {
        self.whole.get(&name).cloned().unwrap_or(name)
    }

    /// `names` in a `tool_search` call's arguments, which name tools the way the request did,
    /// restored to the whole names the registry knows.
    fn restore_searched(&self, input: &mut serde_json::Value) {
        if let Some(names) = input
            .get_mut("names")
            .and_then(serde_json::Value::as_array_mut)
        {
            for name in names {
                if let Some(cut) = name.as_str() {
                    *name = serde_json::Value::String(self.restore(cut.to_string()));
                }
            }
        }
    }

    /// `event` with every name restored.
    fn restore_event(&self, event: StreamEvent) -> StreamEvent {
        match event {
            StreamEvent::ToolUseStart { id, name } => StreamEvent::ToolUseStart {
                id,
                name: self.restore(name),
            },
            StreamEvent::ToolUseEnd {
                id,
                name,
                mut input,
                provider_fields,
            } => {
                let name = self.restore(name);
                if name == SEARCH_TOOL_NAME {
                    self.restore_searched(&mut input);
                }
                StreamEvent::ToolUseEnd {
                    id,
                    name,
                    input,
                    provider_fields,
                }
            }
            StreamEvent::ToolCallRefused {
                id,
                name,
                unparsed,
                provider_fields,
            } => StreamEvent::ToolCallRefused {
                id,
                name: self.restore(name),
                unparsed,
                provider_fields,
            },
            other => other,
        }
    }

    fn restore_completion(&self, mut completion: Completion) -> Completion {
        for block in &mut completion.message.content {
            if let ContentBlock::ToolUse { name, input, .. } = block {
                *name = self.restore(std::mem::take(name));
                if name == SEARCH_TOOL_NAME {
                    self.restore_searched(input);
                }
            }
        }
        completion
    }
}

#[async_trait]
impl Provider for ToolNameCap {
    fn model(&self) -> Option<&str> {
        self.inner.model()
    }

    async fn complete(
        &self,
        request: CompletionRequest<'_>,
        cancellation: CancellationToken,
    ) -> Result<Completion> {
        let Some(cut) = CutRequest::of(&request, self.max_bytes) else {
            return self.inner.complete(request, cancellation).await;
        };
        let completion = self
            .inner
            .complete(cut.request(&request), cancellation)
            .await?;
        Ok(cut.restore_completion(completion))
    }

    async fn stream(
        &self,
        request: CompletionRequest<'_>,
        event_sender: mpsc::Sender<StreamEvent>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        let Some(cut) = CutRequest::of(&request, self.max_bytes) else {
            return self.inner.stream(request, event_sender, cancellation).await;
        };
        let (sender, mut receiver) = mpsc::channel(event_sender.max_capacity());
        // Owns the receiver, so a consumer that stops reading closes the driver's channel as it
        // would have closed the driver's own, rather than leaving the driver blocked on a full one.
        let restore = &cut;
        let forward = async move {
            while let Some(event) = receiver.recv().await {
                if event_sender
                    .send(restore.restore_event(event))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        };
        let (result, ()) = tokio::join!(
            self.inner
                .stream(cut.request(&request), sender, cancellation),
            forward
        );
        result
    }

    async fn fetch_usage(&self) -> Result<Option<AccountUsage>> {
        self.inner.fetch_usage().await
    }

    async fn fetch_identity(&self) -> Result<Option<AccountIdentity>> {
        self.inner.fetch_identity().await
    }

    async fn fetch_usage_stats(&self) -> Result<Option<UsageStats>> {
        self.inner.fetch_usage_stats().await
    }

    fn deferred_tool_delivery(&self) -> DeferredToolDelivery {
        self.inner.deferred_tool_delivery()
    }

    fn rejects_tool_name(&self, name: &str) -> bool {
        self.inner.rejects_tool_name(name)
    }

    fn resolved_effort(&self) -> Option<String> {
        self.inner.resolved_effort()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::mock::{MockEvent, MockProvider, MockStopReason};

    fn tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: "a tool".to_string(),
            parameters: serde_json::json!({"type": "object"}),
            ..Default::default()
        }
    }

    /// A name that fits goes as it is; one past the cap is cut to exactly the cap, the same way
    /// every time, and two names that share their head still differ.
    #[test]
    fn a_name_past_the_cap_is_cut_to_it_and_stays_distinct() {
        assert_eq!(wire_name("mcp__exa__search", 64), "mcp__exa__search");
        let long = format!("mcp__server__{}", "a".repeat(80));
        let other = format!("mcp__server__{}b", "a".repeat(80));
        let cut = wire_name(&long, 64);
        assert_eq!(cut.len(), 64);
        assert_eq!(cut, wire_name(&long, 64));
        assert_ne!(cut, wire_name(&other, 64));
    }

    /// A consumer that stops reading ends the stream underneath as it would have ended the
    /// driver's own: the driver's sends fail and it returns, rather than waiting on a full channel
    /// nobody drains.
    #[tokio::test]
    async fn a_consumer_that_stops_reading_ends_the_driver() {
        let long = format!("mcp__server__{}", "x".repeat(80));
        let mut events: Vec<MockEvent> = (0..16)
            .map(|index| MockEvent::Text {
                text: format!("part {index}"),
            })
            .collect();
        events.push(MockEvent::MessageEnd {
            stop_reason: MockStopReason::EndTurn,
        });
        let provider = cap_tool_names(Arc::new(MockProvider::from_rounds(vec![events])), 64);
        let tools = [tool(&long)];
        let history = [Message::user("go")];
        let (sender, receiver) = mpsc::channel(1);
        drop(receiver);
        let streamed = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            provider.stream(
                CompletionRequest::new("", &history, &tools),
                sender,
                CancellationToken::new(),
            ),
        )
        .await;
        assert!(streamed.is_ok(), "the driver returned");
    }

    /// A search that names a tool by its cut name, the name the model was shown, finds the tool by
    /// its whole name.
    #[tokio::test]
    async fn a_search_names_a_cut_tool_by_its_whole_name() {
        let long = format!("mcp__server__{}", "x".repeat(140));
        let short = wire_name(&long, 128).into_owned();
        let mock = Arc::new(MockProvider::from_rounds(vec![vec![
            MockEvent::ToolUseStart {
                id: "call-1".to_string(),
                name: SEARCH_TOOL_NAME.to_string(),
            },
            MockEvent::ToolUseEnd {
                input: serde_json::json!({"names": [short]}),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::ToolUse,
            },
        ]]));
        let provider = cap_tool_names(mock, 128);
        let tools = [tool(&long)];
        let history = [Message::user("go")];
        let (sender, mut receiver) = mpsc::channel(16);
        provider
            .stream(
                CompletionRequest::new("", &history, &tools),
                sender,
                CancellationToken::new(),
            )
            .await
            .expect("streams");
        let mut searched = None;
        while let Some(event) = receiver.recv().await {
            if let StreamEvent::ToolUseEnd { input, .. } = event {
                searched = Some(input);
            }
        }
        assert_eq!(searched, Some(serde_json::json!({"names": [long]})));
    }

    /// A search the history holds goes out naming the tools as the model was shown them, cut,
    /// as the call it restores on the way in was sent, whether or not the catalog still holds
    /// the tool.
    #[tokio::test]
    async fn a_search_in_the_history_goes_out_under_the_cut_names() {
        let long = format!("mcp__server__{}", "x".repeat(80));
        let short = wire_name(&long, 64).into_owned();
        let mock = Arc::new(MockProvider::from_rounds(vec![vec![
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::EndTurn,
            },
        ]]));
        let provider = cap_tool_names(mock.clone(), 64);
        let history = [Message::user("go"), Message {
            role: crate::conversation::Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call-1".to_string(),
                name: SEARCH_TOOL_NAME.to_string(),
                input: serde_json::json!({"names": [long, "short_one"]}),
                unparsed: None,
                provider_fields: None,
            }],
            provider_fields: None,
        }];
        let (sender, mut receiver) = mpsc::channel(16);
        provider
            .stream(
                CompletionRequest::new("", &history, &[]),
                sender,
                CancellationToken::new(),
            )
            .await
            .expect("streams");
        while receiver.recv().await.is_some() {}
        let sent = mock.streams();
        let ContentBlock::ToolUse { input, .. } = &sent[0].messages[1].content[0] else {
            panic!("{:?}", sent[0].messages[1]);
        };
        assert_eq!(input, &serde_json::json!({"names": [short, "short_one"]}));
    }

    /// The wire sees only cut names, in the catalog and in the history, and the call the model
    /// makes under one comes back under the whole name the rest of Meka knows.
    #[tokio::test]
    async fn a_request_carries_cut_names_and_the_reply_comes_back_whole() {
        let long = format!("mcp__server__{}", "x".repeat(80));
        let short = wire_name(&long, 64).into_owned();
        let mock = Arc::new(MockProvider::from_rounds(vec![vec![
            MockEvent::ToolUseStart {
                id: "call-2".to_string(),
                name: short.clone(),
            },
            MockEvent::ToolUseEnd {
                input: serde_json::json!({}),
            },
            MockEvent::MessageEnd {
                stop_reason: MockStopReason::ToolUse,
            },
        ]]));
        let provider = cap_tool_names(mock.clone(), 64);
        let tools = [tool(&long)];
        let history = [Message::user("go"), Message {
            role: crate::conversation::Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call-1".to_string(),
                name: long.clone(),
                input: serde_json::json!({}),
                unparsed: None,
                provider_fields: None,
            }],
            provider_fields: None,
        }];
        let (sender, mut receiver) = mpsc::channel(16);
        provider
            .stream(
                CompletionRequest::new("", &history, &tools),
                sender,
                CancellationToken::new(),
            )
            .await
            .expect("streams");
        let mut started = Vec::new();
        while let Some(event) = receiver.recv().await {
            if let StreamEvent::ToolUseStart { name, .. } = event {
                started.push(name);
            }
        }
        assert_eq!(started, std::slice::from_ref(&long));
        let sent = mock.streams();
        assert_eq!(sent[0].tools[0].name, short);
        assert!(
            matches!(&sent[0].messages[1].content[0], ContentBlock::ToolUse { name, .. } if *name == short),
            "the history's call goes under the cut name too"
        );
    }
}
