//! The registry's view of MCP.
//!
//! [`crate::mcp`] is a client: it connects servers, lists what they offer and makes calls. What
//! makes a remote tool callable by the model, and what keeps every registry current as servers
//! connect, refresh and reconnect, lives here, so the client never has to know what a registry is.

use std::sync::Arc;

use async_trait::async_trait;

use super::{Tool, ToolContext, ToolOutput, ToolRegistry, mcp_resources};
use crate::{
    error::Result,
    mcp::{
        CallContext, McpClientManager, McpTool, ServerTools, ServerToolsObserver,
        handler::convert_tool_result_content,
    },
    permission::Permission,
    provider::ToolDefinition,
};

/// One remote tool as the registry sees it.
pub(crate) struct McpToolAdapter {
    tool: Arc<McpTool>,
}

impl McpToolAdapter {
    pub(crate) fn new(tool: Arc<McpTool>) -> Self {
        Self { tool }
    }
}

#[async_trait]
impl Tool for McpToolAdapter {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.tool.namespaced_name.clone(),
            description: self.tool.description.clone(),
            parameters: self.tool.parameters.clone(),
            title: self.tool.title.clone(),
            annotations: self.tool.annotations.clone(),
            meta: self.tool.meta.clone(),
        }
    }

    fn required_permission(&self) -> Permission {
        self.tool.permission
    }

    /// An MCP call runs in the server's own process, which meka spawns but does not sandbox.
    fn runs_outside_confinement(&self) -> bool {
        true
    }

    async fn execute(&self, input: serde_json::Value, context: ToolContext) -> Result<ToolOutput> {
        let arguments = forwarded_arguments(&input, &self.tool.parameters);
        let call = CallContext {
            session_id: context.session_id,
            tool_call_id: context.tool_call_id,
            frontend: context.frontend,
            cancellation: context.cancellation,
        };
        let result = self.tool.call(arguments, &call).await?;
        Ok(tool_output_from_result(
            &result,
            format!("mcp_{}_{}", self.tool.server_name(), self.tool.raw_name()),
        ))
    }
}

/// Put one server's listing into a registry.
///
/// Marks after replacing, because [`ToolRegistry::replace_server_tools`] drops the deferred marks
/// of the names it removes.
fn apply(registry: &ToolRegistry, server_name: &str, tools: &ServerTools) {
    let adapted = tools
        .tools
        .iter()
        .map(|tool| Arc::new(McpToolAdapter::new(Arc::clone(tool))) as Arc<dyn Tool>)
        .collect();
    registry.replace_server_tools(server_name, adapted);
    for name in &tools.deferred {
        registry.mark_deferred(name);
    }
}

impl ServerToolsObserver for ToolRegistry {
    fn identity(&self) -> usize {
        self.inner_identity()
    }

    fn server_tools_changed(&self, server_name: &str, tools: &ServerTools) {
        apply(self, server_name, tools);
    }
}

/// Wire a session's registry to the manager: it learns everything discovered so far, follows every
/// change until [`detach_session_registry`], and carries the resource and prompt meta-tools, which
/// delegate through `ServerEntry::require_connected` themselves and so tolerate servers that are
/// still Pending or have Failed until a specific one is called.
///
/// Takes the manager by `Arc` so the registry can hold a `Weak` back to it. `load_tool` needs that
/// to explain that a name it cannot find belongs to a server that is not connected, rather than
/// reporting it as unknown.
pub(crate) async fn attach_session_registry(
    manager: &Arc<McpClientManager>,
    registry: ToolRegistry,
) {
    registry.set_mcp_manager(Arc::downgrade(manager));
    mcp_resources::register_all(&registry, Arc::clone(manager));
    manager.subscribe(Arc::new(registry)).await;
}

/// Undo [`attach_session_registry`]. Identity is the registry's inner allocation, so any clone of
/// the handle that attached will do.
pub(crate) async fn detach_session_registry(manager: &McpClientManager, registry: &ToolRegistry) {
    manager.unsubscribe(registry.inner_identity()).await;
}

/// Install MCP tools onto a freshly built worker registry. Mirrors [`attach_session_registry`]
/// minus the subscription: only already-`Connected` servers contribute tools, and Pending or Failed
/// servers are skipped silently, so their tools simply do not appear in the worker's catalog. A
/// server the registry's denials name is not even listed.
///
/// Mirrors the connector's deferred-mark step, so a worker sees the same eager-vs-deferred split as
/// its parent. Idempotent and safe to call concurrently from separate `agent_spawn` invocations
/// operating on distinct registries.
pub(crate) async fn install_on_worker_registry(
    manager: &Arc<McpClientManager>,
    registry: &ToolRegistry,
) {
    // A worker that reaches for a dead server's tool should get the same answer the parent would,
    // not a bare "not registered".
    registry.set_mcp_manager(Arc::downgrade(manager));
    mcp_resources::register_all(registry, Arc::clone(manager));
    for name in manager.server_names() {
        // Skip the round trip entirely rather than discovering and then filtering: a denied
        // server should not even be listed, and `list_all_tools` on a server the worker cannot
        // use is latency the spawn pays for nothing.
        if registry.denials().denies_server(&name) {
            tracing::info!("MCP server '{name}' denied for sub-agent registry");
            continue;
        }
        let tools = match manager.discover_server_tools(&name).await {
            Ok(tools) => tools,
            Err(error) => {
                // Pending / Failed servers fall through `require_connected` as Err; that's
                // normal, not worth a warn. The worker just won't see this server's tools until it
                // next runs (and the parent's connector finishes the handshake).
                tracing::debug!("MCP server '{name}' skipped for sub-agent registry: {error}");
                continue;
            }
        };
        if tools.tools.is_empty() {
            continue;
        }
        apply(registry, &name, &tools);
    }
}

/// The registry form of the tool a scheduled job's gate names, or `None` when no connected server
/// offers it.
pub(crate) async fn tool_by_name(manager: &McpClientManager, name: &str) -> Option<Arc<dyn Tool>> {
    manager
        .tool_by_name(name)
        .await
        .map(|tool| Arc::new(McpToolAdapter::new(tool)) as Arc<dyn Tool>)
}

/// Convert a server's `CallToolResult` into meka's [`ToolOutput`].
///
/// Split out of `execute` so it can be tested without a live server: everything above it in
/// `execute` is transport and retry, and none of that bears on how a result is shaped.
fn tool_output_from_result(
    result: &rmcp::model::CallToolResult,
    scratchpad_hint: String,
) -> ToolOutput {
    let mut content = convert_tool_result_content(&result.content);

    // If the server included structured_content, append it as a fenced JSON block so providers
    // can reason over it without needing a dedicated ToolResultContent variant. Matches Claude
    // Code's pragmatic passthrough.
    //
    // This block is for the model to *read*. Callers that compute on the result take the
    // `structured` field below instead, so the wording and fencing here stay free to change
    // without altering what a scheduled job's gate predicate decides.
    if let Some(structured) = &result.structured_content {
        let pretty = serde_json::to_string_pretty(structured).unwrap_or_default();
        if !pretty.is_empty() {
            let appended = format!("\n\n---\n**Structured content:**\n```json\n{pretty}\n```");
            content.push(crate::conversation::ToolResultContent::Text { text: appended });
        }
    }

    // Unicode sanitization on every text block that came from the server.
    for block in content.iter_mut() {
        if let crate::conversation::ToolResultContent::Text { text } = block {
            *text = crate::text::sanitize_text(text);
        }
    }

    ToolOutput {
        content,
        is_error: result.is_error.unwrap_or(false),
        scratchpad_hint: Some(scratchpad_hint),
        frontend_metadata: None,
        structured: result.structured_content.clone(),
    }
}

/// The arguments an MCP call actually carries: everything the model sent, minus meka's own.
///
/// `scratchpad` and `background` are accepted on every tool and consumed by the agent loop, so a
/// remote server never declared them. Forwarding them sends a property the server did not ask for,
/// which a strict schema validator on the far side rejects outright -- failing a call whose only
/// fault was that the model used a meka feature. The tool documentation already said this happened;
/// it did not.
///
/// `schema` is the tool's own advertised `input_schema`, and a name it declares belongs to *it*.
/// `offer_background` in `src/tools.rs` already refuses to splice `background` onto a tool that
/// advertises the name, precisely so a server owning it keeps its meaning -- but this side stripped
/// unconditionally, so the value the model sent for the *server's* parameter was deleted on the way
/// out and the call arrived missing an argument it had asked for. Both halves have to consult the
/// schema or the pair is incoherent.
fn forwarded_arguments(
    input: &serde_json::Value,
    schema: &serde_json::Value,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    let declares = |name: &str| {
        schema
            .get("properties")
            .and_then(serde_json::Value::as_object)
            .is_some_and(|properties| properties.contains_key(name))
    };
    let strip_scratchpad = !declares(crate::tools::SCRATCHPAD_PARAMETER);
    let strip_background = !declares(crate::tools::BACKGROUND_PARAMETER);
    input.as_object().map(|object| {
        object
            .iter()
            .filter(|(key, _)| {
                !((strip_scratchpad && key.as_str() == crate::tools::SCRATCHPAD_PARAMETER)
                    || (strip_background && key.as_str() == crate::tools::BACKGROUND_PARAMETER))
            })
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A server's structured output reaches callers as data, not only as rendered prose.
    ///
    /// The fenced block is what the model reads and is deliberately presentational, so anything
    /// deciding *on* a result -- a scheduled job's gate predicate is the caller this exists for --
    /// has to take the field. Recovering the JSON by parsing the block back out would make that
    /// format string a wire format between two parts of meka while it reads as formatting, and a
    /// readability edit would then silently change what a gate decides. Both halves are asserted
    /// here so neither can quietly stop happening.
    #[test]
    fn structured_content_is_carried_as_data_and_still_rendered_for_the_model() {
        let structured = serde_json::json!({ "chats": [{ "id": "a" }], "checked_at": "now" });
        let mut result =
            rmcp::model::CallToolResult::success(vec![rmcp::model::ContentBlock::text(
                "1 unseen chat",
            )]);
        result.structured_content = Some(structured.clone());

        let output = tool_output_from_result(&result, "mcp_bridge_unseen".to_string());

        assert_eq!(
            output.structured.as_ref(),
            Some(&structured),
            "a predicate must reach the value without parsing the rendering"
        );
        let rendered = output
            .content
            .iter()
            .filter_map(|block| match block {
                crate::conversation::ToolResultContent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert!(
            rendered.contains("**Structured content:**") && rendered.contains("\"chats\""),
            "the model still sees the fenced block: {rendered}"
        );
    }

    /// The common case: a server that sends only text leaves `structured` empty rather than
    /// inventing a value, so a pointer predicate knows to fall back to parsing the text itself.
    #[test]
    fn a_text_only_result_carries_no_structured_value() {
        let result = rmcp::model::CallToolResult::success(vec![rmcp::model::ContentBlock::text(
            "{\"chats\": []}",
        )]);

        let output = tool_output_from_result(&result, "mcp_bridge_unseen".to_string());

        assert!(output.structured.is_none());
    }

    /// A remote server sees the model's arguments and nothing of meka's.
    ///
    /// `scratchpad` and `background` are accepted on every tool and consumed here, so no server
    /// declares them; sending one is an undeclared property, and a server validating its schema
    /// strictly refuses the call over it. `tools/overview.md` documented this stripping before the
    /// code did it. A server that declares `background` or `scratchpad` itself owns the name, and
    /// the value the model sent for it must reach the server.
    ///
    /// `offer_background` (src/tools.rs) already declines to splice `background` onto a tool that
    /// advertises it, exactly so the server keeps the name. This side stripped unconditionally, so
    /// the pair disagreed: meka left the server's own parameter in the schema the model reads, then
    /// deleted the model's answer on the way out, and the call arrived missing a required argument.
    #[test]
    fn a_parameter_the_server_declares_is_forwarded_not_stripped() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "prompt": {}, "background": { "type": "string" } }
        });
        let arguments = forwarded_arguments(
            &serde_json::json!({
                "prompt": "a cat",
                "background": "transparent",
                "scratchpad": "out",
            }),
            &schema,
        )
        .expect("an object of arguments");

        assert_eq!(
            arguments.get("background").and_then(|v| v.as_str()),
            Some("transparent"),
            "the server declared `background`, so it is the server's parameter"
        );
        assert!(
            !arguments.contains_key("scratchpad"),
            "`scratchpad` is still meka's here, and is still stripped"
        );
    }

    #[test]
    fn meka_only_parameters_are_not_forwarded_to_the_server() {
        let arguments = forwarded_arguments(
            &serde_json::json!({
                "query": "rust",
                "limit": 10,
                "scratchpad": "results",
                "background": true,
            }),
            // A schema declaring neither name: the ordinary case, where both are meka's.
            &serde_json::json!({"type": "object", "properties": {"query": {}, "limit": {}}}),
        )
        .expect("an object of arguments");

        assert!(!arguments.contains_key("scratchpad"));
        assert!(!arguments.contains_key("background"));
        assert_eq!(arguments.get("query"), Some(&serde_json::json!("rust")));
        assert_eq!(arguments.get("limit"), Some(&serde_json::json!(10)));
    }

    /// A tool taking no arguments still sends `{}` rather than nothing, and a non-object input is
    /// passed through as "no arguments" the way it always was.
    #[test]
    fn stripping_leaves_an_argumentless_call_intact() {
        assert_eq!(
            forwarded_arguments(&serde_json::json!({}), &serde_json::json!({})),
            Some(serde_json::Map::new()),
        );
        assert_eq!(
            forwarded_arguments(&serde_json::Value::Null, &serde_json::json!({})),
            None
        );
    }
}
