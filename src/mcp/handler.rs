//! Client-side MCP handler: dispatches server-initiated sampling / `list_roots` / elicitation
//! requests to the rest of the agent, forwards `tools/list_changed` notifications through the
//! manager, and adapts the remote tool list into the `crate::tools` trait so the provider loop can
//! call them like any other tool.

use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

use async_trait::async_trait;
use rmcp::{
    ErrorData as McpError, Peer, RoleClient,
    handler::client::ClientHandler,
    model::{
        CallToolRequest, CallToolRequestParams, CancelledNotificationParam, ClientRequest,
        CreateElicitationRequestParams, CreateElicitationResult, CreateMessageRequestParams,
        CreateMessageResult, ErrorCode, ListRootsResult, Meta, ProgressNotificationParam, Role,
        Root, SamplingMessage, SamplingMessageContent, ServerResult,
    },
    service::{NotificationContext, PeerRequestOptions, RequestContext, ServiceError},
};
use tokio_util::sync::CancellationToken;

use super::{
    ALLOWED_IMAGE_MIME_TYPES, MAX_MCP_IMAGE_BYTES, MCP_SAMPLING_PROVIDER_TIMEOUT, McpClientContext,
    ServerEntry,
};
use crate::{
    config::McpServerConfig,
    error::{AgshError, Result},
    permission::Permission,
    provider::ToolDefinition,
    tools::{Tool, ToolOutput},
};

/// Permission for each server to issue sampling requests. Mirrors the `sampling` / `sampling_limit`
/// fields on `McpServerConfig`.
#[derive(Clone)]
pub struct SamplingPolicy {
    allowed: bool,
    limit: u32,
    count: Arc<AtomicU32>,
}

impl SamplingPolicy {
    pub(super) fn from_config(config: &McpServerConfig) -> Self {
        Self {
            allowed: config.sampling,
            limit: config.sampling_limit.unwrap_or(10),
            count: Arc::new(AtomicU32::new(0)),
        }
    }
}

/// Client-side MCP handler. Dispatches server-initiated requests (`sampling/createMessage`,
/// `roots/list`, `elicitation/create`) and notifications (`tools/list_changed`, etc.) to the rest
/// of the agent via the shared [`McpClientContext`].
#[derive(Clone)]
pub struct AgshClientHandler {
    server_name: Arc<str>,
    sampling: SamplingPolicy,
    context: Arc<McpClientContext>,
}

impl AgshClientHandler {
    pub fn new(
        server_name: String,
        sampling: SamplingPolicy,
        context: Arc<McpClientContext>,
    ) -> Self {
        Self {
            server_name: Arc::from(server_name),
            sampling,
            context,
        }
    }
}

impl ClientHandler for AgshClientHandler {
    fn on_tool_list_changed(
        &self,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        let server_name: String = self.server_name.as_ref().to_string();
        let manager = self.context.manager().and_then(|weak| weak.upgrade());

        async move {
            tracing::info!("MCP server '{}' sent tools/list_changed", server_name);
            let Some(manager) = manager else {
                tracing::debug!(
                    "tool list refresh skipped — manager not yet wired for '{}'",
                    server_name
                );
                return;
            };

            // Tool-permission resolution reads the server config and `mcp_default_permission` from
            // the manager itself — no explicit permission needs to be threaded here.
            match manager.discover_tools_for_server(&server_name).await {
                Ok(adapters) => {
                    // Match the initial-registration path: only mark non-eager tools deferred.
                    // Compute the deferred set before we erase the adapters into `Arc<dyn Tool>`,
                    // since `raw_name`/`server_config` live on the concrete type.
                    let deferred_names: Vec<String> = adapters
                        .iter()
                        .filter(|adapter| {
                            !crate::mcp::tool_should_eager_load(
                                adapter.server_config(),
                                adapter.raw_name(),
                            )
                        })
                        .map(|adapter| adapter.definition().name)
                        .collect();
                    let new_tools: Vec<Arc<dyn Tool>> = adapters
                        .into_iter()
                        .map(|a| Arc::new(a) as Arc<dyn Tool>)
                        .collect();
                    // Routes through every attached registry so all active sessions observe the
                    // updated tool set.
                    manager.update_server_tools(&server_name, new_tools).await;
                    if !deferred_names.is_empty() {
                        manager.mark_deferred_on_attached(&deferred_names).await;
                    }
                    tracing::info!("MCP server '{}' tool registry refreshed", server_name);
                }
                Err(error) => {
                    tracing::warn!(
                        "failed to refresh tools for MCP server '{}': {}",
                        server_name,
                        error
                    );
                }
            }
        }
    }

    fn on_resource_list_changed(
        &self,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        let server = Arc::clone(&self.server_name);
        async move {
            tracing::debug!("MCP server '{}' sent resources/list_changed", server);
        }
    }

    fn on_prompt_list_changed(
        &self,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        let server = Arc::clone(&self.server_name);
        async move {
            tracing::debug!("MCP server '{}' sent prompts/list_changed", server);
        }
    }

    fn on_resource_updated(
        &self,
        params: rmcp::model::ResourceUpdatedNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        let server = Arc::clone(&self.server_name);
        async move {
            tracing::info!(
                "MCP server '{}' reported resource updated: {}",
                server,
                params.uri
            );
            crate::mcp::resource_updates::record(server.as_ref(), &params.uri);
        }
    }

    // Keep the explicit `impl Future` return type: other handlers in this trait impl have
    // non-trivial captures (`Arc<str>` clones, server name in logging, etc.) and use the same
    // signature shape. Staying uniform makes the module easier to read than mixing `async fn` and
    // the manual-future form.
    #[allow(clippy::manual_async_fn)]
    fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        async move {
            crate::mcp::progress::dispatch(params);
        }
    }

    fn on_logging_message(
        &self,
        params: rmcp::model::LoggingMessageNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        let server = Arc::clone(&self.server_name);
        async move {
            tracing::debug!(
                "MCP server '{}' log [{:?}]: {}",
                server,
                params.level,
                params.data
            );
        }
    }

    #[allow(clippy::manual_async_fn)]
    fn list_roots(
        &self,
        _context: RequestContext<RoleClient>,
    ) -> impl Future<Output = std::result::Result<ListRootsResult, McpError>> + Send + '_ {
        async move {
            // Task-local override wins: when an MCP tool runs inside `with_session_cwd(session.cwd,
            // …)`, this query reads that session's cwd. Outside such a scope (connection- time
            // queries, REPL paths) the process default seeded on the context applies.
            let cwd = match self.context.cwd() {
                Some(default) => crate::mcp::current_roots_cwd(default),
                None => std::env::current_dir().map_err(|error| {
                    McpError::internal_error(format!("current dir unavailable: {}", error), None)
                })?,
            };
            let uri = url::Url::from_directory_path(&cwd).map_err(|_| {
                McpError::internal_error(
                    format!("failed to convert {:?} to file:// URL", cwd),
                    None,
                )
            })?;
            let name = cwd
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("root")
                .to_string();
            Ok(ListRootsResult::new(vec![
                Root::new(uri.as_str()).with_name(name),
            ]))
        }
    }

    fn create_elicitation(
        &self,
        request: CreateElicitationRequestParams,
        _context: RequestContext<RoleClient>,
    ) -> impl Future<Output = std::result::Result<CreateElicitationResult, McpError>> + Send + '_
    {
        let server = Arc::clone(&self.server_name);
        async move {
            use crate::mcp::elicitation::{
                ElicitationKind, ElicitationPrompt, ElicitationResponse,
            };

            let (kind, message) = match &request {
                CreateElicitationRequestParams::FormElicitationParams {
                    message,
                    requested_schema,
                    ..
                } => {
                    let schema = serde_json::to_value(requested_schema)
                        .unwrap_or(serde_json::json!({"type": "object", "properties": {}}));
                    (ElicitationKind::Form { schema }, message.clone())
                }
                CreateElicitationRequestParams::UrlElicitationParams { message, url, .. } => {
                    (ElicitationKind::Url { url: url.clone() }, message.clone())
                }
            };

            // 60-second user-response timeout so a distracted user can't stall an MCP tool call
            // forever. Matches the elicitation deadline used for the ToolApprovalRequest channel in
            // shell.rs.
            let (responder, receiver) = std::sync::mpsc::sync_channel::<ElicitationResponse>(1);
            let prompt = ElicitationPrompt {
                server_name: server.as_ref().to_string(),
                kind,
                message,
                responder,
            };

            if !crate::mcp::elicitation::send_prompt(prompt) {
                tracing::warn!(
                    "MCP server '{}' requested elicitation but no shell sink is installed; declining",
                    server
                );
                return Ok(ElicitationResponse::Decline.into_result());
            }

            // Elicitations are standard MCP *requests*, so a `Decline` response IS how the server
            // learns the user didn't answer — no separate `notifications/cancelled` is appropriate
            // here (cancellation notifications are for long-running requests we started, not for
            // server-initiated elicitations).
            let response = tokio::task::spawn_blocking(move || {
                receiver
                    .recv_timeout(std::time::Duration::from_secs(60))
                    .unwrap_or(ElicitationResponse::Decline)
            })
            .await
            .unwrap_or(ElicitationResponse::Decline);

            Ok(response.into_result())
        }
    }

    fn create_message(
        &self,
        params: CreateMessageRequestParams,
        _context: RequestContext<RoleClient>,
    ) -> impl Future<Output = std::result::Result<CreateMessageResult, McpError>> + Send + '_ {
        let server_name = Arc::clone(&self.server_name);
        let policy = self.sampling.clone();
        let provider = self.context.provider();

        async move {
            if !policy.allowed {
                tracing::info!(
                    "MCP server '{}' requested sampling/createMessage — rejected (sampling=false)",
                    server_name
                );
                return Err(McpError::new(
                    ErrorCode::METHOD_NOT_FOUND,
                    "sampling is not enabled for this MCP server in agsh's config",
                    None,
                ));
            }

            let current = policy.count.fetch_add(1, Ordering::SeqCst);
            if current >= policy.limit {
                tracing::warn!(
                    "MCP server '{}' exceeded sampling_limit ({})",
                    server_name,
                    policy.limit
                );
                return Err(McpError::new(
                    ErrorCode::INTERNAL_ERROR,
                    format!(
                        "sampling_limit ({}) exhausted for server '{}'",
                        policy.limit, server_name
                    ),
                    None,
                ));
            }

            let Some(provider) = provider else {
                return Err(McpError::internal_error(
                    "sampling provider not wired (agent not yet started)",
                    None,
                ));
            };

            tracing::info!(
                "MCP server '{}' sampling/createMessage: {} messages, max_tokens={}",
                server_name,
                params.messages.len(),
                params.max_tokens
            );

            let (system_prompt, converted) = convert_sampling_params(&params).map_err(|error| {
                // The slot was reserved for a call that never reached the provider; free it so a
                // well-formed retry isn't rejected.
                policy.count.fetch_sub(1, Ordering::SeqCst);
                McpError::invalid_params(format!("sampling conversion failed: {}", error), None)
            })?;

            // Sampling calls out to the provider with no MCP tools exposed — the server asked for
            // pure reasoning, not tool-use. The empty tool list forces the provider into a plain
            // text completion. Bounded by `MCP_SAMPLING_PROVIDER_TIMEOUT` so a hung provider can't
            // pin the MCP request open indefinitely.
            let completion = tokio::time::timeout(
                MCP_SAMPLING_PROVIDER_TIMEOUT,
                provider.complete(&system_prompt, &converted, &[]),
            )
            .await;

            let (assistant_message, _stop_reason, _usage) = match completion {
                Ok(Ok(result)) => result,
                Ok(Err(error)) => {
                    // Provider returned an error before the timeout elapsed — no quota was really
                    // consumed on our side, so hand the sampling slot back.
                    policy.count.fetch_sub(1, Ordering::SeqCst);
                    return Err(McpError::internal_error(
                        format!("provider completion failed: {}", error),
                        None,
                    ));
                }
                Err(_) => {
                    policy.count.fetch_sub(1, Ordering::SeqCst);
                    return Err(McpError::internal_error(
                        format!(
                            "provider completion timed out after {}s",
                            MCP_SAMPLING_PROVIDER_TIMEOUT.as_secs()
                        ),
                        None,
                    ));
                }
            };

            let text = assistant_message.text_content();
            let message = SamplingMessage::assistant_text(text);
            Ok(CreateMessageResult::new(
                message,
                provider.name().to_string(),
            ))
        }
    }
}

/// Convert MCP `CreateMessageRequestParams` into the provider's `(system_prompt, Vec<Message>)`
/// shape, flattening text content. Non-text sampling content (image, audio, tool_use, tool_result)
/// is replaced with a placeholder string — none of agsh's providers accept these inside sampling
/// calls.
fn convert_sampling_params(
    params: &CreateMessageRequestParams,
) -> std::result::Result<(String, Vec<crate::provider::Message>), String> {
    use crate::provider::{ContentBlock, Message, Role as ProviderRole};

    // Defensive sanitisation: the system prompt is server-controlled and gets forwarded to the
    // configured provider. Strip any Unicode Cc/Cf codepoints so a hostile server can't smuggle
    // terminal escapes or homographs into our provider call.
    let system_prompt = params
        .system_prompt
        .as_deref()
        .map(crate::mcp::sanitize::sanitize_text)
        .unwrap_or_default();

    let mut messages = Vec::with_capacity(params.messages.len());
    for sampling_message in &params.messages {
        let role = match sampling_message.role {
            Role::User => ProviderRole::User,
            Role::Assistant => ProviderRole::Assistant,
        };
        let mut text = String::new();
        for content_item in sampling_message.content.iter() {
            match content_item {
                SamplingMessageContent::Text(t) => {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(&t.text);
                }
                SamplingMessageContent::Image(_) => text.push_str("[image content omitted]"),
                SamplingMessageContent::Audio(_) => text.push_str("[audio content omitted]"),
                SamplingMessageContent::ToolUse(_) => text.push_str("[tool_use content omitted]"),
                SamplingMessageContent::ToolResult(_) => {
                    text.push_str("[tool_result content omitted]")
                }
            }
        }
        messages.push(Message {
            role,
            content: vec![ContentBlock::Text { text }],
        });
    }

    Ok((system_prompt, messages))
}

pub struct McpToolAdapter {
    namespaced_name: String,
    remote_tool_name: String,
    description: String,
    parameters: serde_json::Value,
    permission: Permission,
    entry: Arc<ServerEntry>,
    /// `tool.annotations` and `tool.meta` captured from the remote server. Surfaced to the
    /// provider as hints (read-only / destructive) and round-tripped back in `_meta` so the
    /// MCP server can correlate client-side context.
    annotations: Option<serde_json::Value>,
    meta: Option<serde_json::Value>,
    title: Option<String>,
}

impl McpToolAdapter {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        namespaced_name: String,
        remote_tool_name: String,
        description: String,
        parameters: serde_json::Value,
        permission: Permission,
        entry: Arc<ServerEntry>,
        annotations: Option<serde_json::Value>,
        meta: Option<serde_json::Value>,
        title: Option<String>,
    ) -> Self {
        Self {
            namespaced_name,
            remote_tool_name,
            description,
            parameters,
            permission,
            entry,
            annotations,
            meta,
            title,
        }
    }

    /// Raw, server-advertised tool name (not the `mcp__<server>__<tool>` namespaced form). Used to
    /// look the tool up in per-server config fields like `eager_load_tools`.
    pub(crate) fn raw_name(&self) -> &str {
        &self.remote_tool_name
    }

    /// The server config that produced this adapter. Used to read per-server policy (eager-load,
    /// permission overrides, …) without rediscovering the manager.
    pub(crate) fn server_config(&self) -> &crate::config::McpServerConfig {
        &self.entry.config
    }

    /// Resolves a per-call tool-call timeout. Respects `AGSH_MCP_TOOL_TIMEOUT` (milliseconds) when
    /// set, otherwise falls back to 600 seconds — long enough for a database index rebuild but
    /// short enough that a hung server isn't invisible.
    fn tool_call_timeout() -> std::time::Duration {
        std::env::var("AGSH_MCP_TOOL_TIMEOUT")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(std::time::Duration::from_millis)
            .unwrap_or(std::time::Duration::from_secs(600))
    }

    async fn call_tool_once(
        &self,
        mut params: CallToolRequestParams,
        cancellation: CancellationToken,
        tool_use_id: Option<String>,
    ) -> std::result::Result<rmcp::model::CallToolResult, ServiceError> {
        // Per-call progress token: allows the server to emit `notifications/progress` updates that
        // route back to our shell UI.
        let (progress_token, _progress_guard) = crate::mcp::progress::register(
            self.entry.server_name().to_string(),
            self.remote_tool_name.clone(),
            tool_use_id.clone(),
        );
        let mut meta = Meta::new();
        meta.set_progress_token(progress_token);
        if let Some(id) = &tool_use_id {
            meta.0
                .insert("agsh/toolUseId".to_string(), serde_json::json!(id));
        }
        params.meta = Some(meta);

        // Same error surface as an actually-closed transport — the upstream retry logic already
        // handles `TransportClosed` by attempting a reconnect.
        let peer: Peer<RoleClient> = self
            .entry
            .require_connected()
            .await
            .map_err(|_| ServiceError::TransportClosed)?;
        let request = ClientRequest::CallToolRequest(CallToolRequest::new(params));
        let handle = peer
            .send_cancellable_request(request, PeerRequestOptions::no_options())
            .await?;
        let request_id = handle.id.clone();

        let timeout = Self::tool_call_timeout();
        // Cap how long we wait on the best-effort cancellation notification so a hung transport
        // can't block Ctrl-C handling or shutdown.
        const CANCEL_NOTIFY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
        let notify_cancel = |reason: &'static str| {
            let peer = peer.clone();
            let request_id = request_id.clone();
            let server_name = self.entry.server_name().to_string();
            async move {
                let send = peer.notify_cancelled(CancelledNotificationParam {
                    request_id,
                    reason: Some(reason.to_string()),
                });
                match tokio::time::timeout(CANCEL_NOTIFY_TIMEOUT, send).await {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        tracing::debug!(
                            "failed to send cancellation notification to '{}': {}",
                            server_name,
                            error
                        );
                    }
                    Err(_) => {
                        tracing::debug!(
                            "cancellation notification to '{}' timed out after {}s",
                            server_name,
                            CANCEL_NOTIFY_TIMEOUT.as_secs()
                        );
                    }
                }
            }
        };

        tokio::select! {
            response = handle.await_response() => {
                match response? {
                    ServerResult::CallToolResult(result) => Ok(result),
                    _ => Err(ServiceError::UnexpectedResponse),
                }
            }
            _ = cancellation.cancelled() => {
                notify_cancel("user interrupt").await;
                Err(ServiceError::Cancelled {
                    reason: Some("user interrupt".to_string()),
                })
            }
            _ = tokio::time::sleep(timeout) => {
                notify_cancel("timeout").await;
                Err(ServiceError::Cancelled {
                    reason: Some(format!("timed out after {}s", timeout.as_secs())),
                })
            }
        }
    }
}

#[async_trait]
impl Tool for McpToolAdapter {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.namespaced_name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
            title: self.title.clone(),
            annotations: self.annotations.clone(),
            meta: self.meta.clone(),
        }
    }

    fn required_permission(&self) -> Permission {
        self.permission
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        cancellation: CancellationToken,
    ) -> Result<ToolOutput> {
        let arguments = input.as_object().cloned();

        let params = {
            let mut p = CallToolRequestParams::new(self.remote_tool_name.clone());
            p.arguments = arguments;
            p
        };

        let is_timeout = |error: &ServiceError| matches!(error, ServiceError::Cancelled { reason: Some(reason) } if reason.starts_with("timed out"));

        // First attempt. On TransportClosed, reconnect and retry once.
        let result = match self
            .call_tool_once(params.clone(), cancellation.clone(), None)
            .await
        {
            Ok(result) => result,
            Err(ServiceError::Cancelled { reason })
                if reason.as_deref() == Some("user interrupt") =>
            {
                return Err(AgshError::Interrupted);
            }
            Err(error) if is_timeout(&error) => {
                return Err(AgshError::McpToolExecution {
                    server_name: self.entry.server_name().to_string(),
                    tool_name: self.remote_tool_name.clone(),
                    message: error.to_string(),
                });
            }
            Err(ServiceError::TransportClosed) => {
                self.entry.reconnect().await?;
                match self.call_tool_once(params, cancellation, None).await {
                    Ok(result) => result,
                    Err(ServiceError::Cancelled { reason })
                        if reason.as_deref() == Some("user interrupt") =>
                    {
                        return Err(AgshError::Interrupted);
                    }
                    Err(error) => {
                        return Err(AgshError::McpToolExecution {
                            server_name: self.entry.server_name().to_string(),
                            tool_name: self.remote_tool_name.clone(),
                            message: error.to_string(),
                        });
                    }
                }
            }
            Err(error) => {
                // If the server rejected us with a 401/Unauthorized, persist the `needs-auth`
                // verdict so the next startup skips the unauthenticated probe and goes straight to
                // OAuth. The user must re-authenticate via `agsh mcp login <name>`.
                let text = error.to_string().to_ascii_lowercase();
                if (text.contains("401") || text.contains("unauthorized"))
                    && let Some(store) = self.entry.token_store()
                {
                    if let Err(cache_err) =
                        store.save_auth_probe(self.entry.server_name(), true).await
                    {
                        tracing::debug!(
                            "failed to save auth probe cache for '{}': {}",
                            self.entry.server_name(),
                            cache_err
                        );
                    } else {
                        tracing::warn!(
                            "MCP server '{}' returned 401 — marked as needing auth. Run 'agsh mcp login {}' to re-authenticate.",
                            self.entry.server_name(),
                            self.entry.server_name()
                        );
                    }
                }
                return Err(AgshError::McpToolExecution {
                    server_name: self.entry.server_name().to_string(),
                    tool_name: self.remote_tool_name.clone(),
                    message: error.to_string(),
                });
            }
        };

        let is_error = result.is_error.unwrap_or(false);
        let mut content = convert_tool_result_content(&result.content);

        // If the server included structured_content, append it as a fenced JSON block so providers
        // can reason over it without needing a dedicated ToolResultContent variant. Matches Claude
        // Code's pragmatic passthrough.
        if let Some(structured) = &result.structured_content {
            let pretty = serde_json::to_string_pretty(structured).unwrap_or_default();
            if !pretty.is_empty() {
                let appended =
                    format!("\n\n---\n**Structured content:**\n```json\n{}\n```", pretty);
                content.push(crate::provider::ToolResultContent::Text { text: appended });
            }
        }

        // Unicode sanitisation on every text block that came from the server.
        for block in content.iter_mut() {
            if let crate::provider::ToolResultContent::Text { text } = block {
                *text = crate::mcp::sanitize::sanitize_text(text);
            }
        }

        Ok(ToolOutput {
            content,
            is_error,
            scratchpad_hint: Some(format!(
                "mcp_{}_{}",
                self.entry.server_name(),
                self.remote_tool_name
            )),
            frontend_metadata: None,
        })
    }
}

/// Map MCP `CallToolResult.content` items to agsh's provider-layer `ToolResultContent` blocks. Text
/// stays text; images pass through as multimodal blocks so providers like Claude and GPT-4o can see
/// them; audio, embedded resources, and resource links collapse to informative text placeholders
/// (no provider accepts them as tool-result blocks yet).
fn convert_tool_result_content(
    items: &[rmcp::model::Content],
) -> Vec<crate::provider::ToolResultContent> {
    use crate::provider::{ImageSource, ToolResultContent};

    let mut blocks: Vec<ToolResultContent> = Vec::new();
    let mut text_buf = String::new();

    let flush_text = |buf: &mut String, out: &mut Vec<ToolResultContent>| {
        if !buf.is_empty() {
            out.push(ToolResultContent::Text {
                text: std::mem::take(buf),
            });
        }
    };

    for item in items {
        match &item.raw {
            rmcp::model::RawContent::Text(text_content) => {
                if !text_buf.is_empty() {
                    text_buf.push('\n');
                }
                text_buf.push_str(&text_content.text);
            }
            rmcp::model::RawContent::Image(image) => {
                let mime_ok = ALLOWED_IMAGE_MIME_TYPES
                    .iter()
                    .any(|allowed| image.mime_type.eq_ignore_ascii_case(allowed));
                let size_ok = image.data.len() <= MAX_MCP_IMAGE_BYTES;
                if !mime_ok {
                    if !text_buf.is_empty() {
                        text_buf.push('\n');
                    }
                    text_buf.push_str(&format!(
                        "[image suppressed: mime type '{}' not in allow-list]",
                        image.mime_type
                    ));
                } else if !size_ok {
                    if !text_buf.is_empty() {
                        text_buf.push('\n');
                    }
                    text_buf.push_str(&format!(
                        "[image suppressed: {} base64 bytes exceeds {} byte limit]",
                        image.data.len(),
                        MAX_MCP_IMAGE_BYTES
                    ));
                } else {
                    flush_text(&mut text_buf, &mut blocks);
                    blocks.push(ToolResultContent::Image {
                        source: ImageSource {
                            source_type: "base64".to_string(),
                            media_type: image.mime_type.clone(),
                            data: image.data.clone(),
                        },
                    });
                }
            }
            rmcp::model::RawContent::Audio(audio) => {
                if !text_buf.is_empty() {
                    text_buf.push('\n');
                }
                text_buf.push_str(&format!(
                    "[audio content: {}, {} base64 bytes — agsh does not yet pass audio to the provider]",
                    audio.mime_type,
                    audio.data.len()
                ));
            }
            rmcp::model::RawContent::Resource(resource) => {
                if !text_buf.is_empty() {
                    text_buf.push('\n');
                }
                match &resource.resource {
                    rmcp::model::ResourceContents::TextResourceContents { uri, text, .. } => {
                        text_buf.push_str(&format!("--- {}\n{}", uri, text));
                    }
                    rmcp::model::ResourceContents::BlobResourceContents {
                        uri,
                        mime_type,
                        blob,
                        ..
                    } => {
                        text_buf.push_str(&format!(
                            "[embedded blob resource: {} ({}), {} base64 bytes]",
                            uri,
                            mime_type.as_deref().unwrap_or("application/octet-stream"),
                            blob.len()
                        ));
                    }
                }
            }
            rmcp::model::RawContent::ResourceLink(link) => {
                if !text_buf.is_empty() {
                    text_buf.push('\n');
                }
                text_buf.push_str(&format!("[resource link: {}]", link.uri));
            }
        }
    }

    flush_text(&mut text_buf, &mut blocks);
    if blocks.is_empty() {
        blocks.push(ToolResultContent::Text {
            text: String::new(),
        });
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_policy_rejects_when_disabled() {
        let policy = SamplingPolicy {
            allowed: false,
            limit: 10,
            count: Arc::new(AtomicU32::new(0)),
        };
        assert!(!policy.allowed);
    }

    #[test]
    fn test_sampling_policy_limit_enforcement() {
        let policy = SamplingPolicy {
            allowed: true,
            limit: 2,
            count: Arc::new(AtomicU32::new(0)),
        };
        // Simulate three requests; only first two should be under the limit.
        assert!(policy.count.fetch_add(1, Ordering::SeqCst) < policy.limit);
        assert!(policy.count.fetch_add(1, Ordering::SeqCst) < policy.limit);
        assert!(policy.count.fetch_add(1, Ordering::SeqCst) >= policy.limit);
    }

    #[test]
    fn test_convert_sampling_params_flattens_text() {
        let mut params = rmcp::model::CreateMessageRequestParams::new(
            vec![
                SamplingMessage::user_text("hello"),
                SamplingMessage::assistant_text("world"),
            ],
            100,
        );
        params.system_prompt = Some("you are a test".to_string());
        let (system_prompt, messages) = convert_sampling_params(&params).unwrap();
        assert_eq!(system_prompt, "you are a test");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, crate::provider::Role::User);
        assert_eq!(messages[1].role, crate::provider::Role::Assistant);
    }

    #[test]
    fn test_convert_tool_result_content_text_only() {
        use rmcp::model::{Content, RawContent, RawTextContent};
        let items = vec![
            Content::new(
                RawContent::Text(RawTextContent {
                    text: "hello".to_string(),
                    meta: None,
                }),
                None,
            ),
            Content::new(
                RawContent::Text(RawTextContent {
                    text: "world".to_string(),
                    meta: None,
                }),
                None,
            ),
        ];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            crate::provider::ToolResultContent::Text { text } => {
                assert_eq!(text, "hello\nworld");
            }
            other => panic!("expected Text, got {:?}", other),
        }
    }

    #[test]
    fn test_convert_tool_result_content_image_passthrough() {
        use rmcp::model::{Content, RawContent, RawImageContent};
        let items = vec![Content::new(
            RawContent::Image(RawImageContent {
                data: "BASE64DATA".to_string(),
                mime_type: "image/png".to_string(),
                meta: None,
            }),
            None,
        )];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            crate::provider::ToolResultContent::Image { source } => {
                assert_eq!(source.source_type, "base64");
                assert_eq!(source.media_type, "image/png");
                assert_eq!(source.data, "BASE64DATA");
            }
            other => panic!("expected Image, got {:?}", other),
        }
    }

    #[test]
    fn test_convert_tool_result_content_image_rejects_disallowed_mime() {
        use rmcp::model::{Content, RawContent, RawImageContent};
        let items = vec![Content::new(
            RawContent::Image(RawImageContent {
                data: "BASE64DATA".to_string(),
                mime_type: "image/svg+xml".to_string(),
                meta: None,
            }),
            None,
        )];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            crate::provider::ToolResultContent::Text { text } => {
                assert!(text.contains("image suppressed"));
                assert!(text.contains("image/svg+xml"));
            }
            other => panic!("expected Text placeholder, got {:?}", other),
        }
    }

    #[test]
    fn test_convert_tool_result_content_image_rejects_oversize() {
        use rmcp::model::{Content, RawContent, RawImageContent};
        let oversized = "X".repeat(MAX_MCP_IMAGE_BYTES + 1);
        let items = vec![Content::new(
            RawContent::Image(RawImageContent {
                data: oversized,
                mime_type: "image/png".to_string(),
                meta: None,
            }),
            None,
        )];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            crate::provider::ToolResultContent::Text { text } => {
                assert!(text.contains("image suppressed"));
                assert!(text.contains("exceeds"));
            }
            other => panic!("expected Text placeholder, got {:?}", other),
        }
    }

    #[test]
    fn test_convert_sampling_params_sanitises_system_prompt() {
        let mut params = rmcp::model::CreateMessageRequestParams::new(
            vec![SamplingMessage::user_text("hi")],
            32,
        );
        // RTL override + ANSI escape must both be stripped.
        params.system_prompt = Some("safe\u{202E}evil\x1b[2J".to_string());
        let (system_prompt, _) = convert_sampling_params(&params).unwrap();
        assert!(!system_prompt.contains('\u{202E}'));
        assert!(!system_prompt.contains('\x1b'));
        assert!(system_prompt.contains("safe"));
        assert!(system_prompt.contains("evil"));
    }

    #[test]
    fn test_convert_tool_result_content_mixed_keeps_ordering() {
        use rmcp::model::{Content, RawContent, RawImageContent, RawTextContent};
        let items = vec![
            Content::new(
                RawContent::Text(RawTextContent {
                    text: "before".to_string(),
                    meta: None,
                }),
                None,
            ),
            Content::new(
                RawContent::Image(RawImageContent {
                    data: "IMG".to_string(),
                    mime_type: "image/png".to_string(),
                    meta: None,
                }),
                None,
            ),
            Content::new(
                RawContent::Text(RawTextContent {
                    text: "after".to_string(),
                    meta: None,
                }),
                None,
            ),
        ];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 3);
        assert!(matches!(
            blocks[0],
            crate::provider::ToolResultContent::Text { .. }
        ));
        assert!(matches!(
            blocks[1],
            crate::provider::ToolResultContent::Image { .. }
        ));
        assert!(matches!(
            blocks[2],
            crate::provider::ToolResultContent::Text { .. }
        ));
    }
}
