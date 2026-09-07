//! Client-side MCP handler: dispatches server-initiated `elicitation/create` requests to the rest
//! of the agent, forwards `tools/list_changed` notifications through the manager, and adapts the
//! remote tool list into the `crate::tools` trait so the provider loop can call them like any other
//! tool.
//!
//! meka does not implement the MCP sampling / roots / logging handlers: those features are
//! deprecated by SEP-2577 and slated for removal from the protocol, so the rmcp defaults apply
//! (sampling → `method_not_found`, roots → empty, logging → ignored).

use std::sync::Arc;

use rmcp::{
    ErrorData as McpError, Peer, RoleClient,
    handler::client::ClientHandler,
    model::{
        CallToolRequest, CallToolRequestParams, CancelledNotificationParam, ClientRequest,
        ConstString, CustomNotification, ElicitRequestParams, ElicitResult, ElicitationAction,
        ElicitationResponseNotificationMethod, ProgressNotificationParam, RequestMetaObject,
        ServerResult,
    },
    service::{NotificationContext, PeerRequestOptions, RequestContext, ServiceError},
};

use super::{ALLOWED_IMAGE_MIME_TYPES, MAX_MCP_IMAGE_BYTES, McpClientContext, ServerEntry};
use crate::{
    error::{MekaError, Result},
    frontend::ElicitationResponse,
    permission::Permission,
};

impl ElicitationResponse {
    /// The answer in the shape the server's `elicitation/create` request is completed with.
    pub(crate) fn into_result(self) -> ElicitResult {
        match self {
            ElicitationResponse::Accept {
                content: Some(content),
            } => ElicitResult::new(ElicitationAction::Accept).with_content(content),
            ElicitationResponse::Accept { content: None } => {
                ElicitResult::new(ElicitationAction::Accept)
            }
            ElicitationResponse::Decline => ElicitResult::new(ElicitationAction::Decline),
            ElicitationResponse::Cancel => ElicitResult::new(ElicitationAction::Cancel),
        }
    }
}

/// Client-side MCP handler. Dispatches server-initiated `elicitation/create` requests and
/// notifications (`tools/list_changed`, progress, etc.) to the rest of the agent via the shared
/// [`McpClientContext`]. Sampling / roots / logging are intentionally not handled (SEP-2577).
#[derive(Clone)]
pub(crate) struct MekaClientHandler {
    server_name: Arc<str>,
    context: Arc<McpClientContext>,
}

impl MekaClientHandler {
    pub(crate) fn new(server_name: String, context: Arc<McpClientContext>) -> Self {
        Self {
            server_name: Arc::from(server_name),
            context,
        }
    }
}

impl ClientHandler for MekaClientHandler {
    /// What the `initialize` request says about this client. Left to rmcp's default, it named the
    /// SDK as the client, floated the protocol version with the SDK's release, and declared no
    /// capabilities at all, so a server that checks before it elicits never did.
    fn get_info(&self) -> rmcp::model::ClientInfo {
        use rmcp::model::{
            ClientCapabilities, ElicitationCapability, FormElicitationCapability, Implementation,
            ProtocolVersion, UrlElicitationCapability,
        };
        let mut capabilities = ClientCapabilities::default();
        capabilities.elicitation = Some(
            ElicitationCapability::new()
                .with_form(FormElicitationCapability::new())
                .with_url(UrlElicitationCapability::new()),
        );
        rmcp::model::ClientInfo::new(
            capabilities,
            Implementation::new("meka", env!("CARGO_PKG_VERSION")),
        )
        .with_protocol_version(ProtocolVersion::V_2025_11_25)
    }

    fn on_tool_list_changed(
        &self,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        let server_name: String = self.server_name.as_ref().to_string();
        let manager = self.context.manager().and_then(|weak| weak.upgrade());

        async move {
            tracing::debug!("MCP server '{server_name}' sent tools/list_changed");
            let Some(manager) = manager else {
                tracing::debug!(
                    "tool list refresh skipped: manager not yet wired for '{server_name}'"
                );
                return;
            };

            // Tool-permission resolution reads the server config and `mcp_default_permission` from
            // the manager itself; no explicit permission needs to be threaded here.
            match manager.discover_tools_for_server(&server_name).await {
                Ok(adapters) => {
                    // The same door the initial registration goes through, so a refresh cannot
                    // classify a tool differently from the listing it replaces. Routes through
                    // every attached registry so all active sessions observe the updated tool set.
                    manager.register_server_tools(&server_name, adapters).await;
                    tracing::info!("MCP server '{server_name}' tool registry refreshed");
                }
                Err(error) => {
                    tracing::warn!(
                        "failed to refresh tools for MCP server '{server_name}': {error}"
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
            tracing::debug!("MCP server '{server}' sent resources/list_changed");
        }
    }

    fn on_prompt_list_changed(
        &self,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        let server = Arc::clone(&self.server_name);
        async move {
            tracing::debug!("MCP server '{server}' sent prompts/list_changed");
        }
    }

    fn on_resource_updated(
        &self,
        params: rmcp::model::ResourceUpdatedNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        let server = Arc::clone(&self.server_name);
        let context = Arc::clone(&self.context);
        async move {
            tracing::info!(
                "MCP server '{server}' reported resource updated: {uri}",
                uri = params.uri
            );
            context
                .resource_updates
                .record(server.as_ref(), &params.uri);
        }
    }

    #[allow(
        clippy::manual_async_fn,
        reason = "every handler in this impl spells the `impl Future` signature out; one `async fn` among them would read as a different kind of method"
    )]
    fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        let context = Arc::clone(&self.context);
        async move {
            context.progress.dispatch(params).await;
        }
    }

    /// Notification that a server-side URL elicitation the user was sent to complete has finished.
    /// rmcp 3.1 has no typed hook for this, so it arrives as an unrecognized method: the wire
    /// notification exists, but the SDK routes it nowhere specific. meka's
    /// [`Self::create_elicitation`] already returned its response synchronously, so nothing needs
    /// to drive here; log it for observability.
    fn on_custom_notification(
        &self,
        notification: CustomNotification,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        let server = Arc::clone(&self.server_name);
        async move {
            if notification.method != ElicitationResponseNotificationMethod::VALUE {
                return;
            }
            let elicitation_id = notification
                .params
                .as_ref()
                .and_then(|params| params.get("elicitationId"))
                .and_then(|id| id.as_str())
                .unwrap_or("<unknown>");
            tracing::debug!("MCP server '{server}' completed URL elicitation '{elicitation_id}'");
        }
    }

    fn create_elicitation(
        &self,
        request: ElicitRequestParams,
        _context: RequestContext<RoleClient>,
    ) -> impl Future<Output = std::result::Result<ElicitResult, McpError>> + Send + '_ {
        let server = Arc::clone(&self.server_name);
        let context = Arc::clone(&self.context);
        async move {
            use crate::frontend::{ElicitationKind, ElicitationPrompt};

            let (kind, message) = match &request {
                ElicitRequestParams::FormElicitationParams {
                    message,
                    requested_schema,
                    ..
                } => {
                    let schema = serde_json::to_value(requested_schema)
                        .unwrap_or(serde_json::json!({"type": "object", "properties": {}}));
                    (ElicitationKind::Form { schema }, message.clone())
                }
                ElicitRequestParams::UrlElicitationParams { message, url, .. } => {
                    (ElicitationKind::Url { url: url.clone() }, message.clone())
                }
                // Forward-compat: an elicitation kind this build doesn't recognize falls back to a
                // generic form prompt rather than failing the request.
                _ => (
                    ElicitationKind::Form {
                        schema: serde_json::json!({"type": "object", "properties": {}}),
                    },
                    "unsupported elicitation request".to_string(),
                ),
            };

            let prompt = ElicitationPrompt {
                server_name: server.as_ref().to_string(),
                kind,
                message,
            };

            // Correlate the elicitation back to the in-flight call's frontend via the per-server
            // lookup on the progress registry. When no call from `server` is in flight (the server
            // elicited outside of a tool call, or the progress guard already dropped), there's no
            // human to ask, and declining is the safe answer.
            let frontend = context.progress.find_frontend_for_server(server.as_ref());
            let Some(frontend) = frontend else {
                tracing::warn!(
                    "MCP server '{server}' requested elicitation but no in-flight call's frontend was \
                     registered; declining"
                );
                return Ok(ElicitationResponse::Decline.into_result());
            };

            // User-response timeout so a distracted user can't stall an MCP tool call forever;
            // sixty seconds is the elicitation deadline in its own right (an approval prompt waits
            // longer, since the turn is already paused on it). Elicitations are MCP *requests*, so
            // a `Decline` response IS how the server learns the user didn't answer; no separate
            // `notifications/canceled` is appropriate here (cancellation notifications are for
            // long-running requests we started, not for server-initiated elicitations).
            const ELICITATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
            let response = match tokio::time::timeout(
                ELICITATION_TIMEOUT,
                frontend.handle_elicitation(prompt),
            )
            .await
            {
                Ok(response) => response,
                Err(_) => {
                    tracing::warn!(
                        "MCP server '{server}' elicitation timed out after {seconds}s; declining",
                        seconds = ELICITATION_TIMEOUT.as_secs()
                    );
                    ElicitationResponse::Decline
                }
            };

            Ok(response.into_result())
        }
    }
}

/// One tool a server advertised, as the client knows it: what to send, what to call it, and the
/// permission its listing resolved to. `crate::tools::mcp_adapter` is what makes one callable by
/// the model.
pub(crate) struct McpTool {
    pub(crate) namespaced_name: String,
    /// Raw, server-advertised tool name (not the `mcp__<server>__<tool>` namespaced form). Used to
    /// look the tool up in per-server config fields like `eager_load_tools`.
    pub(crate) remote_tool_name: String,
    pub(crate) description: String,
    pub(crate) parameters: serde_json::Value,
    pub(crate) permission: Permission,
    pub(crate) entry: Arc<ServerEntry>,
    /// `tool.annotations` and `tool.meta` captured from the remote server. Surfaced to the
    /// provider as hints (read-only / destructive) and round-tripped back in `_meta` so the
    /// MCP server can correlate client-side context.
    pub(crate) annotations: Option<serde_json::Value>,
    pub(crate) meta: Option<serde_json::Value>,
    pub(crate) title: Option<String>,
}

/// What one remote call carries besides its arguments: the ids the server may correlate on, where
/// its progress goes, and what stops it.
pub(crate) struct CallContext {
    pub(crate) session_id: Option<uuid::Uuid>,
    pub(crate) tool_call_id: Option<String>,
    pub(crate) frontend: Arc<dyn crate::frontend::Frontend>,
    pub(crate) cancellation: tokio_util::sync::CancellationToken,
}

/// `MEKA_MCP_TOOL_TIMEOUT` as a duration, or the default when unset.
///
/// A value that does not parse, or a zero, is warned about and ignored rather than silently
/// defaulted: a bare number has no unit and either guess is invisible when wrong, and a zero would
/// time out every call before it is sent.
fn parse_tool_call_timeout(raw: Option<&str>) -> std::time::Duration {
    const DEFAULT: std::time::Duration = std::time::Duration::from_secs(600);
    let Some(raw) = raw else {
        return DEFAULT;
    };
    match humantime_serde::re::humantime::parse_duration(raw.trim()) {
        Ok(timeout) if !timeout.is_zero() => timeout,
        Ok(_) => {
            tracing::warn!(
                "ignoring MEKA_MCP_TOOL_TIMEOUT='{raw}': a zero timeout fails every call"
            );
            DEFAULT
        }
        Err(error) => {
            tracing::warn!(
                "ignoring MEKA_MCP_TOOL_TIMEOUT='{raw}': {error} (expected a duration like \"10m\")"
            );
            DEFAULT
        }
    }
}

impl McpTool {
    /// The remote name; see [`Self::remote_tool_name`].
    pub(crate) fn raw_name(&self) -> &str {
        &self.remote_tool_name
    }

    /// The server config that produced this adapter. Used to read per-server policy (eager-load,
    /// permission overrides, …) without rediscovering the manager.
    pub(crate) fn server_config(&self) -> &crate::config::McpServerConfig {
        &self.entry.config
    }

    pub(crate) fn server_name(&self) -> &str {
        self.entry.server_name()
    }

    /// Resolves a per-call tool-call timeout. Respects `MEKA_MCP_TOOL_TIMEOUT` (a humantime
    /// string such as `"10m"`) when set, otherwise falls back to 600 seconds, long enough for a
    /// database index rebuild but short enough that a hung server isn't invisible.
    fn tool_call_timeout() -> std::time::Duration {
        parse_tool_call_timeout(std::env::var("MEKA_MCP_TOOL_TIMEOUT").ok().as_deref())
    }

    async fn call_tool_once(
        &self,
        mut params: CallToolRequestParams,
        context: &CallContext,
    ) -> std::result::Result<rmcp::model::CallToolResult, ServiceError> {
        let cancellation = context.cancellation.clone();
        // Per-call progress token: allows the server to emit `notifications/progress` updates that
        // route back to our shell UI. The frontend snapshot is taken from the task-local installed
        // by `Agent::run_tool` and stored on the registry entry so the rmcp notification handler
        // (which runs on a separately-spawned task; see `rmcp::service::spawn_service_task`) can
        // look it up by token. `None` outside an agent-driven call site falls through to a debug
        // log in `dispatch`.
        let tool_use_id = context.tool_call_id.clone();
        let (progress_token, _progress_guard) = self.entry.client_context.progress.register(
            self.entry.server_name().to_string(),
            self.remote_tool_name.clone(),
            tool_use_id.clone(),
            Some(Arc::clone(&context.frontend)),
        );
        let mut meta = RequestMetaObject::new();
        meta.set_progress_token(progress_token);
        if let Some(id) = &tool_use_id {
            meta.0
                .insert("meka/toolUseId".to_string(), serde_json::json!(id));
        }
        // Lets a server scope per-session state (a cache, a workspace, a connection pool, an audit
        // trail) to the conversation the call came from. `_meta` is the spec's extension point and
        // already carries `meka/toolUseId`, so this adds no new wire contract.
        if let Some(session_id) = context.session_id {
            meta.0.insert(
                "meka/sessionId".to_string(),
                serde_json::json!(session_id.to_string()),
            );
        }
        params.meta = Some(meta);

        // Same error surface as an actually-closed transport. The upstream retry logic already
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
                let send = peer.notify_cancelled(CancelledNotificationParam::new(
                    Some(request_id),
                    Some(reason.to_string()),
                ));
                match tokio::time::timeout(CANCEL_NOTIFY_TIMEOUT, send).await {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        tracing::debug!(
                            "failed to send cancellation notification to '{server_name}': {error}"
                        );
                    }
                    Err(_) => {
                        tracing::debug!(
                            "cancellation notification to '{server_name}' timed out after {seconds}s",
                            seconds = CANCEL_NOTIFY_TIMEOUT.as_secs()
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

impl McpTool {
    /// Make the call, reconnecting and retrying once if the transport has closed under it.
    ///
    /// Cancellation and timeout both notify the server before returning, so it can stop work it
    /// would otherwise finish for nobody. An interrupt is [`MekaError::Interrupted`]; every other
    /// failure is [`MekaError::McpToolExecution`] naming the server and tool.
    pub(crate) async fn call(
        &self,
        arguments: Option<serde_json::Map<String, serde_json::Value>>,
        context: &CallContext,
    ) -> Result<rmcp::model::CallToolResult> {
        let params = {
            let mut p = CallToolRequestParams::new(self.remote_tool_name.clone());
            p.arguments = arguments;
            p
        };

        let is_timeout = |error: &ServiceError| matches!(error, ServiceError::Cancelled { reason: Some(reason) } if reason.starts_with("timed out"));

        // First attempt. On TransportClosed, reconnect and retry once.
        let result = match self.call_tool_once(params.clone(), context).await {
            Ok(result) => result,
            Err(ServiceError::Cancelled { reason })
                if reason.as_deref() == Some("user interrupt") =>
            {
                return Err(MekaError::Interrupted);
            }
            Err(error) if is_timeout(&error) => {
                return Err(MekaError::McpToolExecution {
                    server_name: self.entry.server_name().to_string(),
                    tool_name: self.remote_tool_name.clone(),
                    message: error.to_string(),
                });
            }
            Err(ServiceError::TransportClosed) => {
                self.entry.reconnect().await?;
                match self.call_tool_once(params, context).await {
                    Ok(result) => result,
                    Err(ServiceError::Cancelled { reason })
                        if reason.as_deref() == Some("user interrupt") =>
                    {
                        return Err(MekaError::Interrupted);
                    }
                    Err(error) => {
                        return Err(MekaError::McpToolExecution {
                            server_name: self.entry.server_name().to_string(),
                            tool_name: self.remote_tool_name.clone(),
                            message: error.to_string(),
                        });
                    }
                }
            }
            Err(error) => {
                // Matched on the text because the transport surfaces the status inside the error
                // rather than as a field, so a bare `Unauthorized` with no code still counts.
                let text = error.to_string().to_ascii_lowercase();
                if text.contains("401") || text.contains("unauthorized") {
                    let server_name = self.entry.server_name();
                    tracing::warn!(
                        "MCP server '{server_name}' rejected the call as unauthorized; run `meka mcp login {server_name}` to re-authenticate"
                    );
                }
                return Err(MekaError::McpToolExecution {
                    server_name: self.entry.server_name().to_string(),
                    tool_name: self.remote_tool_name.clone(),
                    message: error.to_string(),
                });
            }
        };

        Ok(result)
    }
}

/// Map MCP `CallToolResult.content` items to meka's provider-layer `ToolResultContent` blocks. Text
/// stays text; images pass through as multimodal blocks so providers like Claude and GPT-4o can see
/// them; audio, embedded resources, and resource links collapse to informative text placeholders
/// (no provider accepts them as tool-result blocks yet).
pub(crate) fn convert_tool_result_content(
    items: &[rmcp::model::ContentBlock],
) -> Vec<crate::conversation::ToolResultContent> {
    use crate::{conversation::ToolResultContent, image::ImageSource};

    // The one unbounded thing a server controls in a tool result. Images have had a ceiling since
    // they were added; text was appended until the server stopped sending, and every byte then went
    // into the session and back to the provider on every subsequent turn. Generous enough that no
    // real result reaches it: four megabytes is roughly a million tokens.
    //
    // Past it the excess is *dropped*, not spilled, which is a knowingly weaker guarantee than the
    // one `tools::shell` gives: an overflowing command writes every byte to a file and hands the
    // model its path. Two things stand in the way of matching it here. This function is a pure
    // transform over content blocks with no session to spill into, and
    // `scratchpad::persist_oversized_results` -- which would preserve the rest -- runs on the
    // `ToolOutput` *after* `execute` returns, so it never sees what was cut. Closing the gap means
    // threading the scratchpad through the conversion, and is worth doing when a real server is
    // observed hitting four megabytes; none has been. The loss is disclosed either way, which is
    // the property that actually matters: the model is told the result was cut rather than
    // answering from a silent truncation.
    const MAX_MCP_TEXT_BYTES: usize = 4 * crate::text::MIB;

    let mut blocks: Vec<ToolResultContent> = Vec::new();
    let mut text_buffer = String::new();
    let mut text_dropped: usize = 0;
    // Counted across the whole result, not per buffer.
    //
    // Compared against `text_buffer.len()` alone the ceiling misses what has already been flushed:
    // the accepted-image arm calls `flush_text`, which `mem::take`s the buffer. A result shaped
    // `[4 MiB text][small PNG][4 MiB text][PNG]...` therefore passed the guard on every round
    // and the cap bounded nothing: the resident total is the sum of the flushed blocks, which
    // is what the model is sent.
    let mut text_kept: usize = 0;

    let flush_text = |buffer: &mut String, out: &mut Vec<ToolResultContent>| {
        if !buffer.is_empty() {
            out.push(ToolResultContent::Text {
                text: std::mem::take(buffer),
            });
        }
    };

    for item in items {
        match item {
            rmcp::model::ContentBlock::Text(text_content) => {
                if text_kept >= MAX_MCP_TEXT_BYTES {
                    text_dropped += text_content.text.len();
                    continue;
                }
                if !text_buffer.is_empty() {
                    text_buffer.push('\n');
                    text_kept += 1;
                }
                let room = MAX_MCP_TEXT_BYTES - text_kept;
                if text_content.text.len() <= room {
                    text_buffer.push_str(&text_content.text);
                    text_kept += text_content.text.len();
                } else {
                    let cut = text_content.text.floor_char_boundary(room);
                    text_buffer.push_str(&text_content.text[..cut]);
                    text_kept += cut;
                    text_dropped += text_content.text.len() - cut;
                }
            }
            rmcp::model::ContentBlock::Image(image) => {
                // The server's declared `mime_type` is a hint; what gets forwarded is decided by
                // the bytes. Providers sniff and reject a mismatch with a 400, and that rejection
                // lands inside a `tool_result` already committed to the session, where it fails
                // every later request. Only the first few characters are decoded, so this costs
                // nothing on a multi-megabyte payload.
                let sniffed = match crate::image::classify_base64_prefix(&image.data) {
                    crate::image::ImageHandling::PassThrough(format) => Some(format.to_mime_type()),
                    // A format needing transcoding (TIFF, ICO, ...) would mean decoding and
                    // re-encoding the whole payload, which is not worth it for a server that
                    // mislabeled its own output.
                    _ => None,
                }
                .filter(|mime| {
                    ALLOWED_IMAGE_MIME_TYPES
                        .iter()
                        .any(|allowed| allowed.eq_ignore_ascii_case(mime))
                });
                // Two ceilings, and both matter. `MAX_MCP_IMAGE_BYTES` is meka's own memory and
                // quota guard. The second is the providers' limit, which every other image
                // producer gets for free by going through `prepare_image_payload`; this path
                // builds its `ImageSource` directly, so without it an MCP image between the two
                // ceilings is forwarded only for the provider to answer 400. Base64 carries 3
                // bytes per 4 characters, which is exact enough to compare against a cap.
                let decoded_len = image.data.len() / 4 * 3;
                let oversize = if image.data.len() > MAX_MCP_IMAGE_BYTES {
                    Some(format!(
                        "{} base64 bytes exceeds {} byte limit",
                        image.data.len(),
                        MAX_MCP_IMAGE_BYTES
                    ))
                } else if decoded_len > crate::image::MAX_IMAGE_RAW_BYTES {
                    Some(format!(
                        "~{} decoded bytes exceeds the {} byte ceiling providers accept",
                        decoded_len,
                        crate::image::MAX_IMAGE_RAW_BYTES
                    ))
                } else {
                    None
                };
                if let Some(reason) = oversize {
                    if !text_buffer.is_empty() {
                        text_buffer.push('\n');
                    }
                    text_buffer.push_str(&format!("[image suppressed: {reason}]"));
                } else if let Some(media_type) = sniffed {
                    flush_text(&mut text_buffer, &mut blocks);
                    blocks.push(ToolResultContent::Image {
                        source: ImageSource::Base64 {
                            media_type: media_type.to_string(),
                            data: image.data.clone(),
                        },
                    });
                } else {
                    if !text_buffer.is_empty() {
                        text_buffer.push('\n');
                    }
                    text_buffer.push_str(&format!(
                        "[image suppressed: declared '{}', but the bytes are not an allowed image \
                         format]",
                        image.mime_type
                    ));
                }
            }
            rmcp::model::ContentBlock::Audio(audio) => {
                if !text_buffer.is_empty() {
                    text_buffer.push('\n');
                }
                text_buffer.push_str(&format!(
                    "[audio content: {}, {} base64 bytes; meka does not yet pass audio to the provider]",
                    audio.mime_type,
                    audio.data.len()
                ));
            }
            rmcp::model::ContentBlock::Resource(resource) => {
                if !text_buffer.is_empty() {
                    text_buffer.push('\n');
                }
                match &resource.resource {
                    rmcp::model::ResourceContents::TextResourceContents { uri, text, .. } => {
                        text_buffer.push_str(&format!("--- {uri}\n{text}"));
                    }
                    rmcp::model::ResourceContents::BlobResourceContents {
                        uri,
                        mime_type,
                        blob,
                        ..
                    } => {
                        text_buffer.push_str(&format!(
                            "[embedded blob resource: {} ({}), {} base64 bytes]",
                            uri,
                            mime_type.as_deref().unwrap_or("application/octet-stream"),
                            blob.len()
                        ));
                    }
                    _ => text_buffer.push_str("[embedded resource omitted]"),
                }
            }
            rmcp::model::ContentBlock::ResourceLink(link) => {
                if !text_buffer.is_empty() {
                    text_buffer.push('\n');
                }
                text_buffer.push_str(&format!("[resource link: {}]", link.uri));
            }
            // `ContentBlock` is non-exhaustive; a block kind this build doesn't recognize collapses
            // to a placeholder rather than being dropped silently.
            _ => {
                if !text_buffer.is_empty() {
                    text_buffer.push('\n');
                }
                text_buffer.push_str("[unsupported content omitted]");
            }
        }
    }

    if text_dropped > 0 {
        if !text_buffer.is_empty() {
            text_buffer.push('\n');
        }
        text_buffer.push_str(&format!(
            "\n... ({text_dropped} further bytes of text were dropped; this server's result exceeded the {MAX_MCP_TEXT_BYTES} \
             byte ceiling)"
        ));
    }
    flush_text(&mut text_buffer, &mut blocks);
    if blocks.is_empty() {
        blocks.push(ToolResultContent::Text {
            text: String::new(),
        });
    }
    blocks
}

#[cfg(test)]
mod tests {

    /// The variable takes a duration string; a bare number and a zero both fall to the default
    /// rather than to a timeout nobody asked for.
    #[test]
    fn the_tool_timeout_variable_takes_a_duration_and_nothing_else() {
        let default = std::time::Duration::from_secs(600);
        assert_eq!(parse_tool_call_timeout(None), default);
        assert_eq!(
            parse_tool_call_timeout(Some("90s")),
            std::time::Duration::from_secs(90)
        );
        assert_eq!(
            parse_tool_call_timeout(Some(" 2m ")),
            std::time::Duration::from_secs(120)
        );
        assert_eq!(parse_tool_call_timeout(Some("600000")), default);
        assert_eq!(parse_tool_call_timeout(Some("0s")), default);
        assert_eq!(parse_tool_call_timeout(Some("soon")), default);
    }
    use base64::Engine as _;

    use super::*;

    #[test]
    fn decline_maps_to_decline_action() {
        match ElicitationResponse::Decline.into_result().action {
            ElicitationAction::Decline => {}
            other => panic!("expected Decline, got {other:?}"),
        }
    }

    #[test]
    fn an_accept_carries_its_content_into_the_result() {
        let content = serde_json::json!({"k": "v"});
        let result = ElicitationResponse::Accept {
            content: Some(content.clone()),
        }
        .into_result();
        assert!(matches!(result.action, ElicitationAction::Accept));
        assert_eq!(result.content, Some(content));
    }

    /// Base64 of a real 4x4 image in `format`. The handler classifies from the bytes, so a
    /// placeholder string is not a usable fixture for the accept path.
    fn base64_image(format: image::ImageFormat) -> String {
        let mut bytes = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut bytes);
        // JPEG has no alpha channel, so it needs an RGB source.
        if format == image::ImageFormat::Jpeg {
            image::RgbImage::from_pixel(4, 4, image::Rgb([128, 64, 200]))
                .write_to(&mut cursor, format)
                .expect("encode");
        } else {
            image::RgbaImage::from_pixel(4, 4, image::Rgba([128, 64, 200, 255]))
                .write_to(&mut cursor, format)
                .expect("encode");
        }
        base64::engine::general_purpose::STANDARD.encode(&bytes)
    }

    #[test]
    fn convert_tool_result_content_text_only() {
        use rmcp::model::ContentBlock;
        let items = vec![ContentBlock::text("hello"), ContentBlock::text("world")];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            crate::conversation::ToolResultContent::Text { text } => {
                assert_eq!(text, "hello\nworld");
            }
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn convert_tool_result_content_image_passthrough() {
        use rmcp::model::ContentBlock;
        let data = base64_image(image::ImageFormat::Png);
        let items = vec![ContentBlock::image(data.clone(), "image/png")];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            crate::conversation::ToolResultContent::Image { source } => {
                assert_eq!(source.media_type(), "image/png");
                assert_eq!(source.base64_data(), Some(data.as_str()));
            }
            other => panic!("expected Image, got {other:?}"),
        }
    }

    /// The bug this whole path guards: a server that declares one format and sends another. The
    /// declared type must not reach the provider, which sniffs and answers 400.
    #[test]
    fn convert_tool_result_content_image_media_type_comes_from_bytes() {
        use rmcp::model::ContentBlock;
        let items = vec![ContentBlock::image(
            base64_image(image::ImageFormat::Jpeg),
            "image/png",
        )];
        let blocks = convert_tool_result_content(&items);
        match &blocks[0] {
            crate::conversation::ToolResultContent::Image { source } => {
                assert_eq!(source.media_type(), "image/jpeg");
            }
            other => panic!("expected Image, got {other:?}"),
        }
    }

    /// The allow-list applies to what the bytes actually are. BMP decodes fine but isn't a format
    /// we forward, so a real BMP is suppressed no matter what the server called it.
    #[test]
    fn convert_tool_result_content_image_rejects_disallowed_format() {
        use rmcp::model::ContentBlock;
        let items = vec![ContentBlock::image(
            base64_image(image::ImageFormat::Bmp),
            "image/png",
        )];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            crate::conversation::ToolResultContent::Text { text } => {
                assert!(text.contains("image suppressed"), "{}", text);
            }
            other => panic!("expected Text placeholder, got {other:?}"),
        }
    }

    #[test]
    fn convert_tool_result_content_image_rejects_non_image_bytes() {
        use rmcp::model::ContentBlock;
        let items = vec![ContentBlock::image("BASE64DATA", "image/svg+xml")];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            crate::conversation::ToolResultContent::Text { text } => {
                assert!(text.contains("image suppressed"));
                assert!(text.contains("image/svg+xml"));
            }
            other => panic!("expected Text placeholder, got {other:?}"),
        }
    }

    #[test]
    fn convert_tool_result_content_image_rejects_oversize() {
        use rmcp::model::ContentBlock;
        let oversized = "X".repeat(MAX_MCP_IMAGE_BYTES + 1);
        let items = vec![ContentBlock::image(oversized, "image/png")];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            crate::conversation::ToolResultContent::Text { text } => {
                assert!(text.contains("image suppressed"));
                assert!(text.contains("exceeds"));
            }
            other => panic!("expected Text placeholder, got {other:?}"),
        }
    }

    /// Between meka's own memory guard and the ceiling providers accept lies a band where an MCP
    /// image would be forwarded purely so the provider could answer 400.
    #[test]
    fn convert_tool_result_content_image_rejects_over_the_provider_ceiling() {
        use rmcp::model::ContentBlock;
        // Comfortably over the provider ceiling, comfortably under meka's own cap.
        let base64_len = crate::image::MAX_IMAGE_RAW_BYTES / 3 * 4 + 4096;
        assert!(base64_len < MAX_MCP_IMAGE_BYTES, "must sit between the two");
        let items = vec![ContentBlock::image("A".repeat(base64_len), "image/png")];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            crate::conversation::ToolResultContent::Text { text } => {
                assert!(text.contains("image suppressed"), "{}", text);
                assert!(text.contains("providers accept"), "{}", text);
            }
            other => panic!("expected Text placeholder, got {other:?}"),
        }
    }

    #[test]
    fn convert_tool_result_content_mixed_keeps_ordering() {
        use rmcp::model::ContentBlock;
        let items = vec![
            ContentBlock::text("before"),
            ContentBlock::image(base64_image(image::ImageFormat::Png), "image/png"),
            ContentBlock::text("after"),
        ];
        let blocks = convert_tool_result_content(&items);
        assert_eq!(blocks.len(), 3);
        assert!(matches!(
            blocks[0],
            crate::conversation::ToolResultContent::Text { .. }
        ));
        assert!(matches!(
            blocks[1],
            crate::conversation::ToolResultContent::Image { .. }
        ));
        assert!(matches!(
            blocks[2],
            crate::conversation::ToolResultContent::Text { .. }
        ));
    }

    /// The `initialize` request is meka's, not the SDK's: the version is pinned here rather than
    /// floating with rmcp, the client is named, and elicitation is declared so a server may use it.
    #[test]
    fn the_client_introduces_itself_and_declares_elicitation() {
        use rmcp::model::ProtocolVersion;
        let handler = MekaClientHandler::new("srv".to_string(), McpClientContext::new());
        let info = handler.get_info();
        assert_eq!(info.protocol_version, ProtocolVersion::V_2025_11_25);
        assert_eq!(info.client_info.name, "meka");
        assert_eq!(info.client_info.version, env!("CARGO_PKG_VERSION"));
        let elicitation = info
            .capabilities
            .elicitation
            .expect("elicitation is declared");
        assert!(elicitation.form.is_some() && elicitation.url.is_some());
    }
}
