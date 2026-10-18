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
    ClientLifecycleMode, ErrorData as McpError, Peer, RoleClient,
    handler::client::ClientHandler,
    model::{
        CallToolRequest, CallToolRequestParams, CancelledNotificationParam, ClientRequest,
        ConstString, CustomNotification, DEFAULT_MRTR_MAX_ROUNDS, ElicitRequestParams,
        ElicitResult, ElicitationAction, ElicitationResponseNotificationMethod, InputRequest,
        InputRequests, InputRequiredResult, InputResponses, ProgressNotificationParam,
        ProtocolVersion, RequestMetaObject, ServerResult,
    },
    service::{NotificationContext, PeerRequestOptions, RequestContext, ServiceError},
};

use super::{ALLOWED_IMAGE_MIME_TYPES, MAX_MCP_IMAGE_BYTES, McpClientContext, ServerEntry};
use crate::{
    error::{MekaError, Result},
    frontend::{ElicitationResponse, Frontend},
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

/// The protocol version the probe asks for first: the one with `server/discover`, per-request
/// `_meta` and `subscriptions/listen`.
pub(super) const MODERN_PROTOCOL_VERSION: ProtocolVersion = ProtocolVersion::V_2026_07_28;

/// The version `initialize` names when the probe finds a server from before the handshake went:
/// the newest one that still has an `initialize`.
pub(super) const LEGACY_PROTOCOL_VERSION: ProtocolVersion = ProtocolVersion::V_2025_11_25;

/// How an HTTP connect opens: probe with `server/discover` and fall back to `initialize` on the
/// same connection, the dual-era client the spec's compatibility matrix describes. A server of
/// either era is served, and the version a connection speaks is the server's answer rather than
/// meka's assumption. An HTTP server answers an unknown method one way or another, so the
/// fallback can ride the probe's connection; a stdio server may not, see [`stdio_probe`].
pub(super) fn http_lifecycle() -> ClientLifecycleMode {
    ClientLifecycleMode::Auto {
        preferred_versions: vec![MODERN_PROTOCOL_VERSION],
        legacy_version: Some(LEGACY_PROTOCOL_VERSION),
    }
}

/// How a stdio connect probes: `server/discover` alone, on a child of its own, which the connector
/// keeps when the child answers as a modern server and discards otherwise. The legacy handshake
/// gets a fresh child, because a server from before the probe existed may swallow a method it does
/// not know and answer nothing afterwards, `initialize` included.
pub(super) fn stdio_probe() -> ClientLifecycleMode {
    ClientLifecycleMode::Discover {
        preferred_versions: vec![MODERN_PROTOCOL_VERSION],
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
    /// What this client says about itself: on `initialize` to a legacy server, and in every
    /// request's `_meta` to a modern one. Left to rmcp's default, it names the SDK as the client,
    /// floats the protocol version with the SDK's release, and declares no capabilities at all, so
    /// a server that checks before it elicits never does. The version here is the one a bare
    /// `initialize` names; [`http_lifecycle`] and [`stdio_probe`] decide which era a connection
    /// ends in.
    fn get_info(&self) -> rmcp::model::ClientConfig {
        use rmcp::model::{
            ClientCapabilities, ElicitationCapability, FormElicitationCapability, Implementation,
            UrlElicitationCapability,
        };
        let mut capabilities = ClientCapabilities::default();
        capabilities.elicitation = Some(
            ElicitationCapability::new()
                .with_form(FormElicitationCapability::new())
                .with_url(UrlElicitationCapability::new()),
        );
        rmcp::model::ClientConfig::new(
            capabilities,
            Implementation::new("meka", env!("CARGO_PKG_VERSION")),
        )
        .with_protocol_version(LEGACY_PROTOCOL_VERSION)
    }

    fn on_tool_list_changed(
        &self,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        let server = Arc::clone(&self.server_name);
        let context = Arc::clone(&self.context);
        async move { refresh_tools_after_list_changed(&server, &context).await }
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
        async move { record_resource_update(&server, &context, &params.uri) }
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
        let server = Arc::clone(&self.server_name);
        let context = Arc::clone(&self.context);
        async move {
            context.progress.dispatch(server.as_ref(), params).await;
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
            // The frontend is the in-flight call's, found through the progress registry. With no
            // call from `server` in flight (the server elicited outside a tool call, or the
            // progress guard already dropped) there is no human to ask, and with calls from two
            // sessions in flight there is no telling which human, so declining is the answer in
            // both cases rather than a guess that lands the prompt in the wrong session.
            let frontend = match context.progress.elicitation_route(server.as_ref()) {
                crate::mcp::progress::ElicitationRoute::To(frontend) => frontend,
                crate::mcp::progress::ElicitationRoute::Nobody => {
                    tracing::warn!(
                        "MCP server '{server}' requested elicitation with no tool call in flight; \
                         declining"
                    );
                    return Ok(ElicitationResponse::Decline.into_result());
                }
                crate::mcp::progress::ElicitationRoute::Ambiguous => {
                    tracing::warn!(
                        "MCP server '{server}' requested elicitation while calls from more than \
                         one session are in flight; declining"
                    );
                    return Ok(ElicitationResponse::Decline.into_result());
                }
            };
            Ok(elicit(server.as_ref(), frontend.as_ref(), request).await)
        }
    }
}

/// Re-list a server's tools after it said they changed, whichever way it said so: a legacy
/// server's unsolicited notification reaches [`ClientHandler::on_tool_list_changed`], a modern
/// one's arrives on the listen stream `subscriptions.rs` drains, and both end here.
pub(super) async fn refresh_tools_after_list_changed(
    server_name: &str,
    context: &McpClientContext,
) {
    tracing::debug!("MCP server '{server_name}' sent tools/list_changed");
    let Some(manager) = context.manager().and_then(|weak| weak.upgrade()) else {
        tracing::debug!("tool list refresh skipped: manager not yet wired for '{server_name}'");
        return;
    };

    // Tool-permission resolution reads the server config and `mcp_default_permission` from the
    // manager itself; no explicit permission needs to be threaded here.
    match manager.discover_tools_for_server(server_name).await {
        Ok(adapters) => {
            // The same door the initial registration goes through, so a refresh cannot classify a
            // tool differently from the listing it replaces. Routes through every attached
            // registry so all active sessions observe the updated tool set.
            manager.register_server_tools(server_name, adapters).await;
            tracing::info!("MCP server '{server_name}' tool registry refreshed");
        }
        Err(error) => {
            tracing::warn!("failed to refresh tools for MCP server '{server_name}': {error}");
        }
    }
}

/// Record that `uri` on `server_name` changed, for `mcp_resource_updates_list`; the counterpart
/// of [`refresh_tools_after_list_changed`] for `resources/updated`, reached from both eras too.
pub(super) fn record_resource_update(server_name: &str, context: &McpClientContext, uri: &str) {
    tracing::info!("MCP server '{server_name}' reported resource updated: {uri}");
    context.resource_updates.record(server_name, uri);
}

/// Put a server's elicitation to `frontend` and answer with what the user said, or a decline when
/// the user did not answer in time.
///
/// One body for both eras: a legacy server sends `elicitation/create` as a request of its own,
/// which [`ClientHandler::create_elicitation`] routes here, and a modern one embeds it in an
/// `input_required` result, which [`fulfill_input_requests`] routes here.
pub(super) async fn elicit(
    server: &str,
    frontend: &dyn Frontend,
    request: ElicitRequestParams,
) -> ElicitResult {
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
        server_name: server.to_string(),
        kind,
        message,
    };

    // Bounded so an unanswered prompt cannot stall an MCP tool call forever; an approval prompt
    // waits longer because the turn is already paused on it. An elicitation is an MCP *request*,
    // so a `Decline` is how the server learns the user did not answer; `notifications/cancelled`
    // is for requests meka started.
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

    response.into_result()
}

/// Answer the input requests an `input_required` result carried, for the retry that follows.
///
/// Only an elicitation can be answered, and only when the call has a frontend to put it to. meka
/// declares no sampling or roots capability, and a server may not ask for what the client did not
/// declare, so one that does has left the protocol and the call fails naming what it asked rather
/// than retrying with a hole the server would ask to fill again.
pub(super) async fn fulfill_input_requests(
    server: &str,
    frontend: Option<&Arc<dyn Frontend>>,
    requests: InputRequests,
) -> std::result::Result<InputResponses, ServiceError> {
    let mut responses = InputResponses::new();
    for (key, request) in requests {
        let asked = match request {
            InputRequest::Elicitation(request) => match frontend {
                Some(frontend) => elicit(server, frontend.as_ref(), request.params).await,
                None => {
                    tracing::warn!(
                        "MCP server '{server}' requested elicitation on a call with no prompt to \
                         show it in; declining"
                    );
                    ElicitationResponse::Decline.into_result()
                }
            },
            InputRequest::CreateMessage(_) => return Err(undeclared_input(server, "sampling")),
            InputRequest::ListRoots(_) => return Err(undeclared_input(server, "roots")),
            _ => return Err(undeclared_input(server, "an input of an unknown kind")),
        };
        let answer = serde_json::to_value(asked).map_err(|error| {
            ServiceError::McpError(McpError::internal_error(
                format!("failed to serialize the elicitation's answer: {error}"),
                None,
            ))
        })?;
        responses.insert(key, answer);
    }
    Ok(responses)
}

fn undeclared_input(server: &str, what: &str) -> ServiceError {
    ServiceError::McpError(McpError::invalid_request(
        format!("MCP server '{server}' asked for {what}, which this client does not declare"),
        None,
    ))
}

/// A round that carried no request is the server waiting on something out of band, a URL the
/// user is still at; retrying at once would spend the round cap in a millisecond.
const STATE_ONLY_ROUND_PAUSE: std::time::Duration = std::time::Duration::from_millis(250);

/// The rounds one request goes through on a modern server: each `input_required` answer is
/// fulfilled through the frontend and the request sent again with the answers and the server's
/// opaque `requestState`, until a result arrives or [`DEFAULT_MRTR_MAX_ROUNDS`] requests have
/// gone. A legacy server, which never answers `input_required`, costs one round.
///
/// The caller owns the request and the sending: it takes [`Self::carry`] into the request before
/// each send and hands an `input_required` answer to [`Self::absorb`] after it.
pub(super) struct Rounds<'a> {
    server: &'a str,
    frontend: Option<&'a Arc<dyn Frontend>>,
    input_responses: Option<InputResponses>,
    request_state: Option<String>,
    sent: usize,
}

impl<'a> Rounds<'a> {
    pub(super) fn new(server: &'a str, frontend: Option<&'a Arc<dyn Frontend>>) -> Self {
        Self {
            server,
            frontend,
            input_responses: None,
            request_state: None,
            sent: 0,
        }
    }

    /// What the next send carries: the answers to the last round's requests, and the server's
    /// state exactly as it gave it.
    pub(super) fn carry(&mut self) -> (Option<InputResponses>, Option<String>) {
        self.sent += 1;
        (self.input_responses.take(), self.request_state.take())
    }

    /// Take in an `input_required` answer so the next [`Self::carry`] has what the retry needs.
    /// An error when the server asked for what meka cannot give, sent an answer a retry could
    /// carry nothing of, or has asked on every round the cap allows.
    pub(super) async fn absorb(
        &mut self,
        result: InputRequiredResult,
    ) -> std::result::Result<(), ServiceError> {
        if self.sent >= DEFAULT_MRTR_MAX_ROUNDS {
            return Err(ServiceError::InputRequiredRoundsExceeded {
                max_rounds: DEFAULT_MRTR_MAX_ROUNDS,
            });
        }
        // rmcp refuses to parse a result carrying neither, so one with no requests has state.
        let requests = result.input_requests.unwrap_or_default();
        if requests.is_empty() {
            tokio::time::sleep(STATE_ONLY_ROUND_PAUSE).await;
        } else {
            self.input_responses =
                Some(fulfill_input_requests(self.server, self.frontend, requests).await?);
        }
        self.request_state = result.request_state;
        Ok(())
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

    /// One attempt at the call on the connection as it stands: every round the server asks for,
    /// under one deadline. A closed transport is the caller's to reopen.
    async fn call_tool_once(
        &self,
        mut params: CallToolRequestParams,
        context: &CallContext,
    ) -> std::result::Result<rmcp::model::CallToolResult, ServiceError> {
        let mut meta = RequestMetaObject::new();
        if let Some(id) = &context.tool_call_id {
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

        // `call` has already refused a server that is not connected in the state's own words; a
        // state that changed between there and here is a transport that closed, which is what
        // the retry in `call` reconnects.
        let peer: Peer<RoleClient> = self
            .entry
            .require_connected()
            .await
            .map_err(|_| ServiceError::TransportClosed)?;

        // One deadline for every round: a server that asks and asks again is still one tool call
        // to the turn, and the model's wait is what the timeout bounds.
        let deadline = tokio::time::Instant::now() + self.entry.tool_call_timeout;
        let mut rounds = Rounds::new(self.entry.server_name(), Some(&context.frontend));
        loop {
            (params.input_responses, params.request_state) = rounds.carry();
            let request = ClientRequest::CallToolRequest(CallToolRequest::new(params.clone()));
            match self.send_bounded(&peer, request, context, deadline).await? {
                ServerResult::CallToolResult(result) => return Ok(result),
                // The wait for the user's answer is the call's too: an interrupt or the deadline
                // ends it, and the asker's end dropping is what withdraws the question from the
                // screen.
                ServerResult::InputRequiredResult(result) => tokio::select! {
                    biased;
                    _ = context.cancellation.cancelled() => {
                        return Err(ServiceError::Cancelled {
                            reason: Some("user interrupt".to_string()),
                        });
                    }
                    _ = tokio::time::sleep_until(deadline) => {
                        return Err(ServiceError::Cancelled {
                            reason: Some(format!(
                                "timed out after {}s",
                                self.entry.tool_call_timeout.as_secs()
                            )),
                        });
                    }
                    absorbed = rounds.absorb(result) => absorbed?,
                },
                _ => return Err(ServiceError::UnexpectedResponse),
            }
        }
    }

    /// Send one request and wait for its answer under the call's cancellation and `deadline`;
    /// whichever of the two fires first tells the server to stop before this returns.
    async fn send_bounded(
        &self,
        peer: &Peer<RoleClient>,
        request: ClientRequest,
        context: &CallContext,
        deadline: tokio::time::Instant,
    ) -> std::result::Result<ServerResult, ServiceError> {
        let handle = peer
            .send_cancellable_request(request, PeerRequestOptions::no_options())
            .await?;
        let request_id = handle.id.clone();
        // The call's entry in the progress registry, under the token rmcp minted for the request:
        // rmcp stamps its own `progressToken` over anything `_meta` carried, so that token is the
        // one the server echoes and the only one a notification can be found by, and it exists
        // only once the request is sent. The frontend is the calling session's, kept on the entry
        // because the rmcp notification handler runs on a separately-spawned task (see
        // `rmcp::service::spawn_service_task`) and cannot read the caller's task-local.
        let _progress_guard = self.entry.client_context.progress.register(
            &handle.progress_token,
            self.entry.server_name().to_string(),
            self.remote_tool_name.clone(),
            context.tool_call_id.clone(),
            Some(Arc::clone(&context.frontend)),
        );

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
            response = handle.await_response() => response,
            _ = context.cancellation.cancelled() => {
                notify_cancel("user interrupt").await;
                Err(ServiceError::Cancelled {
                    reason: Some("user interrupt".to_string()),
                })
            }
            _ = tokio::time::sleep_until(deadline) => {
                notify_cancel("timeout").await;
                Err(ServiceError::Cancelled {
                    reason: Some(format!(
                        "timed out after {}s",
                        self.entry.tool_call_timeout.as_secs()
                    )),
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

        // Refused in the state's own words when the server is not connected: a server still
        // connecting, disabled or refused is not a closed transport, and the reconnect the retry
        // below makes for one is a no-op for these that would then report "transport closed" for
        // a server somebody shut down.
        self.entry.require_connected().await?;

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
                    message: crate::text::redact_urls(&error.to_string()),
                });
            }
            Err(ServiceError::TransportClosed) => {
                self.entry.reconnect(&context.cancellation).await?;
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
                            message: crate::text::redact_urls(&error.to_string()),
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
                    message: crate::text::redact_urls(&error.to_string()),
                });
            }
        };

        Ok(result)
    }
}

/// Map MCP `CallToolResult.content` items to meka's provider-layer `ToolResultContent` blocks. Text
/// stays text and images pass through as image blocks; audio, embedded resources and resource
/// links collapse to text placeholders, because no provider accepts them as tool-result blocks.
pub(crate) fn convert_tool_result_content(
    items: &[rmcp::model::ContentBlock],
) -> Vec<crate::conversation::ToolResultContent> {
    use crate::{conversation::ToolResultContent, image::ImageSource};

    // The one unbounded thing a server controls in a tool result: every byte of text goes into
    // the session and back to the provider on every later turn. Generous enough that no real
    // result reaches it; four megabytes is roughly a million tokens.
    //
    // Past it the excess is *dropped*, not spilled, which is a weaker guarantee than the one
    // `tools::shell` gives. This function is a pure transform over content blocks with no session
    // to spill into, and `scratchpad::persist_oversized_results` runs on the `ToolOutput` *after*
    // `execute` returns, so it never sees what was cut. Threading the scratchpad through the
    // conversion is worth doing once a real server is observed hitting four megabytes. The loss
    // is disclosed either way, so the model never answers from a silent truncation.
    const MAX_MCP_TEXT_BYTES: usize = 4 * crate::text::MIB;

    let mut blocks: Vec<ToolResultContent> = Vec::new();
    let mut text_buffer = String::new();
    let mut text_dropped: usize = 0;
    // Counted across the whole result, not per buffer: the accepted-image arm calls `flush_text`,
    // which `mem::take`s the buffer, so a ceiling compared against `text_buffer.len()` alone lets
    // a result shaped `[4 MiB text][small PNG][4 MiB text][PNG]...` pass on every round. The
    // resident total is the sum of the flushed blocks, which is what the model is sent.
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

    fn input_required(value: serde_json::Value) -> InputRequiredResult {
        serde_json::from_value(value).expect("an input_required result")
    }

    /// A round carrying only state is retried with the state and nothing else; one asking for what
    /// meka does not declare fails the call; an elicitation on a call with no frontend is
    /// declined, not failed; and the cap counts requests sent.
    #[tokio::test]
    async fn rounds_carry_state_refuse_undeclared_asks_and_stop_at_the_cap() {
        let mut rounds = Rounds::new("srv", None);
        assert_eq!(rounds.carry(), (None, None));
        rounds
            .absorb(input_required(serde_json::json!({
                "resultType": "input_required", "requestState": "s1"
            })))
            .await
            .expect("a state-only round is retried");
        assert_eq!(rounds.carry(), (None, Some("s1".to_string())));

        let sampling = input_required(serde_json::json!({
            "resultType": "input_required",
            "inputRequests": {"k": {
                "method": "sampling/createMessage",
                "params": {"messages": [], "maxTokens": 1}
            }}
        }));
        let error = rounds
            .absorb(sampling)
            .await
            .expect_err("sampling is not declared");
        assert!(error.to_string().contains("sampling"), "{error}");
        let roots = input_required(serde_json::json!({
            "resultType": "input_required",
            "inputRequests": {"k": {"method": "roots/list", "params": {}}}
        }));
        let error = rounds
            .absorb(roots)
            .await
            .expect_err("roots are not declared");
        assert!(error.to_string().contains("roots"), "{error}");

        let elicitation = input_required(serde_json::json!({
            "resultType": "input_required",
            "inputRequests": {"q": {
                "method": "elicitation/create",
                "params": {
                    "mode": "form", "message": "?",
                    "requestedSchema": {"type": "object", "properties": {}}
                }
            }}
        }));
        rounds
            .absorb(elicitation)
            .await
            .expect("with no one to ask, the elicitation is declined rather than failed");
        let (answers, state) = rounds.carry();
        assert_eq!(
            answers.expect("an answer per request")["q"]["action"],
            "decline"
        );
        assert_eq!(state, None);

        let mut capped = Rounds::new("srv", None);
        for _ in 0..DEFAULT_MRTR_MAX_ROUNDS {
            capped.carry();
        }
        let again = input_required(serde_json::json!({
            "resultType": "input_required", "requestState": "s"
        }));
        assert!(matches!(
            capped.absorb(again).await,
            Err(ServiceError::InputRequiredRoundsExceeded { .. })
        ));
    }

    /// A frontend that keeps the last prompt it was shown and accepts it.
    #[derive(Default)]
    struct CapturingFrontend(std::sync::Mutex<Option<crate::frontend::ElicitationPrompt>>);

    impl CapturingFrontend {
        fn shown(&self) -> crate::frontend::ElicitationPrompt {
            crate::sync::lock(&self.0)
                .take()
                .expect("a prompt was put to the frontend")
        }
    }

    #[async_trait::async_trait]
    impl Frontend for CapturingFrontend {
        async fn emit(&self, _event: crate::frontend::FrontendEvent) {}

        async fn request_permission(
            &self,
            _request: crate::frontend::PermissionRequest,
        ) -> crate::frontend::PermissionOutcome {
            crate::frontend::PermissionOutcome::Allow
        }

        async fn handle_elicitation(
            &self,
            prompt: crate::frontend::ElicitationPrompt,
        ) -> ElicitationResponse {
            *crate::sync::lock(&self.0) = Some(prompt);
            ElicitationResponse::Accept { content: None }
        }
    }

    /// A form elicitation reaches the frontend as the form it is, message and schema intact, and
    /// a URL one as the URL to open; neither as the placeholder an unknown kind gets.
    #[tokio::test]
    async fn an_elicitation_reaches_the_frontend_as_the_kind_it_is() {
        use crate::frontend::ElicitationKind;

        let frontend = CapturingFrontend::default();
        let form: ElicitRequestParams = serde_json::from_value(serde_json::json!({
            "mode": "form", "message": "Name?",
            "requestedSchema": {"type": "object", "properties": {"name": {"type": "string"}}}
        }))
        .expect("a form request");
        elicit("srv", &frontend, form).await;
        let prompt = frontend.shown();
        assert_eq!(prompt.server_name, "srv");
        assert_eq!(prompt.message, "Name?");
        match prompt.kind {
            ElicitationKind::Form { schema } => {
                assert!(schema["properties"]["name"].is_object(), "{schema}");
            }
            other => panic!("a form was expected: {other:?}"),
        }

        let url: ElicitRequestParams = serde_json::from_value(serde_json::json!({
            "mode": "url", "message": "Sign in", "url": "https://example.test/login",
            "elicitationId": "e1"
        }))
        .expect("a url request");
        elicit("srv", &frontend, url).await;
        let prompt = frontend.shown();
        assert_eq!(prompt.message, "Sign in");
        assert!(
            matches!(prompt.kind, ElicitationKind::Url { ref url } if url == "https://example.test/login"),
            "{:?}",
            prompt.kind
        );
    }
}
