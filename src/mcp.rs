//! Model Context Protocol (MCP) client integration. Manages the lifecycle of configured MCP servers
//! (stdio child processes or streamable HTTP), publishes their tools to whoever subscribes (the
//! registries, through `crate::tools::mcp_adapter`), and handles OAuth/JWT authentication for HTTP
//! transports.

pub(crate) mod auth;
pub(crate) mod connector;
pub(crate) mod expand;
pub(crate) mod handler;
pub(crate) mod policy;
pub(crate) mod progress;
pub(crate) mod resource_updates;
pub(crate) mod sanitize;
pub(crate) mod subscriptions;
pub(crate) mod transport;

use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, Weak},
};

pub(crate) use handler::{CallContext, McpTool};
use policy::*;
pub(crate) use policy::{
    AdvertisedTool, PermissionSource, resolve_tool_permission, tool_exposure, tool_is_allowed,
    warn_on_stale_tool_config,
};
use rmcp::{
    Peer, RoleClient,
    model::{
        GetPromptRequestParams, GetPromptResponse, GetPromptResult, Prompt, ProtocolVersion,
        ReadResourceRequestParams, ReadResourceResponse, ReadResourceResult, Resource,
    },
    service::ServiceError,
};
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;

use crate::{
    config::{McpServerConfig, McpTransport},
    error::{MekaError, Result},
    permission::Permission,
    store::TokenStore,
};

/// Cap MCP-provided text (tool descriptions, resource/prompt descriptions) to this many characters
/// so a chatty server can't blow up the system prompt. Mirrors Claude Code's
/// `MAX_MCP_DESCRIPTION_LENGTH`.
pub(crate) const MAX_MCP_DESCRIPTION_CHARS: usize = 2048;

/// Cap on base64 payload size for an MCP image tool-result block. A server returning a giant image
/// would otherwise be cloned verbatim, forwarded to the provider, billed against the user's API
/// quota, and risk OOM. Mirrors the 10 MiB body cap on `web_fetch`.
pub(crate) const MAX_MCP_IMAGE_BYTES: usize = 10 * crate::text::MIB;

/// Tools one MCP server may advertise before the list is cut.
///
/// `list_all_tools` pages until the server stops offering a cursor, so a server that keeps offering
/// one keeps meka reading, and every tool it returns costs a `ToolDefinition` resident for the
/// session plus a line in the catalog the model reads on every turn. The cap is far above any
/// real server (the largest published ones advertise dozens) and exists so the ceiling belongs to
/// meka rather than to whatever is on the other end of the socket.
pub(crate) const MAX_MCP_TOOLS_PER_SERVER: usize = 512;

/// Keep at most [`MAX_MCP_TOOLS_PER_SERVER`] of what a server advertised, warning when it bites.
///
/// A free function rather than an inline block so the bound is assertable: reaching it through
/// `list_tools_bounded` needs a live server, so raising the constant to `usize::MAX` would leave
/// every suite green. The tool list is held per session and re-sent in every request's tools
/// array, so an unbounded one is resident cost on every turn, not just at connect.
fn cap_advertised_tools<T>(listed: Vec<T>, server_name: &str) -> Vec<T> {
    if listed.len() > MAX_MCP_TOOLS_PER_SERVER {
        tracing::warn!(
            "MCP server '{server_name}' advertised {listed} tools; keeping the first {MAX_MCP_TOOLS_PER_SERVER}",
            listed = listed.len()
        );
        return listed.into_iter().take(MAX_MCP_TOOLS_PER_SERVER).collect();
    }
    listed
}

/// Bound on an MCP request made outside the connector, when no configured timeout is available.
///
/// Matches `[mcp].connect_timeout`'s own default, so a manager that never started a
/// connector behaves like one that did rather than waiting forever.
pub(crate) const DEFAULT_MCP_REQUEST_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(30);

/// Allow-list of image MIME types passed straight through to the provider. Anything else (notably
/// `image/svg+xml`, which can embed script/link elements) is converted to a text placeholder.
pub(crate) const ALLOWED_IMAGE_MIME_TYPES: &[&str] =
    &["image/png", "image/jpeg", "image/gif", "image/webp"];

pub(crate) type McpRunningService =
    rmcp::service::RunningService<RoleClient, handler::MekaClientHandler>;

/// Total wall-clock budget for closing every MCP server on the way out. Serial teardown at up to
/// `CLOSE_TIMEOUT` per server would otherwise make exit latency scale with how many of them hang.
pub(crate) const SHUTDOWN_BUDGET: std::time::Duration = std::time::Duration::from_secs(10);

/// How long a shutdown waits for the connects in flight to unwind once it has ended them. Each
/// ends at its next await, and the child it was spawning dies with its transport, so this is the
/// scheduling of a few threads, plus at most one [`CLOSE_TIMEOUT`] for a connection that landed
/// as the stop fired and is closed rather than adopted.
const CONNECT_UNWIND_GRACE: std::time::Duration = std::time::Duration::from_secs(3);

/// How long `RunningService::close` is given: by a shutdown once the shared references are
/// released, and by [`ServerEntry::adopt_connection`] for a connection it refuses.
const CLOSE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// The connects in flight across one manager's servers, and the token that ends them.
///
/// A connect runs its future on a blocking thread, which nothing can drop from outside: at exit
/// the thread is detached and the process ends with the child it was spawning still owned by a
/// transport nobody dropped, so the child outlives meka. Every connect path selects on the token
/// and counts itself here, so a shutdown can end them all and know when the last has unwound.
pub(crate) struct Connects {
    stopping: CancellationToken,
    in_flight: tokio::sync::watch::Sender<usize>,
}

/// One connect in flight; dropping it is the connect ending, however it ended.
pub(crate) struct Connecting(Arc<Connects>);

impl Connects {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            stopping: CancellationToken::new(),
            in_flight: tokio::sync::watch::Sender::new(0),
        })
    }

    /// The token a connect in flight ends on.
    pub(crate) fn stopping(&self) -> &CancellationToken {
        &self.stopping
    }

    /// Count a connect that is starting, until the returned guard drops.
    pub(crate) fn connecting(self: &Arc<Self>) -> Connecting {
        self.in_flight.send_modify(|count| *count += 1);
        Connecting(Arc::clone(self))
    }

    /// End every connect in flight and wait up to `grace` for the last to unwind. `true` when
    /// none remains.
    pub(crate) async fn stop(&self, grace: std::time::Duration) -> bool {
        self.stopping.cancel();
        let mut in_flight = self.in_flight.subscribe();
        tokio::time::timeout(grace, in_flight.wait_for(|count| *count == 0))
            .await
            .is_ok_and(|waited| waited.is_ok())
    }
}

impl Drop for Connecting {
    fn drop(&mut self) {
        self.0
            .in_flight
            .send_modify(|count| *count = count.saturating_sub(1));
    }
}

/// Close `service` within [`CLOSE_TIMEOUT`]; what does not close in time is left to its drop
/// guard, and said so.
async fn close_service(server_name: &str, mut service: McpRunningService) {
    match service.close_with_timeout(CLOSE_TIMEOUT).await {
        Ok(Some(_)) => {}
        Ok(None) => {
            tracing::warn!("MCP server '{server_name}' shutdown timed out after {CLOSE_TIMEOUT:?}");
        }
        Err(error) => {
            tracing::warn!("failed to shut down MCP server '{server_name}': {error}");
        }
    }
}

pub(crate) struct McpClientManager {
    servers: HashMap<String, Arc<ServerEntry>>,
    /// The context every server's handler reports into; the ledgers a tool reads live on it.
    pub(crate) client_context: Arc<McpClientContext>,
    /// Global fallback permission from `[mcp].default_permission`. Consulted by
    /// `resolve_tool_permission` at tool-registration time when neither the server nor the user
    /// has configured a more specific permission and the server didn't advertise a
    /// `readOnlyHint`. `None` means "no user default": resolution falls through to the
    /// hardcoded strict `Unrestricted`.
    mcp_default_permission: Option<Permission>,
    /// `[mcp].deferral`, for the meta-tools, which belong to no server. Set once by the host after
    /// [`Self::prepare`]; left unset by a test or a `meka mcp` command, where it reads as the
    /// default.
    deferral: OnceLock<crate::config::Deferral>,
    /// Flipped to `true` by the background connector once every enabled entry has reached a
    /// terminal state (Connected or Failed). The turn gate watches this via
    /// [`Self::await_settled`] / [`Self::all_ready`].
    settled: tokio::sync::watch::Sender<bool>,
    /// Entries waiting to be connected by [`Self::start_connector`]. `None` once the connector has
    /// been started so a second call is a no-op; avoids re-spawning the connector if a test or the
    /// screen re-enters the same manager.
    pending_entries: std::sync::Mutex<Option<Vec<Arc<ServerEntry>>>>,
    /// Live snapshot of every connected server's currently-registered tools. The connector writes
    /// here as each server reaches `Connected`, and `on_tool_list_changed` writes here on dynamic
    /// updates. A new subscriber is replayed this snapshot at [`Self::subscribe`] time.
    tools_snapshot: tokio::sync::RwLock<HashMap<String, ServerTools>>,
    /// Who is told when a server's tool list changes. A session's registry subscribes when the
    /// session opens and unsubscribes when it closes; updates from the connector or notification
    /// handler reach every entry.
    observers: tokio::sync::RwLock<Vec<Arc<dyn ServerToolsObserver>>>,
    /// `[mcp].connect_timeout`, kept so the request paths that run *after* the connector
    /// (a `tools/list_changed` refresh, `meka mcp tools`) can bound themselves by the same value
    /// the connect did. Set by [`Self::start_connector`]; a manager that never started one
    /// (tests) falls back to [`DEFAULT_MCP_REQUEST_TIMEOUT`].
    connect_timeout: OnceLock<std::time::Duration>,
    /// The connects in flight on every entry, which [`Self::shutdown`] ends first.
    connects: Arc<Connects>,
}

/// What one server's latest `tools/list` produced: the tools, and which of them ship deferred.
///
/// Both halves in one value because they are one fact about one listing, and a registry given only
/// the first half is wrong in a way nothing reports: the tools arrive and every one of them looks
/// eager. Marks fanned out at discovery time reach only the registries attached then, which on
/// `meka serve` and `meka acp` is none of them (`start_connector` runs in `build_shared_deps`,
/// before any session exists), so every `mcp__*` schema would ship on every request.
///
/// The classification happens here, where [`tool_exposure`] still has the raw name and the server
/// config; a registry sees only the namespaced name and could not ask. A hidden tool is dropped
/// here too, ahead of every registration path, so no door can register it.
#[derive(Clone)]
pub(crate) struct ServerTools {
    pub(crate) tools: Vec<Arc<McpTool>>,
    /// Namespaced (`mcp__server__tool`) names, matching what a registry's deferred set holds.
    pub(crate) deferred: Vec<String>,
}

impl ServerTools {
    pub(crate) fn from_tools(tools: Vec<McpTool>) -> Self {
        let mut kept = Vec::with_capacity(tools.len());
        let mut deferred = Vec::new();
        for tool in tools {
            match tool_exposure(tool.server_config(), tool.raw_name()) {
                crate::config::Deferral::Hidden => continue,
                crate::config::Deferral::Deferred => deferred.push(tool.namespaced_name.clone()),
                crate::config::Deferral::Eager => {}
            }
            kept.push(Arc::new(tool));
        }
        Self {
            tools: kept,
            deferred,
        }
    }
}

/// Someone holding a copy of the servers' tools who needs to be told when they change.
///
/// Registries implement this in `crate::tools`; the client never names a registry.
pub(crate) trait ServerToolsObserver: Send + Sync {
    /// A stable identity, so [`McpClientManager::unsubscribe`] can find the entry
    /// [`McpClientManager::subscribe`] made whichever handle to the observer it is given.
    fn identity(&self) -> usize;
    fn server_tools_changed(&self, server_name: &str, tools: &ServerTools);
}

/// Lifecycle state of a single MCP server. Transitions:
/// - Built as `Disabled` (config says so) or `Pending` (will be connected by the background
///   connector).
/// - `Pending` → `Connected` once the probe or `initialize` has settled the era, through
///   [`ServerEntry::adopt_connection`].
/// - `Pending` → `Failed` on connect error or connect-timeout.
/// - `Connected` → `Connected` (with a new `service` Arc) on reconnect.
#[derive(Clone)]
pub(crate) enum ServerState {
    Disabled,
    Pending,
    Connected {
        service: Arc<McpRunningService>,
    },
    Failed {
        error: String,
        #[allow(
            dead_code,
            reason = "when the failure happened; recorded so the state carries it, read by nothing yet"
        )]
        at: std::time::Instant,
    },
}

impl ServerState {
    /// Why this server can't serve a tool call right now, or `None` when it can.
    ///
    /// Deliberately terse and free of instructions: it states the condition and leaves the agent
    /// to decide what to do. "Still connecting" and "failed" are kept distinct because they call
    /// for opposite behavior, and collapsing them produces an agent that either gives up too
    /// early or retries forever.
    pub(crate) fn unavailable_reason(&self) -> Option<String> {
        match self {
            ServerState::Connected { .. } => None,
            ServerState::Pending => Some("is still connecting".to_string()),
            ServerState::Failed { error, .. } => Some(format!("is unavailable: {error}")),
            ServerState::Disabled => Some("is disabled in config".to_string()),
        }
    }

    pub(crate) fn label(&self) -> &'static str {
        match self {
            ServerState::Disabled => "disabled",
            ServerState::Pending => "pending",
            ServerState::Connected { .. } => "connected",
            ServerState::Failed { .. } => "failed",
        }
    }
}

/// One enabled server that isn't `Connected`, as reported by
/// [`McpClientManager::enabled_not_connected`].
#[derive(Clone)]
pub(crate) struct NotConnected {
    pub(crate) name: String,
    /// Whether this server gates the turn. See [`crate::config::McpServerConfig::required`].
    pub(crate) required: bool,
    pub(crate) state: ServerState,
}

/// The manager a retried entry heals into, and the default its tools are classified against.
pub(crate) struct BackgroundRetry {
    pub(crate) manager: std::sync::Weak<McpClientManager>,
    pub(crate) mcp_default_permission: Option<Permission>,
}

/// `error` with the URLs in its words redacted, as `connector::record_connect_failure` redacts
/// what it records: a reconnect's refusal reaches the model and the session's history too, and a
/// server's `url` may name its key in the query string.
fn redact_mcp_error(error: MekaError) -> MekaError {
    match error {
        MekaError::McpConnection {
            server_name,
            message,
        } => MekaError::McpConnection {
            server_name,
            message: crate::text::redact_urls(&message),
        },
        MekaError::McpAuth {
            server_name,
            message,
        } => MekaError::McpAuth {
            server_name,
            message: crate::text::redact_urls(&message),
        },
        other => other,
    }
}

/// Holds the lifecycle state of a single MCP server plus reconnection machinery. Wrapped in an
/// [`Arc`] and shared between the manager, the per-server tool adapters, and the resource/prompt
/// builtin tools so every caller sees the current service (or the current failure) via
/// [`Self::require_connected`].
pub(crate) struct ServerEntry {
    pub(crate) server_name: String,
    pub(crate) config: McpServerConfig,
    pub(crate) token_store: Option<TokenStore>,
    pub(crate) client_context: Arc<McpClientContext>,
    pub(crate) state: RwLock<ServerState>,
    pub(crate) reconnect_lock: Mutex<()>,
    /// Cap on one call to this server's tools: `ResolvedConfig::mcp_tool_timeout`, carried here
    /// because the adapter that makes the call holds the entry and nothing of the process.
    pub(crate) tool_call_timeout: std::time::Duration,
    /// Why this server's configuration may not be sent at all, when it may not: its `headers` or
    /// `env` name a variable the environment did not supply, so the request would carry the
    /// literal `${NAME}`. Read by every door that connects, because marking the entry `Failed`
    /// alone is not enough: the connector takes every non-disabled entry, and the retry loop would
    /// keep sending the literal for the life of the process.
    pub(crate) refused: Option<String>,
    /// The instructions the handshake carried (`InitializeResult.instructions`, or
    /// `DiscoverResult.instructions` on a modern connection), restamped on every `Connected`
    /// transition.
    ///
    /// Not a `OnceLock`: the MCP spec's "instructions are immutable for the lifetime of the
    /// connection" is about the connection, and a reconnect *is* a new connection with a new
    /// `InitializeResult`, while this entry outlives both. Set once, a redeployed server's first
    /// handshake rides [`crate::prompt::WorldSnapshot`] into the model's context every turn for
    /// the rest of the process.
    ///
    /// A `std::sync::RwLock` and not a `tokio` one because
    /// [`McpClientManager::server_instructions`] is synchronous and is called while building
    /// the per-turn context; the guard is never held across an await.
    pub(crate) instructions: std::sync::RwLock<Option<String>>,
    /// `[mcp].connect_timeout`, copied here so the request helpers that run outside the
    /// manager can honor it. Without it [`bounded`] falls back to its own constant and a
    /// configured timeout applies to `tools/list` but silently not to `resources/read` or
    /// `prompts/get`.
    pub(crate) request_timeout: OnceLock<std::time::Duration>,
    /// What [`connector::retry_until_connected`] needs to own this entry once a reconnect gives
    /// up, set when the connector starts; an entry of a manager that never started one (a
    /// throwaway for `meka mcp reconnect`) has no background to hand it to.
    pub(crate) background_retry: OnceLock<BackgroundRetry>,
    /// How many tools the last `tools/list` dropped to stay under [`MAX_MCP_TOOLS_PER_SERVER`].
    /// Recorded so the cap is *disclosed* rather than only logged: a tool that vanished between
    /// what the server offers and what meka registered is indistinguishable, from the outside,
    /// from a tool the server never had.
    pub(crate) dropped_tools: std::sync::atomic::AtomicUsize,
    /// What the model subscribed to on this server, carried onto every connection the entry
    /// adopts; see `subscriptions.rs`.
    pub(crate) subscriptions: subscriptions::Subscriptions,
    /// Whether this stdio server has completed a legacy handshake after not answering the probe,
    /// so a later connect skips the probe and its wait. Kept for the life of the process, which
    /// the spec allows a client across restarts of one server configuration. Set only by a
    /// handshake that succeeded: one that fails, or that the connect's bound cuts short, leaves
    /// no verdict, so a server replaced or merely slow in the meantime is probed again.
    pub(crate) legacy_era: std::sync::atomic::AtomicBool,
    /// The manager's [`Connects`], which every connect of this entry counts itself in and ends on.
    pub(crate) connects: Arc<Connects>,
}

impl ServerEntry {
    /// The configured per-request bound, or [`DEFAULT_MCP_REQUEST_TIMEOUT`] before the connector
    /// has started (tests, and the window before `start_connector`).
    pub(crate) fn request_timeout(&self) -> std::time::Duration {
        self.request_timeout
            .get()
            .copied()
            .unwrap_or(DEFAULT_MCP_REQUEST_TIMEOUT)
    }

    /// Returns the server's `InitializeResult.instructions` (sanitized + truncated to
    /// [`MAX_MCP_DESCRIPTION_CHARS`]) if the current connection advertised one.
    pub(crate) fn instructions(&self) -> Option<String> {
        crate::sync::read(&self.instructions).clone()
    }

    /// Record what a handshake's `InitializeResult` said, replacing whatever the previous one said.
    ///
    /// Takes the raw string rather than the service, so the sanitizing and the truncating live in
    /// one place and can be exercised without a peer.
    pub(crate) fn record_instructions(&self, raw: Option<String>) {
        let captured =
            raw.map(|raw| truncate(&crate::text::sanitize_text(&raw), MAX_MCP_DESCRIPTION_CHARS));
        // Unconditional, including the `None` a reconnect to a server that has stopped advertising
        // instructions produces. Anything else would leave the previous connection's text standing
        // as if the new one had repeated it.
        *crate::sync::write(&self.instructions) = captured;
    }

    /// Snapshot of the current lifecycle state. `Connected` carries an `Arc<McpRunningService>`
    /// which is cheap to clone.
    pub(crate) async fn state(&self) -> ServerState {
        self.state.read().await.clone()
    }

    pub(crate) fn server_name(&self) -> &str {
        &self.server_name
    }

    /// `tools/list`, bounded in both time and count.
    ///
    /// `list_all_tools` follows the server's pagination cursor to exhaustion, so a server that
    /// answers slowly holds the caller open with no deadline, and one that keeps handing back
    /// cursors grows the tool set without limit. Every caller (the connect path, a
    /// `tools/list_changed` refresh and `meka mcp tools`) wants both bounds.
    pub(crate) async fn list_tools_bounded(
        &self,
        timeout: std::time::Duration,
    ) -> Result<Vec<rmcp::model::Tool>> {
        let peer = self.require_connected().await?;
        let listed = tokio::time::timeout(timeout, peer.list_all_tools())
            .await
            .map_err(|_elapsed| MekaError::McpConnection {
                server_name: self.server_name.clone(),
                message: format!("tools/list timed out after {timeout:?}"),
            })?
            .map_err(|error| MekaError::McpConnection {
                server_name: self.server_name.clone(),
                message: format!("failed to list tools: {error}"),
            })?;

        let advertised = listed.len();
        let kept = cap_advertised_tools(listed, &self.server_name);
        self.dropped_tools.store(
            advertised.saturating_sub(kept.len()),
            std::sync::atomic::Ordering::Relaxed,
        );
        Ok(kept)
    }
}

impl ServerEntry {
    /// Return the live peer if the server is currently `Connected`; otherwise return an error
    /// describing the current lifecycle state. Every tool dispatch / list-call path funnels through
    /// this so the "MCP X not ready" error surfaces at one site.
    pub(crate) async fn require_connected(&self) -> Result<Peer<RoleClient>> {
        let state = self.state.read().await;
        if let ServerState::Connected { service } = &*state {
            return Ok(service.peer().clone());
        }
        // Same wording the unregistered-tool path uses, so one condition reads one way however the
        // agent reached it.
        Err(MekaError::McpConnection {
            server_name: self.server_name.clone(),
            message: state
                .unavailable_reason()
                .unwrap_or_else(|| "is unavailable".to_string()),
        })
    }

    /// The refusal a connect ended by a shutdown answers with.
    pub(crate) fn shutdown_refusal(&self) -> MekaError {
        MekaError::McpConnection {
            server_name: self.server_name.clone(),
            message: "is shutting down".to_string(),
        }
    }

    /// The protocol version this entry's connection speaks, when it has one: the server's answer
    /// to the probe, or the one `initialize` settled on.
    pub(crate) async fn protocol_version(&self) -> Option<ProtocolVersion> {
        match &*self.state.read().await {
            ServerState::Connected { service } => service
                .peer()
                .peer_info()
                .map(|info| info.protocol_version.clone()),
            _ => None,
        }
    }

    /// Make `service` this entry's connection: the state, the instructions the handshake or the
    /// discover carried, and what the model subscribed to, carried over onto it.
    ///
    /// The one door onto `Connected`, for the first connect, the cold-start retry and a reconnect
    /// alike, so none of them can leave a subscription behind on the connection that closed, and
    /// none can bring a server up behind a shutdown: a connect that lands once the stop has fired
    /// is refused here, under the state lock the shutdown takes after firing it, and `service`
    /// is closed, child and all. The set is carried over beside the connection rather
    /// than ahead of it, so a server slow to acknowledge its stream delays no tool discovery and
    /// no call waiting on a reconnect.
    pub(crate) async fn adopt_connection(
        self: &Arc<Self>,
        service: McpRunningService,
    ) -> Result<()> {
        let mut state = self.state.write().await;
        if self.connects.stopping().is_cancelled() {
            drop(state);
            // Closed rather than dropped: a drop leaves the transport to a task the runtime may
            // never poll again on its way out, and with it the child a stdio transport owns.
            close_service(&self.server_name, service).await;
            return Err(self.shutdown_refusal());
        }
        self.record_instructions(
            service
                .peer()
                .peer_info()
                .and_then(|info| info.instructions.clone()),
        );
        *state = ServerState::Connected {
            service: Arc::new(service),
        };
        drop(state);
        tokio::spawn({
            let entry = Arc::clone(self);
            async move { subscriptions::resume(&entry).await }
        });
        Ok(())
    }

    /// Transport-close check used by [`Self::reconnect`]. Returns false if the server isn't
    /// `Connected` (there's nothing to reconnect).
    async fn needs_reconnect(&self) -> bool {
        match &*self.state.read().await {
            ServerState::Connected { service } => service.peer().is_transport_closed(),
            _ => false,
        }
    }

    /// Reopen this server's transport after it closed under a `Connected` state, serialized on
    /// `reconnect_lock` so concurrent tool calls do not stampede; a caller that finds the
    /// transport reopened already returns at once. Each attempt is bounded by the connect timeout
    /// and raced against `cancellation`, so a peer that accepts the connection and never answers
    /// `initialize` cannot hold a turn past a stop. Giving up leaves the entry `Failed` and hands
    /// it to the background retry, so `meka mcp list` says what is true and a later call is
    /// refused in the state's words rather than paying the whole backoff again.
    ///
    /// Schedule: five attempts, 1 s, 2 s, 4 s and 8 s apart. Only remote (HTTP) transports go
    /// through backoff; a dead stdio child has to be respawned and retry-after-sleep doesn't help.
    pub(crate) async fn reconnect(
        self: &Arc<Self>,
        cancellation: &tokio_util::sync::CancellationToken,
    ) -> Result<()> {
        let _guard = tokio::select! {
            guard = self.reconnect_lock.lock() => guard,
            _ = cancellation.cancelled() => return Err(MekaError::Interrupted),
        };

        if !self.needs_reconnect().await {
            return Ok(());
        }
        if let Some(reason) = &self.refused {
            return Err(MekaError::McpConnection {
                server_name: self.server_name.clone(),
                message: reason.clone(),
            });
        }

        tracing::warn!(
            "MCP server '{server_name}' transport closed; attempting reconnect",
            server_name = self.server_name
        );
        self.reconnect_attempts(cancellation).await
    }

    /// The attempts [`Self::reconnect`] makes and what giving up leaves behind, under the lock the
    /// caller holds.
    ///
    /// The connect future is `!Send` for an OAuth-authenticated server (rmcp's auth module holds a
    /// `!Sync` closure slot across an await), so each attempt is driven on a `spawn_blocking`
    /// thread with the outer runtime's `Handle` to keep `Tool::execute`'s `Send` bound satisfied.
    /// The caller's stop leaves that thread to its own end; the bound and a shutdown end it from
    /// inside its own future.
    async fn reconnect_attempts(
        self: &Arc<Self>,
        cancellation: &tokio_util::sync::CancellationToken,
    ) -> Result<()> {
        let max_attempts: u32 = match self.config.transport {
            McpTransport::Stdio => 1,
            McpTransport::Http => 5,
        };
        let bound = self.request_timeout();
        let mut last_error: Option<MekaError> = None;
        for attempt in 0..max_attempts {
            if attempt > 0 {
                // 1 s, 2 s, 4 s, 8 s between the five attempts.
                let delay_secs = std::cmp::min(30u64, 1u64 << (attempt - 1));
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_secs(delay_secs)) => {}
                    _ = cancellation.cancelled() => return Err(MekaError::Interrupted),
                    _ = self.connects.stopping().cancelled() => return Err(self.shutdown_refusal()),
                }
            }
            let handle = tokio::runtime::Handle::current();
            let entry = Arc::clone(self);
            let connecting = self.connects.connecting();
            // The bound and the stop sit inside the blocking thread's own future, which is the
            // one thing that can end it: dropping the join handle out here abandons the thread,
            // and with it the child the connect was spawning.
            let connect = tokio::task::spawn_blocking(move || {
                let _connecting = connecting;
                handle.block_on(async {
                    tokio::select! {
                        biased;
                        _ = entry.connects.stopping().cancelled() => None,
                        result = tokio::time::timeout(bound, connector::connect_server(&entry)) => {
                            Some(result.unwrap_or_else(|_elapsed| Err(MekaError::McpConnection {
                                server_name: entry.server_name.clone(),
                                message: format!("reconnect timed out after {bound:?}"),
                            })))
                        }
                    }
                })
            });
            // Counted until the service is adopted or refused: the thread's own count ends when
            // its future returns, and a shutdown must not pass between the two.
            let landing = self.connects.connecting();
            let result = tokio::select! {
                result = connect => result,
                _ = cancellation.cancelled() => return Err(MekaError::Interrupted),
            };

            match result {
                // Ended by a shutdown: the entry is about to be closed, so nothing is recorded
                // against it and no retry spawned.
                Ok(None) => return Err(self.shutdown_refusal()),
                Ok(Some(Ok(new_service))) => {
                    self.adopt_connection(new_service).await?;
                    drop(landing);
                    tracing::info!(
                        "reconnected to MCP server '{server_name}' on attempt {attempt}",
                        server_name = self.server_name,
                        attempt = attempt + 1
                    );
                    self.relist_after_reconnect().await;
                    return Ok(());
                }
                Ok(Some(Err(error))) => {
                    let error = redact_mcp_error(error);
                    tracing::warn!(
                        "MCP server '{server_name}' reconnect attempt {attempt} failed: {error}",
                        server_name = self.server_name,
                        attempt = attempt + 1
                    );
                    last_error = Some(error);
                }
                Err(join_error) => {
                    tracing::warn!(
                        "MCP server '{server_name}' reconnect task join error on attempt {attempt}: {join_error}",
                        server_name = self.server_name,
                        attempt = attempt + 1
                    );
                    last_error = Some(MekaError::McpConnection {
                        server_name: self.server_name.clone(),
                        message: format!("reconnect task join error: {join_error}"),
                    });
                }
            }
        }

        // Given up: the entry says so, through the door every connect failure is recorded by,
        // and the background retry owns it from here. Left `Connected` over the dead transport,
        // `meka mcp list` says connected, every later call pays the whole backoff again, and
        // nothing retries in between.
        let error = last_error.unwrap_or_else(|| MekaError::McpConnection {
            server_name: self.server_name.clone(),
            message: "reconnect made no attempt".to_string(),
        });
        let cause = match &error {
            MekaError::McpConnection { message, .. } | MekaError::McpAuth { message, .. } => {
                message.clone()
            }
            other => other.to_string(),
        };
        connector::record_connect_failure(self, &self.server_name, cause).await;
        if let Some(retry) = self.background_retry.get() {
            tokio::spawn(connector::retry_until_connected(
                Arc::clone(self),
                retry.manager.clone(),
                retry.mcp_default_permission,
                bound,
                connector::INITIAL_RETRY_BACKOFF,
            ));
        }
        Err(error)
    }

    /// Re-list this server's tools after a reconnect has swapped in a new transport.
    ///
    /// Ending a reconnect at the transport, on the reasoning that the tool adapters already exist
    /// and resolve the live peer at dispatch time, leaves the tool list stale. That is true of
    /// *dispatch* and not of the *list*: the peer on the other side is a new session with a new
    /// `InitializeResult`, and a fresh `initialize` produces no `tools/list_changed` because the
    /// client is expected to list. A server redeployed with a tool dropped would keep being
    /// advertised, and one added would never be learned.
    ///
    /// Inline rather than spawned: the four resource and prompt retry sites reconnect and then
    /// immediately retry their request, and a listing that lands after the retry would leave the
    /// window this exists to close. The cost is one `tools/list` on a connection that has just
    /// completed a full handshake.
    ///
    /// A failure here is a `warn!` rather than a failed reconnect. The transport is back, which is
    /// what the caller asked for, and the stale tool set is the condition that already held.
    async fn relist_after_reconnect(&self) {
        let Some(manager) = self
            .client_context
            .manager()
            .and_then(|manager| manager.upgrade())
        else {
            // No manager means nothing is holding a registry to update: the shape the unit tests
            // build, and the shape a shutting-down process ends in.
            return;
        };
        match manager.refresh_server_tools(&self.server_name).await {
            Ok(count) => tracing::info!(
                "MCP server '{server_name}' re-registered {count} tool(s) after reconnect",
                server_name = self.server_name
            ),
            Err(error) => tracing::warn!(
                "failed to re-list tools for MCP server '{server_name}' after reconnect; keeping the \
                 previous set: {error}",
                server_name = self.server_name
            ),
        }
    }
}

/// Runtime tuning for the background MCP connector. Pulled from `ResolvedConfig` by the binary; the
/// manager uses it directly.
pub(crate) struct McpRuntimeConfig {
    /// Per-server wrap around connect + the probe or `initialize` + `list_tools`.
    pub(crate) connect_timeout: std::time::Duration,
    /// Max concurrent stdio spawns, from `[mcp].stdio_concurrency`.
    pub(crate) stdio_concurrency: usize,
    /// Max concurrent HTTP connects, from `[mcp].http_concurrency`.
    pub(crate) http_concurrency: usize,
}

impl McpRuntimeConfig {
    pub(crate) fn from_config(config: &crate::config::ResolvedConfig) -> Self {
        Self {
            connect_timeout: config.mcp_connect_timeout,
            stdio_concurrency: config.mcp_stdio_concurrency,
            http_concurrency: config.mcp_http_concurrency,
        }
    }
}

impl McpClientManager {
    /// Validate configs and build a manager with every entry `Disabled` or `Pending`.
    ///
    /// Spawns no process or network work; [`Self::start_connector`] does, once the registries have
    /// subscribed, so the connector can register tools into them as each server comes up without
    /// any registry having to exist before config validation.
    #[allow(
        clippy::unused_async,
        reason = "the signature is the boundary every host and test calls, and the credential check it is likely to grow will await the store"
    )]
    pub(crate) async fn prepare(
        configs: &[McpServerConfig],
        mcp_default_permission: Option<Permission>,
        token_store: Option<TokenStore>,
        client_context: Arc<McpClientContext>,
        tool_call_timeout: std::time::Duration,
    ) -> Result<Arc<Self>> {
        let mut servers = HashMap::new();
        let mut pending: Vec<Arc<ServerEntry>> = Vec::new();
        let connects = Connects::new();

        for original_config in configs {
            // Apply env-var substitution (`${VAR}` / `${VAR:-default}`) once, up-front, so the rest
            // of the pipeline sees only resolved values.
            let mut config = original_config.clone();
            let unresolved = crate::mcp::expand::expand_server_config(&mut config);
            if !unresolved.names.is_empty() {
                tracing::warn!(
                    "MCP server '{name}': unresolved env vars {names:?} left literal in config",
                    name = config.name,
                    names = unresolved.names
                );
            }

            if config.name.is_empty() {
                return Err(MekaError::McpConnection {
                    server_name: "(empty)".to_string(),
                    message: "server name must not be empty".to_string(),
                });
            }

            // Reject anything that would collide with meka-internal names or our
            // `mcp__<server>__<tool>` namespace separator.
            if crate::mcp::sanitize::is_reserved_server_name(&config.name) {
                return Err(MekaError::McpConnection {
                    server_name: config.name.clone(),
                    message: "server name is reserved (meka, ide, or mcp_*)".to_string(),
                });
            }

            let normalized = crate::mcp::sanitize::normalize_server_name(&config.name);
            if normalized != config.name {
                return Err(MekaError::McpConnection {
                    server_name: config.name.clone(),
                    message: format!(
                        "server name contains characters not allowed in tool prefixes (would normalize to '{normalized}')"
                    ),
                });
            }

            if config.name.contains("__") {
                return Err(MekaError::McpConnection {
                    server_name: config.name.clone(),
                    message: "server name must not contain '__'".to_string(),
                });
            }

            if servers.contains_key(&config.name) {
                return Err(MekaError::McpConnection {
                    server_name: config.name.clone(),
                    message: "duplicate server name".to_string(),
                });
            }

            let is_disabled = config.disabled.unwrap_or(false);
            if is_disabled {
                tracing::info!(
                    "MCP server '{name}' is disabled in config",
                    name = config.name
                );
            }
            // A credential the environment did not supply is not sent as the literal `${NAME}`:
            // the request cannot succeed, and the string names a secret the operator meant to
            // keep out of the file. `config.toml`'s own `${VAR}` fails closed the same way.
            let refused = (!is_disabled && unresolved.in_secret_bearing_fields).then(|| {
                let what = unresolved.describe();
                tracing::warn!(
                    "MCP server '{name}' will not connect: {what}, in its `args`, `env`, `url` or \
                     `headers`",
                    name = config.name
                );
                format!("{what}, in `args`, `env`, `url` or `headers`")
            });
            let initial_state = if is_disabled {
                ServerState::Disabled
            } else if let Some(error) = &refused {
                ServerState::Failed {
                    error: error.clone(),
                    at: std::time::Instant::now(),
                }
            } else {
                ServerState::Pending
            };

            let entry = Arc::new(ServerEntry {
                server_name: config.name.clone(),
                config: config.clone(),
                token_store: token_store.clone(),
                client_context: Arc::clone(&client_context),
                state: RwLock::new(initial_state),
                reconnect_lock: Mutex::new(()),
                tool_call_timeout,
                refused,
                instructions: std::sync::RwLock::new(None),
                request_timeout: OnceLock::new(),
                background_retry: OnceLock::new(),
                dropped_tools: std::sync::atomic::AtomicUsize::new(0),
                subscriptions: subscriptions::Subscriptions::default(),
                legacy_era: std::sync::atomic::AtomicBool::new(false),
                connects: Arc::clone(&connects),
            });
            // A refused entry is not pending: it is failed for good, and the connector's
            // post-settle retry only looks at what it was handed.
            if !is_disabled && entry.refused.is_none() {
                pending.push(Arc::clone(&entry));
            }
            servers.insert(config.name.clone(), entry);
        }

        // Initialize the watch with `true` when nothing will ever be pending (all servers disabled,
        // or no servers configured) so callers of `all_ready` / `await_settled` short-circuit
        // immediately. `send` on a Sender with no receivers errors and drops the value, so the
        // initial-value path is the only safe pre-subscription way to publish settled.
        let initial_settled = pending.is_empty();
        let (settled_tx, _) = tokio::sync::watch::channel(initial_settled);
        let manager = Arc::new(Self {
            servers,
            client_context,
            mcp_default_permission,
            deferral: OnceLock::new(),
            settled: settled_tx,
            pending_entries: std::sync::Mutex::new(Some(pending)),
            tools_snapshot: tokio::sync::RwLock::new(HashMap::new()),
            observers: tokio::sync::RwLock::new(Vec::new()),
            connect_timeout: OnceLock::new(),
            connects,
        });
        Ok(manager)
    }

    /// Update the live snapshot for one server's tools and tell every observer. Called by the
    /// connector when a server reaches `Connected` and by `on_tool_list_changed` when a server
    /// signals a dynamic update.
    ///
    /// The snapshot is what a new subscriber is replayed; the fan-out keeps existing ones in sync
    /// without requiring them to re-subscribe. Both are given the whole [`ServerTools`], so a
    /// registry that subscribes later cannot end up with a different eager-vs-deferred split from
    /// one that was already here.
    async fn update_server_tools(&self, server_name: &str, tools: ServerTools) {
        // Snapshot first, then fan out. [`Self::subscribe`] does the mirror image, and the pairing
        // is what closes the window where an observer subscribing concurrently is missed by the
        // fan-out *and* subscribes before the snapshot names the update.
        self.tools_snapshot
            .write()
            .await
            .insert(server_name.to_string(), tools.clone());
        let observers = self.observers.read().await;
        for observer in observers.iter() {
            observer.server_tools_changed(server_name, &tools);
        }
    }

    /// [`Self::update_server_tools`] for a caller holding the tools themselves.
    ///
    /// The one door for all three discovery paths (the connector's first `tools/list`, the refresh
    /// a `tools/list_changed` notification triggers, and the re-list [`Self::refresh_server_tools`]
    /// runs after a reconnect), so none of them can classify a tool differently from the others.
    pub(super) async fn register_server_tools(&self, server_name: &str, tools: Vec<McpTool>) {
        self.update_server_tools(server_name, ServerTools::from_tools(tools))
            .await;
    }

    /// Tell `observer` about every server's tools now, and about every change from here on.
    ///
    /// Pushes the observer *before* replaying the snapshot, so any concurrent
    /// [`Self::update_server_tools`] either fans out to it (push happened first) or has its result
    /// replayed (push happened second). The opposite ordering (read snapshot, then push) has a
    /// window where an update can land between the snapshot read and the push, with the observer
    /// missing it forever. An observer applies a listing idempotently, so the double delivery when
    /// both paths fire is harmless.
    ///
    /// Sessions subscribe their registry at `session/new` and pair it with [`Self::unsubscribe`] at
    /// `session/close`.
    pub(crate) async fn subscribe(&self, observer: Arc<dyn ServerToolsObserver>) {
        self.observers.write().await.push(Arc::clone(&observer));
        let snapshot = self.tools_snapshot.read().await;
        for (server_name, tools) in snapshot.iter() {
            observer.server_tools_changed(server_name, tools);
        }
    }

    /// Stop telling the observer with this [`ServerToolsObserver::identity`]. No-op if it is not
    /// subscribed.
    pub(crate) async fn unsubscribe(&self, identity: usize) {
        let mut observers = self.observers.write().await;
        observers.retain(|observer| observer.identity() != identity);
    }

    /// Record `[mcp].deferral` for the meta-tools; a second call is ignored.
    pub(crate) fn set_deferral(&self, deferral: crate::config::Deferral) {
        // `set` fails only when the slot is already filled, which is the documented no-op.
        let _already_set = self.deferral.set(deferral);
    }

    /// What becomes of the meta-tools: `[mcp].deferral`, or its default when nothing recorded it.
    pub(crate) fn deferral(&self) -> crate::config::Deferral {
        self.deferral.get().copied().unwrap_or_default()
    }

    /// The bound to put on an MCP request made outside the connector.
    fn request_timeout(&self) -> std::time::Duration {
        self.connect_timeout
            .get()
            .copied()
            .unwrap_or(DEFAULT_MCP_REQUEST_TIMEOUT)
    }

    /// Spawn the background connector. Consumes the `Pending` entry list stashed by
    /// [`Self::prepare`] so subsequent calls are no-ops. Safe to call on managers with no pending
    /// entries.
    ///
    /// The connector writes tool discoveries through [`Self::update_server_tools`], which fans out
    /// to every observer subscribed via [`Self::subscribe`]. The caller does not pass a specific
    /// registry: attach yours first, then start the connector.
    pub(crate) fn start_connector(self: &Arc<Self>, runtime: McpRuntimeConfig) {
        // Recorded before the early return, so a second `start_connector` call still leaves the
        // timeout set for the request paths that read it.
        if self.connect_timeout.set(runtime.connect_timeout).is_err() {
            tracing::debug!("MCP connector already configured; keeping the first timeout");
        }
        // Every entry gets the same bound, so the request helpers that only hold an
        // `Arc<ServerEntry>` honor the configured timeout rather than falling back to the
        // module default.
        for entry in self.servers.values() {
            if entry.request_timeout.set(runtime.connect_timeout).is_err() {
                tracing::debug!(
                    "MCP server {server_name} already has its request timeout",
                    server_name = entry.server_name
                );
            }
            if entry
                .background_retry
                .set(BackgroundRetry {
                    manager: Arc::downgrade(self),
                    mcp_default_permission: self.mcp_default_permission,
                })
                .is_err()
            {
                tracing::debug!(
                    "MCP server {server_name} already has its background retry",
                    server_name = entry.server_name
                );
            }
        }
        let Some(pending) = crate::sync::lock(&self.pending_entries).take() else {
            return;
        };
        let manager = Arc::clone(self);
        let settled = self.settled.clone();
        let mcp_default_permission = self.mcp_default_permission;
        tokio::spawn(async move {
            connector::run_connector(pending, manager, mcp_default_permission, runtime, settled)
                .await;
        });
    }

    /// True when every enabled server has reached a terminal state (`Connected` or `Failed`).
    /// Returns `true` if there are no enabled servers configured. Non-blocking.
    pub(crate) fn all_ready(&self) -> bool {
        *self.settled.borrow()
    }

    /// Parks until the background connector finishes processing every enabled server. Returns
    /// immediately if already settled. Safe to call concurrently from multiple turn dispatches.
    pub(crate) async fn await_settled(&self) {
        let mut rx = self.settled.subscribe();
        if *rx.borrow() {
            return;
        }
        if rx.wait_for(|done| *done).await.is_err() {
            tracing::debug!("the MCP connector ended before it settled");
        }
    }

    /// Snapshot of enabled servers that are not currently `Connected` (still `Pending` or
    /// `Failed`), each paired with whether it is `required`. [`crate::agent`]'s turn gate stops
    /// only for the required ones; the rest are logged at `debug` and the session runs without
    /// them. Deliberately not `warn`: this is consulted on every turn, and a server that is down
    /// stays down, so warning here would print a line before every reply for the life of the
    /// session.
    pub(crate) async fn enabled_not_connected(&self) -> Vec<NotConnected> {
        let mut out = Vec::new();
        for (name, entry) in &self.servers {
            let state = entry.state().await;
            match state {
                ServerState::Connected { .. } | ServerState::Disabled => {}
                other => out.push(NotConnected {
                    name: name.clone(),
                    // `required` is settled in `ResolvedConfig::from_cli`, so `None` here can only
                    // mean a config built outside that path (tests, `meka mcp add`); treat it as
                    // optional, matching the default.
                    required: entry.config.required.unwrap_or(false),
                    state: other,
                }),
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    /// Why a `mcp__<server>__<tool>` name isn't callable, when the reason is the server rather
    /// than the tool. `None` means this is not an MCP name, or names a server meka has never heard
    /// of: both genuinely "unknown tool".
    ///
    /// Exists because a server that never connected registers no tools, so its names fall through
    /// to the agent's unknown-tool arm and the model is told the tool does not exist. It does
    /// exist; it is unreachable. An agent told "unknown" reasonably stops asking, which is the
    /// wrong lesson when the server is still connecting or is one `meka mcp reconnect` away. The
    /// prompt-level instructions, a skill, or a resumed conversation can all name a tool whose
    /// server is currently down.
    pub(crate) async fn unavailable_tool_reason(&self, tool_name: &str) -> Option<String> {
        let rest = tool_name.strip_prefix("mcp__")?;
        // Server names cannot contain `__` (`sanitize::normalize_server_name`), so the first
        // occurrence splits server from tool.
        let (server_name, _tool) = rest.split_once("__")?;
        let entry = self.servers.get(server_name)?;
        Some(format!(
            "MCP server '{}' {}",
            server_name,
            entry.state().await.unavailable_reason()?
        ))
    }

    pub(crate) fn server_entry(&self, server_name: &str) -> Option<Arc<ServerEntry>> {
        self.servers.get(server_name).cloned()
    }

    /// Whether the server owning a namespaced tool name has yet to finish its first connection.
    ///
    /// For callers that report a *standing* condition rather than act on one. A tool missing from
    /// the snapshot because its server is still handshaking is not the same as one whose server
    /// failed or was never configured, and saying so out loud before the handshake finishes tells
    /// the reader something untrue that fixes itself a moment later. Anything that has to *run* the
    /// tool should keep treating both as unavailable, which is what it is.
    ///
    /// Split on the first `__`, like [`Self::unavailable_tool_reason`]: a server name cannot
    /// contain one (`sanitize::normalize_server_name`), so the first occurrence separates server
    /// from tool. A name with no `__` after the prefix is not one meka minted and answers `false`.
    pub(crate) async fn server_is_still_connecting(&self, namespaced_tool: &str) -> bool {
        let Some((server, _tool)) = namespaced_tool
            .strip_prefix("mcp__")
            .and_then(|rest| rest.split_once("__"))
        else {
            return false;
        };
        match self.servers.get(server) {
            Some(entry) => matches!(entry.state().await, ServerState::Pending),
            None => false,
        }
    }

    /// The refusal for a server name that matches nothing, naming the servers that exist.
    fn unknown_server(&self, server_name: &str) -> String {
        let mut known: Vec<&str> = self.servers.keys().map(String::as_str).collect();
        known.sort_unstable();
        crate::text::unknown_name("MCP server", server_name, known)
    }

    /// Find a registered MCP tool by the name the model uses (`mcp__server__tool`).
    ///
    /// Reads [`Self::tools_snapshot`], which `update_server_tools` keeps current, so this is a map
    /// lookup rather than a round trip. That matters because the caller is a scheduled job's gate,
    /// asked once per poll interval per job: resolving through
    /// [`Self::list_advertised_tools`] would put a `tools/list` on the wire every tick.
    ///
    /// `None` covers both "no such tool" and "its server is not connected", which are the same
    /// answer to the only question a gate asks: can this be evaluated right now.
    pub(crate) async fn tool_by_name(&self, name: &str) -> Option<Arc<McpTool>> {
        let snapshot = self.tools_snapshot.read().await;
        snapshot
            .values()
            .flat_map(|server| server.tools.iter())
            .find(|tool| tool.namespaced_name == name)
            .map(Arc::clone)
    }

    pub(crate) fn server_names(&self) -> Vec<String> {
        self.servers.keys().cloned().collect()
    }

    /// Returns `(server_name, instructions)` pairs for every connected server whose current
    /// handshake advertised an `InitializeResult.instructions` string. Already sanitized and
    /// truncated to [`MAX_MCP_DESCRIPTION_CHARS`]. Used by the agent loop to splice MCP server
    /// instructions into the per-turn context.
    pub(crate) fn server_instructions(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for (name, entry) in &self.servers {
            if let Some(text) = entry.instructions()
                && !text.trim().is_empty()
            {
                out.push((name.clone(), text));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Re-run one server's `tools/list` and publish the result.
    ///
    /// The repair a *reconnect* needs. A fresh `initialize` produces no `tools/list_changed` (the
    /// client is expected to list), so a server redeployed with a different tool set is invisible
    /// until something asks again. Errors propagate rather than publishing an empty set: a failed
    /// list means meka does not know what the server has, which is not the same as knowing it has
    /// nothing.
    pub(crate) async fn refresh_server_tools(&self, server_name: &str) -> Result<usize> {
        let adapters = self.discover_tools_for_server(server_name).await?;
        let count = adapters.len();
        self.register_server_tools(server_name, adapters).await;
        Ok(count)
    }

    /// One server's tools as [`McpTool`]s, from a fresh `tools/list`. Empty for a name nothing
    /// configured.
    pub(crate) async fn discover_tools_for_server(
        &self,
        server_name: &str,
    ) -> Result<Vec<McpTool>> {
        let Some(entry) = self.servers.get(server_name) else {
            return Ok(Vec::new());
        };
        connector::build_mcp_tools(entry, self.mcp_default_permission, self.request_timeout()).await
    }

    /// [`Self::discover_tools_for_server`], classified the way the connector classifies a listing.
    pub(crate) async fn discover_server_tools(&self, server_name: &str) -> Result<ServerTools> {
        Ok(ServerTools::from_tools(
            self.discover_tools_for_server(server_name).await?,
        ))
    }

    /// Heal one server on demand, picking the right repair for the state it is actually in.
    ///
    /// The dispatch is the whole point, because the two repairs are not interchangeable.
    /// [`ServerEntry::reconnect`] reopens a transport that has closed under a state still claiming
    /// `Connected`, and no-ops when it has not. An entry that is `Failed` has no transport to
    /// reopen and never reached tool discovery in the first place, so [`connector::connect_one`],
    /// which does the whole handshake, is the right call there.
    /// Both end in a `tools/list`, because a new connection is a new session and its tool set is
    /// only knowable by asking.
    ///
    /// A `Failed` server is already being retried in the background with exponential backoff, so
    /// this is an impatience button rather than the only route back: it collapses the wait for an
    /// operator who has just fixed whatever was wrong.
    pub(crate) async fn reconnect_server(
        self: &Arc<Self>,
        server_name: &str,
        connect_timeout: std::time::Duration,
    ) -> Result<ServerState> {
        let Some(entry) = self.servers.get(server_name).cloned() else {
            return Err(MekaError::McpConnection {
                server_name: server_name.to_string(),
                message: self.unknown_server(server_name),
            });
        };
        match entry.state().await {
            // Refused, not honored. `run_connector` owns every `Pending` entry and connects it
            // without taking `reconnect_lock` (it iterates the list captured at `prepare` time),
            // so a second `connect_one` here races it: two child processes for a stdio server,
            // and if the second attempt loses, `record_connect_failure` overwrites a working
            // `Connected` with `Failed`. Servers past `stdio_concurrency` sit `Pending` for
            // seconds, which is when a dashboard polling `GET /v1/mcp` sees "not connected" and
            // tries to help. The one caller (`host::http::handlers::info::mcp_reconnect`) reads
            // the state first and answers 200 `pending`, because over the wire a refusal reads as
            // "the server failed" when nothing was attempted; this arm is the backstop, and the
            // handler's own check is what produces the 200.
            ServerState::Pending => {
                return Err(MekaError::McpConnection {
                    server_name: server_name.to_string(),
                    message: "is still connecting; wait for it to settle".to_string(),
                });
            }
            ServerState::Disabled => {
                return Err(MekaError::McpConnection {
                    server_name: server_name.to_string(),
                    message: format!("is disabled in config; run `meka mcp enable {server_name}`"),
                });
            }
            ServerState::Connected { .. } => {
                // Reconnect is still the right call: it no-ops unless the transport has actually
                // closed underneath a state that still says `Connected`, which is exactly the case
                // an operator reaching for this button cannot see from outside.
                //
                // Bounded here rather than inside: `ServerEntry::reconnect` retries an HTTP
                // transport up to five times with its own backoff and wraps none of it in a
                // timeout, so an endpoint that blackholes connections would hold this request far
                // past the budget the caller passed in. The re-list it ends with carries its own
                // bound, so this covers the whole of it either way.
                tokio::time::timeout(
                    connect_timeout,
                    entry.reconnect(&tokio_util::sync::CancellationToken::new()),
                )
                .await
                .map_err(|_| MekaError::McpConnection {
                    server_name: server_name.to_string(),
                    message: format!("reconnect did not complete within {connect_timeout:?}"),
                })??;
            }
            ServerState::Failed { .. } => {
                // Under `reconnect_lock`, which is the same guard `retry_until_connected` takes
                // around this call. Without it two of these requests, or one racing the background
                // retry, drive two `connect_one`s into the same entry: two child processes for a
                // stdio server, both writing `state`, the loser's service orphaned, and
                // `update_server_tools` fanned out twice.
                let _guard = entry.reconnect_lock.lock().await;
                // Re-checked under the lock: whoever held it may have just connected this entry,
                // in which case a second connect would replace a healthy transport for nothing.
                if matches!(entry.state().await, ServerState::Failed { .. }) {
                    connector::connect_one(
                        Arc::clone(&entry),
                        Arc::clone(self),
                        self.mcp_default_permission,
                        connect_timeout,
                    )
                    .await;
                }
            }
        }
        Ok(entry.state().await)
    }

    /// Connect to the named server and list EVERY advertised tool, including ones currently
    /// filtered out by `allowed_tools` / `disabled_tools` so users editing those lists can see what
    /// names are available. Permission is resolved through the normal 5-step chain with the winning
    /// step recorded on each entry.
    ///
    /// Differs from [`Self::discover_tools_for_server`] by (a) not filtering by allow/block lists,
    /// (b) not registering adapters, and (c) capturing the resolution source for display.
    pub(crate) async fn list_advertised_tools(
        &self,
        server_name: &str,
    ) -> Result<Vec<AdvertisedTool>> {
        let Some(entry) = self.servers.get(server_name) else {
            return Err(MekaError::McpConnection {
                server_name: server_name.to_string(),
                message: self.unknown_server(server_name),
            });
        };

        let server_config = &entry.config;
        let tools = entry.list_tools_bounded(self.request_timeout()).await?;

        let mut out = Vec::with_capacity(tools.len());
        for tool in tools {
            let raw_name = tool.name.as_ref().to_string();
            let raw_description = tool
                .description
                .as_ref()
                .map(|d| d.as_ref().to_string())
                .unwrap_or_default();
            let description = truncate(
                &crate::text::sanitize_text(&raw_description),
                MAX_MCP_DESCRIPTION_CHARS,
            );
            let (resolved_permission, permission_source) = resolve_tool_permission_with_source(
                &raw_name,
                tool.annotations.as_ref(),
                server_config,
                self.mcp_default_permission,
            );
            let allowed = tool_is_allowed(server_config, &raw_name);
            let exposure = tool_exposure(server_config, &raw_name);
            let read_only_hint_declined =
                read_only_hint_was_declined(tool.annotations.as_ref(), permission_source);
            out.push(AdvertisedTool {
                raw_name,
                description,
                resolved_permission,
                permission_source,
                allowed,
                exposure,
                read_only_hint_declined,
            });
        }

        out.sort_by(|a, b| a.raw_name.cmp(&b.raw_name));
        Ok(out)
    }

    /// Shutdown helper for callers that hold the manager through an `Arc`. Just calls
    /// [`Self::shutdown_within`], which needs only `&self`.
    pub(crate) async fn shutdown_arc(self: Arc<Self>) {
        self.shutdown_within(SHUTDOWN_BUDGET).await;
    }

    /// [`Self::shutdown`] under a total wall-clock bound.
    ///
    /// The loop is serial and each server can spend up to `CLOSE_TIMEOUT`, so an exit's cost scales
    /// with the number of servers that refuse to answer. Every caller is on its way out and some of
    /// them are being timed by something else (systemd, a container runtime, a user holding a
    /// terminal), so the whole teardown gets one budget rather than each server getting its own.
    /// Overrunning it is not an error: the remaining servers fall to rmcp's drop guards, which is
    /// exactly where they were before any of this ran.
    pub(crate) async fn shutdown_within(&self, budget: std::time::Duration) {
        if tokio::time::timeout(budget, self.shutdown()).await.is_err() {
            tracing::warn!(
                "MCP shutdown exceeded {budget:?}; a stdio child may outlive this process"
            );
        }
    }

    /// Close every connected server, in place.
    ///
    /// Takes `&self` deliberately. Consuming `self` would make callers `try_unwrap` an `Arc<Self>`
    /// first, which never succeeds: the manager holds the tool registries it serves (the
    /// observers) and those registries hold the six `mcp_resource_*` / `mcp_prompt_*` tools, each
    /// of which holds an `Arc` back to the manager. With sole ownership unreachable,
    /// `close_with_timeout` never runs and stdio children are left to rmcp's drop guard, which
    /// spawns onto a runtime already tearing down.
    ///
    /// The service is taken out from under each entry's `state` lock rather than by owning the
    /// entry, so a `ServerEntry` clone held by an in-flight call doesn't block teardown either. The
    /// entry is left `Disabled` so a tool call arriving during teardown is refused rather than
    /// handed a service that is closing.
    ///
    /// The connects in flight are ended first, and waited for: a connect runs on a blocking thread
    /// the process would otherwise abandon at exit with the child it was spawning still alive. A
    /// connect that ends on the token drops its transport, which kills the child, and leaves the
    /// entry as it was; a connector still working through its queue finds every connect ending at
    /// once and drains.
    pub(crate) async fn shutdown(&self) {
        /// Max time to wait for in-flight tool calls to complete before we drop the shared service
        /// Arc and let the drop-guard cancel it.
        const SHUTDOWN_GRACE: std::time::Duration = std::time::Duration::from_millis(2000);

        if !self.connects.stop(CONNECT_UNWIND_GRACE).await {
            tracing::warn!(
                "an MCP connect did not end within {CONNECT_UNWIND_GRACE:?}; its child may outlive \
                 this process"
            );
        }

        for (server_name, entry) in &self.servers {
            // Only Connected entries have a service to close; Pending / Failed / Disabled entries
            // are tear-down no-ops. Taken under the write lock so a concurrent reconnect can't
            // hand this loop a service it is about to replace.
            let service = {
                let mut state = entry.state.write().await;
                let taken = std::mem::replace(&mut *state, ServerState::Disabled);
                let ServerState::Connected { service } = taken else {
                    *state = taken;
                    continue;
                };
                drop(state);
                service
            };

            // Meant to let in-flight tool calls finish before the transport goes, and currently
            // ineffective: dispatch goes through `require_connected`, which clones
            // `service.peer()`, and rmcp 3.1's `Peer` holds channels rather than an
            // `Arc<RunningService>`, so the count is already 1 and the loop exits at once. The fix
            // belongs in what dispatch holds, not here.
            let deadline = tokio::time::Instant::now() + SHUTDOWN_GRACE;
            while Arc::strong_count(&service) > 1 && tokio::time::Instant::now() < deadline {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }

            match Arc::try_unwrap(service) {
                Ok(owned_service) => close_service(server_name, owned_service).await,
                Err(_arc) => {
                    tracing::debug!(
                        "MCP server '{server_name}' still had in-flight calls after {SHUTDOWN_GRACE:?}; \
                         left to its drop guard"
                    );
                }
            }
        }
    }
}

/// Whether a server offered `readOnlyHint: true` and resolution refused it.
///
/// A hint that *won* is reported as [`PermissionSource::ReadOnlyHint`], so anything else means the
/// hint was present and something below it decided. Only the `true` direction can be declined:
/// `readOnlyHint: false` only ever raises the requirement, so it is always honored and always wins
/// when present.
fn read_only_hint_was_declined(
    tool_annotations: Option<&rmcp::model::ToolAnnotations>,
    source: PermissionSource,
) -> bool {
    tool_annotations
        .and_then(|annotations| annotations.read_only_hint)
        .unwrap_or(false)
        && !matches!(source, PermissionSource::ReadOnlyHint)
}

/// Shared context threaded into every [`handler::MekaClientHandler`] so notification callbacks and
/// server-to-client requests (`elicitation/create`, `tools/list_changed`) can reach the rest of the
/// agent. The manager slot is optional because the handler is constructed before the manager
/// exists; it is filled in post-construction via [`McpClientContext::set_manager`].
#[derive(Default)]
pub(crate) struct McpClientContext {
    /// Weak reference to the MCP manager so the notification callback can rediscover tools without
    /// creating an Arc cycle through the handler. Tool list updates flow through the manager's
    /// observers; no per-context registry slot is needed.
    manager: OnceLock<Weak<McpClientManager>>,
    /// The in-flight calls whose progress notifications are routed back to a frontend.
    pub(crate) progress: progress::ProgressRegistry,
    /// What servers have reported changed, for `mcp_resource_updates_list`.
    pub(crate) resource_updates: resource_updates::ResourceUpdates,
    /// Who completes an interactive login, when anyone can. See [`auth::LoginPrompt`].
    login_prompt: OnceLock<Arc<dyn auth::LoginPrompt>>,
}

impl McpClientContext {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Install the one prompt that can complete an interactive login. Only `meka mcp login` has
    /// one; a second install is a programming error and is ignored with a warning.
    pub(crate) fn set_login_prompt(&self, prompt: Arc<dyn auth::LoginPrompt>) {
        if self.login_prompt.set(prompt).is_err() {
            tracing::warn!("MCP client context: login prompt already set");
        }
    }

    pub(crate) fn login_prompt(&self) -> Option<Arc<dyn auth::LoginPrompt>> {
        self.login_prompt.get().cloned()
    }

    pub(crate) fn set_manager(&self, manager: Weak<McpClientManager>) {
        if self.manager.set(manager).is_err() {
            tracing::warn!("MCP client context: manager already set");
        }
    }

    pub(crate) fn manager(&self) -> Option<Weak<McpClientManager>> {
        self.manager.get().cloned()
    }
}

/// Truncate a string to `max_chars` Unicode scalar values, appending an ellipsis marker if
/// truncation occurred. Operates on `char` boundaries so the result is always valid UTF-8.
pub(crate) fn truncate(text: &str, max_chars: usize) -> String {
    let mut byte_end = text.len();
    for (count, (index, _)) in text.char_indices().enumerate() {
        if count == max_chars {
            byte_end = index;
            break;
        }
    }
    if byte_end < text.len() {
        let mut truncated = String::with_capacity(byte_end + 3);
        truncated.push_str(&text[..byte_end]);
        truncated.push_str("...");
        truncated
    } else {
        text.to_string()
    }
}

/// Bound one MCP round-trip in time and against the turn's cancellation.
///
/// Every helper below is a request to a process meka does not control, over a transport that can
/// accept and then go quiet. Without this a server that never answers parks the tool call, and with
/// it the turn, for the life of the process, and pressing stop does not reach it either, because
/// the token the tool was handed goes unused.
///
/// `biased` so a token already fired wins over a response arriving in the same instant: once the
/// user has stopped the turn, the answer is not wanted whichever got there first.
pub(super) async fn bounded<T>(
    entry: &Arc<ServerEntry>,
    what: &str,
    cancellation: &CancellationToken,
    work: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => Err(MekaError::Interrupted),
        outcome = tokio::time::timeout(entry.request_timeout(), work) => match outcome {
            Ok(result) => result,
            Err(_elapsed) => Err(MekaError::McpConnection {
                server_name: entry.server_name.clone(),
                message: format!("{} timed out after {:?}", what, entry.request_timeout()),
            }),
        },
    }
}

/// List all resources advertised by a server. Returned verbatim from the current peer; no caching
/// is done here.
pub(crate) async fn list_resources(
    entry: &Arc<ServerEntry>,
    cancellation: &CancellationToken,
) -> Result<Vec<Resource>> {
    bounded(entry, "resources/list", cancellation, async {
        let peer = entry.require_connected().await?;
        match peer.list_all_resources().await {
            Ok(resources) => Ok(resources),
            Err(ServiceError::TransportClosed) => {
                entry
                    .reconnect(&tokio_util::sync::CancellationToken::new())
                    .await?;
                let peer = entry.require_connected().await?;
                peer.list_all_resources()
                    .await
                    .map_err(|error| MekaError::McpConnection {
                        server_name: entry.server_name.clone(),
                        message: format!("failed to list resources: {error}"),
                    })
            }
            Err(error) => Err(MekaError::McpConnection {
                server_name: entry.server_name.clone(),
                message: format!("failed to list resources: {error}"),
            }),
        }
    })
    .await
}

pub(crate) async fn read_resource(
    entry: &Arc<ServerEntry>,
    uri: String,
    cancellation: &CancellationToken,
) -> Result<ReadResourceResult> {
    /// One attempt on the connection as it stands, through the rounds a modern server may ask
    /// for. No frontend: a read has no call in flight to route an elicitation to, on either era.
    async fn rounds(
        entry: &Arc<ServerEntry>,
        peer: Peer<RoleClient>,
        mut params: ReadResourceRequestParams,
    ) -> std::result::Result<ReadResourceResult, ServiceError> {
        let mut rounds = handler::Rounds::new(&entry.server_name, None);
        loop {
            (params.input_responses, params.request_state) = rounds.carry();
            match peer.read_resource_once(params.clone()).await? {
                ReadResourceResponse::Complete(result) => return Ok(result),
                ReadResourceResponse::InputRequired(result) => rounds.absorb(result).await?,
                _ => return Err(ServiceError::UnexpectedResponse),
            }
        }
    }

    bounded(entry, "resources/read", cancellation, async {
        let params = ReadResourceRequestParams::new(uri.clone());
        let failed = |error: ServiceError| MekaError::McpConnection {
            server_name: entry.server_name.clone(),
            message: format!("failed to read resource '{uri}': {error}"),
        };
        let peer = entry.require_connected().await?;
        match rounds(entry, peer, params.clone()).await {
            Ok(result) => Ok(result),
            Err(ServiceError::TransportClosed) => {
                entry
                    .reconnect(&tokio_util::sync::CancellationToken::new())
                    .await?;
                let peer = entry.require_connected().await?;
                rounds(entry, peer, params).await.map_err(failed)
            }
            Err(error) => Err(failed(error)),
        }
    })
    .await
}

pub(crate) async fn list_prompts(
    entry: &Arc<ServerEntry>,
    cancellation: &CancellationToken,
) -> Result<Vec<Prompt>> {
    bounded(entry, "prompts/list", cancellation, async {
        let peer = entry.require_connected().await?;
        match peer.list_all_prompts().await {
            Ok(prompts) => Ok(prompts),
            Err(ServiceError::TransportClosed) => {
                entry
                    .reconnect(&tokio_util::sync::CancellationToken::new())
                    .await?;
                let peer = entry.require_connected().await?;
                peer.list_all_prompts()
                    .await
                    .map_err(|error| MekaError::McpConnection {
                        server_name: entry.server_name.clone(),
                        message: format!("failed to list prompts: {error}"),
                    })
            }
            Err(error) => Err(MekaError::McpConnection {
                server_name: entry.server_name.clone(),
                message: format!("failed to list prompts: {error}"),
            }),
        }
    })
    .await
}

pub(crate) async fn get_prompt(
    entry: &Arc<ServerEntry>,
    name: String,
    arguments: Option<serde_json::Map<String, serde_json::Value>>,
    cancellation: &CancellationToken,
) -> Result<GetPromptResult> {
    /// One attempt on the connection as it stands, through the rounds a modern server may ask
    /// for; no frontend, as for [`read_resource`].
    async fn rounds(
        entry: &Arc<ServerEntry>,
        peer: Peer<RoleClient>,
        mut params: GetPromptRequestParams,
    ) -> std::result::Result<GetPromptResult, ServiceError> {
        let mut rounds = handler::Rounds::new(&entry.server_name, None);
        loop {
            (params.input_responses, params.request_state) = rounds.carry();
            match peer.get_prompt_once(params.clone()).await? {
                GetPromptResponse::Complete(result) => return Ok(result),
                GetPromptResponse::InputRequired(result) => rounds.absorb(result).await?,
                _ => return Err(ServiceError::UnexpectedResponse),
            }
        }
    }

    bounded(entry, "prompts/get", cancellation, async {
        let mut params = GetPromptRequestParams::new(name.clone());
        params.arguments = arguments;
        let failed = |error: ServiceError| MekaError::McpConnection {
            server_name: entry.server_name.clone(),
            message: format!("failed to render prompt '{name}': {error}"),
        };
        let peer = entry.require_connected().await?;
        match rounds(entry, peer, params.clone()).await {
            Ok(result) => Ok(result),
            Err(ServiceError::TransportClosed) => {
                entry
                    .reconnect(&tokio_util::sync::CancellationToken::new())
                    .await?;
                let peer = entry.require_connected().await?;
                rounds(entry, peer, params).await.map_err(failed)
            }
            Err(error) => Err(failed(error)),
        }
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn annotations_with_read_only_hint(hint: Option<bool>) -> rmcp::model::ToolAnnotations {
        // `ToolAnnotations` is `#[non_exhaustive]`; use the builder.
        let mut annotations = rmcp::model::ToolAnnotations::new();
        annotations.read_only_hint = hint;
        annotations
    }

    #[test]
    fn resolve_tool_permission_prefers_per_tool_override() {
        let mut server = McpServerConfig::for_test("s");
        server.permission = Some(Permission::Unrestricted);
        let mut per_tool = std::collections::HashMap::new();
        per_tool.insert("search".to_string(), Permission::Read);
        server.tool_permissions = Some(per_tool);

        // Per-tool override wins even when both the server default AND the server's hint disagree.
        let annotations = annotations_with_read_only_hint(Some(false));
        let resolved = resolve_tool_permission(
            "search",
            Some(&annotations),
            &server,
            Some(Permission::Unrestricted),
        );
        assert_eq!(resolved, Permission::Read);
    }

    #[test]
    fn resolve_tool_permission_falls_through_to_server_level() {
        let mut server = McpServerConfig::for_test("s");
        server.permission = Some(Permission::Read);
        // Server level beats the hint.
        let annotations = annotations_with_read_only_hint(Some(false));
        let resolved = resolve_tool_permission(
            "any",
            Some(&annotations),
            &server,
            Some(Permission::Unrestricted),
        );
        assert_eq!(resolved, Permission::Read);
    }

    #[test]
    fn resolve_tool_permission_honors_read_only_hint() {
        let server = McpServerConfig::for_test("s");
        // readOnlyHint = true → Read, even though the global default would otherwise be
        // Unrestricted.
        let annotations = annotations_with_read_only_hint(Some(true));
        let resolved = resolve_tool_permission(
            "search",
            Some(&annotations),
            &server,
            Some(Permission::Unrestricted),
        );
        assert_eq!(resolved, Permission::Read);

        // readOnlyHint = false → Unrestricted, even though the global default is the lenient Read.
        let annotations = annotations_with_read_only_hint(Some(false));
        let resolved = resolve_tool_permission(
            "write-page",
            Some(&annotations),
            &server,
            Some(Permission::Read),
        );
        assert_eq!(resolved, Permission::Unrestricted);
    }

    /// `trust_read_only_hint = false` is the knob that keeps an unsandboxed MCP tool out of the
    /// `read` tier on a server whose self-classification the user does not accept. Without it, a
    /// server advertising `readOnlyHint: true` for a tool that in fact writes gets to write while
    /// The advertised-tool bound is a residency bound: the list is held per session and re-sent in
    /// every request's tools array, so an unbounded one is a per-turn cost, not a one-off.
    #[test]
    fn an_over_advertising_server_is_capped_at_the_tool_ceiling() {
        let under: Vec<usize> = (0..MAX_MCP_TOOLS_PER_SERVER).collect();
        assert_eq!(
            cap_advertised_tools(under, "s").len(),
            MAX_MCP_TOOLS_PER_SERVER,
            "a server exactly at the ceiling keeps everything"
        );

        let over: Vec<usize> = (0..MAX_MCP_TOOLS_PER_SERVER + 250).collect();
        let capped = cap_advertised_tools(over, "s");
        assert_eq!(capped.len(), MAX_MCP_TOOLS_PER_SERVER);
        assert_eq!(
            capped.first().copied(),
            Some(0),
            "the kept ones are the first, not an arbitrary slice"
        );
    }

    /// meka sits at `read`, because MCP tools run in the server's process with no sandbox.
    #[test]
    fn a_declined_read_only_hint_cannot_reach_the_read_tier() {
        let mut server = McpServerConfig::for_test("s");
        server.trust_read_only_hint = Some(false);
        let annotations = annotations_with_read_only_hint(Some(true));

        // Every global default, including the two that are themselves at or below `read`.
        //
        // Those two are the whole point. With `default_permission = "read"`, sending a refused hint
        // back to `Read` is bit-for-bit what trusting it would do, so the knob would change nothing
        // but a label. `"none"` was worse: a required level of `None` is permitted at every tier,
        // so the tool ran even at `--permission none`. This test asserted the invariant in its name
        // while only ever passing `Some(Unrestricted)` and `None`.
        for default in [
            Some(Permission::Unrestricted),
            Some(Permission::Read),
            Some(Permission::None),
            None,
        ] {
            let resolved = resolve_tool_permission("search", Some(&annotations), &server, default);
            assert_eq!(
                resolved,
                Permission::Unrestricted,
                "a declined hint must reach the strict fallback whatever the global default is, \
                 but with {default:?} it resolved to {resolved}"
            );
        }
    }

    /// The knob is per-server, so it has to beat the global default; the per-server *overrides*
    /// still beat it in turn.
    ///
    /// Without this second half the fix would be a lockout: a distrusted server's tool could never
    /// be brought back within reach of `read` at all. `tool_permissions` and `permission` are
    /// checked before the hint, and they remain the documented escape hatch.
    #[test]
    fn an_explicit_override_still_outranks_a_declined_hint() {
        let mut server = McpServerConfig::for_test("s");
        server.trust_read_only_hint = Some(false);
        let annotations = annotations_with_read_only_hint(Some(true));

        server.tool_permissions = Some(std::collections::HashMap::from([(
            "search".to_string(),
            Permission::Read,
        )]));
        let resolved = resolve_tool_permission(
            "search",
            Some(&annotations),
            &server,
            Some(Permission::Read),
        );
        assert_eq!(
            resolved,
            Permission::Read,
            "an explicit per-tool override is how a distrusted server's tool is re-admitted"
        );
    }

    /// Declining the hint withholds only the direction that *lowers* the requirement. A
    /// `readOnlyHint: false` can only ever raise it, so believing it costs nothing and it stays
    /// honored, including the attribution, so `meka mcp tools` still explains the classification.
    #[test]
    fn a_declined_read_only_hint_still_honors_the_raising_direction() {
        let mut server = McpServerConfig::for_test("s");
        server.trust_read_only_hint = Some(false);
        let annotations = annotations_with_read_only_hint(Some(false));

        let (resolved, source) = resolve_tool_permission_with_source(
            "write-page",
            Some(&annotations),
            &server,
            Some(Permission::Read),
        );
        assert_eq!(resolved, Permission::Unrestricted);
        assert_eq!(source, PermissionSource::ReadOnlyHint);
    }

    /// Every MCP round-trip has to answer to the turn's cancellation and to a clock.
    ///
    /// A server can accept a request and then go quiet, and the resource and prompt helpers awaited
    /// that unconditionally: the tool call parked, the turn with it, and pressing stop did not
    /// reach it either, because the token the tool was handed went unused. `call_tool_once` had
    /// both bounds from the start; these six were the asymmetry.
    #[tokio::test]
    async fn an_mcp_round_trip_answers_to_cancellation_and_to_the_clock() {
        let entry = pending_entry("quiet-srv", McpTransport::Http);

        let canceled = CancellationToken::new();
        canceled.cancel();
        let outcome = bounded(&entry, "resources/read", &canceled, async {
            std::future::pending::<Result<()>>().await
        })
        .await;
        assert!(
            matches!(outcome, Err(MekaError::Interrupted)),
            "a stopped turn must not wait on the server: {outcome:?}",
        );

        // And the clock, for a server nobody stopped waiting on.
        tokio::time::pause();
        let live = CancellationToken::new();
        let waiting = tokio::spawn({
            let entry = Arc::clone(&entry);
            async move {
                bounded(&entry, "resources/read", &live, async {
                    std::future::pending::<Result<()>>().await
                })
                .await
            }
        });
        tokio::time::advance(DEFAULT_MCP_REQUEST_TIMEOUT + std::time::Duration::from_secs(1)).await;
        let outcome = waiting.await.expect("join");
        assert!(
            matches!(outcome, Err(MekaError::McpConnection { .. })),
            "an unanswered request must end: {outcome:?}",
        );
    }

    /// A user who sets `trust_read_only_hint = false` has to be able to see what it moved.
    ///
    /// `meka mcp tools` reports the step that *won*, and a declined hint by definition did not, so
    /// a server advertising no hint at all and a server whose hint was refused both printed
    /// `default_permission` and read identically. The one place the setting is observable showed
    /// nothing of it.
    #[test]
    fn a_declined_read_only_hint_is_reported_as_declined() {
        let offered = annotations_with_read_only_hint(Some(true));

        assert!(
            read_only_hint_was_declined(Some(&offered), PermissionSource::GlobalDefault),
            "offered and outvoted by the global default is the case this exists for",
        );
        assert!(
            read_only_hint_was_declined(Some(&offered), PermissionSource::Fallback),
            "and the same when there is no global default to fall to",
        );
        assert!(
            !read_only_hint_was_declined(Some(&offered), PermissionSource::ReadOnlyHint),
            "a hint that won was not declined",
        );
        assert!(
            !read_only_hint_was_declined(None, PermissionSource::GlobalDefault),
            "and a server that offered nothing has nothing to decline",
        );
        assert!(
            !read_only_hint_was_declined(
                Some(&annotations_with_read_only_hint(Some(false))),
                PermissionSource::GlobalDefault,
            ),
            "only the lowering direction can be declined; `false` always wins when present",
        );
    }

    /// Declining the hint must not also discard the user's own overrides: steps 1 and 2 sit above
    /// the hint in the chain and are the documented way to make a distrusted server's tool usable.
    #[test]
    fn a_declined_read_only_hint_leaves_user_overrides_in_charge() {
        let mut server = McpServerConfig::for_test("s");
        server.trust_read_only_hint = Some(false);
        server.tool_permissions = Some(
            [("search".to_string(), Permission::Read)]
                .into_iter()
                .collect(),
        );
        let annotations = annotations_with_read_only_hint(Some(true));

        let (resolved, source) =
            resolve_tool_permission_with_source("search", Some(&annotations), &server, None);
        assert_eq!(resolved, Permission::Read);
        assert_eq!(source, PermissionSource::ToolOverride);
    }

    #[test]
    fn resolve_tool_permission_falls_through_to_mcp_default() {
        let server = McpServerConfig::for_test("s");
        // No user overrides, no hint → fall through to `[mcp].default`.
        let resolved = resolve_tool_permission("any", None, &server, Some(Permission::Read));
        assert_eq!(resolved, Permission::Read);
    }

    #[test]
    fn resolve_tool_permission_hardcoded_unrestricted_fallback() {
        let server = McpServerConfig::for_test("s");
        // Nothing configured anywhere, no hint → the hardcoded `Unrestricted` fallback.
        let resolved = resolve_tool_permission("any", None, &server, None);
        assert_eq!(resolved, Permission::Unrestricted);
    }

    #[test]
    fn resolve_tool_permission_with_source_attributes_each_step() {
        // 1. Per-tool override.
        let mut server = McpServerConfig::for_test("s");
        let mut per_tool = std::collections::HashMap::new();
        per_tool.insert("a".to_string(), Permission::Workspace);
        server.tool_permissions = Some(per_tool);
        let (permission, source) = resolve_tool_permission_with_source("a", None, &server, None);
        assert_eq!(permission, Permission::Workspace);
        assert_eq!(source, PermissionSource::ToolOverride);

        // 2. Server-level override.
        let mut server = McpServerConfig::for_test("s");
        server.permission = Some(Permission::Read);
        let (permission, source) = resolve_tool_permission_with_source("b", None, &server, None);
        assert_eq!(permission, Permission::Read);
        assert_eq!(source, PermissionSource::ServerOverride);

        // 3. readOnlyHint fires when no user override is set.
        let server = McpServerConfig::for_test("s");
        let annotations = annotations_with_read_only_hint(Some(true));
        let (permission, source) =
            resolve_tool_permission_with_source("c", Some(&annotations), &server, None);
        assert_eq!(permission, Permission::Read);
        assert_eq!(source, PermissionSource::ReadOnlyHint);

        // 4. Global default when no hint.
        let server = McpServerConfig::for_test("s");
        let (permission, source) =
            resolve_tool_permission_with_source("d", None, &server, Some(Permission::Read));
        assert_eq!(permission, Permission::Read);
        assert_eq!(source, PermissionSource::GlobalDefault);

        // 5. Hardcoded fallback.
        let server = McpServerConfig::for_test("s");
        let (permission, source) = resolve_tool_permission_with_source("e", None, &server, None);
        assert_eq!(permission, Permission::Unrestricted);
        assert_eq!(source, PermissionSource::Fallback);
    }

    #[test]
    fn permission_source_labels_match_config_keys() {
        // The labels printed by `meka mcp tools` must match the config keys users would edit to
        // change a classification.
        assert_eq!(PermissionSource::ToolOverride.as_str(), "tool_permission");
        assert_eq!(
            PermissionSource::ServerOverride.as_str(),
            "server_permission"
        );
        assert_eq!(PermissionSource::ReadOnlyHint.as_str(), "readOnlyHint");
        assert_eq!(
            PermissionSource::GlobalDefault.as_str(),
            "default_permission"
        );
        assert_eq!(PermissionSource::Fallback.as_str(), "fallback");
    }

    #[test]
    fn tool_is_allowed_default_passes_everything() {
        let server = McpServerConfig::for_test("s");
        assert!(tool_is_allowed(&server, "search"));
        assert!(tool_is_allowed(&server, "create-page"));
    }

    #[test]
    fn tool_is_allowed_allowlist_restricts() {
        let mut server = McpServerConfig::for_test("s");
        server.allowed_tools = Some(vec!["search".into(), "fetch".into()]);
        assert!(tool_is_allowed(&server, "search"));
        assert!(tool_is_allowed(&server, "fetch"));
        assert!(!tool_is_allowed(&server, "create-page"));
    }

    #[test]
    fn tool_is_allowed_empty_allowlist_means_all() {
        // An empty `allowed_tools` array is treated as "unset", i.e. no restriction. A totally
        // absent field behaves the same way.
        let mut server = McpServerConfig::for_test("s");
        server.allowed_tools = Some(Vec::new());
        assert!(tool_is_allowed(&server, "anything"));
    }

    #[test]
    fn tool_is_allowed_blocklist_removes() {
        let mut server = McpServerConfig::for_test("s");
        server.disabled_tools = Some(vec!["delete-page".into()]);
        assert!(tool_is_allowed(&server, "search"));
        assert!(!tool_is_allowed(&server, "delete-page"));
    }

    #[test]
    fn tool_is_allowed_both_lists_compose() {
        // allow restricts to {search, fetch, write-page}, then block subtracts {write-page}. Net
        // effect: only search + fetch.
        let mut server = McpServerConfig::for_test("s");
        server.allowed_tools = Some(vec!["search".into(), "fetch".into(), "write-page".into()]);
        server.disabled_tools = Some(vec!["write-page".into()]);
        assert!(tool_is_allowed(&server, "search"));
        assert!(tool_is_allowed(&server, "fetch"));
        assert!(!tool_is_allowed(&server, "write-page"));
        assert!(!tool_is_allowed(&server, "delete-page")); // not in allow
    }

    #[test]
    fn warn_on_stale_tool_config_smoke() {
        // The function just emits `warn!` lines; we can't easily assert on tracing output from a
        // unit test. Smoke-test that the happy path (empty config) doesn't panic and that it
        // accepts a server_config with all four list fields populated plus tool_permissions.
        let mut server = McpServerConfig::for_test("s");
        server.allowed_tools = Some(vec!["a".into(), "unknown".into()]);
        server.disabled_tools = Some(vec!["b".into(), "gone".into()]);
        server.eager_load_tools = Some(vec!["a".into(), "stale".into(), "b".into()]);
        let mut permissions = std::collections::HashMap::new();
        permissions.insert("a".to_string(), Permission::Read);
        permissions.insert("missing".to_string(), Permission::Unrestricted);
        server.tool_permissions = Some(permissions);

        let advertised: std::collections::HashSet<&str> =
            ["a", "b", "search"].into_iter().collect();
        // Just confirm the call doesn't panic; "stale" should warn (unknown), and "b" should warn
        // (disabled∩eager overlap).
        warn_on_stale_tool_config("s", &server, &advertised);
    }

    /// A server that states nothing defers every tool; an empty eager list changes nothing.
    #[test]
    fn a_tool_no_setting_names_is_deferred() {
        use crate::config::Deferral;
        let mut server = McpServerConfig::for_test("s");
        assert_eq!(tool_exposure(&server, "search"), Deferral::Deferred);
        server.eager_load_tools = Some(Vec::new());
        assert_eq!(tool_exposure(&server, "search"), Deferral::Deferred);
    }

    /// `eager_load_tools` names a tool eager whatever the server's `deferral` says, by its raw
    /// name alone: the namespaced `mcp__notion__search` must not match an entry of `search`, or
    /// one entry would reach across servers.
    #[test]
    fn a_named_tool_is_eager_under_every_deferral_and_only_by_its_raw_name() {
        use crate::config::Deferral;
        let mut server = McpServerConfig::for_test("notion");
        server.eager_load_tools = Some(vec!["search".into(), "fetch".into()]);
        for deferral in Deferral::ALL {
            server.deferral = Some(deferral);
            assert_eq!(
                tool_exposure(&server, "search"),
                Deferral::Eager,
                "{deferral}"
            );
            assert_eq!(
                tool_exposure(&server, "fetch"),
                Deferral::Eager,
                "{deferral}"
            );
            assert_eq!(
                tool_exposure(&server, "create-page"),
                deferral,
                "an unnamed tool takes the server's setting"
            );
            assert_eq!(tool_exposure(&server, "mcp__notion__search"), deferral);
        }
    }

    #[test]
    fn truncate_under_limit() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_at_limit() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn truncate_over_limit() {
        assert_eq!(truncate("hello world", 5), "hello...");
    }

    #[test]
    fn truncate_unicode_boundary() {
        // Three emoji, each multiple bytes: truncation should cut on char boundary.
        let input = "🦀🦀🦀🦀🦀";
        let out = truncate(input, 2);
        assert_eq!(out, "🦀🦀...");
    }

    /// Build a bare server entry in `Pending` state for pure-state tests. No network, no process
    /// spawn.
    /// The configured `connect_timeout` has to reach the request helpers, not just `tools/list`.
    ///
    /// `bounded` hardcoded the module default, so `[mcp].connect_timeout` governed
    /// discovery and silently not `resources/read`, `prompts/get` or any of the other four. An
    /// operator who raised it for a slow server still had those calls cut at the default, and one
    /// who lowered it still waited the default.
    #[tokio::test]
    async fn a_request_helper_waits_the_configured_timeout_not_the_default() {
        let entry = pending_entry("slow-srv", McpTransport::Http);
        let configured = std::time::Duration::from_secs(3);
        entry
            .request_timeout
            .set(configured)
            .expect("first and only set");
        assert_ne!(
            configured, DEFAULT_MCP_REQUEST_TIMEOUT,
            "the test is only meaningful while the two differ"
        );

        tokio::time::pause();
        let live = CancellationToken::new();
        let waiting = tokio::spawn({
            let entry = Arc::clone(&entry);
            async move {
                bounded(&entry, "resources/read", &live, async {
                    std::future::pending::<Result<()>>().await
                })
                .await
            }
        });

        // Measured, not merely awaited. A paused clock auto-advances to the next timer whenever
        // every task is idle, so *some* timeout always fires and asserting only on the error tells
        // the two apart not at all -- the default would fire too, just later in virtual time.
        // Elapsed virtual time is the thing that differs.
        let started = tokio::time::Instant::now();
        let outcome = waiting.await.expect("join");
        let waited = started.elapsed();

        assert!(
            matches!(outcome, Err(MekaError::McpConnection { .. })),
            "an unanswered request must end: {outcome:?}",
        );
        assert!(
            waited < DEFAULT_MCP_REQUEST_TIMEOUT,
            "waited {waited:?}, which is the module default rather than the configured {configured:?}",
        );
    }

    /// Every MCP tool declares itself unconfinable, which is the input the `workspace` gate needs.
    ///
    /// The gate in `crate::agent` refuses a tool at `workspace` when it both exceeds the level and
    /// `runs_outside_confinement()`. The trait default is the *permissive* answer (`false`), so the
    /// three-line override on `McpToolAdapter` is the only thing making a real MCP call qualify --
    /// delete it and the gate never fires, and `workspace` again forwards unannotated calls to a
    /// server process meka spawns but does not sandbox.
    ///
    /// Nothing asserted it. The agent-side test builds its own fixture tool with its own override,
    /// so it exercises the gate's logic and never touches the adapter that feeds it.
    #[test]
    fn an_mcp_tool_reports_itself_as_running_outside_confinement() {
        use crate::tools::Tool;

        let adapter = crate::tools::mcp_adapter::McpToolAdapter::new(Arc::new(McpTool {
            namespaced_name: "mcp__demo__do_thing".to_string(),
            remote_tool_name: "do_thing".to_string(),
            description: "does a thing".to_string(),
            parameters: serde_json::json!({"type": "object"}),
            permission: Permission::Read,
            entry: pending_entry("demo", McpTransport::Stdio),
            annotations: None,
            meta: None,
            title: None,
        }));
        assert!(
            adapter.runs_outside_confinement(),
            "an MCP call runs in the server's own process, which meka does not sandbox; the \
             `workspace` gate depends on this being true"
        );
    }

    fn pending_entry(name: &str, transport: McpTransport) -> Arc<ServerEntry> {
        let mut config = McpServerConfig::for_test(name);
        config.transport = transport;
        pending_entry_for(config)
    }

    fn pending_entry_for(config: McpServerConfig) -> Arc<ServerEntry> {
        Arc::new(ServerEntry {
            server_name: config.name.clone(),
            config,
            token_store: None,
            client_context: McpClientContext::new(),
            state: RwLock::new(ServerState::Pending),
            reconnect_lock: Mutex::new(()),
            tool_call_timeout: crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
            refused: None,
            instructions: std::sync::RwLock::new(None),
            request_timeout: OnceLock::new(),
            background_retry: OnceLock::new(),
            dropped_tools: std::sync::atomic::AtomicUsize::new(0),
            subscriptions: subscriptions::Subscriptions::default(),
            legacy_era: std::sync::atomic::AtomicBool::new(false),
            connects: Connects::new(),
        })
    }

    /// `eager_load_tools` decides the split, against a real `tools/list`.
    ///
    /// The sibling of `a_registry_attaching_after_discovery_still_learns_which_tools_are_deferred`,
    /// which hands [`ServerTools`] its `deferred` list ready-made and so proves only that the list
    /// is *carried*. This proves it is *computed*: the classification needs the raw name and the
    /// server config, neither of which survives the erasure into `Arc<dyn Tool>`, so it has to
    /// happen inside `from_tools` on the discovery path.
    ///
    /// Deliberately not a `#[cfg(test)]` construction of an adapter: the thing that broke on
    /// `serve` and `acp` was the *path*, so this drives `connect_one` against a live stdio peer
    /// with no registry attached at discovery time, which is the ordering both hosts use.
    #[cfg(unix)]
    #[tokio::test]
    async fn discovery_defers_every_tool_the_eager_allowlist_does_not_name() {
        let temp = tempfile::tempdir().expect("tempdir");
        let state = temp.path().join("stub.json");
        std::fs::write(&state, r#"{"tools": ["search", "create_page"]}"#)
            .expect("write the stub's state");

        let mut config = McpServerConfig::for_test("notion");
        config.transport = McpTransport::Stdio;
        config.url = None;
        config.command = Some("python3".to_string());
        config.args = Some(vec![
            format!(
                "{}/tests/fixtures/mcp_stub_server.py",
                env!("CARGO_MANIFEST_DIR")
            ),
            state.display().to_string(),
        ]);
        config.eager_load_tools = Some(vec!["search".to_string()]);

        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(
            &[config],
            None,
            None,
            Arc::clone(&context),
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("prepare");
        context.set_manager(Arc::downgrade(&manager));
        let entry = manager
            .servers
            .get("notion")
            .cloned()
            .expect("the configured entry");

        // No registry attached yet: `build_shared_deps` starts the connector before any session
        // exists, so this is the ordering that made the marks vanish.
        connector::connect_one(
            Arc::clone(&entry),
            Arc::clone(&manager),
            None,
            std::time::Duration::from_secs(20),
        )
        .await;

        let registry = crate::tools::ToolRegistry::new();
        crate::tools::mcp_adapter::attach_session_registry(&manager, registry.clone()).await;

        assert!(
            registry.get("mcp__notion__search").is_some()
                && registry.get("mcp__notion__create_page").is_some(),
            "both tools must arrive, or the split below means nothing: {:?}",
            entry.state().await.label()
        );
        assert!(
            !registry.is_deferred("mcp__notion__search"),
            "`eager_load_tools` names `search`, so it must ship in the cacheable prefix"
        );
        assert!(
            registry.is_deferred("mcp__notion__create_page"),
            "and everything the allowlist does not name must ship deferred"
        );
    }

    /// The server's `deferral` decides what `eager_load_tools` does not name, on the same discovery
    /// path: `eager` declares both tools, `hidden` registers neither, and a named tool is declared
    /// under either.
    #[cfg(unix)]
    #[tokio::test]
    async fn discovery_follows_the_server_s_deferral_for_the_tools_it_does_not_name() {
        for deferral in [
            crate::config::Deferral::Eager,
            crate::config::Deferral::Hidden,
        ] {
            let temp = tempfile::tempdir().expect("tempdir");
            let state = temp.path().join("stub.json");
            std::fs::write(&state, r#"{"tools": ["search", "create_page"]}"#)
                .expect("write the stub's state");
            let mut config = McpServerConfig::for_test("notion");
            config.transport = McpTransport::Stdio;
            config.url = None;
            config.command = Some("python3".to_string());
            config.args = Some(vec![
                format!(
                    "{}/tests/fixtures/mcp_stub_server.py",
                    env!("CARGO_MANIFEST_DIR")
                ),
                state.display().to_string(),
            ]);
            config.eager_load_tools = Some(vec!["search".to_string()]);
            config.deferral = Some(deferral);
            let context = McpClientContext::new();
            let manager = McpClientManager::prepare(
                &[config],
                None,
                None,
                Arc::clone(&context),
                crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
            )
            .await
            .expect("prepare");
            context.set_manager(Arc::downgrade(&manager));
            let entry = manager
                .servers
                .get("notion")
                .cloned()
                .expect("the configured entry");
            connector::connect_one(
                Arc::clone(&entry),
                Arc::clone(&manager),
                None,
                std::time::Duration::from_secs(20),
            )
            .await;
            let registry = crate::tools::ToolRegistry::new();
            crate::tools::mcp_adapter::attach_session_registry(&manager, registry.clone()).await;
            assert!(
                registry.get("mcp__notion__search").is_some()
                    && !registry.is_deferred("mcp__notion__search"),
                "{deferral}: the named tool is declared: {:?}",
                entry.state().await.label()
            );
            match deferral {
                crate::config::Deferral::Eager => assert!(
                    registry.get("mcp__notion__create_page").is_some()
                        && !registry.is_deferred("mcp__notion__create_page"),
                    "eager declares the unnamed tool too"
                ),
                crate::config::Deferral::Hidden => assert!(
                    registry.get("mcp__notion__create_page").is_none(),
                    "hidden registers nothing it was not named"
                ),
                crate::config::Deferral::Deferred => unreachable!("not in the loop"),
            }
        }
    }
    /// A reconnect is a new session with the server, and its tool set is only knowable by asking.
    ///
    /// Ending the reconnect path at the transport is justified by "the tool adapters already exist
    /// and resolve the live peer at dispatch time", which is true of dispatch and not of the list.
    /// A fresh `initialize` produces no `tools/list_changed`, because the client is the one
    /// expected to list, so a server redeployed with a tool dropped kept being advertised for the
    /// life of the process and one added was never learned.
    ///
    /// Driven against a real peer, because the whole claim is about what happens on the wire. The
    /// stub serves one `tools/list` and exits, which both delivers the first tool set and closes
    /// the transport under meka, and it re-reads its state file on every request, so the second
    /// connection advertises the second set.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_reconnect_re_lists_a_server_whose_tools_have_changed() {
        let temp = tempfile::tempdir().expect("tempdir");
        let state = temp.path().join("stub.json");
        let write_state = |tools: &str| {
            std::fs::write(
                &state,
                format!(r#"{{"tools": [{tools}], "exit_after": 1}}"#),
            )
            .expect("write the stub's state")
        };
        write_state(r#""search""#);

        let mut config = McpServerConfig::for_test("stub");
        config.transport = McpTransport::Stdio;
        config.url = None;
        config.command = Some("python3".to_string());
        config.args = Some(vec![
            format!(
                "{}/tests/fixtures/mcp_stub_server.py",
                env!("CARGO_MANIFEST_DIR")
            ),
            state.display().to_string(),
        ]);

        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(
            &[config],
            None,
            None,
            Arc::clone(&context),
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("prepare");
        context.set_manager(Arc::downgrade(&manager));
        let registry = crate::tools::ToolRegistry::new();
        crate::tools::mcp_adapter::attach_session_registry(&manager, registry.clone()).await;
        let entry = manager
            .servers
            .get("stub")
            .cloned()
            .expect("the configured entry");

        connector::connect_one(
            Arc::clone(&entry),
            Arc::clone(&manager),
            None,
            std::time::Duration::from_secs(20),
        )
        .await;
        assert!(
            registry.get("mcp__stub__search").is_some(),
            "the stub's first tool set must land, or nothing below means anything: {:?}",
            entry.state().await.label()
        );

        // The stub exited after serving that list, so the transport is closing. What the *next*
        // connection advertises is written before the reconnect that spawns it.
        write_state(r#""create_page""#);
        for _ in 0..200 {
            if entry.needs_reconnect().await {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(
            entry.needs_reconnect().await,
            "the stub was supposed to exit and close the transport"
        );

        entry
            .reconnect(&tokio_util::sync::CancellationToken::new())
            .await
            .expect("reconnect");

        assert!(
            registry.get("mcp__stub__create_page").is_some(),
            "a reconnect must learn the tool the redeployed server added"
        );
        assert!(
            registry.get("mcp__stub__search").is_none(),
            "and must stop advertising the one it dropped"
        );
    }

    /// Instructions belong to a connection, and the entry outlives the connection.
    ///
    /// This was a `OnceLock`, justified by the spec's "immutable for the lifetime of the
    /// connection". A reconnect is a new connection, so a server redeployed with different
    /// instructions -- or with none -- kept feeding the first handshake's text into the model's
    /// context every turn for the life of the process.
    #[test]
    fn a_later_handshake_replaces_what_the_previous_one_said() {
        let entry = pending_entry("notion", McpTransport::Http);

        entry.record_instructions(Some("call search first".to_string()));
        assert_eq!(entry.instructions().as_deref(), Some("call search first"));

        entry.record_instructions(Some("search was removed".to_string()));
        assert_eq!(
            entry.instructions().as_deref(),
            Some("search was removed"),
            "a second handshake's instructions must supersede the first's"
        );

        entry.record_instructions(None);
        assert_eq!(
            entry.instructions(),
            None,
            "a handshake that advertises none must clear the previous text rather than leave it \
             standing as if it had been repeated"
        );
    }

    /// Shutdown must run while the manager is still shared.
    ///
    /// Consuming `self` would make the caller win an `Arc::try_unwrap` first, which it never can:
    /// the manager holds the registries it serves, and each of those holds six `mcp_resource_*` /
    /// `mcp_prompt_*` tools that hold the manager back. Sole ownership was unreachable by
    /// construction, so the close handshake never ran on any launch that had an MCP server to
    /// close, and every exit warned about it instead.
    ///
    /// The cycle is built here rather than described, so a future change that reintroduces it is
    /// caught: attaching a session's registry puts the registry inside the manager and tools
    /// holding the manager inside that registry, and shutdown still has to reach the entries.
    #[tokio::test]
    async fn shutdown_runs_while_the_manager_is_still_shared() {
        let manager = McpClientManager::prepare(
            &[McpServerConfig::for_test("probe")],
            None,
            None,
            McpClientContext::new(),
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("prepare");
        let registry = crate::tools::ToolRegistry::new();
        crate::tools::mcp_adapter::attach_session_registry(&manager, registry.clone()).await;

        // The cycle a `try_unwrap` would lose to.
        assert!(
            Arc::strong_count(&manager) > 1,
            "the manager must be shared for this test to mean anything"
        );

        // Runs anyway. That this compiles at all is half the guard: `shutdown` taking `&self` is
        // what makes it reachable, and a change back to `self` would fail here rather than silently
        // restoring the warn-and-skip behavior at runtime.
        manager.shutdown().await;

        // A `Pending` entry has no service, so it is left as it was; only `Connected` entries carry
        // something to close, and those are left `Disabled` once their service has been taken.
        let entry = manager.server_entry("probe").expect("entry");
        assert!(matches!(entry.state().await, ServerState::Pending));
    }

    /// A server that never connected registers no tools, so its names reach the agent's
    /// unknown-tool arm. Answering "unknown" would be false and would teach the agent the
    /// capability is gone; this reports the server's state instead.
    #[tokio::test]
    async fn unavailable_tool_reason_names_the_server_state() {
        let manager = McpClientManager::prepare(
            &[McpServerConfig::for_test("ida")],
            None,
            None,
            McpClientContext::new(),
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("prepare");

        let reason = manager
            .unavailable_tool_reason("mcp__ida__decompile")
            .await
            .expect("a configured server must explain itself");
        assert!(reason.contains("ida"), "{reason}");
        assert!(reason.contains("still connecting"), "{reason}");

        // Failed reads differently from Pending: the two call for opposite behavior, and
        // collapsing them yields an agent that either gives up early or retries forever.
        *manager
            .server_entry("ida")
            .expect("entry")
            .state
            .write()
            .await = ServerState::Failed {
            error: "'ida-mcp' not found".to_string(),
            at: std::time::Instant::now(),
        };
        let reason = manager
            .unavailable_tool_reason("mcp__ida__decompile")
            .await
            .expect("failed server must explain itself");
        assert!(reason.contains("unavailable"), "{reason}");
        assert!(reason.contains("'ida-mcp' not found"), "{reason}");
    }

    /// Sub-agent registries come through `install_tools_on`, not `attach_registry`, so they need
    /// the manager back-reference wired there too - otherwise a sub-agent reaching for a dead
    /// server's tool gets a bare "not registered" while the parent gets the reason.
    #[tokio::test]
    async fn subagent_registry_also_explains_an_unconnected_server() {
        use crate::tools::ToolRegistry;

        let manager = McpClientManager::prepare(
            &[McpServerConfig::for_test("ida")],
            None,
            None,
            McpClientContext::new(),
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("prepare");

        let registry = ToolRegistry::new();
        registry.register_load_tool_for_test();
        crate::tools::mcp_adapter::install_on_worker_registry(&manager, &registry).await;

        let tool_load = registry.get("tool_load").expect("tool_load registered");
        let output = tool_load
            .execute(
                serde_json::json!({"name": "mcp__ida__decompile"}),
                crate::tools::ToolContext::detached(tokio_util::sync::CancellationToken::new()),
            )
            .await
            .expect("tool_load returns Ok with an error payload");
        let text = format!("{:?}", output.content);
        assert!(text.contains("ida"), "{text}");
        assert!(!text.contains("not registered"), "{text}");

        // The same back-reference is what `Agent::resolve_and_execute_tool` reads when the model
        // calls the tool outright instead of loading it: a sub-agent has no `mcp_manager` of its
        // own, so the registry is the only route to the reason.
        let from_registry = registry
            .mcp_manager()
            .expect("install_tools_on must record the manager")
            .unavailable_tool_reason("mcp__ida__decompile")
            .await;
        assert!(from_registry.is_some_and(|reason| reason.contains("ida")));
    }

    /// `tool_load` is the path a model actually takes: the tool is absent from its catalog, so
    /// it reaches for the documented way to load a deferred tool first. Found by watching a real
    /// model do exactly that and get "not registered" back.
    #[tokio::test]
    async fn load_tool_explains_an_unconnected_server() {
        use crate::tools::ToolRegistry;

        let manager = McpClientManager::prepare(
            &[McpServerConfig::for_test("ida")],
            None,
            None,
            McpClientContext::new(),
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("prepare");

        let registry = ToolRegistry::new();
        registry.register_load_tool_for_test();
        crate::tools::mcp_adapter::attach_session_registry(&manager, registry.clone()).await;

        let tool_load = registry.get("tool_load").expect("tool_load registered");
        let output = tool_load
            .execute(
                serde_json::json!({"name": "mcp__ida__decompile"}),
                crate::tools::ToolContext::detached(tokio_util::sync::CancellationToken::new()),
            )
            .await
            .expect("tool_load returns Ok with an error payload");

        assert!(output.is_error);
        let text = format!("{:?}", output.content);
        assert!(text.contains("ida"), "{text}");
        assert!(text.contains("still connecting"), "{text}");
        assert!(!text.contains("not registered"), "{text}");
    }

    /// Genuinely unknown names must stay unknown, or the agent loses the signal that it invented
    /// a tool.
    #[tokio::test]
    async fn unavailable_tool_reason_ignores_non_mcp_and_unconfigured_names() {
        let manager = McpClientManager::prepare(
            &[McpServerConfig::for_test("ida")],
            None,
            None,
            McpClientContext::new(),
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("prepare");

        for name in [
            "file_read",
            "memory_write",
            "mcp__nosuch__tool",
            "mcp__malformed",
            "mcp__",
        ] {
            assert!(
                manager.unavailable_tool_reason(name).await.is_none(),
                "'{name}' must fall through to unknown-tool"
            );
        }
    }

    /// The gate needs to know which unavailable servers actually stop a turn.
    #[tokio::test]
    async fn enabled_not_connected_reports_required() {
        let mut optional = McpServerConfig::for_test("ida");
        optional.required = Some(false);
        let mut gating = McpServerConfig::for_test("bridge");
        gating.required = Some(true);

        let manager = McpClientManager::prepare(
            &[optional, gating],
            None,
            None,
            McpClientContext::new(),
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("prepare");

        let not_ready = manager.enabled_not_connected().await;
        assert_eq!(not_ready.len(), 2);
        let bridge = not_ready
            .iter()
            .find(|s| s.name == "bridge")
            .expect("bridge listed");
        assert!(bridge.required);
        let ida = not_ready
            .iter()
            .find(|s| s.name == "ida")
            .expect("ida listed");
        assert!(!ida.required);
    }

    #[tokio::test]
    async fn require_connected_errors_for_pending() {
        let entry = pending_entry("pending-srv", McpTransport::Http);
        let err = entry
            .require_connected()
            .await
            .expect_err("pending should not yield a peer");
        match err {
            MekaError::McpConnection {
                server_name,
                message,
            } => {
                assert_eq!(server_name, "pending-srv");
                assert!(message.contains("connecting"), "got: {message}");
            }
            other => panic!("expected McpConnection, got: {other:?}"),
        }
    }

    /// A call to a server still connecting is refused in the state's own words, ahead of the
    /// retry that reconnects a closed transport: reported as a closed transport, a server that
    /// never came up reads as a fault in the connection rather than as what it is.
    #[tokio::test]
    async fn a_call_to_a_server_still_connecting_is_refused_in_its_own_words() {
        let tool = handler::McpTool {
            namespaced_name: "mcp__pending-srv__ping".to_string(),
            remote_tool_name: "ping".to_string(),
            description: String::new(),
            parameters: serde_json::json!({"type": "object"}),
            permission: Permission::Read,
            entry: pending_entry("pending-srv", McpTransport::Http),
            annotations: None,
            meta: None,
            title: None,
        };
        let context = handler::CallContext {
            session_id: None,
            tool_call_id: None,
            frontend: Arc::new(crate::frontend::testing::RecordingFrontend::new()),
            cancellation: tokio_util::sync::CancellationToken::new(),
        };
        let error = tool
            .call(None, &context)
            .await
            .expect_err("a server still connecting takes no call");
        match error {
            MekaError::McpConnection {
                server_name,
                message,
            } => {
                assert_eq!(server_name, "pending-srv");
                assert!(message.contains("connecting"), "got: {message}");
            }
            other => panic!("expected McpConnection, got: {other:?}"),
        }
    }

    /// A stdio entry whose child never speaks MCP: the connect waits on `initialize` until the
    /// child exits.
    #[cfg(unix)]
    fn silent_stdio_entry(name: &str) -> Arc<ServerEntry> {
        let mut config = McpServerConfig::for_test(name);
        config.transport = McpTransport::Stdio;
        config.url = None;
        config.command = Some("sleep".to_string());
        config.args = Some(vec!["3".to_string()]);
        Arc::new(ServerEntry {
            server_name: name.to_string(),
            config,
            token_store: None,
            client_context: McpClientContext::new(),
            state: RwLock::new(ServerState::Pending),
            reconnect_lock: Mutex::new(()),
            tool_call_timeout: crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
            refused: None,
            instructions: std::sync::RwLock::new(None),
            request_timeout: OnceLock::new(),
            background_retry: OnceLock::new(),
            dropped_tools: std::sync::atomic::AtomicUsize::new(0),
            subscriptions: subscriptions::Subscriptions::default(),
            legacy_era: std::sync::atomic::AtomicBool::new(false),
            connects: Connects::new(),
        })
    }

    /// A reconnect is bounded, and leaves the entry `Failed` when it gives up. Unbounded, a peer
    /// that accepts the connection and never answers `initialize` holds the turn for as long as it
    /// pleases; a give-up that leaves the state `Connected` over the dead transport has `meka mcp
    /// list` say connected and every later call pay the whole backoff again.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_reconnect_gives_up_within_its_bound_and_leaves_the_entry_failed() {
        let entry = silent_stdio_entry("silent");
        entry
            .request_timeout
            .set(std::time::Duration::from_millis(200))
            .expect("unset");
        let started = std::time::Instant::now();
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            entry.reconnect_attempts(&tokio_util::sync::CancellationToken::new()),
        )
        .await
        .expect("the bound ends the attempt, not the test's deadline");
        assert!(
            matches!(outcome, Err(MekaError::McpConnection { .. })),
            "{outcome:?}"
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "the bound, not the child's own exit, ended the attempt: {:?}",
            started.elapsed()
        );
        assert!(
            matches!(
                entry.state().await,
                ServerState::Failed { ref error, .. } if error.contains("timed out")
            ),
            "giving up is recorded, so the background retry owns the entry"
        );
    }

    /// A stop reaches a reconnect in flight: the call's token ends the wait at once, rather than
    /// after the bound, so Ctrl+C and `session/cancel` land during a reconnect as they land
    /// during the call itself.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_stop_ends_a_reconnect_at_once() {
        let entry = silent_stdio_entry("silent");
        entry
            .request_timeout
            .set(std::time::Duration::from_secs(10))
            .expect("unset");
        let cancellation = tokio_util::sync::CancellationToken::new();
        let stop = tokio::spawn({
            let cancellation = cancellation.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                cancellation.cancel();
            }
        });
        let started = std::time::Instant::now();
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            entry.reconnect_attempts(&cancellation),
        )
        .await
        .expect("the stop ends the attempt, not the test's deadline");
        stop.await.expect("the stop landed");
        assert!(
            matches!(outcome, Err(MekaError::Interrupted)),
            "{outcome:?}"
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "{:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn require_connected_errors_for_failed() {
        let entry = pending_entry("failed-srv", McpTransport::Http);
        *entry.state.write().await = ServerState::Failed {
            error: "boom".to_string(),
            at: std::time::Instant::now(),
        };
        let err = entry.require_connected().await.unwrap_err();
        assert!(matches!(err, MekaError::McpConnection { .. }));
    }

    #[tokio::test]
    async fn require_connected_errors_for_disabled() {
        let entry = pending_entry("off-srv", McpTransport::Http);
        *entry.state.write().await = ServerState::Disabled;
        let err = entry.require_connected().await.unwrap_err();
        match err {
            MekaError::McpConnection { message, .. } => assert!(message.contains("disabled")),
            other => panic!("expected McpConnection, got: {other:?}"),
        }
    }

    #[test]
    fn server_state_label_matches_variant() {
        assert_eq!(ServerState::Pending.label(), "pending");
        assert_eq!(ServerState::Disabled.label(), "disabled");
        assert_eq!(
            ServerState::Failed {
                error: "x".into(),
                at: std::time::Instant::now()
            }
            .label(),
            "failed"
        );
    }

    /// The per-call cap is the config's, carried on the entry the adapter holds: one that stopped
    /// at `prepare` would leave every call on a default nobody set.
    #[tokio::test]
    async fn the_tool_call_timeout_reaches_the_server_entry() {
        let manager = McpClientManager::prepare(
            &[McpServerConfig::for_test("srv")],
            None,
            None,
            McpClientContext::new(),
            std::time::Duration::from_secs(7),
        )
        .await
        .expect("prepare");
        assert_eq!(
            manager
                .server_entry("srv")
                .expect("entry")
                .tool_call_timeout,
            std::time::Duration::from_secs(7)
        );
    }

    #[tokio::test]
    async fn prepare_all_disabled_publishes_settled_immediately() {
        let mut config = McpServerConfig::for_test("off");
        config.disabled = Some(true);
        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(
            &[config],
            None,
            None,
            context,
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("prepare should succeed with a disabled-only config");
        assert!(manager.all_ready(), "manager should be settled immediately");
        let not_ready = manager.enabled_not_connected().await;
        assert!(
            not_ready.is_empty(),
            "disabled servers don't count as not-ready"
        );
    }

    #[tokio::test]
    async fn prepare_pending_entries_not_ready_until_connector_runs() {
        let config = McpServerConfig::for_test("waiting");
        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(
            &[config],
            None,
            None,
            context,
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("prepare should succeed");
        assert!(
            !manager.all_ready(),
            "pending server shouldn't be ready yet"
        );
        let not_ready = manager.enabled_not_connected().await;
        assert_eq!(not_ready.len(), 1);
        assert_eq!(not_ready[0].name, "waiting");
    }

    #[tokio::test]
    async fn await_settled_returns_immediately_when_already_settled() {
        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(
            &[],
            None,
            None,
            context,
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("prepare with no servers should succeed");
        let res = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            manager.await_settled(),
        )
        .await;
        assert!(
            res.is_ok(),
            "await_settled blocked past the no-pending fast path"
        );
    }

    #[tokio::test]
    async fn await_settled_unblocks_when_connector_finishes() {
        // `/bin/false` exits immediately, so the connector reaches `settled.send(true)` via Failed
        // state on the first entry.
        let mut config = McpServerConfig::for_test("quick-fail");
        config.transport = McpTransport::Stdio;
        config.command = Some("/bin/false".to_string());
        config.url = None;

        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(
            &[config],
            None,
            None,
            context,
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("prepare should succeed");
        assert!(!manager.all_ready());

        manager.start_connector(McpRuntimeConfig {
            connect_timeout: std::time::Duration::from_secs(2),
            stdio_concurrency: 1,
            http_concurrency: 1,
        });

        let res =
            tokio::time::timeout(std::time::Duration::from_secs(5), manager.await_settled()).await;
        assert!(
            res.is_ok(),
            "await_settled didn't unblock after connector finished"
        );
        assert!(manager.all_ready());

        let entry = manager.server_entry("quick-fail").expect("entry");
        let state = entry.state().await;
        assert!(
            matches!(state, ServerState::Failed { .. }),
            "expected Failed, got: {}",
            state.label()
        );
    }

    /// Sub-agent registry inherits the parent's MCP resource / prompt meta-tools, even when no
    /// server is connected yet. The per-server adapters only show up for `Connected` servers; that
    /// case is covered separately by manual verification since spinning up a real stdio MCP server
    /// here is heavy.
    #[tokio::test]
    async fn install_tools_on_registers_resource_meta_tools() {
        let mut config = McpServerConfig::for_test("subagent-fixture");
        // Disable so `prepare` skips entirely without spawning a connector. `server_names()` still
        // includes it, which is all `register_all` needs to gate on.
        config.disabled = Some(true);

        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(
            &[config],
            None,
            None,
            context,
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("prepare should succeed for a disabled server");

        let registry = crate::tools::ToolRegistry::new();
        crate::tools::mcp_adapter::install_on_worker_registry(&manager, &registry).await;

        for name in [
            "mcp_resource_list",
            "mcp_resource_read",
            "mcp_prompt_list",
            "mcp_prompt_get",
            "mcp_resource_subscribe",
            "mcp_resource_unsubscribe",
            "mcp_resource_updates_list",
        ] {
            assert!(
                registry.get(name).is_some(),
                "expected '{name}' on sub-agent registry after install_tools_on"
            );
        }
    }

    /// With zero servers configured, `register_all`'s `server_names().is_empty()` guard kicks in
    /// and nothing is registered.
    #[tokio::test]
    async fn install_tools_on_noop_without_servers() {
        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(
            &[],
            None,
            None,
            context,
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("prepare with no servers should succeed");

        let registry = crate::tools::ToolRegistry::new();
        crate::tools::mcp_adapter::install_on_worker_registry(&manager, &registry).await;

        assert!(
            registry.get("mcp_resource_list").is_none(),
            "no MCP meta-tools should land on the registry when no servers configured"
        );
    }

    /// A tool with a chosen name and nothing else. An empty `Vec` to `update_server_tools` is a
    /// no-op and would not exercise the propagation path at all, so the fixture has to publish
    /// something distinctively named.
    fn fixture_tool(name: &str) -> Arc<McpTool> {
        Arc::new(McpTool {
            namespaced_name: name.to_string(),
            remote_tool_name: name.to_string(),
            description: "fixture".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
            permission: Permission::Read,
            entry: pending_entry("fixture", McpTransport::Stdio),
            annotations: None,
            meta: None,
            title: None,
        })
    }

    /// Discovery happens before any session exists on `meka serve` and `meka acp` --
    /// `start_connector` runs in `build_shared_deps`, `attach_registry` at `session/new` -- so a
    /// deferred mark that is only pushed to the registries attached *at discovery time* reaches
    /// nothing on either host. Lazy MCP loading was silently inert there: every `mcp__*` schema
    /// shipped on every request, and `tool_catalog` called them enabled.
    ///
    /// The sibling of `attach_registry_races_with_update_without_losing_tools`, which asserts the
    /// tools arrive and says nothing about how they arrive marked.
    #[tokio::test]
    async fn a_registry_attaching_after_discovery_still_learns_which_tools_are_deferred() {
        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(
            &[],
            None,
            None,
            context,
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("prepare");

        manager
            .update_server_tools("notion", ServerTools {
                tools: vec![
                    fixture_tool("mcp__notion__search"),
                    fixture_tool("mcp__notion__create_page"),
                ],
                deferred: vec!["mcp__notion__create_page".to_string()],
            })
            .await;

        // Attached only now, which is the ordering both long-lived hosts actually use.
        let registry = crate::tools::ToolRegistry::new();
        crate::tools::mcp_adapter::attach_session_registry(&manager, registry.clone()).await;

        assert!(
            registry.get("mcp__notion__create_page").is_some(),
            "the backfill must still deliver the tool itself"
        );
        assert!(
            registry.is_deferred("mcp__notion__create_page"),
            "a tool discovered before this registry existed must still arrive deferred"
        );
        assert!(
            !registry.is_deferred("mcp__notion__search"),
            "and an eager one must not be swept up with it"
        );
    }

    /// `update_server_tools` racing against `attach_registry` must not lose updates: every
    /// published tool list must reach every session that attaches before or during the publish,
    /// with no silent miss window. Regression guard for the race fixed in
    /// [`McpClientManager::subscribe`] where the original "read snapshot → push registry"
    /// order let updates land in the gap.
    #[tokio::test]
    async fn attach_registry_races_with_update_without_losing_tools() {
        use std::sync::Arc;

        // Empty config: we don't need real servers to exercise the snapshot/registry plumbing,
        // just the manager methods.
        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(
            &[],
            None,
            None,
            context,
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("prepare");

        let server_names: Vec<String> = (0..4).map(|index| format!("srv-{index}")).collect();
        let registry_count = 8;
        let registries: Vec<crate::tools::ToolRegistry> = (0..registry_count)
            .map(|_| crate::tools::ToolRegistry::new())
            .collect();

        // Each updater publishes one tool named mcp__<server>__ping.
        let mut update_handles = Vec::new();
        for name in &server_names {
            let manager = Arc::clone(&manager);
            let name = name.clone();
            update_handles.push(tokio::spawn(async move {
                manager
                    .update_server_tools(&name, ServerTools {
                        tools: vec![fixture_tool(&format!("mcp__{name}__ping"))],
                        deferred: Vec::new(),
                    })
                    .await;
            }));
        }
        let mut attach_handles = Vec::new();
        for registry in &registries {
            let manager = Arc::clone(&manager);
            let registry = registry.clone();
            attach_handles.push(tokio::spawn(async move {
                crate::tools::mcp_adapter::attach_session_registry(&manager, registry).await;
            }));
        }

        for handle in update_handles {
            handle.await.expect("update task");
        }
        for handle in attach_handles {
            handle.await.expect("attach task");
        }

        // The snapshot is the source of truth for "what got published".  Every server's
        // update must land there, and every registry must hold every server's tool.
        let snapshot_keys: std::collections::HashSet<String> = manager
            .tools_snapshot
            .read()
            .await
            .keys()
            .cloned()
            .collect();
        assert_eq!(
            snapshot_keys.len(),
            server_names.len(),
            "every update_server_tools call should land in the snapshot",
        );
        for registry in &registries {
            for server in &server_names {
                let tool_name = format!("mcp__{server}__ping");
                assert!(
                    registry.get(&tool_name).is_some(),
                    "registry missing '{tool_name}' after concurrent attach/update: race regressed",
                );
            }
        }
    }

    /// A `${VAR}` left unresolved in a header names a credential the operator kept out of the
    /// file. Connected anyway, the literal `Bearer ${TOKEN}` went to the third party; the server is
    /// refused up front instead, the way `config.toml`'s own `${VAR}` fails closed.
    #[tokio::test]
    async fn an_unresolved_secret_refuses_the_server_before_it_connects() {
        let config: McpServerConfig = toml::from_str(
            "name = \"tenant\"\ntransport = \"http\"\nurl = \"https://example.test/mcp\"\n[headers]\nAuthorization = \"Bearer ${MEKA_TEST_UNSET_SECRET_TOKEN}\"\n",
        )
        .expect("the fixture parses");
        let manager = McpClientManager::prepare(
            &[config],
            None,
            None,
            McpClientContext::new(),
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("prepare");
        let entry = manager.server_entry("tenant").expect("entry");
        let state = entry.state().await;
        assert!(
            matches!(state, ServerState::Failed { ref error, .. } if error.contains("MEKA_TEST_UNSET_SECRET_TOKEN")),
            "expected a refusal naming the variable, got: {}",
            state.label()
        );
    }

    /// A frontend that accepts every elicitation with one answer, the way a user filling the form
    /// would.
    #[cfg(unix)]
    struct AcceptingFrontend(serde_json::Value);

    #[cfg(unix)]
    #[async_trait::async_trait]
    impl crate::frontend::Frontend for AcceptingFrontend {
        async fn emit(&self, _event: crate::frontend::FrontendEvent) {}

        async fn request_permission(
            &self,
            _request: crate::frontend::PermissionRequest,
        ) -> crate::frontend::PermissionOutcome {
            crate::frontend::PermissionOutcome::Allow
        }

        async fn handle_elicitation(
            &self,
            _prompt: crate::frontend::ElicitationPrompt,
        ) -> crate::frontend::ElicitationResponse {
            crate::frontend::ElicitationResponse::Accept {
                content: Some(self.0.clone()),
            }
        }
    }

    /// One stdio entry driving the stub under `tests/fixtures`, prepared and attached to a
    /// registry but not connected, with the state the stub is to serve and a log of what it is
    /// sent.
    #[cfg(unix)]
    struct StubServer {
        state: std::path::PathBuf,
        log: std::path::PathBuf,
        context: Arc<McpClientContext>,
        manager: Arc<McpClientManager>,
        entry: Arc<ServerEntry>,
        registry: crate::tools::ToolRegistry,
        _temp: tempfile::TempDir,
    }

    #[cfg(unix)]
    impl StubServer {
        async fn prepare(state: serde_json::Value) -> Self {
            let temp = tempfile::tempdir().expect("tempdir");
            let path = temp.path().join("stub.json");
            let log = temp.path().join("requests.jsonl");
            let mut config = McpServerConfig::for_test("stub");
            config.transport = McpTransport::Stdio;
            config.url = None;
            config.command = Some("python3".to_string());
            config.args = Some(vec![
                format!(
                    "{}/tests/fixtures/mcp_stub_server.py",
                    env!("CARGO_MANIFEST_DIR")
                ),
                path.display().to_string(),
            ]);
            let context = McpClientContext::new();
            let manager = McpClientManager::prepare(
                &[config],
                None,
                None,
                Arc::clone(&context),
                crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
            )
            .await
            .expect("prepare");
            context.set_manager(Arc::downgrade(&manager));
            let registry = crate::tools::ToolRegistry::new();
            crate::tools::mcp_adapter::attach_session_registry(&manager, registry.clone()).await;
            let entry = manager
                .servers
                .get("stub")
                .cloned()
                .expect("the configured entry");
            let stub = Self {
                state: path,
                log,
                context,
                manager,
                entry,
                registry,
                _temp: temp,
            };
            stub.write_state(state);
            stub
        }

        /// Connect through the connector's door, as every host does.
        async fn connect(&self) {
            connector::connect_one(
                Arc::clone(&self.entry),
                Arc::clone(&self.manager),
                None,
                std::time::Duration::from_secs(20),
            )
            .await;
        }

        /// Give the stub new state. Written whole and renamed into place, so the stub's watcher
        /// never reads half a file.
        fn write_state(&self, mut state: serde_json::Value) {
            state["log"] = serde_json::Value::String(self.log.display().to_string());
            let staged = self.state.with_extension("tmp");
            std::fs::write(&staged, state.to_string()).expect("write the stub's state");
            std::fs::rename(&staged, &self.state).expect("move the stub's state into place");
        }

        /// The methods the stub has been sent so far, in order.
        fn logged_methods(&self) -> Vec<String> {
            let Ok(log) = std::fs::read_to_string(&self.log) else {
                return Vec::new();
            };
            log.lines()
                .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .filter_map(|request| request["method"].as_str().map(str::to_string))
                .collect()
        }

        fn sent(&self, method: &str) -> usize {
            self.logged_methods()
                .iter()
                .filter(|sent| *sent == method)
                .count()
        }

        async fn version(&self) -> String {
            self.entry
                .protocol_version()
                .await
                .map(|version| version.to_string())
                .unwrap_or_default()
        }

        async fn tool(&self, name: &str) -> Arc<McpTool> {
            self.manager
                .tool_by_name(&format!("mcp__stub__{name}"))
                .await
                .expect("the stub's tool is registered")
        }
    }

    /// Wait for `condition` to hold, up to `wait`.
    #[cfg(unix)]
    async fn eventually(
        wait: std::time::Duration,
        mut condition: impl AsyncFnMut() -> bool,
    ) -> bool {
        let deadline = tokio::time::Instant::now() + wait;
        while tokio::time::Instant::now() < deadline {
            if condition().await {
                return true;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        condition().await
    }

    #[cfg(unix)]
    fn call_context(frontend: Arc<dyn crate::frontend::Frontend>) -> CallContext {
        CallContext {
            session_id: None,
            tool_call_id: Some("tu-1".to_string()),
            frontend,
            cancellation: CancellationToken::new(),
        }
    }

    #[cfg(unix)]
    fn text_of(result: &rmcp::model::CallToolResult) -> String {
        result
            .content
            .iter()
            .filter_map(|block| match block {
                rmcp::model::ContentBlock::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The probe settles the era with each server: one that answers `server/discover` is spoken
    /// to as 2026-07-28 and never sent an `initialize`, one that does not know the method gets
    /// the `initialize` of 2025-11-25 on the same process. Both carry their instructions.
    #[cfg(unix)]
    #[tokio::test]
    async fn the_probe_settles_the_era_with_each_server() {
        let modern = StubServer::prepare(serde_json::json!({
            "tools": ["search"], "instructions": "call search first"
        }))
        .await;
        modern.connect().await;
        assert_eq!(modern.version().await, "2026-07-28");
        assert_eq!(
            modern.entry.instructions().as_deref(),
            Some("call search first")
        );
        assert!(modern.registry.get("mcp__stub__search").is_some());
        let sent = modern.logged_methods();
        assert_eq!(sent.first().map(String::as_str), Some("server/discover"));
        assert!(
            !sent.iter().any(|method| method == "initialize"),
            "{sent:?}"
        );
        assert!(
            eventually(std::time::Duration::from_secs(5), async || modern
                .sent("subscriptions/listen")
                == 1)
            .await,
            "{:?}",
            modern.logged_methods()
        );

        let legacy = StubServer::prepare(serde_json::json!({
            "tools": ["search"], "instructions": "legacy", "legacy_only": true
        }))
        .await;
        legacy.connect().await;
        assert_eq!(legacy.version().await, "2025-11-25");
        assert_eq!(legacy.entry.instructions().as_deref(), Some("legacy"));
        assert!(legacy.registry.get("mcp__stub__search").is_some());
        assert_eq!(&legacy.logged_methods()[..2], [
            "server/discover",
            "initialize"
        ]);
        assert_eq!(legacy.sent("subscriptions/listen"), 0);
    }

    /// A child that exits on the probe takes the transport the fallback needed, so it is spawned
    /// once more for the legacy handshake alone: the probe on one process, `initialize` first on
    /// the next.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_server_that_dies_on_the_probe_is_spawned_again_as_legacy() {
        let stub = StubServer::prepare(serde_json::json!({
            "tools": ["search"], "exit_on_probe": true
        }))
        .await;
        stub.connect().await;
        assert_eq!(
            stub.version().await,
            "2025-11-25",
            "{:?}",
            stub.entry.state().await.label()
        );
        assert_eq!(&stub.logged_methods()[..2], [
            "server/discover",
            "initialize"
        ]);
        assert!(stub.registry.get("mcp__stub__search").is_some());
    }

    /// A server that swallows the probe and everything after it is taken for legacy once the
    /// window passes, and spoken to on a child of its own; the verdict is kept, so the next connect
    /// sends `initialize` first and pays no window.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_server_that_swallows_the_probe_is_legacy_after_the_window_and_stays_so() {
        let stub = StubServer::prepare(serde_json::json!({
            "tools": ["search"], "ignore_probe": true
        }))
        .await;
        // Half of this is the probe's window.
        stub.entry
            .request_timeout
            .set(std::time::Duration::from_secs(2))
            .expect("unset until now");
        let started = std::time::Instant::now();
        stub.connect().await;
        assert_eq!(
            stub.version().await,
            "2025-11-25",
            "{:?}",
            stub.entry.state().await.label()
        );
        assert!(
            started.elapsed() >= std::time::Duration::from_secs(1),
            "the probe is given its window before the server is taken for legacy"
        );
        assert_eq!(&stub.logged_methods()[..2], [
            "server/discover",
            "initialize"
        ]);

        let again = connector::connect_server(&stub.entry)
            .await
            .expect("the second connect");
        drop(again);
        assert_eq!(
            stub.sent("server/discover"),
            1,
            "{:?}",
            stub.logged_methods()
        );
        assert_eq!(stub.sent("initialize"), 2);
    }

    /// A connect still in flight when the manager shuts down ends with it, and the child it was
    /// spawning dies: the thread the connect runs on would otherwise be abandoned at exit with
    /// the child alive and nothing left to kill it.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_shutdown_ends_a_connect_in_flight_and_its_child() {
        let stub = StubServer::prepare(serde_json::json!({
            "tools": ["search"], "ignore_probe": true
        }))
        .await;
        // The stub's processes are the ones whose command line names this test's state file.
        let marker = stub.state.display().to_string();
        let children = move || -> usize {
            std::process::Command::new("pgrep")
                .args(["-f", &marker])
                .output()
                .map(|output| String::from_utf8_lossy(&output.stdout).lines().count())
                .unwrap_or(0)
        };

        let connecting = tokio::spawn({
            let entry = Arc::clone(&stub.entry);
            let manager = Arc::clone(&stub.manager);
            async move {
                connector::connect_one(entry, manager, None, std::time::Duration::from_secs(60))
                    .await;
            }
        });
        assert!(
            eventually(std::time::Duration::from_secs(5), async || children() >= 1).await,
            "the probe child was never spawned"
        );

        stub.manager.shutdown_within(SHUTDOWN_BUDGET).await;
        tokio::time::timeout(std::time::Duration::from_secs(5), connecting)
            .await
            .expect("the connect ends with the shutdown rather than running its course")
            .expect("the connect task");
        assert!(
            eventually(std::time::Duration::from_secs(3), async || children() == 0).await,
            "the child outlived the shutdown"
        );
        assert!(
            matches!(stub.entry.state().await, ServerState::Pending),
            "nothing is recorded against a server that was never given its chance"
        );
    }

    /// A reconnect still in flight when the manager shuts down ends with it too, child included:
    /// it runs on a blocking thread of its own, bounded by the request timeout, which a shutdown
    /// would otherwise wait out.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_shutdown_ends_a_reconnect_in_flight_and_its_child() {
        let stub = StubServer::prepare(serde_json::json!({
            "tools": ["search"], "legacy_only": true, "exit_after": 1
        }))
        .await;
        let marker = stub.state.display().to_string();
        let children = move || -> usize {
            std::process::Command::new("pgrep")
                .args(["-f", &marker])
                .output()
                .map(|output| String::from_utf8_lossy(&output.stdout).lines().count())
                .unwrap_or(0)
        };
        stub.connect().await;
        assert!(
            eventually(std::time::Duration::from_secs(10), async || stub
                .entry
                .needs_reconnect()
                .await)
            .await,
            "the stub was supposed to exit after its first listing"
        );
        // The process the reconnect spawns answers nothing, so the reconnect sits in its bound.
        stub.write_state(serde_json::json!({"tools": ["search"], "mute": true}));
        let reconnecting = tokio::spawn({
            let entry = Arc::clone(&stub.entry);
            async move { entry.reconnect(&CancellationToken::new()).await }
        });
        assert!(
            eventually(std::time::Duration::from_secs(5), async || children() >= 1).await,
            "the reconnect never spawned its child"
        );

        stub.manager.shutdown_within(SHUTDOWN_BUDGET).await;
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), reconnecting)
            .await
            .expect("the reconnect ends with the shutdown rather than waiting out its bound")
            .expect("the reconnect task");
        assert!(
            outcome.is_err(),
            "a reconnect the shutdown ended did not succeed"
        );
        assert!(
            eventually(std::time::Duration::from_secs(3), async || children() == 0).await,
            "the child outlived the shutdown"
        );
    }

    /// A frontend that never answers, as a user who has stepped away.
    #[cfg(unix)]
    struct AbsentFrontend;

    #[cfg(unix)]
    #[async_trait::async_trait]
    impl crate::frontend::Frontend for AbsentFrontend {
        async fn emit(&self, _event: crate::frontend::FrontendEvent) {}

        async fn request_permission(
            &self,
            _request: crate::frontend::PermissionRequest,
        ) -> crate::frontend::PermissionOutcome {
            crate::frontend::PermissionOutcome::Allow
        }

        async fn handle_elicitation(
            &self,
            _prompt: crate::frontend::ElicitationPrompt,
        ) -> crate::frontend::ElicitationResponse {
            std::future::pending().await
        }
    }

    /// An interrupt while a modern server's question is on screen ends the call at once, as it
    /// does with the request in flight; the round is not the user's wait to sit out.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_interrupt_during_a_form_ends_the_call_at_once() {
        let stub = StubServer::prepare(serde_json::json!({
            "tools": ["search"],
            "elicit": {"message": "Name?", "schema": {"type": "object", "properties": {}}}
        }))
        .await;
        stub.connect().await;
        let tool = stub.tool("search").await;
        let cancellation = CancellationToken::new();
        let calling = tokio::spawn({
            let tool = Arc::clone(&tool);
            let cancellation = cancellation.clone();
            async move {
                let context = CallContext {
                    session_id: None,
                    tool_call_id: Some("tu-1".to_string()),
                    frontend: Arc::new(AbsentFrontend),
                    cancellation,
                };
                tool.call(None, &context).await
            }
        });
        assert!(
            eventually(std::time::Duration::from_secs(5), async || stub
                .sent("tools/call")
                == 1)
            .await,
            "the first round was never sent"
        );
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        cancellation.cancel();
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(2), calling)
            .await
            .expect("the call ends on the interrupt, not on the form's timeout")
            .expect("the call task");
        assert!(
            matches!(outcome, Err(MekaError::Interrupted)),
            "an interrupted call is interrupted: {outcome:?}"
        );
    }

    /// A resource read and a prompt render take the rounds a modern server asks for too, with no
    /// one to ask: the elicitation is declined and the server answers the retry.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_resource_read_and_a_prompt_render_take_the_rounds_a_modern_server_asks_for() {
        let stub = StubServer::prepare(serde_json::json!({
            "tools": [], "resources": ["file:///a"], "prompts": ["p"],
            "elicit_reads": true, "elicit_prompts": true,
            "elicit": {"message": "Name?", "schema": {"type": "object", "properties": {}}}
        }))
        .await;
        stub.connect().await;
        let cancellation = CancellationToken::new();

        let read = read_resource(&stub.entry, "file:///a".to_string(), &cancellation)
            .await
            .expect("the read completes on the second round");
        let text = match &read.contents[0] {
            rmcp::model::ResourceContents::TextResourceContents { text, .. } => text.clone(),
            other => panic!("text was expected: {other:?}"),
        };
        assert!(
            text.contains(r#""action": "decline""#) && text.contains("after round-1"),
            "{text}"
        );
        assert_eq!(stub.sent("resources/read"), 2);

        let rendered = get_prompt(&stub.entry, "p".to_string(), None, &cancellation)
            .await
            .expect("the render completes on the second round");
        let text = match &rendered.messages[0].content {
            rmcp::model::ContentBlock::Text(text) => text.text.clone(),
            other => panic!("text was expected: {other:?}"),
        };
        assert!(
            text.contains(r#""action": "decline""#) && text.contains("after round-1"),
            "{text}"
        );
        assert_eq!(stub.sent("prompts/get"), 2);
    }

    /// A modern server pushes nothing on its own: its tool changes arrive on the stream meka
    /// opened at connect, and reach the registry the way a legacy server's notification does.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_modern_server_s_tool_changes_arrive_on_its_stream() {
        let stub = StubServer::prepare(serde_json::json!({"tools": ["search"]})).await;
        stub.connect().await;
        assert!(stub.registry.get("mcp__stub__search").is_some());
        assert!(
            eventually(std::time::Duration::from_secs(5), async || stub
                .sent("subscriptions/listen")
                == 1)
            .await,
            "{:?}",
            stub.logged_methods()
        );

        stub.write_state(serde_json::json!({
            "tools": ["search", "create_page"],
            "notify": {"seq": 1, "tools_list_changed": true}
        }));
        assert!(
            eventually(std::time::Duration::from_secs(10), async || stub
                .registry
                .get("mcp__stub__create_page")
                .is_some())
            .await,
            "the tool the server added never reached the registry: {:?}",
            stub.logged_methods()
        );
    }

    /// On a modern server a subscription is the stream's filter: accepted when the acknowledgment
    /// names the URI, refused when it leaves it out. The set keeps only what was accepted, so the
    /// updates arrive for the one and the other is not asked for again; every change re-issues
    /// the stream.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_modern_subscription_is_the_stream_s_filter() {
        let stub = StubServer::prepare(serde_json::json!({
            "tools": [], "resources": ["file:///a"]
        }))
        .await;
        stub.connect().await;
        let cancellation = CancellationToken::new();

        subscriptions::subscribe(&stub.entry, "file:///a".to_string(), &cancellation)
            .await
            .expect("a URI the server offers is accepted");
        let refused =
            subscriptions::subscribe(&stub.entry, "file:///nope".to_string(), &cancellation)
                .await
                .expect_err("a URI the server does not offer is refused");
        assert!(
            refused
                .to_string()
                .contains("does not accept a subscription to 'file:///nope'"),
            "{refused}"
        );
        assert_eq!(stub.entry.subscriptions.uris(), vec!["file:///a"]);

        stub.write_state(serde_json::json!({
            "tools": [], "resources": ["file:///a"],
            "notify": {"seq": 1, "resources_updated": ["file:///a"]}
        }));
        assert!(
            eventually(std::time::Duration::from_secs(10), async || stub
                .context
                .resource_updates
                .snapshot()
                .iter()
                .any(|(server, uri, _)| server == "stub" && uri == "file:///a"))
            .await,
            "the update never reached the ledger: {:?}",
            stub.logged_methods()
        );

        subscriptions::unsubscribe(&stub.entry, "file:///a".to_string(), &cancellation)
            .await
            .expect("unsubscribed");
        assert!(stub.entry.subscriptions.uris().is_empty());
        assert!(
            eventually(std::time::Duration::from_secs(5), async || stub
                .sent("subscriptions/listen")
                == 4)
            .await,
            "one stream at connect and one per change: {:?}",
            stub.logged_methods()
        );
        assert_eq!(stub.sent("resources/subscribe"), 0);
    }

    /// A legacy subscription is a request the server forgets with the connection, so the set is
    /// asked for again on the next one and the updates keep coming.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_legacy_subscription_is_made_again_on_the_next_connection() {
        let state = serde_json::json!({
            "tools": ["search"], "resources": ["file:///a"], "legacy_only": true, "exit_after": 2
        });
        let stub = StubServer::prepare(state.clone()).await;
        stub.connect().await;
        assert_eq!(stub.version().await, "2025-11-25");
        subscriptions::subscribe(
            &stub.entry,
            "file:///a".to_string(),
            &CancellationToken::new(),
        )
        .await
        .expect("subscribed");
        assert_eq!(stub.sent("resources/subscribe"), 1);

        // A second listing is the stub's cue to exit, which closes the transport under meka.
        let _closing = stub
            .entry
            .list_tools_bounded(std::time::Duration::from_secs(5))
            .await;
        assert!(
            eventually(std::time::Duration::from_secs(10), async || stub
                .entry
                .needs_reconnect()
                .await)
            .await,
            "the stub was supposed to exit and close the transport"
        );
        stub.entry
            .reconnect(&CancellationToken::new())
            .await
            .expect("reconnect");

        assert!(
            eventually(std::time::Duration::from_secs(5), async || stub
                .sent("resources/subscribe")
                == 2)
            .await,
            "the new connection was not asked for the subscription: {:?}",
            stub.logged_methods()
        );
        assert_eq!(stub.entry.subscriptions.uris(), vec!["file:///a"]);

        let mut notifying = state;
        notifying["notify"] = serde_json::json!({"seq": 1, "resources_updated": ["file:///a"]});
        stub.write_state(notifying);
        assert!(
            eventually(std::time::Duration::from_secs(10), async || stub
                .context
                .resource_updates
                .snapshot()
                .iter()
                .any(|(server, uri, _)| server == "stub" && uri == "file:///a"))
            .await,
            "the update never reached the ledger on the new connection"
        );

        subscriptions::unsubscribe(
            &stub.entry,
            "file:///a".to_string(),
            &CancellationToken::new(),
        )
        .await
        .expect("unsubscribed");
        assert_eq!(
            stub.sent("resources/unsubscribe"),
            1,
            "a legacy server is told the subscription is over: {:?}",
            stub.logged_methods()
        );
        assert!(stub.entry.subscriptions.uris().is_empty());
    }

    /// A modern server asks for input in its answer rather than with a request of its own, and
    /// the call's own frontend answers: an accepting one puts the content in the retry, a
    /// declining one its decline, and the server's state goes back as it came.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_input_required_answer_is_fulfilled_through_the_call_s_frontend() {
        let stub = StubServer::prepare(serde_json::json!({
            "tools": ["search"],
            "elicit": {"message": "Name?", "schema": {
                "type": "object", "properties": {"name": {"type": "string"}}
            }}
        }))
        .await;
        stub.connect().await;
        let tool = stub.tool("search").await;

        let accepting = Arc::new(AcceptingFrontend(serde_json::json!({"name": "octocat"})));
        let result = tool
            .call(None, &call_context(accepting))
            .await
            .expect("the call completes on the second round");
        let text = text_of(&result);
        assert!(
            text.contains(r#""name": "octocat""#) && text.contains("after round-1"),
            "{text}"
        );
        assert_eq!(stub.sent("tools/call"), 2);

        let declining = Arc::new(crate::frontend::testing::RecordingFrontend::new());
        let text = text_of(
            &tool
                .call(None, &call_context(declining))
                .await
                .expect("a decline is an answer too"),
        );
        assert!(text.contains(r#""action": "decline""#), "{text}");
    }

    /// A server that asks on every round is cut off at the cap rather than asked forever.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_server_that_never_stops_asking_is_cut_off_at_the_round_cap() {
        let stub = StubServer::prepare(serde_json::json!({
            "tools": ["search"],
            "elicit": {"message": "Again?", "repeat": true, "schema": {
                "type": "object", "properties": {}
            }}
        }))
        .await;
        stub.connect().await;
        let tool = stub.tool("search").await;
        let accepting = Arc::new(AcceptingFrontend(serde_json::json!({})));
        let error = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            tool.call(None, &call_context(accepting)),
        )
        .await
        .expect("the cap ends the call long before the tool's timeout would")
        .expect_err("the cap ends the call");
        assert!(error.to_string().contains("round"), "{error}");
        assert_eq!(
            stub.sent("tools/call"),
            rmcp::model::DEFAULT_MRTR_MAX_ROUNDS,
            "{:?}",
            stub.logged_methods()
        );
    }

    /// Progress is found by the token rmcp mints, the one the server echoes: a token chosen here
    /// matches no notification, and the frontend never sees the line.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_progress_notification_reaches_the_call_s_frontend() {
        let stub = StubServer::prepare(serde_json::json!({
            "tools": ["search"], "progress": true
        }))
        .await;
        stub.connect().await;
        let tool = stub.tool("search").await;
        let recorder = Arc::new(crate::frontend::testing::RecordingFrontend::new());
        tool.call(
            None,
            &call_context(Arc::clone(&recorder) as Arc<dyn crate::frontend::Frontend>),
        )
        .await
        .expect("the call completes");
        assert!(
            eventually(std::time::Duration::from_secs(5), async || recorder
                .events()
                .iter()
                .any(|event| matches!(
                    event,
                    crate::frontend::FrontendEvent::McpProgress(update)
                        if update.message.as_deref() == Some("halfway")
                            && update.tool_use_id.as_deref() == Some("tu-1")
                )))
            .await,
            "no progress reached the frontend: {:?}",
            recorder.events()
        );
    }

    /// A shutdown ends a reconnect's attempts at once, whether it has begun before the one attempt
    /// a stdio server gets or lands in the backoff between two of an HTTP server's: no backoff is
    /// slept out, nothing is recorded over the state the shutdown leaves, and no background retry
    /// is spawned for an entry about to be closed.
    #[tokio::test]
    async fn a_shutdown_ends_a_reconnect_s_attempts_at_once() {
        async fn ended_at_once(entry: &Arc<ServerEntry>) {
            let started = std::time::Instant::now();
            let outcome = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                entry.reconnect_attempts(&tokio_util::sync::CancellationToken::new()),
            )
            .await
            .expect("the stop ends the attempts, not their backoff");
            assert!(
                matches!(
                    outcome,
                    Err(MekaError::McpConnection { ref message, .. }) if message == "is shutting down"
                ),
                "{outcome:?}"
            );
            assert!(
                started.elapsed() < std::time::Duration::from_secs(1),
                "{:?}",
                started.elapsed()
            );
            assert!(
                matches!(entry.state().await, ServerState::Pending),
                "nothing is recorded against an entry the shutdown is closing"
            );
        }

        let local = pending_entry("local", McpTransport::Stdio);
        local.connects.stopping().cancel();
        ended_at_once(&local).await;

        // The first attempt is refused at once by a port nothing listens on, and the stop lands
        // in the second's backoff: late enough that five refusals in a row, were there no
        // backoff, would have ended the reconnect with the refusal instead.
        let mut config = McpServerConfig::for_test("remote");
        config.url = Some("http://127.0.0.1:9/mcp".to_string());
        let remote = pending_entry_for(config);
        let stop = tokio::spawn({
            let connects = Arc::clone(&remote.connects);
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                connects.stopping().cancel();
            }
        });
        ended_at_once(&remote).await;
        stop.await.expect("the stop landed");
    }

    /// A connect that lands once the shutdown has begun is refused at the one door onto
    /// `Connected`, and the child it brought up dies with its service: the shutdown's own loop
    /// may already have passed the entry, and a server brought up behind it would outlive meka.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_connect_that_lands_after_the_shutdown_began_is_refused() {
        let stub = StubServer::prepare(serde_json::json!({"tools": ["search"]})).await;
        let marker = stub.state.display().to_string();
        let children = move || -> usize {
            std::process::Command::new("pgrep")
                .args(["-f", &marker])
                .output()
                .map(|output| String::from_utf8_lossy(&output.stdout).lines().count())
                .unwrap_or(0)
        };
        let service = connector::connect_server(&stub.entry)
            .await
            .expect("the connect itself succeeds");
        assert!(children() >= 1, "the connected child is alive");

        stub.entry.connects.stopping().cancel();
        let refused = stub.entry.adopt_connection(service).await;
        assert!(
            matches!(
                refused,
                Err(MekaError::McpConnection { ref message, .. }) if message == "is shutting down"
            ),
            "{refused:?}"
        );
        assert!(
            matches!(stub.entry.state().await, ServerState::Pending),
            "the entry stays as the shutdown found it"
        );
        assert!(
            eventually(std::time::Duration::from_secs(3), async || children() == 0).await,
            "the child outlived the refused connect"
        );
    }

    /// A connect bound that cuts the legacy handshake leaves no verdict: the next connect probes
    /// again, so a server that was only slow once, or that speaks only 2026-07-28, is not sent
    /// `initialize` first for the rest of the process.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_bound_that_cuts_the_legacy_handshake_leaves_no_verdict() {
        let stub =
            StubServer::prepare(serde_json::json!({"tools": ["search"], "mute": true})).await;
        // Half of this is the probe's window, which the second connect below has to answer the
        // probe within; the bound cuts the handshake that follows the first probe.
        stub.entry
            .request_timeout
            .set(std::time::Duration::from_secs(6))
            .expect("unset until now");
        connector::connect_one(
            Arc::clone(&stub.entry),
            Arc::clone(&stub.manager),
            None,
            std::time::Duration::from_secs(7),
        )
        .await;
        assert!(
            matches!(stub.entry.state().await, ServerState::Failed { .. }),
            "{:?}",
            stub.entry.state().await.label()
        );
        assert!(
            !stub
                .entry
                .legacy_era
                .load(std::sync::atomic::Ordering::Relaxed),
            "a handshake the bound cut is no verdict"
        );

        stub.write_state(serde_json::json!({"tools": ["search"]}));
        let service = connector::connect_server(&stub.entry)
            .await
            .expect("the next connect");
        assert_eq!(
            service
                .peer()
                .peer_info()
                .map(|info| info.protocol_version.to_string())
                .as_deref(),
            Some("2026-07-28"),
            "the next connect probed again and found a modern server"
        );
        assert_eq!(
            stub.sent("server/discover"),
            2,
            "{:?}",
            stub.logged_methods()
        );
        drop(service);
    }

    /// A modern server may end the stream it was asked for, as one about to restart does, and a
    /// proxy may cut the long request it rides on; the connection stands either way, and the
    /// server sends nothing again until asked, so meka asks again and the changes keep arriving.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_stream_the_server_ends_is_opened_again_while_the_connection_lives() {
        let stub = StubServer::prepare(serde_json::json!({"tools": ["search"]})).await;
        stub.connect().await;
        assert!(
            eventually(std::time::Duration::from_secs(5), async || stub
                .sent("subscriptions/listen")
                == 1)
            .await,
            "{:?}",
            stub.logged_methods()
        );

        stub.write_state(serde_json::json!({
            "tools": ["search"],
            "notify": {"seq": 1, "end_streams": true}
        }));
        assert!(
            eventually(std::time::Duration::from_secs(10), async || stub
                .sent("subscriptions/listen")
                == 2)
            .await,
            "no stream was opened in place of the one the server ended: {:?}",
            stub.logged_methods()
        );

        stub.write_state(serde_json::json!({
            "tools": ["search", "create_page"],
            "notify": {"seq": 2, "tools_list_changed": true}
        }));
        assert!(
            eventually(std::time::Duration::from_secs(10), async || stub
                .registry
                .get("mcp__stub__create_page")
                .is_some())
            .await,
            "the change after the server's end never arrived: {:?}",
            stub.logged_methods()
        );
        assert!(
            matches!(stub.entry.state().await, ServerState::Connected { .. }),
            "the connection stood throughout"
        );
    }

    /// A stop waits for the connects in flight to unwind, and no longer than its grace: one that
    /// ends while it waits is waited for, one that does not is given up on and said so.
    #[tokio::test]
    async fn a_stop_waits_for_the_connects_in_flight_and_no_longer_than_its_grace() {
        let connects = Connects::new();
        let in_flight = connects.connecting();
        let ending = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            drop(in_flight);
        });
        let started = std::time::Instant::now();
        assert!(
            connects.stop(std::time::Duration::from_secs(5)).await,
            "the one connect in flight ended within the grace"
        );
        assert!(
            started.elapsed() >= std::time::Duration::from_millis(150),
            "the stop returned before the connect had ended: {:?}",
            started.elapsed()
        );
        ending.await.expect("the connect's task");

        let stuck = Connects::new();
        let _never_ends = stuck.connecting();
        let started = std::time::Instant::now();
        assert!(
            !stuck.stop(std::time::Duration::from_millis(100)).await,
            "a connect that does not end is given up on"
        );
        assert!(started.elapsed() >= std::time::Duration::from_millis(100));
    }

    /// The subscribe and unsubscribe tools reach the server the model named through its entry:
    /// each changes the set, sends what the era calls for, and answers in its own words.
    #[cfg(unix)]
    #[tokio::test]
    async fn the_subscription_tools_reach_the_server_the_model_named() {
        use crate::{
            conversation::ToolResultContent,
            tools::{Tool, ToolContext},
        };

        fn text_of(output: &crate::tools::ToolOutput) -> String {
            output
                .content
                .iter()
                .filter_map(|block| match block {
                    ToolResultContent::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        }

        let stub = StubServer::prepare(serde_json::json!({
            "tools": [], "resources": ["file:///a"]
        }))
        .await;
        stub.connect().await;
        let scope = Arc::new(crate::tools::ToolScope::new(
            Vec::<String>::new(),
            Vec::<String>::new(),
        ));
        let arguments = serde_json::json!({"server": "stub", "uri": "file:///a"});

        let subscribe = crate::tools::mcp_resources::SubscribeMcpResourceTool {
            manager: Arc::clone(&stub.manager),
            tool_scope: Arc::clone(&scope),
        };
        let output = subscribe
            .execute(
                arguments.clone(),
                ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("subscribed");
        assert!(
            text_of(&output).contains("subscribed to 'file:///a' on server 'stub'"),
            "{}",
            text_of(&output)
        );
        assert_eq!(stub.entry.subscriptions.uris(), vec!["file:///a"]);

        let unsubscribe = crate::tools::mcp_resources::UnsubscribeMcpResourceTool {
            manager: Arc::clone(&stub.manager),
            tool_scope: scope,
        };
        let output = unsubscribe
            .execute(arguments, ToolContext::detached(CancellationToken::new()))
            .await
            .expect("unsubscribed");
        assert!(
            text_of(&output).contains("unsubscribed from 'file:///a' on server 'stub'"),
            "{}",
            text_of(&output)
        );
        assert!(stub.entry.subscriptions.uris().is_empty());
        assert!(
            eventually(std::time::Duration::from_secs(5), async || stub
                .sent("subscriptions/listen")
                == 3)
            .await,
            "one stream at connect and one per change: {:?}",
            stub.logged_methods()
        );
    }
}
