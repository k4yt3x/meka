//! Model Context Protocol (MCP) client integration. Manages the lifecycle of
//! configured MCP servers (stdio child processes or streamable HTTP), exposes
//! their tools through the regular [`crate::tools`] registry, and handles
//! OAuth/JWT authentication for HTTP transports.

pub mod auth;
pub mod cli;
pub mod connector;
pub mod elicitation;
pub mod expand;
pub mod handler;
pub mod progress;
pub mod resource_updates;
pub mod sanitize;
pub mod transport;

pub use handler::McpToolAdapter;

use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::{Arc, Weak};

use rmcp::model::{
    GetPromptRequestParams, GetPromptResult, Prompt, ReadResourceRequestParams, ReadResourceResult,
    Resource,
};
use rmcp::service::ServiceError;
use rmcp::{Peer, RoleClient};
use tokio::sync::{Mutex, RwLock};

use crate::config::{McpServerConfig, McpTransport};
use crate::error::{AgshError, Result};
use crate::permission::Permission;
use crate::provider::Provider;
use crate::session::TokenStore;
use crate::tools::ToolRegistry;

/// Cap MCP-provided text (tool descriptions, resource/prompt descriptions) to
/// this many characters so a chatty server can't blow up the system prompt.
/// Mirrors Claude Code's `MAX_MCP_DESCRIPTION_LENGTH`.
pub const MAX_MCP_DESCRIPTION_LENGTH: usize = 2048;

/// Cap on base64 payload size for an MCP image tool-result block. A server
/// returning a giant image would otherwise be cloned verbatim, forwarded to
/// the provider, billed against the user's API quota, and risk OOM. Mirrors
/// the 10 MiB body cap on `fetch_url`.
pub const MAX_MCP_IMAGE_BYTES: usize = 10 * 1024 * 1024;

/// Wall-clock timeout on `provider.complete` calls invoked from a server's
/// `sampling/createMessage` request. Without it, a hung provider keeps the
/// MCP request open forever; with it, the server gets a timely error and
/// the sampling slot is freed.
pub const MCP_SAMPLING_PROVIDER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Allow-list of image MIME types passed straight through to the provider.
/// Anything else (notably `image/svg+xml`, which can embed script/link
/// elements) is converted to a text placeholder.
pub const ALLOWED_IMAGE_MIME_TYPES: &[&str] =
    &["image/png", "image/jpeg", "image/gif", "image/webp"];

/// Cache TTL for MCP "needs auth" probe verdicts. A value of 15 min matches
/// Claude Code's `MCP_AUTH_CACHE_TTL_MS` and keeps a restart after a failed
/// auth flow from re-probing servers in a tight loop.
pub const MCP_AUTH_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(15 * 60);

pub(crate) type McpRunningService =
    rmcp::service::RunningService<RoleClient, handler::AgshClientHandler>;

pub struct McpClientManager {
    servers: HashMap<String, Arc<ServerEntry>>,
    /// Global fallback permission from `[mcp].default_permission`.
    /// Consulted by `resolve_tool_permission` at tool-registration time
    /// when neither the server nor the user has configured a more
    /// specific permission and the server didn't advertise a
    /// `readOnlyHint`. `None` means "no user default" — resolution
    /// falls through to the hardcoded strict `Write`.
    mcp_default_permission: Option<Permission>,
    /// Flipped to `true` by the background connector once every enabled
    /// entry has reached a terminal state (Connected or Failed). The turn
    /// gate watches this via [`Self::await_settled`] / [`Self::all_ready`].
    settled: tokio::sync::watch::Sender<bool>,
    /// Entries waiting to be connected by [`Self::start_connector`].
    /// `None` once the connector has been started so a second call is
    /// a no-op; avoids re-spawning the connector if a test or the REPL
    /// re-enters the same manager.
    pending_entries: std::sync::Mutex<Option<Vec<Arc<ServerEntry>>>>,
}

/// Lifecycle state of a single MCP server. Transitions:
/// - Built as `Disabled` (config says so) or `Pending` (will be
///   connected by the background connector).
/// - `Pending` → `Connected` on successful `initialize` + `list_tools`.
/// - `Pending` → `Failed` on connect error or connect-timeout.
/// - `Connected` → `Connected` (with a new `service` Arc) on reconnect.
#[derive(Clone)]
pub enum ServerState {
    Disabled,
    Pending,
    Connected {
        service: Arc<McpRunningService>,
    },
    Failed {
        error: String,
        #[allow(dead_code)]
        at: std::time::Instant,
    },
}

impl ServerState {
    pub fn label(&self) -> &'static str {
        match self {
            ServerState::Disabled => "disabled",
            ServerState::Pending => "pending",
            ServerState::Connected { .. } => "connected",
            ServerState::Failed { .. } => "failed",
        }
    }
}

/// Holds the lifecycle state of a single MCP server plus reconnection
/// machinery. Wrapped in an [`Arc`] and shared between the manager, the
/// per-server tool adapters, and the resource/prompt builtin tools so
/// every caller sees the current service (or the current failure) via
/// [`Self::require_connected`].
pub struct ServerEntry {
    pub(crate) server_name: String,
    pub(crate) config: McpServerConfig,
    pub(crate) token_store: Option<TokenStore>,
    pub(crate) client_context: Arc<McpClientContext>,
    pub(crate) state: RwLock<ServerState>,
    pub(crate) reconnect_lock: Mutex<()>,
    /// Optional `InitializeResult.instructions` captured on the first
    /// `Connected` transition. Immutable for the lifetime of the
    /// connection per the MCP spec, so reconnects don't reset it.
    pub(crate) instructions: OnceLock<Option<String>>,
}

impl ServerEntry {
    /// Returns the server's `InitializeResult.instructions` (sanitised +
    /// truncated to [`MAX_MCP_DESCRIPTION_LENGTH`]) if the server advertised
    /// one during the handshake.
    pub fn instructions(&self) -> Option<&str> {
        self.instructions.get().and_then(|opt| opt.as_deref())
    }

    /// Snapshot of the current lifecycle state. `Connected` carries an
    /// `Arc<McpRunningService>` which is cheap to clone.
    pub async fn state(&self) -> ServerState {
        self.state.read().await.clone()
    }

    pub(crate) fn server_name(&self) -> &str {
        &self.server_name
    }

    pub(crate) fn token_store(&self) -> Option<&TokenStore> {
        self.token_store.as_ref()
    }
}

impl ServerEntry {
    /// Return the live peer if the server is currently `Connected`;
    /// otherwise return an error describing the current lifecycle
    /// state. Every tool dispatch / list-call path funnels through
    /// this so the "MCP X not ready" error surfaces at one site.
    pub(crate) async fn require_connected(&self) -> Result<Peer<RoleClient>> {
        match &*self.state.read().await {
            ServerState::Connected { service } => Ok(service.peer().clone()),
            ServerState::Pending => Err(AgshError::McpConnection {
                server_name: self.server_name.clone(),
                message: "server is still connecting; try again".to_string(),
            }),
            ServerState::Failed { error, .. } => Err(AgshError::McpConnection {
                server_name: self.server_name.clone(),
                message: format!("server failed to connect: {}", error),
            }),
            ServerState::Disabled => Err(AgshError::McpConnection {
                server_name: self.server_name.clone(),
                message: "server is disabled in config".to_string(),
            }),
        }
    }

    /// Transport-close check used by [`Self::reconnect`]. Returns false
    /// if the server isn't `Connected` (there's nothing to reconnect).
    async fn needs_reconnect(&self) -> bool {
        match &*self.state.read().await {
            ServerState::Connected { service } => service.peer().is_transport_closed(),
            _ => false,
        }
    }

    /// Attempt to reconnect this server with exponential backoff. Serialised
    /// via `reconnect_lock` so concurrent tool calls don't stampede. If
    /// another caller already reopened the transport, returns immediately.
    ///
    /// Schedule: 1s, 2s, 4s, 8s, 16s, capped at 30s, max 5 attempts. Only
    /// remote (HTTP) transports go through backoff — a dead stdio child has
    /// to be respawned and retry-after-sleep doesn't help.
    ///
    /// The connect future itself can be `!Send` for OAuth-authenticated
    /// servers (rmcp 1.5 holds a `form_urlencoded::Serializer` across an
    /// await in its auth module, whose `Option<&dyn Fn(&str) -> Cow<[u8]>>`
    /// closure slot is not `Sync`). To keep `Tool::execute`'s `Send` bound
    /// satisfied, we drive the reconnect on a `spawn_blocking` thread using
    /// the outer runtime's `Handle`.
    pub(crate) async fn reconnect(self: &Arc<Self>) -> Result<()> {
        let _guard = self.reconnect_lock.lock().await;

        if !self.needs_reconnect().await {
            return Ok(());
        }

        tracing::warn!(
            "MCP server '{}' transport closed, attempting reconnect",
            self.server_name
        );

        let max_attempts: u32 = match self.config.transport {
            McpTransport::Stdio => 1,
            McpTransport::Http => 5,
        };
        let mut last_error: Option<AgshError> = None;
        for attempt in 0..max_attempts {
            if attempt > 0 {
                // 1s, 2s, 4s, 8s, 16s, capped at 30s.
                let delay_secs = std::cmp::min(30u64, 1u64 << (attempt - 1));
                tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
            }
            let handle = tokio::runtime::Handle::current();
            let server_name = self.server_name.clone();
            let config = self.config.clone();
            let token_store = self.token_store.clone();
            let client_context = Arc::clone(&self.client_context);

            let result = tokio::task::spawn_blocking(move || {
                handle.block_on(connector::connect_server(
                    &server_name,
                    &config,
                    token_store.as_ref(),
                    &client_context,
                ))
            })
            .await;

            match result {
                Ok(Ok(new_service)) => {
                    *self.state.write().await = ServerState::Connected {
                        service: Arc::new(new_service),
                    };
                    tracing::info!(
                        "reconnected to MCP server '{}' on attempt {}",
                        self.server_name,
                        attempt + 1
                    );
                    return Ok(());
                }
                Ok(Err(error)) => {
                    tracing::warn!(
                        "MCP server '{}' reconnect attempt {} failed: {}",
                        self.server_name,
                        attempt + 1,
                        error
                    );
                    last_error = Some(error);
                }
                Err(join_error) => {
                    tracing::warn!(
                        "MCP server '{}' reconnect task join error on attempt {}: {}",
                        self.server_name,
                        attempt + 1,
                        join_error
                    );
                    last_error = Some(AgshError::McpConnection {
                        server_name: self.server_name.clone(),
                        message: format!("reconnect task join error: {}", join_error),
                    });
                }
            }
        }
        Err(last_error.unwrap_or_else(|| AgshError::McpConnection {
            server_name: self.server_name.clone(),
            message: format!("exhausted {} reconnect attempts", max_attempts),
        }))
    }
}

/// Runtime tuning for the background MCP connector. Pulled from
/// `ResolvedConfig` by the binary; the manager uses it directly.
pub struct McpRuntimeConfig {
    /// Per-server wrap around connect + `initialize` + `list_tools`.
    pub connect_timeout: std::time::Duration,
    /// Max concurrent stdio spawns. Defaults to 3 (env
    /// `AGSH_MCP_STDIO_CONCURRENCY`).
    pub stdio_concurrency: usize,
    /// Max concurrent HTTP connects. Defaults to 20 (env
    /// `AGSH_MCP_HTTP_CONCURRENCY`).
    pub http_concurrency: usize,
}

impl McpRuntimeConfig {
    pub fn from_config(config: &crate::config::ResolvedConfig) -> Self {
        Self {
            connect_timeout: config.mcp_connect_timeout,
            stdio_concurrency: resolve_concurrency_env("AGSH_MCP_STDIO_CONCURRENCY", 3),
            http_concurrency: resolve_concurrency_env("AGSH_MCP_HTTP_CONCURRENCY", 20),
        }
    }
}

/// Parse a positive-integer concurrency override from `env_var`. Falls
/// back to `default` when the variable is unset, unparseable, or zero.
/// Extracted from `McpRuntimeConfig::from_config` so tests can exercise
/// the env-var override path without constructing a full `ResolvedConfig`.
fn resolve_concurrency_env(env_var: &str, default: usize) -> usize {
    std::env::var(env_var)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default)
}

impl McpClientManager {
    /// Validate configs and build a manager with every non-empty entry
    /// in `Disabled` or `Pending` state. Does NOT spawn any network /
    /// process work — that happens in [`Self::start_connector`] once
    /// the caller has built the [`crate::tools::ToolRegistry`]. Callers
    /// typically:
    /// 1. `let manager = McpClientManager::prepare(...).await?;`
    /// 2. Register the manager on the `McpClientContext`.
    /// 3. Build the tool registry.
    /// 4. `manager.start_connector(registry, runtime);`
    ///
    /// The split exists so the connector can register MCP tools into
    /// the registry as each server comes online, without forcing the
    /// registry to exist before config validation.
    pub async fn prepare(
        configs: &[McpServerConfig],
        mcp_default_permission: Option<Permission>,
        token_store: Option<TokenStore>,
        client_context: Arc<McpClientContext>,
    ) -> Result<Arc<Self>> {
        let mut servers = HashMap::new();
        let mut pending: Vec<Arc<ServerEntry>> = Vec::new();

        for original_config in configs {
            // Apply env-var substitution (`${VAR}` / `${VAR:-default}`) once,
            // up-front, so the rest of the pipeline sees only resolved values.
            let mut config = original_config.clone();
            let missing = crate::mcp::expand::expand_server_config(&mut config);
            if !missing.is_empty() {
                tracing::warn!(
                    "MCP server '{}': unresolved env vars {:?} left literal in config",
                    config.name,
                    missing
                );
            }

            if config.name.is_empty() {
                return Err(AgshError::McpConnection {
                    server_name: "(empty)".to_string(),
                    message: "server name must not be empty".to_string(),
                });
            }

            // Reject anything that would collide with agsh-internal names or
            // our `mcp__<server>__<tool>` namespace separator.
            if crate::mcp::sanitize::is_reserved_server_name(&config.name) {
                return Err(AgshError::McpConnection {
                    server_name: config.name.clone(),
                    message: "server name is reserved (agsh, ide, or mcp_*)".to_string(),
                });
            }

            let normalised = crate::mcp::sanitize::normalize_server_name(&config.name);
            if normalised != config.name {
                return Err(AgshError::McpConnection {
                    server_name: config.name.clone(),
                    message: format!(
                        "server name contains characters not allowed in tool prefixes (would normalise to '{}')",
                        normalised
                    ),
                });
            }

            if config.name.contains("__") {
                return Err(AgshError::McpConnection {
                    server_name: config.name.clone(),
                    message: "server name must not contain '__' (reserved as namespace separator)"
                        .to_string(),
                });
            }

            if servers.contains_key(&config.name) {
                return Err(AgshError::McpConnection {
                    server_name: config.name.clone(),
                    message: "duplicate server name".to_string(),
                });
            }

            let is_disabled = config.disabled;
            if is_disabled {
                tracing::info!(
                    "MCP server '{}' is disabled in config, skipping",
                    config.name
                );
            }

            let entry = Arc::new(ServerEntry {
                server_name: config.name.clone(),
                config: config.clone(),
                token_store: token_store.clone(),
                client_context: Arc::clone(&client_context),
                state: RwLock::new(if is_disabled {
                    ServerState::Disabled
                } else {
                    ServerState::Pending
                }),
                reconnect_lock: Mutex::new(()),
                instructions: OnceLock::new(),
            });
            if !is_disabled {
                pending.push(Arc::clone(&entry));
            }
            servers.insert(config.name.clone(), entry);
        }

        // Initialise the watch with `true` when nothing will ever be
        // pending (all servers disabled, or no servers configured) so
        // callers of `all_ready` / `await_settled` short-circuit
        // immediately. `send` on a Sender with no receivers errors and
        // drops the value, so the initial-value path is the only safe
        // pre-subscription way to publish settled.
        let initial_settled = pending.is_empty();
        let (settled_tx, _) = tokio::sync::watch::channel(initial_settled);
        let manager = Arc::new(Self {
            servers,
            mcp_default_permission,
            settled: settled_tx,
            pending_entries: std::sync::Mutex::new(Some(pending)),
        });
        Ok(manager)
    }

    /// Spawn the background connector. Consumes the `Pending` entry list
    /// stashed by [`Self::prepare`] so subsequent calls are no-ops. Safe
    /// to call on managers with no pending entries.
    pub fn start_connector(
        self: &Arc<Self>,
        tool_registry: crate::tools::ToolRegistry,
        runtime: McpRuntimeConfig,
    ) {
        let Some(pending) = self
            .pending_entries
            .lock()
            .expect("pending_entries lock poisoned")
            .take()
        else {
            return;
        };
        let settled = self.settled.clone();
        let mcp_default_permission = self.mcp_default_permission;
        tokio::spawn(async move {
            connector::run_connector(
                pending,
                tool_registry,
                mcp_default_permission,
                runtime,
                settled,
            )
            .await;
        });
    }

    /// True when every enabled server has reached a terminal state
    /// (`Connected` or `Failed`). Returns `true` if there are no enabled
    /// servers configured. Non-blocking.
    pub fn all_ready(&self) -> bool {
        *self.settled.borrow()
    }

    /// Parks until the background connector finishes processing every
    /// enabled server. Returns immediately if already settled. Safe to
    /// call concurrently from multiple turn dispatches.
    pub async fn await_settled(&self) {
        let mut rx = self.settled.subscribe();
        if *rx.borrow() {
            return;
        }
        let _ = rx.wait_for(|done| *done).await;
    }

    /// Snapshot of enabled servers that are not currently `Connected`
    /// (still `Pending` or `Failed`). Used by the per-turn strict gate
    /// to compose the rejection message.
    pub async fn enabled_not_connected(&self) -> Vec<(String, ServerState)> {
        let mut out = Vec::new();
        for (name, entry) in &self.servers {
            let state = entry.state().await;
            match state {
                ServerState::Connected { .. } | ServerState::Disabled => {}
                other => out.push((name.clone(), other)),
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    pub fn server_entry(&self, server_name: &str) -> Option<Arc<ServerEntry>> {
        self.servers.get(server_name).cloned()
    }

    pub fn server_names(&self) -> Vec<String> {
        self.servers.keys().cloned().collect()
    }

    /// Returns `(server_name, instructions)` pairs for every connected server
    /// that advertised an `InitializeResult.instructions` string during the
    /// handshake. Already sanitised and truncated to
    /// [`MAX_MCP_DESCRIPTION_LENGTH`]. Used by the agent loop to splice MCP
    /// server instructions into the system prompt.
    pub fn server_instructions(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for (name, entry) in &self.servers {
            if let Some(text) = entry.instructions()
                && !text.trim().is_empty()
            {
                out.push((name.clone(), text.to_string()));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    pub async fn discover_tools_for_server(
        &self,
        server_name: &str,
    ) -> Result<Vec<McpToolAdapter>> {
        let Some(entry) = self.servers.get(server_name) else {
            return Ok(Vec::new());
        };

        let server_config = &entry.config;

        let peer = entry.require_connected().await?;
        let tools = peer
            .list_all_tools()
            .await
            .map_err(|error| AgshError::McpConnection {
                server_name: server_name.to_string(),
                message: format!("list_tools failed: {}", error),
            })?;

        // Collect advertised raw names up-front so we can flag stale
        // `allowed_tools` / `disabled_tools` / `tool_permissions` entries
        // that no longer match anything the server returns.
        let advertised: std::collections::HashSet<&str> =
            tools.iter().map(|t| t.name.as_ref()).collect();
        warn_on_stale_tool_config(server_name, server_config, &advertised);

        let mut adapters = Vec::new();
        for tool in tools {
            let raw_tool_name = tool.name.as_ref().to_string();

            if !tool_is_allowed(server_config, &raw_tool_name) {
                continue;
            }

            // Sanitise the tool's advertised name defensively — rare in the
            // wild, but a server returning `my.tool` or anything with
            // Unicode would cause the provider to reject the schema.
            let sanitised_tool_name = crate::mcp::sanitize::normalize_server_name(&raw_tool_name);
            let namespaced_name = format!("mcp__{}__{}", server_name, sanitised_tool_name);

            let raw_description = tool
                .description
                .as_ref()
                .map(|d| d.as_ref().to_string())
                .unwrap_or_default();
            let description = truncate(
                &crate::mcp::sanitize::sanitize_text(&raw_description),
                MAX_MCP_DESCRIPTION_LENGTH,
            );

            let parameters = match serde_json::to_value(&*tool.input_schema) {
                Ok(value) => value,
                Err(error) => {
                    tracing::warn!(
                        "MCP server '{}' tool '{}' has unserializable input schema ({}); \
                         skipping registration",
                        server_name,
                        raw_tool_name,
                        error
                    );
                    continue;
                }
            };

            // Per-tool permission via the layered resolution. Hints come
            // from `tool.annotations.readOnlyHint` as published by the
            // server; the function handles all the precedence rules.
            let permission = resolve_tool_permission(
                server_name,
                &raw_tool_name,
                tool.annotations.as_ref(),
                server_config,
                self.mcp_default_permission,
            )?;

            let annotations = tool
                .annotations
                .as_ref()
                .and_then(|ann| serde_json::to_value(ann).ok());
            let meta = tool
                .meta
                .as_ref()
                .and_then(|m| serde_json::to_value(m).ok());
            let title = tool
                .title
                .as_ref()
                .map(|t| crate::mcp::sanitize::sanitize_text(t));

            adapters.push(McpToolAdapter::new(
                namespaced_name,
                raw_tool_name,
                description,
                parameters,
                permission,
                Arc::clone(entry),
                annotations,
                meta,
                title,
            ));
        }

        Ok(adapters)
    }

    /// Connect to the named server and list EVERY advertised tool —
    /// including ones currently filtered out by `allowed_tools` /
    /// `disabled_tools` so users editing those lists can see what
    /// names are available. Permission is resolved through the normal
    /// 5-step chain with the winning step recorded on each entry.
    ///
    /// Differs from [`Self::discover_tools_for_server`] by (a) not
    /// filtering by allow/block lists, (b) not registering adapters, and
    /// (c) capturing the resolution source for display.
    pub async fn list_advertised_tools(&self, server_name: &str) -> Result<Vec<AdvertisedTool>> {
        let Some(entry) = self.servers.get(server_name) else {
            return Err(AgshError::McpConnection {
                server_name: server_name.to_string(),
                message: format!("no MCP server named '{}'", server_name),
            });
        };

        let server_config = &entry.config;
        let peer = entry.require_connected().await?;
        let tools = peer
            .list_all_tools()
            .await
            .map_err(|error| AgshError::McpConnection {
                server_name: server_name.to_string(),
                message: format!("list_tools failed: {}", error),
            })?;

        let mut out = Vec::with_capacity(tools.len());
        for tool in tools {
            let raw_name = tool.name.as_ref().to_string();
            let raw_description = tool
                .description
                .as_ref()
                .map(|d| d.as_ref().to_string())
                .unwrap_or_default();
            let description = truncate(
                &crate::mcp::sanitize::sanitize_text(&raw_description),
                MAX_MCP_DESCRIPTION_LENGTH,
            );
            let (resolved_permission, permission_source) = resolve_tool_permission_with_source(
                server_name,
                &raw_name,
                tool.annotations.as_ref(),
                server_config,
                self.mcp_default_permission,
            )?;
            let allowed = tool_is_allowed(server_config, &raw_name);
            out.push(AdvertisedTool {
                raw_name,
                description,
                resolved_permission,
                permission_source,
                allowed,
            });
        }

        out.sort_by(|a, b| a.raw_name.cmp(&b.raw_name));
        Ok(out)
    }

    /// Shutdown helper for callers that hold the manager through an
    /// `Arc`. Try-unwraps; if the Arc is unshared, drives the owned
    /// [`Self::shutdown`]. If something still holds a reference, drops
    /// the Arc and lets rmcp's drop guards clean up.
    pub async fn shutdown_arc(self: Arc<Self>) {
        match Arc::try_unwrap(self) {
            Ok(manager) => manager.shutdown().await,
            Err(_shared) => {
                tracing::debug!("mcp manager still referenced; relying on drop guards");
            }
        }
    }

    pub async fn shutdown(self) {
        /// Max time to wait for in-flight tool calls to complete before we
        /// drop the shared service Arc and let the drop-guard cancel it.
        const SHUTDOWN_GRACE: std::time::Duration = std::time::Duration::from_millis(2000);
        /// Max time to wait for `RunningService::close` to finish after the
        /// shared references are released.
        const CLOSE_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(2000);

        for (server_name, entry) in self.servers {
            let Ok(entry) = Arc::try_unwrap(entry) else {
                tracing::debug!(
                    "MCP server '{}' entry still referenced; relying on drop guard for cleanup",
                    server_name
                );
                continue;
            };

            // Only Connected entries have a service to close; Pending /
            // Failed / Disabled entries are tear-down no-ops.
            let service = match entry.state.into_inner() {
                ServerState::Connected { service } => service,
                _ => continue,
            };

            // In-flight tool calls hold their own Arc<RunningService> clone.
            // Wait up to `SHUTDOWN_GRACE` for those to complete so the normal
            // `RunningService::close` path can run instead of falling straight
            // to the drop-guard abort.
            let deadline = tokio::time::Instant::now() + SHUTDOWN_GRACE;
            while Arc::strong_count(&service) > 1 && tokio::time::Instant::now() < deadline {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }

            match Arc::try_unwrap(service) {
                Ok(mut owned_service) => {
                    match owned_service.close_with_timeout(CLOSE_TIMEOUT).await {
                        Ok(Some(_)) => {}
                        Ok(None) => {
                            tracing::warn!(
                                "MCP server '{}' shutdown timed out after {:?}",
                                server_name,
                                CLOSE_TIMEOUT
                            );
                        }
                        Err(error) => {
                            tracing::warn!(
                                "failed to shut down MCP server '{}': {}",
                                server_name,
                                error
                            );
                        }
                    }
                }
                Err(_arc) => {
                    tracing::debug!(
                        "MCP server '{}' still had in-flight calls after {:?} grace; \
                         relying on drop guard for cleanup",
                        server_name,
                        SHUTDOWN_GRACE
                    );
                }
            }
        }
    }
}

/// Decide whether a tool advertised by a server should be registered.
/// Applies `allowed_tools` (restrict-in, when set and non-empty) then
/// `disabled_tools` (always-remove). Both fields can coexist — the
/// allow-list acts as a restriction, and the block-list subtracts from
/// whatever remains. A tool passes iff it survives both checks.
pub(crate) fn tool_is_allowed(server_config: &McpServerConfig, tool_raw_name: &str) -> bool {
    if let Some(allow) = server_config.allowed_tools.as_deref()
        && !allow.is_empty()
        && !allow.iter().any(|t| t == tool_raw_name)
    {
        return false;
    }
    if let Some(deny) = server_config.disabled_tools.as_deref()
        && deny.iter().any(|t| t == tool_raw_name)
    {
        return false;
    }
    true
}

/// Whether the given raw tool name is in this server's
/// [`eager_load_tools`][McpServerConfig::eager_load_tools] list. Mirrors
/// [`tool_is_allowed`]'s shape. When true, the registration sites skip
/// `mark_deferred` so the tool ships in the cacheable tools-array prefix
/// from the first turn instead of after a `load_tool` round-trip.
pub(crate) fn tool_should_eager_load(server_config: &McpServerConfig, tool_raw_name: &str) -> bool {
    server_config
        .eager_load_tools
        .as_ref()
        .is_some_and(|list| list.iter().any(|n| n == tool_raw_name))
}

/// Emit a `warn!` once per entry in `allowed_tools` / `disabled_tools`
/// / `eager_load_tools` / `tool_permissions` that doesn't match anything
/// the server currently advertises. Users get a visible heads-up without
/// failing the connect — tool lists can change between server releases,
/// and forcing a hard error on every rename would be hostile. Also warns
/// on the disabled∩eager-load overlap, which is meaningless (disabled
/// tools aren't registered, so eager-loading them is a no-op).
pub(crate) fn warn_on_stale_tool_config(
    server_name: &str,
    server_config: &McpServerConfig,
    advertised: &std::collections::HashSet<&str>,
) {
    if let Some(allow) = server_config.allowed_tools.as_deref() {
        for name in allow {
            if !advertised.contains(name.as_str()) {
                tracing::warn!(
                    "MCP server '{}': allowed_tools entry '{}' doesn't match any advertised tool",
                    server_name,
                    name
                );
            }
        }
    }
    if let Some(deny) = server_config.disabled_tools.as_deref() {
        for name in deny {
            if !advertised.contains(name.as_str()) {
                tracing::warn!(
                    "MCP server '{}': disabled_tools entry '{}' doesn't match any advertised tool",
                    server_name,
                    name
                );
            }
        }
    }
    if let Some(eager) = server_config.eager_load_tools.as_deref() {
        let disabled = server_config.disabled_tools.as_deref().unwrap_or(&[]);
        for name in eager {
            if !advertised.contains(name.as_str()) {
                tracing::warn!(
                    "MCP server '{}': eager_load_tools entry '{}' doesn't match any advertised tool",
                    server_name,
                    name
                );
            }
            if disabled.iter().any(|d| d == name) {
                tracing::warn!(
                    "MCP server '{}': eager_load_tools entry '{}' is also in disabled_tools \
                     (the tool won't be registered at all, so eager-loading it is a no-op)",
                    server_name,
                    name
                );
            }
        }
    }
    if let Some(perms) = server_config.tool_permissions.as_ref() {
        for key in perms.keys() {
            if !advertised.contains(key.as_str()) {
                tracing::warn!(
                    "MCP server '{}': tool_permissions key '{}' doesn't match any advertised tool",
                    server_name,
                    key
                );
            }
        }
    }
}

/// Resolve the required permission for a single MCP tool. Applies the
/// layered policy documented in `docs/book/src/configuration/config-file.md`:
///
/// 1. `server.tool_permissions[tool]` — per-tool user override.
/// 2. `server.permission` — server-level user override.
/// 3. `tool.annotations.readOnlyHint` advertised by the server:
///    `true` → Read, `false` → Write.
/// 4. `mcp.default_permission` — global fallback when no hint exists.
/// 5. Hardcoded `Write` — ultimate strict fallback.
///
/// User config at steps 1/2 always beats the server's hints. Hints
/// beat the global fallback so a `readOnlyHint = false` destructive
/// tool isn't silently promoted to Read just because the user opted
/// into a lenient global default.
pub(crate) fn resolve_tool_permission(
    server_name: &str,
    tool_raw_name: &str,
    tool_annotations: Option<&rmcp::model::ToolAnnotations>,
    server_config: &McpServerConfig,
    mcp_default: Option<Permission>,
) -> Result<Permission> {
    resolve_tool_permission_with_source(
        server_name,
        tool_raw_name,
        tool_annotations,
        server_config,
        mcp_default,
    )
    .map(|(permission, _)| permission)
}

/// Identifies which step of the 5-step resolution chain produced a
/// tool's permission. Used by `agsh mcp tools <name>` so users can see
/// which knob is driving each tool's classification when editing
/// allow/block lists or per-tool overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionSource {
    ToolOverride,
    ServerOverride,
    ReadOnlyHint,
    GlobalDefault,
    Fallback,
}

impl PermissionSource {
    /// Short human label matching the config keys users would edit.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ToolOverride => "tool_permission",
            Self::ServerOverride => "server_permission",
            Self::ReadOnlyHint => "readOnlyHint",
            Self::GlobalDefault => "default_permission",
            Self::Fallback => "fallback",
        }
    }
}

/// A tool advertised by an MCP server, paired with the resolved
/// permission and the source step of the resolution chain. Returned
/// by [`McpClientManager::list_advertised_tools`] and printed by
/// `agsh mcp tools <server>`. The raw `readOnlyHint` value isn't
/// carried here because [`PermissionSource::ReadOnlyHint`] already
/// signals when the hint drove the decision; downstream renderers
/// that want the raw value can re-query.
pub struct AdvertisedTool {
    /// Raw name as advertised by the server — use this value in
    /// `allowed_tools` / `disabled_tools` / `tool_permissions` config.
    pub raw_name: String,
    /// Sanitised + truncated description (same pipeline as registered
    /// tools).
    pub description: String,
    /// Output of the 5-step permission resolution.
    pub resolved_permission: Permission,
    /// Which step of the chain won.
    pub permission_source: PermissionSource,
    /// `false` if currently filtered out by `allowed_tools` /
    /// `disabled_tools` — i.e. the agent would never see this tool.
    pub allowed: bool,
}

/// Same resolution as [`resolve_tool_permission`] but also returns which
/// step of the chain fired, so `agsh mcp tools` can show the user
/// exactly why a given tool has its current permission.
fn resolve_tool_permission_with_source(
    server_name: &str,
    tool_raw_name: &str,
    tool_annotations: Option<&rmcp::model::ToolAnnotations>,
    server_config: &McpServerConfig,
    mcp_default: Option<Permission>,
) -> Result<(Permission, PermissionSource)> {
    // 1. Per-tool override.
    if let Some(map) = &server_config.tool_permissions
        && let Some(raw) = map.get(tool_raw_name)
    {
        let permission = raw
            .parse::<Permission>()
            .map_err(|_| AgshError::McpConnection {
                server_name: server_name.to_string(),
                message: format!(
                    "invalid tool_permissions['{}'] = '{}': expected \
                     'none', 'read', 'ask', or 'write'",
                    tool_raw_name, raw
                ),
            })?;
        return Ok((permission, PermissionSource::ToolOverride));
    }
    // 2. Server-level override.
    if let Some(raw) = server_config.permission.as_deref() {
        let permission = raw
            .parse::<Permission>()
            .map_err(|_| AgshError::McpConnection {
                server_name: server_name.to_string(),
                message: format!(
                    "invalid permission '{}': expected 'none', 'read', \
                     'ask', or 'write'",
                    raw
                ),
            })?;
        return Ok((permission, PermissionSource::ServerOverride));
    }
    // 3. Server-advertised readOnlyHint.
    if let Some(annotations) = tool_annotations
        && let Some(hint) = annotations.read_only_hint
    {
        let permission = if hint {
            Permission::Read
        } else {
            Permission::Write
        };
        return Ok((permission, PermissionSource::ReadOnlyHint));
    }
    // 4. Global [mcp].default_permission.
    if let Some(permission) = mcp_default {
        return Ok((permission, PermissionSource::GlobalDefault));
    }
    // 5. Hardcoded strict fallback.
    Ok((Permission::Write, PermissionSource::Fallback))
}

/// Shared context threaded into every [`handler::AgshClientHandler`] so
/// notification callbacks and server-to-client requests (sampling,
/// list_roots, elicitation) can reach the rest of the agent. All slots are
/// optional because the handler is constructed before the agent/provider
/// exist — they are filled in post-construction by `main.rs` using the
/// `set_*` helpers.
#[derive(Default)]
pub struct McpClientContext {
    /// LLM provider used to serve `sampling/createMessage` requests. Only
    /// consulted when a server has `sampling = true` in its config.
    provider: OnceLock<Arc<dyn Provider>>,
    /// Tool registry to hot-swap when a server emits `tools/list_changed`.
    registry: OnceLock<ToolRegistry>,
    /// Weak reference to the MCP manager so the notification callback can
    /// rediscover tools without creating an Arc cycle through the handler.
    manager: OnceLock<Weak<McpClientManager>>,
}

impl McpClientContext {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn set_provider(&self, provider: Arc<dyn Provider>) {
        if self.provider.set(provider).is_err() {
            tracing::warn!("MCP client context: provider already set");
        }
    }

    pub fn set_registry(&self, registry: ToolRegistry) {
        if self.registry.set(registry).is_err() {
            tracing::warn!("MCP client context: registry already set");
        }
    }

    pub fn set_manager(&self, manager: Weak<McpClientManager>) {
        if self.manager.set(manager).is_err() {
            tracing::warn!("MCP client context: manager already set");
        }
    }

    pub(crate) fn provider(&self) -> Option<Arc<dyn Provider>> {
        self.provider.get().cloned()
    }

    pub(crate) fn registry(&self) -> Option<ToolRegistry> {
        self.registry.get().cloned()
    }

    pub(crate) fn manager(&self) -> Option<Weak<McpClientManager>> {
        self.manager.get().cloned()
    }
}

/// Truncate a string to `max_chars` Unicode scalar values, appending an
/// ellipsis marker if truncation occurred. Operates on `char` boundaries so
/// the result is always valid UTF-8.
pub fn truncate(text: &str, max_chars: usize) -> String {
    let mut byte_end = text.len();
    for (count, (idx, _)) in text.char_indices().enumerate() {
        if count == max_chars {
            byte_end = idx;
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

/// List all resources advertised by a server. Returned verbatim from the
/// current peer; no caching is done here.
pub async fn list_resources(entry: &Arc<ServerEntry>) -> Result<Vec<Resource>> {
    let peer = entry.require_connected().await?;
    match peer.list_all_resources().await {
        Ok(resources) => Ok(resources),
        Err(ServiceError::TransportClosed) => {
            entry.reconnect().await?;
            let peer = entry.require_connected().await?;
            peer.list_all_resources()
                .await
                .map_err(|error| AgshError::McpConnection {
                    server_name: entry.server_name.clone(),
                    message: format!("list_resources failed: {}", error),
                })
        }
        Err(error) => Err(AgshError::McpConnection {
            server_name: entry.server_name.clone(),
            message: format!("list_resources failed: {}", error),
        }),
    }
}

pub async fn read_resource(entry: &Arc<ServerEntry>, uri: String) -> Result<ReadResourceResult> {
    let params = ReadResourceRequestParams::new(uri.clone());
    let peer = entry.require_connected().await?;
    match peer.read_resource(params.clone()).await {
        Ok(result) => Ok(result),
        Err(ServiceError::TransportClosed) => {
            entry.reconnect().await?;
            let peer = entry.require_connected().await?;
            peer.read_resource(params)
                .await
                .map_err(|error| AgshError::McpConnection {
                    server_name: entry.server_name.clone(),
                    message: format!("read_resource({}) failed: {}", uri, error),
                })
        }
        Err(error) => Err(AgshError::McpConnection {
            server_name: entry.server_name.clone(),
            message: format!("read_resource({}) failed: {}", uri, error),
        }),
    }
}

pub async fn list_prompts(entry: &Arc<ServerEntry>) -> Result<Vec<Prompt>> {
    let peer = entry.require_connected().await?;
    match peer.list_all_prompts().await {
        Ok(prompts) => Ok(prompts),
        Err(ServiceError::TransportClosed) => {
            entry.reconnect().await?;
            let peer = entry.require_connected().await?;
            peer.list_all_prompts()
                .await
                .map_err(|error| AgshError::McpConnection {
                    server_name: entry.server_name.clone(),
                    message: format!("list_prompts failed: {}", error),
                })
        }
        Err(error) => Err(AgshError::McpConnection {
            server_name: entry.server_name.clone(),
            message: format!("list_prompts failed: {}", error),
        }),
    }
}

pub async fn subscribe_resource(entry: &Arc<ServerEntry>, uri: String) -> Result<()> {
    let peer = entry.require_connected().await?;
    let params = rmcp::model::SubscribeRequestParams::new(uri.clone());
    peer.subscribe(params)
        .await
        .map_err(|error| AgshError::McpConnection {
            server_name: entry.server_name.clone(),
            message: format!("subscribe({}) failed: {}", uri, error),
        })
}

pub async fn unsubscribe_resource(entry: &Arc<ServerEntry>, uri: String) -> Result<()> {
    let peer = entry.require_connected().await?;
    let params = rmcp::model::UnsubscribeRequestParams::new(uri.clone());
    peer.unsubscribe(params)
        .await
        .map_err(|error| AgshError::McpConnection {
            server_name: entry.server_name.clone(),
            message: format!("unsubscribe({}) failed: {}", uri, error),
        })
}

pub async fn get_prompt(
    entry: &Arc<ServerEntry>,
    name: String,
    arguments: Option<serde_json::Map<String, serde_json::Value>>,
) -> Result<GetPromptResult> {
    let mut params = GetPromptRequestParams::new(name.clone());
    params.arguments = arguments;

    let peer = entry.require_connected().await?;
    match peer.get_prompt(params.clone()).await {
        Ok(result) => Ok(result),
        Err(ServiceError::TransportClosed) => {
            entry.reconnect().await?;
            let peer = entry.require_connected().await?;
            peer.get_prompt(params)
                .await
                .map_err(|error| AgshError::McpConnection {
                    server_name: entry.server_name.clone(),
                    message: format!("get_prompt({}) failed: {}", name, error),
                })
        }
        Err(error) => Err(AgshError::McpConnection {
            server_name: entry.server_name.clone(),
            message: format!("get_prompt({}) failed: {}", name, error),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bare_server_config(name: &str) -> McpServerConfig {
        McpServerConfig {
            name: name.to_string(),
            transport: McpTransport::Http,
            command: None,
            args: None,
            env: None,
            url: Some("https://example".to_string()),
            auth_token: None,
            headers: None,
            headers_helper: None,
            auth: None,
            permission: None,
            allowed_tools: None,
            disabled_tools: None,
            eager_load_tools: None,
            tool_permissions: None,
            sampling: false,
            sampling_limit: None,
            disabled: false,
        }
    }

    fn annotations_with_read_only_hint(hint: Option<bool>) -> rmcp::model::ToolAnnotations {
        // `ToolAnnotations` is `#[non_exhaustive]`; use the builder.
        let mut ann = rmcp::model::ToolAnnotations::new();
        ann.read_only_hint = hint;
        ann
    }

    #[test]
    fn resolve_tool_permission_prefers_per_tool_override() {
        let mut server = bare_server_config("s");
        server.permission = Some("write".into());
        let mut per_tool = std::collections::HashMap::new();
        per_tool.insert("search".to_string(), "read".to_string());
        server.tool_permissions = Some(per_tool);

        // Per-tool override wins even when both the server default AND
        // the server's hint disagree.
        let annotations = annotations_with_read_only_hint(Some(false));
        let resolved = resolve_tool_permission(
            "s",
            "search",
            Some(&annotations),
            &server,
            Some(Permission::Write),
        )
        .expect("should resolve");
        assert_eq!(resolved, Permission::Read);
    }

    #[test]
    fn resolve_tool_permission_falls_through_to_server_level() {
        let mut server = bare_server_config("s");
        server.permission = Some("read".into());
        // Server level beats the hint.
        let annotations = annotations_with_read_only_hint(Some(false));
        let resolved = resolve_tool_permission(
            "s",
            "any",
            Some(&annotations),
            &server,
            Some(Permission::Write),
        )
        .expect("should resolve");
        assert_eq!(resolved, Permission::Read);
    }

    #[test]
    fn resolve_tool_permission_honours_read_only_hint() {
        let server = bare_server_config("s");
        // readOnlyHint = true → Read, even though the global default
        // would otherwise be Write.
        let annotations = annotations_with_read_only_hint(Some(true));
        let resolved = resolve_tool_permission(
            "s",
            "search",
            Some(&annotations),
            &server,
            Some(Permission::Write),
        )
        .expect("should resolve");
        assert_eq!(resolved, Permission::Read);

        // readOnlyHint = false → Write, even though the global default
        // is the lenient Read.
        let annotations = annotations_with_read_only_hint(Some(false));
        let resolved = resolve_tool_permission(
            "s",
            "write-page",
            Some(&annotations),
            &server,
            Some(Permission::Read),
        )
        .expect("should resolve");
        assert_eq!(resolved, Permission::Write);
    }

    #[test]
    fn resolve_tool_permission_falls_through_to_mcp_default() {
        let server = bare_server_config("s");
        // No user overrides, no hint → fall through to `[mcp].default`.
        let resolved = resolve_tool_permission("s", "any", None, &server, Some(Permission::Read))
            .expect("should resolve");
        assert_eq!(resolved, Permission::Read);
    }

    #[test]
    fn resolve_tool_permission_hardcoded_write_fallback() {
        let server = bare_server_config("s");
        // Nothing configured anywhere, no hint → hardcoded strict Write.
        let resolved =
            resolve_tool_permission("s", "any", None, &server, None).expect("should resolve");
        assert_eq!(resolved, Permission::Write);
    }

    #[test]
    fn resolve_tool_permission_rejects_invalid_tool_override() {
        let mut server = bare_server_config("s");
        let mut per_tool = std::collections::HashMap::new();
        per_tool.insert("search".to_string(), "typo".to_string());
        server.tool_permissions = Some(per_tool);
        let err = resolve_tool_permission("s", "search", None, &server, None)
            .expect_err("invalid level should error");
        assert!(format!("{}", err).contains("tool_permissions['search']"));
    }

    #[test]
    fn resolve_tool_permission_with_source_attributes_each_step() {
        // 1. Per-tool override.
        let mut server = bare_server_config("s");
        let mut per_tool = std::collections::HashMap::new();
        per_tool.insert("a".to_string(), "ask".to_string());
        server.tool_permissions = Some(per_tool);
        let (perm, source) =
            resolve_tool_permission_with_source("s", "a", None, &server, None).unwrap();
        assert_eq!(perm, Permission::Ask);
        assert_eq!(source, PermissionSource::ToolOverride);

        // 2. Server-level override.
        let mut server = bare_server_config("s");
        server.permission = Some("read".into());
        let (perm, source) =
            resolve_tool_permission_with_source("s", "b", None, &server, None).unwrap();
        assert_eq!(perm, Permission::Read);
        assert_eq!(source, PermissionSource::ServerOverride);

        // 3. readOnlyHint fires when no user override is set.
        let server = bare_server_config("s");
        let ann = annotations_with_read_only_hint(Some(true));
        let (perm, source) =
            resolve_tool_permission_with_source("s", "c", Some(&ann), &server, None).unwrap();
        assert_eq!(perm, Permission::Read);
        assert_eq!(source, PermissionSource::ReadOnlyHint);

        // 4. Global default when no hint.
        let server = bare_server_config("s");
        let (perm, source) =
            resolve_tool_permission_with_source("s", "d", None, &server, Some(Permission::Read))
                .unwrap();
        assert_eq!(perm, Permission::Read);
        assert_eq!(source, PermissionSource::GlobalDefault);

        // 5. Hardcoded fallback.
        let server = bare_server_config("s");
        let (perm, source) =
            resolve_tool_permission_with_source("s", "e", None, &server, None).unwrap();
        assert_eq!(perm, Permission::Write);
        assert_eq!(source, PermissionSource::Fallback);
    }

    #[test]
    fn permission_source_labels_match_config_keys() {
        // The labels printed by `agsh mcp tools` must match the config
        // keys users would edit to change a classification.
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
        let server = bare_server_config("s");
        assert!(tool_is_allowed(&server, "search"));
        assert!(tool_is_allowed(&server, "create-page"));
    }

    #[test]
    fn tool_is_allowed_allowlist_restricts() {
        let mut server = bare_server_config("s");
        server.allowed_tools = Some(vec!["search".into(), "fetch".into()]);
        assert!(tool_is_allowed(&server, "search"));
        assert!(tool_is_allowed(&server, "fetch"));
        assert!(!tool_is_allowed(&server, "create-page"));
    }

    #[test]
    fn tool_is_allowed_empty_allowlist_means_all() {
        // An empty `allowed_tools` array is treated as "unset" — i.e.
        // no restriction. A totally absent field behaves the same way.
        let mut server = bare_server_config("s");
        server.allowed_tools = Some(Vec::new());
        assert!(tool_is_allowed(&server, "anything"));
    }

    #[test]
    fn tool_is_allowed_blocklist_removes() {
        let mut server = bare_server_config("s");
        server.disabled_tools = Some(vec!["delete-page".into()]);
        assert!(tool_is_allowed(&server, "search"));
        assert!(!tool_is_allowed(&server, "delete-page"));
    }

    #[test]
    fn tool_is_allowed_both_lists_compose() {
        // allow restricts to {search, fetch, write-page}, then block
        // subtracts {write-page}. Net effect: only search + fetch.
        let mut server = bare_server_config("s");
        server.allowed_tools = Some(vec!["search".into(), "fetch".into(), "write-page".into()]);
        server.disabled_tools = Some(vec!["write-page".into()]);
        assert!(tool_is_allowed(&server, "search"));
        assert!(tool_is_allowed(&server, "fetch"));
        assert!(!tool_is_allowed(&server, "write-page"));
        assert!(!tool_is_allowed(&server, "delete-page")); // not in allow
    }

    #[test]
    fn warn_on_stale_tool_config_smoke() {
        // The function just emits `warn!` lines; we can't easily
        // assert on tracing output from a unit test. Smoke-test that
        // the happy path (empty config) doesn't panic and that it
        // accepts a server_config with all four list fields populated
        // plus tool_permissions.
        let mut server = bare_server_config("s");
        server.allowed_tools = Some(vec!["a".into(), "unknown".into()]);
        server.disabled_tools = Some(vec!["b".into(), "gone".into()]);
        server.eager_load_tools = Some(vec!["a".into(), "stale".into(), "b".into()]);
        let mut perms = std::collections::HashMap::new();
        perms.insert("a".to_string(), "read".to_string());
        perms.insert("missing".to_string(), "write".to_string());
        server.tool_permissions = Some(perms);

        let advertised: std::collections::HashSet<&str> =
            ["a", "b", "search"].into_iter().collect();
        // Just confirm the call doesn't panic; "stale" should warn
        // (unknown), and "b" should warn (disabled∩eager overlap).
        warn_on_stale_tool_config("s", &server, &advertised);
    }

    #[test]
    fn tool_should_eager_load_unset_returns_false() {
        let server = bare_server_config("s");
        assert!(!tool_should_eager_load(&server, "search"));
        assert!(!tool_should_eager_load(&server, "anything"));
    }

    #[test]
    fn tool_should_eager_load_empty_list_returns_false() {
        let mut server = bare_server_config("s");
        server.eager_load_tools = Some(Vec::new());
        assert!(!tool_should_eager_load(&server, "search"));
    }

    #[test]
    fn tool_should_eager_load_matching_name_returns_true() {
        let mut server = bare_server_config("s");
        server.eager_load_tools = Some(vec!["search".into(), "fetch".into()]);
        assert!(tool_should_eager_load(&server, "search"));
        assert!(tool_should_eager_load(&server, "fetch"));
    }

    #[test]
    fn tool_should_eager_load_nonmatching_returns_false() {
        let mut server = bare_server_config("s");
        server.eager_load_tools = Some(vec!["search".into()]);
        assert!(!tool_should_eager_load(&server, "create-page"));
    }

    #[test]
    fn tool_should_eager_load_uses_raw_not_namespaced_name() {
        // The check is against the server-advertised raw name; the
        // namespaced `mcp__notion__search` form must NOT match an entry
        // of `"search"` — that would create a footgun where users could
        // accidentally over-match across servers.
        let mut server = bare_server_config("notion");
        server.eager_load_tools = Some(vec!["search".into()]);
        assert!(!tool_should_eager_load(&server, "mcp__notion__search"));
    }

    #[test]
    fn test_truncate_under_limit() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_at_limit() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_over_limit() {
        assert_eq!(truncate("hello world", 5), "hello...");
    }

    #[test]
    fn test_truncate_unicode_boundary() {
        // Three emoji, each multiple bytes: truncation should cut on char boundary.
        let input = "🦀🦀🦀🦀🦀";
        let out = truncate(input, 2);
        assert_eq!(out, "🦀🦀...");
    }

    /// Build a bare server entry in `Pending` state for pure-state tests.
    /// No network, no process spawn.
    fn pending_entry(name: &str, transport: McpTransport) -> Arc<ServerEntry> {
        let mut config = bare_server_config(name);
        config.transport = transport;
        Arc::new(ServerEntry {
            server_name: name.to_string(),
            config,
            token_store: None,
            client_context: McpClientContext::new(),
            state: RwLock::new(ServerState::Pending),
            reconnect_lock: Mutex::new(()),
            instructions: OnceLock::new(),
        })
    }

    #[tokio::test]
    async fn require_connected_errors_for_pending() {
        let entry = pending_entry("pending-srv", McpTransport::Http);
        let err = entry
            .require_connected()
            .await
            .expect_err("pending should not yield a peer");
        match err {
            AgshError::McpConnection {
                server_name,
                message,
            } => {
                assert_eq!(server_name, "pending-srv");
                assert!(message.contains("connecting"), "got: {}", message);
            }
            other => panic!("expected McpConnection, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn require_connected_errors_for_failed() {
        let entry = pending_entry("failed-srv", McpTransport::Http);
        *entry.state.write().await = ServerState::Failed {
            error: "boom".to_string(),
            at: std::time::Instant::now(),
        };
        let err = entry.require_connected().await.unwrap_err();
        assert!(matches!(err, AgshError::McpConnection { .. }));
    }

    #[tokio::test]
    async fn require_connected_errors_for_disabled() {
        let entry = pending_entry("off-srv", McpTransport::Http);
        *entry.state.write().await = ServerState::Disabled;
        let err = entry.require_connected().await.unwrap_err();
        match err {
            AgshError::McpConnection { message, .. } => assert!(message.contains("disabled")),
            other => panic!("expected McpConnection, got: {:?}", other),
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

    #[tokio::test]
    async fn prepare_all_disabled_publishes_settled_immediately() {
        let mut config = bare_server_config("off");
        config.disabled = true;
        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(&[config], None, None, context)
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
        let config = bare_server_config("waiting");
        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(&[config], None, None, context)
            .await
            .expect("prepare should succeed");
        assert!(
            !manager.all_ready(),
            "pending server shouldn't be ready yet"
        );
        let not_ready = manager.enabled_not_connected().await;
        assert_eq!(not_ready.len(), 1);
        assert_eq!(not_ready[0].0, "waiting");
    }

    #[test]
    fn resolve_concurrency_env_uses_default_when_unset() {
        // Unique var names so parallel tests can't race on env state.
        let var = "AGSH_TEST_CONCURRENCY_UNSET";
        unsafe {
            std::env::remove_var(var);
        }
        assert_eq!(resolve_concurrency_env(var, 7), 7);
    }

    #[test]
    fn resolve_concurrency_env_parses_positive_override() {
        let var = "AGSH_TEST_CONCURRENCY_OVERRIDE";
        unsafe {
            std::env::set_var(var, "11");
        }
        assert_eq!(resolve_concurrency_env(var, 3), 11);
        unsafe {
            std::env::remove_var(var);
        }
    }

    #[test]
    fn resolve_concurrency_env_falls_back_on_garbage() {
        let var = "AGSH_TEST_CONCURRENCY_GARBAGE";
        unsafe {
            std::env::set_var(var, "not-a-number");
        }
        assert_eq!(resolve_concurrency_env(var, 5), 5);
        unsafe {
            std::env::remove_var(var);
        }
    }

    #[test]
    fn resolve_concurrency_env_rejects_zero() {
        // Zero would deadlock `buffer_unordered(0)`; must fall back.
        let var = "AGSH_TEST_CONCURRENCY_ZERO";
        unsafe {
            std::env::set_var(var, "0");
        }
        assert_eq!(resolve_concurrency_env(var, 4), 4);
        unsafe {
            std::env::remove_var(var);
        }
    }

    #[tokio::test]
    async fn await_settled_returns_immediately_when_already_settled() {
        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(&[], None, None, context)
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
        // `/bin/false` exits immediately, so the connector reaches
        // `settled.send(true)` via Failed state on the first entry.
        let mut config = bare_server_config("quick-fail");
        config.transport = McpTransport::Stdio;
        config.command = Some("/bin/false".to_string());
        config.url = None;

        let context = McpClientContext::new();
        let manager = McpClientManager::prepare(&[config], None, None, context)
            .await
            .expect("prepare should succeed");
        assert!(!manager.all_ready());

        manager.start_connector(
            crate::tools::ToolRegistry::new(),
            McpRuntimeConfig {
                connect_timeout: std::time::Duration::from_secs(2),
                stdio_concurrency: 1,
                http_concurrency: 1,
            },
        );

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
}
