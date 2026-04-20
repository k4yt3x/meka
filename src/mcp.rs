//! Model Context Protocol (MCP) client integration. Manages the lifecycle of
//! configured MCP servers (stdio child processes or streamable HTTP), exposes
//! their tools through the regular [`crate::tools`] registry, and handles
//! OAuth/JWT authentication for HTTP transports.

pub mod cli;
pub mod elicitation;
pub mod expand;
pub mod progress;
pub mod resource_updates;
pub mod sanitize;

use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Weak};

use async_trait::async_trait;
use rmcp::ErrorData as McpError;
use rmcp::handler::client::ClientHandler;
use rmcp::model::{
    CallToolRequest, CallToolRequestParams, CancelledNotificationParam, ClientRequest,
    CreateElicitationRequestParams, CreateElicitationResult, CreateMessageRequestParams,
    CreateMessageResult, ErrorCode, GetPromptRequestParams, GetPromptResult, ListRootsResult, Meta,
    ProgressNotificationParam, Prompt, ReadResourceRequestParams, ReadResourceResult, Resource,
    Role, Root, SamplingMessage, SamplingMessageContent, ServerResult,
};
use rmcp::service::{NotificationContext, PeerRequestOptions, RequestContext, ServiceError};
use rmcp::transport::auth::OAuthState;
use rmcp::transport::{
    AuthClient, AuthError, AuthorizationManager, ClientCredentialsConfig, CredentialStore,
    StoredCredentials,
};
use rmcp::{Peer, RoleClient};
use tokio::process::Command;
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;

use crate::config::{McpAuthConfig, McpServerConfig, McpTransport};
use crate::error::{AgshError, Result};
use crate::permission::Permission;
use crate::provider::{Provider, ToolDefinition};
use crate::session::TokenStore;
use crate::tools::{Tool, ToolOutput, ToolRegistry};

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

/// Best-effort revoke of a stored OAuth access/refresh token for an MCP
/// server. Looks up the stored credentials, discovers the provider's
/// revocation endpoint via the OAuth authorization server metadata, and
/// posts `token=…&token_type_hint=access_token` per RFC 7009. Errors are
/// propagated so the caller can log them; local credential cleanup should
/// run regardless.
pub async fn revoke_stored_token(
    token_store: &TokenStore,
    server_name: &str,
) -> std::result::Result<(), String> {
    let Some(json) = token_store
        .load_mcp_credentials(server_name)
        .await
        .map_err(|error| format!("load credentials: {}", error))?
    else {
        return Ok(());
    };
    let parsed: serde_json::Value = serde_json::from_str(&json)
        .map_err(|error| format!("stored credentials are not valid JSON: {}", error))?;
    let issuer = parsed
        .get("server_url")
        .and_then(|v| v.as_str())
        .or_else(|| parsed.get("issuer").and_then(|v| v.as_str()))
        .ok_or_else(|| "stored credentials missing issuer/server_url".to_string())?;
    let access_token = parsed
        .get("tokens")
        .and_then(|t| t.get("access_token"))
        .and_then(|v| v.as_str());
    let client_id = parsed
        .get("client")
        .and_then(|c| c.get("client_id"))
        .and_then(|v| v.as_str());

    let Some(access_token) = access_token else {
        return Ok(());
    };

    // Discover the revocation endpoint. RFC 8414 says it lives under
    // /.well-known/oauth-authorization-server; many providers also expose it
    // under /.well-known/openid-configuration. Try OAuth first.
    //
    // Threat model: the `issuer` URL comes from credentials we stored during
    // the original auth flow, so we trust the origin. We do NOT trust the
    // network path or any redirect: reqwest follows redirects by default,
    // which would let a MITM redirect the metadata fetch to an attacker host
    // and coax us into POSTing the access token there. Redirects are turned
    // off, the response body is size-capped, and the returned
    // `revocation_endpoint` is pinned to the same host as the issuer.
    const METADATA_BODY_CAP: usize = 256 * 1024;
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| format!("build http client: {}", error))?;

    let base = issuer.trim_end_matches('/');
    let candidates = [
        format!("{}/.well-known/oauth-authorization-server", base),
        format!("{}/.well-known/openid-configuration", base),
    ];
    let mut revocation_endpoint: Option<String> = None;
    for url in &candidates {
        let Ok(response) = http.get(url).send().await else {
            continue;
        };
        if !response.status().is_success() {
            continue;
        }
        // Read bytes before parsing so we can size-cap: reqwest's own
        // Content-Length is server-supplied and therefore untrusted.
        let Ok(bytes) = response.bytes().await else {
            continue;
        };
        if let Some(endpoint) = extract_revocation_endpoint(&bytes, METADATA_BODY_CAP) {
            revocation_endpoint = Some(endpoint);
            break;
        }
    }

    let Some(endpoint) = revocation_endpoint else {
        return Err(format!(
            "server '{}' does not advertise a revocation_endpoint",
            server_name
        ));
    };

    validate_revocation_endpoint_origin(issuer, &endpoint)?;

    // Build application/x-www-form-urlencoded body manually so we don't need
    // an extra dependency. `form_urlencoded` uses %-encoded UTF-8, same as
    // `percent_encoding::NON_ALPHANUMERIC`.
    use percent_encoding::{NON_ALPHANUMERIC, utf8_percent_encode};
    let enc = |s: &str| utf8_percent_encode(s, NON_ALPHANUMERIC).to_string();
    let mut body = format!("token={}&token_type_hint=access_token", enc(access_token));
    if let Some(id) = client_id {
        body.push_str("&client_id=");
        body.push_str(&enc(id));
    }

    let response = http
        .post(&endpoint)
        .header(
            reqwest::header::CONTENT_TYPE,
            "application/x-www-form-urlencoded",
        )
        .body(body)
        .send()
        .await
        .map_err(|error| format!("revoke POST failed: {}", error))?;
    // RFC 7009: successful revocation is 200 OK with an empty body;
    // unknown tokens also return 200 OK. Non-2xx is a genuine failure.
    if !response.status().is_success() {
        return Err(format!(
            "revoke POST returned HTTP {}",
            response.status().as_u16()
        ));
    }
    Ok(())
}

/// Classification of an unauthenticated probe to an HTTP MCP endpoint.
/// The MCP authorization spec (2025-03-26) layers on RFC 6750 +
/// RFC 9728: a server that requires auth answers unauthenticated
/// requests with `401` and a `WWW-Authenticate: Bearer …` challenge,
/// optionally advertising a `resource_metadata` URL we can fetch to
/// learn which authorization servers + scopes to use.
#[derive(Debug, PartialEq, Eq)]
pub enum McpAuthProbe {
    /// Server answered 2xx — reachable and doesn't require auth.
    Open,
    /// Server answered 401 / 403 with a `Bearer` challenge. The optional
    /// URL is the RFC 9728 protected-resource-metadata document.
    AuthRequired { resource_metadata: Option<String> },
    /// Reachable but some other status (405, 404, …). Record it so the
    /// caller can surface it without claiming auth is or isn't needed.
    Unexpected { status: u16 },
    /// Couldn't even talk to the server (DNS, TLS, timeout, …).
    Unreachable { message: String },
}

/// Probe an MCP HTTP endpoint to see whether it requires OAuth.
///
/// Runs an unauthenticated `GET` with a 3 s wall-clock timeout and
/// redirects disabled; we never follow off-origin so a compromised DNS
/// can't bait us into treating an attacker host as authoritative about
/// the real server. The body is ignored — the verdict comes entirely
/// from the status line and the `WWW-Authenticate` header.
pub async fn probe_http_auth(url: &str) -> McpAuthProbe {
    let http = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .redirect(reqwest::redirect::Policy::none())
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            return McpAuthProbe::Unreachable {
                message: format!("build http client: {}", error),
            };
        }
    };
    let response = match http.get(url).send().await {
        Ok(response) => response,
        Err(error) => {
            return McpAuthProbe::Unreachable {
                message: error.to_string(),
            };
        }
    };
    let status = response.status().as_u16();
    let www_authenticate = response
        .headers()
        .get(reqwest::header::WWW_AUTHENTICATE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    classify_probe_response(status, www_authenticate.as_deref())
}

/// Pure classifier for the probe — takes the status code + optional
/// `WWW-Authenticate` header and returns the [`McpAuthProbe`] verdict.
/// Extracted so the RFC-6750 / RFC-9728 parsing can be unit-tested
/// without a live HTTP server.
fn classify_probe_response(status: u16, www_authenticate: Option<&str>) -> McpAuthProbe {
    if (200..300).contains(&status) {
        return McpAuthProbe::Open;
    }
    if status == 401 || status == 403 {
        let header = www_authenticate.unwrap_or("");
        // RFC 6750 §3: the challenge must start with `Bearer`. Match
        // case-insensitively; tolerate the scheme with or without any
        // `key=value` parameters following.
        let first = header.split(',').next().unwrap_or("").trim();
        let is_bearer = first.eq_ignore_ascii_case("Bearer")
            || first
                .split_once(|c: char| c.is_whitespace())
                .map(|(scheme, _)| scheme.eq_ignore_ascii_case("Bearer"))
                .unwrap_or(false);
        if !is_bearer {
            return McpAuthProbe::Unexpected { status };
        }
        return McpAuthProbe::AuthRequired {
            resource_metadata: extract_bearer_param(header, "resource_metadata"),
        };
    }
    McpAuthProbe::Unexpected { status }
}

/// Extract a quoted parameter value from an RFC 6750 `WWW-Authenticate:
/// Bearer …` challenge. Handles the forms seen in the wild:
/// `key="value"`, `key=value`, trailing commas, mixed whitespace.
/// Returns `None` if the key isn't present.
fn extract_bearer_param(header: &str, key: &str) -> Option<String> {
    // Drop the `Bearer` scheme prefix; everything after is a
    // comma-separated parameter list.
    let params = match header.find(|c: char| c.is_whitespace()) {
        Some(idx) => &header[idx..],
        None => return None,
    };
    for pair in params.split(',') {
        let pair = pair.trim();
        let Some((k, v)) = pair.split_once('=') else {
            continue;
        };
        if !k.trim().eq_ignore_ascii_case(key) {
            continue;
        }
        let v = v.trim();
        let unquoted = v
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .unwrap_or(v);
        return Some(unquoted.to_string());
    }
    None
}

/// Parse an OAuth authorization-server metadata JSON document and return
/// the `revocation_endpoint` string, if any. Rejects bodies larger than
/// `max_bytes` and invalid JSON. Split from [`revoke_stored_token`] so the
/// size-cap and extraction logic are testable without a live HTTP server.
fn extract_revocation_endpoint(bytes: &[u8], max_bytes: usize) -> Option<String> {
    if bytes.len() > max_bytes {
        return None;
    }
    let metadata = serde_json::from_slice::<serde_json::Value>(bytes).ok()?;
    metadata
        .get("revocation_endpoint")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Verify that `endpoint` has the same scheme, host, and effective port as
/// `issuer`. Prevents a compromised metadata document from redirecting the
/// access-token POST to an attacker-controlled host.
fn validate_revocation_endpoint_origin(
    issuer: &str,
    endpoint: &str,
) -> std::result::Result<(), String> {
    let issuer_url = reqwest::Url::parse(issuer)
        .map_err(|error| format!("stored issuer '{}' is not a valid URL: {}", issuer, error))?;
    let endpoint_url = reqwest::Url::parse(endpoint).map_err(|error| {
        format!(
            "revocation_endpoint '{}' is not a valid URL: {}",
            endpoint, error
        )
    })?;
    if endpoint_url.scheme() != issuer_url.scheme()
        || endpoint_url.host_str() != issuer_url.host_str()
        || endpoint_url.port_or_known_default() != issuer_url.port_or_known_default()
    {
        return Err(format!(
            "revocation_endpoint '{}' is on a different origin than issuer '{}'; refusing to send token",
            endpoint, issuer
        ));
    }
    Ok(())
}

type McpRunningService = rmcp::service::RunningService<RoleClient, AgshClientHandler>;

pub struct McpClientManager {
    servers: HashMap<String, Arc<ServerEntry>>,
    /// Global fallback permission from `[mcp].default_permission`.
    /// Consulted by `resolve_tool_permission` at tool-registration time
    /// when neither the server nor the user has configured a more
    /// specific permission and the server didn't advertise a
    /// `readOnlyHint`. `None` means "no user default" — resolution
    /// falls through to the hardcoded strict `Write`.
    mcp_default_permission: Option<Permission>,
}

/// Holds the live service for a single MCP server plus reconnection state.
/// Wrapped in an [`Arc`] and shared between the manager, per-server tool
/// adapters, and the resource/prompt builtin tools so they all see the
/// current service after a reconnect.
pub struct ServerEntry {
    server_name: String,
    config: McpServerConfig,
    token_store: Option<TokenStore>,
    client_context: Arc<McpClientContext>,
    service: RwLock<Arc<McpRunningService>>,
    reconnect_lock: Mutex<()>,
    /// Optional `InitializeResult.instructions` captured at connect-time.
    /// Immutable for the lifetime of the connection per the MCP spec.
    instructions: OnceLock<Option<String>>,
}

impl ServerEntry {
    /// Returns the server's `InitializeResult.instructions` (sanitised +
    /// truncated to [`MAX_MCP_DESCRIPTION_LENGTH`]) if the server advertised
    /// one during the handshake.
    pub fn instructions(&self) -> Option<&str> {
        self.instructions.get().and_then(|opt| opt.as_deref())
    }
}

impl ServerEntry {
    async fn current_peer(&self) -> Peer<RoleClient> {
        self.service.read().await.peer().clone()
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
    async fn reconnect(self: &Arc<Self>) -> Result<()> {
        let _guard = self.reconnect_lock.lock().await;

        if !self.service.read().await.peer().is_transport_closed() {
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
                handle.block_on(connect_server(
                    &server_name,
                    &config,
                    token_store.as_ref(),
                    &client_context,
                ))
            })
            .await;

            match result {
                Ok(Ok(new_service)) => {
                    *self.service.write().await = Arc::new(new_service);
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

impl McpClientManager {
    pub async fn connect_all(
        configs: &[McpServerConfig],
        mcp_default_permission: Option<Permission>,
        token_store: Option<&TokenStore>,
        client_context: Arc<McpClientContext>,
    ) -> Result<Self> {
        let mut servers = HashMap::new();

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
            // our `<server>__<tool>` namespace separator.
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

            let service =
                match connect_server(&config.name, &config, token_store, &client_context).await {
                    Ok(service) => service,
                    Err(error) => {
                        tracing::warn!(
                            "failed to connect to MCP server '{}': {}",
                            config.name,
                            error
                        );
                        continue;
                    }
                };
            tracing::info!("connected to MCP server '{}'", config.name);
            let instructions_slot: OnceLock<Option<String>> = OnceLock::new();
            let captured = service
                .peer()
                .peer_info()
                .and_then(|info| info.instructions.as_ref())
                .map(|raw| {
                    crate::mcp::truncate(
                        &crate::mcp::sanitize::sanitize_text(raw),
                        MAX_MCP_DESCRIPTION_LENGTH,
                    )
                });
            let _ = instructions_slot.set(captured);

            let entry = Arc::new(ServerEntry {
                server_name: config.name.clone(),
                config: config.clone(),
                token_store: token_store.cloned(),
                client_context: Arc::clone(&client_context),
                service: RwLock::new(Arc::new(service)),
                reconnect_lock: Mutex::new(()),
                instructions: instructions_slot,
            });
            servers.insert(config.name.clone(), entry);
        }

        Ok(Self {
            servers,
            mcp_default_permission,
        })
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

        let peer = entry.current_peer().await;
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
            let namespaced_name = format!("{}__{}", server_name, sanitised_tool_name);

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

            adapters.push(McpToolAdapter {
                namespaced_name,
                remote_tool_name: raw_tool_name,
                description,
                parameters,
                permission,
                entry: Arc::clone(entry),
                annotations,
                meta,
                title,
            });
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
        let peer = entry.current_peer().await;
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

            let service = entry.service.into_inner();

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

/// Build a [`Command`] for a stdio MCP server, wrapping shell shims in
/// `cmd /c` on Windows so `npx`, `*.cmd`, and `*.bat` executables can be
/// launched directly as a command string. Unix paths pass through unchanged.
pub fn build_stdio_command(command_str: &str, args: &[String]) -> Command {
    #[cfg(windows)]
    {
        let lower = command_str.to_ascii_lowercase();
        let is_shim = lower == "npx"
            || lower == "yarn"
            || lower == "pnpm"
            || lower.ends_with(".cmd")
            || lower.ends_with(".bat")
            || lower.ends_with(".ps1");
        if is_shim {
            // `cmd /c <command> <args...>` — Windows wraps argument quoting.
            // We don't try to shell-quote the args; the `Command` API does
            // the OS-appropriate escaping via CreateProcess's lpCommandLine.
            let mut cmd = Command::new("cmd");
            cmd.arg("/c").arg(command_str).args(args);
            return cmd;
        }
    }
    let _ = (command_str, args);
    let mut cmd = Command::new(command_str);
    cmd.args(args);
    cmd
}

/// Connect to an MCP server, dispatching to the auth or no-auth path. This
/// function is only called from top-level startup code (e.g. `connect_all`)
/// where a `Send` future isn't required — the OAuth path pulls in an rmcp
/// auth future that is `!Send`.
/// Connect to an MCP server. The returned future is `!Send` when the server
/// config uses OAuth (rmcp 1.5's auth module holds a `!Sync` closure across
/// an await). Callers that need a `Send` future (e.g. `Tool::execute` during
/// reconnect) drive this on a `spawn_blocking` thread via
/// [`ServerEntry::reconnect`].
async fn connect_server(
    server_name: &str,
    config: &McpServerConfig,
    token_store: Option<&TokenStore>,
    client_context: &Arc<McpClientContext>,
) -> Result<McpRunningService> {
    use rmcp::ServiceExt;

    let handler = AgshClientHandler::new(
        server_name.to_string(),
        SamplingPolicy::from_config(config),
        Arc::clone(client_context),
    );

    match config.transport {
        McpTransport::Stdio => {
            let command_str =
                config
                    .command
                    .as_deref()
                    .ok_or_else(|| AgshError::McpConnection {
                        server_name: server_name.to_string(),
                        message: "stdio transport requires 'command' field".to_string(),
                    })?;

            let args_vec: Vec<String> = config.args.clone().unwrap_or_default();
            let command = build_stdio_command(command_str, &args_vec);
            let mut command = command;
            if let Some(env) = &config.env {
                command.envs(env);
            }

            let transport = rmcp::transport::TokioChildProcess::new(command).map_err(|error| {
                AgshError::McpConnection {
                    server_name: server_name.to_string(),
                    message: format!("failed to spawn process: {}", error),
                }
            })?;

            handler
                .serve(transport)
                .await
                .map_err(|error| AgshError::McpConnection {
                    server_name: server_name.to_string(),
                    message: format!("handshake failed: {}", error),
                })
        }
        McpTransport::Http => {
            let url = config
                .url
                .as_deref()
                .ok_or_else(|| AgshError::McpConnection {
                    server_name: server_name.to_string(),
                    message: "http transport requires 'url' field".to_string(),
                })?;

            // Consult the auth-probe cache: if a prior connect returned 401
            // recently and we have no stored creds, skip the unauthenticated
            // probe and drive straight into the OAuth flow. The cache entry
            // is cleared on a successful connect below.
            if config.auth.is_some()
                && let Some(store) = token_store
            {
                match store.load_auth_probe(server_name, MCP_AUTH_CACHE_TTL).await {
                    Ok(Some(true)) => {
                        tracing::info!(
                            "MCP server '{}': cached 'needs-auth' verdict (<{:?} old), going straight to OAuth",
                            server_name,
                            MCP_AUTH_CACHE_TTL
                        );
                    }
                    Ok(_) => {}
                    Err(error) => tracing::debug!(
                        "auth probe cache lookup for '{}' failed: {}",
                        server_name,
                        error
                    ),
                }
            }

            let transport_config = build_http_transport_config(server_name, config)?;

            if let Some(auth_config) = &config.auth {
                connect_http_with_oauth(
                    server_name,
                    url,
                    auth_config,
                    transport_config,
                    token_store,
                    handler,
                )
                .await
            } else {
                let transport =
                    rmcp::transport::StreamableHttpClientTransport::from_config(transport_config);

                handler
                    .serve(transport)
                    .await
                    .map_err(|error| AgshError::McpConnection {
                        server_name: server_name.to_string(),
                        message: format!("HTTP connection failed: {}", error),
                    })
            }
        }
    }
}

/// Build the shared HTTP transport config (URL, bearer token, custom headers)
/// used by both the auth and no-auth paths.
fn build_http_transport_config(
    server_name: &str,
    config: &McpServerConfig,
) -> Result<rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig> {
    let url = config
        .url
        .as_deref()
        .ok_or_else(|| AgshError::McpConnection {
            server_name: server_name.to_string(),
            message: "http transport requires 'url' field".to_string(),
        })?;

    let mut transport_config =
        rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig::with_uri(url);

    if let Some(token) = &config.auth_token {
        transport_config = transport_config.auth_header(token.clone());
    }

    // Merge dynamic headers from the optional `headers_helper` script on
    // top of the static `headers` map (dynamic values override static ones).
    let mut merged_headers: std::collections::HashMap<String, String> =
        config.headers.clone().unwrap_or_default();
    if let Some(script) = &config.headers_helper {
        let dynamic = run_headers_helper(server_name, url, script)?;
        merged_headers.extend(dynamic);
    }

    if !merged_headers.is_empty() {
        let mut header_map = std::collections::HashMap::new();
        for (key, value) in &merged_headers {
            let header_name =
                reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|error| {
                    AgshError::McpConnection {
                        server_name: server_name.to_string(),
                        message: format!("invalid header name '{}': {}", key, error),
                    }
                })?;
            let header_value = reqwest::header::HeaderValue::from_str(value).map_err(|error| {
                AgshError::McpConnection {
                    server_name: server_name.to_string(),
                    message: format!("invalid header value for '{}': {}", key, error),
                }
            })?;
            header_map.insert(header_name, header_value);
        }
        transport_config = transport_config.custom_headers(header_map);
    }

    Ok(transport_config)
}

/// Execute `headers_helper` and parse its stdout as an `Name: Value\n`
/// stream, returning a map merged into the HTTP transport's custom headers.
///
/// The script is spawned synchronously (it's a startup-path helper, not
/// called per-request) with a 15-second wall-clock timeout. `AGSH_MCP_SERVER_NAME`
/// and `AGSH_MCP_SERVER_URL` are injected so one helper can serve multiple
/// servers.
fn run_headers_helper(
    server_name: &str,
    url: &str,
    script: &str,
) -> Result<std::collections::HashMap<String, String>> {
    use std::process::Stdio;
    let err_ctx = |msg: String| AgshError::McpConnection {
        server_name: server_name.to_string(),
        message: msg,
    };

    // Resolve the script path. If it's relative and doesn't exist as-is,
    // try resolving against the agsh config directory for safety (same place
    // config.toml lives).
    let script_path = std::path::Path::new(script);
    let resolved: std::path::PathBuf = if script_path.is_absolute() || script_path.exists() {
        script_path.to_path_buf()
    } else if let Some(config_dir) = crate::config::agsh_config_dir() {
        let candidate = config_dir.join(script);
        if candidate.exists() {
            candidate
        } else {
            script_path.to_path_buf()
        }
    } else {
        script_path.to_path_buf()
    };

    let mut command = std::process::Command::new(&resolved);
    command
        .env("AGSH_MCP_SERVER_NAME", server_name)
        .env("AGSH_MCP_SERVER_URL", url)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().map_err(|error| {
        err_ctx(format!(
            "headers_helper '{}' spawn failed: {}",
            script, error
        ))
    })?;

    // Poll for exit with a 15-second budget. std::process::Child doesn't
    // expose a blocking wait_timeout, so loop on try_wait with a short sleep.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    return Err(err_ctx(format!(
                        "headers_helper '{}' timed out after 15s",
                        script
                    )));
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(error) => {
                return Err(err_ctx(format!(
                    "headers_helper '{}' wait failed: {}",
                    script, error
                )));
            }
        }
    };

    // Caps on how much helper output we're willing to buffer. stdout is the
    // header list (rarely more than a few KiB); stderr is surfaced verbatim
    // in the error message so keep it tight.
    const MAX_HELPER_STDOUT_BYTES: u64 = 64 * 1024;
    const MAX_HELPER_STDERR_BYTES: u64 = 4 * 1024;

    if !status.success() {
        let mut stderr_buf = Vec::new();
        if let Some(stderr) = child.stderr.take() {
            use std::io::Read;
            let _ = stderr
                .take(MAX_HELPER_STDERR_BYTES)
                .read_to_end(&mut stderr_buf);
        }
        let stderr_text = String::from_utf8_lossy(&stderr_buf);
        return Err(err_ctx(format!(
            "headers_helper '{}' exited with status {}: {}",
            script,
            status.code().unwrap_or(-1),
            stderr_text.trim()
        )));
    }

    let mut stdout_buf = Vec::new();
    if let Some(pipe) = child.stdout.take() {
        use std::io::Read;
        pipe.take(MAX_HELPER_STDOUT_BYTES)
            .read_to_end(&mut stdout_buf)
            .map_err(|error| {
                err_ctx(format!(
                    "headers_helper '{}' stdout read failed: {}",
                    script, error
                ))
            })?;
    }
    let stdout = String::from_utf8_lossy(&stdout_buf);

    parse_header_lines(&stdout)
        .map_err(|msg| err_ctx(format!("headers_helper '{}' output: {}", script, msg)))
}

fn parse_header_lines(
    text: &str,
) -> std::result::Result<std::collections::HashMap<String, String>, String> {
    let mut out = std::collections::HashMap::new();
    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| format!("line {} missing ':' separator", line_no + 1))?;
        out.insert(name.trim().to_string(), value.trim().to_string());
    }
    Ok(out)
}

async fn connect_http_with_oauth(
    server_name: &str,
    url: &str,
    auth_config: &McpAuthConfig,
    transport_config: rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig,
    token_store: Option<&TokenStore>,
    handler: AgshClientHandler,
) -> Result<McpRunningService> {
    use rmcp::ServiceExt;

    let auth_manager = match auth_config {
        McpAuthConfig::ClientCredentials {
            client_id,
            client_secret,
            scopes,
            resource,
        } => {
            authenticate_client_credentials(
                server_name,
                url,
                client_id,
                client_secret,
                scopes.as_deref(),
                resource.as_deref(),
            )
            .await?
        }
        McpAuthConfig::ClientCredentialsJwt {
            client_id,
            signing_key_path,
            signing_algorithm,
            scopes,
            resource,
        } => {
            authenticate_client_credentials_jwt(
                server_name,
                url,
                client_id,
                signing_key_path,
                signing_algorithm.as_deref(),
                scopes.as_deref(),
                resource.as_deref(),
            )
            .await?
        }
        McpAuthConfig::OAuth {
            client_id,
            client_secret,
            scopes,
            redirect_port,
        } => {
            authenticate_oauth_authorization_code(
                server_name,
                url,
                client_id.as_deref(),
                client_secret.as_deref(),
                scopes.as_deref(),
                *redirect_port,
                token_store,
            )
            .await?
        }
    };

    let auth_client = AuthClient::new(reqwest::Client::new(), auth_manager);
    let transport =
        rmcp::transport::StreamableHttpClientTransport::with_client(auth_client, transport_config);

    handler
        .serve(transport)
        .await
        .map_err(|error| AgshError::McpConnection {
            server_name: server_name.to_string(),
            message: format!("HTTP connection failed: {}", error),
        })
}

async fn authenticate_client_credentials(
    server_name: &str,
    url: &str,
    client_id: &str,
    client_secret: &str,
    scopes: Option<&[String]>,
    resource: Option<&str>,
) -> Result<AuthorizationManager> {
    let config = ClientCredentialsConfig::ClientSecret {
        client_id: client_id.to_string(),
        client_secret: client_secret.to_string(),
        scopes: scopes.map(|s| s.to_vec()).unwrap_or_default(),
        resource: resource.map(|s| s.to_string()),
    };

    let mut oauth_state = OAuthState::new(url, None)
        .await
        .map_err(|error| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("failed to initialize OAuth: {}", error),
        })?;

    oauth_state
        .authenticate_client_credentials(config)
        .await
        .map_err(|error| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("client credentials authentication failed: {}", error),
        })?;

    oauth_state
        .into_authorization_manager()
        .ok_or_else(|| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: "unexpected OAuth state after client credentials authentication".to_string(),
        })
}

async fn authenticate_client_credentials_jwt(
    server_name: &str,
    url: &str,
    client_id: &str,
    signing_key_path: &str,
    signing_algorithm: Option<&str>,
    scopes: Option<&[String]>,
    resource: Option<&str>,
) -> Result<AuthorizationManager> {
    require_private_key_permissions(server_name, signing_key_path)?;
    let signing_key = std::fs::read(signing_key_path).map_err(|error| AgshError::McpAuth {
        server_name: server_name.to_string(),
        message: format!(
            "failed to read signing key '{}': {}",
            signing_key_path, error
        ),
    })?;

    let algorithm = parse_jwt_signing_algorithm(server_name, signing_algorithm)?;

    let config = ClientCredentialsConfig::PrivateKeyJwt {
        client_id: client_id.to_string(),
        signing_key,
        signing_algorithm: algorithm,
        scopes: scopes.map(|s| s.to_vec()).unwrap_or_default(),
        resource: resource.map(|s| s.to_string()),
        token_endpoint_audience: None,
    };

    let mut oauth_state = OAuthState::new(url, None)
        .await
        .map_err(|error| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("failed to initialize OAuth: {}", error),
        })?;

    oauth_state
        .authenticate_client_credentials(config)
        .await
        .map_err(|error| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("JWT client credentials authentication failed: {}", error),
        })?;

    oauth_state
        .into_authorization_manager()
        .ok_or_else(|| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: "unexpected OAuth state after JWT authentication".to_string(),
        })
}

/// On Unix, refuse to read a JWT signing key that is group- or world-
/// accessible. Matches the 0600-only policy already applied to the session
/// DB and config.toml: if the key can be read by another local user, a
/// local attacker can forge JWTs to the MCP server and impersonate us.
///
/// No-op on non-Unix: Windows uses ACLs and we don't try to audit them.
fn require_private_key_permissions(server_name: &str, path: &str) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata = std::fs::metadata(path).map_err(|error| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("failed to stat signing key '{}': {}", path, error),
        })?;
        let mode = metadata.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            return Err(AgshError::McpAuth {
                server_name: server_name.to_string(),
                message: format!(
                    "signing key '{}' has permissions {:o}; must be 0600 (group/other bits must be clear)",
                    path, mode
                ),
            });
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (server_name, path);
    }
    Ok(())
}

fn parse_jwt_signing_algorithm(
    server_name: &str,
    algorithm: Option<&str>,
) -> Result<rmcp::transport::JwtSigningAlgorithm> {
    use rmcp::transport::JwtSigningAlgorithm;

    match algorithm.unwrap_or("RS256") {
        "RS256" => Ok(JwtSigningAlgorithm::RS256),
        "RS384" => Ok(JwtSigningAlgorithm::RS384),
        "RS512" => Ok(JwtSigningAlgorithm::RS512),
        "ES256" => Ok(JwtSigningAlgorithm::ES256),
        "ES384" => Ok(JwtSigningAlgorithm::ES384),
        other => Err(AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!(
                "unsupported signing algorithm '{}': expected RS256, RS384, RS512, ES256, or ES384",
                other
            ),
        }),
    }
}

async fn authenticate_oauth_authorization_code(
    server_name: &str,
    url: &str,
    client_id: Option<&str>,
    client_secret: Option<&str>,
    scopes: Option<&[String]>,
    redirect_port: Option<u16>,
    token_store: Option<&TokenStore>,
) -> Result<AuthorizationManager> {
    // Bind the callback listener up-front so we can support a random ephemeral
    // port (`redirect_port = None` → bind 0) and learn the actual port before
    // constructing `redirect_uri`. This avoids the "port 8400 already in use"
    // failure mode and lets multiple concurrent agsh sessions coexist.
    let bind_port = redirect_port.unwrap_or(0);
    let callback_listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", bind_port))
        .await
        .map_err(|error| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!(
                "failed to bind callback server on port {}: {}",
                bind_port, error
            ),
        })?;
    let actual_port = callback_listener
        .local_addr()
        .map_err(|error| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("callback listener local_addr failed: {}", error),
        })?
        .port();
    let redirect_uri = format!("http://127.0.0.1:{}/callback", actual_port);
    let scope_strings: Vec<String> = scopes.map(|s| s.to_vec()).unwrap_or_default();

    let mut manager = AuthorizationManager::new(url)
        .await
        .map_err(|error| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("failed to initialize OAuth manager: {}", error),
        })?;

    // Set up persistent credential storage if available
    if let Some(store) = token_store {
        manager.set_credential_store(SqliteCredentialStore {
            token_store: store.clone(),
            server_name: server_name.to_string(),
        });
    }

    // Try loading existing credentials from the store
    let has_stored_credentials =
        manager
            .initialize_from_store()
            .await
            .map_err(|error| AgshError::McpAuth {
                server_name: server_name.to_string(),
                message: format!("failed to load stored credentials: {}", error),
            })?;

    if has_stored_credentials {
        tracing::info!(
            "loaded stored OAuth credentials for MCP server '{}'",
            server_name
        );
        return Ok(manager);
    }

    // No stored credentials — run the interactive browser flow
    tracing::info!(
        "starting OAuth authorization flow for MCP server '{}'",
        server_name
    );

    // Wrap in OAuthState to use its start_authorization flow which handles
    // metadata discovery, dynamic client registration, and PKCE setup
    let mut oauth_state = OAuthState::Unauthorized(manager);

    // If we have a pre-configured client_id, configure it before starting
    if let Some(id) = client_id {
        // We need to discover metadata first, then configure the client
        if let OAuthState::Unauthorized(ref mut manager) = oauth_state {
            let metadata =
                manager
                    .discover_metadata()
                    .await
                    .map_err(|error| AgshError::McpAuth {
                        server_name: server_name.to_string(),
                        message: format!("OAuth metadata discovery failed: {}", error),
                    })?;
            manager.set_metadata(metadata);

            let mut oauth_client_config =
                rmcp::transport::auth::OAuthClientConfig::new(id.to_string(), redirect_uri.clone());
            if let Some(secret) = client_secret {
                oauth_client_config = oauth_client_config.with_client_secret(secret.to_string());
            }
            if !scope_strings.is_empty() {
                oauth_client_config = oauth_client_config.with_scopes(scope_strings.clone());
            }
            manager
                .configure_client(oauth_client_config)
                .map_err(|error| AgshError::McpAuth {
                    server_name: server_name.to_string(),
                    message: format!("failed to configure OAuth client: {}", error),
                })?;

            let scope_refs: Vec<&str> = scope_strings.iter().map(|s| s.as_str()).collect();
            let auth_url = manager
                .get_authorization_url(&scope_refs)
                .await
                .map_err(|error| AgshError::McpAuth {
                    server_name: server_name.to_string(),
                    message: format!("failed to get authorization URL: {}", error),
                })?;

            let session = rmcp::transport::AuthorizationSession::for_scope_upgrade(
                std::mem::replace(
                    manager,
                    AuthorizationManager::new("http://localhost")
                        .await
                        .map_err(|error| AgshError::McpAuth {
                            server_name: server_name.to_string(),
                            message: format!("internal error: {}", error),
                        })?,
                ),
                auth_url,
                &redirect_uri,
            );
            oauth_state = OAuthState::Session(session);
        }
    } else {
        // No client_id configured — use dynamic registration via start_authorization
        let scope_refs: Vec<&str> = scope_strings.iter().map(|s| s.as_str()).collect();
        oauth_state
            .start_authorization(&scope_refs, &redirect_uri, Some("agsh"))
            .await
            .map_err(|error| AgshError::McpAuth {
                server_name: server_name.to_string(),
                message: format!("failed to start OAuth authorization: {}", error),
            })?;
    }

    let auth_url =
        oauth_state
            .get_authorization_url()
            .await
            .map_err(|error| AgshError::McpAuth {
                server_name: server_name.to_string(),
                message: format!("failed to get authorization URL: {}", error),
            })?;

    // Print the URL exactly once and try to open the browser silently.
    // Browser-launch failures are expected on headless hosts (SSH, CI,
    // containers), so they stay at `debug` — the user has the URL and
    // can copy it either way.
    eprintln!("open this URL in your browser to authorize:\n\n{auth_url}\n");
    if let Err(error) = open::that(&auth_url) {
        tracing::debug!("open::that failed to launch browser: {}", error);
    }

    // Wait for the authorization code on our pre-bound listener.
    let (code, state) = await_oauth_callback(callback_listener)
        .await
        .map_err(|error| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("OAuth callback failed: {}", error),
        })?;

    // Exchange the authorization code for tokens
    oauth_state
        .handle_callback(&code, &state)
        .await
        .map_err(|error| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: format!("OAuth token exchange failed: {}", error),
        })?;

    oauth_state
        .into_authorization_manager()
        .ok_or_else(|| AgshError::McpAuth {
            server_name: server_name.to_string(),
            message: "unexpected OAuth state after authorization".to_string(),
        })
}

/// Max bytes we're willing to read from a single HTTP callback request
/// before giving up. Large enough to handle big `Cookie:` headers (which can
/// exceed 4 KiB), small enough to cap a resource-exhaustion attempt.
const CALLBACK_READ_CAP: usize = 64 * 1024;
/// End-of-headers marker for HTTP/1.x.
const CRLF_CRLF: &[u8] = b"\r\n\r\n";

/// Overall wall-clock budget for the OAuth callback wait, shared by
/// both the TCP accept path and the paste-URL fallback.
const OAUTH_CALLBACK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Wait for the authorization code.
///
/// The common case is that the OAuth provider redirects the user's
/// browser to our localhost listener and we pick the code out of the
/// HTTP request. But when agsh runs on a different host than the
/// browser — SSH sessions, containers, remote Codespaces — the browser
/// can't reach back, so we race the TCP accept against a stdin prompt
/// that lets the user paste the full callback URL (it's visible in
/// the browser's address bar even when the connection is refused).
/// Paste mode is only offered when stdin is a TTY.
async fn await_oauth_callback(
    listener: tokio::net::TcpListener,
) -> std::result::Result<(String, String), String> {
    let deadline = tokio::time::Instant::now() + OAUTH_CALLBACK_TIMEOUT;
    let paste_enabled = std::io::IsTerminal::is_terminal(&std::io::stdin());

    if paste_enabled {
        eprintln!(
            "waiting up to {}s for the callback — or paste the callback URL here and press Enter:",
            OAUTH_CALLBACK_TIMEOUT.as_secs()
        );
    } else {
        eprintln!(
            "waiting up to {}s for the callback.",
            OAUTH_CALLBACK_TIMEOUT.as_secs()
        );
        return accept_http_callback(listener, deadline).await;
    }

    tokio::select! {
        result = accept_http_callback(listener, deadline) => result,
        result = read_pasted_callback(deadline) => result,
    }
}

/// Accept one HTTP request on the bound listener, validate it's the
/// OAuth callback, extract `code` and `state`, and send back a success
/// page. Loops past non-callback requests (favicons, preflights) until
/// the shared deadline elapses.
async fn accept_http_callback(
    listener: tokio::net::TcpListener,
    overall_deadline: tokio::time::Instant,
) -> std::result::Result<(String, String), String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    loop {
        let remaining = overall_deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Err(format!(
                "authorization timed out after {}s",
                OAUTH_CALLBACK_TIMEOUT.as_secs()
            ));
        }

        let accept_result = tokio::time::timeout(remaining, listener.accept()).await;
        let (mut stream, _addr) = match accept_result {
            Err(_) => {
                return Err(format!(
                    "authorization timed out after {}s",
                    OAUTH_CALLBACK_TIMEOUT.as_secs()
                ));
            }
            Ok(Err(error)) => return Err(format!("failed to accept connection: {}", error)),
            Ok(Ok(pair)) => pair,
        };

        // Read until we've seen CRLF-CRLF (end of request headers) or hit the
        // byte cap. Browsers sometimes send favicon / preflight requests to
        // the callback origin; if the path isn't `/callback?...`, respond
        // with a minimal 404 so the browser stops retrying and keep waiting.
        let mut buffer = Vec::with_capacity(4096);
        let mut temp = [0u8; 4096];
        let headers_complete = loop {
            if buffer.windows(CRLF_CRLF.len()).any(|w| w == CRLF_CRLF) {
                break true;
            }
            if buffer.len() >= CALLBACK_READ_CAP {
                break false;
            }
            let read_remaining =
                overall_deadline.saturating_duration_since(tokio::time::Instant::now());
            if read_remaining.is_zero() {
                return Err(format!(
                    "authorization timed out after {}s",
                    OAUTH_CALLBACK_TIMEOUT.as_secs()
                ));
            }
            match tokio::time::timeout(read_remaining, stream.read(&mut temp)).await {
                Err(_) => {
                    return Err(format!(
                        "authorization timed out after {}s",
                        OAUTH_CALLBACK_TIMEOUT.as_secs()
                    ));
                }
                Ok(Err(error)) => return Err(format!("failed to read request: {}", error)),
                Ok(Ok(0)) => break buffer.windows(CRLF_CRLF.len()).any(|w| w == CRLF_CRLF),
                Ok(Ok(n)) => buffer.extend_from_slice(&temp[..n]),
            }
        };

        if !headers_complete {
            tracing::debug!(
                "OAuth callback: dropped request with incomplete/oversized headers \
                 ({} bytes)",
                buffer.len()
            );
            let _ = stream
                .write_all(b"HTTP/1.1 431 Request Header Fields Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .await;
            continue;
        }

        let request = String::from_utf8_lossy(&buffer);
        match parse_callback_query(&request) {
            Ok((code, state)) => {
                let response_body = "<!DOCTYPE html><html><body>\
                    <h1>Authorization successful</h1>\
                    <p>You can close this tab and return to agsh.</p>\
                    </body></html>";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                if let Err(error) = stream.write_all(response.as_bytes()).await {
                    tracing::debug!("failed to send callback response: {}", error);
                }
                return Ok((code, state));
            }
            Err(CallbackParseError::NotCallbackPath) => {
                // Almost certainly a browser preflight or favicon request.
                // Respond 404 and keep waiting for the real callback.
                tracing::debug!("OAuth callback: ignored non-callback request on callback port");
                let _ = stream
                    .write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .await;
                continue;
            }
            Err(CallbackParseError::Malformed(message)) => {
                let _ = stream
                    .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    .await;
                return Err(message);
            }
        }
    }
}

/// Paste-URL fallback for the OAuth callback: prompt on stderr, read a
/// line from stdin, extract `code` + `state` from the pasted URL. Used
/// when the browser can't reach back to our bound listener (e.g. agsh
/// is on an SSH host and the browser is on the user's laptop).
async fn read_pasted_callback(
    deadline: tokio::time::Instant,
) -> std::result::Result<(String, String), String> {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
    if remaining.is_zero() {
        return Err(format!(
            "authorization timed out after {}s",
            OAUTH_CALLBACK_TIMEOUT.as_secs()
        ));
    }
    let mut reader = BufReader::new(tokio::io::stdin());
    let mut line = String::new();
    match tokio::time::timeout(remaining, reader.read_line(&mut line)).await {
        Err(_) => Err(format!(
            "authorization timed out after {}s",
            OAUTH_CALLBACK_TIMEOUT.as_secs()
        )),
        Ok(Err(error)) => Err(format!("stdin read failed: {}", error)),
        // `read_line` returning 0 means EOF — stdin was closed before the
        // user pasted anything. Don't treat this as a fatal error; let
        // the TCP branch of the `select!` continue waiting.
        Ok(Ok(0)) => std::future::pending().await,
        Ok(Ok(_)) => parse_pasted_callback(&line),
    }
}

/// Extract `(code, state)` from a pasted callback URL. Accepts either
/// the full URL or just the query string, percent-decodes the values,
/// and surfaces the `error=…` parameter (sanitised) when the
/// authorization server declines.
fn parse_pasted_callback(input: &str) -> std::result::Result<(String, String), String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("no callback URL pasted".to_string());
    }
    // Drop any URL fragment, then narrow to whatever sits after `?`.
    let before_hash = trimmed.split('#').next().unwrap_or(trimmed);
    let query = match before_hash.find('?') {
        Some(idx) => &before_hash[idx + 1..],
        None => before_hash,
    };
    let mut code = None;
    let mut state = None;
    let mut error_param: Option<String> = None;
    for pair in query.split('&') {
        let Some((key, value)) = pair.split_once('=') else {
            continue;
        };
        let decoded = percent_encoding::percent_decode_str(value)
            .decode_utf8_lossy()
            .into_owned();
        match key {
            "code" => code = Some(decoded),
            "state" => state = Some(decoded),
            "error" => error_param = Some(decoded),
            _ => {}
        }
    }
    if let Some(error) = error_param {
        return Err(format!(
            "authorization server returned error: {}",
            crate::mcp::sanitize::sanitize_text(&error)
        ));
    }
    let code = code.ok_or_else(|| "missing 'code' parameter in pasted URL".to_string())?;
    let state = state.ok_or_else(|| "missing 'state' parameter in pasted URL".to_string())?;
    Ok((code, state))
}

#[derive(Debug)]
enum CallbackParseError {
    /// Request wasn't to /callback at all (e.g. /favicon.ico, /).
    NotCallbackPath,
    /// Request targets /callback but failed to parse (missing code/state, etc).
    Malformed(String),
}

fn parse_callback_query(
    request: &str,
) -> std::result::Result<(String, String), CallbackParseError> {
    let first_line = request
        .lines()
        .next()
        .ok_or_else(|| CallbackParseError::Malformed("empty HTTP request".into()))?;

    let path = first_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| CallbackParseError::Malformed("malformed HTTP request line".into()))?;

    // Compare path component only, case-insensitive, anchored to /callback.
    // `/` or `/favicon.ico` fall through to `NotCallbackPath`.
    let (path_component, query_string) = match path.split_once('?') {
        Some((p, q)) => (p, q),
        None => (path, ""),
    };
    if !path_component.eq_ignore_ascii_case("/callback") {
        return Err(CallbackParseError::NotCallbackPath);
    }
    if query_string.is_empty() {
        return Err(CallbackParseError::Malformed(
            "no query parameters in callback URL".into(),
        ));
    }

    let mut code = None;
    let mut state = None;
    let mut error_param: Option<String> = None;

    for pair in query_string.split('&') {
        let Some((key, value)) = pair.split_once('=') else {
            continue;
        };
        let decoded = percent_encoding::percent_decode_str(value)
            .decode_utf8_lossy()
            .into_owned();
        match key {
            "code" => code = Some(decoded),
            "state" => state = Some(decoded),
            "error" => error_param = Some(decoded),
            _ => {}
        }
    }

    if let Some(error) = error_param {
        // Strip Cc/Cf so a hostile authorization server can't inject ANSI
        // escapes or RTL overrides through the error message.
        return Err(CallbackParseError::Malformed(format!(
            "authorization server returned error: {}",
            crate::mcp::sanitize::sanitize_text(&error)
        )));
    }

    let code = code.ok_or_else(|| {
        CallbackParseError::Malformed("missing 'code' parameter in callback".into())
    })?;
    let state = state.ok_or_else(|| {
        CallbackParseError::Malformed("missing 'state' parameter in callback".into())
    })?;

    Ok((code, state))
}

struct SqliteCredentialStore {
    token_store: TokenStore,
    server_name: String,
}

#[async_trait]
impl CredentialStore for SqliteCredentialStore {
    async fn load(&self) -> std::result::Result<Option<StoredCredentials>, AuthError> {
        match self
            .token_store
            .load_mcp_credentials(&self.server_name)
            .await
        {
            Ok(Some(json)) => {
                let credentials: StoredCredentials =
                    serde_json::from_str(&json).map_err(|error| {
                        AuthError::InternalError(format!(
                            "failed to deserialize stored credentials: {}",
                            error
                        ))
                    })?;
                Ok(Some(credentials))
            }
            Ok(None) => Ok(None),
            Err(error) => Err(AuthError::InternalError(format!(
                "failed to load credentials from database: {}",
                error
            ))),
        }
    }

    async fn save(&self, credentials: StoredCredentials) -> std::result::Result<(), AuthError> {
        let json = serde_json::to_string(&credentials).map_err(|error| {
            AuthError::InternalError(format!("failed to serialize credentials: {}", error))
        })?;

        self.token_store
            .save_mcp_credentials(&self.server_name, &json)
            .await
            .map_err(|error| {
                AuthError::InternalError(format!(
                    "failed to save credentials to database: {}",
                    error
                ))
            })
    }

    async fn clear(&self) -> std::result::Result<(), AuthError> {
        self.token_store
            .clear_mcp_credentials(&self.server_name)
            .await
            .map_err(|error| {
                AuthError::InternalError(format!(
                    "failed to clear credentials from database: {}",
                    error
                ))
            })
    }
}

/// Decide whether a tool advertised by a server should be registered.
/// Applies `allowed_tools` (restrict-in, when set and non-empty) then
/// `disabled_tools` (always-remove). Both fields can coexist — the
/// allow-list acts as a restriction, and the block-list subtracts from
/// whatever remains. A tool passes iff it survives both checks.
fn tool_is_allowed(server_config: &McpServerConfig, tool_raw_name: &str) -> bool {
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

/// Emit a `warn!` once per entry in `allowed_tools` / `disabled_tools`
/// / `tool_permissions` that doesn't match anything the server
/// currently advertises. Users get a visible heads-up without failing
/// the connect — tool lists can change between server releases, and
/// forcing a hard error on every rename would be hostile.
fn warn_on_stale_tool_config(
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
fn resolve_tool_permission(
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

/// Shared context threaded into every [`AgshClientHandler`] so notification
/// callbacks and server-to-client requests (sampling, list_roots, elicitation)
/// can reach the rest of the agent. All slots are optional because the
/// handler is constructed before the agent/provider exist — they are filled
/// in post-construction by `main.rs` using the `set_*` helpers.
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
}

/// Permission for each server to issue sampling requests. Mirrors the
/// `sampling` / `sampling_limit` fields on `McpServerConfig`.
#[derive(Clone)]
pub struct SamplingPolicy {
    allowed: bool,
    limit: u32,
    count: Arc<AtomicU32>,
}

impl SamplingPolicy {
    fn from_config(config: &McpServerConfig) -> Self {
        Self {
            allowed: config.sampling,
            limit: config.sampling_limit.unwrap_or(10),
            count: Arc::new(AtomicU32::new(0)),
        }
    }
}

/// Client-side MCP handler. Dispatches server-initiated requests
/// (`sampling/createMessage`, `roots/list`, `elicitation/create`) and
/// notifications (`tools/list_changed`, etc.) to the rest of the agent via
/// the shared [`McpClientContext`].
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
        let manager = self.context.manager.get().and_then(|weak| weak.upgrade());
        let registry = self.context.registry.get().cloned();

        async move {
            tracing::info!("MCP server '{}' sent tools/list_changed", server_name);
            let (Some(manager), Some(registry)) = (manager, registry) else {
                tracing::debug!(
                    "tool list refresh skipped — context not yet wired for '{}'",
                    server_name
                );
                return;
            };

            // Tool-permission resolution reads the server config and
            // `mcp_default_permission` from the manager itself — no
            // explicit permission needs to be threaded here.
            match manager.discover_tools_for_server(&server_name).await {
                Ok(adapters) => {
                    let new_tools: Vec<Arc<dyn Tool>> = adapters
                        .into_iter()
                        .map(|a| Arc::new(a) as Arc<dyn Tool>)
                        .collect();
                    let new_names: Vec<String> =
                        new_tools.iter().map(|t| t.definition().name).collect();
                    registry.replace_server_tools(&server_name, new_tools);
                    // Mark freshly-registered tools as deferred so they match
                    // the behaviour of the initial startup registration.
                    for name in new_names {
                        registry.mark_deferred(&name);
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

    // Keep the explicit `impl Future` return type: other handlers in this
    // trait impl have non-trivial captures (`Arc<str>` clones, server name
    // in logging, etc.) and use the same signature shape. Staying uniform
    // makes the module easier to read than mixing `async fn` and the
    // manual-future form.
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
            let cwd = std::env::current_dir().map_err(|error| {
                McpError::internal_error(format!("current dir unavailable: {}", error), None)
            })?;
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

            // 60-second user-response timeout so a distracted user can't stall
            // an MCP tool call forever. Matches the elicitation deadline used
            // for the ToolApprovalRequest channel in shell.rs.
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

            // Elicitations are standard MCP *requests*, so a `Decline`
            // response IS how the server learns the user didn't answer —
            // no separate `notifications/cancelled` is appropriate here
            // (cancellation notifications are for long-running requests
            // we started, not for server-initiated elicitations).
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
        let provider = self.context.provider.get().cloned();

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
                // The slot was reserved for a call that never reached the
                // provider; free it so a well-formed retry isn't rejected.
                policy.count.fetch_sub(1, Ordering::SeqCst);
                McpError::invalid_params(format!("sampling conversion failed: {}", error), None)
            })?;

            // Sampling calls out to the provider with no MCP tools exposed —
            // the server asked for pure reasoning, not tool-use. The empty
            // tool list forces the provider into a plain text completion.
            // Bounded by `MCP_SAMPLING_PROVIDER_TIMEOUT` so a hung provider
            // can't pin the MCP request open indefinitely.
            let completion = tokio::time::timeout(
                MCP_SAMPLING_PROVIDER_TIMEOUT,
                provider.complete(&system_prompt, &converted, &[]),
            )
            .await;

            let (assistant_message, _stop_reason, _usage) = match completion {
                Ok(Ok(result)) => result,
                Ok(Err(error)) => {
                    // Provider returned an error before the timeout elapsed —
                    // no quota was really consumed on our side, so hand the
                    // sampling slot back.
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

/// Convert MCP `CreateMessageRequestParams` into the provider's
/// `(system_prompt, Vec<Message>)` shape, flattening text content.
/// Non-text sampling content (image, audio, tool_use, tool_result) is
/// replaced with a placeholder string — none of agsh's providers accept
/// these inside sampling calls.
fn convert_sampling_params(
    params: &CreateMessageRequestParams,
) -> std::result::Result<(String, Vec<crate::provider::Message>), String> {
    use crate::provider::{ContentBlock, Message, Role as ProviderRole};

    // Defensive sanitisation: the system prompt is server-controlled and
    // gets forwarded to the configured provider. Strip any Unicode Cc/Cf
    // codepoints so a hostile server can't smuggle terminal escapes or
    // homographs into our provider call.
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
    let peer = entry.current_peer().await;
    match peer.list_all_resources().await {
        Ok(resources) => Ok(resources),
        Err(ServiceError::TransportClosed) => {
            entry.reconnect().await?;
            let peer = entry.current_peer().await;
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
    let peer = entry.current_peer().await;
    match peer.read_resource(params.clone()).await {
        Ok(result) => Ok(result),
        Err(ServiceError::TransportClosed) => {
            entry.reconnect().await?;
            let peer = entry.current_peer().await;
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
    let peer = entry.current_peer().await;
    match peer.list_all_prompts().await {
        Ok(prompts) => Ok(prompts),
        Err(ServiceError::TransportClosed) => {
            entry.reconnect().await?;
            let peer = entry.current_peer().await;
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
    let peer = entry.current_peer().await;
    let params = rmcp::model::SubscribeRequestParams::new(uri.clone());
    peer.subscribe(params)
        .await
        .map_err(|error| AgshError::McpConnection {
            server_name: entry.server_name.clone(),
            message: format!("subscribe({}) failed: {}", uri, error),
        })
}

pub async fn unsubscribe_resource(entry: &Arc<ServerEntry>, uri: String) -> Result<()> {
    let peer = entry.current_peer().await;
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

    let peer = entry.current_peer().await;
    match peer.get_prompt(params.clone()).await {
        Ok(result) => Ok(result),
        Err(ServiceError::TransportClosed) => {
            entry.reconnect().await?;
            let peer = entry.current_peer().await;
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

pub struct McpToolAdapter {
    namespaced_name: String,
    remote_tool_name: String,
    description: String,
    parameters: serde_json::Value,
    permission: Permission,
    entry: Arc<ServerEntry>,
    /// `tool.annotations` and `tool.meta` captured from the remote
    /// server. Surfaced to the provider as hints (read-only / destructive)
    /// and round-tripped back in `_meta` so the MCP server can correlate
    /// client-side context.
    annotations: Option<serde_json::Value>,
    meta: Option<serde_json::Value>,
    title: Option<String>,
}

impl McpToolAdapter {
    /// Resolves a per-call tool-call timeout. Respects `AGSH_MCP_TOOL_TIMEOUT`
    /// (milliseconds) when set, otherwise falls back to 600 seconds — long
    /// enough for a database index rebuild but short enough that a hung
    /// server isn't invisible.
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
        // Per-call progress token: allows the server to emit
        // `notifications/progress` updates that route back to our shell UI.
        let (progress_token, _progress_guard) = crate::mcp::progress::register(
            self.entry.server_name.clone(),
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

        let peer = self.entry.current_peer().await;
        let request = ClientRequest::CallToolRequest(CallToolRequest::new(params));
        let handle = peer
            .send_cancellable_request(request, PeerRequestOptions::no_options())
            .await?;
        let request_id = handle.id.clone();

        let timeout = Self::tool_call_timeout();
        // Cap how long we wait on the best-effort cancellation notification
        // so a hung transport can't block Ctrl-C handling or shutdown.
        const CANCEL_NOTIFY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
        let notify_cancel = |reason: &'static str| {
            let peer = peer.clone();
            let request_id = request_id.clone();
            let server_name = self.entry.server_name.clone();
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
                    server_name: self.entry.server_name.clone(),
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
                            server_name: self.entry.server_name.clone(),
                            tool_name: self.remote_tool_name.clone(),
                            message: error.to_string(),
                        });
                    }
                }
            }
            Err(error) => {
                // If the server rejected us with a 401/Unauthorized, persist
                // the `needs-auth` verdict so the next startup skips the
                // unauthenticated probe and goes straight to OAuth. The user
                // must re-authenticate via `agsh mcp login <name>`.
                let text = error.to_string().to_ascii_lowercase();
                if (text.contains("401") || text.contains("unauthorized"))
                    && let Some(store) = &self.entry.token_store
                {
                    if let Err(cache_err) =
                        store.save_auth_probe(&self.entry.server_name, true).await
                    {
                        tracing::debug!(
                            "failed to save auth probe cache for '{}': {}",
                            self.entry.server_name,
                            cache_err
                        );
                    } else {
                        tracing::warn!(
                            "MCP server '{}' returned 401 — marked as needing auth. Run 'agsh mcp login {}' to re-authenticate.",
                            self.entry.server_name,
                            self.entry.server_name
                        );
                    }
                }
                return Err(AgshError::McpToolExecution {
                    server_name: self.entry.server_name.clone(),
                    tool_name: self.remote_tool_name.clone(),
                    message: error.to_string(),
                });
            }
        };

        let is_error = result.is_error.unwrap_or(false);
        let mut content = convert_tool_result_content(&result.content);

        // If the server included structured_content, append it as a fenced
        // JSON block so providers can reason over it without needing a
        // dedicated ToolResultContent variant. Matches Claude Code's
        // pragmatic passthrough.
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
                self.entry.server_name, self.remote_tool_name
            )),
        })
    }
}

/// Map MCP `CallToolResult.content` items to agsh's provider-layer
/// `ToolResultContent` blocks. Text stays text; images pass through as
/// multimodal blocks so providers like Claude and GPT-4o can see them;
/// audio, embedded resources, and resource links collapse to informative
/// text placeholders (no provider accepts them as tool-result blocks yet).
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
            tool_permissions: None,
            sampling: false,
            sampling_limit: None,
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
        // accepts a server_config with all three fields populated.
        let mut server = bare_server_config("s");
        server.allowed_tools = Some(vec!["a".into(), "unknown".into()]);
        server.disabled_tools = Some(vec!["b".into(), "gone".into()]);
        let mut perms = std::collections::HashMap::new();
        perms.insert("a".to_string(), "read".to_string());
        perms.insert("missing".to_string(), "write".to_string());
        server.tool_permissions = Some(perms);

        let advertised: std::collections::HashSet<&str> =
            ["a", "b", "search"].into_iter().collect();
        // Just confirm the call doesn't panic.
        warn_on_stale_tool_config("s", &server, &advertised);
    }

    #[test]
    fn test_parse_callback_query_valid() {
        let request = "GET /callback?code=abc123&state=xyz789 HTTP/1.1\r\nHost: localhost\r\n";
        let (code, state) = parse_callback_query(request).expect("should parse");
        assert_eq!(code, "abc123");
        assert_eq!(state, "xyz789");
    }

    #[test]
    fn test_parse_callback_query_reversed_order() {
        let request = "GET /callback?state=xyz789&code=abc123 HTTP/1.1\r\n";
        let (code, state) = parse_callback_query(request).expect("should parse");
        assert_eq!(code, "abc123");
        assert_eq!(state, "xyz789");
    }

    #[test]
    fn test_parse_callback_query_missing_code() {
        let request = "GET /callback?state=xyz789 HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        match err {
            CallbackParseError::Malformed(m) => assert!(m.contains("code")),
            other => panic!("expected Malformed, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_callback_query_missing_state() {
        let request = "GET /callback?code=abc123 HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        match err {
            CallbackParseError::Malformed(m) => assert!(m.contains("state")),
            other => panic!("expected Malformed, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_callback_query_no_query_string() {
        let request = "GET /callback HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        match err {
            CallbackParseError::Malformed(_) => {}
            other => panic!("expected Malformed, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_callback_query_ignores_favicon() {
        let request = "GET /favicon.ico HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        assert!(matches!(err, CallbackParseError::NotCallbackPath));
    }

    #[test]
    fn test_parse_callback_query_ignores_root() {
        let request = "GET / HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        assert!(matches!(err, CallbackParseError::NotCallbackPath));
    }

    #[test]
    fn test_parse_callback_query_url_decodes_state() {
        let request = "GET /callback?code=abc&state=xyz%3D%3D HTTP/1.1\r\n";
        let (code, state) = parse_callback_query(request).expect("should parse");
        assert_eq!(code, "abc");
        assert_eq!(state, "xyz==");
    }

    #[test]
    fn test_parse_callback_query_surfaces_oauth_error() {
        let request = "GET /callback?error=access_denied HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        match err {
            CallbackParseError::Malformed(m) => {
                assert!(m.contains("access_denied"));
            }
            other => panic!("expected Malformed with error, got {:?}", other),
        }
    }

    #[test]
    fn extract_revocation_endpoint_returns_string() {
        let body = br#"{"revocation_endpoint":"https://auth.example.com/revoke","other":"x"}"#;
        assert_eq!(
            extract_revocation_endpoint(body, 1024),
            Some("https://auth.example.com/revoke".to_string())
        );
    }

    #[test]
    fn extract_revocation_endpoint_rejects_oversize_body() {
        let body = br#"{"revocation_endpoint":"https://auth.example.com/revoke"}"#;
        assert!(extract_revocation_endpoint(body, 8).is_none());
    }

    #[test]
    fn extract_revocation_endpoint_returns_none_without_field() {
        let body = br#"{"issuer":"https://auth.example.com"}"#;
        assert!(extract_revocation_endpoint(body, 1024).is_none());
    }

    #[test]
    fn extract_revocation_endpoint_rejects_malformed_json() {
        assert!(extract_revocation_endpoint(b"not json", 1024).is_none());
    }

    #[test]
    fn validate_revocation_endpoint_accepts_same_origin() {
        validate_revocation_endpoint_origin(
            "https://auth.example.com",
            "https://auth.example.com/revoke",
        )
        .expect("same host should pass");
    }

    #[test]
    fn validate_revocation_endpoint_accepts_same_origin_with_path() {
        validate_revocation_endpoint_origin(
            "https://auth.example.com/realms/r",
            "https://auth.example.com/realms/r/protocol/openid-connect/revoke",
        )
        .expect("subpath on same host should pass");
    }

    #[test]
    fn validate_revocation_endpoint_rejects_different_host() {
        let err = validate_revocation_endpoint_origin(
            "https://auth.example.com",
            "https://attacker.com/steal",
        )
        .expect_err("cross-host should be rejected");
        assert!(err.contains("different origin"));
    }

    #[test]
    fn validate_revocation_endpoint_rejects_http_to_https_downgrade() {
        let err = validate_revocation_endpoint_origin(
            "https://auth.example.com",
            "http://auth.example.com/revoke",
        )
        .expect_err("scheme change should be rejected");
        assert!(err.contains("different origin"));
    }

    #[test]
    fn validate_revocation_endpoint_rejects_different_port() {
        let err = validate_revocation_endpoint_origin(
            "https://auth.example.com",
            "https://auth.example.com:8443/revoke",
        )
        .expect_err("port change should be rejected");
        assert!(err.contains("different origin"));
    }

    #[cfg(unix)]
    #[test]
    fn require_private_key_permissions_rejects_world_readable() {
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let key_path = dir.path().join("key.pem");
        let mut f = std::fs::File::create(&key_path).expect("create key");
        f.write_all(b"---BEGIN---").expect("write");
        drop(f);
        // 0644 — readable by other users.
        std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o644))
            .expect("chmod 0644");
        let err = require_private_key_permissions("srv", key_path.to_str().unwrap())
            .expect_err("loose perms must be rejected");
        let message = format!("{}", err);
        assert!(
            message.contains("0600") || message.contains("permissions"),
            "unexpected error: {}",
            message
        );
    }

    #[cfg(unix)]
    #[test]
    fn require_private_key_permissions_accepts_0600() {
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let key_path = dir.path().join("key.pem");
        let mut f = std::fs::File::create(&key_path).expect("create key");
        f.write_all(b"---BEGIN---").expect("write");
        drop(f);
        std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o600))
            .expect("chmod 0600");
        require_private_key_permissions("srv", key_path.to_str().unwrap()).expect("0600 must pass");
    }

    #[cfg(unix)]
    #[test]
    fn require_private_key_permissions_reports_missing_file() {
        let err = require_private_key_permissions("srv", "/nonexistent/key.pem")
            .expect_err("missing file must error");
        let message = format!("{}", err);
        assert!(message.contains("stat signing key") || message.contains("key"));
    }

    #[test]
    fn validate_revocation_endpoint_rejects_malformed_urls() {
        assert!(
            validate_revocation_endpoint_origin("not-a-url", "https://auth.example.com/revoke")
                .is_err()
        );
        assert!(
            validate_revocation_endpoint_origin("https://auth.example.com", "also-not-a-url")
                .is_err()
        );
    }

    #[test]
    fn test_parse_callback_query_sanitises_oauth_error() {
        // A malicious authorization server includes an ANSI escape and an
        // RTL override in its `error` parameter. The resulting error message
        // must not carry those codepoints to the terminal.
        let request = "GET /callback?error=bad%1B%5B2Jstuff%E2%80%AErtl HTTP/1.1\r\n";
        let err = parse_callback_query(request).expect_err("should fail");
        match err {
            CallbackParseError::Malformed(m) => {
                assert!(!m.contains('\u{001B}'), "ANSI escape leaked: {:?}", m);
                assert!(!m.contains('\u{202E}'), "RTL override leaked: {:?}", m);
                assert!(m.contains("bad"));
                assert!(m.contains("stuff"));
                assert!(m.contains("rtl"));
            }
            other => panic!("expected Malformed with error, got {:?}", other),
        }
    }

    #[test]
    fn classify_probe_open_on_2xx() {
        assert_eq!(classify_probe_response(200, None), McpAuthProbe::Open);
        assert_eq!(classify_probe_response(204, None), McpAuthProbe::Open);
        assert_eq!(
            classify_probe_response(200, Some("Bearer realm=\"x\"")),
            McpAuthProbe::Open,
            "2xx always wins over any WWW-Authenticate header"
        );
    }

    #[test]
    fn classify_probe_auth_required_with_resource_metadata() {
        // What Notion-style MCP servers actually emit.
        let header = r#"Bearer realm="mcp", resource_metadata="https://mcp.notion.com/.well-known/oauth-protected-resource""#;
        assert_eq!(
            classify_probe_response(401, Some(header)),
            McpAuthProbe::AuthRequired {
                resource_metadata: Some(
                    "https://mcp.notion.com/.well-known/oauth-protected-resource".to_string()
                ),
            }
        );
    }

    #[test]
    fn classify_probe_auth_required_without_resource_metadata() {
        assert_eq!(
            classify_probe_response(401, Some("Bearer realm=\"mcp\"")),
            McpAuthProbe::AuthRequired {
                resource_metadata: None,
            }
        );
    }

    #[test]
    fn classify_probe_auth_required_bare_bearer() {
        // Some servers emit just `Bearer` with no parameters.
        assert_eq!(
            classify_probe_response(401, Some("Bearer")),
            McpAuthProbe::AuthRequired {
                resource_metadata: None,
            }
        );
    }

    #[test]
    fn classify_probe_is_case_insensitive_on_scheme() {
        assert_eq!(
            classify_probe_response(401, Some("bearer realm=\"x\"")),
            McpAuthProbe::AuthRequired {
                resource_metadata: None,
            }
        );
    }

    #[test]
    fn classify_probe_401_without_bearer_is_unexpected() {
        // A 401 with e.g. Basic / Digest auth is not MCP-spec compliant —
        // surface it as Unexpected rather than pretending it's OAuth.
        assert_eq!(
            classify_probe_response(401, Some("Basic realm=\"x\"")),
            McpAuthProbe::Unexpected { status: 401 }
        );
        assert_eq!(
            classify_probe_response(401, None),
            McpAuthProbe::Unexpected { status: 401 }
        );
    }

    #[test]
    fn classify_probe_403_with_bearer_is_auth_required() {
        // Some implementations return 403 for missing auth.
        assert_eq!(
            classify_probe_response(403, Some("Bearer realm=\"mcp\"")),
            McpAuthProbe::AuthRequired {
                resource_metadata: None,
            }
        );
    }

    #[test]
    fn classify_probe_other_statuses_are_unexpected() {
        assert_eq!(
            classify_probe_response(405, None),
            McpAuthProbe::Unexpected { status: 405 }
        );
        assert_eq!(
            classify_probe_response(500, None),
            McpAuthProbe::Unexpected { status: 500 }
        );
    }

    #[test]
    fn extract_bearer_param_handles_quoting_and_spacing() {
        let header = r#"Bearer realm="x",resource_metadata="https://y/z""#;
        assert_eq!(
            extract_bearer_param(header, "resource_metadata"),
            Some("https://y/z".to_string())
        );
        let unquoted = r#"Bearer realm=x, resource_metadata=https://y/z"#;
        assert_eq!(
            extract_bearer_param(unquoted, "resource_metadata"),
            Some("https://y/z".to_string())
        );
    }

    #[test]
    fn extract_bearer_param_returns_none_when_missing() {
        assert_eq!(
            extract_bearer_param("Bearer realm=\"x\"", "resource_metadata"),
            None
        );
    }

    #[test]
    fn parse_pasted_callback_accepts_full_url() {
        // Exact shape returned by Notion after the user authorises.
        let input = "http://127.0.0.1:46437/callback?code=1d5d872b-594c-8153-a5e0-0002d8f4be0f%3AGEE3YdaPJhHZpMMa%3AjZ2YV0BC0TYheYBtoSB16LmRDTgIZ6zM&state=82Nw67m4su5AeMSfCFcXAw";
        let (code, state) = parse_pasted_callback(input).expect("should parse");
        // Percent-encoded colons in the code must be decoded.
        assert_eq!(
            code,
            "1d5d872b-594c-8153-a5e0-0002d8f4be0f:GEE3YdaPJhHZpMMa:jZ2YV0BC0TYheYBtoSB16LmRDTgIZ6zM"
        );
        assert_eq!(state, "82Nw67m4su5AeMSfCFcXAw");
    }

    #[test]
    fn parse_pasted_callback_accepts_query_only() {
        let (code, state) = parse_pasted_callback("code=abc&state=xyz").expect("should parse");
        assert_eq!(code, "abc");
        assert_eq!(state, "xyz");
    }

    #[test]
    fn parse_pasted_callback_trims_whitespace_and_fragment() {
        let (code, state) =
            parse_pasted_callback("   http://127.0.0.1:1/callback?code=a&state=b#fragment  \n")
                .expect("should parse");
        assert_eq!(code, "a");
        assert_eq!(state, "b");
    }

    #[test]
    fn parse_pasted_callback_rejects_empty_input() {
        let err = parse_pasted_callback("   \n").expect_err("empty input should fail");
        assert!(err.contains("no callback URL"));
    }

    #[test]
    fn parse_pasted_callback_surfaces_server_error_sanitised() {
        let input = "http://127.0.0.1:1/callback?error=bad%1B%5B2Jstuff%E2%80%AErtl&state=z";
        let err = parse_pasted_callback(input).expect_err("should surface error");
        assert!(!err.contains('\u{001B}'), "ANSI leaked: {}", err);
        assert!(!err.contains('\u{202E}'), "RTL leaked: {}", err);
        assert!(err.contains("bad"));
    }

    #[test]
    fn parse_pasted_callback_missing_code() {
        let err = parse_pasted_callback("state=xyz").expect_err("should fail");
        assert!(err.contains("missing 'code'"));
    }

    #[test]
    fn parse_pasted_callback_missing_state() {
        let err = parse_pasted_callback("code=abc").expect_err("should fail");
        assert!(err.contains("missing 'state'"));
    }

    #[test]
    fn test_parse_jwt_signing_algorithm_defaults() {
        let alg = parse_jwt_signing_algorithm("test", None).expect("should parse");
        assert!(matches!(alg, rmcp::transport::JwtSigningAlgorithm::RS256));
    }

    #[test]
    fn test_parse_jwt_signing_algorithm_all_values() {
        for name in &["RS256", "RS384", "RS512", "ES256", "ES384"] {
            assert!(parse_jwt_signing_algorithm("test", Some(name)).is_ok());
        }
    }

    #[test]
    fn test_parse_jwt_signing_algorithm_invalid() {
        assert!(parse_jwt_signing_algorithm("test", Some("HS256")).is_err());
    }

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
}
