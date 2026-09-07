//! Server-level discovery endpoints: `/v1/info`, `/v1/skills`, `/v1/mcp`.

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use serde::Serialize;
use utoipa::ToSchema;

use crate::{
    host::http::{
        errors::{ErrorKind, ProblemDetail},
        scope::{self},
        state::ServerState,
    },
    view::{McpToolView, McpToolsResponse, SkillView},
};

/// What this server is, for a client deciding how to talk to it.
///
/// Deliberately carries no backend, profile or model. A server-wide field for either invites a
/// client to post it back to `POST /v1/sessions`, whose `profile` names a profile (`work`) and
/// never a backend (`anthropic-messages`): one field, two meanings, one API, and a 422 for the
/// client that mixed them. `GET /v1/profiles` answers both questions and names them apart, `name`
/// for the profile and `backend` for the backend, and marks the default with `active`.
#[derive(Serialize, ToSchema)]
pub(crate) struct InfoResponse {
    pub(crate) version: String,
    pub(crate) default_permission: String,
    pub(crate) enabled_permissions: Vec<String>,
    /// Whether the process default profile accepts image attachments. The HTTP analog of ACP's
    /// `promptCapabilities.image`, so a client can tell whether attaching one is worth the base64
    /// payload instead of discovering it from a 422.
    ///
    /// An answer for the process, like everything else on this endpoint, and therefore only for a
    /// session created without naming a `profile`. `POST /turn` asks the session itself
    /// (`ResidentSession::accepts_images`), so a session on another profile can differ.
    pub(crate) vision: bool,
}

/// `GET /v1/info`: server identity and the permission surface. Authenticated; admits any token
/// holding at least one read scope (see [`crate::host::http::scope::ANY_READ_SCOPES`]). Tokens with
/// only write scopes get 403. The broad-read fallback is intentional: a token configured for
/// `sessions:r` to surface session listings can also see the server's own identity without
/// operators having to grant it anything else.
///
/// For which profiles exist and which one a session gets by default, see `GET /v1/profiles`.
#[utoipa::path(
    get,
    path = "/v1/info",
    tag = "discovery",
    responses(
        (status = 200, description = "Server identity and capability flags", body = InfoResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:r", "mcp:r", "skills:r", "memory:r", "schedule:r"]))
)]
pub(crate) async fn info(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::AnyRead>,
) -> Result<Json<InfoResponse>, ProblemDetail> {
    let config = &state.shared.config;
    Ok(Json(InfoResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        default_permission: config.permission.to_string(),
        enabled_permissions: config
            .enabled_permissions
            .iter()
            .map(|p| p.to_string())
            .collect(),
        vision: config.vision,
    }))
}

/// `GET /v1/skills`: installed skill palette. Mirrors what the REPL `/skill` command and
/// the ACP `available_commands_update` notification surface.
#[utoipa::path(
    get,
    path = "/v1/skills",
    tag = "discovery",
    responses(
        (status = 200, description = "Installed skill palette", body = [SkillView]),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:r", "mcp:r", "skills:r", "memory:r", "schedule:r"]))
)]
pub(crate) async fn skills(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::AnyRead>,
) -> Result<Json<Vec<SkillView>>, ProblemDetail> {
    let snapshot = state.shared.skills.current().await;
    let skills = snapshot.skills.iter().map(SkillView::from).collect();
    Ok(Json(skills))
}

/// One configured MCP server and where its connection stands. Not the server's configuration,
/// which `meka mcp list` prints and which a read token has no business seeing.
#[derive(Serialize, ToSchema)]
pub(crate) struct McpServerStateView {
    pub(crate) name: String,
    /// `pending`, `connected`, `failed` or `disabled`.
    pub(crate) state: String,
}

/// `GET /v1/mcp`: configured MCP servers and their current connection state.
#[utoipa::path(
    get,
    path = "/v1/mcp",
    tag = "discovery",
    responses(
        (status = 200, description = "Per-server connection state", body = [McpServerStateView]),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:r", "mcp:r", "skills:r", "memory:r", "schedule:r"]))
)]
pub(crate) async fn mcp(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::AnyRead>,
) -> Result<Json<Vec<McpServerStateView>>, ProblemDetail> {
    let Some(manager) = state.shared.mcp_manager.as_ref() else {
        return Ok(Json(Vec::new()));
    };
    let names = manager.server_names();
    let mut servers = Vec::with_capacity(names.len());
    for name in names {
        let server_state = match manager.server_entry(&name) {
            Some(entry) => entry.state().await.label().to_string(),
            None => "unknown".to_string(),
        };
        servers.push(McpServerStateView {
            name,
            state: server_state,
        });
    }
    Ok(Json(servers))
}

/// `GET /v1/mcp/{name}/tools`: what one MCP server advertises, with resolved permissions.
///
/// Queries the server live rather than reporting the registered set, matching `meka mcp tools`:
/// the point is to see everything it offers, including tools config is currently filtering out.
#[utoipa::path(
    get,
    path = "/v1/mcp/{name}/tools",
    tag = "discovery",
    params(("name" = String, Path, description = "MCP server name")),
    responses(
        (status = 200, description = "Advertised tools", body = McpToolsResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such server", body = ProblemDetail),
        (status = 502, description = "Server unreachable or list_tools failed", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["mcp:r"]))
)]
pub(crate) async fn mcp_tools(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::McpRead>,
    Path(name): Path<String>,
) -> Result<Json<McpToolsResponse>, ProblemDetail> {
    let manager = state
        .shared
        .mcp_manager
        .as_ref()
        .ok_or_else(|| no_such_server(&name))?;
    if manager.server_entry(&name).is_none() {
        return Err(no_such_server(&name));
    }
    let tools = manager
        .list_advertised_tools(&name)
        .await
        // A connection or `list_tools` failure is upstream, not the caller's: 502, the same
        // classification `MekaError::Provider` gets.
        .map_err(|error| {
            ProblemDetail::new(ErrorKind::Provider, StatusCode::BAD_GATEWAY, error.to_string())
                .with("server", name.clone())
        })?;
    Ok(Json(McpToolsResponse {
        server: name,
        tools: tools.iter().map(McpToolView::from).collect(),
    }))
}

fn no_such_server(name: &str) -> ProblemDetail {
    ProblemDetail::new(
        ErrorKind::NotFound,
        StatusCode::NOT_FOUND,
        format!("no MCP server named '{name}'"),
    )
    .with("server", name.to_string())
}

#[derive(Serialize, ToSchema)]
pub(crate) struct McpReconnectResponse {
    pub(crate) server: String,
    /// Where the server stands now: `connected`, `failed`, or `pending`.
    ///
    /// A 200 says meka acted on the request, not that the server came back, so read this to find
    /// out which. `pending` means no attempt was made because one was already under way. A
    /// `disabled` server is a 422 and never appears here.
    pub(crate) state: String,
}

/// `POST /v1/mcp/{name}/reconnect`: heal one server now.
///
/// An impatience button rather than the only route back: a server that failed its initial connect
/// is already being retried in the background with exponential backoff. This collapses the wait for
/// an operator who has just fixed whatever was wrong.
#[utoipa::path(
    post,
    path = "/v1/mcp/{name}/reconnect",
    tag = "discovery",
    params(("name" = String, Path, description = "MCP server name")),
    responses(
        (status = 200, description = "Read `state` for where the server now stands", body = McpReconnectResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such server", body = ProblemDetail),
        (status = 422, description = "Server is disabled in config", body = ProblemDetail),
        (status = 502, description = "Re-establishing a connected server's transport did not finish in time", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["mcp:w"]))
)]
pub(crate) async fn mcp_reconnect(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::McpWrite>,
    Path(name): Path<String>,
) -> Result<Json<McpReconnectResponse>, ProblemDetail> {
    let manager = state
        .shared
        .mcp_manager
        .as_ref()
        .ok_or_else(|| no_such_server(&name))?;
    let entry = manager
        .server_entry(&name)
        .ok_or_else(|| no_such_server(&name))?;

    // Read once, before the attempt, and classify the outcome against it rather than against the
    // wording of the error. `reconnect_server` refuses two states outright, and telling them apart
    // by sniffing the message for "disabled" makes a status code hostage to a string an upstream
    // server could also produce.
    let state_before = entry.state().await;
    if matches!(state_before, crate::mcp::ServerState::Pending) {
        // Not an error. The startup connector owns every `Pending` entry and is already connecting
        // it, so nothing is wrong and nothing needs doing; reporting the refusal as 502 would tell
        // a dashboard the server had failed when it is merely still starting, which is exactly
        // when a dashboard polling `GET /v1/mcp` reaches for this button.
        return Ok(Json(McpReconnectResponse {
            server: name,
            state: state_before.label().to_string(),
        }));
    }

    let timeout = state.shared.config.mcp_connect_timeout;
    let resolved = manager
        .reconnect_server(&name, timeout)
        .await
        .map_err(|error| {
            // A disabled server is the caller asking for something the config forbids (422); a
            // transport failure is upstream (502), the same classification `mcp_tools` gives it.
            // Collapsing both into 422 would tell a client to fix its request when the fix is to
            // start the server.
            let (kind, status) = if matches!(state_before, crate::mcp::ServerState::Disabled) {
                (ErrorKind::InvalidBody, StatusCode::UNPROCESSABLE_ENTITY)
            } else {
                (ErrorKind::Provider, StatusCode::BAD_GATEWAY)
            };
            ProblemDetail::new(kind, status, error.to_string()).with("server", name.clone())
        })?;
    tracing::info!(
        "reconnected MCP server '{name}' via HTTP: {state}",
        state = resolved.label()
    );
    Ok(Json(McpReconnectResponse {
        server: name,
        state: resolved.label().to_string(),
    }))
}
