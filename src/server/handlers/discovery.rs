//! Health-probe endpoints: `/v1/health/live` (process-up) and `/v1/health/ready`
//! (subsystems-healthy). The other discovery endpoints — `/v1/info`, `/v1/skills`, `/v1/mcp` —
//! live in [`super::info`].

use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use serde::Serialize;
use utoipa::ToSchema;

use crate::server::state::ServerState;

#[derive(Serialize, ToSchema)]
pub struct LiveResponse {
    pub status: String,
}

/// Liveness probe. Always returns 200 if the process is up and the listener is accepting —
/// the response handler running at all is sufficient proof. Used by Kubernetes / systemd to
/// distinguish "process crashed" from "process degraded".
#[utoipa::path(
    get,
    path = "/v1/health/live",
    tag = "health",
    responses((status = 200, description = "Process is up and accepting connections", body = LiveResponse))
)]
pub async fn live() -> Json<LiveResponse> {
    Json(LiveResponse {
        status: "ok".to_string(),
    })
}

#[derive(Serialize, ToSchema)]
pub struct ReadyResponse {
    pub status: String,
    /// Per-subsystem readiness flags. `false` here means the subsystem is in a state that
    /// would fail a real request; a `503` is returned in that case.
    pub session_db: bool,
    pub provider_configured: bool,
    /// `true` when all configured MCP servers are connected (or none are configured).
    /// Per-server detail (names, connection states) is available via `GET /v1/mcp`
    /// (requires auth) — deliberately omitted here because `/v1/health/ready` is
    /// unauthenticated and server names leak infrastructure topology.
    pub mcp_servers_healthy: bool,
}

/// Readiness probe. Returns 200 iff the server is in a state where new turn requests can
/// reasonably be expected to succeed (session DB queryable, provider configured, no MCP
/// servers stuck in `Failed`). Returns 503 with a body that names which subsystem is the
/// blocker.
#[utoipa::path(
    get,
    path = "/v1/health/ready",
    tag = "health",
    responses(
        (status = 200, description = "All dependencies healthy", body = ReadyResponse),
        (status = 503, description = "One or more subsystems unavailable", body = ReadyResponse),
    )
)]
pub async fn ready(State(state): State<ServerState>) -> impl IntoResponse {
    // Touch the session DB with a cheap read. `session_exists(nil_uuid)` runs one statement
    // and returns Ok(false); any error means the connection is broken / DB is gone.
    let session_db = state
        .shared
        .session_manager
        .session_exists(uuid::Uuid::nil())
        .await
        .is_ok();
    let provider_configured = state.shared.config.provider_name.is_some();

    let mut mcp_healthy = true;
    if let Some(manager) = state.shared.mcp_manager.as_ref() {
        for name in manager.server_names() {
            let is_failed = match manager.server_entry(&name) {
                Some(entry) => entry.state().await.label() == "failed",
                None => true,
            };
            if is_failed {
                mcp_healthy = false;
                break;
            }
        }
    }

    let healthy = session_db && provider_configured && mcp_healthy;
    let body = ReadyResponse {
        status: if healthy {
            "ok".to_string()
        } else {
            "degraded".to_string()
        },
        session_db,
        provider_configured,
        mcp_servers_healthy: mcp_healthy,
    };
    let status = if healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(body))
}
