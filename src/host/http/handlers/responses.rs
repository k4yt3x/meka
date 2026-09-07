//! `POST /v1/sessions/{id}/responses/{request_id}`: client responses to mid-turn
//! `permission_required` SSE events. The HTTP API models only permission approvals; MCP
//! elicitation auto-declines server-side without reaching the wire (service-to-service
//! callers can't render interactive prompts). With only one outcome category the body has no
//! `kind` discriminator: just `{"outcome": "..."}`.

use axum::{
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
};
use serde::Deserialize;
use utoipa::ToSchema;
use uuid::Uuid;

use crate::host::http::{
    errors::{ErrorKind, ProblemDetail},
    http_frontend::PermissionResolution,
    reattach::require_session_exists,
    scope,
    state::ServerState,
};

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ResponseBody {
    pub(crate) outcome: PermissionDecision,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PermissionDecision {
    Allow,
    Deny,
    AllowAlways,
    DenyAlways,
}

#[utoipa::path(
    post,
    path = "/v1/sessions/{id}/responses/{request_id}",
    tag = "turn",
    params(
        ("id" = Uuid, Path, description = "Session UUID"),
        ("request_id" = String, Path, description = "Pending request id emitted by the matching permission_required SSE event."),
    ),
    request_body = ResponseBody,
    responses(
        (status = 204, description = "Resolved; the parked turn continues"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session or request not found / already resolved", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid body, or the id names a sub-agent's conversation (`/errors/session-not-drivable`), which no payload makes acceptable", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:w"]))
)]
pub(crate) async fn respond(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SessionsWrite>,
    Path((session_id, request_id)): Path<(Uuid, String)>,
    raw_body: Bytes,
) -> Result<StatusCode, ProblemDetail> {
    // Parse via `serde_json` directly so `deny_unknown_fields` rejections produce a
    // `application/problem+json` 422 instead of axum's default text/plain response.
    let body: ResponseBody = serde_json::from_slice(&raw_body)
        .map_err(|error| ProblemDetail::invalid_body("response", error))?;
    // Resident sessions only, never a reconstruction. A pending request lives in the resident
    // frontend's parking lot and nowhere else, so a session that is not resident has nothing to
    // resolve, and reviving one to find that out took its cross-process lock for up to
    // `idle_timeout` and pinned it in memory: a `sessions:w` holder could walk the session list and
    // lock an operator out of every one of them with bogus request ids.
    require_session_exists(&state, session_id).await?;
    // A sub-agent is never resident here: it runs under its parent's runtime, and its prompts are
    // parked on the parent's frontend. Without this the resident lookup below answered
    // `request-not-found`, which invites the client to check the id and try again, for an id no
    // request will ever be found under.
    if let Some(refusal) = refuse_subagent_session(session_id, &state).await? {
        return Err(refusal);
    }
    let entry = state.sessions.read().await.get(&session_id).cloned();
    let Some(entry) = entry else {
        return Err(request_not_found(session_id, &request_id));
    };

    let resolution = match body.outcome {
        PermissionDecision::Allow => PermissionResolution::Allow,
        PermissionDecision::AllowAlways => PermissionResolution::AllowAlways,
        PermissionDecision::Deny => PermissionResolution::Deny,
        PermissionDecision::DenyAlways => PermissionResolution::DenyAlways,
    };

    if entry.frontend.resolve_permission(&request_id, resolution) {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(request_not_found(session_id, &request_id))
    }
}

/// The `session-not-drivable` refusal for a sub-agent's id, when the id names one.
///
/// Keyed on the spawn terms rather than the parent link, like every sibling door, so an imported
/// sub-agent whose parent did not survive the archive is refused too. The remedy names the parent
/// because that is where the prompt is parked: a sub-agent's frontend forwards approvals to the
/// session that spawned it.
async fn refuse_subagent_session(
    session_id: Uuid,
    state: &ServerState,
) -> Result<Option<ProblemDetail>, ProblemDetail> {
    let Some(terms) = state
        .shared
        .store
        .spawn_terms(session_id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to read session spawn terms", error)
        })?
    else {
        return Ok(None);
    };
    let detail = match terms.parent {
        Some(parent) => format!(
            "session '{session_id}' is a sub-agent of '{parent}'; answer its prompts at \
             `POST /v1/sessions/{parent}/responses/{{request_id}}`"
        ),
        None => format!(
            "session '{session_id}' is a sub-agent whose parent is not in this store, so it has no \
             prompt of its own to answer"
        ),
    };
    Ok(Some(
        ProblemDetail::new(
            ErrorKind::SessionNotDrivable,
            StatusCode::UNPROCESSABLE_ENTITY,
            detail,
        )
        .with("session_id", session_id.to_string()),
    ))
}

fn request_not_found(session_id: Uuid, request_id: &str) -> ProblemDetail {
    ProblemDetail::new(
        ErrorKind::RequestNotFound,
        StatusCode::NOT_FOUND,
        format!(
            "pending request '{request_id}' for session '{session_id}' is unknown or already resolved"
        ),
    )
    .with("session_id", session_id.to_string())
    .with("request_id", request_id.to_string())
}
