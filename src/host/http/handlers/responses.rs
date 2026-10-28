//! `POST /v1/sessions/{id}/responses/{request_id}`: client responses to the mid-turn
//! `permission_required` and `elicitation_required` SSE events. One body for both, since the id
//! says which prompt it names: an approval takes `allow`, `deny`, `allow_always` or
//! `deny_always`, an elicitation `accept` with its `content`, `decline` or `cancel`, and an answer
//! of the other kind is refused naming the kind.

use axum::{
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
};
use serde::Deserialize;
use utoipa::ToSchema;
use uuid::Uuid;

use crate::{
    frontend::ElicitationResponse,
    host::http::{
        errors::{ErrorKind, ProblemDetail},
        http_frontend::{PendingKind, PermissionResolution},
        reattach::require_session_exists,
        scope,
        state::ServerState,
    },
};

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ResponseBody {
    pub(crate) outcome: ResponseOutcome,
    /// What the form was filled with, on an `accept`: an object of the fields the schema named.
    #[serde(default)]
    pub(crate) content: Option<serde_json::Value>,
}

/// The answers a parked prompt takes: the four an approval takes, and the three an elicitation
/// takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ResponseOutcome {
    Allow,
    Deny,
    AllowAlways,
    DenyAlways,
    Accept,
    Decline,
    Cancel,
}

#[utoipa::path(
    post,
    path = "/v1/sessions/{id}/responses/{request_id}",
    tag = "turn",
    params(
        ("id" = Uuid, Path, description = "Session UUID"),
        ("request_id" = String, Path, description = "Pending request id emitted by the matching `permission_required` or `elicitation_required` SSE event."),
    ),
    request_body = ResponseBody,
    responses(
        (status = 204, description = "Resolved; the parked turn continues"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session or request not found / already resolved", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid body, an answer of the other prompt's kind, `content` on anything but `accept`, or the id names a sub-agent's conversation (`/errors/session-not-drivable`), which no payload makes acceptable", body = ProblemDetail),
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

    let Some(kind) = entry.frontend.pending_kind(&request_id) else {
        return Err(request_not_found(session_id, &request_id));
    };
    let mismatch = |sentence: &str| {
        ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("request '{request_id}' {sentence}"),
        )
        .with("session_id", session_id.to_string())
        .with("request_id", request_id.clone())
    };
    if body.content.is_some() && body.outcome != ResponseOutcome::Accept {
        return Err(mismatch(
            "takes `content` only on `accept`; drop it, or accept",
        ));
    }
    let resolved = match (kind, body.outcome) {
        (PendingKind::Permission, ResponseOutcome::Allow) => entry
            .frontend
            .resolve_permission(&request_id, PermissionResolution::Allow),
        (PendingKind::Permission, ResponseOutcome::AllowAlways) => entry
            .frontend
            .resolve_permission(&request_id, PermissionResolution::AllowAlways),
        (PendingKind::Permission, ResponseOutcome::Deny) => entry
            .frontend
            .resolve_permission(&request_id, PermissionResolution::Deny),
        (PendingKind::Permission, ResponseOutcome::DenyAlways) => entry
            .frontend
            .resolve_permission(&request_id, PermissionResolution::DenyAlways),
        (PendingKind::Permission, _) => {
            return Err(mismatch(
                "is an approval; answer with allow, deny, allow_always or deny_always",
            ));
        }
        // A form's answer is its fields: an `accept` with none would reach the server as an
        // empty acceptance of a schema that asked for something.
        (PendingKind::Elicitation { form: true }, ResponseOutcome::Accept)
            if body.content.is_none() =>
        {
            return Err(mismatch(
                "is a form; `accept` carries `content`, the fields its schema names",
            ));
        }
        (PendingKind::Elicitation { .. }, ResponseOutcome::Accept) => entry
            .frontend
            .resolve_elicitation(&request_id, ElicitationResponse::Accept {
                content: body.content,
            }),
        (PendingKind::Elicitation { .. }, ResponseOutcome::Decline) => entry
            .frontend
            .resolve_elicitation(&request_id, ElicitationResponse::Decline),
        (PendingKind::Elicitation { .. }, ResponseOutcome::Cancel) => entry
            .frontend
            .resolve_elicitation(&request_id, ElicitationResponse::Cancel),
        (PendingKind::Elicitation { .. }, _) => {
            return Err(mismatch(
                "is an elicitation; answer with accept, decline or cancel",
            ));
        }
    };
    if resolved {
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
    let remedy = match terms.parent {
        Some(parent) => {
            format!("; answer its prompts at `POST /v1/sessions/{parent}/responses/{{request_id}}`")
        }
        None => ", so it has no prompt of its own to answer".to_string(),
    };
    let detail = format!("{}{remedy}", terms.describe(session_id));
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
