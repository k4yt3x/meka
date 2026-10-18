//! The session inbox over HTTP: `POST`, `GET` and `DELETE` under `/v1/sessions/{id}/inbox`.
//!
//! The asynchronous door beside `POST /turn`. A client that lives with a session (a chat bridge, a
//! UI) enqueues here and watches the session feed, rather than holding a turn open end to end:
//! the item is durable before the 202, a `steer` reaches the running turn at its next round
//! boundary, an `interrupt` cuts the answer it is streaming, a `followup` rides the next turn, and
//! `inbox.delivered` on the feed says when the model read it. Nothing here waits on a turn.

use axum::{
    Json,
    body::Bytes,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use crate::{
    host::http::{
        errors::{ErrorKind, ProblemDetail},
        reattach::ensure_session_loaded,
        scope,
        sse::SseEventType,
        state::ServerState,
    },
    store::inbox::{InboxClass, InboxItem, NewInboxItem, Withdrawal},
};

/// The body of `POST /v1/sessions/{id}/inbox`.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct InboxRequest {
    /// What the model reads, verbatim under a header naming when it arrived and, when given,
    /// `source`.
    pub(crate) message: String,
    /// `steer` reaches a running turn at its next round boundary; `interrupt` cuts the answer it
    /// is streaming and is read at the boundary while a tool runs; `followup` waits for the turn
    /// to end. Any of them rides the next turn's opening, or opens one, when nothing is running.
    #[schema(value_type = String)]
    pub(crate) class: InboxClass,
    /// Who the message is from, as the header names them. Left out or blank, the header names
    /// nobody: who sent a message is the client's to say, and meka does not guess it from the
    /// token.
    #[serde(default)]
    pub(crate) source: Option<String>,
}

/// One inbox item as the API shows it.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct InboxItemView {
    pub(crate) id: Uuid,
    pub(crate) session_id: Uuid,
    pub(crate) class: String,
    /// Absent when the client named nobody.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) source: Option<String>,
    /// `pending`, `appended`, `delivered` or `withdrawn`.
    pub(crate) state: String,
    pub(crate) created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) appended_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) delivered_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) withdrawn_at: Option<String>,
    /// When meka stops retrying it: its enqueue plus `[serve] inbox_retry_ceiling`.
    pub(crate) retry_until: String,
    /// Why it was given up on, when it was.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) failure: Option<String>,
    /// When its next attempt may run, while it waits after a failed turn or a held session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) not_before: Option<String>,
    /// Why the last turn opened on it failed, as a Problem Detail (`type`, `title`, `status`,
    /// `detail`), while it waits to be tried again or once it was given up on for it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_error: Option<serde_json::Value>,
}

impl InboxItemView {
    /// The item as the API shows it, with the deadline `ceiling` sets from its enqueue.
    pub(crate) fn new(item: InboxItem, ceiling: std::time::Duration) -> Self {
        Self {
            id: item.id,
            session_id: item.session_id,
            class: item.class.name().to_string(),
            source: item.source.clone(),
            state: item.state().name().to_string(),
            created_at: item.created_at.to_rfc3339(),
            appended_at: item.appended_at.map(|at| at.to_rfc3339()),
            delivered_at: item.delivered_at.map(|at| at.to_rfc3339()),
            withdrawn_at: item.withdrawn_at.map(|at| at.to_rfc3339()),
            retry_until: crate::host::scheduler::inbox_deadline(&item, ceiling).to_rfc3339(),
            failure: item.failure,
            not_before: item.not_before.map(|at| at.to_rfc3339()),
            // The row holds the object this host wrote; anything else is not one, and is shown
            // as nothing rather than as prose a client would have to parse.
            last_error: item
                .last_error
                .as_deref()
                .and_then(|text| serde_json::from_str::<serde_json::Value>(text).ok())
                .filter(serde_json::Value::is_object),
        }
    }
}

/// The answer to `POST /v1/sessions/{id}/inbox`: the item's identity and state, and what a
/// producer that holds the key needs to know about its fate without listing the inbox.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct InboxResponse {
    pub(crate) item_id: Uuid,
    pub(crate) session_id: Uuid,
    pub(crate) class: String,
    /// The item's state as answered: `pending` on a fresh item, unless a turn already running
    /// read it in the meantime. A replayed `Idempotency-Key` answers the earlier item's.
    pub(crate) state: String,
    /// Whether the `Idempotency-Key` had been used before, so the item is the earlier one.
    pub(crate) replayed: bool,
    /// When meka stops retrying it: its enqueue plus `[serve] inbox_retry_ceiling`.
    pub(crate) retry_until: String,
    /// On a replayed item, as `GET /v1/sessions/{id}/inbox` shows them: when its next attempt
    /// may run and why the last one failed while it waits, or why it was given up on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) not_before: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_error: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) failure: Option<String>,
}

impl InboxResponse {
    fn new(item: InboxItemView, replayed: bool) -> Self {
        Self {
            item_id: item.id,
            session_id: item.session_id,
            class: item.class,
            state: item.state,
            replayed,
            retry_until: item.retry_until,
            not_before: item.not_before,
            last_error: item.last_error,
            failure: item.failure,
        }
    }
}

/// The answer to `GET /v1/sessions/{id}/inbox`: the items the model has not been shown, oldest
/// first.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct InboxListResponse {
    pub(crate) session_id: Uuid,
    pub(crate) items: Vec<InboxItemView>,
}

#[utoipa::path(
    post,
    path = "/v1/sessions/{id}/inbox",
    tag = "inbox",
    params(("id" = Uuid, Path, description = "Session UUID")),
    request_body = InboxRequest,
    responses(
        (status = 202, description = "Item recorded; the feed reports its delivery", body = InboxResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 409, description = "Another process holds the session", body = ProblemDetail),
        (status = 422, description = "Blank message, unknown class, or a sub-agent's session", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:w"]))
)]
pub(crate) async fn enqueue(
    State(state): State<ServerState>,
    scope::Scoped { principal, .. }: scope::Scoped<scope::SessionsWrite>,
    Path(session_id): Path<Uuid>,
    headers: HeaderMap,
    raw_body: Bytes,
) -> Result<(StatusCode, Json<InboxResponse>), ProblemDetail> {
    let body: InboxRequest = serde_json::from_slice(&raw_body)
        .map_err(|error| ProblemDetail::invalid_body("inbox", error))?;
    // The same doors `POST /turn` goes through, in the same order: a sub-agent's session and one
    // another process holds are refused before anything is written, and an evicted session is
    // revived so its driver has a runtime to run the item in.
    let entry = ensure_session_loaded(&state, session_id).await?;
    let source = body.source.filter(|source| !source.trim().is_empty());
    let mut item = NewInboxItem::from_parts(session_id, body.class, source, body.message.clone())
        .map_err(|error| {
        ProblemDetail::for_error(&error, state.config.relay_provider_errors)
    })?;
    if let Some(key) = idempotency_key(&headers)? {
        item = item.idempotent(principal.token_id.clone(), key);
    }
    let enqueued = state
        .shared
        .store
        .inbox_store()
        .enqueue(item)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to enqueue inbox item", error)
                .with("session_id", session_id.to_string())
        })?;
    // The row as written, fresh or replayed: the answer is the item, and a replayed key answers
    // with what the earlier item has become.
    let item = state
        .shared
        .store
        .inbox_store()
        .get(enqueued.id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to read the inbox item", error)
                .with("session_id", session_id.to_string())
        })?
        .ok_or_else(|| {
            ProblemDetail::new(
                ErrorKind::Internal,
                StatusCode::INTERNAL_SERVER_ERROR,
                "the enqueued inbox item is missing".to_string(),
            )
            .with("session_id", session_id.to_string())
        })?;
    // A retry that carries other words under the same key is the contract `POST /turn` refuses
    // too.
    if enqueued.replayed && (item.body != body.message || item.class != body.class) {
        return Err(ProblemDetail::new(
            ErrorKind::Idempotency,
            StatusCode::CONFLICT,
            "Idempotency-Key was used with another message on this session".to_string(),
        )
        .with("session_id", session_id.to_string())
        .with("item_id", item.id.to_string()));
    }
    entry.touch();
    // Whether or not a turn is running: a running one reads the inbox itself at its next
    // boundary, and the driver finds the session busy and leaves it to that.
    state.inbox_wake.notify_one();
    Ok((
        StatusCode::ACCEPTED,
        Json(InboxResponse::new(
            InboxItemView::new(item, state.config.inbox_retry_ceiling),
            enqueued.replayed,
        )),
    ))
}

/// The `Idempotency-Key` header, validated the way `POST /turn` validates it.
fn idempotency_key(headers: &HeaderMap) -> Result<Option<String>, ProblemDetail> {
    let Some(value) = headers.get("idempotency-key") else {
        return Ok(None);
    };
    let key = value.to_str().ok().map(str::trim).unwrap_or_default();
    if key.is_empty() || key.len() > 255 || !key.is_ascii() {
        return Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "Idempotency-Key must be non-empty ASCII of at most 255 characters",
        ));
    }
    Ok(Some(key.to_string()))
}

#[utoipa::path(
    get,
    path = "/v1/sessions/{id}/inbox",
    tag = "inbox",
    params(("id" = Uuid, Path, description = "Session UUID")),
    responses(
        (status = 200, description = "Items the model has not been shown, oldest first", body = InboxListResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:r"]))
)]
pub(crate) async fn list(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SessionsRead>,
    Path(session_id): Path<Uuid>,
) -> Result<Json<InboxListResponse>, ProblemDetail> {
    // Read from the store, not the map: a listing must not revive an evicted session.
    crate::host::http::reattach::require_session_exists(&state, session_id).await?;
    let items = state
        .shared
        .store
        .inbox_store()
        .list_open(session_id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to list inbox items", error)
                .with("session_id", session_id.to_string())
        })?;
    let ceiling = state.config.inbox_retry_ceiling;
    Ok(Json(InboxListResponse {
        session_id,
        items: items
            .into_iter()
            .map(|item| InboxItemView::new(item, ceiling))
            .collect(),
    }))
}

#[utoipa::path(
    delete,
    path = "/v1/sessions/{id}/inbox/{item_id}",
    tag = "inbox",
    params(
        ("id" = Uuid, Path, description = "Session UUID"),
        ("item_id" = Uuid, Path, description = "Inbox item id"),
    ),
    responses(
        (status = 204, description = "Item withdrawn; it will not reach the model"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such item on this session, or already closed", body = ProblemDetail),
        (status = 409, description = "The item is already in the conversation", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:w"]))
)]
pub(crate) async fn withdraw(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SessionsWrite>,
    Path((session_id, item_id)): Path<(Uuid, Uuid)>,
) -> Result<StatusCode, ProblemDetail> {
    let inbox = state.shared.store.inbox_store();
    let item = inbox.get(item_id).await.map_err(|error| {
        ProblemDetail::internal_sanitized("failed to read inbox item", error)
            .with("session_id", session_id.to_string())
    })?;
    // An item on another session is not found here, so an id cannot be used to reach across
    // sessions with a token that may not hold the other one.
    let not_found = || {
        ProblemDetail::new(
            ErrorKind::NotFound,
            StatusCode::NOT_FOUND,
            format!("no inbox item {item_id} is open on session {session_id}"),
        )
        .with("session_id", session_id.to_string())
    };
    match item {
        Some(item) if item.session_id == session_id => {}
        _ => return Err(not_found()),
    }
    match inbox.withdraw(item_id, None, None).await.map_err(|error| {
        ProblemDetail::internal_sanitized("failed to withdraw inbox item", error)
            .with("session_id", session_id.to_string())
    })? {
        Withdrawal::Withdrawn => {
            if let Some(entry) = state.sessions.read().await.get(&session_id).cloned() {
                entry.frontend.push_sse(
                    SseEventType::InboxWithdrawn,
                    serde_json::json!({ "item_id": item_id.to_string() }),
                );
            }
            Ok(StatusCode::NO_CONTENT)
        }
        Withdrawal::AlreadyAppended => Err(ProblemDetail::new(
            ErrorKind::InboxAppended,
            StatusCode::CONFLICT,
            format!(
                "inbox item {item_id} is already in the conversation; only a turn can answer it now"
            ),
        )
        .with("session_id", session_id.to_string())),
        Withdrawal::Closed | Withdrawal::Missing => Err(not_found()),
    }
}
