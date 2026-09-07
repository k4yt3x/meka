//! `POST /v1/sessions/{id}/turn`: submit a turn. Two response shapes:
//!
//! - **Blocking** (default, `stream: false`): `application/json` with the assembled
//!   [`TurnResponse`] once `Agent::run_turn` returns. The client gets the full transcript, tool
//!   calls, usage counters, and stop reason in one body.
//! - **Streaming** (`stream: true`): `text/event-stream` carrying live `turn.started` /
//!   `assistant_text.delta` / `tool_call.*` / `turn.finished` events (the full taxonomy is in the
//!   HTTP API docs § SSE events). Lifecycle events are 0-based and monotonic per turn.
//!
//! Both modes share an idempotency cache (Stripe-style, `Idempotency-Key` header). The cache
//! key is `(token_id, session_id, key)` and stores the *blocking* JSON envelope, so a replay of a
//! previously-streaming request returns the cached blocking body. Mid-turn permission gates
//! are handled out-of-band via `POST /v1/responses/{request_id}` on a side channel; the
//! streaming client sees a `permission_required` event and resolves via that endpoint without
//! interrupting the SSE response.

use std::{convert::Infallible, sync::Arc};

use axum::{
    Json,
    body::Bytes,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use futures::Stream;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use utoipa::ToSchema;
use uuid::Uuid;

use crate::{
    agent::TurnOutcome,
    conversation::ToolResultContent,
    frontend::{FrontendEvent, Notice, NoticeView},
    host::{
        TurnGuard,
        http::{
            errors::{ErrorKind, ProblemDetail},
            handlers::sessions::turn_in_flight_conflict,
            http_frontend::Recorder,
            idempotency::{LookupOutcome, hash_body},
            reattach::ensure_session_loaded,
            scope,
            state::ServerState,
        },
    },
};

/// Live broadcast capacity for a streaming turn. A consumer that falls this far behind is killed
/// rather than served a transcript with a hole in it; see the lag branch in [`build_sse_stream`].
const SSE_BROADCAST_CAPACITY: usize = 256;

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct TurnRequest {
    pub(crate) message: String,
    /// Image attachments for this turn. A sibling of `message` rather than a member of
    /// [`TurnOptions`] because these are user content, not a per-turn knob. Requires the session's
    /// profile to have vision enabled; see [`decode_turn_images`].
    #[serde(default)]
    pub(crate) images: Vec<ImageInput>,
    /// `false` (default) → blocking JSON response. `true` → SSE.
    #[serde(default)]
    pub(crate) stream: bool,
    /// Per-turn knobs. See [`TurnOptions`]. Omitting the field is the same as `{}`.
    #[serde(default)]
    pub(crate) options: TurnOptions,
}

/// One inline image attachment. Base64 rather than a path or URL because the API is a network
/// surface: `[serve].bind` may be non-loopback, so the caller generally shares no filesystem with
/// the agent and cannot name a file for it to read.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImageInput {
    /// Declared MIME type, e.g. `image/png`. Used as the primary format hint; the payload's magic
    /// bytes win if this doesn't name a supported format.
    pub(crate) media_type: String,
    /// Base64-encoded image bytes (standard alphabet, padding required).
    pub(crate) data: String,
}

/// Per-turn options. `#[serde(deny_unknown_fields)]` here (and only here) so a typo in
/// `option.skil` surfaces as a 422 rather than being silently dropped.
#[derive(Debug, Deserialize, ToSchema, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct TurnOptions {
    /// Optional installed-skill name. When set, `message` is combined with the skill's body (user
    /// text first, then the skill body) before the agent runs, matching `/skill <name> <prompt>`
    /// in the REPL and the `--skill` CLI flag. An unknown skill is a 422.
    #[serde(default)]
    pub(crate) skill: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct TurnResponse {
    pub(crate) turn_id: Uuid,
    pub(crate) session_id: Uuid,
    pub(crate) stop_reason: String,
    /// Concatenated assistant text produced this turn. **Excludes** refusal explanation:
    /// when the model refuses, the refusal text rides on the dedicated `refusal_text` field
    /// instead. Clients that just want "what the user sees" should consume both:
    /// `final_text` for the normal response, `refusal_text` when `stop_reason == "refusal"`.
    pub(crate) final_text: String,
    /// Refusal explanation when `stop_reason == "refusal"`; `None` otherwise. Mirrors the
    /// `refusal_text` field on the streaming `turn.finished` SSE event, so blocking and
    /// streaming clients share the same shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) refusal_text: Option<String>,
    /// Structured view of the assistant's message(s) produced this turn. Per the spec, this is
    /// the "richer access" companion to `final_text`. Clients that want the text plus its
    /// formatting context (text/thinking content blocks) consume this; clients that just want
    /// a single string consume `final_text`. Tool calls live in their own `tool_calls` array.
    pub(crate) messages: Vec<crate::host::http::handlers::messages::MessageView>,
    pub(crate) tool_calls: Vec<ToolCallView>,
    pub(crate) usage: UsageView,
    pub(crate) notices: Vec<NoticeView>,
}

#[derive(Debug, Serialize, Default, ToSchema)]
pub(crate) struct UsageView {
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) cache_creation_input_tokens: u64,
    pub(crate) cache_read_input_tokens: u64,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ToolCallView {
    pub(crate) id: String,
    pub(crate) name: String,
    #[schema(value_type = Object)]
    pub(crate) input: serde_json::Value,
    /// The one-line label a terminal shows for the call, when the tool has one. Omitted otherwise,
    /// like every optional field on this API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) display_summary: Option<String>,
    pub(crate) is_error: bool,
    pub(crate) content: Vec<ToolCallContentView>,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ToolCallContentView {
    Text { text: String },
    Image { media_type: String },
}

#[utoipa::path(
    post,
    path = "/v1/sessions/{id}/turn",
    tag = "turn",
    params(
        ("id" = Uuid, Path, description = "Session UUID"),
        ("Idempotency-Key" = Option<String>, Header, description = "Stripe-style replay key. Same key + same body returns the cached response; same key + different body returns 409."),
    ),
    request_body = TurnRequest,
    responses(
        (status = 200, description = "Blocking turn response (stream=false) or live SSE stream (stream=true). The application/json schema applies only to blocking mode.", body = TurnResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 409, description = "A turn is already in flight (`/errors/turn-in-flight`), the Idempotency-Key was used with another body or is in flight on a concurrent request (`/errors/idempotency`), or another meka process holds the session (`/errors/session-locked`)", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid body, the id names a sub-agent's conversation, or the request still exceeds the profile's `max_request_bytes` after redaction. Read `type`: `/errors/invalid-body` is worth resending with a corrected payload, `/errors/request-too-large` with a smaller one or after `/compact`, `/errors/session-not-drivable` never is", body = ProblemDetail),
        (status = 429, description = "Concurrency limit reached or idempotency-key cache full", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
        (status = 502, description = "The provider rejected or failed this turn. Read `type`: `/errors/provider-unavailable` was classified as transient and is worth resending after a pause, `/errors/provider` is the catch-all for everything else and usually needs the account or endpoint fixed, and `/errors/context-overflow` needs the conversation shortened first", body = ProblemDetail),
        (status = 503, description = "An MCP server marked `required` was not connected, so the turn never reached the provider", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:w"]))
)]
pub(crate) async fn submit_turn(
    State(state): State<ServerState>,
    scope::Scoped { principal, .. }: scope::Scoped<scope::SessionsWrite>,
    Path(session_id): Path<Uuid>,
    headers: HeaderMap,
    raw_body: Bytes,
) -> Result<Response, ProblemDetail> {
    // Parse the header + body before consulting the idempotency cache. A malformed header /
    // body returns 422 cheaply; a successful parse lets us peek `stream` so we can skip
    // idempotency entirely for SSE replays.
    let idempotency_key = idempotency_header(&headers)?;
    let body: TurnRequest = serde_json::from_slice(&raw_body)
        .map_err(|error| ProblemDetail::invalid_body("turn", error))?;

    // Streaming turns can't be replayed (no envelope to cache), so the key is silently ignored
    // there.
    //
    // `lookup_and_mark` atomically inserts a `Pending` sentinel on miss and hands us a
    // ticket; concurrent same-keyed requests see `InFlight` and 409.  The ticket commits on
    // completion via the rollback-on-Drop pattern so a panic doesn't block retries forever.
    let body_hash = hash_body(&raw_body);
    // The session this turn acts on is part of the key. An `Idempotency-Key` names the *client's*
    // unit of work, so reusing one across two sessions is the natural thing to do, and without the
    // scope the second request replayed the first session's transcript and never ran its turn.
    let idempotency_scope = session_id.to_string();
    let cacheable_key = if body.stream { None } else { idempotency_key };
    let idempotency_ticket: Option<crate::host::http::idempotency::IdempotencyTicket> =
        if let Some(key) = cacheable_key.as_deref() {
            match state
                .idempotency
                .lookup_and_mark(&principal.token_id, &idempotency_scope, key, &body_hash)
                .await
            {
                LookupOutcome::Hit(entry) => {
                    tracing::debug!(
                        "idempotency hit: token={token} key={key} bytes={bytes}",
                        token = principal.token_id,
                        bytes = entry.body.len()
                    );
                    return Ok(cached_response_into_axum(entry));
                }
                LookupOutcome::Conflict => {
                    return Err(ProblemDetail::new(
                        ErrorKind::Idempotency,
                        StatusCode::CONFLICT,
                        "Idempotency-Key was used with a different request body; a replay must be \
                         byte-identical",
                    ));
                }
                LookupOutcome::InFlight => {
                    return Err(ProblemDetail::new(
                        ErrorKind::Idempotency,
                        StatusCode::CONFLICT,
                        "Idempotency-Key is in flight on a concurrent request; retry after it \
                         completes",
                    ));
                }
                LookupOutcome::CapExceeded => {
                    let mut problem = ProblemDetail::new(
                        ErrorKind::Idempotency,
                        StatusCode::TOO_MANY_REQUESTS,
                        "per-token idempotency-key cache is full; retry once in-flight requests \
                         complete",
                    )
                    .with_retry_after(60);
                    // Override the generic "conflict" title: this is cache pressure, not a
                    // body-mismatch conflict.
                    problem.title = "Idempotency-Key cache capacity exceeded".to_string();
                    return Err(problem);
                }
                LookupOutcome::Miss(ticket) => Some(ticket),
            }
        } else {
            None
        };

    let message = if let Some(skill_name) = body.options.skill.as_deref() {
        let snapshot = state.shared.skills.current().await;
        let skill = snapshot.find(skill_name).ok_or_else(|| {
            // `unavailable`, not a flat "unknown skill": a `SKILL.md` that will not parse is in no
            // index, and telling the caller their skill does not exist sends them looking for a
            // file that is sitting in the store.
            ProblemDetail::new(
                ErrorKind::InvalidBody,
                StatusCode::UNPROCESSABLE_ENTITY,
                snapshot.unavailable(skill_name),
            )
        })?;
        // Sanitized: the reason names the file, and the skills directory is the operator's to
        // read about in the log rather than the caller's.
        let skill_body = crate::skills::load_skill_body(skill)
            .await
            .map_err(|error| {
                ProblemDetail::internal_sanitized(
                    &format!("failed to load skill '{skill_name}'"),
                    error,
                )
            })?;
        // The composition `--skill [PROMPT]` and the REPL's `/skill` use: the body alone when
        // nothing was typed, so a skill invoked without a message is a turn rather than a blank
        // line above one.
        if body.message.trim().is_empty() {
            skill_body
        } else {
            format!("{}\n\n{}", body.message, skill_body)
        }
    } else {
        body.message
    };

    // Resolved before anything is admitted, because the rest of the body's validation needs it:
    // whether an attachment is admissible is a fact about *this session's* profile. Loading an
    // evicted session is the one side effect, and it is one a refused request may leave behind.
    let resident = state.sessions.read().await.get(&session_id).cloned();
    let entry = match resident {
        Some(entry) => entry,
        None => ensure_session_loaded(&state, session_id).await?,
    };
    let images = decode_turn_images(&body.images, entry.accepts_images()).await?;
    // An image with no text passes; against prior context "look at this" is a complete request.
    let input = crate::agent::TurnInput::from_parts(message, images)
        .map_err(|error| ProblemDetail::for_error(&error, state.config.relay_provider_errors))?;

    // Taken only once the request is known to be one meka will act on. The guard marks the
    // session as busy, and a request rejected above never runs a turn, so acquiring first made a
    // malformed body answer 409 on the session's own PATCH, DELETE, compact and rewind for as long
    // as the guard lived, and answered 409 to a malformed body whenever a real turn was running.
    //
    // Hold the sessions read-lock across `TurnGuard::acquire` to close the TOCTOU gap: DELETE's
    // write-lock blocks behind any reader, so by the time it fires we've already bumped
    // `in_flight > 0` and DELETE's re-check returns 409.
    let turn_guard = {
        let _map = state.sessions.read().await;
        crate::host::http::state::admit_turn(&state, &entry)?
    };

    // Taken here rather than inside the two arms below, and *before* the claim: it is the only
    // per-session admission gate (`TurnGuard::acquire` counts, it does not refuse), so claiming
    // first meant a 409 stamped an outcome delivered that no turn ever delivered, and
    // `list_undelivered_background_tasks` never returns a stamped row again. The guard moves into
    // whichever arm runs, holding from admission to the end of the turn.
    let conversation = Arc::clone(&entry.conversation)
        .try_lock_owned()
        .map_err(|_| turn_in_flight_conflict(session_id, "run a turn"))?;

    // An outcome that did not warrant a turn of its own rides on this one, ahead of the caller's
    // message rather than as a message of its own: see `background::render_outcomes_riding`. Both
    // arms below take the joined prompt, so neither surface can be the one that forgets.
    let outcomes = if state.shared.config.background.enabled {
        crate::host::claim_undelivered_outcomes(&entry.agent, &state.shared.store, session_id).await
    } else {
        Vec::new()
    };
    // This turn and the poller are the two ways an outcome leaves the undelivered pool, and
    // subscribers are owed it whichever one takes it: a task that finishes between two ticks and is
    // claimed here would otherwise never be announced at all.
    // Best-effort here, unlike the poller: this claim is already stamped, and refusing the user's
    // turn over a webhook bookkeeping failure would be the wrong trade. It is logged.
    let _announced = state
        .webhooks
        .announce_finished_tasks(&state.shared.store.background_store(), &outcomes)
        .await;
    let input = input.riding(outcomes);

    if body.stream {
        // SSE responses are streamed live and aren't a single envelope we can cache.
        run_streaming_turn(state, entry, conversation, session_id, input, turn_guard)
    } else {
        // The ticket travels *into* the turn, and is committed by the task that runs it rather
        // than here. The turn now outlives a client that hangs up, so leaving the commit on this
        // side would mean a request timeout drops the ticket, rolls the `Pending` slot back, and
        // lets the documented retry run a second full turn over work the first one already
        // committed -- duplicating its tool calls and its provider bill. Committing where the work
        // finishes is what makes the key mean what the retry-safety table says it means.
        run_blocking_turn(
            state,
            entry,
            conversation,
            session_id,
            input,
            turn_guard,
            idempotency_ticket,
        )
        .await
        .map(IntoResponse::into_response)
    }
}

/// Cancel the turn when the consumer that just lagged was the only one reading it.
///
/// Turn events are broadcast, so a re-attached client or a second consumer is a separate receiver.
/// Canceling unconditionally took the turn away from clients that were keeping up, on the say-so
/// of one slow reader.
///
/// The lagging receiver is still live when this runs -- it is the local binding the `Lagged` arm
/// was reached through -- so it counts itself. That is why the threshold is `<= 1` rather than
/// `== 0`: one means "the lagger and nobody else". Do not "fix" this to exclude it without moving
/// the threshold in the same commit, or the decision inverts in both directions.
///
/// A named function rather than three lines inline: the call site sits inside an SSE generator's
/// `Lagged` arm, which no test can reach without forcing a broadcast overflow against two live
/// consumers. Returns whether the turn was canceled, which decides what the caller may truthfully
/// tell the client: a turn that is still running for someone else has not failed, and saying it
/// has sends this client to retry into a 409 `turn-in-flight`.
fn cancel_if_nobody_else_is_reading(
    frontend: &crate::host::http::http_frontend::HttpFrontend,
    cancellation: &CancellationToken,
) -> bool {
    let remaining = frontend.subscriber_count();
    if remaining <= 1 {
        frontend.note_canceled_for_lag();
        cancellation.cancel();
        true
    } else {
        tracing::debug!(
            "not canceling the turn: {others} other SSE consumer(s) are still reading",
            others = remaining.saturating_sub(1)
        );
        false
    }
}

/// The event a lagging consumer's stream ends with, as `(event type, payload)`.
///
/// Two different facts. When the turn was canceled it really did fail, and a retry is the remedy.
/// When it was not, the turn is still running for the consumer that kept up: `turn.failed` would
/// be a lie, and the retry it invites returns 409 `turn-in-flight`. Re-attaching with
/// `Last-Event-ID` is the remedy there, and it recovers the dropped events rather than redoing
/// them.
///
/// The notice is the `level`/`text` shape every other `notice` event carries, with the ids beside
/// it so the client can confirm which turn it is being told to rejoin.
fn lag_event_parts(
    canceled: bool,
    skipped: u64,
    turn_id: Uuid,
    session_id: Uuid,
) -> (crate::host::http::sse::SseEventType, serde_json::Value) {
    if canceled {
        let problem = crate::host::http::errors::ProblemDetail::new(
            crate::host::http::errors::ErrorKind::SseLag,
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "SSE consumer fell behind and {skipped} event(s) were dropped, so the turn was \
                 canceled; retry it"
            ),
        )
        .instance(format!("/v1/sessions/{session_id}/turn"));
        return (
            crate::host::http::sse::SseEventType::TurnFailed,
            serde_json::json!({
                "turn_id": turn_id.to_string(),
                "session_id": session_id.to_string(),
                "error": serde_json::to_value(problem).unwrap_or(serde_json::Value::Null),
            }),
        );
    }
    let notice = Notice::warn(format!(
        "SSE consumer fell behind and {skipped} event(s) were dropped; the turn is still running, \
         so re-attach with Last-Event-ID"
    ));
    let mut data =
        serde_json::to_value(NoticeView::from(notice)).unwrap_or(serde_json::Value::Null);
    if let Some(object) = data.as_object_mut() {
        object.insert("turn_id".into(), turn_id.to_string().into());
        object.insert("session_id".into(), session_id.to_string().into());
    }
    (crate::host::http::sse::SseEventType::Notice, data)
}

/// Record a finished blocking turn against its `Idempotency-Key`, if it had one.
///
/// Caches success (2xx) and client-error (4xx) envelopes. Server-side errors (5xx) and
/// `TurnInFlight` are skipped: a transient provider 502 would otherwise be replayed for the full
/// 24h TTL, defeating the point of an idempotent retry; `TurnInFlight` means the turn was never
/// attempted at all; and `TurnCanceled` means it was interrupted, which is a fact about one
/// attempt rather than about the request. Caching that one pinned "canceled" as the answer for the
/// next 24 hours, so the retry the cancellation invites could never run. In all three cases the
/// ticket's `Drop` clears the `Pending` entry so a retry re-executes.
async fn commit_idempotency(
    ticket: Option<crate::host::http::idempotency::IdempotencyTicket>,
    session_id: Uuid,
    response: &Result<Json<TurnResponse>, ProblemDetail>,
) {
    let Some(ticket) = ticket else {
        return;
    };
    let skip_cache = matches!(
        response,
        Err(problem) if problem.status >= 500
            || problem.is(ErrorKind::TurnInFlight)
            || problem.is(ErrorKind::TurnCanceled)
    );
    if skip_cache {
        tracing::debug!(
            session_id = %session_id,
            "not caching this turn against its Idempotency-Key; a retry will re-execute"
        );
        return;
    }
    let (status, bytes) = match response {
        Ok(json) => (StatusCode::OK, serde_json::to_vec(&json.0)),
        Err(problem) => (
            StatusCode::from_u16(problem.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            serde_json::to_vec(problem),
        ),
    };
    if let Ok(bytes) = bytes {
        ticket.commit(status.as_u16(), bytes).await;
    }
    // If serialization failed (extraordinarily unlikely; TurnResponse / ProblemDetail are both
    // pure-data serde types), drop the ticket without commit so the Pending entry is removed and
    // clients can retry instead of hitting a permanent 409.
}

/// Validate and normalize a turn's image attachments through the shared client-image pipeline
/// ([`crate::image::decode_base64_image`]), so an HTTP attachment gets exactly the size cap and
/// format conversion an ACP `image` content block does.
///
/// `vision` is the session's answer from `ResidentSession::accepts_images`. Attachments are
/// refused outright when it is off, as ACP refuses `image` content blocks with `InvalidParams` for
/// a text-only profile. This gates on configuration only: whether the named model *actually*
/// understands images is left to the provider to complain about, which is the stance documented
/// for ACP.
///
/// Every failure is a 422: these are all malformed input, not server faults.
///
/// Off the runtime, for the reason `read_file` and `fetch_url` document at their own call sites:
/// the pipeline base64-decodes and then decodes each image to verify it, which is tens of
/// milliseconds of pure CPU on a multi-megapixel attachment, and on the runtime it blocks every
/// other task on that worker. One client posting a screenshot must not stall an unrelated
/// session's stream.
///
/// The `allow` matches the rest of this module's validation helpers: `ProblemDetail` is a large
/// struct by design (RFC 9457 members plus an extensions map) and boxing it here alone would make
/// the error type inconsistent with every other handler.
async fn decode_turn_images(
    images: &[ImageInput],
    vision: bool,
) -> Result<Vec<crate::image::ImageSource>, ProblemDetail> {
    if images.is_empty() {
        return Ok(Vec::new());
    }
    if !vision {
        return Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "image attachments require a profile with vision enabled; set `vision = true` under \
             `[profiles.<name>]`",
        ));
    }
    let owned: Vec<(String, String)> = images
        .iter()
        .map(|image| (image.data.clone(), image.media_type.clone()))
        .collect();
    tokio::task::spawn_blocking(move || {
        owned
            .iter()
            .enumerate()
            .map(|(index, (data, media_type))| {
                crate::image::decode_base64_image(data, media_type).map_err(|message| {
                    ProblemDetail::new(
                        ErrorKind::InvalidBody,
                        StatusCode::UNPROCESSABLE_ENTITY,
                        format!("`images[{index}]` is invalid: {message}"),
                    )
                })
            })
            .collect()
    })
    .await
    .map_err(|error| {
        ProblemDetail::new(
            ErrorKind::Internal,
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to decode the images: {error}"),
        )
    })?
}

/// Extract the `Idempotency-Key` header, validating that it isn't empty and stays within
/// reasonable size bounds. Returns `Ok(None)` when the header is absent.
///
/// All validation failures map to 422 `invalid-body` so the status code is consistent with the
/// body-parse error path and matches the spec's error-catalog table for `invalid-body`.
fn idempotency_header(headers: &HeaderMap) -> Result<Option<String>, ProblemDetail> {
    let Some(value) = headers.get("idempotency-key") else {
        return Ok(None);
    };
    let value = value.to_str().map_err(|_| {
        ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "Idempotency-Key header must be ASCII",
        )
    })?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "Idempotency-Key header must not be empty",
        ));
    }
    if trimmed.len() > 255 {
        return Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "Idempotency-Key header is too long (max 255 chars)",
        ));
    }
    Ok(Some(trimmed.to_string()))
}

/// Re-build an `axum::Response` from a [`crate::host::http::idempotency::CachedResponse`]. Sets the
/// same status code the original handler returned, and picks the content-type to match: a 2xx
/// body is a serialized `TurnResponse` (`application/json`); a 4xx/5xx body is a serialized
/// `ProblemDetail` (`application/problem+json` per RFC 9457).
fn cached_response_into_axum(entry: crate::host::http::idempotency::CachedResponse) -> Response {
    let status = StatusCode::from_u16(entry.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let content_type = if status.is_success() {
        "application/json"
    } else {
        "application/problem+json"
    };
    (
        status,
        [(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static(content_type),
        )],
        axum::body::Body::from(entry.body),
    )
        .into_response()
}

/// RAII guard that clears the per-session `StreamSink` on drop so both normal completion
/// and panics reset the cell. Without this, a panic leaves a zero-subscriber sink that
/// causes subsequent blocking turns to 500 via `client_disconnected()`.
struct StreamGuard {
    frontend: Arc<crate::host::http::http_frontend::HttpFrontend>,
}

impl StreamGuard {
    fn new(frontend: Arc<crate::host::http::http_frontend::HttpFrontend>) -> Self {
        Self { frontend }
    }
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        self.frontend.end_stream();
    }
}

// Eight now that the relay policy travels with the turn. Bundling them into a struct would hide
// which ones the spawned task actually captures, and taking `ServerState` whole would hand this
// function the session map and the idempotency cache it has no business touching.
async fn run_blocking_turn(
    state: ServerState,
    entry: crate::host::http::state::SessionEntry,
    mut conversation: tokio::sync::OwnedMutexGuard<crate::conversation::Conversation>,
    session_id: Uuid,
    input: crate::agent::TurnInput,
    turn_guard: TurnGuard,
    idempotency_ticket: Option<crate::host::http::idempotency::IdempotencyTicket>,
) -> Result<Json<TurnResponse>, ProblemDetail> {
    // The turn runs on a spawned task, for the same reason the streaming path does it: axum drops
    // a handler's future when the client disconnects, and a turn is not a computation that can be
    // abandoned halfway. The runtime guard comes from `submit_turn`, which takes it as the
    // admission check before anything is written.
    //
    // A blocking turn outlives most client timeouts -- the default request has no SSE to keep the
    // connection warm, so a 30s reqwest timeout against a turn that runs a build is the ordinary
    // case, not an exotic one. Dropped mid-`execute_tool_calls` the future would take the running
    // command's future with it, orphaning its process group (nothing calls `kill_child_tree` on
    // the drop path), leave the in-memory conversation holding an assistant `tool_use` whose
    // result was never appended, and skip `notify_turn_end` so a webhook subscriber never hears
    // the turn end at all. The DB stays consistent, since `run_turn` commits each round trip as it
    // completes, but the resident session then sends a dangling `tool_use` on its next turn and
    // eats a provider rejection to repair it.
    //
    // Spawning makes a dropped response detach rather than abort, which is exactly what
    // `SessionResponse::turn_in_flight` already documents for the streaming path: the work
    // completes, and a client that reconnects reads the reply out of `GET /messages`. It also
    // turns a panic inside the turn into a clean 500 instead of a reset connection.

    let _stale = entry.frontend.drain();

    // Publish the cancellation token *after* the mutex has been acquired, which `submit_turn` did
    // before dispatching here. Publishing without the lock would let a rejected Turn B overwrite a
    // running Turn A's token, making Turn A uncancelable. The window between the acquire and this
    // publish is harmless whatever its length: `POST /cancel` reading the old (session-creation or
    // prior-turn) token is a no-op on an already-finished turn.
    let cancellation = CancellationToken::new();
    let published = entry
        .cancel
        .publish(cancellation.clone(), turn_guard.admission);
    let turn_id = uuid::Uuid::new_v4();

    let join = tokio::spawn(async move {
        let _turn_guard = turn_guard;
        let _published = published;
        let outcome = entry
            .agent
            .run_turn(&mut conversation, input, cancellation)
            .await;

        let recorder = entry.frontend.drain();
        entry.touch();

        // Announced from the blocking path too, and from inside the task so it still fires when
        // the client that asked has gone. The requester has its answer in the response body, but
        // it is not necessarily the only party interested in the session, and a webhook subscriber
        // should not have to care which transport a turn happened to use.
        let response = match outcome {
            Ok(turn_outcome) => {
                notify_turn_end(
                    &state.webhooks,
                    crate::host::http::sse::SseEventType::TurnFinished,
                    turn_id,
                    session_id,
                );
                Ok(Json(assemble_response(
                    turn_id,
                    session_id,
                    turn_outcome,
                    recorder,
                    entry.capabilities,
                )))
            }
            Err(error) => {
                // `Interrupted` is a cancellation, and `notify_turn_end` drops those on the floor;
                // routing it through keeps the classification in one place.
                let event_type = if matches!(error, crate::error::MekaError::Interrupted) {
                    crate::host::http::sse::SseEventType::TurnCanceled
                } else {
                    crate::host::http::sse::SseEventType::TurnFailed
                };
                notify_turn_end(&state.webhooks, event_type, turn_id, session_id);
                Err(ProblemDetail::for_error(
                    &error,
                    state.config.relay_provider_errors,
                ))
            }
        };
        // Inside the task, so a client that hung up still records its outcome against the key.
        commit_idempotency(idempotency_ticket, session_id, &response).await;
        response
    });

    match join.await {
        Ok(result) => result,
        Err(panic) => {
            tracing::error!("blocking turn task panicked: {panic:?}");
            Err(ProblemDetail::new(
                ErrorKind::Internal,
                StatusCode::INTERNAL_SERVER_ERROR,
                "turn task panicked",
            )
            .with("session_id", session_id.to_string()))
        }
    }
}

/// Run a turn with `stream: true`. Returns an SSE response that emits events live as the agent
/// produces them, plus a terminal `turn.finished` (or `turn.failed` / `turn.canceled`) event
/// before closing.
fn run_streaming_turn(
    state: ServerState,
    entry: crate::host::http::state::SessionEntry,
    owned_conversation: tokio::sync::OwnedMutexGuard<crate::conversation::Conversation>,
    session_id: Uuid,
    input: crate::agent::TurnInput,
    turn_guard: TurnGuard,
) -> Result<Response, ProblemDetail> {
    // Subscribe to the broadcast BEFORE installing: install_stream returns a receiver that
    // captures the first event onwards.
    let _stale = entry.frontend.drain();
    // Minted before the stream is installed so the ring is keyed by it from the first event; a
    // re-attaching client reads the id back to confirm it rejoined the turn it thought it had.
    let turn_id = uuid::Uuid::new_v4();
    let (receiver, ids) = entry.frontend.install_stream(
        SSE_BROADCAST_CAPACITY,
        state.config.stream_replay_events,
        state.config.stream_reattach_grace,
        turn_id,
    );

    // Publish after the lock succeeds. Same rationale as `run_blocking_turn`.
    let cancellation = CancellationToken::new();
    let published = entry
        .cancel
        .publish(cancellation.clone(), turn_guard.admission);

    let entry_for_task = entry.clone();
    let cancel_for_task = cancellation.clone();
    let shutdown_for_task = state.shutdown.clone();
    let relay_for_task = state.config.relay_provider_errors;
    let webhooks_for_task = state.webhooks;

    // Spawn the turn so the SSE response can return immediately.
    //
    // Declaration order is load-bearing: locals drop in reverse, so `_stream_guard` goes first,
    // then `conversation`, then `_turn_guard`. That is what keeps the conversation mutex held
    // across `end_stream()`, so a turn admitted the instant this one ends cannot install its
    // stream into a frontend the outgoing turn is still tearing down.
    let join = tokio::spawn(async move {
        let _turn_guard = turn_guard;
        let _published = published;
        let mut conversation = owned_conversation;
        let _stream_guard = StreamGuard::new(Arc::clone(&entry_for_task.frontend));
        let outcome = entry_for_task
            .agent
            .run_turn(&mut conversation, input, cancel_for_task)
            .await;
        entry_for_task.touch();
        let usage = drain_recorder_and_extract_usage(&entry_for_task.frontend);
        // Computed and recorded *here*, in the task, rather than in the response stream below.
        // In the case re-attach exists for, the client's connection has already dropped and axum
        // has discarded that stream, so a terminal event computed there would be computed for
        // nobody and a reconnecting client would wait forever for an end that never comes.
        let cancel_reason = if shutdown_for_task.is_cancelled() {
            CancelReason::ServerShutdown
        } else if entry_for_task.frontend.canceled_for_lag() {
            CancelReason::SseLag
        } else {
            CancelReason::Client
        };
        let (event_type, data) = terminal_event_parts(
            Ok(outcome),
            cancel_reason,
            usage,
            turn_id,
            session_id,
            relay_for_task,
        );
        notify_turn_end(&webhooks_for_task, event_type, turn_id, session_id);
        entry_for_task.frontend.record_terminal(event_type, data)
    });

    // Build the SSE stream. Emits the per-FrontendEvent events from the broadcast, then the
    // terminal event the spawned task recorded when the join handle resolves. The loop no longer
    // watches either token itself: the drain fires every session's cancellation token directly
    // (`host::http::drain_active_sessions`), and the task reads the shutdown token to decide
    // whether its terminal says `server_shutdown` or `client`.
    let stream = build_sse_stream(
        turn_id,
        session_id,
        receiver,
        join,
        cancellation,
        ids,
        Arc::clone(&entry.frontend),
    );
    let sse = Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(20))
            .text("keep-alive"),
    );

    let mut response = sse.into_response();
    response.headers_mut().insert(
        "X-Accel-Buffering",
        axum::http::HeaderValue::from_static("no"),
    );
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-cache, no-transform"),
    );
    // No explicit `Connection: keep-alive`: it's forbidden on HTTP/2 (RFC 9113 §8.2.2)
    // and hyper sets it automatically on HTTP/1.1.
    Ok(response)
}

fn build_sse_stream(
    turn_id: Uuid,
    session_id: Uuid,
    mut receiver: tokio::sync::broadcast::Receiver<crate::host::http::sse::SseEvent>,
    join: tokio::task::JoinHandle<crate::host::http::sse::SseEvent>,
    cancellation: CancellationToken,
    ids: Arc<crate::host::http::sse::EventIdGenerator>,
    frontend: Arc<crate::host::http::http_frontend::HttpFrontend>,
) -> impl Stream<Item = Result<Event, Infallible>> + Send {
    async_stream::stream! {
        // Per spec §SSE production-concerns: hint clients to reconnect after 3s on disconnect.
        // Must be the first thing on the wire (before any `id:`/`event:` lines). The `retry:`
        // field has no `id:` by SSE spec.
        yield Ok::<_, Infallible>(Event::default().retry(std::time::Duration::from_secs(3)));

        // Emit the turn.started lifecycle event. The agent's own TurnStarted is filtered out
        // by sse.rs::translate; this richer envelope carries turn_id + session_id + started_at
        // for clients building lifecycle timelines. The id is drawn from the same generator the
        // broadcast events use so per-turn ids stay monotonic and dense.
        let started_at = chrono::Utc::now().to_rfc3339();
        yield Ok(crate::host::http::sse::SseEvent {
            id: ids.next(),
            event_type: crate::host::http::sse::SseEventType::TurnStarted,
            data: serde_json::json!({
                "turn_id": turn_id,
                "session_id": session_id,
                "started_at": started_at,
            }),
        }.into_axum());

        let mut join = Box::pin(join);
        loop {
            tokio::select! {
                biased;
                event = receiver.recv() => {
                    match event {
                        Ok(sse) => yield Ok(sse.into_axum()),
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                            tracing::warn!(
                                "SSE consumer lagged, skipped {skipped} events; terminating stream"
                            );
                            // Stop burning provider tokens for a consumer that has lost data and
                            // will need to retry, but only when nobody else is still reading.
                            let canceled = cancel_if_nobody_else_is_reading(&frontend, &cancellation);
                            let (event_type, data) =
                                lag_event_parts(canceled, skipped, turn_id, session_id);
                            yield Ok(if canceled {
                                Event::default()
                                    .id(ids.next().to_string())
                                    .event(event_type.as_str())
                                    .json_data(data)
                                    .unwrap_or_else(|_| Event::default().comment("lag-failed serialize-failed"))
                            } else {
                                // Deliberately carries no `id`.
                                //
                                // Event ids are session-wide and monotonic, so an id here would be
                                // strictly greater than every event this consumer just lost. A
                                // client doing the obvious thing (remember the last id, re-attach
                                // with it) would then ask to resume *after* the gap, get an empty
                                // backlog and `gap: false`, and carry on missing exactly the events
                                // this notice exists to tell it about. Per the SSE spec an event
                                // with no `id:` field leaves the client's last-event-id buffer
                                // alone, so it re-attaches from the last event it actually received
                                // and the backlog replays the gap.
                                Event::default()
                                    .event(event_type.as_str())
                                    .json_data(data)
                                    .unwrap_or_else(|_| Event::default().comment("lag-notice serialize-failed"))
                            });
                            break;
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                            yield Ok(join_terminal(&mut join, turn_id, session_id).await);
                            break;
                        }
                    }
                }
                turn_result = &mut join => {
                    // Agent finished; flush remaining buffered events before the terminal one.
                    while let Ok(sse) = receiver.try_recv() {
                        yield Ok(sse.into_axum());
                    }
                    yield Ok(match turn_result {
                        Ok(terminal) => terminal.into_axum(),
                        Err(panic) => panic_terminal(panic, turn_id, session_id),
                    });
                    break;
                }
            }
        }
    }
}

/// Why a turn that ended `Interrupted` was stopped, for the recorded `turn.canceled` event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CancelReason {
    /// `POST /cancel`.
    Client,
    /// The graceful drain.
    ServerShutdown,
    /// The only SSE consumer fell behind and the turn was stopped for it. Recorded as `client`, a
    /// re-attaching reader would conclude a human had stopped it.
    SseLag,
}

impl CancelReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Client => "client",
            Self::ServerShutdown => "server_shutdown",
            Self::SseLag => "sse_lag",
        }
    }
}

/// Resolve a finished turn into the `(event type, envelope)` of its terminal SSE event.
///
/// Returns the parts rather than a rendered `Event` because the terminal has to be *stored* as
/// well as sent: [`crate::host::http::http_frontend::HttpFrontend::record_terminal`] keeps it so a
/// client that reconnects after the turn ended still learns how it ended.
///
/// A successful agent outcome always wins over a concurrent cancel signal, so a race between
/// completion and cancellation does not discard an already-persisted result.
fn terminal_event_parts(
    turn_result: std::result::Result<crate::error::Result<TurnOutcome>, tokio::task::JoinError>,
    cancel_reason: CancelReason,
    usage: UsageView,
    turn_id: Uuid,
    session_id: Uuid,
    relay_provider_errors: bool,
) -> (crate::host::http::sse::SseEventType, serde_json::Value) {
    match turn_result {
        Ok(Ok(outcome)) => finished_parts(&outcome, usage, turn_id, session_id),
        Ok(Err(crate::error::MekaError::Interrupted)) => {
            // Every stop surfaces as `Interrupted` by the time the agent loop unwinds; who asked
            // for it is carried alongside.
            canceled_parts(cancel_reason.as_str(), turn_id, session_id)
        }
        Ok(Err(error)) => {
            let instance = format!("/v1/sessions/{session_id}/turn");
            let problem =
                crate::host::http::errors::ProblemDetail::for_error(&error, relay_provider_errors)
                    .instance(instance);
            (
                crate::host::http::sse::SseEventType::TurnFailed,
                serde_json::json!({
                    "turn_id": turn_id.to_string(),
                    "session_id": session_id.to_string(),
                    "error": serde_json::to_value(problem).unwrap_or(serde_json::Value::Null),
                }),
            )
        }
        Err(panic) => {
            tracing::error!("streaming turn task panicked: {panic:?}");
            (
                crate::host::http::sse::SseEventType::TurnFailed,
                serde_json::json!({
                    "turn_id": turn_id.to_string(),
                    "session_id": session_id.to_string(),
                    "error": {
                        "type": "https://meka.so/errors/internal",
                        "title": "Internal server error",
                        "status": 500,
                        "detail": "turn task panicked",
                        "instance": format!("/v1/sessions/{}/turn", session_id),
                    },
                }),
            )
        }
    }
}

/// Announce a finished turn to any configured webhook endpoint.
///
/// Only the terminal *outcome* travels: ids, and whether it ended or failed. A subscriber that
/// wants the reply reads `GET /v1/sessions/{id}/messages` with its own token, over the API it
/// already authenticates against. A webhook URL is a config-file string that can be mistyped or
/// outlive whatever owned it, so it is told that something happened, not what was said.
///
/// Cancellation is not an event: the client that canceled already knows, and nobody else needs
/// paging about a turn a human deliberately stopped.
fn notify_turn_end(
    webhooks: &crate::host::http::webhook::WebhookDispatcher,
    event_type: crate::host::http::sse::SseEventType,
    turn_id: Uuid,
    session_id: Uuid,
) {
    let event = match event_type {
        crate::host::http::sse::SseEventType::TurnFinished => {
            crate::host::http::webhook::WebhookEvent::TurnFinished
        }
        crate::host::http::sse::SseEventType::TurnFailed => {
            crate::host::http::webhook::WebhookEvent::TurnFailed
        }
        _ => return,
    };
    webhooks.send(
        event,
        serde_json::json!({
            "turn_id": turn_id,
            "session_id": session_id,
        }),
    );
}

/// Await the turn task and render whatever terminal it recorded, or synthesize one if it panicked
/// before it could.
async fn join_terminal(
    join: &mut std::pin::Pin<Box<tokio::task::JoinHandle<crate::host::http::sse::SseEvent>>>,
    turn_id: Uuid,
    session_id: Uuid,
) -> Event {
    match join.await {
        Ok(terminal) => terminal.into_axum(),
        Err(panic) => panic_terminal(panic, turn_id, session_id),
    }
}

/// The turn task panicked, so it never reached [`terminal_event_parts`]. Rendered straight to the
/// wire rather than recorded: with the task gone there is nothing left holding the frontend's
/// stream slot open for a reconnect to read.
fn panic_terminal(panic: tokio::task::JoinError, turn_id: Uuid, session_id: Uuid) -> Event {
    let (event_type, data) =
        // The flag is a don't-care here: a `JoinError` takes the panic arm, which renders a fixed
        // `/errors/internal` payload and never reaches an upstream message to relay or withhold.
        terminal_event_parts(
            Err(panic),
            CancelReason::Client,
            UsageView::default(),
            turn_id,
            session_id,
            false,
        );
    // Sent without an `id:` field. The generator lives on the task that just died, and id 0 is
    // already `turn.started`; reusing it would have a client store 0 as its resume position and
    // replay the whole turn on reconnect. An SSE event with no id leaves the client's stored
    // position untouched, which is the honest answer when the sequence has been abandoned.
    Event::default()
        .event(event_type.as_str())
        .json_data(data)
        .unwrap_or_else(|_| Event::default().comment("panic terminal serialize-failed"))
}

fn canceled_parts(
    reason: &'static str,
    turn_id: Uuid,
    session_id: Uuid,
) -> (crate::host::http::sse::SseEventType, serde_json::Value) {
    (
        crate::host::http::sse::SseEventType::TurnCanceled,
        serde_json::json!({
            "turn_id": turn_id.to_string(),
            "session_id": session_id.to_string(),
            "reason": reason,
        }),
    )
}

/// Wire `stop_reason` string for a finished turn. Shared by the blocking (`assemble_response`)
/// and streaming (`terminal_event_for_outcome`) paths so the two can't drift.
fn stop_reason_str(outcome: &TurnOutcome) -> &'static str {
    match outcome {
        TurnOutcome::EndTurn => "end_turn",
        TurnOutcome::MaxTokens => "max_tokens",
        TurnOutcome::Refusal(_) => "refusal",
    }
}

fn finished_parts(
    outcome: &TurnOutcome,
    usage: UsageView,
    turn_id: Uuid,
    session_id: Uuid,
) -> (crate::host::http::sse::SseEventType, serde_json::Value) {
    let stop_reason = stop_reason_str(outcome);
    let mut data = serde_json::json!({
        "turn_id": turn_id.to_string(),
        "session_id": session_id.to_string(),
        "stop_reason": stop_reason,
    });
    if let TurnOutcome::Refusal(text) = outcome
        && !text.is_empty()
        && let Some(obj) = data.as_object_mut()
    {
        obj.insert(
            "refusal_text".into(),
            serde_json::Value::String(text.clone()),
        );
    }
    // Always emit `usage` so clients don't have to handle a conditionally-absent field.
    if let Some(obj) = data.as_object_mut()
        && let Ok(value) = serde_json::to_value(&usage)
    {
        obj.insert("usage".into(), value);
    }
    (crate::host::http::sse::SseEventType::TurnFinished, data)
}

/// Drain the per-session recorder at end-of-turn and pluck the most recent `TokenUsage` event
/// off the back. Mirrors what `run_blocking_turn` does explicitly via `entry.frontend.drain()`.
/// Both transport branches reset the recorder so the next turn starts clean. Returns `None`
/// when the turn never reported usage (mock provider tests, refused turns, server-shutdown
/// cancel before the agent emitted anything).
///
/// Only one of the two select-arm callers ever runs per turn (terminal events break the loop),
/// so the drain happens exactly once.
fn drain_recorder_and_extract_usage(
    frontend: &Arc<crate::host::http::http_frontend::HttpFrontend>,
) -> UsageView {
    let recorder = frontend.drain();
    recorder
        .into_iter()
        .rev()
        .find_map(|event| {
            if let FrontendEvent::TokenUsage(usage) = event {
                Some(UsageView {
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    cache_creation_input_tokens: usage.cache_creation_input_tokens,
                    cache_read_input_tokens: usage.cache_read_input_tokens,
                })
            } else {
                None
            }
        })
        .unwrap_or_default()
}

fn assemble_response(
    turn_id: Uuid,
    session_id: Uuid,
    outcome: TurnOutcome,
    recorder: Recorder,
    capabilities: crate::host::http::http_frontend::SessionCapabilities,
) -> TurnResponse {
    let stop_reason = stop_reason_str(&outcome).to_string();

    let mut final_text = String::new();
    let mut tool_calls_by_id: std::collections::HashMap<String, ToolCallView> =
        std::collections::HashMap::new();
    let mut tool_call_order: Vec<String> = Vec::new();
    let mut usage = UsageView::default();
    let mut notices: Vec<NoticeView> = Vec::new();
    let mut thinking_segments: Vec<String> = Vec::new();

    for event in recorder {
        match event {
            FrontendEvent::AssistantTextDelta(text) => {
                final_text.push_str(&text);
            }
            FrontendEvent::ThinkingBlock { content, .. }
                if capabilities.supports_reasoning_stream =>
            {
                thinking_segments.push(content);
            }
            // The block above already carries this text whole. Accumulating the deltas as well
            // would report every segment twice.
            FrontendEvent::ThinkingDelta(_) => {}
            FrontendEvent::ToolCallStarted {
                id,
                name,
                input,
                display_summary,
            } => {
                tool_call_order.push(id.clone());
                tool_calls_by_id.insert(id.clone(), ToolCallView {
                    id,
                    name,
                    input,
                    display_summary,
                    is_error: false,
                    content: Vec::new(),
                });
            }
            FrontendEvent::ToolCallCompleted {
                id,
                is_error,
                content,
                ..
            } => {
                if let Some(view) = tool_calls_by_id.get_mut(&id) {
                    view.is_error = is_error;
                    view.content = content
                        .into_iter()
                        .map(|item| match item {
                            ToolResultContent::Text { text } => ToolCallContentView::Text { text },
                            ToolResultContent::Image { source } => ToolCallContentView::Image {
                                media_type: source.media_type().to_string(),
                            },
                        })
                        .collect();
                }
            }
            FrontendEvent::TokenUsage(token_usage) => {
                // Last-wins assignment: the agent emits exactly one `TokenUsage` per turn
                // (accumulated total). If that ever changes, switch to `saturating_add`.
                usage.input_tokens = token_usage.input_tokens;
                usage.output_tokens = token_usage.output_tokens;
                usage.cache_creation_input_tokens = token_usage.cache_creation_input_tokens;
                usage.cache_read_input_tokens = token_usage.cache_read_input_tokens;
            }
            FrontendEvent::Notice(notice) => {
                notices.push(NoticeView::from(notice));
            }
            // Remaining lifecycle / UI-chrome variants (TurnStarted/Finished, TodoListUpdated,
            // McpProgress, SessionStarted, ToolCallOutputDelta, SubAgentActivity) aren't part of
            // the blocking JSON envelope.
            // ToolCallComposing does reach here -- `stream: false` picks the response shape, not
            // whether meka streams from the provider -- and is dropped: a body assembled after the
            // turn is over has nothing to mark the beginning of a wait on.
            // Compacted is dropped too: the summary message on `GET /messages` carries the same
            // `compaction` marker, so a blocking client already has the fact where it will look
            // for it, and a streaming one gets `context.compacted`.
            _ => {}
        }
    }

    // Mark orphan tool calls (started but never completed) as errors so clients can
    // distinguish "tool returned nothing" from "interrupted mid-execution".
    for view in tool_calls_by_id.values_mut() {
        if !view.is_error && view.content.is_empty() {
            view.is_error = true;
            view.content = vec![ToolCallContentView::Text {
                text: "tool execution interrupted before completion".to_string(),
            }];
        }
    }

    let refusal_text = match &outcome {
        TurnOutcome::Refusal(text) if !text.is_empty() => Some(text.clone()),
        _ => None,
    };

    let tool_calls = tool_call_order
        .into_iter()
        .filter_map(|id| tool_calls_by_id.remove(&id))
        .collect();

    let mut content_blocks: Vec<crate::host::http::handlers::messages::ContentBlockView> =
        Vec::new();
    for segment in thinking_segments {
        content_blocks.push(
            crate::host::http::handlers::messages::ContentBlockView::Thinking { thinking: segment },
        );
    }
    if !final_text.is_empty() {
        content_blocks.push(
            crate::host::http::handlers::messages::ContentBlockView::Text {
                text: final_text.clone(),
            },
        );
    }
    let messages = if content_blocks.is_empty() {
        Vec::new()
    } else {
        vec![crate::host::http::handlers::messages::MessageView {
            role: "assistant".to_string(),
            content: content_blocks,
            // Not available yet: the DB write may still be in progress.
            created_at: None,
            // Only the current message is available; full history index lives on
            // `GET /v1/sessions/{id}/messages`.
            turn_id: None,
            // This is the assistant's reply, never a compaction summary. A compaction that fired
            // during this turn is reported by the `context.compacted` SSE event and by the marker
            // on the summary when the history is read back.
            compaction: None,
        }]
    };

    TurnResponse {
        turn_id,
        session_id,
        stop_reason,
        final_text,
        refusal_text,
        messages,
        tool_calls,
        usage,
        notices,
    }
}

/// `POST /v1/sessions/{id}/cancel`: interrupt the in-flight turn (if any) by firing the
/// session's cancellation token. Always returns 204 even if no turn is in flight (the
/// operation is idempotent and absence is observationally indistinguishable from a turn that
/// finished microseconds before the cancel arrived).
#[utoipa::path(
    post,
    path = "/v1/sessions/{id}/cancel",
    tag = "turn",
    params(("id" = Uuid, Path, description = "Session UUID")),
    responses(
        (status = 204, description = "Cancellation token fired (idempotent)"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:w"]))
)]
pub(crate) async fn cancel_turn(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SessionsWrite>,
    Path(session_id): Path<Uuid>,
) -> Result<StatusCode, ProblemDetail> {
    // Fast-path: look up the in-memory session map directly. If the session was GC-evicted
    // (no in-memory entry), there's no in-flight turn to cancel. Return 204 idempotently
    // instead of re-attaching from disk (which would build an unconnected cancellation token
    // and waste a file-lock + DB load).
    let entry = state.sessions.read().await.get(&session_id).cloned();

    if let Some(entry) = entry {
        entry.cancel.cancel();
    }
    // 204 whether or not there was anything to cancel: POST /cancel is idempotent.
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::{FrontendEvent, Notice};

    /// One consumer falling behind must not end the turn for a second one that is keeping up.
    /// The lagging receiver is already dropped by the time the decision runs, so the count it
    /// sees is of the *other* readers.
    #[tokio::test]
    async fn a_lagging_consumer_does_not_cancel_a_turn_someone_else_is_reading() {
        let frontend = crate::host::http::http_frontend::HttpFrontend::new();
        let (_turn_consumer, _ids) = frontend.install_stream(
            16,
            16,
            std::time::Duration::from_secs(1),
            Uuid::from_u128(0xfeed),
        );
        let _reattached = frontend
            .attach_stream(None)
            .expect("a live stream accepts a re-attach");

        let cancellation = CancellationToken::new();
        let canceled = cancel_if_nobody_else_is_reading(&frontend, &cancellation);
        assert!(
            !cancellation.is_cancelled(),
            "the turn was canceled out from under a consumer that was keeping up"
        );
        assert!(
            !canceled,
            "and the caller must be told so, or it reports a turn.failed for a turn still running"
        );
    }

    /// The other half: when the consumer that lagged was the only one, there is nobody left to
    /// deliver to, so the turn should stop rather than keep spending provider tokens.
    #[tokio::test]
    async fn a_lagging_consumer_that_was_the_only_reader_cancels_the_turn() {
        let frontend = crate::host::http::http_frontend::HttpFrontend::new();
        let (_turn_consumer, _ids) = frontend.install_stream(
            16,
            16,
            std::time::Duration::from_secs(1),
            Uuid::from_u128(0xfeed),
        );

        let cancellation = CancellationToken::new();
        let canceled = cancel_if_nobody_else_is_reading(&frontend, &cancellation);
        assert!(
            cancellation.is_cancelled(),
            "nobody is reading, so the turn should not keep running"
        );
        assert!(
            canceled,
            "and the caller must be told so, or it withholds the turn.failed the client needs"
        );
    }

    /// A lagging consumer is told what happened in a shape a client already parses.
    ///
    /// The canceled half is a `turn.failed` whose `error` is the cataloged `sse-lag` problem
    /// rather than a hand-written object that can drift from it. The other half is a `notice` in
    /// the `level`/`text` shape every other `notice` event carries, which is the shape the docs
    /// promise.
    #[test]
    fn a_lag_ends_the_stream_with_a_cataloged_failure_or_a_shaped_notice() {
        let turn_id = Uuid::from_u128(1);
        let session_id = Uuid::from_u128(2);

        let (event_type, data) = lag_event_parts(true, 7, turn_id, session_id);
        assert_eq!(event_type, crate::host::http::sse::SseEventType::TurnFailed);
        assert_eq!(
            data["error"]["type"],
            crate::host::http::errors::ErrorKind::SseLag.type_uri()
        );
        assert_eq!(
            data["error"]["title"],
            crate::host::http::errors::ErrorKind::SseLag.title()
        );
        assert_eq!(data["error"]["status"], 500);
        assert!(
            data["error"]["detail"]
                .as_str()
                .unwrap_or_default()
                .contains("7 event(s)"),
            "{data}"
        );
        assert_eq!(data["turn_id"], turn_id.to_string());
        assert_eq!(data["session_id"], session_id.to_string());

        let (event_type, data) = lag_event_parts(false, 3, turn_id, session_id);
        assert_eq!(event_type, crate::host::http::sse::SseEventType::Notice);
        assert_eq!(data["level"], "warn", "{data}");
        let text = data["text"].as_str().unwrap_or_default();
        assert!(
            text.contains("3 event(s)") && text.contains("Last-Event-ID"),
            "{data}"
        );
        assert!(
            data.get("notice").is_none(),
            "the undocumented shape must not survive beside the documented one: {data}"
        );
        assert_eq!(data["turn_id"], turn_id.to_string());
        assert_eq!(data["session_id"], session_id.to_string());
    }

    #[test]
    fn assemble_response_concatenates_text_deltas() {
        let recorder: Recorder = vec![
            FrontendEvent::AssistantTextDelta("Hello ".into()),
            FrontendEvent::AssistantTextDelta("world".into()),
        ];
        let response = assemble_response(
            Uuid::nil(),
            Uuid::nil(),
            TurnOutcome::EndTurn,
            recorder,
            crate::host::http::http_frontend::SessionCapabilities::default(),
        );
        assert_eq!(response.final_text, "Hello world");
        assert_eq!(response.stop_reason, "end_turn");
        assert!(response.tool_calls.is_empty());
    }

    /// A tool call without a label omits `display_summary` from the blocking response rather than
    /// writing `null`, like every optional field on this API.
    #[test]
    fn assemble_response_omits_an_absent_display_summary() {
        let recorder: Recorder = vec![
            FrontendEvent::ToolCallStarted {
                id: "tu_1".into(),
                name: "todo_write".into(),
                input: serde_json::json!({}),
                display_summary: None,
            },
            FrontendEvent::ToolCallCompleted {
                id: "tu_1".into(),
                name: "todo_write".into(),
                is_error: false,
                content: vec![ToolResultContent::Text { text: "ok".into() }],
                metadata: None,
            },
        ];
        let response = assemble_response(
            Uuid::nil(),
            Uuid::nil(),
            TurnOutcome::EndTurn,
            recorder,
            crate::host::http::http_frontend::SessionCapabilities::default(),
        );
        let document = serde_json::to_value(&response).expect("serializes");
        assert!(
            document["tool_calls"][0].get("display_summary").is_none(),
            "an absent summary must be omitted, not null: {document}"
        );
    }

    #[test]
    fn assemble_response_pairs_tool_calls_with_completion() {
        let input = serde_json::json!({"path": "src/main.rs"});
        let recorder: Recorder = vec![
            FrontendEvent::ToolCallStarted {
                id: "tu_1".into(),
                name: "read_file".into(),
                input: input.clone(),
                display_summary: Some("src/main.rs".into()),
            },
            FrontendEvent::ToolCallCompleted {
                id: "tu_1".into(),
                name: "read_file".into(),
                is_error: false,
                content: vec![ToolResultContent::Text {
                    text: "fn main() {}".into(),
                }],
                metadata: None,
            },
        ];
        let response = assemble_response(
            Uuid::nil(),
            Uuid::nil(),
            TurnOutcome::EndTurn,
            recorder,
            crate::host::http::http_frontend::SessionCapabilities::default(),
        );
        assert_eq!(response.tool_calls.len(), 1);
        let call = &response.tool_calls[0];
        assert_eq!(call.id, "tu_1");
        assert_eq!(call.name, "read_file");
        assert_eq!(call.input, input);
        assert!(!call.is_error);
        match &call.content[0] {
            ToolCallContentView::Text { text } => assert_eq!(text, "fn main() {}"),
            other => panic!("expected text content, got {other:?}"),
        }
    }

    #[test]
    fn assemble_response_surfaces_notices() {
        let recorder: Recorder = vec![FrontendEvent::Notice(Notice::warn("auto-denied tool"))];
        let response = assemble_response(
            Uuid::nil(),
            Uuid::nil(),
            TurnOutcome::EndTurn,
            recorder,
            crate::host::http::http_frontend::SessionCapabilities::default(),
        );
        assert_eq!(response.notices.len(), 1);
        assert_eq!(response.notices[0].level, "warn");
        assert_eq!(response.notices[0].text, "auto-denied tool");
    }

    #[test]
    fn assemble_response_separates_refusal_text_from_final_text() {
        let recorder: Recorder = vec![FrontendEvent::AssistantTextDelta(
            "I cannot help with that.".into(),
        )];
        let response = assemble_response(
            Uuid::nil(),
            Uuid::nil(),
            TurnOutcome::Refusal("policy violation".into()),
            recorder,
            crate::host::http::http_frontend::SessionCapabilities::default(),
        );
        assert_eq!(response.stop_reason, "refusal");
        assert_eq!(response.final_text, "I cannot help with that.");
        assert_eq!(response.refusal_text.as_deref(), Some("policy violation"));
    }

    #[test]
    fn assemble_response_omits_refusal_text_on_normal_stop() {
        let recorder: Recorder = vec![FrontendEvent::AssistantTextDelta("hello".into())];
        let response = assemble_response(
            Uuid::nil(),
            Uuid::nil(),
            TurnOutcome::EndTurn,
            recorder,
            crate::host::http::http_frontend::SessionCapabilities::default(),
        );
        assert_eq!(response.refusal_text, None);
        assert_eq!(response.final_text, "hello");
    }

    fn png_input() -> ImageInput {
        use base64::Engine as _;
        let image = image::RgbaImage::from_pixel(4, 4, image::Rgba([10, 20, 30, 255]));
        let mut bytes = Vec::new();
        image
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .expect("encode png");
        ImageInput {
            media_type: "image/png".to_string(),
            data: base64::engine::general_purpose::STANDARD.encode(&bytes),
        }
    }

    #[tokio::test]
    async fn decode_turn_images_accepts_a_png() {
        let decoded = decode_turn_images(&[png_input()], true)
            .await
            .expect("should decode");
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].media_type(), "image/png");
        assert!(
            decoded[0]
                .base64_data()
                .is_some_and(|data| !data.is_empty())
        );
    }

    #[tokio::test]
    async fn decode_turn_images_is_a_noop_without_attachments() {
        // The vision flag is irrelevant when nothing is attached: a text-only profile must still
        // be able to take ordinary turns.
        assert!(
            decode_turn_images(&[], false)
                .await
                .expect("no images")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn decode_turn_images_rejects_attachments_when_vision_is_off() {
        let problem = decode_turn_images(&[png_input()], false)
            .await
            .expect_err("should reject");
        assert_eq!(problem.status, 422);
        assert!(problem.detail.unwrap_or_default().contains("vision"));
    }

    #[tokio::test]
    async fn decode_turn_images_rejects_invalid_base64() {
        let bad = ImageInput {
            media_type: "image/png".to_string(),
            data: "!!!not-base64!!!".to_string(),
        };
        let problem = decode_turn_images(&[bad], true)
            .await
            .expect_err("should reject");
        assert_eq!(problem.status, 422);
        assert!(problem.detail.unwrap_or_default().contains("images[0]"));
    }

    /// The offending index is named so a client sending several attachments knows which one to
    /// fix, rather than being told only that "an image" was bad.
    #[tokio::test]
    async fn decode_turn_images_names_the_failing_index() {
        use base64::Engine as _;
        let garbage = ImageInput {
            media_type: "application/octet-stream".to_string(),
            data: base64::engine::general_purpose::STANDARD.encode(b"not an image"),
        };
        let problem = decode_turn_images(&[png_input(), garbage], true)
            .await
            .expect_err("should reject");
        assert_eq!(problem.status, 422);
        let detail = problem.detail.unwrap_or_default();
        assert!(detail.contains("images[1]"), "{}", detail);
    }

    /// A declared MIME type that names no supported format still decodes when the payload's magic
    /// bytes do, so a client that labels its upload `application/octet-stream` isn't stuck.
    #[tokio::test]
    async fn decode_turn_images_falls_back_to_magic_bytes() {
        let mut input = png_input();
        input.media_type = "application/octet-stream".to_string();
        let decoded = decode_turn_images(&[input], true)
            .await
            .expect("should decode via magic bytes");
        assert_eq!(decoded[0].media_type(), "image/png");
    }

    #[tokio::test]
    async fn decode_turn_images_rejects_oversized_payloads() {
        use base64::Engine as _;
        let raw = vec![0u8; crate::image::MAX_IMAGE_RAW_BYTES + 1];
        let oversized = ImageInput {
            media_type: "image/png".to_string(),
            data: base64::engine::general_purpose::STANDARD.encode(&raw),
        };
        let problem = decode_turn_images(&[oversized], true)
            .await
            .expect_err("should reject");
        assert_eq!(problem.status, 422);
    }
}

#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub(crate) struct StreamQuery {
    /// Last event id the client received, for clients that cannot set a `Last-Event-ID` header
    /// (browser `EventSource` sets it automatically; `fetch`-based clients often cannot).
    /// The header wins when both are present.
    #[serde(default)]
    pub(crate) last_event_id: Option<u64>,
}

/// `GET /v1/sessions/{id}/stream`: rejoin the current turn's SSE stream.
///
/// Replays the events after `Last-Event-ID` from a bounded per-turn ring, then follows the live
/// stream. When the turn has already ended, the backlog plus its terminal event are delivered and
/// the connection closes, so a client that dropped at the last moment still learns the outcome.
///
/// Two limits worth stating plainly. The ring holds `[serve] stream_replay_events` events, so a
/// client that was away longer than that gets a `notice` saying its replay has a hole rather than a
/// transcript that silently skips. And only the most recent turn is retained: reconnecting after a
/// *newer* turn has started returns that turn's stream, which the `turn_id` on the re-issued
/// `turn.started` identifies.
#[utoipa::path(
    get,
    path = "/v1/sessions/{id}/stream",
    tag = "turn",
    params(
        ("id" = Uuid, Path, description = "Session UUID"),
        ("Last-Event-ID" = Option<String>, Header, description = "Resume after this event id"),
        StreamQuery,
    ),
    responses(
        (status = 200, description = "SSE stream (text/event-stream)"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found, or no turn has streamed on it", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:r"]))
)]
pub(crate) async fn stream_turn(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SessionsRead>,
    Path(id): Path<Uuid>,
    Query(query): Query<StreamQuery>,
    headers: axum::http::HeaderMap,
) -> Result<axum::response::Response, ProblemDetail> {
    // Deliberately the in-memory map rather than `ensure_session_loaded`: a re-attached session has
    // no turn stream by construction, so reviving one to discover that would be pure cost.
    let entry = state.sessions.read().await.get(&id).cloned();
    let Some(entry) = entry else {
        // Distinguish "unknown session" from "known but nothing to rejoin", because the fixes
        // differ: one is a bad id, the other means submit a turn.
        crate::host::http::reattach::require_session_exists(&state, id).await?;
        return Err(no_stream_to_join(id));
    };

    let last_event_id = headers
        .get("last-event-id")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .or(query.last_event_id);

    let Some(attachment) = entry.frontend.attach_stream(last_event_id) else {
        return Err(no_stream_to_join(id));
    };

    let session_id = id;
    let stream = build_reattach_stream(session_id, attachment, Arc::clone(&entry.frontend));
    let sse = Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(20))
            .text("keep-alive"),
    );
    let mut response = sse.into_response();
    response.headers_mut().insert(
        "X-Accel-Buffering",
        axum::http::HeaderValue::from_static("no"),
    );
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-cache, no-transform"),
    );
    Ok(response)
}

fn no_stream_to_join(id: Uuid) -> ProblemDetail {
    ProblemDetail::new(
        ErrorKind::NotFound,
        StatusCode::NOT_FOUND,
        "no turn stream to join on this session; submit a turn with `stream: true` first",
    )
    .with("session_id", id.to_string())
}

/// Backlog, then live events, then the terminal.
fn build_reattach_stream(
    session_id: Uuid,
    attachment: crate::host::http::http_frontend::StreamAttachment,
    frontend: Arc<crate::host::http::http_frontend::HttpFrontend>,
) -> impl Stream<Item = Result<Event, Infallible>> + Send {
    async_stream::stream! {
        yield Ok::<_, Infallible>(Event::default().retry(std::time::Duration::from_secs(3)));

        // Announced before anything else so a client can confirm *which* turn it rejoined. A
        // reconnect that lands after a newer turn started gets that turn's id here, which is the
        // only way to tell "my stream resumed" from "I am now watching something else".
        yield Ok(Event::default()
            .event("turn.started")
            .json_data(serde_json::json!({
                "turn_id": attachment.turn_id,
                "session_id": session_id,
                "resumed": true,
            }))
            .unwrap_or_else(|_| Event::default().comment("resumed turn.started serialize-failed")));

        if attachment.gap {
            // Said out loud rather than papered over. A transcript with a silent hole in it is
            // worse than one the client knows is incomplete, because only the second can be
            // repaired by reading `GET /messages`.
            yield Ok(Event::default()
                .event("notice")
                .json_data(serde_json::json!({
                    "level": "warn",
                    "text": "the replay does not reach your Last-Event-ID, so events were \
                             dropped; read `GET /v1/sessions/{id}/messages` for the full transcript",
                }))
                .unwrap_or_else(|_| Event::default().comment("gap-notice serialize-failed")));
        }

        // The terminal is in the backlog too when the turn has ended, since `record_terminal`
        // pushes it into the ring. Track it so we do not send it twice.
        let mut sent_terminal = false;
        for event in attachment.backlog {
            sent_terminal |= event.event_type.is_terminal();
            yield Ok(event.into_axum());
        }

        if let Some(mut receiver) = attachment.receiver {
            loop {
                match receiver.recv().await {
                    Ok(event) => {
                        sent_terminal |= event.event_type.is_terminal();
                        yield Ok(event.into_axum());
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                        tracing::warn!(
                            "re-attached SSE consumer lagged, skipped {skipped} events"
                        );
                        // Unlike the primary stream's lag branch, this does not cancel the turn:
                        // the original consumer may still be reading it perfectly well, and one
                        // slow observer should not kill work someone else is watching.
                        yield Ok(Event::default()
                            .event("notice")
                            .json_data(serde_json::json!({
                                "level": "warn",
                                "text": format!(
                                    "Fell behind; {} event(s) were dropped from this replay.",
                                    skipped
                                ),
                            }))
                            .unwrap_or_else(|_| Event::default().comment("lag serialize-failed")));
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        }

        if !sent_terminal {
            // Re-read rather than reusing the snapshot: a client that attached mid-turn captured
            // `terminal: None`, because the turn had not ended yet. The terminal is recorded but
            // not broadcast (see `record_terminal`), so asking again is the only way to get it.
            // Scoped to the turn we attached to, so a turn that started in the meantime cannot
            // hand us its terminal instead.
            let recorded = frontend.recorded_terminal(attachment.turn_id);
            // Filtered by the resume position like every other replayed event. A client whose last
            // id *is* the terminal has already seen the turn end, and re-sending it would break
            // the one promise resumption makes -- that nothing at or before your position comes
            // back -- on the single event a client is most likely to act on twice.
            let terminal = recorded
                .clone()
                .or(attachment.terminal)
                .filter(|terminal| {
                    attachment
                        .resume_from
                        .is_none_or(|last| terminal.id > last)
                });
            match terminal {
                Some(terminal) => yield Ok(terminal.into_axum()),
                // Nothing to send: the client already holds the terminal, its `resume_from`
                // covering it. Close cleanly rather than inventing an event.
                None if recorded.is_some() => {}
                None => {
                    yield Ok(Event::default()
                        .event("turn.failed")
                        .json_data(serde_json::json!({
                            "turn_id": attachment.turn_id.to_string(),
                            "session_id": session_id.to_string(),
                            "error": {
                                "type": crate::host::http::errors::ErrorKind::StreamDetached.type_uri(),
                                "title": crate::host::http::errors::ErrorKind::StreamDetached.title(),
                                "status": 500,
                                "detail": "the turn's stream closed without recording an outcome; \
                                           read `GET /v1/sessions/{id}/messages` for what completed",
                            },
                        }))
                        .unwrap_or_else(|_| Event::default().comment("detached serialize-failed")));
                }
            }
        }
    }
}
