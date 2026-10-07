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
    frontend::{FrontendEvent, NoticeView},
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
    /// profile to have vision enabled; see [`resolve_turn_images`].
    #[serde(default)]
    pub(crate) images: Vec<ImageInput>,
    /// `false` (default) → blocking JSON response. `true` → SSE.
    #[serde(default)]
    pub(crate) stream: bool,
    /// Per-turn knobs. See [`TurnOptions`]. Omitting the field is the same as `{}`.
    #[serde(default)]
    pub(crate) options: TurnOptions,
}

/// One image attachment: the bytes inline, or a reference to an image the session's history
/// holds. Bytes are base64 rather than a path or URL because the API is a network surface:
/// `[serve].bind` may be non-loopback, so the caller generally shares no filesystem with the agent
/// and cannot name a file for it to read. A reference is the `hash` an image block of
/// `GET /v1/sessions/{id}/messages` carries, so a client sends an image again, after an edit or a
/// rewind, without fetching and uploading its bytes.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImageInput {
    /// Declared MIME type, e.g. `image/png`, beside `data`. Used as the primary format hint; the
    /// payload's magic bytes win if this doesn't name a supported format. Refused beside `hash`,
    /// whose media type is the stored one.
    #[serde(default)]
    pub(crate) media_type: Option<String>,
    /// Base64-encoded image bytes (standard alphabet, padding required).
    #[serde(default)]
    pub(crate) data: Option<String>,
    /// The hash of an image the session's history holds, whose stored bytes are attached.
    #[serde(default)]
    pub(crate) hash: Option<String>,
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
    /// What becomes of `message` if the turn ends, failed or canceled, before anything from the
    /// model reached the conversation. `keep` (default) leaves it in place, as the REPL does for a
    /// prompt whose turn failed; `withdraw` takes it back, for a client that will resend it. Once
    /// a partial answer or a tool call is in the conversation, the message stays regardless.
    #[serde(default)]
    #[schema(value_type = String)]
    pub(crate) unanswered_message: crate::conversation::PromptRetention,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct TurnResponse {
    pub(crate) turn_id: Uuid,
    pub(crate) session_id: Uuid,
    pub(crate) stop_reason: String,
    /// Refusal explanation when `stop_reason == "refusal"`; `None` otherwise. Mirrors the
    /// `refusal_text` field on the streaming `turn.finished` SSE event, so blocking and
    /// streaming clients share the same shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) refusal_text: Option<String>,
    /// The messages this turn added to the conversation, in order and in the shape
    /// `GET /v1/sessions/{id}/messages` reads them back: the assistant's messages with their
    /// text, their thinking when the session streams reasoning, and their tool calls, and the
    /// tool-result messages that answered those calls. The user's own message is not repeated.
    pub(crate) messages: Vec<crate::host::http::handlers::messages::MessageView>,
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
    //
    // Scoped to that one question. The entry holds the session's file lock through its agent, and
    // a clone kept across the decode would keep the lock after the idle sweep evicted the session,
    // so the re-attach the admission below then needs would refuse this process's own claim with
    // `session-locked`.
    let accepts_images = {
        let resident = state.sessions.read().await.get(&session_id).cloned();
        let entry = match resident {
            Some(entry) => entry,
            None => ensure_session_loaded(&state, session_id).await?,
        };
        entry.accepts_images()
    };
    let image_count = body.images.len();
    let images = resolve_turn_images(
        &body.images,
        accepts_images,
        &state.shared.store,
        session_id,
    )
    .await?;
    // An image with no text passes; against prior context "look at this" is a complete request.
    //
    // The retention is stated before any outcome rides along: a carried outcome overrides the
    // client's answer with `Keep`, and the override has to be the last word.
    let input = crate::agent::TurnInput::from_parts(message, images)
        .map_err(|error| ProblemDetail::for_error(&error, state.config.relay_provider_errors))?
        .retaining(body.options.unanswered_message);

    // Taken only once the request is known to be one meka will act on. The guard marks the
    // session as busy, and a request rejected above never runs a turn, so acquiring first made a
    // malformed body answer 409 on the session's own PATCH, DELETE, compact and rewind for as long
    // as the guard lived, and answered 409 to a malformed body whenever a real turn was running.
    //
    // Under the sessions read-lock, and against the entry the map holds *now* rather than the one
    // resolved above: the image decoding awaited in between, and a DELETE landing in that window
    // has removed the entry, so admitting the earlier clone would run the turn against a row that
    // is gone and fail it on a foreign key after the provider had been asked. An entry missing
    // here is either that or an eviction; `ensure_session_loaded` tells the two apart, answering
    // 404 for a row that is gone and re-attaching the other, and the next pass finds it under the
    // lock. From admission on the lock does the rest: DELETE's write-lock blocks behind any
    // reader, so by the time it fires `in_flight > 0` and its re-check returns 409.
    #[cfg(any(debug_assertions, feature = "mock-provider"))]
    hold_before_admission(state.shared.config.mock_turn_hold.as_deref()).await;
    let (entry, turn_guard) = loop {
        {
            let map = state.sessions.read().await;
            if let Some(current) = map.get(&session_id) {
                let entry = current.clone();
                let turn_guard = crate::host::http::state::admit_turn(&state, &entry)?;
                break (entry, turn_guard);
            }
        }
        ensure_session_loaded(&state, session_id).await?;
    };

    // Taken here rather than inside the two arms below, and *before* the claim: it is the only
    // per-session admission gate (`TurnGuard::acquire` counts, it does not refuse), so claiming
    // first meant a 409 stamped an outcome delivered that no turn ever delivered, and
    // `list_undelivered_background_tasks` never returns a stamped row again. The guard moves into
    // whichever arm runs, holding from admission to the end of the turn.
    let conversation = Arc::clone(&entry.conversation)
        .try_lock_owned()
        .map_err(|_| turn_in_flight_conflict(session_id, "run a turn"))?;

    // Under the lock and before the first round, the way ACP's prompt door does it: the turn runs
    // on the profile the row names, or not at all.
    if let Err(error) =
        crate::host::apply_recorded_profile(&state.shared, &entry.agent, session_id).await
    {
        return Err(crate::host::http::reattach::agent_build_problem(
            session_id,
            "cannot run this turn on the profile this session is recorded against",
            error,
        ));
    }

    // Asked again of the entry admitted, not only of the one the decode was judged against: the
    // session may have been evicted and moved onto a text-only profile in between, and the
    // images would otherwise reach a model the row says cannot take them. Ahead of the outcome
    // claim below, which is the first thing here a refusal could not undo.
    refuse_images_without_vision(image_count, entry.accepts_images())?;

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

/// The refusal for attachments on a session whose profile does not accept them. Asked twice of a
/// turn that carries any: of the entry resolved before the decode, so a text-only profile is
/// refused without decoding anything, and of the entry admitted after it, which may run on
/// another profile by then (the session evicted and moved with `PATCH` in between), so what the
/// decode was told is never the last word on what the model is sent.
fn refuse_images_without_vision(images: usize, vision: bool) -> Result<(), ProblemDetail> {
    if images > 0 && !vision {
        return Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "image attachments require a profile with vision enabled; set `vision = true` under \
             `[profiles.<name>]`",
        ));
    }
    Ok(())
}

/// A test's hand on the gap between a turn's validation and its admission. Only where the mock
/// provider exists, and only when `MEKA_MOCK_TURN_HOLD` named a path: the handler marks that it
/// has reached the gap by creating that path with `.waiting` appended, then waits until the path
/// itself exists. A test can then act on the session (delete it, let the idle sweep evict it)
/// while a turn stands in exactly this window, which no amount of image decoding can guarantee in
/// a build where decoding is fast. Bounded, so a test that never releases the hold fails on its
/// own clock rather than hanging the server.
#[cfg(any(debug_assertions, feature = "mock-provider"))]
async fn hold_before_admission(hold: Option<&std::path::Path>) {
    let Some(path) = hold else {
        return;
    };
    if let Err(error) = tokio::fs::write(path.with_extension("waiting"), b"").await {
        tracing::warn!("MEKA_MOCK_TURN_HOLD: failed to mark the hold: {error}");
    }
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
    while !path.exists() && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
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

/// The terminal a stream ends with when its consumer fell behind and the turn was canceled for
/// it, as `(event type, payload)`: a `turn.failed` whose `error` is the cataloged `sse-lag`
/// problem, so a retry is the remedy it invites. A consumer that fell behind while someone else
/// kept reading gets no terminal, since the turn goes on; it is caught up from the ring instead.
fn lag_terminal_parts(
    skipped: u64,
    turn_id: Uuid,
    session_id: Uuid,
) -> (crate::host::http::sse::SseEventType, serde_json::Value) {
    let problem = crate::host::http::errors::ProblemDetail::new(
        crate::host::http::errors::ErrorKind::SseLag,
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        format!(
            "SSE consumer fell behind and {skipped} event(s) were dropped, so the turn was \
             canceled; retry it"
        ),
    )
    .instance(format!("/v1/sessions/{session_id}/turn"));
    (
        crate::host::http::sse::SseEventType::TurnFailed,
        serde_json::json!({
            "turn_id": turn_id.to_string(),
            "session_id": session_id.to_string(),
            "error": serde_json::to_value(problem).unwrap_or(serde_json::Value::Null),
        }),
    )
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

/// What one entry of `images` asks for, once its shape is checked: bytes to decode, or an image
/// the session already holds.
enum ImageEntry {
    Upload { media_type: String, data: String },
    Reference { hash: String },
}

/// An entry is bytes with their media type, or a hash, and nothing else: the two forms the API
/// speaks, told apart by which fields are present so a refusal names what is wrong rather than
/// that nothing matched.
fn classify_image(index: usize, input: &ImageInput) -> Result<ImageEntry, ProblemDetail> {
    match (&input.media_type, &input.data, &input.hash) {
        (Some(media_type), Some(data), None) => Ok(ImageEntry::Upload {
            media_type: media_type.clone(),
            data: data.clone(),
        }),
        (None, None, Some(hash)) => Ok(ImageEntry::Reference { hash: hash.clone() }),
        (Some(_), None, Some(_)) => Err(invalid_image(
            index,
            "names a `hash`, which carries its own `media_type`",
        )),
        _ => Err(invalid_image(
            index,
            "is a `media_type` with `data`, or a `hash`",
        )),
    }
}

/// The 422 every bad entry of `images` answers with, naming the offender so a client sending
/// several attachments knows which one to fix.
fn invalid_image(index: usize, what: &str) -> ProblemDetail {
    ProblemDetail::new(
        ErrorKind::InvalidBody,
        StatusCode::UNPROCESSABLE_ENTITY,
        format!("`images[{index}]` {what}"),
    )
}

/// One entry after the blocking work, waiting only on the store.
enum PendingImage {
    Decoded(crate::image::ImageSource),
    Stored { hash: String },
}

/// Resolve the request's `images` into what the turn carries, in their order: uploaded bytes
/// decoded and validated through the shared image pipeline
/// ([`crate::image::decode_base64_image`]), so an HTTP attachment gets exactly the size cap and
/// format conversion an ACP `image` content block does, and a hash answered from the store.
///
/// `vision` is the session's answer from `ResidentSession::accepts_images`. Attachments are
/// refused outright when it is off, as ACP refuses `image` content blocks with `InvalidParams` for
/// a text-only profile. This gates on configuration only: whether the named model *actually*
/// understands images is left to the provider to complain about, which is the stance documented
/// for ACP.
///
/// Every failure is a 422: these are all malformed input, not server faults.
///
/// Off the runtime, for the reason `file_read` and `web_fetch` document at their own call sites:
/// the pipeline base64-decodes and then decodes each image to verify it, which is tens of
/// milliseconds of pure CPU on a multi-megapixel attachment, and on the runtime it blocks every
/// other task on that worker. One client posting a screenshot must not stall an unrelated
/// session's stream.
async fn resolve_turn_images(
    images: &[ImageInput],
    vision: bool,
    store: &crate::store::Store,
    session_id: Uuid,
) -> Result<Vec<crate::image::ImageSource>, ProblemDetail> {
    if images.is_empty() {
        return Ok(Vec::new());
    }
    refuse_images_without_vision(images.len(), vision)?;
    let entries = images
        .iter()
        .enumerate()
        .map(|(index, input)| classify_image(index, input))
        .collect::<Result<Vec<_>, _>>()?;
    let pending = tokio::task::spawn_blocking(move || {
        entries
            .into_iter()
            .enumerate()
            .map(|(index, entry)| match entry {
                ImageEntry::Upload { media_type, data } => {
                    crate::image::decode_base64_image(&data, &media_type)
                        .map(PendingImage::Decoded)
                        .map_err(|message| invalid_image(index, &format!("is invalid: {message}")))
                }
                ImageEntry::Reference { hash } => Ok(PendingImage::Stored { hash }),
            })
            .collect::<Result<Vec<_>, _>>()
    })
    .await
    .map_err(|error| {
        ProblemDetail::new(
            ErrorKind::Internal,
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to decode the images: {error}"),
        )
    })??;
    let mut resolved = Vec::with_capacity(pending.len());
    for (index, image) in pending.into_iter().enumerate() {
        resolved.push(match image {
            PendingImage::Decoded(source) => source,
            PendingImage::Stored { hash } => stored_image(store, session_id, index, &hash).await?,
        });
    }
    Ok(resolved)
}

/// The stored bytes behind a hash, as the inline form a turn carries, through the session-scoped
/// lookup `GET /v1/sessions/{id}/blobs/{hash}` serves, so a client attaches exactly what it can
/// read. The bytes were prepared when first stored, so they are not decoded again. A hash the
/// session holds no image for is refused like any other bad entry.
async fn stored_image(
    store: &crate::store::Store,
    session_id: Uuid,
    index: usize,
    hash: &str,
) -> Result<crate::image::ImageSource, ProblemDetail> {
    use base64::Engine as _;
    let blob = store
        .load_session_blob(session_id, hash)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to load an image", error))?
        .ok_or_else(|| invalid_image(index, "names a hash this session holds no image for"))?;
    Ok(crate::image::ImageSource::Base64 {
        media_type: blob.media_type,
        data: base64::engine::general_purpose::STANDARD.encode(&blob.bytes),
    })
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
pub(super) struct StreamGuard {
    frontend: Arc<crate::host::http::http_frontend::HttpFrontend>,
}

impl StreamGuard {
    pub(super) fn new(frontend: Arc<crate::host::http::http_frontend::HttpFrontend>) -> Self {
        Self { frontend }
    }
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        self.frontend.end_turn();
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
    let turn_id = uuid::Uuid::new_v4();
    let input = input.identified(turn_id);
    let published = entry
        .cancel
        .publish_turn(cancellation.clone(), turn_guard.admission, turn_id);
    // On the feed like every other turn, so a subscriber sees a blocking client's turn as it
    // runs. Unattended: the client waits on the response, not on a stream, so a feed with no
    // readers says nothing about whether anyone is still waiting.
    let (_feed, _ids) = entry.frontend.begin_turn(
        turn_id,
        crate::host::http::http_frontend::TurnSource::Client,
        false,
        state.config.stream_reattach_grace,
        state.config.stream_replay_events,
    );

    let join = tokio::spawn(async move {
        let _turn_guard = turn_guard;
        let _published = published;
        let _stream_guard = StreamGuard::new(Arc::clone(&entry.frontend));
        let outcome = entry
            .agent
            .run_turn(&mut conversation, input, cancellation)
            .await;

        let recorder = entry.frontend.drain();
        entry.touch();
        // The same terminal a streaming turn records, for the same readers.
        let cancel_reason = if state.shutdown.is_cancelled() {
            CancelReason::ServerShutdown
        } else {
            CancelReason::Client
        };
        let (event_type, data) = terminal_event_parts(
            Ok(outcome.as_ref()),
            cancel_reason,
            usage_from(&recorder),
            turn_id,
            session_id,
            state.config.relay_provider_errors,
            message_withdrawn(&recorder),
            revision_for_terminal(&state.shared.store, session_id).await,
        );
        entry.frontend.record_terminal(event_type, data);

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
                let problem = ProblemDetail::for_error(&error, state.config.relay_provider_errors);
                Err(match message_withdrawn(&recorder) {
                    Some(withdrawn) => problem.with("message_withdrawn", withdrawn),
                    None => problem,
                })
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
    let input = input.identified(turn_id);
    // Publish after the lock succeeds, same rationale as `run_blocking_turn`, and before the
    // stream announces the id: a client that answers `turn.started` with a cancel naming it
    // must find the turn there to cancel.
    let cancellation = CancellationToken::new();
    let published = entry
        .cancel
        .publish_turn(cancellation.clone(), turn_guard.admission, turn_id);
    let (receiver, ids) = entry.frontend.install_stream(
        SSE_BROADCAST_CAPACITY,
        state.config.stream_replay_events,
        state.config.stream_reattach_grace,
        turn_id,
    );
    // Where the stream stands before it has received anything: just ahead of the turn's own
    // `turn.started`, which was published after the receiver subscribed. Read now, while the live
    // turn is certainly this one, rather than at the stream's first poll.
    let joined_after = entry
        .frontend
        .turn_started_id()
        .and_then(|id| id.checked_sub(1));

    let entry_for_task = entry.clone();
    let cancel_for_task = cancellation.clone();
    let shutdown_for_task = state.shutdown.clone();
    let relay_for_task = state.config.relay_provider_errors;
    let store_for_task = state.shared.store.clone();
    let webhooks_for_task = state.webhooks;

    // Spawn the turn so the SSE response can return immediately.
    //
    // Declaration order is load-bearing: locals drop in reverse, so `_stream_guard` goes first,
    // then `conversation`, then `_turn_guard`. That is what keeps the conversation mutex held
    // across `end_turn()`, so a turn admitted the instant this one ends cannot install its
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
        let recorder = entry_for_task.frontend.drain();
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
            Ok(outcome.as_ref()),
            cancel_reason,
            usage_from(&recorder),
            turn_id,
            session_id,
            relay_for_task,
            message_withdrawn(&recorder),
            revision_for_terminal(&store_for_task, session_id).await,
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
        joined_after,
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

#[allow(
    clippy::too_many_arguments,
    reason = "every fact the stream is built from is a parameter, so no caller can leave one out"
)]
fn build_sse_stream(
    turn_id: Uuid,
    session_id: Uuid,
    mut receiver: tokio::sync::broadcast::Receiver<crate::host::http::sse::SseEvent>,
    joined_after: Option<u64>,
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

        // `turn.started` arrives off the feed: `begin_turn` published it after handing out this
        // receiver, so it is the first event here, numbered and in the ring like the rest. The
        // terminal arrives the same way, recorded and broadcast by the turn's task; the join
        // handle is the fallback for a task that died without recording one.
        let mut sent_terminal = false;
        let mut last_delivered = joined_after;
        let mut join = Box::pin(join);
        loop {
            tokio::select! {
                biased;
                event = receiver.recv() => {
                    match event {
                        Ok(sse) => {
                            let terminal = sse.event_type.is_terminal();
                            last_delivered = sse.id.or(last_delivered);
                            yield Ok(sse.into_axum());
                            if terminal {
                                break;
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                            // Stop burning provider tokens for a consumer that has fallen behind,
                            // but only when nobody else is still reading. For one that is, the
                            // turn goes on and this reader is caught up from the ring, the way a
                            // reconnect would be, so the lag is a pause and not a hole.
                            if cancel_if_nobody_else_is_reading(&frontend, &cancellation) {
                                tracing::warn!(
                                    "SSE consumer lagged, skipped {skipped} events; terminating stream"
                                );
                                let (event_type, data) =
                                    lag_terminal_parts(skipped, turn_id, session_id);
                                yield Ok(Event::default()
                                    .id(ids.next().to_string())
                                    .event(event_type.as_str())
                                    .json_data(data)
                                    .unwrap_or_else(|_| Event::default().comment("lag-failed serialize-failed")));
                                break;
                            }
                            tracing::warn!("SSE consumer lagged, skipped {skipped} events; catching up");
                            let Some(caught_up) = frontend.catch_up(last_delivered) else {
                                yield Ok(join_terminal(&mut join, turn_id, session_id).await);
                                break;
                            };
                            receiver = caught_up.receiver;
                            if let Some(gap) = caught_up.gap {
                                yield Ok(gap.event(Some(session_id)).into_axum());
                            }
                            let mut ended = false;
                            for event in caught_up.backlog {
                                ended = event.event_type.is_terminal();
                                last_delivered = event.id.or(last_delivered);
                                yield Ok(event.into_axum());
                                if ended {
                                    break;
                                }
                            }
                            if ended {
                                break;
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                            yield Ok(join_terminal(&mut join, turn_id, session_id).await);
                            break;
                        }
                    }
                }
                turn_result = &mut join => {
                    // Agent finished; flush remaining buffered events, the terminal among them
                    // when the task recorded one, and fall back to the join's copy otherwise.
                    while let Ok(sse) = receiver.try_recv() {
                        sent_terminal |= sse.event_type.is_terminal();
                        yield Ok(sse.into_axum());
                    }
                    if !sent_terminal {
                        yield Ok(match turn_result {
                            Ok(terminal) => terminal.into_axum(),
                            Err(panic) => panic_terminal(panic, turn_id, session_id),
                        });
                    }
                    break;
                }
            }
        }
    }
}

/// Why a turn that ended `Interrupted` was stopped, for the recorded `turn.canceled` event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CancelReason {
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
///
/// `message_withdrawn` rides the failed and canceled terminals of a turn that began: it is the one
/// fact a client that resends needs, and the stream cannot be read for it, since thinking and a
/// half-composed tool call look like output and neither reaches the conversation. `None` is a turn
/// that never began, or one whose record died with its task, and is omitted rather than guessed.
#[allow(
    clippy::too_many_arguments,
    reason = "every fact a terminal carries is a parameter, so no emitter can leave one out"
)]
pub(crate) fn terminal_event_parts(
    turn_result: std::result::Result<
        std::result::Result<&TurnOutcome, &crate::error::MekaError>,
        tokio::task::JoinError,
    >,
    cancel_reason: CancelReason,
    usage: UsageView,
    turn_id: Uuid,
    session_id: Uuid,
    relay_provider_errors: bool,
    message_withdrawn: Option<bool>,
    revision: Option<u64>,
) -> (crate::host::http::sse::SseEventType, serde_json::Value) {
    let (event_type, mut data) = match turn_result {
        Ok(Ok(outcome)) => finished_parts(outcome, usage, turn_id, session_id),
        Ok(Err(crate::error::MekaError::Interrupted)) => {
            // Every stop surfaces as `Interrupted` by the time the agent loop unwinds; who asked
            // for it is carried alongside.
            canceled_parts(
                cancel_reason.as_str(),
                message_withdrawn,
                turn_id,
                session_id,
            )
        }
        Ok(Err(error)) => {
            let instance = format!("/v1/sessions/{session_id}/turn");
            let problem =
                crate::host::http::errors::ProblemDetail::for_error(error, relay_provider_errors)
                    .instance(instance);
            let mut data = serde_json::json!({
                "turn_id": turn_id.to_string(),
                "session_id": session_id.to_string(),
                "error": serde_json::to_value(problem).unwrap_or(serde_json::Value::Null),
            });
            // Omitted, not `null`, when there is nothing to say: the rule every optional field on
            // this API follows.
            if let Some(withdrawn) = message_withdrawn {
                data["message_withdrawn"] = serde_json::Value::Bool(withdrawn);
            }
            (crate::host::http::sse::SseEventType::TurnFailed, data)
        }
        Err(panic) => {
            tracing::error!("streaming turn task panicked: {panic:?}");
            let problem = ProblemDetail::new(
                ErrorKind::Internal,
                StatusCode::INTERNAL_SERVER_ERROR,
                "turn task panicked",
            )
            .instance(format!("/v1/sessions/{session_id}/turn"));
            (
                crate::host::http::sse::SseEventType::TurnFailed,
                serde_json::json!({
                    "turn_id": turn_id.to_string(),
                    "session_id": session_id.to_string(),
                    "error": serde_json::to_value(problem).unwrap_or(serde_json::Value::Null),
                }),
            )
        }
    };
    // On every terminal, because a repair, a redaction or a withdrawal rewrites the view inside
    // the turn and has no event of its own: the terminal is where a client decides what to
    // re-read. Omitted, like every optional field, when the store could not say.
    if let Some(revision) = revision {
        data["revision"] = serde_json::Value::from(revision);
    }
    (event_type, data)
}

/// The conversation's revision for a terminal event, or `None` when the store could not say; a
/// client then re-reads as it does for any event that lacks it. Never fails the turn: the terminal
/// still has to be recorded.
pub(crate) async fn revision_for_terminal(
    store: &crate::store::Store,
    session_id: Uuid,
) -> Option<u64> {
    match store.count_rewrites(session_id).await {
        Ok(revision) => Some(revision),
        Err(error) => {
            tracing::warn!("failed to count rewrites for session {session_id}: {error}");
            None
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
pub(crate) fn notify_turn_end(
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
        // The relay flag is a don't-care here: a `JoinError` takes the panic arm, which renders a
        // fixed `/errors/internal` payload and never reaches an upstream message to relay or
        // withhold. The record of the turn died with its task, so nothing is said about the
        // message either.
        terminal_event_parts(
            Err(panic),
            CancelReason::Client,
            UsageView::default(),
            turn_id,
            session_id,
            false,
            None,
            None,
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
    message_withdrawn: Option<bool>,
    turn_id: Uuid,
    session_id: Uuid,
) -> (crate::host::http::sse::SseEventType, serde_json::Value) {
    let mut data = serde_json::json!({
        "turn_id": turn_id.to_string(),
        "session_id": session_id.to_string(),
        "reason": reason,
    });
    if let Some(withdrawn) = message_withdrawn {
        data["message_withdrawn"] = serde_json::Value::Bool(withdrawn);
    }
    (crate::host::http::sse::SseEventType::TurnCanceled, data)
}

/// Wire `stop_reason` string for a finished turn. Shared by the blocking (`assemble_response`)
/// and streaming (`terminal_event_for_outcome`) paths so the two can't drift.
fn finished_parts(
    outcome: &TurnOutcome,
    usage: UsageView,
    turn_id: Uuid,
    session_id: Uuid,
) -> (crate::host::http::sse::SseEventType, serde_json::Value) {
    let stop_reason = outcome.stop_reason();
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

/// The turn's usage, from the events it recorded. The default when the turn never reported any
/// (mock provider tests, refused turns, a server-shutdown cancel before the agent emitted
/// anything).
pub(crate) fn usage_from(recorder: &Recorder) -> UsageView {
    recorder
        .iter()
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

/// Whether the turn took its message back, from the events it recorded. `None` for a turn refused
/// before it began, which never added the message and so has nothing to say about it.
///
/// Read from the recorder rather than kept as a flag on the frontend, because the recorder is
/// drained before the turn and again after it, so it holds exactly this turn's events. A flag
/// cleared on `TurnStarted` would answer for the previous turn when this one was refused ahead of
/// that event, as the required-MCP gate refuses; here that refusal reads as no `TurnStarted` at
/// all.
pub(crate) fn message_withdrawn(recorder: &Recorder) -> Option<bool> {
    recorder
        .iter()
        .any(|event| matches!(event, FrontendEvent::TurnStarted { .. }))
        .then(|| {
            recorder
                .iter()
                .any(|event| matches!(event, FrontendEvent::PromptWithdrawn))
        })
}

/// The messages a turn adds to the conversation, built from the recorder in the order the agent
/// emitted them. An assistant message gathers text, thinking and tool calls until a tool result
/// arrives, which closes it and opens the user-role message that carries the round's results;
/// the next assistant output closes that one in turn.
struct TurnMessages {
    turn_id: Uuid,
    messages: Vec<crate::host::http::handlers::messages::MessageView>,
    assistant: Vec<crate::host::http::handlers::messages::ContentBlockView>,
    results: Vec<crate::host::http::handlers::messages::ContentBlockView>,
}

impl TurnMessages {
    fn new(turn_id: Uuid) -> Self {
        Self {
            turn_id,
            messages: Vec::new(),
            assistant: Vec::new(),
            results: Vec::new(),
        }
    }

    fn text(&mut self, text: &str) {
        use crate::host::http::handlers::messages::ContentBlockView;
        self.close_results();
        if let Some(ContentBlockView::Text { text: last }) = self.assistant.last_mut() {
            last.push_str(text);
        } else {
            self.assistant.push(ContentBlockView::Text {
                text: text.to_string(),
            });
        }
    }

    fn thinking(&mut self, thinking: String) {
        self.close_results();
        self.assistant
            .push(crate::host::http::handlers::messages::ContentBlockView::Thinking { thinking });
    }

    fn tool_use(&mut self, id: String, name: String, input: serde_json::Value) {
        self.close_results();
        self.assistant.push(
            crate::host::http::handlers::messages::ContentBlockView::ToolUse { id, name, input },
        );
    }

    fn tool_result(&mut self, tool_use_id: String, is_error: bool, content: &[ToolResultContent]) {
        self.close_assistant();
        self.results.push(
            crate::host::http::handlers::messages::ContentBlockView::ToolResult {
                tool_use_id,
                is_error,
                content: crate::host::http::handlers::messages::tool_result_views(content),
            },
        );
    }

    /// A nudge meka wrote closes the assistant message it answers and is a user-role message of
    /// its own, as `GET /messages` shows it; the reply after it opens a new assistant message.
    fn nudge(&mut self, kind: crate::conversation::NudgeKind, text: String) {
        self.close_assistant();
        self.close_results();
        self.messages.push(turn_message(self.turn_id, "user", vec![
            crate::host::http::handlers::messages::ContentBlockView::Nudge {
                kind: kind.name().to_string(),
                text,
            },
        ]));
    }

    fn close_assistant(&mut self) {
        if !self.assistant.is_empty() {
            let content = std::mem::take(&mut self.assistant);
            self.messages
                .push(turn_message(self.turn_id, "assistant", content));
        }
    }

    fn close_results(&mut self) {
        if !self.results.is_empty() {
            let content = std::mem::take(&mut self.results);
            self.messages
                .push(turn_message(self.turn_id, "user", content));
        }
    }

    fn finish(mut self) -> Vec<crate::host::http::handlers::messages::MessageView> {
        self.close_assistant();
        self.close_results();
        self.messages
    }
}

fn turn_message(
    turn_id: Uuid,
    role: &str,
    content: Vec<crate::host::http::handlers::messages::ContentBlockView>,
) -> crate::host::http::handlers::messages::MessageView {
    crate::host::http::handlers::messages::MessageView {
        role: role.to_string(),
        content,
        // Not available yet: the DB write may still be in progress.
        created_at: None,
        turn_id: Some(turn_id),
        // Only this turn is in hand; the label counts in the view `GET /v1/sessions/{id}/messages`
        // serves.
        turn_label: None,
        // The turn's own output, never a compaction summary. A compaction that fired during this
        // turn is reported by the `context.compacted` SSE event and by the marker on the summary
        // when the history is read back.
        compaction: None,
    }
}

fn assemble_response(
    turn_id: Uuid,
    session_id: Uuid,
    outcome: TurnOutcome,
    recorder: Recorder,
    capabilities: crate::host::http::http_frontend::SessionCapabilities,
) -> TurnResponse {
    let stop_reason = outcome.stop_reason().to_string();

    let mut messages = TurnMessages::new(turn_id);
    let mut usage = UsageView::default();
    let mut notices: Vec<NoticeView> = Vec::new();

    for event in recorder {
        match event {
            FrontendEvent::AssistantTextDelta(text) => messages.text(&text),
            FrontendEvent::ThinkingBlock { content } if capabilities.supports_reasoning_stream => {
                messages.thinking(content);
            }
            // The block above already carries this text whole. Accumulating the deltas as well
            // would report every segment twice.
            FrontendEvent::ThinkingDelta(_) => {}
            FrontendEvent::ToolCallStarted {
                id, name, input, ..
            } => messages.tool_use(id, name, input),
            FrontendEvent::ToolCallCompleted {
                id,
                is_error,
                content,
                ..
            } => messages.tool_result(id, is_error, &content),
            FrontendEvent::Nudged { kind, text } => messages.nudge(kind, text),
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
            // Remaining lifecycle / UI-chrome variants (TurnStarted/Finished, ChecklistUpdated,
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

    let refusal_text = match &outcome {
        TurnOutcome::Refusal(text) if !text.is_empty() => Some(text.clone()),
        _ => None,
    };

    TurnResponse {
        turn_id,
        session_id,
        stop_reason,
        refusal_text,
        messages: messages.finish(),
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
    request_body(content = CancelRequest, description = "Optional: the turn to cancel"),
    responses(
        (status = 204, description = "Cancellation token fired (idempotent)"),
        (status = 409, description = "The named turn is not the one in flight", body = ProblemDetail),
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
    raw_body: Bytes,
) -> Result<StatusCode, ProblemDetail> {
    // An empty body keeps the old contract: cancel whatever is running. A body names a turn, for
    // a client that watched one and must not stop the fire or the inbox turn that replaced it.
    let turn_id = if raw_body.is_empty() {
        None
    } else {
        serde_json::from_slice::<CancelRequest>(&raw_body)
            .map_err(|error| ProblemDetail::invalid_body("cancel", error))?
            .turn_id
    };
    // Fast-path: look up the in-memory session map directly. If the session was GC-evicted
    // (no in-memory entry), there's no in-flight turn to cancel. Return 204 idempotently
    // instead of re-attaching from disk (which would build an unconnected cancellation token
    // and waste a file-lock + DB load).
    let entry = state.sessions.read().await.get(&session_id).cloned();

    if let Some(entry) = entry {
        match turn_id {
            None => {
                entry.cancel.cancel();
            }
            Some(turn_id) => match entry.cancel.cancel_turn(turn_id) {
                crate::host::CancelOutcome::Canceled | crate::host::CancelOutcome::NoTurn => {}
                crate::host::CancelOutcome::Mismatch => {
                    let mut problem = ProblemDetail::new(
                        ErrorKind::TurnMismatch,
                        StatusCode::CONFLICT,
                        format!("turn {turn_id} is not in flight on session {session_id}"),
                    )
                    .with("session_id", session_id.to_string());
                    if let Some(live) = entry.cancel.live_turn_id() {
                        problem = problem.with("turn_id", live.to_string());
                    }
                    return Err(problem);
                }
            },
        }
    }
    // 204 whether or not there was anything to cancel: POST /cancel is idempotent.
    Ok(StatusCode::NO_CONTENT)
}

/// The optional body of `POST /v1/sessions/{id}/cancel`.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CancelRequest {
    /// Cancel only if this turn is the one in flight; another answers 409 `turn-mismatch`.
    #[serde(default)]
    pub(crate) turn_id: Option<Uuid>,
}

#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub(crate) struct StreamQuery {
    /// Last event id the client received, for clients that cannot set a `Last-Event-ID` header
    /// (browser `EventSource` sets it automatically; `fetch`-based clients often cannot).
    /// The header wins when both are present.
    #[serde(default)]
    pub(crate) last_event_id: Option<u64>,
    /// Attend the session: show `permission_required` events and answer them. While an attending
    /// reader is connected, a gated call on any turn parks for an answer instead of being refused
    /// without asking. Needs `sessions:w`, the scope that answers.
    #[serde(default)]
    pub(crate) attend: bool,
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
///
/// A sub-agent's id names its own feed for as long as this process runs it: the same events a
/// session's feed carries, read-only, with `turn.started` naming the parent and its `agent_spawn`
/// call. Its prompts park on the parent's feed, so `attend` is refused here.
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
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 409, description = "Another meka process holds the session (`/errors/session-locked`), the token may only read and the session is not loaded (`/errors/session-not-loaded`), or the id names a sub-agent this process is not running (`/errors/subagent-not-running`)", body = ProblemDetail),
        (status = 422, description = "`attend` on a sub-agent's feed, which is read-only (`/errors/session-not-drivable`)", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:r"]))
)]
pub(crate) async fn stream_turn(
    State(state): State<ServerState>,
    scoped: scope::Scoped<scope::SessionsRead>,
    Path(id): Path<Uuid>,
    Query(query): Query<StreamQuery>,
    headers: axum::http::HeaderMap,
) -> Result<axum::response::Response, ProblemDetail> {
    // Attending is answering, so it takes the scope answering takes; refused before the session
    // is loaded, so a token that may not attend loads nothing by asking.
    if query.attend {
        scope::require(&scoped.principal, "sessions:w")?;
    }
    let last_event_id = last_event_id_of(&headers, query.last_event_id);
    // A sub-agent's feed exists for as long as this process runs it, and is read rather than
    // attended: its prompts are parked on its parent's feed, where the one answerer is.
    let child = crate::sync::lock(&state.child_feeds).get(&id).cloned();
    if let Some(feed) = child {
        if query.attend {
            return Err(ProblemDetail::new(
                ErrorKind::SessionNotDrivable,
                StatusCode::UNPROCESSABLE_ENTITY,
                format!(
                    "session '{id}' is a sub-agent, whose prompts are answered on its parent's \
                     feed; read this one without `attend`"
                ),
            )
            .with("session_id", id.to_string()));
        }
        let Some(attachment) = feed.attach_stream(last_event_id, false) else {
            return Err(no_stream_to_join(id));
        };
        let stream = build_reattach_stream(id, attachment, feed, state.shutdown.clone());
        return Ok(sse_response(stream));
    }
    if let Some(terms) = state
        .shared
        .store
        .spawn_terms(id)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to read session", error))?
    {
        return Err(ProblemDetail::new(
            ErrorKind::SubagentNotRunning,
            StatusCode::CONFLICT,
            format!(
                "{}, which is not running in this process; its feed exists while its parent runs \
                 it",
                terms.describe(id)
            ),
        )
        .with("session_id", id.to_string()));
    }
    // Loaded for a token that may drive the session, looked up for one that may only read it.
    // Reviving takes the session's cross-process file lock and pins it in memory for as long as
    // the stream stays open, since the GC never evicts a session with a subscriber; a read token
    // that could do that to every session it lists would hold them all against `meka -r`. A
    // driver gets its feed back after an eviction, which is what a bridge reconnecting wants
    // rather than a 404 that tells it to run a turn it has no message for; a reader watches a
    // session somebody else is driving.
    let entry = if scoped.principal.has_scope("sessions:w") {
        ensure_session_loaded(&state, id).await?
    } else {
        let resident = state.sessions.read().await.get(&id).cloned();
        match resident {
            Some(entry) => entry,
            None => {
                crate::host::http::reattach::require_session_exists(&state, id).await?;
                return Err(ProblemDetail::new(
                    ErrorKind::SessionNotLoaded,
                    StatusCode::CONFLICT,
                    "session is not loaded; a token with `sessions:w` loads it by attaching or by \
                     submitting a turn",
                )
                .with("session_id", id.to_string()));
            }
        }
    };

    let Some(attachment) = entry.frontend.attach_stream(last_event_id, query.attend) else {
        return Err(no_stream_to_join(id));
    };
    let stream = build_reattach_stream(
        id,
        attachment,
        Arc::clone(&entry.frontend),
        state.shutdown.clone(),
    );
    Ok(sse_response(stream))
}

/// The id a client resumes from: the `Last-Event-ID` header, or the query parameter for a
/// transport that cannot send one.
fn last_event_id_of(headers: &HeaderMap, query: Option<u64>) -> Option<u64> {
    headers
        .get("last-event-id")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .or(query)
}

/// An event stream as every feed answers with it: a keep-alive comment every twenty seconds and
/// the headers that keep a proxy from buffering it.
fn sse_response(
    stream: impl Stream<Item = Result<Event, Infallible>> + Send + 'static,
) -> Response {
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
    response
}

fn no_stream_to_join(id: Uuid) -> ProblemDetail {
    ProblemDetail::new(
        ErrorKind::NotFound,
        StatusCode::NOT_FOUND,
        "no event feed on this session; it is installed when the session is loaded",
    )
    .with("session_id", id.to_string())
}

/// Backlog, then the feed, live: this stream does not end with a turn.
///
/// A client that named a `Last-Event-ID` gets what it missed first. A turn in flight is announced
/// so the client can tell "my stream resumed" from "I am now watching something else"; with no
/// turn in flight, the most recent turn's terminal is handed over when the ring no longer holds
/// it, so a client that reconnects late still learns the outcome. Then the feed carries every
/// later turn, whoever starts it, until the client hangs up or the session leaves this process.
#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub(crate) struct ServerStreamQuery {
    /// The last event id received, for a client whose transport cannot send the
    /// `Last-Event-ID` header.
    #[serde(default)]
    pub(crate) last_event_id: Option<u64>,
}

/// `GET /v1/stream`: the server feed, the listing's change feed across every session this process
/// holds; see [`crate::host::http::feed::ServerFeed`] for what it carries.
#[utoipa::path(
    get,
    path = "/v1/stream",
    tag = "sessions",
    params(
        ServerStreamQuery,
        ("Last-Event-ID" = Option<u64>, Header, description = "Resume from this id; the ring replays what followed it"),
    ),
    responses(
        (status = 200, description = "Server-sent events: `session.created`, `session.updated`, `session.deleted`, the four turn lifecycle events and the two permission events, each naming its session", content_type = "text/event-stream"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:r"]))
)]
pub(crate) async fn stream_server(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SessionsRead>,
    Query(query): Query<ServerStreamQuery>,
    headers: axum::http::HeaderMap,
) -> Result<axum::response::Response, ProblemDetail> {
    let last_event_id = last_event_id_of(&headers, query.last_event_id);
    let attachment = crate::sync::lock(&state.server_feed).attach(last_event_id);
    let shutdown = state.shutdown;
    let server_feed = Arc::clone(&state.server_feed);
    let stream = async_stream::stream! {
        yield Ok::<_, Infallible>(Event::default().retry(std::time::Duration::from_secs(3)));
        if let Some(gap) = attachment.gap {
            yield Ok(gap.event(None).into_axum());
        }
        for event in attachment.backlog {
            yield Ok(event.into_axum());
        }
        let mut last_delivered = attachment.joined_after;
        let mut receiver = attachment.receiver;
        loop {
            let received = tokio::select! {
                received = receiver.recv() => received,
                _ = shutdown.cancelled() => break,
            };
            match received {
                Ok(event) => {
                    last_delivered = event.id.or(last_delivered);
                    yield Ok(event.into_axum());
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                    // Caught up from the ring, the way a reconnect would be, so the lag is a
                    // pause and not a hole; a hole is said only where the ring cannot reach.
                    tracing::warn!(
                        "server feed SSE consumer lagged, skipped {skipped} events; catching up"
                    );
                    let caught_up = crate::sync::lock(&server_feed).attach(last_delivered);
                    receiver = caught_up.receiver;
                    if let Some(gap) = caught_up.gap {
                        yield Ok(gap.event(None).into_axum());
                    }
                    for event in caught_up.backlog {
                        last_delivered = event.id.or(last_delivered);
                        yield Ok(event.into_axum());
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    };
    Ok(sse_response(stream))
}

fn build_reattach_stream(
    session_id: Uuid,
    attachment: crate::host::http::feed::StreamAttachment,
    frontend: Arc<crate::host::http::http_frontend::HttpFrontend>,
    shutdown: CancellationToken,
) -> impl Stream<Item = Result<Event, Infallible>> + Send {
    async_stream::stream! {
        // Held for the life of the response: the client attends until it hangs up.
        let _attendance = attachment.attendance;
        yield Ok::<_, Infallible>(Event::default().retry(std::time::Duration::from_secs(3)));

        if let Some(turn_id) = attachment.turn_id {
            let mut data = serde_json::json!({
                "turn_id": turn_id,
                "session_id": session_id,
                "resumed": true,
            });
            if let Some(source) = &attachment.turn_source {
                source.describe(&mut data);
            }
            yield Ok(Event::default()
                .event("turn.started")
                .json_data(data)
                .unwrap_or_else(|_| Event::default().comment("resumed turn.started serialize-failed")));
        }

        if let Some(gap) = attachment.gap {
            // Said out loud rather than papered over. A transcript with a silent hole in it is
            // worse than one the client knows is incomplete, because only the second can be
            // repaired by reading `GET /messages`.
            yield Ok(gap.event(Some(session_id)).into_axum());
        }

        // The terminal is in the backlog too when the turn has ended, since `record_terminal`
        // pushes it into the ring. Track it so the fallback below does not send it twice.
        let mut sent_terminal = false;
        for event in attachment.backlog {
            sent_terminal |= event.event_type.is_terminal();
            yield Ok(event.into_axum());
        }

        // Filtered by the resume position like every other replayed event. A client whose last id
        // *is* the terminal has already seen the turn end, and re-sending it would break the one
        // promise resumption makes -- that nothing at or before your position comes back -- on the
        // single event a client is most likely to act on twice.
        if !sent_terminal
            && let Some(terminal) = attachment.terminal.filter(|terminal| {
                attachment
                    .resume_from
                    .is_none_or(|last| terminal.id.is_some_and(|id| id > last))
            })
        {
            yield Ok(terminal.into_axum());
        }

        let mut last_delivered = attachment.joined_after;
        let mut receiver = attachment.receiver;
        loop {
            // The feed outlives every turn, and this response holds the frontend that owns it, so
            // nothing closes the channel from the sending side while the session is resident: the
            // process ending is watched here, or a bridge attached across a restart would hold
            // axum's graceful shutdown to the drain timeout and the process would leave through
            // `exit(1)`. A session leaving the process closes the feed and ends this from the
            // other side.
            let received = tokio::select! {
                received = receiver.recv() => received,
                _ = shutdown.cancelled() => break,
            };
            match received {
                Ok(event) => {
                    sent_terminal |= event.event_type.is_terminal();
                    last_delivered = event.id.or(last_delivered);
                    yield Ok(event.into_axum());
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                    // Caught up from the ring, the way a reconnect would be, so the lag is a
                    // pause and not a hole; a hole is said only where the ring cannot reach.
                    // Nothing is canceled for a feed reader: the turn was not run for it.
                    tracing::warn!("feed SSE consumer lagged, skipped {skipped} events; catching up");
                    let Some(caught_up) = frontend.catch_up(last_delivered) else {
                        break;
                    };
                    receiver = caught_up.receiver;
                    if let Some(gap) = caught_up.gap {
                        yield Ok(gap.event(Some(session_id)).into_axum());
                    }
                    for event in caught_up.backlog {
                        sent_terminal |= event.event_type.is_terminal();
                        last_delivered = event.id.or(last_delivered);
                        yield Ok(event.into_axum());
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }

        // The feed closed under a turn this client was following, which happens when the session
        // leaves the process. Its outcome, if the task recorded one, or an honest failure.
        if let Some(turn_id) = attachment.turn_id
            && !sent_terminal
        {
            match frontend.recorded_terminal(turn_id) {
                Some(terminal) => yield Ok(terminal.into_axum()),
                None => {
                    yield Ok(Event::default()
                        .event("turn.failed")
                        .json_data(serde_json::json!({
                            "turn_id": turn_id.to_string(),
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
            .attach_stream(None, false)
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
    /// The terminal a lag that canceled the turn ends the stream with is a `turn.failed` whose
    /// `error` is the cataloged `sse-lag` problem rather than a hand-written object that can drift
    /// from it.
    #[test]
    fn a_lag_that_cancels_the_turn_ends_the_stream_with_the_cataloged_failure() {
        let turn_id = Uuid::from_u128(1);
        let session_id = Uuid::from_u128(2);

        let (event_type, data) = lag_terminal_parts(7, turn_id, session_id);
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
    }

    fn text_of(message: &crate::host::http::handlers::messages::MessageView) -> String {
        message
            .content
            .iter()
            .filter_map(|block| match block {
                crate::host::http::handlers::messages::ContentBlockView::Text { text } => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect()
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
        assert_eq!(response.stop_reason, "end_turn");
        assert_eq!(response.messages.len(), 1);
        assert_eq!(response.messages[0].role, "assistant");
        assert_eq!(text_of(&response.messages[0]), "Hello world");
    }

    /// A tool round reads back as the history does: the call on the assistant's message, its
    /// result on the user-role message after it, and the next reply on a message of its own, so a
    /// client renders one shape for a turn it ran and for a turn it reads back.
    #[test]
    fn assemble_response_lays_a_tool_round_out_as_the_history_does() {
        let input = serde_json::json!({"path": "src/main.rs"});
        let recorder: Recorder = vec![
            FrontendEvent::AssistantTextDelta("Reading.".into()),
            FrontendEvent::ToolCallStarted {
                id: "tu_1".into(),
                name: "file_read".into(),
                input: input.clone(),
                display_summary: Some("src/main.rs".into()),
            },
            FrontendEvent::ToolCallCompleted {
                id: "tu_1".into(),
                name: "file_read".into(),
                is_error: false,
                content: vec![ToolResultContent::Text {
                    text: "fn main() {}".into(),
                }],
                metadata: None,
            },
            FrontendEvent::AssistantTextDelta("Done.".into()),
        ];
        let response = assemble_response(
            Uuid::nil(),
            Uuid::nil(),
            TurnOutcome::EndTurn,
            recorder,
            crate::host::http::http_frontend::SessionCapabilities::default(),
        );
        let document = serde_json::to_value(&response).expect("serializes");
        let nil = Uuid::nil().to_string();
        assert_eq!(
            document["messages"],
            serde_json::json!([
                {"role": "assistant", "turn_id": nil, "content": [
                    {"type": "text", "text": "Reading."},
                    {"type": "tool_use", "id": "tu_1", "name": "file_read", "input": input},
                ]},
                {"role": "user", "turn_id": nil, "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "is_error": false,
                     "content": [{"type": "text", "text": "fn main() {}"}]},
                ]},
                {"role": "assistant", "turn_id": nil, "content": [{"type": "text", "text": "Done."}]},
            ]),
            "{document}"
        );
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
    fn assemble_response_separates_refusal_text_from_the_reply() {
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
        assert_eq!(text_of(&response.messages[0]), "I cannot help with that.");
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
        assert_eq!(text_of(&response.messages[0]), "hello");
    }

    /// [`resolve_turn_images`] over a store with nothing in it, which is all an upload needs.
    async fn resolve(
        images: &[ImageInput],
        vision: bool,
    ) -> Result<Vec<crate::image::ImageSource>, ProblemDetail> {
        let dir = tempfile::tempdir().expect("tempdir");
        let store =
            crate::store::Store::open(Some(&dir.path().join("meka.db")), &Default::default())
                .await
                .expect("store");
        resolve_turn_images(images, vision, &store, Uuid::from_u128(0x1)).await
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
            media_type: Some("image/png".to_string()),
            data: Some(base64::engine::general_purpose::STANDARD.encode(&bytes)),
            hash: None,
        }
    }

    #[tokio::test]
    async fn resolve_turn_images_accepts_a_png() {
        let decoded = resolve(&[png_input()], true).await.expect("should decode");
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].media_type(), "image/png");
        assert!(
            decoded[0]
                .base64_data()
                .is_some_and(|data| !data.is_empty())
        );
    }

    #[tokio::test]
    async fn resolve_turn_images_is_a_noop_without_attachments() {
        // The vision flag is irrelevant when nothing is attached: a text-only profile must still
        // be able to take ordinary turns.
        assert!(resolve(&[], false).await.expect("no images").is_empty());
    }

    #[tokio::test]
    async fn resolve_turn_images_rejects_attachments_when_vision_is_off() {
        let problem = resolve(&[png_input()], false)
            .await
            .expect_err("should reject");
        assert_eq!(problem.status, 422);
        assert!(problem.detail.unwrap_or_default().contains("vision"));
    }

    #[tokio::test]
    async fn resolve_turn_images_rejects_invalid_base64() {
        let bad = ImageInput {
            media_type: Some("image/png".to_string()),
            data: Some("!!!not-base64!!!".to_string()),
            hash: None,
        };
        let problem = resolve(&[bad], true).await.expect_err("should reject");
        assert_eq!(problem.status, 422);
        assert!(problem.detail.unwrap_or_default().contains("images[0]"));
    }

    /// The offending index is named so a client sending several attachments knows which one to
    /// fix, rather than being told only that "an image" was bad.
    #[tokio::test]
    async fn resolve_turn_images_names_the_failing_index() {
        use base64::Engine as _;
        let garbage = ImageInput {
            media_type: Some("application/octet-stream".to_string()),
            data: Some(base64::engine::general_purpose::STANDARD.encode(b"not an image")),
            hash: None,
        };
        let problem = resolve(&[png_input(), garbage], true)
            .await
            .expect_err("should reject");
        assert_eq!(problem.status, 422);
        let detail = problem.detail.unwrap_or_default();
        assert!(detail.contains("images[1]"), "{}", detail);
    }

    /// A declared MIME type that names no supported format still decodes when the payload's magic
    /// bytes do, so a client that labels its upload `application/octet-stream` isn't stuck.
    /// An entry is one of the two forms the API speaks, and a refusal names the entry and what is
    /// wrong with it rather than that nothing matched.
    #[tokio::test]
    async fn an_image_entry_is_bytes_with_a_media_type_or_a_hash() {
        let bare_data = ImageInput {
            media_type: None,
            data: Some("AAAA".to_string()),
            hash: None,
        };
        let problem = resolve(&[bare_data], true)
            .await
            .expect_err("bytes without their media type");
        assert_eq!(problem.status, 422);
        assert!(
            problem
                .detail
                .as_deref()
                .is_some_and(|detail| detail
                    .contains("`images[0]` is a `media_type` with `data`, or a `hash`")),
            "{problem:?}"
        );

        let typed_hash = ImageInput {
            media_type: Some("image/png".to_string()),
            data: None,
            hash: Some("abc".to_string()),
        };
        let problem = resolve(&[png_input(), typed_hash], true)
            .await
            .expect_err("a hash carries its own media type");
        assert!(
            problem
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("`images[1]` names a `hash`")),
            "{problem:?}"
        );
    }

    /// A hash the session holds no image for is refused at the same door as a bad upload, naming
    /// the entry, before anything reaches the model.
    #[tokio::test]
    async fn a_hash_the_session_holds_no_image_for_is_refused() {
        let unknown = ImageInput {
            media_type: None,
            data: None,
            hash: Some("0".repeat(64)),
        };
        let problem = resolve(&[png_input(), unknown], true)
            .await
            .expect_err("refused");
        assert_eq!(problem.status, 422);
        assert!(
            problem.detail.as_deref().is_some_and(|detail| {
                detail.contains("`images[1]` names a hash this session holds no image for")
            }),
            "{problem:?}"
        );
    }

    #[tokio::test]
    async fn resolve_turn_images_falls_back_to_magic_bytes() {
        let mut input = png_input();
        input.media_type = Some("application/octet-stream".to_string());
        let decoded = resolve(&[input], true)
            .await
            .expect("should decode via magic bytes");
        assert_eq!(decoded[0].media_type(), "image/png");
    }

    #[tokio::test]
    async fn resolve_turn_images_rejects_oversized_payloads() {
        use base64::Engine as _;
        let raw = vec![0u8; crate::image::MAX_IMAGE_RAW_BYTES + 1];
        let oversized = ImageInput {
            media_type: Some("image/png".to_string()),
            data: Some(base64::engine::general_purpose::STANDARD.encode(&raw)),
            hash: None,
        };
        let problem = resolve(&[oversized], true)
            .await
            .expect_err("should reject");
        assert_eq!(problem.status, 422);
    }
}
