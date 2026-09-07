//! Conversation history: `GET /v1/sessions/{id}/messages`.
//!
//! Returns the materialized `Conversation` view (post-compaction-aware) for clients that want
//! to read past turns. Pagination via `?limit=` and `?offset=` is intentionally simple; the
//! source of truth is the SQLite event log, which holds the full history.

use axum::{
    Json,
    extract::{Path, Query, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use crate::{
    conversation::{ContentBlock, Message, Role, ToolResultContent},
    host::http::{
        errors::{ErrorKind, ProblemDetail},
        scope,
        state::ServerState,
    },
};

#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub(crate) struct MessagesQuery {
    #[serde(default)]
    pub(crate) limit: Option<usize>,
    #[serde(default)]
    pub(crate) offset: Option<usize>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct MessagesResponse {
    pub(crate) session_id: Uuid,
    pub(crate) messages: Vec<MessageView>,
    pub(crate) total: usize,
    /// How many times this conversation has been rewritten rather than appended to.
    ///
    /// Increments on every compaction, rewind, and mid-turn repair. A polling client that sees it
    /// change knows its copy is no longer a prefix of the server's and must re-fetch rather than
    /// diff; `total` alone cannot tell it apart from data loss.
    ///
    /// The per-message `compaction` marker explains one of those three. This covers the other two:
    /// a rewind removes messages with nothing left behind to attach a marker to, which would
    /// otherwise reproduce exactly the silent-rewrite failure the marker was added to prevent.
    pub(crate) revision: u64,
}

/// Lightweight wire view of a `Message`. Strips provider-internal blocks like `Thinking`
/// signatures down to their textual content, since callers consuming this endpoint are
/// typically just rendering a transcript.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct MessageView {
    pub(crate) role: String,
    pub(crate) content: Vec<ContentBlockView>,
    /// RFC 3339 timestamp at which this message was persisted. `None` for messages produced by
    /// `assemble_response` in the same turn (no DB round-trip yet). Once the conversation has
    /// been read back via `GET /v1/sessions/{id}/messages`, every row has a `created_at`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) created_at: Option<String>,
    /// Virtual per-conversation turn correlator (`t_001`, `t_002`, …). Derived at query time
    /// by grouping every user-role message into a new turn that includes the assistant +
    /// tool-result messages that follow it. `None` on messages from the assembled-response
    /// path (no turn boundary known yet).
    ///
    /// Note: these are dense sequential indexes (`t_0001`, not UUIDs). The UUID-shaped
    /// `turn_id` on `POST /v1/sessions/{id}/turn` is a different identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) turn_id: Option<String>,
    /// Present only on a message that *is* a compaction summary.
    ///
    /// Without this a client polling `/messages` watches history rewrite itself: a compaction
    /// truncates the materialised tail and pushes a summary in its place, so `total` shrinks and
    /// messages the client already rendered stop coming back. The marker is what lets it tell
    /// "the window was summarized" from "the server lost my conversation".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) compaction: Option<CompactionMarker>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub(crate) struct CompactionMarker {
    /// How many materialised messages the boundary removed.
    ///
    /// The whole pre-compaction window, including the recent tail that compaction then re-appends
    /// verbatim, so it over-counts what the summary itself stands for. Use it to detect *that* the
    /// view was rewritten, not to compute how much was lost.
    pub(crate) replaced_count: usize,
    /// Which compaction produced it, counting from 1. Fidelity degrades with each pass, so a
    /// client rendering a transcript can say how far from the original it is.
    pub(crate) generation: u64,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ContentBlockView {
    Text {
        text: String,
    },
    /// What meka injected ahead of the user's words for that turn (permission and environment
    /// context, todos, catalog changes, background outcomes, the resume notice), which the model
    /// saw as text ahead of them. Typed so a client can show or hide it; `text` alone is the words.
    TurnContext {
        text: String,
    },
    Image {
        // Signal an input image was present without the base64 payload, mirroring
        // `ToolResultContentView::Image`, so history responses stay tractable.
        media_type: String,
        /// Content hash of the stored bytes; `GET /v1/sessions/{id}/blobs/{hash}` returns them.
        /// Absent only for an image the store never externalized.
        #[serde(skip_serializing_if = "Option::is_none")]
        hash: Option<String>,
    },
    Thinking {
        thinking: String,
    },
    /// Encrypted reasoning whose contents the API withholds; surfaced as a presence marker so
    /// history responses note it occurred without exposing the opaque payload.
    RedactedThinking {},
    ToolUse {
        id: String,
        name: String,
        #[schema(value_type = Object)]
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        is_error: bool,
        content: Vec<ToolResultContentView>,
    },
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ToolResultContentView {
    Text {
        text: String,
    },
    Image {
        // Just signal that an image was present; clients fetching the JSON history don't get
        // the full base64 payload to keep responses tractable.
        media_type: String,
        /// Content hash of the stored bytes; `GET /v1/sessions/{id}/blobs/{hash}` returns them.
        #[serde(skip_serializing_if = "Option::is_none")]
        hash: Option<String>,
    },
}

/// The hash an image view names, when the source is a reference into the store.
fn blob_hash(source: &crate::image::ImageSource) -> Option<String> {
    match source {
        crate::image::ImageSource::Blob { hash, .. } => Some(hash.clone()),
        crate::image::ImageSource::Base64 { .. } => None,
    }
}

#[utoipa::path(
    get,
    path = "/v1/sessions/{id}/messages",
    tag = "sessions",
    params(
        ("id" = Uuid, Path, description = "Session UUID"),
        MessagesQuery,
    ),
    responses(
        (status = 200, description = "Page of conversation messages", body = MessagesResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:r"]))
)]
pub(crate) async fn list_messages(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SessionsRead>,
    Path(id): Path<Uuid>,
    Query(query): Query<MessagesQuery>,
) -> Result<Json<MessagesResponse>, ProblemDetail> {
    let events_with_ts = state
        .shared
        .store
        .load_events_with_timestamps(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to load session events", error)
        })?;
    if events_with_ts.is_empty() {
        // No events → either empty session or unknown. Distinguish via `session_exists`.
        // Propagate DB failure as 500 so the client retries instead of assuming 404.
        let exists = state
            .shared
            .store
            .session_exists(id)
            .await
            .map_err(|error| {
                ProblemDetail::internal_sanitized("failed to verify session existence", error)
            })?;
        if !exists {
            return Err(crate::host::http::reattach::session_not_found(id));
        }
    }
    // The same replay the model's view goes through, so a repair or a boundary reads the same on
    // the wire as it does in the window; a second copy of the rules had already drifted from it.
    let crate::conversation::AnnotatedView {
        messages: materialized,
        timestamps,
        markers,
        revision,
    } = crate::conversation::materialize_annotated(&events_with_ts);
    let total = materialized.len();
    let offset = query.offset.unwrap_or(0).min(total);
    let limit = query.limit.unwrap_or(200).min(1000);
    let end = (offset + limit).min(total);
    // Derive virtual turn indexes: every user-role message opens a new turn. Computed
    // across the *full* materialized view (not just the page) so paging doesn't shift the
    // correlator.
    let turn_indexes = derive_turn_indexes(&materialized);
    let messages = materialized[offset..end]
        .iter()
        .zip(timestamps[offset..end].iter())
        .zip(turn_indexes[offset..end].iter())
        .zip(markers[offset..end].iter())
        .map(|(((message, timestamp), turn_index), marker)| MessageView {
            role: match message.role {
                Role::User => "user".to_string(),
                Role::Assistant => "assistant".to_string(),
            },
            content: message.content.iter().map(view_for_block).collect(),
            created_at: Some(timestamp.clone()),
            turn_id: Some(turn_index.clone()),
            compaction: marker.as_ref().map(|marker| CompactionMarker {
                replaced_count: marker.replaced_count,
                generation: marker.generation,
            }),
        })
        .collect();
    Ok(Json(MessagesResponse {
        session_id: id,
        messages,
        total,
        revision,
    }))
}

/// Group materialized messages into virtual turns. Every user-role message opens a new turn;
/// subsequent assistant + tool-result messages belong to that turn. Returns a parallel `Vec`
/// the same length as `messages` with `t_NNNN` indexes.
fn derive_turn_indexes(messages: &[Message]) -> Vec<String> {
    let mut counter: u32 = 0;
    let mut current = String::from("t_0001"); // placeholder for messages before the first user message
    let mut ids = Vec::with_capacity(messages.len());
    for message in messages {
        if matches!(message.role, Role::User) {
            counter = counter.saturating_add(1);
            current = format!("t_{counter:04}");
        }
        ids.push(current.clone());
    }
    ids
}

fn view_for_block(block: &ContentBlock) -> ContentBlockView {
    match block {
        ContentBlock::Text { text } => ContentBlockView::Text { text: text.clone() },
        ContentBlock::TurnContext { text } => ContentBlockView::TurnContext { text: text.clone() },
        ContentBlock::Image { source } => ContentBlockView::Image {
            media_type: source.media_type().to_string(),
            hash: blob_hash(source),
        },
        ContentBlock::Thinking { thinking, .. } => ContentBlockView::Thinking {
            thinking: thinking.clone(),
        },
        ContentBlock::RedactedThinking { .. } => ContentBlockView::RedactedThinking {},
        ContentBlock::ToolUse { id, name, input } => ContentBlockView::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
        },
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => ContentBlockView::ToolResult {
            tool_use_id: tool_use_id.clone(),
            is_error: *is_error,
            content: content
                .iter()
                .map(|item| match item {
                    ToolResultContent::Text { text } => {
                        ToolResultContentView::Text { text: text.clone() }
                    }
                    ToolResultContent::Image { source } => ToolResultContentView::Image {
                        media_type: source.media_type().to_string(),
                        hash: blob_hash(source),
                    },
                })
                .collect(),
        },
    }
}

/// `GET /v1/sessions/{id}/blobs/{hash}`: the bytes behind an image block, with their media type.
///
/// Scoped to the session: the blob has to be referenced by one of this session's messages, so a
/// token that may read one session cannot walk every image in the store by hash.
#[utoipa::path(
    get,
    path = "/v1/sessions/{id}/blobs/{hash}",
    tag = "sessions",
    params(
        ("id" = Uuid, Path, description = "Session UUID"),
        ("hash" = String, Path, description = "Content hash of an image block, as `GET /v1/sessions/{id}/messages` reports it"),
    ),
    responses(
        (status = 200, description = "The image bytes, under their media type", content_type = "image/*"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No message of this session references such a blob", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:r"]))
)]
pub(crate) async fn get_blob(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SessionsRead>,
    Path((id, hash)): Path<(Uuid, String)>,
) -> Result<Response, ProblemDetail> {
    let blob = state
        .shared
        .store
        .load_session_blob(id, &hash)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to load the image", error))?;
    match blob {
        Some(blob) => Ok(([(header::CONTENT_TYPE, blob.media_type)], blob.bytes).into_response()),
        None => Err(ProblemDetail::new(
            ErrorKind::NotFound,
            StatusCode::NOT_FOUND,
            format!("session '{id}' has no image blob '{hash}'"),
        )
        .with("session_id", id.to_string())),
    }
}
