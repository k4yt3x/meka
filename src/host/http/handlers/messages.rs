//! Conversation history: `GET /v1/sessions/{id}/messages`.
//!
//! Returns the materialized `Conversation` view (post-compaction-aware) for clients that want
//! to read past turns. Pagination via `?limit=` and `?offset=` is intentionally simple; the
//! source of truth is the SQLite event log, which holds the full history.

use axum::{
    Json,
    extract::{Path, Query, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
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
    /// Increments on every compaction, rewind, mid-turn repair and image redaction. A polling
    /// client that sees it change knows its copy is no longer a prefix of the server's and must
    /// re-fetch rather than diff; `total` alone cannot tell it apart from data loss.
    ///
    /// The per-message `compaction` marker explains one of those four. This covers the other
    /// three: a rewind removes messages with nothing left behind to attach a marker to, which
    /// would otherwise reproduce exactly the silent-rewrite failure the marker was added to
    /// prevent, and a repair or a redaction rewrites a message in place.
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
    /// The turn that added this message: the id `turn.started` announced it under and
    /// `GET /v1/sessions/{id}/turns` lists it by. Omitted on a row no turn added (a compaction
    /// summary, a repair's replacement).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) turn_id: Option<Uuid>,
    /// Dense positional turn label (`t_0001`, `t_0002`, …), derived at query time. A message
    /// that opens a turn starts a new label, and the assistant and tool-result messages after it
    /// share it; what opens a turn is `Message::opens_turn`, the one rule rewind counts by, so a
    /// tool round's results never read as a turn of their own, and the labels from a chosen one
    /// to the last are the `turns` a rewind to that point takes. `None` on messages from the
    /// assembled-response path, which holds one turn and no view to count in.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) turn_label: Option<String>,
    /// Present only on a message that *is* a compaction summary.
    ///
    /// Without this a client polling `/messages` watches history rewrite itself: a compaction
    /// truncates the materialized tail and pushes a summary in its place, so `total` shrinks and
    /// messages the client already rendered stop coming back. The marker is what lets it tell
    /// "the window was summarized" from "the server lost my conversation".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) compaction: Option<CompactionMarker>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub(crate) struct CompactionMarker {
    /// How many materialized messages the boundary removed.
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
        (status = 200, description = "Page of conversation messages. The `ETag` header identifies the whole conversation's current state, for `If-Match` on `POST /rewind` and `POST /fork`", body = MessagesResponse),
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
) -> Result<Response, ProblemDetail> {
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
        stamps,
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
        .zip(stamps[offset..end].iter())
        .zip(turn_indexes[offset..end].iter())
        .zip(markers[offset..end].iter())
        .map(|(((message, stamp), turn_label), marker)| MessageView {
            role: match message.role {
                Role::User => "user".to_string(),
                Role::Assistant => "assistant".to_string(),
            },
            content: message.content.iter().map(view_for_block).collect(),
            created_at: Some(stamp.created_at.clone()),
            turn_id: stamp.turn_id,
            turn_label: Some(turn_label.clone()),
            compaction: marker.as_ref().map(|marker| CompactionMarker {
                replaced_count: marker.replaced_count,
                generation: marker.generation,
            }),
        })
        .collect();
    Ok((
        [(header::ETAG, conversation_etag(revision, total))],
        Json(MessagesResponse {
            session_id: id,
            messages,
            total,
            revision,
        }),
    )
        .into_response())
}

/// The entity tag of a conversation's materialized view: its revision and its length, which
/// together say whether a client's copy is still the server's. A rewrite moves the first and an
/// append moves the second, and `revision` alone would let a turn that landed between a read and
/// an edit go unnoticed.
pub(crate) fn conversation_etag(revision: u64, total: usize) -> HeaderValue {
    HeaderValue::from_str(&format!("\"{revision}-{total}\""))
        .unwrap_or_else(|_| HeaderValue::from_static("\"\""))
}

/// The `If-Match` precondition on an edit of the conversation: refused with 412 when none of the
/// tags the client sent is the view's current one, so an edit addressed at the view the client
/// read never lands on one it has not seen. No header is no precondition. `*` matches any view,
/// and a weak tag matches none, as RFC 9110 has it for a state-changing request.
pub(crate) fn check_if_match(
    headers: &HeaderMap,
    revision: u64,
    total: usize,
    session_id: Uuid,
) -> Result<(), ProblemDetail> {
    let mut tags = headers
        .get_all(header::IF_MATCH)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .peekable();
    if tags.peek().is_none() {
        return Ok(());
    }
    let current = conversation_etag(revision, total);
    let current = current.to_str().unwrap_or_default();
    if tags.any(|tag| tag == "*" || tag == current) {
        return Ok(());
    }
    Err(ProblemDetail::new(
        ErrorKind::PreconditionFailed,
        StatusCode::PRECONDITION_FAILED,
        format!(
            "the conversation has changed since it was read; read `GET /v1/sessions/{session_id}/messages` again and retry with its `ETag`"
        ),
    )
    .with("session_id", session_id.to_string())
    .with("revision", revision)
    .with("total", total as u64))
}

/// Group materialized messages into virtual turns. A message that opens a turn starts a new
/// label; the assistant and tool-result messages after it share it. Returns a parallel `Vec` the
/// same length as `messages` with `t_NNNN` labels.
fn derive_turn_indexes(messages: &[Message]) -> Vec<String> {
    let mut counter: u32 = 0;
    // A view that begins before its first opener (a tail a repair left standing) is labeled as
    // the first turn rather than left blank.
    let mut current = String::from("t_0001");
    let mut ids = Vec::with_capacity(messages.len());
    for message in messages {
        if message.opens_turn() {
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
            content: tool_result_views(content),
        },
    }
}

/// A tool result's content as the wire shows it, with an image reduced to its type and hash.
pub(crate) fn tool_result_views(content: &[ToolResultContent]) -> Vec<ToolResultContentView> {
    content
        .iter()
        .map(|item| match item {
            ToolResultContent::Text { text } => ToolResultContentView::Text { text: text.clone() },
            ToolResultContent::Image { source } => ToolResultContentView::Image {
                media_type: source.media_type().to_string(),
                hash: blob_hash(source),
            },
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_round() -> (Message, Message) {
        let call = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "tu_1".to_string(),
                name: "file_read".to_string(),
                input: serde_json::json!({ "path": "a.txt" }),
            }],
        };
        let result = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "tu_1".to_string(),
                content: vec![ToolResultContent::Text {
                    text: "contents".to_string(),
                }],
                is_error: false,
            }],
        };
        (call, result)
    }

    /// A tool round answers with a user-role message, which continues the turn that asked
    /// rather than opening one; a task that took one round is still one turn, as rewind counts.
    #[test]
    fn a_tool_result_shares_the_label_of_the_turn_that_asked_for_it() {
        let (call, result) = tool_round();
        let messages = vec![
            Message::user("first"),
            call,
            result,
            Message::assistant_text("done"),
            Message::user("second"),
        ];
        assert_eq!(derive_turn_indexes(&messages), [
            "t_0001", "t_0001", "t_0001", "t_0001", "t_0002"
        ]);
    }

    /// No header is no precondition; the current tag passes; a tag from before an append is
    /// refused with the current state, since `revision` alone would not have moved; a weak tag
    /// never matches a state-changing request; `*` matches any view.
    #[test]
    fn the_precondition_accepts_the_current_tag_and_refuses_a_stale_one() {
        let mut headers = HeaderMap::new();
        assert!(check_if_match(&headers, 3, 42, Uuid::nil()).is_ok());

        headers.insert(header::IF_MATCH, conversation_etag(3, 42));
        assert!(check_if_match(&headers, 3, 42, Uuid::nil()).is_ok());
        let stale = check_if_match(&headers, 3, 43, Uuid::nil()).expect_err("an append since");
        assert_eq!(stale.status, 412);
        assert_eq!(stale.type_uri, ErrorKind::PreconditionFailed.type_uri());
        assert_eq!(stale.extensions["revision"], 3);
        assert_eq!(stale.extensions["total"], 43);

        headers.insert(header::IF_MATCH, HeaderValue::from_static("W/\"3-42\""));
        assert!(check_if_match(&headers, 3, 42, Uuid::nil()).is_err());

        headers.insert(header::IF_MATCH, HeaderValue::from_static("*"));
        assert!(check_if_match(&headers, 9, 9, Uuid::nil()).is_ok());
    }
}
