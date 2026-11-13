//! A session's conversation, read: `GET /v1/sessions/{id}/history`, what was said, and
//! `GET /v1/sessions/{id}/context/messages`, what the model sees, with the block views both and the
//! feed share, and the conversation's position, which tags both and guards an edit of it.

use axum::{
    Json,
    extract::{Path, Query, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use base64::Engine;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use crate::{
    conversation::{ContentBlock, Message, ToolResultContent},
    host::http::{
        errors::{ErrorKind, ProblemDetail},
        scope,
        state::ServerState,
    },
};

/// A message's role and its blocks as the API types them: a message on the history, the opening a
/// feed announces, a round of a compaction's checkpoint.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct MessageBody {
    pub(crate) role: String,
    pub(crate) content: Vec<ContentBlockView>,
    /// The fields a provider put on the message itself that nothing else holds, as it returned
    /// them (a Chat Completions message's `reasoning_details`).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<Object>)]
    pub(crate) provider_fields: Option<crate::conversation::ProviderFields>,
}

/// `message` as the API types it.
pub(crate) fn message_body(message: &Message) -> MessageBody {
    MessageBody {
        role: message.role.name().to_string(),
        content: message.content.iter().map(view_for_block).collect(),
        provider_fields: message.provider_fields.clone(),
    }
}

/// A message on the history, and for a reply what its provider said about it.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct MessageItem {
    #[serde(flatten)]
    pub(crate) message: MessageBody,
    /// On a reply, what its provider said about it beside its content; absent on any other
    /// message and on a reply written before Meka recorded it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reply: Option<Box<ReplyView>>,
}

/// One part of a thinking block.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ReasoningPartView {
    /// `reasoning`, the reasoning itself, or `summary`, a summary the provider wrote of reasoning
    /// it keeps.
    pub(crate) kind: String,
    pub(crate) text: String,
}

/// What a provider said about a reply beside its content.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ReplyView {
    /// How the reply ended, as its provider said; absent on one cut before it said.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stop: Option<StopView>,
    /// What cut the reply short: `stopped`, `interrupted` or `failed`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cut: Option<String>,
    /// The model the response names.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) model: Option<String>,
    /// The profile that sent the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) profile: Option<String>,
    /// What the response reported spending, as Meka counts it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) usage: Option<crate::host::http::handlers::turn::UsageView>,
    /// The response's fields nothing else holds, as its provider returned them: its id, the raw
    /// `usage` the counts are read from, and whatever else it said about the completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<Object>)]
    pub(crate) provider_fields: Option<crate::conversation::ProviderFields>,
}

/// How a reply ended.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct StopView {
    /// Meka's class of the stop: `end_turn`, `tool_use`, `max_tokens`, `context_window`,
    /// `refusal`, `content_filter` or `unknown`.
    pub(crate) reason: String,
    /// The provider's own word for it, as sent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) word: Option<String>,
    /// What the provider said about the stop beside its word, under its own names.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<Object>)]
    pub(crate) details: Option<serde_json::Map<String, serde_json::Value>>,
}

impl From<&crate::conversation::ReplyRecord> for ReplyView {
    fn from(record: &crate::conversation::ReplyRecord) -> Self {
        Self {
            stop: record.stop.as_ref().map(|stop| StopView {
                reason: stop.reason.name().to_string(),
                word: stop.word.clone(),
                details: stop.details.clone(),
            }),
            cut: record.cut.map(|cut| cut.name().to_string()),
            model: record.model.clone(),
            profile: record.profile.clone(),
            usage: record.usage.as_ref().map(|usage| {
                crate::host::http::handlers::turn::UsageView {
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    cache_creation_input_tokens: usage.cache_creation_input_tokens,
                    cache_read_input_tokens: usage.cache_read_input_tokens,
                }
            }),
            provider_fields: record.provider_fields.clone(),
        }
    }
}

/// `recorded` as the API types it.
fn message_item(recorded: &crate::conversation::RecordedMessage) -> MessageItem {
    MessageItem {
        message: message_body(&recorded.message),
        reply: recorded
            .reply
            .as_ref()
            .map(|record| Box::new(ReplyView::from(record))),
    }
}

/// An entry of a session's history.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct HistoryItem {
    /// The entry's position in the session's log: never reused, and the order of the history.
    pub(crate) position: i64,
    /// RFC 3339, when the entry was said: when it was written, or for a copy a compaction kept
    /// standing alone, when the message it copies was.
    pub(crate) created_at: String,
    /// The turn that said the entry, as `GET /v1/sessions/{id}/turns` lists it: the one that wrote
    /// it, or for a copy a compaction kept standing alone, the one that said the message it
    /// copies; omitted on one said between turns or before turns were recorded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) turn_id: Option<Uuid>,
    #[serde(flatten)]
    pub(crate) kind: HistoryItemKind,
}

/// What a [`HistoryItem`] is, by its `type`.
#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum HistoryItemKind {
    /// A message, where it was said.
    Message(MessageItem),
    /// A compaction, where it happened.
    Compaction(CompactionItem),
    /// A compaction that failed or was stopped, where it ended, with what it did on the way.
    CompactionFailed(CompactionFailedItem),
    /// A rewind, and the positions it took out of the history.
    Rewind(RewindItem),
    /// A prompt a turn took back to join a message to it; the message it joined follows.
    Join(JoinItem),
}

/// A compaction as the history says it.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct CompactionItem {
    /// The summary as the agent or the summarizer wrote it. The message the model reads in its
    /// place is on `GET /v1/sessions/{id}/context/messages`.
    pub(crate) summary: String,
    /// The rounds the checkpoint turn ran on the way, as they happened: each reply as returned
    /// with its record, and the results sent back. Empty when no checkpoint round ran.
    pub(crate) checkpoint: Vec<MessageItem>,
    /// The summarizer's replies, as returned with their records, when the summarizer wrote the
    /// summary.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) summarizer: Vec<MessageItem>,
    #[serde(flatten)]
    pub(crate) compaction: crate::view::CompactionView,
}

/// A compaction that failed or was stopped, as the history says it.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct CompactionFailedItem {
    /// What set it going: `reactive`, `proactive`, `manual`, `requested`, `emergency` or
    /// `retention`.
    pub(crate) trigger: String,
    /// The memories its checkpoint wrote, which stay written.
    pub(crate) memories_written: Vec<String>,
    /// See [`CompactionItem::checkpoint`].
    pub(crate) checkpoint: Vec<MessageItem>,
    /// See [`CompactionItem::summarizer`].
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) summarizer: Vec<MessageItem>,
    /// What it failed on; absent when it was stopped.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
}

/// A rewind as the history says it.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct RewindItem {
    /// The positions it took out of the history, as ranges in order: one for the turns after the
    /// last compaction, and one more for each run of turns the compaction kept, which the history
    /// shows where they were said, before it.
    pub(crate) ranges: Vec<PositionRange>,
    /// How many turns it took.
    pub(crate) turns: usize,
}

/// A joined prompt as the history says it.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct JoinItem {
    /// The positions the turn took back, as ranges in order.
    pub(crate) ranges: Vec<PositionRange>,
}

/// The positions of a session's history from `from` through `through`, both included; another
/// session's positions between them are no part of it.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct PositionRange {
    pub(crate) from: i64,
    pub(crate) through: i64,
}

/// `rows` as the ranges an item carries.
fn position_ranges(rows: &[std::ops::RangeInclusive<i64>]) -> Vec<PositionRange> {
    rows.iter()
        .map(|rows| PositionRange {
            from: *rows.start(),
            through: *rows.end(),
        })
        .collect()
}

/// `entry` as the API says it.
pub(crate) fn history_item(entry: crate::store::HistoryEntry) -> HistoryItem {
    use crate::store::HistoryKind;
    let kind = match entry.kind {
        HistoryKind::Message { message, reply } => HistoryItemKind::Message(MessageItem {
            message: message_body(&message),
            reply: reply.map(|record| Box::new(ReplyView::from(&*record))),
        }),
        HistoryKind::Compaction { record, generation } => {
            HistoryItemKind::Compaction(CompactionItem {
                summary: record.written_summary.clone(),
                checkpoint: record.checkpoint.iter().map(message_item).collect(),
                summarizer: record.summarizer.iter().map(message_item).collect(),
                compaction: crate::view::CompactionView::new(&record, generation),
            })
        }
        HistoryKind::CompactionFailed { failure } => {
            HistoryItemKind::CompactionFailed(CompactionFailedItem {
                trigger: failure.trigger.name().to_string(),
                memories_written: failure.memories_written.clone(),
                checkpoint: failure.checkpoint.iter().map(message_item).collect(),
                summarizer: failure.summarizer.iter().map(message_item).collect(),
                error: failure
                    .error
                    .as_ref()
                    .map(|error| error.sentence(crate::host::http::errors::api_wording)),
            })
        }
        HistoryKind::Rewind { rows, turns } => HistoryItemKind::Rewind(RewindItem {
            ranges: position_ranges(&rows),
            turns,
        }),
        HistoryKind::Join { rows } => HistoryItemKind::Join(JoinItem {
            ranges: position_ranges(&rows),
        }),
    };
    HistoryItem {
        position: entry.position,
        created_at: entry.created_at,
        turn_id: entry.turn_id,
        kind,
    }
}

/// Where to read a page of history from: below `before`, above `after`, or the last page.
#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub(crate) struct HistoryQuery {
    /// The page below this position: the `next_before` of the page above it.
    #[serde(default)]
    pub(crate) before: Option<i64>,
    /// The page above this position: the `next_after` of the page below it, or the last position
    /// a client holds, to read what it has not.
    #[serde(default)]
    pub(crate) after: Option<i64>,
    /// How many entries at most: 200 by default, clamped to 1..1000.
    #[serde(default)]
    pub(crate) limit: Option<usize>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct HistoryResponse {
    pub(crate) session_id: Uuid,
    /// The conversation's position as the page was read, which its `ETag` also carries.
    pub(crate) position: i64,
    /// The entries, oldest first.
    pub(crate) items: Vec<HistoryItem>,
    /// Present when older entries remain: the `before` of the page below.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) next_before: Option<i64>,
    /// Present when newer entries remain: the `after` of the page above.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) next_after: Option<i64>,
}

/// The most entries a page of history holds.
const HISTORY_PAGE_MOST_ENTRIES: usize = 1000;

/// How many entries a page of history holds when `limit` is not given.
const HISTORY_PAGE_ENTRIES: usize = 200;

#[utoipa::path(
    get,
    path = "/v1/sessions/{id}/history",
    tag = "sessions",
    params(
        ("id" = Uuid, Path, description = "Session UUID"),
        HistoryQuery,
    ),
    responses(
        (status = 200, description = "A page of the session's history. The `ETag` header is the conversation's position, for `If-Match` on `POST /rewind` and `POST /fork`", body = HistoryResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 422, description = "Both `before` and `after` were given", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:r"]))
)]
pub(crate) async fn list_history(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SessionsRead>,
    Path(id): Path<Uuid>,
    Query(query): Query<HistoryQuery>,
) -> Result<Response, ProblemDetail> {
    let edge = match (query.before, query.after) {
        (Some(_), Some(_)) => {
            return Err(ProblemDetail::new(
                ErrorKind::InvalidBody,
                StatusCode::UNPROCESSABLE_ENTITY,
                "`before` and `after` each name a page; send one of them",
            ));
        }
        (before, None) => crate::store::HistoryEdge::Before(before),
        (None, Some(after)) => crate::store::HistoryEdge::After(after),
    };
    let limit = query
        .limit
        .unwrap_or(HISTORY_PAGE_ENTRIES)
        .clamp(1, HISTORY_PAGE_MOST_ENTRIES);
    crate::host::http::reattach::require_session_exists(&state, id).await?;
    // Read ahead of the page, so a row written while it is read can only make the tag older than
    // the page, which refuses an edit decided on it rather than letting one through.
    let position = conversation_position(&state.shared.store, id).await?;
    let page = state
        .shared
        .store
        .history_page(id, edge, limit)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to read the history", error))?;
    let more = page.more;
    let first = page.entries.first().map(|entry| entry.position);
    let last = page.entries.last().map(|entry| entry.position);
    let (next_before, next_after) = match edge {
        crate::store::HistoryEdge::Before(_) => (first.filter(|_| more), None),
        crate::store::HistoryEdge::After(_) => (None, last.filter(|_| more)),
    };
    Ok((
        [(header::ETAG, conversation_etag(position))],
        Json(HistoryResponse {
            session_id: id,
            position,
            items: page.entries.into_iter().map(history_item).collect(),
            next_before,
            next_after,
        }),
    )
        .into_response())
}

/// A message of the context.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ContextMessage {
    /// The position of the history item this message shows: its own row, the row a copy a
    /// compaction kept stands for, the row a message Meka rewrote in place was said in, or a
    /// summary's compaction.
    pub(crate) position: i64,
    /// RFC 3339, when the entry it shows was said.
    pub(crate) created_at: String,
    /// The turn that said the entry it shows; omitted on a summary, on one said between turns,
    /// and on one said before turns were recorded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) turn_id: Option<Uuid>,
    pub(crate) role: String,
    pub(crate) content: Vec<ContentBlockView>,
    /// Present only on a compaction summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) compaction: Option<crate::view::CompactionView>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ContextMessagesResponse {
    pub(crate) session_id: Uuid,
    /// The conversation's position as the context was read, which its `ETag` also carries.
    pub(crate) position: i64,
    /// Every message of the context, in the order the model reads them.
    pub(crate) messages: Vec<ContextMessage>,
}

#[utoipa::path(
    get,
    path = "/v1/sessions/{id}/context/messages",
    tag = "sessions",
    params(("id" = Uuid, Path, description = "Session UUID")),
    responses(
        (status = 200, description = "Every message of the context. The `ETag` header is the conversation's position, for `If-Match` on `POST /rewind` and `POST /fork`", body = ContextMessagesResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:r"]))
)]
pub(crate) async fn list_context_messages(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SessionsRead>,
    Path(id): Path<Uuid>,
) -> Result<Response, ProblemDetail> {
    crate::host::http::reattach::require_session_exists(&state, id).await?;
    let view = context_view(&state.shared.store, id).await?;
    Ok((
        [(header::ETAG, conversation_etag(view.position))],
        Json(ContextMessagesResponse {
            session_id: id,
            position: view.position,
            messages: view.messages,
        }),
    )
        .into_response())
}

/// The context as the API says it, and the conversation's position as it was read.
pub(crate) struct ContextView {
    pub(crate) position: i64,
    pub(crate) messages: Vec<ContextMessage>,
}

/// Read the session's context from the store: the rows from its last compaction on, replayed by the
/// rules the model's view is, each message stamped with the history item it shows.
pub(crate) async fn context_view(
    store: &crate::store::Store,
    id: Uuid,
) -> Result<ContextView, ProblemDetail> {
    let events = store
        .load_view_events_with_stamps(id)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to read the context", error))?;
    // The view's rows run to the end of the log, so the last is the conversation's position.
    let position = events.last().map_or(0, |(stamp, _)| stamp.id);
    let crate::conversation::AnnotatedView {
        messages,
        stamps,
        markers,
        ..
    } = crate::conversation::materialize_annotated(&events);
    // A copy a compaction kept stands in history for the row it copies, and says itself when and
    // in which turn it was said.
    let originals = store
        .original_positions(
            id,
            stamps.iter().filter_map(|stamp| stamp.copy_of).collect(),
        )
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to read the context", error))?;
    // Counted below the summary's own boundary, as the history numbers it, so a compaction landing
    // while this is read cannot move the number.
    let summarized_at = stamps
        .iter()
        .zip(&markers)
        .find_map(|(stamp, marker)| marker.as_ref().map(|_| stamp.id));
    let compactions_before = match summarized_at {
        Some(boundary) => store
            .compactions_before(id, boundary)
            .await
            .map_err(|error| {
                ProblemDetail::internal_sanitized("failed to count compactions", error)
            })?,
        None => 0,
    };
    let messages = messages
        .iter()
        .zip(stamps)
        .zip(markers)
        .map(|((message, stamp), marker)| {
            let position = stamp
                .copy_of
                .and_then(|original| originals.get(&original).copied())
                .unwrap_or(stamp.id);
            let (turn_id, created_at) = match stamp.said {
                Some(said) => (said.turn_id, said.created_at),
                None => (stamp.turn_id, stamp.created_at),
            };
            ContextMessage {
                position,
                created_at,
                // A summary stands for the turns before it, not for the turn whose compaction
                // wrote it, which a rewind to that turn would otherwise take the summary with.
                turn_id: turn_id.filter(|_| marker.is_none()),
                role: message.role.name().to_string(),
                content: message.content.iter().map(view_for_block).collect(),
                compaction: marker.map(|marker| {
                    crate::view::CompactionView::new(
                        &marker.record,
                        compactions_before.saturating_add(1),
                    )
                }),
            }
        })
        .collect();
    Ok(ContextView { position, messages })
}

/// The conversation's position: its log's last row, which every change to its history or its
/// context writes, or `0` for a session that has none.
pub(crate) async fn conversation_position(
    store: &crate::store::Store,
    id: Uuid,
) -> Result<i64, ProblemDetail> {
    store
        .last_log_row(id)
        .await
        .map(|position| position.unwrap_or(0))
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to read the conversation's position", error)
                .with("session_id", id.to_string())
        })
}

/// The entity tag of a conversation: its position, which every change to it moves.
pub(crate) fn conversation_etag(position: i64) -> HeaderValue {
    HeaderValue::from_str(&format!("\"{position}\""))
        .unwrap_or_else(|_| HeaderValue::from_static("\"\""))
}

/// The `If-Match` precondition on an edit of the conversation: refused with 412 when none of the
/// tags the client sent is the conversation's current one, so an edit decided on a conversation the
/// client read never lands on one it has not seen. No header is no precondition. `*` matches any
/// conversation, and a weak tag matches none, as RFC 9110 has it for a state-changing request.
pub(crate) fn check_if_match(
    headers: &HeaderMap,
    position: i64,
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
    let current = conversation_etag(position);
    let current = current.to_str().unwrap_or_default();
    if tags.any(|tag| tag == "*" || tag == current) {
        return Ok(());
    }
    Err(ProblemDetail::new(
        ErrorKind::PreconditionFailed,
        StatusCode::PRECONDITION_FAILED,
        format!(
            "the conversation has changed since it was read; read `GET /v1/sessions/{session_id}/history` again and retry with its `ETag`"
        ),
    )
    .with("session_id", session_id.to_string())
    .with("position", position))
}

/// The hash an image view names. A message read back holds a reference into the store; one
/// announced as it is appended still holds the bytes, which the store is filing under the same
/// name, so the view names it rather than sending nothing. A payload that is not base64 stays
/// inline in the row and is named by nothing.
fn blob_hash(source: &crate::image::ImageSource) -> Option<String> {
    match source {
        crate::image::ImageSource::Blob { hash, .. } => Some(hash.clone()),
        crate::image::ImageSource::Base64 { data, .. } => base64::engine::general_purpose::STANDARD
            .decode(data.as_bytes())
            .ok()
            .map(|bytes| crate::store::content_hash(&bytes)),
    }
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ContentBlockView {
    Text {
        text: String,
        /// The part's fields nothing else holds, as its provider returned them (a Responses
        /// message's `phase`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[schema(value_type = Option<Object>)]
        provider_fields: Option<crate::conversation::ProviderFields>,
    },
    /// Words a model returned in the field its provider keeps for declining (OpenAI's `refusal`).
    Refusal {
        text: String,
        /// The part's fields nothing else holds, as its provider returned them.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[schema(value_type = Option<Object>)]
        provider_fields: Option<crate::conversation::ProviderFields>,
    },
    /// What a provider returned that Meka reads no block from, verbatim and in the order it
    /// arrived; never sent back to a model.
    Unrecognized {
        #[schema(value_type = Vec<Object>)]
        parts: Vec<serde_json::Value>,
    },
    /// What Meka injected ahead of the user's words for that turn (permission and environment
    /// context, catalog changes, background outcomes, the resume notice), which the model
    /// saw as text ahead of them. Typed so a client can show or hide it; `text` alone is the words.
    TurnContext { text: String },
    /// Words Meka wrote into the model's own turn in the user's role, to send it back to work:
    /// `kind` is `visible_reply` after a reply with no visible text, `checklist` after one that
    /// left checklist items open. Typed so a client can show it as Meka's rather than the person's.
    Nudge { kind: String, text: String },
    /// Words Meka wrote into a user message in place of what it removed (an image the provider
    /// rejected or the request budget could not carry): `note` the sentence, `text` as the model
    /// read it, behind Meka's label. Typed so a client shows it as Meka's rather than the person's.
    HarnessNote { note: String, text: String },
    /// The message a compaction put in place of the turns it summarized, as the model reads it.
    /// The message's `compaction` says what the compaction was, its summary as written included.
    Summary { text: String },
    /// An inbox item the model read: `text` as it read it, header included, `body` the person's
    /// words, with the item's id, its class (`steer`, `followup`, `interrupt`) and its source
    /// where they were recorded. Typed so a client shows the words as the person's and labels
    /// them without parsing the header.
    Inbox {
        #[serde(skip_serializing_if = "Option::is_none")]
        item_id: Option<Uuid>,
        #[serde(skip_serializing_if = "Option::is_none")]
        class: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        source: Option<String>,
        body: String,
        text: String,
    },
    /// A scheduled job's fire the model read: `text` as it read it, `prompt` the job's words,
    /// `job_id` the job. Typed so a client shows it as a fire rather than as words a person typed.
    Scheduled {
        job_id: String,
        prompt: String,
        text: String,
    },
    /// Finished background work the model read, the tasks `task_ids` name: the `text` a
    /// `task.reported` feed event carried, or a turn of its own delivered.
    BackgroundReport { task_ids: Vec<String>, text: String },
    /// A skill or an MCP prompt somebody invoked by name: `kind` is `skill` or `mcp_prompt`,
    /// `name` what was invoked (an MCP prompt as `server:prompt`), `directive` the words typed
    /// with it, a file an editor attached among them named by its mention (`@name`), blank when
    /// none were, and `text` what the model read, the expanded body and file contents included.
    Invocation {
        kind: String,
        name: String,
        directive: String,
        text: String,
    },
    /// A resource an editor attached among the person's words, where it was attached: its `uri`
    /// and `name`, with `text` the tag the model read, contents included.
    Resource {
        uri: String,
        name: String,
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
    /// Reasoning the model showed, in the parts its provider returned it in.
    Thinking {
        parts: Vec<ReasoningPartView>,
        /// The part's fields nothing else holds, as its provider returned them.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[schema(value_type = Option<Object>)]
        provider_fields: Option<crate::conversation::ProviderFields>,
    },
    /// Encrypted reasoning whose contents the API withholds; surfaced as a presence marker so
    /// history responses note it occurred without exposing the opaque payload.
    RedactedThinking {
        /// The part's fields nothing else holds, as its provider returned them.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[schema(value_type = Option<Object>)]
        provider_fields: Option<crate::conversation::ProviderFields>,
    },
    ToolUse {
        id: String,
        name: String,
        /// The arguments; `{}` when `unparsed` is present.
        #[schema(value_type = Object)]
        input: serde_json::Value,
        /// What the model sent as the arguments when it was not a JSON object, and why the call
        /// was refused.
        #[serde(skip_serializing_if = "Option::is_none")]
        unparsed: Option<UnparsedArgumentsView>,
        /// Every field the provider put on the call beyond its id, name and arguments, as it came.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[schema(value_type = Option<Object>)]
        provider_fields: Option<crate::conversation::ProviderFields>,
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
    /// A deferred tool the result made callable, by name.
    ToolReference {
        tool_name: String,
    },
    /// Words Meka wrote into the result for the model: `note` the sentence, `text` as the model
    /// read it, behind Meka's label.
    HarnessNote {
        note: String,
        text: String,
    },
    /// The line naming where the image after it came from (a file read, a page fetched).
    ImageCaption {
        caption: String,
    },
    /// An output too large to keep inline, saved whole to the scratchpad entry it names, with its
    /// size and its opening.
    LargeOutput {
        scratchpad_entry: String,
        size_bytes: usize,
        preview: String,
    },
}

/// What a model sent as a call's arguments when it was not a JSON object.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct UnparsedArgumentsView {
    /// The text as the model sent it; absent on a call recorded before the text was kept, or
    /// that carried no arguments.
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    /// Why the text is not arguments, in the words the call was refused with.
    error: String,
}

fn view_for_block(block: &ContentBlock) -> ContentBlockView {
    match block {
        ContentBlock::Text {
            text,
            provider_fields,
        } => ContentBlockView::Text {
            text: text.clone(),
            provider_fields: provider_fields.clone(),
        },
        ContentBlock::Refusal {
            text,
            provider_fields,
        } => ContentBlockView::Refusal {
            text: text.clone(),
            provider_fields: provider_fields.clone(),
        },
        ContentBlock::Unrecognized { parts } => ContentBlockView::Unrecognized {
            parts: parts.clone(),
        },
        ContentBlock::TurnContext { text } => ContentBlockView::TurnContext { text: text.clone() },
        ContentBlock::Nudge { kind, text } => ContentBlockView::Nudge {
            kind: kind.name().to_string(),
            text: text.clone(),
        },
        ContentBlock::HarnessNote { note } => ContentBlockView::HarnessNote {
            note: note.clone(),
            text: crate::conversation::harness_note_text(note),
        },
        ContentBlock::Summary { text } => ContentBlockView::Summary { text: text.clone() },
        ContentBlock::Inbox {
            item_id,
            class,
            source,
            body,
            text,
        } => ContentBlockView::Inbox {
            item_id: *item_id,
            class: class.map(|class| class.name().to_string()),
            source: source.clone(),
            body: body.clone(),
            text: text.clone(),
        },
        ContentBlock::Scheduled {
            job_id,
            prompt,
            text,
        } => ContentBlockView::Scheduled {
            job_id: job_id.clone(),
            prompt: prompt.clone(),
            text: text.clone(),
        },
        ContentBlock::BackgroundReport { task_ids, text } => ContentBlockView::BackgroundReport {
            task_ids: task_ids.clone(),
            text: text.clone(),
        },
        ContentBlock::Invocation {
            kind,
            name,
            directive,
            text,
        } => ContentBlockView::Invocation {
            kind: kind.name().to_string(),
            name: name.clone(),
            directive: directive.clone(),
            text: text.clone(),
        },
        ContentBlock::Resource { uri, name, text } => ContentBlockView::Resource {
            uri: uri.clone(),
            name: name.clone(),
            text: text.clone(),
        },
        ContentBlock::Image { source } => ContentBlockView::Image {
            media_type: source.media_type().to_string(),
            hash: blob_hash(source),
        },
        ContentBlock::Thinking {
            parts,
            provider_fields,
            ..
        } => ContentBlockView::Thinking {
            parts: parts
                .iter()
                .map(|part| ReasoningPartView {
                    kind: part.kind.name().to_string(),
                    text: part.text.clone(),
                })
                .collect(),
            provider_fields: provider_fields.clone(),
        },
        ContentBlock::RedactedThinking {
            provider_fields, ..
        } => ContentBlockView::RedactedThinking {
            provider_fields: provider_fields.clone(),
        },
        ContentBlock::ToolUse {
            id,
            name,
            input,
            unparsed,
            provider_fields,
        } => ContentBlockView::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
            unparsed: unparsed.as_ref().map(|unparsed| UnparsedArgumentsView {
                text: unparsed.text.clone(),
                error: unparsed.error.clone(),
            }),
            provider_fields: provider_fields.clone(),
        },
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
            ..
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
            ToolResultContent::ToolReference { tool_name } => {
                ToolResultContentView::ToolReference {
                    tool_name: tool_name.clone(),
                }
            }
            ToolResultContent::HarnessNote { note } => ToolResultContentView::HarnessNote {
                note: note.clone(),
                text: crate::conversation::harness_note_text(note),
            },
            ToolResultContent::ImageCaption { caption } => ToolResultContentView::ImageCaption {
                caption: caption.clone(),
            },
            ToolResultContent::LargeOutput {
                scratchpad_entry,
                size_bytes,
                preview,
            } => ToolResultContentView::LargeOutput {
                scratchpad_entry: scratchpad_entry.clone(),
                size_bytes: *size_bytes,
                preview: preview.clone(),
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
        ("hash" = String, Path, description = "Content hash of an image block, as `GET /v1/sessions/{id}/history` reports it"),
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
        Some(blob) => Ok((
            [
                (header::CONTENT_TYPE, blob.media_type),
                (header::X_CONTENT_TYPE_OPTIONS, "nosniff".to_string()),
            ],
            blob.bytes,
        )
            .into_response()),
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

    /// A session compacted twice numbers its summary 2: one past the compactions the log holds
    /// below its boundary, as the history numbers its item.
    #[tokio::test]
    async fn a_summary_is_numbered_by_the_compactions_below_it() {
        use crate::conversation::Event;
        let store = crate::store::Store::for_test().await;
        let id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        for words in ["one?", "two?"] {
            store
                .save_event(id, &Event::Append(Message::user(words)), None)
                .await
                .expect("save");
            let boundary = Event::CompactBoundary {
                summary: Message::user("the summary"),
                replaced_count: 1,
                record: crate::conversation::CompactionRecord::for_test(),
                loaded_tools_snapshot: Default::default(),
                checklist_snapshot: Default::default(),
            };
            store
                .save_compaction(id, &boundary, &[], None)
                .await
                .expect("compact");
        }
        let view = context_view(&store, id).await.expect("the context");
        let numbered: Vec<u64> = view
            .messages
            .iter()
            .filter_map(|message| {
                message
                    .compaction
                    .as_ref()
                    .map(|compaction| compaction.generation)
            })
            .collect();
        assert_eq!(numbered, [2]);
    }

    /// A message a compaction kept is shown in the context as the history item it copies: at that
    /// item's position, under the turn that said it and when, not under the compaction's. The
    /// summary points at the compaction and names no turn.
    #[tokio::test]
    async fn a_kept_message_shows_the_turn_and_time_it_was_said() {
        use crate::conversation::Event;
        let store = crate::store::Store::for_test().await;
        let id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let said_in = Uuid::new_v4();
        store
            .turn_store()
            .open_turn(
                id,
                said_in,
                crate::store::turns::TurnSource::Client,
                crate::permission::Permission::Read,
                "2026-10-29T00:00:00+00:00",
            )
            .await
            .expect("open");
        let kept = vec![Message::user("one?"), Message::assistant_text("One.")];
        for message in &kept {
            store
                .save_event(id, &Event::Append(message.clone()), Some(said_in))
                .await
                .expect("save");
        }
        let history = store
            .history_page(id, crate::store::HistoryEdge::Before(None), 10)
            .await
            .expect("history");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        let boundary = Event::CompactBoundary {
            summary: Message::user("the summary"),
            replaced_count: 2,
            record: crate::conversation::CompactionRecord::for_test(),
            loaded_tools_snapshot: Default::default(),
            checklist_snapshot: Default::default(),
        };
        // Inside a later turn, as an automatic compaction writes it: its rows name that turn.
        let compacting = Uuid::new_v4();
        store
            .turn_store()
            .open_turn(
                id,
                compacting,
                crate::store::turns::TurnSource::Client,
                crate::permission::Permission::Read,
                "2026-10-29T00:01:00+00:00",
            )
            .await
            .expect("open");
        store
            .save_compaction(id, &boundary, &kept, Some(compacting))
            .await
            .expect("compact");

        let context = context_view(&store, id).await.expect("context");
        assert_eq!(context.messages.len(), 3);
        let summary = &context.messages[0];
        assert!(summary.compaction.is_some());
        assert_eq!(summary.turn_id, None);
        for (shown, said) in context.messages[1..].iter().zip(&history.entries) {
            assert_eq!(shown.position, said.position);
            assert_eq!(shown.turn_id, Some(said_in));
            assert_eq!(shown.created_at, said.created_at);
        }
        assert!(summary.position > history.entries[1].position);
        assert_eq!(
            context.position,
            store.last_log_row(id).await.expect("end").expect("rows")
        );
    }

    /// A message a retry rewrote in place, an earlier turn's answer among them, is shown as the
    /// history item it rewrote, under the turn that said it, so a rewind to the retrying turn
    /// leaves the earlier turn whole.
    #[tokio::test]
    async fn a_message_a_retry_rewrote_shows_the_turn_that_said_it() {
        use crate::conversation::Event;
        let store = crate::store::Store::for_test().await;
        let id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let (said, retrying) = (Uuid::new_v4(), Uuid::new_v4());
        for turn in [said, retrying] {
            store
                .turn_store()
                .open_turn(
                    id,
                    turn,
                    crate::store::turns::TurnSource::Client,
                    crate::permission::Permission::Read,
                    "2026-10-29T00:00:00+00:00",
                )
                .await
                .expect("open");
        }
        for (message, turn) in [
            (Message::user("one?"), said),
            (Message::assistant_text("One."), said),
            (Message::user("two?"), retrying),
        ] {
            store
                .save_event(id, &Event::Append(message), Some(turn))
                .await
                .expect("save");
        }
        let history = store
            .history_page(id, crate::store::HistoryEdge::Before(None), 10)
            .await
            .expect("history");
        store
            .save_event(
                id,
                &Event::Repair {
                    replaced_count: 2,
                    messages: vec![
                        Message::assistant_text("One, degraded."),
                        Message::user("two, degraded?"),
                    ],
                },
                Some(retrying),
            )
            .await
            .expect("repair");

        let context = context_view(&store, id).await.expect("context");
        let shown: Vec<(i64, Option<Uuid>, &str)> = context
            .messages
            .iter()
            .map(|message| {
                (
                    message.position,
                    message.turn_id,
                    message.created_at.as_str(),
                )
            })
            .collect();
        let said_as: Vec<(i64, Option<Uuid>, &str)> = history
            .entries
            .iter()
            .map(|entry| (entry.position, entry.turn_id, entry.created_at.as_str()))
            .collect();
        assert_eq!(shown, said_as);
        assert_eq!(
            crate::host::http::handlers::conversation::messages_from_turn(&store, id, retrying)
                .await
                .expect("in context"),
            1
        );
    }

    /// An invocation and a resource keep their kind on the wire, with what a client shows of them
    /// beside the text the model read.
    #[test]
    fn the_view_types_an_invocation_and_a_resource() {
        let message = Message::invoked_with_a_resource_for_test();
        let views: Vec<serde_json::Value> = message
            .content
            .iter()
            .map(|block| serde_json::to_value(view_for_block(block)).expect("json"))
            .collect();
        assert_eq!(
            views[1],
            serde_json::json!({
                "type": "invocation",
                "kind": "skill",
                "name": "review",
                "directive": "the parser",
                "text": "the parser\n\nthe skill's body",
            })
        );
        assert_eq!(
            views[3],
            serde_json::json!({
                "type": "resource",
                "uri": "file:///a.rs",
                "name": "a.rs",
                "text": "<resource uri=\"file:///a.rs\">fn a() {}</resource>",
            })
        );
    }

    /// What Meka wrote on another's behalf keeps its kind on the wire, with the facts a client
    /// labels it by: an inbox item's words beside its class and source, a fire's job and words, a
    /// report's tasks; and `text` on each is what the model read.
    #[test]
    fn the_view_types_what_meka_relayed() {
        let views: Vec<serde_json::Value> = [
            ContentBlock::Inbox {
                item_id: None,
                class: Some(crate::conversation::InboxClass::Steer),
                source: Some("telegram".to_string()),
                body: "the words".to_string(),
                text: "[Message from telegram, arrived now]\nthe words".to_string(),
            },
            ContentBlock::Scheduled {
                job_id: "job".to_string(),
                prompt: "stretch".to_string(),
                text: "[Scheduled job job fired now]\n\nstretch".to_string(),
            },
            ContentBlock::BackgroundReport {
                task_ids: vec!["task".to_string()],
                text: "[Background task reporting at now]\nok".to_string(),
            },
        ]
        .iter()
        .map(|block| serde_json::to_value(view_for_block(block)).expect("json"))
        .collect();
        assert_eq!(
            views[0],
            serde_json::json!({
                "type": "inbox",
                "class": "steer",
                "source": "telegram",
                "body": "the words",
                "text": "[Message from telegram, arrived now]\nthe words",
            })
        );
        assert_eq!(views[1]["type"], "scheduled");
        assert_eq!(views[1]["job_id"], "job");
        assert_eq!(views[1]["prompt"], "stretch");
        assert_eq!(views[2]["type"], "background_report");
        assert_eq!(views[2]["task_ids"], serde_json::json!(["task"]));
    }

    /// No header is no precondition; the current tag passes; a tag from before a write is refused
    /// with the current position; a weak tag never matches a state-changing request; `*` matches
    /// any conversation.
    #[test]
    fn the_precondition_accepts_the_current_tag_and_refuses_a_stale_one() {
        let mut headers = HeaderMap::new();
        assert!(check_if_match(&headers, 42, Uuid::nil()).is_ok());

        headers.insert(header::IF_MATCH, conversation_etag(42));
        assert!(check_if_match(&headers, 42, Uuid::nil()).is_ok());
        let stale = check_if_match(&headers, 43, Uuid::nil()).expect_err("a write since");
        assert_eq!(stale.status, 412);
        assert_eq!(stale.type_uri, ErrorKind::PreconditionFailed.type_uri());
        assert_eq!(stale.extensions["position"], 43);

        headers.insert(header::IF_MATCH, HeaderValue::from_static("W/\"42\""));
        assert!(check_if_match(&headers, 42, Uuid::nil()).is_err());

        headers.insert(header::IF_MATCH, HeaderValue::from_static("*"));
        assert!(check_if_match(&headers, 9, Uuid::nil()).is_ok());
    }
}
