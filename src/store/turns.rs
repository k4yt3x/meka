//! The `turns` table: one row per turn that began, however it ended, and the id every message
//! row a turn added carries. A turn is the unit a client reasons about (the feed names it, the
//! history groups by it, usage and outcome belong to it), so it is a row rather than something
//! each reader derives for itself.

use std::sync::Arc;

use uuid::Uuid;

use crate::error::{MekaError, Result};

/// The turns table, on a store's connection.
pub(crate) struct TurnStore {
    connection: Arc<tokio_rusqlite::Connection>,
}

/// One turn as its row records it: when it began and, once it has, how it ended.
///
/// `ended_at` and `status` are absent together on a turn that began and has not ended: the one
/// in flight, or one the process died under. Whether it is the former is `turn_in_flight` on
/// the session, a live fact no row stores.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
pub(crate) struct TurnRecord {
    pub(crate) id: Uuid,
    /// Who opened it: `client`, `inbox`, `schedule`, `background`, or `parent` for a sub-agent's.
    pub(crate) source: String,
    /// RFC 3339, when the turn began.
    pub(crate) started_at: String,
    /// RFC 3339, when the turn ended.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) ended_at: Option<String>,
    #[cfg_attr(feature = "serve", schema(value_type = Option<String>))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) status: Option<TurnStatus>,
    /// `end_turn`, `max_tokens` or `refusal`, on a turn that succeeded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stop_reason: Option<String>,
    /// What a turn that failed failed on, in the words a Problem Detail uses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<TurnError>,
    /// What the turn spent, every round of it.
    pub(crate) usage: TurnUsage,
}

/// The failure a turn ended on: its `type`, as every Problem Detail names it, and meka's own
/// sentence about it. Never the provider's text, which `[serve] relay_provider_errors` decides
/// about per deployment and a stored copy would hand to every reader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TurnError {
    pub(crate) kind: crate::error::ErrorKind,
    pub(crate) detail: String,
}

impl serde::Serialize for TurnError {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut error = serializer.serialize_struct("TurnError", 2)?;
        error.serialize_field("type", self.kind.type_uri())?;
        error.serialize_field("detail", &self.detail)?;
        error.end()
    }
}

#[cfg(feature = "serve")]
impl utoipa::PartialSchema for TurnError {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        use utoipa::openapi::schema::{ObjectBuilder, Type};
        ObjectBuilder::new()
            .property(
                "type",
                ObjectBuilder::new()
                    .schema_type(Type::String)
                    .description(Some("The error's `type`, as a Problem Detail names it")),
            )
            .required("type")
            .property(
                "detail",
                ObjectBuilder::new()
                    .schema_type(Type::String)
                    .description(Some("meka's own sentence about the failure")),
            )
            .required("detail")
            .into()
    }
}

#[cfg(feature = "serve")]
impl utoipa::ToSchema for TurnError {}

/// The tokens a turn spent, as the provider reported them round by round.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
pub(crate) struct TurnUsage {
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) cache_creation_input_tokens: u64,
    pub(crate) cache_read_input_tokens: u64,
}

impl From<&crate::stats::TokenUsage> for TurnUsage {
    fn from(usage: &crate::stats::TokenUsage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cache_creation_input_tokens: usage.cache_creation_input_tokens,
            cache_read_input_tokens: usage.cache_read_input_tokens,
        }
    }
}

/// How a turn ended, in the three words the terminal events use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(into = "String")]
pub(crate) enum TurnStatus {
    Succeeded,
    Failed,
    Canceled,
}

impl TurnStatus {
    const ALL: [Self; 3] = [Self::Succeeded, Self::Failed, Self::Canceled];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Canceled => "canceled",
        }
    }
}

impl std::fmt::Display for TurnStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for TurnStatus {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|status| status.name() == value)
            .ok_or_else(|| {
                crate::text::unknown_name("turn status", value, Self::ALL.map(Self::name))
            })
    }
}

impl From<TurnStatus> for String {
    fn from(status: TurnStatus) -> Self {
        status.name().to_string()
    }
}

/// How a turn ended, for the row that opened it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TurnEnding {
    pub(crate) status: TurnStatus,
    pub(crate) stop_reason: Option<String>,
    pub(crate) error: Option<TurnError>,
    pub(crate) usage: TurnUsage,
}

impl TurnEnding {
    /// A turn that ran to its stop reason.
    pub(crate) fn succeeded(stop_reason: &str, usage: TurnUsage) -> Self {
        Self {
            status: TurnStatus::Succeeded,
            stop_reason: Some(stop_reason.to_string()),
            error: None,
            usage,
        }
    }

    /// A turn that ended on an error, recorded under the one classification the HTTP surface
    /// reports it by, so the row and the terminal event name the failure the same way.
    pub(crate) fn failed(error: &MekaError, usage: TurnUsage) -> Self {
        Self {
            status: TurnStatus::Failed,
            stop_reason: None,
            error: Some(TurnError {
                kind: error.kind(),
                detail: error.sentence(),
            }),
            usage,
        }
    }

    /// A turn stopped before it ended, by a client, a shutdown or a lost reader.
    pub(crate) fn canceled(usage: TurnUsage) -> Self {
        Self {
            status: TurnStatus::Canceled,
            stop_reason: None,
            error: None,
            usage,
        }
    }
}

/// The columns of one turn, aliased `t`, in the order [`turn_from_row`] reads them.
pub(super) const TURN_COLUMNS: &str = "t.id, t.source, t.started_at, t.ended_at, t.status, \
     t.stop_reason, t.error_type, t.detail, t.input_tokens, t.output_tokens, \
     t.cache_creation_input_tokens, t.cache_read_input_tokens";

/// The join that puts a session's latest turn beside its row as `t`, every column NULL for a
/// session no turn has begun on. The latest by when it began, whatever its state: a turn the
/// process died under is the latest fact about the session, not something to hide behind the
/// one before it.
pub(super) const LATEST_TURN_JOIN: &str = "LEFT JOIN turns t ON t.id = (SELECT id FROM turns \
     WHERE session_id = s.id ORDER BY started_at DESC, id DESC LIMIT 1)";

/// One turn from the [`TURN_COLUMNS`] starting at column `offset`, or `None` when the join
/// found no turn. A status word meka cannot read is a row nothing here wrote, and is warned
/// about rather than invented.
pub(super) fn turn_from_row(
    row: &rusqlite::Row<'_>,
    offset: usize,
) -> rusqlite::Result<Option<TurnRecord>> {
    let Some(id) = row.get::<_, Option<String>>(offset)? else {
        return Ok(None);
    };
    let id = Uuid::parse_str(&id)
        .map_err(|error| rusqlite::Error::InvalidParameterName(error.to_string()))?;
    let status = match row.get::<_, Option<String>>(offset + 4)? {
        Some(word) => match word.parse::<TurnStatus>() {
            Ok(status) => Some(status),
            Err(error) => {
                tracing::warn!("turn {id} records an ending meka cannot read: {error}");
                None
            }
        },
        None => None,
    };
    let error = match (
        row.get::<_, Option<String>>(offset + 6)?,
        row.get::<_, Option<String>>(offset + 7)?,
    ) {
        (Some(name), Some(detail)) => match crate::error::ErrorKind::from_name(&name) {
            Some(kind) => Some(TurnError { kind, detail }),
            None => {
                tracing::warn!("turn {id} records an error type meka cannot read: {name}");
                None
            }
        },
        _ => None,
    };
    Ok(Some(TurnRecord {
        id,
        source: row.get(offset + 1)?,
        started_at: row.get(offset + 2)?,
        ended_at: row.get(offset + 3)?,
        status,
        stop_reason: row.get(offset + 5)?,
        error,
        usage: TurnUsage {
            input_tokens: row.get::<_, i64>(offset + 8)?.max(0) as u64,
            output_tokens: row.get::<_, i64>(offset + 9)?.max(0) as u64,
            cache_creation_input_tokens: row.get::<_, i64>(offset + 10)?.max(0) as u64,
            cache_read_input_tokens: row.get::<_, i64>(offset + 11)?.max(0) as u64,
        },
    }))
}

impl TurnStore {
    /// A handle on the table over `connection`.
    pub(crate) fn new(connection: Arc<tokio_rusqlite::Connection>) -> Self {
        Self { connection }
    }

    /// Record that a turn began; the rows the turn writes name it by this id, and
    /// [`Self::close_turn`] records how it ended.
    pub(crate) async fn open_turn(
        &self,
        session_id: Uuid,
        turn_id: Uuid,
        source: &str,
        started_at: &str,
    ) -> Result<()> {
        let source = source.to_string();
        let started_at = started_at.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO turns (id, session_id, source, started_at) VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![
                        turn_id.to_string(),
                        session_id.to_string(),
                        source,
                        started_at
                    ],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to open the turn: {error}")))
    }

    /// Every turn of a session, oldest first: what an archive carries.
    pub(crate) async fn load_turns(&self, session_id: Uuid) -> Result<Vec<TurnRecord>> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(&format!(
                    "SELECT {TURN_COLUMNS} FROM turns t
                     WHERE t.session_id = ?1
                     ORDER BY t.started_at ASC, t.id ASC"
                ))?;
                let turns = statement
                    .query_map(rusqlite::params![session_id.to_string()], |row| {
                        turn_from_row(row, 0)
                    })?
                    .collect::<rusqlite::Result<Vec<Option<TurnRecord>>>>()?;
                Ok(turns.into_iter().flatten().collect::<Vec<TurnRecord>>())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to load turns: {error}")))
    }

    /// Record how a turn ended.
    pub(crate) async fn close_turn(&self, turn_id: Uuid, ending: &TurnEnding) -> Result<()> {
        let ending = ending.clone();
        let ended_at = chrono::Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE turns SET
                         ended_at = ?2,
                         status = ?3,
                         stop_reason = ?4,
                         error_type = ?5,
                         detail = ?6,
                         input_tokens = ?7,
                         output_tokens = ?8,
                         cache_creation_input_tokens = ?9,
                         cache_read_input_tokens = ?10
                     WHERE id = ?1",
                    rusqlite::params![
                        turn_id.to_string(),
                        ended_at,
                        ending.status.name(),
                        ending.stop_reason,
                        ending.error.as_ref().map(|error| error.kind.name()),
                        ending.error.as_ref().map(|error| error.detail.clone()),
                        ending.usage.input_tokens as i64,
                        ending.usage.output_tokens as i64,
                        ending.usage.cache_creation_input_tokens as i64,
                        ending.usage.cache_read_input_tokens as i64,
                    ],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to close the turn: {error}")))
    }

    /// A session's turns, newest first: at most `limit` of them, from the one before `before`
    /// when a cursor is given, and the id to pass back as `before` when more remain.
    pub(crate) async fn list_turns(
        &self,
        session_id: Uuid,
        limit: u32,
        before: Option<Uuid>,
    ) -> Result<(Vec<TurnRecord>, Option<Uuid>)> {
        let fetch = i64::from(limit).saturating_add(1);
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(&format!(
                    "SELECT {TURN_COLUMNS} FROM turns t
                     WHERE t.session_id = ?1
                       AND (?2 IS NULL OR (t.started_at, t.id) <
                           (SELECT started_at, id FROM turns WHERE id = ?2))
                     ORDER BY t.started_at DESC, t.id DESC
                     LIMIT ?3"
                ))?;
                let turns = statement
                    .query_map(
                        rusqlite::params![
                            session_id.to_string(),
                            before.map(|id| id.to_string()),
                            fetch
                        ],
                        |row| turn_from_row(row, 0),
                    )?
                    .collect::<rusqlite::Result<Vec<Option<TurnRecord>>>>()?;
                Ok(turns.into_iter().flatten().collect::<Vec<TurnRecord>>())
            })
            .await
            .map(|mut turns| {
                let next = if turns.len() > limit as usize {
                    turns.truncate(limit as usize);
                    turns.last().map(|turn| turn.id)
                } else {
                    None
                };
                (turns, next)
            })
            .map_err(|error| MekaError::Database(format!("failed to list turns: {error}")))
    }
}

/// Write a turn an archive carries, in the caller's transaction, as the row it was: the import's
/// door, which already holds the session's own transaction.
pub(super) fn insert_turn_row(
    transaction: &rusqlite::Transaction<'_>,
    session_id: &str,
    turn: &super::export::ExportedTurn,
) -> rusqlite::Result<()> {
    transaction.execute(
        "INSERT INTO turns (id, session_id, source, started_at, ended_at, status, stop_reason, \
         error_type, detail, input_tokens, output_tokens, cache_creation_input_tokens, \
         cache_read_input_tokens)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
        rusqlite::params![
            turn.id,
            session_id,
            turn.source,
            turn.started_at,
            turn.ended_at,
            turn.status,
            turn.stop_reason,
            turn.error_type,
            turn.detail,
            turn.usage.input_tokens as i64,
            turn.usage.output_tokens as i64,
            turn.usage.cache_creation_input_tokens as i64,
            turn.usage.cache_read_input_tokens as i64,
        ],
    )?;
    Ok(())
}
