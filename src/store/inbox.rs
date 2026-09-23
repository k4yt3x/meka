//! The session inbox: the `inbox_items` table.
//!
//! A message for a session from outside its turn: an HTTP client, or a parent agent steering a
//! worker. A `steer` is read by the running turn at its next round boundary; an `interrupt` cuts
//! the answer the turn is streaming and is read at the boundary otherwise; a `followup` waits for
//! the turn to end. Any of them rides the opening of the next turn, or opens one when nothing is
//! running.
//!
//! An item is `pending` until its text is in the conversation log (`appended_at`), `appended` until
//! a request carrying it is accepted by the provider (`delivered_at`), and `withdrawn` when it was
//! taken back or given up on. `appended_at` is stamped by the same transaction that writes the
//! conversation row, in [`stamp_appended`], never on its own: a stamp without the row would make
//! the store claim the model was shown words it never saw.

use chrono::{DateTime, Utc};
use rusqlite::OptionalExtension;

use super::*;

/// How an item reaches the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "String", into = "String")]
pub(crate) enum InboxClass {
    /// Injected into the running turn at its next round boundary.
    Steer,
    /// Waits for the running turn to end.
    Followup,
    /// Cuts the answer the running turn is streaming and is read right after what streamed; read
    /// at the round boundary, like a steer, while tools run.
    Interrupt,
}

impl InboxClass {
    /// Every value, in the order a refusal lists them.
    pub(crate) const ALL: [InboxClass; 3] = [Self::Steer, Self::Followup, Self::Interrupt];

    /// The one spelling of this value: the `class` column and the HTTP body.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Steer => "steer",
            Self::Followup => "followup",
            Self::Interrupt => "interrupt",
        }
    }

    /// The `class IN (...)` clause selecting `classes`, or nothing for every class. Built from
    /// `name()`, which is why no value is bound.
    fn clause(classes: &[InboxClass]) -> String {
        if classes.is_empty() {
            return String::new();
        }
        let names: Vec<String> = classes
            .iter()
            .map(|class| format!("'{}'", class.name()))
            .collect();
        format!(" AND class IN ({})", names.join(", "))
    }

    fn supported() -> String {
        Self::ALL
            .iter()
            .map(|class| class.name())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl std::fmt::Display for InboxClass {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl std::str::FromStr for InboxClass {
    type Err = String;

    /// Refuses with the names that would have been accepted.
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|class| class.name() == value)
            .ok_or_else(|| {
                format!(
                    "'{value}' is not an inbox class. Supported: {}",
                    Self::supported()
                )
            })
    }
}

impl TryFrom<String> for InboxClass {
    type Error = String;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<InboxClass> for String {
    fn from(class: InboxClass) -> Self {
        class.name().to_string()
    }
}

/// Where an item stands, derived from its stamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InboxState {
    Pending,
    Appended,
    Delivered,
    Withdrawn,
}

impl InboxState {
    /// The one spelling of this value: the HTTP view.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Appended => "appended",
            Self::Delivered => "delivered",
            Self::Withdrawn => "withdrawn",
        }
    }
}

/// One row of the inbox.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InboxItem {
    pub(crate) id: Uuid,
    pub(crate) session_id: Uuid,
    pub(crate) class: InboxClass,
    /// Who sent it, as the header names them: what the client said, or `your parent agent`.
    /// `None` when the client named nobody, and the header then names nobody: who sent a message
    /// is the client's fact, not one meka guesses from the token that carried it.
    pub(crate) source: Option<String>,
    pub(crate) body: String,
    pub(crate) created_at: DateTime<Utc>,
    /// Turns that opened on this item and failed before the provider accepted anything.
    pub(crate) attempts: u32,
    pub(crate) appended_at: Option<DateTime<Utc>>,
    pub(crate) delivered_at: Option<DateTime<Utc>>,
    pub(crate) withdrawn_at: Option<DateTime<Utc>>,
    /// Why it was given up on, when it was.
    pub(crate) failure: Option<String>,
}

impl InboxItem {
    /// Where the item is in its life, read off its stamps: `pending` until its text is in the
    /// conversation, `appended` until a provider accepted a request carrying it, then
    /// `delivered`; `withdrawn` once taken back, whatever it was before.
    pub(crate) fn state(&self) -> InboxState {
        if self.withdrawn_at.is_some() {
            InboxState::Withdrawn
        } else if self.delivered_at.is_some() {
            InboxState::Delivered
        } else if self.appended_at.is_some() {
            InboxState::Appended
        } else {
            InboxState::Pending
        }
    }
}

/// An item about to be enqueued, validated once for every door.
#[derive(Debug, Clone)]
pub(crate) struct NewInboxItem {
    pub(crate) session_id: Uuid,
    pub(crate) class: InboxClass,
    pub(crate) source: Option<String>,
    pub(crate) body: String,
    /// The bearer token that enqueued it, which scopes its idempotency key.
    pub(crate) token_id: Option<String>,
    pub(crate) idempotency_key: Option<String>,
}

impl NewInboxItem {
    /// Refuses a blank body: it costs a provider round-trip to deliver nothing, and the model
    /// cannot tell it from a message whose content went missing.
    pub(crate) fn from_parts(
        session_id: Uuid,
        class: InboxClass,
        source: Option<String>,
        body: impl Into<String>,
    ) -> Result<Self> {
        let body = body.into();
        if body.trim().is_empty() {
            return Err(MekaError::EmptyPrompt);
        }
        Ok(Self {
            session_id,
            class,
            source,
            body,
            token_id: None,
            idempotency_key: None,
        })
    }

    /// Scope a replayable submission to the token that made it.
    pub(crate) fn idempotent(mut self, token_id: String, key: String) -> Self {
        self.token_id = Some(token_id);
        self.idempotency_key = Some(key);
        self
    }
}

/// What [`InboxStore::enqueue`] answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Enqueued {
    pub(crate) id: Uuid,
    /// The key had been used before on this session, so the row is the earlier one.
    pub(crate) replayed: bool,
}

/// What a client's withdrawal found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Withdrawal {
    Withdrawn,
    /// The text is already in the conversation; only a turn can answer it now.
    AlreadyAppended,
    /// Delivered or withdrawn before, which is the same "nothing to take back" from outside.
    Closed,
    Missing,
}

/// The inbox's slice of the session database, handed out by
/// [`crate::store::Store::inbox_store`].
#[derive(Clone)]
pub(crate) struct InboxStore {
    pub(crate) connection: std::sync::Arc<tokio_rusqlite::Connection>,
}

const COLUMNS: &str = "id, session_id, class, source, body, created_at, attempts, appended_at, \
                       delivered_at, withdrawn_at, failure";

/// A row is offered to a turn while its text is not yet in the log, it has not been taken back,
/// and any retry wait has passed.
const PENDING: &str = "appended_at IS NULL AND withdrawn_at IS NULL \
                       AND (not_before IS NULL OR not_before <= ?2)";

impl InboxStore {
    /// A handle on the table over `connection`.
    pub(crate) fn new(connection: std::sync::Arc<tokio_rusqlite::Connection>) -> Self {
        Self { connection }
    }

    /// Record an item. A key already used on this session by the same token answers the earlier
    /// row rather than a second one, so a client that retries across a meka restart enqueues once.
    pub(crate) async fn enqueue(&self, item: NewInboxItem) -> Result<Enqueued> {
        let id = Uuid::new_v4();
        let created_at = Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<Enqueued> {
                // Immediate, because the read decides the write: a deferred transaction
                // upgrading under a second process's write would fail rather than wait.
                let transaction = connection
                    .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                if let (Some(token_id), Some(key)) = (&item.token_id, &item.idempotency_key) {
                    let earlier: Option<String> = transaction
                        .query_row(
                            "SELECT id FROM inbox_items WHERE session_id = ?1 AND token_id = ?2 \
                             AND idempotency_key = ?3",
                            rusqlite::params![item.session_id.to_string(), token_id, key],
                            |row| row.get(0),
                        )
                        .optional()?;
                    if let Some(earlier) = earlier {
                        transaction.commit()?;
                        let id = Uuid::parse_str(&earlier).map_err(|error| {
                            rusqlite::Error::FromSqlConversionFailure(
                                0,
                                rusqlite::types::Type::Text,
                                Box::new(error),
                            )
                        })?;
                        return Ok(Enqueued { id, replayed: true });
                    }
                }
                transaction.execute(
                    "INSERT INTO inbox_items \
                     (id, session_id, class, source, body, token_id, idempotency_key, created_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    rusqlite::params![
                        id.to_string(),
                        item.session_id.to_string(),
                        item.class.name(),
                        // The column was created NOT NULL and only this module reads it, so an
                        // empty string stands for nobody rather than a rebuild of the table.
                        item.source.unwrap_or_default(),
                        item.body,
                        item.token_id,
                        item.idempotency_key,
                        created_at,
                    ],
                )?;
                transaction.commit()?;
                Ok(Enqueued {
                    id,
                    replayed: false,
                })
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to enqueue inbox item: {error}")))
    }

    /// One item by id.
    pub(crate) async fn get(&self, id: Uuid) -> Result<Option<InboxItem>> {
        let rows = self
            .query(
                &format!("SELECT {COLUMNS} FROM inbox_items WHERE id = ?1"),
                vec![id.to_string()],
            )
            .await?;
        Ok(rows.into_iter().next())
    }

    /// A session's items the model has not been shown a request for, oldest first.
    pub(crate) async fn list_open(&self, session_id: Uuid) -> Result<Vec<InboxItem>> {
        self.query(
            &format!(
                "SELECT {COLUMNS} FROM inbox_items WHERE session_id = ?1 \
                 AND delivered_at IS NULL AND withdrawn_at IS NULL ORDER BY created_at, rowid"
            ),
            vec![session_id.to_string()],
        )
        .await
    }

    /// The items a turn may append now, oldest first; every class when `classes` is empty. Reads
    /// only: the stamp is written with the conversation row, by [`stamp_appended`].
    pub(crate) async fn take_pending(
        &self,
        session_id: Uuid,
        classes: &[InboxClass],
        now: DateTime<Utc>,
    ) -> Result<Vec<InboxItem>> {
        let class_clause = InboxClass::clause(classes);
        self.query(
            &format!(
                "SELECT {COLUMNS} FROM inbox_items WHERE session_id = ?1 AND {PENDING}{class_clause} \
                 ORDER BY created_at, rowid"
            ),
            vec![session_id.to_string(), now.to_rfc3339()],
        )
        .await
    }

    /// Whether an item of `class` other than `except` is waiting to be appended: what a turn asks
    /// once a second while a provider call is in flight, so it costs one indexed count and reads
    /// no row. `except` is what the turn already carries: an item whose stamp is waiting on the
    /// prompt's own save is still pending, and is not news.
    pub(crate) async fn has_pending(
        &self,
        session_id: Uuid,
        class: InboxClass,
        now: DateTime<Utc>,
        except: &[Uuid],
    ) -> Result<bool> {
        let session_id = session_id.to_string();
        let now = now.to_rfc3339();
        let class_clause = InboxClass::clause(&[class]);
        let except: Vec<String> = except.iter().map(Uuid::to_string).collect();
        self.connection
            .call(move |connection| -> rusqlite::Result<bool> {
                let mut statement = connection.prepare(&format!(
                    "SELECT id FROM inbox_items WHERE session_id = ?1 AND {PENDING}{class_clause}"
                ))?;
                let mut rows = statement.query(rusqlite::params![session_id, now])?;
                while let Some(row) = rows.next()? {
                    let id: String = row.get(0)?;
                    if !except.contains(&id) {
                        return Ok(true);
                    }
                }
                Ok(false)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to read the inbox: {error}")))
    }

    /// How many items are waiting to be appended, for the session view.
    pub(crate) async fn pending_count(&self, session_id: Uuid) -> Result<u64> {
        let session_id = session_id.to_string();
        let now = Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<i64> {
                connection.query_row(
                    &format!(
                        "SELECT COUNT(*) FROM inbox_items WHERE session_id = ?1 AND {PENDING}"
                    ),
                    rusqlite::params![session_id, now],
                    |row| row.get(0),
                )
            })
            .await
            .map(|count| count.max(0) as u64)
            .map_err(|error| MekaError::Database(format!("failed to count inbox items: {error}")))
    }

    /// Stamp every appended, undelivered item of a session as delivered, answering their ids.
    ///
    /// Called when the provider accepts a request: one turn runs per session, so everything
    /// appended and not yet delivered is in that request.
    pub(crate) async fn mark_delivered(&self, session_id: Uuid) -> Result<Vec<Uuid>> {
        let session_id = session_id.to_string();
        let delivered_at = Utc::now().to_rfc3339();
        let ids: Vec<String> = self
            .connection
            .call(move |connection| -> rusqlite::Result<Vec<String>> {
                let transaction = connection.transaction()?;
                let ids = {
                    let mut statement = transaction.prepare(
                        "SELECT id FROM inbox_items WHERE session_id = ?1 \
                         AND appended_at IS NOT NULL AND delivered_at IS NULL \
                         AND withdrawn_at IS NULL ORDER BY created_at, rowid",
                    )?;
                    let rows = statement.query_map([&session_id], |row| row.get::<_, String>(0))?;
                    rows.collect::<rusqlite::Result<Vec<String>>>()?
                };
                {
                    let mut statement = transaction
                        .prepare("UPDATE inbox_items SET delivered_at = ?2 WHERE id = ?1")?;
                    for id in &ids {
                        statement.execute(rusqlite::params![id, delivered_at])?;
                    }
                }
                transaction.commit()?;
                Ok(ids)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to mark inbox items delivered: {error}"))
            })?;
        Ok(ids
            .iter()
            .filter_map(|id| Uuid::parse_str(id).ok())
            .collect())
    }

    /// Stamp items as appended after a conversation rewrite carried their message to disk.
    ///
    /// The one stamp not written with the row: a compaction that runs before the first provider
    /// call persists the prompt inside its own transaction, which knows nothing of the inbox. The
    /// text is already on disk by the time this runs, so the stamp follows it rather than the
    /// other way round.
    pub(crate) async fn stamp_appended_after_rewrite(&self, ids: &[Uuid]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let ids: Vec<String> = ids.iter().map(Uuid::to_string).collect();
        let now = Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<()> {
                let transaction = connection.transaction()?;
                stamp_appended(&transaction, &ids, &now)?;
                transaction.commit()
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to stamp inbox items: {error}")))
    }

    /// Offer items again after the prompt that carried them left the log without reaching the
    /// store, not before `not_before`. The counterpart of [`stamp_appended`] for a withdrawal the
    /// store never saw; one it did see goes through
    /// [`crate::store::Store::save_event_resetting_inbox`], which resets inside the same
    /// transaction as the withdrawal row.
    pub(crate) async fn reset_pending(
        &self,
        ids: &[Uuid],
        not_before: DateTime<Utc>,
    ) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let ids: Vec<String> = ids.iter().map(Uuid::to_string).collect();
        let not_before = not_before.to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<()> {
                let transaction = connection.transaction()?;
                reset_pending_in(&transaction, &ids, &not_before)?;
                transaction.commit()
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to reset inbox items: {error}")))
    }

    /// Hold pending items back until `not_before` after a turn on them failed, counting the
    /// attempt. Rows already in the conversation are left alone: they are not waiting to be
    /// appended, and their turn did not fail on their account.
    pub(crate) async fn defer_pending(
        &self,
        ids: &[Uuid],
        not_before: DateTime<Utc>,
    ) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let ids: Vec<String> = ids.iter().map(Uuid::to_string).collect();
        let not_before = not_before.to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<()> {
                let transaction = connection.transaction()?;
                {
                    let mut statement = transaction.prepare(
                        "UPDATE inbox_items SET not_before = ?2, attempts = attempts + 1 \
                         WHERE id = ?1 AND appended_at IS NULL AND withdrawn_at IS NULL",
                    )?;
                    for id in &ids {
                        statement.execute(rusqlite::params![id, not_before])?;
                    }
                }
                transaction.commit()
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to defer inbox items: {error}")))
    }

    /// [`Self::defer_pending`] for everything pending on a session, when the session itself could
    /// not be brought up to run a turn on them.
    pub(crate) async fn defer_session_pending(
        &self,
        session_id: Uuid,
        not_before: DateTime<Utc>,
    ) -> Result<()> {
        let session_id = session_id.to_string();
        let now = Utc::now().to_rfc3339();
        let not_before = not_before.to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<()> {
                connection.execute(
                    &format!(
                        "UPDATE inbox_items SET not_before = ?3, attempts = attempts + 1 \
                         WHERE session_id = ?1 AND {PENDING}"
                    ),
                    rusqlite::params![session_id, now, not_before],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to defer inbox items: {error}")))
    }

    /// Take a pending item back, from outside or because meka gave up on it.
    pub(crate) async fn withdraw(&self, id: Uuid, reason: Option<String>) -> Result<Withdrawal> {
        let id = id.to_string();
        let withdrawn_at = Utc::now().to_rfc3339();
        self.connection
            .call(move |connection| -> rusqlite::Result<Withdrawal> {
                let transaction = connection
                    .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let stamps: Option<(Option<String>, Option<String>, Option<String>)> = transaction
                    .query_row(
                        "SELECT appended_at, delivered_at, withdrawn_at FROM inbox_items \
                         WHERE id = ?1",
                        [&id],
                        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                    )
                    .optional()?;
                let outcome = match stamps {
                    None => Withdrawal::Missing,
                    Some((_, Some(_), _) | (_, _, Some(_))) => Withdrawal::Closed,
                    Some((Some(_), None, None)) => Withdrawal::AlreadyAppended,
                    Some((None, None, None)) => {
                        transaction.execute(
                            "UPDATE inbox_items SET withdrawn_at = ?2, failure = ?3 WHERE id = ?1",
                            rusqlite::params![id, withdrawn_at, reason],
                        )?;
                        Withdrawal::Withdrawn
                    }
                };
                transaction.commit()?;
                Ok(outcome)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to withdraw inbox item: {error}")))
    }

    /// Root sessions with an item a turn may append now. A worker's turns are run by its parent,
    /// so its items wait for the parent's next follow-up rather than for a driver.
    pub(crate) async fn sessions_with_work(&self, now: DateTime<Utc>) -> Result<Vec<Uuid>> {
        let now = now.to_rfc3339();
        let ids: Vec<String> = self
            .connection
            .call(move |connection| -> rusqlite::Result<Vec<String>> {
                let mut statement = connection.prepare(&format!(
                    "SELECT DISTINCT session_id FROM inbox_items WHERE {PENDING} \
                     AND session_id IN (SELECT id FROM sessions WHERE parent_session_id IS NULL) \
                     ORDER BY session_id"
                ))?;
                // `?1` is unused by `PENDING`, which binds `?2`; a placeholder is still counted.
                let rows = statement.query_map(rusqlite::params![String::new(), now], |row| {
                    row.get::<_, String>(0)
                })?;
                rows.collect::<rusqlite::Result<Vec<String>>>()
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to list sessions with inbox work: {error}"))
            })?;
        Ok(ids
            .iter()
            .filter_map(|id| Uuid::parse_str(id).ok())
            .collect())
    }

    async fn query(&self, sql: &str, params: Vec<String>) -> Result<Vec<InboxItem>> {
        let sql = sql.to_string();
        let rows: Vec<InboxRow> = self
            .connection
            .call(move |connection| -> rusqlite::Result<Vec<InboxRow>> {
                let mut statement = connection.prepare(&sql)?;
                let rows = statement.query_map(rusqlite::params_from_iter(params), |row| {
                    Ok(InboxRow {
                        id: row.get(0)?,
                        session_id: row.get(1)?,
                        class: row.get(2)?,
                        source: row.get(3)?,
                        body: row.get(4)?,
                        created_at: row.get(5)?,
                        attempts: row.get(6)?,
                        appended_at: row.get(7)?,
                        delivered_at: row.get(8)?,
                        withdrawn_at: row.get(9)?,
                        failure: row.get(10)?,
                    })
                })?;
                rows.collect()
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to read inbox items: {error}")))?;
        rows.into_iter()
            .map(|row| {
                row.decode().map_err(|reason| {
                    MekaError::Database(format!("unreadable inbox row: {reason}"))
                })
            })
            .collect()
    }
}

/// Stamp `ids` as appended inside the transaction that writes their conversation row.
///
/// `pub(super)` rather than a method, because the transaction belongs to the sessions store: the
/// two writes have to commit together or not at all, and that is only possible from inside its
/// call.
pub(super) fn stamp_appended(
    transaction: &rusqlite::Transaction<'_>,
    ids: &[String],
    now: &str,
) -> rusqlite::Result<()> {
    if ids.is_empty() {
        return Ok(());
    }
    let mut statement = transaction
        .prepare("UPDATE inbox_items SET appended_at = ?2 WHERE id = ?1 AND appended_at IS NULL")?;
    for id in ids {
        statement.execute(rusqlite::params![id, now])?;
    }
    Ok(())
}

/// Offer `ids` again inside the transaction that takes their message out of the log: the
/// counterpart of [`stamp_appended`], and `pub(super)` for the same reason.
pub(super) fn reset_pending_in(
    transaction: &rusqlite::Transaction<'_>,
    ids: &[String],
    not_before: &str,
) -> rusqlite::Result<()> {
    if ids.is_empty() {
        return Ok(());
    }
    let mut statement = transaction.prepare(
        "UPDATE inbox_items SET appended_at = NULL, not_before = ?2 \
         WHERE id = ?1 AND delivered_at IS NULL AND withdrawn_at IS NULL",
    )?;
    for id in ids {
        statement.execute(rusqlite::params![id, not_before])?;
    }
    Ok(())
}

/// Offer every appended, undelivered item of a session again, inside the transaction that
/// removes the tail of its log. Those are exactly the items whose text has not yet been in an
/// accepted request, which is the tail a rewind takes.
pub(super) fn reset_appended_in(
    transaction: &rusqlite::Transaction<'_>,
    session_id: &str,
) -> rusqlite::Result<()> {
    transaction.execute(
        "UPDATE inbox_items SET appended_at = NULL, not_before = NULL WHERE session_id = ?1 \
         AND appended_at IS NOT NULL AND delivered_at IS NULL AND withdrawn_at IS NULL",
        [session_id],
    )?;
    Ok(())
}

/// A raw row, decoded outside the connection call so a bad row names itself.
struct InboxRow {
    id: String,
    session_id: String,
    class: String,
    source: String,
    body: String,
    created_at: String,
    attempts: i64,
    appended_at: Option<String>,
    delivered_at: Option<String>,
    withdrawn_at: Option<String>,
    failure: Option<String>,
}

impl InboxRow {
    fn decode(self) -> std::result::Result<InboxItem, String> {
        let stamp = |value: Option<String>| -> std::result::Result<Option<DateTime<Utc>>, String> {
            value
                .map(|text| {
                    DateTime::parse_from_rfc3339(&text)
                        .map(|at| at.with_timezone(&Utc))
                        .map_err(|error| format!("bad timestamp '{text}': {error}"))
                })
                .transpose()
        };
        Ok(InboxItem {
            id: Uuid::parse_str(&self.id).map_err(|error| format!("bad id: {error}"))?,
            session_id: Uuid::parse_str(&self.session_id)
                .map_err(|error| format!("bad session id: {error}"))?,
            class: self.class.parse()?,
            source: (!self.source.is_empty()).then_some(self.source),
            body: self.body,
            created_at: stamp(Some(self.created_at))?.ok_or("missing created_at")?,
            attempts: u32::try_from(self.attempts.max(0)).unwrap_or(u32::MAX),
            appended_at: stamp(self.appended_at)?,
            delivered_at: stamp(self.delivered_at)?,
            withdrawn_at: stamp(self.withdrawn_at)?,
            failure: self.failure,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn store_with_session() -> (Store, Uuid) {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        (store, session_id)
    }

    fn item(session_id: Uuid, class: InboxClass, body: &str) -> NewInboxItem {
        NewInboxItem::from_parts(session_id, class, Some("test".to_string()), body)
            .expect("a non-blank body")
    }

    #[tokio::test]
    async fn a_replayed_key_answers_the_first_row_and_enqueues_nothing() {
        let (store, session_id) = store_with_session().await;
        let inbox = store.inbox_store();
        let first = inbox
            .enqueue(
                item(session_id, InboxClass::Steer, "hello")
                    .idempotent("token".to_string(), "k1".to_string()),
            )
            .await
            .expect("enqueue");
        let again = inbox
            .enqueue(
                item(session_id, InboxClass::Steer, "hello")
                    .idempotent("token".to_string(), "k1".to_string()),
            )
            .await
            .expect("enqueue again");
        assert_eq!(again.id, first.id);
        assert!(again.replayed && !first.replayed);
        assert_eq!(inbox.list_open(session_id).await.expect("list").len(), 1);

        // The same key from another token is another submission.
        let other = inbox
            .enqueue(
                item(session_id, InboxClass::Steer, "hello")
                    .idempotent("other".to_string(), "k1".to_string()),
            )
            .await
            .expect("enqueue from another token");
        assert_ne!(other.id, first.id);
    }

    /// The column holds an empty string for an item nobody was named on, and only this module
    /// knows that: a reader gets `None` back, never the empty name.
    #[tokio::test]
    async fn an_item_named_by_nobody_reads_back_with_no_source() {
        let (store, session_id) = store_with_session().await;
        let inbox = store.inbox_store();
        let unnamed = inbox
            .enqueue(
                NewInboxItem::from_parts(session_id, InboxClass::Steer, None, "who's there?")
                    .expect("item"),
            )
            .await
            .expect("enqueue");
        let named = inbox
            .enqueue(item(session_id, InboxClass::Steer, "me"))
            .await
            .expect("enqueue");
        let unnamed = inbox.get(unnamed.id).await.expect("get").expect("row");
        assert_eq!(unnamed.source, None);
        let named = inbox.get(named.id).await.expect("get").expect("row");
        assert_eq!(named.source, Some("test".to_string()));
    }

    #[tokio::test]
    async fn a_blank_body_is_refused_before_it_reaches_the_store() {
        let session_id = Uuid::new_v4();
        assert!(matches!(
            NewInboxItem::from_parts(session_id, InboxClass::Steer, None, "  \n"),
            Err(MekaError::EmptyPrompt)
        ));
    }

    #[tokio::test]
    async fn pending_items_come_oldest_first_and_honor_their_class_and_wait() {
        let (store, session_id) = store_with_session().await;
        let inbox = store.inbox_store();
        let steer = inbox
            .enqueue(item(session_id, InboxClass::Steer, "first"))
            .await
            .expect("enqueue")
            .id;
        let followup = inbox
            .enqueue(item(session_id, InboxClass::Followup, "second"))
            .await
            .expect("enqueue")
            .id;
        let now = Utc::now();

        let interrupt = inbox
            .enqueue(item(session_id, InboxClass::Interrupt, "third"))
            .await
            .expect("enqueue")
            .id;
        let all = inbox
            .take_pending(session_id, &[], now)
            .await
            .expect("pending");
        assert_eq!(all.iter().map(|item| item.id).collect::<Vec<_>>(), vec![
            steer, followup, interrupt
        ]);
        let boundary = inbox
            .take_pending(session_id, &[InboxClass::Steer, InboxClass::Interrupt], now)
            .await
            .expect("pending at a boundary");
        assert_eq!(
            boundary.iter().map(|item| item.id).collect::<Vec<_>>(),
            vec![steer, interrupt]
        );
        assert!(
            inbox
                .has_pending(session_id, InboxClass::Interrupt, now, &[])
                .await
                .expect("has pending")
        );
        assert!(
            !inbox
                .has_pending(session_id, InboxClass::Interrupt, now, &[interrupt])
                .await
                .expect("has pending"),
            "an item the turn already carries is not news"
        );
        inbox
            .withdraw(interrupt, None)
            .await
            .expect("withdraw the interrupt");
        assert!(
            !inbox
                .has_pending(session_id, InboxClass::Interrupt, now, &[])
                .await
                .expect("has pending")
        );

        // A deferred item waits out `not_before`, then is offered again with the attempt counted.
        inbox
            .defer_pending(&[steer], now + chrono::Duration::minutes(5))
            .await
            .expect("defer");
        let soon = inbox
            .take_pending(session_id, &[], now)
            .await
            .expect("pending");
        assert_eq!(soon.iter().map(|item| item.id).collect::<Vec<_>>(), vec![
            followup
        ]);
        let later = inbox
            .take_pending(session_id, &[], now + chrono::Duration::minutes(6))
            .await
            .expect("pending later");
        assert_eq!(later.len(), 2);
        assert_eq!(later[0].attempts, 1, "the deferral counted an attempt");
        // A session nobody can bring up defers everything it has pending now, counted the same
        // way; the steer, already waiting on its own deferral, is not pending and is left alone.
        inbox
            .defer_session_pending(session_id, now + chrono::Duration::minutes(10))
            .await
            .expect("defer the session");
        let soon = inbox
            .take_pending(session_id, &[], now + chrono::Duration::minutes(6))
            .await
            .expect("pending");
        assert_eq!(soon.iter().map(|item| item.id).collect::<Vec<_>>(), vec![
            steer
        ]);
        let later = inbox
            .take_pending(session_id, &[], now + chrono::Duration::minutes(11))
            .await
            .expect("pending");
        assert_eq!(
            later
                .iter()
                .map(|item| (item.id, item.attempts))
                .collect::<Vec<_>>(),
            vec![(steer, 1), (followup, 1)]
        );
    }

    /// A withdrawal written with the reset, and a rewind, both take an item's text out of the log
    /// and offer the item again in the same write; a delivered item is history and stays.
    #[tokio::test]
    async fn taking_an_items_text_out_of_the_log_offers_it_again_in_the_same_write() {
        let (store, session_id) = store_with_session().await;
        let inbox = store.inbox_store();
        let carried = inbox
            .enqueue(item(session_id, InboxClass::Steer, "carried"))
            .await
            .expect("enqueue")
            .id;
        let prompt = crate::conversation::Message::user_turn("", "hello", Vec::new());
        store
            .save_event_marking_inbox(session_id, &crate::conversation::Event::Append(prompt), &[
                carried,
            ])
            .await
            .expect("save with stamp");
        let not_before = Utc::now() + chrono::Duration::seconds(30);
        store
            .save_event_resetting_inbox(
                session_id,
                &crate::conversation::Event::Repair {
                    replaced_count: 1,
                    messages: Vec::new(),
                },
                &[carried],
                not_before,
            )
            .await
            .expect("withdraw with reset");
        let row = inbox.get(carried).await.expect("get").expect("row");
        assert_eq!(row.state(), InboxState::Pending);
        assert!(
            inbox
                .take_pending(session_id, &[], Utc::now())
                .await
                .expect("pending")
                .is_empty(),
            "held back until the driver's wait has passed"
        );
        assert_eq!(
            store.load_events(session_id).await.expect("events").len(),
            2,
            "the withdrawal row is there"
        );

        // A rewind offers everything appended and undelivered again, and nothing delivered.
        let delivered = inbox
            .enqueue(item(session_id, InboxClass::Steer, "delivered"))
            .await
            .expect("enqueue")
            .id;
        let undelivered = inbox
            .enqueue(item(session_id, InboxClass::Followup, "undelivered"))
            .await
            .expect("enqueue")
            .id;
        store
            .save_event_marking_inbox(
                session_id,
                &crate::conversation::Event::Append(crate::conversation::Message::user_turn(
                    "",
                    "again",
                    Vec::new(),
                )),
                &[delivered],
            )
            .await
            .expect("save");
        inbox.mark_delivered(session_id).await.expect("deliver");
        store
            .save_event_marking_inbox(
                session_id,
                &crate::conversation::Event::Append(crate::conversation::Message::user_turn(
                    "",
                    "later",
                    Vec::new(),
                )),
                &[undelivered],
            )
            .await
            .expect("save");
        store
            .save_rewind(session_id, &crate::conversation::Event::Repair {
                replaced_count: 1,
                messages: Vec::new(),
            })
            .await
            .expect("rewind");
        assert_eq!(
            inbox
                .get(undelivered)
                .await
                .expect("get")
                .expect("row")
                .state(),
            InboxState::Pending,
            "its text left the log before any request carried it"
        );
        assert_eq!(
            inbox
                .get(delivered)
                .await
                .expect("get")
                .expect("row")
                .state(),
            InboxState::Delivered,
            "the model read it; a rewind does not unsay that"
        );
    }

    #[tokio::test]
    async fn delivery_stamps_only_what_was_appended_and_withdrawal_only_what_was_not() {
        let (store, session_id) = store_with_session().await;
        let inbox = store.inbox_store();
        let appended = inbox
            .enqueue(item(session_id, InboxClass::Steer, "in the log"))
            .await
            .expect("enqueue")
            .id;
        let waiting = inbox
            .enqueue(item(session_id, InboxClass::Steer, "not yet"))
            .await
            .expect("enqueue")
            .id;
        // The stamp travels with a conversation write; here the write is a user message.
        store
            .save_event_marking_inbox(
                session_id,
                &crate::conversation::Event::Append(crate::conversation::Message::user_turn(
                    "",
                    "in the log",
                    Vec::new(),
                )),
                &[appended],
            )
            .await
            .expect("save with stamp");
        assert_eq!(
            inbox
                .get(appended)
                .await
                .expect("get")
                .expect("row")
                .state(),
            InboxState::Appended
        );
        assert_eq!(
            inbox.get(waiting).await.expect("get").expect("row").state(),
            InboxState::Pending
        );

        let delivered = inbox.mark_delivered(session_id).await.expect("deliver");
        assert_eq!(delivered, vec![appended]);
        assert!(
            inbox
                .mark_delivered(session_id)
                .await
                .expect("deliver again")
                .is_empty(),
            "a delivered item is not delivered twice"
        );
        assert_eq!(inbox.pending_count(session_id).await.expect("count"), 1);

        assert_eq!(
            inbox.withdraw(appended, None).await.expect("withdraw"),
            Withdrawal::Closed
        );
        assert_eq!(
            inbox
                .withdraw(waiting, Some("gave up".to_string()))
                .await
                .expect("withdraw"),
            Withdrawal::Withdrawn
        );
        assert_eq!(
            inbox
                .withdraw(Uuid::new_v4(), None)
                .await
                .expect("withdraw"),
            Withdrawal::Missing
        );
        let row = inbox.get(waiting).await.expect("get").expect("row");
        assert_eq!(row.state(), InboxState::Withdrawn);
        assert_eq!(row.failure.as_deref(), Some("gave up"));
        assert!(inbox.list_open(session_id).await.expect("list").is_empty());
    }

    #[tokio::test]
    async fn only_root_sessions_with_pending_items_have_work_for_a_driver() {
        let (store, root) = store_with_session().await;
        let inbox = store.inbox_store();
        let (child, _lock) = store
            .create_child_session(
                root,
                None,
                Vec::new(),
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("create child");
        inbox
            .enqueue(item(child, InboxClass::Steer, "for the worker"))
            .await
            .expect("enqueue");
        let now = Utc::now();
        assert!(
            inbox
                .sessions_with_work(now)
                .await
                .expect("work")
                .is_empty(),
            "a worker's items wait for its parent, not a driver"
        );
        inbox
            .enqueue(item(root, InboxClass::Followup, "for the root"))
            .await
            .expect("enqueue");
        assert_eq!(inbox.sessions_with_work(now).await.expect("work"), vec![
            root
        ]);
    }
}
