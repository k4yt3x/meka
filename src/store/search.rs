//! The search index over the conversation's words, for `meka session search`.
//!
//! A contentless FTS5 table, `messages_fts`, keyed by `messages.id`: it holds the inverted index
//! and nothing else, so it costs a fraction of the text it indexes, and a hit is a row id the
//! caller reads the words back from. What it indexes is what people said: the `Text` blocks of
//! user and assistant messages and a compaction's summary. Tool calls, tool results and thinking
//! are left out; `conversation_search` scans those within one session.
//!
//! Every insert into `messages` goes through [`insert_message`], which writes the index row in the
//! same transaction. A delete reaches the index through a trigger on `messages`, so a cascade from
//! `sessions` is covered. [`reconcile_index`] runs on every open and puts right whatever neither
//! did: the first open after the index was introduced, an index another build made to another
//! definition of the words, a door this module does not know about, a restore that lost the
//! index.

use std::collections::HashMap;

use uuid::Uuid;

use super::{
    Store,
    memory::{Terms, canonical_trigger_sql},
    sessions::{SessionSummary, StoredMessage, decode_event_from_row, spawned_session_sql},
};
use crate::{
    conversation::{ContentBlock, Event, Message, Role, is_harness_stand_in, strip_inbox_header},
    error::{MekaError, Result},
};

/// Characters an excerpt keeps, enough of a line to recognize the conversation by.
const EXCERPT_CHARS: usize = 160;
/// Characters of a query term an excerpt line is matched on when the whole term is not in it:
/// the index stems, so `compaction` finds a line saying `compacted`, and the excerpt has to too.
const EXCERPT_STEM_CHARS: usize = 5;

/// One session a search found, with the words it was found by.
#[derive(Debug, Clone)]
pub(crate) struct SessionMatch {
    pub(crate) session: SessionSummary,
    /// The line of the best-matching message that holds a query term, whitespace collapsed and
    /// cut to [`EXCERPT_CHARS`]. `None` for a session found by its title alone.
    pub(crate) excerpt: Option<String>,
}

/// The one trigger the index needs, and the single place its text lives, so [`reconcile_index`]
/// compares against exactly what it will write. An insert has no trigger because the words are
/// meka's reading of the row, which SQL cannot make: [`insert_message`] is the insert.
///
/// The comment in it is load-bearing. The text is what every open compares, so it carries the
/// definition the index was built to: when [`indexed_words`] changes what a row contributes,
/// the number is bumped, every store's trigger stops matching on its next open, and the index is
/// built again to the new definition. Without that, an index built by an older build would pass
/// the row-count check forever with words nobody can find.
const DELETE_TRIGGER: (&str, &str) = (
    "messages_after_delete",
    "CREATE TRIGGER IF NOT EXISTS messages_after_delete AFTER DELETE ON messages BEGIN
         /* indexed words, definition 3 */
         DELETE FROM messages_fts WHERE rowid = old.id;
     END;",
);

/// The words a row contributes to the index: [`spoken_words_of_row`] with the scripts the
/// tokenizer cannot segment spaced out, so a Chinese word is found inside a sentence.
fn indexed_words(kind: &str, content: &str) -> String {
    space_unspaced_scripts(&spoken_words_of_row(kind, content))
}

/// What the user or the model said in a row, and nothing meka or a tool wrote. Empty for every
/// other kind of row, which is still indexed as an empty document so the index holds one row per
/// message and [`reconcile_index`] can compare counts.
fn spoken_words_of_row(kind: &str, content: &str) -> String {
    let row = StoredMessage {
        kind: kind.to_string(),
        content: content.to_string(),
        created_at: String::new(),
    };
    let message = match decode_event_from_row(&row) {
        Ok(Some(Event::Append(message))) => message,
        Ok(Some(Event::CompactBoundary { summary, .. })) => summary,
        _ => return String::new(),
    };
    spoken_words(&message)
}

/// Whether `character` belongs to a script written without spaces between words, which the
/// `unicode61` tokenizer therefore cannot segment: a run of it between two punctuation marks is
/// one token, and a word inside the run is unfindable. Han, kana, Hangul, Thai, Lao, Khmer and
/// Myanmar; the list can grow.
fn is_unspaced_script(character: char) -> bool {
    matches!(
        character as u32,
        0x1000..=0x109F        // Myanmar
            | 0x0E00..=0x0EFF  // Thai, Lao
            | 0x1100..=0x11FF  // Hangul jamo
            | 0x1780..=0x17FF  // Khmer
            | 0x3040..=0x30FF  // hiragana, katakana
            | 0x3130..=0x318F  // Hangul compatibility jamo
            | 0x31F0..=0x31FF  // katakana phonetic extensions
            | 0x3400..=0x4DBF  // CJK unified ideographs extension A
            | 0x4E00..=0x9FFF  // CJK unified ideographs
            | 0xAC00..=0xD7AF  // Hangul syllables
            | 0xF900..=0xFAFF  // CJK compatibility ideographs
            | 0xFF66..=0xFF9F  // halfwidth katakana
            | 0x20000..=0x3134F // CJK unified ideographs extensions B to G
    )
}

/// `text` with a space between every pair of adjacent characters of which at least one is from
/// an unspaced script and neither is a separator already, so the tokenizer sees one token per
/// character there and a query rendered by [`render_term`] matches by adjacency. Latin words and
/// punctuation pass through untouched. The one segmenter, for the memory index to share when it
/// is fed a projection of its own.
fn space_unspaced_scripts(text: &str) -> String {
    let mut spaced = String::with_capacity(text.len() + text.len() / 2);
    let mut previous: Option<char> = None;
    for character in text.chars() {
        if let Some(previous) = previous
            && (is_unspaced_script(character) || is_unspaced_script(previous))
            && character.is_alphanumeric()
            && previous.is_alphanumeric()
        {
            spaced.push(' ');
        }
        spaced.push(character);
        previous = Some(character);
    }
    spaced
}

/// One query term as the `MATCH` syntax the segmented index answers: a phrase whose tokens are
/// what [`space_unspaced_scripts`] makes of the term, so `深圳` becomes `"深 圳"` and matches only
/// where those characters are adjacent, and `iphone手机` becomes `"iphone 手 机"`. A Latin word
/// is the one-token phrase it always was. `prefix` stars the phrase, which FTS5 applies to its
/// last token: `"科 技"*` finds 科技园.
fn render_term(term: &str, prefix: bool) -> String {
    // FTS5 escapes a double quote inside a string literal by doubling it. `Terms::parse` cannot
    // hand one over, but the escape stays: the invariant lives in another function.
    let quoted = format!("\"{}\"", space_unspaced_scripts(term).replace('"', "\"\""));
    if prefix { format!("{quoted}*") } else { quoted }
}

/// The three tiers a query is tried at, narrowest first: every term, any term, any term as a
/// prefix. Deduplicated, because one term makes the first two the same query, which cannot answer
/// differently.
fn tiered_expressions(words: &[String]) -> Vec<String> {
    let join = |prefix: bool, operator: &str| {
        words
            .iter()
            .map(|word| render_term(word, prefix))
            .collect::<Vec<_>>()
            .join(operator)
    };
    let mut expressions = vec![
        join(false, " AND "),
        join(false, " OR "),
        join(true, " OR "),
    ];
    expressions.dedup();
    expressions
}

/// The `Text` blocks of a message, rendered by the one block-to-text function the
/// `conversation_search` tool also reads through, so what the index finds the tool can find.
/// A user message keeps the words a person sent and not what meka put beside them: a stand-in
/// meka wrote in place of content, and the header above an inbox item, are left out, the way the
/// title leaves them out.
fn spoken_words(message: &Message) -> String {
    let from_a_person = message.role == Role::User;
    message
        .content
        .iter()
        .filter(|block| matches!(block, ContentBlock::Text { .. }))
        .filter_map(ContentBlock::search_text)
        .filter(|text| !from_a_person || !is_harness_stand_in(text))
        .map(|text| {
            if from_a_person {
                strip_inbox_header(&text).to_string()
            } else {
                text
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The one door into `messages`: the row and its index entry, in the caller's transaction.
/// Answers the new row's id.
pub(super) fn insert_message(
    connection: &rusqlite::Connection,
    session_id: &str,
    kind: &str,
    content: &str,
    created_at: &str,
) -> rusqlite::Result<i64> {
    connection.execute(
        "INSERT INTO messages (session_id, kind, content, created_at) VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![session_id, kind, content, created_at],
    )?;
    let message_id = connection.last_insert_rowid();
    connection.execute(
        "INSERT INTO messages_fts (rowid, text) VALUES (?1, ?2)",
        rusqlite::params![message_id, indexed_words(kind, content)],
    )?;
    Ok(message_id)
}

/// Bring the index into line with the `messages` table and with this build's definition of the
/// words, on every open.
///
/// The table is created by the schema ledger; the trigger is created here and nowhere else, for
/// the reasons `crate::store::memory::reconcile_index` gives. Two questions, in order. Is the
/// trigger this build's, comment and all? If not, the index was built by another build, or never,
/// and it is built again from every row, in the one transaction that also writes the trigger, so
/// a crash between the two cannot leave a matching trigger over an index built to the old
/// definition. Otherwise, does the index hold one row per message? If not, the rows without an
/// entry are indexed and the entries without a row dropped. The first open after the index was
/// introduced takes the first path, which is also why the ledger step leaves it empty: the words
/// are meka's reading of a row, and a migration may not call meka.
///
/// Cheap when nothing is wrong, which is every open but the first: one `sqlite_master` read and
/// two counts.
pub(crate) fn reconcile_index(connection: &rusqlite::Connection) -> rusqlite::Result<()> {
    let existing: HashMap<String, String> = {
        let mut statement = connection.prepare(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'messages'",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                canonical_trigger_sql(&row.get::<_, String>(1)?),
            ))
        })?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    let (name, sql) = DELETE_TRIGGER;
    // Every trigger on the table is compared, not only the one by this build's name, so one an
    // earlier build left under another name does not stay beside it.
    if existing.len() == 1 && existing.get(name) == Some(&canonical_trigger_sql(sql)) {
        return repair_a_desynced_index(connection);
    }
    tracing::info!("building the session search index to this build's definition");
    // `Immediate`, like the memory index's rebuild: under WAL a deferred transaction's lock
    // upgrade can fail without consulting the busy handler, and this one writes.
    let transaction =
        rusqlite::Transaction::new_unchecked(connection, rusqlite::TransactionBehavior::Immediate)?;
    for name in existing.keys() {
        transaction.execute_batch(&format!("DROP TRIGGER IF EXISTS {name};"))?;
    }
    transaction.execute_batch(sql)?;
    transaction.execute_batch("INSERT INTO messages_fts(messages_fts) VALUES('delete-all');")?;
    let indexed = index_rows(&transaction, "")?;
    transaction.commit()?;
    tracing::info!("built the session search index over {indexed} message(s)");
    Ok(())
}

/// Index every message without an entry and drop every entry without a message, when the two
/// counts disagree: a door this module does not know about, or a restore that lost part of the
/// index. Nothing to do on a healthy store, which two counts establish.
fn repair_a_desynced_index(connection: &rusqlite::Connection) -> rusqlite::Result<()> {
    let (stored, indexed): (i64, i64) = connection.query_row(
        // `messages_fts_docsize` holds one row per indexed document, so it counts what the index
        // believes; `messages` is what is true.
        "SELECT (SELECT count(*) FROM messages), (SELECT count(*) FROM messages_fts_docsize)",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    if stored == indexed {
        return Ok(());
    }
    let transaction =
        rusqlite::Transaction::new_unchecked(connection, rusqlite::TransactionBehavior::Immediate)?;
    let stale: Vec<i64> = {
        let mut statement = transaction.prepare(
            "SELECT rowid FROM messages_fts WHERE rowid NOT IN (SELECT id FROM messages)",
        )?;
        let rows = statement.query_map([], |row| row.get(0))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    for rowid in &stale {
        transaction.execute("DELETE FROM messages_fts WHERE rowid = ?1", [rowid])?;
    }
    let added = index_rows(
        &transaction,
        "WHERE id NOT IN (SELECT rowid FROM messages_fts)",
    )?;
    transaction.commit()?;
    tracing::info!(
        "reconciled the session search index: indexed {added} message(s), dropped {} stale row(s)",
        stale.len()
    );
    Ok(())
}

/// Index the `messages` rows `where_clause` selects, every row when it is empty, and say how
/// many. `where_clause` is one of this module's own literals, never anything derived from input.
fn index_rows(
    transaction: &rusqlite::Transaction<'_>,
    where_clause: &str,
) -> rusqlite::Result<usize> {
    let mut select = transaction.prepare(&format!(
        "SELECT id, kind, content FROM messages {where_clause} ORDER BY id ASC"
    ))?;
    let mut insert =
        transaction.prepare("INSERT INTO messages_fts (rowid, text) VALUES (?1, ?2)")?;
    let rows = select.query_map([], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;
    let mut indexed = 0usize;
    for row in rows {
        let (id, kind, content) = row?;
        insert.execute(rusqlite::params![id, indexed_words(&kind, &content)])?;
        indexed += 1;
    }
    Ok(indexed)
}

impl Store {
    /// [`reconcile_index`] on this store, for a test about what it repairs.
    #[cfg(test)]
    pub(crate) async fn reconcile_search_index_for_test(&self) {
        self.connection
            .call(|connection| reconcile_index(connection))
            .await
            .expect("the index reconciles");
    }

    /// How many messages the store holds and how many the index does, for a test about the two
    /// staying together.
    #[cfg(test)]
    pub(crate) async fn search_index_counts_for_test(&self) -> (i64, i64) {
        self.connection
            .call(|connection| {
                connection.query_row(
                    "SELECT (SELECT count(*) FROM messages), \
                            (SELECT count(*) FROM messages_fts_docsize)",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
            })
            .await
            .expect("the counts read")
    }

    /// The sessions whose words match `query`, best first, at most `limit` of them.
    ///
    /// A session whose set title holds every term comes first, newest first among those. The rest
    /// are ranked by their best-matching message, through the tiers the memory search uses: every
    /// term, then any term, then any term as a prefix, stopping at the first tier that finds
    /// anything. Sub-agent sessions are left out unless `include_children`.
    pub(crate) async fn search_sessions(
        &self,
        query: &str,
        limit: usize,
        include_children: bool,
    ) -> Result<Vec<SessionMatch>> {
        let terms = Terms::parse(&[query.to_string()]);
        if terms.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let words = terms.words().to_vec();
        let words_for_titles = words.clone();
        let expressions = tiered_expressions(&words);
        let (titled, ranked) = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                let spawned = spawned_session_sql("s.");
                let children = if include_children {
                    String::new()
                } else {
                    format!("AND NOT {spawned}")
                };
                // Matched here rather than in SQL: the set titles are few, `Terms::parse` folded
                // the words with Unicode case rules, and SQLite's `lower()` folds ASCII alone.
                let titled: Vec<String> = {
                    let mut statement = connection.prepare(&format!(
                        "SELECT s.id, s.title FROM sessions s
                         WHERE s.title IS NOT NULL {children}
                         ORDER BY s.updated_at DESC, s.id DESC"
                    ))?;
                    let rows = statement.query_map([], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    })?;
                    let mut titled = Vec::new();
                    for row in rows {
                        let (id, title) = row?;
                        let folded = title.to_lowercase();
                        if words_for_titles
                            .iter()
                            .all(|word| folded.contains(word.as_str()))
                        {
                            titled.push(id);
                            if titled.len() == limit {
                                break;
                            }
                        }
                    }
                    titled
                };
                // `MATERIALIZED`, because the planner otherwise flattens the subquery and `rank`
                // leaves the `MATCH` it is only defined under. Grouping by session takes the bare
                // `m.id` beside `min()`, which SQLite guarantees is the row of the minimum.
                let mut statement = connection.prepare(&format!(
                    "WITH hits AS MATERIALIZED (
                         SELECT rowid AS id, rank FROM messages_fts WHERE messages_fts MATCH ?1
                     )
                     SELECT m.session_id, m.id, min(hits.rank)
                     FROM hits
                     JOIN messages m ON m.id = hits.id
                     JOIN sessions s ON s.id = m.session_id
                     WHERE 1 {children}
                     GROUP BY m.session_id
                     ORDER BY 3 ASC, s.updated_at DESC
                     LIMIT ?2"
                ))?;
                let mut ranked: Vec<(String, i64)> = Vec::new();
                for expression in &expressions {
                    let rows = statement
                        .query_map(rusqlite::params![expression, limit as i64], |row| {
                            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                        })?;
                    ranked = rows.collect::<rusqlite::Result<_>>()?;
                    if !ranked.is_empty() {
                        break;
                    }
                }
                Ok((titled, ranked))
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to search sessions: {error}")))?;

        let best_row: HashMap<&str, i64> = ranked
            .iter()
            .map(|(session, row)| (session.as_str(), *row))
            .collect();
        let mut order: Vec<(Uuid, Option<i64>)> = Vec::new();
        for id in &titled {
            if let Ok(id) = Uuid::parse_str(id) {
                order.push((id, best_row.get(id.to_string().as_str()).copied()));
            }
        }
        for (session, row) in &ranked {
            if titled.contains(session) {
                continue;
            }
            if let Ok(id) = Uuid::parse_str(session) {
                order.push((id, Some(*row)));
            }
        }
        order.truncate(limit);

        let mut matches = Vec::with_capacity(order.len());
        for (id, row) in order {
            // A session deleted between the two reads is simply not a result.
            let Some(session) = self.session_info(id).await? else {
                continue;
            };
            let excerpt = match row {
                Some(row) => self.excerpt_of(row, &words).await?,
                None => None,
            };
            matches.push(SessionMatch { session, excerpt });
        }
        Ok(matches)
    }

    /// The line of message `row` a query term is in, cut to [`EXCERPT_CHARS`]; the first line
    /// when no line holds one, which stemming makes possible; `None` when the row is gone.
    async fn excerpt_of(&self, row: i64, words: &[String]) -> Result<Option<String>> {
        let text = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement =
                    connection.prepare("SELECT kind, content FROM messages WHERE id = ?1")?;
                let mut rows = statement.query_map([row], |row| {
                    Ok(spoken_words_of_row(
                        &row.get::<_, String>(0)?,
                        &row.get::<_, String>(1)?,
                    ))
                })?;
                rows.next().transpose()
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to read a match: {error}")))?;
        Ok(text.as_deref().and_then(|text| excerpt_line(text, words)))
    }
}

/// The first line of `text` holding a query term, else one holding the head of a term, else the
/// first line with words; whitespace collapsed and cut to [`EXCERPT_CHARS`].
fn excerpt_line(text: &str, words: &[String]) -> Option<String> {
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    let holding = |needles: &[String]| {
        lines.iter().copied().find(|line| {
            let lowered = line.to_lowercase();
            needles
                .iter()
                .any(|needle| lowered.contains(needle.as_str()))
        })
    };
    let stems: Vec<String> = words
        .iter()
        .map(|word| word.chars().take(EXCERPT_STEM_CHARS).collect())
        .collect();
    let line = holding(words)
        .or_else(|| holding(&stems))
        .or_else(|| lines.first().copied())?;
    let collapsed = line.split_whitespace().collect::<Vec<_>>().join(" ");
    Some(if collapsed.chars().count() > EXCERPT_CHARS {
        let kept: String = collapsed.chars().take(EXCERPT_CHARS).collect();
        format!("{}…", kept.trim_end())
    } else {
        collapsed
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::{Event, Message};

    async fn session_saying(store: &Store, lines: &[&str]) -> Uuid {
        let id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        for line in lines {
            store
                .save_event(id, &Event::Append(Message::user(*line)))
                .await
                .expect("save");
        }
        id
    }

    async fn ids_found(store: &Store, query: &str) -> Vec<Uuid> {
        store
            .search_sessions(query, 10, false)
            .await
            .expect("search")
            .into_iter()
            .map(|found| found.session.id)
            .collect()
    }

    /// What meka wrote into a user message is not what the person said: a harness note and the
    /// header above an inbox item are not indexed, while the item's own words are.
    #[tokio::test]
    async fn what_meka_wrote_beside_the_users_words_is_not_indexed() {
        let store = Store::for_test().await;
        let noted = session_saying(&store, &[&format!(
            "{} the model produced no visible output",
            crate::conversation::HARNESS_NOTE
        )])
        .await;
        let headed = session_saying(&store, &[
            "[Message from a client, arrived 2026-09-24 12:00 +00:00]\nthe words about egrets",
        ])
        .await;

        assert!(
            ids_found(&store, "visible output").await.is_empty(),
            "{noted}"
        );
        assert_eq!(ids_found(&store, "egrets").await, vec![headed]);
        assert!(ids_found(&store, "arrived").await.is_empty());
    }

    /// The words people exchanged are what a session is found by; a tool's output is not.
    #[tokio::test]
    async fn a_session_is_found_by_its_words_and_not_by_a_tool_result() {
        use crate::conversation::{ContentBlock, Role, ToolResultContent};

        let store = Store::for_test().await;
        let spoken = session_saying(&store, &["let us talk about giraffes today"]).await;
        let tooled = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        store
            .save_event(
                tooled,
                &Event::Append(Message {
                    role: Role::User,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: "t1".to_string(),
                        content: vec![ToolResultContent::Text {
                            text: "giraffes everywhere in this file".to_string(),
                        }],
                        is_error: false,
                    }],
                }),
            )
            .await
            .expect("save");

        assert_eq!(ids_found(&store, "giraffes").await, vec![spoken]);
        let found = store
            .search_sessions("giraffes", 10, false)
            .await
            .expect("search");
        assert_eq!(
            found[0].excerpt.as_deref(),
            Some("let us talk about giraffes today")
        );
    }

    /// Every term first, any term when that finds nothing, a prefix when that finds nothing
    /// either: the memory search's ladder, so a truncated word still finds its session.
    #[tokio::test]
    async fn the_tiers_widen_only_when_the_narrower_one_finds_nothing() {
        let store = Store::for_test().await;
        let both = session_saying(&store, &["the zebra and the okapi"]).await;
        let one = session_saying(&store, &["only the zebra here"]).await;

        assert_eq!(ids_found(&store, "zebra okapi").await, vec![both]);
        let any = ids_found(&store, "zebra wombat").await;
        assert!(any.contains(&both) && any.contains(&one), "{any:?}");
        assert_eq!(ids_found(&store, "okap").await, vec![both]);
        assert!(ids_found(&store, "wombat").await.is_empty());
    }

    /// A title someone set outranks a body match, and a sub-agent's session is hidden as the
    /// listing hides it.
    #[tokio::test]
    async fn a_titled_session_comes_first_and_a_sub_agent_session_stays_hidden() {
        let store = Store::for_test().await;
        let body = session_saying(&store, &["notes on the kestrel migration"]).await;
        let titled = session_saying(&store, &["unrelated words"]).await;
        store
            .update_session(titled, crate::store::SessionPatch {
                title: Some(Some("Kestrel research".to_string())),
                ..Default::default()
            })
            .await
            .expect("title");
        let spawned = session_saying(&store, &["a kestrel seen by a sub-agent"]).await;
        store.mark_spawned_for_test(spawned, body).await;

        let found = store
            .search_sessions("kestrel", 10, false)
            .await
            .expect("search");
        let ids: Vec<Uuid> = found.iter().map(|found| found.session.id).collect();
        assert_eq!(ids, vec![titled, body]);
        assert_eq!(found[0].excerpt, None, "found by its title alone");
        assert!(
            ids_found(&store, "kestrel").await.len() == 2
                && store
                    .search_sessions("kestrel", 10, true)
                    .await
                    .expect("search")
                    .len()
                    == 3
        );
    }

    /// The index follows the table through every door: an appended message is found, a deleted
    /// session is not, and an index emptied behind meka's back is refilled on the next open.
    #[tokio::test]
    async fn the_index_follows_the_table_and_is_refilled_on_open() {
        let store = Store::for_test().await;
        let session = session_saying(&store, &["a sentence about pelicans"]).await;
        assert_eq!(ids_found(&store, "pelicans").await, vec![session]);

        store
            .run_sql_for_test("DELETE FROM messages_fts WHERE rowid IN (SELECT id FROM messages)")
            .await;
        assert!(ids_found(&store, "pelicans").await.is_empty());
        store.reconcile_search_index_for_test().await;
        assert_eq!(ids_found(&store, "pelicans").await, vec![session]);

        store.delete_session(session).await.expect("delete");
        assert!(ids_found(&store, "pelicans").await.is_empty());
        let (stored, indexed) = store.search_index_counts_for_test().await;
        assert_eq!((stored, indexed), (0, 0), "the cascade reached the index");
    }

    /// A word of a script written without spaces is found inside a sentence, in its own order,
    /// beside a Latin word, and the excerpt shows the sentence as it was written.
    #[tokio::test]
    async fn a_cjk_word_is_found_inside_a_sentence_and_its_order_matters() {
        let store = Store::for_test().await;
        let sentence = session_saying(&store, &[
            "办公室在深圳南山区的科技园，compaction 也在那里。",
        ])
        .await;
        let reversed = session_saying(&store, &["圳深 is not the same word"]).await;
        let _other = session_saying(&store, &["nothing of the kind here"]).await;

        assert_eq!(ids_found(&store, "深圳").await, vec![sentence]);
        assert_eq!(ids_found(&store, "圳深").await, vec![reversed]);
        assert_eq!(ids_found(&store, "科技").await, vec![sentence]);
        assert_eq!(ids_found(&store, "compaction 深圳").await, vec![sentence]);
        assert_eq!(ids_found(&store, "compaction 深圳 wombat").await, vec![
            sentence
        ]);
        let found = store
            .search_sessions("深圳", 10, false)
            .await
            .expect("search");
        assert_eq!(
            found[0].excerpt.as_deref(),
            Some("办公室在深圳南山区的科技园，compaction 也在那里。"),
            "the excerpt is the sentence as written, not as indexed"
        );
    }

    /// The segmenter spaces only where the tokenizer needs it: between two characters of which
    /// one is from an unspaced script and neither is a separator already.
    #[test]
    fn only_unspaced_scripts_are_spaced_and_only_between_word_characters() {
        assert_eq!(space_unspaced_scripts("iphone手机，ok"), "iphone 手 机，ok");
        assert_eq!(
            space_unspaced_scripts("plain latin text."),
            "plain latin text."
        );
        assert_eq!(space_unspaced_scripts("深 圳"), "深 圳");
        assert_eq!(
            space_unspaced_scripts("日本語のテキスト"),
            "日 本 語 の テ キ ス ト"
        );
        assert_eq!(space_unspaced_scripts("한국어 텍스트"), "한 국 어 텍 스 트");
        assert_eq!(space_unspaced_scripts("ภาษาไทย"), "ภ า ษ า ไ ท ย");
        assert_eq!(render_term("深圳", false), "\"深 圳\"");
        assert_eq!(render_term("iphone手机", true), "\"iphone 手 机\"*");
        assert_eq!(render_term("word", false), "\"word\"");
        assert_eq!(
            tiered_expressions(&["a".to_string(), "b".to_string()]),
            vec!["\"a\" AND \"b\"", "\"a\" OR \"b\"", "\"a\"* OR \"b\"*"]
        );
        assert_eq!(tiered_expressions(&["a".to_string()]), vec![
            "\"a\"", "\"a\"*"
        ]);
    }

    /// An index a build with another definition of the words left behind is built again to this
    /// one on the next open, and the definition is what the trigger's text says, so a bumped
    /// number is enough to make every store rebuild.
    #[tokio::test]
    async fn an_index_built_to_another_definition_is_built_again_on_open() {
        let store = Store::for_test().await;
        let session = session_saying(&store, &["办公室在深圳南山区的科技园"]).await;
        assert_eq!(ids_found(&store, "深圳").await, vec![session]);

        // What an earlier build leaves: its own trigger text, and the words indexed unsegmented.
        store
            .run_sql_for_test(
                "DROP TRIGGER messages_after_delete;
                 CREATE TRIGGER messages_after_delete AFTER DELETE ON messages BEGIN
                     DELETE FROM messages_fts WHERE rowid = old.id;
                 END;
                 INSERT INTO messages_fts(messages_fts) VALUES('delete-all');
                 INSERT INTO messages_fts(rowid, text)
                     SELECT id, '办公室在深圳南山区的科技园' FROM messages;",
            )
            .await;
        assert!(ids_found(&store, "深圳").await.is_empty());
        let (stored, indexed) = store.search_index_counts_for_test().await;
        assert_eq!(stored, indexed, "the counts alone cannot tell");

        store.reconcile_search_index_for_test().await;
        assert_eq!(ids_found(&store, "深圳").await, vec![session]);
        store.reconcile_search_index_for_test().await;
        assert_eq!(
            ids_found(&store, "深圳").await,
            vec![session],
            "and a second open finds the trigger current and leaves the index alone"
        );
    }

    /// The excerpt is the line the term is in, found through the stem when the index matched an
    /// inflection, and cut to its budget.
    #[test]
    fn an_excerpt_is_the_matching_line_cut_to_fit() {
        let text = "first line\nwe compacted the log yesterday\nthird";
        assert_eq!(
            excerpt_line(text, &["compaction".to_string()]).as_deref(),
            Some("we compacted the log yesterday")
        );
        assert_eq!(
            excerpt_line("nothing here", &["absent".to_string()]).as_deref(),
            Some("nothing here")
        );
        let long = "word ".repeat(100);
        let excerpt = excerpt_line(&long, &["word".to_string()]).expect("a line");
        assert!(excerpt.chars().count() <= EXCERPT_CHARS + 1, "{excerpt}");
        assert!(excerpt.ends_with('…'));
    }
}
