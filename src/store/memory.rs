//! The memory store: the `memories` table, its FTS5 index, and the ranked retrieval over both.
//!
//! **The table is the source of truth**, not a mirror of anything on disk. See
//! `docs/book/src/usage/memory.md` for the user-facing consequence, which is that `meka memory
//! export` replaces `grep`.
//!
//! # What SQLite holds, so this module does not
//!
//! - **The FTS index is external-content** (`content='memories'`), so three triggers keep it in
//!   step and there is no sync function to get wrong. It stays derived and disposable: `INSERT INTO
//!   memories_fts(memories_fts) VALUES('rebuild')` repairs it from the table.
//! - **`name` is `UNIQUE COLLATE NOCASE`**, which is the case-collision check.
//! - **Omit-to-keep is in the upsert**, not in a Rust branch. See [`MemoryStore::write`].
//! - **A transaction is the read-modify-write lock**, across processes, so no write door needs an
//!   `flock`. The one in-process mutex left is [`MemoryStore::lock_duplicate_check`], which guards
//!   an advisory read-then-report rather than the write.
//!
//! # Four traps
//!
//! **`bm25()` returns a negative number, and more negative is better.** [`Ranking::score`] negates
//! before multiplying by the importance weights; without that the multiplication orders the results
//! backwards, and it does so silently.
//!
//! **A model-supplied query is not FTS5 syntax.** `*`, `"`, `NEAR` and `OR` are all operators, so a
//! query containing them would either error or quietly mean something else. [`Terms`] tokenizes on
//! the Rust side and re-emits each token as a quoted string literal.
//!
//! **Never `INSERT OR REPLACE` into `memories`, and never `UPDATE` its `id`.** Both are a
//! delete-then-insert that can change the rowid, which unanchors every row of the external-content
//! index; `ON CONFLICT DO UPDATE` is the only upsert shape allowed here. A column added to
//! `memories` and left out of the three triggers desyncs the index just as silently, and FTS5's own
//! `integrity-check` will *not* tell you: on an external-content table it verifies the index's
//! internal structure and never compares it to the content. [`MemoryStore::integrity_check`]
//! therefore counts documents as well, [`repair_a_desynced_index`] runs that comparison at every
//! open, `meka memory verify` runs it by hand, and `rebuild_index` is the only real repair.
//!
//! **`sqlite_master.sql` is not the text you submitted.** SQLite strips `IF NOT EXISTS` and drops
//! the trailing `;`, so comparing a build's own literal against the stored one to detect drift
//! answers "drifted" every time, and the repair then runs on every process start. See
//! [`canonical_trigger_sql`].

use std::{
    sync::Arc,
    time::{Duration, SystemTime},
};

use rusqlite::OptionalExtension;
use tokio_rusqlite::Connection;

use crate::{
    entry::{DEFAULT_PRIORITY, MAX_PRIORITY},
    error::{MekaError, Result},
    memory::Memory,
};

/// Column indices in `memories_fts`, in declaration order. Named because `snippet()` and `bm25()`
/// take positional column arguments and a bare `3` at a call site is unreadable and unsearchable.
const COLUMN_BODY: i64 = 3;

/// `bm25` weights, one per indexed column: name, description, tags, body.
///
/// A hit on the *name* is close to an exact-recall event (the model half-remembered what it
/// called the note), so it outranks everything. The description is the one line the model wrote to
/// stand on its own in the index, which makes it a better signal than prose buried in a body.
const WEIGHT_NAME: f64 = 10.0;
const WEIGHT_DESCRIPTION: f64 = 3.0;
const WEIGHT_TAGS: f64 = 2.0;
const WEIGHT_BODY: f64 = 1.0;

/// How many BM25-ranked candidates are pulled back before the importance weights re-order them.
///
/// Two-stage retrieval: relevance alone picks the pool, then [`Ranking::score`] applies priority,
/// usage and freshness within it. The weights span roughly 0.5x to 6x, which reorders neighbors
/// but will not lift the thousandth-best match to the top, so a pool this size costs a little work
/// and buys the whole practical effect. Anything dropped here was dropped for having no textual
/// relevance, which is the one criterion the model's query actually stated.
const CANDIDATE_POOL: usize = 200;

/// Priority at or below which a memory is a standing directive and does not decay. Read with
/// [`FRESHNESS_HALF_LIFE_DAYS`].
const STANDING_PRIORITY_MAX: u8 = 1;

/// How far the declared priority can move a score, as a multiplier span above 1.0. A priority-0
/// memory is worth `1.0 + PRIORITY_WEIGHT_SPAN` times a priority-9 one at equal relevance.
const PRIORITY_WEIGHT_SPAN: f64 = 2.0;

/// Read count at which the usage bonus saturates. Logarithmic, so the first few recalls move the
/// needle and the hundredth does not.
const USAGE_SATURATION_READS: f64 = 20.0;

/// Half-life of a non-standing memory's freshness, in days.
///
/// Deliberately long, and deliberately *not applied to standing directives at all*. A two-year-old
/// priority-0 rule is exactly as binding as a new one, while a two-year-old priority-8 note
/// probably is not, so decay is keyed on what the memory claims to be rather than applied flat.
/// This is the one place the design departs from the usual recency term, and it is the difference
/// between a memory store and a feed.
const FRESHNESS_HALF_LIFE_DAYS: f64 = 365.0;

/// Floor on the freshness multiplier. An old memory is demoted, never buried: it is still the
/// answer when nothing newer matches.
const FRESHNESS_FLOOR: f64 = 0.5;

/// How many characters the prefix tier may trim from a query term on its second attempt, and the
/// shortest a term may be trimmed to. See [`Terms::trimmed_prefix_match_expressions`].
const PREFIX_TRIM_CHARS: usize = 5;
const PREFIX_MIN_CHARS: usize = 4;

/// Distinct terms one search may carry.
///
/// The substring tier builds one `OR` clause per term against four columns, and SQLite caps an
/// expression tree at depth 1000; the full-text tiers OR the same terms into one `MATCH`. Far more
/// than any real question needs (the tool asks for synonyms, not for a document) and low enough
/// that pasting five paragraphs in as a "query" degrades to searching its first hundred words
/// rather than erroring.
const MAX_TERMS: usize = 100;

/// How much of a body [`MemoryStore::substring_search`] carries as its excerpt. FTS5's `snippet()`
/// is only available under a `MATCH`, and this scan is what runs when there is none.
const SUBSTRING_SNIPPET_CHARS: usize = 160;

/// A window of `chars` characters from `body` containing the first of `terms` that appears in it.
///
/// The renderer presents this under a preamble saying the text contains the search term, so an
/// excerpt that does not is a statement the tool makes and the body contradicts. Taking the body's
/// opening satisfied that only when the match happened to be near the start, and reliably failed
/// for the case this tier exists for: a long CJK body, which `unicode61` tokenizes as a single
/// token so `MATCH` cannot reach it, where the match sits wherever the author put it.
///
/// A quarter of the window leads the match, so what comes back reads as an excerpt rather than as
/// a fragment starting mid-word. When no term is in the body (the row matched on its name,
/// description or tags) the opening is the honest answer and there is nothing to center on.
///
/// Case-insensitively, and specifically *ASCII* case-insensitively, because that is what SQLite's
/// `LIKE` did to select this row. Anything else would look for a match the query never made.
fn excerpt_around_a_match(body: &str, terms: &[String], chars: usize) -> String {
    // The first term that hits, not the earliest hit among all of them: both satisfy the contract,
    // and only one of them stops scanning.
    //
    // Every term is tried, with no cap. `Terms::parse` preserves query order across all `queries`
    // entries, so three phrasings ending in the one non-ASCII word put that word past any small
    // ceiling, and a row whose body is entirely CJK reaches this tier precisely because no
    // full-text tier can see it, matches on `LIKE`, and would then get an excerpt from the body's
    // opening under a preamble asserting it contained the term. The cost is bounded by
    // `MAX_TERMS`, and a false statement is the wrong thing to spend it on.
    let found = terms
        .iter()
        .find_map(|term| find_ignoring_ascii_case(body, term));
    let lead = chars / 4;
    let start = match found {
        Some(offset) => body[..offset].chars().count().saturating_sub(lead),
        None => 0,
    };
    body.chars().skip(start).take(chars).collect()
}

/// Byte offset of the first ASCII-case-insensitive occurrence of `needle`, at a character
/// boundary.
///
/// Byte-wise rather than lowercasing the haystack, because the haystack is a whole memory body and
/// this runs once per candidate row: a store of 200 KB notes would otherwise allocate a second copy
/// of each one to answer a question about a 160-character window. The boundary check is what keeps
/// the offset sliceable: an ASCII needle cannot match inside a multi-byte character, but a
/// non-ASCII one can start part-way through.
fn find_ignoring_ascii_case(haystack: &str, needle: &str) -> Option<usize> {
    let (hay, need) = (haystack.as_bytes(), needle.as_bytes());
    if need.is_empty() || need.len() > hay.len() {
        return None;
    }
    (0..=hay.len() - need.len()).find(|&start| {
        haystack.is_char_boundary(start)
            && hay
                .get(start..start + need.len())
                .is_some_and(|window| window.eq_ignore_ascii_case(need))
    })
}

/// Bring the memory search index into line with this build, on every open.
///
/// The `memories` table, its `idx_memories_priority_created_at` index and the `memories_fts`
/// virtual table are *not* created here. They are created by [`crate::store::migrations`], which
/// owns the `CREATE` for every table the agent reads, so that one ledger describes that schema and
/// an older store is carried forward by the same code path a new one is built by. This function is
/// what is left, and it is a different kind of operation: reconciliation of a derived index against
/// the table it is derived from, which is why it sits on the open path rather than in the ledger or
/// in a command the user has to remember to run.
///
/// Two steps, both of which name no release and hold no knowledge of a past shape.
/// [`sync_triggers`] asks whether this database's FTS triggers are the ones this build requires and
/// replaces them if not, rebuilding the index because whatever the old ones did or failed to do is
/// already in it. [`repair_a_desynced_index`] then probes for a drift the trigger comparison cannot
/// see, and is skipped when that rebuild just ran, since it would only re-ask a question one pass
/// of the table has already answered.
///
/// The triggers are created by [`sync_triggers`] and nowhere else, which is also why the ledger's
/// baseline creates the tables but not the triggers. A `CREATE TRIGGER IF NOT EXISTS` pass ahead of
/// it looks harmless and is the opposite: a trigger that had gone *missing* is silently put back
/// before `sync_triggers` reads `sqlite_master`, so the comparison matches, no rebuild runs, and
/// every write that landed while it was absent stays out of the index for good.
/// [`repair_a_desynced_index`] cannot see that either, because a missed *update* leaves the
/// document counts equal.
///
/// Letting [`sync_triggers`] own creation costs nothing on a fresh database: it finds none of the
/// three, takes the replacement path, and creates all three plus a rebuild of an empty table inside
/// one transaction, which is better than three autocommit statements anyway.
pub(crate) fn reconcile_index(connection: &rusqlite::Connection) -> rusqlite::Result<()> {
    if sync_triggers(connection)? {
        return Ok(());
    }
    repair_a_desynced_index(connection)
}

/// A store at head with its index reconciled, exactly as `initialize_schema` leaves one.
///
/// Test-only, and routed through [`crate::store::migrations`] rather than a local copy of the
/// DDL: a memory test that built its own tables could pass against a schema production does not
/// produce, which is the whole failure mode the single ledger exists to remove.
#[cfg(test)]
fn create_tables(connection: &mut rusqlite::Connection) -> rusqlite::Result<()> {
    crate::store::migrations::create_for_test(connection)
        .expect("the schema ledger builds a fresh store");
    reconcile_index(connection)
}

/// The three triggers that uphold the external-content contract, and the single place their text
/// lives, so [`sync_triggers`] compares against exactly what it will write. Nothing else creates
/// them, including the schema ledger that creates the tables they hang off.
///
/// Every mutation of `memories` must be mirrored, and a delete must be told the *old* values so
/// FTS can find the posting list to remove. A column added to the table and not added to all three
/// of these desyncs the index in silence.
const TRIGGER_DEFINITIONS: [(&str, &str); 3] = [
    (
        "memories_after_insert",
        "CREATE TRIGGER IF NOT EXISTS memories_after_insert AFTER INSERT ON memories BEGIN
             INSERT INTO memories_fts(rowid, name, description, tags, body)
             VALUES (new.id, new.name, new.description, new.tags, new.body);
         END;",
    ),
    (
        "memories_after_delete",
        "CREATE TRIGGER IF NOT EXISTS memories_after_delete AFTER DELETE ON memories BEGIN
             INSERT INTO memories_fts(memories_fts, rowid, name, description, tags, body)
             VALUES ('delete', old.id, old.name, old.description, old.tags, old.body);
         END;",
    ),
    (
        // Gated on the indexed columns actually changing. `record_read` bumps `read_count` on
        // every `memory_read`, and an ungated trigger would answer that by deleting and
        // re-inserting all four columns: two full posting-list rewrites of a body that has not
        // changed, on the one operation a large memory performs most often.
        //
        // Every indexed column has to appear in the `WHEN` or an edit to the one left out stops
        // reaching the index. `id` is deliberately absent: it is the `content_rowid`, and the
        // module header forbids updating it at all.
        //
        // `name` is compared `COLLATE BINARY` because `IS NOT` otherwise inherits the column's
        // `NOCASE` and a case-only rename would not fire. That is defense in depth rather than a
        // live guarantee, and worth saying so: `unicode61` folds case when it tokenizes, so
        // `Policy` and `policy` produce identical index entries and skipping the re-index for a
        // case-only change is currently unobservable through search or `integrity-check`. It stays
        // because the comparison should follow the *content*, not the tokenizer's opinion of it:
        // a future tokenizer change, or a case-sensitive column added beside this one, would make
        // the difference real, and nothing would fail in between.
        "memories_after_update",
        "CREATE TRIGGER IF NOT EXISTS memories_after_update AFTER UPDATE ON memories
         WHEN old.name COLLATE BINARY IS NOT new.name COLLATE BINARY
           OR old.description IS NOT new.description
           OR old.tags        IS NOT new.tags
           OR old.body        IS NOT new.body
         BEGIN
             INSERT INTO memories_fts(memories_fts, rowid, name, description, tags, body)
             VALUES ('delete', old.id, old.name, old.description, old.tags, old.body);
             INSERT INTO memories_fts(rowid, name, description, tags, body)
             VALUES (new.id, new.name, new.description, new.tags, new.body);
         END;",
    ),
];

/// One trigger definition reduced to the form SQLite will hand back, so this build's text and the
/// stored text can be compared at all.
///
/// **`sqlite_master.sql` is not the statement as submitted.** SQLite removes `IF NOT EXISTS` and
/// drops the trailing `;`, so comparing a [`TRIGGER_DEFINITIONS`] literal against it directly never
/// matches, in no database state whatsoever. That makes [`sync_triggers`]'s early return dead code
/// and turns every process open into three dropped triggers, three recreated ones and a full
/// `'rebuild'` of the index: measured on a 20,000-memory store, 306 ms for `meka memory get
/// <name>`, a point lookup, against 3 ms for a command that never opens the store.
pub(super) fn canonical_trigger_sql(sql: &str) -> String {
    // Whitespace is collapsed as well: this file's indentation is not something a definition should
    // be considered stale over.
    sql.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace("CREATE TRIGGER IF NOT EXISTS ", "CREATE TRIGGER ")
        .trim_end_matches(';')
        .trim_end()
        .to_string()
}

/// Replace the FTS triggers when a previous build left different ones behind. Reports whether it
/// rebuilt the index, so the caller can skip a probe that just became redundant.
///
/// The reconciliation `CREATE TRIGGER IF NOT EXISTS` cannot do, without the window that dropping
/// them unconditionally opens: six DDL statements are six auto-commit transactions, so a
/// drop-and-recreate on every open leaves every other connection a gap in which a write never
/// reaches the index, and a `SIGKILL` mid-sequence leaves the triggers gone for good. The
/// lost-update class of that is the one [`MemoryStore::integrity_check`] cannot see.
///
/// Three properties, all load-bearing:
///
/// - **It compares first and writes nothing when the definitions already match**, which is every
///   open after the first. A store that is already correct is never briefly wrong, and a process
///   start costs one `sqlite_master` read rather than six DDL transactions. Comparing needs
///   [`canonical_trigger_sql`], because the stored text is not the submitted text; without it this
///   property reads as true and is false on every run.
/// - **The replacement is one transaction.** SQLite's DDL is transactional, so a concurrent
///   connection sees the old set or the new set and never a partial one, and a crash rolls back.
/// - **The rebuild is inside that transaction.** A rebuild issued after the commit leaves, on a
///   crash in between, triggers that match over an index built by the superseded ones, so every
///   later open takes the early return and search answers from stale text for good. Nothing detects
///   that: the document counts still agree, FTS5's `integrity-check` passes, and `meka memory
///   verify` reports the store sound.
///
/// The `sqlite_master` read sits outside the transaction, which is safe because
/// [`crate::store::Store::initialize_schema`] holds an exclusive OS file lock across all
/// schema work: no second meka process can be between this check and its replacement.
fn sync_triggers(connection: &rusqlite::Connection) -> rusqlite::Result<bool> {
    let existing: std::collections::HashMap<String, String> = {
        let mut statement = connection.prepare(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'memories'",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                canonical_trigger_sql(&row.get::<_, String>(1)?),
            ))
        })?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    // Every trigger on the table is compared, not only the three by this build's names: one an
    // earlier build left under another name would otherwise stay beside the new set and feed the
    // index twice.
    if existing.len() == TRIGGER_DEFINITIONS.len()
        && TRIGGER_DEFINITIONS
            .iter()
            .all(|(name, sql)| existing.get(*name) == Some(&canonical_trigger_sql(sql)))
    {
        return Ok(false);
    }

    tracing::info!("replacing the memory search index triggers with this build's definitions");
    // `Immediate`, matching [`MemoryStore::write`] and for the same reason it gives: a deferred
    // transaction takes its write lock on the first write rather than at `BEGIN`, and under WAL
    // that upgrade can return `SQLITE_BUSY` without consulting the busy handler at all. This one
    // drops three triggers and rebuilds the whole index, so the process it would lose to is a
    // `meka serve` mid-write, not another process's schema work, which the schema `flock`
    // already excludes. Rare path, clean rollback, and no reason to differ from its sibling.
    let transaction =
        rusqlite::Transaction::new_unchecked(connection, rusqlite::TransactionBehavior::Immediate)?;
    for name in existing.keys() {
        transaction.execute_batch(&format!("DROP TRIGGER IF EXISTS {name};"))?;
    }
    for (_, sql) in TRIGGER_DEFINITIONS {
        transaction.execute_batch(sql)?;
    }
    // The index cannot be trusted once a definition has changed: whatever the old triggers did, or
    // failed to do, is already in it. A rebuild is one pass over the table and the only thing that
    // makes the new definitions true of the rows already there, and it commits with them, or not
    // at all.
    transaction.execute_batch("INSERT INTO memories_fts(memories_fts) VALUES('rebuild');")?;
    transaction.commit()?;
    Ok(true)
}

/// How many documents the table holds and how many the index believes it holds, read together.
///
/// One statement, so both counts come from one snapshot. As two `query_row` calls in autocommit
/// mode they would straddle any concurrent commit, which is a spurious disagreement on a healthy
/// store, and this number decides whether to rebuild.
fn document_counts(connection: &rusqlite::Connection) -> rusqlite::Result<(i64, i64)> {
    connection.query_row(
        // `memories_fts_docsize` holds one row per *indexed* document, so it counts what the index
        // believes; `memories` is what is true.
        "SELECT (SELECT count(*) FROM memories), (SELECT count(*) FROM memories_fts_docsize)",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )
}

/// Rebuild the index at open when it does not hold one document per stored memory.
///
/// The index is derived and disposable, so a disagreement has exactly one correct response and no
/// reason to wait for a human to run `meka memory verify`. This catches the case [`sync_triggers`]
/// cannot: a `memories_fts` dropped and recreated empty by a partial restore, which is otherwise
/// silent and permanent: search simply stops finding things.
///
/// Cheap enough to run unconditionally: measured at 20,000 memories, the two counts are 0.27 ms and
/// 0.11 ms, against 280 ms for the rebuild that only a genuinely broken store pays.
///
/// It cannot see a *changed* document whose update trigger did not fire, because the counts still
/// agree. Nothing can, short of rebuilding; see [`MemoryStore::integrity_check`].
fn repair_a_desynced_index(connection: &rusqlite::Connection) -> rusqlite::Result<()> {
    let (stored, indexed) = document_counts(connection)?;
    if stored == indexed {
        return Ok(());
    }
    tracing::warn!(
        "the memory search index holds {indexed} documents but the store holds {stored}; rebuilding it"
    );
    connection.execute_batch("INSERT INTO memories_fts(memories_fts) VALUES('rebuild');")
}

/// A model-supplied query, reduced to tokens that cannot be FTS5 syntax.
///
/// Splitting on anything non-alphanumeric already strips every operator character; quoting each
/// token on the way back out is what stops a *word* like `or` or `near` being read as one.
/// Which tier of the search ladder answered. Reported rather than kept internal: a prefix or a
/// substring answer is a guess about what the caller meant, and a reader that cannot tell it
/// from an exact hit reports the guess as a recalled fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SearchTier {
    Exact,
    Prefix,
    Substring,
}

/// What a search for words found, for every reader of memories to render in its own way.
#[derive(Debug)]
pub(crate) struct Found {
    pub(crate) tier: SearchTier,
    pub(crate) results: SearchResults,
    /// The terms the index could not split into words that were scanned for and found.
    pub(crate) scanned: Vec<String>,
    /// The terms scanned for and found nowhere.
    pub(crate) missed: Vec<String>,
    /// Whether the query also held words the index could split, beside the scanned terms.
    pub(crate) beside_words: bool,
    /// Every searchable word of the query, for a reader that tries spellings when nothing
    /// matched.
    pub(crate) words: Vec<String>,
}

/// The most hits a search answers with, and the window each half of a mixed-script query is
/// asked for before the merge is cut to the limit.
pub(crate) const MAX_SEARCH_LIMIT: usize = 25;

impl MemoryStore {
    /// The ladder every reader of memories climbs, so the agent's tool and the HTTP listing find
    /// the same memories for the same words. Four tiers, tried in order: exact handles word
    /// endings through the stemmer, prefix handles a truncation or a trailing typo, a trimmed
    /// prefix handles a query that is the longer derived form, and a substring scan handles text
    /// the tokenizer does not split into words at all. A term of a script written without spaces
    /// takes the scan every time rather than as a last resort, so a query mixing scripts is
    /// answered for both halves. `None` when `queries` hold no searchable words.
    pub(crate) async fn find(
        &self,
        queries: &[String],
        filter: &MemoryFilter,
        limit: usize,
    ) -> Result<Option<Found>> {
        let terms = Terms::parse(queries);
        if terms.is_empty() {
            return Ok(None);
        }
        let (indexable, unspaced) = terms.split_by_script();
        // Alone, the tiers answer with `limit` hits. When the halves of a mixed query are merged,
        // each side is asked for the widest window ever shown, so a memory past `limit` on words
        // alone still meets its substring hit and rises; the cut to `limit` comes after.
        let window = if unspaced.is_empty() {
            limit
        } else {
            MAX_SEARCH_LIMIT
        };
        let mut tier = SearchTier::Exact;
        let mut results = SearchResults::default();
        if !indexable.is_empty() {
            results = self
                .search(indexable.match_expression(), filter, window)
                .await?;
            if results.hits.is_empty() {
                tier = SearchTier::Prefix;
                results = self
                    .search(indexable.prefix_match_expression(), filter, window)
                    .await?;
            }
            // Still the prefix tier, in the other direction: longest prefix first, stopping at
            // the first that answers, so the most specific query that can find anything does.
            if results.hits.is_empty() {
                for expression in indexable.trimmed_prefix_match_expressions() {
                    results = self.search(expression, filter, window).await?;
                    if !results.hits.is_empty() {
                        break;
                    }
                }
            }
        }
        // The scanned terms that found something and those that found nothing, named apart so a
        // reader is never told a term matched when no memory holds it.
        let mut scanned: Vec<String> = Vec::new();
        let mut missed: Vec<String> = Vec::new();
        if !indexable.is_empty() && results.hits.is_empty() {
            // Every term takes the scan, the indexable ones included: the last resort for text
            // the tokenizer joined into one token, an identifier or a path as much as a sentence
            // of Chinese.
            tier = SearchTier::Substring;
            results = self.substring_search(terms.words(), filter, limit).await?;
        } else {
            // One scan per term, so the result can say which terms it answered for. With no
            // indexable half this is the whole answer rather than a fallback, so the tier stays
            // the one whose preamble does not say the words matched nothing.
            for term in &unspaced {
                let found = self
                    .substring_search(std::slice::from_ref(term), filter, window)
                    .await?;
                if found.hits.is_empty() {
                    missed.push(term.clone());
                } else {
                    scanned.push(term.clone());
                    results = results.merge(found, window);
                }
            }
            results.hits.truncate(limit);
        }
        Ok(Some(Found {
            tier,
            results,
            scanned,
            missed,
            beside_words: !indexable.is_empty(),
            words: terms.words().to_vec(),
        }))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Terms(Vec<String>);

impl Terms {
    /// Tokenize one or more phrasings into a single term set. Duplicate tokens collapse, so
    /// supplying "terse", "terseness" and "be terse" does not triple-count the shared word.
    ///
    /// Capped at [`MAX_TERMS`]. The tool tells the model that supplying synonyms costs nothing,
    /// which is true of a handful and not of a thousand: the substring tier builds one `OR` clause
    /// per term, and a few pasted paragraphs pushed the expression past SQLite's depth limit so the
    /// model got a raw `Expression tree is too large (maximum depth 1000)` back from what it had
    /// been encouraged to do.
    pub(crate) fn parse(queries: &[String]) -> Self {
        let mut terms: Vec<String> = Vec::new();
        for query in queries {
            for token in query.split(|character: char| {
                !character.is_alphanumeric() && character != '_' && character != '\''
            }) {
                if terms.len() >= MAX_TERMS {
                    return Self(terms);
                }
                let token = token.trim_matches('\'');
                if token.is_empty() {
                    continue;
                }
                let lowered = token.to_lowercase();
                if !terms.contains(&lowered) {
                    terms.push(lowered);
                }
            }
        }
        Self(terms)
    }

    /// Whether the query reduced to no terms at all.
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The `MATCH` expression for exact (stemmed) matching: every term OR'd.
    ///
    /// OR rather than AND because the caller is encouraged to pass several phrasings of the same
    /// question, and requiring all of them would return nothing whenever one guess was wrong.
    /// Ranking is what separates a document matching four terms from one matching one.
    pub(crate) fn match_expression(&self) -> String {
        self.render(false)
    }

    /// The same, with each term prefix-expanded (`"pref"*`), for the typo-and-truncation retry.
    pub(crate) fn prefix_match_expression(&self) -> String {
        self.render(true)
    }

    /// Prefix expressions with each term shortened by one more character each time, longest first.
    ///
    /// The plain prefix tier only works in one direction: `Tok*` reaches `Tokyo` because the
    /// stored word is the longer one. The symmetric case is just as ordinary: `deployment`
    /// searched against a memory tagged `deploy` whose body says `Deploys` is missed by every
    /// other tier, because SQLite's porter strips `-s`, `-ing` and `-ed` but not `-ment`,
    /// `deployment*` does not match `deploy` with the star on the wrong end, `LIKE '%deployment%'`
    /// cannot match a shorter word, and the edit distance is 4 against a threshold of 3.
    ///
    /// Progressive rather than one fixed cut, because the ending's length is not fixed either:
    /// `deployment` needs four characters off and `deployments` five. Longest prefix first, so the
    /// most specific query that can answer is the one that does. Bounded by [`PREFIX_TRIM_CHARS`],
    /// past which a prefix stops being a near-miss and starts being a different word, and floored
    /// at [`PREFIX_MIN_CHARS`] so a short term is never reduced to noise.
    pub(crate) fn trimmed_prefix_match_expressions(&self) -> Vec<String> {
        let mut expressions = Vec::new();
        for trim in 1..=PREFIX_TRIM_CHARS {
            let trimmed: Vec<String> = self
                .0
                .iter()
                .map(|term| {
                    let keep = term
                        .chars()
                        .count()
                        .saturating_sub(trim)
                        .max(PREFIX_MIN_CHARS);
                    term.chars().take(keep).collect()
                })
                .collect();
            // Every term already at the floor: shortening further changes nothing, and repeating
            // the same query is a round trip that cannot answer differently.
            if trimmed == self.0
                || expressions
                    .last()
                    .is_some_and(|last| *last == Terms(trimmed.clone()).render(true))
            {
                continue;
            }
            expressions.push(Terms(trimmed).render(true));
        }
        expressions
    }

    fn render(&self, prefix: bool) -> String {
        self.0
            .iter()
            .map(|term| {
                // FTS5 escapes a double quote inside a string literal by doubling it. Tokens
                // cannot contain one after `parse`, but the escape stays: the invariant lives in
                // another function, and a future tokenizer change should not become an injection.
                let quoted = format!("\"{}\"", term.replace('"', "\"\""));
                if prefix { format!("{quoted}*") } else { quoted }
            })
            .collect::<Vec<_>>()
            .join(" OR ")
    }

    /// The terms as plain words, for the edit-distance fallback that runs when FTS finds nothing.
    pub(crate) fn words(&self) -> &[String] {
        &self.0
    }

    /// The terms the index can answer for, and apart from them the ones it cannot: a term holding
    /// a character of a script written without spaces is matched as a literal substring, since
    /// `unicode61` makes a run of such text one token and a word inside it is unfindable by
    /// `MATCH`. Split rather than sent through every tier, so a query mixing scripts is answered
    /// for both halves instead of for the half the index happens to know.
    pub(crate) fn split_by_script(&self) -> (Terms, Vec<String>) {
        let (unspaced, indexable): (Vec<String>, Vec<String>) = self
            .0
            .iter()
            .cloned()
            .partition(|term| term.chars().any(crate::text::is_unspaced_script));
        (Terms(indexable), unspaced)
    }
}

/// One search result, before the caller decides how much of it to render.
#[derive(Debug, Clone)]
pub(crate) struct Hit {
    pub(crate) name: String,
    /// The project whose tier the memory is in; empty for the global tier.
    pub(crate) project: String,
    pub(crate) description: String,
    pub(crate) body: String,
    /// An excerpt of the body around the match, or its opening when the match was in the
    /// description or the name.
    pub(crate) snippet: String,
    pub(crate) priority: u8,
    pub(crate) created: SystemTime,
    pub(crate) read_count: u32,
    /// When the fact was last verified, which is what freshness decays from.
    pub(crate) verified: SystemTime,
    /// When the memory expires, if an expiry was set.
    pub(crate) expires: Option<SystemTime>,
    /// The composed [`Ranking::score`]. Higher is better, unlike the raw `bm25` it derives from.
    pub(crate) score: f64,
    /// How many parts of the query found this memory: one from the tier that answered, one more
    /// for each scanned term a merge brought in. What [`SearchResults::merge`] orders by first.
    pub(crate) found_by: u32,
}

/// What one search found: the ranked hits the caller asked for, and how many there were before the
/// cut.
///
/// `matched` is separate from `hits.len()` so a truncated result can say so. It is itself bounded
/// by [`CANDIDATE_POOL`], which `pool_exhausted` reports: at that point "200" is the size of the
/// window rather than a count of the store, and stating it as a total would be a second, quieter
/// version of the same lie.
#[derive(Debug, Default)]
pub(crate) struct SearchResults {
    pub(crate) hits: Vec<Hit>,
    pub(crate) matched: usize,
    pub(crate) pool_exhausted: bool,
}

/// Which tiers a reader sees, and the predicate every read, update and delete by name carries.
///
/// One definition so no door sees a tier another hides: a session on a project sees the global
/// tier and its own, a session with no project sees the global tier alone, and the operator sees
/// the tier a flag or query parameter names, or everything. The default is the narrowest view, so
/// a door that forgets to say which it is hides a tier rather than leaking one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) enum MemoryView {
    /// The global tier alone: a session with no project, or the operator asking for it.
    #[default]
    Global,
    /// The global tier and one project's: what a session on that project sees.
    Session(String),
    /// One project's tier alone: the operator's project filter.
    Project(String),
    /// Every tier: the operator with no filter.
    All,
}

impl MemoryView {
    /// What a session on `project` sees; `None` or empty is a session with no project.
    pub(crate) fn for_session(project: Option<&str>) -> Self {
        match project {
            Some(project) if !project.is_empty() => Self::Session(project.to_string()),
            _ => Self::Global,
        }
    }

    /// The tiers an operator's `--project` flag or `?project=` parameter names: every tier without
    /// it, the global tier for an empty value, one project's for a name.
    pub(crate) fn for_operator(project: Option<&str>) -> Self {
        match project {
            None => Self::All,
            Some("") => Self::Global,
            Some(project) => Self::Project(project.to_string()),
        }
    }

    /// The project a write through this view creates a memory in when it names no tier: the
    /// view's own project, or global for a view on no project.
    pub(crate) fn project(&self) -> &str {
        match self {
            Self::Session(project) | Self::Project(project) => project,
            Self::Global | Self::All => "",
        }
    }

    /// The tiers whose memories can cite the one in `project`: every tier for a global memory,
    /// the global tier and the project's own for a project's, since a reference is a bare name
    /// read within the citer's view.
    pub(crate) fn citing(project: &str) -> Self {
        if project.is_empty() {
            Self::All
        } else {
            Self::Session(project.to_string())
        }
    }

    /// Whether a memory in `project` (empty for the global tier) is within the view.
    pub(crate) fn sees(&self, project: &str) -> bool {
        match self {
            Self::All => true,
            Self::Global => project.is_empty(),
            Self::Session(own) => project.is_empty() || project == own,
            Self::Project(own) => project == own,
        }
    }

    /// The `AND ...` tail selecting the rows within the view and the parameter it binds, numbered
    /// `position`; `table` as in [`MemoryFilter::clause`]. Empty for [`Self::All`].
    fn clause(&self, table: &str, position: usize) -> (String, Vec<String>) {
        match self {
            Self::All => (String::new(), Vec::new()),
            Self::Global => (format!(" AND {table}project = ''"), Vec::new()),
            Self::Session(project) => (
                format!(" AND ({table}project = '' OR {table}project = ?{position})"),
                vec![project.clone()],
            ),
            Self::Project(project) => (format!(" AND {table}project = ?{position}"), vec![
                project.clone(),
            ]),
        }
    }
}

/// What a listing or a search is narrowed to, the same three questions at every door:
/// `memory_search`'s filters, `meka memory list`'s flags and `GET /v1/memory`'s query parameters,
/// within the tiers the caller sees.
///
/// One definition so the doors cannot disagree about what "referencing" or "unverified" selects,
/// and so a search with words and a listing without them read the same rows.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct MemoryFilter {
    /// The tiers the caller sees. Not a filter in [`Self::is_empty`]'s sense: a view narrows what
    /// exists for the caller, where the questions below pick rows out of it.
    pub(crate) view: MemoryView,
    /// Memories whose references name this, matched whole and without regard to case.
    pub(crate) referencing: Option<String>,
    /// Memories not verified within this long of now.
    pub(crate) unverified_for: Option<Duration>,
    /// Memories whose expiry has passed.
    pub(crate) expired: bool,
}

impl MemoryFilter {
    /// Whether the filter selects everything.
    pub(crate) fn is_empty(&self) -> bool {
        self.referencing.is_none() && self.unverified_for.is_none() && !self.expired
    }

    /// The `AND ...` tail of a `WHERE` clause and the parameters it binds, numbered from
    /// `first`; `table` is the alias prefix the statement uses for `memories`, `"m."` or `""`.
    /// Empty for an empty filter.
    ///
    /// Stamps are compared as the RFC 3339 text the columns hold, which orders correctly because
    /// every stamp meka writes is UTC with the same field widths; the parameters are strings so
    /// the closure that binds them stays `Send`.
    fn clause(&self, table: &str, first: usize, now: SystemTime) -> (String, Vec<String>) {
        let (mut clause, mut parameters) = self.view.clause(table, first);
        if let Some(target) = &self.referencing {
            clause.push_str(" AND ");
            clause.push_str(&Self::referencing_sql(table, first + parameters.len()));
            parameters.push(target.clone());
        }
        if let Some(window) = self.unverified_for {
            clause.push_str(&format!(
                " AND {table}verified_at < ?{}",
                first + parameters.len()
            ));
            parameters.push(crate::memory::render_stamp(
                now.checked_sub(window).unwrap_or(SystemTime::UNIX_EPOCH),
            ));
        }
        if self.expired {
            clause.push_str(&format!(
                " AND {table}expires_at IS NOT NULL AND {table}expires_at <= ?{}",
                first + parameters.len()
            ));
            parameters.push(crate::memory::render_stamp(now));
        }
        (clause, parameters)
    }

    /// The condition selecting rows whose reference list names parameter `position`.
    ///
    /// `json_each` over the row's own column, so the match is against each reference whole;
    /// `NOCASE` because memory names are matched without regard to case everywhere else and a
    /// listing that shows one row too many is harmless where one too few is not.
    fn referencing_sql(table: &str, position: usize) -> String {
        format!(
            "EXISTS (SELECT 1 FROM json_each({table}reference_list) \
             WHERE json_each.value = ?{position} COLLATE NOCASE)"
        )
    }
}

impl SearchResults {
    /// These results and `other`'s as one ranked list, at most `limit` long. A memory in both is
    /// listed once and ordered first, ahead of any found by one side alone, whatever the scores
    /// say: a word hit's score is a bm25 magnitude and a scan hit's is flat, so adding them would
    /// let a strong name match outrank a memory that holds every part of the query. Among
    /// memories found by the same number of parts, the added [`Ranking::score`]s order them.
    ///
    /// `matched` stays a lower bound: neither side's unseen matches can be told apart from the
    /// other's, so the larger count, or the shown union when that is larger, is what is certainly
    /// true. The pool is exhausted when either side's was.
    pub(crate) fn merge(mut self, other: SearchResults, limit: usize) -> SearchResults {
        let matched = self.matched.max(other.matched);
        for hit in other.hits {
            // Keyed on the tier as well as the name: under a view over every tier, two projects
            // may each hold a memory of one name, and those are two hits.
            match self
                .hits
                .iter_mut()
                .find(|known| known.name == hit.name && known.project == hit.project)
            {
                Some(known) => {
                    known.found_by += hit.found_by;
                    known.score += hit.score;
                }
                None => self.hits.push(hit),
            }
        }
        self.hits.sort_by(|left, right| {
            right
                .found_by
                .cmp(&left.found_by)
                .then(right.score.total_cmp(&left.score))
        });
        self.matched = matched.max(self.hits.len());
        self.pool_exhausted = self.pool_exhausted || other.pool_exhausted;
        self.hits.truncate(limit);
        self
    }
}

/// The three multipliers that turn textual relevance into "which of these did you actually mean".
///
/// Split out from the query so each factor is separately testable and so the weights live in one
/// readable place rather than inside a SQL string.
pub(crate) struct Ranking;

impl Ranking {
    /// `relevance x priority x usage x freshness`, all of them positive and increasing-is-better.
    /// `age` is how long since the fact was last verified, so a re-verified old note ranks as
    /// fresh and a note nobody has checked decays from the day it was written.
    ///
    /// `bm25` is negated on the way in. FTS5 returns a *negative* number whose magnitude grows with
    /// relevance, so multiplying it by a positive weight ranks the least relevant result first, and
    /// nothing about the output looks wrong.
    pub(crate) fn score(bm25: f64, priority: u8, read_count: u32, age: Duration) -> f64 {
        let relevance = -bm25;
        relevance
            * Self::priority_weight(priority)
            * Self::usage_weight(read_count)
            * Self::freshness(priority, age)
    }

    /// What the agent said this was worth when it wrote it. Spans `1.0 ..= 1.0 + span`.
    fn priority_weight(priority: u8) -> f64 {
        let inverted = f64::from(MAX_PRIORITY.saturating_sub(priority)) / f64::from(MAX_PRIORITY);
        1.0 + PRIORITY_WEIGHT_SPAN * inverted
    }

    /// What the agent has since *done* with it. Spans `1.0 ..= 2.0`.
    ///
    /// The counterweight to a priority chosen once and never revised. Priorities drift toward 0
    /// over a long-lived instance until the index stops ranking anything; a memory opened forty
    /// times is important whatever it was labeled two years ago, and that is a fact meka owns
    /// rather than one it has to guess.
    fn usage_weight(read_count: u32) -> f64 {
        if read_count == 0 {
            return 1.0;
        }
        let saturation = (1.0 + USAGE_SATURATION_READS).ln();
        1.0 + (f64::from(read_count) + 1.0).ln().min(saturation) / saturation
    }

    /// How much an old observation is still worth. `1.0` for a standing directive; otherwise a
    /// gentle half-life floored at [`FRESHNESS_FLOOR`].
    fn freshness(priority: u8, age: Duration) -> f64 {
        if priority <= STANDING_PRIORITY_MAX {
            return 1.0;
        }
        let days = age.as_secs_f64() / 86_400.0;
        (0.5_f64.powf(days / FRESHNESS_HALF_LIFE_DAYS)).max(FRESHNESS_FLOOR)
    }
}

/// What [`MemoryStore::write`] is being asked to store. `None` means "leave whatever is there".
///
/// `Default` exists for a test that sets a few fields; every production door names them all.
#[derive(Debug, Clone, Default)]
pub(crate) struct WriteRequest {
    pub(crate) name: String,
    /// The tiers the writer sees, which decides what an unnamed tier means and which memory of
    /// the name it may move; see [`resolve_write_tier`].
    pub(crate) view: MemoryView,
    /// The tier the memory is to be in, empty for global. `None` keeps the stored tier, or for a
    /// creating write takes the view's own project.
    pub(crate) project: Option<String>,
    /// `None` means "keep what is stored", exactly as it does for the optional fields below.
    ///
    /// Required only when the write creates the memory, which [`MemoryStore::write`] checks inside
    /// its own transaction. As a plain `String`, a caller refining a stored note would have to
    /// resend the description, and the only copy an agent can see is the index's, elided to 500
    /// characters: refining the body of a memory whose description ran to 900 characters would
    /// rewrite it as 503 ending in `...`, silently, on a call that did not mean to touch it.
    pub(crate) description: Option<String>,
    pub(crate) tags: Option<Vec<String>>,
    pub(crate) body: Option<String>,
    pub(crate) priority: Option<u8>,
    pub(crate) load: Option<crate::memory::Load>,
    pub(crate) references: Option<Vec<String>>,
    pub(crate) source: Option<String>,
    /// Outer `None` keeps the stored expiry; `Some(None)` clears it.
    pub(crate) expires_at: Option<Option<SystemTime>>,
    /// Whether this write verifies the fact holds, which is the one thing that moves
    /// `verified_at` on an existing row. A creating write stamps it regardless, since a new note
    /// is as verified as it will ever be.
    pub(crate) verified: bool,
}

/// What [`MemoryStore::write_body`] found when it went to save.
///
/// Three answers rather than a `bool`, because "nothing was written" has two causes and the caller
/// has to say which: the note is gone, or its text moved while the editor was open. Collapsing them
/// would report a lost update as a deletion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BodyWrite {
    Saved,
    /// No memory by that name any more.
    Gone,
    /// The stored body is no longer the one the caller started from.
    ChangedUnderneath,
}

/// Read one `memories` row in the column order every query here selects.
///
/// One function so the order lives in one place: a `SELECT` that reorders its columns and a reader
/// that does not is a silent field swap, and every one of these columns is a string or a number
/// that would survive being put in the wrong slot.
///
/// **This returns what is stored, byte for byte.** Sanitizing here instead is silent data loss:
/// `meka memory edit` reads a body, hands it to `$EDITOR` and writes back what comes out, so a
/// read that strips format characters makes an edit to one unrelated word destroy every
/// zero-width joiner in the note (the one holding an emoji sequence together, the one a Persian
/// word needs), and `meka memory show` then displays the already-stripped text, so nothing
/// reveals the loss.
///
/// Sanitization belongs at the render boundary instead, and the rule is that a path which
/// *displays* text sanitizes while a path which *round-trips* it does not. See
/// [`crate::memory::render_for_model`].
fn row_to_memory(row: &rusqlite::Row<'_>) -> rusqlite::Result<Memory> {
    let tags: String = row.get(2)?;
    let load: String = row.get(8)?;
    let references: String = row.get(9)?;
    let expires_at: Option<String> = row.get(12)?;
    Ok(Memory {
        name: row.get(0)?,
        project: row.get(13)?,
        description: row.get(1)?,
        tags: tags
            .split_whitespace()
            .map(keepable_tag)
            .filter(|tag| !tag.is_empty())
            .collect::<Vec<_>>(),
        priority: clamp_priority(row.get(3)?),
        created_at: stamp_or_epoch(&row.get::<_, String>(4)?),
        updated_at: stamp_or_epoch(&row.get::<_, String>(5)?),
        read_count: clamp_read_count(row.get(6)?),
        body: row.get(7)?,
        load: load.parse().unwrap_or_default(),
        references: keepable_references(&references),
        source: row.get(10)?,
        verified_at: stamp_or_epoch(&row.get::<_, String>(11)?),
        expires_at: expires_at.as_deref().and_then(crate::memory::parse_stamp),
    })
}

/// Every column of a `memories` row, in the order [`row_to_memory`] reads them.
///
/// One constant so a statement cannot select the columns in another order: every one of them is
/// a string or a number that would survive being put in the wrong slot.
const MEMORY_COLUMNS: &str = "name, description, tags, priority, created_at, updated_at, \
                              read_count, body, load, reference_list, source, verified_at, \
                              expires_at, project";

/// The references a stored row holds, or none when the column is not the JSON array of one-line
/// strings every write door stores.
///
/// All-or-nothing per element, by the write door's own rule, for the reason [`keepable_tag`]
/// gives: a reference that fails [`crate::memory::validate_reference`] is dropped whole rather
/// than trimmed into a different, plausible pointer.
fn keepable_references(stored: &str) -> Vec<String> {
    let Ok(serde_json::Value::Array(values)) = serde_json::from_str::<serde_json::Value>(stored)
    else {
        return Vec::new();
    };
    values
        .into_iter()
        .filter_map(|value| match value {
            serde_json::Value::String(reference)
                if crate::memory::validate_reference(&reference).is_ok() =>
            {
                Some(reference)
            }
            _ => None,
        })
        .collect()
}

/// The JSON array a write stores a reference list as.
fn references_column(references: &[String]) -> String {
    serde_json::Value::Array(
        references
            .iter()
            .map(|reference| serde_json::Value::String(reference.clone()))
            .collect(),
    )
    .to_string()
}

/// Reduce a stored tag to the characters a tag is allowed to hold.
///
/// The same shape as [`clamp_priority`] and [`clamp_read_count`], and against the same threat: a
/// row that reached this table without going through meka's write doors. Unlike the others, tags
/// are *rendered*, into the tag histogram and the world-state diff, both of which reach the
/// model's context and the operator's terminal. `split_whitespace` already stops a newline, but an
/// ANSI escape, a bidi override or a zero-width character is not whitespace and goes through
/// untouched.
///
/// All-or-nothing, against the write door's own rule. Filtering character by character is worse
/// than refusing, because for the most likely input it does not remove an odd character; it
/// silently produces a *different, entirely plausible* tag. A stored `Infra` comes back as `nfra`:
/// still a valid tag, still rendered without complaint, and wrong. `Q3-2026` becomes `3-2026`.
/// Nothing in the histogram or the diff marks a tag as altered, so the corruption is invisible at
/// every surface that displays it.
///
/// The predicate is [`crate::memory::validate_tag`] itself rather than a second character list, so
/// what a read keeps is exactly what a write would have accepted, one rule in one place. That
/// also settles the uppercase case the same way the write door does: `validate_tag` *refuses*
/// `Infra` rather than lowercasing it, so a read that folded case would be inventing a third
/// policy. The tag drops, and the caller filters the empty string out, which keeps a nameless entry
/// out of the tag histogram (`most common tags  (3)`) and a stray separator out of the diff's
/// `[deploy, ]`.
fn keepable_tag(stored: &str) -> String {
    if crate::memory::validate_tag(stored).is_ok() {
        stored.to_string()
    } else {
        String::new()
    }
}

/// Clamp a stored priority into the band. Stored values come from meka's own validated write door,
/// so this is a guard against a hand-edited database rather than an expected path.
fn clamp_priority(stored: i64) -> u8 {
    stored.clamp(0, i64::from(MAX_PRIORITY)) as u8
}

fn clamp_read_count(stored: i64) -> u32 {
    stored.clamp(0, i64::from(u32::MAX)) as u32
}

/// Parse a stored RFC 3339 stamp, falling back to the epoch.
///
/// Every stamp this reads was written by [`crate::memory::render_stamp`], so the fallback is
/// unreachable through meka's own doors. It exists because the alternative is failing a whole
/// query over one malformed cell in a database somebody edited by hand, and an epoch date renders
/// as an obviously wrong age rather than as a plausible one.
fn stamp_or_epoch(raw: &str) -> SystemTime {
    crate::memory::parse_stamp(raw).unwrap_or(SystemTime::UNIX_EPOCH)
}

/// Handle on the memory tables, sharing the one database `Store` owns.
///
/// A handle rather than a second database file: meka has one database, and a second would be a new
/// thing to back up, lock, and explain for the sake of two tables. Cloneable and cheap; the
/// underlying connection is shared, and a transaction is what serializes concurrent writers.
pub(crate) struct MemoryStore {
    /// `None` for a store with no database behind it: `meka tool list`, which prints the
    /// catalog without running anything, and test fixtures that never touch the store. Reads
    /// answer empty; writes report that there is nowhere to write.
    connection: Option<Arc<Connection>>,
    /// Whether the subsystem is switched on at all, from `[memory] enabled`.
    ///
    /// Deliberately separate from [`Self::connection`], because they answer different questions.
    /// A store with no database is an *empty* store and its `memory_*` tools still belong in the
    /// registry; conflating the two makes `meka tool list` hide tools a real session would have.
    /// A *disabled* store is one whose tools are not registered and whose `[Memory]` section
    /// never renders, but which the CLI and the HTTP API still read and write, because those are
    /// the operator rather than the agent.
    enabled: bool,
    /// `[memory] inline_max_bytes`: what the bodies of the in-force inline memories may total.
    ///
    /// Enforced inside [`Self::write`] and [`Self::write_body`], the two statements every door's
    /// write goes through, so no door can leave the tier larger than the band renders whole.
    inline_max_bytes: usize,
    /// Serializes `memory_write`'s near-duplicate check against the write that follows it. See
    /// [`Self::lock_duplicate_check`].
    duplicate_check: tokio::sync::Mutex<()>,
}

/// What a guarded write statement decided, inside its transaction.
///
/// A refusal is returned rather than raised so the closure's `rusqlite` errors keep their type;
/// the transaction is dropped uncommitted on that path, which is what makes "refused" mean
/// "nothing landed".
enum Guarded<T> {
    Committed(T),
    Refused(String),
}

/// The bytes the in-force inline bodies total as the band of a session sees them: the UTF-8 bytes
/// of the stored body of every `inline` row whose expiry has not passed, in the global tier and
/// in `project`'s. For the global tier itself the project counted is the one with the most, since
/// a global memory is in every project's band and the budget holds only if it holds in the
/// fullest. Stamps compare as text for the reason [`MemoryFilter::clause`] gives.
fn inline_bytes_in_force(
    connection: &rusqlite::Connection,
    now: &str,
    project: &str,
) -> rusqlite::Result<usize> {
    let inline = crate::memory::Load::Inline.name();
    let total: i64 = if project.is_empty() {
        connection.query_row(
            "SELECT (SELECT COALESCE(SUM(LENGTH(CAST(body AS BLOB))), 0) FROM memories
                     WHERE load = ?1 AND (expires_at IS NULL OR expires_at > ?2)
                       AND project = '')
                  + (SELECT COALESCE(MAX(total), 0) FROM
                        (SELECT SUM(LENGTH(CAST(body AS BLOB))) AS total FROM memories
                         WHERE load = ?1 AND (expires_at IS NULL OR expires_at > ?2)
                           AND project != ''
                         GROUP BY project))",
            rusqlite::params![inline, now],
            |row| row.get(0),
        )?
    } else {
        connection.query_row(
            "SELECT COALESCE(SUM(LENGTH(CAST(body AS BLOB))), 0) FROM memories
             WHERE load = ?1 AND (expires_at IS NULL OR expires_at > ?2)
               AND (project = '' OR project = ?3)",
            rusqlite::params![inline, now, project],
            |row| row.get(0),
        )?
    };
    Ok(usize::try_from(total).unwrap_or(0))
}

/// The project whose in-force inline bodies total the most, or `None` when no project has one:
/// the project a global write's budget was measured with.
fn largest_inline_project(
    connection: &rusqlite::Connection,
    now: &str,
) -> rusqlite::Result<Option<String>> {
    connection
        .query_row(
            "SELECT project FROM memories
             WHERE load = ?1 AND (expires_at IS NULL OR expires_at > ?2) AND project != ''
             GROUP BY project
             ORDER BY SUM(LENGTH(CAST(body AS BLOB))) DESC, project ASC
             LIMIT 1",
            rusqlite::params![crate::memory::Load::Inline.name(), now],
            |row| row.get(0),
        )
        .optional()
}

/// Whether the write this transaction holds may commit, or the refusal it gets.
///
/// Refused only when the tier ends over the budget *and* larger than it was: a write that trims an
/// overfull tier, or leaves its size alone, always lands, or a store the budget was lowered under
/// could never be brought back within it one memory at a time.
fn inline_budget_refusal(
    connection: &rusqlite::Connection,
    now: &str,
    before: usize,
    max_bytes: usize,
    project: &str,
    view: &MemoryView,
) -> rusqlite::Result<Option<String>> {
    let after = inline_bytes_in_force(connection, now, project)?;
    if after <= max_bytes || after <= before {
        return Ok(None);
    }
    // The band the total was measured for: the project's own, or for a global write the fullest
    // project's, so the names below are the ones that band shows.
    let measured = if project.is_empty() {
        largest_inline_project(connection, now)?
    } else {
        Some(project.to_string())
    };
    // Named only within the writer's view: a memory it cannot read is not one it can shorten.
    let (visible, view_parameters) = view.clause("", 5);
    let mut statement = connection.prepare(&format!(
        "SELECT name, LENGTH(CAST(body AS BLOB)) FROM memories
         WHERE load = ?1 AND (expires_at IS NULL OR expires_at > ?2)
           AND (project = '' OR project = ?4) {visible}
         ORDER BY LENGTH(CAST(body AS BLOB)) DESC, name ASC
         LIMIT ?3"
    ))?;
    let mut bound: Vec<rusqlite::types::Value> = vec![
        crate::memory::Load::Inline.name().to_string().into(),
        now.to_string().into(),
        i64::try_from(crate::memory::INLINE_REFUSAL_NAMES)
            .unwrap_or(i64::MAX)
            .into(),
        measured.clone().unwrap_or_default().into(),
    ];
    bound.extend(
        view_parameters
            .into_iter()
            .map(rusqlite::types::Value::from),
    );
    let largest = statement
        .query_map(rusqlite::params_from_iter(bound), |row| {
            Ok((
                row.get::<_, String>(0)?,
                usize::try_from(row.get::<_, i64>(1)?).unwrap_or(0),
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(Some(crate::memory::render_inline_budget_refusal(
        after,
        max_bytes,
        &largest,
        measured.as_deref(),
    )))
}

impl MemoryStore {
    /// `enabled` is `[memory] enabled`, and it gates *tool registration* only. The connection is
    /// attached either way, because that switch decides whether an agent keeps memories, not
    /// whether the operator can reach what is already stored: `meka memory list` and
    /// `GET /v1/memory` both work on a disabled installation, and a backup you cannot take because
    /// you turned the feature off is a trap rather than a safeguard.
    pub(crate) fn new(
        connection: Arc<Connection>,
        enabled: bool,
        inline_max_bytes: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            connection: Some(connection),
            enabled,
            inline_max_bytes,
            duplicate_check: tokio::sync::Mutex::new(()),
        })
    }

    /// A store with no database: empty at every read, and refusing every write with a message that
    /// names the cause rather than reporting silent success.
    pub(crate) fn detached() -> Arc<Self> {
        Arc::new(Self {
            connection: None,
            enabled: true,
            inline_max_bytes: crate::config::MEMORY_INLINE_MAX_BYTES_DEFAULT,
            duplicate_check: tokio::sync::Mutex::new(()),
        })
    }

    /// A disabled store with no database behind it either. For tests and for callers that have
    /// neither; production hands [`Self::new`] a `false` instead, so the operator
    /// doors keep working.
    pub(crate) fn disabled() -> Arc<Self> {
        Arc::new(Self {
            connection: None,
            enabled: false,
            inline_max_bytes: crate::config::MEMORY_INLINE_MAX_BYTES_DEFAULT,
            duplicate_check: tokio::sync::Mutex::new(()),
        })
    }

    /// Whether the subsystem is switched on. See the field docs on [`Self::enabled`].
    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    /// What the in-force inline bodies may total; see [`Self::inline_max_bytes`].
    pub(crate) fn inline_max_bytes(&self) -> usize {
        self.inline_max_bytes
    }

    /// The same database under a different inline budget, which is what a store looks like after
    /// `[memory] inline_max_bytes` is lowered beneath what it already holds.
    #[cfg(test)]
    pub(crate) fn with_inline_max_bytes_for_test(&self, inline_max_bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            connection: self.connection.clone(),
            enabled: self.enabled,
            inline_max_bytes,
            duplicate_check: tokio::sync::Mutex::new(()),
        })
    }

    /// Serialize a `memory_write`'s near-duplicate check against the write that follows it.
    ///
    /// Deliberately *not* a write lock. [`Self::write`] is one immediate transaction, and SQLite
    /// serializes writers across processes, so nothing about correctness needs this. What
    /// needs it is the advisory check, which is a read followed by a report: a model emits several
    /// tool calls in one message and meka runs them concurrently, so without this two
    /// `memory_write`s in one turn both query the store before either lands, each sees no
    /// duplicate, and the check that exists to catch exactly this says nothing. Observed live.
    ///
    /// That shape is not rare. A compaction checkpoint saves several memories at once, which is
    /// precisely when near-copies are most likely.
    ///
    /// Held by the caller across its whole check-then-write; an in-process mutex is enough because
    /// the check only exists in the tool, and the concurrency it guards against is one model's
    /// batched tool calls inside one process.
    pub(crate) async fn lock_duplicate_check(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.duplicate_check.lock().await
    }

    /// A standalone in-memory store, for tests and for callers that want one without a session
    /// database. Schema is created eagerly, so the handle is usable on return.
    ///
    /// Built by [`crate::store::migrations`], the same ledger `initialize_schema` runs, so this
    /// store is the shape a real one is rather than a second opinion about it. A fresh
    /// `:memory:` database is unversioned and empty, so the ledger classifies it as new and runs
    /// every step.
    #[cfg(test)]
    pub(crate) async fn for_test() -> Result<Arc<Self>> {
        let connection = Connection::open_in_memory().await.map_err(|error| {
            MekaError::Database(format!("failed to open the memory store: {error}"))
        })?;
        connection
            .call(|connection| -> std::result::Result<_, MekaError> {
                let plan = crate::store::migrations::plan(connection)?;
                // Nothing to carry forward: this store is created empty by this very call, so the
                // step that gives existing sessions a provider has no rows to give one to.
                crate::store::migrations::apply(
                    connection,
                    plan,
                    &crate::store::migrations::Context::default(),
                )?;
                reconcile_index(connection).map_err(|error| {
                    MekaError::Database(format!("failed to reconcile the memory index: {error}"))
                })
            })
            .await
            .map_err(|error| match error {
                tokio_rusqlite::Error::Error(inner) => inner,
                other => {
                    MekaError::Database(format!("failed to create the memory tables: {other}"))
                }
            })?;
        Ok(Arc::new(Self {
            connection: Some(Arc::new(connection)),
            enabled: true,
            inline_max_bytes: crate::config::MEMORY_INLINE_MAX_BYTES_DEFAULT,
            duplicate_check: tokio::sync::Mutex::new(()),
        }))
    }

    /// The connection, or the error a write reports when there is none.
    ///
    /// Reads short-circuit to an empty result instead of calling this: an empty store and a store
    /// with no database answer a question the same way, and there is nothing for the caller to do
    /// about the difference. A *write* that quietly did nothing is the failure worth naming.
    fn writable(&self) -> Result<&Connection> {
        self.connection.as_deref().ok_or_else(|| {
            MekaError::Database(if self.enabled {
                "no memory database is open in this process".to_string()
            } else {
                "memory is disabled in this configuration".to_string()
            })
        })
    }

    /// How many memories the view holds.
    pub(crate) async fn count(&self, view: &MemoryView) -> Result<u64> {
        let Some(connection) = self.connection.as_deref() else {
            return Ok(0);
        };
        let view = view.clone();
        connection
            .call(move |connection| -> rusqlite::Result<_> {
                let (clause, parameters) = view.clause("", 1);
                connection.query_row(
                    &format!("SELECT count(*) FROM memories WHERE 1 = 1 {clause}"),
                    rusqlite::params_from_iter(parameters),
                    |row| row.get::<_, i64>(0),
                )
            })
            .await
            .map(|count| u64::try_from(count).unwrap_or(0))
            .map_err(|error| MekaError::Database(format!("failed to count memories: {error}")))
    }

    /// Every memory within the view, ordered as the `[Memory]` index renders them: priority
    /// ascending, then most recently created, then by name so the result is total and stable.
    ///
    /// Bodies are loaded only for the inline band ([`crate::memory::Load::Inline`]), which is
    /// the only tier that renders one in full. Carrying every body would put the whole
    /// store in resident memory for the sake of a handful of entries, and the rest are reachable
    /// through [`Self::get`] and [`Self::search`].
    pub(crate) async fn index(&self, view: &MemoryView) -> Result<Vec<Memory>> {
        let Some(connection) = self.connection.as_deref() else {
            return Ok(Vec::new());
        };
        let inline = crate::memory::Load::Inline.name().to_string();
        let (clause, view_parameters) = view.clause("", 2);
        connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(&format!(
                    "SELECT name, description, tags, priority, created_at, updated_at, read_count,
                            CASE WHEN load = ?1 THEN body ELSE NULL END,
                            load, reference_list, source, verified_at, expires_at, project
                     FROM memories
                     WHERE 1 {clause}
                     ORDER BY priority ASC, created_at DESC, name ASC"
                ))?;
                let mut bound = vec![inline];
                bound.extend(view_parameters);
                let rows = statement.query_map(rusqlite::params_from_iter(bound), row_to_memory)?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to read memories: {error}")))
    }

    /// The memories a filter selects, in index order and without their bodies, for a listing.
    ///
    /// Beside [`Self::index`] rather than folded into it: the index is every memory with the
    /// inline bodies, read once per turn, and a listing is what an operator or a hygiene pass
    /// asked for, which is usually a few rows out of the store.
    pub(crate) async fn list(&self, filter: &MemoryFilter) -> Result<Vec<Memory>> {
        let Some(connection) = self.connection.as_deref() else {
            return Ok(Vec::new());
        };
        let (clause, parameters) = filter.clause("", 1, SystemTime::now());
        connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(&format!(
                    "SELECT name, description, tags, priority, created_at, updated_at, read_count,
                            NULL, load, reference_list, source, verified_at, expires_at, project
                     FROM memories
                     WHERE 1 {clause}
                     ORDER BY priority ASC, created_at DESC, name ASC"
                ))?;
                let rows =
                    statement.query_map(rusqlite::params_from_iter(parameters), row_to_memory)?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to list memories: {error}")))
    }

    /// The memories within the view whose references name `target`, in index order: name and
    /// description, which is what a reader needs to name them and what a sweep needs to decide
    /// which to revisit.
    pub(crate) async fn referenced_by(
        &self,
        target: &str,
        view: &MemoryView,
    ) -> Result<Vec<(String, String)>> {
        let Some(connection) = self.connection.as_deref() else {
            return Ok(Vec::new());
        };
        let target = target.to_string();
        let (clause, view_parameters) = view.clause("", 2);
        connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(&format!(
                    "SELECT name, description FROM memories
                     WHERE {} {clause}
                     ORDER BY priority ASC, created_at DESC, name ASC",
                    MemoryFilter::referencing_sql("", 1)
                ))?;
                let mut bound = vec![target];
                bound.extend(view_parameters);
                let rows = statement.query_map(rusqlite::params_from_iter(bound), |row| {
                    Ok((row.get(0)?, row.get(1)?))
                })?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to read memory references: {error}"))
            })
    }

    /// The memories a filter alone selects, bodies included, as search results in index order:
    /// what `memory_search` answers when it was given a filter and no words.
    ///
    /// Not re-ranked. Every row matched the filter equally, so the importance order the index
    /// uses is the one a reader expects, and a hygiene pass reads the list top to bottom.
    pub(crate) async fn filtered(
        &self,
        filter: &MemoryFilter,
        limit: usize,
    ) -> Result<SearchResults> {
        let Some(connection) = self.connection.as_deref() else {
            return Ok(SearchResults::default());
        };
        let (clause, parameters) = filter.clause("", 1, SystemTime::now());
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        connection
            .call(move |connection| -> rusqlite::Result<_> {
                let matched: i64 = connection.query_row(
                    &format!("SELECT count(*) FROM memories WHERE 1 {clause}"),
                    rusqlite::params_from_iter(parameters.iter()),
                    |row| row.get(0),
                )?;
                let mut statement = connection.prepare(&format!(
                    "SELECT name, description, body, priority, created_at, read_count,
                            verified_at, expires_at, project
                     FROM memories
                     WHERE 1 {clause}
                     ORDER BY priority ASC, created_at DESC, name ASC
                     LIMIT ?{}",
                    parameters.len() + 1
                ))?;
                let bound: Vec<&dyn rusqlite::ToSql> = parameters
                    .iter()
                    .map(|parameter| parameter as &dyn rusqlite::ToSql)
                    .chain(std::iter::once(&limit as &dyn rusqlite::ToSql))
                    .collect();
                let rows = statement.query_map(bound.as_slice(), |row| {
                    let body: String = row.get(2)?;
                    Ok(Hit {
                        name: row.get(0)?,
                        project: row.get(8)?,
                        description: row.get(1)?,
                        // The opening, since nothing was matched: what a listing's excerpt is.
                        snippet: body.chars().take(SUBSTRING_SNIPPET_CHARS).collect(),
                        body,
                        priority: clamp_priority(row.get(3)?),
                        created: stamp_or_epoch(&row.get::<_, String>(4)?),
                        read_count: clamp_read_count(row.get(5)?),
                        verified: stamp_or_epoch(&row.get::<_, String>(6)?),
                        expires: row
                            .get::<_, Option<String>>(7)?
                            .as_deref()
                            .and_then(crate::memory::parse_stamp),
                        score: 0.0,
                        found_by: 1,
                    })
                })?;
                let hits = rows.collect::<rusqlite::Result<Vec<_>>>()?;
                Ok(SearchResults {
                    hits,
                    matched: usize::try_from(matched).unwrap_or(usize::MAX),
                    pool_exhausted: false,
                })
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to list memories: {error}")))
    }

    /// Insert a row straight at the column, past every write door.
    ///
    /// Exists so a test can produce the state the read guards are for: a name, description or tag
    /// that meka's own doors would have refused, which is what a row written by something other
    /// than this build looks like. Nothing in meka can create one, so without this the guards would
    /// be unreachable from the test suite and rest on their comments alone.
    #[cfg(test)]
    pub(crate) async fn plant_row_for_test(&self, name: &str, description: &str) -> Result<()> {
        let Some(connection) = self.connection.as_deref() else {
            return Err(MekaError::Database(
                "memory store has no database".to_string(),
            ));
        };
        let name = name.to_string();
        let description = description.to_string();
        connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO memories (name, description, created_at, updated_at, verified_at) \
                     VALUES (?1, ?2, '2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00', \
                             '2024-01-01T00:00:00+00:00')",
                    rusqlite::params![name, description],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to plant a row: {error}")))
    }

    /// Overwrite a row's reference list straight at the column, past every write door, so the
    /// read guard for a list no door would store is reachable from the suite.
    #[cfg(test)]
    pub(crate) async fn plant_references_for_test(&self, name: &str, raw: &str) -> Result<()> {
        let Some(connection) = self.connection.as_deref() else {
            return Err(MekaError::Database(
                "memory store has no database".to_string(),
            ));
        };
        let (name, raw) = (name.to_string(), raw.to_string());
        connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE memories SET reference_list = ?2 WHERE name = ?1",
                    rusqlite::params![name, raw],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to plant references: {error}")))
    }

    /// Rename a row straight at the column.
    ///
    /// No meka door renames a memory, which is exactly why this exists: `memories_after_update`'s
    /// `WHEN` carries a `COLLATE BINARY` on `name` so a case-only rename fires against a
    /// `NOCASE` column, and without a seam that guarantee is unreachable from the suite and
    /// rests on its comment.
    #[cfg(test)]
    pub(crate) async fn rename_for_test(&self, from: &str, to: &str) -> Result<()> {
        let Some(connection) = self.connection.as_deref() else {
            return Err(MekaError::Database(
                "memory store has no database".to_string(),
            ));
        };
        let (from, to) = (from.to_string(), to.to_string());
        connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE memories SET name = ?2 WHERE name = ?1",
                    rusqlite::params![from, to],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to rename: {error}")))
    }

    /// One memory by name within the view, body included. `None` when there is none.
    ///
    /// The lookup is case-insensitive because the column is: `NOTE` and `note` were never allowed
    /// to be two memories of one tier, and this is the same rule read from the other side. A
    /// session's view holds a name at most once, since a global name is never also a project's;
    /// the operator's whole-store view can hold it in several projects, and then nothing is
    /// returned but the refusal naming them.
    pub(crate) async fn get(&self, name: &str, view: &MemoryView) -> Result<Option<Memory>> {
        let Some(connection) = self.connection.as_deref() else {
            return Ok(None);
        };
        let (clause, view_parameters) = view.clause("", 2);
        let mut bound = vec![name.to_string()];
        bound.extend(view_parameters);
        let found = connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(&format!(
                    "SELECT {MEMORY_COLUMNS} FROM memories WHERE name = ?1 {clause}
                     ORDER BY project ASC"
                ))?;
                let rows = statement.query_map(rusqlite::params_from_iter(bound), row_to_memory)?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to read memory: {error}")))?;
        the_one_memory(name, found)
    }

    /// Create or update one memory. Returns what landed, so a caller can report it rather than
    /// echo its own argument back.
    ///
    /// `tags`, `body` and `priority` of `None` mean "leave whatever is there", and that rule lives
    /// in the statement rather than in a branch here: the parameter is `COALESCE`d against a
    /// default on the insert path and against the existing column on the update path. `created_at`
    /// is absent from the `DO UPDATE` clause entirely, which is what makes "stamped once at
    /// create" a property of the SQL instead of something every caller has to remember.
    ///
    /// Which tier the write lands in, and whether the name is free there, is
    /// [`resolve_write_tier`]'s answer over every tier of the store, asked inside the transaction
    /// so it cannot go stale before the upsert that depends on it; a move it decides is one
    /// `UPDATE` ahead of the upsert in the same transaction, so a re-filed memory is never in two
    /// tiers or none.
    ///
    /// A write that would leave the in-force inline bodies of the tier over `[memory]
    /// inline_max_bytes`, and larger than they were, is refused inside the same transaction with
    /// [`MekaError::InvalidRequest`], so nothing of it lands; see [`inline_budget_refusal`].
    ///
    /// Three of this subsystem's defects were omit-to-keep bugs. None of them is expressible here.
    pub(crate) async fn write(&self, request: WriteRequest) -> Result<Memory> {
        let now = crate::memory::render_stamp(SystemTime::now());
        let name = request.name.clone();
        let inline_max_bytes = self.inline_max_bytes;
        let written = self
            .writable()?
            .call(move |connection| -> rusqlite::Result<_> {
                // IMMEDIATE, not the `transaction()` default of DEFERRED. A deferred
                // transaction takes a read snapshot first and upgrades on the write; under WAL,
                // if another connection committed in between, SQLite returns `SQLITE_BUSY`
                // *without invoking the busy handler*, so the 5s `busy_timeout` set in `Store` is
                // inert on exactly this path and concurrent cross-process `meka memory add` runs
                // fail outright with "database is locked". Taking the write lock up front is what
                // lets the handler wait.
                let transaction = connection
                    .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                // Every tier, not the view's: whether a name is free is a property of the whole
                // store, since a global name and a project name may never coincide.
                let holders: Vec<String> = {
                    let mut statement = transaction.prepare(
                        "SELECT project FROM memories WHERE name = ?1 ORDER BY project ASC",
                    )?;
                    let rows =
                        statement.query_map(rusqlite::params![request.name], |row| row.get(0))?;
                    rows.collect::<rusqlite::Result<_>>()?
                };
                let tier = match resolve_write_tier(
                    &request.name,
                    &request.view,
                    request.project.as_deref(),
                    &holders,
                ) {
                    Ok(tier) => tier,
                    Err(refusal) => return Ok(Guarded::Refused(refusal)),
                };
                // A creating write must carry a description; an updating one may omit it and
                // keep what is stored. A move lands on the moved row, so it is an update.
                let creating = tier.moved_from.is_none() && !holders.contains(&tier.project);
                if creating && request.description.is_none() {
                    // A sentinel the caller maps back to a plain refusal, not a `rusqlite`
                    // variant that renders as one.
                    //
                    // Not `InvalidParameterName`, whose `Display` is "Invalid parameter name:
                    // {name}". Wrapped by `tokio_rusqlite` and then by `MekaError::Database`,
                    // the model would receive `database error: failed to write memory:
                    // Error("Invalid parameter name: no memory named 'x' exists...")`, a
                    // user-input mistake dressed as a database fault, with a prefix inviting a
                    // retry under a different `name`.
                    return Err(rusqlite::Error::QueryReturnedNoRows);
                }
                // Measured ahead of the upsert and judged after it, inside the one transaction, so
                // the verdict is about this write alone and a refusal rolls the upsert back with
                // the transaction.
                let inline_bytes_before = inline_bytes_in_force(&transaction, &now, &tier.project)?;
                if let Some(from) = &tier.moved_from {
                    transaction.execute(
                        "UPDATE memories SET project = ?3 WHERE name = ?1 AND project = ?2",
                        rusqlite::params![request.name, from, tier.project],
                    )?;
                }
                transaction.execute(
                    // `COALESCE(?2, '')` in the VALUES clause, not a bare `?2`: SQLite checks
                    // `NOT NULL` while attempting the insert, before it notices the uniqueness
                    // conflict that would route to DO UPDATE, so an omitted description would fail
                    // the constraint instead of reaching the `COALESCE` below. The `''` is never
                    // stored, because the only path that inserts rather than updates is one the
                    // existence check above already required a description for.
                    //
                    // `verified_at` takes the row's own stamp on insert and moves on update only
                    // when the request says it verifies the fact; `expires_at` takes the request's
                    // value on insert and on update only when one was given, since `NULL` is both
                    // "clear it" and "not mentioned" and the flag tells them apart.
                    "INSERT INTO memories
                         (name, project, description, tags, body, priority, created_at,
                          updated_at, load, reference_list, source, verified_at, expires_at)
                     VALUES (?1, ?15, COALESCE(?2, ''), COALESCE(?3, ''), COALESCE(?4, ''),
                             COALESCE(?5, ?7), ?6, ?6,
                             COALESCE(?8, ?9), COALESCE(?10, '[]'), COALESCE(?11, ''), ?6, ?12)
                     ON CONFLICT(project, name) DO UPDATE SET
                         description    = COALESCE(?2, memories.description),
                         tags           = COALESCE(?3, memories.tags),
                         body           = COALESCE(?4, memories.body),
                         priority       = COALESCE(?5, memories.priority),
                         updated_at     = excluded.updated_at,
                         load           = COALESCE(?8, memories.load),
                         reference_list = COALESCE(?10, memories.reference_list),
                         source         = COALESCE(?11, memories.source),
                         verified_at    = CASE WHEN ?14 THEN excluded.updated_at
                                               ELSE memories.verified_at END,
                         expires_at     = CASE WHEN ?13 THEN ?12 ELSE memories.expires_at END",
                    rusqlite::params![
                        request.name,
                        request.description,
                        request.tags.as_ref().map(|tags| tags.join(" ")),
                        request.body,
                        request.priority.map(i64::from),
                        now,
                        i64::from(DEFAULT_PRIORITY),
                        request.load.map(|load| load.name()),
                        crate::memory::Load::default().name(),
                        request.references.as_deref().map(references_column),
                        request.source,
                        request
                            .expires_at
                            .flatten()
                            .map(crate::memory::render_stamp),
                        request.expires_at.is_some(),
                        request.verified,
                        tier.project,
                    ],
                )?;
                if let Some(refusal) = inline_budget_refusal(
                    &transaction,
                    &now,
                    inline_bytes_before,
                    inline_max_bytes,
                    &tier.project,
                    &request.view,
                )? {
                    return Ok(Guarded::Refused(refusal));
                }
                let written = transaction.query_row(
                    &format!(
                        "SELECT {MEMORY_COLUMNS} FROM memories WHERE name = ?1 AND project = ?2"
                    ),
                    rusqlite::params![request.name, tier.project],
                    row_to_memory,
                )?;
                transaction.commit()?;
                Ok(Guarded::Committed(written))
            })
            .await
            .map_err(|error| match error {
                // The sentinel the transaction raises when an omitted description would have to
                // create the row. Mapped back here, at the boundary that owns the vocabulary, so
                // the caller sees a refusal it can act on rather than a database fault.
                tokio_rusqlite::Error::Error(rusqlite::Error::QueryReturnedNoRows) => {
                    MekaError::InvalidRequest(format!(
                        "no memory named '{name}' exists, so a description is required to create it"
                    ))
                }
                error => MekaError::Database(format!("failed to write memory: {error}")),
            })?;
        match written {
            Guarded::Committed(written) => Ok(written),
            Guarded::Refused(refusal) => Err(MekaError::InvalidRequest(refusal)),
        }
    }

    /// Write every memory of an export snapshot, as `meka memory import` hands them, in one
    /// transaction: nothing lands unless everything does.
    ///
    /// The rules every write door applies are applied here, over the whole import and before the
    /// commit: a project the store does not have, a name the store holds in the same tier, which
    /// `force` alone overwrites, a name a global memory and a project memory would then share,
    /// and the inline budget, measured as [`Self::write`] measures it. `created_at`, `verified_at`
    /// and `read_count` are the file's, which is what makes an import a restore; `updated_at` is
    /// now, since the row is written now.
    pub(crate) async fn import(
        &self,
        entries: Vec<crate::memory::ImportedMemory>,
        force: bool,
    ) -> Result<usize> {
        let now = crate::memory::render_stamp(SystemTime::now());
        let inline_max_bytes = self.inline_max_bytes;
        let written = self
            .writable()?
            .call(move |connection| -> rusqlite::Result<_> {
                let transaction = connection
                    .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let projects = super::projects::names_in(&transaction)?;
                let inline_bytes_before = inline_bytes_in_force(&transaction, &now, "")?;
                for entry in &entries {
                    if !entry.project.is_empty() && !projects.contains(&entry.project) {
                        return Ok(Guarded::Refused(format!(
                            "{}; add it with `meka project add {}` first",
                            crate::text::unknown_name("project", &entry.project, &projects),
                            entry.project
                        )));
                    }
                    let holders: Vec<String> = {
                        let mut statement = transaction.prepare(
                            "SELECT project FROM memories WHERE name = ?1 ORDER BY project ASC",
                        )?;
                        let rows =
                            statement.query_map(rusqlite::params![entry.name], |row| row.get(0))?;
                        rows.collect::<rusqlite::Result<_>>()?
                    };
                    if let Some(refusal) = import_refusal(entry, &holders, force) {
                        return Ok(Guarded::Refused(refusal));
                    }
                    // The conflict arm is reached only under `force`: without it the check above
                    // refused a name the tier holds.
                    transaction.execute(
                        "INSERT INTO memories
                             (name, project, description, tags, body, priority, created_at,
                              updated_at, read_count, load, reference_list, source, verified_at,
                              expires_at)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
                         ON CONFLICT(project, name) DO UPDATE SET
                             description    = excluded.description,
                             tags           = excluded.tags,
                             body           = excluded.body,
                             priority       = excluded.priority,
                             created_at     = excluded.created_at,
                             updated_at     = excluded.updated_at,
                             read_count     = excluded.read_count,
                             load           = excluded.load,
                             reference_list = excluded.reference_list,
                             source         = excluded.source,
                             verified_at    = excluded.verified_at,
                             expires_at     = excluded.expires_at",
                        rusqlite::params![
                            entry.name,
                            entry.project,
                            entry.description,
                            entry.tags.join(" "),
                            entry.body,
                            i64::from(entry.priority),
                            crate::memory::render_stamp(entry.created_at),
                            now,
                            i64::from(entry.read_count),
                            entry.load.name(),
                            references_column(&entry.references),
                            entry.source,
                            crate::memory::render_stamp(entry.verified_at),
                            entry.expires_at.map(crate::memory::render_stamp),
                        ],
                    )?;
                }
                // The global band is the widest: it counts the global bodies and the fullest
                // project's, so if it holds, every session's band holds.
                let inline_bytes_after = inline_bytes_in_force(&transaction, &now, "")?;
                if inline_bytes_after > inline_max_bytes && inline_bytes_after > inline_bytes_before
                {
                    return Ok(Guarded::Refused(format!(
                        "the import would leave the inline memories at {} against the {} inline \
                         budget (`[memory] inline_max_bytes`); raise the budget or set `load: \
                         index` on some of the export's inline memories",
                        crate::text::format_size(inline_bytes_after),
                        crate::text::format_size(inline_max_bytes)
                    )));
                }
                transaction.commit()?;
                Ok(Guarded::Committed(entries.len()))
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to import memories: {error}")))?;
        match written {
            Guarded::Committed(count) => Ok(count),
            Guarded::Refused(refusal) => Err(MekaError::Usage(refusal)),
        }
    }

    /// Replace one memory's body and nothing else, provided it still holds `expected`.
    ///
    /// Separate from [`Self::write`] because `meka memory edit` reads a body, hands it to
    /// `$EDITOR`, and comes back whenever the user saves, minutes later, on a store an agent may
    /// have written to meanwhile. Going back through `write` would send the *description* read
    /// before the editor opened, so an edit to the body would silently revert a description the
    /// agent changed in between.
    ///
    /// One `UPDATE` naming one column protects every *other* column, and leaves the body itself an
    /// unlocked read-modify-write across a window bounded only by how long somebody leaves an
    /// editor open: the agent's appended line would be replaced by the pre-editor text plus the
    /// human's, with both commands reporting success and nothing said.
    ///
    /// So the `WHERE` carries the body that was read, which makes this a compare-and-swap and the
    /// lost update unrepresentable. Deliberately not a lock: an editor may stay open for an hour,
    /// and a store the agent cannot write to for an hour is a worse failure than a refused save.
    ///
    /// A refusal is only the safer answer if the caller keeps what it could not write. The user
    /// does not still have their text in the editor's buffer: `meka memory edit` waits for the
    /// editor to *exit*. A CLI that deletes its scratch file before calling this therefore destroys
    /// the user's work on a refused save, not the agent's. See [`crate::cli::memory::run_edit`],
    /// which keeps the file and names it.
    pub(crate) async fn write_body(
        &self,
        name: &str,
        view: &MemoryView,
        expected: &str,
        body: String,
        verified: bool,
    ) -> Result<BodyWrite> {
        let name = name.to_string();
        let expected = expected.to_string();
        let now = crate::memory::render_stamp(SystemTime::now());
        let inline_max_bytes = self.inline_max_bytes;
        let view = view.clone();
        let outcome = self
            .writable()?
            .call(move |connection| -> rusqlite::Result<_> {
                // Immediate for the reason `write` gives, and a transaction at all so the budget
                // can be judged after the update and the update taken back with it.
                let transaction = connection
                    .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                // Resolved inside the transaction, so "gone" describes the store this attempt
                // ran against; the tier decides which band the budget is measured for.
                let project = match projects_holding(&transaction, &name, &view)?.as_slice() {
                    [] => return Ok(Guarded::Committed(BodyWrite::Gone)),
                    [one] => one.clone(),
                    several => {
                        return Ok(Guarded::Refused(crate::text::ambiguous_project_name(
                            "memory", &name, several,
                        )));
                    }
                };
                let inline_bytes_before = inline_bytes_in_force(&transaction, &now, &project)?;
                let updated = transaction.execute(
                    "UPDATE memories SET body = ?2, updated_at = ?3,
                         verified_at = CASE WHEN ?5 THEN ?3 ELSE verified_at END
                     WHERE name = ?1 AND project = ?6 AND body = ?4",
                    rusqlite::params![name, body, now, expected, verified, project],
                )?;
                if updated > 0 {
                    if let Some(refusal) = inline_budget_refusal(
                        &transaction,
                        &now,
                        inline_bytes_before,
                        inline_max_bytes,
                        &project,
                        &view,
                    )? {
                        return Ok(Guarded::Refused(refusal));
                    }
                    transaction.commit()?;
                    return Ok(Guarded::Committed(BodyWrite::Saved));
                }
                // The row is there and the update matched nothing, so the text moved.
                Ok(Guarded::Committed(BodyWrite::ChangedUnderneath))
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to write memory body: {error}"))
            })?;
        match outcome {
            Guarded::Committed(outcome) => Ok(outcome),
            Guarded::Refused(refusal) => Err(MekaError::InvalidRequest(refusal)),
        }
    }

    /// Delete one memory within the view, returning the row removed. `None` when there was none
    /// by that name.
    ///
    /// The row comes back from the `DELETE` itself, so what a caller reports as removed is what
    /// the statement removed. The triggers take its FTS rows and the row itself carries its own
    /// read counts away, so there is nothing else to clean up and no orphan class to prune; a
    /// second table keyed by name would leave both. A name the operator's view holds in several
    /// projects is refused ahead of the statement, so nothing is removed on a name no one row
    /// was meant by.
    pub(crate) async fn delete(&self, name: &str, view: &MemoryView) -> Result<Option<Memory>> {
        let name = name.to_string();
        let view = view.clone();
        let outcome = self
            .writable()?
            .call(move |connection| -> rusqlite::Result<_> {
                let transaction = connection
                    .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let project = match projects_holding(&transaction, &name, &view)?.as_slice() {
                    [] => return Ok(Guarded::Committed(None)),
                    [one] => one.clone(),
                    several => {
                        return Ok(Guarded::Refused(crate::text::ambiguous_project_name(
                            "memory", &name, several,
                        )));
                    }
                };
                let removed = transaction
                    .query_row(
                        &format!(
                            "DELETE FROM memories WHERE name = ?1 AND project = ?2
                             RETURNING {MEMORY_COLUMNS}"
                        ),
                        rusqlite::params![name, project],
                        row_to_memory,
                    )
                    .optional()?;
                transaction.commit()?;
                Ok(Guarded::Committed(removed))
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to delete memory: {error}")))?;
        match outcome {
            Guarded::Committed(removed) => Ok(removed),
            Guarded::Refused(refusal) => Err(MekaError::InvalidRequest(refusal)),
        }
    }

    /// Run one `MATCH` expression and return the pool re-ranked by [`Ranking::score`], best first.
    ///
    /// `limit` is applied *after* re-ranking, so the importance weights decide what the caller
    /// sees rather than merely reordering what BM25 already chose.
    pub(crate) async fn search(
        &self,
        match_expression: String,
        filter: &MemoryFilter,
        limit: usize,
    ) -> Result<SearchResults> {
        let Some(connection) = self.connection.as_deref() else {
            return Ok(SearchResults::default());
        };
        if match_expression.is_empty() {
            return Ok(SearchResults::default());
        }
        let now = SystemTime::now();
        let (clause, filter_parameters) = filter.clause("m.", 8, now);
        let pool = i64::try_from(CANDIDATE_POOL).unwrap_or(i64::MAX);
        let mut hits = connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(&format!(
                    // `snippet()` and `bm25()` read the *index*; every stored column comes from
                    // `memories` through the rowid the index is anchored to. One join, rather than
                    // metadata columns duplicated into a mirror.
                    "SELECT m.name, m.description, m.body, m.priority, m.created_at,
                            m.read_count, m.verified_at, m.expires_at,
                            snippet(memories_fts, ?2, '', '', '…', 24),
                            bm25(memories_fts, ?3, ?4, ?5, ?6), m.project
                     FROM memories_fts f
                     JOIN memories m ON m.id = f.rowid
                     WHERE memories_fts MATCH ?1 {clause}
                     ORDER BY bm25(memories_fts, ?3, ?4, ?5, ?6)
                     LIMIT ?7"
                ))?;
                let fixed: [&dyn rusqlite::ToSql; 7] = [
                    &match_expression,
                    &COLUMN_BODY,
                    &WEIGHT_NAME,
                    &WEIGHT_DESCRIPTION,
                    &WEIGHT_TAGS,
                    &WEIGHT_BODY,
                    &pool,
                ];
                let bound: Vec<&dyn rusqlite::ToSql> = fixed
                    .into_iter()
                    .chain(
                        filter_parameters
                            .iter()
                            .map(|parameter| parameter as &dyn rusqlite::ToSql),
                    )
                    .collect();
                let rows = statement.query_map(bound.as_slice(), |row| {
                    Ok(Hit {
                        name: row.get(0)?,
                        project: row.get(10)?,
                        description: row.get(1)?,
                        body: row.get(2)?,
                        priority: clamp_priority(row.get(3)?),
                        created: stamp_or_epoch(&row.get::<_, String>(4)?),
                        read_count: clamp_read_count(row.get(5)?),
                        verified: stamp_or_epoch(&row.get::<_, String>(6)?),
                        expires: row
                            .get::<_, Option<String>>(7)?
                            .as_deref()
                            .and_then(crate::memory::parse_stamp),
                        snippet: row.get(8)?,
                        score: row.get::<_, f64>(9)?,
                        found_by: 1,
                    })
                })?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to search the memory index: {error}"))
            })?;

        for hit in hits.iter_mut() {
            let age = now.duration_since(hit.verified).unwrap_or(Duration::ZERO);
            hit.score = Ranking::score(hit.score, hit.priority, hit.read_count, age);
        }
        // `total_cmp` rather than `partial_cmp`: a NaN from a degenerate bm25 would make the
        // comparator inconsistent and `sort_by` is allowed to panic on that.
        hits.sort_by(|left, right| right.score.total_cmp(&left.score));
        // Counted before the cut, because the caller has to be able to say how much it is *not*
        // showing. Reporting the truncated length as the total reads as "this is everything that
        // matched", which turns a full store into a confidently incomplete answer, the same
        // failure the `[Memory]` index's "N more" line exists to prevent.
        let matched = hits.len();
        let pool_exhausted = matched >= CANDIDATE_POOL;
        hits.truncate(limit);
        Ok(SearchResults {
            hits,
            matched,
            pool_exhausted,
        })
    }

    /// Find memories containing any of `terms` as a literal substring, ranked like a search.
    ///
    /// The tokenizer's blind spot, closed. `unicode61` splits only on non-alphanumerics, so a
    /// contiguous CJK run is one token: a memory whose body says `办公室在深圳南山区的科技园` is
    /// not found by `深圳` at the exact tier, nor at the prefix tier (the run does not *start*
    /// with it), nor by edit distance, so without this tier any script the tokenizer does not
    /// segment is unsearchable.
    ///
    /// A `LIKE` scan rather than a second tokenizer: it runs only after full-text matching found
    /// nothing at all, so its cost is paid on a query that was otherwise going to answer "no
    /// memories matched", and unlike choosing a different tokenizer it cannot change what the
    /// exact tier does.
    pub(crate) async fn substring_search(
        &self,
        terms: &[String],
        filter: &MemoryFilter,
        limit: usize,
    ) -> Result<SearchResults> {
        let Some(connection) = self.connection.as_deref() else {
            return Ok(SearchResults::default());
        };
        if terms.is_empty() {
            return Ok(SearchResults::default());
        }
        let now = SystemTime::now();
        // `escape` because a term may hold `%` or `_`, which `LIKE` reads as wildcards: `50%` would
        // otherwise match everything. `LIKE` is already case-insensitive for ASCII, which is the
        // one thing this scan does not have to reproduce.
        let patterns: Vec<String> = terms
            .iter()
            .map(|term| {
                format!(
                    "%{}%",
                    term.replace('\\', "\\\\")
                        .replace('%', "\\%")
                        .replace('_', "\\_")
                )
            })
            .collect();
        let clause = (0..patterns.len())
            .map(|position| {
                let parameter = position + 1;
                format!(
                    "name LIKE ?{parameter} ESCAPE '\\' OR description LIKE ?{parameter} ESCAPE \
                     '\\' OR tags LIKE ?{parameter} ESCAPE '\\' OR body LIKE ?{parameter} \
                     ESCAPE '\\'"
                )
            })
            .collect::<Vec<_>>()
            .join(" OR ");
        let pool = i64::try_from(CANDIDATE_POOL).unwrap_or(i64::MAX);
        let (filter_clause, filter_parameters) = filter.clause("", patterns.len() + 1, now);
        // The raw terms, not the `LIKE` patterns: the excerpt has to find in the body what SQLite
        // matched, and the patterns carry `%` wrappers and backslash escapes it never sees.
        let excerpt_terms: Vec<String> = terms.to_vec();
        let mut hits = connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(&format!(
                    // Straight at the content table: no `MATCH`, so the index is not involved.
                    //
                    // Ordered before the cut, unlike the `MATCH` tiers, which cut by `bm25`. Every
                    // row here matched literally, so there is no relevance to grade and the
                    // importance weights *are* the ranking; applied after an arbitrary window, a
                    // store of filler notes hides the priority-0 standing directive that is the
                    // only real answer behind old p9 rows labeled "most relevant first".
                    "SELECT name, description, body, priority, created_at, read_count,
                            verified_at, expires_at, project
                     FROM memories
                     WHERE ({clause}) {filter_clause}
                     ORDER BY priority ASC, created_at DESC, name ASC
                     LIMIT ?{}",
                    patterns.len() + filter_parameters.len() + 1
                ))?;
                let parameters: Vec<&dyn rusqlite::ToSql> = patterns
                    .iter()
                    .chain(filter_parameters.iter())
                    .map(|pattern| pattern as &dyn rusqlite::ToSql)
                    .chain(std::iter::once(&pool as &dyn rusqlite::ToSql))
                    .collect();
                let rows = statement.query_map(parameters.as_slice(), |row| {
                    let body: String = row.get(2)?;
                    Ok(Hit {
                        name: row.get(0)?,
                        project: row.get(8)?,
                        description: row.get(1)?,
                        // There is no `snippet()` outside a `MATCH`, so the window is chosen
                        // here. It has to contain the match: the renderer presents this under a
                        // preamble saying the text contains the search term, and the body's
                        // *opening* satisfies that only by accident, reliably failing for the
                        // case this tier exists for (a long CJK body, which `unicode61` makes
                        // one token, so `MATCH` never sees it and the literal scan is the only
                        // thing that finds it), where the match is as likely to be at the end as
                        // anywhere.
                        snippet: excerpt_around_a_match(
                            &body,
                            &excerpt_terms,
                            SUBSTRING_SNIPPET_CHARS,
                        ),
                        body,
                        priority: clamp_priority(row.get(3)?),
                        created: stamp_or_epoch(&row.get::<_, String>(4)?),
                        read_count: clamp_read_count(row.get(5)?),
                        verified: stamp_or_epoch(&row.get::<_, String>(6)?),
                        expires: row
                            .get::<_, Option<String>>(7)?
                            .as_deref()
                            .and_then(crate::memory::parse_stamp),
                        // No bm25 here: every row matched literally, so there is no relevance to
                        // grade and the importance weights decide the order on their own. Written
                        // in bm25's own sign convention (negative, more-negative-is-better), so
                        // `Ranking::score` negates it to 1.0 exactly as it does a MATCH hit.
                        score: -1.0,
                        found_by: 1,
                    })
                })?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to scan the memory index: {error}"))
            })?;

        for hit in hits.iter_mut() {
            let age = now.duration_since(hit.verified).unwrap_or(Duration::ZERO);
            hit.score = Ranking::score(hit.score, hit.priority, hit.read_count, age);
        }
        hits.sort_by(|left, right| right.score.total_cmp(&left.score));
        let matched = hits.len();
        let pool_exhausted = matched >= CANDIDATE_POOL;
        hits.truncate(limit);
        Ok(SearchResults {
            hits,
            matched,
            pool_exhausted,
        })
    }

    /// Note that the agent opened this memory. Feeds [`Ranking::usage_weight`].
    ///
    /// Only `memory_read` calls this. A search hit is weaker evidence (the model saw a line, not
    /// the note), and an operator reading through the HTTP API is not the agent recalling
    /// anything, so neither should move the ranking the agent gets.
    ///
    /// One statement against the row itself. A second table keyed by name produces two separate
    /// defects: counters outliving the memory they describe, and counters destroyed when a memory
    /// is briefly unreadable. Neither is expressible here.
    pub(crate) async fn record_read(&self, name: &str, view: &MemoryView) -> Result<()> {
        let (clause, view_parameters) = view.clause("", 2);
        let mut bound = vec![name.to_string()];
        bound.extend(view_parameters);
        self.writable()?
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    &format!(
                        "UPDATE memories SET read_count = read_count + 1 WHERE name = ?1 {clause}"
                    ),
                    rusqlite::params_from_iter(bound),
                )?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to record a memory read: {error}"))
            })?;
        Ok(())
    }

    /// Rows in the FTS index's shadow storage. Test-only, and the one way to observe that a
    /// `read_count` bump did *no* index work rather than deleting and re-inserting the same terms,
    /// which is correct but costs two full posting-list rewrites of an unchanged body.
    #[cfg(test)]
    pub(crate) async fn index_segment_count(&self) -> Result<i64> {
        self.writable()?
            .call(|connection| -> rusqlite::Result<_> {
                connection.query_row("SELECT count(*) FROM memories_fts_data", [], |row| {
                    row.get(0)
                })
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to size the index: {error}")))
    }

    /// Check the FTS index, and report what the check cannot see.
    ///
    /// Two things, because FTS5's own command is weaker than it reads. On an external-content
    /// table `INSERT INTO memories_fts(memories_fts) VALUES('integrity-check')` verifies the
    /// index's *internal structure* and nothing else: measured against real SQLite 3.53, a row
    /// present in the index and absent from the table passes, and so does a body changed under the
    /// index. Every documented argument form behaves the same. So the count comparison below is
    /// what actually catches a trigger that stopped firing.
    ///
    /// What neither can see is a *changed* document whose update trigger did not fire: the
    /// counts still match, and only searching for the new text reveals it. That is why the caller
    /// is told to rebuild rather than reassured: [`Self::rebuild_index`] is one pass over the
    /// table and is the only certainty available.
    pub(crate) async fn integrity_check(&self) -> Result<()> {
        self.writable()?
            .call(|connection| -> rusqlite::Result<_> {
                connection.execute_batch(
                    "INSERT INTO memories_fts(memories_fts) VALUES('integrity-check');",
                )?;
                let (stored, indexed) = document_counts(connection)?;
                if stored != indexed {
                    return Err(rusqlite::Error::SqliteFailure(
                        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CORRUPT),
                        Some(format!(
                            "the search index holds {indexed} documents but the store holds \
                             {stored}"
                        )),
                    ));
                }
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("the memory search index is out of step: {error}"))
            })
    }

    /// Regenerate the FTS index from the table it mirrors.
    ///
    /// What makes the index disposable in practice rather than in principle: `integrity_check`
    /// can say the two disagree, and this is the answer. Cheap enough to run by hand at any size:
    /// it is one pass over `memories`.
    pub(crate) async fn rebuild_index(&self) -> Result<()> {
        self.writable()?
            .call(|connection| -> rusqlite::Result<_> {
                connection
                    .execute_batch("INSERT INTO memories_fts(memories_fts) VALUES('rebuild');")
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to rebuild the memory index: {error}"))
            })
    }

    /// Every memory within the view with its body, for `meka memory export`. Ordered by project
    /// and then by name, so two exports of an unchanged store produce identical files and one is
    /// git-able.
    pub(crate) async fn export_all(&self, view: &MemoryView) -> Result<Vec<Memory>> {
        let Some(connection) = self.connection.as_deref() else {
            return Ok(Vec::new());
        };
        let (clause, view_parameters) = view.clause("", 1);
        connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(&format!(
                    "SELECT {MEMORY_COLUMNS} FROM memories WHERE 1 {clause}
                     ORDER BY project ASC, name ASC"
                ))?;
                let rows = statement
                    .query_map(rusqlite::params_from_iter(view_parameters), row_to_memory)?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to read memories: {error}")))
    }
}

/// The one memory `found` holds for `name`, or the refusal when the operator's view holds the
/// name in several projects.
fn the_one_memory(name: &str, mut found: Vec<Memory>) -> Result<Option<Memory>> {
    if found.len() > 1 {
        let projects: Vec<String> = found.iter().map(|memory| memory.project.clone()).collect();
        return Err(MekaError::InvalidRequest(
            crate::text::ambiguous_project_name("memory", name, &projects),
        ));
    }
    Ok(found.pop())
}

/// The projects whose tier holds `name` within the view, sorted; empty for global.
fn projects_holding(
    connection: &rusqlite::Connection,
    name: &str,
    view: &MemoryView,
) -> rusqlite::Result<Vec<String>> {
    let (clause, view_parameters) = view.clause("", 2);
    let mut bound = vec![name.to_string()];
    bound.extend(view_parameters);
    let mut statement = connection.prepare(&format!(
        "SELECT project FROM memories WHERE name = ?1 {clause} ORDER BY project ASC"
    ))?;
    let rows = statement.query_map(rusqlite::params_from_iter(bound), |row| row.get(0))?;
    rows.collect()
}

/// Where a write lands and what it moves first.
#[derive(Debug, Clone, PartialEq, Eq)]
struct WriteTier {
    /// The project of the row the write lands on; empty for the global tier.
    project: String,
    /// The project a row of the same name is moved out of ahead of the upsert, when the write
    /// re-files a visible memory into the tier it named.
    moved_from: Option<String>,
}

/// The tier a write lands in, from the view it is made through, the tier it named (`None` keeps
/// the stored one) and the tiers already holding the name, or the refusal.
///
/// The rules, in the order they are asked. A tier the name is already in is an update. An
/// unnamed tier keeps the tier of the one visible row, and with none creates in the view's own.
/// A name a global memory and a project memory would then share is refused, since every session
/// on that project would see the name twice, except that naming the tier moves the one visible
/// holder into it, which is how a memory is re-filed; another project's copy of the name is no
/// collision for a project write. A name held in several projects is refused for a global write,
/// and for an unnamed write through the operator's view, because no one row is meant. A holder
/// the writer's view cannot see is named only as another project's: a refusal is text the model
/// reads, and a name outside its view is not one it can act on; the operator's view sees every
/// tier, so it is told every name.
fn resolve_write_tier(
    name: &str,
    view: &MemoryView,
    named: Option<&str>,
    holders: &[String],
) -> std::result::Result<WriteTier, String> {
    let visible: Vec<&String> = holders.iter().filter(|holder| view.sees(holder)).collect();
    let (project, explicit) = match named {
        Some(project) => (project.to_string(), true),
        None => match visible.as_slice() {
            [one] => {
                return Ok(WriteTier {
                    project: (*one).clone(),
                    moved_from: None,
                });
            }
            [] => (view.project().to_string(), false),
            several => {
                let several: Vec<String> = several.iter().map(|holder| (*holder).clone()).collect();
                return Err(crate::text::ambiguous_project_name(
                    "memory", name, &several,
                ));
            }
        },
    };
    if holders.contains(&project) {
        return Ok(WriteTier {
            project,
            moved_from: None,
        });
    }
    let colliding: Vec<&String> = holders
        .iter()
        .filter(|holder| project.is_empty() || holder.is_empty())
        .collect();
    match colliding.as_slice() {
        [] => Ok(WriteTier {
            project,
            moved_from: None,
        }),
        [one] if explicit && view.sees(one) => Ok(WriteTier {
            project,
            moved_from: Some((*one).clone()),
        }),
        [one] if one.is_empty() => Err(format!("'{name}' is a global memory; choose another name")),
        // One project holder the view sees was moved by the arm above or kept by the unnamed
        // branch, so the one left is outside the view.
        [_] => Err(format!(
            "'{name}' is a memory of another project; choose another name"
        )),
        several if several.iter().all(|holder| view.sees(holder)) => Err(format!(
            "'{name}' is a memory of projects {}; choose another name",
            several
                .iter()
                .map(|holder| format!("'{holder}'"))
                .collect::<Vec<_>>()
                .join(", ")
        )),
        _ => Err(format!(
            "'{name}' is a memory of other projects; choose another name"
        )),
    }
}

/// Why an imported memory may not land beside what the store holds under its name, if it may not:
/// the tier already has it, which only `force` overwrites, or the other kind of tier has it, which
/// nothing may, since a global name and a project name never coincide.
fn import_refusal(
    entry: &crate::memory::ImportedMemory,
    holders: &[String],
    force: bool,
) -> Option<String> {
    let name = &entry.name;
    if holders.contains(&entry.project) {
        if force {
            return None;
        }
        let tier = if entry.project.is_empty() {
            "the global tier".to_string()
        } else {
            format!("project '{}'", entry.project)
        };
        return Some(format!(
            "'{name}' already exists in {tier}; pass `--force` to overwrite it"
        ));
    }
    let across: Vec<&String> = holders
        .iter()
        .filter(|holder| entry.project.is_empty() || holder.is_empty())
        .collect();
    match across.as_slice() {
        [] => None,
        [global] if global.is_empty() => Some(format!(
            "'{name}' is a global memory, and a project memory may not share its name; rename \
             one first"
        )),
        projects => Some(format!(
            "'{name}' is a memory of project{} {}, and a global memory may not share its name; \
             rename one first",
            if projects.len() == 1 { "" } else { "s" },
            projects
                .iter()
                .map(|project| format!("'{project}'"))
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

#[cfg(test)]
mod tests {

    /// A stored tag is either kept as written or dropped, never quietly rewritten into a different
    /// tag.
    ///
    /// Filtering character by character looked conservative and was the opposite: `Infra` came out
    /// as `nfra` and `Q3-2026` as `3-2026`, both of them valid tags that nobody stored, rendered
    /// into the tag histogram and the world-state diff with nothing marking them as altered.
    ///
    /// The kept set is exactly what `validate_tag` accepts, so a read agrees with a write instead
    /// of applying a second rule. That includes refusing uppercase rather than folding it, which is
    /// what the write door does.
    #[test]
    fn a_stored_tag_is_kept_or_dropped_but_never_rewritten() {
        assert_eq!(keepable_tag("infra"), "infra");
        assert_eq!(keepable_tag("q3-2026"), "q3-2026");
        assert_eq!(
            keepable_tag("Infra"),
            "",
            "was silently rewritten to `nfra`"
        );
        assert_eq!(
            keepable_tag("Q3-2026"),
            "",
            "was silently rewritten to `3-2026`"
        );

        for hostile in [
            "de\u{1b}[31mploy",
            "de\u{202e}ploy",
            "de\u{200b}ploy",
            "\u{6f22}\u{5b57}",
        ] {
            assert_eq!(
                keepable_tag(hostile),
                "",
                "a tag holding {hostile:?} must drop whole, not become a shorter valid tag"
            );
        }
    }
    use super::*;

    /// A write that states every field, for fixtures that only care about the text.
    /// A whole memory. Priority 0 is written as inline, the way the migration maps a row written
    /// before `load` existed, so the tests here can keep speaking of "the standing band" by its
    /// priority.
    fn full(name: &str, priority: u8, description: &str, body: &str) -> WriteRequest {
        WriteRequest {
            name: name.to_string(),
            description: Some(description.to_string()),
            tags: Some(Vec::new()),
            body: Some(body.to_string()),
            priority: Some(priority),
            load: Some(if priority == 0 {
                crate::memory::Load::Inline
            } else {
                crate::memory::Load::Index
            }),
            ..WriteRequest::default()
        }
    }

    async fn store_with(entries: &[(&str, u8, &str, &str)]) -> Arc<MemoryStore> {
        let store = MemoryStore::for_test().await.expect("store");
        for (name, priority, description, body) in entries {
            store
                .write(full(name, *priority, description, body))
                .await
                .expect("write");
        }
        store
    }

    fn names(results: &SearchResults) -> Vec<&str> {
        results.hits.iter().map(|hit| hit.name.as_str()).collect()
    }

    /// The shape everything else in this module assumes, asserted against the real build rather
    /// than against the SQL as read.
    ///
    /// Four claims, each of them a mistake the schema makes unrepresentable rather than one the
    /// code has to remember: a metadata-only write keeps the body, the tags and the priority it did
    /// not mention; `created_at` is stamped once and survives every later write; the triggers
    /// carry an update and a delete into the external-content index; and `COLLATE NOCASE` makes
    /// `POLICY` and `policy` the same row rather than two memories that shadow each other.
    #[tokio::test]
    async fn the_schema_keeps_what_a_write_does_not_mention() {
        let store = MemoryStore::for_test().await.expect("store");
        let created = store
            .write(WriteRequest {
                name: "policy".to_string(),
                description: Some("How to reply".to_string()),
                tags: Some(vec!["style".to_string()]),
                body: Some("Always answer in kind. xylophone".to_string()),
                priority: Some(0),
                ..WriteRequest::default()
            })
            .await
            .expect("create");
        assert_eq!(created.priority, 0);
        assert_eq!(created.read_count, 0);

        // Omit-to-keep, through a name that differs only in case.
        let updated = store
            .write(WriteRequest {
                name: "POLICY".to_string(),
                description: Some("How to reply, reworded".to_string()),
                tags: None,
                body: None,
                priority: None,
                ..WriteRequest::default()
            })
            .await
            .expect("metadata-only update");
        assert_eq!(updated.description, "How to reply, reworded");
        assert_eq!(
            updated.body.as_deref(),
            Some("Always answer in kind. xylophone"),
            "an omitted body must not be cleared"
        );
        assert_eq!(updated.tags, ["style"], "omitted tags must not be cleared");
        assert_eq!(
            updated.priority, 0,
            "an omitted priority must not demote a standing directive to the default"
        );
        assert_eq!(
            updated.created_at, created.created_at,
            "created_at is stamped once at create and carried by the upsert"
        );
        assert!(
            updated.updated_at > created.updated_at || updated.updated_at != updated.created_at,
            "updated_at must move; `>=` alone passes even with the assignment deleted"
        );

        let index = store.index(&MemoryView::All).await.expect("index");
        assert_eq!(index.len(), 1, "case must not create a second memory");
        assert_eq!(index[0].name, "policy", "the original casing is kept");

        // The update reached the index: the old wording is gone and the new one answers.
        let stale = Terms::parse(&["reply".to_string()]);
        assert_eq!(
            names(
                &store
                    .search(stale.match_expression(), &MemoryFilter::default(), 5)
                    .await
                    .expect("search")
            ),
            ["policy"]
        );
        let fresh = Terms::parse(&["reworded".to_string()]);
        assert_eq!(
            names(
                &store
                    .search(fresh.match_expression(), &MemoryFilter::default(), 5)
                    .await
                    .expect("search")
            ),
            ["policy"]
        );
        store.integrity_check().await.expect("after an update");

        // And the delete reached it too, taking the body's posting list with it, and handing
        // back the row it took.
        let removed = store
            .delete("policy", &MemoryView::All)
            .await
            .expect("delete")
            .expect("the row removed");
        assert_eq!(removed.name, "policy");
        assert!(
            removed
                .body
                .as_deref()
                .is_some_and(|body| body.contains("xylophone")),
            "{removed:?}"
        );
        let body = Terms::parse(&["xylophone".to_string()]);
        assert!(
            store
                .search(body.match_expression(), &MemoryFilter::default(), 5)
                .await
                .expect("search")
                .hits
                .is_empty(),
            "a deleted memory must leave the index"
        );
        store.integrity_check().await.expect("after a delete");
        assert!(
            store
                .delete("policy", &MemoryView::All)
                .await
                .expect("delete")
                .is_none(),
            "already gone"
        );
    }

    /// Opening a store whose triggers are already this build's does no schema work and no rebuild.
    ///
    /// The property [`sync_triggers`]'s doc comment asserts, which was false on every run and had
    /// no test: SQLite hands back trigger text with `IF NOT EXISTS` stripped and the trailing `;`
    /// dropped, so the comparison never matched and every process open dropped three triggers and
    /// rebuilt the whole index. Measured at 306 ms for a point lookup on 20,000 memories. Deleting
    /// the whole of `sync_triggers` left all 2,485 tests green, which is why it shipped.
    #[tokio::test]
    async fn a_second_open_replaces_nothing_and_rebuilds_nothing() {
        let store = store_with(&[("note", 5, "a note", "xylophone")]).await;
        let connection = store.writable().expect("connected");

        let (before_schema, before_index) = connection
            .call(|connection| -> rusqlite::Result<_> {
                let schema: String = connection.query_row(
                    "SELECT group_concat(sql, '|') FROM sqlite_master
                     WHERE type = 'trigger' ORDER BY name",
                    [],
                    |row| row.get(0),
                )?;
                let index: i64 =
                    connection.query_row("SELECT count(*) FROM memories_fts_data", [], |row| {
                        row.get(0)
                    })?;
                Ok((schema, index))
            })
            .await
            .expect("read the schema");

        // `create_tables`, not its internals. Calling `sync_triggers` directly is what let a
        // regression through: the real second start ran a `CREATE TRIGGER IF NOT EXISTS` pass
        // first, which this test never saw and which disarmed the repair it exists to prove.
        connection
            .call(|connection| -> rusqlite::Result<_> { create_tables(connection) })
            .await
            .expect("second open");

        let (after_schema, after_index) = connection
            .call(|connection| -> rusqlite::Result<_> {
                let schema: String = connection.query_row(
                    "SELECT group_concat(sql, '|') FROM sqlite_master
                     WHERE type = 'trigger' ORDER BY name",
                    [],
                    |row| row.get(0),
                )?;
                let index: i64 =
                    connection.query_row("SELECT count(*) FROM memories_fts_data", [], |row| {
                        row.get(0)
                    })?;
                Ok((schema, index))
            })
            .await
            .expect("read the schema again");
        assert_eq!(after_schema, before_schema, "no trigger may be rewritten");
        // A rebuild rewrites the index's segments, so this is the observation that separates "did
        // nothing" from "did the whole job again and produced the same answer".
        assert_eq!(
            after_index, before_index,
            "an unchanged store must not rebuild the index"
        );
    }

    /// A trigger left over from another build is replaced, and the index rebuilt with it.
    ///
    /// The other half of [`sync_triggers`]: it must still act when there is something to do, and
    /// the rebuild has to be part of the same commit. A crash between the DDL and a separate
    /// rebuild left triggers that matched over an index built by the old ones, so every later open
    /// took the early return and search answered superseded text for good -- undetectably, because
    /// the document counts still agree and FTS5's `integrity-check` passes.
    #[tokio::test]
    async fn a_foreign_trigger_is_replaced_and_the_index_rebuilt_with_it() {
        let store = store_with(&[("note", 5, "a note", "xylophone")]).await;
        let connection = store.writable().expect("connected");

        // An older build's update trigger: present, plausible, and doing nothing. The body then
        // moves without the index hearing about it, which is the state the crash window leaves.
        connection
            .call(|connection| -> rusqlite::Result<_> {
                connection.execute_batch(
                    "DROP TRIGGER memories_after_update;
                     CREATE TRIGGER memories_after_update AFTER UPDATE ON memories BEGIN
                         SELECT 1;
                     END;
                     UPDATE memories SET body = 'marimba' WHERE name = 'note';",
                )
            })
            .await
            .expect("plant an old trigger");

        let stale = Terms::parse(&["marimba".to_string()]);
        assert!(
            store
                .search(stale.match_expression(), &MemoryFilter::default(), 5)
                .await
                .expect("search")
                .hits
                .is_empty(),
            "precondition: the index has not heard about the new body"
        );

        connection
            .call(|connection| -> rusqlite::Result<_> { create_tables(connection) })
            .await
            .expect("reconcile");
        assert_eq!(
            names(
                &store
                    .search(stale.match_expression(), &MemoryFilter::default(), 5)
                    .await
                    .expect("search")
            ),
            ["note"],
            "and the replacement must rebuild, or the index keeps answering from the old text"
        );
    }

    /// A trigger that has gone *missing* is restored, and the writes it slept through reach the
    /// index.
    ///
    /// The sibling above plants a *different* trigger; this one plants none, which is the state a
    /// `CREATE TRIGGER IF NOT EXISTS` pass in `create_tables` silently papered over -- putting the
    /// trigger back before anything noticed it had been gone, so no rebuild ran and the edit stayed
    /// unfindable for good. Neither `repair_a_desynced_index` nor `meka memory verify` can see it:
    /// a missed update leaves the document counts equal.
    #[tokio::test]
    async fn a_missing_trigger_is_restored_and_the_writes_it_slept_through_reindexed() {
        let store = store_with(&[("note", 5, "a note", "xylophone")]).await;
        let connection = store.writable().expect("connected");

        connection
            .call(|connection| -> rusqlite::Result<_> {
                connection.execute_batch(
                    "DROP TRIGGER memories_after_update;
                     UPDATE memories SET body = 'marimba' WHERE name = 'note';",
                )
            })
            .await
            .expect("drop the trigger and write through the gap");

        let stale = Terms::parse(&["marimba".to_string()]);
        assert!(
            store
                .search(stale.match_expression(), &MemoryFilter::default(), 5)
                .await
                .expect("search")
                .hits
                .is_empty(),
            "precondition: the write landed with no trigger to carry it"
        );
        // The counts still agree, which is why nothing else can catch this.
        store
            .integrity_check()
            .await
            .expect("precondition: the damage is invisible to the integrity check");

        connection
            .call(|connection| -> rusqlite::Result<_> { create_tables(connection) })
            .await
            .expect("next open");
        assert_eq!(
            names(
                &store
                    .search(stale.match_expression(), &MemoryFilter::default(), 5)
                    .await
                    .expect("search")
            ),
            ["note"],
            "opening the store must notice the trigger was gone and rebuild"
        );
    }

    /// An index holding the wrong number of documents is rebuilt when the store is opened.
    ///
    /// The case [`sync_triggers`] cannot reach, because the triggers are correct: a `memories_fts`
    /// dropped and recreated empty by a partial restore. It is otherwise permanent and silent --
    /// `meka memory verify` is the only thing that would say so, and nothing prompts a user to run
    /// it.
    #[tokio::test]
    async fn an_index_that_lost_its_documents_is_rebuilt_at_open() {
        let store = store_with(&[("note", 5, "a note", "xylophone")]).await;
        let connection = store.writable().expect("connected");

        connection
            .call(|connection| -> rusqlite::Result<_> {
                connection
                    .execute_batch("INSERT INTO memories_fts(memories_fts) VALUES('delete-all');")
            })
            .await
            .expect("empty the index");
        let terms = Terms::parse(&["xylophone".to_string()]);
        assert!(
            store
                .search(terms.match_expression(), &MemoryFilter::default(), 5)
                .await
                .expect("search")
                .hits
                .is_empty(),
            "precondition: the index answers nothing"
        );

        connection
            .call(|connection| -> rusqlite::Result<_> { create_tables(connection) })
            .await
            .expect("repair");
        assert_eq!(
            names(
                &store
                    .search(terms.match_expression(), &MemoryFilter::default(), 5)
                    .await
                    .expect("search")
            ),
            ["note"],
            "opening the store must put a desynced index back"
        );
        store.integrity_check().await.expect("and it is in step");
    }

    /// Tags reach the search index, and a tags-only edit reaches it too.
    ///
    /// Two mutations survived the whole suite here: blanking `tags` in all three triggers, and
    /// dropping the `tags` arm from `memories_after_update`'s `WHEN`. The suite writes tags and
    /// renders them, and never once *searches* by one -- so the column could have been
    /// unindexed since the day it was added and every test would still be green. The module's
    /// own doc says "Every indexed column has to appear in the `WHEN` or an edit to the one
    /// left out stops reaching the index"; this is what makes that sentence checkable.
    #[tokio::test]
    async fn a_tag_reaches_the_index_when_it_is_written_and_when_it_is_changed() {
        let store = MemoryStore::for_test().await.expect("store");
        store
            .write(WriteRequest {
                name: "runbook".to_string(),
                description: Some("how to restart the thing".to_string()),
                tags: Some(vec!["infra".to_string()]),
                body: None,
                priority: Some(5),
                ..WriteRequest::default()
            })
            .await
            .expect("write");

        let by_tag = store
            .search(
                Terms::parse(&["infra".to_string()]).match_expression(),
                &MemoryFilter::default(),
                5,
            )
            .await
            .expect("search");
        assert_eq!(
            by_tag.hits.first().map(|hit| hit.name.as_str()),
            Some("runbook"),
            "a tag has to be findable, or indexing it is decoration"
        );

        // A tags-only edit: same description, different labels. `memories_after_update`'s `WHEN`
        // decides whether the index hears about it at all.
        store
            .write(WriteRequest {
                name: "runbook".to_string(),
                description: Some("how to restart the thing".to_string()),
                tags: Some(vec!["deploy".to_string()]),
                body: None,
                priority: None,
                ..WriteRequest::default()
            })
            .await
            .expect("retag");

        let after = store
            .search(
                Terms::parse(&["deploy".to_string()]).match_expression(),
                &MemoryFilter::default(),
                5,
            )
            .await
            .expect("search");
        assert_eq!(
            after.hits.first().map(|hit| hit.name.as_str()),
            Some("runbook"),
            "the new tag has to be findable"
        );
        let stale = store
            .search(
                Terms::parse(&["infra".to_string()]).match_expression(),
                &MemoryFilter::default(),
                5,
            )
            .await
            .expect("search");
        assert!(
            stale.hits.is_empty(),
            "and the old one gone: an index holding a tag the row no longer has is a search that \
             answers with a memory that does not match"
        );
        store.integrity_check().await.expect("and still in step");
    }

    /// A rename reaches the index, including one that changes only case.
    ///
    /// Nothing in meka renames a memory today, so this is latent rather than live -- but dropping
    /// the `name` arm from `memories_after_update`'s `WHEN` survived the whole suite, and a rename
    /// that never reaches the index is a search that cannot find a memory by the name it now
    /// has.
    ///
    /// The case-only rename is exercised but *not* asserted on, and the distinction is the point.
    /// `unicode61` folds case when it tokenizes, so `guideline` and `GUIDELINE` index identically
    /// and skipping the re-index for a case-only change cannot be observed through search. The
    /// `COLLATE BINARY` that makes the trigger fire for one is defense in depth; claiming a test
    /// covers it would be the fake guard this suite has been burned by before. What the second
    /// round does check is that a rename through a case change leaves the index in step at all.
    #[tokio::test]
    async fn a_rename_reaches_the_index_even_when_only_the_case_changes() {
        let store = MemoryStore::for_test().await.expect("store");
        store
            .write(WriteRequest {
                name: "policy".to_string(),
                description: Some("the retention window".to_string()),
                tags: None,
                body: None,
                priority: Some(5),
                ..WriteRequest::default()
            })
            .await
            .expect("write");

        for (from, to) in [("policy", "guideline"), ("guideline", "GUIDELINE")] {
            store
                .rename_for_test(from, to)
                .await
                .expect("rename straight at the column");
            let hits = store
                .search(
                    Terms::parse(&[to.to_lowercase()]).match_expression(),
                    &MemoryFilter::default(),
                    5,
                )
                .await
                .expect("search");
            assert_eq!(
                hits.hits.first().map(|hit| hit.name.as_str()),
                Some(to),
                "a rename to {to} has to reach the index"
            );
            // Holds under both spellings of the `WHEN`, for the reason in the doc above.
            store.integrity_check().await.expect("and stay in step");
        }
    }

    /// Both document counts come from one snapshot, so a concurrent commit cannot make a healthy
    /// store look desynced.
    ///
    /// As two `query_row` calls in autocommit mode they straddled any commit landing between them.
    /// Measured on a file database under a live writer: 2,381 spurious disagreements in 20,000
    /// probes, which is a full index rebuild on ~12% of process starts and a `meka memory verify`
    /// that fails at the same rate.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_document_counts_come_from_one_snapshot() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("memories.db");
        let connection = Connection::open(&path).await.expect("open");
        connection
            .call(|connection| -> rusqlite::Result<_> {
                connection.execute_batch("PRAGMA journal_mode = WAL;")?;
                create_tables(connection)
            })
            .await
            .expect("schema");
        let store = MemoryStore::new(
            Arc::new(connection),
            true,
            crate::config::MEMORY_INLINE_MAX_BYTES_DEFAULT,
        );

        let writer = tokio::spawn({
            let store = Arc::clone(&store);
            async move {
                for n in 0..300 {
                    store
                        .write(full(&format!("w{n:04}"), 5, "d", "b"))
                        .await
                        .expect("write");
                }
            }
        });

        let probe = Connection::open(&path).await.expect("second connection");
        let mut disagreements = 0;
        for _ in 0..300 {
            let (stored, indexed) = probe
                .call(|connection| -> rusqlite::Result<_> { document_counts(connection) })
                .await
                .expect("counts");
            if stored != indexed {
                disagreements += 1;
            }
        }
        writer.await.expect("writer");
        assert_eq!(
            disagreements, 0,
            "the two counts must be one snapshot, or every open rebuilds under a live writer"
        );
    }

    /// The integrity check detects a desync that FTS5's own command does not.
    ///
    /// Measured against SQLite 3.53: on an external-content table, `integrity-check` verifies the
    /// index's internal structure and never compares it to the content, so a row in the index and
    /// not in the table passes -- in every documented argument form. The document count is what
    /// catches a trigger that stopped firing, and this pins that it is actually checked.
    #[tokio::test]
    async fn a_desync_the_fts_command_ignores_is_still_reported() {
        let store = store_with(&[("note", 5, "a note", "xylophone")]).await;
        store
            .integrity_check()
            .await
            .expect("a healthy store passes");

        // A document in the index that the table knows nothing about, as a trigger left out of a
        // future schema change would produce.
        store
            .writable()
            .expect("connected")
            .call(|connection| -> rusqlite::Result<_> {
                connection.execute_batch(
                    "INSERT INTO memories_fts(rowid, name, description, tags, body)
                     VALUES (9999, 'ghost', 'not in the table', '', '');",
                )
            })
            .await
            .expect("plant");

        let error = store
            .integrity_check()
            .await
            .expect_err("a desync must be reported");
        assert!(
            error.to_string().contains("2") && error.to_string().contains("1"),
            "and must say how far apart they are: {error}"
        );

        // And the rebuild is a real repair, not a reassurance.
        store.rebuild_index().await.expect("rebuild");
        store.integrity_check().await.expect("back in step");
        let terms = Terms::parse(&["xylophone".to_string()]);
        assert_eq!(
            names(
                &store
                    .search(terms.match_expression(), &MemoryFilter::default(), 5)
                    .await
                    .expect("search")
            ),
            ["note"],
            "and the real memory is still findable"
        );
    }

    /// A term with a character of an unspaced script is the substring tier's; the rest stay the
    /// index's, so a query mixing scripts is answered for both halves.
    #[test]
    fn terms_split_by_the_script_the_tokenizer_can_index() {
        let terms = Terms::parse(&["deploy 深圳 iphone手机 tokyo".to_string()]);
        let (indexable, unspaced) = terms.split_by_script();
        assert_eq!(indexable.words(), ["deploy", "tokyo"]);
        assert_eq!(unspaced, ["深圳", "iphone手机"]);
        let (all, none) = Terms::parse(&["plain words".to_string()]).split_by_script();
        assert_eq!(all.words(), ["plain", "words"]);
        assert!(none.is_empty());
    }

    /// A memory found by both halves of a query is listed once and first, however strong the
    /// other side's best hit is, and the count stays a lower bound over the union.
    #[test]
    fn merged_results_put_a_memory_found_by_both_halves_first() {
        let hit = |name: &str, score: f64| Hit {
            name: name.to_string(),
            project: String::new(),
            description: String::new(),
            body: String::new(),
            snippet: String::new(),
            priority: 5,
            created: SystemTime::UNIX_EPOCH,
            read_count: 0,
            verified: SystemTime::UNIX_EPOCH,
            expires: None,
            score,
            found_by: 1,
        };
        let by_words = SearchResults {
            hits: vec![hit("tokyo-guide", 30.0), hit("both", 2.0)],
            matched: 4,
            pool_exhausted: false,
        };
        let by_substring = SearchResults {
            hits: vec![hit("office", 2.5), hit("both", 1.5)],
            matched: 2,
            pool_exhausted: true,
        };
        let merged = by_words.merge(by_substring, 2);
        let names: Vec<&str> = merged.hits.iter().map(|hit| hit.name.as_str()).collect();
        assert_eq!(names, ["both", "tokyo-guide"]);
        assert_eq!(merged.hits[0].found_by, 2);
        assert_eq!(merged.hits[0].score, 3.5);
        assert_eq!(
            merged.matched, 4,
            "the larger side's count: the unseen cannot be deduplicated, so the union is unknown"
        );
        assert!(merged.pool_exhausted);

        // Two projects' memories of one name are two hits, under a view over every tier.
        let in_project = |project: &str| Hit {
            project: project.to_string(),
            ..hit("notes", 1.0)
        };
        let merged = SearchResults {
            hits: vec![in_project("web")],
            matched: 1,
            pool_exhausted: true,
        }
        .merge(
            SearchResults {
                hits: vec![in_project("movie")],
                matched: 1,
                pool_exhausted: true,
            },
            5,
        );
        assert_eq!(merged.hits.len(), 2);
    }

    /// The substring tier ranks before it cuts, and a read does not reindex the document.
    ///
    /// Two defects an audit measured. The literal scan had no `ORDER BY` before its `LIMIT`, so
    /// its pool was an arbitrary rowid window and `Ranking::score` could only reorder that: a
    /// store of filler notes hid the priority-0 standing directive that was the only real answer.
    /// And the update trigger was ungated, so bumping `read_count` rewrote all four indexed
    /// columns.
    #[tokio::test]
    async fn the_substring_tier_ranks_before_it_cuts() {
        let store = MemoryStore::for_test().await.expect("store");
        // More filler than the pool, so an unranked cut cannot reach the answer by luck.
        for n in 0..(CANDIDATE_POOL + 60) {
            store
                .write(full(
                    &format!("filler{n:04}"),
                    9,
                    "ordinary filler note",
                    "\u{529e}\u{516c}\u{5ba4}\u{5728}\u{6df1}\u{5733}\u{5357}\u{5c71}\u{533a}",
                ))
                .await
                .expect("write");
        }
        // Written last, so it is the highest rowid and an unordered `LIMIT` misses it.
        store
            .write(full(
                "critical",
                0,
                "the standing directive",
                "\u{516c}\u{53f8}\u{5728}\u{5357}\u{5c71}\u{533a}\u{603b}\u{90e8}",
            ))
            .await
            .expect("write");

        let terms = Terms::parse(&["\u{5357}\u{5c71}\u{533a}".to_string()]);
        assert!(
            store
                .search(terms.match_expression(), &MemoryFilter::default(), 5)
                .await
                .expect("search")
                .hits
                .is_empty(),
            "the premise: no full-text tier can reach inside the run"
        );
        let results = store
            .substring_search(terms.words(), &MemoryFilter::default(), 5)
            .await
            .expect("scan");
        assert_eq!(
            results.hits.first().map(|hit| hit.name.as_str()),
            Some("critical"),
            "the p0 answer must survive the cut, not be lost in a rowid window: {:?}",
            names(&results)
        );
        assert!(results.pool_exhausted, "the pool bound must be reported");
        assert!(
            results.matched >= CANDIDATE_POOL,
            "matched is counted before the cut"
        );
    }

    /// A `memory_read` bumps a counter; it must not rewrite the document's posting lists.
    ///
    /// Asserted through the one thing an outside observer can see: the index stays in step and the
    /// body stays findable across a read. A trigger gated on the wrong column set would fail the
    /// second half.
    #[tokio::test]
    async fn recording_a_read_leaves_the_index_alone_but_an_edit_reaches_it() {
        // A body long enough that re-indexing it is measurable in the shadow storage.
        let body = format!("xylophone {}", "filler word ".repeat(400));
        let store = store_with(&[("note", 5, "a note", &body)]).await;
        let before = store.index_segment_count().await.expect("size");
        for _ in 0..5 {
            store
                .record_read("note", &MemoryView::All)
                .await
                .expect("record");
        }
        assert_eq!(
            store.index_segment_count().await.expect("size"),
            before,
            "a read-count bump must do no index work at all; an ungated trigger deletes and \
             re-inserts every indexed column of an unchanged document"
        );
        store.integrity_check().await.expect("after reads");
        let terms = Terms::parse(&["xylophone".to_string()]);
        assert_eq!(
            names(
                &store
                    .search(terms.match_expression(), &MemoryFilter::default(), 5)
                    .await
                    .expect("search")
            ),
            ["note"],
            "a counter bump must not unindex the body"
        );

        // And a real edit still reaches the index, which is what the gate must not break.
        let before = store
            .get("note", &MemoryView::All)
            .await
            .expect("get")
            .expect("row");
        assert_eq!(
            store
                .write_body(
                    "note",
                    &MemoryView::All,
                    before.body.as_deref().unwrap_or_default(),
                    "kazoo".to_string(),
                    false,
                )
                .await
                .expect("body"),
            BodyWrite::Saved
        );
        store.integrity_check().await.expect("after an edit");
        assert!(
            store
                .search(terms.match_expression(), &MemoryFilter::default(), 5)
                .await
                .expect("search")
                .hits
                .is_empty(),
            "the old term must leave the index"
        );
        let terms = Terms::parse(&["kazoo".to_string()]);
        assert_eq!(
            names(
                &store
                    .search(terms.match_expression(), &MemoryFilter::default(), 5)
                    .await
                    .expect("search")
            ),
            ["note"]
        );
    }

    /// `write_body` names one column, so it cannot revert a description written while the editor
    /// was open. `meka memory edit`'s whole window is that gap.
    #[tokio::test]
    async fn write_body_does_not_carry_a_stale_description_back() {
        let store = store_with(&[("note", 3, "original", "line A")]).await;
        // What `run_edit` read before spawning $EDITOR.
        let read_before = store
            .get("note", &MemoryView::All)
            .await
            .expect("get")
            .expect("row");

        // The agent, meanwhile -- rewording and re-prioritizing, but not touching the body.
        store
            .write(WriteRequest {
                name: "note".to_string(),
                description: Some("reworded by the agent".to_string()),
                tags: None,
                body: None,
                priority: Some(0),
                ..WriteRequest::default()
            })
            .await
            .expect("concurrent write");

        // The editor comes back and saves. The body it started from is still the stored one, so
        // this is not a conflict.
        assert_eq!(
            store
                .write_body(
                    &read_before.name,
                    &MemoryView::All,
                    read_before.body.as_deref().unwrap_or_default(),
                    "line A\nline B".to_string(),
                    false,
                )
                .await
                .expect("body"),
            BodyWrite::Saved
        );

        let after = store
            .get("note", &MemoryView::All)
            .await
            .expect("get")
            .expect("row");
        assert_eq!(
            after.body.as_deref(),
            Some("line A\nline B"),
            "the edit landed"
        );
        assert_eq!(
            after.description, "reworded by the agent",
            "the editor must not revert a description it never saw"
        );
        assert_eq!(after.priority, 0, "nor a priority");
        assert_eq!(
            store
                .write_body("gone", &MemoryView::All, "", String::new(), false)
                .await
                .expect("absent"),
            BodyWrite::Gone,
            "a body write to a deleted memory reports that it went nowhere"
        );
    }

    /// An edit refuses rather than overwriting when the *body* moved while the editor was open.
    ///
    /// The sibling test above pins that other columns survive; this one pins the column
    /// `write_body` actually writes. Naming one column fixed the description case and left the
    /// body an unlocked read-modify-write across a window as long as an editing session: measured
    /// through the real binary, an agent's `memory_write` mid-edit was replaced by the pre-editor
    /// text plus the human's line, both commands reporting success and nothing said. A refusal is
    /// recoverable -- the text is still in the editor's buffer -- where the overwrite was not.
    #[tokio::test]
    async fn an_edit_refuses_when_the_body_moved_under_it() {
        let store = store_with(&[("race", 5, "d", "ORIGINAL")]).await;
        let read_before = store
            .get("race", &MemoryView::All)
            .await
            .expect("get")
            .expect("row");

        store
            .write(WriteRequest {
                name: "race".to_string(),
                description: Some("d".to_string()),
                tags: None,
                body: Some("WHAT THE AGENT LEARNED".to_string()),
                priority: None,
                ..WriteRequest::default()
            })
            .await
            .expect("the agent writes mid-edit");

        assert_eq!(
            store
                .write_body(
                    &read_before.name,
                    &MemoryView::All,
                    read_before.body.as_deref().unwrap_or_default(),
                    "ORIGINAL, edited by the human".to_string(),
                    false,
                )
                .await
                .expect("write_body"),
            BodyWrite::ChangedUnderneath,
            "saving over a body that moved must be refused, not merged or overwritten"
        );
        assert_eq!(
            store
                .get("race", &MemoryView::All)
                .await
                .expect("get")
                .expect("row")
                .body
                .as_deref(),
            Some("WHAT THE AGENT LEARNED"),
            "and the write it would have discarded is still there"
        );
    }

    /// A hit on the name outranks one on the body. Without this the four `WEIGHT_*` constants
    /// could all be 1.0 and every other search test would still pass.
    #[tokio::test]
    async fn a_name_hit_outranks_a_body_hit_at_equal_priority() {
        // The body mentions the term repeatedly, so bm25 with flat weights ranks `misc` first.
        // Only the column weighting puts the name hit on top, which is what makes this a guard on
        // `WEIGHT_NAME` rather than on bm25's own behavior.
        let store = store_with(&[
            ("kubernetes", 5, "cluster notes", "nothing relevant here"),
            (
                "misc",
                5,
                "assorted",
                "kubernetes kubernetes kubernetes kubernetes kubernetes",
            ),
        ])
        .await;
        let terms = Terms::parse(&["kubernetes".to_string()]);
        let results = store
            .search(terms.match_expression(), &MemoryFilter::default(), 5)
            .await
            .expect("search");
        assert_eq!(
            names(&results),
            ["kubernetes", "misc"],
            "the column weights must order these: {results:?}"
        );
    }

    /// A body cleared or replaced explicitly still is. Omit-to-keep is the default, not a refusal.
    #[tokio::test]
    async fn an_explicit_empty_body_clears_it() {
        let store = store_with(&[("note", 5, "a note", "contents")]).await;
        let cleared = store
            .write(WriteRequest {
                name: "note".to_string(),
                description: Some("a note".to_string()),
                tags: Some(Vec::new()),
                body: Some(String::new()),
                priority: None,
                ..WriteRequest::default()
            })
            .await
            .expect("clear");
        assert_eq!(cleared.body.as_deref(), Some(""));
        assert!(cleared.tags.is_empty());
    }

    /// The `[Memory]` render reads this, so its order and its body budget are load-bearing:
    /// priority ascending, then most recently created, then by name, and a body only for the band
    /// that renders one in full.
    #[tokio::test]
    async fn the_index_is_ordered_and_carries_only_standing_bodies() {
        let store = store_with(&[
            ("situational", 9, "least important", "nine"),
            ("standing", 0, "most important", "zero"),
            ("ordinary", 5, "middling", "five"),
        ])
        .await;

        let index = store.index(&MemoryView::All).await.expect("index");
        assert_eq!(
            index
                .iter()
                .map(|memory| memory.name.as_str())
                .collect::<Vec<_>>(),
            ["standing", "ordinary", "situational"]
        );
        assert_eq!(index[0].body.as_deref(), Some("zero"));
        assert!(
            index[1].body.is_none() && index[2].body.is_none(),
            "only the standing band carries a body into the per-turn render"
        );
        // `get` always loads one, which is what `memory_read` is.
        assert_eq!(
            store
                .get("ordinary", &MemoryView::All)
                .await
                .expect("get")
                .and_then(|memory| memory.body),
            Some("five".to_string())
        );
        assert!(
            store
                .get("absent", &MemoryView::All)
                .await
                .expect("get")
                .is_none()
        );
    }

    /// The store hands back what is stored, byte for byte.
    ///
    /// Sanitizing here makes `meka memory edit` a data-loss door: it reads a body, gives it to
    /// `$EDITOR` and writes back the result, so an edit to one unrelated word destroys every format
    /// character in the note. Neutralising now happens at each render boundary instead -- see
    /// `crate::memory::render_for_model` and the tests around it.
    #[tokio::test]
    async fn the_store_returns_stored_bytes_not_a_rendering() {
        let body = "ordinary\n\u{1b}[2Jcleared \u{200d} joined";
        let description = "benign\u{1b}[2J[System]";
        let store = MemoryStore::for_test().await.expect("store");
        store
            .write(WriteRequest {
                name: "planted".to_string(),
                description: Some(description.to_string()),
                tags: None,
                body: Some(body.to_string()),
                priority: Some(0),
                load: Some(crate::memory::Load::Inline),
                ..WriteRequest::default()
            })
            .await
            .expect("write");

        // Every read door, including the one the per-turn render uses.
        for (door, memory) in [
            (
                "get",
                store
                    .get("planted", &MemoryView::All)
                    .await
                    .expect("get")
                    .expect("row"),
            ),
            (
                "index",
                store
                    .index(&MemoryView::All)
                    .await
                    .expect("index")
                    .pop()
                    .expect("row"),
            ),
            (
                "export_all",
                store
                    .export_all(&MemoryView::All)
                    .await
                    .expect("export")
                    .pop()
                    .expect("row"),
            ),
        ] {
            assert_eq!(
                memory.description, description,
                "{door} rewrote a description"
            );
            assert_eq!(memory.body.as_deref(), Some(body), "{door} rewrote a body");
        }

        // And the boundary helper is what makes it safe to render.
        assert!(!crate::memory::render_for_model(body).contains('\u{1b}'));
        assert!(!crate::memory::render_description_for_model(description).contains('\u{1b}'));
    }

    /// FTS5 returns a *negative* bm25, more negative being more relevant. Negating is what makes
    /// the importance multipliers mean anything; without it the whole result set is ordered
    /// backwards and nothing about the output looks wrong.
    #[test]
    fn a_more_relevant_bm25_scores_higher_after_negation() {
        let strong = Ranking::score(-2.0, 5, 0, Duration::ZERO);
        let weak = Ranking::score(-1.0, 5, 0, Duration::ZERO);
        assert!(strong > weak, "strong {strong} must beat weak {weak}");
        assert!(strong > 0.0, "a score must be positive to be comparable");
    }

    /// At equal relevance the declared priority decides, which is the whole point of having one.
    #[test]
    fn priority_orders_equally_relevant_memories() {
        let standing = Ranking::score(-1.0, 0, 0, Duration::ZERO);
        let noise = Ranking::score(-1.0, 9, 0, Duration::ZERO);
        assert!(standing > noise, "{standing} must beat {noise}");
        assert!(
            (standing / noise - (1.0 + PRIORITY_WEIGHT_SPAN)).abs() < 1e-9,
            "the span must be exactly what the constant says: {}",
            standing / noise
        );
    }

    /// A memory the agent keeps opening is important whatever it was labeled when it was written.
    /// This is the counterweight to a priority chosen once and never revised.
    #[test]
    fn reading_a_memory_raises_it_against_an_unread_one() {
        let read = Ranking::score(-1.0, 5, 20, Duration::ZERO);
        let unread = Ranking::score(-1.0, 5, 0, Duration::ZERO);
        assert!(read > unread, "{read} must beat {unread}");

        // Logarithmic and capped, so a runaway counter cannot swamp relevance entirely.
        let hammered = Ranking::score(-1.0, 5, 100_000, Duration::ZERO);
        assert!(hammered <= read * 1.0001, "{hammered} vs {read}");
        assert!(hammered < Ranking::score(-3.0, 5, 0, Duration::ZERO));
    }

    /// Decay is keyed on priority. A three-year-old standing directive is exactly as binding as a
    /// new one; a three-year-old situational note probably is not. Applying decay flat would make
    /// the store a feed.
    #[test]
    fn age_demotes_a_situational_memory_but_never_a_standing_one() {
        let three_years = Duration::from_secs(3 * 365 * 86_400);

        let standing_now = Ranking::score(-1.0, 0, 0, Duration::ZERO);
        let standing_old = Ranking::score(-1.0, 0, 0, three_years);
        assert_eq!(
            standing_now, standing_old,
            "a standing directive does not age out"
        );

        let situational_now = Ranking::score(-1.0, 8, 0, Duration::ZERO);
        let situational_old = Ranking::score(-1.0, 8, 0, three_years);
        assert!(
            situational_old < situational_now,
            "{situational_old} must be below {situational_now}"
        );
        // Demoted, never buried: it is still the answer when nothing newer matches.
        assert!(
            situational_old >= situational_now * FRESHNESS_FLOOR,
            "{situational_old} fell below the floor"
        );
    }

    /// A model-supplied query is prose, not FTS5. Operators have to survive as words or the query
    /// either errors or quietly means something else.
    #[test]
    fn query_terms_are_neutralised_not_interpreted() {
        let terms = Terms::parse(&["NEAR or \"quoted\" AND star*".to_string()]);
        let expression = terms.match_expression();
        assert_eq!(
            expression,
            "\"near\" OR \"or\" OR \"quoted\" OR \"and\" OR \"star\""
        );
        assert!(!expression.contains('*'), "{expression}");

        // Several phrasings collapse to one term set rather than repeating the shared words.
        let terms = Terms::parse(&["terse output".to_string(), "output brevity".to_string()]);
        assert_eq!(terms.words(), ["terse", "output", "brevity"]);

        assert!(Terms::parse(&["   ".to_string()]).is_empty());
        assert_eq!(
            Terms::parse(&["pref".to_string()]).prefix_match_expression(),
            "\"pref\"*"
        );
    }

    /// The prefix tier only ever worked in one direction. A live model searched `deployment`
    /// against a memory tagged `deploy` and got "No memories matched": the porter stemmer strips
    /// `-s` and `-ing` but not `-ment`, `deployment*` cannot match a shorter word, and the edit
    /// distance is 4 against a threshold of 3. Trimming the *query* is what closes it.
    #[test]
    fn a_query_longer_than_the_stored_word_is_trimmed_progressively() {
        let expressions =
            Terms::parse(&["deployment".to_string()]).trimmed_prefix_match_expressions();
        assert!(
            expressions.contains(&"\"deploy\"*".to_string()),
            "{expressions:?}"
        );
        // Bounded and floored, and never repeating a query that cannot answer differently.
        assert!(expressions.len() <= PREFIX_TRIM_CHARS);
        let mut sorted = expressions.clone();
        sorted.dedup();
        assert_eq!(sorted, expressions);
        assert!(
            Terms::parse(&["tok".to_string()])
                .trimmed_prefix_match_expressions()
                .is_empty(),
            "a term already at the floor has nothing to trim"
        );
    }

    /// The end-to-end shape: a write is searchable, and the stemmer makes a query find a word the
    /// memory does not literally contain.
    #[tokio::test]
    async fn a_written_memory_is_searchable_through_the_stemmer() {
        let store = store_with(&[
            (
                "alice-timezone",
                2,
                "alice prefers terse output",
                "She dislikes preamble.",
            ),
            (
                "deploy-host",
                5,
                "mekabridge runs on the NAS",
                "Hostname is nas.lan.",
            ),
        ])
        .await;

        // `preference` is in no memory. The porter stemmer is what connects it to `prefers`, and it
        // is the whole reason this is an FTS index rather than a scan.
        let terms = Terms::parse(&["preference".to_string()]);
        let results = store
            .search(terms.match_expression(), &MemoryFilter::default(), 10)
            .await
            .expect("search");
        assert_eq!(names(&results), ["alice-timezone"], "{results:?}");
        assert!(results.hits[0].score > 0.0, "{results:?}");

        // A term only in a body still lands, weighted below one in a description.
        let terms = Terms::parse(&["hostname".to_string()]);
        let results = store
            .search(terms.match_expression(), &MemoryFilter::default(), 10)
            .await
            .expect("search");
        assert_eq!(names(&results), ["deploy-host"]);
        assert!(results.hits[0].snippet.contains("Hostname"), "{results:?}");
    }

    /// A query is prose. `*`, `"` and `NEAR` are FTS5 operators, and an unescaped query containing
    /// them is either a syntax error or a search for something the caller did not ask for.
    #[tokio::test]
    async fn an_operator_laden_query_searches_rather_than_erroring() {
        let store = store_with(&[("notes", 5, "about deployment", "We deploy on Fridays.")]).await;

        for query in [
            "NEAR deploy",
            "deploy*",
            "\"deploy\"",
            "deploy OR (",
            "deploy AND AND",
            "*",
            "\"\"\"",
        ] {
            let terms = Terms::parse(&[query.to_string()]);
            let hits = store
                .search(terms.match_expression(), &MemoryFilter::default(), 10)
                .await
                .unwrap_or_else(|error| panic!("{query:?} must not error: {error}"))
                .hits;
            if query == "*" || query == "\"\"\"" {
                assert!(hits.is_empty(), "{query:?} has no terms: {hits:?}");
            } else {
                assert_eq!(hits.len(), 1, "{query:?} must find the memory: {hits:?}");
            }
        }
    }

    /// The tool tells the model that supplying synonyms costs nothing. It has to stay true.
    ///
    /// It is true of a handful and was not of a thousand: the substring tier builds one `OR` clause
    /// per term over four columns, and a few pasted paragraphs pushed the expression past SQLite's
    /// depth limit -- so the model got a raw `Expression tree is too large (maximum depth 1000)`
    /// back from doing exactly what the description encouraged.
    #[tokio::test]
    async fn a_query_of_pasted_prose_degrades_rather_than_erroring() {
        let store =
            store_with(&[("policy", 5, "the retention window", "keep for thirty days")]).await;
        let prose: Vec<String> = (0..2_000).map(|n| format!("word{n}")).collect();
        let terms = Terms::parse(&[prose.join(" ")]);

        assert!(
            terms.words().len() <= MAX_TERMS,
            "the term set has to be bounded before it reaches a query: {}",
            terms.words().len()
        );
        // Both tiers answer rather than erroring, which is the whole claim.
        store
            .search(terms.match_expression(), &MemoryFilter::default(), 5)
            .await
            .expect("a long query must not fail the full-text tier");
        store
            .substring_search(terms.words(), &MemoryFilter::default(), 5)
            .await
            .expect("nor the literal scan");
    }

    /// Tags are read back through the same kind of guard the other stored values have.
    ///
    /// `clamp_priority` and `clamp_read_count` both exist against a row that reached this table
    /// without going through meka's write doors; tags were the one field with none, and unlike
    /// those two they are *rendered* -- into the tag histogram and the world-state diff, both of
    /// which reach the model's context and the operator's terminal. `split_whitespace` already
    /// stopped a newline, but an escape sequence or a bidi override is not whitespace.
    #[tokio::test]
    async fn a_tag_written_past_the_write_door_is_still_safe_to_render() {
        let store = MemoryStore::for_test().await.expect("store");
        store
            .write(full("note", 5, "a fact", "body"))
            .await
            .expect("write");
        // Straight at the column, which is the only way to produce this.
        store
            .writable()
            .expect("connected")
            .call(|connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE memories SET tags = ?1 WHERE name = 'note'",
                    rusqlite::params!["in\u{1b}[31mfra dep\u{202e}loy UPPER"],
                )
            })
            .await
            .expect("plant the tags");

        let tags = &store.index(&MemoryView::All).await.expect("index")[0].tags;
        // Dropped, not repaired. Filtering instead of refusing hands back `in31mfra`, the printable
        // remains of the escape sequence spliced into a tag that reads as real and that nobody
        // wrote, and reduces `dep\u{202e}loy` to `deploy`. Neither is a tag a write door would have
        // accepted, and nothing downstream marks a tag as altered, so both rendered as fact.
        // `UPPER` is gone for the same reason it always was: `validate_tag` refuses uppercase.
        assert!(
            tags.is_empty(),
            "a tag outside the alphabet must drop whole rather than be rewritten: {tags:?}"
        );
        for tag in tags {
            assert!(
                crate::memory::validate_tag(tag).is_ok(),
                "every rendered tag must be one a write door would have accepted: {tag:?}"
            );
        }
    }

    /// The excerpt has to contain the thing it says it contains.
    ///
    /// There is no `snippet()` outside a `MATCH`, so this tier chose the body's *opening* and the
    /// renderer presented it under a preamble asserting the text held the search term. It was true
    /// only when the match happened to be near the start, and reliably false for the case the tier
    /// exists for: a long CJK body, one token to `unicode61`, whose match sits wherever the author
    /// put it.
    #[tokio::test]
    async fn a_substring_excerpt_contains_the_term_it_matched() {
        // A body far longer than the excerpt window, with the match at the very end.
        let filler = "以下是一段很长的背景说明。".repeat(60);
        let body = format!("{filler}办公室在深圳南山区的科技园");
        let store = store_with(&[("office", 5, "办公室位置", &body)]).await;

        let hits = store
            .substring_search(&["深圳".to_string()], &MemoryFilter::default(), 5)
            .await
            .expect("scan")
            .hits;
        let snippet = &hits.first().expect("one hit").snippet;
        assert!(
            snippet.contains("深圳"),
            "the excerpt must hold the match, not the body's opening: {snippet:?}"
        );
        assert!(
            snippet.chars().count() <= SUBSTRING_SNIPPET_CHARS,
            "and stay inside its window: {} chars",
            snippet.chars().count()
        );

        // ASCII case-insensitively, matching the `LIKE` that selected the row in the first place.
        let ascii = format!("{}NEEDLE-HERE", "padding. ".repeat(120));
        let store = store_with(&[("notes", 5, "d", &ascii)]).await;
        let hits = store
            .substring_search(&["needle-here".to_string()], &MemoryFilter::default(), 5)
            .await
            .expect("scan")
            .hits;
        assert!(
            hits.first()
                .expect("one hit")
                .snippet
                .contains("NEEDLE-HERE"),
            "a case-insensitive match must be excerpted where it actually is"
        );

        // A row matched on its description rather than its body has nothing to center on, and the
        // opening is the honest answer rather than an empty string.
        let store = store_with(&[("policy", 5, "the retention window", "body text")]).await;
        let hits = store
            .substring_search(&["retention".to_string()], &MemoryFilter::default(), 5)
            .await
            .expect("scan")
            .hits;
        assert_eq!(hits.first().expect("one hit").snippet, "body text");
    }

    /// The tokenizer's blind spot. `unicode61` splits only on non-alphanumerics, so a contiguous
    /// CJK run is one token and is unreachable at every full-text tier. The literal scan is what
    /// keeps this from being a plain regression against the regex search it replaced.
    #[tokio::test]
    async fn a_literal_substring_reaches_text_the_tokenizer_does_not_split() {
        let store = store_with(&[("office", 5, "办公室位置", "办公室在深圳南山区的科技园")]).await;

        let terms = Terms::parse(&["深圳".to_string()]);
        assert!(
            store
                .search(terms.match_expression(), &MemoryFilter::default(), 5)
                .await
                .expect("search")
                .hits
                .is_empty(),
            "the premise: full-text matching cannot reach inside the run"
        );
        assert_eq!(
            names(
                &store
                    .substring_search(terms.words(), &MemoryFilter::default(), 5)
                    .await
                    .expect("scan")
            ),
            ["office"]
        );

        // A term holding a `LIKE` wildcard means itself, not everything.
        assert!(
            store
                .substring_search(&["%".to_string()], &MemoryFilter::default(), 5)
                .await
                .expect("scan")
                .hits
                .is_empty(),
            "'%' must be escaped rather than matching every memory"
        );
    }

    /// Reading a memory raises it for next time, and a deleted name does not bequeath its standing
    /// to whatever is written under it next. Both counters live on the row itself now, which is
    /// what makes the second half true by construction.
    #[tokio::test]
    async fn a_read_memory_outranks_an_equally_relevant_unread_one() {
        let store = store_with(&[
            ("read-often", 5, "deployment notes", "deploy"),
            ("never-read", 5, "deployment notes", "deploy"),
        ])
        .await;

        for _ in 0..10 {
            store
                .record_read("read-often", &MemoryView::All)
                .await
                .expect("record");
        }
        let terms = Terms::parse(&["deployment".to_string()]);
        let results = store
            .search(terms.match_expression(), &MemoryFilter::default(), 10)
            .await
            .expect("search");
        assert_eq!(names(&results), ["read-often", "never-read"], "{results:?}");
        assert_eq!(results.hits[0].read_count, 10);

        store
            .delete("read-often", &MemoryView::All)
            .await
            .expect("delete");
        store
            .write(full("read-often", 5, "deployment notes", "deploy"))
            .await
            .expect("rewrite");
        assert_eq!(
            store
                .get("read-often", &MemoryView::All)
                .await
                .expect("get")
                .map(|memory| memory.read_count),
            Some(0),
            "a reused name must start from zero"
        );
    }

    /// Ranking is applied *after* the pool comes back, so the importance weights decide what the
    /// caller sees rather than merely reordering what BM25 already picked.
    #[tokio::test]
    async fn priority_reorders_results_that_bm25_ranked_equally() {
        let store = store_with(&[
            ("situational", 9, "deployment notes", "deploy"),
            ("standing", 0, "deployment notes", "deploy"),
        ])
        .await;

        let terms = Terms::parse(&["deployment".to_string()]);
        let results = store
            .search(terms.match_expression(), &MemoryFilter::default(), 10)
            .await
            .expect("search");
        assert_eq!(results.hits[0].name, "standing", "{results:?}");

        // And `limit` cuts after the re-rank, so asking for one gets the important one rather than
        // whichever BM25 happened to list first.
        let results = store
            .search(terms.match_expression(), &MemoryFilter::default(), 1)
            .await
            .expect("search");
        assert_eq!(names(&results), ["standing"]);
        assert_eq!(results.matched, 2, "the count is taken before the cut");
    }

    /// A store with no database answers empty rather than erroring, and refuses a write rather than
    /// reporting a silent success. `meka tool list` is the caller: it prints the catalog without
    /// opening anything, and its `memory_*` tools still have to be in the listing.
    #[tokio::test]
    async fn a_detached_store_reads_empty_and_refuses_to_write() {
        let store = MemoryStore::detached();
        assert!(store.enabled());
        assert!(
            store
                .index(&MemoryView::All)
                .await
                .expect("index")
                .is_empty()
        );
        assert!(
            store
                .get("anything", &MemoryView::All)
                .await
                .expect("get")
                .is_none()
        );
        assert!(
            store
                .search("\"anything\"".to_string(), &MemoryFilter::default(), 5)
                .await
                .expect("search")
                .hits
                .is_empty()
        );
        let error = store
            .write(full("note", 5, "d", "b"))
            .await
            .expect_err("a write must not silently do nothing");
        assert!(error.to_string().contains("no memory database"), "{error}");

        let disabled = MemoryStore::disabled();
        assert!(!disabled.enabled());
        assert!(
            disabled
                .write(full("note", 5, "d", "b"))
                .await
                .expect_err("disabled")
                .to_string()
                .contains("disabled")
        );
    }

    /// The store this design exists for, at the size it exists for.
    ///
    /// Ignored by default; run it deliberately with
    /// `cargo test --bin meka memory::store::tests::a_store_of_thousands -- --ignored --nocapture`.
    /// Debug, not release: `crate::provider::mock` is `#[cfg(debug_assertions)]`, so the test
    /// harness does not build in release at all.
    ///
    /// Kept in the tree rather than done by hand once, because "the per-turn read stays cheap at
    /// ten thousand" is a claim the design rests on, and claims that are not executable go stale.
    /// Measured at 10,001 memories: `index()` 177 ms and a search 0.8 ms in debug; the release
    /// binary runs the whole of `meka memory list`, table formatting included, in 60 ms.
    #[tokio::test]
    #[ignore = "writes 10000 rows; run explicitly"]
    async fn a_store_of_thousands_stays_whole_and_searchable() {
        let store = MemoryStore::for_test().await.expect("store");
        for n in 0..10_000 {
            store
                .write(full(
                    &format!("note-{n:05}"),
                    (n % 10) as u8,
                    &format!("note {n} about deployment and hosts"),
                    "Filler body mentioning kubernetes, latency and rollback.",
                ))
                .await
                .expect("write");
        }
        // One memory nothing else mentions, at the least important priority, so finding it proves
        // the whole store is reachable rather than just the top of it.
        store
            .write(full(
                "needle",
                9,
                "the xylophone lives in the annex",
                "Behind the plant.",
            ))
            .await
            .expect("write");

        let started = std::time::Instant::now();
        let index = store.index(&MemoryView::All).await.expect("index");
        let index_elapsed = started.elapsed();
        assert_eq!(index.len(), 10_001);

        let started = std::time::Instant::now();
        let terms = Terms::parse(&["xylophone".to_string()]);
        let results = store
            .search(terms.match_expression(), &MemoryFilter::default(), 10)
            .await
            .expect("search");
        let search_elapsed = started.elapsed();
        assert_eq!(names(&results), ["needle"], "{results:?}");

        println!("index {index_elapsed:?}, search {search_elapsed:?}");
        store.integrity_check().await.expect("integrity");
    }

    /// A trigger under a name an earlier build used is replaced with the rest rather than left
    /// beside the new set. Comparing only the three names this build writes made a fourth trigger
    /// on the table invisible: it survived the rebuild and fed the index a second copy of every
    /// write.
    #[tokio::test]
    async fn a_trigger_under_an_earlier_build_s_name_is_removed() {
        let store = store_with(&[("note", 5, "a note", "xylophone")]).await;
        let connection = store.writable().expect("connected");
        connection
            .call(|connection| -> rusqlite::Result<_> {
                connection.execute_batch(
                    "DROP TRIGGER memories_after_update;
                     CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
                         INSERT INTO memories_fts(memories_fts, rowid, name, description, tags, body)
                         VALUES ('delete', old.id, old.name, old.description, old.tags, old.body);
                         INSERT INTO memories_fts(rowid, name, description, tags, body)
                         VALUES (new.id, new.name, new.description, new.tags, new.body);
                     END;",
                )
            })
            .await
            .expect("the trigger an earlier build left");

        connection
            .call(|connection| -> rusqlite::Result<_> { create_tables(connection) })
            .await
            .expect("second open");

        let names: Vec<String> = connection
            .call(|connection| -> rusqlite::Result<_> {
                connection
                    .prepare(
                        "SELECT name FROM sqlite_master WHERE type = 'trigger' \
                         AND tbl_name = 'memories' ORDER BY name",
                    )?
                    .query_map([], |row| row.get(0))?
                    .collect()
            })
            .await
            .expect("the triggers on the table");
        assert_eq!(
            names,
            [
                "memories_after_delete",
                "memories_after_insert",
                "memories_after_update"
            ],
            "only this build's triggers remain"
        );
    }

    /// Every new field keeps its stored value when a write omits it, so a reword cannot cost a
    /// note its load, its dependencies, its source or its expiry; the explicit empties clear.
    #[tokio::test]
    async fn omitting_a_new_field_keeps_it_and_the_explicit_empty_clears_it() {
        let store = MemoryStore::for_test().await.expect("store");
        let expires = SystemTime::now() + Duration::from_secs(30 * 86_400);
        let written = store
            .write(WriteRequest {
                name: "note".to_string(),
                description: Some("a fact".to_string()),
                load: Some(crate::memory::Load::Search),
                references: Some(vec!["alice-timezone".to_string(), "/srv/x".to_string()]),
                source: Some("alice, in chat".to_string()),
                expires_at: Some(Some(expires)),
                ..WriteRequest::default()
            })
            .await
            .expect("create");
        assert_eq!(written.load, crate::memory::Load::Search);
        assert_eq!(written.references, ["alice-timezone", "/srv/x"]);
        assert_eq!(written.source, "alice, in chat");
        assert_eq!(
            written.expires_at.map(crate::memory::render_stamp),
            Some(crate::memory::render_stamp(expires))
        );
        assert_eq!(written.verified_at, written.created_at);

        let reworded = store
            .write(WriteRequest {
                name: "note".to_string(),
                description: Some("the fact, reworded".to_string()),
                ..WriteRequest::default()
            })
            .await
            .expect("reword");
        assert_eq!(reworded.load, crate::memory::Load::Search);
        assert_eq!(reworded.references, ["alice-timezone", "/srv/x"]);
        assert_eq!(reworded.source, "alice, in chat");
        assert_eq!(reworded.expires_at, written.expires_at);

        let cleared = store
            .write(WriteRequest {
                name: "note".to_string(),
                load: Some(crate::memory::Load::Index),
                references: Some(Vec::new()),
                source: Some(String::new()),
                expires_at: Some(None),
                ..WriteRequest::default()
            })
            .await
            .expect("clear");
        assert_eq!(cleared.load, crate::memory::Load::Index);
        assert!(cleared.references.is_empty());
        assert!(cleared.source.is_empty());
        assert!(cleared.expires_at.is_none());
    }

    /// `verified_at` is the row's own stamp when the row is created and moves on `verified` and
    /// on nothing else: a body rewrite, a reword and a retag all leave it where it was.
    #[tokio::test]
    async fn only_a_verification_moves_the_verification_stamp() {
        let store = MemoryStore::for_test().await.expect("store");
        let created = store
            .write(full("note", 5, "a fact", "the detail"))
            .await
            .expect("create");
        assert_eq!(created.verified_at, created.created_at);

        tokio::time::sleep(Duration::from_millis(5)).await;
        let rewritten = store
            .write(WriteRequest {
                name: "note".to_string(),
                description: Some("a fact, reworded".to_string()),
                body: Some("new detail".to_string()),
                tags: Some(vec!["infra".to_string()]),
                priority: Some(2),
                ..WriteRequest::default()
            })
            .await
            .expect("rewrite");
        assert_eq!(
            rewritten.verified_at, created.verified_at,
            "a rewrite is not a verification"
        );
        assert!(rewritten.updated_at > created.updated_at);

        tokio::time::sleep(Duration::from_millis(5)).await;
        let verified = store
            .write(WriteRequest {
                name: "note".to_string(),
                verified: true,
                ..WriteRequest::default()
            })
            .await
            .expect("verify");
        assert!(verified.verified_at > created.verified_at);
        assert_eq!(verified.verified_at, verified.updated_at);
        assert_eq!(verified.body.as_deref(), Some("new detail"));
        assert_eq!(verified.created_at, created.created_at);
    }

    /// The filter's three questions select the same rows at the listing, the filter-only
    /// results, the backlink query and both search tiers.
    #[tokio::test]
    async fn a_filter_selects_by_reference_verification_and_expiry() {
        let store = MemoryStore::for_test().await.expect("store");
        let past = SystemTime::now() - Duration::from_secs(86_400);
        for (name, references, expires, body) in [
            (
                "a",
                vec!["Alice-Timezone".to_string(), "/srv/x".to_string()],
                None,
                "deploy from 深圳",
            ),
            (
                "b",
                vec!["alice-timezone".to_string()],
                None,
                "deploy from home",
            ),
            ("c", Vec::new(), Some(Some(past)), "deploy from 深圳 too"),
        ] {
            store
                .write(WriteRequest {
                    name: name.to_string(),
                    description: Some(format!("note {name}")),
                    body: Some(body.to_string()),
                    references: Some(references),
                    expires_at: expires,
                    ..WriteRequest::default()
                })
                .await
                .expect("write");
        }
        // A row verified long ago, which no write door can produce today.
        store
            .plant_row_for_test("old", "note old")
            .await
            .expect("planted");

        let listed = |memories: Vec<Memory>| -> Vec<String> {
            memories.into_iter().map(|memory| memory.name).collect()
        };
        let referencing = |target: &str| MemoryFilter {
            referencing: Some(target.to_string()),
            ..MemoryFilter::default()
        };
        // Index order: newest first within the band.
        assert_eq!(
            listed(
                store
                    .list(&referencing("ALICE-timezone"))
                    .await
                    .expect("list")
            ),
            ["b", "a"]
        );
        assert_eq!(
            listed(store.list(&referencing("/srv/x")).await.expect("list")),
            ["a"]
        );
        assert_eq!(
            store
                .referenced_by("alice-timezone", &MemoryView::All)
                .await
                .expect("backlinks"),
            [
                ("b".to_string(), "note b".to_string()),
                ("a".to_string(), "note a".to_string())
            ]
        );
        let expired = MemoryFilter {
            expired: true,
            ..MemoryFilter::default()
        };
        assert_eq!(listed(store.list(&expired).await.expect("list")), ["c"]);
        let stale = MemoryFilter {
            unverified_for: Some(Duration::from_secs(30 * 86_400)),
            ..MemoryFilter::default()
        };
        assert_eq!(listed(store.list(&stale).await.expect("list")), ["old"]);
        assert_eq!(
            listed(store.list(&MemoryFilter::default()).await.expect("list")),
            ["c", "b", "a", "old"]
        );

        let only_expired = store.filtered(&expired, 10).await.expect("filtered");
        assert_eq!(names(&only_expired), ["c"]);
        assert_eq!(only_expired.matched, 1);
        assert_eq!(only_expired.hits[0].body, "deploy from 深圳 too");

        let words = Terms::parse(&["deploy".to_string()]);
        let by_words = store
            .search(words.match_expression(), &referencing("alice-timezone"), 10)
            .await
            .expect("search");
        let mut by_words = names(&by_words);
        by_words.sort_unstable();
        assert_eq!(by_words, ["a", "b"]);
        let by_scan = store
            .substring_search(&["深圳".to_string()], &expired, 10)
            .await
            .expect("scan");
        assert_eq!(names(&by_scan), ["c"]);
        let found = store
            .find(&["deploy".to_string()], &expired, 10)
            .await
            .expect("find")
            .expect("words");
        assert_eq!(names(&found.results), ["c"]);

        // Two and three filters at once, at every door that binds them after its own
        // parameters: the clause numbers each placeholder from where the statement left off.
        let referencing_and_expired = MemoryFilter {
            referencing: Some("alice-timezone".to_string()),
            expired: true,
            ..MemoryFilter::default()
        };
        assert!(
            listed(store.list(&referencing_and_expired).await.expect("list")).is_empty(),
            "a and b reference the target but have not expired"
        );
        let all_three = MemoryFilter {
            view: MemoryView::Global,
            referencing: Some("alice-timezone".to_string()),
            unverified_for: Some(Duration::from_secs(1)),
            expired: false,
        };
        tokio::time::sleep(Duration::from_millis(1_100)).await;
        assert_eq!(listed(store.list(&all_three).await.expect("list")), [
            "b", "a"
        ]);
        let by_words = store
            .search(words.match_expression(), &all_three, 10)
            .await
            .expect("search");
        let mut by_words = names(&by_words);
        by_words.sort_unstable();
        assert_eq!(by_words, ["a", "b"]);
        let by_scan = store
            .substring_search(&["deploy".to_string()], &all_three, 10)
            .await
            .expect("scan");
        let mut by_scan = names(&by_scan);
        by_scan.sort_unstable();
        assert_eq!(by_scan, ["a", "b"]);
        let by_scan_expired = store
            .substring_search(&["深圳".to_string()], &referencing_and_expired, 10)
            .await
            .expect("scan");
        assert!(names(&by_scan_expired).is_empty());
    }

    /// The index carries a body for an inline memory whatever its priority, and none for a
    /// priority-0 memory that is merely listed: the band follows `load`, not the number.
    #[tokio::test]
    async fn the_index_inlines_by_load_not_priority() {
        let store = MemoryStore::for_test().await.expect("store");
        for (name, priority, load) in [
            ("listed-rule", 0, crate::memory::Load::Index),
            ("inlined-note", 5, crate::memory::Load::Inline),
            ("searched", 0, crate::memory::Load::Search),
        ] {
            store
                .write(WriteRequest {
                    name: name.to_string(),
                    description: Some(name.to_string()),
                    body: Some(format!("body of {name}")),
                    priority: Some(priority),
                    load: Some(load),
                    ..WriteRequest::default()
                })
                .await
                .expect("write");
        }
        let index = store.index(&MemoryView::All).await.expect("index");
        let body_of = |name: &str| {
            index
                .iter()
                .find(|memory| memory.name == name)
                .expect("listed")
                .body
                .clone()
        };
        assert_eq!(body_of("listed-rule"), None);
        assert_eq!(
            body_of("inlined-note").as_deref(),
            Some("body of inlined-note")
        );
        assert_eq!(body_of("searched"), None);
    }

    /// A stored reference is kept as written or dropped whole, never trimmed into a different
    /// pointer, and a column that is not a JSON array of strings reads as no references at all.
    #[tokio::test]
    async fn a_planted_reference_list_is_read_by_the_write_door_s_rule() {
        let store = MemoryStore::for_test().await.expect("store");
        store
            .write(full("note", 5, "a note", "body"))
            .await
            .expect("write");
        store
            .plant_references_for_test(
                "note",
                "[\"kept\", \"two\\nlines\", 7, \"\", \"also-kept\"]",
            )
            .await
            .expect("planted");
        let read = store
            .get("note", &MemoryView::All)
            .await
            .expect("get")
            .expect("row");
        assert_eq!(read.references, ["kept", "also-kept"]);
        store
            .plant_references_for_test("note", "{\"not\": \"a list\"}")
            .await
            .expect("planted");
        let read = store
            .get("note", &MemoryView::All)
            .await
            .expect("get")
            .expect("row");
        assert!(read.references.is_empty(), "{:?}", read.references);
    }

    /// Freshness decays from the verification, not the creation, so verifying an old note lifts
    /// it in the ranking without a rewrite.
    #[tokio::test]
    async fn verifying_a_memory_makes_it_fresh_for_ranking() {
        let store = MemoryStore::for_test().await.expect("store");
        store
            .plant_row_for_test("old", "the deploy host moved")
            .await
            .expect("planted");
        let terms = Terms::parse(&["deploy host".to_string()]);
        let score = |results: SearchResults| results.hits[0].score;
        let before = score(
            store
                .search(terms.match_expression(), &MemoryFilter::default(), 5)
                .await
                .expect("search"),
        );
        store
            .write(WriteRequest {
                name: "old".to_string(),
                verified: true,
                ..WriteRequest::default()
            })
            .await
            .expect("verify");
        let after = score(
            store
                .search(terms.match_expression(), &MemoryFilter::default(), 5)
                .await
                .expect("search"),
        );
        assert!(
            after > before,
            "a verified note ranks fresher: {before} before, {after} after"
        );
    }

    /// An inline write, for the budget tests: the body, the load, and nothing else of note.
    fn inline(name: &str, body: &str) -> WriteRequest {
        WriteRequest {
            name: name.to_string(),
            description: Some(format!("About {name}")),
            body: Some(body.to_string()),
            load: Some(crate::memory::Load::Inline),
            ..WriteRequest::default()
        }
    }

    /// A write that would leave the in-force inline bodies over the budget is refused inside its
    /// own transaction, so nothing of it lands: not the row, not a changed body, not a moved
    /// `updated_at`. The refusal names the budget and the largest inline memories as they would
    /// have stood, the one being written among them.
    #[tokio::test]
    async fn an_inline_write_past_the_budget_is_refused_and_nothing_lands() {
        let store = MemoryStore::for_test()
            .await
            .expect("store")
            .with_inline_max_bytes_for_test(100);
        store
            .write(inline("tone", &"t".repeat(60)))
            .await
            .expect("60 of 100");
        let refused = store
            .write(inline("deploy", &"d".repeat(50)))
            .await
            .expect_err("110 of 100");
        let message = refused.to_string();
        assert!(
            message.contains("inline memories would total 110 B against the 100 B inline budget"),
            "{message}"
        );
        assert!(
            message.contains("shorten 'tone' (60 B) or 'deploy' (50 B)"),
            "{message}"
        );
        assert!(message.contains("`load: index`"), "{message}");
        assert!(
            store
                .get("deploy", &MemoryView::All)
                .await
                .expect("get")
                .is_none(),
            "the refused row must not exist"
        );

        // Growing an existing body past the budget is refused the same way, and the row stays as
        // it was, stamps included.
        let before = store
            .get("tone", &MemoryView::All)
            .await
            .expect("get")
            .expect("tone");
        let refused = store
            .write(inline("tone", &"t".repeat(101)))
            .await
            .expect_err("101 of 100");
        assert!(refused.to_string().contains("'tone' (101 B)"), "{refused}");
        let after = store
            .get("tone", &MemoryView::All)
            .await
            .expect("get")
            .expect("tone");
        assert_eq!(after, before, "a refused write leaves the row as it was");

        // Exactly the budget is admitted.
        store
            .write(inline("deploy", &"d".repeat(40)))
            .await
            .expect("100 of 100");
    }

    /// Only a write that grows an overfull tier is refused. Shrinking it, leaving its size alone
    /// or moving a memory out of the band always lands, or a store the budget was lowered under
    /// could never be brought back within it one memory at a time.
    #[tokio::test]
    async fn an_overfull_inline_tier_can_be_trimmed_but_not_grown() {
        let roomy = MemoryStore::for_test().await.expect("store");
        for name in ["a", "b", "c"] {
            roomy
                .write(inline(name, &"x".repeat(300)))
                .await
                .expect("write under the default budget");
        }
        // The budget lowered beneath what the store holds.
        let store = roomy.with_inline_max_bytes_for_test(100);

        store
            .write(inline("a", &"x".repeat(250)))
            .await
            .expect("shrinking lands though the tier is still over");
        store
            .write(WriteRequest {
                name: "b".to_string(),
                description: Some("reworded".to_string()),
                ..WriteRequest::default()
            })
            .await
            .expect("a write that leaves the tier where it is lands");
        store
            .write(WriteRequest {
                name: "c".to_string(),
                load: Some(crate::memory::Load::Index),
                ..WriteRequest::default()
            })
            .await
            .expect("moving a memory out of the band lands");

        store
            .write(inline("d", "x"))
            .await
            .expect_err("a new inline memory grows the tier");
        store
            .write(inline("a", &"x".repeat(251)))
            .await
            .expect_err("a body growing by one byte grows the tier");
        store
            .write(WriteRequest {
                name: "c".to_string(),
                load: Some(crate::memory::Load::Inline),
                ..WriteRequest::default()
            })
            .await
            .expect_err("moving a body back into the band grows the tier");

        // The body door follows the same rule.
        let shrunk = "x".repeat(240);
        store
            .write_body(
                "a",
                &MemoryView::All,
                &"x".repeat(250),
                shrunk.clone(),
                false,
            )
            .await
            .expect("shrinking through the body door lands");
        let refused = store
            .write_body("a", &MemoryView::All, &shrunk, "x".repeat(241), false)
            .await
            .expect_err("growing through the body door is refused");
        assert!(
            refused.to_string().contains("inline memories would total"),
            "{refused}"
        );
        assert_eq!(
            store
                .get("a", &MemoryView::All)
                .await
                .expect("get")
                .expect("a")
                .body
                .as_deref(),
            Some(shrunk.as_str()),
            "a refused body write leaves the body as it was"
        );
    }

    /// The budget counts the bodies the band renders: an inline memory past its expiry, an
    /// `index` memory however long and an empty body spend nothing, and a cleared expiry brings a
    /// body back into the count.
    #[tokio::test]
    async fn the_inline_budget_counts_in_force_inline_bodies_only() {
        let store = MemoryStore::for_test()
            .await
            .expect("store")
            .with_inline_max_bytes_for_test(100);
        let past = Some(Some(SystemTime::now() - Duration::from_secs(60)));
        store
            .write(WriteRequest {
                expires_at: past,
                ..inline("old", &"o".repeat(90))
            })
            .await
            .expect("an expired inline body spends nothing");
        store
            .write(WriteRequest {
                load: Some(crate::memory::Load::Index),
                ..inline("handbook", &"h".repeat(5_000))
            })
            .await
            .expect("an index body spends nothing");
        store
            .write(inline("tone", &"t".repeat(90)))
            .await
            .expect("90 of 100");

        // Clearing the expiry would put `old` back in force: 180 of 100.
        let refused = store
            .write(WriteRequest {
                name: "old".to_string(),
                expires_at: Some(None),
                ..WriteRequest::default()
            })
            .await
            .expect_err("a cleared expiry brings the body back into the count");
        assert!(refused.to_string().contains("'old' (90 B)"), "{refused}");

        // Expiring `tone` instead frees its bytes, and `old` can come back.
        store
            .write(WriteRequest {
                name: "tone".to_string(),
                expires_at: past,
                ..WriteRequest::default()
            })
            .await
            .expect("expiring shrinks the tier");
        store
            .write(WriteRequest {
                name: "old".to_string(),
                expires_at: Some(None),
                ..WriteRequest::default()
            })
            .await
            .expect("90 of 100 again");
    }

    fn session(project: &str) -> MemoryView {
        MemoryView::Session(project.to_string())
    }

    /// Each door spells its view through one constructor: a session's project or its absence,
    /// the operator's flag, and the tiers that can cite a memory.
    #[test]
    fn the_view_constructors_read_each_door_s_spelling() {
        assert_eq!(MemoryView::for_session(None), MemoryView::Global);
        assert_eq!(MemoryView::for_session(Some("")), MemoryView::Global);
        assert_eq!(MemoryView::for_session(Some("p")), session("p"));
        assert_eq!(MemoryView::for_operator(None), MemoryView::All);
        assert_eq!(MemoryView::for_operator(Some("")), MemoryView::Global);
        assert_eq!(
            MemoryView::for_operator(Some("p")),
            MemoryView::Project("p".to_string())
        );
        assert_eq!(MemoryView::citing(""), MemoryView::All);
        assert_eq!(MemoryView::citing("p"), session("p"));
    }

    fn in_project(project: &str, request: WriteRequest) -> WriteRequest {
        WriteRequest {
            view: session(project),
            project: None,
            ..request
        }
    }

    fn names_of(memories: &[crate::memory::Memory]) -> Vec<&str> {
        memories.iter().map(|memory| memory.name.as_str()).collect()
    }

    /// Where a write lands, over every combination the doors can present: the stored tier is
    /// kept unless one is named, naming one moves the one visible holder, a global and a project
    /// memory never share a name, and a name held in several projects is refused where no one
    /// row is meant.
    #[test]
    fn a_write_s_tier_follows_the_view_the_request_and_the_holders() {
        let global = String::new();
        let p = "p".to_string();
        let q = "q".to_string();
        let landed = |project: &str| WriteTier {
            project: project.to_string(),
            moved_from: None,
        };
        let moved = |project: &str, from: &str| WriteTier {
            project: project.to_string(),
            moved_from: Some(from.to_string()),
        };
        let tier = |view: &MemoryView, named: Option<&str>, holders: &[String]| {
            resolve_write_tier("n", view, named, holders)
        };

        // Nothing holds the name: the view's own tier, or the one named.
        assert_eq!(tier(&MemoryView::Global, None, &[]), Ok(landed("")));
        assert_eq!(tier(&session("p"), None, &[]), Ok(landed("p")));
        assert_eq!(tier(&session("p"), Some(""), &[]), Ok(landed("")));
        assert_eq!(tier(&MemoryView::All, None, &[]), Ok(landed("")));
        assert_eq!(tier(&MemoryView::All, Some("p"), &[]), Ok(landed("p")));
        // An unnamed tier keeps the visible holder's.
        assert_eq!(
            tier(&session("p"), None, std::slice::from_ref(&global)),
            Ok(landed(""))
        );
        assert_eq!(
            tier(&session("p"), None, std::slice::from_ref(&p)),
            Ok(landed("p"))
        );
        assert_eq!(
            tier(&MemoryView::All, None, std::slice::from_ref(&q)),
            Ok(landed("q"))
        );
        // Naming the tier of the one visible holder moves it: the re-file, both ways.
        assert_eq!(
            tier(&session("p"), Some("p"), std::slice::from_ref(&global)),
            Ok(moved("p", ""))
        );
        assert_eq!(
            tier(&session("p"), Some(""), std::slice::from_ref(&p)),
            Ok(moved("", "p"))
        );
        assert_eq!(
            tier(&MemoryView::All, Some("q"), std::slice::from_ref(&global)),
            Ok(moved("q", ""))
        );
        // Another project's copy of the name is no collision for a project write.
        assert_eq!(
            tier(&session("p"), Some("p"), std::slice::from_ref(&q)),
            Ok(landed("p"))
        );
        assert_eq!(
            tier(&session("p"), None, std::slice::from_ref(&q)),
            Ok(landed("p"))
        );
        // A global and a project memory never share a name, and a holder the view cannot see is
        // not moved.
        assert_eq!(
            tier(&session("p"), Some(""), std::slice::from_ref(&q)),
            Err("'n' is a memory of another project; choose another name".to_string()),
            "a holder outside the view is not named"
        );
        assert_eq!(
            tier(&MemoryView::Global, None, std::slice::from_ref(&q)),
            Err("'n' is a memory of another project; choose another name".to_string())
        );
        assert_eq!(
            tier(&MemoryView::All, None, std::slice::from_ref(&q)),
            Ok(landed("q"))
        );
        assert_eq!(
            tier(
                &MemoryView::Project("p".to_string()),
                Some("p"),
                std::slice::from_ref(&global)
            ),
            Err("'n' is a global memory; choose another name".to_string())
        );
        // Several projects hold it: no one row is meant.
        assert_eq!(
            tier(&MemoryView::All, None, &[p.clone(), q.clone()]),
            Err("memory 'n' exists in projects 'p' and 'q'; name the project".to_string())
        );
        assert_eq!(
            tier(&MemoryView::All, Some(""), &[p.clone(), q.clone()]),
            Err("'n' is a memory of projects 'p', 'q'; choose another name".to_string())
        );
        assert_eq!(
            tier(&session("p"), Some(""), &[p.clone(), q.clone()]),
            Err("'n' is a memory of other projects; choose another name".to_string()),
            "a session names only the holders it sees, and here it cannot name them all"
        );
        // A named tier the name is already in is an update, whatever else holds it.
        assert_eq!(
            tier(&MemoryView::All, Some("p"), &[p.clone(), q.clone()]),
            Ok(landed("p"))
        );
    }

    /// Every read door answers within the view it was given: a session on a project sees the
    /// global tier and its own, a session with no project sees the global tier alone, and the
    /// operator's filters see one tier or all of them.
    #[tokio::test]
    async fn every_read_door_answers_within_its_view() {
        let store = MemoryStore::for_test().await.expect("store");
        store
            .write(full("shared", 5, "Global note", "zebra body"))
            .await
            .expect("global");
        store
            .write(in_project("p1", full("first", 5, "P1 note", "zebra body")))
            .await
            .expect("p1");
        store
            .write(in_project("p2", full("second", 5, "P2 note", "zebra body")))
            .await
            .expect("p2");
        store
            .write(WriteRequest {
                references: Some(vec!["shared".to_string()]),
                ..in_project("p2", full("citing", 5, "P2 citing note", ""))
            })
            .await
            .expect("p2 citing");

        let index = |view: MemoryView| {
            let store = store.clone();
            async move {
                let mut names: Vec<String> = store
                    .index(&view)
                    .await
                    .expect("index")
                    .iter()
                    .map(|memory| memory.name.clone())
                    .collect();
                names.sort();
                names
            }
        };
        assert_eq!(index(MemoryView::Global).await, ["shared"]);
        assert_eq!(index(session("p1")).await, ["first", "shared"]);
        assert_eq!(index(session("p2")).await, ["citing", "second", "shared"]);
        assert_eq!(index(MemoryView::Project("p2".to_string())).await, [
            "citing", "second"
        ]);
        assert_eq!(index(MemoryView::All).await, [
            "citing", "first", "second", "shared"
        ]);

        let got = store
            .get("first", &session("p1"))
            .await
            .expect("get")
            .expect("visible");
        assert_eq!(got.project, "p1");
        assert!(
            store
                .get("first", &MemoryView::Global)
                .await
                .expect("get")
                .is_none()
        );
        assert!(
            store
                .get("first", &session("p2"))
                .await
                .expect("get")
                .is_none()
        );
        assert!(
            store
                .get("shared", &session("p2"))
                .await
                .expect("get")
                .is_some()
        );

        let hits = |view: MemoryView| {
            let store = store.clone();
            async move {
                let filter = MemoryFilter {
                    view,
                    ..MemoryFilter::default()
                };
                let found = store
                    .find(&["zebra".to_string()], &filter, 10)
                    .await
                    .expect("find")
                    .expect("words");
                let mut names: Vec<String> = found
                    .results
                    .hits
                    .iter()
                    .map(|hit| hit.name.clone())
                    .collect();
                names.sort();
                names
            }
        };
        assert_eq!(hits(MemoryView::Global).await, ["shared"]);
        assert_eq!(hits(session("p1")).await, ["first", "shared"]);
        assert_eq!(hits(MemoryView::Project("p2".to_string())).await, [
            "second"
        ]);
        assert_eq!(hits(MemoryView::All).await, ["first", "second", "shared"]);
        let substring = store
            .substring_search(
                &["ebra".to_string()],
                &MemoryFilter {
                    view: session("p1"),
                    ..MemoryFilter::default()
                },
                10,
            )
            .await
            .expect("scan");
        assert_eq!(names(&substring), ["first", "shared"]);

        // Backlinks within the view: the citing memory is p2's, so p1 does not see the citation.
        assert!(
            store
                .referenced_by("shared", &session("p1"))
                .await
                .expect("backlinks")
                .is_empty()
        );
        assert_eq!(
            store
                .referenced_by("shared", &session("p2"))
                .await
                .expect("backlinks")
                .len(),
            1
        );
        assert_eq!(
            names_of(
                &store
                    .export_all(&MemoryView::Project("p1".to_string()))
                    .await
                    .expect("export")
            ),
            ["first"]
        );
        assert_eq!(
            names_of(&store.export_all(&MemoryView::All).await.expect("export")),
            ["shared", "first", "citing", "second"],
            "an export orders by project and then by name"
        );
        assert_eq!(
            store
                .export_all(&MemoryView::Project("p2".to_string()))
                .await
                .expect("export")
                .len(),
            2
        );

        // A read counts only within the view: a session with no project never opened p1's note.
        store
            .record_read("first", &MemoryView::Global)
            .await
            .expect("read");
        store
            .record_read("first", &session("p1"))
            .await
            .expect("read");
        assert_eq!(
            store
                .get("first", &MemoryView::All)
                .await
                .expect("get")
                .expect("row")
                .read_count,
            1
        );
    }

    /// Naming the other tier for a memory the session sees moves it, the row keeping its
    /// identity and its index entry; leaving the tier unnamed keeps the memory where it is; and a
    /// name a session cannot see is refused with nothing landed.
    #[tokio::test]
    async fn a_write_moves_a_visible_memory_between_tiers_and_refuses_an_invisible_name() {
        let store = MemoryStore::for_test().await.expect("store");
        store
            .write(full("rule", 5, "Global rule", "quokka body"))
            .await
            .expect("global");
        // Unnamed: the global row is updated in place.
        let updated = store
            .write(in_project("p1", WriteRequest {
                name: "rule".to_string(),
                body: Some("quokka body, refined".to_string()),
                ..WriteRequest::default()
            }))
            .await
            .expect("update");
        assert_eq!(updated.project, "");
        // Named: the global row becomes p1's.
        let moved = store
            .write(WriteRequest {
                project: Some("p1".to_string()),
                ..in_project("p1", WriteRequest {
                    name: "rule".to_string(),
                    description: Some("P1 rule".to_string()),
                    ..WriteRequest::default()
                })
            })
            .await
            .expect("move");
        assert_eq!(moved.project, "p1");
        assert_eq!(moved.description, "P1 rule");
        assert_eq!(moved.body.as_deref(), Some("quokka body, refined"));
        assert!(
            store
                .get("rule", &MemoryView::Global)
                .await
                .expect("get")
                .is_none()
        );
        assert_eq!(
            store
                .get("rule", &MemoryView::All)
                .await
                .expect("get")
                .expect("row")
                .project,
            "p1"
        );
        let filter = |view: MemoryView| MemoryFilter {
            view,
            ..MemoryFilter::default()
        };
        assert!(
            store
                .find(&["quokka".to_string()], &filter(MemoryView::Global), 10)
                .await
                .expect("find")
                .expect("words")
                .results
                .hits
                .is_empty()
        );
        assert_eq!(
            names(
                &store
                    .find(&["quokka".to_string()], &filter(session("p1")), 10)
                    .await
                    .expect("find")
                    .expect("words")
                    .results
            ),
            ["rule"]
        );
        // And back to global by naming it.
        let back = store
            .write(WriteRequest {
                project: Some(String::new()),
                ..in_project("p1", WriteRequest {
                    name: "rule".to_string(),
                    ..WriteRequest::default()
                })
            })
            .await
            .expect("move back");
        assert_eq!(back.project, "");

        // A global write of a name a project holds is refused, and nothing lands.
        store
            .write(in_project("p2", full("secret", 5, "P2 only", "")))
            .await
            .expect("p2");
        let refused = store
            .write(full("secret", 5, "Would shadow", ""))
            .await
            .expect_err("a global name may not equal a project's");
        // Said without naming p2: the writer is on no project, and a name outside its view is not
        // one it can act on.
        assert!(
            refused
                .to_string()
                .ends_with("'secret' is a memory of another project; choose another name"),
            "{refused}"
        );
        assert_eq!(
            store
                .get("secret", &MemoryView::All)
                .await
                .expect("get")
                .expect("row")
                .description,
            "P2 only"
        );
        // From p1, the same name is free: another project's copy is no collision.
        let own = store
            .write(in_project("p1", full("secret", 5, "P1's own", "")))
            .await
            .expect("p1 may hold the name");
        assert_eq!(own.project, "p1");
        // A creating write still needs a description, in a project as everywhere.
        let refused = store
            .write(in_project("p1", WriteRequest {
                name: "absent".to_string(),
                body: Some("b".to_string()),
                ..WriteRequest::default()
            }))
            .await
            .expect_err("no description");
        assert!(
            refused.to_string().contains("description is required"),
            "{refused}"
        );
    }

    /// The operator's whole-store view can hold a name in several projects; every door that
    /// takes a bare name then refuses, naming them, and removes or changes nothing.
    #[tokio::test]
    async fn an_operator_s_bare_name_in_several_projects_is_refused_at_every_door() {
        let store = MemoryStore::for_test().await.expect("store");
        for project in ["a", "b"] {
            store
                .write(WriteRequest {
                    view: MemoryView::All,
                    project: Some(project.to_string()),
                    ..full("notes", 5, &format!("{project} notes"), "body")
                })
                .await
                .expect("write");
        }
        let expected = "memory 'notes' exists in projects 'a' and 'b'; name the project";
        let refused = store.get("notes", &MemoryView::All).await.expect_err("get");
        assert!(refused.to_string().ends_with(expected), "{refused}");
        let refused = store
            .delete("notes", &MemoryView::All)
            .await
            .expect_err("delete");
        assert!(refused.to_string().ends_with(expected), "{refused}");
        let refused = store
            .write_body(
                "notes",
                &MemoryView::All,
                "body",
                "changed".to_string(),
                false,
            )
            .await
            .expect_err("edit");
        assert!(refused.to_string().ends_with(expected), "{refused}");
        let refused = store
            .write(WriteRequest {
                view: MemoryView::All,
                project: None,
                ..full("notes", 5, "which one", "")
            })
            .await
            .expect_err("write");
        assert!(refused.to_string().ends_with(expected), "{refused}");
        for project in ["a", "b"] {
            assert_eq!(
                store
                    .export_all(&MemoryView::Project(project.to_string()))
                    .await
                    .expect("export")
                    .len(),
                1,
                "nothing moved in project '{project}'"
            );
        }
        // Named, each door reaches the one row.
        let b = MemoryView::Project("b".to_string());
        assert_eq!(
            store
                .get("notes", &b)
                .await
                .expect("get")
                .expect("row")
                .description,
            "b notes"
        );
        assert_eq!(
            store
                .write_body("notes", &b, "body", "changed".to_string(), false)
                .await
                .expect("edit"),
            BodyWrite::Saved
        );
        assert_eq!(
            store
                .delete("notes", &b)
                .await
                .expect("delete")
                .expect("row")
                .project,
            "b"
        );
        assert_eq!(
            store
                .get("notes", &MemoryView::All)
                .await
                .expect("get")
                .expect("one left")
                .project,
            "a"
        );
    }

    /// The inline budget is measured as a session's band sees it: a project write counts the
    /// global bodies and the project's, and a global write counts the global bodies and the
    /// fullest project's, since a global memory is in every project's band.
    #[tokio::test]
    async fn the_inline_budget_is_measured_per_tier() {
        let store = MemoryStore::for_test()
            .await
            .expect("store")
            .with_inline_max_bytes_for_test(100);
        store
            .write(inline("g", &"g".repeat(40)))
            .await
            .expect("global 40");
        store
            .write(in_project("p1", inline("one", &"1".repeat(50))))
            .await
            .expect("p1 sees 90");
        store
            .write(in_project("p2", inline("two", &"2".repeat(30))))
            .await
            .expect("p2 sees 70");
        // A global write is judged against the fullest project's band.
        let refused = store
            .write(inline("g2", &"h".repeat(15)))
            .await
            .expect_err("global 55 + p1 50 = 105");
        let message = refused.to_string();
        assert!(
            message.contains(
                "inline memories (global and project 'p1') would total 105 B against the 100 B"
            ),
            "{message}"
        );
        assert!(
            message.contains("'g' (40 B)") && !message.contains("'one'"),
            "a writer on no project is told to shorten what it sees, not p1's: {message}"
        );
        assert!(
            store
                .get("g2", &MemoryView::All)
                .await
                .expect("get")
                .is_none()
        );
        // The same bytes fit p2's band.
        store
            .write(in_project("p2", inline("three", &"3".repeat(25))))
            .await
            .expect("p2 sees 95");
        // And p1's band refuses what would overfill it, naming itself.
        let refused = store
            .write(in_project("p1", inline("four", &"4".repeat(15))))
            .await
            .expect_err("p1 sees 105");
        assert!(
            refused.to_string().contains("(global and project 'p1')"),
            "{refused}"
        );
        // A body edit is judged in the row's own tier.
        let refused = store
            .write_body(
                "two",
                &session("p2"),
                &"2".repeat(30),
                "2".repeat(40),
                false,
            )
            .await
            .expect_err("p2 would see 105");
        assert!(
            refused.to_string().contains("(global and project 'p2')"),
            "{refused}"
        );
    }

    /// An import restores what the files state, the stamps and the read count included, in one
    /// transaction: a project the store lacks, a name the tier already holds without `force`,
    /// a name the other kind of tier holds and the inline budget each refuse the whole import, and
    /// `force` alone overwrites a name the tier holds.
    #[tokio::test]
    async fn an_import_restores_the_files_and_refuses_what_a_write_would() {
        use crate::memory::{ImportedMemory, Load};

        let database = crate::store::Store::for_test().await;
        database.add_project("p").await.expect("a project");
        let store = database.memory_store(true, crate::config::MEMORY_INLINE_MAX_BYTES_DEFAULT);
        let stamp = |raw: &str| crate::memory::parse_stamp(raw).expect("a stamp");
        let entry = |name: &str, project: &str, description: &str| ImportedMemory {
            name: name.to_string(),
            project: project.to_string(),
            description: description.to_string(),
            priority: 3,
            load: Load::Index,
            created_at: stamp("2024-05-06T07:08:09+00:00"),
            verified_at: stamp("2024-06-06T07:08:09+00:00"),
            expires_at: None,
            source: "a test".to_string(),
            references: vec!["other".to_string()],
            tags: vec!["t".to_string()],
            read_count: 7,
            body: "the body".to_string(),
        };

        let count = store
            .import(
                vec![
                    entry("tone", "", "Be brief"),
                    entry("stack", "p", "Uses pnpm"),
                ],
                false,
            )
            .await
            .expect("imported");
        assert_eq!(count, 2);
        let tone = store
            .get("tone", &MemoryView::All)
            .await
            .expect("get")
            .expect("tone");
        assert_eq!(tone.created_at, stamp("2024-05-06T07:08:09+00:00"));
        assert_eq!(tone.verified_at, stamp("2024-06-06T07:08:09+00:00"));
        assert_eq!(tone.read_count, 7);
        assert_eq!(tone.body.as_deref(), Some("the body"));
        assert_eq!(tone.tags, ["t"]);
        assert_eq!(tone.references, ["other"]);
        assert_eq!(tone.source, "a test");
        let stack = store
            .get("stack", &MemoryView::Project("p".to_string()))
            .await
            .expect("get")
            .expect("stack");
        assert_eq!(stack.project, "p");
        assert_eq!(
            names_of(&store.export_all(&MemoryView::All).await.expect("all")),
            ["tone", "stack"]
        );

        // The same files again: refused without `force`, every row untouched; overwritten with it.
        let refused = store
            .import(vec![entry("tone", "", "Be briefer")], false)
            .await
            .expect_err("taken");
        assert_eq!(
            refused.to_string(),
            "'tone' already exists in the global tier; pass `--force` to overwrite it"
        );
        let unchanged = store
            .get("tone", &MemoryView::All)
            .await
            .expect("get")
            .expect("tone");
        assert_eq!(unchanged.description, "Be brief");
        store
            .import(vec![entry("tone", "", "Be briefer")], true)
            .await
            .expect("overwritten");
        let overwritten = store
            .get("tone", &MemoryView::All)
            .await
            .expect("get")
            .expect("tone");
        assert_eq!(overwritten.description, "Be briefer");

        // All or nothing: the good file beside a bad one does not land.
        let refused = store
            .import(
                vec![
                    entry("fine", "", "Would land"),
                    entry("stray", "ghost", "No project"),
                ],
                false,
            )
            .await
            .expect_err("an unknown project");
        assert!(
            refused.to_string().contains("no project named 'ghost'")
                && refused.to_string().contains("`meka project add ghost`"),
            "{refused}"
        );
        assert!(
            store
                .get("fine", &MemoryView::All)
                .await
                .expect("get")
                .is_none(),
            "nothing landed"
        );
        let refused = store
            .import(vec![entry("stack", "", "A global one")], false)
            .await
            .expect_err("a project holds the name");
        assert_eq!(
            refused.to_string(),
            "'stack' is a memory of project 'p', and a global memory may not share its name; \
             rename one first"
        );
        let refused = store
            .import(vec![entry("tone", "p", "A project one")], false)
            .await
            .expect_err("the global tier holds the name");
        assert_eq!(
            refused.to_string(),
            "'tone' is a global memory, and a project memory may not share its name; rename one \
             first"
        );

        // The inline budget holds over the whole import, measured as a write measures it.
        let tight = store.with_inline_max_bytes_for_test(16);
        let refused = tight
            .import(
                vec![ImportedMemory {
                    load: Load::Inline,
                    body: "x".repeat(40),
                    ..entry("rule", "", "A standing rule")
                }],
                false,
            )
            .await
            .expect_err("past the budget");
        assert!(refused.to_string().contains("inline budget"), "{refused}");
        assert!(
            store
                .get("rule", &MemoryView::All)
                .await
                .expect("get")
                .is_none()
        );
    }
}
