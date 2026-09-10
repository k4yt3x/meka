//! Sessions and their event logs: the `sessions` and `messages` tables.

use super::*;
use crate::permission::Permission;

/// Raw row from the `messages` table, the on-disk shape of a single
/// [`crate::conversation::Event`]. Internal to the session module: only the encoder and decoder
/// helpers handle these directly. External consumers go through [`Store::save_event`] /
/// [`Store::load_events`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct StoredMessage {
    pub(super) role: String,
    pub(super) content: String,
    pub(super) created_at: String,
}
/// Result of `Store::create_session_with_metadata`. Carries the canonical RFC 3339
/// `created_at` so the caller's in-memory state shares one timestamp with the DB row; without
/// this, the handler's `SessionEntry.created_at` and the DB `sessions.created_at` would each
/// capture `Utc::now()` independently and drift by a few ms. Re-attach reads the DB value,
/// so the in-memory value has to match for round-trip tests to be deterministic.
#[derive(Debug, Clone)]
pub(crate) struct CreatedSession {
    pub(crate) id: Uuid,
    /// RFC 3339 timestamp written to both `sessions.created_at` and `sessions.updated_at`.
    pub(crate) created_at: String,
}
/// Metadata for one session row, used by JSON session export to reconstruct a session and its
/// sub-agent tree. Omits the derived `title` and the `token_id` fingerprint (which is tied to the
/// exporting deployment and must not travel).
#[derive(Debug, Clone)]
pub(crate) struct SessionMetaRow {
    pub(crate) id: Uuid,
    pub(crate) parent_id: Option<Uuid>,
    pub(crate) created_at: String,
    pub(crate) updated_at: String,
    pub(crate) cwd: Option<PathBuf>,
    /// The level the row records, parsed once here; `None` for a row that records none or one
    /// meka cannot read, which the read warns about.
    pub(crate) permission: Option<Permission>,
    /// Whether calls above the level are submitted for approval.
    pub(crate) approvals: bool,
    pub(crate) capabilities_json: Option<String>,
    /// Workspace roots beyond `cwd`. Empty for every non-ACP session, all of which are
    /// single-root.
    pub(crate) additional_roots: Vec<PathBuf>,
    /// The terms a sub-agent was spawned under, as stored JSON. `None` on every root session.
    pub(crate) subagent_spec_json: Option<String>,
    /// What the session runs on, so an export carries it and an import can restore it rather than
    /// landing every imported session on the empty profile no configuration can name.
    pub(crate) profile: String,
}
/// Per-surface overrides applied to the copy produced by [`Store::fork_session_locked`]. Each
/// `None` field inherits the source session's value.
///
/// `cwd` and `additional_roots` exist because ACP models `session/fork` as a session-*creation*
/// request: it carries its own workspace, which may legitimately differ from the source's.
/// `token_id` is never inherited (it fingerprints the bearer token that created a session, so the
/// forking caller's token is the only correct value); `None` simply leaves it NULL.
#[derive(Debug, Default, Clone)]
pub(crate) struct ForkOverrides {
    pub(crate) cwd: Option<std::path::PathBuf>,
    pub(crate) additional_roots: Option<Vec<PathBuf>>,
    pub(crate) token_id: Option<String>,
}
/// One session's worth of data for [`Store::import_sessions`]. IDs are already freshly
/// minted and parent links remapped by the caller; the records must be ordered parents-first so
/// the `parent_session_id` foreign key is satisfied on insert.
pub(crate) struct ImportSessionRecord {
    pub(crate) new_id: Uuid,
    pub(crate) new_parent_id: Option<Uuid>,
    pub(crate) created_at: String,
    pub(crate) cwd: Option<PathBuf>,
    /// The level the row starts at; an archive that recorded none is given the config default by
    /// `plan_import` before it reaches here.
    pub(crate) permission: Permission,
    /// Whether calls above the level are submitted for approval.
    pub(crate) approvals: bool,
    pub(crate) capabilities_json: Option<String>,
    /// Workspace roots beyond `cwd`, carried across an export/import round trip. Defaults to empty
    /// for exports written before the field existed.
    pub(crate) additional_roots: Vec<PathBuf>,
    /// A sub-agent's spawn terms, carried so an imported sub-agent is still followable. `None` for
    /// root sessions and for archives written before the field existed; an imported sub-agent
    /// without it can be read and deleted but not resumed.
    pub(crate) subagent_spec_json: Option<String>,
    /// What the imported session runs on. The caller settles this: an archive that carries a
    /// profile keeps it, and one written before the field existed adopts the importing
    /// installation's default, which is the only thing that can be known about it here.
    pub(crate) profile: String,
    pub(crate) stats: crate::stats::SessionStatsSnapshot,
    /// `(created_at, event)` pairs in chronological order; timestamps are preserved verbatim.
    pub(crate) events: Vec<(String, crate::conversation::Event)>,
    /// `(name, content)` scratchpad entries referenced by name from tool-call inputs.
    pub(crate) tool_outputs: Vec<(String, String)>,
}
/// A row's evidence that another session spawned it. See [`Store::spawn_terms`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct SpawnTerms {
    /// The session that spawned it, when that row is in this store.
    ///
    /// `None` for a sub-agent whose parent link did not survive an export and re-import. The
    /// conversation is still a sub-agent's, which is what makes this an `Option` inside a type
    /// that only exists when the answer is yes, rather than the answer itself.
    pub(crate) parent: Option<Uuid>,
}
/// "Is this row a sub-agent's conversation?", for a `WHERE` clause.
///
/// The SQL half of [`Store::spawn_terms`]; see that function for why both columns are
/// read. `qualifier` is the table alias with its dot (`"s."`) or empty for an unaliased query.
/// Interpolated rather than parameterized because it names columns, not values, and takes the
/// alias as an argument rather than being a `const` so a joined query cannot silently pick the
/// wrong table's columns.
pub(crate) fn spawned_session_sql(qualifier: &str) -> String {
    format!(
        "({qualifier}parent_session_id IS NOT NULL OR {qualifier}subagent_spec_json IS NOT NULL)"
    )
}
#[derive(Debug, Clone)]
pub(crate) struct SessionSummary {
    pub(crate) id: Uuid,
    /// RFC 3339 timestamp the session row was first written. Surfaced alongside `updated_at`
    /// so re-attach can restore the original creation time rather than stamping a fresh
    /// `Utc::now()` on every reconstruction.
    pub(crate) created_at: String,
    pub(crate) updated_at: String,
    /// [`crate::conversation::Conversation::title`] of the session, from its first user message.
    pub(crate) title: String,
    /// Working directory captured at session creation. `None` when an archive omitted one and
    /// [`Store::import_sessions`] stored that absence verbatim; ACP-facing code falls
    /// back to the process cwd for display.
    pub(crate) cwd: Option<std::path::PathBuf>,
    /// The level the session runs at: written by every door that creates a row and by every
    /// surface that moves it, so a resume and a scheduled gate read the same answer. Parsed once,
    /// here, so no reader handles the column as text. `None` for a row no host wrote, or one whose
    /// value meka cannot read (warned about at the read), which the scheduler treats as a level at
    /// which nothing runs and a host resumes at its configured default.
    pub(crate) permission: Option<Permission>,
    /// Whether a call above the level is submitted for approval rather than refused. Recorded
    /// beside the level, for the same readers.
    pub(crate) approvals: bool,
    /// The name of the profile this session runs on, and nothing else: a profile is an
    /// indivisible bundle, so the name is the whole binding.
    pub(crate) profile: String,
    /// Per-session capability flags, as a serialized
    /// [`crate::host::http::http_frontend::SessionCapabilities`]. Deliberately not enumerated
    /// here: the flag set has grown twice, and each restatement went stale silently. NULL for
    /// every session the HTTP API did not create.
    pub(crate) capabilities_json: Option<String>,
    /// Workspace roots beyond `cwd`, from an ACP client's `additionalDirectories`. Empty whenever
    /// a session carries no extra roots, which is every non-ACP session.
    pub(crate) additional_roots: Vec<PathBuf>,
    /// SHA-256 fingerprint of the bearer token that created this session. `None` for every session
    /// not created via the HTTP API, including sub-agents, whose row omits the column entirely. A
    /// fork through the HTTP API does carry one: the token doing the forking, never the source's.
    pub(crate) token_id: Option<String>,
    /// The session this one was spawned from, for a sub-agent. `None` for a root session.
    /// Surfaced so a client listing with `include_children` can rebuild the spawn tree rather than
    /// receiving a flat list in which a sub-agent is indistinguishable from the agent that
    /// dispatched it.
    pub(crate) parent_id: Option<Uuid>,
}
/// What a sweep over many sessions did, so its caller can say what it left behind.
///
/// A bare count reads as "everything that matched was deleted", which makes the retention sweep
/// destructive: it announces `deleted 1 session(s)` in an unrelated terminal and says nothing about
/// the conversation an operator has open in another one. A sweep that spares something has to be
/// able to say so.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SessionSweep {
    pub(crate) deleted: u64,
    /// Sessions that matched but were left alone, because another meka process has them open or
    /// because their lock could not be established either way.
    pub(crate) attached_elsewhere: u64,
}
/// What a host may move on a session's row once it exists, for [`Store::update_session`]. `None`
/// leaves that column alone.
///
/// One shape for every writer, so the `updated_at` rule is stated once rather than once per
/// column. The profile is a name and nothing else, because a profile is an indivisible bundle and
/// the name is the whole binding.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct SessionPatch {
    pub(crate) permission: Option<Permission>,
    pub(crate) approvals: Option<bool>,
    pub(crate) cwd: Option<PathBuf>,
    pub(crate) profile: Option<String>,
    /// The complete resulting list of roots beyond `cwd`, so `Some(vec![])` clears the column.
    /// Per the ACP spec a non-empty `additionalDirectories` "is the complete resulting
    /// additional-root list", while omitted or empty means none are activated.
    pub(crate) roots: Option<Vec<PathBuf>>,
}
impl SessionPatch {
    /// Whether the patch moves nothing.
    pub(crate) fn is_empty(&self) -> bool {
        self.permission.is_none()
            && self.approvals.is_none()
            && self.cwd.is_none()
            && self.profile.is_none()
            && self.roots.is_none()
    }
}
impl std::fmt::Display for SessionPatch {
    /// The columns the patch names, for a message about a write that failed.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut parts = Vec::new();
        if let Some(level) = self.permission {
            parts.push(format!("permission '{level}'"));
        }
        if let Some(approvals) = self.approvals {
            parts.push(format!(
                "approvals {}",
                if approvals { "on" } else { "off" }
            ));
        }
        if let Some(cwd) = &self.cwd {
            parts.push(format!("working directory '{}'", cwd.display()));
        }
        if let Some(profile) = &self.profile {
            parts.push(format!("profile '{profile}'"));
        }
        if let Some(roots) = &self.roots {
            parts.push(format!("{} additional root(s)", roots.len()));
        }
        if parts.is_empty() {
            return formatter.write_str("nothing");
        }
        formatter.write_str(&parts.join(", "))
    }
}
/// Whether the caller of [`Store::fork_session_locked`] already holds the source session.
///
/// A copy of a conversation being written ends on a user message nothing answered, because a turn
/// persists its prompt before the provider replies, so the source has to stand still for the copy
/// and another process holding it is refused. `flock` is per open file description, though: a
/// probe from the very process that holds the source would refuse that process its own session. A
/// door whose host has the source open says so here instead of probing, and answers for it
/// standing still.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceLock {
    /// Nothing in this process holds the source: probe it, and refuse [`MekaError::SessionLocked`]
    /// when another process does.
    Probe,
    /// This process holds the source: the REPL's own session, or one resident in `serve` or `acp`.
    HeldByCaller,
}
/// Sessions a scheduled job still depends on, as a `WHERE` fragment over `sessions`.
///
/// A session with a job ahead of it is *not* expired, whatever `updated_at` says. Only turns bump
/// that column ([`ScheduleStore::claim_occurrence`] and [`ScheduleStore::complete_claim`] touch
/// `scheduled_jobs` alone), so a gated watcher that evaluates every tick and rarely fires looks
/// untouched for exactly as long as it is working. The cascade would then take the job with the
/// session, and the sweep report `deleted 1 session(s)` without ever mentioning that a schedule
/// went with it.
///
/// Sparing only the row that *owns* the job is not enough. `parent_session_id` carries
/// `ON DELETE CASCADE`, so deleting a stale parent silently takes its sub-agent children, and a
/// job created against a child (reachable over HTTP, whose only gate is that the session exists)
/// goes with them. The recursive term walks parent links up from every job-owning session and
/// spares that whole chain.
///
/// A constant because it is applied twice, in two statements, and the pair is only sound while
/// they agree: see [`Store::delete_the_unattached_among`].
///
/// [`ScheduleStore::claim_occurrence`]: crate::store::schedule::ScheduleStore::claim_occurrence
/// [`ScheduleStore::complete_claim`]: crate::store::schedule::ScheduleStore::complete_claim
pub(super) const NOT_SPOKEN_FOR_BY_A_SCHEDULE: &str = "id NOT IN (SELECT session_id FROM scheduled_jobs) \
     AND id NOT IN ( \
         WITH RECURSIVE ancestors(id) AS ( \
             SELECT parent_session_id FROM sessions \
               WHERE parent_session_id IS NOT NULL \
                 AND id IN (SELECT session_id FROM scheduled_jobs) \
             UNION \
             SELECT s.parent_session_id FROM sessions s \
               JOIN ancestors a ON s.id = a.id \
              WHERE s.parent_session_id IS NOT NULL \
         ) \
         SELECT id FROM ancestors \
     )";
/// How many sessions a prefix scan fetches before it stops counting.
///
/// A resolution needs at most two: one match resolves, and any second makes it ambiguous. The rest
/// exist so the refusal can name what collided. Reported as "at least" when the scan hits it, since
/// past this point the count is the cap rather than the truth.
pub(crate) const PREFIX_MATCH_CAP: usize = 17;
/// Pseudo-role written to the `messages` table's `role` column for `Event::CompactBoundary` rows,
/// distinct from every role an `Event::Append` uses.
pub(super) const COMPACT_BOUNDARY_ROLE: &str = "compact_boundary";
/// Pseudo-role for a `Role::User` message that carries non-text blocks: every turn's message, whose
/// context block comes first, and any input images. Its full `Vec<ContentBlock>` is stored as JSON,
/// because flattening to `text_content()` (as the plain `user` role does) would drop them. A
/// text-only user message, which is one meka authored (a nudge, a summary), stays plaintext under
/// `user`. A `role`-column pseudo-role, mirroring [`COMPACT_BOUNDARY_ROLE`].
pub(super) const USER_BLOCKS_ROLE: &str = "user_blocks";
/// Pseudo-role for `Event::Repair` rows, mirroring [`COMPACT_BOUNDARY_ROLE`]. The superseded
/// messages keep their own rows, so `meka session export` still shows what was replaced.
pub(super) const REPAIR_ROLE: &str = "repair";
/// Pseudo-role for `Event::Redact` rows, mirroring [`REPAIR_ROLE`]: the images it names keep
/// their rows, and the replay puts the placeholder over them.
pub(super) const REDACT_ROLE: &str = "redact";
/// The `messages` row a session's title is read from, as a `WHERE` over `messages` correlated to
/// `sessions s`: the first user message that carries words. A `user` row is text by construction; a
/// `user_blocks` row is a JSON array of blocks and is passed over while none of them is a `text`
/// block, so an image sent alone does not leave the session unlabeled once words follow. Blank
/// text and a stand-in meka wrote do not count as words in either shape, which is
/// [`crate::conversation::is_harness_stand_in`] spelled in SQL. A compaction summary is under its
/// own pseudo-role and is never selected. This is [`crate::conversation::Conversation::title`]'s
/// rule over the log, and [`title_of_first_user_row`] then applies that function to the row it
/// selects, so the two cannot pick different rows.
static TITLE_ROW_WHERE_SQL: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    let quoted = |text: &str| format!("'{}'", text.replace('\'', "''"));
    // What `Conversation::title` takes as words, over one SQL expression holding the text.
    let is_words = |text: &str| {
        format!(
            "(trim({text}, ' ' || char(9, 10, 11, 12, 13)) <> '' AND {text} <> {placeholder} \
             AND substr({text}, 1, {note_chars}) <> {note})",
            placeholder = quoted(crate::conversation::IMAGE_REDACTION_PLACEHOLDER),
            note_chars = crate::conversation::HARNESS_NOTE.chars().count(),
            note = quoted(crate::conversation::HARNESS_NOTE),
        )
    };
    format!(
        "session_id = s.id
         AND ((role = 'user' AND {user_words})
              OR (role = 'user_blocks'
                  AND EXISTS (SELECT 1 FROM json_each(messages.content)
                              WHERE json_extract(json_each.value, '$.type') = 'text'
                                AND {block_words})))",
        user_words = is_words("messages.content"),
        block_words = is_words("json_extract(json_each.value, '$.text')"),
    )
});
/// A session's title from the row [`TITLE_ROW_WHERE_SQL`] selects, through the one definition in
/// [`crate::conversation::Conversation::title`]. A `user_blocks` row holds the message's blocks as
/// JSON; a `user` row is the text itself.
pub(super) fn title_of_first_user_row(session: &str, role: &str, content: String) -> String {
    use crate::conversation::{ContentBlock, Conversation, Event, Message, Role};
    let message = if role == USER_BLOCKS_ROLE {
        match serde_json::from_str::<Vec<ContentBlock>>(&content) {
            Ok(blocks) => Message {
                role: Role::User,
                content: blocks,
            },
            Err(error) => {
                tracing::warn!(
                    "failed to decode the first user message of session {session} for its title: \
                     {error}"
                );
                return String::new();
            }
        }
    } else {
        Message::user(content)
    };
    Conversation::from_events(vec![Event::Append(message)]).title()
}
/// Encode an [`crate::conversation::Event`] into the `(role, content)` columns of the `messages`
/// table. `Event::Append` writes the message's natural role; `Event::CompactBoundary`,
/// `Event::Repair` and `Event::Redact` write a JSON envelope under their pseudo-role.
pub(super) fn encode_event_for_db(
    event: &crate::conversation::Event,
) -> std::result::Result<(String, String), serde_json::Error> {
    use crate::conversation::{ContentBlock, Event, Role};

    match event {
        Event::Append(message) => {
            let (role, content) = match message.role {
                Role::User => {
                    if message
                        .content
                        .iter()
                        .any(|block| matches!(block, ContentBlock::ToolResult { .. }))
                    {
                        ("tool_results", serde_json::to_string(&message.content)?)
                    } else if message
                        .content
                        .iter()
                        .any(|block| !matches!(block, ContentBlock::Text { .. }))
                    {
                        // A user turn carrying non-text blocks (input images) can't be flattened to
                        // plain text without losing them, so persist the full block list as JSON.
                        (USER_BLOCKS_ROLE, serde_json::to_string(&message.content)?)
                    } else {
                        ("user", message.text_content())
                    }
                }
                Role::Assistant => ("assistant", serde_json::to_string(&message.content)?),
            };
            Ok((role.to_string(), content))
        }
        Event::CompactBoundary { .. } => {
            let content = serde_json::to_string(event)?;
            Ok((COMPACT_BOUNDARY_ROLE.to_string(), content))
        }
        Event::Repair { .. } => {
            let content = serde_json::to_string(event)?;
            Ok((REPAIR_ROLE.to_string(), content))
        }
        Event::Redact { .. } => {
            let content = serde_json::to_string(event)?;
            Ok((REDACT_ROLE.to_string(), content))
        }
    }
}
/// Decode one persisted row back into an [`crate::conversation::Event`]. Returns `Ok(None)` when
/// the row's role is unrecognized (forward compatibility for new variants).
pub(super) fn decode_event_from_row(
    row: &StoredMessage,
) -> std::result::Result<Option<crate::conversation::Event>, serde_json::Error> {
    use crate::conversation::{ContentBlock, Event, Message, Role};

    match row.role.as_str() {
        "user" => Ok(Some(Event::Append(Message::user(&row.content)))),
        "assistant" => match serde_json::from_str::<Vec<ContentBlock>>(&row.content) {
            Ok(content) => Ok(Some(Event::Append(Message {
                role: Role::Assistant,
                content,
            }))),
            Err(_) => {
                // Deliberately softer than the `tool_results` and `user_blocks` arms below, which
                // drop an unparseable row. Corruption here costs the turn's `tool_use` blocks, and
                // dropping the row entirely would take the assistant's prose with them. Keeping the
                // raw string as text leaves the transcript readable and the turn boundary intact.
                Ok(Some(Event::Append(Message::assistant_text(&row.content))))
            }
        },
        "tool_results" => match serde_json::from_str::<Vec<ContentBlock>>(&row.content) {
            Ok(content) => Ok(Some(Event::Append(Message {
                role: Role::User,
                content,
            }))),
            Err(error) => Err(error),
        },
        role if role == USER_BLOCKS_ROLE => {
            match serde_json::from_str::<Vec<ContentBlock>>(&row.content) {
                Ok(content) => Ok(Some(Event::Append(Message {
                    role: Role::User,
                    content,
                }))),
                Err(error) => Err(error),
            }
        }
        role if role == COMPACT_BOUNDARY_ROLE || role == REPAIR_ROLE || role == REDACT_ROLE => {
            let event: Event = serde_json::from_str(&row.content)?;
            Ok(Some(event))
        }
        _ => Ok(None),
    }
}
/// Pagination cursor for [`Store::list_sessions`]: encodes the `(updated_at, id)` of the
/// last row in a page as base64-url JSON. The shape is opaque to clients; they only round-trip it
/// back as `next_cursor`.
#[derive(Serialize, Deserialize)]
pub(super) struct ListSessionsCursor {
    #[serde(rename = "u")]
    pub(super) updated_at: String,
    #[serde(rename = "i")]
    pub(super) id: String,
}
/// A page's last `(updated_at, id)` as the opaque token a client hands back.
pub(super) fn encode_list_cursor(updated_at: &str, id: &str) -> String {
    use base64::Engine;
    let payload = ListSessionsCursor {
        updated_at: updated_at.to_string(),
        id: id.to_string(),
    };
    #[allow(
        clippy::expect_used,
        reason = "`ListSessionsCursor` is two owned `String`s, which `serde_json::to_vec` cannot fail on"
    )]
    let json = serde_json::to_vec(&payload)
        .expect("ListSessionsCursor is two owned Strings; serialization cannot fail");
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json)
}
/// The `(updated_at, id)` a client's token encodes, refused when it is not one this made.
pub(super) fn decode_list_cursor(token: &str) -> Result<(String, String)> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(token)
        .map_err(|error| MekaError::Database(format!("invalid list cursor: {error}")))?;
    let cursor: ListSessionsCursor = serde_json::from_slice(&bytes)
        .map_err(|error| MekaError::Database(format!("invalid list cursor: {error}")))?;
    Ok((cursor.updated_at, cursor.id))
}
/// Encode an additional-root list for the `additional_roots_json` column. `None` for the empty case
/// keeps "no extra roots" as NULL, so the column has one representation for one meaning rather than
/// NULL and `[]` both appearing.
pub(super) fn encode_additional_roots(roots: &[PathBuf]) -> Result<Option<String>> {
    if roots.is_empty() {
        return Ok(None);
    }
    serde_json::to_string(roots)
        .map(Some)
        .map_err(|error| MekaError::Database(format!("failed to encode additional roots: {error}")))
}
/// Decode the persisted `additional_roots_json` column into workspace roots.
///
/// NULL (every session that never carried extra roots) and unparseable JSON both yield an empty
/// list. Failing soft is right here: a session whose root list can't be read is still perfectly
/// usable as a single-root session, and refusing to load it would be a far worse outcome than
/// silently narrowing its search scope.
pub(super) fn decode_additional_roots(json: Option<&str>) -> Vec<PathBuf> {
    json.and_then(|raw| serde_json::from_str::<Vec<PathBuf>>(raw).ok())
        .unwrap_or_default()
}
impl Store {
    /// The key under which a sub-agent's spawn terms record the profile the spawn call chose. The
    /// spec is the sub-agent tool's document and the store never reads it, except to move this
    /// one name when the profile is renamed; a test beside the spec pins the key to this.
    pub(crate) const SUBAGENT_SPEC_PROFILE_KEY: &'static str = "profile";

    /// Create a new session, optionally recording its working directory. `cwd` is persisted as an
    /// absolute path string; pass `None` only for code paths that genuinely have no cwd context.
    ///
    /// Leaves the row unlocked, so a sweep can reach it before anyone claims it. Every host that
    /// creates a session a turn will run against wants [`Self::create_session_locked`] instead;
    /// this one is reached from tests.
    #[cfg(test)]
    pub(crate) async fn create_session(
        &self,
        cwd: Option<std::path::PathBuf>,
        profile: impl Into<String>,
    ) -> Result<Uuid> {
        self.create_session_with_metadata(
            cwd,
            crate::permission::Permission::Read.to_string(),
            false,
            None,
            None,
            profile,
        )
        .await
        .map(|created| created.id)
    }

    /// Like [`Self::create_session`] but also persists the HTTP API's per-session metadata
    /// (`permission` level, `capabilities_json` blob, and `token_id` fingerprint). The REPL
    /// and ACP paths derive permission from process config and don't have a bearer token.
    ///
    /// No production caller: every host that creates a session now goes through
    /// [`Self::create_session_locked`], which takes the lock before the row exists. This and
    /// [`Self::create_session`] remain as the unlocked doors, reached from tests and from callers
    /// that genuinely want a row nobody is holding.
    #[cfg(test)]
    pub(crate) async fn create_session_with_metadata(
        &self,
        cwd: Option<std::path::PathBuf>,
        permission: String,
        approvals: bool,
        capabilities_json: Option<String>,
        token_id: Option<String>,
        profile: impl Into<String>,
    ) -> Result<CreatedSession> {
        self.insert_session_row(
            Uuid::new_v4(),
            cwd,
            permission,
            approvals,
            capabilities_json,
            token_id,
            profile.into(),
        )
        .await
    }

    /// Create a session and take its lock, in that order: the lock **before** the row.
    ///
    /// The ordering is the entire point. Committing the row first leaves a window (microseconds
    /// wide, but real) in which the session is visible to `SELECT id FROM sessions` and held by
    /// nobody. [`Self::delete_all_sessions`] enumerates at delete time, so it lands inside that
    /// window, takes the lock legitimately, and cascades the conversation away underneath the
    /// process creating it: the turn ends `FOREIGN KEY constraint failed` with the user's prompt
    /// gone. A targeted `meka session delete <id>` cannot, because its id list is gathered before
    /// the creator exists, which is what identifies the window as belonging to creation rather
    /// than to deletion.
    ///
    /// Locking first closes it with nothing left over: a sweeper either cannot see the row yet, or
    /// sees it and finds the lock held. A lock file whose row never lands is swept by
    /// [`Self::prune_orphan_lock_files`] like any other orphan.
    ///
    /// An `Err` in the second half means the claim could not be *made* (an unwritable lock
    /// directory, descriptors exhausted) and never that somebody else holds it, because no other
    /// process can know this id yet. It is returned rather than logged-and-dropped so a caller that
    /// refuses can report the reason it actually hit. Callers differ on what it is worth: a host
    /// that must be alone refuses, and the agent's own path warns and runs the turn regardless
    /// rather than breaking installations that work today.
    pub(crate) async fn create_session_locked(
        &self,
        cwd: Option<std::path::PathBuf>,
        permission: String,
        approvals: bool,
        capabilities_json: Option<String>,
        token_id: Option<String>,
        profile: impl Into<String>,
    ) -> Result<(CreatedSession, std::result::Result<FileLock, MekaError>)> {
        let session_id = Uuid::new_v4();
        let lock = self.claim_a_fresh_id(session_id);
        // A failed insert leaves the claim's file in place here, unlike the two sibling doors that
        // call `discard_unused_claim`: `a_session_is_locked_before_its_row_is_written` reads that
        // leftover as its proof that the lock came first, and the sweep at `open()` collects it.
        let created = self
            .insert_session_row(
                session_id,
                cwd,
                permission,
                approvals,
                capabilities_json,
                token_id,
                profile.into(),
            )
            .await?;
        Ok((created, lock))
    }

    /// Drop a claim on an id that has no row, file included.
    ///
    /// Nothing is written under it, so the file the claim created is garbage the moment it exists.
    /// Left behind it accumulates once per failed insert or per lookup of an id a client can name,
    /// and the sweep that would collect it runs only at `open()` and after a delete. Nothing else
    /// can be holding it: the id is a fresh v4, or one this process has just locked itself. One
    /// definition for the doors that mint an id ahead of its row, and for the ones that lock an id
    /// and then find no row behind it.
    fn discard_unused_claim(&self, lock: std::result::Result<FileLock, MekaError>, id: Uuid) {
        drop(lock);
        let path = self.lock_dir.join(format!("{id}.lock"));
        // A file already gone is the outcome wanted: a sweep between the drop and here took it.
        if let Err(error) = std::fs::remove_file(&path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!("failed to remove '{path}': {error}", path = path.display());
        }
    }

    /// How many sessions run on one profile.
    ///
    /// For `meka profile remove`, which otherwise strands them silently: the refusal only arrives
    /// when the user next resumes one, which can be long after the removal and somewhere else.
    pub(crate) async fn count_sessions_on_profile(&self, profile: &str) -> Result<u64> {
        let profile = profile.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                // Root sessions only, matching what `meka session list` shows and what the
                // warning's own advice can act on. A sub-agent row copies its parent's binding,
                // so counting children would report a number many times what the user can see,
                // about rows that `meka -r <id> --profile <name>` is not for.
                //
                // [`spawned_session_sql`], so "root" means here what it means in that listing.
                // An imported sub-agent has no parent link, so keying on the link alone would
                // count a row the user cannot see and cannot repin, in the one place whose whole
                // job is to state a number the user is about to act on.
                connection.query_row(
                    &format!(
                        "SELECT COUNT(*) FROM sessions
                         WHERE profile = ?1 AND NOT {}",
                        spawned_session_sql("")
                    ),
                    rusqlite::params![profile],
                    |row| row.get::<_, i64>(0),
                )
            })
            .await
            .map(|count| count.max(0) as u64)
            .map_err(|error| {
                MekaError::Database(format!("failed to count sessions on a profile: {error}"))
            })
    }

    /// Move every session on `profile` to `new_profile`, the pinned spawn terms of a sub-agent
    /// included, and answer how many rows moved.
    ///
    /// One transaction, because a follow-up runs a pinned worker on its row while `session show`
    /// and an export read the spec: rows moved without specs would show a worker pinned to a
    /// name nothing is configured under.
    pub(crate) async fn rename_profile(&self, profile: &str, new_profile: &str) -> Result<u64> {
        let profile = profile.to_string();
        let new_profile = new_profile.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let transaction = connection.transaction()?;
                let moved = transaction.execute(
                    "UPDATE sessions SET profile = ?2 WHERE profile = ?1",
                    rusqlite::params![profile, new_profile],
                )?;
                // `json_valid` first: `json_extract` raises on a spec that is not JSON, which an
                // import writes verbatim from its archive, and one such row must not block every
                // rename. It is left as it is, like any spec the rename has no name to move.
                transaction.execute(
                    &format!(
                        "UPDATE sessions
                         SET subagent_spec_json = json_set(subagent_spec_json, '$.{key}', ?2)
                         WHERE json_valid(subagent_spec_json)
                           AND json_extract(subagent_spec_json, '$.{key}') = ?1",
                        key = Self::SUBAGENT_SPEC_PROFILE_KEY
                    ),
                    rusqlite::params![profile, new_profile],
                )?;
                transaction.commit()?;
                Ok(moved as u64)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to rename a profile: {error}")))
    }

    /// How many rows record `profile` anywhere: on the row, root or sub-agent, or in pinned spawn
    /// terms. What a rename onto this name would otherwise adopt, and what its undo would then
    /// carry away with the rows it meant to put back.
    pub(crate) async fn count_sessions_recording_profile(&self, profile: &str) -> Result<u64> {
        let profile = profile.to_string();
        self.connection
            .call(move |connection| {
                connection.query_row(
                    &format!(
                        "SELECT COUNT(*) FROM sessions
                         WHERE profile = ?1
                            OR (json_valid(subagent_spec_json)
                                AND json_extract(subagent_spec_json, '$.{key}') = ?1)",
                        key = Self::SUBAGENT_SPEC_PROFILE_KEY
                    ),
                    rusqlite::params![profile],
                    |row| row.get::<_, i64>(0),
                )
            })
            .await
            .map(|count| count.max(0) as u64)
            .map_err(|error| {
                MekaError::Database(format!(
                    "failed to count sessions recording a profile: {error}"
                ))
            })
    }

    /// What this session runs on, as recorded on its row.
    ///
    /// `None` means the session is gone. Every other answer is a profile name that resolved when it
    /// was written: every door that mints a session resolves one first and refuses without it, so
    /// there is no such thing as a session this meka created with none. The empty string is still
    /// reachable, on a carried-forward row the migration could resolve no profile for, and needs no
    /// branch here: like a name that has since left `config.toml`, it is refused by
    /// [`crate::provider::ProviderRegistry::settings`], by name.
    pub(crate) async fn recorded_profile(&self, session_id: Uuid) -> Result<Option<String>> {
        let id = session_id.to_string();
        self.connection
            .call(move |connection| {
                connection
                    .query_row("SELECT profile FROM sessions WHERE id = ?1", [&id], |row| {
                        row.get::<_, String>(0)
                    })
                    .optional()
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to read the session's profile: {error}"))
            })
    }

    /// Take the lock on an id that is about to become a row.
    ///
    /// The ordering rule in one place, because four doors mint a session and each has to keep it.
    /// Committed and then claimed, a row is visible to `SELECT id FROM sessions` and held by
    /// nobody, which a concurrent [`Self::delete_all_sessions`] enumerates, locks and cascades
    /// away underneath its creator.
    ///
    /// The `Err` is carried rather than logged and dropped so a caller that refuses can name the
    /// reason it hit. It never means "somebody else holds it": no other process can know this id.
    pub(super) fn claim_a_fresh_id(
        &self,
        session_id: Uuid,
    ) -> std::result::Result<FileLock, MekaError> {
        self.lock_session(session_id).inspect_err(|error| {
            tracing::warn!(
                "failed to lock session {session_id} as it was created: {error}; another meka process could \
                 attach to it or sweep it mid-turn"
            );
        })
    }

    /// The row half of session creation, shared by the locked and unlocked doors so the columns
    /// are written in one place.
    ///
    /// One parameter per column the doors decide, in column order; a struct here would carry the
    /// same seven names one step upstream and add nothing but a second place to keep them in order.
    /// The level is a `String` and not an `Option`: every door writes one, so the scheduler can
    /// read the row alone, and the type is what keeps the next door from forgetting.
    #[allow(
        clippy::too_many_arguments,
        reason = "one parameter per column the doors decide; a struct would carry the same names one step upstream"
    )]
    pub(super) async fn insert_session_row(
        &self,
        session_id: Uuid,
        cwd: Option<std::path::PathBuf>,
        permission: String,
        approvals: bool,
        capabilities_json: Option<String>,
        token_id: Option<String>,
        profile: String,
    ) -> Result<CreatedSession> {
        let created_at = chrono::Utc::now().to_rfc3339();
        let cwd_string = cwd.map(|path| path.display().to_string());

        let created_at_for_db = created_at.clone();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "INSERT INTO sessions (id, created_at, updated_at, cwd, permission, approvals, \
                     capabilities_json, token_id, profile)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                    rusqlite::params![
                        session_id.to_string(),
                        created_at_for_db,
                        created_at_for_db,
                        cwd_string,
                        permission,
                        approvals,
                        capabilities_json,
                        token_id,
                        profile,
                    ],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to create session: {error}")))?;

        Ok(CreatedSession {
            id: session_id,
            created_at,
        })
    }

    /// Create a session whose `parent_session_id` references an existing session, used by
    /// `agent_spawn` so sub-agent conversations persist as children of the parent for auditing.
    /// Cascades on parent delete (see [`Self::delete_session`]). The optional `cwd` is the parent's
    /// cwd snapshot at spawn time, or the first of the sub-agent's `writable_roots`, and
    /// `additional_roots` the rest of them: empty for a sub-agent sharing its parent's workspace.
    ///
    /// `subagent_spec_json` records the terms the sub-agent was spawned under so `agent_followup`
    /// can rebuild it from them rather than from whatever the parent looks like at follow-up
    /// time. It is written with the row rather than updated afterwards, as the roots are: a spawn
    /// that fails between the two would otherwise leave a child that can be followed up on with
    /// no recorded terms.
    pub(crate) async fn create_child_session(
        &self,
        parent: Uuid,
        cwd: Option<std::path::PathBuf>,
        additional_roots: Vec<PathBuf>,
        subagent_spec_json: Option<String>,
        // The level the sub-agent runs at, already clamped against its parent, so the row answers
        // for a sub-agent wherever a row is read.
        permission: String,
        // The profile the sub-agent will actually be built on, which is the parent *agent's* live
        // binding and not necessarily the parent *row's*. The two differ for exactly as long as a
        // repin that could not take the runtime lock: ACP's `session/set_config_option` moves the
        // row mid-turn, `try_lock` fails, and the agent stays where it was until the next turn.
        // Selecting the column here would record the profile the sub-agent is not running on, so
        // a later `agent_followup` on that child would resolve a different account from the one
        // that did the work. Passed in for the same reason the window is: everything about a
        // sub-agent's binding comes off one cell.
        profile: String,
    ) -> Result<(Uuid, std::result::Result<FileLock, MekaError>)> {
        let session_id = Uuid::new_v4();
        // Locked before the row, like every other door that mints one, and the exposure here is
        // not the microsecond window the others have: an unlocked sub-agent row sits claimable for
        // the whole of the sub-agent's run, which is seconds to minutes. A concurrent
        // `meka session delete --all` enumerates it, takes the lock nobody holds, and cascades it
        // away; the sub-agent's next message insert then dies on `FOREIGN KEY constraint failed`
        // with its work gone.
        let lock = self.claim_a_fresh_id(session_id);
        let now = chrono::Utc::now().to_rfc3339();
        let cwd_string = cwd.map(|path| path.display().to_string());
        let additional_roots_json = match encode_additional_roots(&additional_roots) {
            Ok(json) => json,
            Err(error) => {
                self.discard_unused_claim(lock, session_id);
                return Err(error);
            }
        };

        let inserted = self
            .connection
            .call(move |connection| -> rusqlite::Result<bool> {
                let rows = connection.execute(
                    // Still `SELECT … FROM sessions WHERE id = ?4` rather than a plain `VALUES`,
                    // because a parent that is gone must select no row and insert nothing. Only
                    // the provider stopped being read off that row; see the parameter's note.
                    // `approvals` is still read off it: a sub-agent shares its parent's switch.
                    "INSERT INTO sessions
                         (id, created_at, updated_at, parent_session_id, cwd,
                          additional_roots_json, subagent_spec_json, permission, approvals,
                          profile)
                     SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, approvals, ?9
                     FROM sessions WHERE id = ?4",
                    rusqlite::params![
                        session_id.to_string(),
                        now,
                        now,
                        parent.to_string(),
                        cwd_string,
                        additional_roots_json,
                        subagent_spec_json,
                        permission,
                        profile,
                    ],
                )?;
                Ok(rows > 0)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to create child session: {error}"))
            })?;

        // A parent that is gone selects no row, so this statement inserts nothing and succeeds.
        // Reading the count back is what keeps that an error, where a plain `VALUES` form would be
        // refused by `parent_session_id`'s foreign key: without the check a spawn would hand back
        // an id with no row behind it, a sub-agent the model is told about, holding a lock file
        // for a session that never existed, whose first `save_message` dies on the constraint
        // instead. [`Self::fork_session_into`] reads its own count for the same reason.
        if !inserted {
            self.discard_unused_claim(lock, session_id);
            return Err(MekaError::Database(format!(
                "cannot spawn a sub-agent of session {parent}: it no longer exists"
            )));
        }

        Ok((session_id, lock))
    }

    /// Whether a row is a sub-agent's conversation, and its parent when the link survived.
    ///
    /// The one answer to that question for every caller that needs it in Rust;
    /// [`spawned_session_sql`] is the same rule for the callers that need it in a `WHERE` clause.
    /// `Ok(None)` means a root session, and also means "no such row", which every caller here
    /// already answers for separately.
    ///
    /// **Both columns, because either alone is wrong.** A sub-agent normally carries a parent link
    /// and a spec together, but `session export` on a sub-agent alone emits a `parent_id` pointing
    /// outside the archive, and `import_sessions` resolves that to `NULL` while copying
    /// `subagent_spec_json` faithfully: a real row that is a sub-agent's conversation with no
    /// parent in this store. [`crate::host::refuse_a_spawned_session`] reads it that way and
    /// refuses it, so every rule *around* that refusal has to agree, or `meka -c` selects an
    /// orphan it cannot then drive, `session import` prints a `meka -r` that is already
    /// illegal, and `POST /schedule` accepts a job that fails on every fire.
    pub(crate) async fn spawn_terms(&self, session_id: Uuid) -> Result<Option<SpawnTerms>> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection
                    .query_row(
                        "SELECT parent_session_id, subagent_spec_json FROM sessions WHERE id = ?1",
                        rusqlite::params![session_id.to_string()],
                        |row| {
                            Ok((
                                row.get::<_, Option<String>>(0)?,
                                row.get::<_, Option<String>>(1)?,
                            ))
                        },
                    )
                    .optional()
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to read session spawn terms: {error}"))
            })
            .map(|row| {
                let (parent, spec) = row?;
                if parent.is_none() && spec.is_none() {
                    return None;
                }
                Some(SpawnTerms {
                    parent: parent.as_deref().and_then(|id| Uuid::parse_str(id).ok()),
                })
            })
    }

    /// The recorded spawn terms for a sub-agent session, or `None` for a root session.
    pub(crate) async fn load_subagent_spec(&self, session_id: Uuid) -> Result<Option<String>> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection
                    .query_row(
                        "SELECT subagent_spec_json FROM sessions WHERE id = ?1",
                        rusqlite::params![session_id.to_string()],
                        |row| row.get::<_, Option<String>>(0),
                    )
                    .optional()
                    .map(Option::flatten)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to load sub-agent spec: {error}")))
    }

    /// Copy `source`'s conversation into a new session and return it, or `Ok(None)` when `source`
    /// doesn't exist (callers map that to their own not-found shape). The whole copy is one
    /// transaction, so a failure leaves no half-built session behind.
    ///
    /// What travels: the event log verbatim (per-event timestamps included), `tool_outputs`,
    /// `permission`, `capabilities_json`, the cumulative stats, and `cwd` / `additional_roots`
    /// unless [`ForkOverrides`] replaces them.
    ///
    /// What deliberately does not:
    ///
    /// - **`created_at` / `updated_at`**, both stamped to now. Retention GC deletes by `updated_at`
    ///   at every agent startup, so inheriting the source's would let a fork of an old session be
    ///   swept before its first turn.
    /// - **Sub-agent children.** A child links to its parent only through `parent_session_id`,
    ///   while the sub-agent's *result* already sits in the parent's own event log as a tool
    ///   result, so the copy is self-contained without them. This is the intended divergence from
    ///   [`Self::import_sessions`], which copies the tree because an archive should restore whole.
    ///
    /// **`parent_session_id` and `subagent_spec_json` travel.** Written NULL, a fork of a sub-agent
    /// would be a *drivable* copy of a sub-agent's whole conversation carrying no spawn terms: no
    /// `[subagents]` denials, no memory or instruction grants, and the host's permission level
    /// rather than the sub-agent's. Carrying both columns is what makes "a session's row says
    /// whether something spawned it" survive a copy, which is the property
    /// [`crate::host::refuse_a_spawned_session`] rests on. A fork of a sub-agent is a sibling under
    /// the same parent, continued the same way: through `agent_followup`.
    ///
    /// Copied in SQL rather than through the export/import structs, which silently drop any column
    /// they do not model. The column list in [`Self::fork_session_into`] lives next to the schema,
    /// and `fork_copies_every_session_column` fails when a new column appears without a decision
    /// about it.
    ///
    /// **The source stands still and the copy is locked before it exists.** Two locks, one on each
    /// side of the copy. The source's, unless `source_lock` says the caller holds it: a turn
    /// persists its prompt before the provider answers, so a copy taken from a source another
    /// process is writing ends on a user message nothing answered, and restores as an unusable
    /// session; that is refused as [`MekaError::SessionLocked`]. The copy's, taken before its row
    /// lands: committing first and locking after is the same commit-then-claim window
    /// [`Self::create_session_locked`] closes, in the same width, where a concurrent
    /// `meka session delete --all` enumerates the copy, takes the lock nobody holds, and deletes
    /// it, after which the fork locks the vanished id successfully and its next turn dies on a
    /// foreign-key violation. Under ACP it is quieter still: `load_events` returns empty and the
    /// editor is handed a silently blank fork.
    ///
    /// The one fork door. Every host reaches it: `meka session fork` and the REPL's `/fork`,
    /// `POST /v1/sessions/{id}/fork` and ACP's `session/fork`.
    ///
    /// `Ok(None)` means the source is gone; both claims are released *and* their files removed,
    /// since nothing was written for them to protect and an unknown id is client-reachable.
    pub(crate) async fn fork_session_locked(
        &self,
        source: Uuid,
        overrides: ForkOverrides,
        source_lock: SourceLock,
    ) -> Result<Option<(CreatedSession, std::result::Result<FileLock, MekaError>)>> {
        // First, so a refusal costs no claim on a copy that will not be made. Sibling forks in this
        // process queue behind one another here before probing, since their flock would otherwise
        // read as another process's; see `Store::probed_forks`.
        let (_probing, source_held) = match source_lock {
            SourceLock::Probe => {
                let probing = self.probed_forks.lock().await;
                (Some(probing), Some(self.lock_session(source)?))
            }
            SourceLock::HeldByCaller => (None, None),
        };
        let new_id = Uuid::new_v4();
        let lock = self.claim_a_fresh_id(new_id);
        let created = match self.fork_session_into(new_id, source, overrides).await {
            Ok(Some(created)) => created,
            Ok(None) => {
                self.discard_unused_claim(lock, new_id);
                if let Some(source_held) = source_held {
                    self.discard_unused_claim(Ok(source_held), source);
                }
                return Ok(None);
            }
            Err(error) => {
                // Nothing was written under the copy's id, so its claim goes, file included, or
                // every failed copy leaves a file per attempt. The source's is only released: its
                // row is still there, and its file is the one every other process locks it by, so
                // unlinking it could leave two holders on two inodes.
                self.discard_unused_claim(lock, new_id);
                drop(source_held);
                return Err(error);
            }
        };
        // The copy has committed, so the source is free to move again.
        drop(source_held);
        Ok(Some((created, lock)))
    }

    /// [`Self::fork_session_locked`] with the copy's lock dropped, for tests about the copy rather
    /// than the locks.
    #[cfg(test)]
    pub(super) async fn fork_session_for_test(
        &self,
        source: Uuid,
        overrides: ForkOverrides,
    ) -> Result<Option<CreatedSession>> {
        Ok(self
            .fork_session_locked(source, overrides, SourceLock::Probe)
            .await?
            .map(|(created, _lock)| created))
    }

    /// The copy itself, on an id the caller has already minted (and may already have locked).
    pub(super) async fn fork_session_into(
        &self,
        new_id: Uuid,
        source: Uuid,
        overrides: ForkOverrides,
    ) -> Result<Option<CreatedSession>> {
        let created_at = chrono::Utc::now().to_rfc3339();
        let cwd_override = overrides.cwd.map(|path| path.display().to_string());
        // A flag rather than a nested `Option`: "inherit" and "override with no roots" both encode
        // to SQL NULL, so `COALESCE` alone can't tell them apart and would resurrect the source's
        // roots when a caller explicitly asked for none.
        let (override_roots, roots_override) = match overrides.additional_roots {
            Some(roots) => (true, encode_additional_roots(&roots)?),
            None => (false, None),
        };

        let created_at_for_db = created_at.clone();
        let inserted = self
            .connection
            .call(move |connection| -> rusqlite::Result<bool> {
                let transaction = connection.transaction()?;
                let source_id = source.to_string();
                let new_id_string = new_id.to_string();

                // The enclosing transaction is what makes the three statements below a consistent
                // snapshot: the first `INSERT` takes SQLite's write lock, so no other connection
                // can append an event to the source between the row copy and the message copy.
                let rows = transaction.execute(
                    "INSERT INTO sessions (
                         id, created_at, updated_at, parent_session_id, subagent_spec_json,
                         cwd, permission, approvals,
                         capabilities_json, token_id, additional_roots_json, profile,
                         stat_turns,
                         stat_input_tokens, stat_output_tokens,
                         stat_cache_creation_input_tokens, stat_cache_read_input_tokens,
                         stat_redactions, stat_redacted_images, stat_redacted_bytes
                     )
                     SELECT ?1, ?2, ?2, parent_session_id, subagent_spec_json,
                            COALESCE(?3, cwd), permission, approvals,
                            capabilities_json, ?4,
                            CASE WHEN ?5 THEN ?6 ELSE additional_roots_json END, profile,
                            stat_turns,
                            stat_input_tokens, stat_output_tokens,
                            stat_cache_creation_input_tokens, stat_cache_read_input_tokens,
                            stat_redactions, stat_redacted_images, stat_redacted_bytes
                     FROM sessions WHERE id = ?7",
                    rusqlite::params![
                        new_id_string,
                        created_at_for_db,
                        cwd_override,
                        overrides.token_id,
                        override_roots,
                        roots_override,
                        source_id,
                    ],
                )?;
                if rows == 0 {
                    // No source row: nothing was inserted, so the rollback is a formality.
                    transaction.rollback()?;
                    return Ok(false);
                }

                // Row by row rather than one `INSERT ... SELECT`, because each copied message needs
                // the source row's blob references under its own new id: a reference is what keeps
                // an image from the sweep when the source is deleted, and what lets this fork read
                // it over the API.
                {
                    let mut select = transaction.prepare(
                        "SELECT id, role, content, created_at FROM messages \
                         WHERE session_id = ?1 ORDER BY id ASC",
                    )?;
                    let mut insert = transaction.prepare(
                        "INSERT INTO messages (session_id, role, content, created_at) \
                         VALUES (?1, ?2, ?3, ?4)",
                    )?;
                    let mut link = transaction.prepare(
                        "INSERT OR IGNORE INTO message_blobs (message_id, hash) \
                         SELECT ?1, hash FROM message_blobs WHERE message_id = ?2",
                    )?;
                    let rows = select.query_map(rusqlite::params![source_id], |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                        ))
                    })?;
                    for row in rows {
                        let (source_message_id, role, content, created_at) = row?;
                        insert.execute(rusqlite::params![
                            new_id_string,
                            role,
                            content,
                            created_at
                        ])?;
                        link.execute(rusqlite::params![
                            transaction.last_insert_rowid(),
                            source_message_id
                        ])?;
                    }
                }
                transaction.execute(
                    "INSERT INTO tool_outputs (session_id, name, content, created_at)
                     SELECT ?1, name, content, created_at
                     FROM tool_outputs WHERE session_id = ?2",
                    rusqlite::params![new_id_string, source_id],
                )?;

                transaction.commit()?;
                Ok(true)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to fork session: {error}")))?;

        if !inserted {
            return Ok(None);
        }
        Ok(Some(CreatedSession {
            id: new_id,
            created_at,
        }))
    }

    /// Persist a single event from the conversation log. Events are
    /// encoded into the existing `messages(role, content, …)` table:
    ///
    /// - `Event::Append(message)` writes one row with the message's role (`user` / `assistant` /
    ///   `tool_results`).
    /// - `Event::CompactBoundary { … }` writes one row with the pseudo-role `compact_boundary` and
    ///   a JSON-serialized envelope in `content`.
    pub(crate) async fn save_event(
        &self,
        session_id: Uuid,
        event: &crate::conversation::Event,
    ) -> Result<()> {
        let (event, blobs) = super::blobs::externalize_images(event);
        let references = super::blobs::blob_references(&event);
        let (role, content) = encode_event_for_db(&event)
            .map_err(|error| MekaError::Database(format!("failed to encode event: {error}")))?;
        self.save_row(session_id, role, content, blobs, references)
            .await
    }

    /// Persist a batch of events in one SQLite transaction. The agent loop uses this to save the
    /// assistant message and the matching tool-results message together: without the transaction,
    /// a failure on the tool-results row would leave the assistant message persisted with
    /// `tool_use` blocks but no matching results, corrupting the conversation for every later
    /// turn.
    ///
    /// An empty batch is a no-op. `updated_at` is bumped once at the end of the batch rather than
    /// once per row, so the row reflects the batch's commit time.
    pub(crate) async fn save_events_atomic(
        &self,
        session_id: Uuid,
        events: Vec<crate::conversation::Event>,
    ) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }
        // Encode all events upfront so a serialization failure aborts before any DB I/O.
        let mut encoded: Vec<(String, String, Vec<String>)> = Vec::with_capacity(events.len());
        let mut blobs = Vec::new();
        for event in &events {
            let (event, mut taken) = super::blobs::externalize_images(event);
            blobs.append(&mut taken);
            let references = super::blobs::blob_references(&event);
            let (role, content) = encode_event_for_db(&event)
                .map_err(|error| MekaError::Database(format!("failed to encode event: {error}")))?;
            encoded.push((role, content, references));
        }
        let now = chrono::Utc::now().to_rfc3339();
        let session_id_str = session_id.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let transaction = connection.transaction()?;
                super::blobs::insert_blobs(&transaction, &blobs, &now)?;
                {
                    let mut insert = transaction.prepare(
                        "INSERT INTO messages (session_id, role, content, created_at) \
                         VALUES (?1, ?2, ?3, ?4)",
                    )?;
                    for (role, content, references) in &encoded {
                        insert.execute(rusqlite::params![session_id_str, role, content, now])?;
                        let message_id = transaction.last_insert_rowid();
                        super::blobs::link_message_blobs(&transaction, message_id, references)?;
                    }
                }
                transaction.execute(
                    "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![now, session_id_str],
                )?;
                transaction.commit()?;
                Ok(())
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to save event batch atomically: {error}"))
            })
    }

    /// Persist a set of imported sessions (a root plus its sub-agent descendants) in a single
    /// transaction: the `sessions` rows (preserving `created_at` and cumulative stats, but never
    /// the `token_id` fingerprint), each session's event log (preserving per-event timestamps), and
    /// its `tool_outputs`. `records` MUST be ordered parents-first so every `new_parent_id`
    /// references an already-inserted row (the `parent_session_id` foreign key is enforced).
    /// All-or-nothing: any failure rolls back the whole import, leaving no partial tree.
    ///
    /// `updated_at` is stamped to the import time rather than restored from the export. Retention
    /// GC deletes by `updated_at` ([`Self::delete_expired_sessions`], run at startup when
    /// `[session].retention_days` is set), so restoring the original value would have an archive
    /// older than the window swept on the next launch, before anyone could resume it. `created_at`
    /// still carries the original for provenance.
    ///
    /// `blobs` are the image bytes the archive carries; every reference in the events has to name
    /// one of them or a blob the store already holds, or the import is refused whole.
    pub(crate) async fn import_sessions(
        &self,
        records: Vec<ImportSessionRecord>,
        blobs: Vec<super::blobs::StoredBlob>,
    ) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        // Encode every event up front so a serialization failure aborts before any DB I/O.
        struct EncodedSession {
            id: String,
            parent_id: Option<String>,
            created_at: String,
            cwd: Option<String>,
            permission: String,
            approvals: bool,
            capabilities_json: Option<String>,
            additional_roots_json: Option<String>,
            subagent_spec_json: Option<String>,
            profile: String,
            stats: crate::stats::SessionStatsSnapshot,
            events: Vec<(String, String, String, Vec<String>)>,
            tool_outputs: Vec<(String, String)>,
        }
        let imported_at = chrono::Utc::now().to_rfc3339();
        let root_ids: Vec<Uuid> = records
            .iter()
            .filter(|record| record.new_parent_id.is_none())
            .map(|record| record.new_id)
            .collect();
        let archive_blobs: Vec<super::blobs::NewBlob> = blobs
            .into_iter()
            .map(|blob| super::blobs::NewBlob {
                hash: blob.hash,
                media_type: blob.media_type,
                bytes: blob.bytes,
            })
            .collect();
        let mut archive_blobs = archive_blobs;
        let mut encoded = Vec::with_capacity(records.len());
        for record in records {
            let mut events = Vec::with_capacity(record.events.len());
            for (at, event) in &record.events {
                // An archive carries references, but one written by hand or by another tool may
                // carry bytes; either way the row gets a reference and the bytes a blob.
                let (event, mut taken) = super::blobs::externalize_images(event);
                archive_blobs.append(&mut taken);
                let references = super::blobs::blob_references(&event);
                let (role, content) = encode_event_for_db(&event).map_err(|error| {
                    MekaError::Database(format!("failed to encode event: {error}"))
                })?;
                events.push((role, content, at.clone(), references));
            }
            encoded.push(EncodedSession {
                id: record.new_id.to_string(),
                parent_id: record.new_parent_id.map(|id| id.to_string()),
                created_at: record.created_at,
                cwd: record.cwd.map(|path| path.to_string_lossy().into_owned()),
                permission: record.permission.to_string(),
                approvals: record.approvals,
                capabilities_json: record.capabilities_json,
                additional_roots_json: encode_additional_roots(&record.additional_roots)?,
                subagent_spec_json: record.subagent_spec_json,
                profile: record.profile,
                stats: record.stats,
                events,
                tool_outputs: record.tool_outputs,
            });
        }
        // Resolved ahead of the transaction, so an archive naming a blob nobody holds is refused as
        // the caller's mistake, in words for them; the same check inside the transaction stays as
        // the fail-closed guard for a blob swept between the two.
        let carried: std::collections::HashSet<&str> = archive_blobs
            .iter()
            .map(|blob| blob.hash.as_str())
            .collect();
        let mut unresolved: Vec<String> = Vec::new();
        for session in &encoded {
            for (_, _, _, references) in &session.events {
                for hash in references {
                    if !carried.contains(hash.as_str()) && !unresolved.contains(hash) {
                        unresolved.push(hash.clone());
                    }
                }
            }
        }
        if let Some(hash) = self.first_missing_blob(unresolved).await? {
            return Err(MekaError::Usage(format!(
                "the archive references image blob {hash}, which it does not carry and this store \
                 does not hold"
            )));
        }
        // Each root of the tree is claimed before its row lands and released once it has, the
        // ordering every door that mints a session keeps (see `claim_a_fresh_id`): a row visible
        // before its lock is one a concurrent `meka session delete --all` enumerates and sweeps.
        // Only the parentless rows, because a sub-agent's row is never opened on its own. Nothing
        // here keeps the claims afterwards: an import hands back an id, not a live session.
        //
        // After the encoding, whose every `?` returns without reaching the match below: a claim
        // taken ahead of it left a file per refused archive.
        let root_claims: Vec<(Uuid, std::result::Result<FileLock, MekaError>)> = root_ids
            .into_iter()
            .map(|id| (id, self.claim_a_fresh_id(id)))
            .collect();
        let written = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                let transaction = connection.transaction()?;
                super::blobs::insert_blobs(&transaction, &archive_blobs, &imported_at)?;
                for session in &encoded {
                    transaction.execute(
                        "INSERT INTO sessions (
                             id, created_at, updated_at, parent_session_id, cwd, permission,
                             capabilities_json, additional_roots_json, subagent_spec_json,
                             profile, approvals,
                             stat_turns, stat_input_tokens, stat_output_tokens,
                             stat_cache_creation_input_tokens, stat_cache_read_input_tokens,
                             stat_redactions, stat_redacted_images, stat_redacted_bytes
                         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)",
                        rusqlite::params![
                            session.id,
                            session.created_at,
                            imported_at,
                            session.parent_id,
                            session.cwd,
                            session.permission,
                            session.capabilities_json,
                            session.additional_roots_json,
                            session.subagent_spec_json,
                            session.profile,
                            session.approvals,
                            session.stats.turns as i64,
                            session.stats.input_tokens as i64,
                            session.stats.output_tokens as i64,
                            session.stats.cache_creation_input_tokens as i64,
                            session.stats.cache_read_input_tokens as i64,
                            session.stats.redactions as i64,
                            session.stats.redacted_images as i64,
                            session.stats.redacted_bytes as i64,
                        ],
                    )?;
                    {
                        let mut insert_event = transaction.prepare(
                            "INSERT INTO messages (session_id, role, content, created_at) \
                             VALUES (?1, ?2, ?3, ?4)",
                        )?;
                        for (role, content, created_at, references) in &session.events {
                            // Refused rather than linked to nothing: a reference the archive did
                            // not carry and the store does not hold would be an image no reader
                            // could ever show.
                            if let Some(hash) = super::blobs::missing_blob(&transaction, references)? {
                                return Err(rusqlite::Error::InvalidParameterName(format!(
                                    "the archive references image blob {hash}, which it does not \
                                     carry and this store does not hold"
                                )));
                            }
                            insert_event.execute(rusqlite::params![
                                session.id,
                                role,
                                content,
                                created_at
                            ])?;
                            let message_id = transaction.last_insert_rowid();
                            super::blobs::link_message_blobs(&transaction, message_id, references)?;
                        }
                    }
                    {
                        let mut insert_output = transaction.prepare(
                            "INSERT INTO tool_outputs (session_id, name, content, created_at) \
                             VALUES (?1, ?2, ?3, ?4)",
                        )?;
                        for (name, content) in &session.tool_outputs {
                            insert_output.execute(rusqlite::params![
                                session.id,
                                name,
                                content,
                                session.created_at
                            ])?;
                        }
                    }
                }
                transaction.commit()?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to import sessions: {error}")));
        match written {
            // The rows exist, so their lock files stay, as every session's does.
            Ok(()) => drop(root_claims),
            Err(_) => {
                for (id, claim) in root_claims {
                    self.discard_unused_claim(claim, id);
                }
            }
        }
        written
    }

    /// Load every event for a session in chronological order. The `user`, `assistant`,
    /// `tool_results` and `user_blocks` roles are what `encode_event_for_db` writes for an
    /// `Event::Append`, so reading them back as `Event::Append` closes that round trip rather than
    /// falling back to anything; `compact_boundary` and `repair` rows are deserialized from their
    /// JSON envelope. Unknown roles are skipped with a warning.
    pub(crate) async fn load_events(
        &self,
        session_id: Uuid,
    ) -> Result<Vec<crate::conversation::Event>> {
        let stored = self.load_messages(session_id).await?;
        let mut events = Vec::with_capacity(stored.len());
        for row in stored {
            match decode_event_from_row(&row) {
                Ok(Some(event)) => events.push(event),
                Ok(None) => {
                    tracing::warn!(
                        "dropping a session row with unknown role '{role}'",
                        role = row.role
                    );
                }
                Err(error) => {
                    tracing::warn!(
                        "failed to decode a session row of role '{role}': {error}",
                        role = row.role
                    );
                }
            }
        }
        Ok(events)
    }

    /// Variant of [`Self::load_events`] that also returns the persisted `created_at` timestamp for
    /// each event. Used by the HTTP `GET /v1/sessions/{id}/messages` endpoint to surface
    /// per-message creation timestamps on `MessageView` per the spec's resource model.
    /// Order matches `load_events` exactly: chronological by insert id.
    pub(crate) async fn load_events_with_timestamps(
        &self,
        session_id: Uuid,
    ) -> Result<Vec<(String, crate::conversation::Event)>> {
        let stored = self.load_messages(session_id).await?;
        let mut events = Vec::with_capacity(stored.len());
        for row in stored {
            match decode_event_from_row(&row) {
                Ok(Some(event)) => events.push((row.created_at, event)),
                Ok(None) => {
                    tracing::warn!(
                        "dropping a session row with unknown role '{role}'",
                        role = row.role
                    );
                }
                Err(error) => {
                    tracing::warn!(
                        "failed to decode a session row of role '{role}': {error}",
                        role = row.role
                    );
                }
            }
        }
        Ok(events)
    }

    /// Load a session together with every descendant sub-agent session (recursively via
    /// `parent_session_id`), ordered root-first (breadth-first by depth). Used by JSON session
    /// export to capture an entire agent tree, and the root-first order lets an importer insert
    /// parents before children so the `parent_session_id` foreign key is always satisfied. Returns
    /// an empty vec when the root session doesn't exist.
    pub(crate) async fn load_session_tree(&self, root: Uuid) -> Result<Vec<SessionMetaRow>> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "WITH RECURSIVE tree(id, depth) AS (
                         SELECT id, 0 FROM sessions WHERE id = ?1
                         UNION ALL
                         SELECT s.id, tree.depth + 1
                         FROM sessions s JOIN tree ON s.parent_session_id = tree.id
                     )
                     SELECT s.id, s.parent_session_id, s.created_at, s.updated_at,
                            s.cwd, s.permission, s.capabilities_json, s.additional_roots_json,
                            s.subagent_spec_json, s.profile, s.approvals
                     FROM sessions s JOIN tree ON s.id = tree.id
                     ORDER BY tree.depth ASC, s.created_at ASC, s.id ASC",
                )?;
                let parse_uuid = |value: String| {
                    Uuid::parse_str(&value)
                        .map_err(|error| rusqlite::Error::InvalidParameterName(error.to_string()))
                };
                let rows = statement.query_map(rusqlite::params![root.to_string()], |row| {
                    let id = parse_uuid(row.get::<_, String>(0)?)?;
                    let parent_id = match row.get::<_, Option<String>>(1)? {
                        Some(value) => Some(parse_uuid(value)?),
                        None => None,
                    };
                    Ok(SessionMetaRow {
                        id,
                        parent_id,
                        created_at: row.get(2)?,
                        updated_at: row.get(3)?,
                        cwd: row.get::<_, Option<String>>(4)?.map(PathBuf::from),
                        permission: crate::permission::parse_recorded_permission(
                            row.get::<_, Option<String>>(5)?.as_deref(),
                            &format_args!("session {id}"),
                        ),
                        capabilities_json: row.get(6)?,
                        additional_roots: decode_additional_roots(
                            row.get::<_, Option<String>>(7)?.as_deref(),
                        ),
                        subagent_spec_json: row.get(8)?,
                        profile: row.get(9)?,
                        approvals: row.get(10)?,
                    })
                })?;
                let mut out = Vec::new();
                for row in rows {
                    out.push(row?);
                }
                Ok(out)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to load session tree: {error}")))
    }

    /// Persist a single plain row into the `messages` table, carrying no image. Tests call this
    /// directly to populate fixtures; everything else goes through the event API.
    #[cfg(test)]
    pub(super) async fn save_message(
        &self,
        session_id: Uuid,
        role: &str,
        content: &str,
    ) -> Result<()> {
        self.save_row(
            session_id,
            role.to_string(),
            content.to_string(),
            Vec::new(),
            Vec::new(),
        )
        .await
    }

    /// The row half of [`Self::save_event`]: the message, the blobs it took its images out into,
    /// and the references that tie the two, in one transaction.
    async fn save_row(
        &self,
        session_id: Uuid,
        role: String,
        content: String,
        blobs: Vec<super::blobs::NewBlob>,
        references: Vec<String>,
    ) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                // One transaction, like `save_events_atomic`: a message committed without its
                // session's `updated_at` moving is ordered wrong in every listing and judged
                // expired by the retention sweep despite the turn that wrote it.
                let transaction = connection.transaction()?;
                super::blobs::insert_blobs(&transaction, &blobs, &now)?;
                transaction.execute(
                    "INSERT INTO messages (session_id, role, content, created_at) VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![session_id.to_string(), role, content, &now],
                )?;
                let message_id = transaction.last_insert_rowid();
                super::blobs::link_message_blobs(&transaction, message_id, &references)?;
                transaction.execute(
                    "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![now, session_id.to_string()],
                )?;
                transaction.commit()
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to save message: {error}")))
    }

    /// Persist the cumulative `/status` counters onto the session row so they survive resume. The
    /// caller treats this as best-effort (a failed write must never fail a turn).
    pub(crate) async fn save_session_stats(
        &self,
        session_id: Uuid,
        stats: &crate::stats::SessionStatsSnapshot,
    ) -> Result<()> {
        // SQLite has no u64; counts never realistically exceed i64::MAX, so cast on the way in/out.
        let stats = stats.clone();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE sessions SET
                         stat_turns = ?2,
                         stat_input_tokens = ?3,
                         stat_output_tokens = ?4,
                         stat_cache_creation_input_tokens = ?5,
                         stat_cache_read_input_tokens = ?6,
                         stat_redactions = ?7,
                         stat_redacted_images = ?8,
                         stat_redacted_bytes = ?9
                     WHERE id = ?1",
                    rusqlite::params![
                        session_id.to_string(),
                        stats.turns as i64,
                        stats.input_tokens as i64,
                        stats.output_tokens as i64,
                        stats.cache_creation_input_tokens as i64,
                        stats.cache_read_input_tokens as i64,
                        stats.redactions as i64,
                        stats.redacted_images as i64,
                        stats.redacted_bytes as i64,
                    ],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to save session stats: {error}")))
    }

    /// Load the persisted cumulative stats for a session, seeding `SessionStats` on resume. Returns
    /// all-zero when the session row doesn't exist yet (fresh session).
    pub(crate) async fn load_session_stats(
        &self,
        session_id: Uuid,
    ) -> Result<crate::stats::SessionStatsSnapshot> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let result = connection.query_row(
                    "SELECT stat_turns, stat_input_tokens, stat_output_tokens,
                            stat_cache_creation_input_tokens, stat_cache_read_input_tokens,
                            stat_redactions, stat_redacted_images, stat_redacted_bytes
                     FROM sessions WHERE id = ?1",
                    rusqlite::params![session_id.to_string()],
                    |row| {
                        Ok(crate::stats::SessionStatsSnapshot {
                            turns: row.get::<_, i64>(0)? as u64,
                            input_tokens: row.get::<_, i64>(1)? as u64,
                            output_tokens: row.get::<_, i64>(2)? as u64,
                            cache_creation_input_tokens: row.get::<_, i64>(3)? as u64,
                            cache_read_input_tokens: row.get::<_, i64>(4)? as u64,
                            redactions: row.get::<_, i64>(5)? as u64,
                            redacted_images: row.get::<_, i64>(6)? as u64,
                            redacted_bytes: row.get::<_, i64>(7)? as u64,
                        })
                    },
                );
                match result {
                    Ok(snapshot) => Ok(snapshot),
                    Err(rusqlite::Error::QueryReturnedNoRows) => {
                        Ok(crate::stats::SessionStatsSnapshot::default())
                    }
                    Err(error) => Err(error),
                }
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to load session stats: {error}")))
    }

    /// Fetch raw rows for a session. Internal helper for [`Self::load_events`]; external consumers
    /// go through the event API.
    pub(super) async fn load_messages(&self, session_id: Uuid) -> Result<Vec<StoredMessage>> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "SELECT role, content, created_at FROM messages WHERE session_id = ?1 ORDER BY id ASC",
                )?;

                let messages = statement
                    .query_map(rusqlite::params![session_id.to_string()], |row| {
                        Ok(StoredMessage {
                            role: row.get(0)?,
                            content: row.get(1)?,
                            created_at: row.get(2)?,
                        })
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;

                Ok(messages)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to load messages: {error}")))
    }

    /// How many times this session has been compacted, i.e. its compaction *generation*.
    ///
    /// Read from the database rather than the in-memory log because
    /// [`crate::conversation::Conversation::prune_compacted_events`] drains every event preceding
    /// the most recent boundary, so the log in memory holds at most one no matter how many
    /// compactions have run. Every boundary is still its own row here.
    ///
    /// Worth surfacing to the model: a fourth summary-of-a-summary has lost far more than a first,
    /// and an agent that knows its generation can compensate by writing to memory more readily.
    pub(crate) async fn count_compactions(&self, session_id: Uuid) -> Result<u64> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.query_row(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?1 AND role = ?2",
                    rusqlite::params![session_id.to_string(), COMPACT_BOUNDARY_ROLE],
                    |row| row.get::<_, i64>(0),
                )
            })
            .await
            .map(|count| count.max(0) as u64)
            .map_err(|error| MekaError::Database(format!("failed to count compactions: {error}")))
    }

    /// The session `meka -c` resumes: the most recently touched one a host can actually drive.
    ///
    /// Root sessions only, like [`Self::list_sessions`]'s default, and for a sharper reason than
    /// tidiness. A sub-agent's row is touched by its own turns, so a sub-agent still running when
    /// its parent's turn ends (which `agent_spawn`'s `background` parameter makes ordinary) sorts
    /// above the session the user was actually in. `crate::host::refuse_a_spawned_session`
    /// refuses a sub-agent, so picking one would dead-end `-c` on a session the user never named,
    /// with no way to ask for the next one down.
    ///
    /// [`spawned_session_sql`] rather than `parent_session_id IS NULL`, so the filter answers the
    /// same question the refusal does. An imported sub-agent has no parent link and is refused all
    /// the same, so keying on the link alone would let `-c` select one and then decline it:
    /// exactly the dead-end above, and permanent, because nothing newer can outrank it.
    pub(crate) async fn last_session_id(&self) -> Result<Option<Uuid>> {
        self.connection
            .call(|connection| -> rusqlite::Result<_> {
                let result: std::result::Result<String, _> = connection.query_row(
                    &format!(
                        "SELECT id FROM sessions
                         WHERE NOT {}
                         ORDER BY updated_at DESC LIMIT 1",
                        spawned_session_sql("")
                    ),
                    [],
                    |row| row.get(0),
                );

                match result {
                    Ok(id_str) => {
                        let uuid = Uuid::parse_str(&id_str).map_err(|error| {
                            rusqlite::Error::InvalidParameterName(error.to_string())
                        })?;
                        Ok(Some(uuid))
                    }
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(error) => Err(error),
                }
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to get last session: {error}")))
    }

    /// Resolve what a user typed for a session to one id: a full UUID that exists, or a prefix that
    /// matches exactly one session.
    ///
    /// A full id that is not there is [`crate::error::MekaError::SessionNotFound`], the refusal
    /// every host maps; the prefix errors are [`crate::error::MekaError::Usage`], because the words
    /// are for the person who typed the value.
    pub(crate) async fn resolve_session_id(&self, value: &str) -> Result<Uuid> {
        if let Ok(id) = value.parse::<Uuid>() {
            if !self.session_exists(id).await? {
                return Err(crate::error::MekaError::SessionNotFound(id));
            }
            return Ok(id);
        }

        let matches = self.find_sessions_by_prefix(value).await?;
        match matches.len() {
            0 => Err(crate::error::MekaError::Usage(format!(
                "no session matches prefix '{value}'"
            ))),
            1 => Ok(matches[0]),
            _ => {
                let listing: Vec<String> = matches.iter().map(Uuid::to_string).collect();
                // "at least" once the scan's own bound is reached, because the count would
                // otherwise be the cap rather than the truth. Every id it did fetch
                // is named either way, which is what the caller needs to pick one.
                Err(crate::error::MekaError::Usage(format!(
                    "ambiguous prefix '{}' matches {}{} sessions: {}",
                    value,
                    if matches.len() >= PREFIX_MATCH_CAP {
                        "at least "
                    } else {
                        ""
                    },
                    matches.len(),
                    listing.join(", "),
                )))
            }
        }
    }

    /// Whether a row with this id is in the store.
    pub(crate) async fn session_exists(&self, session_id: Uuid) -> Result<bool> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let count: i64 = connection.query_row(
                    "SELECT COUNT(*) FROM sessions WHERE id = ?1",
                    rusqlite::params![session_id.to_string()],
                    |row| row.get(0),
                )?;
                Ok(count > 0)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to check session existence: {error}"))
            })
    }

    /// Every session id in the store, which is the set a prefix is resolved against.
    ///
    /// A listing filters (`-n`, and sub-agent sessions hidden by default) while
    /// [`Self::find_sessions_by_prefix`] does not. Sizing the printed id column against the rows
    /// alone would therefore print a prefix that every command taking one refuses as ambiguous.
    pub(crate) async fn all_session_ids(&self) -> Result<Vec<String>> {
        self.connection
            .call(|connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare("SELECT id FROM sessions")?;
                let ids = statement
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<String>>>()?;
                Ok(ids)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to list session ids: {error}")))
    }

    /// Resolve a session-ID prefix (e.g. `d64`) to the matching full UUIDs.
    ///
    /// Behind every command that takes a session id, so none of them needs the whole UUID typed.
    /// Capped at [`PREFIX_MATCH_CAP`] matches; ordered most-recent-first so the caller's "ambiguous
    /// prefix" listing leads with the session the user most likely meant.
    ///
    /// Anything outside the UUID alphabet (`0-9a-fA-F-`) returns an empty list, both because such
    /// a prefix can't match any real session ID and to keep SQL `LIKE` wildcards (`%`, `_`) from
    /// sneaking through.
    pub(crate) async fn find_sessions_by_prefix(&self, prefix: &str) -> Result<Vec<Uuid>> {
        if !crate::text::is_usable_id_prefix(prefix) {
            return Ok(Vec::new());
        }
        let pattern = format!("{prefix}%");
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    // Two is all a caller needs: one match resolves, and any second makes it
                    // ambiguous. A cap reported verbatim reads as the count, so the refusal says
                    // "at least" once the cap is hit and names every id it fetched, which is why
                    // fetching a bounded few is enough.
                    &format!(
                        "SELECT id FROM sessions WHERE id LIKE ?1 \
                         ORDER BY updated_at DESC LIMIT {PREFIX_MATCH_CAP}"
                    ),
                )?;
                let rows = statement.query_map(rusqlite::params![pattern], |row| {
                    let id: String = row.get(0)?;
                    Ok(id)
                })?;
                let mut ids = Vec::new();
                for row in rows {
                    if let Ok(uuid) = Uuid::parse_str(&row?) {
                        ids.push(uuid);
                    }
                }
                Ok(ids)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to find sessions by prefix: {error}"))
            })
    }

    /// List sessions, most-recent first. When `include_children` is `false`, sub-agent sessions
    /// (rows with non-NULL `parent_session_id`) are hidden; they're persisted for audit/debug but
    /// shouldn't clutter the user's view of their own conversations. Set to `true` to surface them,
    /// e.g. via `meka session list --include-children`.
    ///
    /// `cwd_filter`, if `Some`, restricts the result set to sessions whose persisted `cwd` matches
    /// the given path. Rows with NULL `cwd` are excluded: a session created by
    /// `create_session(None, "test-profile".to_string())` recorded no cwd to match against.
    ///
    /// `cursor`, if `Some`, is a previous `next_cursor` value from this method; rows are returned
    /// strictly *after* the cursor in `(updated_at, id) DESC` order. Returns `(rows, next_cursor)`;
    /// `next_cursor` is `Some` iff there is at least one more row past `limit`. Invalid cursors
    /// are rejected with [`MekaError::Database`].
    pub(crate) async fn list_sessions(
        &self,
        limit: u32,
        include_children: bool,
        cwd_filter: Option<&Path>,
        cursor: Option<&str>,
    ) -> Result<(Vec<SessionSummary>, Option<String>)> {
        let cursor_decoded = match cursor {
            Some(token) => Some(decode_list_cursor(token)?),
            None => None,
        };
        let cwd_filter_string = cwd_filter.map(|path| path.display().to_string());

        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                // The shared rule, so `--include-children` covers the same rows the refusal treats
                // as children. Keying on the parent link alone would list an imported sub-agent
                // among the root sessions and then decline to run it, with the refusal's own text
                // telling the reader to pass `--include-children` to see a row that is already in
                // front of them.
                let not_spawned = format!("NOT {}", spawned_session_sql("s."));
                let mut clauses: Vec<&str> = Vec::new();
                if !include_children {
                    clauses.push(&not_spawned);
                }
                if cwd_filter_string.is_some() {
                    clauses.push("s.cwd = :cwd");
                }
                if cursor_decoded.is_some() {
                    // Keyset on (updated_at, id) DESC: strictly past the cursor row. Tie-break on
                    // id keeps pagination stable when multiple sessions share an updated_at.
                    clauses.push(
                        "(s.updated_at < :cursor_updated_at \
                          OR (s.updated_at = :cursor_updated_at AND s.id < :cursor_id))",
                    );
                }
                let where_clause = if clauses.is_empty() {
                    String::new()
                } else {
                    format!("WHERE {}", clauses.join(" AND "))
                };
                let title_row = TITLE_ROW_WHERE_SQL.as_str();
                let query = format!(
                    "SELECT s.id, s.created_at, s.updated_at, s.cwd, s.permission, s.capabilities_json, s.additional_roots_json, s.token_id, s.parent_session_id, s.profile, s.approvals,
                            COALESCE(
                              (SELECT content FROM messages
                               WHERE {title_row}
                               ORDER BY id ASC LIMIT 1),
                              ''
                            ) AS title_content,
                            COALESCE(
                              (SELECT role FROM messages
                               WHERE {title_row}
                               ORDER BY id ASC LIMIT 1),
                              ''
                            ) AS title_role
                     FROM sessions s
                     {where_clause}
                     ORDER BY s.updated_at DESC, s.id DESC
                     LIMIT :limit",
                );
                let mut statement = connection.prepare(&query)?;

                // Fetch one extra row to detect whether a next page exists without a second COUNT
                // query.
                let fetch_limit: i64 = i64::from(limit).saturating_add(1);
                let mut params: Vec<(&str, &dyn rusqlite::ToSql)> = Vec::new();
                params.push((":limit", &fetch_limit));
                if let Some(ref cwd) = cwd_filter_string {
                    params.push((":cwd", cwd));
                }
                if let Some((ref updated_at, ref id)) = cursor_decoded {
                    params.push((":cursor_updated_at", updated_at));
                    params.push((":cursor_id", id));
                }

                let rows = statement.query_map(params.as_slice(), |row| {
                    let id_str: String = row.get(0)?;
                    let created_at: String = row.get(1)?;
                    let updated_at: String = row.get(2)?;
                    let cwd: Option<String> = row.get(3)?;
                    let permission: Option<String> = row.get(4)?;
                    let capabilities_json: Option<String> = row.get(5)?;
                    let additional_roots_json: Option<String> = row.get(6)?;
                    let token_id: Option<String> = row.get(7)?;
                    let parent_id: Option<String> = row.get(8)?;
                    let profile: String = row.get(9)?;
                    let approvals: bool = row.get(10)?;
                    let title_role: String = row.get(12)?;
                    let title = title_of_first_user_row(&id_str, &title_role, row.get(11)?);
                    Ok((
                        id_str,
                        created_at,
                        updated_at,
                        cwd,
                        permission,
                        capabilities_json,
                        additional_roots_json,
                        token_id,
                        parent_id,
                        profile,
                        approvals,
                        title,
                    ))
                })?;

                let mut summaries = Vec::new();
                for row in rows {
                    let (
                        id_str,
                        created_at,
                        updated_at,
                        cwd,
                        permission,
                        capabilities_json,
                        additional_roots_json,
                        token_id,
                        parent_id,
                        profile,
                        approvals,
                        title,
                    ) = row?;
                    let id = Uuid::parse_str(&id_str).map_err(|error| {
                        rusqlite::Error::InvalidParameterName(error.to_string())
                    })?;
                    summaries.push(SessionSummary {
                        id,
                        created_at,
                        updated_at,
                        title,
                        cwd: cwd.map(PathBuf::from),
                        permission: crate::permission::parse_recorded_permission(
                            permission.as_deref(),
                            &format_args!("session {id}"),
                        ),
                        approvals,
                        profile,
                        capabilities_json,
                        additional_roots: decode_additional_roots(additional_roots_json.as_deref()),
                        token_id,
                        parent_id: parent_id.as_deref().and_then(|raw| Uuid::parse_str(raw).ok()),
                    });
                }
                Ok(summaries)
            })
            .await
            .map(|mut rows| {
                let next_cursor = if rows.len() > limit as usize {
                    rows.truncate(limit as usize);
                    rows.last()
                        .map(|row| encode_list_cursor(&row.updated_at, &row.id.to_string()))
                } else {
                    None
                };
                (rows, next_cursor)
            })
            .map_err(|error| MekaError::Database(format!("failed to list sessions: {error}")))
    }

    /// Fetch a single session by id without scanning the full list. Returns `Ok(None)` if the
    /// session doesn't exist. Used by ACP's `session/load` to verify the requested session exists
    /// and to surface its persisted cwd back to the client.
    pub(crate) async fn session_info(&self, id: Uuid) -> Result<Option<SessionSummary>> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let title_row = TITLE_ROW_WHERE_SQL.as_str();
                let mut statement = connection.prepare(&format!(
                    "SELECT s.id, s.created_at, s.updated_at, s.cwd, s.permission, s.capabilities_json, s.additional_roots_json, s.token_id, s.parent_session_id, s.profile, s.approvals,
                            COALESCE(
                              (SELECT content FROM messages
                               WHERE {title_row}
                               ORDER BY id ASC LIMIT 1),
                              ''
                            ) AS title_content,
                            COALESCE(
                              (SELECT role FROM messages
                               WHERE {title_row}
                               ORDER BY id ASC LIMIT 1),
                              ''
                            ) AS title_role
                     FROM sessions s
                     WHERE s.id = ?1",
                ))?;
                let mut rows = statement.query_map(rusqlite::params![id.to_string()], |row| {
                    let id_str: String = row.get(0)?;
                    let created_at: String = row.get(1)?;
                    let updated_at: String = row.get(2)?;
                    let cwd: Option<String> = row.get(3)?;
                    let permission: Option<String> = row.get(4)?;
                    let capabilities_json: Option<String> = row.get(5)?;
                    let additional_roots_json: Option<String> = row.get(6)?;
                    let token_id: Option<String> = row.get(7)?;
                    let parent_id: Option<String> = row.get(8)?;
                    let profile: String = row.get(9)?;
                    let approvals: bool = row.get(10)?;
                    let title_role: String = row.get(12)?;
                    let title = title_of_first_user_row(&id_str, &title_role, row.get(11)?);
                    Ok((
                        id_str,
                        created_at,
                        updated_at,
                        cwd,
                        permission,
                        capabilities_json,
                        additional_roots_json,
                        token_id,
                        parent_id,
                        profile,
                        approvals,
                        title,
                    ))
                })?;
                match rows.next() {
                    Some(row) => {
                        let (
                            id_str,
                            created_at,
                            updated_at,
                            cwd,
                            permission,
                            capabilities_json,
                            additional_roots_json,
                            token_id,
                            parent_id,
                            profile,
                            approvals,
                            title,
                        ) = row?;
                        let id = Uuid::parse_str(&id_str).map_err(|error| {
                            rusqlite::Error::InvalidParameterName(error.to_string())
                        })?;
                        Ok(Some(SessionSummary {
                            id,
                            created_at,
                            updated_at,
                            title,
                            cwd: cwd.map(PathBuf::from),
                            permission: crate::permission::parse_recorded_permission(
                                permission.as_deref(),
                                &format_args!("session {id}"),
                            ),
                            approvals,
                            profile,
                            additional_roots: decode_additional_roots(
                                additional_roots_json.as_deref(),
                            ),
                            capabilities_json,
                            token_id,
                            parent_id: parent_id
                                .as_deref()
                                .and_then(|raw| Uuid::parse_str(raw).ok()),
                        }))
                    }
                    None => Ok(None),
                }
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to fetch session: {error}")))
    }

    /// Take a session's lock, then read its row, in that order, so what the caller reads is what it
    /// now owns.
    ///
    /// Read first, a row's level, directory or profile can move under another process between the
    /// read and the lock (`meka -r --profile`, a `PATCH` on a server sharing the store), and the
    /// new holder then runs on a snapshot nobody else agrees with. The door for every host that
    /// makes a persisted session resident: a REPL or one-shot resume, `serve`'s re-attach,
    /// ACP's `session/load` and `session/resume`. A host that refuses a sub-agent does so
    /// before this, on the spawn terms, which never move;
    /// `crate::host::refuse_a_spawned_session` says why that answer has to arrive ahead of
    /// `SessionLocked`.
    ///
    /// [`MekaError::SessionLocked`] when another process holds it. [`MekaError::SessionNotFound`]
    /// when the row is gone, in which case the claim's file goes with it, since an id a client can
    /// name would otherwise leave one per attempt.
    pub(crate) async fn open_session_row(
        &self,
        session_id: Uuid,
    ) -> Result<(FileLock, SessionSummary)> {
        let lock = self.lock_session(session_id)?;
        let Some(summary) = self.session_info(session_id).await? else {
            self.discard_unused_claim(Ok(lock), session_id);
            return Err(MekaError::SessionNotFound(session_id));
        };
        Ok((lock, summary))
    }

    /// Backdate a session's `updated_at`, for tests that need one to look old to the retention
    /// sweep. Lives here because `connection` is private to this module, so tests in other modules
    /// have no other way to age a row.
    #[cfg(test)]
    pub(crate) async fn set_session_updated_at_for_test(
        &self,
        session_id: uuid::Uuid,
        updated_at: &str,
    ) -> Result<()> {
        let updated_at = updated_at.to_string();
        let rows = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![updated_at, session_id.to_string()],
                )
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to backdate session: {error}")))?;
        // A typo'd id would otherwise backdate nothing and leave the test asserting against a
        // session that was never aged, which reads as the sweep failing to match.
        if rows != 1 {
            return Err(MekaError::Database(format!(
                "expected to backdate 1 session, updated {rows}"
            )));
        }
        Ok(())
    }

    /// Sweep sessions no turn has touched inside the retention window, leaving alone any that
    /// another meka process currently has open.
    ///
    /// The lock check is not a nicety. `updated_at` is bumped by turns only, and resuming a session
    /// does not touch it, so a REPL left sitting at its prompt past the window is a perfect
    /// candidate for deletion while a human is looking at it. Any start that goes through
    /// `async_main` runs this sweep, so without it a completely unrelated `meka` in another
    /// terminal destroys the live session and announces `deleted 1 session(s)`; the operator's
    /// next turn runs against the provider and *then* fails on a foreign-key violation, with the
    /// answer paid for and lost, and every later turn in that REPL fails the same way.
    ///
    /// A locked *child* is not separately checked. The cascade would take one with its parent, but
    /// children are sub-agent rows, and the only thing that locks one is `agent_followup`, for the
    /// length of a sub-agent turn that runs inside its parent's turn. The parent is locked for as
    /// long as that lasts, so a live child always has a live parent, and the parent's lock is what
    /// spares the pair.
    pub(crate) async fn delete_expired_sessions(
        &self,
        retention: std::time::Duration,
    ) -> Result<SessionSweep> {
        // Both steps can blow up on an absurd `retention`, and this takes user input straight
        // from `--older-than-days`, so a run of digits must not panic. `TimeDelta` overflows around
        // 10^11 days; subtracting from `Utc::now()` overflows far sooner, around 96.4 million. Both
        // fall back to a ~100-year window, which matches nothing and so keeps every session: the
        // sane reading of "delete anything older than forever".
        let now = chrono::Utc::now();
        #[allow(
            clippy::expect_used,
            reason = "36,500 days back from now is within `DateTime`'s range"
        )]
        let fallback = now
            .checked_sub_signed(chrono::TimeDelta::days(36_500))
            .expect("100 years before now is representable");
        let cutoff = chrono::TimeDelta::from_std(retention)
            .ok()
            .and_then(|retention| now.checked_sub_signed(retention))
            .unwrap_or(fallback);
        let cutoff_str = cutoff.to_rfc3339();
        let cutoff_for_delete = cutoff_str.clone();

        let expired: Vec<Uuid> = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                // FK CASCADE sweeps messages, tool_outputs, and any sub-agent child sessions of the
                // expired parents.
                //
                // A session with a scheduled job still ahead of it is *not* expired, whatever
                // `updated_at` says; `NOT_SPOKEN_FOR_BY_A_SCHEDULE` says why the whole parent
                // chain of a job-owning session is spared.
                //
                // Selected rather than deleted outright, because which of these rows may go is not
                // a question the database can answer: it depends on which of them another process
                // has open. [`Self::delete_the_unattached_among`] re-applies the same condition
                // inside the delete, so splitting one statement into two does not open a window
                // where a job created in between is cascaded away by a decision taken before it
                // existed.
                let mut statement = connection.prepare(&format!(
                    "SELECT id FROM sessions WHERE updated_at < ?1 AND {NOT_SPOKEN_FOR_BY_A_SCHEDULE}"
                ))?;
                let ids = statement
                    .query_map(rusqlite::params![cutoff_str], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<String>>>()?;
                Ok(ids)
            })
            .await
            .map_err(|error| {
                MekaError::Database(format!("failed to list expired sessions: {error}"))
            })?
            .into_iter()
            .filter_map(|id| Uuid::parse_str(&id).ok())
            .collect();

        self.delete_the_unattached_among(
            &expired,
            NOT_SPOKEN_FOR_BY_A_SCHEDULE,
            Some(cutoff_for_delete.as_str()),
        )
        .await
    }

    /// Delete every one of `candidates` whose lock this process can take, and report what it left.
    ///
    /// The lock is held across the delete rather than probed and released, because the window
    /// between a probe and a `DELETE` is exactly long enough for another process to attach and
    /// start a turn on a row that is about to vanish underneath it.
    ///
    /// `still_eligible` is a SQL predicate re-applied inside the delete, so the statement decides
    /// on the rows as they are rather than on a list read earlier. Selecting candidates and then
    /// deleting them by id is two statements, and a condition checked only in the first can stop
    /// being true in between: the retention sweep's "no schedule ahead of it" is the case that
    /// matters, because a job created against a sub-agent child in that gap would be cascaded away
    /// with a parent nothing has locked. A `&'static str` and never anything derived from input;
    /// `""` for a caller with no further condition, which is `--all`.
    ///
    /// Chunked because a lock is an open file descriptor: a sweep over ten thousand expired
    /// sessions would otherwise hold ten thousand at once and hit the process limit, turning a
    /// housekeeping pass into a hard failure. The chunk is the unit of both locking and deleting,
    /// so no lock is held longer than its own statement.
    pub(super) async fn delete_the_unattached_among(
        &self,
        candidates: &[Uuid],
        still_eligible: &'static str,
        // The `updated_at` cutoff the candidates were selected under, re-applied inside the
        // delete for the same reason `still_eligible` is: a session resumed and released between
        // the select and the delete has a fresh `updated_at`, and the stale list still names it.
        updated_before: Option<&str>,
    ) -> Result<SessionSweep> {
        /// Sessions locked and deleted per statement. Well under SQLite's parameter ceiling
        /// (32,766) and well under any sane descriptor limit.
        const CHUNK: usize = 100;

        let mut sweep = SessionSweep::default();
        for chunk in candidates.chunks(CHUNK) {
            let mut held = Vec::with_capacity(chunk.len());
            for id in chunk {
                match self.lock_session(*id) {
                    Ok(lock) => held.push((*id, lock)),
                    // Not "someone has it" but "we could not ask", which is the same answer here:
                    // a session this process cannot establish a claim on is one it must not
                    // delete. Counted as attached so the caller still reports incomplete coverage.
                    Err(_) => sweep.attached_elsewhere += 1,
                }
            }
            if held.is_empty() {
                continue;
            }
            let ids: Vec<String> = held.iter().map(|(id, _)| id.to_string()).collect();
            let updated_before = updated_before.map(str::to_string);
            let deleted = self
                .connection
                .call(move |connection| -> rusqlite::Result<_> {
                    let placeholders = vec!["?"; ids.len()].join(",");
                    let cutoff_clause = match &updated_before {
                        Some(_) => format!(" AND updated_at < ?{}", ids.len() + 1),
                        None => String::new(),
                    };
                    let parameters: Vec<String> = ids
                        .iter()
                        .cloned()
                        .chain(updated_before.iter().cloned())
                        .collect();
                    let transaction = connection.transaction()?;
                    let deleted = transaction.execute(
                        &format!(
                            "DELETE FROM sessions WHERE id IN ({}){}{}",
                            placeholders,
                            match still_eligible {
                                "" => String::new(),
                                predicate => format!(" AND {predicate}"),
                            },
                            cutoff_clause
                        ),
                        rusqlite::params_from_iter(parameters.iter()),
                    )?;
                    super::blobs::sweep_unreferenced_blobs(&transaction)?;
                    transaction.commit()?;
                    Ok(deleted)
                })
                .await
                .map_err(|error| {
                    MekaError::Database(format!("failed to delete sessions: {error}"))
                })?;
            sweep.deleted += deleted as u64;
            // Before the sweep below, not after: `prune_orphan_lock_files` refuses to unlink a
            // file whose lock it cannot take, and this process is holding every one of these.
            drop(held);
        }
        self.prune_orphan_lock_files().await;
        Ok(sweep)
    }

    /// Move what a host may move on a session's row, in one statement, bumping `updated_at`.
    ///
    /// The one writer for every column a session carries once it exists: the level, the approvals
    /// switch, the working directory, the profile and the additional roots. Every door builds a
    /// [`SessionPatch`] and lands here, so the `updated_at` rule is stated once: any column that
    /// moves bumps it, roots included. Activating roots looks like how a session is being opened
    /// rather than activity in it, but a column that moves without the timestamp is invisible to
    /// every reader keyed on it (the idle sweep, `session/list` order, a client's change
    /// detection), and a door that wants no bump sends nothing, which is what every door does for a
    /// value the row already holds.
    ///
    /// `cwd` is stored as the path's `to_string_lossy()` form, UTF-8 being the only text SQLite
    /// has. An empty patch writes nothing. A patch for a row that is gone is
    /// [`MekaError::SessionNotFound`].
    ///
    /// **What a failed write costs is decided at the host**, once, in
    /// `crate::host::record_session_change`, and it differs by column. The level, the switch, the
    /// directory and the roots have already moved in the process that asked, so its door warns and
    /// continues rather than failing the user's command over a stale row. The profile is the
    /// billing record, so a door that cannot write it fails the request, or the session would
    /// run on an account its row does not name.
    pub(crate) async fn update_session(&self, session_id: Uuid, patch: SessionPatch) -> Result<()> {
        if patch.is_empty() {
            return Ok(());
        }
        let SessionPatch {
            permission,
            approvals,
            cwd,
            profile,
            roots,
        } = patch;
        let permission = permission.map(|level| level.to_string());
        let cwd = cwd.map(|path| path.to_string_lossy().into_owned());
        // `Some(None)` clears the column: the difference between "leave the roots" and "no roots".
        let roots = match roots {
            Some(roots) => Some(encode_additional_roots(&roots)?),
            None => None,
        };
        let id = session_id.to_string();
        let updated_at = chrono::Utc::now().to_rfc3339();
        let changed = self
            .connection
            .call(move |connection| -> rusqlite::Result<usize> {
                // Column names come from this list and never from a caller; only the values are
                // bound.
                let mut assignments = vec!["updated_at = :updated_at"];
                let mut params: Vec<(&str, &dyn rusqlite::ToSql)> =
                    vec![(":updated_at", &updated_at), (":id", &id)];
                if let Some(permission) = &permission {
                    assignments.push("permission = :permission");
                    params.push((":permission", permission));
                }
                if let Some(approvals) = &approvals {
                    assignments.push("approvals = :approvals");
                    params.push((":approvals", approvals));
                }
                if let Some(cwd) = &cwd {
                    assignments.push("cwd = :cwd");
                    params.push((":cwd", cwd));
                }
                if let Some(profile) = &profile {
                    assignments.push("profile = :profile");
                    params.push((":profile", profile));
                }
                if let Some(roots) = &roots {
                    assignments.push("additional_roots_json = :roots");
                    params.push((":roots", roots));
                }
                connection.execute(
                    &format!(
                        "UPDATE sessions SET {} WHERE id = :id",
                        assignments.join(", ")
                    ),
                    params.as_slice(),
                )
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to update session: {error}")))?;
        if changed == 0 {
            return Err(MekaError::SessionNotFound(session_id));
        }
        Ok(())
    }

    /// Erase a row's level, which no door does: the shape an older meka left behind, for the tests
    /// that need one.
    #[cfg(test)]
    pub(crate) async fn erase_recorded_permission(&self, session_id: Uuid) -> Result<()> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE sessions SET permission = NULL WHERE id = ?1",
                    rusqlite::params![session_id.to_string()],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to erase the level: {error}")))
    }

    /// Empty a session's event log and scratchpad, for tests that reuse a row.
    #[cfg(test)]
    pub(crate) async fn clear_messages(&self, session_id: Uuid) -> Result<()> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "DELETE FROM tool_outputs WHERE session_id = ?1",
                    rusqlite::params![session_id.to_string()],
                )?;

                connection.execute(
                    "DELETE FROM messages WHERE session_id = ?1",
                    rusqlite::params![session_id.to_string()],
                )?;

                connection.execute(
                    "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![chrono::Utc::now().to_rfc3339(), session_id.to_string()],
                )?;
                Ok(())
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to clear messages: {error}")))
    }

    /// Delete a session this caller already owns.
    ///
    /// Deliberately does *not* consult the session lock, because every caller here is holding it
    /// already or is acting on a row nothing can have locked: the HTTP handler evicting its own
    /// entry, the GC dropping a session it served, a sub-agent tool removing a child row it
    /// created moments ago. Taking the lock in those cases would *refuse* the caller its own
    /// session (`flock` is per open file description rather than per process, so a second
    /// descriptor contends with the first), and `try_write` is non-blocking, so what it produces
    /// is a spurious [`MekaError::SessionLocked`] rather than a hang.
    ///
    /// A caller acting on a session it has never met wants
    /// [`Self::delete_session_unless_attached`] instead.
    pub(crate) async fn delete_session(&self, session_id: Uuid) -> Result<bool> {
        let deleted = self.delete_session_row(session_id).await?;
        self.prune_orphan_lock_files().await;
        Ok(deleted)
    }

    /// The row half of a delete, without the lock-directory sweep, so a caller holding this
    /// session's lock can drop it before the sweep runs rather than blocking its own cleanup.
    pub(super) async fn delete_session_row(&self, session_id: Uuid) -> Result<bool> {
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                // ON DELETE CASCADE on `messages.session_id`, `tool_outputs.session_id`, and
                // `sessions.parent_session_id` sweeps own-session rows + any sub-agent children +
                // their messages/tool_outputs in a single statement.
                let transaction = connection.transaction()?;
                let deleted = transaction
                    .execute("DELETE FROM sessions WHERE id = ?1", rusqlite::params![
                        session_id.to_string()
                    ])?;
                // The cascade took the references; the bytes nothing else names go with them, in
                // the same transaction so a crash cannot leave them behind.
                super::blobs::sweep_unreferenced_blobs(&transaction)?;
                transaction.commit()?;
                Ok(deleted > 0)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to delete session: {error}")))
    }

    /// Delete a session, refusing with [`MekaError::SessionLocked`] if another meka process has it
    /// open. The door for a caller acting on a session it does not hold: `meka session delete`.
    ///
    /// Without it, `meka session delete <id>` against a live REPL exits 0 having said nothing at
    /// all (the count goes through `tracing::info!`, invisible at the default level) while the row
    /// and its messages cascade away underneath a conversation that carries on until its next turn
    /// fails on a foreign-key violation.
    pub(crate) async fn delete_session_unless_attached(&self, session_id: Uuid) -> Result<bool> {
        let lock = self.lock_session(session_id)?;
        let deleted = self.delete_session_row(session_id).await?;
        // Released before the sweep: it will not unlink a file it cannot lock, and that file is
        // this one.
        drop(lock);
        self.prune_orphan_lock_files().await;
        Ok(deleted)
    }

    /// Delete every session no other meka process has open, and report what was left behind.
    pub(crate) async fn delete_all_sessions(&self) -> Result<SessionSweep> {
        let ids: Vec<Uuid> = self
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare("SELECT id FROM sessions")?;
                let ids = statement
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<String>>>()?;
                Ok(ids)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to list sessions: {error}")))?
            .into_iter()
            .filter_map(|id| Uuid::parse_str(&id).ok())
            .collect();
        // No further condition: `--all` means every session nobody else has open, schedules
        // included. Sparing a job-owning session here would make the command quietly not mean what
        // it says.
        self.delete_the_unattached_among(&ids, "", None).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The regression this whole arrangement exists for: a session created on one profile must
    /// still name that profile afterwards, so resuming it does not silently move the conversation.
    #[tokio::test]
    async fn a_session_keeps_the_profile_it_was_created_on() {
        let store = Store::for_test().await;
        let id = store
            .create_session(None, "openaiprof".to_string())
            .await
            .expect("create");

        assert_eq!(
            store.recorded_profile(id).await.expect("read"),
            Some("openaiprof".to_string())
        );
    }

    /// Repinning is what every "change the provider" surface lands on, and it has to stick.
    #[tokio::test]
    async fn a_session_can_be_moved_to_another_profile() {
        let store = Store::for_test().await;
        let id = store
            .create_session(None, "openaiprof".to_string())
            .await
            .expect("create");

        store
            .update_session(id, SessionPatch {
                profile: Some("claudeprof".to_string()),
                ..Default::default()
            })
            .await
            .expect("repin");
        assert_eq!(
            store.recorded_profile(id).await.expect("read"),
            Some("claudeprof".to_string())
        );
    }

    /// A session that is gone reads as absent rather than as a profile named nothing, so a caller
    /// can tell "no such session" from "this session runs on ''".
    #[tokio::test]
    async fn a_missing_session_has_no_recorded_profile() {
        let store = Store::for_test().await;
        assert_eq!(
            store.recorded_profile(Uuid::new_v4()).await.expect("read"),
            None
        );
        assert!(
            matches!(
                store
                    .update_session(Uuid::new_v4(), SessionPatch {
                        profile: Some("anything".to_string()),
                        ..Default::default()
                    })
                    .await,
                Err(MekaError::SessionNotFound(_))
            ),
            "a repin that matched no row says so rather than reporting success"
        );
    }

    /// A child records the profile it was given, not the one on its parent's row.
    ///
    /// The two are the same on every ordinary path. They come apart for exactly as long as a repin
    /// that could not take the runtime lock, which is what ACP's `session/set_config_option` does
    /// mid-turn: the row moves, the agent does not, and a sub-agent spawned in that window ran on
    /// one account while its row claimed the other. Whoever calls this owes it the profile the
    /// sub-agent is actually built on; `agent_spawn` reads both from the same cell.
    ///
    /// Asserted with the two deliberately different, because passing the parent's own profile
    /// cannot tell the new behavior from the old.
    #[tokio::test]
    async fn a_child_session_records_the_profile_it_was_given_not_its_parents_row() {
        let store = Store::for_test().await;
        let parent = store
            .create_session(None, "openaiprof".to_string())
            .await
            .expect("parent");
        let (child, _lock) = store
            .create_child_session(
                parent,
                None,
                Vec::new(),
                None,
                "read".to_string(),
                "repinned-midturn".to_string(),
            )
            .await
            .expect("child");

        assert_eq!(
            store.recorded_profile(child).await.expect("read"),
            Some("repinned-midturn".to_string()),
            "the caller's profile, which is the one the sub-agent runs on"
        );
        assert_eq!(
            store.recorded_profile(parent).await.expect("read"),
            Some("openaiprof".to_string()),
            "and the parent's own row is left alone"
        );
    }

    /// Populate a session with the full spread of state a fork has to carry.
    async fn seeded_session(store: &Store) -> Uuid {
        use crate::conversation::{Event, Message};

        let id = store
            .create_session(
                Some(PathBuf::from("/work/main")),
                "test-profile".to_string(),
            )
            .await
            .expect("create session");
        // Enough alternating turns that a copy which reordered the conversation would show it
        // rather than coincidentally matching. This does not prove the `ORDER BY id` in
        // `fork_session_into` is load-bearing: SQLite's index scan yields rowid order anyway, so
        // the clause states the requirement rather than repairing a scramble.
        for turn in 0..20 {
            for event in [
                Event::Append(Message::user(format!("ask {turn}"))),
                Event::Append(Message::assistant_text(format!("reply {turn}"))),
            ] {
                store.save_event(id, &event).await.expect("save event");
            }
        }
        store
            .save_scratchpad_entry(id, "tool_1_output", "scratch")
            .await
            .expect("tool output");
        store
            .update_session(id, SessionPatch {
                roots: Some(vec![PathBuf::from("/work/shared")]),
                ..Default::default()
            })
            .await
            .expect("roots");
        store
            .save_session_stats(id, &crate::stats::SessionStatsSnapshot {
                turns: 3,
                input_tokens: 4242,
                ..Default::default()
            })
            .await
            .expect("stats");
        id
    }

    #[tokio::test]
    async fn fork_copies_the_conversation_and_leaves_the_source_alone() {
        let store = Store::for_test().await;
        let source = seeded_session(&store).await;

        let forked = store
            .fork_session_for_test(source, ForkOverrides::default())
            .await
            .expect("fork")
            .expect("source exists");
        assert_ne!(forked.id, source);

        let copy_events = store.load_events(forked.id).await.expect("copy events");
        let source_events = store.load_events(source).await.expect("source events");
        assert_eq!(
            serde_json::to_string(&copy_events).expect("serialize copy"),
            serde_json::to_string(&source_events).expect("serialize source"),
            "the copy starts from the source's exact conversation"
        );
        assert_eq!(
            store
                .load_all_scratchpad_entries(forked.id)
                .await
                .expect("copy outputs"),
            store
                .load_all_scratchpad_entries(source)
                .await
                .expect("source outputs"),
            "scratchpad entries are referenced by name from tool inputs, so they must travel"
        );

        let copy = store
            .session_info(forked.id)
            .await
            .expect("info")
            .expect("row");
        let original = store
            .session_info(source)
            .await
            .expect("info")
            .expect("row");
        assert_eq!(copy.cwd, original.cwd);
        assert_eq!(copy.additional_roots, original.additional_roots);
        assert_eq!(copy.title, original.title);
        let copy_stats = store
            .load_session_stats(forked.id)
            .await
            .expect("copy stats");
        assert_eq!(copy_stats.turns, 3);
        assert_eq!(copy_stats.input_tokens, 4242);
        assert!(
            copy.token_id.is_none(),
            "the bearer-token fingerprint is never inherited"
        );

        // The source is untouched: same event count, and still a listable root session.
        assert_eq!(
            store.load_events(source).await.expect("source").len(),
            40,
            "forking must not mutate the source"
        );
    }

    /// Regression: retention GC deletes by `updated_at` and runs at every agent startup, so a fork
    /// that inherited a stale timestamp was swept before its first turn.
    #[tokio::test]
    async fn fork_stamps_fresh_timestamps_so_retention_gc_spares_it() {
        let store = Store::for_test().await;
        let source = seeded_session(&store).await;

        let stale = (chrono::Utc::now() - chrono::TimeDelta::days(100)).to_rfc3339();
        let source_string = source.to_string();
        store
            .connection
            .call(move |connection| {
                connection.execute(
                    "UPDATE sessions SET created_at = ?1, updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![stale, source_string],
                )
            })
            .await
            .expect("age the source");

        let forked = store
            .fork_session_for_test(source, ForkOverrides::default())
            .await
            .expect("fork")
            .expect("source exists");

        let deleted = store
            .delete_expired_sessions(std::time::Duration::from_secs(90 * 86_400))
            .await
            .expect("retention sweep");
        assert_eq!(deleted.deleted, 1, "only the stale source is swept");
        assert!(
            store.session_exists(forked.id).await.expect("exists"),
            "the fork must survive the sweep that removes its stale source"
        );
    }

    /// A child links to its parent only through `parent_session_id`, and the sub-agent's *result*
    /// already sits in the parent's own event log, so the copy is self-contained without it.
    #[tokio::test]
    async fn fork_does_not_copy_sub_agent_children() {
        let store = Store::for_test().await;
        let source = seeded_session(&store).await;
        // Bound rather than discarded only because the tuple's second element is a `Result` that
        // must be used. Nothing here needs the child's lock: `fork_session_into` and
        // `load_session_tree` take none.
        let _child = store
            .create_child_session(
                source,
                None,
                Vec::new(),
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child");

        let forked = store
            .fork_session_for_test(source, ForkOverrides::default())
            .await
            .expect("fork")
            .expect("source exists");

        let tree = store.load_session_tree(forked.id).await.expect("tree");
        assert_eq!(tree.len(), 1, "the fork stands alone");
        assert_eq!(
            tree[0].parent_id, None,
            "a fork is a root session: `parent_session_id` means sub-agent parent, and list_sessions \
             hides rows that have one"
        );
        assert_eq!(
            store
                .load_session_tree(source)
                .await
                .expect("source tree")
                .len(),
            2,
            "the source keeps its child"
        );
    }

    #[tokio::test]
    async fn fork_overrides_replace_the_workspace() {
        let store = Store::for_test().await;
        let source = seeded_session(&store).await;

        let forked = store
            .fork_session_for_test(source, ForkOverrides {
                cwd: Some(PathBuf::from("/elsewhere")),
                additional_roots: Some(vec![PathBuf::from("/elsewhere/docs")]),
                token_id: Some("token-fingerprint".to_string()),
            })
            .await
            .expect("fork")
            .expect("source exists");

        let copy = store
            .session_info(forked.id)
            .await
            .expect("info")
            .expect("row");
        assert_eq!(copy.cwd, Some(PathBuf::from("/elsewhere")));
        assert_eq!(copy.additional_roots, vec![PathBuf::from(
            "/elsewhere/docs"
        )]);
        assert_eq!(copy.token_id.as_deref(), Some("token-fingerprint"));
    }

    /// `Some(vec![])` means "activate no additional roots", which is what ACP's fork request sends
    /// when `additionalDirectories` is omitted. It must not be confused with "inherit": both encode
    /// to SQL NULL, so a `COALESCE` alone would resurrect the source's roots.
    #[tokio::test]
    async fn fork_can_override_additional_roots_to_empty() {
        let store = Store::for_test().await;
        let source = seeded_session(&store).await;

        let forked = store
            .fork_session_for_test(source, ForkOverrides {
                additional_roots: Some(Vec::new()),
                ..Default::default()
            })
            .await
            .expect("fork")
            .expect("source exists");

        let copy = store
            .session_info(forked.id)
            .await
            .expect("info")
            .expect("row");
        assert!(
            copy.additional_roots.is_empty(),
            "an explicit empty override must narrow the workspace, not inherit it"
        );
        assert_eq!(
            store
                .session_info(source)
                .await
                .expect("info")
                .expect("row")
                .additional_roots,
            vec![PathBuf::from("/work/shared")],
            "and it must not disturb the source"
        );
    }

    /// Several clients forking one session at once must each get a whole, distinct copy. The copy
    /// spans three statements, so an interleaving that let a second fork observe the first's
    /// half-built session would show up as a short or empty event log.
    ///
    /// Every one of them probes the source, so this also proves that sibling forks wait behind one
    /// another's copy rather than reading each other's probe as another process mid-turn.
    #[tokio::test]
    async fn concurrent_forks_of_one_source_each_get_a_complete_copy() {
        let store = Store::for_test().await;
        let source = seeded_session(&store).await;
        let expected = store.load_events(source).await.expect("source").len();

        let mut handles = Vec::new();
        for _ in 0..8 {
            let store = store.clone();
            handles.push(tokio::spawn(async move {
                store
                    .fork_session_for_test(source, ForkOverrides::default())
                    .await
                    .expect("fork")
                    .expect("source exists")
                    .id
            }));
        }
        let mut ids = Vec::new();
        for handle in handles {
            ids.push(handle.await.expect("join"));
        }

        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(
            unique.len(),
            ids.len(),
            "every fork must be its own session"
        );
        for id in &ids {
            assert_eq!(
                store.load_events(*id).await.expect("copy").len(),
                expected,
                "every concurrent fork must carry the whole conversation"
            );
        }
    }

    /// A fork is a root session, not a child, so deleting what it was forked from must not
    /// take it with it. Had fork reused `parent_session_id` for lineage, the FK cascade would
    /// destroy every fork the moment its source was deleted.
    #[tokio::test]
    async fn deleting_the_source_leaves_the_fork_intact() {
        let store = Store::for_test().await;
        let source = seeded_session(&store).await;
        let expected = store.load_events(source).await.expect("source").len();

        let forked = store
            .fork_session_for_test(source, ForkOverrides::default())
            .await
            .expect("fork")
            .expect("source exists");
        assert!(store.delete_session(source).await.expect("delete"));

        assert!(store.session_exists(forked.id).await.expect("exists"));
        assert_eq!(
            store.load_events(forked.id).await.expect("copy").len(),
            expected,
            "the fork keeps its own copy of the conversation"
        );
        assert_eq!(
            store
                .load_all_scratchpad_entries(forked.id)
                .await
                .expect("outputs")
                .len(),
            1,
        );
    }

    /// A fork's copy of an image is its own: the reference rows travel with the copied messages,
    /// so deleting the source does not sweep the bytes, and the fork reads them under its own id.
    #[tokio::test]
    async fn a_fork_keeps_its_images_when_the_source_is_deleted() {
        use crate::{
            conversation::{Event, Message},
            image::ImageSource,
        };

        let store = Store::for_test().await;
        let source = seeded_session(&store).await;
        store
            .save_event(
                source,
                &Event::Append(Message::user_with_images("look", vec![
                    ImageSource::Base64 {
                        media_type: "image/png".to_string(),
                        data: "aGVsbG8=".to_string(),
                    },
                ])),
            )
            .await
            .expect("save");
        let hash = blobs::content_hash(b"hello");

        let forked = store
            .fork_session_for_test(source, ForkOverrides::default())
            .await
            .expect("fork")
            .expect("source exists");
        assert!(store.delete_session(source).await.expect("delete"));

        assert!(
            store
                .load_session_blob(forked.id, &hash)
                .await
                .expect("load")
                .is_some(),
            "the fork references the blob under its own message rows"
        );
        let mut events = store.load_events(forked.id).await.expect("events");
        store.inline_blobs(&mut events).await.expect("inline");
        let last = events.last().expect("the copied image turn");
        assert!(
            serde_json::to_string(last)
                .expect("serialize")
                .contains("aGVsbG8="),
            "and the bytes come back whole after the source is gone"
        );
    }

    /// Deleting a fork must sweep the rows it copied. `discard_failed_fork` in the ACP handler
    /// relies on this to undo a fork whose runtime wouldn't build; if the cascade missed, a failed
    /// fork would leave an orphaned transcript with no session row pointing at it.
    #[tokio::test]
    async fn deleting_a_fork_removes_the_rows_it_copied() {
        let store = Store::for_test().await;
        let source = seeded_session(&store).await;
        let forked = store
            .fork_session_for_test(source, ForkOverrides::default())
            .await
            .expect("fork")
            .expect("source exists");

        assert!(store.delete_session(forked.id).await.expect("delete"));
        assert!(
            store
                .load_events(forked.id)
                .await
                .expect("events")
                .is_empty(),
            "the copied event log must go with the row"
        );
        assert!(
            store
                .load_all_scratchpad_entries(forked.id)
                .await
                .expect("outputs")
                .is_empty(),
            "and so must the copied scratchpad entries"
        );
        // The source is untouched by the fork's deletion.
        assert!(!store.load_events(source).await.expect("source").is_empty());
    }

    /// Forking a sub-agent gives another sub-agent, not a laundered root session.
    ///
    /// A fork writing `NULL` into `parent_session_id` and omitting `subagent_spec_json` leaves a
    /// copy that reads as a root session, which `crate::host::refuse_a_spawned_session` then
    /// admits. That handed anyone with `sessions:w` -- or a shell, via `meka session fork` -- a
    /// drivable copy of a sub-agent's whole conversation with no `[subagents]` denials, no memory
    /// or instruction grants, and the host's permission, which is the escalation the refusal
    /// exists to stop. The refusal is a boundary only while a copy cannot cross it.
    ///
    /// Both arms, because carrying the columns unconditionally would be its own bug: a fork of an
    /// ordinary session must stay a root session or `list_sessions` would hide every fork. `-c`
    /// resumes the newest session it can actually drive, not merely the newest row.
    ///
    /// A sub-agent's row is touched by its own turns, so a sub-agent still running when its
    /// parent's turn ends -- which `agent_spawn`'s `background` parameter makes ordinary --
    /// sorts above the session the user was last in. Picking it would dead-end on
    /// `crate::host::refuse_a_spawned_session`'s refusal, with no way to ask for the next one down.
    ///
    /// The child is created second so it is unambiguously the newer row: without the filter this
    /// returns it, which is the whole failure.
    #[tokio::test]
    async fn the_last_session_is_the_newest_one_a_host_can_drive() {
        let store = Store::for_test().await;
        let parent = store
            .create_session(None, "profile".to_string())
            .await
            .expect("a root session");
        let (sub_agent, _lock) = store
            .create_child_session(
                parent,
                None,
                Vec::new(),
                None,
                "read".to_string(),
                "profile".to_string(),
            )
            .await
            .expect("a sub-agent, written after its parent and so the newer row");
        assert_ne!(parent, sub_agent);

        assert_eq!(
            store.last_session_id().await.expect("read"),
            Some(parent),
            "a sub-agent is not resumable, so offering it to `-c` is offering a dead end"
        );
    }

    /// A rename moves the rows and the pinned spawn terms together, and only those: unpinned terms
    /// gain no `profile`, and a root's absent terms stay absent.
    #[tokio::test]
    async fn renaming_a_profile_moves_rows_and_pinned_specs_together() {
        let store = Store::for_test().await;
        let root = store
            .create_session(None, "old".to_string())
            .await
            .expect("root");
        let (pinned, _lock) = store
            .create_child_session(
                root,
                None,
                Vec::new(),
                Some(r#"{"profile":"old"}"#.to_string()),
                "read".to_string(),
                "old".to_string(),
            )
            .await
            .expect("pinned");
        let (unpinned, _lock) = store
            .create_child_session(
                root,
                None,
                Vec::new(),
                Some("{}".to_string()),
                "read".to_string(),
                "old".to_string(),
            )
            .await
            .expect("unpinned");
        let other = store
            .create_session(None, "other".to_string())
            .await
            .expect("other");
        // What an import copies verbatim from an archive somebody edited: not JSON at all.
        let (malformed, _lock) = store
            .create_child_session(
                root,
                None,
                Vec::new(),
                Some("not json".to_string()),
                "read".to_string(),
                "old".to_string(),
            )
            .await
            .expect("malformed");

        let moved = store.rename_profile("old", "new").await.expect("rename");

        assert_eq!(moved, 4, "the root and all three children moved");
        assert_eq!(
            store
                .load_subagent_spec(malformed)
                .await
                .expect("read")
                .as_deref(),
            Some("not json"),
            "a spec that is not JSON neither blocks the rename nor changes"
        );
        for id in [root, pinned, unpinned, malformed] {
            assert_eq!(
                store.recorded_profile(id).await.expect("read").as_deref(),
                Some("new")
            );
        }
        assert_eq!(
            store
                .recorded_profile(other)
                .await
                .expect("read")
                .as_deref(),
            Some("other")
        );
        assert_eq!(
            store
                .load_subagent_spec(pinned)
                .await
                .expect("read")
                .as_deref(),
            Some(r#"{"profile":"new"}"#)
        );
        assert_eq!(
            store
                .load_subagent_spec(unpinned)
                .await
                .expect("read")
                .as_deref(),
            Some("{}")
        );
        assert_eq!(store.load_subagent_spec(root).await.expect("read"), None);
    }

    /// The rows `-c` and `session list` skip are the rows the refusal declines, not a near-miss.
    ///
    /// A sub-agent exported alone and re-imported keeps `subagent_spec_json` and loses its parent
    /// link, and `refuse_a_spawned_session` reads that row as a sub-agent -- correctly, since
    /// nothing else can reconstruct the tools and level it ran under. A filter keyed on the
    /// parent alone therefore disagreed with the refusal about the same row, which is worse
    /// than either answer: `-c` offered the orphan, printed `Resuming session:`, and then
    /// declined it, *permanently*, because no later session can outrank a row that is always
    /// the newest thing in the store. The list did the mirror of it, presenting the orphan as a
    /// root session while the refusal's own text told the reader to pass `--include-children`
    /// to see it.
    ///
    /// Written through the connection rather than by exporting and importing, because what is
    /// under test is the shape, not the route to it: [`Store::spawn_terms`] and
    /// [`spawned_session_sql`] have to agree about a row holding spawn terms and no parent,
    /// however it came to exist.
    #[tokio::test]
    async fn an_imported_worker_is_a_worker_to_every_filter() {
        let store = Store::for_test().await;
        let drivable = store
            .create_session(None, "profile".to_string())
            .await
            .expect("a root session");
        let orphan = store
            .create_session(None, "profile".to_string())
            .await
            .expect("a second row, newer, which the import will turn into a sub-agent");
        store
            .connection
            .call(move |connection| {
                connection.execute(
                    "UPDATE sessions SET subagent_spec_json = ?2 WHERE id = ?1",
                    rusqlite::params![orphan.to_string(), "{\"tools\":[]}"],
                )
            })
            .await
            .expect("plant the spawn terms an import would copy");

        assert_eq!(
            store
                .spawn_terms(orphan)
                .await
                .expect("read")
                .map(|terms| terms.parent),
            Some(None),
            "spawn terms with no parent are still spawn terms"
        );
        assert!(
            store.spawn_terms(drivable).await.expect("read").is_none(),
            "and an ordinary session must not be caught by the same rule"
        );

        assert_eq!(
            store.last_session_id().await.expect("read"),
            Some(drivable),
            "`-c` has to skip the orphan and reach the session it can actually drive"
        );

        let (listed, _) = store
            .list_sessions(50, false, None, None)
            .await
            .expect("list");
        let ids: Vec<Uuid> = listed.iter().map(|row| row.id).collect();
        assert!(
            ids.contains(&drivable) && !ids.contains(&orphan),
            "the default listing is the drivable ones: {ids:?}"
        );
        let (with_children, _) = store
            .list_sessions(50, true, None, None)
            .await
            .expect("list");
        assert!(
            with_children.iter().any(|row| row.id == orphan),
            "and `--include-children` is where the refusal says to look for it"
        );
    }

    #[tokio::test]
    async fn a_fork_of_a_worker_is_a_worker() {
        let store = Store::for_test().await;
        let parent = store
            .create_session(None, "profile".to_string())
            .await
            .expect("a root session");
        let (sub_agent, lock) = store
            .create_child_session(
                parent,
                None,
                Vec::new(),
                Some("{\"tools\":[]}".to_string()),
                "read".to_string(),
                "profile".to_string(),
            )
            .await
            .expect("a sub-agent of that session");
        // Released, as it is once the parent stops running the sub-agent: a fork probes its source,
        // and a sub-agent its parent is still driving is refused like any session being written.
        drop(lock);

        let forked_worker = store
            .fork_session_for_test(sub_agent, ForkOverrides::default())
            .await
            .expect("fork")
            .expect("the sub-agent exists");
        let info = store
            .session_info(forked_worker.id)
            .await
            .expect("read")
            .expect("the copy exists");
        assert_eq!(
            info.parent_id,
            Some(parent),
            "a fork of a sub-agent is a sibling under the same parent, which is what keeps it \
             undrivable by a host"
        );
        assert_eq!(
            store
                .load_subagent_spec(forked_worker.id)
                .await
                .expect("read the copy's spec"),
            Some("{\"tools\":[]}".to_string()),
            "and it carries the terms it was spawned under, or `agent_followup` would rebuild it \
             as something it never was"
        );

        let forked_root = store
            .fork_session_for_test(parent, ForkOverrides::default())
            .await
            .expect("fork")
            .expect("the parent exists");
        assert_eq!(
            store
                .session_info(forked_root.id)
                .await
                .expect("read")
                .expect("the copy exists")
                .parent_id,
            None,
            "a fork of an ordinary session stays a root session; inventing a parent would hide every \
             fork from `meka session list`"
        );
    }

    #[tokio::test]
    async fn fork_of_an_unknown_session_is_none() {
        let store = Store::for_test().await;
        assert!(
            store
                .fork_session_for_test(Uuid::new_v4(), ForkOverrides::default())
                .await
                .expect("fork")
                .is_none(),
            "callers map None to their own not-found shape rather than parsing an error"
        );
    }

    /// Drift guard for [`Store::fork_session_locked`], whose `INSERT ... SELECT` names every
    /// column explicitly. A new column silently omitted there would be dropped from every fork,
    /// which is exactly how `additional_roots` came to be lost by export/import. If this fails,
    /// decide whether the new column should be copied, reset, or overridden, then update both the
    /// fork statement and this list.
    #[tokio::test]
    async fn fork_copies_every_session_column() {
        let store = Store::for_test().await;
        let columns = store
            .connection
            .call(|connection| {
                let mut statement =
                    connection.prepare("SELECT name FROM pragma_table_info('sessions')")?;
                let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
                rows.collect::<rusqlite::Result<Vec<String>>>()
            })
            .await
            .expect("read schema");

        assert_eq!(columns, vec![
            "id",
            "created_at",
            "updated_at",
            "parent_session_id",
            "cwd",
            "permission",
            "capabilities_json",
            "token_id",
            "additional_roots_json",
            // Copied by a fork, together with `parent_session_id`: a fork of a sub-agent is a
            // sibling under the same parent, and a copy that dropped either would be a drivable
            // sub-agent with no spawn terms. See `fork_session_locked`'s doc comment.
            "subagent_spec_json",
            "stat_turns",
            "stat_input_tokens",
            "stat_output_tokens",
            "stat_cache_creation_input_tokens",
            "stat_cache_read_input_tokens",
            "stat_redactions",
            "stat_redacted_images",
            "stat_redacted_bytes",
            // Last, and after the stats, because a migration appended it and the fresh path
            // replays that same step rather than creating the column inline, so both
            // orders agree.
            //
            // Copied by a fork: a fork continues the same conversation and so has to keep running
            // on the same thing. Resetting it would make forking one more door that switches
            // profile without saying so.
            "profile",
            // Copied by a fork for the same reason `permission` is: the copy continues under the
            // terms the source was running on.
            "approvals",
        ]);
    }

    /// The sibling of `fork_copies_every_session_column`, for the door that had no such guard and
    /// was therefore the one that forgot: `import_sessions` wrote no profile at all, so every
    /// imported session landed on the empty profile no configuration can name, and the resume hint
    /// the command printed named a session that could not resume.
    #[tokio::test]
    async fn import_writes_the_sessions_profile() {
        let store = Store::for_test().await;
        let id = Uuid::new_v4();
        store
            .import_sessions(
                vec![ImportSessionRecord {
                    new_id: id,
                    new_parent_id: None,
                    created_at: chrono::Utc::now().to_rfc3339(),
                    cwd: None,
                    permission: crate::permission::Permission::Read,
                    approvals: false,
                    capabilities_json: None,
                    additional_roots: Vec::new(),
                    subagent_spec_json: None,
                    profile: "work".to_string(),
                    stats: crate::stats::SessionStatsSnapshot::default(),
                    events: Vec::new(),
                    tool_outputs: Vec::new(),
                }],
                Vec::new(),
            )
            .await
            .expect("import");

        assert_eq!(
            store.recorded_profile(id).await.expect("read"),
            Some("work".to_string())
        );
    }

    /// A `PATCH /v1/sessions/{id}` naming a profile moves the row in the same statement that
    /// carries permission and cwd, so a client that changed all three cannot end up with a session
    /// that took some of them.
    #[tokio::test]
    async fn moving_a_session_atomically_rewrites_its_profile() {
        let store = Store::for_test().await;
        let created = store
            .create_session(Some(PathBuf::from("/work/old")), "alpha".to_string())
            .await
            .expect("create");

        store
            .update_session(created, SessionPatch {
                permission: Some(crate::permission::Permission::Workspace),
                approvals: Some(true),
                cwd: Some(PathBuf::from("/work/new")),
                profile: Some("beta".to_string()),
                roots: None,
            })
            .await
            .expect("move the session");

        let row = store
            .session_info(created)
            .await
            .expect("info")
            .expect("row");
        assert_eq!(row.profile, "beta");
        assert_eq!(
            row.permission,
            Some(crate::permission::Permission::Workspace)
        );
        assert!(row.approvals);
        assert_eq!(row.cwd, Some(PathBuf::from("/work/new")));
    }

    /// Every column the one writer moves bumps `updated_at`, roots included.
    ///
    /// Roots were the exception, on the theory that activating them is how a session is opened
    /// rather than activity in it. A column that moves without the timestamp is invisible to every
    /// reader keyed on it, and a door that wants no bump sends nothing.
    #[tokio::test]
    async fn every_patched_column_bumps_updated_at_roots_included() {
        let store = Store::for_test().await;
        let id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let aged = "2020-01-01T00:00:00+00:00";
        for patch in [
            SessionPatch {
                roots: Some(vec![PathBuf::from("/work/shared")]),
                ..Default::default()
            },
            SessionPatch {
                permission: Some(crate::permission::Permission::Unrestricted),
                ..Default::default()
            },
            SessionPatch {
                approvals: Some(true),
                ..Default::default()
            },
            SessionPatch {
                cwd: Some(PathBuf::from("/work/elsewhere")),
                ..Default::default()
            },
            SessionPatch {
                profile: Some("other".to_string()),
                ..Default::default()
            },
        ] {
            store
                .set_session_updated_at_for_test(id, aged)
                .await
                .expect("age the row");
            let described = patch.to_string();
            store.update_session(id, patch).await.expect("patch");
            let row = store.session_info(id).await.expect("info").expect("row");
            assert_ne!(
                row.updated_at, aged,
                "writing {described} must move updated_at"
            );
        }

        // And nothing at all leaves it alone.
        store
            .set_session_updated_at_for_test(id, aged)
            .await
            .expect("age the row");
        store
            .update_session(id, SessionPatch::default())
            .await
            .expect("an empty patch");
        let row = store.session_info(id).await.expect("info").expect("row");
        assert_eq!(row.updated_at, aged, "an empty patch is not a write");
    }

    #[tokio::test]
    async fn sessions_on_a_profile_are_counted_for_the_removal_warning() {
        let store = Store::for_test().await;
        for profile in ["work", "work", "side"] {
            store
                .create_session(None, profile.to_string())
                .await
                .expect("create");
        }
        assert_eq!(
            store
                .count_sessions_on_profile("work")
                .await
                .expect("count"),
            2
        );
        assert_eq!(
            store
                .count_sessions_on_profile("gone")
                .await
                .expect("count"),
            0
        );
    }

    /// The count is what `meka profile remove` warns with, beside advice that only applies to a
    /// root session. A sub-agent row copies its parent's binding, so counting children made
    /// the warning cite a number many times what `meka session list` shows.
    #[tokio::test]
    async fn a_profile_count_leaves_out_the_workers_a_session_spawned() {
        let store = Store::for_test().await;
        let parent = store
            .create_session(None, "work".to_string())
            .await
            .expect("create parent");
        for _ in 0..3 {
            // On the parent's own profile, which is what makes this test mean anything: created on
            // a different one, `provider = ?1` excludes them by itself and the spawn filter is
            // never consulted. It said "on the parent's profile" while doing the opposite, so
            // deleting the filter outright left the suite green.
            let (_id, lock) = store
                .create_child_session(
                    parent,
                    None,
                    Vec::new(),
                    None,
                    "read".to_string(),
                    "work".to_string(),
                )
                .await
                .expect("spawn a sub-agent");
            lock.expect("claim the sub-agent's lock");
        }
        // And an imported sub-agent, which holds spawn terms and no parent link. A parent-only
        // filter counts this row, and it is one a user cannot act on: `profile remove`
        // warns how many sessions it would strand, and this is not one of them.
        let orphan = store
            .create_session(None, "work".to_string())
            .await
            .expect("create the row an import would write");
        store
            .connection
            .call(move |connection| {
                connection.execute(
                    "UPDATE sessions SET subagent_spec_json = ?2 WHERE id = ?1",
                    rusqlite::params![orphan.to_string(), "{\"tools\":[]}"],
                )
            })
            .await
            .expect("plant the spawn terms an import would copy");

        assert_eq!(
            store
                .count_sessions_on_profile("work")
                .await
                .expect("count"),
            1,
            "sub-agents on the parent's profile are not more sessions to move, however they got here"
        );
    }

    /// A spawn against a parent that is gone must fail, not hand back an id with no row behind it.
    ///
    /// The statement is an `INSERT … SELECT` even though the child is now *told* its provider
    /// rather than copying it, and this is why: that form selects nothing and succeeds where the
    /// `VALUES` it replaced was refused by `parent_session_id`'s foreign key. Unchecked, the model
    /// is told about a sub-agent
    /// that does not exist, a lock file is held for it, and the failure resurfaces as a raw
    /// constraint violation on the sub-agent's first saved message.
    #[tokio::test]
    async fn spawning_from_a_session_that_is_gone_is_refused() {
        let store = Store::for_test().await;
        let Err(error) = store
            .create_child_session(
                Uuid::new_v4(),
                None,
                Vec::new(),
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
        else {
            panic!("a missing parent must refuse the spawn");
        };
        assert!(
            error.to_string().contains("no longer exists"),
            "the refusal must name what went wrong: {error}"
        );
    }

    #[tokio::test]
    async fn additional_roots_round_trip_through_the_database() {
        let store = Store::for_test().await;
        let id = store
            .create_session(
                Some(PathBuf::from("/work/main")),
                "test-profile".to_string(),
            )
            .await
            .expect("create session");

        // A fresh session has none: the column starts NULL.
        let summary = store.session_info(id).await.expect("info").expect("row");
        assert!(summary.additional_roots.is_empty());

        let roots = vec![PathBuf::from("/work/shared"), PathBuf::from("/work/docs")];
        store
            .update_session(id, SessionPatch {
                roots: Some(roots.clone()),
                ..Default::default()
            })
            .await
            .expect("store roots");
        let summary = store.session_info(id).await.expect("info").expect("row");
        assert_eq!(summary.additional_roots, roots);

        // `session/list` reports the same set, since that is what a client rebuilds a workspace
        // from when picking a session out of its history.
        let (listed, _cursor) = store
            .list_sessions(10, false, None, None)
            .await
            .expect("list");
        let row = listed
            .iter()
            .find(|row| row.id == id)
            .expect("session should be listed");
        assert_eq!(row.additional_roots, roots);
    }

    /// Load and resume carry the complete resulting list, so an empty one clears rather than
    /// merges: reopening a session from a window that no longer has the second folder has to
    /// narrow the session, not silently keep searching a folder the user removed.
    #[tokio::test]
    async fn empty_additional_roots_clears_the_stored_list() {
        let store = Store::for_test().await;
        let id = store
            .create_session(
                Some(PathBuf::from("/work/main")),
                "test-profile".to_string(),
            )
            .await
            .expect("create session");
        store
            .update_session(id, SessionPatch {
                roots: Some(vec![PathBuf::from("/work/shared")]),
                ..Default::default()
            })
            .await
            .expect("store roots");

        store
            .update_session(id, SessionPatch {
                roots: Some(Vec::new()),
                ..Default::default()
            })
            .await
            .expect("clear roots");

        let summary = store.session_info(id).await.expect("info").expect("row");
        assert!(
            summary.additional_roots.is_empty(),
            "an empty list must clear, not merge"
        );
    }

    /// Unparseable JSON must not make a session unloadable. Like NULL it means a single root, which
    /// is what such a session is anyway.
    #[test]
    fn decode_additional_roots_fails_soft() {
        assert!(decode_additional_roots(None).is_empty());
        assert!(decode_additional_roots(Some("not json")).is_empty());
        assert_eq!(decode_additional_roots(Some(r#"["/a","/b"]"#)), vec![
            PathBuf::from("/a"),
            PathBuf::from("/b"),
        ]);
    }

    /// Persist one of every event variant via `save_event` and read it back through `load_events`.
    /// Verifies the encoding/decoding round trip, including the JSON envelope used for
    /// `CompactBoundary`, matches the in-memory shape.
    #[tokio::test]
    async fn save_and_load_events_round_trip() {
        use std::collections::HashSet;

        use crate::conversation::{ContentBlock, Event, Message, Role, ToolResultContent};

        let store = Store::for_test().await;
        let sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");

        let user_event = Event::Append(Message::user("hello"));
        let assistant_event = Event::Append(Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "thinking aloud".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "u1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/x"}),
                },
            ],
        });
        let tool_result_event = Event::Append(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "u1".to_string(),
                content: vec![ToolResultContent::Text {
                    text: "ok".to_string(),
                }],
                is_error: false,
            }],
        });
        let snapshot: HashSet<String> = ["mcp__notion__fetch".to_string()].into_iter().collect();
        let boundary_event = Event::CompactBoundary {
            summary: Message::user("[summary]"),
            replaced_count: 3,
            loaded_tools_snapshot: snapshot,
        };

        let repair_event = Event::Repair {
            replaced_count: 2,
            messages: vec![Message::assistant_text("[degraded]")],
        };

        for event in [
            &user_event,
            &assistant_event,
            &tool_result_event,
            &boundary_event,
            &repair_event,
        ] {
            store.save_event(sid, event).await.expect("save event");
        }

        let loaded = store.load_events(sid).await.expect("load events");
        assert_eq!(loaded.len(), 5);

        match &loaded[0] {
            Event::Append(m) => assert_eq!(m.text_content(), "hello"),
            _ => panic!("expected user Append"),
        }
        match &loaded[1] {
            Event::Append(m) => {
                assert_eq!(m.role, Role::Assistant);
                assert_eq!(m.content.len(), 2);
                assert!(matches!(&m.content[1], ContentBlock::ToolUse { id, .. } if id == "u1"));
            }
            _ => panic!("expected assistant Append"),
        }
        match &loaded[2] {
            Event::Append(m) => {
                assert_eq!(m.role, Role::User);
                assert!(matches!(
                    &m.content[0],
                    ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "u1"
                ));
            }
            _ => panic!("expected tool_results Append"),
        }
        match &loaded[3] {
            Event::CompactBoundary {
                replaced_count,
                loaded_tools_snapshot,
                summary,
            } => {
                assert_eq!(*replaced_count, 3);
                assert!(loaded_tools_snapshot.contains("mcp__notion__fetch"));
                assert_eq!(summary.text_content(), "[summary]");
            }
            _ => panic!("expected CompactBoundary"),
        }
        match &loaded[4] {
            Event::Repair {
                replaced_count,
                messages,
            } => {
                assert_eq!(*replaced_count, 2);
                assert_eq!(messages.len(), 1);
                assert_eq!(messages[0].text_content(), "[degraded]");
            }
            _ => panic!("expected Repair"),
        }
    }

    /// A user turn carrying an input image is persisted under the `user_blocks` role as full JSON
    /// so the image survives the round trip, while a text-only user turn still stores as plaintext
    /// under `user` (keeping `list_sessions`'s title intact).
    #[tokio::test]
    async fn user_input_image_round_trips_via_user_blocks_role() {
        use crate::{
            conversation::{ContentBlock, Event, Message, Role},
            image::ImageSource,
        };

        let store = Store::for_test().await;
        let sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");

        let image = ImageSource::Base64 {
            media_type: "image/png".to_string(),
            data: "aGVsbG8=".to_string(),
        };
        let image_event = Event::Append(Message::user_with_images("look at this", vec![image]));
        let text_event = Event::Append(Message::user("plain text only"));
        store
            .save_event(sid, &image_event)
            .await
            .expect("save image event");
        store
            .save_event(sid, &text_event)
            .await
            .expect("save text event");

        // Storage roles: the image-bearing turn is JSON under `user_blocks`; the text-only turn
        // stays plaintext under `user`.
        let rows = store.load_messages(sid).await.expect("load messages");
        assert_eq!(rows[0].role, "user_blocks");
        assert_eq!(rows[1].role, "user");
        assert_eq!(rows[1].content, "plain text only");

        // The row holds a reference rather than the bytes, and the bytes come back on hydration.
        assert!(
            rows[0].content.contains("\"type\":\"blob\""),
            "the image should rest as a blob reference: {}",
            rows[0].content
        );
        let mut loaded = store.load_events(sid).await.expect("load events");
        store
            .inline_blobs(&mut loaded)
            .await
            .expect("inline the blob");
        match &loaded[0] {
            Event::Append(message) => {
                assert_eq!(message.role, Role::User);
                assert_eq!(message.content.len(), 2);
                assert!(matches!(
                    &message.content[0],
                    ContentBlock::Text { text } if text == "look at this"
                ));
                assert!(matches!(
                    &message.content[1],
                    ContentBlock::Image { source }
                        if source.base64_data() == Some("aGVsbG8=")
                            && source.media_type() == "image/png"
                ));
            }
            other => panic!("expected user Append, got {other:?}"),
        }
        match &loaded[1] {
            Event::Append(message) => assert_eq!(message.text_content(), "plain text only"),
            other => panic!("expected user Append, got {other:?}"),
        }
    }

    /// The plain `user` and `assistant` roles are what `encode_event_for_db` writes for an
    /// `Event::Append`, so `load_events` must hand every such row back as one: this is the primary
    /// live path through the decoder, not a fallback.
    #[tokio::test]
    async fn load_events_decodes_stored_append_rows() {
        use crate::conversation::Event;

        let store = Store::for_test().await;
        let sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");

        // Written by hand rather than through `save_event`, so what is under test is the decoder
        // alone: nothing here would notice the encoder changing which role it writes.
        // `user_input_image_round_trips_via_user_blocks_role` is the test that closes that
        // loop.
        store
            .save_message(sid, "user", "first")
            .await
            .expect("save user");
        let assistant_blocks = serde_json::json!([
            {"type": "text", "text": "answer"}
        ])
        .to_string();
        store
            .save_message(sid, "assistant", &assistant_blocks)
            .await
            .expect("save assistant");

        let events = store.load_events(sid).await.expect("load events");
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|e| matches!(e, Event::Append(_))));
    }

    /// A row with an unknown role should be skipped (with a warning) so a future schema bump that
    /// adds new event variants doesn't crash older binaries reading newer DBs.
    #[tokio::test]
    async fn load_events_skips_unknown_role() {
        let store = Store::for_test().await;
        let sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");
        store
            .save_message(sid, "user", "real")
            .await
            .expect("save real row");
        store
            .save_message(sid, "future_event_kind", "{}")
            .await
            .expect("save unknown row");
        let events = store.load_events(sid).await.expect("load events");
        assert_eq!(events.len(), 1);
    }

    #[tokio::test]
    async fn a_created_session_exists() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed to create session");
        assert!(
            store
                .session_exists(session_id)
                .await
                .expect("failed to check")
        );
    }

    #[tokio::test]
    async fn session_stats_persist_round_trip() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create session");

        // A fresh row starts at all-zero (columns default to 0).
        let fresh = store
            .load_session_stats(session_id)
            .await
            .expect("load fresh");
        assert_eq!(fresh.turns, 0);
        assert_eq!(fresh.input_tokens, 0);

        let snapshot = crate::stats::SessionStatsSnapshot {
            turns: 5,
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 10,
            cache_read_input_tokens: 200,
            redactions: 2,
            redacted_images: 3,
            redacted_bytes: 4096,
        };
        store
            .save_session_stats(session_id, &snapshot)
            .await
            .expect("save stats");

        let loaded = store
            .load_session_stats(session_id)
            .await
            .expect("load stats");
        assert_eq!(loaded.turns, 5);
        assert_eq!(loaded.input_tokens, 100);
        assert_eq!(loaded.output_tokens, 50);
        assert_eq!(loaded.cache_creation_input_tokens, 10);
        assert_eq!(loaded.cache_read_input_tokens, 200);
        assert_eq!(loaded.redactions, 2);
        assert_eq!(loaded.redacted_images, 3);
        assert_eq!(loaded.redacted_bytes, 4096);

        // An unknown session id is not an error; it reads as all-zero.
        let unknown = store
            .load_session_stats(uuid::Uuid::new_v4())
            .await
            .expect("load unknown");
        assert_eq!(unknown.turns, 0);
    }

    #[tokio::test]
    async fn save_and_load_messages() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed to create session");

        store
            .save_message(session_id, "user", "hello")
            .await
            .expect("failed to save message");
        store
            .save_message(session_id, "assistant", "hi there")
            .await
            .expect("failed to save message");

        let messages = store
            .load_messages(session_id)
            .await
            .expect("failed to load messages");

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "hello");
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[1].content, "hi there");
    }

    #[tokio::test]
    async fn the_last_session_id_is_none_until_a_session_exists() {
        let store = Store::for_test().await;
        assert!(store.last_session_id().await.expect("failed").is_none());

        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed to create session");
        let last = store
            .last_session_id()
            .await
            .expect("failed to get last session");
        assert_eq!(last, Some(session_id));
    }

    #[tokio::test]
    async fn find_sessions_by_prefix_empty_db() {
        let store = Store::for_test().await;
        let matches = store
            .find_sessions_by_prefix("abc")
            .await
            .expect("failed prefix lookup");
        assert!(matches.is_empty());
    }

    #[tokio::test]
    async fn find_sessions_by_prefix_unique_match() {
        let store = Store::for_test().await;
        let id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed to create session");
        // First 8 hex chars (before the first dash), guaranteed unique for a freshly-generated
        // random UUID with only one row in the DB.
        let prefix: String = id.to_string().chars().take(8).collect();
        let matches = store
            .find_sessions_by_prefix(&prefix)
            .await
            .expect("failed prefix lookup");
        assert_eq!(matches, vec![id]);
    }

    #[tokio::test]
    async fn find_sessions_by_prefix_no_match() {
        let store = Store::for_test().await;
        store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed to create session");
        let matches = store
            .find_sessions_by_prefix("ffffffff")
            .await
            .expect("failed prefix lookup");
        // Real UUIDs are random; collision with this prefix is astronomically unlikely but
        // theoretically possible; re-create a session if so.
        assert!(matches.is_empty() || matches.len() == 1);
    }

    #[tokio::test]
    async fn find_sessions_by_prefix_rejects_non_hex_chars() {
        let store = Store::for_test().await;
        store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed to create session");
        // SQL `%` and `_` wildcards must not slip through as prefix chars.
        for bad in ["%", "_", "abc%", "ab_c", "g0g0", "x123"] {
            let matches = store
                .find_sessions_by_prefix(bad)
                .await
                .expect("failed prefix lookup");
            assert!(
                matches.is_empty(),
                "non-hex prefix {bad:?} should match nothing"
            );
        }
    }

    #[tokio::test]
    async fn find_sessions_by_prefix_empty_prefix_matches_nothing() {
        let store = Store::for_test().await;
        store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed to create session");
        let matches = store
            .find_sessions_by_prefix("")
            .await
            .expect("failed prefix lookup");
        assert!(
            matches.is_empty(),
            "empty prefix must not match every session"
        );
    }

    #[tokio::test]
    async fn delete_session_removes_lock_file() {
        let store = Store::for_test().await;
        let session = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let lock_path = store.lock_dir.join(format!("{session}.lock"));
        std::fs::write(&lock_path, "").expect("write lock");

        store.delete_session(session).await.expect("delete");
        assert!(
            !lock_path.exists(),
            "deleting a session must remove its lock file"
        );
    }

    /// A session another meka process has open is not one this process may delete.
    ///
    /// `meka session delete <id>` against a live REPL exited 0 having said nothing at all: the
    /// count goes through `tracing::info!`, invisible at the default level. The row and its
    /// messages cascaded away underneath a conversation that carried on as though nothing had
    /// happened, until its next turn ran against the provider and *then* failed on a foreign-key
    /// violation -- tokens spent, answer lost, and every later turn in that REPL failing the same
    /// way with no recovery.
    #[tokio::test]
    async fn deleting_a_session_another_process_holds_is_refused() {
        let store = Store::for_test().await;
        let session = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let _held = store.lock_session(session).expect("hold the session");

        match store.delete_session_unless_attached(session).await {
            Err(MekaError::SessionLocked(id)) => assert_eq!(id, session),
            other => panic!("expected SessionLocked, got {:?}", other.map(|_| "Ok(_)")),
        }
        assert!(
            store.session_exists(session).await.expect("exists"),
            "the refusal must leave the conversation alone, not merely report one"
        );
    }

    /// The sweep that runs on every start, against a session someone is sitting in.
    ///
    /// Only turns bump `updated_at` -- resuming does not touch it -- so a REPL left at its prompt
    /// past the retention window looks expired while a human is looking at it. Any `meka` start
    /// that goes through `async_main` runs this sweep, so an unrelated invocation in another
    /// terminal announced `deleted 1 session(s)` and destroyed the live one.
    #[tokio::test]
    async fn the_retention_sweep_spares_a_session_that_is_open() {
        let store = Store::for_test().await;
        let open = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let stale = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let long_ago = (chrono::Utc::now() - chrono::Duration::days(90)).to_rfc3339();
        for session in [open, stale] {
            store
                .set_session_updated_at_for_test(session, &long_ago)
                .await
                .expect("backdate");
        }
        let _held = store.lock_session(open).expect("hold the session");

        let sweep = store
            .delete_expired_sessions(std::time::Duration::from_secs(30 * 86_400))
            .await
            .expect("retention sweep");

        assert_eq!(sweep.deleted, 1, "the session nobody has open still goes");
        assert_eq!(
            sweep.attached_elsewhere, 1,
            "and the sweep has to be able to say what it left, or a count of deletions reads as \
             'everything matched went'"
        );
        assert!(store.session_exists(open).await.expect("exists"));
        assert!(!store.session_exists(stale).await.expect("exists"));
    }

    /// A session's lock is taken *before* its row is written, not after.
    ///
    /// The window between the two is what `meka session delete --all` fell into: it enumerates
    /// `SELECT id FROM sessions` at delete time, so it saw a row committed microseconds earlier
    /// whose creator had not yet reached `flock`, took the lock nobody held, and cascaded the
    /// conversation away underneath the process creating it. A contention run measured **42 lost
    /// turns in 11,948** with four creators against two sweep loops, each ending
    /// `FOREIGN KEY constraint failed` with the user's prompt gone.
    ///
    /// Observed by ordering rather than by racing, deliberately. The window is microseconds wide,
    /// so a test that waits for the row and then sweeps passes just as happily with the fix
    /// reverted -- the first version of this test did exactly that. Volume does find it, but at
    /// roughly one event per six hundred turns, which is a coin flip rather than a guard. Breaking
    /// the insert instead makes the ordering directly visible: if the lock comes first, the
    /// attempt leaves a lock file behind even though no row was ever written, and if it comes
    /// second there is nothing in the directory at all.
    #[tokio::test]
    async fn a_session_is_locked_before_its_row_is_written() {
        let store = Store::for_test().await;
        let locks_before = std::fs::read_dir(&store.lock_dir)
            .expect("read the lock dir")
            .count();
        // Renamed rather than dropped, so the failure is a plain "no such table" from the insert
        // rather than anything the foreign keys have an opinion about.
        store
            .connection
            .call(|connection| connection.execute_batch("ALTER TABLE sessions RENAME TO hidden;"))
            .await
            .expect("hide the table");

        let refused = store
            .create_session_locked(
                None,
                "read".to_string(),
                false,
                None,
                None,
                "test-profile".to_string(),
            )
            .await;

        assert!(
            refused.is_err(),
            "the premise: with no `sessions` table the row cannot be written"
        );
        assert_eq!(
            std::fs::read_dir(&store.lock_dir)
                .expect("read the lock dir")
                .count(),
            locks_before + 1,
            "a creation that never wrote a row must still have taken its lock first; nothing in \
             the lock directory means the row went first, and a row that lands before its lock is \
             one a sweep can take"
        );
    }

    /// A fork whose copy cannot be written leaves no claim behind, and still claimed first.
    ///
    /// The copy's id is claimed before its row exists, for the reason
    /// [`Store::create_session_locked`] does it: a row committed ahead of its lock is one a
    /// concurrent `meka session delete --all` enumerates and sweeps, after which the fork locks
    /// the vanished id and hands its caller a session whose next turn dies on a foreign key.
    ///
    /// That ordering used to be read off the file the claim left behind when the insert failed,
    /// the way [`a_session_is_locked_before_its_row_is_written`] still reads the create door's.
    /// Here the file is the defect: a copy nothing wrote a row for left one per failed attempt,
    /// and the sweep that collects them runs only at `open()` and after a delete. So the file has
    /// to go, and the ordering is read off the source instead, as the re-attach guards read
    /// theirs. The source's own file stays, as it does after a copy that succeeds: its row is
    /// still there, and unlinking the file another process locks it by could leave two holders.
    #[tokio::test]
    async fn a_fork_whose_copy_fails_leaves_no_claim_behind_and_still_claimed_first() {
        let store = Store::for_test().await;
        let source = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        // The source's own lock file, taken and released here so it is in place before the probe
        // and the comparison below measures the copy's claim alone.
        drop(store.lock_session(source).expect("probe the source once"));
        let lock_files = || -> std::collections::BTreeSet<std::ffi::OsString> {
            std::fs::read_dir(&store.lock_dir)
                .expect("read the lock dir")
                .map(|entry| entry.expect("a directory entry").file_name())
                .collect()
        };
        let before = lock_files();
        // Renamed rather than dropped, so the copy fails on a plain "no such table" rather than on
        // anything the foreign keys have an opinion about.
        store
            .connection
            .call(|connection| connection.execute_batch("ALTER TABLE sessions RENAME TO hidden;"))
            .await
            .expect("hide the table");

        let refused = store
            .fork_session_locked(source, ForkOverrides::default(), SourceLock::Probe)
            .await;

        assert!(
            refused.is_err(),
            "the premise: with no `sessions` table the copy cannot be written"
        );
        assert_eq!(
            lock_files(),
            before,
            "a copy that was never written must leave nothing in the lock directory, and the \
             source's own file must stay where it was"
        );
        assert!(
            store.lock_session(source).is_ok(),
            "and the probe's hold on the source is released with the refusal"
        );

        // The ordering, asserted against the source. Normalized first: CI's Windows checkout is
        // CRLF, and the closing-brace delimiter below would otherwise never match.
        let body = include_str!("sessions.rs")
            .replace("\r\n", "\n")
            .split("pub(crate) async fn fork_session_locked(")
            .nth(1)
            .expect("the door this test is about")
            .split("\n    }\n")
            .next()
            .expect("splitting always yields a first part")
            .to_string();
        let claim = body
            .find("claim_a_fresh_id(new_id)")
            .expect("the copy's id is claimed in the door");
        let copy = body
            .find("fork_session_into(new_id")
            .expect("and the copy is written in the door");
        assert!(
            claim < copy,
            "the claim must come before the copy's row is written; found claim@{claim} copy@{copy}"
        );
    }

    /// A source another process is writing is refused, and a source the caller holds is not.
    ///
    /// `Agent::run_turn` persists the user message before the provider answers, so a copy taken
    /// mid-turn ends on a message nothing answered. Every fork door reaches this one method, so the
    /// probe lives here rather than at the doors that remembered it (`meka session fork`) and the
    /// ones that did not (`POST /fork`, `session/fork`). A second descriptor stands in for the
    /// other process: `flock` conflicts across descriptors whether or not they share a process.
    #[tokio::test]
    async fn a_fork_refuses_a_source_another_process_holds_and_trusts_a_caller_that_does() {
        let store = Store::for_test().await;
        let source = seeded_session(&store).await;
        let held = store.lock_session(source).expect("hold the source");

        let refused = store
            .fork_session_locked(source, ForkOverrides::default(), SourceLock::Probe)
            .await
            .err()
            .map(|error| error.to_string());
        assert_eq!(
            refused,
            Some(MekaError::SessionLocked(source).to_string()),
            "a probe against a held source must refuse with the source's id"
        );
        let (listed, _cursor) = store
            .list_sessions(10, false, None, None)
            .await
            .expect("list");
        assert_eq!(listed.len(), 1, "and no copy was written");

        let (copied, copy_lock) = store
            .fork_session_locked(source, ForkOverrides::default(), SourceLock::HeldByCaller)
            .await
            .expect("a caller that holds the source forks it")
            .expect("the source exists");
        assert!(copy_lock.is_ok(), "the copy's own lock is still taken");
        assert_ne!(copied.id, source);
        drop(held);

        let (copied_again, _lock) = store
            .fork_session_locked(source, ForkOverrides::default(), SourceLock::Probe)
            .await
            .expect("a released source forks")
            .expect("the source exists");
        assert_ne!(copied_again.id, copied.id);
    }

    /// A probe of an id that names nothing leaves no lock file behind, like the copy's claim.
    #[tokio::test]
    async fn a_fork_of_an_unknown_id_leaves_neither_claim_behind() {
        let store = Store::for_test().await;
        let locks_before = std::fs::read_dir(&store.lock_dir)
            .expect("read the lock dir")
            .count();
        assert!(
            store
                .fork_session_locked(Uuid::new_v4(), ForkOverrides::default(), SourceLock::Probe)
                .await
                .expect("fork")
                .is_none()
        );
        assert_eq!(
            std::fs::read_dir(&store.lock_dir)
                .expect("read the lock dir")
                .count(),
            locks_before,
            "an unknown id is client-reachable and must not leave a file per attempt"
        );
    }

    /// Opening a row takes its lock first and reads second, and says which of the two failed.
    #[tokio::test]
    async fn opening_a_session_row_locks_then_reads_and_names_what_it_could_not_do() {
        let store = Store::for_test().await;
        let id = store
            .create_session(
                Some(PathBuf::from("/work/main")),
                "test-profile".to_string(),
            )
            .await
            .expect("create");

        let (lock, summary) = store.open_session_row(id).await.expect("open");
        assert_eq!(summary.id, id);
        assert_eq!(summary.cwd, Some(PathBuf::from("/work/main")));
        assert!(
            matches!(store.lock_session(id), Err(MekaError::SessionLocked(_))),
            "the returned lock is the session's"
        );
        drop(lock);

        let held = store.lock_session(id).expect("hold it from elsewhere");
        assert!(
            matches!(
                store.open_session_row(id).await,
                Err(MekaError::SessionLocked(locked)) if locked == id
            ),
            "a held session is refused before it is read"
        );
        drop(held);

        let locks_before = std::fs::read_dir(&store.lock_dir)
            .expect("read the lock dir")
            .count();
        let missing = Uuid::new_v4();
        assert!(
            matches!(
                store.open_session_row(missing).await,
                Err(MekaError::SessionNotFound(gone)) if gone == missing
            ),
            "a row that is gone is its own answer"
        );
        assert_eq!(
            std::fs::read_dir(&store.lock_dir)
                .expect("read the lock dir")
                .count(),
            locks_before,
            "and the claim on an id nobody has leaves no file behind"
        );
    }

    /// An import claims each root of the tree before its row lands, the ordering every other door
    /// that mints a session keeps, and releases it once written.
    ///
    /// The lock file is the proof, as in `a_session_is_locked_before_its_row_is_written`: a claim
    /// creates it, and nothing else in an import does. Only roots are claimed, because a
    /// sub-agent's row is never opened on its own.
    #[tokio::test]
    async fn an_import_claims_its_roots_before_their_rows_and_lets_go_after() {
        let store = Store::for_test().await;
        let root = Uuid::new_v4();
        let child = Uuid::new_v4();
        let record = |id, parent| ImportSessionRecord {
            new_id: id,
            new_parent_id: parent,
            created_at: chrono::Utc::now().to_rfc3339(),
            cwd: None,
            permission: crate::permission::Permission::Read,
            approvals: false,
            capabilities_json: None,
            additional_roots: Vec::new(),
            subagent_spec_json: parent.map(|_| "{\"tools\":[]}".to_string()),
            profile: "work".to_string(),
            stats: crate::stats::SessionStatsSnapshot::default(),
            events: Vec::new(),
            tool_outputs: Vec::new(),
        };
        store
            .import_sessions(
                vec![record(root, None), record(child, Some(root))],
                Vec::new(),
            )
            .await
            .expect("import");

        assert!(
            store.lock_dir.join(format!("{root}.lock")).exists(),
            "the root was claimed ahead of its row"
        );
        assert!(
            !store.lock_dir.join(format!("{child}.lock")).exists(),
            "a sub-agent's row is never opened on its own, so it is not claimed"
        );
        assert!(
            store.lock_session(root).is_ok(),
            "and the claim was released once the row existed"
        );
    }

    /// An archive refused while it is still being encoded leaves no claim behind.
    ///
    /// The roots were claimed ahead of the encoding, whose every `?` returned past the arm that
    /// let the claims go, so a refused archive left one file per root for a sweep that runs only
    /// at `open()` and after a delete. A path that is not UTF-8 is what `serde` refuses to encode,
    /// which an archive written by hand can carry; Unix only, because the other platforms cannot
    /// spell one.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_import_refused_while_encoding_leaves_no_claim_behind() {
        use std::os::unix::ffi::OsStrExt;

        let store = Store::for_test().await;
        let root = Uuid::new_v4();
        let refused = store
            .import_sessions(
                vec![ImportSessionRecord {
                    new_id: root,
                    new_parent_id: None,
                    created_at: chrono::Utc::now().to_rfc3339(),
                    cwd: None,
                    permission: crate::permission::Permission::Read,
                    approvals: false,
                    capabilities_json: None,
                    additional_roots: vec![PathBuf::from(std::ffi::OsStr::from_bytes(
                        b"/not-\xffutf8",
                    ))],
                    subagent_spec_json: None,
                    profile: "work".to_string(),
                    stats: crate::stats::SessionStatsSnapshot::default(),
                    events: Vec::new(),
                    tool_outputs: Vec::new(),
                }],
                Vec::new(),
            )
            .await;

        assert!(
            refused.is_err(),
            "the premise: a root list that cannot be encoded refuses the archive"
        );
        assert!(
            !store.lock_dir.join(format!("{root}.lock")).exists(),
            "a refused archive must not leave its root's claim behind"
        );
        assert!(
            !store.session_exists(root).await.expect("exists"),
            "and wrote nothing"
        );
    }

    /// The sweep decides on the rows as they are, not on a list read a moment earlier.
    ///
    /// Selecting candidates and then deleting them by id is two statements where there was one, so
    /// a condition checked only in the first can stop being true in between. The one that matters
    /// is "no schedule ahead of it": `parent_session_id` cascades, so a job created against a
    /// sub-agent child in that gap would be swept away with a parent nothing has locked, and the
    /// lock cannot stand in for the check because it is the *parent* being deleted and the
    /// *child* that acquired the job.
    ///
    /// Driven through [`Store::delete_the_unattached_among`] directly, with a candidate
    /// that already owns a job, because the gap itself is microseconds wide and not something a
    /// test can sit inside. What it pins is the property that closes it: the predicate is in the
    /// delete, not only in the select.
    #[tokio::test]
    async fn the_sweep_re_checks_its_own_condition_inside_the_delete() {
        let store = Store::for_test().await;
        let scheduled = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let ordinary = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        store
            .schedule_store()
            .create_scheduled_job(&crate::schedule::ScheduledJob {
                attempts: 0,
                id: "job-1".to_string(),
                session_id: scheduled,
                schedule: crate::schedule::Schedule::parse_every("1h").expect("parses"),
                prompt: "check the thing".to_string(),
                gate: None,
                created_at: chrono::Utc::now(),
                last_fired_at: None,
                next_fire_at: chrono::Utc::now() + chrono::Duration::hours(1),
            })
            .await
            .expect("create the job");

        let sweep = store
            .delete_the_unattached_among(&[scheduled, ordinary], NOT_SPOKEN_FOR_BY_A_SCHEDULE, None)
            .await
            .expect("sweep");

        assert_eq!(sweep.deleted, 1, "only the one with nothing ahead of it");
        assert!(
            store.session_exists(scheduled).await.expect("exists"),
            "a session a job still depends on must survive a delete it was listed for"
        );
        assert!(!store.session_exists(ordinary).await.expect("exists"));
    }

    /// The cutoff the candidates were selected under is re-applied inside the delete. A session
    /// resumed and released between the select and the delete has a fresh `updated_at`, and the
    /// stale list still names it; deleting on the list alone took the conversation with it.
    #[tokio::test]
    async fn a_session_touched_since_it_was_listed_survives_the_retention_delete() {
        let store = Store::for_test().await;
        let fresh = store
            .create_session_locked(
                None,
                "read".to_string(),
                false,
                None,
                None,
                "test-profile".to_string(),
            )
            .await
            .expect("create")
            .0
            .id;
        let cutoff = (chrono::Utc::now() - chrono::Duration::days(30)).to_rfc3339();
        let sweep = store
            .delete_the_unattached_among(&[fresh], "", Some(cutoff.as_str()))
            .await
            .expect("sweep");
        assert_eq!(sweep.deleted, 0);
        assert!(
            store.session_exists(fresh).await.expect("exists"),
            "a row newer than the cutoff is not the one the list meant"
        );
    }

    /// `meka session delete --all` is the same rule with no window: everything except what someone
    /// else is using.
    #[tokio::test]
    async fn delete_all_spares_a_session_that_is_open() {
        let store = Store::for_test().await;
        let open = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let other = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let _held = store.lock_session(open).expect("hold the session");

        let sweep = store.delete_all_sessions().await.expect("delete all");

        assert_eq!(sweep.deleted, 1);
        assert_eq!(sweep.attached_elsewhere, 1);
        assert!(store.session_exists(open).await.expect("exists"));
        assert!(!store.session_exists(other).await.expect("exists"));
    }

    #[tokio::test]
    async fn a_random_id_is_not_an_existing_session() {
        let store = Store::for_test().await;
        let fake_id = Uuid::new_v4();
        assert!(
            !store
                .session_exists(fake_id)
                .await
                .expect("failed to check")
        );
    }

    #[tokio::test]
    async fn multiple_sessions() {
        let store = Store::for_test().await;
        let session1 = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed");
        let session2 = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed");

        store
            .save_message(session1, "user", "msg1")
            .await
            .expect("failed");
        store
            .save_message(session2, "user", "msg2")
            .await
            .expect("failed");

        let messages1 = store.load_messages(session1).await.expect("failed");
        let messages2 = store.load_messages(session2).await.expect("failed");

        assert_eq!(messages1.len(), 1);
        assert_eq!(messages1[0].content, "msg1");
        assert_eq!(messages2.len(), 1);
        assert_eq!(messages2[0].content, "msg2");
    }

    #[tokio::test]
    async fn an_expired_session_is_deleted() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed");
        store
            .save_message(session_id, "user", "hello")
            .await
            .expect("failed");

        // Backdate the session to 100 days ago
        let old_date = (chrono::Utc::now() - chrono::TimeDelta::days(100)).to_rfc3339();
        store
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![old_date, session_id.to_string()],
                )?;
                Ok(())
            })
            .await
            .expect("failed to backdate");

        let deleted = store
            .delete_expired_sessions(std::time::Duration::from_secs(30 * 86_400))
            .await
            .expect("failed to delete");
        assert_eq!(deleted.deleted, 1);
        assert!(!store.session_exists(session_id).await.expect("failed"));

        let messages = store.load_messages(session_id).await.expect("failed");
        assert!(messages.is_empty());
    }

    /// The FK cascade must not take a job-owning child with its stale parent.
    ///
    /// A session holding a scheduled job is not idle, whatever its `updated_at` says: only turns
    /// bump that column, so a gated watcher that evaluates every tick and rarely fires looks
    /// untouched precisely while it is doing its job.
    ///
    /// Sparing only the row named by `scheduled_jobs.session_id` left the guard half-built:
    /// `parent_session_id` carries `ON DELETE CASCADE`, so a root session that has gone quiet
    /// still deletes its sub-agent children, and a job created against a child -- which the HTTP
    /// surface allows, gating only on the session existing -- went with them. The sweep then
    /// reported one deletion and said nothing about the schedule it destroyed.
    #[tokio::test]
    async fn retention_spares_the_parent_of_a_child_that_has_a_scheduled_job() {
        let store = Store::for_test().await;
        let parent = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create parent");
        let child = store
            .create_child_session(
                parent,
                None,
                Vec::new(),
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("create child")
            .0;

        let job = crate::schedule::ScheduledJob {
            attempts: 0,
            id: uuid::Uuid::new_v4().to_string(),
            session_id: child,
            schedule: crate::schedule::Schedule::parse_every("30m").expect("parses"),
            prompt: "check the build".to_string(),
            gate: None,
            created_at: chrono::Utc::now(),
            last_fired_at: None,
            next_fire_at: chrono::Utc::now() + chrono::Duration::minutes(30),
        };
        store
            .schedule_store()
            .create_scheduled_job(&job)
            .await
            .expect("save job");

        // Only the parent looks ancient; the child is what owns the future.
        let ancient = (chrono::Utc::now() - chrono::Duration::days(400)).to_rfc3339();
        store
            .set_session_updated_at_for_test(parent, &ancient)
            .await
            .expect("backdate");

        store
            .delete_expired_sessions(std::time::Duration::from_secs(30 * 86_400))
            .await
            .expect("sweep");

        assert!(
            store.session_exists(child).await.expect("exists"),
            "the child owning the job was cascaded away with its stale parent"
        );
        assert!(
            store.session_exists(parent).await.expect("exists"),
            "the parent must be spared too, since deleting it is what takes the child"
        );
    }

    /// Retention must leave a session alone while it still owns a scheduled job.
    ///
    /// It did not: the cascade took the job along with the session, and the sweep reported only
    /// "deleted N session(s)".
    #[tokio::test]
    async fn retention_spares_a_session_that_still_has_a_scheduled_job() {
        let store = Store::for_test().await;
        let watcher = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let plain = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");

        let job = crate::schedule::ScheduledJob {
            attempts: 0,
            id: uuid::Uuid::new_v4().to_string(),
            session_id: watcher,
            schedule: crate::schedule::Schedule::parse_every("30m").expect("parses"),
            prompt: "check the build".to_string(),
            gate: None,
            created_at: chrono::Utc::now(),
            last_fired_at: None,
            next_fire_at: chrono::Utc::now() + chrono::Duration::minutes(30),
        };
        store
            .schedule_store()
            .create_scheduled_job(&job)
            .await
            .expect("save job");

        // Both look ancient by `updated_at`; only one of them has a future.
        let ancient = (chrono::Utc::now() - chrono::Duration::days(400)).to_rfc3339();
        for id in [watcher, plain] {
            store
                .set_session_updated_at_for_test(id, &ancient)
                .await
                .expect("backdate");
        }

        store
            .delete_expired_sessions(std::time::Duration::from_secs(30 * 86_400))
            .await
            .expect("sweep");

        assert!(
            store.session_exists(watcher).await.expect("exists"),
            "a session with a pending job must survive retention"
        );
        assert!(
            !store.session_exists(plain).await.expect("exists"),
            "a genuinely idle session should still be swept"
        );
        assert_eq!(
            store
                .schedule_store()
                .list_scheduled_jobs(watcher)
                .await
                .expect("list")
                .len(),
            1,
            "the job must survive with its session"
        );
    }

    #[tokio::test]
    async fn delete_expired_sessions_keeps_recent() {
        let store = Store::for_test().await;
        let old_session = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed");
        let new_session = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed");

        store
            .save_message(old_session, "user", "old")
            .await
            .expect("failed");
        store
            .save_message(new_session, "user", "new")
            .await
            .expect("failed");

        // Backdate only the old session
        let old_date = (chrono::Utc::now() - chrono::TimeDelta::days(100)).to_rfc3339();
        store
            .connection
            .call(move |connection| -> rusqlite::Result<_> {
                connection.execute(
                    "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                    rusqlite::params![old_date, old_session.to_string()],
                )?;
                Ok(())
            })
            .await
            .expect("failed to backdate");

        let deleted = store
            .delete_expired_sessions(std::time::Duration::from_secs(30 * 86_400))
            .await
            .expect("failed to delete");
        assert_eq!(deleted.deleted, 1);
        assert!(!store.session_exists(old_session).await.expect("failed"));
        assert!(store.session_exists(new_session).await.expect("failed"));
    }

    /// `--older-than-days` puts a raw number in the user's hands, so a mistyped run of digits must
    /// not panic. `TimeDelta` overflows near 10^11 days and `Utc::now() - delta` near 96.4 million,
    /// so both bounds need covering; either way nothing is old enough to match.
    #[tokio::test]
    async fn delete_expired_sessions_survives_absurd_windows() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        store
            .save_message(session_id, "user", "hello")
            .await
            .expect("save");

        for days in [96_500_000, u64::MAX] {
            let deleted = store
                .delete_expired_sessions(std::time::Duration::from_secs(
                    days.saturating_mul(86_400),
                ))
                .await
                .expect("must not panic or error");
            assert_eq!(deleted.deleted, 0, "{days} days should match nothing");
        }
        assert!(store.session_exists(session_id).await.expect("exists"));
    }

    #[tokio::test]
    async fn clearing_a_session_removes_its_messages() {
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("failed");

        store
            .save_message(session_id, "user", "hello")
            .await
            .expect("failed");
        store
            .save_message(session_id, "assistant", "hi")
            .await
            .expect("failed");

        let messages = store.load_messages(session_id).await.expect("failed");
        assert_eq!(messages.len(), 2);

        store
            .clear_messages(session_id)
            .await
            .expect("failed to clear");

        let messages = store.load_messages(session_id).await.expect("failed");
        assert!(messages.is_empty());

        // Session itself should still exist
        assert!(store.session_exists(session_id).await.expect("failed"));
    }

    /// Reconstructs what `agent::Agent::run_turn` appends for a fresh user turn: the context block
    /// and the user's raw prompt, as `Message::user_turn` shapes them.
    fn mock_run_turn_user_message(
        permission: crate::permission::Permission,
        user_input: &str,
    ) -> crate::conversation::Message {
        let block = crate::prompt::build_turn_context(crate::prompt::TurnContext {
            permission,
            approvals: false,
            todos: &crate::todo::TodoState::default(),
            cwd: std::path::Path::new("."),
            roots: &[],
            world_state: "",
            budget: Some(crate::prompt::ContextBudget {
                used: 42_000,
                window: 200_000,
                compact_at_percent: Some(80),
                generation: 0,
            }),
            background: &[],
            outcomes: None,
            resumed: true,
        });
        crate::conversation::Message::user_turn(block, user_input, Vec::new())
    }

    #[tokio::test]
    async fn list_sessions_title_is_user_prompt_not_context_wrapper() {
        // The canonical regression: user types a prompt, turn runs, `meka session list` must show
        // the prompt, not `<context>`, not the permission/environment metadata.
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");

        let user_prompt = "find all Rust files under src/";
        let stored = mock_run_turn_user_message(crate::permission::Permission::Read, user_prompt);
        store
            .save_event(session_id, &crate::conversation::Event::Append(stored))
            .await
            .expect("save_message");

        let (summaries, _next_cursor) = store
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let summary = summaries
            .iter()
            .find(|s| s.id == session_id)
            .expect("session missing from list");

        assert_eq!(
            summary.title, user_prompt,
            "title regressed: expected user prompt, got {:?}",
            summary.title
        );
        assert!(
            !summary.title.contains("<context>"),
            "wrapper leaked into the title: {:?}",
            summary.title
        );
        assert!(
            !summary.title.contains("[Permission context]"),
            "permission metadata leaked into the title: {:?}",
            summary.title
        );
    }

    #[tokio::test]
    async fn list_sessions_title_covers_all_permission_levels() {
        // The context block's shape differs per permission level (`none` omits the [Environment
        // context] entirely, `workspace` adds a write-boundary paragraph to it). Every level should
        // still surface the user's prompt cleanly. "all permission levels" in the name is a claim,
        // so the list has to actually hold all of them: `workspace` was missing here, and it is the
        // level whose block grew a new section.
        let store = Store::for_test().await;
        for (label, permission) in &[
            ("none", crate::permission::Permission::None),
            ("read", crate::permission::Permission::Read),
            ("workspace", crate::permission::Permission::Workspace),
            ("unrestricted", crate::permission::Permission::Unrestricted),
        ] {
            let session_id = store
                .create_session(None, "test-profile".to_string())
                .await
                .expect("create_session");
            let prompt = format!("ask at {label} level");
            let stored = mock_run_turn_user_message(*permission, &prompt);
            store
                .save_event(session_id, &crate::conversation::Event::Append(stored))
                .await
                .expect("save_message");

            let (summaries, _next_cursor) = store
                .list_sessions(100, false, None, None)
                .await
                .expect("list_sessions");
            let summary = summaries
                .iter()
                .find(|s| s.id == session_id)
                .unwrap_or_else(|| panic!("session missing for level {label}"));
            assert_eq!(
                summary.title, prompt,
                "title mismatch at permission level {label}"
            );
        }
    }

    #[tokio::test]
    async fn list_sessions_title_truncates_long_prompt_with_ellipsis() {
        // Long prompts are capped at 80 chars with a trailing ellipsis. The cap must apply to the
        // user's prompt, not the wrapper.
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");

        let long_prompt = "a".repeat(150);
        let stored = mock_run_turn_user_message(crate::permission::Permission::Read, &long_prompt);
        store
            .save_event(session_id, &crate::conversation::Event::Append(stored))
            .await
            .expect("save_message");

        let (summaries, _next_cursor) = store
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();

        assert!(
            summary.title.starts_with("aaa"),
            "the title should start with the user's content, not the wrapper: {:?}",
            summary.title
        );
        assert!(
            summary.title.ends_with('…'),
            "a long title should end with an ellipsis: {:?}",
            summary.title
        );
        assert!(summary.title.chars().count() <= 81);
    }

    #[tokio::test]
    async fn list_sessions_title_is_first_user_turn_not_later() {
        // Multiple turns in one session: the title must be the FIRST user prompt, not a later one.
        // `ORDER BY id ASC LIMIT 1` guarantees this; guard against that being changed.
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");

        for (i, prompt) in ["first prompt", "second prompt", "third prompt"]
            .iter()
            .enumerate()
        {
            let stored = mock_run_turn_user_message(crate::permission::Permission::Read, prompt);
            store
                .save_event(session_id, &crate::conversation::Event::Append(stored))
                .await
                .expect("save_message");
            // Interleave an assistant reply: real sessions alternate.
            store
                .save_message(session_id, "assistant", &format!("reply {i}"))
                .await
                .expect("save_message");
        }

        let (summaries, _next_cursor) = store
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(summary.title, "first prompt");
    }

    #[tokio::test]
    async fn list_sessions_title_collapses_a_multiline_prompt_to_one_line() {
        // Multi-line user prompts become one line in the list view, keeping every word: the same
        // `Conversation::title` every other surface shows.
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");

        let stored = mock_run_turn_user_message(
            crate::permission::Permission::Read,
            "line one of the title\nline two joins it\n\n  line three too",
        );
        store
            .save_event(session_id, &crate::conversation::Event::Append(stored))
            .await
            .expect("save_message");

        let (summaries, _next_cursor) = store
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(
            summary.title,
            "line one of the title line two joins it line three too"
        );
    }

    #[tokio::test]
    async fn list_sessions_title_skips_a_first_turn_that_carried_no_words() {
        // An image sent alone is a `user_blocks` row with no `text` block. The title is the first
        // user *words*, as `Conversation::title` defines it, so the next turn's prompt labels the
        // session rather than leaving it blank forever.
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");
        let image_only = crate::conversation::Message::user_turn("<context/>", "", vec![
            crate::image::ImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: "AAAA".to_string(),
            },
        ]);
        store
            .save_event(session_id, &crate::conversation::Event::Append(image_only))
            .await
            .expect("save the image-only turn");
        let words = mock_run_turn_user_message(
            crate::permission::Permission::Read,
            "what is in this picture?",
        );
        store
            .save_event(session_id, &crate::conversation::Event::Append(words))
            .await
            .expect("save the worded turn");

        let (summaries, _next_cursor) = store
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(summary.title, "what is in this picture?");
        let info = store
            .session_info(session_id)
            .await
            .expect("session_info")
            .expect("row");
        assert_eq!(
            info.title, summary.title,
            "both readers select the same row"
        );
    }

    /// The SQL twin of `Conversation::title` skips what it skips: a turn whose only text is the
    /// placeholder a redaction left, and a compaction summary, so the title is the first words a
    /// user said even when the store replays a redacted, compacted session.
    #[tokio::test]
    async fn list_sessions_title_skips_a_placeholder_and_a_compaction_summary() {
        use crate::conversation::{ContentBlock, Event, Message, Role};
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");
        let placeholder_only = Message {
            role: Role::User,
            content: vec![
                ContentBlock::TurnContext {
                    text: "<context/>".to_string(),
                },
                ContentBlock::Text {
                    text: crate::conversation::IMAGE_REDACTION_PLACEHOLDER.to_string(),
                },
            ],
        };
        let events = [
            Event::Append(placeholder_only),
            Event::Append(Message::user(format!(
                "{} An image here was removed.",
                crate::conversation::HARNESS_NOTE
            ))),
            Event::CompactBoundary {
                summary: Message::user("[Conversation summary from session compaction] nothing"),
                replaced_count: 2,
                loaded_tools_snapshot: std::collections::HashSet::new(),
            },
            Event::Append(mock_run_turn_user_message(
                crate::permission::Permission::Read,
                "what is in this picture?",
            )),
        ];
        for event in &events {
            store.save_event(session_id, event).await.expect("save");
        }

        let (summaries, _next_cursor) = store
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(summary.title, "what is in this picture?");
        let info = store
            .session_info(session_id)
            .await
            .expect("session_info")
            .expect("row");
        assert_eq!(
            info.title, summary.title,
            "both readers select the same row"
        );
    }

    /// An archive naming an image blob it does not carry and the store does not hold is refused in
    /// the caller's words, before any row of it is written.
    #[tokio::test]
    async fn an_archive_referencing_a_blob_nobody_holds_is_refused_before_any_write() {
        let store = Store::for_test().await;
        let new_id = uuid::Uuid::new_v4();
        let hash = "0".repeat(64);
        let record = ImportSessionRecord {
            new_id,
            new_parent_id: None,
            created_at: "2026-08-31T00:00:00Z".to_string(),
            cwd: None,
            permission: crate::permission::Permission::Read,
            approvals: false,
            capabilities_json: None,
            additional_roots: Vec::new(),
            subagent_spec_json: None,
            profile: "test-profile".to_string(),
            stats: Default::default(),
            events: vec![(
                "2026-08-31T00:00:00Z".to_string(),
                crate::conversation::Event::Append(crate::conversation::Message::user_with_images(
                    "look",
                    vec![crate::image::ImageSource::Blob {
                        hash: hash.clone(),
                        media_type: "image/png".to_string(),
                        size: 3,
                    }],
                )),
            )],
            tool_outputs: Vec::new(),
        };
        let error = store
            .import_sessions(vec![record], Vec::new())
            .await
            .expect_err("a reference nobody can resolve is refused");
        assert!(
            matches!(&error, MekaError::Usage(message) if message.contains(&hash)),
            "the refusal is the caller's to act on and names the blob: {error}"
        );
        assert!(
            store
                .session_info(new_id)
                .await
                .expect("session_info")
                .is_none(),
            "nothing was written"
        );
    }

    #[tokio::test]
    async fn list_sessions_title_independent_per_session() {
        // Each session's title is its own first user turn: no cross-contamination from neighbor
        // sessions.
        let store = Store::for_test().await;
        let a = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");
        let b = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");
        let c = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");

        for (sid, prompt) in [(a, "alpha"), (b, "beta"), (c, "gamma")] {
            let stored = mock_run_turn_user_message(crate::permission::Permission::Read, prompt);
            store
                .save_event(sid, &crate::conversation::Event::Append(stored))
                .await
                .expect("save_message");
        }

        let (summaries, _next_cursor) = store
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let title_of = |id: uuid::Uuid| {
            summaries
                .iter()
                .find(|s| s.id == id)
                .map(|s| s.title.clone())
                .unwrap_or_default()
        };
        assert_eq!(title_of(a), "alpha");
        assert_eq!(title_of(b), "beta");
        assert_eq!(title_of(c), "gamma");
    }

    #[tokio::test]
    async fn list_sessions_title_empty_session_has_empty_title() {
        // A session with zero user messages (e.g. created but Ctrl-C'd before first dispatch) falls
        // back to an empty title; it should not panic or render `<no user msg>` scaffolding.
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");

        let (summaries, _next_cursor) = store
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(summary.title, "");
    }

    #[tokio::test]
    async fn list_sessions_title_compacted_session() {
        // After `/compact`, the agent clears messages and inserts a single new user message
        // starting with `[Conversation summary from session compaction]`. That has no `<context>`
        // wrapper; `list_sessions` should surface the summary's words, not an empty title.
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");

        let summary_message = "[Conversation summary from session compaction]\n\nSummary text here\n\n\
             [Post-compaction context]\n\n…";
        store
            .save_message(session_id, "user", summary_message)
            .await
            .expect("save_message");

        let (summaries, _next_cursor) = store
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(
            summary.title,
            "[Conversation summary from session compaction] Summary text here [Post-compactio…",
            "compacted session should surface the summary marker as its title"
        );
    }

    #[tokio::test]
    async fn list_sessions_title_unwrapped_user_message() {
        // The `<context>` block is added by `Agent::run_turn`, not by storage. A `user` row written
        // by any other path carries none -- `import_sessions` replays whatever an archive held --
        // and then the stored string IS the prompt, so the title equals it.
        let store = Store::for_test().await;
        let session_id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create_session");
        store
            .save_message(session_id, "user", "prompt without any wrapper")
            .await
            .expect("save_message");

        let (summaries, _next_cursor) = store
            .list_sessions(10, false, None, None)
            .await
            .expect("list_sessions");
        let summary = summaries.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(summary.title, "prompt without any wrapper");
    }

    #[tokio::test]
    async fn session_metadata_round_trips() {
        let store = Store::for_test().await;

        // The NULL shape the REPL and ACP get from `create_session_locked`, which they call with
        // permission, capabilities and token id all `None`: unset, so the re-attach helper falls
        // back to the process default.
        let plain = store
            .create_session(
                Some(std::path::PathBuf::from("/tmp/plain")),
                "test-profile".to_string(),
            )
            .await
            .expect("create plain");
        let plain_info = store
            .session_info(plain)
            .await
            .expect("session_info")
            .expect("plain row");
        assert_eq!(
            plain_info.permission,
            Some(crate::permission::Permission::Read)
        );
        assert_eq!(plain_info.capabilities_json, None);

        // Metadata path: persisted permission + capabilities + token_id round-trip verbatim.
        let with_meta = store
            .create_session_with_metadata(
                Some(std::path::PathBuf::from("/tmp/meta")),
                "read".to_string(),
                false,
                Some(r#"{"supports_reasoning_stream":true}"#.to_string()),
                Some("token_fp_1234".to_string()),
                "test-profile".to_string(),
            )
            .await
            .expect("create with metadata");
        let meta_info = store
            .session_info(with_meta.id)
            .await
            .expect("session_info")
            .expect("meta row");
        assert_eq!(
            meta_info.permission,
            Some(crate::permission::Permission::Read)
        );
        assert_eq!(
            meta_info.capabilities_json.as_deref(),
            Some(r#"{"supports_reasoning_stream":true}"#)
        );
        assert_eq!(
            meta_info.token_id.as_deref(),
            Some("token_fp_1234"),
            "token_id round-trips through the DB"
        );
        // The DB-returned `created_at` matches what session_info reads back.
        assert_eq!(meta_info.created_at, with_meta.created_at);

        // `update_session` flips the persisted value.
        store
            .update_session(with_meta.id, SessionPatch {
                permission: Some(crate::permission::Permission::Workspace),
                ..Default::default()
            })
            .await
            .expect("update permission");
        let after_flip = store
            .session_info(with_meta.id)
            .await
            .expect("session_info")
            .expect("post-flip row");
        assert_eq!(
            after_flip.permission,
            Some(crate::permission::Permission::Workspace)
        );
    }

    #[tokio::test]
    async fn create_child_session_writes_parent_id() {
        let store = Store::for_test().await;
        let parent = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create parent");
        let child = store
            .create_child_session(
                parent,
                None,
                Vec::new(),
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("create child")
            .0;

        // Cross-check the column via list_sessions(include_children=true).
        let (summaries, _next_cursor) = store
            .list_sessions(100, true, None, None)
            .await
            .expect("list_sessions");
        let ids: Vec<_> = summaries.iter().map(|s| s.id).collect();
        assert!(ids.contains(&parent), "parent missing from listing");
        assert!(ids.contains(&child), "child missing from listing");
    }

    #[tokio::test]
    async fn list_sessions_default_hides_children() {
        let store = Store::for_test().await;
        let parent = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create parent");
        let _child = store
            .create_child_session(
                parent,
                None,
                Vec::new(),
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("create child")
            .0;

        let (default_view, _) = store
            .list_sessions(10, false, None, None)
            .await
            .expect("list");
        let ids: Vec<_> = default_view.iter().map(|s| s.id).collect();
        assert_eq!(ids.len(), 1);
        assert!(ids.contains(&parent), "parent should still be visible");

        let (full_view, _) = store
            .list_sessions(10, true, None, None)
            .await
            .expect("list");
        assert_eq!(full_view.len(), 2);
    }

    #[tokio::test]
    async fn create_session_round_trips_cwd_through_session_info() {
        let store = Store::for_test().await;
        let cwd = PathBuf::from("/home/agent/proj-a");
        let sid = store
            .create_session(Some(cwd.clone()), "test-profile".to_string())
            .await
            .expect("create");

        let info = store
            .session_info(sid)
            .await
            .expect("session_info")
            .expect("present");
        assert_eq!(info.cwd, Some(cwd));
    }

    #[tokio::test]
    async fn session_info_returns_none_for_unknown_id() {
        let store = Store::for_test().await;
        let absent = store
            .session_info(Uuid::new_v4())
            .await
            .expect("session_info");
        assert!(absent.is_none());
    }

    #[tokio::test]
    async fn list_sessions_filters_by_cwd() {
        let store = Store::for_test().await;
        let cwd_a = PathBuf::from("/home/agent/proj-a");
        let cwd_b = PathBuf::from("/home/agent/proj-b");
        let a = store
            .create_session(Some(cwd_a.clone()), "test-profile".to_string())
            .await
            .expect("create a");
        let _b = store
            .create_session(Some(cwd_b.clone()), "test-profile".to_string())
            .await
            .expect("create b");

        let (only_a, next) = store
            .list_sessions(10, false, Some(&cwd_a), None)
            .await
            .expect("list filtered");
        assert_eq!(only_a.len(), 1);
        assert_eq!(only_a[0].id, a);
        assert!(
            next.is_none(),
            "single result must not advertise a next page"
        );

        let (all, _) = store
            .list_sessions(10, false, None, None)
            .await
            .expect("list unfiltered");
        assert_eq!(all.len(), 2, "unfiltered must include both sessions");
    }

    #[tokio::test]
    async fn list_sessions_cwd_filter_excludes_null_cwd_rows() {
        // A session created by `create_session(None, "test-profile".to_string())` recorded no cwd,
        // so it can never match a cwd filter: NULL is not equal to a TEXT value in SQL.
        let store = Store::for_test().await;
        let cwd = PathBuf::from("/home/agent/proj");
        let with_cwd = store
            .create_session(Some(cwd.clone()), "test-profile".to_string())
            .await
            .expect("create with cwd");
        let _without_cwd = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create without cwd");

        let (filtered, _) = store
            .list_sessions(10, false, Some(&cwd), None)
            .await
            .expect("list");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, with_cwd);
    }

    #[tokio::test]
    async fn list_sessions_pagination_cursor_round_trips() {
        let store = Store::for_test().await;
        // Create five sessions; cap each page at 2. Walking forward must visit all five exactly
        // once with monotonically older updated_at.
        let mut ids = Vec::new();
        for _ in 0..5 {
            let id = store
                .create_session(None, "test-profile".to_string())
                .await
                .expect("create");
            // `created_at`/`updated_at` use chrono::Utc::now(); pause to ensure each row's
            // timestamp is strictly newer (RFC3339 millisecond resolution).
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            ids.push(id);
        }

        let mut walked = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let (page, next) = store
                .list_sessions(2, false, None, cursor.as_deref())
                .await
                .expect("list");
            for summary in &page {
                walked.push(summary.id);
            }
            if next.is_none() {
                break;
            }
            cursor = next;
            assert!(walked.len() <= 5, "infinite pagination loop");
        }
        // The walk emits sessions newest-first; the creation order is oldest-first, so reverse to
        // compare.
        ids.reverse();
        assert_eq!(walked, ids, "pagination must visit every row in order");
    }

    #[tokio::test]
    async fn list_sessions_invalid_cursor_returns_error() {
        let store = Store::for_test().await;
        let result = store
            .list_sessions(10, false, None, Some("not_base64_at_all!!"))
            .await;
        assert!(
            result.is_err(),
            "garbage cursor must be rejected rather than silently ignored"
        );
    }

    #[tokio::test]
    async fn delete_session_cascades_to_children() {
        let store = Store::for_test().await;
        let parent = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create parent");
        let child = store
            .create_child_session(
                parent,
                None,
                Vec::new(),
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("create child")
            .0;

        // Populate the child with a message and a tool_output so the cascade has something to clean
        // up. This proves the descendant deletions run, not just the parent row.
        store
            .save_message(child, "user", "hello from sub-agent")
            .await
            .expect("save_message");
        store
            .save_scratchpad_entry(child, "fixture", "tool body")
            .await
            .expect("save_scratchpad_entry");

        let deleted = store.delete_session(parent).await.expect("delete parent");
        assert!(deleted);
        assert!(
            !store.session_exists(parent).await.expect("exists check"),
            "parent should be gone"
        );
        assert!(
            !store.session_exists(child).await.expect("exists check"),
            "child should be cascaded"
        );
        assert!(
            store
                .load_scratchpad_entry(child, "fixture")
                .await
                .expect("load")
                .is_none(),
            "child's tool_output should be gone"
        );
    }
}
