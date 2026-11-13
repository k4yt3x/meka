//! The portable form of a session: the JSON envelope `meka session export` writes and `meka
//! session import` reads, and the planning that turns an archive back into rows.

use super::*;

/// On-wire format version for `meka session export --format json`. Bumped when the envelope shape
/// or the underlying [`crate::conversation::Event`] serialization changes, a field added or taken
/// away included: the reader takes this version's shape alone, and an older archive the migration
/// module knows is brought forward before it is read. `meka session import` refuses any other
/// version.
pub(crate) const SESSION_EXPORT_FORMAT_VERSION: u32 = 29;
/// Decode an archive, refusing one written for another `format_version` before its shape is read.
///
/// The version is read on its own first, because a release that changed the shape also changed
/// the version, and an archive from one is better refused by the number that names the remedy
/// than by whichever field it spelled differently. [`plan_import`] checks the version again for a
/// caller that built the envelope itself.
pub(crate) fn parse_session_export(raw: &[u8]) -> crate::error::Result<SessionExport> {
    #[derive(serde::Deserialize)]
    struct Envelope {
        format_version: u32,
    }
    let invalid = |error: serde_json::Error| {
        MekaError::Usage(format!("invalid session export JSON: {error}"))
    };
    let envelope: Envelope = serde_json::from_slice(raw).map_err(invalid)?;
    if envelope.format_version == SESSION_EXPORT_FORMAT_VERSION {
        return serde_json::from_slice(raw).map_err(invalid);
    }
    // Only the migration module knows what an older Meka wrote, so this door hands the document
    // over and reads the answer: brought forward, or not a version it knows.
    let mut document: serde_json::Value = serde_json::from_slice(raw).map_err(invalid)?;
    if !super::migrations::bring_archive_forward(
        &mut document,
        u64::from(SESSION_EXPORT_FORMAT_VERSION),
    ) {
        return Err(unsupported_format_version(envelope.format_version));
    }
    serde_json::from_value(document).map_err(invalid)
}

/// Whether this build reads archives of `version` as they are.
fn check_format_version(version: u32) -> crate::error::Result<()> {
    if version != SESSION_EXPORT_FORMAT_VERSION {
        return Err(unsupported_format_version(version));
    }
    Ok(())
}

fn unsupported_format_version(version: u32) -> MekaError {
    MekaError::Usage(format!(
        "unsupported session export format_version {version} (this build supports \
         {SESSION_EXPORT_FORMAT_VERSION})"
    ))
}

/// Sessions one `POST /v1/sessions/import` will accept.
///
/// Enforced by the HTTP handler, not by [`plan_import`], because the reason for it is
/// contention-specific: `import_sessions` runs the whole tree in one closure on the process's
/// single SQLite connection, so every other in-flight request queues behind it. A one-shot
/// `meka session import` restoring its own backup has nothing to contend with, and refusing it
/// would mean a tree that exported fine cannot be restored.
pub(crate) const MAX_IMPORT_SESSIONS: usize = 1_000;
/// Root envelope for a JSON session export. Carries the session plus any sub-agent descendants as a
/// flat, root-first list; parent links are by original id and get remapped on import. Deliberately
/// secret-free: credentials live in separate global tables and the `token_id` fingerprint is
/// omitted.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct SessionExport {
    pub(crate) format_version: u32,
    pub(crate) meka_version: String,
    pub(crate) exported_at: String,
    pub(crate) root_session_id: String,
    /// Reachable from outside because `POST /v1/sessions/import` enforces
    /// [`MAX_IMPORT_SESSIONS`] on the parsed body before handing it to [`plan_import`]; see that
    /// constant for why the cap is the handler's and not the planner's.
    pub(crate) sessions: Vec<ExportedSession>,
    /// The bytes behind every image block in `sessions`, once each; empty when there are none.
    pub(crate) blobs: Vec<ExportedBlob>,
}
/// One image's bytes in an archive, under the content hash its blocks reference.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct ExportedBlob {
    pub(crate) hash: String,
    pub(crate) media_type: String,
    /// The bytes, base64-encoded: the archive is JSON.
    pub(crate) data: String,
}
/// What this installation configures, which [`plan_import`] settles every imported session against.
///
/// A session runs on the profile its row names and files its memories under the project it names,
/// and an import is a door that writes such a row, so it answers to the same rule as every other:
/// each name is one this installation has, or the import is refused. `--profile` is the explicit
/// act that moves a session onto another profile; a project the archive names is added first.
///
/// The level and the approvals switch are this installation's to say as well, as a new session's
/// are, whatever the archive recorded. A directory the archive names is recorded where it is a
/// directory here, as one a client names is.
pub(crate) struct ImportInstallation<'a> {
    /// `--profile <name>`, when it was passed: every imported session takes it, whatever the
    /// archive recorded.
    pub(crate) selected: Option<&'a str>,
    /// What a session the archive records no profile for adopts when no flag was passed: this
    /// installation's default, which is the only thing that can be known about such a session.
    pub(crate) default: Option<&'a str>,
    /// The profiles this installation configures, which every name written has to be among.
    /// `None` only where there is no installation to check against, which is the planner's own
    /// tests: `meka session import` refuses ahead of planning when `config.toml` cannot be read,
    /// and the HTTP server always holds a parsed one.
    pub(crate) configured:
        Option<&'a std::collections::BTreeMap<String, crate::config::ProfileConfig>>,
    /// The projects this installation's store has, which a project the archive records has to be
    /// among.
    pub(crate) projects: Vec<String>,
    /// The level a new session starts at on the importing run, which every root session the
    /// archive carries takes whatever it recorded, and which a sub-agent's row is held within.
    /// `--permission` is the explicit act for another level, as `--profile` is for the profile.
    pub(crate) permission: crate::permission::Permission,
    /// `[permissions].approvals`, which every imported session takes.
    pub(crate) approvals: bool,
    /// The rule every door that records a directory holds one to, answering its canonical
    /// spelling, or why it is not a directory here in words that follow the path; a directory it
    /// refuses is not recorded. Handed in because the module that defines it sits above the
    /// store.
    pub(crate) accept_directory:
        fn(&std::path::Path) -> std::result::Result<std::path::PathBuf, String>,
}
#[cfg(test)]
impl ImportInstallation<'static> {
    /// The planner with no installation behind it, for a test to state over what it is about:
    /// nothing selected, defaulted or configured, the `read` level with approvals off, and every
    /// directory taken as spelled.
    pub(crate) fn for_test() -> Self {
        Self {
            selected: None,
            default: None,
            configured: None,
            projects: Vec::new(),
            permission: crate::permission::Permission::Read,
            approvals: false,
            accept_directory: |directory| Ok(directory.to_path_buf()),
        }
    }
}
/// What [`plan_import`] settled: the sessions to write, the image bytes they reference, and the
/// id the archive's root was given.
pub(crate) struct ImportPlan {
    pub(crate) records: Vec<crate::store::ImportSessionRecord>,
    pub(crate) blobs: Vec<super::blobs::StoredBlob>,
    pub(crate) root_new_id: uuid::Uuid,
}
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct ExportedSession {
    pub(crate) id: String,
    pub(crate) parent_id: Option<String>,
    pub(crate) created_at: String,
    pub(crate) updated_at: String,
    /// A string in the archive, a path in the struct, like `additional_workspace_directories`
    /// below.
    pub(crate) working_directory: Option<std::path::PathBuf>,
    /// The level the session ran at, as its one spelling; an archive naming anything else is
    /// refused at parse rather than written to a row no reader could resolve.
    pub(crate) permission: Option<crate::permission::Permission>,
    /// Whether calls above the level were submitted for approval.
    pub(crate) approvals: bool,
    pub(crate) capabilities_json: Option<String>,
    /// The workspace beyond the working directory; empty for a single-directory session.
    pub(crate) additional_workspace_directories: Vec<std::path::PathBuf>,
    /// A sub-agent's spawn terms; `null` for a session nothing spawned, which an absent key also
    /// reads as, by serde's rule for an `Option`.
    pub(crate) subagent_spec_json: Option<String>,
    /// The profile the session ran on; empty when the archive names none, which [`plan_import`]
    /// settles against the importing installation's default rather than storing a profile no
    /// configuration can name.
    pub(crate) profile: String,
    /// The project the session was on; absent for none, which is also what an archive written
    /// before projects reads as.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) project: String,
    /// The title a user set; absent when none was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) title: Option<String>,
    /// When the session was pinned; absent when it was not, restored as written.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) pinned_at: Option<String>,
    pub(crate) stats: crate::stats::SessionStatsSnapshot,
    /// Every turn that began on the session, oldest first, as its rows record them.
    pub(crate) turns: Vec<ExportedTurn>,
    pub(crate) events: Vec<ExportedEvent>,
    pub(crate) scratchpad_entries: std::collections::BTreeMap<String, String>,
}
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct ExportedEvent {
    /// RFC 3339 timestamp the event row was persisted; preserved across import.
    pub(crate) at: String,
    pub(crate) event: crate::conversation::Event,
    /// The turn that added the row, one of the session's `turns`, or none.
    pub(crate) turn_id: Option<String>,
    /// For a copy a compaction kept, the position among the session's `events` of the one it
    /// copies, always an earlier one: history shows that one and the model reads this.
    pub(crate) copy_of: Option<usize>,
    /// For a copy a compaction kept, the turn that said it, one of the session's `turns`; absent
    /// for one said before turns were recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) said_turn_id: Option<String>,
    /// For a copy a compaction kept, when it was said; absent on every other event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) said_at: Option<String>,
    /// For a reply, what its provider said about it beside its content; absent on every other
    /// event and on a reply written before Meka recorded it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) reply: Option<crate::conversation::ReplyRecord>,
}
/// A turn as its row records it: the stored names (`status`, `error_type`) rather than the API's
/// spellings, so an import writes the row back as it was.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct ExportedTurn {
    pub(crate) id: String,
    pub(crate) source: String,
    pub(crate) started_at: String,
    pub(crate) ended_at: Option<String>,
    pub(crate) status: Option<String>,
    pub(crate) stop_reason: Option<String>,
    pub(crate) error_type: Option<String>,
    pub(crate) detail: Option<String>,
    /// What a turn that failed on a refusal records in place of `detail`.
    pub(crate) refusal: Option<crate::error::RecordedRefusal>,
    pub(crate) cancel_reason: Option<String>,
    /// The profile that served the turn when it was not the session's own.
    pub(crate) profile: Option<String>,
    /// The permission level the turn ran under.
    pub(crate) permission: String,
    pub(crate) usage: crate::store::turns::TurnUsage,
}
impl From<crate::store::turns::TurnRecord> for ExportedTurn {
    fn from(turn: crate::store::turns::TurnRecord) -> Self {
        let (error_type, detail, refusal) = match turn.error {
            Some(error) => {
                let kind = Some(error.kind.name().to_string());
                match error.failure {
                    crate::error::RecordedFailure::Sentence(sentence) => {
                        (kind, Some(sentence), None)
                    }
                    crate::error::RecordedFailure::Refusal(refusal) => (kind, None, Some(refusal)),
                }
            }
            None => (None, None, None),
        };
        Self {
            id: turn.id.to_string(),
            source: turn.source.name().to_string(),
            started_at: turn.started_at,
            ended_at: turn.ended_at,
            status: turn.status.map(|status| status.name().to_string()),
            stop_reason: turn
                .stop_reason
                .map(|stop_reason| stop_reason.name().to_string()),
            error_type,
            detail,
            refusal,
            cancel_reason: turn.reason.map(|reason| reason.name().to_string()),
            profile: turn.profile,
            permission: turn.permission.name().to_string(),
            usage: turn.usage,
        }
    }
}
/// Assemble the structured JSON export envelope for a session and every sub-agent descendant.
/// Per-event timestamps and cumulative stats are preserved; `token_id` is intentionally excluded.
pub(crate) async fn build_session_export(
    store: &Store,
    root: uuid::Uuid,
) -> crate::error::Result<SessionExport> {
    let tree = store.load_session_tree(root).await?;
    let mut sessions = Vec::with_capacity(tree.len());
    let mut referenced: Vec<String> = Vec::new();
    for meta in tree {
        let loaded = store.load_events_with_timestamps(meta.id).await?;
        let positions: std::collections::HashMap<i64, usize> = loaded
            .iter()
            .enumerate()
            .map(|(position, (stamp, _))| (stamp.id, position))
            .collect();
        let events: Vec<ExportedEvent> = loaded
            .into_iter()
            .map(|(stamp, event)| ExportedEvent {
                at: stamp.created_at,
                event,
                turn_id: stamp.turn_id.map(|id| id.to_string()),
                copy_of: stamp
                    .copy_of
                    .and_then(|original| positions.get(&original).copied()),
                said_turn_id: stamp
                    .said
                    .as_ref()
                    .and_then(|said| said.turn_id)
                    .map(|id| id.to_string()),
                said_at: stamp.said.map(|said| said.created_at),
                reply: stamp.reply,
            })
            .collect();
        let turns: Vec<ExportedTurn> = store
            .turn_store()
            .load_turns(meta.id)
            .await?
            .into_iter()
            .map(ExportedTurn::from)
            .collect();
        for exported in &events {
            for hash in super::blobs::blob_references(&exported.event) {
                if !referenced.contains(&hash) {
                    referenced.push(hash);
                }
            }
        }
        let scratchpad_entries = store
            .load_all_scratchpad_entries(meta.id)
            .await?
            .into_iter()
            .collect();
        let stats = store.load_session_stats(meta.id).await?;
        sessions.push(ExportedSession {
            id: meta.id.to_string(),
            parent_id: meta.parent_id.map(|id| id.to_string()),
            created_at: meta.created_at,
            updated_at: meta.updated_at,
            working_directory: meta.cwd,
            permission: meta.permission,
            approvals: meta.approvals,
            capabilities_json: meta.capabilities_json,
            additional_workspace_directories: meta.additional_roots,
            subagent_spec_json: meta.subagent_spec_json,
            profile: meta.profile,
            project: meta.project,
            title: meta.title,
            pinned_at: meta.pinned_at,
            stats,
            turns,
            events,
            scratchpad_entries,
        });
    }
    let blobs = store
        .load_blobs(referenced)
        .await?
        .into_iter()
        .map(|blob| ExportedBlob {
            hash: blob.hash,
            media_type: blob.media_type,
            data: {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD.encode(&blob.bytes)
            },
        })
        .collect();
    Ok(SessionExport {
        format_version: SESSION_EXPORT_FORMAT_VERSION,
        meka_version: env!("CARGO_PKG_VERSION").to_string(),
        exported_at: chrono::Utc::now().to_rfc3339(),
        root_session_id: root.to_string(),
        sessions,
        blobs,
    })
}
/// An archive stamp as the store writes one: RFC 3339 in UTC, the shape every reader compares as
/// text, so an archive's `+05:00` or `Z` is converted here rather than misordered against the
/// store's own stamps. A stamp that is not RFC 3339 is refused.
fn utc_rfc3339(session: &str, what: &str, stamp: &mut String) -> crate::error::Result<()> {
    let parsed = chrono::DateTime::parse_from_rfc3339(stamp).map_err(|error| {
        crate::error::MekaError::Usage(format!(
            "session '{session}' carries a `{what}` that is not RFC 3339 ('{stamp}'): {error}"
        ))
    })?;
    *stamp = parsed.with_timezone(&chrono::Utc).to_rfc3339();
    Ok(())
}

/// Turn a deserialized [`SessionExport`] into the parents-first
/// [`crate::store::ImportSessionRecord`] list to persist, plus the freshly-minted root session
/// ID. Validates the format version, mints a new ID per session, and remaps parent links (a parent
/// pointing outside the exported set collapses to `None`, importing that session as a new root
/// session). Pure and I/O-free so the ID-remap and ordering are unit-testable.
pub(crate) fn plan_import(
    mut export: SessionExport,
    // How each session's profile and authority are settled, here rather than at the reader
    // because an import is where a session enters *this* installation.
    installation: ImportInstallation<'_>,
) -> crate::error::Result<ImportPlan> {
    check_format_version(export.format_version)?;
    if export.sessions.is_empty() {
        return Err(crate::error::MekaError::Usage(
            "session export contains no sessions".to_string(),
        ));
    }
    // Caught here rather than at the `sessions.id` primary key, which would surface a caller's
    // malformed envelope as an internal error.
    let mut seen = std::collections::HashSet::with_capacity(export.sessions.len());
    for session in &export.sessions {
        if !seen.insert(session.id.clone()) {
            return Err(crate::error::MekaError::Usage(format!(
                "session export contains duplicate id '{}'",
                session.id
            )));
        }
    }

    // A stamp Meka did not write is refused ahead of any row: every listing reads them back, and
    // a column cut from one has to be cut from a shape it knows.
    for session in &mut export.sessions {
        let id = session.id.clone();
        for (what, stamp) in [
            ("created_at", Some(&mut session.created_at)),
            ("updated_at", Some(&mut session.updated_at)),
            ("pinned_at", session.pinned_at.as_mut()),
        ] {
            if let Some(stamp) = stamp {
                utc_rfc3339(&id, what, stamp)?;
            }
        }
        for event in &mut session.events {
            utc_rfc3339(&id, "an event's `at`", &mut event.at)?;
            if let Some(said_at) = event.said_at.as_mut() {
                utc_rfc3339(&id, "an event's `said_at`", said_at)?;
            }
        }
    }

    let remap: std::collections::HashMap<String, uuid::Uuid> = export
        .sessions
        .iter()
        .map(|session| (session.id.clone(), uuid::Uuid::new_v4()))
        .collect();
    let root_new_id = remap.get(&export.root_session_id).copied().ok_or_else(|| {
        crate::error::MekaError::Usage(
            "root_session_id is not present in the sessions list".to_string(),
        )
    })?;

    let nodes: Vec<(String, Option<String>)> = export
        .sessions
        .iter()
        .map(|session| (session.id.clone(), session.parent_id.clone()))
        .collect();
    let order = parents_first_order(&nodes)?;

    // Checked once, ahead of any session being planned: a flag naming nothing moves nothing.
    if let (Some(name), Some(configured)) = (installation.selected, installation.configured) {
        crate::config::require_profile(name, configured).map_err(|refusal| {
            crate::error::MekaError::Usage(format!(
                "{refusal}; pass a configured name to `--profile`"
            ))
        })?;
    }

    // Decoded and checked against their hashes before any session is planned: a blob whose bytes
    // do not hash to the name its blocks reference would be served under a name it does not
    // deserve, by every reader that trusts the name.
    let mut blobs = Vec::with_capacity(export.blobs.len());
    for blob in export.blobs {
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(blob.data.as_bytes())
            .map_err(|error| {
                crate::error::MekaError::Usage(format!(
                    "archive blob {} is not base64: {error}",
                    blob.hash
                ))
            })?;
        // Served back as the `Content-Type` of the blob route, so a label that is not an image
        // would have a client's browser render the bytes as whatever it names.
        if !crate::image::is_raster_media_type(&blob.media_type) {
            return Err(crate::error::MekaError::Usage(format!(
                "archive blob {} is labeled '{}', which is not an image type",
                blob.hash, blob.media_type
            )));
        }
        let actual = super::blobs::content_hash(&bytes);
        if actual != blob.hash {
            return Err(crate::error::MekaError::Usage(format!(
                "archive blob {} does not hash to its name (bytes hash to {actual})",
                blob.hash
            )));
        }
        blobs.push(super::blobs::StoredBlob {
            hash: blob.hash,
            media_type: blob.media_type,
            bytes,
        });
    }

    let mut slots: Vec<Option<ExportedSession>> = export.sessions.into_iter().map(Some).collect();
    let mut records = Vec::with_capacity(order.len());
    for index in order {
        let session = slots.get_mut(index).and_then(Option::take).ok_or_else(|| {
            crate::error::MekaError::Usage(
                "duplicate session index while ordering import".to_string(),
            )
        })?;
        let new_id = remap.get(&session.id).copied().ok_or_else(|| {
            crate::error::MekaError::Usage(
                "internal error: session id missing from id remap".to_string(),
            )
        })?;
        let new_parent_id = session
            .parent_id
            .as_ref()
            .and_then(|parent| remap.get(parent).copied());
        // Turns take ids of their own, as sessions do, and the events that name them follow.
        let mut turn_ids: std::collections::HashMap<String, uuid::Uuid> =
            std::collections::HashMap::new();
        let mut turns = Vec::with_capacity(session.turns.len());
        for mut turn in session.turns {
            utc_rfc3339(&session.id, "a turn's started_at", &mut turn.started_at)?;
            if let Some(ended_at) = &mut turn.ended_at {
                utc_rfc3339(&session.id, "a turn's ended_at", ended_at)?;
            }
            if turn
                .source
                .parse::<crate::store::turns::TurnSource>()
                .is_err()
            {
                return Err(crate::error::MekaError::Usage(format!(
                    "session {} in the archive records a turn source Meka does not know: '{}'",
                    session.id, turn.source
                )));
            }
            if turn
                .permission
                .parse::<crate::permission::Permission>()
                .is_err()
            {
                return Err(crate::error::MekaError::Usage(format!(
                    "session {} in the archive records a permission level Meka does not know: \
                     '{}'",
                    session.id, turn.permission
                )));
            }
            if let Some(reason) = &turn.cancel_reason
                && reason.parse::<crate::store::turns::CancelReason>().is_err()
            {
                return Err(crate::error::MekaError::Usage(format!(
                    "session {} in the archive records a cancel reason Meka does not know: \
                     '{reason}'",
                    session.id
                )));
            }
            if let Some(status) = &turn.status
                && status.parse::<crate::store::turns::TurnStatus>().is_err()
            {
                return Err(crate::error::MekaError::Usage(format!(
                    "session {} in the archive records a turn status Meka does not know: '{status}'",
                    session.id
                )));
            }
            if let Some(error_type) = &turn.error_type
                && crate::error::ErrorKind::from_name(error_type).is_none()
            {
                return Err(crate::error::MekaError::Usage(format!(
                    "session {} in the archive records an error type Meka does not know: \
                     '{error_type}'",
                    session.id
                )));
            }
            let new_turn = uuid::Uuid::new_v4();
            turn_ids.insert(
                std::mem::replace(&mut turn.id, new_turn.to_string()),
                new_turn,
            );
            turns.push(turn);
        }
        let mut events: Vec<crate::store::ImportedEvent> = Vec::with_capacity(session.events.len());
        // What the view the latest compaction replaced showed of each message row it held, by the
        // row's position: what a copy of that row may hold, whatever the view rewrote on the way.
        // `None` when that view cannot be replayed, a turn sweep having taken rows a repair in it
        // rewrote: a copy then stands for any message row the view was built from.
        let mut shown_before_boundary: Option<
            std::collections::HashSet<(usize, (String, String))>,
        > = Some(std::collections::HashSet::new());
        let mut replaced = 0..0;
        let mut since_boundary = 0;
        for event in session.events {
            if matches!(
                event.event,
                crate::conversation::Event::CompactBoundary { .. }
            ) {
                let window: Vec<(usize, crate::conversation::Event)> = events
                    .iter()
                    .enumerate()
                    .skip(since_boundary)
                    .map(|(position, earlier)| (position, earlier.event.clone()))
                    .collect();
                let view = crate::conversation::materialize_annotated(&window);
                replaced = since_boundary..events.len();
                shown_before_boundary = view.whole.then(|| {
                    view.messages
                        .iter()
                        .zip(&view.stamps)
                        .filter(|(_, position)| {
                            matches!(
                                events.get(**position).map(|earlier| &earlier.event),
                                Some(crate::conversation::Event::Append(_))
                            )
                        })
                        .filter_map(|(message, position)| {
                            match super::sessions::encode_message_for_db(message) {
                                Ok(held) => Some((*position, held)),
                                Err(_) => None,
                            }
                        })
                        .collect()
                });
                since_boundary = events.len();
            }
            let carried = |turn: Option<String>| -> crate::error::Result<Option<uuid::Uuid>> {
                match turn {
                    Some(old) => Ok(Some(turn_ids.get(&old).copied().ok_or_else(|| {
                        crate::error::MekaError::Usage(format!(
                            "session {} in the archive has an event naming turn '{old}', which \
                             the archive does not carry",
                            session.id
                        ))
                    })?)),
                    None => Ok(None),
                }
            };
            let turn = carried(event.turn_id)?;
            let said = match (event.said_turn_id, event.said_at) {
                (turn, Some(at))
                    if matches!(event.event, crate::conversation::Event::Append(_)) =>
                {
                    Some(crate::store::SaidStamp {
                        turn_id: carried(turn)?,
                        created_at: at,
                    })
                }
                (_, Some(_)) => {
                    return Err(crate::error::MekaError::Usage(format!(
                        "session {} in the archive has event {position} saying where it was said, \
                         which only a message does",
                        session.id,
                        position = events.len()
                    )));
                }
                (None, None) => None,
                (Some(_), None) => {
                    return Err(crate::error::MekaError::Usage(format!(
                        "session {} in the archive has event {position} naming the turn it was \
                         said in but not when",
                        session.id,
                        position = events.len()
                    )));
                }
            };
            // A link hides the copy from every history, so it is taken only in the shapes a
            // compaction writes: a message holding what an earlier row was shown as in the view
            // the latest compaction replaced, or repeating an earlier one word for word below a
            // boundary the original is above. On anything else it would hide what it names.
            if let Some(original) = event.copy_of {
                let stands_for = match (&event.event, &shown_before_boundary) {
                    (crate::conversation::Event::Append(message), Some(shown)) => {
                        super::sessions::encode_message_for_db(message)
                            .is_ok_and(|held| shown.contains(&(original, held)))
                    }
                    (crate::conversation::Event::Append(_), None) => {
                        replaced.contains(&original)
                            && matches!(
                                events.get(original).map(|earlier| &earlier.event),
                                Some(crate::conversation::Event::Append(_))
                            )
                    }
                    _ => false,
                };
                let repeats = |earlier: &crate::conversation::Event| {
                    matches!(
                        (earlier, &event.event),
                        (
                            crate::conversation::Event::Append(_),
                            crate::conversation::Event::Append(_)
                        )
                    ) && matches!(
                        (
                            super::sessions::encode_event_for_db(earlier),
                            super::sessions::encode_event_for_db(&event.event),
                        ),
                        (Ok(earlier), Ok(copy)) if earlier == copy
                    )
                };
                let copies_a_message = events
                    .get(original)
                    .is_some_and(|earlier: &crate::store::ImportedEvent| repeats(&earlier.event))
                    && events.iter().skip(original + 1).any(|between| {
                        matches!(
                            between.event,
                            crate::conversation::Event::CompactBoundary { .. }
                        )
                    });
                if !stands_for && !copies_a_message {
                    return Err(crate::error::MekaError::Usage(format!(
                        "session {} in the archive has event {position} copying event {original}, \
                         which is not an earlier message it repeats from before a compaction",
                        session.id,
                        position = events.len()
                    )));
                }
            }
            events.push(crate::store::ImportedEvent {
                at: event.at,
                event: event.event,
                turn_id: turn,
                copy_of: event.copy_of,
                said,
                reply: event.reply,
            });
        }
        // A sub-agent's row by the rule `Store::spawn_terms` reads one by. Its level is the
        // lower of what it recorded and this installation's, since a follow-up narrows a worker
        // and never widens it; a root's is this installation's, as a new session's is.
        let spawned = super::sessions::is_spawned(
            new_parent_id.is_some(),
            session.subagent_spec_json.is_some(),
        );
        let permission = match session.permission {
            Some(recorded) if spawned => recorded.greatest_within_both(installation.permission),
            _ => installation.permission,
        };
        let profile = match (
            installation.selected,
            session.profile.is_empty(),
            installation.default,
        ) {
            (Some(name), ..) => name.to_string(),
            // Refused rather than written: a row naming a profile this installation lacks is a
            // session every resume refuses by name, and a state no other door can produce.
            (None, false, _) => {
                if let Some(configured) = installation.configured
                    && !configured.contains_key(&session.profile)
                {
                    return Err(crate::error::MekaError::ImportNamesUnknownProfile {
                        name: session.profile,
                        known: configured.keys().cloned().collect(),
                    });
                }
                session.profile
            }
            (None, true, Some(default)) => default.to_string(),
            // Refused rather than left blank, for the same reason: a session with no profile
            // cannot run, and a blank is a state no other door can produce.
            (None, true, None) => {
                return Err(crate::error::MekaError::ImportNamesNoProfile);
            }
        };
        // A directory this installation would refuse from a client is left out rather than
        // recorded: a row without a working directory opens where its host does.
        let accepted = |noun: &str, directory: &std::path::Path| match (installation
            .accept_directory)(
            directory
        ) {
            Ok(canonical) => Some(canonical),
            Err(reason) => {
                tracing::info!(
                    "session {id} in the archive is imported without its {noun} '{path}', \
                         which {reason}",
                    id = session.id,
                    path = directory.display()
                );
                None
            }
        };
        // Refused rather than written, as a profile is: a row naming a project this installation
        // lacks is a session every resume refuses by name.
        if !session.project.is_empty() && !installation.projects.contains(&session.project) {
            return Err(crate::error::MekaError::ImportNamesUnknownProject {
                name: session.project,
                known: installation.projects,
            });
        }
        records.push(crate::store::ImportSessionRecord {
            new_id,
            new_parent_id,
            created_at: session.created_at,
            cwd: session
                .working_directory
                .as_deref()
                .and_then(|directory| accepted("working directory", directory)),
            permission,
            approvals: installation.approvals,
            capabilities_json: session.capabilities_json,
            additional_roots: session
                .additional_workspace_directories
                .iter()
                .filter_map(|root| accepted("workspace directory", root))
                .collect(),
            subagent_spec_json: session.subagent_spec_json,
            profile,
            project: session.project,
            // Through the same acceptor every door that sets one uses, so an archive cannot
            // plant a title no `PATCH` could.
            title: match session.title.as_deref() {
                Some(title) => crate::store::normalize_title(title)?,
                None => None,
            },
            pinned_at: session.pinned_at,
            stats: session.stats,
            turns,
            events,
            scratchpad_entries: session.scratchpad_entries.into_iter().collect(),
        });
    }

    Ok(ImportPlan {
        records,
        blobs,
        root_new_id,
    })
}
/// Order sessions parents-first (a topological sort over `parent_id` edges, considering only
/// parents present in the set) so an importer can insert each session after its parent and satisfy
/// the `parent_session_id` foreign key. Returns indices into `nodes`. Errors on a cyclic
/// relationship. Sessions whose parent is absent from the set are treated as roots.
pub(crate) fn parents_first_order(
    nodes: &[(String, Option<String>)],
) -> crate::error::Result<Vec<usize>> {
    use std::collections::{HashMap, VecDeque};

    let index_of: HashMap<&str, usize> = nodes
        .iter()
        .enumerate()
        .map(|(index, (id, _))| (id.as_str(), index))
        .collect();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    let mut indegree = vec![0usize; nodes.len()];
    for (index, (_, parent)) in nodes.iter().enumerate() {
        if let Some(parent) = parent
            && let Some(&parent_index) = index_of.get(parent.as_str())
            && let Some(siblings) = children.get_mut(parent_index)
            && let Some(count) = indegree.get_mut(index)
        {
            siblings.push(index);
            *count += 1;
        }
    }
    let mut queue: VecDeque<usize> = indegree
        .iter()
        .enumerate()
        .filter(|&(_, &count)| count == 0)
        .map(|(index, _)| index)
        .collect();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(node) = queue.pop_front() {
        order.push(node);
        for &child in children.get(node).into_iter().flatten() {
            if let Some(count) = indegree.get_mut(child) {
                *count -= 1;
                if *count == 0 {
                    queue.push_back(child);
                }
            }
        }
    }
    if order.len() != nodes.len() {
        return Err(crate::error::MekaError::Usage(
            "session export has a cyclic parent relationship".to_string(),
        ));
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A turn a compaction kept that the turn sweep then takes is told by its copy, which holds
    /// what was said, not the context's rewrite of it: the reasoning the view took out, and the
    /// reply's record.
    #[tokio::test]
    async fn a_released_copy_holds_what_its_row_held() {
        use crate::conversation::{
            ContentBlock, Event, Message, ReplyRecord, Role, Stop, StopReason,
        };
        let store = crate::store::Store::for_test().await;
        let id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("session");
        let (old, new) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
        let long_ago = (chrono::Utc::now() - chrono::TimeDelta::days(90)).to_rfc3339();
        let recent = chrono::Utc::now().to_rfc3339();
        for (turn, at) in [(old, &long_ago), (new, &recent)] {
            store
                .turn_store()
                .open_turn(
                    id,
                    turn,
                    crate::store::turns::TurnSource::Client,
                    crate::permission::Permission::Read,
                    at,
                )
                .await
                .expect("turn");
        }
        let answer = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::RedactedThinking {
                    data: "sealed".to_string(),
                    provider_fields: None,
                },
                ContentBlock::Text {
                    text: "Seen.".to_string(),
                    provider_fields: None,
                },
            ],
            provider_fields: None,
        };
        let record = ReplyRecord {
            stop: Some(Stop::of(StopReason::EndTurn)),
            cut: None,
            model: Some("a-model".to_string()),
            profile: None,
            usage: None,
            provider_fields: None,
        };
        store
            .save_events_atomic(
                id,
                vec![
                    crate::store::RecordedEvent::from(Event::Append(Message::user("look?"))),
                    crate::store::RecordedEvent::reply(answer.clone(), record.clone()),
                    // Claude rejected a signature: the view takes the thinking out.
                    Event::Unsign.into(),
                ],
                Some(old),
            )
            .await
            .expect("save");
        store
            .turn_store()
            .close_turn(
                old,
                &crate::store::turns::TurnEnding::succeeded(
                    StopReason::EndTurn,
                    crate::store::turns::TurnUsage::default(),
                ),
            )
            .await
            .expect("close");
        store
            .run_sql_for_test(&format!(
                "UPDATE turns SET ended_at = '{long_ago}' WHERE id = '{old}'"
            ))
            .await;
        store
            .save_event(id, &Event::Append(Message::user("next?")), Some(new))
            .await
            .expect("save");
        let boundary = Event::CompactBoundary {
            summary: Message::user("summary"),
            replaced_count: 3,
            record: crate::conversation::CompactionRecord::for_test(),
            loaded_tools_snapshot: Default::default(),
            checklist_snapshot: Default::default(),
        };
        // The view's tail, as the compaction copies it: the reply without its thinking.
        store
            .save_compaction(
                id,
                &boundary,
                &[
                    Message::user("look?"),
                    Message::assistant_text("Seen."),
                    Message::user("next?"),
                ],
                Some(new),
            )
            .await
            .expect("compaction");
        let swept = store
            .delete_expired_turns(chrono::Utc::now() - chrono::TimeDelta::days(30), None)
            .await
            .expect("sweep");
        assert!(swept.rows > 0, "the earlier turn goes");
        let told = store
            .load_events_with_timestamps(id)
            .await
            .expect("load")
            .into_iter()
            .find_map(|(stamp, event)| match event {
                Event::Append(message) if message.role == Role::Assistant => {
                    Some((message, stamp.reply))
                }
                _ => None,
            })
            .expect("the copy tells the reply");
        assert_eq!(told, (answer, Some(record)));
    }

    /// A turn sweep that takes the turns a retry's rewrite reached back into leaves that rewrite
    /// unable to replay below the compaction that kept its message, and the archive of what is
    /// left still imports, the copy still standing for the row it was said in.
    #[tokio::test]
    async fn an_archive_exported_after_a_sweep_imports_a_rewritten_copy() {
        use crate::conversation::{Event, Message};
        let store = crate::store::Store::for_test().await;
        let id = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("session");
        let (old, new) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
        let long_ago = (chrono::Utc::now() - chrono::TimeDelta::days(90)).to_rfc3339();
        let recent = chrono::Utc::now().to_rfc3339();
        for (turn, at) in [(old, &long_ago), (new, &recent)] {
            store
                .turn_store()
                .open_turn(
                    id,
                    turn,
                    crate::store::turns::TurnSource::Client,
                    crate::permission::Permission::Read,
                    at,
                )
                .await
                .expect("turn");
        }
        for message in [Message::user("one?"), Message::assistant_text("One.")] {
            store
                .save_event(id, &Event::Append(message), Some(old))
                .await
                .expect("save");
        }
        store
            .turn_store()
            .close_turn(
                old,
                &crate::store::turns::TurnEnding::succeeded(
                    crate::conversation::StopReason::EndTurn,
                    crate::store::turns::TurnUsage::default(),
                ),
            )
            .await
            .expect("close");
        store
            .run_sql_for_test(&format!(
                "UPDATE turns SET ended_at = '{long_ago}' WHERE id = '{old}'"
            ))
            .await;
        for event in [
            Event::Append(Message::user("two?")),
            // The retry reaches back past its own prompt into the turn before.
            Event::Repair {
                replaced_count: 2,
                messages: vec![
                    Message::assistant_text("One."),
                    Message::user("two, degraded?"),
                ],
            },
        ] {
            store.save_event(id, &event, Some(new)).await.expect("save");
        }
        let boundary = Event::CompactBoundary {
            summary: Message::user("summary"),
            replaced_count: 3,
            record: crate::conversation::CompactionRecord::for_test(),
            loaded_tools_snapshot: Default::default(),
            checklist_snapshot: Default::default(),
        };
        store
            .save_compaction(id, &boundary, &[Message::user("two, degraded?")], Some(new))
            .await
            .expect("compaction");
        let swept = store
            .delete_expired_turns(chrono::Utc::now() - chrono::TimeDelta::days(30), None)
            .await
            .expect("sweep");
        assert_eq!(swept.rows, 2, "the earlier turn goes");
        let archive = build_session_export(&store, id).await.expect("export");
        let links: Vec<Option<usize>> = archive.sessions[0]
            .events
            .iter()
            .map(|event| event.copy_of)
            .collect();
        assert_eq!(
            links,
            [None, None, None, Some(0), None],
            "the copy stands for its row, and the repair after it restates the view"
        );
        let plan = plan_import(archive, ImportInstallation {
            selected: None,
            default: Some("test-profile"),
            configured: None,
            projects: Vec::new(),
            ..ImportInstallation::for_test()
        })
        .expect("imports");
        assert_eq!(plan.records[0].events[3].copy_of, Some(0));
    }

    /// An archive from another release is refused by its version, which names the remedy, and
    /// not by the first field that release spelled differently: the 0.53 key `tool_outputs` is
    /// missing its 0.54 name here, and the message must not say so.
    #[test]
    fn an_archive_of_another_version_is_refused_by_its_version_before_its_shape() {
        let archive = br#"{"format_version": 2, "sessions": [{"tool_outputs": {}}]}"#;
        let Err(error) = parse_session_export(archive) else {
            panic!("an archive of another version must be refused");
        };
        let error = error.to_string();
        assert!(error.contains("format_version 2"), "{error}");
        assert!(!error.contains("missing field"), "{error}");
    }

    /// The reader takes the current shape alone: a current-version archive missing a field the
    /// exporter always writes is refused by that field, never defaulted. Anything older reaches
    /// the migration module's conversion instead, which is the one place that may fill one in.
    /// `subagent_spec_json` is not in the list: an absent `Option` reads as `null` by serde's own
    /// rule, the same way `title` and `pinned_at` do.
    #[test]
    fn a_current_archive_missing_a_field_is_refused_rather_than_defaulted() {
        let mut archive = serde_json::to_value(archive_on("work")).expect("serialize");
        for field in [
            "approvals",
            "additional_workspace_directories",
            "profile",
            "turns",
        ] {
            let mut lacking = archive.clone();
            lacking["sessions"][0]
                .as_object_mut()
                .expect("session")
                .remove(field);
            let Err(error) = parse_session_export(lacking.to_string().as_bytes()) else {
                panic!("`{field}` must be required");
            };
            assert!(error.to_string().contains(field), "{error}");
        }
        archive.as_object_mut().expect("root").remove("blobs");
        assert!(
            parse_session_export(archive.to_string().as_bytes()).is_err(),
            "`blobs` must be required"
        );
    }

    /// A 0.88 archive, `format_version` 21, reads as this build's version, each copy a compaction
    /// kept saying where it was said: the event it copies.
    #[test]
    fn a_0_88_archive_is_read_with_its_copies_saying_where_they_were_said() {
        let mut archive = serde_json::to_value(archive_on("work")).expect("serialize");
        archive["format_version"] = 21.into();
        let said = |text: &str, at: &str, copy_of: Option<usize>| {
            serde_json::json!({
                "at": at,
                "event": {"Append": crate::conversation::Message::user(text)},
                "turn_id": null,
                "copy_of": copy_of,
            })
        };
        archive["sessions"][0]["events"] = serde_json::json!([
            said("kept", "when it was said", None),
            said("kept", "when it was kept", Some(0)),
        ]);
        let read = parse_session_export(archive.to_string().as_bytes()).expect("brought forward");
        assert_eq!(read.format_version, SESSION_EXPORT_FORMAT_VERSION);
        let says: Vec<Option<&str>> = read.sessions[0]
            .events
            .iter()
            .map(|event| event.said_at.as_deref())
            .collect();
        assert_eq!(says, [None, Some("when it was said")]);
    }

    /// Where a message was said is a stamp like the others: one that is not RFC 3339 is refused,
    /// one in another zone is read in UTC, and one on anything but a message is refused, since
    /// only a message is said.
    #[test]
    fn an_archive_s_said_stamps_are_read_as_every_stamp_is() {
        let profiles = configured(&["work"]);
        let plan = |events: serde_json::Value| {
            let mut archive = serde_json::to_value(archive_on("work")).expect("serialize");
            archive["sessions"][0]["events"] = events;
            plan_import(
                serde_json::from_value(archive).expect("an archive"),
                ImportInstallation {
                    selected: None,
                    default: Some("work"),
                    configured: Some(&profiles),
                    projects: Vec::new(),
                    ..ImportInstallation::for_test()
                },
            )
        };
        let said_at = |event: serde_json::Value, at: &str| {
            serde_json::json!([{
                "at": "2020-01-01T00:00:00Z",
                "event": event,
                "turn_id": null,
                "copy_of": null,
                "said_at": at,
            }])
        };
        let message = || serde_json::json!({"Append": crate::conversation::Message::user("kept")});
        let refused = refusal(plan(said_at(message(), "yesterday")));
        assert!(refused.contains("`said_at`"), "{refused}");
        let read = plan(said_at(message(), "2020-01-01T02:00:00+02:00")).expect("plans");
        assert_eq!(
            read.records[0].events[0]
                .said
                .as_ref()
                .map(|said| said.created_at.as_str()),
            Some("2020-01-01T00:00:00+00:00")
        );
        let refused = refusal(plan(said_at(
            serde_json::json!("Unsign"),
            "2020-01-01T00:00:00Z",
        )));
        assert!(refused.contains("only a message"), "{refused}");
    }

    /// A one-session archive whose session ran on `profile`.
    fn archive_on(profile: &str) -> SessionExport {
        serde_json::from_value(serde_json::json!({
            "format_version": SESSION_EXPORT_FORMAT_VERSION,
            "meka_version": "test",
            "exported_at": "now",
            "root_session_id": "11111111-1111-4111-8111-111111111111",
            "sessions": [{
                "id": "11111111-1111-4111-8111-111111111111",
                "parent_id": null,
                "created_at": "2020-01-01T00:00:00Z",
                "updated_at": "2020-01-01T00:00:00Z",
                "working_directory": null,
                "permission": "read",
                "approvals": false,
                "capabilities_json": null,
                "additional_workspace_directories": [],
                "subagent_spec_json": null,
                "profile": profile,
                "stats": crate::stats::SessionStatsSnapshot::default(),
                "turns": [],
                "events": [],
                "scratchpad_entries": {},
            }],
            "blobs": [],
        }))
        .expect("a well-formed archive")
    }

    fn configured(
        names: &[&str],
    ) -> std::collections::BTreeMap<String, crate::config::ProfileConfig> {
        names
            .iter()
            .map(|name| {
                (name.to_string(), crate::config::ProfileConfig {
                    account: name.to_string(),
                    ..Default::default()
                })
            })
            .collect()
    }

    fn refusal(outcome: crate::error::Result<ImportPlan>) -> String {
        match outcome {
            Ok(_) => panic!("the import must be refused"),
            Err(error) if error.kind() == crate::error::ErrorKind::InvalidBody => error.to_string(),
            Err(other) => panic!("the refusal must be the caller's to act on, got {other}"),
        }
    }

    /// An archive from another installation names the profile it ran on there. Written as it
    /// stands, the row is a session every resume refuses by name; refused here, the remedy is the
    /// one flag that moves a session.
    #[test]
    fn an_archive_naming_a_profile_this_installation_lacks_is_refused() {
        let profiles = configured(&["other"]);
        let message = refusal(plan_import(archive_on("work"), ImportInstallation {
            selected: None,
            default: Some("other"),
            configured: Some(&profiles),
            projects: Vec::new(),
            ..ImportInstallation::for_test()
        }));
        assert!(
            message.contains("'work'") && message.contains("--profile <name> session import"),
            "{message}"
        );
    }

    /// An archive stamp in another zone is stored in UTC, the shape every reader compares as text,
    /// so it orders against the store's own stamps by the instant it names.
    #[test]
    fn an_archive_s_stamps_are_stored_in_utc() {
        let profiles = configured(&["work"]);
        let mut archive = archive_on("work");
        archive.sessions[0].created_at = "2020-01-01T05:00:00+05:00".to_string();
        let plan = plan_import(archive, ImportInstallation {
            selected: None,
            default: Some("work"),
            configured: Some(&profiles),
            projects: Vec::new(),
            ..ImportInstallation::for_test()
        })
        .expect("plans");
        assert_eq!(plan.records[0].created_at, "2020-01-01T00:00:00+00:00");
    }

    /// A link hides the copy from every history, so an archive may draw one only from a message to
    /// an earlier message it repeats word for word, across a compaction: one onto a repair,
    /// forward, onto other words, or with no boundary between, is refused.
    #[test]
    fn an_archive_link_is_refused_unless_it_repeats_an_earlier_message_across_a_compaction() {
        let profiles = configured(&["work"]);
        let plan = |events: serde_json::Value| {
            let mut archive = archive_on("work");
            archive.sessions[0].events =
                serde_json::from_value(events).expect("well-formed events");
            plan_import(archive, ImportInstallation {
                selected: None,
                default: Some("work"),
                configured: Some(&profiles),
                projects: Vec::new(),
                ..ImportInstallation::for_test()
            })
        };
        let saying = |text: &str, copy_of: Option<usize>| {
            serde_json::json!({
                "at": "2020-01-01T00:00:00Z",
                "event": {"Append": {"role": "user", "content": [{"type": "text", "text": text}]}},
                "turn_id": null,
                "copy_of": copy_of,
            })
        };
        let message = |copy_of: Option<usize>| saying("hi", copy_of);
        let repair = serde_json::json!({
            "at": "2020-01-01T00:00:00Z",
            "event": {"Repair": {"replaced_count": 1, "messages": []}},
            "turn_id": null,
            "copy_of": null,
        });
        let boundary = serde_json::json!({
            "at": "2020-01-01T00:00:00Z",
            "event": crate::conversation::Event::CompactBoundary {
                summary: crate::conversation::Message::user("the summary"),
                replaced_count: 1,
                record: crate::conversation::CompactionRecord::for_test(),
                loaded_tools_snapshot: Default::default(),
                checklist_snapshot: Default::default(),
            },
            "turn_id": null,
            "copy_of": null,
        });
        assert!(
            plan(serde_json::json!([
                message(None),
                boundary,
                message(Some(0))
            ]))
            .is_ok()
        );
        assert!(plan(serde_json::json!([message(None), message(Some(0))])).is_err());
        assert!(
            plan(serde_json::json!([
                message(None),
                boundary,
                saying("bye", Some(0))
            ]))
            .is_err()
        );
        assert!(plan(serde_json::json!([repair, boundary, message(Some(0))])).is_err());
        assert!(
            plan(serde_json::json!([
                message(Some(2)),
                boundary,
                message(None)
            ]))
            .is_err()
        );
    }

    /// A blob is served back under the label it carries, so an archive whose blob claims to be a
    /// page rather than an image is refused, however well its bytes hash.
    #[test]
    fn an_archive_blob_labeled_as_anything_but_an_image_is_refused() {
        use base64::Engine;
        let profiles = configured(&["work"]);
        let bytes = b"<script>alert(1)</script>";
        let mut archive = archive_on("work");
        archive.blobs.push(ExportedBlob {
            hash: super::super::blobs::content_hash(bytes),
            media_type: "text/html".to_string(),
            data: base64::engine::general_purpose::STANDARD.encode(bytes),
        });
        let message = refusal(plan_import(archive, ImportInstallation {
            selected: None,
            default: Some("work"),
            configured: Some(&profiles),
            projects: Vec::new(),
            ..ImportInstallation::for_test()
        }));
        assert!(message.contains("'text/html'"), "{message}");
    }

    /// A current archive states every field its turns and boundaries record, so one that leaves a
    /// turn's level, a boundary's checkpoint or its written summary out is refused by the name of
    /// what it lacks, never read with a default in its place.
    #[test]
    fn a_current_archive_missing_a_recorded_field_is_refused_by_its_name() {
        let mut archive = archive_on("work");
        archive.sessions[0].turns.push(ExportedTurn {
            id: "22222222-2222-4222-8222-222222222222".to_string(),
            source: "client".to_string(),
            started_at: "2020-01-01T00:00:00+00:00".to_string(),
            ended_at: None,
            status: None,
            stop_reason: None,
            error_type: None,
            detail: None,
            refusal: None,
            cancel_reason: None,
            profile: None,
            permission: "workspace".to_string(),
            usage: crate::store::turns::TurnUsage::default(),
        });
        archive.sessions[0].events.push(ExportedEvent {
            at: "2020-01-01T00:00:00Z".to_string(),
            event: crate::conversation::Event::CompactBoundary {
                summary: crate::conversation::Message::user("summary"),
                replaced_count: 0,
                loaded_tools_snapshot: Default::default(),
                checklist_snapshot: crate::checklist::ChecklistState::default(),
                record: crate::conversation::CompactionRecord::for_test(),
            },
            turn_id: None,
            copy_of: None,
            said_turn_id: None,
            said_at: None,
            reply: None,
        });
        let whole = serde_json::to_value(&archive).expect("json");
        parse_session_export(whole.to_string().as_bytes()).expect("the whole archive reads");
        for (field, remove) in [
            (
                "permission",
                (|document: &mut serde_json::Value| {
                    document["sessions"][0]["turns"][0]
                        .as_object_mut()
                        .and_then(|turn| turn.remove("permission"))
                }) as fn(&mut serde_json::Value) -> Option<serde_json::Value>,
            ),
            ("checkpoint", |document| {
                document["sessions"][0]["events"][0]["event"]["CompactBoundary"]
                    .as_object_mut()
                    .and_then(|boundary| boundary.remove("checkpoint"))
            }),
            ("written_summary", |document| {
                document["sessions"][0]["events"][0]["event"]["CompactBoundary"]
                    .as_object_mut()
                    .and_then(|boundary| boundary.remove("written_summary"))
            }),
        ] {
            let mut document = whole.clone();
            assert!(
                remove(&mut document).is_some(),
                "the archive states {field}"
            );
            let refusal = match parse_session_export(document.to_string().as_bytes()) {
                Ok(_) => panic!("an archive without {field} must be refused"),
                Err(error) => error.to_string(),
            };
            assert!(refusal.contains(field), "{refusal}");
        }
    }

    /// A turn's source is read back typed, so an archive naming one Meka does not know is refused
    /// at the door rather than written into a row every listing of the session then fails on.
    #[test]
    fn an_archive_turn_with_an_unknown_source_is_refused() {
        let profiles = configured(&["work"]);
        let mut archive = archive_on("work");
        archive.sessions[0].turns.push(ExportedTurn {
            id: uuid::Uuid::new_v4().to_string(),
            source: "screen".to_string(),
            started_at: "2020-01-01T00:00:00+00:00".to_string(),
            ended_at: None,
            status: None,
            stop_reason: None,
            error_type: None,
            detail: None,
            refusal: None,
            cancel_reason: None,
            profile: None,
            permission: "read".to_string(),
            usage: crate::store::turns::TurnUsage::default(),
        });
        let message = refusal(plan_import(archive, ImportInstallation {
            selected: None,
            default: Some("work"),
            configured: Some(&profiles),
            projects: Vec::new(),
            ..ImportInstallation::for_test()
        }));
        assert!(message.contains("'screen'"), "{message}");
    }

    /// A project this installation does not define is refused ahead of any row, as a profile is;
    /// the archive names it, so the remedy is to define it.
    #[test]
    fn an_archive_naming_a_project_this_installation_lacks_is_refused() {
        let profiles = configured(&["work"]);
        let mut archive = archive_on("work");
        archive.sessions[0].project = "movie".to_string();
        let message = refusal(plan_import(archive, ImportInstallation {
            selected: None,
            default: Some("work"),
            configured: Some(&profiles),
            projects: Vec::new(),
            ..ImportInstallation::for_test()
        }));
        assert!(
            message.contains("no project named 'movie'")
                && message.contains("`meka project add movie`"),
            "{message}"
        );
        let mut archive = archive_on("work");
        archive.sessions[0].project = "movie".to_string();
        plan_import(archive, ImportInstallation {
            selected: None,
            default: Some("work"),
            configured: Some(&profiles),
            projects: vec!["movie".to_string()],
            ..ImportInstallation::for_test()
        })
        .expect("a project the store has is planned");
    }

    /// A stamp that is not RFC 3339 is refused ahead of any row: Meka's own are, every listing
    /// reads them back, and `scratchpad_list` cuts one for its column.
    #[test]
    fn an_archive_stamp_that_is_not_rfc_3339_is_refused() {
        let profiles = configured(&["work"]);
        let mut archive = archive_on("work");
        archive.sessions[0].created_at = "yesterday".to_string();
        let message = refusal(plan_import(archive, ImportInstallation {
            selected: None,
            default: Some("work"),
            configured: Some(&profiles),
            projects: Vec::new(),
            ..ImportInstallation::for_test()
        }));
        assert!(
            message.contains("RFC 3339") && message.contains("created_at"),
            "{message}"
        );
    }

    /// `--profile` is the explicit act that moves a session: every imported session takes it,
    /// whatever the archive recorded, and a flag naming nothing moves nothing.
    /// An archive's title goes through the acceptor every door uses: collapsed like a typed one,
    /// and refused past the cap rather than planted on a row no `PATCH` could produce.
    #[test]
    fn an_archived_title_is_accepted_the_way_a_typed_one_is() {
        let profiles = configured(&["work"]);
        let import = |title: &str| {
            let mut archive = archive_on("work");
            archive.sessions[0].title = Some(title.to_string());
            archive.sessions[0].pinned_at = Some("2026-02-02T00:00:00+00:00".to_string());
            plan_import(archive, ImportInstallation {
                selected: None,
                default: Some("work"),
                configured: Some(&profiles),
                projects: Vec::new(),
                ..ImportInstallation::for_test()
            })
        };
        let plan = import("  Research   notes ").expect("a plan");
        assert_eq!(plan.records[0].title.as_deref(), Some("Research notes"));
        assert_eq!(
            plan.records[0].pinned_at.as_deref(),
            Some("2026-02-02T00:00:00+00:00")
        );
        let message = refusal(import(&"x".repeat(201)));
        assert!(message.contains("200 characters"), "{message}");
    }

    #[test]
    fn a_profile_flag_moves_every_imported_session_onto_it() {
        let profiles = configured(&["other"]);
        let plan = plan_import(archive_on("work"), ImportInstallation {
            selected: Some("other"),
            default: None,
            configured: Some(&profiles),
            projects: Vec::new(),
            ..ImportInstallation::for_test()
        })
        .expect("the flag names a configured profile");
        assert!(
            plan.records.iter().all(|record| record.profile == "other"),
            "every session moved onto the selected profile"
        );

        let message = refusal(plan_import(archive_on("work"), ImportInstallation {
            selected: Some("ghost"),
            default: None,
            configured: Some(&profiles),
            projects: Vec::new(),
            ..ImportInstallation::for_test()
        }));
        assert!(message.contains("'ghost'"), "{message}");
    }

    /// An archive recording no profile adopts the installation's default, as before, and is
    /// refused when there is none.
    #[test]
    fn an_archive_naming_no_profile_takes_the_default_or_is_refused() {
        let profiles = configured(&["other"]);
        let plan = plan_import(archive_on(""), ImportInstallation {
            selected: None,
            default: Some("other"),
            configured: Some(&profiles),
            projects: Vec::new(),
            ..ImportInstallation::for_test()
        })
        .expect("the default supplies one");
        assert_eq!(plan.records[0].profile, "other");

        let message = refusal(plan_import(archive_on(""), ImportInstallation {
            selected: None,
            default: None,
            configured: Some(&profiles),
            projects: Vec::new(),
            ..ImportInstallation::for_test()
        }));
        assert!(message.contains("--profile"), "{message}");
    }

    /// `archive_on("work")` with `sessions` in place of its one session: each a session object
    /// holding the fields a test states over those of the archive's own.
    fn archive_of(sessions: &[serde_json::Value]) -> SessionExport {
        let mut archive = serde_json::to_value(archive_on("work")).expect("serialize");
        let template = archive["sessions"][0].clone();
        archive["sessions"] = sessions
            .iter()
            .map(|stated| {
                let mut session = template.clone();
                for (field, value) in stated.as_object().expect("a session object") {
                    session[field] = value.clone();
                }
                session
            })
            .collect();
        serde_json::from_value(archive).expect("a well-formed archive")
    }

    const ROOT: &str = "11111111-1111-4111-8111-111111111111";

    /// An imported session's level and approvals switch are the importing installation's,
    /// whatever the archive recorded: a root takes its level, a sub-agent keeps the lower of the
    /// two, and every session takes its approvals switch. A directory it would not take from a
    /// client is not recorded.
    #[test]
    fn an_import_takes_its_level_approvals_and_directories_from_the_installation() {
        let archive = || {
            archive_of(&[
                serde_json::json!({
                    "permission": "unrestricted",
                    "approvals": false,
                    "working_directory": "/refused/project",
                    "additional_workspace_directories": ["/taken", "/refused/more"],
                }),
                // A worker whose parent the archive does not carry is a worker still.
                serde_json::json!({
                    "id": "orphan", "parent_id": "elsewhere", "permission": "workspace",
                    "subagent_spec_json": "{\"permission\":\"unrestricted\"}",
                }),
                // A second root, which recorded less than this installation gives.
                serde_json::json!({"id": "low", "permission": "none"}),
                serde_json::json!({
                    "id": "wide", "parent_id": ROOT, "permission": "workspace",
                    "working_directory": "/taken",
                }),
                serde_json::json!({"id": "narrow", "parent_id": ROOT, "permission": "none"}),
            ])
        };
        let installation = |permission| ImportInstallation {
            default: Some("work"),
            permission,
            approvals: true,
            accept_directory: |directory| {
                if directory.starts_with("/refused") {
                    return Err("is refused".to_string());
                }
                Ok(directory.to_path_buf())
            },
            ..ImportInstallation::for_test()
        };
        use crate::permission::Permission;
        let levels = |permission| -> Vec<Permission> {
            plan_import(archive(), installation(permission))
                .expect("plans")
                .records
                .iter()
                .map(|record| record.permission)
                .collect()
        };
        assert_eq!(
            levels(Permission::Read),
            [
                Permission::Read,
                Permission::Read,
                Permission::Read,
                Permission::Read,
                Permission::None
            ],
            "root, orphan, low, wide, narrow"
        );
        assert_eq!(
            levels(Permission::Unrestricted),
            [
                Permission::Unrestricted,
                Permission::Workspace,
                Permission::Unrestricted,
                Permission::Workspace,
                Permission::None
            ],
            "root, orphan, low, wide, narrow"
        );
        let plan = plan_import(archive(), installation(Permission::Read)).expect("plans");
        assert!(
            plan.records.iter().all(|record| record.approvals),
            "every session takes the installation's switch"
        );
        let root = &plan.records[0];
        assert_eq!(root.cwd, None, "a refused directory is not recorded");
        assert_eq!(root.additional_roots, [std::path::PathBuf::from("/taken")]);
        assert_eq!(
            plan.records[3].cwd,
            Some(std::path::PathBuf::from("/taken"))
        );
    }

    #[test]
    fn parents_first_order_orders_parents_before_children() {
        // Given out of order (child, root, middle), each node must land after its parent.
        let nodes = vec![
            ("c".to_string(), Some("b".to_string())),
            ("a".to_string(), None),
            ("b".to_string(), Some("a".to_string())),
        ];
        let order = crate::store::export::parents_first_order(&nodes).expect("order");
        let position = |id: &str| order.iter().position(|&i| nodes[i].0 == id).unwrap();
        assert!(position("a") < position("b"));
        assert!(position("b") < position("c"));
    }

    #[test]
    fn parents_first_order_treats_external_parent_as_root() {
        // A parent absent from the set (e.g. the exported root was itself a sub-agent) is not an
        // error; the node is ordered as a root.
        let nodes = vec![("only".to_string(), Some("outside".to_string()))];
        assert_eq!(
            crate::store::export::parents_first_order(&nodes).expect("order"),
            vec![0]
        );
    }
}
