//! The portable form of a session: the JSON envelope `meka session export` writes and `meka
//! session import` reads, and the planning that turns an archive back into rows.

use super::*;

/// On-wire format version for `meka session export --format json`. Bumped when the envelope shape
/// or the underlying [`crate::conversation::Event`] serialization changes, a field added or taken
/// away included: the reader takes this version's shape alone, and an older archive the migration
/// module knows is brought forward before it is read. `meka session import` refuses any other
/// version.
pub(crate) const SESSION_EXPORT_FORMAT_VERSION: u32 = 7;
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
    // Only the migration module knows what an older meka wrote, so this door hands the document
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
/// The profile facts [`plan_import`] settles every imported session against.
///
/// A session runs on the profile its row names, and an import is a door that writes such a row, so
/// it answers to the same rule as every other: the name is one this installation configures, or
/// the import is refused. `--profile` is the explicit act that moves a session onto another.
pub(crate) struct ImportProfiles<'a> {
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
    /// A string in the archive, a path in the struct, like `additional_roots` below.
    pub(crate) cwd: Option<std::path::PathBuf>,
    /// The level the session ran at, as its one spelling; an archive naming anything else is
    /// refused at parse rather than written to a row no reader could resolve.
    pub(crate) permission: Option<crate::permission::Permission>,
    /// Whether calls above the level were submitted for approval.
    pub(crate) approvals: bool,
    pub(crate) capabilities_json: Option<String>,
    /// Workspace roots beyond `cwd`; empty for a single-root session.
    pub(crate) additional_roots: Vec<std::path::PathBuf>,
    /// A sub-agent's spawn terms; `null` for a session nothing spawned, which an absent key also
    /// reads as, by serde's rule for an `Option`.
    pub(crate) subagent_spec_json: Option<String>,
    /// The profile the session ran on; empty when the archive names none, which [`plan_import`]
    /// settles against the importing installation's default rather than storing a profile no
    /// configuration can name.
    pub(crate) profile: String,
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
    pub(crate) usage: crate::store::turns::TurnUsage,
}
impl From<crate::store::turns::TurnRecord> for ExportedTurn {
    fn from(turn: crate::store::turns::TurnRecord) -> Self {
        let (error_type, detail) = match turn.error {
            Some(error) => (Some(error.kind.name().to_string()), Some(error.detail)),
            None => (None, None),
        };
        Self {
            id: turn.id.to_string(),
            source: turn.source,
            started_at: turn.started_at,
            ended_at: turn.ended_at,
            status: turn.status.map(|status| status.name().to_string()),
            stop_reason: turn.stop_reason,
            error_type,
            detail,
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
        let events: Vec<ExportedEvent> = store
            .load_events_with_timestamps(meta.id)
            .await?
            .into_iter()
            .map(|(stamp, event)| ExportedEvent {
                at: stamp.created_at,
                event,
                turn_id: stamp.turn_id.map(|id| id.to_string()),
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
            cwd: meta.cwd,
            permission: meta.permission,
            approvals: meta.approvals,
            capabilities_json: meta.capabilities_json,
            additional_roots: meta.additional_roots,
            subagent_spec_json: meta.subagent_spec_json,
            profile: meta.profile,
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
/// The refusal for an archive stamp that is not RFC 3339, the one shape every row's stamp has.
fn require_rfc3339(session: &str, what: &str, stamp: &str) -> crate::error::Result<()> {
    chrono::DateTime::parse_from_rfc3339(stamp)
        .map(|_| ())
        .map_err(|error| {
            crate::error::MekaError::Usage(format!(
                "session '{session}' carries a `{what}` that is not RFC 3339 ('{stamp}'): {error}"
            ))
        })
}

/// Turn a deserialized [`SessionExport`] into the parents-first
/// [`crate::store::ImportSessionRecord`] list to persist, plus the freshly-minted root session
/// ID. Validates the format version, mints a new ID per session, and remaps parent links (a parent
/// pointing outside the exported set collapses to `None`, importing that session as a new root
/// session). Pure and I/O-free so the ID-remap and ordering are unit-testable.
pub(crate) fn plan_import(
    export: SessionExport,
    // How each session's profile is settled, here rather than at the reader because an import is
    // where a session enters *this* installation.
    profiles: ImportProfiles<'_>,
    // What an archive session that records no level adopts, settled here for the same reason as
    // the profile: a row with no level runs nothing under the scheduler, and this installation's
    // default is what such a session would start at. `None` refuses the archive rather than
    // guessing.
    default_permission: Option<crate::permission::Permission>,
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

    // A stamp meka did not write is refused ahead of any row: every listing reads them back, and
    // a column cut from one has to be cut from a shape it knows.
    for session in &export.sessions {
        for (what, stamp) in [
            ("created_at", Some(&session.created_at)),
            ("updated_at", Some(&session.updated_at)),
            ("pinned_at", session.pinned_at.as_ref()),
        ] {
            if let Some(stamp) = stamp {
                require_rfc3339(&session.id, what, stamp)?;
            }
        }
        for event in &session.events {
            require_rfc3339(&session.id, "an event's `at`", &event.at)?;
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
    if let (Some(name), Some(configured)) = (profiles.selected, profiles.configured) {
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
        let session = slots[index].take().ok_or_else(|| {
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
            require_rfc3339(&session.id, "a turn's started_at", &turn.started_at)?;
            if let Some(ended_at) = &turn.ended_at {
                require_rfc3339(&session.id, "a turn's ended_at", ended_at)?;
            }
            if let Some(status) = &turn.status
                && status.parse::<crate::store::turns::TurnStatus>().is_err()
            {
                return Err(crate::error::MekaError::Usage(format!(
                    "session {} in the archive records a turn status meka does not know: '{status}'",
                    session.id
                )));
            }
            if let Some(error_type) = &turn.error_type
                && crate::error::ErrorKind::from_name(error_type).is_none()
            {
                return Err(crate::error::MekaError::Usage(format!(
                    "session {} in the archive records an error type meka does not know: \
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
        let mut events = Vec::with_capacity(session.events.len());
        for event in session.events {
            let turn = match event.turn_id {
                Some(old) => Some(turn_ids.get(&old).copied().ok_or_else(|| {
                    crate::error::MekaError::Usage(format!(
                        "session {} in the archive has an event naming turn '{old}', which the \
                         archive does not carry",
                        session.id
                    ))
                })?),
                None => None,
            };
            events.push((event.at, event.event, turn));
        }
        let permission = match (session.permission, default_permission) {
            (Some(level), _) => level,
            (None, Some(level)) => level,
            (None, None) => {
                return Err(crate::error::MekaError::Usage(format!(
                    "session {} in the archive records no permission level, and no default could \
                     be read from `config.toml`",
                    session.id
                )));
            }
        };
        let profile = match (
            profiles.selected,
            session.profile.is_empty(),
            profiles.default,
        ) {
            (Some(name), ..) => name.to_string(),
            // Refused rather than written: a row naming a profile this installation lacks is a
            // session every resume refuses by name, and a state no other door can produce.
            (None, false, _) => {
                if let Some(configured) = profiles.configured {
                    crate::config::require_profile(&session.profile, configured).map_err(
                        |refusal| {
                            crate::error::MekaError::Usage(format!(
                                "{refusal}; the archive names it, so import it with `meka --profile \
                                 <name> session import`"
                            ))
                        },
                    )?;
                }
                session.profile
            }
            (None, true, Some(default)) => default.to_string(),
            // Refused rather than left blank, for the same reason: a session with no profile
            // cannot run, and a blank is a state no other door can produce.
            (None, true, None) => {
                return Err(crate::error::MekaError::Usage(
                    "this archive names no profile and no default profile is set; import it with \
                     `meka --profile <name> session import`"
                        .to_string(),
                ));
            }
        };
        records.push(crate::store::ImportSessionRecord {
            new_id,
            new_parent_id,
            created_at: session.created_at,
            cwd: session.cwd,
            permission,
            approvals: session.approvals,
            capabilities_json: session.capabilities_json,
            additional_roots: session.additional_roots,
            subagent_spec_json: session.subagent_spec_json,
            profile,
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
        {
            children[parent_index].push(index);
            indegree[index] += 1;
        }
    }
    let mut queue: VecDeque<usize> = (0..nodes.len()).filter(|&i| indegree[i] == 0).collect();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(node) = queue.pop_front() {
        order.push(node);
        for &child in &children[node] {
            indegree[child] -= 1;
            if indegree[child] == 0 {
                queue.push_back(child);
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
        for field in ["approvals", "additional_roots", "profile", "turns"] {
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
                "cwd": null,
                "permission": "read",
                "approvals": false,
                "capabilities_json": null,
                "additional_roots": [],
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
            Err(crate::error::MekaError::Usage(message)) => message,
            Err(other) => panic!("the refusal must be the caller's to act on, got {other}"),
        }
    }

    /// An archive from another installation names the profile it ran on there. Written as it
    /// stands, the row is a session every resume refuses by name; refused here, the remedy is the
    /// one flag that moves a session.
    #[test]
    fn an_archive_naming_a_profile_this_installation_lacks_is_refused() {
        let profiles = configured(&["other"]);
        let message = refusal(plan_import(
            archive_on("work"),
            ImportProfiles {
                selected: None,
                default: Some("other"),
                configured: Some(&profiles),
            },
            None,
        ));
        assert!(
            message.contains("'work'") && message.contains("--profile <name> session import"),
            "{message}"
        );
    }

    /// A stamp that is not RFC 3339 is refused ahead of any row: meka's own are, every listing
    /// reads them back, and `scratchpad_list` cuts one for its column.
    #[test]
    fn an_archive_stamp_that_is_not_rfc_3339_is_refused() {
        let profiles = configured(&["work"]);
        let mut archive = archive_on("work");
        archive.sessions[0].created_at = "yesterday".to_string();
        let message = refusal(plan_import(
            archive,
            ImportProfiles {
                selected: None,
                default: Some("work"),
                configured: Some(&profiles),
            },
            Some(crate::permission::Permission::Read),
        ));
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
            plan_import(
                archive,
                ImportProfiles {
                    selected: None,
                    default: Some("work"),
                    configured: Some(&profiles),
                },
                Some(crate::permission::Permission::Read),
            )
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
        let plan = plan_import(
            archive_on("work"),
            ImportProfiles {
                selected: Some("other"),
                default: None,
                configured: Some(&profiles),
            },
            None,
        )
        .expect("the flag names a configured profile");
        assert!(
            plan.records.iter().all(|record| record.profile == "other"),
            "every session moved onto the selected profile"
        );

        let message = refusal(plan_import(
            archive_on("work"),
            ImportProfiles {
                selected: Some("ghost"),
                default: None,
                configured: Some(&profiles),
            },
            None,
        ));
        assert!(message.contains("'ghost'"), "{message}");
    }

    /// An archive recording no profile adopts the installation's default, as before, and is
    /// refused when there is none.
    #[test]
    fn an_archive_naming_no_profile_takes_the_default_or_is_refused() {
        let profiles = configured(&["other"]);
        let plan = plan_import(
            archive_on(""),
            ImportProfiles {
                selected: None,
                default: Some("other"),
                configured: Some(&profiles),
            },
            None,
        )
        .expect("the default supplies one");
        assert_eq!(plan.records[0].profile, "other");

        let message = refusal(plan_import(
            archive_on(""),
            ImportProfiles {
                selected: None,
                default: None,
                configured: Some(&profiles),
            },
            None,
        ));
        assert!(message.contains("--profile"), "{message}");
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
