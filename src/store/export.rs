//! The portable form of a session: the JSON envelope `meka session export` writes and `meka
//! session import` reads, and the planning that turns an archive back into rows.

use super::*;

/// On-wire format version for `meka session export --format json`. Bumped when the envelope shape
/// or the underlying [`crate::conversation::Event`] serialization changes incompatibly; `meka
/// session import` rejects versions it doesn't recognize.
pub(crate) const SESSION_EXPORT_FORMAT_VERSION: u32 = 2;
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
    /// The bytes behind every image block in `sessions`, once each. `#[serde(default)]` so an
    /// archive with no images, or one written by hand, needs no empty list.
    #[serde(default)]
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
    /// Whether calls above the level were submitted for approval. `#[serde(default)]` so an
    /// archive that omits it imports with the switch off.
    #[serde(default)]
    pub(crate) approvals: bool,
    pub(crate) capabilities_json: Option<String>,
    /// Workspace roots beyond `cwd`. `#[serde(default)]` rather than a `format_version` bump:
    /// [`plan_import`] rejects any version it doesn't equal exactly, so bumping would make every
    /// export written before this field unimportable, while an absent field already means the
    /// single-root sessions those exports describe.
    #[serde(default)]
    pub(crate) additional_roots: Vec<std::path::PathBuf>,
    /// A sub-agent's spawn terms. `#[serde(default)]` for the same reason as `additional_roots`:
    /// an archive written before the field existed is still importable, and its sub-agents simply
    /// come back unfollowable rather than unimportable.
    #[serde(default)]
    pub(crate) subagent_spec_json: Option<String>,
    /// The profile the session ran on. `#[serde(default)]` for the same reason as the two fields
    /// above: an archive that names none is still importable, and [`plan_import`] settles the
    /// empty case against the importing installation's default rather than storing a profile
    /// no configuration can name.
    #[serde(default)]
    pub(crate) profile: String,
    pub(crate) stats: crate::stats::SessionStatsSnapshot,
    pub(crate) events: Vec<ExportedEvent>,
    pub(crate) tool_outputs: std::collections::BTreeMap<String, String>,
}
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct ExportedEvent {
    /// RFC 3339 timestamp the event row was persisted; preserved across import.
    pub(crate) at: String,
    pub(crate) event: crate::conversation::Event,
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
            .map(|(at, event)| ExportedEvent { at, event })
            .collect();
        for exported in &events {
            for hash in super::blobs::blob_references(&exported.event) {
                if !referenced.contains(&hash) {
                    referenced.push(hash);
                }
            }
        }
        let tool_outputs = store
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
            stats,
            events,
            tool_outputs,
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
/// Turn a deserialized [`SessionExport`] into the parents-first
/// [`crate::store::ImportSessionRecord`] list to persist, plus the freshly-minted root session
/// ID. Validates the format version, mints a new ID per session, and remaps parent links (a parent
/// pointing outside the exported set collapses to `None`, importing that session as a new root
/// session). Pure and I/O-free so the ID-remap and ordering are unit-testable.
pub(crate) fn plan_import(
    export: SessionExport,
    // What an archive that names no profile adopts. Settled here rather than at the reader because
    // an import is where a session enters *this* installation, and this installation's default is
    // the only thing that can be known about an archive that names none. `None` refuses the import
    // rather than writing a session that cannot run.
    default_profile: Option<&str>,
    // What an archive session that records no level adopts, settled here for the same reason as
    // the profile: a row with no level runs nothing under the scheduler, and this installation's
    // default is what such a session would start at. `None` refuses the archive rather than
    // guessing.
    default_permission: Option<crate::permission::Permission>,
) -> crate::error::Result<ImportPlan> {
    if export.format_version != SESSION_EXPORT_FORMAT_VERSION {
        return Err(crate::error::MekaError::Usage(format!(
            "unsupported session export format_version {} (this build supports \
             {SESSION_EXPORT_FORMAT_VERSION})",
            export.format_version
        )));
    }
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
                "internal error: session id missing from ID remap".to_string(),
            )
        })?;
        let new_parent_id = session
            .parent_id
            .as_ref()
            .and_then(|parent| remap.get(parent).copied());
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
            profile: if session.profile.is_empty() {
                // Refused rather than left blank. A session with no profile cannot run, so
                // importing one is writing a row whose only future is a refusal the user has to
                // work backwards from, and it would put a state into the store that no other
                // door can produce, which every reader would then have to know about.
                let Some(default_profile) = default_profile else {
                    return Err(crate::error::MekaError::Usage(
                        "this archive names no profile and no default profile is configured; import \
                         it with `meka --profile <name> session import`"
                            .to_string(),
                    ));
                };
                default_profile.to_string()
            } else {
                session.profile
            },
            stats: session.stats,
            events: session
                .events
                .into_iter()
                .map(|event| (event.at, event.event))
                .collect(),
            tool_outputs: session.tool_outputs.into_iter().collect(),
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
