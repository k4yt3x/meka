//! `meka session`: the stored-conversation CLI.
//!
//! Nothing here is reachable from a turn: these are the commands a human runs against conversations
//! that already exist, so they take a [`Store`] and nothing else of the agent.
//!
//! Export and import are the substantial part. The JSON form is versioned
//! ([`crate::store::export::SESSION_EXPORT_FORMAT_VERSION`]) and carries sub-agent descendants
//! alongside their parent; the Markdown form is a rendering for people, and is not re-importable.

use crate::{cli, conversation, store::Store, text::Precision, view::SessionView};

/// What [`list_sessions`] aims to fit in.
///
/// The id takes whatever distinguishes it from the others on screen, the timestamp its fixed width
/// plus two, the profile what its longest name needs up to [`PROFILE_TRUNCATE`], and the preview
/// the rest. A full UUID here cost 36 of the 120 to repeat what eight characters usually say, and
/// every command that takes a session id accepts the prefix this prints; [`show_session`] has the
/// whole one.
const TABLE_WIDTH: usize = 120;

/// Ceiling on the rendered profile column.
///
/// A profile name is a config key the user chose, so nothing bounds it but this. Wide enough for
/// the descriptive names people actually use (`openrouter-anthropic-messages` is 29), since a
/// name cut short enough to stop distinguishing two profiles is worse than a shorter preview.
const PROFILE_TRUNCATE: usize = 32;

/// Floor on the preview, so a pathological profile name cannot squeeze it to nothing.
const PREVIEW_MINIMUM: usize = 24;

/// `meka session fork <id>`: copy a session's conversation into a new one and print the new id.
///
/// Output split mirrors [`import_session`]: the bare id on stdout so `id=$(meka session fork …)`
/// works, the resume hint on stderr.
pub(crate) async fn fork_session_command(
    store: &Store,
    session_id: uuid::Uuid,
) -> anyhow::Result<()> {
    // Nothing in this process holds the source, so the store probes it; see
    // `Store::fork_session_locked` for why a source being written cannot be copied.
    let (forked, copy_lock) = store
        .fork_session_locked(
            session_id,
            crate::store::ForkOverrides::default(),
            crate::store::SourceLock::Probe,
        )
        .await
        .map_err(cannot_copy_while_written)?
        .ok_or(crate::error::MekaError::SessionNotFound(session_id))?;
    // The copy is handed back as an id, not held: `meka -r` on it takes its own lock.
    drop(copy_lock);

    tracing::info!(
        "forked session {session_id} into {forked}",
        forked = forked.id
    );
    // A fork of a sub-agent is a sibling under the same parent, so the copy is a worker too and
    // `meka -r` refuses it; the hint has to say that continuing it is the parent's job. A read
    // failure decides only which hint is printed, so it must not fail a fork that has already
    // landed, but it is logged, because the fallback hint is the wrong advice for a worker's copy.
    let spawned = match store.spawn_terms(forked.id).await {
        Ok(terms) => terms,
        Err(error) => {
            tracing::debug!(
                "failed to read the fork's row to choose a hint, assuming a root: {error}"
            );
            None
        }
    };
    match spawned.map(|terms| terms.parent) {
        Some(Some(parent)) => crate::streams::write_stderr_line(format!(
            "Forked session; it is a sub-agent of {parent}, so continue it with `agent_followup` \
             from there."
        )),
        // A copy of a worker whose parent is not in this store: still a worker, still not
        // resumable, and with no id to name.
        Some(None) => crate::streams::write_stderr_line(
            "Forked session; its parent is not in this store, so it cannot be resumed.",
        ),
        None => crate::streams::write_stderr_line(format!(
            "Forked session; resume it with `meka -r {}`",
            forked.id
        )),
    }
    crate::render::write_stdout_line(forked.id)?;
    Ok(())
}

pub(crate) async fn run_session_subcommand(
    store: &Store,
    action: &cli::SessionAction,
    // How `import` settles each archived session's profile: the `--profile` flag as passed, the
    // installation's default and its configured profiles. The flag is carried unresolved so that
    // `meka --profile work session import` moves the archive per run, while the migration context
    // the caller also builds ignores it, because that stamps rows once and irreversibly. Unused by
    // every other action here.
    profiles: crate::store::export::ImportProfiles<'_>,
    // The level such an archive's sessions adopt when they record none, for the same reason.
    default_permission: Option<crate::permission::Permission>,
) -> anyhow::Result<()> {
    match action {
        cli::SessionAction::List {
            limit,
            include_children,
            format,
        } => list_sessions(store, *limit, *include_children, *format).await,
        cli::SessionAction::Export {
            session_id,
            output,
            format,
        } => {
            let session_id = store.resolve_session_id(session_id).await?;
            // Held for the same reason `fork` holds its source: an export is a snapshot, and a
            // snapshot of a conversation mid-turn carries an unanswered user message that `meka
            // session import` then restores as an unusable session. See `hold_still_for_a_copy`.
            let _source = hold_still_for_a_copy(store, session_id)?;
            // The written path is only interesting to the REPL; out here the shell (and the `-o`
            // the user typed) already knows where it went.
            export_session(store, session_id, output.as_deref(), *format).await?;
            Ok(())
        }
        cli::SessionAction::Delete {
            session_ids,
            all,
            older_than_days,
        } => {
            // One id that cannot be resolved does not cancel the rest, matching the policy
            // `delete_sessions` states for a session that is in use: every id the user named is a
            // separate request.
            let mut resolved = Vec::with_capacity(session_ids.len());
            let mut unresolved = Vec::new();
            for session_id in session_ids {
                match store.resolve_session_id(session_id).await {
                    Ok(id) => resolved.push(id),
                    Err(error) => unresolved.push(error.to_string()),
                }
            }
            for problem in &unresolved {
                crate::streams::write_stderr_line(problem);
            }
            // A name that resolved to nothing is a refusal like any other, and has to reach the
            // exit code the same way: `delete_sessions` never learns these ids, so leaving it to
            // report them exited 0 on `meka session delete <good> <typo>` and a script could not
            // tell the typo from success. Clap cannot catch this, because a prefix is not a `Uuid`,
            // so the check lives here.
            //
            // Skipped only when every id the user named failed to resolve and no sweep was asked
            // for, because `delete_sessions` would then read the empty list as "no ids given" and
            // answer `specify session ids, ...`, which contradicts the command line. With no
            // ids at all that message is exactly right, so it still goes through.
            if !unresolved.is_empty() && resolved.is_empty() && !*all && older_than_days.is_none() {
                anyhow::bail!(
                    "{} of the session(s) named did not resolve; see the errors above",
                    unresolved.len()
                );
            }
            let outcome = delete_sessions(store, &resolved, *all, *older_than_days).await;
            if unresolved.is_empty() {
                return outcome;
            }
            outcome?;
            anyhow::bail!(
                "{} of {} session(s) named did not resolve; see the errors above",
                unresolved.len(),
                unresolved.len() + resolved.len()
            )
        }
        cli::SessionAction::Import { input } => {
            import_session(store, input, profiles, default_permission).await
        }
        cli::SessionAction::Show { session_id, format } => {
            let session_id = store.resolve_session_id(session_id).await?;
            show_session(store, session_id, *format).await
        }
        cli::SessionAction::Fork { session_id } => {
            let session_id = store.resolve_session_id(session_id).await?;
            fork_session_command(store, session_id).await
        }
        cli::SessionAction::Rewind { session_id, turns } => {
            // The turn count is checked before the id is resolved, so `-n 0` blames the argument
            // rather than reporting a session it never needed to find. Pinned by
            // `session_rewind_rejects_zero_turns_without_describing_the_conversation`.
            if *turns == 0 {
                anyhow::bail!("`-n` must be 1 or more");
            }
            let session_id = store.resolve_session_id(session_id).await?;
            rewind_session_command(store, session_id, *turns).await
        }
    }
}

/// `meka session show <id>`: one session, with the id in full.
///
/// The listing shortens an id to whatever distinguishes it, which is only safe because this prints
/// the whole thing. It also carries what the table has no room for: the working directory and the
/// permission the row records. Not the whole first message, which `session_info` has already cut to
/// its title of 80 characters before this sees it; `meka session export` is the surface that has
/// all of it.
async fn show_session(
    store: &Store,
    session_id: uuid::Uuid,
    format: cli::OutputFormat,
) -> anyhow::Result<()> {
    let session = store
        .session_info(session_id)
        .await?
        .ok_or(crate::error::MekaError::SessionNotFound(session_id))?;
    if format == cli::OutputFormat::Json {
        cli::write_json(&SessionView::from(&session))?;
        return Ok(());
    }

    let fields = [
        ("id", session.id.to_string()),
        (
            "created",
            format_stored_timestamp(&session.created_at, Precision::Seconds),
        ),
        (
            "updated",
            format_stored_timestamp(&session.updated_at, Precision::Seconds),
        ),
        // Sanitized but not capped: a profile name is a config key the user chose, and this is the
        // surface that does not truncate.
        (
            "profile",
            crate::text::sanitize_to_line(&session.profile, usize::MAX),
        ),
        ("cwd", match &session.cwd {
            Some(cwd) => crate::text::sanitize_to_line(&cwd.to_string_lossy(), usize::MAX),
            None => "-".to_string(),
        }),
        // A level the store could parse, or nothing: the row is read into `Permission` where it is
        // read, so an archive-chosen value never reaches this cell as text.
        (
            "permission",
            session.permission.map_or_else(
                || "from process config".to_string(),
                |level| level.to_string(),
            ),
        ),
        (
            "approvals",
            if session.approvals { "on" } else { "off" }.to_string(),
        ),
        (
            "title",
            crate::text::sanitize_to_line(&session.title, usize::MAX),
        ),
    ];
    crate::render::write_stdout(crate::text::format_fields(&fields))?;
    Ok(())
}

/// `meka session rewind`: drop the last `turns` turns from a session that isn't currently open.
///
/// The escape hatch for content `Agent::run_turn` can't repair itself, namely anything the provider
/// refuses that was committed before the current turn. Appends an `Event::Repair` with an empty
/// replacement, so nothing is deleted and `meka session export` still shows the dropped turns.
pub(crate) async fn rewind_session_command(
    store: &Store,
    session_id: uuid::Uuid,
    turns: usize,
) -> anyhow::Result<()> {
    // Rejected before anything else: `Conversation::rewind(0)` returns `None` unconditionally, so
    // the error below would otherwise say the session has "fewer than 0 turn(s)".
    if turns == 0 {
        anyhow::bail!("`-n` must be 1 or more");
    }
    // Held for the whole read-modify-write. A REPL, `meka serve`, or `meka acp` holding this
    // session has its own in-memory conversation that would overwrite the rewind on its next turn.
    if !store.session_exists(session_id).await? {
        return Err(crate::error::MekaError::SessionNotFound(session_id).into());
    }
    let _lock = store.lock_session(session_id)?;

    let events = store.load_events(session_id).await?;
    let mut conversation = conversation::Conversation::from_events(events);

    let Some(event) = conversation.rewind(turns) else {
        anyhow::bail!("nothing to rewind: session {session_id} has fewer than {turns} turn(s)");
    };
    store.save_event(session_id, &event).await?;

    tracing::info!("rewound {turns} turn(s) from session {session_id}");
    crate::streams::write_stderr_line(format!(
        "Rewound {} turn(s); {} message(s) remain, and `meka session export` still shows the \
         dropped ones.",
        turns,
        conversation.len(),
    ));
    Ok(())
}

pub(crate) async fn list_sessions(
    store: &Store,
    limit: u32,
    include_children: bool,
    format: cli::OutputFormat,
) -> anyhow::Result<()> {
    let (sessions, _next_cursor) = store
        .list_sessions(limit, include_children, None, None)
        .await?;

    if format == cli::OutputFormat::Json {
        let views: Vec<SessionView> = sessions.iter().map(SessionView::from).collect();
        cli::write_json_listing("sessions", &views)?;
        return Ok(());
    }
    if sessions.is_empty() {
        crate::streams::write_stderr_line("No sessions.");
        return Ok(());
    }

    // The universe a printed prefix will be resolved against, which is every session rather than
    // the ones this listing chose to show.
    let resolvable = store.all_session_ids().await?;
    let headers: &[&str] = &["ID", "Updated", "Profile", "Title"];
    crate::render::write_stdout(crate::text::format_columns(
        headers,
        &session_rows(&sessions, &resolvable),
    ))?;

    Ok(())
}

/// One row per session, separated from printing so the sanitizing can be asserted.
///
/// Both authored cells go through the same helper, which drops `\n` and caps in terminal columns.
fn session_rows(
    sessions: &[crate::store::SessionSummary],
    resolvable: &[String],
) -> Vec<Vec<String>> {
    // A profile name is a config key the user chose, so nothing bounds it but `PROFILE_TRUNCATE`.
    // Sized to the longest one actually present rather than to that ceiling: `format_columns` pads
    // to the widest cell either way, so reserving the ceiling would spend width on nobody.
    let profiles: Vec<String> = sessions
        .iter()
        .map(|session| crate::text::sanitize_to_line(&session.profile, PROFILE_TRUNCATE))
        .collect();
    let profile_width = profiles
        .iter()
        .map(|profile| unicode_width::UnicodeWidthStr::width(profile.as_str()))
        .chain(std::iter::once("Profile".len()))
        .max()
        .unwrap_or(PROFILE_TRUNCATE);
    let ids: Vec<String> = sessions
        .iter()
        .map(|session| session.id.to_string())
        .collect();
    let id_width = crate::text::unique_prefix_len_within(
        ids.iter().map(String::as_str),
        resolvable.iter().map(String::as_str),
    )
    .max("ID".len());
    let preview_width = TABLE_WIDTH
        .saturating_sub(id_width + 2)
        .saturating_sub(Precision::Minutes.width() + 2)
        .saturating_sub(profile_width + 2)
        .max(PREVIEW_MINIMUM);

    sessions
        .iter()
        .zip(profiles)
        .zip(&ids)
        .map(|((session, profile), id)| {
            vec![
                id.get(..id_width).unwrap_or(id).to_string(),
                format_stored_timestamp(&session.updated_at, Precision::Minutes),
                profile,
                // A first message's words: a model composed it, or an API caller sent it through
                // `POST /v1/sessions`.
                crate::text::sanitize_to_line(&session.title, preview_width),
            ]
        })
        .collect()
}

/// Write a session out as Markdown or as the JSON envelope, to `output` (`-` for stdout), or to
/// `session-<id>.<ext>` in the working directory when no path was given.
pub(crate) async fn export_session(
    store: &Store,
    session_id: uuid::Uuid,
    output: Option<&str>,
    format: cli::SessionExportFormat,
) -> anyhow::Result<Option<std::path::PathBuf>> {
    if !store.session_exists(session_id).await? {
        return Err(crate::error::MekaError::SessionNotFound(session_id).into());
    }

    let (body, default_ext) = match format {
        cli::SessionExportFormat::Markdown => {
            // Export the full event log so pre-compaction turns are included. Compaction only hides
            // older turns from the model (it appends a boundary, never deletes), so the export
            // walks the raw log and renders every turn plus a marker at each compaction point.
            let events = store.load_events(session_id).await?;
            let tool_outputs: std::collections::HashMap<String, String> = store
                .load_all_scratchpad_entries(session_id)
                .await?
                .into_iter()
                .collect();
            (
                conversation::format_session_as_markdown(session_id, &events, &tool_outputs),
                "md",
            )
        }
        cli::SessionExportFormat::Json => {
            let export = crate::store::export::build_session_export(store, session_id).await?;
            (serde_json::to_string_pretty(&export)?, "json")
        }
    };

    match output {
        Some("-") => {
            crate::render::write_stdout(&body)?;
            Ok(None)
        }
        Some(path) => {
            write_export(std::path::Path::new(path), &body)?;
            tracing::info!("exported session to {path}");
            Ok(Some(std::path::PathBuf::from(path)))
        }
        None => {
            let path = std::path::PathBuf::from(format!("session-{session_id}.{default_ext}"));
            write_export(&path, &body)?;
            tracing::info!("exported session to {path}", path = path.display());
            Ok(Some(path))
        }
    }
}

/// Write an export to the destination the user named.
///
/// A regular file, or a path with nothing at it yet, is written atomically and owner-only, as
/// `memory export` writes: a transcript carries tool output, file contents and whatever was pasted,
/// and `std::fs::write` left it at the umask's mode. Anything else, a FIFO or a device, is what the
/// user chose to stream into and is written as given; replacing it with a file would not be the
/// export they asked for, and a reader that hangs up on it has to fail the command.
fn write_export(path: &std::path::Path, body: &str) -> std::io::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.is_file() && !metadata.file_type().is_symlink() => {
            std::fs::write(path, body)
        }
        _ => crate::fs::write_file_atomic(path, body),
    }
}

/// Recreate a session tree from a JSON export, read from `input` (`-` for stdin), under fresh ids.
/// The new root id is the command's output; everything else said about it goes to stderr.
pub(crate) async fn import_session(
    store: &Store,
    input: &str,
    profiles: crate::store::export::ImportProfiles<'_>,
    default_permission: Option<crate::permission::Permission>,
) -> anyhow::Result<()> {
    // A `config.toml` that did not parse leaves nothing to check an archive's profile name
    // against, so the import is refused ahead of the first write rather than writing that name
    // unchecked. Here rather than in `main.rs`, whose unreadable-config arm serves every
    // subcommand and has to stay open for the ones that repair the file.
    crate::config::ResolvedConfig::resolve(crate::config::CliOverrides::default())
        .require_readable_config()?;
    let raw = if input == "-" {
        let mut buffer = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buffer)
            .map_err(|error| anyhow::anyhow!("failed to read stdin: {error}"))?;
        buffer
    } else {
        std::fs::read_to_string(input)
            .map_err(|error| anyhow::anyhow!("failed to read '{input}': {error}"))?
    };
    let export: crate::store::export::SessionExport = serde_json::from_str(&raw)
        .map_err(|error| anyhow::anyhow!("invalid session export JSON: {error}"))?;

    // Version mismatch, an empty archive and a profile nothing here configures all surface here,
    // worded for the person who ran the command: the archive is theirs, so a refusal is about
    // their input. An image blob nobody holds is refused the same way by `import_sessions`, ahead
    // of its write.
    let crate::store::export::ImportPlan {
        records,
        blobs,
        root_new_id,
    } = crate::store::export::plan_import(export, profiles, default_permission)?;
    let count = records.len();
    store.import_sessions(records, blobs).await?;

    tracing::info!("imported {count} session(s) as root {root_new_id}");
    crate::streams::write_stderr_line(format!(
        "Imported {count} session(s); resume the root with `meka -r {root_new_id}`"
    ));
    // The one line a script reads: the id it can resume.
    crate::render::write_stdout_line(root_new_id)?;
    Ok(())
}

/// Delete the named sessions, every session, or those not updated for `older_than_days`.
pub(crate) async fn delete_sessions(
    store: &Store,
    session_ids: &[uuid::Uuid],
    all: bool,
    older_than_days: Option<u64>,
) -> anyhow::Result<()> {
    if all {
        let sweep = store.delete_all_sessions().await?;
        tracing::info!("deleted {deleted} session(s)", deleted = sweep.deleted);
        report_sessions_left_open(sweep);
        return Ok(());
    }

    // The manual counterpart to `[session].retention`, since nothing prunes by size on its own.
    // Reports the count through `info!` like the `--all` and by-id branches below: the user ran
    // this to delete, not to obtain a number, and the exit code already carries success.
    if let Some(days) = older_than_days {
        // Zero would sweep everything, which is `--all` by another name and far too easy to type
        // by accident when you meant "today's".
        if days == 0 {
            anyhow::bail!("`--older-than-days 0` would delete every session; use `--all` for that");
        }
        let sweep = store
            .delete_expired_sessions(std::time::Duration::from_secs(days.saturating_mul(86_400)))
            .await?;
        tracing::info!(
            "deleted {deleted} session(s) not updated in {days} days",
            deleted = sweep.deleted
        );
        report_sessions_left_open(sweep);
        return Ok(());
    }

    if session_ids.is_empty() {
        anyhow::bail!("specify session ids, `--older-than-days <DAYS>` or `--all`");
    }

    let mut deleted = 0u64;
    // Reported at the end rather than returned at the first refusal. Every id the user named is a
    // separate request, and one of them being in use is no reason to leave the rest of the list
    // untried, nor to swallow the count of what did go, which returning early also did.
    let mut refused = Vec::new();
    for session_id in session_ids {
        // The refusing door, not the plain one: this is a session this process has never had open,
        // and deleting one another meka is mid-conversation on cascades its messages away
        // underneath a live agent, which then fails every remaining turn on a foreign-key
        // violation.
        match store.delete_session_unless_attached(*session_id).await {
            Ok(true) => deleted += 1,
            // A refusal like any other: the user named this id and it did not go away, which a
            // script must be able to tell from success. Reachable without a race by naming one
            // session twice (a prefix and its full id), where the second pass finds it gone.
            Ok(false) => {
                crate::streams::write_stderr_line(
                    crate::error::MekaError::SessionNotFound(*session_id).to_string(),
                );
                refused.push(*session_id);
            }
            Err(error) => {
                crate::streams::write_stderr_line(format!("Cannot delete {session_id}: {error}"));
                refused.push(*session_id);
            }
        }
    }

    tracing::info!("deleted {deleted} session(s)");
    // A non-zero exit, because the user named these and a silent skip is indistinguishable from
    // success. The per-id reasons are already on stderr; this is what a script reads.
    if !refused.is_empty() {
        // Does not name a cause. Every `Err` lands here, and `delete_session_unless_attached`
        // returns a database error as readily as a lock refusal, so claiming "another meka has
        // them open" would report a full disk as a busy session. The per-id lines above carry the
        // real reason; this is the summary a script reads.
        anyhow::bail!(
            "{} of {} session(s) were not deleted; see the errors above",
            refused.len(),
            session_ids.len()
        );
    }
    Ok(())
}

/// Say what a sweep spared, because a count of deletions alone reads as "everything matched went".
///
/// `warn!` rather than `info!`: the user asked for these to be gone and some of them are not, which
/// is a fallback they should see at the default level rather than a lifecycle signpost.
pub(crate) fn report_sessions_left_open(sweep: crate::store::SessionSweep) {
    if sweep.attached_elsewhere > 0 {
        tracing::warn!(
            "left {attached} session(s) alone: another meka process has them open; close it and \
             run this again",
            attached = sweep.attached_elsewhere
        );
    }
}

/// Hold a session still while its conversation is copied out of it.
///
/// Both CLI doors that copy a conversation read a run of rows that a concurrent turn may be halfway
/// through writing. `Agent::run_turn` persists the user message *eagerly*, before the provider has
/// answered, so a copy taken mid-turn ends on an unanswered user message: the fork reads `user,
/// user, assistant` from its first resumed turn onward, and an exported snapshot reproduces that
/// shape through `meka session import`. It happens every time a session that is thinking is forked.
///
/// `meka session rewind` takes this lock too. Fork probes inside `Store::fork_session_locked`,
/// where every fork door reaches it; export is the door left here.
///
/// Deliberately *not* pushed down into [`export_session`]: the REPL's `/export` acts on the session
/// the REPL is already holding, and `flock` is per open file description, so asking again inside
/// the same process would refuse a session that is legitimately ours.
pub(crate) fn hold_still_for_a_copy(
    store: &Store,
    session_id: uuid::Uuid,
) -> anyhow::Result<crate::fs::FileLock> {
    store
        .lock_session(session_id)
        .map_err(cannot_copy_while_written)
}

/// The remedy for a copy refused because another process holds the source, for the two CLI doors
/// that copy a conversation; any other error passes through as itself.
fn cannot_copy_while_written(error: crate::error::MekaError) -> anyhow::Error {
    match error {
        crate::error::MekaError::SessionLocked(_) => anyhow::anyhow!(
            "{error}: a conversation cannot be copied while it is being written; close the meka \
             that has it open and try again"
        ),
        other => other.into(),
    }
}

/// A stored timestamp as [`crate::text::format_timestamp`] renders one, or the raw value when it
/// will not parse.
///
/// The fallback is sanitized and cut to the width a real timestamp occupies, because it is the one
/// branch that prints a column's bytes rather than a rendering of them. `import_sessions` copies
/// `created_at` out of an archive verbatim, so an unparseable value is attacker-chosen on a
/// supported path: unsanitized it can erase the line above and print a second `id:` line, on the
/// command whose whole purpose is showing that id. Every other cell on both surfaces is already
/// sanitized; this is the door they have in common.
pub(crate) fn format_stored_timestamp(rfc3339: &str, precision: Precision) -> String {
    chrono::DateTime::parse_from_rfc3339(rfc3339)
        .map(|datetime| {
            crate::text::format_timestamp(datetime.with_timezone(&chrono::Utc), precision)
        })
        .unwrap_or_else(|_| crate::text::sanitize_to_line(rfc3339, precision.width()))
}
#[cfg(test)]
mod tests {
    use super::*;

    /// A timestamp that will not parse is still a cell, not a license to draw.
    ///
    /// `import_sessions` copies `created_at` out of an archive verbatim, so this branch prints
    /// bytes an attacker chose. Unsanitized it erases the line above it and writes a second `id:`
    /// line, which is a forged answer from the command whose whole job is printing that id.
    #[test]
    fn an_unparseable_timestamp_cannot_draw_over_the_line_above_it() {
        let forged = "\u{1b}[7mCREATED\u{1b}[0m\rid:          00000000-0000-0000-0000-000000000000";
        let rendered = format_stored_timestamp(forged, Precision::Seconds);
        assert!(
            !rendered.contains('\u{1b}') && !rendered.contains('\r'),
            "no escape or carriage return may survive: {rendered:?}"
        );
        assert!(
            crate::text::display_width(&rendered) <= Precision::Seconds.width(),
            "and it stays inside the column a timestamp occupies: {rendered:?}"
        );
        let instant = chrono::DateTime::parse_from_rfc3339("2026-09-02T22:57:28Z")
            .expect("a valid instant")
            .with_timezone(&chrono::Utc);
        assert_eq!(
            format_stored_timestamp("2026-09-02T22:57:28Z", Precision::Seconds),
            crate::text::format_timestamp(instant, Precision::Seconds),
            "a real timestamp is rendered, not sanitized"
        );
    }

    /// The rows as their own resolution universe, for tests about layout rather than about the
    /// store the ids came from.
    fn ids_of(sessions: &[crate::store::SessionSummary]) -> Vec<String> {
        sessions
            .iter()
            .map(|session| session.id.to_string())
            .collect()
    }
    /// Neither authored cell in `meka session list` can forge a row or carry an escape.
    ///
    /// The preview is the first line of a first message: a model composed it, or an API caller sent
    /// it through `POST /v1/sessions`. Sent to the terminal raw and uncapped while the provider
    /// beside it is sanitized, it is the shape of gap that a test asserting on the helper rather
    /// than on the row would not find, since the helper is not the part that breaks.
    ///
    /// Capped in terminal columns rather than characters, so a preview of CJK cannot be twice as
    /// wide as the cap applied to it.
    #[tokio::test]
    async fn no_session_row_can_be_forged_by_its_own_preview() {
        let manager = Store::for_test().await;
        let id = manager
            .create_session(None, "prof\u{1b}[31m\nffffffff  x  y  z".to_string())
            .await
            .expect("create");
        manager
            .save_event(
                id,
                &conversation::Event::Append(crate::conversation::Message::user(
                    "hello\u{1b}[31m\nffffffff-0000-0000-0000-000000000000  2026-01-01 00:00:00  \
                     other  harmless",
                )),
            )
            .await
            .expect("seed a first message");

        let (sessions, _) = manager
            .list_sessions(10, false, None, None)
            .await
            .expect("list");
        let rows = session_rows(&sessions, &ids_of(&sessions));
        let [row] = rows.as_slice() else {
            panic!("one session, one row: {rows:?}");
        };
        for cell in row {
            assert!(
                !cell.contains('\u{1b}') && !cell.contains('\n'),
                "a cell reaches a terminal verbatim, so neither may survive: {cell:?}"
            );
        }
        assert!(
            unicode_width::UnicodeWidthStr::width(row[3].as_str()) <= TABLE_WIDTH,
            "and the preview is capped in columns: {:?}",
            row[3]
        );
    }
    /// The profile column takes what real names need, and the preview spends what is left.
    ///
    /// Profile names are descriptive in practice (`openrouter-anthropic-messages` is 29 columns),
    /// and one cut short enough to stop distinguishing two profiles is worse than a shorter
    /// preview. Reserving the ceiling instead would spend that width even on an installation with
    /// one four-character name.
    #[tokio::test]
    async fn the_provider_column_takes_what_it_needs_and_the_preview_takes_the_rest() {
        async fn widths(providers: &[&str]) -> (usize, usize) {
            let manager = Store::for_test().await;
            for provider in providers {
                let id = manager
                    .create_session(None, (*provider).to_string())
                    .await
                    .expect("create");
                manager
                    .save_event(
                        id,
                        &conversation::Event::Append(crate::conversation::Message::user(
                            "word ".repeat(40),
                        )),
                    )
                    .await
                    .expect("seed");
            }
            let (sessions, _) = manager
                .list_sessions(10, false, None, None)
                .await
                .expect("list");
            let rows = session_rows(&sessions, &ids_of(&sessions));
            let width = |index: usize| {
                rows.iter()
                    .map(|row| unicode_width::UnicodeWidthStr::width(row[index].as_str()))
                    .max()
                    .unwrap_or(0)
            };
            (width(2).max("Profile".len()), width(3))
        }

        let (short_provider, wide_preview) = widths(&["stub"]).await;
        let (long_provider, narrow_preview) =
            widths(&["openrouter-anthropic-messages", "claude-max"]).await;

        assert_eq!(
            long_provider, 29,
            "a real profile name is shown in full, not cut to a fixed ceiling"
        );
        assert!(
            narrow_preview < wide_preview,
            "and the preview pays for it: {narrow_preview} should be under {wide_preview}"
        );
        for (provider, preview) in [
            (short_provider, wide_preview),
            (long_provider, narrow_preview),
        ] {
            assert_eq!(
                crate::text::ID_PREFIX
                    + 2
                    + Precision::Minutes.width()
                    + 2
                    + provider
                    + 2
                    + preview,
                TABLE_WIDTH,
                "every combination spends the budget exactly"
            );
        }
    }

    /// A `config.toml` that does not parse leaves nothing to check an archive's profile against.
    /// The import is refused before anything is written and names the parse error, rather than
    /// writing the archive's profile name unchecked over a warning.
    #[tokio::test]
    async fn an_unreadable_config_refuses_an_import_before_anything_is_written() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.toml"), "bogus = 1\n").expect("write config");
        let archive = dir.path().join("archive.json");
        std::fs::write(
            &archive,
            serde_json::json!({
                "format_version": crate::store::export::SESSION_EXPORT_FORMAT_VERSION,
                "meka_version": "0.0.0",
                "exported_at": "2020-01-01T00:00:00Z",
                "root_session_id": "11111111-1111-4111-8111-111111111111",
                "sessions": [{
                    "id": "11111111-1111-4111-8111-111111111111",
                    "parent_id": null,
                    "created_at": "2020-01-01T00:00:00Z",
                    "updated_at": "2020-01-01T00:00:00Z",
                    "cwd": null,
                    "permission": null,
                    "capabilities_json": null,
                    "profile": "work",
                    "stats": crate::stats::SessionStatsSnapshot::default(),
                    "events": [],
                    "tool_outputs": {},
                }],
            })
            .to_string(),
        )
        .expect("write archive");
        let store = Store::for_test().await;
        // What `main.rs` hands this door when the file could not be read: nothing to check against.
        let profiles = crate::store::export::ImportProfiles {
            selected: None,
            default: None,
            configured: None,
        };

        // SAFETY: `MEKA_CONFIG_DIR` is process-global; `CONFIG_DIR_ENV_LOCK` serializes every test
        // that sets it.
        let _guard = crate::config::CONFIG_DIR_ENV_LOCK.lock().await;
        unsafe { std::env::set_var("MEKA_CONFIG_DIR", dir.path()) };
        let result = import_session(
            &store,
            &archive.to_string_lossy(),
            profiles,
            Some(crate::permission::Permission::Read),
        )
        .await;
        unsafe { std::env::remove_var("MEKA_CONFIG_DIR") };

        let error = match result {
            Ok(()) => panic!("an unreadable config must refuse the import"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("failed to parse") && error.contains("bogus"),
            "the parse error is the answer: {error}"
        );
        let (sessions, _) = store
            .list_sessions(10, true, None, None)
            .await
            .expect("list");
        assert!(
            sessions.is_empty(),
            "a refused import writes nothing: {sessions:?}"
        );
    }
}
