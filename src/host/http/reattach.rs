//! GC-evicted-session re-attach. When a session's in-memory `SessionEntry` is dropped (because
//! the GC scanner exceeded `idle_timeout`) but its DB row is retained (the default `delete_on_idle
//! = false`), a later request with the same session UUID rebuilds the runtime from disk and
//! continues the conversation. Mirrors ACP's `session/load` semantics
//! (`crate::host::acp::session::handle_load_session`).
//!
//! Inserted between every mutating handler and its session-map lookup: instead of
//! `state.sessions.read().await.get(&id).cloned()` returning `None` → 404, the handler calls
//! [`ensure_session_loaded`] which falls through to reconstruction.
//!
//! Permission and per-session capabilities are persisted on the `sessions` row
//! (`src/session.rs::SessionSummary`), so a session created with `permission = "read"` re-attaches
//! with `permission = "read"`, not the process default.

use std::sync::{Arc, RwLock};

use axum::http::StatusCode;
use uuid::Uuid;

use super::{
    errors::{ErrorKind, ProblemDetail},
    http_frontend::{HttpFrontend, SessionCapabilities},
    state::{ServerState, SessionEntry},
};
use crate::{
    permission::{Permission, SharedPermission},
    workspace::SharedCwd,
};

/// How a failed [`crate::host::build_session_agent`] reaches the caller.
///
/// Something the caller can fix is the caller's to fix, not a server fault, and its message is
/// meka's own rather than an upstream provider's, so it goes back verbatim. Three cases: a session
/// pinned to a profile that has since left `config.toml`, a profile that is configured but has no
/// stored credential, and a session another one spawned, which only its parent can drive. All three
/// name what to do about it, and sanitizing them to "internal server error; consult server logs"
/// left the one actionable sentence in a file the caller cannot read.
///
/// **Caller-remediable, which is narrower than "meka's own words".**
/// [`crate::error::MekaError::Installation`] is meka's own sentence too and is deliberately absent
/// from the list: it names the operator's `ca_cert_file`, proxy or `base_url`, which no caller can
/// act on and none should read. It falls to the sanitized arm below, as every other server fault
/// does.
///
/// Shared by re-attach and by session creation, so both answer a builder refusal the same way.
/// Creation is the half a fresh `meka serve` hits first, since a profile's credential is checked
/// when a session first asks for it rather than at startup.
///
/// Which status and `type` each refusal gets is [`ProblemDetail::for_error`]'s to say; this only
/// decides which variants are refusals a builder can raise, and sanitizes the rest.
pub(crate) fn agent_build_problem(
    id: Uuid,
    context: &str,
    error: impl Into<anyhow::Error>,
) -> ProblemDetail {
    use crate::error::MekaError;
    // Builders answer in `anyhow` and the profile resolver in `MekaError`; both downcast the same
    // way once wrapped, so one discriminator serves every door.
    let error: anyhow::Error = error.into();
    let problem = match error.downcast_ref::<MekaError>() {
        Some(
            refusal @ (MekaError::Config(_)
            | MekaError::SessionNotDrivable(_)
            | MekaError::ProfileNotConfigured { .. }
            | MekaError::SessionNotFound(_)
            | MekaError::SessionLocked(_)),
        ) => ProblemDetail::for_error(refusal, false),
        _ => ProblemDetail::internal_sanitized(context, error),
    };
    problem.with("session_id", id.to_string())
}

/// The 404 for an id that names no session, from the one mapping every refusal goes through.
///
/// `relay_provider_errors` is `false` because the variant carries no upstream body for the switch
/// to decide about.
pub(crate) fn session_not_found(id: Uuid) -> ProblemDetail {
    ProblemDetail::for_error(&crate::error::MekaError::SessionNotFound(id), false)
}

/// Assert a session exists, without building anything.
///
/// The counterpart to [`ensure_session_loaded`] for read-only handlers. Reconstruction builds an
/// `Agent`, a `ToolRegistry` and an MCP-attached registry, then pins the result in the session map
/// until the GC scanner evicts it again; a handler that only reads rows out of SQLite (messages,
/// export, background tasks, scheduled jobs) has no use for any of that and should not resurrect a
/// session as a side effect of being asked about it. Sub-agent transcripts make the difference
/// concrete: listing a spawn tree would otherwise revive every sub-agent in it.
pub(crate) async fn require_session_exists(
    state: &ServerState,
    id: Uuid,
) -> Result<(), ProblemDetail> {
    let exists = state
        .shared
        .store
        .session_exists(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to look up session", error)
                .with("session_id", id.to_string())
        })?;
    if exists {
        return Ok(());
    }
    Err(session_not_found(id))
}

/// One reconstruction at a time per session id, owned by the server state so a process serving two
/// stores could never share it.
///
/// Reconstruction takes the session's file lock, so two requests arriving together for a session
/// this process has not loaded raced: the loser got a `session-locked` 409 whose documented remedy
/// ("another process holds it") was wrong, because the holder was this process, a millisecond
/// ahead. Serializing makes the loser wait and then find the winner's entry on the re-check below.
#[derive(Clone, Default)]
pub(crate) struct ReconstructionLocks(
    Arc<std::sync::Mutex<std::collections::HashMap<Uuid, Arc<tokio::sync::Mutex<()>>>>>,
);

impl ReconstructionLocks {
    /// Acquire the reconstruction lock for `id`, creating it on first use. Entries whose only
    /// remaining owner is the map are dropped on the way past, so it stays bounded by concurrent
    /// reconstructions rather than by every session the process has ever loaded.
    ///
    /// Also taken by the dormant fast paths in `handlers::sessions` and `handlers::conversation`,
    /// which decide what to write on the strength of the session *not* being resident and must not
    /// have that stop being true while they write. `assert_dormant_fast_path_is_serialized` below
    /// is the one statement of what such a path owes.
    pub(crate) async fn lock(&self, id: Uuid) -> tokio::sync::OwnedMutexGuard<()> {
        let lock = {
            let mut registry = crate::sync::lock(&self.0);
            registry.retain(|_, held| Arc::strong_count(held) > 1);
            Arc::clone(
                registry
                    .entry(id)
                    .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
            )
        };
        lock.lock_owned().await
    }

    #[cfg(test)]
    fn held(&self) -> usize {
        crate::sync::lock(&self.0).len()
    }
}

/// Assert that a handler which writes to a session *without* loading it holds both locks that make
/// the decision it wrote on still true at the moment it writes.
///
/// Two handlers bypass [`ensure_session_loaded`] when a session is not resident, and both repeat
/// one shape: take the reconstruction lock, re-check residency under it and hand the caller back to
/// the resident path if the answer moved, take the session's on-disk lock, and only then write.
/// Stated once here rather than per handler, because the likely regression is a new site rather
/// than a changed rule.
///
/// A source-text assertion, because what it protects is unreachable from a single process. The
/// reconstruction lock is per process by construction, and the session lock only conflicts across
/// them, so reproducing either failure needs a second `meka serve` on one store. `signature` names
/// the function, `body_sentinel` is a string near the end of it that proves the scanned region
/// still covers the whole body, and `write` is the call the whole arrangement exists to protect.
#[cfg(test)]
pub(crate) fn assert_dormant_fast_path_is_serialized(
    source: &str,
    signature: &str,
    body_sentinel: &str,
    write: &str,
) {
    // Line endings normalized first, because the delimiter below is the one thing here that a CRLF
    // checkout breaks *silently*. The repo has no `.gitattributes` and CI's `windows-latest` job
    // checks out under Git for Windows' `core.autocrlf = true`, so `\r\n}\r\n` never matches and
    // `split` yields the whole remainder instead: the region widens from one function to the rest
    // of the file, `body_sentinel` matches trivially, and this assertion reads as coverage on two
    // platforms while proving much less on the third.
    let source = source.replace("\r\n", "\n");
    // To the function's closing brace at column zero. Every brace inside the body is indented, so
    // this is exact, where cutting at the next doc comment would drag in whatever follows.
    let body = source
        .split(signature)
        .nth(1)
        .expect("the function this assertion is about")
        .split("\n}\n")
        .next()
        .expect("splitting always yields a first part");
    assert!(
        body.contains(body_sentinel),
        "the scanned region no longer covers {signature}, so this assertion proves nothing"
    );

    let reconstruction = body.find("reconstruction_locks.lock(id)").expect(
        "a dormant write must hold the reconstruction lock, or the residency it decided on can \
         stop being true while it writes",
    );
    let recheck = body
        .find("state.sessions.read().await.contains_key(&id)")
        .expect(
            "and it must re-check residency under that lock, returning `Ok(None)` so the caller \
             takes the resident path and the live agent moves with the row",
        );
    let write_at = body
        .find(write)
        .expect("the write this whole arrangement exists to protect is no longer in this function");
    assert!(
        reconstruction < recheck && recheck < write_at,
        "the lock must be taken first, the residency re-checked under it, and only then the write \
         issued; found reconstruction@{reconstruction} recheck@{recheck} write@{write_at}"
    );
    // The re-check has to *act*, not merely look. Replacing its `return Ok(None)` with a log line
    // leaves all three positions intact and reinstates the race.
    assert!(
        body.get(recheck..write_at)
            .is_some_and(|arm| arm.contains("return Ok(None)")),
        "the residency re-check must hand the caller back to the resident path, not just observe \
         that the session came back"
    );

    // The reconstruction lock is per process, so it says nothing about a second `meka serve` on the
    // same store, which would otherwise answer `200` for a session this one is running.
    let session = body
        .find("lock_session(id)")
        .expect("a dormant write must own the session it writes to, not just the process's map");
    assert!(
        session < write_at,
        "the session lock must be held before the write; found session@{session} write@{write_at}"
    );

    // Presence is not enough: `let _ = lock_session_reconstruction(id).await` compiles, reads as a
    // lock, and drops the guard at the end of that very statement, so every await after it runs
    // unprotected. That one-character edit keeps this assertion green while reinstating the whole
    // race, which is the only way these functions can regress silently.
    for guard in ["_reconstruction", "_session"] {
        assert!(
            body.contains(&format!("let {guard} =")),
            "the {guard} guard must be bound to a name, not `let _ =`, which drops it immediately \
             and leaves every await below it unserialized"
        );
        assert!(
            !body.contains(&format!("drop({guard}")),
            "and {guard} must live to the end of the function; dropping it early is the same \
             defect as never binding it"
        );
    }
}

/// Look up a session, reconstructing it from the persisted DB row if the in-memory entry has been
/// evicted. Returns the (now in-memory) `SessionEntry` on success, a 404 problem detail when the
/// session id is unknown to both the map and the DB, or a 500-class problem detail when
/// reconstruction fails.
///
/// On reconstruction, emits a `tracing::info!` so operators can see re-attach events in their
/// observability pipeline.
///
/// **Takes `state.sessions.write()` internally**, so a caller must not already hold a guard on that
/// map. Three call sites `drop(map)` immediately before reaching here for exactly this reason, and
/// nothing in the type system enforces it.
pub(crate) async fn ensure_session_loaded(
    state: &ServerState,
    id: Uuid,
) -> Result<SessionEntry, ProblemDetail> {
    ensure_session_loaded_holding(state, id, None).await
}

/// [`ensure_session_loaded`] for a caller that already holds the session's file lock: the fork
/// handler, which takes the copy's lock before the copy's row exists so a concurrent sweep cannot
/// delete it in between. `flock` is per open file description, so re-attach cannot take a lock the
/// same process holds and has to be handed it instead.
pub(crate) async fn ensure_session_loaded_holding(
    state: &ServerState,
    id: Uuid,
    held_lock: Option<crate::fs::FileLock>,
) -> Result<SessionEntry, ProblemDetail> {
    // Fast path: in-memory entry.
    if let Some(entry) = state.sessions.read().await.get(&id).cloned() {
        return Ok(entry);
    }

    let _reconstruction = state.reconstruction_locks.lock(id).await;

    // Re-check now that this task owns the reconstruction: a concurrent caller may have finished
    // while this one waited, and its entry is the one to use.
    if let Some(entry) = state.sessions.read().await.get(&id).cloned() {
        return Ok(entry);
    }

    let started = std::time::Instant::now();

    // Before the lock, not merely before the build. `build_session_agent` refuses a sub-agent at
    // the end of this function, and what the intervening work costs is one wrong *answer*: a
    // sub-agent whose parent is currently running it has its lock held, so `lock_session` below
    // fails first and the caller gets `409 session-locked` -- "another process has it, try
    // again" -- for a condition no retry ever resolves. The 422 is the true answer and the
    // guard is what makes it the one that arrives. Hydrating the event log for a session about
    // to be refused is wasted besides.
    //
    // Deliberately *not* justified by the `claim_session` sweep below. That sweep is keyed on this
    // id and a sub-agent has no `background_tasks` rows to key: `Agent::enable_background` is
    // reached only through `assemble_agent`, which is the host builders' path, and
    // `Agent::new_subagent` never calls it. The UPDATE matches zero rows.
    //
    // Here rather than at each door, because every write-side session endpoint funnels through this
    // function: `/turn`, `/compact`, `/rewind`, `/responses`, the fork's read-back and the
    // scheduler fire. Placing it at the doors is what left `patch_session` needing its own copy for
    // the one branch that never calls this.
    //
    // On the cold path only, which the two fast-path returns above skip. That is safe because a
    // sub-agent cannot become *resident*: the only code that inserts into `state.sessions` is `POST
    // /v1/sessions` (a fresh row with no parent), the fork handler (which refuses a sub-agent
    // source before copying, so its copy is never one) and this function. The other two writers
    // of that map only ever remove. Residency therefore implies the row already passed here.
    // Moving the check above the fast path would buy nothing and put a store read on every
    // request to a live session; a future inserter belongs behind this rule rather than in
    // front of it.
    crate::host::refuse_a_spawned_session(&state.shared.store, Some(id))
        .await
        .map_err(|error| agent_build_problem(id, "failed to read session", error))?;

    // Cold path: lock the row, then read it, so what is rebuilt here is what this process now
    // owns. The fork handler arrives holding the copy's lock already (`flock` is per open file
    // description, so re-attach cannot take a lock the same process holds) and reads without one.
    // `session-locked` (not `turn-in-flight`) for a held row: this is a cross-process file-lock
    // conflict, not an in-process turn concurrency issue.
    let (session_lock, summary) = match held_lock {
        Some(lock) => {
            let summary = state
                .shared
                .store
                .session_info(id)
                .await
                .map_err(|error| {
                    ProblemDetail::internal_sanitized(
                        "failed to look up session during re-attach",
                        error,
                    )
                    .with("session_id", id.to_string())
                })?
                .ok_or_else(|| session_not_found(id))?;
            (lock, summary)
        }
        None => state
            .shared
            .store
            .open_session_row(id)
            .await
            .map_err(|error| agent_build_problem(id, "failed to re-attach session", error))?,
    };

    // Resolve persisted permission. The row is the authority: every door records a level, so this
    // process's default answers only for a row an outside hand left bare, or one whose level the
    // operator has since withdrawn from `[permissions].enabled`, which the admitter warns about.
    let enabled = state.shared.config.enabled_permissions;
    let permission: Permission = enabled
        .admit_recorded(summary.permission, &format!("session {id}"))
        .unwrap_or(state.shared.config.permission);
    let shared_permission =
        SharedPermission::new(permission, enabled).with_approvals(summary.approvals);

    // Resolve persisted capabilities. NULL → defaults. Parse failures are surfaced via
    // `warn!` rather than silently falling back, so schema mismatches are operator-visible.
    let capabilities = match summary.capabilities_json.as_deref() {
        Some(json) => match serde_json::from_str::<SessionCapabilities>(json) {
            Ok(parsed) => parsed,
            Err(error) => {
                tracing::warn!(
                    session_id = %id,
                    error = %error,
                    "failed to parse capabilities_json; using the default capabilities"
                );
                SessionCapabilities::default()
            }
        },
        None => SessionCapabilities::default(),
    };

    // A row can carry no `cwd`: `meka session import` stores the archive's value verbatim, and an
    // archive may omit it. Default to the server's process working directory, and propagate
    // `current_dir()` failure as 500 so the operator can fix it.
    let cwd_path = match summary.cwd.clone() {
        Some(path) => path,
        None => std::env::current_dir().map_err(|error| {
            ProblemDetail::new(
                ErrorKind::Internal,
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "server cannot resolve a default working directory for session re-attach: {error}"
                ),
            )
            .with("session_id", id.to_string())
        })?,
    };
    let cwd = SharedCwd::new(cwd_path.clone());

    let http_frontend = Arc::new(HttpFrontend::with_capabilities(capabilities));
    let frontend_dyn: Arc<dyn crate::frontend::Frontend> = http_frontend.clone();

    let context_used = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let context_overhead = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let resident = crate::host::open_session(
        &state.shared,
        id,
        crate::host::SessionSpec {
            session_id: Some(id),
            permission: shared_permission.clone(),
            frontend: frontend_dyn,
            cwd: cwd.clone(),
            roots: crate::workspace::SharedRoots::default(),
            context_tokens: Arc::clone(&context_used),
            context_overhead: Arc::clone(&context_overhead),
            context_window: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        },
        crate::host::Opening::Hydrate,
        session_lock,
        crate::host::CancelCell::default(),
    )
    .await
    .map_err(|error| agent_build_problem(id, "failed to rebuild session agent", error))?;

    let now = chrono::Utc::now();
    let parsed_created_at = chrono::DateTime::parse_from_rfc3339(&summary.created_at)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .unwrap_or(now);
    let parsed_updated_at = chrono::DateTime::parse_from_rfc3339(&summary.updated_at)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .unwrap_or(now);
    let new_entry = SessionEntry {
        resident,
        token_id: summary.token_id.clone(),
        created_at: parsed_created_at,
        updated_at: Arc::new(RwLock::new(parsed_updated_at)),
        last_turn_at_wall: Arc::new(RwLock::new(None)),
        capabilities,
        frontend: http_frontend,
    };

    // Re-checked before the write lock rather than under it: the map's lock is write-preferring,
    // and a store round trip held under the write guard queued every handler on this process
    // behind one re-attach. The reconstruction lock above is what serializes this against a
    // concurrent load; a delete that lands in the gap between this read and the insert leaves an
    // entry the next `save_events_atomic` fails on its foreign key, which is what it did before.
    let still_exists = state
        .shared
        .store
        .session_exists(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized(
                "failed to verify session existence during re-attach",
                error,
            )
            .with("session_id", id.to_string())
        })?;
    if !still_exists {
        // Released like an evicted entry: `open_session` attached its registry to the MCP manager,
        // and dropping the entry without detaching left a registry `tools/list_changed` fanned out
        // to for the rest of the process.
        new_entry.release(state.shared.mcp_manager.as_ref()).await;
        return Err(session_not_found(id));
    }
    let mut sessions = state.sessions.write().await;
    if let Some(existing) = sessions.get(&id).cloned() {
        return Ok(existing);
    }
    sessions.insert(id, new_entry.clone());
    drop(sessions);

    tracing::info!(
        "session re-attached: id={id} elapsed_ms={elapsed_ms} permission={permission} \
         cwd={cwd_path:?}",
        elapsed_ms = started.elapsed().as_millis(),
    );

    Ok(new_entry)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sub-agent's refusal reaches the client as its own words, not as "consult server logs".
    ///
    /// The refusal names the parent and `agent_followup`, and every sentence of that is meka's
    /// own; sanitizing it would leave a `sessions:w` holder with a 500 and no way to learn that
    /// the id it used names a sub-agent. 422 rather than 403 for the reason [`agent_build_problem`]
    /// gives generally: no token and no permission level lifts it, so routing it as an auth
    /// failure would send a client to re-provision something that cannot help.
    ///
    /// Its own `type`, and not [`ErrorKind::InvalidBody`], for the same reason one step down: that
    /// URI reads as "your payload is malformed", and a client acting on it rewrites the payload
    /// forever against an id that will never accept one.
    #[test]
    fn a_worker_refusal_reaches_the_client_verbatim_as_422() {
        let id = Uuid::from_u128(0xb0b);
        let problem = agent_build_problem(
            id,
            "failed to rebuild session agent",
            crate::error::MekaError::SessionNotDrivable(
                "session A was spawned by session B; continue it with `agent_followup`".to_string(),
            ),
        );
        assert_eq!(problem.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(problem.type_uri, ErrorKind::SessionNotDrivable.type_uri());
        let detail = problem.detail.as_deref().unwrap_or_default();
        assert!(
            detail.contains("agent_followup"),
            "the one actionable sentence must survive the mapping: {detail}"
        );
    }

    /// An installation fault is not a refusal the caller can act on, so it does not go back
    /// verbatim.
    ///
    /// It arrives here the same way a `Config` refusal does -- from a builder, in meka's own words
    /// -- which is exactly why the list has to be by variant rather than by "is this meka's own
    /// sentence". A `[web]` client that cannot be built named the operator's `ca_cert_file` in a
    /// 422 `invalid-body`, telling a client to correct a payload that was never wrong.
    #[test]
    fn an_installation_fault_is_sanitized_rather_than_answered_as_a_refusal() {
        let problem = agent_build_problem(
            Uuid::from_u128(0xca),
            "failed to build session agent",
            crate::error::MekaError::Installation(
                "[web].proxy: invalid URL 'socks5://10.0.0.1:1080': …".to_string(),
            ),
        );
        assert_eq!(problem.status, StatusCode::INTERNAL_SERVER_ERROR);
        let detail = problem.detail.unwrap_or_default();
        assert!(!detail.contains("10.0.0.1"), "{detail}");
    }

    /// Two requests for the same unloaded session must not reconstruct it concurrently.
    ///
    /// Both would open the session's file lock; the loser gets a 409 whose documented remedy
    /// ("retry against the process that holds it") does not apply, because the winner is this very
    /// process. Serializing reconstruction per session id makes the loser wait and then find the
    /// winner's entry.
    #[tokio::test]
    async fn reconstructing_one_session_twice_is_serialized() {
        let id = Uuid::from_u128(0x5eed);
        let locks = ReconstructionLocks::default();
        let inside = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let overlapped = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let inside = Arc::clone(&inside);
            let overlapped = Arc::clone(&overlapped);
            let locks = locks.clone();
            handles.push(tokio::spawn(async move {
                let _guard = locks.lock(id).await;
                if inside.fetch_add(1, std::sync::atomic::Ordering::SeqCst) != 0 {
                    overlapped.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                // Long enough that an unserialized run overlaps essentially every time.
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                inside.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            }));
        }
        for handle in handles {
            handle.await.expect("task");
        }

        assert!(
            !overlapped.load(std::sync::atomic::Ordering::SeqCst),
            "two tasks reconstructed the same session id at once; both would race its file lock"
        );
    }

    /// The per-id locks must not accumulate forever: one entry per session ever re-attached would
    /// be an unbounded map keyed by attacker-suppliable ids. `retain` drops the ones nobody holds.
    #[tokio::test]
    async fn the_reconstruction_lock_registry_does_not_grow_without_bound() {
        let locks = ReconstructionLocks::default();
        for index in 0..64u128 {
            let guard = locks.lock(Uuid::from_u128(index)).await;
            drop(guard);
        }
        // The next acquisition sweeps the released entries, so the map holds only the live one.
        let _guard = locks.lock(Uuid::from_u128(9999)).await;
        let held = locks.held();
        assert!(
            held <= 2,
            "released reconstruction locks are never swept; the map grows per session id: {held}"
        );
    }

    /// A configuration refusal is the caller's to fix and its message is meka's own, so it goes
    /// back verbatim with a 422. Sanitizing it to a 500 hides where a fresh `meka serve` lands
    /// first: the profile is configured but has no credential, and the one sentence saying so goes
    /// to a log file the caller cannot read.
    #[test]
    fn a_configuration_refusal_reaches_the_caller_instead_of_becoming_a_500() {
        let id = Uuid::new_v4();
        let problem = agent_build_problem(
            id,
            "failed to build session agent",
            anyhow::Error::new(crate::error::MekaError::Config(
                "profile 'work' has no stored credential".to_string(),
            )),
        );
        assert_eq!(problem.status, StatusCode::UNPROCESSABLE_ENTITY);
        let detail = problem.detail.unwrap_or_default();
        assert!(
            detail.contains("no stored credential"),
            "the actionable sentence must survive: {detail}"
        );

        // Anything else is a server fault and stays sanitized, so an upstream provider's message
        // cannot reach a client through this door.
        let opaque = agent_build_problem(
            id,
            "failed to build session agent",
            anyhow::anyhow!("upstream said something with a token in it"),
        );
        assert_eq!(opaque.status, StatusCode::INTERNAL_SERVER_ERROR);
        let sanitized = opaque.detail.unwrap_or_default();
        assert!(!sanitized.contains("token in it"), "{sanitized}");
    }
}
