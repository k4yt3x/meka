//! Session CRUD: create, list, get, delete.
//!
//! Mirrors the ACP session lifecycle (`session/new` / `session/list` / etc.) but over HTTP+JSON
//! and with `Authorization: Bearer` gating per scope.

use std::sync::{Arc, RwLock};

use axum::{
    Json,
    body::Bytes,
    extract::{Path, Query, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use crate::{
    host::http::{
        errors::{ErrorKind, ProblemDetail},
        http_frontend::{HttpFrontend, SessionCapabilities},
        reattach::{
            agent_build_problem, ensure_session_loaded, require_session_exists, session_not_found,
        },
        scope,
        state::{ServerState, SessionEntry},
    },
    permission::{EnabledPermissions, Permission, SharedPermission},
    store::Store,
    view::SessionView,
    workspace::SharedCwd,
};

/// RAII guard that deletes a freshly-created session DB row when an in-flight create handler
/// returns an error after the row has been written. Without this, a failure between
/// `create_session_with_metadata` and the final success response leaves an orphan row.
///
/// `Drop` can't `.await` the async `delete_session` call directly, so we spawn it on the runtime.
/// The cleanup task runs after the response has flushed; that's fine because nothing else can
/// observe the orphaned row until the next `GET /v1/sessions` scan.
struct SessionRollback {
    uuid: Uuid,
    manager: Store,
    armed: bool,
}

impl SessionRollback {
    fn new(uuid: Uuid, manager: Store) -> Self {
        Self {
            uuid,
            manager,
            armed: true,
        }
    }

    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for SessionRollback {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // Guard `tokio::spawn` with `Handle::try_current`: during graceful shutdown the
        // runtime may already be tearing down and an unguarded spawn would panic.
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::debug!(
                "session rollback: skipping orphan-row delete for {uuid} during shutdown",
                uuid = self.uuid
            );
            return;
        };
        let uuid = self.uuid;
        let manager = self.manager.clone();
        handle.spawn(async move {
            // The guarded door: the row's lock went into the agent build that failed, and `meka
            // -c` in another process can take the newest root before this task runs. The plain
            // door would cascade the row away under it; this one refuses and leaves the row with
            // whoever holds it.
            if let Err(error) = manager.delete_session_unless_attached(uuid).await {
                tracing::warn!("session rollback: failed to delete orphan row {uuid}: {error}");
            } else {
                tracing::info!("session rollback: deleted orphan row {uuid}");
            }
        });
    }
}

/// `deny_unknown_fields` rejects typos like `permision: "read"` with 422 instead of silently
/// falling back to defaults.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CreateSessionRequest {
    /// Absolute path. Defaults to the server process's `current_dir` if omitted.
    #[schema(value_type = Option<String>)]
    pub(crate) cwd: Option<std::path::PathBuf>,
    /// Permission level the session starts in. Defaults to the server's configured default
    /// from `[permissions].default` (typically `read`). Must be in the enabled set.
    pub(crate) permission: Option<String>,
    /// Whether a call above the level is submitted for approval rather than refused. Defaults to
    /// the server's `[permissions].approvals`.
    pub(crate) approvals: Option<bool>,
    /// Profile the session runs on, and keeps for the rest of its life unless a later PATCH moves
    /// it. Defaults to the server's own default profile. Must name a profile in `config.toml`;
    /// `GET /v1/profiles` lists them.
    pub(crate) profile: Option<String>,
    /// Per-session capability flags. See the HTTP API docs § "Capabilities".
    #[serde(default)]
    pub(crate) capabilities: CapabilitiesBody,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CapabilitiesBody {
    /// When `true`, the SSE stream includes `thinking.delta` events for extended-thinking
    /// content. Default `false`: chat-transcript clients (Telegram bridges) don't want
    /// reasoning inline.
    #[serde(default)]
    pub(crate) supports_reasoning_stream: bool,
    /// When `false`, mid-turn permission requests are denied immediately with a notice instead of
    /// parking on the SSE channel. Streaming clients with no approval interface set this. Default
    /// `true`, so a client that says nothing keeps the chance to approve.
    #[serde(default = "default_true")]
    pub(crate) supports_permission_prompts: bool,
}

/// `#[serde(default)]` on a `bool` yields `false`; this is for the fields that default to `true`.
fn default_true() -> bool {
    true
}

impl Default for CapabilitiesBody {
    fn default() -> Self {
        Self {
            supports_reasoning_stream: false,
            supports_permission_prompts: true,
        }
    }
}

impl From<CapabilitiesBody> for SessionCapabilities {
    fn from(body: CapabilitiesBody) -> Self {
        Self {
            supports_reasoning_stream: body.supports_reasoning_stream,
            supports_permission_prompts: body.supports_permission_prompts,
        }
    }
}

/// Decode the persisted `capabilities_json` column back into a `SessionCapabilities`. NULL or
/// invalid JSON yields the defaults. Used on the DB-fallback path for evicted sessions.
fn capabilities_from_row(json: Option<&str>) -> SessionCapabilities {
    json.and_then(|raw| serde_json::from_str::<SessionCapabilities>(raw).ok())
        .unwrap_or_default()
}

/// A session as this server answers for it: the row every host prints, under the facts only the
/// process holding the session can add.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct SessionResponse {
    /// The row's own fields, the same object `meka session show --format json` prints. A resident
    /// session reports its level and clock from its cells rather than the row; re-attach is where
    /// the process default answers for a bare row, and that answer is written to the cells.
    #[serde(flatten)]
    pub(crate) session: SessionView,
    /// Wall-clock timestamp (RFC 3339) of the last successful turn on this session. Omitted when
    /// the session has never run a turn (just-created or just-re-attached). Distinct from
    /// `updated_at`, which advances on any session-level mutation (PATCH included).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_turn_at: Option<String>,
    /// Per-session capability flags declared at create time (or re-attach), echoed back so clients
    /// can confirm the settings their session actually ended up with.
    pub(crate) capabilities: SessionCapabilities,
    /// Whether a turn is running on this session right now.
    ///
    /// Exists so a client whose SSE stream dropped mid-turn can tell "my turn is still running"
    /// from "my turn died" without submitting a speculative turn and reading the 409. A dropped
    /// stream does not cancel the turn: the spawned task keeps the runtime lock and dropping the
    /// `JoinHandle` detaches rather than aborts, so the work completes and resubmitting would
    /// duplicate a reply the user is about to receive anyway.
    ///
    /// Always `false` for a GC-evicted session, since eviction requires an idle session.
    pub(crate) turn_in_flight: bool,
}

#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub(crate) struct ListSessionsQuery {
    #[serde(default)]
    pub(crate) limit: Option<u32>,
    #[serde(default)]
    pub(crate) cursor: Option<String>,
    /// Include sub-agent sessions. Default `false`, which lists only root conversations.
    ///
    /// Off by default because a dispatcher that spawns freely would otherwise bury its own
    /// sessions under the sub-agents it started, and a client paging through the list wants the
    /// conversations it created. Turn it on to audit what was spawned; `parent_id` on each row is
    /// what reconnects a sub-agent to the session that dispatched it.
    #[serde(default)]
    pub(crate) include_children: Option<bool>,
    /// Only list sessions whose working directory is this one, compared in the canonical spelling
    /// every session records.
    #[serde(default)]
    pub(crate) cwd: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ListSessionsResponse {
    pub(crate) sessions: Vec<SessionResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) next_cursor: Option<String>,
}

/// A caller's `cwd` through the one acceptor, refused as a 422 like every other field of the body.
fn accepted_cwd(path: &std::path::Path) -> Result<std::path::PathBuf, ProblemDetail> {
    // A cwd refusal carries no provider text, so the relay flag has nothing to decide.
    crate::workspace::accept_cwd(path).map_err(|error| ProblemDetail::for_error(&error, false))
}

/// POST /v1/sessions: create a session.
///
/// Requires scope `sessions:w`. The created session's runtime (Agent, ToolRegistry,
/// HttpFrontend) is constructed eagerly so subsequent `POST /turn` doesn't pay the build cost.
#[utoipa::path(
    post,
    path = "/v1/sessions",
    tag = "sessions",
    request_body = CreateSessionRequest,
    responses(
        (status = 201, description = "Session created", body = SessionResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid body", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:w"]))
)]
pub(crate) async fn create_session(
    State(state): State<ServerState>,
    scope::Scoped { principal, .. }: scope::Scoped<scope::SessionsWrite>,
    raw_body: Bytes,
) -> Result<(StatusCode, Json<SessionResponse>), ProblemDetail> {
    let body: CreateSessionRequest = serde_json::from_slice(&raw_body)
        .map_err(|error| ProblemDetail::invalid_body("session creation", error))?;

    let cwd_path = match body.cwd {
        Some(path) => accepted_cwd(&path)?,
        // Propagate `current_dir()` failure as 500 rather than falling back to a relative
        // path, which would surprise tools that resolve paths absolutely.
        None => std::env::current_dir().map_err(|error| {
            ProblemDetail::new(
                ErrorKind::Internal,
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("server cannot resolve a default working directory: {error}"),
            )
        })?,
    };
    let cwd = SharedCwd::new(cwd_path.clone());

    let permission: Permission = match body.permission.as_deref() {
        Some(value) => value.parse().map_err(|error| {
            ProblemDetail::new(
                ErrorKind::InvalidBody,
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("invalid `permission` value: {error}"),
            )
        })?,
        None => state.shared.config.permission,
    };
    let enabled: EnabledPermissions = state.shared.config.enabled_permissions;
    if !enabled.is_enabled(permission) {
        return Err(ProblemDetail::for_error(
            &enabled.disabled_level(permission),
            state.config.relay_provider_errors,
        ));
    }
    // Off unless the body or the config says otherwise; recorded on the row beside the level.
    let approvals = body.approvals.unwrap_or(state.shared.config.approvals);
    let shared_permission = SharedPermission::new(permission, enabled).with_approvals(approvals);

    let capabilities: SessionCapabilities = body.capabilities.into();
    let http_frontend = Arc::new(HttpFrontend::with_capabilities(capabilities));
    let frontend_dyn: Arc<dyn crate::frontend::Frontend> = http_frontend.clone();

    // Persist `permission` and `capabilities` so a GC-evicted session re-attaches with the
    // same shape the client created it with.
    let capabilities_json = serde_json::to_string(&capabilities).ok();
    // The profile this session will run with for the rest of its life, unless a PATCH moves it.
    let profile = match body.profile {
        Some(name) => {
            // Before the row is written: a name that resolves to nothing would fail every later
            // turn on this session, and the write is what makes it stick.
            crate::config::require_profile(&name, &state.shared.config.profiles).map_err(
                |error| ProblemDetail::for_error(&error, state.config.relay_provider_errors),
            )?;
            name
        }
        None => state
            .shared
            .default_profile()
            .map_err(|error| ProblemDetail::internal_sanitized("no default profile", error))?
            .to_string(),
    };
    // Created and locked in one step, the lock taken *before* the row exists. A row committed
    // ahead of its lock is visible to `meka session delete --all`, which enumerates at delete
    // time, takes the lock nobody holds yet, and cascades the session away underneath this
    // handler. See `Store::create_session_locked`.
    let (created, created_lock) = state
        .shared
        .store
        .create_session_locked(
            Some(cwd_path.clone()),
            permission.to_string(),
            approvals,
            capabilities_json,
            Some(principal.token_id.clone()),
            profile.clone(),
        )
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to create session", error))?;
    let session_uuid = created.id;
    // Parse the canonical `created_at` returned by the DB so the in-memory entry's timestamp
    // matches the persisted row exactly.
    let created_at_wall = chrono::DateTime::parse_from_rfc3339(&created.created_at)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .unwrap_or_else(|_| chrono::Utc::now());
    // Arm the rollback guard: every `?` below will clean up the orphan DB row on failure.
    let rollback = SessionRollback::new(session_uuid, state.shared.store.clone());
    // `None` means the claim could not be made at all -- an unwritable lock directory, descriptors
    // exhausted -- never that somebody else holds it, since no other process can know this id yet.
    // A served session that cannot be held alone is one this server must not admit.
    let session_lock = created_lock
        .map_err(|error| ProblemDetail::internal_sanitized("failed to lock session", error))?;

    // Build the per-session Agent + ToolRegistry.
    // Retained so `GET /context` and `GET /tools` can read them without the runtime mutex; see
    // the note on `SessionEntry::context_used`.
    let context_used = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let context_overhead = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let resident = crate::host::open_session(
        &state.shared,
        session_uuid,
        crate::host::SessionSpec {
            session_id: Some(session_uuid),
            permission: shared_permission.clone(),
            frontend: frontend_dyn,
            cwd: cwd.clone(),
            roots: crate::workspace::SharedRoots::default(),
            context_tokens: Arc::clone(&context_used),
            context_overhead: Arc::clone(&context_overhead),
            context_window: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        },
        crate::host::Opening::Fresh,
        session_lock,
        crate::host::CancelCell::default(),
    )
    .await
    .map_err(|error| agent_build_problem(session_uuid, "failed to build session agent", error))?;

    let entry = SessionEntry {
        resident,
        token_id: Some(principal.token_id.clone()),
        created_at: created_at_wall,
        updated_at: Arc::new(RwLock::new(created_at_wall)),
        last_turn_at_wall: Arc::new(RwLock::new(None)),
        capabilities,
        frontend: http_frontend,
    };

    state.sessions.write().await.insert(session_uuid, entry);

    // Just-created session has zero messages, so title is always empty. Skip the DB round-trip.
    let title = String::new();

    tracing::info!(
        "session created: id={session_uuid} cwd={cwd_path:?} permission={permission} \
         token={token}",
        token = principal.token_id,
    );

    // Past the point of no return: disarm so the rollback Drop doesn't fire.
    rollback.disarm();
    // Use the canonical `created_at` from the DB insert so all three surfaces agree.
    let timestamp = created.created_at;
    Ok((
        StatusCode::CREATED,
        Json(SessionResponse {
            session: SessionView {
                id: session_uuid,
                created_at: timestamp.clone(),
                updated_at: timestamp,
                cwd: Some(cwd_path),
                permission: Some(permission),
                approvals,
                profile,
                title,
                // A root by construction: `POST /v1/sessions` has no way to name a parent, and
                // a sub-agent session is only ever minted by `agent_spawn` inside a turn.
                parent_id: None,
            },
            last_turn_at: None,
            capabilities,
            turn_in_flight: false,
        }),
    ))
}

/// Body for `POST /v1/sessions/{id}/fork`. Every field is optional; omitted means "inherit from
/// the session being forked".
///
/// Only `cwd` is offered, mirroring ACP's `session/fork` request, which likewise carries a
/// workspace but no permission or capability fields. Both of those are inherited and remain
/// changeable afterwards via `PATCH /v1/sessions/{id}`.
#[derive(Debug, Default, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ForkSessionBody {
    /// Absolute path for the forked session. Absent → inherit the source's.
    #[schema(value_type = Option<String>)]
    pub(crate) cwd: Option<std::path::PathBuf>,
}

/// POST /v1/sessions/{id}/fork: copy a session's conversation into a new one.
///
/// Requires scope `sessions:w`. The copy starts with the source's full conversation and is
/// immediately usable; the source is left untouched and is not required to be in memory, so an
/// evicted session forks just as well as a live one. A source with a turn in flight is refused,
/// like every other write to it: the copy would end on a prompt nothing answered.
#[utoipa::path(
    post,
    path = "/v1/sessions/{id}/fork",
    tag = "sessions",
    params(("id" = Uuid, Path, description = "Session UUID to fork")),
    request_body = Option<ForkSessionBody>,
    responses(
        (status = 201, description = "Forked session", body = SessionResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 409, description = "A turn is in flight on the source; cancel first (`/errors/turn-in-flight`). Or another meka process holds the source (`/errors/session-locked`)", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid body, or the source is a sub-agent's conversation, whose copy is another sub-agent and so has no live session to hand back (`/errors/session-not-drivable`)", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:w"]))
)]
pub(crate) async fn fork_session(
    State(state): State<ServerState>,
    scope::Scoped { principal, .. }: scope::Scoped<scope::SessionsWrite>,
    Path(id): Path<Uuid>,
    raw_body: Bytes,
) -> Result<(StatusCode, Json<SessionResponse>), ProblemDetail> {
    // An empty body is the common case (`inherit everything`), and axum hands it to us as zero
    // bytes, which is not valid JSON.
    let body: ForkSessionBody = if raw_body.is_empty() {
        ForkSessionBody::default()
    } else {
        serde_json::from_slice(&raw_body)
            .map_err(|error| ProblemDetail::invalid_body("session fork", error))?
    };
    let cwd = body.cwd.as_deref().map(accepted_cwd).transpose()?;

    // Before the copy, not after it. A fork of a sub-agent is a sibling under the same parent
    // (`Store::fork_session_locked`), so the copy is a sub-agent too and `ensure_session_loaded`
    // below refuses to build it -- correctly, but far too late to say anything useful: the caller
    // would get a refusal naming an id it has never seen, for a row this handler had already
    // rolled back. Answering here names the id the caller actually sent.
    //
    // The store still copies the columns; what this door declines is the part of its own contract
    // it cannot honor, which is handing back a live session. `meka session fork` takes no runtime
    // and so still makes the copy.
    if let Some(terms) = state
        .shared
        .store
        .spawn_terms(id)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to read session", error))?
    {
        let detail = match terms.parent {
            Some(parent) => format!(
                "session '{id}' is a sub-agent of '{parent}', so a copy of it is another \
                 sub-agent and this endpoint has no live session to return. Read it with \
                 `GET /v1/sessions/{id}/messages`, or continue the conversation with \
                 `agent_followup` from '{parent}'."
            ),
            // `spawn_terms` rather than the parent link, so this names the id the caller sent even
            // for an imported sub-agent. Keyed on the link alone, that case falls through: the copy
            // is made, `ensure_session_loaded` refuses *it*, and the rollback runs, so the caller
            // gets a 422 naming an id it has never seen, for a row that no longer exists.
            None => format!(
                "session '{id}' is a sub-agent whose parent is not in this store, so nothing here \
                 can drive it or a copy of it. Read it with `GET /v1/sessions/{id}/messages`."
            ),
        };
        return Err(ProblemDetail::new(
            ErrorKind::SessionNotDrivable,
            StatusCode::UNPROCESSABLE_ENTITY,
            detail,
        )
        .with("session_id", id.to_string()));
    }

    // The locked fork, as the REPL's and ACP's: the copy's lock is taken before its row exists,
    // and handed to re-attach below, which cannot take a lock this process already holds (`flock`
    // is per open file description). Committing first and locking after left a window in which
    // `meka session delete --all` enumerated the copy, took the lock nobody held, and deleted it,
    // after which re-attach built a runtime on a vanished row and the first turn died on a foreign
    // key.
    //
    // A resident source is this process's own, held through its entry, so the store is told rather
    // than asked; a dormant one is probed, and a source another process is mid-turn on answers
    // `409 session-locked` rather than being copied half-written.
    //
    // Told only once it stands still. A turn persists its prompt before the provider answers, so
    // a resident source copied mid-turn ends on a user message nothing answered, the very copy the
    // probe refuses another process. Claimed under the sessions read lock the way compact is, so
    // DELETE's re-check sees it, then the conversation is `try_lock`ed for the out-of-band turn
    // that takes the mutex before marking itself busy. Both are held across the copy: a busy entry
    // is one the idle sweep leaves alone, where a momentary residency check could see the entry
    // evicted, and its file lock released, between the answer and the copy.
    let (source_idle, source_still, source_lock) = {
        let map = state.sessions.read().await;
        match map.get(&id).cloned() {
            Some(entry) => {
                let idle = entry
                    .claim_idle()
                    .ok_or_else(|| turn_in_flight_conflict(id, "fork the session"))?;
                let still = Arc::clone(&entry.conversation)
                    .try_lock_owned()
                    .map_err(|_| turn_in_flight_conflict(id, "fork the session"))?;
                (
                    Some(idle),
                    Some(still),
                    crate::store::SourceLock::HeldByCaller,
                )
            }
            None => (None, None, crate::store::SourceLock::Probe),
        }
    };
    let (forked, copy_lock) = state
        .shared
        .store
        .fork_session_locked(
            id,
            crate::store::ForkOverrides {
                cwd,
                // The HTTP API is single-root, but the copy keeps whatever the source recorded so
                // a later ACP `session/load` still sees the workspace shape. HTTP runtimes ignore
                // the column either way, exactly as re-attach already does for ACP-created
                // sessions.
                additional_roots: None,
                // Never inherited: this fingerprints the token that created a session, and the
                // token doing the forking is the only correct answer.
                token_id: Some(principal.token_id.clone()),
            },
            source_lock,
        )
        .await
        .map_err(|error| agent_build_problem(id, "failed to fork session", error))?
        .ok_or_else(|| session_not_found(id))?;
    // The copy has committed, so the source is free to move again, and stays free while the
    // copy's runtime is built below.
    drop(source_still);
    drop(source_idle);

    // A lock that could not be taken is reported by re-attach when it tries for itself.
    let copy_lock = match copy_lock {
        Ok(lock) => Some(lock),
        Err(error) => {
            tracing::warn!(
                "failed to lock forked session {forked_id} ahead of its row: {error}",
                forked_id = forked.id
            );
            None
        }
    };

    // The row's title, profile and parent, read back rather than copied off the source entry: the
    // fork's row is what the next turn on it will resolve, so the response quotes that. Before the
    // runtime is built, so a read that fails can still take the row with it; once the copy is
    // resident the caller would be told of a session the response then denied. A row this handler
    // wrote a moment ago and holds the lock on cannot be missing, so an absence is a fault too.
    let forked_info = match state.shared.store.session_info(forked.id).await {
        Ok(Some(info)) => Ok(info),
        Ok(None) => Err(crate::error::MekaError::SessionNotFound(forked.id)),
        Err(error) => Err(error),
    };
    let forked_info = match forked_info {
        Ok(info) => info,
        Err(error) => {
            drop(copy_lock);
            roll_back_fork(&state, forked.id).await;
            return Err(ProblemDetail::internal_sanitized(
                "failed to read the forked session",
                error,
            ));
        }
    };

    // Build the copy's runtime through the re-attach path rather than duplicating it: the row was
    // just written, so re-attach resolves permission, capabilities, cwd, and `token_id` straight
    // from it, hydrates the conversation, adopts the lock, and registers the entry.
    let entry = match crate::host::http::reattach::ensure_session_loaded_holding(
        &state, forked.id, copy_lock,
    )
    .await
    {
        Ok(entry) => entry,
        Err(problem) => {
            // The row exists but is unusable; drop it rather than leaving an orphan the caller
            // was never told about.
            roll_back_fork(&state, forked.id).await;
            return Err(problem);
        }
    };

    tracing::info!(
        "session forked: source={id} id={forked_id} token={token}",
        forked_id = forked.id,
        token = principal.token_id,
    );

    Ok((
        StatusCode::CREATED,
        Json(SessionResponse {
            session: SessionView {
                id: forked.id,
                created_at: forked.created_at.clone(),
                updated_at: forked.created_at,
                cwd: Some(entry.cells().cwd.get()),
                permission: Some(entry.cells().permission.get()),
                approvals: entry.cells().permission.approvals(),
                profile: forked_info.profile,
                title: forked_info.title,
                // Read back rather than assumed: forking a sub-agent session keeps it under the
                // same parent, so the copy is a sibling of the original, not a new root. That
                // sentence was here before the store did it: the copy took a NULL parent, so this
                // reported one thing and the store held another, and the difference was a
                // `sessions:w` holder's way around `crate::host::refuse_a_spawned_session`.
                parent_id: forked_info.parent_id,
            },
            last_turn_at: None,
            capabilities: entry.capabilities,
            turn_in_flight: false,
        }),
    ))
}

/// Delete a fork this handler could not put a session behind, so the caller is not left with a
/// full copy of the conversation under an id it was never told.
///
/// Through the guarded door. The copy's lock has gone by the time this runs, into the runtime
/// build that dropped it, and `meka -c` in another process can take the newest root in that gap;
/// the plain door would then cascade the row away under a session that process is running. The
/// guarded one refuses, and the row stays with whoever holds it. Best-effort: a failed cleanup is
/// worth a warning, not a second error replacing the one the caller needs to see.
async fn roll_back_fork(state: &ServerState, forked: Uuid) {
    if let Err(error) = state
        .shared
        .store
        .delete_session_unless_attached(forked)
        .await
    {
        tracing::warn!(
            "failed to roll back fork {forked} after its runtime could not be built: {error}"
        );
    }
}

/// GET /v1/sessions: paginated list. Returns persisted sessions from the DB (not just
/// in-memory entries) so audit consumers can see everything regardless of GC state.
#[utoipa::path(
    get,
    path = "/v1/sessions",
    tag = "sessions",
    params(ListSessionsQuery),
    responses(
        (status = 200, description = "Page of sessions", body = ListSessionsResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:r"]))
)]
pub(crate) async fn list_sessions(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SessionsRead>,
    Query(query): Query<ListSessionsQuery>,
) -> Result<Json<ListSessionsResponse>, ProblemDetail> {
    // At least one: a zero-row page came back with no cursor, which reads as "no sessions" to a
    // paging client on a store full of them.
    let limit = query.limit.unwrap_or(50).clamp(1, 200);
    let cwd_filter = query
        .cwd
        .as_deref()
        .map(|given| crate::workspace::cwd_filter(std::path::Path::new(given)));
    let (rows, next_cursor) = state
        .shared
        .store
        .list_sessions(
            limit,
            query.include_children.unwrap_or(false),
            cwd_filter.as_deref(),
            query.cursor.as_deref(),
        )
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to list sessions", error))?;

    // Enrich DB rows with live in-memory metadata where available; sessions with no in-memory entry
    // fall back to persisted columns, which are NULL for rows the HTTP server did not create (REPL,
    // ACP, sub-agent, imported).
    let in_memory = state.sessions.read().await;
    let sessions = rows
        .into_iter()
        .map(|row| {
            let live = in_memory.get(&row.id);
            // The row answers for an evicted session; a resident one is reported from its cells
            // and clock, which are what its next turn runs against.
            let mut session = SessionView::from(&row);
            if let Some(entry) = live {
                session.permission = Some(entry.cells().permission.get());
                session.approvals = entry.cells().permission.approvals();
                session.created_at = entry.created_at.to_rfc3339();
                if let Ok(updated_at) = entry.updated_at.read() {
                    session.updated_at = updated_at.to_rfc3339();
                }
            }
            let last_turn_at = live.and_then(|entry| {
                entry
                    .last_turn_at_wall
                    .read()
                    .ok()
                    .and_then(|guard| guard.map(|ts| ts.to_rfc3339()))
            });
            // Recover capabilities from the persisted JSON column for evicted rows.
            let capabilities = live
                .map(|entry| entry.capabilities)
                .unwrap_or_else(|| capabilities_from_row(row.capabilities_json.as_deref()));
            let turn_in_flight = live.is_some_and(|entry| {
                entry.in_flight.load(std::sync::atomic::Ordering::Acquire) > 0
            });
            SessionResponse {
                session,
                last_turn_at,
                capabilities,
                turn_in_flight,
            }
        })
        .collect();
    drop(in_memory);

    Ok(Json(ListSessionsResponse {
        sessions,
        next_cursor,
    }))
}

/// GET /v1/sessions/{id}: single session metadata.
#[utoipa::path(
    get,
    path = "/v1/sessions/{id}",
    tag = "sessions",
    params(("id" = Uuid, Path, description = "Session UUID")),
    responses(
        (status = 200, description = "Session record", body = SessionResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:r"]))
)]
pub(crate) async fn get_session(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SessionsRead>,
    Path(id): Path<Uuid>,
) -> Result<Json<SessionResponse>, ProblemDetail> {
    if let Some(entry) = state.sessions.read().await.get(&id).cloned() {
        let updated_at = entry
            .updated_at
            .read()
            .ok()
            .map(|guard| guard.to_rfc3339())
            .unwrap_or_default();
        // The row, read rather than tolerated. `profile` is the billing record and `""` is not a
        // profile, so a read that fails is a 500 rather than a 200 nobody can act on; a resident
        // entry whose row is gone is a session that is not there.
        let info = state
            .shared
            .store
            .session_info(id)
            .await
            .map_err(|error| {
                ProblemDetail::internal_sanitized("failed to look up session", error)
                    .with("session_id", id.to_string())
            })?
            .ok_or_else(|| session_not_found(id))?;
        let last_turn_at = entry
            .last_turn_at_wall
            .read()
            .ok()
            .and_then(|guard| guard.map(|ts| ts.to_rfc3339()));
        return Ok(Json(SessionResponse {
            session: SessionView {
                id: entry.id,
                created_at: entry.created_at.to_rfc3339(),
                updated_at,
                cwd: Some(entry.cells().cwd.get()),
                permission: Some(entry.cells().permission.get()),
                approvals: entry.cells().permission.approvals(),
                profile: info.profile,
                title: info.title,
                parent_id: info.parent_id,
            },
            last_turn_at,
            capabilities: entry.capabilities,
            turn_in_flight: entry.in_flight.load(std::sync::atomic::Ordering::Acquire) > 0,
        }));
    }
    let summary = state
        .shared
        .store
        .session_info(id)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to look up session", error))?
        .ok_or_else(|| session_not_found(id))?;
    // Evicted-but-persisted row: fall back to DB columns for permission/capabilities.
    let capabilities = capabilities_from_row(summary.capabilities_json.as_deref());
    Ok(Json(SessionResponse {
        session: SessionView::from(&summary),
        last_turn_at: None,
        capabilities,
        turn_in_flight: false,
    }))
}

/// Move a session that is not resident onto another profile, without building an agent
/// for it.
///
/// The rescue path for a session whose recorded profile is no longer configured. Nothing here needs
/// the agent: the row is the thing being changed, there is no in-flight turn to conflict with and
/// no in-memory mirror to update, and the response comes off the row the way `GET` already answers
/// for an evicted session. The profile is still resolved first, so a body naming one that cannot
/// produce a profile is refused before the row moves, exactly as on the resident path.
///
/// `Ok(None)` means the session became resident after all, and the caller must take the resident
/// path so the agent moves with the row. Everything below rests on "not resident", and the check
/// that established it is several awaits behind: resolving a profile can build a profile and load
/// a credential, and any turn, scheduler fire, compaction or rewind arriving meanwhile rebuilds the
/// agent from the *old* profile and inserts it. Writing the row then left the session running,
/// billing and gauging the profile it had left, with `GET /v1/sessions` reporting the new one, for
/// as long as it stayed resident. Holding the reconstruction lock for the whole body is what makes
/// the re-check below conclusive rather than another sample.
///
/// The reconstruction lock answers for *this* process only, so the session lock is taken too. Every
/// other writer of `sessions.profile` already holds it -- a CLI resume, `/profile`, ACP's
/// `session/set_config_option`, and this handler's own resident path, which is resident here and so
/// holds it through the entry -- which makes "you may move a session's profile only while you own
/// the session" an invariant rather than a coincidence. Without it, a second `meka serve` on the
/// same store answered `200` for a session the first one was running: the row moved, `GET
/// /v1/sessions` on *both* reported the new profile, and the host actually holding the session went
/// on building requests for the old one until eviction. Reproduced over HTTP against two servers.
async fn repin_dormant_session(
    state: &ServerState,
    id: Uuid,
    profile: &str,
) -> Result<Option<Json<SessionResponse>>, ProblemDetail> {
    let _reconstruction = state.reconstruction_locks.lock(id).await;
    if state.sessions.read().await.contains_key(&id) {
        return Ok(None);
    }
    require_session_exists(state, id).await?;
    // `session-locked`, the same code `ensure_session_loaded` answers a cross-process conflict
    // with, and for the same reason: this is another process owning the session, not an in-process
    // turn. Held for the rest of the body, so nothing can attach between here and the write.
    //
    // Only a held lock, though. The lock directory failing to open is the operator's fault and
    // names their path, so it goes to the log as a 500 rather than to the caller as a conflict a
    // retry would never resolve.
    let _session = state
        .shared
        .store
        .lock_session(id)
        .map_err(|error| match error {
            crate::error::MekaError::SessionLocked(_) => ProblemDetail::new(
                ErrorKind::SessionLocked,
                StatusCode::CONFLICT,
                "another meka process is running this session, so its profile cannot be changed \
             from here; move it where it is running, or stop that process",
            )
            .with("session_id", id.to_string()),
            other => ProblemDetail::internal_sanitized("failed to lock session for repin", other)
                .with("session_id", id.to_string()),
        })?;
    crate::config::require_profile(profile, &state.shared.config.profiles)
        .map_err(|error| ProblemDetail::for_error(&error, state.config.relay_provider_errors))?;
    let resolved = crate::provider::resolved_profile(
        &state.shared.providers,
        profile.to_string(),
    )
    .await
    // Discriminated, not a blanket 422; see the sibling site in `patch_session`.
    .map_err(|error| {
        agent_build_problem(
            id,
            &format!("failed to resolve profile '{profile}'"),
            error,
        )
    })?;

    // Skipped when nothing changes, so a no-op PATCH does not advance `updated_at` that clients
    // watch for changes. Nothing to reconcile beyond the name: a dormant session has no live agent
    // to put back in step, which is what the resident path has to do unconditionally. A failed
    // write fails the request, because the row is the billing record.
    crate::host::record_profile_switch(&state.shared.store, id, &resolved)
        .await
        .map_err(|error| {
            agent_build_problem(id, "failed to record the session's profile", error)
        })?;

    // Re-read rather than patching the pre-write copy, so `updated_at` and `profile` are what the
    // store now holds.
    let summary = state
        .shared
        .store
        .session_info(id)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to look up session", error))?
        .ok_or_else(|| session_not_found(id))?;
    Ok(Some(Json(SessionResponse {
        session: SessionView::from(&summary),
        last_turn_at: None,
        capabilities: capabilities_from_row(summary.capabilities_json.as_deref()),
        // Not resident, and nothing could have made it so while the reconstruction lock was held.
        turn_in_flight: false,
    })))
}

/// PATCH /v1/sessions/{id}: update mutable session knobs (permission, approvals, cwd, profile) on
/// a live session without re-creating it. Returns the updated metadata.
///
/// Permission and cwd are hoisted on [`SessionEntry`] outside the runtime mutex precisely so the
/// PATCH handler can apply them without contending with a long-running turn; the change is
/// visible to the next agent operation that reads the cells.
#[utoipa::path(
    patch,
    path = "/v1/sessions/{id}",
    tag = "sessions",
    params(("id" = Uuid, Path, description = "Session UUID")),
    request_body = PatchSessionRequest,
    responses(
        (status = 200, description = "Updated session record", body = SessionResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 409, description = "A turn is in flight; cancel first (`/errors/turn-in-flight`). Or another meka process holds the session, so only that process may move it (`/errors/session-locked`)", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid body, or the id names a sub-agent's conversation, whose permission, cwd and profile come from the terms its parent spawned it with (`/errors/session-not-drivable`)", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:w"]))
)]
pub(crate) async fn patch_session(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SessionsWrite>,
    Path(id): Path<Uuid>,
    raw_body: Bytes,
) -> Result<Json<SessionResponse>, ProblemDetail> {
    let body: PatchSessionRequest = serde_json::from_slice(&raw_body)
        .map_err(|error| ProblemDetail::invalid_body("session patch", error))?;

    // Ahead of the dormant fast path below, which is the one branch of this handler that writes a
    // session row without ever building an agent -- so it is the one branch
    // `crate::host::refuse_a_spawned_session` cannot answer for from inside the builders. A
    // sub-agent is almost never resident (`build_subagent` runs it under its parent's runtime
    // rather than registering it here), so that branch was where every `PATCH` aimed at a
    // sub-agent actually landed: it wrote `sessions.profile` on a sub-agent's row and answered
    // `200`, while the same request against a *resident* session was refused. The write did not
    // even survive -- the next `agent_followup` records the profile it really ran on, which is
    // the parent's -- so what a `sessions:w` holder got was a row that disagreed with the
    // sub-agent until something quietly put it back.
    crate::host::refuse_a_spawned_session(&state.shared.store, Some(id))
        .await
        .map_err(|error| agent_build_problem(id, "failed to read session", error))?;

    // A body that names only a profile is the documented way to move a session whose recorded
    // profile has left `config.toml`, and reviving it to apply the change is the one thing that
    // cannot work in that state: `ensure_session_loaded` rebuilds the agent, which resolves the
    // very profile that is gone, so the rescue was refused by the failure it was meant to
    // repair. A session already resident falls through to the path below, because
    // `ensure_session_loaded` returns it without rebuilding and its live agent has to move with
    // the row.
    if body.permission.is_none()
        && body.approvals.is_none()
        && body.cwd.is_none()
        && let Some(name) = body.profile.as_deref()
        && !state.sessions.read().await.contains_key(&id)
        && let Some(response) = repin_dormant_session(&state, id, name).await?
    {
        return Ok(response);
    }

    let entry = ensure_session_loaded(&state, id).await?;

    // Reject PATCH while a turn is in-flight: the agent snapshots cwd/permission at turn
    // start, but tools read them live, creating a split-brain within one iteration.
    //
    //
    // A read and not an `InFlightGuard`, deliberately. That guard means "this session is busy with
    // turn-like work", and claiming it here would make two concurrent PATCHes conflict when they
    // should simply serialize -- metadata edits are not turns, and a client that sends two is not
    // doing anything wrong. The cost is that a turn admitted between this load and the agent swap
    // below makes that swap wait for it, which is slow rather than wrong: the row has already
    // moved, and the swap lands correctly afterwards.
    if entry.in_flight.load(std::sync::atomic::Ordering::Acquire) > 0 {
        return Err(turn_in_flight_conflict(id, "patch the session"));
    }

    // Validate all fields up-front before any DB write so a mixed valid/invalid request
    // (e.g. valid permission + invalid cwd) doesn't leave a half-applied state.
    let new_permission = match body.permission.as_deref() {
        Some(level) => {
            let parsed: Permission = level.parse().map_err(|error| {
                ProblemDetail::new(
                    ErrorKind::InvalidBody,
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!("invalid `permission` value: {error}"),
                )
            })?;
            let enabled = state.shared.config.enabled_permissions;
            if !enabled.is_enabled(parsed) {
                return Err(ProblemDetail::for_error(
                    &enabled.disabled_level(parsed),
                    state.config.relay_provider_errors,
                ));
            }
            Some(parsed)
        }
        None => None,
    };
    let new_cwd = match body.cwd.clone() {
        Some(path) => Some(accepted_cwd(&path)?),
        None => None,
    };
    // Built before the DB write for the same reason the other two are validated first: a profile
    // that cannot produce a profile must not reach the row, and building it here means the 422
    // covers a missing credential as well as a missing profile.
    let new_profile = match body.profile.as_deref() {
        Some(name) => {
            crate::config::require_profile(name, &state.shared.config.profiles).map_err(
                |error| ProblemDetail::for_error(&error, state.config.relay_provider_errors),
            )?;
            // The profile as configured. A `PATCH` naming a profile moves the session to that
            // bundle entire, which is the only thing naming a profile can mean.
            Some(
                crate::provider::resolved_profile(
                    &state.shared.providers,
                    name.to_string(),
                )
                .await
                // Not a blanket 422: `resolved_profile` reaches `profile_credential_version` and
                // `load_profile_credential`, so a locked or unreadable store arrives here as
                // `MekaError::Database`, and formatting it into the body answered "your request is
                // invalid" with an internal message attached. `agent_build_problem` is the
                // discriminator the rest of this surface already uses -- `Config` verbatim, because
                // naming the profile is the whole point, everything else sanitized into a 500.
                .map_err(|error| {
                    agent_build_problem(
                        id,
                        &format!("failed to resolve profile '{name}'"),
                        error,
                    )
                })?,
            )
        }
        None => None,
    };

    // Filter out no-op fields so a PATCH that doesn't change anything skips the DB write
    // and doesn't advance `updated_at` (used by clients for change detection).
    let permission_change: Option<Permission> =
        new_permission.filter(|parsed| entry.cells().permission.get() != *parsed);
    let approvals_change: Option<bool> = body
        .approvals
        .filter(|wanted| entry.cells().permission.approvals() != *wanted);
    let cwd_change: Option<std::path::PathBuf> =
        new_cwd.filter(|path| entry.cells().cwd.get() != *path);
    // Only whether the *row* already says this, because that is all this decides. The live agent
    // is moved further down regardless, so a `PATCH` naming the profile the row already records is
    // how a session whose agent and row have come apart is put back together. Gating the swap on
    // this too made that the one state no request could repair: the retry after a failed `PATCH`
    // found the row already correct, wrote nothing, swapped nothing, and left every turn running
    // on the profile the client had just been told it was off.
    let profile_row_write = match new_profile.as_ref() {
        Some(resolved) => {
            let current = state
                .shared
                .store
                .recorded_profile(id)
                .await
                .map_err(|error| {
                    ProblemDetail::internal_sanitized("failed to read session profile", error)
                })?;
            current.as_ref() != Some(&resolved.profile)
        }
        None => false,
    };
    let patch = crate::store::SessionPatch {
        permission: permission_change,
        approvals: approvals_change,
        cwd: cwd_change.clone(),
        profile: profile_row_write
            .then(|| {
                new_profile
                    .as_ref()
                    .map(|resolved| resolved.profile.clone())
            })
            .flatten(),
        roots: None,
    };
    let mutated = !patch.is_empty();
    if mutated {
        // One write for every column, and one policy for a write that fails: a profile that could
        // not be recorded fails the request, since the row is the billing record; the other three
        // are applied to the live cells below regardless and the failure is logged, because the
        // cells are what this session's next call reads. `record_session_change` decides which.
        crate::host::record_session_change(&state.shared.store, id, patch)
            .await
            .map_err(|error| {
                ProblemDetail::internal_sanitized("failed to record the session's profile", error)
            })?;
        // Apply the in-memory mirror. `try_set` re-validates against the enabled set as
        // belt-and-braces; a failure here would indicate a config reload race (not currently
        // supported) and is treated as a 500.
        if let Some(parsed) = permission_change {
            entry.cells().permission.try_set(parsed).map_err(|error| {
                ProblemDetail::new(
                    ErrorKind::Internal,
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("permission failed in-memory validation after the row moved: {error}"),
                )
            })?;
        }
        if let Some(approvals) = approvals_change {
            entry.cells().permission.set_approvals(approvals);
        }
        if let Some(path) = cwd_change {
            entry.cells().cwd.set(path);
        }
    }

    // The live agent moves last, and only after the row it must agree with. An `await` and not a
    // `try_lock`: the row has already moved, and a resident session's next turn runs on the agent
    // rather than re-reading the row, so refusing to swap would leave the two disagreeing until
    // eviction. See the in-flight note at the top of the handler.
    //
    // Outside the `mutated` block, because "the row already says this" is exactly when a repair is
    // wanted and never a reason to skip one. Re-publishing the profile the agent already runs on
    // costs one lock and changes nothing.
    if let Some(resolved) = new_profile {
        // One call, not three. The window and the vision flag the entry reports both come off the
        // cell `set_profile` publishes into, so moving the agent moves them.
        // Under the conversation lock, so the switch lands between turns: a turn is a conversation
        // with one profile, and moving the agent under a running one would hand the next round to
        // a different wire.
        let _turn_exclusive = entry.conversation.lock().await;
        entry.agent.set_provider(resolved);
    }

    // Bump `updated_at` only on actual changes; leave `last_turn_at` alone so the GC
    // scanner's idle timer tracks profile activity, not metadata edits.
    if mutated && let Ok(mut guard) = entry.updated_at.write() {
        *guard = chrono::Utc::now();
    }
    let cwd_snapshot = entry.cells().cwd.get();
    let updated_at = entry
        .updated_at
        .read()
        .ok()
        .map(|guard| guard.to_rfc3339())
        .unwrap_or_default();
    // Read, not tolerated, for the reason `get_session` gives: `""` is not a profile.
    let info = state
        .shared
        .store
        .session_info(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to look up session", error)
                .with("session_id", id.to_string())
        })?
        .ok_or_else(|| session_not_found(id))?;
    let last_turn_at = entry
        .last_turn_at_wall
        .read()
        .ok()
        .and_then(|guard| guard.map(|ts| ts.to_rfc3339()));
    Ok(Json(SessionResponse {
        session: SessionView {
            id: entry.id,
            created_at: entry.created_at.to_rfc3339(),
            updated_at,
            cwd: Some(cwd_snapshot),
            permission: Some(entry.cells().permission.get()),
            approvals: entry.cells().permission.approvals(),
            profile: info.profile,
            title: info.title,
            parent_id: info.parent_id,
        },
        last_turn_at,
        capabilities: entry.capabilities,
        turn_in_flight: entry.in_flight.load(std::sync::atomic::Ordering::Acquire) > 0,
    }))
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct PatchSessionRequest {
    /// New permission level (`none` / `read` / `workspace` / `unrestricted`). Must be in the
    /// server's enabled set. Absent → keep current.
    #[serde(default)]
    pub(crate) permission: Option<String>,
    /// Whether calls above the level are submitted for approval. Absent → keep current.
    #[serde(default)]
    pub(crate) approvals: Option<bool>,
    /// New working directory. Must be absolute. Absent → keep current.
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub(crate) cwd: Option<std::path::PathBuf>,
    /// Profile to move the session onto. Must name a profile in `config.toml`. Absent → keep
    /// current.
    ///
    /// Switching mid-conversation is allowed and is the client's call. Thinking blocks are tagged
    /// with the profile that produced them and are not replayed to a different one, so the
    /// reasoning recorded so far stops being visible to the model from the next turn onward.
    #[serde(default)]
    pub(crate) profile: Option<String>,
}

/// The 409 a mutating session operation gets when it races a turn, from the one mapping every
/// refusal goes through. `doing` names what was refused; the variant carries no id, so it is
/// attached here. `relay_provider_errors` is `false` because there is no upstream body to decide
/// about.
pub(crate) fn turn_in_flight_conflict(id: Uuid, doing: &'static str) -> ProblemDetail {
    ProblemDetail::for_error(&crate::error::MekaError::TurnInFlight { doing }, false)
        .with("session_id", id.to_string())
}

/// DELETE /v1/sessions/{id}: drop the in-memory entry and (optionally) the DB row.
#[utoipa::path(
    delete,
    path = "/v1/sessions/{id}",
    tag = "sessions",
    params(("id" = Uuid, Path, description = "Session UUID")),
    responses(
        (status = 204, description = "Session deleted (idempotent, also returned for unknown ids)"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 409, description = "A turn is in flight; cancel first (`/errors/turn-in-flight`). Or another meka process holds the session (`/errors/session-locked`)", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:w"]))
)]
pub(crate) async fn delete_session(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SessionsWrite>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, ProblemDetail> {
    // Refuse DELETE while a turn is in flight: silently destroying agent work would surprise
    // callers. DB-delete runs BEFORE the in-memory remove so a transient DB failure leaves
    // the session usable (client can retry).
    {
        let map = state.sessions.read().await;
        if let Some(entry) = map.get(&id)
            && entry.in_flight.load(std::sync::atomic::Ordering::Acquire) > 0
        {
            return Err(turn_in_flight_conflict(id, "delete the session"));
        }
        let present_in_memory = map.contains_key(&id);
        drop(map);
        if !present_in_memory {
            let exists = state
                .shared
                .store
                .session_exists(id)
                .await
                .map_err(|error| {
                    ProblemDetail::internal_sanitized(
                        "failed to check session existence during delete",
                        error,
                    )
                })?;
            // Truly idempotent: return 204 even when the id is unknown.
            if !exists {
                return Ok(StatusCode::NO_CONTENT);
            }
        }
    }

    // The write lock covers the in-flight re-check and the map removal, and nothing else. Not
    // spanning the DB delete, which is a cascading `DELETE` plus a lock-directory sweep doing
    // blocking `read_dir` / `remove_file` on the connection thread; that thread is shared, so the
    // call can queue behind a large `GET /export` or `POST /import`. Held across that, and with
    // tokio's `RwLock` being write-preferring, a single DELETE stalls every reader in the process:
    // `POST /cancel`, `GET /stream`, the GC scan, the background poller.
    //
    // Shortening it is safe because the map lock is not what serializes this against a concurrent
    // re-attach: the removed entry still owns the session's cross-process `FileLock`, and holds it
    // until the end of this function. A request arriving in the gap finds no map entry, loads the
    // row, fails to take the file lock, and gets `session-locked` -- never a second live entry for
    // a session being deleted.
    //
    // The one case that argument does not cover is a session that was already evicted, where there
    // is no entry and so no lock to hold. A re-attach that has taken the file lock and passed its
    // own existence re-check could then insert an entry for a row this delete is about to remove.
    // The window is the few instructions between the two, and both sides funnel through the single
    // database connection, which orders the delete ahead of the re-check in practice.
    let removed = {
        let mut map = state.sessions.write().await;
        if let Some(entry) = map.get(&id)
            && entry.in_flight.load(std::sync::atomic::Ordering::Acquire) > 0
        {
            return Err(turn_in_flight_conflict(id, "delete the session"));
        }
        map.remove(&id)
    };

    // A failure here leaves the row in place with the entry already evicted, so the session is
    // still usable: the next request re-attaches it from the row it just failed to delete.
    //
    // Two doors, by who holds the session's file lock. A resident entry holds it through
    // `entry.session_lock`, so the locked door would refuse this process's own claim. A session
    // this process has not loaded may be open in a REPL or another server, and this is exactly the
    // caller `delete_session_unless_attached` exists for: taking the plain door cascaded the rows
    // away under a conversation that carried on until its next turn failed on a foreign key.
    let deleted = match removed.as_ref() {
        Some(_) => state.shared.store.delete_session(id).await,
        None => state.shared.store.delete_session_unless_attached(id).await,
    };
    if let Err(error) = deleted {
        // The entry is out of the map and about to drop, so its registry has to leave the MCP
        // manager here or it stays attached for the life of the process. Only the detach: the
        // session's background work is still the session's, and it re-attaches from the row this
        // delete failed to remove.
        if let (Some(entry), Some(manager)) = (removed.as_ref(), state.shared.mcp_manager.as_ref())
        {
            crate::tools::mcp_adapter::detach_session_registry(
                manager,
                entry.agent.tool_registry(),
            )
            .await;
        }
        return Err(match error {
            locked @ crate::error::MekaError::SessionLocked(_) => {
                ProblemDetail::for_error(&locked, false)
            }
            other => ProblemDetail::internal_sanitized("failed to delete session", other),
        });
    }

    if let Some(entry) = removed.as_ref() {
        // Signaled after the row is gone, not before: a failed DB delete leaves the session
        // usable, and killing its work first would make that rollback a lie.
        //
        // The in-flight check above only covers *turns*. A detached background task never sets
        // `in_flight`, so DELETE sails past one, and the cascade has just taken the
        // `background_tasks` rows with the session -- which is what makes this the last chance to
        // stop it. Without this the task and its whole process group run on with no row to find
        // them by, so `DELETE /v1/sessions/{id}/tasks/{task_id}` now 404s and only restarting the
        // server (which still would not reap the process group) ends it. The REPL does the same
        // thing on its way out.
        let signaled = entry.release(state.shared.mcp_manager.as_ref()).await;
        if signaled > 0 {
            tracing::info!(
                "deleting session {id} signaled {signaled} running background task(s) to stop"
            );
        }
    }
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    /// The dormant repin must hold both locks; see
    /// [`crate::host::http::reattach::assert_dormant_fast_path_is_serialized`] for why this is
    /// asserted against the source.
    ///
    /// What it defends here: the function decides what to write on the strength of the session
    /// *not* being resident, then awaits five times before writing. Without the reconstruction
    /// lock, any turn, scheduler fire, compaction or rewind arriving meanwhile rebuilds the agent
    /// from the *old* profile; the row then moves and the session runs, bills and gauges the
    /// profile it had left until it is evicted -- hours, at the default idle timeout.
    ///
    /// `tests/serve.rs`'s `a_dormant_repin_and_the_agent_rebuilt_after_it_agree` covers what is
    /// reachable -- that the PATCH takes the dormant path at all, and that the row and the agent
    /// rebuilt from it agree afterwards -- but not the serialization, which only shows up under a
    /// race it cannot create.
    #[test]
    fn the_dormant_repin_serializes_against_reconstruction() {
        crate::host::http::reattach::assert_dormant_fast_path_is_serialized(
            include_str!("sessions.rs"),
            "async fn repin_dormant_session(",
            "session_info(id)",
            "record_profile_switch(",
        );
    }

    /// Both rollbacks delete through the guarded door. What that defends needs a second process:
    /// the row's lock has gone by the time either runs, and `meka -c` there can take the newest
    /// root in the gap, after which the plain door cascaded the row away under a session that
    /// process was running. Asserted against the source for the reason the test above is; the
    /// guarded door's own refusal is `deleting_a_session_another_process_holds_is_refused`.
    #[test]
    fn a_failed_build_rolls_its_row_back_through_the_guarded_door() {
        let source = include_str!("sessions.rs").replace("\r\n", "\n");
        for (signature, sentinel) in [
            (
                "impl Drop for SessionRollback {",
                "session rollback: deleted orphan row",
            ),
            ("async fn roll_back_fork(", "failed to roll back fork"),
        ] {
            let body = source
                .split(signature)
                .nth(1)
                .expect("the rollback this assertion is about")
                .split("\n}\n")
                .next()
                .expect("splitting always yields a first part");
            assert!(
                body.contains(sentinel),
                "the scanned region no longer covers {signature}, so this assertion proves nothing"
            );
            assert!(
                body.contains("delete_session_unless_attached("),
                "{signature} must delete through the guarded door"
            );
            assert!(
                !body.contains(".delete_session("),
                "{signature} must not take the plain door, which deletes under whoever holds the row"
            );
        }
    }
}
