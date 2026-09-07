//! The on-disk Markdown stores meka owns: skills and memories.
//!
//! Both are process-wide rather than per-session, which is why they carry their own scopes
//! (`skills:r` / `skills:w`, `memory:r` / `memory:w`) instead of riding on `sessions:*`. A bridge
//! token that should be able to run turns has no business emptying the memory store.
//!
//! These endpoints are *not* gated by `[skills] agent_managed` or `[memory] access`. Those flags
//! describe what the model may do on its own initiative; a token presented to this API is the
//! operator acting remotely, and is the wire equivalent of running `meka skill add` in a shell.

use axum::{
    Json,
    body::Bytes,
    extract::{Path as AxumPath, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    host::http::{
        errors::{ErrorKind, ProblemDetail},
        scope::{self},
        state::ServerState,
    },
    view::{MemoryDetail, ProfileView, SkillDetail, ToolView},
};

/// Turn a store-layer `Err(String)` into a 422. Every one of them is a statement about the caller's
/// input (bad name, empty description, a symlink in the way), not a server fault.
fn store_error(message: String) -> ProblemDetail {
    ProblemDetail::new(
        ErrorKind::InvalidBody,
        StatusCode::UNPROCESSABLE_ENTITY,
        message,
    )
}

fn not_found(noun: &str, name: &str) -> ProblemDetail {
    ProblemDetail::new(
        ErrorKind::NotFound,
        StatusCode::NOT_FOUND,
        format!("no {noun} named '{name}'"),
    )
}

/// The store is switched off, or meka failed to resolve its config directory.
///
/// `not-found` rather than `invalid-body`: nothing is wrong with the request, there is simply
/// nowhere to write. A client switching on `type` would read `invalid-body` as "fix your JSON" and
/// retry forever against a store that does not exist.
fn store_disabled(noun: &str) -> ProblemDetail {
    ProblemDetail::new(
        ErrorKind::NotFound,
        StatusCode::NOT_FOUND,
        format!("{noun} is disabled, or meka failed to resolve its config directory"),
    )
}

// ---------------------------------------------------------------------------
// Skills
// ---------------------------------------------------------------------------

/// `GET /v1/skills/{name}`: one skill, body included.
#[utoipa::path(
    get,
    path = "/v1/skills/{name}",
    tag = "skills",
    params(("name" = String, Path, description = "Skill name")),
    responses(
        (status = 200, description = "Skill with its body", body = SkillDetail),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such skill", body = ProblemDetail),
        (status = 422, description = "The skill exists but its SKILL.md does not parse", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["skills:r"]))
)]
pub(crate) async fn get_skill(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SkillsRead>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<SkillDetail>, ProblemDetail> {
    // `skills:r`, not any read scope: the palette at `GET /v1/skills` is a listing, but this
    // returns the body, which is the instruction text itself.
    let installed = state.shared.skills.current().await;
    let skill = installed.find(&name).ok_or_else(|| {
        // The distinction `get_memory` already draws: a file that failed to parse is a different
        // answer from one that never existed, and the reason is what the operator needs to fix it.
        match installed.skip_reason(&name) {
            Some(_) => store_error(installed.unavailable(&name)),
            None => not_found("skill", &name),
        }
    })?;
    // `load_skill_source`, not `load_skill_body`: the latter is what the *agent* is handed, with a
    // base-directory header prepended. Returning that here would make `GET` and `PUT` disagree, and
    // an editing client's read-modify-write would bake the header into the file, once per cycle.
    let body = crate::skills::load_skill_source(skill)
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to read skill body", error))?;
    Ok(Json(SkillDetail::new(skill, Some(body))))
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct WriteSkillRequest {
    /// One line saying when to reach for this skill. Required; an empty description produces a
    /// skill that cannot be loaded again.
    pub(crate) description: String,
    /// 0..=9, lower listed first. Omit to keep an existing skill's priority; defaults to 5 on
    /// creation.
    #[serde(default)]
    pub(crate) priority: Option<u8>,
    /// The `SKILL.md` body. Omit to keep the existing body when updating; send `""` to clear it.
    /// The same semantics as `memory_write` and `skill_write`, which exist so a caller correcting
    /// a description does not have to resend a body it never wanted to touch.
    #[serde(default)]
    pub(crate) body: Option<String>,
    /// Attribution. Recorded on creation only; an existing skill keeps whatever author it already
    /// had, so editing a hand-written skill never reassigns it.
    #[serde(default)]
    pub(crate) author: Option<String>,
}

/// `PUT /v1/skills/{name}`: create or update a skill.
#[utoipa::path(
    put,
    path = "/v1/skills/{name}",
    tag = "skills",
    params(("name" = String, Path, description = "Skill name, lowercase letters, digits, hyphens")),
    request_body = WriteSkillRequest,
    responses(
        (status = 200, description = "Skill written", body = SkillDetail),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "The skill store is disabled", body = ProblemDetail),
        (status = 409, description = "The skill lives in a read-only root from `[skills] extra_paths`", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid name, empty description, or an unwritable path", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["skills:w"]))
)]
pub(crate) async fn put_skill(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SkillsWrite>,
    AxumPath(name): AxumPath<String>,
    raw_body: Bytes,
) -> Result<Json<SkillDetail>, ProblemDetail> {
    let body: WriteSkillRequest = serde_json::from_slice(&raw_body)
        .map_err(|error| ProblemDetail::invalid_body("skill", error))?;
    let root = state
        .shared
        .skills
        .root()
        .ok_or_else(|| store_disabled("skills"))?
        .to_path_buf();

    if body
        .priority
        .is_some_and(|priority| priority > crate::entry::MAX_PRIORITY)
    {
        return Err(store_error(format!(
            "priority must be between {} and {}",
            crate::entry::MIN_PRIORITY,
            crate::entry::MAX_PRIORITY
        )));
    }
    let installed = state.shared.skills.current().await;
    let existing = installed.find(&name);

    // A skill discovered under a read-only `extra_paths` root is not this endpoint's to change: the
    // write would land in meka's own store and *shadow* it, so the caller would be told it updated
    // a skill while the file every other client reads stayed as it was. Unlike the module note
    // above about an operator's deliberate choices, nobody chooses a silent fork.
    //
    // Shared with the CLI and the tools rather than spelled out here, which is also what makes it
    // cover a shadowed file that does not parse: a check against the loaded skills alone lets a
    // broken one through.
    if let Some(refusal) = crate::skills::refuse_foreign_write(&installed, &name, &root) {
        // Logged here rather than composed into the body: the caller holds a token, not the
        // machine, and the path is out of the operator's `config.toml`. See `ForeignSkill`.
        tracing::warn!(
            "refusing to write skill '{name}' through HTTP: it lives at {path}",
            path = refusal.source_dir.display()
        );
        return Err(ProblemDetail::new(
            ErrorKind::StoreReadOnly,
            StatusCode::CONFLICT,
            refusal.to_string(),
        ));
    }

    // Omitted means "leave it alone", the way `body` and `author` already do. Resetting to the
    // default instead would make the obvious edit -- `GET`, change the text, `PUT` it back --
    // silently demote a prioritized skill, and priority both orders the `[Skills]` index the model
    // reads and decides which entries the index cap drops. Defaults only on creation.
    let priority = match body.priority {
        Some(priority) => priority,
        None => existing.map_or(crate::entry::DEFAULT_PRIORITY, |skill| skill.priority),
    };
    let description = crate::entry::normalize_description(&body.description);
    // On the blocking pool, like every other caller of this function. It takes a cross-process
    // `flock` and then `fsync`s, so on a runtime worker it parks one that cannot poll anything
    // else meanwhile, and the `flock` has no bound: a `meka skill add --edit` with an editor open
    // holds it for as long as the editor lives. Enough concurrent requests parked that way and the
    // router itself stops answering, health check included.
    {
        let root = root.clone();
        let name = name.clone();
        let description = description.clone();
        let author = body.author.clone();
        let skill_body = body.body.clone();
        tokio::task::spawn_blocking(move || {
            crate::skills::write_skill(
                &root,
                &name,
                &description,
                priority,
                author.as_deref(),
                skill_body.as_deref(),
            )
        })
        .await
        .map_err(|error| ProblemDetail::internal_sanitized("failed to write the skill", error))?
        .map_err(store_error)?;
    }
    // Before the read-back below, which goes through the same cache: without this a second write
    // inside one mtime tick that renders to the same length is invisible, and this response would
    // echo the *previous* values as though they had been applied.
    state.shared.skills.invalidate().await;
    tracing::info!("wrote skill '{name}' via HTTP");

    // Read back through the cache rather than echoing the request: the response then reports what
    // is actually on disk, including the `author` a pre-existing skill kept.
    let installed = state.shared.skills.current().await;
    let skill = installed.find(&name).ok_or_else(|| {
        ProblemDetail::internal_sanitized(
            "skill vanished between write and read-back",
            format!("skill '{name}' is not in the cache after a successful write"),
        )
    })?;
    Ok(Json(SkillDetail::new(skill, None)))
}

/// `DELETE /v1/skills/{name}`: remove a skill and everything bundled with it.
#[utoipa::path(
    delete,
    path = "/v1/skills/{name}",
    tag = "skills",
    params(("name" = String, Path, description = "Skill name")),
    responses(
        (status = 204, description = "Skill removed"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such skill", body = ProblemDetail),
        (status = 409, description = "The skill lives in a read-only root from `[skills] extra_paths`", body = ProblemDetail),
        (status = 422, description = "Invalid name, or a symlinked skill directory", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["skills:w"]))
)]
pub(crate) async fn delete_skill(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SkillsWrite>,
    AxumPath(name): AxumPath<String>,
) -> Result<StatusCode, ProblemDetail> {
    let root = state
        .shared
        .skills
        .root()
        .ok_or_else(|| store_disabled("skills"))?
        .to_path_buf();
    // Validated before the probe, the way `delete_memory` does. Probing first answered "does this
    // directory exist" for any string a caller sent, including one the name rules would have
    // refused, which is a filesystem oracle reachable with only `skills:w`.
    crate::skills::validate_addressable_name(&name).map_err(store_error)?;
    // The same refusal `PUT` gives, rather than the 404 the path check below would produce: `GET
    // /v1/skills` lists this skill, so telling a caller it does not exist is a story the listing
    // contradicts.
    let installed = state.shared.skills.current().await;
    if let Some(refusal) = crate::skills::refuse_foreign_delete(&installed, &name, &root) {
        tracing::warn!(
            "refusing to delete skill '{name}' through HTTP: it lives at {path}",
            path = refusal.source_dir.display()
        );
        return Err(ProblemDetail::new(
            ErrorKind::StoreReadOnly,
            StatusCode::CONFLICT,
            refusal.to_string(),
        ));
    }
    // Classified on the path rather than on the error text, the way `delete_memory` does. The
    // store layer races its own existence check against `remove_dir_all`, whose ENOENT renders as
    // "No such file or directory" and would fall through a substring match for "not found" into a
    // 422 for what is plainly a 404.
    let missing = !root.join(&name).is_dir();
    // On the blocking pool, for the reason `put_skill` is: `delete_skill` takes the store's
    // cross-process `flock`, whose wait is unbounded.
    {
        let root = root.clone();
        let owned = name.clone();
        tokio::task::spawn_blocking(move || crate::skills::delete_skill(&root, &owned))
            .await
            .map_err(|error| {
                ProblemDetail::internal_sanitized("failed to delete the skill", error)
            })?
            .map_err(|message| {
                if missing {
                    not_found("skill", &name)
                } else {
                    store_error(message)
                }
            })?;
    }
    state.shared.skills.invalidate().await;
    tracing::info!("deleted skill '{name}' via HTTP");
    Ok(StatusCode::NO_CONTENT)
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct MemoryListResponse {
    pub(crate) memories: Vec<MemoryDetail>,
}

/// Turn a store error into a 500 that says which call failed without leaking the query.
fn memory_failure(error: crate::error::MekaError) -> ProblemDetail {
    ProblemDetail::internal_sanitized("memory store unavailable", error)
}

/// `GET /v1/memory`: the memory index, most important first.
#[utoipa::path(
    get,
    path = "/v1/memory",
    tag = "memory",
    responses(
        (status = 200, description = "Memory index", body = MemoryListResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["memory:r"]))
)]
pub(crate) async fn list_memories(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::MemoryRead>,
) -> Result<Json<MemoryListResponse>, ProblemDetail> {
    let index = state
        .shared
        .memories
        .index()
        .await
        .map_err(memory_failure)?;
    Ok(Json(MemoryListResponse {
        memories: index
            .iter()
            .map(|memory| MemoryDetail::new(memory, None))
            .collect(),
    }))
}

/// `GET /v1/memory/{name}`: one memory, body included.
#[utoipa::path(
    get,
    path = "/v1/memory/{name}",
    tag = "memory",
    params(("name" = String, Path, description = "Memory name")),
    responses(
        (status = 200, description = "Memory with its body", body = MemoryDetail),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such memory", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["memory:r"]))
)]
pub(crate) async fn get_memory(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::MemoryRead>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<MemoryDetail>, ProblemDetail> {
    // Deliberately not `record_read`: an operator reading through the HTTP API is not the agent
    // recalling anything, and moving the ranking the agent gets would be a lie about its own use.
    let memory = state
        .shared
        .memories
        .get(&name)
        .await
        .map_err(memory_failure)?
        .ok_or_else(|| not_found("memory", &name))?;
    let body = memory.body.clone().unwrap_or_default();
    Ok(Json(MemoryDetail::new(&memory, Some(body))))
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct WriteMemoryRequest {
    /// One line saying what this memory is for.
    pub(crate) description: String,
    /// 0..=9, lower first. Omit to keep an existing memory's priority; defaults to 5 on creation.
    #[serde(default)]
    pub(crate) priority: Option<u8>,
    /// Lowercase labels (`[a-z0-9-]`, at most 10). Omit to keep an existing memory's tags; send
    /// `[]` to clear them.
    #[serde(default)]
    pub(crate) tags: Option<Vec<String>>,
    /// Omit to keep the existing body when updating; send `""` to clear it.
    #[serde(default)]
    pub(crate) body: Option<String>,
}

/// `PUT /v1/memory/{name}`: create or update a memory.
#[utoipa::path(
    put,
    path = "/v1/memory/{name}",
    tag = "memory",
    params(("name" = String, Path, description = "Memory name, [A-Za-z0-9_-]")),
    request_body = WriteMemoryRequest,
    responses(
        (status = 200, description = "Memory written", body = MemoryDetail),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid name, empty description, or an unwritable path", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["memory:w"]))
)]
pub(crate) async fn put_memory(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::MemoryWrite>,
    AxumPath(name): AxumPath<String>,
    raw_body: Bytes,
) -> Result<Json<MemoryDetail>, ProblemDetail> {
    let body: WriteMemoryRequest = serde_json::from_slice(&raw_body)
        .map_err(|error| ProblemDetail::invalid_body("memory", error))?;
    crate::memory::validate_memory_name(&name).map_err(store_error)?;
    if !crate::memory::description_says_something(&body.description) {
        return Err(store_error("description cannot be empty".to_string()));
    }
    if body
        .priority
        .is_some_and(|priority| priority > crate::entry::MAX_PRIORITY)
    {
        return Err(store_error(format!(
            "priority must be between {} and {}",
            crate::entry::MIN_PRIORITY,
            crate::entry::MAX_PRIORITY
        )));
    }
    let tags = match &body.tags {
        Some(tags) => Some(crate::memory::normalize_tags(tags).map_err(store_error)?),
        None => None,
    };

    // One upsert, one transaction, and no lock. No `flock` and no blocking pool: the write is a
    // single statement rather than a read-modify-write over a file, which is what would make two
    // overlapping PUTs to one name last-writer-wins over a stale read. Omit-to-keep now lives in
    // the SQL, so there is no read to go stale and SQLite serializes the writers.
    let written = state
        .shared
        .memories
        .write(crate::store::memory::WriteRequest {
            name: name.clone(),
            description: Some(crate::entry::normalize_description(&body.description)),
            tags,
            body: body.body.clone(),
            priority: body.priority,
        })
        .await
        .map_err(memory_failure)?;
    tracing::info!("wrote memory '{name}' via HTTP");
    Ok(Json(MemoryDetail::new(&written, None)))
}

/// `DELETE /v1/memory/{name}`: remove a memory.
#[utoipa::path(
    delete,
    path = "/v1/memory/{name}",
    tag = "memory",
    params(("name" = String, Path, description = "Memory name")),
    responses(
        (status = 204, description = "Memory removed"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No such memory", body = ProblemDetail),
        (status = 422, description = "Invalid name, or a symlinked path", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["memory:w"]))
)]
pub(crate) async fn delete_memory(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::MemoryWrite>,
    AxumPath(name): AxumPath<String>,
) -> Result<StatusCode, ProblemDetail> {
    crate::memory::validate_memory_lookup(&name).map_err(store_error)?;
    // 404 from `rows_affected`, not from a pre-check. As separate statements a name can stop
    // existing between them, and this endpoint then answers 422 `invalid-body` for something
    // already gone, which a client switching on `type` reads as "fix your request".
    if !state
        .shared
        .memories
        .delete(&name)
        .await
        .map_err(memory_failure)?
    {
        return Err(not_found("memory", &name));
    }
    tracing::info!("deleted memory '{name}' via HTTP");
    Ok(StatusCode::NO_CONTENT)
}

// ---------------------------------------------------------------------------
// Tools, instructions, profiles
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ToolsResponse {
    pub(crate) tools: Vec<ToolView>,
}

/// `GET /v1/sessions/{id}/tools`: what this session's agent can call.
///
/// Per-session rather than server-global because the catalog includes MCP tools and reflects that
/// registry's deferred / loaded state, neither of which is a process-wide fact.
#[utoipa::path(
    get,
    path = "/v1/sessions/{id}/tools",
    tag = "discovery",
    params(("id" = uuid::Uuid, Path, description = "Session UUID")),
    responses(
        (status = 200, description = "Tool catalog", body = ToolsResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 409, description = "Session is not loaded; run a turn first", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:r"]))
)]
pub(crate) async fn list_tools(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SessionsRead>,
    AxumPath(id): AxumPath<uuid::Uuid>,
) -> Result<Json<ToolsResponse>, ProblemDetail> {
    crate::host::http::reattach::require_session_exists(&state, id).await?;
    // Deliberately does NOT revive: re-attaching takes the session's cross-process file lock for
    // up to `idle_timeout`, and a read scope must not be able to seize a write-exclusive resource.
    // See the same note on `GET /context`. A catalog needs a live registry, so an evicted session
    // gets a 409 rather than a guess assembled from process-wide defaults.
    let entry = state
        .sessions
        .read()
        .await
        .get(&id)
        .cloned()
        .ok_or_else(|| {
            ProblemDetail::new(
                ErrorKind::SessionNotLoaded,
                StatusCode::CONFLICT,
                "session is not loaded; submit a turn to load it",
            )
            .with("session_id", id.to_string())
        })?;
    // The hoisted clone, not `runtime.tool_registry`: a streaming turn holds the runtime mutex from
    // admission to completion, so reading through it would make this hang for minutes on exactly
    // the sessions a client is most likely to ask about. The registry is `Arc`-backed, so this
    // observes the same one the agent dispatches through, live MCP updates included.
    let catalog = entry.tool_registry().tool_catalog();
    Ok(Json(ToolsResponse {
        tools: catalog
            .into_iter()
            .map(|(name, description, permission, deferred)| {
                ToolView::new(name, description, permission, deferred)
            })
            .collect(),
    }))
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct InstructionsResponse {
    /// Where the instructions came from, for an operator diagnosing "why is it behaving like
    /// that". Absent when none are configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) source: Option<String>,
    /// The resolved instruction text the agent runs under, omitted when there is none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) content: Option<String>,
}

/// `GET /v1/instructions`: the standing instructions this server's agents run under.
#[utoipa::path(
    get,
    path = "/v1/instructions",
    tag = "discovery",
    responses(
        (status = 200, description = "Resolved standing instructions", body = InstructionsResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:r"]))
)]
pub(crate) async fn instructions(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SessionsRead>,
) -> Result<Json<InstructionsResponse>, ProblemDetail> {
    // `sessions:r` rather than any read scope: this returns the *whole* system-instruction text,
    // which is the same class of content as a skill body and is gated for the same reason (see
    // `get_skill`). A `schedule:r` token has no business reading the persona the agent runs under.
    // `sessions:r` is the right pairing because anyone who can read a session's messages can
    // already infer much of it.
    Ok(Json(InstructionsResponse {
        source: state.shared.user_instructions_source.clone(),
        content: state.shared.user_instructions.clone(),
    }))
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ProfilesResponse {
    pub(crate) profiles: Vec<ProfileView>,
}

/// `GET /v1/profiles`: configured profiles, with their account, backend and model only.
///
/// Read-only by design, and the distinction is which fact is being written. A session *may* name a
/// profile (`POST /v1/sessions`) and move to another (`PATCH /v1/sessions/{id}`), because that
/// records a choice on one session's own row where the user can see it. What no request may do is
/// edit the profiles themselves: what `work` means, and therefore which account it bills, comes
/// from `config.toml` and nothing else. Letting a request redefine a profile is how an ambient
/// value silently rebinds an account, which is the same reason there is no environment tier.
///
/// No credentials are returned: they live in the database, keyed by account name, and never
/// transit this API.
#[utoipa::path(
    get,
    path = "/v1/profiles",
    tag = "discovery",
    responses(
        (status = 200, description = "Configured profiles", body = ProfilesResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:r", "mcp:r", "skills:r", "memory:r", "schedule:r"]))
)]
pub(crate) async fn profiles(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::AnyRead>,
) -> Result<Json<ProfilesResponse>, ProblemDetail> {
    let config = &state.shared.config;
    let active = config.default_profile.as_deref();
    let profiles: Vec<ProfileView> = config
        .profile_summaries
        .iter()
        .map(|summary| ProfileView::from_summary(summary, Some(summary.name.as_str()) == active))
        .collect();
    Ok(Json(ProfilesResponse { profiles }))
}
