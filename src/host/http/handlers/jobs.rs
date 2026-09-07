//! Scheduled jobs and background tasks: the two kinds of work that outlive the request that
//! started them.
//!
//! `meka serve` is already the durable host for both. It runs the scheduler over *every* job in
//! the database rather than only those belonging to an open conversation, because it can revive any
//! session on demand (`crate::host::http::schedule`), and it polls background-task outcomes on its
//! own timer. Until now it was also the only surface with no way to ask what it was about to run.
//!
//! Scheduled jobs are keyed by `schedule:r` / `schedule:w` rather than `sessions:*`: a job survives
//! the conversation that created it and fires unattended, so the ability to plant one is a
//! materially different grant from the ability to run a turn. Background tasks stay on `sessions:*`
//! because they live and die inside one session's runtime.

use axum::{
    Json,
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use crate::{
    host::http::{
        errors::{ErrorKind, ProblemDetail},
        reattach::require_session_exists,
        scope,
        state::ServerState,
    },
    permission::Permission,
    schedule::{Gate, GatePredicate, GateProbe, Schedule, ScheduledJob},
    store::background::TaskStatus,
    view::ScheduledJobView,
};

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ScheduledJobsResponse {
    pub(crate) jobs: Vec<ScheduledJobView>,
}

/// `GET /v1/schedule`: every scheduled job in the database, across all sessions.
#[utoipa::path(
    get,
    path = "/v1/schedule",
    tag = "schedule",
    responses(
        (status = 200, description = "All scheduled jobs", body = ScheduledJobsResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["schedule:r"]))
)]
pub(crate) async fn list_all(
    State(state): State<ServerState>,
    scope::Scoped { principal, .. }: scope::Scoped<scope::ScheduleRead>,
) -> Result<Json<ScheduledJobsResponse>, ProblemDetail> {
    let reveal_command = principal.has_scope("sessions:r");
    let jobs = state
        .shared
        .store
        .schedule_store()
        .list_all_scheduled_jobs()
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to list scheduled jobs", error)
        })?;
    Ok(Json(ScheduledJobsResponse {
        jobs: render_batch(&state, jobs, reveal_command).await,
    }))
}

/// `GET /v1/sessions/{id}/schedule`: jobs belonging to one session.
#[utoipa::path(
    get,
    path = "/v1/sessions/{id}/schedule",
    tag = "schedule",
    params(("id" = Uuid, Path, description = "Session UUID")),
    responses(
        (status = 200, description = "Scheduled jobs for this session", body = ScheduledJobsResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["schedule:r"]))
)]
pub(crate) async fn list_for_session(
    State(state): State<ServerState>,
    scope::Scoped { principal, .. }: scope::Scoped<scope::ScheduleRead>,
    Path(id): Path<Uuid>,
) -> Result<Json<ScheduledJobsResponse>, ProblemDetail> {
    let reveal_command = principal.has_scope("sessions:r");
    require_session_exists(&state, id).await?;
    let jobs = state
        .shared
        .store
        .schedule_store()
        .list_scheduled_jobs(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to list scheduled jobs", error)
                .with("session_id", id.to_string())
        })?;
    Ok(Json(ScheduledJobsResponse {
        jobs: render_batch(&state, jobs, reveal_command).await,
    }))
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CreateJobRequest {
    /// What the agent is asked to do when the job fires.
    pub(crate) prompt: String,
    /// One-shot time: an RFC 3339 instant, or a duration from now (`"20m"`, `"1h 30m"`).
    #[serde(default)]
    pub(crate) at: Option<String>,
    /// Recurring interval (`"30m"`, `"6h"`).
    #[serde(default)]
    pub(crate) every: Option<String>,
    /// 5-field cron pattern, evaluated in the host's local time.
    #[serde(default)]
    pub(crate) cron: Option<String>,
    /// Guard the job on a probe: a shell command, or a call to a read-only tool.
    ///
    /// The level required depends on which. A shell command runs unattended and with no sandbox,
    /// so it needs `unrestricted` and no level that promises a boundary can honestly authorize it.
    /// A tool probe needs only what the tool itself resolves to, which a gate requires to be
    /// `read`.
    #[serde(default)]
    pub(crate) gate: Option<CreateGate>,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CreateGate {
    /// What to run: `{"command": "…"}` for a shell command, or `{"tool": "mcp__server__tool",
    /// "arguments": {…}}` for a tool call.
    pub(crate) check: serde_json::Value,
    /// When to fire: `"changed"` (the default), `"succeeded"`, `{"matches": "<regex>"}`, or
    /// `{"at": "<json pointer>", "is": "not-empty" | "empty" | "changed"}`.
    ///
    /// Untyped here because one parser answers for this field on every door
    /// ([`crate::schedule::GatePredicate::parse_request`]); a second, derived shape would drift
    /// from the tool schema and give different errors for the same mistake.
    #[serde(default)]
    pub(crate) when: Option<serde_json::Value>,
}

/// `POST /v1/sessions/{id}/schedule`: plant a job on a session.
#[utoipa::path(
    post,
    path = "/v1/sessions/{id}/schedule",
    tag = "schedule",
    params(("id" = Uuid, Path, description = "Session UUID")),
    request_body = CreateJobRequest,
    responses(
        (status = 201, description = "Job created", body = ScheduledJobView),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope, or a session that cannot do unattended work", body = ProblemDetail),
        (status = 404, description = "Session not found, or scheduling is disabled on this server (`[schedule] enabled = false`), so there is nowhere for a job to go", body = ProblemDetail),
        (status = 413, description = "Request body exceeds `[serve] max_body_bytes`", body = ProblemDetail),
        (status = 422, description = "Invalid schedule, or the session's job limit is reached, or the id names a sub-agent's conversation, which nothing outside its parent can fire a job on (`/errors/session-not-drivable`)", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["schedule:w"]))
)]
pub(crate) async fn create(
    State(state): State<ServerState>,
    scope::Scoped { principal, .. }: scope::Scoped<scope::ScheduleWrite>,
    Path(id): Path<Uuid>,
    raw_body: Bytes,
) -> Result<(StatusCode, Json<ScheduledJobView>), ProblemDetail> {
    // Refused rather than accepted-and-ignored. With `[schedule] enabled = false` the scheduler
    // task is a no-op and the `schedule_*` tools are not registered at all, so this endpoint is the
    // only way a job can still be created -- and it would be created into a server that will never
    // run it, persisted, and listed forever with a `next_fire_at` receding into the past. Listing
    // and canceling stay open, because clearing out jobs left over from before the flag was
    // flipped is exactly what an operator wants then.
    //
    // `not-found`, as the disabled skill and memory stores answer: nothing is wrong with the
    // request, there is nowhere for the job to go, and a client reading `invalid-body` would
    // rewrite its payload forever.
    if !state.shared.config.schedule.enabled {
        return Err(ProblemDetail::new(
            ErrorKind::NotFound,
            StatusCode::NOT_FOUND,
            "scheduling is disabled on this server (`[schedule] enabled = false`), so a job \
             created here would never fire",
        )
        .with("session_id", id.to_string()));
    }

    let body: CreateJobRequest = serde_json::from_slice(&raw_body)
        .map_err(|error| ProblemDetail::invalid_body("schedule", error))?;
    if body.prompt.trim().is_empty() {
        return Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "`prompt` cannot be empty",
        ));
    }

    // Read, not revived. Reviving would take the session's cross-process file lock for up to
    // `idle_timeout` and hand a `schedule:w` token the same lock-pinning reach a read token was
    // just denied on `GET /context`.
    //
    // Two reads, not one: this answers whether the session exists and what level a gate would be
    // authorized against, and `spawn_terms` below answers whether a job may belong to it at all.
    // `SessionSummary` carries `parent_id` but not `subagent_spec_json`, and the parent link alone
    // is the wrong question -- an imported sub-agent has spawn terms and no parent. Widening the
    // summary to fold these back into one read would put the column on every listing that returns
    // one, for a check two handlers make.
    let summary = state
        .shared
        .store
        .session_info(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to look up session", error)
                .with("session_id", id.to_string())
        })?
        .ok_or_else(|| crate::host::http::reattach::session_not_found(id))?;

    let spawned = state.shared.store.spawn_terms(id).await.map_err(|error| {
        ProblemDetail::internal_sanitized("failed to read session spawn terms", error)
    })?;
    if let Some(refusal) = refuse_subagent_session(id, spawned) {
        return Err(refusal);
    }

    let resident = state.sessions.read().await.get(&id).cloned();

    let now = Utc::now();
    let schedule = parse_schedule(&body, now)?;

    // The session's level, for both checks below rather than only the gate's.
    //
    // The shared predicate rather than a comparison, exactly as `schedule_create` does.
    let permission = match &resident {
        Some(entry) => entry.cells().permission.get(),
        None => permission_from_summary(&state, id, Some(&summary)),
    };

    // Refused for an *ungated* job too, which is the case a gate-shaped check misses.
    //
    // The fire door declines every job on a session at `none`, so accepting one here creates a row
    // that can never run. This endpoint is the only door that could: `schedule_create` requires
    // `read` to dispatch at all, so the agent cannot reach it, and a token's scopes say nothing
    // about the session's level. A client that means to raise the session later can do so and then
    // create the job; one that does not now finds out at the point it can act on it, rather than by
    // noticing months later that nothing ever fired.
    if !permission.allows_unattended_work() {
        return Err(ProblemDetail::new(
            ErrorKind::SessionPermission,
            StatusCode::FORBIDDEN,
            format!(
                "this session is at {permission}, where no tool is executable, so a scheduled turn could \
                 neither act on the job nor cancel it. Raise the session with `PATCH \
                 /v1/sessions/{{id}}` first."
            ),
        )
        .with("session_id", id.to_string()));
    }

    let gate = match &body.gate {
        None => None,
        Some(requested) => {
            // `sessions:w` on top of `schedule:w`, because a gate is not really a scheduling
            // feature: the command runs on a timer, before the turn, as the server's user, and it
            // runs whether or not the turn works at all -- so a job whose gate is the payload
            // needs no provider, no credit and no model. `schedule:w` alone is meant to say "may
            // plant work on a session", and `GET /v1/schedule` already hands out every session id
            // in the database, so without this an operator who scoped a calendar bridge to
            // `schedule:*` and nothing else has in fact granted it unattended arbitrary shell.
            //
            // Requiring `sessions:w` puts gates in the tier that can already drive the agent, and
            // leaves the ordinary prompt-only job reachable by a schedule-only token.
            scope::require(&principal, "sessions:w").map_err(|_| {
                ProblemDetail::new(
                    ErrorKind::AuthScope,
                    StatusCode::FORBIDDEN,
                    "a `gate` runs a probe unattended, so it needs `sessions:w` as well as \
                     `schedule:w`. Create the job without `gate` and check the condition inside \
                     the prompt instead.",
                )
                .with("session_id", id.to_string())
            })?;
            let invalid = |message: String| {
                ProblemDetail::new(
                    ErrorKind::InvalidBody,
                    StatusCode::UNPROCESSABLE_ENTITY,
                    message,
                )
            };
            let probe = GateProbe::parse_request(Some(&requested.check)).map_err(invalid)?;
            let predicate =
                GatePredicate::parse_request(requested.when.as_ref()).map_err(invalid)?;

            // Authority is re-checked at fire time as well; this is the early, specific refusal so
            // a client learns at `POST` rather than from a job that silently never fires.
            if let Err(refusal) = crate::schedule::gate_probe_is_authorized(
                &probe,
                permission,
                state.shared.gate_tools.as_deref(),
            ) {
                // Routed by what would actually fix it, because that is what clients switch on.
                //
                // Never `AuthScope`: the token is fine and a better one will never help. Reporting
                // a scope failure sends a client off to re-provision a token it already holds.
                //
                // `SessionPermission` only where raising the session is the remedy the docs promise
                // for that type. A misspelled tool, or one that resolves above `read`, is a bad
                // request: no level and no token changes the answer, and `PATCH /v1/sessions/{id}`
                // is a wild goose chase. Those are `InvalidBody`, alongside the malformed-`check`
                // refusals the parser above already returns that way.
                let (kind, status) = match refusal {
                    crate::schedule::GateRefusal::ShellNeedsUnrestricted
                    | crate::schedule::GateRefusal::SessionBelowTool => {
                        (ErrorKind::SessionPermission, StatusCode::FORBIDDEN)
                    }
                    crate::schedule::GateRefusal::ToolUnavailable
                    | crate::schedule::GateRefusal::ToolNotReadOnly(_) => {
                        (ErrorKind::InvalidBody, StatusCode::UNPROCESSABLE_ENTITY)
                    }
                };
                // A server mid-handshake resolves nothing, and every tool it provides is
                // `ToolUnavailable` for the second that takes. The status stays 422 rather than
                // gaining a taxonomy entry for a sub-second window, but the message says so: a
                // client that retries once is right, and one that concludes the tool does not
                // exist is wrong.
                let transient = matches!(refusal, crate::schedule::GateRefusal::ToolUnavailable)
                    && state
                        .shared
                        .gate_tools
                        .as_ref()
                        .is_some_and(|tools| tools.is_still_connecting(&probe.summary()));
                let advice = if transient {
                    "Its MCP server has not finished connecting; retry shortly."
                } else {
                    "Create the job without `gate` and check the condition inside the prompt \
                     instead."
                };
                return Err(ProblemDetail::new(
                    kind,
                    status,
                    format!("{}. {}", refusal.explain(&probe, permission), advice),
                )
                .with("session_id", id.to_string()));
            }
            Some(Gate {
                probe,
                predicate,
                last_output: None,
                // See `schedule_create`: the level is recorded so `prepare` can re-check it at fire
                // time. The guard above has admitted this level for this probe, but the session's
                // is mutable through `PATCH /v1/sessions/{id}` and a tool's resolved level moves
                // with config, so the check above cannot stand in for one made when the probe
                // actually runs.
                permission,
            })
        }
    };

    let existing = state
        .shared
        .store
        .schedule_store()
        .list_scheduled_jobs(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to count scheduled jobs", error)
                .with("session_id", id.to_string())
        })?
        .len();
    let max_jobs = state.shared.config.schedule.max_jobs;
    if existing >= max_jobs {
        return Err(ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            format!(
                "session already has {existing} scheduled jobs (the configured limit). Cancel one first."
            ),
        )
        .with("session_id", id.to_string()));
    }

    let next_fire_at = schedule.next_after(now).ok_or_else(|| {
        ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            "that schedule has no next occurrence; a one-shot time must be in the future",
        )
    })?;

    let job = ScheduledJob {
        id: Uuid::new_v4().to_string(),
        session_id: id,
        schedule,
        prompt: body.prompt,
        gate,
        created_at: now,
        last_fired_at: None,
        next_fire_at,
        attempts: 0,
    };
    state
        .shared
        .store
        .schedule_store()
        .create_scheduled_job(&job)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to create scheduled job", error)
                .with("session_id", id.to_string())
        })?;

    tracing::info!(
        "created scheduled job {job_id} on session {id} via HTTP, next fire {next_fire}",
        job_id = job.id,
        next_fire = job.next_fire_at.to_rfc3339()
    );
    // The caller just supplied this command, so echoing it back discloses nothing new. `withheld`
    // is `None` by construction: both refusals above have already run against this same level, so a
    // job that reached here can fire.
    Ok((
        StatusCode::CREATED,
        Json(ScheduledJobView::new(&job, true, None)),
    ))
}

/// Render a batch of jobs, each paired with why it cannot fire.
///
/// One permission lookup per distinct *session* rather than per job: a listing of fifty jobs on one
/// session is the ordinary case, and the level is a property of the session. The resident copy wins
/// where there is one, matching the creation door two functions up.
///
/// A session whose level cannot be read does not silence the answer. `job_withheld` asks the
/// questions that need no level -- parking, above all -- before it asks for one, so a job that is
/// provably dead is still reported as such. Passing the level as an `Option` is what lets that
/// happen; resolving it to a default first, or skipping the call, reports a parked job as healthy
/// on the one surface a client would ask.
async fn render_batch(
    state: &ServerState,
    jobs: Vec<ScheduledJob>,
    reveal_command: bool,
) -> Vec<ScheduledJobView> {
    // Snapshotted rather than held: the loop below awaits a database read, and keeping the sessions
    // lock across that would put a listing in the way of every attach and detach.
    let resident: std::collections::HashMap<Uuid, Permission> = {
        let sessions = state.sessions.read().await;
        sessions
            .iter()
            .map(|(id, entry)| (*id, entry.cells().permission.get()))
            .collect()
    };
    let tools = state.shared.gate_tools.as_deref();
    let mut levels: std::collections::HashMap<Uuid, Option<Permission>> =
        std::collections::HashMap::new();
    let mut out = Vec::with_capacity(jobs.len());
    for job in jobs {
        let level = match levels.get(&job.session_id) {
            Some(level) => *level,
            None => {
                let level = match resident.get(&job.session_id) {
                    Some(level) => Some(*level),
                    None => session_permission_from_row(state, job.session_id)
                        .await
                        .ok(),
                };
                levels.insert(job.session_id, level);
                level
            }
        };
        let withheld = match crate::schedule::job_withheld(
            state.shared.store.scheduler_memory(),
            &job,
            level,
            tools,
        ) {
            crate::schedule::Withheld::Yes(reason) => Some(reason),
            crate::schedule::Withheld::No | crate::schedule::Withheld::Undetermined => None,
        };
        out.push(ScheduledJobView::new(
            &job,
            reveal_command,
            withheld_for_scope(withheld, reveal_command),
        ));
    }
    out
}

/// The withheld reason as far as this reader's scope allows.
///
/// The reason is not a neutral sentence. `GateRefusal::explain` embeds `probe.summary()`, which for
/// a tool gate *is* the tool name that `check` beside it is nulled to hide; the level refusals name
/// the session's permission, which is otherwise behind `sessions:r`; and a standing probe failure
/// carries the first line of the check's own output. Attaching it unconditionally therefore handed
/// a `schedule:r` token everything the field next to it withholds, and one MCP server failing to
/// connect turned a server-wide endpoint into a listing of every internal tool name on the box.
///
/// The bare fact survives at that scope, because it is why a client polls this at all, and it
/// discloses nothing `kind` does not already.
fn withheld_for_scope(reason: Option<String>, reveal_command: bool) -> Option<String> {
    match reveal_command {
        true => reason,
        false => reason
            .map(|_| "this job cannot currently fire; the reason needs `sessions:r`".to_string()),
    }
}

/// Refuse a job on a session that is somebody's sub-agent, returning the problem when it is one.
///
/// A job belongs to a root session, and this endpoint is the only door that could plant one
/// elsewhere. A sub-agent has no `schedule_*` tools by construction
/// ([`crate::tools::ToolRegistry::build_for_subagent`] passes no schedule config), so until this
/// the rule was held by omission at the tool door and by nothing at all here.
///
/// **The authority escalation is closed a level down**, by
/// [`crate::host::refuse_a_spawned_session`]: both agent builders refuse a session that records a
/// parent, so a job keyed to a sub-agent cannot wake it unrestricted because nothing can wake it at
/// all. That is where the rule belongs, since `POST /v1/sessions/{id}/turn`, ACP `session/load`,
/// re-attach and `meka -r` reach the same builders and a scheduling-only guard left every one of
/// them open.
///
/// What this refusal is worth is therefore narrow: a job on a sub-agent is refused *when it is
/// created*, naming the session to use instead, rather than being accepted and then failing on
/// every fire until someone reads a log. A diagnosis, not a boundary.
///
/// `SessionNotDrivable` rather than `AuthScope` or `SessionPermission`: no token and no permission
/// level changes the answer, so routing it at either would send a client off to re-provision a
/// token or to `PATCH /v1/sessions/{id}` for a refusal neither can lift. Nor `InvalidBody`, which
/// reads as "resend with a corrected payload"; the sibling doors that refuse the same id answer the
/// same `type`. [`create`] routes its gate refusals by the same rule.
///
/// Takes [`crate::store::SpawnTerms`] rather than a parent, so it asks the question
/// [`crate::host::refuse_a_spawned_session`] asks. Keyed on the parent alone it admits a job on an
/// imported sub-agent: `201 Created`, then `session unavailable: Session is a sub-agent's
/// conversation` in the log on every fire, at the poll cadence, forever.
fn refuse_subagent_session(
    id: Uuid,
    spawned: Option<crate::store::SpawnTerms>,
) -> Option<ProblemDetail> {
    let terms = spawned?;
    let detail = match terms.parent {
        Some(parent) => format!(
            "session '{id}' is a sub-agent of '{parent}', and a sub-agent runs only while the \
             agent that spawned it is waiting on it. Schedule the job on '{parent}' instead and \
             let its turn dispatch the sub-agent."
        ),
        None => format!(
            "session '{id}' carries the terms another session spawned it under, so it is a \
             sub-agent's conversation and runs only while that session is waiting on it. Its \
             parent is not in this store, so nothing here can fire a job on it."
        ),
    };
    Some(
        ProblemDetail::new(
            ErrorKind::SessionNotDrivable,
            StatusCode::UNPROCESSABLE_ENTITY,
            detail,
        )
        .with("session_id", id.to_string()),
    )
}

/// The level a non-resident session's row asks for, narrowed to what this installation permits.
///
/// Split from the fetch so a caller that already holds the row does not read it again. The answer
/// is the scheduler's own, so the two doors cannot drift: a row that records no level, or one this
/// installation no longer enables, is `none`.
fn permission_from_summary(
    state: &ServerState,
    id: Uuid,
    summary: Option<&crate::store::SessionSummary>,
) -> Permission {
    crate::scheduler::live_permission(
        crate::scheduler::SessionLookup::Read(summary),
        &state.shared.config.schedule,
        id,
    )
}

async fn session_permission_from_row(
    state: &ServerState,
    id: Uuid,
) -> Result<Permission, ProblemDetail> {
    let summary = state.shared.store.session_info(id).await.map_err(|error| {
        ProblemDetail::internal_sanitized("failed to read session permission", error)
            .with("session_id", id.to_string())
    })?;
    Ok(permission_from_summary(state, id, summary.as_ref()))
}

/// Resolve exactly one of `at` / `every` / `cron`.
///
/// Ambiguity is refused rather than resolved by precedence, matching `schedule_create`: silently
/// honoring one and dropping the other would produce a job firing on a schedule nobody asked for.
fn parse_schedule(
    body: &CreateJobRequest,
    now: chrono::DateTime<Utc>,
) -> Result<Schedule, ProblemDetail> {
    let given: Vec<(&str, &str)> = [
        ("at", body.at.as_deref()),
        ("every", body.every.as_deref()),
        ("cron", body.cron.as_deref()),
    ]
    .into_iter()
    .filter_map(|(key, value)| {
        value
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| (key, value))
    })
    .collect();

    let invalid = |message: String| {
        ProblemDetail::new(
            ErrorKind::InvalidBody,
            StatusCode::UNPROCESSABLE_ENTITY,
            message,
        )
    };
    match given.as_slice() {
        [("at", value)] => Schedule::parse_at(value, now).map_err(invalid),
        [("every", value)] => Schedule::parse_every(value).map_err(invalid),
        [("cron", value)] => Schedule::parse_cron(value).map_err(invalid),
        [] => Err(invalid(
            "give one of `at` (once), `every` (interval), or `cron` (expression)".to_string(),
        )),
        several => Err(invalid(format!(
            "give exactly one schedule, got {}",
            several
                .iter()
                .map(|(key, _)| *key)
                .collect::<Vec<_>>()
                .join(" and ")
        ))),
    }
}

/// `DELETE /v1/schedule/{job_id}`: cancel a job.
///
/// Keyed on the job id alone rather than nested under its session, because that is how a client
/// that read `GET /v1/schedule` holds it.
///
/// Takes a unique id prefix as well as the full id, and 404s when nothing matches. Both halves
/// matter, and for the same reason: the 8-character short form is what every surface that renders a
/// job to a human shows (`meka schedule list`, the REPL's `/schedule`, the `schedule_list` tool),
/// so an operator will paste one here, and answering 204 to an id that matched nothing would report
/// a still-firing job as canceled. A gated job kept alive that way goes on running a shell command
/// unattended. `schedule_cancel` and `meka schedule cancel` already resolve prefixes and already
/// report a miss; this is the surface that did not.
#[utoipa::path(
    delete,
    path = "/v1/schedule/{job_id}",
    tag = "schedule",
    params(("job_id" = String, Path, description = "Scheduled job id, or a unique prefix of one")),
    responses(
        (status = 204, description = "Job canceled"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "No job matches that id", body = ProblemDetail),
        (status = 422, description = "The prefix matches more than one job", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["schedule:w"]))
)]
pub(crate) async fn cancel(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::ScheduleWrite>,
    Path(job_id): Path<String>,
) -> Result<StatusCode, ProblemDetail> {
    let jobs = state
        .shared
        .store
        .schedule_store()
        .list_all_scheduled_jobs()
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to resolve scheduled job", error)
                .with("job_id", job_id.clone())
        })?;
    let wanted = crate::text::id_prefix_for_matching(&job_id);
    let matches: Vec<&ScheduledJob> = jobs
        .iter()
        .filter(|job| crate::text::is_usable_id_prefix(&job_id) && job.id.starts_with(&wanted))
        .collect();
    let resolved = match matches.as_slice() {
        [job] => job.id.clone(),
        [] => {
            return Err(ProblemDetail::new(
                ErrorKind::NotFound,
                StatusCode::NOT_FOUND,
                format!("no scheduled job matches '{job_id}'"),
            )
            .with("job_id", job_id.clone()));
        }
        several => {
            return Err(ProblemDetail::new(
                ErrorKind::InvalidBody,
                StatusCode::UNPROCESSABLE_ENTITY,
                format!(
                    "'{}' matches {} scheduled jobs; use a longer id",
                    job_id,
                    several.len()
                ),
            )
            .with("job_id", job_id.clone()));
        }
    };

    let removed = state
        .shared
        .store
        .schedule_store()
        .delete_scheduled_job(&resolved)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to cancel scheduled job", error)
                .with("job_id", resolved.clone())
        })?;
    // The listing above and the delete are two statements, and a scheduler sweep can retire the row
    // in between. `204` then reported a cancellation this request did not perform, which is exactly
    // what a client polls this endpoint to establish. `404` is the same answer it would have got a
    // moment earlier, and is true.
    if !removed {
        return Err(ProblemDetail::new(
            ErrorKind::NotFound,
            StatusCode::NOT_FOUND,
            format!("scheduled job '{resolved}' was already gone"),
        )
        .with("job_id", resolved));
    }
    tracing::info!("canceled scheduled job {resolved} via HTTP");
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct BackgroundTaskView {
    pub(crate) id: String,
    pub(crate) session_id: Uuid,
    /// The tool that was backgrounded, e.g. `execute_command`.
    pub(crate) tool_name: String,
    /// Human-readable summary of what was started.
    pub(crate) label: String,
    /// `running`, `completed`, `failed`, `cancelled`, or `interrupted`.
    pub(crate) status: String,
    /// The tool's output for a terminal task, truncated when it was also spilled to the
    /// scratchpad.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) outcome: Option<String>,
    /// Scratchpad entry holding the full output, when it was too large to carry inline.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) scratchpad_name: Option<String>,
    pub(crate) started_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) finished_at: Option<String>,
    /// When subscribers were told, which is when `task.finished` fired for this task. Absent when
    /// nobody has been told yet.
    ///
    /// A row migrated from before the split carries its `delivered_at` here, which records that
    /// the outcome was reported rather than that any endpoint received it: the store cannot
    /// know whether one was subscribed at the time.
    ///
    /// Separate from `delivered_at` because the two answer different questions and stopped
    /// happening together: announcing needs nothing from a live session, while telling the agent
    /// needs a turn. A client polling this after a `task.finished` can otherwise not tell
    /// "announced, waiting for a turn" from "never announced at all".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) announced_at: Option<String>,
    /// When the outcome was handed to the agent. Absent when it is still waiting to be delivered
    /// on the session's next turn.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) delivered_at: Option<String>,
}

impl From<crate::store::background::BackgroundTask> for BackgroundTaskView {
    fn from(task: crate::store::background::BackgroundTask) -> Self {
        Self {
            id: task.id,
            session_id: task.session_id,
            tool_name: task.tool_name,
            label: task.label,
            status: task.status.name().to_string(),
            outcome: task.outcome,
            scratchpad_name: task.scratchpad_name,
            started_at: task.started_at.to_rfc3339(),
            finished_at: task.finished_at.map(|at| at.to_rfc3339()),
            announced_at: task.announced_at.map(|at| at.to_rfc3339()),
            delivered_at: task.delivered_at.map(|at| at.to_rfc3339()),
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct BackgroundTasksResponse {
    pub(crate) tasks: Vec<BackgroundTaskView>,
}

/// `GET /v1/sessions/{id}/tasks`: this session's background tasks, newest first.
#[utoipa::path(
    get,
    path = "/v1/sessions/{id}/tasks",
    tag = "tasks",
    params(("id" = Uuid, Path, description = "Session UUID")),
    responses(
        (status = 200, description = "Background tasks", body = BackgroundTasksResponse),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session not found", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:r"]))
)]
pub(crate) async fn list_tasks(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SessionsRead>,
    Path(id): Path<Uuid>,
) -> Result<Json<BackgroundTasksResponse>, ProblemDetail> {
    require_session_exists(&state, id).await?;
    let tasks = state
        .shared
        .store
        .background_store()
        .list_background_tasks(id)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to list background tasks", error)
                .with("session_id", id.to_string())
        })?;
    Ok(Json(BackgroundTasksResponse {
        tasks: tasks.into_iter().map(BackgroundTaskView::from).collect(),
    }))
}

/// `DELETE /v1/sessions/{id}/tasks/{task_id}`: stop a running background task.
#[utoipa::path(
    delete,
    path = "/v1/sessions/{id}/tasks/{task_id}",
    tag = "tasks",
    params(
        ("id" = Uuid, Path, description = "Session UUID"),
        ("task_id" = String, Path, description = "Background task id"),
    ),
    responses(
        (status = 204, description = "Task canceled, or already terminal"),
        (status = 401, description = "Authorization missing or invalid", body = ProblemDetail),
        (status = 403, description = "Insufficient scope", body = ProblemDetail),
        (status = 404, description = "Session or task not found", body = ProblemDetail),
        (status = 422, description = "The prefix matches more than one task", body = ProblemDetail),
        (status = 500, description = "Internal server error", body = ProblemDetail),
    ),
    security(("bearerAuth" = ["sessions:w"]))
)]
pub(crate) async fn cancel_task(
    State(state): State<ServerState>,
    _scoped: scope::Scoped<scope::SessionsWrite>,
    Path((id, task_id)): Path<(Uuid, String)>,
) -> Result<StatusCode, ProblemDetail> {
    let Some(task) = state
        .shared
        .store
        .background_store()
        .resolve_background_task(id, &task_id)
        .await
        .map_err(|error| match &error {
            // `resolve_background_task` accepts an id prefix, and reports an ambiguous one as
            // `Config`. That is a statement about the caller's input, so it is a 422; routing it
            // through `internal_sanitized` would report "use a longer id" as a server fault and
            // hide the one detail that fixes it.
            crate::error::MekaError::Config(message) => ProblemDetail::new(
                ErrorKind::InvalidBody,
                StatusCode::UNPROCESSABLE_ENTITY,
                message.clone(),
            )
            .with("session_id", id.to_string()),
            _ => ProblemDetail::internal_sanitized("failed to resolve background task", error)
                .with("session_id", id.to_string()),
        })?
    else {
        return Err(ProblemDetail::new(
            ErrorKind::NotFound,
            StatusCode::NOT_FOUND,
            format!("no background task in session {id} matches '{task_id}'"),
        )
        .with("session_id", id.to_string())
        .with("task_id", task_id.clone()));
    };
    if task.status.is_terminal() {
        return Ok(StatusCode::NO_CONTENT);
    }

    // Recorded before signaling, exactly as `task_cancel` does: `finish_background_task` only
    // writes over a `running` row, so a task finishing in the same instant cannot report success
    // after the caller was told it was stopped.
    state
        .shared
        .store
        .background_store()
        .finish_background_task(&task.id, TaskStatus::Cancelled, None, None)
        .await
        .map_err(|error| {
            ProblemDetail::internal_sanitized("failed to record task cancellation", error)
                .with("task_id", task.id.clone())
        })?;

    // Read off the entry, not through `runtime.agent`: the registry is hoisted onto `SessionEntry`
    // precisely so this path does not wait on the mutex an in-flight turn holds, which is the
    // state a session is in whenever anyone actually wants to stop a detached task.
    //
    // A task belonging to an evicted session has a row here but no live handle. Recording the
    // cancellation is still right, and is what stops a resumed session waiting on an outcome that
    // will never arrive.
    let entry = state.sessions.read().await.get(&id).cloned();
    let signaled = match entry {
        Some(entry) => entry.cells().background_tasks.cancel(&task.id).await,
        None => false,
    };
    if !signaled {
        // `warn`, not `debug`: this is the one outcome that differs from what the 204 claims. The
        // row now says `cancelled` and `GET /tasks` will agree, but nothing in this process could
        // reach the task, so if it is still running it will run to completion unnoticed -- and a
        // second cancel short-circuits on the now-terminal status. Worth seeing by default.
        tracing::warn!(
            "background task {task_id} had no live handle in this process; recorded as \
             canceled, but anything still running was not signaled",
            task_id = task.id
        );
    }
    tracing::info!(
        "canceled background task {task_id} via HTTP",
        task_id = task.id
    );
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A job may be planted on a root session and on nothing else.
    ///
    /// The refusal is a diagnosis rather than a boundary: `crate::host::refuse_a_spawned_session`
    /// is what stops a sub-agent being driven unrestricted, in the builders every door shares.
    /// This keeps the failure at creation time, where it can name the session to schedule on
    /// instead, rather than letting a job be accepted and fail on every fire.
    ///
    /// All three arms are asserted. A guard that refused everything would pass a
    /// refusal-only test while breaking every ordinary job, and one that read only the parent
    /// admitted an imported sub-agent -- which is how a `201 Created` came to be followed by a fire
    /// failure at the poll cadence for as long as the job lived.
    #[test]
    fn a_job_is_refused_on_a_sub_agent_session_and_admitted_on_a_top_level_one() {
        let sub_agent = Uuid::new_v4();
        let parent = Uuid::new_v4();

        assert!(
            refuse_subagent_session(sub_agent, None).is_none(),
            "a session with no spawn terms is the ordinary case and must be admitted"
        );

        let refusal = refuse_subagent_session(
            sub_agent,
            Some(crate::store::SpawnTerms {
                parent: Some(parent),
            }),
        )
        .expect("a session with a parent cannot own a job");
        assert_eq!(refusal.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(refusal.type_uri, ErrorKind::SessionNotDrivable.type_uri());
        // Both ids, because the remedy is to re-issue the request against the parent and a client
        // that was handed the sub-agent's id may not know what spawned it.
        let detail = refusal.detail.as_deref().unwrap_or_default();
        assert!(
            detail.contains(&sub_agent.to_string()) && detail.contains(&parent.to_string()),
            "the refusal must name the sub-agent and the session to use instead: {detail}"
        );

        let orphaned =
            refuse_subagent_session(sub_agent, Some(crate::store::SpawnTerms { parent: None }))
                .expect("spawn terms with no parent are still a sub-agent's conversation");
        assert_eq!(orphaned.type_uri, ErrorKind::SessionNotDrivable.type_uri());
        let detail = orphaned.detail.as_deref().unwrap_or_default();
        assert!(
            detail.contains(&sub_agent.to_string()) && !detail.contains("Schedule the job on"),
            "with no parent in the store there is no session to redirect to: {detail}"
        );
    }

    /// Nothing about *why* a job is held escapes to a token that may not see what the gate runs.
    ///
    /// Every refusal sentence carries something the scope withholds elsewhere: a tool name, a
    /// session's permission level, or a line of the check's own output. Asserting on the exact
    /// wording would be brittle, so this asserts the property -- none of the reason survives --
    /// against the three shapes the reasons actually take.
    #[test]
    fn a_low_scope_reader_learns_that_a_job_is_held_and_nothing_else() {
        let reasons = [
            "no gate tool named 'mcp__internal_bridge__unseen'. A gate can call a read-only tool",
            "a gate command runs unattended with no sandbox, so it needs `unrestricted` \
             (currently workspace)",
            "its gate keeps failing and cannot say whether to fire: gate points at `/x` but the \
             probe did not return JSON: sk-live-DO-NOT-DISCLOSE",
        ];
        for reason in reasons {
            let redacted = withheld_for_scope(Some(reason.to_string()), false)
                .expect("the fact that it is held still reaches the client");
            for leaked in [
                "mcp__internal_bridge__unseen",
                "workspace",
                "sk-live-DO-NOT-DISCLOSE",
                "unrestricted",
            ] {
                assert!(
                    !redacted.contains(leaked),
                    "{leaked:?} reached a reader that cannot see `check`: {redacted:?}"
                );
            }
            assert_eq!(
                withheld_for_scope(Some(reason.to_string()), true).as_deref(),
                Some(reason),
                "and a reader that can see `check` still gets the whole answer"
            );
        }
        assert_eq!(
            withheld_for_scope(None, false),
            None,
            "a job that can fire is not reported as held to anyone"
        );
    }
}
