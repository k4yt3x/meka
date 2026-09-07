//! Resident ACP sessions: the per-connection state, the session lifecycle handlers, the idle
//! sweep, and shutdown.

use super::*;

/// Validate a client's `additionalDirectories`, rejecting any relative entry.
///
/// The spec requires each to be absolute, and meka has no defensible base to resolve a relative one
/// against: joining to `cwd` would invent a root the client never named. Failing the request is the
/// honest answer, and it matches how `cwd` itself is validated.
pub(super) fn validate_additional_roots(
    roots: &[PathBuf],
) -> Result<(), agent_client_protocol::Error> {
    for root in roots {
        if !root.is_absolute() {
            return Err(invalid_params_error(format!(
                "additionalDirectories entries must be absolute paths; got '{}'",
                root.display()
            )));
        }
    }
    Ok(())
}
/// The `sessionId` a request names, as the id every session this host serves is keyed by.
/// Anything that is not a UUID names nothing this process could have handed out, so it is refused
/// as malformed rather than looked up.
pub(super) fn parse_session_id(
    session_id: &str,
) -> Result<uuid::Uuid, agent_client_protocol::Error> {
    uuid::Uuid::parse_str(session_id)
        .map_err(|_| invalid_params_error(format!("malformed sessionId: {session_id}")))
}
/// Move a session's agent onto whatever its row currently names, before a turn runs on it.
///
/// **The row is the carrier, and the only one.** `session/set_config_option` moves the agent
/// itself, but every other writer of the row (`meka -r --profile`, a `PATCH` on a server sharing
/// the store) reaches this process only through the row, and this is what applies it. A resolved
/// profile parked on the session entry is drained only by `session/prompt`, so a scheduled fire or
/// a background-outcome turn runs on, and bills, the profile the user has left, while the row,
/// both pickers and the reported window all say otherwise. A parked value was a second carrier of
/// a fact the row already held, and it could lose to any other writer of that row.
///
/// Cheap when nothing has changed: one indexed row read and a comparison, with no resolution at all
/// unless the two differ.
///
/// Must be called under the runtime mutex, which is what makes "the agent this turn is about to
/// use" the thing being moved.
pub(super) async fn apply_recorded_profile(
    state: &ServerState,
    agent: &crate::agent::Agent,
    session_uuid: uuid::Uuid,
) -> anyhow::Result<()> {
    let Some(recorded) = state.shared.store.recorded_profile(session_uuid).await? else {
        // No row, so nothing names a profile to move to. Reachable only for a session deleted from
        // under a live entry; its turn is going to fail on the write either way.
        return Ok(());
    };
    if recorded == agent.profile() {
        return Ok(());
    }
    let profile = recorded.clone();
    let resolved = crate::provider::resolved_profile(&state.shared.providers, recorded).await?;
    agent.set_provider(resolved);
    tracing::info!("moved session {session_uuid} onto profile '{profile}'");
    Ok(())
}
/// Aborts a background task when the value is dropped. `run_acp` has several exit paths, and a
/// scheduler that outlived them would keep running turns against a connection nobody is reading.
pub(super) struct AbortOnDrop(pub(super) tokio::task::JoinHandle<()>);
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}
/// Process-wide ACP server state. The outer `sessions` `RwLock` is held only for map insert /
/// lookup / remove; per-session mutable state lives behind each entry's inner `Mutex` so a
/// long-running prompt on one session never blocks operations on another.
pub(super) struct ServerState {
    pub(super) shared: Arc<crate::host::SharedDeps>,
    pub(super) client_state: SharedClientState,
    pub(super) sessions: crate::host::Sessions<String, SessionEntry>,
    /// Shared with every per-session `AcpFrontend`; see the field on `AcpFrontend` for the
    /// stdio-level rationale.
    pub(super) transport_dead: Arc<std::sync::atomic::AtomicBool>,
}

impl ServerState {
    /// A server with no sessions yet.
    pub(super) fn new(
        shared: Arc<crate::host::SharedDeps>,
        client_state: SharedClientState,
        transport_dead: Arc<std::sync::atomic::AtomicBool>,
    ) -> Self {
        Self {
            shared,
            client_state,
            sessions: crate::host::Sessions::new(),
            transport_dead,
        }
    }

    /// The link a new session's frontend talks over.
    pub(super) fn link(
        &self,
        connection: ConnectionTo<Client>,
        session_id: SessionId,
    ) -> super::frontend::AcpClientLink {
        super::frontend::AcpClientLink {
            connection,
            session_id,
            client_state: self.client_state.clone(),
            transport_dead: Arc::clone(&self.transport_dead),
        }
    }
}
/// One session `meka acp` holds open: the shared [`crate::host::ResidentSession`] plus what only
/// the ACP host tracks. Derefs to the resident part.
#[derive(Clone)]
pub(super) struct SessionEntry {
    pub(super) resident: crate::host::ResidentSession,
    /// Whether `session/update` with the title has been sent for this session yet.
    pub(super) title_sent: Arc<std::sync::atomic::AtomicBool>,
    pub(super) frontend: Arc<AcpFrontend>,
}

impl std::ops::Deref for SessionEntry {
    type Target = crate::host::ResidentSession;

    fn deref(&self) -> &Self::Target {
        &self.resident
    }
}

/// What [`build_session_runtime`] opens for a session: the resident session, and the frontend the
/// client is reached through.
pub(super) struct BuiltSession {
    pub(super) resident: crate::host::ResidentSession,
    pub(super) frontend: Arc<AcpFrontend>,
}

/// `futures::io::AsyncRead` wrapper over the ACP stdin transport that fires `eof` (a
/// `CancellationToken`) when the underlying reader reports end-of-stream. The
/// `agent-client-protocol` connection future does not resolve on idle stdin EOF by itself (its
/// outgoing actor stays alive while we hold `ConnectionTo` handles), so we observe EOF here and let
/// `acp_run_until_disconnect` use it to shut down. Without this, a `meka acp` whose client
/// disconnected lingers forever holding its session `flock`, and reopening that session later fails
/// with `SessionLocked`.
pub(super) struct EofSignalingRead<R> {
    pub(super) inner: R,
    pub(super) eof: CancellationToken,
}
impl<R: AsyncRead + Unpin> AsyncRead for EofSignalingRead<R> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        let this = self.get_mut();
        let result = Pin::new(&mut this.inner).poll_read(cx, buffer);
        // A zero-length read into a non-empty buffer is end-of-stream: the client closed stdio (or
        // the parent died, closing the pipe). Fire the shutdown token; `cancel()` is idempotent so
        // repeated EOF reads are harmless.
        if matches!(result, Poll::Ready(Ok(0))) && !buffer.is_empty() {
            this.eof.cancel();
        }
        result
    }
}
/// Max time to wait for in-flight turns to unwind during ACP shutdown before abandoning them. They
/// are abandoned safely regardless (the OS releases the session `flock` when the process exits),
/// but the grace window lets a running turn reach its interrupt path and persist its partial output
/// first.
pub(super) const ACP_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
/// `connect_with` `main_fn`: resolve (shutting the connection down) when the ACP client disconnects
/// (stdin EOF, signaled via `stdin_eof`) or a termination signal arrives. Before returning it
/// cancels every in-flight turn and waits briefly so a running turn can persist its partial output.
/// The connection's spawned turns run inside the `background` future this return races against, so
/// the drain must happen here, before we return and that future is dropped.
pub(super) async fn acp_run_until_disconnect(
    state: Arc<ServerState>,
    stdin_eof: CancellationToken,
) -> std::result::Result<(), agent_client_protocol::Error> {
    tokio::select! {
        _ = stdin_eof.cancelled() => {
            tracing::info!("ACP client disconnected (stdin EOF); shutting down");
        }
        _ = acp_shutdown_signal() => {
            tracing::info!("received termination signal; shutting down ACP server");
        }
    }
    drain_acp_sessions(&state).await;
    if tokio::time::timeout(ACP_DRAIN_TIMEOUT, wait_for_sessions_idle(&state))
        .await
        .is_err()
    {
        tracing::warn!("ACP shutdown drain timed out; abandoning in-flight turn(s)");
    }

    // Reclaim every session the client never closed.
    //
    // `session/close` is an *optional* capability, so a client that simply exits leaves each entry
    // resident: an `Agent`, a `ToolRegistry` the MCP manager holds a strong clone of, and an open
    // file lock. Dropping the map here releases the flock -- which is what lets the same session be
    // reopened by the next `meka` without a `SessionLocked` error -- and lets the registry go, so
    // `tools/list_changed` stops fanning out to sessions that no longer exist.
    let abandoned = {
        let mut sessions = state.sessions.write().await;
        std::mem::take(&mut *sessions)
    };
    if !abandoned.is_empty() {
        for entry in abandoned.values() {
            // Nothing here waits on the conversation: an in-flight turn that outlived the drain
            // timeout still holds it, and blocking would trade a leaked registry for a hung
            // shutdown.
            entry.release(state.shared.mcp_manager.as_ref()).await;
        }
        let released = abandoned.len();
        tracing::info!("released {released} session(s) the client did not close");
    }
    drop(abandoned);

    Ok(())
}
/// How long an ACP session may sit untouched before it is released.
///
/// Matches `[serve].idle_timeout`'s default. Not configurable, deliberately: a day is long past the
/// point where an editor still means to use a session, and a knob here would be a setting nobody
/// sets for a mechanism nobody should notice.
pub(super) const ACP_SESSION_IDLE_TIMEOUT: std::time::Duration =
    crate::host::session::DEFAULT_IDLE_TIMEOUT;
/// How often the idle sweep runs. Matches `[serve].gc_scan_interval`'s default.
pub(super) const ACP_IDLE_SCAN_INTERVAL: std::time::Duration =
    crate::host::session::DEFAULT_GC_SCAN_INTERVAL;
/// Release sessions the editor opened and stopped using.
///
/// `session/close` is an *optional* ACP capability, so an editor is entitled never to send one, and
/// several do not. Each session it opens holds an `Agent`, a `ToolRegistry` the MCP manager keeps a
/// strong clone of, and an open file lock; over a long editing session that is a descriptor per
/// file the user glanced at, none of them released, and the session cannot be reopened from
/// anywhere else meanwhile.
///
/// Only the in-memory entry goes. The row stays, so `session/load` brings the conversation back
/// exactly as it does for a session from a previous run -- which is the same trade `meka serve`
/// makes with `delete_on_idle = false`.
pub(super) fn spawn_idle_session_sweep(state: Arc<ServerState>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(ACP_IDLE_SCAN_INTERVAL);
        ticker.tick().await;
        loop {
            ticker.tick().await;
            for (id, entry) in state.sessions.sweep_idle(ACP_SESSION_IDLE_TIMEOUT).await {
                // Same teardown `session/close` does, and for the same reason: without it the
                // manager keeps fanning `tools/list_changed` out to a registry nobody reads.
                entry.release(state.shared.mcp_manager.as_ref()).await;
                tracing::info!(
                    "released idle ACP session {id} after {ACP_SESSION_IDLE_TIMEOUT:?}; `session/load` reopens it"
                );
                drop(entry);
            }
        }
    })
}

/// Cancel every active session's in-flight turn. Mirrors `crate::host::http`'s drain.
///
/// Through the cell, not the live token alone: a prompt admitted but not yet published has no
/// token, and only the cell's epoch bump reaches it. Firing the live tokens left such a prompt to
/// run its whole turn during the drain and then be abandoned.
pub(super) async fn drain_acp_sessions(state: &ServerState) {
    let sessions = state.sessions.read().await;
    for entry in sessions.values() {
        entry.cancel.cancel();
    }
}
/// Resolve once no session is running a turn. The prompt handler holds `entry.runtime`'s lock for
/// the whole turn, so a successful `try_lock` on every session means all turns have unwound.
pub(super) async fn wait_for_sessions_idle(state: &ServerState) {
    loop {
        let all_idle = {
            let sessions = state.sessions.read().await;
            sessions
                .values()
                .all(|entry| entry.conversation.try_lock().is_ok())
        };
        if all_idle {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
}
/// Wait for a cross-platform termination signal: SIGTERM or Ctrl-C on unix, Ctrl-C elsewhere.
/// Mirrors `crate::host::http`'s `shutdown_signal`.
pub(super) async fn acp_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut terminate) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = terminate.recv() => {}
                }
            }
            Err(error) => {
                tracing::warn!(
                    "failed to install SIGTERM handler: {error}; relying on Ctrl+C only"
                );
                if let Err(error) = tokio::signal::ctrl_c().await {
                    tracing::warn!("failed to listen for Ctrl+C: {error}");
                }
            }
        }
    }
    #[cfg(not(unix))]
    {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::warn!("failed to listen for Ctrl+C: {error}");
        }
    }
}
/// What a reopening client's workspace changes on the row: its `cwd` wins (consistent with
/// `session/new`'s captured cwd), and its `additionalDirectories` are the complete resulting list,
/// so an empty one clears the column. Only what differs is named, so a reopen that changes nothing
/// writes nothing and does not bump `updated_at`, which would reorder the client's own listing on
/// every load.
fn reopened_workspace(
    summary: &crate::store::SessionSummary,
    cwd: &std::path::Path,
    additional_roots: &[PathBuf],
) -> crate::store::SessionPatch {
    crate::store::SessionPatch {
        cwd: (summary.cwd.as_deref() != Some(cwd)).then(|| cwd.to_path_buf()),
        roots: (summary.additional_roots != additional_roots).then(|| additional_roots.to_vec()),
        ..Default::default()
    }
}
/// `session/load`: reopen a previously persisted session and add it to the active sessions map.
/// Replays the persisted history as `session/update` notifications so the client's UI rebuilds the
/// conversation before the response goes out.
pub(super) async fn handle_load_session(
    state: Arc<ServerState>,
    request: LoadSessionRequest,
    responder: agent_client_protocol::Responder<LoadSessionResponse>,
    cx: ConnectionTo<Client>,
) -> Result<(), agent_client_protocol::Error> {
    let session_id_str = request.session_id.0.as_ref().to_string();
    let session_uuid = match parse_session_id(&session_id_str) {
        Ok(uuid) => uuid,
        Err(error) => return responder.respond_with_error(error),
    };

    // Refuse if a session with the same id is already loaded. Collisions between different
    // connections aren't possible (one process serves one ACP client) but a re-load of the same
    // session would discard in-flight state.
    if state.sessions.read().await.contains_key(&session_id_str) {
        return responder.respond_with_error(invalid_params_error(
            "session is already loaded; call session/close first",
        ));
    }

    let cwd = match crate::workspace::accept_cwd(&request.cwd) {
        Ok(cwd) => cwd,
        Err(error) => return responder.respond_with_error(acp_error_for(&error, false)),
    };
    if let Err(error) = validate_additional_roots(&request.additional_directories) {
        return responder.respond_with_error(error);
    }

    // Before the lock, and before the writes below. `build_session_runtime` refuses a sub-agent at
    // the end of this handler, but by then it has taken the sub-agent's file lock, rewritten its
    // `cwd` and replaced its `additional_roots_json`: durable writes to a session the caller may
    // not drive, followed by an *internal* error for something the caller got wrong. `cwd` is the
    // writable boundary at `workspace` and the directory a scheduled gate is re-checked in, so
    // moving it is not cosmetic.
    //
    // The `claim_session` call in between is *not* part of the harm, though it looks like it: its
    // sweep is keyed on this id, and a sub-agent has no `background_tasks` rows because
    // `Agent::new_subagent` never enables them.
    //
    // Through the shared predicate rather than `summary.parent_id`, so the load doors cannot drift
    // and an imported sub-agent with no surviving parent is refused here too.
    if let Err(error) =
        crate::host::refuse_a_spawned_session(&state.shared.store, Some(session_uuid)).await
    {
        return responder.respond_with_error(build_failure_error(
            "failed to read session",
            &error,
            state.shared.relay_provider_errors(),
        ));
    }

    // Locked, then read, so the row this session reopens from is the one this process now owns,
    // and no concurrent process can write events while history is replayed.
    let (session_lock, summary) = match state.shared.store.open_session_row(session_uuid).await {
        Ok(opened) => opened,
        Err(error) => return responder.respond_with_error(acp_error_for(&error, false)),
    };

    let session_id: SessionId = session_id_str.clone().into();

    // The level this session was last set to, not the one this process starts at: the row carries
    // what the user last chose via `session/set_mode`, and the scheduler's live gate re-check reads
    // that row, so a session whose row said `unrestricted` while its live cell sat at the config
    // default would have its gates evaluated against authority the session is not running at. A
    // level this configuration no longer enables drops to the default, as on every other host.
    let permission = state
        .shared
        .config
        .enabled_permissions
        .admit_recorded(summary.permission, &format!("session {session_uuid}"))
        .unwrap_or(state.shared.config.permission);
    let runtime = match build_session_runtime(
        &state.shared,
        state.link(cx.clone(), session_id.clone()),
        session_uuid,
        cwd.clone(),
        request.additional_directories.clone(),
        permission,
        summary.approvals,
        crate::host::Opening::Hydrate,
        session_lock,
    )
    .await
    {
        Ok(runtime) => runtime,
        Err(error) => {
            return responder.respond_with_error(build_failure_error(
                "failed to build session runtime",
                &error,
                state.shared.relay_provider_errors(),
            ));
        }
    };

    // The client's cwd wins (consistent with `session/new`'s captured cwd), and the roots it names
    // are the complete resulting list, so `session/list` reflects the live state. Only what differs
    // is written: a reopen that changes nothing must not bump `updated_at` and reorder the client's
    // own listing.
    //
    // After the build, not before. The sub-agent refusal above is the only refusal this handler can
    // make on its own; the builder makes the rest -- a profile that has left `config.toml`, an
    // account with no stored credential -- and a load refused for one of those had already
    // rewritten the session's `cwd` and roots on its way to saying so. `cwd` is the writable
    // boundary at `workspace` and the directory a scheduled gate is re-checked in, so a session
    // this handler declined to open must not come away pointing somewhere else. The runtime is
    // released on a failed write for the same reason a failed build is: it holds the session's lock
    // and an MCP-attached registry.
    if let Err(error) = crate::host::record_session_change(
        &state.shared.store,
        session_uuid,
        reopened_workspace(&summary, &cwd, &request.additional_directories),
    )
    .await
    {
        runtime
            .resident
            .release(state.shared.mcp_manager.as_ref())
            .await;
        return responder.respond_with_error(acp_error_for(&error, false));
    }

    // Replay before inserting so the client sees the rebuild stream before any new turn-related
    // update could race in.
    replay_session_updates(
        &cx,
        &session_id,
        &runtime.resident.cells().cwd,
        &*runtime.resident.conversation.lock().await,
    );

    let permission = runtime.resident.cells().permission.clone();
    let frontend = Arc::clone(&runtime.frontend);
    // History already carries the first user message, so the title is known; push it once now,
    // sharing the flag with the entry so a later prompt won't re-emit it.
    let title_sent = Arc::new(std::sync::atomic::AtomicBool::new(false));
    maybe_emit_session_title(
        &cx,
        &session_id,
        &title_sent,
        &*runtime.resident.conversation.lock().await,
    );
    let entry = SessionEntry {
        resident: runtime.resident,
        title_sent,
        frontend,
    };
    state.sessions.write().await.insert(session_id_str, entry);

    // Refresh the palette + advertise the current mode set: the editor was reopened, so its UI
    // starts blank.
    let modes = build_mode_state(&permission);
    let config_options = build_config_options(&state.shared, &permission, Some(session_uuid)).await;
    emit_available_commands(&cx, &session_id, &state.shared.skills).await;

    responder.respond(
        LoadSessionResponse::new()
            .modes(modes)
            .config_options(config_options),
    )
}
/// `session/list`: paginated index of persisted sessions, filtered by cwd when the client asks.
/// Sub-agent sessions are excluded; they're internal audit rows, not user-facing conversations.
pub(super) async fn handle_list_sessions(
    state: Arc<ServerState>,
    request: ListSessionsRequest,
    responder: agent_client_protocol::Responder<ListSessionsResponse>,
) -> Result<(), agent_client_protocol::Error> {
    const PAGE_SIZE: u32 = 50;
    let cwd_filter = request.cwd.as_deref().map(crate::workspace::cwd_filter);
    let cursor = request.cursor.as_deref();
    let (rows, next_cursor) = match state
        .shared
        .store
        .list_sessions(PAGE_SIZE, false, cwd_filter.as_deref(), cursor)
        .await
    {
        Ok(pair) => pair,
        // Through the classifier: a `Database` failure's `Display` names the store's own path,
        // which is the operator's to read in the log rather than the editor's.
        Err(error) => {
            return responder
                .respond_with_error(acp_error_for(&error, state.shared.relay_provider_errors()));
        }
    };
    // Fallback for a row carrying no `cwd`, which `meka session import` produces when the archive
    // omits it. The process cwd matches what the agent would use for relative-path resolution if
    // the client picked one of these to load. That is better than refusing to surface them.
    let fallback_cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let sessions = rows
        .into_iter()
        .map(|summary| {
            let cwd = summary.cwd.unwrap_or_else(|| fallback_cwd.clone());
            let mut info =
                SessionInfo::new(summary.id.to_string(), cwd).updated_at(summary.updated_at);
            if !summary.additional_roots.is_empty() {
                info = info.additional_directories(summary.additional_roots);
            }
            if !summary.title.is_empty() {
                info = info.title(summary.title);
            }
            info
        })
        .collect::<Vec<_>>();

    let mut response = ListSessionsResponse::new(sessions);
    if let Some(token) = next_cursor {
        response = response.next_cursor(token);
    }
    responder.respond(response)
}
/// `session/resume`: adopt an existing session as active without replaying. Used when the client
/// already has the history in its UI and just wants the agent to pick up the conversation context.
pub(super) async fn handle_resume_session(
    state: Arc<ServerState>,
    request: ResumeSessionRequest,
    responder: agent_client_protocol::Responder<ResumeSessionResponse>,
    cx: ConnectionTo<Client>,
) -> Result<(), agent_client_protocol::Error> {
    let session_id_str = request.session_id.0.as_ref().to_string();
    let session_uuid = match parse_session_id(&session_id_str) {
        Ok(uuid) => uuid,
        Err(error) => return responder.respond_with_error(error),
    };

    if state.sessions.read().await.contains_key(&session_id_str) {
        return responder.respond_with_error(invalid_params_error(
            "session is already loaded; call session/close first",
        ));
    }

    let cwd = match crate::workspace::accept_cwd(&request.cwd) {
        Ok(cwd) => cwd,
        Err(error) => return responder.respond_with_error(acp_error_for(&error, false)),
    };
    if let Err(error) = validate_additional_roots(&request.additional_directories) {
        return responder.respond_with_error(error);
    }

    // Before the lock, and before any of the writes below, for the reason `session/load` gives:
    // the builder's refusal arrives after the side effects on a session the caller may not drive.
    //
    // Through the shared predicate rather than `summary.parent_id`, so the two doors cannot drift
    // and so an imported sub-agent with no surviving parent is refused here too.
    if let Err(error) =
        crate::host::refuse_a_spawned_session(&state.shared.store, Some(session_uuid)).await
    {
        return responder.respond_with_error(build_failure_error(
            "failed to read session",
            &error,
            state.shared.relay_provider_errors(),
        ));
    }

    // Locked, then read, as `session/load` does and for the same reason.
    let (session_lock, summary) = match state.shared.store.open_session_row(session_uuid).await {
        Ok(opened) => opened,
        Err(error) => return responder.respond_with_error(acp_error_for(&error, false)),
    };

    let session_id: SessionId = session_id_str.clone().into();

    // The level and switch the row records, clamped to this configuration, as `session/load`
    // reads them and for the same reason.
    let permission = state
        .shared
        .config
        .enabled_permissions
        .admit_recorded(summary.permission, &format!("session {session_uuid}"))
        .unwrap_or(state.shared.config.permission);
    let runtime = match build_session_runtime(
        &state.shared,
        state.link(cx.clone(), session_id.clone()),
        session_uuid,
        cwd.clone(),
        request.additional_directories.clone(),
        permission,
        summary.approvals,
        crate::host::Opening::Hydrate,
        session_lock,
    )
    .await
    {
        Ok(runtime) => runtime,
        Err(error) => {
            return responder.respond_with_error(build_failure_error(
                "failed to build session runtime",
                &error,
                state.shared.relay_provider_errors(),
            ));
        }
    };

    // Written after the build, for the reason `session/load` gives at length: the builder's own
    // refusals arrive after this handler's, and a resume refused for one of them had already moved
    // the session's `cwd` and roots.
    if let Err(error) = crate::host::record_session_change(
        &state.shared.store,
        session_uuid,
        reopened_workspace(&summary, &cwd, &request.additional_directories),
    )
    .await
    {
        runtime
            .resident
            .release(state.shared.mcp_manager.as_ref())
            .await;
        return responder.respond_with_error(acp_error_for(&error, false));
    }

    let permission = runtime.resident.cells().permission.clone();
    let frontend = Arc::clone(&runtime.frontend);
    // History already carries the first user message, so the title is known; push it once now,
    // sharing the flag with the entry so a later prompt won't re-emit it.
    let title_sent = Arc::new(std::sync::atomic::AtomicBool::new(false));
    maybe_emit_session_title(
        &cx,
        &session_id,
        &title_sent,
        &*runtime.resident.conversation.lock().await,
    );
    let entry = SessionEntry {
        resident: runtime.resident,
        title_sent,
        frontend,
    };
    state.sessions.write().await.insert(session_id_str, entry);

    let modes = build_mode_state(&permission);
    let config_options = build_config_options(&state.shared, &permission, Some(session_uuid)).await;
    emit_available_commands(&cx, &session_id, &state.shared.skills).await;

    responder.respond(
        ResumeSessionResponse::new()
            .modes(modes)
            .config_options(config_options),
    )
}
/// Delete a fork whose runtime could not be built, so a failed `session/fork` doesn't leave a full
/// copy of the conversation in the database under an id the client was never told.
///
/// `session/new` has the same failure shape but leaves its row behind; there the orphan is an empty
/// session, whereas a fork's is an entire transcript, and an auto-retrying client would multiply it
/// on every attempt. Best-effort: a failed cleanup is worth a warning, not a second error replacing
/// the one the client needs to see.
pub(super) async fn discard_failed_fork(state: &Arc<ServerState>, session_uuid: uuid::Uuid) {
    discard_unusable_session(state, session_uuid, "session/fork").await;
}

/// Delete a row this door created and then could not put a session behind. Left standing, it
/// shows in every listing forever with no conversation and no way to have been used.
///
/// Through the guarded door. The row's lock has gone by the time this runs, into the builder that
/// dropped it or never taken at all, and `meka -c` in another process can take the newest root in
/// that gap; the plain door would then cascade the row away under a session that process is
/// running. The guarded one refuses, and the row stays with whoever holds it.
pub(super) async fn discard_unusable_session(
    state: &Arc<ServerState>,
    session_uuid: uuid::Uuid,
    door: &str,
) {
    match state
        .shared
        .store
        .delete_session_unless_attached(session_uuid)
        .await
    {
        Ok(_) => tracing::info!("{door}: discarded unusable session {session_uuid}"),
        Err(error) => {
            tracing::warn!("{door}: failed to discard unusable session {session_uuid}: {error}")
        }
    }
}
/// `session/fork`: copy an existing session's conversation into a new session and adopt the copy as
/// active. The source is left open and untouched.
///
/// **UNSTABLE** in the protocol: gated behind the SDK's `unstable_session_fork` feature and subject
/// to change.
///
/// Shaped like [`handle_resume_session`], with one difference that matters: ACP models fork as a
/// session-*creation* request, so `cwd` and `additionalDirectories` come from the request rather
/// than the source session, and may legitimately differ from it.
pub(super) async fn handle_fork_session(
    state: Arc<ServerState>,
    request: ForkSessionRequest,
    responder: agent_client_protocol::Responder<ForkSessionResponse>,
    cx: ConnectionTo<Client>,
) -> Result<(), agent_client_protocol::Error> {
    let source_uuid = match parse_session_id(request.session_id.0.as_ref()) {
        Ok(uuid) => uuid,
        Err(error) => return responder.respond_with_error(error),
    };

    let cwd = match crate::workspace::accept_cwd(&request.cwd) {
        Ok(cwd) => cwd,
        Err(error) => return responder.respond_with_error(acp_error_for(&error, false)),
    };
    if let Err(error) = validate_additional_roots(&request.additional_directories) {
        return responder.respond_with_error(error);
    }

    // Before the copy, for the reason `crate::host::http::handlers::sessions::fork_session` refuses
    // there: a fork of a sub-agent is a sibling under the same parent, so the copy is a sub-agent
    // too and `build_session_runtime` below refuses to build it. That refusal is correct but
    // arrives far too late to say anything useful: it is reported as an *internal* error, for
    // something the caller got wrong, and it names the copy's id, which the client has never seen
    // and which `discard_failed_fork` has already deleted by the time it reads it.
    //
    // `spawn_terms` and not the parent link, so the same rows the builders refuse are refused
    // here, an imported sub-agent among them.
    match state.shared.store.spawn_terms(source_uuid).await {
        Ok(Some(terms)) => {
            return responder.respond_with_error(invalid_params_error(match terms.parent {
                Some(parent) => format!(
                    "session {source_uuid} is a sub-agent of session {parent}, so a copy of it \
                     cannot be driven; continue it with `agent_followup` there"
                ),
                None => format!(
                    "session {source_uuid} is a sub-agent whose parent is not in this store, so a \
                     copy of it cannot be driven"
                ),
            }));
        }
        // Not this door's refusal to make: `fork_session_locked` answers an unknown id below.
        Ok(None) => {}
        Err(error) => return responder.respond_with_error(acp_error_for(&error, false)),
    }

    // Locked before the copy's row exists. Otherwise a sweep between the two takes the copy, this
    // handler locks the vanished id successfully, `load_events` returns empty, and the editor is
    // handed a silently blank fork. See `Store::fork_session_locked`, which also holds the source
    // still: a source this editor has open is this process's own, so the store is told rather than
    // asked, and one it does not have open is probed, so a session another process is mid-turn on
    // is refused rather than copied half-written.
    //
    // Told only once it stands still. A turn persists its prompt before the provider answers, so
    // a source this editor is prompting, copied mid-turn, ends on a prompt nothing answered: the
    // very copy the probe refuses another process. `try_lock` as the profile option does, since
    // blocking on a turn here is the deadlock `session/close` documents. Held across the copy, so
    // the idle sweep cannot release the entry, and its file lock with it, between this answer and
    // the copy.
    let resident = state
        .sessions
        .read()
        .await
        .get(request.session_id.0.as_ref())
        .cloned();
    let (source_still, source_lock) = match &resident {
        Some(entry) => match entry.conversation.try_lock() {
            Ok(still) => (Some(still), crate::store::SourceLock::HeldByCaller),
            Err(_) => {
                return responder.respond_with_error(acp_error_for(
                    &MekaError::TurnInFlight {
                        doing: "fork the session",
                    },
                    false,
                ));
            }
        },
        None => (None, crate::store::SourceLock::Probe),
    };
    let (forked, forked_lock) = match state
        .shared
        .store
        .fork_session_locked(
            source_uuid,
            crate::store::ForkOverrides {
                cwd: Some(cwd.clone()),
                // Always `Some`: per the spec an omitted or empty list means "no additional roots
                // are activated", which is an override to none rather than a request to inherit.
                additional_roots: Some(request.additional_directories.clone()),
                token_id: None,
            },
            source_lock,
        )
        .await
    {
        Ok(Some(pair)) => pair,
        Ok(None) => {
            return responder.respond_with_error(acp_error_for(
                &MekaError::SessionNotFound(source_uuid),
                false,
            ));
        }
        Err(error) => return responder.respond_with_error(acp_error_for(&error, false)),
    };
    // The copy has committed, so the source is free to move again while the copy's runtime is
    // built below.
    drop(source_still);

    let session_uuid = forked.id;
    let session_id_str = session_uuid.to_string();
    let session_id: SessionId = session_id_str.clone().into();

    let session_lock = match forked_lock {
        Ok(lock) => lock,
        Err(error) => {
            discard_failed_fork(&state, session_uuid).await;
            return responder.respond_with_error(acp_error_for(&error, false));
        }
    };

    // The copy's row carries the source's level and switch; the runtime is seeded from them so the
    // fork runs under what it copied, with a level this configuration no longer enables dropping to
    // the default as on every other door.
    let (permission, approvals) = match state.shared.store.session_info(session_uuid).await {
        Ok(Some(copied)) => (
            state
                .shared
                .config
                .enabled_permissions
                .admit_recorded(copied.permission, &format!("session {session_uuid}"))
                .unwrap_or(state.shared.config.permission),
            copied.approvals,
        ),
        Ok(None) => {
            tracing::warn!("session/fork: the copy {session_uuid} has no row to read");
            (
                state.shared.config.permission,
                state.shared.config.approvals,
            )
        }
        Err(error) => {
            tracing::warn!("session/fork: failed to read the copy's row {session_uuid}: {error}");
            (
                state.shared.config.permission,
                state.shared.config.approvals,
            )
        }
    };
    let runtime = match build_session_runtime(
        &state.shared,
        state.link(cx.clone(), session_id.clone()),
        session_uuid,
        cwd,
        request.additional_directories.clone(),
        permission,
        approvals,
        crate::host::Opening::Hydrate,
        session_lock,
    )
    .await
    {
        Ok(runtime) => runtime,
        Err(error) => {
            discard_failed_fork(&state, session_uuid).await;
            return responder.respond_with_error(build_failure_error(
                "failed to build session runtime",
                &error,
                state.shared.relay_provider_errors(),
            ));
        }
    };

    if !request.mcp_servers.is_empty() {
        let provided = request.mcp_servers.len();
        tracing::warn!(
            "session/fork: ignoring {provided} client-provided mcpServers; MCP servers come from \
             config.toml"
        );
    }

    let permission = runtime.resident.cells().permission.clone();
    let frontend = Arc::clone(&runtime.frontend);
    // The copied history already carries the first user message, so the title is known now.
    let title_sent = Arc::new(std::sync::atomic::AtomicBool::new(false));
    maybe_emit_session_title(
        &cx,
        &session_id,
        &title_sent,
        &*runtime.resident.conversation.lock().await,
    );
    let entry = SessionEntry {
        resident: runtime.resident,
        title_sent,
        frontend,
    };
    state.sessions.write().await.insert(session_id_str, entry);

    tracing::info!("session/fork: {source_uuid} forked into {session_uuid}");

    let modes = build_mode_state(&permission);
    let config_options = build_config_options(&state.shared, &permission, Some(session_uuid)).await;
    emit_available_commands(&cx, &session_id, &state.shared.skills).await;

    responder.respond(
        ForkSessionResponse::new(session_id)
            .modes(modes)
            .config_options(config_options),
    )
}
/// `session/close`: remove a session from the active map. Cancels any in-flight prompt for that
/// session before removing it from the map so the agent loop unwinds. Detaches the session's tool
/// registry from the MCP manager so live `tools/list_changed` updates stop targeting it.
pub(super) async fn handle_close_session(
    state: Arc<ServerState>,
    request: CloseSessionRequest,
    responder: agent_client_protocol::Responder<CloseSessionResponse>,
) -> Result<(), agent_client_protocol::Error> {
    let session_id_str = request.session_id.0.as_ref().to_string();
    let session_uuid = match parse_session_id(&session_id_str) {
        Ok(uuid) => uuid,
        Err(error) => return responder.respond_with_error(error),
    };
    let removed = state.sessions.write().await.remove(&session_id_str);
    let Some(entry) = removed else {
        return responder.respond_with_error(acp_error_for(
            &MekaError::SessionNotFound(session_uuid),
            false,
        ));
    };
    // Fire cancel via the sibling cell first; it never blocks on the conversation mutex (which an
    // in-flight prompt may hold for the whole turn).
    entry.cancel.cancel();
    // Waits on the conversation mutex, which an in-flight prompt holds for the whole turn. That is
    // safe only because the handler is `cx.spawn`ed: on the dispatch loop it would starve the very
    // response the turn is waiting for. The cancel above does not make the wait short either --
    // `read_file` and the `fs/*` delegates do not observe the token -- so this genuinely blocks
    // until the turn ends, off the loop, which is the correct place to do it.
    drop(entry.conversation.lock().await);
    let stopped = entry.release(state.shared.mcp_manager.as_ref()).await;
    if stopped > 0 {
        tracing::info!("session/close signaled {stopped} running background task(s) to stop");
    }
    // The inner Arcs live until any in-flight prompt's lock guard drops; the agent loop sees the
    // cancel and returns. The map entry is gone, so further requests for this session id error.
    drop(entry);
    responder.respond(CloseSessionResponse::new())
}
/// The refusal for a mode id naming no enabled level, listing the levels the picker offers.
fn unknown_level(given: &str, permission: &crate::permission::SharedPermission) -> String {
    crate::text::unknown_name(
        "permission level",
        given,
        permission.enabled().iter().map(|level| level.to_string()),
    )
}
/// `session/set_mode`: switch the active session to a different permission level. Validates against
/// the configured enabled set; levels outside it become JSON-RPC errors rather than silently
/// failing. On success, emit `current_mode_update` so every connected client (the picker UI)
/// reflects the new state.
pub(super) async fn handle_set_session_mode(
    state: Arc<ServerState>,
    request: SetSessionModeRequest,
    responder: agent_client_protocol::Responder<SetSessionModeResponse>,
) -> Result<(), agent_client_protocol::Error> {
    let session_id_str = request.session_id.0.as_ref().to_string();
    let session_uuid = match parse_session_id(&session_id_str) {
        Ok(uuid) => uuid,
        Err(error) => return responder.respond_with_error(error),
    };
    let entry = {
        let sessions = state.sessions.read().await;
        match sessions.get(&session_id_str) {
            Some(entry) => {
                entry.touch();
                entry.clone()
            }
            None => {
                return responder.respond_with_error(acp_error_for(
                    &MekaError::SessionNotFound(session_uuid),
                    false,
                ));
            }
        }
    };
    let permission = match parse_mode_id(request.mode_id.0.as_ref()) {
        Some(permission) => permission,
        None => {
            return responder.respond_with_error(invalid_params_error(unknown_level(
                request.mode_id.0.as_ref(),
                &entry.cells().permission,
            )));
        }
    };
    // No runtime mutex acquired: `SharedPermission` is `Arc<AtomicU8>` and the frontend cell holds
    // the connection. A user's mid-turn level change takes effect on the next tool-call permission
    // probe without waiting for the in-flight turn to finish.
    if let Err(error) = entry.cells().permission.try_set(permission) {
        return responder.respond_with_error(acp_error_for(&error, false));
    }
    // Persisted alongside the in-memory cell, as `PATCH /v1/sessions/{id}` does: the scheduler's
    // live gate re-check reads the session *row* and nothing else, and `session/list` reports it.
    //
    // In-memory first: the change the user asked for has already taken effect on the next tool
    // call, and what a failed write costs is `record_session_change`'s to decide.
    if let Err(error) = crate::host::record_session_change(
        &state.shared.store,
        session_uuid,
        crate::store::SessionPatch {
            permission: Some(permission),
            ..Default::default()
        },
    )
    .await
    {
        return responder.respond_with_error(acp_error_for(&error, false));
    }
    // The canonical id for the level that was set, not the string the client sent: an editor ticks
    // its mode picker by comparing `currentMode` against the ids advertised in `availableModes`.
    send_session_update(
        &entry.frontend.connection,
        &entry.frontend.session_id,
        SessionUpdate::CurrentModeUpdate(CurrentModeUpdate::new(mode_id_for(permission))),
    );
    // The same change again for clients reading `configOptions` rather than `modes`. Both are
    // advertised, so both have to be kept current or the two pickers disagree about the level the
    // session is at.
    emit_config_options(&state, &entry, Some(session_uuid)).await;
    responder.respond(SetSessionModeResponse::new())
}
/// `session/set_config_option`: the `configOptions` counterpart to `session/set_mode`, which is how
/// a client changes the session's profile or its approvals switch.
///
/// The permission option does the same three things `session/set_mode` does -- set the cell, record
/// the row, push a `current_mode_update` -- so a client driving either one gets the same result and
/// both pickers agree. It does not *call* that handler: this one answers with the refreshed
/// `configOptions` list rather than pushing it, which is the response shape the method has.
pub(super) async fn handle_set_session_config_option(
    state: Arc<ServerState>,
    request: SetSessionConfigOptionRequest,
    responder: agent_client_protocol::Responder<SetSessionConfigOptionResponse>,
) -> Result<(), agent_client_protocol::Error> {
    let session_id_str = request.session_id.0.as_ref().to_string();
    let session_uuid = match parse_session_id(&session_id_str) {
        Ok(uuid) => uuid,
        Err(error) => return responder.respond_with_error(error),
    };
    let entry = {
        let sessions = state.sessions.read().await;
        match sessions.get(&session_id_str) {
            Some(entry) => {
                entry.touch();
                entry.clone()
            }
            None => {
                return responder.respond_with_error(acp_error_for(
                    &MekaError::SessionNotFound(session_uuid),
                    false,
                ));
            }
        }
    };
    // The switch is a boolean where the two pickers take a value id, so it is read first and on
    // its own.
    if request.config_id.0.as_ref() == APPROVALS_CONFIG_ID {
        let Some(approvals) = request.value.as_bool() else {
            return responder.respond_with_error(invalid_params_error(
                "the approvals option takes a boolean value",
            ));
        };
        entry.cells().permission.set_approvals(approvals);
        // The row, for the same readers as the level: a resume from any surface.
        if let Err(error) = crate::host::record_session_change(
            &state.shared.store,
            session_uuid,
            crate::store::SessionPatch {
                approvals: Some(approvals),
                ..Default::default()
            },
        )
        .await
        {
            return responder.respond_with_error(acp_error_for(&error, false));
        }
        let options =
            build_config_options(&state.shared, &entry.cells().permission, Some(session_uuid))
                .await;
        return responder.respond(SetSessionConfigOptionResponse::new(options));
    }
    let Some(value) = request.value.as_value_id() else {
        return responder.respond_with_error(invalid_params_error(
            "the permission and profile options take a string value",
        ));
    };
    let value = value.0.as_ref().to_string();

    match request.config_id.0.as_ref() {
        PERMISSION_CONFIG_ID => {
            let Some(permission) = parse_mode_id(&value) else {
                return responder.respond_with_error(invalid_params_error(unknown_level(
                    &value,
                    &entry.cells().permission,
                )));
            };
            if let Err(error) = entry.cells().permission.try_set(permission) {
                return responder.respond_with_error(acp_error_for(&error, false));
            }
            // The row matters here for the same reason it does in `session/set_mode`: a scheduled
            // gate is re-checked against it, and `session/list` reports it.
            if let Err(error) = crate::host::record_session_change(
                &state.shared.store,
                session_uuid,
                crate::store::SessionPatch {
                    permission: Some(permission),
                    ..Default::default()
                },
            )
            .await
            {
                return responder.respond_with_error(acp_error_for(&error, false));
            }
            // `modes` is still advertised, so its picker has to hear about a change made through
            // the other one.
            send_session_update(
                &entry.frontend.connection,
                &entry.frontend.session_id,
                SessionUpdate::CurrentModeUpdate(CurrentModeUpdate::new(mode_id_for(permission))),
            );
        }
        PROFILE_CONFIG_ID => {
            // The profile as configured. Picking one through the picker moves the session to that
            // bundle entire, which is the only thing a provider selection can mean now that a
            // profile is indivisible.
            let resolved = match crate::host::resolve_profile_switch(&state.shared, &value).await {
                Ok(resolved) => resolved,
                Err(error) => {
                    return responder.respond_with_error(build_failure_error(
                        "failed to resolve the profile",
                        &error,
                        state.shared.relay_provider_errors(),
                    ));
                }
            };
            // Unlike permission, which lives in an atomic the tool-call path reads live, the
            // provider is owned by the `Agent` behind the runtime mutex. An in-flight prompt holds
            // that mutex for the whole turn, and blocking the dispatch loop on it is the deadlock
            // `session/close` documents; `try_lock` turns the wait into a refusal, the one the
            // prompt handler gives a second prompt. Taken ahead of the row write, so a switch
            // refused for a turn in flight leaves the row where it was.
            let Ok(_turn_exclusive) = entry.conversation.try_lock() else {
                return responder.respond_with_error(acp_error_for(
                    &MekaError::TurnInFlight {
                        doing: "switch profile",
                    },
                    false,
                ));
            };
            // The row first, so a write that fails leaves the session on the profile it was
            // already recorded against rather than on one no later resume would resolve.
            if let Err(error) =
                crate::host::record_profile_switch(&state.shared.store, session_uuid, &resolved)
                    .await
            {
                return responder.respond_with_error(build_failure_error(
                    "failed to record the profile",
                    &error,
                    state.shared.relay_provider_errors(),
                ));
            }
            entry.agent.set_provider(resolved);
        }
        unknown => {
            return responder.respond_with_error(invalid_params_error(crate::text::unknown_name(
                "configuration option",
                unknown,
                [PERMISSION_CONFIG_ID, PROFILE_CONFIG_ID, APPROVALS_CONFIG_ID],
            )));
        }
    }

    let options =
        build_config_options(&state.shared, &entry.cells().permission, Some(session_uuid)).await;
    responder.respond(SetSessionConfigOptionResponse::new(options))
}
/// Build a fresh [`BuiltSession`] from the process-wide [`crate::host::SharedDeps`]. Called from
/// `session/new`, `session/load`, `session/resume` and `session/fork`. Each follows the same shape:
/// 1. Construct the per-session `AcpFrontend` bound to this connection + session id.
/// 2. Build a per-session `SharedPermission` cell seeded from config defaults.
/// 3. Build the per-session `Agent` via [`crate::host::build_session_agent`], which also attaches
///    its registry to the MCP manager.
/// 4. Bundle the agent, the conversation and the frontend.
#[allow(
    clippy::too_many_arguments,
    reason = "each door passes what it alone knows; a parameter struct would be built once per door"
)]
pub(super) async fn build_session_runtime(
    shared: &Arc<crate::host::SharedDeps>,
    link: super::frontend::AcpClientLink,
    session_uuid: uuid::Uuid,
    cwd_path: PathBuf,
    additional_roots: Vec<PathBuf>,
    // The level and switch the session starts at: the configured defaults for `session/new`, and
    // what the row records for a session reopened or copied, so the runtime is seeded with the
    // level it runs at rather than corrected after the fact.
    permission: Permission,
    approvals: bool,
    opening: crate::host::Opening,
    session_lock: crate::fs::FileLock,
) -> anyhow::Result<BuiltSession> {
    let cwd = SharedCwd::new(cwd_path);
    // `--writable-root` is a flag on the process the editor launched, so it belongs to every
    // session that process serves, not only the REPL's. Merged into the live handle and
    // deliberately not into the persisted row: the row is what `session/load` hands back to the
    // client as its `additionalDirectories`, and reporting a folder the client never asked for
    // would misdescribe its own request to it.
    let roots = SharedRoots::new(
        additional_roots
            .into_iter()
            .chain(shared.config.request.writable_roots.iter().cloned())
            .collect(),
    );
    let permission = SharedPermission::new(permission, shared.config.enabled_permissions)
        .with_approvals(approvals);

    // Shared with the agent (adopted inside `build_session_agent`) so the frontend can read the
    // current context occupancy when emitting `usage_update`.
    let context_tokens = Arc::new(std::sync::atomic::AtomicU64::new(0));
    // The window the same `usage_update` divides by, made here for the same reason: the frontend
    // exists before the agent and has to hold the cell the agent publishes into, rather than a copy
    // something has to remember to re-store beside every provider switch.
    let context_window = Arc::new(std::sync::atomic::AtomicU64::new(0));
    // Created here rather than beside the `SessionEntry` so the frontend and the entry share one
    // cell: the entry's cancel handler writes it, the frontend's client round-trips read it.
    let cancel = crate::host::CancelCell::default();
    let acp_frontend = Arc::new(AcpFrontend::new(
        link,
        cwd.clone(),
        Arc::clone(&context_tokens),
        Arc::clone(&context_window),
        cancel.clone(),
    ));
    let frontend: Arc<dyn Frontend> = acp_frontend.clone();

    let resident = crate::host::open_session(
        shared,
        session_uuid,
        crate::host::SessionSpec {
            session_id: Some(session_uuid),
            permission: permission.clone(),
            frontend,
            cwd: cwd.clone(),
            roots: roots.clone(),
            context_tokens: Arc::clone(&context_tokens),
            context_overhead: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            context_window,
        },
        opening,
        session_lock,
        cancel.clone(),
    )
    .await?;

    Ok(BuiltSession {
        resident,
        frontend: acp_frontend,
    })
}

#[cfg(test)]
mod tests {
    /// The discard deletes through the guarded door. What that defends needs a second process: the
    /// row's lock has gone by the time it runs, and `meka -c` there can take the newest root in
    /// the gap, after which the plain door cascaded the row away under a session that process was
    /// running. Asserted against the source, as the HTTP twin
    /// `a_failed_build_rolls_its_row_back_through_the_guarded_door` is; the guarded door's own
    /// refusal is `deleting_a_session_another_process_holds_is_refused`.
    #[test]
    fn an_unusable_session_is_discarded_through_the_guarded_door() {
        let source = include_str!("session.rs").replace("\r\n", "\n");
        let body = source
            .split("async fn discard_unusable_session(")
            .nth(1)
            .expect("the discard this assertion is about")
            .split("\n}\n")
            .next()
            .expect("splitting always yields a first part");
        assert!(
            body.contains("discarded unusable session"),
            "the scanned region no longer covers the discard, so this assertion proves nothing"
        );
        assert!(
            body.contains("delete_session_unless_attached("),
            "the discard must delete through the guarded door"
        );
        assert!(
            !body.contains(".delete_session("),
            "and must not take the plain door, which deletes under whoever holds the row"
        );
    }
}
