//! `meka acp` subcommand. Speaks the Agent Client Protocol (ACP) on stdio so editor / web /
//! messenger clients can drive a meka turn end to end.
//!
//! The advertised capability surface, the delegation rules and the wire shapes are documented in
//! `docs/book/src/usage/acp.md`.
//!
//! **`execute_command` is never delegated to the client's `terminal/*`**, whatever it advertises
//! and whatever the permission level. meka owns the process so its Landlock / bwrap / sandbox-exec
//! / Low-Integrity jail, env scrub, cwd resolution and process-group kill keep applying; the
//! client's terminal offers no equivalent, so routing through it would run a command unsandboxed at
//! a level meka treats as sandboxed.
//!
//! Any number of sessions coexist in one process, each with its own cwd, permission cell,
//! conversation, cancellation token, `Agent` and `AcpFrontend`, over process-wide dependencies held
//! by `Arc`. Nothing serializes turns, so two `session/prompt` calls run in parallel. A sub-agent
//! reaches the parent's client through [`crate::frontend::PermissionForwardingFrontend`], so its
//! permission prompts and fs delegates surface in the parent session's editor UI.

mod elicitation;
pub(crate) mod frontend;
mod prompt;
mod schedule;
mod session;

use std::{
    path::PathBuf,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use agent_client_protocol::{
    Agent as AcpAgentRole, ByteStreams, Client, ConnectionTo,
    schema::v1::{
        AgentCapabilities, AvailableCommand, AvailableCommandInput, AvailableCommandsUpdate,
        CancelNotification, ClientCapabilities, CloseSessionRequest, CloseSessionResponse,
        ConfigOptionUpdate, ContentBlock, ContentChunk, CurrentModeUpdate, Diff, EmbeddedResource,
        EmbeddedResourceResource, ForkSessionRequest, ForkSessionResponse, ImageContent,
        Implementation, InitializeRequest, InitializeResponse, ListSessionsRequest,
        ListSessionsResponse, LoadSessionRequest, LoadSessionResponse, NewSessionRequest,
        NewSessionResponse, PermissionOption, PermissionOptionKind, Plan, PlanEntry,
        PlanEntryPriority, PlanEntryStatus, PromptCapabilities, PromptRequest, PromptResponse,
        ReadTextFileRequest, RequestPermissionOutcome, RequestPermissionRequest,
        ResumeSessionRequest, ResumeSessionResponse, SessionAdditionalDirectoriesCapabilities,
        SessionCapabilities, SessionCloseCapabilities, SessionConfigOption,
        SessionConfigOptionCategory, SessionConfigSelectOption, SessionConfigValueId,
        SessionForkCapabilities, SessionId, SessionInfo, SessionInfoUpdate,
        SessionListCapabilities, SessionMode, SessionModeId, SessionModeState, SessionNotification,
        SessionResumeCapabilities, SessionUpdate, SetSessionConfigOptionRequest,
        SetSessionConfigOptionResponse, SetSessionModeRequest, SetSessionModeResponse, StopReason,
        ToolCall, ToolCallContent, ToolCallLocation, ToolCallStatus, ToolCallUpdate,
        ToolCallUpdateFields, ToolKind, UnstructuredCommandInput, Usage, UsageUpdate,
        WriteTextFileRequest,
    },
};
use async_trait::async_trait;
use futures::io::AsyncRead;
use tokio_util::{
    compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt},
    sync::CancellationToken,
};

use self::{frontend::*, prompt::*, session::*};
use crate::{
    agent::Agent,
    config::ResolvedConfig,
    conversation::{ContentBlock as MekaContentBlock, Conversation, Role, ToolResultContent},
    error::MekaError,
    frontend::{
        Frontend, FrontendError, FrontendEvent, PermissionOutcome, PermissionRequest,
        ToolOutputMetadata,
    },
    mcp,
    permission::{Permission, SharedPermission},
    skills::SkillCache,
    store::Store,
    todo::{TodoItem, TodoStatus},
    workspace::{SharedCwd, SharedRoots, resolve_against_cwd},
};

/// Build a JSON-RPC `InvalidParams` error (`-32602`) with a free-form human-readable message in the
/// `data` field. Mirrors [`agent_client_protocol::util::internal_error`] but for the
/// input-validation cases (unknown sessionId, malformed UUID, unsupported level, non-text content).
/// Clients can rely on the JSON-RPC code to distinguish "bad input" from "server failure".
fn invalid_params_error(message: impl ToString) -> agent_client_protocol::Error {
    agent_client_protocol::Error::invalid_params().data(message.to_string())
}

/// The JSON-RPC error a [`MekaError`] becomes on this host, decided once for every handler.
///
/// A refusal the caller can act on is `InvalidParams`: a session that is not there, a turn already
/// holding it, a level or profile this configuration does not have, an empty prompt, a sub-agent's
/// id, a session another process holds, a request over the profile's own size ceiling, and the
/// configuration and usage refusals that name their remedy. Their `Display` is the `data`, so no
/// handler rewrites the sentence.
///
/// **Everything else is withheld on the same terms `ProblemDetail::for_error` withholds it**, which
/// is why this takes `relay_provider_errors` rather than reading it: a `.into()` that silently
/// picked a policy is how one surface of a deployment configured to withhold goes on publishing.
/// The three classes and why each is what it is:
///
/// - An upstream's own response body, and the `format_reqwest_error` chain that stands in for one
///   when nothing answered, travel only when the operator said so. That text can name the
///   *operator's* account with the provider and its rate-limit posture, and an editor is a client
///   like any other. Bounded by [`crate::error::bounded_upstream_body`], the same bound the HTTP
///   host applies, since a misconfigured `base_url` can answer with megabytes.
/// - An MCP connector's reason never travels. It is meka's own subprocess text and has carried a
///   spawn failure complete with its command line and path; the server *names* do travel, which is
///   the part an operator acts on.
/// - `Database` and `Io` never travel: they name meka's own store path or the operator's
///   filesystem. Logged, and answered with the sentence that says where the detail went.
///
/// Every arm logs what it withholds, so nothing is lost, only moved.
fn acp_error_for(error: &MekaError, relay_provider_errors: bool) -> agent_client_protocol::Error {
    // Closured rather than repeated per arm, for the reason `for_error`'s `attach` is: the arms
    // must not drift into disagreeing about what relaying means. `sentence` is meka's own and is
    // identical either way, so a client reading only the message is unaffected by the switch.
    let withhold_or_relay = |sentence: &str, upstream: &str| {
        tracing::warn!("{sentence}: {upstream}");
        let data = if relay_provider_errors {
            format!(
                "{sentence}: {}",
                crate::error::bounded_upstream_body(upstream)
            )
        } else {
            sentence.to_string()
        };
        agent_client_protocol::util::internal_error(data)
    };
    match error {
        MekaError::Config(_)
        | MekaError::Usage(_)
        | MekaError::SessionLocked(_)
        | MekaError::SessionNotDrivable(_)
        | MekaError::SessionNotFound(_)
        | MekaError::TurnInFlight { .. }
        | MekaError::EmptyPrompt
        | MekaError::RequestTooLarge(_)
        | MekaError::DisabledLevel { .. }
        | MekaError::ProfileNotConfigured { .. } => invalid_params_error(error),
        // The operator's `ca_cert_file`, proxy or `base_url`. Its own words are for the terminal
        // `meka acp` was started from, not for the editor on the other end of the pipe, which can
        // do nothing with a path in someone else's configuration file.
        MekaError::Installation(message) => {
            tracing::error!("installation error: {message}");
            agent_client_protocol::util::internal_error(
                "meka is misconfigured; the detail is in the meka log",
            )
        }
        MekaError::Provider(message) | MekaError::InvalidRequest(message) => withhold_or_relay(
            "the provider rejected or failed this turn; its response is in the meka log",
            message,
        ),
        MekaError::StreamError(message) | MekaError::RetryableProvider { message, .. } => {
            withhold_or_relay(
                "the provider did not complete this turn; its response is in the meka log",
                message,
            )
        }
        MekaError::ContextOverflow(message) => withhold_or_relay(
            "the conversation exceeds the model's context window and compaction failed to shorten \
             it further; shorten it before retrying",
            message,
        ),
        // The names travel and the reasons do not, which is the policy the arms above state. A
        // reason here is the connector's own text and has carried a spawn failure complete with the
        // command line and its path.
        MekaError::McpTurnGated { servers } => {
            tracing::warn!("mcp gate refused a turn: {error}");
            let names: Vec<&str> = servers.iter().map(|(name, _)| name.as_str()).collect();
            agent_client_protocol::util::internal_error(format!(
                "required MCP server(s) not ready: {}; each server's reason is in the meka log",
                names.join(", ")
            ))
        }
        // A store path, a filesystem path, or a fault nobody classified. The `Display` names
        // meka's own directories either way.
        other => {
            tracing::error!("unhandled agent error reported to the client: {other}");
            agent_client_protocol::util::internal_error(
                "meka could not complete this request; the detail is in the meka log",
            )
        }
    }
}

/// [`acp_error_for`] for a builder or a door that answers in `anyhow`: a [`MekaError`] inside is
/// mapped as itself, and anything else is an `InternalError` naming neither, since an `anyhow`
/// chain from a builder ends in whatever the provider registry or the store said. `context` goes to
/// the log with it. The twin of the HTTP host's `agent_build_problem`.
fn build_failure_error(
    context: &str,
    error: &anyhow::Error,
    relay_provider_errors: bool,
) -> agent_client_protocol::Error {
    match error.downcast_ref::<MekaError>() {
        Some(error) => acp_error_for(error, relay_provider_errors),
        None => {
            tracing::error!("{context}: {error:#}");
            agent_client_protocol::util::internal_error(format!(
                "{context}; the detail is in the meka log"
            ))
        }
    }
}

/// Run meka as an ACP agent over stdio. Returns (and the process then exits) when the client
/// disconnects (stdin EOF) or a termination signal arrives.
pub(crate) async fn run_acp(
    config: ResolvedConfig,
    store: Store,
    mcp_manager: Option<Arc<mcp::McpClientManager>>,
) -> anyhow::Result<()> {
    // The process default profile's vision flag, for the `image` prompt capability `initialize`
    // advertises before any session exists. Whether a `session/prompt` admits an image is the
    // session's own answer, `ResidentSession::accepts_images`, asked on the prompt.
    let vision = config.vision;

    // Build process-wide shared deps once. Sessions hold an `Arc<SharedDeps>` and read fields by
    // reference; no work happens here that needs to be re-run per session.
    let shared =
        Arc::new(crate::host::build_shared_deps(Arc::new(config), store, mcp_manager).await?);
    // A host that creates sessions on the operator's behalf needs a profile to create them on, and
    // is refused up front rather than at the first `session/new`. `-c` / `-r` are refused for both
    // hosts precisely so a resume cannot make this exception apply here.
    shared.default_profile()?;

    let client_state = SharedClientState::default();
    let transport_dead = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let state = Arc::new(ServerState::new(
        Arc::clone(&shared),
        client_state.clone(),
        transport_dead,
    ));

    // Fires jobs for whichever sessions the editor currently has open; see `acp::schedule`. Aborted
    // on the way out so a scheduled turn cannot outlive the connection it would report to.
    let scheduler_handle = schedule::spawn(Arc::clone(&state));
    let _scheduler_guard = AbortOnDrop(scheduler_handle);

    // Reports background-task outcomes into whichever sessions the editor has open, on the same
    // terms and with the same abort-on-drop lifetime.
    let background_handle = schedule::spawn_background_poller(Arc::clone(&state));
    let _background_guard = AbortOnDrop(background_handle);

    // Releases sessions the editor opened and never closed; see `spawn_idle_session_sweep`.
    let idle_handle = spawn_idle_session_sweep(Arc::clone(&state));
    let _idle_guard = AbortOnDrop(idle_handle);

    // Observe stdin EOF so the connection shuts down when the client disconnects (or the parent
    // dies). The connection future does not resolve on idle EOF by itself, so wrap the incoming
    // side; `acp_run_until_disconnect` (the `connect_with` closure below) waits on this token.
    // tokio stdio + `tokio_util::compat` provide the `futures::io` byte streams the transport wants
    // without pulling in the `blocking` crate.
    let stdin_eof = CancellationToken::new();
    let transport = ByteStreams::new(tokio::io::stdout().compat_write(), EofSignalingRead {
        inner: tokio::io::stdin().compat(),
        eof: stdin_eof.clone(),
    });

    let acp_result = AcpAgentRole
        .builder()
        .name("meka")
        .on_receive_request(
            {
                let client_state = client_state.clone();
                async move |request: InitializeRequest, responder, _cx| {
                    // Stash the client's advertised capabilities (so `AcpFrontend`'s delegate_*
                    // methods can gate on them) and the client's self-identifying `Implementation`
                    // (logged here, available for diagnostics elsewhere). Both are small clones.
                    let client = describe_client(request.client_info.as_ref());
                    tracing::info!("ACP client connected: {client}");
                    client_state.record_initialize(
                        request.client_capabilities.clone(),
                        request.client_info.clone(),
                    );

                    // Advertise the optional session methods. Each marker is an empty struct;
                    // presence signals support.
                    let session_caps = SessionCapabilities::new()
                        .additional_directories(Some(
                            SessionAdditionalDirectoriesCapabilities::new(),
                        ))
                        .list(Some(SessionListCapabilities::new()))
                        .resume(Some(SessionResumeCapabilities::new()))
                        .fork(Some(SessionForkCapabilities::new()))
                        .close(Some(SessionCloseCapabilities::new()));
                    // meka accepts `text`, `resource_link`, and embedded `resource` (@-mention)
                    // blocks in `session/prompt`, so `embedded_context` is advertised true. `image`
                    // follows the process default profile's `vision` flag; a session on another
                    // profile answers for itself at `session/prompt`. `audio` stays false. Each
                    // field is set explicitly so the contract is visible in the initialize response
                    // and a future SDK default change cannot quietly flip it.
                    //
                    // `mcp_capabilities` is omitted: meka sources MCP servers from its own config
                    // file and ignores `session/new`'s `mcpServers`, so advertising `{ http: true,
                    // sse: true }` would be false.
                    let capabilities = AgentCapabilities::new()
                        .load_session(true)
                        .session_capabilities(session_caps)
                        .prompt_capabilities(
                            PromptCapabilities::new()
                                .image(vision)
                                .audio(false)
                                .embedded_context(true),
                        );
                    // Reject the V0 sentinel explicitly. The schema uses V0 as the "couldn't parse
                    // the requested version" fallback; a clamped `min(V0, LATEST)` would silently
                    // echo it back and let the handshake proceed against a malformed input.
                    if request.protocol_version
                        == agent_client_protocol::schema::ProtocolVersion::V0
                    {
                        return responder.respond_with_error(invalid_params_error(
                            "protocolVersion 0 is the schema's parse-failure sentinel; \
                             specify a supported version",
                        ));
                    }
                    // The requested version when this build supports it, otherwise the latest it
                    // knows: a plain echo tells a newer client that meka speaks a version it does
                    // not.
                    let negotiated = std::cmp::min(
                        request.protocol_version,
                        agent_client_protocol::schema::ProtocolVersion::LATEST,
                    );
                    let response = InitializeResponse::new(negotiated)
                        .agent_capabilities(capabilities)
                        .agent_info(Implementation::new("meka", env!("CARGO_PKG_VERSION")));
                    responder.respond(response)
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |request: NewSessionRequest, responder, cx: ConnectionTo<Client>| {
                    let cwd = match crate::workspace::accept_cwd(&request.cwd) {
                        Ok(cwd) => cwd,
                        Err(error) => {
                            return responder.respond_with_error(acp_error_for(&error, false));
                        }
                    };
                    if let Err(error) = validate_additional_roots(&request.additional_directories) {
                        return responder.respond_with_error(error);
                    }
                    // A new session runs on the host's default. `session/load` reads the one the
                    // session recorded instead, which is what stops a resume moving the
                    // conversation to another provider.
                    let profile = match state.shared.default_profile() {
                        Ok(profile) => profile.to_string(),
                        Err(error) => {
                            return responder.respond_with_error(build_failure_error(
                                "no default profile",
                                &error,
                                state.shared.relay_provider_errors(),
                            ));
                        }
                    };
                    // Created and locked in one step, the lock taken *before* the row exists: a
                    // row committed ahead of its lock is one `meka session delete --all` can
                    // enumerate and sweep out from under this handler. See
                    // `Store::create_session_locked`.
                    let (session_uuid, session_lock) = match state
                        .shared
                        .store
                        .create_session_locked(
                            Some(cwd.clone()),
                            // The level and switch the runtime below is seeded with, on the row
                            // from the start so a scheduled gate and a resume read them there.
                            state.shared.config.permission.to_string(),
                            state.shared.config.approvals,
                            None,
                            None,
                            profile,
                        )
                        .await
                    {
                        Ok((created, lock)) => (created.id, lock),
                        Err(error) => {
                            return responder.respond_with_error(acp_error_for(&error, false));
                        }
                    };
                    // Persist the roots so `session/list` can report the workspace shape this
                    // session was opened with. Only when there are any: the row starts with none.
                    if !request.additional_directories.is_empty()
                        && let Err(error) = crate::host::record_session_change(
                            &state.shared.store,
                            session_uuid,
                            crate::store::SessionPatch {
                                roots: Some(request.additional_directories.clone()),
                                ..Default::default()
                            },
                        )
                        .await
                    {
                        return responder.respond_with_error(acp_error_for(&error, false));
                    }

                    // The lock taken above, which a second `meka acp` process (or a REPL) needs to
                    // be unable to take. `None` means the claim could not be made at all -- an
                    // unwritable lock directory -- and an editor session that cannot be held alone
                    // is one this host must not open.
                    let session_lock = match session_lock {
                        Ok(lock) => lock,
                        Err(error) => {
                            // The row was committed a moment ago; the HTTP twin rolls its own back
                            // with `SessionRollback`, and a fork with `discard_failed_fork`.
                            discard_unusable_session(&state, session_uuid, "session/new").await;
                            return responder.respond_with_error(acp_error_for(&error, false));
                        }
                    };
                    let session_id_str = session_uuid.to_string();
                    let session_id: SessionId = session_id_str.clone().into();

                    let runtime = match build_session_runtime(
                        &state.shared,
                        state.link(cx.clone(), session_id.clone()),
                        session_uuid,
                        cwd,
                        request.additional_directories.clone(),
                        state.shared.config.permission,
                        state.shared.config.approvals,
                        crate::host::Opening::Fresh,
                        session_lock,
                    )
                    .await
                    {
                        Ok(runtime) => runtime,
                        Err(error) => {
                            // The lock went into the builder and is already released.
                            discard_unusable_session(&state, session_uuid, "session/new").await;
                            return responder.respond_with_error(build_failure_error(
                                "failed to build session runtime",
                                &error,
                                state.shared.relay_provider_errors(),
                            ));
                        }
                    };

                    let permission = runtime.resident.cells().permission.clone();
                    let frontend = Arc::clone(&runtime.frontend);
                    let entry = SessionEntry {
                        resident: runtime.resident,
                        title_sent: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                        frontend,
                    };
                    state.sessions.write().await.insert(session_id_str, entry);

                    if !request.mcp_servers.is_empty() {
                        let provided = request.mcp_servers.len();
                        tracing::warn!(
                            "session/new: ignoring {provided} client-provided mcpServers; MCP \
                             servers come from config.toml"
                        );
                    }

                    // Push the initial skill palette + the configured mode picker so the editor's
                    // UI is populated before the user types their first prompt.
                    let modes = build_mode_state(&permission);
                    let config_options =
                        build_config_options(&state.shared, &permission, Some(session_uuid)).await;
                    emit_available_commands(&cx, &session_id, &state.shared.skills).await;

                    responder.respond(
                        NewSessionResponse::new(session_id)
                            .modes(modes)
                            .config_options(config_options),
                    )
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |request: PromptRequest, responder, cx: ConnectionTo<Client>| {
                    // Counted before the spawn, so a `session/cancel` sent straight after this
                    // prompt finds it pending. Both run on this dispatch loop, so the count is up
                    // before the cancel can look, which is not true of anything the turn itself
                    // does. `None` for an unknown session, which the turn refuses below anyway.
                    let pending = {
                        let sessions = state.sessions.read().await;
                        sessions
                            .get(request.session_id.0.as_ref())
                            .and_then(|entry| entry.admit_turn(None).ok())
                    };
                    let state_for_spawn = Arc::clone(&state);
                    cx.spawn(async move {
                        run_prompt_turn(state_for_spawn, request, responder, pending).await
                    })?;
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |request: LoadSessionRequest, responder, cx: ConnectionTo<Client>| {
                    handle_load_session(Arc::clone(&state), request, responder, cx).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |request: ListSessionsRequest, responder, _cx: ConnectionTo<Client>| {
                    handle_list_sessions(Arc::clone(&state), request, responder).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |request: ResumeSessionRequest, responder, cx: ConnectionTo<Client>| {
                    handle_resume_session(Arc::clone(&state), request, responder, cx).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |request: ForkSessionRequest, responder, cx: ConnectionTo<Client>| {
                    handle_fork_session(Arc::clone(&state), request, responder, cx).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |request: CloseSessionRequest, responder, cx: ConnectionTo<Client>| {
                    // Spawned, exactly like the prompt handler above, because this one waits on a
                    // lock an in-flight turn holds.
                    //
                    // Handler callbacks run on the SDK's dispatch loop, and that loop is also what
                    // routes *responses to meka's own outgoing requests*. A turn blocked on
                    // `fs/read_text_file` cannot release the runtime mutex until its response
                    // arrives, and the response cannot arrive while the loop is parked inside this
                    // handler waiting for that same mutex. Running inline deadlocked every session
                    // in the process, recoverable only by killing the client.
                    let state_for_spawn = Arc::clone(&state);
                    cx.spawn(async move {
                        handle_close_session(state_for_spawn, request, responder).await
                    })?;
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |request: SetSessionModeRequest, responder, _cx: ConnectionTo<Client>| {
                    handle_set_session_mode(Arc::clone(&state), request, responder).await
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                async move |request: SetSessionConfigOptionRequest,
                            responder,
                            cx: ConnectionTo<Client>| {
                    // Spawned rather than run inline for the reason `session/close` spells out:
                    // this one reaches the session runtime, and parking the dispatch loop on
                    // anything a turn holds deadlocks every session in the process.
                    let state_for_spawn = Arc::clone(&state);
                    cx.spawn(async move {
                        handle_set_session_config_option(state_for_spawn, request, responder).await
                    })?;
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            {
                let state = Arc::clone(&state);
                async move |notif: CancelNotification, _cx: ConnectionTo<Client>| {
                    // Cancel fires through the sibling `cancellation` cell on the `SessionEntry`;
                    // we never touch the per-session runtime mutex, which the prompt handler
                    // holds for the duration of the turn.
                    //
                    // A live token takes the cancel and that is the end of it. An empty cell with a
                    // prompt still on its way means the editor canceled a `session/prompt` that
                    // has not reached its handler yet, so latch the signal for that prompt to
                    // apply to its own token. An empty cell with nothing pending has nothing to
                    // stop, and latching there would arm the cancel against whatever the user
                    // submitted next instead.
                    let entry = {
                        let sessions = state.sessions.read().await;
                        sessions.get(notif.session_id.0.as_ref()).cloned()
                    };
                    if let Some(entry) = entry {
                        entry.cancel.cancel();
                    }
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(transport, {
            let state = Arc::clone(&state);
            async move |_cx: ConnectionTo<Client>| acp_run_until_disconnect(state, stdin_eof).await
        })
        .await;

    // The connection has unwound, so nothing will issue another tool call. An editor that quits
    // takes meka's stdio with it but not the grandchildren meka spawned, which is what this closes.
    if let Some(manager) = &state.shared.mcp_manager {
        manager.shutdown_within(crate::mcp::SHUTDOWN_BUDGET).await;
    }

    acp_result.map_err(|error| anyhow::anyhow!("ACP server error: {error}"))
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;

    use super::*;
    use crate::frontend::PermissionOutcome;

    // `AcpFrontend` itself can't be unit-tested (requires a live `ConnectionTo<Client>`);
    // per-session behavior is covered end-to-end in `tests/acp.rs`. The pure helpers below are
    // what this unit-test module owns.

    /// A request the user has already stopped must not wait on the client's answer.
    ///
    /// Every `fs/read_text_file`, `fs/write_text_file` and elicitation is a round trip to an editor
    /// that owes no reply once the turn is canceled, so without the race the stop button left the
    /// turn parked on a request nobody was going to answer.
    #[tokio::test]
    async fn a_canceled_turn_abandons_a_request_the_client_has_not_answered() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();

        // Bounded so a lost race fails the test rather than hanging it: the work below models a
        // client that never answers, so without the race there is nothing to wait for.
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            race_against_cancellation(
                "fs/read_text_file",
                &cancellation,
                std::future::pending::<()>(),
            ),
        )
        .await
        .expect("a canceled turn must not wait on the client");

        let error = outcome.expect_err("a canceled turn must not wait on the client");
        assert!(
            error.is_canceled(),
            "the caller has to be able to tell a stop from a failure: {error}"
        );
    }

    /// The other half: an uncanceled turn must still get its answer, or the race would make every
    /// client round trip fail.
    #[tokio::test]
    async fn a_live_turn_receives_the_clients_answer() {
        let cancellation = CancellationToken::new();

        let outcome =
            race_against_cancellation("fs/read_text_file", &cancellation, async { "contents" })
                .await;

        assert_eq!(
            outcome.expect("a live turn must get its answer"),
            "contents"
        );
    }

    /// What an editor may read of a failed turn, per class.
    ///
    /// "The caller can act on it" and "meka wrote the sentence" are not the same question: an
    /// upstream body names the *operator's* account with the provider, an MCP connector's reason
    /// can carry a spawn command line and its path, a `Database` error names the store, and an
    /// installation fault names a file in someone else's `config.toml`. The HTTP host answers all
    /// four the same way; this is the same policy on the other wire.
    ///
    /// Relaying is asked for throughout, so nothing here passes merely because the switch is off.
    #[test]
    fn a_failed_turn_tells_an_editor_only_what_it_may_act_on() {
        let leaky = "acct-0f3c-operator-only";
        for error in [
            MekaError::McpConnection {
                server_name: "exa".to_string(),
                message: format!("spawn /opt/{leaky}/bin/exa failed"),
            },
            MekaError::Database(format!("unable to open /var/{leaky}/meka.db")),
            MekaError::Io(std::io::Error::other(format!("/srv/{leaky}/skills"))),
            MekaError::Installation(format!("[web].ca_cert_file '/etc/{leaky}/ca.pem'")),
        ] {
            let reported = acp_error_for(&error, true);
            let data = serde_json::to_string(&reported.data).unwrap_or_default();
            assert!(!data.contains(leaky), "{error} reached the client: {data}");
            assert_eq!(
                reported.code,
                agent_client_protocol::ErrorCode::InternalError,
                "{error}"
            );
        }

        // The gate names its servers, because that is the part an operator acts on, and withholds
        // each server's reason for the same reason the HTTP host does.
        let gated = acp_error_for(
            &MekaError::McpTurnGated {
                servers: vec![("exa".to_string(), format!("spawn /opt/{leaky} failed"))],
            },
            true,
        );
        let data = serde_json::to_string(&gated.data).unwrap_or_default();
        assert!(data.contains("exa") && !data.contains(leaky), "{data}");
    }

    /// The upstream's own words travel only when the operator asked for them, and are bounded when
    /// they do.
    ///
    /// The same switch, the same bound and the same withheld sentence as the HTTP host, because
    /// the text is the same text, and a deployment that turns relaying off expects it withheld
    /// everywhere.
    #[test]
    fn the_upstream_body_travels_over_acp_only_when_the_operator_asked_for_it() {
        let leaky = "{\"account_uuid\":\"acct-0f3c\",\"message\":\"quota exceeded\"}";
        for error in [
            MekaError::Provider(leaky.to_string()),
            MekaError::InvalidRequest(leaky.to_string()),
            MekaError::StreamError(leaky.to_string()),
            MekaError::RetryableProvider {
                message: leaky.to_string(),
                retry_after: None,
                server_error_on_completion: false,
            },
            MekaError::ContextOverflow(leaky.to_string()),
        ] {
            let withheld = acp_error_for(&error, false);
            let data = serde_json::to_string(&withheld.data).unwrap_or_default();
            assert!(!data.contains("acct-0f3c"), "{error}: {data}");
            assert!(
                data.contains("meka log") || data.contains("shorten it"),
                "and the client must be told where the detail went, or what to do: {data}"
            );

            let relayed = acp_error_for(&error, true);
            let data = serde_json::to_string(&relayed.data).unwrap_or_default();
            assert!(
                data.contains("acct-0f3c"),
                "{error}: the operator asked for the body and did not get it: {data}"
            );
        }

        // Bounded, because a misconfigured `base_url` can answer with megabytes and this text is
        // copied per failure.
        let huge = MekaError::Provider("x".repeat(64 * crate::text::KIB));
        let data = serde_json::to_string(&acp_error_for(&huge, true).data).unwrap_or_default();
        assert!(
            data.len() < 8 * crate::text::KIB,
            "an unbounded upstream body reached the client: {} bytes",
            data.len()
        );
    }

    /// meka's own request ceiling is the caller's to act on, so it keeps its sentence and its code.
    ///
    /// `InvalidParams` rather than `InternalError` for the reason every other refusal in that list
    /// gets it: nothing in meka failed, and the remedy is in the message.
    #[test]
    fn the_request_ceiling_reaches_the_client_as_a_refusal_it_can_act_on() {
        let error = MekaError::RequestTooLarge(
            "request body is 31.4 MiB after redacting old tool-result images; this profile's \
             ceiling is 30.0 MiB (`max_request_bytes`)."
                .to_string(),
        );
        let reported = acp_error_for(&error, false);
        assert_eq!(
            reported.code,
            agent_client_protocol::ErrorCode::InvalidParams
        );
        let data = serde_json::to_string(&reported.data).unwrap_or_default();
        assert!(data.contains("max_request_bytes"), "{data}");
    }

    /// A sticky permission option must name the tool it actually covers: the option is keyed on
    /// the tool name and applies for the rest of the session, while the prompt beside it names one
    /// specific command.
    #[test]
    fn a_sticky_permission_option_names_the_tool_it_covers() {
        assert_eq!(
            sticky_option_label("allow", "execute_command"),
            "Always allow any execute_command"
        );
        assert_eq!(
            sticky_option_label("deny", "write_file"),
            "Always deny any write_file"
        );
        for verb in ["allow", "deny"] {
            assert!(
                sticky_option_label(verb, "execute_command").contains("execute_command"),
                "dropping the tool name makes the option read as approving the one call on screen"
            );
        }
    }

    #[test]
    fn tool_kind_for_covers_builtins() {
        assert_eq!(tool_kind_for("read_file"), ToolKind::Read);
        assert_eq!(tool_kind_for("edit_file"), ToolKind::Edit);
        assert_eq!(tool_kind_for("write_file"), ToolKind::Edit);
        assert_eq!(tool_kind_for("find_files"), ToolKind::Search);
        assert_eq!(tool_kind_for("search_contents"), ToolKind::Search);
        assert_eq!(tool_kind_for("execute_command"), ToolKind::Execute);
        assert_eq!(tool_kind_for("fetch_url"), ToolKind::Fetch);
        assert_eq!(tool_kind_for("agent_spawn"), ToolKind::Think);
        // MCP-loaded tools and anything else fall through.
        assert_eq!(tool_kind_for("mcp__github__create_issue"), ToolKind::Other);
        assert_eq!(tool_kind_for("scratchpad_write"), ToolKind::Other);
        assert_eq!(tool_kind_for("totally_unknown"), ToolKind::Other);
    }

    #[test]
    fn todo_items_to_plan_maps_status_and_priority() {
        let items = vec![
            TodoItem {
                text: "first".to_string(),
                status: TodoStatus::Pending,
            },
            TodoItem {
                text: "second".to_string(),
                status: TodoStatus::InProgress,
            },
            TodoItem {
                text: "third".to_string(),
                status: TodoStatus::Completed,
            },
            TodoItem {
                text: "fourth".to_string(),
                status: TodoStatus::Canceled,
            },
        ];
        let entries = todo_items_to_plan(&items);
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0].content, "first");
        assert_eq!(entries[0].status, PlanEntryStatus::Pending);
        assert_eq!(entries[1].status, PlanEntryStatus::InProgress);
        assert_eq!(entries[2].status, PlanEntryStatus::Completed);
        // Canceled has no ACP analog; it collapses to Completed.
        assert_eq!(entries[3].status, PlanEntryStatus::Completed);
        // meka tracks no per-item priority, so every entry is Medium.
        assert!(
            entries
                .iter()
                .all(|entry| entry.priority == PlanEntryPriority::Medium)
        );
    }

    /// A real PNG, because the payload has to survive a decode: a client's attachment goes through
    /// the same [`crate::image::prepare_image_source`] door a tool result does.
    fn tiny_png() -> Vec<u8> {
        let mut out = Vec::new();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([1, 2, 3, 255]))
            .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .expect("encode");
        out
    }

    #[tokio::test]
    async fn decode_acp_image_passes_through_within_cap() {
        let data = base64::engine::general_purpose::STANDARD.encode(tiny_png());
        let image = ImageContent::new(data, "image/png".to_string());
        let source = decode_acp_image(&image).await.expect("decode");
        assert_eq!(source.media_type(), "image/png");
    }

    /// A client's attachment is refused for the same reason a `read_file` result is: forwarding
    /// bytes that only look like a PNG lands the provider's rejection inside a committed message.
    #[tokio::test]
    async fn decode_acp_image_rejects_a_payload_that_does_not_decode() {
        let mut truncated = tiny_png();
        truncated.truncate(16);
        let data = base64::engine::general_purpose::STANDARD.encode(&truncated);
        let image = ImageContent::new(data, "image/png".to_string());
        let error = decode_acp_image(&image)
            .await
            .expect_err("should reject undecodable bytes");
        assert!(error.contains("decode"), "got: {error}");
    }

    #[tokio::test]
    async fn decode_acp_image_rejects_oversized() {
        let raw = vec![0u8; crate::image::MAX_IMAGE_RAW_BYTES + 1];
        let data = base64::engine::general_purpose::STANDARD.encode(&raw);
        let image = ImageContent::new(data, "image/png".to_string());
        let error = decode_acp_image(&image)
            .await
            .expect_err("should reject oversized");
        assert!(error.contains("too large"), "got: {error}");
    }

    #[tokio::test]
    async fn decode_acp_image_rejects_bad_base64() {
        let image = ImageContent::new("not%%%valid".to_string(), "image/png".to_string());
        assert!(decode_acp_image(&image).await.is_err());
    }

    #[test]
    fn format_embedded_resource_text_inlines_contents() {
        let embedded = EmbeddedResource::new(EmbeddedResourceResource::TextResourceContents(
            agent_client_protocol::schema::v1::TextResourceContents::new(
                "fn main() {}",
                "file:///proj/src/main.rs",
            )
            .mime_type("text/x-rust".to_string()),
        ));
        let tag = format_embedded_resource(&embedded);
        assert_eq!(
            tag,
            "<resource uri=\"file:///proj/src/main.rs\" mime=\"text/x-rust\">fn main() {}</resource>"
        );
    }

    #[test]
    fn format_embedded_resource_blob_emits_marker_without_payload() {
        let embedded = EmbeddedResource::new(EmbeddedResourceResource::BlobResourceContents(
            agent_client_protocol::schema::v1::BlobResourceContents::new(
                "QUJD",
                "file:///proj/logo.png",
            )
            .mime_type("image/png".to_string()),
        ));
        let tag = format_embedded_resource(&embedded);
        // The base64 payload must NOT be inlined; only a self-closing marker.
        assert_eq!(
            tag,
            "<resource uri=\"file:///proj/logo.png\" mime=\"image/png\" encoding=\"base64\"/>"
        );
        assert!(!tag.contains("QUJD"));
    }

    #[test]
    fn embedded_resource_tag_survives_the_turn_message_shape() {
        // A `<resource>` tag is part of the user's prompt body. Stored beside the agent's own
        // context block, the words come back intact.
        let embedded = EmbeddedResource::new(EmbeddedResourceResource::TextResourceContents(
            agent_client_protocol::schema::v1::TextResourceContents::new(
                "hello",
                "file:///note.txt",
            ),
        ));
        let prompt_body = format!("see this\n{}", format_embedded_resource(&embedded));
        let message = crate::conversation::Message::user_turn(
            "<context>\n[Environment context]\n</context>",
            prompt_body.clone(),
            Vec::new(),
        );
        assert_eq!(message.text_content(), prompt_body);
    }

    #[test]
    fn tool_locations_resolves_relative_against_cwd() {
        let cwd = SharedCwd::new(PathBuf::from("/home/agent/proj"));
        let input = serde_json::json!({"path": "src/main.rs"});
        let locations = tool_locations("read_file", &input, &cwd);
        assert_eq!(locations.len(), 1);
        assert_eq!(
            locations[0].path,
            PathBuf::from("/home/agent/proj/src/main.rs")
        );
    }

    #[test]
    fn tool_locations_passes_absolute_paths_through() {
        let cwd = SharedCwd::new(PathBuf::from("/some/other/dir"));
        let input = serde_json::json!({"path": "/etc/hosts"});
        let locations = tool_locations("edit_file", &input, &cwd);
        assert_eq!(locations[0].path, PathBuf::from("/etc/hosts"));
    }

    #[test]
    fn tool_locations_empty_for_non_path_tools() {
        let cwd = SharedCwd::new(PathBuf::from("/"));
        let input = serde_json::json!({"command": "ls"});
        assert!(tool_locations("execute_command", &input, &cwd).is_empty());
        assert!(tool_locations("search_web", &input, &cwd).is_empty());
    }

    #[test]
    fn tool_locations_read_file_line_from_offset() {
        let cwd = SharedCwd::new(PathBuf::from("/home/agent/proj"));
        // `read_file` offset is 0-based; ACP `line` is 1-based.
        let input = serde_json::json!({"path": "src/main.rs", "offset": 41});
        let locations = tool_locations("read_file", &input, &cwd);
        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0].line, Some(42));
        // No offset -> no line.
        let no_offset = serde_json::json!({"path": "src/main.rs"});
        assert_eq!(tool_locations("read_file", &no_offset, &cwd)[0].line, None);
        // Other path tools never set a line, even with an offset present.
        let edit = serde_json::json!({"path": "src/main.rs", "offset": 41});
        assert_eq!(tool_locations("edit_file", &edit, &cwd)[0].line, None);
    }

    #[test]
    fn build_completion_content_prefers_diff_metadata() {
        let metadata = Some(ToolOutputMetadata::Diff {
            path: PathBuf::from("/tmp/foo.txt"),
            old_text: Some("old".to_string()),
            new_text: "new".to_string(),
        });
        let content = vec![ToolResultContent::Text {
            text: "ignored".to_string(),
        }];
        let blocks = build_completion_content("edit_file", &content, metadata);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(blocks[0], ToolCallContent::Diff(_)));
    }

    /// The title opens with the tool's name, the one the REPL's indicator and approval prompt
    /// show, so a user reading both surfaces meets one vocabulary; the argument follows it.
    #[test]
    fn tool_call_title_per_tool() {
        assert_eq!(
            tool_call_title("execute_command", Some("git status && git diff")),
            "execute_command git status && git diff"
        );
        assert_eq!(
            tool_call_title("read_file", Some("src/main.rs")),
            "read_file src/main.rs"
        );
        assert_eq!(
            tool_call_title("edit_file", Some("src/lib.rs")),
            "edit_file src/lib.rs"
        );
        assert_eq!(
            tool_call_title("write_file", Some("out.txt")),
            "write_file out.txt"
        );
        assert_eq!(
            tool_call_title("find_files", Some("**/*.rs")),
            "find_files **/*.rs"
        );
        assert_eq!(
            tool_call_title("search_contents", Some("TODO")),
            "search_contents TODO"
        );
        assert_eq!(
            tool_call_title("fetch_url", Some("https://example.com")),
            "fetch_url https://example.com"
        );
        assert_eq!(
            tool_call_title("search_web", Some("rust acp")),
            "search_web rust acp"
        );
        // An MCP tool's name is the server's, shown as the server spells it.
        assert_eq!(
            tool_call_title("mcp__exa__web_search_exa", Some("query")),
            "mcp__exa__web_search_exa query"
        );
        // No primary argument resolved -> the name alone.
        assert_eq!(tool_call_title("read_file", None), "read_file");
    }

    #[test]
    fn tool_call_title_sanitizes_whitespace_and_length() {
        // A multi-line command collapses to a single line.
        assert_eq!(
            tool_call_title("execute_command", Some("git status\n  && git diff")),
            "execute_command git status && git diff"
        );
        // Over-long titles are truncated with an ellipsis.
        let long = "x".repeat(400);
        let title = tool_call_title("execute_command", Some(&long));
        assert!(title.chars().count() <= 256);
        assert!(title.ends_with('…'));
    }

    #[test]
    fn build_completion_content_execute_command_wraps_console() {
        let content = vec![ToolResultContent::Text {
            text: "hello\nworld\n".to_string(),
        }];
        let blocks = build_completion_content("execute_command", &content, None);
        assert_eq!(blocks.len(), 1);
        let ToolCallContent::Content(chunk) = &blocks[0] else {
            panic!("expected ToolCallContent::Content; got {:?}", blocks[0]);
        };
        let ContentBlock::Text(text) = &chunk.content else {
            panic!("expected ContentBlock::Text; got {:?}", chunk.content);
        };
        assert_eq!(text.text, "```console\nhello\nworld\n```");
    }

    #[test]
    fn build_completion_content_execute_command_empty_output_no_block() {
        let content = vec![ToolResultContent::Text {
            text: "   \n".to_string(),
        }];
        assert!(build_completion_content("execute_command", &content, None).is_empty());
    }

    fn empty_live_output() -> LiveOutput {
        LiveOutput::new(LiveOutputMode::Text)
    }

    /// The first chunk goes out immediately (a command that prints one line and exits must still
    /// show it), and chunks inside the interval accumulate silently rather than being dropped.
    #[test]
    fn live_output_throttles_but_keeps_every_byte() {
        let start = std::time::Instant::now();
        let mut live = empty_live_output();

        assert_eq!(live.push("first\n", start).as_deref(), Some("first\n"));
        assert_eq!(
            live.push("swallowed\n", start + LIVE_OUTPUT_INTERVAL / 2),
            None,
            "a second chunk inside the interval must not produce an update",
        );
        assert_eq!(
            live.push("later\n", start + LIVE_OUTPUT_INTERVAL * 2)
                .as_deref(),
            Some("first\nswallowed\nlater\n"),
            "the throttled chunk must reappear in the next update, not be lost",
        );
    }

    /// The live view is capped, so a command that dumps far more than the cap doesn't make every
    /// subsequent update carry the whole history. Cutting must land on a line boundary and must
    /// never split a multi-byte character (slicing off one panics).
    #[test]
    fn live_output_trims_to_a_tail_on_a_line_boundary() {
        let start = std::time::Instant::now();
        let mut live = empty_live_output();
        // Multi-byte content so a naive byte-offset cut would panic rather than merely look wrong.
        let line = "ünïcödé filler line to push past the cap\n";
        let mut now = start;
        for _ in 0..(LIVE_OUTPUT_TAIL_BYTES / line.len() + 10) {
            now += LIVE_OUTPUT_INTERVAL * 2;
            live.push(line, now);
        }
        let tail = live
            .push("final\n", now + LIVE_OUTPUT_INTERVAL * 2)
            .expect("update");
        assert!(
            tail.len() <= LIVE_OUTPUT_TAIL_BYTES + line.len(),
            "tail should stay near the cap; got {} bytes",
            tail.len(),
        );
        assert!(tail.ends_with("final\n"), "the newest output must survive");
        assert!(
            tail.starts_with(line),
            "the cut must land at a line start; got {:?}",
            &tail[..line.len().min(tail.len())],
        );
    }

    /// Terminal mode appends into a buffer the client owns, so each send must carry only what has
    /// arrived since the last one. Re-sending the running total (which is what text mode does)
    /// would make the terminal show every line duplicated more times the longer the command ran.
    #[test]
    fn live_output_terminal_mode_sends_each_byte_once() {
        let start = std::time::Instant::now();
        let mut live = LiveOutput::new(LiveOutputMode::Terminal);

        assert_eq!(live.push("alpha\n", start).as_deref(), Some("alpha\n"));
        assert_eq!(
            live.push("beta\n", start + LIVE_OUTPUT_INTERVAL * 2)
                .as_deref(),
            Some("beta\n"),
            "the second send must not repeat the first chunk",
        );
        // Throttled chunks coalesce into the next send rather than being dropped or re-sent.
        assert_eq!(live.push("gamma\n", start + LIVE_OUTPUT_INTERVAL * 2), None);
        assert_eq!(
            live.push("delta\n", start + LIVE_OUTPUT_INTERVAL * 4)
                .as_deref(),
            Some("gamma\ndelta\n"),
        );
    }

    /// The throttle can swallow the final chunk of a command that exits right after printing. In
    /// terminal mode the client's scrollback is the only copy of those bytes, so completion has to
    /// flush them; in text mode the completion update re-sends everything anyway.
    #[test]
    fn live_output_terminal_mode_flushes_what_the_throttle_held_back() {
        let start = std::time::Instant::now();
        let mut live = LiveOutput::new(LiveOutputMode::Terminal);
        live.push("first\n", start).expect("first send");
        assert_eq!(live.push("last gasp\n", start), None, "inside the interval");
        assert_eq!(live.take_pending().as_deref(), Some("last gasp\n"));
        assert_eq!(live.take_pending(), None, "flushing twice would duplicate");

        let mut text_mode = LiveOutput::new(LiveOutputMode::Text);
        text_mode.push("buffered\n", start);
        assert_eq!(
            text_mode.take_pending(),
            None,
            "text mode's completion update carries the whole output already",
        );
    }

    /// The tail cap exists only because text mode re-sends its buffer every tick. Applying it to
    /// terminal mode would silently drop output the client can no longer recover.
    #[test]
    fn live_output_terminal_mode_never_drops_output_to_the_tail_cap() {
        let start = std::time::Instant::now();
        let mut live = LiveOutput::new(LiveOutputMode::Terminal);
        let line = "x".repeat(1024) + "\n";
        let mut now = start;
        let mut delivered = String::new();
        let rounds = (LIVE_OUTPUT_TAIL_BYTES / line.len()) + 8;
        for _ in 0..rounds {
            now += LIVE_OUTPUT_INTERVAL * 2;
            if let Some(chunk) = live.push(&line, now) {
                delivered.push_str(&chunk);
            }
        }
        assert_eq!(
            delivered.len(),
            line.len() * rounds,
            "every byte must reach the client exactly once, past the text-mode cap",
        );
    }

    /// A live update reports content only. Setting a status would tell the client the call had
    /// finished while the command is still running.
    #[test]
    fn live_output_update_carries_no_status() {
        let mut live = empty_live_output();
        let text = live
            .push("building...\n", std::time::Instant::now())
            .expect("first push always emits");
        let fields = ToolCallUpdateFields::new().content(vec![console_content_block(&text)]);
        let update = ToolCallUpdate::new("call_1", fields);
        let wire = serde_json::to_value(&update).expect("serialize");
        assert!(
            wire["status"].is_null(),
            "live updates must not carry a status; got {wire}",
        );
        assert_eq!(
            wire["content"][0]["content"]["text"], "```console\nbuilding...\n```",
            "the live view must render the same way the completed one does",
        );
    }

    #[test]
    fn translate_permission_outcome_maps_each_option() {
        use agent_client_protocol::schema::v1::SelectedPermissionOutcome;

        // Capture sticky pushes via a `Cell` so each call site borrows it fresh; this sidesteps the
        // closure-vs-direct-read borrow conflict that comes from sharing one `&mut Vec`.
        let sticky: std::cell::RefCell<Vec<&'static str>> = std::cell::RefCell::new(Vec::new());
        let record = |s: StickyDecision| {
            sticky.borrow_mut().push(match s {
                StickyDecision::AllowAlways => "allow",
                StickyDecision::RejectAlways => "deny",
            });
        };

        assert_eq!(
            translate_permission_outcome(
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                    OPTION_ALLOW_ONCE,
                )),
                "read_file",
                record,
            ),
            PermissionOutcome::Allow,
        );
        assert!(
            sticky.borrow().is_empty(),
            "allow_once must not record a sticky"
        );

        assert_eq!(
            translate_permission_outcome(
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                    OPTION_ALLOW_ALWAYS,
                )),
                "read_file",
                record,
            ),
            PermissionOutcome::Allow,
        );
        assert_eq!(sticky.borrow().last().copied(), Some("allow"));

        assert_eq!(
            translate_permission_outcome(
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                    OPTION_REJECT_ONCE,
                )),
                "write_file",
                record,
            ),
            PermissionOutcome::Deny,
        );
        assert_eq!(
            sticky.borrow().last().copied(),
            Some("allow"),
            "reject_once must not push"
        );

        assert_eq!(
            translate_permission_outcome(
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                    OPTION_REJECT_ALWAYS,
                )),
                "write_file",
                record,
            ),
            PermissionOutcome::Deny,
        );
        assert_eq!(sticky.borrow().last().copied(), Some("deny"));

        assert_eq!(
            translate_permission_outcome(RequestPermissionOutcome::Cancelled, "read_file", record,),
            PermissionOutcome::Canceled,
        );
    }

    #[test]
    fn translate_permission_outcome_unknown_option_denies() {
        use agent_client_protocol::schema::v1::SelectedPermissionOutcome;
        let result = translate_permission_outcome(
            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new("future_option")),
            "read_file",
            &mut |_| {},
        );
        assert_eq!(result, PermissionOutcome::Deny);
    }

    #[test]
    fn build_completion_content_falls_back_to_text() {
        let content = vec![ToolResultContent::Text {
            text: "hello".to_string(),
        }];
        let blocks = build_completion_content("read_file", &content, None);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(blocks[0], ToolCallContent::Content(_)));
    }

    /// Image tool results go out as ACP `image` content blocks rather than a text marker, so the
    /// client can render the picture the model was shown. Walks into the block to confirm the
    /// payload and MIME type survive the conversion intact.
    #[test]
    fn build_completion_content_forwards_image_content() {
        use crate::image::ImageSource;
        let content = vec![ToolResultContent::Image {
            source: ImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: "aGVsbG8=".to_string(),
            },
        }];
        let blocks = build_completion_content("read_file", &content, None);
        assert_eq!(blocks.len(), 1);
        let ToolCallContent::Content(chunk) = &blocks[0] else {
            panic!("expected ToolCallContent::Content; got {:?}", blocks[0]);
        };
        let ContentBlock::Image(image) = &chunk.content else {
            panic!("expected ContentBlock::Image; got {:?}", chunk.content);
        };
        assert_eq!(image.data, "aGVsbG8=");
        assert_eq!(image.mime_type, "image/png");
    }

    /// A tool whose output interleaves text and an image keeps both blocks, in order: the text
    /// marker `read_file` emits alongside an image (`[Image: path]`) is what names the file in the
    /// transcript, so dropping either half loses information.
    #[test]
    fn build_completion_content_preserves_mixed_text_and_image() {
        use crate::image::ImageSource;
        let content = vec![
            ToolResultContent::Text {
                text: "[Image: logo.png]".to_string(),
            },
            ToolResultContent::Image {
                source: ImageSource::Base64 {
                    media_type: "image/webp".to_string(),
                    data: "d2VicA==".to_string(),
                },
            },
        ];
        let blocks = build_completion_content("read_file", &content, None);
        assert_eq!(blocks.len(), 2);
        let ToolCallContent::Content(first) = &blocks[0] else {
            panic!("expected ToolCallContent::Content; got {:?}", blocks[0]);
        };
        assert!(
            matches!(&first.content, ContentBlock::Text(text) if text.text == "[Image: logo.png]")
        );
        let ToolCallContent::Content(second) = &blocks[1] else {
            panic!("expected ToolCallContent::Content; got {:?}", blocks[1]);
        };
        assert!(
            matches!(&second.content, ContentBlock::Image(image) if image.mime_type == "image/webp")
        );
    }

    #[test]
    fn parse_mode_id_covers_all_levels() {
        assert_eq!(parse_mode_id("none"), Some(Permission::None));
        assert_eq!(parse_mode_id("read"), Some(Permission::Read));
        assert_eq!(parse_mode_id("workspace"), Some(Permission::Workspace));
        assert_eq!(
            parse_mode_id("unrestricted"),
            Some(Permission::Unrestricted)
        );
    }

    /// An id naming no mode is refused, rather than resolving to some rung the client did not ask
    /// for.
    #[test]
    fn parse_mode_id_rejects_garbage() {
        // One spelling per level, shared with `--permission`: a case variant names no mode either.
        for unknown in ["admin", "write", "elevated", "", "READ"] {
            assert!(
                parse_mode_id(unknown).is_none(),
                "'{unknown}' names no mode and must not resolve"
            );
        }
    }

    #[test]
    fn build_mode_state_lists_only_enabled_modes() {
        use crate::permission::{EnabledPermissions, SharedPermission};
        let enabled = EnabledPermissions::from_levels([Permission::Read, Permission::Unrestricted])
            .expect("non-empty");
        let permission = SharedPermission::new(Permission::Read, enabled);

        let state = build_mode_state(&permission);
        let ids: Vec<&str> = state
            .available_modes
            .iter()
            .map(|m| m.id.0.as_ref())
            .collect();
        assert_eq!(ids, vec!["read", "unrestricted"]);
        assert_eq!(state.current_mode_id.0.as_ref(), "read");
        // Descriptions populated.
        assert!(
            state
                .available_modes
                .iter()
                .all(|m| m.description.is_some()),
            "every mode advertised must carry a description"
        );
    }

    #[test]
    fn build_mode_state_reflects_current_after_set() {
        use crate::permission::{EnabledPermissions, SharedPermission};
        let permission = SharedPermission::new(Permission::Read, EnabledPermissions::ALL);
        permission
            .try_set(Permission::Unrestricted)
            .expect("unrestricted enabled");
        assert_eq!(
            build_mode_state(&permission).current_mode_id.0.as_ref(),
            "unrestricted"
        );
    }

    fn select_options(option: &SessionConfigOption) -> Vec<(String, Option<String>)> {
        use agent_client_protocol::schema::v1::{SessionConfigKind, SessionConfigSelectOptions};
        match &option.kind {
            SessionConfigKind::Select(select) => match &select.options {
                SessionConfigSelectOptions::Ungrouped(options) => options
                    .iter()
                    .map(|option| {
                        (
                            option.value.0.as_ref().to_string(),
                            option.description.clone(),
                        )
                    })
                    .collect(),
                other => panic!("expected an ungrouped select, got {other:?}"),
            },
            other => panic!("expected a select option, got {other:?}"),
        }
    }

    fn current_value(option: &SessionConfigOption) -> String {
        use agent_client_protocol::schema::v1::SessionConfigKind;
        match &option.kind {
            SessionConfigKind::Select(select) => select.current_value.0.as_ref().to_string(),
            other => panic!("expected a select option, got {other:?}"),
        }
    }

    fn profiles_for_test(
        entries: &[(&str, Option<&str>)],
    ) -> std::collections::BTreeMap<String, crate::config::ProfileConfig> {
        entries
            .iter()
            .map(|(name, model)| {
                (name.to_string(), crate::config::ProfileConfig {
                    account: name.to_string(),
                    model: model.map(str::to_string),
                    ..Default::default()
                })
            })
            .collect()
    }

    /// The `configOptions` permission entry must offer exactly what the `modes` picker offers, or a
    /// client driving one of the two pickers is looking at a different set of levels from a client
    /// driving the other.
    #[test]
    fn the_permission_config_option_matches_the_mode_picker() {
        use crate::permission::{EnabledPermissions, SharedPermission};
        let enabled = EnabledPermissions::from_levels([Permission::Read, Permission::Unrestricted])
            .expect("non-empty");
        let permission = SharedPermission::new(Permission::Read, enabled);

        let option = permission_config_option(&permission);
        let offered: Vec<String> = select_options(&option)
            .into_iter()
            .map(|(value, _description)| value)
            .collect();
        let modes: Vec<String> = build_mode_state(&permission)
            .available_modes
            .iter()
            .map(|mode| mode.id.0.as_ref().to_string())
            .collect();

        assert_eq!(offered, modes);
        assert_eq!(current_value(&option), "read");
        assert_eq!(option.id.0.as_ref(), PERMISSION_CONFIG_ID);
    }

    /// The switch is advertised as a boolean option reading the same cell the dispatch door reads.
    #[test]
    fn the_approvals_config_option_reports_the_switch() {
        use agent_client_protocol::schema::v1::SessionConfigKind;

        use crate::permission::{EnabledPermissions, SharedPermission};
        fn current(option: &SessionConfigOption) -> bool {
            match &option.kind {
                SessionConfigKind::Boolean(boolean) => boolean.current_value,
                other => panic!("expected a boolean option, got {other:?}"),
            }
        }
        let permission = SharedPermission::new(Permission::Read, EnabledPermissions::ALL);
        assert!(!current(&approvals_config_option(&permission)));
        permission.set_approvals(true);
        let option = approvals_config_option(&permission);
        assert_eq!(option.id.0.as_ref(), APPROVALS_CONFIG_ID);
        assert!(current(&option));
    }

    /// A profile that states no model must not acquire one here: the description is shown to the
    /// user as a fact about the profile, and meka does not know what an unstated model resolves to.
    #[test]
    fn a_provider_option_describes_only_a_stated_model() {
        let profiles = profiles_for_test(&[("work", Some("claude-opus-5")), ("personal", None)]);

        let option = profile_config_option(&profiles, "personal");

        assert_eq!(option.id.0.as_ref(), PROFILE_CONFIG_ID);
        assert_eq!(current_value(&option), "personal");
        assert_eq!(select_options(&option), vec![
            ("personal".to_string(), None),
            ("work".to_string(), Some("claude-opus-5".to_string())),
        ],);
    }

    /// A session pinned to a profile that has since left `config.toml` selects nothing rather than
    /// silently presenting some other profile as the one it runs on.
    #[test]
    fn a_provider_no_longer_configured_selects_nothing() {
        let profiles = profiles_for_test(&[("work", None)]);

        let option = profile_config_option(&profiles, "retired");

        assert_eq!(current_value(&option), "retired");
        assert!(
            !select_options(&option)
                .iter()
                .any(|(value, _description)| value == "retired"),
            "a profile that is gone must not appear among the choices"
        );
    }

    #[tokio::test]
    async fn slash_to_prompt_text_passes_through_non_slash() {
        let cache = SkillCache::for_root(None);
        let out = slash_to_prompt_text("just a normal prompt".to_string(), &cache)
            .await
            .expect("ok");
        assert_eq!(out, "just a normal prompt");
    }

    #[tokio::test]
    async fn slash_to_prompt_text_passes_through_paste_shaped_input() {
        // A pasted path like `/etc/hosts is a config file` has an invalid skill-name first token
        // (slash inside the name), so the helper must NOT touch it.
        let cache = SkillCache::for_root(None);
        let out = slash_to_prompt_text("/etc/hosts is the config file".to_string(), &cache)
            .await
            .expect("pass-through");
        assert_eq!(out, "/etc/hosts is the config file");
    }

    #[tokio::test]
    async fn slash_to_prompt_text_passes_through_double_slash_comment() {
        // `//foo` parses as name="/foo", which is invalid; pass through.
        let cache = SkillCache::for_root(None);
        let out = slash_to_prompt_text("//comment line".to_string(), &cache)
            .await
            .expect("pass-through");
        assert_eq!(out, "//comment line");
    }

    #[tokio::test]
    async fn slash_to_prompt_text_unknown_but_valid_name_errors() {
        // A clean `/<name>` shape with a syntactically valid skill name that isn't installed:
        // error, since the only realistic source of this shape is a typo'd palette pick.
        let cache = SkillCache::for_root(None);
        let err = slash_to_prompt_text("/nonexistent".to_string(), &cache)
            .await
            .expect_err("should error");
        assert!(
            matches!(err, SlashInvocationError::SkillNotFound(ref reason)
                if reason == "no skill named 'nonexistent'"),
            "an absent name reads as absent, not as an unreadable file: {err}"
        );
    }

    /// `/name` for a skill whose `SKILL.md` will not parse reports the file, not "unknown skill".
    ///
    /// This is a person typing a name they know exists, so answering that it does not sends them
    /// looking for something they already have.
    #[tokio::test]
    async fn slash_to_prompt_text_reports_a_broken_skill_as_broken() {
        let temp = tempfile::tempdir().expect("tempdir");
        let skill_dir = temp.path().join("wrecked");
        std::fs::create_dir_all(&skill_dir).expect("mkdir skill");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: wrecked\ndescription: [unclosed\n---\nbody\n",
        )
        .expect("write skill");
        let cache = SkillCache::for_root(Some(temp.path().to_path_buf()));

        let err = slash_to_prompt_text("/wrecked".to_string(), &cache)
            .await
            .expect_err("should error");
        let message = err.to_string();
        assert!(
            message.contains("failed to load"),
            "a file that is right there is not an unknown skill: {message}"
        );
        assert!(message.contains("frontmatter"), "{message}");
    }

    #[tokio::test]
    async fn slash_to_prompt_text_known_skill_composes_body() {
        // Drop a SKILL.md under a tempdir, point a fresh cache at it.
        let temp = tempfile::tempdir().expect("tempdir");
        let skill_dir = temp.path().join("demo");
        std::fs::create_dir_all(&skill_dir).expect("mkdir skill");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\ndescription: demo skill\n---\nrun ls in scripts/\n",
        )
        .expect("write SKILL.md");

        let cache = SkillCache::for_root(Some(temp.path().to_path_buf()));
        let out = slash_to_prompt_text("/demo only fetch UK news".to_string(), &cache)
            .await
            .expect("ok");
        assert!(
            out.starts_with("only fetch UK news\n\n"),
            "extra context must lead: {out}"
        );
        assert!(
            out.contains("run ls in scripts/"),
            "body must be passed through verbatim: {out}"
        );
        assert!(
            out.contains("Base directory for this skill"),
            "skill_context_header must be present: {out}"
        );
    }

    #[tokio::test]
    async fn slash_to_prompt_text_known_skill_no_extra() {
        let temp = tempfile::tempdir().expect("tempdir");
        let skill_dir = temp.path().join("ping");
        std::fs::create_dir_all(&skill_dir).expect("mkdir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\ndescription: ping\n---\npong\n",
        )
        .expect("write");

        let cache = SkillCache::for_root(Some(temp.path().to_path_buf()));
        let out = slash_to_prompt_text("/ping".to_string(), &cache)
            .await
            .expect("ok");
        // No `extra\n\n` prefix when the user passed only the skill name; the body stands alone.
        assert!(
            !out.starts_with("\n\n"),
            "bare /skill must not have a leading newline: {out:?}"
        );
        assert!(out.contains("pong"));
    }

    #[test]
    fn shared_client_state_round_trip() {
        let shared = SharedClientState::default();
        // Default snapshot has every capability false and no client identity recorded.
        let initial = shared.capabilities();
        assert!(!initial.fs.read_text_file);
        assert!(!initial.fs.write_text_file);
        assert!(!initial.terminal);
        assert!(shared.client_info().is_none());

        let updated_caps = ClientCapabilities::new()
            .fs(
                agent_client_protocol::schema::v1::FileSystemCapabilities::new()
                    .read_text_file(true)
                    .write_text_file(true),
            )
            .terminal(true);
        let updated_info = Implementation::new("test-editor", "9.9.9");
        shared.record_initialize(updated_caps, Some(updated_info));

        let after_caps = shared.capabilities();
        assert!(after_caps.fs.read_text_file);
        assert!(after_caps.fs.write_text_file);
        assert!(after_caps.terminal);
        let after_info = shared.client_info().expect("info present");
        assert_eq!(after_info.name, "test-editor");
        assert_eq!(after_info.version, "9.9.9");
    }

    #[test]
    fn describe_client_formats_known_and_unknown() {
        assert_eq!(describe_client(None), "<unknown> <unknown>");
        let info = Implementation::new("zed", "0.999.0");
        assert_eq!(describe_client(Some(&info)), "zed 0.999.0");
    }
}
