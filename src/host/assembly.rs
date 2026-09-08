//! Assembling a session: the dependencies every session on this process shares, building an agent
//! for a fresh or resumed session, opening and releasing it, switching its provider, and the
//! resume-time reconciliation of what its row recorded.

use super::*;

/// Process-wide dependencies that every ACP session shares. Built once at `meka acp` startup by
/// [`build_shared_deps`]; sessions hold an [`Arc<SharedDeps>`] and read fields by reference.
/// Cheap to clone (every field is either an `Arc`, an owned-but-small value, or a clonable handle).
/// Every host builds one, the REPL and `--oneshot` included, so all four assemble a session through
/// the same [`build_session_agent`].
#[derive(Clone)]
pub(crate) struct SharedDeps {
    pub(crate) config: Arc<ResolvedConfig>,
    /// The profile a session created here takes when it names none. `None` for a run that resumes
    /// a session and has no default of its own, which only the REPL and `--oneshot` allow; the
    /// long-lived hosts refuse to start without one, through [`Self::default_profile`].
    pub(crate) default_profile: Option<String>,
    pub(crate) store: Store,
    /// Providers by profile, built on demand.
    ///
    /// A registry rather than one `Arc<dyn Provider>` because a session records the profile it
    /// runs with, and one `meka serve` may host sessions naming different ones.
    pub(crate) providers: Arc<crate::provider::ProviderRegistry>,
    pub(crate) mcp_manager: Option<Arc<crate::mcp::McpClientManager>>,
    pub(crate) skills: Arc<crate::skills::SkillCache>,
    pub(crate) memories: Arc<crate::store::memory::MemoryStore>,
    pub(crate) builtin_filter: crate::config::BuiltinToolFilter,
    pub(crate) agent_options: AgentOptions,
    /// What the sandbox probe found at startup; every session's materials are built from it.
    pub(crate) sandbox: crate::sandbox::SandboxResolution,
    /// The dispatcher a scheduled gate's tool probe resolves against.
    pub(crate) gate_tools: Option<Arc<dyn crate::schedule::GateTools>>,
    /// The user's instructions, resolved once at startup, and where they came from.
    pub(crate) user_instructions: Option<String>,
    pub(crate) user_instructions_source: Option<String>,
    /// Shared by every session this host builds, so two of them writing one file serialize.
    pub(crate) write_locks: crate::workspace::WriteLocks,
}
/// Whether a host may repeat an upstream's own response text to its caller: `[serve]
/// relay_provider_errors`, on unless the operator turned it off.
///
/// One reading of the key for both hosts that answer a caller over a wire. `meka serve` resolves it
/// once into [`crate::host::http::config::ResolvedServeConfig`] and `meka acp` asks per failure, so
/// stating the default in each of them is how the two came to be able to disagree about what a
/// deployment configured to withhold actually withholds. The REPL and `--oneshot` never consult it:
/// their reader is the operator.
pub(crate) fn relay_provider_errors(serve: Option<&crate::config::ServeConfig>) -> bool {
    serve
        .and_then(|serve| serve.relay_provider_errors)
        .unwrap_or(true)
}
/// Warn once about `[tools]` and `[subagents]` entries that match nothing.
///
/// Called from both agent-assembly entry points. `meka acp` and `meka serve` build their agents
/// through `build_shared_deps`, so without a call here a typo in either block denies nothing,
/// silently, at every verbosity.
pub(crate) fn warn_on_stale_tool_config(
    config: &ResolvedConfig,
    builtin_filter: &crate::config::BuiltinToolFilter,
) {
    crate::tools::warn_on_stale_builtin_tool_config(builtin_filter);
    crate::tools::warn_on_stale_subagent_config(
        &crate::tools::ToolDenials::new(
            config.subagents.disabled_servers.clone(),
            config.subagents.disabled_tools.clone(),
        ),
        &config
            .mcp_servers
            .iter()
            .map(|server| server.name.clone())
            .collect::<Vec<_>>(),
    );
}
/// Refuse to build a plain agent for a session another one spawned.
///
/// A sub-agent's authority is not its host's. The `[subagents]` denials it was spawned under, its
/// memory and instruction grants, and the permission ceiling its spawn call set all live in
/// `sessions.subagent_spec_json`, and both builders below assemble from `config` alone. A sub-agent
/// built by either would run the conversation it was *given narrow tools for* with the full
/// built-in set at the host's level, which is the escalation `[subagents]` exists to prevent.
///
/// Called at the doors *and* kept in both builders. The doors are where a refusal has to land to
/// come before the side effects each one performs -- a lock taken, a `cwd` rewritten, background
/// work retired, a row repinned -- and the builders are the backstop that catches a door nobody
/// thought of. A rule placed only in the builders arrives too late to prevent a write; a rule
/// placed only at the doors is one the next door forgets, which is how this came to be enforced
/// for scheduled jobs alone.
///
/// Nothing about a previous release is encoded here: a sub-agent session must not be driven this
/// way on a store meka created a minute ago, and the check reads the same column either way.
///
/// [`crate::tools::subagent`]'s `agent_followup` is unaffected. It goes through `build_subagent`,
/// which reads the spec and clamps the level against the parent's live one, and is the only thing
/// that can reconstruct what the sub-agent was spawned with.
///
/// Reading the session is the whole check, so an id with no row is not this function's business:
/// `None` means there is nothing to refuse, and whatever follows answers for a session that is not
/// there.
pub(crate) async fn refuse_a_spawned_session(
    store: &Store,
    session_id: Option<uuid::Uuid>,
) -> anyhow::Result<()> {
    let Some(session_id) = session_id else {
        return Ok(());
    };
    // Refused here rather than at the import, because import restoring a whole tree is a case it
    // has to keep: a child whose parent *is* in the archive keeps its link and is caught by the
    // parent check. What no legitimate door produces is spawn terms with no parent.
    let Some(crate::store::SpawnTerms { parent }) = store.spawn_terms(session_id).await? else {
        return Ok(());
    };
    let door = match parent {
        Some(parent) => {
            format!("is a sub-agent of session {parent}; continue it with `agent_followup` there")
        }
        // An imported sub-agent whose parent did not come with it. There is no session to point at,
        // so say what it is rather than naming a door that is not there.
        None => {
            "is a sub-agent whose parent is not in this store, so it cannot be driven".to_string()
        }
    };
    Err(crate::error::MekaError::SessionNotDrivable(format!("session {session_id} {door}")).into())
}
/// Build the process-wide [`SharedDeps`] for `meka acp`. Sets up the provider, MCP wiring, skill
/// cache, sandbox capability probe, and the shared `agent_options` template. Each ACP session later
/// calls [`build_session_agent`] against the resulting struct to spin up its own per-session
/// `Agent` + `ToolRegistry`.
pub(crate) async fn build_shared_deps(
    config: Arc<ResolvedConfig>,
    store: Store,
    mcp_manager: Option<Arc<crate::mcp::McpClientManager>>,
) -> anyhow::Result<SharedDeps> {
    config.validate()?;
    let default_profile = config.default_profile.clone();

    // Nothing is built here. The registry resolves a profile and loads its credential when a
    // session first asks, because which profiles this process will need is a property of the
    // sessions it ends up serving rather than of its configuration.
    let providers = Arc::new(crate::provider::ProviderRegistry::new(
        &config,
        store.token_store(),
    ));
    // Once, off the worker threads: a first ask would otherwise resolve a subscription profile's
    // device id inline, writing `config.toml` under the config lock on whatever thread asked.
    providers.preload_device_ids().await;
    // Test-only: hand the registry a scripted provider to return for every profile, whichever host
    // this process is. Installed rather than swapped into a rebuilt registry, so a harness driving
    // sessions on different profiles gets the script for all of them.
    #[cfg(any(debug_assertions, feature = "mock-provider"))]
    if std::env::var("MEKA_MOCK_PROVIDER").as_deref() == Ok("1") {
        let rounds = crate::provider::mock::load_script_from_env()
            .map_err(|error| anyhow::anyhow!("failed to load the mock provider script: {error}"))?
            .unwrap_or_default();
        tracing::info!("MEKA_MOCK_PROVIDER=1: using scripted mock provider");
        providers.install_scripted(Arc::new(crate::provider::mock::MockProvider::from_rounds(
            rounds,
        )));
    }

    // Built once here and dropped, because what it produces is a refusal rather than a client:
    // every registry builds its own from the same `[web]` settings, and the one thing no session
    // can do is fail the *process* before it exists. Otherwise a `ca_cert_file` that is not there
    // or a proxy URL that is not one surfaces on the first turn, per session, as whatever that host
    // reports a registry failure as, which on `serve` hands the operator's path to a remote caller.
    crate::tools::build_web_client(&config.web_client)?;

    // Detection, not configuration: which backend this machine can run is probed once here, and
    // the startup warning about an unusable or improvable sandbox is given from the same answer.
    let sandbox = crate::sandbox::resolve_backend(config.sandbox_backend, config.sandbox);
    crate::sandbox::warn_if_sandbox_issues(
        &crate::sandbox::SandboxState::new(config.sandbox, &sandbox),
        crate::sandbox::WarnContext::Startup,
    );
    let sandbox_capability = crate::sandbox::capability_from_probe(&sandbox.probe);
    let sandboxed_shell = config.sandbox
        && !matches!(
            sandbox_capability,
            crate::sandbox::SandboxCapability::Unavailable
        );

    // Both stores are instance-scoped, so one cache each serves every session this process runs.
    // `disabled()` is distinct from an empty store: it keeps the subsystem's tools out of the
    // registry entirely, which is the point of the config switch.
    let skills = if config.skills_enabled {
        crate::skills::SkillCache::discover(config.skills_extra_paths.clone())
    } else {
        crate::skills::SkillCache::disabled()
    };
    // Always connected, `enabled` carrying the config switch: it gates the agent's tools, not the
    // operator's access to a store that already exists.
    let memories = store.memory_store(config.memory_enabled);
    let builtin_filter = crate::config::BuiltinToolFilter::from_config(
        config.builtin_allowed_tools.clone(),
        config.builtin_disabled_tools.clone(),
        config.builtin_tool_permissions.clone(),
    );
    warn_on_stale_tool_config(&config, &builtin_filter);

    // One dispatcher for the whole process, for the same reason the MCP manager is process-wide: a
    // gate's tool probe is answered by the process that picks the job up, not by the session that
    // wrote it.
    let gate_tools: Option<Arc<dyn crate::schedule::GateTools>> =
        Some(Arc::new(crate::tools::GateToolset::new(
            mcp_manager.clone(),
            CoreMaterials::from_config(&config, builtin_filter.clone(), &sandbox),
        )));

    // Resolved here rather than during configuration: the persistent tiers are files and
    // environment, and a run that cannot load the guidance the user named must not start.
    let instructions = crate::instructions::resolve(config.request.instructions.as_deref())?;
    if let Some(found) = &instructions {
        crate::instructions::warn_if_large(found);
        tracing::info!("instructions loaded from {source}", source = found.source);
    }
    let user_instructions_source = instructions.as_ref().map(|found| found.source.to_string());
    let user_instructions = instructions.map(|found| found.text);

    let agent_options = AgentOptions::from_config(
        &config,
        sandboxed_shell,
        gate_tools.clone(),
        user_instructions.clone(),
    );

    // Kick off the MCP background connector once for the whole process. The connector writes tool
    // discoveries through `update_server_tools`, which fans them out to every attached registry,
    // so per-session registries built later via `build_session_agent` see the tools as servers
    // come online. Idempotent on second call.
    if let Some(manager) = &mcp_manager {
        manager.start_connector(crate::mcp::McpRuntimeConfig::from_config(&config));
    }

    Ok(SharedDeps {
        config,
        sandbox,
        gate_tools,
        user_instructions,
        user_instructions_source,
        default_profile,
        store,
        providers,
        mcp_manager,
        skills,
        memories,
        builtin_filter,
        agent_options,
        write_locks: crate::workspace::WriteLocks::default(),
    })
}
/// Per-session agent assembly behind [`build_session_agent`]. Builds the tool registry (with the
/// session's cwd / permission / frontend baked into the builtins, and `agent_spawn` and the
/// `context_*` tools registered by the builder itself), adds the MCP resource meta-tools, attaches
/// the registry to the MCP manager, and finally constructs the `Agent` itself.
///
/// The order this runs in relative to `start_connector` does not matter. `build_shared_deps` runs
/// the connector once for ACP and `serve`, before any session exists; the REPL runs it after this
/// returns. Either way every attached registry converges on the same tool set, because
/// [`crate::mcp::McpClientManager::update_server_tools`] writes the snapshot before fanning out and
/// [`crate::mcp::McpClientManager::subscribe`] replays the whole of it. What a late attach
/// costs is latency, not state: a session created while a slow server is still connecting sees that
/// server's tools when it lands.
pub(crate) async fn assemble_agent(
    materials: SessionMaterials,
    cells: SessionCells,
    agent_options: AgentOptions,
) -> anyhow::Result<Agent> {
    let tool_registry = ToolRegistry::build_default(&materials, &cells, &agent_options)?;
    if let Some(manager) = materials
        .mcp_manager
        .as_ref()
        .and_then(std::sync::Weak::upgrade)
    {
        // Attach this session's registry so the MCP connector and tools/list_changed handler
        // propagate updates into it, and so it picks up whatever has already been discovered.
        crate::tools::mcp_adapter::attach_session_registry(&manager, tool_registry.clone()).await;
    }
    Ok(Agent::new(
        &materials,
        cells,
        tool_registry,
        agent_options,
        crate::agent::AgentRole::Root,
    ))
}
/// What one session is built around: its row, if it exists yet, and the handles the host holds
/// before the agent exists. The gauges are the host's because a frontend built before the agent
/// (the REPL prompt, ACP's `usage_update`) has to hold the cell the agent publishes into rather
/// than a copy something re-stores beside every provider switch; `serve` keeps them so
/// `GET /v1/sessions/{id}/context` never waits on a turn.
pub(crate) struct SessionSpec {
    /// Which session this agent serves, or `None` for one that does not exist yet. It decides
    /// which profile the agent runs on: see `crate::provider::resolve_session_profile`.
    pub(crate) session_id: Option<uuid::Uuid>,
    pub(crate) permission: SharedPermission,
    pub(crate) frontend: Arc<dyn crate::frontend::Frontend>,
    pub(crate) cwd: crate::workspace::SharedCwd,
    pub(crate) roots: crate::workspace::SharedRoots,
    pub(crate) context_tokens: Arc<std::sync::atomic::AtomicU64>,
    pub(crate) context_overhead: Arc<std::sync::atomic::AtomicU64>,
    pub(crate) context_window: Arc<std::sync::atomic::AtomicU64>,
}
/// A session's lifetime counters: the row's, when it has one, so a resumed session continues its
/// `/status` totals wherever it is resumed; fresh otherwise, and fresh when the row cannot be read,
/// which is logged rather than fatal because the counters inform nothing that decides.
pub(crate) async fn session_stats_for(
    store: &Store,
    session_id: Option<uuid::Uuid>,
) -> Arc<crate::stats::SessionStats> {
    let Some(id) = session_id else {
        return Arc::new(crate::stats::SessionStats::default());
    };
    match store.load_session_stats(id).await {
        Ok(snapshot) => Arc::new(crate::stats::SessionStats::from_snapshot(&snapshot)),
        Err(error) => {
            tracing::warn!("failed to load session stats, starting fresh: {error}");
            Arc::new(crate::stats::SessionStats::default())
        }
    }
}
/// Build one session's `Agent` from the already-prepared [`SharedDeps`]: the one door every host
/// creates or reopens a session through. The registry it dispatches through and the cells it drives
/// are reachable from the agent; the registry is already attached to the MCP manager, and a host
/// detaches it again on the way out.
pub(crate) async fn build_session_agent(
    shared: &SharedDeps,
    spec: SessionSpec,
) -> anyhow::Result<Agent> {
    let SessionSpec {
        session_id,
        permission: shared_permission,
        frontend,
        cwd,
        roots,
        context_tokens,
        context_overhead,
        context_window,
    } = spec;
    refuse_a_spawned_session(&shared.store, session_id).await?;
    let resolved = crate::provider::resolved_profile(
        &shared.providers,
        crate::provider::profile_for_config(&shared.store, &shared.config, session_id).await?,
    )
    .await?;
    let mut cells = SessionCells::new(
        shared_permission,
        cwd,
        roots,
        // Published before the tools that read it are registered, and handed to the agent, which
        // is its only writer. This is what makes `/profile` and its two siblings reach
        // `agent_spawn` and the `context_*` gauge instead of moving the agent alone. The window
        // cell comes from the caller, so the host's own gauge *is* this one rather than a copy.
        crate::provider::PublishedProfile::new(&resolved, context_window),
        frontend,
    );
    cells.context_tokens = context_tokens;
    cells.context_overhead = context_overhead;
    let cells = match session_id {
        Some(id) => cells.with_session(id),
        None => cells,
    };
    let session_stats = session_stats_for(&shared.store, session_id).await;
    assemble_agent(
        shared.materials(session_stats),
        cells,
        shared.agent_options.clone(),
    )
    .await
}
/// What [`ResidentSession::release`] does, for an agent that never became a session: a REPL that
/// exits before its first turn has a lock slot, a registry and a cancel cell to let go of, and no
/// id.
pub(crate) async fn release_agent(
    agent: &Agent,
    cancel: &CancelCell,
    mcp_manager: Option<&Arc<crate::mcp::McpClientManager>>,
) -> usize {
    cancel.cancel();
    let stopped = agent.background_tasks().cancel_all().await;
    if let Some(manager) = mcp_manager {
        crate::tools::mcp_adapter::detach_session_registry(manager, agent.tool_registry()).await;
    }
    stopped
}
/// Where a session's conversation comes from when it is opened.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Opening {
    /// A row that was just created: nothing to load.
    Fresh,
    /// A row with history: hydrate it, retire what the last owner left running, and drop an
    /// orphaned tool call the provider would reject.
    Hydrate,
}
/// Open session `id` in this process: hydrate its conversation, build its agent and hold its lock.
/// Every host that makes a persisted session resident goes through here, so what it takes to reopen
/// one is decided once: `session/load`, `session/resume` and `session/fork` for ACP, re-attach and
/// `POST /v1/sessions` for HTTP, and a fresh row through [`Opening::Fresh`].
pub(crate) async fn open_session(
    shared: &SharedDeps,
    id: uuid::Uuid,
    spec: SessionSpec,
    opening: Opening,
    session_lock: crate::fs::FileLock,
    cancel: CancelCell,
) -> anyhow::Result<ResidentSession> {
    let conversation = match opening {
        Opening::Fresh => crate::conversation::Conversation::new(),
        Opening::Hydrate => hydrate_conversation(&shared.store, id).await?,
    };
    // Every door that opens a row names its id in the spec it hands over; the row id and the
    // spec's id are one fact, so a caller that lets them drift is a bug, not a case to paper over.
    debug_assert_eq!(
        spec.session_id,
        Some(id),
        "open_session was handed a spec for another id"
    );
    let agent = build_session_agent(shared, spec).await?;
    Ok(ResidentSession::new(
        id,
        agent,
        conversation,
        cancel,
        session_lock,
    ))
}
/// The profile a live session is asked to move to, as configured: refused by name when the
/// configuration has no such profile, and resolved through the registry otherwise so a missing
/// credential is refused here rather than on the next turn. The three mid-session switches ask
/// this: `/profile`, a resident `PATCH /v1/sessions/{id}` and ACP's `session/set_config_option`.
/// `--profile` on a resume and a dormant `PATCH` repin the row before any agent exists, so they
/// ask [`crate::config::require_profile`] directly and resolve at build.
pub(crate) async fn resolve_profile_switch(
    shared: &SharedDeps,
    name: &str,
) -> anyhow::Result<crate::provider::ResolvedProfile> {
    crate::config::require_profile(name, &shared.config.profiles)?;
    Ok(crate::provider::resolved_profile(&shared.providers, name.to_string()).await?)
}
/// Record a change a host has already applied to a live session on its row, and decide what a
/// failed write costs.
///
/// The persist-failure policy, applied once for every host; [`Store::update_session`] states it.
/// The level, the approvals switch, the working directory and the roots have moved in this process
/// by the time this runs, so failing the user's command over the write would be worse than a stale
/// row: the failure is warned about, loudly, because another process may still read the old value
/// (a scheduled gate is re-checked against the row, and the next resume opens from it), and the
/// caller continues. The profile is different: the row is the billing record, and a session running
/// on a profile its row does not name bills an account no reader can see. A patch carrying one
/// returns the failure, so the door refuses.
///
/// Every door handles the result the same way, surfacing whatever comes back; which patches can
/// fail is this function's to know, not theirs.
pub(crate) async fn record_session_change(
    store: &Store,
    session_id: uuid::Uuid,
    patch: crate::store::SessionPatch,
) -> crate::error::Result<()> {
    let carries_profile = patch.profile.is_some();
    let described = patch.to_string();
    match store.update_session(session_id, patch).await {
        Ok(()) => Ok(()),
        Err(error) if carries_profile => Err(error),
        Err(error) => {
            tracing::warn!("failed to record {described} for session '{session_id}': {error}");
            Ok(())
        }
    }
}
/// Record a switch on the session's row, which is what the next resume and every other process
/// read. Writing what the row already says is skipped: each write bumps `updated_at`, which idle
/// sweeps read, so a client re-sending its current value could keep a session resident forever.
pub(crate) async fn record_profile_switch(
    store: &Store,
    session_id: uuid::Uuid,
    resolved: &crate::provider::ResolvedProfile,
) -> anyhow::Result<()> {
    let recorded = store.recorded_profile(session_id).await?;
    if recorded.as_deref() == Some(resolved.profile.as_str()) {
        return Ok(());
    }
    record_session_change(store, session_id, crate::store::SessionPatch {
        profile: Some(resolved.profile.clone()),
        ..Default::default()
    })
    .await?;
    Ok(())
}
/// Say what a canceled turn left running.
///
/// Ctrl+C stops the turn, not the detached work, which is the shell's contract and the right
/// default. But the user may not have registered that anything was detached, so what survived has
/// to be visible rather than merely discoverable.
pub(crate) async fn report_background_survivors(agent: &Agent) {
    let running = agent.background_tasks().running_count_all().await;
    if running > 0 {
        crate::streams::write_stderr_line(format!(
            "{running} background task(s) still running; stop them with `/tasks cancel --all`."
        ));
    }
}
/// [`crate::host::claim_undelivered_outcomes`] for the REPL, which reaches these paths before
/// it has a session to claim against.
///
/// Stamped delivered *before* the turn runs, matching the scheduler's own claim and for the same
/// reason: an outcome that reliably wedges the process would otherwise be redelivered on every
/// restart, turning one bad result into a boot loop. Losing one report is the cheaper failure.
pub(crate) async fn collect_background_outcomes(
    agent: &crate::agent::Agent,
    store: &Store,
    session_id: Option<uuid::Uuid>,
) -> Vec<crate::store::background::BackgroundTask> {
    let Some(session_id) = session_id else {
        return Vec::new();
    };
    crate::host::claim_undelivered_outcomes(agent, store, session_id).await
}
/// Close the MCP servers on the way out.
///
/// [`crate::mcp::McpClientManager::shutdown`] takes `&self`, so this runs regardless of how many
/// owners the `Arc` still has. A `try_unwrap` here always fails: the manager holds the registries
/// it serves, and those registries hold six tools that each hold the manager back, so the graceful
/// path is unreachable and every run ends by leaving its stdio children to rmcp's drop guard.
pub(crate) async fn shutdown_mcp_manager(manager: Arc<crate::mcp::McpClientManager>) {
    manager.shutdown_within(crate::mcp::SHUTDOWN_BUDGET).await;
}
/// What `meka -c` / `-r` resolved to, and the settings the resumed session brings with it.
pub(crate) struct ResumedSession {
    pub(crate) session_id: Option<uuid::Uuid>,
    pub(crate) messages: crate::conversation::Conversation,
    pub(crate) lock: Option<crate::fs::FileLock>,
    /// The level this run starts at: the session's recorded one, unless `--permission` asked for
    /// something else, and the configured default for a run that resumed nothing.
    pub(crate) permission: crate::permission::Permission,
    /// The approvals switch this run starts with: the row's for a resume, the configured default
    /// for a run that resumed nothing.
    pub(crate) approvals: bool,
    /// The profile `--profile` asked this session to move to, still uncommitted. Carried out
    /// rather than written here because the row must not move until the profile is known to
    /// produce a provider, and that needs a registry this runs before.
    pub(crate) repin: Option<String>,
    /// The level `--permission` asked this session to run at, when it differs from the row's, also
    /// uncommitted and for the same reason: written here, a run that then failed to start left
    /// the row at a level it never ran at, for every other reader of it.
    pub(crate) permission_to_record: Option<crate::permission::Permission>,
    /// The directory the session recorded, which is where it reopens.
    ///
    /// Carried out rather than applied here because the caller owns the
    /// [`crate::workspace::SharedCwd`] and the fallback: `None` here means the row carried no
    /// directory (an imported archive may omit it) and the launch directory decides.
    ///
    /// **A resume never writes this column back.** The recorded directory is the session's, and at
    /// `workspace` it is also the writable boundary; correcting it to whatever shell happened to
    /// start the process would silently widen that boundary to, say, `$HOME`, and would let an
    /// unattended `--oneshot -c` from a unit at `/` repoint a session and every gate it holds.
    /// `meka serve` already reads the column this way ([`crate::host::http::reattach`]); ACP is
    /// the deliberate exception, because its client passes an authoritative project root per
    /// request.
    pub(crate) cwd: Option<std::path::PathBuf>,
}
pub(crate) async fn resolve_session_resume(
    store: &Store,
    config: &ResolvedConfig,
    console: &std::sync::Mutex<crate::console::Console>,
) -> anyhow::Result<ResumedSession> {
    let fresh = || ResumedSession {
        permission_to_record: None,
        session_id: None,
        messages: crate::conversation::Conversation::new(),
        lock: None,
        permission: config.permission,
        approvals: config.approvals,
        // A run that resumed nothing has no row to repin: the flags become this session's profile
        // when `resolve_session_profile` builds it, and are recorded when the row is created.
        repin: None,
        // Nor a directory to reopen: a fresh session starts where the shell is, and records that.
        cwd: None,
    };
    let resolved = match &config.request.session_resume {
        None => return Ok(fresh()),
        // `--continue` on a store with no sessions yet is not an error: there is simply nothing to
        // pick up, so the run starts fresh.
        Some(crate::config::SessionResume::Last) => store.last_session_id().await?,
        Some(crate::config::SessionResume::Id(value)) => {
            Some(store.resolve_session_id(value).await?)
        }
    };
    let Some(id) = resolved else {
        return Ok(fresh());
    };

    // Before the lock, the repin and the permission write. Refusing *after* this function returns
    // covers `--profile`, computed here and committed there, and misses `--permission`, which is
    // committed a few lines down: a run that declines to touch a session would rewrite its recorded
    // level on the way to saying so.
    refuse_a_spawned_session(store, Some(id)).await?;

    // Locked, then read, so the row this run resumes from is the one it now owns.
    let (lock, recorded) = store.open_session_row(id).await?;
    // `--profile` on a resume repins the session rather than applying for this run alone. A
    // per-run override would leave the row disagreeing with the conversation it describes, and the
    // next resume would silently move back; rewriting it keeps the row the answer to "what does
    // this session run on".
    //
    // Only computed here. `apply_session_repin` commits it, once the profile is known to produce a
    // provider.
    let repin = if let Some(requested) = &config.request.requested_profile {
        crate::config::require_profile(requested, &config.profiles)?;
        Some(requested.clone())
    } else {
        None
    };

    // The level the session recorded, unless this run asked for a different one. Every surface
    // resolves permission this way, through one admitter: ACP, the HTTP API, the scheduler's fire
    // door and `meka schedule` read the row and clamp it to `[permissions].enabled`, so a level the
    // operator has since withdrawn drops the session to the configured default rather than granting
    // authority the configuration no longer does. This function is what the REPL and `--oneshot`
    // share, so both read the row here; `--oneshot` is the half that matters most, a scripted run
    // whose level silently differing from the one the session was created with has nobody
    // watching it.
    let permission = config.request.requested_permission.or_else(|| {
        config
            .enabled_permissions
            .admit_recorded(recorded.permission, &format!("session {id}"))
    });
    // `--permission` rewrites the row for the reason `/permission` already does: a scheduled gate
    // is re-checked against it, and leaving it stale means another process acts on a level the user
    // has moved away from. Carried out to `record_resume_permission` rather than written here.
    let permission_to_record = config
        .request
        .requested_permission
        .filter(|requested| recorded.permission != Some(*requested));

    with_console(console, |console| {
        console.session_id("Continuing session", &id.to_string())
    });
    let messages = hydrate_conversation(store, id).await?;
    Ok(ResumedSession {
        session_id: Some(id),
        messages,
        lock: Some(lock),
        permission: permission.unwrap_or(config.permission),
        approvals: recorded.approvals,
        repin,
        permission_to_record,
        // Read off the row already loaded above, not fetched again.
        cwd: recorded.cwd,
    })
}
/// Record where `/cd` moved the session, so the row keeps saying where the session is.
///
/// A free function rather than a block inside the REPL's event loop because the loop is not
/// reachable from a test: reedline needs a terminal, and a piped stdin fails the read outright, so
/// everything inline there is verified only by running meka by hand.
///
/// The directory has already moved in this process, so a failed write is warned about and the REPL
/// continues; [`record_session_change`] states the policy. The consequence is real: this session's
/// next resume, and any scheduled gate it holds, still read the old directory.
pub(crate) async fn record_session_cwd(
    store: &Store,
    session_id: Option<uuid::Uuid>,
    path: &std::path::Path,
) {
    // No row yet: the directory the first turn creates the session with is read from the same cell
    // `/cd` has already written, so there is nothing to correct.
    let Some(id) = session_id else {
        return;
    };
    if let Err(error) = record_session_change(store, id, crate::store::SessionPatch {
        cwd: Some(path.to_path_buf()),
        ..Default::default()
    })
    .await
    {
        tracing::warn!("{error}");
    }
}
/// The directory a run opens in: the one its session recorded, or where meka was launched.
///
/// A session's directory is its own, so a resume reopens it rather than adopting whatever shell
/// started the process. At `workspace` that directory is also the writable boundary, and taking the
/// shell's instead would silently widen it: resume a project session from `$HOME` and the whole
/// home directory becomes writable, with a scheduled job able to fire before the user can react.
/// `meka serve` reads the column the same way (`crate::host::http::reattach`).
///
/// The launch directory is the fallback for two cases, both of which have to keep the run going:
/// a row that carries no directory (`meka session import` stores an archive's value verbatim, and
/// an archive may omit it) and one naming a directory that has since been removed.
pub(crate) fn resume_working_directory(
    recorded: Option<std::path::PathBuf>,
    launch_cwd: &std::path::Path,
    session_id: Option<uuid::Uuid>,
) -> std::path::PathBuf {
    let Some(recorded) = recorded else {
        return launch_cwd.to_path_buf();
    };
    if recorded.is_dir() {
        return recorded;
    }
    // Warn rather than fail: the conversation is still worth resuming, and the alternative is
    // refusing to open a session because a directory moved. Matches how a scheduled gate reports
    // the same loss (`crate::schedule`'s `run_shell_probe`).
    let session = session_id.map_or_else(|| "?".to_string(), |id| id.to_string());
    let recorded = recorded.display();
    let launch = launch_cwd.display();
    tracing::warn!(
        "session {session} records working directory '{recorded}', which no longer exists; \
         opening in '{launch}' instead"
    );
    launch_cwd.to_path_buf()
}
/// Commit a resume's `--profile` repin, ahead of building the agent, which resolves the row.
///
/// The repin and the level a resume asks for are written on opposite sides of the build. The repin
/// goes first, or the agent would be built on the profile the session is leaving; the level goes
/// after, through [`record_resume_permission`], or a start that then fails leaves the row at a
/// level it never ran at. Written together ahead of the build, a resume whose own profile could
/// not produce a provider had moved the level while refusing to run, and a `meka serve` sharing the
/// store authorized the session's gates at the new level.
pub(crate) async fn commit_resume_repin(
    store: &Store,
    providers: &crate::provider::ProviderRegistry,
    session_id: Option<uuid::Uuid>,
    repin: Option<String>,
) -> anyhow::Result<()> {
    if let (Some(session_id), Some(profile)) = (session_id, repin) {
        apply_session_repin(store, providers, session_id, profile).await?;
    }
    Ok(())
}

/// Record a resume's `--permission` on the row, once the agent has been built and the run is
/// known to start. See [`commit_resume_repin`] for why the two halves are split.
pub(crate) async fn record_resume_permission(
    store: &Store,
    session_id: Option<uuid::Uuid>,
    permission_to_record: Option<crate::permission::Permission>,
) {
    let (Some(session_id), Some(requested)) = (session_id, permission_to_record) else {
        return;
    };
    if let Err(error) = record_session_change(store, session_id, crate::store::SessionPatch {
        permission: Some(requested),
        ..Default::default()
    })
    .await
    {
        tracing::warn!("{error}");
    }
}

/// Commit a `--profile` repin, once the profile it names is known to produce a provider.
///
/// The row moves last, deliberately. Writing it first and only then discovering that the profile
/// has no stored credential leaves the session pinned to something that cannot run, and the profile
/// it had is nowhere in the output, so the next plain resume fails the same way with nothing left
/// to name the previous profile. Both other surfaces that offer this switch build the provider
/// before they write: `PATCH /v1/sessions/{id}` and ACP's `session/set_config_option`.
pub(crate) async fn apply_session_repin(
    store: &Store,
    providers: &crate::provider::ProviderRegistry,
    session_id: uuid::Uuid,
    profile: String,
) -> anyhow::Result<()> {
    let resolved = crate::provider::resolved_profile(providers, profile).await?;
    record_profile_switch(store, session_id, &resolved).await?;
    tracing::info!(
        "moved session {session_id} onto profile '{profile}'",
        profile = resolved.profile
    );
    Ok(())
}
pub(crate) async fn hydrate_conversation(
    store: &Store,
    session_id: uuid::Uuid,
) -> anyhow::Result<crate::conversation::Conversation> {
    // Retire whatever the last process left running before hydrating anything else; see
    // `crate::host::claim_session`.
    crate::host::claim_session(store, session_id).await;

    // Hydrate the event log directly: `load_events` decodes each stored row back into the `Event`
    // that wrote it, so resume rebuilds the log the last process ended with rather than a flattened
    // approximation of it.
    let mut events = store.load_events(session_id).await?;
    // The rows hold image references; a turn needs the bytes, so they come back here, once.
    store.inline_blobs(&mut events).await?;
    let mut log = crate::conversation::Conversation::from_events(events);

    // Drop assistant messages whose tool_use blocks lack matching tool_result blocks in the next
    // message. Anthropic's API rejects orphans; this sanitizes the log after a crash mid-tool-call.
    let dropped = log.sanitize_orphans();
    for message in &dropped {
        let tool_use_ids: Vec<String> = message
            .content
            .iter()
            .filter_map(|block| {
                if let crate::conversation::ContentBlock::ToolUse { id, .. } = block {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect();
        tracing::warn!(
            "dropping an assistant message with orphaned tool_use ids: {tool_use_ids:?}"
        );
    }

    // Materializing the log also replaces images whose bytes contradict their declared media type.
    // Providers sniff and reject those with a 400, and since the block is already in the log that
    // 400 would repeat on every request, leaving the session unusable. A stored session carrying
    // such a block heals on the way in, for free; all that is left to do is say so.
    let replaced = log.invalid_images_replaced();
    if replaced > 0 {
        tracing::warn!(
            "replaced {replaced} image(s) whose bytes did not match their declared media type"
        );
    }

    Ok(log)
}

impl SharedDeps {
    /// Whether this host may repeat an upstream's own response text to whoever it is answering.
    ///
    /// [`crate::host::relay_provider_errors`] states the key and its default; this is the reading
    /// every session-facing door takes, since the two long-lived hosts both hold a `SharedDeps` and
    /// the answer must not differ between them.
    pub(crate) fn relay_provider_errors(&self) -> bool {
        crate::host::relay_provider_errors(self.config.serve.as_ref())
    }

    /// The default profile, or the configuration's own explanation of why there is none. A host
    /// that creates sessions on the operator's behalf asks this once at startup.
    pub(crate) fn default_profile(&self) -> anyhow::Result<&str> {
        self.default_profile.as_deref().ok_or_else(|| {
            anyhow::anyhow!(self.config.provider_error.clone().unwrap_or_else(|| {
                "no profile is configured; run `meka profile add`".to_string()
            }))
        })
    }
}
impl SharedDeps {
    /// What every session this host builds is made of, around the one session's own counters.
    pub(crate) fn materials(
        &self,
        session_stats: Arc<crate::stats::SessionStats>,
    ) -> SessionMaterials {
        SessionMaterials {
            core: CoreMaterials {
                write_locks: self.write_locks.clone(),
                ..CoreMaterials::from_config(
                    &self.config,
                    self.builtin_filter.clone(),
                    &self.sandbox,
                )
            },
            skills: self.skills.clone(),
            skills_agent_managed: self.config.skills_agent_managed,
            memories: self.memories.clone(),
            store: self.store.clone(),
            providers: Arc::clone(&self.providers),
            mcp_manager: self.mcp_manager.as_ref().map(Arc::downgrade),
            session_stats,
            schedule: self.config.schedule.clone(),
            background: self.config.background.clone(),
            subagents: self.config.subagents.clone(),
            subagent_max_depth: self.config.subagent_max_depth,
        }
    }
}
