//! Assembling a session: the dependencies every session on this process shares, building an agent
//! for a fresh or resumed session, opening and releasing it, switching its provider, and the
//! resume-time reconciliation of what its row recorded.

use super::*;

/// Process-wide dependencies that every ACP session shares. Built once at `meka acp` startup by
/// [`build_shared_deps`]; sessions hold an [`Arc<SharedDeps>`] and read fields by reference.
/// Cheap to clone (every field is either an `Arc`, an owned-but-small value, or a clonable handle).
/// Every host builds one, the screen and `--oneshot` included, so all four assemble a session
/// through the same [`build_session_agent`].
#[derive(Clone)]
pub(crate) struct SharedDeps {
    pub(crate) config: Arc<ResolvedConfig>,
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
    /// What `[permissions].confine_reads` and `[shell].additional_readable_directories` resolve
    /// to, once: every session this host builds reads within it.
    pub(crate) read_boundary: crate::workspace::ReadBoundary,
    /// The dispatcher a scheduled gate's tool probe resolves against.
    pub(crate) gate_tools: Option<Arc<dyn crate::schedule::GateTools>>,
    /// The standing instructions and where they came from, resolved at startup and again when
    /// `PUT /v1/instructions` writes them, so a session that opens after the write runs under
    /// it; the text alone, since a session appends the files `[instructions].files` names when
    /// it opens, through [`session_agent_options`]. A session already open keeps what it opened
    /// under, as it keeps its provider.
    pub(crate) instructions: Arc<std::sync::RwLock<Option<crate::instructions::Instructions>>>,
    /// Shared by every session this host builds, so two of them writing one file serialize.
    pub(crate) write_locks: crate::workspace::WriteLocks,
    /// Shared by every session this host builds, so the host can say which sub-agents are in a
    /// turn.
    pub(crate) running_subagents: crate::session::RunningSubagents,
    /// Holds each write of the installation together with the read that serves the file after
    /// it: the file's own lock is released between an edit and its serving, and two writes
    /// interleaved across it would leave the manager running an entry the file no longer states,
    /// or the catalog a staler parse. Taken by [`Self::write_installation`] alone.
    installation_doors: Arc<tokio::sync::Mutex<()>>,
    /// The last whole read of `config.toml`: the one the process started on, and each serving of
    /// the file since.
    whole_read: Arc<std::sync::Mutex<WholeRead>>,
    /// What a host serves of `config.toml` beyond what every host serves, applied to each file
    /// [`Self::serve_installation`] serves, under the same doors: `meka serve`'s tokens.
    serve_also: Arc<std::sync::OnceLock<ServeHook>>,
}

/// A host's part of serving `config.toml`; see [`SharedDeps::serve_also`].
type ServeHook = Box<dyn Fn(&crate::config::ConfigFile) + Send + Sync>;

/// One whole read of `config.toml`: when it was made, and the file's modification time as it was
/// then, which a host compares with the file's now to say whether an edit is unread.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WholeRead {
    pub(crate) at: chrono::DateTime<chrono::Utc>,
    pub(crate) modified: Option<std::time::SystemTime>,
}

/// Why `config.toml` cannot be served as it stands: it fails to parse, holds an MCP server entry
/// the manager would refuse, or holds anything else startup would refuse. A write is refused for
/// it before anything is written.
#[derive(Debug)]
pub(crate) enum Unservable {
    /// Where the file fails to parse, and why, without quoting it.
    Parse(String),
    /// What the entry is refused for, naming it.
    Entry(String),
    /// What startup refuses the file for, typed, so a surface words a refusal whose remedy is a
    /// command with its own.
    Startup(crate::error::MekaError),
}

impl std::fmt::Display for Unservable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.worded(crate::error::MekaError::sentence))
    }
}

impl std::error::Error for Unservable {}

impl Unservable {
    /// Why the file cannot be served, a startup refusal in `word`'s terms: the shell's sentence
    /// by [`Display`](std::fmt::Display), a surface's own by its wording.
    pub(crate) fn worded(&self, word: impl Fn(&crate::error::MekaError) -> String) -> String {
        match self {
            Self::Parse(sentence) => {
                format!("config.toml fails to parse; repair it first: {sentence}")
            }
            Self::Entry(message) => format!(
                "config.toml holds an MCP server entry meka refuses; repair it first: {message}"
            ),
            Self::Startup(error) => format!(
                "config.toml holds a setting meka refuses at startup; repair it first: {}",
                word(error)
            ),
        }
    }

    /// The sentence for a file found unservable after a write landed in it, in `word`'s terms:
    /// an edit from outside broke it in between, so the write is in the file and not served.
    pub(crate) fn after_the_write(
        &self,
        word: impl Fn(&crate::error::MekaError) -> String,
    ) -> String {
        let (what, sentence) = match self {
            Self::Parse(sentence) => ("fails to parse", sentence.clone()),
            Self::Entry(message) => ("holds an MCP server entry meka refuses", message.clone()),
            Self::Startup(error) => ("holds a setting meka refuses at startup", word(error)),
        };
        format!(
            "the change is written, but config.toml as it now stands {what}, so it is not served; \
             repair it: {sentence}"
        )
    }
}

/// What serving the file brought the running manager to, by server name.
#[derive(Debug, Default)]
pub(crate) struct ServedServers {
    /// Servers the file names that were not running, started.
    pub(crate) added: Vec<String>,
    /// Servers whose connection or credential changed, restarted on the file's entry.
    pub(crate) restarted: Vec<String>,
    /// Servers whose tool policy alone changed, given it on the connection they hold.
    pub(crate) reconfigured: Vec<String>,
    /// Servers the file no longer names, taken out.
    pub(crate) removed: Vec<String>,
}

/// Why a write of the installation did not end with the file served.
#[derive(Debug)]
pub(crate) enum InstallationFailure<E> {
    /// The file as it stood could not be served, so nothing was written.
    Unservable(Unservable),
    /// The edit refused or failed, in its own terms.
    Edit(E),
    /// The edit landed and the file could not be served after it.
    Unserved(ServeFailure),
}

/// Why the file could not be served after a write landed in it.
#[derive(Debug)]
pub(crate) enum ServeFailure {
    /// An edit from outside, landing between the write and its serving, left the file as it now
    /// stands unservable.
    Unservable(Unservable),
    /// The store or the manager failed the serving.
    Fault(crate::error::MekaError),
}

impl std::fmt::Display for ServeFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unservable(unservable) => {
                f.write_str(&unservable.after_the_write(crate::error::MekaError::sentence))
            }
            Self::Fault(error) => error.fmt(f),
        }
    }
}

/// Why a write of the standing instructions did not end with them read again.
#[derive(Debug)]
pub(crate) enum InstructionsFailure<E> {
    /// The write failed, in its own terms.
    Write(E),
    /// The write landed and the instructions could not be read again.
    Reload(crate::error::MekaError),
}

impl From<InstallationFailure<crate::error::MekaError>> for crate::error::MekaError {
    /// A write of the installation whose edit speaks in [`crate::error::MekaError`], as one: the
    /// file's own defect as the refusal it is, the edit's and the serving's as they are.
    fn from(failure: InstallationFailure<crate::error::MekaError>) -> Self {
        match failure {
            InstallationFailure::Unservable(unservable) => {
                crate::installation::refusal(unservable.to_string())
            }
            InstallationFailure::Unserved(ServeFailure::Unservable(unservable)) => {
                crate::installation::refusal(
                    unservable.after_the_write(crate::error::MekaError::sentence),
                )
            }
            InstallationFailure::Edit(error)
            | InstallationFailure::Unserved(ServeFailure::Fault(error)) => error,
        }
    }
}

impl SharedDeps {
    /// The last whole read of `config.toml`.
    pub(crate) fn whole_read(&self) -> WholeRead {
        *crate::sync::lock(&self.whole_read)
    }

    /// The sections of `config.toml` that state something else than the file this process
    /// started on and are read at startup alone, so only a restart applies them.
    /// None while the file fails to parse: what it would change cannot be told, and the parse error
    /// is what a write or a reload answers.
    pub(crate) fn sections_awaiting_restart(&self) -> Vec<&'static str> {
        match crate::config::ConfigRead::now().sections() {
            Some(now) => crate::config::sections_awaiting_restart(&self.config.started_on, &now),
            None => Vec::new(),
        }
    }

    /// Write the installation: the one sequence every edit of `config.toml`'s accounts, profiles
    /// and MCP servers, and of the credentials stored for them, runs through on a host that
    /// serves the file. Under [`Self::installation_doors`], the file is read once and refused as
    /// [`Unservable`] when it could not be served; `edit` is handed that parse to decide on, the
    /// existence of what it names included, and runs its door; the file is then read whole,
    /// held to everything startup holds it to, and served ([`Self::serve_installation`]). An
    /// `edit` that changes nothing serves the file as it stands, which is a reload. The parse is
    /// [`Self::servable`]'s.
    ///
    /// An `edit` that fails undoes what it wrote, and the file is served all the same when it has
    /// moved since the read, so the host never runs something other than what the file states.
    pub(crate) async fn write_installation<T, E>(
        &self,
        edit: impl AsyncFnOnce(&crate::config::ConfigFile) -> std::result::Result<T, E>,
    ) -> std::result::Result<(T, ServedServers), InstallationFailure<E>> {
        let _doors = self.installation_doors.lock().await;
        let read = crate::config::ConfigRead::now();
        let config_file = self
            .servable(&read, read.modified != self.whole_read().modified)
            .map_err(InstallationFailure::Unservable)?;
        let value = match edit(&config_file).await {
            Ok(value) => value,
            Err(error) => {
                if crate::config::config_file_modified() != read.modified
                    && let Err(failure) = self.serve_installation().await
                {
                    tracing::warn!(
                        "failed to serve config.toml after a write that failed: {failure}"
                    );
                }
                return Err(InstallationFailure::Edit(error));
            }
        };
        let served = self
            .serve_installation()
            .await
            .map_err(InstallationFailure::Unserved)?;
        Ok((value, served))
    }

    /// Write the standing instructions with `write` and read them again for the sessions that
    /// open after it, under [`Self::installation_doors`]: serving the installation reads the same
    /// file, and a read it made before this write must not land after this one's.
    pub(crate) async fn write_instructions<E>(
        &self,
        write: impl FnOnce() -> std::result::Result<(), E>,
    ) -> std::result::Result<(), InstructionsFailure<E>> {
        let _doors = self.installation_doors.lock().await;
        write().map_err(InstructionsFailure::Write)?;
        self.reload_instructions()
            .map_err(InstructionsFailure::Reload)
    }

    /// The file as parsed, or why it could not be served, which a write is refused for and a door
    /// that begins what it later commits asks first: a parse error, an MCP server entry the
    /// manager would refuse, and, when the file changed since it was last read whole, whatever
    /// else startup refuses, asked of the file resolved under the command line this process
    /// started with. A file unchanged since then was held to that already, by startup or by the
    /// serving that recorded the read.
    pub(crate) fn installation_as_written(
        &self,
    ) -> std::result::Result<crate::config::ConfigFile, Unservable> {
        let read = crate::config::ConfigRead::now();
        self.servable(&read, read.modified != self.whole_read().modified)
    }

    /// `read` as parsed, or why it could not be served: a parse error, an MCP server entry the
    /// manager would refuse, and, when `in_full`, whatever else startup refuses, asked of the file
    /// resolved under the command line this process started with.
    fn servable(
        &self,
        read: &crate::config::ConfigRead,
        in_full: bool,
    ) -> std::result::Result<crate::config::ConfigFile, Unservable> {
        if let Some(refusal) = read.parse_refusal() {
            return Err(Unservable::Parse(refusal));
        }
        let config_file = read
            .parse_or_err()
            .map_err(|error| Unservable::Parse(error.sentence()))?;
        crate::mcp::sanitize::validate_server_entries(&config_file.mcp_servers())
            .map_err(Unservable::Entry)?;
        if in_full {
            let resolved =
                crate::config::ResolvedConfig::resolve_from(self.config.overrides.clone(), read);
            resolved.validate().map_err(Unservable::Startup)?;
            // Each reason names an operator's path or an expanded URL, a webhook's carrying its
            // secret, so it goes to the log and the refusal names the section.
            refuse_an_unbuildable_web_client(&resolved).map_err(|error| {
                tracing::warn!("config.toml's `[web]` cannot be started on: {error}");
                Unservable::Startup(crate::error::MekaError::Config(
                    "the `[web]` settings build no client; the server log says why".to_string(),
                ))
            })?;
            #[cfg(feature = "serve")]
            if self.config.overrides.serve {
                crate::host::http::resolve_serve(resolved.serve).map_err(|refusal| {
                    tracing::warn!("config.toml's `[serve]` cannot be started on: {refusal}");
                    Unservable::Startup(crate::error::MekaError::Config(
                        "the `[serve]` settings cannot be started on; the server log says why"
                            .to_string(),
                    ))
                })?;
            }
        }
        Ok(config_file)
    }

    /// Read `config.toml` whole and serve what it holds: the read is held to everything startup
    /// holds a file to, the accounts and profiles replace the catalog, an account stating no
    /// device id is given one, the MCP servers are brought to the file's entries, the standing
    /// instructions are read again, and the read is recorded. Asked under
    /// [`Self::installation_doors`] by [`Self::write_installation`] alone.
    async fn serve_installation(&self) -> std::result::Result<ServedServers, ServeFailure> {
        let read = crate::config::ConfigRead::now();
        // In full whatever changed: a door validates what it writes, and an edit from outside
        // landing between the read a door decided on and this one is in the file served.
        let config_file = self
            .servable(&read, true)
            .map_err(ServeFailure::Unservable)?;
        let wanted = config_file.mcp_servers();
        self.providers.replace_catalog(&config_file);
        if let Some(hook) = self.serve_also.get() {
            hook(&config_file);
        }
        // As at startup, so a session's first use of an account writes nothing into the file this
        // read did not see; the write is meka's own and counts as read.
        let wrote_device_ids = self.providers.preload_device_ids().await;
        let served = self
            .reconcile_servers(&wanted)
            .await
            .map_err(ServeFailure::Fault)?;
        *crate::sync::lock(&self.whole_read) = WholeRead {
            at: read.at,
            modified: if wrote_device_ids {
                crate::config::config_file_modified()
            } else {
                read.modified
            },
        };
        // The file is read and served; the standing instructions are another source's, read next.
        // One that cannot be read keeps the last text in force and is the log's to say: the write
        // this serves has landed, and answering it as a fault would invite a retry of a done write.
        if let Err(error) = self.reload_instructions() {
            tracing::warn!(
                "failed to read the standing instructions again; sessions keep the last text: \
                 {error}"
            );
        }
        tracing::debug!(
            "served config.toml: {added} MCP server(s) added, {restarted} restarted, \
             {reconfigured} reconfigured, {removed} removed",
            added = served.added.len(),
            restarted = served.restarted.len(),
            reconfigured = served.reconfigured.len(),
            removed = served.removed.len()
        );
        Ok(served)
    }

    /// Have every serving of `config.toml` also run `hook`, for what one host serves of the file
    /// beyond the rest. One per process; a second is ignored.
    pub(crate) fn serve_also(
        &self,
        hook: impl Fn(&crate::config::ConfigFile) + Send + Sync + 'static,
    ) {
        if self.serve_also.set(Box::new(hook)).is_err() {
            tracing::debug!("a host's serving hook is set already; the second is ignored");
        }
    }

    /// The profile `--profile` named when this host started, which the catalog selects as the
    /// default for as long as it runs.
    pub(crate) fn requested_profile(&self) -> Option<&str> {
        self.config.request.requested_profile.as_deref()
    }

    /// Bring the running manager to `wanted`, the file's entries: a server the file names and the
    /// manager does not run is started; one whose credentials have been written since its entry
    /// was built, or whose connection differs as the file would be written from each, is
    /// restarted on the file's entry; one whose tool policy alone differs takes it in place; one
    /// the file no longer names is taken out. Any other server is left as it is, connected or not,
    /// so serving the file is not a reconnect. A host without a manager (MCP disabled at startup)
    /// has nothing to bring anywhere.
    async fn reconcile_servers(
        &self,
        wanted: &[crate::config::McpServerConfig],
    ) -> crate::error::Result<ServedServers> {
        let Some(manager) = self.mcp_manager.as_ref() else {
            return Ok(ServedServers::default());
        };
        let generations = self
            .store
            .token_store()
            .mcp_credential_generations()
            .await?;
        let generation = |name: &str| generations.get(name).copied().unwrap_or(0);
        let running: std::collections::BTreeMap<String, Arc<crate::mcp::ServerEntry>> = manager
            .server_names()
            .into_iter()
            .filter_map(|name| manager.server_entry(&name).map(|entry| (name, entry)))
            .collect();
        let mut served = ServedServers::default();
        for config in wanted {
            let Some(entry) = running.get(&config.name) else {
                manager.add_server(config, generation(&config.name))?;
                served.added.push(config.name.clone());
                continue;
            };
            let running_entry = entry.configured();
            let restart = entry.credential_generation != generation(&config.name)
                || !crate::installation::mcp::same_connection(&running_entry, config);
            if restart {
                manager
                    .replace_server(config, generation(&config.name))
                    .await?;
                served.restarted.push(config.name.clone());
            } else if !crate::installation::mcp::same_entry(&running_entry, config) {
                // A listing the new policy cannot be had under restarts the server instead.
                if manager.reconfigure_server(config).await? {
                    served.reconfigured.push(config.name.clone());
                } else {
                    served.restarted.push(config.name.clone());
                }
            }
        }
        for name in running.keys() {
            if wanted.iter().any(|config| &config.name == name) {
                continue;
            }
            manager.remove_server(name).await?;
            served.removed.push(name.clone());
        }
        Ok(served)
    }
}
/// Whether a host may repeat an upstream's own response text to its caller: `[serve]
/// relay_provider_errors`, on unless the operator turned it off.
///
/// One reading of the key for both hosts that answer a caller over a wire. `meka serve` resolves it
/// once into [`crate::host::http::config::ResolvedServeConfig`] and `meka acp` asks per failure, so
/// stating the default in each of them is how the two came to be able to disagree about what a
/// deployment configured to withhold actually withholds. The screen and `--oneshot` never consult
/// it: their reader is the operator.
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
        &crate::tools::ToolScope::new(
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
    let Some(terms) = store.spawn_terms(session_id).await? else {
        return Ok(());
    };
    let remedy = match terms.parent {
        Some(_) => "; continue it with `agent_followup` there",
        // An imported sub-agent whose parent did not come with it. There is no session to point at,
        // so say what it is rather than naming a door that is not there.
        None => ", so it cannot be driven",
    };
    Err(crate::error::MekaError::SessionNotDrivable(format!(
        "{}{remedy}",
        terms.describe(session_id)
    ))
    .into())
}
/// Move a session's agent onto whatever its row currently names, before a turn runs on it.
///
/// **The row is the carrier, and the only one.** A host's own profile switch moves the agent
/// itself, but the row is the billing record, and a turn that ran on another profile than the
/// row names would bill an account the record does not. Another meka process may move a resident
/// row without its lock, as `meka profile rename` moves every row on the name, and a hand-edit may
/// too, so the row is read where the data is read, on every host's turn doors, the out-of-band
/// ones included, and a turn cannot run on, and bill, a profile the row has left.
///
/// Cheap when nothing has changed: one indexed row read and a comparison, with no resolution at
/// all unless the two differ. Must be called under the conversation lock, which is what makes
/// "the agent this turn is about to use" the thing being moved.
pub(crate) async fn apply_recorded_profile(
    shared: &SharedDeps,
    agent: &crate::agent::Agent,
    session_id: uuid::Uuid,
) -> anyhow::Result<()> {
    let Some(recorded) = shared.store.recorded_profile(session_id).await? else {
        // No row, so nothing names a profile to move to. Reachable only for a session deleted
        // from under a live entry; its turn is going to fail on the write either way.
        return Ok(());
    };
    if recorded == agent.profile() {
        return Ok(());
    }
    let profile = recorded.clone();
    let resolved = crate::provider::resolved_profile(&shared.providers, recorded).await?;
    agent.set_provider(resolved);
    tracing::info!("moved session {session_id} onto profile '{profile}'");
    Ok(())
}
/// Move a session's agent onto the project its row currently names, before a turn runs on it,
/// for the reason [`apply_recorded_profile`] gives: the row is the carrier, and a turn that ran
/// under another project would file its memories in a tier the record does not name. A project
/// the store no longer has is refused by name, as a profile the file lacks is.
pub(crate) async fn apply_recorded_project(
    shared: &SharedDeps,
    agent: &crate::agent::Agent,
    session_id: uuid::Uuid,
) -> anyhow::Result<()> {
    let Some(recorded) = shared.store.recorded_project(session_id).await? else {
        return Ok(());
    };
    let recorded = (!recorded.is_empty()).then_some(recorded);
    if recorded == agent.project() {
        return Ok(());
    }
    if let Some(name) = &recorded {
        shared.store.require_project(name).await?;
    }
    agent.set_project(recorded.clone());
    match recorded {
        Some(project) => tracing::info!("moved session {session_id} onto project '{project}'"),
        None => tracing::info!("moved session {session_id} off its project"),
    }
    Ok(())
}
/// Both of a row's bindings, applied ahead of a turn: the one call every turn door makes, so no
/// door applies the profile and forgets the project.
pub(crate) async fn apply_recorded_bindings(
    shared: &SharedDeps,
    agent: &crate::agent::Agent,
    session_id: uuid::Uuid,
) -> anyhow::Result<()> {
    apply_recorded_profile(shared, agent, session_id).await?;
    apply_recorded_project(shared, agent, session_id).await
}

/// Refuse `[web]` settings no client can be built from. Built here and dropped, because what it
/// produces is a refusal rather than a client: every registry builds its own from the same
/// settings, and the one thing no session can do is fail the *process* before it exists. Otherwise
/// a `ca_cert_file` that is not there or a proxy URL that is not one surfaces on the first turn,
/// per session, as whatever that host reports a registry failure as, which on `serve` hands the
/// operator's path to a remote caller. Asked at startup and of every file served.
fn refuse_an_unbuildable_web_client(config: &ResolvedConfig) -> crate::error::Result<()> {
    crate::tools::build_web_client(
        &config.web_client,
        crate::tools::web::web_fetch_refuses_own_addresses(config.network),
    )
    .map(|_| ())
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

    // Nothing is built here. The registry resolves a profile and loads its credential when a
    // session first asks, because which profiles this process will need is a property of the
    // sessions it ends up serving rather than of its configuration.
    let providers = Arc::new(crate::provider::ProviderRegistry::new(
        &config,
        store.token_store(),
    ));
    // Once, off the worker threads: a first ask would otherwise resolve a subscription profile's
    // device id inline, writing `config.toml` under the config lock on whatever thread asked.
    let wrote_device_ids = providers.preload_device_ids().await;
    // Test-only: hand the registry a scripted provider to return for every profile, whichever host
    // this process is. Installed rather than swapped into a rebuilt registry, so a harness driving
    // sessions on different profiles gets the script for all of them.
    #[cfg(any(debug_assertions, feature = "mock-provider"))]
    if let Some(request) = &config.mock_provider {
        let rounds = match &request.script {
            Some(path) => crate::provider::mock::load_script(path).map_err(|error| {
                anyhow::anyhow!("failed to load the mock provider script: {error}")
            })?,
            None => Vec::new(),
        };
        tracing::info!("MEKA_MOCK_PROVIDER=1: using scripted mock provider");
        providers.install_scripted(Arc::new(crate::provider::mock::MockProvider::from_rounds(
            rounds,
        )));
    }

    refuse_an_unbuildable_web_client(&config)?;

    // Detection, not configuration: which backend this machine can run is probed once here, and
    // the startup warning about an unusable or improvable sandbox is given from the same answer.
    let sandbox = crate::sandbox::resolve_backend(config.sandbox_backend, config.sandbox);
    crate::sandbox::warn_if_sandbox_issues(
        &crate::sandbox::SandboxState::new(
            config.sandbox,
            config.confine_reads,
            config.network,
            &sandbox,
        ),
        crate::sandbox::WarnContext::Startup,
    );
    let sandbox_capability = crate::sandbox::capability_from_probe(&sandbox.probe);
    // Resolved once: the base paths are canonicalized here, and an extra that does not resolve
    // is warned about once rather than per session.
    let read_boundary = crate::workspace::ReadBoundary::from_config(&config);
    let sandboxed_shell = config.sandbox
        && !matches!(
            sandbox_capability,
            crate::sandbox::SandboxCapability::Unavailable
        );

    // Both stores are instance-scoped, so one cache each serves every session this process runs.
    // `disabled()` is distinct from an empty store: it keeps the subsystem's tools out of the
    // registry entirely, which is the point of the config switch.
    let skills = if config.skills_enabled {
        crate::skills::SkillCache::discover(config.skills_additional_directories.clone())
    } else {
        crate::skills::SkillCache::disabled()
    };
    // Always connected, `enabled` carrying the config switch: it gates the agent's tools, not the
    // operator's access to a store that already exists.
    let memories = store.memory_store(config.memory_enabled, config.memory_inline_max_bytes);
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
            read_boundary.clone(),
            config.network,
        )));

    // Read here rather than during configuration: the tier is config's choice, the text is a
    // file's, and a run that cannot load the guidance the user named must not start.
    let instructions = crate::instructions::resolve(&config.standing_instructions)?;
    if let Some(found) = &instructions {
        tracing::info!("instructions loaded from {source}", source = found.source);
    }
    let user_instructions_count = instructions.as_ref().map_or(0, |found| found.count());
    let user_instructions = instructions.as_ref().map(|found| found.text.clone());

    let mut agent_options = AgentOptions::from_config(
        &config,
        sandboxed_shell,
        gate_tools.clone(),
        user_instructions,
    );
    agent_options.user_instructions_count = user_instructions_count;

    // Kick off the MCP background connector once for the whole process. The connector writes tool
    // discoveries through `update_server_tools`, which fans them out to every attached registry,
    // so per-session registries built later via `build_session_agent` see the tools as servers
    // come online. Idempotent on second call.
    if let Some(manager) = &mcp_manager {
        manager.start_connector(crate::mcp::McpRuntimeConfig::from_config(&config));
    }

    // A device id the preload wrote is meka's own write of the file this read saw, so the read
    // takes the file's time after it rather than reporting the write as an edit unread.
    let whole_read = Arc::new(std::sync::Mutex::new(WholeRead {
        at: config.read_at,
        modified: if wrote_device_ids {
            crate::config::config_file_modified()
        } else {
            config.file_modified
        },
    }));
    Ok(SharedDeps {
        config,
        sandbox,
        read_boundary,
        gate_tools,
        instructions: Arc::new(std::sync::RwLock::new(instructions)),
        store,
        providers,
        mcp_manager,
        skills,
        memories,
        builtin_filter,
        agent_options,
        write_locks: crate::workspace::WriteLocks::default(),
        running_subagents: crate::session::RunningSubagents::default(),
        installation_doors: Arc::new(tokio::sync::Mutex::new(())),
        whole_read,
        serve_also: Arc::new(std::sync::OnceLock::new()),
    })
}
/// Per-session agent assembly behind [`build_session_agent`]. Builds the tool registry (with the
/// session's cwd / permission / frontend baked into the builtins, and `agent_spawn` and the
/// `context_*` tools registered by the builder itself), adds the MCP resource meta-tools, attaches
/// the registry to the MCP manager, and finally constructs the `Agent` itself.
///
/// The order this runs in relative to `start_connector` does not matter. `build_shared_deps` runs
/// the connector once for ACP and `serve`, before any session exists; the screen runs it after this
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
/// (the screen prompt, ACP's `usage_update`) has to hold the cell the agent publishes into rather
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
    /// What the session reads beyond its workspace: each `--additional-readable-directory` on a
    /// run, under the screen, a one-shot run or ACP; nothing for a served session.
    pub(crate) read_only_roots: crate::workspace::SharedRoots,
    pub(crate) context_tokens: Arc<std::sync::atomic::AtomicU64>,
    pub(crate) context_overhead: Arc<std::sync::atomic::AtomicU64>,
    pub(crate) context_window: Arc<std::sync::atomic::AtomicU64>,
    /// The project the session is on: what a session that does not exist yet opens on and records
    /// at creation. Beside a `session_id` the row is read instead, since the row is the carrier,
    /// so a value here is at most the row's own repeated.
    pub(crate) project: Option<String>,
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
        read_only_roots,
        context_tokens,
        context_overhead,
        context_window,
        project,
    } = spec;
    refuse_a_spawned_session(&shared.store, session_id).await?;
    let resolved = crate::provider::resolved_profile(
        &shared.providers,
        crate::provider::profile_for_config(&shared.store, &shared.config, session_id).await?,
    )
    .await?;
    // The row's project where there is a row, refused by name when the store no longer has it,
    // as the profile is: a session cannot be opened on a tier that does not exist.
    let project = match session_id {
        Some(id) => shared
            .store
            .recorded_project(id)
            .await?
            .filter(|project| !project.is_empty()),
        None => project,
    };
    if let Some(name) = &project {
        shared.store.require_project(name).await?;
    }
    let agent_options = session_agent_options(shared, &cwd.get(), &resolved);
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
    cells.project = crate::session::SharedProject::new(project);
    // The process's read boundary, before the tools that judge by it are registered.
    cells.read_boundary = shared.read_boundary.clone();
    cells.read_only_roots = read_only_roots;
    cells.network = shared.config.network;
    if let Some(id) = session_id {
        cells.seed_context_tokens(&shared.store, id).await;
    }
    let cells = match session_id {
        Some(id) => cells.with_session(id),
        None => cells,
    };
    let session_stats = session_stats_for(&shared.store, session_id).await;
    assemble_agent(shared.materials(session_stats), cells, agent_options).await
}
/// This session's copy of the process-wide options: the standing text, with the files
/// `[instructions].files` names read against `directory` and appended, the whole of it measured
/// against the window of `profile`, which the session runs on.
///
/// Read here rather than beside the standing text at startup because a relative entry means the
/// session's directory, which `serve` and ACP take from the client per session and a resume from
/// the row, so the process has no one answer to give; and a set is large or not by the window it
/// rides in, which is the session's.
fn session_agent_options(
    shared: &SharedDeps,
    directory: &std::path::Path,
    profile: &crate::provider::ResolvedProfile,
) -> AgentOptions {
    let mut options = shared.agent_options.clone();
    // The standing text as it stands now, not as the process started: a `PUT /v1/instructions`
    // between the two is for the sessions that open after it.
    let standing = crate::sync::read(&shared.instructions).clone();
    let listed = if shared.config.instruction_files.is_empty() {
        None
    } else {
        crate::instructions::read_files(&shared.config.instruction_files, directory)
    };
    if let Some(found) = &listed {
        tracing::info!("instructions loaded from {source}", source = found.source);
    }
    crate::instructions::warn_if_large(
        &standing.iter().chain(listed.iter()).collect::<Vec<_>>(),
        &profile.profile,
        profile.context_window,
    );
    options.user_instructions_count = standing.as_ref().map_or(0, |found| found.count());
    options.user_instructions = standing.map(|found| found.text);
    options.user_instructions_count += listed.as_ref().map_or(0, |found| found.count());
    options.user_instructions = crate::instructions::join(options.user_instructions.take(), listed);
    options
}
/// What [`ResidentSession::release`] does, for an agent that never became a session: a screen that
/// exits before its first turn has a lock slot, a registry and a cancel cell to let go of, and no
/// id.
pub(crate) async fn release_agent(
    agent: &Agent,
    cancel: &CancelCell,
    mcp_manager: Option<&Arc<crate::mcp::McpClientManager>>,
) -> usize {
    // A release that still finds a turn is one a client asked for (`session/close`, a screen
    // leaving); a shutdown stops every turn as the server's own ahead of releasing anything.
    cancel.cancel(crate::store::turns::CancelReason::Client);
    // Recorded as the session let go of, ahead of the signal, as every bulk stop does: the work
    // would otherwise record its own interruption as a failure.
    let stopped = agent
        .background_tasks()
        .stop_all(
            &agent.store().background_store(),
            crate::store::background::TaskStatus::Interrupted,
        )
        .await;
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
    // The record a client reads next carries the checklist from the agent's cell, which starts
    // empty; the conversation just loaded is what the list is, so the cell reads it now rather
    // than at the first turn.
    agent.hydrate_checklist(&conversation).await;
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
    shared.providers.require_profile(name)?;
    Ok(crate::provider::resolved_profile(&shared.providers, name.to_string()).await?)
}
/// The project a live session is asked to move onto, refused by name when the store has no such
/// project. The mid-session moves ask this: `/project open`, a resident `PATCH
/// /v1/sessions/{id}` and ACP's `session/set_config_option`; moving off a project needs no
/// resolution.
pub(crate) async fn resolve_project_switch(
    shared: &SharedDeps,
    name: &str,
) -> anyhow::Result<String> {
    shared.store.require_project(name).await?;
    Ok(name.to_string())
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
/// on a profile its row does not name bills an account no reader can see. The project is different
/// for a like reason: the row says which tier the session's memories are filed in, and every
/// other process reads it there. The title and the pin are different for the opposite reason: they
/// live on the row and nowhere else, so a write that failed is the whole command failing. A patch
/// carrying any of the four returns the failure, so the door refuses.
///
/// Every door handles the result the same way, surfacing whatever comes back; which patches can
/// fail is this function's to know, not theirs.
pub(crate) async fn record_session_change(
    store: &Store,
    session_id: uuid::Uuid,
    patch: crate::store::SessionPatch,
) -> crate::error::Result<()> {
    let must_land = patch.profile.is_some()
        || patch.project.is_some()
        || patch.title.is_some()
        || patch.pinned.is_some();
    let described = patch.to_string();
    match store.update_session(session_id, patch).await {
        Ok(()) => Ok(()),
        Err(error) if must_land => Err(error),
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
/// Record a project move on the session's row, `None` for off its project, skipping a write of
/// what the row already says for the reason [`record_profile_switch`] gives.
pub(crate) async fn record_project_switch(
    store: &Store,
    session_id: uuid::Uuid,
    project: Option<&str>,
) -> anyhow::Result<()> {
    let recorded = store.recorded_project(session_id).await?;
    if recorded.as_deref() == Some(project.unwrap_or_default()) {
        return Ok(());
    }
    record_session_change(store, session_id, crate::store::SessionPatch {
        project: Some(project.unwrap_or_default().to_string()),
        ..Default::default()
    })
    .await?;
    match project {
        Some(project) => tracing::info!("recorded project '{project}' on session {session_id}"),
        None => tracing::info!("recorded no project on session {session_id}"),
    }
    Ok(())
}
/// [`crate::host::claim_undelivered_outcomes`] for the screen, which reaches these paths before
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
    /// The project the session of this run is on: `--project` for a fresh session, checked
    /// against the store; a resumed row's own, moved by `--project` where `-r <id>` named it.
    /// What the banner shows and the opening skill is checked against; the build reads a row's
    /// project off the row.
    pub(crate) project: Option<String>,
    /// The project `--project` asked a named session to move onto, still uncommitted like the
    /// profile's repin and written beside it.
    pub(crate) project_repin: Option<String>,
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
    /// The session to announce as resumed, when `[display].show_session_id_on_resume` asks for
    /// it. The caller prints it, since where a banner goes is the host's business: ahead of the
    /// first prompt on a stream, or as the first item on a screen.
    pub(crate) announce: Option<uuid::Uuid>,
}
pub(crate) async fn resolve_session_resume(
    store: &Store,
    config: &ResolvedConfig,
) -> anyhow::Result<ResumedSession> {
    // Checked first, whatever the run resumes: a name the store lacks is refused before a lock is
    // taken or a row touched.
    let requested_project = config.request.requested_project.as_deref();
    if let Some(name) = requested_project {
        if name.is_empty() {
            anyhow::bail!("`--project` needs a project name; leave it out for a session on none");
        }
        store.require_project(name).await?;
    }
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
        // The project likewise: the session opens on it and the row records it at creation.
        project: requested_project.map(str::to_string),
        project_repin: None,
        // Nor a directory to reopen: a fresh session starts where the shell is, and records that.
        cwd: None,
        announce: None,
    };
    let resolved = match &config.request.session_resume {
        None => return Ok(fresh()),
        // `--continue` on a store with no sessions yet is not an error: there is simply nothing to
        // pick up, so the run starts fresh. With `--project`, the latest session on that project,
        // or a fresh one on it.
        Some(crate::config::SessionResume::Last) => {
            store.last_session_id(requested_project).await?
        }
        Some(crate::config::SessionResume::Id(value)) => {
            Some(store.resolve_session_id(value).await?)
        }
        // The pick is made on the screen once it is up; until then there is nothing to resume.
        Some(crate::config::SessionResume::Pick) => return Ok(fresh()),
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
    // `--project` moves a session the user named, and filters a selection meka made: with `-r
    // <id>` the row is repinned, with `-c` the project chose the session above and the row
    // already says so.
    let project_repin = match &config.request.session_resume {
        Some(crate::config::SessionResume::Id(_)) => requested_project.map(str::to_string),
        _ => None,
    };

    // The level the session recorded, unless this run asked for a different one. Every surface
    // resolves permission this way, through one admitter: ACP, the HTTP API, the scheduler's fire
    // door and `meka schedule` read the row and clamp it to `[permissions].enabled`, so a level the
    // operator has since withdrawn drops the session to the configured default rather than granting
    // authority the configuration no longer does. This function is what the screen and `--oneshot`
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

    let messages = hydrate_conversation(store, id).await?;
    let project = project_repin
        .clone()
        .or_else(|| (!recorded.project.is_empty()).then(|| recorded.project.clone()));
    Ok(ResumedSession {
        session_id: Some(id),
        messages,
        lock: Some(lock),
        permission: permission.unwrap_or(config.permission),
        approvals: recorded.approvals,
        repin,
        permission_to_record,
        project,
        project_repin,
        // Read off the row already loaded above, not fetched again.
        cwd: recorded.cwd,
        announce: config.show_session_id_on_resume.then_some(id),
    })
}
/// Record where `/cd` moved the session, so the row keeps saying where the session is.
///
/// A free function rather than a block inside the host loop because the loop is reachable from a
/// test only through a pseudo-terminal: the screen refuses a piped stdin outright, so what is
/// inline there is verified through `tests/tui_pty.rs` alone.
///
/// The directory has already moved in this process, so a failed write is warned about and the
/// screen continues; [`record_session_change`] states the policy. The consequence is real: this
/// session's next resume, and any scheduled gate it holds, still read the old directory.
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
    project_repin: Option<String>,
) -> anyhow::Result<()> {
    if let (Some(session_id), Some(profile)) = (session_id, repin) {
        apply_session_repin(store, providers, session_id, profile).await?;
    }
    if let (Some(session_id), Some(project)) = (session_id, project_repin) {
        record_project_switch(store, session_id, Some(&project)).await?;
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
/// Make a persisted conversation resident for a host: retire whatever the last process left
/// running on the session, then load the view through `Store::load_conversation`.
pub(crate) async fn hydrate_conversation(
    store: &Store,
    session_id: uuid::Uuid,
) -> anyhow::Result<crate::conversation::Conversation> {
    // Retire whatever the last process left running before hydrating anything else; see
    // `crate::host::claim_session`.
    crate::host::claim_session(store, session_id).await;
    Ok(store.load_conversation(session_id).await?)
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

    /// The default profile, or why there is none, for the host to word. `meka acp` asks this
    /// once at startup; `meka serve` asks it for each session that names no profile, of the
    /// catalog as it stands then.
    pub(crate) fn default_profile(
        &self,
    ) -> std::result::Result<String, crate::config::ProfileRefusal> {
        self.providers.catalog().default_profile.clone()
    }

    /// Read the standing instructions again from the tier this process resolves them from, after
    /// a write this process made to the conventional file, for the sessions that open after it.
    pub(crate) fn reload_instructions(&self) -> crate::error::Result<()> {
        let found = crate::instructions::resolve(&self.config.standing_instructions)?;
        *crate::sync::write(&self.instructions) = found;
        Ok(())
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
                running_subagents: self.running_subagents.clone(),
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both arms, against real rows rather than a fabricated `parent_id`.
    ///
    /// The refusal is what keeps a sub-agent's spawn terms meaningful, so a predicate that answered
    /// wrongly in either direction is serious in both directions: admitting a sub-agent reopens the
    /// escalation, and refusing a root session would break every host at once.
    ///
    /// `None` is asserted too, because that is what every fresh session passes and a check that
    /// tried to read a row for it would refuse the case it exists to allow.
    #[tokio::test]
    async fn a_session_another_one_spawned_cannot_be_built_as_a_plain_agent() {
        let manager = crate::store::Store::for_test().await;
        let parent = manager
            .create_session(None, "profile")
            .await
            .expect("a root session");
        let (sub_agent, _lock) = manager
            .create_child_session_for_test(
                parent,
                None,
                Vec::new(),
                None,
                "read".to_string(),
                "profile".to_string(),
            )
            .await
            .expect("a sub-agent of that session");

        refuse_a_spawned_session(&manager, None)
            .await
            .expect("a session that does not exist yet has nothing to refuse");
        refuse_a_spawned_session(&manager, Some(parent))
            .await
            .expect("a root session is the ordinary case and must be admitted");

        let error = refuse_a_spawned_session(&manager, Some(sub_agent))
            .await
            .expect_err("a session with a parent cannot be built as a plain agent");
        assert!(
            matches!(
                error.downcast_ref::<crate::error::MekaError>(),
                Some(crate::error::MekaError::SessionNotDrivable(_))
            ),
            "the refusal has to be the variant the HTTP layer answers 422 for: {error}"
        );
        // Both ids and the way through, because a client handed a sub-agent's id may not know what
        // spawned it, and "no" without a remedy sends it looking for a bug in meka.
        let text = error.to_string();
        assert!(
            text.contains(&sub_agent.to_string())
                && text.contains(&parent.to_string())
                && text.contains("agent_followup"),
            "the refusal must name the sub-agent, its parent, and the door that can: {text}"
        );
    }

    /// A sub-agent whose parent did not survive an import is still a sub-agent.
    ///
    /// `session export` on a sub-agent alone writes a `parent_id` pointing outside the archive, and
    /// `import_sessions` resolves an unknown parent to `NULL` while copying `subagent_spec_json`
    /// verbatim. Keying the refusal on the parent alone therefore left export-then-import as a
    /// two-command promotion of a sub-agent into a drivable root session -- the same laundering
    /// `fork_session` was fixed for, one door over, and reproducible with a real shell.
    ///
    /// The spec is the fact worth refusing on: a row holding the terms another session spawned it
    /// under is a sub-agent's conversation whether or not the link survived.
    #[tokio::test]
    async fn spawn_terms_without_a_parent_are_still_a_sub_agent() {
        let manager = crate::store::Store::for_test().await;
        // Written through the real importer, because that is the only door that produces this
        // shape: `plan_import` resolves a parent outside the archive to `None` and carries
        // `subagent_spec_json` regardless.
        let orphaned = uuid::Uuid::new_v4();
        manager
            .import_sessions(
                vec![crate::store::ImportSessionRecord {
                    new_id: orphaned,
                    new_parent_id: None,
                    created_at: "2026-08-31T00:00:00Z".to_string(),
                    cwd: None,
                    permission: crate::permission::Permission::Read,
                    approvals: false,
                    capabilities_json: None,
                    additional_roots: Vec::new(),
                    subagent_spec_json: Some("{\"tools\":[]}".to_string()),
                    profile: "profile".to_string(),
                    project: String::new(),
                    title: None,
                    pinned_at: None,
                    stats: Default::default(),
                    turns: Vec::new(),
                    events: Vec::new(),
                    scratchpad_entries: Vec::new(),
                }],
                Vec::new(),
            )
            .await
            .expect("the archive a lone sub-agent exports to");

        let error = refuse_a_spawned_session(&manager, Some(orphaned))
            .await
            .expect_err("spawn terms with no parent are what an imported sub-agent looks like");
        assert!(
            matches!(
                error.downcast_ref::<crate::error::MekaError>(),
                Some(crate::error::MekaError::SessionNotDrivable(_))
            ),
            "the same refusal the parented case gets: {error}"
        );
        let text = error.to_string();
        assert!(
            text.contains(&orphaned.to_string()) && !text.contains("agent_followup"),
            "there is no parent to point at, so it must not name a door that is not there: {text}"
        );
    }

    /// `/cd` reaches the row, which is what makes the recorded directory mean "where the session
    /// is" rather than "where it was created". Nothing else covers this: the screen loop that sends
    /// the event needs a terminal, so a test cannot drive `/cd` itself.
    #[tokio::test]
    async fn a_cd_records_the_directory_on_the_session_row() {
        let manager = crate::store::Store::for_test().await;
        let temp = tempfile::tempdir().expect("tempdir");
        let moved = crate::workspace::canonical_for_test(temp.path());
        let id = manager
            .create_session(
                Some(std::path::PathBuf::from("/somewhere/else")),
                "p".to_string(),
            )
            .await
            .expect("create");

        record_session_cwd(&manager, Some(id), &moved).await;

        let recorded = manager
            .session_info(id)
            .await
            .expect("read the row")
            .and_then(|info| info.cwd);
        assert_eq!(
            recorded,
            Some(moved),
            "the row must say where `/cd` moved the session",
        );
    }

    /// Before the first turn there is no row to correct: the directory the creation snapshot reads
    /// is the cell `/cd` has already written. So this is a no-op, and in particular it must not
    /// reach for some other session's row -- a screen sharing a store with a `meka serve` has
    /// plenty to choose from.
    #[tokio::test]
    async fn a_cd_before_the_first_turn_writes_nobody_elses_row() {
        let manager = crate::store::Store::for_test().await;
        let temp = tempfile::tempdir().expect("tempdir");
        let bystander = manager
            .create_session(
                Some(std::path::PathBuf::from("/its/own/place")),
                "p".to_string(),
            )
            .await
            .expect("create");

        record_session_cwd(&manager, None, temp.path()).await;

        let untouched = manager
            .session_info(bystander)
            .await
            .expect("read the row")
            .and_then(|info| info.cwd);
        assert_eq!(
            untouched,
            Some(std::path::PathBuf::from("/its/own/place")),
            "a `/cd` with no session of its own must leave every other row alone",
        );
    }

    /// A resumed session opens where it was recorded. At `workspace` that directory is also the
    /// writable boundary, so adopting the shell's would widen it behind the user's back.
    #[test]
    fn a_resume_prefers_the_recorded_directory_over_the_launch_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let recorded = temp.path().join("project");
        let launch = temp.path().join("elsewhere");
        std::fs::create_dir_all(&recorded).expect("recorded dir");
        std::fs::create_dir_all(&launch).expect("launch dir");

        assert_eq!(
            resume_working_directory(Some(recorded.clone()), &launch, None),
            recorded,
        );
    }

    /// The two cases that have to keep the run going: a row carrying no directory (an imported
    /// archive may omit it) and one naming a directory that has since been removed.
    #[test]
    fn a_resume_falls_back_to_the_launch_directory_when_the_recording_cannot_serve() {
        let temp = tempfile::tempdir().expect("tempdir");
        let launch = temp.path().join("elsewhere");
        std::fs::create_dir_all(&launch).expect("launch dir");

        assert_eq!(resume_working_directory(None, &launch, None), launch);
        assert_eq!(
            resume_working_directory(Some(temp.path().join("gone")), &launch, None),
            launch,
        );
        // A file is not a directory to open in either, and `is_dir` is what separates them.
        let file = temp.path().join("a-file");
        std::fs::write(&file, b"x").expect("write file");
        assert_eq!(resume_working_directory(Some(file), &launch, None), launch);
    }
}
