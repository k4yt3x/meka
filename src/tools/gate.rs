//! The tool set a scheduler gate probe runs against: silent, unattended, and limited to what a
//! probe may touch.

use super::*;

/// Dispatches a scheduled gate's [`crate::schedule::GateProbe::Tool`] probe.
///
/// Two lookup paths because the two tool families differ in one respect that matters here. An MCP
/// adapter wraps a shared server connection and is independent of any session, so it comes straight
/// from the process-wide manager's snapshot. A built-in bakes cwd, permission and frontend in at
/// construction (`register_core_tools`), so it is built per call against the job's own session.
///
/// Built once per host and shared by every job, which is why it holds the config slice rather than
/// a prebuilt registry: the registry cannot be shared, because the cwd belongs to the job.
pub(crate) struct GateToolset {
    pub(super) mcp: Option<std::sync::Arc<crate::mcp::McpClientManager>>,
    pub(super) core: crate::session::CoreMaterials,
    /// The level built-ins are constructed at.
    ///
    /// Always `Read`, and not the session's level: a gate may only call a tool that resolves to
    /// `read`, so a probe that needed more than this has already been refused by
    /// `gate_probe_is_authorized`. Handing the registry a higher level would let a future
    /// write-capable probe through the door the authority check is guarding. Built on first use by
    /// [`GateToolset::resolution_registry`]; see there for why it is cached.
    pub(super) resolution: std::sync::OnceLock<Option<std::sync::Arc<ToolRegistry>>>,
}
impl std::fmt::Debug for GateToolset {
    /// Deliberately opaque. The interesting field is a live MCP manager, and the rest is a config
    /// slice already printed by whoever owns it; a scheduler log line wants to know a dispatcher
    /// exists, not to unfold one.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GateToolset")
            .field("mcp", &self.mcp.is_some())
            .finish_non_exhaustive()
    }
}
impl GateToolset {
    pub(crate) fn new(
        mcp: Option<std::sync::Arc<crate::mcp::McpClientManager>>,
        core: crate::session::CoreMaterials,
    ) -> Self {
        Self {
            mcp,
            core,
            resolution: std::sync::OnceLock::new(),
        }
    }

    /// The level a gate's built-ins are constructed and dispatched at.
    ///
    /// `Read` is the only level a probe can have been admitted at, and it is also the only one that
    /// leaves `execute_command` confined: `Confinement::resolve` spawns a bare shell only at
    /// `unrestricted`, so handing this registry that level would turn the one read-level tool that
    /// runs arbitrary code into an unsandboxed command on a timer.
    ///
    /// Derived at the point of use rather than stored on the struct: every test builds the struct
    /// by literal and supplies its own fields, so a stored value could be raised to `Unrestricted`
    /// without any test noticing.
    pub(super) fn dispatch_permission() -> crate::permission::SharedPermission {
        crate::permission::SharedPermission::new(
            crate::permission::Permission::Read,
            crate::permission::EnabledPermissions::DEFAULT,
        )
    }

    /// Build the per-call built-in registry, rooted at the job's directory.
    ///
    /// Only [`crate::schedule::GateTools::call`] needs this. Resolution goes through
    /// [`Self::resolution_registry`] instead, because what a tool *requires* does not depend on
    /// where it would run, and building one of these is not free: `register_core_tools`
    /// constructs a web client, which builds a TLS stack and reads `[web].ca_cert_file` from disk.
    pub(super) fn builtins(&self, cwd: Option<&std::path::Path>) -> Result<ToolRegistry> {
        let directory = cwd
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        ToolRegistry::for_gate(
            &self.core,
            crate::session::ToolSite::detached(
                Self::dispatch_permission(),
                crate::workspace::SharedCwd::new(directory),
            ),
        )
    }

    /// The registry that answers what a built-in *requires*, built once and reused.
    ///
    /// Separate from [`Self::builtins`] because resolution is asked far more often than dispatch:
    /// once per gated job per agent turn, once per `schedule_list`, and once per scheduler sweep.
    /// A directory of `.` is fine here, since nothing about a permission depends on the cwd.
    pub(super) fn resolution_registry(&self) -> &ToolRegistry {
        self.resolution
            .get_or_init(|| self.builtins(None).map(std::sync::Arc::new).ok())
            .as_ref()
            .map_or(&EMPTY_GATE_REGISTRY, |registry| registry.as_ref())
    }
}
/// Stands in when the gate's resolution registry could not be built, so every built-in name
/// resolves to "unavailable" rather than to a level nothing verified. Failing closed here costs a
/// refused gate; failing open would run one on authority meka never established.
pub(super) static EMPTY_GATE_REGISTRY: std::sync::LazyLock<ToolRegistry> =
    std::sync::LazyLock::new(ToolRegistry::new);
#[async_trait]
impl crate::schedule::GateTools for GateToolset {
    fn resolve(&self, name: &str) -> Option<crate::permission::Permission> {
        // The tool has to exist before its level means anything, and the two families are looked up
        // differently: an MCP adapter comes from the process-wide snapshot, a built-in from the
        // gate's own registry.
        //
        // The MCP read is sync, so the snapshot read runs on the caller's runtime. It is a map read
        // behind an `RwLock` whose writer never holds it across an await, so it cannot block for
        // long.
        let hardcoded = if name.starts_with("mcp__") {
            let manager = self.mcp.as_ref()?;
            futures::executor::block_on(manager.tool_by_name(name))?.permission
        } else {
            self.resolution_registry().get(name)?.required_permission()
        };
        // Then the operator's override, in the same order dispatch applies it
        // (`ToolRegistry::required_permission_for`): the tool alone answers its hardcoded level,
        // and a `tool_permissions` entry that raised a tool out of reach in conversation must not
        // leave it admitted as a gate. Applied by name rather than by family because dispatch
        // consults the map for every registered name, MCP tools included.
        //
        // Requiring the tool to exist first is what keeps this from failing open: the map alone
        // would admit a probe that matches nothing, if a stale entry happened to give it `read`.
        Some(
            self.core
                .builtin_filter
                .permission_overrides
                .get(name)
                .copied()
                .unwrap_or(hardcoded),
        )
    }

    fn is_still_connecting(&self, name: &str) -> bool {
        match &self.mcp {
            Some(manager) => futures::executor::block_on(manager.server_is_still_connecting(name)),
            None => false,
        }
    }

    async fn call(
        &self,
        name: &str,
        arguments: &serde_json::Value,
        timeout: std::time::Duration,
        cwd: Option<&std::path::Path>,
        session_id: Option<uuid::Uuid>,
    ) -> std::result::Result<crate::schedule::ProbeOutcome, String> {
        let tool = if name.starts_with("mcp__") {
            match &self.mcp {
                Some(manager) => super::mcp_adapter::tool_by_name(manager, name).await,
                None => None,
            }
        } else {
            self.builtins(cwd)
                .map_err(|error| format!("failed to build the gate's tools: {error}"))?
                .get(name)
        }
        .ok_or_else(|| format!("gate tool '{name}' is not available"))?;
        let (arguments, _detach) =
            match crate::tools::admit_arguments(name, arguments, &tool.definition().parameters) {
                Ok(admitted) => admitted,
                Err(refusal) => {
                    return Err(format!(
                        "gate tool '{}' refused its arguments: {}",
                        name,
                        flatten_tool_text(&refusal)
                    ));
                }
            };

        // The same budget a shell gate gets. A tool that hangs must not hold the sweep open, and
        // `execute` is canceled by dropping the future.
        let cancellation = tokio_util::sync::CancellationToken::new();
        // A probe has no user watching it, so its prompts, progress and output go nowhere.
        let context = crate::tools::ToolContext {
            session_id,
            tool_call_id: None,
            prompt_id: None,
            frontend: std::sync::Arc::new(crate::frontend::SilentFrontend),
            cancellation: cancellation.clone(),
        };
        let output = match tokio::time::timeout(timeout, tool.execute(arguments, context)).await {
            Ok(Ok(output)) => output,
            Ok(Err(error)) => return Err(format!("gate tool '{name}' failed: {error}")),
            Err(_) => {
                cancellation.cancel();
                return Err(format!(
                    "gate tool '{}' exceeded its {} budget",
                    name,
                    humantime_serde::re::humantime::format_duration(timeout)
                ));
            }
        };

        // `is_error` is the tool saying "I ran and it did not work", which is a *result* a
        // predicate may legitimately watch for, not a broken gate. Only an unreachable tool
        // is an error here.
        Ok(crate::schedule::ProbeOutcome::new(
            &flatten_tool_text(&output),
            output.structured.clone(),
            !output.is_error,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BuiltinToolFilter;

    /// A gate resolves a built-in through the registry, so `[tools] tool_permissions` applies.
    ///
    /// The override lives on the registry and the tool object knows nothing about it, so asking
    /// the tool admits a probe the operator put out of reach. Every other tool-gate test drives a
    /// stub resolver and cannot see this: replacing `GateToolset::resolve`'s body with
    /// `Some(Permission::Read)` leaves them all green.
    #[test]
    fn a_gate_resolves_a_builtin_through_the_registry_so_overrides_apply() {
        let toolset = |overrides: HashMap<String, Permission>| GateToolset {
            mcp: None,
            core: crate::session::CoreMaterials {
                sandbox_enabled: false,
                sandbox_capability: crate::sandbox::SandboxCapability::Unavailable,
                builtin_filter: BuiltinToolFilter::from_config(None, Vec::new(), overrides),
                ..crate::session::CoreMaterials::for_test()
            },
            resolution: std::sync::OnceLock::new(),
        };
        use crate::schedule::GateTools;

        assert_eq!(
            toolset(HashMap::new()).resolve("fetch_url"),
            Some(Permission::Read),
            "unconfigured, `fetch_url` is a read-only probe and a legitimate gate"
        );

        let raised = toolset(HashMap::from([(
            "fetch_url".to_string(),
            Permission::Unrestricted,
        )]));
        assert_eq!(
            raised.resolve("fetch_url"),
            Some(Permission::Unrestricted),
            "an operator who raised it must be obeyed here as well as at dispatch"
        );
        assert!(
            crate::schedule::gate_probe_is_authorized(
                &crate::schedule::GateProbe::Tool {
                    name: "fetch_url".to_string(),
                    arguments: serde_json::json!({}),
                },
                Permission::Unrestricted,
                Some(&raised),
            )
            .is_err(),
            "and the door must then refuse it at every level, since a gate needs `read` exactly"
        );
    }

    /// A gate that reaches `execute_command` runs it confined, or not at all.
    ///
    /// `execute_command` resolves to `read` wherever a sandbox is usable, so a session at `read`
    /// can name it as a gate probe and get an arbitrary command on a timer. That is allowed, and it
    /// rests entirely on the pairing asserted here: the level that admits it as a probe is the same
    /// level that forces `Confinement::ReadOnly`, and where the sandbox is unavailable the tool
    /// resolves above `read` and the gate door refuses it instead. The registry a gate dispatches
    /// through is built at `read` precisely so the second door cannot be reached with
    /// `unrestricted`, the only level that would spawn unconfined.
    ///
    /// A gate's tool call goes all the way through: registry, dispatch, flatten, and the success
    /// flag the `succeeded` predicate reads.
    ///
    /// Every other tool-gate test drives a stub, so without this a mutation sweep could empty
    /// `flatten_tool_text` (every `changed` gate fires once and never again) or invert
    /// `!output.is_error` and stay green.
    #[tokio::test]
    async fn a_gate_tool_call_returns_the_tools_text_and_its_success() {
        use crate::schedule::GateTools;
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("watched.json");
        std::fs::write(&path, r#"{"chats": ["one"]}"#).expect("write");

        let toolset = GateToolset {
            mcp: None,
            core: crate::session::CoreMaterials {
                sandbox_enabled: false,
                sandbox_capability: crate::sandbox::SandboxCapability::Unavailable,
                builtin_filter: BuiltinToolFilter::from_config(None, Vec::new(), HashMap::new()),
                ..crate::session::CoreMaterials::for_test()
            },
            resolution: std::sync::OnceLock::new(),
        };

        let outcome = toolset
            .call(
                "read_file",
                &serde_json::json!({ "path": path.to_string_lossy() }),
                std::time::Duration::from_secs(10),
                Some(directory.path()),
                None,
            )
            .await
            .expect("the tool is registered and the file is there");
        assert!(
            outcome.text.contains("chats"),
            "the probe carries what the tool returned, not an empty string: {:?}",
            outcome.text
        );
        assert!(
            outcome.succeeded,
            "a tool that did not error reads as success"
        );

        let missing = toolset
            .call(
                "read_file",
                &serde_json::json!({ "path": directory.path().join("absent").to_string_lossy() }),
                std::time::Duration::from_secs(10),
                Some(directory.path()),
                None,
            )
            .await;
        // Whether an absent path is an `Err` or an `is_error` result is the tool's business; what
        // matters is that it is not reported as a success.
        assert!(
            missing.as_ref().map(|outcome| outcome.succeeded) != Ok(true),
            "and one that failed does not: {missing:?}"
        );
    }

    /// A dispatcher with no MCP manager cannot have anything mid-handshake.
    #[test]
    fn a_gate_without_mcp_never_claims_a_server_is_still_connecting() {
        use crate::schedule::GateTools;
        let toolset = GateToolset {
            mcp: None,
            core: crate::session::CoreMaterials {
                sandbox_enabled: false,
                sandbox_capability: crate::sandbox::SandboxCapability::Unavailable,
                builtin_filter: BuiltinToolFilter::from_config(None, Vec::new(), HashMap::new()),
                ..crate::session::CoreMaterials::for_test()
            },
            resolution: std::sync::OnceLock::new(),
        };
        assert!(
            !toolset.is_still_connecting("mcp__anything__at_all"),
            "otherwise every unresolvable tool gate is reported as merely starting up, forever"
        );
    }

    /// Both halves are asserted, because either alone is satisfiable by a broken build: a probe
    /// that is always refused is confined vacuously, and one that always runs is not confined at
    /// all.
    #[test]
    fn a_gate_may_admit_execute_command_only_where_it_will_be_sandboxed() {
        let toolset =
            |sandbox_enabled: bool, capability: crate::sandbox::SandboxCapability| GateToolset {
                mcp: None,
                core: crate::session::CoreMaterials {
                    sandbox_enabled,
                    sandbox_capability: capability,
                    ..crate::session::CoreMaterials::for_test()
                },
                resolution: std::sync::OnceLock::new(),
            };
        use crate::schedule::GateTools;
        let probe = crate::schedule::GateProbe::Tool {
            name: "execute_command".to_string(),
            arguments: serde_json::json!({ "command": "true" }),
        };

        // Whatever this platform's usable backend is; the rule under test is "anything but
        // `Unavailable`", not any particular one.
        #[cfg(target_os = "linux")]
        let usable = crate::sandbox::SandboxCapability::Landlock { abi_version: 5 };
        #[cfg(target_os = "macos")]
        let usable = crate::sandbox::SandboxCapability::SandboxExec;
        #[cfg(target_os = "windows")]
        let usable = crate::sandbox::SandboxCapability::LowIntegrity;

        let sandboxed = toolset(true, usable);
        assert_eq!(
            sandboxed.resolve("execute_command"),
            Some(Permission::Read),
            "with a usable sandbox the shell is a read-level tool"
        );
        assert!(
            crate::schedule::gate_probe_is_authorized(&probe, Permission::Read, Some(&sandboxed))
                .is_ok(),
            "so a gate at `read` may call it, the door this test exists to bound"
        );
        assert_eq!(
            GateToolset::dispatch_permission().get(),
            Permission::Read,
            "and it is dispatched at `read`, which is what makes `Confinement::resolve` confine it: \
             `unrestricted` is the only level that spawns a bare shell"
        );

        let unconfined = toolset(false, crate::sandbox::SandboxCapability::Unavailable);
        assert_eq!(
            unconfined.resolve("execute_command"),
            Some(Permission::Unrestricted),
            "with no sandbox it is not a read-level tool"
        );
        assert!(
            matches!(
                crate::schedule::gate_probe_is_authorized(
                    &probe,
                    Permission::Unrestricted,
                    Some(&unconfined)
                ),
                Err(crate::schedule::GateRefusal::ToolNotReadOnly(_))
            ),
            "so the gate door refuses it at every level, rather than running a command with no \
             boundary on a timer"
        );
    }
}
