//! What a session is made of: the materials every agent and tool registry of the session is built
//! from, and the live cells they all read through the same handles.

use std::sync::Arc;

use uuid::Uuid;

use crate::store::Store;

/// What the core tools are built from: the web client, the sandbox, and the `[tools]` filter.
/// Every registry gets these, including a gate probe's, which has no session behind it.
#[derive(Clone)]
pub(crate) struct CoreMaterials {
    pub(crate) web_client: crate::config::WebClientConfig,
    pub(crate) sandbox_enabled: bool,
    pub(crate) sandbox_capability: crate::sandbox::SandboxCapability,
    pub(crate) sandbox_backend: crate::config::SandboxBackend,
    pub(crate) backend_probe: crate::sandbox::BackendProbe,
    /// The `[tools]` filter; sub-agents inherit it.
    pub(crate) builtin_filter: crate::config::BuiltinToolFilter,
    /// The per-path write locks every registry built from these materials takes.
    pub(crate) write_locks: crate::workspace::WriteLocks,
}

impl CoreMaterials {
    pub(crate) fn from_config(
        config: &crate::config::ResolvedConfig,
        builtin_filter: crate::config::BuiltinToolFilter,
        sandbox: &crate::sandbox::SandboxResolution,
    ) -> Self {
        Self {
            web_client: config.web_client.clone(),
            sandbox_enabled: config.sandbox,
            sandbox_capability: crate::sandbox::capability_from_probe(&sandbox.probe),
            sandbox_backend: sandbox.backend,
            backend_probe: sandbox.probe.clone(),
            builtin_filter,
            write_locks: crate::workspace::WriteLocks::default(),
        }
    }

    /// Sandbox off and unavailable, default web client, no filter.
    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        Self {
            web_client: crate::config::WebClientConfig::default(),
            sandbox_enabled: false,
            sandbox_capability: crate::sandbox::SandboxCapability::Unavailable,
            sandbox_backend: crate::config::SandboxBackend::Landlock,
            backend_probe: crate::sandbox::BackendProbe::Missing {
                reason: "test fixture".to_string(),
            },
            builtin_filter: crate::config::BuiltinToolFilter::default(),
            write_locks: crate::workspace::WriteLocks::default(),
        }
    }
}

/// What a session is made of and never changes while it runs. Every agent built for the session,
/// and every sub-agent spawned from it, shares these by handle; a sub-agent narrows what it may
/// reach through its own registry and permission, never through a different set of materials.
#[derive(Clone)]
pub(crate) struct SessionMaterials {
    pub(crate) core: CoreMaterials,
    pub(crate) skills: Arc<crate::skills::SkillCache>,
    /// Whether the root agent gets `skill_write` / `skill_delete`. Never a sub-agent.
    pub(crate) skills_agent_managed: bool,
    pub(crate) memories: Arc<crate::store::memory::MemoryStore>,
    pub(crate) store: Store,
    /// The MCP manager, held weakly: it holds every attached registry, which holds the tools that
    /// hold this, and a strong reference would keep a closed session alive until exit.
    pub(crate) mcp_manager: Option<std::sync::Weak<crate::mcp::McpClientManager>>,
    /// The session's counters. Sub-agents share them, so `/status` shows the whole cost.
    pub(crate) session_stats: Arc<crate::stats::SessionStats>,
    pub(crate) schedule: crate::config::ResolvedScheduleConfig,
    pub(crate) background: crate::config::ResolvedBackgroundConfig,
    pub(crate) subagents: crate::config::ResolvedSubagentsConfig,
    pub(crate) subagent_max_depth: usize,
}

/// The id of the session the cells belong to, shared by handle between the agent, every tool that
/// scopes its work to the session, and the host. `None` until the first turn creates the row; a
/// host that knows the row seeds it, and `/fork` moves it to the copy.
///
/// A `std` lock rather than a `tokio` one: a read copies one `Uuid` and is never held across an
/// `.await`, so no reader needs an executor to take it. Poisoning is recovered through
/// [`crate::sync`] like every other `std` lock in the tree.
#[derive(Clone, Default)]
pub(crate) struct SharedSessionId(Arc<std::sync::RwLock<Option<Uuid>>>);

impl SharedSessionId {
    pub(crate) fn new(id: Option<Uuid>) -> Self {
        Self(Arc::new(std::sync::RwLock::new(id)))
    }

    pub(crate) fn get(&self) -> Option<Uuid> {
        *crate::sync::read(&self.0)
    }

    pub(crate) fn set(&self, id: Uuid) {
        *crate::sync::write(&self.0) = Some(id);
    }
}

/// The live state of one session: the cells that change as it runs, read by every tool and by the
/// agent through the same handles, so a change made through any one of them reaches all of them.
#[derive(Clone)]
pub(crate) struct SessionCells {
    pub(crate) permission: crate::permission::SharedPermission,
    pub(crate) cwd: crate::workspace::SharedCwd,
    pub(crate) roots: crate::workspace::SharedRoots,
    pub(crate) session_id: SharedSessionId,
    pub(crate) todo_list: crate::todo::SharedTodoList,
    /// What the session runs on, published so a switch reaches everything holding this.
    pub(crate) profile: crate::provider::PublishedProfile,
    /// Tokens in context after the last provider round; a frontend gauge holds the same cell.
    pub(crate) context_tokens: Arc<std::sync::atomic::AtomicU64>,
    /// Estimated fixed overhead (system prompt and tool schemas); `context_check` reads it.
    pub(crate) context_overhead: Arc<std::sync::atomic::AtomicU64>,
    /// Background tool calls in flight; inert unless `[background] enabled`.
    pub(crate) background_tasks: crate::background::BackgroundTasks,
    /// Where prompts, progress, live output and every other event of this session go.
    pub(crate) frontend: Arc<dyn crate::frontend::Frontend>,
    /// The lock on the session, when this process created or adopted it. One slot rather than a
    /// field on each holder, so `/fork` can replace it and a host can drop it after its last
    /// message rather than whenever the agent falls out of scope.
    pub(crate) session_lock: crate::store::SessionLockSlot,
    /// A compaction `context_compact` asked for, drained by the turn loop once the batch's results
    /// are in. Inert on a sub-agent, which is registered no `context_*` tool.
    pub(crate) pending_compaction: PendingCompaction,
}

impl SessionCells {
    /// The cells of a session that has not run yet: no session id, an empty todo list, fresh
    /// gauges, nothing in the background.
    pub(crate) fn new(
        permission: crate::permission::SharedPermission,
        cwd: crate::workspace::SharedCwd,
        roots: crate::workspace::SharedRoots,
        profile: crate::provider::PublishedProfile,
        frontend: Arc<dyn crate::frontend::Frontend>,
    ) -> Self {
        Self {
            permission,
            cwd,
            roots,
            session_id: SharedSessionId::default(),
            todo_list: crate::todo::SharedTodoList::default(),
            profile,
            context_tokens: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            context_overhead: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            background_tasks: crate::background::BackgroundTasks::default(),
            frontend,
            session_lock: crate::store::SessionLockSlot::default(),
            pending_compaction: PendingCompaction::default(),
        }
    }

    /// The same cells, bound to a session that already exists. Every host but the REPL's first
    /// turn knows the row before it builds the agent, and seeding it here is what lets a tool read
    /// the id before any turn has run.
    pub(crate) fn with_session(mut self, id: Uuid) -> Self {
        self.session_id = SharedSessionId::new(Some(id));
        self
    }

    /// The handles the built-in tools read, as one value.
    pub(crate) fn site(&self) -> ToolSite {
        ToolSite {
            permission: self.permission.clone(),
            cwd: self.cwd.clone(),
            roots: self.roots.clone(),
            session_id: self.session_id.clone(),
        }
    }
}

/// Where a tool call runs: the handles a built-in reads for the session it serves.
///
/// One value for the four cells the built-ins each carried a copy of, so a session-level fact
/// reaches every tool by being added here rather than to each struct and each constructor. A
/// session's site is its cells'; a scheduled job's gate has no session and builds a detached one.
#[derive(Clone)]
pub(crate) struct ToolSite {
    pub(crate) permission: crate::permission::SharedPermission,
    pub(crate) cwd: crate::workspace::SharedCwd,
    pub(crate) roots: crate::workspace::SharedRoots,
    pub(crate) session_id: SharedSessionId,
}

impl ToolSite {
    /// A site outside any session, for a gate probe: `permission` in `cwd`, with no roots and no
    /// session id.
    pub(crate) fn detached(
        permission: crate::permission::SharedPermission,
        cwd: crate::workspace::SharedCwd,
    ) -> Self {
        Self {
            permission,
            cwd,
            roots: crate::workspace::SharedRoots::default(),
            session_id: SharedSessionId::default(),
        }
    }
}

#[cfg(test)]
impl ToolSite {
    /// Unrestricted, in the current directory, with no roots and no session; each `with_*` replaces
    /// one handle.
    pub(crate) fn for_test() -> Self {
        Self::detached(
            crate::permission::SharedPermission::new(
                crate::permission::Permission::Unrestricted,
                crate::permission::EnabledPermissions::ALL,
            ),
            crate::workspace::SharedCwd::new(std::path::PathBuf::from(".")),
        )
    }

    pub(crate) fn with_permission(
        mut self,
        permission: crate::permission::SharedPermission,
    ) -> Self {
        self.permission = permission;
        self
    }

    pub(crate) fn with_cwd(mut self, cwd: crate::workspace::SharedCwd) -> Self {
        self.cwd = cwd;
        self
    }

    pub(crate) fn with_roots(mut self, roots: crate::workspace::SharedRoots) -> Self {
        self.roots = roots;
        self
    }

    pub(crate) fn with_session_id(mut self, session_id: SharedSessionId) -> Self {
        self.session_id = session_id;
        self
    }
}

#[cfg(test)]
impl SessionMaterials {
    /// Materials for a test: sandbox off and unavailable, both stores detached, no MCP, no
    /// sub-agents, every subsystem config at its default.
    pub(crate) fn for_test(store: Store) -> Self {
        Self {
            core: CoreMaterials::for_test(),
            skills: crate::skills::SkillCache::for_root(None),
            skills_agent_managed: false,
            memories: crate::store::memory::MemoryStore::detached(),
            store,
            mcp_manager: None,
            session_stats: Arc::new(crate::stats::SessionStats::default()),
            schedule: crate::config::ResolvedScheduleConfig::default(),
            background: crate::config::ResolvedBackgroundConfig::default(),
            subagents: crate::config::ResolvedSubagentsConfig::default(),
            subagent_max_depth: 0,
        }
    }
}

#[cfg(test)]
impl SessionCells {
    /// Fresh cells bound to a mock provider that answers nothing, for a test that needs a session
    /// but never asks the model anything.
    pub(crate) fn for_test(
        permission: crate::permission::SharedPermission,
        cwd: crate::workspace::SharedCwd,
        roots: crate::workspace::SharedRoots,
        frontend: Arc<dyn crate::frontend::Frontend>,
    ) -> Self {
        Self::new(
            permission,
            cwd,
            roots,
            crate::provider::PublishedProfile::unbound(),
            frontend,
        )
    }
}

/// Trigger auto-compaction once a turn's input tokens exceed this fraction of the configured
/// context window.
pub(crate) const AUTO_COMPACT_THRESHOLD_PERCENT: u64 = 80;

/// Per-turn configuration knobs for `Agent`. Constructed once by `main` from the
/// [`crate::config::ResolvedConfig`] and held immutably for the agent's lifetime; mid-session
/// permission cycling and tool loading are handled by shared state (see
/// [`crate::permission::SharedPermission`] and `ToolRegistry`) rather than by mutating fields here.
#[derive(Clone)]
pub(crate) struct AgentOptions {
    /// When true, assistant responses stream token-by-token via `Provider::stream`; otherwise the
    /// agent uses the blocking `Provider::complete`.
    pub(crate) streaming: bool,
    /// Whether `execute_command` calls at `read` run inside the platform sandbox. Forced off when
    /// no sandbox backend is available.
    pub(crate) sandboxed_shell: bool,
    /// Cap on messages sent to the provider, re-applied on every round of a turn rather than once
    /// at its start.
    ///
    /// A maximum, not a target: `truncate_messages_for_context` cuts *forward* to the first
    /// message that neither splits a `tool_use` → `tool_result` chain nor starts the window on
    /// a role the provider rejects, so a window ending inside a long tool loop can hold fewer
    /// messages than asked for. It reaches backward only when the whole tail is one unbroken
    /// chain, where exceeding the cap beats sending something that will be refused.
    ///
    /// `None` is unlimited, but nothing reaches it from `config.toml`: an absent
    /// `[session].context_messages` resolves to a default, so removing the key lowers the cap
    /// rather than lifting it. Only a directly-constructed `AgentOptions` (tests, and a sub-agent
    /// inheriting one) can be `None`.
    pub(crate) context_messages: Option<usize>,
    /// When true, the agent auto-compacts the conversation once a turn's input tokens cross
    /// [`AUTO_COMPACT_THRESHOLD_PERCENT`] of the session's context window. Requires a window above
    /// zero.
    pub(crate) auto_compact: bool,
    /// When true, a compaction is preceded by a *checkpoint turn*: the agent itself, holding its
    /// real system prompt and memory index, decides what survives and writes durable notes for
    /// anything that must outlive the window (see `Agent::run_checkpoint_turn`).
    ///
    /// Off means every compaction uses `Agent::summarize_via_provider`, which is also the
    /// unconditional path for [`crate::session::CompactOrigin::Emergency`] regardless of
    /// this flag.
    pub(crate) compact_checkpoint: bool,
    /// User-authored instructions, surfaced in the system prompt and to sub-agents. Per-run
    /// `--instructions` overrides the config-file value.
    pub(crate) user_instructions: Option<String>,
    /// Max time to wait for still-`Pending` MCP servers to settle before the readiness gate
    /// decides. Which servers actually gate is per-server (`[[mcp.servers]].required`), so there
    /// is no strictness flag here.
    pub(crate) mcp_grace: std::time::Duration,
    /// When `Some`, `run_turn` uses this string verbatim instead of invoking
    /// [`crate::prompt::build_system_prompt`]. Sub-agents set this to their stripped-down prompt
    /// from `build_subagent_system_prompt`. The override is static; it does not see per-turn todo
    /// updates or permission changes, which is fine for one-shot sub-agents whose tool list and
    /// permission level are fixed at spawn time.
    pub(crate) system_prompt_override: Option<String>,
    /// How to resolve a scheduled gate's tool when telling the model which of its jobs are held.
    ///
    /// `None` leaves every tool gate reported as unresolvable, which is honest for a process that
    /// genuinely has no dispatcher, and is what a sub-agent gets: it does not own the parent's
    /// jobs and has no `[Scheduled]` section to fill.
    pub(crate) gate_tools: Option<Arc<dyn crate::schedule::GateTools>>,
}

/// A compaction the agent asked for, parked until the loop reaches a point that can run it.
///
/// Tools hold no `&mut Conversation` (the agent loop owns it for the duration of the turn), so
/// `context_compact` cannot compact where it stands. It records the request here and the tool loop
/// drains it once the batch's results are in, which is what lets the turn carry on against the
/// summary instead of ending at the request. A turn that fails before reaching that point leaves
/// the request behind, and the drain after the loop takes it so it cannot fire against a later one.
pub(crate) type PendingCompaction = Arc<std::sync::Mutex<Option<CompactRequest>>>;
/// Token budget for the verbatim tail a compaction keeps: about a tenth of the window, floored and
/// capped so a small window still keeps something usable and a large one doesn't carry half the
/// conversation past the boundary.
///
/// Shared with `context_check`, which reports it so the model can tell whether its current thread
/// of work would survive a compaction intact.
pub(crate) fn compaction_tail_budget(context_window: u64) -> u64 {
    (context_window / 10).clamp(4_000, 16_000)
}
/// What set a compaction going. Selects the summarization strategy, so it is not merely
/// diagnostic.
///
/// Every origin but [`Self::Emergency`] can afford the checkpoint turn. `Emergency` cannot: it runs
/// *after* the provider rejected the request for exceeding the window, and a checkpoint turn sends
/// the same conversation again, so it would be refused for the same reason. That path needs a call
/// that is deliberately smaller than the one that just failed, which is exactly what
/// `Agent::summarize_via_provider` is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompactOrigin {
    /// The previous turn's reported usage crossed the threshold.
    Reactive,
    /// This turn's projected request would cross it.
    Proactive,
    /// A human ran `/compact`.
    Manual,
    /// The agent asked, via `context_compact`.
    Requested,
    /// The provider refused the request as too large.
    Emergency,
}
/// One compaction, and the instructions shaping it.
#[derive(Debug, Clone)]
pub(crate) struct CompactRequest {
    pub(crate) origin: CompactOrigin,
    /// Free-text guidance on what to preserve or drop, from `/compact <instructions>` or
    /// `context_compact`. Reaches the checkpoint turn and the fallback summarizer alike, so the
    /// channel works whichever strategy runs.
    pub(crate) instructions: Option<String>,
    /// Whether to keep the recent turns verbatim after the summary. `None` means "unspecified",
    /// which resolves to `true`; `context_replace` may override it with a better-informed answer,
    /// since only the checkpoint turn knows whether the summary already covers them.
    pub(crate) keep_recent: Option<bool>,
    /// The prompt the compaction serves, when a turn asked for it; a host-driven `/compact` has
    /// none and its requests mint one.
    pub(crate) prompt_id: Option<Uuid>,
}
impl CompactRequest {
    pub(crate) fn new(origin: CompactOrigin) -> Self {
        Self {
            origin,
            instructions: None,
            keep_recent: None,
            prompt_id: None,
        }
    }

    /// Bill the compaction's requests to the turn that asked for it.
    pub(crate) fn attributed_to(mut self, prompt_id: Option<Uuid>) -> Self {
        self.prompt_id = prompt_id;
        self
    }
}
impl AgentOptions {
    /// The options the root agent runs with under this configuration. `sandboxed_shell` is the
    /// caller's because it depends on a probe the configuration only records.
    pub(crate) fn from_config(
        config: &crate::config::ResolvedConfig,
        sandboxed_shell: bool,
        gate_tools: Option<Arc<dyn crate::schedule::GateTools>>,
        user_instructions: Option<String>,
    ) -> Self {
        Self {
            streaming: config.streaming,
            sandboxed_shell,
            gate_tools,
            context_messages: config.context_messages,
            auto_compact: config.auto_compact,
            compact_checkpoint: config.compact_checkpoint,
            user_instructions,
            mcp_grace: config.mcp_grace,
            system_prompt_override: None,
        }
    }

    /// Everything off, for a registry or agent a test builds without a configuration.
    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        Self {
            streaming: false,
            sandboxed_shell: false,
            gate_tools: None,
            context_messages: None,
            auto_compact: false,
            compact_checkpoint: false,
            user_instructions: None,
            mcp_grace: std::time::Duration::ZERO,
            system_prompt_override: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cli::session::delete_sessions,
        conversation::{self, format_session_as_markdown},
        host::{ForkHandoff, fork_and_lock},
        store::export::{
            SESSION_EXPORT_FORMAT_VERSION, SessionExport, build_session_export,
            parents_first_order, plan_import,
        },
    };

    fn user_msg(text: &str) -> crate::conversation::Message {
        crate::conversation::Message::user(text)
    }

    fn assistant_text(text: &str) -> crate::conversation::Message {
        crate::conversation::Message::assistant_text(text)
    }

    #[test]
    fn parents_first_order_rejects_cycle() {
        let nodes = vec![
            ("a".to_string(), Some("b".to_string())),
            ("b".to_string(), Some("a".to_string())),
        ];
        assert!(parents_first_order(&nodes).is_err());
    }

    #[test]
    fn plan_import_rejects_unknown_format_version() {
        let export = SessionExport {
            format_version: SESSION_EXPORT_FORMAT_VERSION + 1,
            meka_version: "test".into(),
            exported_at: "now".into(),
            root_session_id: "r".into(),
            sessions: Vec::new(),
            blobs: Vec::new(),
        };
        assert!(plan_import(export, None, Some(crate::permission::Permission::Read)).is_err());
    }

    #[tokio::test]
    async fn session_export_import_round_trip() {
        use crate::{
            conversation::{ContentBlock, Event, Message, Role, ToolResultContent},
            image::ImageSource,
        };

        let manager = Store::for_test().await;

        // Root session with a representative mix of events: plain text, an input image, a
        // reasoning block, a tool_use/tool_result pair, and a compaction boundary.
        let root = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("root");
        let image = ImageSource::Base64 {
            media_type: "image/png".to_string(),
            data: "aGk=".to_string(),
        };
        let root_events = vec![
            Event::Append(Message::user("hello")),
            Event::Append(Message::user_with_images("look", vec![image])),
            Event::Append(Message {
                role: Role::Assistant,
                // Both opaque halves of a Responses reasoning block. Neither is readable and
                // neither is reconstructible, so an export that dropped them would leave the
                // imported session unable to replay its own reasoning, silently.
                content: vec![
                    ContentBlock::Thinking {
                        thinking: "weighing it up".to_string(),
                        opaque: Some(crate::conversation::OpaqueReasoning::Sealed {
                            encrypted_content: "OPAQUE".to_string(),
                            id: Some("rs_1".to_string()),
                        }),
                    },
                    ContentBlock::ToolUse {
                        id: "u1".to_string(),
                        name: "read".to_string(),
                        input: serde_json::json!({"path": "/x"}),
                    },
                ],
            }),
            Event::Append(Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "u1".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "ok".to_string(),
                    }],
                    is_error: false,
                }],
            }),
            Event::CompactBoundary {
                summary: Message::user("[summary]"),
                replaced_count: 2,
                loaded_tools_snapshot: Default::default(),
            },
        ];
        for event in &root_events {
            manager
                .save_event(root, event)
                .await
                .expect("save root event");
        }
        manager
            .save_scratchpad_entry(root, "tool_1_output", "big output")
            .await
            .expect("tool output");
        let stats = crate::stats::SessionStatsSnapshot {
            turns: 3,
            input_tokens: 1000,
            ..Default::default()
        };
        manager
            .save_session_stats(root, &stats)
            .await
            .expect("stats");

        // A sub-agent child of the root, with the spawn terms `agent_followup` reconstructs from.
        // An archive that drops these imports a sub-agent nobody can resume.
        let child_spec = r#"{"permission":"read","enabled_permissions":["read"],"denied_servers":["mekabridge"],"denied_tools":[],"memory":"none","inherited_scratchpad":[],"remaining_depth":0,"absolute_depth":1}"#;
        let child = manager
            .create_child_session(
                root,
                None,
                Some(child_spec.to_string()),
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;
        for event in [
            Event::Append(Message::user("sub task")),
            Event::Append(Message::assistant_text("sub done")),
        ] {
            manager.save_event(child, &event).await.expect("save child");
        }

        // Export -> JSON -> back.
        let export = build_session_export(&manager, root).await.expect("export");
        assert_eq!(export.sessions.len(), 2, "root + child");
        assert_eq!(export.sessions[0].id, root.to_string(), "root first");
        let json = serde_json::to_string_pretty(&export).expect("serialize");
        assert!(
            !json.contains("token_id"),
            "the fingerprint must not be exported"
        );
        let reparsed: SessionExport = serde_json::from_str(&json).expect("deserialize");

        // Import under fresh IDs.
        let crate::store::export::ImportPlan {
            records,
            blobs,
            root_new_id,
        } = plan_import(reparsed, None, Some(crate::permission::Permission::Read)).expect("plan");
        assert_ne!(root_new_id, root, "import mints a new id");
        manager
            .import_sessions(records, blobs)
            .await
            .expect("import");

        // The tree came back: root + child, with the child's parent rewired to the new root.
        let tree = manager.load_session_tree(root_new_id).await.expect("tree");
        assert_eq!(tree.len(), 2);
        let child_new = tree
            .iter()
            .find(|meta| meta.id != root_new_id)
            .expect("child present");
        assert_eq!(child_new.parent_id, Some(root_new_id));
        // The spawn terms survived export -> JSON -> import. This is also the column-alignment
        // check on `import_sessions`' 18-parameter INSERT: reading the spec back verbatim off a
        // different column would surface here as a mismatch rather than silently.
        assert_eq!(
            manager
                .load_subagent_spec(child_new.id)
                .await
                .expect("load spec"),
            Some(child_spec.to_string()),
        );
        assert_eq!(
            manager
                .load_subagent_spec(root_new_id)
                .await
                .expect("load root spec"),
            None,
            "a root session has no spawn terms",
        );
        assert_eq!(
            child_new.cwd, None,
            "and neighboring columns are undisturbed"
        );
        assert_eq!(
            child_new.permission,
            Some(crate::permission::Permission::Read),
            "a worker's row records the level it was spawned at"
        );

        // The event log round-trips byte-for-byte against the untouched original.
        let imported = manager
            .load_events(root_new_id)
            .await
            .expect("load imported");
        let original = manager.load_events(root).await.expect("load original");
        assert_eq!(
            serde_json::to_string(&imported).unwrap(),
            serde_json::to_string(&original).unwrap(),
        );
        assert!(
            imported.iter().any(|event| match event {
                Event::Append(message) => message
                    .content
                    .iter()
                    .any(|block| matches!(block, ContentBlock::Image { .. })),
                _ => false,
            }),
            "the input image must survive the round trip",
        );
        assert!(
            imported.iter().any(|event| match event {
                Event::Append(message) => message.content.iter().any(|block| matches!(
                    block,
                    ContentBlock::Thinking {
                        opaque: Some(crate::conversation::OpaqueReasoning::Sealed { encrypted_content, id }),
                        ..
                    } if encrypted_content == "OPAQUE" && id.as_deref() == Some("rs_1")
                )),
                _ => false,
            }),
            "and so must the opaque reasoning, or the imported session cannot replay it",
        );

        // Child events, stats, and tool_outputs are preserved.
        assert_eq!(
            manager
                .load_events(child_new.id)
                .await
                .expect("load child events")
                .len(),
            2,
        );
        let imported_stats = manager
            .load_session_stats(root_new_id)
            .await
            .expect("load stats");
        assert_eq!(imported_stats.turns, 3);
        assert_eq!(imported_stats.input_tokens, 1000);
        assert_eq!(
            manager
                .load_all_scratchpad_entries(root_new_id)
                .await
                .expect("load outputs"),
            vec![("tool_1_output".to_string(), "big output".to_string())],
        );
    }

    /// One id being refused must not cost the user the rest of the list.
    ///
    /// Each id on the command line is a separate request, and a session another meka has open is a
    /// refusal about that one. Returning at the first refusal would skip every id after it and
    /// swallow the count of what had been deleted on the way.
    #[tokio::test]
    async fn a_refused_session_does_not_abandon_the_rest_of_the_list() {
        let manager = Store::for_test().await;
        let held = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let after = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let _lock = manager
            .lock_session(held)
            .expect("another process holds it");

        let outcome = delete_sessions(&manager, &[held, after], false, None).await;

        assert!(
            outcome.is_err(),
            "a refusal the user named has to reach the exit code"
        );
        assert!(
            manager.session_exists(held).await.expect("exists"),
            "the conversation somebody is having survives"
        );
        assert!(
            !manager.session_exists(after).await.expect("exists"),
            "and the id listed after it is still deleted rather than skipped"
        );
    }

    /// `/fork` must own the copy's lock before the REPL lets go of the one it is holding. That
    /// ordering is structural: [`fork_and_lock`] is handed no lock, so it has no way to release
    /// the caller's. This pins the pair of facts that make the structure sound: the returned lock
    /// is genuinely held on the copy, and the source's lock is untouched.
    #[tokio::test]
    async fn fork_and_lock_holds_both_locks_at_the_handoff() {
        let manager = Store::for_test().await;
        let source = manager
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create");
        let source_lock = manager.lock_session(source).expect("lock source");

        let handoff = fork_and_lock(&manager, source).await.expect("fork");
        let ForkHandoff::Switched { id, lock } = handoff else {
            panic!("expected a switch");
        };

        assert!(
            manager.lock_session(id).is_err(),
            "the returned lock must actually be held on the copy"
        );
        assert!(
            manager.lock_session(source).is_err(),
            "and the source's lock must still be held: releasing it first is the bug"
        );

        // Only once the caller drops the old guard does the source become available again.
        drop(source_lock);
        manager.lock_session(source).expect("source is free again");
        drop(lock);
    }

    #[tokio::test]
    async fn fork_and_lock_reports_a_missing_source() {
        let manager = Store::for_test().await;
        assert!(matches!(
            fork_and_lock(&manager, uuid::Uuid::new_v4())
                .await
                .expect("fork"),
            ForkHandoff::SourceGone,
        ));
    }

    /// Without it a multi-root session comes back from an export as single-root: the column exists,
    /// but no export/import struct carries it.
    #[tokio::test]
    async fn session_export_preserves_additional_roots() {
        use std::path::PathBuf;

        let manager = Store::for_test().await;
        let root = manager
            .create_session(
                Some(PathBuf::from("/work/main")),
                "test-profile".to_string(),
            )
            .await
            .expect("root");
        let roots = vec![PathBuf::from("/work/shared"), PathBuf::from("/work/docs")];
        manager
            .update_session(root, crate::store::SessionPatch {
                roots: Some(roots.clone()),
                ..Default::default()
            })
            .await
            .expect("roots");

        let export = build_session_export(&manager, root).await.expect("export");
        let json = serde_json::to_string(&export).expect("serialize");
        let reparsed: SessionExport = serde_json::from_str(&json).expect("deserialize");
        let crate::store::export::ImportPlan {
            records,
            blobs,
            root_new_id: new_id,
        } = plan_import(reparsed, None, Some(crate::permission::Permission::Read)).expect("plan");
        manager
            .import_sessions(records, blobs)
            .await
            .expect("import");

        assert_eq!(
            manager
                .session_info(new_id)
                .await
                .expect("info")
                .expect("row")
                .additional_roots,
            roots,
        );
    }

    /// An export written before `additional_roots` existed must still import. This is why the field
    /// is `#[serde(default)]` instead of a `format_version` bump, which `plan_import` would reject.
    #[test]
    fn plan_import_accepts_an_export_without_additional_roots() {
        let json = serde_json::json!({
            "format_version": SESSION_EXPORT_FORMAT_VERSION,
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
                "stats": crate::stats::SessionStatsSnapshot::default(),
                "events": [],
                "tool_outputs": {},
            }],
        });
        let export: SessionExport = serde_json::from_value(json).expect("deserialize");
        let crate::store::export::ImportPlan {
            records,
            blobs: _,
            root_new_id: _,
        } = plan_import(
            export,
            Some("work"),
            Some(crate::permission::Permission::Read),
        )
        .expect("plan");
        assert!(records[0].additional_roots.is_empty());
        assert_eq!(
            records[0].profile, "work",
            "an archive naming no profile adopts this installation's default"
        );
    }

    /// An archive naming no profile, imported where nothing can supply one, is refused: a row with
    /// an empty profile cannot run, and its existence would force every reader to know about a
    /// state nothing else produces.
    #[test]
    fn an_archive_with_no_profile_is_refused_when_nothing_can_supply_one() {
        let json = serde_json::json!({
            "format_version": SESSION_EXPORT_FORMAT_VERSION,
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
                "stats": crate::stats::SessionStatsSnapshot::default(),
                "events": [],
                "tool_outputs": {},
            }],
        });
        let export: SessionExport = serde_json::from_value(json).expect("deserialize");
        let Err(error) = plan_import(export, None, Some(crate::permission::Permission::Read))
        else {
            panic!("no default and no recorded profile must refuse the import");
        };
        assert!(
            error.to_string().contains("--profile"),
            "the refusal must name what supplies one: {error}"
        );
    }

    /// An archive cannot choose where a session's turns are sent.
    ///
    /// The credential comes from whichever configured profile the row names, so honoring an
    /// archive-supplied endpoint would post that profile's stored key wherever the archive said,
    /// and `POST /v1/sessions/import` takes its archive from a request body behind nothing but a
    /// `sessions:w` token. Refusing such an endpoint would cost a trusted/untrusted split across
    /// the two import doors.
    ///
    /// A session records a profile and nothing else, so the vector is closed by construction
    /// rather than by a refusal that has to be remembered. This pins that: an archive *naming* an
    /// endpoint key imports cleanly and takes the endpoint from its profile, and no import door
    /// needs to know which caller it is serving.
    #[test]
    fn an_archive_cannot_pin_a_sessions_endpoint() {
        let archive = |base_url: serde_json::Value| {
            serde_json::json!({
                "format_version": SESSION_EXPORT_FORMAT_VERSION,
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
                    "base_url_override": base_url,
                    "stats": crate::stats::SessionStatsSnapshot::default(),
                    "events": [],
                    "tool_outputs": {},
                }],
            })
        };

        // An archive written by a meka that still had the field. The key is not modeled, so serde
        // ignores it rather than refusing the archive: an import that failed here would strand a
        // backup the user took a fortnight ago.
        let hostile: SessionExport =
            serde_json::from_value(archive(serde_json::json!("https://elsewhere.invalid/v1")))
                .expect("an archive naming the retired key still deserializes");
        let crate::store::export::ImportPlan {
            records,
            blobs: _,
            root_new_id: _,
        } = plan_import(hostile, None, Some(crate::permission::Permission::Read)).expect("plan");
        assert_eq!(
            records[0].profile, "work",
            "the session runs on its profile, and the endpoint comes from that profile alone"
        );

        let plain: SessionExport =
            serde_json::from_value(archive(serde_json::Value::Null)).expect("deserialize");
        let crate::store::export::ImportPlan {
            records,
            blobs: _,
            root_new_id: _,
        } = plan_import(plain, None, Some(crate::permission::Permission::Read)).expect("plan");
        assert_eq!(records[0].profile, "work");
    }

    /// Retention GC deletes by `updated_at` when `[session].retention_days` is set, so an import
    /// that restored the export's value would be undone by the next launch.
    #[tokio::test]
    async fn import_survives_retention_gc() {
        let manager = Store::for_test().await;
        let stale = (chrono::Utc::now() - chrono::TimeDelta::days(100)).to_rfc3339();
        let records = vec![crate::store::ImportSessionRecord {
            new_id: uuid::Uuid::new_v4(),
            new_parent_id: None,
            created_at: stale.clone(),
            cwd: None,
            permission: crate::permission::Permission::Read,
            approvals: false,
            capabilities_json: None,
            additional_roots: Vec::new(),
            subagent_spec_json: None,
            profile: "test-profile".to_string(),
            stats: crate::stats::SessionStatsSnapshot::default(),
            events: Vec::new(),
            tool_outputs: Vec::new(),
        }];
        let imported_id = records[0].new_id;
        manager
            .import_sessions(records, Vec::new())
            .await
            .expect("import");

        assert_eq!(
            manager
                .delete_expired_sessions(std::time::Duration::from_secs(90 * 86_400))
                .await
                .expect("retention sweep")
                .deleted,
            0,
            "a freshly imported archive must not be swept on the next launch"
        );
        assert!(manager.session_exists(imported_id).await.expect("exists"));

        // `created_at` still carries the original for provenance.
        assert_eq!(
            manager
                .session_info(imported_id)
                .await
                .expect("info")
                .expect("row")
                .created_at,
            stale,
        );
    }

    #[test]
    fn full_export_includes_pre_compaction_turns() {
        // A compacted session: the early turns are hidden from the model behind a CompactBoundary,
        // but `meka session export` must still render them. Build the same event log compaction
        // produces and assert the export contains both the summarized turns and a boundary marker.
        let mut log = conversation::Conversation::new();
        log.append(user_msg("first question"));
        log.append(assistant_text("first answer"));
        log.append(user_msg("second question"));
        log.append(assistant_text("second answer"));
        log.replace_for_compaction(
            user_msg("[Conversation summary from session compaction]\n\nYou discussed things."),
            vec![assistant_text("kept tail answer")],
            std::collections::HashSet::new(),
        );

        let markdown = format_session_as_markdown(
            uuid::Uuid::nil(),
            log.events(),
            &std::collections::HashMap::new(),
        );

        // Pre-compaction turns survive in the export even though the model no longer sees them.
        assert!(
            markdown.contains("first question") && markdown.contains("second answer"),
            "full export must include pre-compaction turns:\n{markdown}"
        );
        // The boundary is marked, and its summary is available (collapsed).
        assert!(
            markdown.contains("Session compaction") && markdown.contains("You discussed things."),
            "full export must mark the compaction boundary:\n{markdown}"
        );
        // The retained tail (re-appended after the boundary) is present.
        assert!(
            markdown.contains("kept tail answer"),
            "full export must include the retained tail:\n{markdown}"
        );
    }
}
