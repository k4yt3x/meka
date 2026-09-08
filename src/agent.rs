//! Per-turn agent loop: streams provider output, dispatches tool calls, and persists the resulting
//! messages to the session store. Also handles mid-conversation auto-compaction when the
//! input-token budget is exceeded.

use std::{collections::HashMap, sync::Arc};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    conversation::{ContentBlock, Conversation, HARNESS_NOTE, Message, Role, ToolResultContent},
    error::{MekaError, Result},
    frontend::{FrontendEvent, PermissionOutcome, PermissionRequest},
    image::ImageSource,
    prompt,
    provider::{
        CompletionRequest, Provider, ResolvedProfile, StopReason, StreamEvent, ToolDefinition,
    },
    skills::SkillCache,
    store::{Store, memory::MemoryStore},
    tools::ToolRegistry,
};

mod compaction;
mod dispatch;
mod recovery;
mod turn;

use self::recovery::*;
pub(crate) use self::{compaction::*, turn::*};
use crate::session::{AUTO_COMPACT_THRESHOLD_PERCENT, AgentOptions, SessionCells};

/// Driver for a single conversation. One [`Agent`] handles one or more sequential turns against a
/// single provider, with a shared tool registry, shared permission state, and a persistent SQLite
/// session. A turn fans out tool calls (in parallel via `join_all`) and persists every assistant
/// and tool-result message to the session store.
///
/// `Agent` is held across turns *and* across providers: [`Self::set_provider`] moves a live one, so
/// a switch keeps the conversation, the session lock and the background-task registry. `/provider`,
/// `PATCH /v1/sessions/{id}` and ACP's `session/set_config_option` all land there.
pub(crate) struct Agent {
    /// The live state of the session this agent drives, shared by handle with every tool and with
    /// the host: permission, working directory, the published provider profile, the gauges, the
    /// frontend. The agent reads and writes through the cells rather than copies of them, so a
    /// change made anywhere reaches everywhere.
    cells: SessionCells,
    /// Whether this is the session's own agent or a worker spawned from it.
    role: AgentRole,
    /// The MCP manager, held weakly like the materials hold it; `None` where the process has none.
    /// A worker never consults it: see [`Self::mcp_manager`].
    mcp_manager: Option<std::sync::Weak<crate::mcp::McpClientManager>>,
    tool_registry: ToolRegistry,
    store: Store,
    options: AgentOptions,
    /// Last todo state pushed to the frontend, so a no-op `todo` call (e.g. a read with no
    /// arguments, or a rewrite that changes nothing) doesn't re-render the list. Private to this
    /// `Agent`; sub-agents route through `Agent::new` and so get their own.
    last_rendered_todo: tokio::sync::RwLock<Option<crate::todo::TodoState>>,
    /// The tool/skill/MCP picture the model was last shown, plus the conversation length at which
    /// it was shown. `None` means "tell it everything": a fresh agent, or a compaction that may
    /// have summarized the earlier rendering away. Same shape and reasoning as
    /// [`Self::last_rendered_todo`].
    ///
    /// The length matters because the render lives in a single user message, and
    /// [`truncate_messages_for_context`] sends only the most recent `context_messages` entries.
    /// Once that message falls out of the window the model can no longer see the catalog,
    /// the skill list, or any MCP server's instructions, so the picture has to be restated.
    /// Tracking where it landed means that costs a full render roughly once per window rather
    /// than once per turn.
    last_rendered_world: tokio::sync::RwLock<Option<(crate::prompt::WorldSnapshot, usize)>>,
    /// Shared skill cache. Re-checks the on-disk snapshot at the top of each turn and re-discovers
    /// when something changed, so adds / removes / frontmatter edits land without restart.
    /// Body-only edits take effect even sooner; `load_skill_body` re-reads from disk on every
    /// invocation regardless of cache state.
    skills: Arc<SkillCache>,
    /// Shared memory cache, same contract as `skills`: re-checked at the top of each turn so a
    /// memory the agent writes mid-turn appears in the very next turn's index.
    memories: Arc<MemoryStore>,
    /// How many times this session has been compacted, for the `[Context budget]` block.
    ///
    /// Held in memory and seeded lazily from the database ([`GENERATION_UNKNOWN`]) rather than
    /// queried per turn: the count only changes when this agent compacts, so one read per process
    /// is enough and a resumed session still reports its true generation. `context_check` goes to
    /// the database directly, since it is on demand and can afford to be authoritative.
    compaction_generation: std::sync::atomic::AtomicU64,
    /// Per-turn map of `tool_use_id` → scratchpad-name hint. Populated by MCP tool adapters so
    /// oversized-output persistence uses `mcp_<server>_<tool>` instead of the plain tool name.
    /// Cleared between turns by `persist_oversized_results`.
    scratchpad_hints: Arc<tokio::sync::RwLock<std::collections::HashMap<String, String>>>,
    /// Tools that have already been the subject of a [`Self::schema_advisory`]. Held rather than
    /// re-sent, because the advisory lives on in the conversation and a second copy teaches
    /// nothing while costing context on every later call.
    ///
    /// Cleared by [`Self::compact_session`], for the reason the read tracker beside it is: a
    /// summary may have taken the advisory with it, and the set is a claim about a conversation
    /// that no longer exists.
    schema_advisories_sent: Arc<std::sync::Mutex<std::collections::HashSet<String>>>,
    /// Images the request budget redacted from the round in flight, as the provider reported them,
    /// waiting to be recorded on the conversation the moment the round's answer lands. Cleared
    /// before each request, so a report from a compaction's own requests never lands on a view it
    /// does not address.
    pending_redactions: Arc<std::sync::Mutex<Vec<crate::image::RedactedImage>>>,
    /// Ceiling on this session's concurrent background tasks, from `[background] max_tasks`. Zero
    /// when the feature is off, which is also what refuses a call that somehow arrives anyway.
    background_max_tasks: usize,
    /// Counters surfaced by `/status`. Shared with the Claude providers, which increment the
    /// redaction-related fields when oversized request bodies trigger image-block redaction.
    session_stats: Arc<crate::stats::SessionStats>,
    /// Where this conversation's most recent provider request id waits for the next request to
    /// name it. Per-`Agent`, so a sub-agent reports its own last response rather than its
    /// spawner's. Read and written only by the Claude subscription provider, which is the one
    /// backend that puts it on the wire (`cc_prev_req`); every other backend leaves it empty.
    previous_request: crate::provider::PreviousRequestSlot,
    /// The message id twin of [`Self::previous_request`], read for
    /// `diagnostics.previous_message_id`.
    previous_message: crate::provider::PreviousMessageSlot,
    /// Conversation length at the time of the most recent request the provider *accepted*, or
    /// [`LAST_ACCEPTED_UNKNOWN`] before the first one. Everything appended past it is what a
    /// `MekaError::InvalidRequest` is allowed to blame: the failing request differs from the last
    /// good one by exactly those messages, which is how `run_turn` locates the offending content
    /// without parsing the provider's error path (Anthropic's `messages.34.content.0…`), a shape
    /// no other backend produces and none of them map cleanly back through context truncation.
    ///
    /// Carried across turns rather than reset per turn so a turn that failed *after* appending its
    /// user message leaves that message a suspect on the retry; a fresh-per-turn floor would put
    /// it out of reach and leave the session stuck. The cost is that it must be invalidated
    /// whenever the conversation is rewritten under the agent, which
    /// [`Self::reset_conversation_markers`] and `compact_session` are responsible for.
    ///
    /// Atomic rather than `&mut` because `run_turn` takes `&self`; never shared between agents
    /// (sub-agents construct their own through [`Self::new`]).
    last_accepted_len: std::sync::atomic::AtomicUsize,
}

/// Whether an agent is the session's own or a worker spawned from it, and everything that follows
/// from that in one place: whose prompt its requests bill to, whether it persists the shared
/// statistics onto its row, whether it consults the MCP manager, and whether it may detach work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AgentRole {
    Root,
    /// A worker answers its parent's prompt, so it carries the id its parent's turn handed the
    /// spawning call rather than minting one.
    Worker {
        inherited_prompt_id: Option<uuid::Uuid>,
    },
}

impl AgentRole {
    pub(crate) fn is_worker(self) -> bool {
        matches!(self, Self::Worker { .. })
    }

    pub(crate) fn is_root(self) -> bool {
        matches!(self, Self::Root)
    }

    fn inherited_prompt_id(self) -> Option<uuid::Uuid> {
        match self {
            Self::Root => None,
            Self::Worker {
                inherited_prompt_id,
            } => inherited_prompt_id,
        }
    }
}

impl Agent {
    pub(crate) fn new(
        materials: &crate::session::SessionMaterials,
        cells: SessionCells,
        tool_registry: ToolRegistry,
        options: AgentOptions,
        role: AgentRole,
    ) -> Self {
        // Off for a worker whatever the configuration says: a worker's session ends with the one
        // turn that spawned it, so a task outliving that turn would have no conversation left to
        // report into.
        let background_max_tasks = match role {
            AgentRole::Root if materials.background.enabled => materials.background.max_tasks,
            _ => 0,
        };
        Self {
            cells,
            role,
            mcp_manager: materials.mcp_manager.clone(),
            options,
            tool_registry,
            store: materials.store.clone(),
            last_rendered_todo: tokio::sync::RwLock::new(None),
            last_rendered_world: tokio::sync::RwLock::new(None),
            skills: materials.skills.clone(),
            memories: materials.memories.clone(),
            compaction_generation: std::sync::atomic::AtomicU64::new(GENERATION_UNKNOWN),
            scratchpad_hints: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            schema_advisories_sent: Arc::new(std::sync::Mutex::new(
                std::collections::HashSet::new(),
            )),
            pending_redactions: Arc::new(std::sync::Mutex::new(Vec::new())),
            background_max_tasks,
            session_stats: Arc::clone(&materials.session_stats),
            // Fresh per agent: it describes one conversation, and a worker's requests are not its
            // spawner's.
            previous_request: crate::provider::PreviousRequestSlot::default(),
            previous_message: crate::provider::PreviousMessageSlot::default(),
            last_accepted_len: std::sync::atomic::AtomicUsize::new(LAST_ACCEPTED_UNKNOWN),
        }
    }

    /// The cells this agent drives, for a host that reads them without going through a turn.
    pub(crate) fn cells(&self) -> &SessionCells {
        &self.cells
    }

    /// The session this agent is on, or `None` before its first turn has created one.
    pub(crate) fn session_id(&self) -> Option<Uuid> {
        self.cells.session_id.get()
    }

    /// The provider this agent talks to now: whatever the published profile says. Read per use
    /// rather than held, so a switch is one store into the cell and nothing to keep in step.
    pub(crate) fn provider(&self) -> Arc<dyn Provider> {
        self.cells.profile.current().provider
    }

    /// The MCP manager, for the readiness gate and the server instructions in the prompt. `None`
    /// for a worker: its tools were fixed at spawn and it has no `[MCP]` section to fill, so it
    /// neither waits for servers nor describes them.
    fn mcp_manager(&self) -> Option<Arc<crate::mcp::McpClientManager>> {
        if self.role.is_worker() {
            return None;
        }
        self.mcp_manager.as_ref()?.upgrade()
    }

    /// Move this agent onto another profile, mid-conversation.
    ///
    /// Every surface that offers the switch lands here: `/profile`, `PATCH /v1/sessions/{id}` and
    /// ACP's `session/set_config_option`. All three parts move together because all three describe
    /// one profile: a profile that disagreed with the provider beside it would make the row a lie,
    /// and a context window left behind would gauge the new model against the old one's size and
    /// compact at the wrong point (or never).
    pub(crate) fn set_provider(&self, resolved: ResolvedProfile) {
        self.cells.profile.store(&resolved);
    }

    /// The window this agent gauges against, and the one every collaborator watching the published
    /// cell sees.
    ///
    /// Read from the cell rather than mirrored into a field of its own: hand-written assignments
    /// enforcing what the cell already guarantees break the day a third holder is added and only
    /// two are remembered.
    pub(crate) fn context_window(&self) -> u64 {
        self.cells.profile.context_window()
    }

    /// Point the occupancy gauge at a cell a test can read, the way a host's frontend gauge does
    /// through [`crate::session::SessionCells`].
    #[cfg(test)]
    pub(crate) fn set_context_tokens_for_test(&mut self, cell: Arc<std::sync::atomic::AtomicU64>) {
        self.cells.context_tokens = cell;
    }

    /// Move the window without a whole profile switch, for the tests that drive the auto-compaction
    /// guards. Goes through the published cell rather than poking an atomic, so the profile a test
    /// leaves behind is one the agent could actually be in.
    #[cfg(test)]
    pub(crate) fn set_context_window_for_test(&self, window: u64) {
        let mut resolved = self.cells.profile.current();
        resolved.context_window = window;
        self.set_provider(resolved);
    }

    /// The occupancy above which a turn compacts, or `None` when auto-compaction cannot apply.
    ///
    /// `None` for auto-compaction switched off, and for a zero window, which is not a small window
    /// but "unknown": a threshold of zero would compact every turn including the first.
    ///
    /// One function for the three sites that need it (the reactive check after a turn, the
    /// proactive projection before one, and the overflow-recovery guard), so the formula and its
    /// two-part guard cannot drift between them.
    pub(crate) fn auto_compact_threshold(&self) -> Option<u64> {
        if !self.options.auto_compact {
            return None;
        }
        let window = self.context_window();
        (window > 0).then(|| window * AUTO_COMPACT_THRESHOLD_PERCENT / 100)
    }

    /// What this agent runs on, for a host that has to report it.
    ///
    /// The session's own profile, which is not the process default once a resume or a switch has
    /// happened, and the only thing a status display may read if it is to agree with the turn.
    pub(crate) fn profile(&self) -> String {
        self.cells.profile.current().profile
    }

    /// The slot holding the lock on a session this agent created, for a host that has to outlive
    /// the turn that took it.
    ///
    /// The REPL needs all three of what this allows: to leave the lock held between turns, to
    /// replace it when `/fork` moves the conversation to a copy, and to drop it after its last
    /// message rather than whenever the agent happens to fall out of scope.
    pub(crate) fn session_lock_slot(&self) -> crate::store::SessionLockSlot {
        Arc::clone(&self.cells.session_lock)
    }

    /// Build an `Agent` configured for sub-agent use: silent, with no MCP readiness gate.
    ///
    /// Inherits `sandboxed_shell`, `context_messages` and the auto-compaction settings from the
    /// parent's options. `user_instructions` is deliberately *not* inherited: they describe the
    /// root agent, and a worker handed one task by one of its turns is not that agent.
    ///
    /// `sub_system_prompt` is the pre-built sub-agent system prompt (typically from
    /// `build_subagent_system_prompt`); `run_turn` uses it verbatim instead of building one
    /// dynamically.
    ///
    /// `frontend` decides where the sub-agent's output and permission requests go. The standard
    /// caller (the `agent_spawn` tool) uses [`crate::frontend::PermissionForwardingFrontend`]
    /// wrapping the parent's frontend. That wrapper drops emits (the sub-agent's report flows back
    /// via the tool result) but forwards permission prompts so the user is asked in their original
    /// UI. Tests can pass [`crate::frontend::SilentFrontend`] for fully-isolated sub-agent
    /// runs.
    ///
    /// Doesn't call `set_mcp_manager`. MCP tool dispatch from the sub-agent's registry works
    /// without an attached manager because the adapters delegate through `Arc<ServerEntry>`
    /// directly, and the paths that do need the manager (`load_tool`, the unknown-tool
    /// explanation) reach it through the registry, which
    /// [`crate::tools::mcp_adapter::install_on_worker_registry`] wires up.
    pub(crate) fn new_subagent(
        materials: &crate::session::SessionMaterials,
        // The worker's own cells. Its profile is a detached cell seeded from what the parent runs
        // on *now*: a sub-agent continues the parent's work on the parent's account, and it
        // inherits the window with the provider rather than from `parent_options`, which is a
        // clone frozen when the session was assembled and cannot hear about a switch.
        cells: crate::session::SessionCells,
        tool_registry: ToolRegistry,
        parent_options: &AgentOptions,
        sub_system_prompt: String,
        // The prompt the spawning call answered, so the worker's requests bill to it.
        inherited_prompt_id: Option<uuid::Uuid>,
    ) -> Self {
        let options = AgentOptions {
            sandboxed_shell: parent_options.sandboxed_shell,
            // A sub-agent has no `[Scheduled]` section: the jobs belong to the parent's
            // session, and `new_subagent` gives it a session of its own.
            gate_tools: None,
            context_messages: parent_options.context_messages,
            // Deliberately not inherited. Instructions are installation-wide and describe the
            // root agent; a worker handed a task by another agent is not that agent. The
            // sub-agent's system prompt is built by
            // `crate::tools::subagent::build_subagent_system_prompt` and skills are the reusable
            // worker-instruction unit.
            user_instructions: None,
            // Sub-agents run silent: no streaming UI, no MCP readiness gate.
            streaming: false,
            // Auto-compaction is inherited: a worker handed a large task has the same context
            // window as its parent and the same need to compact within it.
            auto_compact: parent_options.auto_compact,
            // Inherited for the same reason as `auto_compact`: a worker that compacts is about to
            // discard its own working state, and the checkpoint is what lets it keep the part that
            // mattered. It reaches its own memory only if the spawn granted it any, so a worker
            // with no memory access still gets the better summary and simply has nowhere to write.
            compact_checkpoint: parent_options.compact_checkpoint,
            mcp_grace: std::time::Duration::ZERO,
            system_prompt_override: Some(sub_system_prompt),
        };
        Self::new(
            materials,
            cells,
            tool_registry,
            options,
            AgentRole::Worker {
                inherited_prompt_id,
            },
        )
    }

    /// Whether this agent detaches tool calls and carries their outcomes: `[background] enabled`,
    /// with a ceiling above zero. Off for a sub-agent and for every host that never enabled it.
    pub(crate) fn background_enabled(&self) -> bool {
        self.background_max_tasks > 0
    }

    /// Snapshot of the per-session counters used by `/status`. Called from the REPL on demand.
    pub(crate) fn session_stats_snapshot(&self) -> crate::stats::SessionStatsSnapshot {
        self.session_stats.snapshot()
    }

    /// Live context occupancy for `/status`: `(tokens_in_context, context_window)`.
    ///
    /// `tokens_in_context` is the total tokens of this agent's most recent provider round (all
    /// input tiers + output) = what the next request re-sends minus the new prompt; `0` before
    /// the first turn. It is per-`Agent`, so sub-agents are excluded; a sub-agent's *returned
    /// result* counts only insofar as it became a tool result in this agent's own context.
    /// `context_window` is the resolved window for the active model (`0` if unknown).
    pub(crate) fn context_usage(&self) -> (u64, u64) {
        (
            self.cells
                .context_tokens
                .load(std::sync::atomic::Ordering::Relaxed),
            self.context_window(),
        )
    }

    /// The occupancy figures behind the `[Context budget]` block the turn path pushes to the model.
    ///
    /// `GET /v1/sessions/{id}/context` deliberately does *not* route through here: it reads the
    /// same counters off `SessionEntry` as atomics so it can answer during a turn instead of
    /// waiting on the runtime mutex. The two agree because they read the same handles, not because
    /// they share this function, so a change here has to be mirrored there.
    ///
    /// `used` is `0` until the first provider response of this process lands, which is also true of
    /// a session that was just re-attached from disk: the conversation is long but nothing has
    /// measured it yet. Callers that render a percentage must treat `0` as unmeasured rather than
    /// empty, the way [`crate::prompt::ContextBudget::render`] does.
    pub(crate) async fn context_budget(&self, session_id: Uuid) -> crate::prompt::ContextBudget {
        crate::prompt::ContextBudget {
            used: self
                .cells
                .context_tokens
                .load(std::sync::atomic::Ordering::Relaxed),
            window: self.context_window(),
            compact_at_percent: self
                .options
                .auto_compact
                .then_some(AUTO_COMPACT_THRESHOLD_PERCENT),
            generation: self.compaction_generation(session_id).await,
        }
    }

    /// The reasoning-effort value this agent's provider will send on the wire, or `None` when it
    /// sends none. Used by the `/status` model block.
    pub(crate) fn resolved_effort(&self) -> Option<String> {
        self.provider().resolved_effort()
    }

    /// Fetch the account's rate-limit usage from the active provider, for the `/usage` command.
    /// `Ok(None)` when the provider has no per-account usage endpoint.
    pub(crate) async fn fetch_usage(&self) -> Result<Option<crate::provider::AccountUsage>> {
        self.provider().fetch_usage().await
    }

    /// Tell the agent that the conversation it holds was rewritten out from under it, as `/rewind`
    /// does. Both markers the agent keeps against message *positions* stop meaning anything and are
    /// cleared: the accepted-prefix length a rejection is measured back from, and the index where
    /// the world-state delta was last rendered.
    ///
    /// Leaving either stale is silently wrong rather than loud. A stale accepted-prefix makes the
    /// degrade-and-retry recovery compute an empty suspect window and quietly not fire; a stale
    /// world-state index makes `run_turn` believe it already told the model about a tool or MCP
    /// server whose announcement the rewind just deleted, so it never mentions it again.
    ///
    /// `compact_session` clears the same two inline, since it rewrites the conversation itself.
    pub(crate) async fn reset_conversation_markers(&self) {
        self.last_accepted_len
            .store(LAST_ACCEPTED_UNKNOWN, std::sync::atomic::Ordering::Relaxed);
        *self.last_rendered_world.write().await = None;
    }

    /// The registry this agent dispatches through. Exposed so a host that attached it to the MCP
    /// manager can detach it again on the way out.
    pub(crate) fn tool_registry(&self) -> &ToolRegistry {
        &self.tool_registry
    }

    /// Background detachment for a test that built its registry by hand.
    #[cfg(test)]
    pub(crate) fn enable_background_for_test(&mut self, max_tasks: usize) {
        self.tool_registry.enable_background();
        self.background_max_tasks = max_tasks;
    }

    /// The shared registry, for the REPL's `/tasks` command and its Ctrl+C handling.
    pub(crate) fn background_tasks(&self) -> crate::background::BackgroundTasks {
        self.cells.background_tasks.clone()
    }

    /// This agent's session store, so a signal handler can record a terminal outcome without the
    /// REPL threading a second handle through every call site.
    pub(crate) fn store(&self) -> Store {
        self.store.clone()
    }

    /// Shared handle to the auto-refreshing skill cache. The REPL's `/skill <name>` dispatch reads
    /// from this so the agent's system prompt and the user-invocable list never diverge.
    pub(crate) fn skills(&self) -> &Arc<SkillCache> {
        &self.skills
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        conversation::ToolResultContent, permission::SharedPermission, provider::PublishedProfile,
    };

    /// The switch has to reach the collaborators that outlive a turn, not just the agent's own
    /// fields: `agent_spawn` and `context_check` hold the published cell, and a provider cloned
    /// when the session was assembled would go on billing the account the session had just left.
    #[tokio::test]
    async fn a_switch_reaches_everything_holding_the_published_binding() {
        let first: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(Vec::new()));
        let (agent, _store) = agent_for_test(Arc::clone(&first)).await;
        // The agent's own cell, which is what `agent_spawn` and `context_check` hold.
        agent.set_provider(ResolvedProfile {
            provider: Arc::clone(&first),
            profile: "alpha".to_string(),
            context_window: 32_000,
            vision: true,
        });
        let published = agent.cells.profile.clone();
        let gauge = published.window();
        assert_eq!(published.current().profile, "alpha");
        assert_eq!(gauge.load(std::sync::atomic::Ordering::Acquire), 32_000);

        let second: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(Vec::new()));
        agent.set_provider(ResolvedProfile {
            provider: Arc::clone(&second),
            profile: "beta".to_string(),
            context_window: 500_000,
            vision: false,
        });

        assert_eq!(published.current().profile, "beta");
        assert!(Arc::ptr_eq(&published.current().provider, &second));
        assert_eq!(gauge.load(std::sync::atomic::Ordering::Acquire), 500_000);
        assert_eq!(agent.profile(), "beta");
        assert_eq!(agent.context_window(), 500_000);
    }

    /// [`agent_for_test`] against a store on disk, for a test that has to break it from outside.
    pub(super) async fn agent_at_for_test(
        provider: Arc<dyn Provider>,
        path: &std::path::Path,
    ) -> (Agent, Store) {
        let store = Store::open(Some(path), &Default::default())
            .await
            .expect("open");
        (
            build_test_agent(provider, crate::tools::ToolRegistry::new(), &store),
            store,
        )
    }

    pub(super) async fn agent_for_test(provider: Arc<dyn Provider>) -> (Agent, Store) {
        agent_with_registry_for_test(provider, crate::tools::ToolRegistry::new()).await
    }

    /// The number three separate sites divide by, pinned exactly.
    ///
    /// The tests that drive compaction all force it, so they prove the machinery runs and say
    /// nothing about when it starts, and a wrong threshold is silent either way: compact every
    /// turn and lose history, or never compact and have the provider reject the turn.
    #[tokio::test]
    async fn the_auto_compaction_threshold_is_eighty_percent_of_the_window() {
        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(Vec::new()));
        let (mut agent, _manager) = agent_for_test(provider).await;

        agent.options.auto_compact = true;
        agent.set_context_window_for_test(200_000);
        assert_eq!(
            agent.auto_compact_threshold(),
            Some(160_000),
            "80% of 200k; a `*`/`/` slip here moves the trigger by orders of magnitude"
        );

        agent.set_context_window_for_test(1_000_000);
        assert_eq!(agent.auto_compact_threshold(), Some(800_000));

        // Not "a tiny window": a zero window means meka does not know the size, and a threshold of
        // zero would compact on the very first turn, before there is anything to summarize.
        agent.set_context_window_for_test(0);
        assert_eq!(
            agent.auto_compact_threshold(),
            None,
            "an unknown window must disable auto-compaction, not set the trigger to zero"
        );

        agent.set_context_window_for_test(200_000);
        agent.options.auto_compact = false;
        assert_eq!(
            agent.auto_compact_threshold(),
            None,
            "the config switch must win over any window"
        );
    }

    /// A harness that can reach the emergency-compaction arm.
    ///
    /// The default one cannot: it sets `auto_compact: false` and a zero window, and the guard
    /// requires both, so a test driving `FailContextOverflow` through `agent_for_test` proves only
    /// that the guard short-circuits.
    pub(super) async fn agent_that_compacts_for_test(
        provider: Arc<dyn Provider>,
    ) -> (Agent, Store) {
        let (mut agent, store) =
            agent_with_registry_for_test(provider, crate::tools::ToolRegistry::new()).await;
        agent.options.auto_compact = true;
        agent.set_context_window_for_test(200_000);
        (agent, store)
    }

    pub(super) async fn agent_with_registry_for_test(
        provider: Arc<dyn Provider>,
        registry: crate::tools::ToolRegistry,
    ) -> (Agent, Store) {
        let store = Store::for_test().await;
        let agent = build_test_agent(provider, registry, &store);
        (agent, store)
    }

    /// The agent every test harness here builds, separated so one can hand it a different store.
    pub(super) fn build_test_agent(
        provider: Arc<dyn Provider>,
        registry: crate::tools::ToolRegistry,
        store: &Store,
    ) -> Agent {
        let options = AgentOptions {
            streaming: true,
            sandboxed_shell: false,
            gate_tools: None,
            context_messages: None,
            auto_compact: false,
            compact_checkpoint: false,
            user_instructions: None,
            mcp_grace: std::time::Duration::from_secs(0),
            system_prompt_override: Some("test".to_string()),
        };
        Agent::new(
            &crate::session::SessionMaterials {
                skills: crate::skills::SkillCache::disabled(),
                memories: crate::store::memory::MemoryStore::disabled(),
                ..crate::session::SessionMaterials::for_test(store.clone())
            },
            crate::session::SessionCells::new(
                SharedPermission::new(
                    crate::permission::Permission::Read,
                    crate::permission::EnabledPermissions::ALL,
                ),
                crate::workspace::SharedCwd::new(std::env::temp_dir()),
                crate::workspace::SharedRoots::default(),
                PublishedProfile::detached(&ResolvedProfile {
                    provider,
                    profile: "test-profile".to_string(),
                    context_window: 0,
                    vision: true,
                }),
                Arc::new(crate::frontend::SilentFrontend),
            ),
            registry,
            options,
            AgentRole::Root,
        )
    }

    pub(super) fn image_source() -> ImageSource {
        ImageSource::Base64 {
            media_type: "image/png".to_string(),
            data: "QUJD".to_string(),
        }
    }

    pub(super) const REJECTION: &str = "API returned status 400 Bad Request: the image was specified using \
                             the image/png media type, but the image appears to be a image/jpeg \
                             image";

    /// A deferred tool with a documented optional parameter, standing in for mekabridge's
    /// `send_file`.
    pub(super) struct SendFileFixture;

    #[async_trait::async_trait]
    impl crate::tools::Tool for SendFileFixture {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "mcp__bridge__send_file".to_string(),
                description: "Send a file to a conversation.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"},
                        "as_photo": {
                            "type": "boolean",
                            "default": false,
                            "description": "Send as a viewable photo rather than a document.",
                        },
                    },
                    "required": ["path"]
                }),
                ..Default::default()
            }
        }

        fn required_permission(&self) -> crate::permission::Permission {
            crate::permission::Permission::Read
        }

        async fn execute(
            &self,
            input: serde_json::Value,
            _context: crate::tools::ToolContext,
        ) -> Result<crate::tools::ToolOutput> {
            // One path that fails, so a test can tell a call that ran and failed from one that
            // never ran: the two are treated differently by the schema advisory's bookkeeping.
            if input.get("path").and_then(serde_json::Value::as_str) == Some("/missing.png") {
                return Ok(crate::tools::ToolOutput::text(
                    "No such file: /missing.png".to_string(),
                    true,
                ));
            }
            Ok(crate::tools::ToolOutput::text(
                "Sent (message id 1)".to_string(),
                false,
            ))
        }
    }

    pub(super) fn send_file_registry() -> crate::tools::ToolRegistry {
        let registry = crate::tools::ToolRegistry::new();
        registry.register_load_tool_for_test();
        registry
            .register(Arc::new(SendFileFixture))
            .expect("register fixture");
        registry.mark_deferred("mcp__bridge__send_file");
        registry
    }

    pub(super) fn user_message(text: &str) -> Message {
        Message::user(text)
    }

    pub(super) fn assistant_message(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    pub(super) fn assistant_tool_use() -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                input: serde_json::json!({"path": "/tmp/test"}),
            }],
        }
    }

    pub(super) fn tool_result_message() -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: vec![ToolResultContent::Text {
                    text: "file contents".to_string(),
                }],
                is_error: false,
            }],
        }
    }
}
