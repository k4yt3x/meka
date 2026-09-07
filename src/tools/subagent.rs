//! `agent_spawn` tool: delegates a self-contained research/exploration task to a fresh sub-agent
//! with its own conversation, returning the sub-agent's final report as a single tool result.

use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;
use uuid::Uuid;

use super::{Tool, ToolDenials, ToolOutput, ToolRegistry};
use crate::{
    agent::Agent,
    config::{BuiltinToolFilter, InstructionAccess, MemoryAccess},
    conversation::Conversation,
    error::{MekaError, Result},
    permission::{EnabledPermissions, Permission, SharedPermission},
    prompt::build_environment_context,
    provider::ToolDefinition,
    session::AgentOptions,
    workspace::{
        BoundedWorkspace, SharedCwd, SharedRoots, accept_writable_roots, admit_within_parent_reach,
    },
};

/// Hard ceiling on sub-agent nesting depth, independent of the tunable `session.subagent_max_depth`
/// budget. Guarantees recursion always terminates even if an agent re-grants `max_depth` at every
/// level: no agent nested deeper than this is given a `agent_spawn` tool, so the tree can never
/// exceed this height.
const SUBAGENT_ABSOLUTE_MAX_DEPTH: usize = 16;

/// What a spawning agent hands its workers: the session's materials, its own live cells, and the
/// ceilings a worker may not exceed.
#[derive(Clone)]
pub(crate) struct ToolBuilderParams {
    pub(crate) materials: crate::session::SessionMaterials,
    /// The spawning agent's own cells. `cells.profile` is what the parent runs on *now*, read at
    /// spawn time rather than at registration: a worker spawned after a `/profile` switch went
    /// to the profile the user had just left, billing that account, while its own row recorded
    /// the new one. `cells.cwd` is snapshotted per spawn so a parent `/cd` mid-turn cannot move a
    /// running worker; the roots are shared, because nothing mutates them after construction. A
    /// worker bounded by `writable_roots` takes neither: its directory and roots are its own.
    pub(crate) cells: crate::session::SessionCells,
    /// How much of the memory store the agent doing the spawning holds, which is the ceiling on
    /// what it can grant. `Write` for the root agent; for a worker, whatever its own spawn call
    /// granted. A grant is clamped against this, so authority only ever narrows going down the
    /// tree.
    pub(crate) memory_access: MemoryAccess,
    /// `[subagents].disabled_servers` / `disabled_tools` as config reads *now*.
    ///
    /// Distinct from `AgentSpawnTool::inherited_denials`, which is the accumulated set for this
    /// agent's children. This one exists for the follow-up path: a worker is rebuilt from the
    /// terms it was spawned under, so without re-applying current config, an operator who adds
    /// a denial and resumes a session would find their existing workers still reaching what
    /// they just took away. Restrictions are combined, never replaced, so this can only ever
    /// narrow.
    pub(crate) config_denials: ToolDenials,
    /// The spawning agent's options, from which a worker inherits `sandboxed_shell`,
    /// `context_messages` and the auto-compaction settings inside [`Agent::new_subagent`].
    /// `user_instructions` is deliberately *not* among them; see [`build_subagent_system_prompt`].
    pub(crate) parent_options: AgentOptions,
}

/// The terms a sub-agent was spawned under, persisted as JSON on its session row
/// (`sessions.subagent_spec_json`).
///
/// **A follow-up rebuilds the worker from this, never from the parent's current state.** That is
/// the whole reason it exists. Rebuilding from the parent would mean `agent_spawn({permission:
/// "read"})` followed by `agent_followup` runs the same worker at whatever the parent is now,
/// turning a second question into a one-call privilege escalation. The same holds for the deny
/// lists and the memory level: every restriction the spawn call chose has to outlive the call.
///
/// The `#[serde(default)]` on every field but `permission` is an integrity guard, not tolerance
/// for an older shape: `meka session import` writes this JSON verbatim from a user-supplied
/// archive, and a hand-edited row can drop any field. Each default is the *restrictive* value, so a
/// spec that lost a field lost authority rather than gaining it; a spec missing `permission` has no
/// restrictive reading and is a hard decode error, which `agent_followup` turns into a refusal.
///
/// What is deliberately *not* here: the task (already in the event log), and the cwd of a worker
/// that shares its parent's workspace (already on the session row). A worker bounded by
/// `writable_roots` records the list here as well as on the row, because the row cannot say
/// whether an empty root list means "no boundary of its own" or "bounded to the cwd alone".
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct SubagentSpec {
    /// The level the worker ran at, already clamped against its parent at spawn time.
    pub(crate) permission: Permission,
    /// The clamped enabled-permission set, so a rebuilt worker cannot climb past the ceiling its
    /// spawn call set even if a future runtime switch path appears.
    #[serde(default)]
    pub(crate) enabled_permissions: Vec<Permission>,
    #[serde(default)]
    pub(crate) denied_servers: Vec<String>,
    #[serde(default)]
    pub(crate) denied_tools: Vec<String>,
    /// A spec that somehow lacks the field costs the worker its memory tools rather than handing
    /// it the store. [`MemoryAccess`] has no `Default` for the same reason: the only sensible one
    /// would be `Write`, which is the root agent's level and the wrong answer everywhere else.
    #[serde(default = "MemoryAccess::none")]
    pub(crate) memory: MemoryAccess,
    /// Whether the worker was handed the installation's instructions file. Defaults to `None` for
    /// a spec that lost the field, for the same fail-closed reason as `memory`.
    #[serde(default)]
    pub(crate) instructions: InstructionAccess,
    /// Parent scratchpad names the worker may read. Persisted because it is baked into both the
    /// registry and the system prompt at spawn; without it a follow-up would silently lose the
    /// entries the first turn was working from.
    #[serde(default)]
    pub(crate) inherited_scratchpad: Vec<String>,
    /// The worker's own recursion budgets, so a followed-up worker can spawn exactly what it could
    /// have spawned on its first turn.
    #[serde(default)]
    pub(crate) remaining_depth: usize,
    #[serde(default)]
    pub(crate) absolute_depth: usize,
    /// The write boundary the spawn call set through `writable_roots`, canonical and working
    /// directory first; empty for a worker that shares its parent's workspace. Not a restrictive
    /// default, and the one field where none exists: a spec that lost it falls back to the row's
    /// cwd plus the parent's roots, which is the boundary its spawn call gave it, so nothing wider
    /// is reachable from the absence.
    #[serde(default)]
    pub(crate) writable_roots: Vec<PathBuf>,
}

impl SubagentSpec {
    fn denials(&self) -> ToolDenials {
        ToolDenials::new(self.denied_servers.clone(), self.denied_tools.clone())
    }

    /// The worker's `SharedPermission`, clamped to `ceiling` on top of what the spec recorded.
    ///
    /// Two ceilings, because there are two ways a worker could end up with authority it should not
    /// have. The spec stops a follow-up escalating a worker its spawn call deliberately
    /// restricted. `ceiling` (the parent's level right now) stops a worker outliving a
    /// restriction: spawn at `unrestricted`, switch the session to `read`, and without this the
    /// worker would still run at `unrestricted` on the next follow-up. The effective level is the
    /// lower of the two.
    ///
    /// Falls back to a singleton set when the persisted list is empty or invalid, rather than to
    /// `EnabledPermissions::ALL`: an unreadable spec must not widen what the worker can do.
    ///
    /// Test-only. Production goes through [`Self::shared_permission_bounded`], which applies this
    /// same clamp and then stays bound to the parent; this one is kept because the clamp is worth
    /// asserting in isolation from the tracking.
    #[cfg(test)]
    fn shared_permission(&self, ceiling: Permission) -> SharedPermission {
        let effective = self.effective_permission(ceiling);
        SharedPermission::new(effective, self.clamped_enabled(effective))
    }

    /// Same, but bounded by the parent's *live* level rather than a snapshot of it.
    ///
    /// This is what production uses. The spawn-time clamp is still applied (a worker granted `read`
    /// under a `write` parent stays at `read`), and the ceiling then tracks the parent afterwards,
    /// so cycling the parent down to `none` stops the worker on its next tool call instead of
    /// letting it run to completion at the level it started with.
    fn shared_permission_bounded(&self, parent: &SharedPermission) -> SharedPermission {
        let effective = self.effective_permission(parent.get());
        SharedPermission::with_ceiling(effective, self.clamped_enabled(effective), parent)
    }

    fn clamped_enabled(&self, effective: Permission) -> EnabledPermissions {
        let enabled = EnabledPermissions::from_levels(self.enabled_permissions.iter().copied())
            .unwrap_or_else(|| {
                EnabledPermissions::from_levels([effective]).unwrap_or(EnabledPermissions::DEFAULT)
            });
        clamp_enabled_permissions(enabled, effective)
    }

    /// The memory level this worker actually gets, capped at `Read`.
    ///
    /// `MemoryAccess::parse_grant` refuses `"write"` at the `agent_spawn` boundary, but a spec is
    /// persisted JSON and `meka session import` writes `subagent_spec_json` verbatim from a
    /// user-supplied archive, where `Write` deserializes fine. The documented guarantee is that no
    /// sub-agent can write to the store, so it is enforced where the level is consumed rather
    /// than resting on every writer having validated first, the same shape as
    /// [`clamp_enabled_permissions`], which bounds a persisted permission set for the same reason.
    fn granted_memory(&self) -> MemoryAccess {
        self.memory.min(MemoryAccess::Read)
    }

    /// The level this worker actually runs at, given the parent's current ceiling. What a
    /// **replayed** grant resolves to under the parent's current level.
    ///
    /// `greatest_within_both`, not `clamp_to`: a follow-up must never run the worker at more than
    /// the spawn call asked for. On the ladder `none` < `read` < `workspace` < `unrestricted`,
    /// `clamp_to` answers the *spawn* question (the request when the parent holds it, else the
    /// parent's own level), so a parent that has since been raised would hand a recorded `read`
    /// its new rung. A later parent change may narrow a recorded grant and never widen it, which
    /// is the minimum of the two. See the helper for why spawn and replay are different questions.
    fn effective_permission(&self, ceiling: Permission) -> Permission {
        let own = if self.writable_roots.is_empty() {
            self.permission
        } else {
            // A boundary is a `workspace` thing: `agent_spawn` refuses `unrestricted` alongside
            // `writable_roots`, but a spec is persisted JSON that `meka session import` writes
            // verbatim, so the cap is applied where the level is consumed, as `granted_memory`
            // does for the store.
            self.permission.greatest_within_both(Permission::Workspace)
        };
        own.greatest_within_both(ceiling)
    }
}

pub(crate) struct AgentSpawnTool {
    pub(crate) parent_permission: SharedPermission,
    pub(crate) tool_builder_params: ToolBuilderParams,
    /// Everything a sub-agent spawned by *this* tool is denied. Seeded from `[subagents]` at the
    /// root and, for a nested `agent_spawn`, from its own parent's effective set unioned with what
    /// that spawn call added.
    ///
    /// Carried on the tool rather than looked up per call because it has to accumulate: a worker
    /// that could spawn a grandchild free of its own denials would make every restriction one
    /// `agent_spawn` deep.
    pub(crate) inherited_denials: ToolDenials,
    /// Soft, agent-tunable recursion budget for sub-agents spawned by this tool. The root tool is
    /// seeded from `session.subagent_max_depth`; each level hands the child `remaining_depth - 1`
    /// unless the caller overrides it via the `max_depth` param. A nested `agent_spawn` is granted
    /// only while the child's budget is `>= 1`.
    pub(crate) remaining_depth: usize,
    /// Monotonic absolute nesting depth of the agent holding this tool (root = 0). Unlike
    /// `remaining_depth` it can't be reset by `max_depth`, so it bounds real recursion at
    /// [`SUBAGENT_ABSOLUTE_MAX_DEPTH`] regardless of what the agent requests.
    pub(crate) absolute_depth: usize,
}

/// `agent_spawn`'s schema, as a free function so a caller holding no tool can still name it.
pub(crate) fn agent_spawn_definition() -> ToolDefinition {
    ToolDefinition {
        name: "agent_spawn".to_string(),
        description: "Spawn a sub-agent to perform a research, analysis, or delegated task. \
                      The sub-agent inherits the parent's permission level, has its own \
                      private todo list and scratchpad, and returns a single text report. \
                      Multiple agent_spawn calls in one turn run in parallel. Pass `skill` \
                      to run an installed skill in the sub-agent. The skill's instructions \
                      become the sub-agent's task; supply at least one of `prompt` or \
                      `skill`. Use `inherit_scratchpad` to grant read-only access to \
                      specific parent scratchpad entries by name so the sub-agent can \
                      consume large captured output via `scratchpad_read` without you \
                      re-inlining it in the prompt. Tip: when you expect to hand output to a \
                      sub-agent later, set the `scratchpad` parameter on the originating \
                      tool call (e.g. `execute_command({command: \"...\", scratchpad: \
                      \"build_log\"})`) so the entry has a semantic name you can pass \
                      through `inherit_scratchpad`. Sub-agents may themselves spawn further \
                      sub-agents up to a configured depth; tune a subtree's depth with \
                      `max_depth`. Pass `permission` to run the sub-agent at a more \
                      restricted level than your own (you can restrict but never escalate), \
                      `writable_roots` to confine its writes to directories within your own \
                      reach, and `deny_servers` / `deny_tools` to withhold MCP servers or \
                      individual tools it would otherwise inherit. Restrictions only ever \
                      accumulate: these add to whatever the installation already denies \
                      sub-agents, and there is no way to grant back."
            .to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task description for the sub-agent. Optional when \
                                    `skill` is given; otherwise required."
                },
                "skill": {
                    "type": "string",
                    "description": "Name of an installed skill to run in the sub-agent. The \
                                    skill's instructions become the sub-agent's task; \
                                    `prompt`, if also given, is prepended as extra direction."
                },
                "scratchpad": {
                    "type": "string",
                    "description": "If provided, save the sub-agent's final report to the \
                                    parent's scratchpad under this name instead of returning \
                                    it inline."
                },
                "inherit_scratchpad": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Names of the parent's scratchpad entries the sub-agent \
                                    is allowed to read. The sub-agent's `scratchpad_read` \
                                    falls back to the parent for these names; \
                                    `scratchpad_list` shows them with origin `inherited`. \
                                    Read-only: `scratchpad_write` / `_edit` / `_delete` \
                                    targeting an inherited name return an error so the \
                                    sub-agent can't silently shadow your copy. Names that \
                                    don't exist in the parent are silently skipped."
                },
                "permission": {
                    "type": "string",
                    "enum": ["none", "read", "workspace", "unrestricted"],
                    "description": "Permission level for the sub-agent, never above your \
                                    own. Defaults to your current level; use a lower one \
                                    (e.g. \"read\") to sandbox risky work."
                },
                "writable_roots": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Directories the sub-agent may write under, and nowhere \
                                    else. The first becomes its working directory, so relative \
                                    paths in its tool calls resolve inside it; the rest are its \
                                    additional workspace roots. Each must be an existing \
                                    directory (relative entries resolve against your working \
                                    directory) and, unless you are at `unrestricted`, must lie \
                                    inside your own workspace: a sub-agent's reach never exceeds \
                                    yours. Needs you at `workspace` or above; the sub-agent runs \
                                    at `workspace` unless `permission` asks for less, and \
                                    `permission: \"unrestricted\"` is refused alongside it since \
                                    the list would then bound nothing. Omit it to share your \
                                    workspace; an empty list is refused."
                },
                "memory": {
                    "type": "string",
                    "enum": ["none", "read"],
                    "default": "none",
                    "description": "Grant the sub-agent read access to your memory store. \
                                    Defaults to \"none\": a sub-agent starts with a clean slate, \
                                    since memories from unrelated work are context it pays for \
                                    and reasons from. Grant \"read\" when the task genuinely \
                                    depends on what you have recorded. Sub-agents can never \
                                    write to the store; record anything worth keeping \
                                    yourself, from the sub-agent's report."
                },
                "instructions": {
                    "type": "string",
                    "enum": ["none", "inherit"],
                    "default": "none",
                    "description": "Give the sub-agent the installation's instructions file. \
                                    Defaults to \"none\", because those instructions describe \
                                    you (your persona, how to address the user), and a \
                                    sub-agent is not you. Pass \"inherit\" when the task needs \
                                    the project's standing rules verbatim and quoting the \
                                    relevant ones into `prompt` would be lossy or expensive."
                },
                "deny_servers": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "MCP server names the sub-agent must not see. Removes \
                                    everything the server offers: its tools, its resources, \
                                    and its prompts. Use this when a server exists to act on \
                                    your behalf or to talk to the user, so a sub-agent cannot \
                                    speak as you."
                },
                "deny_tools": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Individual tool names the sub-agent must not see, as they \
                                    appear in your own tool list (e.g. \"write_file\", \
                                    \"mcp__notion__create_page\"). For a whole server, prefer \
                                    `deny_servers`."
                },
                "max_depth": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Override how many further levels of sub-agents this \
                                    sub-agent may itself spawn. Defaults to one less than your \
                                    own remaining budget. 0 forbids it from spawning further; \
                                    larger values are still bounded by a built-in absolute \
                                    recursion cap."
                }
            }
        }),
        ..Default::default()
    }
}

#[async_trait]
impl Tool for AgentSpawnTool {
    fn definition(&self) -> ToolDefinition {
        agent_spawn_definition()
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let cancellation = context.cancellation.clone();
        // Both `prompt` and `skill` are optional, but at least one must be present. This mirrors
        // the CLI's `--oneshot` guard in `src/main.rs`. An empty/whitespace `prompt` counts
        // as absent.
        let prompt = input["prompt"]
            .as_str()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string);
        let skill_name = input["skill"]
            .as_str()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string);
        if prompt.is_none() && skill_name.is_none() {
            return Err(MekaError::ToolExecution {
                tool_name: "agent_spawn".to_string(),
                message: "agent_spawn requires 'prompt', 'skill', or both".to_string(),
            });
        }

        // Resolve the skill against the shared cache up front, before any session is created, so a
        // bad name fails fast without leaving an orphan child session behind.
        let skill = match &skill_name {
            Some(name) => {
                let installed = self.tool_builder_params.materials.skills.current().await;
                match installed.find(name) {
                    Some(skill) => Some(skill.clone()),
                    // A skill whose `SKILL.md` will not parse is in no index, so listing what *is*
                    // available answers a question the caller did not ask and invites it to pick a
                    // substitute. Name the file instead, the way `skill_read` does.
                    None if installed.skip_reason(name).is_some() => {
                        return Ok(ToolOutput::text(
                            format!(
                                "Error: {}. Tell the user; they need to fix that file. Do not \
                                 delegate a procedure you invented in its place.",
                                installed.unavailable(name)
                            ),
                            true,
                        ));
                    }
                    None => {
                        let available: Vec<&str> = installed
                            .skills
                            .iter()
                            .map(|skill| skill.name.as_str())
                            .collect();
                        let hint = if available.is_empty() {
                            "No skills are installed.".to_string()
                        } else {
                            format!("Available skills: {}", available.join(", "))
                        };
                        return Err(MekaError::ToolExecution {
                            tool_name: "agent_spawn".to_string(),
                            message: format!("skill '{name}' not found. {hint}"),
                        });
                    }
                }
            }
            None => None,
        };

        // `inherit_scratchpad`: optional array of parent-scratchpad names.
        let inherited_scratchpad = string_array(&input, "inherit_scratchpad", "agent_spawn")?;

        // Resolve the sub-agent's permission: an optional `permission` param clamped to the
        // parent's level as a ceiling (restrict-only, never escalate); absent keeps the parent's
        // level. Approval prompts route through `PermissionForwardingFrontend` so they surface in
        // the parent's UI.
        let requested_permission =
            parse_subagent_permission(optional_str(&input, "permission", "agent_spawn")?)?;
        let parent_level = self.parent_permission.get();

        // `writable_roots`: an optional list that bounds the worker's writes and sets its working
        // directory. Judged here, ahead of every side effect, by the one acceptor `agent_followup`
        // also uses. Absent and null mean "share the parent's workspace"; a present list is
        // accepted whole or refused whole, so an empty one is a refusal rather than a silent
        // fall-through to the parent's reach.
        let bounded = match input.get("writable_roots") {
            None | Some(serde_json::Value::Null) => None,
            Some(_) => {
                if requested_permission == Some(Permission::Unrestricted) {
                    return Err(MekaError::ToolExecution {
                        tool_name: "agent_spawn".to_string(),
                        message: "writable_roots bounds writes, and a sub-agent at `unrestricted` \
                                  has no boundary for it to bound. Drop one or the other."
                            .to_string(),
                    });
                }
                let entries: Vec<PathBuf> = string_array(&input, "writable_roots", "agent_spawn")?
                    .into_iter()
                    .map(PathBuf::from)
                    .collect();
                let cells = &self.tool_builder_params.cells;
                Some(
                    accept_writable_roots(parent_level, &cells.cwd, &cells.roots, &entries)
                        .map_err(|error| MekaError::ToolExecution {
                            tool_name: "agent_spawn".to_string(),
                            message: error.to_string(),
                        })?,
                )
            }
        };
        // A bounded worker runs at `workspace` unless the call asked for less: the list is a
        // boundary, and `workspace` is the level that has one.
        let sub_perm = resolve_subagent_permission(
            requested_permission.or(bounded.as_ref().map(|_| Permission::Workspace)),
            parent_level,
        );

        // Union, never replace: the call site adds to what config (and, when nested, this agent's
        // own parent) already denied. There is deliberately no allow-list parameter, because one
        // would let a parent hand a worker something the installation took away.
        let call_site_denials = ToolDenials::new(
            string_array(&input, "deny_servers", "agent_spawn")?,
            string_array(&input, "deny_tools", "agent_spawn")?,
        );
        // A name that matches nothing denies nothing, and the model gets no signal either way: it
        // asked for a sandboxed worker and would receive an unsandboxed one believing otherwise.
        // Warned rather than refused, because "deny it if it is there" is a legitimate thing to
        // write against a server list that varies by machine.
        if let Some(weak) = self.tool_builder_params.materials.mcp_manager.as_ref()
            && let Some(manager) = weak.upgrade()
        {
            let configured = manager.server_names();
            for name in call_site_denials.server_list() {
                if !configured.contains(&name) {
                    tracing::warn!(
                        "agent_spawn deny_servers entry '{name}' matches no configured MCP server, so \
                         it denies nothing"
                    );
                }
            }
        }
        let effective_denials = self.inherited_denials.union(&call_site_denials);

        // Context grants. Both default to nothing and are clamped against what this agent itself
        // holds, so a worker can never hand a grandchild more than it was given. Unlike the deny
        // lists, these are grants rather than restrictions: config cannot meaningfully withhold
        // them (a parent holding the text can copy it into the prompt), so the decision is the
        // parent's, and the safe state is the default.
        let memory_access = match optional_str(&input, "memory", "agent_spawn")? {
            Some(text) => {
                let requested = MemoryAccess::parse_grant(text).map_err(|message| {
                    MekaError::ToolExecution {
                        tool_name: "agent_spawn".to_string(),
                        message,
                    }
                })?;
                requested.min(self.tool_builder_params.memory_access)
            }
            None => MemoryAccess::None,
        };
        // The clamp for instructions is the text itself: a worker spawned without them has `None`
        // here (see `build_subagent`), so asking to pass them on is silently a no-op rather than a
        // hole. An installation with no instructions file behaves the same way.
        let parent_has_instructions = self
            .tool_builder_params
            .parent_options
            .user_instructions
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty());
        let instructions = match optional_str(&input, "instructions", "agent_spawn")? {
            Some(text) => {
                let requested = InstructionAccess::parse_grant(text).map_err(|message| {
                    MekaError::ToolExecution {
                        tool_name: "agent_spawn".to_string(),
                        message,
                    }
                })?;
                if parent_has_instructions {
                    requested
                } else {
                    InstructionAccess::None
                }
            }
            None => InstructionAccess::None,
        };

        // Optional `max_depth`: the caller's override for how deep this sub-agent's own subtree may
        // recurse. Consumed by `child_spawn_depth` when deciding whether to grant a nested
        // `agent_spawn` below.
        let max_depth_override =
            optional_u64(&input, "max_depth", "agent_spawn")?.map(|value| value as usize);

        // Resolve parent session ID. By the time a tool runs, `Agent::run_turn` has already written
        // `shared_session_id` before dispatching tools. A missing value here means an agent ran a
        // tool without first creating its session, an internal invariant break worth surfacing
        // rather than silently producing an orphan.
        let parent_sid = self
            .tool_builder_params
            .cells
            .session_id
            .get()
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "agent_spawn".to_string(),
                message: "parent session ID not yet assigned (run_turn invariant)".to_string(),
            })?;

        // Read the skill body before any row is written, for the same reason the name was resolved
        // up front: an unreadable or oversized skill file fails here, and failing after
        // `create_child_session` would leave a childless session row that `agent_list` then
        // advertises as a worker you can follow up on.
        //
        // `load_skill_body` prepends the base-directory header so the skill's relative references
        // resolve against the skill rather than the sub-agent's working directory.
        let skill_body = match &skill {
            Some(skill) => {
                let body = crate::skills::load_skill_body(skill)
                    .await
                    .map_err(|error| MekaError::ToolExecution {
                        tool_name: "agent_spawn".to_string(),
                        message: format!("failed to load skill: {error}"),
                    })?;
                Some(body)
            }
            None => None,
        };

        // Compose the first-turn task: parent directive first, skill body second. The at-least-one
        // check above guarantees a `Some`, but it is resolved here, before the row exists, so that
        // every fallible step of a spawn happens while there is still nothing to clean up.
        let task =
            compose_subagent_task(prompt.as_deref(), skill_body.as_deref()).ok_or_else(|| {
                MekaError::ToolExecution {
                    tool_name: "agent_spawn".to_string(),
                    message: "agent_spawn requires 'prompt', 'skill', or both".to_string(),
                }
            })?;

        // Bound the sub-agent's own recursion budget before the spec is written:
        // `child_spawn_depth` turns this tool's counters plus the optional `max_depth`
        // override into the counters the *child* holds, and the spec has to carry those so
        // a followed-up worker can spawn exactly what it could have spawned on its first
        // turn.
        let (child_remaining_depth, child_absolute_depth, _allow_nested_spawn) = child_spawn_depth(
            self.remaining_depth,
            self.absolute_depth,
            max_depth_override,
        );

        // Snapshot the parent's cwd once, here, so a parent `/cd` mid-sub-agent execution can't
        // shift the sub-agent's path resolution mid-flight. The same value is written to the
        // child's session row, handed to its tool registry, and used to render its environment
        // context; a follow-up reads it back off the row. A bounded worker takes its directory
        // and its roots from the accepted list instead, and nothing of the parent's: the roots it
        // holds are exactly what its row and its spec record.
        let (sub_cwd_snapshot, sub_additional_roots, sub_roots) = match &bounded {
            Some(workspace) => (
                workspace.cwd.clone(),
                workspace.additional_roots.clone(),
                SharedRoots::new(workspace.additional_roots.clone()),
            ),
            None => (
                self.tool_builder_params.cells.cwd.get(),
                Vec::new(),
                self.tool_builder_params.cells.roots.clone(),
            ),
        };
        let workspace = WorkerWorkspace {
            cwd: SharedCwd::new(sub_cwd_snapshot.clone()),
            roots: sub_roots,
        };

        let spec = SubagentSpec {
            permission: sub_perm,
            enabled_permissions: clamp_enabled_permissions(
                self.parent_permission.enabled(),
                sub_perm,
            )
            .iter()
            .collect(),
            denied_servers: effective_denials.server_list(),
            denied_tools: effective_denials.tool_list(),
            memory: memory_access,
            instructions,
            inherited_scratchpad: inherited_scratchpad.clone(),
            remaining_depth: child_remaining_depth,
            absolute_depth: child_absolute_depth,
            writable_roots: bounded
                .as_ref()
                .map(BoundedWorkspace::roots)
                .unwrap_or_default(),
        };
        let spec_json = serde_json::to_string(&spec).map_err(|error| MekaError::ToolExecution {
            tool_name: "agent_spawn".to_string(),
            message: format!("failed to encode sub-agent spec: {error}"),
        })?;

        // Create the sub-agent's own DB session, linked back to the parent via `parent_session_id`.
        // Cascade-on-delete in `delete_session` sweeps it when the parent is removed.
        let (sub_session_id, sub_session_lock) = self
            .tool_builder_params
            .materials
            .store
            .create_child_session(
                parent_sid,
                Some(sub_cwd_snapshot.clone()),
                sub_additional_roots,
                Some(spec_json),
                // The level the worker runs at, on the row like every other session's, so the row
                // answers for it wherever a row is read.
                sub_perm.to_string(),
                // The same cell `build_subagent` reads a moment later, so the row this writes and
                // the provider the worker is built on cannot come apart.
                self.tool_builder_params.cells.profile.current().profile,
            )
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_spawn".to_string(),
                message: format!("failed to create sub-agent session: {error}"),
            })?;
        // Held for the whole of the worker's run, then released with this scope. Unlocked, a
        // sub-agent's row can be taken by a concurrent `meka session delete --all` and its
        // conversation cascaded away mid-turn. A failure to claim is a warning rather than a
        // refusal, matching the root agent's own creation path: the id is one nobody else can be
        // holding, so the only way here is a filesystem problem, and refusing to spawn over that
        // would break installations that work today.
        let _sub_session_lock = match sub_session_lock {
            Ok(lock) => Some(lock),
            Err(error) => {
                tracing::warn!("sub-agent session {sub_session_id} is running unlocked: {error}");
                None
            }
        };
        tracing::info!("spawning sub-agent {sub_session_id} for parent {parent_sid}");

        let sub_roots_snapshot = workspace.roots.get();
        let environment_context =
            build_environment_context(sub_perm, &sub_cwd_snapshot, &sub_roots_snapshot);
        let augmented_prompt = format!("{environment_context}\n{task}");

        // The last step that can fail before the worker exists in its own right. Nothing here is
        // reachable in practice (the web client is built from config the root already used, and a
        // fresh registry cannot collide), but the row is already on disk, so a failure would leave
        // a childless session that `agent_list` advertises and `agent_followup` would resume into
        // an empty conversation. Roll it back rather than rely on the failure staying unreachable.
        //
        // Deliberately *not* extended to `run_turn` below: once the worker has started, a provider
        // error or a cancellation leaves a real conversation the parent may still want to read or
        // follow up on, and deleting that would discard work.
        let sub_agent = match build_subagent(
            &self.tool_builder_params,
            &spec,
            parent_sid,
            sub_session_id,
            workspace,
            "agent_spawn",
            &context,
        )
        .await
        {
            Ok(agent) => agent,
            Err(error) => {
                if let Err(cleanup) = self
                    .tool_builder_params
                    .materials
                    .store
                    .delete_session(sub_session_id)
                    .await
                {
                    tracing::warn!(
                        "failed to build sub-agent {sub_session_id} and then failed to remove its \
                         session row: {cleanup}"
                    );
                }
                return Err(error);
            }
        };

        // Run the sub-agent's single turn via the shared `Agent::run_turn` path. Conversation
        // persistence (user message, assistant messages, tool results) happens inside `run_turn`
        // against the sub-session, so the audit trail is identical to the root agent's. Silent
        // rendering and the omitted MCP gate are baked into the options via `new_subagent`.
        let mut messages = Conversation::new();
        // Mark every provider request made during this run as a sub-agent request so the Claude
        // OAuth billing header carries `cc_is_subagent=true;` (the provider is a shared `Arc`, so
        // the flag rides a task-local rather than provider state).
        sub_agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts(augmented_prompt, Vec::new()).map_err(
                    |empty| MekaError::ToolExecution {
                        tool_name: "agent_spawn".to_string(),
                        message: empty.to_string(),
                    },
                )?,
                cancellation,
            )
            .await?;

        let report = messages
            .last_assistant_text()
            .unwrap_or_else(|| "(sub-agent produced no final text)".to_string());
        // Lead with the id so the report stays the tail of the output, where a model reading a long
        // result looks for the conclusion.
        //
        // Skipped when the caller redirected the report to a scratchpad. That redirect is universal
        // (`scratchpad::save_explicit_scratchpad_results` keys off the `scratchpad` argument, not
        // the tool) and stores the whole result text, so a header here would be written into the
        // entry and handed to whatever later reads it, while the model, which now sees only a
        // reference, would not get the id anyway. A scratchpad holds output the parent means to
        // pass around; the id is metadata about the call, and `agent_list` is where to find it.
        let output = if super::util::redirects_to_scratchpad(&input) {
            report
        } else {
            format!("agent: {sub_session_id}\n\n{report}")
        };
        Ok(ToolOutput::text(output, false))
    }
}

/// Register `agent_spawn` and the three lifecycle tools that operate on what it produced.
///
/// One function so the four always arrive together. `agent_followup` and `agent_delete` are useless
/// without `agent_spawn`, and `agent_spawn` without them is the one-shot worker this replaced: a
/// registry with three of the four is a shape nobody wants.
///
/// Filtering goes through `registry.admits`, which asks the registry *being written to* rather than
/// the tool being written. The distinction is the whole point: `AgentSpawnTool::inherited_denials`
/// is what this agent's future *children* are denied, and on the root agent that is the
/// `[subagents]` config. Reading it here would make `[subagents] disabled_tools = ["agent_spawn"]`
/// (the natural way to write "workers may not spawn workers") delete `agent_spawn` from the root
/// agent and turn delegation off entirely.
///
/// Takes the already-built `AgentSpawnTool` because its depth counters differ between the root
/// (seeded from config) and a nested level (derived from the parent's).
pub(crate) fn register_subagent_tools(
    registry: &ToolRegistry,
    spawn: AgentSpawnTool,
) -> Result<()> {
    let params = spawn.tool_builder_params.clone();
    // The permission of the agent this registry belongs to. `AgentSpawnTool` clamps new workers
    // against it; `AgentFollowupTool` clamps rehydrated ones against it too.
    let spawn_permission = spawn.parent_permission.clone();
    // One shared map per registry, so two parallel `agent_followup` calls on the same worker see
    // each other. A map per tool would make the guard a no-op.
    let in_flight: InFlightFollowups =
        Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
    let delete_in_flight = Arc::clone(&in_flight);

    let mut tools: Vec<(&str, Arc<dyn Tool>)> = vec![("agent_spawn", Arc::new(spawn))];
    tools.push((
        "agent_list",
        Arc::new(AgentListTool {
            tool_builder_params: params.clone(),
        }),
    ));
    tools.push((
        "agent_followup",
        Arc::new(AgentFollowupTool {
            parent_permission: spawn_permission,
            tool_builder_params: params.clone(),
            in_flight,
        }),
    ));
    tools.push((
        "agent_delete",
        Arc::new(AgentDeleteTool {
            tool_builder_params: params,
            in_flight: delete_in_flight,
        }),
    ));

    // All four or none, gated on `agent_spawn`. The three lifecycle tools only ever operate on what
    // `agent_spawn` produced, so an agent that cannot delegate has nothing for them to act on, and
    // `disabled_tools = ["agent_spawn"]` means "no delegation", not "no new delegation while
    // keeping the ability to drive workers a previous run left behind".
    if !registry.admits("agent_spawn") {
        return Ok(());
    }
    for (name, tool) in tools {
        if !registry.admits(name) {
            continue;
        }
        registry.register(tool)?;
    }
    Ok(())
}

/// Whether a session assembled with these settings gets the `agent_*` family at all.
///
/// The two all-or-nothing conditions [`register_subagent_tools`] and `assemble_agent` between them
/// impose, named once so `meka tools list` can answer the question without assembling a session.
/// Per-tool `[tools]` entries are *not* here: denying `agent_list` alone removes only that one, and
/// the caller applies [`BuiltinToolFilter::admits`] per name on top of this.
///
/// Takes the filter rather than a [`ToolRegistry`] because the listing builds its reference
/// registry unfiltered on purpose, to read every tool's hardcoded permission level; asking that
/// registry would answer "yes" no matter what the user configured.
pub(crate) fn agent_tools_registered(
    filter: &BuiltinToolFilter,
    subagent_max_depth: usize,
) -> bool {
    subagent_max_depth >= 1 && filter.admits("agent_spawn")
}

/// The family [`agent_tools_registered`] decides about, for a listing that shows it as denied.
pub(crate) const AGENT_TOOL_NAMES: [&str; 4] = [
    "agent_spawn",
    "agent_list",
    "agent_followup",
    "agent_delete",
];

/// Parse the `id` argument, without checking ownership.
///
/// Split out so a caller can claim the per-worker guard *before* verifying ownership: verifying
/// first leaves an await boundary between the check and the claim, which is exactly long enough for
/// a concurrent `agent_delete` to remove the row the caller just validated.
fn parse_agent_id(input: &serde_json::Value, tool_name: &'static str) -> Result<Uuid> {
    let raw = super::util::require_str(input, "id", tool_name)?;
    Uuid::parse_str(raw.trim()).map_err(|_| MekaError::ToolExecution {
        tool_name: tool_name.to_string(),
        message: format!("'{raw}' is not a valid agent id"),
    })
}

/// Refuse an `id` argument that isn't a live child of the session running the tool.
///
/// One check, three failures it has to catch: an id that was never a sub-agent (a fabricated or
/// mistyped UUID), a sub-agent belonging to a *different* parent, and a fork holding ids it does
/// not own (`fork_session` copies the conversation, which names the children, but not the children
/// themselves, so a forked parent's log advertises sessions that are still linked to the original).
/// Letting any of those through would let one session drive or delete another's workers.
async fn require_child_session(
    params: &ToolBuilderParams,
    tool_name: &'static str,
    input: &serde_json::Value,
) -> Result<(Uuid, crate::store::SessionMetaRow)> {
    let agent_id = parse_agent_id(input, tool_name)?;
    let parent_sid = current_session_id(params, tool_name)?;
    let children = params
        .materials
        .store
        .load_session_tree(parent_sid)
        .await
        .map_err(|error| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("failed to list sub-agents: {error}"),
        })?;
    children
        .into_iter()
        .find(|row| row.id == agent_id && row.parent_id == Some(parent_sid))
        .map(|row| (parent_sid, row))
        .ok_or_else(|| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!(
                "no sub-agent '{agent_id}' belongs to this session. Use `agent_list` to see the ones that \
                 do."
            ),
        })
}

/// The session id of the agent running the tool. By the time a tool runs, `Agent::run_turn` has
/// written `shared_session_id`; a missing value means an agent ran a tool without first creating
/// its session, an internal invariant break worth surfacing rather than papering over.
fn current_session_id(params: &ToolBuilderParams, tool_name: &'static str) -> Result<Uuid> {
    params
        .cells
        .session_id
        .get()
        .ok_or_else(|| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: "session ID not yet assigned (run_turn invariant)".to_string(),
        })
}

/// Lists the sub-agents this session has spawned and can still follow up on.
pub(crate) struct AgentListTool {
    pub(crate) tool_builder_params: ToolBuilderParams,
}

/// `agent_list`'s schema. Free-standing for the reason [`agent_spawn_definition`] is.
pub(crate) fn agent_list_definition() -> ToolDefinition {
    ToolDefinition {
        name: "agent_list".to_string(),
        description: "List the sub-agents you have spawned in this session, with each one's \
                      id, working directory, turn count, and last activity. Pass an id to \
                      `agent_followup` to ask it another question, or to `agent_delete` to \
                      discard it and free what it held."
            .to_string(),
        parameters: serde_json::json!({ "type": "object", "properties": {} }),
        ..Default::default()
    }
}

#[async_trait]
impl Tool for AgentListTool {
    fn definition(&self) -> ToolDefinition {
        agent_list_definition()
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        _context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let session_id = current_session_id(&self.tool_builder_params, "agent_list")?;
        let rows = self
            .tool_builder_params
            .materials
            .store
            .load_session_tree(session_id)
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_list".to_string(),
                message: format!("failed to list sub-agents: {error}"),
            })?;

        // Direct children only. Grandchildren belong to the worker that spawned them and are
        // reachable through *its* `agent_list`; listing them here would advertise ids this session
        // cannot follow up on.
        let mut lines = Vec::new();
        for row in rows.iter().filter(|row| row.parent_id == Some(session_id)) {
            // Tool results are persisted as user-role messages too (`Agent::run_turn` wraps them in
            // one), so counting every user message would report a single task that took four tool
            // rounds as five turns. A real turn is a user message that carries something other than
            // tool results.
            let turns = self
                .tool_builder_params
                .materials
                .store
                .load_events(row.id)
                .await
                .map(|events| {
                    events
                        .iter()
                        .filter(|event| match event {
                            crate::conversation::Event::Append(message) => {
                                message.role == crate::conversation::Role::User
                                    && !message.content.iter().all(|block| {
                                        matches!(
                                            block,
                                            crate::conversation::ContentBlock::ToolResult { .. }
                                        )
                                    })
                            }
                            _ => false,
                        })
                        .count()
                })
                .unwrap_or(0);
            lines.push(format!(
                "{}\t{}\tturns={}\tlast_active={}",
                row.id,
                row.cwd
                    .as_deref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_default(),
                turns,
                row.updated_at,
            ));
        }

        if lines.is_empty() {
            return Ok(ToolOutput::text(
                "(no sub-agents spawned in this session)".to_string(),
                false,
            ));
        }
        Ok(ToolOutput::text(lines.join("\n"), false))
    }
}

/// Asks a sub-agent another question, on top of everything it already did.
pub(crate) struct AgentFollowupTool {
    /// The level of the agent holding this tool, read live at each call. A worker never runs above
    /// it, so switching the session down to `read` reaches workers spawned while it was at
    /// `unrestricted`.
    pub(crate) parent_permission: SharedPermission,
    pub(crate) tool_builder_params: ToolBuilderParams,
    /// Sub-agents currently running a follow-up, keyed by session id. Shared with every other
    /// `agent_followup` on this registry.
    pub(crate) in_flight: InFlightFollowups,
}

/// Sub-agent sessions with a follow-up in progress.
///
/// Parallel tool calls in one turn run concurrently, so two follow-ups on the same worker would
/// interleave: both hydrate the same event log, both append to it, and the second overwrites the
/// first's view of what happened. A `std::sync::Mutex` over a set is enough because every critical
/// section is a set insert or removal, never an await.
pub(crate) type InFlightFollowups = Arc<std::sync::Mutex<std::collections::HashSet<Uuid>>>;

/// Claims a sub-agent for the duration of one follow-up, releasing it on drop so an error or a
/// cancellation mid-turn cannot leave the worker permanently marked busy.
struct FollowupGuard {
    in_flight: InFlightFollowups,
    agent_id: Uuid,
}

impl FollowupGuard {
    fn claim(in_flight: &InFlightFollowups, agent_id: Uuid) -> Option<Self> {
        let mut guard = crate::sync::lock(in_flight);
        if !guard.insert(agent_id) {
            return None;
        }
        drop(guard);
        Some(Self {
            in_flight: Arc::clone(in_flight),
            agent_id,
        })
    }
}

impl Drop for FollowupGuard {
    fn drop(&mut self) {
        crate::sync::lock(&self.in_flight).remove(&self.agent_id);
    }
}

/// The spec a follow-up actually runs under: the recorded grant, narrowed by whatever the operator
/// has denied since.
///
/// Named rather than inlined at the call site because the narrowing is the security property and it
/// is not observable there: the stored spec is deliberately left alone, so the only assertion a
/// test could make against the call site is that the recording did not change, which is true
/// whether or not the narrowing happened.
///
/// The spec is a floor on restriction, never a license: config can only narrow it, and `..spec`
/// carries everything config has no opinion about.
fn combined_for_followup(
    spec: SubagentSpec,
    config_denials: &ToolDenials,
    memory_access: MemoryAccess,
) -> SubagentSpec {
    SubagentSpec {
        denied_servers: spec.denials().union(config_denials).server_list(),
        denied_tools: spec.denials().union(config_denials).tool_list(),
        memory: spec.memory.min(memory_access),
        ..spec
    }
}

/// `agent_followup`'s schema. Free-standing for the reason [`agent_spawn_definition`] is.
pub(crate) fn agent_followup_definition() -> ToolDefinition {
    ToolDefinition {
        name: "agent_followup".to_string(),
        description: "Ask a sub-agent you already spawned another question. It keeps its own \
                      conversation, so it still remembers what it found and can build on it \
                      rather than starting over from a summary. Returns its new report. The \
                      sub-agent runs under the terms it was spawned with (same permission \
                      level, same write boundary, same restrictions), which your current \
                      settings cannot widen. \
                      Get ids from `agent_spawn`'s result or from `agent_list`."
            .to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The sub-agent's id, as returned by `agent_spawn` or \
                                    `agent_list`."
                },
                "prompt": {
                    "type": "string",
                    "description": "The follow-up question or task."
                },
                "scratchpad": {
                    "type": "string",
                    "description": "If provided, save the sub-agent's new report to your \
                                    scratchpad under this name instead of returning it inline."
                }
            },
            "required": ["id", "prompt"]
        }),
        ..Default::default()
    }
}

#[async_trait]
impl Tool for AgentFollowupTool {
    fn definition(&self) -> ToolDefinition {
        agent_followup_definition()
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let cancellation = context.cancellation.clone();
        let prompt = super::util::require_str(&input, "prompt", "agent_followup")?;
        // Claimed before the ownership check, not after: the check awaits, and a concurrent
        // `agent_delete` finishing inside that window would leave this turn writing to a session
        // that no longer exists.
        let agent_id = parse_agent_id(&input, "agent_followup")?;
        let Some(_guard) = FollowupGuard::claim(&self.in_flight, agent_id) else {
            return Err(MekaError::ToolExecution {
                tool_name: "agent_followup".to_string(),
                message: format!(
                    "sub-agent '{agent_id}' is busy. Wait for the call already running against it to \
                     return before asking again."
                ),
            });
        };
        let (parent_sid, row) =
            require_child_session(&self.tool_builder_params, "agent_followup", &input).await?;
        debug_assert_eq!(row.id, agent_id);

        // The terms come off the session row, never from this agent's current state. See
        // `SubagentSpec`.
        let spec_json = self
            .tool_builder_params
            .materials
            .store
            .load_subagent_spec(agent_id)
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_followup".to_string(),
                message: format!("failed to load sub-agent spec: {error}"),
            })?
            .ok_or_else(|| MekaError::ToolExecution {
                tool_name: "agent_followup".to_string(),
                message: format!(
                    "sub-agent '{agent_id}' has no recorded spawn terms, so it cannot be resumed \
                     safely. Spawn a new one."
                ),
            })?;
        let spec: SubagentSpec =
            serde_json::from_str(&spec_json).map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_followup".to_string(),
                message: format!("sub-agent '{agent_id}' has an unreadable spec: {error}"),
            })?;
        // The spec records what the worker was granted; it is not a license to ignore what applies
        // now. Config may have gained deny lists since the spawn (most likely across the restart
        // an old worker had to survive to be here), and the memory grant is re-clamped against
        // what this agent currently holds, so a worker cannot outlive its granter's own limits.
        // Both combines take the more restrictive side, so neither can widen the worker.
        let spec = combined_for_followup(
            spec,
            &self.tool_builder_params.config_denials,
            self.tool_builder_params.memory_access,
        );

        let ceiling = self.parent_permission.get();
        // The worker's own cwd, as recorded when it was spawned, not the parent's current one: a
        // `/cd` between the spawn and the follow-up must not move a worker mid-task. A bounded
        // worker's directory and roots come off its spawn terms instead. Either way the directory
        // goes back through `admit_within_parent_reach` against the parent's reach *now*: a parent
        // that has moved to a directory that no longer contains it is refused rather than resuming
        // a writer it could not spawn today, and a bounded worker's parent that has dropped below
        // `workspace` likewise, the way `effective_permission` narrows a plain worker to what the
        // parent currently holds. One rule for both doors, or a plain worker keeps writing under
        // the very directory a bounded one is refused.
        let cells = &self.tool_builder_params.cells;
        let workspace = if spec.writable_roots.is_empty() {
            let cwd = row
                .cwd
                .as_deref()
                .map(PathBuf::from)
                .unwrap_or_else(|| cells.cwd.get());
            admit_within_parent_reach(ceiling, &cells.cwd, &cells.roots, &cwd).map_err(
                |error| MekaError::ToolExecution {
                    tool_name: "agent_followup".to_string(),
                    message: format!(
                        "sub-agent '{agent_id}' works in a directory this session can no longer \
                         grant: {error}"
                    ),
                },
            )?;
            WorkerWorkspace {
                cwd: SharedCwd::new(cwd),
                roots: cells.roots.clone(),
            }
        } else {
            let bounded =
                accept_writable_roots(ceiling, &cells.cwd, &cells.roots, &spec.writable_roots)
                    .map_err(|error| MekaError::ToolExecution {
                        tool_name: "agent_followup".to_string(),
                        message: format!(
                            "sub-agent '{agent_id}' was spawned with writable_roots this session \
                             can no longer grant: {error}"
                        ),
                    })?;
            WorkerWorkspace {
                cwd: SharedCwd::new(bounded.cwd),
                roots: SharedRoots::new(bounded.additional_roots),
            }
        };
        let sub_cwd_snapshot = workspace.cwd.get();
        let roots_snapshot = workspace.roots.get();

        let effective_permission = spec.effective_permission(ceiling);
        // `!=`, not `<`. The derived `Ord` is display order, which the enum doc says must not
        // decide authority. `greatest_within_both` never resolves above the recorded rung, so any
        // difference is a narrowing, and every one is worth saying out loud.
        if effective_permission != spec.permission {
            let recorded = spec.permission;
            tracing::info!(
                "sub-agent {agent_id} runs at `{effective_permission}` rather than its recorded \
                 `{recorded}`: this session has since been restricted"
            );
        }

        let sub_agent = build_subagent(
            &self.tool_builder_params,
            &spec,
            parent_sid,
            agent_id,
            workspace,
            "agent_followup",
            &context,
        )
        .await?;

        // Rehydrate the worker's own conversation: the same three calls the REPL's resume path
        // makes. `from_events` arms the resume notice, and it is left armed deliberately: every
        // follow-up really is a fresh registry, a fresh read tracker and an empty todo list, so the
        // worker is being told something true each time rather than a stale banner.
        let store = &self.tool_builder_params.materials.store;
        let mut events =
            store
                .load_events(agent_id)
                .await
                .map_err(|error| MekaError::ToolExecution {
                    tool_name: "agent_followup".to_string(),
                    message: format!("failed to load sub-agent conversation: {error}"),
                })?;
        // Bytes back into every image reference, as the root agent's hydration does.
        store
            .inline_blobs(&mut events)
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_followup".to_string(),
                message: format!("failed to load the sub-agent's images: {error}"),
            })?;
        let mut messages = Conversation::from_events(events);
        for dropped in messages.sanitize_orphans() {
            let count = dropped.content.len();
            tracing::warn!(
                "sub-agent {agent_id}: dropped an assistant message with {count} orphaned tool_use \
                 block(s) while rehydrating"
            );
        }

        let environment_context =
            build_environment_context(effective_permission, &sub_cwd_snapshot, &roots_snapshot);
        let augmented_prompt = format!("{environment_context}\n{prompt}");

        // Held for this turn, as `agent_spawn` holds it for the spawn. A follow-up runs a full turn
        // against a row nothing else claims, so without this the worker would sit unlocked for
        // seconds to minutes and a concurrent `meka session delete --all` could take it and cascade
        // the conversation away mid-run.
        //
        // A refusal here, where spawn only warns, and the asymmetry is the point: spawn's id is
        // brand new, so a failure can only be a filesystem problem, while this id already exists
        // and a refusal genuinely means somebody else is running a turn on this worker. Two turns
        // interleaved into one conversation is the thing the lock is for.
        let _worker_lock = self
            .tool_builder_params
            .materials
            .store
            .lock_session(agent_id)
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_followup".to_string(),
                message: format!("cannot follow up on sub-agent {agent_id}: {error}"),
            })?;

        // The row has to follow the build, for the reason `agent_spawn` writes it from the same
        // cell: `build_subagent` runs the worker on the parent's profile now, and a follow-up
        // after a `/profile` switch would otherwise bill an account the worker's row does not
        // name, so every reader (`meka session list`, `GET /v1/sessions`, `session export`, a
        // later resume) would disagree with what ran.
        //
        // And it has to follow the lock and the hydration, immediately ahead of the turn: written
        // earlier, a follow-up the lock then refused, or whose conversation could not be loaded,
        // would have moved the row onto a profile the worker never ran on.
        //
        // Read off the built agent rather than the live cell a second time. The cell is what
        // `build_subagent` consulted, but a repin landing in the gap (`/profile`, `PATCH
        // /v1/sessions/{id}`, ACP's `session/set_config_option`) would make the second read a
        // different answer, and the row would name a profile this turn is not running on.
        //
        // The row is the billing record, so a write that fails fails the call: the policy every
        // host applies to a profile through `record_session_change`, which a tool cannot reach.
        //
        // The level rides along for the reason `agent_spawn` writes it: the row answers for the
        // worker wherever a row is read, and `effective_permission` reads the spec, not the row, so
        // the row can follow the live answer without the recorded grant moving.
        let ran_on = sub_agent.profile();
        self.tool_builder_params
            .materials
            .store
            .update_session(agent_id, crate::store::SessionPatch {
                profile: Some(ran_on.clone()),
                permission: Some(effective_permission),
                ..Default::default()
            })
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_followup".to_string(),
                message: format!(
                    "failed to record that sub-agent {agent_id} runs on '{ran_on}': {error}"
                ),
            })?;

        sub_agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts(augmented_prompt, Vec::new()).map_err(
                    |empty| MekaError::ToolExecution {
                        tool_name: "agent_followup".to_string(),
                        message: empty.to_string(),
                    },
                )?,
                cancellation,
            )
            .await?;

        let report = messages
            .last_assistant_text()
            .unwrap_or_else(|| "(sub-agent produced no final text)".to_string());
        Ok(ToolOutput::text(report, false))
    }
}

/// Discards a sub-agent and everything it accumulated.
pub(crate) struct AgentDeleteTool {
    pub(crate) tool_builder_params: ToolBuilderParams,
    /// Shared with this registry's `agent_followup`, so the two cannot run on one worker at once.
    pub(crate) in_flight: InFlightFollowups,
}

/// `agent_delete`'s schema. Free-standing for the reason [`agent_spawn_definition`] is.
pub(crate) fn agent_delete_definition() -> ToolDefinition {
    ToolDefinition {
        name: "agent_delete".to_string(),
        description: "Delete a sub-agent you spawned, discarding its conversation, its \
                      scratchpad entries, and any sub-agents it spawned in turn. Use this once \
                      you have what you needed from a sub-agent, so a long session doesn't carry \
                      every sub-agent it ever ran. This removes only meka's own record of that \
                      sub-agent and its descendants: files it wrote to disk, and your own \
                      conversation and scratchpad, are untouched."
            .to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The sub-agent's id, as returned by `agent_spawn` or \
                                    `agent_list`."
                }
            },
            "required": ["id"]
        }),
        ..Default::default()
    }
}

#[async_trait]
impl Tool for AgentDeleteTool {
    fn definition(&self) -> ToolDefinition {
        agent_delete_definition()
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        // Claimed before the ownership check, mirroring `agent_followup`, so the two serialize on
        // one worker whichever arrives first. Parallel tool calls in a single turn make this
        // reachable: without it, a delete completing inside a follow-up's ownership check leaves
        // that follow-up writing to a deleted session and failing on a foreign-key violation, a
        // raw database error, on a path where the model did nothing wrong.
        let agent_id = parse_agent_id(&input, "agent_delete")?;
        let Some(_guard) = FollowupGuard::claim(&self.in_flight, agent_id) else {
            return Err(MekaError::ToolExecution {
                tool_name: "agent_delete".to_string(),
                message: format!(
                    "sub-agent '{agent_id}' is busy. Wait for the call already running against it to \
                     return before deleting it."
                ),
            });
        };
        let (_parent_sid, row) =
            require_child_session(&self.tool_builder_params, "agent_delete", &input).await?;
        // One statement; `sessions.parent_session_id`, `messages.session_id` and
        // `tool_outputs.session_id` all carry `ON DELETE CASCADE`, so the worker's messages, its
        // scratchpad entries and its own descendants go with it.
        self.tool_builder_params
            .materials
            .store
            .delete_session(row.id)
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_delete".to_string(),
                message: format!("failed to delete sub-agent: {error}"),
            })?;
        let id = row.id;
        tracing::info!("deleted sub-agent session {id}");
        Ok(ToolOutput::text(format!("deleted agent {}", row.id), false))
    }
}

/// The workspace a worker is built on, resolved by the door that builds it: the parent's own
/// handles for a worker that shares its parent's workspace, a directory and roots of its own for
/// one bounded by `writable_roots`. One value rather than two arguments because the pair is
/// decided together and a worker holding one door's directory with the other's roots is a
/// boundary nobody granted.
struct WorkerWorkspace {
    cwd: SharedCwd,
    roots: SharedRoots,
}

/// Build the worker described by `spec`: its tool registry, its inherited MCP toolset, its own
/// `agent_spawn` when the recursion budget allows, its system prompt, and the `Agent` over all of
/// it.
///
/// Shared by `agent_spawn` and `agent_followup` on purpose. The two differ only in where the spec
/// comes from (freshly built vs. read off the session row) and what conversation the agent is
/// handed (empty vs. rehydrated). Anything that drifted between two copies of this would be a
/// follow-up that quietly runs under different terms than the spawn did, which is the exact failure
/// the spec exists to prevent.
///
/// `params` supplies the *ambient* collaborators (provider, caches, store, frontend) and
/// `spec` supplies every restriction. `params.memory_access` is deliberately not read here: the
/// spec's copy is authoritative, because config may have changed since the spawn.
async fn build_subagent(
    params: &ToolBuilderParams,
    spec: &SubagentSpec,
    parent_session_id: Uuid,
    sub_session_id: Uuid,
    workspace: WorkerWorkspace,
    tool_name: &'static str,
    // The call that spawns the worker: its tool-use id, so the worker's own tool calls roll up
    // into that call's display, and the prompt it answers, so the worker's requests bill to it.
    call: &crate::tools::ToolContext,
) -> Result<Agent> {
    // The parent's live handle, not a snapshot of its level. Taking a `Permission` here is what
    // let a worker outlive its parent's downgrade: the value was read once and frozen into a fresh
    // atomic, so nothing the user did afterwards could reach the running child.
    let parent_permission = &params.cells.permission;
    // Read at spawn time, not at registration: the parent may have switched profile since this
    // tool was registered, and a worker runs on what its parent runs on *now*.
    let parent_profile = params.cells.profile.current();
    // A worker's window comes off `parent_profile` inside `Agent::new_subagent`, not from
    // `params.parent_options`, which was cloned when the session was assembled and cannot hear
    // about a switch. Taking the provider from one and the window from the other would have a
    // worker talk to a 32k profile while auto-compacting at 80% of the 1M one the session had left.
    let parent_options = params.parent_options.clone();
    let sub_shared_perm = spec.shared_permission_bounded(parent_permission);
    let effective_permission = spec.effective_permission(parent_permission.get());
    let denials = spec.denials();
    // Resolved once: it feeds both this worker's system prompt and what its own children can be
    // given. `None` here is what makes nesting self-enforcing: a worker that was not granted the
    // instructions has no copy to pass on, so the restriction propagates through the data rather
    // than through a check every future call site has to remember.
    let memory_access = spec.granted_memory();
    let granted_instructions = match spec.instructions {
        InstructionAccess::Inherit => params.parent_options.user_instructions.clone(),
        InstructionAccess::None => None,
    };
    // A fresh, private todo list so the worker's `todo` calls don't touch the parent's task
    // tracking. Not persisted, so a follow-up starts with an empty one; the resume notice the
    // rehydrated conversation carries is what tells the worker its tool state is gone.
    let sub_todo_list = crate::todo::SharedTodoList::default();
    let sub_shared_session_id = crate::session::SharedSessionId::new(Some(sub_session_id));
    // Wrap so permission prompts surface in the parent's UI while emits stay silent (the
    // sub-agent's output flows back as this tool's result, not as live notifications). The one
    // exception is the sub-agent's tool calls, which are rolled up into this call's own display
    // so a long run is not an opaque spinner, hence the tool-call id.
    let sub_frontend: Arc<dyn crate::frontend::Frontend> =
        Arc::new(crate::frontend::PermissionForwardingFrontend::new(
            Arc::clone(&params.cells.frontend),
            call.tool_call_id.clone(),
        ));
    // The worker's own cells: its clamped permission, the directory and roots it was handed, a
    // session of its own, fresh gauges, and a profile seeded from what the parent runs on now. A
    // worker has no prompt gauge and no session entry watching it, and it never switches, so
    // nothing outside needs the handle. The write fence and the shell sandbox are both built from
    // `roots` by `register_core_tools`, so a bounded worker's boundary is what these cells say.
    let sub_cells = crate::session::SessionCells {
        permission: sub_shared_perm.clone(),
        cwd: workspace.cwd,
        roots: workspace.roots,
        session_id: sub_shared_session_id,
        todo_list: sub_todo_list,
        profile: crate::provider::PublishedProfile::detached(&parent_profile),
        context_tokens: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        context_overhead: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        background_tasks: crate::background::BackgroundTasks::default(),
        frontend: sub_frontend,
        // Fresh per worker for the same reason the id is: a lock belongs to the session that
        // was created, and every agent creates at most its own.
        session_lock: crate::store::SessionLockSlot::default(),
        pending_compaction: crate::session::PendingCompaction::default(),
    };

    let sub_registry = ToolRegistry::build_for_subagent(
        &params.materials,
        &sub_cells,
        super::registry::RegistryScope {
            denials: denials.clone(),
            memory_access,
            parent_session_id: if spec.inherited_scratchpad.is_empty() {
                None
            } else {
                Some(parent_session_id)
            },
            inherited_scratchpad_names: spec.inherited_scratchpad.clone(),
        },
    )
    .map_err(|error| MekaError::ToolExecution {
        tool_name: tool_name.to_string(),
        message: format!("failed to build sub-agent tool registry: {error}"),
    })?;

    // Inherit the parent's MCP toolset, minus anything the spec denies (the install reads the
    // denials back off the registry). Skipped silently when no MCP manager is attached (no servers
    // configured) or when the parent's servers are still Pending / Failed. Non-spawning and
    // idempotent.
    if let Some(weak) = params.materials.mcp_manager.as_ref() {
        // Upgrade only if the manager is still alive. If the parent's `meka acp` process is
        // mid-shutdown, the Arc may already be gone. Skip silently.
        if let Some(manager) = weak.upgrade() {
            super::mcp_adapter::install_on_worker_registry(&manager, &sub_registry).await;
        }
    }

    // Grant the sub-agent its own `agent_spawn` when its recursion budget allows, so it can
    // orchestrate a team of its own. Registered here (before the tool catalog is snapshotted for
    // the system prompt below) and outside `build_for_subagent`, mirroring the root registration in
    // `assemble_agent`. Two counters bound nesting: `remaining_depth` is the soft,
    // `max_depth`-tunable budget; `absolute_depth` is the hard cap that guarantees termination.
    // The three lifecycle tools ride the same gate: a worker that cannot spawn has no children to
    // list, follow up on, or delete.
    let allow_nested_spawn =
        spec.remaining_depth >= 1 && spec.absolute_depth < SUBAGENT_ABSOLUTE_MAX_DEPTH;
    if allow_nested_spawn {
        let child_params = ToolBuilderParams {
            materials: params.materials.clone(),
            cells: crate::session::SessionCells {
                // The *parent's* handle, not a snapshot of what it currently holds: a switch the
                // user makes later must reach the whole subtree's future spawns, not just its
                // first level.
                profile: params.cells.profile.clone(),
                ..sub_cells.clone()
            },
            // The worker's own granted level, not its parent's. `params.memory_access` is what the
            // spawning agent holds, so letting the spread supply it would let a worker grant its
            // children up to its parent's level rather than its own, reaching through a child
            // what it was denied directly. The deny lists come from `spec` for the same reason.
            memory_access,
            config_denials: params.config_denials.clone(),
            // Both ceilings for whatever this worker spawns in turn. `parent_options` carries the
            // instruction text, so overwriting it with the grant is what stops a worker handing a
            // grandchild something it was not given itself.
            parent_options: AgentOptions {
                user_instructions: granted_instructions.clone(),
                ..parent_options.clone()
            },
        };
        register_subagent_tools(&sub_registry, AgentSpawnTool {
            // The worker's own (already clamped) permission is the ceiling for anything it spawns,
            // so a downgrade the parent made reaches the whole subtree.
            parent_permission: sub_shared_perm.clone(),
            tool_builder_params: child_params,
            inherited_denials: denials,
            remaining_depth: spec.remaining_depth,
            absolute_depth: spec.absolute_depth,
        })?;
    }

    // Build the system prompt against the fully-loaded registry (which now includes MCP adapters).
    // The override on `AgentOptions` is static, so this single build captures the whole catalog
    // the sub-agent can see.
    let tools = sub_registry
        .definitions_for_permission(effective_permission, parent_permission.approvals());
    // Gated on the registry rather than on the spec alone: `[memory] enabled = false` or a
    // `[tools]` filter can leave a granted worker without `memory_read`, and an index describing
    // memories it has no tool to open is pure cost. This mirrors how the root agent's index is
    // gated on the same tool being in its catalog.
    let memory_index = if sub_registry.get("memory_read").is_some() {
        render_subagent_memory_index(&params.materials.memories.index().await.unwrap_or_else(
            |error| {
                // Search still works, and the worker is told what it has. A store that cannot be
                // read is not a reason to refuse to spawn.
                tracing::warn!("failed to read the memory index for a sub-agent: {error}");
                Vec::new()
            },
        ))
    } else {
        String::new()
    };
    let sub_system_prompt = build_subagent_system_prompt(
        effective_permission,
        &tools,
        &spec.inherited_scratchpad,
        granted_instructions.as_deref(),
        &memory_index,
    );

    Ok(Agent::new_subagent(
        &params.materials,
        sub_cells,
        sub_registry,
        &parent_options,
        sub_system_prompt,
        call.prompt_id,
    ))
}

/// Read an optional string-valued parameter, refusing a value of the wrong type.
///
/// The obvious `input[key].as_str()` treats a non-string as absent, which for a restriction is the
/// wrong way to fail: `permission: 0` would silently run the worker at the parent's own level
/// rather than the restricted one the caller was reaching for. Absent and explicitly-null still
/// mean "not specified"; anything else that is not a string is an error naming the parameter.
fn optional_str<'a>(
    input: &'a serde_json::Value,
    key: &str,
    tool_name: &'static str,
) -> Result<Option<&'a str>> {
    match input.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(text)) => {
            let trimmed = text.trim();
            Ok(if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            })
        }
        Some(other) => Err(MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!(
                "'{key}' must be a string, got {other}. Leave it out to use the default."
            ),
        }),
    }
}

/// Pull a `Vec<String>` out of an optional array parameter. A missing or null value yields an
/// empty list, and non-string entries inside an array are skipped so a partially malformed array
/// does not sink the whole spawn, but anything that is not an array is an error: read as absent, a
/// bare string in `deny_tools` would spawn an unrestricted worker in place of the restricted one
/// the caller asked for, with nothing to say so. The same guard `optional_str` gives the strings.
fn string_array(
    input: &serde_json::Value,
    key: &str,
    tool_name: &'static str,
) -> Result<Vec<String>> {
    match input.get(key) {
        None | Some(serde_json::Value::Null) => Ok(Vec::new()),
        Some(serde_json::Value::Array(items)) => Ok(items
            .iter()
            .filter_map(|item| item.as_str().map(str::trim))
            .filter(|text| !text.is_empty())
            .map(str::to_string)
            .collect()),
        Some(other) => Err(MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!(
                "'{}' must be an array of strings, got {}. Leave it out to restrict nothing.",
                key,
                json_type_name(other)
            ),
        }),
    }
}

/// An unsigned integer parameter, or an error if the model sent something else: `"0"` read as
/// absent would give a worker the nesting budget its caller had just tried to deny it.
fn optional_u64(
    input: &serde_json::Value,
    key: &str,
    tool_name: &'static str,
) -> Result<Option<u64>> {
    match input.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Number(number)) if number.as_u64().is_some() => Ok(number.as_u64()),
        Some(other) => Err(MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!(
                "'{}' must be a non-negative integer, got {}. Leave it out to use the default.",
                key,
                json_type_name(other)
            ),
        }),
    }
}

fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// Parse the caller's optional `permission` string. An unrecognized string is a hard error, not a
/// silent fallback.
fn parse_subagent_permission(requested: Option<&str>) -> Result<Option<Permission>> {
    requested
        .map(|text| {
            text.parse::<Permission>()
                .map_err(|message| MekaError::ToolExecution {
                    tool_name: "agent_spawn".to_string(),
                    message,
                })
        })
        .transpose()
}

/// Clamp a requested level to the parent's as a ceiling. `None` keeps the parent's level (inherit
/// verbatim). A sub-agent can only ever run at an equal-or-more-restricted level than its parent
/// (`min` over the discriminant order `None < Read < Workspace < Unrestricted`), so a parent turn
/// can hand risky work to a locked-down sub-agent but can never escalate one.
fn resolve_subagent_permission(requested: Option<Permission>, parent: Permission) -> Permission {
    match requested {
        // `clamp_to` rather than a bare comparison, so the door reads as the question it asks.
        // `SharedPermission::with_ceiling` flattens a grandchild's ceiling to the *root* cell on
        // the precondition that this clamp already folded the intermediate level in.
        Some(requested) => requested.clamp_to(parent),
        None => parent,
    }
}

/// Restrict an `EnabledPermissions` set to the levels *contained by* `ceiling`, through
/// [`Permission::is_within`] so this door asks the same question every other bound does.
///
/// Defense-in-depth for the permission clamp: sub-agents have no runtime permission-switch path
/// today, so their initial level is what governs, but bounding the enabled set means any future
/// switch path cannot climb a sub-agent back past the ceiling its parent set. Falls back to a
/// singleton `{ceiling}` set when the intersection is empty: a parent that enabled only
/// `unrestricted` and hands down a `read` ceiling leaves nothing behind.
fn clamp_enabled_permissions(
    enabled: EnabledPermissions,
    ceiling: Permission,
) -> EnabledPermissions {
    EnabledPermissions::from_levels(enabled.iter().filter(|level| level.is_within(ceiling)))
        .unwrap_or_else(|| {
            EnabledPermissions::from_levels([ceiling]).unwrap_or(EnabledPermissions::DEFAULT)
        })
}

/// Compute the recursion budget for a sub-agent one level below a `AgentSpawnTool` with the given
/// `remaining_depth` / `absolute_depth`, honoring an optional `max_depth` override.
///
/// Returns `(child_remaining, child_absolute, allow_nested)`. `remaining_depth` is the budget
/// seeded from `session.subagent_max_depth`; `max_depth` may lower it but never raise it.
/// `absolute_depth` is the monotonic hard counter: it always increments and, once it reaches
/// [`SUBAGENT_ABSOLUTE_MAX_DEPTH`], no further `agent_spawn` is granted. A nested `agent_spawn` is
/// granted only when both budgets allow it.
///
/// The override is clamped because `[session] subagent_max_depth` is documented as a ceiling
/// ("`subagent_max_depth = 1` means sub-agents cannot spawn further sub-agents"), and an
/// unclamped override would make it merely a default that one `agent_spawn` passing `max_depth:
/// 15` re-grants at every level. Recursion is still bounded by [`SUBAGENT_ABSOLUTE_MAX_DEPTH`], so
/// this is about the config key meaning what it says rather than about termination.
fn child_spawn_depth(
    remaining_depth: usize,
    absolute_depth: usize,
    max_depth_override: Option<usize>,
) -> (usize, usize, bool) {
    let inherited = remaining_depth.saturating_sub(1);
    let child_remaining = match max_depth_override {
        Some(requested) => requested.min(inherited),
        None => inherited,
    };
    let child_absolute = absolute_depth + 1;
    let allow_nested = child_remaining >= 1 && child_absolute < SUBAGENT_ABSOLUTE_MAX_DEPTH;
    (child_remaining, child_absolute, allow_nested)
}

/// Compose the sub-agent's first-turn task from an optional parent directive and an optional
/// rendered skill body. Mirrors the CLI's `--skill` ordering (`host::build_skill_prompt`): the
/// parent directive comes first, the skill body second. Returns `None` only
/// when both inputs are absent; the caller treats that as an error.
fn compose_subagent_task(prompt: Option<&str>, skill_body: Option<&str>) -> Option<String> {
    match (prompt, skill_body) {
        (Some(prompt), Some(body)) => Some(format!("{prompt}\n\n{body}")),
        (Some(prompt), None) => Some(prompt.to_string()),
        (None, Some(body)) => Some(body.to_string()),
        (None, None) => None,
    }
}

/// Ceiling on the memory index handed to a granted worker. Smaller than the root agent's 8 KiB
/// budget on purpose: a worker was spawned for one task, and the store is background for it rather
/// than the running context it is for the agent that owns the session.
const SUBAGENT_MEMORY_INDEX_MAX_BYTES: usize = 4_096;

/// Render the memory index for a worker granted `memory: "read"`.
///
/// Separate from the root agent's `[Memory]` section rather than shared with it, for two
/// reasons. A sub-agent's system prompt is a static override, so it never receives the per-turn
/// world state the parent's index rides in. And the parent's header tells the reader to call
/// `memory_write` when it learns something durable, which a worker cannot do: pointing it at a
/// tool it does not have is how a model burns a turn discovering the tool is missing.
fn render_subagent_memory_index(memories: &[crate::memory::Memory]) -> String {
    if memories.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "## Memory\n\nDurable notes the agent that spawned you has saved, most important first. \
         Call `memory_read` with a name to load one in full, or `memory_search` to search across \
         all of them. You cannot add to or change this store: if you learn something worth \
         keeping, say so in your report and let the agent that spawned you decide.\n\n",
    );
    let mut shown = 0;
    for memory in memories {
        // Sanitized at the boundary, like the root agent's index: the store hands back stored
        // bytes, and this is a worker's context. Elided too, as the parent's index and both search
        // renderers do: descriptions are unbounded at the write door, so a single 4,000-character
        // one at the top of the store would exceed the whole budget on its own.
        let line = format!(
            "- **{}**: {}\n",
            memory.name,
            crate::entry::elide_description_for_index(
                &crate::memory::render_description_for_model(&memory.description)
            )
        );
        // Always emit the first, for the same reason `render_hits` does. The elide above is what
        // actually makes a collapse to zero entries unreachable (`MAX_DESCRIPTION_CHARS` bounds
        // one line far below this budget), so this branch is the belt to that brace, and holds if
        // that bound ever moves. It is deliberately not something the tests can distinguish.
        if shown > 0 && out.len() + line.len() > SUBAGENT_MEMORY_INDEX_MAX_BYTES {
            break;
        }
        out.push_str(&line);
        shown += 1;
    }
    // Same reason the parent states its remainder: a silently truncated index reads as "this is
    // everything", which is what turns a full store into a confidently incomplete answer.
    let remaining = memories.len() - shown;
    if remaining > 0 {
        out.push_str(&format!(
            "\n{remaining} more not listed here. Use `memory_search` to reach them.\n"
        ));
    }
    out.push('\n');
    out
}

/// The sub-agent's system prompt.
///
/// `user_instructions` is `None` unless the `agent_spawn` call asked for them. Instructions are
/// installation-wide and describe the root agent: its persona, how it should address the user,
/// what it should volunteer. A worker handed a task by another agent is not that agent, and
/// inheriting the persona unasked is how a sub-agent ends up talking to the user as though it were
/// the one they are speaking to.
///
/// They remain *grantable* because they are also where project conventions live, and a parent that
/// judges a task needs the standing rules can hand them over verbatim rather than paraphrasing them
/// into the prompt.
fn build_subagent_system_prompt(
    permission: Permission,
    tools: &[ToolDefinition],
    inherited_scratchpad: &[String],
    user_instructions: Option<&str>,
    memory_index: &str,
) -> String {
    let mut prompt = String::new();
    prompt.push_str(
        "You are a research sub-agent. Complete the assigned task using the \
         available tools, then produce a concise final report summarizing your \
         findings. Do not ask follow-up questions. Work with what you have. \
         For multi-step work, use the `todo` tool to plan and track progress: \
         pass `items` together with a `title` to (re)write the list, `set` to \
         update statuses by task number, and call `todo` with no arguments to \
         read the current list. Your todo list is private to this sub-agent.\n\n",
    );

    prompt.push_str(&format!("## Permission Level: {permission}\n\n"));

    if let Some(instructions) = user_instructions
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        prompt.push_str("## User Instructions\n\n");
        prompt.push_str(
            "These are installation-specific rules, handed to you by the agent that spawned you. \
             Treat them as hard constraints unless they conflict with safety requirements. They \
             describe that agent's own conduct, so where they concern how to address the user or \
             what to volunteer, they are context rather than instructions to you: your output goes \
             back to that agent as a report, not to the user.\n\n",
        );
        prompt.push_str(instructions);
        prompt.push_str("\n\n");
    }

    prompt.push_str(memory_index);

    if !inherited_scratchpad.is_empty() {
        prompt.push_str("## Inherited Scratchpad Entries\n\n");
        prompt.push_str(
            "Your parent agent has granted you read-only access to the following \
             scratchpad entries from its own session. Use `scratchpad_read` with \
             the exact names below to load them on demand. Do not assume their \
             contents without reading. `scratchpad_write`, `_edit`, and `_delete` \
             against these names will return an error; if you need to derive new \
             state, save it under a different name (e.g. `<name>_local`).\n\n",
        );
        for name in inherited_scratchpad {
            prompt.push_str(&format!("- {name}\n"));
        }
        prompt.push('\n');
    }

    if !tools.is_empty() {
        prompt.push_str("## Available Tools\n\n");
        for tool in tools {
            prompt.push_str(&format!("- **{}**: {}\n", tool.name, tool.description));
        }
        prompt.push('\n');
    }

    prompt
}

#[cfg(test)]
mod tests {
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::{
        provider::{Provider, mock::text_round},
        store::Store,
    };

    #[test]
    fn subagent_system_prompt_reflects_inherited_permission() {
        let prompt = build_subagent_system_prompt(Permission::Unrestricted, &[], &[], None, "");
        assert!(
            prompt.contains(&format!(
                "## Permission Level: {}",
                Permission::Unrestricted
            )),
            "expected Write level in prompt, got: {prompt}"
        );

        let read_prompt = build_subagent_system_prompt(Permission::Read, &[], &[], None, "");
        assert!(read_prompt.contains(&format!("## Permission Level: {}", Permission::Read)));
    }

    /// The installation's instructions describe the root agent, not a worker one of its turns
    /// handed a task to. A sub-agent that inherited them would answer the user in the leader's
    /// voice, under rules written for a conversation it is not part of.
    #[test]
    fn subagent_system_prompt_carries_no_user_instructions() {
        let prompt = build_subagent_system_prompt(Permission::Unrestricted, &[], &[], None, "");
        assert!(!prompt.contains("User Instructions"));
        assert!(!prompt.contains("installation-specific"));
    }

    #[test]
    fn subagent_system_prompt_mentions_todo_tools() {
        let prompt = build_subagent_system_prompt(Permission::Read, &[], &[], None, "");
        assert!(
            prompt.contains("`todo` tool"),
            "expected todo tool mention in prompt, got: {prompt}"
        );
    }

    #[test]
    fn subagent_system_prompt_omits_inheritance_section_when_empty() {
        let prompt = build_subagent_system_prompt(Permission::Read, &[], &[], None, "");
        assert!(
            !prompt.contains("Inherited Scratchpad"),
            "no inherited section expected for empty allowlist, got: {prompt}"
        );
    }

    #[test]
    fn subagent_system_prompt_lists_inherited_names() {
        let names = vec!["captured_output".to_string(), "research_notes".to_string()];
        let prompt = build_subagent_system_prompt(Permission::Read, &[], &names, None, "");
        assert!(prompt.contains("## Inherited Scratchpad Entries"));
        assert!(prompt.contains("- captured_output"));
        assert!(prompt.contains("- research_notes"));
        assert!(prompt.contains("scratchpad_read"));
    }

    #[test]
    fn subagent_system_prompt_warns_inherited_writes_will_error() {
        let names = vec!["build_log".to_string()];
        let prompt = build_subagent_system_prompt(Permission::Read, &[], &names, None, "");
        assert!(
            prompt.contains("will return an error"),
            "expected write-rejection wording, got: {prompt}",
        );
        assert!(
            prompt.contains("_local"),
            "expected naming suggestion, got: {prompt}",
        );
    }

    #[test]
    fn compose_subagent_task_combinations() {
        assert_eq!(
            compose_subagent_task(Some("focus on UK news"), Some("skill body")),
            Some("focus on UK news\n\nskill body".to_string()),
            "parent directive must come first, skill body second",
        );
        assert_eq!(
            compose_subagent_task(Some("just a prompt"), None),
            Some("just a prompt".to_string()),
        );
        assert_eq!(
            compose_subagent_task(None, Some("skill body")),
            Some("skill body".to_string()),
        );
        assert_eq!(compose_subagent_task(None, None), None);
    }

    #[test]
    fn resolve_subagent_permission_inherits_when_absent() {
        assert_eq!(
            resolve_subagent_permission(None, Permission::Unrestricted),
            Permission::Unrestricted
        );
        assert_eq!(
            resolve_subagent_permission(None, Permission::Read),
            Permission::Read
        );
    }

    /// `agent_followup` resolves a stored grant with the meet, not the spawn clamp.
    ///
    /// Replaying a recorded grant narrows it to the parent's current level and never widens it.
    #[test]
    fn a_followup_resolves_a_stored_grant_without_widening_it() {
        let spec = SubagentSpec {
            permission: Permission::Workspace,
            enabled_permissions: Vec::new(),
            denied_servers: Vec::new(),
            denied_tools: Vec::new(),
            memory: MemoryAccess::None,
            instructions: InstructionAccess::None,
            inherited_scratchpad: Vec::new(),
            remaining_depth: 2,
            absolute_depth: 1,
            writable_roots: Vec::new(),
        };

        assert_eq!(
            spec.effective_permission(Permission::Unrestricted),
            Permission::Workspace,
            "a parent that still holds the recorded level replays it unchanged"
        );
        assert_eq!(
            spec.effective_permission(Permission::Read),
            Permission::Read,
            "and a parent that has dropped below it narrows the worker with it"
        );
    }

    /// A spec carrying `writable_roots` never resolves above `workspace`, whatever level it
    /// claims: `agent_spawn` refuses the pair, but `meka session import` writes a spec verbatim,
    /// and a boundary on a worker with no boundary would bound nothing.
    #[test]
    fn a_bounded_spec_never_resolves_above_workspace() {
        let spec = SubagentSpec {
            permission: Permission::Unrestricted,
            writable_roots: vec![PathBuf::from("/work/sub")],
            ..spec_for_test(Permission::Unrestricted)
        };
        assert_eq!(
            spec.effective_permission(Permission::Unrestricted),
            Permission::Workspace
        );
        assert_eq!(
            spec.effective_permission(Permission::Read),
            Permission::Read,
            "and the parent's ceiling still applies beneath the cap"
        );
    }

    /// The bound holds at the depth where nothing downstream re-clamps it.
    ///
    /// At depth 1 a miss is invisible: `SharedPermission::get` re-clamps against the parent. At
    /// depth 2 it is not, because `with_ceiling` flattens a grandchild's ceiling to the *root*
    /// cell on the stated precondition that the spawn-time clamp already folded the intermediate
    /// level in. A `read` child under an `unrestricted` root must therefore not be able to spawn
    /// a `workspace` grandchild.
    ///
    /// **What this does not cover**, and the honest limit of the guarantee: the root cell here
    /// never moves. `with_ceiling`'s precondition is taken once, at spawn, so cycling the root
    /// `unrestricted -> read -> unrestricted` between two spawns leaves a grandchild bound to a
    /// root that has changed shape underneath it. Nothing exceeds the root, which is the human's
    /// own level, so the headline invariant holds and the next `agent_followup` re-clamps it, but
    /// the direct-parent bound does not hold across that sequence, and no test asserts it does.
    #[test]
    fn a_grandchild_cannot_escape_an_intermediate_parent() {
        assert_eq!(
            resolve_subagent_permission(Some(Permission::Workspace), Permission::Read),
            Permission::Read,
            "a `workspace` request under a `read` parent must resolve to the parent's own level"
        );

        // The full chain, through the handles production actually builds.
        fn spec_at(permission: Permission, parent_enabled: EnabledPermissions) -> SubagentSpec {
            SubagentSpec {
                permission,
                enabled_permissions: clamp_enabled_permissions(parent_enabled, permission)
                    .iter()
                    .collect(),
                denied_servers: Vec::new(),
                denied_tools: Vec::new(),
                memory: MemoryAccess::None,
                instructions: InstructionAccess::None,
                inherited_scratchpad: Vec::new(),
                remaining_depth: 2,
                absolute_depth: 1,
                writable_roots: Vec::new(),
            }
        }

        let root = SharedPermission::new(Permission::Unrestricted, EnabledPermissions::ALL);
        let child_spec = spec_at(
            resolve_subagent_permission(Some(Permission::Read), root.get()),
            root.enabled(),
        );
        let child = child_spec.shared_permission_bounded(&root);
        assert_eq!(
            child.get(),
            Permission::Read,
            "child holds what it asked for"
        );

        let grandchild_spec = spec_at(
            resolve_subagent_permission(Some(Permission::Workspace), child.get()),
            child.enabled(),
        );
        let grandchild = grandchild_spec.shared_permission_bounded(&child);
        assert_eq!(
            grandchild.get(),
            Permission::Read,
            "a grandchild must not reach `workspace` past a `read` parent, however the ceiling \
             cell is flattened"
        );
        assert!(
            !grandchild.enabled().is_enabled(Permission::Workspace),
            "nor may `workspace` remain switchable in its enabled set"
        );
    }

    #[test]
    fn resolve_subagent_permission_clamps_to_parent_ceiling() {
        // Requesting a higher level than the parent is clamped down: a sub-agent can never be
        // escalated above its parent.
        assert_eq!(
            resolve_subagent_permission(Some(Permission::Unrestricted), Permission::Read),
            Permission::Read
        );
        assert_eq!(
            resolve_subagent_permission(Some(Permission::Workspace), Permission::Read),
            Permission::Read
        );
        // Requesting a lower level restricts the sub-agent below the parent.
        assert_eq!(
            resolve_subagent_permission(Some(Permission::Read), Permission::Unrestricted),
            Permission::Read
        );
        assert_eq!(
            resolve_subagent_permission(Some(Permission::None), Permission::Unrestricted),
            Permission::None
        );
    }

    #[test]
    fn parse_subagent_permission_rejects_invalid() {
        assert!(parse_subagent_permission(Some("admin")).is_err());
        assert_eq!(parse_subagent_permission(None).unwrap(), None);
        assert_eq!(
            parse_subagent_permission(Some("read")).unwrap(),
            Some(Permission::Read)
        );
    }

    /// A restriction passed with the wrong type is refused, not read as absent. Reading it as
    /// absent is the dangerous direction for `permission`, where "not specified" means "inherit the
    /// parent's level", so a malformed restriction would hand the worker more than the caller was
    /// reaching for.
    #[test]
    fn optional_str_refuses_a_wrong_typed_value() {
        let input = serde_json::json!({
            "permission": 0,
            "memory": true,
            "good": "read",
            "blank": "   ",
            "explicit_null": null,
        });
        for key in ["permission", "memory"] {
            let error = optional_str(&input, key, "agent_spawn")
                .expect_err("a non-string restriction must be refused");
            assert!(error.to_string().contains(key), "{error}");
            assert!(error.to_string().contains("must be a string"), "{error}");
        }
        assert_eq!(
            optional_str(&input, "good", "agent_spawn").expect("string"),
            Some("read")
        );
        // Absent, null, and whitespace all mean "not specified", which is what lets the defaults
        // apply without the caller having to say so.
        for key in ["missing", "blank", "explicit_null"] {
            assert_eq!(optional_str(&input, key, "agent_spawn").expect("ok"), None);
        }
    }

    #[test]
    fn string_array_skips_non_strings_and_blanks() {
        let input = serde_json::json!({
            "deny_servers": ["notion", 7, "  ", "  linear  ", null],
        });
        assert_eq!(
            string_array(&input, "deny_servers", "agent_spawn").expect("a list"),
            vec!["notion".to_string(), "linear".to_string()],
        );
        assert!(
            string_array(&input, "absent", "agent_spawn")
                .expect("absent is fine")
                .is_empty()
        );
        // A non-array value is not a one-element list. A bare string is a refusal, not an empty
        // list: read as empty, `deny_tools: "write_file"` would spawn an unrestricted worker.
        assert!(string_array(&serde_json::json!({"x": "notion"}), "x", "agent_spawn").is_err());
        assert!(
            optional_u64(
                &serde_json::json!({"max_depth": "0"}),
                "max_depth",
                "agent_spawn"
            )
            .is_err(),
            "a quoted zero is not a depth"
        );
        assert_eq!(
            optional_u64(
                &serde_json::json!({"max_depth": 0}),
                "max_depth",
                "agent_spawn"
            )
            .expect("an integer"),
            Some(0)
        );
    }

    #[test]
    fn clamp_enabled_permissions_drops_higher_levels() {
        let clamped = clamp_enabled_permissions(EnabledPermissions::ALL, Permission::Read);
        assert!(clamped.is_enabled(Permission::None));
        assert!(clamped.is_enabled(Permission::Read));
        assert!(!clamped.is_enabled(Permission::Workspace));
        assert!(!clamped.is_enabled(Permission::Unrestricted));
    }

    #[test]
    fn clamp_enabled_permissions_falls_back_to_singleton() {
        // Parent enabled only Write; clamping to Read leaves an empty intersection, so we fall back
        // to a singleton set of the ceiling itself rather than an invalid empty set.
        let only_write = EnabledPermissions::from_levels([Permission::Unrestricted]).unwrap();
        let clamped = clamp_enabled_permissions(only_write, Permission::Read);
        assert!(clamped.is_enabled(Permission::Read));
        assert!(!clamped.is_enabled(Permission::Unrestricted));
    }

    #[test]
    fn child_spawn_depth_natural_decrement() {
        let (remaining, absolute, allow) = child_spawn_depth(3, 0, None);
        assert_eq!(remaining, 2);
        assert_eq!(absolute, 1);
        assert!(allow);
    }

    #[test]
    fn child_spawn_depth_leaf_when_budget_exhausted() {
        // remaining_depth = 1 is "root spawns, sub-agents cannot": the child gets 0 and is not
        // granted a nested agent_spawn.
        let (remaining, _absolute, allow) = child_spawn_depth(1, 0, None);
        assert_eq!(remaining, 0);
        assert!(!allow);
    }

    /// `max_depth` narrows the child's budget and can never widen it. `[session]
    /// subagent_max_depth` is documented as a ceiling, so a model asking for more than the operator
    /// allowed gets the operator's answer.
    #[test]
    fn child_spawn_depth_override_only_narrows() {
        // Asking for more than is left yields what is left, not what was asked for.
        let (remaining, _absolute, allow) = child_spawn_depth(3, 0, Some(9));
        assert_eq!(remaining, 2);
        assert!(allow);

        // At the documented `subagent_max_depth = 1`, no override can grant a grandchild.
        let (remaining, _absolute, allow) = child_spawn_depth(1, 0, Some(5));
        assert_eq!(remaining, 0);
        assert!(!allow, "subagent_max_depth = 1 must forbid nesting");

        // Asking for less than is left is honored: the agent may still restrict itself.
        let (remaining, _absolute, _allow) = child_spawn_depth(5, 0, Some(1));
        assert_eq!(remaining, 1);

        // max_depth = 0 explicitly forbids the sub-agent from spawning further.
        let (_remaining, _absolute, allow_zero) = child_spawn_depth(3, 0, Some(0));
        assert!(!allow_zero);
    }

    #[test]
    fn child_spawn_depth_absolute_cap_forces_leaf() {
        // Even with a large soft budget, the monotonic absolute counter stops recursion at the cap.
        let (_remaining, absolute, allow) =
            child_spawn_depth(100, SUBAGENT_ABSOLUTE_MAX_DEPTH - 1, Some(100));
        assert_eq!(absolute, SUBAGENT_ABSOLUTE_MAX_DEPTH);
        assert!(!allow);
    }

    async fn store_for_test() -> Store {
        Store::for_test().await
    }

    // (Permission gating and "Unknown tool" fold-into-ToolOutput semantics belong to the shared
    // `Agent::run_turn` tool-dispatch path, covered by the `src/agent.rs` and `src/tools.rs`
    // suites.)

    #[tokio::test]
    async fn subagent_registry_has_independent_todo_list() {
        use crate::{
            config::BuiltinToolFilter,
            sandbox::{BackendProbe, SandboxCapability},
        };

        let parent_list = crate::todo::SharedTodoList::default();
        let sub_list = crate::todo::SharedTodoList::default();

        let sub_registry = ToolRegistry::build_for_subagent(
            &crate::session::SessionMaterials {
                core: crate::session::CoreMaterials {
                    web_client: crate::config::WebClientConfig::default(),
                    sandbox_enabled: true,
                    sandbox_capability: SandboxCapability::Unavailable,
                    sandbox_backend: crate::config::SandboxBackend::Landlock,
                    backend_probe: BackendProbe::Missing {
                        reason: "test fixture".to_string(),
                    },
                    builtin_filter: BuiltinToolFilter::default(),
                    write_locks: crate::workspace::WriteLocks::default(),
                },
                skills: crate::skills::SkillCache::for_root(None),
                memories: crate::store::memory::MemoryStore::detached(),
                ..crate::session::SessionMaterials::for_test(store_for_test().await)
            },
            &crate::session::SessionCells {
                session_id: crate::session::SharedSessionId::default(),
                todo_list: sub_list.clone(),
                ..crate::session::SessionCells::for_test(
                    SharedPermission::new(
                        Permission::Read,
                        crate::permission::EnabledPermissions::ALL,
                    ),
                    crate::workspace::cwd_for_test(),
                    crate::workspace::roots_for_test(),
                    Arc::new(crate::frontend::SilentFrontend),
                )
            },
            crate::tools::registry::RegistryScope {
                denials: ToolDenials::default(),
                memory_access: MemoryAccess::Write,
                parent_session_id: None,
                inherited_scratchpad_names: Vec::new(),
            },
        )
        .expect("subagent registry should build");

        let todo = sub_registry.get("todo").expect("subagent should have todo");
        todo.execute(
            serde_json::json!({ "title": "Sub work", "items": ["sub task"] }),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("todo should succeed");

        assert_eq!(sub_list.get().items.len(), 1);
        assert!(
            parent_list.get().items.is_empty(),
            "parent list must remain untouched"
        );
    }

    fn spec_for_test(permission: Permission) -> SubagentSpec {
        SubagentSpec {
            permission,
            enabled_permissions: vec![Permission::None, permission],
            denied_servers: vec!["mekabridge".to_string()],
            denied_tools: vec!["write_file".to_string()],
            memory: MemoryAccess::None,
            instructions: InstructionAccess::Inherit,
            inherited_scratchpad: vec!["build_log".to_string()],
            remaining_depth: 2,
            absolute_depth: 1,
            writable_roots: Vec::new(),
        }
    }

    /// The window has to come off the same live profile as the provider.
    ///
    /// `build_subagent` reads the provider from `live_binding.current()`, so the window must come
    /// from there too rather than from `parent_options`, a clone frozen when the session was
    /// assembled. After a `/profile`, `PATCH` or `set_config_option` switch the two disagree, and a
    /// worker would talk to the new profile while auto-compacting against the size of the one the
    /// session had left.
    ///
    /// `parent_options` carries no window at all, so the two cannot be taken from different places.
    /// This is the behavioral check that the worker gauges against what `binding_on` published
    /// (200_000) rather than any default.
    #[tokio::test]
    async fn a_worker_gauges_against_the_window_its_parent_runs_on_now() {
        let store = Store::for_test().await;
        let parent_session = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("create parent");
        let params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_session)),
        );
        let worker = build_subagent(
            &params,
            &spec_for_test(Permission::Read),
            parent_session,
            Uuid::new_v4(),
            WorkerWorkspace {
                cwd: crate::workspace::cwd_for_test(),
                roots: crate::workspace::roots_for_test(),
            },
            "agent_spawn",
            &crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("build the worker");

        assert_eq!(
            worker.context_usage().1,
            200_000,
            "the worker must gauge against the profile its parent runs on now"
        );
    }

    #[test]
    fn subagent_spec_round_trips_through_json() {
        let spec = spec_for_test(Permission::Read);
        let encoded = serde_json::to_string(&spec).expect("encode");
        let decoded: SubagentSpec = serde_json::from_str(&encoded).expect("decode");
        assert_eq!(spec, decoded);
        // The two enums persist as the lowercase words config uses, so a stored spec is readable.
        assert!(encoded.contains("\"permission\":\"read\""), "{encoded}");
        assert!(encoded.contains("\"memory\":\"none\""), "{encoded}");
    }

    /// A spec missing fields still decodes rather than failing, but every absent field takes its
    /// *restrictive* value: losing a field must cost the worker authority, never grant it.
    #[test]
    fn subagent_spec_decodes_a_minimal_document_and_fails_closed() {
        let decoded: SubagentSpec =
            serde_json::from_str(r#"{"permission":"unrestricted"}"#).expect("decode");
        assert_eq!(decoded.permission, Permission::Unrestricted);
        assert_eq!(
            decoded.memory,
            MemoryAccess::None,
            "an absent memory level must cost the worker the store, not hand it over"
        );
        assert_eq!(decoded.remaining_depth, 0, "and must not let it spawn");
        // The deny lists are the one pair that can't fail closed on their own (an empty list is
        // indistinguishable from "nothing was denied"), which is why `agent_followup` re-unions
        // them with current config rather than trusting the spec alone.
        assert!(decoded.denied_servers.is_empty());
    }

    /// `permission` is the one field with no default: a spec that lost it cannot be second-guessed,
    /// so the decode fails and `agent_followup` refuses the worker outright.
    #[test]
    fn subagent_spec_without_a_permission_refuses_to_decode() {
        assert!(serde_json::from_str::<SubagentSpec>(r#"{"memory":"write"}"#).is_err());
    }

    /// The core follow-up invariant: the worker is rebuilt at the level its spawn call chose. A
    /// parent that has since moved to `unrestricted` does not drag the worker up with it.
    #[test]
    fn spec_permission_survives_a_parent_that_has_since_escalated() {
        let spec = spec_for_test(Permission::Read);
        let rebuilt = spec.shared_permission(Permission::Unrestricted);
        assert_eq!(rebuilt.get(), Permission::Read);
        assert!(!rebuilt.enabled().is_enabled(Permission::Unrestricted));
        assert!(!rebuilt.enabled().is_enabled(Permission::Workspace));
    }

    /// And the other direction, which matters more: a worker spawned at `unrestricted` must drop to
    /// `read` when the user switches the session down. Otherwise pressing Shift+Tab would stop
    /// the agent writing while leaving every worker it already has free to write on its behalf.
    #[test]
    fn spec_permission_follows_a_parent_that_has_since_been_restricted() {
        let spec = SubagentSpec {
            enabled_permissions: vec![Permission::Read, Permission::Unrestricted],
            ..spec_for_test(Permission::Unrestricted)
        };
        let rebuilt = spec.shared_permission(Permission::Read);
        assert_eq!(rebuilt.get(), Permission::Read);
        assert!(
            !rebuilt.enabled().is_enabled(Permission::Unrestricted),
            "the downgrade has to reach the enabled set too, or a switch path could climb back"
        );
        // None is a floor like any other.
        assert_eq!(
            spec.shared_permission(Permission::None).get(),
            Permission::None
        );
        // Unrestricted parent: the spec's own level still governs.
        assert_eq!(
            spec.shared_permission(Permission::Unrestricted).get(),
            Permission::Unrestricted
        );
    }

    /// The clamp above happens once, at build time. This is the half that was missing: a worker
    /// already running when the user presses Shift+Tab has to see the new level on its next tool
    /// call, not finish at the level it started with. A `shared_permission` minted from a snapshot
    /// would let a downgrade reach the parent's next call and nothing else, while `permissions.md`
    /// presents cycling the parent as the way to restrict sub-agents.
    #[test]
    fn a_running_sub_agent_sees_a_parent_downgrade() {
        let parent = SharedPermission::new(Permission::Unrestricted, EnabledPermissions::ALL);
        let spec = SubagentSpec {
            enabled_permissions: vec![Permission::Read, Permission::Unrestricted],
            ..spec_for_test(Permission::Unrestricted)
        };

        let worker = spec.shared_permission_bounded(&parent);
        assert_eq!(
            worker.get(),
            Permission::Unrestricted,
            "spawned under a write parent"
        );

        // The user cycles the session down mid-run.
        parent.set_unchecked(Permission::None);
        assert_eq!(
            worker.get(),
            Permission::None,
            "the worker must not outlive the authority it was granted under"
        );

        // And back up: the rule is min(own grant, what the human currently permits), read in both
        // directions. The worker never exceeds its own grant either way.
        parent.set_unchecked(Permission::Unrestricted);
        assert_eq!(worker.get(), Permission::Unrestricted);
    }

    /// A worker granted less than its parent keeps its own lower level when the parent is raised:
    /// the ceiling bounds from above and never lifts.
    #[test]
    fn a_parent_raise_does_not_lift_a_worker_above_its_own_grant() {
        let parent = SharedPermission::new(Permission::Read, EnabledPermissions::ALL);
        let spec = spec_for_test(Permission::Read);

        let worker = spec.shared_permission_bounded(&parent);
        parent.set_unchecked(Permission::Unrestricted);

        assert_eq!(
            worker.get(),
            Permission::Read,
            "the spec's own grant is still the worker's ceiling"
        );
    }

    /// A spec whose enabled set is empty or unparseable must not fall back to "everything". The
    /// safe floor is the recorded level alone.
    #[test]
    fn spec_permission_falls_back_narrow_not_wide() {
        let spec = SubagentSpec {
            enabled_permissions: Vec::new(),
            ..spec_for_test(Permission::Read)
        };
        let rebuilt = spec.shared_permission(Permission::Unrestricted);
        assert_eq!(rebuilt.get(), Permission::Read);
        assert!(rebuilt.enabled().is_enabled(Permission::Read));
        assert!(!rebuilt.enabled().is_enabled(Permission::Unrestricted));
    }

    #[test]
    fn spec_denials_reconstruct_both_lists() {
        let denials = spec_for_test(Permission::Read).denials();
        assert!(denials.denies_server("mekabridge"));
        assert!(denials.denies_tool("mcp__mekabridge__send_message"));
        assert!(denials.denies_tool("write_file"));
        assert!(!denials.denies_tool("read_file"));
    }

    /// A published profile wrapping one provider, for a test that does not care about the profile.
    fn binding_for_test() -> crate::provider::PublishedProfile {
        binding_on(Arc::new(crate::provider::mock::MockProvider::from_rounds(
            Vec::new(),
        )))
    }

    fn binding_on(provider: Arc<dyn Provider>) -> crate::provider::PublishedProfile {
        binding_named(provider, "test-profile")
    }

    /// The same, with the profile *name* chosen, for a test that needs the parent's live profile to
    /// differ from what its row records.
    fn binding_named(
        provider: Arc<dyn Provider>,
        profile: &str,
    ) -> crate::provider::PublishedProfile {
        crate::provider::PublishedProfile::detached(&crate::provider::ResolvedProfile {
            provider,
            profile: profile.to_string(),
            context_window: 200_000,
            vision: true,
        })
    }

    impl ToolBuilderParams {
        /// Point these params at one provider, the way `assemble_agent` points the real ones at the
        /// session's. A method rather than a field write so a test reads as "the parent is running
        /// on this", which is what the cells' profile means.
        fn on_provider(mut self, provider: Arc<dyn Provider>) -> Self {
            self.cells.profile = binding_on(provider);
            self
        }
    }

    /// A parent agent's `ToolBuilderParams` pointing at `store`, with `parent_permission`
    /// as the ceiling and no MCP manager attached.
    fn params_for_test(
        store: Store,
        parent_session: crate::session::SharedSessionId,
    ) -> ToolBuilderParams {
        let mut cells = crate::session::SessionCells::new(
            SharedPermission::new(Permission::Read, EnabledPermissions::ALL),
            crate::workspace::cwd_for_test(),
            crate::workspace::roots_for_test(),
            binding_for_test(),
            Arc::new(crate::frontend::SilentFrontend),
        );
        cells.session_id = parent_session;
        ToolBuilderParams {
            materials: crate::session::SessionMaterials::for_test(store),
            cells,
            // A root agent: holds the whole store, and has instructions it could pass on.
            memory_access: MemoryAccess::Write,
            config_denials: ToolDenials::default(),
            parent_options: AgentOptions {
                streaming: false,
                sandboxed_shell: false,
                gate_tools: None,
                context_messages: None,
                auto_compact: false,
                compact_checkpoint: false,
                user_instructions: Some("never inherited by a worker".to_string()),
                mcp_grace: std::time::Duration::ZERO,
                system_prompt_override: None,
            },
        }
    }

    /// The whole Phase 4 loop against a scripted provider: spawn returns an id, `agent_list`
    /// reports the worker, a follow-up sees the first turn's history, and `agent_delete`
    /// removes it.
    #[tokio::test]
    async fn spawn_followup_and_delete_round_trip() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent session");
        let parent_session = crate::session::SharedSessionId::new(Some(parent_sid));
        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("first answer"),
                text_round("second answer"),
            ]));
        let params = params_for_test(store.clone(), parent_session.clone());

        let spawn = AgentSpawnTool {
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params.clone().on_provider(Arc::clone(&provider)),
            inherited_denials: ToolDenials::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        };
        let output = spawn
            .execute(
                serde_json::json!({ "prompt": "look into it", "permission": "read" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        let text = output.text_content();
        assert!(text.contains("first answer"), "{text}");

        let agent_id: Uuid = text
            .lines()
            .find_map(|line| line.strip_prefix("agent: "))
            .and_then(|id| Uuid::parse_str(id.trim()).ok())
            .unwrap_or_else(|| panic!("spawn must return a usable agent id, got: {text}"));

        // The spawn call restricted the worker below the parent; the persisted spec says so, and
        // that is what a follow-up rebuilds from.
        let spec: SubagentSpec = serde_json::from_str(
            &store
                .load_subagent_spec(agent_id)
                .await
                .expect("load spec")
                .expect("a spawned worker has a spec"),
        )
        .expect("spec decodes");
        assert_eq!(spec.permission, Permission::Read);

        let list = AgentListTool {
            tool_builder_params: params.clone(),
        };
        let listed = list
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("list succeeds")
            .text_content();
        assert!(listed.contains(&agent_id.to_string()), "{listed}");

        let followup = AgentFollowupTool {
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params.clone().on_provider(Arc::clone(&provider)),
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        let second = followup
            .execute(
                serde_json::json!({ "id": agent_id.to_string(), "prompt": "and then?" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("followup succeeds")
            .text_content();
        assert!(second.contains("second answer"), "{second}");

        // The worker's own history carried into the second turn rather than starting over: its log
        // now holds both tasks and both answers.
        let events = store.load_events(agent_id).await.expect("events");
        let transcript = format!("{events:?}");
        assert!(transcript.contains("look into it"), "{transcript}");
        assert!(transcript.contains("first answer"), "{transcript}");
        assert!(transcript.contains("and then?"), "{transcript}");

        let delete = AgentDeleteTool {
            tool_builder_params: params,
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        delete
            .execute(
                serde_json::json!({ "id": agent_id.to_string() }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("delete succeeds");
        assert!(
            store
                .load_session_tree(parent_sid)
                .await
                .expect("tree")
                .iter()
                .all(|row| row.id != agent_id),
            "the deleted worker must be gone from the parent's tree"
        );
        // The parent itself is untouched.
        assert!(
            store
                .load_session_tree(parent_sid)
                .await
                .expect("tree")
                .iter()
                .any(|row| row.id == parent_sid)
        );
    }

    /// A follow-up holds the worker's session for the length of its turn: it runs a full turn
    /// against a row nothing else claims, for seconds to minutes, and unlocked, a concurrent `meka
    /// session delete --all` would take the lock nobody holds and cascade the conversation away,
    /// so the follow-up's next message insert would die on a foreign-key violation with the
    /// worker's output lost.
    ///
    /// Refused rather than warned, unlike spawn: this id already exists, so a lock it cannot take
    /// genuinely means somebody else is running a turn on this worker, and two turns interleaved
    /// into one conversation is the thing the lock is for.
    #[tokio::test]
    async fn a_followup_is_refused_while_something_else_holds_the_worker() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let parent_session = crate::session::SharedSessionId::new(Some(parent_sid));
        let params = params_for_test(store.clone(), parent_session.clone());
        // Never reached: the refusal happens before any turn runs, which is the point.
        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("unreachable"),
            ]));
        // The claim `create_child_session` takes *is* the contention: holding it here is exactly
        // what a second meka looks like from the follow-up's side, since `flock` conflicts across
        // open file descriptions rather than across processes. A real spec, or the follow-up
        // refuses at the resumability check before it ever reaches the lock.
        let spec = SubagentSpec {
            permission: Permission::Read,
            enabled_permissions: vec![Permission::Read],
            denied_servers: Vec::new(),
            denied_tools: Vec::new(),
            memory: MemoryAccess::None,
            instructions: InstructionAccess::None,
            inherited_scratchpad: Vec::new(),
            remaining_depth: 0,
            absolute_depth: 1,
            writable_roots: Vec::new(),
        };
        let (worker, held) = store
            .create_child_session(
                parent_sid,
                None,
                Vec::new(),
                Some(serde_json::to_string(&spec).expect("serialize the spec")),
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child");
        let _held = held.expect("the spawn's own claim");

        let followup = AgentFollowupTool {
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params.clone().on_provider(Arc::clone(&provider)),
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        let refused = followup
            .execute(
                serde_json::json!({ "id": worker.to_string(), "prompt": "carry on" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a worker somebody else is running must not be run again");
        assert!(
            refused.to_string().contains("cannot follow up"),
            "and the refusal has to name the worker: {refused}"
        );
    }

    /// A follow-up the lock refuses leaves the worker's row on the profile it was on. The row is
    /// the billing record for a turn; written ahead of the lock, a refused follow-up moved it onto
    /// the parent's profile for a turn that never ran.
    #[tokio::test]
    async fn a_refused_followup_leaves_the_workers_profile_alone() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let parent_session = crate::session::SharedSessionId::new(Some(parent_sid));
        let params = params_for_test(store.clone(), parent_session.clone());
        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("unreachable"),
            ]));
        let spec = SubagentSpec {
            permission: Permission::Read,
            enabled_permissions: vec![Permission::Read],
            denied_servers: Vec::new(),
            denied_tools: Vec::new(),
            memory: MemoryAccess::None,
            instructions: InstructionAccess::None,
            inherited_scratchpad: Vec::new(),
            remaining_depth: 0,
            absolute_depth: 1,
            writable_roots: Vec::new(),
        };
        // On a profile the parent is not on, so a write that should not happen is visible.
        let (worker, held) = store
            .create_child_session(
                parent_sid,
                None,
                Vec::new(),
                Some(serde_json::to_string(&spec).expect("serialize the spec")),
                "read".to_string(),
                "stale-profile".to_string(),
            )
            .await
            .expect("child");
        let _held = held.expect("the spawn's own claim");

        let followup = AgentFollowupTool {
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params.clone().on_provider(Arc::clone(&provider)),
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        followup
            .execute(
                serde_json::json!({ "id": worker.to_string(), "prompt": "carry on" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("the held lock refuses the follow-up");

        assert_eq!(
            store
                .recorded_profile(worker)
                .await
                .expect("read the row")
                .as_deref(),
            Some("stale-profile"),
            "a turn that did not run must not move the row"
        );
    }

    /// A follow-up turn runs the worker at the parent's *current* level, not at the one it was
    /// spawned with.
    ///
    /// Asserted against the worker's persisted conversation rather than against its spec, since the
    /// registry is what actually decides what it can do.
    #[tokio::test]
    async fn followup_drops_a_worker_when_the_session_is_restricted() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let parent_session = crate::session::SharedSessionId::new(Some(parent_sid));
        // A path unique to this run: a shared one would leave a file behind on the failing case and
        // make the *next* run fail for the wrong reason.
        let temporary = tempfile::tempdir().expect("tempdir");
        let target = temporary.path().join("escalation-probe.txt");
        // The worker tries to write on its follow-up turn, then reports.
        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                vec![
                    crate::provider::mock::MockEvent::ToolUseStart {
                        id: "call-1".into(),
                        name: "write_file".into(),
                    },
                    crate::provider::mock::MockEvent::ToolUseEnd {
                        input: serde_json::json!({
                            "path": target.to_string_lossy(),
                            "content": "escalated",
                        }),
                    },
                    crate::provider::mock::MockEvent::MessageEnd {
                        stop_reason: crate::provider::mock::MockStopReason::ToolUse,
                    },
                ],
                text_round("could not write"),
            ]));
        let params = params_for_test(store.clone(), parent_session.clone());

        // Spawned at `unrestricted`.
        let spec = SubagentSpec {
            permission: Permission::Unrestricted,
            enabled_permissions: vec![Permission::Read, Permission::Unrestricted],
            denied_servers: Vec::new(),
            denied_tools: Vec::new(),
            memory: MemoryAccess::Write,
            instructions: InstructionAccess::None,
            inherited_scratchpad: Vec::new(),
            remaining_depth: 0,
            absolute_depth: 1,
            writable_roots: Vec::new(),
        };
        let child = store
            .create_child_session(
                parent_sid,
                None,
                Vec::new(),
                Some(serde_json::to_string(&spec).expect("encode")),
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;

        // The session is then restricted to Read, as `/permission` or Shift+Tab would.
        let restricted = SharedPermission::new(
            Permission::Read,
            EnabledPermissions::from_levels([Permission::Read]).expect("set"),
        );
        let followup = AgentFollowupTool {
            parent_permission: restricted,
            tool_builder_params: params.on_provider(provider),
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        followup
            .execute(
                serde_json::json!({ "id": child.to_string(), "prompt": "write the file" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("followup runs");

        // The worker really was refused, by the dispatcher, at Read. Asserted against its persisted
        // conversation rather than against the spec: what matters is what the worker was able to
        // *do*, and a spec-only assertion would still pass if the clamp never reached the registry.
        let transcript = format!("{:?}", store.load_events(child).await.expect("events"));
        assert!(
            transcript.contains("'write_file' requires `workspace`; the session is at `read`"),
            "the worker should have been refused write_file at Read, got: {transcript}"
        );
        assert!(!target.exists(), "and nothing should have been written");

        // The recorded spec is untouched, so the worker returns to `unrestricted` if the session
        // does: the clamp is a live ceiling, not a rewrite of the spawn terms.
        let reloaded: SubagentSpec = serde_json::from_str(
            &store
                .load_subagent_spec(child)
                .await
                .expect("load")
                .expect("spec"),
        )
        .expect("decode");
        assert_eq!(reloaded.permission, Permission::Unrestricted);
    }

    /// A restriction added to config after a worker was spawned still reaches it on follow-up. The
    /// spec is a floor on restriction, not a license to ignore what the operator has since decided.
    #[tokio::test]
    async fn followup_applies_denials_config_gained_since_the_spawn() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("ok"),
            ]));
        let mut params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        // Config now denies a server and all memory access; the spec predates both.
        params.config_denials = ToolDenials::new(vec!["mekabridge".to_string()], Vec::new());
        params.memory_access = MemoryAccess::None;

        let spec = SubagentSpec {
            permission: Permission::Read,
            enabled_permissions: vec![Permission::Read],
            denied_servers: Vec::new(),
            denied_tools: Vec::new(),
            memory: MemoryAccess::Write,
            instructions: InstructionAccess::None,
            inherited_scratchpad: Vec::new(),
            remaining_depth: 0,
            absolute_depth: 1,
            writable_roots: Vec::new(),
        };
        let child = store
            .create_child_session(
                parent_sid,
                None,
                Vec::new(),
                Some(serde_json::to_string(&spec).expect("encode")),
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;

        let followup = AgentFollowupTool {
            parent_permission: SharedPermission::new(Permission::Read, EnabledPermissions::ALL),
            tool_builder_params: params.on_provider(provider),
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        followup
            .execute(
                serde_json::json!({ "id": child.to_string(), "prompt": "go" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("followup runs");

        // The combine happens in memory; the stored spec still records the original spawn terms, so
        // loosening config later restores them rather than leaving the worker permanently narrowed.
        let reloaded: SubagentSpec = serde_json::from_str(
            &store
                .load_subagent_spec(child)
                .await
                .expect("load")
                .expect("spec"),
        )
        .expect("decode");
        assert!(reloaded.denied_servers.is_empty());
        assert_eq!(reloaded.memory, MemoryAccess::Write);
    }

    /// Config denials narrow a recorded grant on every axis, not just the ones a test happened to
    /// look at.
    ///
    /// Asserted against the combined spec directly. The call site deliberately leaves the stored
    /// spec alone so a later loosening restores the original terms, which means the only thing
    /// observable there is that the recording did not change, true whether or not the narrowing
    /// ran.
    #[test]
    fn a_followup_narrows_a_recorded_grant_by_config_on_every_axis() {
        let spec = SubagentSpec {
            permission: Permission::Read,
            enabled_permissions: vec![Permission::Read],
            denied_servers: Vec::new(),
            denied_tools: Vec::new(),
            memory: MemoryAccess::Write,
            instructions: InstructionAccess::None,
            inherited_scratchpad: Vec::new(),
            remaining_depth: 0,
            absolute_depth: 1,
            writable_roots: Vec::new(),
        };
        let config = ToolDenials::new(vec!["mekabridge".to_string()], vec![
            "web_search".to_string(),
        ]);

        let combined = combined_for_followup(spec.clone(), &config, MemoryAccess::None);

        assert!(
            combined.denied_servers.contains(&"mekabridge".to_string()),
            "a server denied since the spawn must reach the worker: {:?}",
            combined.denied_servers
        );
        assert!(
            combined.denied_tools.contains(&"web_search".to_string()),
            "and so must a tool: {:?}",
            combined.denied_tools
        );
        assert_eq!(
            combined.memory,
            MemoryAccess::None,
            "memory narrows to the lesser of the grant and config"
        );
        // Never the other direction: config cannot hand back what the spawn call withheld.
        assert_eq!(combined.permission, Permission::Read);
        assert_eq!(
            combined_for_followup(spec, &ToolDenials::default(), MemoryAccess::Write).memory,
            MemoryAccess::Write,
            "an empty config leaves the recorded grant exactly as it was"
        );
    }

    /// Drive a spawn and hand back the spec that was recorded for the worker.
    async fn spawn_and_read_spec(
        params: ToolBuilderParams,
        parent_permission: Permission,
        parent_sid: Uuid,
        input: serde_json::Value,
    ) -> Result<SubagentSpec> {
        let store = params.materials.store.clone();
        let spawn = AgentSpawnTool {
            parent_permission: SharedPermission::new(parent_permission, EnabledPermissions::ALL),
            tool_builder_params: params.on_provider(Arc::new(
                crate::provider::mock::MockProvider::from_rounds(vec![text_round("done")]),
            )),
            inherited_denials: ToolDenials::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        };
        let output = spawn
            .execute(
                input,
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await?;
        let text = output.text_content();
        let agent_id: Uuid = text
            .lines()
            .find_map(|line| line.strip_prefix("agent: "))
            .and_then(|id| Uuid::parse_str(id.trim()).ok())
            .unwrap_or_else(|| panic!("no agent id in: {text}"));
        let _ = parent_sid;
        let json = store
            .load_subagent_spec(agent_id)
            .await?
            .expect("a spawned worker has a spec");
        Ok(serde_json::from_str(&json).expect("spec decodes"))
    }

    /// A worker's row records the profile its parent is *running on*, not the one its parent's row
    /// says.
    ///
    /// The sibling of [`a_worker_gauges_against_the_window_its_parent_runs_on_now`]: provider,
    /// window and the recorded profile all have to come off the one live cell, not the parent's
    /// row.
    ///
    /// The two come apart for as long as a repin that could not take the runtime lock. ACP's
    /// `session/set_config_option` moves the row mid-turn and `try_lock`s the runtime; when a turn
    /// is in flight that fails, and the agent stays where it was until the next turn. A worker
    /// spawned in that window runs on the previous profile and bills its account while its row
    /// claims the new one, so a later `agent_followup` on that child would resolve a different
    /// account from the one that did the work.
    ///
    /// The row here is deliberately left on `stale-row-profile` while the live profile says
    /// `live-profile`, because a test where the two agree cannot tell the fix from the bug.
    #[tokio::test]
    async fn a_spawned_worker_records_the_profile_its_parent_runs_on_now() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "stale-row-profile".to_string())
            .await
            .expect("parent");
        let mut params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        params.cells.profile = binding_named(
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("done"),
            ])),
            "live-profile",
        );

        let spawn = AgentSpawnTool {
            parent_permission: SharedPermission::new(Permission::Read, EnabledPermissions::ALL),
            tool_builder_params: params,
            inherited_denials: ToolDenials::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        };
        let output = spawn
            .execute(
                serde_json::json!({"prompt": "do a thing", "permission": "read"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn");
        let text = output.text_content();
        let agent_id: Uuid = text
            .lines()
            .find_map(|line| line.strip_prefix("agent: "))
            .and_then(|id| Uuid::parse_str(id.trim()).ok())
            .unwrap_or_else(|| panic!("no agent id in: {text}"));

        assert_eq!(
            store
                .recorded_profile(agent_id)
                .await
                .expect("read the child's row"),
            Some("live-profile".to_string()),
            "the worker's row must name what it was built on, not what the parent's row still says"
        );
        assert_eq!(
            store
                .recorded_profile(parent_sid)
                .await
                .expect("read the parent's row"),
            Some("stale-row-profile".to_string()),
            "and spawning must not rewrite the parent's own row on the way past"
        );
    }

    /// The sibling of the test above, on the follow-up door.
    ///
    /// `build_subagent` takes the parent's live profile for a follow-up exactly as it does for a
    /// spawn, so unless the follow-up writes it down too, a worker spawned on `first-profile` and
    /// followed up after a `/profile` switch runs on and bills `second-profile` while its row still
    /// says `first-profile`: a session whose turns do not run on the profile its row names, which
    /// meka otherwise forbids.
    ///
    /// The switch between the two calls is the whole point; a test where the profile never moves
    /// passes with the write deleted.
    #[tokio::test]
    async fn a_followed_up_worker_records_the_profile_it_is_now_running_on() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "first-profile".to_string())
            .await
            .expect("parent");
        let mut params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        params.cells.profile = binding_named(
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("spawned"),
                text_round("followed up"),
            ])),
            "first-profile",
        );

        let spawn = AgentSpawnTool {
            parent_permission: SharedPermission::new(Permission::Read, EnabledPermissions::ALL),
            tool_builder_params: params.clone(),
            inherited_denials: ToolDenials::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        };
        let output = spawn
            .execute(
                serde_json::json!({"prompt": "do a thing", "permission": "read"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn");
        let text = output.text_content();
        let agent_id: Uuid = text
            .lines()
            .find_map(|line| line.strip_prefix("agent: "))
            .and_then(|id| Uuid::parse_str(id.trim()).ok())
            .unwrap_or_else(|| panic!("no agent id in: {text}"));

        // What `/profile`, `PATCH /v1/sessions/{id}` and ACP's `session/set_config_option` all
        // leave behind: the cell `build_subagent` reads now names a different profile.
        params.cells.profile = binding_named(
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("followed up"),
            ])),
            "second-profile",
        );

        let followup = AgentFollowupTool {
            parent_permission: SharedPermission::new(Permission::Read, EnabledPermissions::ALL),
            tool_builder_params: params,
            in_flight: InFlightFollowups::default(),
        };
        followup
            .execute(
                serde_json::json!({"id": agent_id.to_string(), "prompt": "again"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("follow up");

        assert_eq!(
            store
                .recorded_profile(agent_id)
                .await
                .expect("read the worker's row"),
            Some("second-profile".to_string()),
            "a follow-up runs the worker on the parent's profile now, so the row has to say so"
        );
    }

    /// A worker starts with nothing it was not given. Both grants default to the restrictive end,
    /// so a parent that never considers the question produces a clean slate rather than a copy of
    /// itself.
    #[tokio::test]
    async fn grants_default_to_nothing() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let params = params_for_test(
            store,
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );

        let spec = spawn_and_read_spec(
            params,
            Permission::Unrestricted,
            parent_sid,
            serde_json::json!({ "prompt": "go" }),
        )
        .await
        .expect("spawn");
        assert_eq!(spec.memory, MemoryAccess::None);
        assert_eq!(spec.instructions, InstructionAccess::None);
    }

    /// And gets exactly what it was given when the parent asks.
    #[tokio::test]
    async fn grants_are_recorded_when_asked_for() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let params = params_for_test(
            store,
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );

        let spec = spawn_and_read_spec(
            params,
            Permission::Unrestricted,
            parent_sid,
            serde_json::json!({ "prompt": "go", "memory": "read", "instructions": "inherit" }),
        )
        .await
        .expect("spawn");
        assert_eq!(spec.memory, MemoryAccess::Read);
        assert_eq!(spec.instructions, InstructionAccess::Inherit);
    }

    /// `write` is refused rather than clamped to `read`, so a parent asking for it learns that no
    /// sub-agent can have it instead of quietly getting something else.
    #[tokio::test]
    async fn memory_write_is_refused_not_clamped() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );

        let error = spawn_and_read_spec(
            params,
            Permission::Unrestricted,
            parent_sid,
            serde_json::json!({ "prompt": "go", "memory": "write" }),
        )
        .await
        .expect_err("write is not grantable");
        assert!(error.to_string().contains("not available to sub-agents"));
        // A refused spawn leaves nothing behind.
        assert_eq!(
            store
                .load_session_tree(parent_sid)
                .await
                .expect("tree")
                .len(),
            1
        );
    }

    /// A worker cannot hand a grandchild more than it holds. Memory clamps against the spawning
    /// agent's own level; instructions clamp against whether it has the text at all.
    #[tokio::test]
    async fn a_worker_cannot_grant_more_than_it_holds() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let mut params = params_for_test(
            store,
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        // Stand in for a worker that was itself granted nothing: no memory, and no copy of the
        // instructions to pass on. This is exactly what `build_subagent` hands a nested
        // `AgentSpawnTool`.
        params.memory_access = MemoryAccess::None;
        params.parent_options.user_instructions = None;

        let spec = spawn_and_read_spec(
            params,
            Permission::Unrestricted,
            parent_sid,
            serde_json::json!({ "prompt": "go", "memory": "read", "instructions": "inherit" }),
        )
        .await
        .expect("spawn");
        assert_eq!(
            spec.memory,
            MemoryAccess::None,
            "a worker with no memory cannot grant read"
        );
        assert_eq!(
            spec.instructions,
            InstructionAccess::None,
            "and one with no instructions cannot pass them on"
        );
    }

    /// The same clamp, but through `build_subagent` rather than a hand-built `ToolBuilderParams`.
    ///
    /// A worker granted nothing spawns a grandchild and asks for everything. Written this way
    /// because the clamp is only as good as the params the nested `AgentSpawnTool` is handed, and a
    /// test that sets those params itself would pass even if `build_subagent` set them wrong.
    #[tokio::test]
    async fn a_worker_granted_nothing_cannot_grant_its_own_child_anything() {
        use crate::provider::mock::{MockEvent, MockStopReason};

        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );

        // Worker turn 1 asks for a grandchild with both grants; then the grandchild runs; then the
        // worker reports. All three drain from the one shared script, in that order.
        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                vec![
                    MockEvent::ToolUseStart {
                        id: "nest-1".into(),
                        name: "agent_spawn".into(),
                    },
                    MockEvent::ToolUseEnd {
                        input: serde_json::json!({
                            "prompt": "grandchild task",
                            "memory": "read",
                            "instructions": "inherit",
                        }),
                    },
                    MockEvent::MessageEnd {
                        stop_reason: MockStopReason::ToolUse,
                    },
                ],
                text_round("grandchild done"),
                text_round("worker done"),
            ]));

        let spawn = AgentSpawnTool {
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params.on_provider(provider),
            inherited_denials: ToolDenials::default(),
            // Deep enough that the worker gets its own `agent_spawn`.
            remaining_depth: 2,
            absolute_depth: 0,
        };
        // The worker itself is granted nothing, which is the default.
        spawn
            .execute(
                serde_json::json!({ "prompt": "worker task" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn");

        let tree = store.load_session_tree(parent_sid).await.expect("tree");
        assert_eq!(tree.len(), 3, "parent, worker, grandchild");
        let worker = tree
            .iter()
            .find(|row| row.parent_id == Some(parent_sid))
            .expect("worker");
        let grandchild = tree
            .iter()
            .find(|row| row.parent_id == Some(worker.id))
            .expect("the worker really spawned one");

        let spec: SubagentSpec = serde_json::from_str(
            &store
                .load_subagent_spec(grandchild.id)
                .await
                .expect("load")
                .expect("spec"),
        )
        .expect("decode");
        assert_eq!(
            spec.memory,
            MemoryAccess::None,
            "the worker held no memory, so it had none to grant"
        );
        assert_eq!(
            spec.instructions,
            InstructionAccess::None,
            "and no copy of the instructions to pass on"
        );
    }

    /// The report handed to a scratchpad must be the report, with no `agent:` header bolted on.
    ///
    /// The redirect is universal and stores the whole result text, so a header would end up inside
    /// the entry that `inherit_scratchpad` later hands to another worker, and the model would not
    /// even receive the id, since it sees only a reference once the redirect fires.
    #[tokio::test]
    async fn a_redirected_report_carries_no_agent_header() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        let spawn = AgentSpawnTool {
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params.on_provider(Arc::new(
                crate::provider::mock::MockProvider::from_rounds(vec![
                    text_round("the findings"),
                    text_round("the findings"),
                ]),
            )),
            inherited_denials: ToolDenials::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        };

        let redirected = spawn
            .execute(
                serde_json::json!({ "prompt": "go", "scratchpad": "findings" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn")
            .text_content();
        assert_eq!(
            redirected.trim(),
            "the findings",
            "a redirected result is the report alone"
        );
        assert!(!redirected.contains("agent:"), "{redirected}");

        // Without the redirect the id leads, which is how a parent reaches the worker again.
        let inline = spawn
            .execute(
                serde_json::json!({ "prompt": "go" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn")
            .text_content();
        assert!(inline.starts_with("agent: "), "{inline}");
        assert!(inline.contains("the findings"));
    }

    /// A spec claiming `Write` cannot produce a worker that can write. `parse_grant` refuses it at
    /// the `agent_spawn` boundary, but a spec is persisted JSON and `meka session import` writes it
    /// verbatim from a user-supplied archive, so the guarantee is enforced where it is consumed.
    #[test]
    fn a_spec_claiming_write_is_capped_at_read() {
        let forged = SubagentSpec {
            memory: MemoryAccess::Write,
            ..spec_for_test(Permission::Unrestricted)
        };
        assert_eq!(forged.granted_memory(), MemoryAccess::Read);
        // And the honest values pass through untouched.
        assert_eq!(
            SubagentSpec {
                memory: MemoryAccess::Read,
                ..spec_for_test(Permission::Read)
            }
            .granted_memory(),
            MemoryAccess::Read
        );
        assert_eq!(
            spec_for_test(Permission::Read).granted_memory(),
            MemoryAccess::None
        );
    }

    /// A `memory: "read"` grant has to arrive with an index, or it is only half a grant:
    fn memory_for_test(name: &str, priority: u8, description: &str) -> crate::memory::Memory {
        crate::memory::Memory {
            name: name.to_string(),
            description: description.to_string(),
            priority,
            tags: Vec::new(),
            recorded_at: std::time::SystemTime::UNIX_EPOCH,
            updated_at: std::time::SystemTime::UNIX_EPOCH,
            read_count: 0,
            body: None,
        }
    }

    /// `memory_read` takes an exact name, and a sub-agent never receives the per-turn world state
    /// the root agent's `[Memory]` section rides in. Without this the worker holds two tools and
    /// no idea what to call them with.
    #[test]
    fn a_granted_worker_gets_a_usable_memory_index() {
        let index = [
            memory_for_test("build-incantation", 1, "How this project is built"),
            memory_for_test("review-style", 2, "What the user wants from a review"),
        ];

        let rendered = render_subagent_memory_index(&index);
        assert!(rendered.contains("build-incantation"));
        assert!(rendered.contains("How this project is built"));
        assert!(rendered.contains("review-style"));
        assert!(rendered.contains("memory_read"), "and how to open one");
        // A worker cannot write, so it must not be told to. The root agent's header says to call
        // `memory_write`, which is exactly why this renders separately.
        assert!(
            !rendered.contains("memory_write"),
            "must not point a read-only worker at a tool it lacks: {rendered}"
        );
        assert!(rendered.contains("report"), "it reports instead");

        // An empty store contributes nothing at all rather than an empty heading.
        assert!(render_subagent_memory_index(&[]).is_empty());
    }

    /// A truncated index must say so. Reading as "this is everything" is what turns a full store
    /// into a confidently incomplete answer.
    #[test]
    fn a_truncated_memory_index_states_its_remainder() {
        let memories: Vec<crate::memory::Memory> = (0..400)
            .map(|n| {
                memory_for_test(
                    &format!("memory-{n:03}"),
                    1,
                    &"a description long enough to make the budget bite".repeat(3),
                )
            })
            .collect();
        let rendered = render_subagent_memory_index(&memories);
        assert!(
            rendered.len() <= SUBAGENT_MEMORY_INDEX_MAX_BYTES + 200,
            "budget respected"
        );
        assert!(rendered.contains("more not listed here"), "{rendered}");
        assert!(rendered.contains("memory_search"), "and how to reach them");
    }

    /// One enormous description must not empty a granted worker's whole index.
    ///
    /// Descriptions are unbounded at the write door. With the budget checked before every push and
    /// nothing elided, a single 4,000-character description at the top of the store would produce
    /// a header promising memories followed by "N more not listed here", in a worker that had been
    /// deliberately granted access to them.
    #[test]
    fn one_enormous_description_does_not_empty_the_subagent_index() {
        let mut memories = vec![memory_for_test("enormous", 1, &"x".repeat(4_000))];
        memories.extend(
            (0..3).map(|n| memory_for_test(&format!("ordinary-{n}"), 3, "a short description")),
        );

        let rendered = render_subagent_memory_index(&memories);

        assert!(
            rendered.contains("**enormous**"),
            "the first entry is always emitted: {rendered}"
        );
        assert!(
            !rendered.contains(&"x".repeat(4_000)),
            "and its description is elided rather than carried whole"
        );
        assert!(
            rendered.contains("**ordinary-0**"),
            "which leaves room for the rest of the store: {rendered}"
        );
        assert!(
            rendered.len() <= SUBAGENT_MEMORY_INDEX_MAX_BYTES + 200,
            "budget still respected: {} bytes",
            rendered.len()
        );
    }

    /// The grant reaches the prompt, and its absence leaves no trace of the section.
    #[test]
    fn system_prompt_carries_instructions_only_when_granted() {
        let ungranted = build_subagent_system_prompt(Permission::Read, &[], &[], None, "");
        assert!(!ungranted.contains("User Instructions"));

        let granted = build_subagent_system_prompt(
            Permission::Read,
            &[],
            &[],
            Some("Never use pip. Always prefer uv."),
            "",
        );
        assert!(granted.contains("## User Instructions"));
        assert!(granted.contains("Never use pip. Always prefer uv."));
        // The worker is told whose rules these are, so persona clauses read as context rather than
        // as an instruction to address the user directly.
        assert!(granted.contains("report"), "{granted}");

        // Whitespace-only instructions are treated as absent, matching the root agent.
        assert!(
            !build_subagent_system_prompt(Permission::Read, &[], &[], Some("  \n "), "")
                .contains("User Instructions")
        );
    }

    /// Denying `agent_spawn` means "no delegation", so the three tools that only ever act on what
    /// it produced go with it. Leaving them behind would give an agent that cannot spawn a worker
    /// the ability to drive workers a previous run left in the database.
    #[tokio::test]
    async fn denying_agent_spawn_takes_the_lifecycle_tools_with_it() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        let spawn = |params: ToolBuilderParams| AgentSpawnTool {
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params.on_provider(Arc::new(
                crate::provider::mock::MockProvider::from_rounds(Vec::new()),
            )),
            inherited_denials: ToolDenials::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        };

        let permitted = ToolRegistry::new();
        register_subagent_tools(&permitted, spawn(params.clone())).expect("register");
        for name in [
            "agent_spawn",
            "agent_list",
            "agent_followup",
            "agent_delete",
        ] {
            assert!(permitted.get(name).is_some(), "expected '{name}'");
        }

        let filtered = ToolRegistry::new_with_filter(BuiltinToolFilter::from_config(
            None,
            vec!["agent_spawn".to_string()],
            std::collections::HashMap::new(),
        ));
        register_subagent_tools(&filtered, spawn(params)).expect("register");
        for name in [
            "agent_spawn",
            "agent_list",
            "agent_followup",
            "agent_delete",
        ] {
            assert!(
                filtered.get(name).is_none(),
                "'{name}' must go with agent_spawn"
            );
        }
    }

    /// Each registered tool has to actually describe itself: `meka tools list` reads the same
    /// registration a session does, so a definition that lost its description or its schema would
    /// tell the model and the listing nothing while still registering cleanly.
    #[tokio::test]
    async fn every_agent_tool_describes_itself_and_the_arguments_it_requires() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        let registry = ToolRegistry::new();
        register_subagent_tools(&registry, AgentSpawnTool {
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params.on_provider(Arc::new(
                crate::provider::mock::MockProvider::from_rounds(Vec::new()),
            )),
            inherited_denials: ToolDenials::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        })
        .expect("register");
        for name in [
            "agent_spawn",
            "agent_list",
            "agent_followup",
            "agent_delete",
        ] {
            let definition = registry
                .get(name)
                .unwrap_or_else(|| panic!("'{name}' is registered"))
                .definition();
            let name = &definition.name;
            assert!(name.starts_with("agent_"), "'{name}' is not in the family");
            // A floor rather than an exact length: the point is that a sentence survived, and the
            // shortest of the four runs to several hundred characters.
            assert!(
                definition.description.len() > 40,
                "'{name}' has no usable description"
            );
            assert_eq!(
                definition.parameters["type"], "object",
                "'{name}' must declare an object schema"
            );
            let properties = definition.parameters["properties"]
                .as_object()
                .unwrap_or_else(|| panic!("'{name}' must declare properties"));
            // `agent_list` takes nothing, so an empty set is legal; a required name that no
            // property defines is not, and would reach the model as an unfillable argument.
            for required in definition.parameters["required"]
                .as_array()
                .map(Vec::as_slice)
                .unwrap_or_default()
            {
                let required = required.as_str().unwrap_or_default();
                assert!(
                    properties.contains_key(required),
                    "'{name}' requires '{required}' but declares no such property"
                );
            }
        }
    }

    /// The listing reproduces the registration rule, so the rule has to answer for both halves of
    /// "all four or none" and leave the per-name case to its caller.
    #[test]
    fn the_agent_family_needs_a_depth_budget_and_an_admitted_agent_spawn() {
        let filter = |disabled: Vec<String>| {
            BuiltinToolFilter::from_config(None, disabled, std::collections::HashMap::new())
        };
        assert!(agent_tools_registered(&BuiltinToolFilter::default(), 1));
        // The documented way to turn delegation off entirely.
        assert!(!agent_tools_registered(&BuiltinToolFilter::default(), 0));
        assert!(!agent_tools_registered(
            &filter(vec!["agent_spawn".to_string()]),
            3
        ));
        // Denying a lifecycle tool alone removes that one and leaves the family, so the predicate
        // has to say yes here and let the caller filter the name.
        assert!(agent_tools_registered(
            &filter(vec!["agent_list".to_string()]),
            3
        ));
        // An exhaustive `allowed_tools` that omits `agent_spawn` takes the family with it.
        assert!(!agent_tools_registered(
            &BuiltinToolFilter::from_config(
                Some(vec!["read_file".to_string()]),
                Vec::new(),
                std::collections::HashMap::new(),
            ),
            3
        ));
    }

    /// `agent_delete` shares the follow-up guard, so the two cannot run on one worker at once.
    /// Without it, a delete landing between a follow-up's ownership check and its first write makes
    /// the follow-up fail on a foreign-key violation, a raw database error, on a path where the
    /// model did nothing wrong.
    #[tokio::test]
    async fn delete_is_refused_while_a_followup_holds_the_worker() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = store
            .create_child_session(
                parent_sid,
                None,
                Vec::new(),
                Some(r#"{"permission":"read"}"#.to_string()),
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;
        let params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        let in_flight: InFlightFollowups =
            Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));

        let delete = AgentDeleteTool {
            tool_builder_params: params,
            in_flight: Arc::clone(&in_flight),
        };

        // Stand in for a follow-up in progress on this worker.
        let held = FollowupGuard::claim(&in_flight, child).expect("claim");
        let error = delete
            .execute(
                serde_json::json!({ "id": child.to_string() }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a busy worker must not be deleted mid-follow-up");
        assert!(error.to_string().contains("busy"), "{error}");
        assert_eq!(
            store
                .load_session_tree(parent_sid)
                .await
                .expect("tree")
                .len(),
            2,
            "and it is still there"
        );

        // Once the follow-up returns, the delete goes through.
        drop(held);
        delete
            .execute(
                serde_json::json!({ "id": child.to_string() }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("delete succeeds once the worker is free");
    }

    /// `turns` counts tasks, not messages. `Agent::run_turn` persists tool results as user-role
    /// messages, so a single task that took three tool rounds must still read as one turn.
    #[tokio::test]
    async fn agent_list_turns_excludes_tool_result_messages() {
        use crate::conversation::{ContentBlock, Event, Message, Role, ToolResultContent};

        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = store
            .create_child_session(
                parent_sid,
                None,
                Vec::new(),
                Some(r#"{"permission":"read"}"#.to_string()),
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;

        let tool_result = Event::Append(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "u1".to_string(),
                content: vec![ToolResultContent::Text {
                    text: "ok".to_string(),
                }],
                is_error: false,
            }],
        });
        for event in [
            Event::Append(Message::user("the one and only task")),
            Event::Append(Message::assistant_text("working")),
            tool_result.clone(),
            Event::Append(Message::assistant_text("still working")),
            tool_result,
            Event::Append(Message::assistant_text("done")),
        ] {
            store.save_event(child, &event).await.expect("save");
        }

        let list = AgentListTool {
            tool_builder_params: params_for_test(
                store,
                crate::session::SharedSessionId::new(Some(parent_sid)),
            ),
        };
        let rendered = list
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("list")
            .text_content();
        assert!(
            rendered.contains("turns=1"),
            "one task through two tool rounds is one turn, got: {rendered}"
        );
    }

    /// A skill that resolves by name but fails to load must not leave a childless session row
    /// behind: `agent_list` would advertise it as a worker, and following it up would resume a
    /// conversation that never happened.
    ///
    /// Unix-only because it needs a directory the process cannot read.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_failed_skill_load_leaves_no_orphan_session() {
        use std::os::unix::fs::PermissionsExt;

        let temporary = tempfile::tempdir().expect("tempdir");
        let root = temporary.path().join("skills");
        let skill_dir = root.join("broken");
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: broken\ndescription: a skill whose body goes away\n---\n\nbody\n",
        )
        .expect("write skill");

        let skills = crate::skills::SkillCache::for_root(Some(root.clone()));
        assert_eq!(
            skills.current().await.skills.len(),
            1,
            "skill is discoverable"
        );
        // Making the root unreadable makes `disk_snapshot` return `None`, and `SkillCache::current`
        // then serves its cached list rather than wiping it. So the name still resolves and only
        // the body read fails: the ordering this test is about, and a real race with a
        // `git checkout` or an editor moving a skill mid-turn. (Removing the root instead would not
        // do: a *missing* root is deliberately read as an empty store, which fails resolution.)
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o000))
            .expect("seal skills root");
        if std::fs::read_dir(&root).is_ok() {
            // Running as root, where the mode is advisory. Nothing to assert.
            return;
        }
        assert_eq!(
            skills.current().await.skills.len(),
            1,
            "resolution still succeeds"
        );

        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let mut params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        params.materials.skills = skills;

        let spawn = AgentSpawnTool {
            parent_permission: SharedPermission::new(Permission::Read, EnabledPermissions::ALL),
            tool_builder_params: params.on_provider(Arc::new(
                crate::provider::mock::MockProvider::from_rounds(vec![text_round("never runs")]),
            )),
            inherited_denials: ToolDenials::default(),
            remaining_depth: 0,
            absolute_depth: 0,
        };
        let error = spawn
            .execute(
                serde_json::json!({ "skill": "broken" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("an unreadable skill body must fail the spawn");
        assert!(
            error.to_string().contains("failed to load skill"),
            "{error}"
        );

        assert_eq!(
            store
                .load_session_tree(parent_sid)
                .await
                .expect("tree")
                .len(),
            1,
            "the parent alone: a failed spawn must not leave a child row behind"
        );

        // Let the tempdir clean itself up.
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700))
            .expect("unseal skills root");
    }

    /// Delegating a skill whose `SKILL.md` will not parse must say so, not offer substitutes.
    ///
    /// The `skill_read` wording exists because a model told "not found" improvises the procedure.
    /// `agent_spawn` is the same audience with more at stake (the improvisation runs in a worker,
    /// out of sight), and "skill 'x' not found. Available skills: ..." reads as an invitation to
    /// pick one of those instead.
    #[tokio::test]
    async fn spawning_a_broken_skill_names_the_file_rather_than_offering_alternatives() {
        let temporary = tempfile::tempdir().expect("tempdir");
        let root = temporary.path().join("skills");
        for (name, body) in [
            (
                "wrecked",
                "---\nname: wrecked\ndescription: [unclosed\n---\nbody\n",
            ),
            (
                "fine",
                "---\nname: fine\ndescription: a working one\n---\nbody\n",
            ),
        ] {
            std::fs::create_dir_all(root.join(name)).expect("skill dir");
            std::fs::write(root.join(name).join("SKILL.md"), body).expect("write skill");
        }

        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let mut params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        params.materials.skills = crate::skills::SkillCache::for_root(Some(root));

        let spawn = AgentSpawnTool {
            parent_permission: SharedPermission::new(Permission::Read, EnabledPermissions::ALL),
            tool_builder_params: params.on_provider(Arc::new(
                crate::provider::mock::MockProvider::from_rounds(vec![text_round("never runs")]),
            )),
            inherited_denials: ToolDenials::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        };
        let output = spawn
            .execute(
                serde_json::json!({ "skill": "wrecked" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a broken skill is a tool result, not a tool error");
        assert!(output.is_error);
        let text = output.text_content();
        assert!(
            text.contains("failed to load"),
            "a present-but-unparseable file must not read as absent: {text}"
        );
        assert!(
            !text.contains("Available skills"),
            "naming substitutes invites the model to delegate one: {text}"
        );
        assert_eq!(
            store
                .load_session_tree(parent_sid)
                .await
                .expect("tree")
                .len(),
            1,
            "the refusal happens before any child session exists"
        );
    }

    /// A session may only drive its own workers. This also covers a forked parent, whose copied
    /// conversation names children that are still linked to the original session.
    #[tokio::test]
    async fn followup_and_delete_refuse_a_session_that_is_not_the_parent() {
        let store = store_for_test().await;
        let owner = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("owner");
        let stranger = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("stranger");
        let child = store
            .create_child_session(
                owner,
                None,
                Vec::new(),
                Some("{\"permission\":\"read\"}".to_string()),
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;

        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("should never run"),
            ]));
        let params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(stranger)),
        );

        let followup = AgentFollowupTool {
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params.clone().on_provider(provider),
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        let error = followup
            .execute(
                serde_json::json!({ "id": child.to_string(), "prompt": "hello" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a stranger's worker must be refused");
        assert!(error.to_string().contains("belongs to this session"));

        let delete = AgentDeleteTool {
            tool_builder_params: params,
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        assert!(
            delete
                .execute(
                    serde_json::json!({ "id": child.to_string() }),
                    crate::tools::ToolContext::detached(CancellationToken::new())
                )
                .await
                .is_err(),
            "and must not be deletable either"
        );
        // Still there.
        assert!(
            store
                .load_session_tree(owner)
                .await
                .expect("tree")
                .iter()
                .any(|row| row.id == child)
        );
    }

    /// A worker spawned before the spec column existed has no recorded terms. Refuse rather than
    /// rebuild it from the parent, which is exactly the escalation the spec exists to prevent.
    #[tokio::test]
    async fn followup_refuses_a_worker_with_no_recorded_spec() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = store
            .create_child_session(
                parent_sid,
                None,
                Vec::new(),
                None,
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;

        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("should never run"),
            ]));
        let followup = AgentFollowupTool {
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params_for_test(
                store,
                crate::session::SharedSessionId::new(Some(parent_sid)),
            )
            .on_provider(provider),
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        let error = followup
            .execute(
                serde_json::json!({ "id": child.to_string(), "prompt": "hello" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a spec-less worker must be refused");
        assert!(
            error.to_string().contains("no recorded spawn terms"),
            "{error}"
        );
    }

    /// A parent at `level` working in `cwd`, for the `writable_roots` tests. The spawn tool's
    /// ceiling and the cells' handle are one and the same, as `register_subagent_tools` wires
    /// them, so the acceptor and the worker's live clamp read the same level.
    fn params_at(
        store: Store,
        parent_session: crate::session::SharedSessionId,
        level: Permission,
        cwd: PathBuf,
    ) -> ToolBuilderParams {
        let mut params = params_for_test(store, parent_session);
        params.cells.permission = SharedPermission::new(level, EnabledPermissions::ALL);
        params.cells.cwd = SharedCwd::new(cwd);
        params
    }

    /// A provider that replays `rounds`, handed back concrete so a test can also read what the
    /// worker was sent.
    fn mock(
        rounds: Vec<Vec<crate::provider::mock::MockEvent>>,
    ) -> Arc<crate::provider::mock::MockProvider> {
        Arc::new(crate::provider::mock::MockProvider::from_rounds(rounds))
    }

    fn spawn_tool_for(params: ToolBuilderParams, provider: Arc<dyn Provider>) -> AgentSpawnTool {
        AgentSpawnTool {
            parent_permission: params.cells.permission.clone(),
            tool_builder_params: params.on_provider(provider),
            inherited_denials: ToolDenials::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        }
    }

    fn followup_tool_for(
        params: ToolBuilderParams,
        provider: Arc<dyn Provider>,
    ) -> AgentFollowupTool {
        AgentFollowupTool {
            parent_permission: params.cells.permission.clone(),
            tool_builder_params: params.on_provider(provider),
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        }
    }

    /// One round in which the worker writes `content` to `path` through `write_file`.
    fn write_file_round(
        id: &str,
        path: &std::path::Path,
        content: &str,
    ) -> Vec<crate::provider::mock::MockEvent> {
        vec![
            crate::provider::mock::MockEvent::ToolUseStart {
                id: id.to_string(),
                name: "write_file".to_string(),
            },
            crate::provider::mock::MockEvent::ToolUseEnd {
                input: serde_json::json!({ "path": path.to_string_lossy(), "content": content }),
            },
            crate::provider::mock::MockEvent::MessageEnd {
                stop_reason: crate::provider::mock::MockStopReason::ToolUse,
            },
        ]
    }

    fn agent_id_in(text: &str) -> Uuid {
        text.lines()
            .find_map(|line| line.strip_prefix("agent: "))
            .and_then(|id| Uuid::parse_str(id.trim()).ok())
            .unwrap_or_else(|| panic!("spawn must return a usable agent id, got: {text}"))
    }

    /// A parent's tree with only the parent in it: a refused spawn left no row behind.
    async fn only_the_parent(store: &Store, parent_sid: Uuid) -> bool {
        store
            .load_session_tree(parent_sid)
            .await
            .expect("tree")
            .len()
            == 1
    }

    /// The directories the `writable_roots` tests share: a parent working in `work`, two
    /// candidate roots under it, and one beside it that the parent's workspace does not contain.
    struct Workspaces {
        _temp: tempfile::TempDir,
        work: PathBuf,
        sub: PathBuf,
        extra: PathBuf,
        elsewhere: PathBuf,
    }

    fn workspaces() -> Workspaces {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let work = base.join("work");
        let sub = work.join("sub");
        let extra = work.join("extra");
        let elsewhere = base.join("elsewhere");
        for directory in [&sub, &extra, &elsewhere] {
            std::fs::create_dir_all(directory).expect("dirs");
        }
        Workspaces {
            _temp: temp,
            work,
            sub,
            extra,
            elsewhere,
        }
    }

    /// Spawn under `level` in `cwd` with `input`, expecting a refusal, and hand back its text once
    /// the store shows the refusal came before the row.
    async fn refused_spawn(level: Permission, cwd: PathBuf, input: serde_json::Value) -> String {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let params = params_at(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            level,
            cwd,
        );
        let spawn = spawn_tool_for(params, mock(vec![text_round("never runs")]));
        let error = spawn
            .execute(
                input,
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("the spawn must be refused");
        assert!(
            only_the_parent(&store, parent_sid).await,
            "a refused spawn must leave no child row behind"
        );
        error.to_string()
    }

    /// The feature end to end: a `workspace` parent hands a worker two directories inside its own
    /// workspace, and the worker writes under those and nowhere else: not beside them in the
    /// parent's directory, and not under a root the parent holds and did not pass on. The first
    /// is its working directory, its environment context says so, and the row and the spec both
    /// record the bounds.
    #[tokio::test]
    async fn a_bounded_worker_writes_under_its_roots_and_nowhere_else() {
        let dirs = workspaces();
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let mut params = params_at(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Permission::Workspace,
            dirs.work.clone(),
        );
        // A root of the parent's own, so a worker that inherited the parent's roots rather than
        // taking the list would be caught reaching it.
        params.cells.roots = SharedRoots::new(vec![dirs.elsewhere.clone()]);
        let inside = dirs.sub.join("inside.txt");
        let also = dirs.extra.join("also.txt");
        let beside = dirs.work.join("beside.txt");
        let leak = dirs.elsewhere.join("leak.txt");
        let provider = mock(vec![
            write_file_round("call-1", &inside, "in"),
            write_file_round("call-2", &also, "also"),
            write_file_round("call-3", &beside, "beside"),
            write_file_round("call-4", &leak, "leak"),
            text_round("done"),
        ]);
        let spawn = spawn_tool_for(params, Arc::clone(&provider) as Arc<dyn Provider>);
        let output = spawn
            .execute(
                serde_json::json!({
                    "prompt": "write",
                    "writable_roots": [dirs.sub.to_string_lossy(), dirs.extra.to_string_lossy()],
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        let agent_id = agent_id_in(&output.text_content());

        assert_eq!(
            std::fs::read_to_string(&inside).expect("a write under the first root lands"),
            "in"
        );
        assert_eq!(
            std::fs::read_to_string(&also).expect("a write under the second root lands"),
            "also"
        );
        assert!(
            !beside.exists(),
            "a write beside the roots, inside the parent's own workspace, must be refused"
        );
        assert!(
            !leak.exists(),
            "a root the parent holds but did not pass on is outside the worker's boundary"
        );
        let transcript = format!("{:?}", store.load_events(agent_id).await.expect("events"));
        assert!(
            transcript.contains("outside the workspace"),
            "the worker must have been told why: {transcript}"
        );

        // What the worker was told about its workspace on its first turn: its own directory and
        // roots, and nothing of the parent's.
        // The text as the worker read it, not a `Debug` rendering: that would escape every
        // backslash and no Windows path could match.
        let first_turn = provider
            .completions()
            .first()
            .expect("the worker's first request")
            .iter()
            .map(crate::conversation::Message::wire_text)
            .collect::<Vec<_>>()
            .join("\n");
        // The worker names the canonical spelling the acceptor produced, which on Windows differs
        // from the temp directory as created (8.3 short names, verbatim prefixes).
        let canonical = |path: &std::path::Path| {
            crate::workspace::accept_cwd(path)
                .expect("the test directory exists")
                .display()
                .to_string()
        };
        let sub = canonical(&dirs.sub);
        let extra = canonical(&dirs.extra);
        let elsewhere = canonical(&dirs.elsewhere);
        assert!(
            first_turn.contains(&format!("Working directory: {sub}")),
            "the first root is the working directory the worker is told about: {first_turn}"
        );
        assert!(
            first_turn.contains(&extra) && !first_turn.contains(&elsewhere),
            "the worker is told its own additional root and not the parent's: {first_turn}"
        );

        let row = store
            .load_session_tree(parent_sid)
            .await
            .expect("tree")
            .into_iter()
            .find(|row| row.id == agent_id)
            .expect("the worker's row");
        assert_eq!(
            row.cwd.as_deref(),
            Some(dirs.sub.as_path()),
            "the first root is the worker's working directory"
        );
        assert_eq!(row.additional_roots, vec![dirs.extra.clone()]);
        assert_eq!(row.permission, Some(Permission::Workspace));
        let spec: SubagentSpec = serde_json::from_str(
            &store
                .load_subagent_spec(agent_id)
                .await
                .expect("load")
                .expect("spec"),
        )
        .expect("decode");
        assert_eq!(
            spec.permission,
            Permission::Workspace,
            "bounded with no `permission` given runs at workspace"
        );
        assert_eq!(spec.writable_roots, vec![
            dirs.sub.clone(),
            dirs.extra.clone()
        ]);
    }

    /// A `workspace` parent cannot name a directory its own boundary does not contain: the refusal
    /// names the entry, and comes before the row.
    #[tokio::test]
    async fn a_root_outside_the_parents_workspace_is_refused_before_any_row_exists() {
        let dirs = workspaces();
        let refusal = refused_spawn(
            Permission::Workspace,
            dirs.work.clone(),
            serde_json::json!({
                "prompt": "write",
                "writable_roots": [dirs.elsewhere.to_string_lossy()],
            }),
        )
        .await;
        assert!(
            refusal.contains(&dirs.elsewhere.display().to_string())
                && refusal.contains("outside this session's workspace"),
            "{refusal}"
        );
    }

    /// A parent below `workspace` has no write reach to delegate.
    #[tokio::test]
    async fn a_parent_below_workspace_cannot_bound_a_worker() {
        let dirs = workspaces();
        let refusal = refused_spawn(
            Permission::Read,
            dirs.work.clone(),
            serde_json::json!({
                "prompt": "write",
                "writable_roots": [dirs.sub.to_string_lossy()],
            }),
        )
        .await;
        assert!(
            refusal.contains("needs a parent at `workspace` or `unrestricted`"),
            "{refusal}"
        );
    }

    /// `unrestricted` has no boundary for a list to bound, so asking for both is a contradiction
    /// rather than a worker that silently runs at one or the other.
    #[tokio::test]
    async fn writable_roots_alongside_an_unrestricted_level_are_refused() {
        let dirs = workspaces();
        let refusal = refused_spawn(
            Permission::Unrestricted,
            dirs.work.clone(),
            serde_json::json!({
                "prompt": "write",
                "permission": "unrestricted",
                "writable_roots": [dirs.sub.to_string_lossy()],
            }),
        )
        .await;
        assert!(refusal.contains("has no boundary"), "{refusal}");
    }

    /// An empty list is a refusal, not a fall-through to the parent's workspace.
    #[tokio::test]
    async fn an_empty_writable_roots_list_is_refused() {
        let dirs = workspaces();
        let refusal = refused_spawn(
            Permission::Workspace,
            dirs.work.clone(),
            serde_json::json!({ "prompt": "write", "writable_roots": [] }),
        )
        .await;
        assert!(refusal.contains("names no directory"), "{refusal}");
    }

    /// An `unrestricted` parent may write anywhere, so it may bound a worker to a directory its
    /// own working directory does not contain; the worker still runs at `workspace`, confined to
    /// that directory.
    #[tokio::test]
    async fn an_unrestricted_parent_may_bound_a_worker_outside_its_own_directory() {
        let dirs = workspaces();
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let params = params_at(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Permission::Unrestricted,
            dirs.work.clone(),
        );
        let inside = dirs.elsewhere.join("out.txt");
        let beside = dirs.work.join("beside.txt");
        let spawn = spawn_tool_for(
            params,
            mock(vec![
                write_file_round("call-1", &inside, "out"),
                write_file_round("call-2", &beside, "beside"),
                text_round("done"),
            ]),
        );
        let output = spawn
            .execute(
                serde_json::json!({
                    "prompt": "write",
                    "writable_roots": [dirs.elsewhere.to_string_lossy()],
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        let agent_id = agent_id_in(&output.text_content());

        assert_eq!(
            std::fs::read_to_string(&inside).expect("the named directory is writable"),
            "out"
        );
        assert!(
            !beside.exists(),
            "the parent's own directory is not in the worker's boundary"
        );
        let row = store
            .load_session_tree(parent_sid)
            .await
            .expect("tree")
            .into_iter()
            .find(|row| row.id == agent_id)
            .expect("the worker's row");
        assert_eq!(row.cwd.as_deref(), Some(dirs.elsewhere.as_path()));
        assert_eq!(
            row.permission,
            Some(Permission::Workspace),
            "bounded, so `workspace` rather than the parent's `unrestricted`"
        );
    }

    /// A relative entry names a directory under the parent's working directory, not the process's.
    #[tokio::test]
    async fn a_relative_writable_root_resolves_against_the_parents_directory() {
        let dirs = workspaces();
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let params = params_at(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Permission::Workspace,
            dirs.work.clone(),
        );
        let spawn = spawn_tool_for(params, mock(vec![text_round("done")]));
        let output = spawn
            .execute(
                serde_json::json!({ "prompt": "look", "writable_roots": ["sub"] }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        let agent_id = agent_id_in(&output.text_content());
        let row = store
            .load_session_tree(parent_sid)
            .await
            .expect("tree")
            .into_iter()
            .find(|row| row.id == agent_id)
            .expect("the worker's row");
        assert_eq!(row.cwd.as_deref(), Some(dirs.sub.as_path()));
    }

    /// Spawn a worker bounded to `dirs.sub` under a `workspace` parent in `dirs.work` that also
    /// holds `dirs.elsewhere` as a root of its own, for the follow-up tests. Returns the parent's
    /// params (whose cells the follow-up shares) and the worker's id.
    async fn spawn_bounded_to_sub(store: &Store, dirs: &Workspaces) -> (ToolBuilderParams, Uuid) {
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let mut params = params_at(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Permission::Workspace,
            dirs.work.clone(),
        );
        params.cells.roots = SharedRoots::new(vec![dirs.elsewhere.clone()]);
        let spawn = spawn_tool_for(params.clone(), mock(vec![text_round("spawned")]));
        let output = spawn
            .execute(
                serde_json::json!({
                    "prompt": "wait",
                    "writable_roots": [dirs.sub.to_string_lossy()],
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        (params, agent_id_in(&output.text_content()))
    }

    /// A follow-up rebuilds a bounded worker from its recorded bounds, not from the parent's
    /// workspace: it still writes under its root, and is still refused beside it and under a root
    /// the parent holds.
    #[tokio::test]
    async fn a_followup_keeps_a_workers_bounds() {
        let dirs = workspaces();
        let store = store_for_test().await;
        let (params, agent_id) = spawn_bounded_to_sub(&store, &dirs).await;
        let beside = dirs.work.join("beside.txt");
        let leak = dirs.elsewhere.join("leak.txt");
        let later = dirs.sub.join("later.txt");
        let followup = followup_tool_for(
            params,
            mock(vec![
                write_file_round("call-1", &beside, "beside"),
                write_file_round("call-2", &leak, "leak"),
                write_file_round("call-3", &later, "later"),
                text_round("done"),
            ]),
        );
        followup
            .execute(
                serde_json::json!({ "id": agent_id.to_string(), "prompt": "write all three" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("the follow-up runs");

        assert!(
            !beside.exists(),
            "the parent's directory is still outside the worker's boundary on a follow-up"
        );
        assert!(
            !leak.exists(),
            "and so is a root the parent holds: the bounds come off the spec, not the parent"
        );
        assert_eq!(
            std::fs::read_to_string(&later).expect("the worker's own root is still writable"),
            "later"
        );
        let transcript = format!("{:?}", store.load_events(agent_id).await.expect("events"));
        assert!(transcript.contains("outside the workspace"), "{transcript}");
    }

    /// The bounds are put back through the acceptor against the parent's reach *now*. A parent
    /// that has moved out from over them, or dropped below `workspace`, is refused rather than
    /// resuming a writer it could not spawn today, and the worker's conversation is left alone.
    #[tokio::test]
    async fn a_followup_on_a_bounded_worker_is_refused_once_the_parent_cannot_grant_the_bounds() {
        let dirs = workspaces();
        let store = store_for_test().await;
        let (params, agent_id) = spawn_bounded_to_sub(&store, &dirs).await;
        let events_before = store.load_events(agent_id).await.expect("events").len();
        let ask = || serde_json::json!({ "id": agent_id.to_string(), "prompt": "carry on" });

        // The parent `/cd`s to a directory that does not contain the worker's root.
        params.cells.cwd.set(dirs.elsewhere.clone());
        let followup = followup_tool_for(params.clone(), mock(vec![text_round("never runs")]));
        let refusal = followup
            .execute(
                ask(),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a parent whose workspace no longer contains the bounds cannot resume")
            .to_string();
        assert!(
            refusal.contains("can no longer grant")
                && refusal.contains("outside this session's workspace"),
            "{refusal}"
        );

        // Back over the root, but dropped to `read`, as `/permission` or Shift+Tab would.
        params.cells.cwd.set(dirs.work.clone());
        params.cells.permission.set_unchecked(Permission::Read);
        let followup = followup_tool_for(params, mock(vec![text_round("never runs")]));
        let refusal = followup
            .execute(
                ask(),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a parent below `workspace` cannot resume a writing worker")
            .to_string();
        assert!(
            refusal.contains("needs a parent at `workspace` or `unrestricted`"),
            "{refusal}"
        );

        assert_eq!(
            store.load_events(agent_id).await.expect("events").len(),
            events_before,
            "neither refusal may have run a turn"
        );
    }

    /// The two follow-up doors judge a parent's move the same way. A plain worker keeps the
    /// directory it was spawned in, and once that directory lies outside the parent's reach at
    /// `workspace` the follow-up is refused as a bounded worker's is, rather than writing where the
    /// parent itself no longer can. An `unrestricted` parent reaches everywhere and is unaffected.
    #[tokio::test]
    async fn a_followup_on_a_plain_worker_is_refused_once_the_parent_has_moved_out_from_over_it() {
        let dirs = workspaces();
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let params = params_at(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Permission::Workspace,
            dirs.work.clone(),
        );
        let spawn = spawn_tool_for(params.clone(), mock(vec![text_round("spawned")]));
        let output = spawn
            .execute(
                serde_json::json!({ "prompt": "wait" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        let agent_id = agent_id_in(&output.text_content());
        let events_before = store.load_events(agent_id).await.expect("events").len();
        let leak = dirs.work.join("leak.txt");
        let ask = || serde_json::json!({ "id": agent_id.to_string(), "prompt": "write it" });

        // The parent `/cd`s to a directory that does not contain the worker's.
        params.cells.cwd.set(dirs.elsewhere.clone());
        let followup = followup_tool_for(
            params.clone(),
            mock(vec![
                write_file_round("call-1", &leak, "leak"),
                text_round("never runs"),
            ]),
        );
        let refusal = followup
            .execute(
                ask(),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a parent that has moved out from over the worker cannot resume it")
            .to_string();
        assert!(
            refusal.contains("can no longer grant")
                && refusal.contains("outside this session's workspace"),
            "{refusal}"
        );
        assert!(
            !leak.exists(),
            "the refused follow-up must not have written"
        );
        assert_eq!(
            store.load_events(agent_id).await.expect("events").len(),
            events_before,
            "the refusal may not have run a turn"
        );

        // The control: an `unrestricted` parent may write anywhere, so its move bounds nothing.
        params
            .cells
            .permission
            .set_unchecked(Permission::Unrestricted);
        let followup = followup_tool_for(params, mock(vec![text_round("resumed")]));
        followup
            .execute(
                ask(),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("an unrestricted parent resumes the worker wherever it has moved");
    }

    /// A worker answers its parent's prompt, so every request it makes carries the prompt id the
    /// spawning call was handed, on the spawn and again on a follow-up. The Claude subscription
    /// backend bills by it, and nothing in the worker's answer shows whether it arrived.
    #[tokio::test]
    async fn a_workers_requests_carry_the_prompt_id_of_the_call_that_spawned_it() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        let provider = mock(vec![text_round("spawned"), text_round("followed up")]);
        let call = |prompt_id: Uuid| crate::tools::ToolContext {
            session_id: None,
            tool_call_id: None,
            prompt_id: Some(prompt_id),
            frontend: Arc::new(crate::frontend::SilentFrontend),
            cancellation: CancellationToken::new(),
        };

        let spawning_prompt = Uuid::new_v4();
        let spawn = spawn_tool_for(params.clone(), provider.clone());
        let output = spawn
            .execute(
                serde_json::json!({ "prompt": "look into it", "permission": "read" }),
                call(spawning_prompt),
            )
            .await
            .expect("spawn succeeds");
        let agent_id = agent_id_in(&output.text_content());
        assert_eq!(
            provider.completion_prompt_ids(),
            vec![Some(spawning_prompt)],
            "the spawned worker's request must bill to the prompt that spawned it"
        );

        let followup_prompt = Uuid::new_v4();
        let followup = followup_tool_for(params, provider.clone());
        followup
            .execute(
                serde_json::json!({ "id": agent_id.to_string(), "prompt": "and then?" }),
                call(followup_prompt),
            )
            .await
            .expect("the follow-up runs");
        assert_eq!(
            provider.completion_prompt_ids(),
            vec![Some(spawning_prompt), Some(followup_prompt)],
            "a follow-up's request must bill to the prompt that asked for it"
        );
    }

    /// The row answers for a worker wherever a row is read (`meka session show`, `GET
    /// /v1/sessions`, an export), so after a follow-up under a tightened parent it records the
    /// level the worker ran at, not the one it was spawned at. The recorded spawn terms are what
    /// keep the grant, so the row is free to follow the live answer.
    #[tokio::test]
    async fn a_followup_records_the_level_the_worker_ran_at_on_its_row() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let params = params_at(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Permission::Unrestricted,
            crate::workspace::cwd_for_test().get(),
        );
        let spawn = spawn_tool_for(params.clone(), mock(vec![text_round("spawned")]));
        let output = spawn
            .execute(
                serde_json::json!({ "prompt": "wait" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        let agent_id = agent_id_in(&output.text_content());
        let recorded_level = |rows: Vec<crate::store::SessionMetaRow>| {
            rows.into_iter()
                .find(|row| row.id == agent_id)
                .expect("the worker's row")
                .permission
        };
        assert_eq!(
            recorded_level(store.load_session_tree(parent_sid).await.expect("tree")),
            Some(Permission::Unrestricted),
            "spawned at the parent's level"
        );

        // The session is then restricted, as `/permission` or Shift+Tab would.
        params.cells.permission.set_unchecked(Permission::Read);
        let followup = followup_tool_for(params, mock(vec![text_round("resumed")]));
        followup
            .execute(
                serde_json::json!({ "id": agent_id.to_string(), "prompt": "carry on" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("the follow-up runs");
        assert_eq!(
            recorded_level(store.load_session_tree(parent_sid).await.expect("tree")),
            Some(Permission::Read),
            "the row must say what the worker ran at"
        );
    }

    /// Two parallel follow-ups on one worker would hydrate the same event log and append to it
    /// independently, so the second's view of the conversation is already stale when it starts. The
    /// guard refuses rather than interleaving.
    #[test]
    fn followup_guard_admits_one_holder_at_a_time() {
        let in_flight: InFlightFollowups =
            Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
        let agent = Uuid::new_v4();
        let other = Uuid::new_v4();

        let first = FollowupGuard::claim(&in_flight, agent).expect("first claim succeeds");
        assert!(
            FollowupGuard::claim(&in_flight, agent).is_none(),
            "a second follow-up on the same worker must be refused"
        );
        // A different worker is unaffected: the guard is per-agent, not a global lock.
        let sibling = FollowupGuard::claim(&in_flight, other);
        assert!(sibling.is_some());

        // Released on drop, so a turn that errors or is canceled doesn't strand the worker.
        drop(first);
        assert!(FollowupGuard::claim(&in_flight, agent).is_some());
    }
}
