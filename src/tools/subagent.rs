//! `agent_spawn` tool: delegates a self-contained research/exploration task to a fresh sub-agent
//! with its own conversation, returning the sub-agent's final report as a single tool result.

use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;
use uuid::Uuid;

use super::{
    Tool, ToolOutput, ToolRegistry, ToolScope,
    util::{optional_str, string_or_list},
};
use crate::{
    agent::Agent,
    config::{BuiltinToolFilter, InstructionAccess, MemoryAccess},
    conversation::Conversation,
    error::{MekaError, Result},
    permission::{EnabledPermissions, Permission, SharedPermission},
    provider::ToolDefinition,
    session::AgentOptions,
    workspace::{
        ParentReach, SharedCwd, SharedRoots, WorkerScope, accept_worker_scope,
        admit_within_parent_reach,
    },
};

/// Hard ceiling on sub-agent nesting depth, independent of the tunable `[subagents].max_depth`
/// budget. Guarantees recursion always terminates even if an agent re-grants `max_depth` at every
/// level: no agent nested deeper than this is given a `agent_spawn` tool, so the tree can never
/// exceed this height.
const SUBAGENT_ABSOLUTE_MAX_DEPTH: usize = 16;

/// What a spawning agent hands its workers: the session's materials, its own live cells, and the
/// ceilings a worker may not exceed.
#[derive(Clone)]
pub(crate) struct ToolBuilderParams {
    pub(crate) materials: crate::session::SessionMaterials,
    /// The spawning agent's own cells. `cells.profile` is the profile the parent names *now*,
    /// which is what a worker whose spawn named no profile runs on; read at spawn time rather
    /// than at registration, because a worker spawned after a `/profile` switch went to the
    /// profile the user had just left, billing that account, while its own row recorded the
    /// new one. `cells.cwd` is snapshotted per spawn so a parent `/cd` mid-turn cannot move a
    /// running worker; the roots are shared, because nothing mutates them after construction. A
    /// worker whose spawn stated a scope takes neither: its directory and roots are its own.
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
    pub(crate) config_scope: ToolScope,
    /// The spawning agent's options, from which a worker inherits `sandboxed_shell` and the
    /// auto-compaction settings inside [`Agent::new_subagent`]. `user_instructions` is
    /// deliberately *not* among them; see [`build_subagent_system_prompt`].
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
/// What is deliberately *not* here: the task (already in the event log), and the cwd of a bare
/// worker, which works where its parent did (already on the session row). A worker whose spawn
/// stated a scope records the scope here as well as on the row, because the row cannot say
/// whether an empty directory list means "the parent's scope" or "the working directory alone".
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
    /// The allow-lists in force, sorted, or `None` for a side the spawn left unrestricted, which
    /// is what a spec that lost the key reads as too: the one default here that is not the
    /// restrictive one, and safe because an allow-list only ever narrows what the deny lists
    /// leave, so a lost key widens nothing past them.
    #[serde(default)]
    pub(crate) allowed_servers: Option<Vec<String>>,
    #[serde(default)]
    pub(crate) allowed_tools: Option<Vec<String>>,
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
    /// Where the worker works, when its spawn stated a scope: the working directory the call
    /// named, canonical. `None` for a bare worker, which works where its parent did and follows
    /// the parent's scope on every follow-up. Not a restrictive default, and the one field where
    /// none exists: a spec that lost it is a bare worker, which is the scope its spawn call gave
    /// it when it stated nothing, and a list left behind without it is refused by
    /// [`Self::stated_scope`] rather than read as bare. Absent on the wire for a bare worker, as
    /// `profile` is, so the document does not say a choice was made when none was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) working_directory: Option<PathBuf>,
    /// The rest of a stated scope's workspace, canonical and merged; empty for a bare worker and
    /// for a scope that named none, which is the restrictive reading of a lost field.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) additional_workspace_directories: Vec<PathBuf>,
    /// What a stated scope reads beyond its workspace, canonical and merged; empty for a bare
    /// worker and for a scope that named none, the restrictive reading of a lost field.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) additional_readable_directories: Vec<PathBuf>,
    /// What the worker's shell and `web_fetch` may reach, never wider than its parent's at the
    /// spawn, and narrowed to the parent's setting now on a follow-up. A spec that lost the field
    /// holds `none`, the restrictive reading.
    #[serde(default = "crate::config::Network::none")]
    pub(crate) network: crate::config::Network,
    /// The profile the spawn call chose, when it chose one. Absent for a worker that runs on what
    /// its parent runs on now and follows it across a switch; named, it pins the worker to that
    /// profile on every follow-up. Absent on the wire too, so the document does not say whether
    /// the choice was open when nothing was chosen.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) profile: Option<String>,
}

impl SubagentSpec {
    /// Whether the spawn call named a profile. The one predicate for "does this worker hold to a
    /// profile of its own", read by the follow-up that rebuilds it and by the cells its children
    /// follow, so the two cannot answer differently.
    pub(crate) fn is_pinned(&self) -> bool {
        self.profile.is_some()
    }

    /// The scope the spawn call stated, or `None` for a bare worker that follows its parent's,
    /// by the one rule the call itself is judged by ([`WorkerScope::from_fields`]): a spec that
    /// lost its working directory beside a list is refused rather than read as bare, since a lost
    /// field must cost authority rather than grant it.
    pub(crate) fn stated_scope(&self) -> std::result::Result<Option<WorkerScope>, String> {
        WorkerScope::from_fields(
            self.working_directory.clone(),
            self.additional_workspace_directories.clone(),
            self.additional_readable_directories.clone(),
        )
        .map_err(|_| "the spec names directories without a working directory".to_string())
    }

    fn tool_scope(&self) -> ToolScope {
        ToolScope::new(self.denied_servers.clone(), self.denied_tools.clone())
            .allowing(self.allowed_servers.clone(), self.allowed_tools.clone())
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
        let own = if self.additional_workspace_directories.is_empty()
            && self.additional_readable_directories.is_empty()
        {
            self.permission
        } else {
            // A list confines, and `workspace` is the highest level that is confined: `agent_spawn`
            // refuses a list beside `unrestricted`, but a spec is persisted JSON that `meka session
            // import` writes verbatim, so the cap is applied where the level is consumed, as
            // `granted_memory` does for the store.
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
    pub(crate) inherited_scope: ToolScope,
    /// Soft, agent-tunable recursion budget for sub-agents spawned by this tool. The root tool is
    /// seeded from `[subagents].max_depth`; each level hands the child `remaining_depth - 1`
    /// unless the caller overrides it via the `max_depth` param. A nested `agent_spawn` is granted
    /// only while the child's budget is `>= 1`.
    pub(crate) remaining_depth: usize,
    /// Monotonic absolute nesting depth of the agent holding this tool (root = 0). Unlike
    /// `remaining_depth` it can't be reset by `max_depth`, so it bounds real recursion at
    /// [`SUBAGENT_ABSOLUTE_MAX_DEPTH`] regardless of what the agent requests.
    pub(crate) absolute_depth: usize,
}

/// The profiles `agent_spawn` offers under `materials`: every configured one when
/// `[subagents].agent_chosen_profile` is on, none otherwise. One definition for the root's tool
/// and every nested one, so the rule is the same at every depth.
pub(crate) fn profile_choices(materials: &crate::session::SessionMaterials) -> Vec<String> {
    if materials.subagents.agent_chosen_profile {
        materials.providers.profile_names()
    } else {
        Vec::new()
    }
}

/// `agent_spawn`'s schema, as a free function so a caller holding no tool can still name it.
/// `profile_choices` is what the `profile` parameter offers; empty, the parameter is absent.
pub(crate) fn agent_spawn_definition(profile_choices: &[String]) -> ToolDefinition {
    let mut description = "Assign a self-contained task to a sub-agent and receive its report, \
        which opens with its id and the scope it holds. It has a private conversation, checklist \
        and scratchpad; filesystem changes are shared. Supply `prompt`, `skill`, or both. Grant \
        context explicitly with `inherit_scratchpad`, `memory` or `instructions`. Independent \
        calls run concurrently; assign separate work to avoid conflicting edits. Omit \
        `working_directory` for your whole scope. Nothing here can exceed your own level, scope, \
        network or tools."
        .to_string();
    let mut parameters = serde_json::json!({
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Self-contained task, scope, and expected result. The parent conversation is not copied. Optional when `skill` is given."
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
                "description": "Parent scratchpad entry names to grant read-only. Missing names are skipped; inherited names cannot be overwritten."
            },
            "permission": {
                "type": "string",
                "enum": ["none", "read", "workspace", "unrestricted"],
                "description": "Permission level for the sub-agent, not its scope, which \
                                `working_directory` states. Defaults to yours; a lower one \
                                narrows it, one above yours is refused."
            },
            "network": {
                "type": "string",
                "enum": crate::config::Network::ALL.map(crate::config::Network::name),
                "description": "What the sub-agent's shell and `web_fetch` may reach. Defaults \
                                to your own; `remote` (other machines, not this one) and `none` \
                                narrow it and need a sandboxed level. An MCP server keeps its \
                                own network: withhold it with `allow_servers` or `deny_servers`."
            },
            "working_directory": {
                "type": "string",
                "description": "Where the sub-agent works: its shell's directory and the base \
                                of its relative paths; writable at `workspace`, read-only at \
                                `read`. Relative to your working directory; \".\" is yours. \
                                Omitted, the sub-agent has your whole scope. Named, the call \
                                states its whole scope: this directory, the two lists below, \
                                and nothing else of yours. Required beside either list."
            },
            "additional_workspace_directories": {
                "type": "array",
                "items": { "type": "string" },
                "description": "More of the sub-agent's workspace, with the working directory's \
                                rights, searched by default. Relative to your working \
                                directory. Needs `working_directory`."
            },
            "additional_readable_directories": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Directories the sub-agent may read beyond its workspace, never \
                                write, searched only when named. Relative to your working \
                                directory. Needs `working_directory`."
            },
            "memory": {
                "type": "string",
                "enum": ["none", "read"],
                "default": "none",
                "description": "Read access to saved memories. Default none. Sub-agents cannot write memories."
            },
            "instructions": {
                "type": "string",
                "enum": ["none", "inherit"],
                "default": "none",
                "description": "`inherit` gives the sub-agent the installation's standing instructions. Default none."
            },
            "allow_servers": {
                "type": "array",
                "items": { "type": "string" },
                "description": "MCP servers the sub-agent may see, and no other; one outside \
                                the list is withheld whole, its tools, resources and prompts. \
                                Built-in tools are unaffected. A server `deny_servers` names \
                                is withheld even if listed here. An empty list is refused; \
                                omit it to withhold no server."
            },
            "allow_tools": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Exactly the tools the sub-agent sees, named as in your own tool \
                                list; every other tool, built-in or MCP, is withheld. A tool \
                                `deny_tools` names, or one from a server outside \
                                `allow_servers`, is withheld even if listed here. An empty \
                                list is refused; omit it to withhold no tool."
            },
            "deny_servers": {
                "type": "array",
                "items": { "type": "string" },
                "description": "MCP servers the sub-agent must not see; each is withheld \
                                whole, its tools, resources and prompts."
            },
            "deny_tools": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Tools the sub-agent must not see, named as in your own tool \
                                list. For a whole server, use `deny_servers`."
            },
            "max_depth": {
                "type": "integer",
                "minimum": 0,
                "description": "How many further levels of sub-agents this one may spawn. \
                                Defaults to one less than your own remaining budget; 0 forbids \
                                spawning; a built-in cap still bounds it."
            }
        }
    });
    if !profile_choices.is_empty() {
        description.push_str(
            " Pass `profile` to run the sub-agent on another configured profile; it keeps that \
             profile on every follow-up.",
        );
        if let Some(properties) = parameters
            .get_mut("properties")
            .and_then(serde_json::Value::as_object_mut)
        {
            properties.insert(
                "profile".into(),
                serde_json::json!({
                    "type": "string",
                    "enum": profile_choices,
                    "description": "The profile the sub-agent runs on. Defaults to your own."
                }),
            );
        }
    }
    ToolDefinition {
        name: "agent_spawn".to_string(),
        description,
        parameters,
        ..Default::default()
    }
}

#[async_trait]
impl Tool for AgentSpawnTool {
    fn definition(&self) -> ToolDefinition {
        agent_spawn_definition(&profile_choices(&self.tool_builder_params.materials))
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
        let prompt =
            optional_str(&input, "prompt", "agent_spawn")?.map(|text| text.trim().to_string());
        let skill_name =
            optional_str(&input, "skill", "agent_spawn")?.map(|text| text.trim().to_string());
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
        let inherited_scratchpad = name_list(&input, "inherit_scratchpad")?.unwrap_or_default();

        // Resolve the sub-agent's permission: an optional `permission` param clamped to the
        // parent's level as a ceiling (restrict-only, never escalate); absent keeps the parent's
        // level. Approval prompts route through `PermissionForwardingFrontend` so they surface in
        // the parent's UI.
        let requested_permission = parse_subagent_permission(
            optional_str(&input, "permission", "agent_spawn")?.map(str::trim),
        )?;
        let parent_level = self.parent_permission.get();

        // The profile the worker runs on, when the call names one. Judged here, ahead of every side
        // effect: with the choice closed the parameter is not in the schema, so a value is a model
        // reaching for something it was not offered, and it is refused rather than read as absent.
        // Whether the name resolves is asked by `worker_binding` once the spec exists, still ahead
        // of the row.
        let chosen_profile = optional_str(&input, "profile", "agent_spawn")?.map(str::to_string);
        if let Some(name) = &chosen_profile
            && !self
                .tool_builder_params
                .materials
                .subagents
                .agent_chosen_profile
        {
            return Err(MekaError::ToolExecution {
                tool_name: "agent_spawn".to_string(),
                message: format!(
                    "`profile` ('{name}') is refused while `[subagents].agent_chosen_profile` is \
                     off; spawn without it to run the sub-agent on this session's profile"
                ),
            });
        }

        // The scope the call states, parsed ahead of every side effect and refused whole when a
        // list stands beside no working directory: a list is a shaped scope, and a shaped scope
        // says where the worker works. Absent, the worker has this agent's whole scope.
        let requested_scope = requested_worker_scope(&input)?;
        let sub_perm = resolve_subagent_permission(requested_permission, parent_level)?;
        let network = resolve_subagent_network(
            optional_str(&input, "network", "agent_spawn")?.map(str::trim),
            self.tool_builder_params.cells.network,
            sub_perm,
        )?;
        // Judged by the one acceptor `agent_followup` also uses, at the level the worker will run
        // and against this agent's reach now, so a refusal here leaves nothing behind.
        let scope = match &requested_scope {
            Some(requested) => {
                let cells = &self.tool_builder_params.cells;
                let reads = cells.site().read_scope();
                let parent = ParentReach {
                    level: parent_level,
                    cwd: &cells.cwd,
                    roots: &cells.roots,
                    reads: &reads,
                };
                Some(
                    accept_worker_scope(&parent, sub_perm, requested).map_err(|error| {
                        MekaError::ToolExecution {
                            tool_name: "agent_spawn".to_string(),
                            message: error.to_string(),
                        }
                    })?,
                )
            }
            None => None,
        };

        // Narrow, never replace: the call site's allow-lists intersect with, and its deny lists
        // add to, what config (and, when nested, this agent's own parent) already set, so neither
        // kind of list can hand a worker something the installation or an ancestor took away. An
        // allow-list is a present list accepted whole or refused whole: `[]` is refused rather
        // than read as "everything", which is what config reads an empty one as, or as "nothing",
        // which a model emitting `[]` for "nothing to add" never meant.
        let allow_list = |key: &'static str, noun: &str| -> Result<Option<Vec<String>>> {
            match name_list(&input, key)? {
                None => Ok(None),
                Some(names) if names.is_empty() => Err(MekaError::ToolExecution {
                    tool_name: "agent_spawn".to_string(),
                    message: format!("{key} names no {noun}; leave it out to withhold none"),
                }),
                Some(names) => Ok(Some(names)),
            }
        };
        let call_site_scope = ToolScope::new(
            name_list(&input, "deny_servers")?.unwrap_or_default(),
            name_list(&input, "deny_tools")?.unwrap_or_default(),
        )
        .allowing(
            allow_list("allow_servers", "server")?,
            allow_list("allow_tools", "tool")?,
        );
        // A name that matches nothing restricts nothing, and the model gets no signal either way:
        // it asked for a narrowed worker and would receive a wider one believing otherwise. Warned
        // rather than refused, because "deny it if it is there" is a legitimate thing to write
        // against a server list that varies by machine.
        // With no MCP manager attached no server is configured, so every namespaced name is stale
        // and a built-in name is still checked against the built-ins.
        let configured_servers = self
            .tool_builder_params
            .materials
            .mcp_manager
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .map(|manager| manager.server_names())
            .unwrap_or_default();
        {
            for (list, name) in
                super::registry::stale_scope_entries(&call_site_scope, &configured_servers)
            {
                let (key, what) = match list {
                    super::registry::ScopeList::AllowedServers => {
                        ("allow_servers", "no configured MCP server")
                    }
                    super::registry::ScopeList::DeniedServers => {
                        ("deny_servers", "no configured MCP server")
                    }
                    super::registry::ScopeList::AllowedTools => (
                        "allow_tools",
                        "no built-in tool and no configured MCP server",
                    ),
                    super::registry::ScopeList::DeniedTools => (
                        "deny_tools",
                        "no built-in tool and no configured MCP server",
                    ),
                };
                tracing::warn!(
                    "agent_spawn {key} entry '{name}' matches {what}, so it restricts nothing"
                );
            }
        }
        let effective_scope = self.inherited_scope.narrow(&call_site_scope);

        // Context grants. Both default to nothing and are clamped against what this agent itself
        // holds, so a worker can never hand a grandchild more than it was given. Unlike the deny
        // lists, these are grants rather than restrictions: config cannot meaningfully withhold
        // them (a parent holding the text can copy it into the prompt), so the decision is the
        // parent's, and the safe state is the default.
        let memory_access = match optional_str(&input, "memory", "agent_spawn")?.map(str::trim) {
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
        let instructions = match optional_str(&input, "instructions", "agent_spawn")?.map(str::trim)
        {
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
        // shift a bare worker's path resolution mid-flight. The same value is written to the
        // child's session row and handed to its cells, which its tool registry and its turn's
        // environment context read; a follow-up reads it back off the row. A worker with a stated
        // scope takes its directory and roots from the scope instead, and nothing of the parent's:
        // what it holds is exactly what its row and its spec record.
        let workspace = worker_workspace(
            &self.tool_builder_params.cells,
            scope.as_ref(),
            self.tool_builder_params.cells.cwd.get(),
            network,
        );
        let sub_cwd_snapshot = workspace.cwd.get();
        let sub_additional_roots = scope
            .as_ref()
            .map(|scope| scope.additional_workspace_directories.clone())
            .unwrap_or_default();
        let scope_lines = scope_report(
            &workspace,
            sub_perm,
            crate::sandbox::held_network(
                &self.tool_builder_params.materials.core.sandbox_capability,
                workspace.network,
            ),
        );

        let spec = SubagentSpec {
            permission: sub_perm,
            enabled_permissions: clamp_enabled_permissions(
                self.parent_permission.enabled(),
                sub_perm,
            )
            .iter()
            .collect(),
            denied_servers: effective_scope.denied_server_list(),
            denied_tools: effective_scope.denied_tool_list(),
            allowed_servers: effective_scope.allowed_server_list(),
            allowed_tools: effective_scope.allowed_tool_list(),
            memory: memory_access,
            instructions,
            inherited_scratchpad: inherited_scratchpad.clone(),
            remaining_depth: child_remaining_depth,
            absolute_depth: child_absolute_depth,
            working_directory: scope.as_ref().map(|scope| scope.working_directory.clone()),
            additional_workspace_directories: scope
                .as_ref()
                .map(|scope| scope.additional_workspace_directories.clone())
                .unwrap_or_default(),
            additional_readable_directories: scope
                .as_ref()
                .map(|scope| scope.additional_readable_directories.clone())
                .unwrap_or_default(),
            network,
            profile: chosen_profile,
        };
        // Resolved before the row is written, so a name this installation does not configure, or a
        // profile whose provider cannot be built, refuses the spawn and leaves nothing behind.
        let binding = worker_binding(
            &self.tool_builder_params,
            spec.profile.as_deref(),
            "agent_spawn",
        )
        .await?;
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
                // The same binding `build_subagent` is handed a moment later, so the row this
                // writes and the provider the worker is built on cannot come apart.
                binding.profile.clone(),
                // The parent's live project, which the worker's own cell is seeded from.
                self.tool_builder_params
                    .cells
                    .project
                    .get()
                    .unwrap_or_default(),
            )
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_spawn".to_string(),
                message: format!("failed to create sub-agent session: {error}"),
            })?;
        // Held for the whole of the worker's run, then released with this scope. Unlocked, a
        // sub-agent's row can be taken by a concurrent `meka session remove --all` and its
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
        // A detached spawn is a task row; the row names the session it runs, so a client
        // connects the two without parsing the task's words. Best-effort: the link is
        // bookkeeping, and the run does not wait on it.
        if let Some(task_id) = &context.task_id
            && let Err(error) = self
                .tool_builder_params
                .materials
                .store
                .background_store()
                .record_task_subagent(task_id, sub_session_id)
                .await
        {
            tracing::warn!("failed to record the task's sub-agent: {error}");
        }
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
            WorkerPlacement { workspace, binding },
            parent_sid,
            sub_session_id,
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
        // Registered for the whole of the run, lock or no lock: a sub-agent is never a resident
        // session of any host, so this is what a session record answers `turn_in_flight` from.
        // Taken after the build, which installs the worker's feed, so a record that says the
        // worker is running never names a feed that is not yet there to read.
        let _running = self
            .tool_builder_params
            .materials
            .core
            .running_subagents
            .run(
                sub_session_id,
                crate::session::SubagentRun::Spawned,
                &sub_agent.cells().profile,
            );

        // Run the sub-agent's single turn via the shared `Agent::run_turn` path. Conversation
        // persistence (user message, assistant messages, tool results) happens inside `run_turn`
        // against the sub-session, so the audit trail is identical to the root agent's. Silent
        // rendering and the omitted MCP gate are baked into the options via `new_subagent`.
        let mut messages = Conversation::new();
        // Every request the sub-agent makes is attributed as one (`Attribution::subagent`, from its
        // role), which the Claude OAuth billing header reports as `cc_is_subagent=true;`.
        sub_agent
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts(task, Vec::new()).map_err(|empty| {
                    MekaError::ToolExecution {
                        tool_name: "agent_spawn".to_string(),
                        message: empty.to_string(),
                    }
                })?,
                crate::agent::TurnCancellation::child_of(
                    &cancellation,
                    crate::store::turns::CancelReason::Parent,
                ),
            )
            .await?;

        let report = messages
            .last_assistant_text()
            .unwrap_or_else(|| "(sub-agent produced no final text)".to_string());
        // Lead with the id and the scope the worker holds, so the report stays the tail of the
        // output, where a model reading a long result looks for the conclusion.
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
            format!("Agent: {sub_session_id}\n{scope_lines}\n{report}")
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
        "agent_steer",
        Arc::new(AgentSteerTool {
            tool_builder_params: params.clone(),
        }),
    ));
    tools.push((
        "agent_delete",
        Arc::new(AgentDeleteTool {
            tool_builder_params: params,
            in_flight: delete_in_flight,
        }),
    ));

    // All five or none, gated on `agent_spawn`. The four lifecycle tools only ever operate on what
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
/// impose, named once so `meka tool list` can answer the question without assembling a session.
/// Per-tool `[tools]` entries are *not* here: denying `agent_list` alone removes only that one, and
/// the caller applies [`BuiltinToolFilter::admits`] per name on top of this.
///
/// Takes the filter rather than a [`ToolRegistry`] because the listing builds its reference
/// registry unfiltered on purpose, to read every tool's hardcoded permission level; asking that
/// registry would answer "yes" no matter what the user configured.
pub(crate) fn agent_tools_registered(filter: &BuiltinToolFilter, max_depth: usize) -> bool {
    max_depth >= 1 && filter.admits("agent_spawn")
}

/// The family [`agent_tools_registered`] decides about, for a listing that shows it as denied.
pub(crate) const AGENT_TOOL_NAMES: [&str; 5] = [
    "agent_spawn",
    "agent_list",
    "agent_followup",
    "agent_steer",
    "agent_delete",
];

/// The sub-agents this session spawned: the rows of its tree that name it as their parent.
/// Grandchildren belong to the worker that spawned them and are left out, so a row this returns is
/// always one the session may drive.
async fn direct_children(
    params: &ToolBuilderParams,
    tool_name: &'static str,
) -> Result<(Uuid, Vec<crate::store::SessionMetaRow>)> {
    let parent_sid = current_session_id(params, tool_name)?;
    let rows = params
        .materials
        .store
        .load_session_tree(parent_sid)
        .await
        .map_err(|error| MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("failed to list sub-agents: {error}"),
        })?;
    Ok((
        parent_sid,
        rows.into_iter()
            .filter(|row| row.parent_id == Some(parent_sid))
            .collect(),
    ))
}

/// Pick the one of `children` that `wanted` names: its full id, or a prefix no other child shares.
///
/// Matching among this session's own sub-agents and nowhere else is what makes the lookup the
/// ownership check. An id that was never a sub-agent (a fabricated or mistyped UUID), a sub-agent
/// belonging to a *different* parent, and a fork holding ids it does not own (`fork_session` copies
/// the conversation, which names the children, but not the children themselves, so a forked
/// parent's log advertises sessions that are still linked to the original) all come out as no such
/// sub-agent, and letting any of them through would let one session drive or delete another's
/// workers. The same scope keeps a prefix honest: another session's worker can neither make it
/// ambiguous nor be what it resolves to.
fn match_child(
    children: Vec<crate::store::SessionMetaRow>,
    wanted: &str,
    tool_name: &'static str,
) -> Result<crate::store::SessionMetaRow> {
    let wanted = wanted.trim();
    let mut matches: Vec<crate::store::SessionMetaRow> = if let Ok(id) = Uuid::parse_str(wanted) {
        children.into_iter().filter(|row| row.id == id).collect()
    } else if crate::text::is_usable_id_prefix(wanted) {
        let prefix = crate::text::id_prefix_for_matching(wanted);
        children
            .into_iter()
            .filter(|row| row.id.to_string().starts_with(&prefix))
            .collect()
    } else {
        Vec::new()
    };
    if matches.len() > 1 {
        let listing: Vec<String> = matches.iter().map(|row| row.id.to_string()).collect();
        return Err(MekaError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!(
                "'{wanted}' names {} of this session's sub-agents: {}. Pass more of the id.",
                matches.len(),
                listing.join(", ")
            ),
        });
    }
    matches.pop().ok_or_else(|| MekaError::ToolExecution {
        tool_name: tool_name.to_string(),
        message: format!(
            "no sub-agent '{wanted}' belongs to this session. Use `agent_list` to see the ones that \
             do."
        ),
    })
}

/// Resolve `wanted` to one of this session's sub-agents, refusing anything else.
async fn resolve_child_session(
    params: &ToolBuilderParams,
    tool_name: &'static str,
    wanted: &str,
) -> Result<(Uuid, crate::store::SessionMetaRow)> {
    let (parent_sid, children) = direct_children(params, tool_name).await?;
    Ok((parent_sid, match_child(children, wanted, tool_name)?))
}

/// [`resolve_child_session`] for the `id` argument.
async fn require_child_session(
    params: &ToolBuilderParams,
    tool_name: &'static str,
    input: &serde_json::Value,
) -> Result<(Uuid, crate::store::SessionMetaRow)> {
    let raw = super::util::require_str(input, "id", tool_name)?;
    resolve_child_session(params, tool_name, &raw).await
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
                      `agent_followup` to ask it another question, to `agent_steer` to send it \
                      a message without waiting, or to `agent_delete` to discard it and free \
                      what it held."
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
            // rounds as five turns. A turn is a user message that opens one, by
            // `Message::opens_turn`: a steer that rides a round's results is part of that round.
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
                            crate::conversation::Event::Append(message) => message.opens_turn(),
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

/// Puts a message in a sub-agent's inbox without waiting on it.
///
/// The child runs the same loop as its parent, so the message is read where any inbox item is:
/// at the child's next round boundary while it runs, or at the opening of its next follow-up
/// once it has finished. Which of the two happens is not this tool's to know, and it does not
/// need to: nothing is stranded either way.
pub(crate) struct AgentSteerTool {
    pub(crate) tool_builder_params: ToolBuilderParams,
}

/// `agent_steer`'s schema. Free-standing for the reason [`agent_spawn_definition`] is.
pub(crate) fn agent_steer_definition() -> ToolDefinition {
    ToolDefinition {
        name: "agent_steer".to_string(),
        description: "Send a message to a sub-agent you already spawned, without waiting for it: \
                      a correction, a change of course, something it should know. A sub-agent \
                      that is running reads it at its next step, or at once with `interrupt`; \
                      one that has finished reads it at the start of your next `agent_followup` \
                      to it. Returns at once, with no answer. Use `agent_followup` when you want \
                      the sub-agent's reply."
            .to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The sub-agent's id, as returned by `agent_spawn` or \
                                    `agent_list`; a unique prefix is enough."
                },
                "message": {
                    "type": "string",
                    "description": "What to tell it."
                },
                "interrupt": {
                    "type": "boolean",
                    "description": "Cut the answer the sub-agent is writing so it reads this \
                                    now rather than after its current step; a tool it is running \
                                    still finishes first. Default false."
                }
            },
            "required": ["id", "message"]
        }),
        ..Default::default()
    }
}

#[async_trait]
impl Tool for AgentSteerTool {
    fn definition(&self) -> ToolDefinition {
        agent_steer_definition()
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let message = super::util::require_str(&input, "message", "agent_steer")?;
        let interrupt = input
            .get("interrupt")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let (_parent_sid, row) =
            require_child_session(&self.tool_builder_params, "agent_steer", &input).await?;
        let class = if interrupt {
            crate::store::inbox::InboxClass::Interrupt
        } else {
            crate::store::inbox::InboxClass::Steer
        };
        let item = crate::store::inbox::NewInboxItem::from_parts(
            row.id,
            class,
            Some("your parent agent".to_string()),
            message,
        )?;
        self.tool_builder_params
            .materials
            .store
            .inbox_store()
            .enqueue(item)
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_steer".to_string(),
                message: format!("failed to record the message: {error}"),
            })?;
        let when = if interrupt {
            "It stops the answer it is writing to read this, finishes a tool it is running first, \
             or reads it at the start of your next `agent_followup` to it if it has finished."
        } else {
            "It reads this at its next step if it is running, or at the start of your next \
             `agent_followup` to it."
        };
        Ok(ToolOutput::text(
            format!("Queued for sub-agent {}. {when}", row.id),
            false,
        ))
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
    config_scope: &ToolScope,
    memory_access: MemoryAccess,
) -> SubagentSpec {
    let scope = spec.tool_scope().narrow(config_scope);
    SubagentSpec {
        denied_servers: scope.denied_server_list(),
        denied_tools: scope.denied_tool_list(),
        allowed_servers: scope.allowed_server_list(),
        allowed_tools: scope.allowed_tool_list(),
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
                      level, same scope, same restrictions), which your current settings \
                      cannot widen. \
                      Get ids from `agent_spawn`'s result or from `agent_list`."
            .to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The sub-agent's id, as returned by `agent_spawn` or \
                                    `agent_list`; a unique prefix is enough."
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
        let agent_id = require_child_session(&self.tool_builder_params, "agent_followup", &input)
            .await?
            .1
            .id;
        let Some(_guard) = FollowupGuard::claim(&self.in_flight, agent_id) else {
            return Err(MekaError::ToolExecution {
                tool_name: "agent_followup".to_string(),
                message: format!(
                    "sub-agent '{agent_id}' is busy. Wait for the call already running against it to \
                     return before asking again."
                ),
            });
        };
        // Resolving the id awaited, and a concurrent `agent_delete` finishing inside that window
        // would leave this turn writing to a session that no longer exists. Asked again under the
        // guard, by the full id, the row is either still there or the refusal is honest.
        let (parent_sid, row) = resolve_child_session(
            &self.tool_builder_params,
            "agent_followup",
            &agent_id.to_string(),
        )
        .await?;

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
            &self.tool_builder_params.config_scope,
            self.tool_builder_params.memory_access,
        );

        let ceiling = self.parent_permission.get();
        let effective_permission = spec.effective_permission(ceiling);
        // Narrowed to the parent's setting now, as the level is: a `none` the installation has
        // since adopted reaches a worker spawned under `host`, and never the other way.
        let network = spec
            .network
            .narrowest(self.tool_builder_params.cells.network);
        // The worker's own cwd, as recorded when it was spawned, not the parent's current one: a
        // `/cd` between the spawn and the follow-up must not move a worker mid-task. A worker
        // with a stated scope takes its directory and roots off its spawn terms instead. Either
        // way what it holds goes back through the acceptor against the parent's reach *now*, at
        // the level the worker runs at now: a parent that has moved to a directory that no longer
        // contains it is refused rather than resuming a writer it could not spawn today, the way
        // `effective_permission` narrows the worker to what the parent currently holds. One rule
        // for both doors, or a bare worker keeps writing under the very directory a shaped one is
        // refused.
        let cells = &self.tool_builder_params.cells;
        let reads = cells.site().read_scope();
        let parent = ParentReach {
            level: ceiling,
            cwd: &cells.cwd,
            roots: &cells.roots,
            reads: &reads,
        };
        let recorded = spec
            .stated_scope()
            .map_err(|reason| MekaError::ToolExecution {
                tool_name: "agent_followup".to_string(),
                message: format!(
                    "sub-agent '{agent_id}' has spawn terms this meka cannot read: {reason}"
                ),
            })?;
        let workspace = match &recorded {
            None => {
                let cwd = row
                    .cwd
                    .as_deref()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| cells.cwd.get());
                admit_within_parent_reach(
                    &parent,
                    effective_permission == Permission::Workspace,
                    &cwd,
                    &cwd,
                )
                .map_err(|error| MekaError::ToolExecution {
                    tool_name: "agent_followup".to_string(),
                    message: format!(
                        "sub-agent '{agent_id}' works in a directory this session can no longer \
                         grant: {error}"
                    ),
                })?;
                worker_workspace(cells, None, cwd, network)
            }
            Some(recorded) => {
                let accepted = accept_worker_scope(&parent, effective_permission, recorded)
                    .map_err(|error| MekaError::ToolExecution {
                        tool_name: "agent_followup".to_string(),
                        message: format!(
                            "sub-agent '{agent_id}' was spawned with a scope this session can no \
                             longer grant: {error}"
                        ),
                    })?;
                let cwd = accepted.working_directory.clone();
                worker_workspace(cells, Some(&accepted), cwd, network)
            }
        };
        let scope_lines = scope_report(
            &workspace,
            effective_permission,
            crate::sandbox::held_network(
                &self.tool_builder_params.materials.core.sandbox_capability,
                workspace.network,
            ),
        );

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

        // A pinned worker runs on what its row names, which its spawn wrote and only an explicit
        // act has moved since. Read here rather than off the spec so such an act is
        // honored.
        let pinned = if spec.is_pinned() {
            Some(
                self.tool_builder_params
                    .materials
                    .store
                    .recorded_profile(agent_id)
                    .await
                    .map_err(|error| MekaError::ToolExecution {
                        tool_name: "agent_followup".to_string(),
                        message: format!(
                            "failed to read the profile of sub-agent '{agent_id}': {error}"
                        ),
                    })?
                    .ok_or_else(|| MekaError::ToolExecution {
                        tool_name: "agent_followup".to_string(),
                        message: format!(
                            "sub-agent '{agent_id}' records no profile, so it cannot be resumed \
                             safely. Spawn a new one."
                        ),
                    })?,
            )
        } else {
            None
        };
        // Held for this turn, as `agent_spawn` holds it for the spawn. A follow-up runs a full turn
        // against a row nothing else claims, so without this the worker would sit unlocked for
        // seconds to minutes and a concurrent `meka session remove --all` could take it and cascade
        // the conversation away mid-run.
        //
        // Taken ahead of the build and the hydration below, so the conversation this turn loads is
        // the one the lock protects: a spawn still finishing would otherwise commit its last events
        // between the load and the lock, and the follow-up would continue from a copy without them.
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
                message: match error {
                    // The holder is a turn on this worker: its spawn or follow-up detached with
                    // `background: true`, still running. The model wants to reach it, and the
                    // door for that is the inbox, not a second turn.
                    MekaError::SessionLocked(_) => format!(
                        "sub-agent '{agent_id}' is still running; reach it with `agent_steer` \
                         until it finishes."
                    ),
                    error => format!("cannot follow up on sub-agent {agent_id}: {error}"),
                },
            })?;
        let binding = worker_binding(
            &self.tool_builder_params,
            pinned.as_deref(),
            "agent_followup",
        )
        .await?;
        let sub_agent = build_subagent(
            &self.tool_builder_params,
            &spec,
            WorkerPlacement { workspace, binding },
            parent_sid,
            agent_id,
            "agent_followup",
            &context,
        )
        .await?;
        // After the build, for the reason `agent_spawn` registers after its own.
        let _running = self
            .tool_builder_params
            .materials
            .core
            .running_subagents
            .run(
                agent_id,
                crate::session::SubagentRun::Resumed,
                &sub_agent.cells().profile,
            );

        // Rehydrate the worker's own conversation through the door every resume uses.
        // `from_events` arms the resume notice, and it is left armed deliberately: every
        // follow-up really is a fresh registry and a fresh read tracker, so the
        // worker is being told something true each time rather than a stale banner.
        let mut messages = self
            .tool_builder_params
            .materials
            .store
            .load_conversation(agent_id)
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_followup".to_string(),
                message: format!("failed to load sub-agent conversation: {error}"),
            })?;

        // The row has to follow the build, for the reason `agent_spawn` writes it from the same
        // binding: an unpinned worker runs on the parent's profile now, and a follow-up after a
        // `/profile` switch would otherwise bill an account the worker's row does not
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
                // The project the same way: the worker reads it off its parent's cell at each
                // follow-up, as it reads the profile, so the row says which tier this turn's
                // memories were filed in.
                project: Some(sub_agent.project().unwrap_or_default()),
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
                crate::agent::TurnInput::from_parts(prompt, Vec::new()).map_err(|empty| {
                    MekaError::ToolExecution {
                        tool_name: "agent_followup".to_string(),
                        message: empty.to_string(),
                    }
                })?,
                crate::agent::TurnCancellation::child_of(
                    &cancellation,
                    crate::store::turns::CancelReason::Parent,
                ),
            )
            .await?;

        let report = messages
            .last_assistant_text()
            .unwrap_or_else(|| "(sub-agent produced no final text)".to_string());
        // Skipped when the caller redirected the report to a scratchpad, for the reason the spawn
        // skips its own header.
        let output = if super::util::redirects_to_scratchpad(&input) {
            report
        } else {
            format!("Agent: {agent_id}\n{scope_lines}\n{report}")
        };
        Ok(ToolOutput::text(output, false))
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
                      scratchpad entries and any sub-agents it spawned in turn. Files it wrote, \
                      and your own conversation and scratchpad, are untouched."
            .to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The sub-agent's id, as returned by `agent_spawn` or \
                                    `agent_list`; a unique prefix is enough."
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
        // Claimed between resolving the id and confirming the row, mirroring `agent_followup`, so
        // the two serialize on one worker whichever arrives first. Parallel tool calls in a single
        // turn make this reachable: without it, a delete completing inside a follow-up's ownership
        // check leaves that follow-up writing to a deleted session and failing on a foreign-key
        // violation, a raw database error, on a path where the model did nothing wrong.
        let agent_id = require_child_session(&self.tool_builder_params, "agent_delete", &input)
            .await?
            .1
            .id;
        let Some(_guard) = FollowupGuard::claim(&self.in_flight, agent_id) else {
            return Err(MekaError::ToolExecution {
                tool_name: "agent_delete".to_string(),
                message: format!(
                    "sub-agent '{agent_id}' is busy. Wait for the call already running against it to \
                     return before deleting it."
                ),
            });
        };
        let (_parent_sid, row) = resolve_child_session(
            &self.tool_builder_params,
            "agent_delete",
            &agent_id.to_string(),
        )
        .await?;
        // One statement; `sessions.parent_session_id`, `messages.session_id` and
        // `scratchpad_entries.session_id` all carry `ON DELETE CASCADE`, so the worker's messages,
        // its scratchpad entries and its own descendants go with it. Through the locked door,
        // because the guard above serializes this against this registry's follow-ups only: a spawn
        // detached with `background: true` holds the worker's own lock while it runs and never
        // enters the guard, and neither does a sub-agent of the worker's own.
        self.tool_builder_params
            .materials
            .store
            .delete_session_tree_unless_attached(row.id)
            .await
            .map_err(|error| MekaError::ToolExecution {
                tool_name: "agent_delete".to_string(),
                message: match error {
                    MekaError::SessionLocked(_) => format!(
                        "sub-agent '{agent_id}' is still running; wait for it to finish before \
                         deleting it."
                    ),
                    error => format!("failed to delete sub-agent: {error}"),
                },
            })?;
        let id = row.id;
        tracing::info!("deleted sub-agent session {id}");
        Ok(ToolOutput::text(format!("deleted agent {}", row.id), false))
    }
}

/// The workspace a worker is built on, resolved by the door that builds it: the parent's own
/// handles for a bare worker, a directory and roots of its own for one whose spawn stated a
/// scope. One value rather than two arguments because the pair is decided together and a worker
/// holding one door's directory with the other's roots is a boundary nobody granted.
struct WorkerWorkspace {
    cwd: SharedCwd,
    roots: SharedRoots,
    /// What the worker reads beyond `roots`; see
    /// [`SubagentSpec::additional_readable_directories`].
    read_only_roots: SharedRoots,
    /// Whether the worker's reads are confined, and to what beyond its roots: the parent's
    /// boundary, or one its scope switched on ([`crate::workspace::ReadBoundary::for_worker`]).
    read_boundary: crate::workspace::ReadBoundary,
    /// What the worker's shell and `web_fetch` may reach; see [`SubagentSpec::network`].
    network: crate::config::Network,
}

/// The workspace a worker holds: its own, from the `scope` its spawn stated, or its parent's
/// handles for a bare worker, with `bare_cwd` as the directory a bare worker works in, the
/// parent's as snapshotted at spawn or as read back off the row on a follow-up. One function for
/// both doors, so neither pairs one scope's directory with the other's roots.
fn worker_workspace(
    cells: &crate::session::SessionCells,
    scope: Option<&WorkerScope>,
    bare_cwd: PathBuf,
    network: crate::config::Network,
) -> WorkerWorkspace {
    match scope {
        Some(scope) => WorkerWorkspace {
            cwd: SharedCwd::new(scope.working_directory.clone()),
            roots: SharedRoots::new(scope.additional_workspace_directories.clone()),
            read_only_roots: SharedRoots::new(scope.additional_readable_directories.clone()),
            read_boundary: cells.read_boundary.for_worker(true),
            network,
        },
        None => WorkerWorkspace {
            cwd: SharedCwd::new(bare_cwd),
            roots: cells.roots.clone(),
            read_only_roots: cells.read_only_roots.clone(),
            read_boundary: cells.read_boundary.for_worker(false),
            network,
        },
    }
}

/// The scope a worker holds, as its parent reads it in the spawn and follow-up result: its level
/// and the environment block's own scope lines, so the parent reads the worker's scope in the
/// words it reads its own. `shell_network` is what the worker's shell holds for its setting.
fn scope_report(
    workspace: &WorkerWorkspace,
    level: Permission,
    shell_network: crate::config::Network,
) -> String {
    let cwd = workspace.cwd.get();
    let roots = workspace.roots.get();
    let readable = crate::workspace::additional_readable_directories(
        workspace.read_boundary.confined,
        level,
        &workspace.cwd,
        &workspace.roots,
        &workspace.read_only_roots,
    );
    format!(
        "Permission: {level}\n{}",
        crate::prompt::render_scope_lines(&crate::prompt::ScopeLines {
            permission: level,
            cwd: &cwd,
            roots: &roots,
            readable: readable.as_deref(),
            network: workspace.network,
            shell_network,
        })
    )
}

/// Where a worker runs: the workspace its door resolved and the profile it is bound to, from
/// [`worker_binding`]. One value because the door decides both before the build, and they describe
/// the same worker.
struct WorkerPlacement {
    workspace: WorkerWorkspace,
    binding: crate::provider::ResolvedProfile,
}

/// What a worker runs on: `pinned`, or the profile its parent names now when there is none.
///
/// One door for `agent_spawn` and `agent_followup`. The spawn passes the profile its call chose;
/// the follow-up passes what a pinned worker's own row names, because a session runs on the profile
/// its row names and that row moves only by an explicit act (an import onto another profile),
/// which a follow-up honors rather than undoes. An unpinned worker's row is not
/// consulted: it is rewritten from the parent after every follow-up, so the parent's live cell is
/// the truth. A name that no longer resolves is refused by name, as resuming a root session on a
/// deleted profile is.
async fn worker_binding(
    params: &ToolBuilderParams,
    pinned: Option<&str>,
    tool_name: &'static str,
) -> Result<crate::provider::ResolvedProfile> {
    match pinned {
        Some(name) => {
            crate::provider::resolved_profile(&params.materials.providers, name.to_string())
                .await
                .map_err(|error| MekaError::ToolExecution {
                    tool_name: tool_name.to_string(),
                    message: format!("cannot run the sub-agent on profile '{name}': {error}"),
                })
        }
        // The named profile, not what serves the parent's turn now: a fallback is one turn's
        // substitution, and the worker's own turn routes by the breaker like any other, so its row
        // names the profile the parent's does and its turn's row says who served it.
        None => Ok(params.cells.profile.named()),
    }
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
    placement: WorkerPlacement,
    parent_session_id: Uuid,
    sub_session_id: Uuid,
    tool_name: &'static str,
    // The call that spawns the worker: its tool-use id, so the worker's own tool calls roll up
    // into that call's display, and the prompt it answers, so the worker's requests bill to it.
    call: &crate::tools::ToolContext,
) -> Result<Agent> {
    let WorkerPlacement { workspace, binding } = placement;
    // The parent's live handle, not a snapshot of its level. Taking a `Permission` here is what
    // let a worker outlive its parent's downgrade: the value was read once and frozen into a fresh
    // atomic, so nothing the user did afterwards could reach the running child.
    let parent_permission = &params.cells.permission;
    // A worker's window comes off `binding` inside `Agent::new_subagent`, not from
    // `params.parent_options`, which was cloned when the session was assembled and cannot hear
    // about a switch. Taking the provider from one and the window from the other would have a
    // worker talk to a 32k profile while auto-compacting against the 1M one the session had left.
    let parent_options = params.parent_options.clone();
    let sub_shared_perm = spec.shared_permission_bounded(parent_permission);
    let tool_scope = spec.tool_scope();
    // Resolved once: it feeds both this worker's system prompt and what its own children can be
    // given. `None` here is what makes nesting self-enforcing: a worker that was not granted the
    // instructions has no copy to pass on, so the restriction propagates through the data rather
    // than through a check every future call site has to remember.
    let memory_access = spec.granted_memory();
    let granted_instructions = match spec.instructions {
        InstructionAccess::Inherit => params.parent_options.user_instructions.clone(),
        InstructionAccess::None => None,
    };
    // A fresh, private checklist so the worker's `checklist_*` calls don't touch the parent's.
    // Empty here; the worker's first turn reads it back from its own conversation, so a
    // follow-up still owes what the worker left open.
    let sub_checklist = crate::checklist::SharedChecklist::default();
    let sub_shared_session_id = crate::session::SharedSessionId::new(Some(sub_session_id));
    // Wrap so permission prompts surface in the parent's UI while emits stay silent (the
    // sub-agent's output flows back as this tool's result, not as live notifications). The one
    // exception is the sub-agent's tool calls, which are rolled up into this call's own display
    // so a long run is not an opaque spinner, hence the tool-call id.
    let sub_frontend: Arc<dyn crate::frontend::Frontend> = params.cells.frontend.for_subagent(
        Arc::clone(&params.cells.frontend),
        sub_session_id,
        call.tool_call_id.clone(),
    );
    // The worker's own cells: its clamped permission, the directory and roots it was handed, a
    // session of its own, fresh gauges, and a profile seeded from its binding. A
    // worker has no prompt gauge and no session entry watching it, and it never switches, so
    // nothing outside needs the handle. The write fence and the shell sandbox are both built from
    // `roots` by `register_core_tools`, so a bounded worker's boundary is what these cells say.
    let sub_cells = crate::session::SessionCells {
        permission: sub_shared_perm.clone(),
        cwd: workspace.cwd,
        roots: workspace.roots,
        read_only_roots: workspace.read_only_roots,
        read_boundary: workspace.read_boundary,
        network: workspace.network,
        session_id: sub_shared_session_id,
        project: crate::session::SharedProject::new(params.cells.project.get()),
        checklist: sub_checklist,
        profile: crate::provider::PublishedProfile::detached(&binding),
        context_tokens: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        context_reserved: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        context_overhead: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        background_tasks: crate::background::BackgroundTasks::default(),
        frontend: sub_frontend,
        // Fresh per worker for the same reason the id is: a lock belongs to the session that
        // was created, and every agent creates at most its own.
        session_lock: crate::store::SessionLockSlot::default(),
        pending_compaction: crate::session::PendingCompaction::default(),
    };

    // A follow-up rebuilds the worker over a row that recorded what the provider last measured
    // for it; a spawn's row recorded nothing yet and leaves the fresh gauge alone.
    sub_cells
        .seed_context_tokens(&params.materials.store, sub_session_id)
        .await;

    let sub_registry = ToolRegistry::build_for_subagent(
        &params.materials,
        &sub_cells,
        super::registry::RegistryScope {
            tool_scope: tool_scope.clone(),
            memory_access,
            parent_session_id: if spec.inherited_scratchpad.is_empty() {
                None
            } else {
                Some(parent_session_id)
            },
            inherited_scratchpad_names: spec.inherited_scratchpad.clone(),
            // The worker's own options inherit both (`Agent::new_subagent`), so its reads are sized
            // against the same line its compaction fires at.
            context_ceiling_percent: params.parent_options.context_ceiling_percent,
            auto_compact: params.parent_options.auto_compact,
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
                // A pinned worker's children default to its pinned profile, so the choice holds
                // through the subtree. Otherwise the *parent's* handle, not a snapshot of what it
                // currently holds: a switch the user makes later must reach the whole subtree's
                // future spawns, not just its first level.
                profile: if spec.is_pinned() {
                    sub_cells.profile.clone()
                } else {
                    params.cells.profile.clone()
                },
                ..sub_cells.clone()
            },
            // The worker's own granted level, not its parent's. `params.memory_access` is what the
            // spawning agent holds, so letting the spread supply it would let a worker grant its
            // children up to its parent's level rather than its own, reaching through a child
            // what it was denied directly. The deny lists come from `spec` for the same reason.
            memory_access,
            config_scope: params.config_scope.clone(),
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
            inherited_scope: tool_scope,
            remaining_depth: spec.remaining_depth,
            absolute_depth: spec.absolute_depth,
        })?;
    }

    let available = crate::prompt::AvailableTools::new(sub_registry.registered_tool_names());
    let sub_system_prompt = build_subagent_system_prompt(
        crate::prompt::SystemPromptInputs {
            sandboxed_shell: parent_options.sandboxed_shell,
            user_instructions: granted_instructions.as_deref(),
            one_shot: parent_options.one_shot,
            background: sub_registry
                .background_enabled()
                .then(|| crate::prompt::BackgroundTools {
                    task_list: available.is_available("task_list"),
                    task_cancel: available.is_available("task_cancel"),
                }),
            scheduling: available.is_available("schedule_create"),
            auto_compact: parent_options.auto_compact,
            turn_retention: parent_options.turn_retention,
            conversation_search: available.is_available("conversation_search"),
            conversation_read: available.is_available("conversation_read"),
            memory_write: available.is_available("memory_write"),
        },
        &available,
        &spec.inherited_scratchpad,
    );

    // A worker two levels down names the worker that spawned it, the way Claude Code's nested
    // sub-agent names its parent; the root is no agent in that sense and is not named.
    let parent_agent_id = (spec.absolute_depth >= 2)
        .then(|| params.cells.session_id.get())
        .flatten()
        .map(crate::provider::WorkerIdentity::agent_id_for);
    Ok(Agent::new_subagent(
        &params.materials,
        sub_cells,
        sub_registry,
        &parent_options,
        sub_system_prompt,
        crate::agent::WorkerLineage {
            prompt_id: call.prompt_id,
            parent_agent_id,
        },
    ))
}

/// The scope a spawn call states, as spelled, or `None` for a bare call, by the one rule the spec
/// is read by ([`WorkerScope::from_fields`]); a list beside no `working_directory` is refused with
/// the remedy. An empty list counts as absent, since a model emits `[]` for "nothing to add".
fn requested_worker_scope(input: &serde_json::Value) -> Result<Option<WorkerScope>> {
    WorkerScope::from_fields(
        optional_str(input, "working_directory", "agent_spawn")?.map(PathBuf::from),
        directory_array(input, "additional_workspace_directories")?,
        directory_array(input, "additional_readable_directories")?,
    )
    .map_err(|_| MekaError::ToolExecution {
        tool_name: "agent_spawn".to_string(),
        message: "name `working_directory` beside a directory list; \".\" is your working \
                  directory alone, so list your other directories to keep them"
            .to_string(),
    })
}

/// The directories a scope field names, by the one rule every list is read by. An entry that is
/// not a string refuses the call rather than being dropped: the list confines, so an entry
/// dropped from it would widen the scope the parent meant to state, and beside no
/// `working_directory` a dropped list would read as a bare spawn with the parent's whole scope. A
/// blank entry names nothing and is dropped; a missing or null field is an empty list.
fn directory_array(input: &serde_json::Value, key: &str) -> Result<Vec<PathBuf>> {
    Ok(string_or_list(input, key, "agent_spawn")?
        .unwrap_or_default()
        .into_iter()
        .map(PathBuf::from)
        .collect())
}

/// The names a list parameter gives, each trimmed, by the one rule every list is read by:
/// `None` when the parameter is not given. A bare string is the one name it gives; read as
/// nothing, `deny_tools: "file_write"` would spawn an unrestricted worker in place of the
/// restricted one the caller asked for.
fn name_list(input: &serde_json::Value, key: &str) -> Result<Option<Vec<String>>> {
    Ok(string_or_list(input, key, "agent_spawn")?.map(|names| {
        names
            .into_iter()
            .map(|name| name.trim().to_string())
            .collect()
    }))
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

/// What a sub-agent's shell and `web_fetch` may reach: the parent's setting when the call names
/// none, which at `unrestricted` is what the parent itself holds, `web_fetch` cut and a shell
/// nothing sandboxes; the named setting when it lies within the parent's; and a refusal for one
/// above the parent's, since a worker never reaches wider than its parent, and for a cut beside
/// `unrestricted`, which would state one the shell cannot hold.
fn resolve_subagent_network(
    requested: Option<&str>,
    parent: crate::config::Network,
    level: Permission,
) -> Result<crate::config::Network> {
    let refuse = |message: String| MekaError::ToolExecution {
        tool_name: "agent_spawn".to_string(),
        message,
    };
    let Some(requested) = requested else {
        return Ok(parent);
    };
    let requested: crate::config::Network = requested
        .parse()
        .map_err(|error: String| refuse(format!("`network`: {error}")))?;
    if requested.narrowest(parent) != requested {
        let within: Vec<String> = crate::config::Network::LADDER
            .iter()
            .skip_while(|value| **value != parent)
            .map(|value| format!("`{value}`"))
            .collect();
        let separator = if within.len() > 1 { "," } else { "" };
        return Err(refuse(format!(
            "`network: \"{requested}\"` is above your own `{parent}`; name {}{separator} or leave \
             it out",
            within.join(" or ")
        )));
    }
    if requested != crate::config::Network::Host && level == Permission::Unrestricted {
        return Err(refuse(format!(
            "`network: \"{requested}\"` cannot hold at `unrestricted`, whose shell is not \
             sandboxed; name a lower level"
        )));
    }
    Ok(requested)
}

/// The level a sub-agent runs at: the parent's when the call names none, the named level when
/// it lies within the parent's, and a refusal above it, named so the model can lower its ask.
/// Never a silent clamp: a worker at a level the call did not ask for is a worker whose scope
/// the parent misreads. `SharedPermission::with_ceiling` flattens a grandchild's ceiling to the
/// *root* cell on the precondition that what this returns never exceeds the parent.
fn resolve_subagent_permission(
    requested: Option<Permission>,
    parent: Permission,
) -> Result<Permission> {
    match requested {
        Some(requested) if !requested.is_within(parent) => Err(MekaError::ToolExecution {
            tool_name: "agent_spawn".to_string(),
            message: format!(
                "`permission: \"{requested}\"` is above your own level `{parent}`; name `{parent}` \
                 or lower, or leave it out"
            ),
        }),
        Some(requested) => Ok(requested),
        None => Ok(parent),
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
/// seeded from `[subagents].max_depth`; `max_depth` may lower it but never raise it.
/// `absolute_depth` is the monotonic hard counter: it always increments and, once it reaches
/// [`SUBAGENT_ABSOLUTE_MAX_DEPTH`], no further `agent_spawn` is granted. A nested `agent_spawn` is
/// granted only when both budgets allow it.
///
/// The override is clamped because `[subagents].max_depth` is documented as a ceiling
/// ("`max_depth = 1` means sub-agents cannot spawn further sub-agents"), and an
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

/// A worker's role and granted standing instructions. Live state uses the same per-turn renderer as
/// the root.
fn build_subagent_system_prompt(
    inputs: crate::prompt::SystemPromptInputs<'_>,
    available: &crate::prompt::AvailableTools,
    inherited_scratchpad: &[String],
) -> String {
    let has_instructions = inputs
        .user_instructions
        .map(str::trim)
        .is_some_and(|text| !text.is_empty());
    let mut prompt = crate::prompt::build_system_prompt(inputs);
    prompt.push_str(
        "\n## Assignment\n\n\
         You were spawned by a parent agent, which receives your final reply as a report; nothing \
         you ask is answered before then. Your conversation, checklist and scratchpad entries are \
         private; filesystem changes are shared within your granted workspace.",
    );
    if has_instructions {
        prompt.push_str(" The standing instructions above address the parent's user.");
    }
    prompt.push('\n');
    if !inherited_scratchpad.is_empty() {
        prompt.push_str("\n## Inherited scratchpad entries\n\n");
        prompt.push_str(
            "These parent entries are read-only; use different names for your own entries.\n",
        );
        if available.is_available("scratchpad_read") {
            prompt
                .push_str("Use `scratchpad_read` to read their contents before relying on them.\n");
        }
        for name in inherited_scratchpad {
            prompt.push_str(&format!("- {name}\n"));
        }
    }
    prompt
}

#[cfg(test)]
mod tests {
    /// The store moves exactly one name out of the spawn terms when a profile is renamed, under
    /// the key it names; this is the spec whose serialization that key has to match.
    #[test]
    fn the_spec_records_its_profile_under_the_key_the_store_renames() {
        let spec: SubagentSpec =
            serde_json::from_str(r#"{"permission":"read","tools":[],"profile":"work"}"#)
                .expect("a pinned spec");
        let document = serde_json::to_value(&spec).expect("serialize");
        assert_eq!(
            document[crate::store::Store::SUBAGENT_SPEC_PROFILE_KEY],
            "work"
        );
    }

    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::{
        provider::{Provider, mock::text_round},
        store::Store,
    };

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
            resolve_subagent_permission(None, Permission::Unrestricted).expect("inherited"),
            Permission::Unrestricted
        );
        assert_eq!(
            resolve_subagent_permission(None, Permission::Read).expect("inherited"),
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
            allowed_servers: None,
            allowed_tools: None,
            memory: MemoryAccess::None,
            instructions: InstructionAccess::None,
            inherited_scratchpad: Vec::new(),
            remaining_depth: 2,
            absolute_depth: 1,
            working_directory: None,
            additional_workspace_directories: Vec::new(),
            additional_readable_directories: Vec::new(),
            network: crate::config::Network::Host,
            profile: None,
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

    /// A spec carrying a list never resolves above `workspace`, whatever level it claims:
    /// `agent_spawn` refuses the pair, but `meka session import` writes a spec verbatim, and a
    /// list on a worker that is confined by nothing would confine nothing.
    #[test]
    fn a_spec_with_a_list_never_resolves_above_workspace() {
        let spec = SubagentSpec {
            permission: Permission::Unrestricted,
            working_directory: Some(PathBuf::from("/work/sub")),
            additional_workspace_directories: vec![PathBuf::from("/work/out")],
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
        assert!(
            resolve_subagent_permission(Some(Permission::Workspace), Permission::Read).is_err(),
            "a `workspace` request under a `read` parent is refused at the door"
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
                allowed_servers: None,
                allowed_tools: None,
                memory: MemoryAccess::None,
                instructions: InstructionAccess::None,
                inherited_scratchpad: Vec::new(),
                remaining_depth: 2,
                absolute_depth: 1,
                working_directory: None,
                additional_workspace_directories: Vec::new(),
                additional_readable_directories: Vec::new(),
                network: crate::config::Network::Host,
                profile: None,
            }
        }

        let root = SharedPermission::new(Permission::Unrestricted, EnabledPermissions::ALL);
        let child_spec = spec_at(
            resolve_subagent_permission(Some(Permission::Read), root.get()).expect("within"),
            root.enabled(),
        );
        let child = child_spec.shared_permission_bounded(&root);
        assert_eq!(
            child.get(),
            Permission::Read,
            "child holds what it asked for"
        );

        let grandchild_spec = spec_at(
            resolve_subagent_permission(Some(Permission::Read), child.get()).expect("within"),
            child.enabled(),
        );
        let grandchild = grandchild_spec.shared_permission_bounded(&child);
        assert_eq!(
            grandchild.get(),
            Permission::Read,
            "a grandchild holds what it asked for under a `read` parent, however the ceiling \
             cell is flattened"
        );
        assert!(
            !grandchild.enabled().is_enabled(Permission::Workspace),
            "nor may `workspace` remain switchable in its enabled set"
        );
    }

    /// A level above the parent's is refused with the parent's own level as the remedy, never
    /// clamped: a sub-agent can never be escalated above its parent, and a worker at a level the
    /// call did not ask for is one whose scope the parent misreads. A lower level is granted as
    /// asked.
    #[test]
    fn resolve_subagent_permission_refuses_above_the_parent_and_grants_within() {
        for requested in [Permission::Unrestricted, Permission::Workspace] {
            let refusal = resolve_subagent_permission(Some(requested), Permission::Read)
                .expect_err("above the parent")
                .to_string();
            assert!(
                refusal.contains(&format!(
                    "`permission: \"{requested}\"` is above your own level"
                )) && refusal.contains("name `read` or lower"),
                "{refusal}"
            );
        }
        for requested in [Permission::Read, Permission::None] {
            assert_eq!(
                resolve_subagent_permission(Some(requested), Permission::Unrestricted)
                    .expect("within the parent"),
                requested
            );
        }
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

    /// A list of names drops a blank entry, trims the rest, refuses an entry that is not a name,
    /// and takes a bare string as the one name it gives: read as nothing, `deny_tools:
    /// "file_write"` would spawn an unrestricted worker in place of the restricted one asked for.
    #[test]
    fn a_name_list_takes_names_and_refuses_what_is_not_one() {
        let input = serde_json::json!({
            "deny_servers": ["notion", "  ", "  linear  "],
            "deny_tools": ["file_write", 7],
            "bare": "notion",
        });
        assert_eq!(
            name_list(&input, "deny_servers").expect("a list"),
            Some(vec!["notion".to_string(), "linear".to_string()]),
        );
        let error = name_list(&input, "deny_tools").expect_err("a number is not a name");
        assert!(error.to_string().contains("deny_tools"), "{error}");
        assert_eq!(name_list(&input, "absent").expect("absent is fine"), None);
        assert_eq!(
            name_list(&input, "bare").expect("one name"),
            Some(vec!["notion".to_string()])
        );
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

    /// `max_depth` narrows the child's budget and can never widen it. `[subagents].max_depth` is
    /// documented as a ceiling, so a model asking for more than the operator allowed gets the
    /// operator's answer.
    #[test]
    fn child_spawn_depth_override_only_narrows() {
        // Asking for more than is left yields what is left, not what was asked for.
        let (remaining, _absolute, allow) = child_spawn_depth(3, 0, Some(9));
        assert_eq!(remaining, 2);
        assert!(allow);

        // At the documented `max_depth = 1`, no override can grant a grandchild.
        let (remaining, _absolute, allow) = child_spawn_depth(1, 0, Some(5));
        assert_eq!(remaining, 0);
        assert!(!allow, "max_depth = 1 must forbid nesting");

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
    async fn subagent_registry_has_an_independent_checklist() {
        use crate::{
            config::BuiltinToolFilter,
            sandbox::{BackendProbe, SandboxCapability},
        };

        let parent_list = crate::checklist::SharedChecklist::default();
        let sub_list = crate::checklist::SharedChecklist::default();

        let sub_registry = ToolRegistry::build_for_subagent(
            &crate::session::SessionMaterials {
                providers: std::sync::Arc::new(crate::provider::ProviderRegistry::for_test(
                    Store::for_test().await.token_store(),
                    &["test-profile"],
                )),
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
                    running_subagents: Default::default(),
                },
                skills: crate::skills::SkillCache::for_root(None),
                memories: crate::store::memory::MemoryStore::detached(),
                ..crate::session::SessionMaterials::for_test(store_for_test().await)
            },
            &crate::session::SessionCells {
                session_id: crate::session::SharedSessionId::default(),
                project: Default::default(),
                checklist: sub_list.clone(),
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
                tool_scope: ToolScope::default(),
                memory_access: MemoryAccess::Write,
                parent_session_id: None,
                inherited_scratchpad_names: Vec::new(),
                context_ceiling_percent: 80,
                auto_compact: false,
            },
        )
        .expect("subagent registry should build");

        let add = sub_registry
            .get("checklist_add")
            .expect("subagent should have checklist_add");
        add.execute(
            serde_json::json!({ "items": ["sub task"] }),
            crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("the add should succeed");

        assert_eq!(sub_list.get().items.len(), 1);
        assert!(
            parent_list.get().items.is_empty(),
            "parent list must remain untouched"
        );
    }

    /// Both registry meta-tools reach a worker through the door the root uses, so a worker can
    /// find and load the deferred tools its parent's servers gave it.
    #[tokio::test]
    async fn a_worker_registry_carries_tool_search() {
        use crate::{
            config::BuiltinToolFilter,
            sandbox::{BackendProbe, SandboxCapability},
        };

        let sub_registry = ToolRegistry::build_for_subagent(
            &crate::session::SessionMaterials {
                providers: std::sync::Arc::new(crate::provider::ProviderRegistry::for_test(
                    Store::for_test().await.token_store(),
                    &["test-profile"],
                )),
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
                    running_subagents: Default::default(),
                },
                skills: crate::skills::SkillCache::for_root(None),
                memories: crate::store::memory::MemoryStore::detached(),
                ..crate::session::SessionMaterials::for_test(store_for_test().await)
            },
            &crate::session::SessionCells::for_test(
                SharedPermission::new(Permission::Read, crate::permission::EnabledPermissions::ALL),
                crate::workspace::cwd_for_test(),
                crate::workspace::roots_for_test(),
                Arc::new(crate::frontend::SilentFrontend),
            ),
            crate::tools::registry::RegistryScope {
                tool_scope: ToolScope::default(),
                memory_access: MemoryAccess::Write,
                parent_session_id: None,
                inherited_scratchpad_names: Vec::new(),
                context_ceiling_percent: 80,
                auto_compact: false,
            },
        )
        .expect("subagent registry should build");

        // The discovery tools ride only while something is deferred, for a worker as for the root.
        sub_registry.register_deferred_fixture("fixture_alpha");
        let active: Vec<String> = sub_registry
            .definitions_active_with_loaded(&[])
            .into_iter()
            .map(|definition| definition.name)
            .collect();
        assert!(
            active.iter().any(|name| name == "tool_search"),
            "{active:?}"
        );
        assert!(active.iter().any(|name| name == "tool_load"), "{active:?}");
    }

    fn spec_for_test(permission: Permission) -> SubagentSpec {
        SubagentSpec {
            permission,
            enabled_permissions: vec![Permission::None, permission],
            denied_servers: vec!["mekabridge".to_string()],
            denied_tools: vec!["file_write".to_string()],
            allowed_servers: None,
            allowed_tools: None,
            memory: MemoryAccess::None,
            instructions: InstructionAccess::Inherit,
            inherited_scratchpad: vec!["build_log".to_string()],
            remaining_depth: 2,
            absolute_depth: 1,
            working_directory: None,
            additional_workspace_directories: Vec::new(),
            additional_readable_directories: Vec::new(),
            network: crate::config::Network::Host,
            profile: None,
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
            WorkerPlacement {
                workspace: WorkerWorkspace {
                    cwd: crate::workspace::cwd_for_test(),
                    roots: crate::workspace::roots_for_test(),
                    read_only_roots: crate::workspace::SharedRoots::default(),
                    read_boundary: crate::workspace::ReadBoundary::unconfined(),
                    network: crate::config::Network::Host,
                },
                binding: params.cells.profile.current(),
            },
            parent_session,
            Uuid::new_v4(),
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
        assert_eq!(
            decoded.profile, None,
            "an absent profile is a worker that follows its parent"
        );
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
        let tool_scope = spec_for_test(Permission::Read).tool_scope();
        assert!(!tool_scope.admits_server("mekabridge"));
        assert!(!tool_scope.admits_tool("mcp__mekabridge__send_message"));
        assert!(!tool_scope.admits_tool("file_write"));
        assert!(tool_scope.admits_tool("file_read"));
        let allowed = SubagentSpec {
            allowed_servers: Some(vec!["notion".to_string()]),
            allowed_tools: Some(vec!["file_read".to_string()]),
            ..spec_for_test(Permission::Read)
        }
        .tool_scope();
        assert!(allowed.admits_tool("file_read") && !allowed.admits_tool("file_find"));
        assert!(allowed.admits_server("notion") && !allowed.admits_server("linear"));
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
            config_scope: ToolScope::default(),
            parent_options: AgentOptions {
                one_shot: false,
                streaming: false,
                sandboxed_shell: false,
                gate_tools: None,
                context_ceiling_percent: 80,
                auto_compact: false,
                compact_checkpoint: false,
                turn_retention: None,
                user_instructions: Some("never inherited by a worker".to_string()),
                user_instructions_count: 1,
                mcp_grace: std::time::Duration::ZERO,
                system_prompt_override: None,
            },
        }
    }

    /// The whole Phase 4 loop against a scripted provider: spawn returns an id, `agent_list`
    /// reports the worker, a follow-up sees the first turn's history, and `agent_delete`
    /// removes it.
    /// A worker is held to its own list on its own turn: the nudge and the cap play out inside
    /// the spawn, and nothing of it reaches the parent's surface, whose list is the parent's.
    #[tokio::test]
    async fn a_worker_is_nudged_on_its_own_list_and_its_parent_sees_none_of_it() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent session");
        let parent_session = crate::session::SharedSessionId::new(Some(parent_sid));
        // The worker adds an item and then answers four times without touching it: three nudges,
        // then the cap lets the turn end.
        let mut rounds = vec![vec![
            crate::provider::mock::MockEvent::ToolUseStart {
                id: "add-1".to_string(),
                name: "checklist_add".to_string(),
            },
            crate::provider::mock::MockEvent::ToolUseEnd {
                input: serde_json::json!({"items": ["reply pong"]}),
            },
            crate::provider::mock::MockEvent::MessageEnd {
                stop_reason: crate::provider::mock::MockStopReason::ToolUse,
            },
        ]];
        rounds.extend((0..4).map(|_| text_round("pong")));
        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(rounds));
        let recorder = Arc::new(crate::frontend::testing::RecordingFrontend::new());
        let mut params = params_for_test(store.clone(), parent_session);
        params.cells.frontend = Arc::clone(&recorder) as Arc<dyn crate::frontend::Frontend>;
        let spawn = spawn_tool_for(params, provider);

        let agent_id = spawned(&spawn, serde_json::json!({ "prompt": "reply pong" })).await;

        let conversation = store
            .load_conversation(agent_id)
            .await
            .expect("the worker's conversation");
        let nudges = conversation
            .as_slice()
            .iter()
            .filter(|message| message.is_nudge())
            .count();
        assert_eq!(nudges, 3, "{:?}", conversation.as_slice());
        let leaked: Vec<String> = recorder
            .events()
            .iter()
            .filter_map(|event| match event {
                crate::frontend::FrontendEvent::Notice(notice)
                    if notice.text.starts_with("Checklist:") =>
                {
                    Some(notice.text.clone())
                }
                crate::frontend::FrontendEvent::ChecklistUpdated { .. } => {
                    Some("a checklist update".to_string())
                }
                crate::frontend::FrontendEvent::Nudged { .. } => Some("a nudge".to_string()),
                _ => None,
            })
            .collect();
        assert!(
            leaked.is_empty(),
            "the worker's list is its own: {leaked:?}"
        );
    }

    #[tokio::test]
    async fn spawn_followup_and_delete_round_trip() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent session");
        let parent_session = crate::session::SharedSessionId::new(Some(parent_sid));
        // The first answer reports a usage, so the worker's row records it and the follow-up's
        // rebuilt worker is checked against that rather than a fresh zero.
        let mock = Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
            vec![
                crate::provider::mock::MockEvent::Usage {
                    input_tokens: 12_345,
                },
                crate::provider::mock::MockEvent::Text {
                    text: "first answer".to_string(),
                },
            ],
            text_round("second answer"),
        ]));
        let provider: Arc<dyn Provider> = Arc::clone(&mock) as Arc<dyn Provider>;
        let params = params_for_test(store.clone(), parent_session.clone());

        let spawn = AgentSpawnTool {
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params.clone().on_provider(Arc::clone(&provider)),
            inherited_scope: ToolScope::default(),
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
            .find_map(|line| line.strip_prefix("Agent: "))
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
        assert_eq!(
            store.load_context_tokens(agent_id).await.expect("load"),
            Some(12_345),
            "the worker's row records the measurement its own turn reported"
        );

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

        // The worker's row carried the first turn's measurement, and the rebuilt worker took it as
        // its own: the follow-up's request gauges itself at that occupancy, where a worker that
        // started from zero would carry no budget line at all.
        let requests = mock.completions();
        assert_eq!(requests.len(), 2, "the spawn's turn and the follow-up's");
        let followup_request = requests[1]
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|block| match block {
                crate::conversation::ContentBlock::Text { text }
                | crate::conversation::ContentBlock::TurnContext { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            followup_request.contains("[Context budget]") && followup_request.contains("~12k"),
            "the follow-up must be gauged at the row's 12,345 tokens: {followup_request}"
        );

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
    /// session remove --all` would take the lock nobody holds and cascade the conversation away,
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
            allowed_servers: None,
            allowed_tools: None,
            memory: MemoryAccess::None,
            instructions: InstructionAccess::None,
            inherited_scratchpad: Vec::new(),
            remaining_depth: 0,
            absolute_depth: 1,
            working_directory: None,
            additional_workspace_directories: Vec::new(),
            additional_readable_directories: Vec::new(),
            network: crate::config::Network::Host,
            profile: None,
        };
        let (worker, held) = store
            .create_child_session_for_test(
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
            refused.to_string().contains(&worker.to_string())
                && refused.to_string().contains("still running"),
            "and the refusal has to name the worker and what it is doing: {refused}"
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
            allowed_servers: None,
            allowed_tools: None,
            memory: MemoryAccess::None,
            instructions: InstructionAccess::None,
            inherited_scratchpad: Vec::new(),
            remaining_depth: 0,
            absolute_depth: 1,
            working_directory: None,
            additional_workspace_directories: Vec::new(),
            additional_readable_directories: Vec::new(),
            network: crate::config::Network::Host,
            profile: None,
        };
        // On a profile the parent is not on, so a write that should not happen is visible.
        let (worker, held) = store
            .create_child_session_for_test(
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
        let error = followup
            .execute(
                serde_json::json!({ "id": worker.to_string(), "prompt": "carry on" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("the held lock refuses the follow-up");
        assert!(
            error.to_string().contains("still running")
                && error.to_string().contains("`agent_steer`"),
            "the refusal says what is happening and names the door that is open: {error}"
        );

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
                        name: "file_write".into(),
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
            allowed_servers: None,
            allowed_tools: None,
            memory: MemoryAccess::Write,
            instructions: InstructionAccess::None,
            inherited_scratchpad: Vec::new(),
            remaining_depth: 0,
            absolute_depth: 1,
            working_directory: None,
            additional_workspace_directories: Vec::new(),
            additional_readable_directories: Vec::new(),
            network: crate::config::Network::Host,
            profile: None,
        };
        let child = store
            .create_child_session_for_test(
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
            transcript.contains("'file_write' requires `workspace`; the session is at `read`"),
            "the worker should have been refused file_write at Read, got: {transcript}"
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
        params.config_scope = ToolScope::new(vec!["mekabridge".to_string()], Vec::new());
        params.memory_access = MemoryAccess::None;

        let spec = SubagentSpec {
            permission: Permission::Read,
            enabled_permissions: vec![Permission::Read],
            denied_servers: Vec::new(),
            denied_tools: Vec::new(),
            allowed_servers: None,
            allowed_tools: None,
            memory: MemoryAccess::Write,
            instructions: InstructionAccess::None,
            inherited_scratchpad: Vec::new(),
            remaining_depth: 0,
            absolute_depth: 1,
            working_directory: None,
            additional_workspace_directories: Vec::new(),
            additional_readable_directories: Vec::new(),
            network: crate::config::Network::Host,
            profile: None,
        };
        let child = store
            .create_child_session_for_test(
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
            allowed_servers: None,
            allowed_tools: None,
            memory: MemoryAccess::Write,
            instructions: InstructionAccess::None,
            inherited_scratchpad: Vec::new(),
            remaining_depth: 0,
            absolute_depth: 1,
            working_directory: None,
            additional_workspace_directories: Vec::new(),
            additional_readable_directories: Vec::new(),
            network: crate::config::Network::Host,
            profile: None,
        };
        let config = ToolScope::new(vec!["mekabridge".to_string()], vec![
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
            combined_for_followup(spec, &ToolScope::default(), MemoryAccess::Write).memory,
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
            inherited_scope: ToolScope::default(),
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
            .find_map(|line| line.strip_prefix("Agent: "))
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
            inherited_scope: ToolScope::default(),
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
            .find_map(|line| line.strip_prefix("Agent: "))
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

    /// A worker's row records the project its parent is on when it is spawned, as its profile
    /// is recorded, so its memories file where the parent's do.
    #[tokio::test]
    async fn a_spawned_worker_records_the_project_its_parent_is_on() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
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
            "test-profile",
        );
        params.cells.project = crate::session::SharedProject::new(Some("webdev".to_string()));

        let spawn = AgentSpawnTool {
            parent_permission: SharedPermission::new(Permission::Read, EnabledPermissions::ALL),
            tool_builder_params: params,
            inherited_scope: ToolScope::default(),
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
            .find_map(|line| line.strip_prefix("Agent: "))
            .and_then(|id| Uuid::parse_str(id.trim()).ok())
            .unwrap_or_else(|| panic!("no agent id in: {text}"));
        assert_eq!(
            store
                .recorded_project(agent_id)
                .await
                .expect("the child's row"),
            Some("webdev".to_string())
        );
    }

    /// A follow-up runs the worker in the project its parent is on then, and rewrites its row to
    /// say so, as it does for the parent's profile.
    #[tokio::test]
    async fn a_follow_up_runs_the_worker_in_its_parent_s_current_project_and_records_it() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
                text_round("first"),
                text_round("second"),
            ]));
        let mut params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        )
        .on_provider(Arc::clone(&provider));
        params.cells.project = crate::session::SharedProject::new(Some("webdev".to_string()));
        let spawn = AgentSpawnTool {
            parent_permission: SharedPermission::new(Permission::Read, EnabledPermissions::ALL),
            tool_builder_params: params.clone(),
            inherited_scope: ToolScope::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        };
        let text = spawn
            .execute(
                serde_json::json!({"prompt": "do a thing", "permission": "read"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn")
            .text_content();
        let agent_id: Uuid = text
            .lines()
            .find_map(|line| line.strip_prefix("Agent: "))
            .and_then(|id| Uuid::parse_str(id.trim()).ok())
            .unwrap_or_else(|| panic!("no agent id in: {text}"));
        assert_eq!(
            store.recorded_project(agent_id).await.expect("row"),
            Some("webdev".to_string())
        );

        // The parent moves; the shared cell is what the follow-up reads.
        params.cells.project.set(Some("movie".to_string()));
        let followup = AgentFollowupTool {
            parent_permission: SharedPermission::new(Permission::Read, EnabledPermissions::ALL),
            tool_builder_params: params,
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        let second = followup
            .execute(
                serde_json::json!({"id": agent_id.to_string(), "prompt": "and then?"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("follow-up")
            .text_content();
        assert!(second.contains("second"), "{second}");
        assert_eq!(
            store.recorded_project(agent_id).await.expect("row"),
            Some("movie".to_string()),
            "the row follows the parent, as it does for the profile"
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
            inherited_scope: ToolScope::default(),
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
            .find_map(|line| line.strip_prefix("Agent: "))
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
            inherited_scope: ToolScope::default(),
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
            inherited_scope: ToolScope::default(),
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
        assert!(inline.starts_with("Agent: "), "{inline}");
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

    /// The grant reaches the prompt, and its absence leaves no trace of the section.
    #[test]
    fn system_prompt_carries_instructions_only_when_granted() {
        let ungranted = build_subagent_system_prompt(
            crate::prompt::SystemPromptInputs::for_test(false, None),
            &crate::prompt::AvailableTools::default(),
            &[],
        );
        assert!(!ungranted.contains("Standing instructions"));

        let granted = build_subagent_system_prompt(
            crate::prompt::SystemPromptInputs::for_test(
                false,
                Some("Never use pip. Always prefer uv."),
            ),
            &crate::prompt::AvailableTools::default(),
            &[],
        );
        assert!(granted.contains("## Standing instructions"));
        assert!(granted.contains("Never use pip. Always prefer uv."));
        // The worker is told whose rules these are, so persona clauses read as context rather than
        // as an instruction to address the user directly; without a grant there is nothing to say.
        assert!(
            granted.contains("The standing instructions above address the parent's user."),
            "{granted}"
        );
        assert!(
            !ungranted.contains("address the parent's user"),
            "{ungranted}"
        );

        // Whitespace-only instructions are treated as absent, matching the root agent.
        assert!(
            !build_subagent_system_prompt(
                crate::prompt::SystemPromptInputs::for_test(false, Some("  \n ")),
                &crate::prompt::AvailableTools::default(),
                &[],
            )
            .contains("Standing instructions")
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
            inherited_scope: ToolScope::default(),
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
            "agent_steer",
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
            "agent_steer",
        ] {
            assert!(
                filtered.get(name).is_none(),
                "'{name}' must go with agent_spawn"
            );
        }
    }

    /// Each registered tool has to actually describe itself: `meka tool list` reads the same
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
            inherited_scope: ToolScope::default(),
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
                Some(vec!["file_read".to_string()]),
                Vec::new(),
                std::collections::HashMap::new(),
            ),
            3
        ));
    }

    /// A worker spawned with `background: true` holds its own session lock for as long as it runs
    /// and never enters the follow-up guard, so the lock is what has to refuse the delete.
    #[tokio::test]
    async fn a_delete_is_refused_while_the_worker_is_held() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = store
            .create_child_session_for_test(
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
        let delete = AgentDeleteTool {
            tool_builder_params: params,
            in_flight: InFlightFollowups::default(),
        };

        // Stands in for the worker's own running turn, which holds this lock from inside the
        // spawn's task.
        let held = store.lock_session(child).expect("hold the worker");
        let error = delete
            .execute(
                serde_json::json!({ "id": child.to_string() }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a running worker must not be deleted");
        assert!(error.to_string().contains("still running"), "{error}");
        assert!(
            store.session_exists(child).await.expect("exists"),
            "and its row is untouched"
        );

        drop(held);
        delete
            .execute(
                serde_json::json!({ "id": child.to_string() }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("delete succeeds once the worker has finished");
        assert!(!store.session_exists(child).await.expect("exists"));
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
            .create_child_session_for_test(
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
    /// messages, so a single task that took three tool rounds must still read as one turn, and a
    /// steer that landed beside one round's results is part of that round rather than a turn.
    #[tokio::test]
    async fn agent_list_turns_excludes_tool_result_messages() {
        use crate::conversation::{ContentBlock, Event, Message, Role, ToolResultContent};

        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = store
            .create_child_session_for_test(
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
        let steered = Event::Append(Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "u2".to_string(),
                    content: vec![ToolResultContent::Text {
                        text: "ok".to_string(),
                    }],
                    is_error: false,
                },
                ContentBlock::Text {
                    text: "and hurry".to_string(),
                },
            ],
        });
        for event in [
            Event::Append(Message::user("the one and only task")),
            Event::Append(Message::assistant_text("working")),
            tool_result,
            Event::Append(Message::assistant_text("still working")),
            steered,
            Event::Append(Message::assistant_text("done")),
        ] {
            store.save_event(child, &event, None).await.expect("save");
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
            inherited_scope: ToolScope::default(),
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
            inherited_scope: ToolScope::default(),
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
    /// The text blocks of the last user message of a request, in order.
    fn last_user_text_blocks(request: &[crate::conversation::Message]) -> Vec<String> {
        request
            .iter()
            .rev()
            .find(|message| message.role == crate::conversation::Role::User)
            .map(|message| {
                message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        crate::conversation::ContentBlock::Text { text }
                        | crate::conversation::ContentBlock::Inbox { text, .. } => {
                            Some(text.clone())
                        }
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// A steer written to a worker that has finished waits in its inbox; the next follow-up's
    /// opening carries it after the follow-up's own words, so nothing is stranded.
    #[tokio::test]
    async fn a_steer_written_after_a_worker_finished_opens_its_next_followup() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent session");
        let parent_session = crate::session::SharedSessionId::new(Some(parent_sid));
        let mock = Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
            text_round("first answer"),
            text_round("second answer"),
        ]));
        let provider: Arc<dyn Provider> = Arc::clone(&mock) as Arc<dyn Provider>;
        let params = params_for_test(store.clone(), parent_session.clone());
        let permission = SharedPermission::new(Permission::Unrestricted, EnabledPermissions::ALL);

        let spawn = AgentSpawnTool {
            parent_permission: permission.clone(),
            tool_builder_params: params.clone().on_provider(Arc::clone(&provider)),
            inherited_scope: ToolScope::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        };
        let text = spawn
            .execute(
                serde_json::json!({ "prompt": "look into it", "permission": "read" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds")
            .text_content();
        let agent_id: Uuid = text
            .lines()
            .find_map(|line| line.strip_prefix("Agent: "))
            .and_then(|id| Uuid::parse_str(id.trim()).ok())
            .unwrap_or_else(|| panic!("spawn must return a usable agent id, got: {text}"));

        let steer = AgentSteerTool {
            tool_builder_params: params.clone(),
        };
        let queued = steer
            .execute(
                serde_json::json!({ "id": agent_id.to_string(), "message": "check the dates too" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("steer succeeds")
            .text_content();
        assert!(queued.contains("Queued for sub-agent"), "{queued}");
        assert_eq!(
            store
                .inbox_store()
                .list_open(agent_id)
                .await
                .expect("list")
                .len(),
            1,
            "the message waits in the worker's inbox"
        );

        let followup = AgentFollowupTool {
            parent_permission: permission,
            tool_builder_params: params.clone().on_provider(Arc::clone(&provider)),
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        followup
            .execute(
                serde_json::json!({ "id": agent_id.to_string(), "prompt": "and then?" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("followup succeeds");

        let requests = mock.completions();
        let blocks = last_user_text_blocks(&requests[1]);
        assert_eq!(blocks.len(), 2, "the words, then the steer: {blocks:?}");
        assert_eq!(blocks[0], "and then?");
        assert!(
            blocks[1].starts_with("[Message from your parent agent, arrived ")
                && blocks[1].ends_with("]\ncheck the dates too"),
            "{}",
            blocks[1]
        );
        assert!(
            store
                .inbox_store()
                .list_open(agent_id)
                .await
                .expect("list")
                .is_empty(),
            "the follow-up's accepted request delivered it"
        );
    }

    /// A steer written while the worker runs lands at its next round boundary, after that
    /// round's tool results, without the parent waiting for anything.
    #[tokio::test]
    async fn a_steer_written_while_a_worker_runs_lands_after_its_next_round() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent session");
        let parent_session = crate::session::SharedSessionId::new(Some(parent_sid));
        // The worker's first round holds until the steer is written, so its tool runs after it;
        // the boundary after that tool is where the steer lands.
        let steer_written = CancellationToken::new();
        let mock = Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
            vec![
                crate::provider::mock::MockEvent::Hold {
                    gate: steer_written.clone(),
                },
                crate::provider::mock::MockEvent::ToolUseStart {
                    id: "call-0".to_string(),
                    name: "does_not_exist".to_string(),
                },
                crate::provider::mock::MockEvent::ToolUseEnd {
                    input: serde_json::json!({}),
                },
                crate::provider::mock::MockEvent::MessageEnd {
                    stop_reason: crate::provider::mock::MockStopReason::ToolUse,
                },
            ],
            text_round("done"),
        ]));
        let provider: Arc<dyn Provider> = Arc::clone(&mock) as Arc<dyn Provider>;
        let mut params = params_for_test(store.clone(), parent_session.clone());
        params.parent_options.streaming = true;

        let spawn = AgentSpawnTool {
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params.clone().on_provider(Arc::clone(&provider)),
            inherited_scope: ToolScope::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        };
        let running = tokio::spawn(async move {
            spawn
                .execute(
                    serde_json::json!({ "prompt": "look into it", "permission": "read" }),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect("spawn succeeds")
                .text_content()
        });

        // The worker's row exists before its turn runs, which is when the parent can name it.
        let mut child = None;
        for _ in 0..250 {
            let tree = store.load_session_tree(parent_sid).await.expect("tree");
            child = tree
                .iter()
                .find(|row| row.parent_id == Some(parent_sid))
                .map(|row| row.id);
            if child.is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let child = child.expect("the worker's row appears while it runs");
        let steer = AgentSteerTool {
            tool_builder_params: params.clone(),
        };
        steer
            .execute(
                serde_json::json!({ "id": child.to_string(), "message": "narrow it to 2026" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("steer succeeds");
        steer_written.cancel();

        let text = running.await.expect("the spawn task");
        assert!(text.contains("done"), "{text}");
        let requests = mock.streams();
        assert_eq!(requests.len(), 2, "the tool round and the request after it");
        let boundary = requests[1]
            .messages
            .last()
            .expect("the second request ends on the round's results");
        assert!(
            matches!(
                boundary.content.first(),
                Some(crate::conversation::ContentBlock::ToolResult { .. })
            ),
            "the results lead the message: {boundary:?}"
        );
        let blocks = last_user_text_blocks(&requests[1].messages);
        assert_eq!(blocks.len(), 1, "one steer, one block: {blocks:?}");
        assert!(
            blocks[0].contains("while you were working]\nnarrow it to 2026"),
            "{}",
            blocks[0]
        );
    }

    /// With `interrupt`, the parent does not wait for the worker's current answer either: the
    /// stream is cut, what had streamed stays as the worker's answer so far, and the message is
    /// the next thing the worker reads, all inside the worker's one turn.
    #[tokio::test]
    async fn an_interrupt_from_the_parent_cuts_the_answer_a_worker_is_writing() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent session");
        let parent_session = crate::session::SharedSessionId::new(Some(parent_sid));
        let mock = Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
            vec![
                crate::provider::mock::MockEvent::Text {
                    text: "half of the report".to_string(),
                },
                crate::provider::mock::MockEvent::Sleep { ms: 5000 },
                crate::provider::mock::MockEvent::Text {
                    text: " and the rest, never sent".to_string(),
                },
                crate::provider::mock::MockEvent::MessageEnd {
                    stop_reason: crate::provider::mock::MockStopReason::EndTurn,
                },
            ],
            text_round("done"),
        ]));
        let provider: Arc<dyn Provider> = Arc::clone(&mock) as Arc<dyn Provider>;
        let mut params = params_for_test(store.clone(), parent_session.clone());
        params.parent_options.streaming = true;

        let spawn = AgentSpawnTool {
            parent_permission: SharedPermission::new(
                Permission::Unrestricted,
                EnabledPermissions::ALL,
            ),
            tool_builder_params: params.clone().on_provider(Arc::clone(&provider)),
            inherited_scope: ToolScope::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        };
        let running = tokio::spawn(async move {
            spawn
                .execute(
                    serde_json::json!({ "prompt": "write the report", "permission": "read" }),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect("spawn succeeds")
                .text_content()
        });

        let mut child = None;
        for _ in 0..250 {
            let tree = store.load_session_tree(parent_sid).await.expect("tree");
            child = tree
                .iter()
                .find(|row| row.parent_id == Some(parent_sid))
                .map(|row| row.id);
            if child.is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let child = child.expect("the worker's row appears while it runs");
        // Past the first text, so there is an answer under way to cut.
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let steer = AgentSteerTool {
            tool_builder_params: params.clone(),
        };
        let queued = steer
            .execute(
                serde_json::json!({
                    "id": child.to_string(),
                    "message": "stop, summarize instead",
                    "interrupt": true
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("steer succeeds")
            .text_content();
        assert!(
            queued.contains("stops the answer it is writing"),
            "{queued}"
        );
        let cut_at = std::time::Instant::now();

        let text = running.await.expect("the spawn task");
        assert!(text.contains("done"), "{text}");
        assert!(
            cut_at.elapsed() < std::time::Duration::from_secs(3),
            "the worker did not wait out its stalled answer: {:?}",
            cut_at.elapsed()
        );
        let requests = mock.streams();
        assert_eq!(requests.len(), 2, "the cut request and the one after it");
        let tail = &requests[1].messages;
        let partial = &tail[tail.len() - 2];
        assert_eq!(partial.role, crate::conversation::Role::Assistant);
        assert_eq!(partial.text_content(), "half of the report");
        let blocks = last_user_text_blocks(tail);
        assert_eq!(
            blocks.len(),
            1,
            "the message is the next thing it reads: {blocks:?}"
        );
        assert!(
            blocks[0].starts_with("[Message from your parent agent, arrived ")
                && blocks[0].contains("while you were working]\nstop, summarize instead"),
            "{}",
            blocks[0]
        );
        assert!(
            store
                .inbox_store()
                .list_open(child)
                .await
                .expect("list")
                .is_empty(),
            "the request after the cut delivered it"
        );
    }

    #[tokio::test]
    async fn agent_steer_refuses_a_session_that_is_not_the_parent() {
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
            .create_child_session_for_test(
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
        let params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(stranger)),
        );
        let steer = AgentSteerTool {
            tool_builder_params: params,
        };
        let error = steer
            .execute(
                serde_json::json!({ "id": child.to_string(), "message": "hello" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a stranger's worker must be refused");
        assert!(error.to_string().contains("belongs to this session"));
        assert!(
            store
                .inbox_store()
                .list_open(child)
                .await
                .expect("list")
                .is_empty(),
            "a refused steer writes nothing"
        );
    }

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
            .create_child_session_for_test(
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

    fn child_row_for_test(id: Uuid, parent: Uuid) -> crate::store::SessionMetaRow {
        crate::store::SessionMetaRow {
            id,
            parent_id: Some(parent),
            created_at: String::new(),
            updated_at: String::new(),
            cwd: None,
            permission: None,
            approvals: false,
            capabilities_json: None,
            additional_roots: Vec::new(),
            subagent_spec_json: None,
            profile: "test-profile".to_string(),
            project: String::new(),
            title: None,
            pinned_at: None,
        }
    }

    /// The three id-taking tools resolve one way: the full id, or a prefix no other sub-agent of
    /// this session shares, folded for case like every other id meka resolves.
    #[test]
    fn a_sub_agent_id_resolves_in_full_or_by_a_prefix_unique_among_the_children() {
        let parent = Uuid::from_u128(0x1);
        let first = Uuid::from_u128(0xaaaaaaaa_0000_4000_8000_000000000001);
        let second = Uuid::from_u128(0xaaaabbbb_0000_4000_8000_000000000002);
        let children = || {
            vec![
                child_row_for_test(first, parent),
                child_row_for_test(second, parent),
            ]
        };

        let resolved = |wanted: &str| match_child(children(), wanted, "agent_steer");
        assert_eq!(resolved(&first.to_string()).expect("full id").id, first);
        assert_eq!(
            resolved(&first.to_string().to_uppercase())
                .expect("uppercase full id")
                .id,
            first
        );
        assert_eq!(resolved("aaaaa").expect("unique prefix").id, first);
        assert_eq!(resolved("AAAAB").expect("uppercase prefix").id, second);

        let ambiguous = resolved("aaaa").expect_err("a shared prefix names two");
        let text = ambiguous.to_string();
        assert!(text.contains("2 of this session's sub-agents"), "{text}");
        assert!(
            text.contains(&first.to_string()) && text.contains(&second.to_string()),
            "both are named so the model can pick: {text}"
        );

        let outsider = Uuid::from_u128(0xcccccccc_0000_4000_8000_000000000003);
        for wanted in ["my-agent", "cccc", &outsider.to_string()] {
            let error = resolved(wanted).expect_err("nothing here is a child");
            assert!(
                error.to_string().contains("belongs to this session"),
                "{error}"
            );
        }
    }

    /// The resolution reaches every door that takes an id: a steer by the printed short form lands
    /// in the inbox of the sub-agent it names, and a delete by it removes that one.
    #[tokio::test]
    async fn a_steer_and_a_delete_by_id_prefix_act_on_the_sub_agent_named() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let child = store
            .create_child_session_for_test(
                parent_sid,
                None,
                Vec::new(),
                Some("{\"permission\":\"read\"}".to_string()),
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child")
            .0;
        let short = child.to_string()[..crate::text::ID_PREFIX].to_string();
        let params = || {
            params_for_test(
                store.clone(),
                crate::session::SharedSessionId::new(Some(parent_sid)),
            )
        };

        let steer = AgentSteerTool {
            tool_builder_params: params(),
        };
        let queued = steer
            .execute(
                serde_json::json!({ "id": short, "message": "hello" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a steer by prefix")
            .text_content();
        assert!(queued.contains(&child.to_string()), "{queued}");
        assert_eq!(
            store
                .inbox_store()
                .list_open(child)
                .await
                .expect("list")
                .len(),
            1
        );

        let delete = AgentDeleteTool {
            tool_builder_params: params(),
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        };
        delete
            .execute(
                serde_json::json!({ "id": short }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a delete by prefix");
        assert!(
            store
                .load_session_tree(parent_sid)
                .await
                .expect("tree")
                .iter()
                .all(|row| row.id != child),
            "the sub-agent the prefix named is gone"
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
            .create_child_session_for_test(
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

    /// A parent at `level` working in `cwd`, for the scope tests. The spawn tool's
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
            inherited_scope: ToolScope::default(),
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

    /// Params whose installation configures `other` beside `test-profile` with the choice open,
    /// and `provider` standing in for every profile, the way `MEKA_MOCK_PROVIDER` does.
    fn params_with_profile_choice(
        store: Store,
        parent_session: crate::session::SharedSessionId,
        provider: Arc<dyn Provider>,
    ) -> ToolBuilderParams {
        let mut params = params_for_test(store.clone(), parent_session);
        let providers = crate::provider::ProviderRegistry::for_test(store.token_store(), &[
            "other",
            "test-profile",
        ]);
        providers.install_scripted(provider);
        params.materials.providers = Arc::new(providers);
        params.materials.subagents.agent_chosen_profile = true;
        params
    }

    /// What `agent_spawn` is registered with: nothing while the choice is closed, and exactly the
    /// configured names, in name order, once it is open. Pinned here because the schema test below
    /// takes its names as given.
    #[tokio::test]
    async fn profile_choices_are_the_configured_names_only_while_the_choice_is_open() {
        let store = store_for_test().await;
        let mut materials = crate::session::SessionMaterials::for_test(store.clone());
        materials.providers = Arc::new(crate::provider::ProviderRegistry::for_test(
            store.token_store(),
            &["test-profile", "other"],
        ));
        assert!(
            profile_choices(&materials).is_empty(),
            "closed: the parameter must not exist"
        );
        materials.subagents.agent_chosen_profile = true;
        assert_eq!(
            profile_choices(&materials),
            vec!["other".to_string(), "test-profile".to_string()],
            "open: every configured name, sorted"
        );
    }

    /// The registered tool's schema, not just the free function behind it: the tool derives its
    /// choices from the session materials, so what the model is offered follows the switch and
    /// the configured names.
    #[tokio::test]
    async fn the_registered_tool_offers_profiles_only_while_the_choice_is_open() {
        let store = store_for_test().await;
        let provider: Arc<dyn Provider> = mock(Vec::new());
        let open = spawn_tool_for(
            params_with_profile_choice(
                store.clone(),
                crate::session::SharedSessionId::new(None),
                Arc::clone(&provider),
            ),
            Arc::clone(&provider),
        );
        assert_eq!(
            open.definition().parameters["properties"]["profile"]["enum"],
            serde_json::json!(["other", "test-profile"])
        );
        let closed = spawn_tool_for(
            params_for_test(store, crate::session::SharedSessionId::new(None)),
            provider,
        );
        assert!(
            closed.definition().parameters["properties"]
                .get("profile")
                .is_none(),
            "closed: the parameter must not exist"
        );
    }

    /// The `profile` parameter exists only while `[subagents].agent_chosen_profile` is on, so a
    /// model is never offered a choice the operator closed, and is offered exactly the configured
    /// names when it is open.
    #[test]
    fn the_schema_offers_profile_only_when_the_choice_is_open() {
        let closed = agent_spawn_definition(&[]);
        assert!(
            closed.parameters["properties"].get("profile").is_none(),
            "{closed:?}"
        );
        assert!(
            !closed.description.contains("`profile`"),
            "{}",
            closed.description
        );

        let open = agent_spawn_definition(&["deep".to_string(), "fast".to_string()]);
        assert_eq!(
            open.parameters["properties"]["profile"]["enum"],
            serde_json::json!(["deep", "fast"]),
            "{open:?}"
        );
        assert!(
            open.description.contains("`profile`"),
            "{}",
            open.description
        );
    }

    /// The one door both `agent_spawn` and `agent_followup` build from: a chosen profile resolves
    /// by name, an unnamed one is the parent's own profile, a fallback serving the parent's turn
    /// notwithstanding, and a name the installation does not configure is refused by name.
    #[tokio::test]
    async fn a_worker_binding_follows_the_spec_then_the_parent() {
        let store = store_for_test().await;
        let provider: Arc<dyn Provider> = mock(Vec::new());
        let params = params_with_profile_choice(
            store,
            crate::session::SharedSessionId::new(None),
            Arc::clone(&provider),
        )
        .on_provider(provider);

        let pinned = worker_binding(&params, Some("other"), "agent_spawn")
            .await
            .expect("a configured profile resolves");
        assert_eq!(pinned.profile, "other");

        let following = worker_binding(&params, None, "agent_spawn")
            .await
            .expect("the parent's own profile resolves");
        assert_eq!(following.profile, "test-profile");

        params.cells.profile.serve(&pinned);
        let during_a_fallback = worker_binding(&params, None, "agent_spawn")
            .await
            .expect("the parent's own profile resolves");
        assert_eq!(
            during_a_fallback.profile, "test-profile",
            "a fallback serving the parent's turn is not what a worker is bound to"
        );
        params.cells.profile.restore();

        let refused = match worker_binding(&params, Some("missing"), "agent_followup").await {
            Ok(binding) => panic!(
                "an unconfigured profile must be refused, got '{}'",
                binding.profile
            ),
            Err(error) => error,
        };
        assert!(refused.to_string().contains("missing"), "{refused}");
    }

    /// A spawn that names a profile records it on the worker's row and in its spec, and leaves the
    /// parent where it was.
    #[tokio::test]
    async fn a_chosen_profile_is_recorded_on_the_row_and_in_the_spec() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let provider: Arc<dyn Provider> = mock(vec![text_round("done on the other profile")]);
        let params = params_with_profile_choice(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Arc::clone(&provider),
        );
        let spawn = spawn_tool_for(params, provider);
        let output = spawn
            .execute(
                serde_json::json!({ "prompt": "look into it", "profile": "other" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        let agent_id = agent_id_in(&output.text_content());

        assert_eq!(
            store
                .recorded_profile(agent_id)
                .await
                .expect("row")
                .as_deref(),
            Some("other"),
            "the row is the billing record"
        );
        let spec: SubagentSpec = serde_json::from_str(
            &store
                .load_subagent_spec(agent_id)
                .await
                .expect("load spec")
                .expect("a worker has a spec"),
        )
        .expect("decode");
        assert_eq!(
            spec.profile.as_deref(),
            Some("other"),
            "the spec carries the choice for every follow-up"
        );
        assert_eq!(
            store
                .recorded_profile(parent_sid)
                .await
                .expect("row")
                .as_deref(),
            Some("test-profile"),
            "the parent stays where it was"
        );
    }

    /// With the choice closed the parameter is not offered, so a value is a model reaching for
    /// something it was not given: refused rather than read as absent, and nothing is written.
    #[tokio::test]
    async fn a_profile_is_refused_while_the_choice_is_closed() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        let spawn = spawn_tool_for(params, mock(vec![text_round("never runs")]));
        let error = spawn
            .execute(
                serde_json::json!({ "prompt": "look into it", "profile": "other" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("the choice is closed");
        assert!(
            error.to_string().contains("agent_chosen_profile"),
            "{error}"
        );
        assert!(
            only_the_parent(&store, parent_sid).await,
            "a refused spawn must leave no child row behind"
        );
    }

    /// A name the installation does not configure is refused ahead of the row, with the text every
    /// other door gives an unknown profile.
    #[tokio::test]
    async fn an_unconfigured_profile_is_refused_before_the_row() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let provider: Arc<dyn Provider> = mock(vec![text_round("never runs")]);
        let params = params_with_profile_choice(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Arc::clone(&provider),
        );
        let spawn = spawn_tool_for(params, provider);
        let error = spawn
            .execute(
                serde_json::json!({ "prompt": "look into it", "profile": "missing" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("an unconfigured profile is refused");
        assert!(error.to_string().contains("missing"), "{error}");
        assert!(
            only_the_parent(&store, parent_sid).await,
            "a refused spawn must leave no child row behind"
        );
    }

    /// A worker spawned with a profile is rebuilt on it by every follow-up, whatever its parent
    /// runs on by then, and its row stays on it; one spawned without keeps following the
    /// parent, which the follow-up tests above pin.
    #[tokio::test]
    async fn a_pinned_worker_keeps_its_profile_on_followup() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let parent_session = crate::session::SharedSessionId::new(Some(parent_sid));
        let provider: Arc<dyn Provider> = mock(vec![
            text_round("first answer"),
            text_round("second answer"),
        ]);
        let params =
            params_with_profile_choice(store.clone(), parent_session, Arc::clone(&provider));
        let spawn = spawn_tool_for(params.clone(), Arc::clone(&provider));
        let output = spawn
            .execute(
                serde_json::json!({ "prompt": "look into it", "profile": "other" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        let agent_id = agent_id_in(&output.text_content());

        // The parent runs on `test-profile`, which is what an unpinned worker would be moved to.
        let followup = followup_tool_for(params, provider);
        let output = followup
            .execute(
                serde_json::json!({ "id": agent_id.to_string(), "prompt": "carry on" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("the follow-up runs");
        assert!(output.text_content().contains("second answer"));
        assert_eq!(
            store
                .recorded_profile(agent_id)
                .await
                .expect("row")
                .as_deref(),
            Some("other"),
            "a pinned worker's row does not follow the parent"
        );
    }

    /// The same rule at every depth: a pinned worker's own child, spawned without a choice, runs on
    /// the pinned profile rather than on what the root runs on, so the choice holds through the
    /// subtree.
    #[tokio::test]
    async fn a_pinned_workers_child_defaults_to_the_pinned_profile() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        // The mock serves every agent in call order: the worker asks for a child, the child
        // answers, the worker reports.
        let provider: Arc<dyn Provider> = mock(vec![
            vec![
                crate::provider::mock::MockEvent::ToolUseStart {
                    id: "call-1".into(),
                    name: "agent_spawn".into(),
                },
                crate::provider::mock::MockEvent::ToolUseEnd {
                    input: serde_json::json!({ "prompt": "the smaller task" }),
                },
                crate::provider::mock::MockEvent::MessageEnd {
                    stop_reason: crate::provider::mock::MockStopReason::ToolUse,
                },
            ],
            text_round("the child's answer"),
            text_round("the worker's report"),
        ]);
        let params = params_with_profile_choice(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Arc::clone(&provider),
        );
        let spawn = AgentSpawnTool {
            // Room for one more level, so the worker holds an `agent_spawn` of its own.
            remaining_depth: 2,
            ..spawn_tool_for(params, provider)
        };
        let output = spawn
            .execute(
                serde_json::json!({ "prompt": "look into it", "profile": "other" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        let worker = agent_id_in(&output.text_content());

        let tree = store.load_session_tree(parent_sid).await.expect("tree");
        let grandchild = tree
            .iter()
            .find(|row| row.parent_id == Some(worker))
            .unwrap_or_else(|| panic!("the worker must have spawned a child: {output:?}"));
        assert_eq!(
            store
                .recorded_profile(grandchild.id)
                .await
                .expect("row")
                .as_deref(),
            Some("other"),
            "a pinned worker's child runs on the pinned profile"
        );
    }

    /// [`followup_tool_for`], with the parent running on `profile` rather than `test-profile`: the
    /// switch a `/profile` between the spawn and the follow-up makes.
    fn followup_tool_on(
        mut params: ToolBuilderParams,
        provider: Arc<dyn Provider>,
        profile: &str,
    ) -> AgentFollowupTool {
        params.cells.profile = binding_named(provider, profile);
        AgentFollowupTool {
            parent_permission: params.cells.permission.clone(),
            tool_builder_params: params,
            in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        }
    }

    /// One spawn with `input`, returning the worker's id.
    async fn spawned(spawn: &AgentSpawnTool, input: serde_json::Value) -> Uuid {
        let output = spawn
            .execute(
                input,
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        agent_id_in(&output.text_content())
    }

    /// One follow-up on `agent_id`, returning what the worker said.
    async fn followed_up(followup: &AgentFollowupTool, agent_id: Uuid) -> String {
        followup
            .execute(
                serde_json::json!({ "id": agent_id.to_string(), "prompt": "carry on" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("the follow-up runs")
            .text_content()
    }

    async fn row_profile(store: &Store, id: Uuid) -> Option<String> {
        store.recorded_profile(id).await.expect("read the row")
    }

    /// The parent switches profile after spawning two workers. The one it pinned, even to the
    /// profile the parent was on at the time, stays; the one it left unpinned follows.
    #[tokio::test]
    async fn after_the_parent_switches_a_pinned_worker_stays_and_an_unpinned_one_follows() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let provider: Arc<dyn Provider> = mock(vec![
            text_round("pinned spawned"),
            text_round("unpinned spawned"),
            text_round("pinned followed"),
            text_round("unpinned followed"),
        ]);
        let params = params_with_profile_choice(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Arc::clone(&provider),
        );
        let spawn = spawn_tool_for(params.clone(), Arc::clone(&provider));
        let pinned = spawned(
            &spawn,
            serde_json::json!({ "prompt": "hold still", "profile": "test-profile" }),
        )
        .await;
        let unpinned = spawned(&spawn, serde_json::json!({ "prompt": "come along" })).await;
        assert_eq!(
            row_profile(&store, pinned).await.as_deref(),
            Some("test-profile")
        );
        assert_eq!(
            row_profile(&store, unpinned).await.as_deref(),
            Some("test-profile")
        );

        // `/profile other` in the parent.
        let followup = followup_tool_on(params, provider, "other");
        followed_up(&followup, pinned).await;
        followed_up(&followup, unpinned).await;
        assert_eq!(
            row_profile(&store, pinned).await.as_deref(),
            Some("test-profile"),
            "a pinned worker stays where its spawn put it"
        );
        assert_eq!(
            row_profile(&store, unpinned).await.as_deref(),
            Some("other"),
            "an unpinned worker follows the parent"
        );
    }

    /// A session runs on the profile its row names. When an explicit act (an import onto another
    /// profile) moves a pinned worker's row, the next follow-up runs there and leaves the row
    /// there, rather than dragging it back to the name the spawn recorded.
    #[tokio::test]
    async fn a_pinned_workers_row_moved_by_an_explicit_act_is_where_it_runs() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let provider: Arc<dyn Provider> = mock(vec![text_round("spawned"), text_round("followed")]);
        let params = params_with_profile_choice(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Arc::clone(&provider),
        );
        let spawn = spawn_tool_for(params.clone(), Arc::clone(&provider));
        let worker = spawned(
            &spawn,
            serde_json::json!({ "prompt": "look into it", "profile": "other" }),
        )
        .await;

        store
            .update_session(worker, crate::store::SessionPatch {
                profile: Some("test-profile".to_string()),
                ..Default::default()
            })
            .await
            .expect("move the row");
        // The parent is on `other` now, so following the parent and reverting to the spec would
        // both land on `other`; only honoring the row lands on `test-profile`.
        let followup = followup_tool_on(params, provider, "other");
        followed_up(&followup, worker).await;
        assert_eq!(
            row_profile(&store, worker).await.as_deref(),
            Some("test-profile"),
            "the row is where the worker runs, and the follow-up leaves it there"
        );
        let spec: SubagentSpec = serde_json::from_str(
            &store
                .load_subagent_spec(worker)
                .await
                .expect("load spec")
                .expect("a worker has a spec"),
        )
        .expect("decode");
        assert_eq!(
            spec.profile.as_deref(),
            Some("other"),
            "the spec keeps recording the choice the spawn made"
        );
    }

    /// The pinned profile is removed from the config between the spawn and the follow-up: the
    /// follow-up is refused by name, as resuming a root session on a deleted profile is, and the
    /// row is left alone rather than moved onto whatever the parent runs on.
    #[tokio::test]
    async fn a_pinned_profile_gone_from_the_config_refuses_the_followup_and_leaves_the_row() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let provider: Arc<dyn Provider> = mock(vec![text_round("spawned")]);
        let params = params_with_profile_choice(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Arc::clone(&provider),
        );
        let spawn = spawn_tool_for(params.clone(), Arc::clone(&provider));
        let worker = spawned(
            &spawn,
            serde_json::json!({ "prompt": "look into it", "profile": "other" }),
        )
        .await;

        // The operator removes `[profiles.other]` and restarts.
        let mut params = params;
        let providers =
            crate::provider::ProviderRegistry::for_test(store.token_store(), &["test-profile"]);
        providers.install_scripted(Arc::clone(&provider));
        params.materials.providers = Arc::new(providers);
        let followup = followup_tool_for(params, provider);
        let error = followup
            .execute(
                serde_json::json!({ "id": worker.to_string(), "prompt": "carry on" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a profile that is gone refuses the follow-up");
        assert!(error.to_string().contains("other"), "{error}");
        assert_eq!(
            row_profile(&store, worker).await.as_deref(),
            Some("other"),
            "a refused follow-up moves nothing"
        );
    }

    /// A configured profile whose provider cannot be built (here, an account with no credential)
    /// refuses the spawn ahead of the row, like an unconfigured name does.
    #[tokio::test]
    async fn a_profile_whose_provider_cannot_be_built_is_refused_before_the_row() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let mut params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        // Configured, but nothing scripted stands in for it and no credential was ever stored.
        params.materials.providers = Arc::new(crate::provider::ProviderRegistry::for_test(
            store.token_store(),
            &["other", "test-profile"],
        ));
        params.materials.subagents.agent_chosen_profile = true;
        let spawn = spawn_tool_for(params, mock(vec![text_round("never runs")]));
        let error = spawn
            .execute(
                serde_json::json!({ "prompt": "look into it", "profile": "other" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a provider that cannot be built refuses the spawn");
        assert!(error.to_string().contains("other"), "{error}");
        assert!(
            only_the_parent(&store, parent_sid).await,
            "a refused spawn must leave no child row behind"
        );
    }

    /// The same choice at every depth: a pinned worker may pin its own child to a third profile,
    /// and neither row moves the other.
    #[tokio::test]
    async fn a_pinned_worker_may_pin_its_own_child_elsewhere() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let provider: Arc<dyn Provider> = mock(vec![
            vec![
                crate::provider::mock::MockEvent::ToolUseStart {
                    id: "call-1".into(),
                    name: "agent_spawn".into(),
                },
                crate::provider::mock::MockEvent::ToolUseEnd {
                    input: serde_json::json!({
                        "prompt": "the smaller task",
                        "profile": "test-profile",
                    }),
                },
                crate::provider::mock::MockEvent::MessageEnd {
                    stop_reason: crate::provider::mock::MockStopReason::ToolUse,
                },
            ],
            text_round("the child's answer"),
            text_round("the worker's report"),
        ]);
        let params = params_with_profile_choice(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Arc::clone(&provider),
        );
        let spawn = AgentSpawnTool {
            remaining_depth: 2,
            ..spawn_tool_for(params, provider)
        };
        let worker = spawned(
            &spawn,
            serde_json::json!({ "prompt": "look into it", "profile": "other" }),
        )
        .await;

        let tree = store.load_session_tree(parent_sid).await.expect("tree");
        let grandchild = tree
            .iter()
            .find(|row| row.parent_id == Some(worker))
            .expect("the worker must have spawned a child");
        assert_eq!(row_profile(&store, worker).await.as_deref(), Some("other"));
        assert_eq!(
            row_profile(&store, grandchild.id).await.as_deref(),
            Some("test-profile"),
            "the grandchild runs where its own spawn call put it"
        );
    }

    /// An unpinned worker's children follow the root's live profile too: the root switches after
    /// the worker exists, and a child the worker spawns on its next follow-up lands on the new
    /// profile, as does the worker itself.
    #[tokio::test]
    async fn an_unpinned_workers_child_follows_the_roots_switch() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let provider: Arc<dyn Provider> = mock(vec![
            text_round("spawned"),
            vec![
                crate::provider::mock::MockEvent::ToolUseStart {
                    id: "call-1".into(),
                    name: "agent_spawn".into(),
                },
                crate::provider::mock::MockEvent::ToolUseEnd {
                    input: serde_json::json!({ "prompt": "the smaller task" }),
                },
                crate::provider::mock::MockEvent::MessageEnd {
                    stop_reason: crate::provider::mock::MockStopReason::ToolUse,
                },
            ],
            text_round("the child's answer"),
            text_round("the worker's report"),
        ]);
        let params = params_with_profile_choice(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Arc::clone(&provider),
        );
        let spawn = AgentSpawnTool {
            remaining_depth: 2,
            ..spawn_tool_for(params.clone(), Arc::clone(&provider))
        };
        let worker = spawned(&spawn, serde_json::json!({ "prompt": "look into it" })).await;
        assert_eq!(
            row_profile(&store, worker).await.as_deref(),
            Some("test-profile")
        );

        // `/profile other` in the root, then the worker is asked again and delegates.
        let followup = followup_tool_on(params, provider, "other");
        followed_up(&followup, worker).await;
        let tree = store.load_session_tree(parent_sid).await.expect("tree");
        let grandchild = tree
            .iter()
            .find(|row| row.parent_id == Some(worker))
            .expect("the worker must have spawned a child");
        assert_eq!(
            row_profile(&store, worker).await.as_deref(),
            Some("other"),
            "the worker followed the root"
        );
        assert_eq!(
            row_profile(&store, grandchild.id).await.as_deref(),
            Some("other"),
            "and so did the child it spawned afterwards"
        );
    }

    /// A blank `profile` is "not specified", like every other optional string: the worker follows
    /// its parent and nothing is pinned.
    #[tokio::test]
    async fn a_blank_profile_leaves_the_worker_unpinned() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let provider: Arc<dyn Provider> = mock(vec![text_round("spawned")]);
        let params = params_with_profile_choice(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Arc::clone(&provider),
        );
        let spawn = spawn_tool_for(params, provider);
        let worker = spawned(
            &spawn,
            serde_json::json!({ "prompt": "look into it", "profile": "   " }),
        )
        .await;
        assert_eq!(
            row_profile(&store, worker).await.as_deref(),
            Some("test-profile")
        );
        let spec: SubagentSpec = serde_json::from_str(
            &store
                .load_subagent_spec(worker)
                .await
                .expect("load spec")
                .expect("a worker has a spec"),
        )
        .expect("decode");
        assert_eq!(spec.profile, None, "blank is not a choice");
    }

    /// An unpinned worker hands its children the root's *live* handle, not a snapshot taken when
    /// the worker was built: the root switches while the worker's turn is under way, and the child
    /// the worker spawns during that turn lands on the new profile.
    #[tokio::test]
    async fn a_root_switch_during_a_workers_turn_reaches_the_child_it_spawns() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        // The worker delegates once, the child answers, the worker reports.
        let provider: Arc<dyn Provider> = mock(vec![
            vec![
                crate::provider::mock::MockEvent::ToolUseStart {
                    id: "call-1".into(),
                    name: "agent_spawn".into(),
                },
                crate::provider::mock::MockEvent::ToolUseEnd {
                    input: serde_json::json!({ "prompt": "the smaller task" }),
                },
                crate::provider::mock::MockEvent::MessageEnd {
                    stop_reason: crate::provider::mock::MockStopReason::ToolUse,
                },
            ],
            text_round("the child's answer"),
            text_round("the worker's report"),
        ]);
        let params = params_with_profile_choice(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Arc::clone(&provider),
        )
        .on_provider(Arc::clone(&provider));
        let spec = SubagentSpec {
            inherited_scratchpad: Vec::new(),
            ..spec_for_test(Permission::Read)
        };
        let (worker_id, lock) = store
            .create_child_session_for_test(
                parent_sid,
                None,
                Vec::new(),
                Some(serde_json::to_string(&spec).expect("encode the spec")),
                "read".to_string(),
                "test-profile".to_string(),
            )
            .await
            .expect("child row");
        let _lock = lock.expect("the spawn's own claim");
        let worker = build_subagent(
            &params,
            &spec,
            WorkerPlacement {
                workspace: WorkerWorkspace {
                    cwd: crate::workspace::cwd_for_test(),
                    roots: crate::workspace::roots_for_test(),
                    read_only_roots: crate::workspace::SharedRoots::default(),
                    read_boundary: crate::workspace::ReadBoundary::unconfined(),
                    network: crate::config::Network::Host,
                },
                binding: params.cells.profile.current(),
            },
            parent_sid,
            worker_id,
            "agent_spawn",
            &crate::tools::ToolContext::detached(CancellationToken::new()),
        )
        .await
        .expect("build the worker");

        // `/profile other` in the root, after the worker exists and before it delegates.
        params
            .cells
            .profile
            .store(&binding_named(Arc::clone(&provider), "other").current());
        let mut messages = crate::conversation::Conversation::new();
        worker
            .run_turn(
                &mut messages,
                crate::agent::TurnInput::from_parts("look into it".to_string(), Vec::new())
                    .expect("a prompt"),
                crate::agent::TurnCancellation::new(),
            )
            .await
            .expect("the worker's turn runs");

        let tree = store.load_session_tree(parent_sid).await.expect("tree");
        let grandchild = tree
            .iter()
            .find(|row| row.parent_id == Some(worker_id))
            .expect("the worker must have spawned a child");
        assert_eq!(
            row_profile(&store, grandchild.id).await.as_deref(),
            Some("other"),
            "the child took the root's live profile, not the worker's snapshot"
        );
    }

    /// One round in which the worker writes `content` to `path` through `file_write`.
    fn write_file_round(
        id: &str,
        path: &std::path::Path,
        content: &str,
    ) -> Vec<crate::provider::mock::MockEvent> {
        vec![
            crate::provider::mock::MockEvent::ToolUseStart {
                id: id.to_string(),
                name: "file_write".to_string(),
            },
            crate::provider::mock::MockEvent::ToolUseEnd {
                input: serde_json::json!({ "path": path.to_string_lossy(), "content": content }),
            },
            crate::provider::mock::MockEvent::MessageEnd {
                stop_reason: crate::provider::mock::MockStopReason::ToolUse,
            },
        ]
    }

    fn read_file_round(id: &str, path: &std::path::Path) -> Vec<crate::provider::mock::MockEvent> {
        vec![
            crate::provider::mock::MockEvent::ToolUseStart {
                id: id.to_string(),
                name: "file_read".to_string(),
            },
            crate::provider::mock::MockEvent::ToolUseEnd {
                input: serde_json::json!({ "path": path.to_string_lossy() }),
            },
            crate::provider::mock::MockEvent::MessageEnd {
                stop_reason: crate::provider::mock::MockStopReason::ToolUse,
            },
        ]
    }

    /// The parent's params with its reads confined to its own roots, which every worker it
    /// spawns inherits.
    fn confine_reads(params: &mut ToolBuilderParams) {
        params.cells.read_boundary = crate::workspace::ReadBoundary {
            confined: true,
            base: Vec::new(),
        };
    }

    fn agent_id_in(text: &str) -> Uuid {
        text.lines()
            .find_map(|line| line.strip_prefix("Agent: "))
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

    /// The directories the scope tests share: a parent working in `work`, two
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

    /// The feature end to end: a `workspace` parent hands a worker a working directory and a
    /// second workspace directory inside its own workspace, and the worker writes under those and
    /// nowhere else: not beside them in the parent's directory, and not under a directory the
    /// parent holds and did not pass on. Its environment context says where it works, the row
    /// and the spec both record the scope, and the result opens with it.
    #[tokio::test]
    async fn a_shaped_worker_writes_under_its_workspace_and_nowhere_else() {
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
                    "working_directory": dirs.sub.to_string_lossy(),
                    "additional_workspace_directories": [dirs.extra.to_string_lossy()],
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
        // Once, in the turn's own context block. The words the parent hands over used to carry a
        // second copy from before workers ran through the shared turn path.
        assert_eq!(
            first_turn.matches("[Environment context]").count(),
            1,
            "the worker reads its environment once: {first_turn}"
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
            "a shaped worker with no `permission` given runs at the parent's level"
        );
        assert_eq!(spec.working_directory.as_deref(), Some(dirs.sub.as_path()));
        assert_eq!(spec.additional_workspace_directories, vec![
            dirs.extra.clone()
        ]);
        let text = output.text_content();
        assert!(
            text.starts_with(&format!(
                "Agent: {agent_id}\nPermission: workspace\nWorking directory: {}\nAdditional \
                 workspace directories:\n  {}\n",
                dirs.sub.display(),
                dirs.extra.display()
            )),
            "the result opens with the scope the worker holds: {text}"
        );
    }

    /// A `workspace` parent cannot name a directory its own workspace does not contain: the
    /// refusal names the field and the entry, and comes before the row.
    #[tokio::test]
    async fn a_directory_outside_the_parents_workspace_is_refused_before_any_row_exists() {
        let dirs = workspaces();
        let refusal = refused_spawn(
            Permission::Workspace,
            dirs.work.clone(),
            serde_json::json!({
                "prompt": "write",
                "working_directory": dirs.elsewhere.to_string_lossy(),
            }),
        )
        .await;
        assert!(
            refusal.contains("`working_directory`")
                && refusal.contains(&dirs.elsewhere.display().to_string())
                && refusal.contains("outside this session's workspace"),
            "{refusal}"
        );
    }

    /// A level above the parent's is refused by name rather than clamped: a worker at a level the
    /// call did not ask for is a worker whose scope the parent misreads.
    #[tokio::test]
    async fn a_level_above_the_parents_is_refused() {
        let dirs = workspaces();
        let refusal = refused_spawn(
            Permission::Read,
            dirs.work.clone(),
            serde_json::json!({ "prompt": "write", "permission": "workspace" }),
        )
        .await;
        assert!(
            refusal.contains("above your own level `read`") && refusal.contains("name `read`"),
            "{refusal}"
        );
    }

    /// `unrestricted` is confined by nothing, so a list beside it, asked for or inherited, is a
    /// contradiction rather than a worker that silently runs at one or the other; a working
    /// directory alone is where the worker starts.
    #[tokio::test]
    async fn a_list_beside_an_unrestricted_level_is_refused() {
        let dirs = workspaces();
        for input in [
            serde_json::json!({
                "prompt": "write",
                "permission": "unrestricted",
                "working_directory": dirs.work.to_string_lossy(),
                "additional_workspace_directories": [dirs.sub.to_string_lossy()],
            }),
            serde_json::json!({
                "prompt": "read",
                "working_directory": ".",
                "additional_readable_directories": [dirs.sub.to_string_lossy()],
            }),
        ] {
            let refusal = refused_spawn(Permission::Unrestricted, dirs.work.clone(), input).await;
            assert!(refusal.contains("name a lower level"), "{refusal}");
        }

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
        let spawn = spawn_tool_for(params, mock(vec![text_round("done")]));
        let output = spawn
            .execute(
                serde_json::json!({
                    "prompt": "look",
                    "working_directory": dirs.elsewhere.to_string_lossy(),
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a working directory alone places an unrestricted worker");
        let agent_id = agent_id_in(&output.text_content());
        let row = store
            .load_session_tree(parent_sid)
            .await
            .expect("tree")
            .into_iter()
            .find(|row| row.id == agent_id)
            .expect("the worker's row");
        assert_eq!(row.cwd.as_deref(), Some(dirs.elsewhere.as_path()));
        assert_eq!(row.permission, Some(Permission::Unrestricted));
    }

    /// An empty list counts as absent, so a call carrying one is a bare spawn; a list that names
    /// something beside no working directory is refused with the remedy, before any row; and a
    /// list with an entry that is not a string is refused rather than read as empty, since empty
    /// beside no working directory would be a bare spawn with the parent's whole scope.
    #[tokio::test]
    async fn an_empty_list_counts_as_absent_and_a_list_needs_a_working_directory() {
        let dirs = workspaces();
        let refusal = refused_spawn(
            Permission::Workspace,
            dirs.work.clone(),
            serde_json::json!({
                "prompt": "read",
                "additional_readable_directories": [dirs.elsewhere.to_string_lossy()],
            }),
        )
        .await;
        assert!(
            refusal.contains("name `working_directory`") && refusal.contains("\".\""),
            "{refusal}"
        );
        let refusal = refused_spawn(
            Permission::Workspace,
            dirs.work.clone(),
            serde_json::json!({
                "prompt": "read",
                "additional_readable_directories": [{"path": dirs.elsewhere.to_string_lossy()}],
            }),
        )
        .await;
        assert!(
            refusal.contains("must be a string"),
            "a malformed list is refused, never read as a bare spawn: {refusal}"
        );

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
                serde_json::json!({
                    "prompt": "look",
                    "additional_workspace_directories": [],
                    "additional_readable_directories": [],
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("empty lists are a bare spawn");
        let agent_id = agent_id_in(&output.text_content());
        let spec: SubagentSpec = serde_json::from_str(
            &store
                .load_subagent_spec(agent_id)
                .await
                .expect("load")
                .expect("spec"),
        )
        .expect("decode");
        assert_eq!(spec.stated_scope().expect("readable"), None);
    }

    /// An `unrestricted` parent may write anywhere, so it may place a worker in a directory its
    /// own working directory does not contain; asked for `workspace`, the worker is confined to
    /// that directory.
    #[tokio::test]
    async fn an_unrestricted_parent_may_place_a_worker_outside_its_own_directory() {
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
                    "permission": "workspace",
                    "working_directory": dirs.elsewhere.to_string_lossy(),
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
            "the parent's own directory is not in the worker's scope"
        );
        let row = store
            .load_session_tree(parent_sid)
            .await
            .expect("tree")
            .into_iter()
            .find(|row| row.id == agent_id)
            .expect("the worker's row");
        assert_eq!(row.cwd.as_deref(), Some(dirs.elsewhere.as_path()));
        assert_eq!(row.permission, Some(Permission::Workspace));
    }

    /// A relative entry names a directory under the parent's working directory, not the process's.
    #[tokio::test]
    async fn a_relative_working_directory_resolves_against_the_parents_directory() {
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
                serde_json::json!({ "prompt": "look", "working_directory": "sub" }),
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

    /// Spawn a worker placed in `dirs.sub` under a `workspace` parent in `dirs.work` that also
    /// holds `dirs.elsewhere` as a workspace directory of its own, for the follow-up tests.
    /// Returns the parent's params (whose cells the follow-up shares) and the worker's id.
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
                    "working_directory": dirs.sub.to_string_lossy(),
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        (params, agent_id_in(&output.text_content()))
    }

    /// A follow-up rebuilds a shaped worker from its recorded scope, not from the parent's
    /// workspace: it still writes under its directory, and is still refused beside it and under a
    /// directory the parent holds.
    #[tokio::test]
    async fn a_followup_keeps_a_workers_scope() {
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

    /// The scope is put back through the acceptor against the parent's reach *now*, at the level
    /// the worker runs at now. A parent that has moved out from over it is refused rather than
    /// resuming a writer it could not spawn today, and the worker's conversation is left alone;
    /// one that has dropped to `read` resumes the worker as a reader.
    #[tokio::test]
    async fn a_followup_on_a_shaped_worker_follows_what_the_parent_can_still_grant() {
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
            .expect_err("a parent whose workspace no longer contains the scope cannot resume")
            .to_string();
        assert!(
            refusal.contains("can no longer grant")
                && refusal.contains("outside this session's workspace"),
            "{refusal}"
        );
        assert_eq!(
            store.load_events(agent_id).await.expect("events").len(),
            events_before,
            "a refusal may not have run a turn"
        );

        // Back over the directory, but dropped to `read`, as `/permission` or Shift+Tab would: the
        // scope is re-admitted at the level the worker runs at now, so the follow-up runs, and
        // the worker that could write its directory can only read it.
        params.cells.cwd.set(dirs.work.clone());
        params.cells.permission.set_unchecked(Permission::Read);
        let later = dirs.sub.join("later.txt");
        let followup = followup_tool_for(
            params,
            mock(vec![
                write_file_round("call-1", &later, "later"),
                text_round("tried"),
            ]),
        );
        let output = followup
            .execute(
                ask(),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a parent dropped to `read` resumes the worker as a reader");
        assert!(
            !later.exists(),
            "the worker's own directory is read-only at `read`"
        );
        assert!(
            output.text_content().contains("Permission: read"),
            "the result says the level the worker ran at: {}",
            output.text_content()
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
            task_id: None,
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
        // The id alone: Claude Code's sub-agent requests name neither an origin nor a position.
        assert_eq!(provider.completion_turn_origins(), vec![None]);
        assert_eq!(provider.completion_turn_positions(), vec![None]);
        // And they name the worker itself, in Claude Code's sub-agent id shape; a first-level
        // worker has no worker above it to name.
        let workers = provider.completion_workers();
        let identity = workers[0].as_ref().expect("a worker names itself");
        assert_eq!(identity.agent_id.len(), 17, "{}", identity.agent_id);
        assert!(identity.agent_id.starts_with('a'));
        assert!(
            identity.agent_id[1..]
                .chars()
                .all(|c| c.is_ascii_hexdigit())
        );
        assert_eq!(identity.parent_agent_id, None);

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

    /// A worker spawned by a worker names its spawner, the way Claude Code's nested sub-agent
    /// carries its parent's agent id; the spawner's id is the one its own requests carry.
    #[tokio::test]
    async fn a_nested_workers_requests_name_the_worker_that_spawned_it() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        let provider = mock(vec![text_round("spawned")]);
        let spawn = AgentSpawnTool {
            // The spawner sits one level down: it is a worker itself.
            absolute_depth: 1,
            remaining_depth: 2,
            ..spawn_tool_for(params, provider.clone())
        };
        spawn
            .execute(
                serde_json::json!({ "prompt": "look into it", "permission": "read" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        let workers = provider.completion_workers();
        let identity = workers[0].as_ref().expect("a worker names itself");
        assert_eq!(
            identity.parent_agent_id.as_deref(),
            Some(crate::provider::WorkerIdentity::agent_id_for(parent_sid).as_str())
        );
        assert_ne!(
            Some(identity.agent_id.as_str()),
            identity.parent_agent_id.as_deref()
        );
    }

    /// Whether a sub-agent streams is its parent's setting, not a rule of its own. `--no-stream`
    /// is the one lever for an endpoint that has no SSE, so a sub-agent that streamed regardless
    /// would fail on exactly the endpoint the flag was set for; and nothing of the sub-agent's
    /// output is rendered, so the flag is about the wire alone. Both directions, because a
    /// constant passes either one on its own.
    #[tokio::test]
    async fn a_sub_agent_streams_exactly_when_its_parent_does() {
        for parent_streams in [false, true] {
            let store = store_for_test().await;
            let parent_session_id = store
                .create_session(None, "test-profile".to_string())
                .await
                .expect("parent");
            let mut params = params_for_test(
                store.clone(),
                crate::session::SharedSessionId::new(Some(parent_session_id)),
            );
            params.parent_options.streaming = parent_streams;
            let provider = mock(vec![text_round("done")]);
            let spawn = spawn_tool_for(params, provider.clone());
            spawn
                .execute(
                    serde_json::json!({ "prompt": "draft a chapter", "permission": "read" }),
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect("spawn succeeds");
            assert_eq!(
                provider.streams().len(),
                usize::from(parent_streams),
                "streamed requests under a parent with streaming = {parent_streams}"
            );
            assert_eq!(
                provider.completions().len(),
                usize::from(!parent_streams),
                "whole-reply requests under a parent with streaming = {parent_streams}"
            );
        }
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

    #[tokio::test]
    async fn a_worker_retains_its_live_world_after_mid_turn_compaction() {
        let store = store_for_test().await;
        let parent = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let mut params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent)),
        );
        params.parent_options.streaming = true;
        params.parent_options.auto_compact = true;
        params.parent_options.compact_checkpoint = true;
        params.materials.memories =
            store.memory_store(true, crate::config::MEMORY_INLINE_MAX_BYTES_DEFAULT);
        params
            .materials
            .memories
            .write(crate::store::memory::WriteRequest {
                name: "worker-note".into(),
                description: Some("Review constraints".into()),
                priority: Some(0),
                tags: None,
                body: Some("Keep the fixture unchanged.".into()),
                load: Some(crate::memory::Load::Inline),
                ..crate::store::memory::WriteRequest::default()
            })
            .await
            .expect("memory");
        let manager = crate::mcp::McpClientManager::prepare(
            &[
                crate::config::McpServerConfig::for_test("visible"),
                crate::config::McpServerConfig::for_test("hidden"),
            ],
            None,
            None,
            crate::mcp::McpClientContext::new(),
            crate::config::DEFAULT_MCP_TOOL_TIMEOUT,
        )
        .await
        .expect("manager");
        manager
            .server_entry("visible")
            .expect("visible")
            .record_instructions(Some("VISIBLE SERVER GUIDANCE".into()));
        manager
            .server_entry("hidden")
            .expect("hidden")
            .record_instructions(Some("HIDDEN SERVER GUIDANCE".into()));
        params.materials.mcp_manager = Some(Arc::downgrade(&manager));
        use crate::provider::mock::{MockEvent, MockStopReason};
        let mock = Arc::new(crate::provider::mock::MockProvider::from_rounds(vec![
            vec![
                MockEvent::Usage {
                    input_tokens: 190_000,
                },
                MockEvent::ToolUseStart {
                    id: "load".into(),
                    name: "tool_load".into(),
                },
                MockEvent::ToolUseEnd {
                    input: serde_json::json!({"name":"file_read"}),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::ToolUse,
                },
            ],
            vec![
                MockEvent::ToolUseStart {
                    id: "summary".into(),
                    name: "context_replace".into(),
                },
                MockEvent::ToolUseEnd {
                    input: serde_json::json!({"summary":"Continue reviewing the fixture."}),
                },
                MockEvent::MessageEnd {
                    stop_reason: MockStopReason::ToolUse,
                },
            ],
            text_round("report"),
        ]));
        let spawn = AgentSpawnTool {
            parent_permission: params.cells.permission.clone(),
            tool_builder_params: params.on_provider(mock.clone()),
            inherited_scope: ToolScope::default(),
            remaining_depth: 1,
            absolute_depth: 0,
        };
        let output = spawn.execute(serde_json::json!({"prompt":"Review the fixture", "memory":"read",
            "deny_servers":["hidden"], "deny_tools":["checklist_add","checklist_edit","checklist_read","memory_search"]}),
            crate::tools::ToolContext::detached(CancellationToken::new())).await.expect("spawn");
        assert!(!output.is_error, "{}", output.text_content());
        let requests = mock.streams();
        assert_eq!(
            requests.len(),
            2,
            "the worker continues after the checkpoint"
        );
        assert_eq!(mock.completions().len(), 1, "one checkpoint");
        assert!(
            requests[1].messages[0]
                .wire_text()
                .contains("[Conversation summary from session compaction]")
        );
        for request in &requests {
            let context = request
                .messages
                .iter()
                .map(crate::conversation::Message::wire_text)
                .collect::<Vec<_>>()
                .join("\n");
            assert!(context.contains("[Execution context]"), "{context}");
            assert!(context.contains("Image input: enabled"), "{context}");
            assert!(context.contains("[Tool discovery]"), "{context}");
            assert!(context.contains("mcp_resource_list"), "{context}");
            assert!(context.contains("worker-note"), "{context}");
            assert!(context.contains("Keep the fixture unchanged."), "{context}");
            assert_eq!(
                context.matches("VISIBLE SERVER GUIDANCE").count(),
                1,
                "{context}"
            );
            assert!(!context.contains("HIDDEN SERVER GUIDANCE"), "{context}");
            for name in [
                "checklist_add",
                "checklist_edit",
                "checklist_read",
                "memory_search",
            ] {
                assert!(
                    !request.system_prompt.contains(name),
                    "{}",
                    request.system_prompt
                );
                assert!(!context.contains(name), "{context}");
                assert!(!request.tools.iter().any(|tool| tool.name == name));
            }
            let read = request
                .tools
                .iter()
                .find(|tool| tool.name == "file_read")
                .expect("read tool");
            assert!(!request.system_prompt.contains(&read.description));
        }
    }

    /// A worker placed in a directory and granted additional readable directories reads the
    /// granted one and is refused one the parent reads but did not pass on; the scope is recorded
    /// with the spawn terms.
    #[tokio::test]
    async fn a_worker_reads_its_additional_readable_directories_and_nothing_beyond() {
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
        confine_reads(&mut params);
        let granted = dirs.extra.join("granted.txt");
        let beside = dirs.work.join("beside.txt");
        std::fs::write(&granted, "GRANTED-WORDS").expect("write");
        std::fs::write(&beside, "BESIDE-WORDS").expect("write");
        let provider = mock(vec![
            read_file_round("call-1", &granted),
            read_file_round("call-2", &beside),
            text_round("done"),
        ]);
        let spawn = spawn_tool_for(params, Arc::clone(&provider) as Arc<dyn Provider>);
        let output = spawn
            .execute(
                serde_json::json!({
                    "prompt": "read",
                    "working_directory": dirs.sub.to_string_lossy(),
                    "additional_readable_directories": [dirs.extra.to_string_lossy()],
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        let agent_id = agent_id_in(&output.text_content());

        let transcript = format!("{:?}", store.load_events(agent_id).await.expect("events"));
        assert!(
            transcript.contains("GRANTED-WORDS"),
            "the granted directory is readable: {transcript}"
        );
        assert!(
            !transcript.contains("BESIDE-WORDS")
                && transcript.contains("outside the readable directories"),
            "a directory the parent reads but did not grant is refused: {transcript}"
        );
        let row = store
            .load_session_tree(parent_sid)
            .await
            .expect("tree")
            .into_iter()
            .find(|row| row.id == agent_id)
            .expect("the worker's row");
        let spec: SubagentSpec =
            serde_json::from_str(row.subagent_spec_json.as_deref().expect("spec")).expect("decode");
        assert_eq!(spec.working_directory.as_deref(), Some(dirs.sub.as_path()));
        assert_eq!(spec.additional_readable_directories, vec![
            dirs.extra.clone()
        ]);
        assert!(spec.additional_workspace_directories.is_empty());
        let text = output.text_content();
        assert!(
            text.starts_with(&format!("Agent: {agent_id}\nPermission: workspace\n"))
                && text.contains(&format!(
                    "Additional readable directories:\n  {}\n",
                    dirs.extra.display()
                )),
            "the result opens with the scope the worker holds: {text}"
        );
    }

    /// A parent whose reads are confined cannot grant a worker a directory outside what it reads,
    /// and the refusal leaves no row behind.
    #[tokio::test]
    async fn a_readable_directory_outside_the_parents_reach_is_refused() {
        let dirs = workspaces();
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let mut params = params_at(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Permission::Read,
            dirs.work.clone(),
        );
        confine_reads(&mut params);
        let spawn = spawn_tool_for(params, mock(vec![text_round("never runs")]));
        let error = spawn
            .execute(
                serde_json::json!({
                    "prompt": "read",
                    "working_directory": ".",
                    "additional_readable_directories": [dirs.elsewhere.to_string_lossy()],
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("the spawn must be refused");
        assert!(
            error
                .to_string()
                .contains("outside what this session reads"),
            "{error}"
        );
        assert!(only_the_parent(&store, parent_sid).await);
    }

    /// With the process's reads unconfined, a stated scope confines the worker on its own: a
    /// parent at `unrestricted` hands a worker a read boundary the way it hands one a write
    /// boundary. The worker reads the granted directory and the system directories, and is
    /// refused beside them.
    #[tokio::test]
    async fn a_scope_confines_a_workers_reads_when_the_process_does_not() {
        let dirs = workspaces();
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let mut params = params_at(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Permission::Unrestricted,
            dirs.work.clone(),
        );
        // The real base, resolved with the switch off, so the platform's paths come along.
        params.cells.read_boundary = crate::workspace::ReadBoundary::new(false, &[], &[]);
        assert!(!params.cells.read_boundary.confined);
        let granted = dirs.extra.join("granted.txt");
        let beside = dirs.work.join("beside.txt");
        std::fs::write(&granted, "GRANTED-WORDS").expect("write");
        std::fs::write(&beside, "BESIDE-WORDS").expect("write");
        let platform = crate::workspace::SYSTEM_READ_PATHS
            .iter()
            .map(std::path::Path::new)
            .find_map(|path| {
                std::fs::read_dir(path).ok()?.flatten().find_map(|entry| {
                    let path = entry.path();
                    (path.is_file() && std::fs::read(&path).is_ok()).then_some(path)
                })
            });
        let mut rounds = vec![
            read_file_round("call-1", &granted),
            read_file_round("call-2", &beside),
        ];
        if let Some(platform) = &platform {
            rounds.push(read_file_round("call-3", platform));
        }
        rounds.push(text_round("done"));
        let provider = mock(rounds);
        let spawn = spawn_tool_for(params, Arc::clone(&provider) as Arc<dyn Provider>);
        let output = spawn
            .execute(
                serde_json::json!({
                    "prompt": "read",
                    "permission": "workspace",
                    "working_directory": dirs.sub.to_string_lossy(),
                    "additional_readable_directories": [dirs.extra.to_string_lossy()],
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        let agent_id = agent_id_in(&output.text_content());

        let transcript = format!("{:?}", store.load_events(agent_id).await.expect("events"));
        assert!(transcript.contains("GRANTED-WORDS"), "{transcript}");
        assert!(
            !transcript.contains("BESIDE-WORDS")
                && transcript.contains("outside the readable directories"),
            "the scope alone confines the worker: {transcript}"
        );
        if let Some(platform) = &platform {
            let refused_platform = transcript
                .matches("outside the readable directories")
                .count()
                > 1;
            assert!(
                !refused_platform,
                "the system file {} stays readable: {transcript}",
                platform.display()
            );
        }
    }

    /// The tool names a worker's first request carried, sorted: what its registry held.
    /// The tools the worker's first request carried, sorted. `context_replace` is compaction's
    /// plumbing and rides on every catalog, so every expectation lists it.
    fn tools_sent(provider: &crate::provider::mock::MockProvider) -> Vec<String> {
        let mut names: Vec<String> = provider.streams()[0]
            .tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect();
        names.sort();
        names
    }

    /// `allow_tools` gives the worker exactly the tools it names, and a `deny_tools` entry among
    /// them still wins; both lists are recorded with the spawn terms.
    #[tokio::test]
    async fn an_allow_list_gives_the_worker_exactly_those_tools_less_any_denied() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let mut params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        // Streamed, since only a streamed request records the tools it carried.
        params.parent_options.streaming = true;
        let provider = mock(vec![text_round("done")]);
        let spawn = spawn_tool_for(params, Arc::clone(&provider) as Arc<dyn Provider>);
        let output = spawn
            .execute(
                serde_json::json!({
                    "prompt": "look",
                    "allow_tools": ["file_read", "file_write"],
                    "deny_tools": ["file_write"],
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        assert_eq!(tools_sent(&provider), vec!["context_replace", "file_read"]);
        let agent_id = agent_id_in(&output.text_content());
        let spec: SubagentSpec = serde_json::from_str(
            &store
                .load_subagent_spec(agent_id)
                .await
                .expect("load")
                .expect("spec"),
        )
        .expect("decode");
        assert_eq!(
            spec.allowed_tools,
            Some(vec!["file_read".to_string(), "file_write".to_string()])
        );
        assert_eq!(spec.denied_tools, vec!["file_write".to_string()]);
        assert_eq!(spec.allowed_servers, None);
    }

    /// An empty allow-list is refused before any row.
    #[tokio::test]
    async fn an_empty_allow_list_is_refused() {
        let dirs = workspaces();
        for key in ["allow_tools", "allow_servers"] {
            let refusal = refused_spawn(
                Permission::Read,
                dirs.work.clone(),
                serde_json::json!({ "prompt": "x", key: [] }),
            )
            .await;
            assert!(refusal.contains("names no"), "{key}: {refusal}");
        }
    }

    /// A nested spawn's allow-list intersects with what its parent holds: a worker allowed two
    /// tools can hand a child one of them, never a third.
    #[tokio::test]
    async fn a_nested_spawn_cannot_widen_its_parents_allow_list() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let mut params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        // Streamed, since only a streamed request records the tools it carried.
        params.parent_options.streaming = true;
        let provider = mock(vec![text_round("done")]);
        let mut spawn = spawn_tool_for(params, Arc::clone(&provider) as Arc<dyn Provider>);
        spawn.inherited_scope = ToolScope::default().allowing(
            None::<Vec<String>>,
            Some(vec!["file_read".to_string(), "file_find".to_string()]),
        );
        let output = spawn
            .execute(
                serde_json::json!({
                    "prompt": "look",
                    "allow_tools": ["file_read", "file_write"],
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        assert_eq!(tools_sent(&provider), vec!["context_replace", "file_read"]);
        let agent_id = agent_id_in(&output.text_content());
        let spec: SubagentSpec = serde_json::from_str(
            &store
                .load_subagent_spec(agent_id)
                .await
                .expect("load")
                .expect("spec"),
        )
        .expect("decode");
        assert_eq!(spec.allowed_tools, Some(vec!["file_read".to_string()]));
    }

    /// A follow-up rebuilds the worker under its recorded allow-list.
    #[tokio::test]
    async fn a_followup_keeps_a_workers_allow_lists() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let mut params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        // Streamed, since only a streamed request records the tools it carried.
        params.parent_options.streaming = true;
        let spawn = spawn_tool_for(params.clone(), mock(vec![text_round("spawned")]));
        let output = spawn
            .execute(
                serde_json::json!({ "prompt": "wait", "allow_tools": ["file_read"] }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        let agent_id = agent_id_in(&output.text_content());
        let provider = mock(vec![text_round("resumed")]);
        let followup = followup_tool_for(params, Arc::clone(&provider) as Arc<dyn Provider>);
        followup
            .execute(
                serde_json::json!({ "id": agent_id.to_string(), "prompt": "carry on" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("follow-up succeeds");
        assert_eq!(tools_sent(&provider), vec!["context_replace", "file_read"]);
    }

    /// A bare spawn is the parent's whole scope: its working directory, its additional workspace
    /// and readable directories and its level, so a parent in a multi-directory session spawns a
    /// worker like itself with nothing named. The spec records no scope, so a follow-up keeps
    /// following the parent.
    #[tokio::test]
    async fn a_bare_spawn_has_the_parents_whole_scope() {
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
        confine_reads(&mut params);
        // A readable directory beside the parent's workspace, since one inside it is the
        // workspace's own and would be named nowhere.
        let readable = dirs.work.parent().expect("base").join("readable");
        std::fs::create_dir_all(&readable).expect("mkdir");
        params.cells.roots = SharedRoots::new(vec![dirs.elsewhere.clone()]);
        params.cells.read_only_roots = SharedRoots::new(vec![readable.clone()]);
        let reference = readable.join("reference.txt");
        std::fs::write(&reference, "REFERENCE-WORDS").expect("write");
        let shared = dirs.elsewhere.join("shared.txt");
        let spawn = spawn_tool_for(
            params,
            mock(vec![
                read_file_round("call-1", &reference),
                write_file_round("call-2", &shared, "shared"),
                text_round("done"),
            ]),
        );
        let output = spawn
            .execute(
                serde_json::json!({ "prompt": "work like me" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        let agent_id = agent_id_in(&output.text_content());

        let transcript = format!("{:?}", store.load_events(agent_id).await.expect("events"));
        assert!(
            transcript.contains("REFERENCE-WORDS"),
            "the parent's readable directory is the worker's: {transcript}"
        );
        assert_eq!(
            std::fs::read_to_string(&shared)
                .expect("the parent's workspace directory is the worker's"),
            "shared"
        );
        let spec: SubagentSpec = serde_json::from_str(
            &store
                .load_subagent_spec(agent_id)
                .await
                .expect("load")
                .expect("spec"),
        )
        .expect("decode");
        assert_eq!(spec.stated_scope().expect("readable"), None);
        let text = output.text_content();
        assert!(
            text.contains(&format!("Working directory: {}\n", dirs.work.display()))
                && text.contains(&format!(
                    "Additional workspace directories:\n  {}\n",
                    dirs.elsewhere.display()
                ))
                && text.contains(&format!(
                    "Additional readable directories:\n  {}\n",
                    readable.display()
                )),
            "the result states the scope the worker inherited: {text}"
        );
    }

    /// A readable directory inside the workspace adds nothing and is merged rather than refused:
    /// inside the working directory, and inside an additional workspace directory alike.
    #[tokio::test]
    async fn a_readable_directory_inside_the_workspace_is_merged() {
        let dirs = workspaces();
        let inner = dirs.sub.join("inner");
        std::fs::create_dir_all(&inner).expect("mkdir");
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
        for input in [
            serde_json::json!({
                "prompt": "look",
                "working_directory": ".",
                "additional_readable_directories": [dirs.sub.to_string_lossy()],
            }),
            serde_json::json!({
                "prompt": "look",
                "working_directory": dirs.extra.to_string_lossy(),
                "additional_workspace_directories": [dirs.sub.to_string_lossy()],
                "additional_readable_directories": [inner.to_string_lossy()],
            }),
        ] {
            let spawn = spawn_tool_for(params.clone(), mock(vec![text_round("done")]));
            let output = spawn
                .execute(
                    input,
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect("merged, never refused");
            let agent_id = agent_id_in(&output.text_content());
            let spec: SubagentSpec = serde_json::from_str(
                &store
                    .load_subagent_spec(agent_id)
                    .await
                    .expect("load")
                    .expect("spec"),
            )
            .expect("decode");
            assert!(spec.additional_readable_directories.is_empty(), "{spec:?}");
            assert!(
                output
                    .text_content()
                    .contains("Additional readable directories: none\n"),
                "{}",
                output.text_content()
            );
        }
    }

    /// A bare worker's recorded directory is judged against the parent's readable reach on a
    /// follow-up, as a shaped worker's scope is against its write reach: a parent at `read`
    /// whose reads are confined and that has moved elsewhere cannot resume a reader it could not
    /// spawn today.
    #[tokio::test]
    async fn a_followup_refuses_a_workers_directory_the_parent_can_no_longer_read() {
        let dirs = workspaces();
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let mut params = params_at(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Permission::Read,
            dirs.work.clone(),
        );
        confine_reads(&mut params);
        let spawn = spawn_tool_for(params.clone(), mock(vec![text_round("spawned")]));
        let output = spawn
            .execute(
                serde_json::json!({ "prompt": "wait" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        let agent_id = agent_id_in(&output.text_content());

        // The parent `/cd`s to a directory that does not contain the worker's.
        params.cells.cwd.set(dirs.elsewhere.clone());
        let followup = followup_tool_for(params, mock(vec![text_round("never runs")]));
        let refusal = followup
            .execute(
                serde_json::json!({ "id": agent_id.to_string(), "prompt": "carry on" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect_err("a parent that can no longer read the worker's directory cannot resume it")
            .to_string();
        assert!(
            refusal.contains("can no longer grant")
                && refusal.contains("outside what this session reads"),
            "{refusal}"
        );
    }

    /// The schema names the three directory fields and nothing of the old pair, and each list
    /// says it needs the working directory, which is the one rule where a field affects another.
    #[test]
    fn the_schema_names_the_three_directory_fields() {
        let properties = &agent_spawn_definition(&[]).parameters["properties"];
        for old in ["writable_roots", "readable_roots"] {
            assert!(properties.get(old).is_none(), "{old} must be gone");
        }
        let description = |field: &str| {
            properties[field]["description"]
                .as_str()
                .unwrap_or_else(|| panic!("{field} has a description"))
                .to_string()
        };
        assert!(description("working_directory").contains("Required beside either list"));
        for list in [
            "additional_workspace_directories",
            "additional_readable_directories",
        ] {
            assert!(
                description(list).contains("Needs `working_directory`"),
                "{list}"
            );
            assert!(
                description(list).contains("Relative to your working directory"),
                "{list}"
            );
        }
    }

    /// A follow-up replays the recorded scope: the worker still reads the granted directory and
    /// is still refused beside it.
    #[tokio::test]
    async fn a_followup_keeps_a_workers_readable_directories() {
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
        confine_reads(&mut params);
        let granted = dirs.extra.join("granted.txt");
        let beside = dirs.work.join("beside.txt");
        std::fs::write(&granted, "GRANTED-WORDS").expect("write");
        std::fs::write(&beside, "BESIDE-WORDS").expect("write");
        let spawn = spawn_tool_for(params.clone(), mock(vec![text_round("spawned")]));
        let output = spawn
            .execute(
                serde_json::json!({
                    "prompt": "wait",
                    "working_directory": dirs.sub.to_string_lossy(),
                    "additional_readable_directories": [dirs.extra.to_string_lossy()],
                }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        let agent_id = agent_id_in(&output.text_content());

        let followup = followup_tool_for(
            params,
            mock(vec![
                read_file_round("call-1", &granted),
                read_file_round("call-2", &beside),
                text_round("read both"),
            ]),
        );
        followup
            .execute(
                serde_json::json!({ "id": agent_id.to_string(), "prompt": "read both" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("follow-up succeeds");
        let transcript = format!("{:?}", store.load_events(agent_id).await.expect("events"));
        assert!(transcript.contains("GRANTED-WORDS"), "{transcript}");
        assert!(
            !transcript.contains("BESIDE-WORDS")
                && transcript.contains("outside the readable directories"),
            "{transcript}"
        );
    }

    /// `network: "none"` is recorded with the spawn terms, withholds `web_fetch` from the worker
    /// and is stated in the result; a bare spawn under a parent at `none` holds `none` too.
    #[tokio::test]
    async fn a_worker_at_network_none_loses_web_fetch_and_the_result_says_so() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let mut params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        params.parent_options.streaming = true;
        let provider = mock(vec![text_round("done")]);
        let spawn = spawn_tool_for(params.clone(), Arc::clone(&provider) as Arc<dyn Provider>);
        let output = spawn
            .execute(
                serde_json::json!({ "prompt": "look", "network": "none" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        let sent = tools_sent(&provider);
        assert!(
            !sent.iter().any(|name| name == "web_fetch")
                && sent.iter().any(|name| name == "file_read"),
            "{sent:?}"
        );
        assert!(
            output
                .text_content()
                .contains("Network: none (shell and web_fetch)\n"),
            "{}",
            output.text_content()
        );
        let agent_id = agent_id_in(&output.text_content());
        let spec: SubagentSpec = serde_json::from_str(
            &store
                .load_subagent_spec(agent_id)
                .await
                .expect("load")
                .expect("spec"),
        )
        .expect("decode");
        assert_eq!(spec.network, crate::config::Network::None);

        params.cells.network = crate::config::Network::None;
        let provider = mock(vec![text_round("done")]);
        let spawn = spawn_tool_for(params, Arc::clone(&provider) as Arc<dyn Provider>);
        let output = spawn
            .execute(
                serde_json::json!({ "prompt": "look" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a bare spawn inherits");
        assert!(!tools_sent(&provider).iter().any(|name| name == "web_fetch"));
        assert!(output.text_content().contains("Network: none"));
    }

    /// A worker never reaches wider than its parent: `host` under a parent at `none` is refused;
    /// asking for `none` cannot hold at `unrestricted`, whose shell is not sandboxed, unless the
    /// call also lowers the level; and a bare spawn under a parent at `none` holds `none` at
    /// `unrestricted` too, as the parent does, with the result saying what the cut holds there.
    #[tokio::test]
    async fn a_worker_cannot_widen_the_network_and_none_needs_a_sandboxed_level() {
        let dirs = workspaces();
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let mut params = params_at(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
            Permission::Unrestricted,
            dirs.work.clone(),
        );
        let refused = |params: ToolBuilderParams, input: serde_json::Value| async move {
            spawn_tool_for(params, mock(vec![text_round("never runs")]))
                .execute(
                    input,
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await
                .expect_err("refused")
                .to_string()
        };
        let refusal = refused(
            params.clone(),
            serde_json::json!({ "prompt": "look", "network": "none" }),
        )
        .await;
        assert!(
            refusal.contains("cannot hold at `unrestricted`"),
            "{refusal}"
        );
        let refusal = refused(
            params.clone(),
            serde_json::json!({ "prompt": "look", "network": "wifi" }),
        )
        .await;
        assert!(
            refusal.contains("'wifi' is not a network setting"),
            "{refusal}"
        );
        spawn_tool_for(params.clone(), mock(vec![text_round("done")]))
            .execute(
                serde_json::json!({ "prompt": "look", "network": "none", "permission": "workspace" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("`none` holds at a sandboxed level");

        params.cells.network = crate::config::Network::None;
        let refusal = refused(
            params.clone(),
            serde_json::json!({ "prompt": "look", "network": "host", "permission": "workspace" }),
        )
        .await;
        assert!(refusal.contains("above your own `none`"), "{refusal}");

        params.parent_options.streaming = true;
        let provider = mock(vec![text_round("done")]);
        let output = spawn_tool_for(params, Arc::clone(&provider) as Arc<dyn Provider>)
            .execute(
                serde_json::json!({ "prompt": "look" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("an inherited `none` is what the parent holds, at `unrestricted` too");
        assert!(!tools_sent(&provider).iter().any(|name| name == "web_fetch"));
        assert!(
            output
                .text_content()
                .contains("Network: none (web_fetch)\n"),
            "{}",
            output.text_content()
        );
    }

    /// A follow-up narrows the recorded network to the parent's setting now, as it narrows the
    /// level: a `none` the parent holds since reaches a worker spawned under `host`.
    #[tokio::test]
    async fn a_followup_narrows_the_network_to_the_parents_setting_now() {
        let store = store_for_test().await;
        let parent_sid = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("parent");
        let mut params = params_for_test(
            store.clone(),
            crate::session::SharedSessionId::new(Some(parent_sid)),
        );
        params.parent_options.streaming = true;
        let spawn = spawn_tool_for(params.clone(), mock(vec![text_round("spawned")]));
        let output = spawn
            .execute(
                serde_json::json!({ "prompt": "wait" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("spawn succeeds");
        assert!(!output.text_content().contains("Network:"));
        let agent_id = agent_id_in(&output.text_content());

        params.cells.network = crate::config::Network::None;
        let provider = mock(vec![text_round("again")]);
        let followup = followup_tool_for(params, Arc::clone(&provider) as Arc<dyn Provider>);
        let output = followup
            .execute(
                serde_json::json!({ "id": agent_id.to_string(), "prompt": "carry on" }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("the follow-up runs");
        assert!(
            output
                .text_content()
                .contains("Network: none (shell and web_fetch)\n"),
            "{}",
            output.text_content()
        );
        assert!(!tools_sent(&provider).iter().any(|name| name == "web_fetch"));
    }

    /// A worker's network is an ordering check on the ladder: anything at or below the parent's
    /// setting is admitted, anything above it is refused naming what would have been, and any cut
    /// beside `unrestricted` is refused since nothing sandboxes the shell there.
    #[test]
    fn a_worker_s_network_is_admitted_at_or_below_its_parent_s() {
        use crate::config::Network;
        let resolve = |requested: Option<&str>, parent, level| {
            resolve_subagent_network(requested, parent, level).map_err(|error| error.to_string())
        };
        // Admitted pairs spelled out, so the test does not borrow the ordering it checks.
        let admitted = |requested, parent| {
            matches!(
                (requested, parent),
                (_, Network::Host)
                    | (Network::Remote | Network::None, Network::Remote)
                    | (Network::None, Network::None)
            )
        };
        for parent in Network::ALL {
            assert_eq!(resolve(None, parent, Permission::Unrestricted), Ok(parent));
            for requested in Network::ALL {
                let outcome = resolve(Some(requested.name()), parent, Permission::Workspace);
                if admitted(requested, parent) {
                    assert_eq!(outcome, Ok(requested), "{requested} under {parent}");
                } else {
                    let refusal = outcome.expect_err("wider than the parent");
                    assert!(
                        refusal.contains(&format!("is above your own `{parent}`"))
                            && refusal.contains(&format!("name `{parent}`")),
                        "{requested} under {parent}: {refusal}"
                    );
                }
            }
        }
        assert!(
            resolve(Some("host"), Network::Remote, Permission::Workspace)
                .expect_err("wider")
                .contains("name `remote` or `none`, or leave it out")
        );
        for requested in ["remote", "none"] {
            let refusal = resolve(Some(requested), Network::Host, Permission::Unrestricted)
                .expect_err("a cut beside `unrestricted`");
            assert!(
                refusal.contains("cannot hold at `unrestricted`"),
                "{refusal}"
            );
        }
        assert_eq!(
            resolve(Some("host"), Network::Host, Permission::Unrestricted),
            Ok(Network::Host)
        );
    }

    /// The schema offers `network` with its three spellings.
    #[test]
    fn the_schema_offers_the_network_setting() {
        let property = &agent_spawn_definition(&[]).parameters["properties"]["network"];
        assert_eq!(
            property["enum"],
            serde_json::json!(["host", "none", "remote"])
        );
        assert!(
            property["description"]
                .as_str()
                .expect("description")
                .contains("`allow_servers`")
        );
    }
}
