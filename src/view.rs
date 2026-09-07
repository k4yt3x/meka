//! The JSON shapes a `--format json` listing prints and the HTTP API serves, each defined once.
//!
//! A listing row and the object a `GET` answers with are one view of one record, and one client
//! type reads both, so the struct that names the fields sits below `host` and `cli`, and each host
//! converts into it here rather than spelling the fields again. What only one host can answer is
//! added by that host around the shared core: the HTTP `SessionResponse` flattens [`SessionView`]
//! under the facts a running server holds, and the `Installed*` and `Configured*` types here
//! flatten a core under what only a terminal should see, such as a path on this machine.
//!
//! Every `Option` field is omitted when it is `None`, never sent as `null`. The OpenAPI description
//! and the HTTP API guide state the rule once; every struct here keeps it.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use serde::Serialize;
use uuid::Uuid;

use crate::{
    config::{AccountConfig, McpServerConfig, ProfileConfig, ProfileSummary},
    mcp::AdvertisedTool,
    memory::Memory,
    permission::Permission,
    schedule::ScheduledJob,
    skills::{self, Skill},
    store::SessionSummary,
};

/// A session's row: what `meka session list` prints and what `GET /v1/sessions/{id}` answers with,
/// less the facts only the process holding the session can add (`last_turn_at`, `capabilities`,
/// `turn_in_flight`), which the HTTP response flattens this under.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
pub(crate) struct SessionView {
    pub(crate) id: Uuid,
    /// RFC 3339, when the row was made.
    pub(crate) created_at: String,
    /// RFC 3339, moved by any session-level change, a `PATCH` included.
    pub(crate) updated_at: String,
    /// Omitted when the row recorded none: an archive that carried no working directory is
    /// imported as it was written.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "serve", schema(value_type = Option<String>))]
    pub(crate) cwd: Option<PathBuf>,
    /// The session's permission level (`none`, `read`, `workspace`, `unrestricted`). Omitted when
    /// the row records no level: every door of this meka records one, so a bare row is an outside
    /// hand's, and inventing a value for it would state a level the session never ran at.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "serve", schema(value_type = Option<String>))]
    pub(crate) permission: Option<Permission>,
    /// Whether calls above the level are submitted for approval.
    pub(crate) approvals: bool,
    /// The profile this session runs on.
    pub(crate) profile: String,
    /// The first user message's words, whitespace collapsed and cut to 80 characters. Empty until
    /// the session has run a turn.
    pub(crate) title: String,
    /// The session this one was spawned from, for a sub-agent; omitted for a root session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) parent_id: Option<Uuid>,
}

impl From<&SessionSummary> for SessionView {
    fn from(session: &SessionSummary) -> Self {
        Self {
            id: session.id,
            created_at: session.created_at.clone(),
            updated_at: session.updated_at.clone(),
            cwd: session.cwd.clone(),
            permission: session.permission,
            approvals: session.approvals,
            profile: session.profile.clone(),
            title: session.title.clone(),
            parent_id: session.parent_id,
        }
    }
}

/// A profile as `meka profile list` prints it and `GET /v1/profiles` answers with: its account, the
/// account's backend and its model, and never a credential.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
pub(crate) struct ProfileView {
    pub(crate) name: String,
    /// The account this profile bills.
    pub(crate) account: String,
    /// The account's backend, e.g. `anthropic-messages`, `openai-responses` or
    /// `chatgpt-subscription`. Omitted when the profile names an account that is not configured,
    /// which a listing reports beside the table and which no server starts on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) model: Option<String>,
    /// Whether this is the profile a session gets when it names none.
    ///
    /// Not "the profile the server is running": a server runs no profile of its own, and each
    /// session runs on the one its row records. This marks only the default a run applies to a
    /// request that names no profile.
    pub(crate) active: bool,
}

impl ProfileView {
    /// The view of a profile a resolved config already summarized, as `meka serve` holds it.
    pub(crate) fn from_summary(summary: &ProfileSummary, active: bool) -> Self {
        Self {
            name: summary.name.clone(),
            account: summary.account.clone(),
            backend: Some(summary.backend.clone()),
            model: summary.model.clone(),
            active,
        }
    }

    /// The view of a profile as `config.toml` states it, with the backend looked up on the account
    /// it names, so a profile on a missing account is listed rather than dropped.
    pub(crate) fn from_config(
        name: &str,
        profile: &ProfileConfig,
        accounts: &BTreeMap<String, AccountConfig>,
        active: bool,
    ) -> Self {
        Self {
            name: name.to_string(),
            account: profile.account.clone(),
            backend: accounts
                .get(&profile.account)
                .map(|account| account.backend.clone()),
            model: profile.model.clone(),
            active,
        }
    }
}

/// An account as `meka account list` prints it: the `[accounts.<name>]` settings and whether a
/// credential is stored, never the credential. Accounts do not transit the HTTP API.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct AccountView {
    pub(crate) name: String,
    pub(crate) backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) base_url: Option<String>,
    /// `yes`, `no`, or `unreadable`: the same three answers as the plain column, because a row
    /// meka cannot deserialize is neither logged in nor logged out.
    pub(crate) authenticated: &'static str,
}

impl AccountView {
    /// The view of one configured account, with the credential question already answered by the
    /// caller, which is the only one holding the store.
    pub(crate) fn new(name: &str, account: &AccountConfig, authenticated: &'static str) -> Self {
        Self {
            name: name.to_string(),
            backend: account.backend.clone(),
            base_url: account.base_url.clone(),
            authenticated,
        }
    }
}

/// A configured MCP server as `meka mcp list` prints it: what `config.toml` says about it, never a
/// credential. `GET /v1/mcp` answers a different question, the live connection state, and has its
/// own shape in the HTTP layer; the two share `name` alone.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct McpServerView {
    pub(crate) name: String,
    pub(crate) transport: &'static str,
    pub(crate) required: bool,
    pub(crate) disabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) permission: Option<Permission>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) command: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) args: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) url: Option<String>,
}

impl From<&McpServerConfig> for McpServerView {
    fn from(config: &McpServerConfig) -> Self {
        Self {
            name: config.name.clone(),
            transport: config.transport.name(),
            // `required` is settled during config resolution, so `None` only shows up for a config
            // assembled outside that path; it means the same thing as false.
            required: config.required.unwrap_or(false),
            disabled: config.disabled.unwrap_or(false),
            permission: config.permission,
            command: config.command.clone(),
            args: config.args.clone(),
            url: config.url.clone(),
        }
    }
}

/// One configured MCP server in full, for `meka mcp get`: the listing's fields plus what only the
/// detail shows. Keys only for `env` and `headers`, as in the plain output, because a value there
/// may be a secret.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct McpServerDetail {
    #[serde(flatten)]
    pub(crate) server: McpServerView,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) env_keys: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) header_keys: Vec<String>,
    /// The kinds of credential stored for the server, by the labels the plain output uses.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) credentials: Vec<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) credential_origin: Option<String>,
    /// Absent when there is no origin, or no `url` for it to disagree with.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) credential_origin_matches_url: Option<bool>,
    /// The `type` value as written in `config.toml`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) auth: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) allowed_tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) disabled_tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_permissions: Option<BTreeMap<String, Permission>>,
}

/// One tool an MCP server advertises, with its permission resolved: what `meka mcp tools` prints
/// and `GET /v1/mcp/{name}/tools` answers with.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
pub(crate) struct McpToolView {
    /// Raw name as the server advertises it. This is the value to put in `allowed_tools`,
    /// `disabled_tools` or `tool_permissions`, which is why it is reported unmangled.
    pub(crate) raw_name: String,
    pub(crate) description: String,
    /// Output of the permission resolution chain.
    #[cfg_attr(feature = "serve", schema(value_type = String))]
    pub(crate) required_permission: Permission,
    /// Which step of that chain decided it, so a misclassified tool can be traced to the rule that
    /// classified it rather than guessed at.
    pub(crate) permission_source: &'static str,
    /// `false` when `allowed_tools` or `disabled_tools` filters this tool out, so the agent never
    /// sees it. Listed anyway: "configured away" and "not advertised" are different problems.
    pub(crate) allowed: bool,
    /// The server advertised `readOnlyHint: true` and `trust_read_only_hint = false` withheld it.
    /// `permission_source` names only the step that won, and a declined hint by definition did
    /// not, so without this a server advertising no hint and one whose hint was refused read
    /// alike.
    pub(crate) read_only_hint_declined: bool,
}

impl From<&AdvertisedTool> for McpToolView {
    fn from(tool: &AdvertisedTool) -> Self {
        Self {
            raw_name: tool.raw_name.clone(),
            description: tool.description.clone(),
            required_permission: tool.resolved_permission,
            permission_source: tool.permission_source.as_str(),
            allowed: tool.allowed,
            read_only_hint_declined: tool.read_only_hint_declined,
        }
    }
}

/// What one MCP server advertises, under the server's name.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
pub(crate) struct McpToolsResponse {
    pub(crate) server: String,
    pub(crate) tools: Vec<McpToolView>,
}

/// A scheduled job: what `meka schedule list` prints and `GET /v1/schedule` answers with.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
pub(crate) struct ScheduledJobView {
    pub(crate) id: String,
    pub(crate) session_id: Uuid,
    /// Human-readable rendering of the schedule, e.g. `every 30m`, `cron 0 9 * * 1-5`, or an RFC
    /// 3339 instant for a one-shot.
    pub(crate) schedule: String,
    pub(crate) prompt: String,
    /// Present when the job is gated, on a shell command or a tool call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) gate: Option<GateView>,
    pub(crate) created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_fired_at: Option<String>,
    pub(crate) next_fire_at: String,
    /// Why this job will not fire on its next occurrence, when something is holding it back.
    ///
    /// Absent means it will, as far as the reader can establish. A held job and a healthy watcher
    /// with nothing to report are otherwise identical from outside: neither fires, and
    /// `last_fired_at` is absent for a brand-new job too. Computed per request from the session's
    /// current level, not stored, so it tracks a `PATCH /v1/sessions/{id}` without the job being
    /// rewritten.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) withheld: Option<String>,
}

/// A job's gate: what it runs and when the result fires the job.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
pub(crate) struct GateView {
    /// What the gate runs: a shell command, or a tool name. Omitted when the caller may not see
    /// it.
    ///
    /// The HTTP API withholds it from a token that does not also hold `sessions:r`. A gate command
    /// is an `execute_command` line that runs unattended, the highest-entropy field in the system
    /// and the one most likely to carry a credential someone pasted into a `curl`, and `GET
    /// /v1/schedule` is server-wide, so a `schedule:r` token would otherwise read every gate on
    /// the box. A tool gate is withheld on the same terms, though it discloses less either
    /// way: this carries the bare tool name, and its arguments reach only `schedule_list`. The
    /// gate's kind and its condition stay visible, so a client can still tell a gated job from
    /// an ungated one, and a shell gate from a tool gate, without being told what it runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) check: Option<String>,
    /// `shell` or `tool`.
    pub(crate) kind: String,
    /// The fire condition, as `changed`, `succeeded`, `matches /…/` or `/pointer not-empty`.
    pub(crate) when: String,
}

impl ScheduledJobView {
    /// The view of a job for one reader, disclosing the gate's command only when `reveal_command`
    /// is set, and carrying the `withheld` reason that reader worked out.
    ///
    /// Not a `From` impl: the rendering depends on the caller's scopes, and a conversion that
    /// cannot see them is exactly how the command came to be disclosed at `schedule:r`.
    pub(crate) fn new(job: &ScheduledJob, reveal_command: bool, withheld: Option<String>) -> Self {
        Self {
            id: job.id.clone(),
            session_id: job.session_id,
            schedule: job.schedule.describe(),
            prompt: job.prompt.clone(),
            gate: job.gate.as_ref().map(|gate| GateView {
                check: reveal_command.then(|| gate.probe.summary()),
                kind: gate.probe.kind_str().to_string(),
                when: gate.predicate.summary(),
            }),
            created_at: job.created_at.to_rfc3339(),
            last_fired_at: job.last_fired_at.map(|at| at.to_rfc3339()),
            next_fire_at: job.next_fire_at.to_rfc3339(),
            withheld,
        }
    }
}

/// A memory: what `meka memory list` prints and `GET /v1/memory/{name}` answers with. The body
/// rides only on the single-memory surfaces, so a listing stays small.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
pub(crate) struct MemoryDetail {
    pub(crate) name: String,
    pub(crate) description: String,
    /// 0..=9, lower first. Unlike a skill's, a memory's priority is shown to the model, because it
    /// says how heavily to weigh a note the model is already reasoning from.
    pub(crate) priority: u8,
    /// RFC 3339, when the row was last written, which a metadata-only edit moves; see
    /// `recorded_at` for when the note was made.
    pub(crate) updated_at: String,
    /// RFC 3339, when the memory was recorded. Stamped once, at creation: the one the model is
    /// shown as an age, and the one ties are broken by.
    pub(crate) recorded_at: String,
    /// Lowercase labels for grouping and filtering.
    pub(crate) tags: Vec<String>,
    /// How many times the agent has recalled this memory through `memory_read`. Feeds search
    /// ranking; an operator reading the memory does not move it.
    pub(crate) read_count: u32,
    /// Present on `GET /v1/memory/{name}` and `meka memory show`, absent from a listing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) body: Option<String>,
}

impl MemoryDetail {
    /// The view of one memory, with the body the caller chose to load, or none for a listing.
    pub(crate) fn new(memory: &Memory, body: Option<String>) -> Self {
        Self {
            name: memory.name.clone(),
            description: memory.description.clone(),
            priority: memory.priority,
            updated_at: chrono::DateTime::<chrono::Utc>::from(memory.updated_at).to_rfc3339(),
            recorded_at: chrono::DateTime::<chrono::Utc>::from(memory.recorded_at).to_rfc3339(),
            tags: memory.tags.clone(),
            read_count: memory.read_count,
            body,
        }
    }
}

/// One entry of a tool catalog: what `GET /v1/sessions/{id}/tools` answers with, and the core of
/// what `meka tools list` prints.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
pub(crate) struct ToolView {
    pub(crate) name: String,
    pub(crate) description: String,
    /// Permission level this tool needs: `none`, `read`, `workspace` or `unrestricted`.
    #[cfg_attr(feature = "serve", schema(value_type = String))]
    pub(crate) required_permission: Permission,
    /// Whether the tool is deferred: present in the catalog by name but with its schema withheld
    /// until the model calls `load_tool`.
    pub(crate) deferred: bool,
}

impl ToolView {
    /// The view of one catalog entry, in the order `ToolRegistry::tool_catalog` yields them.
    pub(crate) fn new(
        name: String,
        description: String,
        required_permission: Permission,
        deferred: bool,
    ) -> Self {
        Self {
            name,
            description,
            required_permission,
            deferred,
        }
    }
}

/// A built-in tool as `meka tools list` prints it: the catalog entry plus the two facts this
/// listing exists to show and a session's catalog cannot, where the required level came from and
/// whether the config admits the tool at all.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ConfiguredToolView {
    #[serde(flatten)]
    pub(crate) tool: ToolView,
    /// `builtin`, or `override` when `[tools.tool_permissions]` names the tool.
    pub(crate) permission_source: &'static str,
    /// Whether a session on this config registers the tool; `false` is the plain `disabled`.
    pub(crate) enabled: bool,
}

/// A skill in the palette: what `GET /v1/skills` answers with, and the core of every other skill
/// view.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
pub(crate) struct SkillView {
    pub(crate) name: String,
    /// Rendered as an index for a client to draw, not the file's bytes: the body is the surface
    /// that is returned verbatim.
    pub(crate) description: String,
    /// Listing rank, 0..=9, lower first. Orders the `[Skills]` index the model sees and decides
    /// which entries that index's cap drops.
    pub(crate) priority: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) author: Option<String>,
    /// What the skill needs from its environment, from the Agent Skills `compatibility` field: the
    /// one optional spec field that changes how the skill's instructions should be carried out, so
    /// a client rendering a palette has a reason to show it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) compatibility: Option<String>,
}

impl From<&Skill> for SkillView {
    fn from(skill: &Skill) -> Self {
        Self {
            name: skill.name.clone(),
            description: crate::memory::render_description_for_model(&skill.description),
            priority: skill.priority,
            version: skill.version(),
            author: skill.author(),
            compatibility: skill.compatibility.clone(),
        }
    }
}

/// A skill as `meka skill list` prints it: the palette entry plus where it is, which a palette
/// served over HTTP has no business disclosing.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct InstalledSkillView {
    #[serde(flatten)]
    pub(crate) skill: SkillView,
    /// Found under `[skills] extra_paths` rather than in meka's own store.
    pub(crate) external: bool,
    pub(crate) source_dir: PathBuf,
}

impl InstalledSkillView {
    /// The view of one installed skill; `native_root` is meka's own store, the only directory
    /// anything writes to, so a skill from anywhere else is external.
    pub(crate) fn new(skill: &Skill, native_root: Option<&Path>) -> Self {
        Self {
            skill: SkillView::from(skill),
            external: native_root.is_none_or(|native| skill.root != native),
            source_dir: skill.source_dir.clone(),
        }
    }
}

/// One skill in full: what `GET /v1/skills/{name}` answers with, and the core of what `meka skill
/// get` and `show` print.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "serve", derive(utoipa::ToSchema))]
pub(crate) struct SkillDetail {
    #[serde(flatten)]
    pub(crate) skill: SkillView,
    /// The Agent Skills `license` field, verbatim. Informational.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) license: Option<String>,
    /// The Agent Skills `allowed-tools` field, verbatim. meka never acts on it; see the skills
    /// guide.
    #[serde(rename = "allowed-tools", skip_serializing_if = "Option::is_none")]
    pub(crate) allowed_tools: Option<String>,
    /// The `SKILL.md` body. Present on `GET /v1/skills/{name}` and `meka skill show`, absent
    /// wherever the palette should stay small.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) body: Option<String>,
}

impl SkillDetail {
    /// The view of one skill, with the body the caller chose to load, or none.
    pub(crate) fn new(skill: &Skill, body: Option<String>) -> Self {
        Self {
            skill: SkillView::from(skill),
            license: skill.license.clone(),
            allowed_tools: skill.allowed_tools.clone(),
            body,
        }
    }
}

/// One installed skill in full, for `meka skill get` and `show`: the detail plus its paths and the
/// unmodeled frontmatter the plain `get` prints. `metadata` is an object of strings when the
/// file's `metadata:` is a mapping and the value as text otherwise, the same split the plain lines
/// make.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct InstalledSkillDetail {
    #[serde(flatten)]
    pub(crate) detail: SkillDetail,
    pub(crate) source_dir: PathBuf,
    pub(crate) body_path: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) extra: BTreeMap<String, String>,
}

impl InstalledSkillDetail {
    /// The view of one installed skill, with the body the caller chose to load, or none.
    pub(crate) fn new(skill: &Skill, body: Option<String>) -> Self {
        let metadata = skill
            .metadata
            .as_ref()
            .map(|value| match skill.metadata_map() {
                Some(map) => serde_json::Value::Object(
                    map.iter()
                        .map(|(key, value)| {
                            (
                                skills::yaml_value_to_string(key),
                                serde_json::Value::String(skills::yaml_value_to_string(value)),
                            )
                        })
                        .collect(),
                ),
                None => serde_json::Value::String(skills::yaml_value_to_string(value)),
            });
        Self {
            detail: SkillDetail::new(skill, body),
            source_dir: skill.source_dir.clone(),
            body_path: skill.body_path.clone(),
            metadata,
            extra: skill
                .extra
                .iter()
                .map(|(key, value)| {
                    (
                        skills::yaml_value_to_string(key),
                        skills::yaml_value_to_string(value),
                    )
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(cwd: Option<PathBuf>, parent_id: Option<Uuid>) -> SessionSummary {
        SessionSummary {
            id: Uuid::nil(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-02T00:00:00Z".to_string(),
            title: "list the files".to_string(),
            cwd,
            permission: Some(Permission::Read),
            approvals: true,
            profile: "work".to_string(),
            capabilities_json: None,
            additional_roots: Vec::new(),
            token_id: None,
            parent_id,
        }
    }

    /// The row's view carries exactly the fields both hosts print, under the names the HTTP API
    /// documents, and nothing a store reader cannot answer.
    #[test]
    fn a_session_view_carries_the_row_under_the_documented_names() {
        let parent = Uuid::new_v4();
        let view = SessionView::from(&summary(Some(PathBuf::from("/work")), Some(parent)));
        let document = serde_json::to_value(&view).expect("serializes");
        let keys: Vec<&str> = document
            .as_object()
            .expect("an object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, [
            "id",
            "created_at",
            "updated_at",
            "cwd",
            "permission",
            "approvals",
            "profile",
            "title",
            "parent_id"
        ]);
        assert_eq!(document["cwd"], "/work");
        assert_eq!(document["permission"], "read");
        assert_eq!(document["approvals"], true);
        assert_eq!(document["profile"], "work");
        assert_eq!(document["parent_id"], parent.to_string());
    }

    /// An optional the row does not carry is left out of the document, never written as `null`.
    #[test]
    fn a_session_view_omits_what_the_row_does_not_record() {
        let mut view = SessionView::from(&summary(None, None));
        view.permission = None;
        let document = serde_json::to_value(&view).expect("serializes");
        let object = document.as_object().expect("an object");
        for absent in ["cwd", "permission", "parent_id"] {
            assert!(
                !object.contains_key(absent),
                "{absent} must be omitted rather than null: {document}"
            );
        }
    }

    /// A gate's command is disclosed only to a reader allowed to see it, and the gate's kind and
    /// condition survive either way.
    #[test]
    fn a_job_view_withholds_the_gate_command_unless_told_to_reveal_it() {
        let now = chrono::Utc::now();
        let job = ScheduledJob {
            id: "job-1".to_string(),
            session_id: Uuid::nil(),
            schedule: crate::schedule::Schedule::parse_every("1h").expect("a schedule"),
            prompt: "watch".to_string(),
            gate: Some(crate::schedule::Gate {
                probe: crate::schedule::GateProbe::Shell {
                    command: "curl -s https://example.test".to_string(),
                },
                predicate: crate::schedule::GatePredicate::Changed,
                last_output: None,
                permission: Permission::Unrestricted,
            }),
            created_at: now,
            last_fired_at: None,
            next_fire_at: now,
            attempts: 0,
        };
        let hidden = serde_json::to_value(ScheduledJobView::new(&job, false, None)).expect("json");
        assert!(hidden["gate"].get("check").is_none(), "{hidden}");
        assert_eq!(hidden["gate"]["kind"], "shell");
        assert!(hidden.get("last_fired_at").is_none(), "{hidden}");
        assert!(hidden.get("withheld").is_none(), "{hidden}");
        let shown = serde_json::to_value(ScheduledJobView::new(&job, true, None)).expect("json");
        assert_eq!(shown["gate"]["check"], "curl -s https://example.test");
    }
}
