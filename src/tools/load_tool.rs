//! The two registry meta-tools. `tool_load` makes a deferred tool's full schema visible to the
//! model on subsequent turns: the active tool set is derived by scanning the conversation for its
//! successful calls ([`crate::tools::load_tool::extract_loaded_tool_names_from_events`]), so its
//! `execute` only renders the description and schema as `tool_result` text and never mutates the
//! registry. `tool_search` finds a tool by keyword across names and descriptions, deferred ones
//! included. Both say what a call would do at the session's live level, through the door dispatch
//! uses, so a model never loads a tool only to find the call refused.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock, Weak},
};

use async_trait::async_trait;

use super::{
    Admission, LOAD_TOOL_NAME, SEARCH_TOOL_NAME, TOOL_SEARCH_MAX_RESULTS, Tool, ToolOutput,
    admit_tool_call, util::require_str,
};
use crate::{
    conversation::{ContentBlock, Event, Message},
    error::Result,
    permission::{Permission, SharedPermission},
    provider::ToolDefinition,
};

type ToolSet = Weak<RwLock<Vec<Arc<dyn Tool>>>>;
type DeferredSet = Weak<RwLock<HashSet<String>>>;

/// Meta-tool that makes a deferred tool's schema visible for use. Held by the
/// [`super::ToolRegistry`] like any other tool, so the same `Arc` lifecycle applies. The `Weak`
/// handles avoid a self-referential cycle (registry → `Arc<dyn Tool>` → `Arc<RwLock<…>>` →
/// registry).
pub(super) struct LoadToolTool {
    pub(super) tools: ToolSet,
    pub(super) deferred: DeferredSet,
    /// Filled once the registry is attached to an MCP manager. Lets an unfindable name be
    /// explained by its server's state instead of reported as unknown; a server that never
    /// connected registers no tools, so `tool_load` is the first place the agent hears about it.
    pub(super) mcp_manager: Weak<std::sync::OnceLock<Weak<crate::mcp::McpClientManager>>>,
    /// The registry's `[tools.tool_permissions]` overrides, so a result states the level a call
    /// is judged against, the way `tool_catalog` computes it.
    pub(super) permission_overrides: Arc<HashMap<String, Permission>>,
    /// The session's live level and approvals switch, read at call time as dispatch reads them, so
    /// a result says what a call would do now rather than at registration.
    pub(super) permission: SharedPermission,
}

impl LoadToolTool {
    /// The message for a name that resolved to nothing: the edit-distance hint when a registered
    /// name is close, else the keyword matches for the name, so a model that typed a bare word gets
    /// the tools that word finds rather than a dead end.
    fn not_registered(&self, name: &str, tools: &Arc<RwLock<Vec<Arc<dyn Tool>>>>) -> String {
        let hint = self.near_miss_hint(name, tools);
        let mut message = format!(
            "Error: tool '{name}' is not registered.{hint} Check the names listed under `[Tool \
             discovery]` in the conversation context, or search with `{SEARCH_TOOL_NAME}`."
        );
        if hint.is_empty() {
            let registered: Vec<Arc<dyn Tool>> = crate::sync::read(tools).iter().cloned().collect();
            let deferred = deferred_names(&self.deferred);
            if let Some(matches) = render_search(
                &registered,
                &deferred,
                &self.permission_overrides,
                &self.permission,
                name,
            ) {
                message.push_str("\n\n");
                message.push_str(&matches);
            }
        }
        message
    }

    /// The state of the MCP server behind `name`, when the name is unfindable *because* its
    /// server isn't connected. `None` for every other reason, leaving the generic message.
    async fn unavailable_server_reason(&self, name: &str) -> Option<String> {
        let slot = self.mcp_manager.upgrade()?;
        let manager = slot.get()?.upgrade()?;
        manager.unavailable_tool_reason(name).await
    }

    /// `" Did you mean …?"` against the registry, for a name that resolved to nothing. Takes the
    /// already-upgraded handle so the caller's read lock discipline stays in one place.
    fn near_miss_hint(
        &self,
        name: &str,
        tools: &std::sync::Arc<RwLock<Vec<Arc<dyn Tool>>>>,
    ) -> String {
        let registered: Vec<String> = crate::sync::read(tools)
            .iter()
            .map(|tool| tool.definition().name)
            .collect();
        crate::tools::did_you_mean_hint(name, registered.iter().map(String::as_str))
    }
}

#[async_trait]
impl Tool for LoadToolTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: LOAD_TOOL_NAME.to_string(),
            description: "Load the full schema for one or more deferred tools listed \
                          under `[Tool discovery]` in the conversation context or found \
                          with `tool_search`. After a successful call, each tool's full \
                          schema becomes available on your next turn, and the result says \
                          whether your current permission level allows a call. \
                          Invoke the tools by name as usual. Pass exact tool names (e.g. \
                          `mcp__notion__fetch`), either one as a string or several as an \
                          array."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "description": format!(
                            "Exact name of the tool to load, or an array of up to {} names.",
                            crate::tools::MAX_LOAD_TOOL_BATCH,
                        ),
                    }
                },
                "required": ["name"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        let names = crate::tools::load_tool_names(&input);
        if names.is_empty() {
            // Re-run the scalar extraction purely to raise its error, so a missing or non-string
            // `name` reports the same way it always has.
            require_str(&input, "name", LOAD_TOOL_NAME)?;
        }

        let Some(tools) = self.tools.upgrade() else {
            return Ok(ToolOutput::text(
                "Error: tool registry is no longer available.".to_string(),
                true,
            ));
        };

        let mut sections: Vec<String> = Vec::new();
        let mut resolved = 0usize;
        for name in &names {
            let tool = {
                let guard = crate::sync::read(&tools);
                guard.iter().find(|t| t.definition().name == *name).cloned()
            };

            let Some(tool) = tool else {
                sections.push(match self.unavailable_server_reason(name).await {
                    Some(reason) => reason,
                    None => self.not_registered(name, &tools),
                });
                continue;
            };
            let definition = tool.definition();

            // A tool that is not deferred is already in the active set, so this is a no-op success:
            // the scanner records a name that was already there.
            let is_deferred = self
                .deferred
                .upgrade()
                .map(|d| crate::sync::read(&d).contains(name))
                .unwrap_or(false);

            resolved += 1;
            // Stated for an active tool as well: "call it directly" without the level would send
            // the model into a refusal `tool_search` would have named.
            let status = callability(
                name,
                required_level(name, &*tool, &self.permission_overrides),
                tool.runs_outside_confinement(),
                &self.permission,
            );
            if !is_deferred {
                sections.push(format!(
                    "Tool '{name}' is already available and {status}. Call it directly."
                ));
                continue;
            }

            let schema = serde_json::to_string_pretty(&definition.parameters)
                .unwrap_or_else(|_| definition.parameters.to_string());
            sections.push(format!(
                "# {}\n\nThis tool {}.\n\n{}\n\n## Schema\n\n```json\n{}\n```",
                name, status, definition.description, schema,
            ));
        }

        let plural = if names.len() == 1 {
            "schema is"
        } else {
            "schemas are"
        };
        // Say so when the cap bit. Loading 10 of 15 while reporting success would leave the model
        // believing it holds five schemas it has never seen, which is the exact failure this tool's
        // advisories exist to prevent.
        let dropped = crate::tools::requested_tool_names(&input)
            .len()
            .saturating_sub(names.len());
        let capped = if dropped > 0 {
            format!(
                " Only the first {} names were loaded; {} more were not. Call `tool_load` again \
                 for those.",
                crate::tools::MAX_LOAD_TOOL_BATCH,
                dropped,
            )
        } else {
            String::new()
        };
        // Only claimed when something loaded: appended unconditionally, the trailer would follow an
        // error with "The full schema is now available on your next turn".
        let body = if resolved == 0 {
            sections.join("\n\n---\n\n")
        } else {
            format!(
                "{}\n\nThe full {} now available on your next turn. Call the tools directly with \
                 the parameters above.{}",
                sections.join("\n\n---\n\n"),
                plural,
                capped,
            )
        };
        // Errors are reported per name, but the call only *fails* when nothing resolved: a batch
        // that loaded three of four tools must stay non-error so the three are recorded as active.
        Ok(ToolOutput::text(body, resolved == 0))
    }
}

/// Meta-tool that finds a tool by keyword. Held by the registry like [`LoadToolTool`], with the
/// same `Weak` handles for the same reason.
pub(super) struct ToolSearchTool {
    pub(super) tools: ToolSet,
    pub(super) deferred: DeferredSet,
    /// See [`LoadToolTool::permission_overrides`].
    pub(super) permission_overrides: Arc<HashMap<String, Permission>>,
    /// See [`LoadToolTool::permission`].
    pub(super) permission: SharedPermission,
}

#[async_trait]
impl Tool for ToolSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: SEARCH_TOOL_NAME.to_string(),
            description: format!(
                "Search every registered tool by keyword, deferred tools included: the words in \
                 `query` are matched against tool names and descriptions, near misses tolerated, \
                 best matches first, up to {TOOL_SEARCH_MAX_RESULTS} per call. Each result says \
                 whether your current permission level allows a call and whether the tool is \
                 deferred, in which case `tool_load` fetches its schema or you can call it \
                 directly."
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Words to look for in tool names and descriptions.",
                    }
                },
                "required": ["query"]
            }),
            ..Default::default()
        }
    }

    fn required_permission(&self) -> Permission {
        Permission::Read
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _context: crate::tools::ToolContext,
    ) -> Result<ToolOutput> {
        // `require_str` refuses a missing or blank query, so the matcher never sees an empty one.
        let query = require_str(&input, "query", SEARCH_TOOL_NAME)?
            .trim()
            .to_string();
        let Some(tools) = self.tools.upgrade() else {
            return Ok(ToolOutput::text(
                "Error: tool registry is no longer available.".to_string(),
                true,
            ));
        };
        let registered: Vec<Arc<dyn Tool>> = crate::sync::read(&tools).iter().cloned().collect();
        let deferred = deferred_names(&self.deferred);
        Ok(
            match render_search(
                &registered,
                &deferred,
                &self.permission_overrides,
                &self.permission,
                &query,
            ) {
                Some(text) => ToolOutput::text(text, false),
                None => ToolOutput::text(format!("No tool matches '{query}'."), true),
            },
        )
    }
}

/// The names a registry currently defers, or none once the registry is gone.
fn deferred_names(deferred: &DeferredSet) -> HashSet<String> {
    deferred
        .upgrade()
        .map(|set| crate::sync::read(&set).clone())
        .unwrap_or_default()
}

/// The level a call of `name` is judged against: the config's override when it names one, else
/// the tool's own, exactly as the registry's catalog computes it.
fn required_level(
    name: &str,
    tool: &dyn Tool,
    overrides: &HashMap<String, Permission>,
) -> Permission {
    overrides
        .get(name)
        .copied()
        .unwrap_or_else(|| tool.required_permission())
}

/// What the permission door says about a call of `name` now, in one clause: the decision is
/// `admit_tool_call`'s, the same door dispatch and the checkpoint use. Only the wording
/// distinguishes the door's two reasons, by the same `allows` it consulted: a level the call is
/// above, or a boundary `workspace` cannot apply to a tool that runs outside meka's confinement.
///
/// "Allowed" and "would need", not "runs" and "is submitted": a tool may still refuse on its own
/// terms once called (a shell with no sandbox, a path outside the roots), and those doors take the
/// arguments, which a search does not have.
fn callability(
    name: &str,
    required: Permission,
    unconfinable: bool,
    permission: &SharedPermission,
) -> String {
    let level = permission.get();
    let approvals = permission.approvals();
    let reason = if level.allows(required) {
        format!("cannot be confined at `{level}`")
    } else {
        format!("is above your level (requires `{required}`, you are at `{level}`)")
    };
    match admit_tool_call(name, required, level, approvals, unconfinable) {
        Admission::Run => "is allowed at your level".to_string(),
        Admission::Ask => format!("{reason}; a call would need the user's approval"),
        Admission::Refuse(_) => format!("{reason}; a call is refused"),
    }
}

/// Points a query that is a tool's whole name, or its final `__` segment, earns on top of its
/// terms: past every keyword score a query of that many terms can reach, so the tool asked for by
/// name is first however many longer names share its words, and never cut by the result limit.
const EXACT_NAME_SCORE: usize = 1_000;
const EXACT_SEGMENT_SCORE: usize = 500;
/// Points a query term earns against a name or a description, by how it matched: a whole word, a
/// substring, or a word within [`crate::tools::fuzzy_threshold`] edits. A name outranks a
/// description at every rung because a name is what the model will type.
const NAME_WORD_SCORE: usize = 4;
const NAME_SUBSTRING_SCORE: usize = 3;
const NAME_NEAR_MISS_SCORE: usize = 2;
const DESCRIPTION_WORD_SCORE: usize = 2;
const DESCRIPTION_SUBSTRING_SCORE: usize = 1;
const DESCRIPTION_NEAR_MISS_SCORE: usize = 1;

/// One tool a query matched, with the score that ranks it.
struct ToolMatch {
    tool: Arc<dyn Tool>,
    definition: ToolDefinition,
    score: usize,
}

/// Every registered tool `query` matches, best first: by score descending, then the shorter name,
/// then by name.
///
/// The query is lowercased and split into terms on anything that is not a letter or a digit; a
/// tool's name and its full description (not the clipped summary) are split into words the same
/// way. A term scores by its best match against each of the two, name over description and whole
/// word over substring over near miss, and the two are not summed: the better of them is the
/// term's score. Scores sum over terms, a query that is a tool's whole name or final segment adds
/// [`EXACT_NAME_SCORE`] or [`EXACT_SEGMENT_SCORE`], and a tool qualifies above zero. The shorter
/// name breaks a tie because `get_item` and `get_account_item` earn the same words from
/// `get item`, and the shorter one is the one those words name. Shared by `tool_search` and
/// `tool_load`'s miss path: one definition of what a match is.
fn match_tools(tools: &[Arc<dyn Tool>], query: &str) -> Vec<ToolMatch> {
    let lowered = query.trim().to_lowercase();
    let terms: Vec<&str> = words_of(&lowered).collect();
    if terms.is_empty() {
        return Vec::new();
    }
    let mut matches: Vec<ToolMatch> = tools
        .iter()
        .filter_map(|tool| {
            let definition = tool.definition();
            let name = definition.name.to_lowercase();
            let description = definition.description.to_lowercase();
            let mut score: usize = terms
                .iter()
                .map(|term| term_score(term, &name, &description))
                .sum();
            if name == lowered {
                score += EXACT_NAME_SCORE;
            } else if name.rsplit("__").next() == Some(lowered.as_str()) {
                score += EXACT_SEGMENT_SCORE;
            }
            (score > 0).then(|| ToolMatch {
                tool: Arc::clone(tool),
                definition,
                score,
            })
        })
        .collect();
    matches.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.definition.name.len().cmp(&right.definition.name.len()))
            .then_with(|| left.definition.name.cmp(&right.definition.name))
    });
    matches
}

/// The words of `text`: runs of letters and digits.
fn words_of(text: &str) -> impl Iterator<Item = &str> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
}

/// The better of a term's score against the name and against the description.
fn term_score(term: &str, name: &str, description: &str) -> usize {
    let against = |text: &str, whole: usize, substring: usize, near_miss: usize| -> usize {
        if words_of(text).any(|word| word == term) {
            return whole;
        }
        if text.contains(term) {
            return substring;
        }
        let threshold = crate::tools::fuzzy_threshold(term);
        // Counted in characters, as the threshold is, and banded before the matrix as every other
        // caller of `edit_distance` is: a 2 KB description is hundreds of words per tool.
        let term_chars = term.chars().count();
        if words_of(text).any(|word| {
            term_chars.abs_diff(word.chars().count()) <= threshold
                && crate::tools::edit_distance(term, word) <= threshold
        }) {
            return near_miss;
        }
        0
    };
    against(
        name,
        NAME_WORD_SCORE,
        NAME_SUBSTRING_SCORE,
        NAME_NEAR_MISS_SCORE,
    )
    .max(against(
        description,
        DESCRIPTION_WORD_SCORE,
        DESCRIPTION_SUBSTRING_SCORE,
        DESCRIPTION_NEAR_MISS_SCORE,
    ))
}

/// The `tool_search` result for `query` over `tools`, or `None` when nothing matched: the
/// `[Tool discovery]` line per match, then what a call would do now and whether the tool is
/// deferred, the top [`TOOL_SEARCH_MAX_RESULTS`] with the rest counted.
fn render_search(
    tools: &[Arc<dyn Tool>],
    deferred: &HashSet<String>,
    overrides: &HashMap<String, Permission>,
    permission: &SharedPermission,
    query: &str,
) -> Option<String> {
    let matches = match_tools(tools, query);
    if matches.is_empty() {
        return None;
    }
    let shown = matches.len().min(TOOL_SEARCH_MAX_RESULTS);
    let mut out = format!("Tools matching '{query}':\n");
    for found in &matches[..shown] {
        let name = &found.definition.name;
        let required = required_level(name, &*found.tool, overrides);
        let summary = crate::prompt::short_description(&found.definition.description);
        let status = callability(
            name,
            required,
            found.tool.runs_outside_confinement(),
            permission,
        );
        out.push_str(&format!("- **{name}** (requires `{required}`)"));
        if !summary.is_empty() {
            out.push_str(&format!(": {summary}"));
        }
        out.push_str(&format!("\n  {status}"));
        if deferred.contains(name) {
            out.push_str("; deferred: call `tool_load` for the schema, or call it directly");
        }
        out.push('\n');
    }
    let hidden = matches.len() - shown;
    if hidden > 0 {
        out.push_str(&format!("\n{hidden} more match; narrow the query.\n"));
    }
    Some(out)
}

/// Walk events and collect the names of tools loaded via successful `tool_load` calls. The only
/// door for this question: a scan of the materialized slice cannot see a load whose exchange a
/// compaction has summarized away or a repair has emptied, so this absorbs
/// [`Event::CompactBoundary::loaded_tools_snapshot`] at a boundary and clears the pending uses
/// inside the summarized window, whose results are below the view's logical start.
///
/// Returns names in load order, de-duplicated. The order is what makes the tools array a stable
/// cache prefix: `tool_load` calls only ever append to the conversation, so the array can only
/// grow at the end, whereas the registry's own order would reinsert an earlier-registered tool
/// ahead of one loaded first and re-cache the whole conversation behind it.
pub(crate) fn extract_loaded_tool_names_from_events(events: &[Event]) -> Vec<String> {
    use std::collections::HashMap;
    let mut loaded: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut pending: HashMap<String, Vec<String>> = HashMap::new();

    let absorb = |message: &Message,
                  pending: &mut HashMap<String, Vec<String>>,
                  seen: &mut HashSet<String>,
                  loaded: &mut Vec<String>| {
        for block in &message.content {
            match block {
                ContentBlock::ToolUse { id, name, input }
                    if name == crate::tools::LOAD_TOOL_NAME =>
                {
                    let names = crate::tools::load_tool_names(input);
                    if !names.is_empty() {
                        pending.insert(id.clone(), names);
                    }
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    is_error,
                    ..
                } => {
                    if let Some(loaded_names) = pending.remove(tool_use_id)
                        && !is_error
                    {
                        // A batch load appends its names in call order, keeping the tools array's
                        // growth append-only exactly as a sequence of single loads would.
                        for loaded_name in loaded_names {
                            if seen.insert(loaded_name.clone()) {
                                loaded.push(loaded_name);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    };

    for event in events {
        match event {
            Event::Append(message) => absorb(message, &mut pending, &mut seen, &mut loaded),
            // A repair never *un*-loads a tool: the array is a cache prefix that may only grow, and
            // rewinding past a `tool_load` would drop an entry from its middle and re-cache the
            // whole conversation behind it. Pending uses inside the replaced window are dropped the
            // same way a boundary drops them, since their results are gone from the view.
            Event::Repair { messages, .. } => {
                pending.clear();
                for message in messages {
                    absorb(message, &mut pending, &mut seen, &mut loaded);
                }
            }
            // Touches images only; no tool call is added or removed by it.
            Event::Redact { .. } => {}
            Event::CompactBoundary {
                loaded_tools_snapshot,
                ..
            } => {
                // Pending uses inside the summarized window are gone from the model's view; their
                // would-be results are also gone. Drop them and absorb the snapshot.
                pending.clear();
                // The snapshot is an unordered set, so sort it for a deterministic tail. Continuity
                // with the pre-boundary order isn't needed: compaction rewrites the head of the
                // conversation and re-caches everything anyway. What matters is that every turn
                // *after* the boundary agrees on the order.
                let mut absorbed: Vec<&String> = loaded_tools_snapshot.iter().collect();
                absorbed.sort();
                for name in absorbed {
                    if seen.insert(name.clone()) {
                        loaded.push(name.clone());
                    }
                }
            }
        }
    }

    loaded
}
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio_util::sync::CancellationToken;

    use super::*;

    /// Minimal fake tool for testing the registry-lookup paths of `LoadToolTool` without dragging
    /// in `ToolRegistry::build_default`.
    struct FakeTool {
        name: String,
        description: String,
        schema: serde_json::Value,
    }

    #[async_trait]
    impl Tool for FakeTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name.clone(),
                description: self.description.clone(),
                parameters: self.schema.clone(),
                ..Default::default()
            }
        }

        fn required_permission(&self) -> Permission {
            Permission::Read
        }

        async fn execute(
            &self,
            _input: serde_json::Value,
            _context: crate::tools::ToolContext,
        ) -> Result<crate::tools::ToolOutput> {
            Ok(crate::tools::ToolOutput::text(String::new(), false))
        }
    }

    type ToolStorage = Arc<RwLock<Vec<Arc<dyn Tool>>>>;
    type DeferredStorage = Arc<RwLock<HashSet<String>>>;

    /// Test fixture: holds the strong `Arc`s for `tools` and `deferred` so the `Weak`s inside
    /// `LoadToolTool` stay live for the duration of a test. `take()` either field to simulate
    /// registry teardown.
    struct Fixture {
        tools: Option<ToolStorage>,
        deferred: Option<DeferredStorage>,
        tool_load: LoadToolTool,
        tool_search: ToolSearchTool,
    }

    fn fake_tool(name: &str) -> Arc<dyn Tool> {
        Arc::new(FakeTool {
            name: name.to_string(),
            description: format!("Fixture tool {name}."),
            schema: serde_json::json!({
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Page URL"}},
                "required": ["url"]
            }),
        }) as Arc<dyn Tool>
    }

    fn build_test_tool(registered: Vec<Arc<dyn Tool>>, deferred_names: &[&str]) -> Fixture {
        build_test_tool_at(
            registered,
            deferred_names,
            crate::tools::tests::shared_permission_for_test(),
        )
    }

    /// The fixture under a session at `permission`, for the callability lines.
    fn build_test_tool_at(
        registered: Vec<Arc<dyn Tool>>,
        deferred_names: &[&str],
        permission: SharedPermission,
    ) -> Fixture {
        let tools: ToolStorage = Arc::new(RwLock::new(registered));
        let deferred: DeferredStorage = Arc::new(RwLock::new(
            deferred_names.iter().map(|n| n.to_string()).collect(),
        ));
        let overrides = Arc::new(HashMap::new());
        let tool_load = LoadToolTool {
            tools: Arc::downgrade(&tools),
            deferred: Arc::downgrade(&deferred),
            // No manager attached: these fixtures exercise the plain registry paths, so an
            // unfindable name must still produce the generic "not registered" message.
            mcp_manager: std::sync::Weak::new(),
            permission_overrides: Arc::clone(&overrides),
            permission: permission.clone(),
        };
        let tool_search = ToolSearchTool {
            tools: Arc::downgrade(&tools),
            deferred: Arc::downgrade(&deferred),
            permission_overrides: overrides,
            permission,
        };
        Fixture {
            tools: Some(tools),
            deferred: Some(deferred),
            tool_load,
            tool_search,
        }
    }

    fn described_tool(name: &str, description: &str) -> Arc<dyn Tool> {
        Arc::new(FakeTool {
            name: name.to_string(),
            description: description.to_string(),
            schema: serde_json::json!({"type": "object", "properties": {}}),
        }) as Arc<dyn Tool>
    }

    /// A tool that requires `required`, for the callability lines.
    struct RestrictedTool {
        name: String,
        required: Permission,
    }

    #[async_trait]
    impl Tool for RestrictedTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name.clone(),
                description: "Fixture requiring a level.".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
                ..Default::default()
            }
        }

        fn required_permission(&self) -> Permission {
            self.required
        }

        async fn execute(
            &self,
            _input: serde_json::Value,
            _context: crate::tools::ToolContext,
        ) -> Result<crate::tools::ToolOutput> {
            Ok(crate::tools::ToolOutput::text(String::new(), false))
        }
    }

    fn ida_tools() -> Vec<Arc<dyn Tool>> {
        vec![
            described_tool("mcp__ida__xrefs_to", "Cross-references to an address."),
            described_tool("mcp__ida__xrefs_from", "Cross-references from an address."),
            described_tool(
                "mcp__ida__decompile",
                "Decompile a function; the pseudocode lists xrefs too.",
            ),
        ]
    }

    async fn search(fixture: &Fixture, query: &str) -> ToolOutput {
        fixture
            .tool_search
            .execute(
                serde_json::json!({"query": query}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a search answers")
    }

    /// The line under `name`'s entry: what a call would do, and whether the tool is deferred.
    fn status_of(text: &str, name: &str) -> String {
        let start = text
            .find(&format!("- **{name}**"))
            .unwrap_or_else(|| panic!("{name} in {text}"));
        text[start..]
            .lines()
            .nth(1)
            .expect("a status line")
            .trim()
            .to_string()
    }

    fn read_level() -> SharedPermission {
        SharedPermission::new(Permission::Read, crate::permission::EnabledPermissions::ALL)
    }

    #[tokio::test]
    async fn a_query_ranks_name_matches_above_description_matches() {
        let fixture = build_test_tool(ida_tools(), &[]);
        let text = search(&fixture, "xrefs").await.text_content();
        let position = |name: &str| {
            text.find(name)
                .unwrap_or_else(|| panic!("{name} in {text}"))
        };
        // Both `xrefs_*` tools carry the word in their name; `decompile` only in its description.
        assert!(
            position("mcp__ida__xrefs_from") < position("mcp__ida__decompile"),
            "{text}"
        );
        assert!(
            position("mcp__ida__xrefs_to") < position("mcp__ida__decompile"),
            "{text}"
        );
    }

    #[tokio::test]
    async fn a_bare_segment_ranks_its_namespaced_tool_first() {
        let fixture = build_test_tool(ida_tools(), &[]);
        let text = search(&fixture, "xrefs_to").await.text_content();
        let first = text
            .lines()
            .find(|line| line.starts_with("- **"))
            .expect("a match");
        assert!(first.starts_with("- **mcp__ida__xrefs_to**"), "{text}");
    }

    /// Fourteen longer names share every word of `get_item`, and the alphabet puts most of them
    /// first; the tool asked for by name, or by the name its server gave it, leads regardless and
    /// is never cut by the result limit.
    #[tokio::test]
    async fn the_tool_asked_for_by_name_leads_the_results() {
        let qualifiers = [
            "account", "address", "basket", "catalog", "contact", "delivery", "event", "file",
            "invoice", "ledger", "note", "order", "payment", "receipt",
        ];
        let mut tools: Vec<Arc<dyn Tool>> = qualifiers
            .iter()
            .map(|qualifier| {
                described_tool(&format!("mcp__records__get_{qualifier}_item"), "Get one.")
            })
            .collect();
        tools.push(described_tool("mcp__records__get_item", "Get one."));
        // Shorter than the tool asked for and made of the same words, so a tiebreak on length
        // would put it first; only the exact final segment ranks the right one above it.
        tools.push(described_tool("mcp__a__item_get", "Get one."));
        let fixture = build_test_tool(tools, &[]);
        for query in ["get_item", "mcp__records__get_item", "  Get_Item "] {
            let text = search(&fixture, query).await.text_content();
            let first = text
                .lines()
                .find(|line| line.starts_with("- **"))
                .expect("a match");
            assert!(
                first.starts_with("- **mcp__records__get_item**"),
                "{query}: {text}"
            );
        }
    }

    #[tokio::test]
    async fn a_misspelled_term_still_finds_the_tool() {
        let fixture = build_test_tool(ida_tools(), &[]);
        let text = search(&fixture, "xerfs").await.text_content();
        assert!(text.contains("mcp__ida__xrefs_to"), "{text}");
    }

    #[tokio::test]
    async fn a_query_that_matches_nothing_says_so() {
        let fixture = build_test_tool(ida_tools(), &[]);
        let output = search(&fixture, "kubernetes").await;
        assert!(output.is_error);
        assert_eq!(output.text_content(), "No tool matches 'kubernetes'.");
    }

    /// A blank query and an absent one are the same mistake, refused the way every string
    /// parameter is refused.
    #[tokio::test]
    async fn an_empty_query_is_refused() {
        let fixture = build_test_tool(ida_tools(), &[]);
        for input in [serde_json::json!({"query": "   "}), serde_json::json!({})] {
            let outcome = fixture
                .tool_search
                .execute(
                    input,
                    crate::tools::ToolContext::detached(CancellationToken::new()),
                )
                .await;
            assert!(outcome.is_err());
        }
    }

    #[tokio::test]
    async fn results_are_capped_and_the_rest_are_counted() {
        let tools: Vec<Arc<dyn Tool>> = (0..14)
            .map(|index| described_tool(&format!("mcp__s__thing_{index:02}"), "A thing."))
            .collect();
        let fixture = build_test_tool(tools, &[]);
        let text = search(&fixture, "thing").await.text_content();
        let shown = text.lines().filter(|line| line.starts_with("- **")).count();
        assert_eq!(shown, TOOL_SEARCH_MAX_RESULTS, "{text}");
        assert!(text.contains("4 more match; narrow the query."), "{text}");
    }

    #[tokio::test]
    async fn a_result_says_whether_a_tool_is_callable_now_or_needs_loading() {
        let fixture = build_test_tool(ida_tools(), &["mcp__ida__xrefs_to"]);
        let text = search(&fixture, "xrefs").await.text_content();
        assert!(
            status_of(&text, "mcp__ida__xrefs_to").contains("deferred: call `tool_load`"),
            "{text}"
        );
        assert!(
            !status_of(&text, "mcp__ida__xrefs_from").contains("deferred"),
            "{text}"
        );
    }

    /// The status comes from the door dispatch uses, read against the live level and switch, so
    /// the same registry answers differently after the user toggles approvals.
    #[tokio::test]
    async fn a_result_says_whether_a_call_would_run_ask_or_be_refused() {
        let tools: Vec<Arc<dyn Tool>> = vec![
            described_tool("thing_read", "Read a thing."),
            Arc::new(RestrictedTool {
                name: "thing_write".to_string(),
                required: Permission::Unrestricted,
            }),
        ];
        let permission = read_level();
        let fixture = build_test_tool_at(tools, &[], permission.clone());
        let text = search(&fixture, "thing").await.text_content();
        assert_eq!(
            status_of(&text, "thing_read"),
            "is allowed at your level",
            "{text}"
        );
        assert_eq!(
            status_of(&text, "thing_write"),
            "is above your level (requires `unrestricted`, you are at `read`); a call is refused",
            "{text}"
        );
        permission.set_approvals(true);
        let text = search(&fixture, "thing").await.text_content();
        assert_eq!(
            status_of(&text, "thing_write"),
            "is above your level (requires `unrestricted`, you are at `read`); a call would need \
             the user's approval",
            "{text}"
        );
    }

    #[tokio::test]
    async fn tool_load_states_callability_under_each_schema() {
        let fixture = build_test_tool_at(
            vec![Arc::new(RestrictedTool {
                name: "mcp__x__wipe".to_string(),
                required: Permission::Unrestricted,
            })],
            &["mcp__x__wipe"],
            read_level(),
        );
        let output = fixture
            .tool_load
            .execute(
                serde_json::json!({"name": "mcp__x__wipe"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a load answers");
        assert!(!output.is_error);
        let text = output.text_content();
        assert!(
            text.contains(
                "This tool is above your level (requires `unrestricted`, you are at `read`); a \
                 call is refused."
            ),
            "{text}"
        );
        assert!(text.contains("## Schema"), "{text}");
    }

    /// A load of a tool that is already active answers "call it directly", and that advice has to
    /// carry the same level check a deferred load does, or a model at `read` is sent into a
    /// refusal `tool_search` would have named.
    #[tokio::test]
    async fn an_active_tool_s_load_still_states_its_callability() {
        let tools: Vec<Arc<dyn Tool>> = vec![
            described_tool("thing_read", "Read a thing."),
            Arc::new(RestrictedTool {
                name: "thing_write".to_string(),
                required: Permission::Unrestricted,
            }),
        ];
        let fixture = build_test_tool_at(tools, &[], read_level());
        let output = fixture
            .tool_load
            .execute(
                serde_json::json!({"name": ["thing_read", "thing_write"]}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a load answers");
        let text = output.text_content();
        assert!(
            text.contains("Tool 'thing_read' is already available and is allowed at your level."),
            "{text}"
        );
        assert!(
            text.contains(
                "Tool 'thing_write' is already available and is above your level (requires \
                 `unrestricted`, you are at `read`); a call is refused."
            ),
            "{text}"
        );
    }

    /// A bare word is not a near miss of any namespaced name, so the edit-distance hint has
    /// nothing; the keyword matches are what turn the miss into a next step.
    #[tokio::test]
    async fn tool_load_s_miss_offers_keyword_matches() {
        let fixture = build_test_tool(ida_tools(), &["mcp__ida__xrefs_to"]);
        let output = fixture
            .tool_load
            .execute(
                serde_json::json!({"name": "xrefs"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("a load answers");
        assert!(output.is_error);
        let text = output.text_content();
        assert!(text.contains("not registered"), "{text}");
        assert!(text.contains("Tools matching 'xrefs':"), "{text}");
        assert!(text.contains("mcp__ida__xrefs_to"), "{text}");
    }

    #[tokio::test]
    async fn load_tool_unknown_name() {
        let fixture = build_test_tool(Vec::new(), &[]);
        let tool_load = &fixture.tool_load;
        let result = tool_load
            .execute(
                serde_json::json!({"name": "nonexistent"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("should return Ok");
        assert!(result.is_error);
        let text = result.text_content();
        assert!(text.contains("not registered"));
        assert!(text.contains("[Tool discovery]"));
        // And it must not also claim the schema arrived.
        assert!(
            !text.contains("next turn"),
            "a load that resolved nothing must not promise a schema: {text}"
        );
    }

    #[tokio::test]
    async fn load_tool_missing_name_field() {
        let fixture = build_test_tool(Vec::new(), &[]);
        let tool_load = &fixture.tool_load;
        let result = tool_load
            .execute(
                serde_json::json!({}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn load_tool_returns_schema_for_deferred_tool() {
        let fake = Arc::new(FakeTool {
            name: "mcp__notion__fetch".to_string(),
            description: "Fetch a Notion page by URL or ID.".to_string(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Page URL"}
                },
                "required": ["url"]
            }),
        }) as Arc<dyn Tool>;
        let fixture = build_test_tool(vec![fake], &["mcp__notion__fetch"]);
        let tool_load = &fixture.tool_load;

        let result = tool_load
            .execute(
                serde_json::json!({"name": "mcp__notion__fetch"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("should return Ok");

        assert!(!result.is_error, "deferred-tool load should succeed");
        let text = result.text_content();
        assert!(text.contains("mcp__notion__fetch"));
        assert!(text.contains("Fetch a Notion page"));
        assert!(text.contains("## Schema"));
        // The schema body must be the actual tool's schema, not a placeholder.
        assert!(text.contains("\"url\""));
        assert!(text.contains("\"required\""));
        assert!(text.contains("next turn"));
    }

    #[tokio::test]
    async fn load_tool_already_available_tool() {
        // Registered but not in the deferred set: a success, so the scanner records a name that
        // was already in the active set.
        let fake = Arc::new(FakeTool {
            name: "file_read".to_string(),
            description: "Read a file from disk.".to_string(),
            schema: serde_json::json!({"type": "object"}),
        }) as Arc<dyn Tool>;
        let fixture = build_test_tool(vec![fake], &[]);
        let tool_load = &fixture.tool_load;

        let result = tool_load
            .execute(
                serde_json::json!({"name": "file_read"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("should return Ok");

        assert!(!result.is_error);
        let text = result.text_content();
        assert!(text.contains("already available"));
        assert!(text.contains("file_read"));
        // Must NOT render the schema block; the model already has it.
        assert!(!text.contains("## Schema"));
    }

    #[tokio::test]
    async fn load_tool_accepts_an_array_of_names() {
        let fixture = build_test_tool(
            vec![
                fake_tool("mcp__notion__fetch"),
                fake_tool("mcp__notion__search"),
            ],
            &["mcp__notion__fetch", "mcp__notion__search"],
        );

        let result = fixture
            .tool_load
            .execute(
                serde_json::json!({"name": ["mcp__notion__fetch", "mcp__notion__search"]}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("should return Ok");

        assert!(!result.is_error);
        let text = result.text_content();
        assert!(text.contains("# mcp__notion__fetch"), "{text}");
        assert!(text.contains("# mcp__notion__search"), "{text}");
        assert!(text.contains("schemas are"), "plural wording: {text}");
    }

    /// A batch must not lose the tools that did resolve just because one name was wrong: the
    /// non-error result is what records them in the active set.
    #[tokio::test]
    async fn load_tool_batch_survives_one_bad_name() {
        let fixture = build_test_tool(vec![fake_tool("mcp__notion__fetch")], &[
            "mcp__notion__fetch",
        ]);

        let result = fixture
            .tool_load
            .execute(
                serde_json::json!({"name": ["mcp__notion__fetch", "nope"]}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("should return Ok");

        assert!(!result.is_error, "one resolved, so the call succeeded");
        let text = result.text_content();
        assert!(text.contains("# mcp__notion__fetch"), "{text}");
        assert!(text.contains("'nope' is not registered"), "{text}");
    }

    /// Half-honoring an over-long batch while reporting plain success would leave the model
    /// believing it holds schemas it has never seen.
    #[tokio::test]
    async fn load_tool_reports_names_dropped_by_the_cap() {
        let names: Vec<String> = (0..crate::tools::MAX_LOAD_TOOL_BATCH + 3)
            .map(|index| format!("tool_{index}"))
            .collect();
        let registered: Vec<Arc<dyn Tool>> = names.iter().map(|name| fake_tool(name)).collect();
        let deferred: Vec<&str> = names.iter().map(String::as_str).collect();
        let fixture = build_test_tool(registered, &deferred);

        let result = fixture
            .tool_load
            .execute(
                serde_json::json!({ "name": names }),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("should return Ok");

        assert!(!result.is_error);
        let text = result.text_content();
        assert!(text.contains("3 more were not"), "{text}");
        assert!(!text.contains("# tool_12"), "past the cap: {text}");
    }

    /// Dropping the `mcp__<server>__` prefix is the likeliest way to get a tool name wrong, and
    /// pure edit distance would never suggest the right answer.
    #[tokio::test]
    async fn load_tool_suggests_the_namespaced_name() {
        let fixture = build_test_tool(vec![fake_tool("mcp__notion__fetch")], &[
            "mcp__notion__fetch",
        ]);

        let result = fixture
            .tool_load
            .execute(
                serde_json::json!({"name": "fetch"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("should return Ok");

        assert!(result.is_error);
        let text = result.text_content();
        assert!(
            text.contains("Did you mean `mcp__notion__fetch`?"),
            "{text}"
        );
    }

    #[tokio::test]
    async fn load_tool_registry_dropped() {
        // The registry has gone while the `LoadToolTool` is still held somewhere.
        let mut fixture = build_test_tool(Vec::new(), &[]);
        fixture.tools.take();
        fixture.deferred.take();

        let result = fixture
            .tool_load
            .execute(
                serde_json::json!({"name": "anything"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("should return Ok with error tool_result");
        assert!(result.is_error);
        let text = result.text_content();
        assert!(text.contains("no longer available"));
    }
}
