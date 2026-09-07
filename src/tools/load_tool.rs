//! `load_tool` meta-tool: makes a deferred tool's full schema visible to the model on subsequent
//! turns. The active tool set is derived by scanning the conversation for successful `load_tool`
//! calls ([`crate::tools::load_tool::extract_loaded_tool_names_from_events`]); this tool's
//! `execute` only renders the description and schema as `tool_result` text. It never mutates the
//! registry.

use std::{
    collections::HashSet,
    sync::{Arc, RwLock, Weak},
};

use async_trait::async_trait;

use super::{LOAD_TOOL_NAME, Tool, ToolOutput, util::require_str};
use crate::{
    conversation::{ContentBlock, Event, Message},
    error::Result,
    permission::Permission,
    provider::ToolDefinition,
};

/// Meta-tool that makes a deferred tool's schema visible for use. Held by the
/// [`super::ToolRegistry`] like any other tool, so the same `Arc` lifecycle applies. The `Weak`
/// handles avoid a self-referential cycle (registry → `Arc<dyn Tool>` → `Arc<RwLock<…>>` →
/// registry).
pub(super) struct LoadToolTool {
    pub(super) tools: Weak<RwLock<Vec<std::sync::Arc<dyn Tool>>>>,
    pub(super) deferred: Weak<RwLock<HashSet<String>>>,
    /// Filled once the registry is attached to an MCP manager. Lets an unfindable name be
    /// explained by its server's state instead of reported as unknown; a server that never
    /// connected registers no tools, so `load_tool` is the first place the agent hears about it.
    pub(super) mcp_manager: Weak<std::sync::OnceLock<Weak<crate::mcp::McpClientManager>>>,
}

impl LoadToolTool {
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
                          under `[Tool discovery]` in the conversation context. After a \
                          successful call, each tool's full schema becomes available on \
                          your next turn. Invoke the tools by name as usual. Pass exact \
                          tool names (e.g. `mcp__notion__fetch`), either one as a string \
                          or several as an array."
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
            let definition = {
                let guard = crate::sync::read(&tools);
                guard
                    .iter()
                    .find(|t| t.definition().name == *name)
                    .map(|t| t.definition())
            };

            let Some(definition) = definition else {
                sections.push(match self.unavailable_server_reason(name).await {
                    Some(reason) => reason,
                    None => format!(
                        "Error: tool '{}' is not registered.{} Check the names listed under \
                         `[Tool discovery]` in the conversation context.",
                        name,
                        self.near_miss_hint(name, &tools),
                    ),
                });
                continue;
            };

            // A tool that is not deferred is already in the active set, so this is a no-op success:
            // the scanner records a name that was already there.
            let is_deferred = self
                .deferred
                .upgrade()
                .map(|d| crate::sync::read(&d).contains(name))
                .unwrap_or(false);

            resolved += 1;
            if !is_deferred {
                sections.push(format!(
                    "Tool '{name}' is already available. Call it directly."
                ));
                continue;
            }

            let schema = serde_json::to_string_pretty(&definition.parameters)
                .unwrap_or_else(|_| definition.parameters.to_string());
            sections.push(format!(
                "# {}\n\n{}\n\n## Schema\n\n```json\n{}\n```",
                name, definition.description, schema,
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
                " Only the first {} names were loaded; {} more were not. Call `load_tool` again \
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

/// Walk events and collect the names of tools loaded via successful `load_tool` calls. The only
/// door for this question: a scan of the materialized slice cannot see a load whose exchange a
/// compaction has summarized away or a repair has emptied, so this absorbs
/// [`Event::CompactBoundary::loaded_tools_snapshot`] at a boundary and clears the pending uses
/// inside the summarized window, whose results are below the view's logical start.
///
/// Returns names in load order, de-duplicated. The order is what makes the tools array a stable
/// cache prefix: `load_tool` calls only ever append to the conversation, so the array can only
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
            // rewinding past a `load_tool` would drop an entry from its middle and re-cache the
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
        load_tool: LoadToolTool,
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
        let tools: ToolStorage = Arc::new(RwLock::new(registered));
        let deferred: DeferredStorage = Arc::new(RwLock::new(
            deferred_names.iter().map(|n| n.to_string()).collect(),
        ));
        let load_tool = LoadToolTool {
            tools: Arc::downgrade(&tools),
            deferred: Arc::downgrade(&deferred),
            // No manager attached: these fixtures exercise the plain registry paths, so an
            // unfindable name must still produce the generic "not registered" message.
            mcp_manager: std::sync::Weak::new(),
        };
        Fixture {
            tools: Some(tools),
            deferred: Some(deferred),
            load_tool,
        }
    }

    #[tokio::test]
    async fn load_tool_unknown_name() {
        let fixture = build_test_tool(Vec::new(), &[]);
        let load_tool = &fixture.load_tool;
        let result = load_tool
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
        let load_tool = &fixture.load_tool;
        let result = load_tool
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
        let load_tool = &fixture.load_tool;

        let result = load_tool
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
            name: "read_file".to_string(),
            description: "Read a file from disk.".to_string(),
            schema: serde_json::json!({"type": "object"}),
        }) as Arc<dyn Tool>;
        let fixture = build_test_tool(vec![fake], &[]);
        let load_tool = &fixture.load_tool;

        let result = load_tool
            .execute(
                serde_json::json!({"name": "read_file"}),
                crate::tools::ToolContext::detached(CancellationToken::new()),
            )
            .await
            .expect("should return Ok");

        assert!(!result.is_error);
        let text = result.text_content();
        assert!(text.contains("already available"));
        assert!(text.contains("read_file"));
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
            .load_tool
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
            .load_tool
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
            .load_tool
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
            .load_tool
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
            .load_tool
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
